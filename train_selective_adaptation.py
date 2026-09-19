#!/usr/bin/env python3
"""
PRIME-Selective Adaptation Trainer
===================================
Fine-tunes the pretrained 125M model (step 10,000) with PRIME-Selective Attention
using Identity-Preserving Warm-Start.

Features:
- Identity-preserving initialization on Step 0 (exact loss match, zero loss spike).
- End-to-end co-adaptation of selective gating (W_delta, b_delta, tau_h, beta_h) and attention representations.
- Infinite token streaming from roneneldan/TinyStories.
- Mixed precision AMP (BF16 / FP16) with GradScaler.
- Live autoregressive generation probes at regular intervals.
- Periodic checkpointing to disk.
"""

import os
import sys
import time
import math
import argparse
import queue
import threading
from typing import Iterator, List, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer
from datasets import load_dataset

from prime_moment_attention import PrimeConfig, PrimeForCausalLM


def token_packing_stream(
    dataset_name: str,
    tokenizer,
    seq_len: int,
    batch_size: int,
    dataset_subset: Optional[str] = None,
) -> Iterator[torch.Tensor]:
    """Infinite token-packed batch generator."""
    buffer: List[int] = []
    chunk_size = seq_len + 1

    while True:
        try:
            if dataset_subset:
                ds = load_dataset(dataset_name, name=dataset_subset, split="train", streaming=True)
            else:
                ds = load_dataset(dataset_name, split="train", streaming=True)
            for sample in ds:
                text = sample.get("text", "")
                if not text or len(text.strip()) == 0:
                    continue
                tokens = tokenizer.encode(text) + [tokenizer.eos_token_id]
                buffer.extend(tokens)

                while len(buffer) >= batch_size * chunk_size:
                    batch_tokens = []
                    for _ in range(batch_size):
                        batch_tokens.append(buffer[:chunk_size])
                        buffer = buffer[chunk_size:]
                    yield torch.tensor(batch_tokens, dtype=torch.long)
        except Exception as e:
            print(f"[Stream Warning] Dataset error: {e}. Reconnecting...", file=sys.stderr)
            time.sleep(1)


class PrefetchDataIter:
    """Asynchronous background prefetcher to keep GPU matrix units saturated."""
    def __init__(self, generator_factory, queue_size: int = 8):
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.generator_factory = generator_factory

        def _worker():
            try:
                gen = self.generator_factory()
                for item in gen:
                    if self.stopped:
                        break
                    self.queue.put(item)
            except Exception as e:
                print(f"[Prefetch Worker Error] {e}", file=sys.stderr)

        self.thread = threading.Thread(target=_worker, daemon=True)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True


def get_disk_free_gb(path: str) -> float:
    try:
        st = os.statvfs(path)
        return (st.f_bavail * st.f_frsize) / (1024 ** 3)
    except Exception:
        return -1.0


def main():
    default_output = "/data/prime_checkpoints" if os.path.exists("/data") else "checkpoints"
    parser = argparse.ArgumentParser(description="PRIME-Selective Warm-Start Adaptation")
    parser.add_argument("--base-checkpoint", type=str, default="checkpoints/prime_125m_step_10000.pt")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume directly from a selective checkpoint")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--steps", type=int, default=1500, help="Total adaptation steps")
    parser.add_argument("--target-tokens", type=int, default=None, help="Target total tokens (overrides --steps if set)")
    parser.add_argument("--batch-size", type=int, default=32, help="Micro-batch size per forward pass")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--lr", type=float, default=5.0e-5, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1.0e-5, help="Minimum learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--eval-every", type=int, default=250, help="Steps between generation probes")
    parser.add_argument("--save-every", type=int, default=1000, help="Steps between checkpoints")
    parser.add_argument("--keep-last-k", type=int, default=5, help="Number of rolling checkpoints to keep")
    parser.add_argument("--milestone-every", type=int, default=10000, help="Steps between permanent milestone checkpoints")
    parser.add_argument("--output-dir", type=str, default=default_output)
    parser.add_argument("--additional-steps", type=int, default=None, help="Additional steps to train from resume checkpoint")
    parser.add_argument("--restore-optimizer", action="store_true", default=False, help="Restore optimizer momentum state from checkpoint")
    parser.add_argument("--dataset-subset", type=str, default=None, help="Subset/config name for dataset (e.g. sample-10BT for fineweb-edu)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    free_gb = get_disk_free_gb(args.output_dir)
    print("=" * 70)
    print("  PRIME-Selective Attention: High-Scale Pretraining / Adaptation")
    print(f"  Target Checkpoint Dir: {args.output_dir} ({free_gb:.1f} GB free)")
    print(f"  Device:                {device}")
    print(f"  Micro-Batch:           {args.batch_size} (Grad Accum: {args.grad_accum} -> Effective Batch: {args.batch_size * args.grad_accum})")
    print(f"  Seq Len:               {args.seq_len} ({tokens_per_step:,} tok/step)")
    print(f"  Learning Rate:         {args.lr} -> {args.min_lr}")
    print("=" * 70)

    # 1. Load Tokenizer
    print("\n[1] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Model Initialization (Resume or Warm-Start)
    start_step = 1
    total_tokens_trained = 0
    saved_rolling_ckpts = []

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\n[2] Resuming PRIME-Selective directly from {args.resume_from}...")
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model = PrimeForCausalLM(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt.get("step", 0) + 1
        total_tokens_trained = ckpt.get("tokens_trained", (start_step - 1) * tokens_per_step)
        print(f"    Resumed at Step {start_step:,} (Previously trained: {total_tokens_trained:,} tokens)")
        # Discover existing rolling checkpoints to maintain proper disk pruning
        if os.path.exists(args.output_dir):
            for f in sorted(os.listdir(args.output_dir)):
                if f.startswith("prime_125m_selective_step_") and f.endswith(".pt"):
                    try:
                        s_num = int(f.replace("prime_125m_selective_step_", "").replace(".pt", ""))
                        if s_num % args.milestone_every != 0 and s_num <= start_step:
                            saved_rolling_ckpts.append(os.path.join(args.output_dir, f))
                    except ValueError:
                        pass
            print(f"    Discovered {len(saved_rolling_ckpts)} existing rolling checkpoint(s) for auto-pruning.")
    else:
        print(f"\n[2] Warm-Starting PRIME-Selective from Base Checkpoint {args.base_checkpoint}...")
        model = PrimeForCausalLM.from_pretrained_base_to_selective(args.base_checkpoint, device=device)

    if args.target_tokens is not None:
        remaining_tokens = max(0, args.target_tokens - total_tokens_trained)
        remaining_steps = math.ceil(remaining_tokens / tokens_per_step)
        args.steps = start_step + remaining_steps - 1
        print(f"    [Target Alignment] Target: {args.target_tokens:,} tokens | Remaining: {remaining_tokens:,} tokens -> Adjusted Final Step: {args.steps:,}")
    elif args.additional_steps is not None:
        args.steps = start_step + args.additional_steps - 1
        print(f"[Step Config] Running {args.additional_steps:,} additional steps -> Ending at Step {args.steps:,}")

    total_params = sum(p.numel() for p in model.parameters())
    selective_params = sum(
        p.numel() for n, p in model.named_parameters()
        if any(k in n for k in ["delta_proj", "log_tau", "log_beta"])
    )
    print(f"    Total Parameters:     {total_params / 1e6:.2f}M")
    print(f"    Selective Parameters: {selective_params:,} ({selective_params / total_params * 100:.3f}%)")

    # 3. Parameter Groups & Optimizer
    base_params = [
        p for n, p in model.named_parameters()
        if not any(k in n for k in ["delta_proj", "log_tau", "log_beta"])
    ]
    gate_params = [
        p for n, p in model.named_parameters()
        if any(k in n for k in ["delta_proj", "log_tau", "log_beta"])
    ]

    optimizer = AdamW([
        {"params": base_params, "lr": args.lr, "weight_decay": 0.01},
        {"params": gate_params, "lr": args.lr * 2.0, "weight_decay": 0.001},
    ], betas=(0.9, 0.95))

    if args.resume_from and os.path.exists(args.resume_from) and args.restore_optimizer and "optimizer_state_dict" in ckpt:
        print("    Restoring AdamW optimizer state...")
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if device == "cuda":
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            print("    Optimizer state restored and mapped to device.")
        except Exception as e:
            print(f"    [Warning] Could not restore full optimizer state: {e}. Reinitializing...")
    else:
        print("    Optimizer initialized with fresh momentum & warmup schedule.")

    effective_warmup = min(args.warmup_steps, max(10, (args.steps - start_step + 1) // 20))
    def get_lr_factor(step: int) -> float:
        rel_step = step - start_step
        if rel_step < effective_warmup:
            return float(rel_step + 1) / float(effective_warmup)
        remaining = max(1, args.steps - start_step - effective_warmup)
        progress = float(rel_step - effective_warmup) / float(remaining)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = args.min_lr / args.lr
        return min_factor + (1.0 - min_factor) * cosine

    print(f"\n[3] Initializing Token Packing Stream from '{args.dataset}' (subset: {args.dataset_subset})...")
    data_iter = PrefetchDataIter(
        lambda: token_packing_stream(
            args.dataset,
            tokenizer,
            args.seq_len,
            args.batch_size,
            dataset_subset=args.dataset_subset
        ),
        queue_size=8
    )

    # 5. Prompts for live monitoring
    test_prompts = [
        "Once upon a time, there was a little girl named Lily.",
        "One sunny morning, a dog found a big",
        "The primary function of mitochondria in a biological cell is",
        "In modern science, the theory of relativity explains",
    ]

    print(f"\n[4] Commencing Selective Adaptation ({start_step:,} to {args.steps:,} steps)...")
    start_time = time.time()
    history = []

    model.train()
    for step in range(start_step, args.steps + 1):
        step_t0 = time.time()

        # Update LR
        factor = get_lr_factor(step)
        optimizer.param_groups[0]["lr"] = args.lr * factor
        optimizer.param_groups[1]["lr"] = args.lr * 2.0 * factor

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        skip_step = False

        for accum_idx in range(args.grad_accum):
            batch = next(data_iter).to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]

            if device == "cuda":
                amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    out = model(x, labels=y)
                    loss = out["loss"] / args.grad_accum

                if not torch.isfinite(loss):
                    print(f"[Warning] Step {step} (micro {accum_idx}): Loss is {loss}, skipping...")
                    optimizer.zero_grad(set_to_none=True)
                    del out, loss
                    torch.cuda.empty_cache()
                    skip_step = True
                    break

                loss.backward()
                step_loss += loss.item()
                del out, loss  # Free activation graph and FP32 logits immediately!
            else:
                out = model(x, labels=y)
                loss = out["loss"] / args.grad_accum
                if not torch.isfinite(loss):
                    print(f"[Warning] Step {step} (micro {accum_idx}): Loss is {loss}, skipping...")
                    del out, loss
                    skip_step = True
                    break
                loss.backward()
                step_loss += loss.item()
                del out, loss

        if skip_step:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            print(f"[Warning] Step {step}: GradNorm is {grad_norm}, skipping...")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        optimizer.step()

        step_time = time.time() - step_t0
        tokens_in_step = args.batch_size * args.seq_len * args.grad_accum
        total_tokens_trained += tokens_in_step
        tokens_per_sec = tokens_in_step / max(step_time, 1e-4)

        loss_val = step_loss
        history.append(loss_val)

        # Log progress every 10 steps
        if step % 10 == 0 or step == 1:
            avg_loss_10 = sum(history[-10:]) / len(history[-10:])
            # Inspect first layer selective gate stats
            l0_attn = model.layers[0].self_attn
            mean_beta = l0_attn.get_beta(device).mean().item()
            mean_tau = l0_attn.get_tau(device).mean().item()
            tok_cur = total_tokens_trained / 1e6
            tok_tot = (args.target_tokens / 1e6) if args.target_tokens else ((args.steps * tokens_per_step) / 1e6)
            print(
                f"Step {step:5d}/{args.steps} ({tok_cur:5.1f}M/{tok_tot:.0f}M tok) | "
                f"Loss: {loss_val:.4f} (Avg10: {avg_loss_10:.4f}) | "
                f"BaseLR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"L0 Beta: {mean_beta:.2f} | "
                f"L0 Tau: {mean_tau:.1f} | "
                f"{tokens_per_sec:6.1f} tok/s | "
                f"{step_time:.3f}s/step",
                flush=True
            )

        # Evaluation & Generation Probe
        if step % args.eval_every == 0 or step == 1:
            model.eval()
            print(f"\n--- [Step {step} Selective Generation Probe] ---")
            for prompt in test_prompts:
                p_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    gen_ids = model.generate(p_ids, max_new_tokens=35, temperature=0.7, top_k=40)
                gen_text = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
                print(f"  Prompt: \"{prompt}\"")
                print(f"  Output: \"{gen_text}\"")
            print("-" * 50 + "\n", flush=True)
            model.train()

        # Save Checkpoint
        if step % args.save_every == 0 or step == args.steps:
            ckpt_file = os.path.join(args.output_dir, f"prime_125m_selective_step_{step}.pt")
            free_gb = get_disk_free_gb(args.output_dir)
            print(f"[Checkpoint] Saving checkpoint to {ckpt_file} (Disk Free: {free_gb:.1f} GB)...", flush=True)
            if free_gb > 0 and free_gb < 10.0:
                print(f"[CRITICAL WARNING] Disk space low on {args.output_dir}: {free_gb:.1f} GB remaining!", flush=True)

            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model.config,
                "loss": loss_val,
                "tokens_trained": total_tokens_trained,
            }, ckpt_file)
            print(f"[Checkpoint] Saved {ckpt_file} successfully.", flush=True)

            # Rolling checkpoint pruning: preserve last K plus permanent milestones
            is_milestone = (step % args.milestone_every == 0) or (step == args.steps)
            if not is_milestone:
                saved_rolling_ckpts.append(ckpt_file)
                if len(saved_rolling_ckpts) > args.keep_last_k:
                    oldest = saved_rolling_ckpts.pop(0)
                    if os.path.exists(oldest):
                        try:
                            os.remove(oldest)
                            print(f"[Storage Management] Pruned rolling checkpoint {oldest} to preserve disk space.", flush=True)
                        except Exception as e:
                            print(f"[Storage Management] Warning: failed to prune {oldest}: {e}", flush=True)
            else:
                print(f"[Storage Management] Permanent milestone checkpoint recorded at step {step:,}.", flush=True)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  PRIME-Selective Adaptation Completed Successfully!")
    print(f"  Total Steps:          {args.steps}")
    print(f"  Tokens Trained:       {total_tokens_trained:,}")
    print(f"  Final Loss:           {history[-1]:.4f}")
    print(f"  Elapsed Time:         {elapsed:.1f}s ({elapsed / 60:.2f} min)")
    print(f"  Final Checkpoint:     {os.path.join(args.output_dir, f'prime_125m_selective_step_{args.steps}.pt')}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
