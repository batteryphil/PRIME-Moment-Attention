#!/usr/bin/env python3
"""
PRIME-125M Continued Pretraining & Fine-Tuning on Code, Math & Reasoning Mixture
================================================================================
Trains the verified PRIME-125M architecture on packed token streams containing:
1. Python algorithms & code instructions
2. C / C++ systems programming
3. GSM8K mathematical reasoning
4. SmolTalk analytical thought

Features:
- Fast memory-mapped reading of packed uint16 tokens (train.bin, val.bin)
- AdamW with bfloat16 AMP and gradient norm clipping
- Deterministic validation loss tracking on held-out tokens
- Live AST & GCC syntax evaluation probes during training
- Checkpoints saved directly to /data/prime_checkpoints/code_math_125m/
- Rolling checkpoint pruning to preserve disk space
"""

import os
import sys
import time
import math
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prime_moment_attention.model import PrimeConfig, PrimeForCausalLM
from benchmarks.eval_code_math_capabilities import (
    test_python_generation,
    test_c_syntax,
    test_math_reasoning,
    PYTHON_BENCHMARKS,
    C_BENCHMARKS,
    MATH_BENCHMARKS,
)


class MemmapDataset:
    """Zero-overhead memory-mapped streaming dataset from packed uint16 binaries."""
    def __init__(self, bin_path: str, seq_len: int):
        self.seq_len = seq_len
        self.chunk_size = seq_len + 1  # input + target
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary file not found: {bin_path}")
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)
        self.max_idx = self.total_tokens - self.chunk_size
        if self.max_idx <= 0:
            raise ValueError(f"Dataset in {bin_path} too small for seq_len {seq_len}")

    def get_batch(self, batch_size: int, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
        indices = np.random.randint(0, self.max_idx, size=batch_size)
        batch = np.stack([self.data[i : i + self.chunk_size] for i in indices])
        tensor = torch.from_numpy(batch.astype(np.int64)).to(device)
        return tensor[:, :-1], tensor[:, 1:]


@torch.no_grad()
def evaluate_val_loss(
    model: nn.Module,
    val_dataset: MemmapDataset,
    num_batches: int = 20,
    batch_size: int = 8,
    device: str = "cuda",
) -> float:
    """Computes deterministic cross-entropy loss over held-out validation tokens."""
    model.eval()
    losses = []
    step_size = max(1, val_dataset.max_idx // num_batches)
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    for b in range(num_batches):
        idx = min(b * step_size, val_dataset.max_idx)
        chunk = val_dataset.data[idx : idx + val_dataset.chunk_size]
        batch = torch.from_numpy(chunk.astype(np.int64)).unsqueeze(0).to(device)
        x, y = batch[:, :-1], batch[:, 1:]

        if device == "cuda":
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out = model(x, labels=y)
                losses.append(out["loss"].item())
        else:
            out = model(x, labels=y)
            losses.append(out["loss"].item())

    model.train()
    return float(np.mean(losses))


def get_disk_free_gb(path: str) -> float:
    try:
        st = os.statvfs(path)
        return (st.f_bavail * st.f_frsize) / (1024 ** 3)
    except Exception:
        return -1.0


def main():
    parser = argparse.ArgumentParser(description="PRIME-125M Code, Math & Reasoning Continued Training")
    parser.add_argument("--base-checkpoint", type=str, default="/data/prime_checkpoints/prime_125m_step_10000.pt")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--additional-steps", type=int, default=None, help="Additional steps to train beyond resumed step")
    parser.add_argument("--data-dir", type=str, default="/data/datasets/prime_code_math_reasoning")
    parser.add_argument("--output-dir", type=str, default="/data/prime_checkpoints/code_math_125m")
    parser.add_argument("--steps", type=int, default=1500, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per forward pass")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=2.0e-5, help="Minimum learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Linear warmup steps")
    parser.add_argument("--eval-every", type=int, default=250, help="Steps between validation and capability evaluations")
    parser.add_argument("--save-every", type=int, default=500, help="Steps between checkpoints")
    parser.add_argument("--keep-last-k", type=int, default=3, help="Number of rolling checkpoints to keep")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum

    # 1. Initialize Tokenizer & Datasets
    print("[1/5] Loading Tokenizer and Memory-Mapped Datasets...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_bin = os.path.join(args.data_dir, "train.bin")
    val_bin = os.path.join(args.data_dir, "val.bin")

    train_ds = MemmapDataset(train_bin, seq_len=args.seq_len)
    val_ds = MemmapDataset(val_bin, seq_len=args.seq_len)
    print(f"  Train Set : {train_ds.total_tokens:,} tokens ({train_ds.total_tokens / 1e6:.2f}M)")
    print(f"  Val Set   : {val_ds.total_tokens:,} tokens ({val_ds.total_tokens / 1e6:.2f}M)")

    # 2. Load Model
    start_step = 1
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\n[2/5] Resuming PRIME-125M from {args.resume_from}...")
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        start_step = ckpt.get("step", 0) + 1
        if args.additional_steps is not None:
            args.steps = start_step + args.additional_steps - 1
    else:
        print(f"\n[2/5] Loading PRIME-125M from {args.base_checkpoint}...")
        ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)

    config = ckpt["config"]
    model = PrimeForCausalLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Loaded model successfully: {param_count / 1e6:.2f}M parameters (Resuming at Step {start_step:,}).")

    print("=" * 80)
    print("  PRIME-125M CONTINUED TRAINING: CODE, MATH & REASONING")
    print(f"  Checkpoint           : {args.resume_from or args.base_checkpoint}")
    print(f"  Dataset Directory    : {args.data_dir}")
    print(f"  Output Directory     : {args.output_dir} ({get_disk_free_gb(args.output_dir):.1f} GB free)")
    print(f"  Device               : {device}")
    print(f"  Effective Batch Size : {args.batch_size * args.grad_accum} ({tokens_per_step:,} tok/step)")
    print(f"  Sequence Length      : {args.seq_len}")
    print(f"  Learning Rate        : {args.lr} -> {args.min_lr} (Warmup: {args.warmup_steps})")
    print(f"  Steps                : {start_step:,} to {args.steps:,} (~{((args.steps - start_step + 1) * tokens_per_step) / 1e6:.1f}M tokens)")
    print("=" * 80 + "\n")

    # 3. Setup Optimizer and LR Schedule
    print("\n[3/5] Initializing Optimizer & Parameter Groups...")
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

    if args.resume_from and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("  Restored AdamW optimizer state from checkpoint.")
        except Exception as e:
            print(f"  Note: Initializing fresh optimizer momentum: {e}")

    effective_warmup = min(args.warmup_steps, max(5, (args.steps - start_step + 1) // 10))
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

    # 4. Initial Baseline Validation & Capability Probes
    print("\n[4/5] Running Baseline Validation & Capabilities (Step 0)...")
    base_val_loss = evaluate_val_loss(model, val_ds, num_batches=20, device=device)
    base_py = test_python_generation(model, tokenizer, device)
    base_c = test_c_syntax(model, tokenizer, device)
    base_math = test_math_reasoning(model, tokenizer, device)

    print(f"  Initial Validation Loss   : {base_val_loss:.4f} (Perplexity: {math.exp(min(base_val_loss, 20)):.2f})")
    print(f"  Initial Python AST Valid  : {base_py['accuracy']:.1f}%")
    print(f"  Initial C/C++ GCC Valid   : {base_c['accuracy']:.1f}%")
    print(f"  Initial Math Solving Acc  : {base_math['accuracy']:.1f}%")

    # 5. Training Loop
    print(f"\n[5/5] Launching Training Loop ({args.steps:,} steps)...")
    saved_checkpoints: List[str] = []
    training_metrics: List[Dict[str, Any]] = [{
        "step": 0,
        "val_loss": base_val_loss,
        "python_ast": base_py["accuracy"],
        "c_gcc": base_c["accuracy"],
        "math_acc": base_math["accuracy"],
    }]

    model.train()
    start_time = time.time()
    recent_losses: List[float] = []

    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    for step in range(start_step, args.steps + 1):
        step_t0 = time.time()
        lr_factor = get_lr_factor(step)
        optimizer.param_groups[0]["lr"] = args.lr * lr_factor
        optimizer.param_groups[1]["lr"] = args.lr * 2.0 * lr_factor

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for accum_idx in range(args.grad_accum):
            x, y = train_ds.get_batch(args.batch_size, device=device)
            if device == "cuda":
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = model(x, labels=y)
                    loss = out["loss"] / args.grad_accum
                loss.backward()
                step_loss += loss.item()
                del out, loss
            else:
                out = model(x, labels=y)
                loss = out["loss"] / args.grad_accum
                loss.backward()
                step_loss += loss.item()
                del out, loss

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        recent_losses.append(step_loss)
        if len(recent_losses) > 20:
            recent_losses.pop(0)

        step_elapsed = time.time() - step_t0
        tok_s = tokens_per_step / max(0.001, step_elapsed)

        # Logging every 10 steps
        if step % 10 == 0 or step == 1:
            avg_loss = sum(recent_losses) / len(recent_losses)
            cum_tok = (step * tokens_per_step) / 1e6
            tot_tok = (args.steps * tokens_per_step) / 1e6
            print(
                f"Step {step:5d}/{args.steps} ({cum_tok:5.2f}M/{tot_tok:5.2f}M tok) | "
                f"Loss: {step_loss:.4f} (Avg20: {avg_loss:.4f}) | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"GradNorm: {grad_norm:.2f} | "
                f"{tok_s:6.1f} tok/s ({step_elapsed:.3f}s/step)",
                flush=True
            )

        # Periodic Evaluation
        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate_val_loss(model, val_ds, num_batches=20, device=device)
            py_eval = test_python_generation(model, tokenizer, device)
            c_eval = test_c_syntax(model, tokenizer, device)
            m_eval = test_math_reasoning(model, tokenizer, device)

            metric_entry = {
                "step": step,
                "train_loss": step_loss,
                "val_loss": val_loss,
                "python_ast": py_eval["accuracy"],
                "c_gcc": c_eval["accuracy"],
                "math_acc": m_eval["accuracy"],
            }
            training_metrics.append(metric_entry)

            print("\n" + "-" * 75)
            print(f"  [EVALUATION REPORT @ STEP {step}]")
            print(f"  Validation Loss    : {val_loss:.4f} (Perplexity: {math.exp(min(val_loss, 20)):.2f})")
            print(f"  Python AST Valid   : {py_eval['accuracy']:5.1f}% (Base: {base_py['accuracy']:.1f}%)")
            print(f"  C/C++ GCC Valid    : {c_eval['accuracy']:5.1f}% (Base: {base_c['accuracy']:.1f}%)")
            print(f"  Math Problem Acc   : {m_eval['accuracy']:5.1f}% (Base: {base_math['accuracy']:.1f}%)")
            print("-" * 75 + "\n", flush=True)

        # Periodic Checkpointing
        if step % args.save_every == 0 or step == args.steps:
            ckpt_filename = f"prime_125m_code_math_step_{step}.pt"
            ckpt_path = os.path.join(args.output_dir, ckpt_filename)
            torch.save({
                "step": step,
                "tokens_trained": step * tokens_per_step,
                "config": config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"[+] Saved checkpoint to {ckpt_path} (Free Disk: {get_disk_free_gb(args.output_dir):.1f} GB)")

            saved_checkpoints.append(ckpt_path)
            # Prune rolling checkpoints to save space if needed
            if len(saved_checkpoints) > args.keep_last_k and step != args.steps:
                old_ckpt = saved_checkpoints.pop(0)
                if os.path.exists(old_ckpt):
                    try:
                        os.remove(old_ckpt)
                        print(f"  [-] Pruned rolling checkpoint {os.path.basename(old_ckpt)}")
                    except Exception as e:
                        print(f"  [!] Failed to prune {old_ckpt}: {e}")

    # Final summary save
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(training_metrics, f, indent=2)

    total_time = time.time() - start_time
    print("=" * 80)
    print("  TRAINING SESSION FINISHED SUCCESSFULLY")
    print(f"  Total Wall Clock Time : {total_time / 60.0:.2f} minutes")
    print(f"  Total Tokens Processed: {(args.steps * tokens_per_step):,} tokens")
    print(f"  Metrics Saved to      : {metrics_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
