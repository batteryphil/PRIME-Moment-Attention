#!/usr/bin/env python3
"""
PRIME-125M: Reasoning & Thoughtful Fine-Tuning (SFT) Pipeline

Transforms the pretrained PRIME-125M base model into a thoughtful,
structured reasoning assistant capable of producing explicit <think>...</think>
thought traces before generating clean, articulate answers.

Key features:
1. Multi-source mixture: Math/Logic CoT (GSM8K) + Thoughtful explanations (Smoltalk).
2. Exact Prompt Loss Masking: labels = -100 on user prompt, loss calculated
   exclusively on <think>...</think> and final answer tokens.
3. Native O(1) PRIME recurrence fine-tuning with conservative learning rates
   to prevent catastrophic forgetting.
4. Live evaluation probes tracking reasoning emergence every 50 steps.
"""

import os
import sys
import math
import time
import re
import random
import argparse
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer
from datasets import load_dataset

# Add project root to import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.prime_moment_attention.model import PrimeForCausalLM, PrimeConfig


def get_disk_free_gb(path: str) -> float:
    try:
        st = os.statvfs(path)
        return (st.f_bavail * st.f_frsize) / (1024 ** 3)
    except Exception:
        return -1.0


def format_gsm8k(item: Dict[str, Any]) -> Tuple[str, str]:
    q = item["question"].strip()
    raw_a = item["answer"].strip()
    parts = raw_a.split("####")
    if len(parts) == 2:
        reasoning = parts[0].strip()
        reasoning = re.sub(r'<<.*?>>', '', reasoning).strip()
        ans = parts[1].strip()
        thought = f"<think>\nLet's analyze this step-by-step:\n{reasoning}\n</think>"
        final_answer = f"The final answer is {ans}."
    else:
        thought = f"<think>\nLet's solve this:\n{raw_a}\n</think>"
        final_answer = raw_a

    prompt = f"User: {q}\n\nAssistant: "
    response = f"{thought}\n\n{final_answer}"
    return prompt, response


def format_smoltalk(item: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    msgs = item.get("messages", [])
    user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), None)
    asst_msg = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
    if not user_msg or not asst_msg:
        return None
    user_msg = user_msg.strip()
    asst_msg = asst_msg.strip()
    if len(user_msg) < 5 or len(asst_msg) < 5:
        return None

    # Synthesize concise, thoughtful planning thought
    topic_summary = user_msg[:80].replace("\n", " ")
    thought = (
        f"<think>\n"
        f"The user is asking: \"{topic_summary}...\".\n"
        f"I will organize my response clearly, providing a direct, helpful, and polite explanation.\n"
        f"</think>"
    )
    prompt = f"User: {user_msg}\n\nAssistant: "
    response = f"{thought}\n\n{asst_msg}"
    return prompt, response


class ReasoningDataIterator:
    """Interleaves GSM8K step-by-step reasoning with Smoltalk thoughtful dialogue."""
    def __init__(self, tokenizer, max_seq_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        print("[Data] Initializing GSM8K reasoning stream...")
        self.ds_gsm = list(load_dataset("openai/gsm8k", "main", split="train"))
        random.shuffle(self.ds_gsm)
        self.gsm_idx = 0

        print("[Data] Initializing Smoltalk conversational stream (caching 10,000 in-memory examples)...")
        raw_smol = load_dataset("HuggingFaceTB/smoltalk", "smol-magpie-ultra", split="train[:10000]")
        self.ds_smol = list(raw_smol)
        random.shuffle(self.ds_smol)
        self.smol_idx = 0

    def next_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        while True:
            # Alternating 50/50 between math reasoning and thoughtful dialogue
            if random.random() < 0.5:
                item = self.ds_gsm[self.gsm_idx % len(self.ds_gsm)]
                self.gsm_idx += 1
                prompt, response = format_gsm8k(item)
            else:
                item = self.ds_smol[self.smol_idx % len(self.ds_smol)]
                self.smol_idx += 1
                res = format_smoltalk(item)
                if res is None:
                    continue
                prompt, response = res

            p_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            r_ids = self.tokenizer.encode(response, add_special_tokens=False) + [self.tokenizer.eos_token_id]

            total_len = len(p_ids) + len(r_ids)
            if total_len > self.max_seq_len:
                # Truncate response if too long, keeping prompt intact
                max_r = self.max_seq_len - len(p_ids)
                if max_r < 30:
                    continue
                r_ids = r_ids[:max_r - 1] + [self.tokenizer.eos_token_id]

            input_ids = torch.tensor([p_ids + r_ids], dtype=torch.long)
            labels = torch.tensor([[-100] * len(p_ids) + r_ids], dtype=torch.long)
            return input_ids, labels


@torch.no_grad()
def run_reasoning_probe(model, tokenizer, device: str):
    model.eval()
    test_prompts = [
        "What is 25 * 16?",
        "If Sarah has 3 brothers, and each brother has 2 sisters, how many sisters does Sarah have?",
        "Explain the primary function of mitochondria in a biological cell.",
        "Why do tree leaves change color during autumn?",
    ]
    print("\n" + "=" * 60)
    print("  [Live Reasoning & Thoughtfulness Probe]")
    print("=" * 60)

    for prompt_text in test_prompts:
        full_p = f"User: {prompt_text}\n\nAssistant: <think>\n"
        input_ids = tokenizer.encode(full_p, return_tensors="pt").to(device)

        # Autoregressive generation up to 250 new tokens
        output_tokens = model.generate(
            input_ids,
            max_new_tokens=250,
            temperature=0.7,
            top_k=40,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        # Display response
        print(f"\nPrompt: {prompt_text}")
        # Print generated text starting after 'Assistant: '
        if "Assistant:" in gen_text:
            asst_text = gen_text.split("Assistant:", 1)[1].strip()
            print(f"Assistant:\n{asst_text}")
        else:
            print(f"Output: {gen_text}")
    print("=" * 60 + "\n")
    model.train()


def main():
    parser = argparse.ArgumentParser(description="PRIME-125M Reasoning SFT")
    parser.add_argument("--base-checkpoint", type=str, default="/data/prime_checkpoints/prime_125m_base_pretrained.pt")
    parser.add_argument("--output-dir", type=str, default="/data/prime_checkpoints")
    parser.add_argument("--steps", type=int, default=1000, help="Total SFT steps")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--lr", type=float, default=4.0e-5, help="Base learning rate")
    parser.add_argument("--min-lr", type=float, default=5.0e-6, help="Minimum learning rate")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Warmup steps")
    parser.add_argument("--eval-every", type=int, default=100, help="Steps between probes")
    parser.add_argument("--save-every", type=int, default=250, help="Steps between checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  PRIME-125M: Reasoning & Thoughtful Fine-Tuning (SFT)")
    print(f"  Base Checkpoint: {args.base_checkpoint}")
    print(f"  Output Directory: {args.output_dir} ({get_disk_free_gb(args.output_dir):.1f} GB free)")
    print(f"  Device:          {device}")
    print(f"  Steps:           {args.steps} (Grad Accum: {args.grad_accum})")
    print(f"  Learning Rate:   {args.lr} -> {args.min_lr} (Warmup: {args.warmup_steps})")
    print("=" * 70)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model
    print(f"\n[1] Loading pretrained base checkpoint from {args.base_checkpoint}...")
    ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = PrimeForCausalLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"    Loaded PRIME-125M successfully ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params).")

    # 3. Parameter Groups (Higher LR for gates)
    base_params = [p for n, p in model.named_parameters() if not any(k in n for k in ["delta_proj", "log_tau", "log_beta"])]
    gate_params = [p for n, p in model.named_parameters() if any(k in n for k in ["delta_proj", "log_tau", "log_beta"])]

    optimizer = AdamW([
        {"params": base_params, "lr": args.lr, "weight_decay": 0.01},
        {"params": gate_params, "lr": args.lr * 2.0, "weight_decay": 0.001},
    ], betas=(0.9, 0.95))

    def get_lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / float(args.warmup_steps)
        progress = float(step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_f = args.min_lr / args.lr
        return min_f + (1.0 - min_f) * cosine

    # 4. Data Stream
    print("\n[2] Initializing Multi-Source Reasoning Iterator...")
    data_iter = ReasoningDataIterator(tokenizer, max_seq_len=args.max_seq_len)

    # 5. Baseline Probe Before Fine-Tuning
    print("\n[3] Running Baseline Reasoning Probe (Pre-SFT)...")
    run_reasoning_probe(model, tokenizer, device)

    # 6. Training Loop
    print("\n[4] Launching Reasoning SFT Training Loop...")
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    recent_losses = []
    start_time = time.time()
    step_start = time.time()

    for step in range(1, args.steps + 1):
        accum_loss = 0.0
        tokens_in_step = 0

        # Learning rate schedule
        lr_factor = get_lr_factor(step)
        for i, param_group in enumerate(optimizer.param_groups):
            mult = 2.0 if i == 1 else 1.0
            param_group["lr"] = args.lr * mult * lr_factor

        for _ in range(args.grad_accum):
            input_ids, labels = data_iter.next_sample()
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            tokens_in_step += input_ids.shape[1]

            out = model(input_ids, labels=labels)
            loss = out["loss"] / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        recent_losses.append(accum_loss)
        if len(recent_losses) > 10:
            recent_losses.pop(0)
        avg_loss = sum(recent_losses) / len(recent_losses)

        elapsed = time.time() - step_start
        step_start = time.time()
        tok_s = tokens_in_step / max(0.001, elapsed)

        if step % 10 == 0 or step == 1:
            l0_tau = math.exp(model.layers[0].self_attn.log_tau.mean().item())
            l0_beta = math.exp(model.layers[0].self_attn.log_beta.mean().item())
            print(
                f"SFT Step {step:4d}/{args.steps} | Loss: {accum_loss:.4f} (Avg10: {avg_loss:.4f}) | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | Tau: {l0_tau:.1f} | Beta: {l0_beta:.2f} | "
                f"{tok_s:.1f} tok/s ({elapsed:.2f}s/step)"
            )

        if step % args.eval_every == 0:
            run_reasoning_probe(model, tokenizer, device)

        if step % args.save_every == 0 or step == args.steps:
            save_path = os.path.join(args.output_dir, f"prime_125m_reasoning_sft_step_{step}.pt")
            free_gb = get_disk_free_gb(args.output_dir)
            print(f"[Checkpoint] Saving SFT checkpoint to {save_path} (Disk Free: {free_gb:.1f} GB)...")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "config": config,
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_loss,
            }, save_path)
            print(f"[Checkpoint] Saved {save_path} successfully.")

    # Save final
    final_path = os.path.join(args.output_dir, "prime_125m_reasoning_sft_final.pt")
    print(f"\n[Done] Saving final model to {final_path}...")
    torch.save({
        "step": args.steps,
        "model_state_dict": model.state_dict(),
        "config": config,
        "avg_loss": avg_loss,
    }, final_path)
    total_time = time.time() - start_time
    print(f"Reasoning SFT Completed in {total_time/60:.1f} minutes!")


if __name__ == "__main__":
    main()
