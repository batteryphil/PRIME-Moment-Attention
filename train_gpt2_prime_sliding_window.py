#!/usr/bin/env python3
"""
Fine-Tune Stock GPT-2 with PRIME Sliding Window Attention
=========================================================
Fine-tunes the transplanted PRIME Sliding Window layers on stock GPT-2 (124M)
to calibrate the 2nd-order polynomial moment expectations with pretrained representations.
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, AutoTokenizer


class VectorizedAdaptiveGPT2HybridPrimeAttention(nn.Module):
    def __init__(self, orig_attn, layer_idx: int, window_size: int = 128, decay: float = 0.9995, init_gate: float = -3.5):
        super().__init__()
        self.layer_idx = layer_idx
        self.window_size = window_size
        self.decay = decay
        self.init_gate = init_gate
        
        self.c_attn = orig_attn.c_attn
        self.c_proj = orig_attn.c_proj
        self.split_size = orig_attn.split_size
        self.num_heads = orig_attn.num_heads
        self.head_dim = orig_attn.head_dim
        self.scaling = 1.0 / (self.head_dim ** 0.5)

        # Per-head learnable gate initialized to -3.5 (sigmoid(-3.5) approx 0.029)
        self.gate = nn.Parameter(torch.full((1, self.num_heads, 1, 1), init_gate))
        self.prime_norm = nn.LayerNorm(self.head_dim)

    def forward(self, hidden_states, past_key_values=None, attention_mask=None, **kwargs):
        B, L, C = hidden_states.shape
        device = hidden_states.device
        
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        shape_kv = (B, L, self.num_heads, self.head_dim)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)
        query_states = query_states.view(shape_kv).transpose(1, 2)

        q_scaled = query_states * self.scaling
        
        idx = torch.arange(L, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        causal_mask = diff >= 0

        # 1. Local Sliding Window Softmax
        scores = torch.matmul(q_scaled, key_states.transpose(2, 3))
        window_mask = (diff >= 0) & (diff < self.window_size)
        scores_masked = scores.masked_fill(~window_mask.view(1, 1, L, L), float('-inf'))
        attn_weights = F.softmax(scores_masked.float(), dim=-1).to(hidden_states.dtype)
        out_local = torch.matmul(attn_weights, value_states)

        # 2. Vectorized 2nd-Order PRIME Taylor Attention
        decay_mat = torch.where(
            causal_mask,
            torch.pow(self.decay, diff.float()),
            torch.zeros(L, L, device=device)
        ).view(1, 1, L, L)
        q_f32 = q_scaled.float()
        k_f32 = key_states.float()
        v_f32 = value_states.float()
        dot1 = torch.matmul(q_f32, k_f32.transpose(-1, -2))
        dot2 = 0.5 * torch.matmul(q_f32**2, (k_f32**2).transpose(-1, -2))
        A = decay_mat * (1.0 + dot1 + dot2)
        num_dense = torch.matmul(A, v_f32)
        den_dense = torch.sum(A, dim=-1, keepdim=True).clamp(min=1e-3)
        out_prime = (num_dense / den_dense).to(hidden_states.dtype)
        out_prime = self.prime_norm(out_prime)

        # 3. Adaptive Gated Fusion
        g = torch.sigmoid(self.gate)
        fused = (1.0 - g) * out_local + g * out_prime

        fused = fused.transpose(1, 2).contiguous().view(B, L, C)
        out = self.c_proj(fused)
        return out, None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150, help="Number of fine-tuning steps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--init-gate", type=float, default=-3.5)
    parser.add_argument("--data-bin", type=str, default="/data/datasets/prime_code_math_reasoning/train.bin")
    parser.add_argument("--output-path", type=str, default="/data/prime_checkpoints/gpt2_prime_sliding_window_calibrated.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("  CALIBRATING STOCK GPT-2 PRIME SLIDING WINDOW (FROZEN TRUNK + ADAPTIVE GATE)")
    print(f"  Steps: {args.steps} | Effective Batch: {args.batch_size * args.grad_accum} | SeqLen: {args.seq_len}")
    print(f"  Learning Rate: {args.lr} | Window Size W: {args.window_size} | Init Gate: {args.init_gate}")
    print(f"  Target Checkpoint: {args.output_path}")
    print("=" * 80 + "\n")

    # 1. Load Stock GPT-2 and apply surgery
    print("[1/3] Loading stock GPT-2 and transplanting Adaptive PRIME Sliding Window...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    for i in range(len(model.transformer.h)):
        orig = model.transformer.h[i].attn
        model.transformer.h[i].attn = VectorizedAdaptiveGPT2HybridPrimeAttention(
            orig, layer_idx=i, window_size=args.window_size, decay=0.9995, init_gate=args.init_gate
        )
    model.to(device)
    print("  Transplanted PRIME sliding window across all 12 layers.")

    # 2. Freeze the foundational trunk; train ONLY gate and prime_norm
    for p in model.parameters():
        p.requires_grad = False

    trainable_params = []
    for i in range(len(model.transformer.h)):
        model.transformer.h[i].attn.gate.requires_grad = True
        model.transformer.h[i].attn.prime_norm.weight.requires_grad = True
        model.transformer.h[i].attn.prime_norm.bias.requires_grad = True
        trainable_params.extend([
            model.transformer.h[i].attn.gate,
            model.transformer.h[i].attn.prime_norm.weight,
            model.transformer.h[i].attn.prime_norm.bias,
        ])

    num_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Frozen Trunk: {total_params - num_trainable:,} params frozen ({((total_params - num_trainable)/total_params)*100:.3f}%).")
    print(f"  Trainable Parameters: {num_trainable:,} params (gate + prime_norm).")

    # 3. Setup memory-mapped dataset
    print(f"\n[2/3] Loading memory-mapped tokens from {args.data_bin}...")
    data = np.memmap(args.data_bin, dtype=np.uint16, mode="r")
    chunk_size = args.seq_len + 1
    max_idx = len(data) - chunk_size
    print(f"  Total tokens available: {len(data):,} tokens.")

    def get_batch():
        idxs = np.random.randint(0, max_idx, size=args.batch_size)
        batch = np.stack([data[i : i + chunk_size] for i in idxs])
        t = torch.from_numpy(batch.astype(np.int64)).to(device)
        return t[:, :-1], t[:, 1:]

    # 4. Setup optimizer
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    
    def get_lr(step):
        warmup = 15
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = float(step - warmup) / float(max(1, args.steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    print(f"\n[3/3] Commencing Calibration Training ({args.steps} steps)...")
    model.train()
    t0 = time.time()
    losses = []

    for step in range(1, args.steps + 1):
        step_t0 = time.time()
        lr_factor = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * lr_factor

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(args.grad_accum):
            x, y = get_batch()
            out = model(x, labels=y)
            loss = out.loss / args.grad_accum
            loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        losses.append(step_loss)

        step_elapsed = time.time() - step_t0
        tok_s = (args.batch_size * args.seq_len * args.grad_accum) / max(0.001, step_elapsed)

        if step % 20 == 0 or step == 1 or step == args.steps:
            avg_loss = sum(losses[-20:]) / len(losses[-20:])
            gate_means = [torch.sigmoid(model.transformer.h[j].attn.gate).mean().item() for j in range(len(model.transformer.h))]
            mean_g = sum(gate_means) / len(gate_means)
            print(f"Step {step:4d}/{args.steps} | Loss: {step_loss:.4f} (Avg20: {avg_loss:.4f}) | Mean Gate g: {mean_g:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | {tok_s:6.1f} tok/s", flush=True)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(model.state_dict(), args.output_path)
    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print("  CALIBRATION COMPLETE")
    print(f"  Final Loss       : {step_loss:.4f} (Avg20: {avg_loss:.4f})")
    print(f"  Saved Weights to : {args.output_path}")
    print(f"  Total Duration   : {total_time:.1f} seconds ({total_time/60:.2f} mins)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
