#!/usr/bin/env python3
"""
Scientific Head-to-Head Benchmark: PRIME Moment Attention vs Standard Softmax Attention
========================================================================================
Compares convergence, loss trajectory, training speed, and memory between:
1. PRIME Moment Attention (2nd-order Taylor linear recurrence)
2. Standard Causal Softmax Attention (Standard Transformer)

Both models have identical:
- Architecture: 12 layers, 768 hidden, 12 heads, head dim 64 (~125M params)
- Tokenizer: GPT-2 (50,257 vocab)
- Data stream: roneneldan/TinyStories (identical token batches)
- Optimizer: AdamW, lr=1.5e-4, bfloat16 mixed precision
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer
from datasets import load_dataset

from prime_moment_attention import PrimeConfig, PrimeForCausalLM
from prime_moment_attention.model import RMSNorm, PrimeMLP


class SoftmaxCausalAttention(nn.Module):
    """Standard Transformer Multi-Head Self-Attention with Causal Mask."""
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, **kwargs):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # PyTorch SDPA (Scaled Dot-Product Attention) with causal mask
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)
        return self.o_proj(attn_out), None


class SoftmaxBlock(nn.Module):
    def __init__(self, config: PrimeConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = SoftmaxCausalAttention(config.hidden_size, config.num_heads, config.head_dim)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = PrimeMLP(config.hidden_size, config.intermediate_size)

    def forward(self, x: torch.Tensor, **kwargs):
        residual = x
        x = self.input_layernorm(x)
        attn_out, _ = self.self_attn(x)
        x = residual + attn_out

        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        return x, None


class SoftmaxForCausalLM(nn.Module):
    def __init__(self, config: PrimeConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([SoftmaxBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x, _ = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().to(torch.float32)
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        return {"loss": loss, "logits": logits}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("  HEAD-TO-HEAD SCIENTIFIC BENCHMARK: PRIME ATTENTION vs SOFTMAX ATTENTION")
    print(f"  Target Device: {device} | Precision: bfloat16 | Dataset: roneneldan/TinyStories")
    print("=" * 80)

    # 1. Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = len(tokenizer)

    # Identical Config (125M parameters)
    cfg = PrimeConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        head_dim=64,
        decay=0.995,
        eps=1.0,
    )

    # 2. Build Models with identical seed
    torch.manual_seed(42)
    prime_model = PrimeForCausalLM(cfg).to(device)

    torch.manual_seed(42)
    softmax_model = SoftmaxForCausalLM(cfg).to(device)

    params_prime = sum(p.numel() for p in prime_model.parameters()) / 1e6
    params_softmax = sum(p.numel() for p in softmax_model.parameters()) / 1e6
    print(f"[Model Setup] PRIME Model: {params_prime:.2f}M params | Softmax Model: {params_softmax:.2f}M params")

    # Optimizers
    lr = 1.5e-4
    opt_prime = AdamW(prime_model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    opt_softmax = AdamW(softmax_model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

    # 3. Stream identical data
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    buffer = []
    chunk_size = 129
    batch_size = 4
    num_steps = 25

    print(f"\n[Training] Running {num_steps} identical steps on both architectures (Batch Size: {batch_size}, Seq Len: 128)...\n")
    print(f"{'Step':>4} | {'PRIME Loss':>10} | {'Softmax Loss':>12} | {'Diff (PRIME-Softmax)':>20} | {'PRIME Time':>10} | {'Softmax Time':>12}")
    print("-" * 84)

    def get_batches():
        buf = []
        for s in ds:
            text = s.get("text", "")
            if not text.strip():
                continue
            tokens = tokenizer.encode(text) + [tokenizer.eos_token_id]
            buf.extend(tokens)
            while len(buf) >= batch_size * chunk_size:
                b = []
                for _ in range(batch_size):
                    b.append(buf[:chunk_size])
                    buf = buf[chunk_size:]
                yield torch.tensor(b, dtype=torch.long)

    batch_gen = get_batches()

    prime_model.train()
    softmax_model.train()

    prime_total_time = 0
    softmax_total_time = 0

    for step in range(1, num_steps + 1):
        batch = next(batch_gen).to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        # 1. Step PRIME
        opt_prime.zero_grad()
        t0 = time.time()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out_prime = prime_model(x, labels=y)
            loss_prime = out_prime["loss"]
        loss_prime.backward()
        torch.nn.utils.clip_grad_norm_(prime_model.parameters(), max_norm=1.0)
        opt_prime.step()
        dt_prime = time.time() - t0
        prime_total_time += dt_prime

        # 2. Step Softmax
        opt_softmax.zero_grad()
        t0 = time.time()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out_softmax = softmax_model(x, labels=y)
            loss_softmax = out_softmax["loss"]
        loss_softmax.backward()
        torch.nn.utils.clip_grad_norm_(softmax_model.parameters(), max_norm=1.0)
        opt_softmax.step()
        dt_softmax = time.time() - t0
        softmax_total_time += dt_softmax

        lp = loss_prime.item()
        ls = loss_softmax.item()
        diff = lp - ls

        if step % 5 == 0 or step == 1 or step == num_steps:
            print(f"{step:4d} | {lp:10.4f} | {ls:12.4f} | {diff:+20.4f} | {dt_prime:9.2f}s | {dt_softmax:11.2f}s")

    print("=" * 84)
    print("  BENCHMARK SUMMARY:")
    print(f"  PRIME Final Loss:   {lp:.4f}  (Total Step Time: {prime_total_time:.1f}s)")
    print(f"  Softmax Final Loss: {ls:.4f}  (Total Step Time: {softmax_total_time:.1f}s)")
    print(f"  Loss Difference:    {diff:+.4f}")
    print("=" * 84)


if __name__ == "__main__":
    main()
