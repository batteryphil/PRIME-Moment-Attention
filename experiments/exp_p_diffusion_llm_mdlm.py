#!/usr/bin/env python3
"""
Full Empirical Evaluation of PRIME Polynomial Recurrence on MDLM
================================================================
(Masked Diffusion Language Model -- NeurIPS 2024 by Sahoo et al.)
Model Checkpoint: TheQweaker/mdlm-owt-noflash (170M params, 12 DiT blocks)

Evaluates:
  Part 1: Masked Token Infilling Fidelity across 25%, 50%, 75% corruption
  Part 2: Full 64-step Ancestral Diffusion Text Generation
  Part 3: Latency, Throughput & Memory Benchmarks

Architectures:
  1. Standard Softmax (Full Baseline)
  2. PRIME Mid-Trunk Anchor (Blocks 4, 5, 6, 7 -- 4 of 12 blocks)
  3. PRIME Deep Half (Blocks 3 to 8 -- 6 of 12 blocks)
  4. 100% Zero-Shot PRIME (All 12 blocks)
"""

import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["MIOPEN_USER_DB_PATH"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["TRITON_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/.triton"

import time
import json
import types
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
snap = "/home/phil/.cache/huggingface/hub/models--TheQweaker--mdlm-owt-noflash/snapshots/8871c28982b8b256e1ccb45a94bf68caaf7c270f"

print("=" * 80)
print(" LOADING PRETRAINED MDLM DIFFUSION LLM (NEURIPS 2024)")
print(f" Checkpoint: {snap}")
print("=" * 80)

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForMaskedLM.from_pretrained(snap, trust_remote_code=True)
model = model.to(device).eval()

MASK_TOKEN_ID = 50257
num_blocks = len(model.backbone.blocks)
orig_forwards = [b.forward for b in model.backbone.blocks]

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def apply_rotary(x, cos, sin):
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin

def make_prime_forward(block, order=2, eps=1e-5):
    def prime_forward(x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        (shift_msa, scale_msa, gate_msa, shift_mlp,
         scale_mlp, gate_mlp) = block.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # --- attention ---
        x_skip = x
        x_norm = modulate(block.norm1(x), shift_msa, scale_msa)

        qkv = block.attn_qkv(x_norm).view(batch_size, seq_len, 3, block.n_heads, -1)
        cos, sin = rotary_cos_sin
        cos, sin = cos.to(qkv.dtype), sin.to(qkv.dtype)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        # [b, h, s, d]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        head_dim = q.shape[-1]
        q_scaled = q * (head_dim ** -0.5)

        dot = torch.matmul(q_scaled, k.transpose(-1, -2))

        if order == 1:
            kernel = 1.0 + dot
        elif order == 2:
            kernel = 1.0 + dot + 0.5 * (dot ** 2)
        else:
            kernel = 1.0 + dot + 0.5 * (dot ** 2) + (1.0 / 6.0) * (dot ** 3)

        kernel = F.relu(kernel)
        denom = kernel.sum(dim=-1, keepdim=True) + eps
        attn_weights = kernel / denom

        attn = torch.matmul(attn_weights, v)
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, -1)

        x = x_skip + gate_msa * block.attn_out(attn)

        # --- mlp ---
        x = x + gate_mlp * block.mlp(modulate(block.norm2(x), shift_mlp, scale_mlp))
        return x
    return prime_forward

def restore_attention():
    for i, b in enumerate(model.backbone.blocks):
        b.forward = orig_forwards[i]

def inject_prime(block_indices, order=2):
    restore_attention()
    for idx in block_indices:
        b = model.backbone.blocks[idx]
        b.forward = make_prime_forward(b, order=order)

architectures = [
    {
        "name": "Standard Softmax",
        "description": "Baseline MDLM with Scaled Dot-Product Attention",
        "blocks": []
    },
    {
        "name": "PRIME Mid-Trunk",
        "description": "PRIME Trunk Anchor (Blocks 4, 5, 6, 7 -- 4 of 12)",
        "blocks": [4, 5, 6, 7]
    },
    {
        "name": "PRIME Deep Half",
        "description": "PRIME Deep Backbone (Blocks 3 to 8 -- 6 of 12)",
        "blocks": list(range(3, 9))
    },
    {
        "name": "100% Zero-Shot PRIME",
        "description": "100% Zero-Shot PRIME (All 12 blocks)",
        "blocks": list(range(num_blocks))
    }
]

# ==============================================================================
# PART 1: MASKED TOKEN INFILLING BENCHMARK
# ==============================================================================
print("\n" + "=" * 80)
print(" PART 1: MASKED TOKEN INFILLING RECONSTRUCTION")
print("=" * 80)

eval_prompts = [
    "The theory of relativity usually encompasses two interrelated physics theories by Albert Einstein: special relativity and general relativity.",
    "In computer science, artificial intelligence refers to the intelligence of machines or software, as opposed to the intelligence of living beings.",
    "Deep learning architectures such as deep neural networks and recurrent neural networks have been applied to computer vision and natural language processing."
]

mask_rates = [0.25, 0.50, 0.75]
infilling_results = {}

for p_idx, text in enumerate(eval_prompts):
    print(f"\n[Prompt {p_idx + 1}]: {repr(text[:60])}...")
    token_ids = tok.encode(text, return_tensors="pt").to(device)
    L = token_ids.shape[1]
    print(f" Sequence Length: {L} tokens")

    for rate in mask_rates:
        print(f"\n  --- Mask Rate: {int(rate * 100)}% ---")
        torch.manual_seed(42)
        num_to_mask = max(1, int((L - 2) * rate))
        perm = torch.randperm(L - 2) + 1
        masked_positions = perm[:num_to_mask]

        corrupted = token_ids.clone()
        corrupted[0, masked_positions] = MASK_TOKEN_ID
        ground_truth = token_ids[0, masked_positions]

        for arch in architectures:
            inject_prime(arch["blocks"], order=2)

            t0 = time.time()
            with torch.no_grad():
                logits = model(input_ids=corrupted, return_dict=False)
            elapsed = (time.time() - t0) * 1000.0

            sub_logits = logits[0, masked_positions]
            pred_tokens = torch.argmax(sub_logits, dim=-1)

            top1_acc = (pred_tokens == ground_truth).float().mean().item()
            top5_preds = torch.topk(sub_logits, k=5, dim=-1).indices
            top5_acc = (top5_preds == ground_truth.unsqueeze(-1)).any(dim=-1).float().mean().item()
            ce_loss = F.cross_entropy(sub_logits, ground_truth).item()

            print(f"    {arch['name']:<22} | Top-1: {top1_acc*100:5.1f}% | Top-5: {top5_acc*100:5.1f}% | CE Loss: {ce_loss:5.2f} | Latency: {elapsed:5.2f}ms")

            key = f"prompt_{p_idx}_mask_{int(rate*100)}"
            if key not in infilling_results:
                infilling_results[key] = {}
            infilling_results[key][arch["name"]] = {
                "top1_acc": round(top1_acc, 4),
                "top5_acc": round(top5_acc, 4),
                "ce_loss": round(ce_loss, 4),
                "latency_ms": round(elapsed, 2)
            }

# ==============================================================================
# PART 2: FULL ANCESTRAL DIFFUSION SAMPLING (64 DIFFUSION STEPS)
# ==============================================================================
print("\n" + "=" * 80)
print(" PART 2: FULL ANCESTRAL DIFFUSION GENERATION (64 STEPS)")
print(" Starting from 100% masked context with prompt prefix")
print("=" * 80)

@torch.no_grad()
def generate_mdlm(model, *, seq_len=64, num_steps=64, prompt_ids=None,
                  temperature=0.8, top_k=50, seed=42):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.full((1, seq_len), MASK_TOKEN_ID, dtype=torch.long, device=device)
    prompt_len = 0
    if prompt_ids is not None:
        prompt_ids = prompt_ids.to(device)
        prompt_len = prompt_ids.numel()
        x[0, :prompt_len] = prompt_ids

    times = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    t0 = time.time()
    for i in range(num_steps):
        t, s = times[i].item(), times[i + 1].item()
        logits = model(input_ids=x, return_dict=False)
        logits[..., MASK_TOKEN_ID] = float("-inf")

        logits = logits / max(temperature, 1e-6)
        if top_k is not None:
            kth = logits.topk(top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        x0 = torch.multinomial(probs.view(-1, probs.shape[-1]), 1,
                               generator=generator).view(1, seq_len)

        is_mask = (x == MASK_TOKEN_ID)
        unmask_prob = (t - s) / t if t > 0 else 1.0
        reveal = is_mask & (torch.rand(x.shape, device=device, generator=generator) < unmask_prob)
        x = torch.where(reveal, x0, x)
        if prompt_len > 0:
            x[0, :prompt_len] = prompt_ids

    total_time = time.time() - t0
    ms_per_step = (total_time / num_steps) * 1000.0
    text = tok.decode(x[0].tolist(), skip_special_tokens=True)
    return text, total_time, ms_per_step

generation_prompt = "The future of artificial intelligence will"
prompt_tokens = tok.encode(generation_prompt, return_tensors="pt")
print(f"\nGeneration Prompt: {repr(generation_prompt)} ({prompt_tokens.shape[1]} tokens)")

generation_results = {}

for arch in architectures:
    print(f"\n--- Running 64-step Generation: {arch['name']} ---")
    inject_prime(arch["blocks"], order=2)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    text, total_time, ms_step = generate_mdlm(
        model, seq_len=64, num_steps=64, prompt_ids=prompt_tokens, temperature=0.8, top_k=50, seed=42
    )
    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"  Time: {total_time:.2f}s ({ms_step:.2f} ms/step) | Peak VRAM: {vram_mb:.1f} MB")
    print(f"  Generated Sample:\n  >>> {repr(text)}")

    generation_results[arch["name"]] = {
        "total_time_s": round(total_time, 2),
        "ms_per_step": round(ms_step, 2),
        "vram_mb": round(vram_mb, 1),
        "text": text
    }

# Save all results to JSON
full_output = {
    "model": "TheQweaker/mdlm-owt-noflash (NeurIPS 2024)",
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "infilling": infilling_results,
    "generation": generation_results
}

out_path = "/home/phil/.gemini/antigravity/scratch/mdlm_prime_full_results.json"
with open(out_path, "w") as f:
    json.dump(full_output, f, indent=2)

print("\n" + "=" * 80)
print(f"[+] COMPLETE! All MDLM Diffusion LLM results saved to: {out_path}")
print("=" * 80)
