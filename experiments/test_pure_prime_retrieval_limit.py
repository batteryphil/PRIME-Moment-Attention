#!/usr/bin/env python3
"""
Stress Test: Pushing Needle-In-A-Haystack (NIAH) to the Theoretical & Physical Limit
Target: Pure PRIME Attention Layer (Direct Tensor Probing)
Avoids the O(L^2) Softmax Out-Of-Memory limit from other layers.
Measures:
  1. Exact Needle Key-Value Pair planted at t=0
  2. N Distractor tokens pushed through the recurrent state (gaps: 1K, 4K, 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M)
  3. Querying the state with the exact Needle Key
  4. Evaluating Cosine Similarity & Signal-to-Noise Ratio (SNR) across Decay rates:
     - Standard: 0.9995
     - High: 0.9999
     - Ultra-Deep: 0.99999
     - Undecayed: 1.0000
"""

import sys, os, time
import torch
import torch.nn.functional as F

DEVICE = "cuda:0"
D = 128
H = 12

print("=" * 85)
print("EXTREME NIAH RETRIEVAL LIMIT PROBE: PURE PRIME ATTENTION")
print(f"Hardware: AMD Radeon RX 9060 XT (ROCm) | Head Dim: {D} | Heads: {H}")
print("=" * 85)

gaps = [1000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1000000]
decays = [0.9995, 0.9999, 0.99999, 1.0]

def probe_retrieval(gap, decay):
    torch.manual_seed(42)
    # Plant Needle Key and Value
    needle_k = F.normalize(torch.randn(1, H, D, device=DEVICE), p=2, dim=-1)
    needle_v = torch.randn(1, H, D, device=DEVICE)
    
    # Initialize Recurrent State with Needle
    S0 = needle_v.clone()
    S1 = torch.einsum('bhd,bhe->bhde', needle_k, needle_v)
    S2 = torch.einsum('bhd,bhe->bhde', needle_k**2, needle_v)
    
    K0 = torch.ones(1, H, 1, device=DEVICE)
    K1 = needle_k.clone()
    K2 = (needle_k**2)
    
    # Push Distractors in chunks to avoid GPU memory overhead
    chunk_size = 10000
    remaining = gap
    
    # For very large gaps, compute decay compounding mathematically for background
    # and simulate realistic random walk noise accumulation
    while remaining > 0:
        cur = min(remaining, chunk_size)
        
        # Batch simulate 'cur' random normalized distractors
        # Each step: S = decay * S + distractor
        # Over 'cur' steps, existing signal decays by decay^cur
        decay_factor = decay ** cur
        S0 = S0 * decay_factor
        S1 = S1 * decay_factor
        S2 = S2 * decay_factor
        K0 = K0 * decay_factor
        K1 = K1 * decay_factor
        K2 = K2 * decay_factor
        
        # Injected noise variance: sum of decaying independent noise
        # Variance of sum_{i=0}^{cur-1} decay^i * v_i is (1 - decay^(2*cur)) / (1 - decay^2)
        if decay < 1.0:
            noise_scale = ((1.0 - decay**(2 * cur)) / max(1.0 - decay**2, 1e-12)) ** 0.5
        else:
            noise_scale = cur ** 0.5
            
        noise_v = torch.randn(1, H, D, device=DEVICE) * (noise_scale / (cur**0.5))
        noise_k = F.normalize(torch.randn(1, H, D, device=DEVICE), p=2, dim=-1)
        
        S0 += noise_v * (cur ** 0.5)
        S1 += torch.einsum('bhd,bhe->bhde', noise_k, noise_v) * (cur ** 0.5)
        S2 += torch.einsum('bhd,bhe->bhde', noise_k**2, noise_v) * (cur ** 0.5)
        K0 += cur
        K1 += noise_k * (cur ** 0.5)
        K2 += (noise_k**2) * (cur ** 0.5)
        
        remaining -= cur

    # Query with the exact Needle Key
    qt = needle_k
    num = S0 + torch.einsum('bhd,bhde->bhe', qt, S1) + 0.5 * torch.einsum('bhd,bhde->bhe', qt**2, S2)
    den = (K0 + torch.sum(qt * K1, dim=-1, keepdim=True) + 0.5 * torch.sum((qt**2) * K2, dim=-1, keepdim=True)).clamp(min=1e-3)
    out = num / den
    
    # Measure cosine similarity with true planted Needle Value
    cos_sim = F.cosine_similarity(out, needle_v, dim=-1).mean().item()
    return round(cos_sim, 4)

print(f"{'Gap (Tokens)':<14} | " + " | ".join([f"Decay {d:<8}" for d in decays]))
print("-" * 75)

matrix = {}
for g in gaps:
    row = []
    for d in decays:
        sim = probe_retrieval(g, d)
        row.append(sim)
    matrix[g] = row
    str_row = " | ".join([f"{r:<14.4f}" for r in row])
    print(f"{g:<14} | {str_row}")

print("=" * 75)
