#!/usr/bin/env python3
"""
INDEPENDENT PRIME NIAH BENCHMARK
=================================
No fabricated scores. No hardcoded formulas.
Runs the actual PRIME 2nd-order moment recurrence, linear attention,
and softmax attention on the same token vectors and measures real
cosine similarity of the retrieved value vs the planted needle value.

Three honest tests:
  1. Vector-level retention: random key/value pairs, measure cosine sim after N noise tokens
  2. Decay curve: plot survival at gaps 50 → 100,000 tokens
  3. Multi-needle: plant 2 needles at different depths, check both survive

This does NOT require a language model. It tests the PRIME mechanism itself.
"""

import math
import time
import json
import torch
import torch.nn.functional as F

print("=" * 80)
print("  INDEPENDENT PRIME NIAH BENCHMARK")
print("  Running actual PRIME recurrence — no hardcoded scores")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float32
print(f"  Device: {device} | Dtype: {dtype}\n")

DECAY = 0.9995   # repo default
SEED  = 1337

# ─────────────────────────────────────────────────────────────────────────────
# Core PRIME recurrence (exact repo formulation from surgery.py)
# ─────────────────────────────────────────────────────────────────────────────
def run_prime(keys, values, query, decay=DECAY):
    """
    keys:   [L, D]  — all tokens in sequence order
    values: [L, D]
    query:  [D]     — the probe query (same as needle key, scaled)
    Returns: retrieved value vector [D]
    """
    D = keys.shape[1]
    S0 = torch.zeros(D,    device=device, dtype=torch.float32)
    S1 = torch.zeros(D, D, device=device, dtype=torch.float32)
    S2 = torch.zeros(D, D, device=device, dtype=torch.float32)
    K0 = torch.zeros(1,    device=device, dtype=torch.float32)
    K1 = torch.zeros(D,    device=device, dtype=torch.float32)
    K2 = torch.zeros(D,    device=device, dtype=torch.float32)

    scale = 1.0 / math.sqrt(D)

    for t in range(keys.shape[0]):
        kt = keys[t].float()
        vt = values[t].float()
        S0 = decay * S0 + vt
        S1 = decay * S1 + torch.outer(kt, vt)
        S2 = decay * S2 + torch.outer(kt ** 2, vt)
        K0 = decay * K0 + 1.0
        K1 = decay * K1 + kt
        K2 = decay * K2 + kt ** 2

    q = query.float() * scale
    num = S0 + (q @ S1) + 0.5 * (q ** 2 @ S2)
    den = (K0 + (q * K1).sum() + 0.5 * ((q**2) * K2).sum()).clamp(min=1e-3)
    return num / den


def run_linear(keys, values, query, decay=DECAY):
    """Linear attention (ELU+1 kernel) for comparison."""
    D = keys.shape[1]
    scale = 1.0 / math.sqrt(D)
    kf = F.elu(keys) + 1.0
    qf = (F.elu(query * scale) + 1.0).float()
    S  = torch.zeros(D, D, device=device, dtype=torch.float32)
    z  = torch.zeros(D,    device=device, dtype=torch.float32)
    for t in range(keys.shape[0]):
        kt = kf[t].float()
        vt = values[t].float()
        S  = decay * S + torch.outer(kt, vt)
        z  = decay * z + kt
    num = qf @ S
    den = (qf * z).sum().clamp(min=1e-5)
    return num / den


def run_softmax(keys, values, query):
    """Full softmax attention (the ground truth, O(L) memory)."""
    scale = 1.0 / math.sqrt(keys.shape[1])
    scores = (query.float() @ keys.float().T) * scale   # [L]
    attn   = F.softmax(scores, dim=0)                   # [L]
    return (attn.unsqueeze(0) @ values.float()).squeeze(0)


def cosine(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Single-needle retention across token gap sizes
# ─────────────────────────────────────────────────────────────────────────────
print("\n── TEST 1: Single-Needle Retention vs Token Gap ─────────────────────────")
print(f"{'Gap':>10} | {'Softmax':>10} | {'Linear':>10} | {'PRIME':>10} | {'PRIME/Linear':>14}")
print("-" * 62)

DIMS  = [64, 128]   # test both
gaps  = [50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000]
results_t1 = []

D = 128   # use 128-dim vectors (realistic head_dim)
torch.manual_seed(SEED)

for gap in gaps:
    torch.manual_seed(SEED + gap)
    needle_k = torch.randn(D, device=device) * 2.0   # make needle distinctive
    needle_v = torch.randn(D, device=device) * 2.0
    noise_k  = torch.randn(gap, D, device=device)
    noise_v  = torch.randn(gap, D, device=device)

    all_k = torch.cat([needle_k.unsqueeze(0), noise_k], dim=0)   # [gap+1, D]
    all_v = torch.cat([needle_v.unsqueeze(0), noise_v], dim=0)

    query = needle_k.clone()   # probe: we know the needle key

    t0 = time.time()
    out_soft  = run_softmax(all_k, all_v, query)
    out_lin   = run_linear (all_k, all_v, query)
    out_prime = run_prime  (all_k, all_v, query)
    elapsed   = time.time() - t0

    cs = cosine(out_soft,  needle_v)
    cl = cosine(out_lin,   needle_v)
    cp = cosine(out_prime, needle_v)

    ratio = f"{cp/max(cl,1e-4):.2f}x" if cl > 0 else "∞"
    print(f"{gap:>10,} | {cs:>10.4f} | {cl:>10.4f} | {cp:>10.4f} | {ratio:>14}  ({elapsed:.2f}s)")

    results_t1.append({"gap": gap, "softmax": cs, "linear": cl, "prime": cp})

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Dual-needle at different depths
# ─────────────────────────────────────────────────────────────────────────────
print("\n── TEST 2: Dual-Needle (two needles at different depths) ────────────────")
print(f"{'Horizon':>10} | {'Needle-A depth':>16} | {'Needle-B depth':>16} | {'PRIME-A':>9} | {'PRIME-B':>9}")
print("-" * 72)

results_t2 = []
horizons = [1000, 5000, 10000, 50000]

for L in horizons:
    torch.manual_seed(SEED + L)

    # Needle A at 10% depth
    pos_a = int(L * 0.10)
    # Needle B at 50% depth
    pos_b = int(L * 0.50)

    nk_a = torch.randn(D, device=device) * 2.0
    nv_a = torch.randn(D, device=device) * 2.0
    nk_b = torch.randn(D, device=device) * 2.0
    nv_b = torch.randn(D, device=device) * 2.0

    noise_k = torch.randn(L, D, device=device)
    noise_v = torch.randn(L, D, device=device)

    all_k = noise_k.clone()
    all_v = noise_v.clone()
    all_k[pos_a] = nk_a
    all_v[pos_a] = nv_a
    all_k[pos_b] = nk_b
    all_v[pos_b] = nv_b

    out_a = run_prime(all_k, all_v, nk_a)
    out_b = run_prime(all_k, all_v, nk_b)

    ca = cosine(out_a, nv_a)
    cb = cosine(out_b, nv_b)

    print(f"{L:>10,} | {pos_a:>10,} (10%) | {pos_b:>10,} (50%) | {ca:>9.4f} | {cb:>9.4f}")
    results_t2.append({"horizon": L, "pos_a": pos_a, "pos_b": pos_b, "prime_a": ca, "prime_b": cb})

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Decay constant sensitivity
# ─────────────────────────────────────────────────────────────────────────────
print("\n── TEST 3: Decay Constant Sensitivity (gap = 10,000 tokens) ────────────")
print(f"{'Decay':>10} | {'PRIME cosine':>14} | {'Surviving %':>12}")
print("-" * 42)

GAP = 10000
torch.manual_seed(SEED)
needle_k = torch.randn(D, device=device) * 2.0
needle_v = torch.randn(D, device=device) * 2.0
noise_k  = torch.randn(GAP, D, device=device)
noise_v  = torch.randn(GAP, D, device=device)
all_k = torch.cat([needle_k.unsqueeze(0), noise_k], dim=0)
all_v = torch.cat([needle_v.unsqueeze(0), noise_v], dim=0)

results_t3 = []
for decay in [0.999, 0.9995, 0.9999, 0.99999, 1.0]:
    out = run_prime(all_k, all_v, needle_k, decay=decay)
    c   = cosine(out, needle_v)
    theoretical_decay = decay ** GAP
    print(f"{decay:>10.5f} | {c:>14.4f} | {theoretical_decay*100:>10.4f}%  (theoretical token survival)")
    results_t3.append({"decay": decay, "cosine": c, "theoretical_token_survival": theoretical_decay})

# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────
out = {
    "benchmark": "INDEPENDENT PRIME NIAH — No fabricated scores",
    "methodology": "Actual PyTorch PRIME recurrence, cosine similarity of retrieved vs planted needle value vector",
    "device": device,
    "head_dim_D": D,
    "decay_default": DECAY,
    "test1_single_needle": results_t1,
    "test2_dual_needle":   results_t2,
    "test3_decay_sensitivity": results_t3,
}
out_path = "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/experiments/independent_niah_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\n[+] Results saved to: {out_path}")
print("=" * 80)
print("  KEY: These are REAL cosine similarity measurements, not formulas.")
print("  'PRIME cosine' = cosine_sim(PRIME_retrieved_vector, planted_needle_value)")
print("  1.0 = perfect recall | 0.0 = random | negative = wrong direction")
print("=" * 80)
