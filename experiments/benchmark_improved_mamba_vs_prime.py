#!/usr/bin/env python3
"""
Upgraded Head-to-Head Benchmark: Calibrated PRIME-Selective vs Mamba vs Softmax
==============================================================================
Incorporates the exact engineering improvements proven in our 1B training:
1. Calibrated Inverse Temperature (beta = 6.0) matching QK-norm scaling.
2. Directional Taylor Moment Readout (preventing unweighted S0 baseline drift).
3. Dynamic Selective Gating (Delta_t -> alpha_t) arresting decay on noise gaps.
"""

import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class CalibratedSoftmaxMemory:
    """Exact Causal Softmax Attention with calibrated temperature beta."""
    def __init__(self, beta: float = 6.0):
        self.beta = beta
        
    def forward(self, keys: torch.Tensor, vals: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # keys: [N, D], vals: [N, D], query: [1, D] (unit normalized)
        scores = torch.matmul(query, keys.T) * self.beta # [1, N]
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, vals) # [1, D]


class MambaSSMMemory:
    """Mamba / SSD 1st-Order Selective Recurrent Memory."""
    def __init__(self, d_k: int, d_v: int, beta: float = 6.0):
        self.d_k = d_k
        self.d_v = d_v
        self.beta = beta
        
    def process_sequence(self, tokens_k, tokens_v, alphas):
        S1 = torch.zeros(self.d_k, self.d_v, device=device, dtype=torch.float32)
        L = tokens_k.shape[0]
        for t in range(L):
            a_t = alphas[t]
            kt = tokens_k[t]
            vt = tokens_v[t]
            S1 = a_t * S1 + torch.outer(kt, vt)
        return S1
        
    def readout(self, S1: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return torch.matmul(query, S1)


class UpgradedPurePRIMEMemory:
    """
    PRIME 2nd-Order Taylor Moment Memory (Static Decay, Directional Readout).
    Readout uses calibrated Taylor expansion: beta * (q^T k) + 0.5 * beta^2 (q^T k)^2.
    """
    def __init__(self, d_k: int, d_v: int, beta: float = 6.0, decay: float = 0.9995):
        self.d_k = d_k
        self.d_v = d_v
        self.beta = beta
        self.b_sq = 0.5 * (beta ** 2)
        self.decay = decay
        
    def process_sequence(self, tokens_k, tokens_v):
        S1 = torch.zeros(self.d_k, self.d_v, device=device, dtype=torch.float32)
        S2 = torch.zeros(self.d_k, self.d_v, device=device, dtype=torch.float32)
        L = tokens_k.shape[0]
        dec = self.decay
        
        for t in range(L):
            kt = tokens_k[t]
            vt = tokens_v[t]
            S1 = dec * S1 + self.beta * torch.outer(kt, vt)
            S2 = (dec**2) * S2 + self.b_sq * torch.outer(kt**2, vt)
            
        return (S1, S2)
        
    def readout(self, state, query: torch.Tensor) -> torch.Tensor:
        S1, S2 = state
        q_sq = query**2
        return torch.matmul(query, S1) + torch.matmul(q_sq, S2)


class UpgradedPRIMEMambaMemory:
    """
    PRIME-Selective (Higher-Order Selective State Space Model):
    Combines input-dependent selective gating (Delta_t -> alpha_t)
    with 2nd-order Taylor moment state (S1 + S2) and calibrated beta.
    """
    def __init__(self, d_k: int, d_v: int, beta: float = 6.0):
        self.d_k = d_k
        self.d_v = d_v
        self.beta = beta
        self.b_sq = 0.5 * (beta ** 2)
        
    def process_sequence(self, tokens_k, tokens_v, alphas):
        S1 = torch.zeros(self.d_k, self.d_v, device=device, dtype=torch.float32)
        S2 = torch.zeros(self.d_k, self.d_v, device=device, dtype=torch.float32)
        L = tokens_k.shape[0]
        
        for t in range(L):
            a_t = alphas[t]
            kt = tokens_k[t]
            vt = tokens_v[t]
            
            S1 = a_t * S1 + self.beta * torch.outer(kt, vt)
            S2 = (a_t**2) * S2 + self.b_sq * torch.outer(kt**2, vt)
            
        return (S1, S2)
        
    def readout(self, state, query: torch.Tensor) -> torch.Tensor:
        S1, S2 = state
        q_sq = query**2
        return torch.matmul(query, S1) + torch.matmul(q_sq, S2)


def run_improved_mqar(d_model: int = 64, seq_len: int = 512, num_pairs_list = [4, 8, 16, 32, 64], num_trials: int = 100, beta: float = 6.0):
    softmax_mem = CalibratedSoftmaxMemory(beta=beta)
    mamba_mem = MambaSSMMemory(d_model, d_model, beta=beta)
    pure_prime_mem = UpgradedPurePRIMEMemory(d_model, d_model, beta=beta, decay=0.9995)
    prime_mamba_mem = UpgradedPRIMEMambaMemory(d_model, d_model, beta=beta)
    
    print("\n" + "=" * 95)
    print(f" EXPERIMENT 1 (CALIBRATED): MULTI-QUERY ASSOCIATIVE RECALL (MQAR) | Dim={d_model} | Beta={beta}")
    print("=" * 95)
    print(f"{'Pairs (N)':<10} | {'Softmax':<12} | {'Mamba (1st)':<14} | {'Pure PRIME':<14} | {'PRIME-Mamba':<16} | Advantage vs Mamba")
    print("-" * 95)
    
    results = {}
    
    for N in num_pairs_list:
        acc = {"softmax": 0, "mamba": 0, "pure_prime": 0, "prime_mamba": 0}
        
        for trial in range(num_trials):
            torch.manual_seed(1000 * N + trial)
            
            keys = F.normalize(torch.randn(N, d_model, device=device), p=2, dim=-1)
            vals = F.normalize(torch.randn(N, d_model, device=device), p=2, dim=-1)
            
            seq_k = torch.randn(seq_len, d_model, device=device) * 0.05
            seq_v = torch.randn(seq_len, d_model, device=device) * 0.05
            alphas = torch.full((seq_len,), 0.9998, device=device)
            
            insert_positions = torch.randperm(int(seq_len * 0.8))[:N].sort().values
            for i, pos in enumerate(insert_positions):
                seq_k[pos] = keys[i]
                seq_v[pos] = vals[i]
                alphas[pos] = 0.98
                
            target_idx = torch.randint(0, N, (1,)).item()
            target_k = keys[target_idx:target_idx+1]
            
            out_sm = softmax_mem.forward(keys, vals, target_k).squeeze(0)
            
            S1_mamba = mamba_mem.process_sequence(seq_k, seq_v, alphas)
            out_m = mamba_mem.readout(S1_mamba, target_k).squeeze(0)
            
            st_p = pure_prime_mem.process_sequence(seq_k, seq_v)
            out_p = pure_prime_mem.readout(st_p, target_k).squeeze(0)
            
            st_pm = prime_mamba_mem.process_sequence(seq_k, seq_v, alphas)
            out_pm = prime_mamba_mem.readout(st_pm, target_k).squeeze(0)
            
            for name, vec in [("softmax", out_sm), ("mamba", out_m), ("pure_prime", out_p), ("prime_mamba", out_pm)]:
                norm_vec = F.normalize(vec.unsqueeze(0), p=2, dim=-1)
                sims = torch.matmul(norm_vec, vals.T).squeeze(0)
                if torch.argmax(sims).item() == target_idx:
                    acc[name] += 1
                    
        acc_pct = {k: (acc[k] / num_trials) * 100.0 for k in acc}
        adv = acc_pct["prime_mamba"] - acc_pct["mamba"]
        adv_str = f"+{adv:.1f}%" if adv >= 0 else f"{adv:.1f}%"
        print(f"{N:<10} | {acc_pct['softmax']:>5.1f}%      | {acc_pct['mamba']:>6.1f}%       | {acc_pct['pure_prime']:>6.1f}%       | {acc_pct['prime_mamba']:>8.1f}%      | {adv_str}")
        results[str(N)] = acc_pct
        
    return results


def run_improved_noise_filtering(d_model: int = 64, noise_gap_list = [100, 250, 500, 1000, 2000], num_trials: int = 100, beta: float = 6.0):
    mamba_mem = MambaSSMMemory(d_model, d_model, beta=beta)
    pure_prime_mem = UpgradedPurePRIMEMemory(d_model, d_model, beta=beta, decay=0.9995)
    prime_mamba_mem = UpgradedPRIMEMambaMemory(d_model, d_model, beta=beta)
    
    print("\n" + "=" * 95)
    print(f" EXPERIMENT 2 (CALIBRATED): NOISE TOKEN RESISTANCE OVER EXTENDED GAPS | Dim={d_model} | Beta={beta}")
    print("=" * 95)
    print(f"{'Gap Tokens':<12} | {'Mamba (1st)':<14} | {'Pure PRIME':<14} | {'PRIME-Mamba':<16} | Mechanism")
    print("-" * 95)
    
    results = {}
    for gap in noise_gap_list:
        cos_sims = {"mamba": [], "pure_prime": [], "prime_mamba": []}
        
        for trial in range(num_trials):
            torch.manual_seed(2000 * gap + trial)
            target_k = F.normalize(torch.randn(1, d_model, device=device), p=2, dim=-1)
            target_v = F.normalize(torch.randn(1, d_model, device=device), p=2, dim=-1)
            
            seq_len = 1 + gap
            seq_k = torch.randn(seq_len, d_model, device=device) * 0.05
            seq_v = torch.randn(seq_len, d_model, device=device) * 0.05
            
            seq_k[0] = target_k[0]
            seq_v[0] = target_v[0]
            
            alphas = torch.full((seq_len,), 0.9999, device=device)
            alphas[0] = 0.98
            
            S1_m = mamba_mem.process_sequence(seq_k, seq_v, alphas)
            out_m = mamba_mem.readout(S1_m, target_k).squeeze(0)
            
            st_p = pure_prime_mem.process_sequence(seq_k, seq_v)
            out_p = pure_prime_mem.readout(st_p, target_k).squeeze(0)
            
            st_pm = prime_mamba_mem.process_sequence(seq_k, seq_v, alphas)
            out_pm = prime_mamba_mem.readout(st_pm, target_k).squeeze(0)
            
            cos_sims["mamba"].append(F.cosine_similarity(out_m.unsqueeze(0), target_v).item())
            cos_sims["pure_prime"].append(F.cosine_similarity(out_p.unsqueeze(0), target_v).item())
            cos_sims["prime_mamba"].append(F.cosine_similarity(out_pm.unsqueeze(0), target_v).item())
            
        m_cos = float(np.mean(cos_sims["mamba"]))
        p_cos = float(np.mean(cos_sims["pure_prime"]))
        pm_cos = float(np.mean(cos_sims["prime_mamba"]))
        
        mech = "Selective gating preserves memory!" if pm_cos > 0.6 else "Degraded"
        print(f"{gap:<12} | {m_cos:>6.4f}         | {p_cos:>6.4f}         | {pm_cos:>8.4f}         | {mech}")
        results[str(gap)] = {"mamba": round(m_cos, 4), "pure_prime": round(p_cos, 4), "prime_mamba": round(pm_cos, 4)}
        
    return results


def main():
    print("=" * 95)
    print(" CALIBRATED EMPIRICAL BENCHMARK: PRIME-MAMBA vs MAMBA vs SOFTMAX")
    print(f" Execution Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 95)
    
    mqar_res = run_improved_mqar(d_model=64, seq_len=512, num_pairs_list=[4, 8, 16, 32, 64], num_trials=100, beta=6.0)
    noise_res = run_improved_noise_filtering(d_model=64, noise_gap_list=[100, 250, 500, 1000, 2000], num_trials=100, beta=6.0)
    
    out_file = "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/experiments/calibrated_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump({"mqar": mqar_res, "noise_gaps": noise_res}, f, indent=2)
    print(f"\n[+] Saved calibrated results to: {out_file}")

if __name__ == "__main__":
    main()
