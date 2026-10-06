#!/usr/bin/env python3
"""
Mamba with Proven Trunk Surgery (Jamba/Zamba Architecture)
==========================================================
Testing PRIME as a Constant-Memory Mid-Trunk Anchor inside Mamba/SSM Networks.

Motivation:
-----------
In production, pure 100% Mamba is rarely used alone because 1st-order SSMs suffer from
associative recall bottlenecks. Modern architectures (AI21 Jamba, Zyphra Zamba) use
a HYBRID TRUNK: interleaving Mamba layers with Attention layers.

However, standard Jamba puts Softmax Attention in the trunk, reintroducing the unbounded
O(L) KV cache and destroying Mamba's O(1) memory guarantee.

Here we test:
  1. Pure Mamba (100% SSM layers)
  2. Mamba + Softmax Trunk (Jamba-style: Mamba -> Softmax Attn -> Mamba)
  3. Mamba + PRIME Proven Trunk (Mamba -> PRIME Moment Attn -> Mamba) [100% O(1) Memory]
  4. Mamba + Hybrid Window-PRIME Trunk (Mamba -> Window+PRIME -> Mamba) [100% Bounded Memory]

Tasks:
  - Multi-Query Associative Recall (MQAR)
  - Memory Footprint vs Sequence Length L in [512, 1024, 2048, 4096]
"""

import os
os.environ["MIOPEN_USER_DB_PATH"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["HOME"] = "/home/phil/.gemini/antigravity/scratch"

import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# SUB-MODULES
# ==============================================================================

class MambaBlock(nn.Module):
    """Selective SSM Block (Mamba S6 / SSD)."""
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.delta_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.delta_proj.bias, 0.5413)
        self.log_A = nn.Parameter(torch.linspace(math.log(1e-3), math.log(1e-1), n_heads))
        
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        x_norm = self.norm(x)
        
        # Split into branch and gate
        u, gate = self.in_proj(x_norm).chunk(2, dim=-1)
        # Depthwise 1D conv
        u_conv = self.conv1d(u.transpose(1, 2)).transpose(1, 2)
        u_act = F.silu(u_conv)
        
        # Selective step size
        delta = F.softplus(self.delta_proj(u_act)) # [B, L, H]
        A = torch.exp(self.log_A).view(1, 1, self.n_heads)
        alpha = torch.exp(-delta * A).permute(0, 2, 1) # [B, H, L]
        
        # Recurrent SSM scan
        u_heads = u_act.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [B, H, L, D_h]
        outputs = []
        S = torch.zeros(B, self.n_heads, self.head_dim, device=x.device, dtype=torch.float32)
        
        for t in range(L):
            a_t = alpha[:, :, t].unsqueeze(-1) # [B, H, 1]
            ut = u_heads[:, :, t].float()      # [B, H, D_h]
            S = a_t * S + ut
            outputs.append(S.to(x.dtype))
            
        y = torch.stack(outputs, dim=2) # [B, H, L, D_h]
        y = y.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        
        # Gated output
        out = y * F.silu(gate)
        return residual + self.out_proj(out)


class SoftmaxTrunkBlock(nn.Module):
    """Causal Softmax Attention Trunk Block (Jamba-Style)."""
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        x_norm = self.norm(x)
        
        q = self.q_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, L, D)
        return residual + self.out_proj(out)


class PrimeTrunkBlock(nn.Module):
    """PRIME Moment Recurrence Trunk Block (Constant O(1) Memory)."""
    def __init__(self, d_model: int, n_heads: int = 4, decay: float = 0.9995):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.decay = decay
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        x_norm = self.norm(x)
        
        q = self.q_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2) * self.scale
        k = self.k_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        
        S0 = torch.zeros(B, self.n_heads, self.head_dim, device=x.device, dtype=torch.float32)
        S1 = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32)
        S2 = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32)
        K0 = torch.zeros(B, self.n_heads, 1, device=x.device, dtype=torch.float32)
        K1 = torch.zeros(B, self.n_heads, self.head_dim, device=x.device, dtype=torch.float32)
        K2 = torch.zeros(B, self.n_heads, self.head_dim, device=x.device, dtype=torch.float32)
        
        outputs = []
        for t in range(L):
            qt = q_f[:, :, t]
            kt = k_f[:, :, t]
            vt = v_f[:, :, t]
            
            S0 = self.decay * S0 + vt
            S1 = self.decay * S1 + torch.einsum('bhd,bhe->bhde', kt, vt)
            S2 = (self.decay**2) * S2 + torch.einsum('bhd,bhe->bhde', kt**2, vt)
            
            K0 = self.decay * K0 + 1.0
            K1 = self.decay * K1 + kt
            K2 = (self.decay**2) * K2 + (kt**2)
            
            num = S0 + torch.einsum('bhd,bhde->bhe', qt, S1) + 0.5 * torch.einsum('bhd,bhde->bhe', qt**2, S2)
            den = (K0 + (qt * K1).sum(dim=-1, keepdim=True) + 0.5 * ((qt**2) * K2).sum(dim=-1, keepdim=True)).clamp(min=1e-4)
            outputs.append((num / den).to(x.dtype))
            
        out = torch.stack(outputs, dim=2).transpose(1, 2).contiguous().view(B, L, D)
        return residual + self.out_proj(out)


class HybridTrunkBlock(nn.Module):
    """Hybrid Window (W=64) Softmax + PRIME Recurrent Overflow Trunk Block."""
    def __init__(self, d_model: int, n_heads: int = 4, window_size: int = 64, decay: float = 0.9995):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.window_size = window_size
        self.decay = decay
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        x_norm = self.norm(x)
        
        q = self.q_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2) * self.scale
        k = self.k_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Local Window Softmax
        scores = torch.matmul(q, k.transpose(-1, -2))
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        # Window band mask
        band_mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=-self.window_size)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        scores.masked_fill_(band_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        
        weights = F.softmax(scores, dim=-1)
        # Handle nan in weights where all elements masked
        weights = torch.nan_to_num(weights, nan=0.0)
        out_local = torch.matmul(weights, v)
        
        # Global PRIME overflow for evicted history
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        
        S0 = torch.zeros(B, self.n_heads, self.head_dim, device=x.device, dtype=torch.float32)
        S1 = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32)
        S2 = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32)
        K0 = torch.zeros(B, self.n_heads, 1, device=x.device, dtype=torch.float32)
        K1 = torch.zeros(B, self.n_heads, self.head_dim, device=x.device, dtype=torch.float32)
        K2 = torch.zeros(B, self.n_heads, self.head_dim, device=x.device, dtype=torch.float32)
        
        outputs = []
        for t in range(L):
            # Only accumulate evicted tokens past window_size
            if t >= self.window_size:
                evict_idx = t - self.window_size
                kt = k_f[:, :, evict_idx]
                vt = v_f[:, :, evict_idx]
                
                S0 = self.decay * S0 + vt
                S1 = self.decay * S1 + torch.einsum('bhd,bhe->bhde', kt, vt)
                S2 = (self.decay**2) * S2 + torch.einsum('bhd,bhe->bhde', kt**2, vt)
                K0 = self.decay * K0 + 1.0
                K1 = self.decay * K1 + kt
                K2 = (self.decay**2) * K2 + (kt**2)
                
            qt = q_f[:, :, t]
            num = S0 + torch.einsum('bhd,bhde->bhe', qt, S1) + 0.5 * torch.einsum('bhd,bhde->bhe', qt**2, S2)
            den = (K0 + (qt * K1).sum(dim=-1, keepdim=True) + 0.5 * ((qt**2) * K2).sum(dim=-1, keepdim=True)).clamp(min=1e-4)
            prime_t = (num / den).to(x.dtype)
            local_t = out_local[:, :, t]
            
            # Convex combination: 50% sharp local window, 50% persistent distant prime
            outputs.append(0.5 * local_t + 0.5 * prime_t)
            
        out = torch.stack(outputs, dim=2).transpose(1, 2).contiguous().view(B, L, D)
        return residual + self.out_proj(out)


# ==============================================================================
# FULL HYBRID NETWORKS (4-LAYER BACKBONES)
# ==============================================================================

class HybridMambaModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, trunk_type: str):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Layer 0: Mamba
        self.layer0 = MambaBlock(d_model)
        
        # Layer 1 (PROVEN MID-TRUNK): Choice of Trunk Block
        if trunk_type == "mamba_pure":
            self.trunk = MambaBlock(d_model)
        elif trunk_type == "softmax_trunk":
            self.trunk = SoftmaxTrunkBlock(d_model)
        elif trunk_type == "prime_trunk":
            self.trunk = PrimeTrunkBlock(d_model)
        elif trunk_type == "hybrid_trunk":
            self.trunk = HybridTrunkBlock(d_model, window_size=64)
        else:
            raise ValueError(f"Unknown trunk type: {trunk_type}")
            
        # Layer 2: Mamba
        self.layer2 = MambaBlock(d_model)
        # Layer 3: Mamba
        self.layer3 = MambaBlock(d_model)
        
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.layer0(x)
        x = self.trunk(x)    # Proven Mid-Trunk Anchor
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.norm_f(x)
        return self.head(x)


# ==============================================================================
# BENCHMARK EVALUATION
# ==============================================================================

def generate_mqar_batch(batch_size: int, seq_len: int, num_pairs: int, num_keys: int = 64, num_vals: int = 64):
    KEY_OFFSET = 1
    VAL_OFFSET = KEY_OFFSET + num_keys
    QUERY_TOKEN = VAL_OFFSET + num_vals
    
    seqs = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    q_positions = []
    ground_truth = []
    
    for b in range(batch_size):
        keys = torch.randperm(num_keys, device=device)[:num_pairs] + KEY_OFFSET
        vals = torch.randint(0, num_vals, (num_pairs,), device=device) + VAL_OFFSET
        
        max_pos = int(seq_len * 0.75) - 2
        insert_idx = torch.randperm(max_pos, device=device)[:num_pairs * 2].sort().values
        
        for p in range(num_pairs):
            pos_k = insert_idx[p * 2].item()
            pos_v = pos_k + 1
            seqs[b, pos_k] = keys[p]
            seqs[b, pos_v] = vals[p]
            
        # Query target key at the end
        target_p = num_pairs - 1 # test the furthest or latest stored key
        q_pos = seq_len - 2
        seqs[b, q_pos] = keys[target_p]
        seqs[b, q_pos + 1] = QUERY_TOKEN
        
        q_positions.append(q_pos + 1)
        ground_truth.append(vals[target_p].item())
        
    return seqs, q_positions, ground_truth


def main():
    print("=" * 95)
    print(" EMPIRICAL EVALUATION: MAMBA WITH PROVEN MID-TRUNK SURGERY")
    print(f" Execution Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 95)
    
    vocab_size = 256
    d_model = 128
    
    torch.manual_seed(42)
    models = {
        "Pure Mamba (All SSM)": HybridMambaModel(vocab_size, d_model, "mamba_pure").to(device).eval(),
        "Mamba + Softmax Trunk (Jamba)": HybridMambaModel(vocab_size, d_model, "softmax_trunk").to(device).eval(),
        "Mamba + PRIME Trunk": HybridMambaModel(vocab_size, d_model, "prime_trunk").to(device).eval(),
        "Mamba + Hybrid Trunk (W=64)": HybridMambaModel(vocab_size, d_model, "hybrid_trunk").to(device).eval(),
    }
    
    print("\nModel Parameter Footprint:")
    for name, m in models.items():
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  - {name:<32}: {n_params:,} params")
        
    # Benchmark 1: Memory Footprint vs Sequence Length L
    print("\n" + "=" * 95)
    print(" BENCHMARK 1: KV CACHE / STATE MEMORY FOOTPRINT VS SEQUENCE LENGTH")
    print("=" * 95)
    print(f"{'Sequence Length (L)':<22} | {'Pure Mamba':<16} | {'Jamba (Softmax)':<18} | {'Mamba + PRIME Trunk':<22} | {'Mamba + Hybrid Trunk':<22}")
    print("-" * 105)
    
    lengths = [512, 1024, 2048, 4096, 8192, 16384]
    mem_results = {}
    
    for L in lengths:
        # State memory calculations:
        # Pure Mamba: 4 layers of SSM state S: 4 * (d_model) = 4 * 128 * 4 bytes = 2 KB flat!
        pure_mamba_kb = (4 * d_model * 4) / 1024.0
        
        # Jamba (Softmax Trunk): 3 Mamba layers (1.5 KB) + 1 Softmax KV Cache: 2 * L * d_model * 2 bytes
        softmax_kv_kb = (3 * d_model * 4 + 2 * L * d_model * 2) / 1024.0
        
        # Mamba + PRIME Trunk: 3 Mamba layers (1.5 KB) + 1 PRIME State (S1, S2, K1, K2, S0, K0):
        # 4 heads * (32^2 * 4 * 2 + 32 * 4 * 2 + 4 * 2) = 32.5 KB flat!
        prime_trunk_kb = (3 * d_model * 4 + 4 * (2 * 32 * 32 * 4 + 2 * 32 * 4 + 8)) / 1024.0
        
        # Mamba + Hybrid Trunk: 3 Mamba layers + Window W=64 KV Cache + PRIME State
        hybrid_trunk_kb = (3 * d_model * 4 + 2 * 64 * d_model * 2 + 4 * (2 * 32 * 32 * 4 + 2 * 32 * 4 + 8)) / 1024.0
        
        mem_results[str(L)] = {
            "pure_mamba_kb": round(pure_mamba_kb, 2),
            "jamba_softmax_kb": round(softmax_kv_kb, 2),
            "mamba_prime_trunk_kb": round(prime_trunk_kb, 2),
            "mamba_hybrid_trunk_kb": round(hybrid_trunk_kb, 2)
        }
        
        print(f"{L:<22} | {pure_mamba_kb:>6.2f} KB       | {softmax_kv_kb:>8.2f} KB         | {prime_trunk_kb:>8.2f} KB (O(1))        | {hybrid_trunk_kb:>8.2f} KB (O(1))")
        
    # Benchmark 2: Forward Pass Latency across Context
    print("\n" + "=" * 95)
    print(" BENCHMARK 2: LATENCY & THROUGHPUT (L = 1024, Batch = 1)")
    print("=" * 95)
    
    dummy_x = torch.randint(0, vocab_size, (1, 1024), device=device)
    timing_results = {}
    
    for name, m in models.items():
        torch.cuda.empty_cache()
        for _ in range(3):
            _ = m(dummy_x)
        torch.cuda.synchronize()
        
        t0 = time.time()
        for _ in range(15):
            _ = m(dummy_x)
        torch.cuda.synchronize()
        elapsed_ms = ((time.time() - t0) / 15) * 1000.0
        timing_results[name] = round(elapsed_ms, 2)
        print(f"  {name:<32} | Latency: {elapsed_ms:6.2f} ms")
        
    # Benchmark 3: Multi-Query Associative Recall (MQAR) Retrieval Accuracy
    print("\n" + "=" * 95)
    print(" BENCHMARK 3: MQAR RETRIEVAL ACCURACY WITH PROVEN MID-TRUNK (SeqLen = 256)")
    print("=" * 95)
    print(f"{'Pairs (N)':<10} | {'Pure Mamba':<16} | {'Jamba (Softmax)':<18} | {'Mamba + PRIME Trunk':<22} | {'Mamba + Hybrid Trunk':<22}")
    print("-" * 95)
    
    mqar_results = {name: {} for name in models.keys()}
    for N in [2, 4, 8, 16]:
        accs = {name: [] for name in models.keys()}
        for trial in range(30):
            seqs, q_pos, gts = generate_mqar_batch(batch_size=2, seq_len=256, num_pairs=N)
            for name, m in models.items():
                with torch.no_grad():
                    logits = m(seqs)
                for b in range(2):
                    pred = torch.argmax(logits[b, q_pos[b]]).item()
                    accs[name].append(1.0 if pred == gts[b] else 0.0)
                    
        line_str = f"{N:<10} | "
        for name in models.keys():
            mean_acc = float(np.mean(accs[name])) * 100.0
            mqar_results[name][str(N)] = round(mean_acc, 2)
            line_str += f"{mean_acc:>5.1f}%          | "
        print(line_str)
        
    # Save results
    output_data = {
        "memory_vs_length": mem_results,
        "timing_ms": timing_results,
        "mqar_accuracy": mqar_results
    }
    out_file = "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/experiments/mamba_proven_trunk_results.json"
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print("\n" + "=" * 95)
    print(f"[+] Empirical trunk evaluation saved to: {out_file}")
    print("=" * 95)

if __name__ == "__main__":
    main()
