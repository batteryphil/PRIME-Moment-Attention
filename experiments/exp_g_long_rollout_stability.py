import math
import time
import torch
import numpy as np

print("=" * 85)
print(" STRESS-TEST 1: ULTRA-LONG ROLLOUT NUMERICAL STABILITY (2,048 STEPS)")
print(" Testing whether PRIME 2nd-Order Moment State explodes, vanishes, or drifts")
print("=" * 85)

torch.manual_seed(42)
B, H, D = 1, 8, 64
decay = 0.999
scale = 1.0 / math.sqrt(D)

# Initialize constant O(1) states
S0 = torch.zeros(B, H, D)
S1 = torch.zeros(B, H, D, D)
S2 = torch.zeros(B, H, D, D)
K0 = torch.zeros(B, H, 1)
K1 = torch.zeros(B, H, D)
K2 = torch.zeros(B, H, D)

T_total = 2048
norms_y = []
norms_S0 = []
norms_S1 = []
norms_S2 = []
dens = []

nan_found = False
inf_found = False

print(f"{'Step (t)':<12} | {'||y_t|| Norm':<16} | {'||S0|| Norm':<16} | {'||S1||_F Norm':<16} | {'Denominator':<14} | {'Status'}")
print("-" * 85)

checkpoints = [1, 64, 128, 256, 512, 1024, 1536, 2048]

t_start = time.time()
for t in range(1, T_total + 1):
    # Simulate realistic query, key, value vectors from standard normal
    q = torch.randn(B, H, D) * scale
    k = torch.randn(B, H, D)
    v = torch.randn(B, H, D)
    
    # Update Numerator states
    S0 = decay * S0 + v
    S1 = decay * S1 + torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))
    S2 = decay * S2 + torch.matmul((k**2).unsqueeze(-1), v.unsqueeze(-2))
    
    # Update Denominator states
    K0 = decay * K0 + 1.0
    K1 = decay * K1 + k
    K2 = decay * K2 + (k**2)
    
    # Readout
    num = S0 + torch.matmul(q.unsqueeze(-2), S1).squeeze(-2) + 0.5 * torch.matmul((q**2).unsqueeze(-2), S2).squeeze(-2)
    den = K0 + (q * K1).sum(dim=-1, keepdim=True) + 0.5 * ((q**2) * K2).sum(dim=-1, keepdim=True)
    den = den.clamp(min=1e-3)
    
    y = num / den
    
    # Record metrics
    y_norm = float(y.norm().item())
    s0_norm = float(S0.norm().item())
    s1_norm = float(S1.norm().item())
    s2_norm = float(S2.norm().item())
    den_val = float(den.mean().item())
    
    norms_y.append(y_norm)
    norms_S0.append(s0_norm)
    norms_S1.append(s1_norm)
    norms_S2.append(s2_norm)
    dens.append(den_val)
    
    if math.isnan(y_norm) or math.isnan(den_val):
        nan_found = True
        break
    if math.isinf(y_norm) or math.isinf(den_val):
        inf_found = True
        break
        
    if t in checkpoints:
        status = "STABLE" if (0.1 < y_norm < 10.0) else "DRIFT"
        print(f"{t:<12} | {y_norm:<16.4f} | {s0_norm:<16.2f} | {s1_norm:<16.2f} | {den_val:<14.2f} | {status}")

elapsed = time.time() - t_start
print("-" * 85)
print(f"Total Rollout Time for 2,048 steps: {elapsed*1000:.2f} ms ({T_total / elapsed:.1f} steps/sec)")
print(f"Any NaNs detected: {nan_found}")
print(f"Any Infs detected: {inf_found}")
print(f"Final Output Norm at t=2048: {norms_y[-1]:.4f} (Mean across all 2048 steps: {np.mean(norms_y):.4f})")
print(f"Min Norm: {np.min(norms_y):.4f} | Max Norm: {np.max(norms_y):.4f}")
print("=" * 85)
