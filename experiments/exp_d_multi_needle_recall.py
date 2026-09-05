import math
import torch
import torch.nn.functional as F

print("=" * 85)
print(" STRESS-TEST 2: MULTI-NEEDLE ASSOCIATIVE RECALL (Context L = 2,048)")
print(" Testing 3 distinct needles stored at t=250, t=1000, t=1750 in 2048 noise tokens")
print("=" * 85)

torch.manual_seed(42)
B, H, L, D = 1, 1, 2048, 64
scale = 1.0 / math.sqrt(D)

# 1. Generate background noise tokens
k = torch.randn(B, H, L, D)
v = torch.randn(B, H, L, D)

# 2. Insert 3 distinct needles
# Needle 1: early context (t=250)
needle1_pos = 250
k_A = torch.randn(1, 1, 1, D) * 3.0
v_A = torch.randn(1, 1, 1, D) * 3.0
k[:, :, needle1_pos:needle1_pos+1] = k_A
v[:, :, needle1_pos:needle1_pos+1] = v_A

# Needle 2: mid context (t=1000)
needle2_pos = 1000
k_B = torch.randn(1, 1, 1, D) * 3.0
v_B = torch.randn(1, 1, 1, D) * 3.0
k[:, :, needle2_pos:needle2_pos+1] = k_B
v[:, :, needle2_pos:needle2_pos+1] = v_B

# Needle 3: deep context (t=1750)
needle3_pos = 1750
k_C = torch.randn(1, 1, 1, D) * 3.0
v_C = torch.randn(1, 1, 1, D) * 3.0
k[:, :, needle3_pos:needle3_pos+1] = k_C
v[:, :, needle3_pos:needle3_pos+1] = v_C

needles = [
    ("Needle A (t=250)", k_A, v_A),
    ("Needle B (t=1000)", k_B, v_B),
    ("Needle C (t=1750)", k_C, v_C),
]

# -----------------------------------------------------------------------------
# Standard Softmax Attention Evaluation
# -----------------------------------------------------------------------------
def eval_softmax(query_target):
    q = torch.randn(B, H, L, D)
    q[:, :, -1:] = query_target
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool))
    scores = torch.where(mask, scores, -torch.inf)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)[:, :, -1]
    return out

# -----------------------------------------------------------------------------
# Standard Linear Attention (ELU+1) Evaluation
# -----------------------------------------------------------------------------
def eval_linear(query_target, decay=0.999):
    q_f = F.elu(query_target * scale) + 1.0
    k_f = F.elu(k) + 1.0
    
    S = torch.zeros(B, H, D, D)
    z = torch.zeros(B, H, D)
    
    for t in range(L):
        kt = k_f[:, :, t].unsqueeze(-1)
        vt = v[:, :, t].unsqueeze(-2)
        S = decay * S + torch.matmul(kt, vt)
        z = decay * z + k_f[:, :, t]
        
    qt = q_f.squeeze(2).unsqueeze(-2)
    num = torch.matmul(qt, S).squeeze(-2)
    den = (q_f.squeeze(2) * z).sum(dim=-1, keepdim=True).clamp(min=1e-5)
    return num / den

# -----------------------------------------------------------------------------
# PRIME 2nd-Order Moment Attention Evaluation
# -----------------------------------------------------------------------------
def eval_prime(query_target, decay=0.999):
    qt = (query_target * scale).squeeze(2)
    
    S0 = torch.zeros(B, H, D)
    S1 = torch.zeros(B, H, D, D)
    S2 = torch.zeros(B, H, D, D)
    K0 = torch.zeros(B, H, 1)
    K1 = torch.zeros(B, H, D)
    K2 = torch.zeros(B, H, D)
    
    for t in range(L):
        kt = k[:, :, t]
        vt = v[:, :, t]
        
        S0 = decay * S0 + vt
        S1 = decay * S1 + torch.matmul(kt.unsqueeze(-1), vt.unsqueeze(-2))
        S2 = decay * S2 + torch.matmul((kt**2).unsqueeze(-1), vt.unsqueeze(-2))
        
        K0 = decay * K0 + 1.0
        K1 = decay * K1 + kt
        K2 = decay * K2 + (kt**2)
        
    num = S0 + torch.matmul(qt.unsqueeze(-2), S1).squeeze(-2) + 0.5 * torch.matmul((qt**2).unsqueeze(-2), S2).squeeze(-2)
    den = K0 + (qt * K1).sum(dim=-1, keepdim=True) + 0.5 * ((qt**2) * K2).sum(dim=-1, keepdim=True)
    den = den.clamp(min=0.5 * K0)
    return num / den

print(f"{'Target Query':<20} | {'Model':<24} | {'Retrieved Norm':<16} | {'Cosine Similarity'}")
print("-" * 85)

for name, k_target, v_target in needles:
    v_true = v_target.squeeze(0).squeeze(1)
    
    out_soft = eval_softmax(k_target)
    cos_soft = F.cosine_similarity(out_soft, v_true, dim=-1).item()
    norm_soft = out_soft.norm().item()
    print(f"{name:<20} | {'Softmax (O(L^2))':<24} | {norm_soft:<16.2f} | {cos_soft:.4f}")
    
    out_lin = eval_linear(k_target)
    cos_lin = F.cosine_similarity(out_lin, v_true, dim=-1).item()
    norm_lin = out_lin.norm().item()
    print(f"{'':<20} | {'Linear (ELU+1)':<24} | {norm_lin:<16.2f} | {cos_lin:.4f}")
    
    out_prm = eval_prime(k_target)
    cos_prm = F.cosine_similarity(out_prm, v_true, dim=-1).item()
    norm_prm = out_prm.norm().item()
    print(f"{'':<20} | {'PRIME Moment Attn':<24} | {norm_prm:<16.2f} | {cos_prm:.4f}")
    print("-" * 85)

print("=" * 85)
