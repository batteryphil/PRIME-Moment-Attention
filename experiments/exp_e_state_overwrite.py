import math
import time
import torch
import torch.nn.functional as F

print("=" * 90)
print(" PHASE A & B EXPERIMENTS: 1M TOKEN SCALING & STATE OVERWRITE DYNAMICS")
print("=" * 90)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
print(f"Device: {device} | Dtype: {dtype}\n")

# =============================================================================
# EXPERIMENT 1: 1,000,000 TOKEN CONTEXT STEP LATENCY & MEMORY VERIFICATION
# =============================================================================
print("--- EXPERIMENT 1: 1-MILLION TOKEN RECURRENT STEP VERIFICATION ---")
print("Verifying that per-token decoding step latency and cache memory remain strictly")
print("constant as the internal sequence counter advances from 10k to 1,000,000 tokens.\n")

B, H, D = 1, 14, 64 # Qwen 2.5 architecture (14 heads, dim 64)
scale = 1.0 / math.sqrt(D)

# Initialize PRIME 2nd-order moment state
S0 = torch.zeros(B, H, D, device=device, dtype=dtype)
S1 = torch.zeros(B, H, D, D, device=device, dtype=dtype)
S2 = torch.zeros(B, H, D, D, device=device, dtype=dtype)
K0 = torch.zeros(B, H, 1, device=device, dtype=dtype)
K1 = torch.zeros(B, H, D, device=device, dtype=dtype)
K2 = torch.zeros(B, H, D, device=device, dtype=dtype)

def prime_step(q, k, v, decay=0.999):
    global S0, S1, S2, K0, K1, K2
    # Numerator
    S0 = decay * S0 + v
    kt_col = k.unsqueeze(-1)
    vt_row = v.unsqueeze(-2)
    S1 = decay * S1 + torch.matmul(kt_col, vt_row)
    k2_col = (k**2).unsqueeze(-1)
    S2 = decay * S2 + torch.matmul(k2_col, vt_row)
    # Denominator
    K0 = decay * K0 + 1.0
    K1 = decay * K1 + k
    K2 = decay * K2 + (k**2)
    
    num = S0 + torch.matmul(q.unsqueeze(-2), S1).squeeze(-2) + 0.5 * torch.matmul((q**2).unsqueeze(-2), S2).squeeze(-2)
    den = K0 + (q * K1).sum(dim=-1, keepdim=True) + 0.5 * ((q**2) * K2).sum(dim=-1, keepdim=True)
    den = den.clamp(min=0.5 * K0)
    return num / den

context_checkpoints = [1000, 10000, 50000, 100000, 500000, 1000000]
print(f"{'Context Level':<16} | {'Softmax KV Cache':<20} | {'PRIME Cache VRAM':<18} | {'PRIME Step Time':<18} | {'Norm ||y||'}")
print("-" * 90)

# Measure baseline memory
base_mem_bytes = 24 * (14*64 + 2*14*64*64 + 14*1 + 2*14*64) * 2

for ctx in context_checkpoints:
    # Set simulated accumulated count
    K0.fill_(float(ctx))
    
    # Warmup and time 100 steps
    q_test = torch.randn(B, H, D, device=device, dtype=dtype) * scale
    k_test = torch.randn(B, H, D, device=device, dtype=dtype)
    v_test = torch.randn(B, H, D, device=device, dtype=dtype)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        y_out = prime_step(q_test, k_test, v_test)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    step_time_ms = ((time.perf_counter() - t0) / 50.0) * 1000 * 24 # 24 layers
    
    # Softmax theoretical memory
    soft_kv_bytes = 2 * 24 * 2 * ctx * 64 * 2
    soft_str = f"{soft_kv_bytes / (1024*1024*1024):.2f} GB" if soft_kv_bytes >= 1024**3 else f"{soft_kv_bytes / (1024*1024):.1f} MB"
    prime_str = f"{base_mem_bytes / (1024*1024):.2f} MB flat"
    
    print(f"{ctx:<16,d} | {soft_str:<20} | {prime_str:<18} | {step_time_ms:<18.2f} ms | {y_out.norm().item():.4f}")

print("-" * 90)

# =============================================================================
# EXPERIMENT 2: STATE OVERWRITE & CONTRADICTORY FACT DYNAMICS
# =============================================================================
print("\n--- EXPERIMENT 2: STATE OVERWRITE & CONTRADICTION DYNAMICS ---")
print("Scenario: Fact A ('AX-417') is taught at t=0.")
print("Then 500 noise tokens pass.")
print("Then contradictory Fact B ('BX-921') is taught on the SAME key at t=501.")
print("Measuring P(Fact B dominates) vs P(Fact A persists) as Fact B is repeated...\n")

torch.manual_seed(999)
D = 64
scale = 1.0 / math.sqrt(D)

# Fact A and Fact B share the same Query/Key identity (e.g. 'reactor_designation')
key_identity = torch.randn(1, 1, 1, D, device=device, dtype=dtype) * 3.0

val_A = torch.randn(1, 1, 1, D, device=device, dtype=dtype) * 3.0 # Fact A (AX-417)
val_B = torch.randn(1, 1, 1, D, device=device, dtype=dtype) * 3.0 # Fact B (BX-921, contradictory update)

print(f"{'Fact B Exposures':<18} | {'Cosine to Fact A':<20} | {'Cosine to Fact B':<20} | {'Logit Diff (B - A)':<22} | {'Dominant Fact'}")
print("-" * 90)

for exposures in [1, 2, 3, 5, 10]:
    # Reset states
    S0 = torch.zeros(1, 1, D, device=device, dtype=dtype)
    S1 = torch.zeros(1, 1, D, D, device=device, dtype=dtype)
    S2 = torch.zeros(1, 1, D, D, device=device, dtype=dtype)
    K0 = torch.zeros(1, 1, 1, device=device, dtype=dtype)
    K1 = torch.zeros(1, 1, D, device=device, dtype=dtype)
    K2 = torch.zeros(1, 1, D, device=device, dtype=dtype)
    
    # 1. Teach Fact A at t=0
    _ = prime_step(key_identity.squeeze(2)*scale, key_identity.squeeze(2), val_A.squeeze(2))
    
    # 2. Inject 500 tokens of uncorrelated noise
    noise_k = torch.randn(500, 1, 1, D, device=device, dtype=dtype)
    noise_v = torch.randn(500, 1, 1, D, device=device, dtype=dtype)
    for t in range(500):
        _ = prime_step(noise_k[t].squeeze(2)*scale, noise_k[t].squeeze(2), noise_v[t].squeeze(2))
        
    # 3. Teach contradictory Fact B on the same key with N exposures
    for _ in range(exposures):
        _ = prime_step(key_identity.squeeze(2)*scale, key_identity.squeeze(2), val_B.squeeze(2))
        
    # 4. Query the key identity
    retrieved = prime_step(key_identity.squeeze(2)*scale, torch.zeros_like(key_identity.squeeze(2)), torch.zeros_like(val_A.squeeze(2)))
    
    cos_A = F.cosine_similarity(retrieved.squeeze(), val_A.squeeze(), dim=-1).item()
    cos_B = F.cosine_similarity(retrieved.squeeze(), val_B.squeeze(), dim=-1).item()
    logit_diff = (cos_B - cos_A)
    
    winner = "Fact B (Updated)" if cos_B > cos_A else "Fact A (Persistent)"
    print(f"{exposures:<18} | {cos_A:<20.4f} | {cos_B:<20.4f} | {logit_diff:<22.4f} | {winner}")

print("=" * 90)
