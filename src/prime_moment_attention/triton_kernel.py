import torch
import triton
import triton.language as tl

@triton.jit
def prime_fwd_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, den_ptr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_kb, stride_kh, stride_kl, stride_kd,
    stride_vb, stride_vh, stride_vl, stride_vd,
    stride_ob, stride_oh, stride_ol, stride_od,
    stride_denb, stride_denh, stride_denl,
    L, D: tl.constexpr,
    decay,
    scaling
):
    # One program per batch and head
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    q_head_ptr = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_head_ptr = k_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_head_ptr = v_ptr + batch_idx * stride_vb + head_idx * stride_vh
    o_head_ptr = out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    den_head_ptr = den_ptr + batch_idx * stride_denb + head_idx * stride_denh
    
    d_offsets = tl.arange(0, D)
    d_offsets_row = tl.arange(0, D)[:, None]
    d_offsets_col = tl.arange(0, D)[None, :]
    
    # Initialize states in SRAM
    # S0: sum_{j} w * V_j -> shape (D,)
    s0 = tl.zeros([D], dtype=tl.float32)
    
    # S1: sum_{j} w * (K_j \otimes V_j) -> shape (D, D)
    s1 = tl.zeros([D, D], dtype=tl.float32)
    
    # S2: sum_{j} w * (K_j^2 \otimes V_j) -> shape (D, D)
    s2 = tl.zeros([D, D], dtype=tl.float32)
    
    # K0: sum_{j} w -> scalar
    k0 = 0.0
    
    # K1: sum_{j} w * K_j -> shape (D,)
    k1 = tl.zeros([D], dtype=tl.float32)
    
    # K2: sum_{j} w * K_j^2 -> shape (D,)
    k2 = tl.zeros([D], dtype=tl.float32)
    
    # Recurrent loop over sequence length L
    for i in range(L):
        # Load Q, K, V for timestep i
        q_ptrs = q_head_ptr + i * stride_ql + d_offsets * stride_qd
        k_ptrs = k_head_ptr + i * stride_kl + d_offsets * stride_kd
        v_ptrs = v_head_ptr + i * stride_vl + d_offsets * stride_vd
        
        q = tl.load(q_ptrs).to(tl.float32) * scaling
        k = tl.load(k_ptrs).to(tl.float32)
        v = tl.load(v_ptrs).to(tl.float32)
        
        # ELU + 1 Feature Map
        q_pos = tl.where(q > 0.0, q + 1.0, tl.exp(q))
        k_pos = tl.where(k > 0.0, k + 1.0, tl.exp(k))
        
        q_pos2 = q_pos * q_pos
        k_pos2 = k_pos * k_pos
        
        # Decay past states
        s0 = s0 * decay
        s1 = s1 * decay
        s2 = s2 * decay
        k0 = k0 * decay
        k1 = k1 * decay
        k2 = k2 * decay
        
        # Update states with current K and V
        s0 = s0 + v
        k0 = k0 + 1.0
        
        k1 = k1 + k_pos
        k2 = k2 + k_pos2
        
        # K \otimes V (Outer product)
        kv_outer = k_pos[:, None] * v[None, :]
        s1 = s1 + kv_outer
        
        k2v_outer = k_pos2[:, None] * v[None, :]
        s2 = s2 + k2v_outer
        
        # Compute Output Numerator and Denominator
        # num = s0 + q @ s1 + 0.5 * q^2 @ s2
        # To do vector-matrix mult in Triton: tl.sum(q[:, None] * s1, axis=0)
        num_1 = tl.sum(q_pos[:, None] * s1, axis=0)
        num_2 = tl.sum(q_pos2[:, None] * s2, axis=0)
        num = s0 + num_1 + 0.5 * num_2
        
        den_1 = tl.sum(q_pos * k1, axis=0)
        den_2 = tl.sum(q_pos2 * k2, axis=0)
        den = k0 + den_1 + 0.5 * den_2
        
        # Guard against zero division
        den = tl.where(den < 1e-3, 1e-3, den)
        
        out = num / den
        
        # Store
        o_ptrs = o_head_ptr + i * stride_ol + d_offsets * stride_od
        tl.store(o_ptrs, out)
        
        den_store_ptr = den_head_ptr + i * stride_denl
        tl.store(den_store_ptr, den)


def triton_prime_forward(q, k, v, decay, scaling):
    B, L, H, D = q.shape
    # Make contiguous and transpose to standard B, H, L, D
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    
    out = torch.empty_like(q)
    den = torch.empty((B, H, L), device=q.device, dtype=torch.float32)
    
    # Grid: (Batch, Head)
    grid = (B, H)
    
    prime_fwd_kernel[grid](
        q, k, v, out, den,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        den.stride(0), den.stride(1), den.stride(2),
        L, D,
        decay=decay,
        scaling=scaling
    )
    
    return out.transpose(1, 2).contiguous()
