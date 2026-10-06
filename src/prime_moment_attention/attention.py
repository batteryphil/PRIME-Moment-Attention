"""
PRIME Moment Attention: Constant-State Second-Order Recurrent Attention
========================================================================

Formulation:
    Given queries q_t, keys k_j, values v_j in R^D:
    Computes 2nd-order Taylor polynomial moment approximation to causal attention:
        exp( (q_t^T k_j) / sqrt(d) ) ~ 1 + s_tj + 0.5 * s_tj^2
    
    Numerator:
        N(q_t) = S_0 + (1 / sqrt(d)) * q_t^T S_1 + (1 / (2d)) * (q_t^2)^T S_2
    Denominator:
        D(q_t) = K_0 + (1 / sqrt(d)) * q_t^T K_1 + (1 / (2d)) * (q_t^2)^T K_2
        
    Output:
        y_t = N(q_t) / max(D(q_t), eps)

State Complexity:
    - S_0: [B, H, D]       (0th value moment: sum v_j)
    - S_1: [B, H, D, D]    (1st covariance tensor: sum k_j v_j^T)
    - S_2: [B, H, D, D]    (2nd diagonal curvature moment: sum (k_j^2) v_j^T)
    - K_0: [B, H, 1]       (0th key count: sum 1)
    - K_1: [B, H, D]       (1st key sum: sum k_j)
    - K_2: [B, H, D]       (2nd key square sum: sum k_j^2)

Memory Complexity:
    - O(1) with respect to context length L.
    - O(D^2) with respect to head dimension D.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .chunked_ssd import chunked_prime_ssd_core
except (ImportError, ValueError):
    from chunked_ssd import chunked_prime_ssd_core

class PrimeMomentAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        decay: float = 0.9995,
        use_qk_norm: bool = False,
        eps: float = 1.0,
        chunk_size: int = 64,
        chunk_threshold: int = 512,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.decay = decay
        self.scaling = 1.0 / math.sqrt(head_dim)
        self.eps = eps
        self.use_qk_norm = use_qk_norm
        self.chunk_size = chunk_size
        self.chunk_threshold = chunk_threshold

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(head_dim)
            self.k_norm = nn.LayerNorm(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        B, L, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype

        # Project Q, K, V
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply GQA repeat if needed
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Scale queries
        q = q * self.scaling

        # Cast to float32 for recurrence to guarantee zero overflow beyond float16 65,504 limit
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        # Initialize or unpack recurrent states
        if state is not None:
            S0, S1, S2, K0, K1, K2 = state
        else:
            S0 = torch.zeros(B, self.num_heads, self.head_dim, device=hidden_states.device, dtype=torch.float32)
            S1 = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=hidden_states.device, dtype=torch.float32)
            S2 = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=hidden_states.device, dtype=torch.float32)
            K0 = torch.zeros(B, self.num_heads, 1, device=hidden_states.device, dtype=torch.float32)
            K1 = torch.zeros(B, self.num_heads, self.head_dim, device=hidden_states.device, dtype=torch.float32)
            K2 = torch.zeros(B, self.num_heads, self.head_dim, device=hidden_states.device, dtype=torch.float32)

        # Autoregressive Decoding Mode (L == 1)
        if L == 1:
            qt = q_f32[:, :, 0]
            kt = k_f32[:, :, 0]
            vt = v_f32[:, :, 0]

            # Positive feature map
            qt_pos = F.elu(qt) + 1.0
            kt_pos = F.elu(kt) + 1.0

            # 2nd Order Taylor approximation updates
            S0 = self.decay * S0 + vt
            S1 = self.decay * S1 + torch.einsum('bhd,bhe->bhde', kt_pos, vt)
            S2 = self.decay * S2 + torch.einsum('bhd,bhe->bhde', kt_pos**2, vt)
            
            K0 = self.decay * K0 + 1.0
            K1 = self.decay * K1 + kt_pos
            K2 = self.decay * K2 + (kt_pos**2)

            num = S0 + torch.einsum('bhd,bhde->bhe', qt_pos, S1) + 0.5 * torch.einsum('bhd,bhde->bhe', qt_pos**2, S2)
            den = (K0 + torch.sum(qt_pos * K1, dim=-1, keepdim=True) + 0.5 * torch.sum((qt_pos**2) * K2, dim=-1, keepdim=True)).clamp(min=self.eps)

            y = (num / den).unsqueeze(2) # [B, H, 1, D]

        # Chunked Parallel SSD Mode (L > chunk_threshold): Linear O(L) Scaling with 3D Batch GEMM Scan
        elif L > self.chunk_threshold:
            y, next_state_chunk = chunked_prime_ssd_core(
                q=q_f32,
                k=k_f32,
                v=v_f32,
                state=state,
                decay=self.decay,
                eps=self.eps,
                chunk_size=self.chunk_size,
                return_state=return_state,
            )
            if return_state:
                S0, S1, S2, K0, K1, K2 = next_state_chunk

        # Prefill Mode (1 < L <= chunk_threshold): Fast Vectorized Dense 2nd-order Taylor Attention
        else:
            idx = torch.arange(L, device=hidden_states.device)
            diff = idx.unsqueeze(1) - idx.unsqueeze(0)
            causal_mask = diff >= 0
            decay_mat = torch.where(
                causal_mask,
                torch.pow(self.decay, diff.float()),
                torch.zeros(L, L, device=hidden_states.device)
            ).view(1, 1, L, L)

            # Positive feature map replacing Taylor expansion
            q_pos = F.elu(q_f32) + 1.0
            k_pos = F.elu(k_f32) + 1.0
            A = decay_mat * torch.matmul(q_pos, k_pos.transpose(-1, -2))

            num_dense = torch.matmul(A, v_f32)
            den_dense = torch.sum(A, dim=-1, keepdim=True).clamp(min=self.eps)

            if state is not None:
                decay_into = torch.pow(self.decay, (idx + 1).float()).view(1, 1, L, 1)
                carry_num = decay_into * torch.matmul(q_pos, S1)
                carry_den = decay_into * torch.sum(q_pos * K1.unsqueeze(2), dim=-1, keepdim=True)
                num_dense = num_dense + carry_num
                den_dense = (den_dense + carry_den).clamp(min=self.eps)

            y = num_dense / den_dense

            # If return_state is requested, compute terminal state in one matrix multiply
            if return_state:
                weights = torch.pow(self.decay, (L - 1 - idx).float()).view(1, 1, L, 1)
                k_w = k_pos * weights
                v_w = v_f32 * weights
                
                decay_total = math.pow(self.decay, L)
                S1 = (decay_total * S1 if state is not None else 0.0) + torch.matmul(k_w.transpose(-1, -2), v_f32)
                K1 = (decay_total * K1 if state is not None else 0.0) + torch.sum(k_w, dim=2)
                # S0, S2, K0, K2 remain 0

        # Reshape to output projection shape
        y = y.to(orig_dtype)
        y = y.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)
        output = self.o_proj(y)

        next_state = (S0, S1, S2, K0, K1, K2) if return_state else None
        return output, next_state
