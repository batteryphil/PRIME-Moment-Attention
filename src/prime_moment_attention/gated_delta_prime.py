#!/usr/bin/env python3
"""
Gated Delta-PRIME: Second-Order Decoupled Delta Attention with Lyapunov Bounding
================================================================================
Combines:
  1. Second-Order Taylor Moment Expansion (PRIME)
  2. Decoupled Channel-Wise Erasing & Writing Gates (Gated DeltaNet-2, May 2026)
  3. L2-Normalized Delta Rule Error Correction (Guaranteed positive feedback stability)
  4. Exponential-Trapezoidal Time Discretization (Mamba-3)
  5. Dual-State Lyapunov Contractive Projection (ACI-inspired stability)

Memory Complexity: O(1) in sequence length, O(D^2) in head dimension.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDeltaPrimeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        base_decay: float = 0.9995,
        use_qk_norm: bool = True,
        use_trapezoidal: bool = True,
        lyapunov_bound: float = 25.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.base_decay = base_decay
        self.scaling = 1.0 / math.sqrt(head_dim)
        self.use_qk_norm = use_qk_norm
        self.use_trapezoidal = use_trapezoidal
        self.lyapunov_bound = lyapunov_bound
        self.eps = eps

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Gated DeltaNet-2 Decoupled Gates (Channel-wise Erase & Write)
        self.erase_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        self.write_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)

        # Learnable log-timescales for multiscale decay bank
        self.log_tau = nn.Parameter(
            torch.linspace(math.log(4.0), math.log(1000.0), num_heads).view(1, num_heads, 1)
        )

        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(head_dim)
            self.k_norm = nn.LayerNorm(head_dim)

        self._init_weights()

    def _init_weights(self):
        # Erase gate: start small (b ~ 0.05) so retention is high initially
        nn.init.constant_(self.erase_proj.bias, -2.5)
        # Write gate: start moderate (w ~ 0.25)
        nn.init.constant_(self.write_proj.bias, -1.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        B, L, _ = hidden_states.shape
        device = hidden_states.device

        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        b_gate = torch.sigmoid(self.erase_proj(hidden_states)).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        w_gate = torch.sigmoid(self.write_proj(hidden_states)).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q * self.scaling

        # Stable Normalized Keys for Delta Rule (prevents exponential feedback blowup)
        k_normed = F.normalize(k, p=2, dim=-1)
        k2_normed = F.normalize(k**2, p=2, dim=-1)

        gamma = torch.exp(-1.0 / torch.exp(self.log_tau)).to(torch.float32)

        q_f = q.to(torch.float32)
        k_f = k_normed.to(torch.float32)
        k2_f = k2_normed.to(torch.float32)
        v_f = v.to(torch.float32)
        b_f = b_gate.to(torch.float32)
        w_f = w_gate.to(torch.float32)

        if state is not None:
            S0, S1, S2, prev_delta_S1, prev_delta_S2 = state
        else:
            S0 = torch.zeros(B, self.num_heads, self.head_dim, device=device, dtype=torch.float32)
            S1 = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=torch.float32)
            S2 = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=torch.float32)
            prev_delta_S1 = torch.zeros_like(S1)
            prev_delta_S2 = torch.zeros_like(S2)

        y_steps = []
        for t in range(L):
            qt = q_f[:, :, t]
            kt = k_f[:, :, t]
            k2t = k2_f[:, :, t]
            vt = v_f[:, :, t]
            bt = b_f[:, :, t]
            wt = w_f[:, :, t]

            # 1. Delta Error Vector (Current retrieval deficit)
            v_hat_1 = torch.einsum('bhde,bhd->bhe', S1, kt)
            v_hat_2 = 0.5 * torch.einsum('bhde,bhd->bhe', S2, k2t)
            v_hat = S0 + v_hat_1 + v_hat_2
            error_t = vt - v_hat

            # 2. Decoupled Channel-Wise Erase: (1 - b_t)
            erase_factor = (1.0 - bt).unsqueeze(-1)
            gamma_b = gamma.unsqueeze(-1) * erase_factor

            # 3. Normalized Delta Updates
            delta_S0 = wt * error_t * 0.1
            delta_S1 = torch.einsum('bhd,bhe->bhde', kt, wt * error_t * 0.1)
            delta_S2 = torch.einsum('bhd,bhe->bhde', k2t, wt * error_t * 0.1)

            # 4. Exponential-Trapezoidal Discretization
            if self.use_trapezoidal:
                trapezoidal_update_1 = 0.5 * (delta_S1 + gamma_b * prev_delta_S1)
                trapezoidal_update_2 = 0.5 * (delta_S2 + gamma_b * prev_delta_S2)
                prev_delta_S1 = delta_S1
                prev_delta_S2 = delta_S2
            else:
                trapezoidal_update_1 = delta_S1
                trapezoidal_update_2 = delta_S2

            S0 = gamma * (1.0 - bt) * S0 + delta_S0
            S1 = gamma_b * S1 + trapezoidal_update_1
            S2 = gamma_b * S2 + trapezoidal_update_2

            # 5. Dual-State Lyapunov Contractive Projection (Guarantees non-divergence)
            if self.lyapunov_bound > 0.0:
                s1_norm = torch.linalg.norm(S1, dim=(-2, -1), keepdim=True)
                scale1 = torch.clamp(self.lyapunov_bound / (s1_norm + 1e-6), max=1.0)
                S1 = S1 * scale1

                s2_norm = torch.linalg.norm(S2, dim=(-2, -1), keepdim=True)
                scale2 = torch.clamp(self.lyapunov_bound / (s2_norm + 1e-6), max=1.0)
                S2 = S2 * scale2

            # 6. Readout Contraction
            term1 = torch.einsum('bhd,bhde->bhe', qt, S1)
            term2 = 0.5 * torch.einsum('bhd,bhde->bhe', qt**2, S2)
            yt = S0 + term1 + term2
            y_steps.append(yt)

        y = torch.stack(y_steps, dim=2)
        y = y.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)
        y = y.to(hidden_states.dtype)
        out = self.o_proj(y)

        next_state = (S0, S1, S2, prev_delta_S1, prev_delta_S2) if return_state else None
        return out, next_state
