"""
Adaptive PRIME: Learnable Contextual Bandwidth Allocator
======================================================
Formulates PRIME moment attention as a continuous hierarchy of contextual operators:
    Order 0 (O0): Global / Mean Contextual Field (low-pass regularizer)
    Order 1 (O1): Directional / Query-Conditioned Field (associative routing)
    Order 2 (O2): Curvature / Geometric Field (higher-order dynamics)

Continuously relaxed kernel:
    K_ij = 1 + alpha_1 * (q_i^T k_j) + (alpha_2 / 2) * (q_i^T k_j)^2
where:
    alpha_1 = g_1 + g_2
    alpha_2 = g_2
    [g_0, g_1, g_2] = Softmax(Router(x))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptivePrimeRouter(nn.Module):
    """
    Predicts contextual bandwidth weights [g_0, g_1, g_2] dynamically
    from sequence representation x.
    """
    def __init__(self, hidden_dim, num_orders=3):
        super().__init__()
        self.num_orders = num_orders
        self.gate_proj = nn.Linear(hidden_dim, num_orders)
        # Initialize bias with uniform distribution across orders
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 0.0)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
        Returns:
            g: (batch_size, num_orders) router weights summing to 1
            alpha_1: (batch_size, 1, 1, 1) linear bandwidth gain
            alpha_2: (batch_size, 1, 1, 1) quadratic curvature gain
        """
        # Pool across sequence length
        pooled = hidden_states.mean(dim=1) # (batch_size, hidden_dim)
        logits = self.gate_proj(pooled)    # (batch_size, 3)
        g = F.softmax(logits, dim=-1)      # (batch_size, 3)

        g0 = g[:, 0].view(-1, 1, 1, 1)
        g1 = g[:, 1].view(-1, 1, 1, 1)
        g2 = g[:, 2].view(-1, 1, 1, 1)

        alpha_1 = g1 + g2
        alpha_2 = g2
        return g, alpha_1, alpha_2


class EmpiricalBandwidthRouter(nn.Module):
    """
    Zero-shot empirical router that dynamically determines [g_0, g_1, g_2]
    based on the spectral curvature vs. noise ratio of the input representation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        """
        Estimates signal complexity from finite differences across sequence dimension:
            - Mean 1st difference ||h_t - h_{t-1}|| (velocity / frequency)
            - Mean 2nd difference ||h_t - 2 h_{t-1} + h_{t-2}|| (acceleration / curvature)
        """
        B, L, D = hidden_states.shape
        if L < 3:
            g = torch.tensor([[0.333, 0.333, 0.334]], device=hidden_states.device).repeat(B, 1)
            return g, g[:, 1:2].view(B, 1, 1, 1) + g[:, 2:3].view(B, 1, 1, 1), g[:, 2:3].view(B, 1, 1, 1)

        # First and second differences
        diff1 = (hidden_states[:, 1:] - hidden_states[:, :-1]).norm(dim=-1).mean(dim=-1) # (B,)
        diff2 = (hidden_states[:, 2:] - 2 * hidden_states[:, 1:-1] + hidden_states[:, :-2]).norm(dim=-1).mean(dim=-1) # (B,)
        base_norm = hidden_states.norm(dim=-1).mean(dim=-1) + 1e-5

        # Relative curvature ratio
        curv_ratio = (diff2 / base_norm).clamp(0.0, 5.0)

        # Gating heuristic:
        # High relative curvature -> high-frequency dynamics -> route to O2
        # Smooth flow -> route to O1
        # Flat / noisy -> route to O0
        logits_0 = 1.0 - curv_ratio
        logits_1 = 1.0 - torch.abs(curv_ratio - 1.0)
        logits_2 = curv_ratio - 0.5

        logits = torch.stack([logits_0, logits_1, logits_2], dim=-1)
        g = F.softmax(logits * 2.0, dim=-1)

        alpha_1 = (g[:, 1] + g[:, 2]).view(B, 1, 1, 1)
        alpha_2 = g[:, 2].view(B, 1, 1, 1)
        return g, alpha_1, alpha_2


class AdaptivePrimeAttention(nn.Module):
    """
    Full Adaptive PRIME Attention module with continuous order relaxation.
    """
    def __init__(self, d_model, n_heads, eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.eps = eps

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.router = AdaptivePrimeRouter(d_model)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        # Dynamic bandwidth allocation
        g, alpha_1, alpha_2 = self.router(x)

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Dot-product
        dot = torch.matmul(q * self.scaling, k.transpose(-1, -2)) # (B, H, L, L)

        # Continuous polynomial kernel
        kernel = 1.0 + alpha_1 * dot + 0.5 * alpha_2 * (dot ** 2)
        kernel = F.relu(kernel)

        if mask is not None:
            kernel = kernel * mask

        attn_weights = kernel / (kernel.sum(dim=-1, keepdim=True) + self.eps)
        out = torch.matmul(attn_weights, v) # (B, H, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), g, attn_weights
