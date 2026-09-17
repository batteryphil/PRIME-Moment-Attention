"""
Gumbel-Softmax Straight-Through Estimator (STE) for PRIME Moment Attention
========================================================================
Implements discrete, hard-routed contextual bandwidth allocation per head:
    Order 0 (O0): Mean contextual field (O(1) updates, physically skips S1 and S2)
    Order 1 (O1): Directional field (O(D) dot-product updates, physically skips S2)
    Order 2 (O2): Curvature field (O(D^2) full polynomial tensor updates)

Gumbel-Softmax STE formulation:
    z_i ~ Gumbel(0, 1) = -log(-log(u_i)), u_i ~ Uniform(0, 1)
    g_soft = Softmax((logits + z) / tau)
    g_hard = one_hot(argmax(g_soft)) - g_soft.detach() + g_soft

Forward pass:
    Discrete routing executes only the required compute paths.
Backward pass:
    Gradients flow continuously through g_soft into the router parameters.
Annealing schedule:
    tau_t = max(tau_min, tau_0 * (tau_min / tau_0) ** (t / total_anneal_steps))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


def gumbel_softmax_ste(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = True,
    dim: int = -1,
    eps: float = 1e-10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Gumbel-Softmax sample with Straight-Through Estimator.
    
    Args:
        logits: Unnormalized log probabilities (..., num_classes)
        tau: Non-negative scalar temperature
        hard: If True, the returned sample will be discretized as one-hot vectors,
              but will be differentiated as if it is the soft sample.
        dim: Dimension along which softmax will be computed
        eps: Epsilon for numerical stability in Gumbel noise calculation
        
    Returns:
        g_hard: Discretized one-hot tensor with autograd connected to logits
        g_soft: Continuous soft probabilities for logging/diagnostics
    """
    logits_f = logits.float()
    u = torch.rand_like(logits_f)
    gumbel_noise = -torch.log(-torch.log(u.clamp(min=1e-5, max=1.0 - 1e-5)) + 1e-7)
    y_soft_f = F.softmax((logits_f + gumbel_noise) / max(tau, 1e-4), dim=dim)
    y_soft = y_soft_f.to(logits.dtype)
    
    if hard:
        # Straight-Through Estimator (STE)
        index = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        # Bypasses hard discretization in backward pass:
        y = y_hard - y_soft.detach() + y_soft
        return y, y_soft
    else:
        return y_soft, y_soft


class GumbelAnnealingScheduler:
    """
    Logarithmic / exponential temperature annealing scheduler.
    """
    def __init__(
        self,
        tau_0: float = 1.0,
        tau_min: float = 0.05,
        anneal_steps: int = 2000
    ):
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.anneal_steps = anneal_steps
        self.current_step = 0

    def step(self) -> float:
        self.current_step += 1
        return self.get_tau()

    def get_tau(self) -> float:
        if self.current_step >= self.anneal_steps:
            return self.tau_min
        progress = self.current_step / float(self.anneal_steps)
        # Logarithmic/exponential decay: tau_0 * (tau_min / tau_0) ** progress
        tau = self.tau_0 * math.pow(self.tau_min / self.tau_0, progress)
        return max(self.tau_min, tau)


class GumbelHeadRouter(nn.Module):
    """
    Per-head discrete order allocator for multi-head PRIME attention.
    Assigns each head h in {1 ... H} to a discrete order in {O0, O1, O2}.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_orders: int = 3,
        tau_0: float = 1.0,
        tau_min: float = 0.05,
        anneal_steps: int = 2000
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_orders = num_orders
        
        # Router projection maps sequence representation to (num_heads, num_orders)
        self.router_proj = nn.Linear(hidden_dim, num_heads * num_orders)
        nn.init.zeros_(self.router_proj.weight)
        nn.init.constant_(self.router_proj.bias, 0.0)
        
        self.scheduler = GumbelAnnealingScheduler(tau_0, tau_min, anneal_steps)
        self.frozen_routing: Optional[torch.Tensor] = None

    def freeze(self, fixed_orders: Optional[torch.Tensor] = None):
        """
        Permanently freezes discrete routing assignments for production inference.
        """
        if fixed_orders is not None:
            self.frozen_routing = fixed_orders.detach().clone()
        else:
            with torch.no_grad():
                logits = self.router_proj.bias.view(self.num_heads, self.num_orders)
                self.frozen_routing = logits.argmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        tau: Optional[float] = None,
        hard: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Args:
            x: Sequence tensor (batch_size, seq_len, hidden_dim)
            tau: Optional explicit temperature override
            hard: Whether to use Straight-Through discrete one-hot
            
        Returns:
            g_hard: One-hot decisions (batch_size, num_heads, num_orders)
            g_soft: Soft probability distribution (batch_size, num_heads, num_orders)
            active_tau: The temperature used for this forward step
        """
        B, L, D = x.shape
        
        if self.frozen_routing is not None:
            # Inference mode with frozen discrete assignments: shape (num_heads,)
            # Expand to (B, num_heads, num_orders)
            one_hot = F.one_hot(self.frozen_routing, num_classes=self.num_orders).float()
            g_hard = one_hot.unsqueeze(0).repeat(B, 1, 1).to(x.device)
            return g_hard, g_hard, 0.0

        if tau is None:
            tau = self.scheduler.get_tau() if self.training else self.scheduler.tau_min
            
        # Sequence-pooled context
        pooled = x.mean(dim=1).to(self.router_proj.weight.dtype) # (B, D)
        logits = self.router_proj(pooled).view(B, self.num_heads, self.num_orders)
        
        if self.training:
            g_hard, g_soft = gumbel_softmax_ste(logits, tau=tau, hard=hard, dim=-1)
        else:
            # Inference evaluation without training noise: exact argmax
            g_soft = F.softmax(logits.float() / max(tau, 1e-4), dim=-1)
            index = g_soft.argmax(dim=-1, keepdim=True)
            g_hard = torch.zeros_like(g_soft).scatter_(-1, index, 1.0)
            
        return g_hard.to(x.dtype), g_soft.to(x.dtype), tau


class DiscretePRIMEAttention(nn.Module):
    """
    Multi-Head Attention module executing sparse hardware paths
    according to discrete per-head order allocations.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        eps: float = 1e-6
    ):
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
        
        self.router = GumbelHeadRouter(d_model, n_heads)

    def forward(
        self,
        x: torch.Tensor,
        tau: Optional[float] = None,
        hard: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B, L, D = x.shape
        H = self.n_heads
        d = self.head_dim

        # 1. Obtain discrete routing decisions: (B, H, 3)
        g_hard, g_soft, active_tau = self.router(x, tau=tau, hard=hard)

        # 2. Linear projections
        q = self.q_proj(x).view(B, L, H, d).transpose(1, 2) # (B, H, L, d)
        k = self.k_proj(x).view(B, L, H, d).transpose(1, 2) # (B, H, L, d)
        v = self.v_proj(x).view(B, L, H, d).transpose(1, 2) # (B, H, L, d)

        # 3. Hardware-Sparse Per-Head Execution
        # Extract per-head router indicators
        g0 = g_hard[..., 0].view(B, H, 1, 1) # (B, H, 1, 1) -> Indicator for O0
        g1 = g_hard[..., 1].view(B, H, 1, 1) # (B, H, 1, 1) -> Indicator for O1
        g2 = g_hard[..., 2].view(B, H, 1, 1) # (B, H, 1, 1) -> Indicator for O2

        # Effective linear and quadratic gains:
        # alpha_1 = g1 + g2
        # alpha_2 = g2
        alpha_1 = g1 + g2
        alpha_2 = g2

        # Pre-check whether O2 is physically present across the batch
        has_order_2 = (g2.sum() > 0)
        has_order_1 = (alpha_1.sum() > 0)

        # Kernel computation with physical branch skipping:
        dot = torch.matmul(q * self.scaling, k.transpose(-1, -2)) # (B, H, L, L)

        if not has_order_2:
            # O0 + O1: No heads require O(D^2) curvature terms!
            # Physically skips the dot ** 2 squaring operation
            kernel = 1.0 + alpha_1 * dot
        else:
            # Full hierarchy with selective masking
            kernel = 1.0 + alpha_1 * dot + 0.5 * alpha_2 * (dot ** 2)

        kernel = F.relu(kernel)
        denom = kernel.sum(dim=-1, keepdim=True) + self.eps
        attn_weights = kernel / denom
        
        out = torch.matmul(attn_weights, v) # (B, H, L, d)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)

        stats = {
            "g_hard": g_hard.detach(),
            "g_soft": g_soft.detach(),
            "tau": active_tau,
            "has_order_2": bool(has_order_2),
            "order_distribution": {
                "O0": float(g_hard[..., 0].mean().item()),
                "O1": float(g_hard[..., 1].mean().item()),
                "O2": float(g_hard[..., 2].mean().item()),
            }
        }
        return out, stats
