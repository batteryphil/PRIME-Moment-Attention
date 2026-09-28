"""
PRIME Hydrodynamic Information Dynamics: Bernoulli Entrainment & Venturi Memory Gate
=====================================================================================
Applies Bernoulli's principle of fluid dynamics to associative neural memory:
Along an informational streamline, an increase in information velocity (surprise/gradient)
creates a local static pressure drop:
    P(k) = P_0 - 0.5 * rho * v_t^2

This informational vacuum (Venturi effect) dynamically collapses the activation barrier
for semantically adjacent memories, pulling/entraining correlated background knowledge
into active working attention without requiring an exhaustive brute-force search.

Features:
1. Information Velocity Metric: v_t = ||k_t - mu_context||_2 * (1.0 + Delta_t)
2. Bernoulli Dynamic Pressure Drop: Delta P_t = 0.5 * rho * v_t^2
3. Venturi Entrainment Drag Operator:
   kappa_i = ReLU(cos_sim(k_t, m_i)) * (1.0 - exp(-Delta P_t / tau_p))
4. Hydrodynamic Value Augmentation:
   v_t_augmented = v_t + W_drag @ sum(kappa_i * m_value_i)
"""

import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class VenturiInformationGate(nn.Module):
    """
    Implements Bernoulli Informational Entrainment:
    High-velocity injected tokens create a local pressure drop that 'drags'
    correlated latent memory vectors into active working state.
    """
    def __init__(
        self,
        hidden_size: int,
        key_dim: int,
        memory_slots: int = 128,
        rho_density: float = 1.25,
        tau_pressure: float = 2.0,
        drag_scale: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.key_dim = key_dim
        self.memory_slots = memory_slots
        self.rho = nn.Parameter(torch.tensor(rho_density, dtype=torch.float32))
        self.tau_p = nn.Parameter(torch.tensor(tau_pressure, dtype=torch.float32))
        
        # Projection to modulate drag force
        self.drag_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.drag_scale = drag_scale
        
        # Latent associative memory bank (e.g. background knowledge / anchor facts)
        self.memory_keys = nn.Parameter(torch.randn(memory_slots, key_dim) / math.sqrt(key_dim))
        self.memory_values = nn.Parameter(torch.randn(memory_slots, hidden_size) / math.sqrt(hidden_size))
        
        # Running context key expectation
        self.register_buffer("running_mu", torch.zeros(key_dim))
        self.register_buffer("steps_seen", torch.tensor(0, dtype=torch.long))

    def compute_information_velocity(self, k: torch.Tensor, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes information velocity:
        v_t = ||k_t - mu_context||_2 * (1.0 + delta_t)
        """
        diff = k - self.running_mu.view(1, 1, -1)
        velocity = torch.norm(diff, p=2, dim=-1)  # [B, L]
        if delta is not None:
            if delta.dim() == 2:
                velocity = velocity * (1.0 + delta)
            elif delta.dim() == 3:
                velocity = velocity * (1.0 + delta.mean(dim=-1))
        return velocity

    def compute_bernoulli_vacuum(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Calculates dynamic pressure drop:
        Delta P = 0.5 * rho * v^2
        """
        rho_clamped = F.softplus(self.rho)
        delta_P = 0.5 * rho_clamped * (velocity ** 2)
        return delta_P

    def forward(
        self,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        delta: Optional[torch.Tensor] = None,
        custom_memory: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Applies Venturi Entrainment:
        hidden_states: [B, L, H]
        keys: [B, L, D_k]
        values: [B, L, H]
        """
        B, L, H = hidden_states.shape
        mem_k, mem_v = custom_memory if custom_memory is not None else (self.memory_keys, self.memory_values)
        M, D_k = mem_k.shape

        # 1. Calculate Information Velocity
        velocity = self.compute_information_velocity(keys, delta)  # [B, L]

        # Update running context mean
        if self.training:
            with torch.no_grad():
                batch_mean = keys.mean(dim=(0, 1))
                decay = 0.95
                self.running_mu.mul_(decay).add_(batch_mean * (1.0 - decay))
                self.steps_seen.add_(1)

        # 2. Calculate Bernoulli Dynamic Pressure Drop (Vacuum)
        delta_P = self.compute_bernoulli_vacuum(velocity)  # [B, L]

        # 3. Compute Normalized Semantic Affinity with Memory Bank
        norm_k = F.normalize(keys, p=2, dim=-1)  # [B, L, D_k]
        norm_mem_k = F.normalize(mem_k, p=2, dim=-1)  # [M, D_k]
        affinity = torch.matmul(norm_k, norm_mem_k.transpose(0, 1))  # [B, L, M]

        # 4. Venturi Suction Coefficient: kappa = ReLU(affinity) * (1 - exp(-Delta_P / tau_p))
        tau_clamped = F.softplus(self.tau_p).clamp(min=1e-3)
        suction_factor = (1.0 - torch.exp(-delta_P.unsqueeze(-1) / tau_clamped)).clamp(min=0.0, max=1.0)  # [B, L, 1]
        
        entrainment_weights = F.relu(affinity) * suction_factor  # [B, L, M]

        # 5. Dragged Memory Values
        dragged_values = torch.matmul(entrainment_weights, mem_v)  # [B, L, H]
        drag_force = self.drag_scale * self.drag_proj(dragged_values)

        # 6. Augmented Working State
        augmented_values = values + drag_force

        diagnostics = {
            "mean_velocity": velocity.mean().item(),
            "mean_pressure_drop": delta_P.mean().item(),
            "mean_entrainment_suction": suction_factor.mean().item(),
            "max_entrained_weight": entrainment_weights.max().item(),
            "drag_norm": torch.norm(drag_force, p=2, dim=-1).mean().item()
        }

        return augmented_values, diagnostics
