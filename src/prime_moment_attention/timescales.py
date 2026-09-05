"""
Differentiable Multiscale Temporal Filter Bank
===============================================

Implements per-head learnable decay rates:
    lambda_h = sigmoid(theta_h)
Initialized uniformly in log-space across time-constants tau in [2, 1000] tokens.
Allows the attention architecture to learn a multiscale temporal basis.
"""

import math
from typing import Tuple
import torch
import torch.nn as nn

class LearnableTimescales(nn.Module):
    def __init__(self, num_heads: int, min_tau: float = 2.0, max_tau: float = 1000.0):
        super().__init__()
        self.num_heads = num_heads
        
        # Initialize uniformly in log-space
        taus = torch.logspace(math.log10(min_tau), math.log10(max_tau), steps=num_heads)
        lambdas_init = 1.0 - (1.0 / taus)
        # Invert sigmoid: theta = log(lambda / (1 - lambda))
        thetas_init = torch.log(lambdas_init / (1.0 - lambdas_init))
        self.theta = nn.Parameter(thetas_init)

    def forward(self) -> torch.Tensor:
        """Returns decay rates lambda_h in (0, 1) of shape [num_heads]"""
        return torch.sigmoid(self.theta)

    def get_time_constants(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (tau, half_lives) in tokens"""
        lambdas = self.forward()
        tau = 1.0 / (1.0 - lambdas)
        half_life = math.log(0.5) / torch.log(lambdas)
        return tau, half_life
