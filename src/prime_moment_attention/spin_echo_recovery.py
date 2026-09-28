"""
PRIME Unitary Information Conservation & Adjoint Spin-Echo Memory Recovery
===========================================================================
Formulates the recovery of 'overwritten' memories from recurrent state manifolds.
By the No-Hiding Theorem and quantum unitarity (U^dagger U = I), information
cannot be destroyed; it is only attenuated and superposed across tensor coordinates.

Features:
1. Adjoint Time-Reversal Operator (Neural Spin Echo):
   S_{t-1} = (1 / lambda) * (S_t - k_t v_t^T)
2. Dual-Basis Moore-Penrose Spectral Deconvolution:
   Extracts exact past values v_j from S_t up to the state manifold rank capacity.
3. Numerical Landauer Event Horizon Calculator:
   Computes tau_crit = -precision_bits * ln(2) / ln(lambda).
"""

import math
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdjointSpinEchoMemoryRecoverer:
    """
    Enables backward unrolling and spectral deconvolution of historical
    memories superposed inside recurrent state matrices.
    """
    def __init__(self, key_dim: int, hidden_size: int, default_decay: float = 0.995):
        self.key_dim = key_dim
        self.hidden_size = hidden_size
        self.default_decay = default_decay

    @staticmethod
    def step_backward(
        state_t: torch.Tensor,
        key_t: torch.Tensor,
        value_t: torch.Tensor,
        decay: float = 0.995,
        is_second_order: bool = False,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """
        Reverses a single time step:
        S_{t-1} = (S_t - Delta_S_t) / decay
        """
        if is_second_order:
            k_sq = key_t ** 2
            delta_s = 0.5 * (beta ** 2) * torch.outer(k_sq, value_t)
        else:
            delta_s = beta * torch.outer(key_t, value_t)
            
        state_prev = (state_t - delta_s) / decay
        return state_prev

    @staticmethod
    def reconstruct_past_states(
        final_state: torch.Tensor,
        keys_sequence: List[torch.Tensor],
        values_sequence: List[torch.Tensor],
        decay: float = 0.995,
    ) -> List[torch.Tensor]:
        """
        Rolls backward from t = T to t = 0, reconstructing every historical state S_t.
        """
        curr_state = final_state.clone()
        recovered_history = [curr_state.clone()]
        T = len(keys_sequence)
        
        for t in reversed(range(T)):
            k = keys_sequence[t]
            v = values_sequence[t]
            curr_state = (curr_state - torch.outer(k, v)) / decay
            recovered_history.append(curr_state.clone())
            
        recovered_history.reverse()
        return recovered_history

    @staticmethod
    def deconvolve_sequence_pinv(
        current_state: torch.Tensor,
        keys_matrix: torch.Tensor,
        decay: float = 0.995,
    ) -> torch.Tensor:
        """
        Given keys_matrix [L, D] and current accumulated state S_T [D, H],
        extracts the recovered sequence V_recovered [L, H] using exact
        Moore-Penrose pseudo-inversion of the attenuated key basis.
        """
        L, D = keys_matrix.shape
        device = keys_matrix.device
        attenuation = torch.tensor([decay ** (L - 1 - i) for i in range(L)], device=device).unsqueeze(-1)
        k_attenuated = keys_matrix * attenuation
        k_pinv = torch.linalg.pinv(k_attenuated)  # [D, L]
        v_recovered = torch.matmul(k_pinv.T, current_state)  # [L, H]
        return v_recovered

    @staticmethod
    def compute_landauer_horizon(decay: float = 0.995, precision_bits: int = 24) -> Dict[str, Any]:
        """
        Computes the theoretical and numerical event horizon before signal
        amplitude drops below IEEE-754 mantissa resolution into the noise floor.
        """
        eps_machine = 2.0 ** (-precision_bits)
        log_eps = math.log(eps_machine)
        log_decay = math.log(decay)
        tau_crit = log_eps / log_decay
        
        return {
            "precision_bits": precision_bits,
            "decay": decay,
            "machine_epsilon": eps_machine,
            "critical_token_horizon": int(tau_crit),
            "exact_horizon_steps": round(tau_crit, 2),
            "derivation_formula": f"tau_crit = ln({eps_machine:.2e}) / ln({decay}) = {tau_crit:.1f} tokens"
        }
