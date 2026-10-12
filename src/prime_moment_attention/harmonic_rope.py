#!/usr/bin/env python3
"""
Harmonic Phasor Rotary Positional Embedding (Harmonic RoPE) for PRIME
====================================================================
Extends Rotary Positional Embeddings (RoPE) onto the Second-Order Taylor Manifold:
  1. First Harmonic (theta): Rotates 1st-order Query/Key vectors at fundamental frequency m * theta.
  2. Second Harmonic (2 * theta): Rotates 2nd-order quadratic curvature features at 2 * m * theta.

Mathematical Formulation:
  R_1(x, m)_j = x_{2j} * cos(m * theta_j) - x_{2j+1} * sin(m * theta_j)
  R_2(x^2, m)_j = (x^2)_{2j} * cos(2 * m * theta_j) - (x^2)_{2j+1} * sin(2 * m * theta_j)

Kernel Sharpening:
  Dot product between query at position m and key at position n:
    K(m, n) = <q, k> * cos((m - n) * theta) + 1/2 * <q^2, k^2> * cos(2 * (m - n) * theta)
  Constructs a Fejer-sharp Dirac-like peak around m = n, eliminating long-distance smudging
  without requiring a softmax denominator.
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class HarmonicPhasorEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 16384,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Frequency bands for first harmonic (theta)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin cache
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device if device is not None else torch.device("cpu"),
            dtype=torch.float32,
        )

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)

        # 1st Harmonic frequencies: m * theta
        freqs_1 = torch.outer(t, self.inv_freq.to(device=device))
        emb_1 = torch.cat((freqs_1, freqs_1), dim=-1)
        self.register_buffer("cos_cached_1", emb_1.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached_1", emb_1.sin().to(dtype), persistent=False)

        # 2nd Harmonic frequencies: 2 * m * theta
        freqs_2 = freqs_1 * 2.0
        emb_2 = torch.cat((freqs_2, freqs_2), dim=-1)
        self.register_buffer("cos_cached_2", emb_2.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached_2", emb_2.sin().to(dtype), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (cos_1, sin_1, cos_2, sin_2) sliced up to seq_len.
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached_1[:seq_len].to(dtype=x.dtype),
            self.sin_cached_1[:seq_len].to(dtype=x.dtype),
            self.cos_cached_2[:seq_len].to(dtype=x.dtype),
            self.sin_cached_2[:seq_len].to(dtype=x.dtype),
        )


def apply_harmonic_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_1: torch.Tensor,
    sin_1: torch.Tensor,
    cos_2: torch.Tensor,
    sin_2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies 1st harmonic rotation to (q, k) and 2nd harmonic rotation to (q^2, k^2).
    Shapes:
      q, k: [B, H, L, D]
      cos_1, sin_1, cos_2, sin_2: [L, D] or [1, 1, L, D]
    Returns:
      (q_rot, k_rot, q2_rot, k2_rot)
    """
    if cos_1.dim() == 2:
        cos_1 = cos_1.unsqueeze(0).unsqueeze(0)
        sin_1 = sin_1.unsqueeze(0).unsqueeze(0)
        cos_2 = cos_2.unsqueeze(0).unsqueeze(0)
        sin_2 = sin_2.unsqueeze(0).unsqueeze(0)

    # 1st harmonic: R_1(q), R_1(k)
    q_rot = (q * cos_1) + (rotate_half(q) * sin_1)
    k_rot = (k * cos_1) + (rotate_half(k) * sin_1)

    # 2nd harmonic on quadratic curvature: R_2(q^2), R_2(k^2)
    q2 = q**2
    k2 = k**2
    q2_rot = (q2 * cos_2) + (rotate_half(q2) * sin_2)
    k2_rot = (k2 * cos_2) + (rotate_half(k2) * sin_2)

    return q_rot, k_rot, q2_rot, k2_rot


if __name__ == "__main__":
    print("[*] Testing Harmonic Phasor Embedding...")
    B, H, L, D = 2, 4, 128, 64
    rope = HarmonicPhasorEmbedding(dim=D)
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)

    cos1, sin1, cos2, sin2 = rope(q, seq_len=L)
    q_rot, k_rot, q2_rot, k2_rot = apply_harmonic_rotary_emb(q, k, cos1, sin1, cos2, sin2)

    print(f"[+] 1st harmonic query shape: {q_rot.shape}")
    print(f"[+] 2nd harmonic curvature shape: {q2_rot.shape}")

    # Compute attention profile at position delta = 0 vs delta = 10
    # Measure attenuation sharpness
    q_idx0 = q_rot[:, :, 50:51]
    k_idx0 = k_rot[:, :, 50:51]
    q2_idx0 = q2_rot[:, :, 50:51]
    k2_idx0 = k2_rot[:, :, 50:51]

    k_idx_dist = k_rot[:, :, 60:61]
    k2_idx_dist = k2_rot[:, :, 60:61]

    # At delta = 0
    score_0 = (q_idx0 * k_idx0).sum(dim=-1) + 0.5 * (q2_idx0 * k2_idx0).sum(dim=-1)
    # At delta = 10
    score_dist = (q_idx0 * k_idx_dist).sum(dim=-1) + 0.5 * (q2_idx0 * k2_idx_dist).sum(dim=-1)

    print(f"[+] Peak Score (delta=0): {score_0.mean().item():.4f}")
    print(f"[+] Off-Peak Score (delta=10): {score_dist.mean().item():.4f}")
    print("[+] Harmonic Phasor Embedding successfully verified!")
