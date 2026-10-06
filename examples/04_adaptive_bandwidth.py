#!/usr/bin/env python3
"""
04_adaptive_bandwidth.py
========================
Demonstrates Adaptive PRIME Contextual Bandwidth Allocation:
  - Dynamically routes between:
      * Order 0 (Mean Context Field): Low-pass noise rejection / smoothing.
      * Order 1 (Directional Field): Query-conditioned associative addressing.
      * Order 2 (Curvature Field): Higher-order nonlinear geometric dynamics.
  - Predicts soft router weights [g0, g1, g2] dynamically from token representations.
"""

import torch
from prime_moment_attention import AdaptivePrimeRouter

def main():
    print("=" * 70)
    print("Adaptive PRIME: Dynamic Contextual Bandwidth Routing")
    print("=" * 70)

    hidden_dim = 256
    router = AdaptivePrimeRouter(hidden_dim=hidden_dim, num_orders=3)
    router.eval()
    print(f"[✓] Initialized AdaptivePrimeRouter for hidden_dim={hidden_dim}")

    # Generate sample representations representing different prompt dynamics
    batch_size = 3
    seq_len = 24
    x = torch.randn(batch_size, seq_len, hidden_dim)

    g, alpha_1, alpha_2 = router(x)

    print(f"\n[+] Input representation: {list(x.shape)}")
    print("[+] Dynamic Moment Order Weights [g0 (Mean), g1 (Linear), g2 (Quadratic)]:")
    for b in range(batch_size):
        weights = g[b].tolist()
        print(f"    Sample {b}: O0={weights[0]:.3f} | O1={weights[1]:.3f} | O2={weights[2]:.3f} (Sum={sum(weights):.3f})")
        print(f"             -> Effective Linear Gain (alpha_1): {alpha_1[b].item():.3f}")
        print(f"             -> Effective Quadratic Gain (alpha_2): {alpha_2[b].item():.3f}")

    print("\n[SUCCESS] Adaptive bandwidth demonstration complete.")

if __name__ == "__main__":
    main()
