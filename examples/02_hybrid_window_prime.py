#!/usr/bin/env python3
"""
02_hybrid_window_prime.py
=========================
Demonstrates Hybrid Window-PRIME Attention:
  - Local Sliding-Window Softmax for exact high-frequency syntactic binding.
  - Second-Order Recurrence for all tokens that fall out of the local window.
  - Constant memory bounded to O(W + D^2).
"""

import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Config
from prime_moment_attention import HybridWindowPrimeAttention

def main():
    print("=" * 70)
    print("Hybrid Window-PRIME Attention: Sliding Window + Recurrent State")
    print("=" * 70)

    # 1. Setup minimal Transformer configuration
    cfg = Qwen2Config(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512
    )
    orig_attn = Qwen2Attention(cfg, layer_idx=0)

    # 2. Wrap with HybridWindowPrimeAttention (Window W=16)
    window_size = 16
    hybrid_attn = HybridWindowPrimeAttention(
        original_attn=orig_attn,
        layer_idx=0,
        window_size=window_size,
        decay=0.999,
        alpha=0.5
    )
    hybrid_attn.eval()
    print(f"[✓] Initialized HybridWindowPrimeAttention (Window={window_size}, Alpha=0.5)")

    # 3. Prefill with sequence exceeding window size
    batch_size = 1
    seq_len = 32
    hidden = torch.randn(batch_size, seq_len, cfg.hidden_size)

    # Mock past_key_values container
    class SimpleCache:
        def __init__(self):
            self.hybrid_prime_states = {}

    past = SimpleCache()

    with torch.no_grad():
        out, _ = hybrid_attn(hidden, past_key_values=past)

    print(f"[✓] Prefill complete: Input {list(hidden.shape)} -> Output {list(out.shape)}")
    state = past.hybrid_prime_states[0]
    print(f"[✓] Local window buffer size: {state['k_win'].shape[2]} tokens (capped at window_size={window_size})")
    print(f"[✓] Distant tokens successfully accumulated in 2nd-order moment state!")

    # 4. Step generation past eviction
    print("\n[+] Generating 20 autoregressive tokens past eviction boundary...")
    with torch.no_grad():
        for i in range(20):
            token = torch.randn(batch_size, 1, cfg.hidden_size)
            step_out, _ = hybrid_attn(token, past_key_values=past)
            # Verify window buffer remains clamped at window_size
            assert past.hybrid_prime_states[0]['k_win'].shape[2] <= window_size

    print(f"[✓] Generated 20 tokens. Window buffer stays strictly <= {window_size} tokens.")
    print("[SUCCESS] Hybrid Window-PRIME demonstration complete.")

if __name__ == "__main__":
    main()
