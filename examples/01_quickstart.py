#!/usr/bin/env python3
"""
01_quickstart.py
================
Demonstrates the fundamental mechanics of PRIME Moment Attention:
  1. Module initialization with QK-normalization.
  2. Full-sequence prefill forward pass.
  3. Autoregressive token-by-token decoding with O(1) constant recurrent state.
  4. State memory footprint verification (invariant to sequence length L).
"""

import torch
from prime_moment_attention import PrimeMomentAttention

def main():
    print("=" * 70)
    print("PRIME Moment Attention: Minimal Quickstart")
    print("=" * 70)

    # 1. Initialize PRIME Attention module
    hidden_size = 512
    num_heads = 8
    head_dim = hidden_size // num_heads  # 64
    decay = 0.9995

    attn = PrimeMomentAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        decay=decay,
        use_qk_norm=True
    )
    attn.eval()
    print(f"[✓] Initialized PrimeMomentAttention (H={num_heads}, D={head_dim}, decay={decay})")

    # 2. Prefill Phase: Process prompt sequence
    batch_size = 2
    prompt_len = 32
    prompt_tokens = torch.randn(batch_size, prompt_len, hidden_size)

    with torch.no_grad():
        prefill_out, state = attn(prompt_tokens, return_state=True)

    print(f"[✓] Prefill completed: Input {list(prompt_tokens.shape)} -> Output {list(prefill_out.shape)}")

    # Calculate exact state size in bytes
    S0, S1, S2, K0, K1, K2 = state
    total_state_bytes = sum(t.element_size() * t.nelement() for t in (S0, S1, S2, K0, K1, K2))
    print(f"[✓] Recurrent attention state footprint: {total_state_bytes / 1024:.2f} KB (Batch={batch_size})")

    # 3. Autoregressive Decode Phase: Step-by-step generation
    print("\n[+] Stepping through 50 autoregressive decode tokens...")
    initial_bytes = total_state_bytes

    with torch.no_grad():
        for step in range(50):
            # Single incoming new token (L == 1)
            new_token = torch.randn(batch_size, 1, hidden_size)
            step_out, state = attn(new_token, state=state, return_state=True)

            current_bytes = sum(t.element_size() * t.nelement() for t in state)
            assert current_bytes == initial_bytes, "Memory grew! Recurrent state must be O(1) constant!"

    print(f"[✓] 50 tokens decoded successfully!")
    print(f"[✓] Final state footprint: {current_bytes / 1024:.2f} KB (100% constant, zero growth!)")
    print("\n[SUCCESS] Quickstart demonstration complete.")

if __name__ == "__main__":
    main()
