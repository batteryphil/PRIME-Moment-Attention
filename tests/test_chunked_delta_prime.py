#!/usr/bin/env python3
import unittest
import torch
from prime_moment_attention.chunked_delta_prime import ChunkedGatedDeltaPrimeAttention
from prime_moment_attention.harmonic_rope import HarmonicPhasorEmbedding, apply_harmonic_rotary_emb


class TestWave2Breakthroughs(unittest.TestCase):
    def test_chunked_gated_delta_prime_prefill_and_step(self):
        B, L, D_model, H, D_head = 2, 256, 128, 4, 32
        layer = ChunkedGatedDeltaPrimeAttention(
            hidden_size=D_model,
            num_heads=H,
            head_dim=D_head,
            chunk_size=32,
            lyapunov_bound=20.0
        )
        x = torch.randn(B, L, D_model)
        out, state = layer(x, return_state=True)
        self.assertEqual(out.shape, (B, L, D_model))
        self.assertEqual(state.shape, (B, H, D_head, 2 * D_head))

        # Test single-step generation with state carryover
        x_step = torch.randn(B, 1, D_model)
        out_step, next_state = layer(x_step, state=state, return_state=True)
        self.assertEqual(out_step.shape, (B, 1, D_model))
        self.assertEqual(next_state.shape, (B, H, D_head, 2 * D_head))

    def test_harmonic_phasor_rope(self):
        B, H, L, D = 1, 2, 64, 32
        rope = HarmonicPhasorEmbedding(dim=D)
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)

        cos1, sin1, cos2, sin2 = rope(q, seq_len=L)
        q_rot, k_rot, q2_rot, k2_rot = apply_harmonic_rotary_emb(q, k, cos1, sin1, cos2, sin2)

        self.assertEqual(q_rot.shape, (B, H, L, D))
        self.assertEqual(k_rot.shape, (B, H, L, D))
        self.assertEqual(q2_rot.shape, (B, H, L, D))
        self.assertEqual(k2_rot.shape, (B, H, L, D))


if __name__ == "__main__":
    unittest.main()
