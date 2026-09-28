#!/usr/bin/env python3
"""
Unit test for Chunked State Space Duality (SSD) Parallel Scan exact parity.
"""
import unittest
import math
import torch
from prime_moment_attention.attention import PrimeMomentAttention
from prime_moment_attention.chunked_ssd import ChunkedPrimeSSD, chunked_prime_ssd_core


class TestChunkedSSDParity(unittest.TestCase):
    def test_chunked_ssd_mathematical_parity(self):
        B, L, hidden_size, H, head_dim = 2, 256, 128, 4, 32
        decay = 0.9995

        dense_attn = PrimeMomentAttention(
            hidden_size=hidden_size, num_heads=H, head_dim=head_dim, decay=decay, use_qk_norm=True
        )
        chunk_attn = ChunkedPrimeSSD(
            hidden_size=hidden_size, num_heads=H, head_dim=head_dim, chunk_size=64, decay=decay, use_qk_norm=True
        )

        chunk_attn.q_proj.weight.data.copy_(dense_attn.q_proj.weight.data)
        chunk_attn.k_proj.weight.data.copy_(dense_attn.k_proj.weight.data)
        chunk_attn.v_proj.weight.data.copy_(dense_attn.v_proj.weight.data)
        chunk_attn.o_proj.weight.data.copy_(dense_attn.o_proj.weight.data)
        chunk_attn.q_norm.weight.data.copy_(dense_attn.q_norm.weight.data)
        chunk_attn.q_norm.bias.data.copy_(dense_attn.q_norm.bias.data)
        chunk_attn.k_norm.weight.data.copy_(dense_attn.k_norm.weight.data)
        chunk_attn.k_norm.bias.data.copy_(dense_attn.k_norm.bias.data)

        x = torch.randn(B, L, hidden_size)
        out_dense, state_dense = dense_attn(x, return_state=True)
        out_chunk, state_chunk = chunk_attn(x, return_state=True)

        diff_out = torch.abs(out_dense - out_chunk).max().item()
        self.assertLess(diff_out, 1e-5, f"Discrepancy too high: {diff_out}")

        for i, (sd, sc) in enumerate(zip(state_dense, state_chunk)):
            diff_s = torch.abs(sd.squeeze() - sc.squeeze()).max().item()
            self.assertLess(diff_s, 2e-3, f"State component {i} diff: {diff_s}")


if __name__ == "__main__":
    unittest.main()
