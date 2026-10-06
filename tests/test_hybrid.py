import unittest
import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Config
from prime_moment_attention import HybridWindowPrimeAttention

class TestHybridWindowPrimeAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = Qwen2Config(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256
        )
        self.orig_attn = Qwen2Attention(self.cfg, layer_idx=0)
        self.window_size = 8
        self.hybrid = HybridWindowPrimeAttention(
            original_attn=self.orig_attn,
            layer_idx=0,
            window_size=self.window_size,
            decay=0.999,
            alpha=0.5
        )

    def test_hybrid_prefill_and_eviction(self):
        class SimplePast:
            def __init__(self):
                self.hybrid_prime_states = {}

        past = SimplePast()
        seq_len = 16  # Exceeds window_size 8
        x = torch.randn(1, seq_len, self.cfg.hidden_size)

        out, _ = self.hybrid(x, past_key_values=past)
        self.assertEqual(out.shape, (1, seq_len, self.cfg.hidden_size))
        self.assertIn(0, past.hybrid_prime_states)

        stored = past.hybrid_prime_states[0]
        # Window must be clamped to window_size
        self.assertEqual(stored['k_win'].shape[2], self.window_size)
        self.assertEqual(stored['v_win'].shape[2], self.window_size)

        # Generate step tokens
        for _ in range(5):
            token = torch.randn(1, 1, self.cfg.hidden_size)
            step_out, _ = self.hybrid(token, past_key_values=past)
            self.assertEqual(step_out.shape, (1, 1, self.cfg.hidden_size))
            self.assertEqual(past.hybrid_prime_states[0]['k_win'].shape[2], self.window_size)

if __name__ == "__main__":
    unittest.main()
