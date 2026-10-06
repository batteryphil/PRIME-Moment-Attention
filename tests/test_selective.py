import unittest
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Config
from prime_moment_attention import PrimeSelectiveAttention

class TestPrimeSelectiveAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = Qwen2Config(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256
        )
        self.orig_attn = Qwen2Attention(self.cfg, layer_idx=0)
        self.selective = PrimeSelectiveAttention(
            original_attn=self.orig_attn,
            layer_idx=0,
            min_tau=2.0,
            max_tau=500.0,
            init_beta=4.0
        )

    def test_selective_prefill_and_step(self):
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, self.cfg.hidden_size)

        out, _ = self.selective(x)
        self.assertEqual(out.shape, (batch_size, seq_len, self.cfg.hidden_size))
        self.assertFalse(torch.isnan(out).any())

    def test_learned_parameters(self):
        tau = self.selective.get_tau(torch.device("cpu"))
        beta = self.selective.get_beta(torch.device("cpu"))
        self.assertEqual(tau.shape, (self.cfg.num_attention_heads,))
        self.assertEqual(beta.shape, (self.cfg.num_attention_heads,))
        self.assertTrue((tau >= 1.5).all() and (tau <= 2000.0).all())
        self.assertTrue((beta >= 1.5).all() and (beta <= 12.0).all())

if __name__ == "__main__":
    unittest.main()
