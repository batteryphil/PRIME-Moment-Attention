import unittest
import torch
from prime_moment_attention import PrimeConfig, PrimeForCausalLM

class TestPrimeForCausalLM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = PrimeConfig(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            head_dim=16,
            decay=0.999,
            use_qk_norm=True
        )
        self.model = PrimeForCausalLM(self.config)

    def test_forward_and_loss(self):
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        out = self.model(input_ids, labels=labels)
        self.assertIn("logits", out)
        self.assertIn("loss", out)
        self.assertEqual(out["logits"].shape, (batch_size, seq_len, self.config.vocab_size))
        self.assertFalse(torch.isnan(out["loss"]))
        self.assertGreater(out["loss"].item(), 0.0)

    def test_backward_gradients(self):
        input_ids = torch.randint(0, self.config.vocab_size, (2, 8))
        labels = input_ids.clone()
        out = self.model(input_ids, labels=labels)
        loss = out["loss"]
        loss.backward()

        self.assertIsNotNone(self.model.embed_tokens.weight.grad)
        self.assertIsNotNone(self.model.layers[0].self_attn.q_proj.weight.grad)
        self.assertIsNotNone(self.model.layers[0].mlp.gate_proj.weight.grad)
        self.assertFalse(torch.isnan(self.model.layers[0].self_attn.q_proj.weight.grad).any())

    def test_generate(self):
        prompt = torch.tensor([[1, 2, 3, 4]])
        gen = self.model.generate(prompt, max_new_tokens=10, temperature=0.7)
        self.assertEqual(gen.shape, (1, 14))
        self.assertTrue((gen >= 0).all() and (gen < self.config.vocab_size).all())

if __name__ == "__main__":
    unittest.main()
