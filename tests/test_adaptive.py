import unittest
import torch
from prime_moment_attention import AdaptivePrimeRouter, EmpiricalBandwidthRouter

class TestAdaptivePrime(unittest.TestCase):
    def test_adaptive_router(self):
        hidden_dim = 128
        router = AdaptivePrimeRouter(hidden_dim=hidden_dim, num_orders=3)
        batch_size = 4
        seq_len = 16
        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

        g, alpha_1, alpha_2 = router(x)

        # Check shapes
        self.assertEqual(g.shape, (batch_size, 3))
        self.assertEqual(alpha_1.shape, (batch_size, 1, 1, 1))
        self.assertEqual(alpha_2.shape, (batch_size, 1, 1, 1))

        # Check probabilities sum to 1
        sums = g.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

        # Check gradient flow
        loss = alpha_1.sum() + alpha_2.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_empirical_bandwidth_router(self):
        router = EmpiricalBandwidthRouter()
        # Test clean vs noisy signal
        clean_signal = torch.sin(torch.linspace(0, 10, 100)).unsqueeze(0).unsqueeze(-1)
        g, a1, a2 = router(clean_signal)
        self.assertEqual(g.shape, (1, 3))
        self.assertTrue(torch.allclose(g.sum(), torch.tensor(1.0), atol=1e-5))

if __name__ == "__main__":
    unittest.main()
