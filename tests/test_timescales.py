import unittest
import torch
from prime_moment_attention import LearnableTimescales

class TestLearnableTimescales(unittest.TestCase):
    def test_timescale_initialization(self):
        num_heads = 8
        ts = LearnableTimescales(num_heads=num_heads, min_tau=2.0, max_tau=1000.0)

        taus = ts.get_taus()
        decays = ts.get_decays()

        self.assertEqual(taus.shape, (num_heads,))
        self.assertEqual(decays.shape, (num_heads,))

        # Verify decay bounds
        self.assertTrue((decays > 0.0).all() and (decays < 1.0).all())
        self.assertTrue((taus >= 1.9).all() and (taus <= 1005.0).all())

    def test_timescale_gradients(self):
        num_heads = 4
        ts = LearnableTimescales(num_heads=num_heads)
        decays = ts.get_decays()
        loss = decays.sum()
        loss.backward()

        self.assertIsNotNone(ts.theta.grad)
        self.assertFalse(torch.isnan(ts.theta.grad).any())

if __name__ == "__main__":
    unittest.main()
