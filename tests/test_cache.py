import unittest
import torch
from prime_moment_attention import PrimeMomentCache

class TestPrimeMomentCache(unittest.TestCase):
    def test_cache_initialization_and_update(self):
        cache = PrimeMomentCache()
        self.assertIsNone(cache.get_state(0))

        # Create sample layer state
        B, H, D = 2, 4, 32
        S0 = torch.randn(B, H, D)
        S1 = torch.randn(B, H, D, D)
        S2 = torch.randn(B, H, D, D)
        K0 = torch.randn(B, H, 1)
        K1 = torch.randn(B, H, D)
        K2 = torch.randn(B, H, D)

        state = (S0, S1, S2, K0, K1, K2)
        cache.set_state(layer_idx=0, state=state)

        retrieved = cache.get_state(layer_idx=0)
        self.assertIsNotNone(retrieved)
        self.assertEqual(len(retrieved), 6)
        self.assertTrue(torch.equal(retrieved[0], S0))

    def test_cache_reset(self):
        cache = PrimeMomentCache()
        state = (torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
        cache.set_state(layer_idx=1, state=state)
        self.assertIsNotNone(cache.get_state(1))
        cache.reset()
        self.assertIsNone(cache.get_state(1))

if __name__ == "__main__":
    unittest.main()
