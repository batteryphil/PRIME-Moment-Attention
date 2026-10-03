import unittest
import torch
from prime_moment_attention.thought_reconstruction_map import GenerativeThoughtReconstructionLayer

class TestThoughtReconstructionMap(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.layer = GenerativeThoughtReconstructionLayer(d_model=512, d_map=32)
        self.layer.eval()

    def test_parallel_forward_pass(self):
        x = torch.randn(2, 64, 512)
        out, state = self.layer(x, return_state=True)
        self.assertEqual(out.shape, x.shape)
        # State is now a tuple (m, Z)
        m, Z = state
        self.assertEqual(m.shape, (2, 32, 512))
        self.assertEqual(Z.shape, (2, 32))

    def test_autoregressive_step(self):
        x = torch.randn(2, 64, 512)
        _, state = self.layer(x, return_state=True)
        x_step = torch.randn(2, 1, 512)
        out_step, next_state = self.layer(x_step, state=state, return_state=True)
        self.assertEqual(out_step.shape, (2, 1, 512))
        m_next, Z_next = next_state
        self.assertEqual(m_next.shape, (2, 32, 512))
        self.assertEqual(Z_next.shape, (2, 32))

if __name__ == "__main__":
    unittest.main()
