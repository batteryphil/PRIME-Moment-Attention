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
        self.assertEqual(state.shape, (2, 32, 512))

    def test_state_memory_footprint(self):
        # 32 * 512 * 4 bytes = 65536 bytes = 64.0 KB
        mem_kb = self.layer.state_bytes / 1024.0
        self.assertEqual(mem_kb, 64.0)

    def test_autoregressive_step(self):
        x = torch.randn(2, 64, 512)
        _, state = self.layer(x, return_state=True)
        x_step = torch.randn(2, 1, 512)
        out_step, next_state = self.layer(x_step, state=state, return_state=True)
        self.assertEqual(out_step.shape, (2, 1, 512))
        self.assertEqual(next_state.shape, (2, 32, 512))


if __name__ == "__main__":
    unittest.main()
