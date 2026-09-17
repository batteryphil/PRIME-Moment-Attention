import unittest
import torch
from prime_moment_attention import PrimeMomentAttention

class TestPrimeMomentAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.hidden_size = 128
        self.num_heads = 4
        self.head_dim = 32
        self.attn = PrimeMomentAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            decay=0.999,
            use_qk_norm=True
        )

    def test_prefill_output_shape(self):
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, self.hidden_size)
        out, state = self.attn(x, return_state=True)

        self.assertEqual(out.shape, (batch_size, seq_len, self.hidden_size))
        self.assertIsNotNone(state)
        S0, S1, S2, K0, K1, K2 = state
        self.assertEqual(S0.shape, (batch_size, self.num_heads, self.head_dim))
        self.assertEqual(S1.shape, (batch_size, self.num_heads, self.head_dim, self.head_dim))
        self.assertEqual(S2.shape, (batch_size, self.num_heads, self.head_dim, self.head_dim))
        self.assertEqual(K0.shape, (batch_size, self.num_heads, 1))
        self.assertEqual(K1.shape, (batch_size, self.num_heads, self.head_dim))
        self.assertEqual(K2.shape, (batch_size, self.num_heads, self.head_dim))

    def test_step_decode_constant_memory(self):
        batch_size = 2
        x_init = torch.randn(batch_size, 4, self.hidden_size)
        _, state = self.attn(x_init, return_state=True)

        initial_bytes = sum(t.element_size() * t.nelement() for t in state)

        # Decode for 20 steps
        for _ in range(20):
            token = torch.randn(batch_size, 1, self.hidden_size)
            step_out, state = self.attn(token, state=state, return_state=True)
            self.assertEqual(step_out.shape, (batch_size, 1, self.hidden_size))
            current_bytes = sum(t.element_size() * t.nelement() for t in state)
            self.assertEqual(current_bytes, initial_bytes, "Recurrent state memory must remain constant")

    def test_gradient_flow(self):
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, self.hidden_size, requires_grad=True)
        out, _ = self.attn(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertIsNotNone(self.attn.q_proj.weight.grad)
        self.assertIsNotNone(self.attn.k_proj.weight.grad)
        self.assertIsNotNone(self.attn.v_proj.weight.grad)

    def test_numerical_stability(self):
        # Large sequence prefill to verify no NaN or Inf
        batch_size = 1
        seq_len = 128
        x = torch.randn(batch_size, seq_len, self.hidden_size) * 5.0
        out, _ = self.attn(x)
        self.assertFalse(torch.isnan(out).any(), "Output must not contain NaNs")
        self.assertFalse(torch.isinf(out).any(), "Output must not contain Infs")

if __name__ == "__main__":
    unittest.main()
