import unittest
import torch
import torch.nn.functional as F
from prime_moment_attention.venturi_entrainment import VenturiInformationGate

class TestVenturiEntrainment(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B = 2
        self.L = 4
        self.H = 64
        self.D_k = 32
        self.M = 16
        self.gate = VenturiInformationGate(
            hidden_size=self.H,
            key_dim=self.D_k,
            memory_slots=self.M,
            rho_density=1.25,
            tau_pressure=2.0,
            drag_scale=0.5
        )

    def test_bernoulli_pressure_scaling(self):
        # v = 0 -> Delta P = 0
        v_zero = torch.zeros(self.B, self.L)
        p_zero = self.gate.compute_bernoulli_vacuum(v_zero)
        self.assertTrue(torch.allclose(p_zero, torch.zeros_like(p_zero)))

        # Quadratic scaling: v -> 2*v -> Delta P -> 4*Delta P
        v1 = torch.ones(self.B, self.L) * 2.0
        v2 = torch.ones(self.B, self.L) * 4.0
        p1 = self.gate.compute_bernoulli_vacuum(v1)
        p2 = self.gate.compute_bernoulli_vacuum(v2)
        ratio = (p2 / p1).mean().item()
        self.assertAlmostEqual(ratio, 4.0, places=4, msg="Bernoulli pressure must scale quadratically with velocity")

    def test_venturi_dragging_related_information(self):
        # Create a custom memory bank with 2 slots:
        # Slot 0: Correlated with injected concept
        # Slot 1: Orthogonal
        mem_k = torch.zeros(2, self.D_k)
        mem_k[0, 0] = 1.0  # Semantic dimension 0
        mem_k[1, 1] = 1.0  # Semantic dimension 1

        mem_v = torch.zeros(2, self.H)
        mem_v[0] = 5.0     # Distinct value for Slot 0
        mem_v[1] = -5.0    # Distinct value for Slot 1

        # Test Case A: Injected token aligned with Slot 0, LOW velocity (small delta / at context mean)
        hidden = torch.zeros(1, 1, self.H)
        k_inj = torch.zeros(1, 1, self.D_k)
        k_inj[0, 0, 0] = 1.0 # Aligned with Slot 0
        v_base = torch.zeros(1, 1, self.H)
        delta_low = torch.tensor([[-0.99]]) # Near zero velocity

        self.gate.running_mu.copy_(k_inj[0, 0]) # Zero displacement from mean
        out_low, diag_low = self.gate(hidden, k_inj, v_base, delta=delta_low, custom_memory=(mem_k, mem_v))

        # Test Case B: Injected token aligned with Slot 0, HIGH velocity (large novel displacement & delta)
        self.gate.running_mu.zero_() # Displaced from mean
        delta_high = torch.tensor([[4.0]]) # High injection delta
        out_high, diag_high = self.gate(hidden, k_inj, v_base, delta=delta_high, custom_memory=(mem_k, mem_v))

        # High velocity must create a massive pressure drop and entrainment
        self.assertGreater(diag_high["mean_pressure_drop"], diag_low["mean_pressure_drop"] * 10.0)
        self.assertGreater(diag_high["mean_entrainment_suction"], diag_low["mean_entrainment_suction"])
        self.assertGreater(diag_high["drag_norm"], diag_low["drag_norm"])

        # Correlated slot 0 must be dragged, while orthogonal slot 1 is ignored
        print(f"[+] Low Velocity Suction Factor:  {diag_low['mean_entrainment_suction']:.4f}, Drag Norm: {diag_low['drag_norm']:.4f}")
        print(f"[+] High Velocity Suction Factor: {diag_high['mean_entrainment_suction']:.4f}, Drag Norm: {diag_high['drag_norm']:.4f}")

    def test_differentiability(self):
        hidden = torch.randn(self.B, self.L, self.H, requires_grad=True)
        keys = torch.randn(self.B, self.L, self.D_k, requires_grad=True)
        values = torch.randn(self.B, self.L, self.H, requires_grad=True)
        out, _ = self.gate(hidden, keys, values)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(keys.grad)
        self.assertIsNotNone(values.grad)
        self.assertIsNotNone(self.gate.rho.grad)

if __name__ == '__main__':
    unittest.main()
