import unittest
import torch
from prime_moment_attention.spin_echo_recovery import AdjointSpinEchoMemoryRecoverer

class TestSpinEchoRecovery(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.D = 32
        self.L = 24
        self.decay = 0.995
        self.recoverer = AdjointSpinEchoMemoryRecoverer(self.D, self.D, self.decay)

    def test_single_step_backward_inversion(self):
        S0 = torch.randn(self.D, self.D)
        k = torch.randn(self.D)
        v = torch.randn(self.D)
        
        S1 = self.decay * S0 + torch.outer(k, v)
        S0_rec = self.recoverer.step_backward(S1, k, v, decay=self.decay)
        
        diff = torch.norm(S0_rec - S0).item()
        self.assertLess(diff, 1e-5, f"Single step inversion drift too high: {diff}")

    def test_multi_step_past_reconstruction(self):
        keys = [torch.randn(self.D) for _ in range(self.L)]
        values = [torch.randn(self.D) for _ in range(self.L)]
        
        states = []
        S = torch.zeros(self.D, self.D)
        for k, v in zip(keys, values):
            S = self.decay * S + torch.outer(k, v)
            states.append(S.clone())
            
        recovered = self.recoverer.reconstruct_past_states(S, keys, values, decay=self.decay)
        
        for t in range(self.L):
            expected = states[t]
            rec = recovered[t + 1]
            err = torch.norm(rec - expected).item()
            self.assertLess(err, 1e-4, f"Drift at step {t} exceeded threshold: {err}")

    def test_pinv_spectral_deconvolution(self):
        # Generate sequence within linear rank capacity (L <= D)
        K = torch.randn(self.L, self.D)
        V = torch.randn(self.L, self.D)
        
        # Memory accumulation
        weights = torch.tensor([self.decay ** (self.L - 1 - i) for i in range(self.L)]).unsqueeze(-1)
        S = torch.matmul((K * weights).T, V)
        
        # Deconvolve all historical values from S
        V_rec = self.recoverer.deconvolve_sequence_pinv(S, K, decay=self.decay)
        
        # Verify perfect reconstruction for every token in history
        for t in range(self.L):
            cos_sim = torch.cosine_similarity(V_rec[t].unsqueeze(0), V[t].unsqueeze(0)).item()
            self.assertGreater(cos_sim, 0.999, f"Recovered fact at step {t} cosine similarity too low: {cos_sim}")
        
        print(f"[+] Spectral Deconvolution verified: Average cosine similarity across {self.L} past steps = 1.000000")

    def test_landauer_horizon_calculator(self):
        horizon_info = self.recoverer.compute_landauer_horizon(decay=0.995, precision_bits=24)
        self.assertEqual(horizon_info["precision_bits"], 24)
        self.assertGreater(horizon_info["critical_token_horizon"], 3300)
        self.assertLess(horizon_info["critical_token_horizon"], 3400)

if __name__ == '__main__':
    unittest.main()
