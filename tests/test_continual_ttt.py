#!/usr/bin/env python3
"""
Unit test for PRIME Online Test-Time Training (TTT) and Elastic Synaptic Plasticity.
"""
import unittest
import torch
from prime_moment_attention.model import PrimeConfig, PrimeForCausalLM
from prime_moment_attention.continual_ttt import OnlineTTTContinualLearner


class TestContinualTTT(unittest.TestCase):
    def test_online_ttt_adaptation_and_catastrophic_forgetting_prevention(self):
        torch.manual_seed(42)
        cfg = PrimeConfig(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=4, head_dim=32)
        model = PrimeForCausalLM(cfg)
        learner = OnlineTTTContinualLearner(model, learning_rate=5e-3, elastic_lambda=50.0)

        # 1. Calibration on anchor
        anchor_seq = torch.randint(0, 1000, (1, 64))
        learner.compute_fisher_initialization(anchor_seq, steps=10)
        anchor_loss_initial = learner.evaluate_loss(anchor_seq)

        # 2. Online adaptation on novel sequence
        novel_seq = torch.randint(100, 200, (1, 32))
        loss_before = learner.evaluate_loss(novel_seq)
        learner.adapt_on_sequence(novel_seq, steps=5, lr_scale=2.0)
        loss_after = learner.evaluate_loss(novel_seq)

        self.assertLess(loss_after, loss_before, "Novel sequence loss failed to decrease")

        # 3. Catastrophic forgetting check
        anchor_loss_after = learner.evaluate_loss(anchor_seq)
        drift = abs(anchor_loss_after - anchor_loss_initial)
        self.assertLess(drift, 0.05, f"Catastrophic drift detected on anchor weights: {drift}")

        # 4. Token-by-token online adaptation check
        token_id = 150
        pred, loss_val, state = learner.process_and_learn_token(token_id=token_id, target_token_id=151, enable_learning=True)
        self.assertIsInstance(pred, int)
        self.assertGreater(loss_val, 0.0)


if __name__ == "__main__":
    unittest.main()
