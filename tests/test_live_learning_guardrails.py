#!/usr/bin/env python3
"""
Unit tests for PRIME Live Learning Guardrails, Surprise Gating, and Anchor Rollback.
"""

import unittest
import torch
from prime_moment_attention.model import PrimeConfig, PrimeForCausalLM
from prime_moment_attention.continual_ttt import OnlineTTTContinualLearner


class TestLiveLearningGuardrails(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = PrimeConfig(vocab_size=500, hidden_size=64, num_layers=2, num_heads=2, head_dim=32)
        self.model = PrimeForCausalLM(self.cfg)

    def test_surprise_threshold_gating(self):
        """Verifies that surprise threshold prevents updates when prediction error is below tau."""
        learner = OnlineTTTContinualLearner(
            self.model,
            learning_rate=1e-3,
            surprise_threshold=5.0,  # High threshold
        )
        
        # When target equals predicted token or loss is very low
        token_id = 10
        pred, loss_val, _ = learner.process_and_learn_token(token_id=token_id, target_token_id=token_id)
        
        # If loss < 5.0, total_plastic_updates should remain 0
        if loss_val <= 5.0:
            self.assertEqual(learner.total_plastic_updates, 0, "Update triggered despite loss <= surprise_threshold")

    def test_anchor_rollback_guardrail(self):
        """Verifies that check_and_rollback_if_drifted rolls back weights when anchor loss spikes."""
        learner = OnlineTTTContinualLearner(
            self.model,
            learning_rate=0.5,  # Aggressively high LR to force drift
            elastic_lambda=0.0, # Disable EWC to allow catastrophic forgetting
            adapted_modules="all",
        )

        anchor_seq = torch.randint(0, 500, (1, 32))
        initial_loss = learner.evaluate_loss(anchor_seq)
        learner.initial_anchor_loss = initial_loss

        # Distort weights with an extreme sequence
        distort_seq = torch.full((1, 32), 123, dtype=torch.long)
        learner.adapt_on_sequence(distort_seq, steps=10, lr_scale=5.0)

        # Check rollback
        safe, drift = learner.check_and_rollback_if_drifted(anchor_seq, max_allowed_drift=0.05)
        self.assertFalse(safe, "Guardrail failed to detect catastrophic anchor drift")
        
        # Verify weights were restored to anchor
        loss_after_rollback = learner.evaluate_loss(anchor_seq)
        self.assertAlmostEqual(loss_after_rollback, initial_loss, places=4, msg="Weights were not properly restored to anchor")

    def test_gradient_norm_clipping(self):
        """Verifies that gradient clipping prevents explosive parameter updates."""
        learner = OnlineTTTContinualLearner(
            self.model,
            learning_rate=1.0,
            max_grad_norm=0.5,
        )
        seq = torch.randint(0, 500, (1, 16))
        loss = learner.adapt_on_sequence(seq, steps=2)
        self.assertFalse(torch.isnan(torch.tensor(loss)), "Loss became NaN")
        for p in learner.adapted_params.values():
            self.assertFalse(torch.isnan(p).any(), "Parameters contained NaN after update")


if __name__ == "__main__":
    unittest.main()
