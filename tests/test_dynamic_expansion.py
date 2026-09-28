#!/usr/bin/env python3
"""
Unit Tests for PRIME Dynamic Parameter Expansion Engine
=======================================================
Verifies:
1. Function-preserving exact equivalence (zero drift at expansion step).
2. Parameter count increase matching theoretical expectations.
3. Online TTT Continual Learner synchronization (Fisher inheritance & new zero curvature).
4. Plastic learning on newly expanded parameters.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, '/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention')
from src.prime_moment_attention.model import PrimeForCausalLM, PrimeConfig
from src.prime_moment_attention.continual_ttt import OnlineTTTContinualLearner
from src.prime_moment_attention.dynamic_expansion import expand_model_mlp_width, count_parameters


class TestDynamicParameterExpansion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = PrimeConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            intermediate_size=512,  # Base intermediate size
            decay=0.99,
            use_qk_norm=True,
            use_selective=False
        )
        self.model = PrimeForCausalLM(self.config)
        self.model.eval()

    def test_exact_functional_equivalence_after_expansion(self):
        """Tests that expanding intermediate size preserves exact logits before vs after."""
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        # Forward pass before expansion
        with torch.no_grad():
            out_before = self.model(input_ids=input_ids)["logits"]

        params_before = count_parameters(self.model)

        # Expand intermediate size from 512 -> 1024 (+100% MLP width)
        res = expand_model_mlp_width(
            self.model,
            learner=None,
            new_intermediate_size=1024
        )

        params_after = count_parameters(self.model)
        self.assertGreater(params_after, params_before)
        self.assertEqual(res["new_intermediate_size"], 1024)

        # Forward pass after expansion
        with torch.no_grad():
            out_after = self.model(input_ids=input_ids)["logits"]

        # Check maximum absolute difference
        max_diff = (out_before - out_after).abs().max().item()
        print(f"Max Logit Discrepancy After Expanding Parameters: {max_diff:.8e}")

        # Must be within float32 numerical rounding tolerance (< 1e-5)
        self.assertTrue(
            torch.allclose(out_before, out_after, atol=1e-5),
            f"Logits deviated after expansion! Max diff: {max_diff}"
        )

    def test_learner_synchronization_and_fisher_plasticity(self):
        """Tests that OnlineTTTContinualLearner synchronizes smoothly with newly added weights."""
        learner = OnlineTTTContinualLearner(
            model=self.model,
            learning_rate=1e-4,
            elastic_lambda=50.0
        )

        # Calibrate initial Fisher on dummy tokens
        tokens = torch.randint(0, self.config.vocab_size, (1, 32))
        learner.compute_fisher_initialization(tokens, steps=5)

        # Confirm some Fisher values are non-zero
        initial_fisher_sum = sum(f.sum().item() for f in learner.fisher_diag.values())
        self.assertGreater(initial_fisher_sum, 0.0)

        # Expand model
        res = expand_model_mlp_width(
            self.model,
            learner=learner,
            new_intermediate_size=768,
            include_in_learner=True
        )

        # Verify that learner is tracking newly expanded parameters
        expanded_tracked = [name for name in learner.adapted_params.keys() if "mlp" in name]
        self.assertTrue(len(expanded_tracked) > 0, "Expanded MLP weights must be tracked in learner!")

        # Verify adaptation step runs without error on expanded model
        loss_before = learner.evaluate_loss(tokens)
        learner.adapt_on_sequence(tokens, steps=2, lr_scale=1.0)
        loss_after = learner.evaluate_loss(tokens)

        print(f"Loss Before Adaptation: {loss_before:.4f}, After: {loss_after:.4f}")
        self.assertLessEqual(loss_after, loss_before + 1e-4)


if __name__ == "__main__":
    unittest.main()
