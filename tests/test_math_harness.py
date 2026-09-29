#!/usr/bin/env python3
"""
Unit Tests for PRIME-Net Neuro-Symbolic Math Harness
====================================================
Tests safe arithmetic/algebraic evaluation, calculation tag interception,
and prompt invariant extraction.
"""

import unittest
from prime_moment_attention.primenet_harness import PrimeNetMathHarness


class TestPrimeNetMathHarness(unittest.TestCase):
    def setUp(self):
        self.harness = PrimeNetMathHarness(verbose=False)

    def test_safe_eval_arithmetic(self):
        # Basic arithmetic
        self.assertEqual(self.harness.safe_eval("(1500 - 320) / 4"), 295)
        self.assertEqual(self.harness.safe_eval("25 * 16"), 400)
        self.assertEqual(self.harness.safe_eval("80000 + 50000"), 130000)
        self.assertEqual(self.harness.safe_eval("12 * 8 + 4"), 100)

    def test_safe_eval_percentage(self):
        # Percentage calculation
        self.assertEqual(self.harness.safe_eval("20 percent of 150"), 30)
        self.assertEqual(self.harness.safe_eval("25% of 200"), 50)
        self.assertEqual(self.harness.safe_eval("15% of 80"), 12)

    def test_safe_eval_algebra(self):
        # Linear equation solving
        self.assertEqual(self.harness.safe_eval("Solve for x: 3 * x + 7 = 22"), 5)
        self.assertEqual(self.harness.safe_eval("2 * x + 10 = 30"), 10)

    def test_intercept_gsm8k_tags(self):
        # GSM8K format <<expr>> or <<expr=wrong>>
        text = "Natalia sold clips to 48 friends <<48 / 2>> in April."
        updated, injections = self.harness.intercept_and_solve(text)
        self.assertIn("<<48 / 2=24>>24", updated)
        self.assertEqual(len(injections), 1)
        self.assertEqual(injections[0]["expr"], "48 / 2")
        self.assertEqual(injections[0]["result"], "24")

    def test_intercept_bracket_tags(self):
        # Bracket format [calc: expr]
        text = "The speed is [calc: 12 * 15] km/h."
        updated, injections = self.harness.intercept_and_solve(text)
        self.assertIn("[PRIME-Net: 12 * 15 = 180]", updated)
        self.assertEqual(len(injections), 1)
        self.assertEqual(injections[0]["result"], "180")

    def test_intercept_equality(self):
        # Equality format "X op Y ="
        text = "She had 15 apples and bought 25 more. Total = 15 + 25 ="
        updated, injections = self.harness.intercept_and_solve(text)
        self.assertIn("15 + 25 = 40", updated)
        self.assertEqual(len(injections), 1)
        self.assertEqual(injections[0]["result"], "40")

    def test_extract_invariants_algebra(self):
        prompt = "Solve for x: 4 * x - 8 = 24. What is x?"
        inv = self.harness.extract_prompt_invariants(prompt)
        self.assertIsNotNone(inv)
        self.assertIn("x = 8", inv)

    def test_extract_invariants_percentage(self):
        prompt = "What is 20 percent of 150?"
        inv = self.harness.extract_prompt_invariants(prompt)
        self.assertIsNotNone(inv)
        self.assertIn("20% of 150 = 30", inv)

    def test_extract_invariants_arithmetic(self):
        prompt = "Calculate 125 * 8 and report the result."
        inv = self.harness.extract_prompt_invariants(prompt)
        self.assertIsNotNone(inv)
        self.assertIn("125 * 8 = 1000", inv)

    def test_extract_invariants_sequence(self):
        prompt = "Find the next number in the sequence: 1, 4, 9, 16, 25."
        inv = self.harness.extract_prompt_invariants(prompt)
        self.assertIsNotNone(inv)
        self.assertIn("n**2", inv)


if __name__ == "__main__":
    unittest.main()
