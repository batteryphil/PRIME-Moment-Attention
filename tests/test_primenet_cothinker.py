import unittest
from prime_moment_attention.primenet_cothinker_bridge import PrimeNetCoThinker


class TestPrimeNetCoThinker(unittest.TestCase):
    def setUp(self):
        self.cothinker = PrimeNetCoThinker()

    def test_safe_arithmetic(self):
        res = self.cothinker.safe_sym_eval("(1500 - 320) / 4")
        self.assertEqual(res, "295")

    def test_gsm8k_interception(self):
        text = "Cost is <<24 * 15>> dollars."
        out, inj = self.cothinker.intercept_and_solve(text)
        self.assertIn("<<24 * 15=360>>", out)
        self.assertEqual(len(inj), 1)
        self.assertEqual(inj[0]["result"], "360")

    def test_bracket_calculation(self):
        text = "Speed is [calc: sqrt(144) + 18 * 3] m/s."
        out, inj = self.cothinker.intercept_and_solve(text)
        self.assertIn("[PRIME-Net: sqrt(144) + 18 * 3 = 66]", out)
        self.assertEqual(len(inj), 1)

    def test_physical_invariant_verification(self):
        res = self.cothinker.verify_physical_invariant("E = 0.5 * m * v**2")
        self.assertEqual(res["status"], "verified")
        self.assertEqual(res["matched"][0]["law"], "kinetic_energy")


if __name__ == "__main__":
    unittest.main()
