"""Regression checks for numerical pricing, uncertainty, and resolution sweeps."""

from math import exp, sqrt
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from options_lab import numerical, pricing
from options_lab.server import compute


class NumericalPricingTests(TestCase):
    def test_one_step_tree_and_put_call_parity(self):
        spot, strike, vol, rate, dividend = 100, 100, .2, .05, .03
        up, down = exp(vol), exp(-vol)
        probability = (exp(rate - dividend) - down) / (up - down)
        args = (spot, strike, 1, vol, rate, dividend)
        self.assertAlmostEqual(numerical.crr_price(*args, steps=1),
                               exp(-rate) * probability * (spot * up - strike))
        for steps in (16, 512):
            call = numerical.crr_price(*args, steps=steps)
            put = numerical.crr_price(*args, kind="put", steps=steps)
            self.assertAlmostEqual(call - put, spot * exp(-dividend) - strike * exp(-rate), places=9)

    def test_tree_converges_with_dividends_and_negative_rates(self):
        for kind in ("call", "put"):
            args = (94, 100, .9, .24, -.02, .04, kind)
            self.assertAlmostEqual(numerical.crr_price(*args, steps=2048), pricing.price(*args), delta=.006)

    def test_discounted_deterministic_contracts(self):
        for years, vol in ((0, .2), (.5, 0)):
            for kind in ("call", "put"):
                args = (96, 100, years, vol, -.01, .06, kind)
                expected = pricing.price(*args)
                self.assertAlmostEqual(numerical.crr_price(*args, steps=64), expected)
                for antithetic in (False, True):
                    result = numerical.monte_carlo(*args, paths=256, seed=42, antithetic=antithetic)
                    self.assertAlmostEqual(result["price"], expected)
                    self.assertEqual(result["ci_lower"], result["ci_upper"])
                    self.assertEqual(result["uncertainty_status"], "deterministic")

    def test_monte_carlo_uses_pairs_for_standard_error(self):
        budget, seed = 1024, 8
        first_stream, second_stream = np.random.SeedSequence(seed).spawn(2)
        z = np.random.default_rng(first_stream).standard_normal(budget // 2)
        independent_z = np.random.default_rng(second_stream).standard_normal(budget // 2)

        def discounted_payoffs(shocks):
            return exp(-.04) * np.maximum(100 * np.exp(.04 - .2**2 / 2 + .2 * shocks) - 100, 0)

        plain = np.concatenate((discounted_payoffs(z), discounted_payoffs(independent_z)))
        pairs = (discounted_payoffs(z) + discounted_payoffs(-z)) / 2
        for antithetic, observations in ((False, plain), (True, pairs)):
            result = numerical.monte_carlo(100, 100, 1, .2, .04, paths=budget, seed=seed, antithetic=antithetic)
            expected_se = observations.std(ddof=1) / sqrt(len(observations))
            self.assertEqual(result["independent_observations"], len(observations))
            self.assertAlmostEqual(result["price"], observations.mean())
            self.assertAlmostEqual(result["standard_error"], expected_se)
            self.assertAlmostEqual(result["ci_upper"] - result["ci_lower"], 2 * 1.95996398454 * expected_se)
        self.assertLess(pairs.std(ddof=1) / sqrt(len(pairs)), plain.std(ddof=1) / sqrt(len(plain)))

    def test_reproducibility_and_reference_accuracy(self):
        args = (100, 100, 90 / 365, .25, .04)
        for antithetic in (False, True):
            settings = dict(paths=131072, seed=42, antithetic=antithetic)
            estimate = numerical.monte_carlo(*args, **settings)
            self.assertEqual(estimate, numerical.monte_carlo(*args, **settings))
            self.assertAlmostEqual(estimate["price"], pricing.price(*args), delta=.07)

    def test_no_confidence_interval_when_no_payoffs_are_observed(self):
        result = numerical.monte_carlo(20, 200, .1, .05, paths=256, seed=42)
        self.assertEqual(result["price"], 0)
        self.assertEqual(result["positive_observations"], 0)
        self.assertIsNone(result["standard_error"])
        self.assertIsNone(result["ci_upper"])
        self.assertEqual(result["uncertainty_status"], "insufficient_sample")

    def test_api_keeps_valid_budgets_and_reports_invalid_coarse_tree(self):
        with patch("options_lab.server.fitted", side_effect=AssertionError("Unexpected neural fitting")):
            result = compute("/api/numerical", {
                "rate": 10, "vol": 2, "days": 365, "tree_steps": 70, "mc_paths": 600, "kind": "put"
            })
        self.assertEqual([row["steps"] for row in result["tree"]], [8, 16, 32, 64, 70])
        self.assertIsNotNone(result["tree"][0]["reason"])
        self.assertIsNone(result["tree"][0]["price"])
        self.assertIsNone(result["tree"][-1]["reason"])
        self.assertGreaterEqual(result["tree"][-1]["price"], 0)
        self.assertEqual([row["paths"] for row in result["monte_carlo"]], [256, 512, 600])
        for settings in ({"mc_paths": 257}, {"tree_steps": 2049}, {"tree_steps": 8.5}, {"seed": True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                compute("/api/numerical", settings)
