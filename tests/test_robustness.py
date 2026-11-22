"""Check that robustness studies share a held-out dataset across fitted models."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from options_lab import pricing, robustness
from options_lab.server import compute


class ShiftedPrice:
    greeks = staticmethod(pricing.greeks)

    @staticmethod
    def price(*args, **kwargs):
        return pricing.price(*args, **kwargs) + 2


class RobustnessTests(TestCase):
    def test_suite_respects_regime_boundaries_and_is_reproducible(self):
        suite = list(robustness.contract_suite(42, 128))
        repeated = list(robustness.contract_suite(42, 128))
        changed = list(robustness.contract_suite(43, 128))
        self.assertEqual([meta["key"] for meta, _ in suite],
                         ["core", "short_atm", "deep_wings", "high_vol", "low_vol"])
        self.assertEqual([meta["in_domain"] for meta, _ in suite], [128, 128, 0, 0, 0])
        for (meta, contract), (_, copy), (_, other) in zip(suite, repeated, changed):
            with self.subTest(regime=meta["key"]):
                np.testing.assert_array_equal(contract, copy)
                self.assertFalse(np.array_equal(contract, other))
        self.assertLessEqual(np.max(suite[1][1][2]) * 365, 7)

    def test_report_separates_seed_and_regime_and_uses_reference_errors(self):
        runs = [(11, {"test": (pricing, {"settings": {"seed": 11}})}),
                (12, {"test": (ShiftedPrice(), {"settings": {"seed": 12}})})]
        result = robustness.evaluate_seeds(runs, test_seed=102, samples=64)
        self.assertEqual(result["test_seed"], 102)
        self.assertEqual([run["seed"] for run in result["runs"]], [11, 12])
        for regime in result["regimes"]:
            exact, shifted = [run["models"]["test"]["regimes"][regime["key"]] for run in result["runs"]]
            self.assertEqual(exact["samples"], 64)
            self.assertEqual(exact["price_bound_violations"], 0)
            for kind in ("call", "put"):
                self.assertEqual(exact["errors"][kind]["price"]["mae"], 0)
                self.assertAlmostEqual(shifted["errors"][kind]["price"]["rmse"], 2)
                self.assertEqual(shifted["errors"][kind]["delta"]["mae"], 0)
                self.assertIn("vomma", shifted["errors"][kind])

    def test_rejects_out_of_range_requests_before_training(self):
        with patch("options_lab.server.fitted") as fitted:
            for body in ({"training_runs": 1}, {"seed": 999_998}, {"samples_per_regime": 4096},
                         {"test_seed": -1}):
                with self.subTest(body=body), self.assertRaises(ValueError):
                    compute("/api/robustness", body)
            fitted.assert_not_called()
