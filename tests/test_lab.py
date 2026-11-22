"""Regression checks for pricing, model derivatives, and fair hedge accounting."""

import unittest

import numpy as np

from options_lab import experiments, pricing
from options_lab.differential import fine_tune, objective
from options_lab.neural import NeuralPricer, scaled_features, train_pricer
from options_lab.server import compute, valuation


class PricingTests(unittest.TestCase):
    def test_reference_atm_contract(self):
        self.assertAlmostEqual(pricing.price(100, 100, 1, .2, .05), 10.4505835722, places=9)
        self.assertAlmostEqual(pricing.price(100, 100, 1, .2, .05, kind="put"), 5.5735260223, places=9)

    def test_dividend_adjusted_parity_and_expiry(self):
        s = np.array([80., 100., 120.])
        params = (s, 100, .75, .25, -.01, .03)
        np.testing.assert_allclose(pricing.price(*params) - pricing.price(*params, "put"),
                                   s * np.exp(-.03 * .75) - 100 * np.exp(.01 * .75), atol=1e-12)
        np.testing.assert_allclose(pricing.price(s, 100, 0, .25, kind="put"), [20, 0, 0], atol=1e-12)
        self.assertAlmostEqual(pricing.price(100, 90, 1, 0, .05), 100 - 90 * np.exp(-.05))

    def test_implied_volatility_roundtrip_and_bounds(self):
        for kind in ("call", "put"):
            inputs = dict(spot=93, strike=100, years=.75, rate=.03, dividend=.01, kind=kind)
            premium = pricing.price(**inputs, vol=.31)
            self.assertAlmostEqual(pricing.implied_volatility(premium, **inputs), .31, places=10)
        with self.assertRaises(ValueError):
            pricing.implied_volatility(200, 100, 100, 1)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            pricing.price(-100, 100, 1, .2)
        with self.assertRaises(ValueError):
            pricing.greeks(100, 100, 0, .2)
        with self.assertRaises(ValueError):
            compute("/api/hedge", {"dividend": 2})


class FineTuningGradientTests(unittest.TestCase):
    def test_joint_objective_parameter_gradient(self):
        rng = np.random.default_rng(12)
        model = NeuralPricer(tuple(rng.normal(0, .15, size) for size in [(2, 3), (3, 2), (2, 1)]),
                             (np.zeros(3), np.zeros(2), np.array([-.3])))
        x = np.array([-.2, -.01, .1, .4])
        w = np.array([.02, .06, .15, .4])
        features = scaled_features(x, w)
        targets = np.vstack((pricing.forward_price(x, w), np.array([.1, .4, .8, .9])))
        bounds = np.vstack((np.maximum(np.expm1(x), 0), np.exp(x)))
        args = features, np.exp(-x), targets, bounds, .3, 20.
        _, gradients = objective(model, *args)
        bump = 1e-6
        for parameter, gradient in zip((*model.weights, *model.biases), gradients):
            for index in np.ndindex(parameter.shape):
                value = parameter[index]
                parameter[index] = value + bump
                plus = objective(model, *args)[0]
                parameter[index] = value - bump
                minus = objective(model, *args)[0]
                parameter[index] = value
                self.assertAlmostEqual(gradient[index], (plus - minus) / (2 * bump), places=7)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _ = train_pricer()
        cls.original_weights = tuple(array.copy() for array in (*cls.model.weights, *cls.model.biases))
        cls.tuned, _ = fine_tune(cls.model)

    def test_benchmark_and_neural_derivatives_match_finite_differences(self):
        for engine in (pricing, self.model, self.tuned):
            for kind in ("call", "put"):
                args = dict(spot=106., strike=100, years=.52, vol=.24, rate=.03, dividend=.01, kind=kind)
                with self.subTest(engine=type(engine), kind=kind):
                    result = engine.greeks(**args)
                    delta_step = .01
                    p1 = engine.price(**{**args, "spot": args["spot"] + delta_step})
                    p0 = engine.price(**args)
                    p2 = engine.price(**{**args, "spot": args["spot"] - delta_step})
                    self.assertAlmostEqual(result["delta"], (p1 - p2) / (2 * delta_step), places=6)
                    self.assertAlmostEqual(result["gamma"], (p1 + p2 - 2 * p0) / delta_step**2, places=6)
                    vol_step = .00001
                    v1 = engine.price(**{**args, "vol": args["vol"] + vol_step})
                    v2 = engine.price(**{**args, "vol": args["vol"] - vol_step})
                    self.assertAlmostEqual(result["vega"], (v1 - v2) / (200 * vol_step), places=7)

    def test_held_out_evaluation_reports_all_methods_and_error_slices(self):
        results = experiments.evaluate(self.model, 42, samples=1024)
        rows = results["metrics"]
        self.assertEqual(len(rows), 24)
        call_price = rows[0]
        self.assertEqual((call_price["option"], call_price["metric"]), ("call", "price"))
        self.assertLess(call_price["mae"], .65)
        self.assertLessEqual(results["price_bound_violations"], 1024)
        self.assertEqual(len(experiments.benchmark(self.model, repeats=3)), 4)

    def test_valuation_returns_implied_vol_and_out_of_range_flag(self):
        output = valuation({"spot": 300, "strike": 100}, self.model)
        self.assertFalse(output["in_domain"])
        premium = float(pricing.price(100, 100, 90 / 365, .32, .04))
        output = valuation({"premium": premium}, self.model)
        self.assertAlmostEqual(output["implied_vol_percent"], 32)
        self.assertEqual(len(output["curve"]["spot"]), 101)

    def test_matched_hedges_and_transaction_costs(self):
        inputs = dict(spot=100., strike=100., years=.5, vol=.25, rate=.03,
                      kind="put", steps=12, paths=200, seed=7)
        free = experiments.hedge(pricing, **inputs, cost_bps=0)
        costly = experiments.hedge(pricing, **inputs, cost_bps=20)
        self.assertEqual(free["difference_rmse"], 0)
        self.assertEqual(costly["difference_rmse"], 0)
        self.assertEqual(free["metrics"][0]["mean_cost"], 0)
        self.assertGreater(costly["metrics"][0]["mean_cost"], 0)
        self.assertTrue(np.all(np.array(costly["pnl"]["Black–Scholes"]) <= free["pnl"]["Black–Scholes"]))
        learned = experiments.hedge(self.model, **inputs, cost_bps=20)
        self.assertEqual(learned["pnl"]["Black–Scholes"], costly["pnl"]["Black–Scholes"])
        self.assertGreater(learned["difference_rmse"], 0)


    def test_fine_tuning_keeps_baseline_and_improves_default_holdout(self):
        for actual, expected in zip((*self.model.weights, *self.model.biases), self.original_weights):
            np.testing.assert_array_equal(actual, expected)
        results = experiments.compare_models(self.model, self.tuned, seed=42)
        baseline = results[experiments.BASELINE]
        tuned = results[experiments.FINE_TUNED]
        self.assertEqual([r["samples"] for r in baseline["metrics"]], [r["samples"] for r in tuned["metrics"]])
        self.assertLess(tuned["price_bound_violations"], baseline["price_bound_violations"])
        self.assertLess(tuned["metrics"][1]["mae"], baseline["metrics"][1]["mae"])

    def test_comparison_preserves_paired_paths_and_baseline(self):
        inputs = dict(spot=100., strike=100., years=90 / 365, vol=.25, rate=.04,
                      kind="call", steps=30, paths=1000, cost_bps=5, seed=42)
        original = experiments.hedge(self.model, **inputs)
        comparison = experiments.hedge(self.model, **inputs, comparison=self.tuned)
        np.testing.assert_array_equal(comparison["pnl"][experiments.BLACK_SCHOLES],
                                      original["pnl"][experiments.BLACK_SCHOLES])
        np.testing.assert_array_equal(comparison["pnl"][experiments.BASELINE], original["pnl"]["Neural network"])
        self.assertEqual(len(comparison["pnl"][experiments.FINE_TUNED]), inputs["paths"])
        self.assertLess(comparison["metrics"][2]["rmse"], comparison["metrics"][1]["rmse"])
