"""Structural and numerical regression checks for the constrained pricer."""

import unittest

import numpy as np

from options_lab import experiments, pricing
from options_lab.mixture import MixturePricer, clock_features, mixture_objective, train_mixture


class MixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MixturePricer(
            [.2, .5, .3], [[.5, .9, 1.5], [.2, .05, 0], [0, .1, .2], [.05, 0, .4]]
        )

    def test_bounds_and_parity_with_distinct_clocks(self):
        s = np.geomspace(5, 2000, 201)[None, :]
        t = np.array([.001, .25, 3.])[:, None]
        q = .025
        df_spot = s * np.exp(-q * t)
        df_strike = 100 * np.exp(.02 * t)
        call = self.model.price(s, 100, t, .95, -.02, q)
        put = self.model.price(s, 100, t, .95, -.02, q, "put")
        for price, lower, upper in ((call, df_spot - df_strike, df_spot), (put, df_strike - df_spot, df_strike)):
            self.assertTrue(np.all(price >= np.maximum(lower, 0) - 1e-10))
            self.assertTrue(np.all(price <= upper + 1e-10))
        np.testing.assert_allclose(call - put, df_spot - df_strike, rtol=1e-13, atol=1e-11)
        np.testing.assert_allclose(self.model.price([90, 110], 100, 0, .5), [0, 10], atol=1e-12)

    def test_strike_and_maturity_shape_and_greek_signs(self):
        checks = experiments.grid_checks(self.model)
        for name in ("strike_slope", "strike_convexity", "forward_maturity"):
            self.assertGreater(checks[name + "_checks"], 7000)
            self.assertEqual(checks[name + "_violations"], 0)
        for kind in ("call", "put"):
            greeks = self.model.greeks(np.geomspace(10, 500, 250), 100, .25, .2, kind=kind)
            self.assertTrue(np.all(greeks["gamma"] >= 0))
            self.assertTrue(np.all(greeks["vega"] >= 0))

    def test_price_greeks_match_finite_differences(self):
        for kind in ("call", "put"):
            s, k, t, vol = 102., 100., .4, .3
            b, v = .01, 1e-5
            def price(spot, volatility):
                return self.model.price(spot, k, t, volatility, .03, .02, kind)
            result = self.model.greeks(s, k, t, vol, .03, .02, kind)
            plus, mid, minus = price(s + b, vol), price(s, vol), price(s - b, vol)
            self.assertAlmostEqual(result["delta"], (plus - minus) / (2 * b), places=6)
            self.assertAlmostEqual(result["gamma"], (plus - 2 * mid + minus) / b**2, places=6)
            self.assertAlmostEqual(result["vega"], (price(s, vol + v) - price(s, vol - v)) / (200 * v), places=7)

    def test_training_gradient(self):
        params = np.r_[[-.2, .15, .1], self.model.clock_weights.ravel()]
        x = np.linspace(-.5, .5, 6)
        w = np.linspace(.015, .85, 6)
        features, _ = clock_features(w)
        targets = pricing.forward_price(x, w), pricing.greeks(np.exp(x), 1., 1., w)["delta"]
        _, expected = mixture_objective(params, features, x, *targets)
        for i in range(len(params)):
            plus, minus = params.copy(), params.copy()
            plus[i] += 1e-6
            minus[i] -= 1e-6
            slope = (mixture_objective(plus, features, x, *targets)[0]
                     - mixture_objective(minus, features, x, *targets)[0]) / 2e-6
            self.assertAlmostEqual(expected[i], slope, places=8)

    def test_fitted_model_recovers_reference_and_tracks_paired_hedge(self):
        model, training = train_mixture(seed=42)
        self.assertLess(training["loss"][-1], training["loss"][0])
        report = experiments.evaluate(model, seed=42)
        self.assertLess(report["metrics"][0]["mae"], .005)
        for key in ("price_bound_violations", "negative_gamma_rows", "negative_vega_rows"):
            self.assertEqual(report[key], 0)
        result = experiments.hedge(model, spot=100., strike=100., years=90 / 365, vol=.25,
                                   rate=.04, kind="call", steps=30, paths=1000, cost_bps=5, seed=42)
        self.assertLess(result["difference_rmse"], .005)
