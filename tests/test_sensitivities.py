"""Verify the added Greeks using independent price and risk bumps."""

from unittest import TestCase

import numpy as np

from options_lab import experiments, pricing, sensitivities
from options_lab.mixture import MixturePricer
from options_lab.neural import NeuralPricer
from options_lab.server import valuation


class ExtendedGreekTests(TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(52)
        cls.neural = NeuralPricer(
            tuple(rng.normal(0, .5, size=shape) for shape in ((2, 5), (5, 4), (4, 1))),
            tuple(rng.normal(0, .1, size=width) for width in (5, 4, 1)),
        )
        cls.mixture = MixturePricer(
            [.2, .5, .3], [[.5, .9, 1.5], [.2, .05, 0], [0, .1, .2], [.05, 0, .4]]
        )

    def test_closed_form_against_price_and_risk_bumps(self):
        spot = np.array([84., 100., 118.])[None, :]
        time = np.array([.025, .37, 1.1])[:, None]
        vol = np.array([.17, .29, .47])
        for engine in (pricing, self.neural, self.mixture):
            for kind in ("call", "put"):
                with self.subTest(engine=type(engine), kind=kind):
                    contract = (spot, 100, time, vol, -.017, .035, kind)
                    actual = sensitivities.extended_greeks(
                        engine, contract, engine.price(*contract), engine.greeks(*contract)
                    )
                    h = 1e-5
                    time_up = engine.price(spot, 100, time + h, vol, -.017, .035, kind)
                    time_down = engine.price(spot, 100, time - h, vol, -.017, .035, kind)
                    rate_up = engine.price(spot, 100, time, vol, -.017 + h, .035, kind)
                    rate_down = engine.price(spot, 100, time, vol, -.017 - h, .035, kind)
                    vol_up = engine.greeks(spot, 100, time, vol + h, -.017, .035, kind)
                    vol_down = engine.greeks(spot, 100, time, vol - h, -.017, .035, kind)
                    np.testing.assert_allclose(actual["theta"], (time_down - time_up) / (730 * h), atol=1e-8)
                    np.testing.assert_allclose(actual["rho"], (rate_up - rate_down) / (200 * h), atol=1e-8)
                    np.testing.assert_allclose(actual["vanna"], (vol_up["delta"] - vol_down["delta"]) / (200 * h),
                                               atol=1e-8)
                    np.testing.assert_allclose(actual["vomma"], (vol_up["vega"] - vol_down["vega"]) / (200 * h),
                                               atol=1e-8)

    def test_put_call_parity_and_calendar_sign(self):
        spot, strike, years, vol, rate, dividend = 100., 100., 1., .2, .05, .025
        expected_rho = years * strike * np.exp(-rate * years) / 100
        expected_theta = (dividend * spot * np.exp(-dividend * years)
                          - rate * strike * np.exp(-rate * years)) / 365
        for engine in (pricing, self.neural, self.mixture):
            call, put = [
                sensitivities.extended_greeks(engine, inputs, engine.price(*inputs), engine.greeks(*inputs))
                for inputs in ((spot, strike, years, vol, rate, dividend, kind) for kind in ("call", "put"))
            ]
            self.assertAlmostEqual(call["rho"] - put["rho"], expected_rho)
            self.assertAlmostEqual(call["theta"] - put["theta"], expected_theta)
            self.assertEqual(call["vanna"], put["vanna"])
            self.assertEqual(call["vomma"], put["vomma"])
        self.assertLess(sensitivities.extended_greeks(
            pricing, (spot, strike, years, vol, rate, dividend, "call"),
            pricing.price(spot, strike, years, vol, rate, dividend),
            pricing.greeks(spot, strike, years, vol, rate, dividend),
        )["theta"], 0)

    def test_valuation_and_held_out_contracts_include_the_new_greeks(self):
        response = valuation({}, self.neural, comparison=self.neural, mixture=self.mixture)
        expected = {"delta", "gamma", "vega", "theta", "rho", "vanna", "vomma"}
        for row in response["metrics"]:
            series = response["greek_curves"][row["method"]]
            self.assertEqual(set(series), expected)
            for key, values in series.items():
                self.assertEqual(len(values), 101)
                self.assertTrue(np.isfinite(values).all())
                self.assertAlmostEqual(row[key], values[50])

        result = experiments.evaluate(self.mixture, seed=42, samples=128, extended=True)
        self.assertEqual(len(result["metrics"]), 24)
        self.assertEqual(len(result["extended_metrics"]), 24)
        self.assertEqual({row["metric"] for row in result["extended_metrics"]}, {"theta", "rho", "vanna", "vomma"})
        self.assertEqual({row["samples"] for row in result["extended_metrics"] if row["slice"] == "All held-out"}, {128})
