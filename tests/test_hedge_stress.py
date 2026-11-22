"""Regression tests for matched shocks, funding intervals, and stress metrics."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from options_lab import experiments, pricing
from options_lab.server import compute


class HedgeStressTests(TestCase):
    settings = dict(spot=100, strike=105, years=90 / 365, vol=.25, rate=.04,
                    kind="put", steps=7, paths=101, cost_bps=5, seed=42)

    def test_scenarios_pair_paths_and_use_same_opening_premium(self):
        result = experiments.hedge_stress(pricing, **self.settings, stress_vol=.40,
                                          stress_cost_bps=25, spot_jump=-.10, coarse_every=3)
        scenarios = {row["key"]: row for row in result["scenarios"]}
        self.assertEqual(list(scenarios), ["base", "volatility", "jump", "costs", "frequency", "combined"])
        base = scenarios["base"]
        standard = experiments.hedge(pricing, **self.settings)
        self.assertEqual(base["pnl"], standard["pnl"])
        self.assertEqual(result["initial_premium"], standard["initial_premium"])
        self.assertEqual(scenarios["jump"]["jump_step"], 4)
        self.assertEqual(scenarios["frequency"]["hedge_dates"], 3)
        self.assertEqual(scenarios["costs"]["terminal_spot"], base["terminal_spot"])
        self.assertEqual(scenarios["frequency"]["terminal_spot"], base["terminal_spot"])
        np.testing.assert_allclose(scenarios["jump"]["terminal_spot"], np.asarray(base["terminal_spot"]) * .9)
        np.testing.assert_allclose(scenarios["combined"]["terminal_spot"],
                                   np.asarray(scenarios["volatility"]["terminal_spot"]) * .9)

        model = experiments.BLACK_SCHOLES
        base_pnl = np.asarray(base["pnl"][model])
        costly_pnl = np.asarray(scenarios["costs"]["pnl"][model])
        self.assertTrue(np.all(costly_pnl <= base_pnl + 1e-10))
        self.assertEqual(result["tail_count"], 6)
        combined = scenarios["combined"]
        combined_pnl = np.asarray(combined["pnl"][model])
        stats = combined["metrics"][0]
        self.assertAlmostEqual(stats["worst_5pct_mean_pnl"], np.sort(combined_pnl)[:6].mean())
        self.assertAlmostEqual(stats["mean_change_from_base"], (combined_pnl - base_pnl).mean())

    def test_neutral_shocks_reproduce_baseline(self):
        result = experiments.hedge_stress(pricing, **self.settings, stress_vol=.25,
                                          stress_cost_bps=5, spot_jump=0, coarse_every=1)
        base = result["scenarios"][0]
        for scenario in result["scenarios"]:
            self.assertEqual(scenario["pnl"], base["pnl"])
            self.assertIsNone(scenario["jump_step"])

    def test_coarse_hedge_finances_unequal_intervals(self):
        # Five stock intervals with trades at time zero and at the third node.
        stock = np.array([[100.], [101.], [98.], [108.], [104.], [106.]])
        strike, years, vol, rate, kind = 100, .5, .25, .08, "call"
        premium = float(pricing.price(stock[0, 0], strike, years, vol, rate))
        result = experiments._hedge_paths(experiments.pricing_engines(pricing), stock, strike=strike,
                                           years=years, vol=vol, rate=rate, kind=kind, cost_bps=12,
                                           premium=premium, rebalance_every=3)
        opening_delta = float(pricing.greeks(100, strike, .5, vol, rate)["delta"])
        later_delta = float(pricing.greeks(108, strike, .2, vol, rate)["delta"])
        entry = opening_delta * 100
        trade = (later_delta - opening_delta) * 108
        exit_sale = later_delta * 106
        first_cash = (premium - entry - abs(entry) * .0012) * np.exp(rate * .3)
        final_cash = (first_cash - trade - abs(trade) * .0012) * np.exp(rate * .2)
        expected = final_cash + exit_sale - abs(exit_sale) * .0012 - 6
        self.assertAlmostEqual(result["pnl"][experiments.BLACK_SCHOLES][0], expected)

    def test_stress_validates_inputs_before_fitting(self):
        with patch("options_lab.server.fitted") as fitted:
            for body in ({"spot_jump": -100}, {"stress_cost_bps": 3}, {"coarse_every": 0},
                         {"paths": 2501}, {"dividend": 2}):
                with self.subTest(body=body), self.assertRaises(ValueError):
                    compute("/api/stress", body)
            fitted.assert_not_called()
