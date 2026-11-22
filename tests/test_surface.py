"""Checks for synthetic quotes, implied-volatility recovery, and input handling."""

import json
from math import exp
from unittest import TestCase
from unittest.mock import patch

from options_lab import pricing
from options_lab.server import compute


class SurfaceTests(TestCase):
    def test_default_grid_recovers_and_reprices_quotes(self):
        with patch("options_lab.server.fitted", side_effect=AssertionError("Unexpected neural fitting")):
            surface = compute("/api/surface", {})
        self.assertEqual(surface["summary"]["contracts"], 77)
        self.assertEqual(surface["summary"]["recovered"], 77)
        self.assertLess(surface["summary"]["max_absolute_error_vol_points"], 1e-8)
        for expiry in surface["expiries"]:
            self.assertEqual(len(expiry["quotes"]), 11)
            for quote in expiry["quotes"]:
                self.assertAlmostEqual(quote["strike"] / expiry["forward"] * 100, quote["strike_pct_forward"])
                self.assertEqual(quote["kind"], "put" if quote["strike_pct_forward"] < 100 else "call")
                repriced = pricing.price(100, quote["strike"], expiry["days"] / 365,
                                         quote["implied_vol_pct"] / 100, .04, 0, quote["kind"])
                self.assertAlmostEqual(repriced, quote["premium"], places=9)
        json.dumps(surface, allow_nan=False)

    def test_shape_controls_and_forward_carry(self):
        result = compute("/api/surface", {"spot": 120, "rate": -2, "dividend": 3, "atm_vol": 30,
                                           "skew": -4, "smile": 3, "term": 2})
        expiries = {expiry["days"]: expiry for expiry in result["expiries"]}
        for days, wing_values in ((90, (37, 30, 29)), (180, (39, 32, 31))):
            expiry = expiries[days]
            self.assertAlmostEqual(expiry["forward"], 120 * exp(-.05 * days / 365))
            for index, expected in zip((0, 5, 10), wing_values):
                self.assertAlmostEqual(expiry["quotes"][index]["implied_vol_pct"], expected, places=8)

    def test_low_time_value_is_flagged_and_atm_still_recovers(self):
        surface = compute("/api/surface", {"atm_vol": 1, "skew": 0, "smile": 0, "term": 0})
        front = surface["expiries"][0]["quotes"]
        self.assertEqual(front[0]["status"], "low_time_value")
        self.assertIsNone(front[0]["implied_vol_pct"])
        self.assertIsNone(front[0]["absolute_error_vol_points"])
        self.assertAlmostEqual(front[5]["implied_vol_pct"], 1, places=8)
        self.assertGreater(surface["summary"]["recovered"], 0)
        self.assertLess(surface["summary"]["recovered"], 77)

    def test_surface_rejects_invalid_parameters_and_grid_values(self):
        for body in ({"spot": 0}, {"atm_vol": True}, {"term": 11}, {"rate": float("nan")}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                compute("/api/surface", body)
        with self.assertRaisesRegex(ValueError, r"14 days and 75% of forward"):
            compute("/api/surface", {"atm_vol": 3, "skew": 20, "smile": 0, "term": 0})
