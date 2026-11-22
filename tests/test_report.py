"""Check that the report matches its source experiments and seed settings."""

from unittest import TestCase
from unittest.mock import patch

from options_lab import __version__, pricing
from options_lab.server import compute


def reference_fixture(seed):
    return pricing, {"settings": {"seed": seed}}


class ResearchReportTests(TestCase):
    def test_report_reuses_reference_experiments(self):
        with patch("options_lab.server.fitted", side_effect=reference_fixture), \
             patch("options_lab.server.improved", side_effect=reference_fixture), \
             patch("options_lab.server.structured", side_effect=reference_fixture):
            report = compute("/api/report", {"seed": 8, "test_seed": 125})
            price = compute("/api/price", {"seed": 8})
            stress = compute("/api/stress", {"seed": 8})
        self.assertEqual(report["app_version"], __version__)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["pricing"]["seed"], 8)
        self.assertEqual(report["pricing"]["metrics"], price["metrics"])
        self.assertEqual(report["robustness"]["test_seed"], 125)
        self.assertEqual(report["robustness"]["samples_per_regime"], 512)
        self.assertEqual([run["seed"] for run in report["robustness"]["runs"]], [8, 9, 10])
        stress_summary = report["hedge_stress"]
        self.assertEqual(stress_summary["settings"], stress["settings"])
        self.assertEqual(len(stress_summary["scenarios"]), 6)
        for summarized, full in zip(stress_summary["scenarios"], stress["scenarios"]):
            self.assertEqual(summarized["metrics"], full["metrics"])
            self.assertNotIn("pnl", summarized)
            self.assertNotIn("terminal_spot", summarized)

    def test_invalid_report_seed_fails_before_training(self):
        with patch("options_lab.server.fitted") as fitted:
            with self.assertRaises(ValueError):
                compute("/api/report", {"seed": 999_998})
            fitted.assert_not_called()
