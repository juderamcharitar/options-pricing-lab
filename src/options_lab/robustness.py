"""Matched synthetic contract suites for comparing fitted models across seeds."""

import numpy as np

from . import experiments, pricing
from .neural import X_LIMIT, W_MAX, W_MIN, in_domain


def contract_suite(seed, samples):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 19]))
    short_years = rng.uniform(1, 7, samples) / 365
    tiny_years = rng.uniform(1, 14, samples) / 365
    designs = [
        ("core", "In-domain", rng.uniform(-.65, .65, samples), rng.uniform(.07, .85, samples),
         rng.uniform(.25, 2.5, samples)),
        ("short_atm", "Near expiry, near ATM", rng.uniform(-.012, .012, samples),
         rng.uniform(.12, .60, samples) * np.sqrt(short_years), short_years),
        ("deep_wings", "Deep moneyness", rng.choice([-1, 1], samples) * rng.uniform(X_LIMIT + .05, X_LIMIT + .55, samples),
         rng.uniform(.09, .65, samples), rng.uniform(.15, 2.5, samples)),
        ("high_vol", "High total volatility", rng.uniform(-.55, .55, samples),
         rng.uniform(W_MAX + .05, W_MAX + .60, samples), rng.uniform(.75, 3, samples)),
        ("low_vol", "Low total volatility", rng.uniform(-.008, .008, samples),
         rng.uniform(W_MIN * .10, W_MIN * .90, samples), tiny_years),
    ]
    for key, label, log_moneyness, total_vol, years in designs:
        strike = rng.uniform(60, 150, samples)
        rate = rng.uniform(-.02, .08, samples)
        dividend = rng.uniform(0, .05, samples)
        spot = strike * np.exp(log_moneyness - (rate - dividend) * years)
        inputs = (spot, strike, years, total_vol / np.sqrt(years), rate, dividend)
        _, x, w, _ = pricing.normal_inputs(*inputs)
        yield {"key": key, "label": label, "contracts": samples,
               "in_domain": int(in_domain(x, w).sum())}, inputs


def evaluate_seeds(model_runs, *, test_seed, samples):
    """Evaluate every fitted model on each shared regime for both option types."""
    suite = list(contract_suite(test_seed, samples))
    all_rows = {"all": np.ones(samples, dtype=bool)}
    runs = []
    for seed, models in model_runs:
        model_reports = {}
        for name, (model, training) in models.items():
            regimes = {}
            for regime, inputs in suite:
                result = experiments.evaluate(model, test_seed, samples=samples,
                                              contracts=(inputs, all_rows), extended=True)
                metrics = result.pop("metrics") + result.pop("extended_metrics")
                regimes[regime["key"]] = {
                    "errors": {kind: {row["metric"]: {key: row[key] for key in ("mae", "rmse", "p95_absolute_error")}
                                      for row in metrics if row["option"] == kind}
                               for kind in ("call", "put")},
                    **result,
                }
            model_reports[name] = {"training": training["settings"], "regimes": regimes}
        runs.append({"seed": seed, "models": model_reports})
    return {"regimes": [regime for regime, _ in suite], "test_seed": test_seed, "samples_per_regime": samples,
            "training_domain": {"absolute_log_forward_limit": X_LIMIT, "minimum_total_volatility": W_MIN,
                                "maximum_total_volatility": W_MAX}, "runs": runs}
