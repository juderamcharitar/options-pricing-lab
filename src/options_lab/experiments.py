"""Reproducible model evaluation, inference timing, and discrete delta hedging."""

from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from . import pricing, sensitivities
from .neural import X_LIMIT, W_MIN, W_MAX, in_domain

BLACK_SCHOLES = "Black–Scholes"
BASELINE = "Price-only network"
FINE_TUNED = "Price + delta network"
MIXTURE = "Constrained mixture"


def errors(actual, expected):
    difference = np.asarray(actual) - expected
    return {"mae": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "p95_absolute_error": float(np.quantile(np.abs(difference), .95))}


def held_out_contracts(seed, samples):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    x = rng.uniform(-X_LIMIT, X_LIMIT, samples)
    w = rng.uniform(W_MIN, W_MAX, samples)
    count = samples // 4
    x[:count] = rng.uniform(-.09, .09, count)
    w[:count] = rng.uniform(W_MIN, .12, count)
    strike = rng.uniform(60, 150, samples)
    years = rng.uniform(.08, 2, samples)
    rate = rng.uniform(-.02, .08, samples)
    dividend = rng.uniform(0, .05, samples)
    spot = strike * np.exp(x - (rate - dividend) * years)
    inputs = (spot, strike, years, w / np.sqrt(years), rate, dividend)
    sample_names = {"All held-out": np.ones(samples, dtype=bool), "Low total volatility (≤ 0.12)": w <= .12,
                    "Near forward ATM (|log F/K| ≤ 0.09)": np.abs(x) <= .09}
    return inputs, sample_names


def _metric_rows(actual, expected, kind, sample_names):
    rows = []
    for label, mask in sample_names.items():
        count = int(mask.sum())
        for metric in expected:
            rows.append({"option": kind, "slice": label, "metric": metric, "samples": count,
                         **(errors(actual[metric][mask], expected[metric][mask]) if count else
                            dict.fromkeys(("mae", "rmse", "p95_absolute_error")))})
    return rows


def evaluate(model, seed, samples=4096, *, contracts=None, extended=False):
    inputs, sample_names = held_out_contracts(seed, samples) if contracts is None else contracts
    spot, strike, years, _, rate, dividend = inputs
    rows = []
    advanced_rows = []
    for kind in ("call", "put"):
        contract = (*inputs, kind)
        with threadpool_limits(limits=1):
            actual = {"price": model.price(*contract), **model.greeks(*contract)}
            if extended:
                actual_advanced = sensitivities.extended_greeks(model, contract, actual["price"], actual)
        expected = {"price": pricing.price(*contract), **pricing.greeks(*contract)}
        rows.extend(_metric_rows(actual, expected, kind, sample_names))
        if extended:
            expected_advanced = sensitivities.extended_greeks(pricing, contract, expected["price"], expected)
            advanced_rows.extend(_metric_rows(actual_advanced, expected_advanced, kind, sample_names))
    with threadpool_limits(limits=1):
        call = model.price(*inputs)
    parity = spot * np.exp(-dividend * years) - strike * np.exp(-rate * years)
    below_lower = np.maximum(np.maximum(parity, 0) - call, 0)
    above_upper = np.maximum(call - spot * np.exp(-dividend * years), 0)
    bound_error = below_lower + above_upper
    return {"metrics": rows, **({"extended_metrics": advanced_rows} if extended else {}), "samples": samples,
            "price_bound_violations": int((bound_error > 1e-8).sum()),
            "largest_bound_violation": float(bound_error.max()),
            "negative_gamma_rows": int((actual["gamma"] < 0).sum()),
            "negative_vega_rows": int((actual["vega"] < 0).sum())}


def pricing_engines(model, comparison=None, mixture=None):
    engines = [(BLACK_SCHOLES, pricing), ("Neural network" if comparison is None else BASELINE, model)]
    if comparison is not None:
        engines.append((FINE_TUNED, comparison))
    if mixture is not None:
        engines.append((MIXTURE, mixture))
    return engines


def grid_checks(model):
    """Audit European call slices at zero carry, holding the forward fixed at one."""
    strikes = np.linspace(.25, 2.5, 181)
    total_vols = np.geomspace(.005, 1.5, 40)[:, None]
    with threadpool_limits(limits=1):
        prices = model.price(1., strikes, 1., total_vols)
    strike_slopes = np.diff(prices, axis=1) / (strikes[1] - strikes[0])
    return {"strike_slope_violations": int(((strike_slopes > 1e-9) | (strike_slopes < -1 - 1e-9)).sum()),
            "strike_slope_checks": int(strike_slopes.size),
            "strike_convexity_violations": int((np.diff(strike_slopes, axis=1) < -1e-9).sum()),
            "strike_convexity_checks": int(prices.shape[0] * (len(strikes) - 2)),
            "forward_maturity_violations": int((np.diff(prices, axis=0) < -1e-9).sum()),
            "forward_maturity_checks": int((len(total_vols) - 1) * len(strikes))}


def compare_models(baseline, fine_tuned, seed, samples=4096, *, mixture=None):
    contracts = held_out_contracts(seed, samples)
    return {label: {**evaluate(model, seed, samples, contracts=contracts, extended=True), "grid_checks": grid_checks(model)}
            for label, model in pricing_engines(baseline, fine_tuned, mixture)[1:]}


def benchmark(model, seed=42, repeats=21, *, comparison=None, mixture=None):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    rows = []
    models = pricing_engines(model, comparison, mixture)
    with threadpool_limits(limits=1):
        for size in (1, 2048):
            inputs = (rng.uniform(75, 130, size), 100.0, .5, .23, .04, .01)
            for label, engine in models:
                engine.price(*inputs)
                timings = []
                for _ in range(repeats):
                    start = perf_counter()
                    engine.price(*inputs)
                    timings.append((perf_counter() - start) * 1_000_000)
                rows.append({"method": label, "batch": size, "median_microseconds": float(np.median(timings)),
                             "repeats": repeats})
    return rows


def _stock_paths(spot, years, vol, rate, innovations):
    steps, paths = innovations.shape
    dt = years / steps
    log_returns = (rate - vol**2 / 2) * dt + vol * np.sqrt(dt) * innovations
    return np.vstack((np.full(paths, spot), spot * np.exp(np.cumsum(log_returns, axis=0))))


def _hedge_paths(models, prices, *, strike, years, vol, rate, kind, cost_bps, premium, rebalance_every=1):
    steps, paths = prices.shape[0] - 1, prices.shape[1]
    dt = years / steps
    growth = np.exp(rate * dt)
    cost_fraction = cost_bps / 10_000
    hedge_dates = np.arange(0, steps, rebalance_every)
    remaining_path_times = (steps - hedge_dates)[:, None] * dt
    _, x, w, _ = pricing.normal_inputs(prices[hedge_dates], strike, remaining_path_times, vol, rate, 0)
    coverage = float(in_domain(x, w).mean())
    outcomes = {}
    summaries = []
    with threadpool_limits(limits=1):
        for label, engine in models:
            shares = np.zeros(paths)
            cash = np.full(paths, premium)
            costs = np.zeros(paths)
            previous_date = 0
            for i in hedge_dates:
                if i:
                    cash *= growth ** (i - previous_date)
                remaining = (steps - i) * dt
                prices_now = prices[i]
                target_shares = engine.greeks(prices_now, strike, remaining, vol, rate, 0, kind)["delta"]
                trade = (target_shares - shares) * prices_now
                fee = np.abs(trade) * cost_fraction
                cash -= trade + fee
                costs += fee
                shares = target_shares
                previous_date = i
            exit_fee = np.abs(shares) * prices[-1] * cost_fraction
            costs += exit_fee
            pnl = cash * growth ** (steps - previous_date) + shares * prices[-1] - exit_fee - pricing.payoff(prices[-1], strike, kind)
            outcomes[label] = pnl
            paired_rmse = errors(pnl, outcomes[BLACK_SCHOLES])["rmse"]
            summaries.append({"method": label, "mean_pnl": float(pnl.mean()),
                              "rmse": float(np.sqrt(np.mean(pnl**2))),
                              "p05_pnl": float(np.quantile(pnl, .05)),
                              "mean_cost": float(costs.mean()), "paired_difference_rmse": paired_rmse})
    return {"metrics": summaries, "pnl": {label: pnl.tolist() for label, pnl in outcomes.items()},
            "difference_rmse": summaries[1]["paired_difference_rmse"],
            "initial_premium": premium, "neural_domain_coverage": coverage}


def hedge(model, *, spot, strike, years, vol, rate, kind, steps, paths, cost_bps, seed, comparison=None, mixture=None):
    """Short one option per path. All hedges collect the same benchmark premium."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    prices = _stock_paths(spot, years, vol, rate, rng.standard_normal((steps, paths)))
    premium = float(pricing.price(spot, strike, years, vol, rate, 0, kind))
    return _hedge_paths(pricing_engines(model, comparison, mixture), prices, strike=strike, years=years,
                        vol=vol, rate=rate, kind=kind, cost_bps=cost_bps, premium=premium)


def hedge_stress(model, *, spot, strike, years, vol, rate, kind, steps, paths, cost_bps, seed,
                 stress_vol, stress_cost_bps, spot_jump, coarse_every, comparison=None, mixture=None):
    """Pair terminal outcomes across six scenarios driven by common shocks."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    innovations = rng.standard_normal((steps, paths))
    base = _stock_paths(spot, years, vol, rate, innovations)
    vol_stress = _stock_paths(spot, years, stress_vol, rate, innovations)
    jump_step = (steps + 1) // 2
    jump_multiplier = np.ones((steps + 1, 1))
    jump_multiplier[jump_step:] = 1 + spot_jump
    premium = float(pricing.price(spot, strike, years, vol, rate, 0, kind))
    models = pricing_engines(model, comparison, mixture)
    specifications = [
        ("base", "Baseline", base, vol, 0., cost_bps, 1),
        ("volatility", "Volatility mismatch", vol_stress, stress_vol, 0., cost_bps, 1),
        ("jump", "Midpoint jump", base * jump_multiplier, vol, spot_jump, cost_bps, 1),
        ("costs", "Higher trading costs", base, vol, 0., stress_cost_bps, 1),
        ("frequency", "Coarser rebalancing", base, vol, 0., cost_bps, coarse_every),
        ("combined", "Combined stress", vol_stress * jump_multiplier, stress_vol, spot_jump, stress_cost_bps, coarse_every),
    ]
    scenarios = []
    baseline_pnl = {}
    tail_count = (paths + 19) // 20
    for key, label, prices, realized_vol, jump, costs, every in specifications:
        result = _hedge_paths(models, prices, strike=strike, years=years, vol=vol, rate=rate,
                              kind=kind, cost_bps=costs, premium=premium, rebalance_every=every)
        if key == "base":
            baseline_pnl = {name: np.asarray(values) for name, values in result["pnl"].items()}
        for row in result["metrics"]:
            name = row["method"]
            terminal = np.asarray(result["pnl"][name])
            row["worst_5pct_mean_pnl"] = float(np.partition(terminal, tail_count - 1)[:tail_count].mean())
            row["mean_change_from_base"] = float((terminal - baseline_pnl[name]).mean())
        scenarios.append({
            "key": key, "label": label, "realized_vol": realized_vol, "spot_jump": jump,
            "jump_step": jump_step if jump else None, "cost_bps": costs, "rebalance_every": every,
            "hedge_dates": (steps - 1) // every + 1, "terminal_spot": prices[-1].tolist(),
            "neural_domain_coverage": result["neural_domain_coverage"], "metrics": result["metrics"], "pnl": result["pnl"],
        })
    return {"scenarios": scenarios, "initial_premium": premium, "tail_count": tail_count,
            "model_names": [name for name, _ in models]}
