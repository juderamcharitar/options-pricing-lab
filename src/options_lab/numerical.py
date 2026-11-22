"""European pricing by CRR and exact-terminal Monte Carlo."""

from math import exp, expm1, isfinite, sqrt
from statistics import median
from time import perf_counter_ns

import numpy as np

from . import pricing


MIN_TREE_STEPS = 8
MAX_TREE_STEPS = 2048
MIN_MC_PATHS = 256
MAX_MC_PATHS = 262144
CONFIDENCE_MULTIPLIER = 1.959963984540054
MIN_POSITIVE_OBSERVATIONS = 20
TIMING_REPEATS = 5


class TreeResolutionError(ValueError):
    """The one-step forward is outside the tree's up/down moves."""


def _check_contract(spot, strike, years, vol, rate, dividend, kind):
    if not all(isfinite(value) for value in (spot, strike, years, vol, rate, dividend)):
        raise ValueError("Contract inputs must be finite.")
    if spot <= 0 or strike <= 0 or years < 0 or vol < 0:
        raise ValueError("Spot and strike must be positive. Time and volatility must be nonnegative.")
    if kind not in ("call", "put"):
        raise ValueError("Choose call or put.")


def _deterministic_value(spot, strike, years, rate, dividend, kind):
    return float(pricing.payoff(spot * exp(-dividend * years), strike * exp(-rate * years), kind))


def crr_price(spot, strike, years, vol, rate=0.0, dividend=0.0, kind="call", *, steps):
    """Discount backward through a recombining Cox–Ross–Rubinstein tree."""
    _check_contract(spot, strike, years, vol, rate, dividend, kind)
    if type(steps) is not int or not 1 <= steps <= MAX_TREE_STEPS:
        raise ValueError(f"Tree steps must be a whole number from 1 to {MAX_TREE_STEPS}.")
    if years == 0 or vol == 0:
        return _deterministic_value(spot, strike, years, rate, dividend, kind)

    dt = years / steps
    jump = vol * sqrt(dt)
    carry_step = (rate - dividend) * dt
    if not -jump < carry_step < jump:
        raise TreeResolutionError("The risk-neutral up probability is outside (0, 1). Increase the tree steps.")

    up_probability = expm1(carry_step + jump) / expm1(2 * jump)
    tree_spots = spot * np.exp((2 * np.arange(steps + 1) - steps) * jump)
    values = pricing.payoff(tree_spots, strike, kind)
    down_probability = 1 - up_probability
    discount = exp(-rate * dt)
    for _ in range(steps):
        values = discount * (down_probability * values[:-1] + up_probability * values[1:])
    return float(values[0])


def monte_carlo(spot, strike, years, vol, rate=0.0, dividend=0.0, kind="call", *,
                paths, seed, antithetic=False):
    """Estimate the discounted payoff and uncertainty from independent observations."""
    _check_contract(spot, strike, years, vol, rate, dividend, kind)
    if type(paths) is not int or not 4 <= paths <= MAX_MC_PATHS or (antithetic and paths % 2):
        raise ValueError(f"Use 4 to {MAX_MC_PATHS} paths, with an even count for antithetic sampling.")
    if type(seed) is not int or not 0 <= seed <= 999999:
        raise ValueError("The seed must be a whole number from 0 to 999999.")
    units = paths // 2 if antithetic else paths
    if years == 0 or vol == 0:
        value = _deterministic_value(spot, strike, years, rate, dividend, kind)
        return {"price": value, "independent_observations": 0, "positive_observations": None,
                "standard_error": 0.0, "ci_lower": value, "ci_upper": value,
                "uncertainty_status": "deterministic"}

    first_stream, second_stream = np.random.SeedSequence(seed).spawn(2)
    first_shocks = np.random.default_rng(first_stream).standard_normal(paths // 2)
    # The two methods share the first stream at each budget. Resetting the
    # streams at each budget keeps smaller runs nested in larger ones.
    other_shocks = -first_shocks if antithetic else (
        np.random.default_rng(second_stream).standard_normal(paths - paths // 2)
    )
    shocks = np.concatenate((first_shocks, other_shocks))
    drift = (rate - dividend - vol**2 / 2) * years
    diffusion = vol * sqrt(years)
    terminal = spot * np.exp(drift + diffusion * shocks)
    discounted_payoffs = exp(-rate * years) * pricing.payoff(terminal, strike, kind)
    observations = (discounted_payoffs[:units] + discounted_payoffs[units:]) / 2 if antithetic else discounted_payoffs
    estimate = float(observations.mean())
    positive = int(np.count_nonzero(observations))
    standard_error = float(observations.std(ddof=1) / sqrt(units))
    reliable_normal_interval = positive >= MIN_POSITIVE_OBSERVATIONS and standard_error > 0
    half_width = CONFIDENCE_MULTIPLIER * standard_error if reliable_normal_interval else None
    return {"price": estimate, "independent_observations": units, "positive_observations": positive,
            "standard_error": standard_error if standard_error > 0 else None,
            "ci_lower": estimate - half_width if half_width is not None else None,
            "ci_upper": estimate + half_width if half_width is not None else None,
            "uncertainty_status": "estimated" if reliable_normal_interval else "insufficient_sample"}


def _timed(run):
    result = run()
    durations = []
    for _ in range(TIMING_REPEATS):
        start = perf_counter_ns()
        run()
        durations.append((perf_counter_ns() - start) / 1000)
    return result, median(durations)


def _budgets(minimum, maximum):
    while minimum <= maximum:
        yield minimum
        minimum *= 2
    if minimum // 2 != maximum:
        yield maximum


def compare(inputs, *, tree_steps, mc_paths, seed):
    """Sweep resolution on the same contract and record single-price timings."""
    if type(tree_steps) is not int or not MIN_TREE_STEPS <= tree_steps <= MAX_TREE_STEPS:
        raise ValueError(f"Maximum tree steps must be a whole number from {MIN_TREE_STEPS} to {MAX_TREE_STEPS}.")
    if type(mc_paths) is not int or not MIN_MC_PATHS <= mc_paths <= MAX_MC_PATHS or mc_paths % 2:
        raise ValueError(f"Maximum Monte Carlo paths must be an even count from {MIN_MC_PATHS} to {MAX_MC_PATHS}.")
    reference, analytic_time = _timed(lambda: float(pricing.price(**inputs)))
    tree = []
    for steps in _budgets(MIN_TREE_STEPS, tree_steps):
        try:
            value, elapsed = _timed(lambda: crr_price(**inputs, steps=steps))
        except TreeResolutionError as error:
            tree.append({"steps": steps, "price": None, "absolute_error": None,
                         "median_microseconds": None, "reason": str(error)})
            continue
        tree.append({"steps": steps, "price": value, "absolute_error": abs(value - reference),
                     "median_microseconds": elapsed, "reason": None})

    simulations = []
    for paths in _budgets(MIN_MC_PATHS, mc_paths):
        row = {"paths": paths}
        for method, antithetic in (("plain", False), ("antithetic", True)):
            result, elapsed = _timed(lambda: monte_carlo(**inputs, paths=paths, seed=seed, antithetic=antithetic))
            row[method] = {**result, "absolute_error": abs(result["price"] - reference),
                           "median_microseconds": elapsed}
        simulations.append(row)
    return {"settings": {**inputs, "tree_steps": tree_steps, "mc_paths": mc_paths, "seed": seed},
            "reference": {"price": reference, "median_microseconds": analytic_time},
            "tree": tree, "monte_carlo": simulations, "timing_repeats": TIMING_REPEATS,
            "numpy_version": np.__version__}
