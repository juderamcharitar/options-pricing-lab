"""Vectorized European Black–Scholes–Merton pricing and first sensitivities."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr


def payoff(spot, strike, kind):
    if kind not in ("call", "put"):
        raise ValueError("Choose call or put.")
    sign = 1 if kind == "call" else -1
    return np.maximum(sign * (np.asarray(spot) - strike), 0.0)


def normal_inputs(spot, strike, years, vol, rate, dividend):
    values = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in
                                   (spot, strike, years, vol, rate, dividend)))
    spot, strike, years, vol, rate, dividend = values
    if not all(np.isfinite(x).all() for x in values):
        raise ValueError("Inputs must be finite.")
    if np.any((spot <= 0) | (strike <= 0) | (years < 0) | (vol < 0)):
        raise ValueError("Spot and strike must be positive. Time and volatility must be nonnegative.")
    log_forward = np.log(spot) - np.log(strike) + (rate - dividend) * years
    discount_strike = strike * np.exp(-rate * years)
    return values, log_forward, vol * np.sqrt(years), discount_strike


def forward_price(x, w, kind="call"):
    """Price divided by the discounted strike. Supports zero total volatility."""
    if kind not in ("call", "put"):
        raise ValueError("Choose call or put.")
    safe_w = np.where(w > 0, w, 1)
    d1 = x / safe_w + w / 2
    d2 = d1 - w
    forward = np.exp(x)
    out_of_money = np.where(
        x <= 0, forward * ndtr(d1) - ndtr(d2), ndtr(-d2) - forward * ndtr(-d1)
    )
    intrinsic = np.maximum(np.expm1(x) if kind == "call" else -np.expm1(x), 0)
    return intrinsic + np.where(w > 0, np.maximum(out_of_money, 0), 0)


def price(spot, strike, years, vol, rate=0.0, dividend=0.0, kind="call"):
    _, x, w, discount_strike = normal_inputs(spot, strike, years, vol, rate, dividend)
    return discount_strike * forward_price(x, w, kind)


def greeks(spot, strike, years, vol, rate=0.0, dividend=0.0, kind="call"):
    values, x, w, _ = normal_inputs(spot, strike, years, vol, rate, dividend)
    spot, _, years, _, _, dividend = values
    if np.any(w <= 0):
        raise ValueError("Greeks require positive time and volatility.")
    if kind not in ("call", "put"):
        raise ValueError("Choose call or put.")
    d1 = x / w + w / 2
    density = np.exp(-d1**2 / 2) / np.sqrt(2 * np.pi)
    carry = np.exp(-dividend * years)
    sign = 1 if kind == "call" else -1
    return {
        "delta": carry * sign * ndtr(sign * d1),
        "gamma": carry * density / (spot * w),
        "vega": spot * carry * density * np.sqrt(years) / 100,
    }


def volatility_curvature(spot, strike, years, vol, rate=0.0, dividend=0.0):
    """Return vanna per vol point and vomma per squared vol point."""
    values, x, w, _ = normal_inputs(spot, strike, years, vol, rate, dividend)
    spot, _, years, vol, _, dividend = values
    if np.any(w <= 0):
        raise ValueError("Greeks require positive time and volatility.")
    d1 = x / w + w / 2
    d2 = d1 - w
    discounted_density = np.exp(-dividend * years - d1**2 / 2) / np.sqrt(2 * np.pi)
    vanna = -discounted_density * d2 / (100 * vol)
    vomma = spot * discounted_density * np.sqrt(years) * d1 * d2 / (10_000 * vol)
    return vanna, vomma


def implied_volatility(premium, spot, strike, years, rate=0.0, dividend=0.0, kind="call"):
    """Recover annualized volatility from a European premium using Brent's method."""
    if not np.isfinite(premium) or years <= 0:
        raise ValueError("Implied volatility requires a finite premium and positive time.")
    lower = float(price(spot, strike, years, 0, rate, dividend, kind))
    upper = spot * np.exp(-dividend * years) if kind == "call" else strike * np.exp(-rate * years)
    if not lower <= premium < upper:
        raise ValueError(f"Premium must be at least {lower:.6g} and below {upper:.6g}.")
    if premium == lower:
        return 0.0
    ceiling = 1.0
    while price(spot, strike, years, ceiling, rate, dividend, kind) < premium:
        ceiling *= 2
        if ceiling > 512:
            raise ValueError("Premium implies volatility outside the numerical bracket.")
    return float(brentq(lambda vol: price(spot, strike, years, vol, rate, dividend, kind) - premium,
                        0, ceiling, xtol=1e-12))
