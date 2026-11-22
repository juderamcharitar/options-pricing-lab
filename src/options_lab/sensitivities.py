"""Calendar, rate, and second-order volatility sensitivities."""

import numpy as np


def extended_greeks(engine, contract, premium, base_greeks):
    """Evaluate each model at fixed contract inputs, using the same units."""
    spot, strike, years, vol, rate, dividend, kind = contract
    curvature = getattr(engine, "volatility_curvature", None)
    if curvature is None:
        # Preserve support for duck-typed engines that expose only price and greeks.
        vol_bump, time_bump, rate_bump = np.minimum(1e-4, vol * 1e-4), years * 1e-4, 1e-4
        upper = engine.greeks(spot, strike, years, vol + vol_bump, rate, dividend, kind)
        lower = engine.greeks(spot, strike, years, vol - vol_bump, rate, dividend, kind)
        vanna = (upper["delta"] - lower["delta"]) / (200 * vol_bump)
        vomma = (upper["vega"] - lower["vega"]) / (200 * vol_bump)
        earlier = engine.price(spot, strike, years + time_bump, vol, rate, dividend, kind)
        later = engine.price(spot, strike, years - time_bump, vol, rate, dividend, kind)
        annual_theta = (later - earlier) / (2 * time_bump)
        higher_rate = engine.price(spot, strike, years, vol, rate + rate_bump, dividend, kind)
        lower_rate = engine.price(spot, strike, years, vol, rate - rate_bump, dividend, kind)
        rho = (higher_rate - lower_rate) / (200 * rate_bump)
    else:
        spot_delta = spot * base_greeks["delta"]
        # Time changes discounting, forward moneyness, and total volatility.
        # Vega is already quoted per one vol point.
        annual_theta = rate * premium - (rate - dividend) * spot_delta - (
            vol * base_greeks["vega"] * 100 / (2 * years)
        )
        rho = years * (spot_delta - premium) / 100
        vanna, vomma = curvature(spot, strike, years, vol, rate, dividend)
    return {"theta": annual_theta / 365, "rho": rho, "vanna": vanna, "vomma": vomma}
