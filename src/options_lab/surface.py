"""Synthetic European volatility surface and premium inversion."""

from math import exp, isfinite, log2

import numpy as np

from . import pricing


EXPIRY_DAYS = (14, 30, 60, 90, 180, 365, 730)
STRIKE_PCT_FORWARD = tuple(range(75, 126, 5))
REFERENCE_DAYS = 90
MIN_IV_PCT = 1
MAX_IV_PCT = 250
TIME_VALUE_TOLERANCE = 32 * np.finfo(float).eps


def synthetic_surface(*, spot, rate, dividend, atm_vol, skew, smile, term):
    """Price an out-of-the-money European chain, then recover IV from each premium.

    Rates and dividend yields are annual fractions. Volatility inputs and results
    are in percentage points. Expiries use ACT/365.
    """
    if not all(isfinite(value) for value in (spot, rate, dividend, atm_vol, skew, smile, term)):
        raise ValueError("Surface settings must be finite.")
    if spot <= 0:
        raise ValueError("Spot must be positive.")

    strike_pct = np.asarray(STRIKE_PCT_FORWARD)
    wing = (strike_pct - 100) / 25
    term_offset = np.array([term * log2(days / REFERENCE_DAYS) for days in EXPIRY_DAYS])
    input_iv = atm_vol + term_offset[:, None] + skew * wing + smile * wing**2
    outside = np.argwhere((input_iv < MIN_IV_PCT) | (input_iv > MAX_IV_PCT))
    if outside.size:
        row, column = outside[0]
        raise ValueError(
            f"Volatility must stay between {MIN_IV_PCT}% and {MAX_IV_PCT}% across the grid. "
            f"At {EXPIRY_DAYS[row]} days and {STRIKE_PCT_FORWARD[column]}% of forward, "
            f"these settings give {input_iv[row, column]:.2f}%."
        )

    expiries = []
    errors = []
    for days, row_iv in zip(EXPIRY_DAYS, input_iv):
        years = days / 365
        forward = spot * exp((rate - dividend) * years)
        time_value_floor = TIME_VALUE_TOLERANCE * spot * exp(-dividend * years)
        quotes = []
        for relative_strike, target_iv in zip(STRIKE_PCT_FORWARD, row_iv):
            target_iv = float(target_iv)
            strike = forward * (relative_strike / 100)
            kind = "put" if relative_strike < 100 else "call"
            contract = dict(spot=spot, strike=strike, years=years, rate=rate, dividend=dividend, kind=kind)
            premium = float(pricing.price(vol=target_iv / 100, **contract))
            recovered = None
            error = None
            if premium > time_value_floor:
                recovered = pricing.implied_volatility(premium, **contract) * 100
                error = abs(recovered - target_iv)
                errors.append(error)
            quotes.append({"strike_pct_forward": relative_strike, "strike": strike,
                           "kind": kind, "premium": premium, "input_iv_pct": target_iv,
                           "implied_vol_pct": recovered, "absolute_error_vol_points": error,
                           "status": "recovered" if recovered is not None else "low_time_value"})
        expiries.append({"days": days, "forward": forward, "quotes": quotes})

    return {"settings": {"spot": spot, "rate": rate, "dividend": dividend, "atm_vol": atm_vol,
                         "skew": skew, "smile": smile, "term": term},
            "strike_pct_forward": STRIKE_PCT_FORWARD, "reference_days": REFERENCE_DAYS,
            "expiries": expiries, "summary": {"recovered": len(errors),
                                             "contracts": len(EXPIRY_DAYS) * len(STRIKE_PCT_FORWARD),
                                             "max_absolute_error_vol_points": max(errors) if errors else None}}
