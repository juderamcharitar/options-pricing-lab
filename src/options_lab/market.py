"""Normalize option quotes and compare them with European pricing models."""

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import io
import math
import re

import numpy as np

from . import pricing
from .neural import in_domain


MAX_QUOTES = 1000
MAX_CSV_BYTES = 180_000
MAX_EXPIRY_DAYS = 1095
EXAMPLE_DATE = date(2026, 1, 15)
EXAMPLE_SYMBOL = "DEMO"
CSV_COLUMNS = ("expiry", "strike", "kind", "bid", "ask", "quote_time")
TIMESTAMP = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})")


@dataclass(frozen=True)
class Quote:
    expiry: date
    strike: float
    kind: str
    bid: float | None
    ask: float | None
    quote_time: datetime | None = None
    contract_symbol: str | None = None


def day(value, label):
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise ValueError(f"{label} must be a valid date in YYYY-MM-DD format.")


def timestamp(value):
    match = TIMESTAMP.fullmatch(value) if isinstance(value, str) else None
    if not match:
        raise ValueError("Quote time must be an ISO timestamp with a time zone, such as 2026-01-15T20:55:00Z.")
    whole, fraction, offset = match.groups()
    if offset != "Z" and (int(offset[1:3]) > 23 or int(offset[4:]) > 59):
        raise ValueError("Quote time must have a valid UTC offset.")
    precision = "." + fraction[:6].ljust(6, "0") if fraction else ""
    try:
        return datetime.fromisoformat(whole + precision + ("+00:00" if offset == "Z" else offset)).astimezone(timezone.utc)
    except (ValueError, OverflowError) as error:
        raise ValueError("Quote time must be a valid timestamp.") from error


def read_csv(contents):
    if not isinstance(contents, str) or not contents.strip():
        raise ValueError("Select a nonempty quote CSV.")
    if len(contents.encode("utf-8")) > MAX_CSV_BYTES:
        raise ValueError(f"The CSV exceeds {MAX_CSV_BYTES:,} bytes.")
    quotes = []
    seen = set()
    try:
        reader = csv.DictReader(io.StringIO(contents.lstrip("\ufeff"), newline=""), strict=True)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not set(CSV_COLUMNS[:-1]).issubset(fields):
            raise ValueError("CSV headers must include expiry,strike,kind,bid,ask. quote_time is optional.")
        for row in reader:
            try:
                if None in row or any(row[column] is None for column in CSV_COLUMNS[:-1]):
                    raise ValueError("Column count does not match the header.")
                expiry = day(row["expiry"].strip(), "Expiry")
                strike = float(row["strike"])
                kind = row["kind"].strip().lower()
                bid, ask = (float(row[key]) if row[key].strip() else None for key in ("bid", "ask"))
                if not math.isfinite(strike) or not .001 <= strike <= 100_000:
                    raise ValueError("Strike must be between 0.001 and 100,000.")
                if kind not in ("call", "put"):
                    raise ValueError("Kind must be call or put.")
                if any(side is not None and (not math.isfinite(side) or not 0 <= side <= 100_000)
                       for side in (bid, ask)):
                    raise ValueError("Bid and ask must be blank or nonnegative finite numbers up to 100,000.")
                raw_time = (row.get("quote_time") or "").strip()
                quote_time = timestamp(raw_time) if raw_time else None
                key = (expiry, strike, kind)
                if key in seen:
                    raise ValueError("Duplicate expiry, strike, and option type.")
                seen.add(key)
                quotes.append(Quote(expiry, strike, kind, bid, ask, quote_time))
            except ValueError as error:
                raise ValueError(f"CSV line {reader.line_num}: {error}") from error
            if len(quotes) > MAX_QUOTES:
                raise ValueError(f"CSV supports up to {MAX_QUOTES:,} quotes.")
    except csv.Error as error:
        raise ValueError("The CSV could not be parsed.") from error
    if not quotes:
        raise ValueError("The CSV has no quote rows.")
    return quotes


def example_quotes():
    as_of = EXAMPLE_DATE
    quote_time = datetime(2026, 1, 15, 20, 55, tzinfo=timezone.utc)
    quotes = []
    for days in (36, 64, 92):
        for strike in range(80, 121, 5):
            for kind in ("call", "put"):
                wings = (strike - 100) / 20
                vol = .25 - .05 * wings + .03 * wings**2 + .015 * math.log2(days / 36)
                premium = float(pricing.price(100, strike, days / 365, vol, .04, 0, kind))
                half_spread = max(.01, .025 * premium)
                bid = max(.01, round(premium - half_spread, 2))
                ask = max(bid + .01, round(premium + half_spread, 2))
                quotes.append(Quote(as_of + timedelta(days=days), strike, kind, bid, round(ask, 2), quote_time))
    return quotes


def example_csv():
    results = io.StringIO(newline="")
    writer = csv.writer(results)
    writer.writerow(CSV_COLUMNS)
    for quote in example_quotes():
        writer.writerow((quote.expiry, quote.strike, quote.kind, quote.bid, quote.ask, quote.quote_time.isoformat()))
    return results.getvalue()


def _quote_row(quote, as_of, spot, rate, dividend):
    days = (quote.expiry - as_of).days
    bid, ask = quote.bid, quote.ask
    if bid is None or ask is None:
        quote_status = "missing_side"
    elif bid > ask:
        quote_status = "crossed"
    elif bid == 0 or ask == 0:
        quote_status = "zero_side"
    else:
        quote_status = "two_sided"

    age_days = (as_of - quote.quote_time.date()).days if quote.quote_time else None
    mid = (bid + ask) / 2 if quote_status == "two_sided" else None
    iv = None
    if not 0 < days <= MAX_EXPIRY_DAYS:
        iv_status = "unsupported_expiry"
    elif age_days is not None and age_days < 0:
        iv_status = "after_valuation_date"
    elif mid is None:
        iv_status = "no_midpoint"
    else:
        years = days / 365
        lower = float(pricing.price(spot, quote.strike, years, 0, rate, dividend, quote.kind))
        upper = spot * math.exp(-dividend * years) if quote.kind == "call" else quote.strike * math.exp(-rate * years)
        tolerance = 32 * np.finfo(float).eps * max(spot, quote.strike)
        if mid < lower - tolerance or mid >= upper:
            iv_status = "outside_european_bounds"
        elif mid <= lower + tolerance:
            iv_status = "no_time_value"
        else:
            try:
                iv = 100 * pricing.implied_volatility(mid, spot, quote.strike, years, rate, dividend, quote.kind)
                iv_status = "recovered"
            except ValueError:
                iv_status = "iv_unavailable"
    return {
        "expiry": quote.expiry.isoformat(), "days": days, "strike": quote.strike, "kind": quote.kind,
        "bid": bid, "ask": ask, "mid": mid, "spread": ask - bid if mid is not None else None,
        "quote_time": quote.quote_time.isoformat() if quote.quote_time else None, "quote_age_days": age_days,
        "quote_status": quote_status, "iv_pct": iv, "iv_status": iv_status,
        "contract_symbol": quote.contract_symbol, "in_domain": None, "model_prices": {},
    }


def analyze(quotes, engines, *, as_of, spot, rate, dividend, reference_vol):
    rows = [_quote_row(quote, as_of, spot, rate, dividend)
            for quote in sorted(quotes, key=lambda row: (row.expiry, row.kind, row.strike))]
    for kind in ("call", "put"):
        group = [row for row in rows if row["kind"] == kind and 0 < row["days"] <= MAX_EXPIRY_DAYS]
        if not group:
            continue
        strikes = np.array([row["strike"] for row in group])
        years = np.array([row["days"] for row in group]) / 365
        inputs = (spot, strikes, years, reference_vol, rate, dividend)
        _, x, w, _ = pricing.normal_inputs(*inputs)
        for row, flag in zip(group, in_domain(x, w)):
            row["in_domain"] = bool(flag)
        for name, engine in engines:
            for row, value in zip(group, engine.price(*inputs, kind)):
                row["model_prices"][name] = float(value)
    return rows
