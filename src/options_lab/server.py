"""Local web interface and JSON endpoints. Run with python -m options_lab.server."""

import argparse
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
import json
import math
from importlib.metadata import version
from threading import Timer
import webbrowser

import numpy as np

from . import __version__, alpaca_data, experiments, market, numerical, pricing, robustness, sensitivities, surface
from .differential import fine_tune
from .experiments import BASELINE, FINE_TUNED, MIXTURE
from .mixture import train_mixture
from .neural import X_LIMIT, W_MIN, W_MAX, in_domain, train_pricer


ASSETS = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
          "/style.css": ("style.css", "text/css")}


def number(body, key, default, lo, hi, *, integer=False):
    value = body.get(key, default)
    if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{key} must be a number from {lo} to {hi}.")
    if integer and int(value) != value:
        raise ValueError(f"{key} must be a whole number.")
    return int(value) if integer else float(value)


def contract(body):
    kind = body.get("kind", "call")
    if kind not in ("call", "put"):
        raise ValueError("Choose call or put.")
    return {
        "spot": number(body, "spot", 100, .01, 100_000),
        "strike": number(body, "strike", 100, .01, 100_000),
        "years": number(body, "days", 90, 1, 1095) / 365,
        "vol": number(body, "vol", 25, .1, 250) / 100,
        "rate": number(body, "rate", 4, -20, 30) / 100,
        "dividend": number(body, "dividend", 0, 0, 30) / 100,
        "kind": kind,
    }


@lru_cache(maxsize=3)
def fitted(seed):
    return train_pricer(seed=seed)


@lru_cache(maxsize=3)
def improved(seed):
    baseline, training = fitted(seed)
    return fine_tune(baseline, seed=seed, samples=training["settings"]["samples"])


@lru_cache(maxsize=3)
def structured(seed):
    return train_mixture(seed=seed)


def valuation(body, model, comparison=None, mixture=None):
    inputs = contract(body)
    parameters = {k: v for k, v in inputs.items() if k != "kind"}
    _, x, w, _ = pricing.normal_inputs(**parameters)
    models = experiments.pricing_engines(model, comparison, mixture)
    spots = np.linspace(inputs["spot"] * .65, inputs["spot"] * 1.35, 101)
    sweep = {**inputs, "spot": spots}
    tail = tuple(inputs[key] for key in ("strike", "years", "vol", "rate", "dividend", "kind"))
    results = []
    curve = {"spot": spots.tolist()}
    greek_curves = {}
    for label, engine in models:
        current = (inputs["spot"], *tail)
        premium = engine.price(*current)
        greeks = engine.greeks(*current)
        extended = sensitivities.extended_greeks(engine, current, premium, greeks)
        results.append({"method": label, "price": float(premium),
                        **{key: float(value) for key, value in {**greeks, **extended}.items()}})

        series = (spots, *tail)
        prices = engine.price(*series)
        greeks = engine.greeks(*series)
        extended = sensitivities.extended_greeks(engine, series, prices, greeks)
        curve[label] = prices.tolist()
        greek_curves[label] = {key: value.tolist() for key, value in {**greeks, **extended}.items()}
    _, curve_x, curve_w, _ = pricing.normal_inputs(**{k: v for k, v in sweep.items() if k != "kind"})
    iv = None
    if body.get("premium") is not None:
        premium = number(body, "premium", 0, 0, 100_000)
        iv_inputs = {k: v for k, v in inputs.items() if k != "vol"}
        iv = pricing.implied_volatility(premium, **iv_inputs) * 100
    return {"metrics": results, "curve": curve, "greek_curves": greek_curves, "in_domain": bool(in_domain(x, w)),
            "sweep_in_domain": bool(in_domain(curve_x, curve_w).all()), "implied_vol_percent": iv,
            "training_domain": {"absolute_log_forward_limit": X_LIMIT,
                                "minimum_total_volatility": W_MIN, "maximum_total_volatility": W_MAX}}


def market_analysis(body):
    source = body.get("source", "example")
    if source not in ("example", "csv", "alpaca"):
        raise ValueError("Choose Example chain, CSV upload, or Alpaca snapshot.")
    rate = number(body, "rate", 4, -20, 30) / 100
    dividend = number(body, "dividend", 0, 0, 30) / 100
    reference_vol = number(body, "reference_vol", 25, .1, 250) / 100
    details = {}
    if source == "example":
        as_of, spot, symbol = market.EXAMPLE_DATE, 100., market.EXAMPLE_SYMBOL
        quotes = market.example_quotes()
    elif source == "csv":
        as_of = market.day(body.get("as_of"), "Valuation date")
        spot = number(body, "spot", None, .01, 100_000)
        symbol = None
        quotes = market.read_csv(body.get("quotes_csv"))
    else:
        as_of = datetime.now(timezone.utc).date()
        raw_symbol = body.get("symbol", "AAPL")
        symbol = raw_symbol.strip().upper() if isinstance(raw_symbol, str) else None
        raw_expiry = body.get("expiry")
        expiry = market.day(raw_expiry, "Expiry") if raw_expiry else None
        if expiry and not 0 < (expiry - as_of).days <= market.MAX_EXPIRY_DAYS:
            raise ValueError("Expiry must be 1 to 1,095 calendar days after today in UTC.")
        feed = body.get("feed", "indicative")
        details = alpaca_data.snapshot(symbol, feed, expiry, as_of)
        quotes, spot = details.pop("quotes"), details.pop("spot")
        details["options_feed"] = feed
    model, _ = fitted(42)
    tuned, _ = improved(42)
    mixture, _ = structured(42)
    engines = experiments.pricing_engines(model, tuned, mixture)
    rows = market.analyze(quotes, engines, as_of=as_of, spot=spot, rate=rate,
                          dividend=dividend, reference_vol=reference_vol)
    return {"source": source, "as_of": as_of.isoformat(), "spot": spot, "symbol": symbol,
            "rate_pct": rate * 100, "dividend_pct": dividend * 100, "reference_vol_pct": reference_vol * 100,
            "rows": rows, **details}


def robustness_analysis(body):
    count = number(body, "training_runs", 3, 2, 5, integer=True)
    first_seed = number(body, "seed", 42, 0, 1_000_000 - count, integer=True)
    test_seed = number(body, "test_seed", 2026, 0, 999_999, integer=True)
    samples = number(body, "samples_per_regime", 512, 64, 2048, integer=True)
    model_runs = ((seed, {BASELINE: fitted(seed), FINE_TUNED: improved(seed), MIXTURE: structured(seed)})
                  for seed in range(first_seed, first_seed + count))
    result = robustness.evaluate_seeds(model_runs, test_seed=test_seed, samples=samples)
    return {**result, "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn")}}


def research_report(body):
    """Run the reference pricing and hedge experiments alongside a robustness study."""
    # Validate and run the seed study first so malformed requests fail before other experiments.
    study = robustness_analysis({"seed": body.get("seed", 42), "test_seed": body.get("test_seed", 2026)})
    seed = study["runs"][0]["seed"]
    inputs = {"seed": seed}
    prices = compute("/api/price", inputs)
    stress = compute("/api/stress", inputs)
    return {
        "schema_version": 1,
        "app_version": __version__,
        "pricing": {"seed": seed, "contract": contract({}), "metrics": prices["metrics"]},
        "robustness": study,
        "hedge_stress": {
            "settings": stress["settings"], "initial_premium": stress["initial_premium"],
            "tail_count": stress["tail_count"],
            "scenarios": [{key: value for key, value in row.items() if key not in ("terminal_spot", "pnl")}
                          for row in stress["scenarios"]],
        },
    }


def compute(path, body):
    if path == "/api/report":
        return research_report(body)
    if path == "/api/market":
        return market_analysis(body)
    if path == "/api/robustness":
        return robustness_analysis(body)
    if path == "/api/surface":
        return surface.synthetic_surface(
            spot=number(body, "spot", 100, .01, 100_000),
            rate=number(body, "rate", 4, -20, 30) / 100,
            dividend=number(body, "dividend", 0, 0, 30) / 100,
            atm_vol=number(body, "atm_vol", 25, 1, 250),
            skew=number(body, "skew", -5, -25, 25),
            smile=number(body, "smile", 4, -25, 25),
            term=number(body, "term", 1.5, -10, 10),
        )
    if path not in ("/api/price", "/api/evaluate", "/api/hedge", "/api/stress", "/api/numerical"):
        raise KeyError(path)
    seed = number(body, "seed", 42, 0, 999_999, integer=True)
    if path == "/api/numerical":
        return numerical.compare(
            contract(body),
            tree_steps=number(body, "tree_steps", 1024, numerical.MIN_TREE_STEPS, numerical.MAX_TREE_STEPS, integer=True),
            mc_paths=number(body, "mc_paths", 131072, numerical.MIN_MC_PATHS, numerical.MAX_MC_PATHS, integer=True),
            seed=seed,
        )
    if path == "/api/price":
        contract(body)
    elif path in ("/api/hedge", "/api/stress"):
        inputs = contract(body)
        if inputs.pop("dividend") != 0:
            raise ValueError("The hedge simulator requires a zero dividend yield.")
        maximum_steps, maximum_paths = (126, 2500) if path == "/api/stress" else (252, 5000)
        steps = number(body, "steps", 30, 2, maximum_steps, integer=True)
        paths = number(body, "paths", 1000, 100, maximum_paths, integer=True)
        costs = number(body, "cost_bps", 5, 0, 100)
        if path == "/api/stress":
            stresses = {
                "stress_vol": number(body, "stress_vol", 40, .1, 250) / 100,
                "stress_cost_bps": number(body, "stress_cost_bps", max(25, costs), costs, 500),
                "spot_jump": number(body, "spot_jump", -10, -80, 100) / 100,
                "coarse_every": number(body, "coarse_every", 3, 1, 20, integer=True),
            }
    model, training = fitted(seed)
    tuned, tuning = improved(seed)
    mixture, mixture_training = structured(seed)
    if path == "/api/price":
        return valuation(body, model, tuned, mixture)
    if path == "/api/evaluate":
        return {"models": experiments.compare_models(model, tuned, seed, mixture=mixture),
                "benchmark": experiments.benchmark(model, seed, comparison=tuned, mixture=mixture),
                "training": {BASELINE: training, FINE_TUNED: tuning, MIXTURE: mixture_training}, "seed": seed,
                "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn")}}
    if path == "/api/stress":
        result = experiments.hedge_stress(model, **inputs, **stresses, steps=steps, paths=paths,
                                          cost_bps=costs, seed=seed, comparison=tuned, mixture=mixture)
        return {**result, "settings": {**inputs, **stresses, "steps": steps, "paths": paths,
                                        "cost_bps": costs, "seed": seed}}
    return {**experiments.hedge(model, **inputs, steps=steps, paths=paths, cost_bps=costs, seed=seed,
                                comparison=tuned, mixture=mixture),
            "settings": {**inputs, "steps": steps, "paths": paths, "cost_bps": costs, "seed": seed}}


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, mime, content):
        self.send_response(status)
        self.send_header("Content-Type", mime + "; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def json_response(self, status, data):
        self.respond(status, "application/json", json.dumps(data, allow_nan=False).encode())

    def do_GET(self):
        if self.path == "/market-example.csv":
            return self.respond(200, "text/csv", market.example_csv().encode())
        if self.path not in ASSETS:
            return self.json_response(404, {"error": "Page unavailable."})
        name, mime = ASSETS[self.path]
        self.respond(200, mime, files("options_lab").joinpath("web", name).read_bytes())

    def do_POST(self):
        if self.path == "/api/market":
            port = self.server.server_port
            allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host", "").lower() not in allowed_hosts:
                return self.json_response(403, {"error": "Open the lab using its localhost address."})
        if self.headers.get_content_type() != "application/json":
            return self.json_response(415, {"error": "Expected JSON."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            limit = 400_000 if self.path == "/api/market" else 8192
            if not 0 < size <= limit:
                raise ValueError(f"Request must be between 1 and {limit:,} bytes.")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("Provide an object with input fields.")
            self.json_response(200, compute(self.path, body))
        except KeyError:
            self.json_response(404, {"error": "Unknown experiment."})
        except alpaca_data.ProviderError as exc:
            self.json_response(502, {"error": str(exc)})
        except (ValueError, OverflowError) as exc:
            self.json_response(400, {"error": str(exc)})
        except Exception:
            import traceback
            traceback.print_exc()
            self.json_response(500, {"error": "Experiment failed. See the terminal for details."})


def main():
    parser = argparse.ArgumentParser(description="Launch the local Options Pricing Lab.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--port", type=int, default=8515)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Options Pricing Lab is running at {url}", flush=True)
        if not args.no_browser:
            Timer(.5, webbrowser.open, args=(url,)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
