"""European option pricer with a learned mixture of monotone volatility clocks."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr, softmax
from threadpoolctl import threadpool_limits

from .neural import training_coordinates
from .pricing import forward_price, normal_inputs


SCALES = np.array([.12, .4, .9])
COMPONENTS = 3
DELTA_WEIGHT = .30


def clock_features(w):
    w = np.asarray(w)
    hidden = np.tanh(w[..., None] / SCALES)
    return np.concatenate((w[..., None], SCALES * hidden), axis=-1), hidden


@dataclass(frozen=True)
class MixturePricer:
    """Same-forward lognormal components with contract-independent probabilities."""

    probabilities: np.ndarray
    clock_weights: np.ndarray

    def __post_init__(self):
        probabilities = np.array(self.probabilities, dtype=float, copy=True)
        clock_weights = np.array(self.clock_weights, dtype=float, copy=True)
        if (probabilities.shape != (COMPONENTS,) or not np.isfinite(probabilities).all()
                or np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1, atol=1e-12, rtol=0)):
            raise ValueError("Mixture probabilities must be positive and sum to one.")
        if (clock_weights.shape != (len(SCALES) + 1, COMPONENTS) or not np.isfinite(clock_weights).all()
                or np.any(clock_weights < 0) or np.any(clock_weights[0] <= 0)):
            raise ValueError("Volatility clocks require nonnegative weights and positive linear slopes.")
        for name, values in (("probabilities", probabilities), ("clock_weights", clock_weights)):
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    def price(self, spot, strike, years, vol, rate=0.0, dividend=0.0, kind="call"):
        _, x, w, discount_strike = normal_inputs(spot, strike, years, vol, rate, dividend)
        features, _ = clock_features(w)
        clocks = features @ self.clock_weights
        values = forward_price(x[..., None], clocks, kind)
        return discount_strike * (values @ self.probabilities)

    def _clock_partials(self, w):
        features, hidden = clock_features(w)
        clocks = features @ self.clock_weights
        hidden_slope = 1 - hidden**2
        clock_slopes = self.clock_weights[0] + hidden_slope @ self.clock_weights[1:]
        return clocks, clock_slopes, hidden, hidden_slope

    def greeks(self, spot, strike, years, vol, rate=0.0, dividend=0.0, kind="call"):
        values, x, w, _ = normal_inputs(spot, strike, years, vol, rate, dividend)
        spot, _, years, _, _, dividend = values
        if np.any(w <= 0):
            raise ValueError("Greeks require positive time and volatility.")
        if kind not in ("call", "put"):
            raise ValueError("Choose call or put.")
        clocks, clock_slopes, _, _ = self._clock_partials(w)
        d1 = x[..., None] / clocks + clocks / 2
        density = np.exp(-d1**2 / 2) / np.sqrt(2 * np.pi)
        carry = np.exp(-dividend * years)
        sign = 1 if kind == "call" else -1
        return {
            "delta": carry * sign * (ndtr(sign * d1) @ self.probabilities),
            "gamma": carry / spot * ((density / clocks) @ self.probabilities),
            "vega": spot * carry * np.sqrt(years) / 100 * ((density * clock_slopes) @ self.probabilities),
        }

    def volatility_curvature(self, spot, strike, years, vol, rate=0.0, dividend=0.0):
        values, x, w, _ = normal_inputs(spot, strike, years, vol, rate, dividend)
        if np.any(w <= 0):
            raise ValueError("Greeks require positive time and volatility.")
        spot, _, years, _, _, dividend = values
        clocks, clock_slopes, hidden, hidden_slope = self._clock_partials(w)
        clock_curvature = (-2 * hidden * hidden_slope / SCALES) @ self.clock_weights[1:]
        d1 = x[..., None] / clocks + clocks / 2
        density = np.exp(-d1**2 / 2) / np.sqrt(2 * np.pi)
        d1_clock_slope = .5 - x[..., None] / clocks**2
        carry = np.exp(-dividend * years)
        vanna = carry * np.sqrt(years) / 100 * (
            (density * d1_clock_slope * clock_slopes) @ self.probabilities
        )
        vomma = spot * carry * years / 10_000 * (
            (density * (clock_curvature - d1 * d1_clock_slope * clock_slopes**2)) @ self.probabilities
        )
        return vanna, vomma


def mixture_objective(parameters, features, x, reference_price, reference_delta):
    probabilities = softmax(parameters[:COMPONENTS])
    weights = parameters[COMPONENTS:].reshape(-1, COMPONENTS)
    clocks = features @ weights
    d1 = x[:, None] / clocks + clocks / 2
    density = np.exp(-d1**2 / 2) / np.sqrt(2 * np.pi)
    component_price = forward_price(x[:, None], clocks)
    component_delta = ndtr(d1)
    price = component_price @ probabilities
    delta = component_delta @ probabilities
    price_error = price - reference_price
    delta_error = delta - reference_delta
    scale = 2 / len(x)
    price_grad = scale * price_error
    delta_grad = scale * DELTA_WEIGHT * delta_error
    probability_grad = component_price.T @ price_grad + component_delta.T @ delta_grad
    logits_grad = probabilities * (probability_grad - probabilities @ probability_grad)
    price_clock_slope = np.exp(x[:, None]) * density
    delta_clock_slope = density * (.5 - x[:, None] / clocks**2)
    clock_grad = probabilities * (price_grad[:, None] * price_clock_slope + delta_grad[:, None] * delta_clock_slope)
    loss = np.mean(price_error**2 + DELTA_WEIGHT * delta_error**2)
    return float(loss), np.concatenate((logits_grad, (features.T @ clock_grad).ravel()))


def train_mixture(seed=42, samples=8000):
    """Fit the positive clocks and mixture probabilities to the shared synthetic draw."""
    x, w = training_coordinates(seed, samples)
    features, _ = clock_features(w)
    targets = forward_price(x, w), ndtr(x / w + w / 2)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 11]))
    initial_weights = np.vstack((np.array([.61, .83, 1.12]), rng.uniform(.015, .065, (len(SCALES), COMPONENTS))))
    initial = np.concatenate((np.zeros(COMPONENTS), initial_weights.ravel()))
    bounds = [(-6., 6.)] * COMPONENTS + [(.05, 3.)] * COMPONENTS + [(0., 2.)] * (len(SCALES) * COMPONENTS)
    history = []

    def record(parameters):
        history.append(mixture_objective(parameters, features, x, *targets)[0])

    started = perf_counter()
    with threadpool_limits(limits=1):
        fit = minimize(mixture_objective, initial, args=(features, x, *targets), jac=True, method="L-BFGS-B",
                       bounds=bounds, callback=record, options={"maxiter": 120, "ftol": 1e-14, "gtol": 1e-9})
    if not np.isfinite(fit.fun):
        raise RuntimeError("Mixture training failed to produce a finite loss.")
    model = MixturePricer(softmax(fit.x[:COMPONENTS]), fit.x[COMPONENTS:].reshape(-1, COMPONENTS))
    return model, {
        "settings": {"seed": seed, "samples": samples, "components": COMPONENTS, "optimizer": "L-BFGS-B",
                     "delta_loss_weight": DELTA_WEIGHT, "clock_scales": SCALES.tolist()},
        "seconds": perf_counter() - started, "iterations": int(fit.nit), "converged": bool(fit.success),
        "message": str(fit.message), "loss": history or [float(fit.fun)],
        "probabilities": model.probabilities.tolist(), "clock_weights": model.clock_weights.tolist(),
    }
