"""Smooth NumPy inference and derivatives for a fitted tanh MLP."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

from .pricing import forward_price, normal_inputs

X_LIMIT = 0.85
W_MIN = 0.005
W_MAX = 1.0
W_CENTER = (W_MAX + W_MIN) / 2
W_SCALE = (W_MAX - W_MIN) / 2


@dataclass(frozen=True)
class NeuralPricer:
    weights: tuple
    biases: tuple

    def normalized(self, x, w):
        hidden = scaled_features(x, w)
        for weights, bias in zip(self.weights[:-1], self.biases[:-1]):
            hidden = np.tanh(hidden @ weights + bias)
        return (hidden @ self.weights[-1] + self.biases[-1]).reshape(np.shape(x))

    def derivatives(self, x, w, *, with_vol_curvature=False):
        features = scaled_features(x, w)
        dx = np.array([1 / X_LIMIT, 0.0])
        dw = np.array([0.0, 1 / W_SCALE])
        dxx = np.zeros(2)
        if with_vol_curvature:
            dxw = np.zeros(2)
            dww = np.zeros(2)
        for weights, bias in zip(self.weights[:-1], self.biases[:-1]):
            hidden = np.tanh(features @ weights + bias)
            slope = 1 - hidden**2
            pre_dx = dx @ weights
            pre_dw = dw @ weights
            dxx = slope * (dxx @ weights - 2 * hidden * pre_dx**2)
            if with_vol_curvature:
                dxw = slope * (dxw @ weights - 2 * hidden * pre_dx * pre_dw)
                dww = slope * (dww @ weights - 2 * hidden * pre_dw**2)
            dx = slope * pre_dx
            dw = slope * pre_dw
            features = hidden
        end = self.weights[-1]
        partials = (dx, dxx, dw, dxw, dww) if with_vol_curvature else (dx, dxx, dw)
        return tuple((array @ end).reshape(np.shape(x)) for array in partials)

    def price(self, spot, strike, years, vol, rate=0.0, dividend=0.0, kind="call"):
        if kind not in ("call", "put"):
            raise ValueError("Choose call or put.")
        _, x, w, discount_strike = normal_inputs(spot, strike, years, vol, rate, dividend)
        normalized = self.normalized(x, w)
        if kind == "put":
            normalized -= np.expm1(x)
        return np.where(w == 0, discount_strike * forward_price(x, 0, kind), discount_strike * normalized)

    def greeks(self, spot, strike, years, vol, rate=0.0, dividend=0.0, kind="call"):
        if kind not in ("call", "put"):
            raise ValueError("Choose call or put.")
        values, x, w, discount_strike = normal_inputs(spot, strike, years, vol, rate, dividend)
        spot, _, years, _, _, dividend = values
        if np.any(w <= 0):
            raise ValueError("Greeks require positive time and volatility.")
        dx, dxx, dw = self.derivatives(x, w)
        delta = discount_strike / spot * dx
        if kind == "put":
            delta -= np.exp(-dividend * years)
        return {"delta": delta, "gamma": discount_strike / spot**2 * (dxx - dx),
                "vega": discount_strike * np.sqrt(years) * dw / 100}

    def volatility_curvature(self, spot, strike, years, vol, rate=0.0, dividend=0.0):
        values, x, w, discount_strike = normal_inputs(spot, strike, years, vol, rate, dividend)
        if np.any(w <= 0):
            raise ValueError("Greeks require positive time and volatility.")
        spot, _, years, _, _, _ = values
        _, _, _, dxw, dww = self.derivatives(x, w, with_vol_curvature=True)
        return discount_strike * np.sqrt(years) * dxw / (100 * spot), discount_strike * years * dww / 10_000


def in_domain(x, w):
    return (np.abs(x) <= X_LIMIT) & (w >= W_MIN) & (w <= W_MAX)


def training_coordinates(seed, samples):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-X_LIMIT, X_LIMIT, samples)
    w = rng.uniform(W_MIN, W_MAX, samples)
    # Add coverage near expiry and forward at-the-money, where gamma changes sharply.
    edge = samples // 4
    x[:edge] = rng.uniform(-0.24, 0.24, edge)
    w[:edge] = rng.uniform(W_MIN, 0.18, edge)
    return x, w


def scaled_features(x, w):
    return np.column_stack((np.ravel(x) / X_LIMIT, (np.ravel(w) - W_CENTER) / W_SCALE))


def train_pricer(seed=42, samples=8000, epochs=180, width=32):
    x, w = training_coordinates(seed, samples)
    features = scaled_features(x, w)
    fitted = MLPRegressor(hidden_layer_sizes=(width, width), activation="tanh",
                          random_state=seed, batch_size=128, learning_rate_init=.003,
                          alpha=1e-7, max_iter=epochs, shuffle=True, n_iter_no_change=30,
                          tol=1e-7)
    start = perf_counter()
    with threadpool_limits(limits=1):
        # The epoch limit is an explicit experiment budget. Capture its termination in metadata.
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            fitted.fit(features, forward_price(x, w))
    model = NeuralPricer(tuple(fitted.coefs_), tuple(fitted.intercepts_))
    return model, {"settings": {"seed": seed, "samples": samples, "max_epochs": epochs,
                                "hidden_layers": [width, width], "activation": "tanh",
                                "optimizer": "adam", "batch_size": 128, "learning_rate": .003},
                   "seconds": perf_counter() - start, "epochs": int(fitted.n_iter_),
                   "epoch_limit_reached": any(issubclass(w.category, ConvergenceWarning) for w in caught),
                   "parameters": sum(a.size for a in (*model.weights, *model.biases)),
                   "loss": [float(v) for v in fitted.loss_curve_]}
