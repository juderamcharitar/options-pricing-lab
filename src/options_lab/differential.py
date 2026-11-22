"""Fine-tune the original MLP on prices, spot deltas, and European price bounds."""

from time import perf_counter

import numpy as np
from scipy.special import ndtr
from threadpoolctl import threadpool_limits

from .neural import NeuralPricer, X_LIMIT, scaled_features, training_coordinates
from .pricing import forward_price


def objective(model, features, discount_delta_scale, targets, bounds, delta_weight, bound_weight):
    """Return batch loss and input-derivative-aware parameter gradients."""
    activations = [features]
    derivatives = [np.broadcast_to([1 / X_LIMIT, 0.0], features.shape)]
    pre_derivatives = []
    for weights, bias in zip(model.weights[:-1], model.biases[:-1]):
        hidden = np.tanh(activations[-1] @ weights + bias)
        pre_dx = derivatives[-1] @ weights
        activations.append(hidden)
        derivatives.append((1 - hidden**2) * pre_dx)
        pre_derivatives.append(pre_dx)

    weights, bias = model.weights[-1], model.biases[-1]
    price = (activations[-1] @ weights + bias)[:, 0]
    dx = (derivatives[-1] @ weights)[:, 0]
    delta = dx * discount_delta_scale
    price_error = price - targets[0]
    delta_error = delta - targets[1]
    lower, upper = bounds
    bound_error = np.minimum(price - lower, 0) + np.maximum(price - upper, 0)
    loss = np.mean(price_error**2 + delta_weight * delta_error**2 + bound_weight * bound_error**2)
    output_grad = ((price_error + bound_weight * bound_error) * (2 / len(price)))[:, None]
    dx_grad = (delta_weight * delta_error * discount_delta_scale * (2 / len(price)))[:, None]

    weight_grads = [activations[-1].T @ output_grad + derivatives[-1].T @ dx_grad]
    bias_grads = [output_grad.sum(axis=0)]
    back_value = output_grad @ weights.T
    back_dx = dx_grad @ weights.T
    for i in reversed(range(len(pre_derivatives))):
        hidden = activations[i + 1]
        slope = 1 - hidden**2
        back_z = (back_value - 2 * hidden * pre_derivatives[i] * back_dx) * slope
        back_pre_dx = back_dx * slope
        weight_grads.append(activations[i].T @ back_z + derivatives[i].T @ back_pre_dx)
        bias_grads.append(back_z.sum(axis=0))
        back_value = back_z @ model.weights[i].T
        back_dx = back_pre_dx @ model.weights[i].T
    return float(loss), [*reversed(weight_grads), *reversed(bias_grads)]


def fine_tune(baseline, seed=42, samples=8000, epochs=120, delta_weight=.30, bound_weight=20.0):
    """Use the baseline's samples and initialization with a second, joint objective."""
    start = perf_counter()
    x, w = training_coordinates(seed, samples)
    features = scaled_features(x, w)
    delta_scale = np.exp(-x)
    targets = np.vstack((forward_price(x, w), ndtr(x / w + w / 2)))
    bounds = np.vstack((np.maximum(np.expm1(x), 0), np.exp(x)))
    model = NeuralPricer(tuple(w.copy() for w in baseline.weights), tuple(b.copy() for b in baseline.biases))
    parameters = (*model.weights, *model.biases)
    moving_mean = [np.zeros_like(p) for p in parameters]
    moving_second = [np.zeros_like(p) for p in parameters]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    batch_size, learning_rate = 128, .001
    step = 0
    history = []
    with threadpool_limits(limits=1):
        for epoch in range(epochs):
            order = rng.permutation(samples)
            losses = []
            lr = learning_rate * max(.1, .5 + .5 * np.cos(np.pi * epoch / epochs))
            for start_index in range(0, samples, batch_size):
                ids = order[start_index:start_index + batch_size]
                loss, grads = objective(model, features[ids], delta_scale[ids], targets[:, ids],
                                        bounds[:, ids], delta_weight, bound_weight)
                losses.append(loss * len(ids))
                step += 1
                for parameter, gradient, mean, second in zip(parameters, grads, moving_mean, moving_second):
                    mean *= .9
                    mean += .1 * gradient
                    second *= .999
                    second += .001 * gradient**2
                    parameter -= lr * (mean / (1 - .9**step)) / (np.sqrt(second / (1 - .999**step)) + 1e-8)
            history.append(float(sum(losses) / samples))
    return model, {"settings": {"seed": seed, "samples": samples, "max_epochs": epochs,
                                "optimizer": "adam", "batch_size": batch_size, "learning_rate": learning_rate,
                                "delta_loss_weight": delta_weight, "price_bound_loss_weight": bound_weight,
                                "learning_rate_schedule": "cosine with floor at 10%"},
                   "seconds": perf_counter() - start, "epochs": epochs, "loss": history}
