"""
Surrogate gradient functions for spiking neural networks.
MLX port of spyx.axn — uses mx.custom_function for custom VJPs.
"""

import mlx.core as mx


def _make_spike_fn(surrogate_grad_fn, k=25.0):
    """
    Build a heaviside spike function with a custom surrogate gradient.

    Args:
        surrogate_grad_fn: function (u, k) -> gradient tensor
        k: sharpness parameter
    Returns:
        callable with custom VJP
    """
    @mx.custom_function
    def spike(u):
        return (u >= 0).astype(u.dtype)

    @spike.vjp
    def spike_vjp(primals, cotangent, output):
        # For a single-argument custom_function, primals IS the input array.
        u = primals
        grad = surrogate_grad_fn(u, k)
        return cotangent * grad

    return spike


def superspike(k=25.0):
    """SuperSpike surrogate: 1/(1 + k|u|)^2 (Zenke & Ganguli 2018)."""
    def _grad(u, k):
        return 1.0 / (1.0 + k * mx.abs(u)) ** 2
    return _make_spike_fn(_grad, k)


def arctan(k=2.0):
    """Arctan surrogate: 1 / ((1 + k^2 pi^2 u^2) * pi)."""
    import math
    def _grad(u, k):
        return 1.0 / ((1.0 + (k ** 2) * (math.pi ** 2) * (u ** 2)) * math.pi)
    return _make_spike_fn(_grad, k)


def triangular(k=1.0):
    """Triangular surrogate: max(0, 1 - k|u|)."""
    def _grad(u, k):
        return mx.maximum(0.0, 1.0 - k * mx.abs(u))
    return _make_spike_fn(_grad, k)


def boxcar(width=2.0, height=0.5):
    """Boxcar surrogate: height if |u| < width/2 else 0. Matches spyx default."""
    half = width / 2.0
    def _grad(u, k):  # k carries half-width
        return height * (mx.abs(u) < k).astype(u.dtype)
    return _make_spike_fn(_grad, half)


def straight_through():
    """Straight-through estimator: gradient passes unchanged."""
    def _grad(u, k):
        return mx.ones_like(u)
    return _make_spike_fn(_grad, 1.0)
