"""
Spiking neuron models for MLX.
MLX port of spyx.nn — uses mlx.nn.Module instead of Haiku RNNCore.

All neurons follow the same interface:
    out, new_state = neuron(x, state)
    state = neuron.initial_state(batch_size)

Design notes:
- IF/LIF/ALIF/CuBaLIF update equations match spyx.nn for strict numerical
    parity tests.
- Beta parametrised in logit space (sigmoid reparametrisation). This constrains
  beta to (0,1) and provides implicit regularisation: the gradient through
  sigmoid at beta=0.9 is 0.09, slowing drift away from long-memory values.
- Default beta/gamma init: logit(0.9) → fast training convergence on SHD.
- RLIF: uses previous-timestep spikes for feedback (no circular dependency).
"""

import math
import warnings
from collections.abc import Sequence
from itertools import product
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .axn import superspike

_LOGIT_09 = math.log(0.9 / 0.1)   # sigmoid^{-1}(0.9) ≈ 2.197


# ---------------------------------------------------------------------------
# IF — Integrate-and-Fire
# ---------------------------------------------------------------------------


class IF(nn.Module):
    """
    Integrate-and-Fire neuron.
        V[t] = V[t-1] + x[t]
        spike[t] = Heaviside(V[t] - threshold)
        V[t] -= spike[t] * threshold   (soft reset)
    """

    def __init__(self, hidden_shape, threshold=1.0, activation=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)
        self.threshold = threshold
        self._spike = activation if activation is not None else superspike()

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + self.hidden_shape)

    def __call__(self, x, V):
        # Match spyx.IF: spike uses previous V, then membrane is updated.
        spike = self._spike(V - self.threshold)
        V = V + x - spike * self.threshold
        return spike, V


# ---------------------------------------------------------------------------
# LIF — Leaky Integrate-and-Fire
# ---------------------------------------------------------------------------


class LIF(nn.Module):
    """
    Leaky Integrate-and-Fire neuron.
        V[t] = beta * V[t-1] + x[t]
        spike[t] = Heaviside(V[t] - threshold)
        V[t] -= spike[t] * threshold

    beta is learnable per-neuron via sigmoid(beta_logit), initialised at 0.9.
    """

    def __init__(self, hidden_shape, beta=None, threshold=1.0, activation=None, beta_init=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)
        self.threshold = threshold
        self._spike = activation if activation is not None else superspike()
        self._beta_fixed = None
        beta_value = beta if beta is not None else beta_init
        if beta_value is not None:
            self._beta_fixed = float(beta_value)
        else:
            self.beta_logit = mx.full(self.hidden_shape, _LOGIT_09)

    def _beta(self):
        if self._beta_fixed is not None:
            return self._beta_fixed
        return mx.sigmoid(self.beta_logit)

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + self.hidden_shape)

    def __call__(self, x, V):
        # Match spyx.LIF: spike from previous V, then leaky integration update.
        spike = self._spike(V - self.threshold)
        V = self._beta() * V + x - spike * self.threshold
        return spike, V


# ---------------------------------------------------------------------------
# LI — Leaky Integrator (non-spiking readout)
# ---------------------------------------------------------------------------


class LI(nn.Module):
    """
    Leaky Integrator — no spike, used as output readout layer.
        V[t] = beta * V[t-1] + x[t]

    beta learnable via sigmoid(beta_logit), initialised at 0.9.
    """

    def __init__(self, layer_shape=None, beta=None, beta_init=None, hidden_shape=None):
        super().__init__()
        if layer_shape is None:
            layer_shape = hidden_shape
        if layer_shape is None:
            raise ValueError("LI requires `layer_shape` (or compatibility alias `hidden_shape`).")
        self.hidden_shape = (layer_shape,) if isinstance(layer_shape, int) else tuple(layer_shape)
        self._beta_fixed = None
        beta_value = beta if beta is not None else beta_init
        if beta_value is not None:
            self._beta_fixed = float(beta_value)
        else:
            self.beta_logit = mx.full(self.hidden_shape, _LOGIT_09)

    def _beta(self):
        if self._beta_fixed is not None:
            return self._beta_fixed
        return mx.sigmoid(self.beta_logit)

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + self.hidden_shape)

    def __call__(self, x, V):
        V = self._beta() * V + x
        return V, V   # (output trace, new state)


# ---------------------------------------------------------------------------
# CuBaLIF — Current-Based LIF (dual time constants)
# ---------------------------------------------------------------------------


class CuBaLIF(nn.Module):
    """
    Current-Based LIF: separates synaptic current and membrane dynamics.
        I[t]  = alpha * I[t-1] + x[t]
        V[t]  = beta  * V[t-1] + I[t]
        spike[t] = Heaviside(V[t] - threshold)
        V[t] -= spike[t] * threshold

    Matches spyx.nn.CuBaLIF update order and reset behavior.
    State layout matches spyx: concatenate [V, I] along the last axis.
    """

    def __init__(self, hidden_shape, alpha=None, beta=None, threshold=1.0, activation=None, alpha_init=None, beta_init=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)
        self.threshold = threshold
        self._spike = activation if activation is not None else superspike()
        self._alpha_fixed = None
        self._beta_fixed = None
        alpha_value = alpha if alpha is not None else alpha_init
        beta_value = beta if beta is not None else beta_init
        if alpha_value is not None:
            self._alpha_fixed = float(alpha_value)
        else:
            self.alpha_logit = mx.full(self.hidden_shape, _LOGIT_09)
        if beta_value is not None:
            self._beta_fixed = float(beta_value)
        else:
            self.beta_logit = mx.full(self.hidden_shape, _LOGIT_09)

    def _alpha(self):
        return self._alpha_fixed if self._alpha_fixed is not None else mx.sigmoid(self.alpha_logit)

    def _beta(self):
        return self._beta_fixed if self._beta_fixed is not None else mx.sigmoid(self.beta_logit)

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + tuple(2 * v for v in self.hidden_shape))

    def __call__(self, x, state):
        V, I = mx.split(state, 2, axis=-1)
        spike = self._spike(V - self.threshold)
        reset = spike * self.threshold
        V = V - reset
        I = self._alpha() * I + x
        V = self._beta() * V + I - reset
        return spike, mx.concatenate([V, I], axis=-1)


# ---------------------------------------------------------------------------
# ALIF — Adaptive LIF
# ---------------------------------------------------------------------------


class ALIF(nn.Module):
    """
    Adaptive LIF (Bellec et al. 2018 LSNN):
        thresh   = threshold + T[t-1]
        V[t]     = beta * V[t-1] + x[t]
        spike[t] = Heaviside(V[t] - thresh)
        V[t]    -= spike[t] * thresh
        T[t]     = gamma * T[t-1] + (1 - gamma) * spike[t]

    State layout matches spyx: concatenate [V, T] along the last axis.
    """

    def __init__(self, hidden_shape, beta=None, gamma=None, threshold=1.0, activation=None, beta_init=None, gamma_init=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)
        self.threshold = threshold
        self._spike = activation if activation is not None else superspike()
        self._beta_fixed = None
        self._gamma_fixed = None
        beta_value = beta if beta is not None else beta_init
        gamma_value = gamma if gamma is not None else gamma_init
        if beta_value is not None:
            self._beta_fixed = float(beta_value)
        else:
            self.beta_logit = mx.full(self.hidden_shape, _LOGIT_09)
        if gamma_value is not None:
            self._gamma_fixed = float(gamma_value)
        else:
            self.gamma_logit = mx.full(self.hidden_shape, _LOGIT_09)

    def _beta(self):
        return self._beta_fixed if self._beta_fixed is not None else mx.sigmoid(self.beta_logit)

    def _gamma(self):
        return self._gamma_fixed if self._gamma_fixed is not None else mx.sigmoid(self.gamma_logit)

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + tuple(2 * v for v in self.hidden_shape))

    def __call__(self, x, state):
        V, T = mx.split(state, 2, axis=-1)
        beta = self._beta()
        gamma = self._gamma()
        thresh = self.threshold + T
        # Match spyx.ALIF: spike uses previous V and dynamic threshold.
        spike = self._spike(V - thresh)
        V = beta * V + x - spike * thresh
        T = gamma * T + (1.0 - gamma) * spike
        return spike, mx.concatenate([V, T], axis=-1)


# ---------------------------------------------------------------------------
# RLIF — Recurrent LIF
# ---------------------------------------------------------------------------


class RLIF(nn.Module):
    """
    Recurrent LIF matching spyx.nn.RLIF semantics:
        spike[t] = Heaviside(V[t-1] - threshold)
        feedback = spike[t] @ W_rec
        V[t] = beta * V[t-1] + x[t] + feedback - spike[t] * threshold

    State is membrane voltage V only.
    """

    def __init__(self, hidden_shape, beta=None, threshold=1.0, activation=None, beta_init=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)
        n = self.hidden_shape[0]
        self.threshold = threshold
        self._spike = activation if activation is not None else superspike()
        self.w_rec = mx.random.normal((n, n)) * (1.0 / math.sqrt(n))
        self._beta_fixed = None
        beta_value = beta if beta is not None else beta_init
        if beta_value is not None:
            self._beta_fixed = float(beta_value)
        else:
            self.beta_logit = mx.full(self.hidden_shape, _LOGIT_09)

    def _beta(self):
        return self._beta_fixed if self._beta_fixed is not None else mx.sigmoid(self.beta_logit)

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + self.hidden_shape)

    def __call__(self, x, V):
        spike = self._spike(V - self.threshold)
        feedback = spike @ self.w_rec
        V = self._beta() * V + x + feedback - spike * self.threshold
        return spike, V


# ---------------------------------------------------------------------------
# RIF — Recurrent Integrate-and-Fire
# ---------------------------------------------------------------------------


class RIF(nn.Module):
    """Recurrent IF matching spyx.nn.RIF semantics."""

    def __init__(self, hidden_shape, threshold=1.0, activation=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)
        n = self.hidden_shape[0]
        self.threshold = threshold
        self._spike = activation if activation is not None else superspike()
        self.w_rec = mx.random.normal((n, n)) * (1.0 / math.sqrt(n))

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + self.hidden_shape)

    def __call__(self, x, V):
        spike = self._spike(V - self.threshold)
        feedback = spike @ self.w_rec
        V = V + x + feedback - spike * self.threshold
        return spike, V


# ---------------------------------------------------------------------------
# RCuBaLIF — Recurrent Current-Based LIF
# ---------------------------------------------------------------------------


class RCuBaLIF(nn.Module):
    """Recurrent CuBaLIF matching spyx.nn.RCuBaLIF semantics."""

    def __init__(self, hidden_shape, alpha=None, beta=None, threshold=1.0, activation=None, alpha_init=None, beta_init=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)
        n = self.hidden_shape[0]
        self.threshold = threshold
        self._spike = activation if activation is not None else superspike()
        self.w_rec = mx.random.normal((n, n)) * (1.0 / math.sqrt(n))
        self._alpha_fixed = None
        self._beta_fixed = None
        alpha_value = alpha if alpha is not None else alpha_init
        beta_value = beta if beta is not None else beta_init
        if alpha_value is not None:
            self._alpha_fixed = float(alpha_value)
        else:
            self.alpha_logit = mx.full(self.hidden_shape, _LOGIT_09)
        if beta_value is not None:
            self._beta_fixed = float(beta_value)
        else:
            self.beta_logit = mx.full(self.hidden_shape, _LOGIT_09)

    def _alpha(self):
        return self._alpha_fixed if self._alpha_fixed is not None else mx.sigmoid(self.alpha_logit)

    def _beta(self):
        return self._beta_fixed if self._beta_fixed is not None else mx.sigmoid(self.beta_logit)

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + tuple(2 * v for v in self.hidden_shape))

    def __call__(self, x, state):
        V, I = mx.split(state, 2, axis=-1)
        spike = self._spike(V - self.threshold)
        V = V - spike * self.threshold
        feedback = spike @ self.w_rec
        I = self._alpha() * I + x + feedback
        V = self._beta() * V + I
        return spike, mx.concatenate([V, I], axis=-1)


class ActivityRegularization(nn.Module):
    """Track cumulative spikes similarly to spyx.nn.ActivityRegularization."""

    def __init__(self):
        super().__init__()
        self.spike_count = None

    def reset(self):
        self.spike_count = None

    def __call__(self, spikes):
        if self.spike_count is None:
            self.spike_count = mx.zeros_like(spikes)
        self.spike_count = self.spike_count + spikes
        return spikes


def PopulationCode(num_classes):
    """Population coding helper matching spyx.nn.PopulationCode signature."""

    def _pop_code(x):
        return mx.sum(mx.reshape(x, (-1, num_classes)), axis=-1)

    return _pop_code


def _infer_shape(
    x,
    size: Union[int, Sequence[int]],
    channel_axis: Optional[int] = -1,
) -> tuple[int, ...]:
    if isinstance(size, int):
        if channel_axis and not 0 <= abs(channel_axis) < x.ndim:
            raise ValueError(f"Invalid channel axis {channel_axis} for {x.shape}")
        if channel_axis and channel_axis < 0:
            channel_axis = x.ndim + channel_axis
        return (1,) + tuple(size if d != channel_axis else 1 for d in range(1, x.ndim))
    if len(size) < x.ndim:
        return (1,) * (x.ndim - len(size)) + tuple(size)
    return tuple(size)


_VMAP_SHAPE_INFERENCE_WARNING = (
    "When running under vmap, passing an int (except for 1) for window_shape or strides "
    "can infer incorrect shapes if batch dims are hidden. Prefer full unbatched tuples."
)


def _warn_if_unsafe(window_shape, strides):
    unsafe = lambda size: isinstance(size, int) and size != 1
    if unsafe(window_shape) or unsafe(strides):
        warnings.warn(_VMAP_SHAPE_INFERENCE_WARNING, DeprecationWarning)


def _same_padding_1d(n: int, k: int, s: int) -> tuple[int, int]:
    out = math.ceil(n / s)
    pad = max((out - 1) * s + k - n, 0)
    return pad // 2, pad - (pad // 2)


def sum_pool(
    value,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: str,
    channel_axis: Optional[int] = -1,
):
    """Sum pool with spyx.nn-compatible signature."""
    if padding not in ("SAME", "VALID"):
        raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")

    _warn_if_unsafe(window_shape, strides)
    window_shape = _infer_shape(value, window_shape, channel_axis)
    strides = _infer_shape(value, strides, channel_axis)

    x = np.array(value)
    if padding == "SAME":
        pads = [_same_padding_1d(int(n), int(k), int(s)) for n, k, s in zip(x.shape, window_shape, strides)]
        x = np.pad(x, pads, mode="constant")

    view = np.lib.stride_tricks.sliding_window_view(x, window_shape)
    stride_slices = tuple(slice(None, None, int(s)) for s in strides)
    view = view[stride_slices + (slice(None),) * x.ndim]
    out = view.sum(axis=tuple(range(x.ndim, 2 * x.ndim)))
    return mx.array(out)


class SumPool(nn.Module):
    """Sum pooling module matching spyx.nn.SumPool API."""

    def __init__(
        self,
        window_shape: Union[int, Sequence[int]],
        strides: Union[int, Sequence[int]],
        padding: str,
        channel_axis: Optional[int] = -1,
    ):
        super().__init__()
        self.window_shape = window_shape
        self.strides = strides
        self.padding = padding
        self.channel_axis = channel_axis

    def __call__(self, value):
        return sum_pool(value, self.window_shape, self.strides, self.padding, self.channel_axis)


# ---------------------------------------------------------------------------
# Utility: unroll a sequence of (linear, neuron) pairs over time
# ---------------------------------------------------------------------------


def dynamic_unroll(cells, x_seq, initial_states):
    """
    Unroll a list of (nn.Linear, Neuron) pairs over a time sequence.

    Args:
        cells: list of (nn.Linear, Neuron) pairs
        x_seq: (T, batch, n_input)
        initial_states: list of initial states, one per neuron

    Returns:
        traces: list of stacked outputs, shape (T, batch, n_out) per cell
        final_states: list of final states
    """
    n_t = x_seq.shape[0]
    states = list(initial_states)
    all_outputs = [[] for _ in range(len(cells))]

    for t in range(n_t):
        x = x_seq[t]
        for i, (linear, neuron) in enumerate(cells):
            x = linear(x)
            out, states[i] = neuron(x, states[i])
            all_outputs[i].append(out)
            x = out

    traces = [mx.stack(outs, axis=0) for outs in all_outputs]
    return traces, states
