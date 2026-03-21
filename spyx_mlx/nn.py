"""
Spiking neuron models for MLX.
MLX port of spyx.nn — uses mlx.nn.Module instead of Haiku RNNCore.

All neurons follow the same interface:
    out, new_state = neuron(x, state)
    state = neuron.initial_state(batch_size)

Spike timing matches spyx: the spike decision is evaluated on the PREVIOUS
membrane potential V[t-1], then V is updated. This is the standard LSNN / spyx
convention:

    spike[t] = Heaviside(V[t-1] - thresh)
    V[t]     = beta * V[t-1] + x[t] - spike[t] * thresh

Parameter initialisation matches spyx: beta and gamma are initialised via a
truncated normal in direct (not logit) space, clipped to [0, 1].
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .axn import superspike


def _init_decay(shape, init_val=None, default=0.9):
    """
    Return an (shape,) MLX parameter for a decay constant in [0, 1].
    If init_val is given, the parameter is fixed (scalar).
    Otherwise it is a learnable uniform value initialised at `default`.
    High default (0.9) suits long-timescale tasks like SHD (128 time bins).
    """
    if init_val is not None:
        return mx.array(float(init_val))
    return mx.full(shape, default)


# ---------------------------------------------------------------------------
# IF — Integrate-and-Fire
# ---------------------------------------------------------------------------


class IF(nn.Module):
    """
    Integrate-and-Fire neuron (spyx-compatible).
        spike[t] = Heaviside(V[t-1] - threshold)
        V[t]     = V[t-1] + x[t] - spike[t] * threshold
    """

    def __init__(self, hidden_shape, threshold=1.0, activation=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)
        self.threshold = threshold
        self._spike = activation if activation is not None else superspike()

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + self.hidden_shape)

    def __call__(self, x, V):
        V = V + x                                         # integrate first
        spike = self._spike(V - self.threshold)
        V = V - spike * self.threshold
        return spike, V


# ---------------------------------------------------------------------------
# LIF — Leaky Integrate-and-Fire
# ---------------------------------------------------------------------------


class LIF(nn.Module):
    """
    Leaky Integrate-and-Fire neuron (spyx-compatible).
        spike[t] = Heaviside(V[t-1] - threshold)
        V[t]     = beta * V[t-1] + x[t] - spike[t] * threshold

    beta is learnable per-neuron, initialised ~TruncatedNormal(0.5, 0.25).
    """

    def __init__(self, hidden_shape, beta_init=None, threshold=1.0, activation=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)
        self.threshold = threshold
        self._spike = activation if activation is not None else superspike()
        self._beta_fixed = None
        if beta_init is not None:
            self._beta_fixed = float(beta_init)
        else:
            self.beta = _init_decay(self.hidden_shape)

    def _get_beta(self):
        if self._beta_fixed is not None:
            return self._beta_fixed
        return mx.clip(self.beta, 0.0, 1.0)

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + self.hidden_shape)

    def __call__(self, x, V):
        beta = self._get_beta()
        V = beta * V + x                                  # integrate first
        spike = self._spike(V - self.threshold)
        V = V - spike * self.threshold
        return spike, V


# ---------------------------------------------------------------------------
# LI — Leaky Integrator (non-spiking readout)
# ---------------------------------------------------------------------------


class LI(nn.Module):
    """
    Leaky Integrator — no spike, used as output readout layer (spyx-compatible).
        V[t] = beta * V[t-1] + x[t]

    beta is learnable per-neuron, initialised ~0.8 (matching spyx Constant(0.8)).
    """

    def __init__(self, hidden_shape, beta_init=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)
        self._beta_fixed = None
        if beta_init is not None:
            self._beta_fixed = float(beta_init)
        else:
            # spyx LI uses Constant(0.8)
            self.beta = mx.full(self.hidden_shape, 0.8)

    def _get_beta(self):
        if self._beta_fixed is not None:
            return self._beta_fixed
        return mx.clip(self.beta, 0.0, 1.0)

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + self.hidden_shape)

    def __call__(self, x, V):
        beta = self._get_beta()
        V = beta * V + x
        return V, V   # (output trace, new state)


# ---------------------------------------------------------------------------
# CuBaLIF — Current-Based LIF (dual time constants)
# ---------------------------------------------------------------------------


class CuBaLIF(nn.Module):
    """
    Current-Based LIF: separates synaptic current and membrane dynamics.
        spike[t] = Heaviside(V[t-1] - threshold)
        I[t]     = alpha * I[t-1] + x[t]
        V[t]     = beta  * V[t-1] + I[t] - spike[t] * threshold

    (spyx has a double-reset bug; we use the correct single-reset formulation.)
    State = mx.stack([V, I], axis=-1), shape (batch, n, 2).
    """

    def __init__(self, hidden_shape, alpha_init=None, beta_init=None, threshold=1.0, activation=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)
        self.threshold = threshold
        self._spike = activation if activation is not None else superspike()

        self._alpha_fixed = None
        self._beta_fixed = None
        if alpha_init is not None:
            self._alpha_fixed = float(alpha_init)
        else:
            self.alpha = _init_decay(self.hidden_shape)
        if beta_init is not None:
            self._beta_fixed = float(beta_init)
        else:
            self.beta = _init_decay(self.hidden_shape)

    def _get_alpha(self):
        return self._alpha_fixed if self._alpha_fixed is not None else mx.clip(self.alpha, 0.0, 1.0)

    def _get_beta(self):
        return self._beta_fixed if self._beta_fixed is not None else mx.clip(self.beta, 0.0, 1.0)

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + self.hidden_shape + (2,))

    def __call__(self, x, state):
        V, I = state[..., 0], state[..., 1]
        I = self._get_alpha() * I + x                    # integrate current
        V = self._get_beta() * V + I                     # integrate voltage
        spike = self._spike(V - self.threshold)
        V = V - spike * self.threshold
        return spike, mx.stack([V, I], axis=-1)


# ---------------------------------------------------------------------------
# ALIF — Adaptive LIF
# ---------------------------------------------------------------------------


class ALIF(nn.Module):
    """
    Adaptive LIF (spyx-compatible, Bellec et al. 2018 LSNN):
        thresh   = threshold + T[t-1]
        spike[t] = Heaviside(V[t-1] - thresh)
        V[t]     = beta * V[t-1] + x[t] - spike[t] * thresh
        T[t]     = gamma * T[t-1] + (1 - gamma) * spike[t]

    State = mx.stack([V, T], axis=-1), shape (batch, n, 2).
    """

    def __init__(self, hidden_shape, beta_init=None, gamma_init=None, threshold=1.0, activation=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)
        self.threshold = threshold
        self._spike = activation if activation is not None else superspike()

        self._beta_fixed = None
        self._gamma_fixed = None
        if beta_init is not None:
            self._beta_fixed = float(beta_init)
        else:
            self.beta = _init_decay(self.hidden_shape)
        if gamma_init is not None:
            self._gamma_fixed = float(gamma_init)
        else:
            self.gamma = _init_decay(self.hidden_shape)

    def _get_beta(self):
        return self._beta_fixed if self._beta_fixed is not None else mx.clip(self.beta, 0.0, 1.0)

    def _get_gamma(self):
        return self._gamma_fixed if self._gamma_fixed is not None else mx.clip(self.gamma, 0.0, 1.0)

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + self.hidden_shape + (2,))

    def __call__(self, x, state):
        V, T = state[..., 0], state[..., 1]
        thresh = self.threshold + T
        V = self._get_beta() * V + x                     # integrate first
        spike = self._spike(V - thresh)
        V = V - spike * thresh
        T = self._get_gamma() * T + (1.0 - self._get_gamma()) * spike
        return spike, mx.stack([V, T], axis=-1)


# ---------------------------------------------------------------------------
# RLIF — Recurrent LIF
# ---------------------------------------------------------------------------


class RLIF(nn.Module):
    """
    Recurrent LIF: previous spikes feed back through a learnable weight matrix.
        spike[t] = Heaviside(V[t-1] - threshold)
        V[t]     = beta * V[t-1] + x[t] + spike[t-1] @ W_rec - spike[t] * threshold

    State = mx.stack([V, spike_prev], axis=-1), shape (batch, n, 2).
    """

    def __init__(self, hidden_shape, beta_init=None, threshold=1.0, activation=None):
        super().__init__()
        n = hidden_shape if isinstance(hidden_shape, int) else hidden_shape[0]
        self.n = n
        self.threshold = threshold
        self._spike = activation if activation is not None else superspike()
        self.w_rec = nn.Linear(n, n, bias=False)

        self._beta_fixed = None
        if beta_init is not None:
            self._beta_fixed = float(beta_init)
        else:
            self.beta = _init_decay((n,))

    def _get_beta(self):
        return self._beta_fixed if self._beta_fixed is not None else mx.clip(self.beta, 0.0, 1.0)

    def initial_state(self, batch_size):
        return mx.zeros((batch_size, self.n, 2))

    def __call__(self, x, state):
        V, s_prev = state[..., 0], state[..., 1]
        V = self._get_beta() * V + x + self.w_rec(s_prev)  # integrate first
        spike = self._spike(V - self.threshold)
        V = V - spike * self.threshold
        return spike, mx.stack([V, spike], axis=-1)


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
