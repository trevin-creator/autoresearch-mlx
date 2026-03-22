"""
Spiking neuron models for MLX.
MLX port of spyx.nn — uses mlx.nn.Module instead of Haiku RNNCore.

All neurons follow the same interface:
    out, new_state = neuron(x, state)
    state = neuron.initial_state(batch_size)

Design decisions vs spyx:
- Spike timing: integrate-then-spike (V = beta*V + x; spike on new V) rather
  than spyx's spike-then-update. Empirically better on SHD with 128 time bins.
- Beta parametrised in logit space (sigmoid reparametrisation). This constrains
  beta to (0,1) and provides implicit regularisation: the gradient through
  sigmoid at beta=0.9 is 0.09, slowing drift away from long-memory values.
- Default beta/gamma init: logit(0.9) → fast training convergence on SHD.
- CuBaLIF: single reset (spyx has a double-reset bug).
- RLIF: uses previous-timestep spikes for feedback (no circular dependency).
"""

import math

import mlx.core as mx
import mlx.nn as nn

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
        V = V + x
        spike = self._spike(V - self.threshold)
        V = V - spike * self.threshold
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

    def __init__(self, hidden_shape, beta_init=None, threshold=1.0, activation=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)
        self.threshold = threshold
        self._spike = activation if activation is not None else superspike()
        self._beta_fixed = None
        if beta_init is not None:
            self._beta_fixed = float(beta_init)
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
        spike = self._spike(V - self.threshold)
        V = V - spike * self.threshold
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

    def __init__(self, hidden_shape, beta_init=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)
        self._beta_fixed = None
        if beta_init is not None:
            self._beta_fixed = float(beta_init)
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

    Single reset (spyx has a double-reset bug).
    State = mx.stack([V, I], axis=-1).
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
            self.alpha_logit = mx.full(self.hidden_shape, _LOGIT_09)
        if beta_init is not None:
            self._beta_fixed = float(beta_init)
        else:
            self.beta_logit = mx.full(self.hidden_shape, _LOGIT_09)

    def _alpha(self):
        return self._alpha_fixed if self._alpha_fixed is not None else mx.sigmoid(self.alpha_logit)

    def _beta(self):
        return self._beta_fixed if self._beta_fixed is not None else mx.sigmoid(self.beta_logit)

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + self.hidden_shape + (2,))

    def __call__(self, x, state):
        V, I = state[..., 0], state[..., 1]
        I = self._alpha() * I + x
        V = self._beta() * V + I
        spike = self._spike(V - self.threshold)
        V = V - spike * self.threshold
        return spike, mx.stack([V, I], axis=-1)


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

    State = mx.stack([V, T], axis=-1).
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
            self.beta_logit = mx.full(self.hidden_shape, _LOGIT_09)
        if gamma_init is not None:
            self._gamma_fixed = float(gamma_init)
        else:
            self.gamma_logit = mx.full(self.hidden_shape, _LOGIT_09)

    def _beta(self):
        return self._beta_fixed if self._beta_fixed is not None else mx.sigmoid(self.beta_logit)

    def _gamma(self):
        return self._gamma_fixed if self._gamma_fixed is not None else mx.sigmoid(self.gamma_logit)

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + self.hidden_shape + (2,))

    def __call__(self, x, state):
        V, T = state[..., 0], state[..., 1]
        beta = self._beta()
        gamma = self._gamma()
        thresh = self.threshold + T
        V = beta * V + x
        spike = self._spike(V - thresh)
        V = V - spike * thresh
        T = gamma * T + (1.0 - gamma) * spike
        return spike, mx.stack([V, T], axis=-1)


# ---------------------------------------------------------------------------
# RLIF — Recurrent LIF
# ---------------------------------------------------------------------------


class RLIF(nn.Module):
    """
    Recurrent LIF: previous-timestep spikes feed back through a learnable W_rec.
        V[t] = beta * V[t-1] + x[t] + spike[t-1] @ W_rec
        spike[t] = Heaviside(V[t] - threshold)
        V[t] -= spike[t] * threshold

    State = mx.stack([V, spike_prev], axis=-1).
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
            self.beta_logit = mx.full((n,), _LOGIT_09)

    def _beta(self):
        return self._beta_fixed if self._beta_fixed is not None else mx.sigmoid(self.beta_logit)

    def initial_state(self, batch_size):
        return mx.zeros((batch_size, self.n, 2))

    def __call__(self, x, state):
        V, s_prev = state[..., 0], state[..., 1]
        V = self._beta() * V + x + self.w_rec(s_prev)
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
