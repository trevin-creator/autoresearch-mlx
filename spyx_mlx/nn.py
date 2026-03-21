"""
Spiking neuron models for MLX.
MLX port of spyx.nn — uses mlx.nn.Module instead of Haiku RNNCore.

All neurons follow the same interface:
    out, new_state = neuron(x, state)
    state = neuron.initial_state(batch_size)

Neurons are stateless modules (weights only). The membrane state is passed
explicitly and accumulated by the caller's time loop, which lets MLX compile
the full unrolled graph efficiently.
"""

import mlx.core as mx
import mlx.nn as nn

from .axn import superspike


# ---------------------------------------------------------------------------
# IF — Integrate-and-Fire
# ---------------------------------------------------------------------------

class IF(nn.Module):
    """
    Simple Integrate-and-Fire neuron.
        V[t] = V[t-1] + x[t] - spike[t-1] * threshold
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
        V[t] = beta * V[t-1] + x[t] - spike[t-1] * threshold

    beta is learnable (one value per neuron) when beta_init is None,
    otherwise fixed to the given scalar.
    """

    def __init__(self, hidden_shape, beta_init=None, threshold=1.0, activation=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)
        self.threshold = threshold
        self._spike = activation if activation is not None else superspike()

        if beta_init is None:
            # learnable: initialise near 0.9 in logit space
            import math
            init_val = math.log(0.9 / 0.1)  # sigmoid^{-1}(0.9)
            self.beta_logit = mx.full(self.hidden_shape, init_val)
        else:
            self.beta_logit = None
            self._beta_fixed = float(beta_init)

    def _beta(self):
        if self.beta_logit is not None:
            return mx.sigmoid(self.beta_logit)
        return self._beta_fixed

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + self.hidden_shape)

    def __call__(self, x, V):
        beta = self._beta()
        V = beta * V + x
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
    """

    def __init__(self, hidden_shape, beta_init=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)

        if beta_init is None:
            import math
            init_val = math.log(0.9 / 0.1)
            self.beta_logit = mx.full(self.hidden_shape, init_val)
        else:
            self.beta_logit = None
            self._beta_fixed = float(beta_init)

    def _beta(self):
        if self.beta_logit is not None:
            return mx.sigmoid(self.beta_logit)
        return self._beta_fixed

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + self.hidden_shape)

    def __call__(self, x, V):
        beta = self._beta()
        V = beta * V + x
        return V, V  # (output, new_state) — output is the voltage trace


# ---------------------------------------------------------------------------
# CuBaLIF — Current-Based LIF (dual time constants)
# ---------------------------------------------------------------------------

class CuBaLIF(nn.Module):
    """
    Current-Based LIF: separates synaptic current and membrane dynamics.
        I[t]  = alpha * I[t-1] + x[t]
        V[t]  = beta  * V[t-1] + I[t] - spike[t-1] * threshold
    State = [V, I] concatenated along last axis.
    """

    def __init__(self, hidden_shape, alpha_init=None, beta_init=None, threshold=1.0, activation=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)
        self.threshold = threshold
        self._spike = activation if activation is not None else superspike()

        import math
        for name, val in [("alpha_logit", alpha_init), ("beta_logit", beta_init)]:
            if val is None:
                init = math.log(0.9 / 0.1)
                setattr(self, name, mx.full(self.hidden_shape, init))
            else:
                setattr(self, name, None)
                setattr(self, f"_{name[:-6]}_fixed", float(val))

    def _alpha(self):
        if self.alpha_logit is not None:
            return mx.sigmoid(self.alpha_logit)
        return self._alpha_fixed

    def _beta(self):
        if self.beta_logit is not None:
            return mx.sigmoid(self.beta_logit)
        return self._beta_fixed

    def initial_state(self, batch_size):
        return mx.zeros((batch_size,) + self.hidden_shape + (2,))

    def __call__(self, x, state):
        V, I = state[..., 0], state[..., 1]
        alpha = self._alpha()
        beta = self._beta()
        I = alpha * I + x
        V = beta * V + I
        spike = self._spike(V - self.threshold)
        V = V - spike * self.threshold
        new_state = mx.stack([V, I], axis=-1)
        return spike, new_state


# ---------------------------------------------------------------------------
# ALIF — Adaptive LIF
# ---------------------------------------------------------------------------

class ALIF(nn.Module):
    """
    Adaptive LIF: threshold adapts based on recent spike history.
        thresh = threshold + T[t-1]
        spike   = Heaviside(V[t-1] + x[t] - thresh)
        V[t]    = V[t-1] + x[t] - spike * thresh
        T[t]    = gamma * T[t-1] + (1 - gamma) * spike
    State = [V, T] concatenated along last axis.
    """

    def __init__(self, hidden_shape, beta_init=None, gamma_init=None, threshold=1.0, activation=None):
        super().__init__()
        self.hidden_shape = (hidden_shape,) if isinstance(hidden_shape, int) else tuple(hidden_shape)
        self.threshold = threshold
        self._spike = activation if activation is not None else superspike()

        import math
        for name, val in [("beta_logit", beta_init), ("gamma_logit", gamma_init)]:
            if val is None:
                init = math.log(0.9 / 0.1)
                setattr(self, name, mx.full(self.hidden_shape, init))
            else:
                setattr(self, name, None)
                setattr(self, f"_{name[:-6]}_fixed", float(val))

    def _beta(self):
        if self.beta_logit is not None:
            return mx.sigmoid(self.beta_logit)
        return self._beta_fixed

    def _gamma(self):
        if self.gamma_logit is not None:
            return mx.sigmoid(self.gamma_logit)
        return self._gamma_fixed

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
        new_state = mx.stack([V, T], axis=-1)
        return spike, new_state


# ---------------------------------------------------------------------------
# Recurrent variants
# ---------------------------------------------------------------------------

class RLIF(nn.Module):
    """
    Recurrent LIF: previous spikes feed back through a learnable weight matrix.
        V[t] = beta * V[t-1] + x[t] + spike[t-1] @ W_rec - spike[t-1] * threshold
    State = [V, spike_prev] concatenated along last axis.
    """

    def __init__(self, hidden_shape, beta_init=None, threshold=1.0, activation=None):
        super().__init__()
        n = hidden_shape if isinstance(hidden_shape, int) else hidden_shape[0]
        self.n = n
        self.threshold = threshold
        self._spike = activation if activation is not None else superspike()
        self.w_rec = nn.Linear(n, n, bias=False)

        import math
        if beta_init is None:
            init_val = math.log(0.9 / 0.1)
            self.beta_logit = mx.full((n,), init_val)
        else:
            self.beta_logit = None
            self._beta_fixed = float(beta_init)

    def _beta(self):
        if self.beta_logit is not None:
            return mx.sigmoid(self.beta_logit)
        return self._beta_fixed

    def initial_state(self, batch_size):
        return mx.zeros((batch_size, self.n, 2))

    def __call__(self, x, state):
        V, s_prev = state[..., 0], state[..., 1]
        beta = self._beta()
        V = beta * V + x + self.w_rec(s_prev)
        spike = self._spike(V - self.threshold)
        V = V - spike * self.threshold
        new_state = mx.stack([V, spike], axis=-1)
        return spike, new_state


# ---------------------------------------------------------------------------
# Utility: unroll a sequence of (neuron, linear) layer pairs over time
# ---------------------------------------------------------------------------

def dynamic_unroll(cells, x_seq, initial_states):
    """
    Unroll a list of (linear, neuron) layer pairs over a time sequence.

    Args:
        cells: list of (nn.Linear, Neuron) pairs, plus a final (nn.Linear, LI) pair
        x_seq: input tensor of shape (T, batch, n_input)
        initial_states: list of initial states, one per neuron

    Returns:
        traces: list of output voltage traces, shape (T, batch, n_out) per readout
        final_states: list of final states
    """
    T = x_seq.shape[0]
    states = list(initial_states)
    all_outputs = [[] for _ in range(len(cells))]

    for t in range(T):
        x = x_seq[t]
        for i, (linear, neuron) in enumerate(cells):
            x = linear(x)
            out, states[i] = neuron(x, states[i])
            all_outputs[i].append(out)
            x = out

    traces = [mx.stack(outs, axis=0) for outs in all_outputs]
    return traces, states
