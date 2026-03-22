"""
Loss and utility functions for spiking neural networks.
MLX port of spyx.fn.
"""

from collections.abc import Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn_mlx


def _iter_arrays(tree_like):
    if isinstance(tree_like, Mapping):
        for v in tree_like.values():
            yield from _iter_arrays(v)
        return
    if isinstance(tree_like, Sequence) and not isinstance(tree_like, (str, bytes)):
        for v in tree_like:
            yield from _iter_arrays(v)
        return
    if hasattr(tree_like, "shape"):
        yield tree_like


def _huber_loss(x, delta=1.0):
    abs_x = mx.abs(x)
    quad = 0.5 * (x ** 2)
    lin = delta * (abs_x - 0.5 * delta)
    return mx.where(abs_x <= delta, quad, lin)


def silence_reg(min_spikes):
    """Spyx-compatible higher-order silence regularizer."""

    def _loss(x):
        flat = mx.reshape(x, (x.shape[0], -1))
        return (mx.maximum(0.0, min_spikes - mx.mean(flat, axis=0))) ** 2

    def _call(spikes):
        losses = [_loss(x) for x in _iter_arrays(spikes)]
        if not losses:
            return mx.array(0.0)
        return mx.sum(mx.concatenate([mx.ravel(v) for v in losses]))

    return _call


def sparsity_reg(max_spikes, norm=None):
    """Spyx-compatible higher-order sparsity regularizer."""
    norm = _huber_loss if norm is None else norm

    def _loss(x):
        flat = mx.reshape(x, (x.shape[0], -1))
        return norm(mx.maximum(0.0, mx.mean(flat, axis=-1) - max_spikes))

    def _call(spikes):
        losses = [_loss(x) for x in _iter_arrays(spikes)]
        if not losses:
            return mx.array(0.0)
        return mx.sum(mx.concatenate([mx.ravel(v) for v in losses]))

    return _call


def integral_accuracy(time_axis=1):
    """Spyx-compatible higher-order integral accuracy."""

    def _integral_accuracy(traces, targets):
        logits = mx.sum(traces, axis=time_axis)
        preds = mx.argmax(logits, axis=-1)
        return mx.mean((preds == targets).astype(mx.float32)), preds

    return _integral_accuracy


def _integral_crossentropy_impl(traces, targets, smoothing=0.3, time_axis=1):
    logits = mx.sum(traces, axis=time_axis)
    loss = nn_mlx.losses.cross_entropy(logits, targets, label_smoothing=smoothing)
    return mx.mean(loss)


def integral_crossentropy(traces=None, targets=None, smoothing=0.3, time_axis=1):
    """
    Spyx-compatible integral cross entropy.

    Supports both styles:
    - `integral_crossentropy(smoothing=..., time_axis=...) -> callable`
    - `integral_crossentropy(traces, targets, smoothing=..., time_axis=...)`

    Args:
        traces: optional tensor for direct-call mode
        targets: optional labels for direct-call mode
        smoothing: label smoothing coefficient
        time_axis: temporal integration axis

    Returns:
        scalar loss or callable
    """
    if traces is not None and targets is not None:
        return _integral_crossentropy_impl(traces, targets, smoothing=smoothing, time_axis=time_axis)

    def _integral_crossentropy(traces_, targets_):
        return _integral_crossentropy_impl(traces_, targets_, smoothing=smoothing, time_axis=time_axis)

    return _integral_crossentropy


def mse_spikerate(sparsity=0.25, smoothing=0.0, time_axis=1):
    """Spyx-compatible MSE spike-rate loss."""

    def _mse_spikerate(traces, targets):
        t = traces.shape[time_axis]
        logits = mx.mean(traces, axis=time_axis)
        n_classes = logits.shape[-1]
        one_hot = (mx.arange(n_classes)[None, :] == targets[:, None]).astype(logits.dtype)
        labels = (1.0 - smoothing) * one_hot + smoothing / n_classes
        return mx.mean((logits - labels * sparsity * t) ** 2)

    return _mse_spikerate
