"""
Loss and utility functions for spiking neural networks.
MLX port of spyx.fn.
"""

import mlx.core as mx
import mlx.nn as nn_mlx


def integral_crossentropy(traces, targets, smoothing=0.3):
    """
    Cross-entropy loss on the time-integrated voltage traces.

    Args:
        traces: (T, batch, n_classes) — readout voltages across time
        targets: (batch,) int32 class indices
        smoothing: label smoothing coefficient

    Returns:
        scalar loss
    """
    logits = traces.sum(axis=0)  # (batch, n_classes)
    n_classes = logits.shape[-1]
    loss = nn_mlx.losses.cross_entropy(logits, targets, label_smoothing=smoothing)
    return loss.mean()


def integral_accuracy(traces, targets):
    """
    Classification accuracy from time-integrated voltage traces.

    Args:
        traces: (T, batch, n_classes)
        targets: (batch,) int32 class indices

    Returns:
        accuracy: scalar in [0, 1]
    """
    logits = traces.sum(axis=0)  # (batch, n_classes)
    preds = logits.argmax(axis=-1)
    return (preds == targets).astype(mx.float32).mean()


def silence_reg(traces, min_rate=0.01):
    """
    Penalise neurons that spike below min_rate on average.
    Encourages all neurons to participate.

    Args:
        traces: (T, batch, n_neurons) spike tensor
        min_rate: minimum desired mean spike rate

    Returns:
        scalar regularisation loss
    """
    mean_rates = traces.mean(axis=(0, 1))  # (n_neurons,)
    deficit = mx.maximum(0.0, min_rate - mean_rates)
    return (deficit ** 2).sum()


def sparsity_reg(traces, max_rate=0.1):
    """
    Penalise neurons that spike above max_rate on average.
    Encourages sparse activity.

    Args:
        traces: (T, batch, n_neurons)
        max_rate: maximum desired mean spike rate

    Returns:
        scalar regularisation loss
    """
    mean_rates = traces.mean(axis=(0, 1))
    excess = mx.maximum(0.0, mean_rates - max_rate)
    return (excess ** 2).sum()
