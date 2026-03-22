"""MLX data utilities mirroring spyx.data signatures."""

from __future__ import annotations

import numpy as np
import mlx.core as mx


def shift_augment(max_shift=10, axes=(-1,)):
    """Roll input along given axes by random shifts in [-max_shift, max_shift)."""

    def _shift(data, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        shift = rng.integers(-max_shift, max_shift, size=(len(axes),))
        arr = np.array(data)
        out = np.roll(arr, shift=tuple(int(s) for s in shift), axis=axes)
        return mx.array(out)

    return _shift


def shuffler(dataset, batch_size):
    """Build a dataset shuffler returning batch-shaped tensors."""
    x, y = dataset
    x_np = np.array(x)
    y_np = np.array(y)
    cutoff = (y_np.shape[0] // batch_size) * batch_size
    data_shape = (-1, batch_size) + x_np.shape[1:]

    def _shuffle(dataset_, shuffle_rng=None):
        del dataset_  # kept for API parity with spyx.data
        if shuffle_rng is None:
            shuffle_rng = np.random.default_rng()
        indices = shuffle_rng.permutation(y_np.shape[0])[:cutoff]
        obs, labels = x_np[indices], y_np[indices]
        obs = np.reshape(obs, data_shape)
        labels = np.reshape(labels, (-1, batch_size))
        return mx.array(obs), mx.array(labels)

    return _shuffle


def rate_code(num_steps, max_r=0.75):
    """Rate-code values in [0,1] into Bernoulli spikes over a repeated time axis."""

    def _call(data, key=None):
        del key  # compatibility arg
        data_np = np.array(data, dtype=np.float32)
        unrolled = np.repeat(data_np, num_steps, axis=1)
        probs = np.clip(unrolled * max_r, 0.0, 1.0)
        spikes = (np.random.random(size=probs.shape) < probs).astype(np.uint8)
        return mx.array(spikes)

    return _call


def angle_code(neuron_count, min_val, max_val):
    """Convert continuous values to one-hot bins between min and max."""
    neurons = np.linspace(min_val, max_val, neuron_count)

    def _call(obs):
        obs_np = np.array(obs)
        digital = np.digitize(obs_np, neurons)
        digital = np.clip(digital, 0, neuron_count - 1)
        eye = np.eye(neuron_count, dtype=np.float32)
        return mx.array(eye[digital])

    return _call
