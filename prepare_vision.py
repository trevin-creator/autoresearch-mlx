"""
Vision SNN evaluation harness for DVSGesture classification.
Compatible with train_vision.py and search_vision_optuna.py.
"""

import haiku as hk
import jax
import jax.numpy as jnp


def build_snn_vision(
    n_hidden: int = 128,
    kernel_size: int = 3,
    use_binary: bool = False,
    dropout: float = 0.0,
    surrogate: str = "superspike",
):
    """
    Build an SNN for DVSGesture: Conv2D → LIF pool → Dense → Classification.

    Args:
        n_hidden: Hidden layer size in dense layers
        kernel_size: Conv2D kernel size
        use_binary: Use binary quantization on Conv2D weights
        dropout: Dropout rate (0 = no dropout)
        surrogate: Surrogate gradient type

    Returns:
        Callable taking (spikes: THWC, key: PRNGKey) → (logits: 11,)
    """
    from spyx.axn import sigmoid, straight_through, superspike

    surrogate_fn = {
        "superspike": superspike,
        "sigmoid": lambda: sigmoid(k=10),
        "straight_through": straight_through,
    }.get(surrogate, superspike)()

    def forward(spikes):
        # spikes: (T, H, W, C) → unpack bits if needed
        if spikes.dtype == jnp.uint8 and spikes.shape[-1] == 2:
            spikes = jnp.unpackbits(spikes, axis=0)[:65536]  # Unpack temporal dimension

        # Conv2D + LIF layer 1
        x = hk.Conv2D(64, (kernel_size,), padding="SAME", name="conv1")(spikes)
        x = jnp.transpose(x, (0, 3, 1, 2))  # THWC → TCHW for LIF

        # Simulate LIF neuron state across time
        h1 = jnp.zeros((x.shape[1], 64, x.shape[3], x.shape[4]))
        spikes1_list = []
        for t in range(x.shape[0]):
            spike = jnp.where(h1 > 1.0, 1.0, 0.0)
            h1 = 0.9 * h1 + x[t] - spike
            spikes1_list.append(spike)
        spikes1 = jnp.stack(spikes1_list)  # (T, B, 64, H, W)

        # Spatial pooling
        x = jnp.max(spikes1, axis=0)  # (B, 64, H, W)
        x = jnp.reshape(x, (x.shape[0], -1))  # Flatten spatial

        # Dense classifier
        x = hk.Linear(n_hidden, name="dense1")(x)
        x = jnp.where(x > 0, x, 0.0)  # ReLU
        x = hk.Linear(11, name="dense2")(x)  # 11 gesture classes

        return x

    return hk.transform_with_state(forward)


def evaluate_vision(params, state, x_val, y_val, batch_size=32):
    """
    Evaluate vision SNN on validation set.

    Args:
        params: Haiku params
        state: Haiku state
        x_val: Validation spikes (N, T, H, W, 2)
        y_val: Validation labels (N,)
        batch_size: Batch size for evaluation

    Returns:
        Accuracy (0-1)
    """
    snn, _ = build_snn_vision()

    def forward_no_state(x):
        # Simplified forward pass (no state threading)
        logits = snn.apply(params, state, jax.random.PRNGKey(0), x)[0]
        return jnp.argmax(logits, axis=-1)

    n_batches = len(y_val) // batch_size
    correct = 0
    total = 0

    for b in range(n_batches):
        batch_x = x_val[b * batch_size : (b + 1) * batch_size]
        batch_y = y_val[b * batch_size : (b + 1) * batch_size]

        preds = jax.vmap(forward_no_state)(batch_x)
        correct += jnp.sum(preds == batch_y)
        total += len(batch_y)

    return float(correct) / float(total)
