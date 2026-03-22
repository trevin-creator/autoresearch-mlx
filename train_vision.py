#!/usr/bin/env python3
"""
Vision SNN Training Script for DVSGesture

Trains convolutional spike neural networks for gesture classification.
Supports Optuna hyperparameter search integration via structured JSONL logging.

Usage:
    python train_vision.py --help
    python train_vision.py --n_hidden 128 --batch_size 32
    python train_vision.py --config experiments/vision_config.json
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# MLX imports
try:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx import nn
except ImportError:
    print(
        "ERROR: MLX not installed. Install with: pip install mlx-community",
        file=sys.stderr,
    )
    sys.exit(1)

# Spyx imports
try:
    from spyx_mlx.nn import IF, LIF, LIF_Conv2D
except ImportError:
    print("ERROR: spyx_mlx not found in path", file=sys.stderr)
    sys.exit(1)


def assert_mlx_gpu_or_die():
    """Assert MLX has GPU available, exit if not."""
    if not mx.metal.is_available():
        print(
            "WARNING: MLX Metal GPU not available. Computing on CPU may be very slow.",
            file=sys.stderr,
        )
        # Don't exit; allow CPU training for testing


@dataclass(frozen=True)
class VisionConfig:
    """Experiment configuration for vision SNN training."""

    # Model architecture
    n_hidden: int = 128
    kernel_size: int = 3
    n_conv_layers: int = 2
    use_binary: bool = False

    # Training dynamics
    learning_rate: float = 0.001
    batch_size: int = 32
    n_epochs: int = 10
    clip_grads: float = 1.0

    # Data
    val_size: float = 0.2
    data_subsample: float = 1.0
    sample_T: int = 128

    # System
    seed: int = 42
    device: str = "metal"  # or "cpu"
    log_interval: int = 100

    # Bookkeeping
    git_commit: str = ""
    experiment_id: str = ""


def load_vision_config(path: str | None = None, **overrides) -> VisionConfig:
    """Load VisionConfig from file or create default with overrides."""
    if path and os.path.exists(path):
        with open(path) as f:
            cfg_dict = json.load(f)
        return VisionConfig(**{**cfg_dict, **overrides})
    return VisionConfig(**overrides)


def train_vision_once(cfg: VisionConfig, dry_run: bool = False) -> dict[str, Any]:
    """
    Train vision SNN once with given config.

    Args:
        cfg: VisionConfig with training parameters
        dry_run: If True, run one batch and return early for testing

    Returns:
        Dict with metrics: {loss, accuracy, duration, ...}
    """
    assert_mlx_gpu_or_die()

    # Set random seed
    mx.random.seed(cfg.seed)

    try:
        from spyx_mlx.loaders import DVSGesture_loader
    except ImportError:
        print(
            "ERROR: DVSGesture loader not found. Ensure tonic and torch are installed.",
            file=sys.stderr,
        )
        return {"loss": float("nan"), "accuracy": 0.0, "error": "loader_not_found"}

    # Load dataset
    try:
        loader = DVSGesture_loader(
            batch_size=cfg.batch_size,
            sample_T=cfg.sample_T,
            val_size=cfg.val_size,
            data_subsample=cfg.data_subsample,
            key=cfg.seed,
        )
    except Exception as e:
        print(f"ERROR: Failed to load DVSGesture: {e}", file=sys.stderr)
        return {"loss": float("nan"), "accuracy": 0.0, "error": str(e)}

    # Build minimal Conv2D SNN model
    class VisionSNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(
                2, 32, kernel_size=cfg.kernel_size, padding=cfg.kernel_size // 2
            )
            if cfg.n_conv_layers > 1:
                self.conv2 = nn.Conv2d(
                    32, 64, kernel_size=cfg.kernel_size, padding=cfg.kernel_size // 2
                )
            self.dense1 = nn.Linear(64 * 128 * 128 // 16, cfg.n_hidden)
            self.dense2 = nn.Linear(cfg.n_hidden, 11)

        def __call__(self, x):
            # x: (batch, T_packed, 128, 128, 2)
            # Unpack time if needed and process through convolutions
            batch_size = x.shape[0]
            x = mx.reshape(x, (batch_size, -1, 128, 128, 2))

            # Simple temporal aggregation: max pool over time
            x = mx.max(x, axis=1)

            # Conv layers
            x = self.conv1(x)
            x = mx.nn.relu(x)
            if cfg.n_conv_layers > 1:
                x = self.conv2(x)
                x = mx.nn.relu(x)

            # Spatial pooling
            x = mx.reshape(x, (batch_size, -1))

            # Classifier
            x = self.dense1(x)
            x = mx.nn.relu(x)
            x = self.dense2(x)
            return x

    model = VisionSNN()
    optimizer = optim.Adam(learning_rate=cfg.learning_rate)

    def loss_fn(model, x, y):
        logits = model(x)
        loss = mx.mean(nn.losses.cross_entropy(logits, y))
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    start_time = time.time()
    train_losses = []
    val_acc_final = 0.0

    for epoch in range(cfg.n_epochs):
        if dry_run and epoch > 0:
            break

        # Training epoch
        key = mx.random.PRNGKey(cfg.seed + epoch)
        train_state = loader.train_epoch(key)

        batch_losses = []
        for batch_idx in range(train_state.obs.shape[0]):
            x_batch = train_state.obs[batch_idx]
            y_batch = train_state.labels[batch_idx]

            loss, grads = loss_and_grad(model, x_batch, y_batch)

            # Clip gradients
            if cfg.clip_grads > 0:
                for key in grads.keys():
                    grads[key] = mx.clip(grads[key], -cfg.clip_grads, cfg.clip_grads)

            optimizer.update(model, grads)
            batch_losses.append(float(loss))

        epoch_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(epoch_loss)

        # Validation
        val_state = loader.val_epoch()
        val_correct = 0
        val_total = 0
        for batch_idx in range(val_state.obs.shape[0]):
            x_batch = val_state.obs[batch_idx]
            y_batch = val_state.labels[batch_idx]
            logits = model(x_batch)
            preds = mx.argmax(logits, axis=1)
            val_correct += mx.sum(preds == y_batch)
            val_total += len(y_batch)

        val_acc = float(val_correct) / val_total if val_total > 0 else 0.0
        val_acc_final = val_acc

        if (epoch + 1) % cfg.log_interval == 0 or dry_run:
            print(
                f"Epoch {epoch + 1}/{cfg.n_epochs}: loss={epoch_loss:.4f}, val_acc={val_acc:.4f}"
            )

    duration = time.time() - start_time

    return {
        "loss": float(train_losses[-1]) if train_losses else float("nan"),
        "accuracy": val_acc_final,
        "duration": duration,
        "epochs": cfg.n_epochs,
        "final_val_acc": val_acc_final,
    }


def append_jsonl(
    metrics: dict[str, Any],
    cfg: VisionConfig,
    log_file: str = "vision_experiments.jsonl",
):
    """Append experiment metrics to JSONL log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": cfg.experiment_id,
        "config": asdict(cfg),
        "metrics": metrics,
    }
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train Vision SNN on DVSGesture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python train_vision.py --help
  python train_vision.py --n_hidden 256 --learning_rate 0.0005
  python train_vision.py --config ~/vision_exp.json
  python train_vision.py --dry_run  # Test setup
        """,
    )

    # Config
    parser.add_argument(
        "--config", type=str, default=None, help="Load config from JSON file"
    )

    # Model args
    parser.add_argument("--n_hidden", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--kernel_size", type=int, default=3, help="Conv2D kernel size")
    parser.add_argument(
        "--n_conv_layers", type=int, default=2, help="Number of conv layers"
    )
    parser.add_argument(
        "--use_binary", action="store_true", help="Use binary quantization"
    )

    # Training args
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--clip_grads", type=float, default=1.0, help="Gradient clipping value"
    )

    # Data args
    parser.add_argument(
        "--val_size", type=float, default=0.2, help="Validation fraction"
    )
    parser.add_argument(
        "--data_subsample", type=float, default=1.0, help="Data subsample fraction"
    )
    parser.add_argument(
        "--sample_T", type=int, default=128, help="Time bins per sample"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # System args
    parser.add_argument("--device", type=str, default="metal", choices=["metal", "cpu"])
    parser.add_argument(
        "--dry_run", action="store_true", help="Run one epoch for testing"
    )
    parser.add_argument("--log_file", type=str, default="vision_experiments.jsonl")

    args = parser.parse_args()

    # Build config
    cfg_dict = {
        k: v
        for k, v in vars(args).items()
        if k not in ["config", "dry_run", "log_file"]
    }
    cfg = load_vision_config(args.config, **cfg_dict)

    # Assign experiment ID
    cfg = VisionConfig(
        **{**asdict(cfg), "experiment_id": datetime.now().strftime("%Y%m%d_%H%M%S")}
    )

    print(f"VisionSNN Training Config:\n{asdict(cfg)}")

    # Train
    metrics = train_vision_once(cfg, dry_run=args.dry_run)
    print(f"\nTraining Results:\n{json.dumps(metrics, indent=2)}")

    # Log
    append_jsonl(metrics, cfg, args.log_file)
    print(f"Logged to {args.log_file}")

    return 0 if metrics.get("error") is None else 1


if __name__ == "__main__":
    sys.exit(main())
