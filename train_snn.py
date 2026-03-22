"""
Autoresearch SNN training script — SHD benchmark, Apple Silicon MLX.
Usage: ~/.local/bin/uv run train_snn.py

This file now supports:
- config-driven runs (CLI overrides)
- structured JSONL experiment logging
- reproducibility metadata capture
- basic training anomaly checks
"""

import argparse
import dataclasses
import gc
import json
import math
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten

from prepare_snn import (
    N_CHANNELS,
    N_CLASSES,
    T_STEPS,
    TIME_BUDGET,
    evaluate_accuracy,
    get_peak_memory_mb,
    load_datasets,
)
from spyx_mlx import fn
from spyx_mlx.nn import ALIF, LI

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    n_hidden: int = 128
    n_layers: int = 2
    t_steps: int = T_STEPS
    batch_size: int = 48
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    warmup_ratio: float = 0.05
    final_lr_frac: float = 0.0
    silence_reg_weight: float = 0.0
    sparsity_reg_weight: float = 0.0
    final_eval_batch_size: int = 256
    weight_mode: str = "float"
    fixed_point_bits: int = 8
    fixed_point_frac_bits: int = 4
    fixed_point_round_mode: str = "nearest"
    fixed_point_use_ste: bool = True
    ternary_threshold: float = 0.08
    ternary_scale_mode: str = "mean_abs"
    ternary_use_ste: bool = True
    seed: int = 42
    time_budget_s: float = float(TIME_BUDGET)


class ShdSnn(nn.Module):
    """
    2-layer ALIF network for SHD. Adaptive threshold improves temporal coding.
    Linear -> ALIF -> Linear -> ALIF -> Linear -> LI
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_layers: int,
        n_classes: int,
        cfg: ExperimentConfig,
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.weight_mode = cfg.weight_mode
        self.fixed_point_bits = cfg.fixed_point_bits
        self.fixed_point_frac_bits = cfg.fixed_point_frac_bits
        self.fixed_point_round_mode = cfg.fixed_point_round_mode
        self.fixed_point_use_ste = cfg.fixed_point_use_ste
        self.ternary_threshold = cfg.ternary_threshold
        self.ternary_scale_mode = cfg.ternary_scale_mode
        self.ternary_use_ste = cfg.ternary_use_ste

        self.input_proj = nn.Linear(n_input, n_hidden, bias=False)
        self.hidden_linears = [
            nn.Linear(n_hidden, n_hidden, bias=False) for _ in range(n_layers - 1)
        ]
        self.alif_layers = [ALIF(n_hidden) for _ in range(n_layers)]

        self.output_proj = nn.Linear(n_hidden, n_classes, bias=False)
        self.readout = LI(n_classes)

    def _ternary_weight(self, w: mx.array) -> mx.array:
        abs_w = mx.abs(w)
        max_abs = mx.maximum(mx.max(abs_w), mx.array(1e-8, dtype=w.dtype))
        threshold = self.ternary_threshold * max_abs
        keep_mask = abs_w >= threshold

        if self.ternary_scale_mode == "max_abs":
            scale = max_abs
        else:
            keep_count = mx.maximum(mx.sum(keep_mask.astype(w.dtype)), mx.array(1.0))
            scale = mx.sum(abs_w * keep_mask.astype(w.dtype)) / keep_count

        ternary = mx.where(keep_mask, mx.sign(w) * scale, mx.zeros_like(w))

        if self.ternary_use_ste and hasattr(mx, "stop_gradient"):
            return w + mx.stop_gradient(ternary - w)
        return ternary

    def _fixed_point_weight(self, w: mx.array) -> mx.array:
        bits = max(2, int(self.fixed_point_bits))
        frac_bits = max(0, min(int(self.fixed_point_frac_bits), bits - 1))

        scale = float(2**frac_bits)
        max_int = float((2 ** (bits - 1)) - 1)
        min_int = float(-(2 ** (bits - 1)))

        w_scaled = w * scale
        if self.fixed_point_round_mode == "floor":
            w_q_int = mx.floor(w_scaled)
        else:
            w_q_int = mx.round(w_scaled)

        w_q_int = mx.clip(w_q_int, min_int, max_int)
        w_q = w_q_int / scale

        if self.fixed_point_use_ste and hasattr(mx, "stop_gradient"):
            return w + mx.stop_gradient(w_q - w)
        return w_q

    def _linear(self, x: mx.array, linear: nn.Linear) -> mx.array:
        if self.weight_mode == "float":
            return linear(x)

        if self.weight_mode == "ternary":
            q_weight = self._ternary_weight(linear.weight)
        elif self.weight_mode == "fixed":
            q_weight = self._fixed_point_weight(linear.weight)
        else:
            return linear(x)

        return x @ q_weight.T

    def __call__(self, x_seq: mx.array) -> mx.array:
        n_t, n_b, _ = x_seq.shape

        alif_states = [alif.initial_state(n_b) for alif in self.alif_layers]
        li_state = self.readout.initial_state(n_b)

        traces = []
        for t in range(n_t):
            x = x_seq[t]
            x = self._linear(x, self.input_proj)
            x, alif_states[0] = self.alif_layers[0](x, alif_states[0])

            for i, (linear, alif) in enumerate(
                zip(self.hidden_linears, self.alif_layers[1:], strict=False)
            ):
                x = self._linear(x, linear)
                x, alif_states[i + 1] = alif(x, alif_states[i + 1])

            x = self._linear(x, self.output_proj)
            v_out, li_state = self.readout(x, li_state)
            traces.append(v_out)

        return mx.stack(traces, axis=0)


def assert_mlx_gpu_or_die() -> str:
    device = mx.default_device()
    if getattr(device, "type", None) != mx.gpu:
        raise RuntimeError(
            f"MLX GPU is required for training, but current default device is {device}."
        )
    print(f"Using MLX device: {device}")
    return str(device)


def get_git_commit_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        return out
    except Exception:
        return "unknown"


def get_lr(step: int, total_steps: int, cfg: ExperimentConfig) -> float:
    progress = step / max(total_steps, 1)
    if progress < cfg.warmup_ratio:
        if cfg.warmup_ratio <= 0:
            return cfg.learning_rate
        return cfg.learning_rate * (progress / cfg.warmup_ratio)
    t = (progress - cfg.warmup_ratio) / max(1e-8, (1.0 - cfg.warmup_ratio))
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return cfg.learning_rate * (cfg.final_lr_frac + (1.0 - cfg.final_lr_frac) * cosine)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def train_once(cfg: ExperimentConfig) -> tuple[dict[str, Any], Any]:
    """Train an SNN model and return (metrics, model)."""
    t_start = time.time()
    mx.random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    device_name = assert_mlx_gpu_or_die()

    print("Loading SHD dataset ...")
    train_ds, test_ds = load_datasets(n_steps=cfg.t_steps)
    t_data = time.time()
    print(
        f"Dataset loaded in {t_data - t_start:.1f}s | "
        f"train: {train_ds.N} samples, test: {test_ds.N} samples"
    )

    model = ShdSnn(N_CHANNELS, cfg.n_hidden, cfg.n_layers, N_CLASSES, cfg)
    mx.eval(model.parameters())
    num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(
        f"Parameters: {num_params / 1e6:.2f}M | hidden: {cfg.n_hidden} x {cfg.n_layers}"
    )

    optimizer = optim.AdamW(
        learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    def loss_fn(local_model: nn.Module, x_seq: mx.array, targets: mx.array) -> mx.array:
        traces = local_model(x_seq)
        return fn.integral_crossentropy(
            smoothing=cfg.label_smoothing,
            time_axis=0,
        )(traces, targets)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    rng = np.random.default_rng(cfg.seed)

    total_training_time = 0.0
    step = 0
    t_compiled = None
    smooth_loss = 0.0
    step_dt_samples: list[float] = []

    steps_per_epoch = max(1, math.ceil(train_ds.N / cfg.batch_size))
    estimated_total_steps = max(1000, steps_per_epoch * 20)

    print(
        f"Time budget: {cfg.time_budget_s:.0f}s | batch: {cfg.batch_size} | T: {cfg.t_steps}"
    )

    while True:
        for x_batch, y_batch in train_ds.batches(cfg.batch_size, shuffle=True, rng=rng):
            if step > 0 and total_training_time >= cfg.time_budget_s:
                break

            t0 = time.time()
            x_seq = x_batch.transpose(1, 0, 2)

            lr = get_lr(step, estimated_total_steps, cfg)
            optimizer.learning_rate = lr

            loss, grads = loss_and_grad(model, x_seq, y_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            if t_compiled is None:
                t_compiled = time.time()
                print(f"Compiled in {t_compiled - t_data:.1f}s")

            dt = time.time() - t0
            total_training_time += dt
            step_dt_samples.append(dt)

            if len(step_dt_samples) == 10:
                avg_dt = sum(step_dt_samples) / len(step_dt_samples)
                estimated_total_steps = max(
                    1, int(cfg.time_budget_s / max(avg_dt, 1e-6))
                )

            loss_f = float(loss.item())
            if not math.isfinite(loss_f):
                raise RuntimeError(f"Non-finite loss detected at step {step}: {loss_f}")

            ema = 0.95
            smooth_loss = ema * smooth_loss + (1 - ema) * loss_f if step > 0 else loss_f
            debiased = smooth_loss / (1 - ema ** (step + 1)) if step > 0 else loss_f

            pct_done = 100 * min(total_training_time / cfg.time_budget_s, 1.0)
            remaining = max(0.0, cfg.time_budget_s - total_training_time)
            print(
                f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased:.4f} | "
                f"lr: {lr:.2e} | dt: {dt * 1000:.0f}ms | remaining: {remaining:.0f}s    ",
                end="",
                flush=True,
            )

            if step == 0:
                gc.collect()
                gc.freeze()
                gc.disable()

            step += 1
            if total_training_time >= cfg.time_budget_s:
                break

        if total_training_time >= cfg.time_budget_s:
            break

    print()
    t_train = time.time()
    print(f"Training completed in {t_train - t_data:.1f}s ({step} steps)")

    print("Starting final evaluation ...")
    val_acc = evaluate_accuracy(model, test_ds, batch_size=cfg.final_eval_batch_size)
    t_eval = time.time()
    print(f"Eval completed in {t_eval - t_train:.1f}s")

    peak_vram_mb = get_peak_memory_mb()

    result = {
        "val_acc": float(val_acc),
        "training_seconds": float(total_training_time),
        "total_seconds": float(t_eval - t_start),
        "peak_vram_mb": float(peak_vram_mb),
        "num_steps": int(step),
        "num_params_M": float(num_params / 1e6),
        "device": device_name,
    }
    return result, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SHD SNN experiment")
    parser.add_argument("--n-hidden", type=int, default=ExperimentConfig.n_hidden)
    parser.add_argument("--n-layers", type=int, default=ExperimentConfig.n_layers)
    parser.add_argument("--t-steps", type=int, default=ExperimentConfig.t_steps)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument(
        "--learning-rate", type=float, default=ExperimentConfig.learning_rate
    )
    parser.add_argument(
        "--weight-decay", type=float, default=ExperimentConfig.weight_decay
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=ExperimentConfig.label_smoothing
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=ExperimentConfig.warmup_ratio
    )
    parser.add_argument(
        "--final-lr-frac", type=float, default=ExperimentConfig.final_lr_frac
    )
    parser.add_argument(
        "--weight-mode",
        type=str,
        default=ExperimentConfig.weight_mode,
        choices=["float", "fixed", "ternary"],
        help="Weight path: float baseline, fixed-point, or ternary projection.",
    )
    parser.add_argument(
        "--fixed-point-bits",
        type=int,
        default=ExperimentConfig.fixed_point_bits,
        help="Total signed fixed-point bit-width (including sign bit).",
    )
    parser.add_argument(
        "--fixed-point-frac-bits",
        type=int,
        default=ExperimentConfig.fixed_point_frac_bits,
        help="Fractional bits for fixed-point quantization.",
    )
    parser.add_argument(
        "--fixed-point-round-mode",
        type=str,
        default=ExperimentConfig.fixed_point_round_mode,
        choices=["nearest", "floor"],
        help="Rounding mode for fixed-point projection.",
    )
    parser.add_argument(
        "--fixed-point-use-ste",
        action=argparse.BooleanOptionalAction,
        default=ExperimentConfig.fixed_point_use_ste,
        help="Use straight-through estimator for fixed-point projection gradients.",
    )
    parser.add_argument(
        "--ternary-threshold",
        type=float,
        default=ExperimentConfig.ternary_threshold,
        help="Relative threshold used for ternary {-alpha,0,+alpha} projection.",
    )
    parser.add_argument(
        "--ternary-scale-mode",
        type=str,
        default=ExperimentConfig.ternary_scale_mode,
        choices=["mean_abs", "max_abs"],
        help="How alpha is computed for ternary non-zero weights.",
    )
    parser.add_argument(
        "--ternary-use-ste",
        action=argparse.BooleanOptionalAction,
        default=ExperimentConfig.ternary_use_ste,
        help="Use straight-through estimator for ternary projection gradients.",
    )
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument(
        "--time-budget-s", type=float, default=ExperimentConfig.time_budget_s
    )
    parser.add_argument(
        "--log-jsonl",
        type=Path,
        default=Path("experiments/snn_runs.jsonl"),
        help="Structured JSONL output path for run metadata and metrics.",
    )
    parser.add_argument(
        "--export-ir",
        type=Path,
        default=None,
        help="Path to write NetworkIR JSON after training (enables hardware export).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        t_steps=args.t_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        warmup_ratio=args.warmup_ratio,
        final_lr_frac=args.final_lr_frac,
        weight_mode=args.weight_mode,
        fixed_point_bits=args.fixed_point_bits,
        fixed_point_frac_bits=args.fixed_point_frac_bits,
        fixed_point_round_mode=args.fixed_point_round_mode,
        fixed_point_use_ste=args.fixed_point_use_ste,
        ternary_threshold=args.ternary_threshold,
        ternary_scale_mode=args.ternary_scale_mode,
        ternary_use_ste=args.ternary_use_ste,
        seed=args.seed,
        time_budget_s=args.time_budget_s,
    )

    result, model = train_once(cfg)

    if args.export_ir is not None:
        from snn_ir.export import export_ir

        ir = export_ir(model, cfg, meta={"git_commit": get_git_commit_short()})
        args.export_ir.parent.mkdir(parents=True, exist_ok=True)
        args.export_ir.write_text(ir.model_dump_json(indent=2))
        print(f"IR exported to {args.export_ir}")

    run_payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit_short(),
        "python_seed": cfg.seed,
        "mlx_version": str(getattr(mx, "__version__", "unknown")),
        "config": dataclasses.asdict(cfg),
        "metrics": result,
    }
    append_jsonl(args.log_jsonl, run_payload)

    print("---")
    print(f"val_acc:          {result['val_acc']:.6f}")
    print(f"training_seconds: {result['training_seconds']:.1f}")
    print(f"total_seconds:    {result['total_seconds']:.1f}")
    print(f"peak_vram_mb:     {result['peak_vram_mb']:.1f}")
    print(f"num_steps:        {result['num_steps']}")
    print(f"num_params_M:     {result['num_params_M']:.2f}")
    print(f"n_hidden:         {cfg.n_hidden}")
    print(f"n_layers:         {cfg.n_layers}")
    print(f"t_steps:          {cfg.t_steps}")
    print(f"log_jsonl:        {args.log_jsonl}")


if __name__ == "__main__":
    main()
