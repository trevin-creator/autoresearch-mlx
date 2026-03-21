"""
Autoresearch SNN training script — SHD benchmark, Apple Silicon MLX.
Usage: ~/.local/bin/uv run train_snn.py

This is the file the autonomous agent modifies. Everything is fair game:
architecture, optimizer, hyperparameters, neuron models, regularisation.

The goal: maximise val_acc (higher is better) within the fixed time budget.
Evaluation is done by prepare_snn.evaluate_accuracy() — do not modify that.
"""

import gc
import math
import os
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from prepare_snn import (
    N_CHANNELS,
    N_CLASSES,
    TIME_BUDGET,
    T_STEPS,
    evaluate_accuracy,
    get_peak_memory_mb,
    load_datasets,
)
from spyx_mlx import fn
from spyx_mlx.nn import ALIF, IF, LI, LIF, CuBaLIF

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


# ---------------------------------------------------------------------------
# Hyperparameters — edit these freely
# ---------------------------------------------------------------------------

N_HIDDEN    = 128       # neurons per hidden layer
N_LAYERS    = 2         # number of hidden spiking layers
T_STEPS_RUN = T_STEPS  # time steps (must match prepare_snn.T_STEPS or override)
BATCH_SIZE  = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTHING = 0.3

# Learning rate schedule: cosine decay with linear warmup
WARMUP_RATIO   = 0.05
FINAL_LR_FRAC  = 0.0    # fraction of peak LR at end of training

# Regularisation
SILENCE_REG_WEIGHT  = 0.0   # penalise silent neurons
SPARSITY_REG_WEIGHT = 0.0   # penalise over-active neurons

# Evaluation
FINAL_EVAL_BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SHD_SNN(nn.Module):
    """
    Baseline 2-layer LIF network for SHD.
    Linear -> LIF -> Linear -> LIF -> Linear -> LI
    """

    def __init__(self, n_input, n_hidden, n_layers, n_classes):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Linear(n_input, n_hidden, bias=False)

        # Hidden layers: alternating Linear + LIF
        self.hidden_linears = [nn.Linear(n_hidden, n_hidden, bias=False) for _ in range(n_layers - 1)]
        self.lif_layers = [LIF(n_hidden) for _ in range(n_layers)]

        # Output
        self.output_proj = nn.Linear(n_hidden, n_classes, bias=False)
        self.readout = LI(n_classes)

    def __call__(self, x_seq):
        """
        Args:
            x_seq: (T, batch, n_input) float32

        Returns:
            traces: (T, batch, n_classes) voltage traces for integral loss
        """
        T, B, _ = x_seq.shape

        # Initialise states
        lif_states = [lif.initial_state(B) for lif in self.lif_layers]
        li_state   = self.readout.initial_state(B)

        traces = []
        for t in range(T):
            x = x_seq[t]                              # (B, n_input)
            x = self.input_proj(x)                    # (B, n_hidden)
            x, lif_states[0] = self.lif_layers[0](x, lif_states[0])

            for i, (linear, lif) in enumerate(zip(self.hidden_linears, self.lif_layers[1:])):
                x = linear(x)
                x, lif_states[i + 1] = lif(x, lif_states[i + 1])

            x = self.output_proj(x)                   # (B, n_classes)
            v_out, li_state = self.readout(x, li_state)
            traces.append(v_out)

        return mx.stack(traces, axis=0)              # (T, B, n_classes)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, total_steps: int, peak_lr: float) -> float:
    progress = step / max(total_steps, 1)
    if progress < WARMUP_RATIO:
        return peak_lr * (progress / WARMUP_RATIO) if WARMUP_RATIO > 0 else peak_lr
    t = (progress - WARMUP_RATIO) / (1.0 - WARMUP_RATIO)
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return peak_lr * (FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * cosine)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def loss_fn(model, x_seq, targets):
    traces = model(x_seq)
    loss = fn.integral_crossentropy(traces, targets, smoothing=LABEL_SMOOTHING)
    if SILENCE_REG_WEIGHT > 0:
        # collect hidden spike traces — not directly exposed here; skip for baseline
        pass
    return loss


t_start = time.time()
mx.random.seed(42)

print("Loading SHD dataset ...")
train_ds, test_ds = load_datasets(n_steps=T_STEPS_RUN)
t_data = time.time()
print(f"Dataset loaded in {t_data - t_start:.1f}s | "
      f"train: {train_ds.N} samples, test: {test_ds.N} samples")

model = SHD_SNN(N_CHANNELS, N_HIDDEN, N_LAYERS, N_CLASSES)
mx.eval(model.parameters())
num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
print(f"Parameters: {num_params / 1e6:.2f}M | hidden: {N_HIDDEN} x {N_LAYERS}")

optimizer = optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

loss_and_grad = nn.value_and_grad(model, loss_fn)

rng = None  # numpy rng, initialised lazily
import numpy as np
rng = np.random.default_rng(42)

total_training_time = 0.0
step = 0
t_compiled = None
smooth_loss = 0.0

print(f"Time budget: {TIME_BUDGET}s | batch: {BATCH_SIZE} | T: {T_STEPS_RUN}")

while True:
    # Iterate over training batches, reshuffling each epoch
    for x_batch, y_batch in train_ds.batches(BATCH_SIZE, shuffle=True, rng=rng):
        if step > 0 and total_training_time >= TIME_BUDGET:
            break

        t0 = time.time()

        # x_batch: (B, T, C) -> transpose to (T, B, C)
        x_seq = x_batch.transpose(1, 0, 2)

        # Estimate total steps for LR schedule (rough)
        steps_per_epoch = math.ceil(train_ds.N / BATCH_SIZE)
        # Estimate epochs from first step timing if available; use 1000 as fallback
        estimated_total_steps = max(1000, steps_per_epoch * 20)

        lr = get_lr(step, estimated_total_steps, LEARNING_RATE)
        optimizer.learning_rate = lr

        loss, grads = loss_and_grad(model, x_seq, y_batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), loss)

        if t_compiled is None:
            t_compiled = time.time()
            print(f"Compiled in {t_compiled - t_data:.1f}s")

        dt = time.time() - t0
        total_training_time += dt

        loss_f = float(loss.item())
        ema = 0.95
        smooth_loss = ema * smooth_loss + (1 - ema) * loss_f if step > 0 else loss_f
        debiased = smooth_loss / (1 - ema ** (step + 1)) if step > 0 else loss_f

        pct_done = 100 * min(total_training_time / TIME_BUDGET, 1.0)
        remaining = max(0.0, TIME_BUDGET - total_training_time)
        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased:.4f} | "
            f"lr: {lr:.2e} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ",
            end="", flush=True,
        )

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()

        step += 1
        if total_training_time >= TIME_BUDGET:
            break

    if total_training_time >= TIME_BUDGET:
        break

print()
t_train = time.time()
print(f"Training completed in {t_train - t_data:.1f}s ({step} steps)")

print("Starting final evaluation ...")
val_acc = evaluate_accuracy(model, test_ds, batch_size=FINAL_EVAL_BATCH_SIZE)
t_eval = time.time()
print(f"Eval completed in {t_eval - t_train:.1f}s")

peak_vram_mb = get_peak_memory_mb()

print("---")
print(f"val_acc:          {val_acc:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_eval - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.2f}")
print(f"n_hidden:         {N_HIDDEN}")
print(f"n_layers:         {N_LAYERS}")
print(f"t_steps:          {T_STEPS_RUN}")
