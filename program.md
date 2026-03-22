# autoresearch-mlx

This is an Apple Silicon (MLX) port of Karpathy's autoresearch — an experiment to have the LLM do its own research. All training runs natively on MLX with unified memory. No PyTorch or CUDA required.

**Key additions:**

* Use **Optuna** to drive structured search over hyperparameters and architectures.
* Explicitly explore **spiking / spike-inspired neural networks** as a core model family.
* **Always run on MLX GPU using Spyx** for maximum performance and correct execution model.

---

## Core Execution Requirement (MLX + Spyx)

All training must:

* Run on **MLX (Apple Silicon)**
* Execute on the **GPU backend (Metal)**
* Use **Spyx** as the execution/runtime layer whenever applicable

This is **non-negotiable**:

* Do not fall back to CPU unless debugging crashes
* Do not introduce PyTorch / CUDA
* Do not bypass Spyx abstractions if they are available

The goal is to:

* maximize throughput under the fixed 5-minute budget
* evaluate models under realistic MLX GPU constraints
* explore architectures (especially spiking) that benefit from MLX execution

If something is slow, the solution is:

* simplify the model
* improve batching
* reduce memory pressure
  —not to switch execution backend.

---

## Monorepo note

This project may live inside a larger repo.

* Always stage only `autoresearch-mlx/` paths
* Never use `git add -A`

---

## Setup

1. **Agree on a run tag**
   Example: `mar5`
   Branch must not exist.

2. **Create branch**

   ```
   git checkout -b autoresearch/<tag>
   ```

3. **Read in-scope files**

   * `README.md`
   * `prepare.py` (read-only)
   * `train.py` (all experimentation happens here)

4. **Verify data**

   ```
   ~/.cache/autoresearch/
   ```

   If missing:

   ```
   uv run prepare.py
   ```

5. **Initialize results.tsv**

   * create file
   * run baseline:

     ```
     uv run train.py
     ```

6. **Confirm setup**

---

## Experimentation

Each run:

* executes on **MLX GPU via Spyx**
* trains for **5 minutes wall clock**

Goal:

**Minimize `val_bpb`**

---

## Model families to explore

You must actively explore across:

### 1. Dense models

* transformer variants
* optimizer / scaling changes

### 2. Spike-inspired models

* thresholded activations
* sparse gating
* event-style updates

### 3. Spiking neural networks

* membrane state
* spike emission
* surrogate gradients
* temporal recurrence

### 4. Hybrid models

* transformer + spike FFN
* spike memory + dense attention
* residual spike gating

---

## Optuna-driven search

All experimentation must be structured via **Optuna**.

### Rule

**One run = one Optuna trial**

Use persistent storage:

```python
study = optuna.create_study(
    direction="minimize",
    storage="sqlite:///optuna.db",
    load_if_exists=True,
)
study.optimize(objective, n_trials=1)
```

---

## Search space (example)

```python
lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

model_family = trial.suggest_categorical(
    "model_family",
    [
        "dense_transformer",
        "spike_mlp",
        "spike_rnn",
        "hybrid_spike_transformer",
    ],
)

spike_threshold = trial.suggest_float("spike_threshold", 0.1, 2.0)
membrane_decay = trial.suggest_float("membrane_decay", 0.7, 0.999)
sparsity = trial.suggest_float("sparsity", 0.01, 0.5)
```

Use conditional parameters depending on model family.

---

## Spiking direction (important)

We are not aiming for strict biological realism.

We are exploring:

* sparse computation
* temporal state
* event-driven updates
* threshold dynamics

Example:

```
v_t = decay * v_{t-1} + input_t
spike_t = H(v_t - threshold)
v_t = reset(v_t, spike_t)
```

Implementation can use:

* surrogate gradients
* straight-through estimators
* approximations compatible with MLX / Spyx

---

## What you CAN do

* Modify `train.py`
* Add Optuna logic
* Implement spiking / hybrid architectures
* Change training loop, optimizer, batching
* Optimize for MLX GPU execution

## What you CANNOT do

* Modify `prepare.py`
* Add dependencies
* Switch away from MLX / Spyx
* Change evaluation

---

## Output

Example:

```
val_bpb:          2.534000
peak_vram_mb:     27528.9
```

Extract:

```
grep "^val_bpb:" run.log
```

---

## Logging results

`results.tsv`:

```
commit	val_bpb	memory_gb	status	description
```

Example:

```
abc1234	2.51	27.1	keep	optuna hybrid_spike lr=0.01 decay=0.93 thr=0.7
```

---

## Experiment loop

LOOP FOREVER:

1. Inspect current best results + Optuna DB
2. Modify `train.py`
3. Commit:

   ```
   git add autoresearch-mlx/train.py
   git commit -m "experiment: <desc>"
   ```
4. Run:

   ```
   uv run train.py > run.log 2>&1
   ```
5. Extract results
6. Handle crash if needed
7. Log to TSV
8. Keep or discard:

   * better → amend
   * worse → reset

---

## Strategy

This is now:

* Optuna-guided search
* across dense + spike + hybrid models
* optimized for MLX GPU via Spyx

Key levers:

* sparsity vs density
* temporal state
* architecture choice
* compute efficiency under fixed time

---

## Simplicity rule

Prefer:

* simpler models
* fewer hacks
* stable training

Reject:

* fragile spike implementations with marginal gains

---

## Crash handling (important for spiking)

Common issues:

* dead neurons (no spikes)
* exploding membrane
* zero gradients
* broken state propagation

Fix small issues. Skip fundamentally broken ideas.

---

## Timeout

* Expected: ~7 minutes/run
* Kill if >15 minutes

---

## Objective

> Use Optuna to search dense, spike-inspired, and spiking architectures on MLX GPU (via Spyx) to minimize `val_bpb` under a fixed 5-minute training budget.

---

## NEVER STOP

The loop is autonomous.

Continue indefinitely:

* refining search space
* improving spike designs
* simplifying models
* exploiting promising regions

Do not pause. Do not ask for confirmation.
