# autoresearch-mlx — SNN Edition

Autonomous SNN research loop. MLX-native spiking neural networks on Apple Silicon, benchmarked on the SHD (Spiking Heidelberg Digits) classification task.

**Monorepo note:** This project may live inside a larger repo. Always stage only `autoresearch-mlx/` paths. Never use blind `git add -A`.

## Setup

Work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar21-snn`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare_snn.py` — fixed constants, SHD data loading, binning, evaluation harness. **Do not modify.**
   - `train_snn.py` — the file you modify. Network architecture, optimizer, hyperparameters.
   - `spyx_mlx/nn.py` — MLX neuron models (LIF, ALIF, IF, LI, CuBaLIF, RLIF). Read to understand the API.
   - `spyx_mlx/axn.py` — surrogate gradient functions (superspike, arctan, triangular, boxcar). Read to know what's available.
   - `spyx_mlx/fn.py` — loss and regularisation functions.
4. **Verify data exists**: Check that `~/.cache/autoresearch-snn/shd_train.h5` and `shd_test.h5` exist. If not, the first `uv run train_snn.py` will download and cache them (~1.6 GB total). Data is cached as binned `.npz` files for subsequent runs.
5. **Initialize results_snn.tsv**: Create it with header + baseline entry. Run `~/.local/bin/uv run train_snn.py` once to establish YOUR baseline on this hardware.
6. **Confirm and go**.

## Experimentation

Each experiment runs on Apple Silicon via MLX. The training script runs for a **fixed time budget of 5 minutes** (wall-clock training time, excluding data loading and evaluation). Launch with:

```bash
~/.local/bin/uv run train_snn.py > run_snn.log 2>&1
```

**What you CAN do:**
- Modify `train_snn.py` — this is the only file you edit. Everything is fair game:
  - Network architecture: depth, width, neuron type (LIF/ALIF/CuBaLIF/IF), recurrence
  - Optimizer: AdamW, Adam, SGD, Lion — use `mlx.optimizers`
  - Hyperparameters: LR, weight decay, batch size, time steps
  - Loss: label smoothing, regularisation weights (silence_reg, sparsity_reg)
  - Data: T_STEPS_RUN (time steps, must be ≤ T_STEPS from prepare_snn.py)
  - Surrogate gradient: swap to `arctan()`, `triangular()`, etc. from `spyx_mlx.axn`

**What you CANNOT do:**
- Modify `prepare_snn.py`. It is read-only.
- Modify `spyx_mlx/` library files. These are infrastructure, not experiment files.
- Install new packages. Use what's in `pyproject.toml`.

**The goal: maximise val_acc (higher is better).**

Since the time budget is fixed, throughput matters: smaller/faster models can outperform larger ones by fitting more gradient steps into 5 minutes. Balance model capacity against iteration speed.

**Memory** is a soft constraint. Avoid OOM; some increase is fine for real val_acc gains.

**Simplicity criterion**: All else equal, simpler is better. A tiny gain with lots of complexity is not worth it.

## Output format

When the script finishes it prints:

```
---
val_acc:          0.823456
training_seconds: 300.1
total_seconds:    412.3
peak_vram_mb:     8192.0
num_steps:        4321
num_params_M:     0.24
n_hidden:         128
n_layers:         2
t_steps:          128
```

Extract results:
```bash
grep "^val_acc:\|^peak_vram_mb:" run_snn.log
```

## Logging results

Log to `results_snn.tsv` (tab-separated):

```
commit	val_acc	memory_gb	status	description
```

1. git commit hash (7 chars)
2. val_acc achieved (e.g. 0.823456) — use 0.000000 for crashes
3. peak memory in GB (peak_vram_mb / 1024), round to .1f — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description

Example:
```
commit	val_acc	memory_gb	status	description
abc1234	0.650000	4.2	keep	baseline LIF 128 hidden 2 layers
def5678	0.720000	4.3	keep	switch to ALIF neurons
```

## The experiment loop

Runs on a dedicated branch (e.g. `autoresearch/mar21-snn`).

LOOP FOREVER:

1. Check git state (branch, latest commit).
2. Form a hypothesis. Ideas to explore (not exhaustive):
   - Swap LIF → ALIF (adaptive threshold helps SHD)
   - Add a third hidden layer
   - Try CuBaLIF (dual time constants model temporal dynamics better)
   - Reduce batch size to fit more steps per budget
   - Try arctan surrogate instead of superspike
   - Add recurrence (RLIF)
   - Reduce hidden size but add depth
   - Increase T_STEPS_RUN (more temporal resolution)
   - Tune label smoothing
   - Add sparsity regularisation
3. Edit `train_snn.py` with the experiment idea.
4. `git add autoresearch-mlx/train_snn.py && git commit -m "experiment: <description>"`
5. Run: `~/.local/bin/uv run train_snn.py > run_snn.log 2>&1`
6. Check results: `grep "^val_acc:\|^peak_vram_mb:" run_snn.log`
7. If empty output → crash. `tail -n 50 run_snn.log` to inspect. Fix or discard.
8. Log to `results_snn.tsv`.
9. If val_acc improved: `git add autoresearch-mlx/results_snn.tsv && git commit --amend --no-edit`
10. If equal or worse: record hash, `git reset --hard <previous kept commit>`.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. Run indefinitely until manually interrupted. If you run out of ideas, revisit near-misses, try combinations, or look at the spyx research notebooks at `spyx/research/` for inspiration (SHD baselines, surrogate comparisons, optimizer studies).

**Timeout**: Each experiment takes ~7 minutes (5 min training + data loading + eval). If a run exceeds 15 minutes, kill it and treat as failure.

**Crashes**: OOM or bugs — fix obvious typos and re-run. If the idea is fundamentally broken, log crash and move on.

## Architecture reference

```
# Typical SHD SNN in train_snn.py:
Input (B, T, 700) → [Linear(700→H) → LIF(H)] × N_LAYERS → Linear(H→20) → LI(20)
→ integral over T → softmax CE loss
```

Available neurons from `spyx_mlx.nn`:
- `LIF(n, beta_init=None)` — learnable beta decay per neuron
- `ALIF(n, beta_init=None, gamma_init=None)` — adaptive threshold
- `IF(n)` — no decay
- `LI(n)` — leaky integrator, no spikes (use as readout)
- `CuBaLIF(n, alpha_init=None, beta_init=None)` — current-based dual TC
- `RLIF(n)` — LIF with recurrent connections

Available surrogate gradients from `spyx_mlx.axn`:
- `superspike(k=25.0)` — default
- `arctan(k=2.0)`
- `triangular(k=1.0)`
- `boxcar(k=1.0)`
- `straight_through()`

Available losses from `spyx_mlx.fn`:
- `integral_crossentropy(traces, targets, smoothing=0.3)`
- `integral_accuracy(traces, targets)`
- `silence_reg(traces, min_rate=0.01)`
- `sparsity_reg(traces, max_rate=0.1)`
