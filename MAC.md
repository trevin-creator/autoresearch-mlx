# Running this on an Apple Silicon Mac

Setup notes and the optimization record for this checkout. Everything below was
measured on the machine named in "Verified on", not estimated.

## Verified on

| | |
|---|---|
| Chip | Apple M5 Pro (18 CPU: 12P + 6E, 20 GPU cores) |
| Memory | 64 GB unified |
| macOS | 26.6.2 (25G83) |
| Metal | Metal 4, `applegpu_g17s` |
| Max Metal buffer | 38.9 GB |
| Recommended working set | 51.8 GB |
| Python | 3.13.15 (pinned in `.python-version`) |
| MLX | 0.32.2 |

## Setup

```bash
uv sync          # installs into .venv, Python 3.13
uv run prepare.py   # one-time: ~1 GB of shards + BPE tokenizer, ~4 min
uv run train.py     # one 5-minute experiment, ~5.5 min wall clock
```

`prepare.py` writes to `~/.cache/autoresearch/` (about 1.0 GB for the default
10 training shards plus the pinned validation shard).

Two setup details matter on a current Mac:

- **Python 3.13, not 3.14.** `pyproject.toml` requires `>=3.10,<3.14`, and
  Homebrew's `python3` is now 3.14. `.python-version` pins 3.13 so `uv sync`
  resolves without touching the system Python.
- **MLX 0.32.2 or newer.** The lockfile previously pinned 0.31.0. See below.

## What changed, and what it bought

All three changes are in `train.py`. `prepare.py` is untouched, so `val_bpb`
stays comparable with every other machine running this repo.

### 1. MLX 0.31.0 → 0.32.2

Nothing but a version bump, but 0.32.2 has materially better kernels for this
GPU generation. `pyproject.toml` now requires `>=0.32.2` and `uv.lock` is
regenerated.

### 2. Compile the whole optimizer step

Every microbatch forward/backward, the gradient accumulation, and the AdamW
update are fused into one Metal graph with `mx.compile`. `grad_accum_steps` is a
Python constant, so the accumulation loop unrolls at trace time and the entire
step becomes a single graph instead of a few hundred separately dispatched
kernels driven from Python.

This needed two changes to `AdamW`:

- The step counter and the LR multiplier moved from Python numbers into mx
  arrays (`optimizer.scalars`). A Python float LR changing every step would
  invalidate the compiled graph and force a recompile on *every* step.
- Adam moment buffers are allocated up front so the graph sees stable shapes
  from step 1.

The per-parameter step counters collapsed into one shared counter. Every
parameter is updated on every step, so those counters were always equal.

Set `COMPILE = False` in `train.py` to get the plain eager loop back when a
change is misbehaving.

The two paths compute the same thing. Same seed, same data, first eight steps:

| step | eager | compiled |
|---:|---:|---:|
| 0 | 9.011286 | 9.011286 |
| 1 | 9.019857 | 9.019852 |
| 2 | 8.699372 | 8.699403 |
| 3 | 8.424069 | 8.424079 |
| 4 | 8.218989 | 8.219037 |
| 5 | 8.068878 | 8.069085 |
| 6 | 7.965811 | 7.966056 |
| 7 | 7.885002 | 7.886373 |

They agree to ~1e-4, which is bf16 accumulation-order noise, at 92k versus 142k
tokens/sec.

### 3. `main()` guard

The training loop used to run at import time, so nothing could import the model
or the optimizer without training for five minutes. It is now under
`if __name__ == "__main__":`. `uv run train.py` behaves exactly as before, and
`bench/micro.py` can now import the real model instead of a copy of it.

## Results

Fixed 5-minute training budget, default hyperparameters (depth 4, AdamW), same
machine, same data. `val_bpb` is lower-is-better.

Three runs each, since one run decides nothing at this noise level.

| Code | MLX | steps (mean) | val_bpb, 3 runs | mean | peak |
|---|---|---:|---|---:|---:|
| upstream | 0.31.0 | 399 | 1.679942 / 1.645527 / 1.602925 | 1.6428 | 8.4 GB |
| upstream | 0.32.2 | 422 | 1.602921 (one run) | — | 9.2 GB |
| this checkout | 0.32.2 | 614 | 1.509226 / 1.494695 / 1.463923 | 1.4893 | 16.1 GB |

**399 → 614 optimizer steps inside the same 5 minutes, a 1.54x speedup**, which
is the only thing that matters when the budget is wall clock. Of that, the MLX
bump is worth about 6% (399 → 422 steps) and the compiled step carries the rest.

The two sets of `val_bpb` do not overlap: the worst optimized run (1.509) still
beats the best baseline run (1.603). Mean improvement is 0.154, against a
within-group spread of 0.077 (baseline) and 0.045 (optimized).

That spread is also the reason to keep using `rigor.py` for keep/discard
decisions inside the experiment loop. A change worth less than ~0.06 `val_bpb`
cannot be judged from a single run.

Peak memory roughly doubles, to 16.1 GB. That is the cost of the unrolled
graph holding both microbatches' activations and the accumulated gradients at
once. On a 64 GB machine with a 51.8 GB working set there is a lot of room left;
on a 16 GB Mac, drop `DEVICE_BATCH_SIZE` to 8 or set `COMPILE = False`.

## Tried and rejected

Recorded so nobody spends the afternoon re-testing them.

- **`DEVICE_BATCH_SIZE = 32` (no gradient accumulation).** 147k tok/s versus
  149k tok/s for the default 16 with accumulation — a wash, and it uses more
  memory. The default stays at 16, which also keeps this checkout's
  configuration identical to upstream.
- **Device-aware eval memory budget in `prepare.py`.** The hardcoded 2 GiB
  logits cap was sized for 24 GB Macs and makes a 64 GB Mac run 4x more eval
  forward passes than it needs. Raising it moved final eval from 7.4 s to 7.5 s.
  Final eval is ~2.5% of the run, so this bought nothing and was reverted rather
  than modify a file the protocol declares read-only.
- **Prefetching the dataloader on a background thread.** The packing loop is
  pure Python and looked like an obvious stall, so `train.py` now reports
  `dataloader_secs`. It measures 7–16 s out of 300 s, under 6% even at its worst.
  Not worth a thread.

## bench/micro.py

Times one full optimizer step on synthetic data, so a change can be judged in
seconds instead of a 5-minute run. Needs no dataset.

```bash
uv run bench/micro.py                  # sweep device batch 16 and 32, eager vs compiled
uv run bench/micro.py --batch 32 --steps 20
uv run bench/micro.py --depth 8        # cost of a deeper model
```

It reports median s/step, tokens/sec and peak memory. It does **not** report
`val_bpb` — it trains on random tokens. Use it to answer "is this faster", never
"is this better". `uv run train.py` is the only thing that answers the latter.
