# Autoresearch Brief
<!-- AUTORESEARCH_STANDARD_VERSION: 1 -->

## Project

- Name: Autoresearch MLX
- Summary: MLX research and training workspace for local model experiments.
- Stack: Python + uv + MLX

## Key Commands

- Dev: `uv run train.py`
- Build: `./scripts/ensure-runtime.sh`
- Lint: `not configured`
- Type check: `not configured`
- Test: `not configured`

## Objective

Keep the local MLX research loop reproducible on Apple Silicon, especially around cache setup, results logging, and memory-safe final evaluation.

## Primary Metric

- `CHECK tokenizer_cache_present=true` from `./autoresearch.bench.sh`
- `CHECK run_log_present=true` and `METRIC results_rows=...` from `./autoresearch.bench.sh`

## Current Workstream

- Owner: Abdias
- Status: Active
- Trigger / ticket: Local MLX autoresearch continuation after the M4 cache and final-eval regressions
- Baseline date: 2026-04-16

## Scope

- In bounds: Training loop reproducibility, cache/probe sanity, experiment logging, and Apple Silicon readiness.
- Out of bounds: Scope creep outside the core `prepare.py` / `train.py` / `results.tsv` research loop.

## Known Regressions / Constraints

- Verify the exact tokenizer cache artifact name (`token_bytes.npy`), not just the cache directory.
- Final evaluation must stay safe on smaller Apple Silicon memory tiers.
- Do not muddy the research loop with unrelated repo-wide changes.
- `uv` may panic on some macOS setups; the check path should fall back to a local venv instead of treating that as a repo failure.

## Candidate Experiments

1. Keep the cache and results probe current before new overnight runs.
2. Use the brief to capture hardware-specific learnings instead of re-discovering them.
3. Bias toward experiments that improve `val_bpb` without turning `train.py` into a mess.

## Proof Runner

- `./autoresearch.sh` — standard entrypoint
- `./autoresearch.checks.sh` — validation gate
- `./autoresearch.bench.sh` — local MLX cache/results readiness probe for Apple Silicon runs

## Notes

- This brief should keep local research honest and resumable.
- Use `program.md` for the deeper autonomous experiment protocol.
