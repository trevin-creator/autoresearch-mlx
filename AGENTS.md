# Autoresearch MLX

MLX research and training workspace for local model experiments.

## Stack

- Python + uv + MLX

## Key Commands

- Dev: `uv run train.py`
- Build: `./scripts/ensure-runtime.sh`
- Lint: `not configured`
- Type check: `not configured`
- Test: `not configured`

## Autoresearch Protocol

When `autoresearch.md` or `autoresearch.sh` exists in this project:

- Read `autoresearch.md` before substantial implementation, refactor, or investigation work.
- Run `./autoresearch.sh` before and after changes when the task should be provable.
- Treat `autoresearch.md` as the task contract: objective, metric, scope, out-of-bounds, regressions, and next experiments.
- Keep `autoresearch.sh` safe and repeatable. Emit only `METRIC`, `CHECK`, or `ARTIFACT` lines.
- Put deeper benchmarks or probes in `autoresearch.bench.sh` and richer validation in `autoresearch.checks.sh`.
- Update `autoresearch.md` when the objective, constraints, or proof commands change.

## Agent Skills & Memory

Global skills are indexed at `~/.agents/skills/_manifest.json`.
Full conventions and episodic logging live at `~/memory/agents/skills.md`.
