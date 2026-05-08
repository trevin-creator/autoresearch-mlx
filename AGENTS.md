# Project Codex Instructions

## Project Truth

This file overrides assumptions.

Before making changes, inspect the repository and infer the actual stack from the code.

Do not assume React, Next.js, Astro, Flutter, Rails, Svelte, Shopify, or any other stack unless the repo proves it.

## Current Project Stack

- Language: Python
- Framework: MLX research/training scripts
- Package manager: uv (uv.lock, pyproject.toml)
- Build command: ./scripts/ensure-runtime.sh per existing notes, though scripts directory was not present in the shallow scan
- Dev command: uv run train.py
- Test command: not configured in repo; use autoresearch checks when present
- Deploy target: local research workspace
- Database: none
- Styling: not applicable
- Important directories: train.py, prepare.py, autoresearch*.sh, program.md, pyproject.toml

## Project Priorities

1. Keep the app working.
2. Keep changes small.
3. Improve clarity before adding features.
4. Avoid unnecessary dependencies.
5. Prefer boring, stable solutions.
6. Protect production data and credentials.

## Required Workflow

For every task:

1. Read this file.
2. Check `git status`.
3. Inspect the relevant files before proposing changes.
4. Explain the plan briefly.
5. Make the smallest useful change.
6. Run the relevant check.
7. Summarize the diff and risks.

## Forbidden Without Approval

Do not do these without explicit approval:

- Change framework
- Add a major dependency
- Change deployment target
- Change database schema
- Change authentication
- Change payment logic
- Rewrite large files
- Delete major functionality
- Rename public routes
- Change environment variable names
- Touch production credentials

## Output Format After Work

Use this format:

### Changed
- ...

### Tested
- ...

### Risks
- ...

### Next
- ...

## Existing Repository Notes

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
