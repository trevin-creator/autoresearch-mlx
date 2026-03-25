# Contributing

## Scope
This repository contains multiple experiment tracks. Most active Python changes in this workspace target `world_model_experiments/`.

## Development Setup
1. Use a local virtual environment (for example `.venv`).
2. Install required dependencies for the area you are modifying.
3. Keep generated artifacts out of commits.

## Change Guidelines
1. Prefer small, focused commits.
2. Keep behavior-compatible refactors separate from functional fixes.
3. Avoid broad formatting-only churn in unrelated files.

## Local Validation
Run targeted checks before committing:

```bash
python3 -m py_compile world_model_experiments/<file>.py
```

When editing multiple files, compile-check each edited module.

## Pre-commit
This repository uses pre-commit hooks. Install and run them locally:

```bash
pre-commit install
pre-commit run --all-files
```

## Pull Request Notes
1. Summarize what changed and why.
2. List validation steps you ran.
3. Call out follow-up work if any checks were intentionally deferred.
