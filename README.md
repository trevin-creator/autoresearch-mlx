# autoresearch-mlx

Archived monolith repo. Active development has moved to focused sibling repos:

- `neuromorph-train` — SNN training, search, calibration, benchmarks
- `neuromorph-pipeline` — Dagster orchestration, DVC stages, pipeline params
- `autoresearch-agent` — research protocols and automation scaffold
- `eventcam` — stereo event-camera data generation
- `snn-ir` — SNN intermediate representation and RTL/codegen
- `spyx` — JAX Spyx fork including `spyx_mlx`

## Local workspace layout

```text
/Users/vincent/Work/
├── autoresearch-mlx        # archive / redirect only
├── neuromorph-train
├── neuromorph-pipeline
├── autoresearch-agent
├── eventcam
├── snn-ir
└── spyx
```

Open `/Users/vincent/Work/neuromorph.code-workspace` to work across the split repos.

## Repo mapping

- Training/search code moved to `neuromorph-train`
- Pipeline/orchestration code moved to `neuromorph-pipeline`
- GPT-era research loop and protocols moved to `autoresearch-agent`
- Data-generation helpers/assets moved to `eventcam`
- `spyx_mlx` duplication was removed in favor of the `spyx` fork

This repo is no longer the active development target.
