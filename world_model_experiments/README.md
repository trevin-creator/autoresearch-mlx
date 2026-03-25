# SNN + LeWorldModel + Dreamer-Like Experiments

This folder provides a practical scaffold for your requested stack:

1. **Spyx SNN feature extraction** from stereo event streams + IMU.
2. **LeWorldModel-style JEPA training** on those features.
3. **Dreamer-like latent planning** on top of the learned embedding dynamics.
4. **ONNX export** for deployment to targets such as NCNN on Raspberry Pi 5.

## Why this structure

- Keeps SNN front-end lightweight and compatible with event+IMU sensing.
- Moves world model + planning to software accelerators (non-FPGA), as requested.
- Matches LeWM spirit: next-embedding prediction + Gaussian-style latent regularization.

## Files

- `snn_feature_pipeline.py`: Spyx stereo/IMU feature extractor and HDF5 writer.
- `tumvie_local.py`: local TUMVIE event/IMU reader for the existing dataset files in this repo.
- `build_tumvie_feature_dataset.py`: CLI to generate feature/action HDF5 from real TUMVIE windows.
- `evaluate_tumvie_probe.py`: linear probe that tests whether the learned embedding predicts real pose deltas.
- `informed_dreamer_model.py`: informed world model + reward/continue heads + actor-critic with smooth action regularization.
- `train_informed_dreamer.py`: end-to-end training for the full informed Dreamer-style stack.
- `evaluate_informed_dreamer.py`: evaluates privileged decoder and reward/continue prediction errors.
- `run_informed_ablation.py`: one-command A/B run (with vs without flight-plan conditioning) and delta report.
- `run_informed_multiseed_report.py`: multi-seed A/B run that writes CSV + Markdown report artifacts.
- `motor_simulator.py`: lightweight 4-motor quad dynamics model with domain randomization helper.
- `motor_constraints.py`: shared action clamping and slew-rate constraints for motor commands.
- `build_sim_rollout_dataset.py`: simulator rollout dataset builder that writes `motor_commands` plus compatible keys.
- `evaluate_closed_loop_motor.py`: closed-loop simulator evaluation with crash/smoothness metrics.
- `run_motor_multiseed_report.py`: motor-mode multiseed train+eval pipeline producing CSV/Markdown artifacts.
- `validate_motor_onnx_parity.py`: PyTorch vs ONNXRuntime parity check for motor-mode JEPA predictor.
- `evaluate_motor_robustness.py`: disturbance robustness evaluator across calm/wind/gust/noisy scenarios.
- `run_motor_robustness_report.py`: per-seed robustness aggregator that writes CSV/Markdown artifacts.
- `benchmark_motor_onnx_runtime.py`: runtime latency/jitter and parity stress benchmark for ONNXRuntime.
- `check_phase35_gates.py`: fail-fast metric gates for closed-loop safety, robustness, and runtime/parity checks.
- `safety_shield.py`: runtime safety shield for bounded, slew-limited, emergency-stopped motor commands.
- `run_motor_curriculum_train.py`: staged disturbance curriculum training runner for motor-mode informed Dreamer.
- `validate_real_replay.py`: replay parity validator (predicted vs observed pose_delta/reward) with speed buckets.
- `check_phase4_release_gates.py`: stricter release-grade gate checker across safety, robustness, runtime, parity, replay.
- `command_interface.py`: normalized-command to PWM interface contract with bounded rate-limited outputs.
- `check_sensor_sync.py`: timestamp monotonicity/jitter/gap checker for sequence datasets.
- `evaluate_ood_guard.py`: feature-distribution OOD detector and fallback trigger-rate estimator.
- `fit_motor_dynamics.py`: coarse system-identification utility for motor-to-motion gains and delay.
- `generate_verification_matrix.py`: requirement-to-evidence markdown matrix generator from release logs.
- `evaluate_fault_injection_replay.py`: actuator fault-injection replay for fallback/arbitration resilience checks.
- `generate_deployment_manifest.py`: pinned deployment manifest writer with hashes, thresholds, and evidence summaries.
- `telemetry.py`: JSONL telemetry logger for replay and closed-loop runtime traces.
- `flight_state_machine.py`: explicit shadow/autonomous/fallback/emergency mode controller with flight-plan decoding.
- `fallback_controller.py`: conservative low-authority hover/land fallback controller.
- `autopilot_bridge.py`: planner/fallback arbitration plus safety shield and PWM command packet bridge.
- `run_shadow_mode_replay.py`: replay planner outputs in shadow mode with fallback arbitration and command packets.
- `lewm_feature_model.py`: Feature JEPA model (PyTorch).
- `train_feature_lewm.py`: Trainer for feature JEPA.
- `dreamer_like_planner.py`: CEM planner over imagined embedding rollouts.
- `export_onnx.py`: ONNX export for inference deployment.

## Expected data flow

```text
Tonic stereo-event + IMU windows
  -> Spyx stereo/IMU SNN feature vectors
  -> versioned parquet or HDF5 feature/action sequences + lineage manifest
  -> LeWM-style feature JEPA training
  -> embedding predictor checkpoint
  -> ONNX export
  -> Dreamer-like planner over embedding dynamics
```

## Managed datasets and lineage

The dataset builders now support managed outputs under `artifacts/datasets/...`
and automatically register lineage metadata in `artifacts/data_catalog/`.

Each derived dataset gets:

- a versioned data artifact (`.parquet` or `.h5`)
- an adjacent manifest file (`data.<ext>.manifest.json`)
- a row in the parquet registry (`artifacts/data_catalog/dataset_registry.parquet`)

The manifest records:

- `dataset_id`, `dataset_name`, and content-addressed `version`
- `source_uri` for raw provenance (local path, S3 URI, Hugging Face URI, simulator URI)
- `parent_ids` for upstream dataset lineage
- transform metadata and tensor schema

That gives MLflow, Optuna, or any higher-level autoresearch loop a stable dataset
identifier to log instead of a loose file path.

### Managed simulator dataset example

```bash
python -m world_model_experiments.build_sim_rollout_dataset \
  --output artifacts/datasets/sim_motor_rollouts/20260325T120000Z/data.parquet \
  --dataset-name sim_motor_rollouts \
  --source-uri simulator://quad_motor_dynamics
```

### Managed TUMVIE dataset example

```bash
python -m world_model_experiments.build_tumvie_feature_dataset \
  --recording-dir spyx/research/data/TUMVIE/mocap-6dof \
  --output artifacts/datasets/tumvie_features/20260325T120000Z/data.parquet \
  --dataset-name tumvie_features
```

Both commands emit a dataset manifest next to the data artifact and register the
dataset in the parquet catalog.

## MLflow-tracked multiseed runs

`run_informed_multiseed_report.py` can now open an MLflow parent run and nested
per-seed child runs while automatically logging the managed dataset reference.

```bash
python -m world_model_experiments.run_informed_multiseed_report \
  --dataset artifacts/datasets/tumvie_features/20260325T120000Z/data.parquet \
  --output-root artifacts/tumvie/informed_multiseed \
  --seeds 0,1,2 \
  --epochs 3 \
  --mlflow-experiment world_model/informed_multiseed \
  --mlflow-tracking-uri artifacts/mlruns
```

This logs:

- parent run configuration and aggregate deltas
- nested runs for each `(seed, variant)` pair
- the dataset manifest as an MLflow artifact
- artifact reports (`seed_metrics.csv`, `delta_summary.csv`, `report.md`)

## Materialize external datasets into lineage catalog

Use `materialize_dataset.py` to bring external sources (local files, `s3://`,
`hf://`, `http(s)://`) into the managed dataset tree with a manifest + catalog row.

```bash
python -m world_model_experiments.materialize_dataset \
  --source-uri s3://my-bucket/datasets/tumvie_features.parquet \
  --output artifacts/datasets/tumvie_features/20260325T140000Z/data.parquet \
  --dataset-name tumvie_features
```

For Hugging Face:

```bash
python -m world_model_experiments.materialize_dataset \
  --source-uri hf://org/dataset/train.parquet \
  --output artifacts/datasets/external/train.parquet \
  --dataset-name hf_train_split \
  --hf-repo-type dataset \
  --hf-revision main
```

## Optuna + MLflow search

`run_optuna_search.py` performs architecture and hyperparameter search for
informed Dreamer, with one nested MLflow run per trial and dataset lineage tags.

```bash
python -m world_model_experiments.run_optuna_search \
  --dataset artifacts/datasets/tumvie_features/20260325T120000Z/data.parquet \
  --output-root artifacts/search/informed_optuna \
  --study-name informed_dreamer_search \
  --n-trials 20 \
  --epochs 3 \
  --metric pose_delta_mse \
  --direction minimize \
  --mlflow-experiment world_model/optuna_informed \
  --mlflow-tracking-uri artifacts/mlruns
```

Optional: provide a JSON search-space spec with `--search-space-json`.

Example `search_space.json`:

```json
{
  "embed_dim": {"type": "int", "low": 64, "high": 192, "step": 32},
  "hidden_dim": {"type": "int", "low": 128, "high": 384, "step": 64},
  "horizon": {"type": "int", "low": 4, "high": 10, "step": 2},
  "lr": {"type": "float", "low": 0.0001, "high": 0.002, "log": true},
  "weight_decay": {"type": "float", "low": 0.000001, "high": 0.001, "log": true},
  "batch_size": {"type": "categorical", "choices": [8, 16, 32]}
}
```

Search space (current defaults):

- `embed_dim`: 64..256 (step 32)
- `hidden_dim`: 128..512 (step 64)
- `horizon`: 4..12 (step 2)
- `lr`: 1e-4..3e-3 (log)
- `weight_decay`: 1e-6..1e-3 (log)
- `batch_size`: {8, 16, 32}

## MLflow for standalone trainers

Both `train_feature_lewm.py` and `train_informed_dreamer.py` now support:

- `--mlflow-experiment`
- `--mlflow-tracking-uri`
- `--run-name`

Example:

```bash
python -m world_model_experiments.train_informed_dreamer \
  --dataset artifacts/datasets/tumvie_features/20260325T120000Z/data.parquet \
  --output-dir artifacts/tumvie/informed_dreamer \
  --epochs 5 \
  --batch-size 16 \
  --mlflow-experiment world_model/informed_single \
  --mlflow-tracking-uri artifacts/mlruns
```

The same MLflow controls are also available in high-level runners:

- `run_informed_ablation.py`
- `run_motor_multiseed_report.py`
- `run_motor_robustness_report.py`
- `run_motor_curriculum_train.py`

These runners create a parent run plus nested runs per variant/seed/stage and
automatically log dataset lineage tags when a managed dataset manifest exists.

## Quick start

Install dependencies in your experiment environment:

```bash
pip install -r world_model_experiments/requirements.txt
```

Generate a smoke dataset from synthetic windows (replace with real Tonic loader next):

```python
from world_model_experiments.snn_feature_pipeline import (
    SnnFeatureConfig,
    SpyxStereoImuFeatureExtractor,
    mock_tonic_stereo_imu_windows,
    windows_to_sequences,
    write_feature_hdf5,
)

cfg = SnnFeatureConfig(input_hw=(32, 32), input_channels=2, imu_dim=6)
extractor = SpyxStereoImuFeatureExtractor(cfg)
windows = mock_tonic_stereo_imu_windows(
    cfg=cfg,
    num_windows=512,
    timesteps=8,
    batch=1,
    action_dim=4,
)
seqs = windows_to_sequences(extractor, windows, sequence_len=16, action_dim=4)
write_feature_hdf5(seqs, "artifacts/feature_data/smoke_features.h5")
```

Train feature JEPA:

```bash
python -m world_model_experiments.train_feature_lewm \
  --dataset artifacts/feature_data/smoke_features.h5 \
  --output-dir artifacts/feature_lewm \
  --epochs 20 --batch-size 64
```

Export ONNX:

```bash
python -m world_model_experiments.export_onnx \
  --checkpoint artifacts/feature_lewm/feature_lewm_best.pt \
  --output artifacts/feature_lewm/feature_lewm.onnx
```

## Real TUMVIE run

This repo already contains local TUMVIE files under `spyx/research/data/TUMVIE/mocap-6dof/`.
The local builder reads:

- `mocap-6dof-events_left.h5`
- `mocap-6dof-events_right.h5`
- `imu_data.txt`
- `mocap_data.txt`

and builds windowed stereo-event + IMU sequences without triggering Tonic downloads.

Adaptation from SkyDreamer:

- We add a flight-plan vector that encodes future relative and absolute waypoints
  (position + yaw) over a configurable horizon.
- This vector is stored as `flight_plan` in the dataset and can be concatenated to
  `actions` during training and probing using `--use-flight-plan`.
- We add informed decoder targets (`pose`, `pose_delta`) plus `reward` and
  `continue` heads to train latent imagination with actor-critic.
- We apply smoothness regularization on actor action means during imagined rollouts.

```bash
python -m world_model_experiments.build_tumvie_feature_dataset \
  --recording-dir spyx/research/data/TUMVIE/mocap-6dof \
  --output artifacts/tumvie/tumvie_features.h5 \
  --sample-t 4 \
  --window-us 20000 \
  --stride-us 40000 \
  --downsample-factor 0.05 \
  --max-windows 16 \
  --sequence-len 4 \
  --flight-plan-horizon 3

python -m world_model_experiments.train_feature_lewm \
  --dataset artifacts/tumvie/tumvie_features.h5 \
  --output-dir artifacts/tumvie/feature_lewm \
  --epochs 1 \
  --batch-size 2 \
  --history-size 2 \
  --embed-dim 32 \
  --hidden-dim 64 \
  --depth 1 \
  --heads 4 \
  --use-flight-plan

python -m world_model_experiments.evaluate_tumvie_probe \
  --dataset artifacts/tumvie/tumvie_features.h5 \
  --checkpoint artifacts/tumvie/feature_lewm/feature_lewm_best.pt \
  --use-flight-plan
```

The generated TUMVIE HDF5 now includes:

- `features`: SNN-derived feature vectors
- `actions`: IMU-summary action proxy
- `pose`: aligned 6DOF pose
- `pose_delta`: aligned pose change between accepted windows
- `timestamps_us`: original recording timestamps
- `flight_plan`: future relative+absolute waypoint summary (SkyDreamer-style adaptation)
- `reward`: progress proxy derived from pose delta
- `continue`: continuation target for imagined rollouts

## Full informed-dreamer run (paper-adapted)

```bash
python -m world_model_experiments.train_informed_dreamer \
  --dataset artifacts/tumvie/tumvie_features.h5 \
  --output-dir artifacts/tumvie/informed_dreamer \
  --epochs 5 \
  --batch-size 8 \
  --embed-dim 128 \
  --hidden-dim 192 \
  --horizon 8 \
  --use-flight-plan

python -m world_model_experiments.evaluate_informed_dreamer \
  --dataset artifacts/tumvie/tumvie_features.h5 \
  --checkpoint artifacts/tumvie/informed_dreamer/informed_dreamer_best.pt \
  --use-flight-plan

python -m world_model_experiments.run_informed_ablation \
  --dataset artifacts/tumvie/tumvie_features.h5 \
  --output-root artifacts/tumvie/informed_ablation \
  --epochs 1 \
  --batch-size 8

python -m world_model_experiments.run_informed_multiseed_report \
  --dataset artifacts/tumvie/tumvie_features.h5 \
  --output-root artifacts/tumvie/informed_multiseed \
  --seeds 0,1,2 \
  --epochs 3 \
  --batch-size 16
```

## Motor-action MVP (sim rollout data)

This path switches training action semantics from IMU-summary proxies to true
4D motor commands generated by simulator rollouts.

```bash
python -m world_model_experiments.build_sim_rollout_dataset \
  --output artifacts/sim/sim_motor_rollouts.h5 \
  --num-sequences 128 \
  --sequence-len 16 \
  --feature-dim 79 \
  --seed 0

python -m world_model_experiments.train_feature_lewm \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --output-dir artifacts/sim/feature_lewm_motor \
  --epochs 5 \
  --batch-size 16 \
  --use-motor-commands

python -m world_model_experiments.train_informed_dreamer \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --output-dir artifacts/sim/informed_dreamer_motor \
  --epochs 5 \
  --batch-size 16 \
  --use-motor-commands

python -m world_model_experiments.evaluate_informed_dreamer \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --checkpoint artifacts/sim/informed_dreamer_motor/informed_dreamer_best.pt \
  --use-motor-commands

python -m world_model_experiments.evaluate_closed_loop_motor \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --checkpoint artifacts/sim/informed_dreamer_motor/informed_dreamer_best.pt \
  --episodes 8 \
  --horizon 16 \
  --use-motor-commands

python -m world_model_experiments.validate_motor_onnx_parity \
  --checkpoint artifacts/sim/feature_lewm_motor/feature_lewm_best.pt \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --output artifacts/sim/feature_lewm_motor/feature_lewm_motor.onnx \
  --use-motor-commands

python -m world_model_experiments.run_motor_multiseed_report \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --output-root artifacts/sim/motor_multiseed \
  --seeds 0,1,2 \
  --epochs 3 \
  --batch-size 16

python -m world_model_experiments.evaluate_motor_robustness \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --checkpoint artifacts/sim/informed_dreamer_motor/informed_dreamer_best.pt \
  --episodes 8 \
  --horizon 8 \
  --use-motor-commands

python -m world_model_experiments.run_motor_robustness_report \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --checkpoint-root artifacts/sim/motor_multiseed \
  --output-root artifacts/sim/motor_robustness \
  --seeds 0,1,2 \
  --episodes 8 \
  --horizon 8

python -m world_model_experiments.run_motor_robustness_report \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --checkpoint-root artifacts/sim/motor_multiseed \
  --output-root artifacts/sim/motor_robustness_matrix \
  --seeds 0,1,2 \
  --episodes 8 \
  --horizon 8 \
  --scenario-mode matrix \
  --wind-stds 0.0,0.5,1.0 \
  --act-noise-stds 0.0,0.08,0.16 \
  --latency-steps 0,1,2

python -m world_model_experiments.benchmark_motor_onnx_runtime \
  --checkpoint artifacts/sim/feature_lewm_motor/feature_lewm_best.pt \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --output artifacts/sim/feature_lewm_motor/feature_lewm_motor_runtime.onnx \
  --batches 16 \
  --warmup-runs 5 \
  --timed-runs 20 \
  --csv-output artifacts/sim/feature_lewm_motor/runtime_trend.csv \
  --tag local-dev \
  --use-motor-commands

python -m world_model_experiments.check_phase35_gates \
  --closed-loop-log /tmp/ci_closed_loop.log \
  --robust-log /tmp/ci_robust.log \
  --runtime-log /tmp/ci_runtime.log \
  --onnx-log /tmp/ci_onnx_parity.log \
  --max-crash-rate 0.05 \
  --max-termination-rate 0.05 \
  --min-survival 0.85 \
  --max-latency-p95-ms 8.0 \
  --max-parity-diff 1e-4

python -m world_model_experiments.run_motor_curriculum_train \
  --output-root artifacts/sim/motor_curriculum \
  --num-sequences 64 \
  --sequence-len 16 \
  --epochs-per-stage 2 \
  --batch-size 16 \
  --seed 0

python -m world_model_experiments.evaluate_closed_loop_motor \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --checkpoint artifacts/sim/informed_dreamer_motor/informed_dreamer_best.pt \
  --episodes 8 \
  --horizon 16 \
  --use-motor-commands \
  --use-safety-shield

python -m world_model_experiments.evaluate_motor_robustness \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --checkpoint artifacts/sim/informed_dreamer_motor/informed_dreamer_best.pt \
  --episodes 8 \
  --horizon 8 \
  --scenario-mode matrix \
  --wind-stds 0.0,0.5,1.0 \
  --act-noise-stds 0.0,0.08,0.16 \
  --latency-steps 0,1,2 \
  --use-motor-commands \
  --use-safety-shield

python -m world_model_experiments.benchmark_motor_onnx_runtime \
  --checkpoint artifacts/sim/feature_lewm_motor/feature_lewm_best.pt \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --output artifacts/sim/feature_lewm_motor/feature_lewm_motor_runtime.onnx \
  --batches 16 \
  --warmup-runs 5 \
  --timed-runs 20 \
  --repeats 3 \
  --csv-output artifacts/sim/feature_lewm_motor/runtime_trend.csv \
  --tag release \
  --use-motor-commands

python -m world_model_experiments.validate_real_replay \
  --dataset artifacts/tumvie/tumvie_features.h5 \
  --checkpoint artifacts/tumvie/informed_dreamer/informed_dreamer_best.pt \
  --use-flight-plan

python -m world_model_experiments.check_phase4_release_gates \
  --closed-loop-log /tmp/release_closed_loop.log \
  --robust-log /tmp/release_robust.log \
  --runtime-log /tmp/release_runtime.log \
  --onnx-log /tmp/release_onnx.log \
  --replay-log /tmp/release_replay.log \
  --max-crash-rate 0.01 \
  --max-termination-rate 0.01 \
  --min-survival 0.95 \
  --max-latency-p95-ms 6.0 \
  --max-parity-diff 1e-4 \
  --max-replay-pose-delta-mse 0.01 \
  --max-replay-reward-mse 0.02 \
  --max-shield-emergency-rate 0.01

python -m world_model_experiments.check_sensor_sync \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --max-jitter-us 5000 \
  --max-gap-us 50000

python -m world_model_experiments.evaluate_ood_guard \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --z-threshold 4.0 \
  --max-ood-rate 0.10

python -m world_model_experiments.fit_motor_dynamics \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --output artifacts/sim/system_id/motor_dynamics.json \
  --max-delay 4

python -m world_model_experiments.generate_verification_matrix \
  --closed-loop-log /tmp/release_closed_loop.log \
  --robust-log /tmp/release_robust.log \
  --runtime-log /tmp/release_runtime.log \
  --replay-log /tmp/release_replay.log \
  --shadow-log /tmp/release_shadow.log \
  --sync-log /tmp/release_sync.log \
  --ood-log /tmp/release_ood.log \
  --system-id-log /tmp/release_system_id.log \
  --output artifacts/sim/verification/phase5_verification_matrix.md

python -m world_model_experiments.run_shadow_mode_replay \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --checkpoint artifacts/sim/informed_dreamer_motor/informed_dreamer_best.pt \
  --episodes 8 \
  --ood-threshold 0.10 \
  --shadow-warmup-steps 4 \
  --telemetry-output artifacts/sim/telemetry/shadow_replay.jsonl \
  --use-motor-commands

python -m world_model_experiments.evaluate_fault_injection_replay \
  --dataset artifacts/sim/sim_motor_rollouts.h5 \
  --checkpoint artifacts/sim/informed_dreamer_motor/informed_dreamer_best.pt \
  --episodes 8 \
  --ood-threshold 0.10 \
  --fault-mode stuck_low \
  --fault-strength 0.7 \
  --fault-start-step 2 \
  --telemetry-output artifacts/sim/telemetry/fault_replay.jsonl \
  --use-motor-commands

python -m world_model_experiments.generate_deployment_manifest \
  --checkpoint artifacts/sim/informed_dreamer_motor/informed_dreamer_best.pt \
  --onnx-model artifacts/sim/feature_lewm_motor/feature_lewm_motor_runtime.onnx \
  --release-gates-log /tmp/release_gate.log \
  --shadow-log /tmp/release_shadow.log \
  --fault-log /tmp/release_fault.log \
  --version-tag local-dev \
  --output artifacts/sim/release/deployment_manifest.json

python -m world_model_experiments.check_phase4_release_gates \
  --closed-loop-log /tmp/release_closed_loop.log \
  --robust-log /tmp/release_robust.log \
  --runtime-log /tmp/release_runtime.log \
  --onnx-log /tmp/release_onnx.log \
  --replay-log /tmp/release_replay.log \
  --shadow-log /tmp/release_shadow.log \
  --fault-log /tmp/release_fault.log \
  --sync-log /tmp/release_sync.log \
  --ood-log /tmp/release_ood.log \
  --system-id-log /tmp/release_system_id.log \
  --manifest-path /tmp/release_deployment_manifest.json \
  --max-shadow-fallback-rate 0.50 \
  --max-shadow-emergency-rate 0.01 \
  --min-shadow-autonomous-rate 0.50 \
  --max-shadow-emergency-stop-rate 0.01 \
  --min-fault-fallback-rate 0.60 \
  --max-fault-emergency-stop-rate 0.01
```

## Integrating real Tonic stereo+IMU data

Replace `mock_tonic_stereo_imu_windows(...)` with an adapter that yields `StereoImuBatch`:

- `left_events`: `[T, B, H, W, C]`
- `right_events`: `[T, B, H, W, C]`
- `imu`: `[T, B, I]`
- `actions`: `[B, A]` or `None` if unavailable

Then reuse the same extraction and training scripts.

## Deployment notes (RPi5 + NCNN)

- Export ONNX with static or constrained sequence length for easier conversion.
- Prefer FP16 inference if your runtime supports it.
- Keep `embed_dim` moderate (e.g., 128-256) for latency.
- Use short planning horizons (8-16) for real-time Dreamer-like control loops.
