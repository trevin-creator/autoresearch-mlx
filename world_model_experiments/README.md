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
- `lewm_feature_model.py`: Feature JEPA model (PyTorch).
- `train_feature_lewm.py`: Trainer for feature JEPA.
- `dreamer_like_planner.py`: CEM planner over imagined embedding rollouts.
- `export_onnx.py`: ONNX export for inference deployment.

## Expected data flow

```text
Tonic stereo-event + IMU windows
  -> Spyx stereo/IMU SNN feature vectors
  -> HDF5 feature/action sequences
  -> LeWM-style feature JEPA training
  -> embedding predictor checkpoint
  -> ONNX export
  -> Dreamer-like planner over embedding dynamics
```

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
    num_windows=512,
    timesteps=8,
    batch=1,
    hw=(32, 32),
    channels=2,
    imu_dim=6,
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
