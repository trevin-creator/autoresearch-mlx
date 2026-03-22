# Data Generation — Stereo Event Camera Simulation

Synthetic training data for the autoresearch-mlx SNN pipeline. Uses the
[Genesis](https://genesis-world.readthedocs.io/) 3D simulator to produce
stereo event streams, depth maps, IMU readings, and ground-truth poses that
feed directly into the `spyx_mlx` training workflow.

## Pipeline overview

```
Genesis 3D scene
    ↓  high-rate stereo RGB + depth renders
Per-pixel ESIM event model (datagen.emulator)
    ↓  (t_us, x, y, polarity) events per eye
Dataset writer (datagen.writer)
    ↓  .npz events, .npy depth, CSV poses/IMU, calibration JSON
prepare_vision.py / custom loader
    ↓  bin events into frames, build (B, T, C, H, W) tensors
train_vision.py / train_snn.py (MLX GPU via spyx_mlx)
```

The event model follows the ESIM principle: tight coupling between the
renderer and a per-pixel log-intensity threshold emulator with per-pixel
threshold mismatch, refractory period, and timestamp interpolation.

## Relationship to the project

| Component | Role |
|---|---|
| `datagen/` | Offline data generation (NumPy + Genesis). Runs on any machine with Genesis installed. |
| `generate_stereo_events.py` | CLI entry point. Config-driven via dataclass defaults + argparse overrides. Same pattern as `train_snn.py`. |
| `spyx_mlx/` | MLX-native SNN training library. Consumes the generated `.npz` event data. |
| `train_snn.py` / `train_vision.py` | Downstream training scripts that load the generated dataset. All training on MLX GPU — non-negotiable. |

Genesis is **not** required at training time. Data generation is a separate
offline step that produces standard NumPy files loadable on any platform.

## v1 spec

| Parameter | Value | Notes |
|---|---|---|
| Resolution | 346 × 260 | Matches DVS346 sensor resolution |
| Stereo baseline | 0.10 m | ~human-eye scale |
| C_pos, C_neg | 0.18 | Typical event camera contrast thresholds |
| Threshold mismatch | Gaussian σ 0.02 | Per-pixel variation |
| Refractory period | 200 µs | Prevents event avalanche |
| Micro-step dt | 1 ms | Higher rate = more fidelity, more events |
| Event format | `(t_us, x, y, polarity)` | Standard int64 tuple per event |
| Timestamp interpolation | Enabled | Sub-step event timing via linear interp |

## Usage

```bash
# Generate 10 seconds of stereo event data (default settings)
uv run generate_stereo_events.py

# Override parameters
uv run generate_stereo_events.py \
    --duration 30.0 \
    --dt 0.0005 \
    --width 320 --height 240 \
    --baseline 0.08 \
    --out-dir ./my_dataset \
    --save-rgb

# Then train on the generated data
uv run train_vision.py --data-dir ./out_genesis_stereo_events
```

## Output layout

```
out_genesis_stereo_events/
  events_left/000000.npz, 000001.npz, ...
  events_right/000000.npz, 000001.npz, ...
  depth_left/000000.npy, ...
  depth_right/000000.npy, ...
  rgb_left_preview/  (optional, --save-rgb)
  rgb_right_preview/ (optional)
  meta/
    calibration.json   — pinhole intrinsics, stereo extrinsics, timing
    poses.csv          — per-frame rig + left/right camera poses (quat)
    imu.csv            — per-frame body-frame accel + gyro
    frames.csv         — per-frame file path index
```

Each `events_*.npz` contains `events` with shape `(N, 4)` and dtype int64:
columns are `[t_us, x, y, polarity]`.

## Module structure — `datagen/`

```
datagen/
  __init__.py       — public API re-exports
  config.py         — CameraConfig, StereoConfig, EventConfig, SimConfig, OutputConfig
  math_utils.py     — normalize, lookat_rotation, quat_from_rotmat, rotmat_to_rvec
  trajectory.py     — ScriptedRigTrajectory (smooth forward flight + lateral oscillation)
  emulator.py       — EventCameraEmulator (ESIM-style per-pixel threshold model)
  writer.py         — DatasetWriter (events/depth/pose/IMU to disk)
  scene.py          — GenesisStereoEventDataset (Genesis world + stereo rig + main loop)
```

The emulator and writer work standalone without Genesis — useful for
unit-testing or converting existing video frames to events.

## Design decisions

**Why Genesis?** It provides camera, depth, segmentation, IMU, and physics
in a single simulator. Cameras render RGB + depth directly; no external
rendering pipeline needed.

**Why ESIM-style?** The per-pixel log-intensity threshold model is the
standard for synthetic event generation. It matches the physics of how DVS
pixels respond to brightness changes, including multi-event emission for
large brightness swings.

**Why not MLX for data generation?** The event emulator is inherently
serial per-pixel (threshold crossings, refractory state). NumPy is the
right tool here. MLX is reserved for the training path where batch
parallelism pays off.

**Config dataclasses** follow the same `@dataclass(frozen=True)` pattern
used by `ExperimentConfig` in `train_snn.py`, with argparse CLI overrides.

## Known limitations and next steps

1. **Fidelity is limited by sim_dt** — renders once per step. Adaptive
   sub-stepping when |Δlog I| is large would improve realism.
2. **No photoreceptor low-pass** — a first-order filter on log intensity
   would model the DVS analog frontend.
3. **No leak/shot noise** — background activity rate and hot pixels are
   not yet modeled.
4. **Event loop is Python** — the per-pixel iteration is not vectorized.
   For large resolutions or long sequences, a Cython or NumPy-vectorized
   version would speed things up significantly.
5. **Ideal IMU** — derived from the scripted trajectory, not from Genesis
   IMU sensor. Swap to `scene.add_sensor(gs.sensors.IMU(...))` when using
   a physics-driven rig.
6. **Static scene geometry** — replace the placeholder boxes/cylinder/sphere
   with realistic Genesis world assets, moving objects, and lighting
   variation for domain randomization.
