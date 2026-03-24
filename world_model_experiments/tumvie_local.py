from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import hdf5plugin  # noqa: F401  Registers Blosc HDF5 filters.
import numpy as np

from world_model_experiments.snn_feature_pipeline import (
    SnnFeatureConfig,
    SpyxStereoImuFeatureExtractor,
    StereoImuBatch,
)


TUMVIE_SENSOR_HW = (720, 1280)


@dataclass(frozen=True)
class TumvieWindowConfig:
    recording_dir: str | Path
    sample_t: int = 8
    window_us: int = 50_000
    stride_us: int = 50_000
    downsample_factor: float = 0.1
    max_windows: int = 64
    imu_dim: int = 6


@dataclass(frozen=True)
class TumvieWindowSample:
    batch: StereoImuBatch
    end_us: int
    pose: np.ndarray
    pose_delta: np.ndarray


def _quat_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quat
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    rx = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    ry = np.arcsin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    rz = np.arctan2(siny_cosp, cosy_cosp)
    return np.asarray([rx, ry, rz], dtype=np.float32)


class _TumvieEventStream:
    def __init__(self, file_path: str | Path):
        self.file = h5py.File(file_path, "r")
        self.events = self.file["events"]
        self.t = self.events["t"]
        self.x = self.events["x"]
        self.y = self.events["y"]
        self.p = self.events["p"]
        self.ms_to_idx = self.file["ms_to_idx"]
        self.length = int(self.t.shape[0])

    def _candidate_bounds(self, start_us: int, end_us: int) -> tuple[int, int]:
        start_ms = max(0, min(int(start_us // 1000), len(self.ms_to_idx) - 1))
        end_ms = max(0, min(int(end_us // 1000) + 1, len(self.ms_to_idx) - 1))
        start_idx = int(self.ms_to_idx[start_ms])
        end_idx = int(self.ms_to_idx[end_ms]) if end_ms < len(self.ms_to_idx) else self.length
        if end_idx <= start_idx:
            end_idx = min(self.length, start_idx + 1)
        return start_idx, end_idx

    def slice_events(self, start_us: int, end_us: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lo, hi = self._candidate_bounds(start_us, end_us)
        t_block = np.asarray(self.t[lo:hi], dtype=np.int64)
        if t_block.size == 0:
            empty_i64 = np.empty((0,), dtype=np.int64)
            empty_u16 = np.empty((0,), dtype=np.uint16)
            empty_u8 = np.empty((0,), dtype=np.uint8)
            return empty_i64, empty_u16, empty_u16, empty_u8

        rel_lo = int(np.searchsorted(t_block, start_us, side="left"))
        rel_hi = int(np.searchsorted(t_block, end_us, side="left"))
        start = lo + rel_lo
        stop = lo + rel_hi
        if stop <= start:
            empty_i64 = np.empty((0,), dtype=np.int64)
            empty_u16 = np.empty((0,), dtype=np.uint16)
            empty_u8 = np.empty((0,), dtype=np.uint8)
            return empty_i64, empty_u16, empty_u16, empty_u8

        return (
            np.asarray(self.t[start:stop], dtype=np.int64),
            np.asarray(self.x[start:stop], dtype=np.uint16),
            np.asarray(self.y[start:stop], dtype=np.uint16),
            np.asarray(self.p[start:stop], dtype=np.uint8),
        )


def _load_imu_series(recording_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    imu = np.loadtxt(recording_dir / "imu_data.txt", comments="#", dtype=np.float64)
    return imu[:, 0].astype(np.int64), imu[:, 1:7].astype(np.float32)


def _load_mocap_series(recording_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    mocap = np.loadtxt(recording_dir / "mocap_data.txt", comments="#", dtype=np.float64)
    times = mocap[:, 0].astype(np.int64)
    positions = mocap[:, 1:4].astype(np.float32)
    euler = np.stack([_quat_to_euler_xyz(q) for q in mocap[:, 4:8]], axis=0)
    poses = np.concatenate([positions, euler], axis=-1)
    return times, poses.astype(np.float32)


def _downsample_hw(factor: float) -> tuple[int, int]:
    h = max(1, int(round(TUMVIE_SENSOR_HW[0] * factor)))
    w = max(1, int(round(TUMVIE_SENSOR_HW[1] * factor)))
    return h, w


def _rasterize_events(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    start_us: int,
    end_us: int,
    sample_t: int,
    output_hw: tuple[int, int],
) -> np.ndarray:
    h_out, w_out = output_hw
    frames = np.zeros((sample_t, h_out, w_out, 2), dtype=np.float32)
    if t.size == 0:
        return frames

    duration_us = max(1, end_us - start_us)
    bins = np.minimum(((t - start_us) * sample_t) // duration_us, sample_t - 1).astype(np.int64)
    x_ds = np.minimum((x.astype(np.int64) * w_out) // TUMVIE_SENSOR_HW[1], w_out - 1)
    y_ds = np.minimum((y.astype(np.int64) * h_out) // TUMVIE_SENSOR_HW[0], h_out - 1)
    pol = np.clip(p.astype(np.int64), 0, 1)

    np.add.at(frames, (bins, y_ds, x_ds, pol), 1.0)
    return np.minimum(frames, 1.0)


def _interp_series(sample_times_us: np.ndarray, times_us: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros((sample_times_us.shape[0], values.shape[1]), dtype=np.float32)
    if times_us.size == 0:
        return out
    for dim in range(values.shape[1]):
        out[:, dim] = np.interp(sample_times_us, times_us, values[:, dim]).astype(np.float32)
    return out


def iter_tumvie_windows(cfg: TumvieWindowConfig) -> Iterator[TumvieWindowSample]:
    recording_dir = Path(cfg.recording_dir)
    left_stream = _TumvieEventStream(recording_dir / f"{recording_dir.name}-events_left.h5")
    right_stream = _TumvieEventStream(recording_dir / f"{recording_dir.name}-events_right.h5")
    imu_times, imu_values = _load_imu_series(recording_dir)
    mocap_times, poses = _load_mocap_series(recording_dir)

    output_hw = _downsample_hw(cfg.downsample_factor)
    sample_offsets = np.linspace(0, cfg.window_us, cfg.sample_t, endpoint=False, dtype=np.float64)
    used = 0
    last_end_us = -1
    prev_pose: np.ndarray | None = None

    min_time = max(int(imu_times[0]), int(left_stream.t[0]), int(right_stream.t[0]))
    for idx, end_us in enumerate(mocap_times):
        if end_us < min_time + cfg.window_us:
            continue
        if last_end_us >= 0 and end_us - last_end_us < cfg.stride_us:
            continue

        start_us = int(end_us - cfg.window_us)
        sample_times = (start_us + sample_offsets).astype(np.int64)

        left_t, left_x, left_y, left_p = left_stream.slice_events(start_us, int(end_us))
        right_t, right_x, right_y, right_p = right_stream.slice_events(start_us, int(end_us))
        if left_t.size == 0 and right_t.size == 0:
            continue

        left_frames = _rasterize_events(left_t, left_x, left_y, left_p, start_us, int(end_us), cfg.sample_t, output_hw)
        right_frames = _rasterize_events(right_t, right_x, right_y, right_p, start_us, int(end_us), cfg.sample_t, output_hw)
        imu_seq = _interp_series(sample_times, imu_times, imu_values)
        action = np.mean(imu_seq, axis=0, keepdims=True).astype(np.float32)
        pose = poses[idx].astype(np.float32)
        pose_delta = np.zeros_like(pose) if prev_pose is None else (pose - prev_pose).astype(np.float32)

        yield TumvieWindowSample(
            batch=StereoImuBatch(
                left_events=left_frames[:, None, ...],
                right_events=right_frames[:, None, ...],
                imu=imu_seq[:, None, :],
                actions=action,
            ),
            end_us=int(end_us),
            pose=pose,
            pose_delta=pose_delta,
        )

        used += 1
        last_end_us = int(end_us)
        prev_pose = pose
        if used >= cfg.max_windows:
            break


def build_tumvie_feature_dataset(
    tumvie_cfg: TumvieWindowConfig,
    snn_cfg: SnnFeatureConfig,
    sequence_len: int,
    output_path: str | Path,
) -> Path:
    extractor = SpyxStereoImuFeatureExtractor(snn_cfg)
    samples = list(iter_tumvie_windows(tumvie_cfg))
    if not samples:
        raise ValueError("No TUMVIE windows were generated; check timing and recording availability.")

    features_per_window: list[np.ndarray] = []
    actions_per_window: list[np.ndarray] = []
    pose_per_window: list[np.ndarray] = []
    pose_delta_per_window: list[np.ndarray] = []
    timestamps_per_window: list[int] = []

    for sample in samples:
        vec = extractor.extract(sample.batch)
        features_per_window.append(vec[0].astype(np.float32))
        actions_per_window.append(np.asarray(sample.batch.actions, dtype=np.float32)[0])
        pose_per_window.append(sample.pose.astype(np.float32))
        pose_delta_per_window.append(sample.pose_delta.astype(np.float32))
        timestamps_per_window.append(sample.end_us)

    if len(features_per_window) < sequence_len:
        raise ValueError("No TUMVIE sequences were generated; try increasing max_windows or reducing sequence_len.")

    seq_features = []
    seq_actions = []
    seq_pose = []
    seq_pose_delta = []
    seq_timestamps = []
    for start in range(0, len(features_per_window) - sequence_len + 1, sequence_len):
        end = start + sequence_len
        seq_features.append(np.stack(features_per_window[start:end], axis=0))
        seq_actions.append(np.stack(actions_per_window[start:end], axis=0))
        seq_pose.append(np.stack(pose_per_window[start:end], axis=0))
        seq_pose_delta.append(np.stack(pose_delta_per_window[start:end], axis=0))
        seq_timestamps.append(np.asarray(timestamps_per_window[start:end], dtype=np.int64))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as h5:
        h5.create_dataset("features", data=np.stack(seq_features, axis=0), compression="gzip")
        h5.create_dataset("actions", data=np.stack(seq_actions, axis=0), compression="gzip")
        h5.create_dataset("pose", data=np.stack(seq_pose, axis=0), compression="gzip")
        h5.create_dataset("pose_delta", data=np.stack(seq_pose_delta, axis=0), compression="gzip")
        h5.create_dataset("timestamps_us", data=np.stack(seq_timestamps, axis=0), compression="gzip")
    return output