from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from spyx import fpga_models as fm


@dataclass(frozen=True)
class StereoImuBatch:
    """One time-windowed stereo event + IMU batch.

    Shapes:
    - left_events: [T, B, H, W, C]
    - right_events: [T, B, H, W, C]
    - imu: [T, B, I]
    - actions: [B, A] optional action aligned to the window end
    """

    left_events: np.ndarray
    right_events: np.ndarray
    imu: np.ndarray
    actions: np.ndarray | None = None


@dataclass(frozen=True)
class FeatureSequenceExample:
    """Episode-like sequence for LeWM-style training.

    Shapes:
    - features: [T, D]
    - actions: [T, A]
    """

    features: np.ndarray
    actions: np.ndarray


@dataclass(frozen=True)
class SnnFeatureConfig:
    input_hw: tuple[int, int]
    input_channels: int
    imu_dim: int
    stereo_channels: int = 16
    visual_channels1: int = 16
    visual_channels2: int = 24
    feature_dim: int = 64


class SpyxStereoImuFeatureExtractor:
    """Extract compact feature vectors with Spyx stereo + IMU modules."""

    def __init__(self, cfg: SnnFeatureConfig) -> None:
        self.cfg = cfg
        self._forward = hk.without_apply_rng(hk.transform(self._build_forward))
        left = jnp.zeros((4, 1, cfg.input_hw[0], cfg.input_hw[1], cfg.input_channels), dtype=jnp.float32)
        right = jnp.zeros_like(left)
        imu = jnp.zeros((4, 1, cfg.imu_dim), dtype=jnp.float32)
        self.params = self._forward.init(jax.random.PRNGKey(0), left, right, imu)

    def _build_forward(self, left_seq: jnp.ndarray, right_seq: jnp.ndarray, imu_seq: jnp.ndarray):
        stereo_cfg = fm.StereoCoincidenceConfig(
            input_hw=self.cfg.input_hw,
            input_channels=self.cfg.input_channels,
            channels=self.cfg.stereo_channels,
            output_dim=self.cfg.feature_dim,
        )
        visual_cfg = fm.ConvConfig(
            input_hw=self.cfg.input_hw,
            input_channels=self.cfg.input_channels,
            channels1=self.cfg.visual_channels1,
            channels2=self.cfg.visual_channels2,
            output_dim=self.cfg.feature_dim,
        )
        imu_cfg = fm.IMUConditionedConfig(
            vision_cfg=visual_cfg,
            imu_dim=self.cfg.imu_dim,
            imu_hidden=max(16, self.cfg.feature_dim // 2),
            gating="late",
        )

        stereo = fm.StereoCoincidenceSNN(stereo_cfg)
        imu_fused = fm.IMUConditionedVisualSNN(imu_cfg)

        stereo_logits, stereo_aux = stereo(left_seq, right_seq)
        vis_seq = jnp.abs(left_seq - right_seq)
        imu_logits, imu_aux = imu_fused(vis_seq, imu_seq)

        return {
            "stereo_logits": stereo_logits,
            "imu_logits": imu_logits,
            "stereo_rate": stereo_aux["spike_rate"],
            "imu_rate": imu_aux["spike_rate"],
        }

    def extract(self, batch: StereoImuBatch) -> np.ndarray:
        left = jnp.asarray(batch.left_events, dtype=jnp.float32)
        right = jnp.asarray(batch.right_events, dtype=jnp.float32)
        imu = jnp.asarray(batch.imu, dtype=jnp.float32)
        outputs = self._forward.apply(self.params, left, right, imu)
        return build_feature_vector(outputs, imu)


def build_feature_vector(outputs: dict[str, jnp.ndarray], imu_seq: jnp.ndarray) -> np.ndarray:
    """Build a dense feature vector per batch item from model outputs."""

    stereo_logits = np.asarray(outputs["stereo_logits"], dtype=np.float32)
    imu_logits = np.asarray(outputs["imu_logits"], dtype=np.float32)
    batch = stereo_logits.shape[0]

    disparity = np.zeros((batch, 1), dtype=np.float32)

    stereo_rate = float(np.asarray(outputs["stereo_rate"]).reshape(-1)[0])
    imu_rate = float(np.asarray(outputs["imu_rate"]).reshape(-1)[0])
    rate_feats = np.full((batch, 2), fill_value=[stereo_rate, imu_rate], dtype=np.float32)

    imu_np = np.asarray(imu_seq, dtype=np.float32)
    imu_mean = imu_np.mean(axis=0)
    imu_std = imu_np.std(axis=0)

    return np.concatenate([stereo_logits, imu_logits, disparity, rate_feats, imu_mean, imu_std], axis=-1)


def windows_to_sequences(
    extractor: SpyxStereoImuFeatureExtractor,
    windows: Iterable[StereoImuBatch],
    sequence_len: int,
    action_dim: int,
) -> list[FeatureSequenceExample]:
    """Aggregate extracted vectors into fixed-length sequences.

    This keeps assumptions minimal: each window contributes one latent/action point.
    """

    vectors: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    for window in windows:
        vec = extractor.extract(window)
        vectors.append(vec)
        if window.actions is None:
            actions.append(np.zeros((vec.shape[0], action_dim), dtype=np.float32))
        else:
            actions.append(np.asarray(window.actions, dtype=np.float32))

    if not vectors:
        return []

    feat_stream = np.concatenate(vectors, axis=0)
    act_stream = np.concatenate(actions, axis=0)

    examples: list[FeatureSequenceExample] = []
    for start in range(0, feat_stream.shape[0] - sequence_len + 1, sequence_len):
        end = start + sequence_len
        examples.append(
            FeatureSequenceExample(
                features=feat_stream[start:end].astype(np.float32),
                actions=act_stream[start:end].astype(np.float32),
            )
        )
    return examples


def write_feature_hdf5(examples: list[FeatureSequenceExample], output_path: str | Path) -> Path:
    """Persist feature/action sequences in a LeWM-friendly HDF5 layout."""

    if not examples:
        raise ValueError("No examples to write.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    features = np.stack([e.features for e in examples], axis=0)
    actions = np.stack([e.actions for e in examples], axis=0)

    with h5py.File(output, "w") as h5:
        h5.create_dataset("features", data=features, compression="gzip")
        h5.create_dataset("actions", data=actions, compression="gzip")

    return output


def mock_tonic_stereo_imu_windows(
    cfg: SnnFeatureConfig,
    num_windows: int,
    timesteps: int,
    batch: int,
    action_dim: int,
    seed: int = 0,
) -> list[StereoImuBatch]:
    """Synthetic stream for smoke-testing the pipeline before real Tonic wiring."""

    rng = np.random.default_rng(seed)
    h, w = cfg.input_hw
    channels = cfg.input_channels
    windows: list[StereoImuBatch] = []
    for _ in range(num_windows):
        left = rng.binomial(1, 0.04, size=(timesteps, batch, h, w, channels)).astype(np.float32)
        right = rng.binomial(1, 0.04, size=(timesteps, batch, h, w, channels)).astype(np.float32)
        imu = rng.normal(0.0, 1.0, size=(timesteps, batch, cfg.imu_dim)).astype(np.float32)
        act = rng.uniform(-1.0, 1.0, size=(batch, action_dim)).astype(np.float32)
        windows.append(StereoImuBatch(left_events=left, right_events=right, imu=imu, actions=act))
    return windows
