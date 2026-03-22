"""Configuration dataclasses for stereo event camera data generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    width: int = 346
    height: int = 260
    fov_deg: float = 90.0
    near: float = 0.01
    far: float = 100.0


@dataclass(frozen=True)
class StereoConfig:
    baseline_m: float = 0.10
    rig_up_world: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class EventConfig:
    c_pos: float = 0.18
    c_neg: float = 0.18
    threshold_sigma: float = 0.02
    refractory_us: int = 200
    eps: float = 1e-3
    use_timestamp_interpolation: bool = True
    photoreceptor_tau_ms: float = 5.0
    leak_rate_hz: float = 0.5
    shot_noise_rate_hz: float = 0.5


@dataclass(frozen=True)
class SimConfig:
    sim_dt: float = 0.001
    duration_s: float = 10.0
    export_every_n_depth: int = 1
    seed: int = 1234
    backend: str = "gpu"
    show_viewer: bool = False


@dataclass(frozen=True)
class OutputConfig:
    out_dir: str = "./out_genesis_stereo_events"
    save_rgb_preview: bool = False
