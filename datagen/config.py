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
    sensor_profile: str = "ideal"
    shutter_model: str = "global"
    pixel_size_um: float = 3.0
    full_well_e: float = 12000.0
    read_noise_e: float = 0.0
    dark_current_e_s: float = 0.0
    distortion: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    vignette_strength: float = 0.0
    lens_blur_sigma_px: float = 0.0
    exposure_ratio: float = 0.95


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
    photoreceptor_noise_std: float = 0.0
    leak_rate_hz: float = 0.5
    shot_noise_rate_hz: float = 0.5
    noise_rate_cov_decades: float = 0.2
    max_event_rate_hz: float = 0.0
    timestamp_jitter_us: float = 0.0


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
