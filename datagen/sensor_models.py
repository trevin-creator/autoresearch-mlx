"""Real-world sensor and lens profile presets for data generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorProfile:
    name: str
    description: str
    shutter_model: str
    native_width: int
    native_height: int
    nominal_fov_deg: float
    pixel_size_um: float
    full_well_e: float
    read_noise_e: float
    dark_current_e_s: float
    distortion: tuple[float, float, float, float, float]
    vignette_strength: float
    lens_blur_sigma_px: float


_PROFILES: dict[str, SensorProfile] = {
    "ideal": SensorProfile(
        name="ideal",
        description="Idealized pinhole camera without lens/sensor artifacts",
        shutter_model="global",
        native_width=346,
        native_height=260,
        nominal_fov_deg=90.0,
        pixel_size_um=3.0,
        full_well_e=12000.0,
        read_noise_e=0.0,
        dark_current_e_s=0.0,
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
        vignette_strength=0.0,
        lens_blur_sigma_px=0.0,
    ),
    "ov9281": SensorProfile(
        name="ov9281",
        description=(
            "OV9281-like 1MP global-shutter profile with practical lens artifacts"
        ),
        shutter_model="global",
        native_width=1280,
        native_height=800,
        nominal_fov_deg=80.0,
        pixel_size_um=3.0,
        full_well_e=10500.0,
        read_noise_e=2.2,
        dark_current_e_s=6.0,
        distortion=(-0.17, 0.045, 0.0005, -0.0005, -0.007),
        vignette_strength=0.38,
        lens_blur_sigma_px=0.75,
    ),
}


def available_sensor_profiles() -> list[str]:
    return sorted(_PROFILES.keys())


def get_sensor_profile(name: str) -> SensorProfile:
    profile = _PROFILES.get(name.lower())
    if profile is None:
        valid = ", ".join(available_sensor_profiles())
        msg = f"Unknown sensor profile '{name}'. Valid values: {valid}"
        raise ValueError(msg)
    return profile
