"""Generate stereo event camera data from a Genesis 3D scene.

Usage:
    uv run generate_stereo_events.py
    uv run generate_stereo_events.py --duration 30 --dt 0.0005 --out-dir ./my_data
"""

from __future__ import annotations

import argparse

from datagen.config import (
    CameraConfig,
    EventConfig,
    OutputConfig,
    SimConfig,
    StereoConfig,
)
from datagen.scene import GenesisStereoEventDataset
from datagen.sensor_models import available_sensor_profiles, get_sensor_profile


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate stereo event camera data via Genesis + ESIM emulator.",
    )

    # Camera
    p.add_argument("--width", type=int, default=346)
    p.add_argument("--height", type=int, default=260)
    p.add_argument("--fov", type=float, default=90.0, help="Horizontal FOV in degrees")
    p.add_argument(
        "--sensor-profile",
        type=str,
        default="ideal",
        choices=available_sensor_profiles(),
        help="Sensor/lens realism preset (includes OV9281 global-shutter profile)",
    )
    p.add_argument(
        "--keep-camera-params",
        action="store_true",
        help="Do not override width/height/FOV with selected sensor profile",
    )
    p.add_argument(
        "--vignette-strength",
        type=float,
        default=None,
        help="Override vignette falloff strength (None uses profile default)",
    )
    p.add_argument(
        "--lens-blur-sigma-px",
        type=float,
        default=None,
        help="Override lens PSF blur sigma in pixels (None uses profile default)",
    )

    # Stereo
    p.add_argument(
        "--baseline", type=float, default=0.10, help="Stereo baseline in metres"
    )

    # Event model
    p.add_argument(
        "--c-pos", type=float, default=0.18, help="Positive contrast threshold"
    )
    p.add_argument(
        "--c-neg", type=float, default=0.18, help="Negative contrast threshold"
    )
    p.add_argument(
        "--threshold-sigma", type=float, default=0.02, help="Per-pixel threshold noise"
    )
    p.add_argument(
        "--refractory-us",
        type=int,
        default=200,
        help="Refractory period in microseconds",
    )
    p.add_argument(
        "--photoreceptor-tau-ms",
        type=float,
        default=5.0,
        help="Photoreceptor low-pass time constant in milliseconds",
    )
    p.add_argument(
        "--photoreceptor-noise-std",
        type=float,
        default=0.0,
        help="Std-dev of additive temporal noise in log-intensity domain",
    )
    p.add_argument(
        "--leak-rate-hz",
        type=float,
        default=0.5,
        help="Per-pixel leak/background activity rate in Hz",
    )
    p.add_argument(
        "--shot-noise-rate-hz",
        type=float,
        default=0.5,
        help="Per-pixel shot-noise activity rate in Hz",
    )
    p.add_argument(
        "--noise-rate-cov-decades",
        type=float,
        default=0.2,
        help="Log-normal per-pixel FPN spread for leak/shot rates (decades)",
    )
    p.add_argument(
        "--max-event-rate-hz",
        type=float,
        default=0.0,
        help="Per-pixel bandwidth cap in Hz (0 disables cap)",
    )
    p.add_argument(
        "--timestamp-jitter-us",
        type=float,
        default=0.0,
        help="Gaussian timestamp jitter std-dev in microseconds",
    )

    # Simulation
    p.add_argument(
        "--dt", type=float, default=0.001, help="Simulation micro-step in seconds"
    )
    p.add_argument(
        "--duration", type=float, default=10.0, help="Total duration in seconds"
    )
    p.add_argument(
        "--depth-every", type=int, default=1, help="Save depth every N steps"
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--viewer", action="store_true", help="Show Genesis viewer")

    # Output
    p.add_argument("--out-dir", type=str, default="./out_genesis_stereo_events")
    p.add_argument("--save-rgb", action="store_true", help="Save RGB preview frames")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    profile = get_sensor_profile(args.sensor_profile)
    if args.keep_camera_params:
        cam_width = args.width
        cam_height = args.height
        cam_fov = args.fov
    else:
        cam_width = profile.native_width
        cam_height = profile.native_height
        cam_fov = profile.nominal_fov_deg

    cam_cfg = CameraConfig(
        width=cam_width,
        height=cam_height,
        fov_deg=cam_fov,
        sensor_profile=profile.name,
        shutter_model=profile.shutter_model,
        pixel_size_um=profile.pixel_size_um,
        full_well_e=profile.full_well_e,
        read_noise_e=profile.read_noise_e,
        dark_current_e_s=profile.dark_current_e_s,
        distortion=profile.distortion,
        vignette_strength=(
            profile.vignette_strength
            if args.vignette_strength is None
            else float(args.vignette_strength)
        ),
        lens_blur_sigma_px=(
            profile.lens_blur_sigma_px
            if args.lens_blur_sigma_px is None
            else float(args.lens_blur_sigma_px)
        ),
    )
    stereo_cfg = StereoConfig(baseline_m=args.baseline)
    event_cfg = EventConfig(
        c_pos=args.c_pos,
        c_neg=args.c_neg,
        threshold_sigma=args.threshold_sigma,
        refractory_us=args.refractory_us,
        photoreceptor_tau_ms=args.photoreceptor_tau_ms,
        photoreceptor_noise_std=args.photoreceptor_noise_std,
        leak_rate_hz=args.leak_rate_hz,
        shot_noise_rate_hz=args.shot_noise_rate_hz,
        noise_rate_cov_decades=args.noise_rate_cov_decades,
        max_event_rate_hz=args.max_event_rate_hz,
        timestamp_jitter_us=args.timestamp_jitter_us,
    )
    sim_cfg = SimConfig(
        sim_dt=args.dt,
        duration_s=args.duration,
        export_every_n_depth=args.depth_every,
        seed=args.seed,
        backend=args.backend,
        show_viewer=args.viewer,
    )
    out_cfg = OutputConfig(out_dir=args.out_dir, save_rgb_preview=args.save_rgb)

    ds = GenesisStereoEventDataset(
        cam_cfg=cam_cfg,
        stereo_cfg=stereo_cfg,
        event_cfg=event_cfg,
        sim_cfg=sim_cfg,
        out_cfg=out_cfg,
    )
    ds.run()


if __name__ == "__main__":
    main()
