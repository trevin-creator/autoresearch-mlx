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


def _add_camera_args(p: argparse.ArgumentParser) -> None:
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


def _add_stereo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--baseline", type=float, default=0.10, help="Stereo baseline in metres"
    )
    p.add_argument("--left-offset-x", type=float, default=0.0)
    p.add_argument("--left-offset-y", type=float, default=0.0)
    p.add_argument("--left-offset-z", type=float, default=0.0)
    p.add_argument("--right-offset-x", type=float, default=0.0)
    p.add_argument("--right-offset-y", type=float, default=0.0)
    p.add_argument("--right-offset-z", type=float, default=0.0)
    p.add_argument("--left-roll-deg", type=float, default=0.0)
    p.add_argument("--left-pitch-deg", type=float, default=0.0)
    p.add_argument("--left-yaw-deg", type=float, default=0.0)
    p.add_argument("--right-roll-deg", type=float, default=0.0)
    p.add_argument("--right-pitch-deg", type=float, default=0.0)
    p.add_argument("--right-yaw-deg", type=float, default=0.0)
    p.add_argument(
        "--right-fov-delta-deg",
        type=float,
        default=0.0,
        help="Extra FOV applied only to right camera",
    )
    p.add_argument(
        "--right-read-noise-scale",
        type=float,
        default=1.0,
        help="Scale factor for right camera read noise",
    )
    p.add_argument(
        "--right-dark-current-scale",
        type=float,
        default=1.0,
        help="Scale factor for right camera dark current",
    )
    p.add_argument(
        "--right-vignette-scale",
        type=float,
        default=1.0,
        help="Scale factor for right camera vignette strength",
    )
    p.add_argument(
        "--right-blur-delta-px",
        type=float,
        default=0.0,
        help="Additive blur sigma offset for right camera",
    )
    p.add_argument(
        "--right-distortion-scale",
        type=float,
        default=1.0,
        help="Scale all right-camera distortion coefficients",
    )


def _add_event_args(p: argparse.ArgumentParser) -> None:
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


def _add_simulation_args(p: argparse.ArgumentParser) -> None:
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
    p.add_argument(
        "--rotor-base-hz",
        type=float,
        default=90.0,
        help="Nominal rotor speed for vibration model",
    )
    p.add_argument(
        "--rotor-throttle-gain",
        type=float,
        default=0.15,
        help="Rotor speed sensitivity to maneuver intensity",
    )
    p.add_argument(
        "--rotor-imbalance",
        type=float,
        default=0.0,
        help="Rotor imbalance fraction in [0, 0.5]",
    )
    p.add_argument(
        "--vibration-trans-amp-mm",
        type=float,
        default=0.0,
        help="Camera-rig vibration translation amplitude in millimeters",
    )
    p.add_argument(
        "--vibration-rot-amp-deg",
        type=float,
        default=0.0,
        help="Camera-rig vibration angular amplitude in degrees",
    )


def _add_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out-dir", type=str, default="./out_genesis_stereo_events")
    p.add_argument("--save-rgb", action="store_true", help="Save RGB preview frames")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate stereo event camera data via Genesis + ESIM emulator.",
    )
    _add_camera_args(p)
    _add_stereo_args(p)
    _add_event_args(p)
    _add_simulation_args(p)
    _add_output_args(p)

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
    stereo_cfg = StereoConfig(
        baseline_m=args.baseline,
        left_mount_offset_m=(
            args.left_offset_x,
            args.left_offset_y,
            args.left_offset_z,
        ),
        right_mount_offset_m=(
            args.right_offset_x,
            args.right_offset_y,
            args.right_offset_z,
        ),
        left_mount_rpy_deg=(args.left_roll_deg, args.left_pitch_deg, args.left_yaw_deg),
        right_mount_rpy_deg=(
            args.right_roll_deg,
            args.right_pitch_deg,
            args.right_yaw_deg,
        ),
        right_fov_delta_deg=args.right_fov_delta_deg,
        right_read_noise_scale=args.right_read_noise_scale,
        right_dark_current_scale=args.right_dark_current_scale,
        right_vignette_scale=args.right_vignette_scale,
        right_blur_delta_px=args.right_blur_delta_px,
        right_distortion_scale=args.right_distortion_scale,
    )
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
        rotor_base_hz=args.rotor_base_hz,
        rotor_throttle_gain=args.rotor_throttle_gain,
        rotor_imbalance=args.rotor_imbalance,
        vibration_trans_amp_m=args.vibration_trans_amp_mm * 1e-3,
        vibration_rot_amp_deg=args.vibration_rot_amp_deg,
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
