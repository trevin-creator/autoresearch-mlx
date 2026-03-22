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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate stereo event camera data via Genesis + ESIM emulator.",
    )

    # Camera
    p.add_argument("--width", type=int, default=346)
    p.add_argument("--height", type=int, default=260)
    p.add_argument("--fov", type=float, default=90.0, help="Horizontal FOV in degrees")

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

    cam_cfg = CameraConfig(width=args.width, height=args.height, fov_deg=args.fov)
    stereo_cfg = StereoConfig(baseline_m=args.baseline)
    event_cfg = EventConfig(
        c_pos=args.c_pos,
        c_neg=args.c_neg,
        threshold_sigma=args.threshold_sigma,
        refractory_us=args.refractory_us,
        photoreceptor_tau_ms=args.photoreceptor_tau_ms,
        leak_rate_hz=args.leak_rate_hz,
        shot_noise_rate_hz=args.shot_noise_rate_hz,
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
