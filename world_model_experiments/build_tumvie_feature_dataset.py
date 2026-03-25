from __future__ import annotations

import argparse
import logging
from pathlib import Path

from world_model_experiments.snn_feature_pipeline import SnnFeatureConfig
from world_model_experiments.tumvie_local import TumvieWindowConfig, build_tumvie_feature_dataset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a real TUMVIE stereo-event+IMU feature dataset")
    parser.add_argument(
        "--recording-dir",
        type=str,
        default="spyx/research/data/TUMVIE/mocap-6dof",
        help="Path to the local TUMVIE recording directory",
    )
    parser.add_argument("--output", type=str, default="artifacts/tumvie/tumvie_features.h5")
    parser.add_argument("--sample-t", type=int, default=8)
    parser.add_argument("--window-us", type=int, default=50000)
    parser.add_argument("--stride-us", type=int, default=50000)
    parser.add_argument("--downsample-factor", type=float, default=0.1)
    parser.add_argument("--max-windows", type=int, default=64)
    parser.add_argument("--sequence-len", type=int, default=8)
    parser.add_argument("--flight-plan-horizon", type=int, default=3)
    parser.add_argument("--stereo-channels", type=int, default=12)
    parser.add_argument("--visual-channels1", type=int, default=12)
    parser.add_argument("--visual-channels2", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_h = max(1, int(round(720 * args.downsample_factor)))
    input_w = max(1, int(round(1280 * args.downsample_factor)))

    tumvie_cfg = TumvieWindowConfig(
        recording_dir=Path(args.recording_dir),
        sample_t=args.sample_t,
        window_us=args.window_us,
        stride_us=args.stride_us,
        downsample_factor=args.downsample_factor,
        max_windows=args.max_windows,
        flight_plan_horizon=args.flight_plan_horizon,
    )
    snn_cfg = SnnFeatureConfig(
        input_hw=(input_h, input_w),
        input_channels=2,
        imu_dim=6,
        stereo_channels=args.stereo_channels,
        visual_channels1=args.visual_channels1,
        visual_channels2=args.visual_channels2,
        feature_dim=args.feature_dim,
    )

    out = build_tumvie_feature_dataset(
        tumvie_cfg=tumvie_cfg,
        snn_cfg=snn_cfg,
        sequence_len=args.sequence_len,
        output_path=args.output,
    )
    logger.info("wrote TUMVIE feature dataset: %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()