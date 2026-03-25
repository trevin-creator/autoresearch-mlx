from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

from world_model_experiments._io import load_sequence_dataset, write_sequence_dataset
from world_model_experiments.data_catalog import (
    DEFAULT_CATALOG_ROOT,
    default_dataset_output_path,
    describe_arrays,
    register_dataset,
)
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
    parser.add_argument("--output", type=str, default=str(default_dataset_output_path("tumvie_features")))
    parser.add_argument("--dataset-name", type=str, default="tumvie_features")
    parser.add_argument("--catalog-root", type=str, default=str(DEFAULT_CATALOG_ROOT))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--source-uri", type=str, default=None)
    parser.add_argument("--parent-dataset-id", action="append", default=[])
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
    output_path = Path(args.output)
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

    if output_path.suffix.lower() == ".parquet":
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_h5 = Path(tmp_dir) / "tumvie_features.h5"
            build_tumvie_feature_dataset(
                tumvie_cfg=tumvie_cfg,
                snn_cfg=snn_cfg,
                sequence_len=args.sequence_len,
                output_path=temp_h5,
            )
            arrays = load_sequence_dataset(temp_h5)
            out = write_sequence_dataset(output_path, arrays)
    else:
        out = build_tumvie_feature_dataset(
            tumvie_cfg=tumvie_cfg,
            snn_cfg=snn_cfg,
            sequence_len=args.sequence_len,
            output_path=output_path,
        )
        arrays = load_sequence_dataset(out)

    record = register_dataset(
        dataset_path=out,
        dataset_name=args.dataset_name,
        transform_name="build_tumvie_feature_dataset",
        split=args.split,
        source_uri=args.source_uri or Path(args.recording_dir).resolve().as_uri(),
        parent_ids=args.parent_dataset_id,
        metadata={
            "downsample_factor": args.downsample_factor,
            "feature_dim": args.feature_dim,
            "flight_plan_horizon": args.flight_plan_horizon,
            "max_windows": args.max_windows,
            "recording_dir": str(Path(args.recording_dir).resolve()),
            "sample_t": args.sample_t,
            "sequence_len": args.sequence_len,
            "stride_us": args.stride_us,
            "window_us": args.window_us,
        },
        schema=describe_arrays(arrays),
        catalog_root=args.catalog_root,
    )
    logger.info("wrote TUMVIE feature dataset: %s", out)
    logger.info("registered dataset: %s", record.dataset_id)
    logger.info("dataset manifest: %s", record.manifest_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
