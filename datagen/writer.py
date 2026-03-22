"""Dataset writer — persists events, depth, poses, IMU, and calibration to disk."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .config import CameraConfig, SimConfig, StereoConfig


class DatasetWriter:
    """Writes stereo event dataset to a structured directory layout.

    Layout::

        <out_dir>/
          events_left/000000.npz, ...
          events_right/000000.npz, ...
                    depth_left/000000.npy, ...
                    depth_right/000000.npy, ...
                    depth_gt/000000.npy, ...
                    disparity_gt/000000.npy, ...
          rgb_left_preview/000000.npy, ...   (optional)
          rgb_right_preview/000000.npy, ...  (optional)
          meta/
            calibration.json
            poses.csv
            imu.csv
            frames.csv
    """

    def __init__(self, out_dir: str):
        self.root = Path(out_dir)

        self.events_left_dir = self.root / "events_left"
        self.events_right_dir = self.root / "events_right"
        self.depth_left_dir = self.root / "depth_left"
        self.depth_right_dir = self.root / "depth_right"
        self.depth_gt_dir = self.root / "depth_gt"
        self.disparity_gt_dir = self.root / "disparity_gt"
        self.rgb_left_dir = self.root / "rgb_left_preview"
        self.rgb_right_dir = self.root / "rgb_right_preview"
        self.meta_dir = self.root / "meta"

        for d in [
            self.events_left_dir,
            self.events_right_dir,
            self.depth_left_dir,
            self.depth_right_dir,
            self.depth_gt_dir,
            self.disparity_gt_dir,
            self.rgb_left_dir,
            self.rgb_right_dir,
            self.meta_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        self._pose_fp = open(
            self.meta_dir / "poses.csv", "w", newline="", encoding="utf-8"
        )
        self._imu_fp = open(
            self.meta_dir / "imu.csv", "w", newline="", encoding="utf-8"
        )
        self._frame_fp = open(
            self.meta_dir / "frames.csv", "w", newline="", encoding="utf-8"
        )

        self.pose_writer = csv.writer(self._pose_fp)
        self.imu_writer = csv.writer(self._imu_fp)
        self.frame_writer = csv.writer(self._frame_fp)

        self.pose_writer.writerow(
            [
                "frame_idx",
                "t_us",
                "rig_px",
                "rig_py",
                "rig_pz",
                "rig_qx",
                "rig_qy",
                "rig_qz",
                "rig_qw",
                "left_px",
                "left_py",
                "left_pz",
                "left_qx",
                "left_qy",
                "left_qz",
                "left_qw",
                "right_px",
                "right_py",
                "right_pz",
                "right_qx",
                "right_qy",
                "right_qz",
                "right_qw",
            ]
        )
        self.imu_writer.writerow(
            [
                "frame_idx",
                "t_us",
                "acc_bx",
                "acc_by",
                "acc_bz",
                "gyro_bx",
                "gyro_by",
                "gyro_bz",
            ]
        )
        self.frame_writer.writerow(
            [
                "frame_idx",
                "t_us",
                "events_left_path",
                "events_right_path",
                "depth_left_path",
                "depth_right_path",
                "depth_gt_path",
                "disparity_gt_path",
            ]
        )

    def close(self) -> None:
        self._pose_fp.close()
        self._imu_fp.close()
        self._frame_fp.close()

    # ------------------------------------------------------------------
    # Per-frame writes
    # ------------------------------------------------------------------

    def write_events(self, frame_idx: int, side: str, events: np.ndarray) -> str:
        d = self.events_left_dir if side == "left" else self.events_right_dir
        out = d / f"{frame_idx:06d}.npz"
        np.savez_compressed(out, events=events)
        return str(out.relative_to(self.root))

    def write_depth(self, frame_idx: int, side: str, depth: np.ndarray) -> str:
        d = self.depth_left_dir if side == "left" else self.depth_right_dir
        out = d / f"{frame_idx:06d}.npy"
        np.save(out, depth)
        return str(out.relative_to(self.root))

    def write_depth_gt(self, frame_idx: int, depth: np.ndarray) -> str:
        out = self.depth_gt_dir / f"{frame_idx:06d}.npy"
        np.save(out, depth)
        return str(out.relative_to(self.root))

    def write_disparity_gt(self, frame_idx: int, disparity: np.ndarray) -> str:
        out = self.disparity_gt_dir / f"{frame_idx:06d}.npy"
        np.save(out, disparity)
        return str(out.relative_to(self.root))

    def write_rgb_preview(self, frame_idx: int, side: str, rgb: np.ndarray) -> str:
        d = self.rgb_left_dir if side == "left" else self.rgb_right_dir
        out = d / f"{frame_idx:06d}.npy"
        np.save(out, rgb)
        return str(out.relative_to(self.root))

    def write_pose_row(
        self,
        frame_idx: int,
        t_us: int,
        rig_pos: np.ndarray,
        rig_quat: np.ndarray,
        left_pos: np.ndarray,
        left_quat: np.ndarray,
        right_pos: np.ndarray,
        right_quat: np.ndarray,
    ) -> None:
        self.pose_writer.writerow(
            [
                frame_idx,
                t_us,
                *rig_pos.tolist(),
                *rig_quat.tolist(),
                *left_pos.tolist(),
                *left_quat.tolist(),
                *right_pos.tolist(),
                *right_quat.tolist(),
            ]
        )

    def write_imu_row(
        self,
        frame_idx: int,
        t_us: int,
        acc_b: np.ndarray,
        gyro_b: np.ndarray,
    ) -> None:
        self.imu_writer.writerow([frame_idx, t_us, *acc_b.tolist(), *gyro_b.tolist()])

    def write_frame_row(
        self,
        frame_idx: int,
        t_us: int,
        events_left_path: str,
        events_right_path: str,
        depth_left_path: str,
        depth_right_path: str,
        depth_gt_path: str,
        disparity_gt_path: str,
    ) -> None:
        self.frame_writer.writerow(
            [
                frame_idx,
                t_us,
                events_left_path,
                events_right_path,
                depth_left_path,
                depth_right_path,
                depth_gt_path,
                disparity_gt_path,
            ]
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def write_calibration(
        self,
        cam_cfg: CameraConfig,
        stereo_cfg: StereoConfig,
        sim_cfg: SimConfig,
    ) -> None:
        fx = fy = 0.5 * cam_cfg.width / math.tan(math.radians(cam_cfg.fov_deg) / 2.0)
        cx = (cam_cfg.width - 1) / 2.0
        cy = (cam_cfg.height - 1) / 2.0

        calib = {
            "camera_model": "pinhole",
            "resolution": {"width": cam_cfg.width, "height": cam_cfg.height},
            "intrinsics_left": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "distortion": [0.0, 0.0, 0.0, 0.0],
            },
            "intrinsics_right": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "distortion": [0.0, 0.0, 0.0, 0.0],
            },
            "stereo": {
                "baseline_m": stereo_cfg.baseline_m,
                "T_right_from_left": [
                    [1.0, 0.0, 0.0, stereo_cfg.baseline_m],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            "depth_ground_truth": {
                "camera": "rig_center",
                "description": "Single depth map from drone center camera",
            },
            "disparity_ground_truth": {
                "description": "Dense disparity from left-depth projection",
                "formula": "disparity_px = fx * baseline_m / depth_left_m",
            },
            "timing": {
                "sim_dt_s": sim_cfg.sim_dt,
                "time_unit": "microseconds",
            },
        }

        with open(self.meta_dir / "calibration.json", "w", encoding="utf-8") as f:
            json.dump(calib, f, indent=2)
