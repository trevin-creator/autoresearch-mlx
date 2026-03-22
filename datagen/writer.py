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
        left_cam_cfg: CameraConfig,
        right_cam_cfg: CameraConfig,
        stereo_cfg: StereoConfig,
        sim_cfg: SimConfig,
    ) -> None:
        fx_l = fy_l = (
            0.5
            * left_cam_cfg.width
            / math.tan(math.radians(left_cam_cfg.fov_deg) / 2.0)
        )
        cx_l = (left_cam_cfg.width - 1) / 2.0
        cy_l = (left_cam_cfg.height - 1) / 2.0

        fx_r = fy_r = (
            0.5
            * right_cam_cfg.width
            / math.tan(math.radians(right_cam_cfg.fov_deg) / 2.0)
        )
        cx_r = (right_cam_cfg.width - 1) / 2.0
        cy_r = (right_cam_cfg.height - 1) / 2.0

        calib = {
            "camera_model": "pinhole",
            "resolution": {
                "left": {"width": left_cam_cfg.width, "height": left_cam_cfg.height},
                "right": {
                    "width": right_cam_cfg.width,
                    "height": right_cam_cfg.height,
                },
            },
            "sensor": {
                "left": {
                    "profile": left_cam_cfg.sensor_profile,
                    "shutter_model": left_cam_cfg.shutter_model,
                    "pixel_size_um": left_cam_cfg.pixel_size_um,
                    "full_well_e": left_cam_cfg.full_well_e,
                    "read_noise_e": left_cam_cfg.read_noise_e,
                    "dark_current_e_s": left_cam_cfg.dark_current_e_s,
                    "exposure_ratio": left_cam_cfg.exposure_ratio,
                },
                "right": {
                    "profile": right_cam_cfg.sensor_profile,
                    "shutter_model": right_cam_cfg.shutter_model,
                    "pixel_size_um": right_cam_cfg.pixel_size_um,
                    "full_well_e": right_cam_cfg.full_well_e,
                    "read_noise_e": right_cam_cfg.read_noise_e,
                    "dark_current_e_s": right_cam_cfg.dark_current_e_s,
                    "exposure_ratio": right_cam_cfg.exposure_ratio,
                },
            },
            "lens": {
                "model": "radial_tangential",
                "left": {
                    "distortion": list(left_cam_cfg.distortion),
                    "vignette_strength": left_cam_cfg.vignette_strength,
                    "psf_blur_sigma_px": left_cam_cfg.lens_blur_sigma_px,
                },
                "right": {
                    "distortion": list(right_cam_cfg.distortion),
                    "vignette_strength": right_cam_cfg.vignette_strength,
                    "psf_blur_sigma_px": right_cam_cfg.lens_blur_sigma_px,
                },
            },
            "intrinsics_left": {
                "fx": fx_l,
                "fy": fy_l,
                "cx": cx_l,
                "cy": cy_l,
                "distortion": list(left_cam_cfg.distortion),
            },
            "intrinsics_right": {
                "fx": fx_r,
                "fy": fy_r,
                "cx": cx_r,
                "cy": cy_r,
                "distortion": list(right_cam_cfg.distortion),
            },
            "stereo": {
                "baseline_m": stereo_cfg.baseline_m,
                "left_mount_offset_m": list(stereo_cfg.left_mount_offset_m),
                "right_mount_offset_m": list(stereo_cfg.right_mount_offset_m),
                "left_mount_rpy_deg": list(stereo_cfg.left_mount_rpy_deg),
                "right_mount_rpy_deg": list(stereo_cfg.right_mount_rpy_deg),
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
            "vibration_model": {
                "rotor_base_hz": sim_cfg.rotor_base_hz,
                "rotor_throttle_gain": sim_cfg.rotor_throttle_gain,
                "rotor_imbalance": sim_cfg.rotor_imbalance,
                "translation_amplitude_m": sim_cfg.vibration_trans_amp_m,
                "rotation_amplitude_deg": sim_cfg.vibration_rot_amp_deg,
            },
        }

        with open(self.meta_dir / "calibration.json", "w", encoding="utf-8") as f:
            json.dump(calib, f, indent=2)
