"""Genesis-based stereo event dataset generator.

Builds a Genesis 3D scene with a stereo camera rig, steps the simulation
at high temporal rate, and feeds rendered frames through the ESIM event
emulator to produce a complete stereo event dataset on disk.

Requires the ``genesis`` package (``pip install genesis-world``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import CameraConfig, EventConfig, OutputConfig, SimConfig, StereoConfig
from .emulator import EventCameraEmulator
from .math_utils import lookat_rotation, normalize, quat_from_rotmat
from .trajectory import ScriptedRigTrajectory
from .writer import DatasetWriter

try:
    import genesis as gs  # type: ignore[import-untyped]
except ImportError:
    gs = None


class GenesisStereoEventDataset:
    """End-to-end stereo event data generation using Genesis."""

    def __init__(
        self,
        cam_cfg: CameraConfig,
        stereo_cfg: StereoConfig,
        event_cfg: EventConfig,
        sim_cfg: SimConfig,
        out_cfg: OutputConfig,
    ):
        self.cam_cfg = cam_cfg
        self.stereo_cfg = stereo_cfg
        self.event_cfg = event_cfg
        self.sim_cfg = sim_cfg
        self.out_cfg = out_cfg

        self.rng = np.random.default_rng(sim_cfg.seed)
        self.writer = DatasetWriter(out_cfg.out_dir)
        self.trajectory = ScriptedRigTrajectory()

        self.scene: Any = None
        self.left_cam: Any = None
        self.right_cam: Any = None

        self.left_emu = EventCameraEmulator(
            height=cam_cfg.height,
            width=cam_cfg.width,
            cfg=event_cfg,
            rng=self.rng,
        )
        self.right_emu = EventCameraEmulator(
            height=cam_cfg.height,
            width=cam_cfg.width,
            cfg=event_cfg,
            rng=self.rng,
        )

    def build_scene(self) -> None:
        if gs is None:
            msg = "Genesis is required. Install with: pip install genesis-world"
            raise ImportError(msg)

        backend = gs.gpu if self.sim_cfg.backend == "gpu" else gs.cpu
        gs.init(backend=backend)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.sim_cfg.sim_dt),
            show_viewer=self.sim_cfg.show_viewer,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.0, -4.0, 2.0),
                camera_lookat=(2.0, 0.0, 1.0),
                camera_fov=40,
            ),
        )

        # Placeholder geometry — replace with real Genesis world assets.
        self.scene.add_entity(gs.morphs.Plane())
        self.scene.add_entity(gs.morphs.Box(pos=(5.0, 0.0, 0.5), size=(1.0, 1.0, 1.0)))
        self.scene.add_entity(gs.morphs.Box(pos=(8.0, 1.2, 0.75), size=(1.5, 1.0, 1.5)))
        self.scene.add_entity(
            gs.morphs.Box(pos=(12.0, -1.0, 1.0), size=(2.0, 1.5, 2.0))
        )
        self.scene.add_entity(
            gs.morphs.Cylinder(pos=(15.0, 0.8, 0.75), radius=0.5, height=1.5)
        )
        self.scene.add_entity(gs.morphs.Sphere(pos=(18.0, -0.5, 1.0), radius=0.8))

        half_b = self.stereo_cfg.baseline_m / 2.0
        self.left_cam = self.scene.add_camera(
            res=(self.cam_cfg.width, self.cam_cfg.height),
            pos=(0.0, -half_b, 1.2),
            lookat=(1.0, -half_b, 1.2),
            up=self.stereo_cfg.rig_up_world,
            fov=self.cam_cfg.fov_deg,
        )
        self.right_cam = self.scene.add_camera(
            res=(self.cam_cfg.width, self.cam_cfg.height),
            pos=(0.0, half_b, 1.2),
            lookat=(1.0, half_b, 1.2),
            up=self.stereo_cfg.rig_up_world,
            fov=self.cam_cfg.fov_deg,
        )

        self.scene.build()

    # ------------------------------------------------------------------
    # Stereo rig helpers
    # ------------------------------------------------------------------

    def stereo_poses_from_rig(
        self,
        rig_pos: np.ndarray,
        R_wc: np.ndarray,
    ) -> dict[str, np.ndarray]:
        right_axis = R_wc[:, 0]
        forward_axis = R_wc[:, 2]
        up_axis = normalize(np.cross(right_axis, forward_axis))

        half_b = 0.5 * self.stereo_cfg.baseline_m
        left_pos = rig_pos - half_b * right_axis
        right_pos = rig_pos + half_b * right_axis

        return {
            "left_pos": left_pos,
            "right_pos": right_pos,
            "left_lookat": left_pos + forward_axis,
            "right_lookat": right_pos + forward_axis,
            "up": up_axis,
        }

    def update_camera_poses(self, t: float) -> dict[str, np.ndarray]:
        rig_pos, R_wc = self.trajectory.pose(t)
        rig_quat = quat_from_rotmat(R_wc)

        poses = self.stereo_poses_from_rig(rig_pos, R_wc)
        left_R = lookat_rotation(poses["left_pos"], poses["left_lookat"], poses["up"])
        right_R = lookat_rotation(
            poses["right_pos"], poses["right_lookat"], poses["up"]
        )

        self.left_cam.set_pose(
            pos=tuple(poses["left_pos"].tolist()),
            lookat=tuple(poses["left_lookat"].tolist()),
            up=tuple(poses["up"].tolist()),
        )
        self.right_cam.set_pose(
            pos=tuple(poses["right_pos"].tolist()),
            lookat=tuple(poses["right_lookat"].tolist()),
            up=tuple(poses["up"].tolist()),
        )

        return {
            "rig_pos": rig_pos,
            "rig_quat": rig_quat,
            "left_pos": poses["left_pos"],
            "left_quat": quat_from_rotmat(left_R),
            "right_pos": poses["right_pos"],
            "right_quat": quat_from_rotmat(right_R),
            "R_wc": R_wc,
        }

    def ideal_imu_from_trajectory(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        kin = self.trajectory.kinematics(t)
        R_wc = kin["R_wc"]
        gravity_w = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        specific_force_w = kin["acc_w"] - gravity_w
        acc_b = R_wc.T @ specific_force_w
        return acc_b, kin["omega_body"]

    def render_pair(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rgb_left = np.asarray(self.left_cam.render(rgb=True))
        depth_left = np.asarray(self.left_cam.render(depth=True))
        rgb_right = np.asarray(self.right_cam.render(rgb=True))
        depth_right = np.asarray(self.right_cam.render(depth=True))
        return rgb_left, depth_left, rgb_right, depth_right

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.build_scene()
        self.writer.write_calibration(self.cam_cfg, self.stereo_cfg, self.sim_cfg)

        n_steps = int(round(self.sim_cfg.duration_s / self.sim_cfg.sim_dt))

        # Initialise at t=0
        self.update_camera_poses(0.0)
        rgb_l0, _, rgb_r0, _ = self.render_pair()
        self.left_emu.initialize(rgb_l0, t0_us=0)
        self.right_emu.initialize(rgb_r0, t0_us=0)

        for step_idx in range(n_steps):
            self.scene.step()
            t = (step_idx + 1) * self.sim_cfg.sim_dt
            t_us = int(round(t * 1e6))

            pose_info = self.update_camera_poses(t)
            rgb_left, depth_left, rgb_right, depth_right = self.render_pair()

            events_left = self.left_emu.step(rgb_left, t_us=t_us)
            events_right = self.right_emu.step(rgb_right, t_us=t_us)

            left_events_path = self.writer.write_events(step_idx, "left", events_left)
            right_events_path = self.writer.write_events(
                step_idx, "right", events_right
            )

            if step_idx % self.sim_cfg.export_every_n_depth == 0:
                left_depth_path = self.writer.write_depth(step_idx, "left", depth_left)
                right_depth_path = self.writer.write_depth(
                    step_idx, "right", depth_right
                )
            else:
                left_depth_path = ""
                right_depth_path = ""

            if self.out_cfg.save_rgb_preview:
                self.writer.write_rgb_preview(step_idx, "left", rgb_left)
                self.writer.write_rgb_preview(step_idx, "right", rgb_right)

            acc_b, gyro_b = self.ideal_imu_from_trajectory(t)

            self.writer.write_pose_row(
                frame_idx=step_idx,
                t_us=t_us,
                rig_pos=pose_info["rig_pos"],
                rig_quat=pose_info["rig_quat"],
                left_pos=pose_info["left_pos"],
                left_quat=pose_info["left_quat"],
                right_pos=pose_info["right_pos"],
                right_quat=pose_info["right_quat"],
            )
            self.writer.write_imu_row(
                frame_idx=step_idx, t_us=t_us, acc_b=acc_b, gyro_b=gyro_b
            )
            self.writer.write_frame_row(
                frame_idx=step_idx,
                t_us=t_us,
                events_left_path=left_events_path,
                events_right_path=right_events_path,
                depth_left_path=left_depth_path,
                depth_right_path=right_depth_path,
            )

            if step_idx % 100 == 0:
                print(
                    f"[{step_idx:06d}/{n_steps:06d}] t={t:.3f}s "
                    f"left_events={len(events_left)} right_events={len(events_right)}"
                )

        self.writer.close()
        print(f"Done. Dataset written to: {self.writer.root.resolve()}")
