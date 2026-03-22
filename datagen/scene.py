"""Genesis-based stereo event dataset generator.

Builds a Genesis 3D scene with a stereo camera rig, steps the simulation
at high temporal rate, and feeds rendered frames through the ESIM event
emulator to produce a complete stereo event dataset on disk.

Requires the ``genesis`` package (``pip install genesis-world``).
"""

from __future__ import annotations

from pathlib import Path
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
        self.center_cam: Any = None

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

    @staticmethod
    def _checker_texture(
        size: int,
        tile: int,
        color_a: tuple[int, int, int],
        color_b: tuple[int, int, int],
    ) -> np.ndarray:
        yy, xx = np.indices((size, size))
        pattern = ((xx // tile) + (yy // tile)) % 2
        tex = np.empty((size, size, 3), dtype=np.uint8)
        tex[pattern == 0] = color_a
        tex[pattern == 1] = color_b
        return tex

    @staticmethod
    def _stripe_texture(
        size: int,
        stripe: int,
        color_a: tuple[int, int, int],
        color_b: tuple[int, int, int],
    ) -> np.ndarray:
        _, xx = np.indices((size, size))
        pattern = (xx // stripe) % 2
        tex = np.empty((size, size, 3), dtype=np.uint8)
        tex[pattern == 0] = color_a
        tex[pattern == 1] = color_b
        return tex

    @staticmethod
    def _surface_from_texture(
        tex: np.ndarray,
        roughness: float = 0.75,
        metallic: float = 0.0,
    ) -> Any:
        _gs = gs
        if _gs is None:
            msg = "Genesis is required. Install with: pip install genesis-world"
            raise ImportError(msg)
        return _gs.surfaces.Rough(
            diffuse_texture=_gs.textures.ImageTexture(image_array=tex),
            roughness=roughness,
            metallic=metallic,
        )

    @staticmethod
    def _surface_from_color(
        color: tuple[float, float, float],
        roughness: float = 0.8,
        metallic: float = 0.0,
    ) -> Any:
        _gs = gs
        if _gs is None:
            msg = "Genesis is required. Install with: pip install genesis-world"
            raise ImportError(msg)
        return _gs.surfaces.Rough(color=color, roughness=roughness, metallic=metallic)

    def _noise_texture(
        self,
        size: int,
        base_color: tuple[int, int, int],
        noise_strength: int = 28,
    ) -> np.ndarray:
        base = np.full((size, size, 3), base_color, dtype=np.int16)
        noise = self.rng.integers(
            -noise_strength,
            noise_strength + 1,
            size=(size, size, 3),
            dtype=np.int16,
        )
        return np.clip(base + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def _blend_textures(
        tex_a: np.ndarray, tex_b: np.ndarray, alpha: float
    ) -> np.ndarray:
        a = tex_a.astype(np.float32)
        b = tex_b.astype(np.float32)
        out = np.clip((1.0 - alpha) * a + alpha * b, 0.0, 255.0)
        return out.astype(np.uint8)

    def _sample_floor_y(self) -> float:
        """Sample floor clutter mostly outside the flight corridor around y=0."""
        if self.rng.random() < 0.5:
            return float(self.rng.uniform(-4.0, -1.6))
        return float(self.rng.uniform(1.6, 4.0))

    def _add_floor_clutter(self) -> None:
        """Populate scene floor with grass-like tufts and random props."""
        _gs = gs
        if _gs is None:
            msg = "Genesis is required. Install with: pip install genesis-world"
            raise ImportError(msg)

        grass_surface = self._surface_from_color((0.28, 0.55, 0.25), roughness=0.95)
        rock_surface = self._surface_from_color((0.45, 0.42, 0.38), roughness=0.9)

        # Grass tufts: thin cylinders with slight height/radius jitter.
        for _ in range(140):
            x = float(self.rng.uniform(0.5, 22.0))
            y = self._sample_floor_y()
            h = float(self.rng.uniform(0.08, 0.24))
            r = float(self.rng.uniform(0.01, 0.03))
            self.scene.add_entity(
                _gs.morphs.Cylinder(
                    pos=(x, y, h * 0.5),
                    radius=r,
                    height=h,
                    fixed=True,
                ),
                surface=grass_surface,
            )

        # Low-frequency terrain clutter: rocks and small debris props.
        for _ in range(45):
            x = float(self.rng.uniform(0.5, 22.0))
            y = self._sample_floor_y()
            rad = float(self.rng.uniform(0.04, 0.2))
            self.scene.add_entity(
                _gs.morphs.Sphere(pos=(x, y, rad), radius=rad, fixed=True),
                surface=rock_surface,
            )

        for _ in range(30):
            x = float(self.rng.uniform(0.5, 22.0))
            y = self._sample_floor_y()
            sx = float(self.rng.uniform(0.08, 0.35))
            sy = float(self.rng.uniform(0.08, 0.35))
            sz = float(self.rng.uniform(0.05, 0.22))
            color = (
                float(self.rng.uniform(0.22, 0.55)),
                float(self.rng.uniform(0.20, 0.45)),
                float(self.rng.uniform(0.14, 0.32)),
            )
            self.scene.add_entity(
                _gs.morphs.Box(pos=(x, y, sz * 0.5), size=(sx, sy, sz), fixed=True),
                surface=self._surface_from_color(color, roughness=0.92),
            )

    def build_scene(self) -> None:
        if gs is None:
            msg = "Genesis is required. Install with: pip install genesis-world"
            raise ImportError(msg)

        backend = gs.metal if self.sim_cfg.backend == "gpu" else gs.cpu
        print(f"[datagen] Genesis backend={backend}")
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

        # Procedural textures keep the scene visually distinct for stereo demos.
        # Blend pattern + random noise to create richer, less synthetic surfaces.
        ground_checker = self._checker_texture(512, 24, (78, 84, 90), (52, 57, 61))
        ground_noise = self._noise_texture(512, (68, 74, 79), noise_strength=18)
        ground_tex = self._blend_textures(ground_checker, ground_noise, alpha=0.35)

        box1_checker = self._checker_texture(256, 14, (154, 108, 72), (118, 80, 52))
        box1_noise = self._noise_texture(256, (136, 96, 62), noise_strength=34)
        box1_tex = self._blend_textures(box1_checker, box1_noise, alpha=0.45)

        box2_checker = self._checker_texture(256, 10, (64, 136, 198), (32, 90, 148))
        box2_noise = self._noise_texture(256, (58, 118, 170), noise_strength=30)
        box2_tex = self._blend_textures(box2_checker, box2_noise, alpha=0.42)

        box3_checker = self._checker_texture(256, 8, (196, 132, 58), (122, 76, 38))
        box3_noise = self._noise_texture(256, (170, 112, 50), noise_strength=36)
        box3_tex = self._blend_textures(box3_checker, box3_noise, alpha=0.48)
        cube_mesh = (
            Path(__file__).resolve().parents[1] / "assets" / "meshes" / "uv_cube.obj"
        )

        self.scene.add_entity(
            gs.morphs.Plane(),
            surface=self._surface_from_texture(ground_tex, roughness=0.95),
        )
        self.scene.add_entity(
            gs.morphs.Mesh(
                file=str(cube_mesh),
                scale=(1.0, 1.0, 1.0),
                pos=(5.0, 0.0, 0.5),
                fixed=True,
            ),
            surface=self._surface_from_texture(box1_tex, roughness=0.85),
        )
        self.scene.add_entity(
            gs.morphs.Mesh(
                file=str(cube_mesh),
                scale=(1.5, 1.0, 1.5),
                pos=(8.0, 1.2, 0.75),
                fixed=True,
            ),
            surface=self._surface_from_texture(box2_tex, roughness=0.8),
        )
        self.scene.add_entity(
            gs.morphs.Mesh(
                file=str(cube_mesh),
                scale=(2.0, 1.5, 2.0),
                pos=(12.0, -1.0, 1.0),
                fixed=True,
            ),
            surface=self._surface_from_texture(box3_tex, roughness=0.82),
        )
        self.scene.add_entity(
            gs.morphs.Cylinder(pos=(15.0, 0.8, 0.75), radius=0.5, height=1.5),
            surface=self._surface_from_color((0.85, 0.72, 0.2), roughness=0.55),
        )
        self.scene.add_entity(
            gs.morphs.Sphere(pos=(18.0, -0.5, 1.0), radius=0.8),
            surface=self._surface_from_color((0.68, 0.22, 0.24), roughness=0.35),
        )

        self._add_floor_clutter()

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
        self.center_cam = self.scene.add_camera(
            res=(self.cam_cfg.width, self.cam_cfg.height),
            pos=(0.0, 0.0, 1.2),
            lookat=(1.0, 0.0, 1.2),
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
        rig_forward = R_wc[:, 2]
        self.center_cam.set_pose(
            pos=tuple(rig_pos.tolist()),
            lookat=tuple((rig_pos + rig_forward).tolist()),
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

    @staticmethod
    def _extract_rgb_depth(render_out: Any) -> tuple[np.ndarray, np.ndarray]:
        """Normalize Genesis camera render output across API variants.

        In genesis-world 0.4.x, ``render`` returns a 4-tuple:
        ``(rgb, depth, seg, normal)`` where disabled channels are ``None``.
        Older variants may return only an RGB frame.
        """
        if isinstance(render_out, tuple):
            rgb = render_out[0]
            depth = render_out[1] if len(render_out) > 1 else None
        else:
            rgb = render_out
            depth = None

        rgb_np = np.asarray(rgb)
        if depth is None:
            depth_np = np.zeros((rgb_np.shape[0], rgb_np.shape[1]), dtype=np.float32)
        else:
            depth_np = np.asarray(depth)
        return rgb_np, depth_np

    def render_pair(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        left_out = self.left_cam.render(rgb=True, depth=True)
        right_out = self.right_cam.render(rgb=True, depth=True)
        rgb_left, depth_left = self._extract_rgb_depth(left_out)
        rgb_right, depth_right = self._extract_rgb_depth(right_out)
        return rgb_left, depth_left, rgb_right, depth_right

    def render_depth_gt(self) -> np.ndarray:
        center_out = self.center_cam.render(depth=True)
        _, depth_gt = self._extract_rgb_depth(center_out)
        return depth_gt

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
            rgb_left, _depth_left, rgb_right, _depth_right = self.render_pair()
            depth_gt = self.render_depth_gt()

            events_left = self.left_emu.step(rgb_left, t_us=t_us)
            events_right = self.right_emu.step(rgb_right, t_us=t_us)

            left_events_path = self.writer.write_events(step_idx, "left", events_left)
            right_events_path = self.writer.write_events(
                step_idx, "right", events_right
            )

            if step_idx % self.sim_cfg.export_every_n_depth == 0:
                depth_gt_path = self.writer.write_depth_gt(step_idx, depth_gt)
            else:
                depth_gt_path = ""

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
                depth_gt_path=depth_gt_path,
            )

            if step_idx % 100 == 0:
                print(
                    f"[{step_idx:06d}/{n_steps:06d}] t={t:.3f}s "
                    f"left_events={len(events_left)} right_events={len(events_right)}"
                )

        self.writer.close()
        print(f"Done. Dataset written to: {self.writer.root.resolve()}")
