"""Scripted stereo rig trajectory for synthetic data generation."""

from __future__ import annotations

import math

import numpy as np

from .math_utils import lookat_rotation, rotmat_to_rvec


class ScriptedRigTrajectory:
    """Forward flight with multi-axis oscillations and mild banking.

    Produces deterministic pose + kinematics at any time t via closed-form
    expressions (no ODE integration needed).
    """

    def __init__(
        self,
        speed_mps: float = 2.0,
        altitude: float = 1.2,
        lateral_amplitude: float = 0.3,
        lateral_freq: float = 0.2,
        vertical_amplitude: float = 0.15,
        vertical_freq: float = 0.17,
        roll_amplitude_deg: float = 6.0,
        roll_freq: float = 0.23,
        lateral_clearance: float = 0.75,
        vertical_clearance: float = 0.40,
        obstacle_influence_sigma: float = 1.1,
    ):
        self.speed = speed_mps
        self.alt = altitude
        self.lat_amp = lateral_amplitude
        self.lat_freq = lateral_freq
        self.vert_amp = vertical_amplitude
        self.vert_freq = vertical_freq
        self.roll_amp = math.radians(roll_amplitude_deg)
        self.roll_freq = roll_freq
        self.lat_clearance = lateral_clearance
        self.vert_clearance = vertical_clearance
        self.obs_sigma = obstacle_influence_sigma

        # (x, y, radius-ish extent, top_z, preferred_side)
        self.obstacles = [
            (5.0, 0.0, 0.9, 1.0, 1.0),
            (8.0, 1.2, 1.2, 1.5, -1.0),
            (12.0, -1.0, 1.5, 2.0, 1.0),
            (15.0, 0.8, 0.8, 1.5, -1.0),
            (18.0, -0.5, 0.9, 1.8, 1.0),
        ]

    @staticmethod
    def _rotate_about_axis(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rodrigues rotation for vector ``v`` around ``axis`` by ``angle``."""
        axis_n = axis / (np.linalg.norm(axis) + 1e-12)
        ca = math.cos(angle)
        sa = math.sin(angle)
        return v * ca + np.cross(axis_n, v) * sa + axis_n * np.dot(axis_n, v) * (1 - ca)

    def _position(self, t: float) -> np.ndarray:
        """Obstacle-aware rig centre position at time t."""
        omega_lat = 2.0 * math.pi * self.lat_freq
        omega_vert = 2.0 * math.pi * self.vert_freq

        # Nominal forward motion with smooth weave and vertical bobbing.
        px = self.speed * t + 0.35 * math.sin(2.0 * math.pi * 0.11 * t)
        py = self.lat_amp * math.sin(omega_lat * t) + 0.12 * math.sin(
            2.0 * omega_lat * t + 0.7
        )
        pz = self.alt + self.vert_amp * math.sin(omega_vert * t + 0.35)

        # Obstacle-aware slalom and climb profile.
        for ox, oy, extent, top_z, side in self.obstacles:
            dx = px - ox
            influence = math.exp(-0.5 * (dx / self.obs_sigma) ** 2)

            target_y = oy + side * (extent + self.lat_clearance)
            py += 0.85 * influence * (target_y - py)

            min_z = top_z + self.vert_clearance
            if pz < min_z:
                pz += 0.90 * influence * (min_z - pz)

        return np.array([px, py, pz], dtype=np.float64)

    def pose(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (position, rotation_matrix) of the rig centre at time t."""
        pos = self._position(t)
        pos_next = self._position(t + 1e-3)
        forward = pos_next - pos
        forward = forward / (np.linalg.norm(forward) + 1e-12)

        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, up_world)
        if np.linalg.norm(right) < 1e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right = right / np.linalg.norm(right)

        roll = self.roll_amp * math.sin(2.0 * math.pi * self.roll_freq * t)
        up = self._rotate_about_axis(up_world, forward, roll)
        up = up / (np.linalg.norm(up) + 1e-12)

        target = pos + forward
        R = lookat_rotation(pos, target, up)
        return pos, R

    def kinematics(self, t: float, dt: float = 1e-4) -> dict[str, np.ndarray]:
        """Finite-difference velocity, acceleration, and body-frame angular velocity."""
        p0, _R0 = self.pose(t - dt)
        p1, R1 = self.pose(t)
        p2, R2 = self.pose(t + dt)

        vel_w = (p2 - p0) / (2.0 * dt)
        acc_w = (p2 - 2.0 * p1 + p0) / (dt * dt)

        dR = R1.T @ R2
        omega_body = rotmat_to_rvec(dR) / dt

        return {
            "vel_w": vel_w,
            "acc_w": acc_w,
            "omega_body": omega_body,
            "R_wc": R1,
        }
