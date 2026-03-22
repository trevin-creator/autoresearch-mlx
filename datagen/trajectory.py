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
    ):
        self.speed = speed_mps
        self.alt = altitude
        self.lat_amp = lateral_amplitude
        self.lat_freq = lateral_freq
        self.vert_amp = vertical_amplitude
        self.vert_freq = vertical_freq
        self.roll_amp = math.radians(roll_amplitude_deg)
        self.roll_freq = roll_freq

    @staticmethod
    def _rotate_about_axis(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rodrigues rotation for vector ``v`` around ``axis`` by ``angle``."""
        axis_n = axis / (np.linalg.norm(axis) + 1e-12)
        ca = math.cos(angle)
        sa = math.sin(angle)
        return v * ca + np.cross(axis_n, v) * sa + axis_n * np.dot(axis_n, v) * (1 - ca)

    def pose(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (position, rotation_matrix) of the rig centre at time t."""
        omega_lat = 2.0 * math.pi * self.lat_freq
        omega_vert = 2.0 * math.pi * self.vert_freq

        # Slight speed wobble + dual-frequency lateral weave + vertical undulation.
        px = self.speed * t + 0.35 * math.sin(2.0 * math.pi * 0.11 * t)
        py = self.lat_amp * math.sin(omega_lat * t) + 0.12 * math.sin(
            2.0 * omega_lat * t + 0.7
        )
        pz = self.alt + self.vert_amp * math.sin(omega_vert * t + 0.35)
        pos = np.array([px, py, pz], dtype=np.float64)

        vx = self.speed + 0.35 * 2.0 * math.pi * 0.11 * math.cos(
            2.0 * math.pi * 0.11 * t
        )
        vy = self.lat_amp * omega_lat * math.cos(
            omega_lat * t
        ) + 0.24 * omega_lat * math.cos(2.0 * omega_lat * t + 0.7)
        vz = self.vert_amp * omega_vert * math.cos(omega_vert * t + 0.35)

        forward = np.array([vx, vy, vz], dtype=np.float64)
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
