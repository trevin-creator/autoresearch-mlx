"""Scripted stereo rig trajectory for synthetic data generation."""

from __future__ import annotations

import math

import numpy as np

from .math_utils import lookat_rotation, rotmat_to_rvec


class ScriptedRigTrajectory:
    """Smooth forward flight with lateral oscillation and gentle yaw.

    Produces deterministic pose + kinematics at any time t via closed-form
    expressions (no ODE integration needed).
    """

    def __init__(
        self,
        speed_mps: float = 2.0,
        altitude: float = 1.2,
        lateral_amplitude: float = 0.3,
        lateral_freq: float = 0.2,
        yaw_amplitude_deg: float = 5.0,
        yaw_freq: float = 0.1,
    ):
        self.speed = speed_mps
        self.alt = altitude
        self.lat_amp = lateral_amplitude
        self.lat_freq = lateral_freq
        self.yaw_amp = math.radians(yaw_amplitude_deg)
        self.yaw_freq = yaw_freq

    def pose(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (position, rotation_matrix) of the rig centre at time t."""
        px = self.speed * t
        py = self.lat_amp * math.sin(2.0 * math.pi * self.lat_freq * t)
        pz = self.alt
        pos = np.array([px, py, pz], dtype=np.float64)

        yaw = self.yaw_amp * math.sin(2.0 * math.pi * self.yaw_freq * t)
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
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
