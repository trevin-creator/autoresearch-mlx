from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Reward shaping weights
REWARD_TRANSLATION_WEIGHT = 0.8
REWARD_YAW_WEIGHT = 0.2

# Numerical stability floor for motor time-constant
MOTOR_TAU_EPS = 1e-4


@dataclass(frozen=True)
class QuadSimConfig:
    dt: float = 1.0 / 90.0
    gravity: float = 9.81
    mass: float = 0.9
    arm_length: float = 0.12
    inertia_x: float = 0.011
    inertia_y: float = 0.011
    inertia_z: float = 0.021
    max_thrust_newton: float = 14.0
    torque_coeff: float = 0.06
    lin_drag: float = 0.12
    ang_drag: float = 0.08
    motor_tau: float = 0.04
    command_scale: float = 1.0


@dataclass
class QuadState:
    position: np.ndarray
    velocity: np.ndarray
    euler: np.ndarray
    body_rates: np.ndarray
    motors: np.ndarray


class QuadMotorDynamics:
    """Lightweight 6-DoF quadrotor model with 4 motor commands in [-1, 1]."""

    def __init__(self, cfg: QuadSimConfig) -> None:
        self.cfg = cfg
        self.state = self.reset()

    def reset(self) -> QuadState:
        self.state = QuadState(
            position=np.zeros(3, dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            euler=np.zeros(3, dtype=np.float32),
            body_rates=np.zeros(3, dtype=np.float32),
            motors=np.zeros(4, dtype=np.float32),
        )
        return self.state

    def _rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = float(euler[0]), float(euler[1]), float(euler[2])
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
        return rz @ ry @ rx

    def _motor_to_forces(self, motor_power: np.ndarray) -> tuple[float, np.ndarray]:
        thrusts = self.cfg.max_thrust_newton * np.clip(motor_power, 0.0, 1.0)
        total = float(np.sum(thrusts))
        arm = self.cfg.arm_length

        # Motor order: [front_left, front_right, rear_right, rear_left]
        tau_x = arm * (thrusts[0] + thrusts[1] - thrusts[2] - thrusts[3])
        tau_y = arm * (thrusts[0] - thrusts[1] - thrusts[2] + thrusts[3])
        tau_z = self.cfg.torque_coeff * ((thrusts[0] - thrusts[1]) + (thrusts[2] - thrusts[3]))
        return total, np.asarray([tau_x, tau_y, tau_z], dtype=np.float32)

    def step(self, command: np.ndarray) -> dict[str, np.ndarray]:
        dt = self.cfg.dt
        cmd = np.asarray(command, dtype=np.float32).reshape(4)
        cmd = np.clip(cmd, -1.0, 1.0)

        target_motors = 0.5 * (cmd + 1.0) * self.cfg.command_scale
        alpha = float(np.clip(dt / max(self.cfg.motor_tau, MOTOR_TAU_EPS), 0.0, 1.0))
        self.state.motors = (1.0 - alpha) * self.state.motors + alpha * target_motors
        self.state.motors = np.clip(self.state.motors, 0.0, 1.0)

        total_thrust, torques = self._motor_to_forces(self.state.motors)

        rmat = self._rotation_matrix(self.state.euler)
        thrust_world = rmat @ np.asarray([0.0, 0.0, total_thrust], dtype=np.float32)
        gravity_world = np.asarray([0.0, 0.0, -self.cfg.mass * self.cfg.gravity], dtype=np.float32)
        drag_world = -self.cfg.lin_drag * self.state.velocity
        force_world = thrust_world + gravity_world + drag_world

        accel = force_world / self.cfg.mass
        self.state.velocity = self.state.velocity + dt * accel
        self.state.position = self.state.position + dt * self.state.velocity

        rates = self.state.body_rates
        ang_drag = -self.cfg.ang_drag * rates
        ang_accel = np.asarray(
            [
                (torques[0] + ang_drag[0]) / self.cfg.inertia_x,
                (torques[1] + ang_drag[1]) / self.cfg.inertia_y,
                (torques[2] + ang_drag[2]) / self.cfg.inertia_z,
            ],
            dtype=np.float32,
        )
        self.state.body_rates = self.state.body_rates + dt * ang_accel
        self.state.euler = self.state.euler + dt * self.state.body_rates

        pose = np.concatenate([self.state.position, self.state.euler], axis=0).astype(np.float32)
        pose_delta = np.concatenate([dt * self.state.velocity, dt * self.state.body_rates], axis=0).astype(np.float32)

        reward = np.float32(
            REWARD_TRANSLATION_WEIGHT * np.linalg.norm(pose_delta[:3])
            + REWARD_YAW_WEIGHT * abs(pose_delta[5])
        )
        cont = np.float32(1.0)

        return {
            "pose": pose,
            "pose_delta": pose_delta,
            "reward": np.asarray(reward, dtype=np.float32),
            "continue": np.asarray(cont, dtype=np.float32),
            "motors": self.state.motors.copy().astype(np.float32),
        }


def domain_randomized_config(rng: np.random.Generator) -> QuadSimConfig:
    base = QuadSimConfig()
    return QuadSimConfig(
        dt=base.dt,
        gravity=base.gravity,
        mass=float(base.mass * rng.uniform(0.85, 1.15)),
        arm_length=float(base.arm_length * rng.uniform(0.9, 1.1)),
        inertia_x=float(base.inertia_x * rng.uniform(0.8, 1.2)),
        inertia_y=float(base.inertia_y * rng.uniform(0.8, 1.2)),
        inertia_z=float(base.inertia_z * rng.uniform(0.8, 1.2)),
        max_thrust_newton=float(base.max_thrust_newton * rng.uniform(0.8, 1.2)),
        torque_coeff=float(base.torque_coeff * rng.uniform(0.8, 1.2)),
        lin_drag=float(base.lin_drag * rng.uniform(0.7, 1.4)),
        ang_drag=float(base.ang_drag * rng.uniform(0.7, 1.4)),
        motor_tau=float(base.motor_tau * rng.uniform(0.7, 1.5)),
        command_scale=base.command_scale,
    )
