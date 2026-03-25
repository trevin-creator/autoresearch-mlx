from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from world_model_experiments.motor_simulator import QuadMotorDynamics, domain_randomized_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build simulator rollout dataset with 4-motor commands")
    p.add_argument("--output", type=str, default="artifacts/sim/sim_motor_rollouts.h5")
    p.add_argument("--num-sequences", type=int, default=64)
    p.add_argument("--sequence-len", type=int, default=16)
    p.add_argument("--feature-dim", type=int, default=79)
    p.add_argument("--action-noise", type=float, default=0.2)
    p.add_argument("--wind-std", type=float, default=0.0)
    p.add_argument("--actuation-noise-std", type=float, default=0.0)
    p.add_argument("--latency-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_sequences <= 0:
        raise ValueError("--num-sequences must be > 0")
    if args.sequence_len <= 0:
        raise ValueError("--sequence-len must be > 0")
    if args.feature_dim <= 0:
        raise ValueError("--feature-dim must be > 0")
    if args.action_noise < 0.0:
        raise ValueError("--action-noise must be >= 0")
    if args.wind_std < 0.0:
        raise ValueError("--wind-std must be >= 0")
    if args.actuation_noise_std < 0.0:
        raise ValueError("--actuation-noise-std must be >= 0")
    if args.latency_steps < 0:
        raise ValueError("--latency-steps must be >= 0")


def _state_to_feature(pose: np.ndarray, pose_delta: np.ndarray, motors: np.ndarray, feature_dim: int, rng: np.random.Generator) -> np.ndarray:
    base = np.concatenate(
        [
            pose,
            pose_delta,
            motors,
            np.square(motors),
            np.sin(pose[3:]),
            np.cos(pose[3:]),
            np.array([np.linalg.norm(pose_delta[:3]), abs(pose_delta[5])], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)

    if base.shape[0] >= feature_dim:
        return base[:feature_dim].astype(np.float32)

    # Fill remaining dimensions with stable random projections from the base vector.
    extra = feature_dim - base.shape[0]
    proj = rng.normal(0.0, 0.2, size=(extra, base.shape[0])).astype(np.float32)
    tail = (proj @ base).astype(np.float32)
    return np.concatenate([base, tail], axis=0).astype(np.float32)


def build_dataset(args: argparse.Namespace) -> Path:
    rng = np.random.default_rng(args.seed)

    seq_features = []
    seq_actions = []
    seq_motor = []
    seq_pose = []
    seq_pose_delta = []
    seq_reward = []
    seq_continue = []
    seq_timestamps = []

    # Keep compatibility with existing pipeline by providing flight_plan and proxy actions.
    seq_flight_plan = []

    for _ in range(args.num_sequences):
        sim = QuadMotorDynamics(domain_randomized_config(rng))
        cmd_queue: list[np.ndarray] = []

        feat_list = []
        action_proxy_list = []
        motor_list = []
        pose_list = []
        pose_delta_list = []
        reward_list = []
        cont_list = []
        t_list = []
        flight_plan_list = []

        prev_cmd = np.zeros(4, dtype=np.float32)
        for t in range(args.sequence_len):
            cmd = np.clip(prev_cmd + rng.normal(0.0, args.action_noise, size=4).astype(np.float32), -1.0, 1.0)
            prev_cmd = 0.85 * prev_cmd + 0.15 * cmd

            applied = cmd.astype(np.float32)
            if args.actuation_noise_std > 0.0:
                applied = np.clip(
                    applied + rng.normal(0.0, args.actuation_noise_std, size=4).astype(np.float32),
                    -1.0,
                    1.0,
                )

            if args.latency_steps > 0:
                cmd_queue.append(applied)
                if len(cmd_queue) <= args.latency_steps:
                    applied = np.zeros(4, dtype=np.float32)
                else:
                    applied = cmd_queue.pop(0)

            out = sim.step(applied)

            if args.wind_std > 0.0:
                sim.state.velocity = sim.state.velocity + rng.normal(0.0, args.wind_std, size=3).astype(np.float32) * sim.cfg.dt
                sim.state.position = sim.state.position + sim.state.velocity * sim.cfg.dt
                out["pose"][:3] = sim.state.position.astype(np.float32)
            pose = out["pose"]
            pose_delta = out["pose_delta"]
            motors = out["motors"]

            feat = _state_to_feature(pose, pose_delta, motors, args.feature_dim, rng)

            # Action proxy retained for backward-compatible training paths.
            action_proxy = np.concatenate([pose_delta[:3], sim.state.body_rates], axis=0).astype(np.float32)

            # Simple placeholder flight plan for compatibility (3 future blocks of 8 dims).
            fp = np.tile(
                np.concatenate([pose_delta[:3], [pose_delta[5]], pose[:3], [pose[5]]], axis=0),
                3,
            ).astype(np.float32)

            feat_list.append(feat)
            action_proxy_list.append(action_proxy)
            motor_list.append(applied.astype(np.float32))
            pose_list.append(pose.astype(np.float32))
            pose_delta_list.append(pose_delta.astype(np.float32))
            reward_list.append(np.float32(out["reward"]))
            cont_list.append(np.float32(1.0 if t < args.sequence_len - 1 else 0.0))
            t_list.append(np.int64(t))
            flight_plan_list.append(fp)

        seq_features.append(np.stack(feat_list, axis=0))
        seq_actions.append(np.stack(action_proxy_list, axis=0))
        seq_motor.append(np.stack(motor_list, axis=0))
        seq_pose.append(np.stack(pose_list, axis=0))
        seq_pose_delta.append(np.stack(pose_delta_list, axis=0))
        seq_reward.append(np.asarray(reward_list, dtype=np.float32))
        seq_continue.append(np.asarray(cont_list, dtype=np.float32))
        seq_timestamps.append(np.asarray(t_list, dtype=np.int64))
        seq_flight_plan.append(np.stack(flight_plan_list, axis=0))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as h5:
        h5.create_dataset("features", data=np.stack(seq_features, axis=0), compression="gzip")
        h5.create_dataset("actions", data=np.stack(seq_actions, axis=0), compression="gzip")
        h5.create_dataset("motor_commands", data=np.stack(seq_motor, axis=0), compression="gzip")
        h5.create_dataset("pose", data=np.stack(seq_pose, axis=0), compression="gzip")
        h5.create_dataset("pose_delta", data=np.stack(seq_pose_delta, axis=0), compression="gzip")
        h5.create_dataset("reward", data=np.stack(seq_reward, axis=0), compression="gzip")
        h5.create_dataset("continue", data=np.stack(seq_continue, axis=0), compression="gzip")
        h5.create_dataset("timestamps_us", data=np.stack(seq_timestamps, axis=0), compression="gzip")
        h5.create_dataset("flight_plan", data=np.stack(seq_flight_plan, axis=0), compression="gzip")

    return output


def main() -> None:
    args = parse_args()
    _validate_args(args)
    out = build_dataset(args)
    logger.info("wrote simulator rollout dataset: %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
