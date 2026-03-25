from __future__ import annotations

import argparse

import numpy as np
import torch

from world_model_experiments._errors import ERR_NO_MOTOR_COMMANDS
from world_model_experiments._io import load_actions, load_sequence_dataset
from world_model_experiments.informed_dreamer_model import InformedDreamerConfig, InformedFeatureDreamer
from world_model_experiments.motor_simulator import REWARD_TRANSLATION_WEIGHT, REWARD_YAW_WEIGHT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay real trajectory features and report prediction parity")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--use-motor-commands", action="store_true")
    p.add_argument("--use-flight-plan", action="store_true")
    return p.parse_args()


def _bucket_name(speed: float) -> str:
    if speed < 0.02:
        return "slow"
    if speed < 0.06:
        return "medium"
    return "fast"


def main() -> None:
    args = parse_args()

    dataset = load_sequence_dataset(args.dataset)
    features = np.asarray(dataset["features"], dtype=np.float32)
    if args.use_motor_commands and "motor_commands" not in dataset:
        raise ValueError(ERR_NO_MOTOR_COMMANDS)
    actions = load_actions(dataset, args.use_motor_commands, args.use_flight_plan)

    pose_delta = np.asarray(dataset["pose_delta"], dtype=np.float32)
    reward = np.asarray(dataset["reward"], dtype=np.float32) if "reward" in dataset else None

    if reward is None:
        transl = np.linalg.norm(pose_delta[..., :3], axis=-1)
        yaw = np.abs(pose_delta[..., 5])
        reward = (REWARD_TRANSLATION_WEIGHT * transl + REWARD_YAW_WEIGHT * yaw).astype(np.float32)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = InformedDreamerConfig(**ckpt["config"])
    model = InformedFeatureDreamer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    feat_t = torch.from_numpy(features)
    act_t = torch.from_numpy(actions)
    with torch.no_grad():
        out = model.world_forward(feat_t, act_t)
        pred_pose_delta = out["pose_delta"].cpu().numpy()
        pred_reward = out["reward"].cpu().numpy()

    pose_delta_mse = np.mean((pred_pose_delta - pose_delta) ** 2, axis=-1)
    reward_mse = (pred_reward - reward) ** 2
    pose_delta_mse = np.asarray(pose_delta_mse, dtype=np.float64)
    reward_mse = np.asarray(reward_mse, dtype=np.float64)

    speed = np.linalg.norm(pose_delta[..., :3], axis=-1)
    bucket_stats: dict[str, dict[str, float]] = {}
    for bucket in ["slow", "medium", "fast"]:
        mask = np.vectorize(_bucket_name)(speed) == bucket
        if not np.any(mask):
            bucket_stats[bucket] = {
                "count": 0.0,
                "pose_delta_mse": 0.0,
                "reward_mse": 0.0,
            }
            continue
        bucket_stats[bucket] = {
            "count": float(np.sum(mask)),
            "pose_delta_mse": float(np.mean(np.asarray(pose_delta_mse[mask], dtype=np.float64))),
            "reward_mse": float(np.mean(np.asarray(reward_mse[mask], dtype=np.float64))),
        }

    result = {
        "global_pose_delta_mse": float(np.mean(pose_delta_mse)),
        "global_reward_mse": float(np.mean(reward_mse)),
        "bucket_slow_pose_delta_mse": bucket_stats["slow"]["pose_delta_mse"],
        "bucket_medium_pose_delta_mse": bucket_stats["medium"]["pose_delta_mse"],
        "bucket_fast_pose_delta_mse": bucket_stats["fast"]["pose_delta_mse"],
        "bucket_fast_count": bucket_stats["fast"]["count"],
    }
    print("real_replay_eval", result)


if __name__ == "__main__":
    main()
