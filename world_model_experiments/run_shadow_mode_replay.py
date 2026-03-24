from __future__ import annotations

import argparse
from collections import Counter

import h5py
import numpy as np
import torch

from world_model_experiments.autopilot_bridge import ArbitrationConfig, MotorAutopilotBridge
from world_model_experiments.command_interface import CommandInterfaceConfig
from world_model_experiments.fallback_controller import ConservativeFallbackController, FallbackControllerConfig
from world_model_experiments.informed_dreamer_model import InformedDreamerConfig, InformedFeatureDreamer
from world_model_experiments.safety_shield import SafetyShieldConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shadow-mode replay with planner/fallback arbitration and command packets")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--ood-threshold", type=float, default=0.10)
    p.add_argument("--use-motor-commands", action="store_true")
    p.add_argument("--use-flight-plan", action="store_true")
    return p.parse_args()


def _load_actions(h5: h5py.File, use_motor_commands: bool, use_flight_plan: bool) -> np.ndarray:
    if use_motor_commands:
        if "motor_commands" not in h5:
            raise ValueError("--use-motor-commands set but dataset has no motor_commands key")
        actions = np.asarray(h5["motor_commands"], dtype=np.float32)
    else:
        actions = np.asarray(h5["actions"], dtype=np.float32)
    if use_flight_plan and "flight_plan" in h5:
        actions = np.concatenate([actions, np.asarray(h5["flight_plan"], dtype=np.float32)], axis=-1)
    return actions


def _compute_ood_scores(features: np.ndarray) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
    mu = np.mean(flat, axis=0, keepdims=True)
    sigma = np.std(flat, axis=0, keepdims=True) + 1e-6
    z = np.abs((flat - mu) / sigma)
    frame = np.mean(z > 4.0, axis=1).astype(np.float32)
    return frame.reshape(features.shape[0], features.shape[1])


def main() -> None:
    args = parse_args()

    with h5py.File(args.dataset, "r") as h5:
        features = np.asarray(h5["features"], dtype=np.float32)
        actions = _load_actions(h5, args.use_motor_commands, args.use_flight_plan)
        pose = np.asarray(h5["pose"], dtype=np.float32)
        pose_delta = np.asarray(h5["pose_delta"], dtype=np.float32)
        timestamps = np.asarray(h5["timestamps_us"], dtype=np.int64)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = InformedDreamerConfig(**ckpt["config"])
    model = InformedFeatureDreamer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ood_scores = _compute_ood_scores(features)

    bridge = MotorAutopilotBridge(
        interface_cfg=CommandInterfaceConfig(),
        shield_cfg=SafetyShieldConfig(),
        arbitration_cfg=ArbitrationConfig(ood_fallback_threshold=args.ood_threshold),
    )
    fallback = ConservativeFallbackController(FallbackControllerConfig())

    n_eps = min(args.episodes, features.shape[0])
    source_counts: Counter[str] = Counter()
    fallback_rate = []
    shield_emergency_rate = []
    pwm_means = []

    for ep in range(n_eps):
        bridge.reset()
        ep_fallback = 0.0
        ep_emergency = 0.0
        ep_pwm = []

        feat_t = torch.from_numpy(features[ep : ep + 1])
        act_t = torch.from_numpy(actions[ep : ep + 1])
        with torch.no_grad():
            wm = model.world_forward(feat_t, act_t)
            states = wm["state"][0]
            actor_out = model.actor(states)
            mu, _ = torch.chunk(actor_out, 2, dim=-1)
            planner_cmds = torch.tanh(mu).cpu().numpy().astype(np.float32)

        for t in range(features.shape[1]):
            planner = planner_cmds[t, :4]
            fallback_cmd = fallback.command(pose[ep, t], pose_delta[ep, t], mode="hover")
            state_norm = float(np.linalg.norm(pose[ep, t, :3]) + np.linalg.norm(pose[ep, t, 3:]))
            packet, decision = bridge.step(
                timestamp_us=int(timestamps[ep, t]),
                planner_command=planner,
                fallback_command=fallback_cmd,
                state_norm=state_norm,
                ood_score=float(ood_scores[ep, t]),
            )
            source_counts[decision.source] += 1
            ep_fallback += decision.used_fallback
            ep_emergency += decision.shield_emergency
            ep_pwm.append(float(np.mean(packet.motor_pwm)))

        fallback_rate.append(ep_fallback / max(1, features.shape[1]))
        shield_emergency_rate.append(ep_emergency / max(1, features.shape[1]))
        pwm_means.append(float(np.mean(ep_pwm)) if ep_pwm else 0.0)

    result = {
        "episodes": float(n_eps),
        "planner_rate": float(source_counts.get("planner", 0) / max(1, sum(source_counts.values()))),
        "fallback_rate": float(np.mean(fallback_rate)) if fallback_rate else 0.0,
        "shield_emergency_rate": float(np.mean(shield_emergency_rate)) if shield_emergency_rate else 0.0,
        "avg_pwm_mean": float(np.mean(pwm_means)) if pwm_means else 0.0,
    }
    print("shadow_mode_eval", result)
    print("shadow_mode_sources", dict(source_counts))


if __name__ == "__main__":
    main()
