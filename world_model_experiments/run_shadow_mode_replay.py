from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch

from world_model_experiments._io import load_actions, load_sequence_dataset
from world_model_experiments.autopilot_bridge import ArbitrationConfig, MotorAutopilotBridge
from world_model_experiments.command_interface import CommandInterfaceConfig
from world_model_experiments.fallback_controller import ConservativeFallbackController, FallbackControllerConfig
from world_model_experiments.flight_state_machine import FlightStateMachine, FlightStateMachineConfig
from world_model_experiments.informed_dreamer_model import InformedDreamerConfig, InformedFeatureDreamer
from world_model_experiments.safety_shield import SafetyShieldConfig
from world_model_experiments.telemetry import TelemetryLogger

STD_FLOOR = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shadow-mode replay with planner/fallback arbitration and command packets")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--ood-threshold", type=float, default=0.10)
    p.add_argument("--use-motor-commands", action="store_true")
    p.add_argument("--use-flight-plan", action="store_true")
    p.add_argument("--initial-mode", type=str, default="shadow", choices=["shadow", "autonomous", "fallback"])
    p.add_argument("--shadow-warmup-steps", type=int, default=4)
    p.add_argument("--fallback-hold-steps", type=int, default=2)
    p.add_argument("--recovery-hold-steps", type=int, default=3)
    p.add_argument("--max-tracking-error", type=float, default=5.0)
    p.add_argument("--telemetry-output", type=str, default="")
    return p.parse_args()


def _compute_ood_scores(features: np.ndarray) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
    mu = np.mean(flat, axis=0, keepdims=True)
    sigma = np.maximum(np.std(flat, axis=0, keepdims=True), STD_FLOOR)
    z = np.abs((flat - mu) / sigma)
    frame = np.mean(z > 4.0, axis=1).astype(np.float32)
    return frame.reshape(features.shape[0], features.shape[1])


def main() -> None:
    args = parse_args()

    dataset = load_sequence_dataset(args.dataset)
    features = np.asarray(dataset["features"], dtype=np.float32)
    actions = load_actions(dataset, args.use_motor_commands, args.use_flight_plan)
    flight_plan = np.asarray(dataset["flight_plan"], dtype=np.float32) if "flight_plan" in dataset else None
    pose = np.asarray(dataset["pose"], dtype=np.float32)
    pose_delta = np.asarray(dataset["pose_delta"], dtype=np.float32)
    timestamps = np.asarray(dataset["timestamps_us"], dtype=np.int64)

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
    flight_sm = FlightStateMachine(
        bridge=bridge,
        fallback=fallback,
        cfg=FlightStateMachineConfig(
            initial_mode=args.initial_mode,
            shadow_warmup_steps=args.shadow_warmup_steps,
            fallback_hold_steps=args.fallback_hold_steps,
            recovery_hold_steps=args.recovery_hold_steps,
            max_tracking_error=args.max_tracking_error,
        ),
    )

    n_eps = min(args.episodes, features.shape[0])
    telemetry = TelemetryLogger(args.telemetry_output) if args.telemetry_output else None
    source_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    fallback_rate = []
    shield_emergency_rate = []
    pwm_means = []
    autonomous_rate = []
    emergency_stop_rate = []
    mode_change_rate = []

    for ep in range(n_eps):
        flight_sm.reset()
        ep_fallback = 0.0
        ep_emergency = 0.0
        ep_pwm = []
        ep_autonomous = 0.0
        ep_emergency_mode = 0.0
        ep_mode_changes = 0.0

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
            state_norm = float(np.linalg.norm(pose[ep, t, :3]) + np.linalg.norm(pose[ep, t, 3:]))
            packet, decision, snapshot = flight_sm.step(
                timestamp_us=int(timestamps[ep, t]),
                planner_command=planner,
                pose=pose[ep, t],
                pose_delta=pose_delta[ep, t],
                flight_plan=flight_plan[ep, t] if flight_plan is not None else None,
                state_norm=state_norm,
                ood_score=float(ood_scores[ep, t]),
            )
            source_counts[decision.source] += 1
            mode_counts[snapshot.mode] += 1
            ep_fallback += decision.used_fallback
            ep_emergency += decision.shield_emergency
            ep_pwm.append(float(np.mean(packet.motor_pwm)))
            ep_autonomous += float(snapshot.mode == "autonomous")
            ep_emergency_mode += float(snapshot.mode == "emergency_stop")
            ep_mode_changes += snapshot.mode_changed

            if telemetry is not None:
                telemetry.log(
                    {
                        "kind": "shadow_replay_step",
                        "episode": ep,
                        "step": t,
                        "timestamp_us": int(timestamps[ep, t]),
                        "source": decision.source,
                        "mode": snapshot.mode,
                        "ood_score": float(ood_scores[ep, t]),
                        "used_fallback": decision.used_fallback,
                        "shield_emergency": decision.shield_emergency,
                        "motor_pwm": packet.motor_pwm,
                    }
                )

        fallback_rate.append(ep_fallback / max(1, features.shape[1]))
        shield_emergency_rate.append(ep_emergency / max(1, features.shape[1]))
        pwm_means.append(float(np.mean(ep_pwm)) if ep_pwm else 0.0)
        autonomous_rate.append(ep_autonomous / max(1, features.shape[1]))
        emergency_stop_rate.append(ep_emergency_mode / max(1, features.shape[1]))
        mode_change_rate.append(ep_mode_changes / max(1, features.shape[1]))

    if telemetry is not None:
        telemetry.close()

    result = {
        "episodes": float(n_eps),
        "planner_rate": float(source_counts.get("planner", 0) / max(1, sum(source_counts.values()))),
        "fallback_rate": float(np.mean(fallback_rate)) if fallback_rate else 0.0,
        "shield_emergency_rate": float(np.mean(shield_emergency_rate)) if shield_emergency_rate else 0.0,
        "autonomous_rate": float(np.mean(autonomous_rate)) if autonomous_rate else 0.0,
        "emergency_stop_rate": float(np.mean(emergency_stop_rate)) if emergency_stop_rate else 0.0,
        "mode_change_rate": float(np.mean(mode_change_rate)) if mode_change_rate else 0.0,
        "avg_pwm_mean": float(np.mean(pwm_means)) if pwm_means else 0.0,
    }
    print("shadow_mode_eval", result)
    print("shadow_mode_sources", dict(source_counts))
    print("shadow_mode_modes", dict(mode_counts))


if __name__ == "__main__":
    main()
