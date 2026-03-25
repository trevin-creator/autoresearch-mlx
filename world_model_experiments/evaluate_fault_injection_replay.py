from __future__ import annotations

import argparse
from collections import Counter

import h5py
import numpy as np
import torch

from world_model_experiments.autopilot_bridge import ArbitrationConfig, MotorAutopilotBridge
from world_model_experiments.command_interface import CommandInterfaceConfig
from world_model_experiments.fallback_controller import ConservativeFallbackController, FallbackControllerConfig
from world_model_experiments.flight_state_machine import FlightStateMachine, FlightStateMachineConfig
from world_model_experiments._io import load_actions
from world_model_experiments.informed_dreamer_model import InformedDreamerConfig, InformedFeatureDreamer
from world_model_experiments.safety_shield import SafetyShieldConfig
from world_model_experiments.telemetry import TelemetryLogger


STD_FLOOR = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Actuator fault-injection replay for planner/fallback arbitration")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--ood-threshold", type=float, default=0.10)
    p.add_argument("--use-motor-commands", action="store_true")
    p.add_argument("--use-flight-plan", action="store_true")
    p.add_argument("--fault-mode", type=str, default="stuck_low", choices=["none", "stuck_low", "stuck_high", "dropout", "scale_loss"])
    p.add_argument("--fault-strength", type=float, default=0.6)
    p.add_argument("--fault-start-step", type=int, default=2)
    p.add_argument("--anomaly-ood-gain", type=float, default=2.0)
    p.add_argument("--telemetry-output", type=str, default="")
    return p.parse_args()


def _compute_ood_scores(features: np.ndarray) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
    mu = np.mean(flat, axis=0, keepdims=True)
    sigma = np.maximum(np.std(flat, axis=0, keepdims=True), STD_FLOOR)
    z = np.abs((flat - mu) / sigma)
    frame = np.mean(z > 4.0, axis=1).astype(np.float32)
    return frame.reshape(features.shape[0], features.shape[1])


def _inject_fault(command: np.ndarray, mode: str, strength: float, step: int, start_step: int) -> tuple[np.ndarray, float]:
    cmd = np.asarray(command, dtype=np.float32)
    if mode == "none" or step < start_step:
        return cmd, 0.0

    out = cmd.copy()
    if mode == "stuck_low":
        out[0] = -1.0
    elif mode == "stuck_high":
        out[0] = 1.0
    elif mode == "dropout":
        out[0] = 0.0
    elif mode == "scale_loss":
        out = np.clip(out * max(0.0, 1.0 - strength), -1.0, 1.0)

    delta = float(np.mean(np.abs(out - cmd)))
    return out, delta


def main() -> None:
    args = parse_args()

    with h5py.File(args.dataset, "r") as h5:
        features = np.asarray(h5["features"], dtype=np.float32)
        actions = load_actions(h5, args.use_motor_commands, args.use_flight_plan)
        flight_plan = np.asarray(h5["flight_plan"], dtype=np.float32) if "flight_plan" in h5 else None
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
    flight_sm = FlightStateMachine(
        bridge=bridge,
        fallback=fallback,
        cfg=FlightStateMachineConfig(initial_mode="autonomous", shadow_warmup_steps=1),
    )

    telemetry = TelemetryLogger(args.telemetry_output) if args.telemetry_output else None

    n_eps = min(args.episodes, features.shape[0])
    source_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    fallback_rate = []
    shield_emergency_rate = []
    autonomous_rate = []
    emergency_stop_rate = []
    fault_active_rate = []
    fault_delta_mean = []

    for ep in range(n_eps):
        flight_sm.reset()
        ep_fallback = 0.0
        ep_emergency = 0.0
        ep_autonomous = 0.0
        ep_emergency_mode = 0.0
        ep_fault_active = 0.0
        ep_fault_delta = []

        feat_t = torch.from_numpy(features[ep : ep + 1])
        act_t = torch.from_numpy(actions[ep : ep + 1])
        with torch.no_grad():
            wm = model.world_forward(feat_t, act_t)
            states = wm["state"][0]
            actor_out = model.actor(states)
            mu, _ = torch.chunk(actor_out, 2, dim=-1)
            planner_cmds = torch.tanh(mu).cpu().numpy().astype(np.float32)

        for t in range(features.shape[1]):
            nominal = planner_cmds[t, :4]
            injected, fault_delta = _inject_fault(nominal, args.fault_mode, args.fault_strength, t, args.fault_start_step)
            state_norm = float(np.linalg.norm(pose[ep, t, :3]) + np.linalg.norm(pose[ep, t, 3:]))
            effective_ood = float(ood_scores[ep, t] + args.anomaly_ood_gain * fault_delta)

            packet, decision, snapshot = flight_sm.step(
                timestamp_us=int(timestamps[ep, t]),
                planner_command=injected,
                pose=pose[ep, t],
                pose_delta=pose_delta[ep, t],
                flight_plan=flight_plan[ep, t] if flight_plan is not None else None,
                state_norm=state_norm,
                ood_score=effective_ood,
            )

            source_counts[decision.source] += 1
            mode_counts[snapshot.mode] += 1
            ep_fallback += decision.used_fallback
            ep_emergency += decision.shield_emergency
            ep_autonomous += float(snapshot.mode == "autonomous")
            ep_emergency_mode += float(snapshot.mode == "emergency_stop")
            ep_fault_active += float(fault_delta > 1e-6)
            ep_fault_delta.append(fault_delta)

            if telemetry is not None:
                telemetry.log(
                    {
                        "kind": "fault_replay_step",
                        "episode": ep,
                        "step": t,
                        "timestamp_us": int(timestamps[ep, t]),
                        "fault_mode": args.fault_mode,
                        "fault_delta": fault_delta,
                        "ood_score": effective_ood,
                        "source": decision.source,
                        "mode": snapshot.mode,
                        "shield_emergency": decision.shield_emergency,
                        "motor_pwm": packet.motor_pwm,
                    }
                )

        fallback_rate.append(ep_fallback / max(1, features.shape[1]))
        shield_emergency_rate.append(ep_emergency / max(1, features.shape[1]))
        autonomous_rate.append(ep_autonomous / max(1, features.shape[1]))
        emergency_stop_rate.append(ep_emergency_mode / max(1, features.shape[1]))
        fault_active_rate.append(ep_fault_active / max(1, features.shape[1]))
        fault_delta_mean.append(float(np.mean(ep_fault_delta)) if ep_fault_delta else 0.0)

    if telemetry is not None:
        telemetry.close()

    total_sources = max(1, sum(source_counts.values()))
    result = {
        "episodes": float(n_eps),
        "fault_mode": float(0.0 if args.fault_mode == "none" else 1.0),
        "fault_active_rate": float(np.mean(fault_active_rate)) if fault_active_rate else 0.0,
        "fault_delta_mean": float(np.mean(fault_delta_mean)) if fault_delta_mean else 0.0,
        "planner_rate": float(source_counts.get("planner", 0) / total_sources),
        "fallback_rate": float(np.mean(fallback_rate)) if fallback_rate else 0.0,
        "shield_emergency_rate": float(np.mean(shield_emergency_rate)) if shield_emergency_rate else 0.0,
        "autonomous_rate": float(np.mean(autonomous_rate)) if autonomous_rate else 0.0,
        "emergency_stop_rate": float(np.mean(emergency_stop_rate)) if emergency_stop_rate else 0.0,
    }
    print("fault_replay_eval", result)
    print("fault_replay_sources", dict(source_counts))
    print("fault_replay_modes", dict(mode_counts))


if __name__ == "__main__":
    main()
