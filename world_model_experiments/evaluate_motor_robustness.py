from __future__ import annotations

import argparse
import json
from collections import deque

import h5py
import numpy as np
import torch

from world_model_experiments._errors import ERR_NO_MOTOR_COMMANDS
from world_model_experiments.informed_dreamer_model import InformedDreamerConfig, InformedFeatureDreamer
from world_model_experiments.motor_constraints import apply_motor_constraints
from world_model_experiments.motor_simulator import QuadMotorDynamics, domain_randomized_config
from world_model_experiments.safety_shield import SafetyShieldConfig, apply_safety_shield


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robustness evaluation under simulator disturbances")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--max-action-delta", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-motor-commands", action="store_true")
    p.add_argument("--use-flight-plan", action="store_true")
    p.add_argument("--scenario-mode", type=str, choices=["preset", "matrix"], default="preset")
    p.add_argument("--wind-stds", type=str, default="0.0,0.5,1.0")
    p.add_argument("--act-noise-stds", type=str, default="0.0,0.05,0.12")
    p.add_argument("--latency-steps", type=str, default="0,1,2")
    p.add_argument("--use-safety-shield", action="store_true")
    p.add_argument("--shield-max-slew", type=float, default=0.2)
    p.add_argument("--shield-max-abs-cmd", type=float, default=1.0)
    p.add_argument("--shield-emergency-stop-norm", type=float, default=30.0)
    return p.parse_args()


def _load_actions(h5: h5py.File, use_motor_commands: bool, use_flight_plan: bool) -> np.ndarray:
    if use_motor_commands:
        if "motor_commands" not in h5:
            raise ValueError(ERR_NO_MOTOR_COMMANDS)
        actions = np.asarray(h5["motor_commands"], dtype=np.float32)
    else:
        actions = np.asarray(h5["actions"], dtype=np.float32)
    if use_flight_plan and "flight_plan" in h5:
        actions = np.concatenate([actions, np.asarray(h5["flight_plan"], dtype=np.float32)], axis=-1)
    return actions


def _scenario_table() -> dict[str, dict[str, float | int]]:
    return {
        "calm": {"wind_std": 0.0, "act_noise_std": 0.0, "latency_steps": 0},
        "wind": {"wind_std": 0.5, "act_noise_std": 0.0, "latency_steps": 0},
        "gust": {"wind_std": 1.0, "act_noise_std": 0.05, "latency_steps": 1},
        "noisy_actuation": {"wind_std": 0.2, "act_noise_std": 0.12, "latency_steps": 2},
    }


def _parse_floats(csv_text: str) -> list[float]:
    return [float(x.strip()) for x in csv_text.split(",") if x.strip()]


def _parse_ints(csv_text: str) -> list[int]:
    return [int(x.strip()) for x in csv_text.split(",") if x.strip()]


def _scenario_matrix(args: argparse.Namespace) -> dict[str, dict[str, float | int]]:
    scenarios: dict[str, dict[str, float | int]] = {}
    winds = _parse_floats(args.wind_stds)
    noises = _parse_floats(args.act_noise_stds)
    latencies = _parse_ints(args.latency_steps)
    for w in winds:
        for n in noises:
            for l in latencies:
                key = f"w{w:.2f}_n{n:.2f}_l{l}"
                scenarios[key] = {
                    "wind_std": float(w),
                    "act_noise_std": float(n),
                    "latency_steps": int(l),
                }
    return scenarios


def _evaluate_scenario(
    model: InformedFeatureDreamer,
    features: np.ndarray,
    actions: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
    scenario: dict[str, float | int],
) -> dict[str, float]:
    n_eps = min(args.episodes, features.shape[0])

    rewards = []
    action_mags = []
    action_slew = []
    action_energy = []
    crash_flags = []
    rollout_survival = []
    shield_modified = []
    shield_emergency = []

    latency_steps = int(scenario["latency_steps"])
    wind_std = float(scenario["wind_std"])
    act_noise_std = float(scenario["act_noise_std"])

    for ep in range(n_eps):
        feat_seq = torch.from_numpy(features[ep : ep + 1])
        act_seq = torch.from_numpy(actions[ep : ep + 1])

        warm = max(1, min(args.warmup, feat_seq.shape[1]))
        with torch.no_grad():
            wm = model.world_forward(feat_seq[:, :warm], act_seq[:, :warm])
            h = wm["h"][:, -1]
            z = wm["z_post"][:, -1]

        sim = QuadMotorDynamics(domain_randomized_config(rng))
        cmd_queue: deque[np.ndarray] = deque()

        prev = None
        ep_reward = 0.0
        ep_mag = []
        ep_slew = []
        ep_energy = []
        crashed = False
        survived_steps = 0
        ep_shield_modified = 0.0
        ep_shield_emergency = 0.0

        shield_cfg = SafetyShieldConfig(
            max_slew=args.shield_max_slew,
            max_abs_cmd=args.shield_max_abs_cmd,
            emergency_stop_norm=args.shield_emergency_stop_norm,
        )

        for _ in range(args.horizon):
            with torch.no_grad():
                s = torch.cat([h, z], dim=-1)
                actor_out = model.actor(s)
                mu, _ = torch.chunk(actor_out, 2, dim=-1)
                action = torch.tanh(mu).unsqueeze(1)
                action = apply_motor_constraints(action, low=-1.0, high=1.0, max_delta=args.max_action_delta)
                action = action.squeeze(1)

                a_emb = model.action_emb(action.unsqueeze(1)).squeeze(1)
                rssm_in = torch.cat([z, a_emb], dim=-1).unsqueeze(1)
                h_next, _ = model.rssm(rssm_in, h.unsqueeze(0))
                h = h_next.squeeze(1)
                z = model.prior(h)

            cmd = action[0, :4].detach().cpu().numpy().astype(np.float32)
            if act_noise_std > 0.0:
                cmd = cmd + rng.normal(0.0, act_noise_std, size=cmd.shape).astype(np.float32)
                cmd = np.clip(cmd, -1.0, 1.0)

            if latency_steps > 0:
                cmd_queue.append(cmd)
                if len(cmd_queue) <= latency_steps:
                    applied = np.zeros_like(cmd)
                else:
                    applied = cmd_queue.popleft()
            else:
                applied = cmd

            if args.use_safety_shield:
                state_norm = float(np.linalg.norm(sim.state.position) + np.linalg.norm(sim.state.euler))
                applied, shield = apply_safety_shield(applied, prev, state_norm, shield_cfg)
                ep_shield_modified += float(shield["modified"])
                ep_shield_emergency += float(shield["emergency_stop"])

            out = sim.step(applied)

            # Disturb translational state to emulate wind/gust impulses.
            if wind_std > 0.0:
                sim.state.velocity = sim.state.velocity + rng.normal(0.0, wind_std, size=3).astype(np.float32) * sim.cfg.dt
                sim.state.position = sim.state.position + sim.state.velocity * sim.cfg.dt
                out["pose"][:3] = sim.state.position.astype(np.float32)

            ep_reward += float(out["reward"])
            ep_mag.append(float(np.mean(np.abs(applied))))
            ep_energy.append(float(np.mean(np.square(applied))))
            if prev is not None:
                ep_slew.append(float(np.mean(np.abs(applied - prev))))
            prev = applied

            pos = out["pose"][:3]
            euler = out["pose"][3:]
            if np.linalg.norm(pos) > 20.0 or np.max(np.abs(euler)) > 1.2:
                crashed = True
                break

            survived_steps += 1

        rewards.append(ep_reward)
        action_mags.append(float(np.mean(ep_mag)) if ep_mag else 0.0)
        action_slew.append(float(np.mean(ep_slew)) if ep_slew else 0.0)
        action_energy.append(float(np.mean(ep_energy)) if ep_energy else 0.0)
        crash_flags.append(1.0 if crashed else 0.0)
        rollout_survival.append(float(survived_steps / max(1, args.horizon)))
        shield_modified.append(ep_shield_modified / max(1, args.horizon))
        shield_emergency.append(ep_shield_emergency / max(1, args.horizon))

    return {
        "episodes": float(n_eps),
        "reward_sum_mean": float(np.mean(rewards)) if rewards else 0.0,
        "action_abs_mean": float(np.mean(action_mags)) if action_mags else 0.0,
        "action_slew_mean": float(np.mean(action_slew)) if action_slew else 0.0,
        "energy_proxy_mean": float(np.mean(action_energy)) if action_energy else 0.0,
        "crash_rate": float(np.mean(crash_flags)) if crash_flags else 0.0,
        "termination_rate": float(np.mean(crash_flags)) if crash_flags else 0.0,
        "rollout_survival_mean": float(np.mean(rollout_survival)) if rollout_survival else 0.0,
        "shield_modified_rate": float(np.mean(shield_modified)) if shield_modified else 0.0,
        "shield_emergency_rate": float(np.mean(shield_emergency)) if shield_emergency else 0.0,
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.dataset, "r") as h5:
        features = np.asarray(h5["features"], dtype=np.float32)
        actions = _load_actions(h5, args.use_motor_commands, args.use_flight_plan)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = InformedDreamerConfig(**ckpt["config"])
    model = InformedFeatureDreamer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scenarios = _scenario_table() if args.scenario_mode == "preset" else _scenario_matrix(args)
    out: dict[str, dict[str, float]] = {}
    for name, sc in scenarios.items():
        out[name] = _evaluate_scenario(model, features, actions, args, rng, sc)

    print("robust_eval", json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
