from __future__ import annotations

import argparse

import h5py
import numpy as np
import torch

from world_model_experiments.informed_dreamer_model import InformedDreamerConfig, InformedFeatureDreamer
from world_model_experiments.motor_constraints import apply_motor_constraints
from world_model_experiments.motor_simulator import QuadMotorDynamics, domain_randomized_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Closed-loop simulator evaluation for motor-mode informed Dreamer")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--max-action-delta", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
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

    n_eps = min(args.episodes, features.shape[0])

    rewards = []
    action_mags = []
    action_slew = []
    crash_flags = []
    pose_errors = []

    for ep in range(n_eps):
        feat_seq = torch.from_numpy(features[ep : ep + 1])
        act_seq = torch.from_numpy(actions[ep : ep + 1])

        warm = max(1, min(args.warmup, feat_seq.shape[1]))
        with torch.no_grad():
            wm = model.world_forward(feat_seq[:, :warm], act_seq[:, :warm])
            h = wm["h"][:, -1]
            z = wm["z_post"][:, -1]

        sim = QuadMotorDynamics(domain_randomized_config(rng))
        prev = None
        ep_reward = 0.0
        ep_mag = []
        ep_slew = []
        crashed = False
        ep_pose_err = []

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

            # For motor-mode, first 4 dimensions are motor commands.
            motor_cmd = action[0, :4].detach().cpu().numpy().astype(np.float32)
            out = sim.step(motor_cmd)

            ep_reward += float(out["reward"])
            ep_mag.append(float(np.mean(np.abs(motor_cmd))))
            if prev is not None:
                ep_slew.append(float(np.mean(np.abs(motor_cmd - prev))))
            prev = motor_cmd

            pos = out["pose"][:3]
            euler = out["pose"][3:]
            if np.linalg.norm(pos) > 20.0 or np.max(np.abs(euler)) > 1.2:
                crashed = True
                break

            pred_pose_np = model.pose_head(torch.cat([h, z], dim=-1))[0].detach().cpu().numpy()
            ep_pose_err.append(float(np.mean((pred_pose_np - out["pose"]) ** 2)))

        rewards.append(ep_reward)
        action_mags.append(float(np.mean(ep_mag)) if ep_mag else 0.0)
        action_slew.append(float(np.mean(ep_slew)) if ep_slew else 0.0)
        crash_flags.append(1.0 if crashed else 0.0)
        pose_errors.append(float(np.mean(ep_pose_err)) if ep_pose_err else 0.0)

    result = {
        "episodes": float(n_eps),
        "reward_sum_mean": float(np.mean(rewards)) if rewards else 0.0,
        "action_abs_mean": float(np.mean(action_mags)) if action_mags else 0.0,
        "action_slew_mean": float(np.mean(action_slew)) if action_slew else 0.0,
        "crash_rate": float(np.mean(crash_flags)) if crash_flags else 0.0,
        "pose_mse_sim_mean": float(np.mean(pose_errors)) if pose_errors else 0.0,
    }
    print("closed_loop_eval", result)


if __name__ == "__main__":
    main()
