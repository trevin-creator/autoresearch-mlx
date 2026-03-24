from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from world_model_experiments.informed_dreamer_model import InformedDreamerConfig, InformedFeatureDreamer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate informed-dreamer privileged decoders on TUMVIE features")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--use-flight-plan", action="store_true")
    p.add_argument("--use-motor-commands", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with h5py.File(args.dataset, "r") as h5:
        features = np.asarray(h5["features"], dtype=np.float32)
        if args.use_motor_commands:
            if "motor_commands" not in h5:
                raise ValueError("--use-motor-commands set but dataset has no motor_commands key")
            actions = np.asarray(h5["motor_commands"], dtype=np.float32)
        else:
            actions = np.asarray(h5["actions"], dtype=np.float32)
        if args.use_flight_plan and "flight_plan" in h5:
            actions = np.concatenate([actions, np.asarray(h5["flight_plan"], dtype=np.float32)], axis=-1)
        pose = np.asarray(h5["pose"], dtype=np.float32)
        pose_delta = np.asarray(h5["pose_delta"], dtype=np.float32)
        reward = np.asarray(h5["reward"], dtype=np.float32)
        cont = np.asarray(h5["continue"], dtype=np.float32)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = InformedDreamerConfig(**ckpt["config"])
    model = InformedFeatureDreamer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        out = model.world_forward(torch.from_numpy(features), torch.from_numpy(actions))

    pose_mse = float(np.mean((out["pose"].cpu().numpy() - pose) ** 2))
    pd_mse = float(np.mean((out["pose_delta"].cpu().numpy() - pose_delta) ** 2))
    rw_mse = float(np.mean((out["reward"].cpu().numpy() - reward) ** 2))
    cont_prob = 1.0 / (1.0 + np.exp(-out["continue_logit"].cpu().numpy()))
    cont_bce = float(np.mean(-(cont * np.log(np.clip(cont_prob, 1e-8, 1.0)) + (1.0 - cont) * np.log(np.clip(1.0 - cont_prob, 1e-8, 1.0)))))

    print("informed_eval", {
        "pose_mse": pose_mse,
        "pose_delta_mse": pd_mse,
        "reward_mse": rw_mse,
        "continue_bce": cont_bce,
    })


if __name__ == "__main__":
    main()
