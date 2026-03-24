from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from world_model_experiments.informed_dreamer_model import InformedDreamerConfig, InformedFeatureDreamer


class InformedDataset(Dataset):
    def __init__(self, h5_path: str | Path, use_flight_plan: bool):
        with h5py.File(h5_path, "r") as h5:
            self.features = np.asarray(h5["features"], dtype=np.float32)
            actions = np.asarray(h5["actions"], dtype=np.float32)
            if use_flight_plan and "flight_plan" in h5:
                actions = np.concatenate([actions, np.asarray(h5["flight_plan"], dtype=np.float32)], axis=-1)
            self.actions = actions
            self.pose = np.asarray(h5["pose"], dtype=np.float32)
            self.pose_delta = np.asarray(h5["pose_delta"], dtype=np.float32)
            self.reward = np.asarray(h5["reward"], dtype=np.float32) if "reward" in h5 else None
            self.cont = np.asarray(h5["continue"], dtype=np.float32) if "continue" in h5 else None

        if self.reward is None:
            transl = np.linalg.norm(self.pose_delta[..., :3], axis=-1)
            yaw = np.abs(self.pose_delta[..., 5])
            self.reward = (0.8 * transl + 0.2 * yaw).astype(np.float32)

        if self.cont is None:
            self.cont = np.ones(self.reward.shape, dtype=np.float32)
            self.cont[:, -1] = 0.0

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.features[idx]),
            "actions": torch.from_numpy(self.actions[idx]),
            "pose": torch.from_numpy(self.pose[idx]),
            "pose_delta": torch.from_numpy(self.pose_delta[idx]),
            "reward": torch.from_numpy(self.reward[idx]),
            "continue": torch.from_numpy(self.cont[idx]),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train informed Dreamer-style model on TUMVIE-derived features")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="artifacts/tumvie/informed_dreamer")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--use-flight-plan", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def train() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = InformedDataset(args.dataset, use_flight_plan=args.use_flight_plan)
    n_val = max(1, int(0.1 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    sample = ds[0]
    cfg = InformedDreamerConfig(
        feature_dim=int(sample["features"].shape[-1]),
        action_dim=int(sample["actions"].shape[-1]),
        pose_dim=int(sample["pose"].shape[-1]),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InformedFeatureDreamer(cfg).to(device)
    opt_wm = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = out_dir / "informed_dreamer_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_wm = []
        train_ac = []
        for batch in train_loader:
            feat = batch["features"].to(device)
            act = batch["actions"].to(device)
            pose = batch["pose"].to(device)
            pose_delta = batch["pose_delta"].to(device)
            reward = batch["reward"].to(device)
            cont = batch["continue"].to(device)

            wm_loss = model.world_loss(feat, act, pose, pose_delta, reward, cont)
            ac_loss = model.actor_critic_loss(feat, act)
            total = wm_loss["loss"] + ac_loss["loss"]

            opt_wm.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_wm.step()

            train_wm.append(float(wm_loss["loss"].item()))
            train_ac.append(float(ac_loss["loss"].item()))

        model.eval()
        val_wm = []
        val_ac = []
        with torch.no_grad():
            for batch in val_loader:
                feat = batch["features"].to(device)
                act = batch["actions"].to(device)
                pose = batch["pose"].to(device)
                pose_delta = batch["pose_delta"].to(device)
                reward = batch["reward"].to(device)
                cont = batch["continue"].to(device)
                wm_loss = model.world_loss(feat, act, pose, pose_delta, reward, cont)
                ac_loss = model.actor_critic_loss(feat, act)
                val_wm.append(float(wm_loss["loss"].item()))
                val_ac.append(float(ac_loss["loss"].item()))

        tr_wm = float(np.mean(train_wm)) if train_wm else float("nan")
        tr_ac = float(np.mean(train_ac)) if train_ac else float("nan")
        va_wm = float(np.mean(val_wm)) if val_wm else float("nan")
        va_ac = float(np.mean(val_ac)) if val_ac else float("nan")
        print(f"epoch={epoch:03d} train_wm={tr_wm:.6f} train_ac={tr_ac:.6f} val_wm={va_wm:.6f} val_ac={va_ac:.6f}")

        ckpt = {
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "epoch": epoch,
            "val_wm": va_wm,
            "val_ac": va_ac,
        }
        torch.save(ckpt, out_dir / "informed_dreamer_last.pt")
        if va_wm + va_ac < best_val:
            best_val = va_wm + va_ac
            torch.save(ckpt, best_path)

    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    train()
