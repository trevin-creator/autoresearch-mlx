from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from world_model_experiments._errors import (
    ERR_MOTOR_FP_EXCLUSIVE,
    ERR_NO_FLIGHT_PLAN,
    ERR_NO_MOTOR_COMMANDS,
    ERR_SEQ_COUNT_MISMATCH,
)
from world_model_experiments.lewm_feature_model import FeatureJEPA, FeatureLeWmConfig


class FeatureSequenceDataset(Dataset):
    def __init__(self, h5_path: str | Path, use_motor_commands: bool):
        with h5py.File(h5_path, "r") as h5:
            self.features = np.asarray(h5["features"], dtype=np.float32)
            if use_motor_commands:
                if "motor_commands" not in h5:
                    raise ValueError(ERR_NO_MOTOR_COMMANDS)
                self.actions = np.asarray(h5["motor_commands"], dtype=np.float32)
            else:
                self.actions = np.asarray(h5["actions"], dtype=np.float32)
            self.flight_plan = np.asarray(h5["flight_plan"], dtype=np.float32) if "flight_plan" in h5 else None

        if self.features.shape[0] != self.actions.shape[0]:
            raise ValueError(ERR_SEQ_COUNT_MISMATCH)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {
            "features": torch.from_numpy(self.features[idx]),
            "actions": torch.from_numpy(self.actions[idx]),
        }
        if self.flight_plan is not None:
            sample["flight_plan"] = torch.from_numpy(self.flight_plan[idx])
        return sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LeWM-style JEPA on Spyx feature vectors")
    p.add_argument("--dataset", type=str, required=True, help="HDF5 path with features/actions datasets")
    p.add_argument("--output-dir", type=str, default="artifacts/feature_lewm")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--history-size", type=int, default=8)
    p.add_argument("--num-preds", type=int, default=1)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--sigreg-weight", type=float, default=10.0)
    p.add_argument("--use-flight-plan", action="store_true", help="Concatenate flight_plan with actions")
    p.add_argument("--use-motor-commands", action="store_true", help="Use motor_commands key as action source")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def train() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.use_motor_commands and args.use_flight_plan:
        raise ValueError(ERR_MOTOR_FP_EXCLUSIVE)

    ds = FeatureSequenceDataset(args.dataset, use_motor_commands=args.use_motor_commands)
    n_val = max(1, int(0.1 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    sample = ds[0]
    feature_dim = int(sample["features"].shape[-1])
    action_dim = int(sample["actions"].shape[-1])
    if args.use_flight_plan:
        if "flight_plan" not in sample:
            raise ValueError(ERR_NO_FLIGHT_PLAN)
        action_dim += int(sample["flight_plan"].shape[-1])

    cfg = FeatureLeWmConfig(
        feature_dim=feature_dim,
        action_dim=action_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        history_size=args.history_size,
        num_preds=args.num_preds,
        depth=args.depth,
        heads=args.heads,
        sigreg_weight=args.sigreg_weight,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureJEPA(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_path = out_dir / "feature_lewm_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            feat = batch["features"].to(device)
            act = batch["actions"].to(device)
            if args.use_flight_plan:
                act = torch.cat([act, batch["flight_plan"].to(device)], dim=-1)
            losses = model.compute_loss(feat, act)
            loss = losses["loss"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                feat = batch["features"].to(device)
                act = batch["actions"].to(device)
                if args.use_flight_plan:
                    act = torch.cat([act, batch["flight_plan"].to(device)], dim=-1)
                loss = model.compute_loss(feat, act)["loss"]
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        ckpt = {
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(ckpt, out_dir / "feature_lewm_last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, best_path)

    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    train()
