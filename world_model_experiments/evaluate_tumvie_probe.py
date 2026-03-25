from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from world_model_experiments._io import load_sequence_dataset
from world_model_experiments.lewm_feature_model import FeatureJEPA, FeatureLeWmConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TUMVIE embeddings with a linear pose-delta probe")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--use-flight-plan", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    dataset = load_sequence_dataset(path)
    features = np.asarray(dataset["features"], dtype=np.float32)
    actions = np.asarray(dataset["actions"], dtype=np.float32)
    pose_delta = np.asarray(dataset["pose_delta"], dtype=np.float32)
    flight_plan = np.asarray(dataset["flight_plan"], dtype=np.float32) if "flight_plan" in dataset else None
    return features, actions, pose_delta, flight_plan


def _encode_embeddings(model: FeatureJEPA, features: np.ndarray, actions: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        feat = torch.from_numpy(features)
        act = torch.from_numpy(actions)
        emb = model.encode(feat, act)["emb"]
    return emb.cpu().numpy().astype(np.float32)


def _ridge_probe(x_train: np.ndarray, y_train: np.ndarray, ridge: float) -> np.ndarray:
    x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=x_train.dtype)], axis=1)
    eye = np.eye(x_aug.shape[1], dtype=x_aug.dtype)
    eye[-1, -1] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + ridge * eye, x_aug.T @ y_train)


def _predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ weights


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = FeatureJEPA(FeatureLeWmConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    features, actions, pose_delta, flight_plan = _load_dataset(args.dataset)
    if args.use_flight_plan:
        if flight_plan is None:
            raise ValueError("--use-flight-plan set but dataset has no flight_plan key")  # noqa: TRY003
        actions = np.concatenate([actions, flight_plan], axis=-1)
    emb = _encode_embeddings(model, features, actions)

    x = emb.reshape(-1, emb.shape[-1])
    y = pose_delta.reshape(-1, pose_delta.shape[-1])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    split = max(1, int(args.train_frac * len(indices)))
    train_idx = indices[:split]
    val_idx = indices[split:]
    if val_idx.size == 0:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    weights = _ridge_probe(x[train_idx], y[train_idx], args.ridge)
    pred = _predict(x[val_idx], weights)

    mse = np.mean((pred - y[val_idx]) ** 2, axis=0)
    denom = np.sum((y[val_idx] - np.mean(y[val_idx], axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - np.sum((pred - y[val_idx]) ** 2, axis=0) / np.maximum(denom, 1e-8)

    names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]
    print("linear_probe_mse", {name: float(val) for name, val in zip(names, mse, strict=False)})
    print("linear_probe_r2", {name: float(val) for name, val in zip(names, r2, strict=False)})
    print("linear_probe_mean_mse", float(np.mean(mse)))


if __name__ == "__main__":
    main()
