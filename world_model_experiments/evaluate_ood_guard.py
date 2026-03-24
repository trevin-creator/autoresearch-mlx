from __future__ import annotations

import argparse

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate feature-distribution OOD guard and fallback trigger rate")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--z-threshold", type=float, default=4.0)
    p.add_argument("--max-ood-rate", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with h5py.File(args.dataset, "r") as h5:
        if "features" not in h5:
            raise ValueError("dataset has no features key")
        feat = np.asarray(h5["features"], dtype=np.float64)

    if feat.ndim != 3:
        raise ValueError(f"features should be [N,T,D], got {feat.shape}")

    flat = feat.reshape(-1, feat.shape[-1])
    mu = np.mean(flat, axis=0, keepdims=True)
    sigma = np.std(flat, axis=0, keepdims=True) + 1e-6
    z = np.abs((flat - mu) / sigma)

    frame_ood = np.max(z, axis=1) > args.z_threshold
    seq_ood = frame_ood.reshape(feat.shape[0], feat.shape[1]).any(axis=1)

    frame_ood_rate = float(np.mean(np.asarray(frame_ood, dtype=np.float64)))
    seq_ood_rate = float(np.mean(np.asarray(seq_ood, dtype=np.float64)))

    result = {
        "num_sequences": float(feat.shape[0]),
        "frame_ood_rate": frame_ood_rate,
        "sequence_ood_rate": seq_ood_rate,
        "z_threshold": float(args.z_threshold),
        "fallback_trigger_rate": seq_ood_rate,
        "pass": float(frame_ood_rate <= args.max_ood_rate),
    }
    print("ood_guard", result)


if __name__ == "__main__":
    main()
