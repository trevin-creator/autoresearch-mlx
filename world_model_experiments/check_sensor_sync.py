from __future__ import annotations

import argparse

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check timestamp consistency and sync quality in sequence datasets")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--max-jitter-us", type=float, default=5000.0)
    p.add_argument("--max-gap-us", type=float, default=50000.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with h5py.File(args.dataset, "r") as h5:
        if "timestamps_us" not in h5:
            raise ValueError("dataset has no timestamps_us key")
        ts = np.asarray(h5["timestamps_us"], dtype=np.float64)

    if ts.ndim != 2:
        raise ValueError(f"timestamps_us should be [N,T], got shape={ts.shape}")

    diffs = np.diff(ts, axis=1)
    monotonic_viol = np.sum(diffs <= 0)

    per_seq_med = np.median(diffs, axis=1, keepdims=True)
    jitter = np.abs(diffs - per_seq_med)

    jitter_p95 = float(np.percentile(jitter, 95)) if jitter.size else 0.0
    max_gap = float(np.max(diffs)) if diffs.size else 0.0
    min_gap = float(np.min(diffs)) if diffs.size else 0.0

    result = {
        "num_sequences": float(ts.shape[0]),
        "sequence_len": float(ts.shape[1]),
        "monotonic_violations": float(monotonic_viol),
        "jitter_p95_us": jitter_p95,
        "min_gap_us": min_gap,
        "max_gap_us": max_gap,
        "pass": float(
            (monotonic_viol == 0)
            and (jitter_p95 <= args.max_jitter_us)
            and (max_gap <= args.max_gap_us)
        ),
    }

    print("sensor_sync", result)


if __name__ == "__main__":
    main()
