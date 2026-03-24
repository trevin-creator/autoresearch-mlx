from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import onnxruntime as ort
import torch

from world_model_experiments.lewm_feature_model import FeatureJEPA, FeatureJepaOnnxWrapper, FeatureLeWmConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stress benchmark for ONNX runtime latency/jitter and parity")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--output", type=str, default="/tmp/feature_lewm_motor_runtime.onnx")
    p.add_argument("--batches", type=int, default=16)
    p.add_argument("--use-motor-commands", action="store_true")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--warmup-runs", type=int, default=5)
    p.add_argument("--timed-runs", type=int, default=20)
    p.add_argument("--csv-output", type=str, default="")
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def _load_data(path: str, use_motor_commands: bool, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        feat = np.asarray(h5["features"], dtype=np.float32)
        if use_motor_commands:
            if "motor_commands" not in h5:
                raise ValueError("--use-motor-commands set but dataset has no motor_commands key")
            act = np.asarray(h5["motor_commands"], dtype=np.float32)
        else:
            act = np.asarray(h5["actions"], dtype=np.float32)
            if "flight_plan" in h5 and act.shape[-1] != action_dim:
                fp = np.asarray(h5["flight_plan"], dtype=np.float32)
                act = np.concatenate([act, fp], axis=-1)
    return feat, act


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = FeatureLeWmConfig(**ckpt["config"])
    model = FeatureJEPA(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    features, actions = _load_data(args.dataset, args.use_motor_commands, cfg.action_dim)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample_feat = torch.from_numpy(features[0:1])
    sample_act = torch.from_numpy(actions[0:1])

    torch.onnx.export(
        FeatureJepaOnnxWrapper(model).eval(),
        (sample_feat, sample_act),
        str(out_path),
        dynamo=False,
        input_names=["features", "actions"],
        output_names=["predicted_embeddings"],
        dynamic_axes={
            "features": {0: "batch", 1: "time"},
            "actions": {0: "batch", 1: "time"},
            "predicted_embeddings": {0: "batch", 1: "time"},
        },
        opset_version=args.opset,
    )

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

    n = min(args.batches, features.shape[0])
    lat_ms: list[float] = []
    max_abs: list[float] = []
    mean_abs: list[float] = []

    for i in range(n):
        feat_i = features[i : i + 1]
        act_i = actions[i : i + 1]

        with torch.no_grad():
            feat_t = torch.from_numpy(feat_i)
            act_t = torch.from_numpy(act_i)
            enc = model.encode(feat_t, act_t)
            pt = model.predict(enc["emb"], enc["act_emb"]).cpu().numpy()

        for _ in range(max(0, args.warmup_runs)):
            sess.run(None, {"features": feat_i, "actions": act_i})

        timed = max(1, args.timed_runs)
        t0 = time.perf_counter()
        ort_raw = None
        for _ in range(timed):
            ort_raw = sess.run(None, {"features": feat_i, "actions": act_i})[0]
        t1 = time.perf_counter()
        ort_out = cast(np.ndarray[Any, Any], ort_raw)
        lat_ms.append(((t1 - t0) * 1000.0) / timed)

        d = np.abs(pt - ort_out)
        max_abs.append(float(d.max()))
        mean_abs.append(float(d.mean()))

    result = {
        "batches": float(n),
        "warmup_runs": float(max(0, args.warmup_runs)),
        "timed_runs": float(max(1, args.timed_runs)),
        "latency_ms_mean": float(np.mean(lat_ms)) if lat_ms else 0.0,
        "latency_ms_p95": float(np.percentile(lat_ms, 95)) if lat_ms else 0.0,
        "latency_ms_jitter_std": float(np.std(lat_ms)) if lat_ms else 0.0,
        "parity_max_abs_mean": float(np.mean(max_abs)) if max_abs else 0.0,
        "parity_mean_abs_mean": float(np.mean(mean_abs)) if mean_abs else 0.0,
        "parity_allclose_1e-4_rate": float(np.mean([x <= 1e-4 for x in max_abs])) if max_abs else 0.0,
    }

    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "tag",
            "batches",
            "warmup_runs",
            "timed_runs",
            "latency_ms_mean",
            "latency_ms_p95",
            "latency_ms_jitter_std",
            "parity_max_abs_mean",
            "parity_mean_abs_mean",
            "parity_allclose_1e-4_rate",
        ]
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            row = {k: result[k] for k in fieldnames if k != "tag"}
            row["tag"] = args.tag
            w.writerow(row)

    print("runtime_benchmark", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
