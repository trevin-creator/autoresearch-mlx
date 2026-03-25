from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from world_model_experiments._errors import ERR_NO_MOTOR_COMMANDS
from world_model_experiments._io import load_sequence_dataset
from world_model_experiments.lewm_feature_model import FeatureJEPA, FeatureJepaOnnxWrapper, FeatureLeWmConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate ONNX parity for motor-mode JEPA predictor")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--output", type=str, default="/tmp/feature_lewm_motor.onnx")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--use-motor-commands", action="store_true")
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = FeatureLeWmConfig(**ckpt["config"])
    model = FeatureJEPA(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = load_sequence_dataset(args.dataset)
    features = np.asarray(dataset["features"][args.sample_idx : args.sample_idx + 1], dtype=np.float32)
    if args.use_motor_commands:
        if "motor_commands" not in dataset:
            raise ValueError(ERR_NO_MOTOR_COMMANDS)
        actions = np.asarray(dataset["motor_commands"][args.sample_idx : args.sample_idx + 1], dtype=np.float32)
    else:
        actions = np.asarray(dataset["actions"][args.sample_idx : args.sample_idx + 1], dtype=np.float32)
        if "flight_plan" in dataset and actions.shape[-1] != cfg.action_dim:
            fp = np.asarray(dataset["flight_plan"][args.sample_idx : args.sample_idx + 1], dtype=np.float32)
            actions = np.concatenate([actions, fp], axis=-1)

    feat_t = torch.from_numpy(features)
    act_t = torch.from_numpy(actions)
    with torch.no_grad():
        enc = model.encode(feat_t, act_t)
        pt_out = model.predict(enc["emb"], enc["act_emb"]).cpu().numpy()

    wrapper = FeatureJepaOnnxWrapper(model).eval()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (feat_t, act_t),
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
    ort_out = sess.run(None, {"features": features, "actions": actions})[0]

    diff = np.abs(pt_out - ort_out)
    result = {
        "shape_match": float(pt_out.shape == ort_out.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose_1e-4": float(np.allclose(pt_out, ort_out, atol=1e-4, rtol=1e-4)),
    }
    print("onnx_parity", result)


if __name__ == "__main__":
    main()
