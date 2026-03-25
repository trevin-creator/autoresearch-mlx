from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from world_model_experiments.lewm_feature_model import FeatureJEPA, FeatureJepaOnnxWrapper, FeatureLeWmConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export FeatureJEPA predictor to ONNX")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="artifacts/feature_lewm/feature_lewm.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--seq-len", type=int, default=8)
    return p.parse_args()


def export() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    cfg = FeatureLeWmConfig(**ckpt["config"])
    model = FeatureJEPA(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    wrapper = FeatureJepaOnnxWrapper(model).eval()

    feat = torch.randn(1, args.seq_len, cfg.feature_dim, dtype=torch.float32)
    act = torch.randn(1, args.seq_len, cfg.action_dim, dtype=torch.float32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (feat, act),
        str(output),
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

    print(f"exported ONNX: {output}")


if __name__ == "__main__":
    export()
