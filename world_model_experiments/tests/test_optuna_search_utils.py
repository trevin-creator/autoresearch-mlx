from __future__ import annotations

import json

from world_model_experiments.run_optuna_search import load_search_space, parse_eval_metrics


def test_parse_eval_metrics_extracts_last_informed_eval() -> None:
    output = "\n".join(
        [
            "foo",
            "informed_eval {'pose_mse': 1.2, 'pose_delta_mse': 0.4, 'reward_mse': 0.8, 'continue_bce': 0.3}",
            "bar",
        ]
    )
    metrics = parse_eval_metrics(output)
    assert metrics["pose_mse"] == 1.2
    assert metrics["pose_delta_mse"] == 0.4


def test_load_search_space_from_json(tmp_path) -> None:
    payload = {
        "embed_dim": {"type": "int", "low": 64, "high": 128, "step": 32},
        "batch_size": {"type": "categorical", "choices": [8, 16]},
    }
    path = tmp_path / "space.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_search_space(str(path))
    assert loaded["embed_dim"]["high"] == 128
    assert loaded["batch_size"]["choices"] == [8, 16]
