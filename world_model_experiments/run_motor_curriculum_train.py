from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import TypedDict

from world_model_experiments.experiment_tracking import (
    configure_mlflow,
    log_dataset_record,
    log_flat_params,
    resolve_dataset_record,
)


class CurriculumStage(TypedDict):
    name: str
    action_noise: float
    wind_std: float
    act_noise: float
    latency: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Disturbance curriculum training for motor-mode informed Dreamer")
    p.add_argument("--output-root", type=str, default="artifacts/sim/motor_curriculum")
    p.add_argument("--feature-dim", type=int, default=79)
    p.add_argument("--num-sequences", type=int, default=64)
    p.add_argument("--sequence-len", type=int, default=16)
    p.add_argument("--epochs-per-stage", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--mlflow-experiment", type=str, default=None)
    p.add_argument("--mlflow-tracking-uri", type=str, default="artifacts/mlruns")
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    print(proc.stdout, end="")
    return proc.stdout


def parse_prefixed_dict(output: str, prefix: str) -> dict[str, float]:
    for line in reversed(output.splitlines()):
        if line.startswith(prefix + " "):
            raw = line[len(prefix) + 1 :].strip()
            parsed = json.loads(raw.replace("'", '"')) if raw.startswith("{") and '"' not in raw else None
            if parsed is None:
                import ast

                parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                return {str(k): float(v) for k, v in parsed.items()}
    message = f"Could not parse {prefix} metrics"
    raise RuntimeError(message)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    py = sys.executable
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    mlflow = configure_mlflow(args.mlflow_experiment, args.mlflow_tracking_uri) if args.mlflow_experiment else None

    stages: list[CurriculumStage] = [
        {"name": "stage0_calm", "action_noise": 0.18, "wind_std": 0.0, "act_noise": 0.0, "latency": 0},
        {"name": "stage1_wind", "action_noise": 0.20, "wind_std": 0.5, "act_noise": 0.04, "latency": 0},
        {"name": "stage2_gust", "action_noise": 0.22, "wind_std": 1.0, "act_noise": 0.08, "latency": 1},
        {"name": "stage3_noisy", "action_noise": 0.24, "wind_std": 1.0, "act_noise": 0.12, "latency": 2},
    ]

    rows: list[dict[str, str]] = []
    parent_context = mlflow.start_run(run_name=args.run_name or out_root.name) if mlflow is not None else nullcontext()

    with parent_context:
        if mlflow is not None:
            log_flat_params(
                mlflow,
                {
                    "output_root": str(out_root),
                    "feature_dim": args.feature_dim,
                    "num_sequences": args.num_sequences,
                    "sequence_len": args.sequence_len,
                    "epochs_per_stage": args.epochs_per_stage,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "horizon": args.horizon,
                    "num_stages": len(stages),
                },
            )

        for i, st in enumerate(stages):
            stage_dir = out_root / st["name"]
            stage_dir.mkdir(parents=True, exist_ok=True)
            dataset = stage_dir / "sim_rollouts.parquet"

            build_cmd = [
                py,
                "-m",
                "world_model_experiments.build_sim_rollout_dataset",
                "--output",
                str(dataset),
                "--dataset-name",
                f"motor_curriculum_{st['name']}",
                "--num-sequences",
                str(args.num_sequences),
                "--sequence-len",
                str(args.sequence_len),
                "--feature-dim",
                str(args.feature_dim),
                "--action-noise",
                str(st["action_noise"]),
                "--wind-std",
                str(st["wind_std"]),
                "--actuation-noise-std",
                str(st["act_noise"]),
                "--latency-steps",
                str(st["latency"]),
                "--seed",
                str(args.seed + i),
            ]
            run_cmd(build_cmd)
            dataset_record = resolve_dataset_record(dataset)

            train_cmd = [
                py,
                "-m",
                "world_model_experiments.train_informed_dreamer",
                "--dataset",
                str(dataset),
                "--output-dir",
                str(stage_dir),
                "--epochs",
                str(args.epochs_per_stage),
                "--batch-size",
                str(args.batch_size),
                "--embed-dim",
                str(args.embed_dim),
                "--hidden-dim",
                str(args.hidden_dim),
                "--horizon",
                str(args.horizon),
                "--seed",
                str(args.seed + i),
                "--use-motor-commands",
            ]
            run_cmd(train_cmd)

            eval_cmd = [
                py,
                "-m",
                "world_model_experiments.evaluate_closed_loop_motor",
                "--dataset",
                str(dataset),
                "--checkpoint",
                str(stage_dir / "informed_dreamer_best.pt"),
                "--episodes",
                "8",
                "--horizon",
                str(args.horizon),
                "--use-motor-commands",
            ]
            eval_out = run_cmd(eval_cmd)
            eval_metrics = parse_prefixed_dict(eval_out, "closed_loop_eval")

            rows.append(
                {
                    "stage": st["name"],
                    "dataset": str(dataset),
                    "checkpoint": str(stage_dir / "informed_dreamer_best.pt"),
                    "wind_std": str(st["wind_std"]),
                    "actuation_noise_std": str(st["act_noise"]),
                    "latency_steps": str(st["latency"]),
                    "closed_loop_eval_tail": eval_out.strip().splitlines()[-1] if eval_out.strip() else "",
                }
            )
            if mlflow is not None:
                with mlflow.start_run(run_name=st["name"], nested=True):
                    if dataset_record is not None:
                        log_dataset_record(mlflow, dataset_record)
                    mlflow.log_params(
                        {
                            "stage": st["name"],
                            "stage_seed": args.seed + i,
                            "stage_wind_std": st["wind_std"],
                            "stage_act_noise": st["act_noise"],
                            "stage_latency": st["latency"],
                            "stage_dataset": str(dataset),
                        }
                    )
                    mlflow.log_metrics({f"closed_{k}": v for k, v in eval_metrics.items()})
                    mlflow.log_artifact(str(stage_dir / "informed_dreamer_best.pt"), artifact_path="checkpoints")

        write_csv(out_root / "curriculum_summary.csv", rows)
        if mlflow is not None:
            mlflow.log_artifact(str(out_root / "curriculum_summary.csv"), artifact_path="reports")
        print(f"wrote: {out_root / 'curriculum_summary.csv'}")


if __name__ == "__main__":
    main()
