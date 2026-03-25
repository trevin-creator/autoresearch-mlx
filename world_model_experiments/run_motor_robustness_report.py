from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

from world_model_experiments._errors import ERR_SEEDS_EMPTY
from world_model_experiments.experiment_tracking import (
    configure_mlflow,
    log_dataset_record,
    log_flat_params,
    resolve_dataset_record,
)

ERR_PARSE_ROBUST_EVAL = "Could not parse robust_eval output"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run disturbance robustness report from per-seed checkpoints")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--checkpoint-root", type=str, required=True)
    p.add_argument("--output-root", type=str, default="artifacts/sim/motor_robustness")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--scenario-mode", type=str, choices=["preset", "matrix"], default="preset")
    p.add_argument("--wind-stds", type=str, default="0.0,0.5,1.0")
    p.add_argument("--act-noise-stds", type=str, default="0.0,0.05,0.12")
    p.add_argument("--latency-steps", type=str, default="0,1,2")
    p.add_argument("--mlflow-experiment", type=str, default=None)
    p.add_argument("--mlflow-tracking-uri", type=str, default="artifacts/mlruns")
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    print(proc.stdout, end="")
    return proc.stdout


def parse_robust_eval(output: str) -> dict[str, dict[str, float]]:
    for line in reversed(output.splitlines()):
        if line.startswith("robust_eval "):
            payload = line[len("robust_eval ") :].strip()
            parsed = json.loads(payload)
            out: dict[str, dict[str, float]] = {}
            for scenario, metrics in parsed.items():
                out[str(scenario)] = {str(k): float(v) for k, v in metrics.items()}
            return out
    raise RuntimeError(ERR_PARSE_ROBUST_EVAL)


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError(ERR_SEEDS_EMPTY)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    dataset_record = resolve_dataset_record(args.dataset)
    mlflow = configure_mlflow(args.mlflow_experiment, args.mlflow_tracking_uri) if args.mlflow_experiment else None

    py = sys.executable
    per_seed: dict[int, dict[str, dict[str, float]]] = {}
    parent_context = mlflow.start_run(run_name=args.run_name or out_root.name) if mlflow is not None else nullcontext()
    with parent_context:
        if mlflow is not None:
            log_dataset_record(mlflow, dataset_record)
            log_flat_params(
                mlflow,
                {
                    "dataset": args.dataset,
                    "checkpoint_root": args.checkpoint_root,
                    "output_root": str(out_root),
                    "seeds": args.seeds,
                    "episodes": args.episodes,
                    "horizon": args.horizon,
                    "scenario_mode": args.scenario_mode,
                    "wind_stds": args.wind_stds,
                    "act_noise_stds": args.act_noise_stds,
                    "latency_steps": args.latency_steps,
                },
            )

        for seed in seeds:
            ckpt = Path(args.checkpoint_root) / f"seed_{seed}" / "informed_dreamer_best.pt"
            cmd: list[str] = [
                py,
                "-m",
                "world_model_experiments.evaluate_motor_robustness",
                "--dataset",
                args.dataset,
                "--checkpoint",
                str(ckpt),
                "--episodes",
                str(args.episodes),
                "--horizon",
                str(args.horizon),
                "--use-motor-commands",
                "--seed",
                str(seed),
                "--scenario-mode",
                args.scenario_mode,
                "--wind-stds",
                args.wind_stds,
                "--act-noise-stds",
                args.act_noise_stds,
                "--latency-steps",
                args.latency_steps,
            ]
            out = run_cmd(cmd)
            parsed = parse_robust_eval(out)
            per_seed[seed] = parsed
            if mlflow is not None:
                with mlflow.start_run(run_name=f"seed_{seed}", nested=True):
                    log_dataset_record(mlflow, dataset_record)
                    mlflow.log_param("seed", seed)
                    mlflow.log_param("checkpoint", str(ckpt))
                    for scenario_name, metrics in parsed.items():
                        mlflow.log_metrics({f"{scenario_name}_{k}": v for k, v in metrics.items()})

    scenarios = sorted(next(iter(per_seed.values())).keys())
    metric_keys = sorted(next(iter(next(iter(per_seed.values())).values())).keys())

    rows: list[list[str]] = []
    for seed in seeds:
        for scenario in scenarios:
            rows.append([str(seed), scenario, *[f"{per_seed[seed][scenario][k]:.9f}" for k in metric_keys]])

    csv_path = out_root / "motor_robustness_by_seed.csv"
    write_csv(csv_path, ["seed", "scenario", *metric_keys], rows)

    report = out_root / "motor_robustness_report.md"
    lines = [
        "# Motor Robustness Report",
        "",
        f"- dataset: {args.dataset}",
        f"- checkpoint_root: {args.checkpoint_root}",
        f"- seeds: {','.join(str(s) for s in seeds)}",
        "",
        "## Scenario Means Across Seeds",
    ]

    for scenario in scenarios:
        lines.append(f"### {scenario}")
        for k in metric_keys:
            vals = [per_seed[s][scenario][k] for s in seeds]
            lines.append(f"- {k}: mean={statistics.mean(vals):.9f}, std={statistics.pstdev(vals):.9f}")
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "- motor_robustness_by_seed.csv",
        ]
    )
    report.write_text("\n".join(lines) + "\n")

    if mlflow is not None:
        for scenario in scenarios:
            for k in metric_keys:
                vals = [per_seed[s][scenario][k] for s in seeds]
                mlflow.log_metric(f"{scenario}_mean_{k}", statistics.mean(vals))
                mlflow.log_metric(f"{scenario}_std_{k}", statistics.pstdev(vals))
        mlflow.log_artifact(str(csv_path), artifact_path="reports")
        mlflow.log_artifact(str(report), artifact_path="reports")

    print(f"wrote: {csv_path}")
    print(f"wrote: {report}")


if __name__ == "__main__":
    main()
