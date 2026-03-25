from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import optuna

from world_model_experiments.experiment_tracking import (
    configure_mlflow,
    log_dataset_record,
    log_flat_params,
    resolve_dataset_record,
)

DEFAULT_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "embed_dim": {"type": "int", "low": 64, "high": 256, "step": 32},
    "hidden_dim": {"type": "int", "low": 128, "high": 512, "step": 64},
    "horizon": {"type": "int", "low": 4, "high": 12, "step": 2},
    "lr": {"type": "float", "low": 1e-4, "high": 3e-3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
    "batch_size": {"type": "categorical", "choices": [8, 16, 32]},
}

ERR_SEARCH_SPACE_OBJECT = "search-space JSON must be an object keyed by parameter name"
ERR_PARSE_INFORMED_EVAL = "Could not parse informed_eval metrics from evaluator output"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna architecture/hyperparameter search with nested MLflow tracking")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--output-root", type=str, default="artifacts/search/informed_optuna")
    p.add_argument("--study-name", type=str, default="informed_dreamer_search")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--timeout-s", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--metric", type=str, default="pose_delta_mse")
    p.add_argument("--direction", type=str, choices=["minimize", "maximize"], default="minimize")
    p.add_argument("--sampler", type=str, choices=["tpe", "random"], default="tpe")
    p.add_argument("--use-flight-plan", action="store_true")
    p.add_argument("--use-motor-commands", action="store_true")
    p.add_argument("--search-space-json", type=str, default=None)
    p.add_argument("--mlflow-experiment", type=str, default="world_model/optuna_informed")
    p.add_argument("--mlflow-tracking-uri", type=str, default="artifacts/mlruns")
    return p.parse_args()


def load_search_space(search_space_json: str | None) -> dict[str, dict[str, Any]]:
    if search_space_json is None:
        return dict(DEFAULT_SEARCH_SPACE)
    payload = json.loads(Path(search_space_json).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(ERR_SEARCH_SPACE_OBJECT)
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def suggest_params(trial: optuna.Trial, search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = str(spec.get("type", "")).lower()
        if kind == "int":
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
            continue
        if kind == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                step=None if "step" not in spec else float(spec["step"]),
                log=bool(spec.get("log", False)),
            )
            continue
        if kind == "categorical":
            choices = list(spec.get("choices", []))
            if not choices:
                message = f"categorical search param {name} must define non-empty choices"
                raise ValueError(message)
            params[name] = trial.suggest_categorical(name, choices)
            continue
        message = f"Unsupported search-space type for {name}: {kind}"
        raise ValueError(message)
    return params


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    print(proc.stdout, end="")
    return proc.stdout


def parse_eval_metrics(output: str) -> dict[str, float]:
    for line in reversed(output.splitlines()):
        if line.startswith("informed_eval "):
            raw = line[len("informed_eval ") :].strip()
            payload = ast.literal_eval(raw)
            if isinstance(payload, dict):
                return {str(k): float(v) for k, v in payload.items()}
    raise RuntimeError(ERR_PARSE_INFORMED_EVAL)


def build_study(args: argparse.Namespace) -> optuna.Study:
    if args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=args.seed)
    return optuna.create_study(direction=args.direction, sampler=sampler, study_name=args.study_name)


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    dataset_record = resolve_dataset_record(args.dataset)
    search_space = load_search_space(args.search_space_json)
    mlflow = configure_mlflow(args.mlflow_experiment, args.mlflow_tracking_uri) if args.mlflow_experiment else None

    study = build_study(args)
    parent_context = mlflow.start_run(run_name=args.study_name) if mlflow is not None else nullcontext()

    with parent_context:
        if mlflow is not None:
            log_dataset_record(mlflow, dataset_record)
            log_flat_params(
                mlflow,
                {
                    "dataset": args.dataset,
                    "study_name": args.study_name,
                    "n_trials": args.n_trials,
                    "timeout_s": args.timeout_s,
                    "epochs": args.epochs,
                    "metric": args.metric,
                    "direction": args.direction,
                    "sampler": args.sampler,
                    "use_flight_plan": args.use_flight_plan,
                    "use_motor_commands": args.use_motor_commands,
                    "search_space_json": "" if args.search_space_json is None else args.search_space_json,
                },
            )

        def objective(trial: optuna.Trial) -> float:
            sampled = suggest_params(trial, search_space)
            embed_dim = int(sampled["embed_dim"])
            hidden_dim = int(sampled["hidden_dim"])
            horizon = int(sampled["horizon"])
            lr = float(sampled["lr"])
            weight_decay = float(sampled["weight_decay"])
            batch_size = int(sampled["batch_size"])

            run_dir = out_root / f"trial_{trial.number:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            train_cmd = [
                sys.executable,
                "-m",
                "world_model_experiments.train_informed_dreamer",
                "--dataset",
                args.dataset,
                "--output-dir",
                str(run_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(batch_size),
                "--embed-dim",
                str(embed_dim),
                "--hidden-dim",
                str(hidden_dim),
                "--horizon",
                str(horizon),
                "--lr",
                str(lr),
                "--weight-decay",
                str(weight_decay),
                "--seed",
                str(args.seed + trial.number),
            ]
            if args.use_flight_plan:
                train_cmd.append("--use-flight-plan")
            if args.use_motor_commands:
                train_cmd.append("--use-motor-commands")

            eval_cmd = [
                sys.executable,
                "-m",
                "world_model_experiments.evaluate_informed_dreamer",
                "--dataset",
                args.dataset,
                "--checkpoint",
                str(run_dir / "informed_dreamer_best.pt"),
            ]
            if args.use_flight_plan:
                eval_cmd.append("--use-flight-plan")
            if args.use_motor_commands:
                eval_cmd.append("--use-motor-commands")

            trial_context = (
                mlflow.start_run(run_name=f"trial_{trial.number}", nested=True) if mlflow is not None else nullcontext()
            )
            with trial_context:
                if mlflow is not None:
                    log_dataset_record(mlflow, dataset_record)
                    log_flat_params(
                        mlflow,
                        {
                            "trial_number": trial.number,
                            "output_dir": str(run_dir),
                            **sampled,
                        },
                    )

                run_cmd(train_cmd)
                eval_output = run_cmd(eval_cmd)
                metrics = parse_eval_metrics(eval_output)
                metric_value = float(metrics[args.metric])

                if mlflow is not None:
                    mlflow.log_metrics(metrics)
                    mlflow.log_metric("objective", metric_value)
                    mlflow.log_artifact(str(run_dir / "informed_dreamer_best.pt"), artifact_path="checkpoints")

                trial.set_user_attr("metrics", metrics)
                return metric_value

        timeout = None if args.timeout_s <= 0 else args.timeout_s
        study.optimize(objective, n_trials=args.n_trials, timeout=timeout)

        best = {
            "study_name": args.study_name,
            "best_value": float(study.best_value),
            "best_trial": int(study.best_trial.number),
            "best_params": study.best_params,
            "metric": args.metric,
            "direction": args.direction,
        }
        summary_path = out_root / "optuna_best.json"
        summary_path.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
        space_path = out_root / "search_space_effective.json"
        space_path.write_text(json.dumps(search_space, indent=2) + "\n", encoding="utf-8")

        if mlflow is not None:
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_objective", float(study.best_value))
            mlflow.log_artifact(str(summary_path), artifact_path="reports")
            mlflow.log_artifact(str(space_path), artifact_path="reports")

        print("optuna_best", best)
        print(f"wrote: {summary_path}")
        print(f"wrote: {space_path}")


if __name__ == "__main__":
    main()
