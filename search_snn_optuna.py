"""Optuna-driven structured search over train_snn.py experiment space."""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import optuna

from train_snn import ExperimentConfig, append_jsonl, get_git_commit_short, train_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna search for SHD SNN")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--study-name", type=str, default="snn-search")
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///experiments/snn_optuna.db",
        help="Optuna storage URL",
    )
    parser.add_argument(
        "--time-budget-s",
        type=float,
        default=float(ExperimentConfig.time_budget_s),
        help="Per-trial training budget in seconds.",
    )
    parser.add_argument(
        "--log-jsonl",
        type=Path,
        default=Path("experiments/snn_optuna_trials.jsonl"),
    )
    parser.add_argument(
        "--ternary-search",
        action="store_true",
        help="Enable ternary mode and search ternary hyperparameters.",
    )
    parser.add_argument(
        "--ternary-threshold-min",
        type=float,
        default=0.02,
        help="Lower bound for ternary threshold search.",
    )
    parser.add_argument(
        "--ternary-threshold-max",
        type=float,
        default=0.35,
        help="Upper bound for ternary threshold search.",
    )
    parser.add_argument(
        "--run-verilator",
        action="store_true",
        help="Run a Verilator command after each ternary trial.",
    )
    parser.add_argument(
        "--verilator-command",
        type=str,
        default="verilator --version",
        help=(
            "Verilator command to execute. Supports {trial_dir} and {trial_number} "
            "format placeholders."
        ),
    )
    parser.add_argument(
        "--verilator-timeout-s",
        type=int,
        default=120,
        help="Timeout for each Verilator run.",
    )
    return parser.parse_args()


def trial_to_payload(
    trial: optuna.trial.Trial,
    cfg: ExperimentConfig,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit_short(),
        "study_name": trial.study.study_name,
        "trial_number": trial.number,
        "trial_params": trial.params,
        "config": dataclasses.asdict(cfg),
        "metrics": metrics,
    }


def build_config(
    trial: optuna.trial.Trial,
    time_budget_s: float,
    *,
    ternary_search: bool,
    ternary_threshold_min: float,
    ternary_threshold_max: float,
) -> ExperimentConfig:
    if ternary_search:
        ternary_threshold = trial.suggest_float(
            "ternary_threshold",
            min(ternary_threshold_min, ternary_threshold_max),
            max(ternary_threshold_min, ternary_threshold_max),
        )
        ternary_scale_mode = trial.suggest_categorical(
            "ternary_scale_mode", ["mean_abs", "max_abs"]
        )
        ternary_use_ste = trial.suggest_categorical("ternary_use_ste", [True, False])
        weight_mode = "ternary"
    else:
        ternary_threshold = ExperimentConfig.ternary_threshold
        ternary_scale_mode = ExperimentConfig.ternary_scale_mode
        ternary_use_ste = ExperimentConfig.ternary_use_ste
        weight_mode = "float"

    return ExperimentConfig(
        n_hidden=trial.suggest_int("n_hidden", 96, 320, step=32),
        n_layers=trial.suggest_int("n_layers", 1, 4),
        batch_size=trial.suggest_categorical("batch_size", [32, 48, 64, 96]),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.2),
        warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.2),
        final_lr_frac=trial.suggest_float("final_lr_frac", 0.0, 0.2),
        weight_mode=weight_mode,
        ternary_threshold=ternary_threshold,
        ternary_scale_mode=ternary_scale_mode,
        ternary_use_ste=ternary_use_ste,
        seed=42 + trial.number,
        time_budget_s=time_budget_s,
    )


def run_verilator_step(
    trial: optuna.trial.Trial,
    cfg: ExperimentConfig,
    metrics: dict[str, Any],
    command_template: str,
    timeout_s: int,
) -> dict[str, Any]:
    trial_dir = Path("experiments") / "verilator" / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "trial_number": trial.number,
        "trial_params": trial.params,
        "config": dataclasses.asdict(cfg),
        "metrics": metrics,
    }
    with (trial_dir / "trial_context.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    command = command_template.format(
        trial_dir=str(trial_dir),
        trial_number=trial.number,
    )

    # Command comes from trusted local CLI input for researcher-controlled flows.
    result = subprocess.run(  # noqa: S603
        shlex.split(command),
        capture_output=True,
        text=True,
        timeout=max(1, timeout_s),
        check=False,
    )

    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "passed": result.returncode == 0,
        "trial_dir": str(trial_dir),
    }


def main() -> None:
    args = parse_args()

    Path("experiments").mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        cfg = build_config(
            trial,
            args.time_budget_s,
            ternary_search=args.ternary_search,
            ternary_threshold_min=args.ternary_threshold_min,
            ternary_threshold_max=args.ternary_threshold_max,
        )
        metrics = train_once(cfg)

        if args.ternary_search and args.run_verilator:
            verilator = run_verilator_step(
                trial,
                cfg,
                metrics,
                command_template=args.verilator_command,
                timeout_s=args.verilator_timeout_s,
            )
            metrics["verilator"] = verilator
            trial.set_user_attr("verilator_passed", bool(verilator["passed"]))

        append_jsonl(args.log_jsonl, trial_to_payload(trial, cfg, metrics))
        trial.set_user_attr("metrics", json.dumps(metrics, sort_keys=True))
        return float(metrics["val_acc"])

    study.optimize(objective, n_trials=args.trials)

    best = study.best_trial
    print("---")
    print(f"best_trial: {best.number}")
    print(f"best_val_acc: {best.value:.6f}")
    print(f"best_params: {best.params}")


if __name__ == "__main__":
    main()
