"""Optuna-driven structured search over train_snn.py experiment space."""

from __future__ import annotations

import argparse
import dataclasses
import json
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


def build_config(trial: optuna.trial.Trial, time_budget_s: float) -> ExperimentConfig:
    return ExperimentConfig(
        n_hidden=trial.suggest_int("n_hidden", 96, 320, step=32),
        n_layers=trial.suggest_int("n_layers", 1, 4),
        batch_size=trial.suggest_categorical("batch_size", [32, 48, 64, 96]),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.2),
        warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.2),
        final_lr_frac=trial.suggest_float("final_lr_frac", 0.0, 0.2),
        seed=42 + trial.number,
        time_budget_s=time_budget_s,
    )


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
        cfg = build_config(trial, args.time_budget_s)
        metrics = train_once(cfg)
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
