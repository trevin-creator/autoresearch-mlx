from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run informed Dreamer flight-plan ablation on the same dataset")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--output-root", type=str, default="artifacts/tumvie/informed_ablation")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--horizon", type=int, default=8)
    return p.parse_args()


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    print(proc.stdout, end="")
    return proc.stdout


def parse_metrics(output: str) -> dict[str, float]:
    for line in reversed(output.splitlines()):
        if line.startswith("informed_eval "):
            raw = line[len("informed_eval "):].strip()
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                return {str(k): float(v) for k, v in parsed.items()}
    raise RuntimeError("Could not parse informed_eval metrics from evaluator output")


def train_and_eval(
    dataset: str,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    embed_dim: int,
    hidden_dim: int,
    horizon: int,
    use_flight_plan: bool,
) -> dict[str, float]:
    py = sys.executable

    train_cmd = [
        py,
        "-m",
        "world_model_experiments.train_informed_dreamer",
        "--dataset",
        dataset,
        "--output-dir",
        str(out_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--embed-dim",
        str(embed_dim),
        "--hidden-dim",
        str(hidden_dim),
        "--horizon",
        str(horizon),
    ]
    if use_flight_plan:
        train_cmd.append("--use-flight-plan")

    eval_cmd = [
        py,
        "-m",
        "world_model_experiments.evaluate_informed_dreamer",
        "--dataset",
        dataset,
        "--checkpoint",
        str(out_dir / "informed_dreamer_best.pt"),
    ]
    if use_flight_plan:
        eval_cmd.append("--use-flight-plan")

    run_cmd(train_cmd)
    eval_out = run_cmd(eval_cmd)
    return parse_metrics(eval_out)


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    fp_dir = out_root / "with_flight_plan"
    nofp_dir = out_root / "without_flight_plan"

    print("=== Training/evaluating WITH flight-plan conditioning ===")
    fp = train_and_eval(
        dataset=args.dataset,
        out_dir=fp_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        use_flight_plan=True,
    )

    print("=== Training/evaluating WITHOUT flight-plan conditioning ===")
    nofp = train_and_eval(
        dataset=args.dataset,
        out_dir=nofp_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        use_flight_plan=False,
    )

    print("metric,with_flight_plan,without_flight_plan,delta(without-with)")
    for key in sorted(fp.keys()):
        delta = nofp[key] - fp[key]
        print(f"{key},{fp[key]:.9f},{nofp[key]:.9f},{delta:.9f}")


if __name__ == "__main__":
    main()
