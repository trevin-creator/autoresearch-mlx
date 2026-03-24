from __future__ import annotations

import argparse
import ast
import csv
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-seed informed Dreamer flight-plan ablation and write report artifacts")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--output-root", type=str, default="artifacts/tumvie/informed_multiseed")
    p.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds, e.g. 0,1,2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--horizon", type=int, default=6)
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
    seed: int,
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
        "--seed",
        str(seed),
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


def write_seed_metrics_csv(rows: list[dict[str, str | float]], metrics_path: Path, metric_keys: list[str]) -> None:
    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "variant", *metric_keys])
        for row in rows:
            writer.writerow([row["seed"], row["variant"], *[f"{float(row[k]):.9f}" for k in metric_keys]])


def write_delta_summary_csv(summary: list[dict[str, float | str]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean_delta_no_fp_minus_with_fp", "std_delta"])
        for row in summary:
            writer.writerow([row["metric"], f"{float(row['mean_delta']):.9f}", f"{float(row['std_delta']):.9f}"])


def write_markdown_report(
    report_path: Path,
    dataset: str,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    embed_dim: int,
    hidden_dim: int,
    horizon: int,
    summary: list[dict[str, float | str]],
) -> None:
    lines = [
        "# Informed Dreamer Multiseed Ablation Report",
        "",
        "## Run Configuration",
        f"- dataset: {dataset}",
        f"- seeds: {','.join(str(s) for s in seeds)}",
        f"- epochs: {epochs}",
        f"- batch_size: {batch_size}",
        f"- embed_dim: {embed_dim}",
        f"- hidden_dim: {hidden_dim}",
        f"- horizon: {horizon}",
        "",
        "## Aggregate Delta (no_fp - with_fp)",
        "| metric | mean_delta | std_delta |",
        "|---|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['metric']} | {float(row['mean_delta']):.9f} | {float(row['std_delta']):.9f} |")

    lines.extend([
        "",
        "## Artifact Files",
        "- seed_metrics.csv",
        "- delta_summary.csv",
    ])

    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("At least one seed must be provided")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results: dict[tuple[int, str], dict[str, float]] = {}
    variants = [("with_fp", True), ("no_fp", False)]

    for seed in seeds:
        for name, use_fp in variants:
            out_dir = out_root / f"seed_{seed}" / name
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"=== seed={seed} variant={name} ===")
            results[(seed, name)] = train_and_eval(
                dataset=args.dataset,
                out_dir=out_dir,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                horizon=args.horizon,
                use_flight_plan=use_fp,
            )

    metric_keys = sorted(next(iter(results.values())).keys())

    seed_rows: list[dict[str, str | float]] = []
    for seed in seeds:
        for name, _ in variants:
            row: dict[str, str | float] = {"seed": str(seed), "variant": name}
            for key in metric_keys:
                row[key] = results[(seed, name)][key]
            seed_rows.append(row)

    summary: list[dict[str, float | str]] = []
    print("metric,mean_delta(no_fp-with_fp),std_delta")
    for key in metric_keys:
        deltas = [results[(seed, "no_fp")][key] - results[(seed, "with_fp")][key] for seed in seeds]
        mean_delta = statistics.mean(deltas)
        std_delta = statistics.pstdev(deltas)
        print(f"{key},{mean_delta:.9f},{std_delta:.9f}")
        summary.append({"metric": key, "mean_delta": mean_delta, "std_delta": std_delta})

    seed_metrics_path = out_root / "seed_metrics.csv"
    delta_summary_path = out_root / "delta_summary.csv"
    report_path = out_root / "report.md"

    write_seed_metrics_csv(seed_rows, seed_metrics_path, metric_keys)
    write_delta_summary_csv(summary, delta_summary_path)
    write_markdown_report(
        report_path=report_path,
        dataset=args.dataset,
        seeds=seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        summary=summary,
    )

    print(f"wrote: {seed_metrics_path}")
    print(f"wrote: {delta_summary_path}")
    print(f"wrote: {report_path}")


if __name__ == "__main__":
    main()
