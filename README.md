# autoresearch-mlx

Apple Silicon (MLX) port of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Full credit to [@karpathy](https://github.com/karpathy) for the core idea: fixed-time autonomous research loops controlled through `program.md`. This port keeps the same basic rules: one mutable `train.py`, one metric (`val_bpb`), a fixed 5-minute training budget, and keep-or-revert via git. It runs natively on Apple Silicon through [MLX](https://github.com/ml-explore/mlx), so there is no PyTorch or CUDA dependency.

## Quick start

Requirements: Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies
uv sync

# one-time data + tokenizer prep
uv run prepare.py

# run one 5-minute training experiment
uv run train.py
```

Then point Claude Code or another coding agent at `program.md` and let it run the loop.

## Ternary SNN Search + Verilator

For SHD SNN experiments, ternary training/search is now available in the existing scripts.

```bash
# single ternary run
uv run train_snn.py \
	--weight-mode ternary \
	--ternary-threshold 0.08 \
	--ternary-scale-mode mean_abs \
	--ternary-use-ste

# architecture + hyperparameter search for ternary models
uv run search_snn_optuna.py \
	--trials 20 \
	--ternary-search \
	--ternary-threshold-min 0.02 \
	--ternary-threshold-max 0.35

# ternary search with a post-trial Verilator step
# placeholders: {trial_dir}, {trial_number}
uv run search_snn_optuna.py \
	--trials 10 \
	--ternary-search \
	--run-verilator \
	--verilator-command "verilator --version"

# ternary search with auto-generated RTL lint per trial
uv run search_snn_optuna.py \
	--trials 10 \
	--ternary-search \
	--run-verilator \
	--verilator-mode lint \
	--verilator-top ternary_trial_top \
	--verilator-max-width 512

# ternary search with auto-generated RTL compile + simulation per trial
# generates SystemVerilog + C++ testbench, builds with verilator -cc --exe --build,
# runs the binary and captures per-step output; result logged to JSONL as metrics.verilator
uv run search_snn_optuna.py \
	--trials 10 \
	--ternary-search \
	--run-verilator \
	--verilator-mode simulate \
	--verilator-top ternary_trial_top \
	--verilator-sim-width 32 \
	--verilator-sim-steps 16
```

Each Verilator run writes trial context metadata to `experiments/verilator/trial_XXXX/trial_context.json` before executing the Verilator command.

## 3D Render

[3D render][stereo event camera simulation]

Top row: stereo RGB (left camera, right camera). Second row: stereo event frames
(left events, right events), with positive polarity in red and negative polarity in blue.
Third row: per-camera stereo depth ground truth (left depth, right depth).
Fourth row: overall drone-centric depth ground truth (center camera) and disparity ground truth.
An IMU overlay shows body-frame acceleration and angular velocity for each frame.
The generator also supports real-world profile presets, including an OV9281-like
1MP global-shutter sensor model with lens distortion, vignetting, and sensor noise.

Ground-truth files used in this visualization:

| File | Meaning |
|---|---|
| `depth_left/*.npy` | Left camera depth in meters |
| `depth_right/*.npy` | Right camera depth in meters |
| `depth_gt/*.npy` | Center (drone-rig) depth in meters |
| `disparity_gt/*.npy` | Dense stereo disparity in pixels, computed as `fx * baseline_m / depth_left_m` |
| `meta/imu.csv` | Body-frame acceleration (`acc_b*`) and angular velocity (`gyro_b*`) |

<video controls width="640" preload="metadata">
	<source src="assets/stereo-event-camera-simulation-10s.mp4" type="video/mp4" />
	Your browser does not support the video tag.
</video>

## What matters

- `prepare.py` - data prep, tokenizer, dataloader, and evaluation. Treat as fixed.
- `train.py` - model, optimizer, and training loop. This is the file the agent edits.
- `program.md` - the autonomous experiment protocol.
- `results.tsv` - logged experiment history.

The loop is the same as upstream: edit `train.py`, run a fixed-budget experiment, read `val_bpb`, keep the change if it wins, revert if it loses, and repeat.

## Public baseline results

The public `results.tsv` captures the initial hardware-local walk from the default baseline down to `1.807902`:

| Commit | val_bpb | Status | Description |
|---|---:|---|---|
| `383abb4` | 2.667000 | keep | baseline (AdamW, default config) |
| `909dd59` | 2.588904 | keep | halve total batch size to `2^16` |
| `4161af3` | 2.533728 | keep | increase matrix LR to `0.04` |
| `5efc7aa` | 1.807902 | keep | reduce depth from `8` to `4` |

That result already shows the core Apple Silicon pattern: with a fixed 5-minute wall clock, smaller faster-training models can beat larger ones simply by fitting more optimizer steps into the budget.

## Longer Apple Silicon runs

Longer overnight runs on the working MLX port pushed much further. The long Mac Mini test is included here because it found a meaningfully different winner stack from the Max-class machines.

| Machine | Current best | Starting point | Repeated wins |
|---|---:|---:|---|
| M4 Max #1 | 1.294526 | 1.596971 | AdamW-only, low matrix LR, 3x MLP, no logit cap, moderate weight decay |
| M4 Max #2 | 1.330509 | 1.807902 | leaner batch, long anneal, SiLU, lower regularization, no logit cap |
| Mac Mini (long run) | 1.353329 | 1.922472 | Muon, sharper attention, smaller MLP, lower scalar LR |

The Mac Mini result matters because it did not just rediscover the same exact recipe. On smaller Apple Silicon hardware, the strongest changes leaned toward more aggressive step-efficiency wins. Later transfer tests showed some of those Mac Mini findings did not carry cleanly onto the Max baseline, which is exactly the kind of hardware-specific behavior this loop is useful for uncovering.

## Differences from upstream

- **MLX instead of PyTorch/CUDA.** Native Apple Silicon training with unified memory.
- **AdamW-only public path.** This public `train.py` keeps the default path simple. The long Mac Mini run above explored a Muon variant in the working port, but that branch is not exposed as a public default here.
- **Smaller eval token budget.** Reduced for faster iteration on Apple Silicon while keeping the same `evaluate_bpb` interface in `prepare.py`.
- **Roughly 6-7 minutes per experiment.** Expect 5 minutes of training plus compile and eval overhead.
- **MFU reporting is placeholder.** There is no Apple Silicon equivalent to the H100 FLOPs reference used upstream.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) - autoresearch and nanochat
- [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) - MLX GPT and optimizer reference
- [awni/picochat](https://github.com/awni/picochat) - MLX training patterns
- [Apple MLX team](https://github.com/ml-explore/mlx)

## License

MIT. See [LICENSE](LICENSE).

[stereo event camera simulation]: assets/stereo-event-camera-simulation-10s.mp4
