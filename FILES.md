# What each file does, and how to check it works

A map of every file in `autoresearch-mlx/`. Each entry says what the file is for,
what it reads and writes, and a quick check you can run. Commands run from the
repo root.

## The workflow at a glance

```
prepare.py ──(one time)──> ~/.cache/autoresearch/  (data shards + tokenizer)
                                    │
program.md ──(instructions)──> AI agent edits train.py
                                    │
                            uv run train.py  ──> prints val_bpb etc. (run.log)
                                    │
                  rigor.py decides keep / discard ──> rigor_ledger.jsonl
                                    │
                            results.tsv  (experiment history)
```

---

## Core experiment

### `prepare.py`: one-time data and tokenizer setup (do not edit)
- **What it does:** downloads the training text shards (from Hugging Face,
  `karpathy/climbmix-400b-shuffle`) and trains a BPE tokenizer (vocab 8192).
  It also defines the fixed pieces every experiment shares: `MAX_SEQ_LEN = 2048`,
  `TIME_BUDGET = 300` seconds, the dataloader `make_dataloader`, and the scorer
  `evaluate_bpb` that turns a model into a `val_bpb` number.
- **Writes:** `~/.cache/autoresearch/data/` (parquet shards) and
  `~/.cache/autoresearch/tokenizer/`. Exits with an error if any shard fails to
  download. `token_bytes.npy` (each token's byte length, used by `val_bpb`) is
  rebuilt on its own if you delete it and re-run `prepare.py`. Caches built
  before the byte-length fix make `val_bpb` read about 0.13% low. Rebuild it
  (`rm ~/.cache/autoresearch/tokenizer/token_bytes.npy && uv run prepare.py`)
  and re-run the baseline, because old and new `val_bpb` numbers aren't
  comparable.
- **Check it works:**
  ```bash
  uv run prepare.py --num-shards 8
  ```
  ```bash
  ls ~/.cache/autoresearch/data ~/.cache/autoresearch/tokenizer
  ```
  You should see `shard_*.parquet` files and tokenizer files. The full download
  (without `--num-shards`) is large; 8 shards is enough to test.

### `train.py`: the experiment (the only file the agent edits)
- **What it does:** builds a small GPT model in MLX, trains it for 5 minutes of
  wall-clock time, then scores it with `evaluate_bpb`. The weight-init seed comes
  from the `SEED` env var (default 42) and is printed as `seed: N`. Model size, optimizer
  (AdamW), learning rates and batch size are constants near line 367.
- **Prints at the end** (this is what `rigor.py` reads):
  ```
  val_bpb:          2.225335   <- main score, lower is better
  training_seconds: 302.8
  total_seconds:    330.2
  peak_vram_mb:     8536.2     <- memory used
  mfu_percent:      0.00       <- placeholder on Apple Silicon
  total_tokens_M:   ...
  num_steps:        102
  num_params_M:     11.5
  depth:            4
  ```
- **Needs:** `prepare.py` has run first.
- **Check it works:**
  ```bash
  uv run train.py > run.log 2>&1; tail -9 run.log
  ```
  It takes about 6–7 minutes. It worked if the last lines show a `val_bpb:` value.

### `program.md`: instructions for the AI agent
- **What it does:** the prompt you give an AI coding agent so it runs the
  research loop by itself: create an `autoresearch/<tag>` branch, run a
  baseline, then repeatedly edit `train.py`, score it with `rigor.py --seeds 5`,
  and keep or revert based on rigor's verdict (not a single run), logging each
  result to `results.tsv`. Discarded rows are written after the `git reset`, so
  they aren't lost.
- **Check it works:** it isn't code. Read it and confirm the steps match how you
  want the agent to behave.

---

## Rigor: deciding keep/discard reliably

### `rigor.py`: significance-gated keep/discard (works for any experiment)
- **Why:** re-running the *same* `train.py` moves `val_bpb` by about 0.03. Keeping
  any run that happens to beat the best just chases noise. `rigor.py` runs each
  change several times and keeps it only if it is better **with high confidence**.
- **What it does on `run`:**
  1. Fingerprints the watched files (`--files`, default `train.py`). If that
     exact version was already scored, it skips it (never-repeat). Crashed
     versions can be retried.
  2. Runs the command (`--cmd`, default `uv run train.py`) up to `--seeds`
     times (default 3), setting `SEED=0, 1, 2, ...` for each run, and reads
     each `--metric` from the output.
  3. **Early discard:** if the first run is worse than the best by more than
     `--early-margin` (default 3× the best's run-to-run spread), it stops and
     discards without spending more runs.
  4. **Decision:** keep only if P(better on the first metric) ≥ `--confidence`
     (default 0.95) **and** it isn't confidently worse on any other metric
     (the guardrails). With 5+ seeds shared with the best it uses a paired
     sign-flip test (`test: paired`); otherwise the unpaired bootstrap
     (`test: unpaired`).
  5. Appends the result to the ledger (`--ledger`, default `rigor_ledger.jsonl`).
- **Commands:**
  ```
  uv run rigor.py run "what I changed"                  # score current train.py
  uv run rigor.py --metric val_bpb --metric peak_vram_mb run "..."   # + memory guardrail
  uv run rigor.py best                                  # current best version
  uv run rigor.py log                                   # every scored version
  ```
  Use `NAME:max` for metrics where higher is better, for example `throughput:max`.
- **Messages you will see:**

  | Message | Meaning |
  |---|---|
  | `BASELINE <hash>` | first version; becomes the best |
  | `KEEP <hash>` | confidently better, and no guardrail got worse; new best |
  | `DISCARD <hash> ... P(better)=0.43` | not confidently better (probably noise) |
  | `DISCARD <hash>: first run ... worse ... margin` | clear loser, stopped after one run |
  | `DISCARD ...; worse on peak_vram_mb` | better on the main metric but failed a guardrail |
  | `CRASH <hash>` | the run failed or didn't print every metric; can be retried |
  | `already scored <hash>` | identical version, skipped |
  | `guardrail X: best has no samples, not checked` | the best was scored before X was tracked |

- **Check it works** (no training needed): run `test_rigor.py`, below.

### `test_rigor.py`: self-check for `rigor.py`
- **What it does:** tests every decision rule with fake runs instead of real
  training, so it finishes in seconds. It covers the baseline, noise vs a clear
  loser, crash retry, never-repeat, higher-is-better, the automatic margin,
  guardrails, old ledger files, and metric parsing. It also runs one real
  subprocess to check that output parsing works end to end.
- **Check it works:**
  ```bash
  uv run test_rigor.py
  ```
  The last line should be `ok`. Any failure stops with an `AssertionError`.

### `rigor_ledger.jsonl`: rigor's memory (created on first `run`; git-ignored)
- One JSON line per scored version:
  `hash, desc, samples, mean, std, status, p_better`, plus `extra` (guardrail
  samples) and `regressed` (guardrails that got worse) when relevant.
- **Check:** `uv run rigor.py log` prints it readably.

---

## Results and logs

### `results.tsv`: experiment history
- Tab-separated: `commit  val_bpb  memory_gb  status  description`, one row per
  experiment the agent ran (keep/discard). The agent writes it while following
  `program.md`. `rigor.py` prints a matching `tsv:` line you can paste in.
- **Check:** `column -t -s $'\t' results.tsv`

### `run.log`: output of the last training run (git-ignored)
- What `uv run train.py > run.log 2>&1` produced. Handy for `tail run.log`.

---

## Project plumbing

| File | What it is | Check |
|---|---|---|
| `README.md` | Project overview, results, and the "Rigorous keep/discard" usage section | Read it |
| `pyproject.toml` | Python 3.10–3.13 project and dependencies (mlx, numpy, pyarrow, regex, requests, rustbpe, tiktoken) | `uv sync` finishes with no errors |
| `uv.lock` | Exact pinned dependency versions for `uv` (generated, don't hand-edit) | `uv sync --locked` |
| `.gitignore` | Keeps `.venv/`, `__pycache__/`, logs and `rigor_ledger.jsonl` out of git | `git status` stays clean after runs |
| `LICENSE` | MIT license | none needed |
| `.venv/`, `__pycache__/` | Local virtualenv and Python bytecode caches (generated) | Safe to delete; `uv sync` rebuilds `.venv/` |

---

## Full "does it all work?" checklist

1. Install: run `uv sync`. It should finish with no errors.
2. Data: run `uv run prepare.py --num-shards 8`. Shards and the tokenizer should appear in `~/.cache/autoresearch/`.
3. Rigor logic: run `uv run test_rigor.py`. It should print `ok`.
4. One real training run: run `uv run train.py`. It should print `val_bpb:`.
5. Rigor end to end: run `uv run rigor.py run "baseline"`. It should print `BASELINE <hash>` after 3 runs (about 20 minutes).
6. Make a small edit to `train.py`, then run `uv run rigor.py run "my change"`. It should print `KEEP` or `DISCARD`.
7. Run `uv run rigor.py log`. It should list both versions.
