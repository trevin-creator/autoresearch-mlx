# autoresearch-mlx

This is an Apple Silicon (MLX) port of Karpathy's autoresearch — an experiment to have the LLM do its own research. All training runs natively on MLX with unified memory. No PyTorch or CUDA required.

**Monorepo note:** This project may live inside a larger repo. Always stage only `autoresearch-mlx/` paths. Never use blind `git add -A`.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with header row and baseline entry. Run `uv run train.py` once to establish YOUR baseline on this hardware. Do NOT use baseline numbers from other platforms.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on Apple Silicon via MLX. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. MLX uses unified memory shared between CPU and GPU. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          2.534000
training_seconds: 312.4
total_seconds:    405.7
peak_vram_mb:     27528.9
mfu_percent:      0.00
total_tokens_M:   39.8
num_steps:        46
num_params_M:     50.3
depth:            8
```

Note that the script runs for a fixed 5-minute training budget. On Apple Silicon the throughput, step count, and absolute val_bpb will differ from NVIDIA results — that's expected. Compare only against your own baseline on the same hardware.

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
383abb4	2.667000	26.9	keep	baseline
909dd59	2.588904	26.9	keep	halve total batch size to 2^16
4161af3	2.533728	26.9	keep	increase matrix LR to 0.04
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

### Step 1: Pre-experiment reasoning (the checklist)

Before every experiment, complete this checklist. Do not skip it. Write your reasoning briefly (a few sentences is fine) before committing the change.

1. **Review strategy state**: Read `strategy/hypotheses.md` and pick the next experiment. If no hypothesis fits, generate a new one and add it before proceeding.
2. **Check interactions**: Consult `strategy/interactions.md`. If the change touches a parameter with known couplings, decide whether to adjust the coupled parameter too. If unsure, note the risk.
3. **Check near-misses**: Scan `strategy/near-misses.md`. Has the config changed enough that a near-miss deserves re-testing instead of a new idea?
4. **State your prediction**: Write down what you expect to happen and why (e.g., "Expect ~0.01 improvement because lower LR compensates for smaller batch noise"). This forces clear thinking and makes post-experiment analysis more useful.
5. **Single vs. bundle decision**: Decide whether this is a single-variable test or a deliberate multi-change bundle. If a bundle, justify why the changes should be tested together (e.g., "SwiGLU needs wider model to compensate param count — testing separately would be misleading").

### Step 2: Execute the experiment

1. Edit `train.py` with the change.
2. `git add train.py && git commit -m "experiment: <description>"` (never `git add -A`)
3. Run: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
4. Read results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
5. If grep is empty → crash. Run `tail -n 50 run.log` for the stack trace. Fix if trivial, skip if fundamental.

### Step 3: Post-experiment analysis and strategy update

After every experiment, whether it succeeded or failed:

1. **Record in results.tsv** (tab-separated, as described in Logging Results).
2. **Keep or discard**:
   - If val_bpb improved: `git add results.tsv && git commit --amend --no-edit`
   - If equal or worse: record the discard hash, then `git reset --hard <previous kept commit>`
3. **Update strategy files** (this is critical — do not skip):
   - **`strategy/learnings.md`**: Add what you learned. Focus on *why* it worked/failed, not just the number. Update confidence levels on existing entries if this experiment provides new evidence.
   - **`strategy/hypotheses.md`**: Move the tested hypothesis to "Tested / Resolved" with outcome. If the result suggests new hypotheses, add them.
   - **`strategy/near-misses.md`**: If the result was within ~0.01 of current best, add it. If config changes make an existing near-miss worth re-testing, note that.
   - **`strategy/interactions.md`**: If the experiment revealed or confirmed a parameter coupling, record it.
4. **Compare prediction to outcome**: Did the result match your prediction? If not, ask why. Surprises are the most valuable learning signal — a wrong prediction means your mental model needs updating.

### Experiment selection strategies

Beyond the basic greedy single-change search, use these strategies to explore the space more effectively:

**Synergy bundles**: When two changes have a theoretical reason to interact positively, test them together. If the bundle wins, you can optionally ablate later to understand which part mattered. If the bundle loses, the individual changes may still be worth testing alone.

**LR sweep sub-loop**: After any architecture change (depth, width, activation function), do a sequential LR binary search before declaring the architecture change a loss. Try 0.5x first, then 1.5x (or narrow based on the first result). This costs 2-3 extra runs (~15-20 minutes) but prevents false negatives from LR mismatch. Many architecture changes only fail because the LR wasn't re-tuned.

**Revisit near-misses**: After every 3-4 experiments, scan the near-misses list. If the config has shifted significantly since a near-miss was tested, re-test it. The same change can flip from loss to win in a different context.

**Controlled regressions**: If a change opens a known optimization pathway (e.g., SwiGLU enables different MLP ratios), accept a small regression (~0.005) and immediately test the follow-up. Mark this explicitly in results.tsv as "regression accepted: <reason>".

**Diminishing returns detection**: If the last 5+ experiments are all discards with results within 0.01 of the best, you've likely reached a local optimum at this architecture scale. Time to try something more radical (different depth, different optimizer, major architectural change).

### Single-machine constraint: no parallel experiments

All experiments run sequentially on a single Apple Silicon machine. Do NOT attempt to run multiple training jobs concurrently. The reasons:

1. **Incomparable results**: `TIME_BUDGET` is wall-clock time. Concurrent jobs share GPU compute and memory bandwidth, so each gets fewer steps in the same wall-clock window. Results would not be comparable to sequential baselines.
2. **Unfair scheduling**: MLX GPU scheduling across concurrent processes is not deterministic. One job may get 40% of compute while another gets 25%, making even relative comparisons between parallel runs unreliable.
3. **Memory pressure**: Training already uses ~21GB of unified memory. Multiple instances risk bandwidth contention and OOM.

The strategy knowledge base compensates for the lack of parallelism by making each sequential experiment pick smarter — consult hypotheses, interactions, and near-misses before every run rather than brute-forcing the search space.

If multi-machine access becomes available in the future, parallel triage (running 2-3 variants simultaneously for directional signal, then confirming the winner sequentially) becomes viable. Until then, invest thinking time, not compute time.

### Loop discipline

**Timeout**: Each experiment should take ~7 minutes total (5 min training + ~1 min compile/eval overhead on Apple Silicon). If a run exceeds 15 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — consult the literature, re-read the in-scope files, review strategy files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

## Literature consultation

You should actively consult recent ML research to source experiment ideas. This is not a replacement for your own reasoning — it supplements it.

**When to consult the literature:**
- At the start of a run, before your first non-baseline experiment, to build a prioritized list of ideas grounded in recent findings.
- When you've exhausted your current ideas or hit a plateau (e.g. 3+ consecutive discards).
- When an experiment produces a surprising result (good or bad) and you want to understand why.

**How to do it:**
1. Use web search to find recent papers from venues like ICML, NeurIPS, ICLR, and arXiv on topics relevant to your current line of experimentation (e.g. "efficient transformer training", "learning rate schedules small LLMs", "muon optimizer improvements").
2. When you find a promising paper, fetch its abstract and key results. If the full paper is available as a web page (e.g. on arXiv HTML or OpenReview), read the methodology section for implementation details.
3. Extract the **specific, actionable idea** — a concrete hyperparameter, architectural modification, or training trick — and translate it into a modification of `train.py`.
4. Save your notes on each paper to the `literature/` directory. Use one file per paper, named by arXiv ID or a short slug (e.g. `literature/2405.20233-grokfast.md`). Include the title, authors, venue, key findings, and how they might apply to this setup. This builds a persistent knowledge base across runs.

**Rules:**
- Do NOT spend more than ~2 minutes on any single literature search. You are an experimentalist, not a literature reviewer. Get the idea, try it, move on.
- Do NOT blindly copy findings. Papers train on different data, at different scales, with different budgets. Adapt the idea to this setup (Apple Silicon, 5-minute budget, ~50M params, BPB metric).
- DO log what paper inspired an experiment in the `description` column of results.tsv (e.g. "GrokFast EMA gradient filter (arXiv:2405.20233)").
- DO revisit the literature when you notice a pattern in your results that you don't fully understand — a paper may explain it.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~7 minutes then you can run approx 8-9/hour, for a total of about 70 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Strategy knowledge base

The `strategy/` directory is a living decision-support system. Unlike `literature/` (external research notes) and `results.tsv` (flat experiment log), `strategy/` contains curated, evolving analysis that you update after every experiment.

### Files

- **`strategy/learnings.md`** — What you've learned about *this specific hardware and config*. Not just "X failed" but "X failed because Y, which implies Z for future experiments." Each entry has a confidence level (low/medium/high) that gets updated as more evidence accumulates.

- **`strategy/hypotheses.md`** — Prioritized queue of untested experiment ideas with rationale. Includes both single-change experiments and deliberate multi-change bundles. Hypotheses move to a "Tested / Resolved" section after testing.

- **`strategy/near-misses.md`** — Experiments that were within noise margin (~0.01 bpb) of the best, or that failed in a config context that has since changed significantly. Each entry records the context it was tested in and what changes might make it worth revisiting.

- **`strategy/interactions.md`** — Known and suspected couplings between parameters (e.g., "batch size and LR are coupled — halving batch likely requires ~0.7x LR scaling"). Consult before testing any change in isolation.

### Usage discipline

- **Read before every experiment**: The pre-experiment checklist (Step 1 of the loop) requires consulting hypotheses, interactions, and near-misses.
- **Write after every experiment**: The post-experiment analysis (Step 3) requires updating learnings, hypotheses, near-misses, and interactions as appropriate.
- **Keep it curated**: These are not append-only logs. Update existing entries when new evidence changes your understanding. Remove entries that are no longer relevant. Adjust confidence levels and priorities based on accumulated evidence.
- **Seed at run start**: At the beginning of a new experiment run, read all strategy files to bootstrap your understanding. If starting fresh (no strategy files), create them after the baseline experiment using literature notes and prior knowledge.
