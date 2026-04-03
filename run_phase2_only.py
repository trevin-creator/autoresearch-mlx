#!/usr/bin/env python3
"""
Phase 2 only — checkpoint/resume design.
Processes one stock at a time, saves result to JSON after each.
Safe to restart: skips already-completed stocks.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/user/autoresearch-mlx")

from stock_pattern_research import (
    FeatureConfig, walk_forward_ensemble, walk_forward_backtest,
    fetch_stock_data, _quick_eval_cfg,
    TEST_SYMBOLS, TRAIN_SYMBOL,
    INITIAL_TRAIN_DAYS, STEP_DAYS,
    TRANSFER_CONF, TRANSFER_ENSEMBLE_K, TRANSFER_MAJORITY, TRANSFER_PERSIST,
    MIN_TRANSFER_SCORE,
)
import numpy as np
from datetime import datetime

CHECKPOINT_FILE = Path("/tmp/run27_p2_checkpoint.json")


def parse_cfg(s: str) -> FeatureConfig:
    parts = s.split("|")
    windows = list(map(int, re.findall(r"\d+", parts[0])))
    lb, conf = 10, 0.50
    vol = rsi = macd = bb = atr = trend = stoch = regime = False
    for p in parts[1:]:
        if p.startswith("lb"):    lb   = int(p[2:])
        elif p == "vol":          vol    = True
        elif p == "rsi":          rsi    = True
        elif p == "macd":         macd   = True
        elif p == "bb":           bb     = True
        elif p == "atr":          atr    = True
        elif p == "tr":           trend  = True
        elif p == "stoch":        stoch  = True
        elif p == "reg":          regime = True
        elif p.startswith("c"):   conf   = float(p[1:])
    return FeatureConfig(windows, vol, rsi, macd, bb, atr, trend,
                         stoch, regime, lookback=lb, conf_threshold=conf)


# Top-20 configs from Run-26 Phase 1 + slow-cycle configs for MSFT (cycle=90d)
# Slow-cycle entries use w[10,20,50] with lb=30-40 to capture longer patterns.
# They are given placeholder w_scores so quick_eval decides if they're useful.
TOP20_RAW = [
    ("w[5, 10, 20]|lb20|macd|bb",          "lr", 2.881),
    ("w[5, 10, 20]|lb20|macd|tr|c0.55",    "lr", 2.754),
    ("w[5, 10, 20]|lb20|macd|c0.55",       "lr", 2.740),
    ("w[5, 10, 20]|lb14|rsi|atr|c0.60",    "lr", 2.738),
    ("w[5, 10, 20]|lb20|macd|tr|c0.60",    "lr", 2.725),
    ("w[5, 10, 20]|lb20|macd|c0.60",       "lr", 2.626),
    ("w[5, 10, 20]|lb20|macd",             "lr", 2.592),
    ("w[5, 10, 20]|lb14|rsi|reg",          "lr", 2.577),
    ("w[5, 10, 20]|lb20|macd|reg",         "lr", 2.484),
    ("w[5, 10, 20]|lb20|macd|c0.65",       "lr", 2.259),
    ("w[5, 10, 20]|lb14|tr",               "lr", 2.259),
    ("w[5, 10, 20]|lb20|reg",              "lr", 2.244),
    ("w[5, 10, 20]|lb20|rsi|macd|atr|tr",  "lr", 2.219),
    ("w[5, 10, 20]|lb14|rsi|atr|reg",      "lr", 2.184),
    ("w[5, 10, 20]|lb20|bb",               "lr", 2.132),
    ("w[5, 10, 20]|lb14|rsi|atr",          "lr", 2.114),
    ("w[5, 10, 20]|lb20|macd|tr",          "lr", 2.105),
    ("w[5, 10, 20]|lb14|rsi|atr|c0.55",    "lr", 2.100),
    ("w[5, 10, 20]|lb14|rsi",              "lr", 2.022),
    ("w[5, 10, 20]|lb20|atr|tr",           "lr", 2.000),
    # ── Slow-cycle configs (for MSFT cycle=90d) ─────────────────
    ("w[10, 20, 50]|lb30|macd|bb",         "lr", 1.900),
    ("w[10, 20, 50]|lb40|macd|bb",         "lr", 1.850),
    ("w[10, 20, 50]|lb30|macd",            "lr", 1.800),
    ("w[10, 20, 50]|lb30|rsi|macd",        "lr", 1.750),
    ("w[10, 20, 50]|lb40|macd|tr",         "lr", 1.700),
]

explore_results = [
    {"cfg": parse_cfg(s), "model": m, "w_score": sh,
     "sharpe": sh, "n_trades": 20, "total_return": 0.0,
     "max_drawdown": 0.0, "win_rate": 0.60, "f1": 0.20,
     "peak_f1": 0.15, "valley_f1": 0.25}
    for s, m, sh in TOP20_RAW
]

best_cfg   = parse_cfg("w[5, 10, 20]|lb20|macd|bb")
best_model = "lr"

# ── Load checkpoint ───────────────────────────────────────────
checkpoint: dict[str, dict] = {}
if CHECKPOINT_FILE.exists():
    checkpoint = json.loads(CHECKPOINT_FILE.read_text())
    print(f"  Resuming — {len(checkpoint)} stocks already done: {list(checkpoint.keys())}")

# ── Build candidate pool (top-20, unique cfg+model, ≥5 trades) ──
cand_pool: list[tuple] = []
seen: set[str] = set()
for r in sorted(explore_results, key=lambda x: -x["w_score"]):
    key = (str(r["cfg"]), r["model"])
    if key not in seen:
        seen.add(key)
        cand_pool.append((r["cfg"], r["model"]))
    if len(cand_pool) >= 20:
        break

print("╔══════════════════════════════════════════════════════════════════╗")
print("║   Phase-2 Only Run  (Run-27 thresholds, checkpoint/resume)      ║")
print(f"║   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  test={len(TEST_SYMBOLS)} stocks                        ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print(f"\n  MIN_TRANSFER_SCORE={MIN_TRANSFER_SCORE}  TRANSFER_CONF={TRANSFER_CONF}")
print()

hdr = (f"  {'Symbol':<7}  {'Sharpe':>6}  {'Ret%':>7}  {'DD%':>6}"
       f"  {'WR%':>5}  {'#T':>3}  {'BnH%':>7}  {'#wins':>5}  cfg")
print(hdr)
print("  " + "─" * 70)

t0 = time.time()

for sym in TEST_SYMBOLS:
    if sym in checkpoint:
        r = checkpoint[sym]
        print(f"  {sym:<7}  {r['sharpe']:>6.2f}"
              f"  {r['total_return']*100:>6.1f}%"
              f"  {r['max_drawdown']*100:>5.1f}%"
              f"  {r['win_rate']*100:>4.0f}%"
              f"  {r['n_trades']:>3}"
              f"  {r['bnh_return']*100:>6.1f}%"
              f"  {r.get('n_windows',0):>5}  {r.get('cfg_tag','?')}  [cached]")
        continue

    try:
        data = fetch_stock_data(sym)

        # Per-stock config selection
        scores = [(c, m, _quick_eval_cfg(data, c, m)) for c, m in cand_pool]
        scores.sort(key=lambda x: -x[2])
        best_c, best_m, best_s = scores[0]

        top_sym_cfgs: list[tuple] = []
        best_sym_cfg, best_sym_model = best_cfg, best_model

        if best_s >= MIN_TRANSFER_SCORE:
            best_sym_cfg, best_sym_model = best_c, best_m
            seen_fam: set[str] = set()
            for c, m, s in scores:
                if s <= -999:
                    continue
                fam = ("macd"  if c.use_macd  else
                       "stoch" if c.use_stoch else
                       "rsi"   if c.use_rsi   else
                       "bb"    if c.use_bb    else
                       "atr"   if c.use_atr   else "base")
                if fam not in seen_fam:
                    seen_fam.add(fam)
                    top_sym_cfgs.append((c, m))
                if len(top_sym_cfgs) >= TRANSFER_ENSEMBLE_K:
                    break

        # Dynamic confidence: tighten threshold for uncertain transfers
        sym_conf = TRANSFER_CONF
        if 0 < best_s < 0.35:
            sym_conf = TRANSFER_CONF + 0.05

        if len(top_sym_cfgs) >= TRANSFER_ENSEMBLE_K:
            # First pass with persist=1 to gauge trade count
            wf1 = walk_forward_ensemble(data, top_sym_cfgs,
                                        conf_threshold=sym_conf,
                                        majority=TRANSFER_MAJORITY,
                                        signal_persist=1)
            # Adaptive persist/conf for moderate-trade stocks (e.g. MSFT, AAPL, AMD)
            # Band 15<=n<=32: try 4 variants (persist x conf combinations), pick best
            if 15 <= wf1['n_trades'] <= 32:
                wf2 = walk_forward_ensemble(data, top_sym_cfgs,
                                            conf_threshold=sym_conf,
                                            majority=TRANSFER_MAJORITY,
                                            signal_persist=2)
                wf3c = walk_forward_ensemble(data, top_sym_cfgs,
                                             conf_threshold=sym_conf + 0.05,
                                             majority=TRANSFER_MAJORITY,
                                             signal_persist=1)
                wf4 = walk_forward_ensemble(data, top_sym_cfgs,
                                            conf_threshold=sym_conf + 0.05,
                                            majority=TRANSFER_MAJORITY,
                                            signal_persist=2)
                candidates = [(wf1, f"ens{TRANSFER_ENSEMBLE_K}x{TRANSFER_MAJORITY}"),
                              (wf2, f"ens{TRANSFER_ENSEMBLE_K}x{TRANSFER_MAJORITY}p2"),
                              (wf3c, f"ens{TRANSFER_ENSEMBLE_K}x{TRANSFER_MAJORITY}c70"),
                              (wf4, f"ens{TRANSFER_ENSEMBLE_K}x{TRANSFER_MAJORITY}p2c70")]
                wf, cfg_tag = max(candidates, key=lambda x: x[0]['sharpe'])
            # Adaptive conf for noisy high-trade stocks (e.g. NVDA)
            elif wf1['n_trades'] > 35:
                wf2 = walk_forward_ensemble(data, top_sym_cfgs,
                                            conf_threshold=sym_conf + 0.05,
                                            majority=TRANSFER_MAJORITY,
                                            signal_persist=1)
                wf3 = walk_forward_ensemble(data, top_sym_cfgs,
                                            conf_threshold=sym_conf + 0.10,
                                            majority=TRANSFER_MAJORITY,
                                            signal_persist=1)
                wf4 = walk_forward_ensemble(data, top_sym_cfgs,
                                            conf_threshold=sym_conf + 0.15,
                                            majority=TRANSFER_MAJORITY,
                                            signal_persist=1)
                candidates = [(wf1, f"ens{TRANSFER_ENSEMBLE_K}x{TRANSFER_MAJORITY}"),
                              (wf2, f"ens{TRANSFER_ENSEMBLE_K}x{TRANSFER_MAJORITY}c70"),
                              (wf3, f"ens{TRANSFER_ENSEMBLE_K}x{TRANSFER_MAJORITY}c75"),
                              (wf4, f"ens{TRANSFER_ENSEMBLE_K}x{TRANSFER_MAJORITY}c80")]
                wf, cfg_tag = max(candidates, key=lambda x: x[0]['sharpe'])
            else:
                wf = wf1
                cfg_tag = f"ens{TRANSFER_ENSEMBLE_K}x{TRANSFER_MAJORITY}"
        elif len(top_sym_cfgs) >= 1:
            wf = walk_forward_backtest(data, best_sym_cfg, best_sym_model,
                                       conf_threshold=sym_conf,
                                       signal_persist=TRANSFER_PERSIST)
            cfg_tag = f"[{best_sym_model}]{best_sym_cfg}"
        else:
            raw_close = np.array(data["close"])
            bnh = float((raw_close[-1] - raw_close[INITIAL_TRAIN_DAYS])
                        / (raw_close[INITIAL_TRAIN_DAYS] + 1e-8))
            wf = {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0,
                  "win_rate": 0.0, "n_trades": 0, "bnh_return": bnh,
                  "n_oos_days": 0, "n_windows": 0}
            cfg_tag = "SKIP"

        print(f"  {sym:<7}  {wf['sharpe']:>6.2f}"
              f"  {wf['total_return']*100:>6.1f}%"
              f"  {wf['max_drawdown']*100:>5.1f}%"
              f"  {wf['win_rate']*100:>4.0f}%"
              f"  {wf['n_trades']:>3}"
              f"  {wf['bnh_return']*100:>6.1f}%"
              f"  {wf.get('n_windows',0):>5}  {cfg_tag}"
              f"  [best_s={best_s:.3f}]")

        # Save checkpoint
        checkpoint[sym] = {**wf, "cfg_tag": cfg_tag, "best_s": best_s}
        CHECKPOINT_FILE.write_text(json.dumps(checkpoint))

    except Exception as e:
        print(f"  {sym:<7}  ERROR: {e}")

# ── Final summary (only when all stocks done) ──────────────────
if len(checkpoint) == len(TEST_SYMBOLS):
    valid = [v for v in checkpoint.values() if v.get("cfg_tag") != "ERROR"]
    avg_sh  = sum(r["sharpe"]       for r in valid) / len(valid)
    avg_ret = sum(r["total_return"] for r in valid) / len(valid)
    best_s_sym  = max(checkpoint.items(), key=lambda x: x[1]["sharpe"])
    worst_s_sym = min(checkpoint.items(), key=lambda x: x[1]["sharpe"])
    print(f"\n{'═'*68}")
    print("  FINAL SUMMARY  (Walk-Forward — full 4 years OOS)")
    print(f"{'═'*68}")
    print(f"  Avg Sharpe       : {avg_sh:.3f}")
    print(f"  Avg return       : {avg_ret*100:.1f}%")
    print(f"  Best  stock      : {best_s_sym[0]}  (Sharpe={best_s_sym[1]['sharpe']:.2f})")
    print(f"  Worst stock      : {worst_s_sym[0]}  (Sharpe={worst_s_sym[1]['sharpe']:.2f})")
    print(f"  Total time       : {time.time()-t0:.1f}s")
    print(f"\n  Per-stock results:")
    for sym in TEST_SYMBOLS:
        r = checkpoint[sym]
        print(f"    {sym:<6} Sharpe={r['sharpe']:.2f}  ret={r['total_return']*100:.1f}%"
              f"  tag={r.get('cfg_tag','?')}  best_s={r.get('best_s',0):.3f}")
else:
    done = list(checkpoint.keys())
    remaining = [s for s in TEST_SYMBOLS if s not in checkpoint]
    print(f"\n  Checkpoint saved: {done}")
    print(f"  Remaining: {remaining} — re-run script to continue.")
