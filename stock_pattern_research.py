#!/usr/bin/env python3
"""
Stock Pattern Research — AI-driven High/Low Point Detection + Backtesting
==========================================================================
Phase 1 : AI autonomously explores feature combinations on TSLA,
          optimising for Sharpe ratio of the derived long-only strategy.
Phase 2 : Apply the best config to other US stocks and report full
          strategy metrics (Sharpe, return, max-drawdown, win-rate).

Framework: autoresearch-mlx (sklearn on Linux, MLX on Apple Silicon)
"""

import csv
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
TRAIN_SYMBOL     = "TSLA"
TEST_SYMBOLS     = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "AMD"]
CACHE_DIR        = Path.home() / ".cache" / "stock_research"
DATA_PERIOD_DAYS = 1460           # ~4 years of daily bars

# Peak / valley detection params
MIN_PEAK_DIST  = 5               # min trading days between extrema
PROMINENCE_PCT = 0.05            # min price swing (5 %) to qualify

# Phase-1 quick selection split (70/30 for speed)
TRAIN_RATIO = 0.70

# Walk-forward full backtest
INITIAL_TRAIN_DAYS = 252         # 1-year initial training window
STEP_DAYS          = 63          # re-fit every quarter (~63 trading days)

# ─────────────────────────────────────────────────────────────
# DATA FETCHING  (Yahoo Finance v8 → synthetic fallback)
# ─────────────────────────────────────────────────────────────

# Synthetic stock profiles — tuned so each "stock" has distinct
# volatility, trend, and cycle length, mimicking real TSLA / big-tech behaviour.
SYNTHETIC_PROFILES: dict[str, dict] = {
    "TSLA": dict(mu=0.0003, sigma=0.033, cycle=60,  amp=0.18, seed=42),
    "AAPL": dict(mu=0.0004, sigma=0.015, cycle=80,  amp=0.10, seed=7),
    "MSFT": dict(mu=0.0005, sigma=0.014, cycle=90,  amp=0.08, seed=13),
    "GOOGL":dict(mu=0.0004, sigma=0.017, cycle=70,  amp=0.09, seed=21),
    "NVDA": dict(mu=0.0006, sigma=0.035, cycle=50,  amp=0.20, seed=99),
    "AMZN": dict(mu=0.0003, sigma=0.020, cycle=75,  amp=0.12, seed=55),
    "META": dict(mu=0.0004, sigma=0.025, cycle=65,  amp=0.15, seed=77),
    "AMD":  dict(mu=0.0002, sigma=0.030, cycle=55,  amp=0.17, seed=88),
}


def _generate_synthetic(symbol: str, n_days: int = 1000) -> dict:
    """
    Geometric Brownian Motion + sinusoidal cycle + volume noise.
    Each stock has a unique profile so patterns differ meaningfully.
    """
    prof = SYNTHETIC_PROFILES.get(symbol,
           dict(mu=0.0003, sigma=0.020, cycle=70, amp=0.12, seed=0))
    rng  = np.random.default_rng(prof["seed"])

    mu, sigma = prof["mu"], prof["sigma"]
    cycle_len, amp = prof["cycle"], prof["amp"]

    # Daily log-returns = drift + GBM noise + sinusoidal cycle
    t = np.arange(n_days)
    cycle  = amp * np.sin(2 * np.pi * t / cycle_len)   # deterministic cycle
    noise  = rng.normal(0, sigma, n_days)
    logret = mu - 0.5 * sigma**2 + noise + np.diff(cycle, prepend=cycle[0])

    close  = 100.0 * np.exp(np.cumsum(logret))

    # Intraday high/low
    daily_range = np.abs(rng.normal(0, sigma * 0.6, n_days))
    high  = close * np.exp( daily_range)
    low   = close * np.exp(-daily_range)
    open_ = close * np.exp(rng.normal(0, sigma * 0.3, n_days))

    # Volume: higher on big moves
    base_vol = 1_000_000 * (1 + rng.exponential(0.5, n_days))
    vol_mult = 1.0 + 3.0 * np.abs(logret) / sigma
    volume   = (base_vol * vol_mult).astype(float)

    # Timestamps (trading days from 2021-01-04)
    start_ts = int(datetime(2021, 1, 4).timestamp())
    dates    = [start_ts + i * 86400 for i in range(n_days)]

    return dict(symbol=symbol, dates=dates,
                open=open_.tolist(), high=high.tolist(),
                low=low.tolist(),   close=close.tolist(),
                volume=volume.tolist())


def fetch_stock_data(symbol: str) -> dict:
    """
    Try Yahoo Finance first; fall back to synthetic data if network unavailable.
    Results are cached for 1 day.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{symbol}.json"

    if cache.exists() and time.time() - cache.stat().st_mtime < 86400:
        data = json.loads(cache.read_text())
        src  = "cache"
        print(f"  {symbol}: {len(data['close'])} days  [{src}]")
        return data

    # ── Try Yahoo Finance ────────────────────────────────────
    try:
        end   = int(time.time())
        start = end - DATA_PERIOD_DAYS * 86400
        url   = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        hdrs  = {"User-Agent": "Mozilla/5.0 (compatible; StockResearch/1.0)"}
        r     = requests.get(url,
                             params={"period1": start, "period2": end, "interval": "1d"},
                             headers=hdrs, timeout=10)
        r.raise_for_status()
        result = r.json()["chart"]["result"][0]

        ts  = result["timestamp"]
        q   = result["indicators"]["quote"][0]
        adj = result["indicators"]["adjclose"][0]["adjclose"]
        raw = dict(dates=ts, open=q["open"], high=q["high"],
                   low=q["low"], close=adj, volume=q["volume"])
        valid = [i for i in range(len(ts))
                 if all(raw[k][i] is not None
                        for k in ["open", "high", "low", "close", "volume"])]
        out  = {k: [raw[k][i] for i in valid] for k in raw}
        out["symbol"] = symbol
        src  = "Yahoo"

    except Exception:
        # ── Fallback: synthetic data ─────────────────────────
        out = _generate_synthetic(symbol, n_days=DATA_PERIOD_DAYS)
        src = "synthetic-GBM"

    cache.write_text(json.dumps(out))
    print(f"  {symbol}: {len(out['close'])} days  [{src}]")
    return out


# ─────────────────────────────────────────────────────────────
# PEAK / VALLEY DETECTION
# ─────────────────────────────────────────────────────────────

def find_peaks_valleys(prices: list,
                       min_dist: int = MIN_PEAK_DIST,
                       prom_pct: float = PROMINENCE_PCT) -> tuple[list, list]:
    """
    Identify significant local maxima (peaks) and minima (valleys).

    A point qualifies if:
      1. It is the highest/lowest in a ±min_dist window.
      2. Its price swing relative to the surrounding 50-bar baseline
         is at least prom_pct (prominence filter).
    """
    p = np.array(prices, dtype=np.float32)
    n = len(p)
    peaks, valleys = [], []

    for i in range(min_dist, n - min_dist):
        window = p[i - min_dist: i + min_dist + 1]

        if p[i] == window.max():
            # Prominence: how far does price drop on either side?
            left_min  = p[max(0, i - 50): i].min()   if i > 0     else p[i]
            right_min = p[i + 1: min(n, i + 51)].min() if i < n-1 else p[i]
            if (p[i] - max(left_min, right_min)) / (p[i] + 1e-8) >= prom_pct:
                peaks.append(i)

        elif p[i] == window.min():
            left_max  = p[max(0, i - 50): i].max()   if i > 0     else p[i]
            right_max = p[i + 1: min(n, i + 51)].max() if i < n-1 else p[i]
            if (min(left_max, right_max) - p[i]) / (p[i] + 1e-8) >= prom_pct:
                valleys.append(i)

    def dedup(indices: list) -> list:
        """Remove indices closer than min_dist."""
        if not indices:
            return []
        kept = [indices[0]]
        for x in indices[1:]:
            if x - kept[-1] >= min_dist:
                kept.append(x)
        return kept

    return dedup(peaks), dedup(valleys)


# ─────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    out   = np.zeros_like(arr)
    alpha = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def calc_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain  = _ema(np.where(delta > 0,  delta, 0.0), period)
    loss  = _ema(np.where(delta < 0, -delta, 0.0), period)
    # Safe division: avoid divide-by-zero and invalid-value warnings
    rs    = np.zeros_like(gain)
    mask  = loss > 1e-10
    rs[mask]  = gain[mask] / loss[mask]
    rs[~mask] = 100.0
    return (100.0 - 100.0 / (1.0 + rs)) / 100.0   # → [0, 1]


def calc_macd(close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    line = _ema(close, 12) - _ema(close, 26)
    hist = line - _ema(line, 9)
    return line / (close + 1e-8), hist / (close + 1e-8)


def calc_bollinger_pos(close: np.ndarray, period: int = 20) -> np.ndarray:
    out = np.zeros(len(close))
    for i in range(period, len(close)):
        w = close[i - period: i + 1]
        mu, sigma = w.mean(), w.std()
        out[i] = (close[i] - mu) / (2 * sigma + 1e-8)
    return np.clip(out, -2.0, 2.0)


# ─────────────────────────────────────────────────────────────
# FEATURE CONFIGURATION  (the search space)
# ─────────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    windows:     list[int]       # rolling window sizes
    use_volume:  bool = False
    use_rsi:     bool = False
    use_macd:    bool = False
    use_bb:      bool = False
    lookback:    int  = 10       # how many past bars to include per sample

    def __str__(self) -> str:
        tags = [f"w{self.windows}", f"lb{self.lookback}"]
        if self.use_volume: tags.append("vol")
        if self.use_rsi:    tags.append("rsi")
        if self.use_macd:   tags.append("macd")
        if self.use_bb:     tags.append("bb")
        return "|".join(tags)


# All configs the AI will try on TSLA
SEARCH_SPACE: list[FeatureConfig] = [
    # Baseline: price momentum only
    FeatureConfig([5],              False, False, False, False, lookback=5),
    FeatureConfig([5, 10],          False, False, False, False, lookback=10),
    FeatureConfig([5, 10, 20],      False, False, False, False, lookback=10),
    # Add volume
    FeatureConfig([5, 10, 20],      True,  False, False, False, lookback=10),
    # Add RSI
    FeatureConfig([5, 10, 20],      False, True,  False, False, lookback=14),
    # Add MACD
    FeatureConfig([5, 10, 20],      False, False, True,  False, lookback=20),
    # Add Bollinger
    FeatureConfig([5, 10, 20],      False, False, False, True,  lookback=20),
    # Combine momentum + volume + RSI
    FeatureConfig([5, 10, 20],      True,  True,  False, False, lookback=14),
    # Full combo
    FeatureConfig([5, 10, 20],      True,  True,  True,  True,  lookback=20),
    # Wider windows
    FeatureConfig([10, 20, 50],     True,  True,  True,  True,  lookback=30),
    # Very wide + long lookback
    FeatureConfig([5, 10, 20, 50],  True,  True,  True,  True,  lookback=30),
    # RSI + BB only (oscillator-focused)
    FeatureConfig([20],             False, True,  False, True,  lookback=14),
]


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

def build_features(data: dict, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct (X, y) from OHLCV data using the given FeatureConfig.

    Labels  :  1 = peak,  -1 = valley,  0 = neutral
    X shape : (n_samples, n_features * lookback)
    """
    close  = np.array(data["close"],  dtype=np.float64)
    high   = np.array(data["high"],   dtype=np.float64)
    low    = np.array(data["low"],    dtype=np.float64)
    vol    = np.array(data["volume"], dtype=np.float64)
    n      = len(close)

    # ── labels ──────────────────────────────────────────────
    peaks, valleys = find_peaks_valleys(close.tolist())
    labels = np.zeros(n, dtype=np.int32)
    for p in peaks:   labels[p] =  1
    for v in valleys: labels[v] = -1

    # ── per-bar feature columns ──────────────────────────────
    cols: list[np.ndarray] = []

    for w in cfg.windows:
        # Price return over w bars
        ret = np.zeros(n)
        ret[w:] = (close[w:] - close[:-w]) / (close[:-w] + 1e-8)
        cols.append(ret)

        # Volatility: (max_high - min_low) / close in window
        hl = np.zeros(n)
        for i in range(w, n):
            hl[i] = (high[i - w: i + 1].max() - low[i - w: i + 1].min()) / (close[i] + 1e-8)
        cols.append(hl)

        # Position of current close in its w-bar range  [0, 1]
        pos = np.zeros(n)
        for i in range(w, n):
            lo_w = close[i - w: i + 1].min()
            hi_w = close[i - w: i + 1].max()
            pos[i] = (close[i] - lo_w) / (hi_w - lo_w + 1e-8)
        cols.append(pos)

    if cfg.use_volume:
        vol_norm = np.log1p(vol / (vol.mean() + 1e-8))
        cols.append(vol_norm)
        for w in cfg.windows[:2]:
            avg = np.zeros(n)
            for i in range(w, n):
                avg[i] = vol[i - w: i].mean()
            cols.append(vol / (avg + 1e-8) - 1.0)

    if cfg.use_rsi:
        cols.append(calc_rsi(close) - 0.5)         # centred

    if cfg.use_macd:
        macd_line, macd_hist = calc_macd(close)
        cols.append(macd_line)
        cols.append(macd_hist)

    if cfg.use_bb:
        cols.append(calc_bollinger_pos(close))

    feats = np.stack(cols, axis=1)                  # (n, F)

    # ── build lookback windows ───────────────────────────────
    lb    = cfg.lookback
    start = max(lb, 51)                             # ensure indicators are warmed up
    X = np.array([feats[i - lb: i].ravel() for i in range(start, n)], dtype=np.float32)
    y = labels[start:]
    close_aligned = close[start:]                   # prices aligned with X rows

    return X, y, close_aligned


# ─────────────────────────────────────────────────────────────
# MODEL + TRAINING
# ─────────────────────────────────────────────────────────────

def _oversample_minorities(X: np.ndarray, y: np.ndarray,
                           target_ratio: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """
    Repeat peak (1) and valley (-1) samples until each class reaches
    at least target_ratio of the total dataset.  This is simple random
    oversampling — no new synthetic points are generated.
    """
    n_total = len(y)
    target_count = int(n_total * target_ratio)

    aug_X, aug_y = [X], [y]
    for cls in (1, -1):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        shortfall = max(0, target_count - len(idx))
        if shortfall > 0:
            extra = np.random.choice(idx, size=shortfall, replace=True)
            aug_X.append(X[extra])
            aug_y.append(y[extra])

    Xout = np.concatenate(aug_X, axis=0)
    yout = np.concatenate(aug_y, axis=0)
    perm = np.random.permutation(len(yout))
    return Xout[perm], yout[perm]


# ─────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────

def backtest_strategy(close: np.ndarray, preds: np.ndarray,
                      stop_loss_pct: float = 0.05,
                      commission: float = 0.001) -> dict:
    """
    Long-only strategy driven by model predictions:
      • Enter  LONG  when prediction == -1  (valley / buy signal)
      • Exit   LONG  when prediction ==  1  (peak   / sell signal)
      • Hard stop-loss at stop_loss_pct below entry price
      • commission applied on each buy and sell

    Returns a dict with Sharpe, total_return, max_drawdown,
    win_rate, n_trades, and buy-and-hold return for reference.
    """
    n       = len(close)
    equity  = np.zeros(n)
    cash    = 10_000.0
    shares  = 0.0
    entry_p = 0.0
    trades: list[tuple[float, float]] = []   # (entry_price, exit_price)

    equity[0] = cash

    for i in range(1, n):
        price = close[i]

        # ── stop-loss check ─────────────────────────────────
        if shares > 0 and price <= entry_p * (1.0 - stop_loss_pct):
            cash    = shares * price * (1.0 - commission)
            trades.append((entry_p, price))
            shares  = 0.0
            entry_p = 0.0

        # ── signal-driven entry / exit ───────────────────────
        if shares == 0 and preds[i] == -1:              # valley → buy
            shares  = cash * (1.0 - commission) / price
            cash    = 0.0
            entry_p = price
        elif shares > 0 and preds[i] == 1:              # peak → sell
            cash    = shares * price * (1.0 - commission)
            trades.append((entry_p, price))
            shares  = 0.0
            entry_p = 0.0

        equity[i] = cash + shares * price

    # close any open position at period end
    if shares > 0:
        cash = shares * close[-1] * (1.0 - commission)
        trades.append((entry_p, close[-1]))
        equity[-1] = cash

    # ── performance metrics ──────────────────────────────────
    daily_ret = np.diff(equity) / (equity[:-1] + 1e-8)
    if daily_ret.std() > 1e-8:
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    running_max  = np.maximum.accumulate(equity)
    drawdowns    = (running_max - equity) / (running_max + 1e-8)
    max_drawdown = float(drawdowns.max())

    total_return = float((equity[-1] - 10_000.0) / 10_000.0)
    bnh_return   = float((close[-1] - close[0]) / (close[0] + 1e-8))

    wins     = sum(1 for ep, ex in trades if ex > ep)
    win_rate = float(wins / max(len(trades), 1))

    return {
        "sharpe":       sharpe,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "win_rate":     win_rate,
        "n_trades":     len(trades),
        "bnh_return":   bnh_return,
    }


def _make_clf(model_name: str):
    if model_name == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=10,
                                       class_weight="balanced",
                                       min_samples_leaf=1,
                                       random_state=42, n_jobs=-1)
    elif model_name == "gb":
        return GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                           learning_rate=0.08, subsample=0.8,
                                           random_state=42)
    return LogisticRegression(class_weight="balanced", max_iter=1000,
                               random_state=42, C=1.0)


def walk_forward_backtest(data: dict, cfg: FeatureConfig, model_name: str,
                           initial_train: int = INITIAL_TRAIN_DAYS,
                           step: int = STEP_DAYS) -> dict:
    """
    Walk-forward backtest covering the FULL data period:

      ┌──────────────────────────────────────────────┐
      │ Year 1 (train) │ Q1 │ Q2 │ Q3 │ Q4 │ …      │
      │    fixed init  │ ← out-of-sample predictions  │
      └──────────────────────────────────────────────┘

    At each step the training window expands and the model is
    re-fitted before predicting the next quarter.  Predictions
    outside the initial window are always out-of-sample.
    """
    X, y, close = build_features(data, cfg)
    n = len(X)

    if initial_train >= n:
        bnh = float((close[-1] - close[0]) / (close[0] + 1e-8))
        return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "n_trades": 0, "bnh_return": bnh,
                "n_oos_days": 0, "n_windows": 0}

    preds   = np.zeros(n, dtype=np.int32)   # 0 = stay flat during warm-up
    n_wins  = 0
    cursor  = initial_train

    while cursor < n:
        te_end = min(cursor + step, n)

        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X[:cursor])
        X_te_s   = scaler.transform(X[cursor:te_end])

        np.random.seed(42)
        X_aug, y_aug = _oversample_minorities(X_tr_s, y[:cursor])

        clf = _make_clf(model_name)
        clf.fit(X_aug, y_aug)
        preds[cursor:te_end] = clf.predict(X_te_s)

        n_wins += 1
        cursor += step

    oos_close = close[initial_train:]
    oos_preds = preds[initial_train:]

    bt = backtest_strategy(oos_close, oos_preds)
    bt["bnh_return"]  = float((close[-1] - close[initial_train]) / (close[initial_train] + 1e-8))
    bt["n_oos_days"]  = int(len(oos_close))
    bt["n_windows"]   = int(n_wins)
    return bt


def train_and_evaluate(X_tr: np.ndarray, y_tr: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       close_val: np.ndarray,
                       model_name: str = "rf") -> tuple[object, dict]:
    """
    Train a classifier and return (model, metrics_dict).

    metrics_dict keys:
        f1, peak_f1, valley_f1,
        sharpe, total_return, max_drawdown, win_rate, n_trades, bnh_return
    """
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    X_tr_s, y_tr = _oversample_minorities(X_tr_s, y_tr)

    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=10,
            class_weight="balanced",
            min_samples_leaf=1,
            random_state=42, n_jobs=-1
        )
    elif model_name == "gb":
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.08, subsample=0.8,
            random_state=42
        )
    else:
        clf = LogisticRegression(
            class_weight="balanced", max_iter=1000,
            random_state=42, C=1.0
        )

    clf.fit(X_tr_s, y_tr)
    preds = clf.predict(X_val_s)

    # ── classification metrics ───────────────────────────────
    def _f1(cls: int) -> float:
        tp   = ((y_val == cls) & (preds == cls)).sum()
        prec = tp / ((preds == cls).sum() + 1e-8)
        rec  = tp / ((y_val == cls).sum() + 1e-8)
        return float(2 * prec * rec / (prec + rec + 1e-8))

    # ── backtest metrics ─────────────────────────────────────
    bt = backtest_strategy(close_val, preds)

    clf._scaler = scaler
    metrics = {
        "f1":          (_f1(1) + _f1(-1)) / 2.0,
        "peak_f1":     _f1(1),
        "valley_f1":   _f1(-1),
        **bt,
    }
    return clf, metrics


# ─────────────────────────────────────────────────────────────
# PHASE 1 — AUTONOMOUS EXPLORATION ON TSLA
# ─────────────────────────────────────────────────────────────

def phase1_explore(data: dict) -> tuple[FeatureConfig, str, list[dict]]:
    """
    Autonomously try every (feature-config × model) combination on TSLA.
    Primary ranking metric: Sharpe ratio of the derived long-only strategy.
    Returns (best_cfg, best_model_name, all_results).
    """
    sym   = data["symbol"]
    close = np.array(data["close"])
    n     = len(close)
    split = int(n * TRAIN_RATIO)

    peaks, valleys = find_peaks_valleys(close.tolist())

    print(f"\n{'═'*80}")
    print(f"  PHASE 1 — Autonomous Exploration on {sym}  (metric = Sharpe ratio)")
    print(f"{'═'*80}")
    print(f"  Data : {n} days  |  Peaks : {len(peaks)}  |  Valleys : {len(valleys)}")
    print(f"  Train: {split} days  |  Val: {n - split} days\n")

    tr_data = {k: v[:split] for k, v in data.items() if isinstance(v, list)}
    vl_data = {k: v[split:] for k, v in data.items() if isinstance(v, list)}
    tr_data["symbol"] = vl_data["symbol"] = sym

    results: list[dict]  = []
    best_sharpe          = -999.0
    best_cfg             = None
    best_model_name      = "lr"

    MODEL_NAMES = ["rf", "gb", "lr"]

    hdr = (f"  {'#':>3}  {'Feature Config':<42}  {'Mdl':>3}"
           f"  {'Sharpe':>6}  {'Ret%':>6}  {'DD%':>6}  {'WR%':>5}  {'#T':>3}  {'F1':>5}")
    print(hdr)
    print("  " + "─" * 82)

    exp_num = 0
    for cfg in SEARCH_SPACE:
        try:
            X_tr, y_tr, _        = build_features(tr_data, cfg)
            X_vl, y_vl, close_vl = build_features(vl_data, cfg)

            if len(X_tr) < 50 or len(X_vl) < 15:
                continue
            if (y_tr == 1).sum() < 3 or (y_tr == -1).sum() < 3:
                continue

        except Exception as e:
            exp_num += 1
            print(f"  {exp_num:>3}  {str(cfg):<42}  BUILD ERROR: {e}")
            continue

        for mname in MODEL_NAMES:
            exp_num += 1
            try:
                np.random.seed(42)
                clf, m = train_and_evaluate(X_tr, y_tr, X_vl, y_vl, close_vl, mname)

                sh  = m["sharpe"];       ret = m["total_return"] * 100
                dd  = m["max_drawdown"] * 100; wr = m["win_rate"] * 100
                nt  = m["n_trades"];     f1  = m["f1"]

                marker = " ←best" if m["sharpe"] > best_sharpe else ""
                print(f"  {exp_num:>3}  {str(cfg):<42}  {mname:>3}"
                      f"  {sh:>6.2f}  {ret:>5.1f}%  {dd:>5.1f}%"
                      f"  {wr:>4.0f}%  {nt:>3}  {f1:.3f}{marker}")

                results.append({"cfg": cfg, "model": mname, **m})

                if m["sharpe"] > best_sharpe:
                    best_sharpe     = m["sharpe"]
                    best_cfg        = cfg
                    best_model_name = mname

            except Exception as e:
                print(f"  {exp_num:>3}  {str(cfg):<42}  {mname:>3}  ERROR: {e}")

    if best_cfg is None:
        raise RuntimeError("All experiments failed.")

    print(f"\n  ✓ Best config : {best_cfg}")
    print(f"  ✓ Best model  : {best_model_name}")
    print(f"  ✓ Best Sharpe : {best_sharpe:.3f}")
    return best_cfg, best_model_name, results


# ─────────────────────────────────────────────────────────────
# PHASE 2 — TRANSFER TO OTHER US STOCKS
# ─────────────────────────────────────────────────────────────

def phase2_transfer(cfg: FeatureConfig, model_name: str, symbols: list[str]) -> list[dict]:
    print(f"\n{'═'*80}")
    print(f"  PHASE 2 — Transfer Test  |  Config: {cfg}  |  Model: {model_name}")
    print(f"{'═'*80}\n")
    hdr = (f"  {'Symbol':<7}  {'Sharpe':>6}  {'Ret%':>7}  {'DD%':>6}"
           f"  {'WR%':>5}  {'#T':>3}  {'BnH%':>7}  {'F1':>5}")
    print(hdr)
    print("  " + "─" * 60)

    results: list[dict] = []
    for sym in symbols:
        try:
            data  = fetch_stock_data(sym)
            n     = len(data["close"])
            split = int(n * TRAIN_RATIO)

            tr_data = {k: v[:split] for k, v in data.items() if isinstance(v, list)}
            vl_data = {k: v[split:] for k, v in data.items() if isinstance(v, list)}
            tr_data["symbol"] = vl_data["symbol"] = sym

            X_tr, y_tr, _        = build_features(tr_data, cfg)
            X_vl, y_vl, close_vl = build_features(vl_data, cfg)

            np.random.seed(42)
            clf, m = train_and_evaluate(X_tr, y_tr, X_vl, y_vl, close_vl, model_name)

            print(f"  {sym:<7}  {m['sharpe']:>6.2f}"
                  f"  {m['total_return']*100:>6.1f}%"
                  f"  {m['max_drawdown']*100:>5.1f}%"
                  f"  {m['win_rate']*100:>4.0f}%"
                  f"  {m['n_trades']:>3}"
                  f"  {m['bnh_return']*100:>6.1f}%"
                  f"  {m['f1']:.3f}")

            results.append({"symbol": sym, "n": n, **m})

        except Exception as e:
            print(f"  {sym:<7}  ERROR: {e}")
            results.append({"symbol": sym, "error": str(e)})

    return results


# ─────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE REPORT
# ─────────────────────────────────────────────────────────────

def print_feature_importance(clf, cfg: FeatureConfig, top_n: int = 10) -> None:
    """Print top features from RF/GB (feature_importances_) or LR (coef_)."""
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        # For LR: sum absolute coefficients across all classes
        imp = np.abs(clf.coef_).sum(axis=0)
        imp = imp / (imp.sum() + 1e-12)
    else:
        return
    # Build feature names
    names: list[str] = []
    for w in cfg.windows:
        names += [f"return_{w}d", f"volatility_{w}d", f"range_pos_{w}d"]
    if cfg.use_volume:
        names.append("log_volume")
        for w in cfg.windows[:2]:
            names.append(f"vol_ratio_{w}d")
    if cfg.use_rsi:  names.append("rsi")
    if cfg.use_macd: names += ["macd_line", "macd_hist"]
    if cfg.use_bb:   names.append("bb_pos")

    F = len(names)                                  # features per time-step
    lb = cfg.lookback

    # Aggregate across lookback steps
    total_feats = len(imp)
    agg = np.zeros(F)
    for t in range(lb):
        start_idx = t * F
        end_idx   = min(start_idx + F, total_feats)
        valid_len = end_idx - start_idx
        agg[:valid_len] += imp[start_idx:end_idx]
    agg /= (lb + 1e-8)

    ranked = sorted(zip(names, agg), key=lambda x: -x[1])
    print(f"\n  Top-{top_n} most important features (averaged over {lb}-bar lookback):")
    for i, (name, score) in enumerate(ranked[:top_n], 1):
        bar = "█" * int(score * 200)
        print(f"    {i:>2}. {name:<20} {score:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────
# RESULTS LOGGING
# ─────────────────────────────────────────────────────────────

def phase2_transfer_wf(cfg: FeatureConfig, model_name: str, symbols: list[str]) -> list[dict]:
    """Walk-forward backtest on each test stock — full 4-year coverage."""
    oos_yr = (DATA_PERIOD_DAYS - INITIAL_TRAIN_DAYS) / 252
    print(f"\n{'═'*68}")
    print(f"  PHASE 2 — Walk-Forward Transfer  "
          f"({oos_yr:.1f}yr OOS per stock)")
    print(f"  Config: {cfg}  |  Model: {model_name}")
    print(f"{'═'*68}\n")

    hdr = (f"  {'Symbol':<7}  {'Sharpe':>6}  {'Ret%':>7}  {'DD%':>6}"
           f"  {'WR%':>5}  {'#T':>3}  {'BnH%':>7}  {'#wins':>5}")
    print(hdr)
    print("  " + "─" * 58)

    results: list[dict] = []
    for sym in symbols:
        try:
            data = fetch_stock_data(sym)
            wf   = walk_forward_backtest(data, cfg, model_name)
            print(f"  {sym:<7}  {wf['sharpe']:>6.2f}"
                  f"  {wf['total_return']*100:>6.1f}%"
                  f"  {wf['max_drawdown']*100:>5.1f}%"
                  f"  {wf['win_rate']*100:>4.0f}%"
                  f"  {wf['n_trades']:>3}"
                  f"  {wf['bnh_return']*100:>6.1f}%"
                  f"  {wf['n_windows']:>5}")
            results.append({"symbol": sym, "n": len(data["close"]), **wf})
        except Exception as e:
            print(f"  {sym:<7}  ERROR: {e}")
            results.append({"symbol": sym, "error": str(e)})

    return results


def save_results(explore: list[dict], transfer: list[dict], best_cfg: FeatureConfig) -> None:
    # Phase-1 exploration results ranked by Sharpe
    with open("stock_exploration_results.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["rank", "config", "model",
                    "sharpe", "total_return", "max_drawdown", "win_rate", "n_trades",
                    "f1", "peak_f1", "valley_f1"])
        for i, r in enumerate(sorted(explore, key=lambda x: -x.get("sharpe", -999)), 1):
            w.writerow([i, str(r["cfg"]), r["model"],
                        f"{r.get('sharpe',0):.3f}",
                        f"{r.get('total_return',0)*100:.2f}",
                        f"{r.get('max_drawdown',0)*100:.2f}",
                        f"{r.get('win_rate',0)*100:.1f}",
                        r.get("n_trades", 0),
                        f"{r['f1']:.4f}", f"{r['peak_f1']:.4f}", f"{r['valley_f1']:.4f}"])

    # Phase-2 walk-forward transfer results
    with open("stock_transfer_results.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["symbol", "n_days", "oos_days",
                    "sharpe", "total_return%", "max_drawdown%", "win_rate%",
                    "n_trades", "n_refits", "bnh_return%"])
        for r in transfer:
            if "error" not in r:
                w.writerow([r["symbol"], r["n"],
                            r.get("n_oos_days", ""),
                            f"{r['sharpe']:.3f}",
                            f"{r['total_return']*100:.2f}",
                            f"{r['max_drawdown']*100:.2f}",
                            f"{r['win_rate']*100:.1f}",
                            r["n_trades"],
                            r.get("n_windows", ""),
                            f"{r['bnh_return']*100:.2f}"])

    print("\n  Saved → stock_exploration_results.tsv")
    print("  Saved → stock_transfer_results.tsv")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   Stock Pattern Research — AI High/Low Point Detection           ║")
    print(f"║   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  train={TRAIN_SYMBOL}  |  test={len(TEST_SYMBOLS)} stocks          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    t0 = time.time()

    # ── Fetch TSLA ────────────────────────────────────────────
    print(f"\nFetching {TRAIN_SYMBOL} data...")
    tsla = fetch_stock_data(TRAIN_SYMBOL)

    # ── Phase 1 : autonomous exploration (70/30 quick selection) ──
    best_cfg, best_model_name, explore_results = phase1_explore(tsla)

    # Feature importance (train on 70% split)
    close  = np.array(tsla["close"])
    n      = len(close)
    split  = int(n * TRAIN_RATIO)
    tr_d   = {k: v[:split] for k, v in tsla.items() if isinstance(v, list)}
    vl_d   = {k: v[split:] for k, v in tsla.items() if isinstance(v, list)}
    tr_d["symbol"] = vl_d["symbol"] = TRAIN_SYMBOL
    X_tr, y_tr, _        = build_features(tr_d, best_cfg)
    X_vl, y_vl, close_vl = build_features(vl_d, best_cfg)
    np.random.seed(42)
    best_clf, _ = train_and_evaluate(
        X_tr, y_tr, X_vl, y_vl, close_vl, model_name=best_model_name)
    print_feature_importance(best_clf, best_cfg)

    # ── TSLA walk-forward full 4-year backtest ─────────────────
    oos_years = (DATA_PERIOD_DAYS - INITIAL_TRAIN_DAYS) / 252
    print(f"\n{'═'*68}")
    print(f"  TSLA Walk-Forward Backtest  "
          f"({INITIAL_TRAIN_DAYS}d init-train → {oos_years:.1f}yr OOS)")
    print(f"  step={STEP_DAYS}d  |  config={best_cfg}  |  model={best_model_name}")
    print(f"{'═'*68}")
    tsla_wf = walk_forward_backtest(tsla, best_cfg, best_model_name)
    print(f"  Sharpe       : {tsla_wf['sharpe']:.3f}")
    print(f"  Total return : {tsla_wf['total_return']*100:.1f}%  "
          f"(vs buy-and-hold {tsla_wf['bnh_return']*100:.1f}%)")
    print(f"  Max drawdown : {tsla_wf['max_drawdown']*100:.1f}%")
    print(f"  Win rate     : {tsla_wf['win_rate']*100:.0f}%")
    print(f"  Trades       : {tsla_wf['n_trades']}  "
          f"({tsla_wf['n_windows']} re-fits)")
    print(f"  OOS days     : {tsla_wf['n_oos_days']}")

    # ── Phase 2 : walk-forward transfer to other stocks ───────
    transfer_results = phase2_transfer_wf(best_cfg, best_model_name, TEST_SYMBOLS)

    # ── Save ──────────────────────────────────────────────────
    save_results(explore_results, transfer_results, best_cfg)

    # ── Final Summary ─────────────────────────────────────────
    valid = [r for r in transfer_results if "error" not in r]

    print(f"\n{'═'*68}")
    print("  FINAL SUMMARY  (Walk-Forward — full 4 years OOS)")
    print(f"{'═'*68}")
    print(f"  Training stock   : {TRAIN_SYMBOL}")
    print(f"  Best feature set : {best_cfg}")
    print(f"  Best model       : {best_model_name}")
    print(f"  Configs tried    : {len(explore_results)}")
    print()
    print(f"  ── {TRAIN_SYMBOL} 4-year walk-forward ─────────────────────")
    print(f"  Sharpe           : {tsla_wf['sharpe']:.3f}")
    print(f"  Total return     : {tsla_wf['total_return']*100:.1f}%")
    print(f"  Max drawdown     : {tsla_wf['max_drawdown']*100:.1f}%")
    print(f"  Win rate         : {tsla_wf['win_rate']*100:.0f}%")
    print(f"  vs Buy-and-hold  : {tsla_wf['bnh_return']*100:.1f}%")
    if valid:
        avg_sh  = sum(r["sharpe"]       for r in valid) / len(valid)
        avg_ret = sum(r["total_return"] for r in valid) / len(valid)
        best_s  = max(valid, key=lambda r: r["sharpe"])
        worst_s = min(valid, key=lambda r: r["sharpe"])
        print()
        print(f"  ── Transfer-test ({len(valid)} stocks, 4yr walk-forward) ──")
        print(f"  Avg Sharpe       : {avg_sh:.3f}")
        print(f"  Avg return       : {avg_ret*100:.1f}%")
        print(f"  Best  stock      : {best_s['symbol']}  (Sharpe={best_s['sharpe']:.2f})")
        print(f"  Worst stock      : {worst_s['symbol']}  (Sharpe={worst_s['sharpe']:.2f})")

    elapsed = time.time() - t0
    print(f"  Total time       : {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
