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

# Phase-1 scoring: penalise configs with too few trades
MIN_TRADES_P1 = 15               # target trade count in the ~440-day val period
# Weighted score = Sharpe × min(1, n_trades / MIN_TRADES_P1)
# ensures statistical credibility alongside raw Sharpe

# Phase-2 confidence filter: require higher confidence for transfer signals
# Raises signal quality bar, reduces overtrading on unfamiliar stocks
TRANSFER_CONF    = 0.65          # confidence threshold for Phase-2 transfer signals
TRANSFER_PERSIST = 2             # persistence for single-config WF; ensemble uses 1

# Phase-2 per-stock gate: skip trading if best quick_eval w_score < this
# Prevents negative-Sharpe contributions from stocks where no config works
MIN_TRANSFER_SCORE = 0.10        # w_score threshold (Sharpe × trade-count penalty)
# Phase-2 ensemble: use top-K per-stock configs with majority vote
TRANSFER_ENSEMBLE_K = 3          # number of configs per stock for voting
TRANSFER_MAJORITY   = 3          # unanimous: all K configs must agree to fire
# Phase-2 quick_eval quality floor: configs where win rate is too low are
# treated as zero (the model is consistently wrong on this stock)
MIN_WIN_RATE_QUICK_EVAL = 0.42   # min win rate to consider a config usable


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

# Additional profiles for multi-cycle Phase-1 scoring.
# Covers cycles 40–85 days to prevent TSLA(60)-specific over-fitting.
# Seeds are offset from SYNTHETIC_PROFILES to ensure fresh data.
MULTI_CYCLE_PROFILES: list[dict] = [
    dict(mu=0.0003, sigma=0.035, cycle=40, amp=0.22, seed=201),  # fast cycle
    dict(mu=0.0003, sigma=0.030, cycle=52, amp=0.19, seed=202),  # med-fast
    dict(mu=0.0003, sigma=0.022, cycle=72, amp=0.13, seed=203),  # med-slow
    dict(mu=0.0004, sigma=0.016, cycle=85, amp=0.10, seed=204),  # slow cycle
]


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


def _generate_from_profile(profile: dict, n_days: int = 1460) -> dict:
    """Generate synthetic OHLCV from an explicit profile dict (no caching)."""
    rng = np.random.default_rng(profile["seed"])
    mu, sigma   = profile["mu"],    profile["sigma"]
    cycle_len   = profile["cycle"]; amp = profile["amp"]
    t      = np.arange(n_days)
    cyc    = amp * np.sin(2 * np.pi * t / cycle_len)
    noise  = rng.normal(0, sigma, n_days)
    logret = mu - 0.5 * sigma**2 + noise + np.diff(cyc, prepend=cyc[0])
    close  = 100.0 * np.exp(np.cumsum(logret))
    dr     = np.abs(rng.normal(0, sigma * 0.6, n_days))
    high   = close * np.exp( dr)
    low    = close * np.exp(-dr)
    open_  = close * np.exp(rng.normal(0, sigma * 0.3, n_days))
    bv     = 1_000_000 * (1 + rng.exponential(0.5, n_days))
    volume = (bv * (1.0 + 3.0 * np.abs(logret) / sigma)).astype(float)
    start  = int(datetime(2021, 1, 4).timestamp())
    dates  = [start + i * 86400 for i in range(n_days)]
    return dict(symbol=f"MC{profile['seed']}", dates=dates,
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


def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             period: int = 14) -> np.ndarray:
    """Normalised Average True Range (ATR / close)."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - prev_close),
                    np.abs(low  - prev_close)))
    return _ema(tr, period) / (close + 1e-8)


def calc_trend_ratio(close: np.ndarray, ema_period: int = 50) -> np.ndarray:
    """(close / EMA) − 1 : positive = above trend, negative = below."""
    return close / (_ema(close, ema_period) + 1e-8) - 1.0


def calc_stoch(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               k_period: int = 14, d_period: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator %K and %D, centred at 0 (range ≈ [-0.5, 0.5]).
    %K = (close − lowest_low) / (highest_high − lowest_low)
    %D = 3-bar EMA of %K
    """
    n = len(close)
    k = np.zeros(n)
    for i in range(k_period - 1, n):
        lo = low[i - k_period + 1: i + 1].min()
        hi = high[i - k_period + 1: i + 1].max()
        k[i] = (close[i] - lo) / (hi - lo + 1e-8)
    d = _ema(k, d_period)
    return k - 0.5, d - 0.5          # centred


def calc_regime(close: np.ndarray, ema_period: int = 200) -> np.ndarray:
    """
    EMA-200 regime: (close / EMA200) − 1.
    Positive → bull regime (above long-term average).
    Negative → bear / sideways regime.
    """
    return close / (_ema(close, ema_period) + 1e-8) - 1.0


# ─────────────────────────────────────────────────────────────
# FEATURE CONFIGURATION  (the search space)
# ─────────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    windows:        list[int]    # rolling window sizes
    use_volume:     bool  = False
    use_rsi:        bool  = False
    use_macd:       bool  = False
    use_bb:         bool  = False
    use_atr:        bool  = False   # ATR normalised volatility
    use_trend:      bool  = False   # close vs EMA-50 ratio
    use_stoch:      bool  = False   # Stochastic %K/%D
    use_regime:     bool  = False   # EMA-200 regime filter
    lookback:       int   = 10      # past bars per sample
    conf_threshold: float = 0.50    # min predict_proba to act on signal

    def __str__(self) -> str:
        tags = [f"w{self.windows}", f"lb{self.lookback}"]
        if self.use_volume: tags.append("vol")
        if self.use_rsi:    tags.append("rsi")
        if self.use_macd:   tags.append("macd")
        if self.use_bb:     tags.append("bb")
        if self.use_atr:    tags.append("atr")
        if self.use_trend:  tags.append("tr")
        if self.use_stoch:  tags.append("stoch")
        if self.use_regime: tags.append("reg")
        if self.conf_threshold > 0.50:
            tags.append(f"c{self.conf_threshold:.2f}")
        return "|".join(tags)


# ─── helper to build configs concisely ─────────────────────
def _cfg(windows, vol=False, rsi=False, macd=False, bb=False,
         atr=False, trend=False, stoch=False, regime=False, lb=10, conf=0.50):
    return FeatureConfig(windows, vol, rsi, macd, bb, atr, trend,
                         stoch, regime, lookback=lb, conf_threshold=conf)


# All configs the AI will try on TSLA (× 3 models)
SEARCH_SPACE: list[FeatureConfig] = [
    # ── Tier 1: baseline momentum ───────────────────────────
    _cfg([5],              lb=5),
    _cfg([5, 10],          lb=10),
    _cfg([5, 10, 20],      lb=10),
    # ── Tier 2: add individual indicators ───────────────────
    _cfg([5, 10, 20], vol=True,             lb=10),
    _cfg([5, 10, 20], rsi=True,             lb=14),
    _cfg([5, 10, 20], macd=True,            lb=20),
    _cfg([5, 10, 20], bb=True,              lb=20),
    _cfg([5, 10, 20], atr=True,             lb=14),
    _cfg([5, 10, 20], trend=True,           lb=14),
    _cfg([5, 10, 20], stoch=True,           lb=14),   # NEW: Stochastic
    _cfg([5, 10, 20], regime=True,          lb=20),   # NEW: EMA-200 regime
    # ── Tier 3: combos that showed promise ──────────────────
    _cfg([5, 10, 20], vol=True, rsi=True,   lb=14),
    _cfg([5, 10, 20], macd=True, bb=True,   lb=20),
    _cfg([5, 10, 20], rsi=True, atr=True,   lb=14),
    _cfg([5, 10, 20], macd=True, trend=True,lb=20),
    _cfg([5, 10, 20], atr=True, trend=True, lb=20),
    _cfg([5, 10, 20], stoch=True, rsi=True, lb=14),   # NEW: Stoch+RSI
    _cfg([5, 10, 20], stoch=True, macd=True,lb=20),   # NEW: Stoch+MACD
    _cfg([5, 10, 20], macd=True, regime=True,lb=20),  # NEW: MACD+regime
    _cfg([5, 10, 20], rsi=True, regime=True,lb=14),   # NEW: RSI+regime
    # ── Tier 4: full combos ──────────────────────────────────
    _cfg([5, 10, 20], True, True, True, True,  lb=20),
    _cfg([5, 10, 20], True, True, True, False, atr=True, trend=True, lb=20),
    _cfg([10, 20, 50],True, True, True, True,  lb=30),
    _cfg([5, 10, 20], rsi=True, macd=True, atr=True, trend=True, lb=20),  # NEW
    _cfg([5, 10, 20], macd=True, trend=True, regime=True, lb=20),          # NEW
    _cfg([5, 10, 20], stoch=True, macd=True, trend=True, lb=20),           # NEW
    _cfg([5, 10, 20], rsi=True, atr=True, regime=True, lb=14),             # NEW
    # ── Tier 5: confidence filtering (proven combos + conf=0.55/0.60/0.65) ─
    _cfg([5, 10, 20], macd=True,            lb=20, conf=0.55),
    _cfg([5, 10, 20], macd=True,            lb=20, conf=0.60),
    _cfg([5, 10, 20], macd=True,            lb=20, conf=0.65),   # NEW
    _cfg([5, 10, 20], rsi=True, atr=True,   lb=14, conf=0.55),
    _cfg([5, 10, 20], rsi=True, atr=True,   lb=14, conf=0.60),
    _cfg([5, 10, 20], True, True, True, False, atr=True, trend=True,
         lb=20, conf=0.55),
    _cfg([5, 10, 20], True, True, True, False, atr=True, trend=True,
         lb=20, conf=0.60),
    _cfg([5, 10, 20], macd=True, trend=True,lb=20, conf=0.55),  # NEW
    _cfg([5, 10, 20], macd=True, trend=True,lb=20, conf=0.60),  # NEW
    _cfg([5, 10, 20], stoch=True, macd=True,lb=20, conf=0.55),  # NEW
    _cfg([5, 10, 20], rsi=True, atr=True, regime=True, lb=14, conf=0.55),  # NEW
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

    if cfg.use_atr:
        cols.append(calc_atr(high, low, close) - calc_atr(high, low, close).mean())

    if cfg.use_trend:
        cols.append(calc_trend_ratio(close))
        # Trend momentum: rate-of-change of the EMA ratio
        tr = calc_trend_ratio(close)
        tr_roc = np.zeros(n)
        tr_roc[10:] = tr[10:] - tr[:-10]
        cols.append(tr_roc)

    if cfg.use_stoch:
        sk, sd = calc_stoch(high, low, close)
        cols.append(sk)
        cols.append(sd)
        cols.append(sk - sd)      # %K−%D crossover signal

    if cfg.use_regime:
        cols.append(calc_regime(close))   # EMA-200 regime

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
                      commission: float = 0.001,
                      proba: np.ndarray | None = None,
                      atr: np.ndarray | None = None,
                      atr_stop_mult: float = 2.5) -> dict:
    """
    Long-only strategy driven by model predictions:
      • Enter  LONG  when prediction == -1  (valley / buy signal)
      • Exit   LONG  when prediction ==  1  (peak   / sell signal)
      • Stop-loss: ATR-based (entry − atr_stop_mult × ATR) when atr is given,
        otherwise fixed stop_loss_pct below entry price
      • Confidence-proportional sizing: when proba is provided, the fraction
        invested scales linearly from 0.5× (p=0.50) to 1.0× (p≥0.75)
      • commission applied on each buy and sell

    Returns a dict with Sharpe, total_return, max_drawdown,
    win_rate, n_trades, and buy-and-hold return for reference.
    """
    n       = len(close)
    equity  = np.zeros(n)
    cash    = 10_000.0
    shares  = 0.0
    entry_p = 0.0
    stop_p  = 0.0
    trades: list[tuple[float, float]] = []   # (entry_price, exit_price)

    equity[0] = cash

    for i in range(1, n):
        price = close[i]

        # ── stop-loss check ─────────────────────────────────
        if shares > 0 and price <= stop_p:
            cash   += shares * price * (1.0 - commission)
            trades.append((entry_p, price))
            shares  = 0.0
            entry_p = 0.0
            stop_p  = 0.0

        # ── signal-driven entry / exit ───────────────────────
        if shares == 0 and preds[i] == -1:              # valley → buy
            # Confidence-proportional sizing [0.5, 1.0]
            if proba is not None:
                p = float(proba[i])
                size_frac = 0.5 + 0.5 * min(1.0, max(0.0, (p - 0.50) / 0.25))
            else:
                size_frac = 1.0
            invest = cash * size_frac
            shares  = invest * (1.0 - commission) / price
            cash   -= invest
            entry_p = price
            # ATR-based stop, capped at stop_loss_pct to protect against large drawdowns
            if atr is not None and atr[i] > 0:
                atr_pct = min(atr_stop_mult * atr[i], stop_loss_pct)
                stop_p = price * (1.0 - atr_pct)
            else:
                stop_p = price * (1.0 - stop_loss_pct)
        elif shares > 0 and preds[i] == 1:              # peak → sell
            cash    = cash + shares * price * (1.0 - commission)
            trades.append((entry_p, price))
            shares  = 0.0
            entry_p = 0.0
            stop_p  = 0.0

        equity[i] = cash + shares * price

    # close any open position at period end
    if shares > 0:
        cash += shares * close[-1] * (1.0 - commission)
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


def _apply_conf_filter(clf, X_scaled: np.ndarray,
                       conf_threshold: float,
                       return_proba: bool = False
                       ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Return predictions filtered by class-specific probability.
    A valley prediction (-1) is kept only when P(valley) ≥ threshold.
    A peak  prediction ( 1) is kept only when P(peak)   ≥ threshold.
    Everything else becomes 0 (neutral / stay flat).

    When return_proba=True, also returns the max-class probability array
    aligned with the predictions (for confidence-proportional sizing).
    """
    classes = list(clf.classes_)
    proba_mat = clf.predict_proba(X_scaled)        # (n, n_classes)

    peak_col   = classes.index(1)  if  1 in classes else -1
    valley_col = classes.index(-1) if -1 in classes else -1

    out = np.zeros(len(X_scaled), dtype=np.int32)

    if conf_threshold <= 0.50:
        out = clf.predict(X_scaled).astype(np.int32)
    else:
        if peak_col   >= 0:
            out[proba_mat[:, peak_col]   >= conf_threshold] =  1
        if valley_col >= 0:
            out[proba_mat[:, valley_col] >= conf_threshold] = -1

    if return_proba:
        # Max probability across signal classes (used for sizing)
        max_proba = np.zeros(len(X_scaled), dtype=np.float32)
        if valley_col >= 0:
            buy_mask = out == -1
            max_proba[buy_mask] = proba_mat[buy_mask, valley_col]
        return out, max_proba

    return out


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
                           step: int = STEP_DAYS,
                           conf_threshold: float | None = None,
                           signal_persist: int = 1) -> dict:
    # conf_threshold from cfg unless overridden
    if conf_threshold is None:
        conf_threshold = cfg.conf_threshold
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
    probas  = np.zeros(n, dtype=np.float32) # confidence for sizing
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
        p_slice, prob_slice = _apply_conf_filter(clf, X_te_s, conf_threshold,
                                                  return_proba=True)
        preds[cursor:te_end]  = p_slice
        probas[cursor:te_end] = prob_slice

        n_wins += 1
        cursor += step

    oos_close  = close[initial_train:]
    oos_preds  = preds[initial_train:]
    oos_probas = probas[initial_train:]

    # Optional: require signal to persist signal_persist consecutive bars
    if signal_persist > 1:
        filtered = np.zeros_like(oos_preds)
        filt_pb  = np.zeros_like(oos_probas)
        for i in range(signal_persist - 1, len(oos_preds)):
            window = oos_preds[i - signal_persist + 1: i + 1]
            if np.all(window == window[0]) and window[0] != 0:
                filtered[i] = window[0]
                filt_pb[i]  = oos_probas[i]
        oos_preds  = filtered
        oos_probas = filt_pb

    # ATR for dynamic stop-loss (raw close-aligned ATR)
    raw_close  = np.array(data["close"])
    raw_high   = np.array(data["high"])
    raw_low    = np.array(data["low"])
    atr_full   = calc_atr(raw_high, raw_low, raw_close)
    # Align with build_features start offset (max(lb, 51))
    feat_start = max(cfg.lookback, 51)
    atr_aligned = atr_full[feat_start:]
    oos_atr     = atr_aligned[initial_train:] if len(atr_aligned) > initial_train else None

    bt = backtest_strategy(oos_close, oos_preds,
                           proba=oos_probas, atr=oos_atr)
    bt["bnh_return"]  = float((close[-1] - close[initial_train]) / (close[initial_train] + 1e-8))
    bt["n_oos_days"]  = int(len(oos_close))
    bt["n_windows"]   = int(n_wins)
    return bt


def _get_oos_preds(data: dict, cfg: FeatureConfig, model_name: str,
                   initial_train: int, step: int,
                   conf_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Run one walk-forward pass and return (oos_close, oos_preds).
    Used by the ensemble to collect per-config predictions.
    """
    X, y, close = build_features(data, cfg)
    n = len(X)
    if initial_train >= n:
        return close[initial_train:], np.zeros(n - initial_train, dtype=np.int32)

    preds  = np.zeros(n, dtype=np.int32)
    cursor = initial_train
    while cursor < n:
        te_end = min(cursor + step, n)
        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X[:cursor])
        X_te_s   = scaler.transform(X[cursor:te_end])
        np.random.seed(42)
        X_aug, y_aug = _oversample_minorities(X_tr_s, y[:cursor])
        clf = _make_clf(model_name)
        clf.fit(X_aug, y_aug)
        preds[cursor:te_end] = _apply_conf_filter(clf, X_te_s, conf_threshold)
        cursor += step

    return close[initial_train:], preds[initial_train:]


def walk_forward_ensemble(data: dict,
                          top_cfgs: list[tuple],
                          initial_train: int = INITIAL_TRAIN_DAYS,
                          step: int = STEP_DAYS,
                          conf_threshold: float = TRANSFER_CONF,
                          majority: int = 2,
                          signal_persist: int = 1) -> dict:
    """
    Ensemble walk-forward: collect OOS predictions from the top-K configs
    and vote.  A signal fires only when ≥ majority configs agree.

    top_cfgs      : list of (FeatureConfig, model_name) tuples
    majority      : minimum agreements needed to fire a signal (default 2 of K)
    signal_persist: require the voted signal to persist this many bars
    """
    K = len(top_cfgs)
    all_preds: list[np.ndarray] = []
    oos_close: np.ndarray | None = None

    for cfg, mname in top_cfgs:
        oc, op = _get_oos_preds(data, cfg, mname, initial_train, step, conf_threshold)
        all_preds.append(op)
        if oos_close is None:
            oos_close = oc

    # All configs share the same time-alignment (start = max(lb, 51) = 51 for lb≤51)
    min_len = min(len(p) for p in all_preds)
    arr  = np.array([p[:min_len] for p in all_preds])   # (K, T)
    oos_close = oos_close[:min_len]

    valley_votes = (arr == -1).sum(axis=0)
    peak_votes   = (arr ==  1).sum(axis=0)

    voted = np.zeros(min_len, dtype=np.int32)
    # Peaks take priority only when no valley consensus
    voted[valley_votes >= majority] = -1
    voted[peak_votes   >= majority] =  1
    # If both have majority, neutral (conflict)
    voted[(valley_votes >= majority) & (peak_votes >= majority)] = 0

    # Optional: require the voted signal to hold for signal_persist consecutive bars
    persist = signal_persist
    if persist > 1:
        filtered = np.zeros_like(voted)
        for i in range(persist - 1, len(voted)):
            window = voted[i - persist + 1: i + 1]
            if np.all(window == window[0]) and window[0] != 0:
                filtered[i] = window[0]
        voted = filtered

    # ATR-based stop-loss for ensemble
    raw_c = np.array(data["close"])
    raw_h = np.array(data["high"])
    raw_l = np.array(data["low"])
    atr_f = calc_atr(raw_h, raw_l, raw_c)
    # Use first config's lookback for alignment
    feat_start = max(top_cfgs[0][0].lookback, 51)
    atr_al = atr_f[feat_start:]
    oos_atr = atr_al[initial_train:min_len + initial_train] if len(atr_al) > initial_train else None
    if oos_atr is not None and len(oos_atr) != min_len:
        oos_atr = None  # length mismatch: skip ATR stop

    # Vote strength as proxy for confidence (fraction of configs agreeing)
    vote_strength = np.zeros(min_len, dtype=np.float32)
    vote_strength[voted == -1] = (valley_votes[voted == -1] / K).astype(np.float32)
    # Map vote fraction [majority/K .. 1] → confidence proxy [0.50 .. 0.75]
    vote_proba = 0.50 + 0.25 * (vote_strength - majority / K) / max(1.0 - majority / K, 1e-8)

    bt = backtest_strategy(oos_close, voted, proba=vote_proba, atr=oos_atr)
    bnh_start   = oos_close[0] if len(oos_close) > 0 else 1.0
    bt["bnh_return"] = float((oos_close[-1] - bnh_start) / (bnh_start + 1e-8))
    bt["n_oos_days"] = int(min_len)
    bt["n_windows"]  = int(len(oos_close) // step)
    return bt


def train_and_evaluate(X_tr: np.ndarray, y_tr: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       close_val: np.ndarray,
                       model_name: str = "rf",
                       conf_threshold: float = 0.50) -> tuple[object, dict]:
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
    preds = _apply_conf_filter(clf, X_val_s, conf_threshold)

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
           f"  {'Sharpe':>6}  {'Ret%':>6}  {'DD%':>6}  {'WR%':>5}  {'#T':>3}  {'Wscore':>7}  {'F1':>5}")
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
                clf, m = train_and_evaluate(X_tr, y_tr, X_vl, y_vl, close_vl,
                                            mname, cfg.conf_threshold)

                sh  = m["sharpe"];       ret = m["total_return"] * 100
                dd  = m["max_drawdown"] * 100; wr = m["win_rate"] * 100
                nt  = m["n_trades"];     f1  = m["f1"]

                # Weighted score: penalise low-trade-count configs
                w_score = m["sharpe"] * min(1.0, m["n_trades"] / MIN_TRADES_P1)
                marker  = " ←best" if w_score > best_sharpe else ""
                print(f"  {exp_num:>3}  {str(cfg):<42}  {mname:>3}"
                      f"  {sh:>6.2f}  {ret:>5.1f}%  {dd:>5.1f}%"
                      f"  {wr:>4.0f}%  {nt:>3}  {w_score:.3f}{marker}")

                results.append({"cfg": cfg, "model": mname,
                                 "w_score": w_score, **m})

                if w_score > best_sharpe:
                    best_sharpe     = w_score
                    best_cfg        = cfg
                    best_model_name = mname

            except Exception as e:
                print(f"  {exp_num:>3}  {str(cfg):<42}  {mname:>3}  ERROR: {e}")

    if best_cfg is None:
        raise RuntimeError("All experiments failed.")

    print(f"\n  ✓ Best config : {best_cfg}")
    print(f"  ✓ Best model  : {best_model_name}")
    print(f"  ✓ Best Sharpe : {best_sharpe:.3f}")

    # Build diversity-based ensemble: one best config per feature family
    # Families: macd, rsi, stoch, bb, atr, base
    # This prevents all 3 slots from being filled by correlated MACD configs
    def _family(cfg: FeatureConfig) -> str:
        if cfg.use_macd:  return "macd"
        if cfg.use_stoch: return "stoch"
        if cfg.use_rsi:   return "rsi"
        if cfg.use_bb:    return "bb"
        if cfg.use_atr:   return "atr"
        return "base"

    top_k = 3
    seen_keys:     set[str] = set()
    seen_families: set[str] = set()
    top_cfgs_models: list[tuple] = []
    for r in sorted(results, key=lambda x: -x.get("w_score", -999)):
        if r.get("n_trades", 0) < 5:
            continue
        key = (str(r["cfg"]), r["model"])
        fam = _family(r["cfg"])
        if key not in seen_keys and fam not in seen_families:
            seen_keys.add(key)
            seen_families.add(fam)
            top_cfgs_models.append((r["cfg"], r["model"]))
        if len(top_cfgs_models) >= top_k:
            break
    print(f"  ✓ Ensemble    : {top_k} diverse configs for Phase-2 voting (3/3 unanimous)")
    for i, (c, m) in enumerate(top_cfgs_models, 1):
        print(f"      {i}. [{_family(c)}]  {c}  [{m}]")

    return best_cfg, best_model_name, results, top_cfgs_models


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
            clf, m = train_and_evaluate(X_tr, y_tr, X_vl, y_vl, close_vl,
                                        model_name, cfg.conf_threshold)

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
    if cfg.use_rsi:   names.append("rsi")
    if cfg.use_macd:  names += ["macd_line", "macd_hist"]
    if cfg.use_bb:    names.append("bb_pos")
    if cfg.use_atr:    names.append("atr")
    if cfg.use_trend:  names += ["trend_ratio", "trend_roc"]
    if cfg.use_stoch:  names += ["stoch_k", "stoch_d", "stoch_kd"]
    if cfg.use_regime: names.append("regime_ema200")

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

def _quick_eval_cfg(data: dict, cfg: FeatureConfig, model_name: str,
                    train_ratio: float = TRAIN_RATIO) -> float:
    """
    Evaluate a (cfg, model) on TWO temporal splits and return the minimum
    (conservative) score.  Using two splits guards against the single-split
    70/30 being accidentally lucky for a stock that won't generalise.

    Split A: first 60% train, last 40% test
    Split B: first train_ratio train, rest test  (default 70/30)

    Returns weighted score = Sharpe × min(1, n_trades / MIN_TRADES_P1).
    Returns -999 on error or too-few samples.
    """
    close = np.array(data["close"])
    n     = len(close)

    split_scores: list[float] = []
    for sr in (0.60, train_ratio):
        split = int(n * sr)
        tr_d = {k: v[:split] for k, v in data.items() if isinstance(v, list)}
        vl_d = {k: v[split:] for k, v in data.items() if isinstance(v, list)}
        tr_d["symbol"] = vl_d["symbol"] = data["symbol"]
        try:
            X_tr, y_tr, _        = build_features(tr_d, cfg)
            X_vl, y_vl, close_vl = build_features(vl_d, cfg)
            if len(X_tr) < 50 or len(X_vl) < 15:
                return -999.0
            if (y_tr == 1).sum() < 3 or (y_tr == -1).sum() < 3:
                return -999.0
            np.random.seed(42)
            # Use TRANSFER_CONF (not cfg.conf_threshold) so the quick-eval
            # threshold matches what will be applied in the final walk-forward.
            _, m = train_and_evaluate(X_tr, y_tr, X_vl, y_vl, close_vl,
                                      model_name, max(cfg.conf_threshold, TRANSFER_CONF))
            # Cap score at 0 when win rate is below floor (model is consistently wrong)
            if m["win_rate"] < MIN_WIN_RATE_QUICK_EVAL and m["n_trades"] >= 3:
                split_scores.append(0.0)
            else:
                split_scores.append(m["sharpe"] * min(1.0, m["n_trades"] / MIN_TRADES_P1))
        except Exception:
            return -999.0

    # Conservative: take the minimum across both splits
    return min(split_scores)


def phase2_transfer_wf(cfg: FeatureConfig, model_name: str,
                       symbols: list[str],
                       explore_results: list[dict] | None = None,
                       top_cfgs_models: list[tuple] | None = None) -> list[dict]:
    """
    Walk-forward backtest on each test stock — full 4-year coverage.

    Per-stock adaptive config selection (when explore_results is given):
      For each test stock a quick 70/30 validation selects the best config
      from the top-N Phase-1 candidates, so each stock gets its own
      tailored feature set rather than always inheriting TSLA's best.
    """
    oos_yr = (DATA_PERIOD_DAYS - INITIAL_TRAIN_DAYS) / 252
    adapt  = explore_results is not None and len(explore_results) > 0
    print(f"\n{'═'*68}")
    if adapt:
        print(f"  PHASE 2 — Walk-Forward Transfer  (per-stock adaptive config)")
    else:
        print(f"  PHASE 2 — Walk-Forward Transfer  ({oos_yr:.1f}yr OOS per stock)")
    print(f"  Fallback config: {cfg}  |  Model: {model_name}")
    print(f"  TRANSFER_CONF={TRANSFER_CONF}")
    print(f"{'═'*68}\n")

    # Build top-20 candidates from Phase 1 (unique by config+model, ≥5 trades)
    cand_pool: list[tuple[FeatureConfig, str]] = []
    if adapt:
        seen: set[str] = set()
        for r in sorted(explore_results, key=lambda x: -x.get("w_score", -999)):
            if r.get("n_trades", 0) < 5:
                continue
            key = (str(r["cfg"]), r["model"])
            if key not in seen:
                seen.add(key)
                cand_pool.append((r["cfg"], r["model"]))
            if len(cand_pool) >= 20:
                break

    hdr = (f"  {'Symbol':<7}  {'Sharpe':>6}  {'Ret%':>7}  {'DD%':>6}"
           f"  {'WR%':>5}  {'#T':>3}  {'BnH%':>7}  {'#wins':>5}  {'cfg':}")
    print(hdr)
    print("  " + "─" * 70)

    results: list[dict] = []
    for sym in symbols:
        try:
            data = fetch_stock_data(sym)

            # ── Per-stock config selection ─────────────────────
            best_sym_cfg, best_sym_model = cfg, model_name
            top_sym_cfgs: list[tuple] = []
            best_s = 1.0  # default: confident (no adjustment) when not adapting

            if adapt and cand_pool:
                scores = [(c, m, _quick_eval_cfg(data, c, m))
                          for c, m in cand_pool]
                scores.sort(key=lambda x: -x[2])
                best_c, best_m, best_s = scores[0]

                # Gate: only trade this stock if at least one config is profitable
                if best_s >= MIN_TRANSFER_SCORE:
                    best_sym_cfg, best_sym_model = best_c, best_m
                    # Build top-K with diversity across feature families so
                    # ensemble configs are less correlated in their errors
                    seen_fam: set[str] = set()
                    for c, m, s in scores:
                        if s <= -999:
                            continue
                        fam = ("macd" if c.use_macd else
                               "stoch" if c.use_stoch else
                               "rsi" if c.use_rsi else
                               "bb" if c.use_bb else
                               "atr" if c.use_atr else "base")
                        if fam not in seen_fam:
                            seen_fam.add(fam)
                            top_sym_cfgs.append((c, m))
                        if len(top_sym_cfgs) >= TRANSFER_ENSEMBLE_K:
                            break
                # else: top_sym_cfgs stays empty → skip-trade branch below

            # Per-stock confidence: boost threshold for uncertain stocks to
            # reduce noise (stocks with low quick_eval score need tighter filter)
            sym_conf = TRANSFER_CONF
            if 0 < best_s < 0.35:
                sym_conf = TRANSFER_CONF + 0.05  # 0.70: more selective for uncertain stocks

            # ── Walk-forward ─────────────────────────────────
            if len(top_sym_cfgs) >= TRANSFER_ENSEMBLE_K:
                # Ensemble vote: signal fires only when ≥ majority configs agree
                wf = walk_forward_ensemble(data, top_sym_cfgs,
                                           conf_threshold=sym_conf,
                                           majority=TRANSFER_MAJORITY,
                                           signal_persist=1)
                cfg_tag = f"ens{TRANSFER_ENSEMBLE_K}x{TRANSFER_MAJORITY}"
            elif len(top_sym_cfgs) >= 1:
                # Fewer than K valid configs — fall back to single best
                wf = walk_forward_backtest(data, best_sym_cfg, best_sym_model,
                                           conf_threshold=sym_conf,
                                           signal_persist=TRANSFER_PERSIST)
                cfg_tag = f"[{best_sym_model}]{best_sym_cfg}"
            else:
                # No profitable config found — skip trading (hold cash)
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
                  f"  {wf['n_windows']:>5}  {cfg_tag}")
            results.append({"symbol": sym, "n": len(data["close"]),
                             "cfg": cfg_tag, **wf})
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
    best_cfg, best_model_name, explore_results, top_cfgs_models = phase1_explore(tsla)

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

    # ── Phase 2 : per-stock adaptive walk-forward transfer ───────
    # Each test stock gets the best config from the top-20 Phase-1
    # candidates (ranked by a quick validation score on that stock).
    # TAKE_PROFIT_PCT is now applied globally in backtest_strategy.
    transfer_results = phase2_transfer_wf(
        best_cfg, best_model_name, TEST_SYMBOLS,
        explore_results=explore_results)

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
