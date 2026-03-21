"""
SHD (Spiking Heidelberg Digits) data prep, loader, and evaluation harness.

Fixed constants — do not modify this file.
The agent modifies train_snn.py only.

Dataset: https://compneuro.net/datasets/
    shd_train.h5.gz  (~1.3 GB)
    shd_test.h5.gz   (~330 MB)

Format:
    spikes/times[i]  — float32 array of spike times (seconds) for sample i
    spikes/units[i]  — int32  array of neuron indices (0-699) for sample i
    labels[i]        — uint8  class label (0-19)

We bin spikes into T discrete time steps and return dense binary tensors.
"""

import gzip
import hashlib
import os
import shutil
import time
from pathlib import Path

import h5py
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import requests

# ---------------------------------------------------------------------------
# Fixed constants — do not modify
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.path.expanduser("~/.cache/autoresearch-snn"))
N_CHANNELS = 700          # SHD input neurons
N_CLASSES  = 20           # SHD classes (0-19 digits in German)
TIME_BUDGET = 300         # seconds of training per experiment
T_STEPS = 128             # time bins per sample (tunable in train_snn.py)
MAX_DURATION = 1.0        # seconds — all SHD samples are <= 1 second

# Evaluation uses the full test set
EVAL_BATCH_SIZE = 128

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

SHD_FILES = {
    "shd_train.h5": "https://compneuro.net/datasets/shd_train.h5.gz",
    "shd_test.h5":  "https://compneuro.net/datasets/shd_test.h5.gz",
}


def _download_and_extract(url: str, dest: Path, chunk_size: int = 1 << 20):
    gz_path = dest.with_suffix(".h5.gz")
    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(gz_path, "wb") as f:
        for chunk in r.iter_content(chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100 * downloaded / total
                print(f"\r  {pct:.1f}%", end="", flush=True)
    print()
    print(f"Extracting {gz_path.name} ...")
    with gzip.open(gz_path, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    print(f"  -> {dest}")


def ensure_data():
    """Download SHD dataset if not already cached. Returns (train_path, test_path)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fname, url in SHD_FILES.items():
        dest = CACHE_DIR / fname
        if not dest.exists():
            _download_and_extract(url, dest)
        paths[fname] = dest
    return paths["shd_train.h5"], paths["shd_test.h5"]


# ---------------------------------------------------------------------------
# Spike binning
# ---------------------------------------------------------------------------

def bin_spikes(h5_file: h5py.File, n_steps: int, max_dur: float = MAX_DURATION) -> tuple:
    """
    Convert event-based SHD spikes to dense (N, T, C) binary tensors.

    Args:
        h5_file: open HDF5 file
        n_steps: number of time bins
        max_dur: clip spike times at this value (seconds)

    Returns:
        X: numpy uint8 array (N, T, C)
        y: numpy int64 array (N,)
    """
    times_grp = h5_file["spikes"]["times"]
    units_grp = h5_file["spikes"]["units"]
    labels    = np.array(h5_file["labels"], dtype=np.int64)
    N = len(labels)
    X = np.zeros((N, n_steps, N_CHANNELS), dtype=np.uint8)

    dt = max_dur / n_steps
    for i in range(N):
        ts = np.array(times_grp[i])
        us = np.array(units_grp[i], dtype=np.int64)
        bins = np.floor(np.clip(ts, 0.0, max_dur - 1e-9) / dt).astype(np.int64)
        bins = np.clip(bins, 0, n_steps - 1)
        # clip channels just in case
        valid = (us >= 0) & (us < N_CHANNELS)
        np.add.at(X[i], (bins[valid], us[valid]), 1)
    X = np.clip(X, 0, 1)  # binarise
    return X, labels


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class SHDDataset:
    """Holds pre-binned SHD data as numpy arrays ready for MLX batch iteration."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, T, C),  y: (N,)
        self.X = X
        self.y = y
        self.N = len(y)

    def batches(self, batch_size: int, shuffle: bool = False, rng=None):
        """Yield (mx_x, mx_y) batches. x shape: (batch, T, C)."""
        idx = np.arange(self.N)
        if shuffle:
            rng = rng or np.random.default_rng()
            rng.shuffle(idx)
        for start in range(0, self.N, batch_size):
            end = min(start + batch_size, self.N)
            batch_idx = idx[start:end]
            x_np = self.X[batch_idx].astype(np.float32)   # (B, T, C)
            y_np = self.y[batch_idx]
            yield mx.array(x_np), mx.array(y_np)


def load_datasets(n_steps: int = T_STEPS) -> tuple:
    """
    Ensure SHD is downloaded, bin spikes, return (train_ds, test_ds).

    Caches the binned arrays on disk so re-runs are fast.
    """
    train_h5_path, test_h5_path = ensure_data()
    cache_file = CACHE_DIR / f"shd_binned_{n_steps}.npz"

    if cache_file.exists():
        data = np.load(cache_file)
        train_ds = SHDDataset(data["X_train"], data["y_train"])
        test_ds  = SHDDataset(data["X_test"],  data["y_test"])
        return train_ds, test_ds

    print(f"Binning SHD into {n_steps} time steps ...")
    with h5py.File(train_h5_path, "r") as f:
        X_train, y_train = bin_spikes(f, n_steps)
    with h5py.File(test_h5_path, "r") as f:
        X_test, y_test = bin_spikes(f, n_steps)

    np.savez_compressed(cache_file, X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)
    print(f"  train: {X_train.shape}, test: {X_test.shape}")
    return SHDDataset(X_train, y_train), SHDDataset(X_test, y_test)


# ---------------------------------------------------------------------------
# Evaluation harness (fixed — do not modify)
# ---------------------------------------------------------------------------

def evaluate_accuracy(model, test_ds: SHDDataset, batch_size: int = EVAL_BATCH_SIZE) -> float:
    """
    Evaluate model accuracy on the full SHD test set.

    Args:
        model: callable, takes (T, batch, C) float32 tensor, returns (T, batch, n_classes)
        test_ds: SHDDataset
        batch_size: evaluation batch size

    Returns:
        accuracy in [0, 1]
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in test_ds.batches(batch_size, shuffle=False):
        # x_batch: (B, T, C) — transpose to (T, B, C) for network
        x_seq = x_batch.transpose(1, 0, 2)
        traces = model(x_seq)          # (T, B, n_classes)
        logits = traces.sum(axis=0)    # (B, n_classes)
        preds  = logits.argmax(axis=-1)
        mx.eval(preds)
        correct = int((preds == y_batch).sum().item())
        total_correct += correct
        total_samples += len(y_batch)

    model.train()
    return total_correct / total_samples


def get_peak_memory_mb():
    return mx.get_peak_memory() / 1024 / 1024
