"""ESIM-style per-pixel event camera emulator.

Converts a sequence of RGB frames (rendered by Genesis or any other source)
into a stream of events using a log-intensity threshold model with per-pixel
threshold mismatch and refractory period.
"""

from __future__ import annotations

import numpy as np

from .config import EventConfig


class EventCameraEmulator:
    """Per-pixel log-intensity threshold event camera model.

    Maintains per-pixel reference log-intensity and emits events when the
    change exceeds the (possibly noisy) threshold.  Supports refractory
    periods and optional sub-step timestamp interpolation.
    """

    def __init__(
        self,
        height: int,
        width: int,
        cfg: EventConfig,
        rng: np.random.Generator | None = None,
    ):
        self.H = height
        self.W = width
        self.cfg = cfg
        rng = rng or np.random.default_rng()

        # Per-pixel threshold maps with Gaussian mismatch
        self.c_pos_map = np.abs(
            rng.normal(cfg.c_pos, cfg.threshold_sigma, (height, width))
        )
        self.c_neg_map = np.abs(
            rng.normal(cfg.c_neg, cfg.threshold_sigma, (height, width))
        )

        self.L_ref: np.ndarray | None = None
        self.t_ref_us: np.ndarray | None = None
        self.t_last_event_us = np.zeros((height, width), dtype=np.int64)

    @staticmethod
    def rgb_to_gray_float(rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to [0, 1] grayscale using Rec. 601 weights."""
        rgb = rgb.astype(np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return np.clip(gray, 0.0, 1.0)

    def initialize(self, rgb0: np.ndarray, t0_us: int) -> None:
        """Set the initial reference frame. Must be called before step()."""
        gray0 = self.rgb_to_gray_float(rgb0)
        self.L_ref = np.log(gray0 + self.cfg.eps)
        self.t_ref_us = np.full((self.H, self.W), t0_us, dtype=np.int64)

    def step(self, rgb: np.ndarray, t_us: int) -> np.ndarray:
        """Process one frame and return events as (N, 4) int64 array.

        Columns: [t_us, x, y, polarity].
        """
        if self.L_ref is None or self.t_ref_us is None:
            raise RuntimeError("Call initialize() before step().")

        gray = self.rgb_to_gray_float(rgb)
        L = np.log(gray + self.cfg.eps)
        dL = L - self.L_ref

        events: list[tuple[int, int, int, int]] = []

        # Positive crossings
        pos_idx = np.where(dL >= self.c_pos_map)
        for y, x in zip(*pos_idx):
            if (t_us - self.t_last_event_us[y, x]) < self.cfg.refractory_us:
                continue
            n = int(dL[y, x] / self.c_pos_map[y, x])
            if n <= 0:
                continue
            for k in range(n):
                if self.cfg.use_timestamp_interpolation:
                    frac = ((k + 1) * self.c_pos_map[y, x]) / max(dL[y, x], 1e-9)
                    t_evt = int(
                        self.t_ref_us[y, x] + frac * (t_us - self.t_ref_us[y, x])
                    )
                else:
                    t_evt = t_us
                if (t_evt - self.t_last_event_us[y, x]) < self.cfg.refractory_us:
                    continue
                events.append((t_evt, x, y, 1))
                self.t_last_event_us[y, x] = t_evt
            self.L_ref[y, x] += n * self.c_pos_map[y, x]
            self.t_ref_us[y, x] = t_us

        # Negative crossings
        neg_idx = np.where(dL <= -self.c_neg_map)
        for y, x in zip(*neg_idx):
            if (t_us - self.t_last_event_us[y, x]) < self.cfg.refractory_us:
                continue
            n = int((-dL[y, x]) / self.c_neg_map[y, x])
            if n <= 0:
                continue
            for k in range(n):
                if self.cfg.use_timestamp_interpolation:
                    frac = ((k + 1) * self.c_neg_map[y, x]) / max(-dL[y, x], 1e-9)
                    t_evt = int(
                        self.t_ref_us[y, x] + frac * (t_us - self.t_ref_us[y, x])
                    )
                else:
                    t_evt = t_us
                if (t_evt - self.t_last_event_us[y, x]) < self.cfg.refractory_us:
                    continue
                events.append((t_evt, x, y, -1))
                self.t_last_event_us[y, x] = t_evt
            self.L_ref[y, x] -= n * self.c_neg_map[y, x]
            self.t_ref_us[y, x] = t_us

        if not events:
            return np.empty((0, 4), dtype=np.int64)
        return np.asarray(events, dtype=np.int64)
