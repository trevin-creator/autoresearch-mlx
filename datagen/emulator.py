"""ESIM-style per-pixel event camera emulator.

Converts a sequence of RGB frames (rendered by Genesis or any other source)
into a stream of events using a log-intensity threshold model with per-pixel
threshold mismatch, refractory period, photoreceptor low-pass, and stochastic
background activity (leak + shot noise).
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
        self.rng = rng or np.random.default_rng()

        # Per-pixel threshold maps with Gaussian mismatch
        self.c_pos_map = np.abs(
            self.rng.normal(cfg.c_pos, cfg.threshold_sigma, (height, width))
        )
        self.c_neg_map = np.abs(
            self.rng.normal(cfg.c_neg, cfg.threshold_sigma, (height, width))
        )

        self.L_ref: np.ndarray | None = None
        self.L_lp: np.ndarray | None = None
        self.t_ref_us: np.ndarray | None = None
        self.t_last_event_us = np.zeros((height, width), dtype=np.int64)
        self.prev_t_us: int | None = None

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
        self.L_lp = self.L_ref.copy()
        self.t_ref_us = np.full((self.H, self.W), t0_us, dtype=np.int64)
        self.prev_t_us = t0_us

    def _sample_noise_events(self, t_start_us: int, t_end_us: int) -> np.ndarray:
        """Generate stochastic background events from leak and shot noise."""
        dt_s = max((t_end_us - t_start_us) * 1e-6, 0.0)
        if dt_s <= 0.0:
            return np.empty((0, 4), dtype=np.int64)

        leak_rate = max(float(self.cfg.leak_rate_hz), 0.0)
        shot_rate = max(float(self.cfg.shot_noise_rate_hz), 0.0)
        if leak_rate <= 0.0 and shot_rate <= 0.0:
            return np.empty((0, 4), dtype=np.int64)

        events: list[tuple[int, int, int, int]] = []

        if leak_rate > 0.0:
            p_leak = np.clip(leak_rate * dt_s, 0.0, 1.0)
            leak_mask = self.rng.random((self.H, self.W)) < p_leak
            ys, xs = np.where(leak_mask)
            for y, x in zip(ys, xs):
                t_evt = int(self.rng.integers(t_start_us, t_end_us + 1))
                if (t_evt - self.t_last_event_us[y, x]) < self.cfg.refractory_us:
                    continue
                events.append((t_evt, x, y, 1))
                self.t_last_event_us[y, x] = t_evt
                if self.L_ref is not None:
                    self.L_ref[y, x] += self.c_pos_map[y, x]

        if shot_rate > 0.0:
            p_shot = np.clip(shot_rate * dt_s, 0.0, 1.0)
            shot_mask = self.rng.random((self.H, self.W)) < p_shot
            ys, xs = np.where(shot_mask)
            for y, x in zip(ys, xs):
                pol = 1 if self.rng.random() < 0.5 else -1
                t_evt = int(self.rng.integers(t_start_us, t_end_us + 1))
                if (t_evt - self.t_last_event_us[y, x]) < self.cfg.refractory_us:
                    continue
                events.append((t_evt, x, y, pol))
                self.t_last_event_us[y, x] = t_evt
                if self.L_ref is not None:
                    if pol > 0:
                        self.L_ref[y, x] += self.c_pos_map[y, x]
                    else:
                        self.L_ref[y, x] -= self.c_neg_map[y, x]

        if not events:
            return np.empty((0, 4), dtype=np.int64)
        return np.asarray(events, dtype=np.int64)

    def _apply_photoreceptor_lowpass(self, L_raw: np.ndarray, t_us: int) -> np.ndarray:
        """Apply first-order low-pass in log-intensity domain."""
        if self.prev_t_us is None:
            self.prev_t_us = t_us

        tau_ms = max(float(self.cfg.photoreceptor_tau_ms), 0.0)
        if tau_ms <= 0.0:
            self.L_lp = L_raw
            return L_raw

        dt_s = max((t_us - self.prev_t_us) * 1e-6, 1e-9)
        alpha = 1.0 - np.exp(-dt_s / (tau_ms * 1e-3))
        if self.L_lp is None:
            self.L_lp = L_raw.copy()
        else:
            self.L_lp = self.L_lp + alpha * (L_raw - self.L_lp)
        assert self.L_lp is not None
        return self.L_lp

    def _emit_polarity_crossings(
        self,
        dL: np.ndarray,
        t_us: int,
        polarity: int,
    ) -> list[tuple[int, int, int, int]]:
        """Emit ON/OFF threshold crossings for one polarity."""
        if self.L_ref is None or self.t_ref_us is None:
            return []

        if polarity > 0:
            mask = dL >= self.c_pos_map
            c_map = self.c_pos_map
            sign = 1.0
        else:
            mask = dL <= -self.c_neg_map
            c_map = self.c_neg_map
            sign = -1.0

        events: list[tuple[int, int, int, int]] = []
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            if (t_us - self.t_last_event_us[y, x]) < self.cfg.refractory_us:
                continue

            dl_abs = dL[y, x] if polarity > 0 else -dL[y, x]
            n = int(dl_abs / c_map[y, x])
            if n <= 0:
                continue

            for k in range(n):
                if self.cfg.use_timestamp_interpolation:
                    frac = ((k + 1) * c_map[y, x]) / max(dl_abs, 1e-9)
                    t_evt = int(
                        self.t_ref_us[y, x] + frac * (t_us - self.t_ref_us[y, x])
                    )
                else:
                    t_evt = t_us

                if (t_evt - self.t_last_event_us[y, x]) < self.cfg.refractory_us:
                    continue
                events.append((t_evt, x, y, polarity))
                self.t_last_event_us[y, x] = t_evt

            self.L_ref[y, x] += sign * n * c_map[y, x]
            self.t_ref_us[y, x] = t_us

        return events

    def step(self, rgb: np.ndarray, t_us: int) -> np.ndarray:
        """Process one frame and return events as (N, 4) int64 array.

        Columns: [t_us, x, y, polarity].
        """
        if self.L_ref is None or self.t_ref_us is None:
            raise RuntimeError("Call initialize() before step().")

        gray = self.rgb_to_gray_float(rgb)
        L_raw = np.log(gray + self.cfg.eps)
        L = self._apply_photoreceptor_lowpass(L_raw, t_us)
        dL = L - self.L_ref

        events = self._emit_polarity_crossings(dL, t_us, polarity=1)
        events.extend(self._emit_polarity_crossings(dL, t_us, polarity=-1))

        assert self.prev_t_us is not None
        noise_events = self._sample_noise_events(self.prev_t_us, t_us)
        if noise_events.size > 0:
            events.extend(noise_events.tolist())

        self.prev_t_us = t_us

        if not events:
            return np.empty((0, 4), dtype=np.int64)

        events_np = np.asarray(events, dtype=np.int64)
        # Stable ordering by timestamp simplifies downstream binning/debugging.
        if events_np.shape[0] > 1:
            order = np.argsort(events_np[:, 0], kind="mergesort")
            events_np = events_np[order]
        return events_np
