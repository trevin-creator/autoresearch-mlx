"""Lens and sensor image-formation effects for event-camera realism."""

from __future__ import annotations

import math

import numpy as np

from .config import CameraConfig


class SensorImageModel:
    """Apply practical lens/sensor effects to rendered RGB frames.

    The model approximates a real camera front-end by applying:
    1) radial-tangential lens distortion,
    2) vignetting,
    3) optical blur,
    4) sensor-domain shot/read/dark noise.
    """

    def __init__(
        self,
        cam_cfg: CameraConfig,
        sim_dt_s: float,
        rng: np.random.Generator,
    ):
        self.cam_cfg = cam_cfg
        self.rng = rng
        self.H = cam_cfg.height
        self.W = cam_cfg.width
        self.exposure_s = max(float(cam_cfg.exposure_ratio) * sim_dt_s, 1e-6)
        self._distorted_uv = self._build_inverse_distortion_map()

    def _build_inverse_distortion_map(self) -> np.ndarray:
        """Build map from distorted image grid to undistorted source UV."""
        ys, xs = np.indices((self.H, self.W), dtype=np.float32)
        cx = (self.W - 1) * 0.5
        cy = (self.H - 1) * 0.5
        scale = max(float(self.W), float(self.H)) * 0.5

        x_d = (xs - cx) / scale
        y_d = (ys - cy) / scale
        x_u = x_d.copy()
        y_u = y_d.copy()

        for _ in range(5):
            x_est, y_est = self._distort_norm_xy(x_u, y_u)
            x_u += x_d - x_est
            y_u += y_d - y_est

        u = x_u * scale + cx
        v = y_u * scale + cy
        return np.stack((u, v), axis=-1)

    def _distort_norm_xy(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        k1, k2, p1, p2, k3 = self.cam_cfg.distortion
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * (r2 * r2) + k3 * (r2 * r2 * r2)
        x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return x_d, y_d

    @staticmethod
    def _bilinear_sample_rgb(rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
        h, w, _ = rgb.shape
        u = np.clip(uv[..., 0], 0.0, w - 1.0)
        v = np.clip(uv[..., 1], 0.0, h - 1.0)

        x0 = np.floor(u).astype(np.int32)
        y0 = np.floor(v).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)

        wa = (x1 - u) * (y1 - v)
        wb = (u - x0) * (y1 - v)
        wc = (x1 - u) * (v - y0)
        wd = (u - x0) * (v - y0)

        Ia = rgb[y0, x0]
        Ib = rgb[y0, x1]
        Ic = rgb[y1, x0]
        Id = rgb[y1, x1]

        return (
            Ia * wa[..., None]
            + Ib * wb[..., None]
            + Ic * wc[..., None]
            + Id * wd[..., None]
        )

    @staticmethod
    def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
        sigma = max(float(sigma), 1e-6)
        radius = int(math.ceil(3.0 * sigma))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-0.5 * (x / sigma) ** 2)
        k /= np.sum(k)
        return k

    @staticmethod
    def _separable_blur_rgb(rgb: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0.0:
            return rgb

        k = SensorImageModel._gaussian_kernel_1d(sigma)
        r = len(k) // 2

        h, w, _ = rgb.shape
        tmp = np.empty_like(rgb)
        out = np.empty_like(rgb)

        pad_x = np.pad(rgb, ((0, 0), (r, r), (0, 0)), mode="reflect")
        for i in range(w):
            window = pad_x[:, i : i + 2 * r + 1, :]
            tmp[:, i, :] = np.tensordot(window, k, axes=((1,), (0,)))

        pad_y = np.pad(tmp, ((r, r), (0, 0), (0, 0)), mode="reflect")
        for j in range(h):
            window = pad_y[j : j + 2 * r + 1, :, :]
            out[j, :, :] = np.tensordot(window, k, axes=((0,), (0,)))

        return out

    def _apply_vignetting(self, rgb: np.ndarray) -> np.ndarray:
        strength = float(self.cam_cfg.vignette_strength)
        if strength <= 0.0:
            return rgb

        ys, xs = np.indices((self.H, self.W), dtype=np.float32)
        cx = (self.W - 1) * 0.5
        cy = (self.H - 1) * 0.5
        xn = (xs - cx) / max(cx, 1.0)
        yn = (ys - cy) / max(cy, 1.0)
        r2 = np.clip(xn * xn + yn * yn, 0.0, 2.0)
        gain = np.clip((1.0 - strength * r2) ** 2, 0.2, 1.0)
        return rgb * gain[..., None]

    def _apply_sensor_noise(self, rgb: np.ndarray) -> np.ndarray:
        full_well = max(float(self.cam_cfg.full_well_e), 1.0)
        read_noise_e = max(float(self.cam_cfg.read_noise_e), 0.0)
        dark_current_e_s = max(float(self.cam_cfg.dark_current_e_s), 0.0)

        electrons = np.clip(rgb, 0.0, 1.0) * full_well
        electrons = electrons + dark_current_e_s * self.exposure_s

        # Gaussian approximation for shot noise around photo-electron count.
        shot_std = np.sqrt(np.clip(electrons, 0.0, None))
        noisy_e = electrons + self.rng.normal(0.0, shot_std)
        if read_noise_e > 0.0:
            noisy_e = noisy_e + self.rng.normal(0.0, read_noise_e, size=noisy_e.shape)

        return np.clip(noisy_e / full_well, 0.0, 1.0)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Apply lens/sensor effects to RGB image, preserving input dtype range."""
        rgb_f32 = rgb.astype(np.float32)
        if rgb_f32.max() > 1.5:
            rgb_f32 = rgb_f32 / 255.0

        distorted = self._bilinear_sample_rgb(rgb_f32, self._distorted_uv)
        vignetted = self._apply_vignetting(distorted)
        blurred = self._separable_blur_rgb(vignetted, self.cam_cfg.lens_blur_sigma_px)
        noisy = self._apply_sensor_noise(blurred)

        if rgb.dtype == np.uint8:
            return np.clip(np.round(noisy * 255.0), 0.0, 255.0).astype(np.uint8)
        return noisy.astype(np.float32)
