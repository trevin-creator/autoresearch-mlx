#!/usr/bin/env python3
"""Build a README demo video from generated stereo event dataset outputs.

The output video has a 2x2 layout:
- Top-left:  left RGB preview
- Top-right: right RGB preview
- Bottom-left:  left event frame (red=positive, blue=negative)
- Bottom-right: right event frame (red=positive, blue=negative)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build stereo RGB + event demo video")
    p.add_argument("--input-dir", default="_readme_video_gen")
    p.add_argument("--output", default="assets/stereo-event-camera-simulation-5s.mp4")
    p.add_argument("--fps", type=int, default=20)
    return p.parse_args()


def event_image(height: int, width: int, events: np.ndarray) -> np.ndarray:
    """Render per-frame events into an RGB image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if events.size == 0:
        return img

    x = events[:, 1].astype(np.int64)
    y = events[:, 2].astype(np.int64)
    p = events[:, 3].astype(np.int64)

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(valid):
        return img
    x = x[valid]
    y = y[valid]
    p = p[valid]

    flat = y * width + x

    pos = p > 0
    if np.any(pos):
        pos_count = np.bincount(flat[pos], minlength=height * width).reshape(height, width)
        img[..., 0] = np.clip(pos_count * 32, 0, 255).astype(np.uint8)

    neg = p < 0
    if np.any(neg):
        neg_count = np.bincount(flat[neg], minlength=height * width).reshape(height, width)
        img[..., 2] = np.clip(neg_count * 32, 0, 255).astype(np.uint8)

    return img


def pad_to_macroblock(frame: np.ndarray, block: int = 16) -> np.ndarray:
    h, w = frame.shape[:2]
    out_h = ((h + block - 1) // block) * block
    out_w = ((w + block - 1) // block) * block
    if out_h == h and out_w == w:
        return frame
    padded = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    padded[:h, :w] = frame
    return padded


def add_labels(frame: np.ndarray) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)

    h, w = frame.shape[:2]
    half_w = w // 2
    half_h = h // 2
    labels = [
        (8, 8, "Left RGB"),
        (half_w + 8, 8, "Right RGB"),
        (8, half_h + 8, "Left Events (+ red / - blue)"),
        (half_w + 8, half_h + 8, "Right Events (+ red / - blue)"),
    ]
    for x, y, text in labels:
        draw.rectangle((x - 4, y - 2, x + 250, y + 16), fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255))
    return np.asarray(image)


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    left_rgb_files = sorted((root / "rgb_left_preview").glob("*.npy"))
    if not left_rgb_files:
        raise FileNotFoundError("No rgb_left_preview frames found in input dir")

    writer = iio.get_writer(out.as_posix(), fps=args.fps, codec="libx264", quality=8)
    try:
        for left_path in left_rgb_files:
            idx = left_path.stem
            right_path = root / "rgb_right_preview" / f"{idx}.npy"
            left_evt_path = root / "events_left" / f"{idx}.npz"
            right_evt_path = root / "events_right" / f"{idx}.npz"

            if not (right_path.exists() and left_evt_path.exists() and right_evt_path.exists()):
                continue

            rgb_l = np.load(left_path)
            rgb_r = np.load(right_path)
            ev_l = np.load(left_evt_path)["events"]
            ev_r = np.load(right_evt_path)["events"]

            h, w = rgb_l.shape[:2]
            img_ev_l = event_image(h, w, ev_l)
            img_ev_r = event_image(h, w, ev_r)

            top = np.concatenate([rgb_l, rgb_r], axis=1)
            bottom = np.concatenate([img_ev_l, img_ev_r], axis=1)
            frame = np.concatenate([top, bottom], axis=0)
            frame = add_labels(frame)
            frame = pad_to_macroblock(frame)
            writer.append_data(frame)
    finally:
        writer.close()

    print(f"wrote: {out}")


if __name__ == "__main__":
    main()