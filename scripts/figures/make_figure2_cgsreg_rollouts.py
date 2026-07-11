#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image


PROJECT_ROOT = Path.home() / "projects" / "CGSReg"
PAPER_ROOT = Path.home() / "projects" / "nips_paper"
BUNDLE = PROJECT_ROOT / "artifacts" / "wm_closeloop_rollout_bundle"
OUT_DIR = PROJECT_ROOT / "results" / "paper_figures"
PAPER_FIGURE_DIR = PAPER_ROOT / "figures"

FRAME_SIZE = 130
GAP = 3

ROWS = {
    "figure2_dreamer_cgsreg_rollout.png": (
        "dreamer",
        "w_recommended",
        "ep00_seed0",
        [80, 86, 92, 98, 104, 116, 122],
    ),
    "figure2_diamond_cgsreg_rollout.png": (
        "diamond",
        "w_recommended",
        "ep00_seed0",
        [80, 110, 140, 170, 200, 240, 280],
    ),
    "figure2_twister_cgsreg_rollout.png": (
        "twister",
        "w_recommended",
        "ep00_seed0",
        [360, 370, 381, 392, 404, 416, 422],
    ),
}


def read_frame(path: Path, frame_idx: int) -> Image.Image:
    frame = np.asarray(iio.imread(path, index=int(frame_idx)))
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    image = Image.fromarray(frame.astype(np.uint8))
    return image.resize((FRAME_SIZE, FRAME_SIZE), Image.Resampling.NEAREST)


def make_strip(project: str, condition: str, episode: str, frames: list[int]) -> Image.Image:
    video = BUNDLE / "videos" / project / condition / f"{episode}.mp4"
    if not video.exists():
        raise FileNotFoundError(video)

    width = len(frames) * FRAME_SIZE + (len(frames) - 1) * GAP
    height = FRAME_SIZE
    strip = Image.new("RGB", (width, height), "white")

    for col, frame_idx in enumerate(frames):
        x = col * (FRAME_SIZE + GAP)
        strip.paste(read_frame(video, frame_idx), (x, 0))
    return strip


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in ROWS.items():
        image = make_strip(*spec)
        out = OUT_DIR / name
        paper_out = PAPER_FIGURE_DIR / name
        save_image(image, out)
        save_image(image, paper_out)
        print(out)
        print(paper_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
