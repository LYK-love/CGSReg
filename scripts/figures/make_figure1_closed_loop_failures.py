#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path.home() / "projects" / "CGSReg"
PAPER_ROOT = Path.home() / "projects" / "nips_paper"
BUNDLE_DIR = PROJECT_ROOT / "paper_rollout_eval_bundle"
OUT_DIR = PROJECT_ROOT / "results" / "paper_figures"
PAPER_FIGURE_DIR = PAPER_ROOT / "figures"

ACTION_NAMES = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT",
    4: "RIGHTFIRE",
    5: "LEFTFIRE",
}

ROW_FIGURE_DIR = "figure1_rows"

ROWS = [
    {
        "label": "DreamerV3\nball loss",
        "latex_label": "\\begin{tabular}[c]{@{}r@{}}\\textbf{DreamerV3}\\\\Ball disappearance\\end{tabular}",
        "slug": "dreamer_ball_loss",
        "tag": "ball\nloss",
        "project": "dreamer",
        "condition": "exp_repro",
        "episode": "ep00_seed0",
        "frames": [112, 113, 114, 116, 117, 118, 120],
    },
    {
        "label": "Simulus\nspurious turn",
        "latex_label": "\\begin{tabular}[c]{@{}r@{}}\\textbf{Simulus}\\\\Spurious turn\\end{tabular}",
        "slug": "simulus_spurious_turn",
        "tag": "spurious\nturn",
        "project": "simulus",
        "condition": "exp_repro",
        "episode": "ep02_seed2",
        "frames": [325, 326, 327, 329, 330, 332, 333],
    },
    {
        "label": "TWISTER\nmissed bounce",
        "latex_label": "\\begin{tabular}[c]{@{}r@{}}\\textbf{TWISTER}\\\\Incorrect bounce\\end{tabular}",
        "slug": "twister_missed_bounce",
        "tag": "missed\nbounce",
        "project": "twister",
        "condition": "w0",
        "episode": "ep01_seed1",
        "frames": [456, 457, 458, 459, 460, 461, 462],
        "caption_note": "In the TWISTER row, the ball reaches the paddle but does not rebound correctly.",
    },
    {
        "label": "DIAMOND\npaddle loss",
        "latex_label": "\\begin{tabular}[c]{@{}r@{}}\\textbf{DIAMOND}\\\\Paddle disappearance\\end{tabular}",
        "slug": "diamond_paddle_loss",
        "tag": "paddle\nloss",
        "project": "diamond",
        "condition": "exp_repro",
        "episode": "ep04_seed4",
        "frames": [64, 96, 128, 160, 192, 256, 320],
    },
    {
        "label": "STORM\naction response",
        "latex_label": "\\begin{tabular}[c]{@{}r@{}}\\textbf{STORM}\\\\Poor action response\\\\{\\scriptsize LEFT at $t=113$--$115$}\\end{tabular}",
        "slug": "storm_action_response",
        "tag": "action\nresponse",
        "project": "storm",
        "condition": "exp_repro",
        "episode": "manual_export_20260624_185905",
        "video_path": "/scorpio/home/luyukuan/projects/oc-storm/debug_outputs/wm_play_exports/play_episode_0001_manual_export_20260624_185905.mp4",
        "frames": [111, 112, 113, 114, 115, 116, 117],
        "show_actions": True,
    },
]


def video_path(row: dict) -> Path:
    if row.get("video_path"):
        return Path(row["video_path"]).expanduser()
    return BUNDLE_DIR / "videos" / row["project"] / row["condition"] / f"{row['episode']}.mp4"


def metadata_path(row: dict) -> Path:
    if row.get("video_path"):
        return Path(row["video_path"]).expanduser().with_suffix(".json")
    return BUNDLE_DIR / "rollout_metadata" / row["project"] / row["condition"] / f"{row['episode']}.json"


def read_frame(path: Path, frame_idx: int, size: int) -> np.ndarray:
    frame = np.asarray(iio.imread(path, index=int(frame_idx)))
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    img = Image.fromarray(frame.astype(np.uint8))
    img = img.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(img)


def action_label(row: dict, frame_idx: int) -> str:
    if not row.get("show_actions"):
        return ""
    path = metadata_path(row)
    if not path.exists():
        return "a=?"
    actions = json.loads(path.read_text()).get("actions", [])
    if not actions:
        return "a=?"
    action = int(actions[min(frame_idx, len(actions) - 1)])
    return f"a={ACTION_NAMES.get(action, str(action))}"


def make_row_strip(row: dict, *, frame_size: int = 128, gap: int = 3, border: int = 1) -> Image.Image:
    frames = [int(x) for x in row["frames"]]
    path = video_path(row)
    if not path.exists():
        raise FileNotFoundError(path)

    cell = frame_size + 2 * border
    width = len(frames) * cell + (len(frames) - 1) * gap
    height = cell
    strip = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(strip)
    for c, t in enumerate(frames):
        x0 = c * (cell + gap)
        y0 = 0
        frame = Image.fromarray(read_frame(path, t, frame_size))
        strip.paste(frame, (x0 + border, y0 + border))
        draw.rectangle(
            [x0, y0, x0 + cell - 1, y0 + cell - 1],
            outline=(120, 120, 120),
            width=border,
        )
    return strip


def make_frame_grid(*, frame_size: int = 128, gap: int = 3, row_gap: int = 9, border: int = 1) -> Image.Image:
    ncols = len(ROWS[0]["frames"])
    if any(len(row["frames"]) != ncols for row in ROWS):
        raise ValueError("All Figure 1 rows must use the same number of frames.")

    strips = [make_row_strip(row, frame_size=frame_size, gap=gap, border=border) for row in ROWS]
    width = max(strip.width for strip in strips)
    height = sum(strip.height for strip in strips) + row_gap * (len(strips) - 1)
    grid = Image.new("RGB", (width, height), "white")
    y = 0
    for strip in strips:
        grid.paste(strip, (0, y))
        y += strip.height + row_gap
    return grid


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".pdf":
        image.save(path, "PDF", resolution=300.0)
    else:
        image.save(path)


def make_row_strip_pdf(row: dict, path: Path) -> None:
    image = make_row_strip(row)
    save_image(image, path)


def make_frame_grid_pdf(path: Path) -> None:
    image = make_frame_grid()
    save_image(image, path)


def write_selection() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Manual frame selections for Figure 1 closed-loop rollout examples.",
        "rows": ROWS,
    }
    path = OUT_DIR / "figure1_closed_loop_failures_selection.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_latex_snippet() -> Path:
    lines = [
        "% Auto-generated by scripts/figures/make_figure1_closed_loop_failures.py",
        "{\\footnotesize",
        "\\begin{tabular}{@{}>{\\raggedleft\\arraybackslash}m{0.20\\linewidth}@{\\hspace{0.5em}}m{0.75\\linewidth}@{}}",
    ]
    for row in ROWS:
        lines.extend(
            [
                f"{row['latex_label']} &",
                f"\\includegraphics[width=\\linewidth]{{figures/{ROW_FIGURE_DIR}/{row['slug']}.png}} \\\\[0.25em]",
            ]
        )
    lines.append("\\end{tabular}")
    lines.append("}")
    path = OUT_DIR / "figure1_closed_loop_failures_latex_body.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_caption_note() -> Path:
    text = (
        "Caption note for Figure 1: Rows show representative closed-loop "
        "rollout windows from five frozen world models. The first three "
        "failure types are observed across all five architectures: the ball "
        "can disappear, turn without a valid collision, or fail to bounce "
        "correctly. DIAMOND commonly also loses paddles, while STORM commonly "
        "fails to respond faithfully to supplied actions. In the STORM "
        "example, the second through fourth frames receive LEFT actions, which "
        "move the Pong paddle downward, but the paddle returns toward its "
        "original position from the fifth frame without a RIGHT input.\n"
    )
    path = OUT_DIR / "figure1_closed_loop_failures_caption_note.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / ROW_FIGURE_DIR).mkdir(parents=True, exist_ok=True)
    (PAPER_FIGURE_DIR / ROW_FIGURE_DIR).mkdir(parents=True, exist_ok=True)
    write_selection()
    write_caption_note()
    write_latex_snippet()
    for row in ROWS:
        row_img = make_row_strip(row)
        for suffix in ["png", "pdf"]:
            name = f"{row['slug']}.{suffix}"
            out = OUT_DIR / ROW_FIGURE_DIR / name
            paper_out = PAPER_FIGURE_DIR / ROW_FIGURE_DIR / name
            save_image(row_img, out)
            save_image(row_img, paper_out)
            print(out)
            print(paper_out)

    grid_img = make_frame_grid()
    for name in ["figure1_closed_loop_failures.png", "figure1_closed_loop_failures.pdf"]:
        out = OUT_DIR / name
        paper_out = PAPER_FIGURE_DIR / name
        save_image(grid_img, out)
        save_image(grid_img, paper_out)
        print(out)
        print(paper_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
