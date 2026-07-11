#!/usr/bin/env python3
"""Export freeze-WM diagnostic eval curves from W&B.

The W&B projects were produced by different codebases, so their native x-axes
and metric names differ. This script normalizes the completed runs to:

project, run_name, condition, freeze_progress, env_step, score_mean, wm_frozen

The generated CSV/JSON/PNG files are intentionally ignored by git; rerun this
script to refresh the local plotting data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ORIGINAL_T = 100_000


@dataclass(frozen=True)
class RunSpec:
    project: str
    wandb_project: str
    run_name: str
    run_id: str
    condition: str
    freeze_progress: float | None
    source_metric: str
    x_mode: str
    samples: int = 200_000

    @property
    def freeze_step(self) -> int | None:
        if self.freeze_progress is None:
            return None
        return int(round(self.freeze_progress * ORIGINAL_T))


RUNS: list[RunSpec] = [
    RunSpec("dreamer", "dreamer-dyna-freeze-wm", "dreamer_size200m_nofreeze_1p5T", "erz51wqh", "nofreeze_1p5T", None, "eval_real/score_mean", "dreamer_step_div4", samples=20_000),
    RunSpec("dreamer", "dreamer-dyna-freeze-wm", "dreamer_size200m_f0p5_to1p5T", "3u0l0upn", "f0p5_to1p5T", 0.5, "eval_real/score_mean", "dreamer_step_div4", samples=20_000),
    RunSpec("dreamer", "dreamer-dyna-freeze-wm", "dreamer_size200m_f0p75_to1p5T", "ygez8r7e", "f0p75_to1p5T", 0.75, "eval_real/score_mean", "dreamer_step_div4", samples=20_000),
    RunSpec("dreamer", "dreamer-dyna-freeze-wm", "dreamer_size200m_f1p0_to1p5T", "jg8ql9ou", "f1p0_to1p5T", 1.0, "eval_real/score_mean", "dreamer_step_div4", samples=20_000),
    RunSpec("simulus", "simulus-dyna-freeze-wm", "simulus_repro_nofreeze_1p5T", "v3toe1st", "nofreeze_1p5T", None, "test_dataset/return", "simulus_eval_index", samples=20_000),
    RunSpec("simulus", "simulus-dyna-freeze-wm", "simulus_repro_f0p5_to1p5T", "0igt7447", "f0p5_to1p5T", 0.5, "test_dataset/return", "simulus_eval_index", samples=20_000),
    RunSpec("simulus", "simulus-dyna-freeze-wm", "simulus_repro_f0p75_to1p5T", "xmjtt338", "f0p75_to1p5T", 0.75, "test_dataset/return", "simulus_eval_index", samples=20_000),
    RunSpec("simulus", "simulus-dyna-freeze-wm", "simulus_repro_f1p0_to1p5T", "d7qxifdm", "f1p0_to1p5T", 1.0, "test_dataset/return", "simulus_eval_index", samples=20_000),
    RunSpec("twister", "twister-dyna-freeze-wm", "twister_repro_nofreeze_1p5T", "bve1x57y", "nofreeze_1p5T", None, "eval_real/score_mean", "twister_eval_index"),
    RunSpec("twister", "twister-dyna-freeze-wm", "twister_repro_f0p5_to1p5T", "327nu8jo", "f0p5_to1p5T", 0.5, "eval_real/score_mean", "twister_eval_index"),
    RunSpec("twister", "twister-dyna-freeze-wm", "twister_repro_f0p75_to1p5T", "9t3p5t4t", "f0p75_to1p5T", 0.75, "eval_real/score_mean", "twister_eval_index"),
    RunSpec("twister", "twister-dyna-freeze-wm", "twister_repro_f1p0_to1p5T", "if1davnj", "f1p0_to1p5T", 1.0, "eval_real/score_mean", "twister_eval_index"),
    RunSpec("storm", "storm-dyna-freeze-wm", "storm_repro_nofreeze_1p5T", "xntvidfn", "nofreeze_1p5T", None, "eval_real/score_mean", "storm_eval_index", samples=1_000),
    RunSpec("storm", "storm-dyna-freeze-wm", "storm_repro_f0p5_to1p5T", "xyjfeywv", "f0p5_to1p5T", 0.5, "eval_real/score_mean", "storm_eval_index", samples=1_000),
    RunSpec("storm", "storm-dyna-freeze-wm", "storm_repro_f0p75_to1p5T", "as19ng2q", "f0p75_to1p5T", 0.75, "eval_real/score_mean", "storm_eval_index", samples=1_000),
    RunSpec("storm", "storm-dyna-freeze-wm", "storm_repro_f1p0_to1p5T", "5a0bhzvd", "f1p0_to1p5T", 1.0, "eval_real/score_mean", "storm_eval_index", samples=1_000),
]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def history_rows(run: Any, samples: int, retries: int) -> tuple[list[dict[str, Any]], str | None]:
    last_error: str | None = None
    for attempt in range(1, retries + 1):
        try:
            rows = run.history(samples=samples, pandas=False)
            return list(rows), None
        except Exception as exc:  # W&B can intermittently return 500/timeouts.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2 * attempt)
    return [], last_error


def score_rows(rows: Iterable[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    return [row for row in rows if is_number(row.get(metric))]


def collapse_close_eval_rows(rows: list[dict[str, Any]], metric: str, max_step_gap: int = 100) -> list[dict[str, Any]]:
    """Deduplicate pairs emitted around the same TWISTER eval trigger."""
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda row: int(row.get("_step", 0)))
    groups: list[list[dict[str, Any]]] = []
    for row in sorted_rows:
        if not groups:
            groups.append([row])
            continue
        prev_step = int(groups[-1][-1].get("_step", 0))
        step = int(row.get("_step", 0))
        if step - prev_step <= max_step_gap:
            groups[-1].append(row)
        else:
            groups.append([row])
    collapsed = []
    for group in groups:
        base = dict(group[-1])
        base[metric] = sum(float(row[metric]) for row in group) / len(group)
        base["_raw_eval_rows"] = len(group)
        collapsed.append(base)
    return collapsed


def infer_env_step(spec: RunSpec, row: dict[str, Any], eval_index: int, total_evals: int) -> int:
    if spec.x_mode == "native_step":
        return int(row.get("_step") or 0)
    if spec.x_mode == "dreamer_step_div4":
        # DreamerV3 logs W&B steps in frame units; the diagnostic contract uses
        # environment-equivalent steps, matching run.steps ~= W&B step / 4.
        return int(round((row.get("_step") or 0) / 4))
    if spec.x_mode == "simulus_eval_index":
        # Simulus command uses evaluation.every=20 and original T=600 epochs.
        epoch = min(eval_index * 20, 900)
        return int(round(epoch / 600 * ORIGINAL_T))
    if spec.x_mode == "twister_eval_index":
        # TWISTER command uses --eval_period_step 10000 up to 150000.
        return min(eval_index * 10_000, 150_000)
    if spec.x_mode == "storm_eval_index":
        # STORM command uses --eval_every_steps 20000 and final 150020.
        if is_number(row.get("eval_real/collection_step")):
            return int(row["eval_real/collection_step"])
        return 150_020 if eval_index >= 8 else eval_index * 20_000
    raise ValueError(f"Unknown x_mode: {spec.x_mode}")


def source_quality(spec: RunSpec, rows_error: str | None, extracted_rows: list[dict[str, Any]]) -> str:
    if rows_error and spec.project == "storm" and len(extracted_rows) == 1:
        return "summary_only_wandb_history_failed"
    if spec.project == "storm" and len(extracted_rows) <= 1:
        return "summary_or_sparse_sampled_history"
    if rows_error:
        return "partial_or_failed_history"
    if spec.x_mode.endswith("eval_index"):
        return f"sampled_history_x_inferred_{spec.x_mode}"
    return "sampled_history_native_step"


def build_curve_rows(api: Any, specs: list[RunSpec], retries: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for spec in specs:
        run = api.run(f"ssl-lab/{spec.wandb_project}/{spec.run_id}")
        rows, error = history_rows(run, spec.samples, retries)
        extracted = score_rows(rows, spec.source_metric)

        if spec.project == "twister":
            extracted = collapse_close_eval_rows(extracted, spec.source_metric)

        if not extracted and spec.project == "storm" and is_number(run.summary.get(spec.source_metric)):
            extracted = [{
                spec.source_metric: float(run.summary[spec.source_metric]),
                "eval_real/collection_step": run.summary.get("eval_real/collection_step", 150_020),
                "_step": run.summary.get(f"{spec.source_metric}/step", None),
            }]

        quality = source_quality(spec, error, extracted)
        for idx, row in enumerate(extracted, start=1):
            env_step = infer_env_step(spec, row, idx, len(extracted))
            progress = env_step / ORIGINAL_T
            freeze_step = spec.freeze_step
            wm_frozen = int(freeze_step is not None and env_step >= freeze_step)
            all_rows.append({
                "project": spec.project,
                "wandb_project": spec.wandb_project,
                "wandb_run_id": spec.run_id,
                "wandb_url": f"https://wandb.ai/ssl-lab/{spec.wandb_project}/runs/{spec.run_id}",
                "run_name": spec.run_name,
                "condition": spec.condition,
                "freeze_step": "" if freeze_step is None else freeze_step,
                "env_step": env_step,
                "freeze_progress": round(progress, 6),
                "wm_frozen": wm_frozen,
                "score_mean": float(row[spec.source_metric]),
                "source_metric": spec.source_metric,
                "source_quality": quality,
                "raw_wandb_step": row.get("_step", ""),
                "raw_eval_rows": row.get("_raw_eval_rows", 1),
            })

        final_score = all_rows[-1]["score_mean"] if extracted else ""
        final_step = all_rows[-1]["env_step"] if extracted else ""
        summary_rows.append({
            "project": spec.project,
            "run_name": spec.run_name,
            "condition": spec.condition,
            "wandb_run_id": spec.run_id,
            "state": run.state,
            "num_eval_points": len(extracted),
            "final_env_step": final_step,
            "final_score_mean": final_score,
            "source_quality": quality,
            "history_error": error or "",
        })
        print(f"{spec.project:8s} {spec.condition:15s} {len(extracted):3d} points  {quality}", flush=True)
    return all_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows: list[dict[str, Any]], outpath: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable; writing SVG fallback: {exc}")
        write_svg_plot(rows, outpath.with_suffix(".svg"))
        return

    outpath.parent.mkdir(parents=True, exist_ok=True)
    projects = ["dreamer", "simulus", "twister", "storm"]
    colors = {
        "nofreeze_1p5T": "#222222",
        "f0p5_to1p5T": "#d62728",
        "f0p75_to1p5T": "#ff7f0e",
        "f1p0_to1p5T": "#1f77b4",
    }
    labels = {
        "nofreeze_1p5T": "no freeze",
        "f0p5_to1p5T": "freeze 0.5T",
        "f0p75_to1p5T": "freeze 0.75T",
        "f1p0_to1p5T": "freeze 1.0T",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, project in zip(axes.ravel(), projects):
        ax.set_title(project.upper() if project != "dreamer" else "DreamerV3")
        project_rows = [row for row in rows if row["project"] == project]
        for condition in labels:
            series = sorted(
                [row for row in project_rows if row["condition"] == condition],
                key=lambda row: float(row["freeze_progress"]),
            )
            if not series:
                continue
            xs = [float(row["freeze_progress"]) for row in series]
            ys = [float(row["score_mean"]) for row in series]
            linestyle = "--" if any("summary_only" in row["source_quality"] for row in series) else "-"
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.6, linestyle=linestyle, color=colors[condition], label=labels[condition])
            freeze_step = series[0]["freeze_step"]
            if freeze_step != "":
                ax.axvline(float(freeze_step) / ORIGINAL_T, color=colors[condition], alpha=0.18, linewidth=1)
        ax.axhline(0, color="#888888", linewidth=0.8, alpha=0.5)
        ax.set_xlim(0, 1.55)
        ax.set_ylim(-22, 22)
        ax.grid(True, linewidth=0.4, alpha=0.35)
    axes[1, 0].set_xlabel("Training progress / original T")
    axes[1, 1].set_xlabel("Training progress / original T")
    axes[0, 0].set_ylabel("Real-env Pong score")
    axes[1, 0].set_ylabel("Real-env Pong score")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Freeze-WM diagnostic: real-env policy score", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, dpi=200)
    print(f"Wrote {outpath}")


def svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg_plot(rows: list[dict[str, Any]], outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1160, 780
    margin_l, margin_t, panel_w, panel_h = 62, 78, 500, 275
    gap_x, gap_y = 55, 74
    projects = ["dreamer", "simulus", "twister", "storm"]
    titles = {"dreamer": "DreamerV3", "simulus": "Simulus", "twister": "TWISTER", "storm": "STORM"}
    colors = {
        "nofreeze_1p5T": "#222222",
        "f0p5_to1p5T": "#d62728",
        "f0p75_to1p5T": "#ff7f0e",
        "f1p0_to1p5T": "#1f77b4",
    }
    labels = {
        "nofreeze_1p5T": "no freeze",
        "f0p5_to1p5T": "freeze 0.5T",
        "f0p75_to1p5T": "freeze 0.75T",
        "f1p0_to1p5T": "freeze 1.0T",
    }

    def sx(x: float, ox: int) -> float:
        return ox + max(0.0, min(1.55, x)) / 1.55 * panel_w

    def sy(y: float, oy: int) -> float:
        return oy + (22 - max(-22.0, min(22.0, y))) / 44 * panel_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#222} .axis{stroke:#333;stroke-width:1} .grid{stroke:#d0d0d0;stroke-width:.6} .tick{font-size:11px;fill:#555} .title{font-size:17px;font-weight:700}.label{font-size:13px}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="580" y="30" text-anchor="middle" font-size="21" font-weight="700">Freeze-WM diagnostic: real-env policy score</text>',
    ]

    legend_x = 268
    for i, (condition, label) in enumerate(labels.items()):
        x = legend_x + i * 165
        parts.append(f'<line x1="{x}" y1="54" x2="{x + 34}" y2="54" stroke="{colors[condition]}" stroke-width="3"/>')
        parts.append(f'<circle cx="{x + 17}" cy="54" r="3.5" fill="{colors[condition]}"/>')
        parts.append(f'<text x="{x + 42}" y="58" class="label">{svg_escape(label)}</text>')

    for idx, project in enumerate(projects):
        row, col = divmod(idx, 2)
        ox = margin_l + col * (panel_w + gap_x)
        oy = margin_t + row * (panel_h + gap_y)
        parts.append(f'<text x="{ox + panel_w / 2}" y="{oy - 16}" text-anchor="middle" class="title">{titles[project]}</text>')
        for t in [-20, -10, 0, 10, 20]:
            y = sy(t, oy)
            parts.append(f'<line x1="{ox}" y1="{y:.2f}" x2="{ox + panel_w}" y2="{y:.2f}" class="grid"/>')
            parts.append(f'<text x="{ox - 8}" y="{y + 4:.2f}" text-anchor="end" class="tick">{t}</text>')
        for t in [0, 0.5, 1.0, 1.5]:
            x = sx(t, ox)
            parts.append(f'<line x1="{x:.2f}" y1="{oy}" x2="{x:.2f}" y2="{oy + panel_h}" class="grid"/>')
            parts.append(f'<text x="{x:.2f}" y="{oy + panel_h + 18}" text-anchor="middle" class="tick">{t:g}</text>')
        parts.append(f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" fill="none" class="axis"/>')
        if col == 0:
            parts.append(f'<text x="{ox - 42}" y="{oy + panel_h / 2}" text-anchor="middle" transform="rotate(-90 {ox - 42} {oy + panel_h / 2})" class="label">Pong score</text>')
        if row == 1:
            parts.append(f'<text x="{ox + panel_w / 2}" y="{oy + panel_h + 46}" text-anchor="middle" class="label">training progress / original T</text>')

        project_rows = [item for item in rows if item["project"] == project]
        for condition, label in labels.items():
            series = sorted(
                [item for item in project_rows if item["condition"] == condition],
                key=lambda item: float(item["freeze_progress"]),
            )
            if not series:
                continue
            freeze_step = series[0]["freeze_step"]
            if freeze_step != "":
                x = sx(float(freeze_step) / ORIGINAL_T, ox)
                parts.append(f'<line x1="{x:.2f}" y1="{oy}" x2="{x:.2f}" y2="{oy + panel_h}" stroke="{colors[condition]}" stroke-width="1.2" opacity=".22"/>')
            points = [(sx(float(item["freeze_progress"]), ox), sy(float(item["score_mean"]), oy)) for item in series]
            if len(points) >= 2:
                path = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
                dash = ' stroke-dasharray="5 4"' if any("summary" in item["source_quality"] for item in series) else ""
                parts.append(f'<polyline points="{path}" fill="none" stroke="{colors[condition]}" stroke-width="2.1"{dash}/>')
            for x, y in points:
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{colors[condition]}"/>')

    parts.append("</svg>")
    outpath.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("experiments/freeze_wm_diagnostic/generated"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    import wandb

    api = wandb.Api(timeout=args.timeout)
    rows, summary_rows = build_curve_rows(api, RUNS, retries=args.retries)
    write_csv(args.outdir / "eval_real_score_curves.csv", rows)
    write_csv(args.outdir / "eval_real_score_curve_summary.csv", summary_rows)
    (args.outdir / "eval_real_score_curves.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not args.no_plot:
        plot_curves(rows, args.outdir / "freeze_wm_eval_real_score.png")
    print(f"Wrote {args.outdir / 'eval_real_score_curves.csv'}")
    print(f"Wrote {args.outdir / 'eval_real_score_curve_summary.csv'}")


if __name__ == "__main__":
    main()
