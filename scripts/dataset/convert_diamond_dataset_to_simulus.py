#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import torch


MASK_KEYS = ("mask1", "mask2", "mask3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a DIAMOND-format offline dataset split to Simulus episode .pt files."
    )
    parser.add_argument(
        "--diamond-train-dir",
        type=Path,
        required=True,
        help="DIAMOND-format train split containing info.pt and episode .pt files.",
    )
    parser.add_argument(
        "--output-train-dir",
        type=Path,
        required=True,
        help="Output Simulus-format train directory.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it exists.")
    return parser.parse_args()


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def episode_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (int(path.stem), str(path))
    except ValueError:
        return (10**9, str(path))


def find_episode_files(train_dir: Path) -> list[Path]:
    files = [path for path in train_dir.rglob("*.pt") if path.name != "info.pt"]
    files = sorted(files, key=episode_sort_key)
    if not files:
        raise FileNotFoundError(f"No episode .pt files found under {train_dir}")
    return files


def convert_episode(src: dict[str, Any]) -> dict[str, Any]:
    required = ("obs", "act", "rew", "end")
    missing = [key for key in required if key not in src]
    if missing:
        raise KeyError(f"DIAMOND episode missing required keys: {missing}")
    obs = src["obs"].to(torch.uint8)
    act = src["act"].to(torch.long)
    rew = src["rew"].to(torch.float32)
    end = src["end"].to(torch.long)
    payload: dict[str, Any] = {
        "observations": {"image": obs},
        "actions": act,
        "rewards": rew,
        "ends": end,
        "mask_padding": torch.ones(len(act), dtype=torch.bool),
        "last_info": {},
    }
    spatial_masks = {}
    for key in MASK_KEYS:
        value = src.get(key)
        if value is not None:
            spatial_masks[key] = value.to(torch.uint8)
    if spatial_masks:
        payload["spatial_masks"] = spatial_masks
    return payload


def atomic_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    train_dir = args.diamond_train_dir.expanduser().resolve()
    out_dir = args.output_train_dir.expanduser()
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"{out_dir} exists; pass --force to overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_episode_files(train_dir)
    lengths: list[int] = []
    for idx, src_path in enumerate(files):
        src = load_torch(src_path)
        dst = convert_episode(src)
        atomic_save(dst, out_dir / f"{idx}.pt")
        lengths.append(int(len(dst["actions"])))
        print(f"{idx:04d} {src_path} -> {out_dir / f'{idx}.pt'} T={lengths[-1]}", flush=True)

    metadata = {
        "format": "simulus_offline_episode_v1",
        "source_format": "diamond_dataset_standard",
        "source_train_dir": str(train_dir),
        "num_episodes": len(lengths),
        "num_steps": int(sum(lengths)),
        "lengths": lengths,
        "mask_keys": list(MASK_KEYS),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
