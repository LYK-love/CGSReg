#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


FORMATS = ("dreamer", "diamond", "simulus", "storm")
MASK_KEYS = ("mask1", "mask2", "mask3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TWISTER native ReplayBuffer shards into other WM project dataset formats."
    )
    parser.add_argument(
        "--twister-replay-dir",
        type=Path,
        required=True,
        help="TWISTER ReplayBuffer directory containing *.torch shards.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root. Format-specific subdirectories are written under this root.",
    )
    parser.add_argument(
        "--formats",
        default=",".join(FORMATS),
        help=f"Comma-separated output formats. Choices: {','.join(FORMATS)}.",
    )
    parser.add_argument(
        "--dreamer-root",
        type=Path,
        default=Path("~/projects/dreamerv3-reborn"),
        help="Dreamer repo root, needed only when writing Dreamer replay chunks.",
    )
    parser.add_argument("--chunk-size", type=int, default=1024, help="Dreamer replay chunk size.")
    parser.add_argument(
        "--mask-mode",
        choices=("zeros", "motion", "none"),
        default="none",
        help="Mask fields to write. `motion` derives mask1 from consecutive frame differences.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Resize images and masks to this square size. Use 0 to preserve input size.",
    )
    parser.add_argument("--game-name", default="Pong", help="DIAMOND info.pt game_name field.")
    parser.add_argument("--limit-steps", type=int, default=0, help="Optional max model steps to convert.")
    parser.add_argument("--force", action="store_true", help="Overwrite output root if it exists.")
    return parser.parse_args()


def _parse_formats(value: str) -> list[str]:
    formats = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(formats) - set(FORMATS))
    if unknown:
        raise SystemExit(f"Unknown --formats values: {unknown}; expected any of {FORMATS}")
    return formats


def _find_shards(replay_dir: Path) -> list[Path]:
    shards = sorted(replay_dir.glob("*.torch"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)
    if not shards:
        raise FileNotFoundError(f"No TWISTER ReplayBuffer *.torch shards found under {replay_dir}")
    return shards


def _stack_or_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        return torch.stack([torch.as_tensor(x) for x in value], dim=0)
    return torch.as_tensor(value)


def _image_chw_uint8(value: Any, image_size: int | None) -> torch.Tensor:
    image = torch.as_tensor(value)
    if image.ndim != 3:
        raise ValueError(f"Expected image shape (C,H,W) or (H,W,C), got {tuple(image.shape)}")
    if image.shape[-1] in (1, 3, 4) and image.shape[0] not in (1, 3, 4):
        image = image.permute(2, 0, 1)
    if image.dtype != torch.uint8:
        image = image.float()
        if image.numel() and image.min() < 0:
            image = (image + 1.0) / 2.0
        if image.numel() and image.max() <= 1.5:
            image = image * 255.0
        image = image.clamp(0, 255).to(torch.uint8)
    image = image[:3].contiguous()
    if image_size is not None and tuple(image.shape[-2:]) != (image_size, image_size):
        image = F.interpolate(image[None].float(), size=(image_size, image_size), mode="bilinear", align_corners=False)[0]
        image = image.clamp(0, 255).to(torch.uint8)
    return image.contiguous()


def _action_index(value: Any) -> int:
    action = torch.as_tensor(value)
    if action.ndim > 0 and action.numel() > 1:
        return int(action.float().argmax().item())
    return int(action.reshape(-1)[0].item())


def _bool_scalar(value: Any) -> bool:
    return bool(torch.as_tensor(value).reshape(-1)[0].item())


def _float_scalar(value: Any) -> float:
    return float(torch.as_tensor(value).reshape(-1)[0].item())


def _step_dict_from_window(window: list[Any], index: int, image_size: int | None) -> dict[str, Any]:
    states = _stack_or_tensor(window[0])
    actions = _stack_or_tensor(window[1])
    rewards = _stack_or_tensor(window[2])
    dones = _stack_or_tensor(window[3])
    firsts = _stack_or_tensor(window[4])
    model_steps = _stack_or_tensor(window[5])
    return {
        "image": _image_chw_uint8(states[index], image_size),
        "action": _action_index(actions[index]),
        "reward": _float_scalar(rewards[index]),
        "done": _bool_scalar(dones[index]),
        "is_first": _bool_scalar(firsts[index]),
        "model_step": int(model_steps[index].item()),
    }


def _load_twister_steps(replay_dir: Path, image_size: int | None, limit_steps: int) -> dict[int, dict[str, Any]]:
    steps: dict[int, dict[str, Any]] = {}
    duplicate_mismatches = 0
    total_windows = 0
    total_window_steps = 0
    for shard in _find_shards(replay_dir):
        payload = torch.load(shard, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict payload in {shard}, got {type(payload)}")
        for _, window in sorted(payload.items(), key=lambda item: int(item[0])):
            if not isinstance(window, (list, tuple)) or len(window) < 6:
                raise ValueError(f"Invalid TWISTER ReplayBuffer window in {shard}")
            total_windows += 1
            model_steps = _stack_or_tensor(window[5])
            length = len(model_steps)
            total_window_steps += length
            for idx in range(length):
                model_step = int(model_steps[idx].item())
                if model_step in steps:
                    continue
                if limit_steps > 0 and model_step >= limit_steps:
                    continue
                step = _step_dict_from_window(list(window), idx, image_size)
                existing = steps.get(model_step)
                if existing is None:
                    steps[model_step] = step
                elif (
                    existing["action"] != step["action"]
                    or existing["done"] != step["done"]
                    or abs(existing["reward"] - step["reward"]) > 1e-6
                    or existing["image"].shape != step["image"].shape
                    or not torch.equal(existing["image"], step["image"])
                ):
                    duplicate_mismatches += 1
        print(f"[load] {shard.name}: windows={len(payload)} unique_steps={len(steps)}", flush=True)
    if not steps:
        raise ValueError(f"No model steps loaded from {replay_dir}")
    if duplicate_mismatches:
        print(f"[WARN] Duplicate model_step payload mismatches: {duplicate_mismatches}", flush=True)
    print(
        f"[load] windows={total_windows} window_steps={total_window_steps} "
        f"unique_steps={len(steps)} range={min(steps)}..{max(steps)}",
        flush=True,
    )
    return steps


def _split_episodes(steps: dict[int, dict[str, Any]]) -> list[list[dict[str, Any]]]:
    episodes: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_step: int | None = None
    for model_step in sorted(steps):
        step = steps[model_step]
        starts_new = (
            not current
            or step["is_first"]
            or previous_step is None
            or model_step != previous_step + 1
        )
        if starts_new and current:
            episodes.append(current)
            current = []
        current.append(step)
        previous_step = model_step
        if step["done"]:
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    return episodes


def _derive_masks(episode: list[dict[str, Any]], mode: str) -> dict[str, torch.Tensor]:
    length = len(episode)
    if mode == "none":
        return {}
    shape = tuple(episode[0]["image"].shape[-2:])
    masks = torch.zeros(length, 3, *shape, dtype=torch.uint8)
    if mode == "zeros":
        return {key: masks[:, idx] for idx, key in enumerate(MASK_KEYS)}
    frames = torch.stack([step["image"][:3].float() for step in episode], dim=0)
    diff = torch.zeros(length, *shape, dtype=torch.float32)
    if length > 1:
        step_diff = (frames[1:] - frames[:-1]).abs().mean(dim=1)
        diff[1:] = torch.maximum(diff[1:], step_diff)
        diff[:-1] = torch.maximum(diff[:-1], step_diff)
    mask1 = (diff > 8.0).float()[:, None]
    mask1 = F.max_pool2d(mask1, kernel_size=3, stride=1, padding=1)
    masks[:, 0] = (mask1[:, 0] > 0.5).to(torch.uint8)
    return {key: masks[:, idx] for idx, key in enumerate(MASK_KEYS)}


def _diamond_episode_path(output_dir: Path, episode_id: int) -> Path:
    n = 3
    subfolders = []
    for i in range(n - 1, -1, -1):
        value = (episode_id % (10 ** (i + 1))) // (10 ** i) * (10 ** i)
        subfolders.append(f"{value:0{i + 1}d}")
    return output_dir / "/".join(subfolders) / f"{episode_id}.pt"


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _episode_tensors(episode: list[dict[str, Any]], mask_mode: str) -> dict[str, Any]:
    images = torch.stack([step["image"] for step in episode], dim=0)
    actions = torch.tensor([step["action"] for step in episode], dtype=torch.long)
    rewards = torch.tensor([step["reward"] for step in episode], dtype=torch.float32)
    dones = torch.tensor([step["done"] for step in episode], dtype=torch.uint8)
    is_first = torch.tensor([step["is_first"] for step in episode], dtype=torch.float32)
    if len(is_first) and not bool(is_first.any()):
        is_first[0] = 1.0
    return {
        "obs": images,
        "action": actions,
        "reward": rewards,
        "done": dones,
        "is_first": is_first,
        "masks": _derive_masks(episode, mask_mode),
    }


def _write_diamond(episodes: list[list[dict[str, Any]]], output_dir: Path, mask_mode: str, game_name: str) -> dict[str, Any]:
    train_dir = output_dir / "train"
    lengths: list[int] = []
    counter_rew = Counter()
    counter_end = Counter()
    for episode_id, episode in enumerate(episodes):
        data = _episode_tensors(episode, mask_mode)
        payload = {
            "obs": data["obs"],
            "act": data["action"],
            "rew": data["reward"],
            "end": data["done"],
            "trunc": torch.zeros_like(data["done"]),
            "info": {},
        }
        for key, value in data["masks"].items():
            payload[key] = value
        _atomic_torch_save(payload, _diamond_episode_path(train_dir, episode_id))
        lengths.append(len(episode))
        counter_rew.update(data["reward"].sign().to(torch.int64).tolist())
        counter_end.update(data["done"].to(torch.int64).tolist())
    starts = np.cumsum([0, *lengths[:-1]], dtype=np.int64)
    info = {
        "is_static": False,
        "num_episodes": len(lengths),
        "num_steps": int(sum(lengths)),
        "start_idx": starts,
        "lengths": lengths,
        "counter_rew": counter_rew,
        "counter_end": counter_end,
        "game_name": game_name,
    }
    _atomic_torch_save(info, train_dir / "info.pt")
    return {"path": str(train_dir), "num_episodes": len(lengths), "num_steps": int(sum(lengths))}


def _write_simulus(episodes: list[list[dict[str, Any]]], output_dir: Path, mask_mode: str) -> dict[str, Any]:
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    lengths: list[int] = []
    for episode_id, episode in enumerate(episodes):
        data = _episode_tensors(episode, mask_mode)
        payload = {
            "observations": {"image": data["obs"]},
            "actions": data["action"],
            "rewards": data["reward"],
            "ends": data["done"].long(),
            "mask_padding": torch.ones(len(episode), dtype=torch.bool),
            "last_info": {},
        }
        if data["masks"]:
            payload["spatial_masks"] = {key: value.to(torch.uint8) for key, value in data["masks"].items()}
        _atomic_torch_save(payload, train_dir / f"{episode_id}.pt")
        lengths.append(len(episode))
    meta = {
        "format": "simulus_offline_episode_v1",
        "num_episodes": len(lengths),
        "num_steps": int(sum(lengths)),
        "lengths": lengths,
        "mask_mode": mask_mode,
    }
    (train_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"path": str(train_dir), "num_episodes": len(lengths), "num_steps": int(sum(lengths))}


def _write_storm(episodes: list[list[dict[str, Any]]], output_dir: Path, mask_mode: str) -> dict[str, Any]:
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    lengths: list[int] = []
    for episode_id, episode in enumerate(episodes):
        data = _episode_tensors(episode, mask_mode)
        payload = {
            "obs": data["obs"],
            "action": data["action"],
            "reward": data["reward"],
            "termination": data["done"].float(),
            "info": {},
        }
        for key, value in data["masks"].items():
            payload[key] = value
        _atomic_torch_save(payload, train_dir / f"{episode_id}.pt")
        lengths.append(len(episode))
    meta = {
        "format": "oc_storm_offline_episode_v1",
        "num_episodes": len(lengths),
        "num_steps": int(sum(lengths)),
        "lengths": lengths,
        "mask_mode": mask_mode,
    }
    (train_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"path": str(train_dir), "num_episodes": len(lengths), "num_steps": int(sum(lengths))}


def _load_dreamer_chunklib(dreamer_root: Path):
    dreamer_root = dreamer_root.expanduser().resolve()
    sys.path.insert(0, str(dreamer_root))
    sys.path.insert(0, str(dreamer_root / "dreamerv3"))
    try:
        import elements  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Writing Dreamer chunks requires the Dreamer environment with `elements` installed."
        ) from exc
    chunk_path = dreamer_root / "embodied" / "core" / "chunk.py"
    spec = importlib.util.spec_from_file_location("dreamerv3_chunklib_for_twister_convert", chunk_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Dreamer chunk module from {chunk_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stepid(chunk_uuid, index: int) -> np.ndarray:
    raw = bytes(chunk_uuid) + int(index).to_bytes(4, "big")
    return np.frombuffer(raw, np.uint8)


def _write_dreamer(
    episodes: list[list[dict[str, Any]]],
    output_dir: Path,
    mask_mode: str,
    dreamer_root: Path,
    chunk_size: int,
) -> dict[str, Any]:
    chunklib = _load_dreamer_chunklib(dreamer_root)
    replay_dir = output_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    current = chunklib.Chunk(chunk_size)
    saved_chunks = 0
    steps_written = 0
    for episode in episodes:
        masks = _derive_masks(episode, mask_mode)
        for index, step in enumerate(episode):
            item = {
                "image": step["image"].permute(1, 2, 0).cpu().numpy().astype(np.uint8),
                "action": np.int32(step["action"]),
                "reward": np.float32(step["reward"]),
                "is_first": np.bool_(index == 0),
                "is_last": np.bool_(step["done"]),
                "is_terminal": np.bool_(step["done"]),
            }
            for key, value in masks.items():
                item[key] = value[index].cpu().numpy().astype(np.uint8)
            item["stepid"] = _stepid(current.uuid, current.length)
            current.append(item)
            steps_written += 1
            if current.length == current.size:
                nxt = chunklib.Chunk(chunk_size)
                current.succ = nxt.uuid
                current.save(replay_dir)
                saved_chunks += 1
                current = nxt
    if current.length:
        current.save(replay_dir)
        saved_chunks += 1
    return {
        "path": str(replay_dir),
        "num_episodes": len(episodes),
        "num_steps": steps_written,
        "num_chunks": saved_chunks,
    }


def _write_manifest(
    output_root: Path,
    *,
    source: Path,
    formats: Iterable[str],
    mask_mode: str,
    image_size: int | None,
    episodes: list[list[dict[str, Any]]],
    outputs: dict[str, Any],
) -> None:
    lengths = [len(ep) for ep in episodes]
    rewards = Counter()
    terminals = Counter()
    for episode in episodes:
        rewards.update(int(np.sign(step["reward"])) for step in episode)
        terminals.update(int(step["done"]) for step in episode)
    manifest = {
        "format": "twister_replay_converted_dataset_v1",
        "source": str(source),
        "formats": list(formats),
        "mask_mode": mask_mode,
        "image_size": image_size,
        "num_episodes": len(episodes),
        "num_steps": int(sum(lengths)),
        "lengths": lengths,
        "reward_sign_counts": dict(rewards),
        "terminal_counts": dict(terminals),
        "outputs": outputs,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    replay_dir = args.twister_replay_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    formats = _parse_formats(args.formats)
    image_size = None if args.image_size == 0 else int(args.image_size)
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"Output root already exists: {output_root}. Use --force to overwrite.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    steps = _load_twister_steps(replay_dir, image_size, int(args.limit_steps))
    episodes = _split_episodes(steps)
    lengths = [len(ep) for ep in episodes]
    print(
        f"[episodes] count={len(episodes)} steps={sum(lengths)} "
        f"min={min(lengths)} max={max(lengths)}",
        flush=True,
    )

    outputs: dict[str, Any] = {}
    if "dreamer" in formats:
        target = output_root if formats == ["dreamer"] else output_root / "dreamer"
        outputs["dreamer"] = _write_dreamer(
            episodes, target, args.mask_mode, args.dreamer_root, int(args.chunk_size)
        )
        print(f"[dreamer] {outputs['dreamer']}", flush=True)
    if "diamond" in formats:
        target = output_root if formats == ["diamond"] else output_root / "diamond"
        outputs["diamond"] = _write_diamond(episodes, target, args.mask_mode, args.game_name)
        print(f"[diamond] {outputs['diamond']}", flush=True)
    if "simulus" in formats:
        target = output_root if formats == ["simulus"] else output_root / "simulus"
        outputs["simulus"] = _write_simulus(episodes, target, args.mask_mode)
        print(f"[simulus] {outputs['simulus']}", flush=True)
    if "storm" in formats:
        target = output_root if formats == ["storm"] else output_root / "storm"
        outputs["storm"] = _write_storm(episodes, target, args.mask_mode)
        print(f"[storm] {outputs['storm']}", flush=True)

    _write_manifest(
        output_root,
        source=replay_dir,
        formats=formats,
        mask_mode=args.mask_mode,
        image_size=image_size,
        episodes=episodes,
        outputs=outputs,
    )
    print(f"[done] manifest={output_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
