#!/usr/bin/env python3
"""Run OC-STORM's CUTIE Pong segmenter on rollout videos."""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECTS_ROOT = ROOT.parent


CLASS_NAMES = {
    1: 'right_paddle',
    2: 'left_paddle',
    3: 'ball',
}


@dataclass(frozen=True)
class VideoSpec:
  name: str
  path: pathlib.Path


def ensure_dir(path: pathlib.Path):
  path.mkdir(parents=True, exist_ok=True)


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]):
  ensure_dir(path.parent)
  keys = sorted({key for row in rows for key in row})
  with path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)


def parse_video(value: str) -> VideoSpec:
  if '=' in value:
    name, path = value.split('=', 1)
  else:
    path = value
    name = pathlib.Path(path).stem
  return VideoSpec(name=name, path=pathlib.Path(path).expanduser().resolve())


def videos_from_bundle(bundle_root: pathlib.Path) -> list[VideoSpec]:
  specs = []
  videos_root = bundle_root / 'videos'
  for path in sorted(videos_root.glob('*/*/*.mp4')):
    condition = path.parent.name
    project = path.parent.parent.name
    episode = path.stem
    specs.append(VideoSpec(
        name=f'{project}__{condition}__{episode}',
        path=path.resolve()))
  return specs


def to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
  arr = np.asarray(frame)
  if arr.ndim == 2:
    arr = np.repeat(arr[..., None], 3, axis=-1)
  if arr.shape[-1] > 3:
    arr = arr[..., :3]
  if arr.dtype != np.uint8:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
  return arr


def frame_to_atari_geometry(frame: np.ndarray, mode: str) -> np.ndarray:
  frame = to_uint8_rgb(frame)
  if mode == 'native':
    return frame
  if mode == 'square-to-atari':
    return np.asarray(
        Image.fromarray(frame).resize((160, 210), resample=Image.Resampling.NEAREST),
        dtype=np.uint8)
  raise ValueError(f'Unknown geometry mode: {mode}')


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
  colors = np.asarray([
      [0, 0, 0],
      [255, 0, 255],
      [0, 255, 0],
      [0, 192, 255],
  ], dtype=np.float32)
  frame = to_uint8_rgb(frame).astype(np.float32)
  mask = np.asarray(mask, np.int64)
  if mask.shape[:2] != frame.shape[:2]:
    mask = np.asarray(
        Image.fromarray(mask.astype(np.uint8)).resize(
            (frame.shape[1], frame.shape[0]), resample=Image.Resampling.NEAREST),
        dtype=np.int64)
  out = frame.copy()
  fg = mask > 0
  out[fg] = 0.20 * out[fg] + 0.80 * colors[np.clip(mask[fg], 0, len(colors) - 1)]
  return np.clip(out, 0, 255).astype(np.uint8)


def mask_rows(mask: np.ndarray, sample_id: str, t: int) -> list[dict[str, Any]]:
  rows = []
  for cls, name in CLASS_NAMES.items():
    ys, xs = np.nonzero(mask == cls)
    row = {
        'sample_id': sample_id,
        't': int(t),
        'class_id': int(cls),
        'object': name,
        'present': bool(xs.size),
        'area': int(xs.size),
    }
    if xs.size:
      row.update({
          'x': float(xs.mean()),
          'y': float(ys.mean()),
          'xmin': int(xs.min()),
          'xmax': int(xs.max()),
          'ymin': int(ys.min()),
          'ymax': int(ys.max()),
      })
    rows.append(row)
  return rows


def load_cutie(oc_storm_root: pathlib.Path):
  sys.path.insert(0, str(oc_storm_root))
  cwd = pathlib.Path.cwd()
  os.chdir(oc_storm_root)
  try:
    from feature_extractor.cutie.build_feature_extractor import load_cuite
    from feature_extractor.cutie.cutie.inference.inference_core import InferenceCore
    from feature_extractor.cutie.cuite_gui.interactive_utils import (
        image_to_torch,
        index_numpy_to_one_hot_torch,
        torch_prob_to_numpy_mask,
    )
    import cv2
    import torch
    from PIL import Image as PILImage
    cutie, cfg = load_cuite('small')
    processor = InferenceCore(cutie, cfg=cfg)
    label_folder = oc_storm_root / 'segmentation_masks/Atari/Pong'
    if not label_folder.exists():
      raise FileNotFoundError(f'Missing OC-STORM Pong masks: {label_folder}')
    with torch.inference_mode():
      with torch.amp.autocast('cuda', enabled=True):
        for mask_path in sorted((label_folder / 'masks').glob('*.png')):
          idx = mask_path.stem
          frame = cv2.imread(str(label_folder / 'imgs' / f'{idx}.png'))
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame = cv2.resize(frame, (320, 420), interpolation=cv2.INTER_NEAREST)
          frame_torch = image_to_torch(frame, device='cuda')
          mask = np.asarray(PILImage.open(mask_path))
          mask = cv2.resize(mask, (320, 420), interpolation=cv2.INTER_NEAREST)
          mask_torch = index_numpy_to_one_hot_torch(mask, len(CLASS_NAMES) + 1).cuda()
          processor.step(frame_torch, mask_torch[1:], idx_mask=False, force_permanent=True)
    return processor, image_to_torch, torch_prob_to_numpy_mask
  finally:
    os.chdir(cwd)


def cutie_outputs_are_current(video_path: pathlib.Path, out_dir: pathlib.Path) -> bool:
  outputs = [out_dir / 'cutie_tracks.csv', out_dir / 'cutie_overlay.mp4']
  if not all(path.exists() for path in outputs):
    return False
  video_mtime = video_path.stat().st_mtime
  return min(path.stat().st_mtime for path in outputs) >= video_mtime


def run_video(spec: VideoSpec, processor, image_to_torch, torch_prob_to_numpy_mask, args):
  import cv2
  import torch

  out_dir = args.output_dir / spec.name
  if args.skip_existing and cutie_outputs_are_current(spec.path, out_dir):
    print(f'Keeping existing {out_dir}', flush=True)
    return

  frames = [to_uint8_rgb(x) for x in iio.imread(spec.path)]
  if args.max_frames:
    frames = frames[:int(args.max_frames)]
  rows = []
  overlays = []
  masks = []
  processor.clear_non_permanent_memory()
  with torch.inference_mode():
    with torch.amp.autocast('cuda', enabled=True):
      for t, frame in enumerate(frames):
        atari_frame = frame_to_atari_geometry(frame, args.geometry)
        cutie_frame = cv2.resize(atari_frame, (320, 420), interpolation=cv2.INTER_NEAREST)
        prediction = processor.step(image_to_torch(cutie_frame, device='cuda'))
        mask = torch_prob_to_numpy_mask(prediction)
        mask_atari = np.asarray(
            Image.fromarray(mask.astype(np.uint8)).resize(
                (atari_frame.shape[1], atari_frame.shape[0]),
                resample=Image.Resampling.NEAREST),
            dtype=np.uint8)
        rows.extend(mask_rows(mask_atari, spec.name, t))
        overlays.append(overlay_mask(atari_frame, mask_atari))
        if args.save_masks:
          masks.append(mask_atari)
  ensure_dir(out_dir)
  write_csv(out_dir / 'cutie_tracks.csv', rows)
  if args.save_masks:
    np.savez_compressed(
        out_dir / 'cutie_masks.npz',
        masks=np.asarray(masks, dtype=np.uint8),
        class_ids=np.asarray(sorted(CLASS_NAMES), dtype=np.int64),
        class_names=np.asarray([CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]))
  iio.imwrite(
      out_dir / 'cutie_overlay.mp4',
      np.asarray(overlays, np.uint8),
      fps=args.fps,
      macro_block_size=1)
  print(f'Wrote {out_dir}', flush=True)


def main(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--video', action='append', default=[],
                      help='name=/path/to/video.mp4. Repeatable.')
  parser.add_argument('--bundle-root', type=pathlib.Path, default=None,
                      help='Optional rollout bundle root with videos/<project>/<condition>/<episode>.mp4.')
  parser.add_argument('--output-dir', type=pathlib.Path,
                      default=ROOT / 'eval_outputs/oc_storm_cutie_pong_smoke')
  parser.add_argument('--oc-storm-root', type=pathlib.Path,
                      default=PROJECTS_ROOT / 'oc-storm')
  parser.add_argument('--geometry', choices=('square-to-atari', 'native'), default='square-to-atari')
  parser.add_argument('--max-frames', type=int, default=0)
  parser.add_argument('--fps', type=int, default=15)
  parser.add_argument('--save-masks', action='store_true',
                      help='Also save per-frame class masks to cutie_masks.npz next to cutie_tracks.csv.')
  parser.add_argument('--skip-existing', action=argparse.BooleanOptionalAction, default=True,
                      help='Skip videos whose cutie_tracks.csv and cutie_overlay.mp4 exist and are newer than the source video.')
  args = parser.parse_args(argv)
  specs = [parse_video(x) for x in args.video]
  if args.bundle_root:
    specs.extend(videos_from_bundle(args.bundle_root.expanduser().resolve()))
  if not specs:
    raise ValueError('Provide at least one --video or a --bundle-root with videos.')
  processor, image_to_torch, torch_prob_to_numpy_mask = load_cutie(
      args.oc_storm_root.expanduser().resolve())
  ensure_dir(args.output_dir)
  for spec in specs:
    run_video(spec, processor, image_to_torch, torch_prob_to_numpy_mask, args)


if __name__ == '__main__':
  main()
