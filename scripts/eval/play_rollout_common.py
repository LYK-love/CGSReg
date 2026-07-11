"""Shared helpers for headless WM rollout scripts."""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECTS_ROOT = ROOT.parent


def project_root_from_env(env_name: str, *candidates: str) -> pathlib.Path:
  env_value = os.environ.get(env_name)
  if env_value:
    return pathlib.Path(env_value)
  for candidate in candidates:
    path = PROJECTS_ROOT / candidate
    if path.exists():
      return path
  return PROJECTS_ROOT / candidates[0]


@dataclass
class PixelRLContext:
  env_name: str
  seed: int
  num_envs: int
  device: str
  wm_checkpoint: str
  wm_horizon: int
  wm_reward_quantize_threshold: float = 0.5
  wm_respect_terminal: bool = False
  extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WMSpec:
  project: str
  name: str
  checkpoint: pathlib.Path
  project_root: pathlib.Path | None = None


PROJECT_PATHS = {
    'diamond': {
        'root': pathlib.Path(os.environ.get('DIAMOND_ROOT', PROJECTS_ROOT / 'diamond')),
        'pythonpath': ('src', 'src/wm_play/src', 'third_party/rl-in-pixel-env/src'),
        'import': 'pixel_rl.adapter',
    },
    'simulus': {
        'root': project_root_from_env('SIMULUS_ROOT', 'simulus', 'Simulus'),
        'pythonpath': ('src', 'third_party/rl-in-pixel-env/src'),
        'import': 'pixel_rl.adapter',
    },
    'twister': {
        'root': pathlib.Path(os.environ.get('TWISTER_ROOT', PROJECTS_ROOT / 'twister')),
        'pythonpath': ('.', 'third_party/rl-in-pixel-env/src'),
        'import': 'pixel_rl.adapter',
    },
    'storm': {
        'root': pathlib.Path(os.environ.get('STORM_ROOT', PROJECTS_ROOT / 'oc-storm')),
        'pythonpath': ('.', 'third_party/rl-in-pixel-env/src'),
        'import': 'pixel_rl.adapter',
    },
}


def ensure_dir(path: pathlib.Path):
  path.mkdir(parents=True, exist_ok=True)


def write_json(path: pathlib.Path, data: Any):
  ensure_dir(path.parent)
  path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def save_video(path: pathlib.Path, frames: np.ndarray, fps: int):
  ensure_dir(path.parent)
  frames = np.asarray(frames, np.uint8)
  try:
    import imageio.v2 as iio

    iio.mimsave(path, list(frames), fps=int(fps), macro_block_size=1)
  except Exception:
    import imageio.v3 as iio

    iio.imwrite(path, frames, fps=int(fps), codec='libx264')


def parse_scalar(raw: str) -> Any:
  lowered = raw.lower()
  if lowered in ('true', 'false'):
    return lowered == 'true'
  try:
    return int(raw)
  except ValueError:
    try:
      return float(raw)
    except ValueError:
      return raw


def parse_extra(values: list[str]) -> dict[str, Any]:
  out: dict[str, Any] = {}
  for value in values:
    if '=' not in value:
      raise ValueError(f'--extra values must be key=value, got {value!r}')
    key, raw = value.split('=', 1)
    out[key] = parse_scalar(raw)
  return out


def infer_name(path: str | pathlib.Path) -> str:
  path = pathlib.Path(path)
  if path.name and path.name != 'latest':
    return path.name
  for parent in path.parents:
    if parent.name not in {'ckpt', 'checkpoints', 'agent_versions'}:
      return parent.name
  return 'wm'


def parse_project_wm(value: str) -> WMSpec:
  if ':' not in value:
    raise ValueError('--wm must look like project:name=/path/to/ckpt or project:/path/to/ckpt.')
  project, rest = value.split(':', 1)
  if '=' in rest:
    name, path = rest.split('=', 1)
  else:
    path = rest
    name = infer_name(path)
  if project not in PROJECT_PATHS:
    raise ValueError(f'Unknown project {project!r}; expected one of {sorted(PROJECT_PATHS)}.')
  return WMSpec(project=project, name=name, checkpoint=pathlib.Path(path).expanduser().resolve())


def add_project_paths(project: str, project_root: pathlib.Path | None = None):
  spec = PROJECT_PATHS[project]
  root = pathlib.Path(project_root or spec['root']).expanduser().resolve()
  if not root.exists():
    raise FileNotFoundError(f'{project} root does not exist: {root}')
  for rel in reversed(spec['pythonpath']):
    path = root / rel
    if path.exists():
      value = str(path.resolve())
      if value not in sys.path:
        sys.path.insert(0, value)
  common = PROJECTS_ROOT / 'wm-play-common' / 'src'
  if common.exists():
    value = str(common.resolve())
    if value not in sys.path:
      sys.path.insert(0, value)
  if project == 'storm':
    import ale_py  # noqa: F401
  return root, spec['import']


def clear_pixel_rl_modules():
  for key in list(sys.modules):
    if key == 'pixel_rl' or key.startswith('pixel_rl.'):
      del sys.modules[key]


def import_make_torch_wm_env(
    project: str,
    project_root: pathlib.Path | None = None,
    *,
    clear_project_modules: bool = True,
):
  root, module_name = add_project_paths(project, project_root)
  cwd = pathlib.Path.cwd()
  os.chdir(root)
  try:
    if clear_project_modules:
      clear_pixel_rl_modules()
    module = __import__(module_name, fromlist=['make_torch_wm_env'])
  finally:
    os.chdir(cwd)
  return root, module.make_torch_wm_env


def require_torch():
  import torch

  return torch


def resolve_policy_checkpoint(path: str | pathlib.Path) -> pathlib.Path:
  root = pathlib.Path(path).expanduser()
  if root.is_file():
    return root.resolve()
  for rel in ('latest.pt', 'pixel_rl_ckpt/latest.pt', 'policy_ckpt/latest.pt', 'ckpt/latest.pt'):
    candidate = root / rel
    if candidate.is_file():
      return candidate.resolve()
  candidates = [p for p in root.rglob('*.pt') if p.is_file()]
  if not candidates:
    raise FileNotFoundError(f'No policy .pt checkpoint found under {root}')

  def key(item: pathlib.Path):
    match = re.search(r'(?:update|epoch)_?0*(\d+)', item.stem)
    return (1, int(match.group(1)), item.stat().st_mtime) if match else (0, -1, item.stat().st_mtime)

  return max(candidates, key=key).resolve()


def load_pixel_rl_policy(checkpoint: pathlib.Path, *, num_actions: int, device: str, deterministic: bool):
  torch = require_torch()
  try:
    from diamond_rl_env.actor_critic import ActorCriticConfig
    from diamond_rl_env.policy import load_actor_critic_policy
  except ModuleNotFoundError:
    rl_env_src = PROJECTS_ROOT / 'rl-in-pixel-env' / 'src'
    if rl_env_src.exists():
      sys.path.insert(0, str(rl_env_src.resolve()))
    from diamond_rl_env.actor_critic import ActorCriticConfig
    from diamond_rl_env.policy import load_actor_critic_policy

  cfg = ActorCriticConfig(num_actions=int(num_actions))
  return load_actor_critic_policy(
      checkpoint,
      cfg=cfg,
      device=torch.device(device),
      deterministic=deterministic,
      module_name='policy')


def obs_to_uint8_frame(obs: Any, size: int = 0) -> np.ndarray:
  torch = require_torch()
  if isinstance(obs, dict):
    obs = obs.get('image', obs)
  if torch.is_tensor(obs):
    arr = obs.detach().cpu()
    while arr.ndim > 3:
      arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
      arr = arr.permute(1, 2, 0)
    arr = arr.float().numpy()
  else:
    arr = np.asarray(obs)
    while arr.ndim > 3:
      arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
      arr = np.moveaxis(arr, 0, -1)
  if arr.ndim == 2:
    arr = np.repeat(arr[..., None], 3, axis=-1)
  if arr.ndim != 3:
    raise ValueError(f'Cannot render observation with shape {arr.shape}')
  if arr.shape[-1] == 1:
    arr = np.repeat(arr, 3, axis=-1)
  if arr.shape[-1] > 3:
    arr = arr[..., :3]
  if arr.dtype != np.uint8:
    arr = arr.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size and finite.min() < -0.05:
      arr = (arr + 1.0) * 127.5
    elif finite.size and finite.max() <= 1.5:
      arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
  if size > 0 and arr.shape[:2] != (size, size):
    from PIL import Image

    arr = np.asarray(Image.fromarray(arr).resize((size, size), resample=Image.NEAREST), np.uint8)
  return arr


def obs_to_policy_tensor(obs: Any, device: str):
  torch = require_torch()
  frame = obs_to_uint8_frame(obs, size=64)
  tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).contiguous().float()
  tensor = tensor.to(torch.device(device), non_blocking=True)
  if tensor.numel() and tensor.max() > 1.5:
    tensor = tensor / 255.0
  return tensor.clamp(0, 1).mul(2).sub(1)


def action_tensor(action: int, device: str):
  torch = require_torch()
  return torch.tensor([int(action)], dtype=torch.long, device=torch.device(device))


def first_bool(value: Any) -> bool:
  torch = require_torch()
  return bool(torch.as_tensor(value).detach().cpu().reshape(-1)[0].item())


def first_float(value: Any) -> float:
  torch = require_torch()
  return float(torch.as_tensor(value).detach().cpu().reshape(-1)[0].item())


def prepare_checkpoint(project: str, checkpoint: str | pathlib.Path, output: pathlib.Path) -> pathlib.Path:
  torch = require_torch()
  path = pathlib.Path(checkpoint).expanduser().resolve()
  if project != 'storm':
    return path
  try:
    data = torch.load(path, map_location='cpu', weights_only=True)
  except TypeError:
    data = torch.load(path, map_location='cpu')
  if not isinstance(data, dict):
    return path
  keys = list(data.keys())
  if not any(str(key).startswith('world_model.') for key in keys):
    return path
  state = {}
  for key, value in data.items():
    key = str(key)
    if not key.startswith('world_model.'):
      continue
    key = key[len('world_model.'):]
    if key.startswith('state_decoder.'):
      key = 'image_decoder.' + key[len('state_decoder.'):]
    state[key] = value
  if not state:
    return path
  normalized = output.with_suffix('.storm_world_model.pth')
  normalized.parent.mkdir(parents=True, exist_ok=True)
  torch.save(state, normalized)
  return normalized.resolve()
