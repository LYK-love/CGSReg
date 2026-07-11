"""Adapter for DIAMOND's public Atari100K actor-critic checkpoints."""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECTS_ROOT = ROOT.parent


def default_diamond_root() -> pathlib.Path:
  import os

  value = os.environ.get('DIAMOND_ROOT')
  if value:
    return pathlib.Path(value).expanduser().resolve()
  return (PROJECTS_ROOT / 'diamond').resolve()


@dataclass
class DiamondPublicPolicy:
  """Small policy wrapper with the same ``act`` API as rl-in-pixel-env policies."""

  actor_critic: object
  device: object
  deterministic: bool = True
  hx_cx: object | None = None

  def reset(self):
    self.hx_cx = None

  def act(self, obs):
    import torch

    with torch.no_grad():
      logits, value, self.hx_cx = self.actor_critic.predict_act_value(obs, self.hx_cx)
      if self.deterministic:
        action = torch.argmax(logits, dim=-1)
        entropy = torch.distributions.Categorical(logits=logits).entropy() / math.log(2)
      else:
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        entropy = dist.entropy() / math.log(2)
      return action, value, {'entropy': entropy.detach()}


def _conv1x1(in_channels: int, out_channels: int):
  import torch.nn as nn

  return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


def _conv3x3(in_channels: int, out_channels: int):
  import torch.nn as nn

  return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


class _SmallResBlock:
  def __new__(cls, in_channels: int, out_channels: int):
    import torch.nn as nn

    class _Block(nn.Module):
      def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            _NamedGroupNorm(in_channels),
            nn.SiLU(inplace=True),
            _conv3x3(in_channels, out_channels))
        self.skip_projection = (
            nn.Identity() if in_channels == out_channels else _conv1x1(in_channels, out_channels))

      def forward(self, x):
        return self.skip_projection(x) + self.f(x)

    return _Block()


class _NamedGroupNorm:
  """DIAMOND's GroupNorm wrapper, preserving ``*.norm.*`` state keys."""

  def __init__(self, in_channels: int):
    import torch.nn as nn

    class _Module(nn.Module):
      def __init__(self):
        super().__init__()
        self.norm = nn.GroupNorm(max(1, in_channels // 32), in_channels, eps=1e-5)

      def forward(self, x):
        return self.norm(x)

    self.module = _Module()

  def __new__(cls, in_channels: int):
    import torch.nn as nn

    class _Module(nn.Module):
      def __init__(self):
        super().__init__()
        self.norm = nn.GroupNorm(max(1, in_channels // 32), in_channels, eps=1e-5)

      def forward(self, x):
        return self.norm(x)

    return _Module()


class _ActorCriticEncoder:
  def __new__(cls):
    import torch.nn as nn

    class _Encoder(nn.Module):
      def __init__(self):
        super().__init__()
        channels = [32, 32, 64, 64]
        down = [1, 1, 1, 1]
        layers = [_conv3x3(3, channels[0])]
        for i, out_channels in enumerate(channels):
          in_channels = channels[max(0, i - 1)]
          layers.append(_SmallResBlock(in_channels, out_channels))
          if down[i]:
            layers.append(nn.MaxPool2d(2))
        self.encoder = nn.Sequential(*layers)

      def forward(self, x):
        return self.encoder(x)

    return _Encoder()


class _DiamondActorCritic:
  def __new__(cls, num_actions: int):
    import torch.nn as nn

    class _ActorCritic(nn.Module):
      def __init__(self):
        super().__init__()
        self.encoder = _ActorCriticEncoder()
        self.lstm = nn.LSTMCell(1024, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, int(num_actions))

      def predict_act_value(self, obs, hx_cx):
        assert obs.ndim == 4
        x = self.encoder(obs).flatten(start_dim=1)
        hx, cx = self.lstm(x, hx_cx)
        return self.actor_linear(hx), self.critic_linear(hx).squeeze(dim=1), (hx, cx)

    return _ActorCritic()


def load_diamond_public_policy(
    checkpoint: str | pathlib.Path,
    *,
    num_actions: int,
    device: str,
    deterministic: bool = True,
    diamond_root: str | pathlib.Path | None = None,
):
  """Load the actor-critic from a DIAMOND checkpoint such as ``checkpoints/Pong.pt``."""

  import torch

  del diamond_root  # The adapter is self-contained and does not import DIAMOND code.
  checkpoint = pathlib.Path(checkpoint).expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f'DIAMOND policy checkpoint not found: {checkpoint}')

  ckpt = torch.load(checkpoint, weights_only=False, map_location=torch.device(device))
  actor_critic = _DiamondActorCritic(int(num_actions))
  actor_critic.load_state_dict(_extract_actor_critic_state_dict(ckpt))
  actor_critic = actor_critic.to(torch.device(device)).eval()

  return DiamondPublicPolicy(
      actor_critic=actor_critic,
      device=torch.device(device),
      deterministic=bool(deterministic))


def _looks_like_state_dict(value) -> bool:
  import torch

  return (
      isinstance(value, dict)
      and bool(value)
      and all(isinstance(key, str) for key in value)
      and any(torch.is_tensor(item) for item in value.values()))


def _extract_actor_critic_state_dict(ckpt):
  if not isinstance(ckpt, dict):
    raise TypeError(f'Expected checkpoint mapping, got {type(ckpt).__name__}.')
  if 'actor_critic' in ckpt and _looks_like_state_dict(ckpt['actor_critic']):
    return ckpt['actor_critic']
  if 'policy' in ckpt and _looks_like_state_dict(ckpt['policy']):
    return ckpt['policy']
  prefixed = {}
  for key, value in ckpt.items():
    if isinstance(key, str) and key.startswith('actor_critic.'):
      prefixed[key[len('actor_critic.'):]] = value
  if prefixed:
    return prefixed
  if _looks_like_state_dict(ckpt):
    return ckpt
  keys = ', '.join(str(key) for key in list(ckpt)[:10])
  raise KeyError(
      'Could not find actor-critic weights in DIAMOND checkpoint. '
      f'Top-level keys: {keys}')
