"""Stable-Baselines3 Atari policy wrapper for 64x64 RGB WM rollouts."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


class SB3AtariPolicy:
  """Expose an SB3 Atari frame-stack policy through the pixel-RL policy API.

  The public ``act()`` method accepts the same 64x64 RGB tensor used by the
  existing rollout scripts. Internally, it converts frames to the RL-Zoo Atari
  convention expected by ``sb3/ppo-PongNoFrameskip-v4``: grayscale 84x84 with a
  4-frame channel-first stack.
  """

  def __init__(self, checkpoint: str | Path, *, device: str = "cpu", deterministic: bool = True):
    import torch

    self.device = torch.device(device)
    self.deterministic = bool(deterministic)
    self.model = _PlainSB3PongPolicy().to(self.device).eval()
    try:
      data = torch.load(Path(checkpoint).expanduser(), map_location=self.device, weights_only=True)
    except TypeError:
      data = torch.load(Path(checkpoint).expanduser(), map_location=self.device)
    state = data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data
    self.model.load_state_dict(state, strict=False)
    self.frames = deque(maxlen=4)

  def reset(self):
    self.frames.clear()

  def act(self, image):
    import torch

    frame = self._tensor_to_gray84(image)
    if not self.frames:
      for _ in range(4):
        self.frames.append(frame)
    else:
      self.frames.append(frame)
    obs = torch.from_numpy(np.stack(list(self.frames), axis=0)[None]).to(self.device).float().div(255.0)
    with torch.no_grad():
      logits, value = self.model(obs)
      if self.deterministic:
        action_tensor = torch.argmax(logits, dim=-1)
      else:
        action_tensor = torch.distributions.Categorical(logits=logits).sample()
    value_tensor = value.reshape(-1)
    return action_tensor, value_tensor, {}

  @staticmethod
  def _tensor_to_gray84(image) -> np.ndarray:
    import torch

    if isinstance(image, dict):
      if "image" in image:
        image = image["image"]
      else:
        for key, value in image.items():
          if str(key).endswith("image") or getattr(key, "name", None) == "image":
            image = value
            break
    if torch.is_tensor(image):
      arr = image.detach().cpu()
      while arr.ndim > 3:
        arr = arr[0]
      if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.permute(1, 2, 0)
      arr = arr.float().numpy()
    else:
      arr = np.asarray(image)
      while arr.ndim > 3:
        arr = arr[0]
      if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
      arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] > 3:
      arr = arr[..., :3]
    if arr.dtype != np.uint8:
      finite = arr[np.isfinite(arr)]
      if finite.size and finite.min() < -0.05:
        arr = (arr + 1.0) * 127.5
      elif finite.size and finite.max() <= 1.5:
        arr = arr * 255.0
      arr = np.clip(arr, 0, 255).astype(np.uint8)
    gray = np.asarray(Image.fromarray(arr).convert("L").resize((84, 84), resample=Image.BILINEAR))
    return gray.astype(np.uint8, copy=False)


def load_sb3_atari_policy(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    deterministic: bool = True,
) -> SB3AtariPolicy:
  return SB3AtariPolicy(checkpoint, device=device, deterministic=deterministic)


class _PlainSB3PongPolicy:
  def __new__(cls):
    import torch.nn as nn

    class _NatureCNN(nn.Module):
      def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())

      def forward(self, x):
        return self.linear(self.cnn(x))

    class _Policy(nn.Module):
      def __init__(self):
        super().__init__()
        self.features_extractor = _NatureCNN()
        self.action_net = nn.Linear(512, 6)
        self.value_net = nn.Linear(512, 1)

      def forward(self, x):
        features = self.features_extractor(x)
        return self.action_net(features), self.value_net(features)

      def load_state_dict(self, state_dict, strict=True):
        filtered = {}
        for key, value in state_dict.items():
          if key.startswith("features_extractor.") or key.startswith("action_net.") or key.startswith("value_net."):
            filtered[key] = value
        return super().load_state_dict(filtered, strict=True)

    return _Policy()
