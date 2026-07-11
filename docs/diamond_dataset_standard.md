# DIAMOND Atari Dataset Standard

This document records the DIAMOND-format offline Atari dataset contract used for
the paper experiments. Converted datasets should follow this standard before
being exported to Dreamer, Simulus, TWISTER, or OC-STORM native formats.

## Episode Semantics

- One `.pt` file is one environment episode buffer.
- For Pong, one complete episode is one real ALE Pong game.
- A reset observation after `end=1` or `trunc=1` belongs to the next `.pt`
  episode. It must not be appended to the previous episode.
- A fixed step-budget collection may end with one incomplete tail episode. Such a
  tail has no terminal marker and is only allowed at the end of a split.

## Transition Alignment

Each timestep stores the transition sampled by DIAMOND's collector:

```text
obs_t, action_t, reward_t, end_t, trunc_t
```

`obs_t` is the observation before executing `action_t`. `reward_t`, `end_t`, and
`trunc_t` are the environment outputs after the action. If the action ends the
episode, the terminal reward and `end_t=1` are stored on the same final timestep.

## Episode File Fields

Each episode file is a PyTorch payload:

```text
obs:   uint8,   shape (T, 3, 64, 64), pixel range [0, 255]
act:   int64,   shape (T,)
rew:   float32, shape (T,)
end:   uint8,   shape (T,), values 0/1
trunc: uint8,   shape (T,), values 0/1
info:  dict
```

Optional mask fields may be present and must be sliced with the same episode
boundaries:

```text
mask1: uint8, shape (T, 64, 64)
mask2: uint8, shape (T, 64, 64)
mask3: uint8, shape (T, 64, 64)
important_event_indicator: uint8, shape (T,)
```

## Metadata

Each split directory contains `info.pt` with at least:

```text
lengths: list[int]
counter_rew: collections.Counter
counter_end: collections.Counter
game_name: str
```

Converted datasets may include extra metadata keys, but downstream tools should
not depend on them unless explicitly documented.

## Pong-Specific Validation

For every complete Pong episode:

```text
-21 <= sum(rew) <= 21
end.sum() + trunc.sum() == 1
end[-1] == 1 or trunc[-1] == 1
```

The complete-game terminal condition is score based:

```text
count(rew > 0) == 21 or count(rew < 0) == 21
```

When repairing a replay segment that contains multiple Pong games, split at the
timestep where either positive or negative reward count reaches 21. The split
timestep remains in the preceding episode and is marked `end=1`; the following
timestep starts the next episode.

