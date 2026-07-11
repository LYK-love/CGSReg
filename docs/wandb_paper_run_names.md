# W&B Paper Run Names

This is the shared W&B naming registry for the WM projects and their
zero-shot RL runs. Local checkpoint directories and archived artifact names are
not renamed here; they remain stable restore paths.

## Naming Convention

Use compact, searchable display names:

- `paper-wm-<wm>-<source>-<variant>` for world-model/SIGReg training runs.
- `paper-zsrl-<wm>-<source>-<variant>-h<horizon>` for zero-shot RL runs in a
  frozen WM. Omit default reward-quantization settings such as `rewq0p1`; keep
  `rewq...` only for runs that are specifically part of a reward-threshold
  sweep.
- Use these WM slugs: `dreamer`, `diamond`, `twister`, `simulus`, `storm`.
- Use these source slugs when applicable: `repro`, `offline`, `twexp`,
  `diamond-static`, `offline-ac-cpc`.
- Encode decimal weights with `p`, for example `w0p01` instead of `w0.01`.
- Keep display names short. Put long local paths, machine names, checkpoint
  paths, seeds, and scheduler details in config, notes, tags, or artifacts.

Examples:

- `paper-wm-diamond-twexp-dilated-k3-w0p01`
- `paper-zsrl-twister-twexp-w1-h512`
- `paper-zsrl-simulus-offline-w0p01-h512`

The current W&B inventory and cleanup review files are generated under
`artifacts/wandb_run_inventory/`. Candidate obsolete runs that should not be
deleted without human confirmation are summarized in
`docs/wandb_run_cleanup_candidates.md`.

## Zero-Shot RL Runs

| Project | Run id | Display name |
| --- | --- | --- |
| `ssl-lab/rl-in-pixel-env-dreamer` | `qa32kfzt` | `paper-zsrl-dreamer-repro-size200m-h512` |
| `ssl-lab/rl-in-pixel-env-dreamer` | `2xw3qk3y` | `paper-zsrl-dreamer-size200m-w0-h512` |
| `ssl-lab/rl-in-pixel-env-dreamer` | `2i3sz2l2` | `paper-zsrl-dreamer-size200m-w0p001-h512` |
| `ssl-lab/rl-in-pixel-env-dreamer` | `maqe4hy1` | `paper-zsrl-dreamer-size200m-w0p01-h512` |
| `ssl-lab/rl-in-pixel-env-dreamer` | `y00l3khw` | `paper-zsrl-dreamer-size200m-w0p1-h512` |
| `ssl-lab/rl-in-pixel-env-dreamer` | `udtgn6r3` | `paper-zsrl-dreamer-size200m-w1-h512` |
| `ssl-lab/rl-in-pixel-env-dreamer` | `vh3l53jd` | `paper-zsrl-dreamer-size400m-cgs0p01-old-log1k` |
| `ssl-lab/rl-in-pixel-env-diamond` | `uwqo1qel` | `paper-zsrl-diamond-hf-h512` |
| `ssl-lab/rl-in-pixel-env-diamond` | `v4y77brt` | `paper-zsrl-diamond-exp-repro-h512` |
| `ssl-lab/rl-in-pixel-env-diamond` | `b52orqfs` | `paper-zsrl-diamond-mask13-w0p001-h512` |
| `ssl-lab/rl-in-pixel-env-diamond` | `kmop8mc3` | `paper-zsrl-diamond-mask13-w0p01-h512` |
| `ssl-lab/rl-in-pixel-env-diamond` | `b0ejwfxn` | `paper-zsrl-diamond-mask13-w0p1-h512` |
| `ssl-lab/rl-in-pixel-env-diamond` | `i0mljzr4` | `paper-zsrl-diamond-mask13-w1-h512` |
| `ssl-lab/rl-in-pixel-env-simulus` | `5l20n3ja` | `paper-zsrl-simulus-repro-h512-rewq0p001` |
| `ssl-lab/rl-in-pixel-env-simulus` | `pufgm4zu` | `paper-zsrl-simulus-repro-h512` |
| `ssl-lab/rl-in-pixel-env-simulus` | `yg6yqfu6` | `paper-zsrl-simulus-repro-h512-rewq0p5` |
| `ssl-lab/rl-in-pixel-env-simulus` | `7i0qq7ho` | `paper-zsrl-simulus-offline-w0-h512` |
| `ssl-lab/rl-in-pixel-env-simulus` | `y5n1ev0v` | `paper-zsrl-simulus-offline-w0p01-h512` |
| `ssl-lab/rl-in-pixel-env-simulus` | `wg1jc761` | `paper-zsrl-simulus-offline-w0p1-h512` |
| `ssl-lab/rl-in-pixel-env-simulus` | `ww09vt09` | `paper-zsrl-simulus-diamondreplay-offline-w0p01-h512` |
| `ssl-lab/rl-in-pixel-env-simulus` | `b87vxbqm` | `paper-zsrl-simulus-diamondreplay-offline-w0p1-h512` |
| `ssl-lab/rl-in-pixel-env-simulus` | `ev3amjfw` | `paper-zsrl-simulus-diamondreplay-w1-h512` |
| `ssl-lab/rl-in-pixel-env-simulus` | `2cilzlwy` | `paper-zsrl-simulus-twexp-w0-h512` |
| `ssl-lab/rl-in-pixel-env-simulus` | `lc28ctea` | `paper-zsrl-simulus-twexp-w0p01-h512` |
| `ssl-lab/rl-in-pixel-env-simulus` | `s0odx1er` | `paper-zsrl-simulus-twexp-w0p1-h512` |
| `ssl-lab/rl-in-pixel-env-twister` | `sk6pr3jh` | `paper-zsrl-twister-repro-h128-rewq0p001` |
| `ssl-lab/rl-in-pixel-env-twister` | `qqopxvlv` | `paper-zsrl-twister-repro-h128` |
| `ssl-lab/rl-in-pixel-env-twister` | `ewxjbe5a` | `paper-zsrl-twister-repro-h128-rewq0p5` |
| `ssl-lab/rl-in-pixel-env-twister` | `qaa6jl0o` | `paper-zsrl-twister-repro-h512-rewq0p001` |
| `ssl-lab/rl-in-pixel-env-twister` | `q0uu7f2c` | `paper-zsrl-twister-repro-h512` |
| `ssl-lab/rl-in-pixel-env-twister` | `21fgh1ab` | `paper-zsrl-twister-repro-h512-rewq0p5` |
| `ssl-lab/rl-in-pixel-env-twister` | `q8c8bege` | `paper-zsrl-twister-diamond-static-w0-h512` |
| `ssl-lab/rl-in-pixel-env-twister` | `nhnu6bgw` | `paper-zsrl-twister-offline-ac-cpc-w0-h512` |
| `ssl-lab/rl-in-pixel-env-twister` | `nv6p4ix2` | `paper-zsrl-twister-offline-ac-cpc-w1-h512` |
| `ssl-lab/rl-in-pixel-env-twister` | `zwd9m1bt` | `paper-zsrl-twister-twexp-w0-h512-rewq0p5` |
| `ssl-lab/rl-in-pixel-env-twister` | `x0bkt2ot` | `paper-zsrl-twister-twexp-w0p01-h512-rewq0p5` |
| `ssl-lab/rl-in-pixel-env-twister` | `h4bqm2fm` | `paper-zsrl-twister-twexp-w0p1-h512-rewq0p5` |
| `ssl-lab/rl-in-pixel-env-twister` | `4qzb8na8` | `paper-zsrl-twister-twexp-w1-h512-rewq0p5` |
| `ssl-lab/rl-in-pixel-env-storm` | `9bx3w28b` | `paper-zsrl-storm-repro-h128-rewq0p001` |
| `ssl-lab/rl-in-pixel-env-storm` | `eoksijt6` | `paper-zsrl-storm-repro-h128` |
| `ssl-lab/rl-in-pixel-env-storm` | `aefa9yuy` | `paper-zsrl-storm-repro-h128-rewq0p5` |
| `ssl-lab/rl-in-pixel-env-storm` | `910r264r` | `paper-zsrl-storm-repro-h512-rewq0p001` |
| `ssl-lab/rl-in-pixel-env-storm` | `phzb9xat` | `paper-zsrl-storm-repro-h512` |
| `ssl-lab/rl-in-pixel-env-storm` | `imj7qsz4` | `paper-zsrl-storm-repro-h512-rewq0p5` |
| `ssl-lab/rl-in-pixel-env-storm` | `sbfsn9bs` | `paper-zsrl-storm-offline-w0-h512` |
| `ssl-lab/rl-in-pixel-env-storm` | `y1rogmug` | `paper-zsrl-storm-offline-w0p01-h512` |

## WM Training Runs

| Project | Run id | Display name |
| --- | --- | --- |
| `ssl-lab/twister` | `bg4lhhcw` | `paper-wm-twister-offline-ac-cpc-w0p003` |
| `ssl-lab/twister` | `4807yvf0` | `paper-wm-twister-offline-ac-cpc-w0p005` |
| `ssl-lab/twister` | `ttw5akh3` | `paper-wm-twister-offline-ac-cpc-w0p02` |
| `ssl-lab/twister` | `oebdkvwl` | `paper-wm-twister-offline-ac-cpc-w0p05` |
| `ssl-lab/twister` | `dv4m21jy` | `paper-wm-twister-offline-ac-cpc-w1` |
| `ssl-lab/twister` | `y8r41fds` | `paper-wm-twister-offline-ac-cpc-w10` |
| `ssl-lab/simulus` | `v2h4k7yr` | `paper-wm-simulus-tokenizer-offline-w0p001-oldsigreg` |
| `ssl-lab/simulus` | `f7gxl3ta` | `paper-wm-simulus-tokenizer-offline-w0p01-oldsigreg` |
| `ssl-lab/simulus` | `eur80nch` | `paper-wm-simulus-tokenizer-offline-w0p1-oldsigreg` |
| `ssl-lab/simulus` | `mikz6u3m` | `paper-wm-simulus-tokenizer-offline-w1-oldsigreg` |
| `ssl-lab/simulus` | `ox50nbri` | `paper-wm-simulus-tokenizer-offline-w10-oldsigreg` |
| `ssl-lab/diamond` | `p5zrqwdl` | `paper-wm-diamond-denoiser-mask13-w0p001` |
| `ssl-lab/diamond` | `371xbvfz` | `paper-wm-diamond-denoiser-mask13-w0p01` |
| `ssl-lab/diamond` | `d6jythur` | `paper-wm-diamond-denoiser-mask13-w0p1` |
| `ssl-lab/diamond` | `z59x22of` | `paper-wm-diamond-denoiser-mask13-w1` |
| `ssl-lab/diamond` | `f01i94v4` | `paper-wm-diamond-twexp-dilated-k3-w0` |
| `ssl-lab/diamond` | `dyclkt69` | `paper-wm-diamond-twexp-dilated-k3-w0p01` |
| `ssl-lab/oc-storm` | `ecnn7tcp` | `paper-wm-storm-size200m-w0p01` |
| `ssl-lab/oc-storm` | `p47seusg` | `paper-wm-storm-size200m-w0p05` |
| `ssl-lab/oc-storm` | `khi5gdld` | `paper-wm-storm-size200m-w0p1` |
| `ssl-lab/oc-storm` | `80np6e56` | `paper-wm-storm-size200m-w1` |
| `ssl-lab/oc-storm` | `n5n1875d` | `paper-wm-storm-size400m-w0p01` |
| `ssl-lab/oc-storm` | `phjmk9qq` | `paper-wm-storm-size400m-w0p05` |
| `ssl-lab/oc-storm` | `q0ssyqnr` | `paper-wm-storm-size400m-w0p1` |

## Unresolved Local References

These run ids are recorded in local TWISTER replay policy archives, but W&B API
could not find them under the checked `ssl-lab` projects on 2026-07-02:

- `jb2u331u`
- `oe7fu37x`
- `kmv64y6h`
- `sklo7o5n`
