# Scripts

Main entry points:

- `eval/pong_wm_metrics.py`: replay-aligned Dreamer WM metrics and frame export.
- `eval/pong_sam2_ball_experiment.py`: generated-video SAM2 ball tracking for
  Dreamer checkpoints.
- `eval/pong_external_sam2_ball_experiment.py`: generated-video SAM2 ball
  tracking for DIAMOND, STORM/OC-STORM, Twister, and Simulus.
- `eval/external_wm_rollout_worker.py`: subprocess worker that runs one Torch
  world model in its own project/env.
- `eval/pong_ball_physical_consistency.py`: postprocesses SAM2 ball tracks into
  flicker, teleport, spontaneous-turn, and acceleration metrics.
- `eval/masks_to_tracks.py`: converts SAM2 mask sequences into centroid/bbox
  track CSVs, useful for paddle collision exclusion.
