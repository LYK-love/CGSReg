#!/usr/bin/env python3
"""Build a compact evaluation bundle from rollout target manifests."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "eval"


def read_json(path: pathlib.Path) -> Any:
  return json.loads(path.read_text())


def write_json(path: pathlib.Path, data: Any):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def rel_or_abs(path: str) -> pathlib.Path:
  value = pathlib.Path(path).expanduser()
  return value if value.is_absolute() else ROOT / value


def link_or_copy(src: pathlib.Path, dst: pathlib.Path, *, copy: bool):
  dst.parent.mkdir(parents=True, exist_ok=True)
  if dst.exists() or dst.is_symlink():
    dst.unlink()
  if copy:
    shutil.copy2(src, dst)
  else:
    dst.symlink_to(src.resolve())


def link_or_copy_tree(src: pathlib.Path, dst: pathlib.Path, *, copy: bool):
  dst.parent.mkdir(parents=True, exist_ok=True)
  if dst.exists() or dst.is_symlink():
    if dst.is_dir() and not dst.is_symlink():
      shutil.rmtree(dst)
    else:
      dst.unlink()
  if copy:
    shutil.copytree(src, dst)
  else:
    dst.symlink_to(src.resolve(), target_is_directory=True)


def parse_csv(value: str) -> list[str]:
  return [item.strip() for item in value.split(",") if item.strip()]


def manifest_entries_by_key(bundle_root: pathlib.Path) -> dict[tuple[str, str, str], dict[str, Any]]:
  path = bundle_root / "bundle_manifest.json"
  if not path.exists():
    raise FileNotFoundError(f"Missing source bundle manifest: {path}")
  manifest = read_json(path)
  return {
      (row["project"], row["condition"], row["episode"]): row
      for row in manifest.get("entries", [])
  }


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
  manifest = read_json(args.manifest)
  output = args.output_dir
  output.mkdir(parents=True, exist_ok=True)
  write_json(output / "config" / "target_rollouts.json", manifest)

  entries = []
  missing = []
  selected_projects = set(parse_csv(args.target_projects)) if args.target_projects else None
  for target in manifest["targets"]:
    project = target["project"]
    if selected_projects is not None and project not in selected_projects:
      continue
    for cond in target["conditions"]:
      condition = cond["condition"]
      root = rel_or_abs(cond["root"])
      model = cond["model"]
      for episode_dir in sorted(root.glob("ep*_seed*")):
        episode = episode_dir.name
        video = episode_dir / "videos" / f"{model}.mp4"
        metadata = episode_dir / "rollouts" / f"{model}.json"
        if not video.exists() or not metadata.exists():
          missing.append({
              "project": project,
              "condition": condition,
              "episode": episode,
              "model": model,
              "video": str(video),
              "metadata": str(metadata),
          })
          continue

        video_dst = output / "videos" / project / condition / f"{episode}.mp4"
        meta_dst = output / "rollout_metadata" / project / condition / f"{episode}.json"
        link_or_copy(video, video_dst, copy=args.copy)
        data = read_json(metadata)
        data.update({
            "bundle_project": project,
            "bundle_condition": condition,
            "bundle_weight": cond.get("weight", ""),
            "bundle_model": model,
            "bundle_source_root": str(root),
            "bundle_source_video": str(video),
            "bundle_source_metadata": str(metadata),
        })
        write_json(meta_dst, data)
        entries.append({
            "project": project,
            "condition": condition,
            "weight": cond.get("weight", ""),
            "episode": episode,
            "model": model,
            "video": str(video_dst),
            "metadata": str(meta_dst),
        })

  shared_ale_added = 0
  if args.shared_ale_bundle:
    shared_bundle = args.shared_ale_bundle.expanduser()
    shared_bundle = shared_bundle if shared_bundle.is_absolute() else ROOT / shared_bundle
    source_entries = manifest_entries_by_key(shared_bundle)
    target_projects = parse_csv(args.shared_ale_target_projects)
    if not target_projects:
      target_projects = [
          target["project"] for target in manifest.get("targets", [])
          if selected_projects is None or target["project"] in selected_projects
      ]
    target_projects = list(dict.fromkeys(target_projects))

    target_episode_names = sorted({row["episode"] for row in entries if row.get("project") in target_projects})
    source_keys = sorted(
        key for key in source_entries
        if key[0] == args.shared_ale_source_project and key[1] == args.shared_ale_source_condition
    )
    if args.shared_ale_match_target_episodes and target_episode_names:
      target_episode_set = set(target_episode_names)
      source_keys = [key for key in source_keys if key[2] in target_episode_set]
    if not source_keys:
      raise ValueError(
          "No shared ALE source entries found for "
          f"{args.shared_ale_source_project}/{args.shared_ale_source_condition} in {shared_bundle}"
      )
    for _, _, episode in source_keys:
      source_row = source_entries[(args.shared_ale_source_project, args.shared_ale_source_condition, episode)]
      source_video = shared_bundle / "videos" / args.shared_ale_source_project / args.shared_ale_source_condition / f"{episode}.mp4"
      source_meta = shared_bundle / "rollout_metadata" / args.shared_ale_source_project / args.shared_ale_source_condition / f"{episode}.json"
      source_cutie = shared_bundle / "cutie_segmentations" / f"{args.shared_ale_source_project}__{args.shared_ale_source_condition}__{episode}"
      for project in target_projects:
        video_dst = output / "videos" / project / args.shared_ale_target_condition / f"{episode}.mp4"
        meta_dst = output / "rollout_metadata" / project / args.shared_ale_target_condition / f"{episode}.json"
        cutie_dst = output / "cutie_segmentations" / f"{project}__{args.shared_ale_target_condition}__{episode}"
        if not source_video.exists() or not source_meta.exists():
          missing.append({
              "project": project,
              "condition": args.shared_ale_target_condition,
              "episode": episode,
              "model": "ale_pong",
              "video": str(source_video),
              "metadata": str(source_meta),
          })
          continue
        link_or_copy(source_video, video_dst, copy=args.copy)
        data = read_json(source_meta)
        data.update({
            "bundle_project": project,
            "bundle_condition": args.shared_ale_target_condition,
            "bundle_weight": "",
            "bundle_model": "ale_pong",
            "bundle_source_bundle": str(shared_bundle),
            "bundle_source_project": args.shared_ale_source_project,
            "bundle_source_condition": args.shared_ale_source_condition,
            "bundle_source_video": str(source_video),
            "bundle_source_metadata": str(source_meta),
            "shared_ale_baseline": True,
        })
        write_json(meta_dst, data)
        if args.link_shared_ale_cutie and source_cutie.exists():
          link_or_copy_tree(source_cutie, cutie_dst, copy=args.copy)
        entries.append({
            "project": project,
            "condition": args.shared_ale_target_condition,
            "weight": "",
            "episode": episode,
            "model": "ale_pong",
            "video": str(video_dst),
            "metadata": str(meta_dst),
            "shared_ale_baseline": True,
            "shared_ale_source": {
                "bundle": str(shared_bundle),
                "project": args.shared_ale_source_project,
                "condition": args.shared_ale_source_condition,
            },
        })
        shared_ale_added += 1

  write_json(output / "bundle_manifest.json", {
      "source_manifest": str(args.manifest),
      "copy": bool(args.copy),
      "shared_ale_bundle": str(args.shared_ale_bundle) if args.shared_ale_bundle else "",
      "shared_ale_source_project": args.shared_ale_source_project if args.shared_ale_bundle else "",
      "shared_ale_source_condition": args.shared_ale_source_condition if args.shared_ale_bundle else "",
      "shared_ale_target_condition": args.shared_ale_target_condition if args.shared_ale_bundle else "",
      "shared_ale_entries": int(shared_ale_added),
      "entries": entries,
      "missing": missing,
  })
  lines = [
      "# Rollout Evaluation Bundle",
      "",
      f"- Source manifest: `{args.manifest}`",
      f"- Entries: {len(entries)}",
      f"- Missing entries: {len(missing)}",
      f"- Videos: `{output / 'videos'}`",
      f"- Metadata: `{output / 'rollout_metadata'}`",
  ]
  if args.shared_ale_bundle:
    lines.extend([
        f"- Shared ALE bundle: `{args.shared_ale_bundle}`",
        f"- Shared ALE entries: {shared_ale_added}",
        f"- Shared ALE target condition: `{args.shared_ale_target_condition}`",
    ])
  (output / "README.md").write_text("\n".join(lines) + "\n")
  return {"entries": len(entries), "missing": len(missing), "shared_ale_entries": shared_ale_added}


def main(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=pathlib.Path, required=True)
  parser.add_argument("--output-dir", type=pathlib.Path, required=True)
  parser.add_argument("--copy", action="store_true",
                      help="Copy videos instead of symlinking them.")
  parser.add_argument("--target-projects", default="",
                      help="Comma-separated project subset to include from the manifest. Defaults to all projects.")
  parser.add_argument("--shared-ale-bundle", type=pathlib.Path, default=None,
                      help="Optional source bundle containing one real ALE baseline to alias into every target project.")
  parser.add_argument("--shared-ale-source-project", default="diamond")
  parser.add_argument("--shared-ale-source-condition", default="ale_emulator")
  parser.add_argument("--shared-ale-target-condition", default="ale_emulator")
  parser.add_argument("--shared-ale-target-projects", default="",
                      help="Comma-separated projects to receive the shared ALE baseline. Defaults to all manifest projects.")
  parser.add_argument("--shared-ale-match-target-episodes", action=argparse.BooleanOptionalAction, default=True,
                      help="Only alias ALE episodes that also appear in selected target rollout entries.")
  parser.add_argument("--link-shared-ale-cutie", action=argparse.BooleanOptionalAction, default=True,
                      help="Also link/copy CUTIE outputs from the shared ALE source bundle when present.")
  parser.add_argument("--run-cutie", action=argparse.BooleanOptionalAction, default=False,
                      help="Run CUTIE over the completed bundle after building it.")
  parser.add_argument("--cutie-output-dir", type=pathlib.Path, default=None,
                      help="CUTIE output directory. Defaults to output-dir/cutie_segmentations.")
  parser.add_argument("--cutie-geometry", choices=("square-to-atari", "native"), default="square-to-atari")
  parser.add_argument("--cutie-fps", type=int, default=15)
  parser.add_argument("--cutie-save-masks", action="store_true")
  args = parser.parse_args(argv)
  result = build_bundle(args)
  print(
      f"Wrote {args.output_dir} with {result['entries']} entries; "
      f"shared_ale={result['shared_ale_entries']} missing={result['missing']}"
  )
  if args.run_cutie:
    cutie_output_dir = (
        args.cutie_output_dir.expanduser()
        if args.cutie_output_dir is not None
        else args.output_dir / "cutie_segmentations"
    )
    cmd = [
        sys.executable,
        str(SCRIPTS / "run_oc_storm_cutie_pong_tracks.py"),
        "--bundle-root",
        str(args.output_dir),
        "--output-dir",
        str(cutie_output_dir),
        "--geometry",
        args.cutie_geometry,
        "--fps",
        str(int(args.cutie_fps)),
        "--skip-existing",
    ]
    if args.cutie_save_masks:
      cmd.append("--save-masks")
    print("Running CUTIE:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
  main()
