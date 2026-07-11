#!/usr/bin/env python3
"""Convert binary mask sequences into centroid/bbox track CSVs."""

from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np


def mask_to_row(mask: np.ndarray, t: int, sample_id: str, prefix: str):
  ys, xs = np.nonzero(np.asarray(mask) > 0)
  row = {
      'sample_id': sample_id,
      't': int(t),
      f'{prefix}_present': bool(len(xs)),
      f'{prefix}_area': int(len(xs)),
  }
  if len(xs):
    row.update({
        f'{prefix}_x': float(xs.mean()),
        f'{prefix}_y': float(ys.mean()),
        'xmin': int(xs.min()),
        'xmax': int(xs.max()),
        'ymin': int(ys.min()),
        'ymax': int(ys.max()),
    })
  return row


def write_csv(path: pathlib.Path, rows: list[dict]):
  path.parent.mkdir(parents=True, exist_ok=True)
  keys = ['sample_id', 't']
  for row in rows:
    for key in row:
      if key not in keys:
        keys.append(key)
  with path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def main() -> None:
  parser = argparse.ArgumentParser(
      description='Convert a SAM2 binary mask npz into a bbox track CSV.')
  parser.add_argument('--mask-npz', type=pathlib.Path, required=True)
  parser.add_argument('--key', default='mask1')
  parser.add_argument('--output-csv', type=pathlib.Path, required=True)
  parser.add_argument('--sample-id', default='window0')
  parser.add_argument('--prefix', default='object',
                      help='Column prefix, for example ball or paddle.')
  parser.add_argument('--downscale', type=float, default=1.0,
                      help='Divide coordinates by this value, e.g. 4 for 256->64.')
  args = parser.parse_args()

  with np.load(args.mask_npz) as data:
    masks = np.asarray(data[args.key])
  rows = []
  for t, mask in enumerate(masks):
    row = mask_to_row(mask, t, args.sample_id, args.prefix)
    if args.downscale != 1.0:
      for key in ('xmin', 'xmax', 'ymin', 'ymax', f'{args.prefix}_x', f'{args.prefix}_y'):
        if key in row:
          row[key] = float(row[key]) / args.downscale
    rows.append(row)
  write_csv(args.output_csv, rows)
  print(args.output_csv)


if __name__ == '__main__':
  main()

