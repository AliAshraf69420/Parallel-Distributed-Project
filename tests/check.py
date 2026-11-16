#!/usr/bin/env python3
"""Check one run of any implementation against the frozen reference, bit for bit.

Compares <run>/stats.json and <run>/checksums.csv with results/reference/<set>/:
  * n, S1, S2         identical integers
  * mean, std         identical doubles
  * lut_f32_hex       identical float32 bits
  * per-image CRCs    identical; a crc_resized mismatch means the resize differs,
                      a crc_normalized-only mismatch means stats/LUT/normalize differ
Exit status 0 = PASS, 1 = FAIL.

Usage
  python tests/check.py --set 100MB --run results/runs/omp/100MB
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_crcs(path: Path) -> dict[int, tuple[str, str]]:
    with open(path, newline="") as f:
        return {int(r["image_id"]): (r["crc_resized"], r["crc_normalized"])
                for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", required=True, help="set name, e.g. full, 100MB, 500MB")
    ap.add_argument("--run", required=True, type=Path, help="directory with the run's outputs")
    ap.add_argument("--reference", type=Path, default=REPO / "results" / "reference")
    args = ap.parse_args()

    ref_dir = args.reference / args.set
    for p in (ref_dir / "stats.json", args.run / "stats.json", args.run / "checksums.csv"):
        if not p.is_file():
            print(f"FAIL: {p} not found", file=sys.stderr)
            return 1
    ref = json.loads((ref_dir / "stats.json").read_text())
    got = json.loads((args.run / "stats.json").read_text())
    problems: list[str] = []

    for key in ("records", "train_records", "n", "S1", "S2", "mean", "std", "lut_f32_hex"):
        if ref.get(key) != got.get(key):
            if key == "lut_f32_hex" and isinstance(got.get(key), list):
                diff = sum(a != b for ra, rb in zip(ref[key], got[key]) for a, b in zip(ra, rb))
                problems.append(f"{key}: {diff} of 768 entries differ")
            else:
                problems.append(f"{key}: expected {ref.get(key)}, got {got.get(key)}")

    rc, gc = load_crcs(ref_dir / "checksums.csv"), load_crcs(args.run / "checksums.csv")
    if rc.keys() != gc.keys():
        problems.append(f"image ids differ: {len(rc.keys() - gc.keys())} missing, "
                        f"{len(gc.keys() - rc.keys())} unexpected")
    common = sorted(rc.keys() & gc.keys())
    bad_resize = [i for i in common if rc[i][0] != gc[i][0]]
    bad_norm = [i for i in common if rc[i][0] == gc[i][0] and rc[i][1] != gc[i][1]]
    if bad_resize:
        problems.append(f"resize differs for {len(bad_resize)} images (first: {bad_resize[:5]})")
    if bad_norm:
        problems.append(f"normalize differs for {len(bad_norm)} images (first: {bad_norm[:5]})")

    label = f"{got.get('version', '?')} on {args.set}"
    if problems:
        print(f"FAIL {label}:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"PASS {label}: stats, LUT and {len(common)} image checksums match the reference exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
