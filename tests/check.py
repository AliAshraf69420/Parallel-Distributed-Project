#!/usr/bin/env python3
"""Check one run of any implementation against the frozen reference, bit for bit.

Compares <run>/stats.json and <run>/checksums.csv with results/reference/<set>/:
  * n, S1, S2         identical integers
  * mean, std         identical doubles
  * lut_f32_hex       identical float32 bits
  * per-image CRCs    identical; a crc_resized mismatch means the resize differs,
                      a crc_normalized-only mismatch means stats/LUT/normalize differ
  * per_class.csv     (Hadoop runs) per-class n, S1, S2 identical to the reference
A run without checksums.csv (statistics job only) is checked on the statistics alone.
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
    for p in (ref_dir / "stats.json", args.run / "stats.json"):
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

    has_crcs = (args.run / "checksums.csv").is_file()
    rc = load_crcs(ref_dir / "checksums.csv") if has_crcs else {}
    gc = load_crcs(args.run / "checksums.csv") if has_crcs else {}
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

    per_class = args.run / "per_class.csv"
    n_class_keys = 0
    if per_class.is_file():
        want = {(int(k), c): (v["n"], v["S1"][c], v["S2"][c])
                for k, v in ref["per_class_train"].items() for c in range(3)}
        with open(per_class, newline="") as f:
            have = {(int(r["class_id"]), int(r["channel"])): (int(r["n"]), int(r["S1"]), int(r["S2"]))
                    for r in csv.DictReader(f)}
        n_class_keys = len(have)
        if want != have:
            wrong = sorted(k for k in want.keys() | have.keys() if want.get(k) != have.get(k))
            problems.append(f"per-class sums differ for {len(wrong)} (class, channel) keys "
                            f"(first: {wrong[:5]})")

    label = f"{got.get('version', '?')} on {args.set}"
    if problems:
        print(f"FAIL {label}:")
        for p in problems:
            print(f"  - {p}")
        return 1
    parts = ["stats", "LUT"]
    if has_crcs:
        parts.append(f"{len(common)} image checksums")
    if n_class_keys:
        parts.append(f"{n_class_keys} per-class sums")
    print(f"PASS {label}: {', '.join(parts[:-1])} and {parts[-1]} match the reference exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
