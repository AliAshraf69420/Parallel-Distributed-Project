#!/usr/bin/env python3
"""Step 3 of preprocessing: cut smaller, class-stratified subsets of the full set.

Each subset is a binary set in the same format as `full` (see cubfmt.py), with
pixels copied byte-for-byte from the full shards, so every phase can run on
any set unchanged.

Selection (deterministic for a given --seed):
  * group records by (class_id, is_train) and shuffle each group with the seed
  * give the record at position i of a group of size n the key (i + u) / n,
    with u ~ U(0,1) per record, and sort all records by key
  * a subset of target T bytes is the shortest prefix of that order whose
    pixel bytes reach T
Every prefix is therefore close to proportional in class *and* split, and the
subsets are nested: 100MB ⊂ 500MB ⊂ full. Records inside a subset are stored
in image_id order, like the full set.

Sizes are decoded pixel bytes (what the C++ code processes), in MiB.

Usage
  python preprocessing/make_subsets.py [--processed DIR] [--source full]
                                       [--sizes 100 500] [--seed 401] [--reports DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

import cubfmt


def stratified_order(records: np.ndarray, seed: int) -> np.ndarray:
    """Indices of `records` in the proportional, seeded order described above."""
    rng = np.random.default_rng(seed)
    keys = np.empty(len(records), dtype=np.float64)
    groups: dict[tuple[int, int], list[int]] = {}
    for idx in range(len(records)):
        groups.setdefault((int(records["class_id"][idx]), int(records["is_train"][idx])), []).append(idx)
    for g in sorted(groups):
        members = np.array(groups[g])
        rng.shuffle(members)
        n = len(members)
        keys[members] = (np.arange(n) + rng.random(n)) / n
    return np.argsort(keys, kind="stable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--processed", type=Path, default=cubfmt.DEFAULT_PROCESSED,
                    help="directory holding the source set (default: %(default)s)")
    ap.add_argument("--source", default="full", help="set to sample from (default: %(default)s)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 500],
                    help="subset sizes in MiB of decoded pixels (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=401, help="sampling seed (default: %(default)s)")
    ap.add_argument("--reports", type=Path, default=cubfmt.DEFAULT_REPORTS,
                    help="where subsets.json goes (default: %(default)s)")
    ap.add_argument("--shard-mb", type=int, default=1024)
    args = ap.parse_args()

    src_dir = args.processed / args.source
    if not (src_dir / "index.bin").is_file():
        print(f"error: {src_dir}/index.bin not found; run predecode.py first", file=sys.stderr)
        return 1
    manifest = cubfmt.read_manifest(args.processed / "manifest.csv")
    sha1 = {int(r["image_id"]): r["sha1"] for r in manifest}

    records, num_shards = cubfmt.read_index(src_dir / "index.bin")
    shards = cubfmt.open_shards(src_dir, num_shards)
    nbytes = records["w"].astype(np.int64) * records["h"] * 3
    total = int(nbytes.sum())
    order = stratified_order(records, args.seed)
    cum = np.cumsum(nbytes[order])

    for size in sorted(args.sizes):
        target = size * 2**20
        if target >= total:
            print(f"skip {size}MB: source has only {total / 2**20:.0f} MiB", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        take = int(np.searchsorted(cum, target)) + 1          # shortest prefix reaching target
        chosen = np.sort(order[:take])                         # image_id order, like `full`
        name = f"{size}MB"
        with cubfmt.SetWriter(args.processed, name, args.shard_mb) as sw:
            for idx in chosen:
                rec = records[idx]
                sw.add({f: rec[f] for f in ("image_id", "class_id", "is_train", "flags",
                                            "w", "h", "bx", "by", "bw", "bh")},
                       cubfmt.record_pixels(shards, rec).tobytes())
            sw.verify({int(records["image_id"][i]): sha1[int(records["image_id"][i])] for i in chosen})
            per_class = Counter(int(c) for c in records["class_id"][chosen])
            meta = sw.commit({
                "parent": args.source,
                "seed": args.seed,
                "target_pixel_mib": size,
                "fraction_of_parent": round(len(chosen) / len(records), 4),
                "images_per_class": {"min": min(per_class.values()), "max": max(per_class.values())},
                "seconds": round(time.perf_counter() - t0, 2),
            })
        print(f"{name}: {meta['count']} images ({meta['train']} train / {meta['test']} test, "
              f"{meta['classes']} classes, {meta['images_per_class']['min']}-"
              f"{meta['images_per_class']['max']} per class), "
              f"{meta['pixel_bytes'] / 2**20:.1f} MiB pixels, verified -> {sw.final_dir}")

    # Nesting check: every smaller subset is contained in every larger one.
    sizes = sorted(s for s in args.sizes if s * 2**20 < total)
    ids = [set(cubfmt.read_index(args.processed / f"{s}MB" / "index.bin")[0]["image_id"].tolist())
           for s in sizes]
    for a, b, sa, sb in zip(ids, ids[1:], sizes, sizes[1:]):
        if not a <= b:
            print(f"error: {sa}MB is not a subset of {sb}MB", file=sys.stderr)
            return 1
    summary = {f"{s}MB": json.loads((args.processed / f"{s}MB" / "meta.json").read_text())
               for s in sizes}
    out = args.reports / "subsets.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"nested: {' ⊂ '.join(f'{s}MB' for s in sizes)} ⊂ {args.source}; summary -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
