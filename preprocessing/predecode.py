#!/usr/bin/env python3
"""Step 2 of preprocessing: decode every validated JPEG once into binary shards.

Reads data/processed/manifest.csv (from validate.py) and writes the "full" set:
  data/processed/full/index.bin        header + one 32-byte record per image
  data/processed/full/shard_KKKK.bin   raw HWC RGB uint8 pixels, 64-byte aligned blocks
  data/processed/full/meta.json        counts, sizes, tool versions
The format is documented in cubfmt.py. Records are in image_id order.

Decoding runs in a process pool; one writer appends blocks in order, starting a
new shard when the current one would exceed --shard-mb. The set is written to
a temporary directory and renamed into place only after it has been verified
(every block re-read and its SHA-1 compared with the manifest), so a crash or
Ctrl-C never leaves a half-written set behind.

Usage
  python preprocessing/predecode.py [--root DIR] [--processed DIR] [--name full]
                                    [--shard-mb 1024] [-j N] [--no-verify]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cubfmt

_IMAGES_DIR: Path | None = None


def _init_worker(images_dir: str) -> None:
    global _IMAGES_DIR
    _IMAGES_DIR = Path(images_dir)


def _decode(job: tuple[int, str, str]) -> tuple[int, bytes]:
    image_id, rel, want_sha1 = job
    arr, _ = cubfmt.decode_rgb(_IMAGES_DIR / rel)
    got = hashlib.sha1(arr.data).hexdigest()
    if got != want_sha1:
        raise RuntimeError(f"image_id {image_id}: pixels differ from validation "
                           f"(sha1 {got} != {want_sha1}); re-run validate.py")
    return image_id, arr.tobytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=cubfmt.DEFAULT_RAW,
                    help="extracted CUB_200_2011 directory (default: %(default)s)")
    ap.add_argument("--processed", type=Path, default=cubfmt.DEFAULT_PROCESSED,
                    help="holds manifest.csv; the set is written below it (default: %(default)s)")
    ap.add_argument("--name", default="full", help="output set name (default: %(default)s)")
    ap.add_argument("--shard-mb", type=int, default=1024,
                    help="max shard size in MiB (default: %(default)s)")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                    help="decoder processes (default: all CPUs)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip re-reading and hashing every block after writing")
    args = ap.parse_args()

    manifest = args.processed / "manifest.csv"
    if not manifest.is_file():
        print(f"error: {manifest} not found; run validate.py first", file=sys.stderr)
        return 1
    rows = sorted(cubfmt.read_manifest(manifest), key=lambda r: int(r["image_id"]))
    if not rows:
        print("error: manifest is empty", file=sys.stderr)
        return 1
    by_id = {int(r["image_id"]): r for r in rows}
    jobs = [(int(r["image_id"]), r["rel_path"], r["sha1"]) for r in rows]
    t0 = time.perf_counter()

    with cubfmt.SetWriter(args.processed, args.name, args.shard_mb) as sw, \
         Pool(args.jobs, initializer=_init_worker, initargs=(str(args.root / "images"),)) as pool:
        # imap (ordered) keeps the shard layout deterministic: image_id order.
        for done, (image_id, block) in enumerate(pool.imap(_decode, jobs, chunksize=16), 1):
            row = by_id[image_id]
            sw.add({**row, "flags": cubfmt.FLAG_CONVERTED if row["mode"] != "RGB" else 0}, block)
            if done % 1000 == 0 or done == len(jobs):
                print(f"  wrote {done}/{len(jobs)}", file=sys.stderr)
        t_write = time.perf_counter() - t0
        if not args.no_verify:
            print("  verifying ...", file=sys.stderr)
            sw.verify({int(r["image_id"]): r["sha1"] for r in rows})
        meta = sw.commit({
            "source_manifest_sha1": hashlib.sha1(manifest.read_bytes()).hexdigest(),
            "seconds": {"decode_write": round(t_write, 2),
                        "total": round(time.perf_counter() - t0, 2)},
        })

    print(f"{args.name}: {meta['count']} images, {len(meta['shard_files'])} shard(s), "
          f"{meta['bytes_on_disk'] / 2**30:.3f} GiB on disk, "
          f"{'verified' if meta['verified'] else 'NOT verified'}, "
          f"{meta['seconds']['total']:.1f} s -> {sw.final_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
