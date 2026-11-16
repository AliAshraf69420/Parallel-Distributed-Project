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
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import PIL
from PIL import features

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


def _verify(set_dir: Path, rows: list[dict]) -> None:
    records, num_shards = cubfmt.read_index(set_dir / "index.bin")
    if len(records) != len(rows):
        raise RuntimeError(f"index has {len(records)} records, manifest {len(rows)}")
    shards = cubfmt.open_shards(set_dir, num_shards)
    for rec, row in zip(records, rows):
        if int(rec["image_id"]) != int(row["image_id"]):
            raise RuntimeError(f"record order mismatch at image_id {row['image_id']}")
        px = cubfmt.record_pixels(shards, rec)
        if hashlib.sha1(px.data).hexdigest() != row["sha1"]:
            raise RuntimeError(f"verification failed for image_id {row['image_id']}")


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

    final_dir = args.processed / args.name
    tmp_dir = args.processed / f".{args.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    shard_limit = args.shard_mb * 2**20
    t0 = time.perf_counter()

    records = np.zeros(len(rows), dtype=cubfmt.RECORD_DTYPE)
    by_id = {int(r["image_id"]): (n, r) for n, r in enumerate(rows)}
    shard_sizes: list[int] = []
    shard_k, pos = -1, 0
    out = None
    jobs = [(int(r["image_id"]), r["rel_path"], r["sha1"]) for r in rows]

    try:
        with Pool(args.jobs, initializer=_init_worker,
                  initargs=(str(args.root / "images"),)) as pool:
            # imap (ordered) keeps the shard layout deterministic: image_id order.
            for done, (image_id, block) in enumerate(pool.imap(_decode, jobs, chunksize=16), 1):
                n, row = by_id[image_id]
                w, h = int(row["w"]), int(row["h"])
                if len(block) != w * h * 3:
                    raise RuntimeError(f"image_id {image_id}: {len(block)} bytes, expected {w}x{h}x3")
                if w > 0xFFFF or h > 0xFFFF:
                    raise RuntimeError(f"image_id {image_id}: {w}x{h} does not fit in u16")

                start = cubfmt.align_up(pos)
                if out is None or (pos > 0 and start + len(block) > shard_limit):
                    if out is not None:
                        out.close()
                        shard_sizes.append(pos)
                    shard_k += 1
                    out = open(tmp_dir / cubfmt.shard_name(shard_k), "wb")
                    pos = start = 0
                out.write(b"\0" * (start - pos))
                out.write(block)
                pos = start + len(block)

                rec = records[n]
                rec["offset"] = start
                rec["image_id"] = image_id
                rec["class_id"] = int(row["class_id"])
                rec["is_train"] = int(row["is_train"])
                rec["flags"] = cubfmt.FLAG_CONVERTED if row["mode"] != "RGB" else 0
                rec["w"], rec["h"] = w, h
                rec["bx"], rec["by"] = int(row["bx"]), int(row["by"])
                rec["bw"], rec["bh"] = int(row["bw"]), int(row["bh"])
                rec["shard"] = shard_k
                if done % 1000 == 0 or done == len(jobs):
                    print(f"  wrote {done}/{len(jobs)} (shard {shard_k})", file=sys.stderr)
        out.close()
        out = None
        shard_sizes.append(pos)
        cubfmt.write_index(tmp_dir / "index.bin", records, len(shard_sizes))
        t_write = time.perf_counter() - t0

        if not args.no_verify:
            print("  verifying ...", file=sys.stderr)
            _verify(tmp_dir, rows)
        t_total = time.perf_counter() - t0

        pixel_bytes = int(sum(int(r["w"]) * int(r["h"]) * 3 for r in rows))
        meta = {
            "set": args.name,
            "format_version": cubfmt.FORMAT_VERSION,
            "record_size": cubfmt.RECORD_DTYPE.itemsize,
            "align": cubfmt.ALIGN,
            "count": len(rows),
            "train": int(records["is_train"].sum()),
            "test": int(len(rows) - records["is_train"].sum()),
            "classes": int(len(np.unique(records["class_id"]))),
            "pixel_bytes": pixel_bytes,
            "shard_files": {cubfmt.shard_name(k): s for k, s in enumerate(shard_sizes)},
            "bytes_on_disk": int(sum(shard_sizes) + (tmp_dir / "index.bin").stat().st_size),
            "source_manifest_sha1": hashlib.sha1(manifest.read_bytes()).hexdigest(),
            "verified": not args.no_verify,
            "seconds": {"decode_write": round(t_write, 2), "total": round(t_total, 2)},
            "tool_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                              "pillow": PIL.__version__,
                              "libjpeg_turbo": features.version("libjpeg_turbo")},
        }
        (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    except BaseException:
        if out is not None:
            out.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    print(f"{args.name}: {meta['count']} images, {len(shard_sizes)} shard(s), "
          f"{meta['bytes_on_disk'] / 2**30:.3f} GiB on disk, "
          f"{'verified' if meta['verified'] else 'NOT verified'}, {t_total:.1f} s -> {final_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
