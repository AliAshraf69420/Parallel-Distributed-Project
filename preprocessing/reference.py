#!/usr/bin/env python3
"""Step 4 of preprocessing: compute the frozen reference output of the core computation.

This file is the *specification* of the core computation. Every implementation
(sequential, OpenMP, threads, MPI, Hadoop, CUDA) must reproduce its outputs
bit for bit. The computation is defined with integer arithmetic plus a few
correctly-rounded IEEE double operations, so exact agreement is achievable
in C++ as long as floating-point contraction is disabled (-ffp-contract=off).

Core computation, for a set of records (see cubfmt.py)
======================================================
OUT = 224, FRAC_BITS = 11, ONE = 1 << FRAC_BITS = 2048

1. Crop + resize (per record, train and test)
   crop = pixels[by : by+bh, bx : bx+bw, :]            (HWC uint8, size bh x bw)
   Coordinate tables, for an axis of source length L and output index i in [0, OUT):
       num = (2*i + 1) * L - OUT                       (source coord = num / (2*OUT))
       if num <= 0:  i0 = 0, f = 0
       else:         i0 = num // (2*OUT),  f = num - i0 * 2*OUT
       if i0 >= L-1: i0 = L-1, f = 0
       i1 = min(i0 + 1, L - 1)
       wt = (f * ONE + OUT) // (2*OUT)                 (weight of i1, in [0, ONE))
   Columns use L = bw (-> x0, x1, wx); rows use L = bh (-> y0, y1, wy).
   For every output pixel (y, x) and channel c, with p(r, s) = crop[r, s, c]:
       top = (ONE - wx) * p(y0, x0) + wx * p(y0, x1)
       bot = (ONE - wx) * p(y1, x0) + wx * p(y1, x1)
       out[y, x, c] = ((ONE - wy) * top + wy * bot + (1 << 21)) >> 22
   All terms are non-negative and fit in 32 bits (max 1 071 644 672).
   Result: resized, 224x224x3 HWC uint8.

2. Pass 1: channel statistics (train records only)
   Per channel c, over every resized pixel value v of every train record:
       n      = (#train records) * OUT * OUT         (same for all channels)
       S1[c]  = sum v            (exact integer)
       S2[c]  = sum v*v          (exact integer)
   Partial results merge by plain integer addition, in any order.
   Then, in IEEE double:
       m[c]   = double(S1[c]) / double(n)            (raw 0..255 scale)
       q[c]   = double(S2[c]) / double(n)
       s[c]   = sqrt(q[c] - m[c] * m[c])             (population std, no FMA)
       mean[c] = m[c] / 255,  std[c] = s[c] / 255    (reported on the [0,1] scale)
   Lookup table: lut[c][v] = float32( (double(v) - m[c]) / s[c] ),  v = 0..255

3. Pass 2: normalize (every record)
   normalized[c, y, x] = lut[c][resized[y, x, c]]      (CHW float32)

4. Checksums (per record): CRC-32 (zlib/IEEE 802.3 polynomial) of
   * the resized HWC uint8 bytes                       -> crc_resized
   * the normalized CHW little-endian float32 bytes    -> crc_normalized

Outputs (results/reference/<set>/)
   stats.json       n, S1, S2 (exact integers), mean/std, the LUT as float32 hex bits,
                    per-class train statistics, SHA-1 of checksums.csv
   checksums.csv    image_id, crc_resized, crc_normalized (8-digit hex), image_id order

Usage
  python preprocessing/reference.py [--processed DIR] [--sets full 100MB 500MB] [-j N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import cubfmt

OUT = 224
FRAC_BITS = 11
ONE = 1 << FRAC_BITS
ROUND = 1 << (2 * FRAC_BITS - 1)
SHIFT = 2 * FRAC_BITS


# --------------------------------------------------------------------------- spec
def axis_table(length: int, out: int = OUT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(i0, i1, weight_of_i1) for one axis, exactly as in the spec above."""
    i = np.arange(out, dtype=np.int64)
    num = (2 * i + 1) * length - out
    pos = num > 0
    i0 = np.where(pos, num // (2 * out), 0)
    f = np.where(pos, num - i0 * 2 * out, 0)
    edge = i0 >= length - 1
    i0 = np.where(edge, length - 1, i0)
    f = np.where(edge, 0, f)
    i1 = np.minimum(i0 + 1, length - 1)
    wt = (f * ONE + out) // (2 * out)
    return i0, i1, wt


def crop_resize(pixels: np.ndarray, bx: int, by: int, bw: int, bh: int) -> np.ndarray:
    crop = pixels[by:by + bh, bx:bx + bw, :].astype(np.int64)
    x0, x1, wx = axis_table(bw)
    y0, y1, wy = axis_table(bh)
    wx = wx[None, :, None]
    wy = wy[:, None, None]
    top = (ONE - wx) * crop[y0][:, x0] + wx * crop[y0][:, x1]
    bot = (ONE - wx) * crop[y1][:, x0] + wx * crop[y1][:, x1]
    return (((ONE - wy) * top + wy * bot + ROUND) >> SHIFT).astype(np.uint8)


def float_resize(pixels: np.ndarray, bx: int, by: int, bw: int, bh: int) -> np.ndarray:
    """Plain float64 bilinear with the same coordinate mapping (self-check only)."""
    crop = pixels[by:by + bh, bx:bx + bw, :].astype(np.float64)

    def tab(length):
        s = np.clip((np.arange(OUT) + 0.5) * length / OUT - 0.5, 0, length - 1)
        i0 = np.floor(s).astype(np.int64)
        return i0, np.minimum(i0 + 1, length - 1), s - i0

    x0, x1, fx = tab(bw)
    y0, y1, fy = tab(bh)
    fx, fy = fx[None, :, None], fy[:, None, None]
    top = (1 - fx) * crop[y0][:, x0] + fx * crop[y0][:, x1]
    bot = (1 - fx) * crop[y1][:, x0] + fx * crop[y1][:, x1]
    return (1 - fy) * top + fy * bot


def finalize_stats(n: int, s1: list[int], s2: list[int]) -> dict:
    """Pass-1 integers -> mean/std/LUT, exactly as in the spec (Python floats are IEEE doubles)."""
    m = [float(s1[c]) / float(n) for c in range(3)]
    q = [float(s2[c]) / float(n) for c in range(3)]
    s = [math.sqrt(q[c] - m[c] * m[c]) for c in range(3)]
    lut = np.array([[(float(v) - m[c]) / s[c] for v in range(256)] for c in range(3)],
                   dtype=np.float64).astype(np.float32)
    return {"m_raw": m, "s_raw": s, "mean": [x / 255 for x in m], "std": [x / 255 for x in s],
            "lut": lut}


# --------------------------------------------------------------------------- workers
_SHARDS = None
_RECORDS = None
_LUT = None


def _init(set_dir: str, lut: np.ndarray | None) -> None:
    global _SHARDS, _RECORDS, _LUT
    _RECORDS, num_shards = cubfmt.read_index(Path(set_dir) / "index.bin")
    _SHARDS = cubfmt.open_shards(Path(set_dir), num_shards)
    _LUT = lut


def _resized(idx: int) -> np.ndarray:
    rec = _RECORDS[idx]
    return crop_resize(cubfmt.record_pixels(_SHARDS, rec),
                       int(rec["bx"]), int(rec["by"]), int(rec["bw"]), int(rec["bh"]))


def _pass1(idx: int) -> tuple[int, list[int], list[int]]:
    """Per-channel exact sums of one train record's resized image."""
    r = _resized(idx).reshape(-1, 3).astype(np.int64)
    return idx, r.sum(axis=0).tolist(), (r * r).sum(axis=0).tolist()


def _pass2(idx: int) -> tuple[int, int, int]:
    r = _resized(idx)
    norm = np.empty((3, OUT, OUT), dtype="<f4")
    for c in range(3):
        norm[c] = _LUT[c][r[:, :, c]]
    return idx, zlib.crc32(r.tobytes()), zlib.crc32(norm.tobytes())


def _selfcheck(idx: int) -> float:
    rec = _RECORDS[idx]
    px = cubfmt.record_pixels(_SHARDS, rec)
    box = (int(rec["bx"]), int(rec["by"]), int(rec["bw"]), int(rec["bh"]))
    return float(np.abs(crop_resize(px, *box).astype(np.float64) - float_resize(px, *box)).max())


# --------------------------------------------------------------------------- driver
def run_set(processed: Path, name: str, out_root: Path, jobs: int, n_selfcheck: int) -> dict:
    set_dir = processed / name
    records, _ = cubfmt.read_index(set_dir / "index.bin")
    train = [int(i) for i in np.flatnonzero(records["is_train"] == 1)]
    t0 = time.perf_counter()

    with Pool(jobs, initializer=_init, initargs=(str(set_dir), None)) as pool:
        rng = np.random.default_rng(0)
        sample = rng.choice(len(records), size=min(n_selfcheck, len(records)), replace=False)
        max_dev = max(pool.map(_selfcheck, sample.tolist())) if len(sample) else 0.0
        s1, s2 = [0, 0, 0], [0, 0, 0]
        cls_s1: dict[int, list[int]] = {}
        cls_s2: dict[int, list[int]] = {}
        cls_n: dict[int, int] = {}
        for idx, a, b in pool.imap_unordered(_pass1, train, chunksize=16):
            k = int(records["class_id"][idx])
            cs1, cs2 = cls_s1.setdefault(k, [0, 0, 0]), cls_s2.setdefault(k, [0, 0, 0])
            cls_n[k] = cls_n.get(k, 0) + OUT * OUT
            for c in range(3):
                s1[c] += a[c]
                s2[c] += b[c]
                cs1[c] += a[c]
                cs2[c] += b[c]
    t_pass1 = time.perf_counter() - t0
    if max_dev > 1.0:
        raise RuntimeError(f"fixed-point resize deviates {max_dev:.3f} from float bilinear")

    n = len(train) * OUT * OUT
    st = finalize_stats(n, s1, s2)
    crcs = np.zeros((len(records), 2), dtype=np.uint32)
    with Pool(jobs, initializer=_init, initargs=(str(set_dir), st["lut"])) as pool:
        for idx, cr, cn in pool.imap_unordered(_pass2, range(len(records)), chunksize=16):
            crcs[idx] = (cr, cn)
    t_total = time.perf_counter() - t0

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["image_id,crc_resized,crc_normalized"]
    lines += [f"{int(records['image_id'][i])},{crcs[i, 0]:08x},{crcs[i, 1]:08x}"
              for i in range(len(records))]
    csv_bytes = ("\n".join(lines) + "\n").encode()
    (out_dir / "checksums.csv").write_bytes(csv_bytes)

    per_class = {}
    for k in sorted(cls_n):
        f = finalize_stats(cls_n[k], cls_s1[k], cls_s2[k])
        per_class[str(k)] = {"n": cls_n[k], "S1": cls_s1[k], "S2": cls_s2[k],
                             "mean": f["mean"], "std": f["std"]}
    stats = {
        "set": name,
        "spec": "preprocessing/reference.py (fixed-point bilinear, exact integer sums, LUT)",
        "out_size": OUT,
        "records": int(len(records)),
        "train_records": len(train),
        "n": n, "S1": s1, "S2": s2,
        "mean": st["mean"], "std": st["std"],
        "lut_f32_hex": [[f"{int(b):08x}" for b in st["lut"][c].view(np.uint32)] for c in range(3)],
        "checksums_csv_sha1": hashlib.sha1(csv_bytes).hexdigest(),
        "selfcheck": {"images": int(len(sample)), "max_abs_dev_vs_float_bilinear": max_dev},
        "per_class_train": per_class,
        "seconds": {"pass1": round(t_pass1, 2), "total": round(t_total, 2)},
        "tool_versions": cubfmt.tool_versions(),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--processed", type=Path, default=cubfmt.DEFAULT_PROCESSED,
                    help="directory holding the sets (default: %(default)s)")
    ap.add_argument("--sets", nargs="+", default=["full", "100MB", "500MB"],
                    help="sets to compute references for (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=cubfmt.REPO_ROOT / "results" / "reference",
                    help="output root (default: %(default)s)")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--selfcheck", type=int, default=64,
                    help="images to compare against float bilinear (default: %(default)s)")
    args = ap.parse_args()

    for name in args.sets:
        if not (args.processed / name / "index.bin").is_file():
            print(f"error: set '{name}' not found in {args.processed}", file=sys.stderr)
            return 1
    for name in args.sets:
        st = run_set(args.processed, name, args.out, args.jobs, args.selfcheck)
        fmt = lambda xs: "[" + ", ".join(f"{x:.6f}" for x in xs) + "]"  # noqa: E731
        print(f"{name}: {st['records']} records ({st['train_records']} train) "
              f"mean={fmt(st['mean'])} std={fmt(st['std'])} "
              f"resize self-check max dev {st['selfcheck']['max_abs_dev_vs_float_bilinear']:.3f} "
              f"| {st['seconds']['total']:.1f} s -> {args.out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
