#!/usr/bin/env python3
"""Step 1 of preprocessing: validate CUB-200-2011 and write the clean manifest.

Checks, in this order (first failure decides the drop reason):
  1. missing_meta   image lacks a class label, split flag or bounding box
  2. missing_file   listed in images.txt but not on disk
  3. corrupt        cannot be fully decoded (truncated / malformed)
  4. bad_bbox       box is degenerate after clamping to the image (< MIN_BBOX px)
  5. duplicate      decoded RGB pixels identical to an earlier (lower id) image
Non-RGB images (grayscale, CMYK, ...) are kept and converted to RGB; they are
counted, and the conversion is flagged in the binary index later.

Outputs
  data/processed/manifest.csv            kept images, one row each, sorted by image_id
  results/preprocessing/validation.json  before/after statistics for the report
  results/preprocessing/dropped.csv      every dropped image with its reason

Usage
  python preprocessing/validate.py [--root DIR] [--out DIR] [--reports DIR] [-j N]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import PIL
from PIL import features

import cubfmt

META_FILES = ["images.txt", "image_class_labels.txt", "train_test_split.txt",
              "bounding_boxes.txt", "classes.txt"]

_IMAGES_DIR: Path | None = None  # set in each worker by _init_worker


def _init_worker(images_dir: str) -> None:
    global _IMAGES_DIR
    _IMAGES_DIR = Path(images_dir)


def _inspect(job: tuple[int, str]) -> dict:
    """Decode one image (in a worker process) and report what we learned."""
    image_id, rel = job
    path = _IMAGES_DIR / rel
    if not path.is_file():
        return {"image_id": image_id, "status": "missing_file", "error": ""}
    file_bytes = path.stat().st_size
    try:
        arr, mode = cubfmt.decode_rgb(path)
    except Exception as e:  # noqa: BLE001 - any decode failure means corrupt
        return {"image_id": image_id, "status": "corrupt", "file_bytes": file_bytes,
                "error": f"{type(e).__name__}: {e}"[:300]}
    h, w, _ = arr.shape
    return {"image_id": image_id, "status": "ok", "mode": mode, "w": w, "h": h,
            "file_bytes": file_bytes, "sha1": hashlib.sha1(arr.data).hexdigest()}


def _read_table(path: Path, ncols: int) -> dict[int, list[str]]:
    """Parse a whitespace-separated CUB metadata file keyed by its first column."""
    table: dict[int, list[str]] = {}
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != ncols:
                raise ValueError(f"{path.name}:{lineno}: expected {ncols} columns, got {len(parts)}")
            table[int(parts[0])] = parts[1:]
    return table


def _describe(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {}
    return {"min": float(a.min()), "p05": float(np.percentile(a, 5)),
            "median": float(np.median(a)), "mean": float(a.mean()),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=cubfmt.DEFAULT_RAW,
                    help="extracted CUB_200_2011 directory (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=cubfmt.DEFAULT_PROCESSED,
                    help="where manifest.csv goes (default: %(default)s)")
    ap.add_argument("--reports", type=Path, default=cubfmt.DEFAULT_REPORTS,
                    help="where the report files go (default: %(default)s)")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                    help="decoder processes (default: all CPUs)")
    args = ap.parse_args()

    root: Path = args.root
    missing = [m for m in META_FILES if not (root / m).is_file()]
    if missing:
        print(f"error: {root} is missing {', '.join(missing)}; "
              "extract CUB_200_2011.tgz into data/raw/ first", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # ---- metadata ---------------------------------------------------------
    images = _read_table(root / "images.txt", 2)
    labels = _read_table(root / "image_class_labels.txt", 2)
    split = _read_table(root / "train_test_split.txt", 2)
    boxes = _read_table(root / "bounding_boxes.txt", 5)
    classes = _read_table(root / "classes.txt", 2)

    ids = sorted(images)
    dropped: list[dict] = []
    jobs: list[tuple[int, str]] = []
    for i in ids:
        if i not in labels or i not in split or i not in boxes:
            dropped.append({"image_id": i, "rel_path": images[i][0],
                            "reason": "missing_meta", "detail": ""})
        else:
            jobs.append((i, images[i][0]))

    # ---- decode everything (parallel) ---------------------------------------
    results: dict[int, dict] = {}
    with Pool(args.jobs, initializer=_init_worker, initargs=(str(root / "images"),)) as pool:
        for n, r in enumerate(pool.imap_unordered(_inspect, jobs, chunksize=32), 1):
            results[r["image_id"]] = r
            if n % 1000 == 0 or n == len(jobs):
                print(f"  decoded {n}/{len(jobs)}", file=sys.stderr)

    # ---- apply checks in image_id order -------------------------------------
    kept: list[dict] = []
    seen_sha: dict[str, int] = {}
    modes_converted: Counter = Counter()
    formats_bytes_before = 0
    files_found = 0
    n_clamped = 0
    for i, rel in jobs:
        r = results[i]
        if r["status"] != "missing_file":
            files_found += 1
            formats_bytes_before += r["file_bytes"]
        if r["status"] != "ok":
            dropped.append({"image_id": i, "rel_path": rel, "reason": r["status"],
                            "detail": r.get("error", "")})
            continue
        x, y, bw, bh = map(float, boxes[i])
        box = cubfmt.sanitize_bbox(x, y, bw, bh, r["w"], r["h"])
        if box is None:
            dropped.append({"image_id": i, "rel_path": rel, "reason": "bad_bbox",
                            "detail": f"box {x},{y},{bw},{bh} on {r['w']}x{r['h']}"})
            continue
        if r["sha1"] in seen_sha:
            dropped.append({"image_id": i, "rel_path": rel, "reason": "duplicate",
                            "detail": f"same pixels as image_id {seen_sha[r['sha1']]}"})
            continue
        seen_sha[r["sha1"]] = i
        if r["mode"] != "RGB":
            modes_converted[r["mode"]] += 1
        n_clamped += box[4]
        kept.append({
            "image_id": i, "rel_path": rel, "class_id": int(labels[i][0]) - 1,
            "is_train": int(split[i][0]), "w": r["w"], "h": r["h"], "mode": r["mode"],
            "bx": box[0], "by": box[1], "bw": box[2], "bh": box[3],
            "bbox_clamped": int(box[4]), "sha1": r["sha1"], "file_bytes": r["file_bytes"],
        })

    dropped.sort(key=lambda d: d["image_id"])
    cubfmt.write_manifest(args.out / "manifest.csv", kept)
    with open(args.reports / "dropped.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["image_id", "rel_path", "reason", "detail"])
        wr.writeheader()
        wr.writerows(dropped)

    # ---- before / after statistics -----------------------------------------
    reasons = Counter(d["reason"] for d in dropped)
    before_train = sum(1 for i in ids if i in split and split[i][0] == "1")
    decoded_bytes = sum(k["w"] * k["h"] * 3 for k in kept)
    report = {
        "source": str(root),
        "tool_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                          "pillow": PIL.__version__,
                          "libjpeg_turbo": features.version("libjpeg_turbo")},
        "seconds": round(time.perf_counter() - t0, 2),
        "before": {
            "records_listed": len(ids),
            "image_files_found": files_found,
            "jpeg_bytes_on_disk": formats_bytes_before,
            "train": before_train,
            "test": len(ids) - before_train,
            "classes": len(classes),
        },
        "checks": {
            "dropped_total": len(dropped),
            "dropped_by_reason": {k: reasons.get(k, 0) for k in
                                  ["missing_meta", "missing_file", "corrupt", "bad_bbox", "duplicate"]},
            "converted_to_rgb": dict(modes_converted),
            "bbox_clamped": n_clamped,
            "min_bbox_px": cubfmt.MIN_BBOX,
        },
        "after": {
            "records": len(kept),
            "train": sum(k["is_train"] for k in kept),
            "test": sum(1 - k["is_train"] for k in kept),
            "classes": len({k["class_id"] for k in kept}),
            "decoded_rgb_bytes": decoded_bytes,
            "decoded_rgb_gib": round(decoded_bytes / 2**30, 3),
        },
        # Spread of per-image work: drives the OpenMP scheduling discussion.
        "size_stats": {
            "width": _describe([k["w"] for k in kept]),
            "height": _describe([k["h"] for k in kept]),
            "image_pixels": _describe([k["w"] * k["h"] for k in kept]),
            "bbox_pixels": _describe([k["bw"] * k["bh"] for k in kept]),
        },
    }
    with open(args.reports / "validation.json", "w") as f:
        json.dump(report, f, indent=2)

    b, c, a = report["before"], report["checks"], report["after"]
    print(f"before: {b['records_listed']} listed, {b['image_files_found']} files, "
          f"{b['jpeg_bytes_on_disk'] / 2**20:.1f} MiB JPEG")
    print(f"dropped: {c['dropped_total']} {c['dropped_by_reason']}")
    print(f"converted to RGB: {c['converted_to_rgb'] or 'none'}; bbox clamped: {c['bbox_clamped']}")
    print(f"after: {a['records']} kept ({a['train']} train / {a['test']} test, "
          f"{a['classes']} classes), {a['decoded_rgb_gib']} GiB decoded RGB")
    print(f"wrote {args.out / 'manifest.csv'} and {args.reports}/ in {report['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
