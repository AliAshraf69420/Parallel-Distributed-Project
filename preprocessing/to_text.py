#!/usr/bin/env python3
"""Step 5 of preprocessing: export binary sets as line-based text for Hadoop Streaming.

Hadoop Streaming reads and writes text lines, so every image becomes one line of
tab-separated fields (no header line):

    image_id  class_id  is_train  w  h  bx  by  bw  bh  base64(pixels)

The first nine fields are the index record of the image (see cubfmt.py); the last is its
raw HWC RGB uint8 pixels (h*w*3 bytes) in standard base64 without line breaks. Lines are in
the set's record order (image_id). A line is about 1 MB; base64 makes the text 4/3 the size
of the binary set.

Output: data/processed/text/<set>.txt and results/preprocessing/text.json (sizes, line
counts, expansion factor). Every line is decoded again and compared before it is counted.

Usage
  python preprocessing/to_text.py [--processed DIR] [--sets 100MB 500MB full]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cubfmt


def export(processed: Path, name: str) -> dict:
    set_dir = processed / name
    records, num_shards = cubfmt.read_index(set_dir / "index.bin")
    shards = cubfmt.open_shards(set_dir, num_shards)
    out_dir = processed / "text"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp, final = out_dir / f".{name}.txt.tmp", out_dir / f"{name}.txt"
    t0 = time.perf_counter()
    pixel_bytes = 0
    with open(tmp, "wb", buffering=8 << 20) as f:
        for rec in records:
            px = cubfmt.record_pixels(shards, rec)
            raw = px.tobytes()
            b64 = base64.b64encode(raw)
            if base64.b64decode(b64) != raw:
                raise RuntimeError(f"base64 round trip failed for image_id {rec['image_id']}")
            head = "\t".join(str(int(rec[k])) for k in
                             ("image_id", "class_id", "is_train", "w", "h", "bx", "by", "bw", "bh"))
            f.write(head.encode() + b"\t" + b64 + b"\n")
            pixel_bytes += len(raw)
    tmp.rename(final)
    size = final.stat().st_size
    return {"set": name, "file": str(final.relative_to(processed.parent.parent)),
            "lines": int(len(records)), "text_bytes": size, "pixel_bytes": pixel_bytes,
            "expansion": round(size / pixel_bytes, 4),
            "mean_line_bytes": round(size / len(records)),
            "seconds": round(time.perf_counter() - t0, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--processed", type=Path, default=cubfmt.DEFAULT_PROCESSED)
    ap.add_argument("--sets", nargs="+", default=["100MB", "500MB", "full"])
    ap.add_argument("--reports", type=Path, default=cubfmt.DEFAULT_REPORTS)
    args = ap.parse_args()

    for name in args.sets:
        if not (args.processed / name / "index.bin").is_file():
            print(f"error: set '{name}' not found in {args.processed}", file=sys.stderr)
            return 1
    report_path = args.reports / "text.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for name in args.sets:
        r = export(args.processed, name)
        report[name] = r
        print(f"{name}: {r['lines']} lines, {r['text_bytes'] / 2**20:.1f} MiB text "
              f"({r['expansion']}x the binary pixels), {r['seconds']} s -> {r['file']}")
    args.reports.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
