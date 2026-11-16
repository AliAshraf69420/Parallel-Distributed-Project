"""Shared definitions for the CUB preprocessing scripts.

Contents:
  * the on-disk binary format (index + shards) that every phase reads
    (src/common/record_io must match this file exactly)
  * the single JPEG -> RGB decode function, so validate.py and predecode.py
    produce bit-identical pixels
  * small helpers: paths, CSV manifest I/O

Binary format, version 1 (all little-endian)
--------------------------------------------
<set_dir>/index.bin
    header (32 bytes):
        char[8]  magic        = b"CUBIDX01"
        u32      version      = 1
        u32      record_size  = 32
        u64      count        number of records
        u32      num_shards
        u32      reserved     = 0
    count x record (32 bytes each, naturally aligned):
        u64 offset     byte offset of the pixel block inside its shard (multiple of 64)
        u32 image_id   original CUB image id (1-based, from images.txt)
        u16 class_id   0-based class (CUB label - 1), 0..199
        u8  is_train   1 = train split, 0 = test split
        u8  flags      bit 0: source image was not RGB and was converted
        u16 w, h       image width / height in pixels
        u16 bx, by     sanitized bounding box top-left (inclusive), in pixels
        u16 bw, bh     sanitized bounding box size, in pixels (>= MIN_BBOX)
        u16 shard      shard number k -> file shard_{k:04d}.bin
        u16 pad        = 0
<set_dir>/shard_KKKK.bin
    concatenated pixel blocks; each block is h*w*3 bytes, HWC, RGB, uint8,
    starting at a 64-byte aligned offset (zero padding between blocks).
<set_dir>/meta.json
    human-readable description of the set (counts, sizes, tool versions).
"""

from __future__ import annotations

import csv
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "data" / "raw" / "CUB_200_2011"
DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed"
DEFAULT_REPORTS = REPO_ROOT / "results" / "preprocessing"

FORMAT_VERSION = 1
MAGIC = b"CUBIDX01"
HEADER = struct.Struct("<8sIIQII")  # 32 bytes
ALIGN = 64
MIN_BBOX = 8  # boxes narrower/shorter than this (after clamping) are dropped
FLAG_CONVERTED = 1

RECORD_DTYPE = np.dtype(
    [
        ("offset", "<u8"),
        ("image_id", "<u4"),
        ("class_id", "<u2"),
        ("is_train", "u1"),
        ("flags", "u1"),
        ("w", "<u2"),
        ("h", "<u2"),
        ("bx", "<u2"),
        ("by", "<u2"),
        ("bw", "<u2"),
        ("bh", "<u2"),
        ("shard", "<u2"),
        ("pad", "<u2"),
    ]
)
assert HEADER.size == 32
assert RECORD_DTYPE.itemsize == 32

MANIFEST_FIELDS = [
    "image_id", "rel_path", "class_id", "is_train", "w", "h", "mode",
    "bx", "by", "bw", "bh", "bbox_clamped", "sha1", "file_bytes",
]


# --------------------------------------------------------------------------- decode
def decode_rgb(path: Path) -> tuple[np.ndarray, str]:
    """Fully decode an image to a contiguous HxWx3 uint8 RGB array.

    Returns (pixels, original_mode). Raises on unreadable or truncated files
    (Pillow's LOAD_TRUNCATED_IMAGES stays at its default, False).
    """
    with Image.open(path) as im:
        mode = im.mode
        im.load()
        rgb = im if mode == "RGB" else im.convert("RGB")
        arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"unexpected decoded shape {arr.shape}")
    return arr, mode


# --------------------------------------------------------------------------- bbox
def sanitize_bbox(x: float, y: float, w: float, h: float, img_w: int, img_h: int):
    """Turn CUB's float box (x, y, w, h; 0-based, top-left origin) into integer
    pixel bounds covering it, clamped to the image.

    Returns (bx, by, bw, bh, clamped) or None if the clamped box is degenerate.
    """
    x0, y0 = math.floor(x), math.floor(y)
    x1, y1 = math.ceil(x + w), math.ceil(y + h)
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(img_w, x1), min(img_h, y1)
    clamped = (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1)
    bw, bh = cx1 - cx0, cy1 - cy0
    if bw < MIN_BBOX or bh < MIN_BBOX:
        return None
    return cx0, cy0, bw, bh, clamped


# --------------------------------------------------------------------------- index
def align_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def shard_name(k: int) -> str:
    return f"shard_{k:04d}.bin"


def write_index(path: Path, records: np.ndarray, num_shards: int) -> None:
    assert records.dtype == RECORD_DTYPE
    with open(path, "wb") as f:
        f.write(HEADER.pack(MAGIC, FORMAT_VERSION, RECORD_DTYPE.itemsize,
                            len(records), num_shards, 0))
        f.write(records.tobytes())


def read_index(path: Path) -> tuple[np.ndarray, int]:
    raw = Path(path).read_bytes()
    magic, version, rec_size, count, num_shards, _ = HEADER.unpack_from(raw, 0)
    if magic != MAGIC or version != FORMAT_VERSION or rec_size != RECORD_DTYPE.itemsize:
        raise ValueError(f"{path}: not a CUB index v{FORMAT_VERSION}")
    records = np.frombuffer(raw, dtype=RECORD_DTYPE, count=count, offset=HEADER.size)
    return records, num_shards


def open_shards(set_dir: Path, num_shards: int) -> list[np.memmap]:
    return [np.memmap(set_dir / shard_name(k), dtype=np.uint8, mode="r")
            for k in range(num_shards)]


def record_pixels(shards: list[np.memmap], rec) -> np.ndarray:
    n = int(rec["h"]) * int(rec["w"]) * 3
    off = int(rec["offset"])
    return shards[int(rec["shard"])][off:off + n].reshape(int(rec["h"]), int(rec["w"]), 3)


# --------------------------------------------------------------------------- manifest
def write_manifest(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        wr.writeheader()
        wr.writerows(rows)


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
