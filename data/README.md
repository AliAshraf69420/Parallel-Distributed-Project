# Data

The full dataset is **not** in git (`data/raw/` and `data/processed/` are gitignored).
Only `data/samples/` is committed.

## 1. Download

1. Download `CUB_200_2011.tgz` (≈ 1.1 GB) from the official CUB-200-2011 page on
   Caltech Data. The segmentation masks are a separate download and are not needed.
2. Put it in `data/raw/` and extract it there:

```bash
cd data/raw
tar -xzf CUB_200_2011.tgz
```

The result must be `data/raw/CUB_200_2011/` containing `images/`, `images.txt`,
`image_class_labels.txt`, `train_test_split.txt`, `bounding_boxes.txt` and `classes.txt`.
Treat this directory as read-only. All changes are made by the scripts below.

## 2. Preprocess

From the repository root, inside the project's conda environment:

```bash
conda env create -f environment.yml   # once
conda activate sw401
```

Then:

```bash
python3 preprocessing/validate.py    # -> data/processed/manifest.csv + results/preprocessing/
python3 preprocessing/predecode.py   # -> data/processed/full/{index.bin, shard_*.bin, meta.json}
```

| Output | Contents |
|---|---|
| `data/processed/manifest.csv` | the kept images: id, path, class, split, size, sanitized bbox, pixel SHA-1 |
| `results/preprocessing/validation.json` | before/after counts and sizes, drop reasons, RGB conversions, size distribution |
| `results/preprocessing/dropped.csv` | every dropped image and why |
| `data/processed/full/` | the predecoded binary set (format documented in `preprocessing/cubfmt.py`) |

## 3. Layout

```
data/
├── raw/CUB_200_2011/   # official files, untouched
├── processed/          # everything generated from raw/
└── samples/            # small committed samples for demos and tests
```
