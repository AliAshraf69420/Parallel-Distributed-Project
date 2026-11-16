# CUB-200-2011 Predecoding & Normalization Pipeline

SW401 (Parallel and Distributed Systems) course project. It turns the CUB-200-2011
bird images into bounding-box-cropped, 224×224, per-channel-normalized tensors, and
implements that same computation as sequential C++, OpenMP, C++ threads, MPI and
Hadoop Streaming. Every implementation must match one frozen reference **bit for bit**.

- Full plan and design: [docs/PLAN.md](docs/PLAN.md)
- Dataset and binary format: [data/README.md](data/README.md), [preprocessing/cubfmt.py](preprocessing/cubfmt.py)
- Exact spec of the core computation: [preprocessing/reference.py](preprocessing/reference.py)

| Part | Status |
|---|---|
| Preprocessing (validate, predecode, subsets, reference) | Done |
| Sequential baseline, OpenMP, C++ thread pipeline, correctness checker | Done |
| Phase 1 benchmarks, plots, memory-hierarchy experiments | Done |
| Phase 1 report and video | To do |
| Phase 2: MPI | To do |
| Phase 3: Hadoop Streaming | To do |

---

## First-time setup and run order

Run every command **from the repository root**.

### 0. Prerequisites

| Tool | Needed for | Install (Arch / Ubuntu) |
|---|---|---|
| git, make, g++ ≥ 11 with OpenMP | everything | `pacman -S base-devel` / `apt install build-essential` |
| conda (Miniconda or Anaconda) | Python preprocessing | <https://docs.conda.io/en/latest/miniconda.html> |
| perf | Phase 1 cache experiment | `pacman -S perf` / `apt install linux-tools-$(uname -r)` |
| OpenMPI | Phase 2 | `pacman -S openmpi` / `apt install openmpi-bin libopenmpi-dev` |
| Docker | Phase 3 (Hadoop cluster) | distro package |

Disk: about **28 GB** free (1.1 GB download, 1.2 GB extracted, 5.9 GB predecoded full set,
0.6 GB subsets, 10.5 GB weak-scaling subsets, plus outputs). RAM: 8 GB or more.

### 1. Clone and create the Python environment

```bash
git clone git@github.com:AliAshraf69420/Parallel-Distributed-Project.git
cd Parallel-Distributed-Project
conda env create -f environment.yml      # once; later: conda env update -f environment.yml --prune
conda activate sw401                     # every new terminal
```

Use this environment: it pins Pillow and libjpeg-turbo, so everyone decodes the JPEGs to
identical pixels.

### 2. Download the dataset

Download `CUB_200_2011.tgz` (≈ 1.1 GB) from the official CUB-200-2011 page on Caltech Data
and extract it into `data/raw/`:

```bash
cd data/raw && tar -xzf /path/to/CUB_200_2011.tgz && cd ../..
# result: data/raw/CUB_200_2011/{images/, images.txt, bounding_boxes.txt, ...}
```

### 3. Preprocess (in this order)

Each step reads the previous step's output. Or run all of them with `make data`.

| # | Command | Produces | Time* |
|---|---|---|---|
| 1 | `python preprocessing/validate.py` | `data/processed/manifest.csv`, `results/preprocessing/{validation.json, dropped.csv}` | ~5 s |
| 2 | `python preprocessing/predecode.py` | `data/processed/full/` (index + 6 shards, 5.9 GiB, verified) | ~25 s |
| 3 | `python preprocessing/make_subsets.py` | `data/processed/{100MB,500MB}/` | ~5 s |
| 4 | `python preprocessing/reference.py` | `results/reference/{full,100MB,500MB}/{stats.json, checksums.csv}` | ~100 s |
| 5 | `python preprocessing/make_subsets.py --sizes 700 1400 2800 5600 --reports results/preprocessing/weak` | weak-scaling sets `data/processed/{700,1400,2800,5600}MB/` (p × 700 MiB) | ~20 s |
| 6 | `python preprocessing/reference.py --sets 700MB 1400MB 2800MB 5600MB` | their references | ~150 s |

\* Ryzen 7 5800H (8 cores / 16 threads), NVMe SSD.

**Expected results.** If any of these differ, stop and ask before going further:
- `validate.py` drops **exactly 1 image**: id 3340, a byte-identical duplicate of id 3310.
  It converts 8 grayscale images to RGB and clamps 30 bounding boxes.
- After preprocessing: **11,787 images** (5,993 train / 5,794 test), 200 classes.
- `reference.py` prints `full: ... mean=[0.471774, 0.470920, 0.405665] std=[0.235584, 0.231691, 0.258334]`.
- Once `results/reference/` is committed, `git status results/reference/` should show no
  changes to any `checksums.csv`. That proves your preprocessing matches everyone else's.
  (`stats.json` also stores run times, so only that file may differ.)

### 4. Build the C++ implementations

```bash
make            # -> build/{seq,omp,threads,memexp}   (flags: -O3 -march=native -ffp-contract=off)
```

### 5. Run and verify

```bash
make test       # runs seq, omp and threads on the 100MB set and checks each against the reference
```

Each run of `build/<version>` writes `stats.json` and `checksums.csv` to
`results/runs/<version>/<set>/` (gitignored). `tests/check.py` compares them with
`results/reference/<set>/` and prints `PASS` or `FAIL`:

```bash
build/seq --set full --preload
python tests/check.py --set full --run results/runs/seq/full

build/omp --set full -t 8 --schedule dynamic,8 --preload --csv results/timings/omp.csv
python tests/check.py --set full --run results/runs/omp/full      # or: make check V=omp SET=full
```

Options common to every implementation (`build/seq --help`):

| Option | Meaning |
|---|---|
| `--set SET` | `full`, `500MB`, `100MB` (or a path to a set directory) |
| `-t N` | threads (parallel versions) |
| `--schedule S` | OpenMP schedule: `static[,k]`, `dynamic[,k]`, `guided[,k]` |
| `--order O` | processing order: `id` (storage order), `area-desc`, `area-asc` (by crop area) |
| `--preload` | read the set into the page cache before timing, so the times measure compute rather than disk |
| `--cache` | resize every image once in pass 1 and reuse it in pass 2 (~1.8 GB RAM for `full`) |
| `--csv FILE` | append one timing row (load, pass1, merge, pass2, write, total) |
| `--write FILE` | also write all normalized tensors (n × 602,112 bytes; ≈ 7.1 GB for `full`) |
| `--record-times FILE` | `seq` only: per-image pass-2 time and crop size |

Reference timings on the full set with `--preload`: `seq` ≈ 12.3 s, and `omp -t 8` / `threads -t 8` ≈ 1.6 s.

### 6. Phase 1 benchmarks and plots

Close other programs first (the runner warns if the load average is high).

```bash
python scripts/bench/phase1.py       # ~10 min: strong, schedule, weak, per-image times
python plots/phase1.py               # -> plots/phase1/*.png, results/timings/phase1_summary.md
make report                          # -> report.pdf (LaTeX source in report/, needs latexmk)
```

Every configuration gets one warm-up run plus 3 measured runs (the median is reported), uses
`--preload`, and pins threads one per physical core (`OMP_PROC_BIND=close OMP_PLACES=cores`).
The first measured run of each configuration is checked bit-exactly; a FAIL aborts the run.
Raw rows go to `results/timings/phase1_*.csv`, and machine details to `phase1_env.json`.

### 7. Memory-hierarchy experiments

```bash
build/memexp wss                     # load latency vs working-set size, pointer chase (L1/L2/L3/DRAM)
build/memexp stride                  # time per access vs stride
build/memexp resize-order            # row-major vs column-major resize (identical output)
build/memexp false-sharing -t 8      # packed vs padded vs local per-thread counters
perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
    build/memexp resize-order --variant col    # repeat with --variant row
```

The report's numbers were produced with each command's output redirected to
`results/timings/memexp/<name>.csv` (and `perf stat -x, -o results/timings/memexp/perf_<name>.csv`);
`python plots/phase1.py` turns them into `plots/phase1/{latency,stride}.png` and tables.

---

## Repository layout

```
├── environment.yml       conda env `sw401` (Python side)
├── Makefile              build / data / test / check
├── docs/PLAN.md          project plan (all phases)
├── preprocessing/        Python: validate → predecode → subsets → reference
├── src/common/           shared C++: set reader, kernels (resize/stats/normalize/CRC), driver
├── src/seq/, src/omp/    one main.cpp each: only the loops differ
├── src/threads/          std::thread pipeline: reader -> workers -> writer, bounded queues
├── src/memexp/           memory-hierarchy and false-sharing experiments
├── scripts/bench/        benchmark runners (phase1.py)
├── plots/                plot + table scripts and their PNG output
├── tests/check.py        bit-exact comparison against results/reference/
├── data/                 raw/ and processed/ are gitignored; README has download steps
└── results/              preprocessing reports, reference outputs, timings (runs/ gitignored)
```

## Troubleshooting

- **`no set at 'full'`**: run from the repo root, or pass `--set data/processed/full`.
- **`ModuleNotFoundError: numpy`**: the conda env is not active (`conda activate sw401`).
- **Times are much slower the first time**: the set was not in the page cache yet. Use `--preload`.
- **`check.py` reports FAIL after you changed C++ code**: `resize differs` points at
  `crop_resize`, `normalize differs` points at the stats, LUT or `normalize_chw`. Don't
  remove `-ffp-contract=off`.
