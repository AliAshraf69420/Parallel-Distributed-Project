# SW401 Project Plan — CUB-200-2011 Predecoding & Normalization Pipeline

> One project with two uses: the SW401 parallel and distributed systems project, and a
> proof-of-concept data pipeline for a separate ML project trained on CUB-200-2011.

---

## 1. Summary

We build a pipeline that turns the CUB-200-2011 bird images into **fixed-size,
per-channel-normalized tensors** that a model can train on directly. The same computation
is implemented four times: sequential C++, OpenMP, MPI and Hadoop Streaming. Each version
is checked against the same reference output.

**Core computation (identical in every phase):**

1. Crop each decoded image to its ground-truth bounding box.
2. Resize the crop to **224×224** with bilinear interpolation (our own C++ kernel).
3. **Pass 1:** compute the per-channel (R, G, B) mean and standard deviation over the
   **training split** (a global reduction).
4. **Pass 2:** normalize every image with those statistics,
   `x' = (x/255 − μ_c) / σ_c`, and write CHW float tensors.

JPEG decoding happens **once**, during preprocessing ("predecoding"). Every later phase
starts from raw pixels, so the measured speedups come from our own code, not from libjpeg.

---

## 2. Why an image dataset needs per-channel mean/std

The mean/std here is **not** a statistic of the target class. It is a property of the
*input distribution*. It is used to standardize every pixel before it enters the network.

### 2.1 What normalization does for training

| Reason | Explanation |
|---|---|
| **Zero-centered inputs** | Raw pixels are all positive (0–255). If every input is positive, the gradients for a neuron's incoming weights all have the same sign, so the weights can only move in zig-zag steps. Subtracting the mean removes this (LeCun et al., *Efficient BackProp*). |
| **Equal scale across channels** | Channels have different spreads. Bird photos are dominated by green foliage and brown bark, for example. Dividing by σ gives each channel roughly unit variance, which improves the conditioning of the loss surface so one learning rate works for all first-layer weights. |
| **Stable activations** | Weight initialization schemes (Xavier/He) assume inputs with roughly unit variance. Without normalization, early activations can saturate or explode. |
| **Matching a pretrained backbone** | A network pretrained on ImageNet expects ImageNet's statistics (mean ≈ [0.485, 0.456, 0.406], std ≈ [0.229, 0.224, 0.225]). For fine-tuning you would usually reuse those. For training from scratch, or for checking how far CUB is from ImageNet, you need CUB's own values, and that is what this pipeline computes. |

### 2.2 Why per *channel* and not per pixel or per image

- **Per pixel** (a "mean image" of 224×224×3 values) was used in early CNNs. Convolutions
  are translation-invariant, though, so position-specific statistics add little. Per-pixel
  stats also tie the result to one input resolution.
- **Per image** (normalize each image by its own mean/std) removes real information
  such as overall brightness and color cast, which can be useful cues for species.
- **Per channel over the whole dataset** keeps differences between images and still fixes
  the scale. It is only 6 numbers (3 means and 3 stds), so it is cheap to store and apply.

### 2.3 Why it cannot be per class

At inference time the class is **unknown**, since predicting it is the whole point. So the
normalization must be the same for every input. If we normalized with per-class statistics,
the label would leak into the input during training, and the transform could not be
applied at test time at all.

(Per-class statistics, such as mean color histograms per species, are still useful for
**analysis**. They appear in this plan only as an optional extra and as a Hadoop key, see §7.)

### 2.4 Rules we follow

- **Train split only.** Statistics are computed on the 5,994 training images and then
  applied to both train and test. Including test images would leak test information.
- **Same representation the model sees.** Statistics are computed *after* crop and resize,
  because that is the data the network receives.
- **Pixel scale.** Statistics are reported on the [0, 1] scale, as in ImageNet's numbers.

---

## 3. Dataset

| Item | Value |
|---|---|
| Source | CUB-200-2011 (Caltech-UCSD Birds), official tarball `CUB_200_2011.tgz` |
| Images | 11,788 JPEGs, 200 species |
| Split | 5,994 train / 5,794 test (`train_test_split.txt`) |
| Annotations used | `images.txt`, `image_class_labels.txt`, `bounding_boxes.txt`, `train_test_split.txt`, `classes.txt` |
| Compressed size | ≈ 1.1 GB |
| Decoded raw RGB size | ≈ 6–7 GB (estimate, to be measured in preprocessing) |

### 3.1 Size justification (Step 0 approval)

The brief asks for ≥ 1 GB or about 10M records. Our argument:

- **Decoded**, the dataset is ≈ 6–7 GB, and decoded pixels are what every phase processes.
- Pixels per channel processed in pass 1 ≈ 5,994 × 224 × 224 ≈ **3.0 × 10⁸**.
- For weak scaling and bigger runs we can grow the dataset with **deterministic
  augmentation** (horizontal flip plus K fixed jittered crops per image, with seeded RNG),
  for example ×4 → about 47k records and ≈ 25 GB decoded.

**Action:** get explicit instructor approval for this framing before Week 2. Fallback if it
is rejected: the same pipeline on a larger bird dataset (e.g. NABirds).

---

## 4. Repository layout

```
Parallel-Project/
├── README.md                 # how to reproduce everything
├── docs/
│   ├── PLAN.md               # this file
│   └── SW401–CourseProjectDescription.pdf
├── preprocessing/            # Python (allowed for preprocessing only)
│   ├── download.sh
│   ├── validate.py           # integrity, mode, duplicates
│   ├── predecode.py          # JPEG → raw RGB shards + index
│   ├── make_subsets.py       # 100 MB / 500 MB / full
│   ├── to_text.py            # binary → Hadoop line format
│   └── reference.py          # NumPy reference output for correctness checks
├── src/
│   ├── common/               # record I/O, bilinear resize, stats merge, timing
│   ├── seq/                  # sequential baseline
│   ├── omp/                  # OpenMP
│   ├── threads/              # C++ thread-pool pipeline
│   ├── mpi/                  # MPI
│   ├── cuda/                 # optional bonus
│   └── hadoop/               # mapper.cpp, combiner.cpp, reducer.cpp
├── scripts/                  # benchmark runners, mpirun + hostfile, hadoop jobs
├── data/                     # small samples + download instructions only
├── results/                  # raw timing CSVs
├── plots/
└── report/
```

---

## 5. Phase 1: Parallel Foundations (deadline: Week 7)

### 5.1 Preprocessing (`preprocessing/`, Python + NumPy + Pillow)

| Step | What | Reported statistic |
|---|---|---|
| 1. Integrity | Open every JPEG and fully decode it. Drop files that fail. | # corrupt |
| 2. Color mode | CUB contains a few non-RGB images (grayscale, etc.). Convert to RGB and log the IDs. | # converted |
| 3. Duplicates | Hash the decoded pixels (SHA-1). Drop exact duplicates and keep the first ID. | # duplicates |
| 4. Annotation join | Join path, class, split and bbox by `image_id`. Drop images with missing metadata. | # missing |
| 5. BBox sanitize | Clamp boxes to the image bounds, round to integers, and drop degenerate boxes (w or h < 8 px). | # clamped / dropped |
| 6. Predecode | Write raw HWC uint8 pixels at **original size** to binary shards, plus an index. | bytes before/after |
| 7. Subsets | Take class-stratified samples of ≈ 100 MB and ≈ 500 MB (seeded). | records and bytes per subset |
| 8. Reference | Compute the expected output in NumPy (see §5.3). | — |

**Binary record format** (little-endian, 64-byte aligned):

```
index.bin   : N × { u64 offset; u32 image_id; u16 class_id; u8 is_train; u8 pad;
                    u16 w; u16 h; u16 bx, by, bw, bh; }
shard_k.bin : concatenated raw HWC uint8 pixel blocks
```

We keep the original size (not pre-resized) on purpose. Images differ in size, so the
work per record is **uneven**, which makes the OpenMP scheduling comparison meaningful.
The index lets any worker jump to any record, which is needed for MPI partitioning.

### 5.2 Sequential baseline (`src/seq/`, C++17)

```
load index
for each record:                         # pass 1 (train records only)
    crop  = bbox_crop(raw)
    img   = bilinear_resize(crop, 224, 224)   # uint8 HWC
    stats.accumulate(img)                # per-channel n, Σx, Σx² in double
μ, σ = stats.finalize()
for each record:                         # pass 2 (all records)
    img = bilinear_resize(bbox_crop(raw))
    out = normalize_to_chw(img, μ, σ)    # float32 CHW
    write(out)
write stats.json, checksums
```

Implementation notes:
- **Bilinear resize** uses half-pixel centers (matching OpenCV `INTER_LINEAR` and Pillow), so
  the reference comparison is exact up to rounding.
- **Statistics** are accumulated in `double`. Partial results are combined with **Chan's
  parallel merge** of (n, mean, M2), which is numerically safer than `E[x²] − E[x]²`. Because
  the merge is associative, the same code serves OpenMP reductions, MPI reductions and the
  Hadoop combiner.
- We avoid recomputing the resize in pass 2: it is cached as uint8 (≈ 1.8 GB for the full
  set) when memory allows, with a flag to recompute instead. The trade-off is discussed
  in the report.

**Output:** `normalized_k.bin` (float32 CHW, 224×224×3 = 588 KB per image, ≈ 6.9 GB total;
optional `--fp16` halves this), `stats.json`, `checksums.txt`.

### 5.3 Correctness check

| Check | Tolerance |
|---|---|
| μ, σ per channel vs NumPy reference | abs. error < 1e-6 |
| Resized uint8 vs Pillow/OpenCV bilinear on 100 sampled images | max |Δ| ≤ 1 |
| Normalized tensors vs reference on samples | max abs. error < 1e-5 |
| Every later version (OMP / threads / MPI / Hadoop) vs the sequential output | stats identical within 1e-9 relative; per-image checksums equal |

### 5.4 OpenMP (`src/omp/`)

- `#pragma omp parallel for schedule(runtime)` over records in both passes.
- **Pass 1:** each thread keeps a local `Stats` and merges once at the end through a
  user-defined reduction, `#pragma omp declare reduction(merge : Stats : ...)`. There is no
  `critical` section in the hot loop.
- **Pass 2:** threads write to disjoint, precomputed output offsets, so no synchronization
  is needed.
- **Scheduling study:** `static`, `static,1`, `dynamic,{1,8,32}`, `guided`. We expect
  `dynamic`/`guided` to win because crop sizes vary a lot. We report the distribution of
  crop areas to explain the result.
- **Pinning:** `OMP_PROC_BIND=close`, `OMP_PLACES=cores`.

### 5.5 C++ thread pool (`src/threads/`), the fourth required tool

A producer–consumer pipeline, which is how a real training data loader is built:

```
[reader thread] → bounded queue → [N worker threads: crop+resize+normalize] → bounded queue → [writer thread]
```

- `std::thread`, `std::mutex` + `std::condition_variable` queues, and a `std::atomic<size_t>`
  progress counter.
- Compare against OpenMP: overlapping I/O with compute vs a pure `parallel for`.
- **This is the component we reuse directly in the ML project.**

### 5.6 Performance measurement

- Threads p ∈ {1, 2, 4, 8}. Input sizes: 100 MB, 500 MB and full. **3+ runs, report the median.**
- Time each phase separately: load, pass 1, pass 2, write.
- Report S = T_S/T_P, E = S/p, and the serial fraction f = (1/S − 1/p)/(1 − 1/p).
- **Amdahl:** expected serial parts are index load, the final stats merge, and the single-threaded writer.
- **Gustafson:** scale the dataset with p (augmentation factor) and show near-constant time.

### 5.7 Memory hierarchy experiment

- **Layout experiment:** HWC → CHW transpose during normalization, done (a) naively with a
  stride-3 read and stride-224² write, and (b) **blocked** in 32×32 tiles. Measure L1/LLC misses.
- **Resize locality:** iterate the resize over rows vs columns of the source.
- `perf stat -e cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses`.
- **False sharing:** a deliberately bad version where per-thread `Stats` sit in one
  contiguous array, compared with the version padded with `alignas(64)`.

### 5.8 Phase 1 deliverables checklist

- [ ] `/src`, `/preprocessing`, `/data`, `/plots`, `report.pdf`
- [ ] Report sections 1–6 as listed in the brief (5–7 pages)
- [ ] 2-minute demo video: dataset → preprocessing → run → speedup plot

---

## 6. Phase 2: Scaling Across Nodes (deadline: Week 10)

### 6.1 Decomposition

- **Rank 0** reads the index and computes a **balanced partition by total pixel count**,
  not by record count, because image sizes vary. It sends each rank its `(start, end)`
  range with `MPI_Send` (**point-to-point**, as required).
- Each rank reads its own byte range of the shards directly (parallel I/O, no pixel
  shipping through rank 0).
- **Pass 1:** local `Stats` → `MPI_Allreduce` with a custom `MPI_Op` (Chan merge) on an
  `MPI_Type_contiguous` struct (**collective**). Every rank gets μ, σ, which is the global
  barrier between pass 1 and pass 2.
- **Pass 2:** each rank normalizes and writes its own output range. `MPI_Gather` collects
  per-rank checksums and timings to rank 0.
- Optional comparison: a naive version where rank 0 reads everything and `MPI_Scatterv`s
  the pixels. This shows why moving data costs more than moving work.

### 6.2 Latency vs bandwidth

- Ping-pong from 1 B to 64 MB. Fit `T(m) = α + m/β` and mark the latency-dominated and
  bandwidth-dominated regions.
- Relate it to the app: the stats message is ~100 bytes (latency-bound), a 224×224×3
  float image is 588 KB, and a batch of 32 is ~19 MB (bandwidth-bound).

### 6.3 Communication/computation overlap

Variant in which workers **send normalized batches to a writer rank**: post `MPI_Isend`
for batch k, compute batch k+1, then `MPI_Wait`. Measure hidden time as
`T_blocking − T_overlap` relative to total communication time.

### 6.4 Scaling study

- **Strong scaling:** full dataset, p ∈ {1, 2, 4, 8} (containers or multiple machines).
- **Weak scaling:** augmentation factor grows with p.
- Plot against the OpenMP numbers from Phase 1 on the same axes.

### 6.5 Bonuses (optional)

- **Hybrid MPI + OpenMP:** ranks across nodes, threads within a node.
- **CUDA:** a resize + normalize + transpose kernel (one thread per output pixel; shared
  memory tiles for the transpose). Compare against CPU, including the PCIe transfer cost.
  Most relevant to the ML project.
- **RPC:** a small TCP service that returns `stats.json` or the tensor for an `image_id`,
  with a client timeout.

### 6.6 Phase 2 deliverables checklist

- [ ] MPI code, `mpirun` scripts, hostfile
- [ ] Report (4–6 pages) with topology/data-flow diagram, strong/weak scaling, communication analysis
- [ ] 2-minute video

---

## 7. Phase 3: Hadoop Streaming (deadline: Week 14)

### 7.1 Text format (`preprocessing/to_text.py`)

One line per image:

```
image_id \t class_id \t is_train \t w \t h \t bx,by,bw,bh \t base64(raw HWC uint8)
```

Lines are ≈ 1 MB each (base64 adds ~33%). Hadoop handles long lines fine, and we
report the size expansion versus the binary format.

### 7.2 Job 1: Channel statistics (the core job)

- **Mapper:** decode base64 → crop → resize → per-channel `(n, mean, M2)` for training
  images. Emits `channel \t n,mean,M2` (3 lines per image), and optionally
  `class_id:channel \t ...` for per-class analysis.
- **Combiner:** Chan merge, which is valid because it is associative and commutative.
- **Reducer:** Chan merge → final μ, σ per key.
- **Local test:** `cat sample.txt | ./mapper | sort | ./reducer`.

### 7.3 Job 2: Normalization (counts as the "second MapReduce job" bonus)

- Map-only job (`-numReduceTasks 0`). μ, σ are shipped with `-files stats.json`.
- Output is `image_id \t base64(float32 CHW)`. We spot-check it against the binary output.

### 7.4 Required experiments

| Experiment | Plan |
|---|---|
| **Correctness** | Job 1 μ, σ vs the sequential / OMP / MPI `stats.json` |
| **HDFS inspection** | `hdfs fsck /cub -files -blocks -locations`, then change replication 1 → 3 → 2 and observe |
| **Comparison table** | Serial / OpenMP / MPI / Hadoop × {100 MB, 500 MB, full}, explaining Hadoop's startup and shuffle overhead on small inputs |
| **Combiner** | With vs without: map output records, combine input/output records, shuffle bytes. Expected drop: 3 × N_images records → 3 × N_mappers |
| **Failure** | Start the full job, `docker stop` one datanode/nodemanager, record failed/retried attempts, check the result is still correct, run `hdfs fsck` before/after re-replication |
| **Consistency** | HDFS write-once/read-many, single-writer leases, what stale replicas mean, cost of stronger guarantees (Lab 8) |

### 7.5 Phase 3 deliverables checklist

- [ ] Mapper/combiner/reducer, job scripts, README with reproduction steps
- [ ] Final report (6–8 pages) covering all phases, the cross-model table, combiner and failure experiments, ≥ 2 MapReduce limitations
- [ ] Final presentation + demo

---

## 8. Tools coverage

| Required tool | Where |
|---|---|
| OpenMP | Phase 1 (§5.4) |
| C++ threads/atomics | Phase 1 (§5.5) |
| MPI | Phase 2 (§6) |
| Hadoop Streaming | Phase 3 (§7) |
| CUDA (optional) | Phase 2 bonus (§6.5) |

---

## 9. What carries over to the ML project

| Pipeline output | ML use |
|---|---|
| Predecoded shards + index | No JPEG decoding during training, so the data loader is no longer the bottleneck |
| Bbox-cropped 224×224 tensors | Ready for fine-grained classification backbones |
| `stats.json` (CUB μ, σ) | Input normalization for training from scratch; compare with ImageNet stats when fine-tuning |
| Thread-pool loader (§5.5) | Template for a C++/pybind11 data loader |
| CUDA kernel (§6.5) | On-GPU preprocessing |

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Dataset rejected as too small | Decoded-size framing + augmentation (§3.1); NABirds fallback |
| Speedup limited by disk I/O | Report compute-only and end-to-end times separately; use page cache warm runs and note it |
| Too few physical cores for p = 8 | State core count; flag hyperthreading in results |
| Hadoop on one laptop is slow | Use the 100 MB subset for development and the full set only for final runs |
| Numerical mismatch across versions | Shared `Stats` merge code in `src/common/`; fixed record order in reductions where bitwise equality matters |

---

## 11. Next steps

1. Get instructor approval for the dataset framing (§3.1).
2. Assign roles: Parallelization Lead, Performance Analyst, Code Integrator / Tester,
   Documentation Coordinator.
3. Write `download.sh`, `validate.py` and `predecode.py`, and measure the real decoded size.
4. Write `reference.py` and freeze the reference `stats.json`.
5. Write the sequential baseline and verify it against the reference.
