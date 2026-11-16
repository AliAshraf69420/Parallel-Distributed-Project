# SW401 Project Plan: CUB-200-2011 Predecoding & Normalization Pipeline

> One project with two uses: the SW401 parallel and distributed systems project, and a
> proof-of-concept data pipeline for a separate ML project trained on CUB-200-2011.

---

## 1. Summary

We build a pipeline that turns the CUB-200-2011 bird images into **fixed-size,
per-channel-normalized tensors** that a model can train on directly. The **same core
computation** runs in every phase, implemented five ways, and every version is checked
against one frozen reference output:

| Version                        | Phase           | Model                    |
| ------------------------------ | --------------- | ------------------------ |
| Sequential C++                 | 1               | Baseline                 |
| OpenMP                         | 1               | Shared memory            |
| C++ thread pool                | 1 (reused in 2) | Shared memory, pipelined |
| MPI (+ optional hybrid / CUDA) | 2               | Distributed memory       |
| Hadoop Streaming (C++)         | 3               | MapReduce                |

**Core computation (identical in every phase):**

1. Crop each decoded image to its ground-truth bounding box.
2. Resize the crop to **224×224** with bilinear interpolation (our own C++ kernel).
3. **Pass 1:** compute the per-channel (R, G, B) mean and standard deviation over the
   **training split** (a global reduction).
4. **Pass 2:** normalize every image with those statistics,
   `x' = (x/255 − μ_c) / σ_c`, and write CHW float tensors.

JPEG decoding happens **once**, during preprocessing ("predecoding"). Every phase starts
from raw pixels, so the measured speedups come from our own code, not from libjpeg.

### 1.1 How the two passes map onto each model

| Step                   | Sequential | OpenMP / threads                | MPI                            | Hadoop                       |
| ---------------------- | ---------- | ------------------------------- | ------------------------------ | ---------------------------- |
| Partition              | —          | loop iterations / work queue    | pixel-balanced rank ranges     | HDFS blocks → map tasks      |
| Pass 1 local work      | loop       | per-thread `Stats`              | per-rank `Stats`               | mapper emits `Stats`         |
| Pass 1 merge           | —          | `declare reduction`             | `MPI_Allreduce` (custom op)    | combiner + reducer           |
| Barrier between passes | implicit   | end of parallel region          | the Allreduce itself           | job boundary (Job 1 → Job 2) |
| Pass 2                 | loop       | `parallel for`, disjoint writes | each rank writes its own range | map-only job                 |

---

## 2. Why an image dataset needs per-channel mean/std

The mean/std here is **not** a statistic of the target class. It is a property of the
_input distribution_. It is used to standardize every pixel before it enters the network.

### 2.1 What normalization does for training

| Reason                             | Explanation                                                                                                                                                                                                                                                                                                             |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Zero-centered inputs**           | Raw pixels are all positive (0–255). If every input is positive, the gradients for a neuron's incoming weights all have the same sign, so the weights can only move in zig-zag steps. Subtracting the mean removes this (LeCun et al., _Efficient BackProp_).                                                           |
| **Equal scale across channels**    | Channels have different spreads. Bird photos are dominated by green foliage and brown bark, for example. Dividing by σ gives each channel roughly unit variance, which improves the conditioning of the loss surface so one learning rate works for all first-layer weights.                                            |
| **Stable activations**             | Weight initialization schemes (Xavier/He) assume inputs with roughly unit variance. Without normalization, early activations can saturate or explode.                                                                                                                                                                   |
| **Matching a pretrained backbone** | A network pretrained on ImageNet expects ImageNet's statistics (mean ≈ [0.485, 0.456, 0.406], std ≈ [0.229, 0.224, 0.225]). For fine-tuning you would usually reuse those. For training from scratch, or for checking how far CUB is from ImageNet, you need CUB's own values, and that is what this pipeline computes. |

### 2.2 Why per _channel_ and not per pixel or per image

- **Per pixel** (a "mean image" of 224×224×3 values) was used in early CNNs. Convolutions
  are translation-invariant, though, so position-specific statistics add little. Per-pixel
  stats also tie the result to one input resolution.
- **Per image** (normalize each image by its own mean/std) removes real information such
  as overall brightness and color cast, which can be useful cues for species.
- **Per channel over the whole dataset** keeps differences between images and still fixes
  the scale. It is only 6 numbers, so it is cheap to store and apply.

### 2.3 Why it cannot be per class

At inference time the class is **unknown**, since predicting it is the whole point. So the
normalization must be the same for every input. If we normalized with per-class statistics,
the label would leak into the input during training, and the transform could not be
applied at test time at all.

(Per-class statistics are still useful for **analysis**. They appear in this plan as an
optional output in every phase and as a second key in the Hadoop job, see §8.2.)

### 2.4 Rules we follow (all phases)

- **Train split only.** Statistics are computed on the 5,994 training images and then
  applied to both train and test. Including test images would leak test information.
- **Same representation the model sees.** Statistics are computed _after_ crop and resize.
- **Pixel scale.** Statistics are reported on the [0, 1] scale, as in ImageNet's numbers.

---

## 3. Dataset

| Item                 | Value                                                                                               |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| Source               | CUB-200-2011 (Caltech-UCSD Birds), official tarball `CUB_200_2011.tgz`                              |
| Approval             | approved by the instructor                                                                          |
| Images               | 11,788 JPEGs, 200 species                                                                           |
| Split                | 5,994 train / 5,794 test (`train_test_split.txt`)                                                   |
| Annotations used     | `images.txt`, `image_class_labels.txt`, `bounding_boxes.txt`, `train_test_split.txt`, `classes.txt` |
| Compressed size      | ≈ 1.1 GB                                                                                            |
| Decoded raw RGB size | ≈ 6–7 GB (estimate, measured in preprocessing)                                                      |

### 3.1 Dataset scale per phase

| Need                                     | Dataset used                                                                                                                                            |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phase 1 thread scaling                   | 100 MB, 500 MB, full (decoded)                                                                                                                          |
| Phase 1 Gustafson / Phase 2 weak scaling | **augmented ×k** sets, k ∈ {1, 2, 4, 8}: horizontal flip plus fixed jittered crops with seeded RNG, so they are reproducible and can be checked bitwise |
| Phase 2 strong scaling                   | full                                                                                                                                                    |
| Phase 3 comparison table                 | 100 MB, 500 MB, full (text format)                                                                                                                      |
| Phase 3 failure experiment               | augmented ×4 text (a job long enough to kill a node mid-run)                                                                                            |

---

## 4. Data artifacts (produced once in preprocessing, used by all phases)

All artifacts come from `preprocessing/` in one reproducible run (`make data`). No phase
creates its own copy of the data.

| Artifact                             | Format                                                                                                          | Used in                                          |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `index.bin`                          | fixed-size records: `u64 offset; u32 image_id; u16 class_id; u8 is_train; u8 pad; u16 w, h; u16 bx, by, bw, bh` | 1, 2                                             |
| `shard_k.bin`                        | raw HWC uint8 pixels at **original size**, concatenated                                                         | 1, 2                                             |
| `subsets/{100MB,500MB}/`             | same format, class-stratified, seeded                                                                           | 1, 2, 3                                          |
| `aug_x{2,4,8}/`                      | same format, deterministic augmentation                                                                         | 1 (Gustafson), 2 (weak scaling), 3 (failure run) |
| `text/{100MB,500MB,full,aug_x4}.txt` | one line per image (see §8.1)                                                                                   | 3                                                |
| `results/reference/<set>/stats.json` | exact integer sums (n, S1, S2), μ, σ, the float32 LUT, per-class train stats; spec in `preprocessing/reference.py` | all (correctness) |
| `results/reference/<set>/checksums.csv` | per-image CRC-32 of the resized uint8 and of the normalized float32 output | all (correctness) |

We keep the original size (not pre-resized) on purpose:

- Image sizes vary, so the work per record is **uneven**. That makes the OpenMP scheduling
  comparison (Phase 1) and pixel-balanced partitioning (Phase 2) meaningful.
- The index lets any worker jump straight to any record, which is needed for MPI
  partitioning.
- Hadoop runs exactly the same crop + resize in its mapper.

---

## 5. Shared foundations (all phases)

### 5.1 Repository layout

```
Parallel-Distributed-Project/
├── README.md                 # reproduce everything, per phase
├── Makefile                  # make data | seq | omp | threads | mpi | hadoop | bench | plots
├── docs/
│   ├── PLAN.md
│   └── SW401–CourseProjectDescription.pdf
├── preprocessing/            # Python (allowed for preprocessing only)
│   ├── download.sh
│   ├── validate.py           # integrity, mode, duplicates, bbox sanitize
│   ├── predecode.py          # JPEG → shards + index
│   ├── make_subsets.py       # 100 MB / 500 MB
│   ├── augment.py            # deterministic ×k sets
│   ├── to_text.py            # binary → Hadoop line format
│   └── reference.py          # NumPy reference stats + checksums
├── src/
│   ├── common/               # shared by every version (see §5.2)
│   ├── seq/                  # Phase 1
│   ├── omp/                  # Phase 1
│   ├── threads/              # Phase 1
│   ├── mpi/                  # Phase 2 (+ hybrid)
│   ├── cuda/                 # Phase 2 bonus
│   ├── rpc/                  # Phase 2/3 bonus
│   └── hadoop/               # Phase 3: mapper, combiner, reducer, normalize_mapper
├── tests/                    # unit tests for src/common + correctness checker
├── scripts/
│   ├── bench/                # one runner per version, same CSV schema
│   ├── mpi/                  # mpirun commands, hostfile
│   ├── hadoop/               # job submission, fsck, failure experiment
│   └── docker/               # MPI cluster + Hadoop cluster compose files
├── data/                     # small samples + download instructions only
├── results/                  # raw CSVs (all phases)
├── plots/                    # generated from results/
└── report/                   # phase1/, phase2/, final/
```

### 5.2 Shared code (`src/common/`)

Writing this once keeps all versions computing exactly the same thing:

| Module      | Contents                                                                                                |
| ----------- | ------------------------------------------------------------------------------------------------------- |
| `record_io` | read the index, mmap shards, read a byte range (MPI), parse a text line + base64 (Hadoop)               |
| `resize`    | bbox crop + **fixed-point** (11-bit) bilinear resize with half-pixel centers, exactly as specified in `reference.py` |
| `stats`     | `Stats {u64 n, u64 S1[3], u64 S2[3]}`, `accumulate(img)`, `merge(a, b)` (integer addition), `finalize()` → μ, σ, LUT |
| `normalize` | HWC uint8 → CHW float32 through a 256-entry LUT per channel (naive and blocked variants)                |
| `timer`     | per-stage wall-clock timing, CSV writer                                                                 |
| `checksum`  | CRC-32 (zlib polynomial) of the resized and of the normalized tensor                                    |

**Everything is exact, so every version must match bit for bit.** Resize is integer
arithmetic; pass 1 keeps exact integer sums (n, Σv, Σv²), so merging is plain addition in
any order; μ, σ and the LUT come from a few correctly rounded double operations. The same
`Stats::merge` serves as the OpenMP reduction, the MPI reduction and the Hadoop
combiner/reducer. C++ is compiled with `-ffp-contract=off`, so FMA contraction cannot change
the double results.

### 5.3 Correctness (every version, every phase)

One checker, `tests/check.py`, compares any version's output against `results/reference/<set>/`:

| Check | Required |
| --- | --- |
| n, S1, S2 per channel | identical integers |
| μ, σ and the LUT | identical bits |
| per-image `crc_resized` (pinpoints resize bugs) | identical |
| per-image `crc_normalized` | identical |

`reference.py` also checks the fixed-point resize against float64 bilinear on 64 sampled images. It must stay within 1 gray level; measured: 0.56.

**Cross-version correctness matrix**, filled in as each phase lands and included in the final report:

| Version               | 100 MB | 500 MB | full | aug ×4 |
| --------------------- | ------ | ------ | ---- | ------ |
| Sequential            |        |        |      |        |
| OpenMP                |        |        |      |        |
| Threads               |        |        |      |        |
| MPI                   |        |        |      |        |
| Hybrid / CUDA (bonus) |        |        |      |        |
| Hadoop                |        |        |      |        |

### 5.4 Benchmark methodology (every phase)

- **Same rules everywhere:** ≥ 3 runs, report the **median**. Record the machine
  (CPU model, core count, RAM, disk), compiler and flags (`-O3 -march=native`), and library
  versions.
- **Per-stage timing:** load, pass 1, merge/communication, pass 2, write. Report
  compute-only and end-to-end times separately, so I/O effects are visible.
- **Cache state:** warm page cache by default. One cold-cache run per configuration is
  noted in the report.
- **One CSV schema** for all phases, so the plotting scripts and the final comparison
  table work unchanged:

```
phase,version,dataset,workers,threads_per_worker,schedule,run,t_load,t_pass1,t_merge,t_pass2,t_write,t_total,host
```

- `scripts/bench/run_all.sh <version>` produces the CSV, and `plots/make_plots.py`
  produces every figure.

### 5.5 Cross-model comparison table (built up across phases)

Started in Phase 1 and extended each phase. It becomes the table required in Phase 3.

| Dataset | Serial | OpenMP (8t) | Threads (8t) | MPI (8p) | Hybrid (bonus) | Hadoop |
| ------- | ------ | ----------- | ------------ | -------- | -------------- | ------ |
| 100 MB  | P1     | P1          | P1           | P2       | P2             | P3     |
| 500 MB  | P1     | P1          | P1           | P2       | P2             | P3     |
| full    | P1     | P1          | P1           | P2       | P2             | P3     |

### 5.6 Environments

| Phase | Environment                                                                         |
| ----- | ----------------------------------------------------------------------------------- |
| 1     | GCC with OpenMP, `perf`, Python 3 + NumPy + Pillow                                  |
| 2     | OpenMPI. Multi-node through Docker containers or lab machines, plus a hostfile      |
| 3     | Hadoop cluster in Docker (1 NameNode + ≥ 3 DataNodes/NodeManagers) or the course VM |

Set up the next phase's environment **while the current phase's benchmarks are running**,
so setup problems don't block coding.

---

## 6. Phase 1: Parallel Foundations (due 2026-11-17)

### 6.1 Preprocessing

| Step                      | What                                                                                       | Reported statistic             |
| ------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------ |
| 1. Integrity              | Open every JPEG and fully decode it. Drop files that fail.                                 | # corrupt                      |
| 2. Color mode             | Convert non-RGB images (grayscale, etc.) to RGB and log the IDs.                           | # converted                    |
| 3. Duplicates             | Hash the decoded pixels (SHA-1). Drop exact duplicates and keep the first ID.              | # duplicates                   |
| 4. Annotation join        | Join path, class, split and bbox by `image_id`. Drop images with missing metadata.         | # missing                      |
| 5. BBox sanitize          | Clamp boxes to the image bounds, round to integers, drop degenerate boxes (w or h < 8 px). | # clamped / dropped            |
| 6. Predecode              | Write the shards + index (§4).                                                             | bytes and records before/after |
| 7. Subsets & augmentation | 100 MB, 500 MB, aug ×{2,4,8}.                                                              | records and bytes per set      |
| 8. Text export            | Hadoop line format, so Phase 3 isn't waiting on it.                                        | bytes, expansion factor        |
| 9. Reference              | NumPy stats + checksums.                                                                   | —                              |

### 6.2 Sequential baseline (`src/seq/`)

```
load index
for each train record:                   # pass 1
    img = resize(bbox_crop(raw), 224, 224)
    stats.accumulate(img)
μ, σ = stats.finalize()
for each record:                         # pass 2
    img = resize(bbox_crop(raw))         # or read from uint8 cache
    write(normalize_to_chw(img, μ, σ))
write stats.json, checksums
```

- The pass-1 resize can be cached as uint8 (≈ 1.8 GB for the full set) to avoid redoing
  it in pass 2. A flag turns the cache on or off, and the memory vs compute trade-off is
  discussed in the report.
- Output is float32 CHW (588 KB per image, ≈ 6.9 GB full). `--fp16` halves it.

### 6.3 OpenMP (`src/omp/`)

- `#pragma omp parallel for schedule(runtime)` over records in both passes.
- **Pass 1:** thread-local `Stats`, merged with
  `#pragma omp declare reduction(merge : Stats : ...)` (or array `reduction(+ : S1[:3], S2[:3])`).
  No `critical` section in the hot loop.
- **Pass 2:** disjoint, precomputed output offsets, so no synchronization is needed.
- **Scheduling study:** `static`, `static,1`, `dynamic,{1,8,32}`, `guided`. We expect
  `dynamic`/`guided` to win because crop sizes vary. The distribution of crop areas is
  plotted to explain the result.
- **Pinning:** `OMP_PROC_BIND=close`, `OMP_PLACES=cores`.
- **Tasks:** an `omp task` variant that splits very large crops into row bands, for
  comparison.

### 6.4 C++ thread pool (`src/threads/`)

```
[reader] → bounded queue → [N workers: crop+resize+normalize] → bounded queue → [writer]
```

- `std::thread`, `std::mutex` + `std::condition_variable` queues, `std::atomic` counters.
- Compared against OpenMP: overlapping I/O with compute vs a pure `parallel for`.
- **Reused in Phase 2:** the same pipeline runs inside each MPI rank (hybrid bonus).
- **Reused in the ML project** as the data-loader template.

### 6.5 Performance and memory hierarchy

- Threads p ∈ {1, 2, 4, 8} × {100 MB, 500 MB, full}. Report S, E and the serial fraction
  f = (1/S − 1/p)/(1 − 1/p).
- **Amdahl:** expected serial parts are index load, final merge and the writer.
  **Gustafson:** aug ×p sets.
- **Layout:** naive vs **blocked (32×32)** HWC → CHW transpose. Row-major vs column-major
  resize traversal.
- `perf stat -e cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses`.
- **False sharing:** packed vs `alignas(64)` per-thread `Stats` array.

### 6.6 Deliverables

- [ ] Repo with `/src`, `/preprocessing`, `/data`, `/plots`, `report.pdf`
- [ ] Report, 5–7 pages: dataset + before/after stats, problem + baseline, OpenMP design, performance table + plots, Amdahl/Gustafson + profiling, remaining bottlenecks
- [ ] 2-minute video: dataset → preprocessing → run → speedup plot
- [ ] Correctness matrix and comparison table: Serial, OpenMP, Threads columns filled in

---

## 7. Phase 2: Scaling Across Nodes (target: 2026-11-22)

### 7.1 Decomposition

- **Rank 0** reads the index and computes a partition **balanced by total pixel count**
  (not record count, because sizes vary). It sends each rank its `(start, end)` with
  `MPI_Send` (**point-to-point**).
- Each rank reads its own byte range of the shards directly (parallel I/O).
- **Pass 1:** local `Stats` (7 × u64) → `MPI_Allreduce(MPI_UINT64_T, MPI_SUM)`
  (**collective**). Because the sums are exact integers, the result does not depend on
  reduction order. This also acts as the barrier between pass 1 and pass 2.
- **Pass 2:** each rank writes its own output range. `MPI_Gather` collects checksums and
  timings to rank 0.
- **Comparison:** a naive version where rank 0 reads everything and `MPI_Scatterv`s the
  pixels. This shows that moving data costs more than moving work.
- **Load balance:** report per-rank compute time, plus a version balanced by record count
  for comparison.

### 7.2 Latency vs bandwidth

- Ping-pong from 1 B to 64 MB, both intra-node and inter-node. Fit `T(m) = α + m/β`.
- Relate it to the app: the `Stats` message is ~100 B (latency-bound), one float image
  is 588 KB, and a batch of 32 is ~19 MB (bandwidth-bound).

### 7.3 Communication/computation overlap

Variant in which workers send normalized batches to a writer rank: `MPI_Isend` batch k,
compute batch k+1, then `MPI_Wait`. Hidden time = `T_blocking − T_overlap`, as a share of
total communication time.

### 7.4 Scaling study

- **Strong scaling:** full dataset, p ∈ {1, 2, 4, 8}.
- **Weak scaling:** aug ×p with p processes.
- Plotted on the same axes as the OpenMP results from Phase 1, using the shared CSV schema.

### 7.5 Bonuses

- **Hybrid:** MPI ranks across nodes, with the OpenMP or thread-pool version inside each rank.
- **CUDA:** a resize + normalize + transpose kernel (one thread per output pixel,
  shared-memory tiles for the transpose). Compared against CPU, including the PCIe
  transfer cost.
- **RPC:** a TCP service returning `stats.json` or the tensor for an `image_id`, with a
  client timeout, plus a discussion of why a slow server looks the same as a crashed one.
  Reused in Phase 3.

### 7.6 Deliverables

- [ ] MPI code, `mpirun` scripts, hostfile
- [ ] Report, 4–6 pages: topology/data-flow diagram, strong/weak scaling, communication analysis
- [ ] 2-minute video
- [ ] Correctness matrix and comparison table: MPI (and bonus) columns filled in

---

## 8. Phase 3: Resilience & Big-Data Integration (target: 2026-11-22)

### 8.1 Text format

```
image_id \t class_id \t is_train \t w \t h \t bx,by,bw,bh \t base64(raw HWC uint8)
```

Lines are ≈ 1 MB each (base64 adds ~33%). The size expansion vs binary is reported.
This file is produced in Phase 1 preprocessing (§6.1, step 8).

### 8.2 Job 1: Channel statistics (the core job)

- **Mapper:** parse → crop → resize (via `src/common`) → for training images emit
  `channel \t n,S1,S2` (3 lines per image, exact integers). Optionally also
  `class_id:channel \t ...` for the per-class analysis.
- **Combiner / reducer:** `Stats::merge` (integer addition), so the combiner is trivially correct.
- **Local test:** `cat sample.txt | ./mapper | sort | ./reducer`. The output must match
  `results/reference/<set>/stats.json` exactly.

### 8.3 Job 2: Normalization (counts as the "second MapReduce job" bonus)

- Map-only (`-numReduceTasks 0`). μ, σ are shipped with `-files stats.json`.
- Output `image_id \t base64(float32 CHW)`. Checksums are compared against
  `results/reference/<set>/checksums.csv`.

### 8.4 Required experiments

| Experiment           | Plan                                                                                                                                                                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Correctness**      | Job 1 output vs serial / OMP / threads / MPI (fills the Hadoop row of the matrix in §5.3)                                                                                                                                                  |
| **HDFS inspection**  | `hdfs fsck /cub -files -blocks -locations`, NameNode UI; change replication 1 → 3 → 2 and observe                                                                                                                                          |
| **Comparison table** | Fill the Hadoop column of §5.5. Explain JVM/container startup and shuffle overhead on small inputs, and estimate the size or cluster size where Hadoop would win                                                                           |
| **Combiner**         | With vs without: map output records, combine in/out records, shuffle bytes. Expected: 3 × N_images → about 3 × N_map_tasks records                                                                                                         |
| **Failure**          | Start the aug ×4 job, `docker stop` a DataNode/NodeManager, record failed/retried attempts, check the result is still correct, run `hdfs fsck` before/after re-replication, and compare with the partial-failure discussion from Lecture 1 |
| **Consistency**      | HDFS write-once/read-many, single-writer leases, stale replicas, the cost of stronger guarantees (Lab 8)                                                                                                                                   |
| **Bonus**            | RPC service from §7.5 serving Hadoop results                                                                                                                                                                                               |

### 8.5 Deliverables

- [ ] Mapper/combiner/reducer, job scripts, README reproducing all phases
- [ ] Final report, 6–8 pages: all phases, the full comparison table and correctness matrix, combiner + failure experiments, ≥ 2 MapReduce limitations, lessons learned
- [ ] Final presentation + live/recorded demo

---

## 9. Roles across phases

The roles are fixed. Each person's focus shifts from phase to phase.

| Role                          | Phase 1                                                      | Phase 2                                       | Phase 3                                                                      |
| ----------------------------- | ------------------------------------------------------------ | --------------------------------------------- | ---------------------------------------------------------------------------- |
| **Parallelization Lead**      | OpenMP + thread pool                                         | MPI decomposition, overlap, hybrid            | Mapper/combiner/reducer, Job 2                                               |
| **Performance Analyst**       | Benchmark harness, speedup, perf/cache, Amdahl/Gustafson     | Ping-pong, strong/weak scaling, comm analysis | Comparison table, combiner counters, Hadoop overhead analysis                |
| **Code Integrator / Tester**  | Preprocessing, `src/common`, `check.py`, sequential baseline | MPI environment, hostfile, correctness matrix | Hadoop cluster, HDFS inspection, failure experiment, clean-clone README test |
| **Documentation Coordinator** | Phase 1 report + video                                       | Data-flow diagram, Phase 2 report + video     | Consistency write-up, final report, presentation                             |

---

## 10. Tools coverage

| Required tool       | Where                                 |
| ------------------- | ------------------------------------- |
| OpenMP              | Phase 1 (§6.3), Phase 2 hybrid (§7.5) |
| C++ threads/atomics | Phase 1 (§6.4), Phase 2 hybrid (§7.5) |
| MPI                 | Phase 2 (§7)                          |
| Hadoop Streaming    | Phase 3 (§8)                          |
| CUDA (optional)     | Phase 2 bonus (§7.5)                  |

---

## 11. What carries over to the ML project

| Pipeline output              | ML use                                                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------------- |
| Predecoded shards + index    | No JPEG decoding during training, so the data loader is no longer the bottleneck            |
| Bbox-cropped 224×224 tensors | Ready for fine-grained classification backbones                                             |
| `stats.json` (CUB μ, σ)      | Input normalization for training from scratch; compare with ImageNet stats when fine-tuning |
| Thread-pool loader (§6.4)    | Template for a C++/pybind11 data loader                                                     |
| MPI partitioning (§7.1)      | Pattern for sharding data in distributed training                                           |
| CUDA kernel (§7.5)           | On-GPU preprocessing                                                                        |

---

## 12. Risks

| Risk                                   | Phase | Mitigation                                                                                                                                                                              |
| -------------------------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Speedup limited by disk I/O            | 1, 2  | Report compute-only and end-to-end separately; warm-cache runs                                                                                                                          |
| Too few physical cores for p = 8       | 1, 2  | State core count; flag hyperthreaded results                                                                                                                                            |
| No real multi-node cluster             | 2     | Docker containers with a hostfile; state clearly that intra-host network is not a real interconnect; use lab machines if available                                                      |
| MPI partition imbalance                | 2     | Pixel-balanced partition; report per-rank times                                                                                                                                         |
| Hadoop cluster setup eats Phase 3 time | 3     | Start the Docker cluster while the MPI benchmarks run (§5.6)                                                                                                                            |
| One week for all three phases          | all   | Required items first, bonuses only at the end (§13); every phase reuses `src/common`, the data artifacts and the bench harness, so Phases 2–3 are mostly new drivers, not new pipelines |
| Hadoop on one machine is slow          | 3     | Develop on 100 MB; full and aug runs only for final numbers                                                                                                                             |
| Numerical mismatch across versions     | all   | Exact integer spec (§5.2) + `-ffp-contract=off`; per-image `crc_resized` shows which stage diverged |
| Report work piling up at deadlines     | all   | Shared CSV schema + plot scripts; fill in the comparison table and correctness matrix as each phase lands                                                                               |

---

## 13. Order of work

Phase 1 is due **2026-11-17**. Phases 2 and 3 are targeted for the same week
(by **2026-11-22**).

### Phase 1: critical path (due tomorrow)

1. `download.sh`, `validate.py`, `predecode.py` on the full set. Record the before/after statistics.
2. `make_subsets.py` (100 MB, 500 MB) and `reference.py`.
3. `src/common` + the sequential baseline, verified with `check.py`.
4. OpenMP + scheduling study.
5. Bench harness: {1, 2, 4, 8} threads × 3 datasets × 3 runs. Speedup/efficiency plots.
6. `perf stat`, naive vs blocked transpose, false-sharing test.
7. Report (5–7 pages) + 2-minute video.

**Can move to the Phase 2 submission if time runs out:** the thread pool (§6.4), since
the four-tool requirement covers the whole project, not each phase; the `omp task`
variant; the aug ×8 set (Gustafson can use ×2 and ×4).

### Phase 2

1. Start the MPI Docker setup while the Phase 1 benchmarks run.
2. MPI decomposition + Allreduce, verified with `check.py`.
3. Ping-pong, overlap variant, strong/weak scaling.
4. Thread pool, if deferred from Phase 1.
5. Report (4–6 pages) + data-flow diagram + video.

### Phase 3

1. Start the Hadoop Docker cluster while the MPI benchmarks run. Run `to_text.py`.
2. Mapper/combiner/reducer, local pipe test against the reference, then HDFS upload + `fsck`.
3. Comparison table, combiner experiment, failure experiment, consistency write-up.
4. Final report (6–8 pages), presentation, README checked on a clean clone.

### Bonuses (only after everything required above is done)

Hybrid MPI + OpenMP, Hadoop Job 2 (normalization), RPC service, CUDA kernel. In that
order, cheapest first.
