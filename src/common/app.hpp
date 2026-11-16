// Shared driver code for every implementation: command line, the per-record work
// of each pass, timing, and output files (stats.json, checksums.csv, timing CSV).
// Each implementation (seq, omp, threads, mpi, ...) only adds its own loops.
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cub_format.hpp"
#include "kernels.hpp"

namespace cub {

struct Options {
    std::string version;       // "seq", "omp", ...
    std::string set_dir;       // data/processed/<set>
    std::string out_dir;       // where stats.json + checksums.csv go ("" = default)
    std::string csv;           // append one timing row here ("" = don't)
    std::string write_path;    // write normalized tensors to this file ("" = don't)
    std::string schedule;      // OpenMP schedule, e.g. "dynamic,8" ("" = OMP_SCHEDULE)
    std::string order = "id";  // record processing order: id | area-desc | area-asc
    std::string record_times;  // seq only: per-record pass-2 time CSV ("" = don't)
    std::string partition = "work";  // mpi: how rank 0 splits the records: work | count | pixels
    std::string collect = "none";    // mpi: ship tensors to rank 0: none | send | isend
    int threads = 0;           // 0 = implementation default
    int run = 0;               // run number, for the CSV
    bool cache = false;        // keep resized images between passes
    bool preload = false;      // fault every shard page in before timing
    bool quiet = false;
};

// Parses argv; prints usage and exits on --help or errors.
Options parse_args(int argc, char** argv, const std::string& version);

struct Timings {
    double load = 0, pass1 = 0, merge = 0, pass2 = 0, write = 0, total = 0;
};

class Stopwatch {
public:
    Stopwatch() : t0_(clock::now()), last_(t0_) {}
    double lap() {  // seconds since the previous lap
        const auto now = clock::now();
        const double s = std::chrono::duration<double>(now - last_).count();
        last_ = now;
        return s;
    }
    double total() const { return std::chrono::duration<double>(clock::now() - t0_).count(); }

private:
    using clock = std::chrono::steady_clock;
    clock::time_point t0_, last_;
};

// Fixed-size output file of normalized tensors: block i at i * kNormalizedBytes,
// written with pwrite, so threads/ranks can write disjoint blocks without locks.
class TensorWriter {
public:
    // create = true: create/truncate and size the file. create = false: open a file that
    // another process (rank 0) already created, to write this process's own blocks.
    TensorWriter(const std::string& path, size_t count, bool create = true);
    ~TensorWriter();
    void put(size_t i, const float* tensor) const;

private:
    int fd_ = -1;
};

// Everything one run produces, indexed like the dataset's records.
struct RunOutput {
    Stats stats;
    Normalization norm{};
    std::vector<uint32_t> crc_resized, crc_normalized;
    Timings t;
};

// ---- per-record work (identical in every implementation) --------------------
// Order in which records are processed (pass 2 visits all of them in this order).
// "id" is the storage order; "area-desc"/"area-asc" sort by crop area (the per-image
// cost), which makes block-static scheduling maximally unbalanced (scheduling study).
// Outputs are always indexed by record, so the order never changes the results.
std::vector<size_t> record_order(const Dataset& ds, const std::string& order);

// Records pass 1 visits, in processing order: train records, or all records with --cache
// (then every image is resized once in pass 1 and pass 2 reuses it).
std::vector<size_t> pass1_indices(const Dataset& ds, const std::vector<size_t>& order,
                                  bool cache);

// Pass 1 for record i: resize into `resized`, accumulate into st if it is a train record.
inline void pass1_record(const Dataset& ds, size_t i, uint8_t* resized, Stats& st) {
    const Record& r = ds[i];
    crop_resize(ds.pixels(r), r, resized);
    if (r.is_train) st.accumulate(resized);
}

// Pass 2 for record i. `resized` holds the resized image if `have_resized`, otherwise it
// is scratch space. `tensor` is scratch for the normalized output.
inline void pass2_record(const Dataset& ds, size_t i, const Normalization& nz,
                         uint8_t* resized, bool have_resized, float* tensor,
                         RunOutput& out, const TensorWriter* writer) {
    const Record& r = ds[i];
    if (!have_resized) crop_resize(ds.pixels(r), r, resized);
    out.crc_resized[i] = crc32(resized, kResizedBytes);
    normalize_chw(resized, nz, tensor);
    out.crc_normalized[i] = crc32(tensor, kNormalizedBytes);
    if (writer) writer->put(i, tensor);
}

// ---- outputs ----------------------------------------------------------------
// Writes <out_dir>/stats.json and <out_dir>/checksums.csv (same format as
// results/reference/<set>/), appends a timing row to opt.csv, prints a summary.
// `workers` = processes/ranks, `threads` = threads per worker.
// `records` are all records of the set, in storage order (out.crc_* are indexed the same way).
void finish_run(const Options& opt, const std::string& set_name,
                const std::vector<Record>& records, RunOutput& out, int workers, int threads,
                const std::string& schedule, Stopwatch& sw);
inline void finish_run(const Options& opt, const Dataset& ds, RunOutput& out, int workers,
                       int threads, const std::string& schedule, Stopwatch& sw) {
    finish_run(opt, ds.name(), ds.records(), out, workers, threads, schedule, sw);
}

}  // namespace cub
