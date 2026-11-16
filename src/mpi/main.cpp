// MPI version (distributed memory). Same per-record work as every other version
// (src/common/app.hpp); this file is the decomposition and the communication.
//
// Decomposition: 1-D block partition of the records, in storage order.
//   Rank 0 is the only rank that reads index.bin. It cuts the record list into P contiguous
//   blocks and sends each rank its block of 32-byte records with MPI_Send (point-to-point).
//   Each rank then maps only the shards its records live in and reads its own pixels, so no
//   pixel data ever passes through rank 0. Contiguous blocks keep each rank's I/O sequential.
//   --partition picks what is balanced: `work` (default; estimated cost: every record is
//   normalized, train records are also resized in pass 1), `count` (records) or `pixels`
//   (source bytes, i.e. I/O volume).
//
// Pass 1: each rank accumulates private Stats over its train records, then one
//   MPI_Allreduce(MPI_UINT64_T, MPI_SUM) over the 7 counters (collective). The sums are exact
//   integers, so the result is identical for any P and any partition. The Allreduce is also
//   the barrier between the passes: every rank leaves it holding the global mean/std.
// Pass 2: each rank normalizes its records. The per-image CRCs travel to rank 0 with
//   MPI_Gatherv, per-rank timings with MPI_Gather (collectives). With --write every rank
//   pwrites its own blocks of the shared output file.
//
// --collect send|isend (communication/computation overlap experiment): rank 0 becomes a
//   collector and ranks 1..P-1 ship every normalized tensor (588 KiB) to it.
//     send:  MPI_Send(tensor k), then compute tensor k+1.
//     isend: MPI_Isend(tensor k), compute tensor k+1 into a second buffer (calling MPI_Test
//            between kernels so the library can make progress), then MPI_Wait.
//   The collector checks every received tensor's CRC, so the experiment is verified too.
//
// Hybrid MPI + OpenMP: -t N runs N OpenMP threads inside every rank (both passes).
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/app.hpp"

#pragma omp declare reduction(merge : cub::Stats : omp_out.merge(omp_in)) \
    initializer(omp_priv = cub::Stats{})

namespace {

using namespace cub;

constexpr int kTagCount = 1, kTagRecords = 2, kTagTensor = 3;
// Relative cost of the two passes per record, from the Phase 1 sequential profile
// (pass 2: 7.59 s / 11787 records; pass 1: 2.07 s / 5993 train records).
constexpr double kCostPass2 = 0.643, kCostPass1 = 0.345;

// Cut `records` into `parts` contiguous blocks with (nearly) equal total weight.
std::vector<size_t> block_cuts(const std::vector<Record>& records, int parts,
                               const std::string& how) {
    std::vector<double> w(records.size());
    for (size_t i = 0; i < records.size(); ++i) {
        const Record& r = records[i];
        if (how == "count") w[i] = 1.0;
        else if (how == "pixels") w[i] = double(r.w) * r.h;
        else if (how == "work") w[i] = kCostPass2 + (r.is_train ? kCostPass1 : 0.0);
        else throw std::runtime_error("unknown partition '" + how + "'");
    }
    const double total = std::accumulate(w.begin(), w.end(), 0.0);
    std::vector<size_t> cuts(size_t(parts) + 1, records.size());
    cuts[0] = 0;
    double acc = 0;
    size_t i = 0;
    for (int p = 1; p < parts; ++p) {
        const double target = total * p / parts;
        while (i < records.size() && acc + w[i] / 2 < target) acc += w[i++];
        cuts[size_t(p)] = i;
    }
    return cuts;
}

struct RankTimes {  // local wall-clock seconds of one rank
    double load, pass1, pass2, comm;
};

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    try {
        Options opt = parse_args(argc, argv, "mpi");
        const int threads = opt.threads > 0 ? opt.threads : 1;
        omp_set_num_threads(threads);
        const bool collect = opt.collect != "none";
        if (collect && opt.collect != "send" && opt.collect != "isend")
            throw std::runtime_error("unknown --collect '" + opt.collect + "'");
        if (collect && P < 2) throw std::runtime_error("--collect needs at least 2 ranks");
        if (collect && threads > 1) throw std::runtime_error("--collect runs 1 thread per rank");
        if (opt.cache) throw std::runtime_error("--cache is not supported by the mpi version");
        const int first_worker = collect ? 1 : 0;   // rank 0 only collects in --collect mode
        const int workers = P - first_worker;

        MPI_Barrier(MPI_COMM_WORLD);
        Stopwatch sw;
        double t = MPI_Wtime();

        // ---- distribute the index: rank 0 -> everyone (point-to-point) -------------------
        std::vector<Record> all;        // rank 0 only
        std::vector<size_t> cuts;       // rank 0 only: block boundaries per worker
        std::vector<Record> mine;
        uint64_t my_begin = 0, total = 0;
        if (rank == 0) {
            all = Dataset::read_index(opt.set_dir);
            total = all.size();
            cuts = block_cuts(all, workers, opt.partition);
            for (int r = first_worker; r < P; ++r) {
                const size_t b = cuts[size_t(r - first_worker)], e = cuts[size_t(r - first_worker) + 1];
                if (r == 0) {
                    mine.assign(all.begin() + long(b), all.begin() + long(e));
                    my_begin = b;
                    continue;
                }
                const uint64_t hdr[3] = {b, e - b, total};
                MPI_Send(hdr, 3, MPI_UINT64_T, r, kTagCount, MPI_COMM_WORLD);
                MPI_Send(all.data() + b, int((e - b) * sizeof(Record)), MPI_BYTE, r, kTagRecords,
                         MPI_COMM_WORLD);
            }
        } else {
            uint64_t hdr[3];
            MPI_Recv(hdr, 3, MPI_UINT64_T, 0, kTagCount, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            my_begin = hdr[0];
            total = hdr[2];
            mine.resize(hdr[1]);
            MPI_Recv(mine.data(), int(hdr[1] * sizeof(Record)), MPI_BYTE, 0, kTagRecords,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        Dataset ds(opt.set_dir, mine);  // maps only this rank's shards
        if (opt.preload) ds.preload();
        const long n = long(ds.size());
        std::vector<size_t> train;
        for (size_t i = 0; i < ds.size(); ++i)
            if (ds[i].is_train) train.push_back(i);
        RunOutput local;                // CRCs of this rank's records
        local.crc_resized.resize(ds.size());
        local.crc_normalized.resize(ds.size());
        std::unique_ptr<TensorWriter> writer;
        if (!opt.write_path.empty()) {
            // Rank 0 creates and sizes the file first; the others then open it.
            if (rank == 0) writer = std::make_unique<TensorWriter>(opt.write_path, total, true);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank != 0 && !collect)
                writer = std::make_unique<TensorWriter>(opt.write_path, total, false);
        }
        RankTimes rt{};
        rt.load = MPI_Wtime() - t;
        MPI_Barrier(MPI_COMM_WORLD);    // everyone has its data: start the timed passes
        Timings tm;
        tm.load = sw.lap();

        // ---- pass 1 ----------------------------------------------------------------------
        t = MPI_Wtime();
        Stats st;
        const long n_train = long(train.size());
#pragma omp parallel
        {
            std::vector<uint8_t> scratch(kResizedBytes);
#pragma omp for schedule(static) reduction(merge : st)
            for (long k = 0; k < n_train; ++k) pass1_record(ds, train[size_t(k)], scratch.data(), st);
        }
        rt.pass1 = MPI_Wtime() - t;
        const double t_local1 = sw.lap();

        uint64_t loc[7] = {st.n, st.s1[0], st.s1[1], st.s1[2], st.s2[0], st.s2[1], st.s2[2]};
        uint64_t glob[7];
        MPI_Allreduce(loc, glob, 7, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);   // collective
        Stats gst;
        gst.n = glob[0];
        for (int c = 0; c < 3; ++c) gst.s1[c] = glob[1 + c], gst.s2[c] = glob[4 + c];
        const Normalization norm = finalize(gst);
        tm.pass1 = t_local1;            // rank 0's own compute; the wait is in merge
        tm.merge = sw.lap();            // Allreduce: includes waiting for the slowest rank

        // ---- pass 2 ----------------------------------------------------------------------
        t = MPI_Wtime();
        std::vector<uint32_t> crc_from_wire;    // collector: CRC of every received tensor
        if (!collect) {
#pragma omp parallel
            {
                std::vector<uint8_t> scratch(kResizedBytes);
                std::vector<float> tensor(kOutPixels * 3);
#pragma omp for schedule(static)
                for (long i = 0; i < n; ++i) {
                    pass2_record(ds, size_t(i), norm, scratch.data(), false, tensor.data(), local, nullptr);
                    if (writer) writer->put(size_t(my_begin) + size_t(i), tensor.data());
                }
            }
        } else if (rank == 0) {
            // Collector: receive every tensor (any source), verify and optionally write it.
            // Messages from one rank arrive in the order it sent them (MPI non-overtaking),
            // so a per-source counter recovers the record index without a header.
            crc_from_wire.resize(total);
            std::vector<size_t> next(size_t(P), 0);
            for (int r = 1; r < P; ++r) next[size_t(r)] = cuts[size_t(r - 1)];
            std::vector<float> tensor(kOutPixels * 3);
            for (uint64_t k = 0; k < total; ++k) {
                MPI_Status status;
                MPI_Recv(tensor.data(), int(kNormalizedBytes), MPI_BYTE, MPI_ANY_SOURCE, kTagTensor,
                         MPI_COMM_WORLD, &status);
                const size_t i = next[size_t(status.MPI_SOURCE)]++;
                crc_from_wire[i] = crc32(tensor.data(), kNormalizedBytes);
                if (writer) writer->put(i, tensor.data());
            }
        } else {
            // Worker: compute each tensor and ship it to rank 0.
            std::vector<uint8_t> scratch(kResizedBytes);
            std::vector<float> buf[2] = {std::vector<float>(kOutPixels * 3),
                                         std::vector<float>(kOutPixels * 3)};
            MPI_Request req = MPI_REQUEST_NULL;
            int cur = 0;
            for (long i = 0; i < n; ++i) {
                const Record& r = ds[size_t(i)];
                float* tensor = buf[cur].data();
                // Same steps as pass2_record, split so MPI_Test can run between kernels.
                crop_resize(ds.pixels(r), r, scratch.data());
                local.crc_resized[size_t(i)] = crc32(scratch.data(), kResizedBytes);
                int flag = 0;
                if (opt.collect == "isend") MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
                normalize_chw(scratch.data(), norm, tensor);
                if (opt.collect == "isend") MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
                local.crc_normalized[size_t(i)] = crc32(tensor, kNormalizedBytes);

                const double tc = MPI_Wtime();
                if (opt.collect == "send") {
                    MPI_Send(tensor, int(kNormalizedBytes), MPI_BYTE, 0, kTagTensor, MPI_COMM_WORLD);
                } else {
                    MPI_Wait(&req, MPI_STATUS_IGNORE);      // previous tensor is out: its buffer is free
                    MPI_Isend(tensor, int(kNormalizedBytes), MPI_BYTE, 0, kTagTensor, MPI_COMM_WORLD, &req);
                    cur ^= 1;                                // compute the next one in the other buffer
                }
                rt.comm += MPI_Wtime() - tc;
            }
            const double tc = MPI_Wtime();
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            rt.comm += MPI_Wtime() - tc;
        }
        rt.pass2 = MPI_Wtime() - t;

        // ---- gather results on rank 0 (collectives) --------------------------------------
        std::vector<int> counts(size_t(P), 0), displs(size_t(P), 0);
        if (rank == 0)
            for (int r = first_worker; r < P; ++r) {
                counts[size_t(r)] = int(cuts[size_t(r - first_worker) + 1] - cuts[size_t(r - first_worker)]);
                displs[size_t(r)] = int(cuts[size_t(r - first_worker)]);
            }
        RunOutput out;
        if (rank == 0) {
            out.crc_resized.resize(total);
            out.crc_normalized.resize(total);
        }
        MPI_Gatherv(local.crc_resized.data(), int(n), MPI_UINT32_T, out.crc_resized.data(),
                    counts.data(), displs.data(), MPI_UINT32_T, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local.crc_normalized.data(), int(n), MPI_UINT32_T, out.crc_normalized.data(),
                    counts.data(), displs.data(), MPI_UINT32_T, 0, MPI_COMM_WORLD);
        const size_t np = static_cast<size_t>(P);
        std::vector<RankTimes> times(np);
        MPI_Gather(&rt, 4, MPI_DOUBLE, times.data(), 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        std::vector<uint64_t> rank_pixels(np);
        uint64_t my_pixels = ds.pixel_bytes();
        MPI_Gather(&my_pixels, 1, MPI_UINT64_T, rank_pixels.data(), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        tm.pass2 = sw.lap();

        writer.reset();
        if (rank == 0) {
            if (collect && crc_from_wire != out.crc_normalized)
                throw std::runtime_error("collector: a received tensor differs from what was sent");
            out.stats = gst;
            out.norm = norm;
            out.t = tm;
            const std::string set_name = std::filesystem::path(opt.set_dir).filename().string();
            const std::string mode = collect ? "collect-" + opt.collect : "part-" + opt.partition;
            finish_run(opt, set_name, all, out, P, threads, mode, sw);

            // Per-rank breakdown: the evidence for the decomposition and the overlap analysis.
            const auto path = std::filesystem::path(opt.out_dir) / "ranks.csv";
            std::FILE* f = std::fopen(path.c_str(), "w");
            if (!f) throw std::runtime_error("cannot write " + path.string());
            std::fputs("rank,records,pixel_bytes,t_load,t_pass1,t_pass2,t_comm\n", f);
            for (int r = 0; r < P; ++r)
                std::fprintf(f, "%d,%d,%llu,%.6f,%.6f,%.6f,%.6f\n", r, counts[size_t(r)],
                             static_cast<unsigned long long>(rank_pixels[size_t(r)]),
                             times[size_t(r)].load, times[size_t(r)].pass1, times[size_t(r)].pass2,
                             times[size_t(r)].comm);
            std::fclose(f);
            if (!opt.quiet) {
                double lo = 1e30, hi = 0, comm = 0;
                for (int r = first_worker; r < P; ++r) {
                    const double c = times[size_t(r)].pass1 + times[size_t(r)].pass2;
                    lo = std::min(lo, c), hi = std::max(hi, c), comm += times[size_t(r)].comm;
                }
                std::printf("  ranks: %d working, pass1+pass2 per rank min %.3f max %.3f s "
                            "(imbalance %.1f%%), time in send/wait %.3f s summed\n",
                            workers, lo, hi, 100.0 * (hi - lo) / hi, comm);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mpi[%d]: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return 0;
}
