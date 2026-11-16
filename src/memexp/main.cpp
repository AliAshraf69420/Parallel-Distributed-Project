// Memory-hierarchy experiments for the Phase 1 report (Lab 2 / Lab 5 style), on our own data.
//
//   memexp wss            load latency vs working-set size (16 KiB .. 1 GiB), pointer chase:
//                         shows the L1 / L2 / L3 / DRAM steps
//   memexp stride         time per access vs stride (1 B .. 8 KiB) over 1 GiB of shard bytes
//   memexp resize-order   the real crop+resize with the production row-major traversal vs a
//                         column-major traversal; the outputs are verified identical, so the
//                         time difference is purely access-pattern (locality)
//   memexp false-sharing  8 threads accumulating per-pixel sums into per-thread counters that
//                         are (a) packed: adjacent in one array, sharing cache lines
//                         (b) padded: alignas(64), one line per thread
//                         (c) local: registers, merged once per image (what src/ does)
//
// Options: --set SET (default 500MB; resize-order / false-sharing), --variant V (run only one
// variant, for `perf stat`), -t N (false-sharing threads, default 8), --reps K.
// Output: one CSV block per experiment on stdout.
//
// perf example (cache misses of each traversal):
//   perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses
//       build/memexp resize-order --variant col --set 500MB   (one command line)
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "../common/cub_format.hpp"
#include "../common/kernels.hpp"

namespace {

using namespace cub;
using Clock = std::chrono::steady_clock;

double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

struct Args {
    std::string cmd, set = "500MB", variant;
    int threads = 8, reps = 3;
};

std::string set_path(const std::string& s) {
    return std::filesystem::exists(std::filesystem::path(s) / "index.bin") ? s : "data/processed/" + s;
}

// Copy `bytes` of real shard data into an owned, page-aligned buffer.
std::vector<uint8_t> shard_bytes(const Dataset& ds, size_t bytes) {
    std::vector<uint8_t> buf(bytes);
    size_t filled = 0;
    for (size_t i = 0; i < ds.size() && filled < bytes; ++i) {
        const Record& r = ds[i];
        const size_t n = std::min(size_t(r.w) * r.h * 3, bytes - filled);
        std::memcpy(buf.data() + filled, ds.pixels(r), n);
        filled += n;
    }
    if (filled < bytes) throw std::runtime_error("set too small for requested buffer");
    return buf;
}

// ------------------------------------------------------------------ wss
// Load-to-load latency vs working-set size: a pointer chase through a random cyclic
// permutation of the 64-byte lines of the working set. Every load's address depends on the
// previous load, so neither the compiler nor the hardware prefetcher can hide the latency;
// the steps show where the working set stops fitting in L1, L2 and L3.
void wss(const Args& a) {
    const size_t max_bytes = size_t(1) << 30;
    std::vector<uint64_t> buf(max_bytes / 8);
    std::printf("experiment,working_set_bytes,hops,ns_per_load\n");
    const size_t hops = size_t(1) << 24;
    for (size_t ws = 16 << 10; ws <= max_bytes; ws *= 2) {
        const size_t lines = ws / 64;
        std::vector<uint32_t> perm(lines);
        for (size_t k = 0; k < lines; ++k) perm[k] = uint32_t(k);
        uint64_t rng = 0x9E3779B97F4A7C15ull ^ ws;  // xorshift64, fixed seed per size
        for (size_t k = lines - 1; k > 0; --k) {
            rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
            std::swap(perm[k], perm[rng % (k + 1)]);
        }
        for (size_t k = 0; k < lines; ++k) buf[size_t(perm[k]) * 8] = perm[(k + 1) % lines];
        double best = 1e30;
        uint64_t idx = perm[0];
        for (int rep = 0; rep < a.reps; ++rep) {
            const auto t0 = Clock::now();
            for (size_t h = 0; h < hops; ++h) idx = buf[idx * 8];
            best = std::min(best, secs(t0));
        }
        if (idx >= lines) throw std::runtime_error("wss: chase left the working set");
        std::printf("wss,%zu,%zu,%.3f\n", ws, hops, best * 1e9 / double(hops));
        std::fflush(stdout);
    }
}

// ------------------------------------------------------------------ stride
void stride(const Args& a) {
    Dataset ds(set_path("full"));
    const size_t bytes = size_t(1) << 30;
    std::vector<uint8_t> buf = shard_bytes(ds, bytes);
    std::printf("experiment,stride_bytes,accesses,ns_per_access,bytes_per_ns_useful,checksum\n");
    for (size_t s = 1; s <= 8192; s *= 2) {
        const size_t n = bytes / s;
        double best = 1e30;
        uint64_t sum = 0;
        for (int rep = 0; rep < a.reps; ++rep) {
            const auto t0 = Clock::now();
            for (size_t off = 0; off < bytes; off += s) sum += buf[off];
            best = std::min(best, secs(t0));
        }
        std::printf("stride,%zu,%zu,%.3f,%.3f,%llu\n", s, n, best * 1e9 / double(n),
                    double(n) / (best * 1e9), static_cast<unsigned long long>(sum));
        std::fflush(stdout);
    }
}

// ------------------------------------------------------------------ resize-order
// Column-major variant of crop_resize: identical arithmetic, x outer / y inner, so
// consecutive output pixels read source rows that are far apart in memory.
void crop_resize_colmajor(const uint8_t* img, const Record& r, uint8_t* dst) {
    const size_t stride = size_t(r.w) * 3;
    const uint8_t* base = img + size_t(r.by) * stride + size_t(r.bx) * 3;
    const AxisTable tx = make_axis_table(r.bw);
    const AxisTable ty = make_axis_table(r.bh);
    for (int x = 0; x < kOut; ++x) {
        const uint32_t wx = tx.w[x], wx0 = kOne - wx;
        const int xa = tx.i0[x] * 3, xb = tx.i1[x] * 3;
        for (int y = 0; y < kOut; ++y) {
            const uint8_t* r0 = base + size_t(ty.i0[y]) * stride;
            const uint8_t* r1 = base + size_t(ty.i1[y]) * stride;
            const uint32_t wy = ty.w[y], wy0 = kOne - wy;
            uint8_t* out = dst + (size_t(y) * kOut + x) * 3;
            for (int c = 0; c < 3; ++c) {
                const uint32_t top = wx0 * r0[xa + c] + wx * r0[xb + c];
                const uint32_t bot = wx0 * r1[xa + c] + wx * r1[xb + c];
                out[c] = uint8_t((wy0 * top + wy * bot + (1u << 21)) >> 22);
            }
        }
    }
}

void resize_order(const Args& a) {
    Dataset ds(set_path(a.set));
    ds.preload();
    std::vector<uint8_t> buf(kResizedBytes);
    std::vector<uint32_t> crc_row(ds.size()), crc_col(ds.size());
    std::printf("experiment,set,variant,images,seconds,us_per_image\n");
    for (const std::string v : {"row", "col"}) {
        if (!a.variant.empty() && a.variant != v) continue;
        auto& crc = v == "row" ? crc_row : crc_col;
        double best = 1e30;
        for (int rep = 0; rep < a.reps; ++rep) {
            const auto t0 = Clock::now();
            for (size_t i = 0; i < ds.size(); ++i) {
                const Record& r = ds[i];
                if (v == "row") crop_resize(ds.pixels(r), r, buf.data());
                else crop_resize_colmajor(ds.pixels(r), r, buf.data());
                crc[i] = crc32(buf.data(), kResizedBytes);
            }
            best = std::min(best, secs(t0));
        }
        std::printf("resize-order,%s,%s,%zu,%.4f,%.2f\n", ds.name().c_str(), v.c_str(), ds.size(), best,
                    best * 1e6 / double(ds.size()));
        std::fflush(stdout);
    }
    if (a.variant.empty() && crc_row != crc_col) throw std::runtime_error("row/col outputs differ");
    if (a.variant.empty()) std::fprintf(stderr, "resize-order: row and col outputs identical\n");
}

// ------------------------------------------------------------------ false-sharing
struct Packed { uint64_t s1[3]; };                 // 24 B: ~2.7 threads per 64-B line
struct alignas(64) Padded { uint64_t s1[3]; };     // one cache line per thread

template <class Slot>
double accumulate_shared(const Dataset& ds, const std::vector<uint8_t>& resized, int threads,
                         uint64_t total[3]) {
    std::vector<Slot> slots(static_cast<size_t>(threads));
    const long n = long(ds.size());
    const auto t0 = Clock::now();
#pragma omp parallel num_threads(threads)
    {
        // volatile: every pixel really loads and stores its thread's counter in memory,
        // as naive code with a shared counter array would.
        volatile uint64_t* c = slots[size_t(omp_get_thread_num())].s1;
#pragma omp for schedule(static)
        for (long i = 0; i < n; ++i) {
            const uint8_t* px = &resized[size_t(i) * kResizedBytes];
            for (size_t k = 0; k < kOutPixels; ++k)
                for (int ch = 0; ch < 3; ++ch) c[ch] = c[ch] + px[k * 3 + ch];
        }
    }
    const double t = secs(t0);
    for (auto& s : slots)
        for (int ch = 0; ch < 3; ++ch) total[ch] += s.s1[ch];
    return t;
}

double accumulate_local(const Dataset& ds, const std::vector<uint8_t>& resized, int threads,
                        uint64_t total[3]) {
    uint64_t s0 = 0, s1 = 0, s2 = 0;
    const long n = long(ds.size());
    const auto t0 = Clock::now();
#pragma omp parallel for num_threads(threads) schedule(static) reduction(+ : s0, s1, s2)
    for (long i = 0; i < n; ++i) {
        const uint8_t* px = &resized[size_t(i) * kResizedBytes];
        uint64_t a = 0, b = 0, c = 0;
        for (size_t k = 0; k < kOutPixels; ++k) a += px[k * 3], b += px[k * 3 + 1], c += px[k * 3 + 2];
        s0 += a, s1 += b, s2 += c;
    }
    const double t = secs(t0);
    total[0] += s0, total[1] += s1, total[2] += s2;
    return t;
}

void false_sharing(const Args& a) {
    Dataset ds(set_path(a.set));
    ds.preload();
    std::vector<uint8_t> resized(ds.size() * kResizedBytes);  // resize once, outside the timing
#pragma omp parallel for schedule(dynamic, 8)
    for (long i = 0; i < long(ds.size()); ++i)
        crop_resize(ds.pixels(ds[size_t(i)]), ds[size_t(i)], &resized[size_t(i) * kResizedBytes]);

    std::printf("experiment,set,variant,threads,seconds,sum_r\n");
    uint64_t ref[3] = {0, 0, 0};
    accumulate_local(ds, resized, 1, ref);
    for (const std::string v : {"packed", "padded", "local"}) {
        if (!a.variant.empty() && a.variant != v) continue;
        double best = 1e30;
        for (int rep = 0; rep < a.reps; ++rep) {
            uint64_t tot[3] = {0, 0, 0};
            const double t = v == "packed"   ? accumulate_shared<Packed>(ds, resized, a.threads, tot)
                             : v == "padded" ? accumulate_shared<Padded>(ds, resized, a.threads, tot)
                                             : accumulate_local(ds, resized, a.threads, tot);
            if (tot[0] != ref[0] || tot[1] != ref[1] || tot[2] != ref[2])
                throw std::runtime_error("false-sharing: wrong sum in variant " + v);
            best = std::min(best, t);
        }
        std::printf("false-sharing,%s,%s,%d,%.4f,%llu\n", ds.name().c_str(), v.c_str(), a.threads, best,
                    static_cast<unsigned long long>(ref[0]));
        std::fflush(stdout);
    }
}

}  // namespace

int main(int argc, char** argv) try {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto val = [&] {
            if (i + 1 >= argc) throw std::runtime_error(s + " needs a value");
            return std::string(argv[++i]);
        };
        if (s == "--set") a.set = val();
        else if (s == "--variant") a.variant = val();
        else if (s == "-t") a.threads = std::atoi(val().c_str());
        else if (s == "--reps") a.reps = std::atoi(val().c_str());
        else if (a.cmd.empty()) a.cmd = s;
        else throw std::runtime_error("unexpected argument " + s);
    }
    if (a.cmd == "wss") wss(a);
    else if (a.cmd == "stride") stride(a);
    else if (a.cmd == "resize-order") resize_order(a);
    else if (a.cmd == "false-sharing") false_sharing(a);
    else {
        std::fprintf(stderr, "usage: memexp wss|stride|resize-order|false-sharing "
                             "[--set SET] [--variant V] [-t N] [--reps K]\n");
        return 2;
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "memexp: %s\n", e.what());
    return 1;
}
