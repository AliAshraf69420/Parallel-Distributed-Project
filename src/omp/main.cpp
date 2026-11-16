// OpenMP version of the sequential baseline (same per-record work, see src/common/app.hpp).
//
// Pass 1: `parallel for` over train records; each thread accumulates a private Stats and
//         the user-defined reduction `merge` combines them (integer addition, so the result
//         is identical for any thread count or schedule). No critical sections.
// Pass 2: `parallel for` over all records; every iteration writes only its own slots
//         (crc arrays, tensor-file block), so no synchronization at all.
// Both loops use schedule(runtime): --schedule / OMP_SCHEDULE picks static, dynamic or
// guided without recompiling. Image sizes vary a lot, so the choice matters.
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "../common/app.hpp"

#pragma omp declare reduction(merge : cub::Stats : omp_out.merge(omp_in)) \
    initializer(omp_priv = cub::Stats{})

namespace {

// "dynamic,8" -> omp_set_schedule(omp_sched_dynamic, 8)
void apply_schedule(const std::string& s) {
    const auto comma = s.find(',');
    const std::string kind = s.substr(0, comma);
    const int chunk = comma == std::string::npos ? 0 : std::atoi(s.c_str() + comma + 1);
    omp_sched_t k;
    if (kind == "static") k = omp_sched_static;
    else if (kind == "dynamic") k = omp_sched_dynamic;
    else if (kind == "guided") k = omp_sched_guided;
    else if (kind == "auto") k = omp_sched_auto;
    else throw std::runtime_error("unknown schedule '" + s + "'");
    omp_set_schedule(k, chunk);
}

std::string current_schedule() {
    omp_sched_t k;
    int chunk;
    omp_get_schedule(&k, &chunk);
    const char* name = "?";
    switch (k & ~omp_sched_monotonic) {
        case omp_sched_static: name = "static"; break;
        case omp_sched_dynamic: name = "dynamic"; break;
        case omp_sched_guided: name = "guided"; break;
        case omp_sched_auto: name = "auto"; break;
        default: break;
    }
    // chunk <= 0 means "implementation default" (static: n/threads; dynamic/guided: 1)
    return chunk > 0 ? std::string(name) + ";" + std::to_string(chunk) : name;
}

}  // namespace

int main(int argc, char** argv) try {
    using namespace cub;
    const Options opt = parse_args(argc, argv, "omp");
    if (opt.threads > 0) omp_set_num_threads(opt.threads);
    if (!opt.schedule.empty()) apply_schedule(opt.schedule);
    const int threads = omp_get_max_threads();
    Stopwatch sw;

    Dataset ds(opt.set_dir);
    if (opt.preload) ds.preload();
    RunOutput out;
    out.crc_resized.resize(ds.size());
    out.crc_normalized.resize(ds.size());
    std::unique_ptr<TensorWriter> writer;
    if (!opt.write_path.empty()) writer = std::make_unique<TensorWriter>(opt.write_path, ds.size());
    std::vector<uint8_t> cache(opt.cache ? ds.size() * kResizedBytes : 0);
    const std::vector<size_t> order = record_order(ds, opt.order);
    const std::vector<size_t> p1 = pass1_indices(ds, order, opt.cache);
    const long n1 = long(p1.size()), n2 = long(ds.size());
    out.t.load = sw.lap();

    // Pass 1: channel statistics over the train split.
    Stats st;
#pragma omp parallel
    {
        std::vector<uint8_t> scratch(opt.cache ? 0 : kResizedBytes);  // per-thread buffer
#pragma omp for schedule(runtime) reduction(merge : st)
        for (long k = 0; k < n1; ++k) {
            const size_t i = p1[k];
            pass1_record(ds, i, opt.cache ? &cache[i * kResizedBytes] : scratch.data(), st);
        }
    }
    out.stats = st;
    out.t.pass1 = sw.lap();

    out.norm = finalize(out.stats);
    out.t.merge = sw.lap();

    // Pass 2: normalize every record.
#pragma omp parallel
    {
        std::vector<uint8_t> scratch(opt.cache ? 0 : kResizedBytes);
        std::vector<float> tensor(kOutPixels * 3);
#pragma omp for schedule(runtime)
        for (long k = 0; k < n2; ++k) {
            const size_t i = order[k];
            uint8_t* buf = opt.cache ? &cache[i * kResizedBytes] : scratch.data();
            pass2_record(ds, i, out.norm, buf, opt.cache, tensor.data(), out, writer.get());
        }
    }
    out.t.pass2 = sw.lap();

    writer.reset();
    finish_run(opt, ds, out, 1, threads, current_schedule(), sw);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "omp: %s\n", e.what());
    return 1;
}
