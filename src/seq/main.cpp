// Sequential baseline: crop + resize -> pass 1 (channel stats) -> pass 2 (normalize).
// The per-record work lives in src/common/app.hpp; this file is only the loops.
#include <chrono>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <memory>
#include <vector>

#include "../common/app.hpp"

int main(int argc, char** argv) try {
    using namespace cub;
    const Options opt = parse_args(argc, argv, "seq");
    Stopwatch sw;

    Dataset ds(opt.set_dir);
    if (opt.preload) ds.preload();
    RunOutput out;
    out.crc_resized.resize(ds.size());
    out.crc_normalized.resize(ds.size());
    std::unique_ptr<TensorWriter> writer;
    if (!opt.write_path.empty()) writer = std::make_unique<TensorWriter>(opt.write_path, ds.size());
    // With --cache, one resized image per record (~1.8 GB for the full set); else one scratch.
    std::vector<uint8_t> resized((opt.cache ? ds.size() : 1) * kResizedBytes);
    std::vector<float> tensor(kOutPixels * 3);
    const std::vector<size_t> order = record_order(ds, opt.order);
    const std::vector<size_t> p1 = pass1_indices(ds, order, opt.cache);
    out.t.load = sw.lap();

    // Pass 1: channel statistics over the train split.
    for (size_t i : p1) pass1_record(ds, i, &resized[opt.cache ? i * kResizedBytes : 0], out.stats);
    out.t.pass1 = sw.lap();

    out.norm = finalize(out.stats);
    out.t.merge = sw.lap();

    // Pass 2: normalize every record. With --record-times, also time each record
    // (measures how uneven the per-image work is; adds ~2 clock reads per image).
    std::vector<double> rec_us(opt.record_times.empty() ? 0 : ds.size());
    for (size_t i : order) {
        const auto t0 = std::chrono::steady_clock::now();
        pass2_record(ds, i, out.norm, &resized[opt.cache ? i * kResizedBytes : 0], opt.cache,
                     tensor.data(), out, writer.get());
        if (!rec_us.empty())
            rec_us[i] = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - t0).count();
    }
    out.t.pass2 = sw.lap();
    if (!rec_us.empty()) {
        std::FILE* f = std::fopen(opt.record_times.c_str(), "w");
        if (!f) throw std::runtime_error("cannot write " + opt.record_times);
        std::fputs("image_id,w,h,bw,bh,pass2_us\n", f);
        for (size_t i = 0; i < ds.size(); ++i)
            std::fprintf(f, "%u,%u,%u,%u,%u,%.2f\n", ds[i].image_id, ds[i].w, ds[i].h,
                         ds[i].bw, ds[i].bh, rec_us[i]);
        std::fclose(f);
        sw.lap();  // file writing is not part of the timed passes
    }

    writer.reset();
    finish_run(opt, ds, out, 1, 1, "-", sw);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "seq: %s\n", e.what());
    return 1;
}
