// Sequential baseline: crop + resize -> pass 1 (channel stats) -> pass 2 (normalize).
// The per-record work lives in src/common/app.hpp; this file is only the loops.
#include <cstdio>
#include <exception>
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
    const std::vector<size_t> p1 = pass1_indices(ds, opt.cache);
    out.t.load = sw.lap();

    // Pass 1: channel statistics over the train split.
    for (size_t i : p1) pass1_record(ds, i, &resized[opt.cache ? i * kResizedBytes : 0], out.stats);
    out.t.pass1 = sw.lap();

    out.norm = finalize(out.stats);
    out.t.merge = sw.lap();

    // Pass 2: normalize every record.
    for (size_t i = 0; i < ds.size(); ++i)
        pass2_record(ds, i, out.norm, &resized[opt.cache ? i * kResizedBytes : 0], opt.cache,
                     tensor.data(), out, writer.get());
    out.t.pass2 = sw.lap();

    writer.reset();
    finish_run(opt, ds, out, 1, 1, "-", sw);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "seq: %s\n", e.what());
    return 1;
}
