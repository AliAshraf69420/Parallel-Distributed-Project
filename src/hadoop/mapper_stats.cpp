// Job 1 mapper (Hadoop Streaming): channel statistics.
//
// stdin : one image per line (see src/common/textio.hpp)
// stdout: for every TRAIN image, the exact integer sums of its resized pixels, twice:
//           all:<c>      \t n,S1,S2      c = channel 0..2   (global statistics)
//           cls<kkk>:<c> \t n,S1,S2      kkk = class id     (per-class statistics)
//         with n = 224*224, S1 = sum v, S2 = sum v*v. Test images are skipped without
//         decoding. One record per image and key, so the combiner has real work to do.
// stderr: Hadoop counters (reporter:counter:CUB,...).
//
// Local test:  cat data/processed/text/100MB.txt | build/hadoop/mapper_stats | sort |
//              build/hadoop/reducer_stats
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "../common/kernels.hpp"
#include "../common/textio.hpp"

int main() try {
    using namespace cub;
    std::ios::sync_with_stdio(false);
    std::string line;
    line.reserve(2 << 20);
    std::vector<uint8_t> pixels, resized(kResizedBytes);
    unsigned long long images = 0, train = 0;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        ++images;
        Record rec;
        const std::string_view b64 = parse_line_header(line, rec);
        if (!rec.is_train) continue;
        ++train;
        base64_decode(b64, pixels);
        if (pixels.size() != size_t(rec.w) * rec.h * 3)
            throw std::runtime_error("pixel count mismatch for image_id " + std::to_string(rec.image_id));
        crop_resize(pixels.data(), rec, resized.data());
        Stats st;
        st.accumulate(resized.data());
        for (int c = 0; c < 3; ++c) {
            const auto n = static_cast<unsigned long long>(st.n);
            const auto s1 = static_cast<unsigned long long>(st.s1[c]);
            const auto s2 = static_cast<unsigned long long>(st.s2[c]);
            std::printf("all:%d\t%llu,%llu,%llu\n", c, n, s1, s2);
            std::printf("cls%03u:%d\t%llu,%llu,%llu\n", unsigned(rec.class_id), c, n, s1, s2);
        }
    }
    std::fprintf(stderr, "reporter:counter:CUB,images_read,%llu\n", images);
    std::fprintf(stderr, "reporter:counter:CUB,train_images_resized,%llu\n", train);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "mapper_stats: %s\n", e.what());
    return 1;
}
