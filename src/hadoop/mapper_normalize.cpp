// Job 2 mapper (Hadoop Streaming, map-only): normalize every image with Job 1's statistics.
//
// argv[1]: Job 1's reducer output (shipped to every task with `-files`); only the three
//          global keys all:0, all:1, all:2 are used
// stdin  : one image per line (train and test)
// stdout : image_id \t crc_resized,crc_normalized   (8 hex digits each; the CRC-32 of the
//          resized uint8 image and of the normalized CHW float32 tensor)
//          With --emit-tensors the base64 of the tensor is appended as a third field.
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../common/kernels.hpp"
#include "../common/textio.hpp"

namespace {

std::string base64_encode(const uint8_t* p, size_t n) {
    static const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((n + 2) / 3 * 4);
    for (size_t i = 0; i + 2 < n; i += 3) {
        const uint32_t v = uint32_t(p[i]) << 16 | uint32_t(p[i + 1]) << 8 | p[i + 2];
        out += a[v >> 18], out += a[(v >> 12) & 63], out += a[(v >> 6) & 63], out += a[v & 63];
    }
    if (n % 3 == 1) {
        const uint32_t v = uint32_t(p[n - 1]) << 16;
        out += a[v >> 18], out += a[(v >> 12) & 63], out += "==";
    } else if (n % 3 == 2) {
        const uint32_t v = uint32_t(p[n - 2]) << 16 | uint32_t(p[n - 1]) << 8;
        out += a[v >> 18], out += a[(v >> 12) & 63], out += a[(v >> 6) & 63], out += '=';
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) try {
    using namespace cub;
    bool emit = false;
    const char* stats_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--emit-tensors")) emit = true;
        else stats_path = argv[i];
    }
    if (!stats_path) throw std::runtime_error("usage: mapper_normalize STATS_FILE [--emit-tensors]");

    Stats st;
    int seen = 0;
    {
        std::ifstream f(stats_path);
        if (!f) throw std::runtime_error(std::string("cannot open ") + stats_path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("all:", 0) != 0) continue;
            const int c = line[4] - '0';
            const size_t tab = line.find('\t');
            if (c < 0 || c > 2 || tab == std::string::npos) throw std::runtime_error("bad stats line: " + line);
            const ChannelSums v = parse_sums(std::string_view(line).substr(tab + 1));
            st.n = v.n, st.s1[c] = v.s1, st.s2[c] = v.s2;
            seen |= 1 << c;
        }
    }
    if (seen != 7) throw std::runtime_error("stats file lacks all:0, all:1 or all:2");
    const Normalization norm = finalize(st);

    std::ios::sync_with_stdio(false);
    std::string line;
    line.reserve(2 << 20);
    std::vector<uint8_t> pixels, resized(kResizedBytes);
    std::vector<float> tensor(kOutPixels * 3);
    unsigned long long images = 0;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        Record rec;
        parse_line(line, rec, pixels);
        crop_resize(pixels.data(), rec, resized.data());
        const uint32_t cr = crc32(resized.data(), kResizedBytes);
        normalize_chw(resized.data(), norm, tensor.data());
        const uint32_t cn = crc32(tensor.data(), kNormalizedBytes);
        if (emit)
            std::printf("%u\t%08x,%08x\t%s\n", rec.image_id, cr, cn,
                        base64_encode(reinterpret_cast<const uint8_t*>(tensor.data()), kNormalizedBytes).c_str());
        else
            std::printf("%u\t%08x,%08x\n", rec.image_id, cr, cn);
        ++images;
    }
    std::fprintf(stderr, "reporter:counter:CUB,images_normalized,%llu\n", images);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "mapper_normalize: %s\n", e.what());
    return 1;
}
