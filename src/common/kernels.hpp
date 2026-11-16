// The core computation, exactly as specified in preprocessing/reference.py.
// Every function here must stay bit-exact with that spec; tests/check.py enforces it.
#pragma once

#include <cstddef>
#include <cstdint>

#include "cub_format.hpp"

namespace cub {

constexpr int kOut = 224;                                  // output side
constexpr int kFracBits = 11;
constexpr uint32_t kOne = 1u << kFracBits;                 // 2048
constexpr size_t kOutPixels = size_t(kOut) * kOut;         // 50 176
constexpr size_t kResizedBytes = kOutPixels * 3;           // HWC uint8
constexpr size_t kNormalizedBytes = kOutPixels * 3 * 4;    // CHW float32

// ---- 1. crop + fixed-point bilinear resize ---------------------------------
struct AxisTable {
    int32_t i0[kOut], i1[kOut];
    uint32_t w[kOut];  // weight of i1, in [0, kOne)
};
AxisTable make_axis_table(int length);

// img: full HWC image of record r; dst: kResizedBytes (224x224x3 HWC).
void crop_resize(const uint8_t* img, const Record& r, uint8_t* dst);

// ---- 2. pass-1 statistics (exact integers; merge = addition) ----------------
struct Stats {
    uint64_t n = 0;       // pixels per channel
    uint64_t s1[3] = {};  // sum v
    uint64_t s2[3] = {};  // sum v*v
    void accumulate(const uint8_t* resized);
    void merge(const Stats& o) {
        n += o.n;
        for (int c = 0; c < 3; ++c) s1[c] += o.s1[c], s2[c] += o.s2[c];
    }
};

struct Normalization {
    double m[3], s[3];         // raw 0..255 scale
    double mean[3], std[3];    // [0,1] scale (reported)
    float lut[3][256];
};
Normalization finalize(const Stats& st);

// ---- 3. normalize: HWC uint8 -> CHW float32 through the LUT ------------------
void normalize_chw(const uint8_t* resized, const Normalization& nz, float* out);

// ---- 4. CRC-32 (zlib / IEEE 802.3), slicing-by-8 -----------------------------
uint32_t crc32(const void* data, size_t n, uint32_t crc = 0);

}  // namespace cub
