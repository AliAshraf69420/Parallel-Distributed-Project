#include "kernels.hpp"

#include <array>
#include <cmath>
#include <cstring>

// finalize() must be compiled with -ffp-contract=off (the Makefile does this) so
// q - m*m is not fused into an FMA; otherwise std/LUT bits could differ from the spec.

namespace cub {

// ---------------------------------------------------------------- resize
AxisTable make_axis_table(int length) {
    AxisTable t;
    for (int i = 0; i < kOut; ++i) {
        const int64_t num = int64_t(2 * i + 1) * length - kOut;
        int64_t i0 = 0, f = 0;
        if (num > 0) {
            i0 = num / (2 * kOut);
            f = num - i0 * 2 * kOut;
        }
        if (i0 >= length - 1) {
            i0 = length - 1;
            f = 0;
        }
        t.i0[i] = int32_t(i0);
        t.i1[i] = int32_t(i0 + 1 < length - 1 ? i0 + 1 : length - 1);
        t.w[i] = uint32_t((f * kOne + kOut) / (2 * kOut));
    }
    return t;
}

void crop_resize(const uint8_t* img, const Record& r, uint8_t* dst) {
    const size_t stride = size_t(r.w) * 3;
    const uint8_t* base = img + size_t(r.by) * stride + size_t(r.bx) * 3;
    const AxisTable tx = make_axis_table(r.bw);
    const AxisTable ty = make_axis_table(r.bh);

    int32_t xa[kOut], xb[kOut];  // byte offsets of the two source columns
    for (int x = 0; x < kOut; ++x) xa[x] = tx.i0[x] * 3, xb[x] = tx.i1[x] * 3;

    for (int y = 0; y < kOut; ++y) {
        const uint8_t* r0 = base + size_t(ty.i0[y]) * stride;
        const uint8_t* r1 = base + size_t(ty.i1[y]) * stride;
        const uint32_t wy = ty.w[y], wy0 = kOne - wy;
        uint8_t* out = dst + size_t(y) * kOut * 3;
        for (int x = 0; x < kOut; ++x) {
            const uint32_t wx = tx.w[x], wx0 = kOne - wx;
            const uint8_t* p00 = r0 + xa[x];
            const uint8_t* p01 = r0 + xb[x];
            const uint8_t* p10 = r1 + xa[x];
            const uint8_t* p11 = r1 + xb[x];
            for (int c = 0; c < 3; ++c) {
                const uint32_t top = wx0 * p00[c] + wx * p01[c];
                const uint32_t bot = wx0 * p10[c] + wx * p11[c];
                out[x * 3 + c] = uint8_t((wy0 * top + wy * bot + (1u << 21)) >> 22);
            }
        }
    }
}

// ---------------------------------------------------------------- stats
void Stats::accumulate(const uint8_t* px) {
    // Per image: sum v <= 50176*255 and sum v^2 <= 50176*65025 both fit in uint32.
    uint32_t a[3] = {0, 0, 0}, b[3] = {0, 0, 0};
    for (size_t i = 0; i < kOutPixels; ++i) {
        for (int c = 0; c < 3; ++c) {
            const uint32_t v = px[i * 3 + c];
            a[c] += v;
            b[c] += v * v;
        }
    }
    n += kOutPixels;
    for (int c = 0; c < 3; ++c) s1[c] += a[c], s2[c] += b[c];
}

Normalization finalize(const Stats& st) {
    Normalization nz{};
    for (int c = 0; c < 3; ++c) {
        const double m = double(st.s1[c]) / double(st.n);
        const double q = double(st.s2[c]) / double(st.n);
        const double mm = m * m;  // separate statement: no fused multiply-subtract
        const double s = std::sqrt(q - mm);
        nz.m[c] = m;
        nz.s[c] = s;
        nz.mean[c] = m / 255.0;
        nz.std[c] = s / 255.0;
        for (int v = 0; v < 256; ++v) nz.lut[c][v] = float((double(v) - m) / s);
    }
    return nz;
}

// ---------------------------------------------------------------- normalize
void normalize_chw(const uint8_t* px, const Normalization& nz, float* out) {
    float* o0 = out;
    float* o1 = out + kOutPixels;
    float* o2 = out + 2 * kOutPixels;
    for (size_t i = 0; i < kOutPixels; ++i) {
        o0[i] = nz.lut[0][px[i * 3 + 0]];
        o1[i] = nz.lut[1][px[i * 3 + 1]];
        o2[i] = nz.lut[2][px[i * 3 + 2]];
    }
}

// ---------------------------------------------------------------- crc32
namespace {

constexpr std::array<std::array<uint32_t, 256>, 8> make_crc_tables() {
    std::array<std::array<uint32_t, 256>, 8> t{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        t[0][i] = c;
    }
    for (int k = 1; k < 8; ++k)
        for (uint32_t i = 0; i < 256; ++i) t[k][i] = (t[k - 1][i] >> 8) ^ t[0][t[k - 1][i] & 0xFF];
    return t;
}

constexpr auto kCrc = make_crc_tables();
static_assert(kCrc[0][1] == 0x77073096u);

}  // namespace

uint32_t crc32(const void* data, size_t n, uint32_t crc) {
    const auto* p = static_cast<const uint8_t*>(data);
    crc = ~crc;
    while (n >= 8) {
        uint32_t lo, hi;
        std::memcpy(&lo, p, 4);
        std::memcpy(&hi, p + 4, 4);
        lo ^= crc;
        crc = kCrc[7][lo & 0xFF] ^ kCrc[6][(lo >> 8) & 0xFF] ^ kCrc[5][(lo >> 16) & 0xFF] ^
              kCrc[4][lo >> 24] ^ kCrc[3][hi & 0xFF] ^ kCrc[2][(hi >> 8) & 0xFF] ^
              kCrc[1][(hi >> 16) & 0xFF] ^ kCrc[0][hi >> 24];
        p += 8;
        n -= 8;
    }
    while (n--) crc = (crc >> 8) ^ kCrc[0][(crc ^ *p++) & 0xFF];
    return ~crc;
}

}  // namespace cub
