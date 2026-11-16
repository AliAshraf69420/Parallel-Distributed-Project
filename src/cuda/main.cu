// CUDA version (bonus): the compute-heavy kernels on the GPU, in batches.
//
// Per batch of images:
//   host    copy each image's crop rows into one pinned staging buffer        (OpenMP)
//   upload  staging buffer + per-image descriptors                            (PCIe)
//   GPU     resize_kernel:    one thread per output pixel; the same fixed-point bilinear
//                             arithmetic as crop_resize(), so the result is bit-identical
//           sums_kernel:      pass 1; one block per image, shared-memory tree reduction to
//                             the image's exact sum v and sum v*v per channel
//           normalize_kernel: pass 2; one thread per output pixel, LUT gather, HWC -> CHW
//   download  pass 1: 24 bytes per image; pass 2: resized image + float tensor (PCIe)
//   host    merge sums (pass 1) or CRC-32 of both outputs (pass 2, OpenMP)
//
// Timings are split into stage / upload / kernel / download / host so the report can show
// where the time goes: for this workload the PCIe transfers and the host-side CRC, not the
// kernels, set the end-to-end time.
//
//   build/cuda --set SET [--batch 256] [-t host_threads] [--preload] [--out DIR] [--csv FILE]
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/app.hpp"

namespace {

using namespace cub;

struct ImageDesc {       // one image of the batch, as the kernels see it
    uint32_t src;        // byte offset of its first crop row in the staging buffer
    uint32_t stride;     // bytes per source row (full image width * 3)
    uint32_t bx3;        // bytes to skip at the start of each row (bx * 3)
    uint16_t bw, bh;     // crop size
};

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// Source index and weight for output index i along an axis of length `len`
// (identical to make_axis_table() in src/common/kernels.cpp).
__device__ inline void axis(int i, int len, int& i0, int& i1, uint32_t& w) {
    const long long num = (long long)(2 * i + 1) * len - kOut;
    long long a = 0, f = 0;
    if (num > 0) {
        a = num / (2 * kOut);
        f = num - a * 2 * kOut;
    }
    if (a >= len - 1) {
        a = len - 1;
        f = 0;
    }
    i0 = int(a);
    i1 = int(a + 1 < len - 1 ? a + 1 : len - 1);
    w = uint32_t((f * kOne + kOut) / (2 * kOut));
}

__global__ void resize_kernel(const uint8_t* src, const ImageDesc* desc, uint8_t* out, int count) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int img = blockIdx.z;
    if (x >= kOut || y >= kOut || img >= count) return;
    const ImageDesc d = desc[img];
    int x0, x1, y0, y1;
    uint32_t wx, wy;
    axis(x, d.bw, x0, x1, wx);
    axis(y, d.bh, y0, y1, wy);
    const uint8_t* base = src + d.src + d.bx3;
    const uint8_t* r0 = base + size_t(y0) * d.stride;
    const uint8_t* r1 = base + size_t(y1) * d.stride;
    uint8_t* o = out + (size_t(img) * kOutPixels + size_t(y) * kOut + x) * 3;
    const uint32_t wx0 = kOne - wx, wy0 = kOne - wy;
    for (int c = 0; c < 3; ++c) {
        const uint32_t top = wx0 * r0[x0 * 3 + c] + wx * r0[x1 * 3 + c];
        const uint32_t bot = wx0 * r1[x0 * 3 + c] + wx * r1[x1 * 3 + c];
        o[c] = uint8_t((wy0 * top + wy * bot + (1u << 21)) >> 22);
    }
}

// One block (256 threads) per image. Per image, sum v <= 50176*255 and sum v*v <= 50176*65025
// both fit in 32 bits, so the reduction is exact in uint32.
__global__ void sums_kernel(const uint8_t* resized, uint32_t* sums /* [count][6] */) {
    __shared__ uint32_t acc[6][256];
    const int img = blockIdx.x, t = threadIdx.x;
    const uint8_t* px = resized + size_t(img) * kResizedBytes;
    uint32_t a[3] = {0, 0, 0}, b[3] = {0, 0, 0};
    for (int k = t; k < int(kOutPixels); k += 256)
        for (int c = 0; c < 3; ++c) {
            const uint32_t v = px[k * 3 + c];
            a[c] += v;
            b[c] += v * v;
        }
    for (int c = 0; c < 3; ++c) acc[c][t] = a[c], acc[3 + c][t] = b[c];
    __syncthreads();
    for (int half = 128; half > 0; half >>= 1) {
        if (t < half)
            for (int j = 0; j < 6; ++j) acc[j][t] += acc[j][t + half];
        __syncthreads();
    }
    if (t < 6) sums[img * 6 + t] = acc[t][0];
}

__global__ void normalize_kernel(const uint8_t* resized, const float* lut /* [3][256] */,
                                 float* tensor, int count) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;   // pixel index within the image
    const int img = blockIdx.y;
    if (p >= int(kOutPixels) || img >= count) return;
    const uint8_t* px = resized + (size_t(img) * kOutPixels + p) * 3;
    float* o = tensor + size_t(img) * kOutPixels * 3;
    o[p] = lut[px[0]];
    o[kOutPixels + p] = lut[256 + px[1]];
    o[2 * kOutPixels + p] = lut[512 + px[2]];
}

struct Breakdown {
    double stage = 0, upload = 0, kernel = 0, download = 0, host = 0;
};

template <class T>
struct Pinned {     // page-locked host memory: required for fast (DMA) transfers
    T* p = nullptr;
    explicit Pinned(size_t n) { check(cudaMallocHost(&p, n * sizeof(T)), "cudaMallocHost"); }
    ~Pinned() { cudaFreeHost(p); }
};

template <class T>
struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc"); }
    ~Device() { cudaFree(p); }
};

double now() { return omp_get_wtime(); }

}  // namespace

int main(int argc, char** argv) try {
    // --batch is ours; everything else goes to the shared option parser.
    int batch = 256;
    std::vector<char*> args;
    for (int i = 0; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--batch") && i + 1 < argc) batch = std::atoi(argv[++i]);
        else args.push_back(argv[i]);
    }
    const Options opt = parse_args(int(args.size()), args.data(), "cuda");
    if (opt.threads > 0) omp_set_num_threads(opt.threads);
    const int threads = omp_get_max_threads();
    if (batch < 1 || batch > 1024) throw std::runtime_error("--batch must be 1..1024");
    Stopwatch sw;

    Dataset ds(opt.set_dir);
    if (opt.preload) ds.preload();
    RunOutput out;
    out.crc_resized.resize(ds.size());
    out.crc_normalized.resize(ds.size());

    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    const size_t max_src = 512u * 512u * 3u;                 // CUB images are at most 500 x 500
    Pinned<uint8_t> h_src(size_t(batch) * max_src);
    Pinned<ImageDesc> h_desc(batch);
    Pinned<uint32_t> h_sums(size_t(batch) * 6);
    Pinned<uint8_t> h_resized(size_t(batch) * kResizedBytes);
    Pinned<float> h_tensor(size_t(batch) * kOutPixels * 3);
    Device<uint8_t> d_src(size_t(batch) * max_src);
    Device<ImageDesc> d_desc(batch);
    Device<uint32_t> d_sums(size_t(batch) * 6);
    Device<uint8_t> d_resized(size_t(batch) * kResizedBytes);
    Device<float> d_tensor(size_t(batch) * kOutPixels * 3);
    Device<float> d_lut(768);
    out.t.load = sw.lap();

    // Stage the crop rows of records idx[first .. first+count) and upload them.
    auto stage_and_upload = [&](const std::vector<size_t>& idx, size_t first, int count, Breakdown& b) {
        double t = now();
        size_t off = 0;
        for (int k = 0; k < count; ++k) {
            const Record& r = ds[idx[first + size_t(k)]];
            if (size_t(r.w) * r.h * 3 > max_src) throw std::runtime_error("image larger than 512 x 512");
            h_desc.p[k] = ImageDesc{uint32_t(off), uint32_t(r.w) * 3u, uint32_t(r.bx) * 3u, r.bw, r.bh};
            off += size_t(r.bh) * r.w * 3;
        }
#pragma omp parallel for schedule(static)
        for (int k = 0; k < count; ++k) {
            const Record& r = ds[idx[first + size_t(k)]];
            std::memcpy(h_src.p + h_desc.p[k].src, ds.pixels(r) + size_t(r.by) * r.w * 3,
                        size_t(r.bh) * r.w * 3);
        }
        b.stage += now() - t;
        t = now();
        check(cudaMemcpy(d_src.p, h_src.p, off, cudaMemcpyHostToDevice), "upload pixels");
        check(cudaMemcpy(d_desc.p, h_desc.p, size_t(count) * sizeof(ImageDesc), cudaMemcpyHostToDevice),
              "upload descriptors");
        b.upload += now() - t;
    };
    auto run_resize = [&](int count) {
        const dim3 block(16, 16), grid((kOut + 15) / 16, (kOut + 15) / 16, unsigned(count));
        resize_kernel<<<grid, block>>>(d_src.p, d_desc.p, d_resized.p, count);
    };

    // ---- pass 1: statistics over the train split -------------------------------------
    std::vector<size_t> train, all(ds.size());
    for (size_t i = 0; i < ds.size(); ++i) {
        all[i] = i;
        if (ds[i].is_train) train.push_back(i);
    }
    Breakdown b1, b2;
    for (size_t first = 0; first < train.size(); first += size_t(batch)) {
        const int count = int(std::min(size_t(batch), train.size() - first));
        stage_and_upload(train, first, count, b1);
        double t = now();
        run_resize(count);
        sums_kernel<<<unsigned(count), 256>>>(d_resized.p, d_sums.p);
        check(cudaDeviceSynchronize(), "pass 1 kernels");
        b1.kernel += now() - t;
        t = now();
        check(cudaMemcpy(h_sums.p, d_sums.p, size_t(count) * 6 * sizeof(uint32_t), cudaMemcpyDeviceToHost),
              "download sums");
        b1.download += now() - t;
        t = now();
        for (int k = 0; k < count; ++k) {
            out.stats.n += kOutPixels;
            for (int c = 0; c < 3; ++c) {
                out.stats.s1[c] += h_sums.p[k * 6 + c];
                out.stats.s2[c] += h_sums.p[k * 6 + 3 + c];
            }
        }
        b1.host += now() - t;
    }
    out.t.pass1 = sw.lap();

    out.norm = finalize(out.stats);
    check(cudaMemcpy(d_lut.p, out.norm.lut, 768 * sizeof(float), cudaMemcpyHostToDevice), "upload LUT");
    out.t.merge = sw.lap();

    // ---- pass 2: normalize every record ----------------------------------------------
    for (size_t first = 0; first < all.size(); first += size_t(batch)) {
        const int count = int(std::min(size_t(batch), all.size() - first));
        stage_and_upload(all, first, count, b2);
        double t = now();
        run_resize(count);
        normalize_kernel<<<dim3((unsigned(kOutPixels) + 255) / 256, unsigned(count)), 256>>>(
            d_resized.p, d_lut.p, d_tensor.p, count);
        check(cudaDeviceSynchronize(), "pass 2 kernels");
        b2.kernel += now() - t;
        t = now();
        check(cudaMemcpy(h_resized.p, d_resized.p, size_t(count) * kResizedBytes, cudaMemcpyDeviceToHost),
              "download resized");
        check(cudaMemcpy(h_tensor.p, d_tensor.p, size_t(count) * kNormalizedBytes, cudaMemcpyDeviceToHost),
              "download tensors");
        b2.download += now() - t;
        t = now();
#pragma omp parallel for schedule(static)
        for (int k = 0; k < count; ++k) {
            const size_t i = all[first + size_t(k)];
            out.crc_resized[i] = crc32(h_resized.p + size_t(k) * kResizedBytes, kResizedBytes);
            out.crc_normalized[i] = crc32(h_tensor.p + size_t(k) * kOutPixels * 3, kNormalizedBytes);
        }
        b2.host += now() - t;
    }
    out.t.pass2 = sw.lap();

    finish_run(opt, ds, out, 1, threads, "gpu-batch" + std::to_string(batch), sw);
    const auto path = std::filesystem::path(opt.out_dir) / "gpu_breakdown.csv";
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + path.string());
    std::fputs("pass,images,stage_s,upload_s,kernel_s,download_s,host_s\n", f);
    std::fprintf(f, "1,%zu,%.6f,%.6f,%.6f,%.6f,%.6f\n", train.size(), b1.stage, b1.upload, b1.kernel,
                 b1.download, b1.host);
    std::fprintf(f, "2,%zu,%.6f,%.6f,%.6f,%.6f,%.6f\n", all.size(), b2.stage, b2.upload, b2.kernel,
                 b2.download, b2.host);
    std::fclose(f);
    if (!opt.quiet) {
        std::printf("  gpu: %s, batch %d, %d host threads\n", prop.name, batch, threads);
        std::printf("  pass 1: stage %.3f  upload %.3f  kernels %.3f  download %.3f  host merge %.3f s\n",
                    b1.stage, b1.upload, b1.kernel, b1.download, b1.host);
        std::printf("  pass 2: stage %.3f  upload %.3f  kernels %.3f  download %.3f  host CRC %.3f s\n",
                    b2.stage, b2.upload, b2.kernel, b2.download, b2.host);
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "cuda: %s\n", e.what());
    return 1;
}
