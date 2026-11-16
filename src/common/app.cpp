#include "app.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string_view>

namespace fs = std::filesystem;

namespace cub {

namespace {

[[noreturn]] void usage(const std::string& version, int code) {
    std::fprintf(code ? stderr : stdout,
        "usage: %s --set SET [options]\n"
        "  --set SET        set name (data/processed/SET) or path to a set directory\n"
        "  --out DIR        stats.json + checksums.csv go here (default results/runs/%s/SET)\n"
        "  --csv FILE       append a timing row to FILE (header written if new)\n"
        "  --write FILE     also write all normalized tensors to FILE (n x 602112 bytes)\n"
        "  -t, --threads N  threads (parallel versions; default: all cores)\n"
        "  --schedule S     OpenMP schedule: static[,k] | dynamic[,k] | guided[,k] | auto\n"
        "  --run K          run number recorded in the CSV (default 0)\n"
        "  --cache          resize every image once in pass 1 and keep it for pass 2\n"
        "  --preload        read all shards into the page cache before timing starts\n"
        "  -q, --quiet      no summary line\n",
        version.c_str(), version.c_str());
    std::exit(code);
}

std::string json_doubles(const double* v, int n) {
    std::string s = "[";
    char buf[40];
    for (int i = 0; i < n; ++i) {
        std::snprintf(buf, sizeof buf, "%s%.17g", i ? ", " : "", v[i]);
        s += buf;
    }
    return s + "]";
}

std::string json_u64(const uint64_t* v, int n) {
    std::string s = "[";
    for (int i = 0; i < n; ++i) s += (i ? ", " : "") + std::to_string(v[i]);
    return s + "]";
}

int phase_of(const std::string& v) {
    if (v == "mpi" || v == "hybrid" || v == "cuda") return 2;
    if (v == "hadoop") return 3;
    return 1;
}

}  // namespace

Options parse_args(int argc, char** argv, const std::string& version) {
    Options o;
    o.version = version;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s: %s needs a value\n", version.c_str(), argv[i]);
                usage(version, 2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") usage(version, 0);
        else if (a == "--set") o.set_dir = value();
        else if (a == "--out") o.out_dir = value();
        else if (a == "--csv") o.csv = value();
        else if (a == "--write") o.write_path = value();
        else if (a == "--schedule") o.schedule = value();
        else if (a == "-t" || a == "--threads") o.threads = std::atoi(value().c_str());
        else if (a == "--run") o.run = std::atoi(value().c_str());
        else if (a == "--cache") o.cache = true;
        else if (a == "--preload") o.preload = true;
        else if (a == "-q" || a == "--quiet") o.quiet = true;
        else {
            std::fprintf(stderr, "%s: unknown option %s\n", version.c_str(), argv[i]);
            usage(version, 2);
        }
    }
    if (o.set_dir.empty()) usage(version, 2);
    if (!fs::exists(fs::path(o.set_dir) / "index.bin")) {
        const fs::path by_name = fs::path("data/processed") / o.set_dir;
        if (o.set_dir.find('/') == std::string::npos && fs::exists(by_name / "index.bin")) {
            o.set_dir = by_name.string();
        } else {
            std::fprintf(stderr, "%s: no set at '%s' (run from the repo root, or pass a path)\n",
                         version.c_str(), o.set_dir.c_str());
            std::exit(2);
        }
    }
    if (o.out_dir.empty())
        o.out_dir = (fs::path("results/runs") / version / fs::path(o.set_dir).filename()).string();
    return o;
}

// ---------------------------------------------------------------- TensorWriter
TensorWriter::TensorWriter(const std::string& path, size_t count) {
    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd_ < 0) throw std::runtime_error("cannot create " + path + ": " + std::strerror(errno));
    if (::ftruncate(fd_, off_t(count * kNormalizedBytes)) != 0)
        throw std::runtime_error("ftruncate " + path + ": " + std::strerror(errno));
}

TensorWriter::~TensorWriter() {
    if (fd_ >= 0) ::close(fd_);
}

void TensorWriter::put(size_t i, const float* tensor) const {
    const auto* p = reinterpret_cast<const char*>(tensor);
    size_t left = kNormalizedBytes;
    off_t off = off_t(i * kNormalizedBytes);
    while (left) {
        const ssize_t w = ::pwrite(fd_, p, left, off);
        if (w < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("pwrite: ") + std::strerror(errno));
        }
        p += w, off += w, left -= size_t(w);
    }
}

// ---------------------------------------------------------------- passes
std::vector<size_t> pass1_indices(const Dataset& ds, bool cache) {
    std::vector<size_t> idx;
    idx.reserve(ds.size());
    for (size_t i = 0; i < ds.size(); ++i)
        if (cache || ds[i].is_train) idx.push_back(i);
    return idx;
}

// ---------------------------------------------------------------- outputs
void finish_run(const Options& opt, const Dataset& ds, RunOutput& out, int workers, int threads,
                const std::string& schedule, Stopwatch& sw) {
    fs::create_directories(opt.out_dir);
    const Normalization& nz = out.norm;
    size_t train = 0;
    for (const auto& r : ds.records()) train += r.is_train;

    {   // stats.json
        const auto path = fs::path(opt.out_dir) / "stats.json";
        std::FILE* f = std::fopen(path.c_str(), "w");
        if (!f) throw std::runtime_error("cannot write " + path.string());
        std::fprintf(f, "{\n  \"set\": \"%s\",\n  \"version\": \"%s\",\n", ds.name().c_str(),
                     opt.version.c_str());
        std::fprintf(f, "  \"records\": %zu,\n  \"train_records\": %zu,\n", ds.size(), train);
        std::fprintf(f, "  \"n\": %llu,\n", static_cast<unsigned long long>(out.stats.n));
        std::fprintf(f, "  \"S1\": %s,\n  \"S2\": %s,\n", json_u64(out.stats.s1, 3).c_str(),
                     json_u64(out.stats.s2, 3).c_str());
        std::fprintf(f, "  \"mean\": %s,\n  \"std\": %s,\n", json_doubles(nz.mean, 3).c_str(),
                     json_doubles(nz.std, 3).c_str());
        std::fprintf(f, "  \"lut_f32_hex\": [");
        for (int c = 0; c < 3; ++c) {
            std::fprintf(f, "%s\n    [", c ? "," : "");
            for (int v = 0; v < 256; ++v) {
                uint32_t bits;
                std::memcpy(&bits, &nz.lut[c][v], 4);
                std::fprintf(f, "%s\"%08x\"", v ? ", " : "", bits);
            }
            std::fprintf(f, "]");
        }
        std::fprintf(f, "\n  ]\n}\n");
        std::fclose(f);
    }
    {   // checksums.csv, byte-identical format to results/reference/<set>/checksums.csv
        const auto path = fs::path(opt.out_dir) / "checksums.csv";
        std::FILE* f = std::fopen(path.c_str(), "w");
        if (!f) throw std::runtime_error("cannot write " + path.string());
        std::fputs("image_id,crc_resized,crc_normalized\n", f);
        for (size_t i = 0; i < ds.size(); ++i)
            std::fprintf(f, "%u,%08x,%08x\n", ds[i].image_id, out.crc_resized[i],
                         out.crc_normalized[i]);
        std::fclose(f);
    }
    out.t.write = sw.lap();
    out.t.total = sw.total();

    if (!opt.csv.empty()) {
        const bool fresh = !fs::exists(opt.csv) || fs::file_size(opt.csv) == 0;
        if (fs::path(opt.csv).has_parent_path()) fs::create_directories(fs::path(opt.csv).parent_path());
        std::FILE* f = std::fopen(opt.csv.c_str(), "a");
        if (!f) throw std::runtime_error("cannot append to " + opt.csv);
        if (fresh)
            std::fputs("phase,version,dataset,workers,threads_per_worker,schedule,cache,preload,"
                       "run,t_load,t_pass1,t_merge,t_pass2,t_write,t_total,host\n", f);
        char host[256] = "unknown";
        ::gethostname(host, sizeof host - 1);
        std::fprintf(f, "%d,%s,%s,%d,%d,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s\n",
                     phase_of(opt.version), opt.version.c_str(), ds.name().c_str(), workers,
                     threads, schedule.c_str(), int(opt.cache), int(opt.preload), opt.run,
                     out.t.load, out.t.pass1, out.t.merge, out.t.pass2, out.t.write, out.t.total,
                     host);
        std::fclose(f);
    }
    if (!opt.quiet)
        std::printf("%s %s: %zu images (%zu train) | workers=%d threads=%d schedule=%s%s | "
                    "load %.3f  pass1 %.3f  merge %.6f  pass2 %.3f  write %.3f  total %.3f s"
                    " | mean %.6f %.6f %.6f -> %s\n",
                    opt.version.c_str(), ds.name().c_str(), ds.size(), train, workers, threads,
                    schedule.c_str(), opt.cache ? " cache" : "", out.t.load, out.t.pass1,
                    out.t.merge, out.t.pass2, out.t.write, out.t.total, nz.mean[0], nz.mean[1],
                    nz.mean[2], opt.out_dir.c_str());
}

}  // namespace cub
