#include "cub_format.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace cub {

namespace {

std::runtime_error sys_error(const std::string& what) {
    return std::runtime_error(what + ": " + std::strerror(errno));
}

std::string shard_name(unsigned k) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "shard_%04u.bin", k);
    return buf;
}

}  // namespace

Dataset::Dataset(const std::filesystem::path& dir) : dir_(dir) {
    const auto index_path = dir / "index.bin";
    std::FILE* f = std::fopen(index_path.c_str(), "rb");
    if (!f) throw sys_error("cannot open " + index_path.string());
    IndexHeader hdr{};
    if (std::fread(&hdr, sizeof hdr, 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error(index_path.string() + ": truncated header");
    }
    if (std::memcmp(hdr.magic, "CUBIDX01", 8) != 0 || hdr.version != 1 ||
        hdr.record_size != sizeof(Record)) {
        std::fclose(f);
        throw std::runtime_error(index_path.string() + ": not a CUB index v1");
    }
    records_.resize(hdr.count);
    if (std::fread(records_.data(), sizeof(Record), hdr.count, f) != hdr.count) {
        std::fclose(f);
        throw std::runtime_error(index_path.string() + ": truncated records");
    }
    std::fclose(f);

    for (unsigned k = 0; k < hdr.num_shards; ++k) {
        const auto path = dir / shard_name(k);
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw sys_error("cannot open " + path.string());
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            throw sys_error("fstat " + path.string());
        }
        Mapping m;
        m.len = static_cast<size_t>(st.st_size);
        m.addr = ::mmap(nullptr, m.len, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (m.addr == MAP_FAILED) throw sys_error("mmap " + path.string());
        shards_.push_back(m);
    }
    for (const auto& r : records_) {
        if (r.shard >= shards_.size() ||
            r.offset + uint64_t(r.w) * r.h * 3 > shards_[r.shard].len ||
            r.bx + r.bw > r.w || r.by + r.bh > r.h || r.bw == 0 || r.bh == 0)
            throw std::runtime_error(dir_.string() + ": invalid record for image_id " +
                                     std::to_string(r.image_id));
    }
}

Dataset::~Dataset() {
    for (auto& m : shards_) ::munmap(m.addr, m.len);
}

const uint8_t* Dataset::pixels(const Record& r) const {
    return static_cast<const uint8_t*>(shards_[r.shard].addr) + r.offset;
}

uint64_t Dataset::pixel_bytes() const {
    uint64_t total = 0;
    for (const auto& r : records_) total += uint64_t(r.w) * r.h * 3;
    return total;
}

void Dataset::preload() const {
    const long page = ::sysconf(_SC_PAGESIZE);
    volatile uint8_t sink = 0;
    for (const auto& m : shards_) {
        ::madvise(m.addr, m.len, MADV_WILLNEED);
        const auto* p = static_cast<const uint8_t*>(m.addr);
        for (size_t off = 0; off < m.len; off += page) sink = sink ^ p[off];
    }
    (void)sink;
}

}  // namespace cub
