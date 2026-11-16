#include "cub_format.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

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

std::vector<Record> Dataset::read_index(const std::filesystem::path& dir) {
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
    std::vector<Record> records(hdr.count);
    if (std::fread(records.data(), sizeof(Record), hdr.count, f) != hdr.count) {
        std::fclose(f);
        throw std::runtime_error(index_path.string() + ": truncated records");
    }
    std::fclose(f);
    return records;
}

Dataset::Dataset(const std::filesystem::path& dir) : dir_(dir), records_(read_index(dir)) {
    map_shards();
}

Dataset::Dataset(const std::filesystem::path& dir, std::vector<Record> records)
    : dir_(dir), records_(std::move(records)) {
    map_shards();
}

void Dataset::map_shards() {
    unsigned num = 0;
    for (const auto& r : records_) num = std::max(num, unsigned(r.shard) + 1);
    std::vector<bool> used(num, false);
    for (const auto& r : records_) used[r.shard] = true;
    shards_.assign(num, Mapping{});
    for (unsigned k = 0; k < num; ++k) {
        if (!used[k]) continue;
        const auto path = dir_ / shard_name(k);
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
        shards_[k] = m;
    }
    for (const auto& r : records_) {
        if (r.offset + uint64_t(r.w) * r.h * 3 > shards_[r.shard].len ||
            r.bx + r.bw > r.w || r.by + r.bh > r.h || r.bw == 0 || r.bh == 0)
            throw std::runtime_error(dir_.string() + ": invalid record for image_id " +
                                     std::to_string(r.image_id));
    }
}

Dataset::~Dataset() {
    for (auto& m : shards_)
        if (m.addr) ::munmap(m.addr, m.len);
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
    const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    volatile uint8_t sink = 0;
    for (const auto& r : records_) {
        const uint8_t* p = pixels(r);
        const size_t len = size_t(r.w) * r.h * 3;
        for (size_t off = 0; off < len; off += page) sink = sink ^ p[off];
        sink = sink ^ p[len - 1];
    }
    (void)sink;
}

}  // namespace cub
