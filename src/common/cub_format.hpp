// Binary set format (index.bin + shard_KKKK.bin), version 1.
// Must match preprocessing/cubfmt.py, which documents the layout.
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cub {

static_assert(std::endian::native == std::endian::little, "format is little-endian");

struct IndexHeader {
    char magic[8];  // "CUBIDX01"
    uint32_t version;
    uint32_t record_size;
    uint64_t count;
    uint32_t num_shards;
    uint32_t reserved;
};

struct Record {
    uint64_t offset;  // byte offset of the pixel block inside its shard
    uint32_t image_id;
    uint16_t class_id;
    uint8_t is_train;
    uint8_t flags;
    uint16_t w, h;
    uint16_t bx, by, bw, bh;
    uint16_t shard;
    uint16_t pad;
};

static_assert(sizeof(IndexHeader) == 32);
static_assert(sizeof(Record) == 32);
static_assert(offsetof(Record, image_id) == 8 && offsetof(Record, w) == 16 &&
              offsetof(Record, shard) == 28);

// A set on disk: the index in memory, the shards memory-mapped read-only.
class Dataset {
public:
    explicit Dataset(const std::filesystem::path& dir);
    ~Dataset();
    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;

    size_t size() const { return records_.size(); }
    const Record& operator[](size_t i) const { return records_[i]; }
    const std::vector<Record>& records() const { return records_; }
    const uint8_t* pixels(const Record& r) const;  // HWC RGB, r.h * r.w * 3 bytes
    const std::filesystem::path& dir() const { return dir_; }
    std::string name() const { return dir_.filename().string(); }
    uint64_t pixel_bytes() const;

    // Touch every shard page so later timings measure compute, not disk reads.
    void preload() const;

private:
    struct Mapping {
        void* addr = nullptr;
        size_t len = 0;
    };
    std::filesystem::path dir_;
    std::vector<Record> records_;
    std::vector<Mapping> shards_;
};

}  // namespace cub
