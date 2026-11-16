// Text-line record format for Hadoop Streaming (written by preprocessing/to_text.py):
//
//   image_id \t class_id \t is_train \t w \t h \t bx \t by \t bw \t bh \t base64(pixels)
//
// plus the key/value format of the statistics job:
//
//   key \t n,S1,S2          key = "all:<channel>" or "cls<class, 3 digits>:<channel>"
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "cub_format.hpp"

namespace cub {

// Parses the nine metadata fields of a line into `rec` (offset/shard are left 0) and returns
// the base64 field. Throws std::runtime_error on a malformed line.
std::string_view parse_line_header(std::string_view line, Record& rec);

// Decodes standard base64 into `out` (resized to the decoded length). Throws on bad input.
void base64_decode(std::string_view b64, std::vector<uint8_t>& out);

// Full parse: header + pixels, checking that the pixel count matches w*h*3.
void parse_line(std::string_view line, Record& rec, std::vector<uint8_t>& pixels);

// "n,S1,S2" <-> three integers.
struct ChannelSums {
    uint64_t n = 0, s1 = 0, s2 = 0;
};
ChannelSums parse_sums(std::string_view value);

}  // namespace cub
