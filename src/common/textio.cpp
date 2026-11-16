#include "textio.hpp"

#include <array>
#include <charconv>
#include <stdexcept>

namespace cub {

namespace {

constexpr std::array<int8_t, 256> make_b64_table() {
    std::array<int8_t, 256> t{};
    for (auto& v : t) v = -1;
    const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) t[static_cast<unsigned char>(alphabet[i])] = int8_t(i);
    return t;
}
constexpr auto kB64 = make_b64_table();

uint64_t to_u64(std::string_view s, const char* what) {
    uint64_t v = 0;
    const auto res = std::from_chars(s.data(), s.data() + s.size(), v);
    if (res.ec != std::errc{} || res.ptr != s.data() + s.size())
        throw std::runtime_error(std::string("bad ") + what + " field '" + std::string(s) + "'");
    return v;
}

// Splits off the text before the next `sep`; advances `rest` past it.
std::string_view next_field(std::string_view& rest, char sep, const char* what) {
    const size_t pos = rest.find(sep);
    if (pos == std::string_view::npos)
        throw std::runtime_error(std::string("line ends before the ") + what + " field");
    const std::string_view field = rest.substr(0, pos);
    rest.remove_prefix(pos + 1);
    return field;
}

}  // namespace

std::string_view parse_line_header(std::string_view line, Record& rec) {
    rec = Record{};
    rec.image_id = uint32_t(to_u64(next_field(line, '\t', "image_id"), "image_id"));
    rec.class_id = uint16_t(to_u64(next_field(line, '\t', "class_id"), "class_id"));
    rec.is_train = uint8_t(to_u64(next_field(line, '\t', "is_train"), "is_train"));
    rec.w = uint16_t(to_u64(next_field(line, '\t', "w"), "w"));
    rec.h = uint16_t(to_u64(next_field(line, '\t', "h"), "h"));
    rec.bx = uint16_t(to_u64(next_field(line, '\t', "bx"), "bx"));
    rec.by = uint16_t(to_u64(next_field(line, '\t', "by"), "by"));
    rec.bw = uint16_t(to_u64(next_field(line, '\t', "bw"), "bw"));
    rec.bh = uint16_t(to_u64(next_field(line, '\t', "bh"), "bh"));
    if (rec.bw == 0 || rec.bh == 0 || rec.bx + rec.bw > rec.w || rec.by + rec.bh > rec.h)
        throw std::runtime_error("bounding box outside the image for image_id " +
                                 std::to_string(rec.image_id));
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.remove_suffix(1);
    return line;
}

void base64_decode(std::string_view b64, std::vector<uint8_t>& out) {
    while (!b64.empty() && b64.back() == '=') b64.remove_suffix(1);
    const size_t full = b64.size() / 4, tail = b64.size() % 4;
    if (tail == 1) throw std::runtime_error("truncated base64");
    out.resize(full * 3 + (tail ? tail - 1 : 0));
    const auto* in = reinterpret_cast<const unsigned char*>(b64.data());
    uint8_t* o = out.data();
    int bad = 0;
    for (size_t i = 0; i < full; ++i, in += 4, o += 3) {
        const int a = kB64[in[0]], b = kB64[in[1]], c = kB64[in[2]], d = kB64[in[3]];
        bad |= a | b | c | d;                       // any -1 makes this negative
        const uint32_t v = uint32_t(a) << 18 | uint32_t(b) << 12 | uint32_t(c) << 6 | uint32_t(d);
        o[0] = uint8_t(v >> 16), o[1] = uint8_t(v >> 8), o[2] = uint8_t(v);
    }
    if (tail) {
        const int a = kB64[in[0]], b = kB64[in[1]], c = tail == 3 ? kB64[in[2]] : 0;
        bad |= a | b | c;
        const uint32_t v = uint32_t(a) << 18 | uint32_t(b) << 12 | uint32_t(c) << 6;
        o[0] = uint8_t(v >> 16);
        if (tail == 3) o[1] = uint8_t(v >> 8);
    }
    if (bad < 0) throw std::runtime_error("invalid base64 character");
}

void parse_line(std::string_view line, Record& rec, std::vector<uint8_t>& pixels) {
    const std::string_view b64 = parse_line_header(line, rec);
    base64_decode(b64, pixels);
    if (pixels.size() != size_t(rec.w) * rec.h * 3)
        throw std::runtime_error("image_id " + std::to_string(rec.image_id) + ": " +
                                 std::to_string(pixels.size()) + " pixel bytes, expected " +
                                 std::to_string(size_t(rec.w) * rec.h * 3));
}

ChannelSums parse_sums(std::string_view value) {
    while (!value.empty() && (value.back() == '\n' || value.back() == '\r' || value.back() == '\t'))
        value.remove_suffix(1);
    ChannelSums s;
    s.n = to_u64(next_field(value, ',', "n"), "n");
    s.s1 = to_u64(next_field(value, ',', "S1"), "S1");
    s.s2 = to_u64(value, "S2");
    return s;
}

}  // namespace cub
