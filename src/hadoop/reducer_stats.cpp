// Job 1 reducer AND combiner (Hadoop Streaming): sums the values of each key.
//
// stdin : key \t n,S1,S2   sorted by key (Hadoop's shuffle, or `sort` locally)
// stdout: key \t n,S1,S2   one line per key, with the three fields summed
//
// The sums are exact integers, so addition is associative and commutative: running this
// program as a combiner on any subset of a key's values, any number of times, cannot change
// the final result. That is what makes it a valid combiner.
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>

#include "../common/textio.hpp"

int main() try {
    using namespace cub;
    std::ios::sync_with_stdio(false);
    std::string line, current;
    ChannelSums acc;
    bool have = false;
    unsigned long long in = 0, out = 0;
    auto flush = [&] {
        if (!have) return;
        std::printf("%s\t%llu,%llu,%llu\n", current.c_str(), static_cast<unsigned long long>(acc.n),
                    static_cast<unsigned long long>(acc.s1), static_cast<unsigned long long>(acc.s2));
        ++out;
    };
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) throw std::runtime_error("line without a tab: " + line);
        const std::string_view key(line.data(), tab);
        const ChannelSums v = parse_sums(std::string_view(line).substr(tab + 1));
        ++in;
        if (!have || key != current) {
            flush();
            current.assign(key);
            acc = ChannelSums{};
            have = true;
        }
        acc.n += v.n, acc.s1 += v.s1, acc.s2 += v.s2;
    }
    flush();
    std::fprintf(stderr, "reporter:counter:CUB,sum_records_in,%llu\n", in);
    std::fprintf(stderr, "reporter:counter:CUB,sum_records_out,%llu\n", out);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "reducer_stats: %s\n", e.what());
    return 1;
}
