// Turns the outputs of the two Hadoop jobs into the same result files as every other
// implementation, so tests/check.py can compare Hadoop with the reference bit for bit.
//
//   hadoop_finalize --set SET --stats FILE [--crcs FILE] [--out DIR]
//
//   --stats FILE  Job 1 output (all part-* files concatenated): key \t n,S1,S2
//   --crcs FILE   Job 2 output (concatenated): image_id \t crc_resized,crc_normalized
//                 Without it, checksums.csv is not written (statistics only).
// Writes <out>/stats.json, <out>/checksums.csv and <out>/per_class.csv
// (class_id,channel,n,S1,S2), default out = results/runs/hadoop/SET.
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "../common/app.hpp"
#include "../common/textio.hpp"

int main(int argc, char** argv) try {
    using namespace cub;
    std::string set, stats_path, crcs_path, out_dir;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--set") set = argv[i + 1];
        else if (a == "--stats") stats_path = argv[i + 1];
        else if (a == "--crcs") crcs_path = argv[i + 1];
        else if (a == "--out") out_dir = argv[i + 1];
        else throw std::runtime_error("unknown option " + a);
    }
    if (set.empty() || stats_path.empty())
        throw std::runtime_error("usage: hadoop_finalize --set SET --stats FILE [--crcs FILE] [--out DIR]");
    namespace fs = std::filesystem;
    const fs::path set_dir = fs::exists(fs::path(set) / "index.bin") ? fs::path(set) : fs::path("data/processed") / set;
    const std::string name = set_dir.filename().string();
    if (out_dir.empty()) out_dir = (fs::path("results/runs/hadoop") / name).string();
    fs::create_directories(out_dir);
    const std::vector<Record> records = Dataset::read_index(set_dir);

    // ---- Job 1: statistics -----------------------------------------------------------
    RunOutput out;
    std::map<std::pair<int, int>, ChannelSums> per_class;   // (class, channel)
    int seen = 0;
    {
        std::ifstream f(stats_path);
        if (!f) throw std::runtime_error("cannot open " + stats_path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            const size_t tab = line.find('\t'), colon = line.find(':');
            if (tab == std::string::npos || colon == std::string::npos || colon > tab)
                throw std::runtime_error("bad stats line: " + line);
            const int c = line[colon + 1] - '0';
            if (c < 0 || c > 2) throw std::runtime_error("bad channel in: " + line);
            const ChannelSums v = parse_sums(std::string_view(line).substr(tab + 1));
            if (line.rfind("all:", 0) == 0) {
                if (seen & (1 << c)) throw std::runtime_error("duplicate key in stats: " + line);
                out.stats.n = v.n, out.stats.s1[c] = v.s1, out.stats.s2[c] = v.s2;
                seen |= 1 << c;
            } else if (line.rfind("cls", 0) == 0) {
                const int k = std::stoi(line.substr(3, colon - 3));
                if (!per_class.emplace(std::make_pair(k, c), v).second)
                    throw std::runtime_error("duplicate key in stats: " + line);
            }
        }
    }
    if (seen != 7) throw std::runtime_error("stats output lacks all:0, all:1 or all:2");
    out.norm = finalize(out.stats);
    {
        std::FILE* f = std::fopen((fs::path(out_dir) / "per_class.csv").c_str(), "w");
        if (!f) throw std::runtime_error("cannot write per_class.csv");
        std::fputs("class_id,channel,n,S1,S2\n", f);
        for (const auto& [key, v] : per_class)
            std::fprintf(f, "%d,%d,%llu,%llu,%llu\n", key.first, key.second,
                         static_cast<unsigned long long>(v.n), static_cast<unsigned long long>(v.s1),
                         static_cast<unsigned long long>(v.s2));
        std::fclose(f);
    }

    // ---- Job 2: per-image CRCs -------------------------------------------------------
    out.crc_resized.assign(records.size(), 0);
    out.crc_normalized.assign(records.size(), 0);
    size_t matched = 0;
    if (!crcs_path.empty()) {
        std::unordered_map<uint32_t, size_t> index_of;
        for (size_t i = 0; i < records.size(); ++i) index_of[records[i].image_id] = i;
        std::vector<bool> got(records.size(), false);
        std::ifstream f(crcs_path);
        if (!f) throw std::runtime_error("cannot open " + crcs_path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            unsigned id = 0, cr = 0, cn = 0;
            if (std::sscanf(line.c_str(), "%u\t%8x,%8x", &id, &cr, &cn) != 3)
                throw std::runtime_error("bad crc line: " + line.substr(0, 60));
            const auto it = index_of.find(id);
            if (it == index_of.end()) throw std::runtime_error("unknown image_id " + std::to_string(id));
            if (got[it->second]) throw std::runtime_error("image_id " + std::to_string(id) + " appears twice");
            got[it->second] = true;
            out.crc_resized[it->second] = cr, out.crc_normalized[it->second] = cn;
            ++matched;
        }
        if (matched != records.size())
            throw std::runtime_error("job 2 output has " + std::to_string(matched) + " images, expected " +
                                     std::to_string(records.size()));
    }

    Options opt;
    opt.version = "hadoop";
    opt.out_dir = out_dir;
    opt.quiet = true;
    Stopwatch sw;
    finish_run(opt, name, records, out, 1, 1, "mapreduce", sw);
    if (crcs_path.empty()) fs::remove(fs::path(out_dir) / "checksums.csv");
    std::printf("hadoop %s: %zu keys per class file, %zu image CRCs | mean %.6f %.6f %.6f -> %s\n",
                name.c_str(), per_class.size(), matched, out.norm.mean[0], out.norm.mean[1],
                out.norm.mean[2], out_dir.c_str());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "hadoop_finalize: %s\n", e.what());
    return 1;
}
