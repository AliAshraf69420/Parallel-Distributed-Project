// Query service in front of the pipeline's results (bonus: client-server / RPC).
//
//   rpc_server --set SET [--results DIR] [--port 4010] [--delay-ms N] [--hang]
//
//   --results DIR  directory holding job1.tsv (Job 1 output: global and per-class sums) and
//                  checksums.csv; default results/runs/hadoop/SET, i.e. it serves what the
//                  MapReduce pipeline produced
//   --delay-ms N   sleep N ms before every reply (a slow server)
//   --hang         accept connections and read requests but never reply (a hung server)
//
// One thread per connection; the loaded results are immutable, so no locking is needed.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

#include "../common/cub_format.hpp"
#include "../common/kernels.hpp"
#include "../common/textio.hpp"
#include "protocol.hpp"

namespace {

using namespace cub;

struct ImageInfo {
    unsigned class_id = 0;
    bool train = false;
    std::string crc_resized, crc_normalized;
};

struct Results {
    Stats all;
    std::map<int, Stats> per_class;
    std::unordered_map<unsigned, ImageInfo> images;
};

Results load(const std::filesystem::path& set_dir, const std::filesystem::path& dir) {
    Results r;
    std::ifstream f(dir / "job1.tsv");
    if (!f) throw std::runtime_error("cannot open " + (dir / "job1.tsv").string());
    std::string line;
    while (std::getline(f, line)) {
        const size_t tab = line.find('\t'), colon = line.find(':');
        if (tab == std::string::npos || colon == std::string::npos) continue;
        const int c = line[colon + 1] - '0';
        if (c < 0 || c > 2) continue;
        const ChannelSums v = parse_sums(std::string_view(line).substr(tab + 1));
        Stats& st = line.rfind("all:", 0) == 0 ? r.all : r.per_class[std::stoi(line.substr(3, colon - 3))];
        st.n = v.n, st.s1[c] = v.s1, st.s2[c] = v.s2;
    }
    if (r.all.n == 0) throw std::runtime_error("job1.tsv has no global statistics");
    for (const Record& rec : Dataset::read_index(set_dir)) {
        ImageInfo& info = r.images[rec.image_id];
        info.class_id = rec.class_id, info.train = rec.is_train;
    }
    std::ifstream g(dir / "checksums.csv");
    std::getline(g, line);  // header
    while (std::getline(g, line)) {
        std::istringstream ss(line);
        std::string id, a, b;
        if (std::getline(ss, id, ',') && std::getline(ss, a, ',') && std::getline(ss, b)) {
            const auto it = r.images.find(unsigned(std::stoul(id)));
            if (it != r.images.end()) it->second.crc_resized = a, it->second.crc_normalized = b;
        }
    }
    return r;
}

std::string describe(const Stats& st) {
    const Normalization nz = finalize(st);
    char buf[256];
    std::snprintf(buf, sizeof buf, "n=%llu mean=%.6f,%.6f,%.6f std=%.6f,%.6f,%.6f",
                  static_cast<unsigned long long>(st.n), nz.mean[0], nz.mean[1], nz.mean[2],
                  nz.std[0], nz.std[1], nz.std[2]);
    return buf;
}

std::string answer(const Results& r, const std::string& request) {
    std::istringstream ss(request);
    std::string op;
    ss >> op;
    if (op == "PING") return "OK pong";
    if (op == "STATS") return "OK " + describe(r.all);
    if (op == "CLASS") {
        int k = -1;
        if (!(ss >> k)) return "ERR usage: CLASS <class_id>";
        const auto it = r.per_class.find(k);
        if (it == r.per_class.end()) return "ERR no training images of class " + std::to_string(k);
        return "OK class=" + std::to_string(k) + " " + describe(it->second);
    }
    if (op == "IMAGE") {
        unsigned id = 0;
        if (!(ss >> id)) return "ERR usage: IMAGE <image_id>";
        const auto it = r.images.find(id);
        if (it == r.images.end()) return "ERR no image " + std::to_string(id) + " in this set";
        const ImageInfo& i = it->second;
        return "OK image=" + std::to_string(id) + " class=" + std::to_string(i.class_id) +
               " split=" + (i.train ? "train" : "test") + " crc_resized=" + i.crc_resized +
               " crc_normalized=" + i.crc_normalized;
    }
    return "ERR unknown request (PING | STATS | CLASS <k> | IMAGE <id>)";
}

void serve(int fd, const Results& r, int delay_ms, bool hang) {
    std::string buf;
    char chunk[256];
    for (;;) {
        const ssize_t n = ::read(fd, chunk, sizeof chunk);
        if (n <= 0) break;
        buf.append(chunk, size_t(n));
        if (buf.size() > rpc::kMaxLine) break;                  // not a valid request
        size_t nl;
        while ((nl = buf.find('\n')) != std::string::npos) {
            std::string request = buf.substr(0, nl);
            buf.erase(0, nl + 1);
            if (!request.empty() && request.back() == '\r') request.pop_back();
            if (hang) continue;                                 // swallow the request
            if (delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            const std::string reply = answer(r, request) + "\n";
            if (::write(fd, reply.data(), reply.size()) < 0) break;
        }
    }
    ::close(fd);
}

}  // namespace

int main(int argc, char** argv) try {
    std::string set, results;
    int port = rpc::kDefaultPort, delay_ms = 0;
    bool hang = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&] {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return std::string(argv[++i]);
        };
        if (a == "--set") set = val();
        else if (a == "--results") results = val();
        else if (a == "--port") port = std::stoi(val());
        else if (a == "--delay-ms") delay_ms = std::stoi(val());
        else if (a == "--hang") hang = true;
        else throw std::runtime_error("unknown option " + a);
    }
    if (set.empty()) throw std::runtime_error("usage: rpc_server --set SET [--results DIR] [--port P] [--delay-ms N] [--hang]");
    namespace fs = std::filesystem;
    const fs::path set_dir = fs::exists(fs::path(set) / "index.bin") ? fs::path(set) : fs::path("data/processed") / set;
    if (results.empty()) results = (fs::path("results/runs/hadoop") / set_dir.filename()).string();
    const Results r = load(set_dir, results);

    std::signal(SIGPIPE, SIG_IGN);
    const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    ::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);              // local queries only
    addr.sin_port = htons(uint16_t(port));
    if (::bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 || ::listen(listener, 64) != 0)
        throw std::runtime_error(std::string("cannot listen on port ") + std::to_string(port) + ": " + std::strerror(errno));
    std::printf("rpc_server: %zu images, %zu classes from %s; listening on 127.0.0.1:%d%s\n",
                r.images.size(), r.per_class.size(), results.c_str(), port,
                hang ? " (hang mode)" : delay_ms ? " (delayed replies)" : "");
    std::fflush(stdout);
    for (;;) {
        const int fd = ::accept(listener, nullptr, nullptr);
        if (fd < 0) continue;
        std::thread(serve, fd, std::cref(r), delay_ms, hang).detach();
    }
} catch (const std::exception& e) {
    std::fprintf(stderr, "rpc_server: %s\n", e.what());
    return 1;
}
