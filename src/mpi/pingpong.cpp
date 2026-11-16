// Ping-pong (Lab 7 Part A style): round-trip time against message size between rank 0 and
// rank 1, from 1 byte to 64 MiB. Small messages are latency-dominated (RTT is flat); large
// ones are bandwidth-dominated (RTT grows linearly with size).
//
//   mpirun -n 2 build/pingpong [--label NAME] [--max-bytes N] > pingpong.csv
//
// Output CSV: label,bytes,reps,rtt_min_us,rtt_median_us,mb_per_s (2 * bytes / median RTT).
// Run it once inside one host (shared-memory transport) and once across two nodes or
// containers (TCP) and compare.
#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        if (rank == 0) std::fprintf(stderr, "pingpong: run with exactly 2 ranks\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    std::string label = "run";
    size_t max_bytes = size_t(64) << 20;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--label")) label = argv[i + 1];
        else if (!std::strcmp(argv[i], "--max-bytes")) max_bytes = std::strtoull(argv[i + 1], nullptr, 10);
    }
    char host[MPI_MAX_PROCESSOR_NAME];
    int len = 0;
    MPI_Get_processor_name(host, &len);
    std::fprintf(stderr, "pingpong: rank %d on %s\n", rank, host);

    std::vector<char> buf(max_bytes, char(rank + 1));
    const int peer = 1 - rank;
    if (rank == 0) std::printf("label,bytes,reps,rtt_min_us,rtt_median_us,mb_per_s\n");
    for (size_t bytes = 1; bytes <= max_bytes; bytes *= 2) {
        // More repetitions for small messages; at least 10, at most 2000.
        const int reps = int(std::clamp<size_t>((size_t(32) << 20) / bytes, 10, 2000));
        const int warmup = std::max(3, reps / 10);
        std::vector<double> rtt;
        rtt.reserve(size_t(reps));
        for (int k = 0; k < warmup + reps; ++k) {
            if (rank == 0) {
                const double t0 = MPI_Wtime();
                MPI_Send(buf.data(), int(bytes), MPI_BYTE, peer, 0, MPI_COMM_WORLD);
                MPI_Recv(buf.data(), int(bytes), MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                const double dt = MPI_Wtime() - t0;
                if (k >= warmup) rtt.push_back(dt);
            } else {
                MPI_Recv(buf.data(), int(bytes), MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf.data(), int(bytes), MPI_BYTE, peer, 0, MPI_COMM_WORLD);
            }
        }
        if (rank == 0) {
            std::sort(rtt.begin(), rtt.end());
            const double med = rtt[rtt.size() / 2];
            std::printf("%s,%zu,%d,%.3f,%.3f,%.2f\n", label.c_str(), bytes, reps, rtt.front() * 1e6,
                        med * 1e6, 2.0 * double(bytes) / med / 1e6);
            std::fflush(stdout);
        }
    }
    MPI_Finalize();
    return 0;
}
