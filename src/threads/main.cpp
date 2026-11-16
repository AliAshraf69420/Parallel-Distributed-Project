// C++ threads version: a producer-consumer pipeline built from std::thread, std::mutex,
// std::condition_variable and std::atomic (no OpenMP).
//
//   [reader] --indices--> BoundedQueue --> [N workers] --tensors--> BoundedQueue --> [writer]
//
// * reader:  walks the records in processing order, asks the kernel to prefetch each
//            image's pixels (madvise WILLNEED: overlaps disk I/O with compute when the set
//            is not in the page cache) and pushes the record index.
// * workers: pop indices and do the per-record work (src/common/app.hpp). In pass 1 each
//            worker accumulates a private Stats; they are merged after join() (integer
//            addition, so the result does not depend on which worker did what).
// * writer:  only with --write; receives finished tensors and writes them with pwrite, so
//            workers never block on disk writes. Tensor buffers are recycled through a
//            free-list queue, which bounds memory.
// The queues are bounded, so a fast stage cannot run arbitrarily far ahead (backpressure).
// Workers are pinned one per physical core first (like OMP_PLACES=cores), then SMT siblings.
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../common/app.hpp"

namespace {

using namespace cub;

template <class T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity) : cap_(capacity) {}

    void push(T v) {
        std::unique_lock lk(m_);
        not_full_.wait(lk, [&] { return q_.size() < cap_; });
        q_.push_back(std::move(v));
        not_empty_.notify_one();
    }
    // Blocks until an item is available; nullopt once the queue is closed and drained.
    std::optional<T> pop() {
        std::unique_lock lk(m_);
        not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return std::nullopt;
        T v = std::move(q_.front());
        q_.pop_front();
        not_full_.notify_one();
        return v;
    }
    void close() {
        std::lock_guard lk(m_);
        closed_ = true;
        not_empty_.notify_all();
    }

private:
    std::mutex m_;
    std::condition_variable not_empty_, not_full_;
    std::deque<T> q_;
    size_t cap_;
    bool closed_ = false;
};

// Logical CPUs ordered one per physical core first, then the SMT siblings.
std::vector<int> cpu_order() {
    std::map<std::pair<int, int>, std::vector<int>> by_core;  // (package, core) -> cpus
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        std::ifstream core(base + "core_id"), pkg(base + "physical_package_id");
        int c, p;
        if (core >> c && pkg >> p) by_core[{p, c}].push_back(cpu);  // offline/absent CPUs skipped
    }
    std::vector<int> order;
    for (size_t round = 0;; ++round) {
        bool any = false;
        for (auto& [_, cpus] : by_core)
            if (round < cpus.size()) order.push_back(cpus[round]), any = true;
        if (!any) break;
    }
    return order;
}

void pin(std::thread& t, int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(t.native_handle(), sizeof set, &set);
}

// Runs `n` workers (pinned) plus a reader feeding them `indices`; rethrows the first error.
template <class Work>
void run_pipeline(const Dataset& ds, const std::vector<size_t>& indices, int n,
                  const std::vector<int>& cpus, Work work) {
    BoundedQueue<size_t> q(size_t(4 * n));
    std::exception_ptr err;
    std::mutex err_m;
    auto guard = [&](auto&& fn) {
        try {
            fn();
        } catch (...) {
            std::lock_guard lk(err_m);
            if (!err) err = std::current_exception();
        }
    };
    std::thread reader([&] {
        guard([&] {
            const long page = sysconf(_SC_PAGESIZE);
            for (size_t i : indices) {
                const Record& r = ds[i];
                auto a = reinterpret_cast<uintptr_t>(ds.pixels(r)) & ~uintptr_t(page - 1);
                const size_t len = size_t(r.w) * r.h * 3 + (reinterpret_cast<uintptr_t>(ds.pixels(r)) - a);
                madvise(reinterpret_cast<void*>(a), len, MADV_WILLNEED);
                q.push(i);
            }
        });
        q.close();
    });
    std::vector<std::thread> workers;
    for (int w = 0; w < n; ++w) {
        workers.emplace_back([&, w] { guard([&] { work(w, q); }); });
        if (!cpus.empty()) pin(workers.back(), cpus[size_t(w) % cpus.size()]);
    }
    reader.join();
    for (auto& t : workers) t.join();
    if (err) std::rethrow_exception(err);
}

}  // namespace

int main(int argc, char** argv) try {
    const Options opt = parse_args(argc, argv, "threads");
    const int n = opt.threads > 0 ? opt.threads : int(std::thread::hardware_concurrency());
    const std::vector<int> cpus = cpu_order();
    Stopwatch sw;

    Dataset ds(opt.set_dir);
    if (opt.preload) ds.preload();
    RunOutput out;
    out.crc_resized.resize(ds.size());
    out.crc_normalized.resize(ds.size());
    std::unique_ptr<TensorWriter> writer;
    if (!opt.write_path.empty()) writer = std::make_unique<TensorWriter>(opt.write_path, ds.size());
    std::vector<uint8_t> cache(opt.cache ? ds.size() * kResizedBytes : 0);
    const std::vector<size_t> order = record_order(ds, opt.order);
    const std::vector<size_t> p1 = pass1_indices(ds, order, opt.cache);
    std::atomic<size_t> done{0};
    out.t.load = sw.lap();

    // Pass 1: private Stats per worker, merged after join (no shared counters in the loop).
    // alignas(64) keeps each worker's Stats on its own cache line (no false sharing).
    struct alignas(64) Local { Stats st; };
    std::vector<Local> local(static_cast<size_t>(n));
    run_pipeline(ds, p1, n, cpus, [&](int w, BoundedQueue<size_t>& q) {
        std::vector<uint8_t> scratch(opt.cache ? 0 : kResizedBytes);
        size_t mine = 0;
        while (auto i = q.pop()) {
            pass1_record(ds, *i, opt.cache ? &cache[*i * kResizedBytes] : scratch.data(), local[w].st);
            ++mine;
        }
        done.fetch_add(mine, std::memory_order_relaxed);
    });
    for (const auto& l : local) out.stats.merge(l.st);
    if (done.load() != p1.size()) throw std::runtime_error("pass 1 lost records");
    out.t.pass1 = sw.lap();

    out.norm = finalize(out.stats);
    out.t.merge = sw.lap();

    // Pass 2: workers normalize; with --write a dedicated writer thread does the I/O.
    using Tensor = std::unique_ptr<float[]>;
    BoundedQueue<std::pair<size_t, Tensor>> to_write(size_t(2 * n));
    BoundedQueue<Tensor> free_buffers(size_t(3 * n));
    std::thread writer_thread;
    if (writer) {
        for (int k = 0; k < 3 * n; ++k) free_buffers.push(Tensor(new float[kOutPixels * 3]));
        writer_thread = std::thread([&] {
            while (auto item = to_write.pop()) {
                writer->put(item->first, item->second.get());
                free_buffers.push(std::move(item->second));
            }
        });
    }
    done = 0;
    try {
        run_pipeline(ds, order, n, cpus, [&](int, BoundedQueue<size_t>& q) {
            std::vector<uint8_t> scratch(opt.cache ? 0 : kResizedBytes);
            Tensor own(writer ? nullptr : new float[kOutPixels * 3]);
            size_t mine = 0;
            while (auto i = q.pop()) {
                uint8_t* buf = opt.cache ? &cache[*i * kResizedBytes] : scratch.data();
                if (writer) {
                    Tensor t = std::move(*free_buffers.pop());
                    pass2_record(ds, *i, out.norm, buf, opt.cache, t.get(), out, nullptr);
                    to_write.push({*i, std::move(t)});
                } else {
                    pass2_record(ds, *i, out.norm, buf, opt.cache, own.get(), out, nullptr);
                }
                ++mine;
            }
            done.fetch_add(mine, std::memory_order_relaxed);
        });
    } catch (...) {
        to_write.close();
        if (writer_thread.joinable()) writer_thread.join();
        throw;
    }
    to_write.close();
    if (writer_thread.joinable()) writer_thread.join();
    if (done.load() != order.size()) throw std::runtime_error("pass 2 lost records");
    out.t.pass2 = sw.lap();

    writer.reset();
    finish_run(opt, ds, out, 1, n, "pipeline", sw);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "threads: %s\n", e.what());
    return 1;
}
