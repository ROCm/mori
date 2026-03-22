// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// umbp_bench: End-to-end benchmark using UMBPClient (full stack).
//
// Modes:
//   Throughput (default):
//     ./umbp_bench                                            # POSIX single
//     UMBP_SPDK_NVME_PCI=... ./umbp_bench --ranks=8          # SPDK multi
//
//   Latency (single-thread QD=1, per-op timing):
//     ./umbp_bench --latency                                  # POSIX
//     UMBP_SPDK_NVME_PCI=... ./umbp_bench --latency          # SPDK
//
// Cold reads use O_DIRECT (POSIX) or bypass ring cache (SPDK).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef __linux__
#include <sched.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "umbp/umbp_client.h"

static double NowSec() {
    auto tp = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(tp.time_since_epoch()).count();
}

static std::string MakeSessionId() {
    std::random_device rd;
    char buf[16];
    snprintf(buf, sizeof(buf), "%08x", rd());
    return buf;
}

struct SizeSpec { size_t size; int count; };

static const SizeSpec kSpecs[] = {
    {4 * 1024, 2000},
    {32 * 1024, 2000},
    {128 * 1024, 2000},
    {512 * 1024, 1024},
    {1024 * 1024, 512},
    {2ULL * 1024 * 1024, 256},
    {8ULL * 1024 * 1024, 64},
    {16ULL * 1024 * 1024, 32},
    {32ULL * 1024 * 1024, 16},
    {64ULL * 1024 * 1024, 8},
    {128ULL * 1024 * 1024, 4},
    {256ULL * 1024 * 1024, 2},
    {288ULL * 1024 * 1024, 2},
    {512ULL * 1024 * 1024, 2},
};

struct BenchResult {
    size_t value_size;
    int count;
    double write_mbps;
    double read_mbps;       // first-read (cold, real business scenario)
    double cache_read_mbps; // best of subsequent iterations (heap cache)
    int read_ok;
    int corrupt;
    double cold_t0, cold_t1;     // wall-clock timestamps for cold read
    double cache_t0, cache_t1;   // wall-clock timestamps for best cache read
};

// Coordination structure for multi-rank phased benchmark.
// Shared between forked processes via anonymous mmap.
struct alignas(64) BenchCoord {
    std::atomic<int> write_gen;       // Leader sets to size_idx+1 after writing
    std::atomic<int> reads_complete;  // All ranks increment after reading
    std::atomic<int> iter_barrier;    // Sync ranks between read iterations
    int num_ranks;
    char session[16];                 // Shared session ID
};

// ---------------------------------------------------------------------------
// Single-rank benchmark: same process writes and reads (original behavior)
// ---------------------------------------------------------------------------
static BenchResult RunBatch(UMBPClient& client, int rank_id,
                            const std::string& session,
                            size_t value_size, int count, int iterations) {
    std::string prefix = "bench_r" + std::to_string(rank_id) + "_" + session +
                         "_" + std::to_string(value_size) + "_";

    uint32_t seed = 0;
    for (char c : session) seed = seed * 31 + static_cast<uint8_t>(c);

    std::vector<std::vector<char>> datas(count);
    std::vector<size_t> sizes(count, value_size);
    for (int i = 0; i < count; ++i)
        datas[i].resize(value_size, static_cast<char>((seed + i + 1) & 0xFF));

    double total_bytes = static_cast<double>(value_size) * count;

    auto* ssd = client.Storage().GetTier(StorageTier::LOCAL_SSD);
    if (!ssd) return {value_size, count, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    std::vector<const void*> cptrs(count);
    for (int i = 0; i < count; ++i)
        cptrs[i] = datas[i].data();

    double best_write = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<std::string> wkeys(count);
        for (int i = 0; i < count; ++i)
            wkeys[i] = prefix + "w" + std::to_string(iter) + "_" + std::to_string(i);

        double t0 = NowSec();
        auto wr = ssd->BatchWrite(wkeys, cptrs, sizes);
        double t1 = NowSec();

        int ok = 0;
        for (auto b : wr) ok += b;
        double mbps = (total_bytes / (1024.0 * 1024.0)) / (t1 - t0);
        if (ok == count && mbps > best_write) best_write = mbps;
    }

    std::vector<std::string> rkeys(count);
    for (int i = 0; i < count; ++i)
        rkeys[i] = prefix + "r_" + std::to_string(i);

    int write_ok = 0;
    for (int attempt = 0; attempt < 3 && write_ok < count; ++attempt) {
        auto wr = ssd->BatchWrite(rkeys, cptrs, sizes);
        write_ok = 0;
        for (auto b : wr) write_ok += b;
        if (write_ok < count && attempt < 2)
            fprintf(stderr, "  [rank %d] read-key write %d/%d, retrying...\n",
                    rank_id, write_ok, count);
    }
    if (write_ok < count)
        fprintf(stderr, "  [rank %d] WARNING: only wrote %d/%d read-keys for %zuKB\n",
                rank_id, write_ok, count, value_size / 1024);

    std::vector<std::vector<char>> read_bufs(count, std::vector<char>(value_size, 0));
    std::vector<uintptr_t> dst_ptrs(count);
    for (int i = 0; i < count; ++i)
        dst_ptrs[i] = reinterpret_cast<uintptr_t>(read_bufs[i].data());

    double first_read = 0;
    double cache_read = 0;
    double cold_t0 = 0, cold_t1 = 0;
    double cache_last_t0 = 0, cache_last_t1 = 0;
    int best_read_ok = 0;
    int corrupt = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        ssd->SetColdRead(iter == 0);

        for (auto& b : read_bufs) std::memset(b.data(), 0, b.size());

        double t0 = NowSec();
        auto rr = ssd->BatchReadIntoPtr(rkeys, dst_ptrs, sizes);
        double t1 = NowSec();

        int ok = 0;
        for (auto b : rr) ok += b;
        double mbps = (total_bytes / (1024.0 * 1024.0)) / (t1 - t0);

        if (iter == 0) {
            if (ok > 0) { first_read = mbps; best_read_ok = ok; }
            cold_t0 = t0; cold_t1 = t1;
            for (int i = 0; i < count; ++i) {
                if (!rr[i]) continue;
                char expected = static_cast<char>((seed + i + 1) & 0xFF);
                const char* buf = read_bufs[i].data();
                if (buf[0] != expected ||
                    buf[value_size / 2] != expected ||
                    buf[value_size - 1] != expected) {
                    ++corrupt;
                    if (corrupt <= 3)
                        fprintf(stderr,
                            "  [rank %d] CORRUPT: key %d, expected 0x%02x, "
                            "got first=0x%02x last=0x%02x\n",
                            rank_id, i, (unsigned char)expected,
                            (unsigned char)buf[0],
                            (unsigned char)buf[value_size - 1]);
                }
            }
        } else {
            if (ok > 0) {
                cache_read = mbps;
                cache_last_t0 = t0; cache_last_t1 = t1;
            }
        }
    }
    ssd->SetColdRead(false);

    return {value_size, count, best_write, first_read, cache_read,
            best_read_ok, corrupt, cold_t0, cold_t1, cache_last_t0, cache_last_t1};
}

// ---------------------------------------------------------------------------
// Multi-rank phased benchmark (real Prefill/Decode scenario)
//   Phase 1: Leader writes data (measure write bandwidth)
//   Phase 2: All ranks concurrently read the SAME data (measure read bandwidth)
//   Phase 3: Barrier — sync before next size
// ---------------------------------------------------------------------------
#ifdef __linux__
static BenchResult RunBatchPhased(UMBPClient& client, int rank_id,
                                   size_t value_size, int count, int iterations,
                                   int size_idx, BenchCoord* coord) {
    std::string session(coord->session);
    std::string prefix = "bench_" + session + "_" + std::to_string(value_size) + "_";

    uint32_t seed = 0;
    for (char c : session) seed = seed * 31 + static_cast<uint8_t>(c);

    double total_bytes = static_cast<double>(value_size) * count;
    auto* ssd = client.Storage().GetTier(StorageTier::LOCAL_SSD);
    if (!ssd) return {value_size, count, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    double best_write = 0;

    // === WRITE PHASE (Leader only, others wait) ===
    if (rank_id == 0) {
        std::vector<std::vector<char>> datas(count);
        std::vector<size_t> sizes(count, value_size);
        for (int i = 0; i < count; ++i)
            datas[i].resize(value_size, static_cast<char>((seed + i + 1) & 0xFF));

        std::vector<const void*> cptrs(count);
        for (int i = 0; i < count; ++i) cptrs[i] = datas[i].data();

        for (int iter = 0; iter < iterations; ++iter) {
            std::vector<std::string> wkeys(count);
            for (int i = 0; i < count; ++i)
                wkeys[i] = prefix + "w" + std::to_string(iter) + "_" + std::to_string(i);

            double t0 = NowSec();
            auto wr = ssd->BatchWrite(wkeys, cptrs, sizes);
            double t1 = NowSec();

            int ok = 0;
            for (auto b : wr) ok += b;
            double mbps = (total_bytes / (1024.0 * 1024.0)) / (t1 - t0);
            if (ok == count && mbps > best_write) best_write = mbps;
        }

        // Write the read-benchmark keys (data all ranks will read)
        std::vector<std::string> rkeys(count);
        for (int i = 0; i < count; ++i)
            rkeys[i] = prefix + "r_" + std::to_string(i);

        int write_ok = 0;
        for (int attempt = 0; attempt < 3 && write_ok < count; ++attempt) {
            auto wr = ssd->BatchWrite(rkeys, cptrs, sizes);
            write_ok = 0;
            for (auto b : wr) write_ok += b;
        }
        if (write_ok < count)
            fprintf(stderr, "  [Leader] WARNING: wrote %d/%d read-keys for %zuKB\n",
                    write_ok, count, value_size / 1024);

        client.Flush();

        coord->write_gen.store(size_idx + 1, std::memory_order_release);
    }

    // === WAIT for Leader to finish writing ===
    while (coord->write_gen.load(std::memory_order_acquire) < size_idx + 1)
        sched_yield();

    // === READ PHASE (All ranks read the SAME data concurrently) ===
    std::vector<std::string> rkeys(count);
    for (int i = 0; i < count; ++i)
        rkeys[i] = prefix + "r_" + std::to_string(i);

    std::vector<std::vector<char>> read_bufs(count, std::vector<char>(value_size, 0));
    std::vector<uintptr_t> dst_ptrs(count);
    std::vector<size_t> sizes(count, value_size);
    for (int i = 0; i < count; ++i)
        dst_ptrs[i] = reinterpret_cast<uintptr_t>(read_bufs[i].data());

    double first_read = 0;   // iteration 0: cold read (O_DIRECT / bypass cache)
    double cache_read = 0;   // last cache iteration (steady-state)
    double cold_t0 = 0, cold_t1 = 0;
    double cache_last_t0 = 0, cache_last_t1 = 0;
    int best_read_ok = 0;
    int corrupt = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        ssd->SetColdRead(iter == 0);

        if (iter > 0) {
            coord->iter_barrier.fetch_add(1, std::memory_order_acq_rel);
            int sync = (size_idx * (iterations - 1) + iter) * coord->num_ranks;
            while (coord->iter_barrier.load(std::memory_order_acquire) < sync)
                sched_yield();
        }

        for (auto& b : read_bufs) std::memset(b.data(), 0, b.size());

        double t0 = NowSec();
        auto rr = ssd->BatchReadIntoPtr(rkeys, dst_ptrs, sizes);
        double t1 = NowSec();

        int ok = 0;
        for (auto b : rr) ok += b;
        double mbps = (total_bytes / (1024.0 * 1024.0)) / (t1 - t0);

        if (iter == 0) {
            if (ok > 0) { first_read = mbps; best_read_ok = ok; }
            cold_t0 = t0; cold_t1 = t1;
            for (int i = 0; i < count; ++i) {
                if (!rr[i]) continue;
                char expected = static_cast<char>((seed + i + 1) & 0xFF);
                const char* buf = read_bufs[i].data();
                if (buf[0] != expected || buf[value_size / 2] != expected ||
                    buf[value_size - 1] != expected) {
                    ++corrupt;
                    if (corrupt <= 3)
                        fprintf(stderr,
                            "  [rank %d] CORRUPT: key %d, expected 0x%02x, "
                            "got first=0x%02x last=0x%02x\n",
                            rank_id, i, (unsigned char)expected,
                            (unsigned char)buf[0],
                            (unsigned char)buf[value_size - 1]);
                }
            }
        } else {
            if (ok > 0) {
                cache_read = mbps;
                cache_last_t0 = t0; cache_last_t1 = t1;
            }
        }
    }

    ssd->SetColdRead(false);

    // === BARRIER: all ranks done reading this size ===
    coord->reads_complete.fetch_add(1, std::memory_order_acq_rel);
    int target = (size_idx + 1) * coord->num_ranks;
    while (coord->reads_complete.load(std::memory_order_acquire) < target)
        sched_yield();

    return {value_size, count, best_write, first_read, cache_read,
            best_read_ok, corrupt, cold_t0, cold_t1, cache_last_t0, cache_last_t1};
}
#endif

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------
static void PrintResults(int rank_id, int num_ranks, const char* role_str,
                         const char* backend,
                         const std::vector<BenchResult>& results) {
    printf("\n");
    printf("========================================================\n");
    printf(" UMBPClient E2E Benchmark — rank=%d/%d role=%s backend=%s\n",
           rank_id, num_ranks, role_str, backend);
    printf("========================================================\n");
    printf("  %8s  %6s  %10s  %10s  %12s\n",
           "ValSize", "Count", "Write MB/s", "ColdRd MB/s", "HotRd MB/s");

    for (const auto& r : results) {
        char sz_label[16];
        if (r.value_size >= 1024 * 1024)
            snprintf(sz_label, sizeof(sz_label), "%zuMB", r.value_size / (1024 * 1024));
        else
            snprintf(sz_label, sizeof(sz_label), "%zuKB", r.value_size / 1024);

        if (r.write_mbps > 0)
            printf("  %8s  %6d  %10.0f  %10.0f  %12.0f",
                   sz_label, r.count, r.write_mbps, r.read_mbps, r.cache_read_mbps);
        else
            printf("  %8s  %6d  %10s  %10.0f  %12.0f",
                   sz_label, r.count, "-", r.read_mbps, r.cache_read_mbps);
        if (r.read_ok != r.count)
            printf("  *** READ %d/%d", r.read_ok, r.count);
        if (r.corrupt > 0)
            printf("  *** CORRUPT %d", r.corrupt);
        printf("\n");
    }
    fflush(stdout);
}

#ifdef __linux__
static void WriteResultsToPipe(int fd, const std::vector<BenchResult>& results) {
    uint32_t n = static_cast<uint32_t>(results.size());
    (void)!write(fd, &n, sizeof(n));
    for (const auto& r : results)
        (void)!write(fd, &r, sizeof(r));
}

static std::vector<BenchResult> ReadResultsFromPipe(int fd) {
    uint32_t n = 0;
    if (read(fd, &n, sizeof(n)) != sizeof(n)) return {};
    std::vector<BenchResult> results(n);
    for (uint32_t i = 0; i < n; ++i)
        (void)!read(fd, &results[i], sizeof(BenchResult));
    return results;
}
#endif

static int RunBenchmarkProcess(int rank_id, int num_ranks, int pipe_fd,
                               void* coord_ptr) {
    auto cfg = UMBPConfig::FromEnvironment();
    cfg.dram_capacity_bytes = 64ULL * 1024 * 1024;

    UMBPClient client(cfg);

    UMBPRole role = cfg.ResolveRole();
    const char* role_str = (role == UMBPRole::Standalone) ? "Standalone" :
                           (role == UMBPRole::SharedSSDLeader) ? "Leader" : "Follower";
    const char* backend = cfg.ssd_backend.c_str();

    const int iterations = 2;  // 0=cold (O_DIRECT/bypass cache), 1=hot

    std::vector<BenchResult> results;

#ifdef __linux__
    auto* coord = static_cast<BenchCoord*>(coord_ptr);
    // SPDK Leader/Follower: phased mode (Leader writes → barrier → all read same data)
    // POSIX / Standalone:   per-rank mode (each rank writes and reads independently)
    bool use_phased = coord && (role == UMBPRole::SharedSSDLeader ||
                                role == UMBPRole::SharedSSDFollower);
    if (use_phased) {
        for (size_t s = 0; s < sizeof(kSpecs) / sizeof(kSpecs[0]); ++s)
            results.push_back(RunBatchPhased(client, rank_id, kSpecs[s].size,
                                              kSpecs[s].count, iterations,
                                              static_cast<int>(s), coord));
    } else
#endif
    {
        (void)coord_ptr;
        std::string session = MakeSessionId();
        for (const auto& s : kSpecs)
            results.push_back(RunBatch(client, rank_id, session,
                                       s.size, s.count, iterations));
    }

    if (pipe_fd >= 0) {
#ifdef __linux__
        WriteResultsToPipe(pipe_fd, results);
#endif
    } else {
        PrintResults(rank_id, num_ranks, role_str, backend, results);
    }

    (void)role_str;
    (void)backend;
    return 0;
}

// ---------------------------------------------------------------------------
// Latency benchmark: single-thread QD=1, per-op timing
// ---------------------------------------------------------------------------
static const SizeSpec kLatencySpecs[] = {
    {4 * 1024, 500},
    {32 * 1024, 500},
    {128 * 1024, 200},
    {512 * 1024, 100},
    {1024 * 1024, 50},
    {2ULL * 1024 * 1024, 50},
    {8ULL * 1024 * 1024, 20},
    {16ULL * 1024 * 1024, 10},
    {32ULL * 1024 * 1024, 8},
    {64ULL * 1024 * 1024, 4},
    {128ULL * 1024 * 1024, 4},
    {256ULL * 1024 * 1024, 2},
    {288ULL * 1024 * 1024, 2},
    {512ULL * 1024 * 1024, 2},
};

struct LatencyStats {
    double avg_us, p50_us, p99_us, max_us;
};

static LatencyStats ComputeStats(std::vector<double>& samples_us) {
    if (samples_us.empty()) return {0, 0, 0, 0};
    std::sort(samples_us.begin(), samples_us.end());
    size_t n = samples_us.size();
    double sum = 0;
    for (double v : samples_us) sum += v;
    return {
        sum / static_cast<double>(n),
        samples_us[n / 2],
        samples_us[std::min(n - 1, static_cast<size_t>(n * 0.99))],
        samples_us[n - 1]
    };
}

static int RunLatencyBench() {
    auto cfg = UMBPConfig::FromEnvironment();
    cfg.dram_capacity_bytes = 64ULL * 1024 * 1024;
    UMBPClient client(cfg);

    auto* ssd = client.Storage().GetTier(StorageTier::LOCAL_SSD);
    if (!ssd) {
        fprintf(stderr, "ERROR: no SSD tier available\n");
        return 1;
    }

    const char* backend = cfg.ssd_backend.c_str();
    std::string session = MakeSessionId();

    ssd->SetColdRead(true);

    printf("\n================================================================\n");
    printf(" WRITE LATENCY — backend=%s  (single-thread QD=1, O_DIRECT)\n", backend);
    printf("================================================================\n");
    printf("  %8s  %5s  %10s  %10s  %10s  %10s\n",
           "ValSize", "Count", "Avg(us)", "P50(us)", "P99(us)", "Max(us)");

    for (const auto& spec : kLatencySpecs) {
        size_t vsize = spec.size;
        int count = spec.count;

        char sz_label[16];
        if (vsize >= 1024 * 1024)
            snprintf(sz_label, sizeof(sz_label), "%zuMB", vsize / (1024 * 1024));
        else
            snprintf(sz_label, sizeof(sz_label), "%zuKB", vsize / 1024);

        std::string prefix = "lat_" + session + "_" + std::to_string(vsize) + "_";

        std::vector<char> data(vsize);
        for (size_t j = 0; j < vsize; ++j)
            data[j] = static_cast<char>((j + 1) & 0xFF);

        std::vector<double> samples;
        samples.reserve(count);
        for (int i = 0; i < count; ++i) {
            std::string key = prefix + std::to_string(i);
            double t0 = NowSec();
            ssd->Write(key, data.data(), vsize);
            double t1 = NowSec();
            samples.push_back((t1 - t0) * 1e6);
        }
        ssd->Flush();

        auto st = ComputeStats(samples);
        printf("  %8s  %5d  %10.1f  %10.1f  %10.1f  %10.1f\n",
               sz_label, count, st.avg_us, st.p50_us, st.p99_us, st.max_us);
    }

    printf("\n================================================================\n");
    printf(" READ LATENCY — backend=%s  (single-thread QD=1, O_DIRECT)\n", backend);
    printf("================================================================\n");
    printf("  %8s  %5s  %10s  %10s  %10s  %10s\n",
           "ValSize", "Count", "Avg(us)", "P50(us)", "P99(us)", "Max(us)");

    for (const auto& spec : kLatencySpecs) {
        size_t vsize = spec.size;
        int count = spec.count;

        char sz_label[16];
        if (vsize >= 1024 * 1024)
            snprintf(sz_label, sizeof(sz_label), "%zuMB", vsize / (1024 * 1024));
        else
            snprintf(sz_label, sizeof(sz_label), "%zuKB", vsize / 1024);

        std::string prefix = "lat_" + session + "_" + std::to_string(vsize) + "_";

        std::vector<char> read_buf(vsize, 0);
        uintptr_t dst = reinterpret_cast<uintptr_t>(read_buf.data());

        std::vector<double> samples;
        samples.reserve(count);
        for (int i = 0; i < count; ++i) {
            std::string key = prefix + std::to_string(i);
            double t0 = NowSec();
            ssd->ReadIntoPtr(key, dst, vsize);
            double t1 = NowSec();
            samples.push_back((t1 - t0) * 1e6);
        }

        auto st = ComputeStats(samples);
        printf("  %8s  %5d  %10.1f  %10.1f  %10.1f  %10.1f\n",
               sz_label, count, st.avg_us, st.p50_us, st.p99_us, st.max_us);
    }
    ssd->SetColdRead(false);

    printf("================================================================\n");
    fflush(stdout);
    return 0;
}

int main(int argc, char** argv) {
    int num_ranks = 1;
    bool latency_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--ranks=", 8) == 0) {
            num_ranks = std::atoi(argv[i] + 8);
        } else if (strcmp(argv[i], "--latency") == 0) {
            latency_mode = true;
        }
    }

    if (latency_mode)
        return RunLatencyBench();

    if (num_ranks <= 1) {
        return RunBenchmarkProcess(0, 1, -1, nullptr);
    }

#ifdef __linux__
    printf("Launching %d ranks (phased: Leader writes → barrier → all read)...\n",
           num_ranks);
    fflush(stdout);

    // Shared coordination via anonymous mmap (inherited by fork children)
    auto* coord = static_cast<BenchCoord*>(
        mmap(nullptr, sizeof(BenchCoord), PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    coord->write_gen.store(0, std::memory_order_relaxed);
    coord->reads_complete.store(0, std::memory_order_relaxed);
    coord->iter_barrier.store(0, std::memory_order_relaxed);
    coord->num_ranks = num_ranks;
    std::string session = MakeSessionId();
    strncpy(coord->session, session.c_str(), sizeof(coord->session) - 1);
    coord->session[sizeof(coord->session) - 1] = '\0';

    struct RankInfo {
        pid_t pid;
        int pipe_fd;
        int rank_id;
    };
    std::vector<RankInfo> ranks;

    for (int r = 0; r < num_ranks; ++r) {
        int pipefd[2];
        if (pipe(pipefd) != 0) {
            fprintf(stderr, "pipe() failed for rank %d\n", r);
            continue;
        }

        pid_t pid = fork();
        if (pid == 0) {
            close(pipefd[0]);
            char rank_str[16];
            snprintf(rank_str, sizeof(rank_str), "%d", r);
            setenv("LOCAL_RANK", rank_str, 1);
            int rc = RunBenchmarkProcess(r, num_ranks, pipefd[1], coord);
            close(pipefd[1]);
            _exit(rc);
        }
        if (pid < 0) {
            fprintf(stderr, "fork() failed for rank %d: %s\n", r, strerror(errno));
            close(pipefd[0]);
            close(pipefd[1]);
            continue;
        }
        close(pipefd[1]);
        ranks.push_back({pid, pipefd[0], r});
    }

    int failures = 0;
    struct RankResult {
        int rank_id;
        std::vector<BenchResult> results;
        bool ok;
    };
    std::vector<RankResult> all_results;

    for (auto& ri : ranks) {
        int status;
        waitpid(ri.pid, &status, 0);
        bool ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
        auto results = ReadResultsFromPipe(ri.pipe_fd);
        close(ri.pipe_fd);
        all_results.push_back({ri.rank_id, std::move(results), ok});
        if (!ok) failures++;
    }

    munmap(coord, sizeof(BenchCoord));

    printf("\n=== %d/%d ranks completed successfully ===\n",
           num_ranks - failures, num_ranks);

    auto cfg = UMBPConfig::FromEnvironment();
    const char* backend = cfg.ssd_backend.c_str();
    bool is_phased = (num_ranks > 1);

    for (auto& rr : all_results) {
        const char* role_str;
        if (is_phased)
            role_str = (rr.rank_id == 0) ? "Leader" : "Follower";
        else
            role_str = "Standalone";
        if (rr.ok && !rr.results.empty()) {
            PrintResults(rr.rank_id, num_ranks, role_str, backend, rr.results);
        } else {
            printf("\n[rank %d] FAILED (no results)\n", rr.rank_id);
        }
    }

    size_t num_specs = sizeof(kSpecs) / sizeof(kSpecs[0]);
    if (is_phased) {
        printf("\n========================================================\n");
        printf(" AGGREGATE (Leader write / %d-rank concurrent read)\n", num_ranks);
        printf("========================================================\n");
    } else {
        printf("\n========================================================\n");
        printf(" AGGREGATE (sum of %d independent ranks)\n", num_ranks);
        printf("========================================================\n");
    }
    printf("  %8s  %10s  %10s  %12s\n",
           "ValSize", "Write MB/s", "ColdRd MB/s", "HotRd MB/s");
    for (size_t s = 0; s < num_specs; ++s) {
        double sum_w = 0;
        double min_cold_t0 = 1e18, max_cold_t1 = 0;
        double min_cache_t0 = 1e18, max_cache_t1 = 0;
        double total_cold_bytes = 0, total_cache_bytes = 0;
        for (auto& rr : all_results) {
            if (!rr.ok || s >= rr.results.size()) continue;
            auto& r = rr.results[s];
            if (is_phased) {
                if (rr.rank_id == 0) sum_w = r.write_mbps;
            } else {
                sum_w += r.write_mbps;
            }

            double bytes = static_cast<double>(r.value_size) * r.count;
            if (r.cold_t0 > 0 && r.cold_t1 > r.cold_t0) {
                if (r.cold_t0 < min_cold_t0) min_cold_t0 = r.cold_t0;
                if (r.cold_t1 > max_cold_t1) max_cold_t1 = r.cold_t1;
                total_cold_bytes += bytes;
            }
            if (r.cache_t0 > 0 && r.cache_t1 > r.cache_t0) {
                if (r.cache_t0 < min_cache_t0) min_cache_t0 = r.cache_t0;
                if (r.cache_t1 > max_cache_t1) max_cache_t1 = r.cache_t1;
                total_cache_bytes += bytes;
            }
        }

        double wall_read = (max_cold_t1 > min_cold_t0)
            ? (total_cold_bytes / (1024.0 * 1024.0)) / (max_cold_t1 - min_cold_t0)
            : 0;
        double wall_cache = (max_cache_t1 > min_cache_t0)
            ? (total_cache_bytes / (1024.0 * 1024.0)) / (max_cache_t1 - min_cache_t0)
            : 0;

        char sz_label[16];
        if (kSpecs[s].size >= 1024 * 1024)
            snprintf(sz_label, sizeof(sz_label), "%zuMB",
                     kSpecs[s].size / (1024 * 1024));
        else
            snprintf(sz_label, sizeof(sz_label), "%zuKB",
                     kSpecs[s].size / 1024);
        printf("  %8s  %10.0f  %10.0f  %12.0f\n",
               sz_label, sum_w, wall_read, wall_cache);
    }
    fflush(stdout);

    return failures > 0 ? 1 : 0;
#else
    fprintf(stderr, "Multi-rank mode requires Linux\n");
    return 1;
#endif
}
