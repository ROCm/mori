// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// umbp_bench: End-to-end benchmark using UMBPClient (full stack).
// Simulates real business scenario: write KV cache blocks, then read them back.
//
// Modes:
//   Single process (default):
//     ./umbp_bench
//     UMBP_SPDK_NVME_PCI=... ./umbp_bench              # Standalone SPDK
//
//   Multi-process auto-fork (Linux only):
//     UMBP_SPDK_NVME_PCI=... ./umbp_bench --ranks=4    # Leader auto-forks proxy
//
// The benchmark writes directly to SSD tier to measure disk I/O bandwidth,
// bypassing the DRAM cache layer.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef __linux__
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
    {512ULL * 1024 * 1024, 2},
};

struct BenchResult {
    size_t value_size;
    int count;
    double write_mbps;
    double read_mbps;
    int read_ok;
};

static BenchResult RunBatch(UMBPClient& client, int rank_id,
                            const std::string& session,
                            size_t value_size, int count, int iterations) {
    std::string prefix = "bench_r" + std::to_string(rank_id) + "_" + session +
                         "_" + std::to_string(value_size) + "_";

    std::vector<std::vector<char>> datas(count);
    std::vector<size_t> sizes(count, value_size);

    for (int i = 0; i < count; ++i)
        datas[i].resize(value_size, static_cast<char>((i + 1) & 0xFF));

    double total_bytes = static_cast<double>(value_size) * count;

    auto* ssd = client.Storage().GetTier(StorageTier::LOCAL_SSD);
    if (!ssd) return {value_size, count, 0, 0, 0};

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

    // Write read-benchmark keys with retry to ensure all are present
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

    double best_read = 0;
    int best_read_ok = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        for (auto& b : read_bufs) std::memset(b.data(), 0, b.size());

        double t0 = NowSec();
        auto rr = ssd->BatchReadIntoPtr(rkeys, dst_ptrs, sizes);
        double t1 = NowSec();

        int ok = 0;
        for (auto b : rr) ok += b;
        double mbps = (total_bytes / (1024.0 * 1024.0)) / (t1 - t0);
        if (ok > 0 && mbps > best_read) { best_read = mbps; best_read_ok = ok; }
    }

    return {value_size, count, best_write, best_read, best_read_ok};
}

static void PrintResults(int rank_id, int num_ranks, const char* role_str,
                         const char* backend,
                         const std::vector<BenchResult>& results) {
    printf("\n");
    printf("========================================================\n");
    printf(" UMBPClient E2E Benchmark — rank=%d/%d role=%s backend=%s\n",
           rank_id, num_ranks, role_str, backend);
    printf("========================================================\n");
    printf("  %8s  %6s  %10s  %10s\n", "ValSize", "Count", "Write MB/s", "Read MB/s");

    for (const auto& r : results) {
        char sz_label[16];
        if (r.value_size >= 1024 * 1024)
            snprintf(sz_label, sizeof(sz_label), "%zuMB", r.value_size / (1024 * 1024));
        else
            snprintf(sz_label, sizeof(sz_label), "%zuKB", r.value_size / 1024);

        printf("  %8s  %6d  %10.0f  %10.0f", sz_label, r.count, r.write_mbps, r.read_mbps);
        if (r.read_ok != r.count)
            printf("  *** READ %d/%d", r.read_ok, r.count);
        printf("\n");
    }
    fflush(stdout);
}

#ifdef __linux__
// Write all results to a pipe so the parent can collect and print them
// sequentially, avoiding interleaved output from concurrent ranks.
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

static int RunBenchmarkProcess(int rank_id, int num_ranks, int pipe_fd) {
    auto cfg = UMBPConfig::FromEnvironment();
    cfg.dram_capacity_bytes = 64ULL * 1024 * 1024;

    UMBPClient client(cfg);

    UMBPRole role = cfg.ResolveRole();
    const char* role_str = (role == UMBPRole::Standalone) ? "Standalone" :
                           (role == UMBPRole::SharedSSDLeader) ? "Leader" : "Follower";
    const char* backend = cfg.ssd_backend.c_str();

    const int iterations = 3;
    std::string session = MakeSessionId();

    std::vector<BenchResult> results;
    for (const auto& s : kSpecs)
        results.push_back(RunBatch(client, rank_id, session, s.size, s.count, iterations));

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

int main(int argc, char** argv) {
    int num_ranks = 1;
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--ranks=", 8) == 0) {
            num_ranks = std::atoi(argv[i] + 8);
        }
    }

    if (num_ranks <= 1) {
        return RunBenchmarkProcess(0, 1, -1);
    }

#ifdef __linux__
    printf("Launching %d ranks...\n", num_ranks);
    fflush(stdout);

    struct RankInfo {
        pid_t pid;
        int pipe_fd;    // parent reads from this
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
            int rc = RunBenchmarkProcess(r, num_ranks, pipefd[1]);
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

    printf("\n=== %d/%d ranks completed successfully ===\n",
           num_ranks - failures, num_ranks);

    // Determine role/backend from environment for display
    auto cfg = UMBPConfig::FromEnvironment();
    const char* backend = cfg.ssd_backend.c_str();

    for (auto& rr : all_results) {
        const char* role_str = (rr.rank_id == 0) ? "Leader" : "Follower";
        if (rr.ok && !rr.results.empty()) {
            PrintResults(rr.rank_id, num_ranks, role_str, backend, rr.results);
        } else {
            printf("\n[rank %d] FAILED (no results)\n", rr.rank_id);
        }
    }

    // Aggregate bandwidth across all ranks
    size_t num_specs = sizeof(kSpecs) / sizeof(kSpecs[0]);
    printf("\n========================================================\n");
    printf(" AGGREGATE Bandwidth (sum of %d ranks)\n", num_ranks);
    printf("========================================================\n");
    printf("  %8s  %10s  %10s\n", "ValSize", "Write MB/s", "Read MB/s");
    for (size_t s = 0; s < num_specs; ++s) {
        double sum_w = 0, sum_r = 0;
        int valid = 0;
        for (auto& rr : all_results) {
            if (rr.ok && s < rr.results.size()) {
                sum_w += rr.results[s].write_mbps;
                sum_r += rr.results[s].read_mbps;
                ++valid;
            }
        }
        if (valid == 0) continue;
        char sz_label[16];
        if (kSpecs[s].size >= 1024 * 1024)
            snprintf(sz_label, sizeof(sz_label), "%zuMB",
                     kSpecs[s].size / (1024 * 1024));
        else
            snprintf(sz_label, sizeof(sz_label), "%zuKB",
                     kSpecs[s].size / 1024);
        printf("  %8s  %10.0f  %10.0f\n", sz_label, sum_w, sum_r);
    }
    fflush(stdout);

    return failures > 0 ? 1 : 0;
#else
    fprintf(stderr, "Multi-rank mode requires Linux\n");
    return 1;
#endif
}
