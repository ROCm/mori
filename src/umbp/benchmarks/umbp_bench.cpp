// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// umbp_bench: End-to-end benchmark using UMBPClient (full stack).
// Simulates real business scenario: write KV cache blocks, then read them back.
//
// Modes:
//   Single process (default):
//     UMBP_SSD_BACKEND=posix ./umbp_bench
//     UMBP_SSD_BACKEND=spdk  ./umbp_bench              # Standalone SPDK
//
//   Multi-process auto-fork (Linux only):
//     UMBP_SSD_BACKEND=spdk  ./umbp_bench --ranks=4    # Leader auto-forks proxy+followers
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
};

static void RunBatch(UMBPClient& client, int rank_id,
                     const std::string& session,
                     size_t value_size, int count, int iterations) {
    std::string prefix = "bench_r" + std::to_string(rank_id) + "_" + session +
                         "_" + std::to_string(value_size) + "_";

    std::vector<std::vector<char>> datas(count);
    std::vector<uintptr_t> src_ptrs(count);
    std::vector<size_t> sizes(count, value_size);

    for (int i = 0; i < count; ++i) {
        datas[i].resize(value_size, static_cast<char>((i + 1) & 0xFF));
        src_ptrs[i] = reinterpret_cast<uintptr_t>(datas[i].data());
    }

    double total_bytes = static_cast<double>(value_size) * count;

    auto* ssd = client.Storage().GetTier(StorageTier::LOCAL_SSD);
    if (!ssd) return;

    std::vector<const void*> cptrs(count);
    for (int i = 0; i < count; ++i)
        cptrs[i] = datas[i].data();

    // Write benchmark — unique keys per iteration, directly to SSD
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

    // Write read-benchmark keys (fixed set)
    std::vector<std::string> rkeys(count);
    for (int i = 0; i < count; ++i)
        rkeys[i] = prefix + "r_" + std::to_string(i);
    ssd->BatchWrite(rkeys, cptrs, sizes);

    // Read benchmark
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
        if (mbps > best_read) { best_read = mbps; best_read_ok = ok; }
    }

    char sz_label[16];
    if (value_size >= 1024 * 1024)
        snprintf(sz_label, sizeof(sz_label), "%zuMB", value_size / (1024 * 1024));
    else
        snprintf(sz_label, sizeof(sz_label), "%zuKB", value_size / 1024);

    printf("  %8s  %6d  %10.0f  %10.0f", sz_label, count, best_write, best_read);
    if (best_read_ok != count)
        printf("  *** READ %d/%d", best_read_ok, count);
    printf("\n");
    fflush(stdout);
}

static int RunBenchmarkProcess(int rank_id, int num_ranks) {
    auto cfg = UMBPConfig::FromEnvironment();
    // For multi-rank: override DRAM to small so we test SSD path
    cfg.dram_capacity_bytes = 64ULL * 1024 * 1024;  // 64MB DRAM cache

    UMBPClient client(cfg);

    UMBPRole role = cfg.ResolveRole();
    const char* role_str = (role == UMBPRole::Standalone) ? "Standalone" :
                           (role == UMBPRole::SharedSSDLeader) ? "Leader" : "Follower";
    const char* backend = cfg.ssd_backend.c_str();

    printf("\n");
    printf("========================================================\n");
    printf(" UMBPClient E2E Benchmark — rank=%d/%d role=%s backend=%s\n",
           rank_id, num_ranks, role_str, backend);
    printf("========================================================\n");
    printf("  %8s  %6s  %10s  %10s\n", "ValSize", "Count", "Write MB/s", "Read MB/s");

    const int iterations = 3;
    std::string session = MakeSessionId();

    for (const auto& s : kSpecs) {
        RunBatch(client, rank_id, session, s.size, s.count, iterations);
    }

    printf("\n");
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
        return RunBenchmarkProcess(0, 1);
    }

#ifdef __linux__
    printf("Launching %d ranks...\n", num_ranks);

    std::vector<pid_t> children;
    for (int r = 0; r < num_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child sets LOCAL_RANK → auto-deduced role
            char rank_str[16];
            snprintf(rank_str, sizeof(rank_str), "%d", r);
            setenv("LOCAL_RANK", rank_str, 1);
            _exit(RunBenchmarkProcess(r, num_ranks));
        }
        if (pid < 0) {
            fprintf(stderr, "fork() failed for rank %d: %s\n", r, strerror(errno));
            continue;
        }
        children.push_back(pid);
    }

    int failures = 0;
    for (auto pid : children) {
        int status;
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) failures++;
    }

    printf("\n=== %d/%d ranks completed successfully ===\n",
           num_ranks - failures, num_ranks);
    return failures > 0 ? 1 : 0;
#else
    fprintf(stderr, "Multi-rank mode requires Linux\n");
    return 1;
#endif
}
