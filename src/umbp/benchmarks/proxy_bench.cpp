// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// proxy_bench: Benchmark for SpdkProxyTier (shared memory IPC to spdk_proxy).
// Measures batch write/read throughput at various value sizes.
//
// Requires: spdk_proxy daemon running.
// Usage:
//   UMBP_SPDK_PROXY_RANK=0 ./proxy_bench

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/storage/spdk_proxy_tier.h"

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

static void RunBatch(SpdkProxyTier& tier, uint32_t rank_id,
                     const std::string& session,
                     size_t value_size, int count, int iterations) {
    std::string prefix = "r" + std::to_string(rank_id) + "_" + session +
                         "_" + std::to_string(value_size) + "_";

    std::vector<std::vector<char>> datas(count);
    std::vector<const void*> ptrs(count);
    std::vector<size_t> sizes(count, value_size);

    for (int i = 0; i < count; ++i) {
        datas[i].resize(value_size, static_cast<char>((i + 1) & 0xFF));
        ptrs[i] = datas[i].data();
    }

    double total_bytes = static_cast<double>(value_size) * count;

    // Write benchmark — unique keys per iteration to avoid dedup
    double best_write = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<std::string> wkeys(count);
        for (int i = 0; i < count; ++i)
            wkeys[i] = prefix + "w" + std::to_string(iter) + "_" + std::to_string(i);
        double t0 = NowSec();
        auto wr = tier.BatchWrite(wkeys, ptrs, sizes);
        double t1 = NowSec();
        int ok = 0;
        for (auto b : wr) ok += b;
        double mbps = (total_bytes / (1024.0 * 1024.0)) / (t1 - t0);
        if (ok == count && mbps > best_write) best_write = mbps;
    }

    // Write read-benchmark keys (fixed set, no iter suffix)
    std::vector<std::string> rkeys(count);
    for (int i = 0; i < count; ++i)
        rkeys[i] = prefix + "r_" + std::to_string(i);
    tier.BatchWrite(rkeys, ptrs, sizes);

    std::vector<std::vector<char>> read_bufs(count, std::vector<char>(value_size, 0));
    std::vector<uintptr_t> dst_ptrs(count);
    for (int i = 0; i < count; ++i)
        dst_ptrs[i] = reinterpret_cast<uintptr_t>(read_bufs[i].data());

    double best_read = 0;
    int best_read_ok = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        for (auto& b : read_bufs) std::memset(b.data(), 0, b.size());
        double t0 = NowSec();
        auto rr = tier.BatchReadIntoPtr(rkeys, dst_ptrs, sizes);
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
}

int main() {
    auto cfg = UMBPConfig::FromEnvironment();
    cfg.ssd_backend = "spdk_proxy";

    printf("SpdkProxyTier Benchmark — rank=%u shm='%s'\n",
           cfg.spdk_proxy_rank_id, cfg.spdk_proxy_shm_name.c_str());

    SpdkProxyTier tier(cfg);
    if (!tier.IsValid()) {
        fprintf(stderr, "FAILED: cannot connect to spdk_proxy daemon\n");
        return 1;
    }

    auto [used, total] = tier.Capacity();
    printf("  capacity: used=%zuMB total=%zuMB\n",
           used / (1024 * 1024), total / (1024 * 1024));

    struct SizeSpec { size_t size; int count; };
    SizeSpec specs[] = {
        {4 * 1024, 2000},
        {32 * 1024, 2000},
        {128 * 1024, 2000},
        {512 * 1024, 1024},
        {1024 * 1024, 512},
        {2 * 1024 * 1024, 256},
        {8 * 1024 * 1024, 64},
        {16 * 1024 * 1024, 32},
        {32 * 1024 * 1024, 16},
        {64 * 1024 * 1024, 8},
        {128 * 1024 * 1024, 4},
        {256 * 1024 * 1024, 2},
        {512 * 1024 * 1024, 2},
    };

    const int iterations = 3;
    std::string session = MakeSessionId();

    printf("\n%s\n SpdkProxyTier BATCH THROUGHPUT (rank=%u session=%s)\n%s\n",
           std::string(72, '-').c_str(), cfg.spdk_proxy_rank_id,
           session.c_str(), std::string(72, '-').c_str());
    printf("  %8s  %6s  %10s  %10s\n", "ValSize", "Count", "Write MB/s", "Read MB/s");

    for (auto& s : specs) {
        RunBatch(tier, cfg.spdk_proxy_rank_id, session, s.size, s.count, iterations);
    }

    printf("\n");
    return 0;
}
