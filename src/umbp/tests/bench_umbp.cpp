// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// UMBP Benchmark Tool
//
// Standardized benchmark for measuring UMBP write-path optimizations:
// zero-copy CopyToSSD, lock splitting, multi-threaded async copy, batch io_uring.
//
// Usage:
//   bench_umbp [OPTIONS]
//     --profile <small|medium|large>   Preset config (default: medium)
//     --num-keys N                     Keys per scenario
//     --value-size N                   Value size in bytes
//     --batch-size N                   Batch size
//     --iters N                        Measurement iterations
//     --filter SUBSTRING               Run only matching scenarios
//     --dir PATH                       Temp directory path
//     -h, --help                       Help

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/common/storage_tier.h"
#include "umbp/storage/dram_tier.h"
#include "umbp/storage/local_storage_manager.h"
#include "umbp/storage/ssd_tier.h"
#include "umbp/umbp_client.h"

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Clock alias
// ---------------------------------------------------------------------------
using Clock = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// BenchConfig
// ---------------------------------------------------------------------------
struct BenchConfig {
  size_t num_keys = 1000;
  size_t value_size = 4096;
  size_t batch_size = 64;
  size_t warmup_iters = 1;
  size_t measure_iters = 3;

  size_t dram_capacity = 64ULL * 1024 * 1024;
  size_t ssd_capacity = 256ULL * 1024 * 1024;
  size_t segment_size = 64ULL * 1024 * 1024;

  std::vector<int> thread_counts = {1, 2, 4, 8};

  std::string base_dir = "/tmp/umbp_bench";
  std::string filter;
};

// ---------------------------------------------------------------------------
// BenchResult
// ---------------------------------------------------------------------------
struct BenchResult {
  std::string name;
  std::string variant;
  size_t ops = 0;
  size_t bytes = 0;
  double elapsed_sec = 0.0;

  double lat_min_us = 0.0;
  double lat_avg_us = 0.0;
  double lat_p50_us = 0.0;
  double lat_p95_us = 0.0;
  double lat_p99_us = 0.0;
  double lat_max_us = 0.0;

  double throughput_ops_sec() const {
    return elapsed_sec > 0.0 ? static_cast<double>(ops) / elapsed_sec : 0.0;
  }
  double throughput_mb_sec() const {
    return elapsed_sec > 0.0 ? static_cast<double>(bytes) / (1024.0 * 1024.0) / elapsed_sec : 0.0;
  }
};

// ---------------------------------------------------------------------------
// ScopedTempDir — RAII directory cleanup
// ---------------------------------------------------------------------------
struct ScopedTempDir {
  std::string path;
  explicit ScopedTempDir(const std::string& p) : path(p) {
    fs::remove_all(path);
    fs::create_directories(path);
  }
  ~ScopedTempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
  ScopedTempDir(const ScopedTempDir&) = delete;
  ScopedTempDir& operator=(const ScopedTempDir&) = delete;
};

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------
static std::string MakeKey(size_t idx) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "bench_%08zu", idx);
  return std::string(buf);
}

static std::vector<std::string> GenerateKeys(size_t n) {
  std::vector<std::string> keys(n);
  for (size_t i = 0; i < n; ++i) {
    keys[i] = MakeKey(i);
  }
  return keys;
}

static std::vector<std::vector<char>> GenerateValues(size_t n, size_t value_size) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<std::vector<char>> values(n);
  for (size_t i = 0; i < n; ++i) {
    values[i].resize(value_size);
    for (size_t j = 0; j < value_size; ++j) {
      values[i][j] = static_cast<char>(dist(rng));
    }
  }
  return values;
}

// ---------------------------------------------------------------------------
// Latency statistics
// ---------------------------------------------------------------------------
static void ComputeLatencyStats(std::vector<double>& latencies, BenchResult& result) {
  if (latencies.empty()) return;
  std::sort(latencies.begin(), latencies.end());
  size_t n = latencies.size();
  result.lat_min_us = latencies.front();
  result.lat_max_us = latencies.back();
  result.lat_avg_us =
      std::accumulate(latencies.begin(), latencies.end(), 0.0) / static_cast<double>(n);
  result.lat_p50_us = latencies[n * 50 / 100];
  result.lat_p95_us = latencies[n * 95 / 100];
  result.lat_p99_us = latencies[std::min(n * 99 / 100, n - 1)];
}

// ---------------------------------------------------------------------------
// Result formatting
// ---------------------------------------------------------------------------
static void PrintHeader(const std::string& section) {
  std::cout << "\n=== " << section << " ===" << std::endl;
  std::printf("%-36s %-20s %8s %10s %9s %9s %9s %9s %9s %9s %12s\n", "Benchmark", "Variant", "Ops",
              "MB/s", "min(us)", "avg(us)", "p50(us)", "p95(us)", "p99(us)", "max(us)", "ops/s");
  std::printf("%s\n", std::string(36 + 20 + 8 + 10 + 9 * 6 + 12 + 10, '-').c_str());
}

static void PrintResult(const BenchResult& r) {
  std::printf("%-36s %-20s %8zu %10.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %12.0f\n",
              r.name.c_str(), r.variant.c_str(), r.ops, r.throughput_mb_sec(), r.lat_min_us,
              r.lat_avg_us, r.lat_p50_us, r.lat_p95_us, r.lat_p99_us, r.lat_max_us,
              r.throughput_ops_sec());
}

// ---------------------------------------------------------------------------
// Scenario filter check
// ---------------------------------------------------------------------------
static bool ShouldRun(const BenchConfig& cfg, const std::string& name) {
  if (cfg.filter.empty()) return true;
  return name.find(cfg.filter) != std::string::npos;
}

// ---------------------------------------------------------------------------
// A. DRAM Tier Benchmarks
// ---------------------------------------------------------------------------
static void BenchDRAMTier(const BenchConfig& cfg, const std::vector<std::string>& keys,
                          const std::vector<std::vector<char>>& values,
                          std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "DRAM")) return;

  PrintHeader("DRAM Tier");

  DRAMTier tier(cfg.dram_capacity);

  // Pre-fill for read benchmarks
  for (size_t i = 0; i < keys.size(); ++i) {
    tier.Write(keys[i], values[i].data(), values[i].size());
  }

  // 1. DRAM Write
  {
    // Clear and re-fill for write benchmark
    tier.Clear();
    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        tier.Write(keys[i], values[i].data(), values[i].size());
      }
    }
    tier.Clear();

    std::vector<double> latencies;
    latencies.reserve(keys.size() * cfg.measure_iters);

    auto wall_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        tier.Write(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        latencies.push_back(us);
      }
    }
    auto wall_end = Clock::now();

    BenchResult r;
    r.name = "DRAM Write";
    r.variant = "single-key";
    r.ops = keys.size() * cfg.measure_iters;
    r.bytes = r.ops * cfg.value_size;
    r.elapsed_sec = std::chrono::duration<double>(wall_end - wall_start).count();
    ComputeLatencyStats(latencies, r);
    PrintResult(r);
    results.push_back(r);
  }

  // Re-fill for reads
  tier.Clear();
  for (size_t i = 0; i < keys.size(); ++i) {
    tier.Write(keys[i], values[i].data(), values[i].size());
  }

  // 2. DRAM Read (ReadIntoPtr)
  {
    std::vector<char> buf(cfg.value_size);

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (size_t i = 0; i < keys.size(); ++i) {
        tier.ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
      }
    }

    std::vector<double> latencies;
    latencies.reserve(keys.size() * cfg.measure_iters);

    auto wall_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        tier.ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto wall_end = Clock::now();

    BenchResult r;
    r.name = "DRAM Read";
    r.variant = "single-key";
    r.ops = keys.size() * cfg.measure_iters;
    r.bytes = r.ops * cfg.value_size;
    r.elapsed_sec = std::chrono::duration<double>(wall_end - wall_start).count();
    ComputeLatencyStats(latencies, r);
    PrintResult(r);
    results.push_back(r);
  }

  // 3. DRAM ReadPtr (zero-copy)
  {
    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (size_t i = 0; i < keys.size(); ++i) {
        size_t sz = 0;
        tier.ReadPtr(keys[i], &sz);
      }
    }

    std::vector<double> latencies;
    latencies.reserve(keys.size() * cfg.measure_iters);

    auto wall_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        size_t sz = 0;
        auto t0 = Clock::now();
        tier.ReadPtr(keys[i], &sz);
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto wall_end = Clock::now();

    BenchResult r;
    r.name = "DRAM ReadPtr (zero-copy)";
    r.variant = "single-key";
    r.ops = keys.size() * cfg.measure_iters;
    r.bytes = r.ops * cfg.value_size;
    r.elapsed_sec = std::chrono::duration<double>(wall_end - wall_start).count();
    ComputeLatencyStats(latencies, r);
    PrintResult(r);
    results.push_back(r);
  }
}

// ---------------------------------------------------------------------------
// B. SSD Tier Benchmarks
// ---------------------------------------------------------------------------
static void BenchSSDTier(const BenchConfig& cfg, const std::vector<std::string>& keys,
                         const std::vector<std::vector<char>>& values,
                         std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "SSD Tier")) return;

  PrintHeader("SSD Tier");

  UMBPConfig ucfg;
  ucfg.ssd.io.backend = UMBPIoBackend::PThread;
  ucfg.ssd.durability.mode = UMBPDurabilityMode::Relaxed;
  ucfg.ssd.segment_size_bytes = cfg.segment_size;

  // Write
  {
    ScopedTempDir tmp(cfg.base_dir + "/ssd_tier_write");
    SSDTier tier(tmp.path, cfg.ssd_capacity, ucfg);

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        tier.Write(keys[i], values[i].data(), values[i].size());
      }
    }
    tier.Clear();

    std::vector<double> latencies;
    latencies.reserve(keys.size() * cfg.measure_iters);

    auto wall_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        tier.Write(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto wall_end = Clock::now();

    BenchResult r;
    r.name = "SSD Tier Write";
    r.variant = "single-key";
    r.ops = keys.size() * cfg.measure_iters;
    r.bytes = r.ops * cfg.value_size;
    r.elapsed_sec = std::chrono::duration<double>(wall_end - wall_start).count();
    ComputeLatencyStats(latencies, r);
    PrintResult(r);
    results.push_back(r);
  }

  // Read
  {
    ScopedTempDir tmp(cfg.base_dir + "/ssd_tier_read");
    SSDTier tier(tmp.path, cfg.ssd_capacity, ucfg);

    // Pre-fill
    for (size_t i = 0; i < keys.size(); ++i) {
      tier.Write(keys[i], values[i].data(), values[i].size());
    }

    std::vector<char> buf(cfg.value_size);

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (size_t i = 0; i < keys.size(); ++i) {
        tier.ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
      }
    }

    std::vector<double> latencies;
    latencies.reserve(keys.size() * cfg.measure_iters);

    auto wall_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        tier.ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto wall_end = Clock::now();

    BenchResult r;
    r.name = "SSD Tier Read";
    r.variant = "single-key";
    r.ops = keys.size() * cfg.measure_iters;
    r.bytes = r.ops * cfg.value_size;
    r.elapsed_sec = std::chrono::duration<double>(wall_end - wall_start).count();
    ComputeLatencyStats(latencies, r);
    PrintResult(r);
    results.push_back(r);
  }
}

// ---------------------------------------------------------------------------
// C. Batch vs Single Write
// ---------------------------------------------------------------------------
static void BenchBatchWrite(const BenchConfig& cfg, const std::vector<std::string>& keys,
                            const std::vector<std::vector<char>>& values,
                            std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "Batch")) return;

  PrintHeader("Batch vs Single Write");

  UMBPConfig ucfg;
  ucfg.ssd.io.backend = UMBPIoBackend::PThread;
  ucfg.ssd.durability.mode = UMBPDurabilityMode::Relaxed;
  ucfg.ssd.segment_size_bytes = cfg.segment_size;

  // Sequential single-key write
  {
    ScopedTempDir tmp(cfg.base_dir + "/batch_seq");
    SSDTier tier(tmp.path, cfg.ssd_capacity, ucfg);

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        tier.Write(keys[i], values[i].data(), values[i].size());
      }
    }
    tier.Clear();

    std::vector<double> latencies;
    latencies.reserve(keys.size() * cfg.measure_iters);

    auto wall_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        tier.Write(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto wall_end = Clock::now();

    BenchResult r;
    r.name = "SSD Write";
    r.variant = "sequential";
    r.ops = keys.size() * cfg.measure_iters;
    r.bytes = r.ops * cfg.value_size;
    r.elapsed_sec = std::chrono::duration<double>(wall_end - wall_start).count();
    ComputeLatencyStats(latencies, r);
    PrintResult(r);
    results.push_back(r);
  }

  // WriteBatch
  {
    ScopedTempDir tmp(cfg.base_dir + "/batch_batch");
    SSDTier tier(tmp.path, cfg.ssd_capacity, ucfg);

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); i += cfg.batch_size) {
        size_t end = std::min(i + cfg.batch_size, keys.size());
        std::vector<std::string> batch_keys(keys.begin() + i, keys.begin() + end);
        std::vector<const void*> batch_ptrs;
        std::vector<size_t> batch_sizes;
        for (size_t j = i; j < end; ++j) {
          batch_ptrs.push_back(values[j].data());
          batch_sizes.push_back(values[j].size());
        }
        tier.WriteBatch(batch_keys, batch_ptrs, batch_sizes);
      }
    }
    tier.Clear();

    std::vector<double> latencies;
    size_t total_batches = (keys.size() + cfg.batch_size - 1) / cfg.batch_size;
    latencies.reserve(total_batches * cfg.measure_iters);

    auto wall_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); i += cfg.batch_size) {
        size_t end = std::min(i + cfg.batch_size, keys.size());
        std::vector<std::string> batch_keys(keys.begin() + i, keys.begin() + end);
        std::vector<const void*> batch_ptrs;
        std::vector<size_t> batch_sizes;
        for (size_t j = i; j < end; ++j) {
          batch_ptrs.push_back(values[j].data());
          batch_sizes.push_back(values[j].size());
        }
        auto t0 = Clock::now();
        tier.WriteBatch(batch_keys, batch_ptrs, batch_sizes);
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto wall_end = Clock::now();

    BenchResult r;
    r.name = "SSD Write";
    r.variant = "WriteBatch(bs=" + std::to_string(cfg.batch_size) + ")";
    r.ops = keys.size() * cfg.measure_iters;
    r.bytes = r.ops * cfg.value_size;
    r.elapsed_sec = std::chrono::duration<double>(wall_end - wall_start).count();
    ComputeLatencyStats(latencies, r);
    PrintResult(r);
    results.push_back(r);
  }
}

// ---------------------------------------------------------------------------
// D. CopyToSSD vs CopyToSSDBatch
// ---------------------------------------------------------------------------
static void BenchCopyToSSD(const BenchConfig& cfg, const std::vector<std::string>& keys,
                           const std::vector<std::vector<char>>& values,
                           std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "CopyToSSD")) return;

  PrintHeader("CopyToSSD vs CopyToSSDBatch");

  // Single CopyToSSD
  {
    ScopedTempDir tmp(cfg.base_dir + "/copy_single");

    UMBPConfig ucfg;
    ucfg.dram.capacity_bytes = cfg.dram_capacity;
    ucfg.ssd.enabled = true;
    ucfg.ssd.storage_dir = tmp.path + "/ssd";
    ucfg.ssd.capacity_bytes = cfg.ssd_capacity;

    ucfg.ssd.io.backend = UMBPIoBackend::PThread;
    ucfg.ssd.durability.mode = UMBPDurabilityMode::Relaxed;
    ucfg.ssd.segment_size_bytes = cfg.segment_size;
    ucfg.eviction.auto_promote_on_read = false;

    fs::create_directories(ucfg.ssd.storage_dir);
    LocalStorageManager mgr(ucfg);

    // Pre-fill DRAM
    for (size_t i = 0; i < keys.size(); ++i) {
      mgr.Write(keys[i], values[i].data(), values[i].size(), StorageTier::CPU_DRAM);
    }

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (size_t i = 0; i < keys.size(); ++i) {
        mgr.CopyToSSD(keys[i]);
      }
    }

    std::vector<double> latencies;
    latencies.reserve(keys.size() * cfg.measure_iters);

    auto wall_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        mgr.CopyToSSD(keys[i]);
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto wall_end = Clock::now();

    BenchResult r;
    r.name = "CopyToSSD";
    r.variant = "single-key";
    r.ops = keys.size() * cfg.measure_iters;
    r.bytes = r.ops * cfg.value_size;
    r.elapsed_sec = std::chrono::duration<double>(wall_end - wall_start).count();
    ComputeLatencyStats(latencies, r);
    PrintResult(r);
    results.push_back(r);
  }

  // Batch CopyToSSDBatch
  {
    ScopedTempDir tmp(cfg.base_dir + "/copy_batch");

    UMBPConfig ucfg;
    ucfg.dram.capacity_bytes = cfg.dram_capacity;
    ucfg.ssd.enabled = true;
    ucfg.ssd.storage_dir = tmp.path + "/ssd";
    ucfg.ssd.capacity_bytes = cfg.ssd_capacity;

    ucfg.ssd.io.backend = UMBPIoBackend::PThread;
    ucfg.ssd.durability.mode = UMBPDurabilityMode::Relaxed;
    ucfg.ssd.segment_size_bytes = cfg.segment_size;
    ucfg.eviction.auto_promote_on_read = false;

    fs::create_directories(ucfg.ssd.storage_dir);
    LocalStorageManager mgr(ucfg);

    // Pre-fill DRAM
    for (size_t i = 0; i < keys.size(); ++i) {
      mgr.Write(keys[i], values[i].data(), values[i].size(), StorageTier::CPU_DRAM);
    }

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (size_t i = 0; i < keys.size(); i += cfg.batch_size) {
        size_t end = std::min(i + cfg.batch_size, keys.size());
        std::vector<std::string> batch(keys.begin() + i, keys.begin() + end);
        mgr.CopyToSSDBatch(batch);
      }
    }

    std::vector<double> latencies;
    size_t total_batches = (keys.size() + cfg.batch_size - 1) / cfg.batch_size;
    latencies.reserve(total_batches * cfg.measure_iters);

    auto wall_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); i += cfg.batch_size) {
        size_t end = std::min(i + cfg.batch_size, keys.size());
        std::vector<std::string> batch(keys.begin() + i, keys.begin() + end);
        auto t0 = Clock::now();
        mgr.CopyToSSDBatch(batch);
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto wall_end = Clock::now();

    BenchResult r;
    r.name = "CopyToSSD";
    r.variant = "batch(bs=" + std::to_string(cfg.batch_size) + ")";
    r.ops = keys.size() * cfg.measure_iters;
    r.bytes = r.ops * cfg.value_size;
    r.elapsed_sec = std::chrono::duration<double>(wall_end - wall_start).count();
    ComputeLatencyStats(latencies, r);
    PrintResult(r);
    results.push_back(r);
  }
}

// ---------------------------------------------------------------------------
// E. IO Backend: POSIX vs io_uring
// ---------------------------------------------------------------------------
static void BenchIOBackend(const BenchConfig& cfg, const std::vector<std::string>& keys,
                           const std::vector<std::vector<char>>& values,
                           std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "IO Backend")) return;

  PrintHeader("IO Backend (POSIX vs io_uring)");

  auto run_backend = [&](UMBPIoBackend backend, const std::string& label) {
    UMBPConfig ucfg;
    ucfg.ssd.io.backend = backend;
    ucfg.ssd.durability.mode = UMBPDurabilityMode::Relaxed;
    ucfg.ssd.segment_size_bytes = cfg.segment_size;
    ucfg.ssd.io.queue_depth = 4096;

    std::string suffix = (backend == UMBPIoBackend::PThread) ? "posix" : "iouring";

    // Write
    {
      ScopedTempDir tmp(cfg.base_dir + "/io_" + suffix + "_w");
      std::unique_ptr<SSDTier> tier;
      try {
        tier = std::make_unique<SSDTier>(tmp.path, cfg.ssd_capacity, ucfg);
      } catch (const std::exception& e) {
        std::printf("[SKIP] %s not available: %s\n", label.c_str(), e.what());
        return;
      }

      // Warmup
      for (size_t w = 0; w < cfg.warmup_iters; ++w) {
        tier->Clear();
        for (size_t i = 0; i < keys.size(); ++i) {
          tier->Write(keys[i], values[i].data(), values[i].size());
        }
      }
      tier->Clear();

      std::vector<double> latencies;
      latencies.reserve(keys.size() * cfg.measure_iters);

      auto wall_start = Clock::now();
      for (size_t m = 0; m < cfg.measure_iters; ++m) {
        tier->Clear();
        for (size_t i = 0; i < keys.size(); ++i) {
          auto t0 = Clock::now();
          tier->Write(keys[i], values[i].data(), values[i].size());
          auto t1 = Clock::now();
          latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
      }
      auto wall_end = Clock::now();

      BenchResult r;
      r.name = "IO Backend Write";
      r.variant = label;
      r.ops = keys.size() * cfg.measure_iters;
      r.bytes = r.ops * cfg.value_size;
      r.elapsed_sec = std::chrono::duration<double>(wall_end - wall_start).count();
      ComputeLatencyStats(latencies, r);
      PrintResult(r);
      results.push_back(r);
    }

    // Read
    {
      ScopedTempDir tmp(cfg.base_dir + "/io_" + suffix + "_r");
      std::unique_ptr<SSDTier> tier;
      try {
        tier = std::make_unique<SSDTier>(tmp.path, cfg.ssd_capacity, ucfg);
      } catch (const std::exception& e) {
        std::printf("[SKIP] %s not available: %s\n", label.c_str(), e.what());
        return;
      }

      // Pre-fill
      for (size_t i = 0; i < keys.size(); ++i) {
        tier->Write(keys[i], values[i].data(), values[i].size());
      }

      std::vector<char> buf(cfg.value_size);

      // Warmup
      for (size_t w = 0; w < cfg.warmup_iters; ++w) {
        for (size_t i = 0; i < keys.size(); ++i) {
          tier->ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
        }
      }

      std::vector<double> latencies;
      latencies.reserve(keys.size() * cfg.measure_iters);

      auto wall_start = Clock::now();
      for (size_t m = 0; m < cfg.measure_iters; ++m) {
        for (size_t i = 0; i < keys.size(); ++i) {
          auto t0 = Clock::now();
          tier->ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
          auto t1 = Clock::now();
          latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
      }
      auto wall_end = Clock::now();

      BenchResult r;
      r.name = "IO Backend Read";
      r.variant = label;
      r.ops = keys.size() * cfg.measure_iters;
      r.bytes = r.ops * cfg.value_size;
      r.elapsed_sec = std::chrono::duration<double>(wall_end - wall_start).count();
      ComputeLatencyStats(latencies, r);
      PrintResult(r);
      results.push_back(r);
    }
  };

  run_backend(UMBPIoBackend::PThread, "POSIX");
  run_backend(UMBPIoBackend::IoUring, "io_uring");
}

// ---------------------------------------------------------------------------
// F. Durability: Strict vs Relaxed
// ---------------------------------------------------------------------------
static void BenchDurability(const BenchConfig& cfg, const std::vector<std::string>& keys,
                            const std::vector<std::vector<char>>& values,
                            std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "Durability")) return;

  PrintHeader("Durability (Strict vs Relaxed)");

  auto run_mode = [&](UMBPDurabilityMode mode, const std::string& label) {
    UMBPConfig ucfg;
    ucfg.ssd.io.backend = UMBPIoBackend::PThread;
    ucfg.ssd.durability.mode = mode;
    ucfg.ssd.segment_size_bytes = cfg.segment_size;

    ScopedTempDir tmp(cfg.base_dir + "/dur_" + label);
    SSDTier tier(tmp.path, cfg.ssd_capacity, ucfg);

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        tier.Write(keys[i], values[i].data(), values[i].size());
      }
    }
    tier.Clear();

    std::vector<double> latencies;
    latencies.reserve(keys.size() * cfg.measure_iters);

    auto wall_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        tier.Write(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto wall_end = Clock::now();

    BenchResult r;
    r.name = "SSD Write Durability";
    r.variant = label;
    r.ops = keys.size() * cfg.measure_iters;
    r.bytes = r.ops * cfg.value_size;
    r.elapsed_sec = std::chrono::duration<double>(wall_end - wall_start).count();
    ComputeLatencyStats(latencies, r);
    PrintResult(r);
    results.push_back(r);
  };

  run_mode(UMBPDurabilityMode::Strict, "Strict");
  run_mode(UMBPDurabilityMode::Relaxed, "Relaxed");
}

// ---------------------------------------------------------------------------
// G. Concurrent Scaling (UMBPClient Put + Get)
// ---------------------------------------------------------------------------
static void BenchConcurrent(const BenchConfig& cfg, const std::vector<std::string>& keys,
                            const std::vector<std::vector<char>>& values,
                            std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "Concurrent")) return;

  PrintHeader("Concurrent Scaling");

  for (int nthreads : cfg.thread_counts) {
    ScopedTempDir tmp(cfg.base_dir + "/concurrent_" + std::to_string(nthreads));

    UMBPConfig ucfg;
    ucfg.dram.capacity_bytes = cfg.dram_capacity;
    ucfg.ssd.enabled = true;
    ucfg.ssd.storage_dir = tmp.path + "/ssd";
    ucfg.ssd.capacity_bytes = cfg.ssd_capacity;

    ucfg.ssd.io.backend = UMBPIoBackend::PThread;
    ucfg.ssd.durability.mode = UMBPDurabilityMode::Relaxed;
    ucfg.ssd.segment_size_bytes = cfg.segment_size;
    ucfg.role = UMBPRole::Standalone;
    ucfg.copy_pipeline.async_enabled = false;
    ucfg.eviction.auto_promote_on_read = false;

    fs::create_directories(ucfg.ssd.storage_dir);
    UMBPClient client(ucfg);

    size_t keys_per_thread = keys.size() / static_cast<size_t>(nthreads);
    if (keys_per_thread == 0) continue;

    std::string variant = std::to_string(nthreads) + " threads";

    // --- Put ---
    {
      // Warmup
      for (size_t w = 0; w < cfg.warmup_iters; ++w) {
        client.Clear();
        for (size_t i = 0; i < keys.size(); ++i) {
          client.Put(keys[i], values[i].data(), values[i].size());
        }
      }
      client.Clear();

      std::vector<std::vector<double>> thread_latencies(nthreads);
      for (int t = 0; t < nthreads; ++t) {
        thread_latencies[t].reserve(keys_per_thread * cfg.measure_iters);
      }

      auto wall_start = Clock::now();
      for (size_t m = 0; m < cfg.measure_iters; ++m) {
        client.Clear();
        std::vector<std::thread> threads;
        for (int t = 0; t < nthreads; ++t) {
          threads.emplace_back([&, t]() {
            size_t start = t * keys_per_thread;
            size_t end = start + keys_per_thread;
            for (size_t i = start; i < end; ++i) {
              auto t0 = Clock::now();
              client.Put(keys[i], values[i].data(), values[i].size());
              auto t1 = Clock::now();
              thread_latencies[t].push_back(
                  std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
          });
        }
        for (auto& th : threads) th.join();
      }
      auto wall_end = Clock::now();

      // Merge latencies
      std::vector<double> all_lat;
      for (auto& tl : thread_latencies) {
        all_lat.insert(all_lat.end(), tl.begin(), tl.end());
      }

      BenchResult r;
      r.name = "Concurrent Put";
      r.variant = variant;
      r.ops = keys_per_thread * static_cast<size_t>(nthreads) * cfg.measure_iters;
      r.bytes = r.ops * cfg.value_size;
      r.elapsed_sec = std::chrono::duration<double>(wall_end - wall_start).count();
      ComputeLatencyStats(all_lat, r);
      PrintResult(r);
      results.push_back(r);
    }

    // --- Get ---
    {
      // Ensure data is present
      client.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        client.Put(keys[i], values[i].data(), values[i].size());
      }

      // Warmup
      std::vector<char> wbuf(cfg.value_size);
      for (size_t w = 0; w < cfg.warmup_iters; ++w) {
        for (size_t i = 0; i < keys.size(); ++i) {
          client.GetIntoPtr(keys[i], reinterpret_cast<uintptr_t>(wbuf.data()), wbuf.size());
        }
      }

      std::vector<std::vector<double>> thread_latencies(nthreads);
      for (int t = 0; t < nthreads; ++t) {
        thread_latencies[t].reserve(keys_per_thread * cfg.measure_iters);
      }

      auto wall_start = Clock::now();
      for (size_t m = 0; m < cfg.measure_iters; ++m) {
        std::vector<std::thread> threads;
        for (int t = 0; t < nthreads; ++t) {
          threads.emplace_back([&, t]() {
            std::vector<char> buf(cfg.value_size);
            size_t start = t * keys_per_thread;
            size_t end = start + keys_per_thread;
            for (size_t i = start; i < end; ++i) {
              auto t0 = Clock::now();
              client.GetIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
              auto t1 = Clock::now();
              thread_latencies[t].push_back(
                  std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
          });
        }
        for (auto& th : threads) th.join();
      }
      auto wall_end = Clock::now();

      std::vector<double> all_lat;
      for (auto& tl : thread_latencies) {
        all_lat.insert(all_lat.end(), tl.begin(), tl.end());
      }

      BenchResult r;
      r.name = "Concurrent Get";
      r.variant = variant;
      r.ops = keys_per_thread * static_cast<size_t>(nthreads) * cfg.measure_iters;
      r.bytes = r.ops * cfg.value_size;
      r.elapsed_sec = std::chrono::duration<double>(wall_end - wall_start).count();
      ComputeLatencyStats(all_lat, r);
      PrintResult(r);
      results.push_back(r);
    }
  }
}

// ---------------------------------------------------------------------------
// H. Leader Mode: sync vs async copy
// ---------------------------------------------------------------------------
static void BenchLeaderMode(const BenchConfig& cfg, const std::vector<std::string>& keys,
                            const std::vector<std::vector<char>>& values,
                            std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "Leader")) return;

  PrintHeader("Leader Mode (sync vs async copy)");

  auto run_mode = [&](bool async_copy, const std::string& label) {
    ScopedTempDir tmp(cfg.base_dir + "/leader_" + label);

    UMBPConfig ucfg;
    ucfg.dram.capacity_bytes = cfg.dram_capacity;
    ucfg.ssd.enabled = true;
    ucfg.ssd.storage_dir = tmp.path + "/ssd";
    ucfg.ssd.capacity_bytes = cfg.ssd_capacity;

    ucfg.ssd.io.backend = UMBPIoBackend::PThread;
    ucfg.ssd.durability.mode = UMBPDurabilityMode::Relaxed;
    ucfg.ssd.segment_size_bytes = cfg.segment_size;
    ucfg.role = UMBPRole::SharedSSDLeader;
    ucfg.copy_pipeline.async_enabled = async_copy;
    ucfg.eviction.auto_promote_on_read = false;

    fs::create_directories(ucfg.ssd.storage_dir);

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      UMBPClient client(ucfg);
      for (size_t i = 0; i < keys.size(); ++i) {
        client.Put(keys[i], values[i].data(), values[i].size());
      }
    }

    std::vector<double> latencies;
    latencies.reserve(keys.size() * cfg.measure_iters);

    auto wall_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      UMBPClient client(ucfg);
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        client.Put(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
      // Destructor waits for async drain
    }
    auto wall_end = Clock::now();

    BenchResult r;
    r.name = "Leader Put";
    r.variant = label;
    r.ops = keys.size() * cfg.measure_iters;
    r.bytes = r.ops * cfg.value_size;
    r.elapsed_sec = std::chrono::duration<double>(wall_end - wall_start).count();
    ComputeLatencyStats(latencies, r);
    PrintResult(r);
    results.push_back(r);
  };

  run_mode(false, "sync copy");
  run_mode(true, "async copy");
}

// ---------------------------------------------------------------------------
// I. Capacity Pressure
// ---------------------------------------------------------------------------
static void BenchCapacityPressure(const BenchConfig& cfg, const std::vector<std::string>& keys,
                                  const std::vector<std::vector<char>>& values,
                                  std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "Capacity")) return;

  PrintHeader("Capacity Pressure");

  // DRAM capacity = 50% of total data => second half triggers eviction
  size_t pressure_dram = keys.size() * cfg.value_size / 2;
  size_t half = keys.size() / 2;
  if (half == 0) return;

  ScopedTempDir tmp(cfg.base_dir + "/pressure");

  UMBPConfig ucfg;
  ucfg.dram.capacity_bytes = pressure_dram;
  ucfg.ssd.enabled = true;
  ucfg.ssd.storage_dir = tmp.path + "/ssd";
  ucfg.ssd.capacity_bytes = cfg.ssd_capacity;
  ucfg.ssd.io.backend = UMBPIoBackend::PThread;
  ucfg.ssd.durability.mode = UMBPDurabilityMode::Relaxed;
  ucfg.ssd.segment_size_bytes = cfg.segment_size;
  ucfg.role = UMBPRole::Standalone;
  ucfg.copy_pipeline.async_enabled = false;
  ucfg.eviction.auto_promote_on_read = false;

  fs::create_directories(ucfg.ssd.storage_dir);

  // No pressure: first half, DRAM not full
  {
    UMBPClient client(ucfg);

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      client.Clear();
      for (size_t i = 0; i < half; ++i) {
        client.Put(keys[i], values[i].data(), values[i].size());
      }
    }
    client.Clear();

    std::vector<double> latencies;
    latencies.reserve(half * cfg.measure_iters);

    auto wall_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      client.Clear();
      for (size_t i = 0; i < half; ++i) {
        auto t0 = Clock::now();
        client.Put(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto wall_end = Clock::now();

    BenchResult r;
    r.name = "Capacity Put";
    r.variant = "no pressure";
    r.ops = half * cfg.measure_iters;
    r.bytes = r.ops * cfg.value_size;
    r.elapsed_sec = std::chrono::duration<double>(wall_end - wall_start).count();
    ComputeLatencyStats(latencies, r);
    PrintResult(r);
    results.push_back(r);
  }

  // Under pressure: write all keys, second half triggers eviction
  {
    UMBPClient client(ucfg);

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      client.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        client.Put(keys[i], values[i].data(), values[i].size());
      }
    }
    client.Clear();

    std::vector<double> latencies;
    latencies.reserve(half * cfg.measure_iters);

    auto wall_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      client.Clear();
      // Write first half to fill DRAM
      for (size_t i = 0; i < half; ++i) {
        client.Put(keys[i], values[i].data(), values[i].size());
      }
      // Write second half — triggers eviction + demotion
      for (size_t i = half; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        client.Put(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto wall_end = Clock::now();

    BenchResult r;
    r.name = "Capacity Put";
    r.variant = "under pressure";
    r.ops = (keys.size() - half) * cfg.measure_iters;
    r.bytes = r.ops * cfg.value_size;
    r.elapsed_sec = std::chrono::duration<double>(wall_end - wall_start).count();
    ComputeLatencyStats(latencies, r);
    PrintResult(r);
    results.push_back(r);
  }
}

// ---------------------------------------------------------------------------
// CLI parsing & profiles
// ---------------------------------------------------------------------------
static void ApplyProfile(BenchConfig& cfg, const std::string& profile) {
  if (profile == "small") {
    cfg.num_keys = 200;
    cfg.value_size = 1024;
    cfg.batch_size = 16;
    cfg.dram_capacity = 4ULL * 1024 * 1024;
    cfg.ssd_capacity = 16ULL * 1024 * 1024;
    cfg.segment_size = 4ULL * 1024 * 1024;
    cfg.thread_counts = {1, 2};
  } else if (profile == "medium") {
    cfg.num_keys = 1000;
    cfg.value_size = 4096;
    cfg.batch_size = 64;
    cfg.dram_capacity = 64ULL * 1024 * 1024;
    cfg.ssd_capacity = 256ULL * 1024 * 1024;
    cfg.segment_size = 64ULL * 1024 * 1024;
    cfg.thread_counts = {1, 2, 4, 8};
  } else if (profile == "large") {
    cfg.num_keys = 10000;
    cfg.value_size = 64 * 1024;
    cfg.batch_size = 128;
    cfg.dram_capacity = 512ULL * 1024 * 1024;
    cfg.ssd_capacity = 2ULL * 1024 * 1024 * 1024;
    cfg.segment_size = 256ULL * 1024 * 1024;
    cfg.thread_counts = {1, 2, 4, 8};
  } else {
    std::cerr << "Unknown profile: " << profile << std::endl;
    std::exit(1);
  }
}

static void PrintUsage(const char* argv0) {
  std::printf(
      "Usage: %s [OPTIONS]\n"
      "  --profile <small|medium|large>   Preset config (default: medium)\n"
      "  --num-keys N                     Keys per scenario\n"
      "  --value-size N                   Value size in bytes\n"
      "  --batch-size N                   Batch size\n"
      "  --iters N                        Measurement iterations\n"
      "  --filter SUBSTRING               Run only matching scenarios\n"
      "  --dir PATH                       Temp directory path\n"
      "  -h, --help                       Help\n",
      argv0);
}

static BenchConfig ParseArgs(int argc, char* argv[]) {
  BenchConfig cfg;
  std::string profile = "medium";
  bool profile_set = false;

  // Track which fields the user explicitly overrides
  bool override_num_keys = false;
  bool override_value_size = false;
  bool override_batch_size = false;

  size_t user_num_keys = 0;
  size_t user_value_size = 0;
  size_t user_batch_size = 0;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else if (arg == "--profile" && i + 1 < argc) {
      profile = argv[++i];
      profile_set = true;
    } else if (arg == "--num-keys" && i + 1 < argc) {
      user_num_keys = std::stoull(argv[++i]);
      override_num_keys = true;
    } else if (arg == "--value-size" && i + 1 < argc) {
      user_value_size = std::stoull(argv[++i]);
      override_value_size = true;
    } else if (arg == "--batch-size" && i + 1 < argc) {
      user_batch_size = std::stoull(argv[++i]);
      override_batch_size = true;
    } else if (arg == "--iters" && i + 1 < argc) {
      cfg.measure_iters = std::stoull(argv[++i]);
    } else if (arg == "--filter" && i + 1 < argc) {
      cfg.filter = argv[++i];
    } else if (arg == "--dir" && i + 1 < argc) {
      cfg.base_dir = argv[++i];
    } else {
      std::cerr << "Unknown option: " << arg << std::endl;
      PrintUsage(argv[0]);
      std::exit(1);
    }
  }

  // Apply profile first, then override individual fields
  ApplyProfile(cfg, profile);
  if (override_num_keys) cfg.num_keys = user_num_keys;
  if (override_value_size) cfg.value_size = user_value_size;
  if (override_batch_size) cfg.batch_size = user_batch_size;

  return cfg;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  BenchConfig cfg = ParseArgs(argc, argv);

  std::printf("UMBP Benchmark\n");
  std::printf("  num_keys     = %zu\n", cfg.num_keys);
  std::printf("  value_size   = %zu bytes\n", cfg.value_size);
  std::printf("  batch_size   = %zu\n", cfg.batch_size);
  std::printf("  warmup_iters = %zu\n", cfg.warmup_iters);
  std::printf("  measure_iters= %zu\n", cfg.measure_iters);
  std::printf("  dram_capacity= %zu bytes\n", cfg.dram_capacity);
  std::printf("  ssd_capacity = %zu bytes\n", cfg.ssd_capacity);
  std::printf("  base_dir     = %s\n", cfg.base_dir.c_str());
  if (!cfg.filter.empty()) {
    std::printf("  filter       = %s\n", cfg.filter.c_str());
  }
  std::printf("  threads      =");
  for (int t : cfg.thread_counts) std::printf(" %d", t);
  std::printf("\n");

  // Pre-generate data
  std::printf("\nGenerating %zu keys x %zu bytes...\n", cfg.num_keys, cfg.value_size);
  auto keys = GenerateKeys(cfg.num_keys);
  auto values = GenerateValues(cfg.num_keys, cfg.value_size);
  std::printf("Data generation complete.\n");

  std::vector<BenchResult> results;

  // Run all scenarios
  BenchDRAMTier(cfg, keys, values, results);
  BenchSSDTier(cfg, keys, values, results);
  BenchBatchWrite(cfg, keys, values, results);
  BenchCopyToSSD(cfg, keys, values, results);
  BenchIOBackend(cfg, keys, values, results);
  BenchDurability(cfg, keys, values, results);
  BenchConcurrent(cfg, keys, values, results);
  BenchLeaderMode(cfg, keys, values, results);
  BenchCapacityPressure(cfg, keys, values, results);

  // Summary
  std::printf("\n=== Summary ===\n");
  std::printf("%-36s %-20s %12s %10s\n", "Benchmark", "Variant", "ops/s", "MB/s");
  std::printf("%s\n", std::string(36 + 20 + 12 + 10 + 3, '-').c_str());
  for (const auto& r : results) {
    std::printf("%-36s %-20s %12.0f %10.1f\n", r.name.c_str(), r.variant.c_str(),
                r.throughput_ops_sec(), r.throughput_mb_sec());
  }

  std::printf("\nDone. %zu benchmarks completed.\n", results.size());
  return 0;
}
