// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// UMBP SPDK Benchmark Tool
//
// Mirror of bench_umbp.cpp with SPDK NVMe backend via SpdkProxyTier.
// Scenarios that are POSIX-specific (IO Backend, Durability, StorageIoDriver)
// are skipped; all remaining scenarios use the SPDK proxy path.
//
// Requires: UMBP_SPDK_NVME_PCI environment variable set.
//
// Usage:
//   UMBP_SPDK_NVME_PCI=0000:88:00.0 ./bench_umbp_spdk [OPTIONS]
//     --profile <small|medium|large>   Preset config (default: medium)
//     --num-keys N                     Keys per scenario
//     --value-size N                   Value size in bytes
//     --batch-size N                   Batch size
//     --iters N                        Measurement iterations
//     --filter SUBSTRING               Run only matching scenarios
//     -h, --help                       Help

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#endif

#include "umbp/common/config.h"
#include "umbp/common/storage_tier.h"
#include "umbp/local/storage/local_storage_manager.h"
#include "umbp/local/storage/tier_backend.h"
#include "umbp/local/umbp_client.h"

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
  std::vector<int> thread_counts = {1, 2, 4, 8};
  std::string filter;
};

// ---------------------------------------------------------------------------
// E2E Config
// ---------------------------------------------------------------------------
enum class E2EModelMode { MHA, MLA };

struct E2EConfig {
  E2EModelMode mode = E2EModelMode::MLA;
  size_t num_layers = 61;
  size_t num_kv_heads = 8;
  size_t head_dim = 128;
  size_t kv_lora_rank = 512;
  size_t qk_rope_head_dim = 64;
  std::string kv_cache_dtype = "bf16";
  size_t page_size = 1;
  size_t num_pages = 512;
  size_t batch_pages = 128;
  double dedup_ratio = 0.5;
  int prefix_depth_base = 10;

  size_t DtypeSize() const {
    if (kv_cache_dtype == "fp8_e4m3" || kv_cache_dtype == "fp8_e5m2" ||
        kv_cache_dtype == "fp8")
      return 1;
    return 2;
  }
  size_t KvCacheDim() const { return kv_lora_rank + qk_rope_head_dim; }
  size_t ValueSizePerKey() const {
    if (mode == E2EModelMode::MLA)
      return num_layers * page_size * KvCacheDim() * DtypeSize();
    return num_layers * page_size * num_kv_heads * head_dim * DtypeSize();
  }
  size_t KeysPerPage() const { return (mode == E2EModelMode::MLA) ? 1 : 2; }
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
    return elapsed_sec > 0.0
               ? static_cast<double>(bytes) / (1024.0 * 1024.0) / elapsed_sec
               : 0.0;
  }
};

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------
static std::vector<std::string> GenerateKeys(size_t n) {
  std::vector<std::string> keys(n);
  for (size_t i = 0; i < n; ++i) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "spdk_%08zu", i);
    keys[i] = buf;
  }
  return keys;
}

static std::vector<std::vector<char>> GenerateValues(size_t n, size_t sz) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<std::vector<char>> values(n);
  for (size_t i = 0; i < n; ++i) {
    values[i].resize(sz);
    for (size_t j = 0; j < sz; ++j)
      values[i][j] = static_cast<char>(dist(rng));
  }
  return values;
}

// ---------------------------------------------------------------------------
// Batch descriptors
// ---------------------------------------------------------------------------
struct WriteBatchDesc {
  std::vector<std::string> keys;
  std::vector<const void*> data_ptrs;
  std::vector<size_t> sizes;
};

struct ReadBatchDesc {
  std::vector<std::string> keys;
  std::vector<uintptr_t> dst_ptrs;
  std::vector<size_t> sizes;
};

static std::vector<WriteBatchDesc>
BuildWriteBatches(const std::vector<std::string>& all_keys,
                  const std::vector<std::vector<char>>& values,
                  size_t batch_size) {
  std::vector<WriteBatchDesc> descs;
  for (size_t i = 0; i < all_keys.size(); i += batch_size) {
    size_t end = std::min(i + batch_size, all_keys.size());
    WriteBatchDesc d;
    d.keys.assign(all_keys.begin() + i, all_keys.begin() + end);
    for (size_t j = i; j < end; ++j) {
      d.data_ptrs.push_back(values[j].data());
      d.sizes.push_back(values[j].size());
    }
    descs.push_back(std::move(d));
  }
  return descs;
}

static std::vector<ReadBatchDesc>
BuildReadBatches(const std::vector<std::string>& all_keys,
                 const std::vector<uintptr_t>& all_ptrs,
                 const std::vector<size_t>& all_sizes, size_t batch_size) {
  std::vector<ReadBatchDesc> descs;
  for (size_t i = 0; i < all_keys.size(); i += batch_size) {
    size_t end = std::min(i + batch_size, all_keys.size());
    ReadBatchDesc d;
    d.keys.assign(all_keys.begin() + i, all_keys.begin() + end);
    d.dst_ptrs.assign(all_ptrs.begin() + i, all_ptrs.begin() + end);
    d.sizes.assign(all_sizes.begin() + i, all_sizes.begin() + end);
    descs.push_back(std::move(d));
  }
  return descs;
}

static std::vector<std::vector<std::string>>
BuildKeyBatches(const std::vector<std::string>& all_keys, size_t batch_size) {
  std::vector<std::vector<std::string>> descs;
  for (size_t i = 0; i < all_keys.size(); i += batch_size) {
    size_t end = std::min(i + batch_size, all_keys.size());
    descs.emplace_back(all_keys.begin() + i, all_keys.begin() + end);
  }
  return descs;
}

// ---------------------------------------------------------------------------
// E2E helpers
// ---------------------------------------------------------------------------
struct E2EKeyGenerator {
  E2EModelMode mode;
  size_t tp_rank = 0;
  size_t pp_rank = 0;
  size_t pp_size = 1;

  std::string MlaSuffix() const {
    return (pp_size > 1) ? std::to_string(pp_rank) : "";
  }
  std::string MhaSuffix() const {
    if (pp_size > 1)
      return std::to_string(tp_rank) + "_" + std::to_string(pp_rank);
    return std::to_string(tp_rank);
  }

  void KeysForPage(size_t page_idx, std::vector<std::string>& out) const {
    char hash[32];
    std::snprintf(hash, sizeof(hash), "e2e_%08zu", page_idx);
    if (mode == E2EModelMode::MLA) {
      out.push_back(std::string(hash) + "_" + MlaSuffix() + "_k");
    } else {
      std::string suffix = MhaSuffix();
      out.push_back(std::string(hash) + "_" + suffix + "_k");
      out.push_back(std::string(hash) + "_" + suffix + "_v");
    }
  }

  std::vector<std::string> KeysForPages(size_t start, size_t count) const {
    std::vector<std::string> keys;
    size_t kpp = (mode == E2EModelMode::MLA) ? 1 : 2;
    keys.reserve(count * kpp);
    for (size_t i = 0; i < count; ++i) KeysForPage(start + i, keys);
    return keys;
  }
};

struct E2EHostBuffer {
  std::vector<char> data;
  size_t value_size;
  size_t keys_per_page;
  size_t num_pages;
  size_t v_offset;

  E2EHostBuffer(size_t np, size_t vs, size_t kpp)
      : value_size(vs), keys_per_page(kpp), num_pages(np) {
    v_offset = np * vs;
    data.resize(np * kpp * vs);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& b : data) b = static_cast<char>(dist(rng));
  }

  void GetBatchMeta(size_t start, size_t count, std::vector<uintptr_t>& ptrs,
                    std::vector<size_t>& sizes) const {
    ptrs.clear();
    sizes.clear();
    auto base = reinterpret_cast<uintptr_t>(data.data());
    for (size_t i = 0; i < count; ++i) {
      size_t page = start + i;
      uintptr_t k_ptr = base + page * value_size;
      ptrs.push_back(k_ptr);
      sizes.push_back(value_size);
      if (keys_per_page == 2) {
        ptrs.push_back(k_ptr + v_offset);
        sizes.push_back(value_size);
      }
    }
  }

  static void MakeReadMeta(size_t count, size_t kpp, size_t vs,
                           std::vector<char>& buf,
                           std::vector<uintptr_t>& ptrs,
                           std::vector<size_t>& sizes) {
    buf.assign(count * kpp * vs, 0);
    ptrs.clear();
    sizes.clear();
    auto base = reinterpret_cast<uintptr_t>(buf.data());
    size_t rv = count * vs;
    for (size_t i = 0; i < count; ++i) {
      uintptr_t k = base + i * vs;
      ptrs.push_back(k);
      sizes.push_back(vs);
      if (kpp == 2) {
        ptrs.push_back(k + rv);
        sizes.push_back(vs);
      }
    }
  }
};

static std::vector<int> GenerateDepths(const E2EConfig& e2e, size_t start,
                                       size_t count) {
  std::vector<int> depths;
  depths.reserve(count * e2e.KeysPerPage());
  for (size_t i = 0; i < count; ++i) {
    int d = e2e.prefix_depth_base + static_cast<int>(start + i);
    depths.push_back(d);
    if (e2e.mode == E2EModelMode::MHA) depths.push_back(d);
  }
  return depths;
}

// ---------------------------------------------------------------------------
// Latency / result helpers
// ---------------------------------------------------------------------------
static void ComputeLatencyStats(std::vector<double>& lat, BenchResult& r) {
  if (lat.empty()) return;
  std::sort(lat.begin(), lat.end());
  size_t n = lat.size();
  r.lat_min_us = lat.front();
  r.lat_max_us = lat.back();
  r.lat_avg_us =
      std::accumulate(lat.begin(), lat.end(), 0.0) / static_cast<double>(n);
  r.lat_p50_us = lat[n * 50 / 100];
  r.lat_p95_us = lat[n * 95 / 100];
  r.lat_p99_us = lat[std::min(n * 99 / 100, n - 1)];
}

static double LatencyElapsed(const std::vector<double>& lat) {
  return std::accumulate(lat.begin(), lat.end(), 0.0) / 1e6;
}

static void PrintHeader(const std::string& section) {
  std::printf("\n=== %s ===\n", section.c_str());
  std::printf("%-36s %-20s %8s %10s %9s %9s %9s %9s %9s %9s %12s\n",
              "Benchmark", "Variant", "Ops", "MB/s", "min(us)", "avg(us)",
              "p50(us)", "p95(us)", "p99(us)", "max(us)", "ops/s");
  std::printf("%s\n", std::string(140, '-').c_str());
}

static void PrintResult(const BenchResult& r) {
  std::printf("%-36s %-20s %8zu %10.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f "
              "%12.0f\n",
              r.name.c_str(), r.variant.c_str(), r.ops, r.throughput_mb_sec(),
              r.lat_min_us, r.lat_avg_us, r.lat_p50_us, r.lat_p95_us,
              r.lat_p99_us, r.lat_max_us, r.throughput_ops_sec());
}

static void RecordResult(const std::string& name, const std::string& variant,
                         size_t ops, size_t bytes, double elapsed_sec,
                         std::vector<double>& lat,
                         std::vector<BenchResult>& results) {
  BenchResult r;
  r.name = name;
  r.variant = variant;
  r.ops = ops;
  r.bytes = bytes;
  r.elapsed_sec = elapsed_sec;
  ComputeLatencyStats(lat, r);
  PrintResult(r);
  results.push_back(r);
}

static bool ShouldRun(const BenchConfig& cfg, const std::string& name) {
  if (cfg.filter.empty()) return true;
  return name.find(cfg.filter) != std::string::npos;
}

// ---------------------------------------------------------------------------
// SPDK Config helpers
// ---------------------------------------------------------------------------
static UMBPConfig MakeSpdkLeaderConfig(const BenchConfig& cfg) {
  auto ucfg = UMBPConfig::FromEnvironment();
  ucfg.dram.capacity_bytes = cfg.dram_capacity;
  ucfg.role = UMBPRole::SharedSSDLeader;
  ucfg.copy_pipeline.async_enabled = false;
  ucfg.eviction.auto_promote_on_read = false;
  return ucfg;
}

static UMBPConfig MakeSpdkFollowerConfig() {
  auto ucfg = UMBPConfig::FromEnvironment();
  ucfg.role = UMBPRole::SharedSSDFollower;
  return ucfg;
}

// ---------------------------------------------------------------------------
// A. SPDK Tier Benchmarks (Write / Read / ReadBatch)
// ---------------------------------------------------------------------------
static void BenchSpdkTier(const BenchConfig& cfg,
                          const std::vector<std::string>& keys,
                          const std::vector<std::vector<char>>& values,
                          std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "SPDK Tier")) return;
  PrintHeader("SPDK Tier");

  UMBPClient client(MakeSpdkLeaderConfig(cfg));
  auto* ssd = client.Storage().GetTier(StorageTier::LOCAL_SSD);
  if (!ssd) {
    std::printf("[SKIP] SPDK SSD tier not available\n");
    return;
  }

  // Write
  {
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      ssd->Clear();
      for (size_t i = 0; i < keys.size(); ++i)
        ssd->Write(keys[i], values[i].data(), values[i].size());
    }
    ssd->Clear();

    std::vector<double> lat;
    lat.reserve(keys.size() * cfg.measure_iters);
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      ssd->Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        ssd->Write(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        lat.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    RecordResult("SPDK Tier Write", "single-key",
                 keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size,
                 LatencyElapsed(lat), lat, results);
  }

  // Read
  {
    ssd->Clear();
    for (size_t i = 0; i < keys.size(); ++i)
      ssd->Write(keys[i], values[i].data(), values[i].size());

    std::vector<char> buf(cfg.value_size);
    for (size_t w = 0; w < cfg.warmup_iters; ++w)
      for (size_t i = 0; i < keys.size(); ++i)
        ssd->ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()),
                         buf.size());

    std::vector<double> lat;
    lat.reserve(keys.size() * cfg.measure_iters);
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        ssd->ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()),
                         buf.size());
        auto t1 = Clock::now();
        lat.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    RecordResult("SPDK Tier Read", "single-key",
                 keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size,
                 LatencyElapsed(lat), lat, results);
  }

  // ReadBatch
  {
    ssd->Clear();
    for (size_t i = 0; i < keys.size(); ++i)
      ssd->Write(keys[i], values[i].data(), values[i].size());

    std::vector<std::vector<char>> bufs(keys.size(),
                                        std::vector<char>(cfg.value_size));
    std::vector<uintptr_t> ptrs(keys.size());
    std::vector<size_t> sizes(keys.size(), cfg.value_size);
    for (size_t i = 0; i < keys.size(); ++i)
      ptrs[i] = reinterpret_cast<uintptr_t>(bufs[i].data());
    auto rdescs = BuildReadBatches(keys, ptrs, sizes, cfg.batch_size);

    for (size_t w = 0; w < cfg.warmup_iters; ++w)
      for (const auto& d : rdescs)
        ssd->ReadBatchIntoPtr(d.keys, d.dst_ptrs, d.sizes);

    std::vector<double> lat;
    lat.reserve(rdescs.size() * cfg.measure_iters);
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (const auto& d : rdescs) {
        auto t0 = Clock::now();
        ssd->ReadBatchIntoPtr(d.keys, d.dst_ptrs, d.sizes);
        auto t1 = Clock::now();
        lat.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    RecordResult("SPDK Tier ReadBatch",
                 "bs=" + std::to_string(cfg.batch_size),
                 keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size,
                 LatencyElapsed(lat), lat, results);
  }
}

// ---------------------------------------------------------------------------
// B. Batch vs Single Write
// ---------------------------------------------------------------------------
static void BenchSpdkBatchWrite(const BenchConfig& cfg,
                                const std::vector<std::string>& keys,
                                const std::vector<std::vector<char>>& values,
                                std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "Batch")) return;
  PrintHeader("SPDK Batch vs Single Write");

  UMBPClient client(MakeSpdkLeaderConfig(cfg));
  auto* ssd = client.Storage().GetTier(StorageTier::LOCAL_SSD);
  if (!ssd) {
    std::printf("[SKIP] SPDK SSD tier not available\n");
    return;
  }

  // Sequential
  {
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      ssd->Clear();
      for (size_t i = 0; i < keys.size(); ++i)
        ssd->Write(keys[i], values[i].data(), values[i].size());
    }
    ssd->Clear();

    std::vector<double> lat;
    lat.reserve(keys.size() * cfg.measure_iters);
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      ssd->Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        ssd->Write(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        lat.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    RecordResult("SPDK Write", "sequential", keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size,
                 LatencyElapsed(lat), lat, results);
  }

  // BatchWrite
  {
    auto wdescs = BuildWriteBatches(keys, values, cfg.batch_size);
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      ssd->Clear();
      for (const auto& d : wdescs)
        ssd->BatchWrite(d.keys, d.data_ptrs, d.sizes);
    }
    ssd->Clear();

    std::vector<double> lat;
    lat.reserve(wdescs.size() * cfg.measure_iters);
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      ssd->Clear();
      for (const auto& d : wdescs) {
        auto t0 = Clock::now();
        ssd->BatchWrite(d.keys, d.data_ptrs, d.sizes);
        auto t1 = Clock::now();
        lat.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    RecordResult("SPDK Write",
                 "BatchWrite(bs=" + std::to_string(cfg.batch_size) + ")",
                 keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size,
                 LatencyElapsed(lat), lat, results);
  }
}

// ---------------------------------------------------------------------------
// C. Batch vs Single Read
// ---------------------------------------------------------------------------
static void BenchSpdkBatchRead(const BenchConfig& cfg,
                               const std::vector<std::string>& keys,
                               const std::vector<std::vector<char>>& values,
                               std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "Batch")) return;
  PrintHeader("SPDK Batch vs Single Read");

  UMBPClient client(MakeSpdkLeaderConfig(cfg));
  auto* ssd = client.Storage().GetTier(StorageTier::LOCAL_SSD);
  if (!ssd) {
    std::printf("[SKIP] SPDK SSD tier not available\n");
    return;
  }

  ssd->Clear();
  for (size_t i = 0; i < keys.size(); ++i)
    ssd->Write(keys[i], values[i].data(), values[i].size());

  // Sequential
  {
    std::vector<char> buf(cfg.value_size);
    for (size_t w = 0; w < cfg.warmup_iters; ++w)
      for (size_t i = 0; i < keys.size(); ++i)
        ssd->ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()),
                         buf.size());

    std::vector<double> lat;
    lat.reserve(keys.size() * cfg.measure_iters);
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        ssd->ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()),
                         buf.size());
        auto t1 = Clock::now();
        lat.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    RecordResult("SPDK Read", "sequential", keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size,
                 LatencyElapsed(lat), lat, results);
  }

  // ReadBatch
  {
    std::vector<std::vector<char>> bufs(keys.size(),
                                        std::vector<char>(cfg.value_size));
    std::vector<uintptr_t> ptrs(keys.size());
    std::vector<size_t> sizes(keys.size(), cfg.value_size);
    for (size_t i = 0; i < keys.size(); ++i)
      ptrs[i] = reinterpret_cast<uintptr_t>(bufs[i].data());
    auto rdescs = BuildReadBatches(keys, ptrs, sizes, cfg.batch_size);

    for (size_t w = 0; w < cfg.warmup_iters; ++w)
      for (const auto& d : rdescs)
        ssd->ReadBatchIntoPtr(d.keys, d.dst_ptrs, d.sizes);

    std::vector<double> lat;
    lat.reserve(rdescs.size() * cfg.measure_iters);
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (const auto& d : rdescs) {
        auto t0 = Clock::now();
        ssd->ReadBatchIntoPtr(d.keys, d.dst_ptrs, d.sizes);
        auto t1 = Clock::now();
        lat.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    RecordResult("SPDK Read",
                 "ReadBatch(bs=" + std::to_string(cfg.batch_size) + ")",
                 keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size,
                 LatencyElapsed(lat), lat, results);
  }
}

// ---------------------------------------------------------------------------
// D. CopyToSSD (DRAM → SPDK NVMe)
// ---------------------------------------------------------------------------
static void BenchSpdkCopyToSSD(const BenchConfig& cfg,
                                const std::vector<std::string>& keys,
                                const std::vector<std::vector<char>>& values,
                                std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "CopyToSSD")) return;
  PrintHeader("SPDK CopyToSSD vs CopyToSSDBatch");

  auto ucfg = MakeSpdkLeaderConfig(cfg);
  LocalStorageManager mgr(ucfg);

  for (size_t i = 0; i < keys.size(); ++i)
    mgr.Write(keys[i], values[i].data(), values[i].size(),
              StorageTier::CPU_DRAM);

  // Single
  {
    for (size_t w = 0; w < cfg.warmup_iters; ++w)
      for (size_t i = 0; i < keys.size(); ++i) mgr.CopyToSSD(keys[i]);

    std::vector<double> lat;
    lat.reserve(keys.size() * cfg.measure_iters);
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        mgr.CopyToSSD(keys[i]);
        auto t1 = Clock::now();
        lat.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    RecordResult("SPDK CopyToSSD", "single-key",
                 keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size,
                 LatencyElapsed(lat), lat, results);
  }

  // Batch
  {
    auto kdescs = BuildKeyBatches(keys, cfg.batch_size);
    for (size_t w = 0; w < cfg.warmup_iters; ++w)
      for (const auto& batch : kdescs) mgr.CopyToSSDBatch(batch);

    std::vector<double> lat;
    lat.reserve(kdescs.size() * cfg.measure_iters);
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (const auto& batch : kdescs) {
        auto t0 = Clock::now();
        mgr.CopyToSSDBatch(batch);
        auto t1 = Clock::now();
        lat.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    RecordResult("SPDK CopyToSSD",
                 "batch(bs=" + std::to_string(cfg.batch_size) + ")",
                 keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size,
                 LatencyElapsed(lat), lat, results);
  }
}

// ---------------------------------------------------------------------------
// E. Concurrent Scaling (UMBPClient Put + Get via SPDK proxy)
// ---------------------------------------------------------------------------
static void BenchSpdkConcurrent(const BenchConfig& cfg,
                                const std::vector<std::string>& keys,
                                const std::vector<std::vector<char>>& values,
                                std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "Concurrent")) return;
  PrintHeader("SPDK Concurrent Scaling");

  for (int nthreads : cfg.thread_counts) {
    auto ucfg = MakeSpdkLeaderConfig(cfg);
    UMBPClient client(ucfg);

    size_t kpt = keys.size() / static_cast<size_t>(nthreads);
    if (kpt == 0) continue;
    std::string variant = std::to_string(nthreads) + " threads";

    // Put
    {
      for (size_t w = 0; w < cfg.warmup_iters; ++w) {
        client.Clear();
        for (size_t i = 0; i < keys.size(); ++i)
          client.Put(keys[i], values[i].data(), values[i].size());
      }
      client.Clear();

      std::vector<std::vector<double>> tl(nthreads);
      for (int t = 0; t < nthreads; ++t)
        tl[t].reserve(kpt * cfg.measure_iters);

      for (size_t m = 0; m < cfg.measure_iters; ++m) {
        client.Clear();
        std::vector<std::thread> threads;
        for (int t = 0; t < nthreads; ++t) {
          threads.emplace_back([&, t]() {
            size_t s = t * kpt, e = s + kpt;
            for (size_t i = s; i < e; ++i) {
              auto t0 = Clock::now();
              client.Put(keys[i], values[i].data(), values[i].size());
              auto t1 = Clock::now();
              tl[t].push_back(
                  std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
          });
        }
        for (auto& th : threads) th.join();
      }
      std::vector<double> all;
      for (auto& v : tl) all.insert(all.end(), v.begin(), v.end());
      RecordResult("SPDK Concurrent Put", variant,
                   kpt * nthreads * cfg.measure_iters,
                   kpt * nthreads * cfg.measure_iters * cfg.value_size,
                   LatencyElapsed(all), all, results);
    }

    // Get
    {
      client.Clear();
      for (size_t i = 0; i < keys.size(); ++i)
        client.Put(keys[i], values[i].data(), values[i].size());

      std::vector<std::vector<double>> tl(nthreads);
      for (int t = 0; t < nthreads; ++t)
        tl[t].reserve(kpt * cfg.measure_iters);

      for (size_t m = 0; m < cfg.measure_iters; ++m) {
        std::vector<std::thread> threads;
        for (int t = 0; t < nthreads; ++t) {
          threads.emplace_back([&, t]() {
            std::vector<char> buf(cfg.value_size);
            size_t s = t * kpt, e = s + kpt;
            for (size_t i = s; i < e; ++i) {
              auto t0 = Clock::now();
              client.GetIntoPtr(keys[i],
                                reinterpret_cast<uintptr_t>(buf.data()),
                                buf.size());
              auto t1 = Clock::now();
              tl[t].push_back(
                  std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
          });
        }
        for (auto& th : threads) th.join();
      }
      std::vector<double> all;
      for (auto& v : tl) all.insert(all.end(), v.begin(), v.end());
      RecordResult("SPDK Concurrent Get", variant,
                   kpt * nthreads * cfg.measure_iters,
                   kpt * nthreads * cfg.measure_iters * cfg.value_size,
                   LatencyElapsed(all), all, results);
    }
  }
}

// ---------------------------------------------------------------------------
// F. Leader Mode (sync vs async copy via SPDK)
// ---------------------------------------------------------------------------
static void BenchSpdkLeaderMode(const BenchConfig& cfg,
                                const std::vector<std::string>& keys,
                                const std::vector<std::vector<char>>& values,
                                std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "Leader")) return;
  PrintHeader("SPDK Leader Mode (sync vs async copy)");

  auto run_mode = [&](bool async_copy, const std::string& label) {
    auto ucfg = MakeSpdkLeaderConfig(cfg);
    ucfg.copy_pipeline.async_enabled = async_copy;

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      UMBPClient client(ucfg);
      for (size_t i = 0; i < keys.size(); ++i)
        client.Put(keys[i], values[i].data(), values[i].size());
    }

    std::vector<double> lat;
    lat.reserve(keys.size() * cfg.measure_iters);
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      UMBPClient client(ucfg);
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        client.Put(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        lat.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    RecordResult("SPDK Leader Put", label, keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size,
                 LatencyElapsed(lat), lat, results);
  };

  run_mode(false, "sync copy");
  run_mode(true, "async copy");
}

// ---------------------------------------------------------------------------
// G. Capacity Pressure (DRAM eviction → SPDK NVMe)
// ---------------------------------------------------------------------------
static void
BenchSpdkCapacityPressure(const BenchConfig& cfg,
                          const std::vector<std::string>& keys,
                          const std::vector<std::vector<char>>& values,
                          std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "Capacity")) return;
  PrintHeader("SPDK Capacity Pressure");

  size_t pressure_dram = keys.size() * cfg.value_size / 2;
  size_t half = keys.size() / 2;
  if (half == 0) return;

  auto ucfg = MakeSpdkLeaderConfig(cfg);
  ucfg.dram.capacity_bytes = pressure_dram;

  // No pressure
  {
    UMBPClient client(ucfg);
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      client.Clear();
      for (size_t i = 0; i < half; ++i)
        client.Put(keys[i], values[i].data(), values[i].size());
    }
    client.Clear();

    std::vector<double> lat;
    lat.reserve(half * cfg.measure_iters);
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      client.Clear();
      for (size_t i = 0; i < half; ++i) {
        auto t0 = Clock::now();
        client.Put(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        lat.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    RecordResult("SPDK Capacity Put", "no pressure",
                 half * cfg.measure_iters,
                 half * cfg.measure_iters * cfg.value_size,
                 LatencyElapsed(lat), lat, results);
  }

  // Under pressure
  {
    UMBPClient client(ucfg);
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      client.Clear();
      for (size_t i = 0; i < keys.size(); ++i)
        client.Put(keys[i], values[i].data(), values[i].size());
    }
    client.Clear();

    std::vector<double> lat;
    lat.reserve(half * cfg.measure_iters);
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      client.Clear();
      for (size_t i = 0; i < half; ++i)
        client.Put(keys[i], values[i].data(), values[i].size());
      for (size_t i = half; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        client.Put(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        lat.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    RecordResult("SPDK Capacity Put", "under pressure",
                 (keys.size() - half) * cfg.measure_iters,
                 (keys.size() - half) * cfg.measure_iters * cfg.value_size,
                 LatencyElapsed(lat), lat, results);
  }
}

// ---------------------------------------------------------------------------
// H. E2E UMBPClient (sglang connector) via SPDK
// ---------------------------------------------------------------------------
static void BenchSpdkE2E(const BenchConfig& cfg, const E2EConfig& e2e,
                         std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "E2E")) return;

  size_t value_size = e2e.ValueSizePerKey();
  size_t keys_per_page = e2e.KeysPerPage();
  std::string mode_str = (e2e.mode == E2EModelMode::MLA) ? "MLA" : "MHA";
  std::string vlabel = mode_str + "/" + std::to_string(e2e.batch_pages) +
                        "pg/" + std::to_string(value_size / 1024) + "KB";

  PrintHeader("SPDK E2E UMBPClient (" + vlabel + ")");
  std::printf("  mode=%s  layers=%zu  value_size=%zuB  pages=%zu  "
              "batch=%zu  dedup=%.0f%%\n",
              mode_str.c_str(), e2e.num_layers, value_size, e2e.num_pages,
              e2e.batch_pages, e2e.dedup_ratio * 100);

  E2EKeyGenerator keygen{e2e.mode, 0, 0};
  E2EHostBuffer host_buf(e2e.num_pages, value_size, keys_per_page);
  size_t total_data = e2e.num_pages * keys_per_page * value_size;

  size_t batches_per_iter =
      (e2e.num_pages + e2e.batch_pages - 1) / e2e.batch_pages;
  size_t e2e_iters = cfg.measure_iters;
  if (batches_per_iter > 0 && batches_per_iter * e2e_iters < 100)
    e2e_iters = (100 + batches_per_iter - 1) / batches_per_iter;
  std::printf("  e2e_iters=%zu (%zu batches/iter)\n", e2e_iters,
              batches_per_iter);

  auto FillAll = [&](UMBPClient& c) {
    for (size_t b = 0; b < e2e.num_pages; b += e2e.batch_pages) {
      size_t cnt = std::min(e2e.batch_pages, e2e.num_pages - b);
      auto keys = keygen.KeysForPages(b, cnt);
      std::vector<uintptr_t> ptrs;
      std::vector<size_t> sizes;
      host_buf.GetBatchMeta(b, cnt, ptrs, sizes);
      auto depths = GenerateDepths(e2e, b, cnt);
      c.BatchPutFromPtrWithDepth(keys, ptrs, sizes, depths);
    }
  };

  auto FillAllTimed = [&](UMBPClient& c, size_t np,
                          std::vector<double>& lat) {
    for (size_t b = 0; b < np; b += e2e.batch_pages) {
      size_t cnt = std::min(e2e.batch_pages, np - b);
      auto keys = keygen.KeysForPages(b, cnt);
      std::vector<uintptr_t> ptrs;
      std::vector<size_t> sizes;
      host_buf.GetBatchMeta(b, cnt, ptrs, sizes);
      auto depths = GenerateDepths(e2e, b, cnt);
      auto t0 = Clock::now();
      c.BatchPutFromPtrWithDepth(keys, ptrs, sizes, depths);
      auto t1 = Clock::now();
      lat.push_back(
          std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
  };

  std::vector<char> read_buf;
  std::vector<uintptr_t> read_ptrs;
  std::vector<size_t> read_sizes;

  auto ReadAllTimed = [&](UMBPClient& c, size_t np,
                          std::vector<double>& lat) {
    for (size_t b = 0; b < np; b += e2e.batch_pages) {
      size_t cnt = std::min(e2e.batch_pages, np - b);
      auto keys = keygen.KeysForPages(b, cnt);
      E2EHostBuffer::MakeReadMeta(cnt, keys_per_page, value_size, read_buf,
                                  read_ptrs, read_sizes);
      auto t0 = Clock::now();
      c.BatchGetIntoPtr(keys, read_ptrs, read_sizes);
      auto t1 = Clock::now();
      lat.push_back(
          std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
  };

  auto MakeDramOnly = [&]() -> UMBPConfig {
    auto ucfg = MakeSpdkLeaderConfig(cfg);
    ucfg.dram.capacity_bytes = total_data * 2;
    ucfg.ssd.enabled = false;
    return ucfg;
  };

  // (a) BatchSet — DRAM only (measures UMBPClient overhead, same as POSIX)
  {
    UMBPClient client(MakeDramOnly());
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      client.Clear();
      FillAll(client);
    }
    client.Clear();
    std::vector<double> lat;
    lat.reserve(batches_per_iter * e2e_iters);
    for (size_t m = 0; m < e2e_iters; ++m) {
      client.Clear();
      FillAllTimed(client, e2e.num_pages, lat);
    }
    RecordResult("E2E BatchSet", vlabel, e2e.num_pages * e2e_iters,
                 e2e.num_pages * e2e_iters * keys_per_page * value_size,
                 LatencyElapsed(lat), lat, results);
  }

  // (b) BatchGet — DRAM only
  {
    UMBPClient client(MakeDramOnly());
    FillAll(client);
    std::vector<double> lat;
    lat.reserve(batches_per_iter * e2e_iters);
    for (size_t m = 0; m < e2e_iters; ++m)
      ReadAllTimed(client, e2e.num_pages, lat);
    RecordResult("E2E BatchGet", vlabel, e2e.num_pages * e2e_iters,
                 e2e.num_pages * e2e_iters * keys_per_page * value_size,
                 LatencyElapsed(lat), lat, results);
  }

  // (c) Capacity pressure — DRAM 50%, eviction to SPDK NVMe
  {
    auto ucfg = MakeSpdkLeaderConfig(cfg);
    ucfg.dram.capacity_bytes = total_data / 2;

    // Write under pressure
    {
      UMBPClient client(ucfg);
      for (size_t w = 0; w < cfg.warmup_iters; ++w) {
        client.Clear();
        FillAll(client);
      }
      client.Clear();
      std::vector<double> lat;
      lat.reserve(batches_per_iter * e2e_iters);
      for (size_t m = 0; m < e2e_iters; ++m) {
        client.Clear();
        FillAllTimed(client, e2e.num_pages, lat);
      }
      RecordResult("E2E SPDK Capacity Set", "DRAM 50%",
                   e2e.num_pages * e2e_iters,
                   e2e.num_pages * e2e_iters * keys_per_page * value_size,
                   LatencyElapsed(lat), lat, results);
    }

    // Read back (DRAM+NVMe mixed)
    {
      UMBPClient client(ucfg);
      FillAll(client);
      std::vector<double> lat;
      lat.reserve(batches_per_iter * e2e_iters);
      for (size_t m = 0; m < e2e_iters; ++m)
        ReadAllTimed(client, e2e.num_pages, lat);
      RecordResult("E2E SPDK Capacity Get", "DRAM+NVMe mixed",
                   e2e.num_pages * e2e_iters,
                   e2e.num_pages * e2e_iters * keys_per_page * value_size,
                   LatencyElapsed(lat), lat, results);
    }
  }

  // (d) Leader — sync vs async copy to SPDK NVMe
  {
    auto run_leader = [&](bool async_copy, const std::string& label) {
      auto ucfg = MakeSpdkLeaderConfig(cfg);
      ucfg.dram.capacity_bytes = total_data * 2;
      ucfg.copy_pipeline.async_enabled = async_copy;

      for (size_t w = 0; w < cfg.warmup_iters; ++w) {
        UMBPClient client(ucfg);
        FillAll(client);
      }
      std::vector<double> lat;
      lat.reserve(batches_per_iter * e2e_iters);
      for (size_t m = 0; m < e2e_iters; ++m) {
        UMBPClient client(ucfg);
        FillAllTimed(client, e2e.num_pages, lat);
      }
      RecordResult("E2E SPDK Leader Set", label, e2e.num_pages * e2e_iters,
                   e2e.num_pages * e2e_iters * keys_per_page * value_size,
                   LatencyElapsed(lat), lat, results);
    };
    run_leader(false, "sync copy");
    run_leader(true, "async copy");
  }

  // (e) Follower — pure NVMe read via SharedSSDFollower
  {
    auto leader_cfg = MakeSpdkLeaderConfig(cfg);
    leader_cfg.dram.capacity_bytes = total_data * 2;
    UMBPClient leader(leader_cfg);
    FillAll(leader);

    auto follower_cfg = MakeSpdkFollowerConfig();
    follower_cfg.dram.capacity_bytes = total_data * 2;
    UMBPClient follower(follower_cfg);

    std::vector<double> lat;
    lat.reserve(batches_per_iter * e2e_iters);
    for (size_t m = 0; m < e2e_iters; ++m)
      ReadAllTimed(follower, e2e.num_pages, lat);
    RecordResult("E2E SPDK Follower Get", "NVMe read",
                 e2e.num_pages * e2e_iters,
                 e2e.num_pages * e2e_iters * keys_per_page * value_size,
                 LatencyElapsed(lat), lat, results);
  }
}

// ---------------------------------------------------------------------------
// CLI / profiles
// ---------------------------------------------------------------------------
static void ApplyProfile(BenchConfig& cfg, E2EConfig& e2e,
                         const std::string& profile) {
  if (profile == "small") {
    cfg.num_keys = 200;
    cfg.value_size = 1024;
    cfg.batch_size = 16;
    cfg.dram_capacity = 4ULL * 1024 * 1024;
    cfg.thread_counts = {1, 2};
    e2e.num_pages = 64;
    e2e.batch_pages = 16;
  } else if (profile == "medium") {
    cfg.num_keys = 1000;
    cfg.value_size = 4096;
    cfg.batch_size = 64;
    cfg.dram_capacity = 64ULL * 1024 * 1024;
    cfg.thread_counts = {1, 2, 4, 8};
    e2e.num_pages = 512;
    e2e.batch_pages = 128;
  } else if (profile == "large") {
    cfg.num_keys = 10000;
    cfg.value_size = 64 * 1024;
    cfg.batch_size = 128;
    cfg.dram_capacity = 512ULL * 1024 * 1024;
    cfg.thread_counts = {1, 2, 4, 8};
    e2e.num_pages = 2048;
    e2e.batch_pages = 128;
  } else {
    std::fprintf(stderr, "Unknown profile: %s\n", profile.c_str());
    std::exit(1);
  }
}

static void PrintUsage(const char* argv0) {
  std::printf(
      "Usage: UMBP_SPDK_NVME_PCI=<pci> %s [OPTIONS]\n\n"
      "  --profile <small|medium|large>   Preset (default: medium)\n"
      "  --num-keys N       --value-size N     --batch-size N\n"
      "  --iters N          --warmup-iters N   --filter SUBSTRING\n"
      "  --dram-capacity N\n"
      "  --model <deepseek-v3|llama-70b|...>\n"
      "  --num-pages N      --batch-pages N    --dedup-ratio F\n"
      "  -h, --help\n\n"
      "Skipped (POSIX-only): DRAM Tier, IO Backend, Durability, "
      "StorageIoDriver\n",
      argv0);
}

int main(int argc, char* argv[]) {
  if (!std::getenv("UMBP_SPDK_NVME_PCI")) {
    std::fprintf(stderr,
                 "ERROR: UMBP_SPDK_NVME_PCI not set. Example:\n"
                 "  UMBP_SPDK_NVME_PCI=0000:88:00.0 %s\n",
                 argv[0]);
    return 1;
  }

  BenchConfig cfg;
  E2EConfig e2e;
  std::string profile = "medium";

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      return 0;
    } else if (arg == "--profile" && i + 1 < argc) {
      profile = argv[++i];
    } else if (arg == "--num-keys" && i + 1 < argc) {
      cfg.num_keys = std::stoull(argv[++i]);
    } else if (arg == "--value-size" && i + 1 < argc) {
      cfg.value_size = std::stoull(argv[++i]);
    } else if (arg == "--batch-size" && i + 1 < argc) {
      cfg.batch_size = std::stoull(argv[++i]);
    } else if (arg == "--iters" && i + 1 < argc) {
      cfg.measure_iters = std::stoull(argv[++i]);
    } else if (arg == "--warmup-iters" && i + 1 < argc) {
      cfg.warmup_iters = std::stoull(argv[++i]);
    } else if (arg == "--filter" && i + 1 < argc) {
      cfg.filter = argv[++i];
    } else if (arg == "--dram-capacity" && i + 1 < argc) {
      cfg.dram_capacity = std::stoull(argv[++i]);
    } else if (arg == "--model" && i + 1 < argc) {
      std::string preset = argv[++i];
      if (preset == "deepseek-v3" || preset == "deepseek-r1") {
        e2e.mode = E2EModelMode::MLA;
        e2e.num_layers = 61;
        e2e.kv_lora_rank = 512;
        e2e.qk_rope_head_dim = 64;
      } else if (preset == "llama-70b") {
        e2e.mode = E2EModelMode::MHA;
        e2e.num_layers = 80;
        e2e.num_kv_heads = 8;
        e2e.head_dim = 128;
      } else if (preset == "llama-8b") {
        e2e.mode = E2EModelMode::MHA;
        e2e.num_layers = 32;
        e2e.num_kv_heads = 8;
        e2e.head_dim = 128;
      }
    } else if (arg == "--num-pages" && i + 1 < argc) {
      e2e.num_pages = std::stoull(argv[++i]);
    } else if (arg == "--batch-pages" && i + 1 < argc) {
      e2e.batch_pages = std::stoull(argv[++i]);
    } else if (arg == "--dedup-ratio" && i + 1 < argc) {
      e2e.dedup_ratio = std::stod(argv[++i]);
    }
  }

  ApplyProfile(cfg, e2e, profile);

  std::printf("UMBP SPDK Benchmark\n");
  std::printf("  backend      = SPDK (NVMe: %s)\n",
              std::getenv("UMBP_SPDK_NVME_PCI"));
  std::printf("  num_keys     = %zu\n", cfg.num_keys);
  std::printf("  value_size   = %zu bytes\n", cfg.value_size);
  std::printf("  batch_size   = %zu\n", cfg.batch_size);
  std::printf("  warmup_iters = %zu\n", cfg.warmup_iters);
  std::printf("  measure_iters= %zu\n", cfg.measure_iters);
  std::printf("  dram_capacity= %zu bytes\n", cfg.dram_capacity);
  std::printf("  threads      =");
  for (int t : cfg.thread_counts) std::printf(" %d", t);
  std::printf("\n");

  std::printf("\n[SKIP] DRAM Tier       — memory-only, identical to POSIX\n");
  std::printf("[SKIP] IO Backend      — pthread/io_uring is POSIX I/O layer\n");
  std::printf("[SKIP] Durability      — Strict/Relaxed is POSIX fsync semantics\n");
  std::printf("[SKIP] StorageIoDriver — raw POSIX file I/O, not used by SPDK\n");

  std::printf("\nGenerating %zu keys x %zu bytes...\n", cfg.num_keys,
              cfg.value_size);
  auto keys = GenerateKeys(cfg.num_keys);
  auto values = GenerateValues(cfg.num_keys, cfg.value_size);
  std::printf("Data generation complete.\n");

  std::vector<BenchResult> results;

  BenchSpdkTier(cfg, keys, values, results);
  BenchSpdkBatchWrite(cfg, keys, values, results);
  BenchSpdkBatchRead(cfg, keys, values, results);
  BenchSpdkCopyToSSD(cfg, keys, values, results);
  BenchSpdkConcurrent(cfg, keys, values, results);
  BenchSpdkLeaderMode(cfg, keys, values, results);
  BenchSpdkCapacityPressure(cfg, keys, values, results);
  BenchSpdkE2E(cfg, e2e, results);

  std::printf("\n=== Summary ===\n");
  std::printf("%-36s %-20s %12s %10s\n", "Benchmark", "Variant", "ops/s",
              "MB/s");
  std::printf("%s\n", std::string(80, '-').c_str());
  for (const auto& r : results)
    std::printf("%-36s %-20s %12.0f %10.1f\n", r.name.c_str(),
                r.variant.c_str(), r.throughput_ops_sec(),
                r.throughput_mb_sec());

  std::printf("\nDone. %zu benchmarks completed.\n", results.size());
  return 0;
}
