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

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
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
#include "umbp/local/storage/dram_tier.h"
#include "umbp/local/storage/io/storage_io_driver.h"
#include "umbp/local/storage/local_storage_manager.h"
#include "umbp/local/storage/ssd_tier.h"
#include "umbp/local/umbp_client.h"

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
  UMBPIoBackend ssd_io_backend = UMBPIoBackend::PThread;
  size_t ssd_io_queue_depth = 4096;
  UMBPDurabilityMode ssd_durability_mode = UMBPDurabilityMode::Relaxed;

  std::vector<int> thread_counts = {1, 2, 4, 8};

  std::string base_dir = "/tmp/umbp_bench";
  std::string filter;
};

// ---------------------------------------------------------------------------
// E2E Config — sglang-aligned parameters
// ---------------------------------------------------------------------------
enum class E2EModelMode { MHA, MLA };

struct E2EConfig {
  E2EModelMode mode = E2EModelMode::MLA;

  // Model architecture (defaults: DeepSeek-V3/R1)
  size_t num_layers = 61;
  size_t num_kv_heads = 8;       // MHA only
  size_t head_dim = 128;         // MHA only
  size_t kv_lora_rank = 512;     // MLA only
  size_t qk_rope_head_dim = 64;  // MLA only

  // Common
  std::string kv_cache_dtype = "bf16";  // bf16, fp16, fp8_e4m3, fp8_e5m2
  size_t page_size = 1;                 // tokens per page

  // Bench parameters
  size_t num_pages = 512;
  size_t batch_pages = 128;  // sglang storage_batch_size
  double dedup_ratio = 0.5;
  int prefix_depth_base = 10;

  size_t DtypeSize() const {
    if (kv_cache_dtype == "fp8_e4m3" || kv_cache_dtype == "fp8_e5m2" || kv_cache_dtype == "fp8")
      return 1;
    return 2;  // bf16, fp16
  }

  size_t KvCacheDim() const { return kv_lora_rank + qk_rope_head_dim; }

  size_t ValueSizePerKey() const {
    if (mode == E2EModelMode::MLA)
      return num_layers * page_size * KvCacheDim() * DtypeSize();
    else
      return num_layers * page_size * num_kv_heads * head_dim * DtypeSize();
  }

  size_t KeysPerPage() const { return (mode == E2EModelMode::MLA) ? 1 : 2; }
};

static void ApplyModelPreset(E2EConfig& e2e, const std::string& preset) {
  if (preset == "deepseek-v3" || preset == "deepseek-r1") {
    e2e.mode = E2EModelMode::MLA;
    e2e.num_layers = 61;
    e2e.kv_lora_rank = 512;
    e2e.qk_rope_head_dim = 64;
    e2e.kv_cache_dtype = "bf16";
    e2e.page_size = 1;
  } else if (preset == "deepseek-v2") {
    e2e.mode = E2EModelMode::MLA;
    e2e.num_layers = 60;
    e2e.kv_lora_rank = 512;
    e2e.qk_rope_head_dim = 64;
    e2e.kv_cache_dtype = "bf16";
    e2e.page_size = 1;
  } else if (preset == "llama-70b") {
    e2e.mode = E2EModelMode::MHA;
    e2e.num_layers = 80;
    e2e.num_kv_heads = 8;
    e2e.head_dim = 128;
    e2e.kv_cache_dtype = "bf16";
    e2e.page_size = 1;
  } else if (preset == "llama-8b") {
    e2e.mode = E2EModelMode::MHA;
    e2e.num_layers = 32;
    e2e.num_kv_heads = 8;
    e2e.head_dim = 128;
    e2e.kv_cache_dtype = "bf16";
    e2e.page_size = 1;
  } else {
    std::cerr << "Unknown model preset: " << preset << std::endl;
    std::exit(1);
  }
}

static std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

struct IoBackendSpec {
  UMBPIoBackend backend;
  const char* cli_name;
  const char* display_name;
  const char* path_suffix;
};

static const std::vector<IoBackendSpec>& IoBackendSpecs() {
  static const std::vector<IoBackendSpec> specs = {
      {UMBPIoBackend::PThread, "pthread", "POSIX", "posix"},
      {UMBPIoBackend::IoUring, "io_uring", "io_uring", "iouring"},
  };
  return specs;
}

static const IoBackendSpec& GetIoBackendSpec(UMBPIoBackend backend) {
  for (const auto& spec : IoBackendSpecs()) {
    if (spec.backend == backend) return spec;
  }
  throw std::invalid_argument("Unknown I/O backend enum value");
}

static bool ParseIoBackend(const std::string& text, UMBPIoBackend& backend) {
  std::string lower = ToLower(text);
  if (lower == "pthread" || lower == "posix") {
    backend = UMBPIoBackend::PThread;
    return true;
  }
  if (lower == "io_uring" || lower == "iouring" || lower == "uring") {
    backend = UMBPIoBackend::IoUring;
    return true;
  }
  return false;
}

static const char* DurabilityLabel(UMBPDurabilityMode mode) {
  return (mode == UMBPDurabilityMode::Strict) ? "Strict" : "Relaxed";
}

static bool ParseDurabilityMode(const std::string& text, UMBPDurabilityMode& mode) {
  std::string lower = ToLower(text);
  if (lower == "strict" || lower == "sync") {
    mode = UMBPDurabilityMode::Strict;
    return true;
  }
  if (lower == "relaxed" || lower == "async") {
    mode = UMBPDurabilityMode::Relaxed;
    return true;
  }
  return false;
}

static std::string BackendVariantLabel(UMBPIoBackend backend, size_t queue_depth) {
  return std::string(GetIoBackendSpec(backend).display_name) +
         "(qd=" + std::to_string(queue_depth) + ")";
}

static UMBPConfig MakeBaseSsdConfig(const BenchConfig& cfg) {
  UMBPConfig ucfg;
  ucfg.ssd.enabled = true;
  ucfg.ssd.capacity_bytes = cfg.ssd_capacity;
  ucfg.ssd.io.backend = cfg.ssd_io_backend;
  ucfg.ssd.io.queue_depth = cfg.ssd_io_queue_depth;
  ucfg.ssd.durability.mode = cfg.ssd_durability_mode;
  ucfg.ssd.segment_size_bytes = cfg.segment_size;
  return ucfg;
}

static UMBPConfig MakeBaseSsdConfig(const BenchConfig& cfg, UMBPIoBackend backend,
                                    UMBPDurabilityMode durability, size_t queue_depth) {
  UMBPConfig ucfg = MakeBaseSsdConfig(cfg);
  ucfg.ssd.io.backend = backend;
  ucfg.ssd.io.queue_depth = queue_depth;
  ucfg.ssd.durability.mode = durability;
  return ucfg;
}

static UMBPConfig MakeStandaloneClientConfig(const BenchConfig& cfg, const std::string& storage_dir,
                                             size_t dram_capacity,
                                             size_t ssd_capacity_override = 0) {
  UMBPConfig ucfg = MakeBaseSsdConfig(cfg);
  ucfg.dram.capacity_bytes = dram_capacity;
  ucfg.ssd.enabled = true;
  ucfg.ssd.storage_dir = storage_dir;
  if (ssd_capacity_override > 0) ucfg.ssd.capacity_bytes = ssd_capacity_override;
  ucfg.role = UMBPRole::Standalone;
  ucfg.copy_pipeline.async_enabled = false;
  ucfg.eviction.auto_promote_on_read = false;
  return ucfg;
}

static UMBPConfig MakeLeaderClientConfig(const BenchConfig& cfg, const std::string& storage_dir,
                                         size_t dram_capacity, bool async_copy,
                                         size_t ssd_capacity_override = 0) {
  UMBPConfig ucfg =
      MakeStandaloneClientConfig(cfg, storage_dir, dram_capacity, ssd_capacity_override);
  ucfg.role = UMBPRole::SharedSSDLeader;
  ucfg.copy_pipeline.async_enabled = async_copy;
  return ucfg;
}

static UMBPConfig MakeFollowerClientConfig(const BenchConfig& cfg, const std::string& storage_dir,
                                           size_t dram_capacity, size_t ssd_capacity_override = 0) {
  UMBPConfig ucfg =
      MakeStandaloneClientConfig(cfg, storage_dir, dram_capacity, ssd_capacity_override);
  ucfg.role = UMBPRole::SharedSSDFollower;
  return ucfg;
}

struct ScopedFd {
  int fd = -1;

  ScopedFd() = default;
  explicit ScopedFd(int fd) : fd(fd) {}
  ~ScopedFd() {
    if (fd >= 0) close(fd);
  }

  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;

  ScopedFd(ScopedFd&& other) noexcept : fd(other.fd) { other.fd = -1; }
  ScopedFd& operator=(ScopedFd&& other) noexcept {
    if (this != &other) {
      if (fd >= 0) close(fd);
      fd = other.fd;
      other.fd = -1;
    }
    return *this;
  }
};

static ScopedFd OpenBenchFile(const std::string& path) {
  int fd = open(path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
  if (fd < 0) {
    throw std::runtime_error("Failed to open benchmark file '" + path +
                             "': " + std::string(std::strerror(errno)));
  }
  return ScopedFd(fd);
}

static void EnsureIoOk(const IoStatus& status, const std::string& context) {
  if (!status.ok()) {
    throw std::runtime_error(context + ": " + status.message());
  }
}

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
// E2E helpers — key generation, host buffer, depth computation
// ---------------------------------------------------------------------------

// Generates UMBP keys matching umbp_store.py's _batch_preprocess format.
// Key suffix rules (from umbp_store.py lines 154-160):
//   MHA pp=1: "{hash}_{tp_rank}_k/v"
//   MHA pp>1: "{hash}_{tp_rank}_{pp_rank}_k/v"
//   MLA pp=1: "{hash}__k"           (mla_suffix="" → double underscore)
//   MLA pp>1: "{hash}_{pp_rank}_k"
struct E2EKeyGenerator {
  E2EModelMode mode;
  size_t tp_rank = 0;
  size_t pp_rank = 0;
  size_t pp_size = 1;

  // Pre-compute suffix string matching umbp_store.py's __init__.
  std::string MhaSuffix() const {
    if (pp_size > 1) return std::to_string(tp_rank) + "_" + std::to_string(pp_rank);
    return std::to_string(tp_rank);
  }

  std::string MlaSuffix() const {
    if (pp_size > 1) return std::to_string(pp_rank);
    return "";  // pp_size==1: mla_suffix = ""
  }

  void KeysForPage(size_t page_idx, std::vector<std::string>& out) const {
    char hash[32];
    std::snprintf(hash, sizeof(hash), "e2e_%08zu", page_idx);
    if (mode == E2EModelMode::MLA) {
      // "{hash}_{mla_suffix}_k" — when mla_suffix="" this becomes "{hash}__k"
      out.push_back(std::string(hash) + "_" + MlaSuffix() + "_k");
    } else {
      std::string suffix = MhaSuffix();
      out.push_back(std::string(hash) + "_" + suffix + "_k");
      out.push_back(std::string(hash) + "_" + suffix + "_v");
    }
  }

  std::vector<std::string> KeysForPages(size_t start, size_t count) const {
    std::vector<std::string> keys;
    size_t keys_per_page = (mode == E2EModelMode::MLA) ? 1 : 2;
    keys.reserve(count * keys_per_page);
    for (size_t i = 0; i < count; ++i) {
      KeysForPage(start + i, keys);
    }
    return keys;
  }
};

// Contiguous host buffer simulating sglang's pinned host KV pool.
//
// Matches sglang memory_pool_host.py get_page_buffer_meta() for page_first layout:
//   MHA: [K0 K1 ... Kn | V0 V1 ... Vn]  — K region then V region, separated by v_offset.
//         k_ptr[i] = base + i * value_size
//         v_ptr[i] = k_ptr[i] + v_offset   (v_offset = num_pages * value_size)
//   MLA: [K0 K1 ... Kn]                  — single contiguous K region, no V.
struct E2EHostBuffer {
  std::vector<char> data;
  size_t value_size;
  size_t keys_per_page;  // 2 for MHA, 1 for MLA
  size_t num_pages;
  size_t v_offset;  // byte offset from K region to V region (MHA only)

  E2EHostBuffer(size_t num_pages, size_t value_size, size_t keys_per_page)
      : value_size(value_size), keys_per_page(keys_per_page), num_pages(num_pages) {
    // MHA: v_offset = num_pages * value_size (all K pages, then all V pages).
    v_offset = num_pages * value_size;
    data.resize(num_pages * keys_per_page * value_size);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& b : data) b = static_cast<char>(dist(rng));
  }

  // Fill ptrs/sizes for pages [start, start+count), matching get_page_buffer_meta.
  // MHA output order: [k_ptr0, v_ptr0, k_ptr1, v_ptr1, ...]
  // MLA output order: [k_ptr0, k_ptr1, ...]
  void GetBatchMeta(size_t start, size_t count, std::vector<uintptr_t>& ptrs,
                    std::vector<size_t>& sizes) const {
    ptrs.clear();
    sizes.clear();
    size_t total_keys = count * keys_per_page;
    ptrs.reserve(total_keys);
    sizes.reserve(total_keys);
    auto base = reinterpret_cast<uintptr_t>(data.data());
    for (size_t i = 0; i < count; ++i) {
      size_t page = start + i;
      uintptr_t k_ptr = base + page * value_size;
      ptrs.push_back(k_ptr);
      sizes.push_back(value_size);
      if (keys_per_page == 2) {
        ptrs.push_back(k_ptr + v_offset);  // V region at fixed offset
        sizes.push_back(value_size);
      }
    }
  }

  // Separate read buffer with the same layout for BatchGetIntoPtr.
  static void MakeReadMeta(size_t count, size_t keys_per_page, size_t value_size,
                           std::vector<char>& buf, std::vector<uintptr_t>& ptrs,
                           std::vector<size_t>& sizes) {
    size_t total_bytes = count * keys_per_page * value_size;
    buf.assign(total_bytes, 0);
    ptrs.clear();
    sizes.clear();
    size_t total = count * keys_per_page;
    ptrs.reserve(total);
    sizes.reserve(total);
    auto base = reinterpret_cast<uintptr_t>(buf.data());
    size_t read_v_offset = count * value_size;
    for (size_t i = 0; i < count; ++i) {
      uintptr_t k_ptr = base + i * value_size;
      ptrs.push_back(k_ptr);
      sizes.push_back(value_size);
      if (keys_per_page == 2) {
        ptrs.push_back(k_ptr + read_v_offset);
        sizes.push_back(value_size);
      }
    }
  }
};

// Generate depth values matching umbp_store.py _compute_expanded_depths.
// depth = prefix_depth_base + page_index; duplicated for K/V in MHA.
static std::vector<int> GenerateDepths(const E2EConfig& e2e, size_t start_page, size_t count) {
  std::vector<int> depths;
  depths.reserve(count * e2e.KeysPerPage());
  for (size_t i = 0; i < count; ++i) {
    int depth = e2e.prefix_depth_base + static_cast<int>(start_page + i);
    depths.push_back(depth);
    if (e2e.mode == E2EModelMode::MHA) depths.push_back(depth);  // V same depth as K
  }
  return depths;
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

// Elapsed time from latency vector (sum of per-call microseconds → seconds).
static double LatencyElapsed(const std::vector<double>& latencies) {
  return std::accumulate(latencies.begin(), latencies.end(), 0.0) / 1e6;
}

// Record a benchmark result: compute stats, print, and append to results vector.
static void RecordResult(const std::string& name, const std::string& variant, size_t ops,
                         size_t bytes, double elapsed_sec, std::vector<double>& latencies,
                         std::vector<BenchResult>& results) {
  BenchResult r;
  r.name = name;
  r.variant = variant;
  r.ops = ops;
  r.bytes = bytes;
  r.elapsed_sec = elapsed_sec;
  ComputeLatencyStats(latencies, r);
  PrintResult(r);
  results.push_back(r);
}

// ---------------------------------------------------------------------------
// Scenario filter check
// ---------------------------------------------------------------------------
static bool ShouldRun(const BenchConfig& cfg, const std::string& name) {
  if (cfg.filter.empty()) return true;
  return name.find(cfg.filter) != std::string::npos;
}

// ---------------------------------------------------------------------------
// Pre-generated batch descriptors
//
// Construct once before warmup/measure so that vector slicing and allocation
// stay outside the timed path, keeping throughput numbers pure.
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

static std::vector<WriteBatchDesc> BuildWriteBatches(const std::vector<std::string>& all_keys,
                                                     const std::vector<std::vector<char>>& values,
                                                     size_t batch_size) {
  std::vector<WriteBatchDesc> descs;
  for (size_t i = 0; i < all_keys.size(); i += batch_size) {
    size_t end = std::min(i + batch_size, all_keys.size());
    WriteBatchDesc d;
    d.keys.assign(all_keys.begin() + i, all_keys.begin() + end);
    d.data_ptrs.reserve(end - i);
    d.sizes.reserve(end - i);
    for (size_t j = i; j < end; ++j) {
      d.data_ptrs.push_back(values[j].data());
      d.sizes.push_back(values[j].size());
    }
    descs.push_back(std::move(d));
  }
  return descs;
}

static std::vector<ReadBatchDesc> BuildReadBatches(const std::vector<std::string>& all_keys,
                                                   const std::vector<uintptr_t>& all_ptrs,
                                                   const std::vector<size_t>& all_sizes,
                                                   size_t batch_size) {
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

// Build key-only batch slices (e.g. for CopyToSSDBatch).
static std::vector<std::vector<std::string>> BuildKeyBatches(
    const std::vector<std::string>& all_keys, size_t batch_size) {
  std::vector<std::vector<std::string>> descs;
  for (size_t i = 0; i < all_keys.size(); i += batch_size) {
    size_t end = std::min(i + batch_size, all_keys.size());
    descs.emplace_back(all_keys.begin() + i, all_keys.begin() + end);
  }
  return descs;
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

    RecordResult("DRAM Write", "single-key", keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size, LatencyElapsed(latencies),
                 latencies, results);
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

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        tier.ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("DRAM Read", "single-key", keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size, LatencyElapsed(latencies),
                 latencies, results);
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

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        size_t sz = 0;
        auto t0 = Clock::now();
        tier.ReadPtr(keys[i], &sz);
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("DRAM ReadPtr (zero-copy)", "single-key", keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size, LatencyElapsed(latencies),
                 latencies, results);
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

  UMBPConfig ucfg = MakeBaseSsdConfig(cfg);

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

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        tier.Write(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("SSD Tier Write", "single-key", keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size, LatencyElapsed(latencies),
                 latencies, results);
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

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        tier.ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("SSD Tier Read", "single-key", keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size, LatencyElapsed(latencies),
                 latencies, results);
  }

  // ReadBatch
  {
    ScopedTempDir tmp(cfg.base_dir + "/ssd_tier_read_batch");
    SSDTier tier(tmp.path, cfg.ssd_capacity, ucfg);

    // Pre-fill
    for (size_t i = 0; i < keys.size(); ++i) {
      tier.Write(keys[i], values[i].data(), values[i].size());
    }

    // Prepare per-key read buffers and pre-build batch descriptors.
    std::vector<std::vector<char>> bufs(keys.size(), std::vector<char>(cfg.value_size));
    std::vector<uintptr_t> ptrs(keys.size());
    std::vector<size_t> sizes(keys.size(), cfg.value_size);
    for (size_t i = 0; i < keys.size(); ++i) {
      ptrs[i] = reinterpret_cast<uintptr_t>(bufs[i].data());
    }
    auto rdescs = BuildReadBatches(keys, ptrs, sizes, cfg.batch_size);

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (const auto& d : rdescs) tier.ReadBatchIntoPtr(d.keys, d.dst_ptrs, d.sizes);
    }

    std::vector<double> latencies;
    latencies.reserve(rdescs.size() * cfg.measure_iters);

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (const auto& d : rdescs) {
        auto t0 = Clock::now();
        tier.ReadBatchIntoPtr(d.keys, d.dst_ptrs, d.sizes);
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("SSD Tier ReadBatch", "bs=" + std::to_string(cfg.batch_size),
                 keys.size() * cfg.measure_iters, keys.size() * cfg.measure_iters * cfg.value_size,
                 LatencyElapsed(latencies), latencies, results);
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

  UMBPConfig ucfg = MakeBaseSsdConfig(cfg);

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

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        tier.Write(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("SSD Write", "sequential", keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size, LatencyElapsed(latencies),
                 latencies, results);
  }

  // WriteBatch
  {
    ScopedTempDir tmp(cfg.base_dir + "/batch_batch");
    SSDTier tier(tmp.path, cfg.ssd_capacity, ucfg);

    auto wdescs = BuildWriteBatches(keys, values, cfg.batch_size);

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      tier.Clear();
      for (const auto& d : wdescs) tier.WriteBatch(d.keys, d.data_ptrs, d.sizes);
    }
    tier.Clear();

    std::vector<double> latencies;
    latencies.reserve(wdescs.size() * cfg.measure_iters);

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      tier.Clear();
      for (const auto& d : wdescs) {
        auto t0 = Clock::now();
        tier.WriteBatch(d.keys, d.data_ptrs, d.sizes);
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("SSD Write", "WriteBatch(bs=" + std::to_string(cfg.batch_size) + ")",
                 keys.size() * cfg.measure_iters, keys.size() * cfg.measure_iters * cfg.value_size,
                 LatencyElapsed(latencies), latencies, results);
  }
}

// ---------------------------------------------------------------------------
// C2. Batch vs Single Read
// ---------------------------------------------------------------------------
static void BenchBatchRead(const BenchConfig& cfg, const std::vector<std::string>& keys,
                           const std::vector<std::vector<char>>& values,
                           std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "Batch")) return;

  PrintHeader("Batch vs Single Read");

  UMBPConfig ucfg = MakeBaseSsdConfig(cfg);

  // Sequential single-key read
  {
    ScopedTempDir tmp(cfg.base_dir + "/batch_read_seq");
    SSDTier tier(tmp.path, cfg.ssd_capacity, ucfg);

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

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        tier.ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("SSD Read", "sequential", keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size, LatencyElapsed(latencies),
                 latencies, results);
  }

  // ReadBatchIntoPtr
  {
    ScopedTempDir tmp(cfg.base_dir + "/batch_read_batch");
    SSDTier tier(tmp.path, cfg.ssd_capacity, ucfg);

    for (size_t i = 0; i < keys.size(); ++i) {
      tier.Write(keys[i], values[i].data(), values[i].size());
    }

    std::vector<std::vector<char>> bufs(keys.size(), std::vector<char>(cfg.value_size));
    std::vector<uintptr_t> ptrs(keys.size());
    std::vector<size_t> sizes(keys.size(), cfg.value_size);
    for (size_t i = 0; i < keys.size(); ++i) {
      ptrs[i] = reinterpret_cast<uintptr_t>(bufs[i].data());
    }
    auto rdescs = BuildReadBatches(keys, ptrs, sizes, cfg.batch_size);

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (const auto& d : rdescs) tier.ReadBatchIntoPtr(d.keys, d.dst_ptrs, d.sizes);
    }

    std::vector<double> latencies;
    latencies.reserve(rdescs.size() * cfg.measure_iters);

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (const auto& d : rdescs) {
        auto t0 = Clock::now();
        tier.ReadBatchIntoPtr(d.keys, d.dst_ptrs, d.sizes);
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("SSD Read", "ReadBatch(bs=" + std::to_string(cfg.batch_size) + ")",
                 keys.size() * cfg.measure_iters, keys.size() * cfg.measure_iters * cfg.value_size,
                 LatencyElapsed(latencies), latencies, results);
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

    UMBPConfig ucfg = MakeStandaloneClientConfig(cfg, tmp.path + "/ssd", cfg.dram_capacity);

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

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        mgr.CopyToSSD(keys[i]);
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("CopyToSSD", "single-key", keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size, LatencyElapsed(latencies),
                 latencies, results);
  }

  // Batch CopyToSSDBatch
  {
    ScopedTempDir tmp(cfg.base_dir + "/copy_batch");

    UMBPConfig ucfg = MakeStandaloneClientConfig(cfg, tmp.path + "/ssd", cfg.dram_capacity);

    fs::create_directories(ucfg.ssd.storage_dir);
    LocalStorageManager mgr(ucfg);

    // Pre-fill DRAM
    for (size_t i = 0; i < keys.size(); ++i) {
      mgr.Write(keys[i], values[i].data(), values[i].size(), StorageTier::CPU_DRAM);
    }

    auto kdescs = BuildKeyBatches(keys, cfg.batch_size);

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (const auto& batch : kdescs) mgr.CopyToSSDBatch(batch);
    }

    std::vector<double> latencies;
    latencies.reserve(kdescs.size() * cfg.measure_iters);

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (const auto& batch : kdescs) {
        auto t0 = Clock::now();
        mgr.CopyToSSDBatch(batch);
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("CopyToSSD", "batch(bs=" + std::to_string(cfg.batch_size) + ")",
                 keys.size() * cfg.measure_iters, keys.size() * cfg.measure_iters * cfg.value_size,
                 LatencyElapsed(latencies), latencies, results);
  }
}

// ---------------------------------------------------------------------------
// E. IO Backend: POSIX vs io_uring
// ---------------------------------------------------------------------------
static void BenchIOBackend(const BenchConfig& cfg, const std::vector<std::string>& keys,
                           const std::vector<std::vector<char>>& values,
                           std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "IO Backend")) return;

  PrintHeader("IO Backend Sweep");

  auto run_backend = [&](UMBPIoBackend backend) {
    const auto& spec = GetIoBackendSpec(backend);
    UMBPConfig ucfg =
        MakeBaseSsdConfig(cfg, backend, cfg.ssd_durability_mode, cfg.ssd_io_queue_depth);
    std::string variant = BackendVariantLabel(backend, cfg.ssd_io_queue_depth);

    // Write
    {
      ScopedTempDir tmp(cfg.base_dir + "/io_" + std::string(spec.path_suffix) + "_w");
      std::unique_ptr<SSDTier> tier;
      try {
        tier = std::make_unique<SSDTier>(tmp.path, cfg.ssd_capacity, ucfg);
      } catch (const std::exception& e) {
        std::printf("[SKIP] %s not available: %s\n", variant.c_str(), e.what());
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

      for (size_t m = 0; m < cfg.measure_iters; ++m) {
        tier->Clear();
        for (size_t i = 0; i < keys.size(); ++i) {
          auto t0 = Clock::now();
          tier->Write(keys[i], values[i].data(), values[i].size());
          auto t1 = Clock::now();
          latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
      }

      RecordResult("IO Backend Write", variant, keys.size() * cfg.measure_iters,
                   keys.size() * cfg.measure_iters * cfg.value_size, LatencyElapsed(latencies),
                   latencies, results);
    }

    // Read
    {
      ScopedTempDir tmp(cfg.base_dir + "/io_" + std::string(spec.path_suffix) + "_r");
      std::unique_ptr<SSDTier> tier;
      try {
        tier = std::make_unique<SSDTier>(tmp.path, cfg.ssd_capacity, ucfg);
      } catch (const std::exception& e) {
        std::printf("[SKIP] %s not available: %s\n", variant.c_str(), e.what());
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

      for (size_t m = 0; m < cfg.measure_iters; ++m) {
        for (size_t i = 0; i < keys.size(); ++i) {
          auto t0 = Clock::now();
          tier->ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
          auto t1 = Clock::now();
          latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
      }

      RecordResult("IO Backend Read", variant, keys.size() * cfg.measure_iters,
                   keys.size() * cfg.measure_iters * cfg.value_size, LatencyElapsed(latencies),
                   latencies, results);
    }
  };

  for (const auto& spec : IoBackendSpecs()) {
    run_backend(spec.backend);
  }
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
    UMBPConfig ucfg = MakeBaseSsdConfig(cfg, cfg.ssd_io_backend, mode, cfg.ssd_io_queue_depth);

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

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        tier.Write(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("SSD Write Durability", label, keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size, LatencyElapsed(latencies),
                 latencies, results);
  };

  run_mode(UMBPDurabilityMode::Strict, "Strict");
  run_mode(UMBPDurabilityMode::Relaxed, "Relaxed");
}

// ---------------------------------------------------------------------------
// G. StorageIoDriver microbench
// ---------------------------------------------------------------------------
static void BenchStorageIoDriver(const BenchConfig& cfg,
                                 const std::vector<std::vector<char>>& values,
                                 std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "StorageIoDriver")) return;

  PrintHeader("StorageIoDriver");
  std::string backend_variant = BackendVariantLabel(cfg.ssd_io_backend, cfg.ssd_io_queue_depth);

  std::unique_ptr<StorageIoDriver> driver;
  try {
    driver =
        CreateStorageIoDriver(cfg.ssd_io_backend, static_cast<uint32_t>(cfg.ssd_io_queue_depth));
    if (cfg.ssd_io_backend == UMBPIoBackend::IoUring && !driver->Capabilities().native_async) {
      std::printf("[SKIP] %s not available: native async initialization failed\n",
                  backend_variant.c_str());
      return;
    }
  } catch (const std::exception& e) {
    std::printf("[SKIP] %s not available: %s\n", backend_variant.c_str(), e.what());
    return;
  }

  ScopedTempDir tmp(cfg.base_dir + "/driver_" +
                    std::string(GetIoBackendSpec(cfg.ssd_io_backend).path_suffix));
  ScopedFd rw_fd = OpenBenchFile(tmp.path + "/rw.bin");

  auto prefill_rw_file = [&]() {
    for (size_t i = 0; i < values.size(); ++i) {
      EnsureIoOk(driver->WriteAt(rw_fd.fd, values[i].data(), values[i].size(), i * cfg.value_size),
                 "prefill rw.bin");
    }
    EnsureIoOk(driver->Sync(rw_fd.fd), "sync rw.bin");
  };

  // --- WriteAt ---
  {
    std::vector<double> latencies;
    latencies.reserve(values.size() * cfg.measure_iters);

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (size_t i = 0; i < values.size(); ++i) {
        EnsureIoOk(
            driver->WriteAt(rw_fd.fd, values[i].data(), values[i].size(), i * cfg.value_size),
            "warmup WriteAt");
      }
    }

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < values.size(); ++i) {
        auto t0 = Clock::now();
        EnsureIoOk(
            driver->WriteAt(rw_fd.fd, values[i].data(), values[i].size(), i * cfg.value_size),
            "WriteAt");
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("Driver WriteAt", backend_variant, values.size() * cfg.measure_iters,
                 values.size() * cfg.measure_iters * cfg.value_size, LatencyElapsed(latencies),
                 latencies, results);
  }

  // --- ReadAt ---
  {
    prefill_rw_file();
    std::vector<char> buf(cfg.value_size);
    std::vector<double> latencies;
    latencies.reserve(values.size() * cfg.measure_iters);

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (size_t i = 0; i < values.size(); ++i) {
        EnsureIoOk(driver->ReadAt(rw_fd.fd, buf.data(), buf.size(), i * cfg.value_size),
                   "warmup ReadAt");
      }
    }

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < values.size(); ++i) {
        auto t0 = Clock::now();
        EnsureIoOk(driver->ReadAt(rw_fd.fd, buf.data(), buf.size(), i * cfg.value_size), "ReadAt");
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("Driver ReadAt", backend_variant, values.size() * cfg.measure_iters,
                 values.size() * cfg.measure_iters * cfg.value_size, LatencyElapsed(latencies),
                 latencies, results);
  }

  // --- WriteBatch ---
  {
    ScopedFd batch_fd = OpenBenchFile(tmp.path + "/batch.bin");
    size_t total_batches = (values.size() + cfg.batch_size - 1) / cfg.batch_size;
    std::vector<std::vector<IoWriteOp>> write_batches;
    write_batches.reserve(total_batches);
    for (size_t i = 0; i < values.size(); i += cfg.batch_size) {
      size_t end = std::min(i + cfg.batch_size, values.size());
      std::vector<IoWriteOp> ops;
      ops.reserve(end - i);
      for (size_t j = i; j < end; ++j) {
        ops.push_back({batch_fd.fd, values[j].data(), values[j].size(), j * cfg.value_size});
      }
      write_batches.push_back(std::move(ops));
    }

    std::vector<double> latencies;
    latencies.reserve(total_batches * cfg.measure_iters);

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (const auto& ops : write_batches) {
        EnsureIoOk(driver->WriteBatch(ops), "warmup WriteBatch");
      }
    }

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (const auto& ops : write_batches) {
        auto t0 = Clock::now();
        EnsureIoOk(driver->WriteBatch(ops), "WriteBatch");
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("Driver WriteBatch", backend_variant + "/bs=" + std::to_string(cfg.batch_size),
                 values.size() * cfg.measure_iters,
                 values.size() * cfg.measure_iters * cfg.value_size, LatencyElapsed(latencies),
                 latencies, results);
  }

  // --- Sync ---
  {
    ScopedFd sync_fd = OpenBenchFile(tmp.path + "/sync.bin");
    std::vector<double> latencies;
    latencies.reserve(values.size() * cfg.measure_iters);

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (size_t i = 0; i < values.size(); ++i) {
        EnsureIoOk(driver->WriteAt(sync_fd.fd, values[i].data(), values[i].size(), 0),
                   "warmup Sync write");
        EnsureIoOk(driver->Sync(sync_fd.fd), "warmup Sync");
      }
    }

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < values.size(); ++i) {
        EnsureIoOk(driver->WriteAt(sync_fd.fd, values[i].data(), values[i].size(), 0),
                   "prepare Sync");
        auto t0 = Clock::now();
        EnsureIoOk(driver->Sync(sync_fd.fd), "Sync");
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("Driver Sync", backend_variant, values.size() * cfg.measure_iters, 0,
                 LatencyElapsed(latencies), latencies, results);
  }

  // --- SyncMany ---
  {
    size_t sync_group_size = std::min(cfg.batch_size, values.size());
    std::vector<ScopedFd> sync_fds;
    sync_fds.reserve(sync_group_size);
    for (size_t i = 0; i < sync_group_size; ++i) {
      sync_fds.push_back(OpenBenchFile(tmp.path + "/sync_many_" + std::to_string(i) + ".bin"));
    }

    size_t total_groups = (values.size() + sync_group_size - 1) / sync_group_size;
    std::vector<double> latencies;
    latencies.reserve(total_groups * cfg.measure_iters);

    auto write_group = [&](size_t start_idx) {
      size_t count = std::min(sync_group_size, values.size() - start_idx);
      for (size_t i = 0; i < count; ++i) {
        EnsureIoOk(driver->WriteAt(sync_fds[i].fd, values[start_idx + i].data(),
                                   values[start_idx + i].size(), 0),
                   "prepare SyncMany");
      }
      return count;
    };

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (size_t start = 0; start < values.size(); start += sync_group_size) {
        size_t count = write_group(start);
        std::vector<int> fds;
        fds.reserve(count);
        for (size_t i = 0; i < count; ++i) fds.push_back(sync_fds[i].fd);
        EnsureIoOk(driver->SyncMany(fds), "warmup SyncMany");
      }
    }

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t start = 0; start < values.size(); start += sync_group_size) {
        size_t count = write_group(start);
        std::vector<int> fds;
        fds.reserve(count);
        for (size_t i = 0; i < count; ++i) fds.push_back(sync_fds[i].fd);
        auto t0 = Clock::now();
        EnsureIoOk(driver->SyncMany(fds), "SyncMany");
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("Driver SyncMany", backend_variant + "/fds=" + std::to_string(sync_group_size),
                 values.size() * cfg.measure_iters, 0, LatencyElapsed(latencies), latencies,
                 results);
  }
}

// ---------------------------------------------------------------------------
// H. Concurrent Scaling (UMBPClient Put + Get)
// ---------------------------------------------------------------------------
static void BenchConcurrent(const BenchConfig& cfg, const std::vector<std::string>& keys,
                            const std::vector<std::vector<char>>& values,
                            std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "Concurrent")) return;

  PrintHeader("Concurrent Scaling");

  for (int nthreads : cfg.thread_counts) {
    ScopedTempDir tmp(cfg.base_dir + "/concurrent_" + std::to_string(nthreads));

    UMBPConfig ucfg = MakeStandaloneClientConfig(cfg, tmp.path + "/ssd", cfg.dram_capacity);

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

      // Merge latencies
      std::vector<double> all_lat;
      for (auto& tl : thread_latencies) {
        all_lat.insert(all_lat.end(), tl.begin(), tl.end());
      }

      RecordResult(
          "Concurrent Put", variant,
          keys_per_thread * static_cast<size_t>(nthreads) * cfg.measure_iters,
          keys_per_thread * static_cast<size_t>(nthreads) * cfg.measure_iters * cfg.value_size,
          LatencyElapsed(all_lat), all_lat, results);
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

      std::vector<double> all_lat;
      for (auto& tl : thread_latencies) {
        all_lat.insert(all_lat.end(), tl.begin(), tl.end());
      }

      RecordResult(
          "Concurrent Get", variant,
          keys_per_thread * static_cast<size_t>(nthreads) * cfg.measure_iters,
          keys_per_thread * static_cast<size_t>(nthreads) * cfg.measure_iters * cfg.value_size,
          LatencyElapsed(all_lat), all_lat, results);
    }
  }
}

// ---------------------------------------------------------------------------
// I. Leader Mode: sync vs async copy
// ---------------------------------------------------------------------------
static void BenchLeaderMode(const BenchConfig& cfg, const std::vector<std::string>& keys,
                            const std::vector<std::vector<char>>& values,
                            std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "Leader")) return;

  PrintHeader("Leader Mode (sync vs async copy)");

  auto run_mode = [&](bool async_copy, const std::string& label) {
    ScopedTempDir tmp(cfg.base_dir + "/leader_" + label);

    UMBPConfig ucfg = MakeLeaderClientConfig(cfg, tmp.path + "/ssd", cfg.dram_capacity, async_copy);

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

    RecordResult("Leader Put", label, keys.size() * cfg.measure_iters,
                 keys.size() * cfg.measure_iters * cfg.value_size, LatencyElapsed(latencies),
                 latencies, results);
  };

  run_mode(false, "sync copy");
  run_mode(true, "async copy");
}

// ---------------------------------------------------------------------------
// J. Capacity Pressure
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

  UMBPConfig ucfg = MakeStandaloneClientConfig(cfg, tmp.path + "/ssd", pressure_dram);

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

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      client.Clear();
      for (size_t i = 0; i < half; ++i) {
        auto t0 = Clock::now();
        client.Put(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }

    RecordResult("Capacity Put", "no pressure", half * cfg.measure_iters,
                 half * cfg.measure_iters * cfg.value_size, LatencyElapsed(latencies), latencies,
                 results);
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

    RecordResult("Capacity Put", "under pressure", (keys.size() - half) * cfg.measure_iters,
                 (keys.size() - half) * cfg.measure_iters * cfg.value_size,
                 LatencyElapsed(latencies), latencies, results);
  }
}

// ---------------------------------------------------------------------------
// K. E2E UMBPClient Benchmark (sglang connector simulation)
// ---------------------------------------------------------------------------
static void BenchE2E(const BenchConfig& cfg, const E2EConfig& e2e,
                     std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "E2E")) return;

  size_t value_size = e2e.ValueSizePerKey();
  size_t keys_per_page = e2e.KeysPerPage();
  std::string mode_str = (e2e.mode == E2EModelMode::MLA) ? "MLA" : "MHA";
  std::string variant_label = mode_str + "/" + std::to_string(e2e.batch_pages) + "pg/" +
                              std::to_string(value_size / 1024) + "KB";

  PrintHeader("E2E UMBPClient (" + variant_label + ")");
  std::printf("  mode          = %s\n", mode_str.c_str());
  std::printf("  num_layers    = %zu\n", e2e.num_layers);
  if (e2e.mode == E2EModelMode::MLA) {
    std::printf("  kv_cache_dim  = %zu (lora=%zu + rope=%zu)\n", e2e.KvCacheDim(), e2e.kv_lora_rank,
                e2e.qk_rope_head_dim);
  } else {
    std::printf("  num_kv_heads  = %zu, head_dim = %zu\n", e2e.num_kv_heads, e2e.head_dim);
  }
  std::printf("  kv_cache_dtype= %s (%zu bytes)\n", e2e.kv_cache_dtype.c_str(), e2e.DtypeSize());
  std::printf("  value_size    = %zu bytes (%zu KB)\n", value_size, value_size / 1024);
  std::printf("  keys_per_page = %zu\n", keys_per_page);
  std::printf("  num_pages     = %zu\n", e2e.num_pages);
  std::printf("  batch_pages   = %zu\n", e2e.batch_pages);
  std::printf("  dedup_ratio   = %.0f%%\n", e2e.dedup_ratio * 100);

  E2EKeyGenerator keygen{e2e.mode, 0, 0};
  E2EHostBuffer host_buf(e2e.num_pages, value_size, keys_per_page);

  // DRAM sized to hold all data with headroom.
  size_t total_data = e2e.num_pages * keys_per_page * value_size;

  // E2E DRAM-only config: reuses cfg's SSD base settings (io_backend, durability, etc.)
  // but disables SSD. If an SSD scenario is needed, MakeStandaloneClientConfig enables it.
  auto MakeDramOnlyConfig = [&]() -> UMBPConfig {
    UMBPConfig ucfg = MakeBaseSsdConfig(cfg);  // inherit SSD params for consistency
    ucfg.dram.capacity_bytes = total_data * 2;
    ucfg.ssd.enabled = false;
    ucfg.role = UMBPRole::Standalone;
    ucfg.copy_pipeline.async_enabled = false;
    ucfg.eviction.auto_promote_on_read = false;
    return ucfg;
  };

  size_t batches_per_iter = (e2e.num_pages + e2e.batch_pages - 1) / e2e.batch_pages;

  // Auto-scale iters to ensure enough latency samples for reliable percentiles.
  // Target: >=100 batch samples for p95, >=1000 for p99.
  // User --iters is treated as a minimum; we scale up if batches_per_iter is small.
  constexpr size_t kMinSamplesForP95 = 100;
  size_t e2e_iters = cfg.measure_iters;
  if (batches_per_iter > 0 && batches_per_iter * e2e_iters < kMinSamplesForP95) {
    e2e_iters = (kMinSamplesForP95 + batches_per_iter - 1) / batches_per_iter;
  }
  size_t total_samples = batches_per_iter * e2e_iters;
  std::printf("  e2e_iters     = %zu (auto-scaled from %zu; %zu batches/iter, %zu total samples)\n",
              e2e_iters, cfg.measure_iters, batches_per_iter, total_samples);

  // Helper: fill all pages into a client (untimed).
  auto FillAll = [&](UMBPClient& client) {
    for (size_t b = 0; b < e2e.num_pages; b += e2e.batch_pages) {
      size_t count = std::min(e2e.batch_pages, e2e.num_pages - b);
      auto keys = keygen.KeysForPages(b, count);
      std::vector<uintptr_t> ptrs;
      std::vector<size_t> sizes;
      host_buf.GetBatchMeta(b, count, ptrs, sizes);
      auto depths = GenerateDepths(e2e, b, count);
      client.BatchPutFromPtrWithDepth(keys, ptrs, sizes, depths);
    }
  };

  // Helper: fill pages [0, num_pages) via BatchPut, collecting per-batch latencies.
  auto FillAllTimed = [&](UMBPClient& client, size_t num_pages, std::vector<double>& latencies) {
    for (size_t b = 0; b < num_pages; b += e2e.batch_pages) {
      size_t count = std::min(e2e.batch_pages, num_pages - b);
      auto keys = keygen.KeysForPages(b, count);
      std::vector<uintptr_t> ptrs;
      std::vector<size_t> sizes;
      host_buf.GetBatchMeta(b, count, ptrs, sizes);
      auto depths = GenerateDepths(e2e, b, count);
      auto t0 = Clock::now();
      client.BatchPutFromPtrWithDepth(keys, ptrs, sizes, depths);
      auto t1 = Clock::now();
      latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
  };

  // Helper: read all pages [0, num_pages) via BatchGet, collecting per-batch latencies.
  std::vector<char> read_buf;
  std::vector<uintptr_t> read_ptrs;
  std::vector<size_t> read_sizes;

  auto ReadAllTimed = [&](UMBPClient& client, size_t num_pages, std::vector<double>& latencies) {
    for (size_t b = 0; b < num_pages; b += e2e.batch_pages) {
      size_t count = std::min(e2e.batch_pages, num_pages - b);
      auto keys = keygen.KeysForPages(b, count);
      E2EHostBuffer::MakeReadMeta(count, keys_per_page, value_size, read_buf, read_ptrs,
                                  read_sizes);
      auto t0 = Clock::now();
      client.BatchGetIntoPtr(keys, read_ptrs, read_sizes);
      auto t1 = Clock::now();
      latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
  };

  // Helper: read all pages [0, num_pages) without timing (for warmup).
  auto ReadAll = [&](UMBPClient& client, size_t num_pages) {
    for (size_t b = 0; b < num_pages; b += e2e.batch_pages) {
      size_t count = std::min(e2e.batch_pages, num_pages - b);
      auto keys = keygen.KeysForPages(b, count);
      E2EHostBuffer::MakeReadMeta(count, keys_per_page, value_size, read_buf, read_ptrs,
                                  read_sizes);
      client.BatchGetIntoPtr(keys, read_ptrs, read_sizes);
    }
  };

  // LatencyElapsed() is now a file-scope helper (above RecordResult).

  // ---------------------------------------------------------------
  // (a) E2E BatchSet — fresh writes via BatchPutFromPtrWithDepth
  // ---------------------------------------------------------------
  {
    ScopedTempDir tmp(cfg.base_dir + "/e2e_batchset");
    UMBPConfig ucfg = MakeDramOnlyConfig();
    UMBPClient client(ucfg);

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      client.Clear();
      FillAll(client);
    }
    client.Clear();

    std::vector<double> latencies;
    latencies.reserve(batches_per_iter * e2e_iters);

    for (size_t m = 0; m < e2e_iters; ++m) {
      client.Clear();
      FillAllTimed(client, e2e.num_pages, latencies);
    }
    RecordResult("E2E BatchSet", variant_label, e2e.num_pages * e2e_iters,
                 e2e.num_pages * e2e_iters * keys_per_page * value_size, LatencyElapsed(latencies),
                 latencies, results);
  }

  // ---------------------------------------------------------------
  // (b) E2E BatchGet — read-back via BatchGetIntoPtr
  // ---------------------------------------------------------------
  {
    ScopedTempDir tmp(cfg.base_dir + "/e2e_batchget");
    UMBPConfig ucfg = MakeDramOnlyConfig();
    UMBPClient client(ucfg);
    FillAll(client);

    for (size_t w = 0; w < cfg.warmup_iters; ++w) ReadAll(client, e2e.num_pages);

    std::vector<double> latencies;
    latencies.reserve(batches_per_iter * e2e_iters);
    for (size_t m = 0; m < e2e_iters; ++m) ReadAllTimed(client, e2e.num_pages, latencies);
    RecordResult("E2E BatchGet", variant_label, e2e.num_pages * e2e_iters,
                 e2e.num_pages * e2e_iters * keys_per_page * value_size, LatencyElapsed(latencies),
                 latencies, results);
  }

  // ---------------------------------------------------------------
  // (c) E2E BatchExists — BatchExistsConsecutive with partial fill
  // ---------------------------------------------------------------
  {
    ScopedTempDir tmp(cfg.base_dir + "/e2e_exists");
    UMBPConfig ucfg = MakeDramOnlyConfig();
    UMBPClient client(ucfg);

    // Fill first half of pages to test early-stop behavior.
    size_t fill_pages = e2e.num_pages / 2;
    for (size_t b = 0; b < fill_pages; b += e2e.batch_pages) {
      size_t count = std::min(e2e.batch_pages, fill_pages - b);
      auto keys = keygen.KeysForPages(b, count);
      std::vector<uintptr_t> ptrs;
      std::vector<size_t> sizes;
      host_buf.GetBatchMeta(b, count, ptrs, sizes);
      auto depths = GenerateDepths(e2e, b, count);
      client.BatchPutFromPtrWithDepth(keys, ptrs, sizes, depths);
    }

    // Query all pages — consecutive hits stop at fill_pages boundary.
    auto all_keys = keygen.KeysForPages(0, e2e.num_pages);

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      client.BatchExistsConsecutive(all_keys);
    }

    std::vector<double> latencies;
    latencies.reserve(e2e_iters);

    for (size_t m = 0; m < e2e_iters; ++m) {
      auto t0 = Clock::now();
      size_t hit = client.BatchExistsConsecutive(all_keys);
      auto t1 = Clock::now();
      latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      (void)hit;
    }
    RecordResult("E2E BatchExists", variant_label, e2e.num_pages * e2e_iters, 0,
                 LatencyElapsed(latencies), latencies, results);
  }

  // ---------------------------------------------------------------
  // (d) E2E Dedup — batch_set with prefix reuse (dedup_ratio pre-filled)
  //
  // Each iteration: clear → re-fill only prefix pages → batch_set all pages.
  // Pages [0, prefill) hit MayExist dedup; pages [prefill, num_pages) are fresh writes.
  // ---------------------------------------------------------------
  {
    ScopedTempDir tmp(cfg.base_dir + "/e2e_dedup");
    UMBPConfig ucfg = MakeDramOnlyConfig();

    size_t prefill_pages = static_cast<size_t>(e2e.num_pages * e2e.dedup_ratio);

    // Helper: clear and re-seed prefix pages in a client.
    auto SeedPrefix = [&](UMBPClient& c) {
      c.Clear();
      // FillAll variant with partial page count (untimed).
      for (size_t b = 0; b < prefill_pages; b += e2e.batch_pages) {
        size_t count = std::min(e2e.batch_pages, prefill_pages - b);
        auto keys = keygen.KeysForPages(b, count);
        std::vector<uintptr_t> ptrs;
        std::vector<size_t> sizes;
        host_buf.GetBatchMeta(b, count, ptrs, sizes);
        auto depths = GenerateDepths(e2e, b, count);
        c.BatchPutFromPtrWithDepth(keys, ptrs, sizes, depths);
      }
    };

    // Warmup
    {
      UMBPClient client(ucfg);
      for (size_t w = 0; w < cfg.warmup_iters; ++w) {
        SeedPrefix(client);
        FillAll(client);
      }
    }

    std::vector<double> latencies;
    latencies.reserve(batches_per_iter * e2e_iters);

    UMBPClient client(ucfg);
    for (size_t m = 0; m < e2e_iters; ++m) {
      SeedPrefix(client);
      FillAllTimed(client, e2e.num_pages, latencies);
    }
    RecordResult("E2E Dedup", std::to_string(static_cast<int>(e2e.dedup_ratio * 100)) + "% dedup",
                 e2e.num_pages * e2e_iters, e2e.num_pages * e2e_iters * keys_per_page * value_size,
                 LatencyElapsed(latencies), latencies, results);
  }

  // ---------------------------------------------------------------
  // (e) E2E Prefetch — exists → get pipeline (sglang prefetch flow)
  // ---------------------------------------------------------------
  {
    ScopedTempDir tmp(cfg.base_dir + "/e2e_prefetch");
    UMBPConfig ucfg = MakeDramOnlyConfig();
    UMBPClient client(ucfg);
    FillAll(client);

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      auto all_keys = keygen.KeysForPages(0, e2e.num_pages);
      client.BatchExistsConsecutive(all_keys);
      ReadAll(client, e2e.num_pages);
    }

    std::vector<double> latencies;
    latencies.reserve(e2e_iters);

    for (size_t m = 0; m < e2e_iters; ++m) {
      auto t0 = Clock::now();

      // Step 1: exists check (determines how many pages to fetch).
      auto all_keys = keygen.KeysForPages(0, e2e.num_pages);
      size_t hit = client.BatchExistsConsecutive(all_keys);
      size_t hit_pages = hit / keys_per_page;

      // Step 2: batch_get the hit pages in chunks (matching sglang flow).
      // Note: untimed inner loop — entire prefetch cycle is timed as one op.
      for (size_t b = 0; b < hit_pages; b += e2e.batch_pages) {
        size_t count = std::min(e2e.batch_pages, hit_pages - b);
        auto keys = keygen.KeysForPages(b, count);
        E2EHostBuffer::MakeReadMeta(count, keys_per_page, value_size, read_buf, read_ptrs,
                                    read_sizes);
        client.BatchGetIntoPtr(keys, read_ptrs, read_sizes);
      }

      auto t1 = Clock::now();
      latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    RecordResult("E2E Prefetch", variant_label, e2e.num_pages * e2e_iters,
                 e2e.num_pages * e2e_iters * keys_per_page * value_size, LatencyElapsed(latencies),
                 latencies, results);
  }

  // ---------------------------------------------------------------
  // (f) E2E Capacity Pressure — DRAM too small, eviction spills to SSD,
  //     then BatchGet reads back from SSD.
  //
  //     DRAM = 50% of total data → second half triggers eviction.
  //     Measures: write under pressure (set), then read-back (get from SSD).
  // ---------------------------------------------------------------
  // SSD scenarios — always run when BenchE2E is entered (gated by "E2E" filter above).
  {
    ScopedTempDir tmp(cfg.base_dir + "/e2e_capacity");

    UMBPConfig ucfg =
        MakeStandaloneClientConfig(cfg, tmp.path + "/ssd", total_data / 2, total_data * 2);
    fs::create_directories(ucfg.ssd.storage_dir);

    // --- Write under capacity pressure ---
    {
      UMBPClient client(ucfg);

      // Warmup
      for (size_t w = 0; w < cfg.warmup_iters; ++w) {
        client.Clear();
        FillAll(client);
      }
      client.Clear();

      std::vector<double> latencies;
      latencies.reserve(batches_per_iter * e2e_iters);

      for (size_t m = 0; m < e2e_iters; ++m) {
        client.Clear();
        FillAllTimed(client, e2e.num_pages, latencies);
      }
      RecordResult("E2E Capacity Set", "DRAM 50%", e2e.num_pages * e2e_iters,
                   e2e.num_pages * e2e_iters * keys_per_page * value_size,
                   LatencyElapsed(latencies), latencies, results);
    }

    // --- Read-back (early pages evicted to SSD) ---
    {
      UMBPClient client(ucfg);
      FillAll(client);  // first half evicted to SSD, second half in DRAM

      for (size_t w = 0; w < cfg.warmup_iters; ++w) ReadAll(client, e2e.num_pages);

      std::vector<double> latencies;
      latencies.reserve(batches_per_iter * e2e_iters);

      for (size_t m = 0; m < e2e_iters; ++m) ReadAllTimed(client, e2e.num_pages, latencies);
      RecordResult("E2E Capacity Get", "DRAM+SSD mixed", e2e.num_pages * e2e_iters,
                   e2e.num_pages * e2e_iters * keys_per_page * value_size,
                   LatencyElapsed(latencies), latencies, results);
    }
  }

  // ---------------------------------------------------------------
  // (g) E2E Leader Mode — SharedSSDLeader with async copy pipeline.
  //
  //     Mirrors sglang MLA + TP>1 deployment:
  //     BatchPutFromPtrWithDepth writes to DRAM, CopyPipeline async-copies to SSD.
  //     Compares sync vs async copy throughput.
  // ---------------------------------------------------------------
  {
    auto run_leader = [&](bool async_copy, const std::string& label) {
      ScopedTempDir tmp(cfg.base_dir + "/e2e_leader_" + label);

      UMBPConfig ucfg = MakeLeaderClientConfig(cfg, tmp.path + "/ssd", total_data * 2, async_copy,
                                               total_data * 2);
      fs::create_directories(ucfg.ssd.storage_dir);

      // Warmup — construct+destroy client so destructor drains async queue.
      for (size_t w = 0; w < cfg.warmup_iters; ++w) {
        UMBPClient client(ucfg);
        FillAll(client);
      }

      std::vector<double> latencies;
      latencies.reserve(batches_per_iter * e2e_iters);

      for (size_t m = 0; m < e2e_iters; ++m) {
        UMBPClient client(ucfg);  // fresh client per iter (destructor drains async)
        FillAllTimed(client, e2e.num_pages, latencies);
      }
      RecordResult("E2E Leader Set", label, e2e.num_pages * e2e_iters,
                   e2e.num_pages * e2e_iters * keys_per_page * value_size,
                   LatencyElapsed(latencies), latencies, results);
    };

    run_leader(false, "sync copy");
    run_leader(true, "async copy");
  }

  // ---------------------------------------------------------------
  // (h) E2E Follower Get — pure SSD read via SharedSSDFollower.
  //
  //     Mirrors sglang MLA + TP>1, rank>0:
  //     Leader writes all pages to shared SSD, Follower reads from SSD only.
  //     Measures pure SSD read throughput (no DRAM hits).
  // ---------------------------------------------------------------
  {
    ScopedTempDir tmp(cfg.base_dir + "/e2e_follower");
    std::string ssd_dir = tmp.path + "/ssd";

    // Step 1: Leader writes all pages to SSD.
    {
      UMBPConfig leader_cfg =
          MakeLeaderClientConfig(cfg, ssd_dir, total_data * 2, false, total_data * 2);
      fs::create_directories(leader_cfg.ssd.storage_dir);
      UMBPClient leader(leader_cfg);
      FillAll(leader);
      // Leader destructor drains copy pipeline, ensuring all data is on SSD.
    }

    // Step 2: Follower reads from SSD.
    UMBPConfig follower_cfg =
        MakeFollowerClientConfig(cfg, ssd_dir, total_data * 2, total_data * 2);
    UMBPClient follower(follower_cfg);

    for (size_t w = 0; w < cfg.warmup_iters; ++w) ReadAll(follower, e2e.num_pages);

    std::vector<double> latencies;
    latencies.reserve(batches_per_iter * e2e_iters);
    for (size_t m = 0; m < e2e_iters; ++m) ReadAllTimed(follower, e2e.num_pages, latencies);
    RecordResult("E2E Follower Get", "SSD read", e2e.num_pages * e2e_iters,
                 e2e.num_pages * e2e_iters * keys_per_page * value_size, LatencyElapsed(latencies),
                 latencies, results);
  }
}

// ---------------------------------------------------------------------------
// CLI parsing & profiles
// ---------------------------------------------------------------------------
static void ApplyProfile(BenchConfig& cfg, E2EConfig& e2e, const std::string& profile) {
  if (profile == "small") {
    cfg.num_keys = 200;
    cfg.value_size = 1024;
    cfg.batch_size = 16;
    cfg.dram_capacity = 4ULL * 1024 * 1024;
    cfg.ssd_capacity = 16ULL * 1024 * 1024;
    cfg.segment_size = 4ULL * 1024 * 1024;
    cfg.thread_counts = {1, 2};
    // E2E: DeepSeek-V3, small scale
    ApplyModelPreset(e2e, "deepseek-v3");
    e2e.num_pages = 64;
    e2e.batch_pages = 16;
  } else if (profile == "medium") {
    cfg.num_keys = 1000;
    cfg.value_size = 4096;
    cfg.batch_size = 64;
    cfg.dram_capacity = 64ULL * 1024 * 1024;
    cfg.ssd_capacity = 256ULL * 1024 * 1024;
    cfg.segment_size = 64ULL * 1024 * 1024;
    cfg.thread_counts = {1, 2, 4, 8};
    // E2E: DeepSeek-V3, default scale
    ApplyModelPreset(e2e, "deepseek-v3");
    e2e.num_pages = 512;
    e2e.batch_pages = 128;
  } else if (profile == "large") {
    cfg.num_keys = 10000;
    cfg.value_size = 64 * 1024;
    cfg.batch_size = 128;
    cfg.dram_capacity = 512ULL * 1024 * 1024;
    cfg.ssd_capacity = 2ULL * 1024 * 1024 * 1024;
    cfg.segment_size = 256ULL * 1024 * 1024;
    cfg.thread_counts = {1, 2, 4, 8};
    // E2E: DeepSeek-V3, large scale
    ApplyModelPreset(e2e, "deepseek-v3");
    e2e.num_pages = 2048;
    e2e.batch_pages = 128;
  } else {
    std::cerr << "Unknown profile: " << profile << std::endl;
    std::exit(1);
  }
}

static void PrintUsage(const char* argv0) {
  std::printf(
      "Usage: %s [OPTIONS]\n"
      "\n"
      "General:\n"
      "  --profile <small|medium|large>   Preset config (default: medium)\n"
      "  --num-keys N                     Keys per scenario\n"
      "  --value-size N                   Value size in bytes\n"
      "  --batch-size N                   Batch size\n"
      "  --warmup-iters N                 Warmup iterations\n"
      "  --iters N                        Measurement iterations\n"
      "  --filter SUBSTRING               Run only matching scenarios\n"
      "  --dir PATH                       Temp directory path\n"
      "  -h, --help                       Help\n"
      "\n"
      "SSD / driver:\n"
      "  --dram-capacity N                DRAM capacity in bytes\n"
      "  --ssd-capacity N                 SSD capacity in bytes\n"
      "  --segment-size N                 SSD segment size in bytes\n"
      "  --ssd-io-backend <pthread|posix|io_uring>\n"
      "  --ssd-queue-depth N              Storage I/O queue depth\n"
      "  --ssd-durability <strict|relaxed>\n"
      "\n"
      "E2E (sglang connector simulation):\n"
      "  --model <deepseek-v3|deepseek-v2|llama-70b|llama-8b>\n"
      "  --mode <mha|mla>                 Override model attention mode\n"
      "  --num-layers N                   Transformer layers\n"
      "  --num-kv-heads N                 KV heads (MHA only)\n"
      "  --head-dim N                     Head dimension (MHA only)\n"
      "  --kv-lora-rank N                 LoRA rank (MLA only)\n"
      "  --qk-rope-head-dim N             RoPE head dim (MLA only)\n"
      "  --kv-cache-dtype <bf16|fp16|fp8_e4m3|fp8_e5m2>\n"
      "  --page-size N                    Tokens per page\n"
      "  --num-pages N                    Total pages for E2E test\n"
      "  --batch-pages N                  Pages per batch call\n"
      "  --dedup-ratio F                  Prefix reuse ratio (0.0-1.0)\n",
      argv0);
}

struct ParsedArgs {
  BenchConfig cfg;
  E2EConfig e2e;
};

static ParsedArgs ParseArgs(int argc, char* argv[]) {
  BenchConfig cfg;
  E2EConfig e2e;
  std::string profile = "medium";
  std::string model_preset;

  // Track which fields the user explicitly overrides
  bool override_num_keys = false;
  bool override_value_size = false;
  bool override_batch_size = false;
  bool override_warmup_iters = false;
  bool override_dram_capacity = false;
  bool override_ssd_capacity = false;
  bool override_segment_size = false;
  bool override_ssd_io_backend = false;
  bool override_ssd_io_queue_depth = false;
  bool override_ssd_durability = false;

  size_t user_num_keys = 0;
  size_t user_value_size = 0;
  size_t user_batch_size = 0;
  size_t user_warmup_iters = 0;
  size_t user_dram_capacity = 0;
  size_t user_ssd_capacity = 0;
  size_t user_segment_size = 0;
  UMBPIoBackend user_ssd_io_backend = UMBPIoBackend::PThread;
  size_t user_ssd_io_queue_depth = 0;
  UMBPDurabilityMode user_ssd_durability = UMBPDurabilityMode::Relaxed;

  // E2E overrides
  bool override_mode = false;
  bool override_num_layers = false;
  bool override_num_kv_heads = false;
  bool override_head_dim = false;
  bool override_kv_lora_rank = false;
  bool override_qk_rope = false;
  bool override_kv_cache_dtype = false;
  bool override_page_size = false;
  bool override_num_pages = false;
  bool override_batch_pages = false;
  bool override_dedup_ratio = false;

  E2EModelMode user_mode{};
  size_t user_num_layers = 0, user_num_kv_heads = 0, user_head_dim = 0;
  size_t user_kv_lora_rank = 0, user_qk_rope = 0;
  std::string user_kv_cache_dtype;
  size_t user_page_size = 0, user_num_pages = 0, user_batch_pages = 0;
  double user_dedup_ratio = 0.0;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else if (arg == "--profile" && i + 1 < argc) {
      profile = argv[++i];
    } else if (arg == "--num-keys" && i + 1 < argc) {
      user_num_keys = std::stoull(argv[++i]);
      override_num_keys = true;
    } else if (arg == "--value-size" && i + 1 < argc) {
      user_value_size = std::stoull(argv[++i]);
      override_value_size = true;
    } else if (arg == "--batch-size" && i + 1 < argc) {
      user_batch_size = std::stoull(argv[++i]);
      override_batch_size = true;
    } else if (arg == "--warmup-iters" && i + 1 < argc) {
      user_warmup_iters = std::stoull(argv[++i]);
      override_warmup_iters = true;
    } else if (arg == "--iters" && i + 1 < argc) {
      cfg.measure_iters = std::stoull(argv[++i]);
    } else if (arg == "--filter" && i + 1 < argc) {
      cfg.filter = argv[++i];
    } else if (arg == "--dir" && i + 1 < argc) {
      cfg.base_dir = argv[++i];
    } else if (arg == "--dram-capacity" && i + 1 < argc) {
      user_dram_capacity = std::stoull(argv[++i]);
      override_dram_capacity = true;
    } else if (arg == "--ssd-capacity" && i + 1 < argc) {
      user_ssd_capacity = std::stoull(argv[++i]);
      override_ssd_capacity = true;
    } else if (arg == "--segment-size" && i + 1 < argc) {
      user_segment_size = std::stoull(argv[++i]);
      override_segment_size = true;
    } else if (arg == "--ssd-io-backend" && i + 1 < argc) {
      std::string backend = argv[++i];
      if (!ParseIoBackend(backend, user_ssd_io_backend)) {
        std::cerr << "Error: --ssd-io-backend must be one of: pthread, posix, io_uring"
                  << " (got '" << backend << "')\n";
        std::exit(1);
      }
      override_ssd_io_backend = true;
    } else if (arg == "--ssd-queue-depth" && i + 1 < argc) {
      user_ssd_io_queue_depth = std::stoull(argv[++i]);
      override_ssd_io_queue_depth = true;
    } else if (arg == "--ssd-durability" && i + 1 < argc) {
      std::string durability = argv[++i];
      if (!ParseDurabilityMode(durability, user_ssd_durability)) {
        std::cerr << "Error: --ssd-durability must be 'strict' or 'relaxed', got '" << durability
                  << "'\n";
        std::exit(1);
      }
      override_ssd_durability = true;
      // E2E flags
    } else if (arg == "--model" && i + 1 < argc) {
      model_preset = argv[++i];
    } else if (arg == "--mode" && i + 1 < argc) {
      std::string m = argv[++i];
      if (m == "mla") {
        user_mode = E2EModelMode::MLA;
      } else if (m == "mha") {
        user_mode = E2EModelMode::MHA;
      } else {
        std::cerr << "Error: --mode must be 'mha' or 'mla', got '" << m << "'\n";
        std::exit(1);
      }
      override_mode = true;
    } else if (arg == "--num-layers" && i + 1 < argc) {
      user_num_layers = std::stoull(argv[++i]);
      override_num_layers = true;
    } else if (arg == "--num-kv-heads" && i + 1 < argc) {
      user_num_kv_heads = std::stoull(argv[++i]);
      override_num_kv_heads = true;
    } else if (arg == "--head-dim" && i + 1 < argc) {
      user_head_dim = std::stoull(argv[++i]);
      override_head_dim = true;
    } else if (arg == "--kv-lora-rank" && i + 1 < argc) {
      user_kv_lora_rank = std::stoull(argv[++i]);
      override_kv_lora_rank = true;
    } else if (arg == "--qk-rope-head-dim" && i + 1 < argc) {
      user_qk_rope = std::stoull(argv[++i]);
      override_qk_rope = true;
    } else if (arg == "--kv-cache-dtype" && i + 1 < argc) {
      user_kv_cache_dtype = argv[++i];
      override_kv_cache_dtype = true;
    } else if (arg == "--page-size" && i + 1 < argc) {
      user_page_size = std::stoull(argv[++i]);
      override_page_size = true;
    } else if (arg == "--num-pages" && i + 1 < argc) {
      user_num_pages = std::stoull(argv[++i]);
      override_num_pages = true;
    } else if (arg == "--batch-pages" && i + 1 < argc) {
      user_batch_pages = std::stoull(argv[++i]);
      override_batch_pages = true;
    } else if (arg == "--dedup-ratio" && i + 1 < argc) {
      user_dedup_ratio = std::stod(argv[++i]);
      override_dedup_ratio = true;
    } else {
      std::cerr << "Unknown option: " << arg << std::endl;
      PrintUsage(argv[0]);
      std::exit(1);
    }
  }

  // Apply profile first (sets both BenchConfig and E2EConfig defaults).
  ApplyProfile(cfg, e2e, profile);

  // Override BenchConfig fields.
  if (override_num_keys) cfg.num_keys = user_num_keys;
  if (override_value_size) cfg.value_size = user_value_size;
  if (override_batch_size) cfg.batch_size = user_batch_size;
  if (override_warmup_iters) cfg.warmup_iters = user_warmup_iters;
  if (override_dram_capacity) cfg.dram_capacity = user_dram_capacity;
  if (override_ssd_capacity) cfg.ssd_capacity = user_ssd_capacity;
  if (override_segment_size) cfg.segment_size = user_segment_size;
  if (override_ssd_io_backend) cfg.ssd_io_backend = user_ssd_io_backend;
  if (override_ssd_io_queue_depth) cfg.ssd_io_queue_depth = user_ssd_io_queue_depth;
  if (override_ssd_durability) cfg.ssd_durability_mode = user_ssd_durability;

  // Apply model preset (overrides E2E model params from profile).
  if (!model_preset.empty()) ApplyModelPreset(e2e, model_preset);

  // Override individual E2E fields.
  if (override_mode) e2e.mode = user_mode;
  if (override_num_layers) e2e.num_layers = user_num_layers;
  if (override_num_kv_heads) e2e.num_kv_heads = user_num_kv_heads;
  if (override_head_dim) e2e.head_dim = user_head_dim;
  if (override_kv_lora_rank) e2e.kv_lora_rank = user_kv_lora_rank;
  if (override_qk_rope) e2e.qk_rope_head_dim = user_qk_rope;
  if (override_kv_cache_dtype) e2e.kv_cache_dtype = user_kv_cache_dtype;
  if (override_page_size) e2e.page_size = user_page_size;
  if (override_num_pages) e2e.num_pages = user_num_pages;
  if (override_batch_pages) e2e.batch_pages = user_batch_pages;
  if (override_dedup_ratio) e2e.dedup_ratio = user_dedup_ratio;

  // --- Input validation ---
  if (cfg.num_keys == 0) {
    std::cerr << "Error: --num-keys must be > 0\n";
    std::exit(1);
  }
  if (cfg.value_size == 0) {
    std::cerr << "Error: --value-size must be > 0\n";
    std::exit(1);
  }
  if (cfg.batch_size == 0) {
    std::cerr << "Error: --batch-size must be > 0\n";
    std::exit(1);
  }
  if (cfg.dram_capacity == 0) {
    std::cerr << "Error: --dram-capacity must be > 0\n";
    std::exit(1);
  }
  if (cfg.ssd_capacity == 0) {
    std::cerr << "Error: --ssd-capacity must be > 0\n";
    std::exit(1);
  }
  if (cfg.segment_size == 0) {
    std::cerr << "Error: --segment-size must be > 0\n";
    std::exit(1);
  }
  if (cfg.ssd_io_queue_depth == 0) {
    std::cerr << "Error: --ssd-queue-depth must be > 0\n";
    std::exit(1);
  }
  if (cfg.ssd_io_queue_depth > std::numeric_limits<uint32_t>::max()) {
    std::cerr << "Error: --ssd-queue-depth must fit in uint32_t\n";
    std::exit(1);
  }
  if (e2e.batch_pages == 0) {
    std::cerr << "Error: --batch-pages must be > 0\n";
    std::exit(1);
  }
  if (e2e.num_pages == 0) {
    std::cerr << "Error: --num-pages must be > 0\n";
    std::exit(1);
  }
  if (e2e.dedup_ratio < 0.0 || e2e.dedup_ratio > 1.0) {
    std::cerr << "Error: --dedup-ratio must be in [0.0, 1.0]\n";
    std::exit(1);
  }
  if (e2e.page_size == 0) {
    std::cerr << "Error: --page-size must be > 0\n";
    std::exit(1);
  }
  {
    const auto& d = e2e.kv_cache_dtype;
    if (d != "bf16" && d != "fp16" && d != "fp8" && d != "fp8_e4m3" && d != "fp8_e5m2") {
      std::cerr << "Error: --kv-cache-dtype must be one of: bf16, fp16, fp8, fp8_e4m3, fp8_e5m2"
                << " (got '" << d << "')\n";
      std::exit(1);
    }
  }

  return {cfg, e2e};
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  auto [cfg, e2e] = ParseArgs(argc, argv);

  std::printf("UMBP Benchmark\n");
  std::printf("  num_keys     = %zu\n", cfg.num_keys);
  std::printf("  value_size   = %zu bytes\n", cfg.value_size);
  std::printf("  batch_size   = %zu\n", cfg.batch_size);
  std::printf("  warmup_iters = %zu\n", cfg.warmup_iters);
  std::printf("  measure_iters= %zu\n", cfg.measure_iters);
  std::printf("  dram_capacity= %zu bytes\n", cfg.dram_capacity);
  std::printf("  ssd_capacity = %zu bytes\n", cfg.ssd_capacity);
  std::printf("  segment_size = %zu bytes\n", cfg.segment_size);
  std::printf("  ssd_backend  = %s\n", GetIoBackendSpec(cfg.ssd_io_backend).display_name);
  std::printf("  ssd_queue_d  = %zu\n", cfg.ssd_io_queue_depth);
  std::printf("  ssd_durability = %s\n", DurabilityLabel(cfg.ssd_durability_mode));
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
  BenchBatchRead(cfg, keys, values, results);
  BenchCopyToSSD(cfg, keys, values, results);
  BenchIOBackend(cfg, keys, values, results);
  BenchDurability(cfg, keys, values, results);
  BenchStorageIoDriver(cfg, values, results);
  BenchConcurrent(cfg, keys, values, results);
  BenchLeaderMode(cfg, keys, values, results);
  BenchCapacityPressure(cfg, keys, values, results);
  BenchE2E(cfg, e2e, results);

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
