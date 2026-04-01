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
//     --iters N                        Measurement iterations (default: 10)
//     --ssd-backend <posix|spdk>       SSD backend (default: posix)
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
#include "umbp/local/storage/segment/segment_format.h"
#include "umbp/local/storage/spdk_proxy_tier.h"
#include "umbp/local/storage/ssd_tier.h"
#include "umbp/local/umbp_client.h"

using namespace mori::umbp;

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Clock alias
// ---------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// BenchConfig
// ---------------------------------------------------------------------------
struct BenchConfig {
  size_t num_keys = 1000;
  size_t value_size = 4096;
  size_t batch_size = 64;
  size_t warmup_iters = 1;
  size_t measure_iters = 10;

  size_t dram_capacity = 64ULL * 1024 * 1024;
  size_t ssd_capacity = 256ULL * 1024 * 1024;
  size_t segment_size = 64ULL * 1024 * 1024;
  UMBPIoBackend ssd_io_backend = UMBPIoBackend::PThread;
  size_t ssd_io_queue_depth = 4096;
  UMBPDurabilityMode ssd_durability_mode = UMBPDurabilityMode::Relaxed;

  std::vector<int> thread_counts = {1, 2, 4, 8};

  std::string ssd_backend = "posix";  // "posix" or "spdk"

  std::string base_dir = "/tmp/umbp_bench";
  std::string filter;
  bool list_scenarios = false;
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

namespace mori::umbp::bench {

inline std::string ToLowerCopy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

inline bool FilterMatches(const std::string& filter, const std::string& scenario_name) {
  if (filter.empty()) return true;
  return ToLowerCopy(scenario_name).find(ToLowerCopy(filter)) != std::string::npos;
}

inline size_t CountTrue(const std::vector<bool>& values) {
  return static_cast<size_t>(std::count(values.begin(), values.end(), true));
}

inline size_t CountSuccessfulPages(const std::vector<bool>& key_results, size_t keys_per_page) {
  if (keys_per_page == 0) return 0;
  size_t ok_pages = 0;
  for (size_t i = 0; i < key_results.size(); i += keys_per_page) {
    bool page_ok = true;
    for (size_t j = 0; j < keys_per_page && i + j < key_results.size(); ++j) {
      page_ok = page_ok && key_results[i + j];
    }
    if (page_ok) ++ok_pages;
  }
  return ok_pages;
}

inline double ThroughputOpsPerSec(size_t successful_ops, double elapsed_sec) {
  return elapsed_sec > 0.0 ? static_cast<double>(successful_ops) / elapsed_sec : 0.0;
}

inline double ThroughputMbPerSec(size_t successful_bytes, double elapsed_sec) {
  return elapsed_sec > 0.0 ? static_cast<double>(successful_bytes) / (1024.0 * 1024.0) / elapsed_sec
                           : 0.0;
}

struct ResultTally {
  size_t requested_ops = 0;
  size_t successful_ops = 0;
  size_t requested_bytes = 0;
  size_t successful_bytes = 0;
  std::vector<double> latencies_us;

  void ReserveLatencySamples(size_t n) { latencies_us.reserve(n); }

  size_t failed_ops() const {
    return (requested_ops >= successful_ops) ? (requested_ops - successful_ops) : 0;
  }

  size_t sample_count() const { return latencies_us.size(); }

  void AddSample(size_t requested, size_t succeeded, size_t req_bytes, size_t ok_bytes,
                 double total_us) {
    requested_ops += requested;
    successful_ops += succeeded;
    requested_bytes += req_bytes;
    successful_bytes += ok_bytes;
    if (requested == 0) return;
    latencies_us.push_back(total_us);
  }

  void AddCall(size_t requested, size_t succeeded, size_t req_bytes, size_t ok_bytes,
               double total_us) {
    requested_ops += requested;
    successful_ops += succeeded;
    requested_bytes += req_bytes;
    successful_bytes += ok_bytes;
    latencies_us.push_back(total_us);
  }

  void AddOp(bool ok, size_t bytes, double total_us) {
    AddSample(1, ok ? 1 : 0, bytes, ok ? bytes : 0, total_us);
  }

  void Merge(const ResultTally& other) {
    requested_ops += other.requested_ops;
    successful_ops += other.successful_ops;
    requested_bytes += other.requested_bytes;
    successful_bytes += other.successful_bytes;
    latencies_us.insert(latencies_us.end(), other.latencies_us.begin(), other.latencies_us.end());
  }
};

enum class BatchWriteMode {
  Fused,
  Fallback,
};

inline size_t EstimateRecordBytes(size_t key_size, size_t value_size, size_t record_header_size) {
  return record_header_size + key_size + value_size;
}

inline size_t EstimateBatchRecordBytes(size_t keys_in_batch, size_t key_size, size_t value_size,
                                       size_t record_header_size) {
  return keys_in_batch * EstimateRecordBytes(key_size, value_size, record_header_size);
}

inline BatchWriteMode ClassifyBatchWriteMode(size_t keys_in_batch, size_t key_size,
                                             size_t value_size, size_t segment_size,
                                             size_t record_header_size) {
  return EstimateBatchRecordBytes(keys_in_batch, key_size, value_size, record_header_size) <=
                 segment_size
             ? BatchWriteMode::Fused
             : BatchWriteMode::Fallback;
}

struct WorkloadSummary {
  size_t num_keys = 0;
  size_t value_size = 0;
  size_t batch_size = 0;
  size_t dram_capacity = 0;
  size_t ssd_capacity = 0;
  size_t segment_size = 0;
  size_t key_size_hint = 0;
  size_t record_header_size = 0;
};

struct ConfigWarning {
  std::string message;
};

inline std::vector<ConfigWarning> CollectConfigWarnings(const WorkloadSummary& summary) {
  std::vector<ConfigWarning> warnings;
  if (summary.value_size > summary.dram_capacity) {
    warnings.push_back(
        {"value_size exceeds DRAM capacity; DRAM-resident and cold CopyToSSD scenarios will "
         "partially fail or be skipped."});
  }
  if (summary.value_size > summary.ssd_capacity) {
    warnings.push_back(
        {"value_size exceeds SSD capacity; single-value SSD writes cannot succeed."});
  }
  if (summary.num_keys * summary.value_size > summary.ssd_capacity) {
    warnings.push_back(
        {"working set exceeds SSD capacity; warm-cache SSD read scenarios will report mixed "
         "residency rather than full-hit throughput."});
  }
  if (summary.batch_size > 0 &&
      ClassifyBatchWriteMode(summary.batch_size, summary.key_size_hint, summary.value_size,
                             summary.segment_size,
                             summary.record_header_size) == BatchWriteMode::Fallback) {
    warnings.push_back(
        {"batch payload exceeds segment_size; WriteBatch-style scenarios will "
         "exercise fallback/per-key behavior instead of fused writes."});
  }
  return warnings;
}

}  // namespace mori::umbp::bench

namespace bench = mori::umbp::bench;

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
  const auto& spec = GetIoBackendSpec(backend);
  if (backend != UMBPIoBackend::IoUring) return "backend=" + std::string(spec.display_name);
  return "backend=" + std::string(spec.display_name) + "/qd=" + std::to_string(queue_depth);
}

static bool IsSpdk(const BenchConfig& cfg) { return cfg.ssd_backend == "spdk"; }

static UMBPConfig MakeBaseSsdConfig(const BenchConfig& cfg) {
  UMBPConfig ucfg = IsSpdk(cfg) ? UMBPConfig::FromEnvironment() : UMBPConfig();
  ucfg.ssd.enabled = true;
  ucfg.ssd_backend = cfg.ssd_backend;
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
  ucfg.ssd_backend = cfg.ssd_backend;
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
  ucfg.role = IsSpdk(cfg) ? UMBPRole::SharedSSDLeader : UMBPRole::Standalone;
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
struct DisplayTag {
  std::string key;
  std::string value;
};

struct BenchDisplay {
  std::string benchmark;
  std::string workload;
  std::vector<DisplayTag> scenario_tags;
  std::string latency_scope;
};

struct BenchResult {
  BenchDisplay display;
  size_t requested_ops = 0;
  size_t successful_ops = 0;
  size_t failed_ops = 0;
  size_t requested_bytes = 0;
  size_t successful_bytes = 0;
  size_t sample_count = 0;
  double elapsed_sec = 0.0;

  double lat_min_us = 0.0;
  double lat_avg_us = 0.0;
  double lat_p50_us = 0.0;
  double lat_p95_us = 0.0;
  double lat_p99_us = 0.0;
  double lat_max_us = 0.0;

  double throughput_ops_sec() const {
    return bench::ThroughputOpsPerSec(successful_ops, elapsed_sec);
  }
  double throughput_mb_sec() const {
    return bench::ThroughputMbPerSec(successful_bytes, elapsed_sec);
  }
};

static DisplayTag Tag(std::string key, std::string value) {
  return DisplayTag{std::move(key), std::move(value)};
}

static int DisplayTagOrder(const std::string& key) {
  static const std::vector<std::string> order = {
      "backend", "qd",       "durability", "path",  "source", "phase", "resident",
      "dram",    "ssd_seed", "dram_seed",  "batch", "bs",     "op",    "threads",
      "copy",    "fds",      "hit",        "dedup", "bytes",
  };
  for (size_t i = 0; i < order.size(); ++i) {
    if (order[i] == key) return static_cast<int>(i);
  }
  return static_cast<int>(order.size());
}

static BenchDisplay MakeDisplay(std::string benchmark, std::string workload = {},
                                std::initializer_list<DisplayTag> scenario_tags = {},
                                std::string latency_scope = {}) {
  BenchDisplay display;
  display.benchmark = std::move(benchmark);
  display.workload = std::move(workload);
  display.scenario_tags.assign(scenario_tags.begin(), scenario_tags.end());
  std::sort(display.scenario_tags.begin(), display.scenario_tags.end(),
            [](const DisplayTag& lhs, const DisplayTag& rhs) {
              int lhs_order = DisplayTagOrder(lhs.key);
              int rhs_order = DisplayTagOrder(rhs.key);
              if (lhs_order != rhs_order) return lhs_order < rhs_order;
              return lhs.key < rhs.key;
            });
  display.latency_scope = std::move(latency_scope);
  return display;
}

static bool ContainsTagKey(const std::vector<DisplayTag>& tags, const std::string& key) {
  return std::any_of(tags.begin(), tags.end(),
                     [&](const DisplayTag& tag) { return tag.key == key; });
}

static BenchDisplay ParseLegacyDisplay(const std::string& benchmark, const std::string& variant) {
  std::vector<std::string> parts;
  std::stringstream ss(variant);
  std::string part;
  while (std::getline(ss, part, '/')) {
    if (!part.empty()) parts.push_back(part);
  }

  std::string workload;
  size_t i = 0;
  if (parts.size() >= 3 && (parts[0] == "MLA" || parts[0] == "MHA") &&
      parts[1].find("pg") != std::string::npos && parts[2].find("KB") != std::string::npos) {
    workload = parts[0] + "/" + parts[1] + "/" + parts[2];
    i = 3;
  }

  std::vector<DisplayTag> tags;
  std::string explicit_latency_scope;
  for (; i < parts.size(); ++i) {
    const std::string& token = parts[i];
    size_t eq = token.find('=');
    if (eq == std::string::npos) continue;

    std::string key = token.substr(0, eq);
    std::string value = token.substr(eq + 1);
    if ((key == "resident" || key == "dram_seed" || key == "ssd_seed") && i + 1 < parts.size() &&
        parts[i + 1].find('=') == std::string::npos) {
      value += "/" + parts[++i];
    }

    if (key == "timing") {
      explicit_latency_scope = value;
      continue;
    }
    tags.push_back(Tag(key, value));
  }

  if (explicit_latency_scope.empty()) {
    if (benchmark.find("ExistsScan") != std::string::npos ||
        benchmark.find("PrefetchCycle") != std::string::npos) {
      explicit_latency_scope = "per-cycle";
    } else if (benchmark.rfind("E2E ", 0) == 0 && benchmark.find("Leader") == std::string::npos) {
      explicit_latency_scope = "per-batch";
    } else if (ContainsTagKey(tags, "batch") || ContainsTagKey(tags, "bs") ||
               ContainsTagKey(tags, "fds")) {
      explicit_latency_scope = "per-batch";
    } else {
      explicit_latency_scope = "per-op";
    }
  }

  BenchDisplay display;
  display.benchmark = benchmark;
  display.workload = workload;
  display.scenario_tags = std::move(tags);
  std::sort(display.scenario_tags.begin(), display.scenario_tags.end(),
            [](const DisplayTag& lhs, const DisplayTag& rhs) {
              int lhs_order = DisplayTagOrder(lhs.key);
              int rhs_order = DisplayTagOrder(rhs.key);
              if (lhs_order != rhs_order) return lhs_order < rhs_order;
              return lhs.key < rhs.key;
            });
  display.latency_scope = explicit_latency_scope;
  return display;
}

static std::string FormatScenario(const BenchDisplay& display) {
  if (display.scenario_tags.empty()) return "-";

  auto residency_tier_label = [&]() -> std::string {
    if (display.benchmark.rfind("DRAM ", 0) == 0) return "DRAM";
    if (display.benchmark.rfind("SSD ", 0) == 0) return "SSD";
    if (display.benchmark.rfind("IO Backend Read", 0) == 0) return "SSD";
    return "storage";
  };

  auto format_tag_key = [](const std::string& key) {
    if (key == "qd") return std::string("queue depth");
    if (key == "source") return std::string("read source");
    if (key == "dram") return std::string("DRAM fit");
    if (key == "ssd_seed") return std::string("SSD preloaded");
    if (key == "dram_seed") return std::string("DRAM preloaded");
    if (key == "batch") return std::string("batch mode");
    if (key == "bs") return std::string("batch size");
    if (key == "op") return std::string("operation");
    if (key == "copy") return std::string("copy mode");
    if (key == "fds") return std::string("files synced");
    if (key == "hit") return std::string("existing pages");
    if (key == "dedup") return std::string("reused pages");
    if (key == "bytes") return std::string("throughput");
    if (key == "resident") return std::string("residency");
    return key;
  };

  auto format_tag_value = [&](const std::string& key, const std::string& value) {
    if (key == "path") {
      if (value == "ssd-miss") return std::string("needs SSD copy");
      if (value == "ssd-hit") return std::string("already on SSD");
    }
    if (key == "resident") {
      if (value == "all") return "all data on " + residency_tier_label();
      return value + " on " + residency_tier_label();
    }
    if (key == "batch") {
      if (value == "fused") return std::string("fused write");
      if (value == "fallback") return std::string("per-key fallback");
    }
    if (key == "op") {
      if (value == "single") return std::string("single call");
      if (value == "batch") return std::string("batch call");
    }
    if (key == "copy") {
      if (value == "sync") return std::string("sync");
      if (value == "async") return std::string("async");
    }
    if (key == "source" && value == "ssd-only") return std::string("SSD only");
    if (key == "phase") {
      if (value == "fits-in-dram") return std::string("fits in DRAM");
      if (value == "spills-to-ssd") return std::string("spills to SSD");
      if (value == "read-after-spill") return std::string("read after spill");
      if (value == "pressure") return std::string("under pressure");
    }
    if (key == "hit" || key == "dedup" || key == "dram") {
      return value;
    }
    if (key == "bytes" && value == "physical") return std::string("physical writes only");
    return value;
  };

  std::ostringstream oss;
  for (size_t i = 0; i < display.scenario_tags.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << format_tag_key(display.scenario_tags[i].key) << "="
        << format_tag_value(display.scenario_tags[i].key, display.scenario_tags[i].value);
  }
  return oss.str();
}

static std::string FormatScopeLabel(const std::string& scope) {
  if (scope == "per-op") return "single call";
  if (scope == "per-batch") return "batch call";
  if (scope == "per-cycle") return "full cycle";
  if (scope == "end-to-end") return "full request";
  return scope;
}

static std::vector<std::string> WrapCell(std::string text, size_t width) {
  if (width == 0) return {text};
  if (text.empty()) return {"-"};

  std::vector<std::string> lines;
  while (text.size() > width) {
    size_t split = text.rfind(' ', width);
    if (split == std::string::npos || split == 0) {
      split = text.rfind(',', width);
      if (split != std::string::npos && split + 1 < text.size()) ++split;
    }
    if (split == std::string::npos || split == 0) split = width;

    lines.push_back(text.substr(0, split));
    text.erase(0, split);
    while (!text.empty() && text.front() == ' ') text.erase(text.begin());
  }
  if (!text.empty()) lines.push_back(text);
  if (lines.empty()) lines.push_back("-");
  return lines;
}

enum class OutputTableKind {
  Detail,
  Summary,
};

constexpr int kBenchmarkColWidth = 30;
constexpr int kWorkloadColWidth = 20;
constexpr int kScenarioColWidth = 58;
constexpr int kScopeColWidth = 12;

static void PrintSectionTitle(const std::string& section) {
  std::cout << "\n=== " << section << " ===" << std::endl;
}

static void PrintTableDivider() {
  constexpr size_t kDividerWidth = kBenchmarkColWidth + kWorkloadColWidth + kScenarioColWidth +
                                   kScopeColWidth + 8 * 4 + 10 + 9 * 6 + 12 + 18;
  std::printf("%s\n", std::string(kDividerWidth, '-').c_str());
}

static void PrintTableHeader(OutputTableKind kind) {
  if (kind == OutputTableKind::Detail) {
    std::printf("%-*s %-*s %-*s %-*s %8s %8s %8s %8s %10s %9s %9s %9s %9s %9s %9s %12s\n",
                kBenchmarkColWidth, "Benchmark", kWorkloadColWidth, "Workload", kScenarioColWidth,
                "Scenario", kScopeColWidth, "Scope", "req", "ok", "fail", "samples", "MB/s",
                "min(us)", "avg(us)", "p50(us)", "p95(us)", "p99(us)", "max(us)", "ops/s");
  } else {
    std::printf("%-*s %-*s %-*s %-*s %8s %8s %12s %10s\n", kBenchmarkColWidth, "Benchmark",
                kWorkloadColWidth, "Workload", kScenarioColWidth, "Scenario", kScopeColWidth,
                "Scope", "ok", "fail", "ops/s", "MB/s");
  }
  PrintTableDivider();
}

static void PrintSectionHeader(const std::string& section) {
  PrintSectionTitle(section);
  PrintTableHeader(OutputTableKind::Detail);
}

static void PrintDetailResult(const BenchResult& r) {
  auto benchmark_lines = WrapCell(r.display.benchmark, kBenchmarkColWidth);
  auto workload_lines =
      WrapCell(r.display.workload.empty() ? "-" : r.display.workload, kWorkloadColWidth);
  auto scenario_lines = WrapCell(FormatScenario(r.display), kScenarioColWidth);
  auto scope_lines =
      WrapCell(r.display.latency_scope.empty() ? "-" : FormatScopeLabel(r.display.latency_scope),
               kScopeColWidth);
  size_t line_count = std::max(std::max(benchmark_lines.size(), workload_lines.size()),
                               std::max(scenario_lines.size(), scope_lines.size()));

  for (size_t i = 0; i < line_count; ++i) {
    const char* benchmark = (i < benchmark_lines.size()) ? benchmark_lines[i].c_str() : "";
    const char* workload = (i < workload_lines.size()) ? workload_lines[i].c_str() : "";
    const char* scenario = (i < scenario_lines.size()) ? scenario_lines[i].c_str() : "";
    const char* scope = (i < scope_lines.size()) ? scope_lines[i].c_str() : "";
    if (i == 0) {
      std::printf(
          "%-*s %-*s %-*s %-*s %8zu %8zu %8zu %8zu %10.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f "
          "%12.0f\n",
          kBenchmarkColWidth, benchmark, kWorkloadColWidth, workload, kScenarioColWidth, scenario,
          kScopeColWidth, scope, r.requested_ops, r.successful_ops, r.failed_ops, r.sample_count,
          r.throughput_mb_sec(), r.lat_min_us, r.lat_avg_us, r.lat_p50_us, r.lat_p95_us,
          r.lat_p99_us, r.lat_max_us, r.throughput_ops_sec());
    } else {
      std::printf("%-*s %-*s %-*s %-*s\n", kBenchmarkColWidth, benchmark, kWorkloadColWidth,
                  workload, kScenarioColWidth, scenario, kScopeColWidth, scope);
    }
  }
}

static void PrintSummaryResult(const BenchResult& r) {
  auto benchmark_lines = WrapCell(r.display.benchmark, kBenchmarkColWidth);
  auto workload_lines =
      WrapCell(r.display.workload.empty() ? "-" : r.display.workload, kWorkloadColWidth);
  auto scenario_lines = WrapCell(FormatScenario(r.display), kScenarioColWidth);
  auto scope_lines =
      WrapCell(r.display.latency_scope.empty() ? "-" : FormatScopeLabel(r.display.latency_scope),
               kScopeColWidth);
  size_t line_count = std::max(std::max(benchmark_lines.size(), workload_lines.size()),
                               std::max(scenario_lines.size(), scope_lines.size()));

  for (size_t i = 0; i < line_count; ++i) {
    const char* benchmark = (i < benchmark_lines.size()) ? benchmark_lines[i].c_str() : "";
    const char* workload = (i < workload_lines.size()) ? workload_lines[i].c_str() : "";
    const char* scenario = (i < scenario_lines.size()) ? scenario_lines[i].c_str() : "";
    const char* scope = (i < scope_lines.size()) ? scope_lines[i].c_str() : "";
    if (i == 0) {
      std::printf("%-*s %-*s %-*s %-*s %8zu %8zu %12.0f %10.1f\n", kBenchmarkColWidth, benchmark,
                  kWorkloadColWidth, workload, kScenarioColWidth, scenario, kScopeColWidth, scope,
                  r.successful_ops, r.failed_ops, r.throughput_ops_sec(), r.throughput_mb_sec());
    } else {
      std::printf("%-*s %-*s %-*s %-*s\n", kBenchmarkColWidth, benchmark, kWorkloadColWidth,
                  workload, kScenarioColWidth, scenario, kScopeColWidth, scope);
    }
  }
}

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
// TierScope — RAII for SSDTier (POSIX) or external TierBackend* (SPDK)
// ---------------------------------------------------------------------------
struct TierScope {
  std::unique_ptr<ScopedTempDir> tmp;
  std::unique_ptr<SSDTier> local_tier;
  TierBackend* tier;

  // POSIX: create local SSDTier
  TierScope(const std::string& dir, size_t capacity, const UMBPConfig& ucfg)
      : tmp(std::make_unique<ScopedTempDir>(dir)),
        local_tier(std::make_unique<SSDTier>(tmp->path, capacity, ucfg)),
        tier(local_tier.get()) {}

  // SPDK: borrow external tier
  explicit TierScope(TierBackend* ext) : tier(ext) {}

  TierScope(const TierScope&) = delete;
  TierScope& operator=(const TierScope&) = delete;
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
static double PercentileValue(const std::vector<double>& latencies, double pct) {
  if (latencies.empty()) return 0.0;
  if (latencies.size() == 1) return latencies.front();
  double index = pct * static_cast<double>(latencies.size() - 1);
  size_t lo = static_cast<size_t>(index);
  size_t hi = std::min(lo + 1, latencies.size() - 1);
  double frac = index - static_cast<double>(lo);
  return latencies[lo] + (latencies[hi] - latencies[lo]) * frac;
}

static void ComputeLatencyStats(std::vector<double>& latencies, BenchResult& result) {
  if (latencies.empty()) return;
  std::sort(latencies.begin(), latencies.end());
  size_t n = latencies.size();
  result.lat_min_us = latencies.front();
  result.lat_max_us = latencies.back();
  result.lat_avg_us =
      std::accumulate(latencies.begin(), latencies.end(), 0.0) / static_cast<double>(n);
  result.lat_p50_us = PercentileValue(latencies, 0.50);
  result.lat_p95_us = PercentileValue(latencies, 0.95);
  result.lat_p99_us = PercentileValue(latencies, 0.99);
}

// ---------------------------------------------------------------------------
// Result formatting
// ---------------------------------------------------------------------------
static double WallSeconds(const Clock::time_point& start, const Clock::time_point& end) {
  return std::chrono::duration<double>(end - start).count();
}

static void PrintHeader(const std::string& section) { PrintSectionHeader(section); }

// Record a benchmark result: compute stats, print, and append to results vector.
static void RecordResult(BenchDisplay display, const bench::ResultTally& tally, double elapsed_sec,
                         std::vector<BenchResult>& results);
static void RecordResult(BenchDisplay display, size_t requested_ops, size_t requested_bytes,
                         double elapsed_sec, std::vector<double>& latencies,
                         std::vector<BenchResult>& results);

static void RecordResult(const std::string& name, const std::string& variant,
                         const bench::ResultTally& tally, double elapsed_sec,
                         std::vector<BenchResult>& results) {
  RecordResult(ParseLegacyDisplay(name, variant), tally, elapsed_sec, results);
}

static void RecordResult(BenchDisplay display, const bench::ResultTally& tally, double elapsed_sec,
                         std::vector<BenchResult>& results) {
  BenchResult r;
  r.display = std::move(display);
  r.requested_ops = tally.requested_ops;
  r.successful_ops = tally.successful_ops;
  r.failed_ops = tally.failed_ops();
  r.requested_bytes = tally.requested_bytes;
  r.successful_bytes = tally.successful_bytes;
  r.sample_count = tally.sample_count();
  r.elapsed_sec = elapsed_sec;
  auto latencies = tally.latencies_us;
  ComputeLatencyStats(latencies, r);
  PrintDetailResult(r);
  results.push_back(r);
}

static void RecordResult(const std::string& name, const std::string& variant, size_t requested_ops,
                         size_t requested_bytes, double elapsed_sec, std::vector<double>& latencies,
                         std::vector<BenchResult>& results) {
  RecordResult(ParseLegacyDisplay(name, variant), requested_ops, requested_bytes, elapsed_sec,
               latencies, results);
}

static void RecordResult(BenchDisplay display, size_t requested_ops, size_t requested_bytes,
                         double elapsed_sec, std::vector<double>& latencies,
                         std::vector<BenchResult>& results) {
  bench::ResultTally tally;
  tally.requested_ops = requested_ops;
  tally.successful_ops = requested_ops;
  tally.requested_bytes = requested_bytes;
  tally.successful_bytes = requested_bytes;
  tally.latencies_us = latencies;
  RecordResult(std::move(display), tally, elapsed_sec, results);
}

// ---------------------------------------------------------------------------
// Scenario filter check
// ---------------------------------------------------------------------------
static bool ShouldRun(const BenchConfig& cfg, const std::string& name) {
  return bench::FilterMatches(cfg.filter, name);
}

static const std::vector<std::string>& ScenarioFilters() {
  static const std::vector<std::string> filters = {
      "DRAM",      "SSD Tier",   "Batch",  "CopyToSSD", "IO Backend", "Durability",
      "StorageIo", "Concurrent", "Leader", "Capacity",  "E2E",
  };
  return filters;
}

static void PrintScenarioList() {
  std::printf("Available scenario filters:\n");
  for (const auto& name : ScenarioFilters()) {
    std::printf("  %s\n", name.c_str());
  }
}

static void PrintSkipMessage(const std::string& section, const std::string& reason) {
  std::cout << "\n=== " << section << " ===" << std::endl;
  std::printf("[skipped] %s\n", reason.c_str());
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

static size_t CountExistingKeys(const TierBackend* tier, const std::vector<std::string>& keys) {
  size_t existing = 0;
  for (const auto& key : keys) {
    if (tier->Exists(key)) ++existing;
  }
  return existing;
}

static std::string ResidentValue(size_t resident_keys, size_t total_keys) {
  if (resident_keys >= total_keys) return "all";
  std::ostringstream oss;
  oss << resident_keys << "/" << total_keys;
  return oss.str();
}

static std::string ResidencyVariantPrefix(size_t resident_keys, size_t total_keys) {
  return "resident=" + ResidentValue(resident_keys, total_keys);
}

static std::string JoinVariant(std::initializer_list<std::string> parts) {
  std::ostringstream oss;
  bool first = true;
  for (const auto& value : parts) {
    if (value.empty()) continue;
    if (!first) oss << "/";
    oss << value;
    first = false;
  }
  return oss.str();
}

static bool BatchFitsSegment(const WriteBatchDesc& desc, size_t segment_size) {
  size_t total_bytes = 0;
  for (size_t i = 0; i < desc.keys.size(); ++i) {
    total_bytes += sizeof(segment::RecordHeader) + desc.keys[i].size() + desc.sizes[i];
  }
  return total_bytes <= segment_size;
}

static std::string BatchModeValue(const std::vector<WriteBatchDesc>& descs, size_t segment_size) {
  for (const auto& desc : descs) {
    if (!BatchFitsSegment(desc, segment_size)) return "fallback";
  }
  return "fused";
}

static std::string BatchModeLabel(const std::vector<WriteBatchDesc>& descs, size_t segment_size) {
  return "batch=" + BatchModeValue(descs, segment_size);
}

static std::string BatchSizeValue(size_t batch_size) { return std::to_string(batch_size); }

static std::string BackendDisplayValue(UMBPIoBackend backend) {
  return GetIoBackendSpec(backend).display_name;
}

static std::string E2EWorkloadLabel(const E2EConfig& e2e) {
  std::string mode = (e2e.mode == E2EModelMode::MLA) ? "MLA" : "MHA";
  return mode + "/" + std::to_string(e2e.batch_pages) + "pg/" +
         std::to_string(e2e.ValueSizePerKey() / 1024) + "KB";
}

static void PrintConfigWarnings(const BenchConfig& cfg) {
  bench::WorkloadSummary summary;
  summary.num_keys = cfg.num_keys;
  summary.value_size = cfg.value_size;
  summary.batch_size = cfg.batch_size;
  summary.dram_capacity = cfg.dram_capacity;
  summary.ssd_capacity = cfg.ssd_capacity;
  summary.segment_size = cfg.segment_size;
  summary.key_size_hint = MakeKey(0).size();
  summary.record_header_size = sizeof(segment::RecordHeader);

  auto warnings = bench::CollectConfigWarnings(summary);
  for (const auto& warning : warnings) {
    std::printf("  warning      = %s\n", warning.message.c_str());
  }
}

// ---------------------------------------------------------------------------
// A. DRAM Tier Benchmarks
// ---------------------------------------------------------------------------
static void BenchDRAMTier(const BenchConfig& cfg, const std::vector<std::string>& keys,
                          const std::vector<std::vector<char>>& values,
                          std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "DRAM")) return;
  if (cfg.value_size > cfg.dram_capacity) {
    PrintSkipMessage("DRAM Tier",
                     "value_size exceeds dram_capacity; a single payload cannot fit in DRAM.");
    return;
  }

  PrintHeader("DRAM Tier");

  DRAMTier tier(cfg.dram_capacity);

  // Pre-fill for read benchmarks
  size_t resident_keys = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    if (tier.Write(keys[i], values[i].data(), values[i].size())) ++resident_keys;
  }
  std::string read_variant =
      JoinVariant({ResidencyVariantPrefix(resident_keys, keys.size()), "op=single"});

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

    bench::ResultTally tally;
    tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);

    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        bool ok = tier.Write(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        tally.AddOp(ok, cfg.value_size, us);
      }
    }
    auto run_end = Clock::now();

    RecordResult("DRAM Write", "op=single", tally, WallSeconds(run_start, run_end), results);
  }

  // Re-fill for reads
  tier.Clear();
  resident_keys = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    if (tier.Write(keys[i], values[i].data(), values[i].size())) ++resident_keys;
  }
  read_variant = JoinVariant({ResidencyVariantPrefix(resident_keys, keys.size()), "op=single"});

  // 2. DRAM Read (ReadIntoPtr)
  {
    std::vector<char> buf(cfg.value_size);

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (size_t i = 0; i < keys.size(); ++i) {
        tier.ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
      }
    }

    bench::ResultTally tally;
    tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);

    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        bool ok = tier.ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
        auto t1 = Clock::now();
        tally.AddOp(ok, cfg.value_size, std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto run_end = Clock::now();

    RecordResult("DRAM Read", read_variant, tally, WallSeconds(run_start, run_end), results);
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

    bench::ResultTally tally;
    tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);

    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        size_t sz = 0;
        auto t0 = Clock::now();
        const void* ptr = tier.ReadPtr(keys[i], &sz);
        auto t1 = Clock::now();
        tally.AddOp(ptr != nullptr && sz == cfg.value_size, cfg.value_size,
                    std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto run_end = Clock::now();

    RecordResult("DRAM ReadPtr (zero-copy)", read_variant, tally, WallSeconds(run_start, run_end),
                 results);
  }
}

// ---------------------------------------------------------------------------
// B. SSD Tier Benchmarks
// ---------------------------------------------------------------------------
static void BenchSSDTier(const BenchConfig& cfg, const std::vector<std::string>& keys,
                         const std::vector<std::vector<char>>& values,
                         std::vector<BenchResult>& results, TierBackend* ext_tier = nullptr) {
  if (!ShouldRun(cfg, "SSD Tier")) return;
  if (!ext_tier && cfg.value_size > cfg.ssd_capacity) {
    PrintSkipMessage("SSD Tier",
                     "value_size exceeds ssd_capacity; a single payload cannot fit in SSD.");
    return;
  }

  PrintHeader("SSD Tier");

  UMBPConfig ucfg = MakeBaseSsdConfig(cfg);

  // Write
  {
    auto ts = ext_tier ? TierScope(ext_tier)
                       : TierScope(cfg.base_dir + "/ssd_tier_write", cfg.ssd_capacity, ucfg);
    TierBackend& tier = *ts.tier;

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        tier.Write(keys[i], values[i].data(), values[i].size());
      }
    }
    tier.Clear();

    bench::ResultTally tally;
    tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);

    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        bool ok = tier.Write(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        tally.AddOp(ok, cfg.value_size, std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto run_end = Clock::now();

    RecordResult("SSD Tier Write", "op=single", tally, WallSeconds(run_start, run_end), results);
  }

  // BatchWrite
  {
    auto ts = ext_tier ? TierScope(ext_tier)
                       : TierScope(cfg.base_dir + "/ssd_tier_write_batch", cfg.ssd_capacity, ucfg);
    TierBackend& tier = *ts.tier;

    auto wdescs = BuildWriteBatches(keys, values, cfg.batch_size);
    std::string batch_variant = JoinVariant(
        {BatchModeLabel(wdescs, cfg.segment_size), "bs=" + std::to_string(cfg.batch_size)});

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      tier.Clear();
      for (const auto& d : wdescs) tier.BatchWrite(d.keys, d.data_ptrs, d.sizes);
    }
    tier.Clear();

    bench::ResultTally tally;
    tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);

    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      tier.Clear();
      for (const auto& d : wdescs) {
        auto t0 = Clock::now();
        auto write_results = tier.BatchWrite(d.keys, d.data_ptrs, d.sizes);
        auto t1 = Clock::now();
        size_t wrote = bench::CountTrue(write_results);
        tally.AddSample(d.keys.size(), wrote, d.keys.size() * cfg.value_size,
                        wrote * cfg.value_size,
                        std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto run_end = Clock::now();

    RecordResult("SSD Tier WriteBatch", batch_variant, tally, WallSeconds(run_start, run_end),
                 results);
  }

  // Read
  {
    auto ts = ext_tier ? TierScope(ext_tier)
                       : TierScope(cfg.base_dir + "/ssd_tier_read", cfg.ssd_capacity, ucfg);
    TierBackend& tier = *ts.tier;

    // Pre-fill
    size_t resident_keys = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (tier.Write(keys[i], values[i].data(), values[i].size())) ++resident_keys;
    }
    std::string read_variant =
        JoinVariant({ResidencyVariantPrefix(resident_keys, keys.size()), "op=single"});

    std::vector<char> buf(cfg.value_size);

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (size_t i = 0; i < keys.size(); ++i) {
        tier.ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
      }
    }

    bench::ResultTally tally;
    tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);

    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        bool ok = tier.ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
        auto t1 = Clock::now();
        tally.AddOp(ok, cfg.value_size, std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto run_end = Clock::now();

    RecordResult("SSD Tier Read", read_variant, tally, WallSeconds(run_start, run_end), results);
  }

  // BatchReadIntoPtr
  {
    auto ts = ext_tier ? TierScope(ext_tier)
                       : TierScope(cfg.base_dir + "/ssd_tier_read_batch", cfg.ssd_capacity, ucfg);
    TierBackend& tier = *ts.tier;

    // Pre-fill
    size_t resident_keys = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (tier.Write(keys[i], values[i].data(), values[i].size())) ++resident_keys;
    }
    std::string read_variant = JoinVariant({ResidencyVariantPrefix(resident_keys, keys.size()),
                                            "bs=" + std::to_string(cfg.batch_size)});

    // Prepare per-key read buffers and pre-build batch descriptors.
    std::vector<std::vector<char>> bufs(keys.size(), std::vector<char>(cfg.value_size));
    std::vector<uintptr_t> ptrs(keys.size());
    std::vector<size_t> sizes(keys.size(), cfg.value_size);
    for (size_t i = 0; i < keys.size(); ++i) {
      ptrs[i] = reinterpret_cast<uintptr_t>(bufs[i].data());
    }
    auto rdescs = BuildReadBatches(keys, ptrs, sizes, cfg.batch_size);

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (const auto& d : rdescs) tier.BatchReadIntoPtr(d.keys, d.dst_ptrs, d.sizes);
    }

    bench::ResultTally tally;
    tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);

    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (const auto& d : rdescs) {
        auto t0 = Clock::now();
        auto batch_results = tier.BatchReadIntoPtr(d.keys, d.dst_ptrs, d.sizes);
        auto t1 = Clock::now();
        size_t ok = bench::CountTrue(batch_results);
        tally.AddSample(d.keys.size(), ok, d.keys.size() * cfg.value_size, ok * cfg.value_size,
                        std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto run_end = Clock::now();

    RecordResult("SSD Tier ReadBatch", read_variant, tally, WallSeconds(run_start, run_end),
                 results);
  }
}

// ---------------------------------------------------------------------------
// C. Batch vs Single Write
// ---------------------------------------------------------------------------
static void BenchBatchWrite(const BenchConfig& cfg, const std::vector<std::string>& keys,
                            const std::vector<std::vector<char>>& values,
                            std::vector<BenchResult>& results, TierBackend* ext_tier = nullptr) {
  if (!ShouldRun(cfg, "Batch")) return;
  if (!ext_tier && cfg.value_size > cfg.ssd_capacity) {
    PrintSkipMessage("Batch vs Single Write",
                     "value_size exceeds ssd_capacity; SSD write comparisons cannot succeed.");
    return;
  }

  PrintHeader("Batch vs Single Write");

  UMBPConfig ucfg = MakeBaseSsdConfig(cfg);

  // Sequential single-key write
  {
    auto ts = ext_tier ? TierScope(ext_tier)
                       : TierScope(cfg.base_dir + "/batch_seq", cfg.ssd_capacity, ucfg);
    TierBackend& tier = *ts.tier;

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        tier.Write(keys[i], values[i].data(), values[i].size());
      }
    }
    tier.Clear();

    bench::ResultTally tally;
    tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);

    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        bool ok = tier.Write(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        tally.AddOp(ok, cfg.value_size, std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto run_end = Clock::now();

    RecordResult("SSD Write Compare", "op=single", tally, WallSeconds(run_start, run_end), results);
  }

  // BatchWrite
  {
    auto ts = ext_tier ? TierScope(ext_tier)
                       : TierScope(cfg.base_dir + "/batch_batch", cfg.ssd_capacity, ucfg);
    TierBackend& tier = *ts.tier;

    auto wdescs = BuildWriteBatches(keys, values, cfg.batch_size);
    std::string batch_variant = JoinVariant({"op=batch", BatchModeLabel(wdescs, cfg.segment_size),
                                             "bs=" + std::to_string(cfg.batch_size)});

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      tier.Clear();
      for (const auto& d : wdescs) tier.BatchWrite(d.keys, d.data_ptrs, d.sizes);
    }
    tier.Clear();

    bench::ResultTally tally;
    tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);

    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      tier.Clear();
      for (const auto& d : wdescs) {
        auto t0 = Clock::now();
        auto write_results = tier.BatchWrite(d.keys, d.data_ptrs, d.sizes);
        auto t1 = Clock::now();
        size_t wrote = bench::CountTrue(write_results);
        tally.AddSample(d.keys.size(), wrote, d.keys.size() * cfg.value_size,
                        wrote * cfg.value_size,
                        std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto run_end = Clock::now();

    RecordResult("SSD Write Compare", batch_variant, tally, WallSeconds(run_start, run_end),
                 results);
  }
}

// ---------------------------------------------------------------------------
// C2. Batch vs Single Read
// ---------------------------------------------------------------------------
static void BenchBatchRead(const BenchConfig& cfg, const std::vector<std::string>& keys,
                           const std::vector<std::vector<char>>& values,
                           std::vector<BenchResult>& results, TierBackend* ext_tier = nullptr) {
  if (!ShouldRun(cfg, "Batch")) return;
  if (!ext_tier && cfg.value_size > cfg.ssd_capacity) {
    PrintSkipMessage("Batch vs Single Read",
                     "value_size exceeds ssd_capacity; SSD read comparisons cannot succeed.");
    return;
  }

  PrintHeader("Batch vs Single Read");

  UMBPConfig ucfg = MakeBaseSsdConfig(cfg);

  // Sequential single-key read
  {
    auto ts = ext_tier ? TierScope(ext_tier)
                       : TierScope(cfg.base_dir + "/batch_read_seq", cfg.ssd_capacity, ucfg);
    TierBackend& tier = *ts.tier;

    size_t resident_keys = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (tier.Write(keys[i], values[i].data(), values[i].size())) ++resident_keys;
    }
    std::string seq_variant =
        JoinVariant({ResidencyVariantPrefix(resident_keys, keys.size()), "op=single"});

    std::vector<char> buf(cfg.value_size);

    // Warmup
    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (size_t i = 0; i < keys.size(); ++i) {
        tier.ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
      }
    }

    bench::ResultTally tally;
    tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);

    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        bool ok = tier.ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
        auto t1 = Clock::now();
        tally.AddOp(ok, cfg.value_size, std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto run_end = Clock::now();

    RecordResult("SSD Read Compare", seq_variant, tally, WallSeconds(run_start, run_end), results);
  }

  // BatchReadIntoPtr
  {
    auto ts = ext_tier ? TierScope(ext_tier)
                       : TierScope(cfg.base_dir + "/batch_read_batch", cfg.ssd_capacity, ucfg);
    TierBackend& tier = *ts.tier;

    size_t resident_keys = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (tier.Write(keys[i], values[i].data(), values[i].size())) ++resident_keys;
    }

    std::vector<std::vector<char>> bufs(keys.size(), std::vector<char>(cfg.value_size));
    std::vector<uintptr_t> ptrs(keys.size());
    std::vector<size_t> sizes(keys.size(), cfg.value_size);
    for (size_t i = 0; i < keys.size(); ++i) {
      ptrs[i] = reinterpret_cast<uintptr_t>(bufs[i].data());
    }
    auto rdescs = BuildReadBatches(keys, ptrs, sizes, cfg.batch_size);
    std::string batch_variant = JoinVariant({ResidencyVariantPrefix(resident_keys, keys.size()),
                                             "op=batch", "bs=" + std::to_string(cfg.batch_size)});

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (const auto& d : rdescs) tier.BatchReadIntoPtr(d.keys, d.dst_ptrs, d.sizes);
    }

    bench::ResultTally tally;
    tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);

    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (const auto& d : rdescs) {
        auto t0 = Clock::now();
        auto batch_results = tier.BatchReadIntoPtr(d.keys, d.dst_ptrs, d.sizes);
        auto t1 = Clock::now();
        size_t ok = bench::CountTrue(batch_results);
        tally.AddSample(d.keys.size(), ok, d.keys.size() * cfg.value_size, ok * cfg.value_size,
                        std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto run_end = Clock::now();

    RecordResult("SSD Read Compare", batch_variant, tally, WallSeconds(run_start, run_end),
                 results);
  }
}

// ---------------------------------------------------------------------------
// D. CopyToSSD vs CopyToSSDBatch
// ---------------------------------------------------------------------------
static void BenchCopyToSSD(const BenchConfig& cfg, const std::vector<std::string>& keys,
                           const std::vector<std::vector<char>>& values,
                           std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "CopyToSSD")) return;

  PrintHeader("CopyToSSD Paths");

  auto seed_dram = [&](LocalStorageManager& mgr) {
    size_t ready = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (mgr.Write(keys[i], values[i].data(), values[i].size(), StorageTier::CPU_DRAM)) ++ready;
    }
    return ready;
  };

  auto seed_ssd = [&](TierBackend* ssd) {
    size_t ready = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (ssd->Write(keys[i], values[i].data(), values[i].size())) ++ready;
    }
    return ready;
  };

  auto hot_variant = [&](size_t seeded, const std::string& suffix) {
    std::ostringstream oss;
    oss << "path=ssd-hit/ssd_seed=" << seeded << "/" << keys.size() << "/" << suffix;
    return oss.str();
  };

  auto cold_variant = [&](size_t seeded, const std::string& suffix) {
    std::ostringstream oss;
    oss << "path=ssd-miss/dram_seed=" << seeded << "/" << keys.size() << "/" << suffix;
    return oss.str();
  };

  // Single CopyToSSD, cold path.
  {
    ScopedTempDir tmp(cfg.base_dir + "/copy_single_cold");
    UMBPConfig ucfg = MakeStandaloneClientConfig(cfg, tmp.path + "/ssd", cfg.dram_capacity);
    fs::create_directories(ucfg.ssd.storage_dir);
    LocalStorageManager mgr(ucfg);
    TierBackend* ssd = mgr.GetTier(StorageTier::LOCAL_SSD);
    size_t dram_ready = seed_dram(mgr);

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (size_t i = 0; i < keys.size(); ++i) {
        mgr.CopyToSSD(keys[i]);
        if (ssd->Exists(keys[i])) ssd->Evict(keys[i]);
      }
    }

    bench::ResultTally tally;
    tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);
    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        bool ok = mgr.CopyToSSD(keys[i]);
        auto t1 = Clock::now();
        bool copied = ok && ssd->Exists(keys[i]);
        tally.AddOp(copied, cfg.value_size,
                    std::chrono::duration<double, std::micro>(t1 - t0).count());
        if (copied) ssd->Evict(keys[i]);
      }
    }
    auto run_end = Clock::now();

    RecordResult("CopyToSSD", cold_variant(dram_ready, "op=single"), tally,
                 WallSeconds(run_start, run_end), results);
  }

  // Single CopyToSSD, already-present fast path.
  {
    ScopedTempDir tmp(cfg.base_dir + "/copy_single_hot");
    UMBPConfig ucfg = MakeStandaloneClientConfig(cfg, tmp.path + "/ssd", cfg.dram_capacity);
    fs::create_directories(ucfg.ssd.storage_dir);
    LocalStorageManager mgr(ucfg);
    TierBackend* ssd = mgr.GetTier(StorageTier::LOCAL_SSD);
    size_t ssd_ready = seed_ssd(ssd);

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (size_t i = 0; i < keys.size(); ++i) {
        mgr.CopyToSSD(keys[i]);
      }
    }

    bench::ResultTally tally;
    tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);
    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        bool ok = mgr.CopyToSSD(keys[i]);
        auto t1 = Clock::now();
        tally.AddOp(ok && ssd->Exists(keys[i]), cfg.value_size,
                    std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto run_end = Clock::now();

    RecordResult("CopyToSSD", hot_variant(ssd_ready, "op=single"), tally,
                 WallSeconds(run_start, run_end), results);
  }

  auto kdescs = BuildKeyBatches(keys, cfg.batch_size);
  auto wdescs = BuildWriteBatches(keys, values, cfg.batch_size);
  std::string batch_mode = BatchModeLabel(wdescs, cfg.segment_size);

  // Batch CopyToSSDBatch, cold path.
  {
    ScopedTempDir tmp(cfg.base_dir + "/copy_batch_cold");
    UMBPConfig ucfg = MakeStandaloneClientConfig(cfg, tmp.path + "/ssd", cfg.dram_capacity);
    fs::create_directories(ucfg.ssd.storage_dir);
    LocalStorageManager mgr(ucfg);
    TierBackend* ssd = mgr.GetTier(StorageTier::LOCAL_SSD);
    size_t dram_ready = seed_dram(mgr);

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (const auto& batch : kdescs) {
        mgr.CopyToSSDBatch(batch);
        for (const auto& key : batch) {
          if (ssd->Exists(key)) ssd->Evict(key);
        }
      }
    }

    bench::ResultTally tally;
    tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);
    double measured_sec = 0.0;
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (const auto& batch : kdescs) {
        auto t0 = Clock::now();
        mgr.CopyToSSDBatch(batch);
        auto t1 = Clock::now();
        measured_sec += WallSeconds(t0, t1);
        tally.AddSample(batch.size(), batch.size(), batch.size() * cfg.value_size,
                        batch.size() * cfg.value_size,
                        std::chrono::duration<double, std::micro>(t1 - t0).count());
        // Reset SSD state for next iteration (untimed).
        for (const auto& key : batch) {
          if (ssd->Exists(key)) ssd->Evict(key);
        }
      }
    }

    RecordResult(
        "CopyToSSD",
        cold_variant(dram_ready, JoinVariant({batch_mode, "bs=" + std::to_string(cfg.batch_size)})),
        tally, measured_sec, results);
  }

  // Batch CopyToSSDBatch, already-present fast path.
  {
    ScopedTempDir tmp(cfg.base_dir + "/copy_batch_hot");
    UMBPConfig ucfg = MakeStandaloneClientConfig(cfg, tmp.path + "/ssd", cfg.dram_capacity);
    fs::create_directories(ucfg.ssd.storage_dir);
    LocalStorageManager mgr(ucfg);
    TierBackend* ssd = mgr.GetTier(StorageTier::LOCAL_SSD);
    size_t ssd_ready = seed_ssd(ssd);

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (const auto& batch : kdescs) mgr.CopyToSSDBatch(batch);
    }

    bench::ResultTally tally;
    tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);
    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (const auto& batch : kdescs) {
        auto t0 = Clock::now();
        mgr.CopyToSSDBatch(batch);
        auto t1 = Clock::now();
        // All keys pre-seeded on SSD; no need for CountExistingKeys here.
        tally.AddSample(batch.size(), batch.size(), batch.size() * cfg.value_size,
                        batch.size() * cfg.value_size,
                        std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto run_end = Clock::now();

    RecordResult(
        "CopyToSSD",
        hot_variant(ssd_ready, JoinVariant({batch_mode, "bs=" + std::to_string(cfg.batch_size)})),
        tally, WallSeconds(run_start, run_end), results);
  }
}

// ---------------------------------------------------------------------------
// E. IO Backend: POSIX vs io_uring
// ---------------------------------------------------------------------------
static void BenchIOBackend(const BenchConfig& cfg, const std::vector<std::string>& keys,
                           const std::vector<std::vector<char>>& values,
                           std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "IO Backend")) return;
  if (IsSpdk(cfg)) {
    PrintSkipMessage("IO Backend Sweep", "not applicable for SPDK backend");
    return;
  }
  if (cfg.value_size > cfg.ssd_capacity) {
    PrintSkipMessage("IO Backend Sweep",
                     "value_size exceeds ssd_capacity; backend SSD comparisons cannot succeed.");
    return;
  }

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

      bench::ResultTally tally;
      tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);

      auto run_start = Clock::now();
      for (size_t m = 0; m < cfg.measure_iters; ++m) {
        tier->Clear();
        for (size_t i = 0; i < keys.size(); ++i) {
          auto t0 = Clock::now();
          bool ok = tier->Write(keys[i], values[i].data(), values[i].size());
          auto t1 = Clock::now();
          tally.AddOp(ok, cfg.value_size,
                      std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
      }
      auto run_end = Clock::now();

      RecordResult("IO Backend Write", variant, tally, WallSeconds(run_start, run_end), results);
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
      size_t resident_keys = 0;
      for (size_t i = 0; i < keys.size(); ++i) {
        if (tier->Write(keys[i], values[i].data(), values[i].size())) ++resident_keys;
      }
      std::string read_variant =
          JoinVariant({variant, ResidencyVariantPrefix(resident_keys, keys.size())});

      std::vector<char> buf(cfg.value_size);

      // Warmup
      for (size_t w = 0; w < cfg.warmup_iters; ++w) {
        for (size_t i = 0; i < keys.size(); ++i) {
          tier->ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
        }
      }

      bench::ResultTally tally;
      tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);

      auto run_start = Clock::now();
      for (size_t m = 0; m < cfg.measure_iters; ++m) {
        for (size_t i = 0; i < keys.size(); ++i) {
          auto t0 = Clock::now();
          bool ok = tier->ReadIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
          auto t1 = Clock::now();
          tally.AddOp(ok, cfg.value_size,
                      std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
      }
      auto run_end = Clock::now();

      RecordResult("IO Backend Read", read_variant, tally, WallSeconds(run_start, run_end),
                   results);
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
  if (IsSpdk(cfg)) {
    PrintSkipMessage("Durability (Strict vs Relaxed)", "not applicable for SPDK backend");
    return;
  }
  if (cfg.value_size > cfg.ssd_capacity) {
    PrintSkipMessage(
        "Durability (Strict vs Relaxed)",
        "value_size exceeds ssd_capacity; durability write comparisons cannot succeed.");
    return;
  }

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

    bench::ResultTally tally;
    tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);

    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      tier.Clear();
      for (size_t i = 0; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        bool ok = tier.Write(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        tally.AddOp(ok, cfg.value_size, std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto run_end = Clock::now();

    RecordResult("SSD Write Durability", "durability=" + label, tally,
                 WallSeconds(run_start, run_end), results);
  };

  run_mode(UMBPDurabilityMode::Strict, "strict");
  run_mode(UMBPDurabilityMode::Relaxed, "relaxed");
}

// ---------------------------------------------------------------------------
// G. StorageIoDriver microbench
// ---------------------------------------------------------------------------
static void BenchStorageIoDriver(const BenchConfig& cfg,
                                 const std::vector<std::vector<char>>& values,
                                 std::vector<BenchResult>& results) {
  if (!ShouldRun(cfg, "StorageIoDriver")) return;
  if (IsSpdk(cfg)) {
    PrintSkipMessage("StorageIoDriver", "not applicable for SPDK backend");
    return;
  }

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

    auto run_start = Clock::now();
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
    auto run_end = Clock::now();

    RecordResult("Driver WriteAt", backend_variant, values.size() * cfg.measure_iters,
                 values.size() * cfg.measure_iters * cfg.value_size,
                 WallSeconds(run_start, run_end), latencies, results);
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

    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t i = 0; i < values.size(); ++i) {
        auto t0 = Clock::now();
        EnsureIoOk(driver->ReadAt(rw_fd.fd, buf.data(), buf.size(), i * cfg.value_size), "ReadAt");
        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto run_end = Clock::now();

    RecordResult("Driver ReadAt", backend_variant, values.size() * cfg.measure_iters,
                 values.size() * cfg.measure_iters * cfg.value_size,
                 WallSeconds(run_start, run_end), latencies, results);
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

    bench::ResultTally tally;
    tally.ReserveLatencySamples(values.size() * cfg.measure_iters);

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (const auto& ops : write_batches) {
        EnsureIoOk(driver->WriteBatch(ops), "warmup WriteBatch");
      }
    }

    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (const auto& ops : write_batches) {
        auto t0 = Clock::now();
        EnsureIoOk(driver->WriteBatch(ops), "WriteBatch");
        auto t1 = Clock::now();
        tally.AddSample(ops.size(), ops.size(), ops.size() * cfg.value_size,
                        ops.size() * cfg.value_size,
                        std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto run_end = Clock::now();

    RecordResult("Driver WriteBatch",
                 JoinVariant({backend_variant, "bs=" + std::to_string(cfg.batch_size)}), tally,
                 WallSeconds(run_start, run_end), results);
  }

  // --- ReadBatch ---
  {
    ScopedFd batch_fd = OpenBenchFile(tmp.path + "/read_batch.bin");
    for (size_t i = 0; i < values.size(); ++i) {
      EnsureIoOk(
          driver->WriteAt(batch_fd.fd, values[i].data(), values[i].size(), i * cfg.value_size),
          "prefill ReadBatch");
    }
    EnsureIoOk(driver->Sync(batch_fd.fd), "sync ReadBatch");

    size_t total_batches = (values.size() + cfg.batch_size - 1) / cfg.batch_size;
    std::vector<std::vector<char>> read_buffers(values.size(), std::vector<char>(cfg.value_size));
    std::vector<std::vector<IoReadOp>> read_batches;
    read_batches.reserve(total_batches);
    for (size_t i = 0; i < values.size(); i += cfg.batch_size) {
      size_t end = std::min(i + cfg.batch_size, values.size());
      std::vector<IoReadOp> ops;
      ops.reserve(end - i);
      for (size_t j = i; j < end; ++j) {
        ops.push_back(
            {batch_fd.fd, read_buffers[j].data(), read_buffers[j].size(), j * cfg.value_size});
      }
      read_batches.push_back(std::move(ops));
    }

    bench::ResultTally tally;
    tally.ReserveLatencySamples(values.size() * cfg.measure_iters);

    for (size_t w = 0; w < cfg.warmup_iters; ++w) {
      for (const auto& ops : read_batches) {
        EnsureIoOk(driver->ReadBatch(ops), "warmup ReadBatch");
      }
    }

    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (const auto& ops : read_batches) {
        auto t0 = Clock::now();
        EnsureIoOk(driver->ReadBatch(ops), "ReadBatch");
        auto t1 = Clock::now();
        tally.AddSample(ops.size(), ops.size(), ops.size() * cfg.value_size,
                        ops.size() * cfg.value_size,
                        std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto run_end = Clock::now();

    RecordResult("Driver ReadBatch",
                 JoinVariant({backend_variant, "bs=" + std::to_string(cfg.batch_size)}), tally,
                 WallSeconds(run_start, run_end), results);
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

    auto run_start = Clock::now();
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
    auto run_end = Clock::now();

    RecordResult("Driver Sync", backend_variant, values.size() * cfg.measure_iters, 0,
                 WallSeconds(run_start, run_end), latencies, results);
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
    bench::ResultTally tally;
    tally.ReserveLatencySamples(values.size() * cfg.measure_iters);

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

    auto run_start = Clock::now();
    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      for (size_t start = 0; start < values.size(); start += sync_group_size) {
        size_t count = write_group(start);
        std::vector<int> fds;
        fds.reserve(count);
        for (size_t i = 0; i < count; ++i) fds.push_back(sync_fds[i].fd);
        auto t0 = Clock::now();
        EnsureIoOk(driver->SyncMany(fds), "SyncMany");
        auto t1 = Clock::now();
        tally.AddSample(count, count, 0, 0,
                        std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    auto run_end = Clock::now();

    RecordResult("Driver SyncMany",
                 JoinVariant({backend_variant, "fds=" + std::to_string(sync_group_size)}), tally,
                 WallSeconds(run_start, run_end), results);
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

    std::string variant = "threads=" + std::to_string(nthreads);

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

      std::vector<bench::ResultTally> thread_tallies(nthreads);
      for (int t = 0; t < nthreads; ++t) {
        thread_tallies[t].ReserveLatencySamples(keys_per_thread * cfg.measure_iters);
      }
      double elapsed_sec = 0.0;

      for (size_t m = 0; m < cfg.measure_iters; ++m) {
        client.Clear();
        auto iter_start = Clock::now();
        std::vector<std::thread> threads;
        for (int t = 0; t < nthreads; ++t) {
          threads.emplace_back([&, t]() {
            size_t start = t * keys_per_thread;
            size_t end = start + keys_per_thread;
            for (size_t i = start; i < end; ++i) {
              auto t0 = Clock::now();
              bool ok = client.Put(keys[i], values[i].data(), values[i].size());
              auto t1 = Clock::now();
              thread_tallies[t].AddOp(ok, cfg.value_size,
                                      std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
          });
        }
        for (auto& th : threads) th.join();
        elapsed_sec += WallSeconds(iter_start, Clock::now());
      }

      bench::ResultTally tally;
      for (const auto& thread_tally : thread_tallies) tally.Merge(thread_tally);

      RecordResult("Concurrent Put", variant, tally, elapsed_sec, results);
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

      std::vector<bench::ResultTally> thread_tallies(nthreads);
      for (int t = 0; t < nthreads; ++t) {
        thread_tallies[t].ReserveLatencySamples(keys_per_thread * cfg.measure_iters);
      }
      double elapsed_sec = 0.0;

      for (size_t m = 0; m < cfg.measure_iters; ++m) {
        auto iter_start = Clock::now();
        std::vector<std::thread> threads;
        for (int t = 0; t < nthreads; ++t) {
          threads.emplace_back([&, t]() {
            std::vector<char> buf(cfg.value_size);
            size_t start = t * keys_per_thread;
            size_t end = start + keys_per_thread;
            for (size_t i = start; i < end; ++i) {
              auto t0 = Clock::now();
              bool ok =
                  client.GetIntoPtr(keys[i], reinterpret_cast<uintptr_t>(buf.data()), buf.size());
              auto t1 = Clock::now();
              thread_tallies[t].AddOp(ok, cfg.value_size,
                                      std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
          });
        }
        for (auto& th : threads) th.join();
        elapsed_sec += WallSeconds(iter_start, Clock::now());
      }

      bench::ResultTally tally;
      for (const auto& thread_tally : thread_tallies) tally.Merge(thread_tally);

      RecordResult("Concurrent Get", variant, tally, elapsed_sec, results);
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

    bench::ResultTally tally;
    tally.ReserveLatencySamples(keys.size() * cfg.measure_iters);
    double elapsed_sec = 0.0;

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      auto iter_start = Clock::now();
      {
        UMBPClient client(ucfg);
        for (size_t i = 0; i < keys.size(); ++i) {
          auto t0 = Clock::now();
          bool ok = client.Put(keys[i], values[i].data(), values[i].size());
          auto t1 = Clock::now();
          tally.AddOp(ok, cfg.value_size,
                      std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
      }
      elapsed_sec += WallSeconds(iter_start, Clock::now());
    }

    RecordResult("Leader Put", JoinVariant({label, "timing=end-to-end"}), tally, elapsed_sec,
                 results);
  };

  run_mode(false, "copy=sync");
  run_mode(true, "copy=async");
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

    bench::ResultTally tally;
    tally.ReserveLatencySamples(half * cfg.measure_iters);
    double elapsed_sec = 0.0;

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      client.Clear();
      auto iter_start = Clock::now();
      for (size_t i = 0; i < half; ++i) {
        auto t0 = Clock::now();
        bool ok = client.Put(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        tally.AddOp(ok, cfg.value_size, std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
      elapsed_sec += WallSeconds(iter_start, Clock::now());
    }

    RecordResult("Capacity Put", "phase=fits-in-dram/dram=50%", tally, elapsed_sec, results);
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

    bench::ResultTally tally;
    tally.ReserveLatencySamples(half * cfg.measure_iters);
    double elapsed_sec = 0.0;

    for (size_t m = 0; m < cfg.measure_iters; ++m) {
      client.Clear();
      // Write first half to fill DRAM
      for (size_t i = 0; i < half; ++i) {
        client.Put(keys[i], values[i].data(), values[i].size());
      }
      // Write second half — triggers eviction + demotion
      auto iter_start = Clock::now();
      for (size_t i = half; i < keys.size(); ++i) {
        auto t0 = Clock::now();
        bool ok = client.Put(keys[i], values[i].data(), values[i].size());
        auto t1 = Clock::now();
        tally.AddOp(ok, cfg.value_size, std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
      elapsed_sec += WallSeconds(iter_start, Clock::now());
    }

    RecordResult("Capacity Put", "phase=spills-to-ssd/dram=50%", tally, elapsed_sec, results);
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

  PrintSectionTitle("E2E UMBPClient (" + variant_label + ")");
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

  // Auto-scale iters to keep percentile rows reasonably stable.
  // Batch-style rows target >=100 batch samples; whole-cycle rows need a higher floor
  // because p99 from only ~100 calls is still noisy.
  // User --iters is treated as a minimum; we scale up if batches_per_iter is small.
  constexpr size_t kMinBatchSamples = 100;
  constexpr size_t kMinWholeCycleSamples = 200;
  size_t e2e_iters = cfg.measure_iters;
  if (batches_per_iter > 0 && batches_per_iter * e2e_iters < kMinBatchSamples) {
    e2e_iters = (kMinBatchSamples + batches_per_iter - 1) / batches_per_iter;
  }
  size_t total_samples = batches_per_iter * e2e_iters;
  size_t e2e_call_iters = std::max(e2e_iters, kMinWholeCycleSamples);
  std::printf("  e2e_iters     = %zu (auto-scaled from %zu; %zu batches/iter, %zu total samples)\n",
              e2e_iters, cfg.measure_iters, batches_per_iter, total_samples);
  if (e2e_call_iters != e2e_iters) {
    std::printf("  e2e_call_iters= %zu (whole-cycle latency rows)\n", e2e_call_iters);
  }
  PrintTableHeader(OutputTableKind::Detail);

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

  // Helper: fill pages [0, num_pages) via BatchPut, collecting per-page metrics.
  auto FillAllTimed = [&](UMBPClient& client, size_t num_pages, bench::ResultTally& tally) {
    for (size_t b = 0; b < num_pages; b += e2e.batch_pages) {
      size_t count = std::min(e2e.batch_pages, num_pages - b);
      auto keys = keygen.KeysForPages(b, count);
      std::vector<uintptr_t> ptrs;
      std::vector<size_t> sizes;
      host_buf.GetBatchMeta(b, count, ptrs, sizes);
      auto depths = GenerateDepths(e2e, b, count);
      auto t0 = Clock::now();
      auto batch_results = client.BatchPutFromPtrWithDepth(keys, ptrs, sizes, depths);
      auto t1 = Clock::now();
      size_t ok_pages = bench::CountSuccessfulPages(batch_results, keys_per_page);
      tally.AddSample(count, ok_pages, count * keys_per_page * value_size,
                      ok_pages * keys_per_page * value_size,
                      std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
  };

  auto FillAllTimedDedup = [&](UMBPClient& client, size_t num_pages, size_t prefill_pages,
                               bench::ResultTally& tally) {
    for (size_t b = 0; b < num_pages; b += e2e.batch_pages) {
      size_t count = std::min(e2e.batch_pages, num_pages - b);
      auto keys = keygen.KeysForPages(b, count);
      std::vector<uintptr_t> ptrs;
      std::vector<size_t> sizes;
      host_buf.GetBatchMeta(b, count, ptrs, sizes);
      auto depths = GenerateDepths(e2e, b, count);
      auto t0 = Clock::now();
      auto batch_results = client.BatchPutFromPtrWithDepth(keys, ptrs, sizes, depths);
      auto t1 = Clock::now();

      size_t ok_pages = bench::CountSuccessfulPages(batch_results, keys_per_page);
      size_t dedup_pages = 0;
      if (b < prefill_pages) {
        dedup_pages = std::min(count, prefill_pages - b);
      }
      size_t physical_write_pages = (ok_pages > dedup_pages) ? (ok_pages - dedup_pages) : 0;
      tally.AddSample(count, ok_pages, count * keys_per_page * value_size,
                      physical_write_pages * keys_per_page * value_size,
                      std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
  };

  // Helper: read all pages [0, num_pages) via BatchGet, collecting per-batch latencies.
  std::vector<char> read_buf;
  std::vector<uintptr_t> read_ptrs;
  std::vector<size_t> read_sizes;

  auto ReadAllTimed = [&](UMBPClient& client, size_t num_pages, bench::ResultTally& tally) {
    for (size_t b = 0; b < num_pages; b += e2e.batch_pages) {
      size_t count = std::min(e2e.batch_pages, num_pages - b);
      auto keys = keygen.KeysForPages(b, count);
      E2EHostBuffer::MakeReadMeta(count, keys_per_page, value_size, read_buf, read_ptrs,
                                  read_sizes);
      auto t0 = Clock::now();
      auto batch_results = client.BatchGetIntoPtr(keys, read_ptrs, read_sizes);
      auto t1 = Clock::now();
      size_t ok_pages = bench::CountSuccessfulPages(batch_results, keys_per_page);
      tally.AddSample(count, ok_pages, count * keys_per_page * value_size,
                      ok_pages * keys_per_page * value_size,
                      std::chrono::duration<double, std::micro>(t1 - t0).count());
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

    bench::ResultTally tally;
    tally.ReserveLatencySamples(e2e.num_pages * e2e_iters);

    auto run_start = Clock::now();
    for (size_t m = 0; m < e2e_iters; ++m) {
      client.Clear();
      FillAllTimed(client, e2e.num_pages, tally);
    }
    auto run_end = Clock::now();
    RecordResult("E2E BatchSet", variant_label, tally, WallSeconds(run_start, run_end), results);
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

    bench::ResultTally tally;
    tally.ReserveLatencySamples(e2e.num_pages * e2e_iters);
    auto run_start = Clock::now();
    for (size_t m = 0; m < e2e_iters; ++m) ReadAllTimed(client, e2e.num_pages, tally);
    auto run_end = Clock::now();
    RecordResult("E2E BatchGet", variant_label, tally, WallSeconds(run_start, run_end), results);
  }

  // ---------------------------------------------------------------
  // (c) E2E ExistsScan — BatchExistsConsecutive with partial fill
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

    bench::ResultTally tally;
    tally.ReserveLatencySamples(e2e_call_iters);
    size_t hit_pct = (e2e.num_pages == 0) ? 0 : ((fill_pages * 100) / e2e.num_pages);

    auto run_start = Clock::now();
    for (size_t m = 0; m < e2e_call_iters; ++m) {
      auto t0 = Clock::now();
      client.BatchExistsConsecutive(all_keys);
      auto t1 = Clock::now();
      tally.AddCall(/*requested=*/1, /*succeeded=*/1, /*req_bytes=*/0, /*ok_bytes=*/0,
                    std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    auto run_end = Clock::now();
    RecordResult("E2E ExistsScan",
                 JoinVariant({variant_label, "hit=" + std::to_string(hit_pct) + "%"}), tally,
                 WallSeconds(run_start, run_end), results);
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

    bench::ResultTally tally;
    tally.ReserveLatencySamples(e2e.num_pages * e2e_iters);

    UMBPClient client(ucfg);
    auto run_start = Clock::now();
    for (size_t m = 0; m < e2e_iters; ++m) {
      SeedPrefix(client);
      FillAllTimedDedup(client, e2e.num_pages, prefill_pages, tally);
    }
    auto run_end = Clock::now();
    RecordResult(
        "E2E Dedup",
        JoinVariant({variant_label,
                     "dedup=" + std::to_string(static_cast<int>(e2e.dedup_ratio * 100)) + "%",
                     "bytes=physical"}),
        tally, WallSeconds(run_start, run_end), results);
  }

  // ---------------------------------------------------------------
  // (e) E2E PrefetchCycle — exists → get pipeline (sglang prefetch flow)
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

    bench::ResultTally tally;
    tally.ReserveLatencySamples(e2e_call_iters);

    auto run_start = Clock::now();
    for (size_t m = 0; m < e2e_call_iters; ++m) {
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
      tally.AddCall(/*requested=*/1, /*succeeded=*/1,
                    /*req_bytes=*/e2e.num_pages * keys_per_page * value_size,
                    /*ok_bytes=*/hit_pages * keys_per_page * value_size,
                    std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    auto run_end = Clock::now();
    RecordResult("E2E PrefetchCycle", variant_label, tally, WallSeconds(run_start, run_end),
                 results);
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

      bench::ResultTally tally;
      tally.ReserveLatencySamples(e2e.num_pages * e2e_iters);

      auto run_start = Clock::now();
      for (size_t m = 0; m < e2e_iters; ++m) {
        client.Clear();
        FillAllTimed(client, e2e.num_pages, tally);
      }
      auto run_end = Clock::now();
      RecordResult("E2E Capacity Put",
                   JoinVariant({variant_label, "phase=spills-to-ssd", "dram=50%"}), tally,
                   WallSeconds(run_start, run_end), results);
    }

    // --- Read-back (early pages evicted to SSD) ---
    {
      UMBPClient client(ucfg);
      FillAll(client);  // first half evicted to SSD, second half in DRAM

      for (size_t w = 0; w < cfg.warmup_iters; ++w) ReadAll(client, e2e.num_pages);

      bench::ResultTally tally;
      tally.ReserveLatencySamples(e2e.num_pages * e2e_iters);

      auto run_start = Clock::now();
      for (size_t m = 0; m < e2e_iters; ++m) ReadAllTimed(client, e2e.num_pages, tally);
      auto run_end = Clock::now();
      RecordResult("E2E Capacity Get",
                   JoinVariant({variant_label, "phase=read-after-spill", "dram=50%"}), tally,
                   WallSeconds(run_start, run_end), results);
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

      bench::ResultTally tally;
      tally.ReserveLatencySamples(e2e.num_pages * e2e_iters);

      auto run_start = Clock::now();
      for (size_t m = 0; m < e2e_iters; ++m) {
        bench::ResultTally iter_tally;
        auto iter_start = Clock::now();
        {
          UMBPClient client(ucfg);  // fresh client per iter (destructor drains async)
          FillAllTimed(client, e2e.num_pages, iter_tally);
        }
        auto iter_end = Clock::now();
        tally.AddSample(iter_tally.requested_ops, iter_tally.successful_ops,
                        iter_tally.requested_bytes, iter_tally.successful_bytes,
                        std::chrono::duration<double, std::micro>(iter_end - iter_start).count());
      }
      auto run_end = Clock::now();
      RecordResult("E2E Leader Set", JoinVariant({variant_label, label, "timing=end-to-end"}),
                   tally, WallSeconds(run_start, run_end), results);
    };

    run_leader(false, "copy=sync");
    run_leader(true, "copy=async");
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

    bench::ResultTally tally;
    tally.ReserveLatencySamples(e2e.num_pages * e2e_iters);
    auto run_start = Clock::now();
    for (size_t m = 0; m < e2e_iters; ++m) ReadAllTimed(follower, e2e.num_pages, tally);
    auto run_end = Clock::now();
    RecordResult("E2E Follower Get", JoinVariant({variant_label, "source=ssd-only"}), tally,
                 WallSeconds(run_start, run_end), results);
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
      "  --list-scenarios                 Print available scenario filters and exit\n"
      "  --num-keys N                     Keys per scenario\n"
      "  --value-size N                   Value size in bytes\n"
      "  --batch-size N                   Batch size\n"
      "  --warmup-iters N                 Warmup iterations\n"
      "  --iters N                        Measurement iterations (default: 10)\n"
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
      "  --ssd-backend <posix|spdk>       SSD backend (default: posix)\n"
      "                                   spdk requires UMBP_SPDK_NVME_PCI env var\n"
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
    } else if (arg == "--list-scenarios") {
      cfg.list_scenarios = true;
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
    } else if (arg == "--ssd-backend" && i + 1 < argc) {
      std::string val = argv[++i];
      std::string lower = ToLower(val);
      if (lower != "posix" && lower != "spdk") {
        std::cerr << "Error: --ssd-backend must be 'posix' or 'spdk'\n";
        std::exit(1);
      }
      cfg.ssd_backend = lower;
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
  if (cfg.list_scenarios) {
    PrintScenarioList();
    return 0;
  }

  std::printf("UMBP Benchmark\n");
  std::printf("  num_keys     = %zu\n", cfg.num_keys);
  std::printf("  value_size   = %zu bytes\n", cfg.value_size);
  std::printf("  batch_size   = %zu\n", cfg.batch_size);
  std::printf("  warmup_iters = %zu\n", cfg.warmup_iters);
  std::printf("  measure_iters= %zu\n", cfg.measure_iters);
  std::printf("  dram_capacity= %zu bytes\n", cfg.dram_capacity);
  std::printf("  ssd_capacity = %zu bytes\n", cfg.ssd_capacity);
  std::printf("  segment_size = %zu bytes\n", cfg.segment_size);
  std::printf("  ssd_io_back  = %s\n", GetIoBackendSpec(cfg.ssd_io_backend).display_name);
  std::printf("  ssd_backend  = %s\n", cfg.ssd_backend.c_str());
  std::printf("  ssd_queue_d  = %zu\n", cfg.ssd_io_queue_depth);
  std::printf("  ssd_durability = %s\n", DurabilityLabel(cfg.ssd_durability_mode));
  std::printf("  base_dir     = %s\n", cfg.base_dir.c_str());
  if (!cfg.filter.empty()) {
    std::printf("  filter       = %s\n", cfg.filter.c_str());
  }
  PrintConfigWarnings(cfg);
  std::printf("  threads      =");
  for (int t : cfg.thread_counts) std::printf(" %d", t);
  std::printf("\n");

  // Pre-generate data
  std::printf("\nGenerating %zu keys x %zu bytes...\n", cfg.num_keys, cfg.value_size);
  auto keys = GenerateKeys(cfg.num_keys);
  auto values = GenerateValues(cfg.num_keys, cfg.value_size);
  std::printf("Data generation complete.\n");

  std::vector<BenchResult> results;

  // SPDK anchor — keeps the proxy daemon alive for the entire benchmark run.
  // All subsequent UMBPClient instances probe the existing proxy and reuse it
  // instead of spawning new ones (see LocalStorageManager proxy probe logic).
  std::unique_ptr<UMBPClient> spdk_anchor;
  TierBackend* ext_ssd = nullptr;
  if (IsSpdk(cfg)) {
    const char* pci = std::getenv("UMBP_SPDK_NVME_PCI");
    if (!pci || !pci[0]) {
      std::fprintf(stderr,
                   "ERROR: --ssd-backend spdk requires UMBP_SPDK_NVME_PCI env var.\n"
                   "  Example: UMBP_SPDK_NVME_PCI=0000:c1:00.0 ./bench_umbp --ssd-backend spdk\n"
                   "  Find NVMe PCI addresses with: lspci | grep -i nvme\n");
      return 1;
    }
    std::printf("\nInitializing SPDK backend (PCI: %s)...\n", pci);
    UMBPConfig acfg = MakeBaseSsdConfig(cfg);
    acfg.dram.capacity_bytes = cfg.dram_capacity;
    acfg.role = UMBPRole::SharedSSDLeader;
    acfg.copy_pipeline.async_enabled = false;
    acfg.eviction.auto_promote_on_read = false;
    spdk_anchor = std::make_unique<UMBPClient>(acfg);
    ext_ssd = spdk_anchor->Storage().GetTier(StorageTier::LOCAL_SSD);
    if (!ext_ssd || !dynamic_cast<SpdkProxyTier*>(ext_ssd)) {
      std::fprintf(stderr,
                   "ERROR: SPDK SSD tier not available. "
                   "Check UMBP_SPDK_NVME_PCI env var.\n");
      return 1;
    }
    std::printf("SPDK backend initialized.\n");
  }

  // Phase 1: tier-level benchmarks
  BenchDRAMTier(cfg, keys, values, results);
  BenchSSDTier(cfg, keys, values, results, ext_ssd);
  BenchBatchWrite(cfg, keys, values, results, ext_ssd);
  BenchBatchRead(cfg, keys, values, results, ext_ssd);

  // Phase 2: POSIX-specific (auto-skipped for SPDK inside each function)
  BenchIOBackend(cfg, keys, values, results);
  BenchDurability(cfg, keys, values, results);
  BenchStorageIoDriver(cfg, values, results);

  // Phase 3: client-level — each creates own UMBPClient that reuses the
  // anchor's proxy (probed as alive, no respawn needed).
  BenchCopyToSSD(cfg, keys, values, results);
  BenchConcurrent(cfg, keys, values, results);
  BenchLeaderMode(cfg, keys, values, results);
  BenchCapacityPressure(cfg, keys, values, results);
  BenchE2E(cfg, e2e, results);

  // Anchor destroyed here (after all benchmarks) — proxy shuts down.

  if (results.empty()) {
    if (!cfg.filter.empty()) {
      std::fprintf(stderr,
                   "Warning: filter '%s' matched no benchmark scenarios. Use --list-scenarios.\n",
                   cfg.filter.c_str());
    }
    return 1;
  }

  // Summary
  std::printf("\n=== Summary ===\n");
  PrintTableHeader(OutputTableKind::Summary);
  for (const auto& r : results) {
    PrintSummaryResult(r);
  }

  std::printf("\nDone. %zu benchmarks completed.\n", results.size());
  return 0;
}
