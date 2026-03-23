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
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

enum class UMBPRole : int {
  Standalone = 0,
  SharedSSDLeader = 1,
  SharedSSDFollower = 2,
};

static constexpr uint32_t kAutoRankId = UINT32_MAX;

enum class UMBPSsdLayoutMode : int {
  SegmentedLog = 1,
};

enum class UMBPIoBackend : int {
  PThread = 0,
  IoUring = 1,
};

enum class UMBPDurabilityMode : int {
  Strict = 0,
  Relaxed = 1,
};

struct UMBPDramConfig {
  size_t capacity_bytes = 4ULL * 1024 * 1024 * 1024;
  bool use_shared_memory = false;
  std::string shm_name = "/umbp_dram";
  double high_watermark = 0.9;
  double low_watermark = 0.7;
};

struct UMBPIoConfig {
  UMBPIoBackend backend = UMBPIoBackend::IoUring;
  size_t queue_depth = 4096;
};

struct UMBPDurabilityConfig {
  UMBPDurabilityMode mode = UMBPDurabilityMode::Strict;
  bool enable_background_gc = true;
};

struct UMBPSsdConfig {
  bool enabled = true;
  std::string storage_dir = "/tmp/umbp_ssd";
  size_t capacity_bytes = 32ULL * 1024 * 1024 * 1024;
  UMBPSsdLayoutMode layout_mode = UMBPSsdLayoutMode::SegmentedLog;
  size_t segment_size_bytes = 256ULL * 1024 * 1024;
  UMBPIoConfig io;
  UMBPDurabilityConfig durability;
};

struct UMBPEvictionConfig {
  std::string policy = "lru";
  size_t candidate_window = 16;
  bool auto_promote_on_read = true;
};

struct UMBPCopyPipelineConfig {
  bool async_enabled = true;
  size_t queue_depth = 4096;
  size_t worker_threads = 2;
  size_t batch_max_ops = 128;
};

struct UMBPConfig {
  UMBPDramConfig dram;
  UMBPSsdConfig ssd;
  UMBPEvictionConfig eviction;
  UMBPCopyPipelineConfig copy_pipeline;

  // SPDK SSD tier configuration (only used when ssd_backend == "spdk")
  std::string ssd_backend = "posix";       // "posix" or "spdk"
  std::string spdk_bdev_name;              // e.g. "Malloc0" or "NVMe0n1"
  std::string spdk_reactor_mask = "0x1";   // CPU core mask for SPDK reactors
  int spdk_mem_size_mb = 256;              // DPDK hugepage limit (MB)
  std::string spdk_nvme_pci_addr;          // PCI BDF, e.g. "0000:47:00.0"
  std::string spdk_nvme_ctrl_name = "NVMe0";
  int spdk_io_workers = 4;                     // Internal I/O worker threads for SpdkSsdTier batch ops

  // SPDK Proxy configuration
  std::string spdk_proxy_shm_name = "/umbp_spdk_proxy";
  uint32_t spdk_proxy_rank_id = kAutoRankId;   // kAutoRankId = CAS auto-allocate (default)
  uint32_t spdk_proxy_max_ranks = 8;
  size_t spdk_proxy_data_per_rank_mb = 512;    // MB of SHM data region per rank
  std::string spdk_proxy_bin;                  // Path to spdk_proxy binary (empty = search PATH)
  int spdk_proxy_startup_timeout_ms = 30000;   // Max ms to wait for proxy READY

  // Role is the source of truth for runtime behavior.
  UMBPRole role = UMBPRole::Standalone;

  // Backward compatibility fields for older Python/C++ callers.
  // New code should set `role` instead.
  bool follower_mode = false;
  bool force_ssd_copy_on_write = false;

  UMBPRole ResolveRole() const {
    if (role != UMBPRole::Standalone) {
      return role;
    }
    if (follower_mode) {
      return UMBPRole::SharedSSDFollower;
    }
    if (force_ssd_copy_on_write) {
      return UMBPRole::SharedSSDLeader;
    }
    return UMBPRole::Standalone;
  }

  bool Validate(std::string* error_message = nullptr) const {
    if (dram.capacity_bytes == 0) {
      if (error_message) *error_message = "dram.capacity_bytes must be > 0";
      return false;
    }
    if (ssd.enabled) {
      if (ssd.capacity_bytes == 0) {
        if (error_message) *error_message = "ssd.capacity_bytes must be > 0";
        return false;
      }
      if (ssd.segment_size_bytes == 0) {
        if (error_message) *error_message = "ssd.segment_size_bytes must be > 0";
        return false;
      }
    }
    if (copy_pipeline.queue_depth == 0) {
      if (error_message) *error_message = "copy_pipeline.queue_depth must be > 0";
      return false;
    }
    if (copy_pipeline.worker_threads == 0) {
      if (error_message) *error_message = "copy_pipeline.worker_threads must be > 0";
      return false;
    }
    if (copy_pipeline.batch_max_ops == 0) {
      if (error_message) *error_message = "copy_pipeline.batch_max_ops must be > 0";
      return false;
    }
    return true;
  }

  static UMBPConfig FromEnvironment() {
    UMBPConfig cfg;
    auto getenv_str = [](const char* name, const std::string& def) -> std::string {
      const char* v = std::getenv(name);
      return v ? v : def;
    };
    auto getenv_size = [](const char* name, size_t def) -> size_t {
      const char* v = std::getenv(name);
      return v ? static_cast<size_t>(std::stoull(v)) : def;
    };
    auto getenv_int = [](const char* name, int def) -> int {
      const char* v = std::getenv(name);
      return v ? std::atoi(v) : def;
    };
    auto getenv_double = [](const char* name, double def) -> double {
      const char* v = std::getenv(name);
      return v ? std::atof(v) : def;
    };

    cfg.dram.capacity_bytes = getenv_size("UMBP_DRAM_CAPACITY", cfg.dram.capacity_bytes);
    cfg.ssd.enabled = getenv_int("UMBP_SSD_ENABLED", cfg.ssd.enabled ? 1 : 0) != 0;
    cfg.ssd.storage_dir = getenv_str("UMBP_SSD_DIR", cfg.ssd.storage_dir);
    cfg.ssd.capacity_bytes = getenv_size("UMBP_SSD_CAPACITY", cfg.ssd.capacity_bytes);
    cfg.eviction.policy = getenv_str("UMBP_EVICTION_POLICY", cfg.eviction.policy);
    cfg.dram.high_watermark = getenv_double("UMBP_DRAM_HIGH_WM", cfg.dram.high_watermark);
    cfg.dram.low_watermark = getenv_double("UMBP_DRAM_LOW_WM", cfg.dram.low_watermark);

    cfg.ssd_backend = getenv_str("UMBP_SSD_BACKEND", cfg.ssd_backend);
    if (cfg.ssd_backend == "posix" && !std::getenv("UMBP_SSD_BACKEND") &&
        std::getenv("UMBP_SPDK_NVME_PCI")) {
      cfg.ssd_backend = "spdk";
    }
    cfg.spdk_bdev_name = getenv_str("UMBP_SPDK_BDEV", cfg.spdk_bdev_name);
    cfg.spdk_reactor_mask = getenv_str("UMBP_SPDK_REACTOR_MASK", cfg.spdk_reactor_mask);
    cfg.spdk_mem_size_mb = getenv_int("UMBP_SPDK_MEM_MB", cfg.spdk_mem_size_mb);
    cfg.spdk_nvme_pci_addr = getenv_str("UMBP_SPDK_NVME_PCI", cfg.spdk_nvme_pci_addr);
    cfg.spdk_nvme_ctrl_name = getenv_str("UMBP_SPDK_NVME_CTRL", cfg.spdk_nvme_ctrl_name);
    cfg.spdk_io_workers = getenv_int("UMBP_SPDK_IO_WORKERS", cfg.spdk_io_workers);

    cfg.spdk_proxy_shm_name = getenv_str("UMBP_SPDK_PROXY_SHM", cfg.spdk_proxy_shm_name);
    const char* rank_env = std::getenv("UMBP_SPDK_PROXY_RANK");
    cfg.spdk_proxy_rank_id = rank_env ? static_cast<uint32_t>(std::atoi(rank_env)) : kAutoRankId;
    cfg.spdk_proxy_max_ranks = static_cast<uint32_t>(
        getenv_int("UMBP_SPDK_PROXY_MAX_RANKS", static_cast<int>(cfg.spdk_proxy_max_ranks)));
    cfg.spdk_proxy_data_per_rank_mb = getenv_size("UMBP_SPDK_PROXY_DATA_MB", cfg.spdk_proxy_data_per_rank_mb);
    cfg.spdk_proxy_bin = getenv_str("UMBP_SPDK_PROXY_BIN", cfg.spdk_proxy_bin);
    cfg.spdk_proxy_startup_timeout_ms = getenv_int("UMBP_SPDK_PROXY_TIMEOUT_MS", cfg.spdk_proxy_startup_timeout_ms);

    std::string role_str = getenv_str("UMBP_ROLE", "");
    if (role_str == "leader") cfg.role = UMBPRole::SharedSSDLeader;
    else if (role_str == "follower") cfg.role = UMBPRole::SharedSSDFollower;
    else if (role_str == "standalone") cfg.role = UMBPRole::Standalone;
    else if (role_str.empty() && cfg.role == UMBPRole::Standalone) {
      const char* local_rank = nullptr;
      for (const char* name : {"LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK",
                                "SLURM_LOCALID", "MPI_LOCALRANKID"}) {
        local_rank = std::getenv(name);
        if (local_rank) break;
      }
      if (local_rank) {
        cfg.role = (std::atoi(local_rank) == 0)
            ? UMBPRole::SharedSSDLeader : UMBPRole::SharedSSDFollower;
      }
    }

    return cfg;
  }
};
