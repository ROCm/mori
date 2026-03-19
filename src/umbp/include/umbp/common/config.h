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
#include <string>

enum class UMBPRole : int {
  Standalone = 0,
  SharedSSDLeader = 1,
  SharedSSDFollower = 2,
};

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

struct UMBPConfig {
  // DRAM
  size_t dram_capacity_bytes = 4ULL * 1024 * 1024 * 1024;  // 4 GB
  bool use_shared_memory = false;                          // shm_open vs MAP_ANONYMOUS
  std::string shm_name = "/umbp_dram";                     // only used when use_shared_memory=true

  // SSD
  bool ssd_enabled = true;
  std::string ssd_storage_dir = "/tmp/umbp_ssd";
  size_t ssd_capacity_bytes = 32ULL * 1024 * 1024 * 1024;
  UMBPSsdLayoutMode ssd_layout_mode = UMBPSsdLayoutMode::SegmentedLog;
  UMBPIoBackend ssd_io_backend = UMBPIoBackend::IoUring;
  UMBPDurabilityMode ssd_durability_mode = UMBPDurabilityMode::Strict;
  size_t ssd_segment_size_bytes = 256ULL * 1024 * 1024;
  size_t ssd_batch_max_ops = 128;
  size_t ssd_queue_depth = 4096;
  size_t ssd_writer_threads = 2;
  bool ssd_enable_background_gc = true;

  // Policy: "lru" (default) or "prefix_aware_lru"
  std::string eviction_policy = "lru";
  // Number of LRU-tail candidates inspected when eviction_policy == "prefix_aware_lru".
  // Must be >= 1; values of 0 are treated as 1.
  size_t eviction_candidate_window = 16;
  bool auto_promote_on_read = true;
  double dram_high_watermark = 0.9;
  double dram_low_watermark = 0.7;

  // Role is the source of truth for runtime behavior.
  UMBPRole role = UMBPRole::Standalone;

  // Backward compatibility fields for older Python/C++ callers.
  // New code should set `role` instead.
  bool follower_mode = false;
  bool force_ssd_copy_on_write = false;
  bool copy_to_ssd_async = true;
  size_t copy_to_ssd_queue_depth = 4096;

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
};
