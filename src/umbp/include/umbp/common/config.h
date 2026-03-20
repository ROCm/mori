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
};
