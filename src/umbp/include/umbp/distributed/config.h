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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp {

// Forward declarations for strategy interfaces used by MasterServerConfig.
class RouteGetStrategy;
class RoutePutStrategy;

struct ClientRegistryConfig {
  std::chrono::seconds heartbeat_ttl{10};
  std::chrono::seconds reaper_interval{5};
  std::chrono::seconds allocation_ttl{30};
  std::chrono::seconds finalized_record_ttl{120};
  uint32_t max_missed_heartbeats = 3;

  // Default DRAM/HBM page_size used by every PageBitmapAllocator the
  // registry creates when the registering Client does not specify its own
  // (RegisterClientRequest.dram_page_size == 0).  All nodes within the same
  // tier must agree on page_size.
  uint64_t default_dram_page_size = 2ULL * 1024 * 1024;  // 2 MiB
};

struct MasterClientConfig {
  std::string master_address;
  std::string node_id;
  std::string node_address;
  bool auto_heartbeat = true;
};

struct EvictionConfig {
  double high_watermark = 0.9;
  double low_watermark = 0.7;
  std::chrono::seconds check_interval{5};
  std::chrono::seconds lease_duration{10};
  size_t evict_batch_size = 32;
};

struct MasterServerConfig {
  std::string listen_address = "0.0.0.0:50051";
  ClientRegistryConfig registry_config;
  EvictionConfig eviction_config;

  std::unique_ptr<RouteGetStrategy> get_strategy;
  std::unique_ptr<RoutePutStrategy> put_strategy;
};

struct ExportableDram {
  void* buffer = nullptr;
  size_t size = 0;
};

struct ExportableSsd {
  std::string dir;
  size_t capacity = 0;
};

struct PoolClientConfig {
  MasterClientConfig master_config;

  std::string io_engine_host;
  uint16_t io_engine_port = 0;

  size_t staging_buffer_size = 64ULL * 1024 * 1024;

  std::vector<ExportableDram> dram_buffers;
  std::vector<ExportableSsd> ssd_stores;

  std::map<TierType, TierCapacity> tier_capacities;

  uint16_t peer_service_port = 0;

  // Page size used by Master's PageBitmapAllocator for this node's DRAM/HBM
  // tier.  Reported via RegisterClient.  Same value applies to both DRAM
  // and HBM.  Master's ClientRegistry falls back to its own
  // `default_dram_page_size` if this is left at 0.  Defaults to 2 MiB.
  uint64_t dram_page_size = 2ULL * 1024 * 1024;
};

}  // namespace mori::umbp
