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
  uint32_t max_missed_heartbeats = 3;
};

struct MasterClientConfig {
  std::string master_address;
  std::string node_id;
  std::string node_address;
  bool auto_heartbeat = true;
};

struct MasterServerConfig {
  std::string listen_address = "0.0.0.0:50051";
  ClientRegistryConfig registry_config;

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
};

}  // namespace mori::umbp
