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
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "umbp/common/pool_allocator.h"

namespace mori::umbp {

enum class TierType : int {
  UNKNOWN = 0,
  HBM = 1,
  DRAM = 2,
  SSD = 3,
};

struct TierCapacity {
  uint64_t total_bytes = 0;
  uint64_t available_bytes = 0;
};

struct Location {
  std::string node_id;
  std::string location_id;  // Opaque handle from target node
  uint64_t size = 0;
  TierType tier = TierType::UNKNOWN;

  bool operator==(const Location& other) const;
};

enum class ClientStatus : int {
  UNKNOWN = 0,
  ALIVE = 1,
  EXPIRED = 2,
};

struct BlockMetrics {
  std::chrono::steady_clock::time_point created_at;
  std::chrono::steady_clock::time_point last_accessed_at;
  uint64_t access_count = 0;
};

struct ClientRecord {
  std::string node_id;
  std::string node_address;
  ClientStatus status = ClientStatus::UNKNOWN;
  std::chrono::steady_clock::time_point last_heartbeat;
  std::chrono::steady_clock::time_point registered_at;
  std::map<TierType, TierCapacity> tier_capacities;

  std::string peer_address;
  std::vector<uint8_t> engine_desc_bytes;

  std::vector<std::vector<uint8_t>> dram_memory_desc_bytes_list;
  std::vector<PoolAllocator> dram_allocators;
  std::vector<PoolAllocator> ssd_allocators;
};

// Helpers for logging
const char* TierTypeName(TierType t);
const char* ClientStatusName(ClientStatus s);

}  // namespace mori::umbp
