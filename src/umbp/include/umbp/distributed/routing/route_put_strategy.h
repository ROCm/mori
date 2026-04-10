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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp {

/// Result returned by RoutePutStrategy::Select.
struct RoutePutResult {
  std::string node_id;
  std::string node_address;
  TierType tier;

  std::string peer_address;
  std::vector<uint8_t> engine_desc_bytes;
  std::vector<uint8_t> dram_memory_desc_bytes;
  uint64_t allocated_offset = 0;
  uint32_t buffer_index = 0;
  std::string allocation_id;
};

/// Abstract interface for RoutePut node placement.
/// Implement this to plug in a custom write-path placement strategy.
class RoutePutStrategy {
 public:
  virtual ~RoutePutStrategy() = default;

  /// Select a target node from @p alive_clients that can accommodate
  /// @p block_size bytes. Tier selection is the strategy's responsibility.
  /// @return nullopt if no suitable node exists.
  virtual std::optional<RoutePutResult> Select(const std::vector<ClientRecord>& alive_clients,
                                               uint64_t block_size) = 0;
};

/// Default strategy: try tiers fastest-first (HBM -> DRAM -> SSD),
/// pick the node with the most available space on the first tier that has capacity.
class TierAwareMostAvailableStrategy : public RoutePutStrategy {
 public:
  std::optional<RoutePutResult> Select(const std::vector<ClientRecord>& alive_clients,
                                       uint64_t block_size) override;
};

}  // namespace mori::umbp
