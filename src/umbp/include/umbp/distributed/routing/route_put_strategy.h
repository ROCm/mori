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

/// Result returned by RoutePutStrategy::Select and the Router::RoutePut path.
///
/// The strategy fills in only the placement-related fields (node_id,
/// node_address, tier) — Router::RoutePut then drives ClientRegistry
/// .AllocateForPut to populate everything below the divider.
///
/// DRAM/HBM layout under the new page-bitmap allocator:
///   `location_id`         canonical "0:p3,4;1:p0" string for this Put
///   `pages`               structured form of the same page set
///   `dram_memory_descs`   deduplicated MemoryDesc bytes for every distinct
///                         buffer_index referenced by `pages`
///   `page_size`           page size of the source allocator (bytes)
///
/// SSD layout (legacy capacity-only allocator):
///   The above DRAM/HBM fields are left empty / zero.  Allocation_id is
///   still set, and ssd_store_index identifies which SSD store was reserved.
struct RoutePutResult {
  std::string node_id;
  std::string node_address;
  TierType tier;

  std::string peer_address;
  std::vector<uint8_t> engine_desc_bytes;
  std::string allocation_id;

  // DRAM/HBM only.
  std::string location_id;
  std::vector<PageLocation> pages;
  std::vector<BufferMemoryDescBytes> dram_memory_descs;
  uint64_t page_size = 0;

  // SSD only.
  uint32_t ssd_store_index = 0;
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
