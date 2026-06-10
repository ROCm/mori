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
#include <unordered_set>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp {

/// Result of RoutePut: a routing advisory only.  Master owns no per-Put
/// state — the writer follows up with peer.AllocateSlot to actually
/// reserve capacity.  ENOSPC at peer triggers a retry with the failed
/// node added to the exclude set.
/// BatchRoutePut per-key outcome.  "Unavailable" is the outer
/// `std::optional<RoutePutResult>::nullopt`.
enum class RoutePutOutcome {
  kRouted,         ///< node_id / tier / peer_address populated
  kAlreadyExists,  ///< master-side dedup hit
};

struct RoutePutResult {
  RoutePutOutcome outcome = RoutePutOutcome::kRouted;
  std::string node_id;
  std::string peer_address;
  TierType tier = TierType::UNKNOWN;
};

/// Abstract interface for RoutePut node placement.
/// Implement this to plug in a custom write-path placement strategy.
class RoutePutStrategy {
 public:
  virtual ~RoutePutStrategy() = default;

  /// Select a target node from @p alive_clients that can accommodate
  /// @p block_size bytes.  Tier selection is the strategy's responsibility.
  /// Nodes whose `node_id` appears in @p exclude_nodes are skipped — the
  /// caller has already failed against them (typically ENOSPC at the
  /// peer's allocator) and would only fail again.
  /// @return nullopt if no suitable node exists.
  virtual std::optional<RoutePutResult> Select(
      const std::vector<ClientRecord>& alive_clients, uint64_t block_size,
      const std::unordered_set<std::string>& exclude_nodes) = 0;

  /// Batch-aware placement with projected capacity: each routed pick deducts
  /// the chosen node/tier's available_bytes in the by-value @p candidates
  /// copy so later entries in the same batch see the reservation.  The copy is
  /// batch-local and never written back to the registry — the peer allocator is
  /// still the final arbiter.  Result length and order match @p block_sizes.
  ///
  /// @p already_exists must be the same length as @p block_sizes (throws
  /// otherwise).  Entries with already_exists[i]==true are master-side dedup
  /// hits: they return kAlreadyExists and consume no projected capacity.
  ///
  /// The default implementation reuses the virtual Select() unchanged; it does
  /// not alter single-key placement semantics.  Override only to implement a
  /// smarter batch planner.
  virtual std::vector<std::optional<RoutePutResult>> SelectBatch(
      const std::vector<uint64_t>& block_sizes, const std::vector<bool>& already_exists,
      std::vector<ClientRecord> candidates, const std::unordered_set<std::string>& exclude_nodes);
};

/// Default strategy: try tiers fastest-first (HBM -> DRAM -> SSD),
/// pick the node with the most available space on the first tier that has capacity.
class TierAwareMostAvailableStrategy : public RoutePutStrategy {
 public:
  std::optional<RoutePutResult> Select(
      const std::vector<ClientRecord>& alive_clients, uint64_t block_size,
      const std::unordered_set<std::string>& exclude_nodes) override;
};

}  // namespace mori::umbp
