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
#include <mutex>
#include <optional>
#include <random>
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
  /// @p already_exists must be the same length as @p block_sizes; a mismatch is
  /// logged as a MORI ERROR and yields an all-nullopt result (best-effort, no
  /// throw).  Entries with already_exists[i]==true are master-side dedup hits:
  /// they return kAlreadyExists and consume no projected capacity.
  ///
  /// @p requester_node_id is the node that issued the batch put; node-affinity
  /// strategies use it to bias placement toward the writer's local node.  The
  /// default implementation ignores it and reuses the virtual Select()
  /// unchanged, preserving single-key placement semantics.  Override only to
  /// implement a smarter batch planner.
  ///
  /// NOTE: this virtual signature is an internal extension point.  Adding
  /// @p requester_node_id is a breaking change for any out-of-tree subclass.
  virtual std::vector<std::optional<RoutePutResult>> SelectBatch(
      const std::string& requester_node_id, const std::vector<uint64_t>& block_sizes,
      const std::vector<bool>& already_exists, std::vector<ClientRecord> candidates,
      const std::unordered_set<std::string>& exclude_nodes);
};

/// Default strategy: try direct-put tiers fastest-first (HBM -> DRAM),
/// pick the node with the most available space on the first tier that has capacity.
class TierAwareMostAvailableStrategy : public RoutePutStrategy {
 public:
  std::optional<RoutePutResult> Select(
      const std::vector<ClientRecord>& alive_clients, uint64_t block_size,
      const std::unordered_set<std::string>& exclude_nodes) override;
};

/// Configurable batch put strategy combining two orthogonal knobs:
///   - base select algorithm: most-available vs capacity-weighted random;
///   - node affinity: none / same-node / local-node.
///
/// Both knobs are wired from env vars at master startup
/// (UMBP_ROUTE_PUT_SELECT_ALGO / UMBP_ROUTE_PUT_NODE_AFFINITY).  Tier order is
/// always HBM -> DRAM; SSD is never a direct-put target.  Projected capacity is
/// deducted on the batch-local candidates copy exactly like the base
/// SelectBatch; nothing is written back to the registry.
class ConfigurableRoutePutStrategy : public RoutePutStrategy {
 public:
  enum class SelectAlgo { kMostAvailable, kRandom };
  enum class NodeAffinity { kNone, kSame, kLocal };

  ConfigurableRoutePutStrategy(SelectAlgo algo, NodeAffinity affinity);
  /// Test-only ctor: pins the RNG seed so capacity-weighted random draws are
  /// reproducible.  Production uses the thread_local RNG (no shared state).
  ConfigurableRoutePutStrategy(SelectAlgo algo, NodeAffinity affinity, uint64_t rng_seed);

  std::optional<RoutePutResult> Select(
      const std::vector<ClientRecord>& alive_clients, uint64_t block_size,
      const std::unordered_set<std::string>& exclude_nodes) override;

  std::vector<std::optional<RoutePutResult>> SelectBatch(
      const std::string& requester_node_id, const std::vector<uint64_t>& block_sizes,
      const std::vector<bool>& already_exists, std::vector<ClientRecord> candidates,
      const std::unordered_set<std::string>& exclude_nodes) override;

  /// Human-readable "algo/affinity" for startup logging.
  std::string Describe() const;

 private:
  /// Try only @p node_id on exactly @p tier; nullopt if it cannot fit
  /// @p block_size.  No cross-node, no cross-tier fallback.
  std::optional<RoutePutResult> TrySelectOnNodeTier(
      const std::vector<ClientRecord>& candidates, const std::string& node_id, TierType tier,
      uint64_t block_size, const std::unordered_set<std::string>& exclude_nodes) const;

  /// Try only @p node_id, HBM then DRAM; nullopt if it cannot fit @p block_size.
  /// Performs no cross-node fallback — that is the caller's explicit job.
  std::optional<RoutePutResult> TrySelectOnNode(
      const std::vector<ClientRecord>& candidates, const std::string& node_id, uint64_t block_size,
      const std::unordered_set<std::string>& exclude_nodes) const;

  /// Base algorithm, tier priority paramount (HBM before DRAM): walk tiers in
  /// order and route on the first tier that has room.  Within a tier, most-
  /// available picks the largest free space; random draws weighted by it.  When
  /// @p preferred_node is set and has room on the current tier, it wins over the
  /// algorithm — but only within that tier, so tier priority is never broken
  /// (a remote HBM node still beats the preferred node's DRAM).  This is the
  /// explicit global-fallback entry point.
  std::optional<RoutePutResult> SelectByAlgo(
      const std::vector<ClientRecord>& candidates, uint64_t block_size,
      const std::unordered_set<std::string>& exclude_nodes,
      const std::optional<std::string>& preferred_node = std::nullopt);

  /// Draw an index in [0, weights.size()) with probability proportional to the
  /// weights.  Uses the pinned RNG when seeded, else a thread_local RNG.
  size_t PickWeighted(const std::vector<uint64_t>& weights);

  SelectAlgo algo_;
  NodeAffinity affinity_;
  bool seeded_ = false;
  std::mt19937 rng_;
  std::mutex rng_mutex_;
};

}  // namespace mori::umbp
