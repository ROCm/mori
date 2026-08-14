// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/pool/pool_policy.h"

namespace mori::umbp {

struct PoolAllocateResult {
  uint32_t backend_id = BackendRegistry::kMaxBackends;
  AllocateResult allocation;
};

struct PoolSlotRef {
  uint32_t backend_id = BackendRegistry::kMaxBackends;
  uint64_t local_slot_id = 0;
};

struct PoolCommitRequest {
  PoolSlotRef slot;
  std::string key;
};

struct PoolCommitResult {
  uint32_t backend_id = BackendRegistry::kMaxBackends;
  CommitResult commit;
};

struct PoolResolvedEntry {
  uint32_t backend_id = BackendRegistry::kMaxBackends;
  TierType tier = TierType::UNKNOWN;
  ResolvedEntry resolved;
};

// The implicit peer-local default Pool. It owns logical key placement and
// delegates backend choice to PoolPolicy; BackendRegistry continues to own the
// physical storage instances. This keeps the first Pool layer behaviorally
// compatible while creating the seam for weighted and tiered policies.
class PeerPool {
 public:
  PeerPool(BackendRegistry* backends, std::unique_ptr<PoolPolicy> policy);

  BackendRegistry* Backends() const { return backends_; }

  std::vector<PoolAllocateResult> BatchAllocate(
      const std::vector<PoolPlacementRequest>& requests);
  std::vector<PoolCommitResult> BatchCommit(const std::vector<PoolCommitRequest>& requests);
  std::vector<bool> BatchAbort(const std::vector<PoolSlotRef>& slots);
  std::vector<PoolResolvedEntry> BatchResolve(const std::vector<std::string>& keys,
                                              bool include_descs);
  std::vector<EvictResult> Evict(const std::vector<std::string>& keys);
  void ClearLocal();

  std::optional<uint32_t> PlacementBackend(const std::string& key) const;
  size_t PlacementCount() const;

 private:
  struct PendingPlacement {
    uint32_t backend_id = BackendRegistry::kMaxBackends;
    uint64_t local_slot_id = 0;
    std::chrono::steady_clock::time_point expires_at =
        std::chrono::steady_clock::time_point::max();
  };

  BackendRegistry* backends_;
  std::unique_ptr<PoolPolicy> policy_;

  // Serializes policy decisions with backend lifecycle operations. The first
  // Pool favors correctness over concurrent dispatch; later policies can shard
  // this by key without changing the public interface.
  mutable std::mutex operation_mutex_;
  std::unordered_map<std::string, uint32_t> placements_;
  std::unordered_map<std::string, PendingPlacement> pending_keys_;
  std::map<std::pair<uint32_t, uint64_t>, std::string> pending_slots_;
};

}  // namespace mori::umbp
