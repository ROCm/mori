// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "umbp/distributed/pool/pool_policy.h"
#include "umbp/distributed/pool/tier_transition.h"

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
  PeerPool(BackendRegistry* backends, std::unique_ptr<PoolPolicy> policy,
           TransferEngine* transfer_engine = nullptr);
  ~PeerPool();

  BackendRegistry* Backends() const { return backends_; }

  std::vector<PoolAllocateResult> BatchAllocate(
      const std::vector<PoolPlacementRequest>& requests);
  std::vector<PoolCommitResult> BatchCommit(const std::vector<PoolCommitRequest>& requests);
  std::vector<bool> BatchAbort(const std::vector<PoolSlotRef>& slots);
  std::vector<PoolResolvedEntry> BatchResolve(const std::vector<std::string>& keys,
                                              bool include_descs);
  std::vector<EvictResult> Evict(const std::vector<std::string>& keys);
  void ClearLocal();
  std::vector<KvEvent> DrainPendingEvents();
  std::vector<KvEvent> SnapshotOwnedKeysForFullSync();
  std::map<std::string, LogicalTierCapacity> LogicalTierCapacities() const;
  std::string LogicalTierForBackend(uint32_t backend_id) const;

  std::optional<uint32_t> PlacementBackend(const std::string& key) const;
  size_t PlacementCount() const;
  TierTransitionMetrics TransitionMetrics() const;

 private:
  struct PendingPlacement {
    uint32_t backend_id = BackendRegistry::kMaxBackends;
    uint64_t local_slot_id = 0;
    std::chrono::steady_clock::time_point expires_at =
        std::chrono::steady_clock::time_point::max();
  };

  enum class TransitionJobKind { kWatermark, kPromotion };
  struct TransitionJob {
    TransitionJobKind kind;
    std::string key;
    uint32_t source_backend_id;
    LogicalTierGraph::TierIndex source_tier;
    uint8_t attempts = 0;
  };

  void EnqueueTransition(TransitionJob job);
  void TransitionWorkerLoop();
  void StopTransitionWorker();
  TierTransitionResult RunTransitionLocked(const TransitionJob& job);

  BackendRegistry* backends_;
  std::unique_ptr<PoolPolicy> policy_;
  TransferEngine* transfer_engine_;
  std::shared_ptr<const LogicalTierGraph> tier_graph_;
  TierTransitionExecutor transition_executor_;

  // Serializes policy decisions with backend lifecycle operations. The first
  // Pool favors correctness over concurrent dispatch; later policies can shard
  // this by key without changing the public interface.
  mutable std::mutex operation_mutex_;
  std::unordered_map<std::string, uint32_t> placements_;
  // A migration may install its target while an independent read lease delays
  // source deletion. Eviction retries must drain only that source, never fan
  // out and delete the durable target.
  std::unordered_map<std::string, std::unordered_set<uint32_t>> draining_sources_;
  std::unordered_map<std::string, PendingPlacement> pending_keys_;
  std::map<std::pair<uint32_t, uint64_t>, std::string> pending_slots_;
  std::unordered_map<std::string, uint64_t> last_access_;
  uint64_t access_clock_ = 0;
  TierTransitionMetrics transition_metrics_;

  std::mutex transition_mutex_;
  std::condition_variable transition_cv_;
  std::deque<TransitionJob> transition_queue_;
  std::unordered_set<std::string> queued_transition_keys_;
  bool stop_transition_worker_ = false;
  std::thread transition_worker_;
};

}  // namespace mori::umbp
