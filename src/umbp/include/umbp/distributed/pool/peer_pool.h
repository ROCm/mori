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

// Why a key is being removed. A tiered pool answers the two intents
// differently, so the caller has to say which one it means instead of leaving
// it to the tier configuration.
enum class PoolEvictMode {
  // Free bytes on the tier that holds the key. Where a tier configures
  // on_evict offload the key is demoted, not dropped: the master budgets per
  // (node, tier), so moving the bytes downstream does relieve the pressure it
  // measured, and the data stays reachable.
  kReclaim,
  // Drop the key from every backend on the peer regardless of tier
  // configuration, for when the value itself must stop existing.
  kDiscard,
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
  std::vector<EvictResult> Evict(const std::vector<std::string>& keys, PoolEvictMode mode);
  void ClearLocal();
  std::vector<KvEvent> DrainPendingEvents();
  std::vector<KvEvent> SnapshotOwnedKeysForFullSync();
  std::map<std::string, LogicalTierCapacity> LogicalTierCapacities() const;
  std::string LogicalTierForBackend(uint32_t backend_id) const;

  std::optional<uint32_t> PlacementBackend(const std::string& key) const;
  size_t PlacementCount() const;
  TierTransitionMetrics TransitionMetrics() const;
  // Reads this peer served, by the logical tier that held the key. Says whether
  // a tier policy is actually keeping hot data where it was supposed to.
  std::map<std::string, uint64_t> TierReadHits() const;

 private:
  struct PendingPlacement {
    uint32_t backend_id = BackendRegistry::kMaxBackends;
    uint64_t local_slot_id = 0;
    std::chrono::steady_clock::time_point expires_at =
        std::chrono::steady_clock::time_point::max();
  };

  struct TransitionJob {
    TierTransitionKind kind = TierTransitionKind::kOffload;
    std::string key;
    uint32_t source_backend_id = BackendRegistry::kMaxBackends;
    LogicalTierGraph::TierIndex source_tier = 0;
    uint8_t attempts = 0;
  };

  // What the copy phase needs, resolved while the pool lock is still held.
  struct TransitionPlan {
    bool valid = false;
    bool remove_source = false;
    std::vector<uint32_t> targets;
  };

  void EnqueueTransition(TransitionJob job);
  void TransitionWorkerLoop();
  void StopTransitionWorker();
  void TouchLocked(const std::string& key);
  void ForgetAccessLocked(const std::string& key);
  // Backend that currently holds the key, or kMaxBackends. The placement index
  // is process-local and can be stale or empty (after a restart, or once
  // another path evicted the key), so the backends are the authority.
  uint32_t FindOwnerLocked(const std::string& key) const;
  // Queues at most `max_count` offload candidates for one tier, oldest first,
  // and reports how many it queued. Bounded on purpose: a peer can hold
  // millions of placements, so neither the scan nor the queue may be
  // proportional to that.
  size_t EnqueueWatermarkCandidatesLocked(LogicalTierGraph::TierIndex tier,
                                          size_t max_count);
  void MaybeEnqueueWatermarkOffloadLocked();
  bool WatermarkDrivenLocked(const TransitionJob& job) const;
  TransitionPlan PlanTransitionLocked(const TransitionJob& job);
  void FinishTransitionLocked(const TransitionJob& job, const TierTransitionResult& result);
  // Plans under `lock`, releases it for the byte copy, then reacquires it to
  // publish the outcome. Holding it across the copy would stall every
  // allocate / commit / resolve on the peer for the duration of an SSD write.
  TierTransitionResult RunTransition(std::unique_lock<std::mutex>& lock,
                                     const TransitionJob& job);

  BackendRegistry* backends_;
  std::unique_ptr<PoolPolicy> policy_;
  TransferEngine* transfer_engine_;
  std::shared_ptr<const LogicalTierGraph> tier_graph_;
  TierTransitionExecutor transition_executor_;

  // Serializes policy decisions with backend lifecycle operations. The first
  // Pool favors correctness over concurrent dispatch; later policies can shard
  // this by key without changing the public interface.
  mutable std::mutex operation_mutex_;
  // Ordered so watermark offload can walk candidates in key order without
  // materializing and sorting every placement first.
  std::map<std::string, uint32_t> placements_;
  // A migration may install its target while an independent read lease delays
  // source deletion. Eviction retries must drain only that source, never fan
  // out and delete the durable target.
  std::unordered_map<std::string, std::unordered_set<uint32_t>> draining_sources_;
  // Keys whose bytes are being copied right now, with the pool lock released.
  // Eviction defers to the copy instead of racing it, and ClearLocal waits for
  // the set to drain so a late target commit cannot resurrect a wiped key.
  std::unordered_set<std::string> migrating_keys_;
  std::condition_variable transition_idle_cv_;
  std::unordered_map<std::string, PendingPlacement> pending_keys_;
  std::map<std::pair<uint32_t, uint64_t>, std::string> pending_slots_;
  std::unordered_map<std::string, uint64_t> last_access_;
  // Reverse index of last_access_, so the least recently used key is the first
  // element rather than the result of sorting every placement.
  std::map<uint64_t, std::string> access_order_;
  uint64_t access_clock_ = 0;
  // Earliest next scan per tier, moved forward only when a scan found nothing
  // to queue. A pressured tier with no candidates must not make every commit
  // batch pay for another walk.
  std::vector<std::chrono::steady_clock::time_point> tier_scan_backoff_;
  TierTransitionMetrics transition_metrics_;
  std::map<std::string, uint64_t> tier_read_hits_;

  std::mutex transition_mutex_;
  std::condition_variable transition_cv_;
  std::deque<TransitionJob> transition_queue_;
  std::unordered_set<std::string> queued_transition_keys_;
  // Outstanding jobs per source tier. A tier that already has work queued needs
  // no rescan, which is what keeps repeated commits off the placement map.
  std::vector<size_t> queued_by_tier_;
  bool stop_transition_worker_ = false;
  std::thread transition_worker_;
};

}  // namespace mori::umbp
