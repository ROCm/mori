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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
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
  // Failed because another writer is storing this key right now.
  bool in_flight = false;
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

// What PeerPool::Evict did with one request.
enum class PoolEvictOutcome {
  kNone,            // nothing freed: no such copy, or a read lease / pin holds it
  kFreed,           // the copy was dropped
  kDemotionQueued,  // queued for the transition worker, which frees the source
  kInFlight,        // another transition owns the key and will move or free it
};

struct PoolEvictResult {
  std::string key;
  uint64_t bytes_freed = 0;  // by the time Evict returned; 0 for a queued demotion
  PoolEvictOutcome outcome = PoolEvictOutcome::kNone;
};

// Why a key is being removed. A tiered pool answers the two intents
// differently, so the caller has to say which one it means instead of leaving
// it to the tier configuration.
enum class PoolEvictMode {
  // Free the one copy the request names: demoted if its tier has an offload
  // target, else dropped.  Other copies are left alone.
  kReclaim,
  // Drop the key from every backend on the peer regardless of tier
  // configuration or scope, for when the value itself must stop existing.
  kDiscard,
};

// One key to evict, and which of its copies (a key can live on several media).
struct PoolEvictRequest {
  std::string key;
  TierType tier = TierType::UNKNOWN;                    // UNKNOWN: the medium holding the placement
  uint32_t backend_id = BackendRegistry::kMaxBackends;  // overrides tier when set
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

  std::vector<PoolAllocateResult> BatchAllocate(const std::vector<PoolPlacementRequest>& requests);
  std::vector<PoolCommitResult> BatchCommit(const std::vector<PoolCommitRequest>& requests);
  std::vector<bool> BatchAbort(const std::vector<PoolSlotRef>& slots);
  // `reader` and `extents` (parallel to `keys`, or null) are passed to every
  // backend asked (see MediumBackend::BatchResolve).
  std::vector<PoolResolvedEntry> BatchResolve(const std::vector<std::string>& keys,
                                              bool include_descs, bool allow_file_refs = false,
                                              ResolveReader reader = ResolveReader::kRemote,
                                              const std::vector<ObjectExtent>* extents = nullptr);
  // Existence only, in request order. Cheaper than BatchResolve, which
  // heap-copies the page vector of every key it finds -- work an existence
  // check discards. Takes no read lease and does not touch recency: asking
  // whether a key is here is not a read of it.
  std::vector<bool> BatchContains(const std::vector<std::string>& keys);
  // The single eviction entry point (master EvictKey and local eviction).
  // Drops happen before it returns; demotions are only queued.  One result per
  // request, in order.
  std::vector<PoolEvictResult> Evict(const std::vector<PoolEvictRequest>& requests,
                                     PoolEvictMode mode);
  // Every key unscoped: freed from the medium that currently holds it.
  std::vector<PoolEvictResult> Evict(const std::vector<std::string>& keys, PoolEvictMode mode);
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

  // Background watermark eviction for one backend: at `high` usage, Evict()
  // its coldest keys down to `low`.  Puts only wake the worker.  False for an
  // unknown backend or watermarks outside 0 < low < high <= 1.
  bool EnableLocalEviction(uint32_t backend_id, double high_watermark, double low_watermark);
  LocalEvictionMetrics LocalEvictionStats() const;
  // Test seam: one synchronous pass.  Returns keys dropped or queued.
  size_t RunLocalEvictionOnceForTest() { return LocalEvictionPass(); }
  // UMBP_EVICT_DEMOTION_RETRY_MS override; set it before queueing work.
  void SetEvictDemotionRetryBudgetForTest(std::chrono::milliseconds budget) {
    evict_demotion_retry_budget_ = budget;
  }

  // Victims for a backend over its watermark, coldest first, before the
  // medium's own order: (tier, logical tier, bytes) -> offers, or nullopt when
  // unavailable.  The master's GetEvictionCandidates in peer-driven mode.
  using EvictionCandidateSource = std::function<std::optional<std::vector<EvictionOffer>>(
      TierType tier, const std::string& logical_tier, uint64_t bytes)>;
  void SetEvictionCandidateSource(EvictionCandidateSource source);

 private:
  struct PendingPlacement {
    uint32_t backend_id = BackendRegistry::kMaxBackends;
    uint64_t local_slot_id = 0;
    std::chrono::steady_clock::time_point expires_at = std::chrono::steady_clock::time_point::max();
  };

  struct TransitionJob {
    TierTransitionKind kind = TierTransitionKind::kOffload;
    std::string key;
    uint32_t source_backend_id = BackendRegistry::kMaxBackends;
    LogicalTierGraph::TierIndex source_tier = 0;
    uint8_t attempts = 0;
    // An eviction demotion's retries are bounded by time from its first attempt.
    std::chrono::steady_clock::time_point first_attempt{};
    bool evict = false;  // from Evict(): ignores watermarks; drops once retries run out
  };

  // What the copy phase needs, resolved while the pool lock is still held.
  struct TransitionPlan {
    bool valid = false;
    bool remove_source = false;
    std::vector<uint32_t> targets;
  };

  // False if the key already has a job or the worker is stopping.
  bool EnqueueTransition(TransitionJob job);
  bool TransitionQueued(const std::string& key);  // queued or running
  void TransitionWorkerLoop();
  void StopTransitionWorker();
  void DropEvictedCopyLocked(const TransitionJob& job);

  struct LocalEvictionWatermarks {
    double high = 0.0;
    double low = 0.0;
  };
  void WakeLocalEviction();  // put path: one atomic exchange, no capacity read
  void LocalEvictionLoop();
  void StopLocalEviction();
  size_t LocalEvictionPass();  // lowest tier first; returns keys evicted
  size_t ReclaimBackend(uint32_t backend_id, const LocalEvictionWatermarks& watermarks);
  std::optional<std::vector<EvictionOffer>> RankedCandidates(uint32_t backend_id, TierType tier,
                                                             uint64_t bytes,
                                                             std::chrono::milliseconds wait);

  void TouchLocked(const std::string& key);
  void ForgetAccessLocked(const std::string& key);
  // Whether this read should promote the key out of `source_tier`, per that
  // tier's trigger. Counts the read for an kOnHits tier, so call it once per read.
  bool PromoteOnReadLocked(const std::string& key, LogicalTierGraph::TierIndex source_tier);
  // Promotes what this read should, before it runs, and repoints `preferred` at
  // each promoted copy so the read is served from there.
  void PromoteForRead(const std::vector<std::string>& keys,
                      std::vector<std::optional<uint32_t>>* preferred);
  // Backend that currently holds the key, or kMaxBackends. The placement index
  // is process-local and can be stale or empty (after a restart, or once
  // another path evicted the key), so the backends are the authority.
  uint32_t FindOwnerLocked(const std::string& key) const;
  // The backend holding the copy a request names, or kMaxBackends.
  uint32_t EvictSourceLocked(const PoolEvictRequest& request) const;
  // Re-points the placement at a surviving copy (or forgets the key); drops
  // run unlocked, so it re-checks the backends.
  void RepairAfterDropLocked(const std::string& key);
  // Queues at most `max_count` offload candidates for one tier, oldest first,
  // and reports how many it queued. Bounded on purpose: a peer can hold
  // millions of placements, so neither the scan nor the queue may be
  // proportional to that.
  size_t EnqueueWatermarkCandidatesLocked(LogicalTierGraph::TierIndex tier, size_t max_count);
  void MaybeEnqueueWatermarkOffloadLocked();
  bool WatermarkDrivenLocked(const TransitionJob& job) const;
  TransitionPlan PlanTransitionLocked(const TransitionJob& job);
  void FinishTransitionLocked(const TransitionJob& job, const TierTransitionResult& result);
  // Plans under `lock`, releases it for the byte copy, then reacquires it to
  // publish the outcome. Holding it across the copy would stall every
  // allocate / commit / resolve on the peer for the duration of an SSD write.
  TierTransitionResult RunTransition(std::unique_lock<std::mutex>& lock, const TransitionJob& job);

  BackendRegistry* backends_;
  std::unique_ptr<PoolPolicy> policy_;
  TransferEngine* transfer_engine_;
  std::shared_ptr<const LogicalTierGraph> tier_graph_;
  TierTransitionExecutor transition_executor_;
  // Only watermark-driven offload consumes recency order. On-evict policies
  // name their victim explicitly, and non-tiered pools never read the LRU.
  bool track_access_order_ = false;

  // Resolve and commit dispatch deliberately drop operation_mutex_ while
  // thread-safe backends do their work. ClearLocal takes this exclusively so it
  // cannot invalidate an SSD staging result while a resolve is between its plan
  // and publish phases, or wipe a key a commit is still writing.
  mutable std::shared_mutex lifecycle_mutex_;
  // Serializes policy decisions with backend lifecycle operations. The first
  // Pool still uses one metadata lock, but slow resolve dispatch does not hold
  // it; later policies can shard the protected state by key without changing
  // the public interface.
  mutable std::mutex operation_mutex_;
  // Every use is a point lookup, insert or erase. Keeping this unordered avoids
  // string comparisons in BatchResolve's publish critical section.
  std::unordered_map<std::string, uint32_t> placements_;
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
  std::unordered_map<std::string, std::list<std::string>::iterator> last_access_;
  // Reads served for a key from a tier whose trigger is kOnHits. Separate from
  // last_access_, which is an LRU position and carries no count. Cleared when the
  // promotion is queued and whenever the key leaves the access index, so it
  // cannot outlive the key it counts for.
  std::unordered_map<std::string, uint32_t> promote_hit_counts_;
  // Hottest key is at the front. Stable list iterators let a touch splice an
  // existing key without allocating a tree node or copying the key.
  std::list<std::string> access_order_;
  // Earliest next scan per tier, moved forward only when a scan found nothing
  // to queue. A pressured tier with no candidates must not make every commit
  // batch pay for another walk.
  std::vector<std::chrono::steady_clock::time_point> tier_scan_backoff_;
  TierTransitionMetrics transition_metrics_;
  std::map<std::string, uint64_t> tier_read_hits_;
  // Invalidates a resolve plan if a clear completed while it was dispatched.
  uint64_t clear_epoch_ = 0;

  std::mutex transition_mutex_;
  std::condition_variable transition_cv_;
  std::deque<TransitionJob> transition_queue_;
  std::unordered_set<std::string> queued_transition_keys_;
  // Outstanding jobs per source tier. A tier that already has work queued needs
  // no rescan, which is what keeps repeated commits off the placement map.
  std::vector<size_t> queued_by_tier_;
  bool stop_transition_worker_ = false;
  std::thread transition_worker_;
  std::chrono::milliseconds evict_demotion_retry_budget_{30000};  // transition worker only

  // Local watermark eviction.  local_evict_mutex_ guards local_evict_,
  // stop_local_evict_ and the worker's start.
  mutable std::mutex local_evict_mutex_;
  std::condition_variable local_evict_cv_;
  std::map<uint32_t, LocalEvictionWatermarks> local_evict_;
  bool stop_local_evict_ = false;
  std::atomic<bool> local_evict_wake_{false};  // lets a put skip the lock
  std::thread local_evict_worker_;
  std::mutex local_evict_pass_mutex_;  // one pass at a time (worker vs test seam)
  std::atomic<uint64_t> local_evict_rounds_{0};
  std::atomic<uint64_t> local_evict_keys_{0};
  std::atomic<uint64_t> local_evict_no_candidate_{0};
  std::atomic<uint64_t> local_evict_stalled_{0};
  std::atomic<uint64_t> local_evict_master_keys_{0};
  EvictionCandidateSource candidate_source_;  // guarded by local_evict_mutex_
  // At most one source query in flight per backend; pass thread only.  Last
  // member, so a pending query is joined before anything else goes away.
  struct RankedQuery {
    std::chrono::steady_clock::time_point started{};
    std::future<std::optional<std::vector<EvictionOffer>>> answer;
  };
  std::map<uint32_t, RankedQuery> ranked_queries_;
};

}  // namespace mori::umbp
