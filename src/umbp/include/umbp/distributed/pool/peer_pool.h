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

// What PeerPool::Evict did with one request.  A demotion is never run inside
// the call: it is queued to the pool's transition worker, so an EvictKey RPC
// returns without waiting on a copy.
enum class PoolEvictOutcome {
  kNone,            // nothing freed: no such copy, or a read lease / pin holds it
  kFreed,           // the copy was dropped; bytes_freed says how much
  kDemotionQueued,  // this call queued the copy's demotion; it frees the source when done
  kInFlight,        // a transition already owned the key; it will move or free it
};

struct PoolEvictResult {
  std::string key;
  // Bytes freed by the time Evict returned.  0 for a queued demotion, whose
  // bytes come back when the transition completes.
  uint64_t bytes_freed = 0;
  PoolEvictOutcome outcome = PoolEvictOutcome::kNone;
};

// Local watermark eviction counters (see PeerPool::EnableLocalEviction).  The
// keys and bytes actually freed are not here: every eviction is an Evict()
// call on a backend, and InstrumentedBackend already counts those.  These are
// what only the policy can see.
struct LocalEvictionMetrics {
  // Passes that found a backend at or above its high watermark.
  uint64_t rounds = 0;
  // Keys the passes evicted: dropped, or queued for demotion.
  uint64_t keys = 0;
  // Passes the medium offered no candidate for: everything it holds is under
  // a read lease or pin, or it keeps no eviction order at all.
  uint64_t no_candidate = 0;
  // Batches in which every candidate freed nothing (mid-transition, or busy by
  // the time Evict reached it).  The pass stops and the next wake retries.
  uint64_t stalled = 0;
};

// Why a key is being removed. A tiered pool answers the two intents
// differently, so the caller has to say which one it means instead of leaving
// it to the tier configuration.
enum class PoolEvictMode {
  // Free the bytes of ONE copy of the key: the one the request names (see
  // PoolEvictRequest). Where that copy's logical tier has an offload target the
  // key is demoted, not dropped -- the pressure being relieved is on the named
  // medium, so moving the bytes downstream relieves it and the data stays
  // reachable. Any other copy of the key is left alone.
  kReclaim,
  // Drop the key from every backend on the peer regardless of tier
  // configuration or scope, for when the value itself must stop existing.
  kDiscard,
};

// One key to evict, and which of its copies.  A key can live on several media
// at once (a copy-mode promotion leaves the lower copy behind), so "evict this
// key" is ambiguous: the request says which medium is under pressure.
struct PoolEvictRequest {
  std::string key;
  // The medium to free the key from -- the tier a master charged it to.
  // UNKNOWN means the medium that currently holds the key's placement, which
  // is how a request with no tier (an older master) is read.
  TierType tier = TierType::UNKNOWN;
  // One specific backend instead of a medium.  Local watermark eviction knows
  // exactly which backend is over its watermark.  Takes precedence over tier.
  uint32_t backend_id = BackendRegistry::kMaxBackends;
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
  std::vector<PoolResolvedEntry> BatchResolve(const std::vector<std::string>& keys,
                                              bool include_descs, bool allow_file_refs = false);
  // Existence only, in request order. Cheaper than BatchResolve, which
  // heap-copies the page vector of every key it finds -- work an existence
  // check discards. Takes no read lease and does not touch recency: asking
  // whether a key is here is not a read of it.
  std::vector<bool> BatchContains(const std::vector<std::string>& keys);
  // The one routing entry point for eviction.  Every policy calls this -- the
  // master's EvictKey RPC and this pool's own local watermark eviction alike --
  // and every byte it frees is freed by the medium's MediumBackend::Evict,
  // directly or as the last step of a demotion.  A copy that drops is freed
  // before this returns; a copy that demotes is only QUEUED (see
  // PoolEvictOutcome), so the call never waits on a byte copy.  One result per
  // request, in request order.
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

  // ---- local watermark eviction ----
  //
  // Turn on watermark eviction for one backend: once its usage reaches `high`,
  // the pool frees its coldest keys (MediumBackend::EvictionCandidates) down to
  // `low` by calling Evict() above -- the same call the master's EvictKey
  // makes, so a key whose tier has a downstream is demoted, not deleted.  It
  // runs on one background thread, started by the first call; a put only wakes
  // it and never evicts itself.  PoolClient enables it for every backend on a
  // node with no master, and for SSD always.
  //
  // Returns false and changes nothing for an unknown backend or watermarks
  // outside 0 < low < high <= 1.
  bool EnableLocalEviction(uint32_t backend_id, double high_watermark, double low_watermark);
  LocalEvictionMetrics LocalEvictionStats() const;
  // Test seam: one synchronous pass over every enabled backend, as the worker
  // runs it.  Returns the number of keys evicted: dropped, or queued for
  // demotion (which completes on the transition worker afterwards).
  size_t RunLocalEvictionOnceForTest() { return LocalEvictionPass(); }

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
    // Queued by Evict(): a demotion someone asked for -- the master's EvictKey
    // or local eviction -- not one the tier's own watermark started.  It runs
    // whatever the tier's usage is by then, and if nothing downstream can take
    // the key the copy is dropped instead: the eviction still has to free it.
    bool evict = false;
  };

  // What the copy phase needs, resolved while the pool lock is still held.
  struct TransitionPlan {
    bool valid = false;
    bool remove_source = false;
    std::vector<uint32_t> targets;
  };

  // False when the key already has a job (queued or running) or the worker is
  // stopping; the job is not queued then.
  bool EnqueueTransition(TransitionJob job);
  // Whether a transition job owns `key` right now, queued or running.
  bool TransitionQueued(const std::string& key);
  void TransitionWorkerLoop();
  void StopTransitionWorker();
  // An eviction's demotion could not move the copy (no room downstream, or the
  // copy moved meanwhile): drop it, as a synchronous eviction would have --
  // unless another transition owns the key, which would rather move it.
  void DropEvictedCopyLocked(const TransitionJob& job);

  struct LocalEvictionWatermarks {
    double high = 0.0;
    double low = 0.0;
  };
  // Cheap enough for the put path: one atomic exchange, and a notify only when
  // no wake is already pending.  Never reads capacity.
  void WakeLocalEviction();
  void LocalEvictionLoop();
  void StopLocalEviction();
  // One pass: every enabled backend, lowest tier first.  Returns keys evicted.
  size_t LocalEvictionPass();
  // Brings one backend down to its low watermark if it is at or above its
  // high one -- counting demotions still in flight -- and returns keys evicted.
  size_t ReclaimBackend(uint32_t backend_id, const LocalEvictionWatermarks& watermarks);

  void TouchLocked(const std::string& key);
  void ForgetAccessLocked(const std::string& key);
  // Whether this read should promote the key out of `source_tier`, per that
  // tier's trigger. Counts the read for an kOnHits tier, so it must only be
  // called once per served read and only when a promotion could actually run.
  bool PromoteOnReadLocked(const std::string& key, LogicalTierGraph::TierIndex source_tier);
  // Backend that currently holds the key, or kMaxBackends. The placement index
  // is process-local and can be stale or empty (after a restart, or once
  // another path evicted the key), so the backends are the authority.
  uint32_t FindOwnerLocked(const std::string& key) const;
  // The backend holding the copy an eviction request names, or kMaxBackends
  // when there is no such copy (see PoolEvictRequest for how scope resolves).
  uint32_t EvictSourceLocked(const PoolEvictRequest& request) const;
  // After copies of `key` were dropped: point the placement at a copy that is
  // still there, or forget the key if none is. Re-checks the backends rather
  // than trusting what was dropped, because drops run without the pool lock.
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

  // Resolve dispatch deliberately drops operation_mutex_ while thread-safe
  // backends do their work. ClearLocal takes this exclusively so it cannot
  // invalidate an SSD staging result while a resolve is between its plan and
  // publish phases.
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

  // ---- local watermark eviction (see EnableLocalEviction) ----
  // Guards local_evict_, stop_local_evict_ and the worker's start.
  mutable std::mutex local_evict_mutex_;
  std::condition_variable local_evict_cv_;
  std::map<uint32_t, LocalEvictionWatermarks> local_evict_;
  bool stop_local_evict_ = false;
  // Set by a wake, cleared by the worker at the start of a pass.  Atomic so a
  // commit that finds a wake already pending touches no lock at all.
  std::atomic<bool> local_evict_wake_{false};
  std::thread local_evict_worker_;
  // One pass at a time: the worker and the test seam share LocalEvictionPass.
  std::mutex local_evict_pass_mutex_;
  std::atomic<uint64_t> local_evict_rounds_{0};
  std::atomic<uint64_t> local_evict_keys_{0};
  std::atomic<uint64_t> local_evict_no_candidate_{0};
  std::atomic<uint64_t> local_evict_stalled_{0};
};

}  // namespace mori::umbp
