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
#include "umbp/distributed/pool/peer_pool.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <numeric>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "umbp/common/env_time.h"
#include "umbp/common/parallel_for.h"

namespace mori::umbp {

namespace {

// How long a retryable eviction demotion keeps retrying before its copy is
// dropped.  Time, not a count: the faults it rides out last a wall-clock
// interval, and a short queue would burn a count in milliseconds.
std::chrono::milliseconds EvictDemotionRetryBudget() {
  static const auto v = GetEnvMilliseconds("UMBP_EVICT_DEMOTION_RETRY_MS",
                                           std::chrono::milliseconds(30000), /*min_allowed=*/0);
  return v;
}
// Pause before a lone retry, so waiting out pressure does not spin the worker.
constexpr std::chrono::milliseconds kEvictRetryIdlePause{20};

// One watermark scan queues at most this many keys; the worker tops the queue
// up after each completed offload, so a tier still drains all the way down to
// its low watermark without ever queueing its whole key set at once.
constexpr size_t kWatermarkCandidateBatch = 64;
// Floor between scans. Without it every commit batch arriving above the high
// watermark would re-walk the placements.
constexpr auto kWatermarkScanInterval = std::chrono::milliseconds(5);

// Keys per Evict() call, so one call never holds the pool lock for a whole drain.
constexpr size_t kLocalEvictBatch = 64;
// Wakes are the main signal; the poll catches growth no wake reported.
constexpr auto kLocalEvictPollInterval = std::chrono::milliseconds(100);
constexpr auto kLocalEvictMinGap = std::chrono::milliseconds(1);  // between passes
// Per backend, between candidate-source queries.
constexpr auto kRankedCandidatesMinInterval = std::chrono::milliseconds(200);
// How long a pass waits on its first query (a small index answers in time); a
// slower answer is taken when ready, unless older than the max age.
constexpr auto kRankedCandidatesWait = std::chrono::milliseconds(20);
constexpr auto kRankedCandidatesMaxAge = std::chrono::seconds(3);

uint64_t UsedBytes(const TierCapacity& capacity) {
  return capacity.total_bytes > capacity.available_bytes
             ? capacity.total_bytes - capacity.available_bytes
             : 0;
}

}  // namespace

PeerPool::PeerPool(BackendRegistry* backends, std::unique_ptr<PoolPolicy> policy,
                   TransferEngine* transfer_engine)
    : backends_(backends),
      policy_(std::move(policy)),
      transfer_engine_(transfer_engine),
      tier_graph_(policy_ == nullptr ? nullptr : policy_->TierGraph()),
      transition_executor_(backends, transfer_engine) {
  if (tier_graph_ != nullptr) {
    // A tier graph without an engine would accept offload and promotion rules
    // and then silently fail every one of them, leaving the topology looking
    // configured while nothing ever moves.
    if (transfer_engine == nullptr) {
      throw std::invalid_argument(
          "PeerPool: a tiered policy requires a TransferEngine to move bytes between "
          "backends");
    }
    tier_scan_backoff_.assign(tier_graph_->TierCount(), std::chrono::steady_clock::time_point{});
    queued_by_tier_.assign(tier_graph_->TierCount(), 0);
    for (size_t tier = 0; tier < tier_graph_->TierCount(); ++tier) {
      const auto& node = tier_graph_->NodeAt(tier);
      if (node.trigger == PoolOffloadTrigger::kWatermark && !node.offload_to.empty()) {
        track_access_order_ = true;
        break;
      }
    }
    evict_demotion_retry_budget_ = EvictDemotionRetryBudget();
    transition_worker_ = std::thread(&PeerPool::TransitionWorkerLoop, this);
  }
}

PeerPool::~PeerPool() {
  StopLocalEviction();  // first: a pass may still be using the transition worker
  StopTransitionWorker();
}

bool PeerPool::EnqueueTransition(TransitionJob job) {
  std::lock_guard<std::mutex> lock(transition_mutex_);
  if (stop_transition_worker_ || !queued_transition_keys_.insert(job.key).second) return false;
  if (job.source_tier < queued_by_tier_.size()) ++queued_by_tier_[job.source_tier];
  transition_queue_.push_back(std::move(job));
  transition_cv_.notify_one();
  return true;
}

bool PeerPool::TransitionQueued(const std::string& key) {
  std::lock_guard<std::mutex> lock(transition_mutex_);
  return queued_transition_keys_.count(key) != 0;
}

void PeerPool::DropEvictedCopyLocked(const TransitionJob& job) {
  if (migrating_keys_.count(job.key) != 0) return;
  auto* source = backends_ == nullptr ? nullptr : backends_->Get(job.source_backend_id);
  if (source == nullptr) return;
  const auto evicted = source->Evict({job.key});
  if (!evicted.empty() && evicted.front().bytes_freed > 0) ++transition_metrics_.dropped;
  RepairAfterDropLocked(job.key);
}

void PeerPool::StopTransitionWorker() {
  {
    std::lock_guard<std::mutex> lock(transition_mutex_);
    stop_transition_worker_ = true;
    transition_queue_.clear();
    queued_transition_keys_.clear();
    std::fill(queued_by_tier_.begin(), queued_by_tier_.end(), 0);
  }
  transition_cv_.notify_all();
  if (transition_worker_.joinable()) transition_worker_.join();
}

void PeerPool::TouchLocked(const std::string& key) {
  if (!track_access_order_) return;
  auto access = last_access_.find(key);
  if (access != last_access_.end()) {
    access_order_.splice(access_order_.begin(), access_order_, access->second);
    return;
  }
  access_order_.push_front(key);
  last_access_.emplace(key, access_order_.begin());
}

void PeerPool::ForgetAccessLocked(const std::string& key) {
  promote_hit_counts_.erase(key);
  auto access = last_access_.find(key);
  if (access == last_access_.end()) return;
  access_order_.erase(access->second);
  last_access_.erase(access);
}

bool PeerPool::PromoteOnReadLocked(const std::string& key,
                                   LogicalTierGraph::TierIndex source_tier) {
  const auto& tier = tier_graph_->NodeAt(source_tier);
  switch (tier.promote_trigger) {
    case PoolPromoteTrigger::kNever:
      return false;
    case PoolPromoteTrigger::kOnRead:
      return true;
    case PoolPromoteTrigger::kOnHits: {
      const uint32_t hits = ++promote_hit_counts_[key];
      if (hits < tier.promote_hits) return false;
      // The count has done its job; leaving it behind would promote again on
      // the next read if this promotion does not move the key.
      promote_hit_counts_.erase(key);
      return true;
    }
  }
  return false;
}

uint32_t PeerPool::FindOwnerLocked(const std::string& key) const {
  if (policy_ == nullptr || backends_ == nullptr) return BackendRegistry::kMaxBackends;
  for (uint32_t backend_id : policy_->ReadOrder(*backends_)) {
    auto* backend = backends_->Get(backend_id);
    if (backend != nullptr && backend->Contains(key)) return backend_id;
  }
  return BackendRegistry::kMaxBackends;
}

size_t PeerPool::EnqueueWatermarkCandidatesLocked(LogicalTierGraph::TierIndex tier_index,
                                                  size_t max_count) {
  if (tier_graph_ == nullptr || max_count == 0) return 0;
  const auto& tier = tier_graph_->NodeAt(tier_index);
  if (tier.trigger != PoolOffloadTrigger::kWatermark || tier.offload_to.empty()) return 0;

  std::vector<TransitionJob> candidates;
  const auto consider = [&](const std::string& key, uint32_t backend_id) {
    if (tier_graph_->TierIndexForBackendId(backend_id) != tier_index) return;
    // Draining a member that is not itself under pressure frees nothing that
    // was scarce, and in a mixed-medium tier it would move keys off the roomy
    // member while the full one stays full.
    if (!tier_graph_->MemberAtOrAbove(backend_id, tier.low_watermark)) return;
    if (migrating_keys_.count(key) != 0) return;
    candidates.push_back({TierTransitionKind::kOffload, key, backend_id, tier_index});
  };

  // Coldest first: access_order_ keeps the hottest key at the front, so walking
  // it backwards reaches the keys least likely to be read again first.
  for (auto it = access_order_.rbegin(); it != access_order_.rend(); ++it) {
    const std::string& key = *it;
    auto placement = placements_.find(key);
    if (placement == placements_.end()) continue;
    consider(key, placement->second);
    if (candidates.size() >= max_count) break;
  }
  for (auto& candidate : candidates) EnqueueTransition(std::move(candidate));
  return candidates.size();
}

void PeerPool::MaybeEnqueueWatermarkOffloadLocked() {
  if (tier_graph_ == nullptr) return;
  const auto now = std::chrono::steady_clock::now();
  for (size_t tier_index = 0; tier_index < tier_graph_->TierCount(); ++tier_index) {
    const auto& tier = tier_graph_->NodeAt(tier_index);
    if (tier.trigger != PoolOffloadTrigger::kWatermark || tier.offload_to.empty() ||
        now < tier_scan_backoff_[tier_index] ||
        !tier_graph_->AtOrAbove(tier_index, tier.high_watermark)) {
      continue;
    }
    {
      // Work is already queued for this tier; the worker tops it up as it
      // drains, so there is nothing to decide here.
      std::lock_guard<std::mutex> lock(transition_mutex_);
      if (queued_by_tier_[tier_index] > 0) continue;
    }
    if (EnqueueWatermarkCandidatesLocked(tier_index, kWatermarkCandidateBatch) == 0) {
      tier_scan_backoff_[tier_index] = now + kWatermarkScanInterval;
    }
  }
}

bool PeerPool::WatermarkDrivenLocked(const TransitionJob& job) const {
  // An eviction's demotion runs whatever the tier's own watermark says.
  return !job.evict && job.kind == TierTransitionKind::kOffload && tier_graph_ != nullptr &&
         tier_graph_->NodeAt(job.source_tier).trigger == PoolOffloadTrigger::kWatermark;
}

PeerPool::TransitionPlan PeerPool::PlanTransitionLocked(const TransitionJob& job) {
  TransitionPlan plan;
  if (tier_graph_ == nullptr) return plan;
  auto placement = placements_.find(job.key);
  if (placement == placements_.end() || placement->second != job.source_backend_id) {
    return plan;
  }
  if (!migrating_keys_.insert(job.key).second) return plan;

  const bool promotion = job.kind == TierTransitionKind::kPromotion;
  const auto& source_tier = tier_graph_->NodeAt(job.source_tier);
  plan.remove_source = !promotion || source_tier.promote_mode == PoolTransitionMode::kMove;
  plan.targets = promotion ? tier_graph_->PromoteTargetOrder(job.source_tier, job.key)
                           : tier_graph_->TransitionTargetOrder(job.source_tier, job.key);
  plan.valid = true;
  ++transition_metrics_.attempted;
  return plan;
}

void PeerPool::FinishTransitionLocked(const TransitionJob& job,
                                      const TierTransitionResult& result) {
  migrating_keys_.erase(job.key);
  if (migrating_keys_.empty()) transition_idle_cv_.notify_all();
  // Before the outcome: a failed move may still have read and written.
  auto* source = backends_ == nullptr ? nullptr : backends_->Get(job.source_backend_id);
  if (source != nullptr && result.bytes_read != 0) {
    transition_metrics_.read_bytes[TierTypeName(source->Tier())] += result.bytes_read;
  }
  for (const auto& [tier, bytes] : result.bytes_written) {
    transition_metrics_.write_bytes[TierTypeName(tier)] += bytes;
  }
  if (!result.success) {
    ++transition_metrics_.failed;
    if (job.kind == TierTransitionKind::kPromotion) ++transition_metrics_.promote_failed;
    return;
  }
  ++transition_metrics_.succeeded;
  // The target medium just grew without passing through BatchCommit.
  WakeLocalEviction();
  if (job.kind == TierTransitionKind::kPromotion) {
    transition_metrics_.promoted_bytes += result.bytes_moved;
  } else {
    transition_metrics_.offloaded_bytes += result.bytes_moved;
  }
  // The copy ran with the pool lock released, so another operation may have
  // re-placed the key meanwhile. Only the placement this transition started
  // from may be advanced to the target.
  auto placement = placements_.find(job.key);
  if (placement != placements_.end() && placement->second == job.source_backend_id) {
    placement->second = result.target_backend_id;
  }
  if (result.source_draining) {
    draining_sources_[job.key].insert(job.source_backend_id);
  } else {
    auto draining = draining_sources_.find(job.key);
    if (draining != draining_sources_.end()) {
      draining->second.erase(job.source_backend_id);
      if (draining->second.empty()) draining_sources_.erase(draining);
    }
  }
  // A parked copy reused as the target (ALREADY_EXISTS) is durable again.
  auto draining = draining_sources_.find(job.key);
  if (draining != draining_sources_.end()) {
    draining->second.erase(result.target_backend_id);
    if (draining->second.empty()) draining_sources_.erase(draining);
  }
}

// The promoting read lands in the upper tier once and is served from there.  It
// does not wait for that tier to drop below its high watermark: the headroom
// above it is for this; a promotion that finds no room leaves the read where it is.
void PeerPool::PromoteForRead(const std::vector<std::string>& keys,
                              std::vector<std::optional<uint32_t>>* preferred) {
  std::vector<size_t> at;
  std::vector<TransitionJob> jobs;
  std::vector<TransitionPlan> plans;
  {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    if (tier_graph_ == nullptr) return;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (!(*preferred)[i].has_value()) continue;
      const auto source_tier = tier_graph_->TierIndexForBackendId(*(*preferred)[i]);
      if (!source_tier.has_value() || *source_tier == tier_graph_->EntryTierIndex() ||
          migrating_keys_.count(keys[i]) != 0 || !PromoteOnReadLocked(keys[i], *source_tier)) {
        continue;
      }
      TransitionJob job{TierTransitionKind::kPromotion, keys[i], *(*preferred)[i], *source_tier};
      TransitionPlan plan = PlanTransitionLocked(job);
      if (!plan.valid) continue;
      at.push_back(i);
      jobs.push_back(std::move(job));
      plans.push_back(std::move(plan));
    }
  }
  if (jobs.empty()) return;

  // Several at once, so a batch keeps more than one device read in flight.
  constexpr int kPromoteThreads = 8;
  std::vector<TierTransitionResult> results(jobs.size());
  ParallelFor(jobs.size(), kPromoteThreads, [&](size_t j) {
    results[j] = transition_executor_.Execute(jobs[j].key, jobs[j].source_backend_id,
                                              plans[j].targets, plans[j].remove_source);
  });

  std::lock_guard<std::mutex> lock(operation_mutex_);
  for (size_t j = 0; j < jobs.size(); ++j) {
    FinishTransitionLocked(jobs[j], results[j]);
    if (results[j].success) (*preferred)[at[j]] = results[j].target_backend_id;
  }
}

TierTransitionResult PeerPool::RunTransition(std::unique_lock<std::mutex>& lock,
                                             const TransitionJob& job) {
  TierTransitionResult result;
  const auto plan = PlanTransitionLocked(job);
  if (!plan.valid) return result;

  lock.unlock();
  result = transition_executor_.Execute(job.key, job.source_backend_id, plan.targets,
                                        plan.remove_source);
  lock.lock();
  FinishTransitionLocked(job, result);
  return result;
}

void PeerPool::TransitionWorkerLoop() {
  for (;;) {
    TransitionJob job;
    {
      std::unique_lock<std::mutex> lock(transition_mutex_);
      transition_cv_.wait(lock,
                          [&] { return stop_transition_worker_ || !transition_queue_.empty(); });
      if (stop_transition_worker_ && transition_queue_.empty()) return;
      job = std::move(transition_queue_.front());
      transition_queue_.pop_front();
    }
    bool retry = false;
    bool evict_retry = false;
    {
      std::unique_lock<std::mutex> lock(operation_mutex_);
      if (tier_graph_ != nullptr) {
        const auto& tier = tier_graph_->NodeAt(job.source_tier);
        const bool watermark_driven = WatermarkDrivenLocked(job);
        if (!watermark_driven || tier_graph_->AtOrAbove(job.source_tier, tier.low_watermark)) {
          if (job.evict && job.first_attempt == std::chrono::steady_clock::time_point{}) {
            job.first_attempt = std::chrono::steady_clock::now();
          }
          const auto result = RunTransition(lock, job);
          // The evicted copy is the only one: retry what can clear, and drop it
          // only once the budget is spent or the target can never take it.
          evict_retry =
              job.evict && !result.success && result.retryable &&
              std::chrono::steady_clock::now() - job.first_attempt < evict_demotion_retry_budget_;
          if (evict_retry) ++transition_metrics_.retried;
          if (job.evict && !result.success && !evict_retry) DropEvictedCopyLocked(job);
          const bool still_pressured =
              watermark_driven && tier_graph_->AtOrAbove(job.source_tier, tier.low_watermark);
          retry = (still_pressured && !result.success && job.attempts < 3) || evict_retry;
          // Keep the drain going without waiting for the next commit batch to
          // notice the tier is still above its low watermark. The scan skips
          // whatever is still queued, so this tops the batch back up rather
          // than duplicating it.
          if (still_pressured && result.success) {
            tier_scan_backoff_[job.source_tier] = std::chrono::steady_clock::time_point{};
            EnqueueWatermarkCandidatesLocked(job.source_tier, kWatermarkCandidateBatch);
          }
        }
      }
    }
    if (evict_retry) {
      bool idle = false;
      {
        std::lock_guard<std::mutex> lock(transition_mutex_);
        idle = transition_queue_.empty();
      }
      if (idle) std::this_thread::sleep_for(kEvictRetryIdlePause);
    } else if (retry) {
      std::this_thread::sleep_for(std::chrono::milliseconds(uint64_t{1} << job.attempts));
    }
    {
      std::lock_guard<std::mutex> lock(transition_mutex_);
      queued_transition_keys_.erase(job.key);
      const size_t source_tier = job.source_tier;
      if (retry && !stop_transition_worker_) {
        if (job.attempts < UINT8_MAX) ++job.attempts;
        queued_transition_keys_.insert(job.key);
        transition_queue_.push_back(std::move(job));
        transition_cv_.notify_one();
      } else if (source_tier < queued_by_tier_.size() && queued_by_tier_[source_tier] > 0) {
        --queued_by_tier_[source_tier];
      }
    }
  }
}

std::vector<PoolAllocateResult> PeerPool::BatchAllocate(
    const std::vector<PoolPlacementRequest>& requests) {
  std::lock_guard<std::mutex> operation_lock(operation_mutex_);
  std::vector<PoolAllocateResult> out(requests.size());
  if (backends_ == nullptr || policy_ == nullptr) return out;

  std::vector<std::vector<uint32_t>> put_orders(requests.size());
  std::vector<size_t> next_candidate(requests.size(), 0);
  std::vector<size_t> active;
  active.reserve(requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    auto existing = placements_.find(requests[i].key);
    if (existing != placements_.end()) {
      auto* existing_backend = backends_->Get(existing->second);
      if (existing_backend != nullptr && existing_backend->Contains(requests[i].key)) {
        out[i].backend_id = existing->second;
        out[i].allocation.outcome = AllocateOutcome::kSuccessAlreadyExists;
        continue;
      }
      placements_.erase(existing);
    }
    auto pending = pending_keys_.find(requests[i].key);
    if (pending != pending_keys_.end()) {
      if (pending->second.expires_at <= std::chrono::steady_clock::now()) {
        auto* pending_backend = backends_->Get(pending->second.backend_id);
        if (pending_backend != nullptr && pending->second.local_slot_id != 0) {
          pending_backend->BatchAbort({pending->second.local_slot_id});
        }
        pending_slots_.erase({pending->second.backend_id, pending->second.local_slot_id});
        pending_keys_.erase(pending);
      } else {
        // A second writer must retry until the reservation commits, aborts, or
        // reaches the same TTL as the backend slot. ALREADY_EXISTS would expose
        // uncommitted bytes; in_flight lets a local put count it as stored.
        out[i].allocation.outcome = AllocateOutcome::kFailed;
        out[i].in_flight = true;
        continue;
      }
    }

    // Rebuild the index before making a new placement decision: a persistent
    // backend may still own the key even though this process has no record.
    const uint32_t owner = FindOwnerLocked(requests[i].key);
    if (owner != BackendRegistry::kMaxBackends) {
      placements_[requests[i].key] = owner;
      out[i].backend_id = owner;
      out[i].allocation.outcome = AllocateOutcome::kSuccessAlreadyExists;
      continue;
    }

    put_orders[i] = policy_->PutOrder(*backends_, requests[i]);
    if (!put_orders[i].empty()) {
      const uint32_t selected = put_orders[i].front();
      pending_keys_[requests[i].key] =
          PendingPlacement{selected, 0, std::chrono::steady_clock::time_point::max()};
      active.push_back(i);
    }
  }

  // Each round batches the requests targeting the same candidate. Only
  // NO_SPACE advances to the next deterministic policy candidate; generic
  // failures preserve their meaning and do not hide backend errors.
  while (!active.empty()) {
    std::map<uint32_t, std::vector<size_t>> by_backend;
    for (size_t index : active) {
      by_backend[put_orders[index][next_candidate[index]]].push_back(index);
    }
    std::vector<size_t> retry;
    // Moves a request onto its next policy candidate, or drops the reservation
    // once the candidates run out.
    auto advance = [&](size_t index) {
      ++next_candidate[index];
      if (next_candidate[index] >= put_orders[index].size()) {
        pending_keys_.erase(requests[index].key);
        return;
      }
      pending_keys_[requests[index].key] =
          PendingPlacement{put_orders[index][next_candidate[index]], 0,
                           std::chrono::steady_clock::time_point::max()};
      retry.push_back(index);
    };

    for (const auto& [backend_id, indices] : by_backend) {
      auto* backend = backends_->Get(backend_id);
      if (backend == nullptr) {
        for (size_t index : indices) advance(index);
        continue;
      }

      std::vector<AllocateRequest> backend_requests;
      backend_requests.reserve(indices.size());
      for (size_t index : indices) {
        backend_requests.push_back(AllocateRequest{requests[index].key, requests[index].size,
                                                   requests[index].preferred_numa_node});
      }

      auto results = backend->BatchAllocate(backend_requests);
      for (size_t i = 0; i < indices.size(); ++i) {
        const size_t index = indices[i];
        if (i >= results.size()) {
          pending_keys_.erase(requests[index].key);
          continue;
        }
        out[index].backend_id = backend_id;
        out[index].allocation = std::move(results[i]);
        switch (out[index].allocation.outcome) {
          case AllocateOutcome::kSuccessAllocated: {
            const uint64_t slot_id = out[index].allocation.slot_id;
            const auto ttl_ms = out[index].allocation.pending_ttl_ms;
            const auto expires_at =
                ttl_ms == 0 ? std::chrono::steady_clock::time_point::max()
                            : std::chrono::steady_clock::now() + std::chrono::milliseconds(ttl_ms);
            pending_keys_[requests[index].key] = PendingPlacement{backend_id, slot_id, expires_at};
            pending_slots_[{backend_id, slot_id}] = requests[index].key;
            break;
          }
          case AllocateOutcome::kSuccessAlreadyExists:
            placements_[requests[index].key] = backend_id;
            pending_keys_.erase(requests[index].key);
            break;
          case AllocateOutcome::kFailed:
            pending_keys_.erase(requests[index].key);
            break;
          case AllocateOutcome::kFailedNoSpace:
            WakeLocalEviction();  // don't wait for the poll; the put never evicts
            advance(index);
            break;
        }
      }
    }
    active = std::move(retry);
  }
  return out;
}

std::vector<PoolCommitResult> PeerPool::BatchCommit(
    const std::vector<PoolCommitRequest>& requests) {
  // Backend commits run without operation_mutex_; this keeps ClearLocal out meanwhile.
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  std::unique_lock<std::mutex> operation_lock(operation_mutex_);
  std::vector<PoolCommitResult> out(requests.size());
  if (backends_ == nullptr) return out;

  const auto release_pending = [&](const PoolSlotRef& slot) {
    auto pending = pending_slots_.find({slot.backend_id, slot.local_slot_id});
    if (pending == pending_slots_.end()) return;
    pending_keys_.erase(pending->second);
    pending_slots_.erase(pending);
  };

  std::map<uint32_t, std::vector<size_t>> by_backend;
  std::set<std::pair<uint32_t, uint64_t>> accepted_slots;
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto slot_key =
        std::make_pair(requests[i].slot.backend_id, requests[i].slot.local_slot_id);
    if (requests[i].slot.backend_id >= BackendRegistry::kMaxBackends ||
        accepted_slots.find(slot_key) != accepted_slots.end()) {
      continue;
    }
    auto slot = pending_slots_.find(slot_key);
    if (slot == pending_slots_.end() || slot->second != requests[i].key) continue;
    auto pending = pending_keys_.find(slot->second);
    if (pending == pending_keys_.end() ||
        pending->second.backend_id != requests[i].slot.backend_id ||
        pending->second.local_slot_id != requests[i].slot.local_slot_id) {
      continue;
    }
    if (pending->second.expires_at <= std::chrono::steady_clock::now()) {
      auto* backend = backends_->Get(requests[i].slot.backend_id);
      if (backend != nullptr) backend->BatchAbort({requests[i].slot.local_slot_id});
      release_pending(requests[i].slot);
      continue;
    }
    // Claimed: a second commit of this slot finds nothing, and the key stays
    // reserved so racing puts see it in flight. No expiry: a hung commit pins it.
    accepted_slots.insert(slot_key);
    pending_slots_.erase(slot);
    pending->second.expires_at = std::chrono::steady_clock::time_point::max();
    by_backend[requests[i].slot.backend_id].push_back(i);
  }

  // Unlocked: an SSD commit is the device write, and every lookup and read
  // needs this lock.
  operation_lock.unlock();
  std::map<uint32_t, std::vector<CommitResult>> results;
  for (const auto& [backend_id, indices] : by_backend) {
    auto* backend = backends_->Get(backend_id);
    if (backend == nullptr) continue;
    std::vector<CommitRequest> backend_requests;
    backend_requests.reserve(indices.size());
    for (size_t index : indices) {
      backend_requests.push_back(
          CommitRequest{requests[index].slot.local_slot_id, requests[index].key});
    }
    results[backend_id] = backend->BatchCommit(backend_requests);
  }
  operation_lock.lock();

  for (const auto& [backend_id, indices] : by_backend) {
    const auto& backend_results = results[backend_id];
    for (size_t i = 0; i < indices.size(); ++i) {
      const PoolCommitRequest& request = requests[indices[i]];
      if (i < backend_results.size()) {
        out[indices[i]].backend_id = backend_id;
        out[indices[i]].commit = backend_results[i];
        if (backend_results[i].success) {
          placements_[request.key] = backend_id;
          TouchLocked(request.key);
        }
      }
      auto pending = pending_keys_.find(request.key);
      if (pending != pending_keys_.end() && pending->second.backend_id == backend_id &&
          pending->second.local_slot_id == request.slot.local_slot_id) {
        pending_keys_.erase(pending);
      }
    }
  }
  MaybeEnqueueWatermarkOffloadLocked();
  WakeLocalEviction();
  return out;
}

std::vector<bool> PeerPool::BatchAbort(const std::vector<PoolSlotRef>& slots) {
  std::lock_guard<std::mutex> operation_lock(operation_mutex_);
  std::vector<bool> out(slots.size(), true);
  if (backends_ == nullptr) return out;

  const auto release_pending = [&](const PoolSlotRef& slot) {
    auto pending = pending_slots_.find({slot.backend_id, slot.local_slot_id});
    if (pending == pending_slots_.end()) return;
    pending_keys_.erase(pending->second);
    pending_slots_.erase(pending);
  };

  std::map<uint32_t, std::vector<size_t>> by_backend;
  for (size_t i = 0; i < slots.size(); ++i) {
    if (slots[i].backend_id < BackendRegistry::kMaxBackends) {
      by_backend[slots[i].backend_id].push_back(i);
    }
  }

  for (const auto& [backend_id, indices] : by_backend) {
    auto* backend = backends_->Get(backend_id);
    if (backend == nullptr) {
      for (size_t index : indices) release_pending(slots[index]);
      continue;
    }

    std::vector<uint64_t> local_slot_ids;
    local_slot_ids.reserve(indices.size());
    for (size_t index : indices) local_slot_ids.push_back(slots[index].local_slot_id);

    auto results = backend->BatchAbort(local_slot_ids);
    for (size_t i = 0; i < indices.size(); ++i) {
      if (i < results.size()) out[indices[i]] = results[i];
      release_pending(slots[indices[i]]);
    }
  }
  return out;
}

std::vector<PoolResolvedEntry> PeerPool::BatchResolve(const std::vector<std::string>& keys,
                                                      bool include_descs, bool allow_file_refs,
                                                      ResolveReader reader,
                                                      const std::vector<ObjectExtent>* extents) {
  if (extents != nullptr && extents->size() != keys.size()) extents = nullptr;
  // A remote reader discards a response holding any kBusy key; a local one
  // reads its hits and comes back for the busy keys.
  const bool busy_discards = reader == ResolveReader::kRemote;
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  std::vector<PoolResolvedEntry> out(keys.size());
  if (keys.empty()) return out;

  // A no-tier, single-backend pool needs placement metadata for allocate,
  // commit, and eviction, but a hit whose placement already names that sole
  // backend cannot repair or transition anything. Resolve it directly and
  // avoid building the preferred/attempted/group vectors plus taking the
  // metadata lock again after backend IO. This is the common no-master,
  // 100%-hit restore shape.
  uint32_t direct_backend_id = BackendRegistry::kMaxBackends;
  MediumBackend* direct_backend = nullptr;
  {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    if (backends_ != nullptr && policy_ != nullptr && tier_graph_ == nullptr &&
        !track_access_order_ && backends_->Size() == 1) {
      const std::vector<uint32_t> order = policy_->ReadOrder(*backends_);
      if (order.size() == 1 && order.front() < BackendRegistry::kMaxBackends) {
        const uint32_t backend_id = order.front();
        bool all_placed_here = true;
        for (const auto& key : keys) {
          const auto placement = placements_.find(key);
          if (placement == placements_.end() || placement->second != backend_id) {
            all_placed_here = false;
            break;
          }
        }
        if (all_placed_here) {
          direct_backend_id = backend_id;
          direct_backend = backends_->Get(backend_id);
        }
      }
    }
  }
  if (direct_backend != nullptr) {
    auto resolved =
        direct_backend->BatchResolve(keys, include_descs, allow_file_refs, reader, extents);
    bool batch_busy = false;
    bool all_found = resolved.size() == keys.size();
    std::vector<bool> failed_but_owned(keys.size(), false);
    for (size_t i = 0; i < resolved.size() && i < out.size(); ++i) {
      const ResolveOutcome outcome = EffectiveResolveOutcome(resolved[i]);
      batch_busy |= outcome == ResolveOutcome::kBusy;
      all_found &= outcome == ResolveOutcome::kFound;
      if (outcome == ResolveOutcome::kFailed) {
        failed_but_owned[i] = direct_backend->Contains(keys[i]);
      }
      out[i].backend_id = direct_backend_id;
      out[i].tier = direct_backend->Tier();
      out[i].resolved = std::move(resolved[i]);
      out[i].resolved.outcome = outcome;
    }
    if (all_found || (batch_busy && busy_discards)) return out;
    // A plain miss is authoritative, while kFailed needs one ownership probe:
    // a backend can still own a key whose descriptor shape it cannot return.
    // Finalize stale placements here instead of running the whole routing
    // pipeline and resolving the same backend a second time.
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    for (size_t i = 0; i < out.size(); ++i) {
      if (out[i].resolved.found || failed_but_owned[i]) continue;
      if (EffectiveResolveOutcome(out[i].resolved) == ResolveOutcome::kBusy) continue;
      const auto placement = placements_.find(keys[i]);
      if (placement == placements_.end() || placement->second != direct_backend_id) continue;
      placements_.erase(placement);
      ForgetAccessLocked(keys[i]);
    }
    return out;
  }

  uint64_t planned_clear_epoch = 0;
  std::vector<std::optional<uint32_t>> preferred(keys.size());
  std::vector<uint32_t> read_order;
  {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    if (backends_ == nullptr || policy_ == nullptr) return out;
    planned_clear_epoch = clear_epoch_;
    for (size_t i = 0; i < keys.size(); ++i) {
      auto it = placements_.find(keys[i]);
      if (it != placements_.end()) preferred[i] = it->second;
    }
    read_order = policy_->ReadOrder(*backends_);
  }
  PromoteForRead(keys, &preferred);

  // A flat mask avoids one heap allocation per key in the common 1023-key
  // ranged-get batch.
  static_assert(BackendRegistry::kMaxBackends <= 16, "attempted mask is 16 bits wide");
  std::vector<uint16_t> attempted(keys.size(), 0);

  const auto resolve = [&](uint32_t backend_id, const std::vector<size_t>& indices) {
    auto* backend = backends_->Get(backend_id);
    if (backend == nullptr || indices.empty()) return;
    bool identity = indices.size() == keys.size();
    for (size_t i = 0; identity && i < indices.size(); ++i) identity = indices[i] == i;
    std::vector<std::string> backend_keys;
    std::vector<ObjectExtent> backend_extents;
    if (!identity) backend_keys.reserve(indices.size());
    if (!identity && extents != nullptr) backend_extents.reserve(indices.size());
    for (size_t index : indices) {
      attempted[index] |= static_cast<uint16_t>(1u << backend_id);
      if (!identity) backend_keys.push_back(keys[index]);
      if (!identity && extents != nullptr) backend_extents.push_back((*extents)[index]);
    }
    const std::vector<ObjectExtent>* sub_extents =
        extents == nullptr ? nullptr : (identity ? extents : &backend_extents);
    auto results = backend->BatchResolve(identity ? keys : backend_keys, include_descs,
                                         allow_file_refs, reader, sub_extents);
    for (size_t i = 0; i < indices.size() && i < results.size(); ++i) {
      const size_t index = indices[i];
      const ResolveOutcome outcome = EffectiveResolveOutcome(results[i]);
      if (outcome == ResolveOutcome::kFound) {
        if (out[index].resolved.found) continue;
        out[index].backend_id = backend_id;
        out[index].tier = backend->Tier();
        out[index].resolved = std::move(results[i]);
        out[index].resolved.outcome = ResolveOutcome::kFound;
        continue;
      }
      if (out[index].resolved.found) continue;
      // Preserve a retryable owner over misses from fallback media.  A
      // permanent backend failure is useful only when no backend reported
      // BUSY; BUSY has priority because a later attempt can still succeed.
      const ResolveOutcome current = EffectiveResolveOutcome(out[index].resolved);
      if (outcome == ResolveOutcome::kBusy || current == ResolveOutcome::kMissing) {
        out[index].backend_id = backend_id;
        out[index].tier = backend->Tier();
        out[index].resolved.outcome = outcome;
      }
    }
  };

  std::map<uint32_t, std::vector<size_t>> preferred_groups;
  for (size_t i = 0; i < preferred.size(); ++i) {
    if (preferred[i].has_value() && *preferred[i] < BackendRegistry::kMaxBackends) {
      preferred_groups[*preferred[i]].push_back(i);
    }
  }
  for (const auto& [backend_id, indices] : preferred_groups) resolve(backend_id, indices);

  for (uint32_t backend_id : read_order) {
    if (backend_id >= BackendRegistry::kMaxBackends) continue;
    std::vector<size_t> unresolved;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (!out[i].resolved.found && (attempted[i] & (1u << backend_id)) == 0) {
        unresolved.push_back(i);
      }
    }
    resolve(backend_id, unresolved);
  }

  // Classify once. A 100%-hit restore must not allocate a key-sized repair
  // vector merely to discover that there is nothing to repair.
  bool batch_busy = false;
  bool all_found = true;
  for (const auto& entry : out) {
    batch_busy |= EffectiveResolveOutcome(entry.resolved) == ResolveOutcome::kBusy;
    all_found &= entry.resolved.found;
  }
  // Do not count hits or repair placements for a response whose page
  // locations will never be consumed.
  if (batch_busy && busy_discards) return out;

  // Contains is backend work too; keep it off the pool metadata lock. A
  // placement snapshot check below prevents this repair from overwriting a
  // commit or migration that publishes while these probes are running.
  std::vector<std::optional<uint32_t>> repaired;
  if (!all_found) {
    repaired.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      if (out[i].resolved.found) continue;
      // kMissing is an authoritative negative result from every attempted
      // backend. Only kFailed is ambiguous: the selected backend may still own
      // the key but be unable to materialize its descriptor.
      if (EffectiveResolveOutcome(out[i].resolved) != ResolveOutcome::kFailed) continue;
      auto* backend = backends_->Get(out[i].backend_id);
      if (backend != nullptr && backend->Contains(keys[i])) {
        repaired[i] = out[i].backend_id;
      }
    }
  }

  std::lock_guard<std::mutex> operation_lock(operation_mutex_);
  // ClearLocal takes lifecycle_mutex_ exclusively, so this is defensive
  // against future lifecycle paths that may invalidate an in-flight plan.
  if (clear_epoch_ != planned_clear_epoch) return out;

  const auto placement_unchanged = [&](size_t index) {
    auto current = placements_.find(keys[index]);
    if (!preferred[index].has_value()) return current == placements_.end();
    return current != placements_.end() && current->second == *preferred[index];
  };

  for (size_t i = 0; i < keys.size(); ++i) {
    if (out[i].resolved.found) {
      // A migration or commit may have published a newer owner while backend
      // IO ran. Repair only the placement snapshot this resolve actually used;
      // never move the logical owner backwards to a source that is draining.
      const bool metadata_current = placement_unchanged(i);
      if (metadata_current) placements_[keys[i]] = out[i].backend_id;
      // A concurrent discard may have removed the key entirely. Do not
      // resurrect only its LRU entry from a resolve that started earlier.
      const bool still_placed = metadata_current || placements_.find(keys[i]) != placements_.end();
      if (still_placed) TouchLocked(keys[i]);
      if (tier_graph_ != nullptr) {
        ++tier_read_hits_[tier_graph_->NameForBackend(out[i].backend_id)];
      }
    } else {
      // A concurrent commit or transition changed the placement after the
      // snapshot; its publication is newer than this miss and must win.
      if (!placement_unchanged(i)) continue;
      // Held, just not staged yet: the local reader comes back for it.
      if (EffectiveResolveOutcome(out[i].resolved) == ResolveOutcome::kBusy) continue;
      if (repaired[i].has_value()) {
        placements_[keys[i]] = *repaired[i];
        continue;
      }
      // A key the backends no longer hold must leave the access index too, or
      // it accumulates there for the life of the process and skews LRU order.
      placements_.erase(keys[i]);
      ForgetAccessLocked(keys[i]);
    }
  }
  return out;
}

std::vector<bool> PeerPool::BatchContains(const std::vector<std::string>& keys) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  std::vector<bool> out(keys.size(), false);
  if (keys.empty()) return out;

  std::vector<uint32_t> read_order;
  {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    if (backends_ == nullptr || policy_ == nullptr) return out;
    read_order = policy_->ReadOrder(*backends_);
  }

  // Carry only the still-unanswered indices from one backend to the next, so a
  // second medium is asked about exactly what the first one missed. Same shape
  // as the resolve paths, and for the same reason: one call per backend, not
  // one per key. The backend calls stay outside operation_mutex_.
  std::vector<size_t> pending(keys.size());
  std::iota(pending.begin(), pending.end(), 0);
  std::vector<std::string> batch;
  std::vector<size_t> still_missing;
  for (uint32_t backend_id : read_order) {
    if (pending.empty()) break;
    auto* backend = backends_->Get(backend_id);
    if (backend == nullptr) continue;
    batch.clear();
    batch.reserve(pending.size());
    for (size_t i : pending) batch.push_back(keys[i]);

    const auto hits = backend->BatchContains(batch);
    still_missing.clear();
    for (size_t j = 0; j < pending.size(); ++j) {
      if (j < hits.size() && hits[j]) {
        out[pending[j]] = true;
      } else {
        still_missing.push_back(pending[j]);
      }
    }
    pending.swap(still_missing);
  }
  return out;
}

uint32_t PeerPool::EvictSourceLocked(const PoolEvictRequest& request) const {
  constexpr uint32_t kNone = BackendRegistry::kMaxBackends;
  const auto holds = [&](uint32_t backend_id) {
    if (backend_id >= kNone) return false;
    auto* backend = backends_->Get(backend_id);
    return backend != nullptr &&
           (request.tier == TierType::UNKNOWN || backend->Tier() == request.tier) &&
           backend->Contains(request.key);
  };
  if (request.backend_id < kNone) return holds(request.backend_id) ? request.backend_id : kNone;
  // Prefer the placement: the only copy a demotion can start from.
  auto placement = placements_.find(request.key);
  if (placement != placements_.end() && holds(placement->second)) return placement->second;
  if (request.tier == TierType::UNKNOWN) return FindOwnerLocked(request.key);
  for (auto* backend : backends_->All()) {
    const uint32_t backend_id = backends_->BackendId(backend);
    if (holds(backend_id)) return backend_id;
  }
  return kNone;
}

void PeerPool::RepairAfterDropLocked(const std::string& key) {
  auto draining = draining_sources_.find(key);
  if (draining != draining_sources_.end()) {
    for (auto it = draining->second.begin(); it != draining->second.end();) {
      auto* source = backends_->Get(*it);
      it = source == nullptr || !source->Contains(key) ? draining->second.erase(it) : std::next(it);
    }
    if (draining->second.empty()) draining_sources_.erase(draining);
  }
  auto placement = placements_.find(key);
  if (placement != placements_.end()) {
    auto* backend = backends_->Get(placement->second);
    if (backend != nullptr && backend->Contains(key)) return;
  }
  const uint32_t owner = FindOwnerLocked(key);
  if (owner != BackendRegistry::kMaxBackends) {
    placements_[key] = owner;
    return;
  }
  placements_.erase(key);
  ForgetAccessLocked(key);
  draining_sources_.erase(key);
}

std::vector<PoolEvictResult> PeerPool::Evict(const std::vector<std::string>& keys,
                                             PoolEvictMode mode) {
  std::vector<PoolEvictRequest> requests(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) requests[i].key = keys[i];
  return Evict(requests, mode);
}

std::vector<PoolEvictResult> PeerPool::Evict(const std::vector<PoolEvictRequest>& requests,
                                             PoolEvictMode mode) {
  // Drops run without operation_mutex_; this keeps ClearLocal out meanwhile.
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  std::vector<PoolEvictResult> out;
  out.reserve(requests.size());
  for (const auto& request : requests) out.push_back(PoolEvictResult{request.key});
  if (backends_ == nullptr || requests.empty()) return out;

  // (request index, backend), decided under the pool lock.
  std::vector<std::pair<size_t, uint32_t>> drops;
  std::vector<std::pair<size_t, uint32_t>> demotions;

  std::unique_lock<std::mutex> operation_lock(operation_mutex_);
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& request = requests[i];
    // A copy already owns this key. Freeing nothing makes the caller retry,
    // which is the same contract as a read lease delaying a source delete, and
    // it keeps a discard from racing a target commit that would resurrect it.
    if (migrating_keys_.count(request.key) != 0) {
      if (mode == PoolEvictMode::kReclaim) out[i].outcome = PoolEvictOutcome::kInFlight;
      continue;
    }
    // A discard drops every copy, ignoring scope.
    if (mode == PoolEvictMode::kDiscard) {
      for (auto* backend : backends_->All()) drops.emplace_back(i, backends_->BackendId(backend));
      continue;
    }

    // A source parked by a lease-blocked offload is only ever deleted; an
    // unscoped request means "finish the drain".
    auto draining = draining_sources_.find(request.key);
    const bool unscoped =
        request.tier == TierType::UNKNOWN && request.backend_id >= BackendRegistry::kMaxBackends;
    if (draining != draining_sources_.end() && unscoped) {
      for (uint32_t source_id : draining->second) drops.emplace_back(i, source_id);
      continue;
    }
    const uint32_t source_id = EvictSourceLocked(request);
    if (source_id == BackendRegistry::kMaxBackends) continue;
    if (draining != draining_sources_.end() && draining->second.count(source_id) != 0) {
      drops.emplace_back(i, source_id);
      continue;
    }

    // Demote the placement copy if its tier has an offload target, whatever
    // the trigger.
    bool demote = false;
    if (tier_graph_ != nullptr) {
      auto placement = placements_.find(request.key);
      const auto tier_index = tier_graph_->TierIndexForBackendId(source_id);
      demote = placement != placements_.end() && placement->second == source_id &&
               tier_index.has_value() && !tier_graph_->NodeAt(*tier_index).offload_to.empty();
    }
    if (demote) {
      demotions.emplace_back(i, source_id);
    } else {
      drops.emplace_back(i, source_id);
    }
  }

  // Queued, so EvictKey never waits on a copy.
  for (const auto& [index, source_id] : demotions) {
    const auto tier_index = tier_graph_->TierIndexForBackendId(source_id);
    if (!tier_index.has_value()) {
      drops.emplace_back(index, source_id);
      continue;
    }
    TransitionJob job{TierTransitionKind::kOffload, requests[index].key, source_id, *tier_index};
    job.evict = true;
    if (EnqueueTransition(std::move(job))) {
      out[index].outcome = PoolEvictOutcome::kDemotionQueued;
    } else if (TransitionQueued(requests[index].key)) {
      // Another job will move it; deleting the source could lose the key.
      out[index].outcome = PoolEvictOutcome::kInFlight;
    }
  }
  if (drops.empty()) return out;

  // Unlocked: an SSD delete is device IO.  One Evict call per backend.
  operation_lock.unlock();
  std::map<uint32_t, std::vector<size_t>> by_backend;
  for (const auto& [index, backend_id] : drops) by_backend[backend_id].push_back(index);
  for (const auto& [backend_id, indices] : by_backend) {
    auto* backend = backends_->Get(backend_id);
    if (backend == nullptr) continue;
    std::vector<std::string> keys;
    keys.reserve(indices.size());
    for (size_t index : indices) keys.push_back(requests[index].key);
    auto results = backend->Evict(keys);
    for (size_t j = 0; j < indices.size() && j < results.size(); ++j) {
      out[indices[j]].bytes_freed += results[j].bytes_freed;
      if (results[j].bytes_freed > 0) out[indices[j]].outcome = PoolEvictOutcome::kFreed;
    }
  }
  operation_lock.lock();

  std::unordered_set<std::string> repaired;
  for (const auto& [index, backend_id] : drops) {
    if (repaired.insert(requests[index].key).second) RepairAfterDropLocked(requests[index].key);
  }
  return out;
}

// ---- Local watermark eviction: picks when and what, then calls Evict(). ----

bool PeerPool::EnableLocalEviction(uint32_t backend_id, double high_watermark,
                                   double low_watermark) {
  if (!(high_watermark > 0.0 && high_watermark <= 1.0 && low_watermark > 0.0 &&
        low_watermark < high_watermark)) {
    MORI_UMBP_WARN(
        "[PeerPool] ignoring local eviction watermarks high={} low={} for backend {} "
        "(require 0 < low < high <= 1)",
        high_watermark, low_watermark, backend_id);
    return false;
  }
  auto* backend = backends_ == nullptr ? nullptr : backends_->Get(backend_id);
  if (backend == nullptr) return false;
  // Before the worker can ask for candidates.
  backend->TrackEvictionOrder();

  std::lock_guard<std::mutex> lock(local_evict_mutex_);
  if (stop_local_evict_) return false;
  local_evict_[backend_id] = LocalEvictionWatermarks{high_watermark, low_watermark};
  if (!local_evict_worker_.joinable()) {
    local_evict_worker_ = std::thread(&PeerPool::LocalEvictionLoop, this);
  }
  const auto* entry = backends_->GetEntry(backend_id);
  MORI_UMBP_INFO("[PeerPool] local eviction ON backend={} tier={} high={} low={}",
                 entry == nullptr ? std::string(backend->Name()) : entry->name,
                 TierTypeName(backend->Tier()), high_watermark, low_watermark);
  return true;
}

LocalEvictionMetrics PeerPool::LocalEvictionStats() const {
  LocalEvictionMetrics metrics;
  metrics.rounds = local_evict_rounds_.load(std::memory_order_relaxed);
  metrics.keys = local_evict_keys_.load(std::memory_order_relaxed);
  metrics.no_candidate = local_evict_no_candidate_.load(std::memory_order_relaxed);
  metrics.stalled = local_evict_stalled_.load(std::memory_order_relaxed);
  metrics.master_keys = local_evict_master_keys_.load(std::memory_order_relaxed);
  return metrics;
}

void PeerPool::WakeLocalEviction() {
  // A pending wake covers this one (with eviction off, nothing clears it).
  if (local_evict_wake_.exchange(true, std::memory_order_acq_rel)) return;
  // Under the worker's lock, so the notify can't slip between check and wait.
  std::lock_guard<std::mutex> lock(local_evict_mutex_);
  local_evict_cv_.notify_one();
}

void PeerPool::StopLocalEviction() {
  {
    std::lock_guard<std::mutex> lock(local_evict_mutex_);
    stop_local_evict_ = true;
  }
  local_evict_cv_.notify_all();
  if (local_evict_worker_.joinable()) local_evict_worker_.join();
}

void PeerPool::LocalEvictionLoop() {
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(local_evict_mutex_);
      local_evict_cv_.wait_for(lock, kLocalEvictPollInterval, [&] {
        return stop_local_evict_ || local_evict_wake_.load(std::memory_order_acquire);
      });
      if (stop_local_evict_) return;
    }
    // Cleared first, so a commit during the pass schedules another.
    local_evict_wake_.store(false, std::memory_order_release);
    LocalEvictionPass();
    std::this_thread::sleep_for(kLocalEvictMinGap);
  }
}

size_t PeerPool::LocalEvictionPass() {
  std::lock_guard<std::mutex> pass(local_evict_pass_mutex_);
  std::vector<std::pair<uint32_t, LocalEvictionWatermarks>> targets;
  {
    std::lock_guard<std::mutex> lock(local_evict_mutex_);
    targets.assign(local_evict_.begin(), local_evict_.end());
  }
  if (targets.empty() || backends_ == nullptr) return 0;

  // Lowest tier first, so upper tiers' demotions find room below.
  const auto depth = [this](uint32_t backend_id) {
    auto* backend = backends_->Get(backend_id);
    return backend == nullptr ? 0 : static_cast<int>(backend->Tier());
  };
  std::stable_sort(targets.begin(), targets.end(),
                   [&](const auto& a, const auto& b) { return depth(a.first) > depth(b.first); });

  size_t freed = 0;
  for (const auto& [backend_id, watermarks] : targets) {
    freed += ReclaimBackend(backend_id, watermarks);
  }
  return freed;
}

size_t PeerPool::ReclaimBackend(uint32_t backend_id, const LocalEvictionWatermarks& watermarks) {
  auto* backend = backends_->Get(backend_id);
  if (backend == nullptr) return 0;
  const TierCapacity capacity = backend->Capacity();
  if (capacity.total_bytes == 0) return 0;
  const double total = static_cast<double>(capacity.total_bytes);
  const uint64_t used = UsedBytes(capacity);
  if (static_cast<double>(used) < watermarks.high * total) return 0;
  const auto low_bytes = static_cast<uint64_t>(watermarks.low * total);
  if (used <= low_bytes) return 0;
  local_evict_rounds_.fetch_add(1, std::memory_order_relaxed);

  // Stop once offers cover the starting overage (usage can lag deletes) or
  // usage reaches the low watermark.  Queued demotions count as covered.
  const uint64_t budget = used - low_bytes;
  const TierType tier = backend->Tier();
  uint64_t covered = 0;
  size_t evicted_keys = 0;
  // The medium keeps offering settled keys until they leave; look past them.
  std::unordered_set<std::string> settled;
  const auto stopping = [this] {
    std::lock_guard<std::mutex> lock(local_evict_mutex_);
    return stop_local_evict_;
  };
  const auto drained = [&] { return UsedBytes(backend->Capacity()) <= low_bytes; };

  // Evicts the coldest of offers[*next...] the budget still needs, at most one
  // batch.  False if none of them freed or moved anything.
  const auto evict_some = [&](const std::vector<EvictionOffer>& offers, size_t* next) {
    std::vector<PoolEvictRequest> requests;
    std::vector<uint64_t> offered_bytes;
    uint64_t taken = 0;
    for (; *next < offers.size(); ++*next) {
      const auto& offer = offers[*next];
      if (requests.size() >= kLocalEvictBatch) break;
      if (!requests.empty() && covered + taken >= budget) break;
      if (settled.count(offer.key) != 0) continue;
      taken += offer.bytes;
      offered_bytes.push_back(offer.bytes);
      requests.push_back(PoolEvictRequest{offer.key, tier, backend_id});
    }
    const auto results = Evict(requests, PoolEvictMode::kReclaim);
    bool progressed = false;
    for (size_t i = 0; i < results.size() && i < offered_bytes.size(); ++i) {
      settled.insert(results[i].key);
      switch (results[i].outcome) {
        case PoolEvictOutcome::kFreed:
        case PoolEvictOutcome::kDemotionQueued:
          ++evicted_keys;
          [[fallthrough]];
        case PoolEvictOutcome::kInFlight:
          covered += offered_bytes[i];  // capacity units, as offered
          progressed = true;
          break;
        case PoolEvictOutcome::kNone:
          break;
      }
    }
    return progressed;
  };

  // The master's ranking whenever an answer is at hand, asked again for what
  // the budget still needs as each one runs out; the medium's own order
  // covers the rest meanwhile.  A pass can span a whole band, far more keys
  // than one answer carries.  A medium whose own order is the only one that
  // frees space (the file SSD tier: a whole segment at a time) skips the
  // master: its per-key recency would leave every segment partly live.
  const bool own_order = backend->OwnsEvictionOrder();
  using Ranking = std::optional<std::vector<EvictionOffer>>;
  const auto ranked_for = [&](uint64_t bytes, std::chrono::milliseconds wait) -> Ranking {
    if (own_order) return std::nullopt;
    return RankedCandidates(backend_id, tier, bytes, wait);
  };
  auto ranked = ranked_for(budget, kRankedCandidatesWait);
  size_t ranked_next = 0;
  size_t master_keys = 0;
  size_t window = kLocalEvictBatch;
  while (covered < budget && !drained()) {
    if (stopping()) break;
    if (!ranked.has_value() || ranked_next >= ranked->size()) {
      ranked = ranked_for(budget - covered, std::chrono::milliseconds(0));
      ranked_next = 0;
    }
    if (ranked.has_value() && ranked_next < ranked->size()) {
      const size_t before = evicted_keys;
      evict_some(*ranked, &ranked_next);  // stale entries just yield kNone
      master_keys += evicted_keys - before;
      continue;
    }
    auto offers = backend->EvictionCandidates(window);
    const bool exhausted = offers.size() < window;
    offers.erase(std::remove_if(offers.begin(), offers.end(),
                                [&](const EvictionOffer& o) { return settled.count(o.key) != 0; }),
                 offers.end());
    if (offers.empty()) {
      if (!exhausted) {
        window *= 2;
        continue;
      }
      if (settled.empty()) {
        local_evict_no_candidate_.fetch_add(1, std::memory_order_relaxed);
        MORI_UMBP_DEBUG(
            "[PeerPool] local eviction: backend {} over its watermark but offers no "
            "candidate (all leased or pinned)",
            backend_id);
      }
      break;
    }
    // Use the whole fetch: queued demotions stay offered, so a refetch per batch rescans them.
    size_t next = 0;
    bool progressed = false;
    while (next < offers.size() && covered < budget && !drained() && !stopping()) {
      progressed = evict_some(offers, &next) || progressed;
    }
    if (!progressed) {
      // All got leased or pinned since offered; retry on the next wake.
      local_evict_stalled_.fetch_add(1, std::memory_order_relaxed);
      break;
    }
  }
  local_evict_master_keys_.fetch_add(master_keys, std::memory_order_relaxed);
  local_evict_keys_.fetch_add(evicted_keys, std::memory_order_relaxed);
  return evicted_keys;
}

std::optional<std::vector<EvictionOffer>> PeerPool::RankedCandidates(
    uint32_t backend_id, TierType tier, uint64_t bytes, std::chrono::milliseconds wait) {
  EvictionCandidateSource source;
  {
    std::lock_guard<std::mutex> lock(local_evict_mutex_);
    source = candidate_source_;
  }
  if (!source) return std::nullopt;
  // Only the pass thread touches this (local_evict_pass_mutex_).  The master
  // answers in time proportional to its index, so the pass never waits long
  // (`wait` only on a query it just sent): it evicts by the medium's order
  // meanwhile.  The interval keeps the master from being asked continuously.
  auto& query = ranked_queries_[backend_id];
  if (!query.answer.valid()) {
    if (std::chrono::steady_clock::now() - query.started < kRankedCandidatesMinInterval) {
      return std::nullopt;
    }
    query.started = std::chrono::steady_clock::now();
    const std::string logical =
        tier_graph_ == nullptr ? "" : tier_graph_->NameForBackend(backend_id);
    query.answer = std::async(std::launch::async, [source = std::move(source), tier, logical,
                                                   bytes] { return source(tier, logical, bytes); });
  } else {
    wait = std::chrono::milliseconds(0);
  }
  if (query.answer.wait_for(wait) != std::future_status::ready) return std::nullopt;
  auto offers = query.answer.get();
  if (std::chrono::steady_clock::now() - query.started > kRankedCandidatesMaxAge) {
    return std::nullopt;  // the index has moved on since
  }
  // The source reports value sizes; the budget counts whole pages.
  auto* backend = backends_->Get(backend_id);
  const uint64_t page = backend == nullptr ? 0 : backend->PageSize();
  if (offers.has_value() && page > 0) {
    for (auto& offer : *offers)
      offer.bytes = std::max<uint64_t>(1, (offer.bytes + page - 1) / page) * page;
  }
  return offers;
}

void PeerPool::SetEvictionCandidateSource(EvictionCandidateSource source) {
  std::lock_guard<std::mutex> lock(local_evict_mutex_);
  candidate_source_ = std::move(source);
}

void PeerPool::ClearLocal() {
  std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  std::unique_lock<std::mutex> operation_lock(operation_mutex_);
  {
    std::lock_guard<std::mutex> lock(transition_mutex_);
    transition_queue_.clear();
    queued_transition_keys_.clear();
  }
  // Wiping state underneath an in-flight copy would let its target commit
  // resurrect a key the user asked to discard, so drain the copies first. The
  // queue is already empty, so no new key can enter the set.
  transition_idle_cv_.wait(operation_lock, [&] { return migrating_keys_.empty(); });

  if (backends_ != nullptr) {
    for (auto* backend : backends_->All()) backend->ClearLocal();
  }
  placements_.clear();
  draining_sources_.clear();
  pending_keys_.clear();
  pending_slots_.clear();
  last_access_.clear();
  access_order_.clear();
  promote_hit_counts_.clear();
  ++clear_epoch_;
}

std::vector<KvEvent> PeerPool::DrainPendingEvents() {
  std::lock_guard<std::mutex> operation_lock(operation_mutex_);
  if (backends_ == nullptr) return {};
  struct EventState {
    std::optional<KvEvent> add;
    bool saw_remove = false;
  };
  std::map<std::tuple<std::string, TierType, std::string>, EventState> states;
  const auto all_backends = backends_->All();
  for (auto* backend : all_backends) {
    auto drained = backend->DrainPendingEvents();
    const std::string logical = tier_graph_ == nullptr
                                    ? std::string{}
                                    : tier_graph_->NameForBackend(backends_->BackendId(backend));
    for (auto& event : drained) {
      event.logical_tier = logical;
      auto& state = states[{logical, event.tier, event.key}];
      if (event.kind == KvEvent::Kind::ADD) {
        state.add = std::move(event);
      } else {
        state.saw_remove = true;
      }
    }
  }
  std::vector<KvEvent> events;
  events.reserve(states.size());
  for (auto& [identity, state] : states) {
    const auto& [logical, tier, key] = identity;
    const bool still_owned =
        std::any_of(all_backends.begin(), all_backends.end(), [&](MediumBackend* backend) {
          if (backend == nullptr || backend->Tier() != tier || !backend->Contains(key)) {
            return false;
          }
          return tier_graph_ == nullptr ||
                 tier_graph_->NameForBackend(backends_->BackendId(backend)) == logical;
        });
    if (state.add.has_value() && (still_owned || !state.saw_remove)) {
      events.push_back(std::move(*state.add));
    } else if (state.saw_remove && !still_owned) {
      events.push_back(KvEvent{KvEvent::Kind::REMOVE, key, tier, 0, logical});
    }
  }
  return events;
}

std::vector<KvEvent> PeerPool::SnapshotOwnedKeysForFullSync() {
  std::lock_guard<std::mutex> operation_lock(operation_mutex_);
  if (backends_ == nullptr) return {};
  std::map<std::tuple<std::string, TierType, std::string>, KvEvent> unique;
  for (auto* backend : backends_->All()) {
    auto snapshot = backend->SnapshotOwnedKeysForFullSync();
    const std::string logical = tier_graph_ == nullptr
                                    ? std::string{}
                                    : tier_graph_->NameForBackend(backends_->BackendId(backend));
    for (auto& event : snapshot) {
      event.logical_tier = logical;
      unique[{logical, event.tier, event.key}] = std::move(event);
    }
  }
  std::vector<KvEvent> events;
  events.reserve(unique.size());
  for (auto& item : unique) events.push_back(std::move(item.second));
  return events;
}

std::map<std::string, LogicalTierCapacity> PeerPool::LogicalTierCapacities() const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  return tier_graph_ == nullptr ? std::map<std::string, LogicalTierCapacity>{}
                                : tier_graph_->CapacitySnapshot();
}

std::string PeerPool::LogicalTierForBackend(uint32_t backend_id) const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  return tier_graph_ == nullptr ? std::string{} : tier_graph_->NameForBackend(backend_id);
}

std::optional<uint32_t> PeerPool::PlacementBackend(const std::string& key) const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  auto it = placements_.find(key);
  return it == placements_.end() ? std::nullopt : std::optional<uint32_t>{it->second};
}

size_t PeerPool::PlacementCount() const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  return placements_.size();
}

TierTransitionMetrics PeerPool::TransitionMetrics() const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  return transition_metrics_;
}

std::map<std::string, uint64_t> PeerPool::TierReadHits() const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  return tier_read_hits_;
}

}  // namespace mori::umbp
