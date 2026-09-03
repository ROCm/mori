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
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace mori::umbp {
namespace {

// One watermark scan queues at most this many keys; the worker tops the queue
// up after each completed offload, so a tier still drains all the way down to
// its low watermark without ever queueing its whole key set at once.
constexpr size_t kWatermarkCandidateBatch = 64;
// Floor between scans. Without it every commit batch arriving above the high
// watermark would re-walk the placements.
constexpr auto kWatermarkScanInterval = std::chrono::milliseconds(5);

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
    transition_worker_ = std::thread(&PeerPool::TransitionWorkerLoop, this);
  }
}

PeerPool::~PeerPool() { StopTransitionWorker(); }

void PeerPool::EnqueueTransition(TransitionJob job) {
  std::lock_guard<std::mutex> lock(transition_mutex_);
  if (stop_transition_worker_ || !queued_transition_keys_.insert(job.key).second) return;
  if (job.source_tier < queued_by_tier_.size()) ++queued_by_tier_[job.source_tier];
  transition_queue_.push_back(std::move(job));
  transition_cv_.notify_one();
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
  return job.kind == TierTransitionKind::kOffload && tier_graph_ != nullptr &&
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
  if (!result.success) {
    ++transition_metrics_.failed;
    return;
  }
  ++transition_metrics_.succeeded;
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
    {
      std::unique_lock<std::mutex> lock(operation_mutex_);
      if (tier_graph_ != nullptr) {
        const auto& tier = tier_graph_->NodeAt(job.source_tier);
        const bool watermark_driven = WatermarkDrivenLocked(job);
        if (!watermark_driven || tier_graph_->AtOrAbove(job.source_tier, tier.low_watermark)) {
          const auto result = RunTransition(lock, job);
          const bool still_pressured =
              watermark_driven && tier_graph_->AtOrAbove(job.source_tier, tier.low_watermark);
          retry = still_pressured && !result.success && job.attempts < 3;
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
    if (retry) {
      std::this_thread::sleep_for(std::chrono::milliseconds(uint64_t{1} << job.attempts));
    }
    {
      std::lock_guard<std::mutex> lock(transition_mutex_);
      queued_transition_keys_.erase(job.key);
      const size_t source_tier = job.source_tier;
      if (retry && !stop_transition_worker_) {
        ++job.attempts;
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
        // uncommitted bytes.
        out[i].allocation.outcome = AllocateOutcome::kFailed;
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
        backend_requests.push_back(AllocateRequest{requests[index].key, requests[index].size});
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
  std::lock_guard<std::mutex> operation_lock(operation_mutex_);
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
    accepted_slots.insert(slot_key);
    by_backend[requests[i].slot.backend_id].push_back(i);
  }

  for (const auto& [backend_id, indices] : by_backend) {
    auto* backend = backends_->Get(backend_id);
    if (backend == nullptr) {
      for (size_t index : indices) release_pending(requests[index].slot);
      continue;
    }

    std::vector<CommitRequest> backend_requests;
    backend_requests.reserve(indices.size());
    for (size_t index : indices) {
      backend_requests.push_back(
          CommitRequest{requests[index].slot.local_slot_id, requests[index].key});
    }

    auto results = backend->BatchCommit(backend_requests);
    for (size_t i = 0; i < indices.size(); ++i) {
      const size_t index = indices[i];
      if (i >= results.size()) {
        release_pending(requests[index].slot);
        continue;
      }
      out[index].backend_id = backend_id;
      out[index].commit = results[i];
      if (results[i].success) {
        placements_[requests[index].key] = backend_id;
        TouchLocked(requests[index].key);
      }
      release_pending(requests[index].slot);
    }
  }
  MaybeEnqueueWatermarkOffloadLocked();
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
                                                      bool include_descs) {
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
    auto resolved = direct_backend->BatchResolve(keys, include_descs);
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
    if (batch_busy || all_found) return out;
    // A plain miss is authoritative, while kFailed needs one ownership probe:
    // a backend can still own a key whose descriptor shape it cannot return.
    // Finalize stale placements here instead of running the whole routing
    // pipeline and resolving the same backend a second time.
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    for (size_t i = 0; i < out.size(); ++i) {
      if (out[i].resolved.found || failed_but_owned[i]) continue;
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
    if (!identity) backend_keys.reserve(indices.size());
    for (size_t index : indices) {
      attempted[index] |= static_cast<uint16_t>(1u << backend_id);
      if (!identity) backend_keys.push_back(keys[index]);
    }
    auto results = backend->BatchResolve(identity ? keys : backend_keys, include_descs);
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
  // A caller retries the whole batch on BUSY and discards this response. Do
  // not count hits, repair placements, or enqueue promotions for a response
  // whose page locations will never be consumed.
  if (batch_busy) return out;

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
        const auto source_tier = tier_graph_->TierIndexForBackendId(out[i].backend_id);
        if (metadata_current && source_tier.has_value() &&
            *source_tier != tier_graph_->EntryTierIndex() && migrating_keys_.count(keys[i]) == 0 &&
            !tier_graph_->AtOrAbove(
                tier_graph_->EntryTierIndex(),
                tier_graph_->NodeAt(tier_graph_->EntryTierIndex()).high_watermark) &&
            PromoteOnReadLocked(keys[i], *source_tier)) {
          EnqueueTransition(
              {TierTransitionKind::kPromotion, keys[i], out[i].backend_id, *source_tier});
        }
      }
    } else {
      // A concurrent commit or transition changed the placement after the
      // snapshot; its publication is newer than this miss and must win.
      if (!placement_unchanged(i)) continue;
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

std::vector<EvictResult> PeerPool::Evict(const std::vector<std::string>& keys, PoolEvictMode mode) {
  std::unique_lock<std::mutex> operation_lock(operation_mutex_);
  std::vector<EvictResult> out;
  out.reserve(keys.size());
  for (const auto& key : keys) out.push_back(EvictResult{key, 0});
  if (backends_ == nullptr || keys.empty()) return out;

  std::vector<bool> handled(keys.size(), false);
  for (size_t i = 0; i < keys.size(); ++i) {
    // A copy already owns this key. Freeing nothing makes the caller retry,
    // which is the same contract as a read lease delaying a source delete, and
    // it keeps a discard from racing a target commit that would resurrect it.
    if (migrating_keys_.count(keys[i]) != 0) {
      handled[i] = true;
      continue;
    }
    // A discard has to leave no copy behind, so it skips demotion and the
    // drain shortcut and goes straight to the delete pass below.
    if (mode == PoolEvictMode::kDiscard) continue;

    auto draining = draining_sources_.find(keys[i]);
    if (draining != draining_sources_.end()) {
      std::vector<uint32_t> drained;
      for (uint32_t source_id : draining->second) {
        auto* source = backends_->Get(source_id);
        if (source != nullptr) {
          auto retried = source->Evict({keys[i]});
          if (!retried.empty()) out[i].bytes_freed += retried.front().bytes_freed;
        }
        if (source == nullptr || !source->Contains(keys[i])) {
          drained.push_back(source_id);
        }
      }
      for (uint32_t source_id : drained) draining->second.erase(source_id);
      if (draining->second.empty()) draining_sources_.erase(draining);
      handled[i] = true;
      continue;
    }

    auto placement = placements_.find(keys[i]);
    const uint32_t source_id =
        placement != placements_.end() ? placement->second : FindOwnerLocked(keys[i]);
    if (tier_graph_ == nullptr) continue;
    const auto tier_index = tier_graph_->TierIndexForBackendId(source_id);
    if (!tier_index.has_value()) continue;
    const auto& tier = tier_graph_->NodeAt(*tier_index);
    if (tier.trigger != PoolOffloadTrigger::kOnEvict || tier.offload_to.empty()) {
      continue;
    }
    auto migrated = RunTransition(operation_lock,
                                  {TierTransitionKind::kOffload, keys[i], source_id, *tier_index});
    handled[i] = migrated.success;
    out[i].bytes_freed = migrated.bytes_freed;
  }

  std::vector<std::string> delete_keys;
  std::vector<size_t> delete_indices;
  for (size_t i = 0; i < keys.size(); ++i) {
    if (!handled[i]) {
      delete_keys.push_back(keys[i]);
      delete_indices.push_back(i);
    }
  }
  for (auto* backend : backends_->All()) {
    auto results = backend->Evict(delete_keys);
    for (size_t i = 0; i < results.size() && i < delete_indices.size(); ++i) {
      out[delete_indices[i]].bytes_freed += results[i].bytes_freed;
    }
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    if (handled[i]) continue;
    const uint32_t owner = FindOwnerLocked(keys[i]);
    if (owner != BackendRegistry::kMaxBackends) {
      placements_[keys[i]] = owner;
    } else {
      placements_.erase(keys[i]);
      ForgetAccessLocked(keys[i]);
      draining_sources_.erase(keys[i]);
    }
  }
  return out;
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
