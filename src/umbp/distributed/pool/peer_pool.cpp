// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include "umbp/distributed/pool/peer_pool.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace mori::umbp {

PeerPool::PeerPool(BackendRegistry* backends, std::unique_ptr<PoolPolicy> policy,
                   TransferEngine* transfer_engine)
    : backends_(backends),
      policy_(std::move(policy)),
      transfer_engine_(transfer_engine),
      tier_graph_(policy_ == nullptr ? nullptr : policy_->TierGraph()),
      transition_executor_(backends, transfer_engine) {
  if (tier_graph_ != nullptr) {
    transition_worker_ = std::thread(&PeerPool::TransitionWorkerLoop, this);
  }
}

PeerPool::~PeerPool() { StopTransitionWorker(); }

void PeerPool::EnqueueTransition(TransitionJob job) {
  std::lock_guard<std::mutex> lock(transition_mutex_);
  if (stop_transition_worker_ || !queued_transition_keys_.insert(job.key).second) return;
  transition_queue_.push_back(std::move(job));
  transition_cv_.notify_one();
}

void PeerPool::StopTransitionWorker() {
  {
    std::lock_guard<std::mutex> lock(transition_mutex_);
    stop_transition_worker_ = true;
    transition_queue_.clear();
    queued_transition_keys_.clear();
  }
  transition_cv_.notify_all();
  if (transition_worker_.joinable()) transition_worker_.join();
}

TierTransitionResult PeerPool::RunTransitionLocked(const TransitionJob& job) {
  TierTransitionResult result;
  if (tier_graph_ == nullptr) return result;
  auto placement = placements_.find(job.key);
  if (placement == placements_.end() || placement->second != job.source_backend_id) {
    return result;
  }
  const bool promotion = job.kind == TransitionJobKind::kPromotion;
  const auto& source_tier = tier_graph_->NodeAt(job.source_tier);
  const bool remove_source =
      !promotion || source_tier.promotion_mode == PoolTransitionMode::kMove;
  const auto targets =
      promotion ? tier_graph_->PromoteTargetOrder(job.source_tier, job.key)
                : tier_graph_->TransitionTargetOrder(job.source_tier, job.key);
  ++transition_metrics_.attempted;
  result = transition_executor_.Execute(job.key, job.source_backend_id, targets,
                                        remove_source);
  if (!result.success) {
    ++transition_metrics_.failed;
    return result;
  }
  ++transition_metrics_.succeeded;
  if (promotion) {
    transition_metrics_.promoted_bytes += result.bytes_moved;
  } else {
    transition_metrics_.offloaded_bytes += result.bytes_moved;
  }
  placements_[job.key] = result.target_backend_id;
  if (result.source_draining) {
    draining_sources_[job.key].insert(job.source_backend_id);
  } else {
    auto draining = draining_sources_.find(job.key);
    if (draining != draining_sources_.end()) {
      draining->second.erase(job.source_backend_id);
      if (draining->second.empty()) draining_sources_.erase(draining);
    }
  }
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
      std::lock_guard<std::mutex> lock(operation_mutex_);
      if (tier_graph_ != nullptr) {
        const auto& tier = tier_graph_->NodeAt(job.source_tier);
        if (job.kind != TransitionJobKind::kWatermark ||
            tier_graph_->AtOrAbove(job.source_tier, tier.low_watermark)) {
          const auto result = RunTransitionLocked(job);
          retry = job.kind == TransitionJobKind::kWatermark && !result.success &&
                  job.attempts < 3 &&
                  tier_graph_->AtOrAbove(job.source_tier, tier.low_watermark);
        }
      }
    }
    if (retry) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(uint64_t{1} << job.attempts));
    }
    {
      std::lock_guard<std::mutex> lock(transition_mutex_);
      queued_transition_keys_.erase(job.key);
      if (retry && !stop_transition_worker_) {
        ++job.attempts;
        queued_transition_keys_.insert(job.key);
        transition_queue_.push_back(std::move(job));
        transition_cv_.notify_one();
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

    // The index is process-local and may be empty after restart while a
    // persistent backend still owns the key. Rebuild it authoritatively before
    // making a new placement decision.
    bool already_exists = false;
    for (uint32_t backend_id : policy_->ReadOrder(*backends_)) {
      auto* backend = backends_->Get(backend_id);
      if (backend != nullptr && backend->Contains(requests[i].key)) {
        placements_[requests[i].key] = backend_id;
        out[i].backend_id = backend_id;
        out[i].allocation.outcome = AllocateOutcome::kSuccessAlreadyExists;
        already_exists = true;
        break;
      }
    }
    if (already_exists) continue;

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

    for (const auto& [backend_id, indices] : by_backend) {
      auto* backend = backends_->Get(backend_id);
      if (backend == nullptr) {
        for (size_t index : indices) {
          ++next_candidate[index];
          if (next_candidate[index] < put_orders[index].size()) {
            const uint32_t next_backend = put_orders[index][next_candidate[index]];
            pending_keys_[requests[index].key] = PendingPlacement{
                next_backend, 0, std::chrono::steady_clock::time_point::max()};
            retry.push_back(index);
          } else {
            pending_keys_.erase(requests[index].key);
          }
        }
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
                ttl_ms == 0
                    ? std::chrono::steady_clock::time_point::max()
                    : std::chrono::steady_clock::now() + std::chrono::milliseconds(ttl_ms);
            pending_keys_[requests[index].key] =
                PendingPlacement{backend_id, slot_id, expires_at};
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
            ++next_candidate[index];
            if (next_candidate[index] < put_orders[index].size()) {
              const uint32_t next_backend = put_orders[index][next_candidate[index]];
              pending_keys_[requests[index].key] = PendingPlacement{
                  next_backend, 0, std::chrono::steady_clock::time_point::max()};
              retry.push_back(index);
            } else {
              pending_keys_.erase(requests[index].key);
            }
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
        last_access_[requests[index].key] = ++access_clock_;
      }
      release_pending(requests[index].slot);
    }
  }
  if (tier_graph_ == nullptr) return out;
  for (size_t tier_index = 0; tier_index < tier_graph_->TierCount(); ++tier_index) {
    const auto& tier = tier_graph_->NodeAt(tier_index);
    if (tier.trigger != PoolOffloadTrigger::kWatermark || tier.offload_to.empty() ||
        !tier_graph_->AtOrAbove(tier_index, tier.high_watermark)) {
      continue;
    }
    std::vector<std::pair<std::string, uint32_t>> candidates;
    for (const auto& [key, backend_id] : placements_) {
      if (tier_graph_->TierIndexForBackendId(backend_id) == tier_index) {
        candidates.emplace_back(key, backend_id);
      }
    }
    std::sort(candidates.begin(), candidates.end(), [&](const auto& left, const auto& right) {
      if (tier.candidate_policy == TierCandidatePolicy::kKeyOrder) {
        return left.first < right.first;
      }
      const auto left_access = last_access_.find(left.first);
      const auto right_access = last_access_.find(right.first);
      const uint64_t left_value =
          left_access == last_access_.end() ? 0 : left_access->second;
      const uint64_t right_value =
          right_access == last_access_.end() ? 0 : right_access->second;
      return left_value == right_value ? left.first < right.first
                                       : left_value < right_value;
    });
    for (const auto& [key, backend_id] : candidates) {
      EnqueueTransition(
          {TransitionJobKind::kWatermark, key, backend_id, tier_index});
    }
  }
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
  std::lock_guard<std::mutex> operation_lock(operation_mutex_);
  std::vector<PoolResolvedEntry> out(keys.size());
  if (backends_ == nullptr || policy_ == nullptr || keys.empty()) return out;

  std::vector<std::optional<uint32_t>> preferred(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto it = placements_.find(keys[i]);
    if (it != placements_.end()) preferred[i] = it->second;
  }

  std::vector<std::vector<bool>> attempted(
      keys.size(), std::vector<bool>(BackendRegistry::kMaxBackends, false));

  const auto resolve = [&](uint32_t backend_id, const std::vector<size_t>& indices) {
    auto* backend = backends_->Get(backend_id);
    if (backend == nullptr || indices.empty()) return;
    std::vector<std::string> backend_keys;
    backend_keys.reserve(indices.size());
    for (size_t index : indices) {
      attempted[index][backend_id] = true;
      backend_keys.push_back(keys[index]);
    }
    auto results = backend->BatchResolve(backend_keys, include_descs);
    for (size_t i = 0; i < indices.size() && i < results.size(); ++i) {
      const size_t index = indices[i];
      if (!results[i].found || out[index].resolved.found) continue;
      out[index].backend_id = backend_id;
      out[index].tier = backend->Tier();
      out[index].resolved = std::move(results[i]);
    }
  };

  std::map<uint32_t, std::vector<size_t>> preferred_groups;
  for (size_t i = 0; i < preferred.size(); ++i) {
    if (preferred[i].has_value() && *preferred[i] < BackendRegistry::kMaxBackends) {
      preferred_groups[*preferred[i]].push_back(i);
    }
  }
  for (const auto& [backend_id, indices] : preferred_groups) resolve(backend_id, indices);

  const auto read_order = policy_->ReadOrder(*backends_);
  for (uint32_t backend_id : read_order) {
    if (backend_id >= BackendRegistry::kMaxBackends) continue;
    std::vector<size_t> unresolved;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (!out[i].resolved.found && !attempted[i][backend_id]) unresolved.push_back(i);
    }
    resolve(backend_id, unresolved);
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    if (out[i].resolved.found) {
      placements_[keys[i]] = out[i].backend_id;
      last_access_[keys[i]] = ++access_clock_;
      if (tier_graph_ != nullptr) {
        const auto source_tier = tier_graph_->TierIndexForBackendId(out[i].backend_id);
        if (source_tier.has_value() && *source_tier != tier_graph_->EntryTierIndex() &&
            tier_graph_->NodeAt(*source_tier).promote_on_read &&
            !tier_graph_->AtOrAbove(tier_graph_->EntryTierIndex(),
                                    tier_graph_->NodeAt(tier_graph_->EntryTierIndex())
                                        .high_watermark)) {
          EnqueueTransition(
              {TransitionJobKind::kPromotion, keys[i], out[i].backend_id, *source_tier});
        }
      }
    } else {
      bool exists = false;
      for (uint32_t backend_id : read_order) {
        auto* backend = backends_->Get(backend_id);
        if (backend != nullptr && backend->Contains(keys[i])) {
          placements_[keys[i]] = backend_id;
          exists = true;
          break;
        }
      }
      if (!exists) placements_.erase(keys[i]);
    }
  }
  return out;
}

std::vector<EvictResult> PeerPool::Evict(const std::vector<std::string>& keys) {
  std::lock_guard<std::mutex> operation_lock(operation_mutex_);
  std::vector<EvictResult> out;
  out.reserve(keys.size());
  for (const auto& key : keys) out.push_back(EvictResult{key, 0});
  if (backends_ == nullptr || keys.empty()) return out;

  std::vector<bool> handled(keys.size(), false);
  for (size_t i = 0; i < keys.size(); ++i) {
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

    uint32_t source_id = BackendRegistry::kMaxBackends;
    auto placement = placements_.find(keys[i]);
    if (placement != placements_.end()) {
      source_id = placement->second;
    } else if (policy_ != nullptr) {
      for (uint32_t backend_id : policy_->ReadOrder(*backends_)) {
        auto* backend = backends_->Get(backend_id);
        if (backend != nullptr && backend->Contains(keys[i])) {
          source_id = backend_id;
          break;
        }
      }
    }
    if (tier_graph_ == nullptr) continue;
    const auto tier_index = tier_graph_->TierIndexForBackendId(source_id);
    if (!tier_index.has_value()) continue;
    const auto& tier = tier_graph_->NodeAt(*tier_index);
    if (tier.trigger != PoolOffloadTrigger::kOnEvict || tier.offload_to.empty()) {
      continue;
    }
    auto migrated = RunTransitionLocked(
        {TransitionJobKind::kWatermark, keys[i], source_id, *tier_index});
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
    bool found = false;
    if (policy_ != nullptr) {
      for (uint32_t backend_id : policy_->ReadOrder(*backends_)) {
        auto* backend = backends_->Get(backend_id);
        if (backend != nullptr && backend->Contains(keys[i])) {
          placements_[keys[i]] = backend_id;
          found = true;
          break;
        }
      }
    }
    if (!found) placements_.erase(keys[i]);
    if (!found) last_access_.erase(keys[i]);
  }
  return out;
}

void PeerPool::ClearLocal() {
  std::lock_guard<std::mutex> operation_lock(operation_mutex_);
  if (backends_ != nullptr) {
    for (auto* backend : backends_->All()) backend->ClearLocal();
  }
  placements_.clear();
  draining_sources_.clear();
  pending_keys_.clear();
  pending_slots_.clear();
  last_access_.clear();
  {
    std::lock_guard<std::mutex> lock(transition_mutex_);
    transition_queue_.clear();
    queued_transition_keys_.clear();
  }
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
    const std::string logical =
        tier_graph_ == nullptr ? std::string{}
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
        std::any_of(all_backends.begin(), all_backends.end(),
                    [&](MediumBackend* backend) {
                      if (backend == nullptr || backend->Tier() != tier ||
                          !backend->Contains(key)) {
                        return false;
                      }
                      return tier_graph_ == nullptr ||
                             tier_graph_->NameForBackend(
                                 backends_->BackendId(backend)) == logical;
                    });
    if (state.add.has_value() && (still_owned || !state.saw_remove)) {
      events.push_back(std::move(*state.add));
    } else if (state.saw_remove && !still_owned) {
      events.push_back(
          KvEvent{KvEvent::Kind::REMOVE, key, tier, 0, logical});
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
    const std::string logical =
        tier_graph_ == nullptr ? std::string{}
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

}  // namespace mori::umbp
