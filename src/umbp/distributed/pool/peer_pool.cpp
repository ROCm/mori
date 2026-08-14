// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include "umbp/distributed/pool/peer_pool.h"

#include <map>
#include <set>
#include <utility>

namespace mori::umbp {

PeerPool::PeerPool(BackendRegistry* backends, std::unique_ptr<PoolPolicy> policy)
    : backends_(backends), policy_(std::move(policy)) {}

std::vector<PoolAllocateResult> PeerPool::BatchAllocate(
    const std::vector<PoolPlacementRequest>& requests) {
  std::lock_guard<std::mutex> operation_lock(operation_mutex_);
  std::vector<PoolAllocateResult> out(requests.size());
  if (backends_ == nullptr || policy_ == nullptr) return out;

  std::map<uint32_t, std::vector<size_t>> by_backend;
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

    auto selected = policy_->SelectPutBackend(*backends_, requests[i]);
    if (selected.has_value()) {
      pending_keys_[requests[i].key] =
          PendingPlacement{*selected, 0, std::chrono::steady_clock::time_point::max()};
      by_backend[*selected].push_back(i);
    }
  }

  for (const auto& [backend_id, indices] : by_backend) {
    auto* backend = backends_->Get(backend_id);
    if (backend == nullptr) {
      for (size_t index : indices) pending_keys_.erase(requests[index].key);
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
        case AllocateOutcome::kFailedNoSpace:
          pending_keys_.erase(requests[index].key);
          break;
      }
    }
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
      }
      release_pending(requests[index].slot);
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

  for (auto* backend : backends_->All()) {
    auto results = backend->Evict(keys);
    for (size_t i = 0; i < results.size() && i < out.size(); ++i) {
      out[i].bytes_freed += results[i].bytes_freed;
    }
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    if (out[i].bytes_freed > 0) placements_.erase(keys[i]);
  }
  return out;
}

void PeerPool::ClearLocal() {
  std::lock_guard<std::mutex> operation_lock(operation_mutex_);
  if (backends_ != nullptr) {
    for (auto* backend : backends_->All()) backend->ClearLocal();
  }
  placements_.clear();
  pending_keys_.clear();
  pending_slots_.clear();
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

}  // namespace mori::umbp
