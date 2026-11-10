// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include "umbp/distributed/peer/backend/engine_hbm_backend.h"

#include <algorithm>
#include <unordered_set>

namespace mori::umbp {
namespace {

std::string SessionKey(const std::array<uint8_t, 16>& session) {
  return std::string(reinterpret_cast<const char*>(session.data()), session.size());
}

}  // namespace

EngineHbmBackend::EngineHbmBackend(EngineRegistration registration)
    : registration_(std::move(registration)), slots_(registration_.geometry.num_pages + 1) {}

void EngineHbmBackend::SetBackendId(uint32_t backend_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  registration_.backend_id = backend_id;
}

bool EngineHbmBackend::Init(MemoryRegistrar* /*registrar*/) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (permanently_shutdown_) return false;
  if (initialized_) return true;
  if (registration_.pool_id.empty() || !registration_.geometry.Validate()) return false;
  if (registration_.geometry.layout_id.empty()) {
    registration_.geometry.layout_id = engine::ComputeLayoutId(registration_.geometry);
  }
  initialized_ = true;
  return true;
}

void EngineHbmBackend::Shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (permanently_shutdown_) return;
  permanently_shutdown_ = true;
  initialized_ = false;
  pending_events_.clear();
  key_to_page_.clear();
  // The framework still owns the physical allocations. Preserve generation,
  // quarantine and grant state until process exit; a new physical pool needs a
  // new backend object and epoch.
  for (size_t i = 1; i < slots_.size(); ++i) {
    if (slots_[i].state != engine::SlotState::kFree) {
      slots_[i].state = engine::SlotState::kQuarantined;
      slots_[i].key.clear();
      if (slots_[i].retire.has_value()) {
        auto retire = retires_.find(*slots_[i].retire);
        if (retire != retires_.end()) retire->second.state = engine::RetireState::kQuarantined;
      }
    }
  }
}

TierCapacity EngineHbmBackend::Capacity() const {
  std::lock_guard<std::mutex> lock(mutex_);
  TierCapacity out;
  if (!initialized_ || permanently_shutdown_) return out;
  out.total_bytes = registration_.geometry.object_bytes * registration_.geometry.num_pages;
  for (size_t i = 1; i < slots_.size(); ++i) {
    if (slots_[i].state == engine::SlotState::kFree) {
      out.available_bytes += registration_.geometry.object_bytes;
    }
  }
  out.max_allocatable_bytes = out.available_bytes == 0 ? 0 : registration_.geometry.object_bytes;
  return out;
}

uint64_t EngineHbmBackend::OwnedKeyCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return key_to_page_.size();
}

std::vector<KvEvent> EngineHbmBackend::DrainPendingEvents() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<KvEvent> out;
  out.swap(pending_events_);
  return out;
}

std::vector<KvEvent> EngineHbmBackend::SnapshotLocked() const {
  std::vector<KvEvent> out;
  out.reserve(key_to_page_.size());
  for (const auto& [key, page] : key_to_page_) {
    if (slots_[page].state == engine::SlotState::kPublished ||
        slots_[page].state == engine::SlotState::kRetiring) {
      out.push_back(
          KvEvent{KvEvent::Kind::ADD, key, TierType::HBM, registration_.geometry.object_bytes, {}});
    }
  }
  return out;
}

std::vector<KvEvent> EngineHbmBackend::SnapshotOwnedKeys() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return SnapshotLocked();
}

std::vector<KvEvent> EngineHbmBackend::SnapshotOwnedKeysForFullSync() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto out = SnapshotLocked();
  pending_events_.clear();
  return out;
}

void EngineHbmBackend::ClearLocal() {
  std::lock_guard<std::mutex> lock(mutex_);
  full_sync_pending_ = true;
  pending_events_.clear();
  // A legacy clear has no quiesce lease. Do not return published or retiring
  // engine pages to service; quarantine them until physical rebuild.
  for (size_t i = 1; i < slots_.size(); ++i) {
    if (slots_[i].state == engine::SlotState::kPublished ||
        slots_[i].state == engine::SlotState::kRetiring) {
      slots_[i].state = engine::SlotState::kQuarantined;
    }
  }
  key_to_page_.clear();
}

void EngineHbmBackend::ClearFullSyncAcked() {
  std::lock_guard<std::mutex> lock(mutex_);
  full_sync_pending_ = false;
}

bool EngineHbmBackend::IsClearFullSyncPending() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return full_sync_pending_;
}

void EngineHbmBackend::SetAutoFlushHook(size_t threshold, std::function<void()> cb) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto_flush_threshold_ = threshold;
  auto_flush_cb_ = std::move(cb);
}

void EngineHbmBackend::SetEventPublishing(bool enabled) {
  std::lock_guard<std::mutex> lock(mutex_);
  event_publishing_ = enabled;
  if (!enabled) pending_events_.clear();
}

void EngineHbmBackend::QueueEventLocked(KvEvent event, std::function<void()>* callback) {
  if (!event_publishing_) return;
  pending_events_.push_back(std::move(event));
  if (auto_flush_cb_ && pending_events_.size() >= auto_flush_threshold_) *callback = auto_flush_cb_;
}

std::vector<EvictionOffer> EngineHbmBackend::EvictionCandidates(size_t max_count) {
  (void)max_count;
  // The framework tree is the eviction-order authority. M2/S1 feed its
  // versioned whole-leaf offers through the engine control seam; returning an
  // unordered backend key here would bypass detach confirmation.
  return {};
}

std::vector<AllocateResult> EngineHbmBackend::BatchAllocate(
    const std::vector<AllocateRequest>& requests) {
  return std::vector<AllocateResult>(requests.size());
}

std::vector<CommitResult> EngineHbmBackend::BatchCommit(
    const std::vector<CommitRequest>& requests) {
  return std::vector<CommitResult>(requests.size());
}

std::vector<bool> EngineHbmBackend::BatchAbort(const std::vector<uint64_t>& slot_ids) {
  return std::vector<bool>(slot_ids.size(), true);
}

std::vector<ResolvedEntry> EngineHbmBackend::BatchResolve(
    const std::vector<std::string>& keys, bool /*include_descs*/, bool /*allow_file_refs*/,
    ResolveReader /*reader*/, const std::vector<ObjectExtent>* /*extents*/) {
  std::vector<ResolvedEntry> out(keys.size());
  for (auto& entry : out) entry.outcome = ResolveOutcome::kFailed;
  return out;
}

bool EngineHbmBackend::Contains(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return key_to_page_.count(key) != 0;
}

std::vector<EvictResult> EngineHbmBackend::Evict(const std::vector<std::string>& keys) {
  std::vector<EvictResult> out;
  out.reserve(keys.size());
  for (const auto& key : keys) out.push_back(EvictResult{key, 0});
  return out;  // eviction requires framework detach + Retire
}

TransferRef EngineHbmBackend::BufferRef(uint32_t buffer_index) const {
  return buffer_index < registration_.geometry.regions.size()
             ? registration_.geometry.regions[buffer_index].buffer
             : TransferRef{};
}

DelegateResult EngineHbmBackend::Delegate(engine::RankId rank, size_t count,
                                          engine::OperationId operation) {
  DelegateResult out;
  std::lock_guard<std::mutex> lock(mutex_);
  if (TooOldLocked(operation)) return out;
  auto replay = delegate_ops_.find(operation);
  if (replay != delegate_ops_.end()) return replay->second;
  if (!initialized_ || permanently_shutdown_ || rank != registration_.owner_rank ||
      full_sync_pending_) {
    return out;
  }
  for (uint32_t page = 1; page < slots_.size() && out.handles.size() < count; ++page) {
    if (slots_[page].state != engine::SlotState::kFree) continue;
    slots_[page].state = engine::SlotState::kDelegated;
    slots_[page].delegate = rank;
    out.handles.push_back(engine::AllocationHandle{registration_.pool_id, registration_.epoch,
                                                   registration_.backend_id, page,
                                                   slots_[page].generation});
  }
  out.success = out.handles.size() == count;
  if (!out.success) {
    for (const auto& handle : out.handles) slots_[handle.page_id] = Slot{};
    out.handles.clear();
  }
  delegate_ops_.emplace(operation, out);
  return out;
}

bool EngineHbmBackend::ValidHandleLocked(const engine::AllocationHandle& handle) const {
  return handle.pool_id == registration_.pool_id && handle.epoch == registration_.epoch &&
         handle.backend_id == registration_.backend_id && handle.page_id > 0 &&
         handle.page_id < slots_.size();
}

bool EngineHbmBackend::CurrentHandleLocked(const engine::AllocationHandle& handle) const {
  return ValidHandleLocked(handle) && slots_[handle.page_id].generation == handle.generation;
}

PublishResult EngineHbmBackend::PublishResident(const std::vector<ResidentRecord>& records,
                                                engine::OperationId operation) {
  PublishResult out;
  out.accepted.assign(records.size(), false);
  std::function<void()> callback;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (TooOldLocked(operation)) return out;
    auto replay = publish_ops_.find(operation);
    if (replay != publish_ops_.end()) return replay->second;
    if (!initialized_ || permanently_shutdown_ || full_sync_pending_) return out;
    std::unordered_set<std::string> batch_keys;
    std::unordered_set<uint32_t> batch_pages;
    for (size_t i = 0; i < records.size(); ++i) {
      const auto& record = records[i];
      if (!ValidHandleLocked(record.handle) || record.key.empty() ||
          !batch_keys.insert(record.key).second ||
          !batch_pages.insert(record.handle.page_id).second) {
        publish_ops_.emplace(operation, out);
        return out;
      }
      const Slot& slot = slots_[record.handle.page_id];
      if (slot.state != engine::SlotState::kDelegated ||
          slot.delegate != registration_.owner_rank ||
          record.handle.generation <= slot.generation || key_to_page_.count(record.key) != 0) {
        publish_ops_.emplace(operation, out);
        return out;
      }
    }
    for (size_t i = 0; i < records.size(); ++i) {
      const auto& record = records[i];
      Slot& slot = slots_[record.handle.page_id];
      slot.state = engine::SlotState::kPublished;
      slot.generation = record.handle.generation;
      slot.key = record.key;
      key_to_page_[record.key] = record.handle.page_id;
      out.accepted[i] = true;
      QueueEventLocked(KvEvent{KvEvent::Kind::ADD,
                               record.key,
                               TierType::HBM,
                               registration_.geometry.object_bytes,
                               {}},
                       &callback);
    }
    out.success = std::all_of(out.accepted.begin(), out.accepted.end(), [](bool v) { return v; });
    publish_ops_.emplace(operation, out);
  }
  if (callback) callback();
  return out;
}

std::vector<GrantResult> EngineHbmBackend::Acquire(
    const ReaderIdentity& reader, const std::vector<engine::AllocationHandle>& handles,
    engine::OperationId operation, GrantTable::Clock::time_point expires_at) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<bool> eligible(handles.size(), false);
  if (initialized_ && !permanently_shutdown_ && !TooOldLocked(operation)) {
    for (size_t i = 0; i < handles.size(); ++i) {
      if (!CurrentHandleLocked(handles[i])) continue;
      eligible[i] = slots_[handles[i].page_id].state == engine::SlotState::kPublished;
    }
  }
  return grants_.AcquireEligible(reader, handles, eligible, operation, expires_at);
}

std::vector<GrantReleaseResult> EngineHbmBackend::Release(const ReaderIdentity& reader,
                                                          const std::vector<GrantId>& grants,
                                                          engine::CompletionProof proof,
                                                          engine::OperationId operation) {
  std::function<void()> callback;
  std::vector<GrantReleaseResult> out;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    out = grants_.Release(reader, grants, proof, operation);
    for (const auto& result : out) {
      if (!result.source.has_value()) continue;
      if (result.quarantined) {
        (void)QuarantineLocked({*result.source}, &callback);
        continue;
      }
      if (!result.released || !CurrentHandleLocked(*result.source)) continue;
      const Slot& slot = slots_[result.source->page_id];
      if (slot.state == engine::SlotState::kRetiring && slot.retire.has_value()) {
        RefreshRetireLocked(*slot.retire);
      }
    }
    for (const auto& handle : grants_.DrainQuarantineCandidates()) {
      (void)QuarantineLocked({handle}, &callback);
    }
  }
  if (callback) callback();
  return out;
}

std::vector<GrantReleaseResult> EngineHbmBackend::Cancel(const ReaderIdentity& reader,
                                                         const std::vector<GrantId>& grants,
                                                         engine::CompletionProof proof,
                                                         engine::OperationId operation) {
  return Release(reader, grants, proof, operation);
}

std::vector<engine::AllocationHandle> EngineHbmBackend::ExpireGrants(
    GrantTable::Clock::time_point now) {
  std::function<void()> callback;
  std::vector<engine::AllocationHandle> expired;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    expired = grants_.Expire(now);
    if (!expired.empty()) (void)QuarantineLocked(expired, &callback);
  }
  if (callback) callback();
  return expired;
}

engine::RetireTicket EngineHbmBackend::Retire(const std::vector<engine::AllocationHandle>& handles,
                                              engine::OperationId operation) {
  std::lock_guard<std::mutex> lock(mutex_);
  engine::RetireTicket rejected{operation, handles, engine::RetireState::kQuarantined};
  auto existing = retires_.find(operation);
  if (existing != retires_.end()) return existing->second;
  if (TooOldLocked(operation) || !initialized_ || permanently_shutdown_) return rejected;
  if (handles.empty()) {
    retires_.emplace(operation, rejected);
    return rejected;
  }
  std::unordered_set<uint32_t> pages;
  for (const auto& handle : handles) {
    if (!CurrentHandleLocked(handle) ||
        slots_[handle.page_id].state != engine::SlotState::kPublished ||
        !pages.insert(handle.page_id).second) {
      retires_.emplace(operation, rejected);
      return rejected;
    }
  }

  grants_.StopNewGrants(handles);
  engine::RetireTicket ticket{operation, handles, engine::RetireState::kWaitingHolds};
  std::vector<std::string> keys;
  keys.reserve(handles.size());
  for (const auto& handle : handles) {
    Slot& slot = slots_[handle.page_id];
    keys.push_back(slot.key);
    slot.state = engine::SlotState::kRetiring;
    slot.retire = operation;
  }
  retires_.emplace(operation, ticket);
  retire_keys_.emplace(operation, std::move(keys));
  RefreshRetireLocked(operation);
  return retires_.at(operation);
}

void EngineHbmBackend::RefreshRetireLocked(engine::OperationId operation) {
  auto it = retires_.find(operation);
  if (it == retires_.end() || it->second.state != engine::RetireState::kWaitingHolds) return;
  for (const auto& handle : it->second.handles) {
    ++retire_handle_checks_;
    if (grants_.Held(handle)) return;
  }
  it->second.state = engine::RetireState::kReady;
  ready_retires_.push_back(operation);
}

std::vector<engine::RetireTicket> EngineHbmBackend::PollReadyRetires() {
  std::vector<engine::RetireTicket> out;
  std::lock_guard<std::mutex> lock(mutex_);
  while (!ready_retires_.empty()) {
    const engine::OperationId operation = ready_retires_.front();
    ready_retires_.pop_front();
    auto ticket = retires_.find(operation);
    if (ticket != retires_.end() && ticket->second.state == engine::RetireState::kReady) {
      out.push_back(ticket->second);
    }
  }
  return out;
}

std::vector<std::string> EngineHbmBackend::RetireKeys(engine::OperationId operation) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto keys = retire_keys_.find(operation);
  return keys == retire_keys_.end() ? std::vector<std::string>{} : keys->second;
}

bool EngineHbmBackend::AcknowledgeReleased(engine::OperationId operation) {
  std::function<void()> callback;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = retires_.find(operation);
    if (it == retires_.end()) return false;
    if (it->second.state == engine::RetireState::kReleased) return true;
    if (it->second.state != engine::RetireState::kReady) return false;
    for (const auto& handle : it->second.handles) {
      if (grants_.Held(handle)) return false;
      Slot& slot = slots_[handle.page_id];
      if (slot.state != engine::SlotState::kRetiring || !slot.retire.has_value() ||
          *slot.retire != operation) {
        return false;
      }
    }
    for (const auto& handle : it->second.handles) {
      Slot& slot = slots_[handle.page_id];
      const std::string key = slot.key;
      key_to_page_.erase(key);
      slot.state = engine::SlotState::kDelegated;
      slot.key.clear();
      slot.retire.reset();
      QueueEventLocked(KvEvent{KvEvent::Kind::REMOVE, key, TierType::HBM, 0, {}}, &callback);
    }
    grants_.AllowNewGrants(it->second.handles);
    it->second.state = engine::RetireState::kReleased;
  }
  if (callback) callback();
  return true;
}

bool EngineHbmBackend::Quarantine(const std::vector<engine::AllocationHandle>& handles) {
  std::function<void()> callback;
  bool ok = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ok = QuarantineLocked(handles, &callback);
  }
  if (callback) callback();
  return ok;
}

bool EngineHbmBackend::QuarantineLocked(const std::vector<engine::AllocationHandle>& handles,
                                        std::function<void()>* callback) {
  std::unordered_set<uint32_t> pages;
  for (const auto& handle : handles) {
    if (!ValidHandleLocked(handle) || !pages.insert(handle.page_id).second) return false;
    const Slot& slot = slots_[handle.page_id];
    switch (slot.state) {
      case engine::SlotState::kPublished:
      case engine::SlotState::kRetiring:
        if (slot.generation != handle.generation) return false;
        break;
      case engine::SlotState::kDelegated:
        // The node has only the last published generation. An unpublished
        // write is owned by the rank and must report the strictly newer one.
        if (handle.generation <= slot.generation) return false;
        break;
      case engine::SlotState::kQuarantined:
        break;
      case engine::SlotState::kFree:
        return false;
    }
  }
  for (const auto& handle : handles) {
    Slot& slot = slots_[handle.page_id];
    if (slot.state == engine::SlotState::kQuarantined) {
      slot.generation = std::max(slot.generation, handle.generation);
      continue;
    }
    if (slot.state == engine::SlotState::kDelegated) slot.generation = handle.generation;
    const std::string key = slot.key;
    key_to_page_.erase(key);
    slot.state = engine::SlotState::kQuarantined;
    slot.key.clear();
    if (!key.empty()) {
      QueueEventLocked(KvEvent{KvEvent::Kind::REMOVE, key, TierType::HBM, 0, {}}, callback);
    }
    if (slot.retire.has_value()) {
      auto retire = retires_.find(*slot.retire);
      if (retire != retires_.end()) retire->second.state = engine::RetireState::kQuarantined;
    }
  }
  return true;
}

std::optional<engine::SlotState> EngineHbmBackend::State(uint32_t page_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (page_id == 0 || page_id >= slots_.size()) return std::nullopt;
  return slots_[page_id].state;
}

std::optional<engine::AllocationHandle> EngineHbmBackend::HandleForKey(
    const std::string& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = key_to_page_.find(key);
  if (it == key_to_page_.end()) return std::nullopt;
  const Slot& slot = slots_[it->second];
  return engine::AllocationHandle{registration_.pool_id, registration_.epoch,
                                  registration_.backend_id, it->second, slot.generation};
}

ObjectView EngineHbmBackend::View(const engine::AllocationHandle& handle) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!ValidHandleLocked(handle)) return {};
  return registration_.geometry.View(handle.page_id);
}

void EngineHbmBackend::AdvanceAckWatermark(engine::OperationId watermark) {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::string session = SessionKey(watermark.session_id);
  uint64_t& current = ack_watermarks_[session];
  if (watermark.seq <= current) return;
  current = watermark.seq;
  grants_.AdvanceAckWatermark(watermark);
  const auto acknowledged = [&](engine::OperationId operation) {
    return SessionKey(operation.session_id) == session && operation.seq <= current;
  };
  for (auto it = delegate_ops_.begin(); it != delegate_ops_.end();) {
    it = acknowledged(it->first) ? delegate_ops_.erase(it) : std::next(it);
  }
  for (auto it = publish_ops_.begin(); it != publish_ops_.end();) {
    it = acknowledged(it->first) ? publish_ops_.erase(it) : std::next(it);
  }
  for (auto it = retires_.begin(); it != retires_.end();) {
    if (!acknowledged(it->first)) {
      ++it;
      continue;
    }
    const bool referenced = std::any_of(slots_.begin(), slots_.end(), [&](const Slot& slot) {
      return slot.retire.has_value() && *slot.retire == it->first;
    });
    if (it->second.state == engine::RetireState::kReleased || !referenced) {
      retire_keys_.erase(it->first);
      it = retires_.erase(it);
    } else {
      ++it;  // a quarantined physical resource keeps its diagnostic record
    }
  }
  ready_retires_.erase(std::remove_if(ready_retires_.begin(), ready_retires_.end(),
                                      [&](engine::OperationId operation) {
                                        return retires_.find(operation) == retires_.end();
                                      }),
                       ready_retires_.end());
}

EngineHbmHistoryStats EngineHbmBackend::HistoryStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return EngineHbmHistoryStats{delegate_ops_.size(),  publish_ops_.size(),   retires_.size(),
                               ready_retires_.size(), retire_handle_checks_, grants_.Stats()};
}

bool EngineHbmBackend::TooOldLocked(engine::OperationId operation) const {
  auto watermark = ack_watermarks_.find(SessionKey(operation.session_id));
  return watermark != ack_watermarks_.end() && operation.seq <= watermark->second;
}

std::unique_ptr<MediumBackend> MakeEngineHbmBackend(EngineRegistration registration) {
  return std::make_unique<EngineHbmBackend>(std::move(registration));
}

}  // namespace mori::umbp
