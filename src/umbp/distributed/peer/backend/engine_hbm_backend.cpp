// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include "umbp/distributed/peer/backend/engine_hbm_backend.h"

#include <unistd.h>

#include <algorithm>
#include <unordered_set>

namespace mori::umbp {
namespace {

std::string SessionKey(const std::array<uint8_t, 16>& session) {
  return std::string(reinterpret_cast<const char*>(session.data()), session.size());
}

}  // namespace

EngineHbmBackend::EngineHbmBackend(EngineRegistration registration)
    : registration_(std::move(registration)), slots_(registration_.geometry.num_pages + 1) {
  active_ = registration_.geometry.Validate();
}

EngineHbmBackend::~EngineHbmBackend() {
  for (auto& region : registration_.vmm_regions) {
    if (region.fd >= 0) close(region.fd);
    region.fd = -1;
  }
}

void EngineHbmBackend::SetBackendId(uint32_t backend_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  registration_.backend_id = backend_id;
}

bool EngineHbmBackend::Activate(EngineRegistration registration) {
  if (!registration.geometry.Validate()) return false;
  if (registration.geometry.layout_id.empty()) {
    registration.geometry.layout_id = engine::ComputeLayoutId(registration.geometry);
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (permanently_shutdown_ || failed_session_) return false;
  if (registration.owner_rank != registration_.owner_rank ||
      (!registration_.pool_id.empty() && registration.pool_id != registration_.pool_id)) {
    return false;
  }
  if (active_) return false;  // exactly one live control session per fixed rank
  registration.backend_id = registration_.backend_id;
  registration_ = std::move(registration);
  slots_.assign(registration_.geometry.num_pages + 1, Slot{});
  active_ = true;
  initialized_ = true;
  return true;
}

bool EngineHbmBackend::Active() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return active_;
}

void EngineHbmBackend::FailSession() {
  std::function<void()> callback;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) return;
    for (uint32_t page_id = 1; page_id < slots_.size(); ++page_id) {
      auto& slot = slots_[page_id];
      if (!slot.key.empty()) {
        QueueEventLocked(KvEvent{KvEvent::Kind::REMOVE, slot.key, TierType::HBM}, &callback);
        key_to_page_.erase(slot.key);
        slot.key.clear();
      }
      if (slot.state != engine::SlotState::kFree) {
        slot.state = engine::SlotState::kQuarantined;
        if (slot.retire.has_value()) {
          auto retire = retires_.find(*slot.retire);
          if (retire != retires_.end() &&
              retire->second.state != engine::RetireState::kQuarantined) {
            retire->second.state = engine::RetireState::kQuarantined;
            ready_retires_.push_back(retire->first);
          }
        }
      }
    }
    active_ = false;
    failed_session_ = true;
  }
  if (callback) callback();
}

bool EngineHbmBackend::Init(MemoryRegistrar* /*registrar*/) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (permanently_shutdown_) return false;
  if (initialized_) return true;
  if (active_ && registration_.geometry.layout_id.empty()) {
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
  active_ = false;
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
        if (retire != retires_.end() && retire->second.state != engine::RetireState::kQuarantined) {
          retire->second.state = engine::RetireState::kQuarantined;
          ready_retires_.push_back(retire->first);
        }
      }
    }
  }
}

TierCapacity EngineHbmBackend::Capacity() const {
  std::lock_guard<std::mutex> lock(mutex_);
  TierCapacity out;
  if (!initialized_ || !active_ || permanently_shutdown_) return out;
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
  if (auto_flush_cb_ && auto_flush_threshold_ > 0 &&
      pending_events_.size() >= auto_flush_threshold_) {
    *callback = auto_flush_cb_;
  }
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
  for (auto& entry : out) entry.outcome = ResolveOutcome::kMissing;
  return out;
}

bool EngineHbmBackend::AcquireMigrationRead(const std::string& key, ResolvedEntry* resolved,
                                            bool allow_file_ref) {
  (void)allow_file_ref;
  if (resolved == nullptr) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  auto found = key_to_page_.find(key);
  if (!active_ || reset_sealed_ || found == key_to_page_.end()) return false;
  const uint32_t page_id = found->second;
  if (page_id >= slots_.size() || slots_[page_id].state != engine::SlotState::kPublished) {
    return false;
  }
  ObjectView view = registration_.geometry.View(page_id);
  if (!ValidateObjectView(view)) return false;
  const engine::AllocationHandle handle{registration_.pool_id, registration_.epoch,
                                        registration_.backend_id, page_id,
                                        slots_[page_id].generation};
  const engine::ResidentIdentity identity(handle);
  ++migration_reads_[identity];
  migration_read_tokens_[key].push_back(identity);
  resolved->outcome = ResolveOutcome::kFound;
  resolved->found = true;
  resolved->size = registration_.geometry.object_bytes;
  resolved->page_size = registration_.geometry.object_bytes;
  resolved->view = std::move(view);
  return true;
}

bool EngineHbmBackend::AcquireRetireMigrationRead(const std::string& key,
                                                  engine::OperationId retire,
                                                  ResolvedEntry* resolved) {
  if (resolved == nullptr) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  auto found = key_to_page_.find(key);
  if (!active_ || reset_sealed_ || found == key_to_page_.end()) return false;
  const uint32_t page_id = found->second;
  if (page_id >= slots_.size()) return false;
  const Slot& slot = slots_[page_id];
  if (slot.state != engine::SlotState::kRetiring || !slot.retire.has_value() ||
      *slot.retire != retire) {
    return false;
  }
  ObjectView view = registration_.geometry.View(page_id);
  if (!ValidateObjectView(view)) return false;
  const engine::AllocationHandle handle{registration_.pool_id, registration_.epoch,
                                        registration_.backend_id, page_id, slot.generation};
  const engine::ResidentIdentity identity(handle);
  ++migration_reads_[identity];
  migration_read_tokens_[key].push_back(identity);
  resolved->outcome = ResolveOutcome::kFound;
  resolved->found = true;
  resolved->size = registration_.geometry.object_bytes;
  resolved->page_size = registration_.geometry.object_bytes;
  resolved->view = std::move(view);
  return true;
}

void EngineHbmBackend::ReleaseMigrationRead(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto tokens = migration_read_tokens_.find(key);
  if (tokens == migration_read_tokens_.end() || tokens->second.empty()) return;
  const engine::ResidentIdentity identity = tokens->second.back();
  tokens->second.pop_back();
  if (tokens->second.empty()) migration_read_tokens_.erase(tokens);
  auto hold = migration_reads_.find(identity);
  if (hold == migration_reads_.end()) return;
  if (--hold->second != 0) return;
  migration_reads_.erase(hold);
  if (identity.page_id >= slots_.size()) return;
  const Slot& slot = slots_[identity.page_id];
  if (slot.generation != identity.generation) return;
  if (slot.state == engine::SlotState::kRetiring && slot.retire.has_value()) {
    RefreshRetireLocked(*slot.retire);
  }
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
  if (!initialized_ || !active_ || permanently_shutdown_ || reset_sealed_ ||
      rank != registration_.owner_rank || full_sync_pending_) {
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
    if (!initialized_ || !active_ || permanently_shutdown_ || reset_sealed_ || full_sync_pending_) {
      return out;
    }
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
  if (initialized_ && active_ && !permanently_shutdown_ && !reset_sealed_ &&
      !TooOldLocked(operation)) {
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
    std::sort(expired.begin(), expired.end(), [](const auto& left, const auto& right) {
      return left.page_id < right.page_id ||
             (left.page_id == right.page_id && left.generation < right.generation);
    });
    expired.erase(std::unique(expired.begin(), expired.end()), expired.end());
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
  if (reset_sealed_) {
    rejected.state = engine::RetireState::kBusy;
    return rejected;
  }
  if (TooOldLocked(operation) || !initialized_ || !active_ || permanently_shutdown_) {
    retires_.emplace(operation, rejected);
    return rejected;
  }
  if (handles.empty()) {
    retires_.emplace(operation, rejected);
    return rejected;
  }
  std::unordered_set<uint32_t> pages;
  std::vector<engine::AllocationHandle> eligible;
  std::vector<engine::AllocationHandle> quarantined;
  eligible.reserve(handles.size());
  quarantined.reserve(handles.size());
  for (const auto& handle : handles) {
    if (!ValidHandleLocked(handle) || !pages.insert(handle.page_id).second) {
      retires_.emplace(operation, rejected);
      return rejected;
    }
    if (CurrentHandleLocked(handle) &&
        slots_[handle.page_id].state == engine::SlotState::kPublished) {
      eligible.push_back(handle);
    } else {
      quarantined.push_back(handle);
    }
  }
  if (eligible.empty()) {
    rejected.handles = std::move(quarantined);
    retires_.emplace(operation, rejected);
    return rejected;
  }
  if (!quarantined.empty()) {
    partial_quarantines_.push_back(
        engine::RetireTicket{operation, quarantined, engine::RetireState::kQuarantined});
  }

  grants_.StopNewGrants(eligible);
  engine::RetireTicket ticket{operation, eligible, engine::RetireState::kWaitingHolds};
  std::vector<std::string> keys;
  keys.reserve(eligible.size());
  for (const auto& handle : eligible) {
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
    if (migration_reads_.count(engine::ResidentIdentity(handle)) != 0) {
      return;
    }
    const bool io_held = std::any_of(io_holds_.begin(), io_holds_.end(), [&](const auto& entry) {
      return std::find(entry.second.begin(), entry.second.end(), handle) != entry.second.end();
    });
    if (io_held) return;
  }
  it->second.state = engine::RetireState::kTransitionPending;
  transition_retires_.push_back(operation);
}

std::vector<engine::RetireTicket> EngineHbmBackend::PollReadyRetires(bool mark_reported) {
  std::vector<engine::RetireTicket> out;
  std::lock_guard<std::mutex> lock(mutex_);
  while (!partial_quarantines_.empty()) {
    out.push_back(std::move(partial_quarantines_.front()));
    partial_quarantines_.pop_front();
  }
  while (!ready_retires_.empty()) {
    const engine::OperationId operation = ready_retires_.front();
    ready_retires_.pop_front();
    auto ticket = retires_.find(operation);
    if (ticket != retires_.end() && (ticket->second.state == engine::RetireState::kReady ||
                                     ticket->second.state == engine::RetireState::kQuarantined)) {
      if (mark_reported) reported_retires_.insert(operation);
      out.push_back(ticket->second);
    }
  }
  return out;
}

std::optional<engine::RetireTicket> EngineHbmBackend::MarkRetireReported(
    engine::OperationId operation) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto ticket = retires_.find(operation);
  if (ticket == retires_.end() || (ticket->second.state != engine::RetireState::kReady &&
                                   ticket->second.state != engine::RetireState::kQuarantined)) {
    return std::nullopt;
  }
  reported_retires_.insert(operation);
  return ticket->second;
}

std::vector<engine::RetireTicket> EngineHbmBackend::PollTransitionRetires() {
  std::vector<engine::RetireTicket> out;
  std::lock_guard<std::mutex> lock(mutex_);
  while (!transition_retires_.empty()) {
    const auto operation = transition_retires_.front();
    transition_retires_.pop_front();
    auto ticket = retires_.find(operation);
    if (ticket != retires_.end() &&
        ticket->second.state == engine::RetireState::kTransitionPending) {
      out.push_back(ticket->second);
    }
  }
  return out;
}

bool EngineHbmBackend::CompleteRetireTransition(engine::OperationId operation, bool success,
                                                bool unknown) {
  std::function<void()> callback;
  bool ok = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto ticket = retires_.find(operation);
    if (ticket == retires_.end() ||
        ticket->second.state != engine::RetireState::kTransitionPending) {
      return false;
    }
    if (unknown) {
      const auto handles = ticket->second.handles;
      ok = QuarantineLocked(handles, &callback);
    } else if (success) {
      ticket->second.state = engine::RetireState::kReady;
      ready_retires_.push_back(operation);
      ok = true;
    } else {
      // F3 may safely refuse a write after proving that no posted DMA remains.
      // This is a policy-approved last-copy drop, not an unknown completion.
      ticket->second.state = engine::RetireState::kReady;
      ready_retires_.push_back(operation);
      ok = true;
    }
  }
  if (callback) callback();
  return ok;
}

bool EngineHbmBackend::IsQualifiedReplica(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto page = key_to_page_.find(key);
  return active_ && page != key_to_page_.end() && page->second < slots_.size() &&
         slots_[page->second].state == engine::SlotState::kPublished;
}

std::vector<std::string> EngineHbmBackend::RetireKeys(engine::OperationId operation) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto keys = retire_keys_.find(operation);
  return keys == retire_keys_.end() ? std::vector<std::string>{} : keys->second;
}

std::optional<engine::RetireState> EngineHbmBackend::RetireStatus(
    engine::OperationId operation) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto ticket = retires_.find(operation);
  return ticket == retires_.end() ? std::nullopt
                                  : std::optional<engine::RetireState>{ticket->second.state};
}

std::optional<engine::OperationId> EngineHbmBackend::RetireOperationForKey(
    const std::string& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto page = key_to_page_.find(key);
  if (page == key_to_page_.end() || page->second >= slots_.size()) return std::nullopt;
  const Slot& slot = slots_[page->second];
  return slot.state == engine::SlotState::kRetiring ? slot.retire : std::nullopt;
}

bool EngineHbmBackend::ReopenRetireTransition(engine::OperationId operation) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto ticket = retires_.find(operation);
  if (ticket == retires_.end() || ticket->second.state != engine::RetireState::kReady ||
      reported_retires_.count(operation) != 0) {
    return false;
  }
  ticket->second.state = engine::RetireState::kTransitionPending;
  ready_retires_.erase(std::remove(ready_retires_.begin(), ready_retires_.end(), operation),
                       ready_retires_.end());
  transition_retires_.erase(
      std::remove(transition_retires_.begin(), transition_retires_.end(), operation),
      transition_retires_.end());
  transition_retires_.push_back(operation);
  return true;
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
      if (migration_reads_.count(engine::ResidentIdentity(handle)) != 0) return false;
      const bool io_held = std::any_of(io_holds_.begin(), io_holds_.end(), [&](const auto& entry) {
        return std::find(entry.second.begin(), entry.second.end(), handle) != entry.second.end();
      });
      if (io_held) return false;
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

std::vector<engine::AllocationHandle> EngineHbmBackend::DrainQuarantineNotices() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<engine::AllocationHandle> out;
  out.reserve(quarantine_notices_.size());
  while (!quarantine_notices_.empty()) {
    out.push_back(std::move(quarantine_notices_.front()));
    quarantine_notices_.pop_front();
  }
  return out;
}

bool EngineHbmBackend::QuarantineLocked(const std::vector<engine::AllocationHandle>& handles,
                                        std::function<void()>* callback) {
  // A Ready ticket that has already crossed the wire is immutable.  If one of
  // its pages becomes unsafe, turn the complete reported ticket into the
  // terminal Quarantined result so the rank cannot ACK a stale subset.
  std::vector<engine::AllocationHandle> expanded = handles;
  std::unordered_set<engine::OperationId, engine::OperationIdHash> reported_terminal;
  for (const auto& handle : handles) {
    if (!ValidHandleLocked(handle)) continue;
    const Slot& slot = slots_[handle.page_id];
    if (!slot.retire.has_value() || reported_retires_.count(*slot.retire) == 0) continue;
    auto retire = retires_.find(*slot.retire);
    if (retire == retires_.end()) continue;
    reported_terminal.insert(retire->first);
    for (const auto& member : retire->second.handles) {
      if (std::find(expanded.begin(), expanded.end(), member) == expanded.end()) {
        expanded.push_back(member);
      }
    }
  }
  std::unordered_set<uint32_t> pages;
  for (const auto& handle : expanded) {
    if (!ValidHandleLocked(handle) || !pages.insert(handle.page_id).second) return false;
    const Slot& slot = slots_[handle.page_id];
    switch (slot.state) {
      case engine::SlotState::kPublished:
      case engine::SlotState::kRetiring:
        if (slot.generation != handle.generation) return false;
        break;
      case engine::SlotState::kDelegated:
        // The rank owns a delegated slot. A late guard/IO completion for the
        // just-released generation is authoritative too; only an older
        // generation is stale.
        if (handle.generation < slot.generation) return false;
        break;
      case engine::SlotState::kQuarantined:
        break;
      case engine::SlotState::kFree:
        return false;
    }
  }
  for (const auto& handle : expanded) {
    Slot& slot = slots_[handle.page_id];
    if (slot.state == engine::SlotState::kQuarantined) {
      slot.generation = std::max(slot.generation, handle.generation);
      continue;
    }
    if (slot.state == engine::SlotState::kDelegated) slot.generation = handle.generation;
    const std::string key = slot.key;
    key_to_page_.erase(key);
    slot.state = engine::SlotState::kQuarantined;
    quarantine_notices_.push_back(
        engine::AllocationHandle{registration_.pool_id, registration_.epoch,
                                 registration_.backend_id, handle.page_id, slot.generation});
    reset_approved_.erase(
        std::remove_if(reset_approved_.begin(), reset_approved_.end(),
                       [&](const auto& approved) { return approved.page_id == handle.page_id; }),
        reset_approved_.end());
    slot.key.clear();
    if (!key.empty()) {
      QueueEventLocked(KvEvent{KvEvent::Kind::REMOVE, key, TierType::HBM, 0, {}}, callback);
    }
    if (slot.retire.has_value()) {
      auto retire = retires_.find(*slot.retire);
      if (retire != retires_.end() && retire->second.state != engine::RetireState::kQuarantined) {
        const engine::OperationId operation = retire->first;
        if (reported_terminal.count(operation) != 0) {
          retire->second.state = engine::RetireState::kQuarantined;
          ready_retires_.push_back(operation);
        } else if (retire->second.handles.size() > 1) {
          partial_quarantines_.push_back(
              engine::RetireTicket{operation, {handle}, engine::RetireState::kQuarantined});
          auto position =
              std::find(retire->second.handles.begin(), retire->second.handles.end(), handle);
          if (position != retire->second.handles.end()) {
            const size_t index = std::distance(retire->second.handles.begin(), position);
            retire->second.handles.erase(position);
            auto keys = retire_keys_.find(operation);
            if (keys != retire_keys_.end() && index < keys->second.size()) {
              keys->second.erase(keys->second.begin() + index);
            }
          }
          slot.retire.reset();
          if (retire->second.handles.empty()) {
            retire->second.state = engine::RetireState::kQuarantined;
            ready_retires_.push_back(operation);
          } else if (retire->second.state == engine::RetireState::kWaitingHolds) {
            RefreshRetireLocked(operation);
          }
        } else {
          retire->second.state = engine::RetireState::kQuarantined;
          ready_retires_.push_back(operation);
        }
      }
    }
  }
  return true;
}

std::optional<engine::SlotState> EngineHbmBackend::State(uint32_t page_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (page_id == 0 || page_id >= slots_.size()) return std::nullopt;
  return slots_[page_id].state;
}

uint64_t EngineHbmBackend::StateCount(engine::SlotState state) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return static_cast<uint64_t>(std::count_if(
      slots_.begin(), slots_.end(), [&](const Slot& slot) { return slot.state == state; }));
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

bool EngineHbmBackend::IsDelegatedHandle(const engine::AllocationHandle& handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!ValidHandleLocked(handle)) return false;
  Slot& slot = slots_[handle.page_id];
  if (slot.state != engine::SlotState::kDelegated || handle.generation < slot.generation) {
    return false;
  }
  slot.generation = handle.generation;
  return true;
}

ObjectView EngineHbmBackend::View(const engine::AllocationHandle& handle) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!ValidHandleLocked(handle)) return {};
  return registration_.geometry.View(handle.page_id);
}

EngineRegistration EngineHbmBackend::RegistrationSnapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return registration_;
}

bool EngineHbmBackend::HasBlockingMigrationReadsLocked() const {
  for (const auto& [identity, count] : migration_reads_) {
    if (count == 0 || identity.page_id == 0 || identity.page_id >= slots_.size()) continue;
    const Slot& slot = slots_[identity.page_id];
    if (slot.generation == identity.generation && slot.state != engine::SlotState::kQuarantined) {
      return true;
    }
  }
  return false;
}

bool EngineHbmBackend::HasBlockingIoHoldsLocked() const {
  for (const auto& [operation, handles] : io_holds_) {
    (void)operation;
    for (const auto& handle : handles) {
      if (!CurrentHandleLocked(handle)) continue;
      if (slots_[handle.page_id].state != engine::SlotState::kQuarantined) return true;
    }
  }
  return false;
}

engine::ResetLease EngineHbmBackend::BeginReset(engine::OperationId operation) {
  std::lock_guard<std::mutex> lock(mutex_);
  engine::ResetLease out;
  out.id = operation;
  out.old_epoch = registration_.epoch;
  out.new_epoch = registration_.epoch;
  ++out.new_epoch.reset_seq;
  if (!initialized_ || !active_ || permanently_shutdown_ || failed_session_ ||
      TooOldLocked(operation)) {
    return out;
  }
  if (reset_lease_.has_value()) {
    if (*reset_lease_ == operation) {
      out.ready = true;
      out.approved = reset_approved_;
    }
    return out;
  }
  reset_sealed_ = true;
  const bool retiring =
      std::any_of(slots_.begin() + std::min<size_t>(1, slots_.size()), slots_.end(),
                  [](const Slot& slot) { return slot.state == engine::SlotState::kRetiring; });
  const bool blocking_grants = std::any_of(
      slots_.begin() + std::min<size_t>(1, slots_.size()), slots_.end(), [&](const Slot& slot) {
        if (slot.state == engine::SlotState::kQuarantined) return false;
        const uint32_t page_id = &slot - slots_.data();
        const engine::AllocationHandle handle{registration_.pool_id, registration_.epoch,
                                              registration_.backend_id, page_id, slot.generation};
        return grants_.Held(handle);
      });
  if (blocking_grants || HasBlockingMigrationReadsLocked() || HasBlockingIoHoldsLocked() ||
      retiring) {
    reset_sealed_ = false;
    return out;
  }
  reset_lease_ = operation;
  reset_approved_.clear();
  for (uint32_t page_id = 1; page_id < slots_.size(); ++page_id) {
    const Slot& slot = slots_[page_id];
    if (slot.state != engine::SlotState::kDelegated &&
        slot.state != engine::SlotState::kPublished) {
      continue;
    }
    reset_approved_.push_back(engine::AllocationHandle{registration_.pool_id, registration_.epoch,
                                                       registration_.backend_id, page_id,
                                                       slot.generation});
  }
  out.ready = true;
  out.approved = reset_approved_;
  return out;
}

bool EngineHbmBackend::HoldIO(const std::vector<engine::AllocationHandle>& handles,
                              engine::OperationId operation) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto existing = io_holds_.find(operation);
  if (existing != io_holds_.end()) return existing->second == handles;
  if (TooOldLocked(operation) || io_completed_.count(operation) != 0 || handles.empty() ||
      reset_sealed_ || !active_ || permanently_shutdown_) {
    return false;
  }
  std::unordered_set<uint32_t> pages;
  for (const auto& handle : handles) {
    if (!CurrentHandleLocked(handle) || !pages.insert(handle.page_id).second) return false;
    const Slot& slot = slots_[handle.page_id];
    if (slot.state != engine::SlotState::kPublished) return false;
  }
  io_holds_.emplace(operation, handles);
  return true;
}

bool EngineHbmBackend::CompleteIO(engine::OperationId hold, engine::CompletionProof proof) {
  std::function<void()> callback;
  bool ok = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto completed = io_completed_.find(hold);
    if (completed != io_completed_.end()) return completed->second == proof;
    if (TooOldLocked(hold)) return false;
    auto active = io_holds_.find(hold);
    if (active == io_holds_.end()) return false;
    const auto handles = active->second;
    io_completed_[hold] = proof;
    if (proof == engine::CompletionProof::kUnknown) {
      ok = QuarantineLocked(handles, &callback);
    } else {
      io_holds_.erase(active);
      for (const auto& handle : handles) {
        if (!CurrentHandleLocked(handle)) continue;
        const Slot& slot = slots_[handle.page_id];
        if (slot.state == engine::SlotState::kRetiring && slot.retire.has_value()) {
          RefreshRetireLocked(*slot.retire);
        }
      }
      ok = true;
    }
  }
  if (callback) callback();
  return ok;
}

bool EngineHbmBackend::FinishReset(engine::OperationId lease, bool commit, engine::PoolEpoch* epoch,
                                   std::vector<std::string>* removed_keys,
                                   std::vector<engine::AllocationHandle>* approved) {
  std::function<void()> callback;
  std::vector<std::string> removed;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!reset_sealed_ || !reset_lease_.has_value() || *reset_lease_ != lease) return false;
    const bool blocking_grants = std::any_of(
        slots_.begin() + std::min<size_t>(1, slots_.size()), slots_.end(), [&](const Slot& slot) {
          if (slot.state == engine::SlotState::kQuarantined) return false;
          const uint32_t page_id = &slot - slots_.data();
          const engine::AllocationHandle handle{registration_.pool_id, registration_.epoch,
                                                registration_.backend_id, page_id, slot.generation};
          return grants_.Held(handle);
        });
    if (commit &&
        (blocking_grants || HasBlockingMigrationReadsLocked() || HasBlockingIoHoldsLocked())) {
      return false;
    }
    if (commit) {
      for (size_t page_id = 1; page_id < slots_.size(); ++page_id) {
        Slot& slot = slots_[page_id];
        if (!slot.key.empty()) {
          removed.push_back(slot.key);
          QueueEventLocked(KvEvent{KvEvent::Kind::REMOVE, slot.key, TierType::HBM, 0, {}},
                           &callback);
          key_to_page_.erase(slot.key);
          slot.key.clear();
        }
        if (slot.state == engine::SlotState::kPublished ||
            slot.state == engine::SlotState::kRetiring) {
          slot.state = engine::SlotState::kDelegated;
          slot.retire.reset();
        }
      }
      ++registration_.epoch.reset_seq;
      if (approved != nullptr) {
        approved->clear();
        for (auto handle : reset_approved_) {
          if (handle.page_id == 0 || handle.page_id >= slots_.size() ||
              slots_[handle.page_id].state != engine::SlotState::kDelegated) {
            continue;
          }
          handle.epoch = registration_.epoch;
          handle.generation = slots_[handle.page_id].generation;
          approved->push_back(std::move(handle));
        }
      }
      delegate_ops_.clear();
      publish_ops_.clear();
      for (auto it = retires_.begin(); it != retires_.end();) {
        const bool referenced = std::any_of(slots_.begin(), slots_.end(), [&](const Slot& slot) {
          return slot.retire.has_value() && *slot.retire == it->first;
        });
        if (!referenced) {
          retire_keys_.erase(it->first);
          it = retires_.erase(it);
        } else {
          ++it;
        }
      }
      ready_retires_.clear();
      reported_retires_.clear();
      io_completed_.clear();
      if (epoch != nullptr) *epoch = registration_.epoch;
    } else if (epoch != nullptr) {
      *epoch = registration_.epoch;
    }
    reset_lease_.reset();
    reset_approved_.clear();
    reset_sealed_ = false;
  }
  if (removed_keys != nullptr) *removed_keys = std::move(removed);
  if (callback) callback();
  return true;
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
  for (auto it = io_completed_.begin(); it != io_completed_.end();) {
    it = acknowledged(it->first) ? io_completed_.erase(it) : std::next(it);
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
      reported_retires_.erase(it->first);
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

std::string EngineHbmBackendName(engine::RankId rank) {
  return "engine-hbm-rank-" + std::to_string(rank);
}

}  // namespace mori::umbp
