// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/peer/backend/medium_backend.h"
#include "umbp/distributed/pool/grant_table.h"
#include "umbp/engine/engine_types.h"

namespace mori::umbp {

struct EngineRegistration {
  std::string pool_id;
  engine::PoolEpoch epoch;
  uint32_t backend_id = 0;
  engine::RankId owner_rank = 0;
  engine::PageGeometry geometry;
};

struct ResidentRecord {
  std::string key;
  std::string parent_key;
  engine::AllocationHandle handle;
};

struct DelegateResult {
  bool success = false;
  std::vector<engine::AllocationHandle> handles;
};

struct PublishResult {
  bool success = false;
  std::vector<bool> accepted;
};

struct EngineHbmHistoryStats {
  size_t delegate_records = 0;
  size_t publish_records = 0;
  size_t retire_records = 0;
  size_t ready_queue = 0;
  uint64_t retire_handle_checks = 0;
  GrantTableStats grants;
};

// Framework-owned HBM exposed as a MediumBackend. It owns slot authority and
// resident metadata, never the tensors or mappings. Legacy Allocate/Resolve are
// intentionally unavailable: an engine-aware client must negotiate span+grant
// support and use Delegate/Publish/Acquire/Retire.
class EngineHbmBackend final : public MediumBackend {
 public:
  explicit EngineHbmBackend(EngineRegistration registration);
  ~EngineHbmBackend() override = default;

  TierType Tier() const override { return TierType::HBM; }
  const char* Name() const override { return "EngineHbmBackend"; }
  void SetBackendId(uint32_t backend_id) override;
  bool Init(MemoryRegistrar* registrar) override;
  void Shutdown() override;

  TierCapacity Capacity() const override;
  uint64_t OwnedKeyCount() const override;
  std::vector<KvEvent> DrainPendingEvents() override;
  std::vector<KvEvent> SnapshotOwnedKeys() const override;
  std::vector<KvEvent> SnapshotOwnedKeysForFullSync() override;
  void ClearLocal() override;
  void ClearFullSyncAcked() override;
  bool IsClearFullSyncPending() const override;
  void SetAutoFlushHook(size_t threshold, std::function<void()> cb) override;
  void SetEventPublishing(bool enabled) override;

  void TrackEvictionOrder() override {}
  std::vector<EvictionOffer> EvictionCandidates(size_t max_count) override;
  bool OwnsEvictionOrder() const override { return true; }
  void Touch(const std::vector<std::string>& keys) override { (void)keys; }

  std::vector<AllocateResult> BatchAllocate(const std::vector<AllocateRequest>& requests) override;
  std::vector<CommitResult> BatchCommit(const std::vector<CommitRequest>& requests) override;
  std::vector<bool> BatchAbort(const std::vector<uint64_t>& slot_ids) override;
  std::vector<ResolvedEntry> BatchResolve(
      const std::vector<std::string>& keys, bool include_descs, bool allow_file_refs = false,
      ResolveReader reader = ResolveReader::kRemote,
      const std::vector<ObjectExtent>* extents = nullptr) override;
  bool Contains(const std::string& key) const override;
  std::vector<EvictResult> Evict(const std::vector<std::string>& keys) override;

  uint64_t PageSize() const override { return registration_.geometry.object_bytes; }
  bool HasIndependentGeometry() const override { return true; }
  std::vector<BufferMemoryDescBytes> AllBufferDescs() const override { return {}; }
  size_t BufferCount() const override { return registration_.geometry.regions.size(); }
  TransferRef BufferRef(uint32_t buffer_index) const override;

  DelegateResult Delegate(engine::RankId rank, size_t count, engine::OperationId operation);
  PublishResult PublishResident(const std::vector<ResidentRecord>& records,
                                engine::OperationId operation);

  std::vector<GrantResult> Acquire(const ReaderIdentity& reader,
                                   const std::vector<engine::AllocationHandle>& handles,
                                   engine::OperationId operation,
                                   GrantTable::Clock::time_point expires_at);
  std::vector<GrantReleaseResult> Release(const ReaderIdentity& reader,
                                          const std::vector<GrantId>& grants,
                                          engine::CompletionProof proof,
                                          engine::OperationId operation);
  std::vector<GrantReleaseResult> Cancel(const ReaderIdentity& reader,
                                         const std::vector<GrantId>& grants,
                                         engine::CompletionProof proof,
                                         engine::OperationId operation);
  std::vector<engine::AllocationHandle> ExpireGrants(GrantTable::Clock::time_point now);

  engine::RetireTicket Retire(const std::vector<engine::AllocationHandle>& handles,
                              engine::OperationId operation);
  std::vector<engine::RetireTicket> PollReadyRetires();
  std::vector<std::string> RetireKeys(engine::OperationId ticket) const;
  bool AcknowledgeReleased(engine::OperationId ticket);
  bool Quarantine(const std::vector<engine::AllocationHandle>& handles);
  std::optional<engine::SlotState> State(uint32_t page_id) const;
  std::optional<engine::AllocationHandle> HandleForKey(const std::string& key) const;
  ObjectView View(const engine::AllocationHandle& handle) const;
  void AdvanceAckWatermark(engine::OperationId watermark);
  EngineHbmHistoryStats HistoryStats() const;

 private:
  struct Slot {
    engine::SlotState state = engine::SlotState::kFree;
    engine::RankId delegate = 0;
    uint64_t generation = 0;
    std::string key;
    std::optional<engine::OperationId> retire;
  };

  bool ValidHandleLocked(const engine::AllocationHandle& handle) const;
  bool CurrentHandleLocked(const engine::AllocationHandle& handle) const;
  bool TooOldLocked(engine::OperationId operation) const;
  std::vector<KvEvent> SnapshotLocked() const;
  void QueueEventLocked(KvEvent event, std::function<void()>* callback);
  void RefreshRetireLocked(engine::OperationId operation);
  bool QuarantineLocked(const std::vector<engine::AllocationHandle>& handles,
                        std::function<void()>* callback);

  EngineRegistration registration_;
  mutable std::mutex mutex_;
  std::vector<Slot> slots_;  // index 0 is the padding slot and is never delegated
  std::unordered_map<std::string, uint32_t> key_to_page_;
  std::unordered_map<engine::OperationId, engine::RetireTicket, engine::OperationIdHash> retires_;
  std::unordered_map<engine::OperationId, std::vector<std::string>, engine::OperationIdHash>
      retire_keys_;
  std::deque<engine::OperationId> ready_retires_;
  std::unordered_map<engine::OperationId, DelegateResult, engine::OperationIdHash> delegate_ops_;
  std::unordered_map<engine::OperationId, PublishResult, engine::OperationIdHash> publish_ops_;
  std::unordered_map<std::string, uint64_t> ack_watermarks_;
  std::vector<KvEvent> pending_events_;
  bool initialized_ = false;
  bool permanently_shutdown_ = false;
  bool full_sync_pending_ = false;
  bool event_publishing_ = true;
  size_t auto_flush_threshold_ = static_cast<size_t>(-1);
  std::function<void()> auto_flush_cb_;
  GrantTable grants_;
  uint64_t retire_handle_checks_ = 0;
};

std::unique_ptr<MediumBackend> MakeEngineHbmBackend(EngineRegistration registration);

}  // namespace mori::umbp
