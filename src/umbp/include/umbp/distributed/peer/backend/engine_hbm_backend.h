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
#include <unordered_set>
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
  std::string executor_id;
  std::vector<uint8_t> engine_desc;
  std::vector<engine::PeerVmmRegion> vmm_regions;
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
  ~EngineHbmBackend() override;

  TierType Tier() const override { return TierType::HBM; }
  const char* Name() const override { return "EngineHbmBackend"; }
  void SetBackendId(uint32_t backend_id) override;
  bool Activate(EngineRegistration registration);
  bool Active() const;
  void FailSession();
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
  bool AcquireMigrationRead(const std::string& key, ResolvedEntry* resolved,
                            bool allow_file_ref) override;
  bool AcquireRetireMigrationRead(const std::string& key, engine::OperationId retire,
                                  ResolvedEntry* resolved);
  void ReleaseMigrationRead(const std::string& key) override;
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
  std::vector<engine::RetireTicket> PollReadyRetires(bool mark_reported = true);
  std::optional<engine::RetireTicket> MarkRetireReported(engine::OperationId operation);
  std::vector<engine::RetireTicket> PollTransitionRetires();
  bool CompleteRetireTransition(engine::OperationId operation, bool success, bool unknown);
  bool IsQualifiedReplica(const std::string& key) const;
  std::vector<std::string> RetireKeys(engine::OperationId ticket) const;
  std::optional<engine::RetireState> RetireStatus(engine::OperationId ticket) const;
  std::optional<engine::OperationId> RetireOperationForKey(const std::string& key) const;
  bool ReopenRetireTransition(engine::OperationId operation);
  bool AcknowledgeReleased(engine::OperationId ticket);
  bool Quarantine(const std::vector<engine::AllocationHandle>& handles);
  std::vector<engine::AllocationHandle> DrainQuarantineNotices();
  std::optional<engine::SlotState> State(uint32_t page_id) const;
  uint64_t StateCount(engine::SlotState state) const;
  std::optional<engine::AllocationHandle> HandleForKey(const std::string& key) const;
  bool IsDelegatedHandle(const engine::AllocationHandle& handle);
  ObjectView View(const engine::AllocationHandle& handle) const;
  EngineRegistration RegistrationSnapshot() const;
  engine::ResetLease BeginReset(engine::OperationId operation);
  bool FinishReset(engine::OperationId lease, bool commit, engine::PoolEpoch* epoch,
                   std::vector<std::string>* removed_keys,
                   std::vector<engine::AllocationHandle>* approved = nullptr);
  bool HoldIO(const std::vector<engine::AllocationHandle>& handles, engine::OperationId operation);
  bool CompleteIO(engine::OperationId hold, engine::CompletionProof proof);
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
  bool HasBlockingMigrationReadsLocked() const;
  bool HasBlockingIoHoldsLocked() const;
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
  std::unordered_set<engine::OperationId, engine::OperationIdHash> reported_retires_;
  std::deque<engine::RetireTicket> partial_quarantines_;
  std::deque<engine::AllocationHandle> quarantine_notices_;
  std::deque<engine::OperationId> transition_retires_;
  std::unordered_map<engine::OperationId, DelegateResult, engine::OperationIdHash> delegate_ops_;
  std::unordered_map<engine::OperationId, PublishResult, engine::OperationIdHash> publish_ops_;
  std::unordered_map<engine::ResidentIdentity, size_t, engine::ResidentIdentityHash>
      migration_reads_;
  std::unordered_map<std::string, std::vector<engine::ResidentIdentity>> migration_read_tokens_;
  std::unordered_map<engine::OperationId, std::vector<engine::AllocationHandle>,
                     engine::OperationIdHash>
      io_holds_;
  std::unordered_map<engine::OperationId, engine::CompletionProof, engine::OperationIdHash>
      io_completed_;
  std::unordered_map<std::string, uint64_t> ack_watermarks_;
  std::vector<KvEvent> pending_events_;
  bool initialized_ = false;
  bool permanently_shutdown_ = false;
  bool active_ = false;
  bool failed_session_ = false;
  bool reset_sealed_ = false;
  std::optional<engine::OperationId> reset_lease_;
  std::vector<engine::AllocationHandle> reset_approved_;
  bool full_sync_pending_ = false;
  bool event_publishing_ = true;
  size_t auto_flush_threshold_ = static_cast<size_t>(-1);
  std::function<void()> auto_flush_cb_;
  GrantTable grants_;
  uint64_t retire_handle_checks_ = 0;
};

std::unique_ptr<MediumBackend> MakeEngineHbmBackend(EngineRegistration registration);
std::string EngineHbmBackendName(engine::RankId rank);

}  // namespace mori::umbp
