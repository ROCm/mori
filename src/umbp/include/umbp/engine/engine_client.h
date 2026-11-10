// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "umbp/distributed/peer/backend/engine_hbm_backend.h"
#include "umbp/engine/engine_types.h"
#include "umbp_engine.pb.h"

namespace mori::umbp::engine {

class EngineExecutor;

struct EngineClientConfig {
  std::string node_address;
  std::string executor_id;
  uint32_t heartbeat_ms = 500;
  uint32_t heartbeat_timeout_ms = 5000;
  std::vector<uint8_t> engine_desc;
};

struct PoolRegistration {
  PoolRegistration() = default;
  ~PoolRegistration();
  PoolRegistration(const PoolRegistration&) = delete;
  PoolRegistration& operator=(const PoolRegistration&) = delete;
  PoolRegistration(PoolRegistration&& other) noexcept;
  PoolRegistration& operator=(PoolRegistration&& other) noexcept;

  bool success = false;
  uint32_t backend_id = 0;
  PoolEpoch epoch;
  std::vector<SharedDramRegion> dram_regions;  // owns every fd
  std::string error;
};

enum class ReservationPurpose : uint8_t { kLoad, kExtend };

struct ReservationResult {
  bool success = false;
  uint64_t id = 0;
  size_t shortfall = 0;
};

enum class LocalPageState : uint8_t {
  kCached,
  kReserved,
  kInUse,
  kPendingFree,
  kPublishSent,
  kPublished,
  kRetiring,
  kQuarantined,
};

struct EnginePageAccounting {
  size_t cached = 0;
  size_t reserved = 0;
  size_t in_use = 0;
  size_t pending_free = 0;
  size_t publish_sent = 0;
  size_t published = 0;
  size_t retiring = 0;
  size_t quarantined = 0;
  size_t available = 0;
  size_t total = 0;
};

struct LocalHoldId {
  uint64_t value = 0;
  bool operator==(const LocalHoldId& other) const { return value == other.value; }
};

// Rank-side owner mirror. All node operations share one mutex/socket/sequence;
// the heartbeat thread uses that same channel and never bypasses ordering.
class EngineClient {
 public:
  explicit EngineClient(EngineClientConfig config);
  ~EngineClient();

  PoolRegistration RegisterPool(RankId rank, std::string pool_id, PoolEpoch epoch,
                                const PageGeometry& geometry);
  DelegateResult RequestDelegation(size_t count);
  ReservationResult Reserve(std::string request_id, size_t load_pages, size_t extend_pages);
  std::vector<AllocationHandle> Take(uint64_t reservation_id, ReservationPurpose purpose,
                                     size_t count);
  bool Refund(uint64_t reservation_id);
  bool ReturnPages(const std::vector<AllocationHandle>& handles);
  uint64_t QueuePublish(const std::vector<ResidentRecord>& records, bool ready = false);
  bool MarkPublishReady(uint64_t ticket, CompletionProof proof);
  bool CancelPublish(uint64_t ticket);
  std::optional<PublishReceipt> PublishStatus(uint64_t ticket) const;
  std::vector<PublishReceipt> PollPublishReceipts();
  bool AcknowledgePublishReceipt(uint64_t ticket);
  bool ConsumeEventOverflow();
  PublishResult Publish(const std::vector<ResidentRecord>& records);
  RetireTicket Retire(const std::vector<AllocationHandle>& handles);
  std::vector<RetireTicket> PollRetires();
  bool AcknowledgeReleased(const std::vector<OperationId>& operations);
  ResetLease QuiesceReset();
  bool FinishReset(ResetLease& lease, bool commit);
  std::optional<OperationId> HoldIO(const std::vector<AllocationHandle>& handles);
  bool CompleteIO(OperationId hold, CompletionProof proof);
  std::optional<LocalHoldId> HoldReuseGuard(const std::vector<AllocationHandle>& handles);
  bool CompleteReuseGuard(LocalHoldId hold, CompletionProof proof);
  ReadPlan Lookup(const std::vector<std::string>& keys, std::string name_space);
  PreparedLoad PrepareLoad(const ReadPlan& plan, std::string request_id,
                           const std::vector<AllocationHandle>& targets, uint32_t n_pages);
  CancelLoadResult CancelLoad(const PreparedLoad& prepared);
  bool MarkLoadStarted(uint64_t prepared_id);
  bool FinishLoad(uint64_t prepared_id, LoadOutcome outcome);
  EnginePageAccounting Accounting() const;

  bool Heartbeat();
  void AttachExecutor(EngineExecutor* executor);
  bool healthy() const { return healthy_.load(); }
  uint64_t last_received_seq() const { return last_received_seq_.load(); }
  uint32_t backend_id() const { return backend_id_; }
  std::vector<SharedDramRegion> dram_regions() const;
  void Close();
#ifdef MORI_UMBP_TESTING
  void SetBeforePublishSendHookForTest(std::function<void()> hook);
  void SetBeforeRetireSendHookForTest(std::function<void()> hook);
  void SetAfterRetireReplyHookForTest(std::function<void()> hook);
  void SetAfterReceiverStartedHookForTest(std::function<void()> hook);
  void SetBeforeCallSendHookForTest(std::function<void(const ::umbp::engine::EngineFrame&)> hook);
  bool QuarantineLocalForTest(const AllocationHandle& handle);
  size_t PublishTaskCountForTest() const;
  void SetEventQueueLimitForTest(size_t limit);
  void SetPublishReceiptLimitForTest(size_t limit);
  void InjectPublishReceiptForTest(uint64_t ticket, PublishReceiptState state);
  uint64_t ReservePageChecksForTest() const { return reserve_page_checks_.load(); }
#endif

 private:
  bool Connect(std::string* error);
  bool Call(::umbp::engine::EngineFrame* request, ::umbp::engine::EngineFrame* response,
            std::vector<int>* response_fds, std::string* error,
            const std::vector<int>& request_fds = {});
  bool CallLocked(::umbp::engine::EngineFrame* request, ::umbp::engine::EngineFrame* response,
                  std::vector<int>* response_fds, std::string* error,
                  const std::vector<int>& request_fds = {});
  PublishResult PublishRpc(const std::vector<ResidentRecord>& records, bool* unknown);
  void FailStopLocked();
  void HeartbeatLoop();
  void ReceiverLoop();
  void ProgressLoop();
  bool ExecuteOneWork(::umbp::engine::EngineFrame response);
  bool ExecuteWorkBatch(std::vector<::umbp::engine::EngineFrame> responses);
  OperationId NextOperation();

  struct LocalPage {
    AllocationHandle handle;
    LocalPageState state = LocalPageState::kCached;
    size_t io_holds = 0;
    size_t reuse_guards = 0;
    uint64_t pending_publish = 0;
    bool return_requested = false;
    bool retire_queued = false;
    bool retire_sent = false;
  };

  struct PublishTask {
    uint64_t id = 0;
    std::vector<ResidentRecord> records;
    std::optional<CompletionProof> readiness;
    bool canceled = false;
    bool sent = false;
    bool done = false;
    bool retain_result = false;
    PublishResult result;
    std::condition_variable done_cv;
  };

  struct LocalHold {
    std::vector<AllocationHandle> handles;
    bool reuse_guard = false;
    bool load_target = false;
  };

  bool HandleMatchesLocked(const AllocationHandle& handle) const;
  bool TransitionLocked(LocalPage* page, std::initializer_list<LocalPageState> allowed_from,
                        LocalPageState to);
  void RebuildStateIndexLocked();
  uint64_t QueuePublishImpl(const std::vector<ResidentRecord>& records, bool ready,
                            bool retain_result);
  void AddDelegatedLocked(const std::vector<AllocationHandle>& handles);
  bool SettleUnpublishedLocked(LocalPage* page, CompletionProof proof);
  void FinishTaskLocked(const std::shared_ptr<PublishTask>& task, PublishResult result);
  void FinishAllPublishTasksLocked();
  void RecordPublishReceiptLocked(uint64_t ticket, PublishReceiptState state,
                                  std::vector<bool> accepted = {});
  bool QueueRetireLocked(LocalPage* page);
  void ProcessRetireProgress();
  std::vector<AllocationHandle> ReturnedPublishedLocked() const;
  void RetryReturnedPublished();
  bool ReportQuarantine(const std::vector<AllocationHandle>& handles);
  bool CompleteIOInternal(uint64_t id, CompletionProof proof, bool deferred);
  std::optional<OperationId> HoldLoadTargets(const std::vector<AllocationHandle>& handles);

  EngineClientConfig config_;
  std::array<uint8_t, 16> session_id_{};
  PoolEpoch epoch_;
  std::string pool_id_;
  std::string layout_id_;
  RankId rank_ = 0;
  uint32_t backend_id_ = 0;
  std::atomic<int> socket_{-1};
  uint64_t next_seq_ = 1;
  std::mutex call_mutex_;
  std::atomic<bool> healthy_{false};
  std::atomic<bool> closing_{false};
  std::atomic<bool> failed_{false};
  std::atomic<uint64_t> last_received_seq_{0};
  std::thread heartbeat_thread_;
  std::thread progress_thread_;
  std::thread receiver_thread_;
  std::atomic<bool> receiver_started_{false};
  std::mutex response_mutex_;
  std::condition_variable response_cv_;
  uint64_t expected_response_seq_ = 0;
  std::optional<::umbp::engine::EngineFrame> received_response_;
  std::vector<int> received_response_fds_;
  EngineExecutor* executor_ = nullptr;  // borrowed; provider closes client first
  std::vector<SharedDramRegion> dram_regions_;
  struct LocalReservation {
    std::string request_id;
    std::deque<AllocationHandle> load;
    std::deque<AllocationHandle> extend;
  };
  mutable std::mutex state_mutex_;
  std::condition_variable progress_cv_;
  std::condition_variable state_cv_;
  bool quiescing_ = false;
  uint64_t next_reservation_id_ = 1;
  uint64_t next_publish_id_ = 1;
  uint64_t next_local_hold_id_ = 1;
  std::unordered_map<uint32_t, LocalPage> pages_;
  std::array<size_t, 8> state_counts_{};
  std::deque<uint32_t> cached_pages_;
  std::atomic<uint64_t> reserve_page_checks_{0};
  std::unordered_map<uint64_t, LocalReservation> reservations_;
  std::deque<std::shared_ptr<PublishTask>> publish_queue_;
  std::deque<AllocationHandle> retire_queue_;
  std::deque<std::vector<AllocationHandle>> quarantine_queue_;
  size_t quarantine_reports_in_flight_ = 0;
  std::deque<::umbp::engine::EngineFrame> work_queue_;
  std::unordered_map<uint64_t, std::shared_ptr<PublishTask>> publish_tasks_;
  std::unordered_map<uint64_t, PublishReceipt> publish_receipts_;
  std::deque<PublishReceipt> publish_receipt_events_;
  std::deque<RetireTicket> retire_events_;
  size_t event_queue_limit_ = 4096;
  size_t publish_receipt_limit_ = 65536;
  bool event_overflowed_ = false;
  std::chrono::steady_clock::time_point next_retire_poll_{};
  std::unordered_map<uint64_t, LocalHold> local_holds_;
  std::unordered_set<uint32_t> load_target_pages_;
  struct LocalPreparedLoad {
    std::optional<OperationId> target_hold;
    std::vector<AllocationHandle> targets;
    std::chrono::steady_clock::time_point created_at = std::chrono::steady_clock::now();
    bool started = false;
    bool terminal = false;
  };
  std::unordered_map<uint64_t, LocalPreparedLoad> prepared_loads_;
  std::unordered_map<OperationId, std::vector<AllocationHandle>, OperationIdHash> pending_released_;
  std::unordered_set<OperationId, OperationIdHash> released_operations_;
  std::deque<OperationId> released_operation_order_;
#ifdef MORI_UMBP_TESTING
  std::function<void()> before_publish_send_hook_;
  std::function<void()> before_retire_send_hook_;
  std::function<void()> after_retire_reply_hook_;
  std::function<void()> after_receiver_started_hook_;
  std::function<void(const ::umbp::engine::EngineFrame&)> before_call_send_hook_;
#endif
};

}  // namespace mori::umbp::engine
