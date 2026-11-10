// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "umbp/distributed/transfer/transfer_engine.h"

namespace mori::umbp::engine {

enum class DispatchOutcome : uint8_t { kCompleted, kNeverSubmitted, kUnknown };

struct DispatchedTransfer {
  uint64_t id = 0;
  std::vector<TransferPlan> plans;
};

class DispatchTicket {
 public:
  void Complete(DispatchOutcome outcome, std::vector<TransferFailure> failures);
  void Wait(std::vector<TransferFailure>* failures);
  bool WaitFor(std::chrono::milliseconds timeout, std::vector<TransferFailure>* failures);

 private:
  friend class EngineDispatchRegistry;
  void ConfigureTimeout(std::vector<size_t> tags, std::function<void()> handler);
  std::mutex mutex_;
  std::condition_variable cv_;
  bool complete_ = false;
  DispatchOutcome outcome_ = DispatchOutcome::kUnknown;
  std::vector<TransferFailure> failures_;
  std::vector<size_t> tags_;
  std::function<void()> timeout_handler_;
};

// Process-local directory between PoolClient's EngineTransferEngine and the
// standalone EngineSession serving the owning rank. It contains only in-flight
// queue state; slot/grant/residency authority remains in PeerPool.
class EngineDispatchRegistry {
 public:
  static EngineDispatchRegistry& Global();

  bool Register(const std::string& executor_id, const std::array<uint8_t, 16>& session_id);
  void Unregister(const std::string& executor_id, const std::array<uint8_t, 16>& session_id);
  std::shared_ptr<DispatchTicket> Submit(const std::string& executor_id,
                                         std::vector<TransferPlan> plans);
  std::optional<DispatchedTransfer> Fetch(const std::string& executor_id,
                                          const std::array<uint8_t, 16>& session_id);
  bool Complete(const std::string& executor_id, const std::array<uint8_t, 16>& session_id,
                uint64_t transfer_id, DispatchOutcome outcome,
                std::vector<TransferFailure> failures);
  int EventFd(const std::string& executor_id, const std::array<uint8_t, 16>& session_id) const;
  bool FailFetched(const std::string& executor_id, const std::array<uint8_t, 16>& session_id,
                   uint64_t transfer_id, DispatchOutcome outcome);
#ifdef MORI_UMBP_TESTING
  size_t PendingCountForTest(const std::string& executor_id);
  bool RegisteredForTest(const std::string& executor_id);
#endif

 private:
  struct Entry {
    std::array<uint8_t, 16> session_id{};
    uint64_t next_transfer_id = 1;
    std::deque<DispatchedTransfer> pending;
    std::unordered_set<uint64_t> fetched;
    // A fetched transfer can time out while its executor still owns the DMA.
    // Keep a bounded tombstone so its eventual completion is ACKed and ignored
    // instead of being mistaken for a protocol violation.
    std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> tombstones;
    std::unordered_map<uint64_t, std::shared_ptr<DispatchTicket>> in_flight;
    int event_fd = -1;
  };

  void Timeout(const std::string& executor_id, const std::array<uint8_t, 16>& session_id,
               uint64_t transfer_id, const std::shared_ptr<DispatchTicket>& ticket);

  mutable std::mutex mutex_;
  std::unordered_map<std::string, Entry> executors_;
};

}  // namespace mori::umbp::engine
