// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include "umbp/engine/engine_dispatch.h"

#include <sys/eventfd.h>
#include <unistd.h>

#include <algorithm>

namespace mori::umbp::engine {
namespace {
constexpr auto kDispatchTombstoneTtl = std::chrono::minutes(5);
}

void DispatchTicket::ConfigureTimeout(std::vector<size_t> tags, std::function<void()> handler) {
  std::lock_guard<std::mutex> lock(mutex_);
  tags_ = std::move(tags);
  timeout_handler_ = std::move(handler);
}

void DispatchTicket::Complete(DispatchOutcome outcome, std::vector<TransferFailure> failures) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (complete_) return;
  outcome_ = outcome;
  failures_ = std::move(failures);
  complete_ = true;
  cv_.notify_all();
}

void DispatchTicket::Wait(std::vector<TransferFailure>* failures) {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [&] { return complete_; });
  if (outcome_ == DispatchOutcome::kUnknown && failures_.empty()) {
    failures_.push_back({tags_, 1, "engine transfer completion is unknown", {}, false, true});
  } else if (outcome_ == DispatchOutcome::kNeverSubmitted && failures_.empty()) {
    failures_.push_back({tags_, 1, "engine transfer was never submitted", {}, true, false});
  } else {
    for (auto& failure : failures_) {
      failure.quiesced = outcome_ != DispatchOutcome::kUnknown;
      failure.submitted = outcome_ != DispatchOutcome::kNeverSubmitted;
    }
  }
  if (failures != nullptr) {
    for (auto& failure : failures_) failures->push_back(std::move(failure));
  }
  failures_.clear();
}

bool DispatchTicket::WaitFor(std::chrono::milliseconds timeout,
                             std::vector<TransferFailure>* failures) {
  std::function<void()> timeout_handler;
  bool timed_out = false;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cv_.wait_for(lock, timeout, [&] { return complete_; })) {
      timed_out = true;
      timeout_handler = timeout_handler_;
    }
  }
  if (timeout_handler) timeout_handler();
  bool complete = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    complete = complete_;
  }
  if (!complete) {
    // A standalone ticket has no registry to cancel it. Preserve the same
    // fail-closed completion contract as a fetched dispatch timeout.
    Complete(DispatchOutcome::kUnknown, {});
  }
  Wait(failures);
  return !timed_out;
}

EngineDispatchRegistry& EngineDispatchRegistry::Global() {
  static EngineDispatchRegistry registry;
  return registry;
}

bool EngineDispatchRegistry::Register(const std::string& executor_id,
                                      const std::array<uint8_t, 16>& session_id) {
  if (executor_id.empty()) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  auto [it, inserted] = executors_.try_emplace(executor_id);
  if (!inserted && it->second.session_id != session_id) return false;
  if (it->second.event_fd < 0) it->second.event_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (it->second.event_fd < 0) {
    if (inserted) executors_.erase(it);
    return false;
  }
  it->second.session_id = session_id;
  return true;
}

void EngineDispatchRegistry::Unregister(const std::string& executor_id,
                                        const std::array<uint8_t, 16>& session_id) {
  std::vector<std::pair<std::shared_ptr<DispatchTicket>, DispatchOutcome>> failed;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = executors_.find(executor_id);
    if (it == executors_.end() || it->second.session_id != session_id) return;
    for (auto& [id, ticket] : it->second.in_flight) {
      failed.emplace_back(std::move(ticket), it->second.fetched.count(id) != 0
                                                 ? DispatchOutcome::kUnknown
                                                 : DispatchOutcome::kNeverSubmitted);
    }
    if (it->second.event_fd >= 0) close(it->second.event_fd);
    executors_.erase(it);
  }
  for (auto& [ticket, outcome] : failed) ticket->Complete(outcome, {});
}

std::shared_ptr<DispatchTicket> EngineDispatchRegistry::Submit(const std::string& executor_id,
                                                               std::vector<TransferPlan> plans) {
  auto ticket = std::make_shared<DispatchTicket>();
  std::vector<size_t> tags;
  for (const auto& plan : plans) {
    for (size_t tag : plan.tags) {
      if (std::find(tags.begin(), tags.end(), tag) == tags.end()) tags.push_back(tag);
    }
  }
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = executors_.find(executor_id);
  if (it == executors_.end()) {
    ticket->ConfigureTimeout(std::move(tags), {});
    ticket->Complete(DispatchOutcome::kNeverSubmitted, {});
    return ticket;
  }
  const uint64_t id = it->second.next_transfer_id++;
  const auto session_id = it->second.session_id;
  std::weak_ptr<DispatchTicket> weak = ticket;
  ticket->ConfigureTimeout(std::move(tags), [this, executor_id, session_id, id, weak] {
    if (auto active = weak.lock()) Timeout(executor_id, session_id, id, active);
  });
  it->second.pending.push_back(DispatchedTransfer{id, std::move(plans)});
  it->second.in_flight.emplace(id, ticket);
  const uint64_t wake = 1;
  (void)write(it->second.event_fd, &wake, sizeof(wake));
  return ticket;
}

void EngineDispatchRegistry::Timeout(const std::string& executor_id,
                                     const std::array<uint8_t, 16>& session_id,
                                     uint64_t transfer_id,
                                     const std::shared_ptr<DispatchTicket>& ticket) {
  DispatchOutcome outcome = DispatchOutcome::kUnknown;
  bool found = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = executors_.find(executor_id);
    if (it == executors_.end() || it->second.session_id != session_id) return;
    auto transfer = it->second.in_flight.find(transfer_id);
    if (transfer == it->second.in_flight.end() || transfer->second != ticket) return;
    if (it->second.fetched.erase(transfer_id) == 0) {
      auto pending =
          std::find_if(it->second.pending.begin(), it->second.pending.end(),
                       [&](const DispatchedTransfer& work) { return work.id == transfer_id; });
      if (pending != it->second.pending.end()) it->second.pending.erase(pending);
      outcome = DispatchOutcome::kNeverSubmitted;
    } else {
      const auto now = std::chrono::steady_clock::now();
      for (auto tombstone = it->second.tombstones.begin();
           tombstone != it->second.tombstones.end();) {
        tombstone = tombstone->second <= now ? it->second.tombstones.erase(tombstone)
                                             : std::next(tombstone);
      }
      it->second.tombstones[transfer_id] = now + kDispatchTombstoneTtl;
    }
    it->second.in_flight.erase(transfer);
    found = true;
  }
  if (found) ticket->Complete(outcome, {});
}

std::optional<DispatchedTransfer> EngineDispatchRegistry::Fetch(
    const std::string& executor_id, const std::array<uint8_t, 16>& session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = executors_.find(executor_id);
  if (it == executors_.end() || it->second.session_id != session_id || it->second.pending.empty()) {
    return std::nullopt;
  }
  DispatchedTransfer out = std::move(it->second.pending.front());
  it->second.pending.pop_front();
  it->second.fetched.insert(out.id);
  return out;
}

bool EngineDispatchRegistry::Complete(const std::string& executor_id,
                                      const std::array<uint8_t, 16>& session_id,
                                      uint64_t transfer_id, DispatchOutcome outcome,
                                      std::vector<TransferFailure> failures) {
  std::shared_ptr<DispatchTicket> ticket;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = executors_.find(executor_id);
    if (it == executors_.end() || it->second.session_id != session_id) return false;
    const auto now = std::chrono::steady_clock::now();
    for (auto tombstone = it->second.tombstones.begin();
         tombstone != it->second.tombstones.end();) {
      tombstone =
          tombstone->second <= now ? it->second.tombstones.erase(tombstone) : std::next(tombstone);
    }
    auto transfer = it->second.in_flight.find(transfer_id);
    if (transfer == it->second.in_flight.end()) {
      if (it->second.tombstones.erase(transfer_id) == 0) return false;
      return true;
    }
    if (it->second.fetched.erase(transfer_id) == 0) return false;
    ticket = std::move(transfer->second);
    it->second.in_flight.erase(transfer);
  }
  ticket->Complete(outcome, std::move(failures));
  return true;
}

int EngineDispatchRegistry::EventFd(const std::string& executor_id,
                                    const std::array<uint8_t, 16>& session_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = executors_.find(executor_id);
  return it == executors_.end() || it->second.session_id != session_id ? -1 : it->second.event_fd;
}

bool EngineDispatchRegistry::FailFetched(const std::string& executor_id,
                                         const std::array<uint8_t, 16>& session_id,
                                         uint64_t transfer_id, DispatchOutcome outcome) {
  std::shared_ptr<DispatchTicket> ticket;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = executors_.find(executor_id);
    if (it == executors_.end() || it->second.session_id != session_id ||
        it->second.fetched.erase(transfer_id) == 0) {
      return false;
    }
    auto transfer = it->second.in_flight.find(transfer_id);
    if (transfer == it->second.in_flight.end()) return false;
    ticket = std::move(transfer->second);
    it->second.in_flight.erase(transfer);
  }
  ticket->Complete(outcome, {});
  return true;
}

#ifdef MORI_UMBP_TESTING
size_t EngineDispatchRegistry::PendingCountForTest(const std::string& executor_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = executors_.find(executor_id);
  return it == executors_.end() ? 0 : it->second.pending.size();
}

bool EngineDispatchRegistry::RegisteredForTest(const std::string& executor_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return executors_.count(executor_id) != 0;
}
#endif

}  // namespace mori::umbp::engine
