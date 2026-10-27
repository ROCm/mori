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
#include "umbp/common/prefetch_manager.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace mori::umbp {

struct PrefetchManager::Operation {
  std::string key;
  PrefetchKeyResult result;
  bool running = false;
  bool done = false;
  std::vector<std::weak_ptr<Task>> subscribers;
};

struct PrefetchManager::Task {
  std::vector<std::string> keys;
  PrefetchOptions options;
  std::vector<std::shared_ptr<Operation>> operations;
  PrefetchStatus status;
  Clock::time_point submitted, deadline, started, finished, lease_until;
  bool finalizing = false;
  bool retain = true;
};

PrefetchManager::PrefetchManager(Execute execute, Protect protect, Limits limits, Observe observe)
    : observe_(std::move(observe)),
      execute_(std::move(execute)),
      protect_(std::move(protect)),
      limits_(limits) {
  if (!execute_ || !protect_ || limits_.workers == 0 || limits_.batch_keys == 0)
    throw std::invalid_argument("prefetch requires executors and nonzero worker/batch limits");
  try {
    for (size_t i = 0; i < limits_.workers; ++i) workers_.emplace_back([this] { Worker(); });
  } catch (...) {
    Stop();
    throw;
  }
}

PrefetchManager::~PrefetchManager() { Stop(); }

void PrefetchManager::Note(const char* event, uint64_t count) {
  if (observe_ && count) {
    try {
      observe_(event, count);
    } catch (...) { /* Metrics must not fail prefetch. */
    }
  }
}

void PrefetchManager::FinishLocked(Task& task, PrefetchState state, Clock::time_point now) {
  task.status.state = state;
  task.finished = now;
  const auto start = task.started == Clock::time_point{} ? now : task.started;
  const auto us = [](auto duration) {
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  };
  task.status.queue_latency_us = us(start - task.submitted);
  task.status.execution_latency_us = us(now - start);
  task.status.total_latency_us = us(now - task.submitted);
  task.status.completed_keys = std::count_if(task.status.results.begin(), task.status.results.end(),
                                             [](const auto& result) { return result.Ready(); });
}

void PrefetchManager::RefreshLocked(Clock::time_point now) {
  for (auto it = tasks_.begin(); it != tasks_.end();) {
    auto& task = *it->second;
    if (!PrefetchTerminal(task.status.state) && now >= task.deadline) {
      // No buffer ownership is attached to a task's terminal state. Running
      // operations remain alive in the worker and can serve other subscribers.
      for (size_t i = 0; i < task.operations.size(); ++i) {
        if (task.operations[i]->done)
          task.status.results[i] = task.operations[i]->result;
        else
          task.status.results[i].code = PrefetchCode::kExpired;
      }
      task.status.error = "prefetch deadline expired";
      FinishLocked(task, PrefetchState::kExpired, now);
      Note("expired_request");
    }
    if (!task.finalizing && PrefetchTerminal(task.status.state) &&
        (!task.retain || now - task.finished >= limits_.retention)) {
      it = tasks_.erase(it);
    } else {
      ++it;
    }
  }
}

PrefetchSubmitResult PrefetchManager::Submit(const std::vector<std::string>& keys,
                                             PrefetchOptions options, std::string id, bool retain) {
  PrefetchSubmitResult out;
  if (keys.empty() || keys.size() > limits_.keys_per_task ||
      std::any_of(keys.begin(), keys.end(),
                  [](const auto& key) { return key.empty() || key.size() > 4096; }) ||
      id.size() > 256) {
    out.error = "invalid prefetch keys or request_id";
    if (!out.accepted) Note("rejected_request");
    return out;
  }
  if (options.timeout.count() == 0) options.timeout = PrefetchOptions{}.timeout;
  if (options.lease_ttl.count() == 0) options.lease_ttl = PrefetchOptions{}.lease_ttl;
  if (options.timeout.count() < 0 || options.lease_ttl.count() < 0 ||
      options.timeout > std::chrono::minutes(5) || options.lease_ttl > std::chrono::minutes(5)) {
    out.error = "prefetch timeout or lease exceeds the accepted bound";
    if (!out.accepted) Note("rejected_request");
    return out;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const auto now = Clock::now();
  RefreshLocked(now);
  if (stopped_ || paused_) {
    out.error = "prefetch is stopped or clearing";
    if (!out.accepted) Note("rejected_request");
    return out;
  }
  if (id.empty()) {
    do {
      id = "prefetch-" + std::to_string(now.time_since_epoch().count()) + "-" +
           std::to_string(next_id_++);
    } while (tasks_.count(id));
  }
  out.request_id = id;
  auto existing = tasks_.find(id);
  if (existing != tasks_.end()) {
    const auto& task = *existing->second;
    out.accepted = task.keys == keys && task.options.timeout == options.timeout &&
                   task.options.lease_ttl == options.lease_ttl;
    if (!out.accepted) out.error = "request_id already names a different prefetch";
    if (!out.accepted) Note("rejected_request");
    return out;
  }
  size_t input_keys = keys.size();
  size_t key_bytes = 0;
  for (const auto& key : keys) key_bytes += key.size();
  for (const auto& [task_id, task] : tasks_) {
    input_keys += task->keys.size();
    for (const auto& key : task->keys) key_bytes += key.size();
  }
  std::unordered_set<std::string> new_keys;
  for (const auto& key : keys)
    if (!operations_.count(key)) new_keys.insert(key);
  if (tasks_.size() >= limits_.tasks ||
      operations_.size() + new_keys.size() > limits_.inflight_keys ||
      input_keys > limits_.retained_input_keys || key_bytes > limits_.retained_key_bytes) {
    out.error = "prefetch task or key queue is full";
    if (!out.accepted) Note("rejected_request");
    return out;
  }
  auto task = std::make_shared<Task>();
  task->keys = keys;
  task->options = options;
  task->submitted = now;
  task->deadline = now + options.timeout;
  task->retain = retain;
  task->status.state = PrefetchState::kQueued;
  task->status.results.resize(keys.size());
  std::unordered_set<Operation*> subscribed;
  for (const auto& key : keys) {
    auto& op = operations_[key];
    if (!op) {
      op = std::make_shared<Operation>();
      op->key = key;
      queue_.push_back(op);
    }
    if (subscribed.insert(op.get()).second) op->subscribers.push_back(task);
    task->operations.push_back(op);
    if (op->running) {
      task->status.state = PrefetchState::kRunning;
      task->started = now;
    }
  }
  tasks_.emplace(id, std::move(task));
  out.accepted = true;
  Note("accepted_request");
  Note("joined_key", keys.size() - new_keys.size());
  cv_.notify_all();
  return out;
}

PrefetchStatus PrefetchManager::Status(const std::string& id) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto now = Clock::now();
  RefreshLocked(now);
  const auto it = tasks_.find(id);
  if (it == tasks_.end()) return {};
  auto status = it->second->status;
  if (it->second->lease_until > now) {
    status.lease_remaining_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(it->second->lease_until - now)
            .count();
  }
  return status;
}

bool PrefetchManager::InterestedLocked(const Operation& op, Clock::time_point now) const {
  for (const auto& weak : op.subscribers) {
    const auto task = weak.lock();
    if (task && !PrefetchTerminal(task->status.state) && now < task->deadline) return true;
  }
  return false;
}

void PrefetchManager::Worker() {
  for (;;) {
    std::vector<std::shared_ptr<Operation>> batch;
    std::vector<std::string> keys;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return stopped_ || (!paused_ && !queue_.empty()); });
      if (stopped_) return;
      const auto now = Clock::now();
      RefreshLocked(now);
      while (!queue_.empty() && batch.size() < limits_.batch_keys) {
        auto op = std::move(queue_.front());
        queue_.pop_front();
        if (!InterestedLocked(*op, now)) {
          op->done = true;
          op->result.code = PrefetchCode::kExpired;
          operations_.erase(op->key);
          continue;
        }
        op->running = true;
        for (const auto& weak : op->subscribers) {
          const auto task = weak.lock();
          if (task && task->status.state == PrefetchState::kQueued) {
            task->status.state = PrefetchState::kRunning;
            task->started = now;
          }
        }
        keys.push_back(op->key);
        batch.push_back(std::move(op));
      }
      if (batch.empty()) {
        cv_.notify_all();
        continue;
      }
      ++active_;
    }
    std::vector<PrefetchKeyResult> results;
    try {
      results = execute_(keys);
    } catch (...) { /* Fail the batch; keep the workers alive. */
    }
    results.resize(keys.size(), PrefetchKeyResult{PrefetchCode::kTransferFailed});
    std::vector<std::shared_ptr<Task>> finishing;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (size_t i = 0; i < batch.size(); ++i) {
        batch[i]->result = results[i];
        batch[i]->done = true;
        operations_.erase(batch[i]->key);
      }
      RefreshLocked(Clock::now());
      std::unordered_set<Task*> notified;
      std::vector<std::shared_ptr<Task>> affected;
      for (const auto& op : batch) {
        for (const auto& weak : op->subscribers) {
          auto task = weak.lock();
          if (task && notified.insert(task.get()).second) affected.push_back(std::move(task));
        }
      }
      for (const auto& task : affected) {
        if (task->finalizing || PrefetchTerminal(task->status.state)) continue;
        for (size_t i = 0; i < task->operations.size(); ++i) {
          if (task->operations[i]->done) task->status.results[i] = task->operations[i]->result;
        }
        task->status.completed_keys =
            std::count_if(task->status.results.begin(), task->status.results.end(),
                          [](const auto& r) { return r.Ready(); });
        if (std::all_of(task->operations.begin(), task->operations.end(),
                        [](const auto& op) { return op->done; })) {
          task->finalizing = true;
          finishing.push_back(task);
        }
      }
      cv_.notify_all();
    }
    for (const auto& task : finishing) {
      // Work on a private result vector: Status may expire this task concurrently.
      auto protected_results = std::vector<PrefetchKeyResult>{};
      for (const auto& op : task->operations) protected_results.push_back(op->result);
      const auto lease_start = Clock::now();
      try {
        protect_(task->keys, &protected_results, task->options.lease_ttl);
      } catch (...) {
        for (auto& r : protected_results) r.code = PrefetchCode::kNotResident;
      }
      const auto now = Clock::now();
      if (now >= lease_start + task->options.lease_ttl) {
        for (auto& result : protected_results)
          if (result.Ready()) result.code = PrefetchCode::kNotResident;
      }
      std::lock_guard<std::mutex> lock(mutex_);
      task->finalizing = false;
      if (PrefetchTerminal(task->status.state)) continue;
      task->status.results = std::move(protected_results);
      if (now >= task->deadline) {
        task->status.error = "prefetch completed after its deadline";
        FinishLocked(*task, PrefetchState::kExpired, now);
        Note("expired_request");
      } else {
        const size_t ready = std::count_if(task->status.results.begin(), task->status.results.end(),
                                           [](const auto& r) { return r.Ready(); });
        Note("ready_key", ready);
        Note("failed_key", task->keys.size() - ready);
        if (ready) task->lease_until = lease_start + task->options.lease_ttl;
        if (ready != task->keys.size())
          task->status.error = "some keys could not be prepared in DRAM";
        FinishLocked(*task,
                     ready == task->keys.size() ? PrefetchState::kReady
                     : ready                    ? PrefetchState::kPartial
                                                : PrefetchState::kFailed,
                     now);
      }
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      --active_;
      RefreshLocked(Clock::now());
      cv_.notify_all();
    }
  }
}

bool PrefetchManager::WaitForKeys(const std::vector<std::string>& keys,
                                  std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  std::vector<std::shared_ptr<Operation>> waiting;
  const auto now = Clock::now();
  for (const auto& key : keys) {
    const auto it = operations_.find(key);
    if (it != operations_.end() && InterestedLocked(*it->second, now))
      waiting.push_back(it->second);
  }
  const bool done = cv_.wait_for(lock, timeout, [&] {
    return stopped_ || paused_ ||
           std::all_of(waiting.begin(), waiting.end(), [](const auto& op) { return op->done; });
  });
  if (!done) Note("wait_timeout");
  return done;
}

bool PrefetchManager::ShouldRun(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = operations_.find(key);
  return !paused_ && !stopped_ && it != operations_.end() &&
         InterestedLocked(*it->second, Clock::now());
}

void PrefetchManager::CancelLocked() {
  for (auto& op : queue_) {
    op->done = true;
    op->result.code = PrefetchCode::kStopped;
    operations_.erase(op->key);
  }
  queue_.clear();
  for (auto& [id, task] : tasks_) {
    task->status.error = "prefetch cleared or stopped";
    for (auto& r : task->status.results) r.code = PrefetchCode::kStopped;
    task->lease_until = {};
    FinishLocked(*task, PrefetchState::kFailed, Clock::now());
  }
  cv_.notify_all();
}

void PrefetchManager::PauseAndDrain() {
  std::unique_lock<std::mutex> lock(mutex_);
  paused_ = true;
  CancelLocked();
  cv_.wait(lock, [&] { return active_ == 0; });
}

void PrefetchManager::Resume() {
  std::lock_guard<std::mutex> lock(mutex_);
  paused_ = false;
  cv_.notify_all();
}

void PrefetchManager::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
    CancelLocked();
  }
  for (auto& worker : workers_)
    if (worker.joinable()) worker.join();
}

}  // namespace mori::umbp
