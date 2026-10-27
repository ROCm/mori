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
#include "umbp/common/prefetch_queue.h"

#include <algorithm>

namespace mori::umbp {

struct PrefetchQueue::Task {
  std::vector<std::string> keys;
  PrefetchOptions options;
  PrefetchStatus status;
  Clock::time_point submitted, deadline, started, finished;
  size_t key_bytes = 0;
  bool done = false;
  bool retain = true;
};

PrefetchQueue::PrefetchQueue(Execute execute)
    : execute_(std::move(execute)), worker_([this] { Run(); }) {}
PrefetchQueue::~PrefetchQueue() { Stop(); }

void PrefetchQueue::PruneLocked() {
  const auto now = Clock::now();
  for (auto it = tasks_.begin(); it != tasks_.end();) {
    const auto& task = *it->second;
    if (task.done && (!task.retain || now - task.finished >= std::chrono::minutes(5)))
      it = tasks_.erase(it);
    else
      ++it;
  }
}

PrefetchSubmitResult PrefetchQueue::Submit(const std::vector<std::string>& keys,
                                           PrefetchOptions options, std::string id, bool retain) {
  if (keys.empty() || keys.size() > 4096 || id.size() > 256 ||
      std::any_of(keys.begin(), keys.end(),
                  [](const auto& k) { return k.empty() || k.size() > 4096; }))
    return {false, id, "invalid prefetch keys or request_id"};
  if (options.timeout.count() == 0) options.timeout = PrefetchOptions{}.timeout;
  if (options.lease_ttl.count() == 0) options.lease_ttl = PrefetchOptions{}.lease_ttl;
  if (options.timeout.count() < 0 || options.lease_ttl.count() < 0 ||
      options.timeout > std::chrono::minutes(5) || options.lease_ttl > std::chrono::minutes(5))
    return {false, id, "invalid prefetch duration"};
  std::lock_guard<std::mutex> lock(mutex_);
  PruneLocked();
  if (paused_ || stopped_) return {false, id, "prefetch is stopped or clearing"};
  const auto now = Clock::now();
  if (id.empty()) {
    do {
      id = "prefetch-" + std::to_string(now.time_since_epoch().count()) + "-" +
           std::to_string(next_id_++);
    } while (tasks_.count(id));
  }
  const auto existing = tasks_.find(id);
  if (existing != tasks_.end()) {
    const auto& task = *existing->second;
    if (task.keys != keys || task.options.timeout != options.timeout ||
        task.options.lease_ttl != options.lease_ttl)
      return {false, id, "request_id already names a different prefetch"};
    return {true, id, {}};
  }
  size_t key_bytes = 0;
  for (const auto& key : keys) key_bytes += key.size();
  size_t total_keys = keys.size(), total_bytes = key_bytes;
  for (const auto& entry : tasks_) {
    total_keys += entry.second->keys.size();
    total_bytes += entry.second->key_bytes;
  }
  if (tasks_.size() >= 1024 || total_keys > 65536 || total_bytes > (64ULL << 20))
    return {false, id, "prefetch queue is full"};
  auto task = std::make_shared<Task>();
  task->keys = keys;
  task->options = options;
  task->key_bytes = key_bytes;
  task->submitted = now;
  task->deadline = now + options.timeout;
  task->retain = retain;
  task->status.state = PrefetchState::kQueued;
  task->status.ok.resize(keys.size(), false);
  tasks_.emplace(id, task);
  queue_.push_back(std::move(task));
  for (const auto& key : keys) ++pending_keys_[key];
  cv_.notify_all();
  return {true, id, {}};
}

PrefetchStatus PrefetchQueue::Status(const std::string& id) {
  std::lock_guard<std::mutex> lock(mutex_);
  PruneLocked();
  const auto it = tasks_.find(id);
  if (it == tasks_.end()) return {};
  auto status = it->second->status;
  // A deadline changes the visible result, never the lifetime of in-flight I/O.
  if (!PrefetchTerminal(status.state) && Clock::now() >= it->second->deadline) {
    status.state = PrefetchState::kExpired;
    status.error = "prefetch deadline expired";
  }
  return status;
}

void PrefetchQueue::FinishLocked(Task& task, std::vector<bool> ok) {
  const auto now = Clock::now();
  ok.resize(task.keys.size(), false);
  if (paused_ || stopped_) std::fill(ok.begin(), ok.end(), false);
  auto& status = task.status;
  status.ok = std::move(ok);
  status.completed_keys = std::count(status.ok.begin(), status.ok.end(), true);
  if (paused_ || stopped_)
    status.state = PrefetchState::kFailed;
  else if (now >= task.deadline)
    status.state = PrefetchState::kExpired;
  else if (status.completed_keys == task.keys.size())
    status.state = PrefetchState::kReady;
  else
    status.state = status.completed_keys ? PrefetchState::kPartial : PrefetchState::kFailed;
  if (status.state != PrefetchState::kReady) status.error = "prefetch incomplete or canceled";
  const auto start = task.started == Clock::time_point{} ? now : task.started;
  const auto us = [](auto d) {
    return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
  };
  status.queue_latency_us = us(start - task.submitted);
  status.execution_latency_us = us(now - start);
  status.total_latency_us = us(now - task.submitted);
  task.finished = now;
  task.done = true;
  for (const auto& key : task.keys) {
    const auto it = pending_keys_.find(key);
    if (--it->second == 0) pending_keys_.erase(it);
  }
  cv_.notify_all();
}

void PrefetchQueue::Run() {
  for (;;) {
    std::shared_ptr<Task> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return stopped_ || (!paused_ && !queue_.empty()); });
      if (stopped_) return;
      task = std::move(queue_.front());
      queue_.pop_front();
      active_ = true;
      task->started = Clock::now();
      task->status.state = PrefetchState::kRunning;
    }
    std::vector<bool> ok;
    try {
      if (Clock::now() < task->deadline) ok = execute_(task->keys, task->options, task->deadline);
    } catch (...) { /* Report failure and keep serving subsequent requests. */
    }
    std::lock_guard<std::mutex> lock(mutex_);
    FinishLocked(*task, std::move(ok));
    active_ = false;
    PruneLocked();
    cv_.notify_all();
  }
}

bool PrefetchQueue::WaitForKeys(const std::vector<std::string>& keys,
                                std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  return cv_.wait_for(lock, timeout, [&] {
    return paused_ || stopped_ || std::none_of(keys.begin(), keys.end(), [&](const auto& key) {
             return pending_keys_.count(key);
           });
  });
}

void PrefetchQueue::CancelLocked() {
  for (const auto& task : queue_) FinishLocked(*task, {});
  queue_.clear();
  for (auto& entry : tasks_) {
    auto& status = entry.second->status;
    status.state = PrefetchState::kFailed;
    std::fill(status.ok.begin(), status.ok.end(), false);
    status.completed_keys = 0;
    status.error = "prefetch cleared or stopped";
  }
  PruneLocked();
  cv_.notify_all();
}

void PrefetchQueue::PauseAndDrain() {
  std::unique_lock<std::mutex> lock(mutex_);
  paused_ = true;
  CancelLocked();
  cv_.wait(lock, [&] { return !active_; });
}
void PrefetchQueue::Resume() {
  std::lock_guard<std::mutex> lock(mutex_);
  paused_ = false;
  cv_.notify_all();
}
void PrefetchQueue::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
    CancelLocked();
  }
  if (worker_.joinable()) worker_.join();
}

}  // namespace mori::umbp
