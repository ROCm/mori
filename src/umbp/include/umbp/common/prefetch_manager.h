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
#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "umbp/common/prefetch.h"

namespace mori::umbp {

// Shared by hints, explicit tasks and demand-read locality prefetch. The
// executor is synchronous: it must retain all destination buffers until the
// actual I/O completes, even when every subscriber's deadline has passed.
class PrefetchManager {
 public:
  struct Limits {
    size_t tasks = 1024;
    size_t keys_per_task = 4096;
    size_t inflight_keys = 8192;
    size_t retained_input_keys = 65536;
    size_t retained_key_bytes = 64ULL << 20;
    size_t batch_keys = 128;
    size_t workers = 2;
    std::chrono::milliseconds retention{300000};
  };
  using Execute = std::function<std::vector<PrefetchKeyResult>(const std::vector<std::string>&)>;
  // Revalidate and lease all successful keys just before task completion.
  // Called outside the manager mutex; may downgrade results that were evicted.
  using Protect = std::function<void(const std::vector<std::string>&,
                                     std::vector<PrefetchKeyResult>*, std::chrono::milliseconds)>;

  // Receives low-cardinality cumulative events. Must not call back into the
  // manager; notifications can run while its state mutex is held.
  using Observe = std::function<void(const char*, uint64_t)>;
  PrefetchManager(Execute execute, Protect protect, Limits limits, Observe observe = {});
  ~PrefetchManager();
  PrefetchSubmitResult Submit(const std::vector<std::string>& keys, PrefetchOptions options = {},
                              std::string request_id = {}, bool retain = true);
  PrefetchStatus Status(const std::string& request_id);
  // Wait only for operations already associated with these keys, within one
  // shared budget. False means the caller should use its ordinary read path.
  bool WaitForKeys(const std::vector<std::string>& keys, std::chrono::milliseconds timeout);
  // Executors recheck this before starting additional I/O batches. An expired
  // subscriber does not cancel another subscriber's work.
  bool ShouldRun(const std::string& key);
  void PauseAndDrain();
  void Resume();
  void Stop();

 private:
  using Clock = std::chrono::steady_clock;
  struct Operation;
  struct Task;
  void Worker();
  void RefreshLocked(Clock::time_point now);
  void FinishLocked(Task& task, PrefetchState state, Clock::time_point now);
  void CancelLocked();
  bool InterestedLocked(const Operation& op, Clock::time_point now) const;

  void Note(const char* event, uint64_t count = 1);
  Observe observe_;
  Execute execute_;
  Protect protect_;
  Limits limits_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::unordered_map<std::string, std::shared_ptr<Task>> tasks_;
  std::unordered_map<std::string, std::shared_ptr<Operation>> operations_;
  std::deque<std::shared_ptr<Operation>> queue_;
  std::vector<std::thread> workers_;
  uint64_t next_id_ = 1;
  size_t active_ = 0;
  bool paused_ = false;
  bool stopped_ = false;
};

}  // namespace mori::umbp
