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

// One bounded FIFO for Router requests, Engine hints and reactive prefetch.
// The executor returns only after I/O and final DRAM lease validation finish.
class PrefetchQueue {
 public:
  using Clock = std::chrono::steady_clock;
  using Execute = std::function<std::vector<bool>(const std::vector<std::string>&, PrefetchOptions,
                                                  Clock::time_point)>;
  explicit PrefetchQueue(Execute execute);
  ~PrefetchQueue();
  PrefetchSubmitResult Submit(const std::vector<std::string>& keys, PrefetchOptions options = {},
                              std::string request_id = {}, bool retain = true);
  PrefetchStatus Status(const std::string& request_id);
  bool WaitForKeys(const std::vector<std::string>& keys, std::chrono::milliseconds timeout);
  void PauseAndDrain();
  void Resume();
  void Stop();

 private:
  struct Task;
  void Run();
  void FinishLocked(Task& task, std::vector<bool> ok);
  void CancelLocked();
  void PruneLocked();
  Execute execute_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::unordered_map<std::string, std::shared_ptr<Task>> tasks_;
  std::unordered_map<std::string, size_t> pending_keys_;
  std::deque<std::shared_ptr<Task>> queue_;
  uint64_t next_id_ = 1;
  bool active_ = false;
  bool paused_ = false;
  bool stopped_ = false;
  std::thread worker_;
};

}  // namespace mori::umbp
