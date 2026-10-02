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

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace mori::umbp {

// Says that a long blocking call is still running.
//
// A few UMBP setup steps take minutes and report nothing until they return --
// pinning a multi-terabyte host tier in one hipHostRegister, a registration RPC
// against a server busy pinning its own. In flight, they are indistinguishable
// from a hang, which is what makes every timeout around them a blind guess.
//
// Costs one thread for the object's lifetime, so this belongs on setup paths,
// not hot ones. `report` runs on that thread, is handed the elapsed time, must
// not throw, and must not touch what the awaited call is mutating.
class ScopedProgressLog {
 public:
  ScopedProgressLog(std::chrono::milliseconds interval,
                    std::function<void(std::chrono::milliseconds)> report)
      : interval_(interval), report_(std::move(report)) {
    // Disabled forms, so call sites need no enable/disable branch of their own.
    if (interval_.count() > 0 && report_) {
      thread_ = std::thread([this] { Loop(); });
    }
  }

  ~ScopedProgressLog() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      done_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
  }

  ScopedProgressLog(const ScopedProgressLog&) = delete;
  ScopedProgressLog& operator=(const ScopedProgressLog&) = delete;

  // Callers want this for the message they log when the call finally returns.
  std::chrono::milliseconds Elapsed() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                 start_);
  }

 private:
  void Loop() {
    std::unique_lock<std::mutex> lock(mu_);
    // wait_for returns false on timeout, i.e. the awaited call is still
    // running. Reporting outside the lock would let the destructor join a
    // thread that is mid-report.
    while (!cv_.wait_for(lock, interval_, [this] { return done_; })) {
      report_(Elapsed());
    }
  }

  const std::chrono::milliseconds interval_;
  const std::function<void(std::chrono::milliseconds)> report_;
  const std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
  std::mutex mu_;
  std::condition_variable cv_;
  bool done_ = false;
  std::thread thread_;
};

}  // namespace mori::umbp
