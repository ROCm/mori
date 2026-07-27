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

// A tiny fixed-size worker pool used to issue a multi-endpoint fan-out's
// per-instance Redis calls concurrently, so a RouteGet's latency is one round
// trip instead of N sequential ones (matters on high-RTT remote stores). It is
// deliberately minimal: persistent workers (no per-call thread churn — that was
// measured to cost more than it saves), a mutex+cv task queue, and futures that
// carry a task's exception back to the caller.
//
// Deadlock-free for the fan-out use case: worker threads only RUN tasks (a
// blocking Redis call) and never wait on other tasks; only the caller waits on
// its futures. An undersized pool merely reduces parallelism (tasks queue), it
// cannot deadlock.

#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace mori::umbp::redis {

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t num_threads) {
    if (num_threads == 0) num_threads = 1;
    workers_.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] { WorkerLoop(); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
      if (w.joinable()) w.join();
    }
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  std::size_t size() const { return workers_.size(); }

  // Enqueue a task; the returned future becomes ready when it runs and rethrows
  // any exception the task threw.
  std::future<void> Enqueue(std::function<void()> task) {
    std::packaged_task<void()> pt(std::move(task));
    std::future<void> fut = pt.get_future();
    {
      std::lock_guard<std::mutex> lk(mu_);
      tasks_.push(std::move(pt));
    }
    cv_.notify_one();
    return fut;
  }

 private:
  void WorkerLoop() {
    for (;;) {
      std::packaged_task<void()> task;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
        if (stop_ && tasks_.empty()) return;
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }

  std::vector<std::thread> workers_;
  std::queue<std::packaged_task<void()>> tasks_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
};

}  // namespace mori::umbp::redis
