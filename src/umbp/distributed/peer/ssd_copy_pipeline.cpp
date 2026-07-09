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
#include "umbp/distributed/peer/ssd_copy_pipeline.h"

#include <algorithm>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/peer/peer_dram_allocator.h"
#include "umbp/distributed/peer/peer_ssd_manager.h"

namespace mori::umbp {

namespace {

// RAII guard: releases a DramCopyPin on every scope exit (success, failure,
// or exception), so a worker can never leak a pin and strand a key's DRAM
// pages against eviction.
class DramCopyPinGuard {
 public:
  DramCopyPinGuard(PeerDramAllocator* dram, std::string key, uint64_t token)
      : dram_(dram), key_(std::move(key)), token_(token) {}
  ~DramCopyPinGuard() { dram_->ReleaseDramCopyPin(key_, token_); }

  DramCopyPinGuard(const DramCopyPinGuard&) = delete;
  DramCopyPinGuard& operator=(const DramCopyPinGuard&) = delete;

 private:
  PeerDramAllocator* dram_;
  std::string key_;
  uint64_t token_;
};

}  // namespace

SsdCopyPipeline::SsdCopyPipeline(PeerDramAllocator* dram, PeerSsdManager* ssd, size_t queue_depth,
                                 size_t worker_threads)
    : dram_(dram),
      ssd_(ssd),
      queue_depth_(std::max<size_t>(1, queue_depth)),
      worker_threads_(std::max<size_t>(1, worker_threads)) {}

SsdCopyPipeline::~SsdCopyPipeline() { Stop(); }

void SsdCopyPipeline::Start() {
  std::lock_guard<std::mutex> lock(mu_);
  if (!workers_.empty()) return;  // already started
  stop_ = false;
  paused_ = false;
  workers_.reserve(worker_threads_);
  for (size_t i = 0; i < worker_threads_; ++i) {
    workers_.emplace_back([this] { WorkerLoop(); });
  }
  MORI_UMBP_INFO("[SsdCopyPipeline] started workers={} queue_depth={}", worker_threads_,
                 queue_depth_);
}

void SsdCopyPipeline::Stop() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (stop_ && workers_.empty()) return;
    stop_ = true;
    queue_.clear();  // drop queued tasks; in-flight task finishes below
  }
  cv_.notify_all();
  for (auto& w : workers_) {
    if (w.joinable()) w.join();
  }
  workers_.clear();
}

void SsdCopyPipeline::Quiesce() {
  std::unique_lock<std::mutex> lock(mu_);
  paused_ = true;
  queue_.clear();
  // Block until the in-flight task (if any) has finished and released its pin.
  idle_cv_.wait(lock, [this] { return active_ == 0; });
}

void SsdCopyPipeline::Resume() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (stop_) return;
    paused_ = false;
  }
  cv_.notify_all();
}

bool SsdCopyPipeline::Enqueue(SsdCopyTask task) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (stop_ || paused_) {
      metrics_.dropped_stopped.fetch_add(1, std::memory_order_relaxed);
      MORI_UMBP_DEBUG("[SsdCopyPipeline] drop key='{}' — pipeline {}", task.key,
                      stop_ ? "stopped" : "paused");
      return false;
    }
    if (queue_.size() >= queue_depth_) {
      metrics_.dropped_queue_full.fetch_add(1, std::memory_order_relaxed);
      MORI_UMBP_DEBUG("[SsdCopyPipeline] drop key='{}' — queue full (depth={})", task.key,
                      queue_depth_);
      return false;  // full — drop, never block commit
    }
    queue_.push_back(std::move(task));
    metrics_.enqueued.fetch_add(1, std::memory_order_relaxed);
  }
  cv_.notify_one();
  return true;
}

void SsdCopyPipeline::WorkerLoop() {
  while (true) {
    SsdCopyTask task;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [this] { return stop_ || (!paused_ && !queue_.empty()); });
      if (stop_) return;                        // queue already cleared by Stop
      if (paused_ || queue_.empty()) continue;  // quiescing / spurious wakeup
      task = std::move(queue_.front());
      queue_.pop_front();
      ++active_;
    }

    RunTask(task);

    {
      std::lock_guard<std::mutex> lock(mu_);
      --active_;
      if (active_ == 0) idle_cv_.notify_all();
    }
  }
}

void SsdCopyPipeline::RunTask(const SsdCopyTask& task) {
  auto pin = dram_->AcquireDramCopyPin(task.key);
  if (!pin.has_value()) {
    // Key was already evicted (or a duplicate task) — nothing to copy.
    MORI_UMBP_DEBUG("[SsdCopyPipeline] key='{}' not pinnable (evicted/duplicate) — drop", task.key);
    return;
  }
  // From here the pin is released no matter how we exit.
  DramCopyPinGuard guard(dram_, task.key, pin->pin_token);

  if (pin->segments.empty() || pin->total_size == 0) {
    metrics_.failed.fetch_add(1, std::memory_order_relaxed);
    MORI_UMBP_WARN("[SsdCopyPipeline] key='{}' has no readable segments (size={}) — skip", task.key,
                   pin->total_size);
    return;
  }

  // Backend IO runs outside the allocator lock; the pin keeps the source
  // pages alive for the duration of this synchronous Write.
  if (ssd_->Write(task.key, pin->segments, pin->total_size, pin->encoding)) {
    metrics_.copied_ok.fetch_add(1, std::memory_order_relaxed);
  } else {
    metrics_.failed.fetch_add(1, std::memory_order_relaxed);
  }
}

}  // namespace mori::umbp
