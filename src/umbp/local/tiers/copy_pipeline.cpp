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
#include "umbp/local/tiers/copy_pipeline.h"

#include <algorithm>

namespace mori::umbp {

CopyPipeline::CopyPipeline(LocalStorageManager& storage, const UMBPCopyPipelineConfig& config,
                           UMBPRole role)
    : storage_(storage), config_(config), role_(role) {
  if (role_ == UMBPRole::SharedSSDLeader && config_.async_enabled) {
    const size_t n_workers = std::max<size_t>(1, config_.worker_threads);
    copy_workers_.reserve(n_workers);
    for (size_t i = 0; i < n_workers; ++i) {
      copy_workers_.emplace_back(&CopyPipeline::CopyWorkerLoop, this);
    }
  }
}

CopyPipeline::~CopyPipeline() {
  if (copy_workers_.empty()) return;
  {
    std::lock_guard<std::mutex> lock(copy_mu_);
    stop_copy_worker_.store(true);
  }
  copy_cv_.notify_all();
  for (auto& worker : copy_workers_) {
    if (worker.joinable()) worker.join();
  }
}

bool CopyPipeline::MaybeCopyToSharedSSD(const std::string& key) {
  if (role_ != UMBPRole::SharedSSDLeader) return true;
  if (!config_.async_enabled) {
    storage_.CopyToSSD(key);
    return true;
  }
  if (!EnqueueCopyToSSD(key)) {
    storage_.CopyToSSD(key);
  }
  return true;
}

void CopyPipeline::MaybeBatchCopyToSharedSSD(const std::vector<std::string>& keys) {
  if (role_ != UMBPRole::SharedSSDLeader || keys.empty()) return;

  if (!config_.async_enabled) {
    storage_.CopyToSSDBatch(keys);
    return;
  }

  size_t enqueued = EnqueueCopyToSSDBatch(keys);
  if (enqueued < keys.size()) {
    std::vector<std::string> overflow(keys.begin() + enqueued, keys.end());
    storage_.CopyToSSDBatch(overflow);
  }
}

bool CopyPipeline::EnqueueCopyToSSD(const std::string& key) {
  std::unique_lock<std::mutex> lock(copy_mu_);
  if (copy_queue_.size() >= config_.queue_depth) return false;
  copy_queue_.push_back({key});
  lock.unlock();
  copy_cv_.notify_one();
  return true;
}

size_t CopyPipeline::EnqueueCopyToSSDBatch(const std::vector<std::string>& keys) {
  std::unique_lock<std::mutex> lock(copy_mu_);
  size_t remaining =
      (config_.queue_depth > copy_queue_.size()) ? config_.queue_depth - copy_queue_.size() : 0;
  size_t to_enqueue = std::min(keys.size(), remaining);
  for (size_t i = 0; i < to_enqueue; ++i) {
    copy_queue_.push_back({keys[i]});
  }
  lock.unlock();
  if (to_enqueue > 0) copy_cv_.notify_all();
  return to_enqueue;
}

void CopyPipeline::CopyWorkerLoop() {
  const size_t batch_max = std::max<size_t>(1, config_.batch_max_ops);
  while (true) {
    std::vector<std::string> batch;
    {
      std::unique_lock<std::mutex> lock(copy_mu_);
      copy_cv_.wait(lock, [&]() { return stop_copy_worker_.load() || !copy_queue_.empty(); });
      if (stop_copy_worker_.load() && copy_queue_.empty()) return;
      size_t n = std::min(batch_max, copy_queue_.size());
      batch.reserve(n);
      for (size_t i = 0; i < n; ++i) {
        batch.push_back(std::move(copy_queue_.front().key));
        copy_queue_.pop_front();
      }
    }

    if (batch.size() == 1) {
      storage_.CopyToSSD(batch[0]);
    } else {
      storage_.CopyToSSDBatch(batch);
    }
  }
}

}  // namespace mori::umbp
