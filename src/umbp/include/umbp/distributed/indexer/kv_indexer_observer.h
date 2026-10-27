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

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <thread>
#include <unordered_set>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/indexer/kv_indexer_reporter.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

// Event-driven reconciliation of caller metadata with committed local DRAM.
// Backend callbacks only wake this worker; snapshots and RPCs run off-path.
class KvIndexerObserver {
 public:
  using Snapshot = std::function<std::unordered_set<std::string>()>;
  KvIndexerObserver(const KvIndexerConfig& config, Snapshot snapshot)
      : reporter_(config.endpoint, config.worker_id, config.worker_address, config.components,
                  config.swa_window_tokens, config.timeout_ms),
        components_(config.components),
        snapshot_(std::move(snapshot)),
        thread_([this] { Run(); }) {}
  ~KvIndexerObserver() { Stop(); }

  void Notify() {
    {
      std::lock_guard<std::mutex> lock(wake_mutex_);
      dirty_ = true;
    }
    cv_.notify_one();
  }
  void Register(const KvBlockMetadata& block) {
    uint32_t mask = 0;
    for (const auto& [component, keys] : block.component_keys) {
      if ((component != 1 && component != 2 && component != 4) || keys.empty() ||
          std::any_of(keys.begin(), keys.end(), [](const auto& key) { return key.empty(); }))
        throw std::invalid_argument("invalid KV component or physical keys");
      mask |= component;
    }
    if (mask && (mask != components_ || !block.block_size || block.parent == block.hash))
      throw std::invalid_argument(
          "metadata must describe all configured components and token count");
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      auto old = blocks_.find(block.hash);
      if (mask && old != blocks_.end() &&
          (old->second.parent != block.parent || old->second.block_size != block.block_size))
        throw std::invalid_argument("a block hash cannot change parent or token count");
      if (mask)
        blocks_.insert_or_assign(block.hash, block);
      else
        blocks_.erase(block.hash);
    }
    Notify();
  }

  bool Sync() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    try {
      const auto resident = snapshot_();
      std::map<int64_t, uint32_t> desired;
      for (const auto& [hash, block] : blocks_) {
        uint32_t mask = 0;
        for (const auto& [component, keys] : block.component_keys) {
          if (std::all_of(keys.begin(), keys.end(),
                          [&](const auto& key) { return resident.count(key); }))
            mask |= component;
        }
        if (mask & 1) desired[hash] = mask;
      }
      const auto now = std::chrono::steady_clock::now();
      // Rebuild from fresh state after ambiguous failures and quiet Indexer restarts.
      if (retry_ || now - last_full_ >= std::chrono::seconds(30)) {
        reporter_.Clear();
        reported_.clear();
        last_full_ = now;
      }
      for (const auto& [hash, mask] : reported_) {
        if (stopping_) return false;
        if (!desired.count(hash)) reporter_.Revoke({hash});
      }
      for (const auto& [hash, mask] : desired) {
        if (stopping_) return false;
        auto old = reported_.find(hash);
        if (old == reported_.end() || old->second != mask) {
          const auto& block = blocks_.at(hash);
          reporter_.Report({hash}, block.parent, block.block_size, mask);
        }
      }
      reported_ = std::move(desired);
      retry_ = false;
      return true;
    } catch (const std::exception& error) {
      retry_ = true;
      MORI_UMBP_WARN("[KVIndexer] sync failed: {}", error.what());
      return false;
    }
  }
  void Stop() {
    {
      std::lock_guard<std::mutex> lock(wake_mutex_);
      stopping_ = true;
    }
    cv_.notify_one();
    if (!thread_.joinable()) return;
    thread_.join();
    try {
      reporter_.Clear();
    } catch (const std::exception& error) {
      MORI_UMBP_WARN("[KVIndexer] withdrawal failed: {}", error.what());
    }
  }

 private:
  void Run() {
    auto periodic = std::chrono::steady_clock::now();
    while (true) {
      std::unique_lock<std::mutex> lock(wake_mutex_);
      cv_.wait_for(lock, std::chrono::seconds(1), [this] { return dirty_ || stopping_; });
      if (stopping_) return;
      bool work = std::exchange(dirty_, false);
      lock.unlock();
      const auto now = std::chrono::steady_clock::now();
      if (work || retry_ || now - periodic >= std::chrono::seconds(30)) {
        Sync();
        periodic = now;
      }
    }
  }
  KvIndexerReporter reporter_;
  const uint32_t components_;
  Snapshot snapshot_;
  std::mutex state_mutex_, wake_mutex_;
  std::condition_variable cv_;
  std::map<int64_t, KvBlockMetadata> blocks_;
  std::map<int64_t, uint32_t> reported_;
  std::chrono::steady_clock::time_point last_full_{};
  std::atomic<bool> retry_{true};
  bool dirty_ = true;
  std::atomic<bool> stopping_{false};
  std::thread thread_;
};
}  // namespace mori::umbp
