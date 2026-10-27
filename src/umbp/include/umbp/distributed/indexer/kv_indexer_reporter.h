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

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "sgl_kv_indexer.grpc.pb.h"
#include "umbp/common/config.h"

namespace mori::umbp {
namespace indexer_pb = ::kv_indexer::v1;

// Event-driven reconciliation of caller metadata with committed local DRAM.
// Backend callbacks only wake this worker; snapshots and RPCs run off-path.
class KvIndexerReporter {
 public:
  using Snapshot = std::function<std::unordered_set<std::string>()>;
  KvIndexerReporter(const KvIndexerConfig& config, Snapshot snapshot)
      : components_(config.components),
        timeout_(config.timeout_ms),
        snapshot_(std::move(snapshot)) {
    if (config.endpoint.empty() || config.worker_id.empty() || config.worker_address.empty() ||
        !(components_ & 1) || (components_ & ~7u) ||
        ((components_ & 2) && !config.swa_window_tokens) || !config.timeout_ms)
      throw std::invalid_argument("invalid KV Indexer identity, cache spec or timeout");
    stub_ = indexer_pb::KVIndexer::NewStub(
        grpc::CreateChannel(config.endpoint, grpc::InsecureChannelCredentials()));
    request_.set_worker_id(config.worker_id);
    request_.set_worker_address(config.worker_address);
    auto* spec = request_.mutable_cache_spec();
    spec->set_version(1);
    spec->set_components(components_);
    spec->set_swa_window_tokens(config.swa_window_tokens);
    spec->set_full_tier_mask(6);  // HBM + DRAM are eligible for prefix reuse.
    spec->set_swa_tier_mask(6);
    spec->set_mamba_tier_mask(6);
    thread_ = std::thread([this] { Run(); });
  }
  ~KvIndexerReporter() { Stop(); }

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
    if (mask && (mask != components_ || !block.block_size))
      throw std::invalid_argument(
          "metadata must describe all configured components and token count");
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      auto old = blocks_.find(block.hash);
      if (mask && old != blocks_.end() &&
          (old->second.parent != block.parent || old->second.block_size != block.block_size))
        throw std::invalid_argument("a block hash cannot change parent or token count");
      if (mask && old != blocks_.end() && old->second.component_keys == block.component_keys)
        return;  // Repeated physical writes need no metadata update or ancestor walk.
      if (mask) {
        // Forward references are allowed, but cannot close a parent cycle.
        for (auto parent = block.parent; parent;) {
          if (*parent == block.hash) throw std::invalid_argument("KV parent cycle");
          auto ancestor = blocks_.find(*parent);
          parent = ancestor == blocks_.end() ? std::nullopt : ancestor->second.parent;
        }
        blocks_.insert_or_assign(block.hash, block);
      } else {
        blocks_.erase(block.hash);
      }
    }
    Notify();
  }

  bool Sync(bool rebuild = false) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    try {
      request_.clear_actions();
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
      // Rebuild from fresh state after ambiguous failures and quiet Indexer restarts.
      if (retry_ || rebuild) {
        AddAction(indexer_pb::ACTION_CLEAR_ALL_AT_TIER);
        reported_.clear();
      }
      for (const auto& [hash, mask] : reported_) {
        if (stopping_) return false;
        if (!desired.count(hash)) AddAction(indexer_pb::ACTION_REVOKE, hash);
      }
      for (const auto& [hash, mask] : desired) {
        if (stopping_) return false;
        auto old = reported_.find(hash);
        if (old == reported_.end() || old->second != mask) {
          AddAction(indexer_pb::ACTION_REPORT, hash, mask);
        }
      }
      SendBatch();
      reported_ = std::move(desired);
      retry_ = false;
      return true;
    } catch (const std::exception& error) {
      retry_ = true;
      Notify();  // A failed caller-initiated flush must also wake the worker.
      MORI_UMBP_WARN("[KVIndexer] sync failed: {}", error.what());
      return false;
    }
  }
  void Stop() {
    stopping_ = true;
    Notify();
    if (!thread_.joinable()) return;
    thread_.join();
    try {
      request_.clear_actions();
      AddAction(indexer_pb::ACTION_CLEAR_ALL_AT_TIER);
      SendBatch();
    } catch (const std::exception& error) {
      MORI_UMBP_WARN("[KVIndexer] withdrawal failed: {}", error.what());
    }
  }

 private:
  // Append an action and flush when the protocol's batch limit is reached.
  void AddAction(indexer_pb::ExternalKvActionType type, int64_t hash = 0, uint32_t mask = 0) {
    auto* action = request_.add_actions();
    action->set_type(type);
    action->set_tier(indexer_pb::TIER_DRAM);
    if (type != indexer_pb::ACTION_CLEAR_ALL_AT_TIER) action->add_hashes(hash);
    if (type == indexer_pb::ACTION_REPORT) {
      const auto& block = blocks_.at(hash);
      if (block.parent) action->set_parent_block_hash(*block.parent);
      action->add_block_sizes(block.block_size);
      action->add_component_masks(mask);
    }
    // One hash per action: also below the Indexer's 16384-hash request limit.
    if (request_.actions_size() == 256) SendBatch();
  }
  void SendBatch() {
    if (request_.actions().empty()) return;
    request_.set_seq(request_.seq() + 1);
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + timeout_);
    indexer_pb::ApplyExternalKvBatchResponse response;
    auto status = stub_->ApplyExternalKvBatch(&context, request_, &response);
    if (!status.ok())
      throw std::runtime_error(std::to_string(status.error_code()) + ": " + status.error_message());
    request_.clear_actions();
  }
  void Run() {
    auto next_rebuild = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    std::unique_lock<std::mutex> lock(wake_mutex_);
    while (!stopping_) {
      const auto now = std::chrono::steady_clock::now();
      const bool rebuild = now >= next_rebuild;
      if (rebuild) next_rebuild = now + std::chrono::seconds(30);
      dirty_ = false;
      lock.unlock();
      const bool ok = Sync(rebuild);
      lock.lock();
      if (ok)
        cv_.wait_until(lock, next_rebuild, [this] { return dirty_ || stopping_; });
      else
        // Writes must not turn an unavailable Indexer into a retry storm.
        cv_.wait_for(lock, std::chrono::seconds(1), [this] { return stopping_.load(); });
    }
  }
  std::unique_ptr<indexer_pb::KVIndexer::Stub> stub_;
  indexer_pb::ApplyExternalKvBatchRequest request_;
  const uint32_t components_;
  const std::chrono::milliseconds timeout_;
  Snapshot snapshot_;
  // State/RPC serialization is separate from the short storage-notification lock.
  std::mutex state_mutex_, wake_mutex_;
  std::condition_variable cv_;
  std::map<int64_t, KvBlockMetadata> blocks_;
  std::map<int64_t, uint32_t> reported_;
  bool retry_ = true, dirty_ = true;
  std::atomic<bool> stopping_{false};
  std::thread thread_;
};
}  // namespace mori::umbp
