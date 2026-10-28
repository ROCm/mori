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
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "sgl_kv_indexer.grpc.pb.h"
#include "umbp/common/config.h"

namespace mori::umbp {
namespace indexer_pb = ::kv_indexer::v1;

// Event-driven reconciliation of caller metadata with committed local DRAM.
// Backend callbacks identify changed keys; residency checks and RPCs run off-path.
class KvIndexerReporter {
  struct Block : KvBlockMetadata {
    explicit Block(const KvBlockMetadata& metadata) : KvBlockMetadata(metadata) {}
    int reported = -1;  // -1: absent; 0: an ambiguous REPORT may have been applied.
    uint32_t desired = 0;
  };
  using Blocks = std::map<int64_t, std::shared_ptr<Block>>;

 public:
  using Contains = std::function<bool(const std::string&)>;
  KvIndexerReporter(const KvIndexerConfig& config, Contains contains, Contains retained = {})
      : endpoint_(config.endpoint),
        pool_id_(config.dram_pool_id),
        timeout_(config.timeout_ms),
        contains_(std::move(contains)),
        retained_(retained ? std::move(retained) : contains_) {
    if (endpoint_.empty() || pool_id_.empty() || !config.timeout_ms)
      throw std::invalid_argument("KV Indexer requires an endpoint, pool id and timeout");
    stub_ = indexer_pb::KVIndexer::NewStub(
        grpc::CreateChannel(config.endpoint, grpc::InsecureChannelCredentials()));
    request_.set_dram_pool_id(pool_id_);
    thread_ = std::thread([this] { Run(); });
  }
  ~KvIndexerReporter() { Stop(); }

  // Forwarders may attach to the same pool without replacing its reporter.
  bool MatchesPool(const KvIndexerConfig& config) const {
    return config.dram_pool_id == pool_id_ && config.endpoint == endpoint_ &&
           std::chrono::milliseconds(config.timeout_ms) == timeout_;
  }

  // Empty means a clear or an explicit request to check all registered blocks.
  void Notify(const std::vector<std::string>& keys = {}) {
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (keys.empty()) rescan_ = true;
      for (const auto& key : keys) {
        auto it = key_blocks_.find(key);
        if (it != key_blocks_.end())
          for (auto hash : it->second) pending_.emplace(hash, blocks_.at(hash));
      }
    }
    cv_.notify_one();
  }
  KvMetadataPin Register(const KvBlockMetadata& block) {
    std::shared_ptr<Block> metadata;
    for (const auto& [component, keys] : block.component_keys) {
      if ((component != 1 && component != 2 && component != 4) || keys.empty() ||
          std::any_of(keys.begin(), keys.end(), [](const auto& key) { return key.empty(); }))
        throw std::invalid_argument("invalid KV component or physical keys");
    }
    if (block.component_keys.empty() || !block.block_size)
      throw std::invalid_argument("metadata requires component keys and token count");
    if (block.parent == block.hash) throw std::invalid_argument("KV parent cycle");
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      auto old = blocks_.find(block.hash);
      if (old != blocks_.end()) {
        if (old->second->parent != block.parent || old->second->block_size != block.block_size ||
            old->second->component_keys != block.component_keys)
          throw std::invalid_argument("a block hash must keep the same KV metadata");
        return old->second;
      }
      metadata = std::make_shared<Block>(block);
      blocks_[block.hash] = metadata;
      for (const auto& [component, keys] : block.component_keys) {
        for (const auto& key : keys) {
          auto& hashes = key_blocks_[key];
          if (hashes.empty() || hashes.back() != block.hash) hashes.push_back(block.hash);
        }
      }
      pending_.emplace(block.hash, metadata);
    }
    cv_.notify_one();
    return metadata;
  }

  bool Sync(bool rebuild = false) {
    std::lock_guard<std::mutex> sync_lock(sync_mutex_);
    if (stopping_) return false;
    rebuild |= repair_;
    Blocks work;
    std::optional<int64_t> last, cursor;
    bool scan;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      scan = rescan_ || rebuild;
      rescan_ = false;
      if (scan && !blocks_.empty()) last = blocks_.rbegin()->first;
      work.swap(pending_);
    }
    if (scan) work.clear();
    try {
      request_.clear_actions();
      batch_hashes_ = 0;
      // Startup alone clears a previous owner. Repair never hides live DRAM.
      if (startup_) {
        AddAction(indexer_pb::ACTION_CLEAR_ALL_AT_TIER);
        SendBatch();
        startup_ = false;
      }
      while (!stopping_) {
        if (scan) {
          std::lock_guard<std::mutex> lock(state_mutex_);
          auto it = cursor ? blocks_.upper_bound(*cursor) : blocks_.begin();
          while (last && it != blocks_.end() && it->first <= *last && work.size() < 4096) {
            cursor = it->first;
            work.insert(*it++);
          }
        }
        if (work.empty()) break;
        Reconcile(work, rebuild);
        work.clear();  // Release retired metadata outside the write-path lock.
        if (!scan) break;
      }
      repair_ = false;
      return !stopping_;
    } catch (const std::exception& error) {
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        rescan_ = true;
        repair_ = rebuild;
      }
      cv_.notify_one();
      MORI_UMBP_WARN("[KVIndexer] sync failed: {}", error.what());
      return false;
    }
  }
  void Stop() {
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      stopping_ = true;
    }
    cv_.notify_one();
    if (!thread_.joinable()) return;
    thread_.join();
    std::lock_guard<std::mutex> lock(sync_mutex_);
    try {
      request_.clear_actions();
      batch_hashes_ = 0;
      AddAction(indexer_pb::ACTION_CLEAR_ALL_AT_TIER);
      SendBatch();
    } catch (const std::exception& error) {
      MORI_UMBP_WARN("[KVIndexer] withdrawal failed: {}", error.what());
    }
  }

 private:
  void Reconcile(const Blocks& work, bool rebuild) {
    for (const auto& [hash, block] : work) {
      if (stopping_) return;
      uint32_t mask = 0;
      for (const auto& [component, keys] : block->component_keys)
        if (std::all_of(keys.begin(), keys.end(), contains_)) mask |= component;
      block->desired = mask;
      if (!mask) {
        if (block->reported >= 0) AddAction(indexer_pb::ACTION_REVOKE, hash);
      } else if (rebuild || block->reported != static_cast<int>(mask)) {
        // An applied-but-failed RPC must remain revocable after an eviction.
        block->reported = 0;
        AddAction(indexer_pb::ACTION_REPORT, hash, mask, block.get());
      }
    }
    SendBatch();
    for (const auto& [hash, block] : work) {
      block->reported = block->desired ? static_cast<int>(block->desired) : -1;
      if (!block->desired && std::none_of(block->component_keys.begin(),
                                          block->component_keys.end(), [&](const auto& part) {
                                            return std::any_of(part.second.begin(),
                                                               part.second.end(), retained_);
                                          }))
        Retire(block);
    }
  }

  void Retire(const std::shared_ptr<Block>& block) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    // The map and this sync own two references. Writers hold additional pins;
    // a concurrent commit/eviction queues another check before we can retire.
    if (block.use_count() != 2 || pending_.count(block->hash)) return;
    blocks_.erase(block->hash);
    for (const auto& [component, keys] : block->component_keys) {
      for (const auto& key : keys) {
        auto it = key_blocks_.find(key);
        if (it == key_blocks_.end()) continue;
        auto& hashes = it->second;
        hashes.erase(std::remove(hashes.begin(), hashes.end(), block->hash), hashes.end());
        if (hashes.empty()) key_blocks_.erase(it);
      }
    }
  }

  // REPORT hashes form a parent-child chain, never an arbitrary list. REVOKE
  // may group unrelated hashes. Both protocol limits apply to each request.
  void AddAction(indexer_pb::ExternalKvActionType type, int64_t hash = 0, uint32_t mask = 0,
                 const KvBlockMetadata* block = nullptr) {
    auto* action =
        request_.actions_size() ? request_.mutable_actions(request_.actions_size() - 1) : nullptr;
    bool append = action && action->type() == type &&
                  (type == indexer_pb::ACTION_REVOKE ||
                   (block && block->parent && action->hashes_size() &&
                    action->hashes(action->hashes_size() - 1) == *block->parent));
    if (!append) {
      if (request_.actions_size() == 256) SendBatch();
      action = request_.add_actions();
      action->set_type(type);
      action->set_tier(indexer_pb::TIER_DRAM);
      if (block && block->parent) action->set_parent_block_hash(*block->parent);
    }
    if (type != indexer_pb::ACTION_CLEAR_ALL_AT_TIER) {
      action->add_hashes(hash);
      ++batch_hashes_;
    }
    if (block) {
      action->add_block_sizes(block->block_size);
      action->add_component_masks(mask);
    }
    if (batch_hashes_ == 16384) SendBatch();
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
    batch_hashes_ = 0;
  }
  void Run() {
    auto next_rebuild = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    std::unique_lock<std::mutex> lock(state_mutex_);
    while (!stopping_) {
      cv_.wait_until(lock, next_rebuild,
                     [this] { return !pending_.empty() || rescan_ || stopping_; });
      const auto now = std::chrono::steady_clock::now();
      const bool rebuild = now >= next_rebuild;
      if (rebuild) next_rebuild = now + std::chrono::seconds(30);
      lock.unlock();
      const bool ok = Sync(rebuild);
      lock.lock();
      if (!ok)
        // Writes must not turn an unavailable Indexer into a retry storm.
        cv_.wait_for(lock, std::chrono::seconds(1), [this] { return stopping_.load(); });
    }
  }
  std::unique_ptr<indexer_pb::KVIndexer::Stub> stub_;
  indexer_pb::ApplyExternalKvBatchRequest request_;
  const std::string endpoint_, pool_id_;
  const std::chrono::milliseconds timeout_;
  Contains contains_, retained_;
  // Serialize RPCs/flush/shutdown independently from write-path metadata.
  std::mutex state_mutex_, sync_mutex_;
  std::condition_variable cv_;
  Blocks blocks_, pending_;
  std::unordered_map<std::string, std::vector<int64_t>> key_blocks_;
  size_t batch_hashes_ = 0;
  bool startup_ = true, rescan_ = true, repair_ = false;
  std::atomic<bool> stopping_{false};
  std::thread thread_;
};
}  // namespace mori::umbp
