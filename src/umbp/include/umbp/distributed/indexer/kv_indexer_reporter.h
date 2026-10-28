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
#include <unordered_set>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "sgl_kv_indexer.grpc.pb.h"
#include "umbp/common/config.h"

namespace mori::umbp {
namespace indexer_pb = ::kv_indexer::v1;

// Event-driven reconciliation of caller metadata with committed local DRAM.
// Backend callbacks identify changed keys; residency checks and RPCs run off-path.
class KvIndexerReporter {
 public:
  using Contains = std::function<bool(const std::string&)>;
  KvIndexerReporter(const KvIndexerConfig& config, Contains contains)
      : endpoint_(config.endpoint),
        pool_id_(config.dram_pool_id),
        window_(config.swa_window_tokens),
        components_(config.components),
        timeout_(config.timeout_ms),
        contains_(std::move(contains)) {
    const bool valid_identity = pool_id_.empty()
                                    ? !config.worker_id.empty() && !config.worker_address.empty()
                                    : config.worker_id.empty() && config.worker_address.empty();
    if (config.endpoint.empty() || !valid_identity || !(components_ & 1) || (components_ & ~7u) ||
        ((components_ & 2) && !config.swa_window_tokens) || !config.timeout_ms)
      throw std::invalid_argument("invalid KV Indexer identity, cache spec or timeout");
    stub_ = indexer_pb::KVIndexer::NewStub(
        grpc::CreateChannel(config.endpoint, grpc::InsecureChannelCredentials()));
    request_.set_worker_id(config.worker_id);
    request_.set_worker_address(config.worker_address);
    request_.set_dram_pool_id(pool_id_);
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

  // Forwarders may attach to the same pool without replacing its reporter.
  bool MatchesPool(const KvIndexerConfig& config) const {
    return !pool_id_.empty() && config.dram_pool_id == pool_id_ && config.endpoint == endpoint_ &&
           config.worker_id.empty() && config.worker_address.empty() &&
           config.components == components_ && config.swa_window_tokens == window_ &&
           std::chrono::milliseconds(config.timeout_ms) == timeout_;
  }

  // Empty means a clear or an explicit request to check all registered blocks.
  void Notify(const std::vector<std::string>& keys = {}) {
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (keys.empty()) rescan_ = true;
      for (const auto& key : keys) {
        auto it = key_blocks_.find(key);
        if (it != key_blocks_.end()) pending_.insert(it->second.begin(), it->second.end());
      }
    }
    Wake();
  }
  void Register(const KvBlockMetadata& block) {
    uint32_t mask = 0;
    for (const auto& [component, keys] : block.component_keys) {
      if ((component != 1 && component != 2 && component != 4) || keys.empty() ||
          std::any_of(keys.begin(), keys.end(), [](const auto& key) { return key.empty(); }))
        throw std::invalid_argument("invalid KV component or physical keys");
      mask |= component;
    }
    if (mask != components_ || !block.block_size)
      throw std::invalid_argument(
          "metadata must describe all configured components and token count");
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      auto old = blocks_.find(block.hash);
      if (old != blocks_.end()) {
        if (old->second->parent != block.parent || old->second->block_size != block.block_size ||
            old->second->component_keys != block.component_keys)
          throw std::invalid_argument("a block hash must keep the same KV metadata");
        return;
      }
      if (block.parent == block.hash || parents_.count(block.hash)) {
        // Forward references are allowed, but cannot close a parent cycle.
        // A new leaf with no incoming references cannot introduce a cycle.
        for (auto parent = block.parent; parent;) {
          if (*parent == block.hash) throw std::invalid_argument("KV parent cycle");
          auto ancestor = blocks_.find(*parent);
          parent = ancestor == blocks_.end() ? std::nullopt : ancestor->second->parent;
        }
      }
      blocks_[block.hash] = std::make_shared<const KvBlockMetadata>(block);
      if (block.parent) parents_.insert(*block.parent);
      for (const auto& [component, keys] : block.component_keys) {
        for (const auto& key : keys) {
          auto& hashes = key_blocks_[key];
          if (hashes.empty() || hashes.back() != block.hash) hashes.push_back(block.hash);
        }
      }
      pending_.insert(block.hash);
    }
    Wake();
  }

  bool Sync(bool rebuild = false) {
    std::lock_guard<std::mutex> sync_lock(sync_mutex_);
    if (stopping_) return false;
    std::map<int64_t, std::shared_ptr<const KvBlockMetadata>> work;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (rescan_ || rebuild) {
        work = blocks_;
        rescan_ = false;
      }
      for (int64_t hash : pending_) work[hash] = blocks_.at(hash);
      pending_.clear();
    }
    try {
      request_.clear_actions();
      batch_hashes_ = 0;
      // Only startup clears a previous owner's bucket. Periodic repair and retries
      // upsert live blocks without temporarily hiding the entire DRAM tier.
      if (startup_) {
        AddAction(indexer_pb::ACTION_CLEAR_ALL_AT_TIER);
        SendBatch();
        startup_ = false;
      }
      if (rebuild)
        for (auto& [hash, mask] : reported_) mask = 0;
      std::map<int64_t, uint32_t> desired, reports;
      std::unordered_map<int64_t, int64_t> next;
      for (const auto& [hash, block] : work) {
        if (stopping_) return false;
        uint32_t mask = 0;
        for (const auto& [component, keys] : block->component_keys)
          if (std::all_of(keys.begin(), keys.end(), contains_)) mask |= component;
        // A shared pool may supply SWA/MAMBA while the worker supplies FULL in HBM.
        if (pool_id_.empty() && !(mask & 1)) mask = 0;
        desired[hash] = mask;
        auto old = reported_.find(hash);
        if (!mask) {
          if (old != reported_.end()) AddAction(indexer_pb::ACTION_REVOKE, hash);
        } else if (rebuild || old == reported_.end() || old->second != mask) {
          reports[hash] = mask;
          if (block->parent) next[*block->parent] = hash;
        }
      }
      while (!reports.empty()) {
        int64_t hash = reports.begin()->first;
        // Find the start of an unreported chain, even for randomly ordered hashes.
        while (work.at(hash)->parent && reports.count(*work.at(hash)->parent))
          hash = *work.at(hash)->parent;
        do {
          if (stopping_) return false;
          uint32_t mask = reports.at(hash);
          reports.erase(hash);
          // An applied-but-failed RPC must remain revocable after an eviction.
          reported_[hash] = 0;
          AddAction(indexer_pb::ACTION_REPORT, hash, mask, work.at(hash).get());
          auto child = next.find(hash);
          if (child == next.end()) break;
          hash = child->second;
        } while (reports.count(hash));
      }
      SendBatch();
      for (const auto& [hash, mask] : desired) {
        if (mask)
          reported_[hash] = mask;
        else
          reported_.erase(hash);
      }
      return true;
    } catch (const std::exception& error) {
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        for (const auto& [hash, block] : work) pending_.insert(hash);
      }
      Wake();  // A failed caller-initiated flush must also wake the worker.
      MORI_UMBP_WARN("[KVIndexer] sync failed: {}", error.what());
      return false;
    }
  }
  void Stop() {
    stopping_ = true;
    Wake();
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
  void Wake() {
    {
      std::lock_guard<std::mutex> lock(wake_mutex_);
      dirty_ = true;
    }
    cv_.notify_one();
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
  const std::string endpoint_, pool_id_;
  const uint32_t window_, components_;
  const std::chrono::milliseconds timeout_;
  Contains contains_;
  // Serialize RPCs/flush/shutdown independently from write-path metadata.
  std::mutex state_mutex_, sync_mutex_, wake_mutex_;
  std::condition_variable cv_;
  std::map<int64_t, std::shared_ptr<const KvBlockMetadata>> blocks_;
  std::unordered_map<std::string, std::vector<int64_t>> key_blocks_;
  std::unordered_set<int64_t> parents_;
  std::unordered_set<int64_t> pending_;
  std::map<int64_t, uint32_t> reported_;
  size_t batch_hashes_ = 0;
  bool startup_ = true, rescan_ = true, dirty_ = true;
  std::atomic<bool> stopping_{false};
  std::thread thread_;
};
}  // namespace mori::umbp
