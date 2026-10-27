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
#include "umbp/distributed/indexer/kv_indexer_exporter.h"

#include <google/protobuf/util/json_util.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {
namespace {
namespace pb = ::kv_indexer::v1;
constexpr uint32_t kFull = 1;
constexpr uint32_t kSwa = 2;
constexpr uint32_t kMamba = 4;
constexpr uint32_t kDramMask = 1u << pb::TIER_DRAM;
constexpr size_t kMaxActions = 256;

void Require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

pb::ExternalKvAction Revoke(int64_t hash) {
  pb::ExternalKvAction action;
  action.set_type(pb::ACTION_REVOKE);
  action.set_tier(pb::TIER_DRAM);
  action.add_hashes(hash);
  return action;
}

pb::ExternalKvAction Report(const indexer::BlockMapping& block, uint32_t mask) {
  pb::ExternalKvAction action;
  action.set_type(pb::ACTION_REPORT);
  action.set_tier(pb::TIER_DRAM);
  action.add_hashes(block.hash());
  action.add_component_masks(mask);
  action.add_block_sizes(block.block_size());
  if (block.has_parent_hash()) action.set_parent_block_hash(block.parent_hash());
  return action;
}

}  // namespace

KvIndexerExporter::Config KvIndexerExporter::LoadConfig(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) throw std::invalid_argument("cannot read KV Indexer exporter config: " + path);
  std::string json{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
  Config config;
  auto status = google::protobuf::util::JsonStringToMessage(json, &config);
  if (!status.ok()) throw std::invalid_argument("invalid KV Indexer config: " + status.ToString());
  Validate(config);
  return config;
}

void KvIndexerExporter::Validate(const Config& config) {
  Require(!config.endpoint().empty(), "Indexer endpoint is required");
  Require(config.endpoint().rfind("http://", 0) != 0 && config.endpoint().rfind("https://", 0) != 0,
          "Indexer endpoint must be a gRPC target, not an HTTP URL");
  Require(!config.worker_id().empty(), "Indexer worker_id is required");
  Require(!config.worker_address().empty(), "Indexer router-facing worker_address is required");
  const auto& spec = config.cache_spec();
  Require(spec.version() <= 1, "unsupported Indexer cache spec version");
  Require((spec.components() & kFull) && !(spec.components() & ~7u),
          "cache spec must contain FULL and only known components");
  Require(spec.full_tier_mask() & kDramMask, "cache spec FULL must allow DRAM");
  if (spec.components() & kSwa) {
    Require(spec.swa_window_tokens() > 0 && (spec.swa_tier_mask() & kDramMask),
            "SWA requires a positive window and DRAM tier");
  }
  if (spec.components() & kMamba) {
    Require(spec.mamba_tier_mask() & kDramMask, "MAMBA must allow DRAM");
  }
  Require(config.rpc_timeout_ms() <= 60000, "rpc_timeout_ms must not exceed 60000");
  std::unordered_map<int64_t, const indexer::BlockMapping*> blocks;
  for (const auto& block : config.blocks()) {
    Require(blocks.emplace(block.hash(), &block).second, "duplicate logical block hash");
    Require(block.block_size() > 0, "block_size must be positive");
    uint32_t mask = 0;
    for (const auto& component : block.components()) {
      const auto bit = component.component();
      Require(bit == kFull || bit == kSwa || bit == kMamba, "invalid component bit");
      Require(!(mask & bit) && (spec.components() & bit), "duplicate or undeclared component");
      mask |= bit;
      Require(!component.keys().empty(), "component needs at least one physical object key");
      std::unordered_set<std::string> keys;
      for (const auto& key : component.keys()) {
        Require(!key.empty() && keys.insert(key).second, "empty or duplicate object key");
      }
    }
    Require(mask == spec.components(), "mapping must describe every declared component");
  }
  // Validate the complete logical ancestry, even when some physical ancestors
  // have already been evicted. This also prevents accidentally declaring every
  // storage object to be an unrelated root.
  std::unordered_set<int64_t> finished;
  for (const auto& block : config.blocks()) {
    const auto* current = &block;
    std::unordered_set<int64_t> path;
    while (!finished.count(current->hash())) {
      Require(path.insert(current->hash()).second, "cyclic block ancestry");
      if (!current->has_parent_hash()) break;
      const auto parent = blocks.find(current->parent_hash());
      Require(parent != blocks.end(), "parent hash missing from logical manifest");
      current = parent->second;
    }
    finished.insert(path.begin(), path.end());
  }
}

KvIndexerExporter::KvIndexerExporter(Config config, SnapshotSource source)
    : config_(std::move(config)), source_(std::move(source)) {
  Validate(config_);
  for (const auto& block : config_.blocks()) {
    identities_.emplace(block.hash(),
                        Identity{block.has_parent_hash(), block.parent_hash(), block.block_size()});
  }
  Require(static_cast<bool>(source_), "committed-key snapshot source is required");
  stub_ = pb::KVIndexer::NewStub(
      grpc::CreateChannel(config_.endpoint(), grpc::InsecureChannelCredentials()));
}

KvIndexerExporter::~KvIndexerExporter() { Stop(); }

grpc::Status KvIndexerExporter::Apply(const std::vector<pb::ExternalKvAction>& actions) {
  pb::ApplyExternalKvBatchRequest request;
  request.set_worker_id(config_.worker_id());
  request.set_worker_address(config_.worker_address());
  request.mutable_cache_spec()->CopyFrom(config_.cache_spec());
  request.set_seq(++seq_);  // Observability only; the receiver does not fence seq.
  for (const auto& action : actions) request.add_actions()->CopyFrom(action);
  grpc::ClientContext context;
  const auto timeout = config_.rpc_timeout_ms() ? config_.rpc_timeout_ms() : 200;
  context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(timeout));
  pb::ApplyExternalKvBatchResponse response;
  contacted_ = true;  // Even an error can follow a server-side commit.
  auto status = stub_->ApplyExternalKvBatch(&context, request, &response);
  if (!status.ok()) needs_full_sync_ = true;
  return status;
}

grpc::Status KvIndexerExporter::Clear() {
  pb::ExternalKvAction action;
  action.set_type(pb::ACTION_CLEAR_ALL_AT_TIER);
  action.set_tier(pb::TIER_DRAM);
  auto status = Apply({action});
  if (status.ok()) applied_.clear();
  return status;
}

grpc::Status KvIndexerExporter::SyncOnce(bool force_full_sync) {
  std::lock_guard<std::mutex> lock(sync_mutex_);
  Placements desired;
  try {
    if (!config_path_.empty()) {
      auto fresh = LoadConfig(config_path_);
      Config old_identity = config_, new_identity = fresh;
      old_identity.clear_blocks();
      new_identity.clear_blocks();
      Require(old_identity.SerializeAsString() == new_identity.SerializeAsString(),
              "only block mappings can change without restarting the exporter");
      for (const auto& block : fresh.blocks()) {
        const auto old = identities_.find(block.hash());
        if (old == identities_.end()) continue;
        Require(old->second.has_parent == block.has_parent_hash() &&
                    old->second.parent == block.parent_hash() &&
                    old->second.block_size == block.block_size(),
                "a logical hash must retain its canonical parent and block size");
      }
      // Mappings may change residency requirements, but never logical identity.
      if (config_.SerializeAsString() != fresh.SerializeAsString()) needs_full_sync_ = true;
      for (const auto& block : fresh.blocks()) {
        identities_.emplace(block.hash(), Identity{block.has_parent_hash(), block.parent_hash(),
                                                   block.block_size()});
      }
      config_ = std::move(fresh);
    }
    std::unordered_set<std::string> resident;
    for (const auto& event : source_()) {
      if (event.kind == KvEvent::Kind::ADD && event.tier == TierType::DRAM)
        resident.insert(event.key);
    }
    for (const auto& block : config_.blocks()) {
      uint32_t mask = 0;
      for (const auto& component : block.components()) {
        if (std::all_of(component.keys().begin(), component.keys().end(),
                        [&](const auto& key) { return resident.count(key) != 0; }))
          mask |= component.component();
      }
      // A partial SWA/MAMBA-only page cannot extend a reusable FULL prefix.
      if (mask & kFull) desired.emplace(block.hash(), Placement{block, mask});
    }
  } catch (const std::exception& error) {
    needs_full_sync_ = true;
    // Invalid manifests/snapshot failures must not keep advertising old state.
    auto status = Clear();
    if (!status.ok()) return status;
    return {grpc::StatusCode::INVALID_ARGUMENT, error.what()};
  }
  const auto now = std::chrono::steady_clock::now();
  const auto interval = config_.full_sync_interval_ms() ? config_.full_sync_interval_ms() : 30000;
  const bool full = force_full_sync || needs_full_sync_ ||
                    now - last_full_sync_ >= std::chrono::milliseconds(interval);
  if (full) {
    auto status = Clear();
    if (!status.ok()) return status;
  }
  std::vector<pb::ExternalKvAction> actions;
  for (const auto& [hash, placement] : applied_) {
    if (!desired.count(hash)) actions.push_back(Revoke(hash));
  }
  for (const auto& [hash, placement] : desired) {
    auto old = applied_.find(hash);
    if (old == applied_.end() || old->second.mask != placement.mask)
      actions.push_back(Report(placement.block, placement.mask));
  }
  for (size_t offset = 0; offset < actions.size(); offset += kMaxActions) {
    if (stopping_.load()) {
      needs_full_sync_ = true;
      return {grpc::StatusCode::CANCELLED, "exporter is stopping"};
    }
    const size_t end = std::min(offset + kMaxActions, actions.size());
    auto status =
        Apply(std::vector<pb::ExternalKvAction>(actions.begin() + offset, actions.begin() + end));
    if (!status.ok()) return status;
  }
  applied_ = std::move(desired);
  needs_full_sync_ = false;
  if (full) last_full_sync_ = now;
  return grpc::Status::OK;
}

void KvIndexerExporter::Start(std::string config_path) {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  if (thread_.joinable()) throw std::logic_error("KV Indexer exporter already started");
  std::lock_guard<std::mutex> sync_lock(sync_mutex_);
  config_path_ = std::move(config_path);
  stopping_ = false;
  thread_ = std::thread(&KvIndexerExporter::Loop, this);
}

void KvIndexerExporter::Loop() {
  uint32_t poll;
  {
    std::lock_guard<std::mutex> lock(sync_mutex_);
    poll = config_.poll_interval_ms() ? config_.poll_interval_ms() : 1000;
  }
  while (!stopping_.load()) {
    auto status = SyncOnce();
    if (!status.ok()) MORI_UMBP_WARN("[KVIndexer] sync failed: {}", status.error_message());
    std::unique_lock<std::mutex> lock(lifecycle_mutex_);
    if (cv_.wait_for(lock, std::chrono::milliseconds(poll), [this] { return stopping_.load(); }))
      break;
  }
}

grpc::Status KvIndexerExporter::Stop() {
  {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    stopping_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
  std::lock_guard<std::mutex> lock(sync_mutex_);
  if (!contacted_) return grpc::Status::OK;
  auto status = Clear();
  if (!status.ok())
    MORI_UMBP_WARN("[KVIndexer] DRAM withdrawal failed: {}", status.error_message());
  contacted_ = false;
  needs_full_sync_ = true;
  return status;
}

}  // namespace mori::umbp
