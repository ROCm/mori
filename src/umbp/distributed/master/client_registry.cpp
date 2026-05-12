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
#include "umbp/distributed/master/client_registry.h"

#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/master/external_kv_block_index.h"
#include "umbp/distributed/master/global_block_index.h"

namespace mori::umbp {

ClientRegistry::ClientRegistry(const ClientRegistryConfig& config) : config_(config) {}

ClientRegistry::ClientRegistry(const ClientRegistryConfig& config, GlobalBlockIndex& index)
    : config_(config), index_(&index) {}

ClientRegistry::~ClientRegistry() { StopReaper(); }

void ClientRegistry::SetBlockIndex(GlobalBlockIndex* index) {
  std::unique_lock lock(mutex_);
  index_ = index;
}

void ClientRegistry::SetExternalKvBlockIndex(ExternalKvBlockIndex* index) {
  std::unique_lock lock(mutex_);
  external_kv_index_ = index;
}

void ClientRegistry::RegisterExternalKvBlocks(const std::string& node_id,
                                              const std::vector<std::string>& hashes,
                                              TierType tier) {
  if (!IsClientAlive(node_id)) {
    MORI_UMBP_WARN("[Registry] RegisterExternalKvBlocks rejected: node not alive: {}", node_id);
    return;
  }
  if (external_kv_index_ != nullptr) {
    external_kv_index_->Register(node_id, hashes, tier);
  }
}

void ClientRegistry::UnregisterExternalKvBlocks(const std::string& node_id,
                                                const std::vector<std::string>& hashes,
                                                TierType tier) {
  if (external_kv_index_ != nullptr) {
    external_kv_index_->Unregister(node_id, hashes, tier);
  }
}

bool ClientRegistry::RegisterClient(const std::string& node_id, const std::string& node_address,
                                    const std::map<TierType, TierCapacity>& tier_capacities,
                                    const std::string& peer_address,
                                    const std::vector<uint8_t>& engine_desc_bytes) {
  std::unique_lock lock(mutex_);
  const auto now = std::chrono::steady_clock::now();

  auto it = clients_.find(node_id);
  if (it != clients_.end()) {
    const bool is_expired = (now - it->second.last_heartbeat > ExpiryDuration()) ||
                            (it->second.status == ClientStatus::EXPIRED);
    if (it->second.status == ClientStatus::ALIVE && !is_expired) {
      MORI_UMBP_WARN("[Registry] Rejecting re-registration for alive node: {}", node_id);
      return false;
    }
    MORI_UMBP_INFO("[Registry] Re-registering expired node: {}", node_id);
  }

  ClientRecord record;
  record.node_id = node_id;
  record.node_address = node_address;
  record.status = ClientStatus::ALIVE;
  record.last_heartbeat = now;
  record.registered_at = now;
  record.tier_capacities = tier_capacities;
  record.peer_address = peer_address;
  record.engine_desc_bytes = engine_desc_bytes;
  record.last_applied_seq = 0;

  clients_[node_id] = std::move(record);

  MORI_UMBP_INFO("[Registry] Registered node: {} at {} (peer={})", node_id, node_address,
                 peer_address);
  return true;
}

void ClientRegistry::UnregisterClient(const std::string& node_id) {
  GlobalBlockIndex* idx = nullptr;
  {
    std::unique_lock lock(mutex_);
    auto it = clients_.find(node_id);
    if (it == clients_.end()) return;
    idx = index_;
    clients_.erase(it);
  }
  if (idx != nullptr) {
    // Empty replacement clears every index location belonging to node_id.
    idx->ReplaceNodeLocations(node_id, {});
  }
  if (external_kv_index_ != nullptr) {
    external_kv_index_->UnregisterByNode(node_id);
  }
  MORI_UMBP_INFO("[Registry] Unregistered node: {}", node_id);
}

ClientStatus ClientRegistry::Heartbeat(const std::string& node_id, uint64_t seq,
                                       uint64_t /*last_acked_seq*/,
                                       const std::map<TierType, TierCapacity>& tier_capacities,
                                       const std::vector<KvEvent>& events, bool is_full_sync,
                                       uint64_t* out_acked_seq, bool* out_request_full_sync) {
  if (out_acked_seq != nullptr) *out_acked_seq = 0;
  if (out_request_full_sync != nullptr) *out_request_full_sync = false;

  GlobalBlockIndex* idx = nullptr;
  std::vector<KvEvent> events_to_apply;
  bool full_sync_apply = false;

  {
    std::unique_lock lock(mutex_);
    auto it = clients_.find(node_id);
    if (it == clients_.end()) {
      MORI_UMBP_WARN("[Registry] Heartbeat from unknown node: {}", node_id);
      return ClientStatus::UNKNOWN;
    }
    auto& record = it->second;

    // Gap detection: every heartbeat batch carries seq = previous + 1.
    // is_full_sync overrides the gap check — it's the recovery channel.
    if (!is_full_sync && seq != record.last_applied_seq + 1) {
      MORI_UMBP_WARN(
          "[Registry] Heartbeat seq gap from {}: got {}, expected {} — requesting full sync",
          node_id, seq, record.last_applied_seq + 1);
      if (out_acked_seq != nullptr) *out_acked_seq = record.last_applied_seq;
      if (out_request_full_sync != nullptr) *out_request_full_sync = true;
      record.last_heartbeat = std::chrono::steady_clock::now();
      record.status = ClientStatus::ALIVE;
      return ClientStatus::ALIVE;
    }

    record.last_heartbeat = std::chrono::steady_clock::now();
    record.status = ClientStatus::ALIVE;
    record.tier_capacities = tier_capacities;
    record.last_applied_seq = seq;

    idx = index_;
    events_to_apply = events;
    full_sync_apply = is_full_sync;
    if (out_acked_seq != nullptr) *out_acked_seq = seq;
  }

  if (idx != nullptr) {
    if (full_sync_apply) {
      idx->ReplaceNodeLocations(node_id, events_to_apply);
    } else if (!events_to_apply.empty()) {
      idx->ApplyEvents(node_id, events_to_apply);
    }
  }
  return ClientStatus::ALIVE;
}

bool ClientRegistry::IsClientAlive(const std::string& node_id) const {
  std::shared_lock lock(mutex_);
  auto it = clients_.find(node_id);
  return it != clients_.end() && it->second.status == ClientStatus::ALIVE;
}

size_t ClientRegistry::ClientCount() const {
  std::shared_lock lock(mutex_);
  return clients_.size();
}

std::vector<ClientRecord> ClientRegistry::GetAliveClients() const {
  std::shared_lock lock(mutex_);
  std::vector<ClientRecord> result;
  for (const auto& [id, record] : clients_) {
    if (record.status == ClientStatus::ALIVE) result.push_back(record);
  }
  return result;
}

void ClientRegistry::StartReaper() {
  reaper_running_ = true;
  reaper_thread_ = std::thread(&ClientRegistry::ReaperLoop, this);
  MORI_UMBP_INFO("[Reaper] Started (interval={}s, expiry={}s)", config_.reaper_interval.count(),
                 ExpiryDuration().count());
}

void ClientRegistry::StopReaper() {
  if (reaper_running_) {
    reaper_running_ = false;
    reaper_cv_.notify_one();
    if (reaper_thread_.joinable()) reaper_thread_.join();
    MORI_UMBP_INFO("[Reaper] Stopped");
  }
}

void ClientRegistry::ReaperLoop() {
  while (reaper_running_) {
    {
      std::unique_lock cv_lock(reaper_cv_mutex_);
      reaper_cv_.wait_for(cv_lock, config_.reaper_interval,
                          [this] { return !reaper_running_.load(); });
    }
    if (!reaper_running_) break;
    ReapExpiredClients();
  }
}

void ClientRegistry::ReapExpiredClients() {
  const auto now = std::chrono::steady_clock::now();
  const auto expiry = ExpiryDuration();
  std::vector<std::string> dead_nodes;

  {
    std::unique_lock lock(mutex_);
    auto it = clients_.begin();
    while (it != clients_.end()) {
      if (now - it->second.last_heartbeat > expiry) {
        MORI_UMBP_WARN("[Reaper] Reaping expired client: {}", it->first);
        dead_nodes.push_back(it->first);
        it = clients_.erase(it);
      } else {
        ++it;
      }
    }
  }

  if (index_ != nullptr) {
    for (const auto& dead_id : dead_nodes) {
      // Clear every index entry belonging to the dead node.  Capacity
      // numbers vanish with the ClientRecord above.
      index_->ReplaceNodeLocations(dead_id, {});
    }
  }
  if (external_kv_index_ != nullptr) {
    for (const auto& dead_id : dead_nodes) {
      external_kv_index_->UnregisterByNode(dead_id);
    }
  }
}

}  // namespace mori::umbp
