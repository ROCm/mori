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

ClientRegistry::ClientRegistry(const ClientRegistryConfig& config, GlobalBlockIndex& index,
                               ExternalKvBlockIndex* external_kv_index)
    : config_(config), index_(&index), external_kv_index_(external_kv_index) {}

ClientRegistry::~ClientRegistry() { StopReaper(); }

void ClientRegistry::SetBlockIndex(GlobalBlockIndex* index) {
  std::unique_lock lock(mutex_);
  index_ = index;
}

void ClientRegistry::SetExternalKvBlockIndex(ExternalKvBlockIndex* index) {
  std::unique_lock lock(mutex_);
  external_kv_index_ = index;
}

bool ClientRegistry::RegisterClient(const std::string& node_id, const std::string& node_address,
                                    const std::map<TierType, TierCapacity>& tier_capacities,
                                    const std::string& peer_address,
                                    const std::vector<uint8_t>& engine_desc_bytes,
                                    const std::vector<std::string>& tags) {
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
  record.tags = tags;

  clients_[node_id] = std::move(record);

  std::string tags_str;
  for (const auto& t : tags) {
    if (!tags_str.empty()) tags_str += ',';
    tags_str += t;
  }
  MORI_UMBP_INFO("[Registry] Registered node: {} at {} (peer={}) tags=[{}]", node_id, node_address,
                 peer_address, tags_str);
  return true;
}

void ClientRegistry::UnregisterClient(const std::string& node_id) {
  GlobalBlockIndex* idx = nullptr;
  ExternalKvBlockIndex* external_idx = nullptr;
  {
    std::unique_lock lock(mutex_);
    auto it = clients_.find(node_id);
    if (it == clients_.end()) return;
    idx = index_;
    external_idx = external_kv_index_;
    clients_.erase(it);
  }
  if (idx != nullptr) {
    idx->RemoveByNode(node_id);
  }
  if (external_idx != nullptr) {
    external_idx->UnregisterByNode(node_id);
  }
  MORI_UMBP_INFO("[Registry] Unregistered node: {}", node_id);
}

ClientStatus ClientRegistry::Heartbeat(const std::string& node_id,
                                       const std::map<TierType, TierCapacity>& tier_capacities,
                                       const std::vector<EventBundle>& bundles, bool is_full_sync,
                                       uint64_t delta_seq_baseline, uint64_t* out_acked_seq,
                                       bool* out_request_full_sync) {
  if (out_acked_seq != nullptr) *out_acked_seq = 0;
  if (out_request_full_sync != nullptr) *out_request_full_sync = false;

  GlobalBlockIndex* idx = nullptr;
  std::vector<EventBundle> bundles_to_apply;
  std::vector<KvEvent> full_sync_adds;
  bool do_full_sync = false;

  {
    std::unique_lock lock(mutex_);
    auto it = clients_.find(node_id);
    if (it == clients_.end()) {
      MORI_UMBP_WARN("[Registry] Heartbeat from unknown node: {}", node_id);
      return ClientStatus::UNKNOWN;
    }
    auto& record = it->second;

    record.last_heartbeat = std::chrono::steady_clock::now();
    record.status = ClientStatus::ALIVE;
    record.tier_capacities = tier_capacities;

    idx = index_;

    if (is_full_sync) {
      for (const auto& bundle : bundles) {
        for (auto ev : bundle.events) {
          if (ev.kind != KvEvent::Kind::ADD) continue;
          full_sync_adds.push_back(std::move(ev));
        }
      }
      record.last_applied_seq = delta_seq_baseline;
      if (out_acked_seq != nullptr) *out_acked_seq = record.last_applied_seq;
      do_full_sync = true;
    } else {
      for (const auto& bundle : bundles) {
        if (bundle.seq <= record.last_applied_seq) continue;
        if (bundle.seq != record.last_applied_seq + 1) {
          MORI_UMBP_WARN(
              "[Registry] Heartbeat bundle seq gap from {}: got {}, expected {} — requesting "
              "full sync",
              node_id, bundle.seq, record.last_applied_seq + 1);
          if (out_acked_seq != nullptr) *out_acked_seq = record.last_applied_seq;
          if (out_request_full_sync != nullptr) *out_request_full_sync = true;
          return ClientStatus::ALIVE;
        }
        bundles_to_apply.push_back(bundle);
        record.last_applied_seq = bundle.seq;
      }
      if (out_acked_seq != nullptr) *out_acked_seq = record.last_applied_seq;
    }
  }

  if (idx != nullptr) {
    if (do_full_sync) {
      idx->ReplaceNodeLocations(node_id, full_sync_adds);
    } else {
      for (const auto& bundle : bundles_to_apply) {
        if (!bundle.events.empty()) idx->ApplyEvents(node_id, bundle.events);
      }
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

std::vector<std::string> ClientRegistry::GetClientTags(const std::string& node_id) const {
  std::shared_lock lock(mutex_);
  auto it = clients_.find(node_id);
  if (it == clients_.end()) return {};
  return it->second.tags;
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

  GlobalBlockIndex* idx = nullptr;
  ExternalKvBlockIndex* external_idx = nullptr;
  {
    std::shared_lock lock(mutex_);
    idx = index_;
    external_idx = external_kv_index_;
  }

  if (idx != nullptr) {
    for (const auto& dead_id : dead_nodes) {
      // Clear every index entry belonging to the dead node.  Capacity
      // numbers vanish with the ClientRecord above.
      idx->RemoveByNode(dead_id);
    }
  }
  if (external_idx != nullptr) {
    for (const auto& dead_id : dead_nodes) {
      external_idx->UnregisterByNode(dead_id);
    }
  }
}

}  // namespace mori::umbp
