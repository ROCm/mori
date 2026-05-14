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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

class GlobalBlockIndex;
class ExternalKvBlockIndex;

// Master-side membership ledger + heartbeat ingestion.  In the
// master-as-advisor design this class no longer owns any allocator
// state; every per-tier capacity number it stores is the value the peer
// reported in its most recent heartbeat.  Heartbeat is also the channel
// through which peer-shipped KvEvents reach GlobalBlockIndex.
class ClientRegistry {
 public:
  explicit ClientRegistry(const ClientRegistryConfig& config);
  ClientRegistry(const ClientRegistryConfig& config, GlobalBlockIndex& index);
  ~ClientRegistry();

  ClientRegistry(const ClientRegistry&) = delete;
  ClientRegistry& operator=(const ClientRegistry&) = delete;

  void SetBlockIndex(GlobalBlockIndex* index);

  // ---- External KV block index (for unmanaged L1/L2 cache blocks) ----
  void SetExternalKvBlockIndex(ExternalKvBlockIndex* index);
  void RegisterExternalKvBlocks(const std::string& node_id, const std::vector<std::string>& hashes,
                                TierType tier);
  void UnregisterExternalKvBlocks(const std::string& node_id,
                                  const std::vector<std::string>& hashes);

  // --- Client lifecycle ---

  // Returns false when a live node with the same id already exists.
  // Returns true for new registrations or re-registration of expired
  // nodes.  In the new design the only state master holds for a node is
  // membership + last-reported tier capacities; the peer owns its own
  // allocators.
  bool RegisterClient(const std::string& node_id, const std::string& node_address,
                      const std::map<TierType, TierCapacity>& tier_capacities,
                      const std::string& peer_address = "",
                      const std::vector<uint8_t>& engine_desc_bytes = {},
                      const std::vector<std::string>& tags = {});

  // Drops the node from the registry and clears every index entry that
  // belonged to it.
  void UnregisterClient(const std::string& node_id);

  // Apply one heartbeat batch.  Returns the resulting status
  // (UNKNOWN if the node isn't registered).  On the success path:
  //   - tier_capacities replace the stored values unconditionally,
  //   - events are applied to the index (or the node's locations are
  //     replaced in full when is_full_sync=true),
  //   - last_applied_seq advances to req.seq.
  // If req.seq != last_applied_seq + 1 (and !is_full_sync), the heartbeat
  // is rejected: out_request_full_sync is set to true and out_acked_seq
  // echoes the previously applied seq so the peer reships.
  ClientStatus Heartbeat(const std::string& node_id, uint64_t seq, uint64_t last_acked_seq,
                         const std::map<TierType, TierCapacity>& tier_capacities,
                         const std::vector<KvEvent>& events, bool is_full_sync,
                         uint64_t* out_acked_seq, bool* out_request_full_sync);

  // --- Queries ---
  bool IsClientAlive(const std::string& node_id) const;
  size_t ClientCount() const;
  std::vector<ClientRecord> GetAliveClients() const;
  // Returns the tags registered for node_id, or empty if not found.
  std::vector<std::string> GetClientTags(const std::string& node_id) const;

  // --- Reaper control ---
  // The reaper only expires nodes whose last_heartbeat has aged past
  // `heartbeat_ttl × max_missed_heartbeats`.  No allocation reaper —
  // pending state lives at the peer in this design.
  void StartReaper();
  void StopReaper();

 private:
  ClientRegistryConfig config_;
  GlobalBlockIndex* index_ = nullptr;
  ExternalKvBlockIndex* external_kv_index_ = nullptr;

  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, ClientRecord> clients_;

  std::thread reaper_thread_;
  std::atomic<bool> reaper_running_{false};
  std::mutex reaper_cv_mutex_;
  std::condition_variable reaper_cv_;

  void ReaperLoop();
  void ReapExpiredClients();

  std::chrono::seconds ExpiryDuration() const {
    return config_.heartbeat_ttl * config_.max_missed_heartbeats;
  }
};

}  // namespace mori::umbp
