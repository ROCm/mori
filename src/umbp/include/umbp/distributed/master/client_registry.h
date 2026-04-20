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
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

class GlobalBlockIndex;

// Result of ClientRegistry::AllocateForPut.
//
// DRAM/HBM tier (page-bitmap allocator):
//   - `location_id`        : canonical "0:p3,4;1:p0" string covering every
//                            page reserved for this Put.
//   - `pages`              : structured form of the same page set, used by
//                            the Client to build scatter-gather RDMA descs.
//   - `dram_memory_descs`  : MemoryDesc bytes for every distinct buffer_index
//                            referenced by `pages`, deduplicated, ascending.
//   - `page_size`          : page size of the source allocator (bytes), so
//                            the Client can compute `page_index * page_size`.
//
// SSD tier (capacity-only PoolAllocator):
//   - `location_id`        : empty (CommitSsdWrite generates the real one).
//   - `pages`              : empty.
//   - `dram_memory_descs`  : empty.
//   - `page_size`          : 0.
//   - `ssd_store_index`    : which SSD store (PoolAllocator) was reserved.
//
// All tiers:
//   - `allocation_id`      : Master-issued lease id, used by Finalize/Abort.
//   - `peer_address` /
//     `engine_desc_bytes`  : routing metadata copied from the ClientRecord.
struct AllocateResult {
  std::string allocation_id;
  std::string peer_address;
  std::vector<uint8_t> engine_desc_bytes;

  // DRAM/HBM only:
  std::string location_id;
  std::vector<PageLocation> pages;
  std::vector<BufferMemoryDescBytes> dram_memory_descs;
  uint64_t page_size = 0;

  // SSD only:
  uint32_t ssd_store_index = 0;
};

struct ClientIOInfo {
  std::string peer_address;
  std::vector<uint8_t> engine_desc_bytes;
  std::vector<uint8_t> dram_memory_desc_bytes;
};

struct FinalizedRecord {
  std::string key;
  Location location;
  std::chrono::steady_clock::time_point finalized_at;
};

class ClientRegistry {
 public:
  explicit ClientRegistry(const ClientRegistryConfig& config);
  ClientRegistry(const ClientRegistryConfig& config, GlobalBlockIndex& index);
  ~ClientRegistry();

  ClientRegistry(const ClientRegistry&) = delete;
  ClientRegistry& operator=(const ClientRegistry&) = delete;

  void SetBlockIndex(GlobalBlockIndex* index);

  // --- Client lifecycle ---
  // Returns false when a live node with the same id already exists.
  // Returns true for new registrations or re-registration of expired nodes.
  //
  // `dram_page_size` is the page_size to use when constructing this node's
  // DRAM/HBM PageBitmapAllocator(s).  Pass 0 to fall back to
  // `config_.default_dram_page_size` (the registry-wide default).  The same
  // value applies to both DRAM and HBM tiers.
  bool RegisterClient(const std::string& node_id, const std::string& node_address,
                      const std::map<TierType, TierCapacity>& tier_capacities,
                      const std::string& peer_address = "",
                      const std::vector<uint8_t>& engine_desc_bytes = {},
                      const std::vector<std::vector<uint8_t>>& dram_memory_desc_bytes_list = {},
                      const std::vector<uint64_t>& dram_buffer_sizes = {},
                      const std::vector<uint64_t>& ssd_store_capacities = {},
                      uint64_t dram_page_size = 0);

  // Gracefully unregister. Returns number of block keys cleaned up.
  size_t UnregisterClient(const std::string& node_id);

  // Process heartbeat. Updates last_heartbeat and tier capacities.
  // Returns CLIENT_STATUS_UNKNOWN if node is not registered.
  // PA-3 fix: uses exclusive lock since it mutates record fields.
  ClientStatus Heartbeat(const std::string& node_id,
                         const std::map<TierType, TierCapacity>& tier_capacities);

  // --- Ownership tracking (called by BlockIndex) ---
  void TrackKey(const std::string& node_id, const std::string& key);
  void UntrackKey(const std::string& node_id, const std::string& key);

  // --- PoolClient allocation ---
  std::optional<AllocateResult> AllocateForPut(const std::string& node_id, TierType tier,
                                               uint64_t size);

  // Free a previously registered location.  For DRAM/HBM tier the
  // `location_id` is parsed via ParseDramLocationId and the corresponding
  // page slots are flipped back to free in the per-tier PageBitmapAllocator.
  // For SSD tier the capacity-only PoolAllocator is decremented.  DRAM/HBM
  // size is implied by `pages.size() * page_size`; SSD size comes from the
  // caller via `location.size`.
  void DeallocateForUnregister(const std::string& node_id, const Location& location);
  bool FinalizeAllocation(const std::string& node_id, const std::string& key,
                          const Location& location, const std::string& allocation_id);
  bool PublishLocalBlock(const std::string& node_id, const std::string& key,
                         const Location& location);
  bool AbortAllocation(const std::string& node_id, const std::string& allocation_id, uint64_t size);
  std::optional<ClientIOInfo> GetClientIOInfo(const std::string& node_id,
                                              uint32_t buffer_index = 0) const;

  // Collect the deduplicated, ascending-by-buffer_index list of MemoryDesc
  // bytes for every distinct buffer_index appearing in `pages`.  Used by
  // master_server's RouteGet handler (and indirectly by AllocateForPut) to
  // populate `RoutePutResponse.dram_memory_descs` / `RouteGetResponse
  // .dram_memory_descs`.  Returns nullopt if `node_id` is unknown / not
  // ALIVE.  Pages that reference a buffer_index outside the registered
  // descriptor list are silently skipped — the caller is expected to have
  // produced `pages` from the same ClientRegistry it queries here, so this
  // path means a Master-side bug that we do not want to crash on in
  // release builds (debug builds get a DCHECK in the implementation, per
  // §11 risk table).
  std::optional<std::vector<BufferMemoryDescBytes>> GetDramMemoryDescsForPages(
      const std::string& node_id, const std::vector<PageLocation>& pages) const;

  // Read the page_size of the given (node, tier)'s PageBitmapAllocator.
  // Returns std::nullopt when `node_id` is unknown / not ALIVE, when no
  // allocator exists for the tier, or when `tier` is not DRAM/HBM.  Used by
  // master_server's RouteGet handlers so the response carries the source
  // node's actual page_size (which may differ from the registry-wide default
  // when the Client overrode `dram_page_size` at registration time).
  std::optional<uint64_t> GetNodeDramPageSize(const std::string& node_id, TierType tier) const;

  // --- Queries ---
  bool IsClientAlive(const std::string& node_id) const;
  size_t ClientCount() const;

  // Returns all clients with status == ALIVE. Used by Router for RoutePut.
  std::vector<ClientRecord> GetAliveClients() const;

  // --- Reaper control ---
  void StartReaper();
  void StopReaper();

 private:
  ClientRegistryConfig config_;
  GlobalBlockIndex* index_ = nullptr;

  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, ClientRecord> clients_;
  std::unordered_map<std::string, std::set<std::string>> client_keys_;
  std::unordered_map<std::string, PendingAllocation> pending_allocations_;
  std::unordered_map<std::string, FinalizedRecord> finalized_allocations_;

  // Reaper thread
  std::thread reaper_thread_;
  std::atomic<bool> reaper_running_{false};
  std::mutex reaper_cv_mutex_;
  std::condition_variable reaper_cv_;
  std::atomic<uint64_t> next_allocation_id_{1};

  void ReaperLoop();
  // PA-4 fix: uses iterator-safe erase pattern.
  void ReapExpiredClients();
  void ReapExpiredPendingAllocations();
  void ReapExpiredFinalizedRecords();
  void ReleasePendingAllocationsForNodeLocked(const std::string& node_id);

  // Helper used by AbortAllocation / Reaper / ReleasePendingAllocationsForNode
  // to turn a single PendingAllocation back into free capacity.  Caller MUST
  // hold the unique ClientRegistry mutex.  No-op if the owning ClientRecord
  // (or its tier-specific allocator) is no longer present — this can happen
  // after an EXPIRED-then-re-registered race and is not an error.
  void DeallocatePendingLocked(const PendingAllocation& pending);

  // Legacy SSD-only helper used by PublishLocalBlock parsing.
  static uint32_t ParseBufferIndex(const std::string& location_id);
  void UpdateAvailableBytesLocked(ClientRecord& record, TierType tier);

  std::chrono::seconds ExpiryDuration() const {
    return config_.heartbeat_ttl * config_.max_missed_heartbeats;
  }
};

}  // namespace mori::umbp
