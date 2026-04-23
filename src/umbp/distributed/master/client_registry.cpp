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

#include <algorithm>
#include <cassert>
#include <set>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/master/global_block_index.h"
#include "umbp/distributed/page_bitmap_allocator.h"

namespace mori::umbp {

namespace {

constexpr uint64_t kBytesPerMB = 1024ULL * 1024;

// Round-up integer division: how many `page_size` pages cover `size` bytes.
uint32_t PagesForSize(uint64_t size, uint64_t page_size) {
  if (page_size == 0) return 0;
  return static_cast<uint32_t>((size + page_size - 1) / page_size);
}

bool IsDramOrHbm(TierType t) { return t == TierType::DRAM || t == TierType::HBM; }

}  // namespace

ClientRegistry::ClientRegistry(const ClientRegistryConfig& config) : config_(config) {}

ClientRegistry::ClientRegistry(const ClientRegistryConfig& config, GlobalBlockIndex& index)
    : config_(config), index_(&index) {}

ClientRegistry::~ClientRegistry() { StopReaper(); }

void ClientRegistry::SetBlockIndex(GlobalBlockIndex* index) {
  std::unique_lock lock(mutex_);
  index_ = index;
}

uint32_t ClientRegistry::ParseBufferIndex(const std::string& location_id) {
  auto colon = location_id.find(':');
  if (colon == std::string::npos) {
    return 0;
  }
  try {
    return static_cast<uint32_t>(std::stoul(location_id.substr(0, colon)));
  } catch (...) {
    return 0;
  }
}

void ClientRegistry::UpdateAvailableBytesLocked(ClientRecord& record, TierType tier) {
  uint64_t total_avail = 0;
  if (IsDramOrHbm(tier)) {
    auto it = record.page_allocators.find(tier);
    if (it != record.page_allocators.end() && it->second) {
      total_avail = it->second->AvailableBytes();
    } else {
      // No allocator for this tier (e.g. it was never registered) — treat as
      // zero available; do NOT touch tier_capacities[tier] in that case.
      return;
    }
  } else if (tier == TierType::SSD) {
    for (auto& alloc : record.ssd_allocators) {
      total_avail += alloc.AvailableBytes();
    }
  }
  record.tier_capacities[tier].available_bytes = total_avail;
}

void ClientRegistry::DeallocatePendingLocked(const PendingAllocation& pending) {
  auto client_it = clients_.find(pending.node_id);
  if (client_it == clients_.end()) return;
  auto& record = client_it->second;

  if (IsDramOrHbm(pending.tier)) {
    auto alloc_it = record.page_allocators.find(pending.tier);
    if (alloc_it != record.page_allocators.end() && alloc_it->second) {
      alloc_it->second->Deallocate(pending.pages);
      UpdateAvailableBytesLocked(record, pending.tier);
    }
  } else if (pending.tier == TierType::SSD) {
    if (pending.ssd_store_index < record.ssd_allocators.size()) {
      record.ssd_allocators[pending.ssd_store_index].Deallocate(0, pending.size);
      UpdateAvailableBytesLocked(record, TierType::SSD);
    }
  }
}

void ClientRegistry::ReleasePendingAllocationsForNodeLocked(const std::string& node_id) {
  auto it = pending_allocations_.begin();
  while (it != pending_allocations_.end()) {
    if (it->second.node_id != node_id) {
      ++it;
      continue;
    }
    DeallocatePendingLocked(it->second);
    it = pending_allocations_.erase(it);
  }
}

bool ClientRegistry::RegisterClient(
    const std::string& node_id, const std::string& node_address,
    const std::map<TierType, TierCapacity>& tier_capacities, const std::string& peer_address,
    const std::vector<uint8_t>& engine_desc_bytes,
    const std::vector<std::vector<uint8_t>>& dram_memory_desc_bytes_list,
    const std::vector<uint64_t>& dram_buffer_sizes,
    const std::vector<uint64_t>& ssd_store_capacities, uint64_t dram_page_size) {
  std::unique_lock lock(mutex_);
  auto now = std::chrono::steady_clock::now();

  auto it = clients_.find(node_id);
  if (it != clients_.end()) {
    const bool is_expired = (now - it->second.last_heartbeat > ExpiryDuration()) ||
                            (it->second.status == ClientStatus::EXPIRED);

    if (it->second.status == ClientStatus::ALIVE && !is_expired) {
      MORI_UMBP_WARN("[Registry] Rejecting re-registration for alive node: {}", node_id);
      return false;
    }

    ReleasePendingAllocationsForNodeLocked(node_id);
    it->second.status = ClientStatus::EXPIRED;
    client_keys_.erase(node_id);
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
  record.dram_memory_desc_bytes_list = dram_memory_desc_bytes_list;

  // Build a per-tier PageBitmapAllocator for every DRAM/HBM tier that has
  // non-zero advertised capacity.  The Client may request a per-node
  // page_size at registration time; 0 falls back to the registry-wide
  // default (refactor-master-page-allocator.md §10 / Q6 — same value for
  // DRAM & HBM).
  const uint64_t effective_page_size =
      (dram_page_size > 0) ? dram_page_size : config_.default_dram_page_size;
  for (auto check_tier : {TierType::HBM, TierType::DRAM}) {
    auto cap_it = tier_capacities.find(check_tier);
    if (cap_it == tier_capacities.end() || cap_it->second.total_bytes == 0) {
      continue;
    }

    std::vector<uint64_t> buffer_sizes;
    if (!dram_buffer_sizes.empty()) {
      // Multi-buffer registration path.  Both DRAM and HBM currently share
      // the same dram_buffer_sizes list (consistent with PoolClientConfig
      // .dram_buffers); a future per-tier split would add separate inputs.
      buffer_sizes = dram_buffer_sizes;
    } else {
      // Single-buffer fallback: synthesize one buffer sized at the
      // advertised total capacity.  Wasted bytes (if total_bytes is not a
      // multiple of page_size) are simply unaddressable.
      buffer_sizes.push_back(cap_it->second.total_bytes);
    }

    auto allocator = std::make_shared<PageBitmapAllocator>(effective_page_size, buffer_sizes);
    record.page_allocators[check_tier] = allocator;
    // Re-derive tier_capacities[check_tier].available_bytes from the
    // allocator immediately so subsequent RoutePut decisions use the same
    // ground truth as Heartbeat will produce.
    record.tier_capacities[check_tier].available_bytes = allocator->AvailableBytes();
  }

  // Per-store SSD allocators (capacity-only, no OffsetTracker)
  if (!ssd_store_capacities.empty()) {
    for (uint64_t cap : ssd_store_capacities) {
      PoolAllocator alloc;
      alloc.total_size = cap;
      record.ssd_allocators.push_back(std::move(alloc));
    }
  } else {
    // Backward compat: single allocator from tier_capacities
    auto ssd_it = tier_capacities.find(TierType::SSD);
    if (ssd_it != tier_capacities.end() && ssd_it->second.total_bytes > 0) {
      PoolAllocator alloc;
      alloc.total_size = ssd_it->second.total_bytes;
      record.ssd_allocators.push_back(std::move(alloc));
    }
  }

  const size_t dram_buffer_count =
      record.page_allocators.empty() ? 0u : record.page_allocators.begin()->second->NumBuffers();
  clients_[node_id] = std::move(record);
  client_keys_[node_id];

  MORI_UMBP_INFO(
      "[Registry] Registered node: {} at {} (page_size={}KB, dram_buffers={}, ssd_stores={})",
      node_id, node_address, effective_page_size / 1024, dram_buffer_count,
      static_cast<unsigned>(ssd_store_capacities.empty()
                                ? (tier_capacities.count(TierType::SSD) ? 1u : 0u)
                                : ssd_store_capacities.size()));
  return true;
}

size_t ClientRegistry::UnregisterClient(const std::string& node_id) {
  size_t keys_removed = 0;
  std::vector<std::string> keys_to_cleanup;

  {
    std::unique_lock lock(mutex_);
    auto it = clients_.find(node_id);
    if (it == clients_.end()) {
      return 0;
    }

    auto keys_it = client_keys_.find(node_id);
    if (keys_it != client_keys_.end()) {
      keys_removed = keys_it->second.size();
      keys_to_cleanup.assign(keys_it->second.begin(), keys_it->second.end());
      client_keys_.erase(keys_it);
    }

    ReleasePendingAllocationsForNodeLocked(node_id);

    clients_.erase(it);
  }

  if (index_ != nullptr) {
    for (const auto& key : keys_to_cleanup) {
      index_->UnregisterByNode(key, node_id);
    }
  }

  MORI_UMBP_INFO("[Registry] Unregistered node: {} (keys_removed={})", node_id, keys_removed);
  return keys_removed;
}

// Heartbeat — owner of `available_bytes` differs by tier:
//   DRAM/HBM: Master is the owner.  Client's reported `available_bytes` is
//             ignored; Master recomputes it from the page allocator.
//             total_bytes mutations are also rejected (logged with throttle
//             so a flapping client cannot spam every second).
//   SSD:      Client is the owner; the whole TierCapacity entry is accepted
//             verbatim.
ClientStatus ClientRegistry::Heartbeat(const std::string& node_id,
                                       const std::map<TierType, TierCapacity>& tier_capacities) {
  std::unique_lock lock(mutex_);
  auto it = clients_.find(node_id);
  if (it == clients_.end()) {
    MORI_UMBP_WARN("[Registry] Heartbeat from unknown node: {}", node_id);
    return ClientStatus::UNKNOWN;
  }

  auto& record = it->second;
  record.last_heartbeat = std::chrono::steady_clock::now();
  record.status = ClientStatus::ALIVE;

  for (const auto& kv : tier_capacities) {
    const TierType tier = kv.first;
    const TierCapacity& reported = kv.second;
    if (IsDramOrHbm(tier)) {
      auto& stored = record.tier_capacities[tier];
      if (reported.total_bytes != stored.total_bytes) {
        // Throttled WARN: log once per distinct reported total_bytes value
        // so a flapping client doesn't spam the log every heartbeat tick.
        if (!record.dram_total_mismatch_logged ||
            record.last_logged_dram_total != reported.total_bytes) {
          MORI_UMBP_WARN(
              "[Registry] DRAM/HBM total_bytes change ignored: node={} tier={} stored={}MB "
              "reported={}MB (hot resize is not yet supported)",
              node_id, TierTypeName(tier), stored.total_bytes / kBytesPerMB,
              reported.total_bytes / kBytesPerMB);
          record.dram_total_mismatch_logged = true;
          record.last_logged_dram_total = reported.total_bytes;
        }
      } else if (record.dram_total_mismatch_logged) {
        MORI_UMBP_INFO("[Registry] DRAM/HBM total back in sync: node={} tier={} total={}MB",
                       node_id, TierTypeName(tier), reported.total_bytes / kBytesPerMB);
        record.dram_total_mismatch_logged = false;
      }
      // Master is the source of truth for available_bytes — recompute even
      // if the Client tried to report something different.
      UpdateAvailableBytesLocked(record, tier);
    } else if (tier == TierType::SSD) {
      // SSD is Client-owned; accept verbatim.
      record.tier_capacities[tier] = reported;
    }
  }

  return ClientStatus::ALIVE;
}

// #6 fix: index_->Lookup() is moved out of the registry unique_lock to avoid
// blocking Heartbeat / AllocateForPut.  Phase 1 (shared_lock) loads the
// index_ pointer and checks client existence; Phase 2 does the ownership
// Lookup with no registry lock held; Phase 3 (unique_lock) re-checks the
// client and mutates client_keys_.  Pairing the index_ load with the
// shared_lock is what makes the lock-free Phase 2 read safe against a
// concurrent SetBlockIndex (see client_registry.h assumption #3).
void ClientRegistry::TrackKey(const std::string& node_id, const std::string& key) {
  GlobalBlockIndex* idx = nullptr;
  {
    std::shared_lock lock(mutex_);
    if (clients_.find(node_id) == clients_.end()) {
      return;
    }
    idx = index_;
  }

  if (idx != nullptr) {
    const auto locations = idx->Lookup(key);
    const bool owns_key =
        std::any_of(locations.begin(), locations.end(),
                    [&node_id](const Location& location) { return location.node_id == node_id; });
    if (!owns_key) {
      return;
    }
  }

  std::unique_lock lock(mutex_);
  // Re-check after the lock-free window: the client may have been
  // unregistered concurrently.  Without this we would resurrect an empty
  // set via operator[] on a dead node_id.
  if (clients_.find(node_id) == clients_.end()) {
    return;
  }
  client_keys_[node_id].insert(key);
}

void ClientRegistry::UntrackKey(const std::string& node_id, const std::string& key) {
  GlobalBlockIndex* idx = nullptr;
  {
    std::shared_lock lock(mutex_);
    idx = index_;
  }

  if (idx != nullptr) {
    const auto locations = idx->Lookup(key);
    const bool still_owns_key =
        std::any_of(locations.begin(), locations.end(),
                    [&node_id](const Location& location) { return location.node_id == node_id; });
    if (still_owns_key) {
      return;
    }
  }

  std::unique_lock lock(mutex_);
  auto it = client_keys_.find(node_id);
  if (it == client_keys_.end()) {
    return;
  }
  it->second.erase(key);
  if (it->second.empty()) {
    client_keys_.erase(it);
  }
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
    if (record.status == ClientStatus::ALIVE) {
      result.push_back(record);
    }
  }
  return result;
}

std::optional<AllocateResult> ClientRegistry::AllocateForPut(const std::string& node_id,
                                                             TierType tier, uint64_t size) {
  std::unique_lock lock(mutex_);
  auto it = clients_.find(node_id);
  if (it == clients_.end() || it->second.status != ClientStatus::ALIVE) {
    return std::nullopt;
  }

  auto& record = it->second;

  if (IsDramOrHbm(tier)) {
    auto alloc_it = record.page_allocators.find(tier);
    if (alloc_it == record.page_allocators.end() || !alloc_it->second) {
      return std::nullopt;
    }
    auto& allocator = *alloc_it->second;
    const uint32_t num_pages = PagesForSize(size, allocator.PageSize());
    if (num_pages == 0) return std::nullopt;

    auto alloc = allocator.Allocate(num_pages);
    if (!alloc) return std::nullopt;

    // Post-allocation invariant guard: these conditions MUST hold for any
    // correct PageBitmapAllocator implementation (page_size is fixed > 0 at
    // construction; Allocate(num_pages>0) succeeds iff it returns exactly
    // num_pages slots; PagesForSize is the same round-up formula as
    // SizeMatchesAllocation).  Kept as a defensive fail-fast so a future
    // allocator regression surfaces here with a clear ERROR instead of
    // bubbling up to the Client as a silent malformed RoutePut response.
    //
    // On violation: free the just-allocated pages and return nullopt (no
    // pending created, no allocation_id consumed).
    const uint64_t page_size = allocator.PageSize();
    if (page_size == 0 || alloc->pages.empty() ||
        !SizeMatchesAllocation(size, alloc->pages.size(), page_size)) {
      MORI_UMBP_ERROR(
          "[Registry] AllocateForPut invariant violated: node={} tier={} size={} "
          "num_pages={} page_size={} (PageBitmapAllocator returned inconsistent result)",
          node_id, TierTypeName(tier), size, alloc->pages.size(), page_size);
      allocator.Deallocate(alloc->pages);
      UpdateAvailableBytesLocked(record, tier);
      return std::nullopt;
    }

    UpdateAvailableBytesLocked(record, tier);

    AllocateResult result;
    result.allocation_id = record.node_id + ":" + std::to_string(next_allocation_id_.fetch_add(1));
    result.peer_address = record.peer_address;
    result.engine_desc_bytes = record.engine_desc_bytes;
    result.location_id = alloc->location_id;
    result.pages = alloc->pages;
    result.page_size = page_size;

    // Build the deduplicated descriptor list for every distinct buffer_index
    // referenced in `pages`.  Map keeps it sorted ascending for free.
    //
    // Note: when the caller registered with an empty
    // dram_memory_desc_bytes_list (common in unit tests that exercise only
    // the bookkeeping side of the allocator) we leave dram_memory_descs
    // empty here — the Client-facing build assertion lives in
    // GetDramMemoryDescsForPages where it can also catch the RouteGet path.
    std::map<uint32_t, std::vector<uint8_t>> by_buf;
    for (const auto& p : alloc->pages) {
      if (p.buffer_index < record.dram_memory_desc_bytes_list.size()) {
        by_buf[p.buffer_index] = record.dram_memory_desc_bytes_list[p.buffer_index];
      }
    }
    result.dram_memory_descs.reserve(by_buf.size());
    for (auto& [buf_idx, bytes] : by_buf) {
      result.dram_memory_descs.push_back({buf_idx, std::move(bytes)});
    }

    pending_allocations_[result.allocation_id] = PendingAllocation{
        result.allocation_id, record.node_id,         tier, result.location_id,
        result.pages,         /*ssd_store_index=*/0u, size, std::chrono::steady_clock::now()};
    return result;
  }

  if (tier == TierType::SSD) {
    for (uint32_t i = 0; i < record.ssd_allocators.size(); ++i) {
      auto offset = record.ssd_allocators[i].Allocate(size);
      if (offset) {
        UpdateAvailableBytesLocked(record, tier);

        AllocateResult result;
        result.allocation_id =
            record.node_id + ":" + std::to_string(next_allocation_id_.fetch_add(1));
        result.peer_address = record.peer_address;
        result.engine_desc_bytes = record.engine_desc_bytes;
        // SSD path leaves location_id / pages / dram_memory_descs empty;
        // CommitSsdWrite will generate the real location_id later.
        result.ssd_store_index = i;
        pending_allocations_[result.allocation_id] =
            PendingAllocation{result.allocation_id,
                              record.node_id,
                              tier,
                              /*location_id=*/{},
                              /*pages=*/{},
                              i,
                              size,
                              std::chrono::steady_clock::now()};
        return result;
      }
    }
    return std::nullopt;
  }

  return std::nullopt;
}

void ClientRegistry::DeallocateForUnregister(const std::string& node_id, const Location& location) {
  std::unique_lock lock(mutex_);
  auto it = clients_.find(node_id);
  if (it == clients_.end()) {
    return;
  }

  auto& record = it->second;

  if (IsDramOrHbm(location.tier)) {
    auto parsed = ParseDramLocationId(location.location_id);
    if (!parsed) {
      MORI_UMBP_ERROR(
          "[Registry] DeallocateForUnregister: malformed DRAM/HBM location_id '{}', skipping",
          location.location_id);
      return;
    }
    auto alloc_it = record.page_allocators.find(location.tier);
    if (alloc_it == record.page_allocators.end() || !alloc_it->second) return;
    alloc_it->second->Deallocate(parsed->pages);
    UpdateAvailableBytesLocked(record, location.tier);
  } else if (location.tier == TierType::SSD) {
    const uint32_t store_idx = ParseBufferIndex(location.location_id);
    if (store_idx < record.ssd_allocators.size()) {
      record.ssd_allocators[store_idx].Deallocate(0, location.size);
      UpdateAvailableBytesLocked(record, TierType::SSD);
    }
  }
}

// Contract: on tier/size/location_id mismatch, master auto-rolls back the
// pending allocation and returns false.  The allocation_id is then dead:
// subsequent FinalizeAllocation calls with the same id see
// pending-not-found + no finalized record, and return false.  Client
// callers must route/allocate afresh rather than retrying the same id.
// On ALL false paths, the Client does NOT need to send AbortAllocation:
// either master already cleaned up the pending itself (mismatch paths),
// or there never was a pending owned by this node_id to clean up.
bool ClientRegistry::FinalizeAllocation(const std::string& node_id, const std::string& key,
                                        const Location& location,
                                        const std::string& allocation_id) {
  if (key.empty() || allocation_id.empty()) {
    return false;
  }

  {
    std::unique_lock lock(mutex_);
    auto client_it = clients_.find(node_id);
    if (client_it == clients_.end() || client_it->second.status != ClientStatus::ALIVE) {
      return false;
    }

    auto finalized_it = finalized_allocations_.find(allocation_id);
    if (finalized_it != finalized_allocations_.end()) {
      if (finalized_it->second.key == key && finalized_it->second.location == location) {
        return true;
      }
      MORI_UMBP_ERROR(
          "[Registry] Idempotent FinalizeAllocation mismatch: allocation_id={}, "
          "expected key='{}' location='{}', got key='{}' location='{}'",
          allocation_id, finalized_it->second.key, finalized_it->second.location.location_id, key,
          location.location_id);
      return false;
    }

    auto pending_it = pending_allocations_.find(allocation_id);
    if (pending_it == pending_allocations_.end()) {
      return false;
    }
    if (pending_it->second.node_id != node_id) {
      MORI_UMBP_ERROR(
          "[Registry] FinalizeAllocation: node_id mismatch for allocation '{}': "
          "pending={}, request={}",
          allocation_id, pending_it->second.node_id, node_id);
      return false;
    }
    const auto& pa = pending_it->second;
    // On any of the three field-level mismatches below, master auto-rolls
    // back the pending: the Client cannot successfully finalize this
    // allocation_id ever again (parameters are wrong), so there is no point
    // waiting for TTL to reap it.  Client therefore does NOT need to send a
    // follow-up AbortAllocation RPC.  Once this allocation_id hits a
    // mismatch, it is permanently dead -- caller must re-route/re-allocate.
    if (location.tier != pa.tier) {
      MORI_UMBP_ERROR(
          "[Registry] FinalizeAllocation: tier mismatch for allocation '{}': "
          "expected={}, got={} (auto-rolling back pending)",
          allocation_id, TierTypeName(pa.tier), TierTypeName(location.tier));
      DeallocatePendingLocked(pa);
      pending_allocations_.erase(pending_it);
      return false;
    }
    if (location.size != pa.size) {
      MORI_UMBP_ERROR(
          "[Registry] FinalizeAllocation: size mismatch for allocation '{}': "
          "expected={}, got={} (auto-rolling back pending)",
          allocation_id, pa.size, location.size);
      DeallocatePendingLocked(pa);
      pending_allocations_.erase(pending_it);
      return false;
    }
    if (IsDramOrHbm(pa.tier)) {
      // location_id must equal the canonical page-bitmap string handed out
      // at AllocateForPut time.  Catches stale clients accidentally
      // finalizing with the legacy "<buf>:<offset>" format.
      if (location.location_id != pa.location_id) {
        MORI_UMBP_ERROR(
            "[Registry] FinalizeAllocation: location_id mismatch for allocation '{}': "
            "expected='{}', got='{}' (auto-rolling back pending)",
            allocation_id, pa.location_id, location.location_id);
        DeallocatePendingLocked(pa);
        pending_allocations_.erase(pending_it);
        return false;
      }
    }

    pending_allocations_.erase(pending_it);

    finalized_allocations_[allocation_id] =
        FinalizedRecord{key, location, std::chrono::steady_clock::now()};
  }

  if (index_ != nullptr) {
    index_->Register(node_id, key, location);
  }
  return true;
}

bool ClientRegistry::PublishLocalBlock(const std::string& node_id, const std::string& key,
                                       const Location& location) {
  if (key.empty()) {
    return false;
  }

  {
    std::unique_lock lock(mutex_);
    auto client_it = clients_.find(node_id);
    if (client_it == clients_.end() || client_it->second.status != ClientStatus::ALIVE) {
      return false;
    }

    if (location.tier == TierType::SSD) {
      uint32_t buffer_index = ParseBufferIndex(location.location_id);
      if (buffer_index >= client_it->second.ssd_allocators.size()) {
        return false;
      }
      auto reserved = client_it->second.ssd_allocators[buffer_index].Allocate(location.size);
      if (!reserved.has_value()) {
        return false;
      }
      UpdateAvailableBytesLocked(client_it->second, TierType::SSD);
    }
    // DRAM/HBM: PublishLocalBlock does not pre-reserve capacity in the
    // page-bitmap allocator; the caller is expected to have produced
    // location_id from a prior AllocateForPut (or to be publishing a
    // location whose pages are already accounted for).
  }

  if (index_ != nullptr) {
    index_->Register(node_id, key, location);
  }
  return true;
}

bool ClientRegistry::AbortAllocation(const std::string& node_id, const std::string& allocation_id,
                                     uint64_t size) {
  (void)size;
  std::unique_lock lock(mutex_);
  auto pending_it = pending_allocations_.find(allocation_id);
  if (pending_it == pending_allocations_.end()) {
    // Normal race: reaper TTL, concurrent FinalizeAllocation mismatch-
    // rollback, or a duplicate Abort RPC all reach here.  Not an error.
    MORI_UMBP_DEBUG(
        "[Registry] AbortAllocation: allocation '{}' not found (already reaped/finalized/aborted)",
        allocation_id);
    return false;
  }
  if (pending_it->second.node_id != node_id) {
    // Protocol violation: caller is asking to roll back another node's
    // allocation.  Refuse and surface for the operator.
    MORI_UMBP_WARN(
        "[Registry] AbortAllocation: node_id mismatch for allocation '{}': pending={}, request={}",
        allocation_id, pending_it->second.node_id, node_id);
    return false;
  }

  PendingAllocation pending = pending_it->second;
  pending_allocations_.erase(pending_it);

  DeallocatePendingLocked(pending);
  return true;
}

std::optional<ClientIOInfo> ClientRegistry::GetClientIOInfo(const std::string& node_id,
                                                            uint32_t buffer_index) const {
  std::shared_lock lock(mutex_);
  auto it = clients_.find(node_id);
  if (it == clients_.end() || it->second.status != ClientStatus::ALIVE) {
    return std::nullopt;
  }

  ClientIOInfo info;
  info.peer_address = it->second.peer_address;
  info.engine_desc_bytes = it->second.engine_desc_bytes;
  if (buffer_index < it->second.dram_memory_desc_bytes_list.size()) {
    info.dram_memory_desc_bytes = it->second.dram_memory_desc_bytes_list[buffer_index];
  } else if (!it->second.dram_memory_desc_bytes_list.empty()) {
    info.dram_memory_desc_bytes = it->second.dram_memory_desc_bytes_list[0];
  }
  return info;
}

std::optional<std::vector<BufferMemoryDescBytes>> ClientRegistry::GetDramMemoryDescsForPages(
    const std::string& node_id, const std::vector<PageLocation>& pages) const {
  std::shared_lock lock(mutex_);
  auto it = clients_.find(node_id);
  if (it == clients_.end() || it->second.status != ClientStatus::ALIVE) {
    return std::nullopt;
  }

  std::set<uint32_t> seen;
  std::vector<BufferMemoryDescBytes> result;
  for (const auto& p : pages) {
    if (!seen.insert(p.buffer_index).second) continue;  // already collected
    if (p.buffer_index < it->second.dram_memory_desc_bytes_list.size()) {
      const auto& bytes = it->second.dram_memory_desc_bytes_list[p.buffer_index];
      // Only emit non-empty descs.  Empty entries occur in test fixtures
      // that do not bother packing real MemoryDescs; surfacing them here
      // would just produce zero-byte protobuf entries the Client cannot
      // unpack.
      if (!bytes.empty()) {
        result.push_back({p.buffer_index, bytes});
      }
    }
    // If the desc list is shorter than the buffer_index, leave it out: the
    // Client falls through to its no-desc error path which is a clear
    // structured failure (no crash, no silent corruption).  This can happen
    // in unit tests that build the registry with empty
    // dram_memory_desc_bytes_list.
  }
  // Result is grouped by buffer_index ascending because std::set iterates in
  // sorted order, so the protocol's "deduplicated + ascending" requirement
  // is satisfied here without an extra sort.
  return result;
}

std::optional<uint64_t> ClientRegistry::GetNodeDramPageSize(const std::string& node_id,
                                                            TierType tier) const {
  if (!IsDramOrHbm(tier)) return std::nullopt;
  std::shared_lock lock(mutex_);
  auto it = clients_.find(node_id);
  if (it == clients_.end() || it->second.status != ClientStatus::ALIVE) {
    return std::nullopt;
  }
  auto alloc_it = it->second.page_allocators.find(tier);
  if (alloc_it == it->second.page_allocators.end() || !alloc_it->second) {
    return std::nullopt;
  }
  return alloc_it->second->PageSize();
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
    if (reaper_thread_.joinable()) {
      reaper_thread_.join();
    }
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
    if (!reaper_running_) {
      break;
    }
    ReapExpiredClients();
    ReapExpiredPendingAllocations();
    ReapExpiredFinalizedRecords();
  }
}

// PA-4 fix: iterator-safe erase (never erase during range-for)
void ClientRegistry::ReapExpiredClients() {
  auto now = std::chrono::steady_clock::now();
  auto expiry = ExpiryDuration();
  std::vector<std::pair<std::string, std::vector<std::string>>> reap_cleanup;

  {
    std::unique_lock lock(mutex_);
    auto it = clients_.begin();
    while (it != clients_.end()) {
      if (now - it->second.last_heartbeat > expiry) {
        const std::string dead_id = it->first;
        MORI_UMBP_WARN("[Reaper] Reaping expired client: {}", dead_id);

        std::vector<std::string> keys_to_cleanup;
        auto keys_it = client_keys_.find(dead_id);
        if (keys_it != client_keys_.end()) {
          keys_to_cleanup.assign(keys_it->second.begin(), keys_it->second.end());
          client_keys_.erase(keys_it);
        }

        ReleasePendingAllocationsForNodeLocked(dead_id);

        reap_cleanup.emplace_back(dead_id, std::move(keys_to_cleanup));
        it = clients_.erase(it);  // returns next valid iterator
      } else {
        ++it;
      }
    }
  }

  if (index_ != nullptr) {
    for (const auto& [dead_id, keys_to_cleanup] : reap_cleanup) {
      for (const auto& key : keys_to_cleanup) {
        index_->UnregisterByNode(key, dead_id);
      }
    }
  }
}

void ClientRegistry::ReapExpiredPendingAllocations() {
  const auto now = std::chrono::steady_clock::now();
  std::unique_lock lock(mutex_);
  auto it = pending_allocations_.begin();
  while (it != pending_allocations_.end()) {
    if (now - it->second.allocated_at <= config_.allocation_ttl) {
      ++it;
      continue;
    }
    DeallocatePendingLocked(it->second);
    MORI_UMBP_WARN("[Reaper] Expired pending allocation: id={}", it->second.allocation_id);
    it = pending_allocations_.erase(it);
  }
}

void ClientRegistry::ReapExpiredFinalizedRecords() {
  const auto now = std::chrono::steady_clock::now();
  std::unique_lock lock(mutex_);
  auto it = finalized_allocations_.begin();
  while (it != finalized_allocations_.end()) {
    if (now - it->second.finalized_at > config_.finalized_record_ttl) {
      it = finalized_allocations_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace mori::umbp
