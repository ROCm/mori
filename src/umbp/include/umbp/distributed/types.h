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

#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mori::umbp {

enum class TierType : int {
  UNKNOWN = 0,
  HBM = 1,
  DRAM = 2,
  SSD = 3,
};

enum class HiCacheTransfer : int {
  UNKNOWN = 0,
  L1_TO_L2 = 1,
  L2_TO_L1 = 2,
  L2_TO_L3 = 3,
  L3_TO_L2 = 4,
};

struct HiCacheTransferRate {
  HiCacheTransfer direction = HiCacheTransfer::UNKNOWN;
  double bytes_per_sec = 0.0;
  uint64_t rate_age_ms = 0;
  uint64_t window_ms = 0;
};

struct ClientTransferRates {
  std::string node_id;
  std::string peer_address;
  std::vector<std::string> tags;
  std::vector<HiCacheTransferRate> rates;
};

struct TierCapacity {
  uint64_t total_bytes = 0;
  uint64_t available_bytes = 0;
};

// In the master-as-advisor design, Location is a (node_id, tier) handle.
// The peer is the canonical owner of every per-key page set; master holds
// no per-key page state, so location_id is gone.  `size` is carried so
// the read path can size its RDMA buffer without a separate Resolve.
//
// Dedup key for the index is (node_id, tier): a single peer cannot host
// two replicas of the same key on the same tier.
struct Location {
  std::string node_id;
  uint64_t size = 0;
  TierType tier = TierType::UNKNOWN;

  bool operator==(const Location& other) const {
    return node_id == other.node_id && size == other.size && tier == other.tier;
  }
};

enum class ClientStatus : int {
  UNKNOWN = 0,
  ALIVE = 1,
  EXPIRED = 2,
};

struct BlockMetrics {
  std::chrono::steady_clock::time_point created_at;
  std::chrono::steady_clock::time_point last_accessed_at;
  uint64_t access_count = 0;
};

// Structured form of one (buffer_index, page_index) slot.  Used by the
// peer DRAM/HBM allocator to describe which page slot a write should
// land in, and by ResolveKey responses to tell readers where to RDMA
// from.  Master never sees this type — it's a peer-internal handle.
struct PageLocation {
  uint32_t buffer_index = 0;
  uint32_t page_index = 0;

  bool operator==(const PageLocation& other) const {
    return buffer_index == other.buffer_index && page_index == other.page_index;
  }
  bool operator!=(const PageLocation& other) const { return !(*this == other); }
  bool operator<(const PageLocation& other) const {
    if (buffer_index != other.buffer_index) return buffer_index < other.buffer_index;
    return page_index < other.page_index;
  }
};

// One peer-side buffer's RDMA MemoryDesc bytes plus the buffer_index it
// belongs to.  Returned by PeerDramAllocator and the peer service in
// AllocateSlot / ResolveKey / GetPeerInfo responses so the Client can
// hydrate its peer-side buffer_index -> MemoryDesc cache in a single
// batch.
struct BufferMemoryDescBytes {
  uint32_t buffer_index = 0;
  std::vector<uint8_t> desc_bytes;
};

// One mutation in a peer's owned-key set, shipped via Heartbeat.  Mirrors
// the wire-level umbp::KvEvent — kept as a plain C++ struct so the peer
// allocator (and its unit tests) do not have to depend on the generated
// proto headers.
struct KvEvent {
  enum class Kind : int { ADD = 0, REMOVE = 1 };
  Kind kind = Kind::ADD;
  std::string key;
  TierType tier = TierType::UNKNOWN;
  uint64_t size = 0;  // ADD only; REMOVE leaves this 0
};

// Master-side snapshot of one peer node.  In the master-as-advisor design
// master holds no allocator state — capacity is whatever the peer most
// recently reported.  `last_applied_seq` is the last heartbeat sequence
// number whose events have been applied to the index; used to detect
// gaps and trigger a full sync.
struct ClientRecord {
  std::string node_id;
  std::string node_address;
  ClientStatus status = ClientStatus::UNKNOWN;
  std::chrono::steady_clock::time_point last_heartbeat;
  std::chrono::steady_clock::time_point registered_at;
  std::map<TierType, TierCapacity> tier_capacities;

  std::string peer_address;
  std::vector<uint8_t> engine_desc_bytes;

  uint64_t last_applied_seq = 0;

  // Opaque key=value labels supplied at registration (e.g. "sgl_role=prefill").
  // Attached to all metrics reported by this node so Prometheus queries can
  // filter/group by role or other client attributes.
  std::vector<std::string> tags;
};

// Helpers for logging
inline const char* TierTypeName(TierType t) {
  switch (t) {
    case TierType::HBM:
      return "HBM";
    case TierType::DRAM:
      return "DRAM";
    case TierType::SSD:
      return "SSD";
    default:
      return "UNKNOWN";
  }
}

inline const char* HiCacheTransferName(HiCacheTransfer d) {
  switch (d) {
    case HiCacheTransfer::L1_TO_L2:
      return "L1_TO_L2";
    case HiCacheTransfer::L2_TO_L1:
      return "L2_TO_L1";
    case HiCacheTransfer::L2_TO_L3:
      return "L2_TO_L3";
    case HiCacheTransfer::L3_TO_L2:
      return "L3_TO_L2";
    default:
      return "UNKNOWN";
  }
}

inline const char* ClientStatusName(ClientStatus s) {
  switch (s) {
    case ClientStatus::ALIVE:
      return "ALIVE";
    case ClientStatus::EXPIRED:
      return "EXPIRED";
    default:
      return "UNKNOWN";
  }
}

}  // namespace mori::umbp
