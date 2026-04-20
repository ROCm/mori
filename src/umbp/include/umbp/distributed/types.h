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
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "umbp/distributed/pool_allocator.h"

namespace mori::umbp {

// Forward declared so ClientRecord can hold a per-tier PageBitmapAllocator
// map without inverting the include order (page_bitmap_allocator.h includes
// types.h, not the other way around).
class PageBitmapAllocator;

enum class TierType : int {
  UNKNOWN = 0,
  HBM = 1,
  DRAM = 2,
  SSD = 3,
};

struct TierCapacity {
  uint64_t total_bytes = 0;
  uint64_t available_bytes = 0;
};

struct Location {
  std::string node_id;
  std::string location_id;  // Opaque handle from target node
  uint64_t size = 0;
  TierType tier = TierType::UNKNOWN;

  bool operator==(const Location& other) const {
    return node_id == other.node_id && location_id == other.location_id && size == other.size &&
           tier == other.tier;
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

// Structured form of a single (buffer_index, page_index) slot referenced
// by a DRAM/HBM location_id such as "0:p1,2;1:p0".  Defined here (rather
// than alongside ParseDramLocationId further down) because PendingAllocation
// and ClientRecord-adjacent types need the complete type for their members.
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
// belongs to.  Returned by ClientRegistry helpers and threaded through
// Master -> Client RoutePut/RouteGet responses so the Client can hydrate
// its peer-side buffer_index -> MemoryDesc cache in a single batch.
struct BufferMemoryDescBytes {
  uint32_t buffer_index = 0;
  std::vector<uint8_t> desc_bytes;
};

struct ClientRecord {
  std::string node_id;
  std::string node_address;
  ClientStatus status = ClientStatus::UNKNOWN;
  std::chrono::steady_clock::time_point last_heartbeat;
  std::chrono::steady_clock::time_point registered_at;
  std::map<TierType, TierCapacity> tier_capacities;

  std::string peer_address;
  std::vector<uint8_t> engine_desc_bytes;

  std::vector<std::vector<uint8_t>> dram_memory_desc_bytes_list;

  // Per-tier page-bitmap allocator (DRAM/HBM only).  SSD continues to use
  // `ssd_allocators` below since SSD allocation happens Client-side
  // (CommitSsdWrite generates its own location_id).
  //
  // shared_ptr keeps ClientRecord copy-constructible (GetAliveClients()
  // returns a snapshot vector) without needing PageBitmapAllocator to be
  // default-constructible.
  std::map<TierType, std::shared_ptr<PageBitmapAllocator>> page_allocators;

  // Capacity-only PoolAllocator for SSD stores (per-store).
  std::vector<PoolAllocator> ssd_allocators;

  // Throttle for the "Client reported a different total_bytes than was
  // registered" WARN.  Per-record (not per-tier): if HBM and DRAM both
  // drift simultaneously we log only one at a time.
  bool dram_total_mismatch_logged = false;
  uint64_t last_logged_dram_total = 0;
};

struct PendingAllocation {
  std::string allocation_id;
  std::string node_id;
  TierType tier = TierType::UNKNOWN;

  // For DRAM/HBM tier: location_id + pages used to call
  // PageBitmapAllocator::Deallocate (on Abort / Reaper / eviction).  Both
  // are populated together by AllocateForPut and never independently.
  std::string location_id;
  std::vector<PageLocation> pages;

  // For SSD tier: which SSD store (PoolAllocator) to release back to.
  uint32_t ssd_store_index = 0;

  uint64_t size = 0;
  std::chrono::steady_clock::time_point allocated_at;
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

struct ParsedLocationId {
  uint32_t buffer_index = 0;
  uint64_t offset = 0;
};

inline std::optional<ParsedLocationId> ParseLocationId(const std::string& location_id) {
  auto colon = location_id.find(':');
  if (colon == std::string::npos) return std::nullopt;
  try {
    ParsedLocationId result;
    result.buffer_index = static_cast<uint32_t>(std::stoul(location_id.substr(0, colon)));
    result.offset = std::stoull(location_id.substr(colon + 1));
    return result;
  } catch (...) {
    return std::nullopt;
  }
}

// ---------------------------------------------------------------------------
// DRAM/HBM page-bitmap location_id parsing
//
// PageLocation itself is declared higher up in this file (next to
// PendingAllocation) because it is a member type of those structs.  Here we
// only define the parsed-form wrapper and the parser entry point.
// ---------------------------------------------------------------------------

// Parsed form of a DRAM/HBM page-bitmap location_id.
//
// Format: "{buf}:p{idx}[,{idx}...][;{buf}:p{idx}...]"
// Examples:
//   "0:p3"            -> buffer 0, page 3
//   "0:p3,4"          -> buffer 0, pages 3+4
//   "0:p1,2;1:p0"     -> buffer 0 pages 1+2, buffer 1 page 0
//
// SSD location_id (e.g. "0:seg1:42") is *not* handled here; it is parsed by
// the SSD-side helpers (CommitSsdWrite generates that format).
struct ParsedDramLocation {
  std::vector<PageLocation> pages;
};

// Defensive parser: any malformed input yields std::nullopt.
//
// Rejected as malformed (each returns std::nullopt):
//   - empty string
//   - segment missing ':' or ":p" prefix
//   - non-numeric buffer_index or page_index
//   - empty page list (e.g. "0:p", "0:p,3", "0:p3,")
//   - duplicate (buffer_index, page_index) pair (e.g. "0:p1,1" or "0:p1;0:p1")
//   - duplicate buffer_index across segments (canonical serialization groups
//     all pages of the same buffer into one segment, so "0:p1;0:p2" is
//     considered malformed)
//   - trailing ';' (e.g. "0:p1;")
inline std::optional<ParsedDramLocation> ParseDramLocationId(const std::string& s) {
  if (s.empty()) return std::nullopt;

  ParsedDramLocation parsed;
  std::vector<uint32_t> seen_buffers;

  // Split by ';'.  We deliberately do NOT skip empty segments — an empty
  // segment (leading, internal, or trailing ';') is malformed.
  size_t seg_begin = 0;
  while (seg_begin <= s.size()) {
    size_t seg_end = s.find(';', seg_begin);
    bool is_last = (seg_end == std::string::npos);
    size_t seg_len = is_last ? (s.size() - seg_begin) : (seg_end - seg_begin);
    if (seg_len == 0) return std::nullopt;  // empty segment (incl. trailing ';')

    std::string segment = s.substr(seg_begin, seg_len);

    auto colon = segment.find(':');
    if (colon == std::string::npos) return std::nullopt;
    if (colon + 1 >= segment.size() || segment[colon + 1] != 'p') return std::nullopt;

    uint32_t buf_idx = 0;
    try {
      const std::string buf_str = segment.substr(0, colon);
      if (buf_str.empty()) return std::nullopt;
      for (char c : buf_str) {
        if (c < '0' || c > '9') return std::nullopt;
      }
      unsigned long v = std::stoul(buf_str);
      if (v > std::numeric_limits<uint32_t>::max()) return std::nullopt;
      buf_idx = static_cast<uint32_t>(v);
    } catch (...) {
      return std::nullopt;
    }

    for (auto prev : seen_buffers) {
      if (prev == buf_idx) return std::nullopt;
    }
    seen_buffers.push_back(buf_idx);

    // Page list: substring after "p", split by ','.  Must contain >=1 page,
    // each numeric, no duplicates within this segment.
    const std::string page_list = segment.substr(colon + 2);
    if (page_list.empty()) return std::nullopt;

    std::vector<uint32_t> pages_in_segment;
    size_t p_begin = 0;
    while (p_begin <= page_list.size()) {
      size_t p_end = page_list.find(',', p_begin);
      bool p_is_last = (p_end == std::string::npos);
      size_t p_len = p_is_last ? (page_list.size() - p_begin) : (p_end - p_begin);
      if (p_len == 0) return std::nullopt;  // empty page entry

      const std::string page_str = page_list.substr(p_begin, p_len);
      for (char c : page_str) {
        if (c < '0' || c > '9') return std::nullopt;
      }
      uint32_t page_idx = 0;
      try {
        unsigned long v = std::stoul(page_str);
        if (v > std::numeric_limits<uint32_t>::max()) return std::nullopt;
        page_idx = static_cast<uint32_t>(v);
      } catch (...) {
        return std::nullopt;
      }

      for (auto prev : pages_in_segment) {
        if (prev == page_idx) return std::nullopt;
      }
      pages_in_segment.push_back(page_idx);
      parsed.pages.push_back({buf_idx, page_idx});

      if (p_is_last) break;
      p_begin = p_end + 1;
    }

    if (is_last) break;
    seg_begin = seg_end + 1;
  }

  if (parsed.pages.empty()) return std::nullopt;
  return parsed;
}

}  // namespace mori::umbp
