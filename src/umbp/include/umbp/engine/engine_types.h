// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "umbp/distributed/object_view.h"

namespace mori::umbp::engine {

using RankId = uint32_t;

struct PoolEpoch {
  std::array<uint8_t, 16> boot_id{};
  uint64_t reset_seq = 0;

  bool operator==(const PoolEpoch& other) const {
    return boot_id == other.boot_id && reset_seq == other.reset_seq;
  }
  bool operator!=(const PoolEpoch& other) const { return !(*this == other); }
};

struct OperationId {
  std::array<uint8_t, 16> session_id{};
  uint64_t seq = 0;

  bool operator==(const OperationId& other) const {
    return session_id == other.session_id && seq == other.seq;
  }
  bool operator!=(const OperationId& other) const { return !(*this == other); }
};

struct AllocationHandle {
  std::string pool_id;
  PoolEpoch epoch;
  uint32_t backend_id = 0;
  uint32_t page_id = 0;
  uint64_t generation = 0;

  bool operator==(const AllocationHandle& other) const {
    return pool_id == other.pool_id && epoch == other.epoch && backend_id == other.backend_id &&
           page_id == other.page_id && generation == other.generation;
  }
};

struct ResidentIdentity {
  std::string pool_id;
  PoolEpoch epoch;
  uint32_t backend_id = 0;
  uint32_t page_id = 0;
  uint64_t generation = 0;

  explicit ResidentIdentity(const AllocationHandle& handle)
      : pool_id(handle.pool_id),
        epoch(handle.epoch),
        backend_id(handle.backend_id),
        page_id(handle.page_id),
        generation(handle.generation) {}

  bool operator==(const ResidentIdentity& other) const {
    return pool_id == other.pool_id && epoch == other.epoch && backend_id == other.backend_id &&
           page_id == other.page_id && generation == other.generation;
  }
};

inline size_t HashBytes(const uint8_t* data, size_t size, size_t seed = 1469598103934665603ULL) {
  size_t out = seed;
  for (size_t i = 0; i < size; ++i) out = (out ^ data[i]) * 1099511628211ULL;
  return out;
}

struct ResidentIdentityHash {
  size_t operator()(const ResidentIdentity& id) const {
    size_t h = std::hash<std::string>{}(id.pool_id);
    h = HashBytes(id.epoch.boot_id.data(), id.epoch.boot_id.size(), h);
    h ^= std::hash<uint64_t>{}(id.epoch.reset_seq + (static_cast<uint64_t>(id.backend_id) << 32));
    h ^= std::hash<uint64_t>{}((static_cast<uint64_t>(id.page_id) << 32) ^ id.generation);
    return h;
  }
};

struct OperationIdHash {
  size_t operator()(const OperationId& id) const {
    return HashBytes(id.session_id.data(), id.session_id.size()) ^ std::hash<uint64_t>{}(id.seq);
  }
};

// One SGLang page contributes one physical run per region. The buffer starts at
// the page-number-zero row; page 0 remains padding and is never delegated.
struct PageRegion {
  std::string region_id;
  TransferRef buffer;
  uint64_t row_bytes = 0;
  uint64_t rows_per_page = 0;
  uint64_t logical_offset = 0;
  int32_t layer_id = -1;
  std::string component;

  uint64_t PageBytes() const {
    if (rows_per_page == 0 || row_bytes > std::numeric_limits<uint64_t>::max() / rows_per_page) {
      return 0;
    }
    return row_bytes * rows_per_page;
  }
};

struct PageGeometry {
  uint32_t page_tokens = 0;
  uint32_t num_pages = 0;
  uint64_t object_bytes = 0;
  std::string layout_id;
  std::vector<PageRegion> regions;

  bool Validate() const {
    if (page_tokens == 0 || num_pages == 0 || object_bytes == 0 || regions.empty()) return false;
    for (const auto& region : regions) {
      const uint64_t bytes = region.PageBytes();
      if (bytes == 0 || num_pages > std::numeric_limits<uint64_t>::max() / bytes) return false;
      const uint64_t last_offset = static_cast<uint64_t>(num_pages) * bytes;
      if (!region.buffer.Valid() || last_offset > region.buffer.size ||
          bytes > region.buffer.size - last_offset) {
        return false;
      }
    }
    return ValidateObjectView(View(1)) && ValidateObjectView(View(num_pages));
  }

  ObjectView View(uint32_t page_id) const {
    ObjectView view;
    view.size = object_bytes;
    view.layout_id = layout_id;
    if (page_id == 0 || page_id > num_pages) return view;
    for (const auto& region : regions) {
      const uint64_t bytes = region.PageBytes();
      if (bytes == 0 || page_id > std::numeric_limits<uint64_t>::max() / bytes) {
        view.spans.clear();
        return view;
      }
      const uint64_t offset = static_cast<uint64_t>(page_id) * bytes;
      if (!region.buffer.Valid() || offset > region.buffer.size ||
          bytes > region.buffer.size - offset) {
        view.spans.clear();
        return view;
      }
      view.spans.push_back(RegionSpan{region.buffer, offset, bytes, region.logical_offset});
    }
    if (!ValidateObjectView(view)) view.spans.clear();
    return view;
  }
};

inline std::string ComputeLayoutId(const PageGeometry& geometry) {
  // FNV-1a over structural fields only: addresses, rank and backend placement
  // are deliberately absent, so replicas of the same layout share the id.
  uint64_t hash = 1469598103934665603ULL;
  const auto add = [&](const void* data, size_t size) {
    hash = HashBytes(static_cast<const uint8_t*>(data), size, hash);
  };
  add(&geometry.page_tokens, sizeof(geometry.page_tokens));
  add(&geometry.object_bytes, sizeof(geometry.object_bytes));
  for (const auto& region : geometry.regions) {
    add(region.region_id.data(), region.region_id.size());
    add(&region.row_bytes, sizeof(region.row_bytes));
    add(&region.rows_per_page, sizeof(region.rows_per_page));
    add(&region.logical_offset, sizeof(region.logical_offset));
    add(&region.layer_id, sizeof(region.layer_id));
    add(region.component.data(), region.component.size());
  }
  std::ostringstream out;
  out << "engine-page-v1-" << std::hex << std::setfill('0') << std::setw(16) << hash;
  return out.str();
}

enum class SlotState : uint8_t { kFree, kDelegated, kPublished, kRetiring, kQuarantined };

enum class RetireState : uint8_t { kWaitingHolds, kReady, kReleased, kQuarantined };

struct RetireTicket {
  OperationId id;
  std::vector<AllocationHandle> handles;
  RetireState state = RetireState::kWaitingHolds;
};

enum class CompletionProof : uint8_t { kCompleted, kNeverSubmitted, kUnknown };

}  // namespace mori::umbp::engine
