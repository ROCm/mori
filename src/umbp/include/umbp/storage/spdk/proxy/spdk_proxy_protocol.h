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
//
// SPDK Proxy IPC Protocol — shared memory layout for multitenant SPDK access.
//
// Architecture:
//   spdk_proxy daemon (1 process) owns SPDK + NVMe device.
//   N client sessions communicate through POSIX shared memory channels.
//   Each channel is permanently bound to one tenant after ATTACH_SESSION.
//
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace umbp {
namespace proxy {

// ---- Constants ----
static constexpr uint64_t kProxyShmMagic = 0x554D4250534B5058ULL;  // "UMBPSKPX"
static constexpr uint32_t kProxyVersion = 4;
static constexpr uint32_t kMaxKeyLen = 256;
static constexpr uint32_t kRingSize = 256;
static constexpr uint32_t kMaxChannels = 64;
static constexpr uint32_t kMaxTenants = 64;
static constexpr size_t kDefaultDataRegionPerChannel = 32ULL * 1024 * 1024;        // 32MB
static constexpr size_t kDefaultDataRegionPerRank = kDefaultDataRegionPerChannel;  // legacy alias
static constexpr size_t kDmaAlignment = 4096;
static constexpr size_t kHugepageSize = 2ULL * 1024 * 1024;
static constexpr size_t kCacheRingAlign = 64;

static constexpr uint32_t kTenantFlagActive = 1u << 0;
static constexpr uint32_t kTenantFlagReaping = 1u << 1;

static constexpr const char* kDefaultShmName = "/umbp_spdk_proxy";
// Default stale threshold for the proxy heartbeat in the SHM header.
// Runtime value is overridable via UMBP_SPDK_PROXY_HEARTBEAT_STALE_MS;
// see the consumer TUs (spdk_proxy_shm.cpp, spdk_proxy_tier.cpp,
// spdk_proxy_daemon.cpp) for the resolver. Kept here only as the numeric
// default so this protocol header stays free of env/log dependencies.
static constexpr uint64_t kDefaultHeartbeatStaleMs = 5000;

inline uint64_t NowEpochMs() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count());
}

// ---- Enums ----
enum class ProxyState : uint32_t {
  UNINIT = 0,
  READY = 1,
  DRAINING = 2,
  SHUTDOWN = 3,
  ERROR = 4,
};

enum class SlotState : uint32_t {
  EMPTY = 0,
  PENDING = 1,
  COMPLETED = 2,
};

enum class RequestType : uint32_t {
  NOP = 0,
  ATTACH_SESSION = 1,
  DETACH_SESSION = 2,
  PUT = 3,
  GET = 4,
  EXISTS = 5,
  REMOVE = 6,
  CLEAR_TENANT = 7,
  CAPACITY_TENANT = 8,
  GET_LRU_KEY = 9,
  GET_LRU_CANDIDATES = 10,
  BATCH_PUT = 11,
  BATCH_GET = 12,
  FLUSH_TENANT = 13,
  GET_TENANT_STATS = 14,
  ADMIN_SHUTDOWN = 15,
  ADMIN_CLEAR_ALL = 16,
  ADMIN_GET_GLOBAL_STATS = 17,
};

enum class ResultCode : int32_t {
  OK = 0,
  NOT_FOUND = -1,
  NO_SPACE = -2,
  ERROR = -3,
  PERMISSION_DENIED = -4,
  PROTOCOL_MISMATCH = -5,
  BUSY = -6,
};

// ---- Per-slot request structure ----
struct alignas(64) RingSlot {
  std::atomic<uint32_t> state;  // SlotState
  uint32_t type;                // RequestType
  uint64_t seq_id;
  uint32_t key_len;
  char key[kMaxKeyLen];
  uint64_t data_offset;  // offset within channel data region
  uint64_t data_size;
  uint64_t request_aux;  // attach quota hint, admin args, etc.
  uint32_t flags;
  uint32_t batch_count;
  uint64_t ring_data_base;

  int32_t result;  // ResultCode
  uint32_t _pad0;
  uint64_t result_size;
  uint64_t result_aux;
};

// ---- Per-channel transport ----
struct ClientChannel {
  alignas(64) std::atomic<uint32_t> head;
  alignas(64) std::atomic<uint32_t> tail;
  alignas(64) std::atomic<uint32_t> owner_pid;  // 0 = free slot
  alignas(64) std::atomic<uint32_t> connected;  // 1 = claimed by client

  uint32_t channel_id;
  uint32_t client_pid;
  uint32_t tenant_id;
  uint32_t tenant_slot;

  std::atomic<uint64_t> session_id;  // 0 until ATTACH_SESSION succeeds
  std::atomic<uint64_t> last_activity_ms;
  uint64_t requested_quota_bytes;
  char _pad[32];

  RingSlot slots[kRingSize];
};

using RankChannel = ClientChannel;  // legacy alias

// ---- Batch descriptor (stored in data region for BATCH_PUT/BATCH_GET) ----
struct BatchEntry {
  uint16_t key_len;
  char key[kMaxKeyLen];
  uint64_t data_offset;
  uint64_t data_size;
  uint8_t result;
  char _pad[5];
};

struct BatchDescriptor {
  uint32_t count;
  uint32_t _reserved;
  uint64_t total_data_size;
  uint64_t ring_data_base;
  alignas(64) std::atomic<uint32_t> items_ready;
  alignas(64) std::atomic<uint32_t> items_done;
  alignas(64) std::atomic<uint64_t> bytes_ready;
  alignas(64) std::atomic<uint64_t> bytes_done;

  BatchEntry entries[];
};

struct alignas(64) TenantStatsPayload {
  uint64_t used_bytes;
  uint64_t quota_bytes;
  uint64_t reserved_bytes;
  uint64_t evicted_bytes;
  uint64_t hit_count;
  uint64_t miss_count;
  uint64_t last_activity_ms;
  uint64_t cache_epoch;
  uint32_t active_sessions;
  uint32_t flags;
  uint32_t tenant_id;
  uint32_t tenant_slot;
};

// ---- SHM ring cache ----
struct alignas(64) CacheRingControl {
  std::atomic<uint64_t> write_pos;
  uint64_t capacity;
  char _pad[48];
};

struct alignas(64) CacheRingIndexEntry {
  std::atomic<uint64_t> gen;
  uint32_t tenant_id;
  uint32_t key_len;
  uint64_t tenant_cache_epoch;
  uint64_t ring_offset;
  uint64_t data_size;
  char key[kMaxKeyLen];
  char _pad[16];
};

inline uint32_t CacheIndexHash(const char* key, uint32_t key_len, uint32_t tenant_id,
                               uint32_t num_slots) {
  uint64_t h = 14695981039346656037ULL;
  h ^= static_cast<uint64_t>(tenant_id);
  h *= 1099511628211ULL;
  for (uint32_t i = 0; i < key_len; ++i) {
    h ^= static_cast<uint64_t>(static_cast<unsigned char>(key[i]));
    h *= 1099511628211ULL;
  }
  return static_cast<uint32_t>(h % num_slots);
}

// ---- Tenant registry ----
struct alignas(64) TenantInfo {
  uint32_t tenant_id;
  uint32_t tenant_slot;
  std::atomic<uint32_t> active_sessions;
  std::atomic<uint32_t> flags;
  std::atomic<uint64_t> used_bytes;
  std::atomic<uint64_t> quota_bytes;
  std::atomic<uint64_t> reserved_bytes;
  std::atomic<uint64_t> evicted_bytes;
  std::atomic<uint64_t> hit_count;
  std::atomic<uint64_t> miss_count;
  std::atomic<uint64_t> last_activity_ms;
  std::atomic<uint64_t> cache_epoch;
};

// ---- Global shared memory header ----
struct ProxyShmHeader {
  uint64_t magic;
  uint32_t version;
  std::atomic<uint32_t> state;  // ProxyState

  uint64_t service_epoch;
  uint32_t max_channels;
  uint32_t max_tenants;
  uint32_t block_size;
  uint32_t hugepage;
  uint64_t bdev_size;

  uint64_t tenant_table_offset;
  uint64_t channels_offset;
  uint64_t data_region_offset;
  uint64_t data_region_per_channel;
  uint64_t total_shm_size;

  std::atomic<uint64_t> next_session_id;
  std::atomic<uint64_t> next_cache_epoch;
  std::atomic<uint32_t> proxy_pid;
  std::atomic<uint32_t> active_sessions;
  std::atomic<uint64_t> capacity_used;
  std::atomic<uint64_t> capacity_total;
  std::atomic<uint64_t> proxy_heartbeat_ms;
  std::atomic<uint64_t> last_activity_ms;

  uint64_t cache_region_offset;
  uint64_t cache_ring_capacity;
  uint32_t cache_index_slots;
  uint32_t _pad_cache;

  char _reserved[32];
};

// ---- Size helpers ----
inline size_t ComputeShmSize(uint32_t max_channels, uint32_t max_tenants, size_t data_per_channel,
                             size_t cache_total_bytes = 0) {
  size_t header_size = sizeof(ProxyShmHeader);
  size_t tenants_offset = (header_size + 4095) & ~4095ULL;
  size_t tenants_size = sizeof(TenantInfo) * max_tenants;
  size_t channels_offset = (tenants_offset + tenants_size + 4095) & ~4095ULL;
  size_t channels_size = sizeof(ClientChannel) * max_channels;
  size_t data_offset = (channels_offset + channels_size + 4095) & ~4095ULL;
  size_t data_size = data_per_channel * max_channels;
  size_t base = data_offset + data_size;
  if (cache_total_bytes > 0) {
    base = (base + 4095) & ~4095ULL;
    base += cache_total_bytes;
  }
  return base;
}

// ---- Pointer helpers ----
inline TenantInfo* GetTenantInfo(void* shm_base, const ProxyShmHeader* hdr, uint32_t tenant_slot) {
  if (tenant_slot >= hdr->max_tenants) return nullptr;
  auto* base = static_cast<char*>(shm_base);
  return reinterpret_cast<TenantInfo*>(base + hdr->tenant_table_offset) + tenant_slot;
}

inline const TenantInfo* GetTenantInfo(const void* shm_base, const ProxyShmHeader* hdr,
                                       uint32_t tenant_slot) {
  if (tenant_slot >= hdr->max_tenants) return nullptr;
  auto* base = static_cast<const char*>(shm_base);
  return reinterpret_cast<const TenantInfo*>(base + hdr->tenant_table_offset) + tenant_slot;
}

inline ClientChannel* GetChannel(void* shm_base, const ProxyShmHeader* hdr, uint32_t channel) {
  if (channel >= hdr->max_channels) return nullptr;
  auto* base = static_cast<char*>(shm_base);
  return reinterpret_cast<ClientChannel*>(base + hdr->channels_offset) + channel;
}

inline const ClientChannel* GetChannel(const void* shm_base, const ProxyShmHeader* hdr,
                                       uint32_t channel) {
  if (channel >= hdr->max_channels) return nullptr;
  auto* base = static_cast<const char*>(shm_base);
  return reinterpret_cast<const ClientChannel*>(base + hdr->channels_offset) + channel;
}

inline void* GetDataRegion(void* shm_base, const ProxyShmHeader* hdr, uint32_t channel) {
  if (channel >= hdr->max_channels) return nullptr;
  auto* base = static_cast<char*>(shm_base);
  return base + hdr->data_region_offset + channel * hdr->data_region_per_channel;
}

inline const void* GetDataRegion(const void* shm_base, const ProxyShmHeader* hdr,
                                 uint32_t channel) {
  if (channel >= hdr->max_channels) return nullptr;
  auto* base = static_cast<const char*>(shm_base);
  return base + hdr->data_region_offset + channel * hdr->data_region_per_channel;
}

inline CacheRingControl* GetCacheRingControl(void* shm_base, const ProxyShmHeader* hdr) {
  if (hdr->cache_region_offset == 0) return nullptr;
  return reinterpret_cast<CacheRingControl*>(static_cast<char*>(shm_base) +
                                             hdr->cache_region_offset);
}

inline const CacheRingControl* GetCacheRingControl(const void* shm_base,
                                                   const ProxyShmHeader* hdr) {
  if (hdr->cache_region_offset == 0) return nullptr;
  return reinterpret_cast<const CacheRingControl*>(static_cast<const char*>(shm_base) +
                                                   hdr->cache_region_offset);
}

inline CacheRingIndexEntry* GetCacheIndex(void* shm_base, const ProxyShmHeader* hdr) {
  if (hdr->cache_region_offset == 0) return nullptr;
  return reinterpret_cast<CacheRingIndexEntry*>(
      static_cast<char*>(shm_base) + hdr->cache_region_offset + sizeof(CacheRingControl));
}

inline const CacheRingIndexEntry* GetCacheIndex(const void* shm_base, const ProxyShmHeader* hdr) {
  if (hdr->cache_region_offset == 0) return nullptr;
  return reinterpret_cast<const CacheRingIndexEntry*>(
      static_cast<const char*>(shm_base) + hdr->cache_region_offset + sizeof(CacheRingControl));
}

inline size_t CacheRingDataOffset(const ProxyShmHeader* hdr) {
  if (hdr->cache_region_offset == 0) return 0;
  size_t idx_end = hdr->cache_region_offset + sizeof(CacheRingControl) +
                   static_cast<size_t>(hdr->cache_index_slots) * sizeof(CacheRingIndexEntry);
  return (idx_end + 4095) & ~4095ULL;
}

inline char* GetCacheRingData(void* shm_base, const ProxyShmHeader* hdr) {
  size_t off = CacheRingDataOffset(hdr);
  return off ? (static_cast<char*>(shm_base) + off) : nullptr;
}

inline const char* GetCacheRingData(const void* shm_base, const ProxyShmHeader* hdr) {
  size_t off = CacheRingDataOffset(hdr);
  return off ? (static_cast<const char*>(shm_base) + off) : nullptr;
}

inline uint64_t AllocRingContiguous(CacheRingControl* ctrl, size_t size) {
  if (!ctrl || size == 0) return UINT64_MAX;
  uint64_t cap = ctrl->capacity;
  size_t aligned = (size + kCacheRingAlign - 1) & ~(kCacheRingAlign - 1);
  if (aligned > (cap * 3) / 4) return UINT64_MAX;

  uint64_t expected = ctrl->write_pos.load(std::memory_order_relaxed);
  while (true) {
    uint64_t alloc_start = expected;
    size_t off = static_cast<size_t>(expected % cap);
    if (off + aligned > cap) alloc_start = expected + (cap - off);
    uint64_t new_pos = alloc_start + aligned;
    if (ctrl->write_pos.compare_exchange_weak(expected, new_pos, std::memory_order_acq_rel)) {
      return alloc_start;
    }
  }
}

}  // namespace proxy
}  // namespace umbp
