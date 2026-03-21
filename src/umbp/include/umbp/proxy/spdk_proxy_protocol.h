// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// SPDK Proxy IPC Protocol — shared memory layout for multi-process SPDK access.
//
// Architecture:
//   spdk_proxy daemon (1 process) owns SPDK + NVMe device.
//   N rank processes communicate via POSIX shared memory.
//
// Lifecycle:
//   Leader (rank 0) auto-forks spdk_proxy daemon.
//   Daemon creates SHM, initializes SPDK, sets state=READY.
//   All ranks attach SHM, register via active_ranks + rank_pids.
//   On shutdown, ranks deregister; daemon exits when active_ranks==0
//   and shutdown_requested is set (or spawner process is dead).
//
// Memory layout:
//   [ProxyShmHeader]          — global state, offsets to channels and data
//   [RankChannel × max_ranks] — per-rank request/response ring buffers
//   [DataRegion]              — bulk data transfer area (per-rank sections)
//   [CacheRingControl]        — ring buffer write pointer and capacity
//   [CacheRingIndex]          — hash index: key → ring position
//   [CacheRingData]           — ring buffer data (FIFO, daemon writes, clients read)
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
static constexpr uint32_t kProxyVersion = 3;
static constexpr uint32_t kMaxKeyLen = 256;
static constexpr uint32_t kRingSize = 256;
static constexpr uint32_t kMaxRanks = 16;
static constexpr size_t kDefaultDataRegionPerRank = 256ULL * 1024 * 1024;  // 256MB
static constexpr size_t kDmaAlignment = 4096;  // NVMe sector alignment for zero-copy DMA
static constexpr size_t kHugepageSize = 2ULL * 1024 * 1024;  // 2MB hugepage

static constexpr const char* kDefaultShmName = "/umbp_spdk_proxy";

// Heartbeat stale threshold: clients treat the proxy as dead if
// proxy_heartbeat_ms hasn't been updated for this many milliseconds.
static constexpr uint64_t kHeartbeatStaleMs = 5000;

// ---- Epoch-millisecond helper (cross-process safe) ----
inline uint64_t NowEpochMs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

// ---- Enums ----

enum class ProxyState : uint32_t {
    UNINIT = 0,
    READY = 1,
    SHUTDOWN = 2,
    ERROR = 3,
};

enum class SlotState : uint32_t {
    EMPTY = 0,
    PENDING = 1,
    COMPLETED = 2,
};

enum class RequestType : uint32_t {
    NOP = 0,
    WRITE = 1,
    READ = 2,
    EXISTS = 3,
    EVICT = 4,
    CLEAR = 5,
    CAPACITY = 6,
    GET_LRU_KEY = 7,
    GET_LRU_CANDIDATES = 8,
    BATCH_WRITE = 9,
    BATCH_READ = 10,
    // Flush: no-op that serializes after the previous batch request.
    // In Write-Back mode, guarantees all prior NVMe writes for this rank
    // are complete and visible in entries_ before returning COMPLETED.
    FLUSH = 11,
};

enum class ResultCode : int32_t {
    OK = 0,
    NOT_FOUND = -1,
    NO_SPACE = -2,
    ERROR = -3,
    PERMISSION_DENIED = -4,
};

// ---- Per-slot request structure (in ring buffer) ----
//
// Lifecycle: EMPTY → (rank writes fields) → PENDING → (proxy processes) → COMPLETED → (rank reads) → EMPTY
//
struct alignas(64) RingSlot {
    std::atomic<uint32_t> state;    // SlotState

    // Request fields (written by rank, read by proxy)
    uint32_t type;                  // RequestType
    uint64_t seq_id;
    uint32_t key_len;
    char key[kMaxKeyLen];
    uint64_t data_offset;           // offset within rank's data region
    uint64_t data_size;
    uint32_t flags;                 // reserved
    uint32_t batch_count;           // for BATCH_WRITE/READ: number of entries

    // Response fields (written by proxy, read by rank)
    int32_t result;                 // ResultCode
    uint64_t result_size;           // for READ: actual bytes written to data region
    uint64_t result_aux;            // for CAPACITY: total_bytes; for GET_LRU_CANDIDATES: count

    char _pad[4];
};

static_assert(sizeof(RingSlot) % 64 == 0, "RingSlot must be cache-line aligned");

// ---- Per-rank channel ----
struct RankChannel {
    // Ring buffer: rank writes at head, proxy consumes at tail
    alignas(64) std::atomic<uint32_t> head;     // next slot for rank to write
    alignas(64) std::atomic<uint32_t> tail;     // next slot for proxy to process
    alignas(64) uint32_t rank_id;
    uint32_t is_leader;                         // 1 = leader (can write), 0 = follower (read-only)
    uint32_t connected;                         // 1 = rank is connected
    std::atomic<uint32_t> owner_pid;            // 0 = free slot, >0 = PID that owns this slot (CAS allocation)
    char _pad[48];

    RingSlot slots[kRingSize];
};

// ---- Batch descriptor (stored in data region for BATCH_WRITE/READ) ----
struct BatchEntry {
    uint16_t key_len;
    char key[kMaxKeyLen];
    uint64_t data_offset;   // relative to batch data start
    uint64_t data_size;
    uint8_t result;         // written by proxy: 0=fail, 1=ok
    char _pad[5];
};

struct BatchDescriptor {
    uint32_t count;
    uint32_t _reserved;
    uint64_t total_data_size;
    // Non-zero: daemon placed read data directly in ring buffer at this
    // absolute offset.  Client should read from ring instead of data_region.
    // Zero (default): data is in data_region as before.
    uint64_t ring_data_base;
    // Streaming signals — separate cache lines to avoid false sharing.
    alignas(64) std::atomic<uint32_t> items_ready;
    alignas(64) std::atomic<uint32_t> items_done;
    alignas(64) std::atomic<uint64_t> bytes_ready;
    alignas(64) std::atomic<uint64_t> bytes_done;

    BatchEntry entries[];
};

// ---- SHM shared cache: ring buffer with hash index ----
//
// Single-producer (daemon) multi-consumer (rank clients).
// Daemon appends data to a ring buffer and updates a hash index table.
// Clients look up the index, validate that the ring region is still live,
// then memcpy out.
//
// Index entry seqlock:
//   gen == 0       → empty
//   gen odd        → daemon is updating (client must bail)
//   gen even != 0  → stable; client may read
//
// Ring liveness: data at absolute offset P is live iff write_pos - P <= capacity.
//
static constexpr size_t kCacheRingAlign = 64;  // data alignment in ring buffer

struct alignas(64) CacheRingControl {
    std::atomic<uint64_t> write_pos;  // absolute monotonic write position (daemon-only)
    uint64_t capacity;                // ring data capacity in bytes
    char _pad[48];
};

struct alignas(64) CacheRingIndexEntry {
    std::atomic<uint64_t> gen;  // seqlock generation
    uint32_t key_len;
    uint32_t _pad0;
    uint64_t ring_offset;       // absolute position when written
    uint64_t data_size;
    char key[kMaxKeyLen];
    char _pad1[32];
};
static_assert(sizeof(CacheRingIndexEntry) == 320, "CacheRingIndexEntry size mismatch");

inline uint32_t CacheIndexHash(const char* key, uint32_t key_len, uint32_t num_slots) {
    uint64_t h = 14695981039346656037ULL;
    for (uint32_t i = 0; i < key_len; ++i) {
        h ^= static_cast<uint64_t>(static_cast<unsigned char>(key[i]));
        h *= 1099511628211ULL;
    }
    return static_cast<uint32_t>(h % num_slots);
}

// ---- Global shared memory header ----
struct ProxyShmHeader {
    // ---- Identity ----
    uint64_t magic;
    uint32_t version;
    std::atomic<uint32_t> state;    // ProxyState

    // ---- Device info ----
    uint32_t max_ranks;
    uint32_t block_size;            // NVMe block size (set by proxy after SPDK init)
    uint32_t hugepage;              // 1 = data region is hugepage-backed (zero-copy DMA capable)
    uint32_t _pad0;
    uint64_t bdev_size;             // NVMe device size in bytes

    // ---- Layout offsets (from start of shared memory) ----
    uint64_t channels_offset;
    uint64_t data_region_offset;
    uint64_t data_region_per_rank;
    uint64_t total_shm_size;

    // ---- Capacity info (updated by proxy) ----
    std::atomic<uint64_t> capacity_used;
    std::atomic<uint64_t> capacity_total;

    // ---- Lifecycle management ----
    std::atomic<uint32_t> proxy_pid;          // PID of proxy daemon
    std::atomic<uint32_t> spawner_pid;        // PID of the process that auto-spawned proxy (0 = manual)
    std::atomic<uint32_t> active_ranks;       // Number of currently connected rank processes
    std::atomic<uint32_t> shutdown_requested; // 1 = graceful shutdown requested by spawner
    std::atomic<uint64_t> proxy_heartbeat_ms; // Epoch ms, updated ~every 500ms by proxy

    // Per-rank PID tracking: proxy periodically checks if rank PIDs are alive
    // to detect crashed ranks and reclaim their resources.
    std::atomic<uint32_t> rank_pids[kMaxRanks];

    // ---- Shared cache (ring buffer + hash index, appended after data regions) ----
    uint64_t cache_region_offset;     // 0 = no cache; else offset to CacheRingControl
    uint64_t cache_ring_capacity;     // ring data area capacity in bytes
    uint32_t cache_index_slots;       // number of hash index entries
    uint32_t _pad_cache;

    char _reserved2[16];
};

// ---- Helper: compute total shared memory size ----
inline size_t ComputeShmSize(uint32_t max_ranks, size_t data_per_rank,
                             size_t cache_total_bytes = 0) {
    size_t header_size = sizeof(ProxyShmHeader);
    size_t channels_offset = (header_size + 4095) & ~4095ULL;
    size_t channels_size = sizeof(RankChannel) * max_ranks;
    size_t data_offset = (channels_offset + channels_size + 4095) & ~4095ULL;
    size_t data_size = data_per_rank * max_ranks;
    size_t base = data_offset + data_size;
    if (cache_total_bytes > 0) {
        base = (base + 4095) & ~4095ULL;
        base += cache_total_bytes;
    }
    return base;
}

// ---- Helper: get channel pointer from header ----
inline RankChannel* GetChannel(void* shm_base, const ProxyShmHeader* hdr, uint32_t rank) {
    if (rank >= hdr->max_ranks) return nullptr;
    auto* base = static_cast<char*>(shm_base);
    return reinterpret_cast<RankChannel*>(base + hdr->channels_offset) + rank;
}

inline const RankChannel* GetChannel(const void* shm_base, const ProxyShmHeader* hdr, uint32_t rank) {
    if (rank >= hdr->max_ranks) return nullptr;
    auto* base = static_cast<const char*>(shm_base);
    return reinterpret_cast<const RankChannel*>(base + hdr->channels_offset) + rank;
}

// ---- Helper: get data region pointer for a rank ----
inline void* GetDataRegion(void* shm_base, const ProxyShmHeader* hdr, uint32_t rank) {
    if (rank >= hdr->max_ranks) return nullptr;
    auto* base = static_cast<char*>(shm_base);
    return base + hdr->data_region_offset + rank * hdr->data_region_per_rank;
}

inline const void* GetDataRegion(const void* shm_base, const ProxyShmHeader* hdr, uint32_t rank) {
    if (rank >= hdr->max_ranks) return nullptr;
    auto* base = static_cast<const char*>(shm_base);
    return base + hdr->data_region_offset + rank * hdr->data_region_per_rank;
}

// ---- Helpers: ring buffer cache access ----
inline CacheRingControl* GetCacheRingControl(void* shm_base, const ProxyShmHeader* hdr) {
    if (hdr->cache_region_offset == 0) return nullptr;
    return reinterpret_cast<CacheRingControl*>(
        static_cast<char*>(shm_base) + hdr->cache_region_offset);
}

inline const CacheRingControl* GetCacheRingControl(const void* shm_base,
                                                    const ProxyShmHeader* hdr) {
    if (hdr->cache_region_offset == 0) return nullptr;
    return reinterpret_cast<const CacheRingControl*>(
        static_cast<const char*>(shm_base) + hdr->cache_region_offset);
}

inline CacheRingIndexEntry* GetCacheIndex(void* shm_base, const ProxyShmHeader* hdr) {
    if (hdr->cache_region_offset == 0) return nullptr;
    return reinterpret_cast<CacheRingIndexEntry*>(
        static_cast<char*>(shm_base) + hdr->cache_region_offset + sizeof(CacheRingControl));
}

inline const CacheRingIndexEntry* GetCacheIndex(const void* shm_base,
                                                 const ProxyShmHeader* hdr) {
    if (hdr->cache_region_offset == 0) return nullptr;
    return reinterpret_cast<const CacheRingIndexEntry*>(
        static_cast<const char*>(shm_base) + hdr->cache_region_offset + sizeof(CacheRingControl));
}

inline size_t CacheRingDataOffset(const ProxyShmHeader* hdr) {
    if (hdr->cache_region_offset == 0) return 0;
    size_t idx_end = hdr->cache_region_offset + sizeof(CacheRingControl)
                   + static_cast<size_t>(hdr->cache_index_slots) * sizeof(CacheRingIndexEntry);
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

// Allocate a contiguous block in the ring buffer (CAS loop).
// Pads past the wrap-around boundary to guarantee contiguous memory.
// Returns the absolute ring offset, or UINT64_MAX if size is too large.
inline uint64_t AllocRingContiguous(CacheRingControl* ctrl, size_t size) {
    if (!ctrl || size == 0) return UINT64_MAX;
    uint64_t cap = ctrl->capacity;
    size_t aligned = (size + kCacheRingAlign - 1) & ~(kCacheRingAlign - 1);
    if (aligned > (cap * 3) / 4) return UINT64_MAX;

    uint64_t expected = ctrl->write_pos.load(std::memory_order_relaxed);
    while (true) {
        uint64_t alloc_start = expected;
        size_t off = static_cast<size_t>(expected % cap);
        if (off + aligned > cap)
            alloc_start = expected + (cap - off);
        uint64_t new_pos = alloc_start + aligned;
        if (ctrl->write_pos.compare_exchange_weak(
                expected, new_pos, std::memory_order_acq_rel))
            return alloc_start;
    }
}

}  // namespace proxy
}  // namespace umbp
