// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// SPDK Proxy IPC Protocol — shared memory layout for multi-process SPDK access.
//
// Architecture:
//   spdk_proxy daemon (1 process) owns SPDK + NVMe device.
//   N rank processes communicate via POSIX shared memory.
//
// Memory layout:
//   [ProxyShmHeader]          — global state, offsets to channels and data
//   [RankChannel × max_ranks] — per-rank request/response ring buffers
//   [DataRegion]              — bulk data transfer area (per-rank sections)
//
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace umbp {
namespace proxy {

// ---- Constants ----
static constexpr uint64_t kProxyShmMagic = 0x554D4250534B5058ULL;  // "UMBPSKPX"
static constexpr uint32_t kProxyVersion = 1;
static constexpr uint32_t kMaxKeyLen = 256;
static constexpr uint32_t kRingSize = 256;
static constexpr uint32_t kMaxRanks = 16;
static constexpr size_t kDefaultDataRegionPerRank = 256ULL * 1024 * 1024;  // 256MB
static constexpr size_t kDmaAlignment = 4096;  // NVMe sector alignment for zero-copy DMA
static constexpr size_t kHugepageSize = 2ULL * 1024 * 1024;  // 2MB hugepage

// Default shared memory name
static constexpr const char* kDefaultShmName = "/umbp_spdk_proxy";

// ---- Enums ----

enum class ProxyState : uint32_t {
    UNINIT = 0,
    READY = 1,
    SHUTDOWN = 2,
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
    char _pad[52];

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
    // Streaming signals — separate cache lines to avoid false sharing
    // Write streaming: client increments after copying each key's data
    alignas(64) std::atomic<uint32_t> items_ready;
    // Read streaming: proxy increments after each key's NVMe read completes
    alignas(64) std::atomic<uint32_t> items_done;

    BatchEntry entries[];
};

// ---- Global shared memory header ----
struct ProxyShmHeader {
    uint64_t magic;
    uint32_t version;
    std::atomic<uint32_t> state;    // ProxyState

    uint32_t max_ranks;
    uint32_t block_size;            // NVMe block size (set by proxy after SPDK init)
    uint32_t hugepage;              // 1 = data region is hugepage-backed (zero-copy DMA capable)
    uint32_t _pad0;
    uint64_t bdev_size;             // NVMe device size in bytes

    // Layout offsets (from start of shared memory)
    uint64_t channels_offset;
    uint64_t data_region_offset;
    uint64_t data_region_per_rank;
    uint64_t total_shm_size;

    // Capacity info (updated by proxy)
    std::atomic<uint64_t> capacity_used;
    std::atomic<uint64_t> capacity_total;

    char _reserved[128];
};

// ---- Helper: compute total shared memory size ----
inline size_t ComputeShmSize(uint32_t max_ranks, size_t data_per_rank) {
    size_t header_size = sizeof(ProxyShmHeader);
    // Align channels to page boundary
    size_t channels_offset = (header_size + 4095) & ~4095ULL;
    size_t channels_size = sizeof(RankChannel) * max_ranks;
    // Align data region to page boundary
    size_t data_offset = (channels_offset + channels_size + 4095) & ~4095ULL;
    size_t data_size = data_per_rank * max_ranks;
    return data_offset + data_size;
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

}  // namespace proxy
}  // namespace umbp
