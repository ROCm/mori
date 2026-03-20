// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// SpdkProxyTier: TierBackend that communicates with spdk_proxy daemon via
// POSIX shared memory. Zero SPDK dependency on the client side.

#include "umbp/storage/spdk_proxy_tier.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

#include "umbp/common/log.h"

using namespace umbp::proxy;

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define CPU_PAUSE() _mm_pause()
#else
#define CPU_PAUSE() ((void)0)
#endif

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
SpdkProxyTier::SpdkProxyTier(const UMBPConfig& config)
    : TierBackend(StorageTier::LOCAL_SSD) {
    std::string shm_name = config.spdk_proxy_shm_name;
    if (shm_name.empty()) shm_name = kDefaultShmName;
    rank_id_ = config.spdk_proxy_rank_id;

    int rc = shm_.Attach(shm_name);
    if (rc != 0) {
        UMBP_LOG_ERROR("SpdkProxyTier: cannot attach to SHM '%s' rc=%d",
                       shm_name.c_str(), rc);
        return;
    }

    auto* hdr = shm_.Header();
    if (hdr->state.load(std::memory_order_acquire) !=
        static_cast<uint32_t>(ProxyState::READY)) {
        UMBP_LOG_ERROR("SpdkProxyTier: proxy not READY (state=%u)",
                       hdr->state.load(std::memory_order_relaxed));
        shm_.Detach();
        return;
    }

    if (rank_id_ >= hdr->max_ranks) {
        UMBP_LOG_ERROR("SpdkProxyTier: rank_id %u >= max_ranks %u",
                       rank_id_, hdr->max_ranks);
        shm_.Detach();
        return;
    }

    auto* ch = shm_.Channel(rank_id_);
    ch->rank_id = rank_id_;
    ch->connected = 1;

    connected_ = true;
    UMBP_LOG_INFO("SpdkProxyTier: connected rank=%u shm='%s' "
                  "bdev_size=%zuMB block_size=%u data_region=%zuMB",
                  rank_id_, shm_name.c_str(),
                  static_cast<size_t>(hdr->bdev_size / (1024 * 1024)),
                  hdr->block_size,
                  static_cast<size_t>(hdr->data_region_per_rank / (1024 * 1024)));
}

SpdkProxyTier::~SpdkProxyTier() {
    if (connected_ && shm_.IsValid()) {
        auto* ch = shm_.Channel(rank_id_);
        if (ch) ch->connected = 0;
    }
    shm_.Detach();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
uint32_t SpdkProxyTier::NextSeqId() const {
    return static_cast<uint32_t>(++seq_counter_);
}

// ---------------------------------------------------------------------------
// Single-op submit/wait
// ---------------------------------------------------------------------------
ResultCode SpdkProxyTier::SubmitAndWait(
    RequestType type, const std::string& key,
    const void* write_data, size_t write_size,
    void* read_buf, size_t read_buf_size,
    uint64_t* out_result_size,
    uint64_t* out_result_aux) const {
    if (!connected_) return ResultCode::ERROR;

    std::lock_guard<std::mutex> lk(submit_mu_);

    auto* ch = shm_.Channel(rank_id_);
    uint32_t h = ch->head.load(std::memory_order_relaxed);
    uint32_t t = ch->tail.load(std::memory_order_acquire);

    // Check ring is not full
    if (((h + 1) % kRingSize) == t) {
        UMBP_LOG_ERROR("SpdkProxyTier: ring full");
        return ResultCode::ERROR;
    }

    auto& slot = ch->slots[h % kRingSize];

    // Copy data to SHM data region if writing
    void* data_region = shm_.DataRegion(rank_id_);
    if (write_data && write_size > 0) {
        size_t max_size = shm_.Header()->data_region_per_rank;
        if (write_size > max_size) {
            UMBP_LOG_ERROR("SpdkProxyTier: data too large %zu > %zu",
                           write_size, max_size);
            return ResultCode::ERROR;
        }
        std::memcpy(data_region, write_data, write_size);
    }

    // Fill slot
    slot.type = static_cast<uint32_t>(type);
    slot.seq_id = NextSeqId();
    slot.key_len = std::min(static_cast<uint32_t>(key.size()), kMaxKeyLen - 1);
    std::memcpy(slot.key, key.data(), slot.key_len);
    slot.key[slot.key_len] = '\0';
    slot.data_offset = 0;
    slot.data_size = write_size;
    slot.batch_count = 0;
    slot.flags = 0;

    // Publish
    slot.state.store(static_cast<uint32_t>(SlotState::PENDING),
                     std::memory_order_release);
    ch->head.store((h + 1) % kRingSize, std::memory_order_release);

    // Poll for completion
    while (true) {
        uint32_t st = slot.state.load(std::memory_order_acquire);
        if (st == static_cast<uint32_t>(SlotState::COMPLETED)) break;
        CPU_PAUSE();
    }

    // Read results
    ResultCode rc = static_cast<ResultCode>(slot.result);
    if (out_result_size) *out_result_size = slot.result_size;
    if (out_result_aux) *out_result_aux = slot.result_aux;

    // Copy read data from SHM to caller's buffer
    if (read_buf && slot.result_size > 0 && rc == ResultCode::OK) {
        size_t copy_sz = std::min(static_cast<size_t>(slot.result_size), read_buf_size);
        std::memcpy(read_buf, data_region, copy_sz);
    }

    slot.state.store(static_cast<uint32_t>(SlotState::EMPTY),
                     std::memory_order_release);
    return rc;
}

// ---------------------------------------------------------------------------
// Batch submit (BATCH_WRITE / BATCH_READ)
// ---------------------------------------------------------------------------
std::vector<bool> SpdkProxyTier::SubmitBatch(
    RequestType type,
    const std::vector<std::string>& keys,
    const std::vector<const void*>& data_ptrs,
    const std::vector<uintptr_t>& dst_ptrs,
    const std::vector<size_t>& sizes) const {

    const int count = static_cast<int>(keys.size());
    std::vector<bool> results(count, false);
    if (!connected_ || count == 0) return results;

    std::lock_guard<std::mutex> lk(submit_mu_);

    auto* hdr = shm_.Header();
    void* data_region = shm_.DataRegion(rank_id_);
    const size_t region_size = hdr->data_region_per_rank;

    // Process in sub-batches if total data exceeds region capacity
    int base = 0;
    while (base < count) {
        // Determine how many keys fit in this sub-batch
        size_t desc_overhead = sizeof(BatchDescriptor);
        size_t data_start = 0;
        int sub_count = 0;
        size_t total_data = 0;

        for (int i = base; i < count; ++i) {
            size_t entry_overhead = sizeof(BatchEntry);
            size_t new_desc = desc_overhead + (sub_count + 1) * entry_overhead;
            // Align data start to 4KB for DMA zero-copy compatibility
            size_t new_data_start = (new_desc + kDmaAlignment - 1) & ~(kDmaAlignment - 1);
            // Each key's data is 4KB-aligned for NVMe DMA
            size_t aligned_data = (sizes[i] + kDmaAlignment - 1) & ~(kDmaAlignment - 1);
            size_t new_total = new_data_start + total_data + aligned_data;
            if (new_total > region_size && sub_count > 0) break;
            sub_count++;
            total_data += aligned_data;
            data_start = new_data_start;
        }

        if (sub_count == 0) {
            results[base] = false;
            base++;
            continue;
        }

        // Build BatchDescriptor metadata (keys + offsets, NO data yet for writes)
        auto* desc = static_cast<BatchDescriptor*>(data_region);
        desc->count = sub_count;
        desc->total_data_size = total_data;
        desc->items_ready.store(0, std::memory_order_relaxed);
        desc->items_done.store(0, std::memory_order_relaxed);

        size_t data_cursor = 0;
        char* data_base = static_cast<char*>(data_region) + data_start;

        for (int i = 0; i < sub_count; ++i) {
            int gi = base + i;
            auto& entry = desc->entries[i];
            entry.key_len = std::min(static_cast<uint16_t>(keys[gi].size()),
                                     static_cast<uint16_t>(kMaxKeyLen - 1));
            std::memcpy(entry.key, keys[gi].data(), entry.key_len);
            entry.key[entry.key_len] = '\0';
            entry.data_offset = data_cursor;
            entry.data_size = sizes[gi];
            entry.result = 0;
            size_t aligned = (sizes[gi] + kDmaAlignment - 1) & ~(kDmaAlignment - 1);
            data_cursor += aligned;
        }

        // Submit ring slot — metadata ready, proxy can start Phase 1 (allocate)
        auto* ch = shm_.Channel(rank_id_);
        uint32_t h = ch->head.load(std::memory_order_relaxed);
        while (((h + 1) % kRingSize) ==
               ch->tail.load(std::memory_order_acquire)) {
            CPU_PAUSE();
        }

        auto& slot = ch->slots[h % kRingSize];
        slot.type = static_cast<uint32_t>(type);
        slot.seq_id = NextSeqId();
        slot.key_len = 0;
        slot.data_offset = 0;
        slot.data_size = data_start + data_cursor;
        slot.batch_count = sub_count;
        slot.flags = 0;

        slot.state.store(static_cast<uint32_t>(SlotState::PENDING),
                         std::memory_order_release);
        ch->head.store((h + 1) % kRingSize, std::memory_order_release);

        if (type == RequestType::BATCH_WRITE) {
            // STREAMING WRITE: copy data per-key, incrementing items_ready.
            // Proxy starts NVMe as soon as each key is ready.
            for (int i = 0; i < sub_count; ++i) {
                int gi = base + i;
                if (!data_ptrs.empty() && data_ptrs[gi] != nullptr) {
                    std::memcpy(data_base + desc->entries[i].data_offset,
                                data_ptrs[gi], sizes[gi]);
                }
                desc->items_ready.store(static_cast<uint32_t>(gi + 1),
                                        std::memory_order_release);
            }

            // Wait for proxy to finish all NVMe writes + metadata update
            while (true) {
                uint32_t st = slot.state.load(std::memory_order_acquire);
                if (st == static_cast<uint32_t>(SlotState::COMPLETED)) break;
                CPU_PAUSE();
            }

            for (int i = 0; i < sub_count; ++i)
                results[base + i] = (desc->entries[i].result != 0);

        } else {
            // STREAMING READ: proxy signals items_done per key.
            // Copy each key's data as soon as its NVMe read completes.
            int copied = 0;
            while (copied < sub_count) {
                uint32_t done = desc->items_done.load(std::memory_order_acquire);
                while (copied < static_cast<int>(done) && copied < sub_count) {
                    int gi = base + copied;
                    if (!dst_ptrs.empty()) {
                        void* dst = reinterpret_cast<void*>(dst_ptrs[gi]);
                        std::memcpy(dst,
                                    data_base + desc->entries[copied].data_offset,
                                    sizes[gi]);
                    }
                    results[gi] = true;
                    ++copied;
                }
                if (copied < sub_count) CPU_PAUSE();
            }

            // Wait for overall completion
            while (true) {
                uint32_t st = slot.state.load(std::memory_order_acquire);
                if (st == static_cast<uint32_t>(SlotState::COMPLETED)) break;
                CPU_PAUSE();
            }

            // Check for failed items
            for (int i = 0; i < sub_count; ++i) {
                int gi = base + i;
                results[gi] = (desc->entries[i].result != 0);
            }
        }

        slot.state.store(static_cast<uint32_t>(SlotState::EMPTY),
                         std::memory_order_release);
        base += sub_count;
    }

    return results;
}

// ---------------------------------------------------------------------------
// TierBackend interface
// ---------------------------------------------------------------------------
bool SpdkProxyTier::Write(const std::string& key, const void* data, size_t size) {
    auto rc = SubmitAndWait(RequestType::WRITE, key, data, size, nullptr, 0);
    return rc == ResultCode::OK;
}

bool SpdkProxyTier::ReadIntoPtr(const std::string& key, uintptr_t dst_ptr,
                                size_t size) {
    uint64_t actual_size = 0;
    auto rc = SubmitAndWait(RequestType::READ, key, nullptr, 0,
                            reinterpret_cast<void*>(dst_ptr), size,
                            &actual_size);
    return rc == ResultCode::OK;
}

bool SpdkProxyTier::Exists(const std::string& key) const {
    auto rc = SubmitAndWait(RequestType::EXISTS, key, nullptr, 0, nullptr, 0);
    return rc == ResultCode::OK;
}

bool SpdkProxyTier::Evict(const std::string& key) {
    auto rc = SubmitAndWait(RequestType::EVICT, key, nullptr, 0, nullptr, 0);
    return rc == ResultCode::OK;
}

std::pair<size_t, size_t> SpdkProxyTier::Capacity() const {
    uint64_t result_size = 0, result_aux = 0;
    auto rc = SubmitAndWait(RequestType::CAPACITY, "", nullptr, 0, nullptr, 0,
                            &result_size, &result_aux);
    if (rc != ResultCode::OK) return {0, 0};
    return {static_cast<size_t>(result_size), static_cast<size_t>(result_aux)};
}

void SpdkProxyTier::Clear() {
    SubmitAndWait(RequestType::CLEAR, "", nullptr, 0, nullptr, 0);
}

std::vector<bool> SpdkProxyTier::BatchWrite(
    const std::vector<std::string>& keys,
    const std::vector<const void*>& data_ptrs,
    const std::vector<size_t>& sizes) {
    return SubmitBatch(RequestType::BATCH_WRITE, keys, data_ptrs, {}, sizes);
}

std::vector<bool> SpdkProxyTier::BatchReadIntoPtr(
    const std::vector<std::string>& keys,
    const std::vector<uintptr_t>& dst_ptrs,
    const std::vector<size_t>& sizes) {
    return SubmitBatch(RequestType::BATCH_READ, keys, {}, dst_ptrs, sizes);
}

std::string SpdkProxyTier::GetLRUKey() const {
    if (!connected_) return "";
    // Use a simple single-key request; proxy returns key in data region
    uint64_t result_size = 0;
    char buf[kMaxKeyLen] = {};
    auto rc = SubmitAndWait(RequestType::GET_LRU_KEY, "", nullptr, 0,
                            buf, sizeof(buf), &result_size);
    if (rc != ResultCode::OK || result_size == 0) return "";
    return std::string(buf, std::min(static_cast<size_t>(result_size),
                                     sizeof(buf) - 1));
}

std::vector<std::string> SpdkProxyTier::GetLRUCandidates(size_t max_candidates) const {
    // Encode max_candidates in data_size field
    uint64_t result_aux = 0;
    auto* hdr = shm_.Header();
    (void)hdr;

    // For simplicity, delegate to multiple GetLRUKey calls or use a batch protocol.
    // Simple implementation: just return single LRU key.
    std::vector<std::string> result;
    std::string k = GetLRUKey();
    if (!k.empty()) result.push_back(std::move(k));
    return result;
}
