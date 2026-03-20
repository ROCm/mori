// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// SpdkProxyTier: TierBackend that communicates with spdk_proxy daemon via
// POSIX shared memory. Zero SPDK dependency on the client side.
//
// Lifecycle:
//   - On construction: attaches SHM, registers rank (active_ranks++, rank_pids).
//   - On destruction: deregisters rank (rank_pids=0, active_ranks--).
//   - All busy-poll loops check proxy heartbeat to avoid hanging on dead proxy.

#include "umbp/storage/spdk_proxy_tier.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

#include "umbp/common/log.h"

#ifdef __linux__
#include <unistd.h>
#endif

using namespace umbp::proxy;

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define CPU_PAUSE() _mm_pause()
#else
#define CPU_PAUSE() ((void)0)
#endif

// ---------------------------------------------------------------------------
// Construction / Destruction
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

    // Register this rank
    auto* ch = shm_.Channel(rank_id_);
    ch->rank_id = rank_id_;
    ch->connected = 1;

#ifdef __linux__
    hdr->rank_pids[rank_id_].store(static_cast<uint32_t>(getpid()),
                                   std::memory_order_relaxed);
#endif
    hdr->active_ranks.fetch_add(1, std::memory_order_release);

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
        auto* hdr = shm_.Header();
        auto* ch = shm_.Channel(rank_id_);
        if (ch) ch->connected = 0;

        hdr->rank_pids[rank_id_].store(0, std::memory_order_relaxed);
        hdr->active_ranks.fetch_sub(1, std::memory_order_release);

        UMBP_LOG_INFO("SpdkProxyTier: disconnected rank=%u", rank_id_);
    }
    shm_.Detach();
}

// ---------------------------------------------------------------------------
// WaitForProxy — static, called before constructing SpdkProxyTier
// ---------------------------------------------------------------------------
bool SpdkProxyTier::WaitForProxy(const std::string& shm_name, int timeout_ms) {
    auto start = std::chrono::steady_clock::now();

    while (true) {
        ProxyShmRegion probe;
        int rc = probe.Attach(shm_name);
        if (rc == 0) {
            auto* hdr = probe.Header();
            uint32_t st = hdr->state.load(std::memory_order_acquire);
            if (st == static_cast<uint32_t>(ProxyState::READY)) {
                return true;
            }
            if (st == static_cast<uint32_t>(ProxyState::ERROR)) {
                UMBP_LOG_ERROR("SpdkProxyTier: proxy reported ERROR state");
                return false;
            }
        }

        auto elapsed = std::chrono::steady_clock::now() - start;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
                >= timeout_ms) {
            UMBP_LOG_ERROR("SpdkProxyTier: timed out waiting for proxy READY (%d ms)",
                           timeout_ms);
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// ---------------------------------------------------------------------------
// IsProxyAlive — check proxy heartbeat
// ---------------------------------------------------------------------------
bool SpdkProxyTier::IsProxyAlive() const {
    if (!shm_.IsValid()) return false;
    auto* hdr = shm_.Header();
    uint64_t hb = hdr->proxy_heartbeat_ms.load(std::memory_order_relaxed);
    if (hb == 0) return true;  // Proxy hasn't started heartbeat yet
    uint64_t now = NowEpochMs();
    return (now - hb) < kHeartbeatStaleMs;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
uint32_t SpdkProxyTier::NextSeqId() const {
    return static_cast<uint32_t>(++seq_counter_);
}

// ---------------------------------------------------------------------------
// Single-op submit/wait (with heartbeat timeout)
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

    if (((h + 1) % kRingSize) == t) {
        UMBP_LOG_ERROR("SpdkProxyTier: ring full");
        return ResultCode::ERROR;
    }

    auto& slot = ch->slots[h % kRingSize];

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

    slot.type = static_cast<uint32_t>(type);
    slot.seq_id = NextSeqId();
    slot.key_len = std::min(static_cast<uint32_t>(key.size()), kMaxKeyLen - 1);
    std::memcpy(slot.key, key.data(), slot.key_len);
    slot.key[slot.key_len] = '\0';
    slot.data_offset = 0;
    slot.data_size = write_size;
    slot.batch_count = 0;
    slot.flags = 0;

    slot.state.store(static_cast<uint32_t>(SlotState::PENDING),
                     std::memory_order_release);
    ch->head.store((h + 1) % kRingSize, std::memory_order_release);

    // Poll for completion with heartbeat safety
    int spin = 0;
    while (true) {
        uint32_t st = slot.state.load(std::memory_order_acquire);
        if (st == static_cast<uint32_t>(SlotState::COMPLETED)) break;
        CPU_PAUSE();
        if (++spin % 8192 == 0 && !IsProxyAlive()) {
            UMBP_LOG_ERROR("SpdkProxyTier: proxy heartbeat stale, aborting");
            slot.state.store(static_cast<uint32_t>(SlotState::EMPTY),
                             std::memory_order_release);
            return ResultCode::ERROR;
        }
    }

    ResultCode rc = static_cast<ResultCode>(slot.result);
    if (out_result_size) *out_result_size = slot.result_size;
    if (out_result_aux) *out_result_aux = slot.result_aux;

    if (read_buf && slot.result_size > 0 && rc == ResultCode::OK) {
        size_t copy_sz = std::min(static_cast<size_t>(slot.result_size), read_buf_size);
        std::memcpy(read_buf, data_region, copy_sz);
    }

    slot.state.store(static_cast<uint32_t>(SlotState::EMPTY),
                     std::memory_order_release);
    return rc;
}

// ---------------------------------------------------------------------------
// Batch submit (with streaming, heartbeat timeout, and correct indices)
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

    int base = 0;
    while (base < count) {
        size_t desc_overhead = sizeof(BatchDescriptor);
        size_t data_start = 0;
        int sub_count = 0;
        size_t total_data = 0;

        for (int i = base; i < count; ++i) {
            size_t entry_overhead = sizeof(BatchEntry);
            size_t new_desc = desc_overhead + (sub_count + 1) * entry_overhead;
            size_t new_data_start = (new_desc + kDmaAlignment - 1) & ~(kDmaAlignment - 1);
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

        // Build BatchDescriptor metadata
        auto* desc = static_cast<BatchDescriptor*>(data_region);
        desc->count = sub_count;
        desc->total_data_size = total_data;
        desc->items_ready.store(0, std::memory_order_relaxed);
        desc->items_done.store(0, std::memory_order_relaxed);
        desc->bytes_ready.store(0, std::memory_order_relaxed);
        desc->bytes_done.store(0, std::memory_order_relaxed);

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

        // Submit ring slot
        auto* ch = shm_.Channel(rank_id_);
        uint32_t h = ch->head.load(std::memory_order_relaxed);

        int ring_spin = 0;
        while (((h + 1) % kRingSize) ==
               ch->tail.load(std::memory_order_acquire)) {
            CPU_PAUSE();
            if (++ring_spin % 8192 == 0 && !IsProxyAlive()) {
                UMBP_LOG_ERROR("SpdkProxyTier: proxy dead while waiting for ring slot");
                return results;
            }
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
            // STREAMING WRITE: copy data in 2MB chunks, update bytes_ready
            // so daemon can start NVMe writes before the full item is copied.
            constexpr size_t kCopyChunk = 2ULL * 1024 * 1024;
            for (int i = 0; i < sub_count; ++i) {
                int gi = base + i;
                if (!data_ptrs.empty() && data_ptrs[gi] != nullptr) {
                    const char* src = static_cast<const char*>(data_ptrs[gi]);
                    char* dst = data_base + desc->entries[i].data_offset;
                    size_t item_sz = sizes[gi];
                    size_t copied = 0;
                    while (copied < item_sz) {
                        size_t chunk = std::min(kCopyChunk, item_sz - copied);
                        std::memcpy(dst + copied, src + copied, chunk);
                        copied += chunk;
                        desc->bytes_ready.store(
                            desc->entries[i].data_offset + copied,
                            std::memory_order_release);
                    }
                }
                desc->items_ready.store(static_cast<uint32_t>(i + 1),
                                        std::memory_order_release);
            }

            // Wait for proxy to finish
            int spin = 0;
            while (true) {
                uint32_t st = slot.state.load(std::memory_order_acquire);
                if (st == static_cast<uint32_t>(SlotState::COMPLETED)) break;
                CPU_PAUSE();
                if (++spin % 8192 == 0 && !IsProxyAlive()) {
                    UMBP_LOG_ERROR("SpdkProxyTier: proxy dead during batch write");
                    slot.state.store(static_cast<uint32_t>(SlotState::EMPTY),
                                     std::memory_order_release);
                    return results;
                }
            }

            for (int i = 0; i < sub_count; ++i)
                results[base + i] = (desc->entries[i].result != 0);

        } else {
            // STREAMING READ: copy data in 2MB chunks using bytes_done,
            // overlapping client memcpy with daemon NVMe reads at sub-item
            // granularity. Falls back to items_done for small items.
            constexpr size_t kReadChunk = 2ULL * 1024 * 1024;
            int current_item = 0;
            size_t item_copied = 0;
            int spin = 0;

            while (current_item < sub_count) {
                int gi = base + current_item;
                auto& entry = desc->entries[current_item];
                size_t item_sz = sizes[gi];

                if (!dst_ptrs.empty() && item_sz > 0) {
                    char* dst = reinterpret_cast<char*>(dst_ptrs[gi]);
                    char* src = data_base + entry.data_offset;

                    while (item_copied < item_sz) {
                        size_t want = std::min(kReadChunk, item_sz - item_copied);
                        uint64_t need = entry.data_offset + item_copied + want;
                        uint64_t bd = desc->bytes_done.load(
                            std::memory_order_acquire);

                        if (bd >= need) {
                            std::memcpy(dst + item_copied, src + item_copied,
                                        want);
                            item_copied += want;
                            spin = 0;
                        } else {
                            CPU_PAUSE();
                            if (++spin % 8192 == 0 && !IsProxyAlive()) {
                                UMBP_LOG_ERROR(
                                    "SpdkProxyTier: proxy dead during batch read");
                                slot.state.store(
                                    static_cast<uint32_t>(SlotState::EMPTY),
                                    std::memory_order_release);
                                return results;
                            }
                        }
                    }
                }

                ++current_item;
                item_copied = 0;
            }

            // Wait for overall slot completion
            spin = 0;
            while (true) {
                uint32_t st = slot.state.load(std::memory_order_acquire);
                if (st == static_cast<uint32_t>(SlotState::COMPLETED)) break;
                CPU_PAUSE();
                if (++spin % 8192 == 0 && !IsProxyAlive()) {
                    UMBP_LOG_ERROR(
                        "SpdkProxyTier: proxy dead waiting for read completion");
                    slot.state.store(
                        static_cast<uint32_t>(SlotState::EMPTY),
                        std::memory_order_release);
                    return results;
                }
            }

            // Authoritative results from proxy
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
    uint64_t result_size = 0;
    char buf[kMaxKeyLen] = {};
    auto rc = SubmitAndWait(RequestType::GET_LRU_KEY, "", nullptr, 0,
                            buf, sizeof(buf), &result_size);
    if (rc != ResultCode::OK || result_size == 0) return "";
    return std::string(buf, std::min(static_cast<size_t>(result_size),
                                     sizeof(buf) - 1));
}

std::vector<std::string> SpdkProxyTier::GetLRUCandidates(size_t max_candidates) const {
    (void)max_candidates;
    std::vector<std::string> result;
    std::string k = GetLRUKey();
    if (!k.empty()) result.push_back(std::move(k));
    return result;
}
