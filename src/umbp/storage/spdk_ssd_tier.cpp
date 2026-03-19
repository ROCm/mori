// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// SpdkSsdTier: deep-queue NVMe pipeline via SPDK.

#include "umbp/storage/spdk_ssd_tier.h"

#include <algorithm>
#include <cstring>

#include "umbp/common/log.h"
#include "umbp/spdk/spdk_env.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define CPU_PAUSE() _mm_pause()
#else
#define CPU_PAUSE() ((void)0)
#endif

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------
SpdkSsdTier::SpdkSsdTier(const UMBPConfig& config)
    : TierBackend(StorageTier::SSD) {
    auto& env = umbp::SpdkEnv::Instance();
    if (!env.IsInitialized()) {
        umbp::SpdkEnvConfig ecfg;
        ecfg.bdev_name = config.spdk_bdev_name;
        ecfg.reactor_mask = config.spdk_reactor_mask;
        ecfg.mem_size_mb = config.spdk_mem_size_mb;
        ecfg.nvme_pci_addr = config.spdk_nvme_pci_addr;
        ecfg.nvme_ctrl_name = config.spdk_nvme_ctrl_name;

        int rc = env.Init(ecfg);
        if (rc != 0) {
            UMBP_LOG_ERROR("SpdkSsdTier: SpdkEnv init failed rc=%d, falling back", rc);
            return;
        }
    }

    block_size_ = env.GetBlockSize();
    if (block_size_ == 0) block_size_ = 4096;

    uint64_t device_size = env.GetBdevSize();
    capacity_ = std::min(config.ssd_capacity_bytes, static_cast<size_t>(device_size));

    allocator_ = umbp::offset_allocator::OffsetAllocator::createAligned(
        0, capacity_, block_size_);
    if (!allocator_) {
        UMBP_LOG_ERROR("SpdkSsdTier: OffsetAllocator creation failed");
        return;
    }

    int prewarm_count = std::min(kMaxQueueDepth, 256);
    size_t prewarm_size = std::min(static_cast<size_t>(2ULL * 1024 * 1024),
                                    capacity_ / 64);
    prewarm_size = AlignUp(prewarm_size);
    if (prewarm_size > 0 && prewarm_count > 0) {
        env.DmaPoolPrewarm(prewarm_size, prewarm_count);
    }

    initialized_ = true;
    UMBP_LOG_INFO("SpdkSsdTier: ready — capacity=%zuMB block_size=%u",
                  capacity_ / (1024 * 1024), block_size_);
}

SpdkSsdTier::~SpdkSsdTier() {
    Clear();
}

// ---------------------------------------------------------------------------
// LRU helpers (caller must hold mu_)
// ---------------------------------------------------------------------------
void SpdkSsdTier::TouchLRU(const std::string& key) {
    auto it = lru_iter_.find(key);
    if (it != lru_iter_.end()) {
        lru_list_.erase(it->second);
    }
    lru_list_.push_front(key);
    lru_iter_[key] = lru_list_.begin();
}

void SpdkSsdTier::RemoveLRU(const std::string& key) {
    auto it = lru_iter_.find(key);
    if (it != lru_iter_.end()) {
        lru_list_.erase(it->second);
        lru_iter_.erase(it);
    }
}

// ---------------------------------------------------------------------------
// Single-key Write — wrapper around BatchWrite for simplicity.
// ---------------------------------------------------------------------------
bool SpdkSsdTier::Write(const std::string& key, const void* data, size_t size) {
    std::vector<std::string> keys = {key};
    std::vector<const void*> ptrs = {data};
    std::vector<size_t> sizes = {size};
    auto results = BatchWrite(keys, ptrs, sizes);
    return !results.empty() && results[0];
}

// ---------------------------------------------------------------------------
// Single-key Read — wrapper around BatchReadIntoPtr.
// ---------------------------------------------------------------------------
bool SpdkSsdTier::ReadIntoPtr(const std::string& key, uintptr_t dst_ptr,
                               size_t size) {
    std::vector<std::string> keys = {key};
    std::vector<uintptr_t> ptrs = {dst_ptr};
    std::vector<size_t> sizes = {size};
    auto results = BatchReadIntoPtr(keys, ptrs, sizes);
    return !results.empty() && results[0];
}

// ---------------------------------------------------------------------------
// Exists / Evict / Capacity / Clear
// ---------------------------------------------------------------------------
bool SpdkSsdTier::Exists(const std::string& key) const {
    std::lock_guard<std::mutex> lk(mu_);
    return entries_.count(key) > 0;
}

bool SpdkSsdTier::Evict(const std::string& key) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = entries_.find(key);
    if (it == entries_.end()) return false;
    RemoveLRU(key);
    entries_.erase(it);  // handle destroyed → space freed via RAII
    return true;
}

std::pair<size_t, size_t> SpdkSsdTier::Capacity() const {
    if (!allocator_) return {0, capacity_};
    auto metrics = allocator_->get_metrics();
    return {metrics.allocated_size_, capacity_};
}

void SpdkSsdTier::Clear() {
    std::lock_guard<std::mutex> lk(mu_);
    entries_.clear();
    lru_list_.clear();
    lru_iter_.clear();
}

std::string SpdkSsdTier::GetLRUKey() const {
    std::lock_guard<std::mutex> lk(mu_);
    if (lru_list_.empty()) return "";
    return lru_list_.back();
}

std::vector<std::string> SpdkSsdTier::GetLRUCandidates(size_t max_candidates) const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> result;
    if (max_candidates == 0) max_candidates = 1;
    result.reserve(std::min(max_candidates, lru_list_.size()));
    for (auto it = lru_list_.rbegin();
         it != lru_list_.rend() && result.size() < max_candidates; ++it) {
        result.push_back(*it);
    }
    return result;
}

// ===========================================================================
// BatchWrite — deep-queue NVMe write pipeline
//
// Phase 1 (lock):   check existing keys, batch_allocate space
// Phase 2 (unlock): memcpy + submit + drain pipeline on calling thread
// Phase 3 (lock):   update entries_ + LRU
// ===========================================================================
std::vector<bool> SpdkSsdTier::BatchWrite(
    const std::vector<std::string>& keys,
    const std::vector<const void*>& data_ptrs,
    const std::vector<size_t>& sizes) {
    const int count = static_cast<int>(keys.size());
    std::vector<bool> results(count, false);
    if (!initialized_ || count == 0) return results;

    auto& env = umbp::SpdkEnv::Instance();

    // --- Phase 1: Allocate space (lock held) ---
    struct PendingItem {
        int idx;                 // index into input arrays
        size_t aligned_size;
        umbp::offset_allocator::OffsetAllocationHandle handle;
    };
    std::vector<PendingItem> pending;
    pending.reserve(count);

    {
        std::lock_guard<std::mutex> lk(mu_);

        std::vector<size_t> alloc_sizes;
        std::vector<int> new_indices;
        alloc_sizes.reserve(count);
        new_indices.reserve(count);

        for (int i = 0; i < count; ++i) {
            if (entries_.count(keys[i])) {
                results[i] = true;
                continue;
            }
            size_t aligned = AlignUp(sizes[i]);
            new_indices.push_back(i);
            alloc_sizes.push_back(aligned);
        }

        if (!alloc_sizes.empty()) {
            auto handles = allocator_->batch_allocate(alloc_sizes);
            for (size_t j = 0; j < handles.size(); ++j) {
                if (handles[j].has_value()) {
                    PendingItem item;
                    item.idx = new_indices[j];
                    item.aligned_size = alloc_sizes[j];
                    item.handle = std::move(handles[j].value());
                    pending.push_back(std::move(item));
                } else {
                    for (size_t k = j; k < new_indices.size(); ++k)
                        results[new_indices[k]] = false;
                    break;
                }
            }
        }
    }
    // mu_ released

    if (pending.empty()) return results;

    // --- Phase 2: Deep-queue I/O pipeline (no lock) ---
    const int pending_count = static_cast<int>(pending.size());
    const int qd = std::min(pending_count, kMaxQueueDepth);

    // Find max aligned size to allocate uniform DMA buffers
    size_t max_aligned = 0;
    for (auto& p : pending)
        max_aligned = std::max(max_aligned, p.aligned_size);

    auto dma_bufs = std::make_unique<void*[]>(qd);
    int got = env.DmaPoolAllocBatch(dma_bufs.get(), max_aligned, qd);
    if (got < qd) {
        UMBP_LOG_ERROR("SpdkSsdTier::BatchWrite: DMA alloc short %d/%d", got, qd);
        for (int i = got; i < qd; ++i)
            dma_bufs[i] = nullptr;
    }

    auto reqs = std::make_unique<umbp::SpdkIoRequest[]>(qd);
    std::vector<bool> io_ok(pending_count, false);

    int head = 0, tail = 0;
    while (tail < pending_count) {
        // Fill the pipeline up to QD
        while (head < pending_count && (head - tail) < qd) {
            int slot = head % qd;
            if (!dma_bufs[slot]) { ++head; continue; }

            auto& p = pending[head];
            int idx = p.idx;

            // memcpy on calling thread
            std::memcpy(dma_bufs[slot], data_ptrs[idx], sizes[idx]);
            if (p.aligned_size > sizes[idx])
                std::memset(static_cast<char*>(dma_bufs[slot]) + sizes[idx],
                            0, p.aligned_size - sizes[idx]);

            auto& req = reqs[slot];
            req.op = umbp::SpdkIoRequest::WRITE;
            req.buf = dma_bufs[slot];
            req.offset = p.handle.address();
            req.nbytes = p.aligned_size;
            req.src_data = nullptr;
            req.src_iov = nullptr;
            req.src_iovcnt = 0;
            req.dst_iov = nullptr;
            req.dst_iovcnt = 0;
            req.completed.store(false, std::memory_order_release);
            req.success = false;

            env.SubmitIoAsync(&req);
            ++head;
        }

        // Drain oldest completion
        if (tail < head) {
            int slot = tail % qd;
            while (!reqs[slot].completed.load(std::memory_order_acquire))
                CPU_PAUSE();
            io_ok[tail] = reqs[slot].success;
            ++tail;
        }
    }

    // Return DMA buffers to pool
    env.DmaPoolFreeBatch(dma_bufs.get(), max_aligned, qd);

    // --- Phase 3: Update metadata (lock held) ---
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (int j = 0; j < pending_count; ++j) {
            int idx = pending[j].idx;
            if (io_ok[j]) {
                Entry entry;
                entry.handle = std::move(pending[j].handle);
                entry.data_size = sizes[idx];

                auto it = entries_.find(keys[idx]);
                if (it != entries_.end()) {
                    it->second = std::move(entry);
                } else {
                    entries_.emplace(keys[idx], std::move(entry));
                }
                TouchLRU(keys[idx]);
                results[idx] = true;
            }
            // Failed: handle destroyed on scope exit → space freed
        }
    }

    return results;
}

// ===========================================================================
// BatchReadIntoPtr — deep-queue NVMe read pipeline
//
// Phase 1 (lock):   look up entries, collect offsets and sizes
// Phase 2 (unlock): submit reads + drain + memcpy to user on calling thread
// Phase 3 (lock):   touch LRU for accessed keys
// ===========================================================================
std::vector<bool> SpdkSsdTier::BatchReadIntoPtr(
    const std::vector<std::string>& keys,
    const std::vector<uintptr_t>& dst_ptrs,
    const std::vector<size_t>& sizes) {
    const int count = static_cast<int>(keys.size());
    std::vector<bool> results(count, false);
    if (!initialized_ || count == 0) return results;

    auto& env = umbp::SpdkEnv::Instance();

    // --- Phase 1: Look up entries (lock held) ---
    struct ReadItem {
        int idx;
        uint64_t offset;
        size_t aligned_size;
        size_t data_size;
    };
    std::vector<ReadItem> items;
    items.reserve(count);

    {
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < count; ++i) {
            auto it = entries_.find(keys[i]);
            if (it == entries_.end()) continue;

            size_t read_size = std::min(sizes[i], it->second.data_size);
            ReadItem ri;
            ri.idx = i;
            ri.offset = it->second.handle.address();
            ri.aligned_size = AlignUp(read_size);
            ri.data_size = read_size;
            items.push_back(ri);
        }
    }
    // mu_ released

    if (items.empty()) return results;

    // --- Phase 2: Deep-queue I/O pipeline (no lock) ---
    const int item_count = static_cast<int>(items.size());
    const int qd = std::min(item_count, kMaxQueueDepth);

    size_t max_aligned = 0;
    for (auto& ri : items)
        max_aligned = std::max(max_aligned, ri.aligned_size);

    auto dma_bufs = std::make_unique<void*[]>(qd);
    int got = env.DmaPoolAllocBatch(dma_bufs.get(), max_aligned, qd);
    if (got < qd) {
        UMBP_LOG_ERROR("SpdkSsdTier::BatchRead: DMA alloc short %d/%d", got, qd);
        for (int i = got; i < qd; ++i)
            dma_bufs[i] = nullptr;
    }

    auto reqs = std::make_unique<umbp::SpdkIoRequest[]>(qd);
    std::vector<bool> io_ok(item_count, false);

    int head = 0, tail = 0;
    while (tail < item_count) {
        // Fill pipeline
        while (head < item_count && (head - tail) < qd) {
            int slot = head % qd;
            if (!dma_bufs[slot]) { ++head; continue; }

            auto& ri = items[head];
            auto& req = reqs[slot];
            req.op = umbp::SpdkIoRequest::READ;
            req.buf = dma_bufs[slot];
            req.offset = ri.offset;
            req.nbytes = ri.aligned_size;
            req.src_data = nullptr;
            req.src_iov = nullptr;
            req.src_iovcnt = 0;
            req.dst_iov = nullptr;
            req.dst_iovcnt = 0;
            req.completed.store(false, std::memory_order_release);
            req.success = false;

            env.SubmitIoAsync(&req);
            ++head;
        }

        // Drain oldest: copy from DMA to user buffer on calling thread
        if (tail < head) {
            int slot = tail % qd;
            while (!reqs[slot].completed.load(std::memory_order_acquire))
                CPU_PAUSE();

            if (reqs[slot].success) {
                auto& ri = items[tail];
                std::memcpy(reinterpret_cast<void*>(dst_ptrs[ri.idx]),
                            dma_bufs[slot], ri.data_size);
                io_ok[tail] = true;
            }
            ++tail;
        }
    }

    env.DmaPoolFreeBatch(dma_bufs.get(), max_aligned, qd);

    // --- Phase 3: Update LRU (lock held) ---
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (int j = 0; j < item_count; ++j) {
            int idx = items[j].idx;
            if (io_ok[j]) {
                TouchLRU(keys[idx]);
                results[idx] = true;
            }
        }
    }

    return results;
}
