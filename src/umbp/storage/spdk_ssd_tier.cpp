// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// SpdkSsdTier: deep-queue NVMe pipeline via SPDK.

#include "umbp/storage/spdk_ssd_tier.h"

#include <algorithm>
#include <cstring>
#include <thread>

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
    : TierBackend(StorageTier::LOCAL_SSD) {
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

    num_io_workers_ = std::max(1, config.spdk_io_workers);

    // Pre-allocate DMA ring buffers (2MB each, kMaxQueueDepth slots)
    size_t ring_buf_size = 2ULL * 1024 * 1024;
    AllocDmaRing(ring_buf_size);

    initialized_ = true;
    UMBP_LOG_INFO("SpdkSsdTier: ready — capacity=%zuMB block_size=%u dma_ring=%d×%zuKB io_workers=%d",
                  capacity_ / (1024 * 1024), block_size_,
                  dma_ring_count_, dma_ring_buf_size_ / 1024, num_io_workers_);
}

SpdkSsdTier::~SpdkSsdTier() {
    Clear();
    FreeDmaRing();
}

void SpdkSsdTier::AllocDmaRing(size_t buf_size) {
    auto& env = umbp::SpdkEnv::Instance();
    if (!env.IsInitialized()) return;

    dma_ring_buf_size_ = buf_size;
    dma_ring_count_ = kMaxQueueDepth * std::max(1, num_io_workers_);
    dma_ring_ = new void*[dma_ring_count_];

    int got = env.DmaPoolAllocBatch(dma_ring_, dma_ring_buf_size_,
                                     dma_ring_count_);
    if (got < dma_ring_count_) {
        UMBP_LOG_WARN("SpdkSsdTier: DMA ring partial %d/%d", got, dma_ring_count_);
        for (int i = got; i < dma_ring_count_; ++i)
            dma_ring_[i] = nullptr;
        dma_ring_count_ = got;
    }
}

void SpdkSsdTier::FreeDmaRing() {
    if (!dma_ring_) return;
    auto& env = umbp::SpdkEnv::Instance();
    if (env.IsInitialized() && dma_ring_count_ > 0) {
        env.DmaPoolFreeBatch(dma_ring_, dma_ring_buf_size_, dma_ring_count_);
    }
    delete[] dma_ring_;
    dma_ring_ = nullptr;
    dma_ring_count_ = 0;
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
    lru_list_.erase(it->second.lru_pos);
    entries_.erase(it);
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
            auto eit = entries_.find(keys[i]);
            if (eit != entries_.end()) {
                results[i] = true;
                lru_list_.splice(lru_list_.begin(),
                                 lru_list_, eit->second.lru_pos);
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

    // --- Phase 2: Chunked deep-queue I/O pipeline (no lock) ---
    // Split large values into DMA-ring-sized chunks so we always use the
    // pre-allocated ring and never need large contiguous DMA allocations.
    const int pending_count = static_cast<int>(pending.size());
    const size_t chunk_sz = dma_ring_buf_size_;

    struct WriteChunk {
        int item_idx;
        uint64_t offset;
        size_t nbytes;
        size_t data_offset;
        size_t data_bytes;
    };
    std::vector<WriteChunk> chunks;
    chunks.reserve(pending_count);
    for (int i = 0; i < pending_count; ++i) {
        auto& p = pending[i];
        size_t rem_aligned = p.aligned_size;
        size_t rem_data = sizes[p.idx];
        size_t src_off = 0;
        uint64_t dev_off = p.handle.address();
        while (rem_aligned > 0) {
            size_t ca = std::min(rem_aligned, chunk_sz);
            size_t cd = std::min(rem_data, ca);
            chunks.push_back({i, dev_off, ca, src_off, cd});
            rem_aligned -= ca;
            rem_data = (rem_data > cd) ? rem_data - cd : 0;
            src_off += cd;
            dev_off += ca;
        }
    }

    const int chunk_count = static_cast<int>(chunks.size());
    auto chunk_ok = std::make_unique<uint8_t[]>(chunk_count);
    std::memset(chunk_ok.get(), 0, chunk_count);

    {
        std::lock_guard<std::mutex> dma_lk(dma_ring_mu_);

        constexpr int kMinChunksPerWorker = 16;
        int max_w = std::min(num_io_workers_, dma_ring_count_ / 2);
        int num_workers = std::clamp(chunk_count / kMinChunksPerWorker, 1, max_w);
        int bufs_per = dma_ring_count_ / num_workers;

        auto run_pipeline = [&](int c_begin, int c_end,
                                void** bufs, int local_qd) {
            auto lreqs = std::make_unique<umbp::SpdkIoRequest[]>(local_qd);
            auto lbatch = std::make_unique<umbp::SpdkIoRequest*[]>(local_qd);
            int head = c_begin, tail = c_begin;

            while (tail < c_end) {
                int bc = 0;
                while (head < c_end && (head - tail) < local_qd) {
                    int slot = (head - c_begin) % local_qd;
                    auto& c = chunks[head];
                    int idx = pending[c.item_idx].idx;

                    const char* src =
                        static_cast<const char*>(data_ptrs[idx]) + c.data_offset;
                    std::memcpy(bufs[slot], src, c.data_bytes);
                    if (c.nbytes > c.data_bytes)
                        std::memset(static_cast<char*>(bufs[slot]) + c.data_bytes,
                                    0, c.nbytes - c.data_bytes);

                    auto& req = lreqs[slot];
                    req.op = umbp::SpdkIoRequest::WRITE;
                    req.buf = bufs[slot];
                    req.offset = c.offset;
                    req.nbytes = c.nbytes;
                    req.src_data = nullptr;
                    req.src_iov = nullptr;
                    req.src_iovcnt = 0;
                    req.dst_iov = nullptr;
                    req.dst_iovcnt = 0;
                    req.completed.store(false, std::memory_order_release);
                    req.success = false;

                    lbatch[bc++] = &req;
                    ++head;

                    if (bc >= 8) {
                        env.SubmitIoBatchAsync(lbatch.get(), bc);
                        bc = 0;
                    }
                }
                if (bc > 0)
                    env.SubmitIoBatchAsync(lbatch.get(), bc);

                while (tail < head) {
                    int slot = (tail - c_begin) % local_qd;
                    if (!lreqs[slot].completed.load(std::memory_order_acquire))
                        break;
                    chunk_ok[tail] = lreqs[slot].success ? 1 : 0;
                    ++tail;
                }
            }
        };

        if (num_workers <= 1) {
            int qd = std::min({chunk_count, kMaxQueueDepth, dma_ring_count_});
            run_pipeline(0, chunk_count, dma_ring_, qd);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(num_workers - 1);
            for (int w = 0; w < num_workers; ++w) {
                int cb = chunk_count * w / num_workers;
                int ce = chunk_count * (w + 1) / num_workers;
                void** wb = dma_ring_ + w * bufs_per;
                int wq = (w == num_workers - 1)
                    ? (dma_ring_count_ - w * bufs_per) : bufs_per;
                if (w < num_workers - 1) {
                    workers.emplace_back([&, cb, ce, wb, wq]() {
                        run_pipeline(cb, ce, wb, wq);
                    });
                } else {
                    run_pipeline(cb, ce, wb, wq);
                }
            }
            for (auto& t : workers) t.join();
        }
    }

    // --- Phase 3: Update metadata (lock held) ---
    // All chunks of an item must succeed for it to be valid.
    std::vector<bool> item_ok(pending_count, true);
    for (int j = 0; j < chunk_count; ++j)
        if (!chunk_ok[j]) item_ok[chunks[j].item_idx] = false;

    {
        std::lock_guard<std::mutex> lk(mu_);
        for (int j = 0; j < pending_count; ++j) {
            int idx = pending[j].idx;
            if (item_ok[j]) {
                Entry entry;
                entry.handle = std::move(pending[j].handle);
                entry.data_size = sizes[idx];

                auto [it, inserted] = entries_.try_emplace(
                    keys[idx], std::move(entry));
                if (inserted) {
                    lru_list_.push_front(keys[idx]);
                    it->second.lru_pos = lru_list_.begin();
                } else {
                    it->second.handle = std::move(entry.handle);
                    it->second.data_size = entry.data_size;
                    lru_list_.splice(lru_list_.begin(),
                                     lru_list_, it->second.lru_pos);
                }
                results[idx] = true;
            }
        }
    }

    return results;
}

// ===========================================================================
// BatchReadIntoPtr — deep-queue NVMe read pipeline
//
// Phase 1 (lock):   look up entries, collect offsets and sizes
// Phase 2 (unlock): submit reads + drain + memcpy DMA→user (pipelined)
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

            lru_list_.splice(lru_list_.begin(),
                             lru_list_, it->second.lru_pos);

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

    // --- Phase 2: Chunked deep-queue I/O pipeline (no lock) ---
    const int item_count = static_cast<int>(items.size());
    const size_t chunk_sz = dma_ring_buf_size_;

    struct ReadChunk {
        int item_idx;
        uint64_t offset;
        size_t nbytes;
        size_t data_offset;
        size_t data_bytes;
    };
    std::vector<ReadChunk> chunks;
    chunks.reserve(item_count);
    for (int i = 0; i < item_count; ++i) {
        auto& ri = items[i];
        size_t rem_aligned = ri.aligned_size;
        size_t rem_data = ri.data_size;
        size_t dst_off = 0;
        uint64_t dev_off = ri.offset;
        while (rem_aligned > 0) {
            size_t ca = std::min(rem_aligned, chunk_sz);
            size_t cd = std::min(rem_data, ca);
            chunks.push_back({i, dev_off, ca, dst_off, cd});
            rem_aligned -= ca;
            rem_data = (rem_data > cd) ? rem_data - cd : 0;
            dst_off += cd;
            dev_off += ca;
        }
    }

    const int chunk_count = static_cast<int>(chunks.size());
    auto chunk_ok = std::make_unique<uint8_t[]>(chunk_count);
    std::memset(chunk_ok.get(), 0, chunk_count);

    {
        std::lock_guard<std::mutex> dma_lk(dma_ring_mu_);

        constexpr int kMinChunksPerWorker = 16;
        int max_w = std::min(num_io_workers_, dma_ring_count_ / 2);
        int num_workers = std::clamp(chunk_count / kMinChunksPerWorker, 1, max_w);
        int bufs_per = dma_ring_count_ / num_workers;

        auto run_pipeline = [&](int c_begin, int c_end,
                                void** bufs, int local_qd) {
            auto lreqs = std::make_unique<umbp::SpdkIoRequest[]>(local_qd);
            auto lbatch = std::make_unique<umbp::SpdkIoRequest*[]>(local_qd);
            int head = c_begin, tail = c_begin;

            while (tail < c_end) {
                int bc = 0;
                while (head < c_end && (head - tail) < local_qd) {
                    int slot = (head - c_begin) % local_qd;
                    auto& c = chunks[head];

                    auto& req = lreqs[slot];
                    req.op = umbp::SpdkIoRequest::READ;
                    req.buf = bufs[slot];
                    req.offset = c.offset;
                    req.nbytes = c.nbytes;
                    req.src_data = nullptr;
                    req.src_iov = nullptr;
                    req.src_iovcnt = 0;
                    req.dst_iov = nullptr;
                    req.dst_iovcnt = 0;
                    req.completed.store(false, std::memory_order_release);
                    req.success = false;

                    lbatch[bc++] = &req;
                    ++head;

                    if (bc >= 8) {
                        env.SubmitIoBatchAsync(lbatch.get(), bc);
                        bc = 0;
                    }
                }
                if (bc > 0)
                    env.SubmitIoBatchAsync(lbatch.get(), bc);

                while (tail < head) {
                    int slot = (tail - c_begin) % local_qd;
                    if (!lreqs[slot].completed.load(std::memory_order_acquire))
                        break;

                    if (lreqs[slot].success) {
                        auto& c = chunks[tail];
                        auto& ri = items[c.item_idx];
                        char* dst = reinterpret_cast<char*>(dst_ptrs[ri.idx])
                                    + c.data_offset;
                        std::memcpy(dst, bufs[slot], c.data_bytes);
                        chunk_ok[tail] = 1;
                    }
                    ++tail;
                }
            }
        };

        if (num_workers <= 1) {
            int qd = std::min({chunk_count, kMaxQueueDepth, dma_ring_count_});
            run_pipeline(0, chunk_count, dma_ring_, qd);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(num_workers - 1);
            for (int w = 0; w < num_workers; ++w) {
                int cb = chunk_count * w / num_workers;
                int ce = chunk_count * (w + 1) / num_workers;
                void** wb = dma_ring_ + w * bufs_per;
                int wq = (w == num_workers - 1)
                    ? (dma_ring_count_ - w * bufs_per) : bufs_per;
                if (w < num_workers - 1) {
                    workers.emplace_back([&, cb, ce, wb, wq]() {
                        run_pipeline(cb, ce, wb, wq);
                    });
                } else {
                    run_pipeline(cb, ce, wb, wq);
                }
            }
            for (auto& t : workers) t.join();
        }
    }

    // Phase 3: mark results
    std::vector<bool> item_ok(item_count, true);
    for (int j = 0; j < chunk_count; ++j)
        if (!chunk_ok[j]) item_ok[chunks[j].item_idx] = false;
    for (int j = 0; j < item_count; ++j)
        if (item_ok[j]) results[items[j].idx] = true;

    return results;
}
