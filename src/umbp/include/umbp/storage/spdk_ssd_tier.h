// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// SpdkSsdTier: SPDK-based SSD tier with deep-queue NVMe pipeline.
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "umbp/allocator/offset_allocator.hpp"
#include "umbp/common/config.h"
#include "umbp/storage/tier_backend.h"

class SpdkSsdTier : public TierBackend {
   public:
    explicit SpdkSsdTier(const UMBPConfig& config);
    ~SpdkSsdTier() override;

    bool IsValid() const { return initialized_; }

    bool Write(const std::string& key, const void* data, size_t size) override;
    bool ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) override;
    bool Exists(const std::string& key) const override;
    bool Evict(const std::string& key) override;
    std::pair<size_t, size_t> Capacity() const override;
    void Clear() override;

    std::vector<bool> BatchWrite(
        const std::vector<std::string>& keys,
        const std::vector<const void*>& data_ptrs,
        const std::vector<size_t>& sizes) override;

    std::vector<bool> BatchReadIntoPtr(
        const std::vector<std::string>& keys,
        const std::vector<uintptr_t>& dst_ptrs,
        const std::vector<size_t>& sizes) override;

    // DMA-ring write with byte-level streaming from shared memory.
    // Identical to BatchWrite except that before memcpy-ing each DMA chunk,
    // the pipeline waits for *bytes_ready >= item_shm_offset[i] + chunk_end.
    // item_shm_offsets[i] is the absolute byte offset of item i's data within
    // the SHM data area (caller computes from BatchEntry::data_offset).
    std::vector<bool> BatchWriteStreaming(
        const std::vector<std::string>& keys,
        const std::vector<const void*>& data_ptrs,
        const std::vector<size_t>& sizes,
        std::atomic<uint64_t>* bytes_ready,
        const std::vector<size_t>& item_shm_offsets,
        void** ext_dma_bufs = nullptr, int ext_dma_count = 0);

    // DMA-ring read with streaming progress at two granularities:
    //   *items_done  — incremented after each item's data is fully in dst.
    //   *bytes_done  — updated after each 2MB DMA chunk is memcpy'd to dst,
    //                  using absolute SHM offsets so the caller can overlap
    //                  downstream copies at sub-item granularity.
    // item_shm_offsets[i] is the byte offset of item i's data in the SHM
    // data area (from BatchEntry::data_offset).  May be nullptr to disable
    // byte-level signaling.
    std::vector<bool> BatchReadIntoPtrStreaming(
        const std::vector<std::string>& keys,
        const std::vector<uintptr_t>& dst_ptrs,
        const std::vector<size_t>& sizes,
        std::atomic<uint32_t>* items_done,
        std::atomic<uint64_t>* bytes_done = nullptr,
        const std::vector<size_t>* item_shm_offsets = nullptr,
        void** ext_dma_bufs = nullptr, int ext_dma_count = 0);

    // Zero-copy DMA variants: data_ptrs must be DMA-registered, 4KB-aligned,
    // with AlignUp(size) bytes of writable space per key.
    // Skips the DMA ring + memcpy entirely — SPDK DMAs directly from/to the buffers.
    std::vector<bool> BatchWriteDmaDirect(
        const std::vector<std::string>& keys,
        const std::vector<void*>& dma_ptrs,
        const std::vector<size_t>& sizes);

    std::vector<bool> BatchReadDmaDirect(
        const std::vector<std::string>& keys,
        const std::vector<uintptr_t>& dma_ptrs,
        const std::vector<size_t>& sizes);

    // Streaming zero-copy: client memcpy and proxy NVMe DMA run in parallel.
    // Write: proxy polls items_ready (client increments per key) before submitting.
    // Read:  proxy increments items_done per key so client can copy in parallel.
    std::vector<bool> BatchWriteDmaStreaming(
        const std::vector<std::string>& keys,
        const std::vector<void*>& dma_ptrs,
        const std::vector<size_t>& sizes,
        std::atomic<uint32_t>* items_ready);

    std::vector<bool> BatchReadDmaStreaming(
        const std::vector<std::string>& keys,
        const std::vector<uintptr_t>& dma_ptrs,
        const std::vector<size_t>& sizes,
        std::atomic<uint32_t>* items_done);

    std::string GetLRUKey() const override;
    std::vector<std::string> GetLRUCandidates(size_t max_candidates) const override;

   private:
    struct Entry {
        umbp::offset_allocator::OffsetAllocationHandle handle;
        size_t data_size = 0;
        std::list<std::string>::iterator lru_pos;

        Entry() = default;
        Entry(Entry&&) = default;
        Entry& operator=(Entry&&) = default;
        Entry(const Entry&) = delete;
        Entry& operator=(const Entry&) = delete;
    };

    size_t AlignUp(size_t size) const {
        return (size + block_size_ - 1) & ~(static_cast<size_t>(block_size_) - 1);
    }

    void AllocDmaRing(size_t buf_size);
    void FreeDmaRing();

    bool initialized_ = false;
    std::shared_ptr<umbp::offset_allocator::OffsetAllocator> allocator_;
    uint32_t block_size_ = 4096;
    size_t capacity_ = 0;

    mutable std::mutex mu_;
    std::unordered_map<std::string, Entry> entries_;
    std::list<std::string> lru_list_;

    static constexpr int kMaxQueueDepth = 128;
    int num_io_workers_ = 1;

    // Pre-allocated DMA ring buffers to avoid per-batch pool alloc overhead
    std::mutex dma_ring_mu_;
    void** dma_ring_ = nullptr;
    size_t dma_ring_buf_size_ = 0;
    int dma_ring_count_ = 0;
};
