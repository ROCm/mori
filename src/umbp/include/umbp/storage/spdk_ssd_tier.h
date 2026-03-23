// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// SpdkSsdTier: SPDK-based SSD tier with deep-queue NVMe pipeline.
//
// Metadata management follows mooncake-store's OffsetAllocatorStorageBackend:
//   - Sharded map (kNumShards) with std::shared_mutex for concurrent reads
//   - RefCounted allocation handles (AllocationPtr) for safe concurrent access
//   - Auto LRU eviction on allocation failure
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
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
    std::vector<bool> BatchWriteStreaming(
        const std::vector<std::string>& keys,
        const std::vector<const void*>& data_ptrs,
        const std::vector<size_t>& sizes,
        std::atomic<uint64_t>* bytes_ready,
        const std::vector<size_t>& item_shm_offsets,
        void** ext_dma_bufs = nullptr, int ext_dma_count = 0);

    // DMA-ring read with streaming progress at two granularities.
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
    std::vector<bool> BatchWriteDmaDirect(
        const std::vector<std::string>& keys,
        const std::vector<void*>& dma_ptrs,
        const std::vector<size_t>& sizes);

    std::vector<bool> BatchReadDmaDirect(
        const std::vector<std::string>& keys,
        const std::vector<uintptr_t>& dma_ptrs,
        const std::vector<size_t>& sizes);

    // Streaming zero-copy variants.
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
    // -- RefCounted allocation handle (mooncake-store pattern) ---------------
    struct RefCountedAllocationHandle {
        umbp::offset_allocator::OffsetAllocationHandle handle;
        explicit RefCountedAllocationHandle(
            umbp::offset_allocator::OffsetAllocationHandle&& h)
            : handle(std::move(h)) {}
        RefCountedAllocationHandle(const RefCountedAllocationHandle&) = delete;
        RefCountedAllocationHandle& operator=(const RefCountedAllocationHandle&) = delete;
        RefCountedAllocationHandle(RefCountedAllocationHandle&&) = default;
        RefCountedAllocationHandle& operator=(RefCountedAllocationHandle&&) = default;
    };
    using AllocationPtr = std::shared_ptr<RefCountedAllocationHandle>;

    // -- Entry stored per key ------------------------------------------------
    struct Entry {
        AllocationPtr allocation;
        size_t data_size = 0;
        std::list<std::string>::iterator lru_pos;

        Entry() = default;
        Entry(Entry&&) = default;
        Entry& operator=(Entry&&) = default;
        Entry(const Entry&) = delete;
        Entry& operator=(const Entry&) = delete;
    };

    // -- Sharded metadata (mooncake-store pattern) ---------------------------
    static constexpr size_t kNumShards = 64;
    static_assert((kNumShards & (kNumShards - 1)) == 0,
                  "kNumShards must be a power of 2");
    struct MetadataShard {
        mutable std::shared_mutex mutex;
        std::unordered_map<std::string, Entry> map;
    };

    size_t ShardForKey(const std::string& key) const {
        return std::hash<std::string>{}(key) & (kNumShards - 1);
    }

    size_t AlignUp(size_t size) const {
        return (size + block_size_ - 1) & ~(static_cast<size_t>(block_size_) - 1);
    }

    void AllocDmaRing(size_t buf_size);
    void FreeDmaRing();

    // -- Write helpers (common Phase 1 / Phase 3 for all BatchWrite*) --------
    struct PendingWrite {
        int idx;
        size_t aligned_size;
        AllocationPtr allocation;
    };

    std::vector<PendingWrite> PrepareWriteAlloc(
        const std::vector<std::string>& keys,
        const std::vector<size_t>& sizes,
        std::vector<bool>& results);

    void CommitWriteEntries(
        const std::vector<std::string>& keys,
        const std::vector<size_t>& sizes,
        std::vector<PendingWrite>& pending,
        const std::vector<bool>& item_ok,
        std::vector<bool>& results);

    // -- Read helper (common Phase 1 for all BatchRead*) ---------------------
    struct ReadInfo {
        int idx;
        uint64_t offset;
        size_t aligned_size;
        size_t data_size;
        AllocationPtr guard;
    };

    std::vector<ReadInfo> PrepareReadLookup(
        const std::vector<std::string>& keys,
        const std::vector<size_t>& sizes,
        std::vector<bool>& results);

    // -- LRU eviction --------------------------------------------------------
    size_t EvictLRU(size_t needed);

    // -- Member data ---------------------------------------------------------
    bool initialized_ = false;
    std::shared_ptr<umbp::offset_allocator::OffsetAllocator> allocator_;
    uint32_t block_size_ = 4096;
    size_t capacity_ = 0;

    std::array<MetadataShard, kNumShards> shards_;
    mutable std::mutex lru_mu_;
    std::list<std::string> lru_list_;

    static constexpr int kMaxQueueDepth = 128;
    int num_io_workers_ = 1;

    std::mutex dma_ring_mu_;
    void** dma_ring_ = nullptr;
    size_t dma_ring_buf_size_ = 0;
    int dma_ring_count_ = 0;
};
