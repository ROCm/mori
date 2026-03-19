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

    std::string GetLRUKey() const override;
    std::vector<std::string> GetLRUCandidates(size_t max_candidates) const override;

   private:
    struct Entry {
        umbp::offset_allocator::OffsetAllocationHandle handle;
        size_t data_size = 0;

        Entry() = default;
        Entry(Entry&&) = default;
        Entry& operator=(Entry&&) = default;
        Entry(const Entry&) = delete;
        Entry& operator=(const Entry&) = delete;
    };

    size_t AlignUp(size_t size) const {
        return (size + block_size_ - 1) & ~(static_cast<size_t>(block_size_) - 1);
    }

    void TouchLRU(const std::string& key);
    void RemoveLRU(const std::string& key);
    void AllocDmaRing(size_t buf_size);
    void FreeDmaRing();

    bool initialized_ = false;
    std::shared_ptr<umbp::offset_allocator::OffsetAllocator> allocator_;
    uint32_t block_size_ = 4096;
    size_t capacity_ = 0;

    mutable std::mutex mu_;
    std::unordered_map<std::string, Entry> entries_;
    std::list<std::string> lru_list_;
    std::unordered_map<std::string, std::list<std::string>::iterator> lru_iter_;

    static constexpr int kMaxQueueDepth = 128;

    // Pre-allocated DMA ring buffers to avoid per-batch pool alloc overhead
    std::mutex dma_ring_mu_;
    void** dma_ring_ = nullptr;
    size_t dma_ring_buf_size_ = 0;
    int dma_ring_count_ = 0;
};
