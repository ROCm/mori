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
// SpdkSsdTier: SPDK-based SSD tier with deep-queue NVMe pipeline.
//
// Metadata management follows mooncake-store's OffsetAllocatorStorageBackend:
//   - Sharded map (kNumShards) with std::shared_mutex for concurrent reads
//   - RefCounted allocation handles (AllocationPtr) for safe concurrent access
//   - Auto LRU eviction on allocation failure
#pragma once

#include <array>
#include <atomic>
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

#include "umbp/common/config.h"
#include "umbp/local/tiers/tier_backend.h"
#include "umbp/storage/spdk/offset_allocator.hpp"

namespace mori {
namespace umbp {

class SpdkSsdTier : public TierBackend {
 public:
  struct Stats {
    uint64_t hit_count = 0;
    uint64_t miss_count = 0;
    uint64_t evicted_bytes = 0;
  };

  struct DmaPool {
    ~DmaPool();

    std::mutex mutex;
    void** bufs = nullptr;
    size_t buf_size = 0;
    int count = 0;
  };
  using SharedDmaPool = std::shared_ptr<DmaPool>;

  explicit SpdkSsdTier(const UMBPConfig& config);
  SpdkSsdTier(const UMBPConfig& config, uint64_t base_offset, size_t capacity_bytes,
              SharedDmaPool shared_dma_pool = nullptr);
  ~SpdkSsdTier() override;

  bool IsValid() const { return initialized_; }
  uint64_t base_offset() const { return base_offset_; }

  static SharedDmaPool CreateSharedDmaPool(size_t buf_size, int count);
  Stats GetStats() const;
  void RecordHit(uint64_t count = 1);
  void RecordMiss(uint64_t count = 1);

  bool Write(const std::string& key, const void* data, size_t size) override;
  bool ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) override;
  bool Exists(const std::string& key) const override;
  bool Evict(const std::string& key) override;
  std::pair<size_t, size_t> Capacity() const override;
  void Clear() override;

  std::vector<bool> BatchWrite(const std::vector<std::string>& keys,
                               const std::vector<const void*>& data_ptrs,
                               const std::vector<size_t>& sizes) override;

  std::vector<bool> BatchReadIntoPtr(const std::vector<std::string>& keys,
                                     const std::vector<uintptr_t>& dst_ptrs,
                                     const std::vector<size_t>& sizes) override;

  // DMA-ring write with byte-level streaming from shared memory.
  std::vector<bool> BatchWriteStreaming(const std::vector<std::string>& keys,
                                        const std::vector<const void*>& data_ptrs,
                                        const std::vector<size_t>& sizes,
                                        std::atomic<uint64_t>* bytes_ready,
                                        const std::vector<size_t>& item_shm_offsets,
                                        void** ext_dma_bufs = nullptr, int ext_dma_count = 0);

  // DMA-ring read with streaming progress at two granularities.
  std::vector<bool> BatchReadIntoPtrStreaming(const std::vector<std::string>& keys,
                                              const std::vector<uintptr_t>& dst_ptrs,
                                              const std::vector<size_t>& sizes,
                                              std::atomic<uint32_t>* items_done,
                                              std::atomic<uint64_t>* bytes_done = nullptr,
                                              const std::vector<size_t>* item_shm_offsets = nullptr,
                                              void** ext_dma_bufs = nullptr, int ext_dma_count = 0);

  // Zero-copy DMA variants: data_ptrs must be DMA-registered, 4KB-aligned,
  // with AlignUp(size) bytes of writable space per key.
  std::vector<bool> BatchWriteDmaDirect(const std::vector<std::string>& keys,
                                        const std::vector<void*>& dma_ptrs,
                                        const std::vector<size_t>& sizes);

  std::vector<bool> BatchReadDmaDirect(const std::vector<std::string>& keys,
                                       const std::vector<uintptr_t>& dma_ptrs,
                                       const std::vector<size_t>& sizes);

  // Streaming zero-copy variants.
  std::vector<bool> BatchWriteDmaStreaming(const std::vector<std::string>& keys,
                                           const std::vector<void*>& dma_ptrs,
                                           const std::vector<size_t>& sizes,
                                           std::atomic<uint32_t>* items_ready);

  std::vector<bool> BatchReadDmaStreaming(const std::vector<std::string>& keys,
                                          const std::vector<uintptr_t>& dma_ptrs,
                                          const std::vector<size_t>& sizes,
                                          std::atomic<uint32_t>* items_done);

  std::string GetLRUKey() const override;
  std::vector<std::string> GetLRUCandidates(size_t max_candidates) const override;

 private:
  // -- RefCounted allocation handle (mooncake-store pattern) ---------------
  struct RefCountedAllocationHandle {
    ::umbp::offset_allocator::OffsetAllocationHandle handle;
    explicit RefCountedAllocationHandle(::umbp::offset_allocator::OffsetAllocationHandle&& h)
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
  static_assert((kNumShards & (kNumShards - 1)) == 0, "kNumShards must be a power of 2");
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

  SharedDmaPool EnsureDmaPool();

  // -- Write helpers (common Phase 1 / Phase 3 for all BatchWrite*) --------
  struct PendingWrite {
    int idx;
    size_t aligned_size;
    AllocationPtr allocation;
  };

  std::vector<PendingWrite> PrepareWriteAlloc(const std::vector<std::string>& keys,
                                              const std::vector<size_t>& sizes,
                                              std::vector<bool>& results);

  void CommitWriteEntries(const std::vector<std::string>& keys, const std::vector<size_t>& sizes,
                          std::vector<PendingWrite>& pending, const std::vector<bool>& item_ok,
                          std::vector<bool>& results);

  // -- Read helper (common Phase 1 for all BatchRead*) ---------------------
  struct ReadInfo {
    int idx;
    uint64_t offset;
    size_t aligned_size;
    size_t data_size;
    AllocationPtr guard;
  };

  std::vector<ReadInfo> PrepareReadLookup(const std::vector<std::string>& keys,
                                          const std::vector<size_t>& sizes,
                                          std::vector<bool>& results);

  // -- LRU eviction --------------------------------------------------------
  size_t EvictLRU(size_t needed);

  // -- Member data ---------------------------------------------------------
  bool initialized_ = false;
  std::shared_ptr<::umbp::offset_allocator::OffsetAllocator> allocator_;
  uint32_t block_size_ = 4096;
  uint64_t base_offset_ = 0;
  size_t capacity_ = 0;

  std::array<MetadataShard, kNumShards> shards_;
  mutable std::mutex lru_mu_;
  std::list<std::string> lru_list_;

  static constexpr int kMaxQueueDepth = 128;
  int num_io_workers_ = 1;

  SharedDmaPool dma_pool_;
  std::atomic<uint64_t> hit_count_{0};
  std::atomic<uint64_t> miss_count_{0};
  std::atomic<uint64_t> evicted_bytes_{0};
};

}  // namespace umbp
}  // namespace mori
