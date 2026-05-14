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
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "umbp/local/host_mem_allocator.h"
#include "umbp/local/tiers/tier_backend.h"

namespace mori::umbp {

// DRAM Tier: mmap pre-allocated large memory block with offset allocator
class DRAMTier : public TierBackend {
 public:
  DRAMTier(size_t capacity, bool use_shm, const std::string& shm_name, bool use_hugepages,
           size_t hugepage_size, int numa_node, bool prefault);

  // Backward-compatible overloads (no hugepages).
  explicit DRAMTier(size_t capacity)
      : DRAMTier(capacity, false, "/umbp_dram", false, 2ULL * 1024 * 1024, -1, true) {}
  DRAMTier(size_t capacity, bool use_shm, const std::string& shm_name)
      : DRAMTier(capacity, use_shm, shm_name, false, 2ULL * 1024 * 1024, -1, true) {}

  ~DRAMTier() override;

  // Non-copyable
  DRAMTier(const DRAMTier&) = delete;
  DRAMTier& operator=(const DRAMTier&) = delete;

  // TierBackend interface
  // Write: allocate slot in pre-allocated memory, memcpy data.
  // Does NOT self-evict on space pressure — returns false if no space.
  // Upper layer (LocalStorageManager) is responsible for demoting keys.
  bool Write(const std::string& key, const void* data, size_t size) override;
  bool ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) override;
  bool Exists(const std::string& key) const override;
  bool Evict(const std::string& key) override;
  std::pair<size_t, size_t> Capacity() const override;
  void Clear() override;

  // Extended interface overrides
  TierCapabilities Capabilities() const override;
  std::vector<char> Read(const std::string& key) override;
  std::string GetLRUKey() const override;
  std::vector<std::string> GetLRUCandidates(size_t max_candidates) const override;
  std::optional<std::string> GetLocationId(const std::string& key) const override;

  // DRAM-specific: zero-copy read returning internal pointer.
  // Only safe for in-process mmap'd memory. Caller must not hold
  // the returned pointer across Evict/Write calls.
  const void* ReadPtr(const std::string& key, size_t* out_size) override;

  // Accessors for distributed integration.
  // Returns the mmap'd base address for RDMA registration.
  void* GetBasePtr() const { return base_ptr_; }
  // Returns the byte offset of a key's slot, or nullopt if not found.
  std::optional<size_t> GetSlotOffset(const std::string& key) const;

 private:
  void* base_ptr_;      // mmap base address
  size_t capacity_;     // usable capacity (= requested size)
  size_t mapped_size_;  // actual mapped size (>= capacity_ with hugepages)
  size_t used_;
  int shm_fd_;  // shm_open fd (-1 for anonymous)
  bool use_shm_;
  std::string shm_name_;
  HostBufferHandle host_buf_handle_;  // owned handle for non-shm path

  // Simple offset allocator: key -> (offset, size)
  struct SlotInfo {
    size_t offset;
    size_t size;
  };
  std::unordered_map<std::string, SlotInfo> slots_;

  // LRU linked list
  std::list<std::string> lru_list_;
  std::unordered_map<std::string, std::list<std::string>::iterator> lru_map_;

  // Free block management (simple free list)
  struct FreeBlock {
    size_t offset;
    size_t size;
  };
  std::list<FreeBlock> free_list_;

  mutable std::mutex mu_;

  size_t Allocate(size_t size);                 // Allocate from free_list_
  void Deallocate(size_t offset, size_t size);  // Return to free_list_
  void EvictLRU();                              // Evict least recently used
  void TouchLRU(const std::string& key);        // Update LRU position
};

}  // namespace mori::umbp
