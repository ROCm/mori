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

#include "umbp/common/config.h"
#include "umbp/local/storage/tier_backend.h"

namespace mori::umbp {

// DRAM Tier: mmap pre-allocated large memory block with chunk-aware offset allocator.
//
// The arena is divided into fixed-size logical chunks, each of which maps 1:1
// to one RDMA memory region, one master-side allocator, and one published
// location_id.  Every block is placed entirely within one chunk.
class DRAMTier : public TierBackend {
 public:
  struct ChunkLocation {
    uint32_t chunk_index = 0;
    size_t offset = 0;
  };

  DRAMTier(size_t capacity, bool use_shm = false, const std::string& shm_name = "/umbp_dram");
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

  // Returns the mmap'd base address for RDMA registration.
  void* GetBasePtr() const { return base_ptr_; }

  // Returns the byte offset of a key's slot (global, across all chunks).
  std::optional<size_t> GetSlotOffset(const std::string& key) const;

  // Re-slice the DRAM arena into fixed-size chunks for RDMA MR registration.
  // The constructor already initializes a single full-arena chunk, so this
  // method only needs to be called when the deployment requires smaller
  // chunks (e.g. AINIC/Pensando 2 GB MR limit).  When chunk_size >= capacity
  // (including SIZE_MAX or 0) the result is a single chunk.
  //
  // Chunk layout is a startup-time decision: ConfigureChunks() may only be
  // called before the layout is sealed.  The layout is sealed either by
  // SealChunkLayout() (called after distributed initialization registers
  // MRs with the master) or by the first successful Write().  Once sealed
  // the layout is permanently locked — Clear() does not re-enable
  // reconfiguration.
  void ConfigureChunks(size_t chunk_size);

  // Permanently lock the current chunk layout.  Called by UMBPClient after
  // distributed initialization so that the local layout cannot diverge
  // from the already-registered MRs and master's buffer_index mapping.
  // Also called implicitly by the first successful Write().
  void SealChunkLayout();

  // Return one ExportableDram per configured chunk.
  std::vector<ExportableDram> GetExportableChunks() const;

  // Return the chunk-local location of a key's slot.
  std::optional<ChunkLocation> GetSlotChunkLocation(const std::string& key) const;

 private:
  void* base_ptr_;  // mmap base address
  size_t capacity_;
  size_t used_;
  int shm_fd_;  // shm_open fd (-1 for anonymous)
  bool use_shm_;
  std::string shm_name_;

  // Chunk-aware slot metadata: key -> (chunk_index, offset_within_chunk, size)
  struct SlotInfo {
    uint32_t chunk_index = 0;
    size_t offset = 0;
    size_t size = 0;
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

  // Chunk layout
  struct DramChunk {
    void* base = nullptr;
    size_t size = 0;
    std::list<FreeBlock> free_list;
  };
  std::vector<DramChunk> chunks_;
  bool chunks_configured_ = false;
  bool chunks_sealed_ = false;  // set by SealChunkLayout() or first Write; never cleared

  mutable std::mutex mu_;

  // Allocate from chunk free lists.  Returns {chunk_index, offset} or nullopt.
  std::optional<std::pair<uint32_t, size_t>> Allocate(size_t size);

  // Return space to the chunk-local free list with coalescing.
  void Deallocate(uint32_t chunk_index, size_t offset, size_t size);

  void EvictLRU();
  void TouchLRU(const std::string& key);
};

}  // namespace mori::umbp
