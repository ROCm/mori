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
#include <vector>
#include <map>
#include <mutex>

namespace mori {
namespace application {

/**
 * @brief Virtual Address Block represents a contiguous VA range
 */
struct VABlock {
  uintptr_t startAddr;    // Starting virtual address
  size_t size;            // Size in bytes
  bool isFree;            // Whether this block is free
  
  VABlock() : startAddr(0), size(0), isFree(true) {}
  
  VABlock(uintptr_t addr, size_t sz, bool free = true)
      : startAddr(addr), size(sz), isFree(free) {}
};

/**
 * @brief VMM Virtual Address Manager
 * 
 * This manager tracks virtual address allocations within the VMM heap.
 * It implements a first-fit allocation strategy and automatically coalesces
 * adjacent free blocks to reduce fragmentation.
 * 
 * Key features:
 * - First-fit allocation: Finds the first free block that can satisfy the request
 * - Automatic coalescing: Merges adjacent free blocks on deallocation
 * - Thread-safe: All operations are protected by mutex
 * - Efficient lookup: Uses sorted list for fast allocation search
 */
class VMMVAManager {
 public:
  /**
   * @brief Construct a new VA Manager
   * 
   * @param baseAddr Base virtual address of the VMM heap
   * @param totalSize Total size of the VMM heap virtual address space
   * @param granularity Physical memory allocation granularity (for RDMA boundary alignment)
   */
  VMMVAManager(uintptr_t baseAddr, size_t totalSize, size_t granularity = 0);
  
  ~VMMVAManager() = default;

  /**
   * @brief Allocate a virtual address block
   * 
   * Searches for the first free block that can satisfy the allocation request.
   * If found, marks it as allocated and returns the address.
   * 
   * @param size Size in bytes to allocate (will be aligned)
   * @param alignment Alignment requirement (default: 256 bytes)
   * @return uintptr_t Allocated virtual address, or 0 if allocation failed
   */
  uintptr_t Allocate(size_t size, size_t alignment = 256);

  /**
   * @brief Free a previously allocated virtual address block
   * 
   * Marks the block as free and automatically coalesces with adjacent
   * free blocks to reduce fragmentation.
   * 
   * @param addr Starting address of the block to free
   * @return true if freed successfully, false if address not found
   */
  bool Free(uintptr_t addr);

  /**
   * @brief Get the size of an allocated block
   * 
   * @param addr Starting address of the block
   * @return size_t Size of the block in bytes, or 0 if not found
   */
  size_t GetBlockSize(uintptr_t addr) const;

  /**
   * @brief Get allocation statistics
   * 
   * @param totalBlocks Output: Total number of VA blocks
   * @param freeBlocks Output: Number of free VA blocks
   * @param allocatedBlocks Output: Number of allocated VA blocks
   * @param totalFreeSpace Output: Total free space in bytes
   * @param largestFreeBlock Output: Size of largest free block in bytes
   */
  void GetStats(size_t& totalBlocks, size_t& freeBlocks, size_t& allocatedBlocks,
                size_t& totalFreeSpace, size_t& largestFreeBlock) const;

  /**
   * @brief Check if an address is within the managed VA range
   * 
   * @param addr Address to check
   * @return true if address is within range, false otherwise
   */
  bool IsValidAddress(uintptr_t addr) const;

  /**
   * @brief Reset the VA manager to initial state
   * 
   * Marks all space as free. Used during VMM heap finalization.
   */
  void Reset();

 private:
  /**
   * @brief Immediately coalesce a block with its adjacent free blocks
   * 
   * This is an optimized version that only checks and merges the immediate
   * neighbors of the specified block using map iterators for O(log n) lookup.
   * Called after marking a block as free.
   * 
   * @param addr Address of the block to coalesce
   */
  void CoalesceAdjacentBlocks(uintptr_t addr);

  /**
   * @brief Coalesce all adjacent free blocks (full scan)
   * 
   * Called after freeing a block to merge adjacent free blocks.
   * This is the legacy method that scans the entire blocks_ vector.
   * Consider using CoalesceAdjacentBlocks for better performance.
   */
  void CoalesceFreeBlocks();

  /**
   * @brief Align size to the specified alignment
   * 
   * @param size Size to align
   * @param alignment Alignment requirement (must be power of 2)
   * @return size_t Aligned size
   */
  static size_t AlignSize(size_t size, size_t alignment);

  uintptr_t baseAddr_;       // Base address of VA space
  size_t totalSize_;         // Total size of VA space
  size_t granularity_;       // Physical memory granularity (for RDMA boundary alignment)
  
  // Map-based implementation (inspired by NVSHMEM for O(log n) operations)
  // Key: start address, Value: VABlock
  std::map<uintptr_t, VABlock> blocks_;
  
  // For fast neighbor lookup during coalescing
  // Key: end address (startAddr + size), Value: start address
  std::map<uintptr_t, uintptr_t> endAddrToStartAddr_;
  
  mutable std::mutex mutex_;  // Protects all data structures
};

}  // namespace application
}  // namespace mori
