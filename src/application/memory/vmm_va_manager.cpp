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

#include "mori/application/memory/vmm_va_manager.hpp"
#include "mori/utils/mori_log.hpp"

#include <algorithm>

namespace mori {
namespace application {

VMMVAManager::VMMVAManager(uintptr_t baseAddr, size_t totalSize, size_t granularity)
    : baseAddr_(baseAddr), totalSize_(totalSize), granularity_(granularity) {
  // Initialize with one large free block
  VABlock initialBlock(baseAddr, totalSize, true);
  blocks_[baseAddr] = initialBlock;
  endAddrToStartAddr_[baseAddr + totalSize] = baseAddr;
  
  MORI_APP_INFO("VMMVAManager initialized: baseAddr={:p}, totalSize={} bytes, granularity={} bytes", 
                reinterpret_cast<void*>(baseAddr), totalSize, granularity);
}

size_t VMMVAManager::AlignSize(size_t size, size_t alignment) {
  return (size + alignment - 1) & ~(alignment - 1);
}

uintptr_t VMMVAManager::Allocate(size_t size, size_t alignment) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (size == 0) {
    return 0;
  }
  
  // Align the requested size
  size_t alignedSize = AlignSize(size, alignment);
  
  MORI_APP_TRACE("VMMVAManager::Allocate requesting {} bytes (aligned: {}), granularity: {}", 
                 size, alignedSize, granularity_);
  
  // First-fit search: iterate through blocks_ map (already sorted by address)
  for (auto it = blocks_.begin(); it != blocks_.end(); ++it) {
    uintptr_t blockAddr = it->first;
    VABlock& block = it->second;
    
    if (!block.isFree || block.size < alignedSize) {
      continue;  // Skip allocated or too-small blocks
    }
    
    // Check address alignment (use user-specified alignment, not granularity)
    uintptr_t alignedAddr = AlignSize(block.startAddr, alignment);
    size_t alignmentWaste = alignedAddr - block.startAddr;
    
    if (alignmentWaste + alignedSize > block.size) {
      continue;  // Not enough space after alignment
    }
    
    // Check if allocation crosses granularity boundary
    // Strategy: If crossing detected, jump to next boundary (only once)
    // This ensures start address is granularity-aligned, minimizing physical blocks needed
    uintptr_t allocAddr = alignedAddr;
    if (granularity_ > 0) {
      uintptr_t allocEnd = alignedAddr + alignedSize;
      
      // Calculate the next granularity boundary
      uintptr_t startBoundary = (alignedAddr / granularity_) * granularity_;
      uintptr_t nextBoundary = startBoundary + granularity_;
      
      // If allocation would cross the boundary, skip to next boundary (only once)
      if (allocEnd > nextBoundary) {
        MORI_APP_TRACE("VMMVAManager::Allocate: allocation would cross granularity boundary "
                       "at {:p}, skipping to next boundary at {:p}",
                       reinterpret_cast<void*>(nextBoundary),
                       reinterpret_cast<void*>(nextBoundary));
        
        allocAddr = nextBoundary;
        
        // Check if the next boundary is still within this free block
        if (allocAddr + alignedSize > block.startAddr + block.size) {
          continue;  // Not enough space after skipping to next boundary
        }
        
        // Update alignment waste to include the skipped region
        alignmentWaste = allocAddr - block.startAddr;
      }
    }
    
    // Handle alignment waste at the beginning (including skipped region for granularity)
    if (alignmentWaste > 0) {
      // Create a small free block for the alignment waste
      VABlock wasteBlock(block.startAddr, alignmentWaste, true);
      
      // Remove old end address mapping
      endAddrToStartAddr_.erase(block.startAddr + block.size);
      
      // Insert waste block
      blocks_[wasteBlock.startAddr] = wasteBlock;
      endAddrToStartAddr_[wasteBlock.startAddr + wasteBlock.size] = wasteBlock.startAddr;
      
      // Update the current block
      block.startAddr = allocAddr;
      block.size -= alignmentWaste;
      
      // Update blocks_ map with new key
      blocks_.erase(it);
      blocks_[block.startAddr] = block;
      
      // Update iterator to point to the adjusted block
      it = blocks_.find(block.startAddr);
      if (it == blocks_.end()) {
        MORI_APP_ERROR("VMMVAManager::Allocate internal error after alignment adjustment");
        return 0;
      }
    }
    
    // Now allocate from the (possibly adjusted) block
    VABlock& currentBlock = it->second;
    
    // Case 1: Exact fit
    if (currentBlock.size == alignedSize) {
      currentBlock.isFree = false;
      // No need to update endAddrToStartAddr_ as size doesn't change
      MORI_APP_TRACE("VMMVAManager::Allocate found exact fit at {:p}, size={}", 
                     reinterpret_cast<void*>(allocAddr), alignedSize);
      return allocAddr;
    }
    
    // Case 2: Split the block
    size_t remainingSize = currentBlock.size - alignedSize;
    
    // Remove old end address mapping
    endAddrToStartAddr_.erase(currentBlock.startAddr + currentBlock.size);
    
    // Update current block (allocated part)
    currentBlock.size = alignedSize;
    currentBlock.isFree = false;
    endAddrToStartAddr_[currentBlock.startAddr + currentBlock.size] = currentBlock.startAddr;
    
    // Create new free block for remaining space
    uintptr_t remainingAddr = currentBlock.startAddr + alignedSize;
    VABlock remainingBlock(remainingAddr, remainingSize, true);
    blocks_[remainingAddr] = remainingBlock;
    endAddrToStartAddr_[remainingAddr + remainingSize] = remainingAddr;
    
    MORI_APP_TRACE("VMMVAManager::Allocate split block at {:p}, allocated={}, remaining={}", 
                   reinterpret_cast<void*>(allocAddr), alignedSize, remainingSize);
    return allocAddr;
  }
  
  // No suitable free block found
  MORI_APP_WARN("VMMVAManager::Allocate failed: no free block of size {} bytes available", alignedSize);
  return 0;
}

bool VMMVAManager::Free(uintptr_t addr) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  // Find the block in the map
  auto it = blocks_.find(addr);
  if (it == blocks_.end()) {
    MORI_APP_ERROR("VMMVAManager::Free failed: address {:p} not found", 
                   reinterpret_cast<void*>(addr));
    return false;
  }
  
  VABlock& block = it->second;
  
  if (block.isFree) {
    MORI_APP_WARN("VMMVAManager::Free: block at {:p} already free (double free?)", 
                  reinterpret_cast<void*>(addr));
    return false;
  }
  
  MORI_APP_TRACE("VMMVAManager::Free freeing block at {:p}, size={}", 
                 reinterpret_cast<void*>(addr), block.size);
  
  // Mark as free
  block.isFree = true;
  
  // Immediately coalesce with adjacent free blocks (optimized strategy)
  CoalesceAdjacentBlocks(addr);
  
  return true;
}

void VMMVAManager::CoalesceAdjacentBlocks(uintptr_t addr) {
  // Must be called with mutex already locked
  // This function immediately coalesces the block at addr with adjacent free blocks
  // Using map iterators for O(log n) neighbor lookup
  
  auto it = blocks_.find(addr);
  if (it == blocks_.end()) {
    return;
  }
  
  VABlock& currentBlock = it->second;
  if (!currentBlock.isFree) {
    return;  // Only coalesce free blocks
  }
  
  // Try to merge with the next block (if it's free and adjacent)
  uintptr_t currentEnd = currentBlock.startAddr + currentBlock.size;
  auto nextIt = blocks_.find(currentEnd);
  
  if (nextIt != blocks_.end() && nextIt->second.isFree) {
    VABlock& nextBlock = nextIt->second;
    
    MORI_APP_TRACE("VMMVAManager coalescing block at {:p} (size={}) with next block at {:p} (size={})", 
                   reinterpret_cast<void*>(currentBlock.startAddr), currentBlock.size,
                   reinterpret_cast<void*>(nextBlock.startAddr), nextBlock.size);
    
    // Remove next block's end address mapping
    endAddrToStartAddr_.erase(nextBlock.startAddr + nextBlock.size);
    
    // Extend current block
    currentBlock.size += nextBlock.size;
    
    // Update current block's end address mapping
    endAddrToStartAddr_.erase(currentEnd);
    endAddrToStartAddr_[currentBlock.startAddr + currentBlock.size] = currentBlock.startAddr;
    
    // Remove next block
    blocks_.erase(nextIt);
  }
  
  // Try to merge with the previous block (if it's free and adjacent)
  // Use endAddrToStartAddr_ to find the block that ends at our start address
  auto prevEndIt = endAddrToStartAddr_.find(currentBlock.startAddr);
  
  if (prevEndIt != endAddrToStartAddr_.end()) {
    uintptr_t prevStartAddr = prevEndIt->second;
    auto prevIt = blocks_.find(prevStartAddr);
    
    if (prevIt != blocks_.end() && prevIt->second.isFree) {
      VABlock& prevBlock = prevIt->second;
      
      MORI_APP_TRACE("VMMVAManager coalescing block at {:p} (size={}) with prev block at {:p} (size={})", 
                     reinterpret_cast<void*>(currentBlock.startAddr), currentBlock.size,
                     reinterpret_cast<void*>(prevBlock.startAddr), prevBlock.size);
      
      // Remove current block's end address mapping
      endAddrToStartAddr_.erase(currentBlock.startAddr + currentBlock.size);
      
      // Remove previous block's end address mapping (which points to current start)
      endAddrToStartAddr_.erase(currentBlock.startAddr);
      
      // Extend previous block
      prevBlock.size += currentBlock.size;
      
      // Update previous block's end address mapping
      endAddrToStartAddr_[prevBlock.startAddr + prevBlock.size] = prevBlock.startAddr;
      
      // Remove current block
      blocks_.erase(it);
    }
  }
  
  MORI_APP_TRACE("VMMVAManager coalescing completed, total blocks: {}", blocks_.size());
}

void VMMVAManager::CoalesceFreeBlocks() {
  // Legacy full-scan method - now less efficient than CoalesceAdjacentBlocks
  // Kept for compatibility but not recommended
  // The new map-based implementation makes this less useful
  
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (blocks_.empty()) {
    return;
  }
  
  // Iterate through all blocks and try to coalesce adjacent free blocks
  bool merged = true;
  while (merged) {
    merged = false;
    
    for (auto it = blocks_.begin(); it != blocks_.end(); ) {
      if (!it->second.isFree) {
        ++it;
        continue;
      }
      
      // Try to merge with next block
      uintptr_t currentEnd = it->second.startAddr + it->second.size;
      auto nextIt = blocks_.find(currentEnd);
      
      if (nextIt != blocks_.end() && nextIt->second.isFree) {
        MORI_APP_TRACE("VMMVAManager full coalesce: merging {:p} and {:p}",
                       reinterpret_cast<void*>(it->first),
                       reinterpret_cast<void*>(nextIt->first));
        
        // Remove end mappings
        endAddrToStartAddr_.erase(currentEnd);
        endAddrToStartAddr_.erase(nextIt->second.startAddr + nextIt->second.size);
        
        // Extend current block
        it->second.size += nextIt->second.size;
        
        // Update end mapping
        endAddrToStartAddr_[it->second.startAddr + it->second.size] = it->second.startAddr;
        
        // Remove next block
        blocks_.erase(nextIt);
        merged = true;
        // Don't increment iterator, check the same block again
      } else {
        ++it;
      }
    }
  }
}

size_t VMMVAManager::GetBlockSize(uintptr_t addr) const {
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto it = blocks_.find(addr);
  if (it == blocks_.end()) {
    return 0;
  }
  
  return it->second.size;
}

void VMMVAManager::GetStats(size_t& totalBlocks, size_t& freeBlocks, 
                           size_t& allocatedBlocks, size_t& totalFreeSpace, 
                           size_t& largestFreeBlock) const {
  std::lock_guard<std::mutex> lock(mutex_);
  
  totalBlocks = blocks_.size();
  freeBlocks = 0;
  allocatedBlocks = 0;
  totalFreeSpace = 0;
  largestFreeBlock = 0;
  
  for (const auto& entry : blocks_) {
    const VABlock& block = entry.second;
    if (block.isFree) {
      freeBlocks++;
      totalFreeSpace += block.size;
      largestFreeBlock = std::max(largestFreeBlock, block.size);
    } else {
      allocatedBlocks++;
    }
  }
}

bool VMMVAManager::IsValidAddress(uintptr_t addr) const {
  return addr >= baseAddr_ && addr < baseAddr_ + totalSize_;
}

void VMMVAManager::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  
  MORI_APP_INFO("VMMVAManager::Reset clearing all allocations");
  
  blocks_.clear();
  endAddrToStartAddr_.clear();
  
  // Reset to one large free block
  VABlock initialBlock(baseAddr_, totalSize_, true);
  blocks_[baseAddr_] = initialBlock;
  endAddrToStartAddr_[baseAddr_ + totalSize_] = baseAddr_;
}

}  // namespace application
}  // namespace mori
