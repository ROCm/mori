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

#include <cstdint>
#include <optional>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp {

// The page-allocation seam for the PAGED media -- DRAM and HBM.
//
// PageBackend is medium-agnostic above PageMemorySource: the two media differ
// in how the pool is obtained and in the loc/device stamped on published refs,
// never in how pages are handed out.  So one pool implementation serves both,
// and swapping it swaps it for both.
//
// WHY ONLY HERE.  The SSD side already has its plug-in seam one level up --
// TierBackend, with five implementations (SSDTier, ShardedSsdTier,
// SpdkSsdTier, SpdkProxyTier, DummySsdTier) -- and its allocation models do
// not fit this contract anyway: the SPDK OffsetAllocator hands out
// variable-size CONTIGUOUS byte extents in a single space, and the segment log
// is an append cursor whose reclamation is whole-segment GC, not per-record
// free.  A single interface over all three would be their union, which is not
// a contract.  This one exists because the opposite is true here: one
// implementation today, several credible ones (bitmap, binned free-list, slab,
// hierarchical bitmap), and exactly one call site to switch them at.
//
// THREAD SAFETY: implementations hold NO internal lock.  The caller
// serializes -- PageBackend::mutex_ on the peer, ClientRegistry::mutex_ on
// master.  Stated here rather than left to accident: a lock-free
// implementation would want to relax this, and that has to be a deliberate
// decision rather than something an implementation quietly assumes.
//
// ALL-OR-NOTHING: a failed Allocate must leave the pool exactly as it found
// it.  Callers treat nullopt as "no space" and do not clean up after it.
class PagePool {
 public:
  virtual ~PagePool() = default;

  PagePool(const PagePool&) = delete;
  PagePool& operator=(const PagePool&) = delete;

  // Hand out `num_pages` pages, or nullopt if the pool cannot.
  //
  // The returned pages need NOT be contiguous and need not share a buffer --
  // callers address them through PageLocation and never assume adjacency.
  // Whether an implementation prefers contiguous runs (better for a single
  // RDMA), packs within one buffer, or scatters across buffers is POLICY that
  // belongs to the implementation; it is deliberately not in this contract, so
  // a new pool can change placement without touching a caller.
  virtual std::optional<std::vector<PageLocation>> Allocate(uint32_t num_pages) = 0;

  // Return pages to the pool.  Must tolerate an empty list, and must ignore
  // out-of-range entries rather than trap -- teardown paths can present pages
  // from a pool generation that has already been reset.
  virtual void Deallocate(const std::vector<PageLocation>& pages) = 0;

  // Capacity, in bytes.  Called every heartbeat, so keep it O(buffers).
  virtual uint64_t TotalBytes() const = 0;
  virtual uint64_t AvailableBytes() const = 0;
  uint64_t UsedBytes() const { return TotalBytes() - AvailableBytes(); }

  virtual uint64_t PageSize() const = 0;

  // Geometry, for publishing transfer endpoints.  Deliberately NOT a handle to
  // internal per-buffer state: PageBackend needs only each buffer's extent to
  // build its TransferRefs, and exposing more would tie this interface to one
  // implementation's bookkeeping (a bitmap's BufferState, say) and make a
  // free-list or slab implementation awkward to write.
  virtual size_t NumBuffers() const = 0;
  virtual uint32_t BufferPageCount(size_t buffer_index) const = 0;

 protected:
  PagePool() = default;
};

}  // namespace mori::umbp
