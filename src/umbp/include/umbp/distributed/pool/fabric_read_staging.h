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
#include <memory>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp {

class BackendRegistry;
class PageMemorySource;
class TransferEngine;
struct PoolResolvedEntry;

// An owner-local, bounded HBM bridge for completion-pinned remote reads of
// DRAM/SSD. It is not a placement backend: the object's storage tier and key
// ownership do not change when a read temporarily publishes its HBM copy.
class FabricReadStaging {
 public:
  static constexpr uint32_t kBackendId = kFabricReadStagingBackendId;

  FabricReadStaging(TransferEngine* engine, std::unique_ptr<PageMemorySource> source,
                    uint64_t page_size, uint64_t capacity_bytes);
  ~FabricReadStaging();
  FabricReadStaging(const FabricReadStaging&) = delete;
  FabricReadStaging& operator=(const FabricReadStaging&) = delete;

  // Allocate/register the arena once. Production requires a fabric-exportable
  // GPU arena; only the simulator may omit its physical fabric handle.
  // The engine and backend registry must outlive this manager. Active grants
  // surviving manager destruction retain the arena permanently.
  bool Init(BackendRegistry* backends, bool require_fabric_handle = true);

  // Configure before serving reads. Retain completed immutable range groups
  // within this budget, sharing preparation between identical concurrent reads.
  // Cache eviction never revokes a published read's completion hold.
  void EnableCache(uint64_t capacity_bytes);
  struct CacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t joins = 0;
    uint64_t bytes = 0;
  };
  CacheStats GetCacheStats() const;
  // Probe using trusted backend size/version metadata, without resolving SSD
  // pages. Only a complete cached group is published by this fast path.
  bool TryCached(std::vector<PoolResolvedEntry>* entries, const std::vector<ObjectExtent>* extents,
                 bool include_descs, const std::vector<std::vector<ObjectExtent>>* ranges);

  // Replace found CPU page endpoints with HBM pages after the local copy has
  // completed. GPU endpoints and misses are unchanged. With no extents, copy
  // the source's covered window; an explicit length==0 requests the whole
  // object. Coverage remains object-relative and size remains the object size.
  // All CPU reservations in one batch are admitted together, without waiting;
  // temporary pressure marks them kBusy; a batch larger than the arena and
  // other permanent failures mark them kFailed. Reserved pages are released
  // when both the completion pins and any retained cache ownership drop.
  bool Stage(std::vector<PoolResolvedEntry>* entries,
             const std::vector<ObjectExtent>* extents = nullptr, bool include_descs = true,
             const std::vector<std::vector<ObjectExtent>>* ranges = nullptr);

  std::vector<BufferMemoryDescBytes> Descriptors() const;
  uint64_t CapacityBytes() const;
  uint64_t InUseBytes() const;
  bool HasQuarantinedTransfers() const;
  void QuarantineBackingMemory() noexcept;

 private:
  struct State;
  struct BatchHold;
  struct InFlight;
  struct Cache;
  struct Signature;
  bool DescribeRequest(const std::vector<PoolResolvedEntry>& entries,
                       const std::vector<ObjectExtent>* extents,
                       const std::vector<std::vector<ObjectExtent>>* ranges, bool require_pages,
                       Signature* signature) const;
  bool StageUncached(std::vector<PoolResolvedEntry>* entries,
                     const std::vector<ObjectExtent>* extents, bool include_descs,
                     const std::vector<std::vector<ObjectExtent>>* ranges);
  std::shared_ptr<State> state_;
  std::unique_ptr<Cache> cache_;
};

}  // namespace mori::umbp
