// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
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

  // Replace found CPU page endpoints with HBM pages after the local copy has
  // completed. GPU endpoints and misses are unchanged. With no extents, copy
  // the source's covered window; an explicit length==0 requests the whole
  // object. Coverage remains object-relative and size remains the object size.
  // All CPU reservations in one batch are admitted together, without waiting;
  // temporary pressure marks them kBusy; a batch larger than the arena and
  // other permanent failures mark them kFailed. Returned holds
  // release the reserved pages when the last completion pin drops.
  bool Stage(std::vector<PoolResolvedEntry>* entries,
             const std::vector<ObjectExtent>* extents = nullptr, bool include_descs = true);

  std::vector<BufferMemoryDescBytes> Descriptors() const;
  uint64_t CapacityBytes() const;
  uint64_t InUseBytes() const;
  bool HasQuarantinedTransfers() const;
  void QuarantineBackingMemory() noexcept;

 private:
  struct State;
  struct BatchHold;
  struct InFlight;
  std::shared_ptr<State> state_;
};

}  // namespace mori::umbp
