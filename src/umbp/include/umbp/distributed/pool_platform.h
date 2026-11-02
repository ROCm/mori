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

#include <memory>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/peer/backend/medium_backend.h"
#include "umbp/distributed/transfer/transfer_engine.h"
#include "umbp/local/host_mem_allocator.h"

namespace mori::umbp {

// Allocation and transport composition; KV placement and control stay in PoolClient.
// One instance per PoolClient. Its owning DistributedClient may share it for
// allocations; engine observers must not be shared with another PoolClient.
class PoolPlatform {
 public:
  struct Engines {
    std::unique_ptr<TransferEngine> transfer;
    PeerDirectory* peers = nullptr;
  };
  virtual ~PoolPlatform() = default;
  virtual Engines CreateTransferEngine(const PoolClientConfig& config,
                                       const std::vector<BackendInstanceConfig>& backends) = 0;
  virtual std::unique_ptr<MediumBackend> CreateBackend(const BackendInstanceConfig& backend,
                                                       uint64_t page_size,
                                                       const PoolClientConfig& config) = 0;
  // Local transfers need real pointer classification or simulated registration metadata.
  virtual TransferRef LocalUserBytes(void* ptr, size_t size,
                                     const TransferRef& registered_ref) const = 0;
  virtual HostBufferHandle AllocateHost(size_t size, const HostBufferOptions& options = {}) = 0;
  virtual void FreeHost(HostBufferHandle& buffer) = 0;
  virtual void AddHostGatherRegion(void*, size_t) {}
  virtual void ClearHostGatherRegions() {}
};

std::shared_ptr<PoolPlatform> MakePoolPlatform(const UMBPSimConfig& sim);

}  // namespace mori::umbp
