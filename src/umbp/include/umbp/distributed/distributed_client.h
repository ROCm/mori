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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#include "umbp/distributed/pool_client.h"
#include "umbp/local/host_mem_allocator.h"
#include "umbp/umbp_client.h"

namespace mori::umbp {

/// Distributed IUMBPClient implementation — master-led global routing
/// with RDMA/MORI-IO data plane.  All routing decisions go through the
/// Master; this client does not use LocalStorageManager or LocalBlockIndex.
class DistributedClient : public IUMBPClient {
 public:
  explicit DistributedClient(const UMBPConfig& config);
  ~DistributedClient() override;

  // ---- IUMBPClient interface ----
  bool Put(const std::string& key, uintptr_t src, size_t size) override;
  bool Get(const std::string& key, uintptr_t dst, size_t size) override;
  bool Exists(const std::string& key) const override;

  std::vector<bool> BatchPut(const std::vector<std::string>& keys,
                             const std::vector<uintptr_t>& srcs,
                             const std::vector<size_t>& sizes) override;
  std::vector<bool> BatchPutWithDepth(const std::vector<std::string>& keys,
                                      const std::vector<uintptr_t>& srcs,
                                      const std::vector<size_t>& sizes,
                                      const std::vector<int>& depths) override;
  std::vector<bool> BatchGet(const std::vector<std::string>& keys,
                             const std::vector<uintptr_t>& dsts,
                             const std::vector<size_t>& sizes) override;
  // Not implemented — see src/umbp/doc/design-tree-connector-port.md §5.  The
  // transfer layer can already express a range (TransferItem carries
  // src_offset/dst_offset/size); what is missing is the object-range to
  // page-range mapping and the master-side metadata question.
  std::vector<bool> BatchGetRanges(const std::vector<std::string>& keys,
                                   const std::vector<std::vector<uintptr_t>>& dsts,
                                   const std::vector<std::vector<size_t>>& sizes,
                                   const std::vector<std::vector<size_t>>& src_offsets) override;
  std::vector<bool> BatchPutRanges(const std::vector<std::string>& keys,
                                   const std::vector<size_t>& object_sizes,
                                   const std::vector<std::vector<uintptr_t>>& srcs,
                                   const std::vector<std::vector<size_t>>& sizes,
                                   const std::vector<std::vector<size_t>>& dst_offsets) override;
  std::vector<bool> BatchExists(const std::vector<std::string>& keys) const override;
  size_t BatchExistsConsecutive(const std::vector<std::string>& keys) const override;

  bool Clear() override;
  bool Flush() override;
  void Close() override;
  bool IsDistributed() const override;
  UMBPDeploymentMode GetDeploymentMode() const override { return UMBPDeploymentMode::Distributed; }
  // Two independent conditions, both required. The arena must exist, because
  // the remote direction has nowhere to land otherwise — that is upstream's
  // opt-in rule. And the selected medium must be able to serve it: ranged I/O
  // maps object ranges onto tier pages a backend publishes as in-process
  // endpoints, and SSD publishes storage refs instead, so it is the one medium
  // this cannot serve. Upstream spells the medium half as `!ssd.enabled`
  // because it predates the single-medium selector.
  bool SupportsRangedIO() const override;

  bool RegisterMemory(uintptr_t ptr, size_t size,
                      mori::io::MemoryLocationType loc = mori::io::MemoryLocationType::CPU,
                      int device = -1) override;
  void DeregisterMemory(uintptr_t ptr) override;

  bool ReportExternalKvBlocks(const std::vector<std::string>& hashes, TierType tier) override;
  bool RevokeExternalKvBlocks(const std::vector<std::string>& hashes, TierType tier) override;
  bool RevokeAllExternalKvBlocksAtTier(TierType tier) override;
  std::vector<ExternalKvMatch> MatchExternalKv(const std::vector<std::string>& hashes,
                                               bool count_as_hit = false) override;
  std::vector<ExternalKvHitCountEntry> GetExternalKvHitCounts(
      const std::vector<std::string>& hashes) override;

 private:
  UMBPConfig config_;
  // The only buffer this class still allocates. Phase 2b moved medium pools
  // into the backends, but the ranged scratch arena is not a medium: it is a
  // client-side staging region for objects fetched from, or assembled for,
  // another node, so it belongs to whoever owns the PoolClient.
  void* ranged_scratch_ = nullptr;
  size_t ranged_scratch_size_ = 0;
  HostBufferHandle ranged_scratch_handle_;
  std::unique_ptr<PoolClient> pool_client_;
  std::atomic<bool> closing_{false};
  mutable std::shared_mutex op_mutex_;
  bool closed_ = false;
};

}  // namespace mori::umbp
