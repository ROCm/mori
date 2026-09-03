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

/// The IUMBPClient implementation: this node's medium, plus master-led global
/// routing over an RDMA/MORI-IO data plane when a master address is configured.
///
/// With none it is an embedded, single-process store — the same backends and
/// the same transfer engine, with nothing routed, registered or heartbeated,
/// and a local miss as the final answer.  Configuration alone decides which,
/// which is why there is no separate local implementation.
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
  // One condition: the scratch arena must exist, because the remote direction
  // has nowhere to land otherwise — that is upstream's opt-in rule, and it
  // defaults to off.
  //
  // There is deliberately no medium condition. This used to also exclude SSD,
  // on the reasoning that ranged I/O maps object ranges onto pages a backend
  // publishes as in-process endpoints and SSD publishes storage refs instead.
  // That describes a backend nobody built: ssd_backend.h weighs a file endpoint
  // against staging and picks staging, so SsdBackend publishes ordinary
  // registered host pages like every other medium. Every ranged path works on
  // an SSD node. See the definition for what the medium does still change (the
  // device read is whole-object until D1) and doc/design-ssd-ranged-io.md.
  bool SupportsRangedIO() const override;

  bool RegisterMemory(uintptr_t ptr, size_t size,
                      mori::io::MemoryLocationType loc = mori::io::MemoryLocationType::CPU,
                      int device = -1,
                      MemoryRegistration mode = MemoryRegistration::kPinned) override;
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
  // The only buffers this class still allocates. Phase 2b moved medium pools
  // into the backends, but the ranged scratch arenas are not a medium: they are
  // client-side staging regions for objects fetched from, or assembled for,
  // another node, so they belong to whoever owns the PoolClient.
  //
  // Two separate arenas — one for remote ranged GET, one for remote ranged PUT
  // — each RDMA-registered and each under its own mutex in PoolClient, so a
  // remote get and a remote put run concurrently instead of serializing on one
  // lock (the load/offload overlap sglang's direct linker wants). Each is sized
  // to config.distributed.ranged_scratch_size; allocated only when that is > 0.
  void* ranged_get_scratch_ = nullptr;
  size_t ranged_get_scratch_size_ = 0;
  HostBufferHandle ranged_get_scratch_handle_;
  void* ranged_put_scratch_ = nullptr;
  size_t ranged_put_scratch_size_ = 0;
  HostBufferHandle ranged_put_scratch_handle_;
  // No master address => no routing => this node can never hold, fetch or
  // serve a remote key.  Set once at construction; read by SupportsRangedIO,
  // which is the one place the distinction changes an answer rather than just
  // a code path.
  bool local_only_ = false;
  std::unique_ptr<PoolClient> pool_client_;
  std::atomic<bool> closing_{false};
  mutable std::shared_mutex op_mutex_;
  bool closed_ = false;
};

}  // namespace mori::umbp
