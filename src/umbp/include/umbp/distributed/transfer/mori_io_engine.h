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

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "mori/io/engine.hpp"
#include "umbp/distributed/transfer/transfer_engine.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

// The mori-io (RDMA) implementation of TransferEngine, and the only place in
// UMBP that names mori::io::IOEngine.
//
// Everything moved here from PoolClient in Phase 6 is something that had to
// know about the wire:
//   * the IOEngine itself, its RDMA backend config, and engine-desc packing
//   * remote engine registration and the peer buffer-descriptor cache — a
//     peer's MemoryDesc is a transfer-layer fact, not a routing fact
//   * GroupTransfersByPair, now Plan(): collapsing a scatter into one transfer
//     per (srcMR, dstMR) is an RDMA optimization, which is why Plan is virtual
//   * the bounce buffer, because the transfer layer is the ONLY layer with a
//     completion signal.  A staging pool inside a storage backend has to guess
//     with a TTL, which is the deleted PrepareSsdRead lease under another name.
//
// Threading: RegisterMemory/Deregister, the remote caches and the bounce pool
// each take their own lock; Plan() is pure and takes none.
class MoriIoEngine final : public TransferEngine, public PeerDirectory {
 public:
  // `bounce_bytes` sizes the single staging region used when a caller's buffer
  // is not registered.  Zero disables staging: an unregistered endpoint is then
  // rejected by Plan rather than bounced.
  MoriIoEngine(std::string engine_key, mori::io::IOEngineConfig io_config, uint64_t bounce_bytes);
  ~MoriIoEngine() override;

  // Create the IOEngine + RDMA backend and register the bounce buffer.
  // Returns false if the engine could not be created; the object is then
  // inert (Active() == false) rather than unusable — a node with no RDMA still
  // registers memory (host_ptr only) and serves purely local transfers through
  // a different engine.
  bool Init();
  void Shutdown();

  bool Active() const { return io_engine_ != nullptr; }

  // ---- TransferEngine ----

  const char* Name() const override { return "MoriIoEngine"; }

  TransferRef RegisterMemory(void* base, size_t size, mori::io::MemoryLocationType loc,
                             int device) override;
  void Deregister(const TransferRef& ref) override;

  // True iff exactly one endpoint is a registered REMOTE memory and the other
  // is local — either registered (zero copy) or, when a bounce buffer is
  // configured, merely addressable (staged).
  bool CanHandle(const TransferRef& src, const TransferRef& dst) const override;

  // Group by (src MR id, dst MR id), first-appearance order, coalescing
  // segments that are exactly contiguous on BOTH sides.  Items whose local side
  // is unregistered are grouped separately and assigned bounce offsets; a
  // bounce group is chunked to fit the pool, and an item larger than the whole
  // pool is rejected.
  TransferPlanSet Plan(const std::vector<TransferItem>& items) const override;

  std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan> plans) override;

  // ---- PeerDirectory ----
  //
  // mori-io's half of "learning how to reach a node": the peer's EngineDesc has
  // to be registered with the local engine and its published MemoryDescs
  // unpacked before any transfer can name them.  This state used to live in
  // PoolClient::PeerConnection, which meant the client had to understand
  // descriptors it never dereferenced.

  std::vector<uint8_t> PackedLocalEngineDesc() const override;
  bool EnsureRemoteEngine(const std::string& node_id,
                          const std::string& packed_engine_desc) override;
  bool HasRemoteEngine(const std::string& node_id) const override;
  void ForgetRemote(const std::string& node_id) override;
  void CacheRemoteBuffers(const std::string& node_id,
                          const std::vector<BufferMemoryDescBytes>& descs) override;
  bool HasRemoteBuffers(const std::string& node_id) const override;
  std::vector<TransferRef> RemoteBufferSnapshot(const std::string& node_id,
                                                uint32_t backend_id) const override;

  const std::string& LocalEngineKey() const { return local_engine_key_; }

 private:
  class RdmaHandle;

  struct RemoteNode {
    mori::io::EngineDesc engine_desc;
    bool engine_registered = false;
    // One shelf per peer-side backend, each indexed by that backend's own
    // buffer_index.  A single flat vector here was the client half of the
    // mixed-media corruption: two backends both publish a buffer 0, and
    // whichever descriptor arrived first answered for both.
    std::array<std::vector<TransferRef>, kMaxBackendsPerPeer> buffers;
  };

  // Post `plan` and hand the statuses to `handle`.  Caller owns sequencing.
  bool PostPlans(const std::vector<TransferPlan>& plans, RdmaHandle* handle);

  // Run one bounce plan to completion: stage in (kPush), post, wait, stage out
  // (kPull), release.  Returns false and fills `failure` on any error.
  bool RunBouncePlanInline(const TransferPlan& plan, TransferFailure* failure);

  std::string local_engine_key_;
  mori::io::IOEngineConfig io_config_;
  std::unique_ptr<mori::io::IOEngine> io_engine_;

  mutable std::mutex remotes_mutex_;
  std::unordered_map<std::string, RemoteNode> remotes_;

  // Bounce pool: one region, one mutex.  A plan that needs it completes inside
  // Submit (see TransferEngine::Submit), so the lock is never held across a
  // return and a submit-all loop over several peers cannot deadlock on it.
  uint64_t bounce_size_ = 0;
  std::unique_ptr<char[]> bounce_buffer_;
  TransferRef bounce_ref_;
  std::mutex bounce_mutex_;
};

}  // namespace mori::umbp
