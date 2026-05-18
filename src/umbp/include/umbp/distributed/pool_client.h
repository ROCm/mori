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
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mori/io/engine.hpp"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/types.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {

class PeerDramAllocator;
class PeerServiceServer;

// In the master-as-advisor design, PoolClient drives the Put/Get
// pipeline: master gives a routing advisory, then the writer talks
// directly to the peer (AllocateSlot → RDMA → CommitSlot or
// ResolveKey → RDMA).  Master holds no per-Put state.  The peer's
// allocator outbox is shipped to master via the heartbeat thread.
class PoolClient {
 public:
  explicit PoolClient(PoolClientConfig config);
  ~PoolClient();

  PoolClient(const PoolClient&) = delete;
  PoolClient& operator=(const PoolClient&) = delete;

  bool Init();
  void Shutdown();

  // Drop every locally-owned key, cancel in-flight pending writes, and
  // ask master to collapse this node's index via a full-sync empty
  // snapshot.  Returns immediately — convergence happens on the next
  // heartbeat tick (which is woken eagerly).  See ClearLocal() /
  // RequestClearFullSync() for the semantics of the write gate and the
  // best-effort caveat around in-flight remote reads.
  void Clear();

  const std::string& NodeId() const { return config_.master_config.node_id; }

  // Pin a caller-owned region for zero-copy RDMA.  Calls into the IO
  // engine's RegisterMemory; the descriptor is cached and looked up by
  // (ptr, size) on the Put/Get hot paths.
  bool RegisterMemory(void* ptr, size_t size);
  void DeregisterMemory(void* ptr);

  // Hot paths.  Both retry up to `max_route_retries` times when the
  // chosen peer reports ENOSPC (Put) or unknown-key (Get); each retry
  // adds the failed node to the exclude set.
  bool Put(const std::string& key, const void* src, size_t size);
  bool Get(const std::string& key, void* dst, size_t size);

  std::vector<bool> BatchPut(const std::vector<std::string>& keys,
                             const std::vector<const void*>& srcs,
                             const std::vector<size_t>& sizes);

  std::vector<bool> BatchGet(const std::vector<std::string>& keys, const std::vector<void*>& dsts,
                             const std::vector<size_t>& sizes);

  // Cluster-wide existence check — issues a RouteGet and reports
  // whether master surfaced any replica.  No RDMA, no lease bump.
  bool Exists(const std::string& key);
  std::vector<bool> BatchExists(const std::vector<std::string>& keys);

  void* SsdStagingPtr() const { return ssd_staging_buffer_.get(); }
  size_t SsdStagingSize() const { return config_.staging_buffer_size; }
  const std::vector<uint8_t>& SsdStagingMemDescBytes() const { return ssd_staging_mem_desc_bytes_; }

  MasterClient& Master();
  PeerDramAllocator* DramAllocator();

  bool IsInitialized() const;

  // External KV block events.
  bool ReportExternalKvBlocks(const std::vector<std::string>& hashes, TierType tier);
  bool RevokeExternalKvBlocks(const std::vector<std::string>& hashes, TierType tier);
  bool RevokeAllExternalKvBlocksAtTier(TierType tier);
  bool MatchExternalKv(const std::vector<std::string>& hashes,
                       std::vector<MasterClient::ExternalKvNodeMatch>* out_matches,
                       bool count_as_hit = false);
  bool GetExternalKvHitCounts(const std::vector<std::string>& hashes,
                              std::vector<MasterClient::ExternalKvHitCountEntry>* out_entries);

  struct SlotPlan {
    uint64_t slot_id = 0;
    std::vector<PageLocation> pages;
    uint64_t page_size = 0;
    std::vector<BufferMemoryDescBytes> descs;
  };

  // Per-entry outcome inside the Put pipeline; projected to `bool` at
  // BatchPut's return boundary (anything but kFailed is success).
  // `kAlreadyExists` (master- or peer-side dedup) is success-to-caller
  // but excluded from bandwidth metrics — no bytes on the wire.
  // Mirrors PeerDramAllocator::Outcome / proto *_ALREADY_EXISTS.
  enum class PutEntryOutcome { kFailed, kSucceeded, kAlreadyExists };

 private:
  PoolClientConfig config_;
  std::atomic<bool> initialized_{false};

  std::unique_ptr<MasterClient> master_client_;

  // Peer-side DRAM/HBM allocator.  Owned here because PoolClient is
  // the natural lifetime anchor for the per-process IO engine + DRAM
  // buffers; PeerServiceServer borrows it.
  std::unique_ptr<PeerDramAllocator> peer_alloc_;
  std::unique_ptr<PeerServiceServer> peer_service_;

  std::unique_ptr<mori::io::IOEngine> io_engine_;
  mori::io::MemoryDesc staging_mem_{};
  std::vector<mori::io::MemoryDesc> export_dram_mems_;
  std::unique_ptr<char[]> staging_buffer_;
  std::mutex staging_mutex_;

  std::unique_ptr<char[]> ssd_staging_buffer_;
  mori::io::MemoryDesc ssd_staging_mem_{};
  std::vector<uint8_t> ssd_staging_mem_desc_bytes_;

  // Lazy peer connections (one per remote node).  Engine descs cached
  // here; DRAM memory descs hydrated on first AllocateSlot/ResolveKey
  // response that references their buffer_index.
  struct PeerConnection {
    std::string peer_address;
    mori::io::EngineDesc engine_desc;
    std::vector<mori::io::MemoryDesc> dram_memories;  // indexed by buffer_index
    bool engine_registered = false;
    std::unique_ptr<void, void (*)(void*)> peer_stub{nullptr, +[](void*) {}};
    std::mutex ssd_op_mutex;
    mori::io::MemoryDesc ssd_staging_mem{};
    size_t ssd_staging_size = 0;
  };
  std::mutex peers_mutex_;
  std::unordered_map<std::string, std::unique_ptr<PeerConnection>> peers_;

  // Caller MUST NOT hold peers_mutex_; this helper acquires it.
  PeerConnection& GetOrConnectPeer(const std::string& node_id, const std::string& peer_address);
  // Hydrate peer.dram_memories[bd.buffer_index] for every entry in
  // `descs`.  Idempotent.  Acquires peers_mutex_ internally.
  void EnsureBufferDescsCached(PeerConnection& peer,
                               const std::vector<BufferMemoryDescBytes>& descs);

  // Same-tier RDMA scatter helpers (keep as much of the prior impl as
  // possible — the IO engine call shape is unchanged).
  bool RemoteDramScatterWrite(PeerConnection& peer, const std::vector<PageLocation>& pages,
                              uint64_t page_size, const void* src, size_t size, bool zero_copy);
  bool RemoteDramScatterRead(PeerConnection& peer, const std::vector<PageLocation>& pages,
                             uint64_t page_size, void* dst, size_t size, bool zero_copy);

  // Self-target fast paths (no RDMA, no peer RPC).
  bool LocalPutPages(const std::vector<PageLocation>& pages, uint64_t page_size, const void* src,
                     size_t size);
  bool LocalGetPages(const std::vector<PageLocation>& pages, uint64_t page_size, void* dst,
                     size_t size);

  // SSD path helpers — preserved so the SSD CommitSsdWrite slot
  // pre-allocation flow keeps working.  The peer side re-shaping of
  // those RPCs is not in scope for this commit.
  bool EnsurePeerServiceConnection(PeerConnection& peer);
  bool RemoteSsdWrite(PeerConnection& peer, const std::string& key, const void* src, size_t size,
                      bool zero_copy, uint32_t store_index = 0);
  bool RemoteSsdRead(PeerConnection& peer, const std::string& key, const std::string& location_id,
                     void* dst, size_t size, bool zero_copy);

  // Zero-copy registered memory regions.
  struct RegisteredRegion {
    void* base;
    size_t size;
    mori::io::MemoryDesc mem_desc;
  };
  std::mutex registered_mem_mutex_;
  std::vector<RegisteredRegion> registered_regions_;
  std::optional<std::pair<mori::io::MemoryDesc, size_t>> FindRegisteredMemory(const void* ptr,
                                                                              size_t size);

  // Single-attempt outcome from a peer call; mapped to PutEntryOutcome
  // by the caller (Partition / Allocate).
  enum class PutAttemptOutcome { kSuccess, kSuccessAlreadyExists, kRetry, kFatal };
  enum class GetAttemptOutcome { kSuccess, kRetry, kFatal };

  PutAttemptOutcome ExecuteLocalPut(const std::string& key, const void* src, size_t size,
                                    TierType tier);
  GetAttemptOutcome ExecuteLocalGet(const std::string& key, void* dst, size_t size);
  struct BatchPutItem {
    size_t index;
    const std::string* key;
    const void* src;
    size_t size;
    RoutePutResult route;
  };
  struct BatchGetItem {
    size_t index;
    const std::string* key;
    void* dst;
    size_t size;
    RouteGetResult route;
  };

  std::unordered_map<std::string, std::vector<BatchPutItem>> PartitionBatchPutTargets(
      const std::vector<std::string>& keys, const std::vector<const void*>& srcs,
      const std::vector<size_t>& sizes, const std::vector<std::optional<RoutePutResult>>& routes,
      std::vector<PutEntryOutcome>* results);

  struct TransferInstruction {
    size_t entry_index;
    mori::io::MemoryDesc local_desc;
    uint64_t local_offset;
    mori::io::MemoryDesc remote_desc;
    uint64_t remote_offset;
    uint64_t size;
  };

  struct RemotePutEntry {
    size_t result_index;
    const BatchPutItem* item;
    SlotPlan plan;
    uint64_t slot_id;
    std::optional<std::pair<mori::io::MemoryDesc, size_t>> zero_copy;
    bool use_staging = false;
    uint64_t staging_offset = 0;
    bool failed = false;
  };

  struct RemoteGetEntry {
    size_t result_index;
    const BatchGetItem* item;
    SlotPlan plan;
    std::optional<std::pair<mori::io::MemoryDesc, size_t>> zero_copy;
    bool use_staging = false;
    uint64_t staging_offset = 0;
    bool failed = false;
  };

  void ProcessRemoteBatchPut(const std::vector<BatchPutItem>& items,
                             std::vector<PutEntryOutcome>* results);
  void ProcessRemoteBatchGet(const std::vector<BatchGetItem>& items, std::vector<bool>* results);
  void ExecuteRemoteBatchPut(const std::vector<BatchPutItem>& items,
                             std::vector<PutEntryOutcome>* results, PeerConnection& peer,
                             ::umbp::UMBPPeer::Stub* stub);
  void ExecuteRemoteBatchGet(const std::vector<BatchGetItem>& items, std::vector<bool>* results,
                             PeerConnection& peer, ::umbp::UMBPPeer::Stub* stub);

  bool AllocateRemotePutEntries(const std::vector<BatchPutItem>& items,
                                ::umbp::UMBPPeer::Stub* stub, std::vector<RemotePutEntry>* entries,
                                std::vector<uint64_t>* abort_slots,
                                std::vector<PutEntryOutcome>* results);
  bool BuildRemotePutTransfers(std::vector<RemotePutEntry>& entries, PeerConnection& peer,
                               std::vector<TransferInstruction>* transfers,
                               uint64_t* staging_bytes);
  void ExecuteRemotePutTransfers(std::vector<RemotePutEntry>& entries,
                                 std::vector<TransferInstruction>& transfers,
                                 uint64_t staging_bytes);
  void FinalizeRemotePutEntries(std::vector<RemotePutEntry>& entries,
                                std::vector<uint64_t>& abort_slots,
                                std::vector<PutEntryOutcome>* results,
                                ::umbp::UMBPPeer::Stub* stub);

  bool PrepareRemoteGetEntries(const std::vector<BatchGetItem>& items, ::umbp::UMBPPeer::Stub* stub,
                               std::vector<RemoteGetEntry>* entries, std::vector<bool>* results);
  bool BuildRemoteGetTransfers(std::vector<RemoteGetEntry>& entries, PeerConnection& peer,
                               std::vector<TransferInstruction>* transfers,
                               uint64_t* staging_bytes);
  void ExecuteRemoteGetTransfers(std::vector<RemoteGetEntry>& entries,
                                 std::vector<TransferInstruction>& transfers,
                                 uint64_t staging_bytes);
  void FinalizeRemoteGetEntries(std::vector<RemoteGetEntry>& entries, std::vector<bool>* results);
};

}  // namespace mori::umbp
