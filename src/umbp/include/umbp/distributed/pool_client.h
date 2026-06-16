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
class PeerSsdManager;
class SsdCopyPipeline;

// Short name for log output. Generic FAILED maps to "FAILED" — the
// detailed reason for that case lives in the peer's allocator log.
inline const char* OutcomeName(::umbp::AllocateSlotOutcome o) {
  switch (o) {
    case ::umbp::ALLOCATE_SLOT_OUTCOME_UNSPECIFIED:
      return "UNSPECIFIED";
    case ::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALLOCATED:
      return "SUCCESS_ALLOCATED";
    case ::umbp::ALLOCATE_SLOT_OUTCOME_SUCCESS_ALREADY_EXISTS:
      return "SUCCESS_ALREADY_EXISTS";
    case ::umbp::ALLOCATE_SLOT_OUTCOME_FAILED:
      return "FAILED";
    case ::umbp::ALLOCATE_SLOT_OUTCOME_FAILED_NO_SPACE:
      return "NO_SPACE";
    default:
      return "UNKNOWN";
  }
}

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

  // Drop every locally-owned key, cancel in-flight pending writes, clear
  // external HiCache placement, and ask master to collapse this node's
  // index via full-sync empty snapshots.  Returns true when the target
  // empty state is reached: vacuously so if the client is uninitialised
  // or no master is configured, otherwise only after master acknowledges
  // both clear full-sync snapshots before this call returns.  Returns
  // false only on an actual synchronous full-sync RPC failure; the
  // heartbeat loop will then retry until convergence.  See ClearLocal()
  // / ClearFullSync() for the semantics of the write gate and the
  // best-effort caveat around in-flight remote reads.
  bool Clear();

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
  size_t SsdStagingSize() const { return config_.ssd_staging_buffer_size; }
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
  // Aggregates all SUCCESS_* / FAILED_* variants of
  // PeerDramAllocator::Outcome / proto AllocateSlotOutcome into 3 buckets;
  // the specific failure reason is logged at the call site, not propagated.
  enum class PutEntryOutcome { kFailed, kSucceeded, kAlreadyExists };

 private:
  PoolClientConfig config_;
  std::atomic<bool> initialized_{false};

  std::unique_ptr<MasterClient> master_client_;

  // Peer-side DRAM/HBM allocator.  Owned here because PoolClient is
  // the natural lifetime anchor for the per-process IO engine + DRAM
  // buffers; PeerServiceServer borrows it.
  std::unique_ptr<PeerDramAllocator> peer_alloc_;
  // Peer-side SSD tier owner.  Built only when config_.ssd.enabled; registered
  // with MasterClient as an owned-location source + SSD capacity provider.
  std::unique_ptr<PeerSsdManager> peer_ssd_;
  // Async copy-on-commit pipeline.  Built only when SSD is enabled;
  // borrows peer_alloc_ (pin source) + peer_ssd_ (write target).  Started in
  // Init, stopped in Shutdown (after the peer service so no commit can enqueue
  // into a stopped pipeline).
  std::unique_ptr<SsdCopyPipeline> ssd_copy_pipeline_;
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
  // Same as above, but the caller MUST already hold peers_mutex_.  Lets the
  // Build* helpers hydrate AND snapshot remote descs inside a single lock
  // window so a concurrent hydrate (which may resize dram_memories) cannot
  // race the reads.
  void EnsureBufferDescsCachedLocked(PeerConnection& peer,
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

  bool EnsurePeerServiceConnection(PeerConnection& peer);

  // Outcome of one remote SSD get attempt.  kRetry is a retryable transient
  // condition (NO_SLOT or a reader-local lease expiry); kMiss (NOT_FOUND) is a
  // definitive miss; kError is the not-served, not-retried catch-all (rpc
  // failure incl. DEADLINE_EXCEEDED, size mismatch, RDMA failure) — not strictly
  // a hard error.  Keeping kMiss distinct lets BatchGet avoid surfacing a
  // non-served key as a cache miss.
  enum class SsdGetOutcome { kSuccess, kMiss, kRetry, kError };

  // Remote SSD get path (reader != owner): key-based PrepareSsdRead -> RDMA out
  // of the peer's published SSD staging buffer -> best-effort ReleaseSsdLease.
  SsdGetOutcome RemoteSsdReadOnce(PeerConnection& peer, const std::string& key, void* dst,
                                  size_t size);
  void ReleaseSsdLeaseBestEffort(::umbp::UMBPPeer::Stub* stub, uint64_t lease_id);

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
  // Self-target SSD get: read straight from the local SSD tier into the user
  // buffer (no staging / RDMA / lease).
  GetAttemptOutcome ExecuteLocalSsdGet(const std::string& key, void* dst, size_t size);
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

  // A set of page transfers sharing the same (localMR, remoteMR) pair.  All
  // its pages collapse into ONE outer IO transfer whose inner offset/size
  // vectors are scatter-gather segments, cutting CQE/status/post count vs
  // one-transfer-per-page.  `entry_indices` is the de-duplicated list of
  // entries contributing pages to this group (for per-key failure mapping).
  struct PairGroup {
    mori::io::MemoryDesc local_desc;
    mori::io::MemoryDesc remote_desc;
    std::vector<size_t> local_offsets;
    std::vector<size_t> remote_offsets;
    std::vector<size_t> sizes;
    std::vector<size_t> entry_indices;
  };
  // Group `active` page transfers by (local_desc.id, remote_desc.id),
  // preserving first-appearance order (stable).  Assumes a fixed page size
  // per (localMR, remoteMR) pair (1 buffer == 1 page size in the current
  // model); mixed page sizes would require folding page size into the key.
  static std::vector<PairGroup> GroupTransfersByPair(
      const std::vector<TransferInstruction>& active);

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
  // Remote SSD reads (one staging slot + lease per key) with bounded retry
  // (default off) on transient NO_SLOT / reader-local lease expiry; an rpc
  // failure is hard not-served, and a NOT_FOUND short-circuits as a miss.
  void ProcessRemoteSsdBatchGet(const std::vector<BatchGetItem>& items, std::vector<bool>* results);

  // Periodic SSD metrics provider (registered in Init() when SSD is enabled, run
  // once per metrics flush tick in the metrics thread).  Reads the cheap atomics
  // on the pipeline / PeerSsdManager / PeerService and ships counter deltas + a
  // staging gauge, keeping AddCounter off the commit/read hot paths.  The
  // last-shipped values below make each tick report only the delta.
  void PublishSsdMetrics();
  struct SsdMetricsLastShipped {
    uint64_t copy_enqueued = 0, copy_succeeded = 0, copy_failed = 0;
    uint64_t copy_dropped_queue_full = 0, copy_dropped_stopped = 0;
    uint64_t read_ok = 0, read_not_found = 0, read_size_too_large = 0, read_error = 0;
    uint64_t read_no_slot = 0;
    uint64_t copy_bytes = 0, read_bytes = 0;
    uint64_t evict_rounds = 0, evict_victims = 0, evict_bytes_freed = 0, evict_backend_failed = 0;
    uint64_t staging_expired_reclaims = 0, staging_slot_full_rejects = 0;
  };
  SsdMetricsLastShipped ssd_metrics_last_;
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
