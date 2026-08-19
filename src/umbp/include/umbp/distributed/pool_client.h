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
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mori/io/engine.hpp"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/peer/backend/medium_backend.h"
#include "umbp/distributed/transfer/transfer_engine.h"
#include "umbp/distributed/types.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {

namespace benchmark {
class WorkloadTraceRecorder;
}

class PeerServiceServer;
class PeerPool;

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
  // (ptr, size) on the Put/Get hot paths. `loc`/`device` describe the
  // caller's allocation (CPU, or a GPU ordinal for a device-resident
  // buffer) so the transfer layer can route to HbmCopyEngine instead of
  // assuming host memory.
  bool RegisterMemory(void* ptr, size_t size,
                      mori::io::MemoryLocationType loc = mori::io::MemoryLocationType::CPU,
                      int device = -1);
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

  MasterClient& Master();

  // Every named storage backend live on this node.  Callers use them through
  // MediumBackend — no concrete backend type is named outside PoolClient::Init.
  BackendRegistry& Backends();
  TierTransitionMetrics TransitionMetrics() const;

  // Legacy default medium: the first configured backend's tier.  Valid after
  // Init and retained for callers that have not adopted named instances.
  TierType Medium() const { return medium_; }

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
    // Which of the peer's backends `pages` index into.  A slot lives entirely
    // in one medium, so one id covers the whole plan; without it the
    // backend-local buffer_index does not name a buffer (see
    // BufferMemoryDescBytes).
    uint32_t backend_id = 0;
  };

  // Per-entry outcome inside the Put pipeline; projected to `bool` at
  // BatchPut's return boundary (anything but kFailed is success).
  // `kAlreadyExists` (master- or peer-side dedup) is success-to-caller
  // but excluded from bandwidth metrics — no bytes on the wire.
  // Aggregates all SUCCESS_* / FAILED_* variants of
  // AllocateOutcome / proto AllocateSlotOutcome into 3 buckets;
  // the specific failure reason is logged at the call site, not propagated.
  enum class PutEntryOutcome { kFailed, kSucceeded, kAlreadyExists };

 private:
  PoolClientConfig config_;
  std::atomic<bool> initialized_{false};

  std::unique_ptr<MasterClient> master_client_;

  // Every storage medium live on this node.  Owned here because PoolClient is
  // the natural lifetime anchor for the per-process IO engine + backend pools.
  // MasterClient borrows it for heartbeat aggregation; default_pool_ borrows it
  // for logical placement and peer/local dispatch.
  BackendRegistry registry_;

  // The implicit default logical pool. It borrows registry_, owns placement
  // state and policy, and is the dispatch surface for local and peer RPC paths.
  std::unique_ptr<PeerPool> default_pool_;
  std::unique_ptr<benchmark::WorkloadTraceRecorder> workload_recorder_;

  // Legacy default tier, cached from the first configured backend. Read by
  // re-cache paths that do not yet carry a PoolPolicy decision.
  TierType medium_ = TierType::DRAM;

  std::unique_ptr<PeerServiceServer> peer_service_;

  // The one byte-moving path (design doc §4).  PoolClient owns the engine;
  // backends receive it narrowed to MemoryRegistrar at Init, so they can
  // publish endpoints but cannot transfer.
  //
  // `peers_` (below) is the control-plane half of talking to another node —
  // the gRPC stub.  `peer_directory_` is the TRANSPORT half, and it is a
  // non-owning pointer to the sub-engine that implements PeerDirectory, not to
  // a concrete engine type: adding a second remote transport means implementing
  // that interface, not editing this file.  Null when this node has no remote
  // transport configured, in which case only local transfers are servicable.
  std::unique_ptr<TransferEngine> transfer_engine_;
  PeerDirectory* peer_directory_ = nullptr;

  // Lazy peer connections (one per remote node).  Purely control plane since
  // Phase 6: the peer's engine desc and buffer descriptors moved into
  // MoriIoEngine, which is the layer that uses them.
  struct PeerConnection {
    std::string node_id;
    std::string peer_address;
    std::unique_ptr<void, void (*)(void*)> peer_stub{nullptr, +[](void*) {}};
    // Guards first-contact connection setup (stub creation + engine/buffer-desc
    // hydration via GetPeerInfo) in EnsurePeerServiceConnection.
    std::mutex conn_mutex;
  };
  std::mutex peers_mutex_;
  std::unordered_map<std::string, std::unique_ptr<PeerConnection>> peers_;

  // Caller MUST NOT hold peers_mutex_; this helper acquires it.
  PeerConnection& GetOrConnectPeer(const std::string& node_id, const std::string& peer_address);

  bool EnsurePeerServiceConnection(PeerConnection& peer);

  // Endpoint for a caller-supplied buffer, plus the offset of `ptr` within it:
  // the registered region when RegisterMemory pinned it (zero copy, ref covers
  // the whole region), otherwise a plain host-bytes ref at offset 0 that the
  // engine will stage through its own bounce buffer.
  //
  // One ref instead of the old optional<MemoryDesc> + use_staging + staging
  // offset triple, because whether a transfer needs staging is the transfer
  // layer's decision, not the client's.
  std::pair<TransferRef, uint64_t> UserBufferRef(void* ptr, size_t size) const;

  // Zero-copy registered memory regions.
  struct RegisteredRegion {
    void* base;
    size_t size;
    TransferRef ref;
  };
  mutable std::mutex registered_mem_mutex_;
  std::vector<RegisteredRegion> registered_regions_;
  // The registered ref covering [ptr, ptr+size) plus the offset of ptr within
  // it, or nullopt when the region was never registered.
  std::optional<std::pair<TransferRef, size_t>> FindRegisteredMemory(const void* ptr,
                                                                     size_t size) const;

  // Single-attempt outcome from a peer call; mapped to PutEntryOutcome
  // by the caller (Partition / Allocate).
  enum class PutAttemptOutcome { kSuccess, kSuccessAlreadyExists, kRetry, kFatal };
  enum class GetAttemptOutcome { kSuccess, kRetry, kFatal };

  PutAttemptOutcome ExecuteLocalPut(const std::string& key, const void* src, size_t size,
                                    TierType tier, const std::string& logical_tier = {});
  GetAttemptOutcome ExecuteLocalGet(const std::string& key, void* dst, size_t size);

  // One TransferItem per page between a caller buffer and `backend`'s own
  // buffers.  `to_backend` is Put (user -> pages), false is Get (pages -> user).
  // False when the backend publishes no in-process endpoint for a referenced
  // buffer — that medium cannot serve the access here, and the caller routes
  // elsewhere rather than reporting a miss.
  bool BuildLocalPageTransfers(MediumBackend* backend, const std::vector<PageLocation>& pages,
                               uint64_t page_size, void* user, size_t size, bool to_backend,
                               std::vector<TransferItem>* items);
  // After a successful remote DRAM fetch, if cache_remote_fetches is enabled and
  // the admission gate admits the block, enqueue it for asynchronous install into
  // this node's local DRAM tier (see ReCacheWorkerLoop). The install (DRAM
  // Allocate + copy + Commit→KvEvent::ADD publish) happens OFF the Get critical
  // path so it does not add latency to concurrent Gets. Idempotent
  // (kSuccessAlreadyExists) and best-effort: the queue is bounded and drops on
  // full; failure does not affect the returned Get result.
  void MaybeReCacheAfterRemote(const std::string& key, const void* src, size_t size);
  // Background worker that drains recache_queue_ and performs the DRAM install
  // via ExecuteLocalPut. Started in Init (when cache_remote_fetches), stopped in
  // Shutdown before peer_alloc_ is torn down.
  void ReCacheWorkerLoop();

  // Async re-cache install queue. MaybeReCacheAfterRemote copies the block bytes
  // into a job here (the source buffer is not owned past the Get return), and the
  // worker thread installs it. Bounded to keep memory + publish churn in check.
  struct ReCacheJob {
    std::string key;
    std::unique_ptr<char[]> bytes;
    size_t size = 0;
  };
  std::deque<ReCacheJob> recache_queue_;
  std::mutex recache_mutex_;
  std::condition_variable recache_cv_;
  std::thread recache_worker_;
  bool recache_stop_ = false;
  size_t recache_queue_max_ = 1024;
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

  // Routing plan for one BatchGet: which keys go to which target.  Pure
  // grouping — no IO is issued here.  Remote reads go through the batched RDMA
  // path (remote_groups, keyed by peer node_id); self-target reads are
  // deferred (collected as indices) so ExecuteBatchGetPlan can place them
  // inside the remote-DRAM in-flight window when overlapping.
  struct BatchGetPlan {
    std::unordered_map<std::string, std::vector<BatchGetItem>> remote_groups;
    std::vector<size_t> local_indices;
  };

  // Pure grouping for one BatchPut (no IO, no local put executed; mirrors
  // BatchGetPlan).  local_items hold full items (not bare indices) so the
  // deferred local memcpy keeps its route tier.
  struct BatchPutPlan {
    std::unordered_map<std::string, std::vector<BatchPutItem>> remote_groups;
    std::vector<BatchPutItem> local_items;
  };

  // Group a BatchPut's routes into a BatchPutPlan; master-side dedup and
  // zero-size skips are projected straight into *results.
  BatchPutPlan PartitionBatchPutTargets(const std::vector<std::string>& keys,
                                        const std::vector<const void*>& srcs,
                                        const std::vector<size_t>& sizes,
                                        const std::vector<std::optional<RoutePutResult>>& routes,
                                        std::vector<PutEntryOutcome>* results);
  // Execute a BatchPutPlan.  Zero-copy submits all peers (not waited), runs the
  // deferred local puts in that window, then waits all + commits; staging runs
  // per peer serially.  Writes per-key outcomes into *results.
  void ExecuteBatchPutPlan(const BatchPutPlan& plan, std::vector<PutEntryOutcome>* results);

  // Group a BatchGet's routes into a BatchGetPlan (no IO issued).  Mirrors
  // PartitionBatchPutTargets, but deliberately does NOT execute local reads:
  // ExecuteBatchGetPlan decides where the local reads run so they can overlap
  // the remote-DRAM RDMA in-flight window.
  BatchGetPlan PartitionBatchGetTargets(const std::vector<std::string>& keys,
                                        const std::vector<void*>& dsts,
                                        const std::vector<size_t>& sizes,
                                        const std::vector<std::optional<RouteGetResult>>& routes);
  // Execute a BatchGetPlan: local reads and the remote-DRAM submit/wait
  // arrangement.  Zero-copy remote DRAM submits all peers, runs local reads
  // INSIDE that submit..wait gap (so they overlap the DRAM wire), then waits
  // all peers.  Staging (non-zero-copy) runs per peer serially (submit ->
  // wait).  Reads the plan; writes per-key outcomes into *results.
  void ExecuteBatchGetPlan(const BatchGetPlan& plan, const std::vector<std::string>& keys,
                           const std::vector<void*>& dsts, const std::vector<size_t>& sizes,
                           std::vector<bool>* results);

  struct RemotePutEntry {
    size_t result_index;
    const BatchPutItem* item;
    SlotPlan plan;
    uint64_t slot_id;
    bool failed = false;
  };

  struct RemoteGetEntry {
    size_t result_index;
    const BatchGetItem* item;
    SlotPlan plan;
    bool failed = false;
  };

  bool AllocateRemotePutEntries(const std::vector<BatchPutItem>& items,
                                ::umbp::UMBPPeer::Stub* stub, std::vector<RemotePutEntry>* entries,
                                std::vector<uint64_t>* abort_slots,
                                std::vector<PutEntryOutcome>* results);
  // Build one TransferItem per page, tagged with the entry index.  Whether an
  // item ends up zero-copy or staged, and how items group into wire transfers,
  // are both the engine's business now — this only names endpoints.
  bool BuildRemotePutTransfers(std::vector<RemotePutEntry>& entries, const std::string& node_id,
                               std::vector<TransferItem>* items);
  void FinalizeRemotePutEntries(std::vector<RemotePutEntry>& entries,
                                std::vector<uint64_t>& abort_slots,
                                std::vector<PutEntryOutcome>* results,
                                ::umbp::UMBPPeer::Stub* stub);

  bool PrepareRemoteGetEntries(const std::vector<BatchGetItem>& items, PeerConnection& peer,
                               ::umbp::UMBPPeer::Stub* stub, std::vector<RemoteGetEntry>* entries,
                               std::vector<bool>* results);
  bool BuildRemoteGetTransfers(std::vector<RemoteGetEntry>& entries, const std::string& node_id,
                               std::vector<TransferItem>* items);
  void FinalizeRemoteGetEntries(std::vector<RemoteGetEntry>& entries, std::vector<bool>* results);

  // One posted-but-not-yet-waited remote read for a single peer; the scheduler
  // waits it later.  The lifetime contract that used to live here — statuses
  // sized once and never moved, drained by the destructor — moved into the
  // engine's TransferHandle, which is where the raw TransferStatus* the RDMA
  // backend holds actually live.
  struct RemoteGetInFlight {
    PeerConnection* peer = nullptr;
    std::vector<RemoteGetEntry> entries;
    std::unique_ptr<TransferHandle> handle;
    bool drained = false;
  };

  // Submit half: GetOrConnectPeer + EnsurePeerServiceConnection +
  // PrepareRemoteGetEntries + BuildRemoteGetTransfers + engine Plan/Submit (NOT
  // waited).  Returns the in-flight handle, or nullptr if nothing is in flight
  // (peer unreachable / resolve / build failure — failed keys already written
  // to *results).
  //
  // There is no longer a permit_staging flag: a batch whose dst is unregistered
  // is staged INSIDE the engine's Submit and comes back already settled, so a
  // submit-all loop over several peers cannot deadlock on the bounce pool and a
  // batch that mixes registered and unregistered buffers is no longer a
  // contract violation — it just works.
  std::unique_ptr<RemoteGetInFlight> SubmitRemoteBatchGet(const std::vector<BatchGetItem>& items,
                                                          std::vector<bool>* results);
  // Wait half: wait the handle (never breaks early), map per-plan failure back
  // to per-key (per-item AND), then FinalizeRemoteGetEntries.
  void WaitRemoteBatchGet(RemoteGetInFlight& inflight, std::vector<bool>* results);

  // Put's counterpart.  Also carries the slot lifecycle: `entries` /
  // `abort_slots` feed FinalizeRemotePutEntries and `stub` issues its
  // commit/abort RPCs.
  struct RemotePutInFlight {
    PeerConnection* peer = nullptr;
    ::umbp::UMBPPeer::Stub* stub = nullptr;
    std::vector<RemotePutEntry> entries;
    // Malformed slots from Allocate (not in `entries`); Finalize appends
    // entry.failed slots and aborts the union (peer Abort is idempotent).
    std::vector<uint64_t> abort_slots;
    std::unique_ptr<TransferHandle> handle;
    bool drained = false;
  };

  // Submit half: allocate + build + engine Plan/Submit (NOT waited).  Returns
  // the in-flight, or nullptr if nothing is posted — in which case any
  // allocated slots are aborted and the keys written kFailed here.  Entries
  // that fail during build but still post ride in the in-flight and are aborted
  // by FinalizeRemotePutEntries at wait time (no early abort, avoids
  // double-abort).
  std::unique_ptr<RemotePutInFlight> SubmitRemoteBatchPut(const std::vector<BatchPutItem>& items,
                                                          std::vector<PutEntryOutcome>* results);
  // Wait half: wait the handle, map per-plan failure back to per-key, then
  // FinalizeRemotePutEntries (commit survivors, abort failures + malformed
  // slots).
  void WaitRemoteBatchPut(RemotePutInFlight& inflight, std::vector<PutEntryOutcome>* results);
};

}  // namespace mori::umbp
