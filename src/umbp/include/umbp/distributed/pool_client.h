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
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mori/io/engine.hpp"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/metrics/component_metrics.h"
#include "umbp/distributed/peer/backend/medium_backend.h"
#include "umbp/distributed/transfer/transfer_engine.h"
#include "umbp/distributed/types.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {

class PeerServiceServer;
class HbmCopyEngine;

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

  // Record a caller-owned region.  The descriptor is cached and looked up by
  // (ptr, size) on the Put/Get hot paths; `loc`/`device` describe the caller's
  // allocation (CPU, or a GPU ordinal) so the transfer layer routes to
  // HbmCopyEngine instead of assuming host memory.
  //
  // kPinned also hands the region to the IO engine for zero-copy RDMA.
  // kLocalCopyOnly skips that, for a region that is only ever a local copy
  // endpoint -- it still has to be recorded, because a range that misses this
  // table is described by its own address rather than its region base and so
  // becomes its own transfer plan.
  bool RegisterMemory(void* ptr, size_t size,
                      mori::io::MemoryLocationType loc = mori::io::MemoryLocationType::CPU,
                      int device = -1, MemoryRegistration mode = MemoryRegistration::kPinned);

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

  // Ranged multi-buffer I/O.  One stored object is backed by scattered tier
  // pages while the caller supplies several disjoint object-relative ranges,
  // each with its own buffer.  This is what a KV connector wants: read layer k
  // of every block without materializing the blocks.
  //
  // Both directions map a range onto pages the same way the whole-object path
  // does, because a TransferItem already carries src_offset/dst_offset/size —
  // it *is* a range.  Whole-object I/O is the degenerate case with the range
  // pinned to [0, size); nothing in the backend, the engine or the wire format
  // changes to support this.
  //
  // Get: ranges must be disjoint and inside the stored object.  Keys held by
  // this node are served straight out of its medium; the rest are fetched whole
  // into the registered scratch arena, installed locally, and served from
  // there.
  //
  // Put: ranges must *tile* the object — disjoint and covering [0, object_size)
  // exactly — since a partial write would commit a slot with undefined gaps.
  // A key routed to this node is written range-by-range directly into its
  // pages, with no assembly step; a key routed elsewhere is assembled in the
  // scratch arena and sent as one object.
  // One caller range against one stored object.  `user` is the caller's buffer
  // for this range (the whole of it — the range's own base), `object_offset`
  // where the range starts inside the stored object.
  struct ObjectRange {
    void* user = nullptr;
    size_t size = 0;
    size_t object_offset = 0;
  };

  // Where a ranged call's internal helpers report the time they spent, so the
  // phases can be attributed without the helpers knowing about the reporting
  // machinery.  Split this finely because the costs scale differently: `resolve`
  // is per key, `classify` is per RANGE, and one call carries thousands of
  // ranges per key batch.  Every member is nullable; null is inert.
  struct RangedPhaseSinks {
    double* resolve = nullptr;   // medium index lookup / slot allocation
    double* classify = nullptr;  // classifying the caller's range pointers
    double* build = nullptr;     // TransferItem assembly (classify excluded)
    double* commit = nullptr;    // slot commit / abort
    size_t* items = nullptr;     // TransferItems handed to the engine
    // Forwarded straight to TransferEngine::Transfer so the engine's own three
    // steps land in the same report as the phases around them.
    const TransferEngine::StepTiming* steps = nullptr;
  };

  std::vector<bool> BatchGetRanges(const std::vector<std::string>& keys,
                                   const std::vector<std::vector<void*>>& dsts,
                                   const std::vector<std::vector<size_t>>& sizes,
                                   const std::vector<std::vector<size_t>>& src_offsets);
  std::vector<bool> BatchPutRanges(const std::vector<std::string>& keys,
                                   const std::vector<size_t>& object_sizes,
                                   const std::vector<std::vector<const void*>>& srcs,
                                   const std::vector<std::vector<size_t>>& sizes,
                                   const std::vector<std::vector<size_t>>& dst_offsets);

  // Cluster-wide existence check — issues a RouteGet and reports
  // whether master surfaced any replica.  No RDMA, no lease bump.
  bool Exists(const std::string& key);
  std::vector<bool> BatchExists(const std::vector<std::string>& keys);

  MasterClient& Master();

  // The storage medium live on this node (exactly one — see PoolClient::Init).
  // Callers reach it by tier (Backends().Get(Medium())) and use it through
  // MediumBackend — no concrete backend type is named outside PoolClient::Init.
  BackendRegistry& Backends();

  // Which medium this node serves.  Valid after Init.
  TierType Medium() const { return medium_; }

  // Whether this node is part of a master-coordinated cluster.  False is the
  // single-node deployment: the data plane is unchanged, but nothing is routed,
  // registered, or heartbeated, and a local miss is the final answer.
  bool HasMaster() const { return master_client_ != nullptr; }

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

  // Sample every instrumented component on this node — each registered storage
  // backend, plus the transfer engine — and ship the result.  Registered as a
  // MasterClient metrics provider, so it runs on the metrics thread and never
  // on a data-plane path.
  //
  // There is no per-component code here and no place to add any: a component is
  // a MetricSource, the labels that identify it come from its own Tier()/Name(),
  // and MetricPublisher does the differencing.  That is what makes a new backend
  // or a new transfer engine visible in Grafana without touching this file.
  void PublishComponentMetrics();

  // Holds the delta baselines for every component published above, keyed by
  // (component, metric, labels).  Touched once per tick; the counters
  // themselves live in the components.
  MetricPublisher metric_publisher_;

  std::unique_ptr<MasterClient> master_client_;

  // Every storage medium live on this node.  Owned here because PoolClient is
  // the natural lifetime anchor for the per-process IO engine + backend pools.
  // PeerServiceServer and MasterClient both borrow the registry and dispatch
  // through it (backend-agnostic refactor Phase 3).
  BackendRegistry registry_;

  // The one tier registry_ holds, cached from the backend at Init.  Read by the
  // few paths that must name a LOCAL destination with no route to dispatch on
  // (the re-cache installer); everything with a route uses route.tier.  Kept in
  // sync with config_.medium by construction — Init sets it from the backend it
  // actually built, not from config.
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
  // Observer into transfer_engine_'s composite, used only to declare which
  // host regions the gather kernel may dereference.  Not a second data path.
  HbmCopyEngine* hbm_engine_ = nullptr;

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

  // Zero-copy registered memory regions, kept sorted by `base`.
  //
  // Sorted because a ranged call looks one up PER RANGE, and a layer-wise
  // reader carries thousands of ranges against a caller that registered one
  // buffer per layer -- a linear scan makes that quadratic.  Shared, because
  // those lookups are reads and every rank on the node shares this client.
  struct RegisteredRegion {
    void* base;
    size_t size;
    TransferRef ref;
  };
  mutable std::shared_mutex registered_mem_mutex_;
  std::vector<RegisteredRegion> registered_regions_;

  // The region covering [ptr, ptr+size), or null.  Caller holds
  // registered_mem_mutex_; the returned pointer is valid only under that lock.
  // Exposed separately from FindRegisteredMemory so a batch can take the lock
  // once and then resolve every range under it.
  const RegisteredRegion* FindRegisteredRegionLocked(const void* ptr, size_t size) const;

  // The registered ref covering [ptr, ptr+size) plus the offset of ptr within
  // it, or nullopt when the region was never registered.
  std::optional<std::pair<TransferRef, size_t>> FindRegisteredMemory(const void* ptr,
                                                                     size_t size) const;

  // Single-attempt outcome from a peer call; mapped to PutEntryOutcome
  // by the caller (Partition / Allocate).
  enum class PutAttemptOutcome { kSuccess, kSuccessAlreadyExists, kRetry, kFatal };
  enum class GetAttemptOutcome { kSuccess, kRetry, kFatal };

  PutAttemptOutcome ExecuteLocalPut(const std::string& key, const void* src, size_t size,
                                    TierType tier);

  // Resolve `keys` against this node's own media.  On return holders[i] is the
  // backend that owns keys[i] (nullptr when nothing here does) and, when it is
  // non-null, resolutions[i] is that backend's entry.  Both are sized to
  // keys.size(); `candidates` selects which indices are asked about, so a
  // caller can skip entries it has already rejected.
  //
  // `resolutions` may be null for a caller that only needs to know WHETHER a
  // key is here -- an Exists does, and materializing an entry per key costs it
  // a pages vector and a descs vector it never reads (measured 6-15%).
  //
  // One BatchResolve per BACKEND rather than one per key: the backend mutex is
  // shared with Allocate / Commit / Evict, and in standalone-process mode every
  // rank on the node shares this client, so a per-key walk is where they
  // serialize.  Backends are tried in registry order and the first hit wins, so
  // a second medium is asked only about what the first one missed.
  //
  // A hit here is conclusive: the backend owns the bytes.  A miss is conclusive
  // only about THIS node; the key may still live on a peer, which is why every
  // caller sends its misses to the master.
  void ResolveLocalBatch(const std::vector<std::string>& keys,
                         const std::vector<size_t>& candidates,
                         std::vector<MediumBackend*>* holders,
                         std::vector<ResolvedEntry>* resolutions);

  // One TransferItem per page between a caller buffer and `backend`'s own
  // buffers.  `to_backend` is Put (user -> pages), false is Get (pages -> user).
  // False when the backend publishes no in-process endpoint for a referenced
  // buffer — that medium cannot serve the access here, and the caller routes
  // elsewhere rather than reporting a miss.
  bool BuildLocalPageTransfers(MediumBackend* backend, const std::vector<PageLocation>& pages,
                               uint64_t page_size, void* user, size_t size, bool to_backend,
                               std::vector<TransferItem>* items);

  // ---- ranged I/O internals ----

  // The route-first arm of BatchGetRanges: one BatchRouteGet over every key,
  // before anything is served.  `route_sink` is the optional phase timer.
  // False means the routing RPC failed.
  bool RouteAllRangesUpFront(const std::vector<std::string>& keys, double* route_sink,
                             std::vector<std::optional<RouteGetResult>>* preroutes);

  // Routes for the keys BatchGetRanges' local phase missed, parallel to
  // `missed`.  Issues the RPC under local_first; under the route-first arm it
  // projects phase 0's `preroutes` instead, so that arm costs one routing RPC
  // rather than two.  False means the routing RPC failed.
  bool RouteMissedRanges(const std::vector<std::string>& route_keys,
                         const std::vector<size_t>& missed,
                         const std::vector<std::optional<RouteGetResult>>& preroutes,
                         double* route_sink, std::vector<std::optional<RouteGetResult>>* routes);

  // The ranged counterpart of BuildLocalPageTransfers, and the only place the
  // object-range -> page-range mapping lives.  Emits one TransferItem per
  // (range x page) intersection; the engines coalesce adjacent ones back into
  // single copies while planning, so a range spanning N contiguous pages costs
  // one copy, not N.  Every item carries `tag` = the caller's key index, so a
  // partial failure comes back per key rather than failing the batch.
  //
  // Appends to `items`; returns false if the medium publishes no endpoint for a
  // referenced buffer, or a range falls outside the stored object.
  //
  // `classify_sink`, when non-null, accumulates the seconds spent classifying
  // the caller's range pointers.  That cost scales with ranges rather than
  // keys, so it is reported as its own phase; null makes it inert.
  bool BuildLocalRangeTransfers(MediumBackend* backend, const std::vector<PageLocation>& pages,
                                uint64_t page_size, uint64_t stored_size,
                                const std::vector<ObjectRange>& ranges, bool to_backend, size_t tag,
                                std::vector<TransferItem>* items, double* classify_sink = nullptr);

  // Copy between one contiguous host object and a set of caller ranges, through
  // the transfer engine so a device-resident caller buffer is handled by
  // HbmCopyEngine rather than memcpy'd.  Used for the scratch arena, which is
  // not a medium and so has no PageLocations to map.
  bool CopyContiguousToRanges(const void* src, size_t object_size,
                              const std::vector<ObjectRange>& ranges);
  // Same, but for a source the backend already described.  A medium slot is
  // NOT necessarily host memory -- an HBM pool's BufferRef carries a device
  // pointer -- so it has to keep the ref's loc/device rather than be re-wrapped
  // as host bytes, or the engine picks the wrong copy direction.
  bool CopyContiguousToRanges(const TransferRef& src, uint64_t src_base, size_t object_size,
                              const std::vector<ObjectRange>& ranges);
  bool CopyRangesToContiguous(const std::vector<ObjectRange>& ranges, void* dst,
                              size_t object_size);

  // One locally-routed ranged put.  `result_index` is the caller's key index,
  // which is what gets written back into the results vector.
  struct LocalRangeWriteRequest {
    size_t result_index = 0;
    const std::string* key = nullptr;
    size_t object_size = 0;
    std::vector<ObjectRange> ranges;
    TierType tier = TierType::UNKNOWN;
  };

  // Allocate a slot per request, write the ranges straight into its pages, and
  // commit.  No assembly buffer: the ranges are written where they belong.
  //
  // Batched across keys deliberately.  Every request's items go into ONE
  // transfer, so a batch of GPU-sourced puts becomes a single gather kernel
  // instead of one per key — the difference the kernel exists to make.  Slots
  // are held allocated across the transfer and committed or aborted per key
  // afterwards, keyed off the engine's failed tags.
  //
  // Ranges must already have been validated as tiling their object.
  //
  // `committed_bytes`, when non-null, accumulates the bytes this call actually
  // wrote.  Deliberately not derivable from `results`: a key already present in
  // the medium reports success without moving anything, and crediting it would
  // inflate the bandwidth histogram.
  //
  // `sinks`, when non-null, attributes the call's time to phases; see
  // RangedPhaseSinks.  Its members are independently nullable too, and a null
  // one costs nothing -- that is what keeps this off the hot path when the
  // ranged debug switch is off.
  void ExecuteLocalPutRangesBatch(const std::vector<LocalRangeWriteRequest>& requests,
                                  std::vector<bool>* results, double* committed_bytes = nullptr,
                                  const RangedPhaseSinks* sinks = nullptr);
  // After a successful remote DRAM fetch, if cache_remote_fetches is enabled and
  // the admission gate admits the block, enqueue it for asynchronous install into
  // this node's local DRAM tier (see ReCacheWorkerLoop). The install (DRAM
  // Allocate + copy + Commit→KvEvent::ADD publish) happens OFF the Get critical
  // path so it does not add latency to concurrent Gets. Idempotent
  // (kSuccessAlreadyExists) and best-effort: the queue is bounded and drops on
  // full; failure does not affect the returned Get result.
  void MaybeReCacheAfterRemote(const std::string& key, const void* src, size_t size);

  // After a remote RANGED read served `fetched_bytes` of `object_size`, queue a
  // background pull of the WHOLE object into this node's medium, so the next
  // read of the key is a local hit instead of another trip to the peer.
  //
  // Why this exists at all: a ranged read holds only the caller's spans, so
  // MaybeReCacheAfterRemote — which installs the buffer it is handed — has
  // nothing legal to install.  Locality has to be rebuilt from the peer.
  //
  // Skipped when this call already covered the object (`fetched_bytes ==
  // object_size`): the whole-object reader would otherwise pull every object
  // twice, once for itself and once for a cache it just filled.
  //
  // Best-effort in every direction: gated by ranged_locality_prefetch and the
  // re-cache admission policy, deduplicated against both the queue and this
  // node's medium, bounded, and silent on failure.  It never runs on, and never
  // blocks, the read that scheduled it.
  void MaybePrefetchWholeObject(const std::string& key, size_t object_size,
                                const RouteGetResult& route);
  // The other half of the same decision: the arena slice already holds the
  // whole object (the caller's ranges tiled it in order), so locality costs one
  // local copy instead of a second trip to the peer.  Synchronous because the
  // slice is live only until the next sub-batch reuses it.  Mutually exclusive
  // with MaybePrefetchWholeObject — exactly one runs per fetched key.
  void MaybeInstallCompleteArenaObject(const std::string& key, const void* arena_slice,
                                       size_t object_size);
  // The two locality paths are gated separately, because they do not cost the
  // same thing.
  //
  // Landing the object in this node's medium when it is ALREADY IN HAND -- the
  // arena slice is the whole object, or the peer can write straight into a slot
  // -- costs nothing on the wire. The pre-ranged code did it on every remote
  // ranged read, unconditionally, so gating it would be a silent regression
  // against a switch whose documented job is something else. The only question
  // worth asking is whether there is a medium to install into.
  bool CanInstallLocally() const;
  // Fetching the object a SECOND time in the background costs up to one extra
  // copy of it on the wire. That is what ranged_locality_prefetch switches off,
  // and what the re-cache admission policy sizes.
  bool LocalityPrefetchAdmits(size_t object_size) const;
  // Background worker that drains recache_queue_ and performs the DRAM install
  // via ExecuteLocalPut. Started in Init (when cache_remote_fetches), stopped in
  // Shutdown before peer_alloc_ is torn down.
  void ReCacheWorkerLoop();

  // Async re-cache install queue. MaybeReCacheAfterRemote copies the block bytes
  // into a job here (the source buffer is not owned past the Get return), and the
  // worker thread installs it. Bounded to keep memory + publish churn in check.
  // Two shapes share one queue and one worker so they also share its lifecycle,
  // its bound and its drain-on-shutdown:
  //   bytes != nullptr  -> install this buffer (MaybeReCacheAfterRemote)
  //   bytes == nullptr  -> pull the object from `route` first
  //                        (MaybePrefetchWholeObject)
  struct ReCacheJob {
    std::string key;
    std::unique_ptr<char[]> bytes;
    size_t size = 0;
    std::optional<RouteGetResult> route;
  };
  // Pull whole objects from their peers straight into freshly allocated local
  // slots and commit them.  Peer pages and medium pages are both registered
  // host memory, so this is RDMA with no bounce buffer and no memcpy — unlike
  // the ReCacheJob path, which has to carry a heap copy of the bytes.
  //
  // BATCHED deliberately, and it has to be.  Per key the fixed cost is a peer
  // resolve RPC plus an RDMA submit/wait; done one at a time on this single
  // worker thread that cost is paid once per key, and for small objects it
  // dominates the bytes so thoroughly that the worker cannot keep up with the
  // reader — the prefetch then loses more races than it wins and the whole
  // feature turns negative.  Batching amortises the resolve over every key
  // routed to the same peer and gives the engine one large scatter-gather to
  // post.
  //
  // Best-effort per key: already-local keys are dropped before allocating (the
  // alternative wastes a whole object of wire), and any key whose allocation or
  // transfer fails is aborted without affecting the others.
  void FetchWholeObjectsIntoMedium(std::vector<ReCacheJob>& jobs);

  std::deque<ReCacheJob> recache_queue_;
  std::mutex recache_mutex_;
  // Keys with a whole-object pull already queued or running.  A layer-wise
  // reader names the same key once per layer group, so without this every group
  // that misses locally would queue another pull of bytes already in flight.
  std::unordered_set<std::string> prefetch_inflight_;

  // Serializes users of each caller-owned ranged scratch arena.  Only the remote
  // half of a ranged operation takes one — keys served by this node's own medium
  // never touch an arena and stay fully concurrent.
  //
  // Separate GET and PUT arenas each get their own mutex, so a remote ranged GET
  // and a remote ranged PUT run concurrently instead of serializing on one lock
  // — the load/offload overlap sglang's direct linker wants.  (Two same-kind ops
  // still serialize on their arena's mutex.)
  std::mutex ranged_get_scratch_mutex_;
  std::mutex ranged_put_scratch_mutex_;
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
  // One byte range of a stored object, named by the caller.
  struct ByteSpan {
    size_t object_offset = 0;
    size_t size = 0;
  };

  struct BatchGetItem {
    size_t index;
    const std::string* key;
    void* dst;
    size_t size;  // the OBJECT's stored size — what the peer must report
    // Ranged reads move only `spans` and land them packed at `dst`, so the
    // bytes written are the span total rather than the object size.  Both
    // fields default to the whole-object behaviour, which is what plain
    // BatchGet wants and why it needs no changes.
    size_t dst_bytes = 0;                          // 0 => size
    const std::vector<ByteSpan>* spans = nullptr;  // null/empty => whole object
    // Pre-resolved destination, used instead of `dst` when set.  UserBufferRef
    // only knows regions registered through PoolClient::RegisterMemory, so a
    // destination inside a medium backend's own pool — which is RDMA-reachable
    // but was never handed to RegisterMemory — has to be named by the ref the
    // backend publishes.  Locality prefetch writes straight into a medium slot
    // and is the only user.
    const TransferRef* dst_ref = nullptr;
    uint64_t dst_ref_offset = 0;
    RouteGetResult route;

    size_t DstBytes() const { return dst_bytes != 0 ? dst_bytes : size; }
    bool Ranged() const { return spans != nullptr && !spans->empty(); }
  };

  // Routing plan for one BatchGet: which keys go to which target.  Pure
  // grouping — no IO is issued here.  Remote reads go through the batched RDMA
  // path (remote_groups, keyed by peer node_id); self-target reads are
  // deferred (collected as indices) so ExecuteBatchGetPlan can place them
  // inside the remote-DRAM in-flight window when overlapping.
  // What a key that no PEER can serve should fall back to.  Route-first has not
  // looked locally yet, so an unroutable or self-routed key still deserves a
  // local read.  Local-first has already asked every medium on this node, so the
  // same key must be dropped -- sending it back would re-resolve a key just
  // proven absent, which is the per-key cost ServeLocalBatchGet exists to avoid.
  enum class LocalFallback { kAllow, kSkip };

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
                                        const std::vector<std::optional<RouteGetResult>>& routes,
                                        LocalFallback fallback);

  // Route `indices` of `keys` and scatter the answers into *routes at their
  // ORIGINAL key index, leaving every other slot nullopt -- which is how
  // ComputeBatchBandwidthBytes already reads a missing route, and what lets one
  // partition pass serve both arms.  `exclude_self` is set by the caller that
  // has already tried this node's own media.  False means the RPC failed.
  bool RouteGetsInto(const std::vector<std::string>& keys, const std::vector<size_t>& indices,
                     bool exclude_self, std::vector<std::optional<RouteGetResult>>* routes);

  // ---- local-first get (PoolClientConfig::local_first) ----

  // Serve `indices` from this node's own media in one resolve and one transfer.
  // Hits go into *results; unservable keys are appended to *missed, which may be
  // null when the caller has no fallback left.  Shared by both BatchGet arms.
  void ServeLocalGets(const std::vector<std::string>& keys, const std::vector<void*>& dsts,
                      const std::vector<size_t>& sizes, const std::vector<size_t>& indices,
                      std::vector<bool>* results, std::vector<size_t>* missed);

  // Whole-batch wrapper for the local-first arm: drops what cannot be served at
  // all, serves the rest, and returns the keys the master still has to route.
  std::vector<size_t> ServeLocalBatchGet(const std::vector<std::string>& keys,
                                         const std::vector<void*>& dsts,
                                         const std::vector<size_t>& sizes,
                                         std::vector<bool>* results);

  // Counter sink that tolerates a node running without a master: metrics
  // accumulate on the MasterClient and ride its flush tick, so with no master
  // there is simply nowhere to put them.
  void CountMetric(std::string name, std::string help, MasterClient::Labels labels, double delta);

  // Placement when there is no master to ask: this node is the only candidate,
  // so every key is routed to it on the medium it serves.  The put paths then
  // proceed exactly as they would for a master-issued self-route.
  void RouteAllPutsLocally(size_t count, std::vector<std::optional<RoutePutResult>>* routes) const;

  // Local/remote bandwidth histograms for one BatchGet.  A method rather than
  // an inline tail because local-first gives the call several exit points.
  void ObserveBatchGetBandwidth(const std::vector<bool>& results, const std::vector<size_t>& sizes,
                                const std::vector<std::optional<RouteGetResult>>& routes,
                                std::chrono::steady_clock::time_point call_start);
  // Execute a BatchGetPlan: local reads and the remote-DRAM submit/wait
  // arrangement.  Zero-copy remote DRAM submits all peers, runs local reads
  // INSIDE that submit..wait gap (so they overlap the DRAM wire), then waits
  // all peers.  Staging (non-zero-copy) runs per peer serially (submit ->
  // wait).  Reads the plan; writes per-key outcomes into *results.
  // `recache_remote=false` suppresses the asynchronous re-cache of remotely
  // fetched blocks.
  void ExecuteBatchGetPlan(const BatchGetPlan& plan, const std::vector<std::string>& keys,
                           const std::vector<void*>& dsts, const std::vector<size_t>& sizes,
                           std::vector<bool>* results, bool recache_remote = true);

  // The remote half of ExecuteBatchGetPlan: submit every peer, then wait every
  // peer.  Split out for the ranged path, whose plan has no local half by
  // construction (BatchGetRanges filters unroutable and self-routed keys out
  // before building it), so threading the keys/dsts/sizes the local half needs
  // would only pass three vectors nothing reads.
  // `in_flight_window`, when set, runs after every peer has been posted and
  // before any of them is waited on -- which is where local reads belong, so
  // they overlap the wire instead of following it.  A ranged plan has no local
  // half by construction and passes nothing.
  void ExecuteRemoteBatchGetPlan(const BatchGetPlan& plan, std::vector<bool>* results,
                                 bool recache_remote,
                                 const std::function<void()>& in_flight_window = {});
  // One place to say "this key was fetched but did not become local".  Every
  // way that can happen -- no slot, a slot that cannot be addressed as one run,
  // a refused commit, a background pull that never landed -- reports here, so
  // the counter measures the feature rather than one branch of it.
  void NoteRangedInstallFailure(size_t count = 1);

  // One key's share of one arena round: the spans to fetch, where they land in
  // the slice, and the caller buffers they are copied out to.  A key needs more
  // than one when its spans do not all fit at once.
  struct RangeFetchUnit {
    size_t original = 0;  // caller key index
    size_t object_size = 0;
    std::optional<RouteGetResult> route;
    std::vector<ByteSpan> spans;
    std::vector<ObjectRange> packed;  // caller pointer + slice-relative offset
    size_t bytes = 0;                 // == sum of spans
    // The slice is byte-for-byte the whole object, which happens when the
    // caller's ranges tile it in ascending order and it all fits in one unit.
    // The whole-object reader (one call for every layer) hits this; the
    // layer-wise reader never does.
    bool holds_whole_object = false;
  };

  // Serve whole-object ranged reads out of a fresh medium slot instead of the
  // arena.  Their span layout already equals the object, so the peer can write
  // straight into the slot: one copy instead of two, the arena stays free for
  // the readers that actually need it, and the arena mutex is never taken.
  // Committing the slot afterwards is the local install, for nothing.
  //
  // Returns the units it could not take -- no slot to be had, or a slot that is
  // not one contiguous run -- for the arena path to serve the ordinary way.
  // That fallback is decided here, on a cheap local allocation, and never after
  // a failed transfer: a transfer does not fail because of where it was
  // pointed, and retrying it elsewhere would only double the latency of a
  // failure the caller treats as fatal.
  // `remote_bytes` accumulates what this path pulled over the wire.  It has to
  // be reported here rather than by the arena loop, because a batch whose units
  // are all slot-served never reaches that loop -- and that is exactly the
  // shape this path exists to make fast, so leaving it out would blank the
  // bandwidth metric precisely where it matters.
  std::vector<RangeFetchUnit> ServeWholeObjectUnitsFromMedium(
      const std::vector<std::string>& keys, std::vector<RangeFetchUnit> units,
      std::unordered_map<size_t, bool>* unit_ok, double* remote_bytes);

  // Remote-only sibling of PartitionBatchGetTargets for ranged reads.  Each key
  // contributes one item carrying its arena slice, its span list, and the span
  // total.  Spans are passed by pointer rather than by value: the caller
  // already owns them for the whole call, and copying one small vector per key
  // per sub-batch is a heap allocation per key that buys nothing.  Both the
  // pointees and `keys` must outlive the returned plan.
  BatchGetPlan PartitionBatchGetRangeTargets(
      const std::vector<std::string>& keys, const std::vector<void*>& arena_slices,
      const std::vector<size_t>& object_sizes, const std::vector<size_t>& packed_bytes,
      const std::vector<const std::vector<ByteSpan>*>& spans,
      const std::vector<std::optional<RouteGetResult>>& routes);

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
  void FinalizeRemoteGetEntries(std::vector<RemoteGetEntry>& entries, std::vector<bool>* results,
                                bool recache_remote);

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
  void WaitRemoteBatchGet(RemoteGetInFlight& inflight, std::vector<bool>* results,
                          bool recache_remote);

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
