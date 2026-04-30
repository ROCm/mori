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
#include <utility>
#include <vector>

#include "mori/io/engine.hpp"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/obs_counters.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

class PoolClient {
 public:
  explicit PoolClient(PoolClientConfig config);
  // Destructor is virtual under MORI_UMBP_TESTING to match the conditional
  // virtuality of IssueBatchWrite (avoids -Wnon-virtual-dtor and supports
  // safe deletion through a base pointer in test subclasses).  In release
  // builds the destructor is non-virtual; PoolClient is not designed to be
  // a public base class outside testing.
  MORI_UMBP_TEST_VIRTUAL ~PoolClient();

  PoolClient(const PoolClient&) = delete;
  PoolClient& operator=(const PoolClient&) = delete;

  bool Init();
  void Shutdown();

  const std::string& NodeId() const { return config_.master_config.node_id; }

  bool RegisterMemory(void* ptr, size_t size);
  void DeregisterMemory(void* ptr);

  // DRAM-only methods for UMBPClient integration.  UMBPClient handles local
  // storage directly and calls these for cluster interactions only;
  // PoolClient never touches local storage.

  // Register an already-written local block with the Master so remote nodes
  // can discover it. UMBPClient provides the location_id (e.g. "0:<offset>").
  bool RegisterWithMaster(const std::string& key, size_t size, const std::string& location_id,
                          TierType tier);
  bool FinalizeAllocation(const std::string& key, size_t size, const std::string& location_id,
                          TierType tier, const std::string& allocation_id);
  bool PublishLocalBlock(const std::string& key, size_t size, const std::string& location_id,
                         TierType tier);
  // Roll back a pending allocation on the Master.  Only call this AFTER
  // Master has handed out a successful allocation (RoutePut / AllocateForPut)
  // AND the local-side commit work (memcpy / RDMA / FinalizeAllocation)
  // failed.  Do NOT call to defend against malformed Master responses or
  // Master-side validation rejections -- those are handled by Master
  // internally (AllocateForPut invariant guard, FinalizeAllocation
  // auto-rollback) and reaching for Abort on top of them is redundant.
  bool AbortAllocation(const std::string& node_id, TierType tier, const std::string& allocation_id,
                       uint64_t size);

  // Check whether a block exists in the cluster (local or remote, no RDMA).
  bool Exists(const std::string& key);

  // Batched existence check — single BatchLookup gRPC for the whole batch.
  // Same semantics as Exists (no access-count / lease side-effects).
  // On wire error all entries are false.  Returns a vector parallel to
  // `keys`.
  std::vector<bool> BatchExists(const std::vector<std::string>& keys);

  bool IsRegistered(const std::string& key) const;

  // Fetch a block from a remote node via RDMA.
  // DRAM: RouteGet -> direct RDMA read.
  // SSD: RouteGet -> PeerService PrepareSsdRead (SSD->staging slot) -> RDMA read.
  //
  // `size` MUST equal the byte size committed at Put time (Master tracks
  // it in Location.size and returns it via RouteGet).  Mismatch is a
  // contract violation — the call fails fast with a clear error rather
  // than silently truncating the last page or pulling stale tail bytes.
  //
  // The caller is responsible for tracking the original Put size (e.g.
  // alongside the key in its own metadata).  `Lookup` deliberately does
  // not return size to keep that RPC cheap and side-effect-free; querying
  // size via `RouteGet` would create a lease as a side effect, which is
  // exactly what `Lookup` was introduced to avoid (see issue #9).
  bool Get(const std::string& key, void* dst, size_t size);

  // Write a block to a remote node via RDMA.
  // DRAM: RoutePut -> direct RDMA write.
  // SSD: RoutePut -> AllocateWriteSlot -> RDMA write -> CommitSsdWrite.
  bool Put(const std::string& key, const void* src, size_t size);

  // Batch write: single BatchRoutePut, fused per-peer BatchWrite (3-phase
  // pipeline overlapping NIC and CPU work), single BatchFinalizeAllocation,
  // and a single BatchAbortAllocation flush for failures.
  std::vector<bool> BatchPut(const std::vector<std::string>& keys,
                             const std::vector<const void*>& srcs, const std::vector<size_t>& sizes,
                             const std::vector<int>& depths = {});

  // Batch read: single gRPC for routing + batched RDMA.
  // Same per-entry size contract as Get: sizes[i] MUST equal the
  // stored Location.size for keys[i].  Per-entry mismatch fails just that
  // entry (results[i]=false); other entries are unaffected.
  std::vector<bool> BatchGet(const std::vector<std::string>& keys, const std::vector<void*>& dsts,
                             const std::vector<size_t>& sizes);

  // Unregister a block from the Master (block no longer remotely accessible).
  bool UnregisterFromMaster(const std::string& key);

  void* SsdStagingPtr() const { return ssd_staging_buffer_.get(); }
  size_t SsdStagingSize() const { return config_.staging_buffer_size; }
  const std::vector<uint8_t>& SsdStagingMemDescBytes() const { return ssd_staging_mem_desc_bytes_; }

  MasterClient& Master();
  bool IsInitialized() const;

  // Observability counters — members always exist (ABI stable) but only
  // increment when built with -DMORI_UMBP_TESTING.  Release builds leave
  // them at 0 and pay zero CPU cost at increment call sites.
  uint64_t AbortAllocationCallsCount() const {
    return abort_allocation_calls_.load(std::memory_order_relaxed);
  }
  uint64_t BatchAbortAllocationCallsCount() const {
    return batch_abort_allocation_calls_.load(std::memory_order_relaxed);
  }
  uint64_t BatchAbortAllocationEntriesCount() const {
    return batch_abort_allocation_entries_.load(std::memory_order_relaxed);
  }
  // BatchPut RDMA fusion observability.  Primary metric:
  //   items_per_pair = BatchPutItemsCount / BatchPutIoEnginePairsCount
  // is the NIC-layer WR-batching proxy (upper bound; backend may split
  // each pair into a few ibv_post_send by postBatchSize).  Secondary:
  // io_engine_calls captures CPU-side IOEngine::BatchWrite invocation count.
  uint64_t BatchPutCallsCount() const { return batch_put_calls_.load(std::memory_order_relaxed); }
  uint64_t BatchPutItemsCount() const { return batch_put_items_.load(std::memory_order_relaxed); }
  uint64_t BatchPutIoEngineCallsCount() const {
    return batch_put_io_engine_calls_.load(std::memory_order_relaxed);
  }
  uint64_t BatchPutIoEnginePairsCount() const {
    return batch_put_io_engine_pairs_.load(std::memory_order_relaxed);
  }
  // BatchGet RDMA fusion observability (mirror of BatchPut counters).
  // Primary metric:
  //   items_per_pair = BatchGetItemsCount / BatchGetIoEnginePairsCount
  uint64_t BatchGetCallsCount() const { return batch_get_calls_.load(std::memory_order_relaxed); }
  uint64_t BatchGetItemsCount() const { return batch_get_items_.load(std::memory_order_relaxed); }
  uint64_t BatchGetIoEngineCallsCount() const {
    return batch_get_io_engine_calls_.load(std::memory_order_relaxed);
  }
  uint64_t BatchGetIoEnginePairsCount() const {
    return batch_get_io_engine_pairs_.load(std::memory_order_relaxed);
  }

  // ---- External KV block events ----
  bool ReportExternalKvBlocks(const std::vector<std::string>& hashes, TierType tier);
  bool RevokeExternalKvBlocks(const std::vector<std::string>& hashes);
  bool MatchExternalKv(const std::vector<std::string>& hashes,
                       std::vector<MasterClient::ExternalKvNodeMatch>* out_matches);

 private:
  PoolClientConfig config_;
  std::atomic<bool> initialized_{false};

  std::unique_ptr<MasterClient> master_client_;

  // IO Engine (data plane)
  std::unique_ptr<mori::io::IOEngine> io_engine_;
  mori::io::MemoryDesc staging_mem_{};
  std::vector<mori::io::MemoryDesc> export_dram_mems_;
  std::unique_ptr<char[]> staging_buffer_;
  std::mutex staging_mutex_;

  // SSD staging buffer — separate from DRAM exportable buffers so that
  // PeerService SSD staging traffic does not conflict with Master-managed
  // DRAM tier offset allocations.
  std::unique_ptr<char[]> ssd_staging_buffer_;
  mori::io::MemoryDesc ssd_staging_mem_{};
  std::vector<uint8_t> ssd_staging_mem_desc_bytes_;

  // Peer connections (lazy init, keyed by node_id)
  struct PeerConnection {
    std::string peer_address;
    mori::io::EngineDesc engine_desc;
    std::vector<mori::io::MemoryDesc> dram_memories;
    bool engine_registered = false;
    std::unique_ptr<void, void (*)(void*)> peer_stub{nullptr, +[](void*) {}};
    std::mutex ssd_op_mutex;

    // Dedicated SSD staging MemoryDesc, independent of dram_memories to avoid
    // offset conflicts between DRAM tier allocations and SSD staging traffic.
    mori::io::MemoryDesc ssd_staging_mem{};
    size_t ssd_staging_size = 0;
  };
  std::mutex peers_mutex_;
  std::unordered_map<std::string, std::unique_ptr<PeerConnection>> peers_;

  PeerConnection& GetOrConnectPeer(const std::string& node_id, const std::string& peer_address,
                                   const std::vector<uint8_t>& engine_desc_bytes);

  // Hydrate `peer.dram_memories[bd.buffer_index]` for every entry in
  // `descs`.  Idempotent: already-cached entries are left alone since
  // MemoryDesc is immutable.  Used by Put/Get paths to absorb the
  // RoutePut/RouteGet response `dram_memory_descs` list before issuing
  // RDMA.  Caller MUST NOT hold peers_mutex_; this helper acquires it.
  void EnsureBufferDescsCached(PeerConnection& peer,
                               const std::vector<BufferMemoryDescBytes>& descs);

  // No-lock variant: caller MUST hold peers_mutex_.  Used by BatchPut
  // fused path to combine hydrate + snapshot under a single critical
  // section.
  void EnsureBufferDescsCachedLocked(PeerConnection& peer,
                                     const std::vector<BufferMemoryDescBytes>& descs);

  bool RemoteDramWrite(PeerConnection& peer, uint32_t buffer_index, const void* src, size_t size,
                       uint64_t offset, bool zero_copy);
  bool RemoteDramRead(PeerConnection& peer, uint32_t buffer_index, void* dst, size_t size,
                      uint64_t offset, bool zero_copy);

  // Multi-page scatter-gather variants used by Put/Get when the master hands
  // back more than one PageLocation.  Pages may span multiple buffer_indices
  // (Strategy 3 of PageBitmapAllocator).  The caller MUST have already
  // populated `peer.dram_memories[buffer_index]` for every buffer_index in
  // `pages` (typically via EnsureBufferDescsCached).
  //
  // Source layout: pages[i] receives src[i*page_size .. min((i+1)*page_size, size)).
  // We accept any `size` in ((N-1)*page_size, N*page_size] where N =
  // pages.size().  PageBitmapAllocator rounds up to ceil(size/page_size)
  // pages, so the *last* logical page is partially filled when size is not
  // a multiple of page_size; only that page's real bytes are transferred.
  //
  // Internally groups by buffer_index, builds one (localOffsets[k],
  // remoteOffsets[k], sizes[k]) batch per distinct buffer (= IOEngine's
  // outer N), and issues a single BatchWrite/BatchRead.  All-or-nothing:
  // if any sub-transfer fails the call returns false (no retry).
  bool RemoteDramScatterWrite(PeerConnection& peer, const std::vector<PageLocation>& pages,
                              uint64_t page_size, const void* src, size_t size, bool zero_copy);
  bool RemoteDramScatterRead(PeerConnection& peer, const std::vector<PageLocation>& pages,
                             uint64_t page_size, void* dst, size_t size, bool zero_copy);

  // BatchPut RDMA fusion internals.  All declared here because they
  // reference each other and need access to private members
  // (FindRegisteredMemory, peers_mutex_, IssueBatchWrite).  Structural
  // contract:
  //   - PlanPeerForBatchPut is the ONLY helper that takes PeerConnection&
  //     and may read peer.dram_memories[].
  //   - SubmitFusedBucket / WaitFusedBucket / MapBucketFailures take only
  //     FusedBucket& and never touch peer state.  Snapshots travel via
  //     FusedPair.remote_mem.
  struct BatchPutItemPlan {
    enum class Kind {
      SKIPPED,                // master-invariant violation, no Abort
      LOCAL,                  // same-node memcpy
      REMOTE_ZC,              // fused per-peer BatchWrite
      REMOTE_STG,             // legacy staging fallback (serial)
      REMOTE_ZC_PREP_FAILED,  // snapshot found buf not hydrated, owe Abort
    };
    size_t idx{0};
    Kind kind{Kind::SKIPPED};
    // Common (set when route validation passes):
    std::string peer_node_id;
    std::string allocation_id;
    uint64_t page_size{0};
    size_t bytes_total{0};
    // REMOTE_ZC only:
    mori::io::MemoryDesc local_mem{};
    size_t local_base_off{0};
    // Set after Phase 2/3.  Drives BatchFinalize inclusion.
    bool wrote{false};
  };
  struct FusedPair {
    uint32_t remote_buf_idx{0};
    mori::io::MemoryDesc remote_mem{};  // snapshotted under peers_mutex_
    mori::io::MemoryDesc local_mem{};
    std::vector<uint64_t> local_offsets;
    std::vector<uint64_t> remote_offsets;
    std::vector<uint64_t> sizes;
    std::vector<size_t> contributing_items;  // indices into batch input vectors
  };
  struct FusedBucket {
    std::string peer_node_id;
    std::vector<FusedPair> pairs;
    std::vector<mori::io::TransferStatus> statuses;
    std::vector<mori::io::TransferUniqueId> ids;
    std::vector<bool> submitted;  // per pair; false skips Wait + Aborts items
  };

  // Phase 1 helper.  Holds peers_mutex_ for the entire hydrate + snapshot
  // window so peer.dram_memories[] reads are consistent and FusedPair.
  // remote_mem carries a stable copy.  Mutates plans[] for items whose
  // buffer_index turns out unhydrated (kind = REMOTE_ZC_PREP_FAILED).
  void PlanPeerForBatchPut(PeerConnection& peer,
                           const std::vector<std::optional<RoutePutResult>>& routes,
                           const std::vector<size_t>& peer_item_indices,
                           const std::vector<std::string>& keys, const std::vector<size_t>& sizes,
                           std::vector<BatchPutItemPlan>& plans, FusedBucket& out_bucket);

  // Phase 1 submit.  Calls IssueBatchWrite under fire-and-return contract.
  // Sets bucket.submitted[k]=true for each pair successfully posted; on
  // exception during prep (uid alloc / vector resize / engine error) the
  // remaining pairs stay submitted=false and Phase 3 routes their items
  // straight to pending_aborts.
  void SubmitFusedBucket(FusedBucket& bucket);

  // Phase 3 helpers.  Stateless w.r.t. PoolClient (declared static); the
  // signature deliberately excludes PeerConnection& - structural
  // enforcement that no peer.dram_memories[] access leaks past Phase 1.
  static void WaitFusedBucket(FusedBucket& bucket);
  // Bucket is non-const because TransferStatus::Succeeded() / Message()
  // are non-const (they may poll progress callbacks); MapBucketFailures
  // does not actually mutate per-pair state besides reading status.
  static void MapBucketFailures(FusedBucket& bucket, const std::vector<BatchPutItemPlan>& plans,
                                std::vector<bool>& results,
                                std::vector<MasterClient::BatchAbortEntry>& pending_aborts);

  // BatchGet RDMA fusion internals (mirror of BatchPut, 3-phase pipeline:
  // BatchRouteGet -> per-peer fused BatchRead fire-and-return -> Wait).
  // FusedPair / FusedBucket types are reused from BatchPut above; in the
  // BatchRead path, FusedPair.local_mem is the NIC *write* destination
  // (caller dst), FusedPair.remote_mem is the NIC *read* source (peer buf).
  //
  // Kind set is one fewer than BatchPut: no REMOTE_ZC_PREP_FAILED because
  // Get has no master-side reservation to abort on snapshot miss.
  struct BatchGetItemPlan {
    enum class Kind {
      SKIPPED,     // route invalid / size mismatch / parse fail / OOB / snapshot miss
      LOCAL,       // same-node memcpy
      REMOTE_ZC,   // fused per-peer BatchRead
      REMOTE_STG,  // legacy staging fallback (serial, post-Wait memcpy)
    };
    size_t idx{0};
    Kind kind{Kind::SKIPPED};
    std::string peer_node_id;
    uint64_t page_size{0};
    size_t bytes_total{0};
    // Pre-parsed in Phase 1a, reused by Phase 1b/c per-peer loop and by
    // Phase 2 LOCAL/STG paths to avoid re-parsing loc.location_id.
    std::vector<PageLocation> parsed_pages;
    // REMOTE_ZC only:
    mori::io::MemoryDesc local_mem{};
    size_t local_base_off{0};
  };

  // Phase 1b/c helper.  Holds peers_mutex_ for the entire hydrate +
  // snapshot window so peer.dram_memories[] reads are consistent and
  // FusedPair.remote_mem carries a stable copy.  May mutate plans[] for
  // items whose buffer_index turns out unhydrated (kind = SKIPPED; no
  // Abort owed for Get).
  void PlanPeerForBatchGet(PeerConnection& peer,
                           const std::vector<std::optional<RouteGetResult>>& routes,
                           const std::vector<size_t>& peer_item_indices,
                           const std::vector<std::string>& keys,
                           std::vector<BatchGetItemPlan>& plans, FusedBucket& out_bucket);

  // Phase 1 submit for Read direction.  Calls IssueBatchRead under
  // fire-and-return contract.  bucket.statuses MUST be sized to
  // bucket.pairs.size() before any &statuses[k] is taken, mirroring
  // SubmitFusedBucket; otherwise vector relocation invalidates pointers
  // already handed to the IO engine.
  void SubmitFusedBucketRead(FusedBucket& bucket);

  // Phase 3 mapper for Read direction.  Differs from MapBucketFailures
  // in two ways: (1) no pending_aborts out-parameter (Get has no
  // reservation to abort), (2) emits per-item GET_BYTES counter for ZC
  // successes and accumulates bw_bytes_remote into the out parameter
  // for the single end-of-batch ObserveBatchBandwidth call.  Metrics
  // are emitted ONLY for items whose every contributing pair Wait
  // succeeded; failed items get results[orig_idx]=false and no metric.
  static void MapBucketFailuresRead(FusedBucket& bucket, const std::vector<BatchGetItemPlan>& plans,
                                    const std::vector<size_t>& sizes, std::vector<bool>& results,
                                    double* bw_bytes_remote_accum, MasterClient* master_client);

  // Test seam for BatchPut RDMA fusion failure injection.  Forwards to
  // io_engine_->BatchWrite under release builds; in MORI_UMBP_TESTING
  // builds it is `virtual` so a test subclass can override and synthesize
  // pair-level failures (see test_pool_client_batch_put_fused.cpp
  // PartialPairFailure).  See obs_counters.h for the build-mode contract.
  MORI_UMBP_TEST_VIRTUAL void IssueBatchWrite(const mori::io::MemDescVec& local_src,
                                              const mori::io::BatchSizeVec& local_offsets,
                                              const mori::io::MemDescVec& remote_dest,
                                              const mori::io::BatchSizeVec& remote_offsets,
                                              const mori::io::BatchSizeVec& sizes,
                                              mori::io::TransferStatusPtrVec& statuses,
                                              mori::io::TransferUniqueIdVec& ids);
  // Test seam for BatchGet RDMA fusion failure injection.  Forwards to
  // io_engine_->BatchRead; in MORI_UMBP_TESTING builds it is `virtual`
  // so a test subclass can synthesize pair-level Read failures (see
  // test_pool_client_batch_get_fused.cpp PartialPairFailure).  Build-mode
  // contract identical to IssueBatchWrite (obs_counters.h).
  MORI_UMBP_TEST_VIRTUAL void IssueBatchRead(const mori::io::MemDescVec& local_dest,
                                             const mori::io::BatchSizeVec& local_offsets,
                                             const mori::io::MemDescVec& remote_src,
                                             const mori::io::BatchSizeVec& remote_offsets,
                                             const mori::io::BatchSizeVec& sizes,
                                             mori::io::TransferStatusPtrVec& statuses,
                                             mori::io::TransferUniqueIdVec& ids);
  bool EnsurePeerServiceConnection(PeerConnection& peer);
  bool RemoteSsdWrite(PeerConnection& peer, const std::string& key, const void* src, size_t size,
                      bool zero_copy, uint32_t store_index = 0,
                      const std::string& allocation_id = "");
  bool RemoteSsdRead(PeerConnection& peer, const std::string& key, const std::string& location_id,
                     void* dst, size_t size, bool zero_copy);

  // Zero-copy registered memory regions
  struct RegisteredRegion {
    void* base;
    size_t size;
    mori::io::MemoryDesc mem_desc;
  };
  std::mutex registered_mem_mutex_;
  std::vector<RegisteredRegion> registered_regions_;

  std::optional<std::pair<mori::io::MemoryDesc, size_t>> FindRegisteredMemory(const void* ptr,
                                                                              size_t size);

  mutable std::mutex cache_mutex_;
  std::unordered_map<std::string, Location> cluster_locations_;

  // Observability counters.  See MORI_UMBP_TESTING macro in
  // obs_counters.h; declared unconditionally so class layout (atomics +
  // getters) is ABI-stable between release and test builds; only the
  // increment call sites are gated.
  std::atomic<uint64_t> abort_allocation_calls_{0};
  std::atomic<uint64_t> batch_abort_allocation_calls_{0};
  std::atomic<uint64_t> batch_abort_allocation_entries_{0};
  // BatchPut fusion counters.
  std::atomic<uint64_t> batch_put_calls_{0};
  std::atomic<uint64_t> batch_put_items_{0};
  std::atomic<uint64_t> batch_put_io_engine_calls_{0};
  std::atomic<uint64_t> batch_put_io_engine_pairs_{0};
  // BatchGet fusion counters (mirror of BatchPut).
  std::atomic<uint64_t> batch_get_calls_{0};
  std::atomic<uint64_t> batch_get_items_{0};
  std::atomic<uint64_t> batch_get_io_engine_calls_{0};
  std::atomic<uint64_t> batch_get_io_engine_pairs_{0};

  // Throttle state for the batch-level WARN emitted when BatchPut sees
  // an unregistered src and falls back to the per-item staging path.
  // Per-instance so unit tests get fresh state with each fresh
  // PoolClient.  Stores the steady_clock ns timestamp of the last
  // emitted warning; 0 means "never emitted".
  std::atomic<int64_t> last_batch_put_staging_warn_ns_{0};
  bool ShouldEmitBatchPutStagingWarn();
};

}  // namespace mori::umbp
