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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp {

// Moves bytes between registered endpoints; also the registrar a backend uses
// to publish its own memory.  Defined in Phase 6 — forward-declared here so
// Init's signature is stable from the start.  See
// doc/design-backend-agnostic-refactor.md §4.
class TransferEngine;

// One storage medium on a peer.
//
// A backend OWNS bytes and PUBLISHES DESCRIPTORS for them.  It does not move
// them.  That split follows from one property of mori::io::MemoryDesc: it is
// self-describing about the medium (loc, deviceId, numaNode, ipcHandle,
// fabricHandle).  So —
//
//   bytes addressed by a descriptor are already medium-agnostic;
//   bytes addressed by a raw pointer are medium-specific;
//   registration is the one-way conversion, and belongs to whoever allocated.
//
// A backend is therefore the only layer that knows what kind of memory it is
// holding, because it is the one that allocated it.  Everything downstream of
// registration works off the descriptor and needs no per-medium branch (see
// design doc §2 for the rule, §3 for the three-component split).
//
// One instance == one medium.  That is why no method takes a TierType: the
// tier is the identity of the object, and the caller has already dispatched on
// it via BackendRegistry.  This inverts PeerDramAllocator's internal
// map<TierType, ...>, relocating the multi-medium generality to the registry
// where it is exercised.
//
// This interface absorbs OwnedLocationSource: the heartbeat drains/snapshots
// every registered backend and concatenates the events into ONE bundle under a
// single monotonic seq — never one seq per medium (that breaks the ack /
// seq-gap full-sync recovery).
//
// Every medium is currently treated as EQUIVALENT: no read priority, no
// put-eligibility flag.  Phase 4 deleted the routing plane's two hardcoded tier
// orders outright rather than re-expressing them as advertised properties, so
// there is deliberately no BackendProperties here.  A medium that genuinely
// differs — SSD, which takes no direct puts — brings the trait back with it
// (design doc §3 / §5 Phase 4); until one exists, an advertised order would be
// scaffolding nothing exercises.
//
// Threading: implementations must be safe to call from the peer service's gRPC
// handler threads and the heartbeat thread concurrently.
//
// NOT part of this contract, deliberately.  The first three were each proposed
// and then rejected once the transfer engine (§4) was settled; they are listed
// so they are not rediscovered and re-added:
//   * LocalCopyIn / LocalCopyOut — local access is a transfer whose endpoints
//     are both local; the engine's planner picks a memcpy or pread path for it.
//     A copy method here would re-admit a second byte-moving path, which is the
//     coupling this whole refactor exists to remove.
//   * a staging / bounce pool — belongs to the transfer engine, the only layer
//     that can observe completion.  A pool here would have to fall back to a
//     TTL, which is the deleted PrepareSsdRead lease under another name.
//   * a "not ready, retry here" resolve state — only needed to express
//     staging-pool exhaustion, which is no longer a backend resource.  Note it
//     must NOT be spelled as found=false: the client excludes a missing node
//     and retries elsewhere, which is wrong for a node that does hold the key.
//   * local mode's TierBackend (blocking read/write of bytes) — a different
//     medium contract at a different layer, fenced off by lint in Phase 5
//   * single-key Allocate/Commit/Abort/Resolve — the single-key RPCs are
//     served by one-element batches
//
// Backends must also not call TransferEngine::{Plan,Submit,Wait}; registration
// and transfer share one type (§4), so that boundary is enforced by the Phase 5
// lint rather than by the type system.

// ---------------------------------------------------------------------------
//  Data-plane value types
//
//  These are the medium-agnostic forms of PeerDramAllocator's nested request /
//  result structs.  Two differences from those, both intentional:
//    * no `tier` field — see the "one instance == one medium" note above
//    * no allocator-internal state (generation, deadline) — the wire carries
//      pending_ttl_ms, and generation is a backend implementation detail
// ---------------------------------------------------------------------------

// Outcome of one slot allocation.  kFailedNoSpace is split out from kFailed so
// the writer can keep retrying on other peers; every other failure collapses
// into kFailed and is diagnosed from the backend's own log.
enum class AllocateOutcome {
  kSuccessAllocated,
  kSuccessAlreadyExists,  // dedup hit against the backend's owned map
  kFailed,                // generic — see backend log for reason
  kFailedNoSpace,         // medium exhausted; retry on another peer
};

struct AllocateRequest {
  std::string key;
  uint64_t size = 0;
};

// Populated fields below `outcome` are meaningful ONLY when
// outcome == kSuccessAllocated.
struct AllocateResult {
  AllocateOutcome outcome = AllocateOutcome::kFailed;
  uint64_t slot_id = 0;
  std::vector<PageLocation> pages;
  uint64_t size = 0;
  uint64_t page_size = 0;
  // Descs for exactly the buffers referenced by `pages`, deduplicated and
  // ascending by buffer_index — enough for the writer to RDMA without a
  // follow-up GetPeerInfo.
  std::vector<BufferMemoryDescBytes> descs;
  // How long the slot survives without a matching BatchCommit / BatchAbort.
  uint64_t pending_ttl_ms = 0;
};

struct CommitRequest {
  uint64_t slot_id = 0;
  std::string key;
};

struct CommitResult {
  bool success = false;  // false = slot unknown: reaped, aborted, or never existed
  uint64_t bytes_committed = 0;
};

struct ResolvedEntry {
  bool found = false;
  std::vector<PageLocation> pages;
  uint64_t size = 0;
  uint64_t page_size = 0;
  // Empty when the caller passed include_descs=false (it already hydrated them
  // from GetPeerInfo), or when found=false.
  std::vector<BufferMemoryDescBytes> descs;
};

struct EvictResult {
  std::string key;
  // 0 if the key was unknown, already freed, or protected (read lease / pin).
  // The master retries protected keys on a later round.
  uint64_t bytes_freed = 0;
};

class MediumBackend {
 public:
  virtual ~MediumBackend() = default;

  MediumBackend(const MediumBackend&) = delete;
  MediumBackend& operator=(const MediumBackend&) = delete;

  // ---- identity + routing advertisement ----

  virtual TierType Tier() const = 0;
  virtual const char* Name() const = 0;

  // ---- ownership ----

  // Allocate this medium's pool with its own policy (hugepage/NUMA for DRAM,
  // hipMalloc for HBM, file extents for SSD) and register it with `engine`,
  // supplying the facts a descriptor cannot recover: location type, device,
  // NUMA node.  PoolClient hands over the engine; it does NOT hand over memory.
  //
  // This is the inversion the refactor turns on.  Today allocation lives in
  // DistributedClient and registration in PoolClient::Init with
  // MemoryLocationType::CPU hardcoded, so a "backend" is a bookkeeper over
  // memory it does not own (design doc §1 item 4).
  virtual bool Init(TransferEngine* engine) = 0;

  // Deregister and release the pool.  Must tolerate being called after a failed
  // Init, and must not run concurrently with any other method.
  virtual void Shutdown() = 0;

  // ---- control plane (shipped by the heartbeat) ----

  // Live capacity, derived from the backend's own allocator so it always
  // reflects pending + owned.  Cheap enough to call every heartbeat.
  virtual TierCapacity Capacity() const = 0;
  virtual uint64_t OwnedKeyCount() const = 0;

  // Events queued since the last call (delta heartbeat).  Clears the outbox.
  virtual std::vector<KvEvent> DrainPendingEvents() = 0;

  // Every owned key as ADD events.  const: a snapshot must not mutate state.
  virtual std::vector<KvEvent> SnapshotOwnedKeys() const = 0;

  // Full-sync snapshot that ALSO atomically drops the outbox, in the SAME
  // critical section.  After a full sync the snapshot is authoritative, so a
  // still-queued event is already reflected in it and must not be re-shipped
  // as a redundant delta.  Two separate locks would drop events committed in
  // between.
  virtual std::vector<KvEvent> SnapshotOwnedKeysForFullSync() = 0;

  // Distributed Clear: drop owned state, invalidate outstanding slots so they
  // fail at commit, and clear the outbox.  Allocation stays gated until
  // ClearFullSyncAcked().
  virtual void ClearLocal() = 0;

  // Called by the heartbeat thread once master has acked the first full-sync
  // empty snapshot.  Re-enables allocation.
  virtual void ClearFullSyncAcked() = 0;

  // True between ClearLocal() and ClearFullSyncAcked().
  virtual bool IsClearFullSyncPending() const = 0;

  // Register a hook the backend invokes once its unshipped-event outbox
  // reaches `threshold`, so a batch of puts flushes the heartbeat instead of
  // waiting for the next tick.  On the interface (rather than on the concrete
  // backend) because every medium's outbox feeds the same single-bundle
  // heartbeat: a newly registered backend must participate without MasterClient
  // learning its type.  `cb` MUST be cheap and MUST NOT re-enter the backend —
  // it should only signal the heartbeat thread.  Pass a very large threshold to
  // disable.
  virtual void SetAutoFlushHook(size_t threshold, std::function<void()> cb) = 0;

  // ---- slot lifecycle (peer-side; driven by PeerServiceServer handlers) ----
  //
  // Deliberately not called "data plane": these calls hand out and retire slots
  // and publish descriptors for them.  The bytes move through the transfer
  // engine, not through here.

  // One result per request, in request order.
  virtual std::vector<AllocateResult> BatchAllocate(const std::vector<AllocateRequest>&) = 0;
  virtual std::vector<CommitResult> BatchCommit(const std::vector<CommitRequest>&) = 0;

  // Idempotent: a slot already reaped or never seen still reports true.  False
  // is reserved for a state not currently produced.
  virtual std::vector<bool> BatchAbort(const std::vector<uint64_t>& slot_ids) = 0;

  // Look up keys and take a read lease on each hit, protecting it from
  // concurrent Evict for the lease window.  `include_descs=false` skips
  // building descs entirely for the whole batch.  A medium whose own bytes are
  // not registerable publishes a descriptor naming its storage (a file/object
  // ref) rather than a memory one — the engine decides how to reach it, and
  // supplies a bounce buffer if the chosen path needs one.
  virtual std::vector<ResolvedEntry> BatchResolve(const std::vector<std::string>& keys,
                                                  bool include_descs) = 0;

  // Master-driven eviction.  Idempotent; see EvictResult::bytes_freed.  One
  // result per key, in request order — the peer service relies on that to sum
  // freed bytes for a key mirrored across media.
  virtual std::vector<EvictResult> Evict(const std::vector<std::string>& keys) = 0;

  // ---- bootstrap (GetPeerInfo) ----

  // Page size of this medium.  Also mirrored into every AllocateResult /
  // ResolvedEntry so a batch spanning backends stays self-describing.
  virtual uint64_t PageSize() const = 0;

  // Descs for ALL configured buffers, ascending by buffer_index, so a
  // first-contact writer can hydrate its peer-side cache in one round trip
  // instead of learning buffers one allocation at a time.
  virtual std::vector<BufferMemoryDescBytes> AllBufferDescs() const = 0;

 protected:
  MediumBackend() = default;
};

// Owns every medium live on this peer, keyed by tier.  Held by PoolClient; the
// registration call sites are the single place in the tree where a concrete
// backend type is named, which is what the Phase 5 lint enforces.
//
// std::map iterates in ascending TierType order, so All() is both the
// enumeration and the (deterministic, arbitrary) order callers walk media in —
// with every medium equivalent there is no priority to express.
class BackendRegistry {
 public:
  // Replaces any backend already registered for that tier.  Null is ignored.
  void Register(std::unique_ptr<MediumBackend> backend) {
    if (backend == nullptr) return;
    const TierType tier = backend->Tier();
    backends_[tier] = std::move(backend);
  }

  // Null if the tier has no live backend on this peer — a normal condition
  // (e.g. a node configured with DRAM only), not an error.  Callers respond to
  // a request for an absent tier with found=false / success=false.
  MediumBackend* Get(TierType tier) const {
    auto it = backends_.find(tier);
    return it == backends_.end() ? nullptr : it->second.get();
  }

  bool Empty() const { return backends_.empty(); }
  size_t Size() const { return backends_.size(); }

  // Every live backend, ascending by TierType.
  std::vector<MediumBackend*> All() const {
    std::vector<MediumBackend*> out;
    out.reserve(backends_.size());
    for (const auto& [tier, backend] : backends_) out.push_back(backend.get());
    return out;
  }

 private:
  std::map<TierType, std::unique_ptr<MediumBackend>> backends_;
};

// ---------------------------------------------------------------------------
//  Heartbeat event aggregation
//
//  These absorb the deleted OwnedLocationSource free functions (design doc §3:
//  "This interface absorbs the existing OwnedLocationSource — that type goes
//  away").  Exposed rather than private to MasterClient so they can be unit
//  tested against backends without standing up a master RPC.
// ---------------------------------------------------------------------------

// Drain every backend and concat into ONE event list, in the given order.  The
// heartbeat shipper wraps the result in a SINGLE EventBundle under one
// monotonic seq — concat here, never one bundle/seq per medium (that breaks the
// ack / seq-gap full-sync recovery).  Null entries are skipped.
inline std::vector<KvEvent> DrainAllBackends(const std::vector<MediumBackend*>& backends) {
  std::vector<KvEvent> merged;
  for (auto* backend : backends) {
    if (backend == nullptr) continue;
    auto events = backend->DrainPendingEvents();
    merged.insert(merged.end(), std::make_move_iterator(events.begin()),
                  std::make_move_iterator(events.end()));
  }
  return merged;
}

// Full-sync counterpart: snapshot every backend and concat, each backend ALSO
// atomically clearing its outbox in the same critical section (the snapshot is
// authoritative, so the queued delta is redundant and must not be re-shipped).
// Null entries are skipped.  Non-const because it mutates the backends.
inline std::vector<KvEvent> SnapshotAllBackendsForFullSync(
    const std::vector<MediumBackend*>& backends) {
  std::vector<KvEvent> merged;
  for (auto* backend : backends) {
    if (backend == nullptr) continue;
    auto events = backend->SnapshotOwnedKeysForFullSync();
    merged.insert(merged.end(), std::make_move_iterator(events.begin()),
                  std::make_move_iterator(events.end()));
  }
  return merged;
}

}  // namespace mori::umbp
