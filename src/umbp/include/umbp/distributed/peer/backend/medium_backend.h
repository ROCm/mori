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
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/metrics/component_metrics.h"
#include "umbp/distributed/transfer/transfer_engine.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

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
// The backend contract treats media as equivalent: it has no built-in read
// priority or put-eligibility flag. Legacy policies retain that behavior. The
// optional JSON TieredPlacementPolicy supplies ordering and offload topology
// above this interface, keeping physical storage implementations policy-free.
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
//   * byte-moving or staging methods — staging ownership stays an
//     implementation detail.  ResolveOutcome::kBusy is intentionally part of
//     the RESULT, however: a staged backend must distinguish transient pressure
//     from kMissing or the client excludes a node that does hold the key.
//   * local mode's TierBackend (blocking read/write of bytes) — a different
//     medium contract at a different layer, fenced off by lint in Phase 5
//   * single-key Allocate/Commit/Abort/Resolve — the single-key RPCs are
//     served by one-element batches
//
// A backend also must not move bytes itself, and since Phase 6 that is a
// COMPILE ERROR rather than a lint rule: Init receives a MemoryRegistrar, which
// has only RegisterMemory/Deregister on it.  The object behind that pointer is
// the same TransferEngine everything else holds — one object, two views (design
// doc §5 Rule C).

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

// A resolve needs one more state than a cache lookup.  kBusy means this backend
// owns the key but cannot publish its bytes until a transient resource (SSD
// staging) becomes available; callers must retry THIS peer rather than exclude
// it as they do for kMissing.  kFailed is a permanent shape/backend error and
// must not be retried indefinitely.
enum class ResolveOutcome {
  kMissing,
  kFound,
  kBusy,
  kFailed,
};

struct ResolvedEntry {
  ResolveOutcome outcome = ResolveOutcome::kMissing;
  // Kept for source and wire compatibility.  New code should set outcome too;
  // readers accept found=true as kFound for older backend implementations.
  bool found = false;
  std::vector<PageLocation> pages;
  uint64_t size = 0;
  uint64_t page_size = 0;
  // Empty when the caller passed include_descs=false (it already hydrated them
  // from GetPeerInfo), or when found=false.
  std::vector<BufferMemoryDescBytes> descs;
};

inline ResolveOutcome EffectiveResolveOutcome(const ResolvedEntry& entry) {
  return entry.found ? ResolveOutcome::kFound : entry.outcome;
}

struct EvictResult {
  std::string key;
  // 0 if the key was unknown, already freed, or protected (read lease / pin).
  // The master retries protected keys on a later round.
  uint64_t bytes_freed = 0;
};

class MediumBackend : public MetricSource {
 public:
  virtual ~MediumBackend() = default;

  MediumBackend(const MediumBackend&) = delete;
  MediumBackend& operator=(const MediumBackend&) = delete;

  // ---- identity + routing advertisement ----

  virtual TierType Tier() const = 0;
  virtual const char* Name() const = 0;

  // ---- ownership ----

  // Allocate this medium's pool with its own policy (hugepage/NUMA for DRAM,
  // hipMalloc for HBM, file extents for SSD) and register it with `registrar`,
  // supplying the facts a descriptor cannot recover: location type, device,
  // NUMA node.  PoolClient hands over the registrar; it does NOT hand over
  // memory.
  //
  // This is the inversion the refactor turns on.  Before Phase 2 allocation
  // lived in DistributedClient and registration in PoolClient::Init with
  // MemoryLocationType::CPU hardcoded, so a "backend" was a bookkeeper over
  // memory it did not own (design doc §1 item 4).
  virtual bool Init(MemoryRegistrar* registrar) = 0;

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

  // ---- observability ----
  //
  // A backend does NOT report its own operation counts, byte volumes or
  // latencies.  InstrumentedBackend sits on this interface and derives all of
  // those from the calls below, which is why a new medium is fully observable
  // the moment PoolClient composes it in — see instrumented_backend.h.
  //
  // MetricSource::SampleMetrics(), inherited here, is for the one thing a
  // decorator cannot see: state INSIDE the medium (the drive's own read
  // outcomes, single-flight coalescing, a staging arena's occupancy).  Publish
  // those under the generic MORI_UMBP_METRIC_BACKEND_MEDIUM_* names with the
  // specifics in an event=/state= label, never under a name that spells the
  // medium — a metric called mori_umbp_ssd_something forces a second dashboard,
  // which is exactly what the single-dashboard layout removed.
  //
  // tier= and backend= are added by the publisher; do not set them here.

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

  // Acquire a peer-local migration pin and resolve the key without changing or
  // cancelling normal RPC read leases. The caller must release a successful
  // acquisition after its synchronous transfer completes.
  virtual bool AcquireMigrationRead(const std::string& key, ResolvedEntry* resolved) {
    (void)key;
    (void)resolved;
    return false;
  }
  virtual void ReleaseMigrationRead(const std::string& key) { (void)key; }

  // Authoritative metadata lookup with no staging, descriptor construction, or
  // read-lease side effects. Pool placement validation must use this instead of
  // interpreting a temporarily unresolvable object as absent.
  virtual bool Contains(const std::string& key) const = 0;

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

  // ---- local (same-process) endpoints ----

  // How many buffers this backend published; PageLocation::buffer_index is an
  // index into that range.  Zero for a backend with no page-addressable
  // buffers.
  virtual size_t BufferCount() const = 0;

  // The endpoint for one buffer, in the form a same-process transfer plans
  // against — i.e. the UNPACKED, in-process counterpart of
  // AllBufferDescs()[i], with no msgpack round trip on the local hot path.
  // Returns an invalid ref for an out-of-range index.
  //
  // This is what replaced PageBackend::LocalBufferViews() in Phase 6, and the
  // difference is the point: a TransferRef is medium-agnostic (§2: "bytes
  // addressed by a descriptor are already medium-agnostic"), a raw base pointer
  // was not.  Publishing refs instead of pointers is what let PoolClient stop
  // naming a concrete backend type, and what makes a second medium's local
  // access work with no tier branch in a copy loop.
  virtual TransferRef BufferRef(uint32_t buffer_index) const = 0;

 protected:
  MediumBackend() = default;
};

// Owns every backend instance live on this peer. Names are stable
// configuration identities; backend ids are dense peer-local wire identities.
class BackendRegistry {
 public:
  // Defined in types.h so the transfer layer can size its per-backend buffer
  // shelves without depending on this header (see kMaxBackendsPerPeer).
  static constexpr uint32_t kMaxBackends = kMaxBackendsPerPeer;

  struct Entry {
    uint32_t backend_id = 0;
    std::string name;
    TierType tier = TierType::UNKNOWN;
    std::unique_ptr<MediumBackend> backend;
  };

  // Legacy registration keeps one conventional name per tier. Registering the
  // same name again replaces that instance and preserves its backend id.
  bool Register(std::unique_ptr<MediumBackend> backend) {
    if (backend == nullptr) return true;
    return Register(DefaultBackendInstanceName(backend->Tier()), std::move(backend));
  }

  bool Register(std::string name, std::unique_ptr<MediumBackend> backend) {
    if (backend == nullptr) return true;
    if (name.empty()) {
      MORI_UMBP_ERROR("[BackendRegistry] backend instance name must not be empty");
      return false;
    }

    const TierType tier = backend->Tier();
    auto existing = id_by_name_.find(name);
    const uint32_t replacing_id =
        existing == id_by_name_.end() ? kMaxBackends : existing->second;

    // Every wire carrying pages has one peer-global page size. When replacing
    // an entry, compare against the other live instances so replacing the sole
    // backend may legitimately establish a new size.
    uint64_t expected_page_size = 0;
    for (const auto& entry : entries_) {
      if (entry->backend_id == replacing_id) continue;
      expected_page_size = entry->backend->PageSize();
      break;
    }
    if (expected_page_size != 0 && backend->PageSize() != expected_page_size) {
      MORI_UMBP_ERROR(
          "[BackendRegistry] backend '{}' tier={} page_size={} disagrees with the peer's "
          "page_size={}; every instance on a node must agree",
          name, static_cast<int>(tier), backend->PageSize(), expected_page_size);
      return false;
    }

    if (existing != id_by_name_.end()) {
      auto& entry = entries_[existing->second];
      entry->tier = tier;
      entry->backend = std::move(backend);
      page_size_ = entries_.front()->backend->PageSize();
      return true;
    }

    if (entries_.size() >= kMaxBackends) {
      MORI_UMBP_ERROR("[BackendRegistry] too many backends (max {})", kMaxBackends);
      return false;
    }
    const uint32_t id = static_cast<uint32_t>(entries_.size());
    auto entry = std::make_unique<Entry>();
    entry->backend_id = id;
    entry->name = std::move(name);
    entry->tier = tier;
    entry->backend = std::move(backend);
    id_by_name_[entry->name] = id;
    entries_.push_back(std::move(entry));
    page_size_ = entries_.front()->backend->PageSize();
    return true;
  }

  // The backend's peer-local id, as stamped onto every buffer descriptor and
  // page set it produces.  kMaxBackends for an unregistered backend.
  uint32_t BackendId(const MediumBackend* backend) const {
    if (backend == nullptr) return kMaxBackends;
    for (const auto& entry : entries_) {
      if (entry->backend.get() == backend) return entry->backend_id;
    }
    return kMaxBackends;
  }

  // Uniform across every registered backend (see Register).  Zero when empty.
  uint64_t PageSize() const { return page_size_; }

  // Null if the tier has no live backend on this peer — a normal condition
  // (e.g. a node configured with DRAM only), not an error.  Callers respond to
  // a request for an absent tier with found=false / success=false.
  MediumBackend* Get(TierType tier) const {
    for (const auto& entry : entries_) {
      if (entry->tier == tier) return entry->backend.get();
    }
    return nullptr;
  }

  MediumBackend* Get(const std::string& name) const {
    const Entry* entry = GetEntry(name);
    return entry == nullptr ? nullptr : entry->backend.get();
  }

  MediumBackend* Get(uint32_t backend_id) const {
    const Entry* entry = GetEntry(backend_id);
    return entry == nullptr ? nullptr : entry->backend.get();
  }

  const Entry* GetEntry(const std::string& name) const {
    auto it = id_by_name_.find(name);
    return it == id_by_name_.end() ? nullptr : GetEntry(it->second);
  }

  Entry* GetEntry(const std::string& name) {
    return const_cast<Entry*>(std::as_const(*this).GetEntry(name));
  }

  const Entry* GetEntry(uint32_t backend_id) const {
    return backend_id < entries_.size() ? entries_[backend_id].get() : nullptr;
  }

  Entry* GetEntry(uint32_t backend_id) {
    return const_cast<Entry*>(std::as_const(*this).GetEntry(backend_id));
  }

  const Entry* GetEntry(const MediumBackend* backend) const {
    const uint32_t id = BackendId(backend);
    return id == kMaxBackends ? nullptr : GetEntry(id);
  }

  bool Empty() const { return entries_.empty(); }
  size_t Size() const { return entries_.size(); }

  // Every live backend in dense backend-id order.
  std::vector<MediumBackend*> All() const {
    std::vector<MediumBackend*> out;
    out.reserve(entries_.size());
    for (const auto& entry : entries_) out.push_back(entry->backend.get());
    return out;
  }

 private:
  std::vector<std::unique_ptr<Entry>> entries_;
  std::unordered_map<std::string, uint32_t> id_by_name_;
  uint64_t page_size_ = 0;
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
  std::vector<KvEvent> raw;
  for (auto* backend : backends) {
    if (backend == nullptr) continue;
    auto events = backend->DrainPendingEvents();
    raw.insert(raw.end(), std::make_move_iterator(events.begin()),
               std::make_move_iterator(events.end()));
  }

  // Master locations are (node, physical tier), not backend instances. A
  // same-tier migration therefore must not publish the source REMOVE after the
  // target ADD as if the node had lost the key. Coalesce by (tier,key) against
  // authoritative current ownership across all instances.
  struct EventState {
    std::optional<KvEvent> add;
    bool saw_remove = false;
  };
  std::map<std::pair<TierType, std::string>, EventState> states;
  for (auto& event : raw) {
    auto& state = states[{event.tier, event.key}];
    if (event.kind == KvEvent::Kind::ADD) {
      state.add = std::move(event);
    } else {
      state.saw_remove = true;
    }
  }
  std::vector<KvEvent> merged;
  merged.reserve(states.size());
  for (auto& [identity, state] : states) {
    const auto& [tier, key] = identity;
    const bool still_owned = std::any_of(
        backends.begin(), backends.end(), [&](MediumBackend* backend) {
          return backend != nullptr && backend->Tier() == tier && backend->Contains(key);
        });
    if (state.add.has_value() && (still_owned || !state.saw_remove)) {
      merged.push_back(std::move(*state.add));
    } else if (state.saw_remove && !still_owned) {
      merged.push_back(KvEvent{KvEvent::Kind::REMOVE, key, tier, 0});
    }
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
