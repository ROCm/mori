# UMBP Backend-Agnostic Refactor — Plan

**Goal:** make UMBP's data plane, control plane, and routing plane backend-agnostic.
Adding a new storage medium becomes: implement one interface + register it. No
call-site edits in `PoolClient`, `PeerServiceServer`, or the routing strategies.

**Method:** remove SSD **from distributed mode only**, then generalize what
remains. Local (standalone) mode keeps its SSD tier, and the whole storage
implementation stack underneath it — SPDK, the io_uring/posix drivers, the
offset allocator, the proxy protocol — stays in tree, built and tested,
untouched.

This is deliberate and temporary: SSD is expected back in distributed mode
*after* the refactor, as an `SsdBackend : MediumBackend`. Taking it out first is
purely to make the refactor simple — the distributed data plane collapses to one
shape (async RDMA page slots) while it is being generalized, and SSD re-enters
through the finished abstraction instead of being carried through every phase.
So the rule for Phase 0 is: **delete the coupling, keep the implementation.**
Anything reusable by a future SSD backend is preserved.

During Phases 1–5 the live distributed media are DRAM and HBM, which are already
two instances of the same page-slot model.

**Scope:** `src/umbp/` — `distributed/` (pool client, peer service, routing),
`local/` (standalone stack), `storage/`.

---

## 1. What is actually coupled

Four specific things, not a diffuse mess.

**1. `PoolClient` hardcodes two different data-plane shapes.**
DRAM is async — `SubmitRemoteBatchGet` / `WaitRemoteBatchGet`, RDMA straight
into the peer's pages. SSD is blocking-only — `RemoteSsdReadOnce`, which does a
key-based `PrepareSsdRead` into a peer-side staging slot, takes a lease, RDMAs
out of the staging buffer, then releases. `BatchGetPlan`'s 4-way partition
(`remote_dram_groups` / `remote_ssd_groups` / `local_dram_indices` /
`local_ssd_indices`) exists only to reconcile these two shapes inside one batch
while preserving overlap.

This asymmetry is roughly 90% of the refactoring cost, and it disappears the
moment SSD leaves the *distributed* path. Nothing about it is inherent to the
storage medium — it is inherent to `PrepareSsdRead`'s key-based, lease-held,
staging-slot shape.

**2. `PeerServiceServer` takes four typed pointers.**
`PeerDramAllocator*`, `PeerSsdManager*`, `SsdCopyPipeline*`, plus the SSD
staging base/size/`MemoryDesc` args. The SSD-only RPCs (`PrepareSsdRead`,
`ReleaseSsdLease`) and the read-slot lease state machine live in the handlers.

**3. Routing is nearly agnostic already.** *(resolved — Phase 4)*
`RoutePutStrategy` and `RouteGetStrategy` only ever see
`ClientRecord.tier_capacities` and `Location.tier`. The single hardcode is two
orderings: `TierPriorityRouteGetStrategy`'s `HBM > DRAM > SSD`, and put's
"HBM then DRAM, SSD is never a direct-put target." Both were **deleted** in
Phase 4 rather than re-expressed as advertised properties — see §5 Phase 4.

**4. Memory ownership lives entirely outside the backend.**
This is the one the original plan missed, and it is the one that makes the
Phase 5 acceptance test unreachable.

| Concern | Where it lives today | Medium-specific? |
|---|---|---|
| Pool allocation | `distributed_client.cpp:41-53` — `HostMemAllocator`, hugepages, NUMA, prefault | yes |
| Pool free | `distributed_client.cpp:271-276` | yes |
| RDMA registration | `pool_client.cpp:360-366` — `MemoryLocationType::CPU` hardcoded | yes |
| Desc/base marshalling into the allocator | `BuildDramTierConfig`, `pool_client.cpp:286-303` | yes |
| Local put/get byte copy | `pool_client.cpp:559-599` + `LocalCopyBlock` `:181-190` (AVX2-NT / `memcpy`) | yes |
| Buffer indexing | `config_.dram_buffers[p.buffer_index]` `:564`, `:585` — **no tier dimension** | yes |
| Copy parallelism | `UMBP_DRAM_{WRITE,READ}_THREADS` `:836`, `:1435` | yes |
| Capacity advertisement | `distributed_client.cpp:70-71` — `{DRAM, {size,size}}` literal | yes |
| Re-cache install tier | `pool_client.cpp:718` — `TierType::DRAM` literal | yes |
| Put tier filter | `pool_client.cpp:821` — `!= DRAM && != HBM` | yes |

`PeerDramAllocator` owns only bookkeeping: the page bitmap, the key→slot map,
pending slots and their reaper, read leases, copy pins, the event outbox, the
clear generation. Everything about the memory itself arrives pre-cooked through
`BuildDramTierConfig`.

So §3's premise — that a backend owns the slots it publishes descriptors for —
is aspirational. Today the client allocates, registers, and copies; the backend
is a bookkeeper over memory it does not own. Adding HBM under this ownership needs
a new config buffer list, a second `RegisterMemory` with `GPU`, a tier branch in
`LocalCopyBlock`, and a fix for the missing tier dimension: four call-site edits
before the registry line.

**Not part of this** — a separate axis that must not be moved into a backend:
`PoolClient::RegisterMemory` (`:526`) registers the **caller's** buffer as
`MemoryLocationType::CPU` unconditionally. That is wrong for a GPU `src`/`dst`
(SGLang HiCache KV pages), but it is a property of the caller's allocation, not
of any storage medium. It needs a location parameter on the public API, not a
backend.

**Key observation:** `PeerDramAllocator` is *shaped* for multiple media — an
internal `map<TierType, PageBitmapAllocator>` plus parallel `tier_descs_` /
`tier_bases_` maps, with a `TierType` parameter threaded through every method —
but it serves exactly **one** live tier. `PoolClient::Init` default-constructs an
empty `hbm_cfg` (`pool_client.cpp:377`, "HBM not currently exposed via
PoolClientConfig"), so no HBM buffer is ever passed in. `TierType::HBM`, its
routing rank, and the tier map are unexercised scaffolding.

The refactor *inverts* that map: one backend instance per medium, instead of one
object holding a map of media. This deletes generality that was never live and
relocates it to the registry, where it is exercised.

**Second key observation — the distributed SSD code is a thin adapter, not a
storage stack.** `PeerSsdManager` (665 LOC) forward-declares `TierBackend` and
holds one (`peer_ssd_manager.h:43,65`); the actual bytes are moved by local
mode's `SsdTier` / `SpdkSsdTier` over `storage/io/**` and `storage/spdk/**`
(~5.5k LOC). Distributed SSD therefore sits *on top of* local SSD. Unwiring it
from distributed mode is a cut at the adapter seam: it touches none of the
storage implementation, and — because the adapter's own dependencies are just
`config.h` and `types.h` — not even the adapter itself has to go. What is
actually deleted is the *lease-shaped call sites*, not SSD code. That is why
"keep it, just stop calling it" is cheap rather than a compromise.

---

## 2. The boundary

Everything below follows from one property of `mori::io::MemoryDesc`
(`include/mori/io/common.hpp:100-127`): it is **self-describing about the
medium**.

```cpp
MemoryLocationType loc;          // CPU | GPU
int deviceId; std::string deviceBusId; int numaNode;
std::array<char, kIpcHandleSize>    ipcHandle;
std::array<char, kFabricHandleSize> fabricHandle;  int vpodId;
```

Hence the criterion for deciding where any piece of code belongs:

> **Bytes addressed by a descriptor are already medium-agnostic. Bytes addressed
> by a raw pointer are medium-specific. Registration is the one-way conversion
> between the two, and it belongs to whoever allocated the memory.**

Three consequences, each checkable against the tree:

**The remote path already works for a medium it was never written for.** A peer
whose pages are HBM can be read today, unmodified, because that peer registered
them and the desc carries `loc=GPU`. `RemoteDramScatterRead`,
`GroupTransfersByPair`, `SubmitRemoteBatchGet` and the peer desc cache never
dereference anything — they only pass descs to the IO engine. None of them
needed this refactor and none of them change.

**The local path breaks because it contains no descriptor at all.**
`LocalCopyBlock(static_cast<char*>(dram.buffer) + off, ...)` (`:575`) operates on
a raw `void*` out of `config_.dram_buffers`, bypassing the descriptor layer.

**That is a deliberate performance choice, not an oversight.** The local fast
path exists because a `memcpy` beats an engine round trip. Medium-awareness is
the price already being paid for it. This matters for §4: the fast path folds
back into the transfer layer only if a local transfer through the abstraction is
genuinely as cheap, which is a measurement, not an argument.

Practical form of the rule, usable in review: **if adding a new medium would
force you to edit it, it belongs to the backend.**

---

## 3. The abstraction

The system has three components. The refactor's job is to make the boundaries
between them real.

| Component | Owns | Knows about media? |
|---|---|---|
| **Control plane** — master, routing, heartbeat | who holds what, capacity, eviction, placement | only via advertised `BackendProperties` |
| **Storage backends** — DRAM, HBM, SSD, … | bytes, the key→slot map, capacity, eviction | yes — this is the only layer that does |
| **Transfer engine** (§4) | moving bytes between descriptors | no — descriptors are self-describing |

New header: `include/umbp/distributed/peer/medium_backend.h`.

A backend's job is to **publish descriptors** for the slots it owns, and to keep
the key→slot bookkeeping around them. It does not move bytes.

```cpp
class MediumBackend {
 public:
  virtual ~MediumBackend() = default;

  // ---- identity ----
  virtual TierType Tier() const = 0;
  virtual const char* Name() const = 0;

  // ---- ownership (new; see §1 item 4) ----
  // The backend allocates its own pool with its own policy (hugepage/NUMA vs
  // hipMalloc vs file extents) and registers it, choosing the location type
  // itself.  PoolClient hands it the engine; it does not hand it memory.
  virtual bool Init(TransferEngine*) = 0;
  virtual void Shutdown() = 0;

  // ---- control plane (shipped by the heartbeat) ----
  virtual TierCapacity Capacity() const = 0;
  virtual uint64_t OwnedKeyCount() const = 0;
  virtual std::vector<KvEvent> DrainPendingEvents() = 0;
  virtual std::vector<KvEvent> SnapshotOwnedKeys() const = 0;
  virtual std::vector<KvEvent> SnapshotOwnedKeysForFullSync() = 0;
  virtual void ClearLocal() = 0;
  virtual void ClearFullSyncAcked() = 0;
  virtual bool IsClearFullSyncPending() const = 0;

  // ---- slot lifecycle (peer-side; driven by PeerServiceServer handlers) ----
  virtual std::vector<AllocateResult> BatchAllocate(const std::vector<AllocateRequest>&) = 0;
  virtual std::vector<CommitResult>   BatchCommit(const std::vector<CommitRequest>&) = 0;
  virtual std::vector<bool>           BatchAbort(const std::vector<uint64_t>& slot_ids) = 0;
  virtual std::vector<ResolvedEntry>  BatchResolve(const std::vector<std::string>& keys,
                                                   bool include_descs) = 0;
  virtual std::vector<EvictResult>    Evict(const std::vector<std::string>& keys) = 0;

  // ---- bootstrap (GetPeerInfo) ----
  virtual uint64_t PageSize() const = 0;
  virtual std::vector<BufferMemoryDescBytes> AllBufferDescs() const = 0;
};
```

`AllocateResult` and `ResolvedEntry` carry `{slot_id, pages, size, page_size,
descs, pending_ttl_ms}`. No method takes a `TierType`: one instance is one
medium, and the caller has already dispatched on it via the registry.

**`BackendProperties` was proposed here and then NOT adopted.** The idea was to
make read order and put-eligibility *advertised* facts rather than a hand-written
tier list in the strategies. Phase 4 found the simpler answer: every medium in
the system today is equivalent, so the tier lists were **deleted outright** and
nothing replaced them. An advertised order would have been scaffolding nothing
exercises — exactly the mistake §1's "key observation" calls out about
`PeerDramAllocator`'s unexercised tier map.

The trait comes back when a medium that genuinely differs does. That is SSD:
it takes no direct puts. `SsdBackend` reintroduces `put_eligible` *with* the
backend that needs it, which is the same "1 file + 1 registry line" test Phase 5
already applies.

**What is deliberately NOT on this interface**, and why — each of these was
proposed and rejected once §4 was settled:

- **`LocalCopyIn` / `LocalCopyOut`.** Local access is a transfer whose endpoints
  are both local; §4 selects a memcpy or pread engine for it. Putting a copy
  method on the backend would re-admit a second byte-moving path.
- **A staging/bounce pool.** Belongs to the transfer layer, which is the only
  layer that can observe transfer completion (§4).
- **A "not ready / retry here" resolve state.** Only needed to express
  staging-pool exhaustion, which is no longer a backend resource.

All three were artifacts of a transfer engine that speaks only memory. They are
listed here so they are not rediscovered and re-added.

This interface absorbs the existing `OwnedLocationSource` — that type goes away.
The dormant `PeerSsdManager` (Phase 0) keeps its `DrainPendingEvents` /
`SnapshotOwnedKeys` methods but drops the `: public OwnedLocationSource` base and
the `override`s — a ~4-line change.

Ownership: a `BackendRegistry` (`map<TierType, unique_ptr<MediumBackend>>`) held
by `PoolClient` is the single place in the tree where a concrete backend type is
named. It exposes `Get(tier)`, `All()`, and `ByReadRank()` — the last being the
peer-local counterpart of `RouteGetStrategy`'s cross-peer ranking, used when a
key is mirrored across media.

---

## 4. The transfer engine

`mori-io` is a **memory-to-memory** engine: `BackendType` is
`{XGMI, RDMA, TCP, FABRIC}` and `MemoryLocationType` is `{CPU, GPU}`
(`include/mori/io/enum.hpp:27-41`). There is no file or object endpoint.

**Decision: do not change mori-io.** Abstract the transfer engine inside UMBP
and make mori-io one implementation. UMBP defines its own descriptor variant, so
mori-io never has to learn about files:

```cpp
struct TransferRef {
  std::variant<MemoryRef,   // wraps mori::io::MemoryDesc verbatim
               FileRef,     // fd/path + offset
               ObjectRef>   // bucket + key
  target;
};

class TransferEngine {
 public:
  virtual ~TransferEngine() = default;

  // ---- registration ----
  // The backend supplies the facts a descriptor cannot recover — location type,
  // device, NUMA node — because it is the allocator (§6).
  virtual TransferRef RegisterMemory(void* base, size_t size, MemoryLocationType, int device) = 0;
  virtual TransferRef RegisterFile(int fd, uint64_t offset, uint64_t size) = 0;
  virtual void        Deregister(const TransferRef&) = 0;

  // ---- transfer ----
  virtual bool CanHandle(const TransferRef& src, const TransferRef& dst) const = 0;
  virtual std::vector<Plan> Plan(const std::vector<TransferItem>&) = 0;   // see below
  virtual Handle Submit(const std::vector<Plan>&) = 0;
  virtual bool   Wait(Handle&) = 0;
};
```

`MoriIoEngine` accepts only the `MemoryRef` variant and unwraps straight to
`mori::io::MemoryDesc`. A `PosixEngine` / `GdsEngine` accepts a `FileRef` on one
side. Engine selection is a function of the `(src, dst)` pair.

**One type, including registration.** Registration and transfer are different
shapes — registering fans *out* across every transport that can reach an
endpoint, while a transfer dispatches to exactly *one*. mori-io already carries
that fan-out in its descriptor: `MemoryDesc` holds `ipcHandle` and
`fabricHandle` side by side, "so XGMI and Fabric backends can register the same
memory without clobbering each other's handle."

That difference does not need a second class. `CompositeTransferEngine :
TransferEngine` holds the concrete engines, implements `RegisterMemory` /
`RegisterFile` by fanning out and merging the per-transport handles into one
`TransferRef`, and implements `Plan` / `Submit` by selecting or chaining. The
composite *is* a `TransferEngine`, so the rest of the system holds exactly one
pointer to exactly one type: `PoolClient` owns it, and backends receive the same
pointer at `Init`.

The cost of folding registration in: a backend now holds something it *could*
call `Submit` on. That invariant — backends do not move bytes — moves from the
type system to Phase 5's lint, as one rule: no file under a backend directory may
call `Submit` / `Plan` / `Wait`.

**`Plan()` is virtual on purpose.** `GroupTransfersByPair` — collapsing a
scatter into one transfer per `(localMR, remoteMR)` — is an RDMA optimization
that cuts CQE and post count. A memcpy engine gains nothing from it; a file
engine would rather group by file and merge adjacent offset ranges. Submit-all-
then-wait *is* universal and belongs in the base class; the grouping strategy is
not. Note the current implementation assumes one page size per MR pair, so
pushing it down requires generalizing it first.

**Bounce buffers belong here, not in a backend.** The transfer layer is the only
layer with a completion signal (`TransferStatus::Wait()`). A staging pool inside
a storage backend must guess with a TTL, because the backend cannot observe when
a remote reader's RDMA finishes — that is exactly the shape of the
`PrepareSsdRead` lease this refactor exists to delete. One pool, owned by the
transfer layer, released on completion rather than on a timer.

**Chaining.** When no single engine spans the pair — a 3FS file on node A, a
reader on node B — the layer composes: a file engine fills a bounce buffer, the
RDMA engine ships it. The extra hop is real, but it is an explicit plan in one
place, and if mori-io ever grows a storage backend the planner simply stops
chaining. Nothing above notices.

**What stays in the client**, because it needs semantics the engine must not
have: which node and tier to target; retry-elsewhere with an exclude set (needs
to know replicas exist); the slot lifecycle, where `Commit` is what publishes a
key; capacity, eviction, and leases.

---

## 5. Phases

### Phase 0 — unwire SSD from distributed mode *(done — `ef081fd2`)*

Cut at the adapter seam. Local mode is not touched; no storage-layer code is
deleted.

**Deleted — the coupling only:**

- `include/umbp/distributed/ssd_read_lease.h` (76 LOC) and
  `test_ssd_read_lease_gating` — the lease *is* the coupling, and the re-added
  backend will not use it
- The construction and wiring of `PeerSsdManager` / `SsdCopyPipeline` in
  `distributed_client.cpp`, `pool_client.{h,cpp}`, `peer_service.{h,cpp}`, and
  the `AddOwnedLocationSource(ssd_manager)` arm in `master_client.cpp`
- Every SSD member and method in `pool_client.{h,cpp}` — `ExecuteLocalSsdGet`,
  `RemoteSsdReadOnce`, `ReleaseSsdLeaseBestEffort`, `ProcessRemoteSsdBatchGet`,
  `PublishSsdMetrics`, `SsdMetricsLastShipped`, `SsdGetOutcome`, the SSD staging
  buffer and its `MemoryDesc` (~470 SSD-touching lines across the pair)
- The SSD read-slot state machine, `StagingMetrics`, and the staging ctor args in
  `peer_service.{h,cpp}`
- `PrepareSsdRead` / `ReleaseSsdLease` from `distributed/proto/umbp_peer.proto`
  — `reserved` the field/method tags rather than freeing them
- SSD counters in `master/master_metrics.h`
- The 3 tests that exercise the *coupling* rather than the adapter:
  `test_peer_ssd_read_rpc`, `test_ssd_read_lease_gating`, `test_ssd_reliability`,
  plus the SSD arms of `test_peer_service`

`BatchGetPlan` collapsed from 4 buckets to 2 (`remote_groups`, `local_indices`).

`TierType::SSD = 3` stays reserved in the enum so wire tags stay stable; every
*live* distributed code path that produces or consumes it is gone.

**Kept — dormant but still compiled and still tested:**

- `distributed/peer/ssd/peer_ssd_manager.{h,cpp}` (665 LOC) and
  `distributed/peer/ssd/ssd_copy_pipeline.{h,cpp}` (~250 LOC). Nothing
  constructs them; they remain in the build. Moved under
  `distributed/peer/ssd/` so "dormant" is legible in the tree and the Phase 5
  lint exemption can be a directory rather than a filename list.
- `PeerSsdConfig` (`distributed/config.h`) — `peer_ssd_manager.h` needs it.
- Their 3 unit tests: `test_peer_ssd_manager`, `test_peer_ssd_eviction`,
  `test_ssd_copy_pipeline`. These test the adapter standalone and stay green.

**Kept — untouched, still built, still tested:** `storage/io/**`,
`storage/spdk/**` and the `3rdparty/spdk` submodule, `local/tiers/*` including
`LocalStorageManager`'s demote/promote, `UMBPSsdConfig` and its Python binding,
and the local SSD test suites.

*Gate:* ctest green — **including the local SSD suites and the 3 dormant adapter
tests**, unmodified. That is the check that the seam was cut in the right place.

### Phase 1 — write the interfaces

Add `medium_backend.h` (§3). Nothing implements it yet.

*Gate:* tree compiles. Note this gate is weak by construction — a header nothing
includes is not in any translation unit. Verify self-containment explicitly
(`-fsyntax-only` against the project's real flags) rather than relying on the
build.

### Phase 2 — DRAM becomes a backend, and takes ownership of its memory

Two pieces, and the second is the one §1 item 4 added:

**(a) Split the map.** `PeerDramAllocator`'s internal `map<TierType, ...>`
becomes one instance per medium. Rename to `PageBackend : MediumBackend`. Every
method loses its `TierType` parameter — this makes the file *smaller*.

**(b) Move ownership down.** The backend allocates its own pool and registers it,
choosing `MemoryLocationType` itself. Consequences up the stack:
`PoolClientConfig::dram_buffers` is deleted; `DistributedClient` stops calling
`HostMemAllocator`; `BuildDramTierConfig` is deleted; the
`{DRAM, {size,size}}` capacity literal is deleted in favour of `Capacity()`
aggregated over the registry.

This does **not** wait for Phase 6. The `TransferEngine` a backend receives at
`Init` is, in this phase, a ~30-line `PoolClient`-owned shim whose registration
methods forward to `io_engine_->RegisterMemory` and whose transfer methods are
unimplemented — nothing calls them yet, since the remote path still runs through
`PoolClient` until Phase 6. Phase 6 replaces the shim with
`CompositeTransferEngine` and no backend changes. Phase 1's header only
forward-declares the type, so it compiles with nothing behind it.

Consequence to accept deliberately: `owned_`, the read lease, and the clear
generation become per-medium, so a key may exist in both DRAM and HBM. The
design already allows this (see the `NodeMatch` mirror comment in `types.h`).
Peer-local tier selection at `Resolve` time becomes a small policy function
driven by `BackendRegistry::ByReadRank()`.

**The local fast path is NOT touched in this phase.** `LocalPutPages` /
`LocalGetPages` stay in `PoolClient`, host-DRAM-only. That is sound as long as
only one *real* medium is live, which is why option (b) below is the default.

**HBM is not a free second backend.** It does not work today — no HBM buffer is
ever allocated or registered (§1). Two options:

- **(a) Wire HBM up for real.** This is the genuine proof of the abstraction,
  but it is *blocked on Phase 6*: a second real medium makes the host-only local
  copy path wrong, and `config_.dram_buffers[buffer_index]` has no tier
  dimension. Doing it before Phase 6 means interim scaffolding that Phase 6 then
  deletes.
- **(b) `MockBackend` as the second registered backend.** Cheap, proves the
  registry and dispatch paths, keeps the local path honestly single-medium.

Default to (b).

*Gate:* DRAM behavior unchanged with a second backend registered and dispatched
to; no buffer pointer crosses into `PoolClientConfig`.

### Phase 3 — peer server and pool client go agnostic

`PeerServiceServer` takes a `BackendRegistry*` instead of typed pointers;
handlers dispatch on the request's tier tag. Single-key RPCs are served by
one-element batches.

`PoolClient`'s local paths go through `registry_.Get(tier)`. The remote paths are
already tier-tagged on the wire, so once the SSD special case is gone they become
generic with no further work.

*Gate:* `bench_pool_client_batch_get` throughput unchanged. The submit/wait
overlap is a throughput property that no assertion catches — it must be measured,
not asserted.

### Phase 4 — routing plane *(done)*

Delete both hardcoded tier orders. **Scope decision: delete, do not replace.**
The original plan derived read order and put-eligibility from an advertised
`BackendProperties`; with every live medium equivalent there is nothing to rank,
so the orders were removed and no mechanism took their place (see §3).

What went:

- `TierReadRank` and the `HBM > DRAM > SSD` read order.
  `TierPriorityRouteGetStrategy` → `LocalPreferringRouteGetStrategy`: it keeps
  the requester-local preference (the thing that makes `cache_remote_fetches`
  pay off) and drops the ranking. The old name described the deleted half.
- `kPutTierOrder = {HBM, DRAM}` and its "SSD is never a direct-put target"
  exclusion. `SelectByAlgo` now scores every `(node, tier)` pair that has room,
  on free space alone.
- `BackendProperties`, `MediumBackend::Properties()`, and
  `BackendRegistry::ByReadRank()` — the last became redundant with `All()`,
  which already iterates in a deterministic (ascending-`TierType`) order.
- `EvictionManager`'s `if (tier == SSD) continue`. It existed because `EvictKey`
  only ever reached the peer's DRAM allocator; since Phase 3 it fans out to
  every backend by key, so an overloaded medium now evicts from itself.

**Behavior changes to know about** (all invisible while DRAM is the only live
medium, all asserted by rewritten tests):

| Situation | Before | After |
|---|---|---|
| Requester holds a replica on a "slower" tier | remote faster tier wins | local replica wins (no RDMA) |
| Node advertises HBM 10G / DRAM 400G | HBM | DRAM (more room) |
| Only SSD has room | unroutable | routed to SSD |
| Same-node affinity, anchor's tier fills | spill to a remote node's faster tier | spill stays on the anchor node, changes tier |

*Gate (revised):* the original gate — "`test_tier_priority_route_get` and
`test_route_put_strategy` pass unmodified" — cannot hold, because those tests
*are* the hardcode. `test_tier_priority_route_get` was replaced by
`test_local_preferring_route_get`, and 6 assertions across the put/get suites
were inverted; each inverted test names the expectation it replaces. Everything
else passes untouched.

### Phase 5 — lock it in *(Rules A and C done in Phase 6; Rule B lint outstanding)*

The original plan was one lint script enforcing three rules. Two of the three
are better enforced by the type system, so the lint shrinks to the one rule
types cannot cheaply express.

**Rule A — only one place may name a concrete backend.** *Types — done.*
Phase 6 deleted `LocalBufferViews()`, the one call that forced `PoolClient` to
hold a `PageBackend*`, so `PoolClient::Init` now builds through
`MakePageBackend() -> unique_ptr<MediumBackend>`. The class stays in its header
because `test_page_backend` drives the allocator bookkeeping through the
`TierConfig` constructor directly; the rule as stated — one place names it — is
satisfied, and that place is the composition root.

The same rule now holds for the transfer layer, which did not exist when this
was written: `PoolClient::Init` is also the only file naming `LocalCopyEngine`,
`MoriIoEngine` or `CompositeTransferEngine`. Reaching mori-io's peer handshake
(`EnsureRemoteEngine`, `CacheRemoteBuffers`, …) needed a `MoriIoEngine*` at
first; those calls became the `PeerDirectory` interface instead, so a second
remote transport implements an interface rather than editing `pool_client.cpp`.

**Rule B — no file under `distributed/` may include `local/tiers/*`.** *Lint.*
The obvious structural fix — split `umbp_common` into `umbp_local` and
`umbp_distributed` so the include cannot resolve — does not work yet:
`page_backend.h` needs `HostMemAllocator`, which lives under `local/`. Cutting
that edge means relocating the allocator to `common/` first, a separate refactor.
Until then this rule stays a lint rule, scoped to `local/tiers/*` (not all of
`local/`), with `distributed/peer/ssd/` exempted as the dormant adapter — its
three `local/tiers/*` includes are the only violations in tree.

**Rule C — no backend may call `TransferEngine::{Plan,Submit,Wait}`.** *Types —
done in Phase 6.* This rule existed only because §4 folded registration into
`TransferEngine`, handing backends an object they could transfer with. That
argument conflated *one class* with *one interface*; keeping one object and
giving it two views turns the rule into a compile error at zero runtime cost:

```cpp
class MemoryRegistrar { /* RegisterMemory, Deregister */ };
class TransferEngine : public MemoryRegistrar { /* + CanHandle, Plan, Submit */ };
```

`MediumBackend::Init` takes a `MemoryRegistrar*`. There is no `Submit` on it to
call, and no accessor that hands back the engine.

So what is left of Phase 5 is: cherry-pick
`tools/umbp_backend_abstraction_lint.py` and its pre-commit hook, scoped to
Rule B. **That is still outstanding** — the tool is not in tree. `MockBackend`
already exists (Phase 2, test-only since Phase 3).

**On the budget table.** The lint carries a per-file table of tolerated
violations that "may only shrink". Prefer starting with none: the only Rule B
violations are in `distributed/peer/ssd/`, which is exempted by directory
anyway. A rule with no exceptions survives; a rule with a table of them drifts,
because the escape hatch becomes the habit.

*Acceptance test:* adding a backend = 1 new file + 1 registry line + 0 call-site
edits. **No linter proves this — adding `SsdBackend` does**, and it is the cheap
test, since it mostly forwards to the `PeerSsdManager` that has been dormant and
under test since Phase 0. Phase 6 removed the blocker this paragraph named (the
local path is no longer host-DRAM-only, and `BufferRef` is how a medium
publishes local endpoints), so the test is now runnable; it has not been run.

Two backend abstractions coexist by design, at different layers:
`MediumBackend` is the *distributed storage* contract (publishes descriptors,
registry-dispatched); `TierBackend` is *local mode's* storage-medium contract
(blocking read/write of bytes). The original coupling came from these two being
tangled inside `PoolClient`, not from both existing.

### Phase 6 — abstract the transfer engine *(done)*

`TransferEngine` and `TransferRef` landed as specified (§4). `MoriIoEngine` is
the first implementation and mori-io itself is not modified;
`CompositeTransferEngine` replaced the Phase 2 registration shim. Moved from
`PoolClient` into the engine: `GroupTransfersByPair` (now `MoriIoEngine::Plan`),
the bounce buffer and its mutex, peer engine registration and desc caching.
`RemoteDramScatterWrite` / `RemoteDramScatterRead` / `GroupPagesByBuffer` were
deleted outright — dead since the batch path landed, and a second byte-moving
path is exactly what §6 forbids.

The local fast path **did** fold in: `LocalPutPages` / `LocalGetPages` /
`LocalCopyBlock` are gone, and `ExecuteLocalPut` / `ExecuteLocalGet` now build
`TransferItem`s against `MediumBackend::BufferRef()` and hand them to the same
planner as everything else. See "the measurement" below.

**Four deviations from §4, each deliberate:**

- **`TransferRef` is a struct of merged handles, not a `std::variant`.**
  Registration fans *out*: the same DRAM buffer is a raw pointer (memcpy-able)
  and an RDMA MR (peer-readable) *at the same time*, and which one is used is a
  property of the (src, dst) PAIR. A variant forces a buffer to be one or the
  other and makes the local path — memcpy between two buffers that are also
  registered — inexpressible. `mori::io::MemoryDesc` already resolves this the
  same way, holding `ipcHandle` and `fabricHandle` side by side.
- **No `FileRef` / `ObjectRef`, no `RegisterFile`.** No engine in tree consumes
  one, and this plan has already refused an abstraction nothing exercises once
  (§3, `BackendProperties`). The same test applies: `SsdBackend` is what needs a
  file endpoint, so `SsdBackend` brings it — a `kind` tag plus a second handle
  set in one header, with `CanHandle` already in place to route it.
- **`Wait` lives on the handle, not on the engine.** `Submit` returns a
  `unique_ptr<TransferHandle>` whose destructor drains. That is not cosmetic:
  the RDMA backend holds raw `TransferStatus*` into the handle's status vector,
  so the drain-on-early-destroy safety net that used to live in
  `~RemoteDramGetInFlight` has to sit with the statuses themselves.
- **The schedulers stayed in `PoolClient`.** §5 put submit-all-then-wait in the
  base class, but the overlap that matters spans *several* `Submit` calls (one
  per peer) with the local reads run in the gap — that is the client's schedule
  over a batch, not one engine's over one call. Same for
  `UMBP_DRAM_{WRITE,READ}_THREADS`, which §5 also listed as folding in: they
  parallelize ACROSS the keys of one batch, and a batch is not a concept the
  engine has. What is in the base class is `Transfer()`, the plan+submit+wait
  convenience for callers with nothing to overlap.

**Two behavior changes, both improvements, both asserted by tests:**

| Situation | Before | After |
|---|---|---|
| Batch mixes registered and unregistered caller buffers | contract violation, batch failed | works; the engine stages the unregistered ones |
| Staged batch needs more than the bounce buffer holds | whole peer batch failed | chunked into pool-sized round trips |

Both fall out of moving staging into the engine. The old code reserved the
whole batch's staging up front and held one mutex from submit to wait, so a
submit-all over several staging peers would deadlock — hence `permit_staging`,
the all-zero-copy-or-all-staging contract, and the two-armed fork in
`ExecuteBatch{Put,Get}Plan`. All of it is gone: a plan that needs the pool
completes *inside* `Submit`, so the lock is never held across a return.
`test_cross_node_smoke`'s `PutStagingOverflowFailsBatchCleanly` asserted the old
behavior and was replaced by `PutStagingLargerThanPoolIsChunkedNotFailed` plus
`PutPageLargerThanStagingPoolFailsBatchCleanly` — the failure that is still a
failure is a single *page* larger than the entire pool, which cannot be chunked.

**Phase 5's two type-enforced rules closed here**, as §5 predicted:

- *Rule C* — `MemoryRegistrar` (register/deregister) and
  `TransferEngine : MemoryRegistrar` (+ `CanHandle`/`Plan`/`Submit`).
  `MediumBackend::Init` takes a `MemoryRegistrar*`, so "a backend must not move
  bytes" is a compile error, not a lint rule.
- *Rule A* — deleting `LocalBufferViews()` removed the last concrete-typed call,
  so `PoolClient::Init` builds through `MakePageBackend() -> unique_ptr<
  MediumBackend>`. `PageBackend`'s class definition stays in its header because
  its unit tests drive the allocator bookkeeping directly; what matters for the
  acceptance test is that the production path names no concrete backend, and it
  no longer does.

`LocalBufferViews()` was replaced on the interface by `BufferRef(buffer_index)`
+ `BufferCount()`. The difference is the point: a `TransferRef` is
medium-agnostic (§2), a raw base pointer is not. Publishing refs is what let
`PoolClient` drop `local_copy_backend_` / `local_buffers_` / `CanCopyLocally`,
and it is why a second medium's local access needs no tier branch in a copy loop.

**Layout.** §3's three components are now three directories, so the boundary is
visible in the tree rather than only in the prose — and so the Phase 5 lint
rules can be scoped by directory, which is what Phase 0 already did for the
dormant SSD adapter:

```
distributed/
  master/ routing/                                     control plane
  transfer/   transfer_engine  mori_io_engine  local_copy_engine  composite_*
  peer/       peer_service.{h,cpp}  batch_resolve_codec.h    (the RPC surface)
    backend/  medium_backend.h  page_backend  mock_backend  peer_page_allocator.h
    ssd/      peer_ssd_manager  ssd_copy_pipeline            (dormant since Phase 0)
```

`transfer/` is a sibling of `peer/`, not a child of it, and the include graph is
why: `peer_service` — the reason `peer/` exists — references the transfer layer
**zero** times, because a peer hands out descriptors and the *initiator* moves
the bytes. `transfer/` depends on nothing but `types.h`; `backend/` and
`pool_client` depend on it. It is the lowest layer, so nesting it inside a
higher-level sibling would invert the layering. `LocalCopyEngine` settles it
from the other direction: a memcpy between two of this node's own buffers has no
peer in it at all.

`backend/` does belong under `peer/`: `PeerServiceServer` dispatches every
Allocate/Commit/Resolve/Evict into `BackendRegistry`, and capacity, eviction and
the heartbeat outbox are all statements about what this node holds *for the
cluster*.

Rule C is a type now, but "no file under `backend/` may name a transfer type
other than `MemoryRegistrar`/`TransferRef`" is a one-line grep against a
directory, and so is Rule B's `local/tiers/*` exemption.

*Gate:* `bench_pool_client_batch_get` / `bench_pool_client_batch_put` unchanged;
see §9.

Only after this phase does Phase 2 option (a), real HBM, become a clean
1-file change.

---

## 6. Decisions

**Branch from `main`, not from `refactor/umbp-storage-backend-api`.**
Most of that branch's remaining work — the `PeerTierBackend` async Submit/Wait
API and the 4-way `BatchGetPlan` partition — exists specifically to preserve
SSD's blocking data-plane shape alongside DRAM's async one. That shape is exactly
what Phase 0 removes. Cherry-pick only the lint tool.

**Backends publish descriptors; the transfer layer decides how to move them.**
This supersedes the earlier "backends must be RDMA-addressable page slots."
The weaker requirement is the correct one: a backend's contract is to produce a
`TransferRef`, not to produce registered memory. A medium whose bytes are not
registerable produces a `FileRef` or `ObjectRef` and the transfer layer picks or
composes an engine. The invariant that must not be broken is that there is one
byte-moving path, chosen by the planner — admitting a second one at the backend
level is what produced the original coupling.

**The transfer engine is abstracted inside UMBP; mori-io is not modified.**
UMBP owns `TransferRef`, so mori-io never needs a file or object endpoint type.
This keeps the blast radius off mori-io's other consumers.

**Staging belongs to the transfer layer.** It is the only layer that can observe
completion. A staging pool in a storage backend must fall back to a TTL, which
is the `PrepareSsdRead` lease under another name.

**Allocation and registration belong to the backend.** Whoever allocates memory
must supply the facts a descriptor cannot recover — hugepage/NUMA policy for
DRAM, device and location type for HBM. Note this is why the caller-buffer
registration at `pool_client.cpp:526` is a *different* bug on a different axis:
same seam, different allocator, so the fix is a location parameter on the public
API rather than a move into a backend.

**The `local/` standalone stack is kept, not folded in.**
It has its own `TierBackend`, is reachable through `UMBPClient`'s standalone mode
and live in the Python bindings, and it is the only remaining consumer of the SSD
storage stack after Phase 0. Phase 5 fences it off with lint rather than
deleting it.

**SSD is unwired from distributed mode, not deleted from it.**
Carrying SSD live through the refactor was estimated at 2.5 weeks for Phases 2–3
alone, because every phase would have to design around the blocking lease shape.
Since neither the storage stack nor `PeerSsdManager` leaves the tree, and both
stay under test, the re-add is an adapter, not a rewrite.

The one thing not preserved is the lease: `SsdReadLease`, `PrepareSsdRead`,
`ReleaseSsdLease`, and the peer-side slot state machine.

**Re-add path (post-Phase 6).** `SsdBackend : MediumBackend` mostly *forwards* to
the still-tested `PeerSsdManager`: `BatchResolve` → `PrepareRead`, `Evict` →
`Evict`, `DrainPendingEvents` / `SnapshotOwnedKeys` → the same methods,
`Capacity` → `Capacity`. The piece the original plan deferred — owning a pool of
registered staging pages — is **no longer the backend's problem**: it publishes a
`FileRef` and the transfer layer supplies the bounce buffer if the chosen engine
needs one. It also brings back `put_eligible=false` (§3): Phase 4 deleted the
router's "SSD is not a direct-put target" rule, so SSD must re-assert it itself
or the router will place direct puts on it. Drivers, SPDK env, allocator, proxy protocol, segment format,
eviction policy, the peer-side ownership map and the copy-on-commit pipeline are
reused as-is.

Concretely, `SsdBackend` is now three things, none of which touch a call site:
add a `kind` tag + file handle set to `TransferRef`; implement
`BufferRef`/`BufferCount` over its extents; register it in `PoolClient::Init`.
The engine that reads those extents (`PosixEngine` / `GdsEngine`) is a fourth
file, and `CompositeTransferEngine::SelectEngine` routes to it with no edit.

**One place still names a concrete type, deliberately: `PoolClient::Init`.**
It constructs `LocalCopyEngine`, `MoriIoEngine`, `CompositeTransferEngine` and
(via `MakePageBackend`) the DRAM backend. That is a composition root, not a
leak — the test the acceptance criterion actually states is that *no other* file
names one, and none does. Everything downstream holds `MediumBackend`,
`TransferEngine`, `MemoryRegistrar`, or `PeerDirectory`.

---

## 7. Effort

| Phase | Work |
|---|---|
| 0 — unwire SSD from distributed | 1–1.5 days *(done)* |
| 1 — interfaces | 0.5 day |
| 2 — DRAM backend **+ ownership move** | 4–5 days |
| 3 — peer + pool client | 2–3 days |
| 4 — routing | 1 day |
| 5 — lint + cleanup | 1 day |
| 6 — transfer engine | 4–6 days |

Phase 2 grew from the original 2–3 days: splitting the tier map is the small
half, and relocating allocation, registration and capacity reaches up into
`PoolClientConfig`, `DistributedClient` and `UMBPConfig`.

Phase 2 assumes option (b) — `MockBackend` as the second registered backend.
Real HBM is a feature on top, and is cleanest after Phase 6.

Not counted: re-adding `SsdBackend`, ~1–1.5 days now that the staging pool moved
to the transfer layer. It is the real acceptance test for Phase 5's
*1 file + 1 registry line* claim.

Keeping SSD alive carries a standing cost worth naming: the SPDK submodule,
`storage/**`, the local SSD suites, and the dormant adapter plus its 3 tests all
stay in build and CI while nothing in distributed mode calls them. Dormant
compiled code also drifts — if `KvEvent` or `types.h` changes under it, someone
must fix a file no one is using. Those 3 tests are what turns "preserved source"
into "preserved working behavior."

---

## 8. Open — shared media and node-scoped routing

Not solved by anything above, and worth knowing before S3 or 3FS is attempted.

The whole routing plane assumes a key lives on a *machine*:

```cpp
struct Location { std::string node_id; uint64_t size; TierType tier; };
```

True for DRAM, HBM and local SSD. Meaningless for S3 or 3FS, where every node
can reach the same objects. Three concrete failures:

- The master records whichever node happened to write an object and routes all
  readers there, creating a hot spot for data any node could have fetched
  directly.
- Per-node `Capacity()` is wrong: the capacity is shared, so N nodes each
  reporting the whole bucket lets the master over-commit it N times.
- Eviction becomes a coordination problem — N nodes believe they own the same
  object.

Fixing this needs `Location` to express "reachable from anywhere" plus a
shared-capacity rule in the master. It is control-plane work, orthogonal to
`MediumBackend` and to §4, and it does not block any phase above.
