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

Three specific things, not a diffuse mess.

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

**3. Routing is nearly agnostic already.**
`RoutePutStrategy` and `RouteGetStrategy` only ever see
`ClientRecord.tier_capacities` and `Location.tier`. The single hardcode is two
orderings: `TierPriorityRouteGetStrategy`'s `HBM > DRAM > SSD`, and put's
"HBM then DRAM, SSD is never a direct-put target."

**Key observation:** `PeerDramAllocator` is *shaped* for multiple media — an
internal `map<TierType, PageBitmapAllocator>` plus parallel `tier_descs_` /
`tier_bases_` maps, with a `TierType` parameter threaded through every method —
but it serves exactly **one** live tier. `PoolClient::Init` default-constructs an
empty `hbm_cfg` (`pool_client.cpp:448`, "HBM not currently exposed via
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

## 2. The abstraction

New header: `include/umbp/distributed/peer/medium_backend.h`, ~80 lines.

The universal data-plane currency is the **RDMA-addressable page slot** —
`PageLocation` + `BufferMemoryDescBytes` — which is exactly what DRAM and HBM
already speak. A backend exposes its bytes as registered pages; everything else
in the system moves bytes with the same code path.

```cpp
class MediumBackend {
 public:
  virtual ~MediumBackend() = default;

  // ---- identity + routing advertisement ----
  virtual TierType Tier() const = 0;
  virtual const char* Name() const = 0;
  virtual BackendProperties Properties() const = 0;   // {read_rank, put_eligible}

  // ---- control plane (shipped by the heartbeat) ----
  virtual TierCapacity Capacity() const = 0;
  virtual uint64_t OwnedKeyCount() const = 0;
  virtual std::vector<KvEvent> DrainPendingEvents() = 0;
  virtual std::vector<KvEvent> SnapshotOwnedKeys() const = 0;
  virtual std::vector<KvEvent> SnapshotOwnedKeysForFullSync() = 0;
  virtual void ClearLocal() = 0;
  virtual void ClearFullSyncAcked() = 0;

  // ---- data plane (peer-side; driven by PeerServiceServer handlers) ----
  virtual std::vector<AllocateResult> BatchAllocate(const std::vector<AllocateRequest>&) = 0;
  virtual std::vector<CommitResult>   BatchCommit(const std::vector<CommitRequest>&) = 0;
  virtual void                        BatchAbort(const std::vector<uint64_t>& slot_ids) = 0;
  virtual std::vector<ResolvedEntry>  BatchResolve(const std::vector<std::string>& keys) = 0;
  virtual std::vector<EvictResult>    Evict(const std::vector<std::string>& keys) = 0;
};
```

`AllocateResult` and `ResolvedEntry` carry `{pages, page_size, descs}`.

`BackendProperties` is what removes the routing hardcode: read order and
put-eligibility become *advertised* facts rather than a hand-written tier list
in the strategies.

This interface absorbs the existing `OwnedLocationSource` — that type goes away.
The dormant `PeerSsdManager` (Phase 0) keeps its `DrainPendingEvents` /
`SnapshotOwnedKeys` methods but drops the `: public OwnedLocationSource` base and
the `override`s — a ~4-line change. Those methods become part of what the future
`SsdBackend` adapter forwards to `MediumBackend`.

Ownership: a `BackendRegistry` (`map<TierType, unique_ptr<MediumBackend>>`) held
by `PoolClient` is the single place in the tree where a concrete backend type is
named.

---

## 3. Phases

### Phase 0 — unwire SSD from distributed mode

Cut at the adapter seam. Local mode is not touched; no storage-layer code is
deleted.

**Delete — the coupling only:**

- `include/umbp/distributed/ssd_read_lease.h` (76 LOC) and
  `test_ssd_read_lease_gating` — the lease *is* the coupling, and the re-added
  backend will not use it (see the re-add note below)
- The construction and wiring of `PeerSsdManager` / `SsdCopyPipeline` in
  `distributed_client.cpp`, `pool_client.{h,cpp}`, `peer_service.{h,cpp}`, and
  the `AddOwnedLocationSource(ssd_manager)` arm in `master_client.cpp:412`
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

Collapse `BatchGetPlan` from 4 buckets to 2 (`remote_groups`, `local_indices`).

Keep `TierType::SSD = 3` reserved in the enum so wire tags stay stable; delete
every *live* distributed code path that produces or consumes it.

**Keep — dormant but still compiled and still tested:**

- `distributed/peer/peer_ssd_manager.{h,cpp}` (665 LOC) and
  `distributed/peer/ssd_copy_pipeline.{h,cpp}` (~250 LOC). Nothing constructs
  them after this phase; they remain in the build. Move both under
  `distributed/peer/ssd/` so "dormant" is legible in the tree and the Phase 5
  lint exemption can be a directory rather than a filename list.
- `PeerSsdConfig` (`distributed/config.h`) — `peer_ssd_manager.h` needs it. Only
  its construction in `distributed_client.cpp` goes away.
- Their 3 unit tests: `test_peer_ssd_manager`, `test_peer_ssd_eviction`,
  `test_ssd_copy_pipeline`. These test the adapter standalone and stay green.

Two changes are needed to keep them compiling once the coupling is gone:
`PeerSsdManager` drops `: public OwnedLocationSource` and its `override`s (§2),
keeping the methods themselves; and `PrepareRead(key, staging_ptr, staging_cap)`
stays as-is — it is a plain "read this key into this buffer", and only the lease
and slot state machine *around* it in `peer_service` die.

The dependency check that makes this cheap: `peer_ssd_manager.h` pulls in only
`distributed/config.h`, `owned_location_source.h`, and `types.h`; the `.cpp`
pulls in local tiers (kept) and `mori_log`. `ssd_copy_pipeline.h` pulls in only
`types.h`. Neither includes `ssd_read_lease.h`. So nothing dormant depends on
anything being deleted.

**Keep — untouched, still built, still tested:**

- `storage/io/**` — `storage_io_driver`, io_uring and posix drivers
- `storage/spdk/**` — `spdk_env`, `offset_allocator`, `proxy/{spdk_proxy_daemon,
  spdk_proxy_shm,spdk_proxy_protocol}` — and the `3rdparty/spdk` submodule
- `local/tiers/{ssd_tier,spdk_ssd_tier,spdk_proxy_tier,dummy_ssd_tier,
  copy_pipeline,tier_backend,local_storage_manager}.*`, **including
  `LocalStorageManager`'s demote/promote and copy pipeline** — local mode's
  tiering keeps working exactly as it does today
- `UMBPSsdConfig` (`common/config.h`) and its Python binding export
- The local SSD tests: `test_{ssd_tier,spdk_ssd_tier,spdk_proxy,spdk_env,
  dummy_ssd_tier,follower_mode}`, `bench_spdk_raw`

Scale: ~800 lines of coupling excised and ~80 LOC of headers deleted, against
~6.4k LOC of SSD implementation preserved (~5.5k storage stack + ~0.9k dormant
adapter).

*Gate:* ctest green — **including the local SSD suites and the 3 dormant adapter
tests**, which must all still pass unmodified. That is the check that the seam
was cut in the right place.

**Re-add path (post-Phase 5, not in scope here).** SSD returns as
`SsdBackend : MediumBackend`, and because `PeerSsdManager` is still in the tree
and still tested, that adapter mostly *forwards*: `BatchResolve` → `PrepareRead`,
`Evict` → `Evict`, `DrainPendingEvents` / `SnapshotOwnedKeys` → the same methods,
`Capacity` → `Capacity`. One piece of design work is genuinely deferred, not
solved by this plan: §4 requires backends to expose RDMA-addressable page slots,
and SSD bytes are not directly registerable — so `SsdBackend` must own a pool of
registered *staging* pages and hand them out through
`BatchAllocate`/`BatchResolve`, with the read issued during resolve. That is a
better shape than `PrepareSsdRead` + a client-held lease, which is why the lease
and its two RPCs are the one part deleted rather than preserved: the re-added
backend will not use them. Everything else — drivers, SPDK env, allocator, proxy
protocol, segment format, eviction policy, the peer-side ownership map, the
copy-on-commit pipeline — is reused as-is.

### Phase 1 — write the interface

Add `medium_backend.h` as specified in §2. Nothing implements it yet.

*Gate:* tree compiles.

### Phase 2 — DRAM becomes a backend

Split `PeerDramAllocator`'s internal `map<TierType, ...>` into one instance per
medium. Rename to `PageBackend : MediumBackend`. Every method loses its
`TierType` parameter — this makes the file *smaller*.

Consequence to accept deliberately: `owned_`, the read lease, and the clear
generation become per-medium, so a key may exist in both DRAM and HBM. The
design already allows this (see the `NodeMatch` mirror comment in `types.h`).
Peer-local tier selection at `Resolve` time becomes a small policy function
driven by `BackendProperties::read_rank`.

Instantiate once, for DRAM, into the `BackendRegistry` owned by `PoolClient`.

**HBM is not a free second backend.** It does not work today — no HBM buffer is
ever allocated or registered (§1). Standing it up means allocating device memory,
RDMA-registering it, and exposing it through `PoolClientConfig`: new work, not
refactoring. Two options, pick one deliberately:

- **(a) Wire HBM up for real** as the second `PageBackend` instance. This is the
  genuine proof of the abstraction and is presumably wanted regardless — but
  budget it as a feature (~2–3 days on top), not as a refactor gate.
- **(b) Pull `MockBackend` forward from Phase 5** and use it as the second
  registered backend. Cheap, proves the registry and dispatch paths, proves
  nothing about a real second medium.

Default to (b) to keep the refactor honest about its own scope, then do (a)
separately.

*Gate:* DRAM behavior unchanged, with a second backend registered and dispatched
to — whichever of (a) or (b) was chosen.

### Phase 3 — peer server and pool client go agnostic

`PeerServiceServer` takes a `BackendRegistry*` instead of typed pointers;
handlers dispatch on the request's tier tag.

`PoolClient`'s local paths go through `registry_.Get(tier)`. The remote paths are
already tier-tagged on the wire, so once the SSD special case is gone they become
generic with no further work.

*Gate:* `bench_pool_client_batch_get` throughput unchanged. The submit/wait
overlap is a throughput property that no assertion catches — it must be measured,
not asserted.

### Phase 4 — routing plane

Delete both hardcoded tier orders. Derive read order and put-eligibility from
`BackendProperties`, advertised through registration and heartbeat alongside
`tier_capacities`. A backend that cannot accept puts advertises no put-eligible
capacity, which deletes the put tier-order list rather than extending it.

~150 lines.

*Gate:* `test_tier_priority_route_get` and `test_route_put_strategy` pass
unmodified.

### Phase 5 — lock it in

- Cherry-pick `tools/umbp_backend_abstraction_lint.py` and its pre-commit hook
  from `refactor/umbp-storage-backend-api`. It already works: it fails if
  anything outside the backend directory names a concrete backend type or
  includes its header, and its budget table may only shrink.
- Add a ~150-line `MockBackend` for tests.
- Enforce the layering boundary instead of collapsing it. `local/tiers/
  tier_backend.h` and `LocalStorageManager` **stay** — they are local mode's
  medium contract and local mode is still a shipping stack. The lint rule is
  therefore scoped: no file under `distributed/` may include `local/tiers/*`,
  and no file outside a backend directory may name a concrete backend type —
  with `distributed/peer/ssd/` exempted from both as the dormant-and-future
  adapter directory. That single directory exemption is what lets the dormant
  `PeerSsdManager` coexist with the lint from day one, and it becomes the home
  of `ssd_backend.cpp` on re-add rather than a new carve-out.

Two backend abstractions coexist by design, at different layers:
`MediumBackend` is the *distributed data-plane* contract (RDMA page slots,
registry-dispatched); `TierBackend` is *local mode's* storage-medium contract
(blocking read/write of bytes). The original coupling came from these two being
tangled inside `PoolClient`, not from both existing. Keeping them separated by a
lint-enforced seam is what makes the SSD re-add a single adapter file.

*Acceptance test:* adding a backend = 1 new file + 1 registry line + 0 call-site
edits.

---

## 4. Decisions

**Branch from `main`, not from `refactor/umbp-storage-backend-api`.**
Most of that branch's remaining work — the `PeerTierBackend` async Submit/Wait
API (§7.2) and the 4-way `BatchGetPlan` partition (§7.3) — exists specifically to
preserve SSD's blocking data-plane shape alongside DRAM's async one. That shape
is exactly what Phase 0 removes from the distributed path. Cherry-pick only the
lint tool.

**Backends must be RDMA-addressable page slots.**
This is the constraint that makes the design simple. A medium whose own bytes
are not registerable — SSD — participates by exposing registered *staging*
pages, filled during `BatchResolve`. It does not get a second data-plane shape.
Admitting one is what produced the current coupling.

**The `local/` standalone stack is kept, not folded in.**
It has its own `TierBackend`, is reachable through `UMBPClient`'s standalone mode
and live in the Python bindings, and it is the only remaining consumer of the SSD
storage stack after Phase 0. It is unchanged by this refactor; Phase 5 fences it
off with lint rather than deleting it.

**SSD is unwired from distributed mode, not deleted from it.**
The alternative — carrying SSD live through the refactor — was estimated at 2.5
weeks for Phases 2–3 alone, because every phase would have to design around the
blocking lease shape. Unwiring it first and re-adding it through the finished
`MediumBackend` interface is strictly less total work. Since neither the storage
stack nor `PeerSsdManager` leaves the tree, and both stay under test, the re-add
is an adapter, not a rewrite.

The one thing not preserved is the lease: `SsdReadLease`, `PrepareSsdRead`,
`ReleaseSsdLease`, and the peer-side slot state machine. Preserving those would
preserve the exact coupling this refactor exists to remove.

---

## 5. Effort

Roughly 1.5–2 weeks.

| Phase | Work |
|---|---|
| 0 — unwire SSD from distributed | 1–1.5 days |
| 1 — interface | 0.5 day |
| 2 — DRAM backend | 2–3 days |
| 3 — peer + pool client | 2–3 days |
| 4 — routing | 1 day |
| 5 — lint + cleanup | 1 day |

Phase 2 assumes option (b) — `MockBackend` as the second registered backend.
Standing HBM up for real instead adds ~2–3 days and is a feature, not part of
this refactor.

Phase 0 is the highest-leverage step: it removes the constraint that every later
phase would otherwise have to design around, and it is a seam cut rather than a
teardown — no storage code is deleted, so the risk is confined to the distributed
call sites.

Not counted above: re-adding `SsdBackend : MediumBackend` afterwards, ~1.5–2 days
— mostly the staging-page pool and the resolve path, since `PeerSsdManager` is
still present and tested and the adapter largely forwards to it. It is a feature
on top of the finished abstraction, and it is the real acceptance test for §5's
*1 file + 1 registry line* claim.

Keeping SSD alive does carry a standing cost, worth naming rather than hiding:
the SPDK submodule, `storage/**`, the local SSD suites, and the dormant adapter
plus its 3 tests all stay in build and CI while nothing in distributed mode calls
them. Dormant compiled code also drifts — if `KvEvent` or `types.h` changes under
it, someone must fix a file no one is using. Those 3 tests are what keeps that
honest and turns "preserved source" into "preserved working behavior"; that is
the difference between this and simply recovering the files from git history
later, and it is the whole reason to keep rather than delete.
