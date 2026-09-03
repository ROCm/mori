# Ranged I/O on the SSD Medium — Standalone UMBP Server and Distributed Master

Ranged (sub-object) I/O is unavailable on any node whose medium is SSD. One
expression turns it off:

```cpp
// distributed_client.cpp:388
return config_.distributed.has_value() && config_.distributed->medium != UMBPMedium::SSD;
```

`pure-ssd-mode.md` describes the deployment this blocks: **every** node sets
`medium: SSD`, so on that deployment `SupportsRangedIO()` is false cluster-wide,
and the sglang tree connector falls back to whole-object I/O for every request.

**Status: D0 landed** on `feat/umbp-ssd-ranged-io`. §3 is implemented and
tested; D1, D1c, X1 and D2 remain open. §3.1 records one thing the
implementation corrected about the shape of the fix.

This document covers the distributed data plane only — a client talking to a
master, whether that client is a `DistributedClient` in-process or a standalone
UMBP server forwarding for one. Written against `refactor/umbp-backend-agnostic`
at `84adf761`.

**Out of scope: local mode.** `StandaloneClient`, `LocalStorageManager` and the
DRAM tier are not touched. Where this design reaches `SSDTier` /
`ShardedSsdTier` it does so through `PeerSsdManager`, which deliberately owns a
bare `TierBackend` and must not pull in `LocalStorageManager` or
`LocalBlockIndex` (`peer_ssd_manager.h:55`); that is a peer-side dependency of
the distributed path, not local mode leaking in. See §9 for the rest of the
non-goals.

Companion reading: [`design-backend-agnostic-refactor.md`](design-backend-agnostic-refactor.md)
for the `MediumBackend` split, [`design-tree-connector-port.md`](design-tree-connector-port.md)
§10.3/§12 for how ranged I/O entered the distributed data plane, and
[`pure-ssd-mode.md`](pure-ssd-mode.md) for the deployment this work serves.

---

## 1. Why SSD is different, and where it is not

`SsdBackend`'s founding decision is that **the staging arena is the medium**
(`ssd_backend.h`, class comment). SSD bytes are not addressable by the transfer
layer, so a Resolve reads the object off disk into a registered host DRAM page
and publishes *that* page. Everything downstream —
`PoolClient::BuildLocalRangeTransfers`, the RDMA path, `HbmCopyEngine` — sees
ordinary host memory and needs no SSD awareness at all.

The consequence for ranged I/O is the crux of this document:

> Once the object is staged, ranged access **already works** — the ranges are
> just offsets into a published page. What does *not* work is the saving. The
> disk read happened at Resolve time, at full object size, before anyone said
> which bytes they wanted.

So the gate at `distributed_client.cpp:388` is not protecting a correctness
hole. Lifting it yields a *correct* ranged path immediately, with **zero
disk-side savings**. That is exactly the follow-up
`design-tree-connector-port.md` §8 item 4 predicted, and it is why this work
splits into "make it work" (§3) and "make it worth it" (§4, §5).

---

## 2. Invariants this design must not break

1. **A backend does not move bytes.** `MediumBackend::Init` receives a
   `MemoryRegistrar`, not memory, and that is a compile-time fence
   (`medium_backend.h`; §5 Rule C of the refactor doc). Nothing here adds a copy
   method to a backend.
2. **Pages tile the object, in order.** `PageLocation` carries
   `(buffer_index, page_index)` and nothing else (`types.h:149`);
   `BuildLocalRangeTransfers` maps an object offset to a page by division
   (`pool_client.cpp:851`). Any "publish only part of an object" scheme has to
   say what it does about this — §4.1.
3. **A ranged put tiles its object.** `RangesTileObject` is checked before
   routing (`pool_client.cpp:1897`). Ranged puts are a *scatter-gather write of
   a whole new object*, never a partial update of an existing one.
4. **A ranged get is a subset, not a tiling.** `RangesAreDisjointAndInBounds` is
   deliberately weaker than `RangesTileObject` (`pool_client.cpp:170`).
5. **Unsupported is "the wrong door", not a miss.** The pattern `TierBackend`
   established for `ranged_read` (`tier_backend.h:39-43`) — an all-false default
   is indistinguishable from "every key missed", so capability is advertised
   separately — applies to every capability added below.
6. **Per-key results.** One bad key fails that key, never the batch; shape
   errors fail the whole call.
7. **No advertised tier order.** `medium_backend.h` rejected `BackendProperties`
   and read priority outright. Nothing below reintroduces one; §6's `Contains`
   is a side-effect-free existence predicate, which is a different thing.

---

## 3. D0 — lift the gate, and be honest about what it buys

Trace what actually happens with the medium test removed.

**Ranged get, key held on this node.** `PoolClient::BatchGetRanges` Phase 1
resolves the key against each registered backend (`pool_client.cpp:1715`).
`SsdBackend::BatchResolve` stages the whole object into a DRAM page and
publishes it under a read lease (`ssd_backend.cpp:380`). `BuildLocalRangeTransfers`
copies only the requested ranges out of that page. **Correct today.**

**Ranged get, key held remotely.** Phase 2 fetches the whole object into the
ranged scratch arena via `ExecuteBatchGetPlan` and copies the ranges out
(`pool_client.cpp:1758`). The owning peer's `SsdBackend` stages it; the wire
carries the whole object. **Correct today.**

**Ranged put, routed to this node.** `ExecuteLocalPutRangesBatch` allocates a
slot — for `SsdBackend`, a staging page (`ssd_backend.cpp:243`) — writes the
ranges straight into it, and commits, which spills the page to disk
(`ssd_backend.cpp:295`). Because a ranged put *tiles* its object (invariant 3),
every byte of the page is written before the spill. **Correct today, and with no
penalty**: the disk sees exactly the one whole-object write it would have seen
anyway.

**Ranged put, routed remotely.** Assembled in the scratch arena and sent as one
contiguous object. **Correct today.**

So D0 is: delete the medium test and let `SupportsRangedIO()` mean what its name
says — the arena check above it already carries the opt-in.

```diff
-  return config_.distributed.has_value() && config_.distributed->medium != UMBPMedium::SSD;
+  return true;   // ranged_scratch_size_ != 0 was already checked above
```

What D0 buys, and what it does not:

| | before D0 | after D0 |
|---|---|---|
| ranged put to an SSD node | unavailable | full speed — one whole-object disk write, as before |
| ranged get, disk bytes | n/a | still the whole object |
| ranged get, wire bytes | n/a | still the whole object |
| ranged get, host↔device copy bytes | n/a | **only the requested ranges** |

### 3.1 The flag is a declaration, not a guard

Worth stating plainly because it changes how the rest of this document reads,
and the implementation is what made it obvious:
`DistributedClient::BatchGetRanges` and `BatchPutRanges` forward to `PoolClient`
**unconditionally** — neither consults `SupportsRangedIO()`. The flag is
something callers *ask*, and the sglang tree connector is the caller that asks.

So D0 does not unblock a code path. It stops the client misreporting a
capability it always had. Two consequences:

- **The characterization tests had to be written to the conjunction.** A test
  that only drove the data path would pass against the pre-D0 gate and pin
  nothing, because the gate never stood in the data path's way. Every case in
  §10 therefore opens by asserting `SupportsRangedIO()`, so what is pinned is
  "the client says yes, *and* the bytes are right" — and all five fail against
  the old gate.
- **Anything reached by bypassing the flag was already live.** A caller that
  never asked has been issuing correct ranged I/O against SSD nodes all along.
  That is not an argument for having left the gate — the connector *did* ask,
  which is the whole reason the deployment lost the feature.

### 3.2 What D0 buys

The put half is a genuine, complete win. The get half is a correctness
enablement whose only saving is the final copy — worth having on a GPU
destination, where `HbmCopyEngine`'s gather is the expensive part, and worth
nothing on the drive. **Do not ship D0 announced as "SSD ranged reads work"
without D1**; that is how a feature gets benchmarked, found flat, and blamed.

---

## 4. D1 — extent-scoped staging

D1 is where the disk saving comes from: read only the requested extent off disk,
stage only that, publish only that.

### 4.1 The tiling assumption, and `ResolvedEntry::covered_offset`

The obstruction is invariant 2. A published page set is assumed to tile
`[0, object_size)` in order, and `BuildLocalRangeTransfers` finds a byte by
`object_offset / page_size`. A staging page holding object bytes `[a, b)` breaks
that assumption **silently** — every offset would be wrong, and the result is
plausible bytes at the wrong place, the worst failure shape available.

Three ways to close it:

**(a) Add an offset to `PageLocation`.** One `uint64_t object_offset` field, and
`BuildLocalRangeTransfers` intersects ranges against page extents instead of
dividing. *Rejected*: `PageLocation` is on the wire in `AllocateSlotResponse` /
`ResolveKeyResponse` and their batch forms, and in the master's metadata. It is
the most widely shared value type in the system, and every DRAM and HBM path
would carry a field only SSD ever sets.

**(b) A per-resolve covered extent on `ResolvedEntry`.** Add
`uint64_t covered_offset = 0` alongside the existing `pages` / `size` /
`page_size`. The pages still tile — they tile
`[covered_offset, covered_offset + covered_size)` rather than the whole object.
`size` continues to mean the *object's* size, so range validation is unchanged.
`BuildLocalRangeTransfers` subtracts `covered_offset` from each range's
`object_offset` before its page arithmetic, and rejects any range not contained
in the covered window. **Chosen.** It is local to the resolve result (21 call
sites, all in-process at D1), DRAM and HBM leave it at zero and stay
byte-identical, and it states the true thing: a resolve answers *for an extent*,
and whole-object is the degenerate case where the extent is everything.

**(c) Resolve-with-ranges as a distinct `MediumBackend` entry point.**
*Rejected*: it forks the backend interface for one medium, which is exactly the
per-medium branching the refactor removed.

### 4.2 The change, end to end

1. `MediumBackend::BatchResolve` gains a requested-extent argument — one
   `std::optional<ObjectExtent>` per key, defaulted to "everything". DRAM and
   HBM ignore it: their pages already hold the object, they answer
   `covered_offset = 0`, and they pay nothing.
2. `PeerSsdManager` gains `PrepareReadRangesBatch`, delegating to
   `TierBackend::ReadBatchRangesIntoPtr` — see §4.3 for what that requires.
3. `SsdBackend::BatchResolve` computes the **enclosing** extent of the requested
   ranges (one span, `[min offset, max end)`), reads that, stages it, and reports
   `covered_offset`. One span rather than the exact set because a staging page is
   one contiguous allocation; the ranges within it are the caller's problem
   again, exactly as they are for DRAM.
4. `PoolClient::BatchGetRanges` Phase 1 passes the enclosing extent down;
   `BuildLocalRangeTransfers` honours `covered_offset`.

The one-key-one-page limit (`ssd_backend.cpp:258`, `:420`) is *relaxed* by this,
not tightened: an extent is by construction no larger than its object, so a key
whose object exceeds `page_size` becomes servable ranged even though it can never
be served whole. Worth a test; not a goal — in distributed mode master's
`page_size` is the KV block size, so the case is rare.

**Staging-arena pressure improves too.** A resolve currently borrows a full
`page_size` page regardless of how much of it the reader wants; with D1 occupancy
tracks demand. The arena remains sized for read *concurrency* (`ssd_backend.h`),
so this does not change the sizing rule, only the headroom.

### 4.3 The peer-side SSD tier dependency — multi-drive

`PeerSsdManager` constructs its `TierBackend` through `CreateSsdBackend`
(`peer_ssd_manager.cpp:35`): one storage dir gives an `SSDTier`, several
comma-separated dirs give a `ShardedSsdTier` (`ssd_backend_factory.h:33`).
Multi-drive is the production topology — it is the whole point of
`pure-ssd-mode.md` — so D1's `ReadBatchRangesIntoPtr` call lands on
`ShardedSsdTier` in every real deployment.

- `SSDTier::ReadBatchRangesIntoPtr` **exists** (`ssd_tier.cpp:710`, from
  `abf38df7`) and is the reference implementation: it resolves keys under `mu_`,
  plans outside it, merges object-contiguous ranges into runs, issues them as one
  `io_driver_->ReadBatch` with per-run fallback, and reads straight into the
  caller's buffer when a single host range is O_DIRECT-compatible.
- `ShardedSsdTier` **does not forward it**, and `Capabilities()` never sets
  `ranged_read` (`sharded_ssd_tier.cpp:237`). Both are omissions, not decisions.

So D1 requires, on `ShardedSsdTier`:

- Override `ReadBatchRangesIntoPtr`: bucket keys by `ShardOf(key)`, issue **one
  call per shard**, scatter results back — the same shard-and-regroup the class
  already does for `ReadBatchIntoPtr`.
- Fold `ranged_read` into the existing capability intersection: ranged-capable
  only if every shard is.

Do not fall back to a per-key loop. One call per shard is what lets each drive
run its own `io_uring` batch concurrently, which is the aggregate-bandwidth
property the mode exists for.

This is scoped strictly to the peer-side read path. `ShardedSsdTier` is reached
here as `PeerSsdManager`'s bare `TierBackend`; nothing about `LocalStorageManager`
routing, DRAM tiers or `StandaloneClient` is in scope.

### 4.4 Single-flight coalescing must key on the extent, not the key

This is the one genuine correctness hazard D1 introduces, and it is easy to miss.

`PeerSsdManager` coalesces concurrent same-key reads when
`UMBPSsdConfig::single_flight_reads` is on — the default
(`config.h:116`, `UMBP_SSD_SINGLE_FLIGHT`). One requester becomes the leader,
others attach to its `ReadEpisode` as `FollowerTarget{dst, cap}`, and the leader
copies its staging buffer into every follower's destination
(`peer_ssd_manager.h:256-286`).

Today that is sound because every read of a key is a whole-object read, so the
leader's bytes are exactly what each follower wanted. Under D1 they are not: two
requesters can want **different extents** of the same key, and a follower
attached to a leader reading `[0, 4096)` would silently receive those bytes as if
they were its own `[8192, 12288)`.

Three admissible answers; pick per what the workload actually does:

- **Attach only on an identical extent.** Key the episode on
  `(key, covered_offset, covered_size)`. Simplest, and exactly right for the
  workload that motivates single-flight in the first place — MLA keys identical
  across TP ranks, where every rank asks for the same thing.
- **Attach on containment.** A follower whose extent is inside the leader's takes
  a sub-slice of the leader's buffer. Strictly better coverage, one offset
  subtraction more code, and it needs `FollowerTarget` to carry the follower's own
  offset.
- **Do not coalesce ranged reads.** Correct and trivial; gives up the `read_dup`
  saving on the exact access pattern that has the most duplication.

**Containment is the recommendation**, with identical-extent as the fallback if
the episode bookkeeping proves awkward. Whichever is chosen, the wrong answer —
attaching on key alone — produces correct-looking bytes and no error, so this
needs a test that asserts *content*, not success.

---

## 5. D2 — ranged resolve on the peer wire

After D1 a locally-held key reads only its extent off disk. A remotely-held one
still crosses the wire whole, because Phase 2 fetches the object into the scratch
arena and slices it client-side (`pool_client.cpp:1758`).

Closing that means the requested extent must travel to the peer, and
`umbp_peer.proto` has no notion of ranges today. Concretely: `ResolveKeyRequest`
and `BatchResolveKeysRequest` gain an optional extent per key;
`ResolveKeyResponse` and `BatchResolveKeysResponse` gain the matching
`covered_offset`. `peer_service.cpp` passes both through to `BatchResolve`, which
D1 already taught to take them.

The client then also stops needing the scratch arena for a remote ranged *get*:
with the peer publishing an extent-scoped page, the reader can RDMA directly into
its own scattered destinations, which is what `BuildLocalRangeTransfers` does for
the local case. The arena stays required for remote ranged *put* — the object
must be contiguous on the wire.

D2 is deliberately last. It is the only piece with a wire-compatibility story: an
old peer ignores the optional field and answers whole-object, which a new client
must detect via `covered_offset == 0 && covered_size == size` and handle by
slicing — i.e. exactly D0's behaviour, so the fallback is code that already
exists. It is also the only piece that helps DRAM and HBM as much as SSD, so it
can be justified on its own and does not need to wait for an SSD argument.

---

## 6. X1 — resolve-as-probe, which D0 exposes

`PoolClient::BatchGetRanges` Phase 1 finds a key's holder by calling
`BatchResolve` on **every registered backend until one answers**
(`pool_client.cpp:1715`). For DRAM that is a map lookup. For `SsdBackend` a
resolve is a *disk read, a staging-page allocation and a read lease* — so using
it as a probe is expensive, and on a node with more than one medium a probe that
reaches SSD first pays all of that for a key another backend may also hold.

Today a node registers exactly one medium (`PoolClientConfig::medium`), so the
loop degenerates and this is latent. D0 is what makes the SSD case reachable at
all. The fix:

> Give `MediumBackend` a `Contains(keys)` predicate that every backend answers
> without side effects. `SsdBackend` forwards it to `PeerSsdManager::SizeOf`
> (`peer_ssd_manager.h:83`), already an index lookup. Phase 1 probes with
> `Contains`, then resolves once against the winner.

Ordering the registry cheapest-medium-first would merely make the bad case rarer,
and would edge toward the advertised tier order invariant 7 forbids. `Contains`
is three lines per backend and is the honest fix.

Whole-object `BatchGet` has the same structure and gets the same benefit; this is
a shared fix that ranged work happens to force.

---

## 7. The standalone UMBP server

The server has no ranged logic of its own — it forwards to the client it was
configured with, so it inherits D0–D2 whole. Three things must be checked rather
than assumed:

- **Capability propagation.** `supports_ranged_io` rides on the `Ping` response
  (`umbp_standalone.proto:38`, `standalone_server.cpp:265`) and is cached in
  `StandaloneProcessClient` (`standalone_process_client.h:80`). It is read from
  the *server's* client, so a server over a distributed backend on an SSD-medium
  node starts reporting true at D0 with no code change. The flag is cached at
  connect time; a client that connected before the server's capability changed
  keeps the stale answer for its lifetime. Pre-existing and acceptable
  (capabilities do not change under a running server), recorded so it is not
  later mistaken for a bug.
- **The opt-in is an env var here.** The server parses
  `UMBP_DISTRIBUTED_RANGED_SCRATCH_BYTES` into `dist.ranged_scratch_size`
  (`standalone_server_main.cpp:160`), which defaults to zero and therefore
  defaults ranged I/O off. D0 does not change that — a pure-SSD deployment must
  set it, and `pure-ssd-mode.md`'s env list should gain it when D0 lands.
- **The wire format needs nothing.** `BatchRangeDataRequest` flattens ranges
  behind a `range_counts` prefix (`umbp_standalone.proto:92`) and is
  medium-agnostic, through D2 included.

---

## 8. Phasing

| Phase | What | Depends on | Value |
|---|---|---|---|
| **D0** | lift the medium gate (§3) — **LANDED** | — | **ranged put on SSD nodes, at full speed**; ranged get correct, no disk saving |
| **D1a** | `ShardedSsdTier` forwards `ReadBatchRangesIntoPtr`, capability intersected (§4.3) | — | prerequisite; no user-visible change alone |
| **D1b** | extent-scoped staging via `ResolvedEntry::covered_offset` (§4.1, §4.2) | D0, D1a | **ranged get reads only its extent off disk** |
| **D1c** | extent-aware single-flight coalescing (§4.4) | D1b | correctness — must land with D1b, not after |
| **X1** | `MediumBackend::Contains` probe (§6) | — | removes a resolve-as-probe cost D0 exposes |
| **D2** | ranged resolve on the peer wire (§5) | D1b | remote ranged get stops shipping whole objects |

Recommended order: **D1a → D0 → D1b+D1c → X1 → D2.**

D0 went first in practice, because it is two lines and unblocks the deployment
immediately; D1a is inert on its own and unblocks the only phase that delivers
a disk-side saving, so it is next. D1b and D1c ship together — D1b without D1c
is a silent-wrong-bytes bug on the default configuration.

**Landed for D0** (`feat/umbp-ssd-ranged-io`):
`distributed_client.cpp` `SupportsRangedIO()` reduced to the arena check;
`distributed_client.h`'s comment corrected (it justified the gate by saying
`SsdBackend` publishes storage refs — it does not, and never did: `ssd_backend.h`
weighs a file endpoint against staging and picks staging, so the gate rested on
a backend nobody built); `tests/cpp/umbp/distributed/test_ssd_ranged_io.cpp`
added and registered as the `umbp_ssd_ranged_io` integration case.

---

## 9. Non-goals

1. **Local mode.** `StandaloneClient`, `LocalStorageManager` ranged-write
   routing, the DRAM tier and the `UMBPRole::SharedSSDLeader/Follower` shared-SSD
   roles are all out of scope. Ranged writes there remain DRAM-only
   (`local_storage_manager.cpp:721`, `:770`) and the follower role remains
   excluded (`standalone_client.h:78`). §4.3 touches `ShardedSsdTier` only as
   `PeerSsdManager`'s `TierBackend`, and adds nothing local mode consumes.
2. **Read-modify-write.** A ranged put tiles its object (invariant 3). Updating
   part of a stored object in place is not added; the segmented-log layout is
   append-only, so a partial update means a new record plus a rewrite of the
   rest — strictly worse than a whole-object put.
3. **Per-block checksums / record format v4.** A record CRC covers the whole
   value (`segment_format.h:91`), so a partial read has nothing to check itself
   against, and `SSDTier`'s ranged read path verifies nothing. This design
   inherits that narrowing rather than fixing it. Note it costs nothing on the
   deployment in question: `pure-ssd-mode.md` recommends
   `UMBP_SSD_VERIFY_CRC=0`, so nothing was verifying anyway. Per-block checksums
   are the right answer only if ranged reads ever become the default access
   shape rather than an opt-in.
4. **SPDK backends.** `SpdkSsdTier` and `SpdkProxyTier` get no ranged support.
   `SpdkProxyTier` moves bytes through a shared-memory ring with its own
   caching, so a "range" is not a device offset there at all; it would mean a
   `spdk_proxy_protocol.h` change. Both correctly inherit the all-false default
   and, under §4.3's intersection, correctly advertise no ranged capability —
   so they are routed around rather than reporting phantom misses.
5. **A `FileRef` transfer endpoint / GDS path.** `ssd_backend.h` argues at length
   for staging over a file endpoint, and ranged I/O does not weaken that
   argument — if anything D1 strengthens it, since an extent-scoped stage is a
   smaller bounce than a whole-object one. A `FileRef` backend remains the right
   answer for zero-copy and remains a strictly larger change.
6. **Collapsing whole-object I/O into the degenerate ranged case.** Still open
   from `design-tree-connector-port.md` §8 item 2; still not this work.
7. **Making ranged I/O the default.** It stays opt-in behind the scratch arena.
   Below some fetched fraction it wins and above it it loses; nothing here
   changes that trade-off, only where it applies.

---

## 10. Verification

**Unit.** `tests/cpp/umbp/distributed/test_pool_client_ranges.cpp` is the
fixture — it drives `BackendRegistry` / `MediumBackend` directly, so it can host
an `SsdBackend` without a master.

- **D0** — *written, in `test_ssd_ranged_io.cpp`; 5 cases, all passing.*
  `SupportsRangedIO()` true for an SSD medium with an arena and false without;
  local ranged put assembling an object from out-of-order tiling ranges
  (verified by a whole-object read-back, since a ranged read-back would agree
  with a consistently wrong mapping); local ranged get of a non-tiling subset;
  a guard-byte case proving a ranged get writes only what was asked for; and a
  remote round trip through an SSD-only peer. Each drives a real
  `IUMBPClient` and asserts the advertised capability first — see §3.1 for why
  that assertion is load-bearing rather than decorative. All five fail against
  the pre-D0 gate.

  A sixth case covers §7's hop, which the other five do not touch: a standalone
  server holding the `DistributedClient` while a worker forwards to it over
  gRPC, asserting `GetBackendMode() == Distributed`, the capability surviving
  the `Ping`, and a ranged round trip. Writing it surfaced a trap worth
  repeating: the standalone-process data plane is **shared memory**, so caller
  buffers must be shm-backed and registered. An ordinary heap pointer fails
  `ResolveRanges` server-side and returns a per-key `false` — correct, but at
  the call site indistinguishable from the store refusing the write.
- **D1a**: a new `test_sharded_ssd_tier_ranges.cpp` asserting keys land on the
  shard `ShardOf` names, that a batch spanning shards issues **one call per
  shard** (counted via a mock `TierBackend`), and that `ranged_read` is the AND
  over shards — including a mixed fixture where one shard is not ranged-capable.
- **D1b**: assert the **disk read size**, not just the bytes returned. An
  instrumented `TierBackend` counting bytes read is what distinguishes D1b from
  D0; without it the D1b test passes on D0's code. Also: a range outside the
  covered window is rejected; DRAM and HBM report `covered_offset == 0` and are
  byte-identical to before; an object larger than `page_size` is servable ranged.
- **D1c**: two concurrent readers of the same key requesting **disjoint**
  extents, asserting each receives its own bytes. This test must compare
  content — the bug it guards against returns success with wrong data. Run it
  with `UMBP_SSD_SINGLE_FLIGHT=1` explicitly, since that is the default the bug
  lives on.
- **X1**: a two-backend registry asserting `SsdBackend` performs no disk read for
  a key it does not hold.

**Measured, 2026-08-19, n06-21** (`umbp_bench.py ranged_perf`, one node,
`umbp_master` + two standalone servers, object 32x36KiB, batch 32, median of
3-7). Same drive, same object, same access shape; only the read path differs:

| deployment | G=1 (3.1% fetched) | G=32 (100%) |
|---|---|---|
| `umbp-local`, 1 drive (true partial pread) | **6.96-9.83x faster** | 0.63-0.92x |
| `umbp-server` over RDMA (stages whole object) | 0.23-0.30x | 0.27x |
| `umbp-local`, 3 drives sharded | **every ranged read misses** | — |

Three things this settles:

1. **D1 is worth about an order of magnitude at low fetched fractions, and it is
   not a projection.** Local mode already reads only the requested extent off
   the same drive and is ~7-10x faster there; distributed cannot, because the
   resolve stages before the ranges are known. The gap between the two rows IS
   the D1 opportunity, measured.
2. **The distributed cost is per key and does not amortize.** A batch sweep at
   G=1 gives 0.71 / 0.70 / 0.69 ms per key at batch 8 / 32 / 128, while
   whole-object improves with batch (0.31 -> 0.215 -> 0.018 ms/key). Something
   in the ranged path is serialized per key; §6's key-by-key Phase 1 probe is
   the prime suspect, not yet profiled. That makes X1 a prerequisite for D1
   paying off, not an optional cleanup.
3. **D1a is not merely a missing optimisation — multi-drive is broken today.**
   `--ssd-dir a,b,c` misses every ranged read where one drive succeeds on the
   identical config, and `supports_ranged_io()` still returns true on that node
   (`StandaloneClient` only checks the role), so the client advertises a
   capability the sharded tier cannot serve. Fixing the forwarding must also fix
   the advertisement.

Caveat on the distributed whole-object baseline: it is partly lease-warmed. Read
leases keep staged pages in DRAM across passes, and whole-object at batch=128
reported ~62 GB/s, impossible from one NVMe. Ranged's flatness and its per-key
constant are unaffected. The node was also heavily loaded (load ~43), which is
why the ratio, not the absolute, is the reported figure.

**Performance.** Two things to hold onto when reading numbers:

- The existing finding is the baseline to beat: ranged **loses** to whole-object
  above roughly an eighth of the object fetched, and the crossover is what D1
  moves. Report the crossover point, not a single ratio.
- Page cache hides everything. `pure-ssd-mode.md`'s direct-I/O settings plus
  `UMBP_SSD_TIMING=1` (diagnostic only — it prints several `[SsdPerf/*]` lines
  per batch) are how device-side numbers become visible. A run showing no
  `dev_GB_s` movement between D0 and D1 is measuring the page cache, not the
  drive.

**End to end.** The sglang tree connector with `ssd_enabled=true` on a
`pure-ssd-mode.md` deployment, plus `UMBP_DISTRIBUTED_RANGED_SCRATCH_BYTES` set:
before D0 it declines ranged I/O outright; after, it should negotiate ranged and
produce identical output. Equality of generated tokens against a whole-object run
is the acceptance test.

---

## 11. Support matrix, after

| Client | Backend | Ranged get | Ranged put |
|---|---|---|---|
| `DistributedClient` | — | opt-in, all media incl. SSD (D0); extent-scoped disk reads (D1); extent-scoped wire (D2) | opt-in, all media incl. SSD (D0) |
| `StandaloneProcessClient` | Distributed | same, inherited | same, inherited |
| `StandaloneProcessClient` | Local | unchanged — out of scope (§9.1) | unchanged — out of scope (§9.1) |
| `StandaloneClient` | — | unchanged — out of scope (§9.1) | unchanged — out of scope (§9.1) |

`design-tree-connector-port.md` §9's matrix predates `abf38df7` and is stale on
the local-mode rows independently of this work; it should be corrected to point
here for the distributed rows once D0 lands.
