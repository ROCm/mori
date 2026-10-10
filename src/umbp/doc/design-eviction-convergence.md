# UMBP Eviction Convergence — Design

**Goal:** one eviction path. Whoever decides that a key has to go — the master,
or a node that has no master — asks `PeerPool::Evict`, and the bytes are freed
by that medium's single `Evict` implementation. Nothing evicts inline on a put.

**Scope:** `src/umbp/distributed/` (master eviction, peer service, `PeerPool`,
`PageBackend`, `SsdBackend` / `PeerSsdManager`) and the SPDK tier the SSD medium
reaches through the proxy daemon (`src/umbp/local/tiers/spdk_ssd_tier.cpp`).

**The three requirements:**

1. The master's eviction request names the medium it measured as over-full.
2. Each medium has exactly one eviction implementation, and every eviction goes
   through it.
3. On a node with no master, watermark eviction calls the same function the
   master's request does.

---

## 1. Where eviction happened before

Five places freed a committed key. Only the first two went through the
medium's `Evict`, and only the first went through `PeerPool`.

| # | Site | Runs when | How it frees | Problem |
|---|---|---|---|---|
| 1 | Master `EvictionManager` → `EvictKey` RPC | Master timer; a (node, tier) is over the master's watermark | `PeerService::EvictKey` → `PeerPool::Evict(kReclaim)` → `MediumBackend::Evict` | The request carries **keys only**. The master picks victims per (node, tier) but `SelectVictims` returns node → keys and `EvictKeyRequest` is `repeated string keys`, so the tier is lost. The peer frees by the key's placement: on a tier without `on_evict` offload it deletes the key from **every** backend. With copy-mode promotion a key lives in DRAM *and* SSD, so SSD pressure can free DRAM instead, and DRAM pressure can delete the SSD copy too. |
| 2 | Tier transitions (offload / promotion) | After a key is copied | `source->Evict` (drop the moved-from copy), `target->Evict` (undo a failed target) | None — this is the primitive, used as intended. |
| 3 | `PageBackend::MaybeEvictToLowWatermark` | No master only; inline on commit and on a failed allocation | Its own copy of `PageBackend::Evict`'s body | A second implementation. It runs below `PeerPool`, so `on_evict` demotion never happens, the pool's placement goes stale, and `InstrumentedBackend` never sees it. It runs *inside* `PeerPool::BatchCommit`, before the pool's `watermark` offload check, so it frees the tier down to its low watermark first and the offload never fires: cold keys are deleted instead of moved to SSD. It also runs on the put path under the backend's exclusive lock. |
| 4 | `PeerSsdManager::EvictToLowWatermark` | Every mode, master or not; inline after every SSD write, plus evict-then-retry when a write fails | `PeerSsdManager::Evict(key)` (the right function) | Bypasses `PeerPool` and `InstrumentedBackend` (its victims and bytes are counted nowhere that is exported). Runs inside `SsdBackend::BatchCommit`, i.e. inside `PeerPool::BatchCommit` holding the pool lock. |
| 5 | `SpdkSsdTier::EvictLRU` in the SPDK proxy daemon | `ssd_backend=spdk`; inline when an allocation fails | Erases keys from the daemon's own map | Nobody is told. `PeerSsdManager` keeps listing the keys, no REMOVE reaches the master, and the master keeps routing reads to bytes that are gone. Fragmentation can trigger it below `PeerSsdManager`'s own watermark. |

So "local eviction does not call `Evict()`" (row 3) was not the only gap:
there were three triggers, two layers, and a hidden fifth path under SSD.

---

## 2. Target shape

```
 policy: which keys, which medium      routing: which copy, demote or drop     freeing: one per medium
 ───────────────────────────────────   ────────────────────────────────────    ───────────────────────
 master EvictionManager ─EvictKey(key, tier)─┐
                                             ├─▶ PeerPool::Evict(requests) ─┬─▶ MediumBackend::Evict(keys)
 PeerPool local eviction worker ─────────────┘                              │     ├ PageBackend::Evict
   (no master: every medium;                                                │     └ SsdBackend::Evict
    with a master: SSD only)                                                │         └ PeerSsdManager::Evict
                                                                            └─ demote ─▶ TierTransitionExecutor
                                                                                          copy down, then source->Evict
```

Three rules hold after this change:

1. **`MediumBackend::Evict` is the only code that frees a committed key for
   capacity.** Freeing that is *not* eviction stays where it is: aborting or
   reaping an uncommitted slot, discarding a stale or duplicate commit, and
   `ClearLocal`.
2. **`PeerPool::Evict` is the only place that decides what happens to a copy**
   — move it down a tier or drop it. Every eviction policy calls it.
3. **No eviction work on a put path.** A put may wake the eviction worker. It
   never selects victims and never frees.

---

## 3. Requirement 1 — per-tier eviction requests

**Master.** A victim is now a key *and* the medium to free it from:

```cpp
// types.h
struct EvictionVictim {
  std::string key;
  TierType tier = TierType::UNKNOWN;
};
```

- `MasterEvictStrategy::SelectVictims` returns `node_id → vector<EvictionVictim>`,
  taking the tier from the candidate's `Location.tier` — the same tier whose
  budget it charged. A key over budget in two tiers of one node appears twice,
  once per tier, which is now meaningful.
- `EvictKeyDispatcher::DispatchEvictKey` takes victims; `MasterPeerStubPool`
  fills both proto fields below. Chunking to the gRPC limit is unchanged.

`MasterEvictStrategy` is a public, pluggable interface. Its return type changes;
there is no other implementation in tree and the Python bindings do not expose it.

**Wire.** `EvictKeyRequest` gains a field parallel to `keys`:

```proto
message EvictKeyRequest {
  repeated string   keys  = 1;
  // tiers[i] is the medium keys[i] should be freed from. Ignored unless it has
  // exactly as many entries as keys.
  repeated TierType tiers = 2;
}
```

| Master | Peer | Result |
|---|---|---|
| new | new | each key freed from the named medium |
| new | old | old peer ignores `tiers`: previous behavior |
| old | new | `tiers` empty: each key is freed from the medium that currently holds it (§4) |

**Peer.** `PeerServiceServer::EvictKey` turns the request into
`PoolEvictRequest{key, tier}` and calls `PeerPool::Evict`.

---

## 4. `PeerPool::Evict` — the one routing entry point

```cpp
struct PoolEvictRequest {
  std::string key;
  // The medium to free the key from. UNKNOWN: the medium that holds the key's
  // placement (what a request without a tier means).
  TierType tier = TierType::UNKNOWN;
  // A specific backend instead of a medium; the local eviction worker knows
  // exactly which backend is over its watermark. Wins over `tier`.
  uint32_t backend_id = BackendRegistry::kMaxBackends;
};

std::vector<EvictResult> Evict(const std::vector<PoolEvictRequest>& requests, PoolEvictMode mode);
// Unchanged convenience form: every key with no scope.
std::vector<EvictResult> Evict(const std::vector<std::string>& keys, PoolEvictMode mode);
```

**Which copy.** `backend_id` names it. Otherwise `tier` selects the backend of
that medium that holds the key, preferring the placement. With neither, it is
the placement (or, without one, the first owner in read order). No copy there →
0 bytes freed and nothing else happens.

**`kReclaim`, per copy:**

1. The key is mid-transition → 0 bytes freed; the caller retries next round
   (the same contract as a read lease).
2. The copy is a *draining* source (left behind by an offload whose source
   delete hit a read lease) → retry deleting that copy.
3. The copy is the key's placement and its logical tier has an offload target →
   **demote**: copy it down, then `source->Evict`. If the target already holds
   the key (a copy-mode promotion left one there) nothing is copied and only the
   source is freed.
4. Otherwise, or if the demotion failed (no room downstream) → **drop** that copy
   only: `Evict({key})` on that one backend. One exception: if the demotion
   failed because *another* transition started moving the key meanwhile, leave
   it (0 freed). Deleting the source then could land before that transition pins
   it, and the key would be lost instead of moved.
5. If the freed copy was the placement, point the placement at the remaining
   copy, or forget the key.

`bytes_freed` is what the named copy released, so it is now accurate per tier.

**`kDiscard`** is unchanged: delete from every backend, no demotion.

**Decision — demote regardless of trigger.** Before, only an `on_evict` tier
demoted; evicting from a `watermark` tier dropped the key from every backend.
Now `offload_trigger` only says whether the pool *also* moves keys proactively at
the tier's watermark; any eviction of a copy whose tier has a downstream moves it
there. Two reasons:

- The data stays reachable instead of being deleted while a lower tier had room.
- On a masterless node, local eviction and the pool's `watermark` offload now do
  the *same* thing to the same tier. Before, the local round deleted exactly the
  cold keys the offload was about to move (§1 row 3).

The cost is that an `EvictKey` handled for a `watermark` tier now copies bytes,
as `on_evict` tiers already did inside the same handler.

**Locking.** `Evict` holds `lifecycle_mutex_` shared for the whole call, so
`ClearLocal` cannot run underneath it. Demotions release `operation_mutex_` for
the byte copy as before (`RunTransition`), and the drops now also run outside
`operation_mutex_`: an SSD delete is device IO and must not stall every commit
on the node. Placement repair re-checks the backends under the lock afterwards,
so a key re-committed in that window keeps its placement.

---

## 5. Requirement 2 — one eviction implementation per medium

| Medium | The one implementation | Removed |
|---|---|---|
| DRAM / HBM (`PageBackend`) | `PageBackend::Evict` | `MaybeEvictToLowWatermark` and its four call sites (`Allocate`, `BatchAllocate`, `Commit`, `BatchCommit`), the round mutex and warning latch |
| SSD (`SsdBackend` → `PeerSsdManager`) | `PeerSsdManager::Evict(key)`, reached only through `SsdBackend::Evict` | `EvictToLowWatermark`, the check after every `Write` / `WriteBatch`, the evict-then-retry on a failed write, the manager's own watermarks and `eviction_mu_` |
| SSD via SPDK proxy (`SpdkSsdTier` in the daemon) | `SpdkSsdTier::Evict(key)`, driven by `PeerSsdManager` over the proxy | `EvictLRU` and the evict-on-allocation-failure retry: a full or fragmented device now fails the write. The daemon's only client is `PeerSsdManager`, which evicts through the pool. |

`MediumBackend::EnableLocalEviction` is removed. In its place the medium offers
**ordering, not eviction**:

```cpp
// types.h: a key, and the capacity freeing it gives back (page-rounded where
// the medium allocates in pages).
struct EvictionOffer { std::string key; uint64_t bytes; };

// Start keeping the order EvictionCandidates() reads. Off by default: a
// master-only node never asks for candidates.
virtual void TrackEvictionOrder() {}
// Up to max_count keys this medium could free right now, coldest first,
// skipping keys under a read lease, a pin, or an in-flight read. Frees nothing;
// Evict() re-checks every key.
virtual std::vector<EvictionOffer> EvictionCandidates(size_t max_count) { return {}; }
```

`PageBackend` keeps its LRU (touched by commit and resolve) behind
`TrackEvictionOrder`; `PeerSsdManager` already keeps one unconditionally.

**Why an offer carries its size.** The worker asks for a batch of 64 but takes
only as many offers as the remaining budget needs. Without sizes it would evict
the whole batch before re-reading usage, and on a small or nearly-drained medium
one batch overshoots the low watermark by far -- the old in-medium rounds avoided
that by summing sizes while selecting, and so must this.

**Why the order comes from the medium, not the pool.** `PeerPool` has an LRU of
its own, `access_order_`, but maintains it only for `watermark` offload, and
turning it on disables `BatchResolve`'s single-backend fast path — which is the
masterless 100%-hit restore path. The medium already sees every read, including
that fast path.

---

## 6. Requirement 3 — masterless watermark eviction through the same call

```cpp
// PeerPool
bool EnableLocalEviction(uint32_t backend_id, double high_watermark, double low_watermark);
```

The first call starts one eviction worker thread. The worker:

- **Wakes** when a commit lands, when an allocation answers NO_SPACE, and when a
  transition finishes; it also polls every 100 ms, and leaves at least 1 ms
  between passes so a stream of commits cannot keep it spinning.
- **Orders** backends lowest tier first (SSD, then DRAM, then HBM), so a
  demotion from an upper tier finds room downstream.
- For each backend at or above its high watermark: repeat
  `EvictionCandidates(64)` → take offers until their bytes cover what is left of
  the budget (usage minus the low watermark, fixed at the start) →
  `PeerPool::Evict({key, backend_id}…, kReclaim)` — **the same call the master's
  request makes** — until usage is at or below the low watermark, the budget is
  covered, the medium has no candidates, or a batch frees nothing (every
  candidate was mid-transition or became busy). The next wake tries again.
  The fixed budget is what stops a medium whose reported usage lags its deletes
  from being drained to empty.

Batches of 64 keep each `Evict` call short, so no single call holds the pool
lock for a whole drain.

**Which backends evict locally.** The policy is unchanged; only the mechanism moved.

| Medium | No master | With a master |
|---|---|---|
| DRAM / HBM | `PoolClientConfig::local_evict_{high,low}_watermark` (from `UMBP_DRAM_{HIGH,LOW}_WM`) | none — the master decides |
| SSD | the SSD config's own watermarks (`UMBP_SSD_{HIGH,LOW}_WM`) | the same, alongside the master, as before |

SSD keeps local eviction with a master because the master's loop is periodic
and SSD previously reclaimed right after each write; removing that would let SSD
fill between master rounds. Both paths emit REMOVE events, so the master's index
stays correct.

---

## 7. Behavior changes

| Situation | Before | After |
|---|---|---|
| Master evicts a key held in DRAM and SSD (copy promotion), DRAM over budget, DRAM tier not `on_evict` | DRAM and SSD copies deleted | DRAM copy demoted (SSD already has it, so only DRAM is freed) |
| Same key, SSD over budget | DRAM copy freed (wrong medium), or both copies deleted | only the SSD copy deleted |
| Master evicts from a `watermark`-trigger tier with an offload target | dropped from every backend | demoted |
| Request from an old master (no tiers) | demote-if-`on_evict` else delete everywhere | free the placement's copy: demote if its tier has a downstream, else drop that copy |
| No master, tier has an offload target | local round deletes cold keys before offload sees the tier full | local eviction demotes them |
| No master, DRAM fills | evicted synchronously on the committing thread | evicted by the background worker; a burst faster than the worker can see NO_SPACE until it catches up |
| SSD write fails for lack of space | evict a round, retry once | the write fails; the worker reclaims SSD at its watermark |
| SPDK daemon allocation fails | daemon silently evicts its own oldest keys | the write fails |

A failed put in a KV cache becomes a later miss, not an error; the master path
already accepts NO_SPACE between its rounds. The headroom above the high
watermark is what absorbs a burst while the worker runs.

---

## 8. Metrics

- `InstrumentedBackend`'s eviction series (`op="evict"`, `status=ok|miss`, and
  bytes) now see **every** eviction, local ones included, because every eviction
  is a call through the interface.
- `PeerPool` adds the counters the backend cannot see:
  `mori_umbp_client_local_evict_total{event=round|key|no_candidate|stalled}` —
  passes that found a backend over its high watermark, keys freed, passes where
  the medium offered no candidate, and passes where a batch freed nothing.
- Removed: the SSD medium's `event="eviction_round"`, which counted the deleted
  inline round.

---

## 9. Tests

- **Master strategy:** victims carry the tier they were charged to; a key over
  budget in two tiers appears once per tier.
- **Master → pool chain:** a dispatched (key, tier) frees only that medium.
- **Peer service:** `EvictKey` with `tiers` scopes each key; without, falls back
  to unscoped.
- **`PeerPool::Evict`:** tier scope, backend scope and unscoped; a copy-promoted
  key keeps the copy that was not named; a `watermark` tier demotes; demotion
  falls back to a drop when the target is full; mid-transition keys are left
  alone.
- **Local eviction worker:** a masterless page backend filled past its high
  watermark drains to its low watermark without any put doing the work; leased
  keys survive; a tier with an offload target demotes instead of dropping; an
  SSD backend drains through the same worker.
- **Media:** `PageBackend` and `PeerSsdManager` no longer evict on their own;
  `EvictionCandidates` is coldest-first and skips busy keys.
- The full `tests/cpp/umbp` suite.

---

## 10. Not done here

- **A DRAM backstop in master mode.** DRAM is still master-only when a master
  exists. The worker supports it; turning it on is a config question
  (an emergency watermark well above the master's) left for a follow-up.
- **One source of watermarks.** There are still four: the master's
  `EvictionConfig`, `local_evict_*` for paged media, the SSD config, and each
  logical tier's watermarks in the policy file.
- **The SPDK change is not compiled by the default dev image**, which has no
  SPDK; it needs a build with SPDK enabled before it ships.
