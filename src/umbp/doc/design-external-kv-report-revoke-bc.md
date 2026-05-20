# UMBP External-KV `report`/`revoke` RPC Restoration — Design

**Status:** Draft v2 — incorporating review (alive-node check, 2-arg
data-plane signatures, outbox-routed `UMBPClient` aliases, narrower
wire-compat claims, reuse of existing metric constants).
**Author:** v2.5 follow-up
**Scope:** Restore the deleted `ReportExternalKvBlocks` /
`RevokeExternalKvBlocks` / `RevokeAllExternalKvBlocksAtTier` RPC surface
on top of the unified `GlobalBlockIndex`, preserving 100% source-level
backward compatibility for existing callers of `UMBPMasterClient` and
`mori.cpp.UMBPClient`. Companion doc to
`design-master-control-plane.md`.

---

## 1. Background

The v2.5 refactor (`refactor(umbp): unify external KV index into
GlobalBlockIndex`, commit `ee357ddd`) collapsed the parallel
`ExternalKvBlockIndex` into `GlobalBlockIndex` via a `LocationOwner`
discriminator (`UMBP_OWNED` vs `EXTERNAL_HICACHE`). As part of that
collapse, three RPCs and the matching client/Python surfaces were
deleted:

| Layer | Deleted symbols |
|---|---|
| proto (`src/umbp/distributed/proto/umbp.proto`) | `ReportExternalKvBlocks{Request,Response}`, `RevokeExternalKvBlocks{Request,Response}`, `RevokeAllExternalKvBlocksAtTier{Request,Response}` and their three service entries |
| `MasterServer` (`src/umbp/distributed/master/master_server.cpp`) | Three RPC handlers |
| `MasterClient` (`src/umbp/distributed/master/master_client.{h,cpp}`) | `ReportExternalKvBlocks`, `RevokeExternalKvBlocks`, `RevokeAllExternalKvBlocksAtTier` |
| Pybind (`src/pybind/pybind_umbp.cpp`) | `UMBPMasterClient.report_external_kv_blocks` / `revoke_external_kv_blocks` / `revoke_all_external_kv_blocks_at_tier` (≈ 40 LOC of `.def(...)` entries) |

These were replaced by the bundle-outbox path on the **data-plane**
`UMBPClient`:

```python
# v2.5 surface (what currently exists)
mori.cpp.UMBPClient.bind_external_hashes(hashes, tier)         # implicit own node_id
mori.cpp.UMBPClient.unbind_external_hashes(hashes, tier)
mori.cpp.UMBPClient.unbind_all_external_hashes_at_tier(tier)
mori.cpp.UMBPClient.flush_external_queue()
```

This broke `UMBPMasterClient`-based code in two ways:

1. The new APIs **only live on `UMBPClient`** (the heavy data-plane
   client that needs a full `DistributedClient` + `PoolClient` setup).
   Lightweight `UMBPMasterClient` callers — exactly the audience the
   docs (`docs/api/umbp.rst`) target — have **no way** to write
   external-KV entries anymore.
2. The signature dropped the explicit `node_id` parameter. The
   pre-v2.5 `UMBPMasterClient` API accepted `node_id` as an argument
   (so a sidecar / control plane could report on behalf of a worker)
   — but the master only honored the call if `node_id` was a
   currently alive client in `ClientRegistry`, silently dropping
   reports for unknown or expired nodes (see
   `client_registry.cpp:49` pre-v2.5, and `docs/api/umbp.rst:217–219`).
   The v2.5 `bind_external_hashes` API ties writes to the calling
   client's own identity — secure but uncallable from a lightweight
   non-data-plane caller.

We have an external consumer requirement to **preserve the original
API exactly**, including the explicit `node_id` parameter and its
semantics. This document specifies how to bring the surface back as a
thin compatibility layer over the unified `GlobalBlockIndex`, without
resurrecting the deleted `ExternalKvBlockIndex` class.

---

## 2. Goals & Non-goals

### Goals

1. **Source-level BC.** Any code written against the pre-v2.5
   `UMBPMasterClient.{report_external_kv_blocks,
   revoke_external_kv_blocks, revoke_all_external_kv_blocks_at_tier}`
   API compiles, runs, and observes the same external `MatchExternalKv`
   outcomes after this change.
2. **Coexistence with v2.5 bundle outbox.** The new
   `bind_external_hashes` / `unbind_external_hashes` /
   `unbind_all_external_hashes_at_tier` / `flush_external_queue` APIs
   keep their batched heartbeat-shipped behavior. Both write paths
   operate on the same `GlobalBlockIndex` under
   `LocationOwner::EXTERNAL_HICACHE`.
3. **Exposed on both clients.** Add aliases on the data-plane
   `mori.cpp.UMBPClient` so callers that already use that client can
   keep the old method names too (the v2.5 namespaces are kept as
   primary; the old names are aliases).
4. **Surface on `MasterClient` (the C++ class)** mirrors what the
   pre-v2.5 code shipped, so C++ users of `MasterClient` directly are
   also unblocked.

### Non-goals

1. **Do not** restore `src/umbp/include/umbp/distributed/master/external_kv_block_index.h`
   or the `ExternalKvBlockIndex` class.
2. **Do not** restore the deleted unit tests
   (`test_client_registry_external_kv.cpp`,
   `test_external_kv_block_index.cpp`) verbatim — they targeted an index
   class that no longer exists. New tests cover the restored RPCs at
   the master/client integration level.
3. **No new wire features.** The restored RPCs send the exact same
   proto shape as before v2.5; no `request_id`, no
   `expected_owner_node`, no `version`, no fan-out. Forward evolution
   is out of scope for this doc.
4. **Do not redesign `MatchExternalKv`.** Its v2.5 shape (one
   `ExternalKvNodeMatch` per node, `hashes_by_tier` per match) stays.

---

## 3. API Surface (final)

### 3.1 `UMBPMasterClient` (lightweight, control-plane)

Methods marked **NEW** are restored under this proposal. Pre-existing
methods are listed for completeness.

| Method | Signature | Notes |
|---|---|---|
| `register_self` | `(tier_capacities: dict[UMBPTierType, tuple[int,int]]) -> None` | unchanged |
| `unregister_self` | `() -> None` | unchanged |
| `is_registered` | `() -> bool` | unchanged |
| `match_external_kv` | `(hashes: list[str], count_as_hit: bool = False) -> list[UMBPExternalKvNodeMatch]` | unchanged |
| `get_external_kv_hit_counts` | `(hashes: list[str]) -> list[UMBPExternalKvHitCountEntry]` | unchanged |
| **NEW** `report_external_kv_blocks` | `(node_id: str, hashes: list[str], tier: UMBPTierType) -> None` | additive per-tier bind. Requires `node_id` to be currently registered and alive in `ClientRegistry`; reports for unknown/expired nodes are **silently ignored** (matches pre-v2.5 behavior; see §5.3). Raises `RuntimeError` on empty `node_id`, empty `hashes`, or RPC failure. |
| **NEW** `revoke_external_kv_blocks` | `(node_id: str, hashes: list[str], tier: UMBPTierType) -> None` | per-tier unbind. **No alive-node check** (revoke is an index delete and was historically allowed on any node_id). No-op for hashes that were never reported at `tier`. Raises `RuntimeError` on empty `node_id`, empty `hashes`, or RPC failure. |
| **NEW** `revoke_all_external_kv_blocks_at_tier` | `(node_id: str, tier: UMBPTierType) -> None` | bulk clear at tier. **No alive-node check.** Raises `RuntimeError` on empty `node_id` or RPC failure. |

### 3.2 `mori.cpp.UMBPClient` (data-plane)

The pre-v2.5 `IUMBPClient` interface only ever exposed **2-arg**
variants of these methods — `node_id` was implicit (the data-plane
client's own registered identity, plumbed by `PoolClient`). The
restored aliases therefore match the **2-arg shape**, not the
`UMBPMasterClient` 3-arg shape. This is what e.g. sglang's
pre-v2.5 `KVEventsSubscriber` called:

```python
self._umbp_client.report_external_kv_blocks(hashes, tier)
```

| Method | Signature | Notes |
|---|---|---|
| `bind_external_hashes` | `(hashes, tier)` | v2.5 primary; batched via heartbeat outbox |
| `unbind_external_hashes` | `(hashes, tier)` | v2.5 primary |
| `unbind_all_external_hashes_at_tier` | `(tier)` | v2.5 primary |
| `flush_external_queue` | `()` | v2.5 primary; force-pushes the outbox in one heartbeat |
| **NEW** `report_external_kv_blocks` | `(hashes, tier)` | **2-arg BC alias.** Internally routes through `BindExternalHashes(hashes, tier) + FlushExternalQueue()` — keeps `external_current_set_` consistent so the next `FULL_SYNC_EXTERNAL_HICACHE` (master gap/restart) includes these entries. Approximates the old per-call synchronous visibility via the explicit flush. |
| **NEW** `revoke_external_kv_blocks` | `(hashes, tier)` | **2-arg BC alias** → `UnbindExternalHashes + FlushExternalQueue`. |
| **NEW** `revoke_all_external_kv_blocks_at_tier` | `(tier)` | **1-arg BC alias** → `UnbindAllExternalHashesAtTier + FlushExternalQueue`. |

> **Design choice — locked in.** The data-plane aliases are **not**
> third-party-node 3-arg methods. Third-party node reporting is the
> exclusive responsibility of `UMBPMasterClient` (§3.1), which has a
> dedicated synchronous RPC path. Adding a 3-arg form on
> `mori.cpp.UMBPClient` would (a) break the pre-v2.5 2-arg signature
> source compatibility and (b) bypass `external_current_set_` and
> therefore lose entries to the next full-sync replay (see §6.2
> "MEDIUM" note).
>
> The v2.5 primary APIs (`bind_external_hashes`, etc.) continue to
> operate without the implicit `flush`; the BC aliases add the flush
> only to approximate pre-v2.5 synchronous visibility for callers that
> relied on it.

### 3.3 Behavior summary table

| API (caller) | Path | Wire shape | `external_current_set_` updated? | Owner field on master | Alive check |
|---|---|---|---|---|---|
| `UMBPClient.bind_external_hashes(hashes, tier)` | data-plane client → `external_pending_events_` → next `Heartbeat` bundle | `KvEvent` in `EventBundle`, ack-retained outbox, seq-numbered | yes | `EXTERNAL_HICACHE`, `node_id = self.config.node_id` | no (always self) |
| `UMBPClient.report_external_kv_blocks(hashes, tier)` *(NEW alias)* | same as bind, **then `FlushExternalQueue()`** | same as bind | yes | same as bind | no (always self) |
| `UMBPClient.unbind_external_hashes(hashes, tier)` | same as bind | `KvEvent{REMOVE}` in bundle | yes | same as bind | no |
| `UMBPClient.revoke_external_kv_blocks(hashes, tier)` *(NEW alias)* | same as unbind, **then flush** | same as unbind | yes | same as bind | no |
| `UMBPClient.unbind_all_external_hashes_at_tier(tier)` | data-plane | `KvEvent{CLEAR_AT_TIER}` in bundle | yes (full clear at tier) | same as bind | no |
| `UMBPClient.revoke_all_external_kv_blocks_at_tier(tier)` *(NEW alias)* | same, **then flush** | same | yes | same as bind | no |
| `UMBPMasterClient.report_external_kv_blocks(node_id, hashes, tier)` *(NEW)* | any client → synchronous `ReportExternalKvBlocks` RPC → `GlobalBlockIndex::ApplyEvents` | Dedicated `ReportExternalKvBlocksRequest` | n/a (master-client has no outbox) | `EXTERNAL_HICACHE`, `node_id = request.node_id` (arbitrary) | **YES** — silently dropped if `node_id` not alive in `ClientRegistry` (§5.3) |
| `UMBPMasterClient.revoke_external_kv_blocks(node_id, hashes, tier)` *(NEW)* | dedicated RPC | dedicated proto | n/a | `node_id = request.node_id` | no |
| `UMBPMasterClient.revoke_all_external_kv_blocks_at_tier(node_id, tier)` *(NEW)* | dedicated RPC | dedicated proto | n/a | `node_id = request.node_id` | no |

All paths converge into `GlobalBlockIndex` under
`LocationOwner::EXTERNAL_HICACHE`, indistinguishable to
`MatchExternalKv`. Observable visibility differences:

- **`UMBPClient` v2.5 primaries** (`bind_external_hashes`, etc.):
  visible after the next heartbeat tick (≤ 5 s by default), or
  immediately via `flush_external_queue()`.
- **`UMBPClient` 2-arg BC aliases**: visible after the implicit
  `flush_external_queue()` returns (one heartbeat RTT).
- **`UMBPMasterClient` 3-arg synchronous RPCs**: visible immediately
  on RPC success.

> **Lifecycle invariant:** Because `UMBPClient`'s BC aliases route
> through the outbox, the entries are tracked in
> `external_current_set_` and therefore survive a master-side gap or
> restart via the `FULL_SYNC_EXTERNAL_HICACHE` snapshot. Entries
> written via `UMBPMasterClient`'s synchronous RPC path are tied to
> the supplied `node_id`'s `ClientRegistry` lifecycle (alive-check on
> write, owner-scoped cleanup on unregister/reaper). The two paths
> never share recovery state.

---

## 4. Wire Protocol

### 4.1 Proto messages (restore verbatim)

Add back to `src/umbp/distributed/proto/umbp.proto`:

```proto
// ---- Restored in v2.5 BC patch ------------------------------------
// Mutation RPCs for external KV index. v2.5 introduced a batched
// bundle path on the heartbeat for the data-plane client; these RPCs
// remain available for direct callers (UMBPMasterClient, ad-hoc
// scripts, tests) that prefer per-call synchronous semantics. Both
// paths populate the same GlobalBlockIndex entries under
// LocationOwner::EXTERNAL_HICACHE.

message ReportExternalKvBlocksRequest {
  string         node_id = 1;
  repeated string hashes = 2;
  TierType       tier    = 3;
}
message ReportExternalKvBlocksResponse {}

message RevokeExternalKvBlocksRequest {
  string         node_id = 1;
  repeated string hashes = 2;
  TierType       tier    = 3;
}
message RevokeExternalKvBlocksResponse {}

message RevokeAllExternalKvBlocksAtTierRequest {
  string   node_id = 1;
  TierType tier    = 2;
}
message RevokeAllExternalKvBlocksAtTierResponse {}
```

Add back to the `UMBPMaster` service block:

```proto
service UMBPMaster {
  // ... existing v2.5 entries ...

  rpc ReportExternalKvBlocks         (ReportExternalKvBlocksRequest)
      returns (ReportExternalKvBlocksResponse);
  rpc RevokeExternalKvBlocks         (RevokeExternalKvBlocksRequest)
      returns (RevokeExternalKvBlocksResponse);
  rpc RevokeAllExternalKvBlocksAtTier(RevokeAllExternalKvBlocksAtTierRequest)
      returns (RevokeAllExternalKvBlocksAtTierResponse);
}
```

The wire shape is **byte-identical to pre-v2.5**, so older clients
(if any are still in flight) can speak to the new server, and the new
client can speak to a pre-v2.5 server.

### 4.2 Reserved tags

No previously-reserved field numbers are reused. The deleted bundles
path messages remain untouched. (Confirm during PR review by running
`buf breaking` against the pre-v2.5 proto if available; otherwise
manually diff against `git show 2505181e:src/umbp/distributed/proto/umbp.proto`.)

---

## 5. Server-side Implementation

### 5.1 Handlers — shape and shared invariants

Add three handlers to
`src/umbp/distributed/master/master_server.cpp`. Each handler:

1. Validates inputs (`node_id` non-empty, `hashes` non-empty where
   applicable) → `INVALID_ARGUMENT` on failure.
2. For **`ReportExternalKvBlocks` only**: verify
   `registry_.IsClientAlive(request->node_id())`; silently drop with
   a warning + `result="rejected_not_alive"` metric label otherwise
   (full code in §5.3 — this is the **MUST** invariant for Report).
3. Constructs a `std::vector<KvEvent>` (size = `hashes.size()` for
   `Report`/`Revoke`, size = 1 with `Kind::CLEAR_AT_TIER` for
   `RevokeAll`).
4. Calls `global_block_index_.ApplyEvents(req.node_id(), events)`.
   Owner is forced to `LocationOwner::EXTERNAL_HICACHE` per event
   (`ApplyEvents` already accepts the owner field on each `KvEvent`).
5. Increments the canonical Prometheus counters
   (`MORI_UMBP_METRIC_EXT_KV_REPORT_TOTAL` etc — see §5.4).
6. Returns `grpc::Status::OK` (even when alive-check silently
   dropped, to match pre-v2.5 behavior).

The complete `ReportExternalKvBlocks` sketch with the alive check is
in **§5.3**. `RevokeExternalKvBlocks` is identical except: no alive
check, `Kind::REMOVE` instead of `Kind::ADD`,
`MORI_UMBP_METRIC_EXT_KV_REVOKE_TOTAL` instead.
`RevokeAllExternalKvBlocksAtTier` builds a single
`Kind::CLEAR_AT_TIER` event (no alive check, no per-hash iteration).

### 5.2 Concurrency with the bundle outbox

`GlobalBlockIndex::ApplyEvents` already takes
`std::unique_lock<std::shared_mutex>` for the entire batch (see
`global_block_index.cpp:77`). Both the heartbeat-driven bundle apply
and the new direct-RPC apply serialize through the same mutex; no
additional locking is needed. There is no risk of partial visibility
of a batch.

Cross-path ordering is **not** specified — a `report` RPC sent after
a `bind_external_hashes` call on the same client may land in
`GlobalBlockIndex` either before or after the corresponding bundle,
depending on the next heartbeat tick. Callers who need strict
ordering should:

- Stick to a single write path per `(node_id, tier, hash)` cell, **or**
- Call `flush_external_queue()` before issuing the direct RPC, **or**
- Use only the synchronous path.

This caveat will be documented in `docs/api/umbp.rst`.

### 5.3 Node-membership check — Report only, matches pre-v2.5

`ReportExternalKvBlocks` **MUST** verify that `request.node_id` is
currently alive in `ClientRegistry` before writing to
`GlobalBlockIndex`. This restores pre-v2.5 behavior verbatim:

```cpp
// Pre-v2.5: src/umbp/distributed/master/client_registry.cpp:49
void ClientRegistry::RegisterExternalKvBlocks(
    const std::string& node_id,
    const std::vector<std::string>& hashes, TierType tier) {
  if (!IsClientAlive(node_id)) {
    MORI_UMBP_WARN(
        "[Registry] RegisterExternalKvBlocks rejected: node not alive: {}",
        node_id);
    return;
  }
  // ... write to external_kv_index_ ...
}
```

And exactly what the pre-v2.5 docs (`docs/api/umbp.rst:217–219`)
promised:

> `node_id` must already be registered and alive; reports for unknown
> or expired nodes are ignored by the master.

Concretely, `ReportExternalKvBlocks` handler:

```cpp
grpc::Status MasterServer::ReportExternalKvBlocks(
    grpc::ServerContext* /*ctx*/,
    const ::umbp::ReportExternalKvBlocksRequest* request,
    ::umbp::ReportExternalKvBlocksResponse* /*response*/) {
  if (request->node_id().empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "node_id must not be empty");
  }
  if (request->hashes_size() == 0) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "hashes must not be empty");
  }

  if (!registry_.IsClientAlive(request->node_id())) {
    MORI_UMBP_WARN(
        "[Server] ReportExternalKvBlocks rejected: node not alive: {}",
        request->node_id());
    metrics_.Inc(MORI_UMBP_METRIC_EXT_KV_REPORT_TOTAL,
                 {{"node", request->node_id()},
                  {"tier", TierTypeName(tier)},
                  {"result", "rejected_not_alive"}});
    return grpc::Status::OK;  // matches pre-v2.5 "silently ignored"
  }

  const TierType tier = TierTypeFromProto(request->tier());
  std::vector<KvEvent> events;
  events.reserve(request->hashes_size());
  for (const auto& hash : request->hashes()) {
    events.push_back(KvEvent{KvEvent::Kind::ADD, hash, tier,
                             /*size=*/0,
                             LocationOwner::EXTERNAL_HICACHE});
  }
  const size_t mutated =
      global_block_index_.ApplyEvents(request->node_id(), events);
  metrics_.IncBy(MORI_UMBP_METRIC_EXT_KV_REPORT_BLOCKS_TOTAL, mutated,
                 {{"node", request->node_id()},
                  {"tier", TierTypeName(tier)}});
  metrics_.Inc(MORI_UMBP_METRIC_EXT_KV_REPORT_TOTAL,
               {{"node", request->node_id()},
                {"tier", TierTypeName(tier)},
                {"result", "ok"}});
  return grpc::Status::OK;
}
```

**Without this check** a malicious or buggy caller could create
permanent phantom `EXTERNAL_HICACHE` entries for a `node_id` that
never registers — those entries would survive forever (no
unregister/reaper hook ever fires for an unknown node), and
`MatchExternalKv` would route requests to a non-existent worker.

`RevokeExternalKvBlocks` and `RevokeAllExternalKvBlocksAtTier`
**MUST NOT** require alive node — this also matches pre-v2.5
(`ClientRegistry::UnregisterExternalKvBlocks` had no alive check) and
makes sense semantically: revoke is an index delete; if you're
cleaning up after a known-dead node, you should be allowed to. They
do still reject empty `node_id` / empty `hashes` with
`INVALID_ARGUMENT`.

`RevokeAllExternalKvBlocksAtTier` may additionally fast-path through
`GlobalBlockIndex::RemoveLocationsLocked(entries_, node_id, tier,
LocationOwner::EXTERNAL_HICACHE)` rather than constructing a synthetic
`CLEAR_AT_TIER` event; both approaches are observably identical, and
the helper already exists.

### 5.4 Metrics — reuse existing constants

`src/umbp/include/umbp/distributed/master/master_metrics.h` already
defines the canonical names for these counters (they predate v2.5
and survived the refactor):

```cpp
#define MORI_UMBP_METRIC_EXT_KV_REPORT_TOTAL        "mori_umbp_external_kv_report_total"
#define MORI_UMBP_METRIC_EXT_KV_REVOKE_TOTAL        "mori_umbp_external_kv_revoke_total"
#define MORI_UMBP_METRIC_EXT_KV_REPORT_BLOCKS_TOTAL "mori_umbp_external_kv_report_blocks_total"
#define MORI_UMBP_METRIC_EXT_KV_REVOKE_BLOCKS_TOTAL "mori_umbp_external_kv_revoke_blocks_total"
#define MORI_UMBP_METRIC_EXT_KV_MATCH_TOTAL         "mori_umbp_external_kv_match_total"
```

The restored handlers **MUST** increment these existing counters with
the labels operators are already used to (`node`, `tier`, `result`
where applicable). Introducing parallel `mori_umbp_report_external_kv_*`
names would split dashboards / alerts in two and was explicitly
flagged in review.

If we want to **additionally** distinguish bundle-bound writes from
direct-RPC writes (useful for sizing the outbox vs. sync RPC load),
add an extra label `path={bundle,rpc}` on the existing counter rather
than a new metric:

```cpp
metrics_.Inc(MORI_UMBP_METRIC_EXT_KV_REPORT_TOTAL,
             {{"node", node_id},
              {"tier", TierTypeName(tier)},
              {"path", "rpc"},
              {"result", "ok"}});
```

Bundle-apply call sites in `master_server.cpp` (heartbeat handler)
also need to gain the `path="bundle"` label for symmetry — that's a
small parallel change but keeps both write paths in one timeseries.

---

## 6. Client-side Implementation

### 6.1 `MasterClient` (C++)

Add three methods to
`src/umbp/include/umbp/distributed/master/master_client.h` and
`master_client.cpp`:

```cpp
grpc::Status ReportExternalKvBlocks(
    const std::string& node_id,
    const std::vector<std::string>& hashes,
    TierType tier);

grpc::Status RevokeExternalKvBlocks(
    const std::string& node_id,
    const std::vector<std::string>& hashes,
    TierType tier);

grpc::Status RevokeAllExternalKvBlocksAtTier(
    const std::string& node_id,
    TierType tier);
```

Each method:

1. Builds the request proto.
2. Sets a `ClientContext` deadline (`RpcShutdownTimeoutMs()`, like the
   other sync RPCs in this file).
3. Invokes `GetStub(stub_.get())->ReportExternalKvBlocks(...)`.
4. Returns the status.

These methods do **not** touch `external_pending_events_` or
`hb_state_mutex_` — they are completely independent of the bundle
outbox. This is the correctness-critical invariant.

### 6.2 `IUMBPClient` (data-plane interface) — restore the 2-arg shape

The pre-v2.5 `IUMBPClient` interface only ever had **2-arg** variants
of these methods (`node_id` was implicit, plumbed by `PoolClient` from
`config_.master_config.node_id`). Restore the exact same signatures:

```cpp
// src/umbp/include/umbp/umbp_client.h
virtual bool ReportExternalKvBlocks(const std::vector<std::string>& hashes,
                                    TierType tier) = 0;
virtual bool RevokeExternalKvBlocks(const std::vector<std::string>& hashes,
                                    TierType tier) = 0;
virtual bool RevokeAllExternalKvBlocksAtTier(TierType tier) = 0;
```

Note: **return type is `bool`**, not `void`, to match the pre-v2.5
ABI exactly (`PoolClient` returned the success bit of the underlying
RPC). The aliases return `true` on success.

Implementations route through the **bundle outbox + flush**, not a
direct synchronous RPC:

```cpp
// src/umbp/distributed/distributed_client.cpp
bool DistributedClient::ReportExternalKvBlocks(
    const std::vector<std::string>& hashes, TierType tier) {
  if (!pool_client_) return false;
  return pool_client_->ReportExternalKvBlocks(hashes, tier);
}

// src/umbp/distributed/pool_client.cpp
bool PoolClient::ReportExternalKvBlocks(
    const std::vector<std::string>& hashes, TierType tier) {
  if (!initialized_) return false;
  if (!master_client_->BindExternalHashes(hashes, tier)) return false;
  return master_client_->FlushExternalQueue();   // approximates pre-v2.5
                                                 // per-call visibility
}

bool PoolClient::RevokeExternalKvBlocks(
    const std::vector<std::string>& hashes, TierType tier) {
  if (!initialized_) return false;
  if (!master_client_->UnbindExternalHashes(hashes, tier)) return false;
  return master_client_->FlushExternalQueue();
}

bool PoolClient::RevokeAllExternalKvBlocksAtTier(TierType tier) {
  if (!initialized_) return false;
  if (!master_client_->UnbindAllExternalHashesAtTier(tier)) return false;
  return master_client_->FlushExternalQueue();
}
```

**Why outbox-routed (vs. direct sync RPC)** — MEDIUM finding in the
review:

> If `UMBPClient`'s restored aliases wrote directly to
> `GlobalBlockIndex` via the sync `ReportExternalKvBlocks` RPC,
> the entries would land in `GlobalBlockIndex` but would **NOT** be
> tracked in the client's local `external_current_set_`. On the next
> `FULL_SYNC_EXTERNAL_HICACHE` (triggered by master gap / restart),
> the client would ship a snapshot built from `external_current_set_`
> only — missing the sync-RPC entries — and the master would
> owner-scoped-clear and replay the (incomplete) snapshot, **silently
> deleting the sync-RPC entries**.

Routing through `BindExternalHashes` / `UnbindExternalHashes` keeps
`external_current_set_` consistent, so the next full-sync correctly
includes these entries. The trailing `FlushExternalQueue()` preserves
the pre-v2.5 "visible-immediately-on-return" guarantee (or as close
as a single heartbeat RTT permits).

`StandaloneClient` adds no-op overrides returning `true`, matching
the existing `BindExternalHashes` stub pattern.

### 6.3 Pybind layer

Two changes to `src/pybind/pybind_umbp.cpp`:

**On `UMBPMasterClient` class:**

```cpp
.def(
    "report_external_kv_blocks",
    [](MasterClient& self, const std::string& node_id,
       const std::vector<std::string>& hashes, TierType tier) {
      auto status = self.ReportExternalKvBlocks(node_id, hashes, tier);
      if (!status.ok())
        throw std::runtime_error("ReportExternalKvBlocks failed: " +
                                 status.error_message());
    },
    py::arg("node_id"), py::arg("hashes"), py::arg("tier"),
    py::call_guard<py::gil_scoped_release>())
// ... and the two revoke methods mirroring this shape ...
```

**On `UMBPClient` data-plane class** (2-arg BC aliases — match
pre-v2.5 `IUMBPClient` signature exactly):

```cpp
.def("report_external_kv_blocks", &IUMBPClient::ReportExternalKvBlocks,
     py::arg("hashes"), py::arg("tier"),
     py::call_guard<py::gil_scoped_release>())
.def("revoke_external_kv_blocks", &IUMBPClient::RevokeExternalKvBlocks,
     py::arg("hashes"), py::arg("tier"),
     py::call_guard<py::gil_scoped_release>())
.def("revoke_all_external_kv_blocks_at_tier",
     &IUMBPClient::RevokeAllExternalKvBlocksAtTier,
     py::arg("tier"),
     py::call_guard<py::gil_scoped_release>())
```

The original v2.5 `bind_external_hashes` / `unbind_external_hashes` /
`unbind_all_external_hashes_at_tier` / `flush_external_queue` entries
**stay as-is** — they are the primary, lower-level surface; the
restored 2-arg methods are the BC layer that calls them with an
implicit flush.

> Important: do **not** add a 3-arg `(node_id, hashes, tier)` overload
> on `UMBPClient`. Third-party-node reports are
> `UMBPMasterClient`'s job (§3.1). Adding it here would (a) break
> pre-v2.5 2-arg signature compatibility for callers like sglang's
> old `KVEventsSubscriber` and (b) bypass `external_current_set_`
> (MEDIUM finding).

---

## 7. Compatibility Matrix

| Caller wrote against | Works after this patch? | Notes |
|---|---|---|
| Pre-v2.5 `UMBPMasterClient.report_external_kv_blocks(node_id, hashes, tier)` | ✅ identical behavior, **including alive-node check** | Reports for unregistered/dead `node_id` are silently dropped, as before |
| Pre-v2.5 `UMBPMasterClient.revoke_external_kv_blocks(node_id, hashes, tier)` | ✅ identical behavior, no alive check | |
| Pre-v2.5 `UMBPMasterClient.revoke_all_external_kv_blocks_at_tier(node_id, tier)` | ✅ identical behavior | |
| Pre-v2.5 `mori.cpp.UMBPClient.report_external_kv_blocks(hashes, tier)` *(2-arg)* | ✅ identical signature, slightly different mechanism (now goes via outbox+flush; full-sync survives) | The pre-v2.5 path was sync-RPC; the new path is outbox-routed. Both end up in the same `GlobalBlockIndex` entries. Sglang's old `KVEventsSubscriber.report_external_kv_blocks(...)` call sites continue to compile and run |
| Pre-v2.5 `mori.cpp.UMBPClient.revoke_external_kv_blocks(hashes, tier)` *(2-arg)* | ✅ same | |
| Pre-v2.5 `mori.cpp.UMBPClient.revoke_all_external_kv_blocks_at_tier(tier)` *(1-arg)* | ✅ same | |
| v2.5 `mori.cpp.UMBPClient.bind_external_hashes(hashes, tier)` | ✅ unchanged | |
| v2.5 `mori.cpp.UMBPClient.flush_external_queue()` | ✅ unchanged | |
| **Restored RPCs across master versions** (pre-v2.5 master ↔ post-patch `UMBPMasterClient`, or vice versa) | ✅ wire-compatible for `ReportExternalKvBlocks` / `RevokeExternalKvBlocks` / `RevokeAllExternalKvBlocksAtTier` / `MatchExternalKv` | These are the only RPCs that survived unchanged on the wire |
| **Full `UMBPClient` data-plane** (heartbeats, full-sync, peer transport) across master versions | ❌ **not** wire-compatible across v2.5 boundary | v2.5 changed `Heartbeat` to use `bundles` + `full_sync_scope` (was `seq` + `events` + `is_full_sync`). Mixed data-plane deployments are unsupported |
| Mixed usage on the **same client** (same `(tier, hash)` written by both `UMBPClient.bind_external_hashes` and `UMBPClient.report_external_kv_blocks`) | ✅ consistent — both paths share `external_current_set_` | The BC alias adds a flush, so a subsequent `bind`+heartbeat naturally extends the same set |
| Mixed usage **across clients** (`UMBPMasterClient` reports for `node-A`, then `node-A`'s own `UMBPClient` calls `bind_external_hashes`) | ⚠️ ordering between the two write paths is wall-clock-arrival-order on the master | Recommend: let each `node_id` be owned by exactly one client (either its own `UMBPClient` or an external `UMBPMasterClient`), never both at the same time |

---

## 8. Test Plan

### 8.1 `UMBPMasterClient` 3-arg path (new RPCs)

New tests under `tests/python/test_umbp_master_client.py`:

1. **`test_report_external_kv_blocks_round_trip`** — register
   `node-a`, report 3 hashes on DRAM via
   `UMBPMasterClient.report_external_kv_blocks("node-a", ...)`,
   query via `match_external_kv`, assert `node-a`'s DRAM bucket
   contains the 3 hashes.
2. **`test_revoke_external_kv_blocks_single_tier`** — report on
   HBM+DRAM, revoke HBM only, assert DRAM bucket survives.
3. **`test_revoke_all_external_kv_blocks_at_tier_bulk`** — report
   many hashes on SSD, call bulk revoke, assert SSD bucket empty.
4. **`test_report_external_kv_blocks_for_unregistered_node_is_ignored`**
   — **without** registering `node-ghost`, from any
   `UMBPMasterClient`, call
   `report_external_kv_blocks("node-ghost", ["h1"], DRAM)` and assert
   the call returns OK (no exception) but `match_external_kv(["h1"])`
   returns **no match for `node-ghost`** — pinning the alive-node
   check in §5.3.
5. **`test_report_external_kv_blocks_for_dead_node_after_reaper_ignored`**
   — register `node-a`, let the reaper expire it
   (or `unregister_self()`), then call
   `report_external_kv_blocks("node-a", ...)`, assert ignored.
6. **`test_revoke_external_kv_blocks_for_unknown_node_is_noop`** —
   call revoke for a `node_id` that was never registered, assert
   `RuntimeError` is NOT raised and `match_external_kv` reflects no
   change (locks in "no alive check on revoke" — matches pre-v2.5).
7. **`test_report_external_kv_blocks_empty_node_id_raises`** —
   `node_id=""` ⇒ `RuntimeError`.
8. **`test_report_external_kv_blocks_empty_hashes_raises`** —
   `hashes=[]` ⇒ `RuntimeError`.

### 8.2 `UMBPClient` 2-arg BC aliases — outbox + full-sync survival

This is the **MEDIUM** finding: the BC alias on `UMBPClient` must
route through the outbox so that subsequent full-sync replay does not
delete its writes. We test it directly:

9. **`test_umbpclient_report_external_kv_blocks_two_arg_visible`**
   — start a data-plane `UMBPClient` as `node-data`, call
   `client.report_external_kv_blocks([h1, h2], DRAM)` (note: **no
   `node_id` arg**), then from a separate `UMBPMasterClient` call
   `match_external_kv([h1, h2])` and assert `node-data` shows both
   hashes on DRAM. Validates the implicit flush.
10. **`test_umbpclient_two_arg_alias_survives_external_full_sync`**
    — same as above, then **trigger a master-driven
    `FULL_SYNC_EXTERNAL_HICACHE`** (e.g. by killing & restarting the
    master, or by injecting a seq gap that forces full-sync recovery
    on the next heartbeat), then re-query `match_external_kv` and
    assert the entries are still there. **Pre-fix this test would
    fail** because the writes wouldn't be in `external_current_set_`
    and the replayed snapshot would wipe them out. This is the
    regression-prevention test for the MEDIUM finding.
11. **`test_umbpclient_revoke_external_kv_blocks_two_arg`** —
    symmetric coverage for the revoke alias.

### 8.3 C++ tests

Extend `src/umbp/tests/test_global_block_index_events.cpp` (already
covers `ApplyEvents` under both owners) with:

- A test that drives `MasterClient::ReportExternalKvBlocks` /
  `RevokeExternalKvBlocks` / `RevokeAllExternalKvBlocksAtTier`
  end-to-end through an in-process `MasterServer` / `MasterClient`
  pair, including the alive-node check (a `Report` for an
  unregistered node must leave the index untouched).
- A test that drives `PoolClient::ReportExternalKvBlocks` (the
  2-arg path) and asserts the entries land in **both**
  `external_current_set_` and `GlobalBlockIndex`.

We do **not** restore `test_external_kv_block_index.cpp` (302 LOC)
or `test_client_registry_external_kv.cpp` (172 LOC) — the underlying
class no longer exists.

### 8.4 Coexistence smoke test

One additional Python test, mirrored from `test_v25_match_external_kv_e2e.py`:

1. Starts master.
2. Spins up a data-plane `UMBPClient` (`node-data`) and registers it.
3. Calls `client.bind_external_hashes([h1, h2], DRAM)` and
   `client.flush_external_queue()` (v2.5 primary).
4. Calls `client.report_external_kv_blocks([h3, h4], DRAM)` (BC alias
   — also via outbox + flush, ends up under `node-data`).
5. From a separate `UMBPMasterClient` (no own identity), registers
   `node-rpc` and calls
   `report_external_kv_blocks("node-rpc", [h5, h6], DRAM)` (3-arg
   sync RPC).
6. Queries `match_external_kv([h1, h2, h3, h4, h5, h6])`.
7. Asserts two distinct `ExternalKvNodeMatch` entries:
   - `node-data` with `{DRAM: {h1, h2, h3, h4}}`
   - `node-rpc` with `{DRAM: {h5, h6}}`

This locks in the "both write paths coexist on one
`GlobalBlockIndex`" invariant and the "BC alias entries are
indistinguishable from primary entries" invariant.

### 8.5 Existing-tests regression

`src/umbp/tests/test_global_block_index_events.cpp`,
`tests/python/test_umbp_master_client.py`,
`test_router_dedup`, `test_peer_dram_allocator` — all must continue
to pass without modification.

---

## 9. Files Touched (estimated)

| File | Change | LOC delta |
|---|---|---|
| `src/umbp/distributed/proto/umbp.proto` | restore 3 messages + 3 service entries | +49 |
| `src/umbp/distributed/master/master_server.cpp` | add 3 RPC handlers — Report with alive-node check, Revoke/RevokeAll without; reuse `MORI_UMBP_METRIC_EXT_KV_*` constants | +100 |
| `src/umbp/distributed/master/master_server.h` | declare 3 handlers | +12 |
| `src/umbp/distributed/master/master_client.cpp` | add 3 sync RPC methods (3-arg, for `UMBPMasterClient`) | +60 |
| `src/umbp/include/umbp/distributed/master/master_client.h` | declare 3 methods | +12 |
| `src/umbp/include/umbp/umbp_client.h` | restore 3 **2-arg** pure virtuals on `IUMBPClient` (`bool` return) | +18 |
| `src/umbp/distributed/distributed_client.cpp` | implement 3 methods (delegate to `PoolClient`) | +30 |
| `src/umbp/include/umbp/distributed/distributed_client.h` | declare overrides | +12 |
| `src/umbp/distributed/pool_client.cpp` | implement 3 methods as `BindExternalHashes` / `UnbindExternalHashes` / `UnbindAllExternalHashesAtTier` + `FlushExternalQueue` (preserves `external_current_set_`) | +40 |
| `src/umbp/include/umbp/distributed/pool_client.h` | declare 3 methods | +12 |
| `src/umbp/include/umbp/local/standalone_client.h` | add 3 no-op overrides returning `true` | +12 |
| `src/pybind/pybind_umbp.cpp` | add 3 def's on `UMBPMasterClient` (3-arg, sync RPC) + 3 def's on `UMBPClient` (2-arg, outbox+flush) | +60 |
| `tests/python/test_umbp_master_client.py` | add 8 tests for `UMBPMasterClient` 3-arg path | +160 |
| `tests/python/test_umbp_packaging.py` *(or new file)* | add 3 tests for `UMBPClient` 2-arg aliases incl. full-sync survival | +120 |
| `src/umbp/tests/test_global_block_index_events.cpp` | add 2 round-trip tests (3-arg sync + 2-arg outbox via `PoolClient`) | +60 |
| `docs/api/umbp.rst` | reuse existing examples; add §"Write paths" note + alive-node check note; no example rewrites | +60 / -5 |
| **Total** | | **~817 / ~5** |

LOC budget grew vs. the previous estimate (595) because:

- `PoolClient` row was missing from the earlier estimate (the 2-arg
  `IUMBPClient` plumbing must go through here, matching pre-v2.5)
- An extra test file for the `UMBPClient` 2-arg aliases is needed —
  in particular the full-sync-survival test that prevents the MEDIUM
  finding from regressing
- The C++ test gained a second case to cover the `PoolClient` path

Implementation effort: still **2–3 hours** including build + tests,
because the additional code is highly mechanical (mirror existing
patterns).

---

## 10. Doc Update Plan (`docs/api/umbp.rst`)

The existing pre-v2.5 `UMBPMasterClient` examples all use the 3-arg
form (`node_a.report_external_kv_blocks("node-a", hashes, tier)`),
which is exactly what we restore. **All six existing usage examples
work as-is.** The only required edits:

1. **Intro paragraph (line 4–8)** — leave as-is; the original
   "register a node, report/revoke externally-managed KV-cache blocks,
   and query which nodes hold a given set of blocks" sentence becomes
   accurate again.

2. **Methods table (line 212–228)** — restore the three method rows
   with their pre-v2.5 wording (the existing prose at lines 217–219
   already documents the alive-node check correctly, so no rewrite
   needed). Append a single sentence per method pointing at the v2.5
   primary:

   > Long-lived data-plane callers should prefer the v2.5
   > `bind_external_hashes` API on `mori.cpp.UMBPClient` — it batches
   > through the heartbeat bundle outbox and survives transient gRPC
   > failures via the ack-retained replay queue.

3. **New subsection** between "Cache Hit Tracking" and
   "UMBPExternalKvHitCountEntry" titled **"Write paths: which
   client, which API"**:

   > UMBP exposes external-KV placement writes through two distinct
   > clients with subtly different semantics:
   >
   > **`UMBPMasterClient` (3-arg, synchronous RPC).**
   > `report_external_kv_blocks(node_id, hashes, tier)` and friends
   > are synchronous gRPC calls. The supplied `node_id` is honored
   > verbatim — useful when an external scheduler or sidecar wants to
   > report on behalf of a worker. **Report** requires the supplied
   > `node_id` to be currently registered and alive in the master's
   > `ClientRegistry`; otherwise the call is silently dropped (matches
   > the lifecycle contract — a phantom entry for a never-registered
   > node can never be cleaned up). **Revoke** has no alive-node check.
   >
   > **`mori.cpp.UMBPClient` (data-plane).** Two APIs:
   >
   > - `bind_external_hashes(hashes, tier)` /
   >   `unbind_external_hashes(hashes, tier)` /
   >   `unbind_all_external_hashes_at_tier(tier)` — non-blocking
   >   writes that accumulate in a per-client outbox and ship in the
   >   next heartbeat bundle (default tick ≤ 5 s, or immediate via
   >   `flush_external_queue()`). The outbox is ack-retained so
   >   transient heartbeat failures are replayed. `node_id` is always
   >   implicit (the client's own).
   > - `report_external_kv_blocks(hashes, tier)` /
   >   `revoke_external_kv_blocks(hashes, tier)` /
   >   `revoke_all_external_kv_blocks_at_tier(tier)` — pre-v2.5
   >   2-arg BC aliases. Internally these route through the bundle
   >   outbox **plus** an immediate `flush_external_queue()`, so
   >   visibility is similar to the synchronous RPC path while still
   >   feeding the client's local `external_current_set_` (and
   >   therefore surviving any subsequent master-initiated full-sync).
   >
   > Both clients write to the same `GlobalBlockIndex` and are
   > indistinguishable to `match_external_kv`. Recommendation: let
   > each `node_id` be claimed by exactly one writer (either its own
   > `mori.cpp.UMBPClient` or an external `UMBPMasterClient`), never
   > both at the same time — there is no defined ordering between the
   > two write paths across the heartbeat boundary.

---

## 11. Rollout

This is a backward-compatible additive patch. Recommended steps:

1. Land this design doc; iterate review comments.
2. Implement proto + server + client + pybind in one commit
   (`feat(umbp): restore report/revoke external KV RPC surface for
   source-level BC over v2.5 GlobalBlockIndex`).
3. Add tests in a separate commit
   (`test(umbp): cover restored report/revoke external KV RPCs and
   coexistence with bundle outbox`).
4. Update `docs/api/umbp.rst` in a third commit.
5. Optional follow-up: `chore(umbp): add Prometheus dashboards for
   report/revoke counters`.

No proto file rename, no version bump on the gRPC service, no
client-side migration required.

---

## 12. Open Questions

| ID | Question | Default if no answer |
|---|---|---|
| Q1 | Should the restored RPCs reject `request.node_id == ""`? | **Yes, reject with `INVALID_ARGUMENT`** on Report / Revoke / RevokeAll. Pre-v2.5 had no explicit empty check but the alive-node check at `ClientRegistry::RegisterExternalKvBlocks` always returned for `""` anyway (`IsClientAlive("") == false`); making it an explicit `INVALID_ARGUMENT` matches v2.5 hygiene without breaking any real caller |
| Q2 | Should the SGLang `UMBPStore` / `HiRadixCache` integrations be migrated back to the restored RPCs (matching pre-v2.5 plumbing), or left on `bind_external_hashes` (v2.5 plumbing)? | **Leave on `bind_external_hashes`** — the bundle outbox is more efficient under the high event rate sglang produces and the v2.5 implementation has already been verified end-to-end |
| Q3 | Should we keep the deleted `test_external_kv_block_index.cpp` and `test_client_registry_external_kv.cpp` files removed, or restore renamed/trimmed versions targeting the unified index? | **Keep removed** — covered by `test_global_block_index_events.cpp` plus the new RPC round-trip tests |
| Q4 | If a `report_external_kv_blocks` RPC supplies a `node_id` that isn't registered, should we log a warning? | **Yes — `MORI_UMBP_WARN`**, identical message to pre-v2.5: `[Server] ReportExternalKvBlocks rejected: node not alive: <node_id>`. Also bump `MORI_UMBP_METRIC_EXT_KV_REPORT_TOTAL{result="rejected_not_alive"}` |
| Q5 | Should the `MORI_UMBP_METRIC_EXT_KV_REPORT_TOTAL` (etc.) counters add a `path={bundle,rpc}` label so operators can split the two paths in dashboards? | **Yes** — minimal extra cost; covered under §5.4 |
| Q6 | Should `MatchExternalKv` `count_as_hit` increments still happen for hashes that were written via the synchronous RPC path? | **Yes** — both paths write into the same `GlobalBlockIndex`, and the hit-index already keys off the matched hash regardless of which path created it. No special-casing needed |

---

## 13. Out of Scope (explicitly)

- Reviving any signature with a different shape (e.g. adding `expected_owner`, `lease_duration`, fan-out, etc.).
- Reviving `ExternalKvBlockIndex` as a separate class.
- Changing `MatchExternalKv` or `GetExternalKvHitCounts`.
- Changing the heartbeat / bundle / outbox protocol.
- Adding RBAC / capability tokens on the restored RPCs (these were absent pre-v2.5; adding now would itself be a BC break).
