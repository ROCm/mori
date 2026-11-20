# RFC: Agent context retention for UMBP

Status: draft for discussion; no runtime or wire change in this PR.

Baseline inspected: `23e73534431fc50637ecd7ed5ffbc7730165741a`.

## Problem and example

An agent can leave a parent context idle while a tool or child runs, then reuse
nearly all of it. Conversely, compaction can make recently read history obsolete.
Recency alone cannot distinguish these cases. A retention hint should preserve
reuse opportunities without preventing reclamation or becoming a data-safety pin.

For example, parent A waits while child B runs. B repeatedly accesses new blocks.
Under pressure, A's older blocks may be preferable to B's just-obsoleted history.
After B ends, its exclusive blocks should lose preference immediately, while
blocks shared with A should retain A's preference.

## Existing mechanisms to extend

- [`IUMBPClient`](../include/umbp/umbp_client.h) already has `BatchTouch`,
  `BatchPrefetch`, and depth-aware puts. Touch changes recency; it does not express
  branch ownership or a context replacement.
- [`MasterEvictStrategy`](../include/umbp/distributed/master/evict_strategy.h)
  supports pluggable master-side victim selection.
- [`PeerPool`](../include/umbp/distributed/pool/peer_pool.h) owns logical placement,
  local eviction, and tier transitions.
- [`MediumBackend`](../include/umbp/distributed/peer/backend/medium_backend.h)
  owns bytes and publishes descriptors. Read holds, leases, migration pins, and
  copy generations remain authoritative for transfer safety.
- [`IMasterMetadataStore`](../include/umbp/distributed/master/master_metadata_store.h)
  abstracts in-memory and Redis state. Retention state must not silently become
  process-local when masters share a metadata store.

This proposal adds semantic retention above these mechanisms. It does not add a
new storage backend or implement an agent scheduler inside MORI.

## Proposed contract

The following names are illustrative, not shipped APIs:

```text
ReplaceContext(owner_epoch, branch_id, revision, context_keys, expires_after)
EndContext(owner_epoch, branch_id, revision)
```

The serving adapter derives `context_keys` from the canonical, namespaced keys
that it already stores. It sends a complete snapshot of the branch's current
context. No prompt text, tool output, or user identifier is required.

1. A branch is identified by deployment/pool scope, an adapter-owner epoch, and
   an opaque branch id. Owner epochs must not be reused after a restart.
2. Revisions increase monotonically within that identity. A duplicate revision
   with the same payload is idempotent. A conflicting payload is rejected.
   Older revisions cannot change state.
3. Replacement removes references to omitted keys and adds references to new
   keys atomically for that branch. A child starts with its own snapshot; ending
   a parent does not remove references belonging to children.
4. `EndContext` is terminal for that identity. Reactivation uses a new branch id.
   Tombstones survive at least the supported retry horizon. After tombstone GC,
   the owner epoch must already be retired or the protocol must reject expired
   messages, so old snapshots cannot resurrect an ended branch.
5. Context expiry is a bounded retention hint, not evidence that execution ended.
   A live adapter renews it; expiry returns the keys to normal policy.
6. Replacing or ending a context never deletes bytes. It only changes eviction
   preference for copies that exist now or arrive while the hint is valid.

The key-set operation needs size and cardinality limits. The first implementation
must reject oversized snapshots explicitly; chunked atomic snapshots require a
separate protocol rather than partial application disguised as success.

## Retention versus transfer protection

These states must remain separate:

- An active read hold, lease, or migration pin protects a particular copy while
  bytes are in use. Retention must never weaken or release it.
- A branch reference is a soft preference for future reuse. Under pressure,
  referenced objects can still be evicted.

The first policy ranks unreferenced candidates before referenced candidates and
uses existing recency order within each class. It records forced eviction of
referenced objects. It makes no prediction about an agent's return time.

References are logical: one branch contributes at most once per key. A derived
reference count is not a count of physical replicas. Extra copies do not gain
permanent protection merely because the logical object has one reference.
Replica-cost or predicted-return ordering is deferred until measured.

## Placement and consistency

The serving adapter is the authority on branch context. A bounded retention
registry supplies advisory ranks to local and master eviction policies. The
backend remains the authority on physical copies and may reject eviction for
existing read-safety reasons.

For master-led deployments, keep branch revision checks and reference updates
behind the shared metadata-store boundary; implement the in-memory and Redis
variants together or reject this mode on unsupported stores. New metadata does
not fabricate `ADD` events or make missing objects resident.

For embedded and standalone-process deployments, apply the same semantics at the
storage owner. Forwarders carry identity but do not maintain independent reference
counts. Missing or stale retention metadata falls back to the existing policy.

This is intentionally not a linearizable promise to keep all referenced bytes.
A delayed hint may lose a reuse opportunity; it must not corrupt data or fail an
otherwise valid put/get. Report rejected or unsupported hints to the adapter.

## Compatibility and resource bounds

- Opt in; clients that send no context events retain current behavior.
- Do not implement retention by periodically calling `BatchTouch`: that loses
  compaction/end semantics and may add work proportional to retained bytes.
- Bound branches, references, snapshot bytes, renewal rate, and tombstones per
  configured pool/owner. Refuse excess metadata without changing storage results.
- Keep ids opaque and scoped by the existing deployment's trust boundary. This
  RFC does not introduce cross-tenant sharing or a new authentication model.
- Expose aggregate counters; do not label metrics with branch ids or raw keys.

## Implementation sequence

1. Add a deterministic registry and tests for replacement, expiry, and end.
2. Add an optional adapter/client transport with explicit capability negotiation;
   define mixed-version rejection before enabling it across processes.
3. Extend master and local eviction inputs with soft reference status, without
   putting policy or byte-moving code in `MediumBackend`.
4. Add one SGLang adapter example and a trace replay; keep the feature off by default.

## Validation and acceptance

Correctness cases: shared parent/child keys; compaction; duplicate and out-of-order
updates; end racing renewal; owner restart; tombstone GC; missing keys; snapshot
limits; all objects referenced; master restart; Redis/in-memory parity; and a
read-protected copy rejected by eviction regardless of its retention class.

Replay a fixed P/D workload containing long tool waits, child bursts, compaction,
and explicit branch completion. Compare current LRU, touch-only, and reference-aware
eviction with identical capacities, inputs, offered load, and warmup. Measure
cache-served versus recomputed tokens, retained bytes, forced referenced evictions,
hint-processing cost, and P95/P99 TTFT in the serving integration. Storage replay
alone cannot measure model TTFT. Publish regressions as well as improvements;
this RFC claims no performance gain.

## Open review questions

- Should the first scope be one storage owner, or must distributed retention ship
  in the first implementation? A local-only pilot must reject global guarantees.
- What branch/key limits and renewal horizon fit real agent traces?
- Should retention expiry use backend-authoritative time in Redis to limit
  cross-master clock skew, while local expiry uses a monotonic clock?
- Which integration owns authoritative compaction and child-end notifications?

## Related work

[Evolving Infra](https://arxiv.org/abs/2611.12304) motivates using agent branch
lifecycle instead of access recency alone. Its
[retention protocol](https://github.com/mit-han-lab/EvolvingInfra/blob/main/docs/kv-retention.md)
is inspiration, not an API compatibility target. This proposal preserves MORI's
backend, pool, transfer, and metadata-store boundaries.
