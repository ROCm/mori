# RFC: SGLang KV Indexer

## Summary

This RFC proposes a SGLang KV Indexer for collecting and querying distributed KV cache location metadata. The indexer does not store KV cache bytes. It tracks which worker holds which KV block hash and at which storage tier, so schedulers and routing components can make cross-worker KV reuse decisions.

SGLang workers already emit KV cache lifecycle events such as `BlockStored`, `BlockRemoved`, and `AllBlocksCleared`. A bridge component converts these events into indexer metadata updates. Query clients can then ask the indexer which workers may hold a given set of KV block hashes.

The initial project will be implemented out of tree as an independent Rust service. It integrates with SGLang through the existing KV event stream and gRPC APIs, without requiring the indexer code to live inside the SGLang source tree.

## Goals

1. Collect KV location metadata from SGLang workers.

   The bridge subscribes to SGLang KV events and converts them into metadata operations such as `report`, `revoke`, and `revoke-all`. The indexer maintains a location index in the form of `hash -> worker -> tier`.

2. Provide cross-worker KV location lookup.

   Schedulers or query clients can ask the indexer where a list of KV block hashes may be found. The response includes worker IDs and storage tiers, and can optionally update hit-count statistics for scheduling policies.

3. Converge after common process and network failures.

   Metadata updates are ordered per worker, idempotent under replay, and recoverable after bridge, indexer, Redis, or worker restarts. Dead or partially reset workers must not be returned as routing candidates.

## Architecture

```text
+------------------+        +----------------------------+
|      SGLang      |        |           Bridge           |
|------------------|        |----------------------------|
| KV Cache         | ZMQ    | - subscribe KVEventBatch   |
| Event Publisher  +------->| - decode KV events         |
| BlockStored      | events | - map hash / medium / tier |
| BlockRemoved     |        | - batch report / revoke    |
| AllBlocksCleared |        |                            |
+------------------+        +-------------+--------------+
                                           |
                                           | gRPC metadata API
                                           v
+------------------------+  +-------------+--------------+
|     Metadata Store     |<-|          KV Indexer        |
|------------------------|  |----------------------------|
| hash -> worker         |  | location index             |
| worker -> tier         |  | hit-count index            |
| hit counts             |  | query API                  |
|------------------------|  |                            |
| Redis / Redis Cluster  |  | gRPC health/readiness      |
+------------------------+  +----------------------------+
```

The data flow is metadata-only. The bridge sends KV block location updates to the indexer. The indexer stores and serves location metadata. KV bytes remain owned by SGLang workers and are not transferred through the indexer.

## KV Indexer

The indexer is a standalone metadata service. Its core responsibility is to maintain KV block location state and expose query APIs.

| Function | Description |
| --- | --- |
| Location update | Record that a worker holds a set of KV block hashes at a specific tier. |
| Location revoke | Remove specific hashes, or remove all hashes from a worker at a specific tier. |
| Location lookup | Return workers and tiers that may hold requested KV block hashes. |
| Hit counting | Optionally count real lookup hits for scheduling or monitoring. |
| Store abstraction | Keep persistence behind a narrow backend interface; the initial durable implementation uses Redis or Redis Cluster. |

The primary data model is:

```text
hash -> worker_id -> set<tier>
```

This allows a KV block hash to exist on multiple workers and on multiple tiers of the same worker.

## Bridge

The bridge adapts SGLang KV events into indexer metadata operations. The initial implementation will run out of tree, most likely as a sidecar near each SGLang worker. This avoids changes to the SGLang worker process and keeps the indexer deployable independently while the API stabilizes.

| Function | Description |
| --- | --- |
| Event subscription | Subscribe to SGLang ZMQ `KVEventBatch`. |
| Event decoding | Decode `BlockStored`, `BlockRemoved`, and `AllBlocksCleared`. |
| Tier mapping | Map SGLang storage medium to indexer tier, for example GPU -> HBM, CPU pinned memory -> DRAM, disk -> SSD. |
| Hash normalization | Convert SGLang block hashes into the indexer hash format. |
| Batching | Group updates by action and tier before sending gRPC requests. |
| Observability | Track decode errors, sequence gaps, and report / revoke failures. |

Event mapping:

| SGLang event | Indexer operation |
| --- | --- |
| `BlockStored(block_hashes, medium)` | `ACTION_REPORT(hashes, tier)` in `ApplyExternalKvBatch` |
| `BlockRemoved(block_hashes, medium)` | `ACTION_REVOKE(hashes, tier)` in `ApplyExternalKvBatch` |
| `AllBlocksCleared()` | `ACTION_CLEAR_ALL_AT_TIER(tier)` in `ApplyExternalKvBatch` for configured tiers |

For multi-worker or DP-rank deployments, run one bridge per event source. Each source has its own worker ID, endpoint, topic, and sequence tracking; all bridges share one indexer.

## Fault Tolerance and Consistency Model

### Scope and invariant

The index is best-effort routing metadata, not authoritative model state. A false negative reduces cache reuse; a false positive must be handled by the normal cache-miss fallback and must not corrupt inference. The design therefore favors a small replay-convergent state machine over distributed transactions or consensus.

Correctness requires **one active bridge writer per `worker_id`**. Sequence checks and mutation commit are separate phases, so two concurrent writers for the same worker are not fenced from both starting an update. Different workers are independent and may update concurrently.

### Durable per-worker sequence

Every SGLang event batch carries a monotonically increasing sequence number. Redis stores the durable `last_applied_seq` in the worker metadata hash.

For each non-heartbeat apply:

1. Read the worker's durable sequence.
2. If the request sequence is less than or equal to it, skip the batch and return `duplicate = true`.
3. Apply all actions in their original order. Placement and reverse-index mutations are idempotent.
4. Advance the durable sequence only after all mutations complete.

If the process fails between mutation and sequence commit, the durable sequence remains behind. The bridge retries the same batch; idempotent `HSET`/`HDEL`/`SADD`/`SREM` semantics repair any partial state before the sequence is committed. The system provides replay convergence rather than globally atomic or exactly-once batch execution.

The response returns `last_applied_seq`, allowing a bridge that re-observes an already applied batch to resynchronize its expected sequence with the backend's durable position.

### Bridge reconnect and replay

The bridge retains an unacknowledged batch across gRPC reconnects and sends it again before consuming newer events. It also tracks the next expected SGLang sequence:

- a lower sequence is stale and is ignored;
- an equal sequence is processed;
- a higher sequence is a gap, and the bridge requests the missing range from SGLang's replay endpoint before processing the live batch.

Replay uses the same apply API and sequence gate as live traffic. When replay is not configured or the requested history is unavailable, the current best-effort policy logs the gap and continues with newer events. This can create false negatives and should be observable, but it does not block inference.

### Worker liveness

Durable worker metadata and liveness are stored separately:

- `{worker}:meta` holds address, sequence, incarnation, and reset state and never expires;
- `{worker}:live` is refreshed by every apply and by periodic empty-action heartbeats and carries the liveness TTL.

`MatchExternalKv` excludes workers whose live key has expired. Placement and reverse-index data are retained, so a transiently disconnected worker can become routable again after a heartbeat without rebuilding the entire index. Keeping metadata durable is essential: expiring it would lose the sequence and incarnation needed to reject stale state.

### Worker restart and incarnation reset

Each worker lifetime has an opaque `incarnation` token. On a token change, the indexer must assume the worker's cache was lost:

1. Atomically record the new incarnation and set durable `reset_pending = 1`.
2. While reset is pending, exclude the worker from all match responses.
3. Walk the worker reverse index and remove the worker from every placement entry.
4. Clear the reverse index and old sequence.
5. Clear `reset_pending` only after the reset fully succeeds.

Reset operations are idempotent. If Redis or the indexer fails midway, the next apply or heartbeat observes `reset_pending` and retries. This prevents both stale placement resurrection and routing to a partially reset worker.

The default bridge-generated token identifies the bridge process. Deployments that can restart the bridge independently from the worker should provide a token tied to the actual worker lifetime; a bridge-only token change is safe but causes conservative cache misses.

### Redis failure and partial writes

Redis is the durable coordination point. When it is unavailable, mutation and query RPCs return `UNAVAILABLE`. The standard `grpc.health.v1.Health` service reports `NOT_SERVING` until Redis answers readiness probes again.

Redis Cluster cannot atomically update placement keys in multiple hash slots as one batch. Instead:

- each placement/hit mutation is atomic within its block-hash slot using Lua;
- worker metadata and the worker reverse index share a worker hash slot;
- forward and reverse index updates are individually idempotent;
- replay always reasserts both postconditions, repairing a failure between them.

Requests have protocol-level hash/action limits, and Redis fan-out is processed in bounded chunks so recovery traffic cannot create unbounded concurrent work.

### Failure behavior summary

| Failure | Behavior |
| --- | --- |
| Duplicate or stale batch | Sequence gate skips it and returns the durable position. |
| Bridge-to-indexer disconnect | Pending batch is retained, connection is retried, and the batch is replayed. |
| Bridge process restart | Volatile pending/expected-sequence state is lost. The default new incarnation conservatively wipes that worker's old placements; subsequent events rebuild them. |
| Missing event sequence | Bridge requests the missing range; without replay it logs degradation and continues. |
| Indexer crash | Redis state survives; bridge reconnects and resumes from the durable sequence. |
| Redis outage | Data RPCs fail with `UNAVAILABLE`; health reports `NOT_SERVING`; reconnect is automatic. |
| Partial forward/reverse update | Replaying the idempotent batch repairs the missing postcondition. |
| Worker or bridge stops heartbeating | Liveness TTL expires and the worker is excluded from matches. |
| Worker restarts | Incarnation change fences routing, retries stale-state cleanup, and resets sequence state. |

### Explicit non-goals

The initial design does not provide:

- multi-writer fencing for the same worker ID;
- global atomicity across all hashes in an apply batch;
- consensus or cross-region linearizability;
- exactly-once event delivery;
- a durable bridge checkpoint across bridge process crashes;
- immediate garbage collection for workers that never return;
- recovery of an event gap that is absent from the SGLang replay buffer.

These boundaries should remain explicit. If stronger guarantees become necessary, they should be justified by routing correctness requirements rather than added preemptively.

## Metadata Store

The metadata store is the persistence layer behind the indexer. The initial implementation uses Redis, Dragonfly, Valkey, or Redis Cluster through a topology-independent backend interface.

The Redis data model maintains both directions required by the serving and recovery paths:

```text
placement: hash -> worker_id -> tier bitmask
reverse:   worker_id -> set<hash>
meta:      worker_id -> {address, seq, incarnation, reset_pending}
live:      worker_id -> expiring heartbeat key
hits:      hash -> aggregate count
```

Placement and hit keys share a block-hash cluster tag; reverse, metadata, and liveness keys share a worker cluster tag. This permits every Lua script to declare keys from a single Redis Cluster slot.

## gRPC API

The indexer exposes a small set of metadata update and query APIs. The API operates on hashes, worker IDs, and tiers. It does not expose KV bytes.

### API Overview

| RPC | Purpose |
| --- | --- |
| `ApplyExternalKvBatch` | Apply an ordered batch of report, revoke, and clear-at-tier actions. This is the sole mutation API. |
| `MatchExternalKv` | Query which workers and tiers hold requested hashes. |
| `GetExternalKvHitCounts` | Query accumulated hit counts for hashes. |

### Proto Sketch

```proto
enum TierType {
  TIER_UNKNOWN = 0;
  TIER_HBM = 1;
  TIER_DRAM = 2;
  TIER_SSD = 3;
}

service KVIndexer {
  rpc ApplyExternalKvBatch(ApplyExternalKvBatchRequest)
      returns (ApplyExternalKvBatchResponse);
  rpc MatchExternalKv(MatchExternalKvRequest)
      returns (MatchExternalKvResponse);
  rpc GetExternalKvHitCounts(GetExternalKvHitCountsRequest)
      returns (GetExternalKvHitCountsResponse);
}

message ApplyExternalKvBatchRequest {
  string worker_id = 1;
  uint64 seq = 2;
  repeated ExternalKvAction actions = 3;
  string worker_address = 4;
  string incarnation = 5;
}

message ApplyExternalKvBatchResponse {
  uint64 last_applied_seq = 1;
  bool duplicate = 2;
}
```

`MatchExternalKv` returns matches grouped by worker and tier, for example `worker_id -> tier -> hashes`. `GetExternalKvHitCounts` returns `hash -> hit_count_total`.

`count_as_hit` is used to distinguish real scheduling lookups from diagnostic queries. When it is true, only matched hashes should increase hit-count statistics.

## Implementation Approach

The indexer service will be implemented in Rust. Rust provides near-C++ performance, strong memory and thread-safety guarantees, and a mature async / gRPC ecosystem. This is a good fit for a high-QPS metadata service with concurrent location updates and lookups.

The implementation uses `tonic` for gRPC, `tokio` for the async runtime, ZeroMQ for SGLang event transport, and Redis for durable metadata. The project remains out of tree from SGLang and keeps the backend interface narrow without committing to unused storage implementations.

## Open Questions

- Should the out-of-tree bridge remain a sidecar long term, or should parts of it be upstreamed into SGLang after the API stabilizes?
- Should the indexer store only block-level hashes initially, or also accept prefix-chain information for future prefix-aware routing?
- Should an unrecoverable replay gap continue best-effort, as today, or fail closed and mark the worker unavailable until a full resynchronization?
- When should stale worker metadata and reverse indexes be garbage-collected after a worker permanently leaves?
- Is the single-writer-per-worker deployment invariant sufficient long term, or will multi-writer fencing be required?
