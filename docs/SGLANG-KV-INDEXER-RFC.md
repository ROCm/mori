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
| possible backends:     |  |                            |
| - in-memory            |  |                            |
| - Redis                |  |                            |
| - MySQL / PostgreSQL   |  |                            |
| - RocksDB              |  |                            |
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
| Store abstraction | Persist metadata in memory, Redis, SQL, RocksDB, or another backend. |

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
| `BlockStored(block_hashes, medium)` | `ReportExternalKvBlocks(hashes, tier)` |
| `BlockRemoved(block_hashes, medium)` | `RevokeExternalKvBlocks(hashes, tier)` |
| `AllBlocksCleared()` | `RevokeAllExternalKvBlocksAtTier(tier)` for configured tiers |

For multi-worker or DP-rank deployments, the bridge can manage multiple event sources. Each source has its own worker ID, endpoint, topic, and sequence tracking.

## Metadata Store

The metadata store is the persistence layer behind the indexer. It stores location metadata and optional hit-count data. The indexer should keep the store behind an interface so deployments can choose a backend based on latency, durability, and operational requirements.

| Store backend | Suitable use case |
| --- | --- |
| in-memory | Prototype, single indexer, fastest lookup, no persistence. |
| Redis | Shared low-latency metadata service. |
| MySQL / PostgreSQL | Durable state, operational visibility, SQL queries. |
| RocksDB | Embedded persistent KV store for a single indexer node. |

The initial implementation can use an in-memory store and keep the backend abstraction stable. Redis, SQL, or RocksDB can be added later without changing the gRPC API.

## gRPC API

The indexer exposes a small set of metadata update and query APIs. The API operates on hashes, worker IDs, and tiers. It does not expose KV bytes.

### API Overview

| RPC | Purpose |
| --- | --- |
| `ReportExternalKvBlocks` | Report that a worker holds hashes at a tier. |
| `RevokeExternalKvBlocks` | Remove hashes from a worker at a tier. |
| `RevokeAllExternalKvBlocksAtTier` | Remove all hashes for a worker at a tier. |
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
  rpc ReportExternalKvBlocks(worker_id, hashes, tier) returns (Empty);
  rpc RevokeExternalKvBlocks(worker_id, hashes, tier) returns (Empty);
  rpc RevokeAllExternalKvBlocksAtTier(worker_id, tier) returns (Empty);
  rpc MatchExternalKv(hashes, count_as_hit) returns (matches);
  rpc GetExternalKvHitCounts(hashes) returns (hit_counts);
}
```

`MatchExternalKv` returns matches grouped by worker and tier, for example `worker_id -> tier -> hashes`. `GetExternalKvHitCounts` returns `hash -> hit_count_total`.

`count_as_hit` is used to distinguish real scheduling lookups from diagnostic queries. When it is true, only matched hashes should increase hit-count statistics.

## Implementation Approach

The indexer service will be implemented in Rust. Rust provides near-C++ performance, strong memory and thread-safety guarantees, and a mature async / gRPC ecosystem. This is a good fit for a high-QPS metadata service with concurrent location updates and lookups.

The first implementation can use `tonic` for gRPC, `tokio` for async runtime, and an in-memory metadata store. The project should remain out of tree from SGLang and keep the store interface stable so Redis, SQL, or RocksDB backends can be added later without changing the gRPC API.

## Open Questions

- Should the out-of-tree bridge remain a sidecar long term, or should parts of it be upstreamed into SGLang after the API stabilizes?
- Should the indexer store only block-level hashes initially, or also accept prefix-chain information for future prefix-aware routing?
- What consistency guarantees are required between SGLang KV events and indexer metadata?
- Which metadata store backend should be recommended for production deployment?
