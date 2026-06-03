# UMBP Master Control Plane — Design

**Scope:** Master control plane (`MasterServer`) and the peer-side state it
projects from (`PoolClient` + `PeerServiceServer` + `PeerDramAllocator`).
Authoritative for the Mori C++ tree under `src/umbp/`.

The earlier draft of this doc described a master-led `BlockIndex` that
serviced `Register`/`Unregister`/`Lookup` RPCs. That design has been
replaced. The current design is **master-as-advisor**: master holds no
per-key allocator or page state, peers are the canonical owners of every
KV block they hold, and the master's `GlobalBlockIndex` is a downstream
projection driven exclusively by heartbeat-shipped `KvEvent`s.

---

## 1. Overview

The master is a stateless-by-design routing advisor over a per-node
allocator that lives on the peers. It runs as a single binary
(`build/src/umbp/umbp_master`) exposing one gRPC service (`UMBPMaster`)
plus an optional Prometheus metrics endpoint.

Peers are processes that load `libmori_pybinds.so` (typically inside an
SGLang or vLLM worker). Each peer owns:

- a **`PeerDramAllocator`** — page-bitmap allocator for HBM/DRAM tiers,
  the canonical owner of every per-key page set on this node,
- a **`PeerSsdManager`** (present when the SSD tier is enabled) — the
  canonical owner of this node's SSD copies: the SSD backend, the
  `key → SSD location` map, SSD capacity accounting, the local
  watermark + LRU eviction sweep, and the SSD read staging buffer,
- a **`PeerServiceServer`** (`UMBPPeer` gRPC) — exposes the DRAM/HBM
  allocator RPCs `AllocateSlot`/`CommitSlot`/`AbortSlot`/`ResolveKey`/
  `EvictKey` plus the key-based SSD read staging RPCs
  `PrepareSsdRead`/`ReleaseSsdLease`,
- a **`MasterClient`** — gRPC stub + heartbeat thread that ships
  `KvEvent`s (DRAM/HBM and SSD), capacity snapshots, and Prometheus
  samples to master.

`PoolClient` glues these together and is what `DistributedClient`
(behind `IUMBPClient`) drives on the Put/Get hot path.

```
                   ┌────────────────────────────────────────────────┐
                   │                MasterServer (gRPC)             │
                   │                                                │
  ┌─────────┐ HB   │  ┌────────────────────────────────────────┐    │
  │ Peer A  │─────►│  │ ClientRegistry                         │    │
  │ DRAM/   │◄─────│  │  - membership + capacity per node      │    │
  │ HBM/SSD │      │  │  - heartbeat seq / gap recovery        │    │
  └────┬────┘      │  │  - reaper                              │    │
       │           │  └──────┬─────────────────────────────────┘    │
       │ peer RPC  │         │ apply KvEvent                        │
       │ (writer→  │  ┌──────▼─────────────────────────────────┐    │
       │  reader)  │  │ GlobalBlockIndex                       │    │
       │           │  │  - hash(key) → [Location{node,tier,sz}]│    │
  ┌────▼────┐ HB   │  │  - lease + last_accessed bookkeeping   │    │
  │ Peer B  │─────►│  └──────┬─────────────────────────────────┘    │
  │         │◄─────│         │                                      │
  └─────────┘      │  ┌──────▼─────────┐  ┌────────────────────┐    │
                   │  │ Router         │  │ ExternalKvBlock    │    │
                   │  │  - RouteGet    │  │ Index              │    │
                   │  │  - RoutePut    │  │  - hash → {node,   │    │
                   │  │  - Batch*      │  │    tier} for       │    │
                   │  └────────────────┘  │    L1/L2-cache     │    │
                   │  ┌────────────────┐  │    blocks not      │    │
                   │  │ EvictionMgr    │  │    owned by UMBP   │    │
                   │  │  - watermark   │  └────────────────────┘    │
                   │  │  - EvictKey    │                            │
                   │  │    dispatch    │                            │
                   │  └────────────────┘                            │
                   │  ┌────────────────────────────────────────┐    │
                   │  │ Prometheus metrics server (optional)   │    │
                   │  └────────────────────────────────────────┘    │
                   └────────────────────────────────────────────────┘
```

**Design principles**

1. **Master holds no per-Put or per-page state.** `RoutePut` is a pure
   advisory; the writer reserves capacity by calling `AllocateSlot` on
   the chosen peer. `RouteGet` returns `(node, tier, size)` plus a
   `peer_address`; the reader follows up with `ResolveKey` on the peer
   to obtain pages and RDMA descriptors.
2. **Peer is canonical.** Every page is owned by `PeerDramAllocator`.
   Master's `GlobalBlockIndex` is a projection — if the projection ever
   disagrees with reality, the peer wins, and the index is reconciled
   via heartbeat events (with full-sync as the gap-recovery fallback).
3. **Heartbeat is the only update channel.** `KvEvent{ADD, key, tier,
   size}` and `KvEvent{REMOVE, key, tier}` are queued by the peer and
   shipped in batches, monotonic seq + last-acked-seq for gap detection.
4. **Eviction is master-driven, peer-final.** Master picks victims
   under watermark pressure and ships `EvictKey`. The peer is free to
   reject (read-leased keys) or no-op (already gone). The actual REMOVE
   events arrive on the next heartbeat and that's when the index
   shrinks.
5. **One TTL.** Pending allocator slots TTL out at the peer
   (`pending_ttl`). Master no longer maintains an allocation TTL —
   `UMBP_ALLOCATION_TTL_SEC` is retained as a config knob for legacy
   compatibility but is unused by the live path.
6. **SSD is a peer-owned, best-effort, eventually-consistent cold
   tier.** An SSD copy is an asynchronous replica filled by
   copy-on-commit on the owner peer after the DRAM/HBM commit succeeds;
   the bytes and all physical IO live on the peer. Master is purely an
   advisor over SSD: it learns of SSD copies only through
   `KvEvent{tier=SSD}` on the heartbeat, never directs an SSD write
   (`RoutePut` cannot target SSD), and never sends `EvictKey` for the
   SSD tier — SSD capacity is reclaimed peer-locally by a
   watermark + LRU sweep. SSD is not guaranteed to mirror DRAM: a copy
   that fails or is dropped under back-pressure simply has no replica
   and no event.

### Design space — advisor vs. strong-consistency master

The two ends of the directory-service spectrum, on the axes that
actually drive the protocol shape and the failure model. Each row
lists the property under each model — strengths and costs both — so
the trade-offs are visible without prescribing a choice.

| Aspect | Master-as-Advisor (this design) | Strong-Consistency Master |
|---|---|---|
| **Master's role** | Routing advisor; peer owns the page state | Authoritative directory + allocator |
| **Per-key state on master** | None — heartbeat-projected only | Full, written through a replicated log |
| **Update channel** | Async `KvEvent` shipped on heartbeat | Synchronous master RPC per mutation |
| **Put hot path** | `RoutePut` → peer `AllocateSlot` → RDMA → peer `CommitSlot` | `BeginPut` → RDMA → `CommitPut` (master in every op) |
| **Read-after-write** | Lag of one heartbeat (~5 s) | Linearizable on commit |
| **Capacity** | Stale; ENOSPC handled via `exclude_nodes` retry | Exact; master is the allocator, no surprise ENOSPC |
| **Eviction** | Fire-and-forget; index shrinks on next heartbeat | Transactional; index drops in lockstep with peer ACK |
| **Master outage** | Soft — hot path degrades but keeps serving | Hard — quorum loss halts the cluster |
| **Master state size** | Small — O(unique keys) | Large — O(replicas × allocator metadata) |
| **Throughput ceiling** | Peer aggregate bandwidth | Master Raft commit rate |
| **Replica policy** | Not expressible in the protocol | Enforced by master placement |
| **Trust model** | Peer self-reports ownership | Master mints every `location_id` |
| **Recovery** | Per-peer heartbeat full-sync | Raft log replay on master |
| **Master code complexity** | Small — a handful of classes | Database-engine sized |
| **Suits** | Throughput-bound, miss-tolerant workloads | Strict-consistency workloads with bounded key counts and low write rate |
| **Real-world analogue** | DNS-style routing directory | HDFS NameNode, Ceph monitor, Bigtable directory |

#### Pros and cons — side by side

**+** marks a strength under that model; **−** marks a cost. Some rows
have both — that's the trade-off.

| Aspect | Master-as-Advisor | Strong-Consistency Master |
|---|---|---|
| **Per-op latency** | **+** No master round trip on the hot path | **−** One Raft commit per mutation (~1–5 ms healthy) |
| **Cluster throughput** | **+** Scales with peer aggregate bandwidth | **−** Bounded by master's Raft commit rate |
| **Read-after-write** | **−** Lag of one heartbeat (~5 s) before new key is visible | **+** Linearizable — visible the moment `CommitPut` returns |
| **Capacity accuracy** | **−** Stale snapshots; surprise ENOSPC retried via `exclude_nodes` | **+** Exact — master is the allocator, no surprise ENOSPC |
| **Eviction** | **+** Idempotent, fire-and-forget; safe to retry<br>**−** Index shrinks on next heartbeat, not immediately | **+** Decisive — index drops in lockstep with peer ACK<br>**−** Slow / partitioned peer can block the eviction transaction |
| **Master availability** | **+** Soft — peer-to-peer traffic rides out brief outages | **−** Hard — quorum loss halts the cluster |
| **Master state size** | **+** Small, O(unique keys), unreplicated | **−** Large, O(keys × replicas × allocator metadata), persisted + replicated |
| **Master code surface** | **+** Handful of classes; easy to reason about and modify | **−** Database-engine class — log, leader election, snapshots, membership change |
| **Replica policy / synchronous publish / quotas** | **−** Not expressible in the protocol | **+** Enforceable centrally by master |
| **Trust model** | **−** Peers self-report; index can be poisoned by a buggy/hostile peer | **+** Master mints every `location_id`; peers cannot unilaterally claim ownership |
| **Recovery** | **−** Full-sync (`SnapshotOwnedKeys`) is the only primitive; thundering herd on master restart | **+** Raft log replay; well-understood path |
| **Peer concurrency** | **−** Coarse mutex over `PeerDramAllocator` serializes peer hot path | **+** Peer is a dumb byte-host; no allocator state to lock |
| **Operational failure modes** | **+** Few — no consensus layer to misbehave | **−** Split-brain on membership change, fsync lies, slow-follower starvation, log corruption |
| **Cross-region / large-cluster** | **+** No consensus, no WAN commit penalty | **−** Raft commit latency degrades over WAN; leader CPU bounds cluster size |

---

## 2. Project layout

Authoritative locations under `src/umbp/`:

```
include/umbp/
├── umbp_client.h                      # IUMBPClient (factory entry point)
├── common/
│   ├── config.h                       # UMBPConfig + sub-configs, FromEnvironment
│   ├── env_time.h                     # GetEnv* helpers (timing knobs)
│   ├── error_code.h
│   └── log.h
├── distributed/
│   ├── config.h                       # MasterServerConfig, PoolClientConfig, ClientRegistryConfig
│   ├── distributed_client.h           # DistributedClient (IUMBPClient impl)
│   ├── pool_client.h                  # PoolClient (master + peer + IO engine glue)
│   ├── pool_allocator.h
│   ├── obs_counters.h                 # MORI_UMBP_OBS_* / test-seam build switch
│   ├── types.h                        # TierType, Location, KvEvent, ClientRecord, ...
│   ├── master/
│   │   ├── master_server.h
│   │   ├── master_client.h
│   │   ├── client_registry.h
│   │   ├── global_block_index.h
│   │   ├── external_kv_block_index.h
│   │   ├── eviction_manager.h
│   │   └── master_metrics.h           # MORI_UMBP_METRIC_* name/help strings
│   ├── peer/
│   │   ├── peer_service.h             # UMBPPeer gRPC server
│   │   ├── peer_dram_allocator.h      # canonical per-node DRAM/HBM owner
│   │   ├── peer_ssd_manager.h         # canonical per-node SSD owner
│   │   ├── ssd_copy_pipeline.h        # async copy-on-commit DRAM→SSD
│   │   └── peer_page_allocator.h      # PageBitmapAllocator
│   └── routing/
│       ├── router.h
│       ├── route_get_strategy.h       # TierPriorityRouteGetStrategy (default), RandomRouteGetStrategy
│       └── route_put_strategy.h       # TierAwareMostAvailableStrategy
└── local/                             # Standalone (no-net) DRAM+SSD path
    ├── standalone_client.h
    ├── host_mem_allocator.h
    ├── storage_tier.h
    ├── block_index/local_block_index.h
    └── tiers/                         # CopyPipeline, DRAM/SSD/SPDK tiers

distributed/
├── proto/umbp.proto                   # UMBPMaster service
├── proto/umbp_peer.proto              # UMBPPeer service
├── bin/
│   ├── master_main.cpp                # umbp_master binary
│   └── client_main.cpp
├── master/                            # impl of master-side classes
├── peer/
└── routing/
```

---

## 3. Core types

Defined in `include/umbp/distributed/types.h`.

```cpp
enum class TierType : int { UNKNOWN = 0, HBM = 1, DRAM = 2, SSD = 3 };

struct TierCapacity {
  uint64_t total_bytes = 0;
  uint64_t available_bytes = 0;
};

// (node_id, size, tier).  No location_id — peer is the canonical owner.
// Dedup key on master is (node_id, tier).
struct Location {
  std::string node_id;
  uint64_t size = 0;
  TierType tier = TierType::UNKNOWN;
};

// One mutation in a peer's owned-key set, shipped via heartbeat.  The
// tier is carried on every event and may be HBM, DRAM, or SSD — the
// index keys locations by (node, tier), so a key can hold a DRAM and an
// SSD location at once and a REMOVE only drops the matching tier.
struct KvEvent {
  enum class Kind : int { ADD = 0, REMOVE = 1, CLEAR_AT_TIER = 2 };
  Kind kind = Kind::ADD;
  std::string key;            // empty for CLEAR_AT_TIER
  TierType tier = TierType::UNKNOWN;
  uint64_t size = 0;          // ADD only; REMOVE / CLEAR_AT_TIER leave this 0
};

// Heartbeat events ride in seq-numbered bundles for ack / gap recovery.
struct EventBundle {
  uint64_t seq = 0;
  std::vector<KvEvent> events;
};

// Peer-internal page handle (writer/reader use it to slice RDMA buffers).
struct PageLocation { uint32_t buffer_index; uint32_t page_index; };

// Master-side projection of one peer node.
struct ClientRecord {
  std::string node_id;
  std::string node_address;
  ClientStatus status;                                // ALIVE | EXPIRED
  std::chrono::steady_clock::time_point last_heartbeat;
  std::chrono::steady_clock::time_point registered_at;
  std::map<TierType, TierCapacity> tier_capacities;   // ground truth from peer
  std::string peer_address;                           // UMBPPeer gRPC addr
  std::vector<uint8_t> engine_desc_bytes;             // packed mori::io::EngineDesc
  uint64_t last_applied_seq = 0;                      // gap-recovery cursor
  std::vector<std::string> tags;                      // opaque metric labels
};
```

---

## 4. Master-side components

### 4.1 ClientRegistry  (`include/umbp/distributed/master/client_registry.h`)

Membership ledger + heartbeat ingestion.

- `RegisterClient(node_id, node_address, tier_capacities, peer_address,
  engine_desc_bytes, tags)` — inserts a fresh `ClientRecord` or refreshes
  an expired one; rejects when a live record with the same `node_id` is
  already present.
- `UnregisterClient(node_id)` — drops the record and clears every index
  entry that belonged to it (delegated to `GlobalBlockIndex` and
  `ExternalKvBlockIndex`).
- `Heartbeat(node_id, tier_capacities, bundles, is_full_sync,
  delta_seq_baseline, out_acked_seq, out_request_full_sync)` — applies
  one heartbeat: replaces capacity, then applies each `EventBundle` in
  `bundles` to `GlobalBlockIndex` in seq order (bundles at or below the
  stored cursor are skipped as retransmissions), advancing
  `last_applied_seq`. On `is_full_sync` it instead replays the full
  owned-key set via `ReplaceNodeLocations` and adopts
  `delta_seq_baseline` as the new cursor. If a delta bundle's seq leaves
  a gap (`!= last_applied_seq + 1`), the call requests recovery:
  `out_request_full_sync = true` and `out_acked_seq` echoes the stored
  cursor so the peer reships a full sync.
- **Reaper** — background thread expires nodes that miss
  `heartbeat_ttl × max_missed_heartbeats` and triggers full GC. There is
  no separate allocation reaper in the new design.

`ClientRegistryConfig` (defaults; all overridable via `UMBP_*` env, see
`runtime-env-vars.md`):

| Field | Default |
|---|---|
| `heartbeat_ttl` | 10 s |
| `reaper_interval` | 5 s |
| `allocation_ttl` | 30 s (legacy; unused by live path) |
| `finalized_record_ttl` | 120 s (legacy; unused by live path) |
| `max_missed_heartbeats` | 3 |
| `default_dram_page_size` | 2 MiB |

### 4.2 GlobalBlockIndex  (`include/umbp/distributed/master/global_block_index.h`)

A `std::unordered_map<key, BlockEntry>` projection, mutated **only** by
heartbeat ingestion.

```cpp
struct BlockEntry {
  std::vector<Location> locations;     // dedup key = (node_id, tier)
  BlockMetrics metrics;                // created_at, ...
  std::atomic<int64_t> lease_expiry_rep{0};
  std::atomic<int64_t> last_accessed_rep{0};
  std::atomic<uint64_t> atomic_access_count{0};
  // GrantLease, IsLeased, RecordAccessAtomic, GetLastAccessed
};
```

The index is **additive over `(key, node, tier)`**. A location's dedup
key is `(node_id, tier)`, so the same key on the same node can carry a
DRAM (or HBM) location and an SSD location side by side, as two
independent `Location` entries. This is what lets a key remain
discoverable on SSD after its DRAM copy is gone, and is the only
master-side support the SSD cold tier needs — ADD/REMOVE/exists/route
all fall out of the per-tier bookkeeping.

Mutators:

- `ApplyEvents(node_id, events[])` — apply a peer's batch.
  - **ADD** inserts a `(node_id, tier)` location with the event's
    `size`. A duplicate ADD for an existing `(node_id, tier)` is an
    idempotent no-op (the existing location is kept and a warning is
    logged); it does not overwrite the stored size.
  - **REMOVE** drops only the location matching `(key, node_id, tier)`.
    Removing the DRAM copy leaves the SSD copy (and vice versa)
    untouched; REMOVE for an unknown `(key, node_id, tier)` is a silent
    no-op. The key's entry is erased only once its last location is
    gone.
  - **CLEAR_AT_TIER** drops every location for `(node_id, tier)` across
    all keys (keyless event); used to wipe one tier's placements for a
    node.
- `ReplaceNodeLocations(node_id, adds[])` — drop every prior location
  for `node_id` and reseed from `adds` (which may mix DRAM/HBM and SSD
  ADDs). Used on full-sync.
- `RecordAccess(key)` / `GrantLease(key, duration)` — under the shared
  lock; both `last_accessed_rep` and `lease_expiry_rep` are atomic
  reps, so neither needs an exclusive lock.

Queries:

- `Lookup(key)` / `BatchLookupExists(keys[])` / `GetMetrics(key)`.
  `BatchLookupExists` returns `true` as long as the key has **any**
  location, so a key that lives only on SSD still reports as resident —
  hicache will prefetch it.
- `BatchLookupForRouteGet(keys[], exclude_nodes, lease_duration)` —
  returns **all** tiers' locations per key (so RouteGet sees SSD
  replicas), and on a non-empty result bumps `RecordAccess` + grants a
  lease, under one shared lock.
- `FindEvictionCandidates(overloaded_node_tiers)` — returns
  `EvictionCandidate{key, location, last_accessed_at, size}` rows for
  the given `(node_id, tier)` set. Skips entries with active leases.
  Only DRAM/HBM `(node, tier)` pairs are ever passed in (see §4.5).

### 4.3 ExternalKvBlockIndex  (`external_kv_block_index.h`)

A separate, lighter index for **unmanaged** L1/L2 cache blocks (e.g.
the SGLang host-mem KV cache). Maps `hash → {node_id → tier}` and
exists alongside `GlobalBlockIndex` because:

- the data lives outside any UMBP-owned allocator (so there's no
  `Location` size/lease to track),
- producers report blocks by hash and consumers want one match list per
  query (`Match(hashes)` returns `[{node_id, matched_hashes, tier}]`).

This index is **not** updated by heartbeat events — clients call
`ReportExternalKvBlocks` / `RevokeExternalKvBlocks` directly. Bulk
clean-up on node expiry is `UnregisterByNode(node_id)`, called from the
reaper / `UnregisterClient` path.

**Do not confuse this index's `tier=SSD` with the UMBP-owned SSD cold
tier.** They are two separate mechanisms:

- The **UMBP-owned SSD tier** (the cold tier described throughout this
  doc) lives in `GlobalBlockIndex`. Its bytes are real, owned by a
  peer's `PeerSsdManager`, populated by copy-on-commit, advertised via
  `KvEvent{tier=SSD}` on the heartbeat, and readable through `RouteGet`
  → `PrepareSsdRead`.
- An **external-KV block with `tier=SSD`** (e.g. an entry hicache
  reports for its own L3) lives in `ExternalKvBlockIndex`. It is pure
  scheduling metadata: no bytes move through UMBP, `RouteGet` never
  consults it, and `PrepareSsdRead` cannot read it.

A hash/key may appear in both, but the serving paths are disjoint.
`RouteGet` only ever resolves `GlobalBlockIndex` locations.

### 4.4 Router  (`include/umbp/distributed/routing/router.h`)

Stateless façade over the two index objects + `ClientRegistry`.
Dispatches to pluggable strategies; defaults are
`TierPriorityRouteGetStrategy` and `TierAwareMostAvailableStrategy`.

- `RouteGet(key, node_id, exclude_nodes)` returns
  `RouteGetResolution{Location, peer_address}` (so the reader doesn't
  need a separate `GetClientInfo` lookup before issuing `ResolveKey`).
  On hit, the router calls `RecordAccess` and `GrantLease(key,
  lease_duration)` so the lookup pins the key against eviction during
  the writer's RDMA round trip.
- `RoutePut(key, node_id, block_size, exclude_nodes)` returns
  `RoutePutResult{node_id, peer_address, tier}`.
- Batch variants take parallel `keys[]` / `block_sizes[]` and return
  `vector<optional<...>>`.
- `exclude_nodes` is the writer's "I tried these and got ENOSPC /
  not-found" steer set: subsequent retries fold the failed peer into
  this set so master picks a different replica or destination.

`RouteGetStrategy::Select(locations, node_id)` —
`TierPriorityRouteGetStrategy` is the default: it picks the
fastest tier present in the **read priority order HBM > DRAM > SSD**
(`UNKNOWN` ranks last), then chooses a random replica within that tier
so load still spreads. So a key that has both a DRAM and an SSD copy is
served from DRAM, and SSD is read only when it is the sole tier
present. `RandomRouteGetStrategy` (uniform across all replicas
regardless of tier) remains available to inject via
`MasterServerConfig::get_strategy`.

`RoutePutStrategy::Select(alive_clients, block_size, exclude_nodes)` —
`TierAwareMostAvailableStrategy` walks **`[HBM, DRAM]`** in order and,
on the first tier with any node holding `>= block_size` available
capacity, picks the node with the most available bytes (load
spreading). **SSD is intentionally not a `RoutePut` target**: there is
no direct-SSD-put path — the SSD copy is filled asynchronously by
copy-on-commit, so even with SSD capacity reported on the heartbeat,
`RoutePut` must never steer a write at a tier with no direct-put
semantics.

### 4.5 EvictionManager  (`eviction_manager.h`)

Background thread that runs on `EvictionConfig::check_interval`. On
each tick:

1. Walk `ClientRegistry` for nodes whose tier(s) have crossed
   `EvictionConfig::high_watermark` (default 0.9). Collect overloaded
   `(node_id, tier)` pairs. **The SSD tier is skipped here** — master
   never turns an SSD overload into an `EvictKey`. `EvictKey` acts only
   on the peer's `PeerDramAllocator`, so dispatching it for an SSD
   overload would wrongly evict the DRAM copy of a key while leaving the
   SSD bytes in place. SSD capacity is reclaimed entirely peer-locally
   (watermark + LRU in `PeerSsdManager`), and the resulting
   `KvEvent{REMOVE, tier=SSD}` shrinks the index on the next heartbeat.
2. Call `GlobalBlockIndex::FindEvictionCandidates(...)` to get a
   sorted-by-LRU candidate list, skipping leased keys.
3. Group the victims by `node_id` and dispatch `EvictKey` to each peer
   via `EvictKeyDispatcher` (see `MasterPeerStubPool` in
   `master_server.cpp`).

Master state does **not** mutate as a result of dispatching `EvictKey`.
The peer is the source of truth: it reports the actually-freed pages on
its next heartbeat as `KvEvent{REMOVE}`, and that is when
`GlobalBlockIndex` shrinks. This makes the eviction loop idempotent
under retries, and makes "the peer rejected this eviction because the
key is read-leased" cost nothing extra to the master.

### 4.6 MasterServer  (`master_server.h`)

Owns `ClientRegistry`, `GlobalBlockIndex`, `ExternalKvBlockIndex`,
`Router`, `EvictionManager`, the gRPC server, the metrics server, and
the outbound `MasterPeerStubPool` (the concrete `EvictKeyDispatcher`).

```cpp
struct MasterServerConfig {
  std::string listen_address = "0.0.0.0:50051";
  int metrics_port = 0;                                // 0 = disabled
  ClientRegistryConfig registry_config;
  EvictionConfig eviction_config;
  std::unique_ptr<RouteGetStrategy> get_strategy;
  std::unique_ptr<RoutePutStrategy> put_strategy;
  static MasterServerConfig FromEnvironment();
};

class MasterServer {
 public:
  explicit MasterServer(MasterServerConfig config);   // by value, holds unique_ptrs
  void Run();                                         // blocks
  void Shutdown();
  uint16_t GetBoundPort() const;                      // for listen_address with port=0
};
```

`bin/master_main.cpp` is the binary entry point: it calls
`FromEnvironment()`, lets `argv[1]` override `listen_address` and
`argv[2]` override `metrics_port`, prints one
`[Master] Resolved timing: ...` line, then runs until SIGINT/SIGTERM.

---

## 5. gRPC contract

### 5.1 `UMBPMaster`  (`distributed/proto/umbp.proto`)

Master-facing service. Sources of truth are the proto file and the
`MasterServer` impl in `master_server.cpp`.

| RPC | Purpose |
|---|---|
| `RegisterClient(RegisterClientRequest)` | Membership announce. Carries `peer_address`, packed `engine_desc`, optional `tags`, and per-tier `tier_capacities` — **SSD capacity rides here as `TierCapacity{tier=TIER_SSD}`**, there is no separate `ssd_store_capacities` field. Response includes recommended heartbeat interval and the initial `ack_seq`. |
| `UnregisterClient(UnregisterClientRequest)` | Graceful shutdown. |
| `Heartbeat(HeartbeatRequest)` | The authoritative update channel. Carries `tier_capacities` (ground truth, including SSD) and `bundles[]` — seq-numbered `EventBundle`s of `KvEvent`s since the last ack, with DRAM/HBM and SSD events merged into one bundle under one monotonic seq. `is_full_sync` flips the request into a complete owned-key set replay; `delta_seq_baseline` carries the cursor for that replay. Response has `acked_seq` and `request_full_sync` for gap recovery. |
| `RouteGet(RouteGetRequest)` | Pick a replica. Read-only against `GlobalBlockIndex`. |
| `RoutePut(RoutePutRequest)` | Pick a target node + tier. Read-only against `ClientRegistry`. |
| `BatchRouteGet`/`BatchRoutePut` | Parallel-key variants. `BatchRoutePut` also performs **master-side Put dedup**: any key already present in `GlobalBlockIndex` is returned with `already_exists=true` (and no node selection) so the caller can skip the Put entirely — primary defense against re-uploading the same kv-cache from multiple ranks (sglang DP-attention). Peer's `AllocateSlot` carries a key field and applies the same dedup as a defensive layer against master-index lag. |
| `BatchLookup(BatchLookupRequest)` | Read-only batched existence probe. Goes straight to `GlobalBlockIndex::BatchLookupExists` — no `RecordAccess`, no `GrantLease`, no per-node RouteGet counters. Used by `PoolClient::Exists` / `BatchExists` for the sglang probe path where the caller only wants to know "is the key resident?" and is not about to RDMA-read it. |
| `ReportExternalKvBlocks`/`RevokeExternalKvBlocks`/`MatchExternalKv` | External (unmanaged) cache index. |
| `ReportMetrics(ReportMetricsRequest)` | Client-side counters/gauges/histograms forwarded to master's Prometheus exposition. |

Existence checks go through `BatchLookup` (or `MatchExternalKv` for L1/L2
lookups); the legacy per-key `Register`/`Unregister`/`Lookup` RPCs are
gone, and `RouteGet`/`BatchRouteGet` are reserved for the real read path
(they bump `RecordAccess` and `GrantLease` on hit, which is the wrong
side-effect for a pure probe).

### 5.2 `UMBPPeer`  (`distributed/proto/umbp_peer.proto`)

Peer-to-peer service hosted by `PeerServiceServer` on every node that
has a DRAM/HBM tier or an SSD tier.

| RPC | Purpose |
|---|---|
| `GetPeerInfo` | First-contact hydration: packed `engine_desc`, SSD staging buffer descriptor, all DRAM/HBM `BufferMemoryDesc`s, and the tier `dram_page_size`. |
| `AllocateSlot(key, size, tier)` | Reserve a pending **DRAM/HBM** slot (`tier` is HBM or DRAM only — SSD is never a direct put target). Response carries an `AllocateSlotOutcome` and, on success, `slot_id`, the `pages` it covers, `page_size`, the dedup'd `descs` for those pages, and a `pending_ttl_ms`. Outcomes: `SUCCESS_ALLOCATED`, `FAILED_NO_SPACE` (ENOSPC — writer retries `RoutePut` with the node added to `exclude_nodes`), or `FAILED` (generic). **Duplicate-key dedup**: if `owned_[key]` is already present (master-index-lag fallback), outcome is `SUCCESS_ALREADY_EXISTS` — caller treats it as a no-op success and skips RDMA. |
| `CommitSlot(slot_id, key)` | Move pending → owned. Queues `KvEvent{ADD, key, tier, size}` for the next heartbeat, and — when the SSD tier is enabled — enqueues an async copy-on-commit task so the bytes are mirrored to SSD (best-effort; the eventual SSD copy emits its own `KvEvent{ADD, tier=SSD}`). |
| `AbortSlot(slot_id)` | Drop a pending slot. Idempotent. |
| `ResolveKey(key)` | DRAM/HBM read-side lookup. Bumps the per-key read-lease counter (Bug #7 mitigation). Returns `pages`, `page_size`, `descs`, `size`. |
| `EvictKey(keys[])` | Master-driven eviction of **DRAM/HBM** keys only. Read-leased (or copy-pinned) keys produce `bytes_freed=0`; everything actually freed yields a `KvEvent{REMOVE}` on the next heartbeat. |
| Batch variants | `BatchAllocateSlots`/`BatchCommitSlots`/`BatchAbortSlots`/`BatchResolveKeys`. |
| `PrepareSsdRead(key, max_size)` | Key-based SSD read staging. The peer looks the key up in `PeerSsdManager`, reads the bytes into the published SSD staging buffer, and returns an `SsdReadStatus` (`OK` / `NOT_FOUND` / `NO_SLOT` / `SIZE_TOO_LARGE` / `ERROR`), the `staging_offset`, the actual `size`, and a `lease_id` + `lease_ttl_ms`. `NOT_FOUND` is a definitive miss; `NO_SLOT` is transient and must be retried, never reported as a miss. |
| `ReleaseSsdLease(lease_id)` | Best-effort fast release of an SSD read staging slot. Slots are also reclaimed by lease TTL, so a missed release is harmless. |

---

## 6. Hot-path data flow

### 6.1 Put

```
  Client (writer)              Master                     Peer (target)
       │                         │                            │
       │   RoutePut(key, sz)     │                            │
       │────────────────────────►│  registry.GetAliveClients  │
       │                         │  strategy.Select(...)      │
       │  RoutePutResult{node,   │                            │
       │  peer_address, tier}    │                            │
       │◄────────────────────────│                            │
       │                                                      │
       │   AllocateSlot(sz, tier)                             │
       │─────────────────────────────────────────────────────►│  PeerDramAllocator.Allocate
       │                                                      │   ├─ ENOSPC → success=false (writer
       │                                                      │   │   retries RoutePut with this
       │                                                      │   │   node added to exclude_nodes)
       │                                                      │   └─ slot_id, pages, descs, page_sz
       │   AllocateSlotResponse{slot_id, pages, descs, ...}   │
       │◄─────────────────────────────────────────────────────│
       │                                                      │
       │   [Writer RDMAs each scatter chunk into pages]       │
       │   (zero-copy if RegisterMemory was called;           │
       │    otherwise routed through staging buffer)          │
       │                                                      │
       │   CommitSlot(slot_id, key)                           │
       │─────────────────────────────────────────────────────►│  pending → owned
       │                                                      │  queue KvEvent{ADD, key, tier, sz}
       │   CommitSlotResponse{success=true}                   │
       │◄─────────────────────────────────────────────────────│
                                                              │
                                  ... peer heartbeat ...      │
                                                              │
       │                         │ Heartbeat(seq, events[ADD…])
       │                         │◄───────────────────────────│
       │                         │ ApplyEvents → GlobalBlockIndex
```

When the SSD tier is enabled, a successful `CommitSlot` on the owner
peer also enqueues an async **copy-on-commit** task. A copy worker pins
the just-committed DRAM/HBM pages, writes them to the SSD backend, and —
only on a successful write — records the SSD location and queues a
`KvEvent{ADD, tier=SSD}`. This is best-effort and off the writer's hot
path: a failed or dropped copy simply leaves no SSD replica. The writer
never targets SSD directly.

### 6.2 Get

```
  Client (reader)              Master                     Peer (source)
       │                         │                            │
       │   RouteGet(key, exclude)│                            │
       │────────────────────────►│  GlobalBlockIndex.Lookup  │
       │                         │  + RecordAccess + Grant…  │
       │  RouteGetResult{node,   │                            │
       │  tier, size, peer_addr} │                            │
       │◄────────────────────────│                            │
       │                                                      │
       │   ResolveKey(key)                                    │
       │─────────────────────────────────────────────────────►│  PeerDramAllocator.Resolve
       │                                                      │  bumps read-lease for `key`
       │   ResolveKeyResponse{pages, descs, page_sz, size}    │
       │◄─────────────────────────────────────────────────────│
       │                                                      │
       │   [Reader RDMAs from the listed pages]               │
```

If `ResolveKey` returns `found=false` (peer evicted between RouteGet
and Resolve), the reader retries `RouteGet` with the failed node added
to `exclude_nodes`. If every replica fails, the get returns `false`.

When `RouteGet` resolves to a `tier=SSD` location (only when no
DRAM/HBM replica exists, per the tier-priority strategy), the read
follows the SSD path instead of `ResolveKey`: a remote reader calls
`PrepareSsdRead(key, max_size)`, RDMA-reads from the returned
`{staging_offset, size}` inside the peer's published SSD staging
buffer, then `ReleaseSsdLease(lease_id)`; a reader that is itself the
owner reads its own SSD bytes directly without staging. Transient
`NO_SLOT` results are retried with bounded attempts and must never be
surfaced as a cache miss — only `NOT_FOUND` is a miss.

### 6.3 Eviction

```
  EvictionManager (master)       Peer A                          Master HB ingest
       │                             │                                  │
       │  Find overloaded(node,tier) │                                  │
       │  Find candidates (LRU)      │                                  │
       │                             │                                  │
       │  EvictKey([k1, k2, k3])     │                                  │
       │────────────────────────────►│  PeerDramAllocator.Evict        │
       │                             │   - k1 : freed → REMOVE queued  │
       │                             │   - k2 : read-leased → 0 bytes  │
       │                             │   - k3 : already gone → 0 bytes │
       │  EvictKeyResponse{[(k1,sz),(k2,0),(k3,0)]}                    │
       │◄────────────────────────────│                                  │
                                                                        │
                              ... peer heartbeat ...                    │
                                                                        ▼
                              Heartbeat(seq, [REMOVE k1])  → ApplyEvents
                                                            → GlobalBlockIndex shrinks
```

The master's eviction tick mutates **no master state directly**; the
index only shrinks once the heartbeat lands. This makes eviction
idempotent and correct under heartbeat loss / replay.

### 6.4 Heartbeat gap recovery

```
  Peer                                   Master
   │                                       │
   │   Heartbeat(seq=N+1, last_acked=N,    │
   │             events[…], is_full=false) │
   │──────────────────────────────────────►│  expected: last_applied_seq + 1
   │                                       │  N+1 == 12 + 1 ?  No → 12 known
   │                                       │     out_acked_seq = 12
   │                                       │     out_request_full_sync = true
   │   HeartbeatResponse{                  │
   │     status=ALIVE,                     │
   │     acked_seq=12,                     │
   │     request_full_sync=true}           │
   │◄──────────────────────────────────────│
   │                                       │
   │   Heartbeat(seq=N+2, is_full=true,    │
   │             events = SnapshotOwnedKeys()) │
   │──────────────────────────────────────►│  ReplaceNodeLocations(node, adds)
   │                                       │  last_applied_seq = N+2
   │   HeartbeatResponse{acked_seq=N+2}    │
   │◄──────────────────────────────────────│
```

The peer's `MasterClient` heartbeat thread tracks `hb_seq_` and
`hb_last_acked_seq_`. On `request_full_sync=true` it calls
`PeerDramAllocator::SnapshotOwnedKeys()` to materialize a complete ADD
list and reships with `is_full_sync=true`.

---

## 7. Peer-side components

### 7.1 PeerDramAllocator  (`peer/peer_dram_allocator.h`)

Canonical owner of every per-key page set on the local node. Backed by
one `PageBitmapAllocator` per configured tier (HBM, DRAM).

State:

```cpp
unordered_map<uint64_t, PendingSlot>           pending_;
unordered_map<string,    OwnedSlot>            owned_;
unordered_map<string,    time_point>           read_lease_until_;
vector<KvEvent>                                pending_events_;  // outbox
```

Lifecycle:

| Op | Effect |
|---|---|
| `Allocate(size, tier)` | Reserve pages, assign `slot_id`, push `PendingSlot` into `pending_`. |
| `Commit(slot_id, key)` | Pop from `pending_`, insert into `owned_`, enqueue `KvEvent{ADD}` in `pending_events_`. |
| `Abort(slot_id)` | Pop from `pending_`, free pages. Idempotent. |
| `Resolve(key)` | Look up in `owned_`, set `read_lease_until_[key] = now()+read_lease_ttl_` (covers any earlier deadline since `steady_clock` is monotonic), return pages. |
| `Evict(keys[])` | For each key: skip if `HasActiveReadLeaseLocked`, else free pages and enqueue `KvEvent{REMOVE}`. |
| `DrainPendingEvents()` | Returns and clears `pending_events_`; called by the heartbeat thread. |
| `SnapshotOwnedKeys()` | Build the full ADD list for full-sync. |
| `TierCapacitiesSnapshot()` | Live `(total, available)` per tier; reported with every heartbeat. |
| `AcquireDramCopyPin(key)` / `ReleaseDramCopyPin(key, token)` | Pin a committed key's pages so the SSD copy worker can read them safely; pinned keys cannot be freed by `Evict` until released. |

`PeerDramAllocator` covers only the DRAM/HBM tiers. SSD ADD/REMOVE
events are **not** routed back through this allocator — they originate
in `PeerSsdManager`, which is a separate `OwnedLocationSource`. The
heartbeat shipper (`MasterClient`) drains both sources and concatenates
their events into one bundle per seq (see `PeerSsdManager` below).

A reaper thread sweeps `pending_` for expired slots (`pending_ttl`) and
`read_lease_until_` for expired entries (`read_lease_ttl_`).

### 7.2 PeerSsdManager  (`peer/peer_ssd_manager.h`)

Canonical owner of every SSD copy on the local node, present only when
`ssd.enabled`. It is a separate, single-responsibility class — it does
**not** reuse the standalone `LocalStorageManager`/`LocalBlockIndex` and
has no DRAM tier or demote/promote logic. It wraps one `TierBackend`
(`SSDTier` for `posix`, `SpdkProxyTier` for `spdk`/`spdk_proxy`) and
implements `OwnedLocationSource`.

State:

```cpp
unordered_map<string, OwnedEntry>   owned_;            // key -> {size, lru_it}
list<string>                        lru_;              // front = MRU, back = LRU
unordered_map<string, int>          inflight_reads_;   // read-priority guard
unordered_set<string>               evicting_;
vector<KvEvent>                     pending_events_;    // SSD ADD/REMOVE outbox
```

Behavior:

| Op | Effect |
|---|---|
| `Capacity()` | `(used, total)` from the backend; reported on the heartbeat as `TierCapacity{tier=SSD}`. |
| `Write(key, segments, total_size)` | Copy-on-commit landing. Assembles the (possibly non-contiguous) DRAM source segments and writes to the backend; on success records the SSD location and queues `KvEvent{ADD, tier=SSD}`. Idempotent on an already-resident key (content-addressed): no rewrite, no duplicate ADD, just an LRU touch. |
| `PrepareRead(key, staging_ptr, cap)` | Resolve size under the lock, run the blocking backend read outside it, return `SsdReadOutcome{status, size}`. `kNotFound` for a missing or currently-evicting key, `kSizeTooLarge` when the key exceeds the reader's cap, `kError` on a backend read failure. |
| `Evict(key)` | Local eviction. Skips keys with in-flight reads (read priority); frees the backend bytes and, only on success, drops `owned_`/`lru_` and queues `KvEvent{REMOVE, tier=SSD}`. |
| `EvictToLowWatermark()` | Check-after-write trigger: when `used/total ≥ high_watermark` it evicts oldest-first down to `low_watermark`. Runs on the copy worker — no dedicated thread — and serializes rounds with a try-lock. |
| `ClearLocal()` | Drop the logical map + pending events and wipe the physical SSD bytes (user `Clear` = discard cache). Drains in-flight reads first; the caller must quiesce the copy pipeline before calling. |
| `DrainPendingEvents()` / `SnapshotOwnedKeys()` | `OwnedLocationSource` — feed SSD ADD/REMOVE into the heartbeat and rebuild the full SSD ADD list on full-sync. |

SSD capacity reclamation is entirely peer-local (this class's
watermark + LRU); master is never involved. The resulting
`KvEvent{REMOVE, tier=SSD}` is what shrinks `GlobalBlockIndex` on the
next heartbeat.

### 7.3 SsdCopyPipeline  (`peer/ssd_copy_pipeline.h`)

Bounded async worker pool that mirrors committed DRAM/HBM keys to SSD.
Owned by `PoolClient`, constructed only when `ssd.enabled`; it borrows
`PeerDramAllocator` (the pin source) and `PeerSsdManager` (the write
target). A successful `CommitSlot` on the owner peer enqueues an
`SsdCopyTask{key, ...}` (the task carries no user pointer — the worker
re-reads the bytes from the owner's DRAM pages by key). Each worker:

1. dequeues a task and calls `AcquireDramCopyPin(key)` — a `nullopt`
   (key already evicted, or a duplicate task) drops the task,
2. holds the pin under a RAII guard (always released),
3. calls `PeerSsdManager::Write(key, pin.segments, pin.total_size)`,
4. releases the pin when the guard leaves scope.

`Enqueue` never blocks the commit hot path: a full or stopped queue
drops the task and bumps a counter. The pin protects the pages against
`EvictKey` for the lifetime of the copy — a copy-pinned key returns
`bytes_freed=0` from `EvictKey` and master retries it on a later
eviction round.

### 7.4 PeerServiceServer  (`peer/peer_service.h`)

Hosts the `UMBPPeer` gRPC service. Holds non-owning pointers to
`PeerDramAllocator` (DRAM/HBM) and `PeerSsdManager` (SSD), plus the SSD
read staging region (base / size / packed `MemoryDesc`). `dram_alloc`
may be null when the process has no DRAM/HBM tier — the DRAM/HBM RPCs
then respond `FAILED` / `found=false` while the SSD read RPCs keep
working. Symmetrically, when `peer_ssd` (or the staging region) is null,
`SsdRpcAvailable()` is false and `PrepareSsdRead` returns
`SSD_READ_ERROR`. The published `ssd_staging_mem_desc` / `ssd_staging_size`
in `GetPeerInfo` come straight from this staging region.

`engine_desc_bytes` is the packed `mori::io::EngineDesc` that the IO
engine published when `PoolClient` constructed it. It's plumbed through
`GetPeerInfo` and the `descs` payloads so writers/readers can wire up
RDMA without a separate engine-discovery step.

### 7.5 PoolClient  (`distributed/pool_client.h`)

Glues the peer-side subsystems on a peer process:

- `MasterClient` (gRPC stub + heartbeat + metrics threads),
- `PeerDramAllocator` (owned),
- `PeerSsdManager` + `SsdCopyPipeline` (owned, only when `ssd.enabled`),
- `PeerServiceServer` (owned, started on `peer_service_port`),
- `mori::io::IOEngine` (owned, published as `engine_desc`).

`MasterClient` registers both `PeerDramAllocator` and `PeerSsdManager`
as `OwnedLocationSource`s, so the heartbeat ships DRAM/HBM and SSD
events together and the capacity snapshot includes SSD. `Clear()`
quiesces the copy pipeline, clears both the allocator and the SSD
manager (the latter also wiping the physical SSD bytes), then resumes
the pipeline so no stale copy re-adds an SSD location afterward.

Hot path (`Put` / `Get`):

1. Master `RoutePut`/`RouteGet` advisory.
2. Self-target fast path: if the target is the local node, skip RPC
   and write/read directly via `LocalPutPages` / `LocalGetPages`.
3. Otherwise, lazy-connect the target peer (cache `PeerConnection` in
   `peers_`), call `AllocateSlot` / `ResolveKey`, hydrate per-buffer
   `MemoryDesc`s on first reference, then build a single
   `TransferInstruction` set to issue one batched RDMA scatter
   read/write.
4. `CommitSlot` (Put) or release (Get).
5. Per-attempt retries on ENOSPC / not-found re-call `RoutePut` /
   `RouteGet` with `exclude_nodes` extended.

`RegisterMemory(ptr, size)` pins a caller-owned region for zero-copy.
On the hot path, `FindRegisteredMemory(src, size)` swaps in the
registered descriptor and skips the staging buffer entirely.

### 7.6 DistributedClient  (`distributed/distributed_client.h`)

`IUMBPClient` adaptor over `PoolClient`. `CreateUMBPClient(config)`
returns a `DistributedClient` when `config.distributed.has_value()` and
a `StandaloneClient` otherwise.

---

## 8. Standalone path (no master)

`StandaloneClient` (`include/umbp/local/standalone_client.h`) implements
the same `IUMBPClient` interface but talks to an in-process
`LocalBlockIndex` + `LocalStorageManager` (DRAM tier, optional
POSIX/SPDK SSD tier, copy-pipeline worker pool). No gRPC, no networking,
no master. Used by:

- single-node smoke tests (`scripts/run_umbp_single_node_hicache.sh`,
  see `MORI-UMBP-SINGLE-NODE-GUIDE.md`),
- the legacy "Shared SSD leader/follower" deployment driven by
  `UMBPRole::SharedSSD{Leader,Follower}` and the `follower_mode` /
  `force_ssd_copy_on_write` back-compat flags in `UMBPConfig`.

---

## 9. Config surface

`UMBPConfig` (`include/umbp/common/config.h`) is the single struct
plumbed through both standalone and distributed factories. Notable
fields:

| Section | Field | Default | Notes |
|---|---|---|---|
| `dram` | `capacity_bytes` | 4 GiB | Local DRAM tier size. |
| `dram` | `high_watermark` / `low_watermark` | 0.9 / 0.7 | LRU eviction trigger band. |
| `ssd` | `enabled` | true | Toggle SSD tier. |
| `ssd` | `storage_dir` | `/tmp/umbp_ssd` | POSIX backend root. |
| `ssd` | `capacity_bytes` | 32 GiB | |
| `ssd` | `segment_size_bytes` | 256 MiB | SegmentedLog segment size. |
| `ssd` | `high_watermark` / `low_watermark` | 0.9 / 0.7 | Peer-local SSD eviction band (`UMBP_SSD_HIGH_WM` / `UMBP_SSD_LOW_WM`). Must satisfy `0 < low < high <= 1` or construction throws. |
| `ssd` | `ssd_backend` | `posix` | `posix`, `spdk`, or `spdk_proxy`. Distributed peers reach SPDK only via the proxy. |
| `ssd` | `spdk_*` | — | SPDK/proxy tuning (BDF, reactor mask, channels, shm name, ...), all under `ssd.*`. |
| `ssd.io` | `backend` | `IoUring` | `PThread` or `IoUring`. |
| `ssd.durability` | `mode` | `Strict` | `Strict` or `Relaxed`. |
| `eviction` | `policy` | `lru` | |
| `copy_pipeline` | `worker_threads` | 2 | Async DRAM→SSD copy-on-commit workers. |
| top-level | `role` | `Standalone` | `Standalone` / `SharedSSDLeader` / `SharedSSDFollower`. |
| top-level | `distributed` | `nullopt` | Set to enable master-led mode. |

`UMBPDistributedConfig`:

| Field | Default | Notes |
|---|---|---|
| `master_config.master_address` | (required) | e.g. `host:50051`. |
| `master_config.node_id` | (required) | Cluster-unique. |
| `master_config.node_address` | (required) | Address peers should reach back on. |
| `master_config.auto_heartbeat` | true | Heartbeat thread starts on `Init`. |
| `io_engine.host` | "" | Empty selects RDMA; sites typically use `127.0.0.1` for local IO engine. |
| `io_engine.port` | 0 | `0` = OS-assigned ephemeral. |
| `staging_buffer_size` | 64 MiB | Host staging for non-zero-copy RDMA. |
| `ssd_staging_buffer_size` | 256 MiB | Dedicated SSD read staging, allocated only when SSD is enabled. A slot (`ssd_staging_buffer_size / ssd_staging_buffer_slots`) must fit the largest single-key value. |
| `ssd_staging_buffer_slots` | 16 | Number of remote SSD read staging slots. |
| `peer_service_port` | 0 | `0` = OS-assigned. |
| `cache_remote_fetches` | true | Locally re-cache blocks fetched from a remote peer. |
| `dram_page_size` | 0 | `0` ⇒ use master's `default_dram_page_size` (2 MiB). |

`UMBPConfig::FromEnvironment()` overlays `UMBP_*` env vars on top of
the defaults; see `runtime-env-vars.md` for the full list.

---

## 10. Concurrency

| Object | Mutex | Notes |
|---|---|---|
| `GlobalBlockIndex::entries_` | `shared_mutex` | `ApplyEvents` / `ReplaceNodeLocations` exclusive; `Lookup` / `BatchLookupExists` / `GetMetrics` shared. `RecordAccess` and `GrantLease` use `std::atomic` reps under shared lock. |
| `ExternalKvBlockIndex::entries_` | `shared_mutex` | `Register`/`Unregister`/`UnregisterByNode` exclusive; `Match` / `GetKvCount` shared. |
| `ClientRegistry::clients_` | `shared_mutex` | `RegisterClient`/`Heartbeat`/`UnregisterClient`/reaper exclusive; `IsClientAlive`/`ClientCount`/`GetAliveClients` shared. |
| `PeerDramAllocator` | `std::mutex` | One coarse mutex over `pending_` / `owned_` / `read_lease_until_` / `pending_events_`. Page bitmaps live under it too — fine-grained pages are not split out. |
| `MasterClient` | per-field | `caps_mutex_` for the cached capacities, `hb_cv_mutex_` + `hb_cv_` for the heartbeat thread, `metrics_mutex_` for the buffered samples. |

Lock ordering: master code never holds two of these mutexes at once.
`ClientRegistry::Heartbeat` releases its lock before calling into
`GlobalBlockIndex::ApplyEvents` (which acquires its own exclusive
lock). The eviction loop reads `ClientRegistry` under shared, then
calls `GlobalBlockIndex::FindEvictionCandidates` under
`GlobalBlockIndex`'s shared lock, then dispatches `EvictKey` outside
both locks.

Custom routing strategies must be thread-safe — the gRPC handler thread
pool calls `Select` concurrently. The defaults are:

- `TierPriorityRouteGetStrategy` (default get) — `thread_local
  std::mt19937` for the within-tier random pick, no mutexes.
- `RandomRouteGetStrategy` (injectable) — `thread_local std::mt19937`,
  no mutexes.
- `TierAwareMostAvailableStrategy` (default put) — stateless.

---

## 11. Observability

`include/umbp/distributed/master/master_metrics.h` defines the
`MORI_UMBP_METRIC_*` constants (counter / histogram / gauge name + help
strings) — that header is the single source of truth for metric names.
(`obs_counters.h` is unrelated: it only holds the `MORI_UMBP_OBS_*`
increment macros and the test-seam build switch.) Samples flow
through:

1. peer-side `MasterClient::AddCounter` / `SetGauge` / `Observe` →
2. buffered in `pending_counters_` / `pending_gauges_` /
   `pending_histograms_` →
3. shipped via `ReportMetrics` every
   `UMBP_METRICS_REPORT_INTERVAL_MS` (default 1000 ms) →
4. master's `MasterMetrics` aggregator →
5. Prometheus exposition on `MasterServerConfig::metrics_port`
   (default 9091 via `bin/master_main.cpp`).

Built-in metric families include:

- per-client `route_get` / `route_put` / `batch_route_get` /
  `batch_route_put` counters and bandwidth histograms,
- per-client capacity gauges
  `mori_umbp_client_capacity_{total,available,used}_bytes` +
  `mori_umbp_client_capacity_utilization_ratio`, each labeled
  `node=<id>, tier=<HBM|DRAM|SSD>` — **SSD capacity is reported here
  with `tier="SSD"`** (upper-cased, matching `TierTypeName`), not via a
  separate metric; plus the `mori_umbp_client_count` master gauge,
- `mori_umbp_heartbeat_events_applied_total` /
  `mori_umbp_heartbeat_seq_gap_total` master counters,
- `mori_umbp_external_kv_*` counters/gauges for the external-KV index.

SSD-tier families are reported by the owner peer (`node=<id>`) via
`ReportMetrics` — names per `master_metrics.h`:

- copy-on-commit: `mori_umbp_ssd_copy_enqueued_total`,
  `mori_umbp_ssd_copy_succeeded_total`,
  `mori_umbp_ssd_copy_failed_total`,
  `mori_umbp_ssd_copy_dropped_total` (`reason=queue_full|stopped`),
  and `mori_umbp_ssd_copy_bytes_total`,
- reads: `mori_umbp_ssd_read_total`
  (`status=ok|not_found|no_slot|size_too_large|error`),
  `mori_umbp_ssd_read_bytes_total`, and the reader-side
  `mori_umbp_ssd_read_client_transient_total`,
- peer-local eviction: `mori_umbp_ssd_eviction_rounds_total`,
  `mori_umbp_ssd_eviction_victims_total`,
  `mori_umbp_ssd_eviction_bytes_freed_total`,
  `mori_umbp_ssd_eviction_backend_failed_total`,
- read staging: `mori_umbp_ssd_staging_slots_in_use`,
  `mori_umbp_ssd_staging_expired_reclaims_total`,
  `mori_umbp_ssd_staging_slot_full_rejects_total`.

Set `MORI_UMBP_LOG_LEVEL=DEBUG` (or `UMBP_LOG_LEVEL=0`) to surface
per-RPC traces from `MORI_UMBP_INFO` / `MORI_UMBP_DEBUG`.

---

## 12. Usage

### 12.1 Run the master

```bash
# Defaults: 0.0.0.0:50051 gRPC, 9091 metrics, all UMBP_* env knobs honored.
./build/src/umbp/umbp_master

# Override the listen address; metrics on 9099.
./build/src/umbp/umbp_master 127.0.0.1:15558 9099

# Tighter heartbeat windows.
UMBP_HEARTBEAT_TTL_SEC=5 UMBP_REAPER_INTERVAL_SEC=2 \
  ./build/src/umbp/umbp_master
```

### 12.2 C++ client (peer-side)

```cpp
#include "umbp/umbp_client.h"

mori::umbp::UMBPConfig cfg;
cfg.dram.capacity_bytes = 16ULL * 1024 * 1024 * 1024;
cfg.distributed.emplace();
cfg.distributed->master_config.master_address = "127.0.0.1:15558";
cfg.distributed->master_config.node_id        = "worker-0";
cfg.distributed->master_config.node_address   = "10.0.0.5";
cfg.distributed->io_engine.host               = "127.0.0.1";
cfg.distributed->io_engine.port               = 16000;
cfg.distributed->peer_service_port            = 17000;

auto client = mori::umbp::CreateUMBPClient(cfg);   // -> DistributedClient

// Hot path
client->Put(key, src_ptr, size);
client->Get(key, dst_ptr, size);

// Zero-copy: pin the caller-owned buffer once, reuse forever.
client->RegisterMemory(buf_ptr, buf_size);
client->Put(key, buf_ptr, n);
client->Get(key, buf_ptr, m);
```

Drop the `cfg.distributed` line and you get a `StandaloneClient` over
local DRAM + SSD only.

### 12.3 Python: read-only master query client

`UMBPMasterClient` (Python) is **not** the full peer/data-plane client
— it's a thin read-only wrapper for the external-KV index. See
`docs/api/umbp.rst` for the full API and
`examples/umbp/umbp_master_client_demo.py` for an end-to-end script.

```python
from mori.cpp import UMBPMasterClient, UMBPTierType

client = UMBPMasterClient("localhost:15558",
                          node_id="worker-0",
                          node_address="worker-0:8080")
client.register_self({UMBPTierType.DRAM: (1<<30, 1<<30)})
client.report_external_kv_blocks("worker-0", ["sha-abc"], UMBPTierType.DRAM)
matches = client.match_external_kv(["sha-abc"])
client.unregister_self()
```

### 12.4 SGLang / hicache integration

The SGLang prefill-decode disaggregation benchmark in
`MORI-UMBP-PD-BENCHMARK.md` is the canonical end-to-end recipe. The
script wiring sets these envs on every node:

```
UMBP_MASTER_ADDRESS=<primary-prefill-ip>:15558
UMBP_MASTER_AUTO_START=true|false       # true on primary prefill, false elsewhere
UMBP_MASTER_BIN=<path>/build/src/umbp/umbp_master
UMBP_NODE_ADDRESS=<this-node-ip>
UMBP_IO_ENGINE_HOST=127.0.0.1
UMBP_IO_ENGINE_PORT=16000
UMBP_PEER_SERVICE_PORT=16001
UMBP_CACHE_REMOTE_FETCHES=false         # disable for clean throughput numbers
```

The Python `mori.umbp` package auto-detects the packaged `umbp_master`
binary inside the wheel; `UMBP_MASTER_BIN` overrides that path.

---

## 13. Testing

Unit and integration tests live in `tests/cpp/umbp/`:

- `distributed/test_global_block_index.cpp` — index ApplyEvents /
  ReplaceNodeLocations / lease / metrics edge cases,
- `distributed/test_client_registry.cpp` — heartbeat seq / gap /
  full-sync / reaper expiry,
- `distributed/test_router_*.cpp` — strategies + the Router façade,
- `distributed/test_eviction_manager.cpp` — watermark-driven dispatch,
- `distributed/test_peer_dram_allocator.cpp` — pending / owned /
  read-lease / reaper sweeps,
- `distributed/test_peer_service.cpp` — `UMBPPeer` end-to-end via real
  gRPC,
- `distributed/test_pool_client_*.cpp` — full Put/Get round-trips
  including `BatchPut` / `BatchGet`,
- `distributed/test_env_time.cpp` — `UMBP_*` parser helpers,
- SSD tier (under `src/umbp/tests/`): `test_peer_ssd_manager.cpp`
  (write / read / capacity), `test_peer_ssd_eviction.cpp` (peer-local
  watermark + LRU), `test_ssd_copy_pipeline.cpp` (copy-on-commit),
  `test_peer_ssd_read_rpc.cpp` / `test_ssd_read_lease_gating.cpp`
  (`PrepareSsdRead` / lease staging), `test_tier_priority_route_get.cpp`
  (HBM > DRAM > SSD selection), `test_ssd_reliability.cpp` (end-to-end),
- Python: `tests/python/test_umbp_master_client.py`,
  `tests/python/test_umbp_packaging.py`.

The runner scripts under `src/umbp/scripts/` build the binary and run
the integration suites end-to-end inside Docker.

---

## 14. Migration notes from the old design

If you are reading old code or PRs, the following names / surfaces have
been removed or repurposed since the master-led era:

| Old | New |
|---|---|
| `BlockIndex` (per-key allocator state on master) | `GlobalBlockIndex` (event-driven projection only) |
| `Register` / `Unregister` / `BatchRegister` / `BatchUnregister` / `Lookup` RPCs | Removed. Use heartbeat events for membership; `RouteGet` for read-side existence. |
| `Location.location_id` (opaque allocator handle minted on master) | Removed. `Location` is `(node_id, size, tier)`; pages live on the peer. |
| `ClientRegistry::TrackKey` / `UntrackKey` (per-key ownership reverse index on master) | Removed. Per-key ownership is implicit in `GlobalBlockIndex`'s `Location.node_id`. |
| Allocation TTL reaper on master | Removed. Pending TTL lives on `PeerDramAllocator`. `UMBP_ALLOCATION_TTL_SEC` is retained as a config knob for legacy compatibility but is not consumed on the live path. |
| `client_id` / `MasterClientConfig::client_id` | Renamed to `node_id` everywhere. |

Consult `runtime-env-vars.md` for the up-to-date `UMBP_*` knob list and
`docs/api/umbp.rst` for the Python read-only `UMBPMasterClient`
surface.
