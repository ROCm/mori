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
- a **`PeerServiceServer`** (`UMBPPeer` gRPC) — exposes
  `AllocateSlot`/`CommitSlot`/`AbortSlot`/`ResolveKey`/`EvictKey` plus
  the SSD staging slot machinery,
- a **`MasterClient`** — gRPC stub + heartbeat thread that ships
  `KvEvent`s, capacity snapshots, and Prometheus samples to master.

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
│   ├── obs_counters.h                 # UMBP_METRIC_* constants
│   ├── types.h                        # TierType, Location, KvEvent, ClientRecord, ...
│   ├── master/
│   │   ├── master_server.h
│   │   ├── master_client.h
│   │   ├── client_registry.h
│   │   ├── global_block_index.h
│   │   ├── external_kv_block_index.h
│   │   ├── eviction_manager.h
│   │   └── master_metrics.h
│   ├── peer/
│   │   ├── peer_service.h             # UMBPPeer gRPC server
│   │   ├── peer_dram_allocator.h      # canonical per-node owner
│   │   └── peer_page_allocator.h      # PageBitmapAllocator
│   └── routing/
│       ├── router.h
│       ├── route_get_strategy.h       # RandomRouteGetStrategy
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

// One mutation in a peer's owned-key set, shipped via heartbeat.
struct KvEvent {
  enum class Kind : int { ADD = 0, REMOVE = 1 };
  Kind kind = Kind::ADD;
  std::string key;
  TierType tier = TierType::UNKNOWN;
  uint64_t size = 0;          // ADD only; REMOVE leaves this 0
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
};
```

---

## 4. Master-side components

### 4.1 ClientRegistry  (`include/umbp/distributed/master/client_registry.h`)

Membership ledger + heartbeat ingestion.

- `RegisterClient(node_id, node_address, tier_capacities, peer_address,
  engine_desc_bytes)` — inserts a fresh `ClientRecord` or refreshes an
  expired one; rejects when a live record with the same `node_id` is
  already present.
- `UnregisterClient(node_id)` — drops the record and clears every index
  entry that belonged to it (delegated to `GlobalBlockIndex` and
  `ExternalKvBlockIndex`).
- `Heartbeat(node_id, seq, last_acked_seq, tier_capacities, events,
  is_full_sync, out_acked_seq, out_request_full_sync)` — applies one
  heartbeat batch: replaces capacity, applies events to
  `GlobalBlockIndex` (or `ReplaceNodeLocations` on `is_full_sync`), and
  advances `last_applied_seq`. If `seq != last_applied_seq + 1` and
  not full sync, the call is rejected: `out_request_full_sync = true`
  and `out_acked_seq` echoes the stored cursor so the peer reships from
  scratch.
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

Mutators:

- `ApplyEvents(node_id, events[])` — apply a peer's batch of
  ADD/REMOVE. ADD with an existing `(node_id, tier)` replaces the
  entry's `size`. REMOVE for an unknown `(key, node_id, tier)` is a
  silent no-op.
- `ReplaceNodeLocations(node_id, adds[])` — drop every prior location
  for `node_id` and reseed from `adds`. Used on full-sync.
- `RecordAccess(key)` / `GrantLease(key, duration)` — under the shared
  lock; both `last_accessed_rep` and `lease_expiry_rep` are atomic
  reps, so neither needs an exclusive lock.

Queries:

- `Lookup(key)` / `BatchLookupExists(keys[])` / `GetMetrics(key)`.
- `FindEvictionCandidates(overloaded_node_tiers)` — returns
  `EvictionCandidate{key, location, last_accessed_at, size}` rows for
  the given `(node_id, tier)` set. Skips entries with active leases.

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

### 4.4 Router  (`include/umbp/distributed/routing/router.h`)

Stateless façade over the two index objects + `ClientRegistry`.
Dispatches to pluggable strategies; defaults are
`RandomRouteGetStrategy` and `TierAwareMostAvailableStrategy`.

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

`RoutePutStrategy::Select(alive_clients, block_size, exclude_nodes)` —
`TierAwareMostAvailableStrategy` walks `[HBM, DRAM, SSD]` in order and,
on the first tier with any node holding `>= block_size` available
capacity, picks the node with the most available bytes (load
spreading).

### 4.5 EvictionManager  (`eviction_manager.h`)

Background thread that runs on `EvictionConfig::check_interval`. On
each tick:

1. Walk `ClientRegistry` for nodes whose tier(s) have crossed
   `EvictionConfig::high_watermark` (default 0.9). Collect overloaded
   `(node_id, tier)` pairs.
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
| `RegisterClient(RegisterClientRequest)` | Membership announce. Carries `peer_address`, packed `engine_desc`, per-tier capacities, and `ssd_store_capacities`. Response includes recommended heartbeat interval and the initial `ack_seq`. |
| `UnregisterClient(UnregisterClientRequest)` | Graceful shutdown. |
| `Heartbeat(HeartbeatRequest)` | The authoritative update channel. `seq` is monotonic and peer-assigned; `last_acked_seq` echoes master's most recent ack; `events[]` is the queue since the last ack; `is_full_sync` flips the request into a complete owned-key set replay. Response has `acked_seq` and `request_full_sync` for gap recovery. |
| `RouteGet(RouteGetRequest)` | Pick a replica. Read-only against `GlobalBlockIndex`. |
| `RoutePut(RoutePutRequest)` | Pick a target node + tier. Read-only against `ClientRegistry`. |
| `BatchRouteGet`/`BatchRoutePut` | Parallel-key variants. |
| `ReportExternalKvBlocks`/`RevokeExternalKvBlocks`/`MatchExternalKv` | External (unmanaged) cache index. |
| `ReportMetrics(ReportMetricsRequest)` | Client-side counters/gauges/histograms forwarded to master's Prometheus exposition. |

There are **no** per-key `Register`/`Unregister`/`Lookup` RPCs; those
have been removed. Existence checks go through `RouteGet` (or
`MatchExternalKv` for L1/L2 lookups).

### 5.2 `UMBPPeer`  (`distributed/proto/umbp_peer.proto`)

Peer-to-peer service hosted by `PeerServiceServer` on every node that
has a DRAM/HBM tier or an SSD tier.

| RPC | Purpose |
|---|---|
| `GetPeerInfo` | First-contact hydration: packed `engine_desc`, SSD staging buffer descriptor, all DRAM/HBM `BufferMemoryDesc`s, and the tier `dram_page_size`. |
| `AllocateSlot(size, tier)` | Reserve a pending DRAM/HBM slot. Response carries `slot_id`, the `pages` it covers, `page_size`, the dedup'd `descs` for those pages, and a `pending_ttl_ms`. ENOSPC is `success=false`, not an RPC error. |
| `CommitSlot(slot_id, key)` | Move pending → owned. Queues `KvEvent{ADD, key, tier, size}`. |
| `AbortSlot(slot_id)` | Drop a pending slot. Idempotent. |
| `ResolveKey(key)` | Read-side lookup. Bumps the per-key read-lease counter (Bug #7 mitigation). Returns `pages`, `page_size`, `descs`, `size`. |
| `EvictKey(keys[])` | Master-driven eviction. Read-leased keys produce `bytes_freed=0`; everything actually freed yields a `KvEvent{REMOVE}` on the next heartbeat. |
| Batch variants | `BatchAllocateSlots`/`BatchCommitSlots`/`BatchAbortSlots`/`BatchResolveKeys`. |
| SSD slot machinery | `AllocateWriteSlot`, `CommitSsdWrite`, `PrepareSsdRead`, `ReleaseSsdLease` — preserved for the SSD tier; uses a separate dedicated staging buffer + lease ID. |

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
unordered_map<string,    deque<time_point>>    read_leases_;
vector<KvEvent>                                pending_events_;  // outbox
```

Lifecycle:

| Op | Effect |
|---|---|
| `Allocate(size, tier)` | Reserve pages, assign `slot_id`, push `PendingSlot` into `pending_`. |
| `Commit(slot_id, key)` | Pop from `pending_`, insert into `owned_`, enqueue `KvEvent{ADD}` in `pending_events_`. |
| `Abort(slot_id)` | Pop from `pending_`, free pages. Idempotent. |
| `Resolve(key)` | Look up in `owned_`, append `now()+read_lease_ttl_` to `read_leases_[key]`, return pages. |
| `Evict(keys[])` | For each key: skip if `HasActiveReadLeaseLocked`, else free pages and enqueue `KvEvent{REMOVE}`. |
| `DrainPendingEvents()` | Returns and clears `pending_events_`; called by the heartbeat thread. |
| `SnapshotOwnedKeys()` | Build the full ADD list for full-sync. |
| `TierCapacitiesSnapshot()` | Live `(total, available)` per tier; reported with every heartbeat. |
| `QueueExternalEvent(ev)` | Lets the SSD `CommitSsdWrite` path enqueue ADD/REMOVE for keys it manages — one outbox per peer. |

A reaper thread sweeps `pending_` for expired slots (`pending_ttl`) and
`read_leases_` for expired entries (`read_lease_ttl_`).

### 7.2 PeerServiceServer  (`peer/peer_service.h`)

Hosts the `UMBPPeer` gRPC service. Holds non-owning pointers to
`LocalStorageManager` (SSD), `LocalBlockIndex` (SSD key→location), and
`PeerDramAllocator` (DRAM/HBM). `dram_alloc` may be null on SSD-only
deployments — the DRAM/HBM RPCs respond with `success=false` /
`found=false` in that case while SSD staging continues to work.

`engine_desc_bytes` is the packed `mori::io::EngineDesc` that the IO
engine published when `PoolClient` constructed it. It's plumbed through
`GetPeerInfo` and the `descs` payloads so writers/readers can wire up
RDMA without a separate engine-discovery step.

### 7.3 PoolClient  (`distributed/pool_client.h`)

Glues the four subsystems on a peer process:

- `MasterClient` (gRPC stub + heartbeat + metrics threads),
- `PeerDramAllocator` (owned),
- `PeerServiceServer` (owned, started on `peer_service_port`),
- `mori::io::IOEngine` (owned, published as `engine_desc`).

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

### 7.4 DistributedClient  (`distributed/distributed_client.h`)

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
| `ssd.io` | `backend` | `IoUring` | `PThread` or `IoUring`. |
| `ssd.durability` | `mode` | `Strict` | `Strict` or `Relaxed`. |
| `eviction` | `policy` | `lru` | |
| `copy_pipeline` | `worker_threads` | 2 | Async DRAM↔SSD copy workers. |
| top-level | `ssd_backend` | `posix` | `posix` or `spdk`. |
| top-level | `spdk_*` | — | SPDK/proxy tuning (BDF, reactor mask, channels, ...). |
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
| `PeerDramAllocator` | `std::mutex` | One coarse mutex over `pending_` / `owned_` / `read_leases_` / `pending_events_`. Page bitmaps live under it too — fine-grained pages are not split out. |
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

- `RandomRouteGetStrategy` — `thread_local std::mt19937`, no mutexes.
- `TierAwareMostAvailableStrategy` — stateless.

---

## 11. Observability

`include/umbp/distributed/obs_counters.h` defines a long list of
`UMBP_METRIC_*` constants (counter / histogram / gauge name + help
strings) that flow through:

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
- per-client `capacity_total` / `capacity_avail` gauges and
  `client_count` master gauge,
- `heartbeat_events_applied_total` / `heartbeat_seq_gap_total` master
  counters,
- `external_kv_*_total` counters for the external-KV index.

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
