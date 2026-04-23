# UMBP Master Control Plane — Design Document (Prototype)

**Author:** Dev3
**Status:** Draft (Prototype-scoped)
**Scope:** Global Indexer (BlockIndex) + Router + Client Registry & Heartbeat

---

## 1. Overview

The Master Control Plane is the centralized brain of the Unified Memory/Bandwidth
Pool (UMBP). It provides three logical components, exposed as a **single gRPC
service**:

1. **Global Indexer (BlockIndex)** — An authoritative registry that tracks
   the physical location(s) of every KV cache block across the cluster. Clients
   report index changes (register/unregister) to the master; the master maintains
   a global view.

2. **Router** — Makes both read and write placement decisions:
   - **RouteGet:** Given a block key, looks up the index and picks which
     existing replica to read from (random selection).
   - **RoutePut:** Given a block key and size, picks an alive node (and tier)
     for the client to write to. Tier selection is handled internally by the
     pluggable strategy. After writing via MORI-IO, the client calls `Register`
     to index the new block.

3. **Client Registry & Heartbeat** — Manages the lifecycle of client nodes.
   Clients register themselves on startup, send periodic heartbeats to prove
   liveness, and are garbage-collected (along with all their indexed blocks)
   when they go silent.

All components are exposed via a **single gRPC service** (`UMBPMaster`) and
follow a server-client architecture where the master is the server and serving
nodes are clients.

---

## 2. Architecture

```
                        ┌──────────────────────────────────────────┐
                        │            UMBP Master Server             │
                        │                                          │
  ┌──────────┐  gRPC   │  ┌────────────────────────────────────┐  │
  │  Client  │─────────────►   ClientRegistry                  │  │
  │  (Node A)│◄────────────│   (client lifecycle + heartbeat)  │  │
  └──────────┘          │  └──────────────┬────────────────────┘  │
       │                │                 │ owns clients           │
       │  heartbeat     │  ┌──────────────▼────────────────────┐  │
       │  (periodic)    │  │   BlockIndex                       │  │
       └────────────────────►  (in-memory hashmap)              │  │
                        │  └──────────────┬────────────────────┘  │
  ┌──────────┐  gRPC   │                 │ lookup                 │
  │  Client  │─────────────┌──────────────▼────────────────────┐  │
  │  (Node B)│◄────────────│   Router                          │  │
  └──────────┘          │  │   (pluggable strategy interfaces)  │  │
                        │  └───────────────────────────────────┘  │
  ┌──────────┐  gRPC   │                                          │
  │  Client  │─────────────►  ... (same services)                  │
  │  (Node C)│◄────────────│                                       │
  └──────────┘          │                                          │
                        │  ┌───────────────────────────────────┐  │
                        │  │   Reaper (background thread)      │  │
                        │  │   scans for expired clients,      │  │
                        │  │   GCs their index entries          │  │
                        │  └───────────────────────────────────┘  │
                        └──────────────────────────────────────────┘
```

**Key design decisions:**

- **Centralized-first:** All routing decisions are made by the master. This
  simplifies the initial design and ensures global consistency.
- **Stateless clients:** Clients do not cache routing decisions. Every get-route
  request goes to the master.
- **Client-owned locations:** Every `Location` in the index is associated with
  the `node_id` that registered it. When a client dies, the master can
  efficiently purge all of its locations.
- **Single service:** All RPCs are in one `UMBPMaster` service. Split into
  multiple services later if the proto gets unwieldy.

---

## 3. BlockIndex Design (Global Indexer)

### 3.1 Purpose

BlockIndex is the authoritative global registry of **where every KV cache block
lives** in the cluster. It is the single source of truth for the Router.

The index maps an opaque `key` (a string uniquely identifying a block) to a list
of `Location` structs — one per replica. Each `Location` identifies a replica by
`(node_id, location_id, size, tier)`. The `location_id` is an opaque handle
minted by the target node during a MORI-IO write — the master never interprets
it. Physical memory addressing (segments, offsets, GPU devices) is the target
node's internal concern. This keeps the master's data model stable regardless of
how target nodes manage their memory.

### 3.2 Storage Choice: In-Memory HashMap

A single `std::unordered_map` protected by a single `std::shared_mutex`.

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| In-memory hashmap | Fastest reads/writes, no external deps, simple | Volatile (lost on restart), memory-bound | **Selected** |
| Redis | Persistent, well-known | Extra dependency, network hop, overkill for single-master | Rejected for now |
| etcd | Strong consistency, watch support | Too slow for hot-path lookups | Rejected |

**Recovery strategy:** On master restart, the index is empty. All clients must
re-register. For the prototype, this means restarting the clients too.

**Future optimization:** If the single mutex becomes a bottleneck under high
concurrency, shard the map into N buckets with per-shard locks.

### 3.3 Data Structure

```
┌────────────────────────────────────────────────────────────────┐
│                         BlockIndex                              │
│                                                                 │
│  mutex_: std::shared_mutex                                      │
│                                                                 │
│  entries_: unordered_map<string, BlockEntry>                    │
│                                                                 │
│  BlockEntry:                                                    │
│    locations : vector<Location>                                 │
│    metrics   : BlockMetrics                                     │
│      { created_at, last_accessed_at, access_count }             │
│                                                                 │
│    "keyA" → { locs: [Loc{nodeX, loc-a1}, Loc{nodeY, loc-a2}],  │
│               metrics: {created: T0, accessed: T5, count: 42} } │
│    "keyB" → { locs: [Loc{nodeX, loc-b1}],                       │
│               metrics: {created: T1, accessed: T3, count: 7} }  │
│                                                                 │
│  registry_: ClientRegistry*  (for ownership tracking)           │
└────────────────────────────────────────────────────────────────┘
```

### 3.4 Operations

#### 3.4.1 Register(node_id, key, location)

Adds a new replica location for a key. Idempotent.

```
Register(node_id, key, location):
    exclusive_lock(mutex_)
    entry& = entries_[key]          // creates BlockEntry if new

    // Initialize metrics for brand-new keys
    if entry.locations.empty():
        entry.metrics.created_at = now()
        entry.metrics.last_accessed_at = now()
        entry.metrics.access_count = 0

    // Idempotency check
    for each loc in entry.locations:
        if loc == location:
            unlock
            return

    entry.locations.push_back(location)
    unlock

    // Track ownership (outside lock to avoid inversion)
    if registry_ != nullptr:
        registry_->TrackKey(node_id, key)
```

#### 3.4.2 Unregister(node_id, key, location)

Removes a specific replica of a block. Called by clients on eviction.

```
Unregister(node_id, key, location):
    exclusive_lock(mutex_)
    it = entries_.find(key)
    if it == end:
        unlock
        return false

    locs& = it->second.locations
    original_size = locs.size()
    locs.erase(remove(locs, location))
    removed = (locs.size() < original_size)

    // Clean up empty keys to prevent memory leak
    if locs.empty():
        entries_.erase(it)

    unlock

    if removed && registry_ != nullptr:
        remaining = count locations in entries_[key] with node_id matching client
        if remaining == 0:
            registry_->UntrackKey(node_id, key)

    return removed
```

#### 3.4.3 UnregisterByNode(key, node_id)

Removes all replicas for a key belonging to a specific node. Used by the Reaper
during client garbage collection.

```
UnregisterByNode(key, node_id):
    exclusive_lock(mutex_)
    it = entries_.find(key)
    if it == end:
        unlock
        return 0

    locs& = it->second.locations
    original_size = locs.size()
    locs.erase(remove_if(locs, loc.node_id == node_id), locs.end())
    removed = original_size - locs.size()

    if locs.empty():
        entries_.erase(it)

    unlock
    return removed
```

#### 3.4.4 Lookup(key)

Returns all known replica locations for a key.

```
Lookup(key):
    shared_lock(mutex_)
    it = entries_.find(key)
    if it == end:
        unlock
        return []

    result = copy(it->second.locations)
    unlock
    return result
```

#### 3.4.5 RecordAccess(key)

Bumps access metrics for a key. Called by the Router after a successful
RouteGet. Requires an exclusive lock because it mutates the entry.

```
RecordAccess(key):
    exclusive_lock(mutex_)
    it = entries_.find(key)
    if it == end:
        unlock
        return

    it->second.metrics.last_accessed_at = now()
    it->second.metrics.access_count += 1
    unlock
```

**Note:** This takes an exclusive lock on every read, which adds contention.
Acceptable for the prototype. If it becomes a bottleneck, batch access
recording or use atomic counters outside the main lock.

#### 3.4.6 GetMetrics(key)

Returns the current metrics for a key. Used by future eviction/replication
strategies.

```
GetMetrics(key):
    shared_lock(mutex_)
    it = entries_.find(key)
    if it == end:
        unlock
        return nullopt

    result = copy(it->second.metrics)
    unlock
    return result
```

#### 3.4.7 BatchRegister(node_id, entries[])

Registers multiple (key, location) pairs in a single call. Acquires the
exclusive lock **once** for the entire batch, avoiding per-item lock overhead.
Each entry follows the same idempotency semantics as single Register.

```
BatchRegister(node_id, entries):
    // entries is a list of {key, location}

    keys_to_track = []

    exclusive_lock(mutex_)
    for each {key, location} in entries:
        entry& = entries_[key]

        // Initialize metrics for brand-new keys
        if entry.locations.empty():
            entry.metrics.created_at = now()
            entry.metrics.last_accessed_at = now()
            entry.metrics.access_count = 0

        // Idempotency check
        if location already in entry.locations:
            continue

        entry.locations.push_back(location)
        keys_to_track.append(key)
    unlock

    // Track ownership (outside lock to avoid inversion)
    if registry_ != nullptr:
        for key in keys_to_track:
            registry_->TrackKey(node_id, key)

    return keys_to_track.size()   // number of new registrations
```

#### 3.4.8 BatchUnregister(node_id, entries[])

Removes multiple (key, location) pairs in a single call. Acquires the
exclusive lock **once** for the entire batch.

```
BatchUnregister(node_id, entries):
    // entries is a list of {key, location}

    removed_count = 0
    keys_to_untrack = []

    exclusive_lock(mutex_)
    for each {key, location} in entries:
        it = entries_.find(key)
        if it == end:
            continue

        locs& = it->second.locations
        original_size = locs.size()
        locs.erase(remove(locs, location))

        if locs.size() < original_size:
            removed_count += 1

        // Clean up empty keys
        if locs.empty():
            entries_.erase(it)

        // Check if client still has locations for this key
        remaining = count locs with node_id matching node_id
        if remaining == 0:
            keys_to_untrack.append(key)
    unlock

    if registry_ != nullptr:
        for key in keys_to_untrack:
            registry_->UntrackKey(node_id, key)

    return removed_count
```

### 3.5 Edge Cases

| Scenario | Behavior |
|----------|----------|
| Register same (key, location) twice | No-op (idempotent) |
| Register same key from different clients | Both locations stored |
| Unregister a location that doesn't exist | Returns false |
| Unregister the last replica for a key | Key is erased from map |
| Lookup a key that was never registered | Returns empty vector |
| Concurrent Register + Lookup on same key | Serialized by mutex |
| Register first location for a key | Initializes created_at and metrics |
| Register additional replica for existing key | Metrics unchanged |
| RecordAccess on unknown key | No-op |
| GetMetrics on unknown key | Returns nullopt |
| BatchRegister with empty entries list | No-op, returns 0 |
| BatchRegister with duplicate entries in same batch | Each processed once (idempotent) |
| BatchRegister with mix of new and existing keys | New keys get metrics initialized; existing keys append location |
| BatchUnregister with empty entries list | No-op, returns 0 |
| BatchUnregister with entries that don't exist | Skipped silently, not counted |
| BatchUnregister removes last replica for some keys | Those keys erased from map |

---

## 4. Router Design

### 4.1 Purpose

The Router makes two symmetric decisions:

- **RouteGet:** "Given a block key, which existing replica should the client
  read from?" (consults BlockIndex)
- **RoutePut:** "Given a block key and size, which node (and tier) should the
  client write to?" (consults ClientRegistry for alive nodes; tier selection
  is internal to the strategy)

```
  Client                    Router                   BlockIndex
    │                         │                         │
    │  RouteGet(key, client) │                         │
    │────────────────────────►│                         │
    │                         │  Lookup(key)            │
    │                         │────────────────────────►│
    │                         │  locations[]            │
    │                         │◄────────────────────────│
    │                         │                         │
    │                         │  random_choice(locs)    │
    │                         │                         │
    │  RouteGetResponse       │                         │
    │◄────────────────────────│                         │
```

### 4.2 Strategy Interfaces

The Router delegates selection logic to pluggable strategy objects. Each
strategy is an abstract base class with a single `Select` method. The Router
owns a `RouteGetStrategy` and a `RoutePutStrategy`, both injected at
construction time.

```
  Router
    │
    ├── RouteGetStrategy*  ──► RandomRouteGetStrategy (default)
    │                          (or any user-provided implementation)
    │
    └── RoutePutStrategy*  ──► TierAwareMostAvailableStrategy (default)
                               (or any user-provided implementation)
```

**RouteGetStrategy** — picks which existing replica to read from:

```
class RouteGetStrategy {
  virtual ~RouteGetStrategy() = default;
  virtual Location Select(const vector<Location>& locations,
                          const string& node_id) = 0;
};
```

**RoutePutStrategy** — picks which alive node to write to:

```
class RoutePutStrategy {
  virtual ~RoutePutStrategy() = default;
  virtual optional<RoutePutResult> Select(
      const vector<ClientRecord>& alive_clients,
      uint64_t block_size) = 0;
};
```

Tier selection is entirely the strategy's internal concern — the application
only specifies "store this block" and the strategy decides where.

Developers implement one or both interfaces and inject them into the Router.

### 4.3 RouteGet — Implementation (RandomRouteGetStrategy)

The default `RouteGetStrategy`. Picks a replica uniformly at random.

```
RouteGet(key, node_id):
    locations = index_.Lookup(key)

    if locations.empty():
        return std::nullopt

    selected = get_strategy_->Select(locations, node_id)
    index_.RecordAccess(key)    // bump last_accessed_at + access_count
    return selected
```

```
// RandomRouteGetStrategy::Select
Select(locations, node_id):
    if locations.size() == 1:
        return locations[0]          // fast path, no RNG needed

    thread_local std::mt19937 rng{std::random_device{}()};
    auto dist = uniform_int_distribution(0, locations.size() - 1);
    return locations[dist(rng)]
```

Using `thread_local` RNG eliminates any mutex contention.

### 4.4 RoutePut — Flow

```
  Client                    Router                   ClientRegistry
    │                         │                         │
    │  RoutePut(key, client, │                         │
    │    block_size)          │                         │
    │────────────────────────►│                         │
    │                         │  GetAliveClients()      │
    │                         │────────────────────────►│
    │                         │  alive_clients[]        │
    │                         │◄────────────────────────│
    │                         │                         │
    │                         │  put_strategy_->Select  │
    │                         │    (alive_clients,      │
    │                         │     block_size)         │
    │                         │                         │
    │  RoutePutResponse       │                         │
    │◄────────────────────────│                         │
    │                         │                         │
    │  [Client writes data    │                         │
    │   to target node via    │                         │
    │   MORI-IO, then calls   │                         │
    │   Register to index it] │                         │
```

### 4.5 RoutePut — Implementation (TierAwareMostAvailableStrategy)

The default `RoutePutStrategy`. Fastest-tier-first, most-available-space.

1. Try tiers from fastest to slowest: HBM → DRAM → SSD.
2. On each tier, find all nodes with enough available space.
3. Pick the node with the **most available space** (spread writes across nodes
   to avoid hotspots and balance memory pressure).

```
RoutePut(key, node_id, block_size):
    alive_clients = registry_.GetAliveClients()

    if alive_clients.empty():
        return std::nullopt

    return put_strategy_->Select(alive_clients, block_size)
```

```
// TierAwareMostAvailableStrategy::Select
TIER_ORDER = [HBM, DRAM, SSD]

Select(alive_clients, block_size):
    for tier in TIER_ORDER:
        candidates = []
        for c in alive_clients:
            cap = c.tier_capacities[tier]
            if cap exists && cap.available_bytes >= block_size:
                candidates.append((c, cap.available_bytes))

        if candidates.empty():
            continue    // try the next (slower) tier

        // Pick the node with the most available space
        // (spreads load, avoids filling any single node too fast)
        target = max(candidates, key=available_bytes)

        return {node_id: target.node_id,
                node_address: target.node_address,
                tier: tier}

    return nullopt
```

**Why most available space?** Spreading writes to the least-loaded node
balances memory pressure across the cluster, avoids creating hotspots on
nearly-full nodes, and reduces the chance of a write failing due to a race
between RoutePut and the actual MORI-IO write.

The master does **not** allocate memory on the target node. It only picks
which node to write to. The actual memory allocation happens on the target
node when the client connects via MORI-IO. The target node mints an opaque
`location_id` that encodes its internal addressing (segment, offset, GPU
device, etc.) and returns it to the client. After the write succeeds, the
client calls `Register(node_id, key, location)` with a `Location`
containing this `location_id`.

**Capacity tracking:** Clients report per-tier capacity on every heartbeat
(see Section 5.6). This allows the master to filter out nodes that are full
on the requested tier. The values are best-effort snapshots — races are
possible (e.g., another write lands between RoutePut and the actual MORI-IO
write), but the target node will reject the write if truly full, and the
client can retry.

### 4.6 Writing a Custom Strategy

To plug in a custom strategy, implement the abstract interface and pass it
to the Router at construction time. Example:

```cpp
// Custom strategy: always pick the node closest to the requesting client
class LocalityAwareGetStrategy : public RouteGetStrategy {
   public:
    Location Select(const std::vector<Location>& locations,
                    const std::string& node_id) override {
        // prefer replica on the same node as the requester
        for (const auto& loc : locations) {
            if (loc.node_id == node_id) return loc;
        }
        // fallback: first in list
        return locations[0];
    }
};

// Inject into Router:
auto get_strategy = std::make_unique<LocalityAwareGetStrategy>();
Router router(index, registry,
              std::move(get_strategy),
              std::make_unique<TierAwareMostAvailableStrategy>());
```

### 4.7 Router Does Not Filter by Client Liveness

The Router does **not** check whether a location's node is alive before returning
it. Stale location cleanup is the Reaper's job. If a client gets a stale
location, the data transfer fails and the client can retry.

### 4.8 Edge Cases

| Scenario | Behavior |
|----------|----------|
| **RouteGet:** Key not in index | Returns `found=false` |
| **RouteGet:** Key has exactly 1 replica | Returns that replica (no RNG) |
| **RouteGet:** Key has N replicas (N > 1) | Returns one at random |
| **RouteGet:** All replicas on dead nodes | Returns stale location; client retries |
| **RoutePut:** No alive clients | Returns `found=false` |
| **RoutePut:** No client has enough capacity on any tier | Returns `found=false` |
| **RoutePut:** HBM has space on some nodes | Strategy picks HBM node (fastest tier first) |
| **RoutePut:** No HBM space, DRAM has space | Strategy falls through to DRAM |
| **RoutePut:** Multiple nodes with HBM capacity | Strategy picks the one with most available HBM (load spreading) |
| **RoutePut:** Two nodes with identical available space | Either may be chosen (tie-break not required) |
| **RoutePut:** Target node fills up before MORI-IO write | Write fails; client retries with new RoutePut |

---

## 5. Client Registry & Heartbeat Design

### 5.1 Problem Statement

Without lifecycle tracking, the index accumulates **stale locations** — routing
clients to dead nodes. We need:
1. A way for clients to **announce** themselves (registration).
2. A way for the master to **detect** that a client is gone (heartbeat + TTL).
3. A way to **clean up** stale index entries (garbage collection).

### 5.2 Client Lifecycle State Machine

```
                RegisterClient()
  ┌─────────┐ ────────────────────► ┌──────────┐
  │  UNKNOWN │                       │  ALIVE   │◄──┐
  └─────────┘                       └──────┬───┘   │
                                           │       │ Heartbeat()
                                           │       │ (resets TTL)
                                           └───────┘
                                           │
                                           │ heartbeat TTL expires
                                           ▼
                                    ┌──────────┐
                                    │  EXPIRED │
                                    └──────┬───┘
                                           │
                                           │ Reaper GC pass
                                           ▼
                                    ┌──────────┐
                                    │  REMOVED │  (client record + all its
                                    └──────────┘   index entries deleted)
```

### 5.3 ClientRegistry Data Structure

```
┌──────────────────────────────────────────────────────────────┐
│                      ClientRegistry                           │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ clients_: map<node_id, ClientRecord>               │    │
│  │           (protected by std::shared_mutex)            │    │
│  │                                                       │    │
│  │  ClientRecord:                                        │    │
│  │  ┌─────────────────────────────────────────────────┐  │    │
│  │  │ node_id      : string                         │  │    │
│  │  │ node_address   : string                         │  │    │
│  │  │ status         : ALIVE | EXPIRED                │  │    │
│  │  │ last_heartbeat : time_point                     │  │    │
│  │  │ registered_at  : time_point                     │  │    │
│  │  │ tier_capacities: map<TierType, TierCapacity>    │  │    │
│  │  │   e.g. { HBM  → {total: 80GB, avail: 32GB},   │  │    │
│  │  │          DRAM → {total: 512GB, avail: 200GB},  │  │    │
│  │  │          SSD  → {total: 4TB, avail: 3TB} }     │  │    │
│  │  └─────────────────────────────────────────────────┘  │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
│  Reverse index (for GC):                                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ client_keys_: map<node_id, set<string>>            │    │
│  │              tracks which block keys each client owns │    │
│  │              (protected by same mutex as clients_)    │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
│  Config:                                                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ heartbeat_ttl       : duration  (default: 10s)       │    │
│  │ reaper_interval     : duration  (default: 5s)        │    │
│  │ max_missed_heartbeats: uint32   (default: 3)         │    │
│  │                                                       │    │
│  │ Effective expiry = heartbeat_ttl * max_missed_beats   │    │
│  │                   = 10s * 3 = 30s default             │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

Per-tier capacity (`tier_capacities`) is used by RoutePut to filter out nodes
that lack space on the requested tier. Clients report all tier capacities at
registration and update them on every heartbeat.

### 5.4 How Client Registration Interacts with BlockIndex

When a client calls `Register` to index a block, the master records
`(node_id → key)` in `client_keys_`:

```
  BlockIndex::Register(node_id, key, location)
      │
      ├──► BlockIndex: entries_[key].push_back(location)
      │
      └──► ClientRegistry: client_keys_[node_id].insert(key)
```

When a client dies (heartbeat expired + reaper runs):

```
  Reaper detects node_id expired
      │
      ├──► keys = client_keys_[node_id]
      │
      ├──► for each key in keys:
      │        BlockIndex::UnregisterByNode(key, node_id)
      │
      ├──► client_keys_.erase(node_id)
      │
      └──► clients_.erase(node_id)
```

### 5.5 Reaper: Background Garbage Collection

The Reaper is a background thread that periodically scans for expired clients
and purges their data.

```
  Reaper thread (runs every reaper_interval):
      │
      │  write_lock(mutex_)
      │  for each (node_id, record) in clients_:
      │      if now() - record.last_heartbeat > heartbeat_ttl * max_missed:
      │          keys = client_keys_[node_id]
      │          for each key in keys:
      │              index_.UnregisterByNode(key, node_id)
      │          client_keys_.erase(node_id)
      │          clients_.erase(node_id)
      │          LOG(WARNING) << "Reaped dead client: " << node_id
      │  unlock
```

Simplified from the original two-phase (read-lock scan → write-lock re-check)
design. A single write lock pass is sufficient for a prototype with few clients.

### 5.6 Heartbeat Protocol

The heartbeat is a lightweight unary gRPC call. The client sends it periodically
(recommended interval: `heartbeat_ttl / 2`).

```
  Client                              Master
    │                                   │
    │  Heartbeat(node_id,            │
    │    tier_capacities)              │
    │──────────────────────────────────►│
    │                                   │  record.last_heartbeat = now()
    │                                   │  record.tier_capacities = updated
    │                                   │
    │  HeartbeatResponse(status)        │
    │◄──────────────────────────────────│
    │                                   │
    │  [repeat every heartbeat_interval]│
```

The client reports per-tier capacity on every heartbeat, keeping the master's
view reasonably fresh for RoutePut placement decisions.

If the client is unknown (not registered), the heartbeat returns
`CLIENT_STATUS_UNKNOWN`. The client should call `RegisterClient` first.

**Deferred:** The `REREGISTER` action for master restart recovery. For the
prototype, if the master restarts, restart clients too.

### 5.7 Graceful Client Shutdown

When a client shuts down cleanly, it calls `UnregisterClient`. The master
immediately removes the client and all its index entries.

```
  Client (shutting down)              Master
    │                                   │
    │  UnregisterClient(node_id)     │
    │──────────────────────────────────►│
    │                                   │  keys = client_keys_[node_id]
    │                                   │  for each key: BlockIndex.UnregisterByNode(...)
    │                                   │  client_keys_.erase(node_id)
    │                                   │  clients_.erase(node_id)
    │                                   │
    │  UnregisterClientResponse(count)  │
    │◄──────────────────────────────────│
```

---

## 6. gRPC Service Contract

All RPCs are in a single `UMBPMaster` service. This can be split later if needed.

### 6.1 Proto Definition

```protobuf
syntax = "proto3";
package umbp;

// ============================================================
//  Core types
// ============================================================

enum TierType {
  TIER_UNKNOWN = 0;
  TIER_HBM     = 1;   // GPU High Bandwidth Memory
  TIER_DRAM    = 2;   // CPU DRAM
  TIER_SSD     = 3;   // NVMe SSD
}

// Per-tier capacity reported by a client node.
message TierCapacity {
  TierType tier                     = 1;
  uint64   total_capacity_bytes     = 2;   // Total capacity on this tier
  uint64   available_capacity_bytes = 3;   // Current free capacity on this tier
}

// Physical location of a single replica of a block.
// The master treats location_id as an opaque handle — physical memory
// addressing (segment, offset, GPU device, etc.) is the target node's
// internal concern.  The target node mints the location_id during a
// MORI-IO write and the client passes it back when calling Register.
message Location {
  string   node_id     = 1;   // Node identifier (e.g. "host1:50052")
  string   location_id = 2;   // Opaque handle minted by target node
  uint64   size        = 3;   // Block size in bytes
  TierType tier        = 4;   // Storage tier
}

// ============================================================
//  Client lifecycle
// ============================================================

enum ClientStatus {
  CLIENT_STATUS_UNKNOWN = 0;
  CLIENT_STATUS_ALIVE   = 1;
  CLIENT_STATUS_EXPIRED = 2;
}

message RegisterClientRequest {
  string node_id                    = 1;
  string node_address                 = 2;   // Network address (e.g. "10.0.1.5:8080")
  repeated TierCapacity tier_capacities = 3; // Capacity per tier (HBM, DRAM, SSD)
}
message RegisterClientResponse {
  uint64 heartbeat_interval_ms = 1;  // Recommended heartbeat interval
}

message UnregisterClientRequest {
  string node_id = 1;
}
message UnregisterClientResponse {
  uint32 keys_removed = 1;
}

message HeartbeatRequest {
  string node_id                    = 1;
  repeated TierCapacity tier_capacities = 2; // Updated capacity per tier
}
message HeartbeatResponse {
  ClientStatus status = 1;
}

// ============================================================
//  Block index
// ============================================================

message RegisterRequest {
  string   node_id = 1;
  string   key       = 2;
  Location location  = 3;
}
message RegisterResponse {}

message UnregisterRequest {
  string   node_id = 1;
  string   key       = 2;
  Location location  = 3;
}
message UnregisterResponse {
  uint32 removed_count = 1;
}

// Single entry for batch operations.
message BlockEntry {
  string   key      = 1;
  Location location = 2;
}

message BatchRegisterRequest {
  string node_id           = 1;
  repeated BlockEntry entries = 2;
}
message BatchRegisterResponse {
  uint32 registered_count = 1;  // Number of new registrations (excludes duplicates)
}

message BatchUnregisterRequest {
  string node_id           = 1;
  repeated BlockEntry entries = 2;
}
message BatchUnregisterResponse {
  uint32 removed_count = 1;
}

message LookupRequest {
  string key = 1;
}
message LookupResponse {
  repeated Location locations = 1;
}

// ============================================================
//  Router
// ============================================================

message RouteGetRequest {
  string key       = 1;
  string node_id = 2;   // Requesting node (for future locality hints)
}
message RouteGetResponse {
  bool     found  = 1;
  Location source = 2;    // Selected replica to read from (if found)
}

// Request the master to pick a target node for writing a block.
// After receiving the response, the client writes via MORI-IO and
// then calls Register to index the new block.
message RoutePutRequest {
  string   key        = 1;   // Block key to store
  string   node_id  = 2;   // Requesting client
  uint64   block_size = 3;   // Size of the block to write (bytes)
}
message RoutePutResponse {
  bool     found        = 1;   // Whether a suitable target node was found
  string   node_id      = 2;   // Target node ID (use in Location.node_id on Register)
  string   node_address = 3;   // Target node network address (for MORI-IO connection)
  TierType tier         = 4;   // Tier selected by the strategy (for MORI-IO allocation)
}

// ============================================================
//  Single unified service
// ============================================================

service UMBPMaster {
  // Client lifecycle
  rpc RegisterClient(RegisterClientRequest) returns (RegisterClientResponse);
  rpc UnregisterClient(UnregisterClientRequest) returns (UnregisterClientResponse);
  rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);

  // Block index
  rpc Register(RegisterRequest) returns (RegisterResponse);
  rpc Unregister(UnregisterRequest) returns (UnregisterResponse);
  rpc BatchRegister(BatchRegisterRequest) returns (BatchRegisterResponse);
  rpc BatchUnregister(BatchUnregisterRequest) returns (BatchUnregisterResponse);
  rpc Lookup(LookupRequest) returns (LookupResponse);

  // Router
  rpc RouteGet(RouteGetRequest) returns (RouteGetResponse);
  rpc RoutePut(RoutePutRequest) returns (RoutePutResponse);
}
```

### 6.2 Contract Summary

| RPC | Purpose | Request | Response |
|-----|---------|---------|----------|
| RegisterClient | Register a new client node | node_id + address + tier_capacities | heartbeat interval |
| UnregisterClient | Graceful client shutdown + GC | node_id | keys removed count |
| Heartbeat | Periodic liveness + capacity update | node_id + tier_capacities | status |
| Register | Add a replica location | node_id + key + location | (empty) |
| Unregister | Remove a specific replica | node_id + key + location | removed count |
| BatchRegister | Add multiple replicas in one call | node_id + entries[] | registered count |
| BatchUnregister | Remove multiple replicas in one call | node_id + entries[] | removed count |
| Lookup | Get all replicas for a key | key | list of locations |
| RouteGet | Pick where to read a block | key + node_id | found + selected location |
| RoutePut | Pick where to write a block | key + node_id + block_size | found + target node |

**Deferred RPCs** (add when needed):
- `BatchLookup` / `BatchRouteGet` — batch variants for prefetch workflows
- `Exists` — lightweight existence check (clients can use `Lookup` instead)
- `GetClientInfo` / `ListClients` — admin/debugging queries

---

## 7. C++ Class Interfaces

### 7.1 Core Types (`include/umbp/types.h`)

```cpp
#pragma once
#include <chrono>
#include <cstdint>
#include <map>
#include <string>

namespace umbp {

enum class TierType : int {
    UNKNOWN = 0,
    HBM     = 1,
    DRAM    = 2,
    SSD     = 3,
};

struct TierCapacity {
    uint64_t total_bytes = 0;
    uint64_t available_bytes = 0;
};

struct Location {
    std::string node_id;
    std::string location_id;  // Opaque handle from target node
    uint64_t    size = 0;
    TierType    tier = TierType::UNKNOWN;

    bool operator==(const Location& other) const;
};

enum class ClientStatus : int {
    UNKNOWN = 0,
    ALIVE   = 1,
    EXPIRED = 2,
};

struct BlockMetrics {
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_accessed_at;
    uint64_t access_count = 0;
};

struct ClientRecord {
    std::string node_id;
    std::string node_address;
    ClientStatus status = ClientStatus::UNKNOWN;
    std::chrono::steady_clock::time_point last_heartbeat;
    std::chrono::steady_clock::time_point registered_at;
    std::map<TierType, TierCapacity> tier_capacities;
};

}  // namespace umbp
```

### 7.2 ClientRegistry (`include/umbp/client_registry.h`)

```cpp
#pragma once
#include <atomic>
#include <chrono>
#include <shared_mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "umbp/types.h"

namespace umbp {

class BlockIndex;  // forward declaration

struct ClientRegistryConfig {
    std::chrono::seconds heartbeat_ttl{10};
    std::chrono::seconds reaper_interval{5};
    uint32_t max_missed_heartbeats = 3;
};

class ClientRegistry {
   public:
    ClientRegistry(const ClientRegistryConfig& config, BlockIndex& index);
    ~ClientRegistry();

    // Non-copyable, non-movable
    ClientRegistry(const ClientRegistry&) = delete;
    ClientRegistry& operator=(const ClientRegistry&) = delete;

    // --- Client lifecycle ---
    void RegisterClient(const std::string& node_id,
                        const std::string& node_address,
                        const std::map<TierType, TierCapacity>& tier_capacities);

    // Gracefully unregister. Returns number of block keys cleaned up.
    size_t UnregisterClient(const std::string& node_id);

    // Process heartbeat. Updates last_heartbeat and tier capacities.
    // Returns CLIENT_STATUS_UNKNOWN if client is not registered.
    ClientStatus Heartbeat(const std::string& node_id,
                           const std::map<TierType, TierCapacity>& tier_capacities);

    // --- Ownership tracking (called by BlockIndex) ---
    void TrackKey(const std::string& node_id, const std::string& key);
    void UntrackKey(const std::string& node_id, const std::string& key);

    // --- Queries ---
    bool IsClientAlive(const std::string& node_id) const;
    size_t ClientCount() const;

    // Returns all clients with status == ALIVE. Used by Router for RoutePut.
    std::vector<ClientRecord> GetAliveClients() const;

    // --- Reaper control ---
    void StartReaper();
    void StopReaper();

   private:
    ClientRegistryConfig config_;
    BlockIndex& index_;

    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, ClientRecord> clients_;
    std::unordered_map<std::string, std::set<std::string>> client_keys_;

    // Reaper thread
    std::thread reaper_thread_;
    std::atomic<bool> reaper_running_{false};

    void ReaperLoop();
    void ReapExpiredClients();

    std::chrono::seconds ExpiryDuration() const;
};

}  // namespace umbp
```

### 7.3 BlockIndex (`include/umbp/block_index.h`)

```cpp
#pragma once
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "umbp/types.h"

namespace umbp {

class ClientRegistry;  // forward declaration

// Per-key entry: replica locations + access metrics.
struct BlockEntry {
    std::vector<Location> locations;
    BlockMetrics metrics;
};

class BlockIndex {
   public:
    BlockIndex() = default;
    ~BlockIndex() = default;

    // Non-copyable, non-movable (owns lock)
    BlockIndex(const BlockIndex&) = delete;
    BlockIndex& operator=(const BlockIndex&) = delete;

    void SetClientRegistry(ClientRegistry* registry);

    // --- Mutators ---
    void Register(const std::string& node_id,
                  const std::string& key,
                  const Location& location);

    bool Unregister(const std::string& node_id,
                    const std::string& key,
                    const Location& location);

    size_t UnregisterByNode(const std::string& key,
                            const std::string& node_id);

    // Batch variants — single lock acquisition for the entire batch.
    size_t BatchRegister(const std::string& node_id,
                         const std::vector<std::pair<std::string, Location>>& entries);
    size_t BatchUnregister(const std::string& node_id,
                           const std::vector<std::pair<std::string, Location>>& entries);

    // Bump last_accessed_at and access_count. Called by Router on RouteGet.
    void RecordAccess(const std::string& key);

    // --- Queries ---
    std::vector<Location> Lookup(const std::string& key) const;

    // Returns metrics for a key, or nullopt if the key doesn't exist.
    std::optional<BlockMetrics> GetMetrics(const std::string& key) const;

   private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, BlockEntry> entries_;
    ClientRegistry* registry_ = nullptr;
};

}  // namespace umbp
```

### 7.4 RouteGetStrategy (`include/umbp/route_get_strategy.h`)

```cpp
#pragma once
#include <string>
#include <vector>

#include "umbp/types.h"

namespace umbp {

// Abstract interface for RouteGet replica selection.
// Implement this to plug in a custom read-path routing strategy.
class RouteGetStrategy {
   public:
    virtual ~RouteGetStrategy() = default;

    // Select one replica from the given non-empty locations list.
    // |node_id| is the requesting client (for locality-aware strategies).
    virtual Location Select(const std::vector<Location>& locations,
                            const std::string& node_id) = 0;
};

// Default: uniform random selection among replicas.
class RandomRouteGetStrategy : public RouteGetStrategy {
   public:
    Location Select(const std::vector<Location>& locations,
                    const std::string& node_id) override;
};

}  // namespace umbp
```

### 7.5 RoutePutStrategy (`include/umbp/route_put_strategy.h`)

```cpp
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "umbp/types.h"

namespace umbp {

struct RoutePutResult {
    std::string node_id;       // Target node's node_id
    std::string node_address;  // Target node's network address
    TierType    tier;          // Tier selected by the strategy
};

// Abstract interface for RoutePut node placement.
// Implement this to plug in a custom write-path placement strategy.
class RoutePutStrategy {
   public:
    virtual ~RoutePutStrategy() = default;

    // Select a target node from |alive_clients| that can accommodate
    // |block_size| bytes. Tier selection is the strategy's responsibility.
    // Returns nullopt if no suitable node exists.
    virtual std::optional<RoutePutResult> Select(
        const std::vector<ClientRecord>& alive_clients,
        uint64_t block_size) = 0;
};

// Default: try tiers fastest-first (HBM → DRAM → SSD), pick the node
// with the most available space on the first tier that has capacity.
class TierAwareMostAvailableStrategy : public RoutePutStrategy {
   public:
    std::optional<RoutePutResult> Select(
        const std::vector<ClientRecord>& alive_clients,
        uint64_t block_size) override;
};

}  // namespace umbp
```

### 7.6 Router (`include/umbp/router.h`)

```cpp
#pragma once
#include <memory>
#include <optional>
#include <string>

#include "umbp/block_index.h"
#include "umbp/client_registry.h"
#include "umbp/route_get_strategy.h"
#include "umbp/route_put_strategy.h"
#include "umbp/types.h"

namespace umbp {

class Router {
   public:
    // Inject strategy objects. If nullptr, defaults are created:
    //   RouteGet  → RandomRouteGetStrategy
    //   RoutePut  → TierAwareMostAvailableStrategy
    Router(BlockIndex& index,
           ClientRegistry& registry,
           std::unique_ptr<RouteGetStrategy> get_strategy = nullptr,
           std::unique_ptr<RoutePutStrategy> put_strategy = nullptr);
    ~Router() = default;

    // RouteGet: pick an existing replica to read from.
    // Returns nullopt if the key is not in the index.
    std::optional<Location> RouteGet(const std::string& key,
                                     const std::string& node_id);

    // RoutePut: pick a target node to write to.
    // Returns nullopt if no suitable node exists.
    std::optional<RoutePutResult> RoutePut(const std::string& key,
                                           const std::string& node_id,
                                           uint64_t block_size);

   private:
    BlockIndex& index_;
    ClientRegistry& registry_;
    std::unique_ptr<RouteGetStrategy> get_strategy_;
    std::unique_ptr<RoutePutStrategy> put_strategy_;
};

}  // namespace umbp
```

### 7.7 MasterServer (`include/umbp/master_server.h`)

```cpp
#pragma once
#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>

#include "umbp/block_index.h"
#include "umbp/client_registry.h"
#include "umbp/router.h"

namespace umbp {

struct MasterServerConfig {
    std::string listen_address = "0.0.0.0:50051";
    ClientRegistryConfig registry_config;

    // Optional: custom routing strategies (nullptr = use defaults)
    std::unique_ptr<RouteGetStrategy> get_strategy;
    std::unique_ptr<RoutePutStrategy> put_strategy;
};

class MasterServer {
   public:
    explicit MasterServer(const MasterServerConfig& config);
    ~MasterServer();

    // Start the gRPC server (blocks until Shutdown is called).
    void Run();

    // Gracefully shut down the server and Reaper.
    void Shutdown();

   private:
    MasterServerConfig config_;
    BlockIndex index_;
    ClientRegistry registry_;
    Router router_;

    // gRPC server and service implementation
    std::unique_ptr<grpc::Server> server_;

    // gRPC service implementation class (defined in master_server.cpp)
    class UMBPMasterServiceImpl;
    std::unique_ptr<UMBPMasterServiceImpl> service_;
};

}  // namespace umbp
```

### 7.8 Client (`client/include/umbp/client.h`)

```cpp
#pragma once
#include <atomic>
#include <grpcpp/grpcpp.h>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "umbp/types.h"

// Forward-declare generated stub
namespace umbp { class UMBPMaster; }

namespace umbp {

struct MasterClientConfig {
    std::string master_address;
    std::string node_id;
    std::string node_address;
    bool auto_heartbeat = true;
};

class MasterClient {
   public:
    explicit MasterClient(const MasterClientConfig& config);
    ~MasterClient();

    // --- Client lifecycle ---
    void RegisterSelf(const std::map<TierType, TierCapacity>& tier_capacities);
    void UnregisterSelf();

    // --- Block index ---
    void Register(const std::string& key, const Location& location);
    uint32_t Unregister(const std::string& key, const Location& location);
    uint32_t BatchRegister(const std::vector<std::pair<std::string, Location>>& entries);
    uint32_t BatchUnregister(const std::vector<std::pair<std::string, Location>>& entries);
    std::vector<Location> Lookup(const std::string& key);

    // --- Router ---
    std::optional<Location> RouteGet(const std::string& key);

    // Returns target node (node_id + node_address) to write to.
    // After writing via MORI-IO, call Register() to index the block.
    std::optional<RoutePutResult> RoutePut(const std::string& key,
                                           uint64_t block_size);

    // --- Heartbeat ---
    void StartHeartbeat();
    void StopHeartbeat();

   private:
    MasterClientConfig config_;

    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<UMBPMaster::Stub> stub_;

    std::thread heartbeat_thread_;
    std::atomic<bool> heartbeat_running_{false};
    uint64_t heartbeat_interval_ms_ = 5000;

    void HeartbeatLoop();
};

}  // namespace umbp
```

---

## 8. Data Flow

### 8.1 Client Startup

```
  Client (Node A)                    Master
      │                                │
      │  RegisterClient(id, addr,      │
      │    tier_capacities=[           │
      │      {HBM, 80G, 80G},         │
      │      {DRAM, 512G, 512G}])     │
      │───────────────────────────────►│
      │                                │  clients_[id] = {ALIVE, now(),
      │                                │    tier_capacities}
      │                                │  client_keys_[id] = {}
      │  RegisterClientResponse(       │
      │    heartbeat_interval=5000ms)  │
      │◄───────────────────────────────│
      │                                │
      │  [start heartbeat thread]      │
      │  Heartbeat(node_id,          │  (every 5s)
      │    tier_capacities=[           │
      │      {HBM, 80G, 32G},         │
      │      {DRAM, 512G, 200G}])     │
      │───────────────────────────────►│
      │                                │  record.last_heartbeat = now()
      │                                │  record.tier_capacities = updated
      │  HeartbeatResponse(ALIVE)      │
      │◄───────────────────────────────│
```

### 8.2 Register Block

```
  Client (Node A)                    Master
      │                                │
      │  Register(node_id,          │
      │    key, location)              │
      │───────────────────────────────►│
      │                                │  validate client is known
      │                                │  write_lock(mutex_)
      │                                │  entries_[key].push_back(location)
      │                                │  unlock
      │                                │  client_keys_[node_id].insert(key)
      │                                │
      │  RegisterResponse (OK)        │
      │◄───────────────────────────────│
```

### 8.3 BatchRegister / BatchUnregister

```
  Client (Node A)                    Master
      │                                │
      │  BatchRegister(node_id,     │
      │    entries=[                   │
      │      {keyA, locA},            │
      │      {keyB, locB},            │
      │      {keyC, locC}])           │
      │───────────────────────────────►│
      │                                │  exclusive_lock(mutex_) — once
      │                                │  for each entry:
      │                                │    entries_[key].push_back(loc)
      │                                │  unlock
      │                                │  track all new keys in client_keys_
      │                                │
      │  BatchRegisterResponse(       │
      │    registered_count=3)        │
      │◄───────────────────────────────│
```

```
  Client (Node A)                    Master
      │                                │
      │  BatchUnregister(node_id,   │
      │    entries=[                   │
      │      {keyA, locA},            │
      │      {keyB, locB}])           │
      │───────────────────────────────►│
      │                                │  exclusive_lock(mutex_) — once
      │                                │  for each entry:
      │                                │    remove loc from entries_[key]
      │                                │    erase key if empty
      │                                │  unlock
      │                                │  untrack orphaned keys
      │                                │
      │  BatchUnregisterResponse(     │
      │    removed_count=2)           │
      │◄───────────────────────────────│
```

### 8.4 RouteGet

```
  Client (Node B)                    Master
      │                                │
      │   RouteGet(key, node_id)     │
      │───────────────────────────────►│
      │                                │  locations = index.Lookup(key)
      │                                │  if empty → return not_found
      │                                │  selected = strategy.Select(locations)
      │                                │  index.RecordAccess(key)
      │                                │
      │   RouteGetResponse(selected)   │
      │◄───────────────────────────────│
      │                                │
      │   [Client reads data directly  │
      │    from selected.node_id via   │
      │    RDMA / MORI-IO]             │
```

### 8.5 RoutePut (write path)

```
  Client (Node B)                    Master
      │                                │
      │   RoutePut(key, node_id,     │
      │     block_size=4096)           │
      │───────────────────────────────►│
      │                                │  alive = registry.GetAliveClients()
      │                                │  strategy selects target node
      │                                │  (tier choice is internal)
      │                                │
      │   RoutePutResponse(            │
      │     node_id, node_address)     │
      │◄───────────────────────────────│
      │                                │
      │   [Client writes block to      │
      │    target node via MORI-IO.    │
      │    Target node mints           │
      │    location_id internally.]    │
      │                                │
      │   Register(node_id, key,     │
      │     Location{node_id,          │
      │       location_id, size, tier})│
      │───────────────────────────────►│
      │                                │  index block in BlockIndex
      │                                │  track ownership in client_keys_
      │   RegisterResponse (OK)        │
      │◄───────────────────────────────│
```

### 8.6 Client Death (Reaper GC)

```
  Client (Node A)         Reaper (bg thread)         Master state
      │                        │                        │
      │  [crash]               │                        │
      ✕                        │                        │
                               │                        │
       ... 30s pass (3 × 10s TTL) ...                   │
                               │                        │
                               │  write_lock            │
                               │  Node A: expired       │
                               │  for each key:         │
                               │    UnregisterByNode    │
                               │  erase client record   │
                               │  unlock                │
                               │                        │
                               │  LOG(WARN) "Reaped     │
                               │   dead client: node-a" │
```

### 8.7 Graceful Client Shutdown

```
  Client (Node A)                    Master
      │                                │
      │  [shutting down]               │
      │  StopHeartbeat()               │
      │                                │
      │  UnregisterClient(node_id)   │
      │───────────────────────────────►│
      │                                │  keys = client_keys_[node_id]
      │                                │  for each key:
      │                                │      index.UnregisterByNode(key, id)
      │                                │  erase client_keys_[node_id]
      │                                │  erase clients_[node_id]
      │                                │
      │  UnregisterClientResponse(     │
      │    keys_removed=42)            │
      │◄───────────────────────────────│
```

---

## 9. Concurrency Design

### 9.1 BlockIndex Thread Safety

- Single `std::shared_mutex` protecting `entries_`.
  - `Lookup`, `GetMetrics`, `BatchLookupExists`, `RecordAccess`: shared (read) lock
  - `Register`, `Unregister`, `UnregisterByNode`: exclusive (write) lock
  - `BatchRegister`, `BatchUnregister`: exclusive (write) lock — held for
    the entire batch to avoid per-item lock/unlock overhead

- **`RecordAccess` is read-only at the lock layer:** as of the #1 fix it
  takes a shared lock and updates `last_accessed_rep` / `atomic_access_count`
  via `std::atomic` stores (`global_block_index.cpp::RecordAccess`,
  `BlockEntry::RecordAccessAtomic`). Earlier drafts of this section
  described `RecordAccess` as exclusive; that is now stale.

- **Batch lock duration:** A large batch (e.g., 1000 entries) holds the exclusive
  lock for longer than a single Register call. This blocks all concurrent Lookup
  and RecordAccess calls for the duration. Acceptable for the prototype — batch
  calls are infrequent (e.g., client startup, bulk eviction). If this becomes a
  concern, process the batch in fixed-size chunks (e.g., 100 entries per lock
  acquisition).

### 9.2 ClientRegistry Thread Safety

- Single `std::shared_mutex` protecting `clients_` and `client_keys_`.
  - `Heartbeat`, `IsClientAlive`: shared (read) lock
  - `RegisterClient`, `UnregisterClient`, Reaper GC: exclusive (write) lock

### 9.3 Lock Ordering

**Principle (stable):** No path may simultaneously hold both the
`ClientRegistry` mutex and the `BlockIndex` mutex in a way that creates a
cycle. Concretely: any code that needs to touch both must release the first
lock before acquiring the second. This rules out the only deadlock geometry
between the two subsystems regardless of which direction a given path goes.

**Current implementation snapshot:**

- `BlockIndex.{Register, Unregister, UnregisterByNode, BatchRegister,
  BatchUnregister, EvictEntries}` acquire the BlockIndex lock, mutate
  `entries_`, **release** the BlockIndex lock, then call
  `registry_->TrackKey()` / `UntrackKey()`.
- `ClientRegistry.{TrackKey, UntrackKey}` (post #6 fix) run in three phases:

  ```
  Phase 1: registry shared_lock → load `idx = index_`, check `clients_.find`, release
  Phase 2: no registry lock     → idx->Lookup(key) (acquires BlockIndex shared lock internally)
  Phase 3: registry unique_lock → re-check clients_.find, mutate client_keys_
  ```

  Phase 1's shared_lock pairs with `SetBlockIndex`'s unique_lock so the
  `index_` pointer is published safely; Phase 3's re-check closes the
  unregister race window opened by Phase 2.
- `ClientRegistry.{UnregisterClient, ReapExpiredClients,
  FinalizeAllocation, PublishLocalBlock}` acquire the registry unique_lock,
  mutate registry-side state, **release** the registry lock, then call
  `index_->{UnregisterByNode, Register}`.
- `Heartbeat` / `AllocateForPut` only touch the registry unique_lock; they
  never reach the `index_`.

No path acquires both locks at once. If the future B-track refactor
(`client_keys_` independent mutex, see distributed-known-issues.md follow-ups)
lands, only this snapshot needs to be updated; the principle above stays.

### 9.4 Router Thread Safety

The Router itself has no shared mutable state — it delegates to strategy objects.
The built-in strategies are thread-safe:
- `RandomRouteGetStrategy` uses `thread_local std::mt19937` — no contention.
- `TierAwareMostAvailableStrategy` is stateless (operates only on its arguments).

Custom strategies **must** be thread-safe, as the Router may call `Select` from
multiple gRPC handler threads concurrently.

### 9.5 gRPC Threading

gRPC uses its own thread pool. All BlockIndex, ClientRegistry, and Router
methods are safe to call from multiple gRPC handler threads concurrently.

---

## 10. Project Structure

```
umbp/
├── CMakeLists.txt                     # Top-level build
├── proto/
│   └── umbp.proto                     # gRPC + protobuf definitions
├── include/umbp/
│   ├── types.h                        # TierType, Location, ClientRecord
│   ├── block_index.h                  # BlockIndex class
│   ├── client_registry.h             # ClientRegistry + Reaper
│   ├── route_get_strategy.h          # RouteGetStrategy interface + RandomRouteGetStrategy
│   ├── route_put_strategy.h          # RoutePutStrategy interface + TierAwareMostAvailableStrategy
│   ├── router.h                       # Router class (uses strategy interfaces)
│   └── master_server.h               # MasterServer (gRPC server)
├── src/
│   ├── CMakeLists.txt
│   ├── block_index.cpp
│   ├── client_registry.cpp
│   ├── route_get_strategy.cpp         # RandomRouteGetStrategy implementation
│   ├── route_put_strategy.cpp         # TierAwareMostAvailableStrategy implementation
│   ├── router.cpp
│   ├── master_server.cpp              # gRPC service implementation
│   └── main.cpp                       # Master binary entry point
├── client/
│   ├── include/umbp/
│   │   └── client.h                   # MasterClient wrapper
│   └── src/
│       └── client.cpp
├── tests/
│   ├── CMakeLists.txt
│   ├── block_index_test.cpp
│   ├── client_registry_test.cpp
│   ├── route_get_strategy_test.cpp
│   ├── route_put_strategy_test.cpp
│   ├── router_test.cpp
│   └── integration_test.cpp
└── docs/
    └── design-master-control-plane.md  # This document
```

---

## 11. Build Dependencies

| Dependency | Purpose | Version |
|------------|---------|---------|
| CMake | Build system | >= 3.16 |
| gRPC | RPC framework | >= 1.50 |
| Protobuf | Serialization / code generation | >= 3.21 |
| glog | Logging | >= 0.6 |
| Google Test | Unit testing | >= 1.12 |

---

## 12. Testing Strategy

### 12.1 Unit Tests

**`block_index_test.cpp`:**
- Register single key + location, verify Lookup returns it
- Register multiple locations for same key, verify all returned
- Register same (key, location) twice — verify idempotent
- Unregister specific replica, verify remaining replicas unchanged
- Unregister a location that doesn't exist — returns false
- Unregister last replica for a key — key disappears
- UnregisterByNode — only removes locations matching node_id
- Lookup unknown key — returns empty vector
- Register first location for a key — initializes created_at and access_count=0
- Register second location for same key — metrics unchanged (created_at preserved)
- RecordAccess bumps access_count and updates last_accessed_at
- RecordAccess on unknown key — no-op, no crash
- GetMetrics returns current metrics; GetMetrics on unknown key returns nullopt
- BatchRegister multiple entries — all appear in Lookup
- BatchRegister with duplicate entries in same batch — idempotent
- BatchRegister with mix of new and existing keys — metrics initialized only for new keys
- BatchRegister empty entries list — returns 0, no-op
- BatchUnregister multiple entries — all removed from Lookup
- BatchUnregister entries that don't exist — skipped, returns 0
- BatchUnregister last replica for some keys — those keys erased
- BatchUnregister empty entries list — returns 0, no-op
- Concurrent BatchRegister/Lookup from multiple threads (stress test)

**`client_registry_test.cpp`:**
- RegisterClient, verify IsClientAlive returns true
- RegisterClient same node_id twice — refreshes record
- Heartbeat updates last_heartbeat time
- Heartbeat for unknown client returns CLIENT_STATUS_UNKNOWN
- UnregisterClient removes client and its keys
- Reaper marks expired clients and GCs their index entries
- Reaper does NOT reap clients that heartbeated in time
- Concurrent RegisterClient/Heartbeat from multiple threads

**`route_get_strategy_test.cpp`:**
- RandomRouteGetStrategy with single replica — returns it
- RandomRouteGetStrategy with multiple replicas — returns a valid one
- RandomRouteGetStrategy statistical distribution — roughly uniform over many calls
- Custom RouteGetStrategy implementation — verify Router delegates to it

**`route_put_strategy_test.cpp`:**
- TierAwareMostAvailable with HBM capacity — returns HBM node
- TierAwareMostAvailable with no HBM space, DRAM available — falls through to DRAM
- TierAwareMostAvailable with all clients full on all tiers — returns nullopt
- TierAwareMostAvailable load spreading: two nodes with HBM space, picks most available
- TierAwareMostAvailable tier ordering: HBM preferred over DRAM even if DRAM has more space
- Custom RoutePutStrategy implementation — verify Router delegates to it

**`router_test.cpp`:**
- RouteGet for unknown key — returns nullopt
- RouteGet delegates to injected RouteGetStrategy
- RoutePut with no alive clients — returns nullopt
- RoutePut delegates to injected RoutePutStrategy
- Router with default strategies (nullptr) — creates RandomRouteGet + TierAwareMostAvailable
- Router with custom strategies — uses injected strategies

### 12.2 Integration Test

**`integration_test.cpp`:**
1. Start `MasterServer` in a background thread
2. Create `MasterClient` connected to localhost
3. `RegisterSelf`
4. Register several keys with locations
5. Verify `Lookup` returns correct locations
6. Verify `RouteGet` returns a valid location
7. Verify `RoutePut` returns a target node with enough capacity
8. Write to target (simulated), then `Register` the new block
9. Verify the new block appears in `Lookup`
10. Unregister a replica, verify `Lookup` reflects the change
11. `BatchRegister` multiple keys, verify all appear in `Lookup`
12. `BatchUnregister` some of those keys, verify they're gone
13. Verify heartbeat keeps the client alive
14. Stop heartbeat, wait for reaper, verify client's keys are cleaned up
15. `UnregisterSelf` for graceful shutdown path
16. Shut down server

---

## 13. What Was Deferred (Add When Needed)

| Feature | Why deferred | When to add |
|---------|-------------|-------------|
| Sharded BlockIndex (64 shards) | Single mutex is sufficient for prototype load | When profiling shows mutex contention |
| `BatchLookup` / `BatchRouteGet` RPCs | No batch use case yet (`BatchRegister`/`BatchUnregister` are implemented) | When prefetch workflows are implemented |
| `Exists` RPC | Clients can use `Lookup` instead | If Lookup overhead becomes a concern |
| `GetClientInfo` / `ListClients` RPCs | Admin/debug only | When operational tooling is needed |
| Generic `ClientMetadata` map | Concrete capacity fields suffice | When arbitrary per-client metadata is needed |
| `REREGISTER` heartbeat action | Master restart recovery | When zero-downtime master restart is required |
| Pimpl pattern (hiding gRPC types) | Adds boilerplate | When header compile times matter |
| Two-phase Reaper locking | Optimization for many clients | When client count exceeds ~100 |
| `UnregisterAll(key)` | No caller in the prototype | When admin cleanup tooling is built |
| Thread-local RNG in separate policy | Already using thread_local in Router | N/A (already simplified) |

---

## 14. Pay Attention: Known Pitfalls and Potential Bug Introducers

The following issues were identified during design review. Implementers **must** address
each one during coding — the pseudocode as written will produce incorrect or undefined
behavior if translated literally.

### PA-1. `Unregister`: use-after-erase + data race on remaining-count check [HIGH]

**Section:** 3.4.2 — `Unregister(node_id, key, location)`

**Problem:** After `entries_.erase(it)` removes an empty key and the lock is released,
the code accesses `entries_[key]` without the lock to count remaining locations. This is
(a) a data race with concurrent threads, and (b) if using `operator[]`, silently
re-creates an empty entry that nothing will ever clean up.

**Fix:** Compute the remaining count **before** erasing and **while still holding the
lock**. Example:

```
exclusive_lock(mutex_)
...
locs.erase(remove(locs, location), locs.end())
removed = (locs.size() < original_size)

// Compute remaining BEFORE erase, UNDER lock
remaining_for_client = count locs with node_id matching node_id

if locs.empty():
    entries_.erase(it)
unlock

if removed && registry_ != nullptr && remaining_for_client == 0:
    registry_->UntrackKey(node_id, key)
```

### PA-2. `BatchUnregister`: use-after-free on `locs` reference [HIGH]

**Section:** 3.4.8 — `BatchUnregister(node_id, entries[])`

**Problem:** `locs` is a reference to `it->second.locations`. After `entries_.erase(it)`,
that memory is freed. The subsequent `remaining = count locs with node_id matching
node_id` reads a dangling reference — undefined behavior.

**Fix:** Same pattern as PA-1: compute the remaining count before the erase. When
`locs.empty()` after removal, remaining is trivially 0 — just push to `keys_to_untrack`
directly.

```
if locs.size() < original_size:
    removed_count += 1
    remaining = count locs with node_id matching node_id
    if remaining == 0:
        keys_to_untrack.append(key)

if locs.empty():
    entries_.erase(it)
```

### PA-3. `Heartbeat` uses shared (read) lock but mutates record [HIGH]

**Section:** 9.2

**Problem:** Section 9.2 specifies that `Heartbeat` takes a shared lock, but the
heartbeat handler writes to `record.last_heartbeat` and `record.tier_capacities`. Under
a shared lock, multiple gRPC threads can execute Heartbeat concurrently, producing a
data race on these fields.

**Fix:** `Heartbeat` must acquire an **exclusive (write) lock**, not a shared lock.
Alternatively, if read contention on `clients_` is a concern, use a per-client mutex or
atomic fields for the hot-path updates.

### PA-4. Reaper erases from `clients_` while iterating [HIGH]

**Section:** 5.5 — Reaper pseudocode

**Problem:** The Reaper loop iterates over `clients_` and calls `clients_.erase(node_id)`
inside the loop body. Erasing from `std::unordered_map` during range-based iteration
invalidates iterators — undefined behavior.

**Fix:** Use the iterator-based erase pattern:

```
auto it = clients_.begin();
while (it != clients_.end()) {
    if (expired) {
        // ... GC work ...
        it = clients_.erase(it);   // returns next valid iterator
    } else {
        ++it;
    }
}
```

Or collect expired client IDs into a separate list first, then erase in a second pass.

### PA-5. Race between `Register` unlock and `TrackKey` [MEDIUM]

**Section:** 3.4.1 — `Register(node_id, key, location)`

**Problem:** After the BlockIndex lock is released but before `TrackKey` executes, a
concurrent `Unregister` on the same (key, location) can remove the location, find
`remaining == 0`, and call `UntrackKey`. When `TrackKey` then runs, it leaves a stale
entry in `client_keys_` that will never be cleaned up. Similarly, the Reaper could run
in this window and miss this key entirely.

The same race exists in `BatchRegister` (Section 3.4.7).

**Fix:** Either call `TrackKey` while still holding the BlockIndex lock (requires
careful lock ordering — see PA-6), or use a separate lightweight lock that serializes
Track/Untrack operations against Register/Unregister for the same key.

**As of #6 (2026-04-23):** the race window is **narrowed but not eliminated**.
`TrackKey` / `UntrackKey` now re-do the index `Lookup` after releasing the
registry lock (Phase 2 in §9.3), as defense-in-depth against exactly the
TOCTOU described above; Phase 3's `clients_.find` re-check additionally
prevents `operator[]` from leaking an empty `client_keys_[node_id]` entry
when the client unregisters concurrently. The residual stale-entry race
remains benign and self-heals via `UnregisterClient` / `ReapExpiredClients`
— see distributed-known-issues.md #6 "假设 4" for the full argument.

### PA-6. Lock ordering description contradicts pseudocode [MEDIUM]

**Section:** 9.3

**Problem:** The prose states:

> `BlockIndex.Register()` acquires BlockIndex lock first, then calls
> `registry_->TrackKey()` which acquires registry lock.

But the pseudocode in Section 3.4.1 explicitly **releases** the BlockIndex lock before
calling `TrackKey` (comment: "outside lock to avoid inversion"). If an implementer
follows the prose description and holds both locks simultaneously, they introduce a
**deadlock** with the Reaper path (which acquires registry lock → BlockIndex lock).

**Fix:** Update the prose in Section 9.3 to accurately reflect the design:

> `BlockIndex.Register()` releases the BlockIndex lock first, then calls
> `registry_->TrackKey()` which acquires the registry lock. The two locks
> are never held simultaneously on this path.

### PA-7. `MasterServerConfig` passed by const-ref but contains `unique_ptr` [MEDIUM]

**Section:** 7.7 — `MasterServer` constructor

**Problem:** `MasterServerConfig` contains `unique_ptr<RouteGetStrategy>` and
`unique_ptr<RoutePutStrategy>`. The constructor takes `const MasterServerConfig&`,
which prevents moving the unique_ptrs out — this will not compile.

**Fix:** Change the constructor signature to take by value or rvalue reference:

```cpp
explicit MasterServer(MasterServerConfig config);        // by value
// or
explicit MasterServer(MasterServerConfig&& config);      // by rvalue-ref
```

### PA-8. Incorrect erase-remove idiom in `Unregister` and `BatchUnregister` [MEDIUM]

**Sections:** 3.4.2 and 3.4.8

**Problem:** The pseudocode uses single-argument erase:

```
locs.erase(remove(locs, location))
```

`std::remove` returns an iterator to the new logical end. The single-iterator overload
of `vector::erase` erases only **one** element at that position. If there were somehow
multiple matching entries (e.g., a bug elsewhere), not all would be removed. More
importantly, this is the wrong API — the correct erase-remove idiom requires the
two-iterator overload. Note that `UnregisterByNode` (Section 3.4.3) already uses the
correct form.

**Fix:** Use the two-iterator erase-remove idiom consistently:

```
locs.erase(std::remove(locs.begin(), locs.end(), location), locs.end())
```

---

### Summary Table

| ID | Issue | Severity | Type |
|----|-------|----------|------|
| PA-1 | `Unregister` reads `entries_[key]` after erase + unlock | HIGH | Data race + logic error |
| PA-2 | `BatchUnregister` reads dangling `locs` ref after erase | HIGH | Use-after-free (UB) |
| PA-3 | `Heartbeat` shared lock on mutable fields | HIGH | Data race |
| PA-4 | Reaper erases from map during iteration | HIGH | Iterator invalidation (UB) |
| PA-5 | Register-to-TrackKey race window | MEDIUM | TOCTOU race |
| PA-6 | Lock ordering prose contradicts pseudocode | MEDIUM | Doc error (deadlock risk) |
| PA-7 | `MasterServerConfig` const-ref with `unique_ptr` | MEDIUM | Won't compile |
| PA-8 | Wrong erase-remove idiom (single-arg erase) | MEDIUM | Logic error |
