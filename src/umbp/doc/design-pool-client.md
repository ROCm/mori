# PoolClient — Data Plane Integration Design Document

**Author:** Dev3
**Status:** Draft
**Scope:** PoolClient (control + data plane), PeerService (SSD coordination), PoolAllocator
**Depends on:** `design-master-control-plane.md` (MasterClient, MasterServer, BlockIndex, Router, ClientRegistry)

---

## 1. Overview

`PoolClient` is a high-level client class built on top of `MasterClient`. It combines the
Master's routing decisions (control plane) with MORI IO Engine data transfers (data plane)
to provide a simple Put/Get API for storing and retrieving KV cache blocks across the
cluster.

**Key components:**

- **MasterClient** — control plane: RoutePut/RouteGet for routing, Register/Unregister
  for metadata, heartbeat for liveness.
- **MORI IO Engine** — data plane: RDMA (or TCP fallback) for actual block transfer
  between nodes.
- **PeerService** — lightweight gRPC service on each storage-providing node, handling
  IO Engine handshake and SSD coordination.

**Four storage paths** (determined by `node_id` and `tier` from RoutePut):

- **Local DRAM**: Target is this node, tier=DRAM. Direct `memcpy` into local DRAM
  buffer, no RDMA needed.
- **Local SSD**: Target is this node, tier=SSD. Direct PosixFile write to local SSD,
  no RDMA needed.
- **Remote DRAM**: Target is another node, tier=DRAM. Pure RDMA one-sided write/read.
  The Master manages DRAM offset allocation centrally — RoutePut atomically selects
  the node and allocates an offset.
- **Remote SSD**: Target is another node, tier=SSD. Two-phase transfer — RDMA write
  into remote DRAM staging area, then PeerService RPC to persist to SSD via PosixFile.

---

## 2. Architecture

```
                                       ┌─────────────────────────────────┐
                                       │       UMBP Master Server        │
  ┌────────────────────┐               │  (pure control plane)           │
  │    Local Node      │   gRPC        │                                 │
  │                    │               │  ClientRegistry                 │
  │  ┌──────────────┐  │  RoutePut     │    + peer_address               │
  │  │  PoolClient  │──────────────────►    + engine_desc                │
  │  │              │  │  RouteGet     │    + dram_memory_desc           │
  │  │  ┌─────────┐ │◄─────────────────    + PoolAllocator per node/tier │
  │  │  │ Master  │ │  │  Register     │                                 │
  │  │  │ Client  │ │  │  Unregister   │  BlockIndex                     │
  │  │  └─────────┘ │  │               │  Router                        │
  │  │              │  │               └─────────────────────────────────┘
  │  │  ┌─────────┐ │  │
  │  │  │IO Engine│ │  │  RDMA          ┌─────────────────────────────────┐
  │  │  │         │─────────────────────►  Remote Node B                  │
  │  │  └─────────┘ │  │               │                                 │
  │  │              │  │  gRPC          │  ┌───────────────────────────┐  │
  │  │  PeerService ├──────────────────►│  │ PeerService               │  │
  │  │  stub        │  │  (SSD ops +   │  │  GetPeerInfo              │  │
  │  └──────────────┘  │   handshake)  │  │  CommitSsdWrite           │  │
  │                    │               │  │  PrepareSsdRead            │  │
  └────────────────────┘               │  └───────────────────────────┘  │
                                       │                                 │
                                       │  RDMA-registered DRAM buffer    │
                                       │  SSD storage (PosixFile)        │
                                       └─────────────────────────────────┘
```

---

## 3. Design Decisions

### 3.1 Master Manages Per-Tier Allocation via PoolAllocator

The Master centrally manages capacity and allocation for **all tiers** (DRAM and SSD)
on all nodes. When `RoutePut` selects a target node, the Master atomically deducts
capacity from that node's tier. This ensures:

- **Consistency**: The Master's view of available capacity is always accurate for
  every tier. No stale-capacity routing decisions between heartbeats.
- **Atomicity**: Node selection and capacity reservation happen in a single RPC.
  No race between concurrent writers targeting the same node.
- **Minimal RPCs**: DRAM Put requires only 2 gRPC calls (RoutePut + Register) plus
  1 RDMA transfer.

A unified `PoolAllocator` is used for both tiers, with one behavioral difference:

- **DRAM**: manages **offset allocation + capacity tracking**. The allocated offset
  is returned in `RoutePut` for RDMA addressing.
- **SSD**: manages **capacity tracking only**. SSD addressing is file-based
  (`{key}.bin`), handled by PeerService. The offset returned by the allocator
  is unused externally.

```cpp
struct PoolAllocator {
    uint64_t total_size = 0;
    uint64_t used_size = 0;

    // Offset management (DRAM uses this; SSD sets to nullopt)
    struct OffsetTracker {
        uint64_t bump = 0;
        std::vector<std::pair<uint64_t, uint64_t>> free_list;  // {offset, size}
    };
    std::optional<OffsetTracker> offset_tracker;

    // DRAM: allocate offset from offset_tracker + deduct used_size.
    // SSD: deduct used_size only (returns 0).
    // Returns nullopt if insufficient capacity.
    std::optional<uint64_t> Allocate(uint64_t size);

    // DRAM: return offset to free_list + add back used_size.
    // SSD: add back used_size only (offset param ignored).
    // Adjacent free blocks may be coalesced (DRAM only).
    void Deallocate(uint64_t offset, uint64_t size);

    uint64_t AvailableBytes() const { return total_size - used_size; }
};
```

Each node has a `std::map<TierType, PoolAllocator>` in its `ClientRecord`. On
`Unregister`, the Master calls `Deallocate` on the corresponding tier's allocator
to reclaim capacity (and offset, for DRAM).

### 3.2 PeerService for SSD Operations and IO Engine Handshake

Each node that provides storage runs a lightweight gRPC `PeerService`. It serves two
purposes:

1. **IO Engine handshake** (`GetPeerInfo`): On first connection, `PoolClient` calls
   `GetPeerInfo` to obtain the remote node's `EngineDesc` and `MemoryDesc`. This is
   needed to register the remote IO Engine and set up RDMA transfers. One-time cost
   per peer.

2. **SSD coordination**: SSD is not RDMA-accessible. Writing to remote SSD requires
   first RDMA-writing data into the remote node's DRAM staging area, then asking the
   remote PeerService to persist it to SSD via PosixFile. Reading from remote SSD
   is the reverse: ask PeerService to load SSD data into DRAM staging, then RDMA-read.

The Master does **not** manage SSD offset allocation. SSD writes produce opaque
`location_id` strings (e.g., file paths) that the PeerService mints and the Master
stores without interpreting.

### 3.3 MORI IO Engine as Data Plane

`PoolClient` uses `mori::io::IOEngine` for all RDMA transfers. The IO Engine supports
both RDMA (production) and TCP (testing/development) backends with an identical API.

Each node allocates and registers a local staging buffer for RDMA transfers. This
staging buffer is used to:
- Stage outgoing data before RDMA write (the source pointer from the caller may not
  be in RDMA-registered memory).
- Receive incoming data from RDMA read before copying to the caller's destination.

---

## 4. Proto Extensions

### 4.1 Master Proto (`umbp.proto`)

Minimal additions to support PoolClient:

```protobuf
message RegisterClientRequest {
  string node_id                      = 1;
  string node_address                 = 2;
  repeated TierCapacity tier_capacities = 3;
  // --- NEW fields ---
  string peer_address                 = 4;  // PeerService gRPC address
  bytes  engine_desc                  = 5;  // packed EngineDesc (for IO Engine setup)
  bytes  dram_memory_desc             = 6;  // packed MemoryDesc (RDMA-registered DRAM)
}

message RoutePutResponse {
  bool     found            = 1;
  string   node_id          = 2;
  string   node_address     = 3;
  TierType tier             = 4;
  // --- NEW fields ---
  string   peer_address     = 5;  // target node's PeerService address
  bytes    engine_desc      = 6;  // target node's packed EngineDesc
  bytes    dram_memory_desc = 7;  // target node's packed MemoryDesc (DRAM)
  uint64   allocated_offset = 8;  // Master-allocated offset (DRAM tier only)
}

message RouteGetResponse {
  bool     found            = 1;
  Location source           = 2;
  // --- NEW fields ---
  string   peer_address     = 3;  // source node's PeerService address
  bytes    engine_desc      = 4;  // source node's packed EngineDesc
  bytes    dram_memory_desc = 5;  // source node's packed MemoryDesc (DRAM)
}
```

### 4.2 PeerService Proto (`umbp_peer.proto`)

```protobuf
syntax = "proto3";
package umbp;

service UMBPPeer {
  // IO Engine handshake (one-time per peer)
  rpc GetPeerInfo(GetPeerInfoRequest) returns (GetPeerInfoResponse);

  // SSD coordination
  rpc CommitSsdWrite(CommitSsdWriteRequest) returns (CommitSsdWriteResponse);
  rpc PrepareSsdRead(PrepareSsdReadRequest) returns (PrepareSsdReadResponse);
}

// --- Handshake ---
message GetPeerInfoRequest {}
message GetPeerInfoResponse {
  bytes  engine_desc         = 1;  // packed EngineDesc
  bytes  dram_memory_desc    = 2;  // packed MemoryDesc
  uint64 ssd_capacity        = 3;
  uint64 ssd_available       = 4;
  uint64 staging_base_offset = 5;  // SSD staging base offset within DRAM buffer
}

// --- SSD Write: RDMA into staging → persist to SSD ---
message CommitSsdWriteRequest {
  string key              = 1;
  uint64 staging_offset   = 2;  // offset within DRAM staging area
  uint64 size             = 3;
}
message CommitSsdWriteResponse {
  bool   success          = 1;
  string ssd_location_id  = 2;  // opaque SSD handle (e.g. file path)
}

// --- SSD Read: load SSD → staging → caller RDMA reads ---
message PrepareSsdReadRequest {
  string key              = 1;
  string ssd_location_id  = 2;
  uint64 size             = 3;
}
message PrepareSsdReadResponse {
  bool   success          = 1;
  uint64 staging_offset   = 2;  // data loaded at this DRAM staging offset
}
```

---

## 5. ClientRecord Extensions

```cpp
struct ClientRecord {
    std::string node_id;
    std::string node_address;
    ClientStatus status = ClientStatus::UNKNOWN;
    std::chrono::steady_clock::time_point last_heartbeat;
    std::chrono::steady_clock::time_point registered_at;
    std::map<TierType, TierCapacity> tier_capacities;

    // --- NEW fields for PoolClient integration ---
    std::string peer_address;                     // PeerService gRPC address
    std::vector<uint8_t> engine_desc_bytes;       // packed EngineDesc
    std::vector<uint8_t> dram_memory_desc_bytes;  // packed MemoryDesc (DRAM)
    std::map<TierType, PoolAllocator> allocators; // per-tier allocation
    // DRAM: PoolAllocator with offset_tracker (offset + capacity)
    // SSD:  PoolAllocator without offset_tracker (capacity only)
};
```

---

## 6. RoutePutStrategy Extensions

`RoutePutResult` gains fields to carry data-plane information:

```cpp
struct RoutePutResult {
    std::string node_id;
    std::string node_address;
    TierType tier;

    // --- NEW ---
    std::string peer_address;
    std::vector<uint8_t> engine_desc_bytes;
    std::vector<uint8_t> dram_memory_desc_bytes;
    uint64_t allocated_offset = 0;  // valid for DRAM tier only
};
```

`Router::RoutePut` combines strategy selection and allocation into a single call.
It internally retries when allocation fails on the selected node:

```
Router::RoutePut(key, node_id, block_size):
    candidates = registry_.GetAliveClients()
    loop:
        result = strategy.Select(candidates, block_size)
        if !result: return nullopt              // all candidates exhausted

        alloc = registry_.AllocateForPut(result.node_id, result.tier, block_size)
        if alloc:
            // merge alloc info into result (peer_address, engine_desc, etc.)
            return result

        // allocation failed — remove this (node, tier) from candidates and retry
        candidates[result.node_id].tier_capacities.erase(result.tier)
```

The `MasterServer` RoutePut handler simply calls `router_.RoutePut()` and fills
the response — no retry logic needed at the handler level.

**Capacity consistency**: `AllocateForPut` and `DeallocateForUnregister` in
`ClientRegistry` synchronize `tier_capacities[tier].available_bytes` with
`PoolAllocator::AvailableBytes()` after every allocation/deallocation. This ensures
the routing strategy always sees accurate remaining capacity, not stale heartbeat
values.

For SSD tier, `allocated_offset` is 0 (capacity-only mode) — SSD addressing is
handled by PeerService using file paths.

---

## 7. PoolClient C++ Interface

### 7.1 Configuration

```cpp
namespace mori::umbp {

struct PoolClientConfig {
    MasterClientConfig master_config;

    // MORI IO Engine
    mori::io::IOEngineConfig io_config;
    mori::io::BackendType backend_type = mori::io::BackendType::RDMA;
    std::unique_ptr<mori::io::BackendConfig> backend_config;

    // Local staging buffer for RDMA transfers
    size_t staging_buffer_size = 64ULL * 1024 * 1024;  // 64 MB

    // Exportable DRAM buffer (optional; makes this node a storage provider)
    void*  exportable_dram_buffer = nullptr;
    size_t exportable_dram_buffer_size = 0;

    // Exportable SSD storage (optional)
    std::string exportable_ssd_dir;
    size_t exportable_ssd_capacity = 0;

    // Tier capacities to report to Master
    std::map<TierType, TierCapacity> tier_capacities;

    // PeerService listen port (0 = no PeerService, consumer-only mode)
    uint16_t peer_service_port = 0;
};

}  // namespace mori::umbp
```

### 7.2 Class Definition

```cpp
namespace mori::umbp {

class PoolClient {
 public:
  explicit PoolClient(PoolClientConfig config);
  ~PoolClient();

  PoolClient(const PoolClient&) = delete;
  PoolClient& operator=(const PoolClient&) = delete;

  // --- Lifecycle ---
  bool Init();
  void Shutdown();

  // --- Core API ---
  bool Put(const std::string& key, const void* src, size_t size);
  bool Get(const std::string& key, void* dst, size_t size);
  bool Remove(const std::string& key);

  // --- Access ---
  MasterClient& Master();
  bool IsInitialized() const;

 private:
  PoolClientConfig config_;
  bool initialized_ = false;

  // Control plane
  std::unique_ptr<MasterClient> master_client_;

  // Data plane — IO Engine
  std::unique_ptr<mori::io::IOEngine> io_engine_;
  mori::io::MemoryDesc staging_mem_;
  std::unique_ptr<char[]> staging_buffer_;
  std::mutex staging_mutex_;

  // Data plane — PeerService server (when this node provides storage)
  std::unique_ptr<PeerServiceServer> peer_service_;

  // Local cache: key → Location (populated on Put, used by Remove)
  std::mutex cache_mutex_;
  std::unordered_map<std::string, Location> location_cache_;

  // Cached peer connections (lazy init)
  struct PeerConnection {
    std::string peer_address;
    mori::io::EngineDesc engine_desc;
    mori::io::MemoryDesc dram_memory;
    uint64_t staging_base_offset = 0;  // SSD staging base within DRAM buffer
    bool engine_registered = false;
    // gRPC stub for remote PeerService
    std::unique_ptr<void, void (*)(void*)> peer_stub{nullptr, +[](void*) {}};
    std::mutex ssd_op_mutex;  // serialize SSD ops to this peer to prevent staging races
  };
  std::mutex peers_mutex_;
  std::unordered_map<std::string, PeerConnection> peers_;

  PeerConnection& GetOrConnectPeer(const std::string& node_id,
                                   const std::string& peer_address,
                                   const std::vector<uint8_t>& engine_desc_bytes,
                                   const std::vector<uint8_t>& dram_memory_desc_bytes);

  // DRAM path: pure RDMA
  bool DramWrite(PeerConnection& peer, const void* src, size_t size, uint64_t offset);
  bool DramRead(PeerConnection& peer, void* dst, size_t size, uint64_t offset);

  // SSD path: RDMA + PeerService coordination
  bool SsdWrite(PeerConnection& peer, const std::string& key,
                const void* src, size_t size);
  bool SsdRead(PeerConnection& peer, const std::string& key,
               void* dst, size_t size);
};

}  // namespace mori::umbp
```

---

## 8. Data Flows

### 8.1 Init

```
  PoolClient                IOEngine              PeerServiceServer    Master
      │                         │                       │                │
      │  IOEngine(key, config)  │                       │                │
      │────────────────────────►│                       │                │
      │  CreateBackend(RDMA)    │                       │                │
      │────────────────────────►│                       │                │
      │                         │                       │                │
      │  alloc staging_buffer   │                       │                │
      │  RegisterMemory(buf)    │                       │                │
      │────────────────────────►│  → staging_mem_       │                │
      │                         │                       │                │
      │  [if provider mode:]    │                       │                │
      │  RegisterMemory(        │                       │                │
      │    exportable_dram)     │                       │                │
      │────────────────────────►│  → export_mem_        │                │
      │                         │                       │                │
      │  [if peer_port > 0:]    │                       │                │
      │  Start PeerService ─────────────────────────────►  listening     │
      │                         │                       │                │
      │  pack engine_desc +     │                       │                │
      │  dram_memory_desc       │                       │                │
      │  RegisterSelf(caps,     │                       │                │
      │    peer_addr,           │                       │                │
      │    engine_desc,         │                       │                │
      │    dram_memory_desc)    │                       │                │
      │──────────────────────────────────────────────────────────────────►
      │                         │                       │  store in      │
      │                         │                       │  ClientRecord  │
      │  heartbeat_interval ◄────────────────────────────────────────────│
      │  StartHeartbeat()       │                       │                │
```

### 8.2 Put/Get Dispatch Logic

The target node returned by RoutePut may be the local node or a remote node, and the
tier may be DRAM or SSD. `PoolClient` dispatches based on
`is_local = (result.node_id == config_.master_config.node_id)` and `result.tier`:

```
Put(key, src, size):
    result = Master::RoutePut(key, size)
    is_local = (result.node_id == config_.master_config.node_id)

    if is_local && tier == DRAM:
        memcpy(local_dram + offset, src, size)       // direct local copy
    elif is_local && tier == SSD:
        PosixFile write {ssd_dir}/{key}.bin           // direct local file write
    elif remote && tier == DRAM:
        RDMA write → remote DRAM at offset            // see 8.3
    elif remote && tier == SSD:
        RDMA write → remote staging + CommitSsdWrite  // see 8.4

    Master::Register(key, location)
    location_cache_[key] = location
```

```
Get(key, dst, size):
    location = Master::RouteGet(key)  // or lookup from location_cache_
    is_local = (location.node_id == config_.master_config.node_id)

    if is_local && tier == DRAM:
        memcpy(dst, local_dram + offset, size)        // direct local copy
    elif is_local && tier == SSD:
        PosixFile read {ssd_dir}/{key}.bin             // direct local file read
    elif remote && tier == DRAM:
        RDMA read ← remote DRAM at offset             // see 8.5
    elif remote && tier == SSD:
        PrepareSsdRead + RDMA read ← remote staging   // see 8.6
```

Local paths require no RDMA or PeerService — just memcpy or PosixFile I/O.
The following sections detail the remote paths with full sequence diagrams.

> **Zero-copy optimization (post-POC)**: The current design does a
> `memcpy(staging, src, size)` before RDMA write because the caller's source pointer
> may not be in RDMA-registered memory. If the caller can guarantee the pointer is
> registered, the staging copy can be skipped. POC prioritizes simplicity and
> correctness; a `PutFromRegisteredMemory` fast path can be added later.
> Same applies to Get.

### 8.3 Put — Remote DRAM (2 RPCs + 1 RDMA)

```
  Caller          PoolClient         Master              IOEngine     Remote Node B
    │                 │                 │                    │              │
    │  Put(key,src,sz)│                 │                    │              │
    │────────────────►│                 │                    │              │
    │                 │  RoutePut(key,  │                    │              │
    │                 │    node_id, sz) │                    │              │
    │                 │────────────────►│                    │              │
    │                 │                 │  select node B     │              │
    │                 │                 │  alloc offset=4096 │              │
    │                 │                 │  deduct capacity   │              │
    │                 │  {node_B, DRAM, │                    │              │
    │                 │   offset=4096,  │                    │              │
    │                 │   engine_desc,  │                    │              │
    │                 │   memory_desc}  │                    │              │
    │                 │◄────────────────│                    │              │
    │                 │                 │                    │              │
    │                 │  [first time: GetOrConnectPeer       │              │
    │                 │   → RegisterRemoteEngine]            │              │
    │                 │                 │                    │              │
    │                 │  memcpy(staging, src, sz)            │              │
    │                 │  Write(staging, 0,                   │              │
    │                 │    remote_dram, 4096, sz)            │              │
    │                 │────────────────────────────────────►│              │
    │                 │                 │                    │  RDMA WRITE  │
    │                 │                 │                    │─────────────►│
    │                 │                 │                    │  complete    │
    │                 │                 │                    │◄─────────────│
    │                 │  status OK ◄─────────────────────────│              │
    │                 │                 │                    │              │
    │                 │  loc = Location{│                    │              │
    │                 │    node_B,      │                    │              │
    │                 │    "4096",      │                    │              │
    │                 │    sz, DRAM}    │                    │              │
    │                 │  Register(key,  │                    │              │
    │                 │    loc)         │                    │              │
    │                 │────────────────►│                    │              │
    │                 │  OK ◄───────────│                    │              │
    │                 │                 │                    │              │
    │                 │  location_cache_│                    │              │
    │                 │    [key] = loc  │                    │              │
    │  true           │                 │                    │              │
    │◄────────────────│                 │                    │              │
```

### 8.4 Put — Remote SSD (3+ RPCs + 1 RDMA)

PoolClient holds `peer.ssd_op_mutex` for the entire SSD operation, ensuring the
staging area is not concurrently overwritten. The staging area is a fixed reusable
buffer — no allocation/deallocation needed.

```
  Caller      PoolClient        Master         PeerService_B    IOEngine    Remote B
    │              │                │                │              │           │
    │  Put(k,s,sz) │                │                │              │           │
    │─────────────►│                │                │              │           │
    │              │  RoutePut ─────►                │              │           │
    │              │                │  select B, SSD │              │           │
    │              │  {B, SSD,      │                │              │           │
    │              │   peer_addr,   │                │              │           │
    │              │   engine_desc, │                │              │           │
    │              │   memory_desc} │                │              │           │
    │              │◄───────────────│                │              │           │
    │              │                │                │              │           │
    │              │  [GetOrConnectPeer if needed]   │              │           │
    │              │  lock(peer_B.ssd_op_mutex)      │              │           │
    │              │  memcpy(staging, src, sz)       │              │           │
    │              │                │                │              │           │
    │              │  --- Phase 1: RDMA write to remote DRAM staging ---       │
    │              │  Write(staging, 0, remote_dram, │              │           │
    │              │    staging_base_offset, sz)     │              │           │
    │              │───────────────────────────────────────────────►│           │
    │              │                │                │              │  RDMA WR  │
    │              │                │                │              │──────────►│
    │              │  OK ◄──────────────────────────────────────────│           │
    │              │                │                │              │           │
    │              │  --- Phase 2: RPC → persist to SSD (PosixFile) ---        │
    │              │  CommitSsdWrite(key,            │              │           │
    │              │    staging_base_offset, sz) ───►              │           │
    │              │                │                │  write SSD   │           │
    │              │                │                │  (PosixFile) │           │
    │              │  {ssd_loc_id} ◄─────────────────│              │           │
    │              │                │                │              │           │
    │              │  loc = Location│                │              │           │
    │              │    {B, ssd_loc,│                │              │           │
    │              │     sz, SSD}   │                │              │           │
    │              │  Register(key, │                │              │           │
    │              │    loc) ───────►                │              │           │
    │              │  OK ◄──────────│                │              │           │
    │              │                │                │              │           │
    │              │  location_cache│                │              │           │
    │              │    [key] = loc │                │              │           │
    │              │  unlock(peer_B.ssd_op_mutex)   │              │           │
    │  true        │                │                │              │           │
    │◄─────────────│                │                │              │           │
```

### 8.5 Get — Remote DRAM (1 RPC + 1 RDMA)

```
  Caller          PoolClient         Master              IOEngine     Remote Node B
    │                 │                 │                    │              │
    │  Get(key,dst,sz)│                 │                    │              │
    │────────────────►│                 │                    │              │
    │                 │  RouteGet(key)  │                    │              │
    │                 │────────────────►│                    │              │
    │                 │  Location{B,    │                    │              │
    │                 │    "4096", sz,  │                    │              │
    │                 │    DRAM} +      │                    │              │
    │                 │   engine_desc,  │                    │              │
    │                 │   memory_desc   │                    │              │
    │                 │◄────────────────│                    │              │
    │                 │                 │                    │              │
    │                 │  [GetOrConnectPeer if needed]        │              │
    │                 │                 │                    │              │
    │                 │  Read(staging, 0,                    │              │
    │                 │    remote_dram, 4096, sz)            │              │
    │                 │────────────────────────────────────►│              │
    │                 │                 │                    │  RDMA READ   │
    │                 │                 │                    │─────────────►│
    │                 │                 │                    │  data        │
    │                 │                 │                    │◄─────────────│
    │                 │  status OK ◄─────────────────────────│              │
    │                 │                 │                    │              │
    │                 │  memcpy(dst, staging, sz)            │              │
    │  true           │                 │                    │              │
    │◄────────────────│                 │                    │              │
```

### 8.6 Get — Remote SSD (2 RPCs + 1 RDMA)

Also holds `peer.ssd_op_mutex` to prevent staging races.

```
  Caller      PoolClient        Master         PeerService_B    IOEngine    Remote B
    │              │                │                │              │           │
    │  Get(k,d,sz) │                │                │              │           │
    │─────────────►│                │                │              │           │
    │              │  RouteGet ─────►                │              │           │
    │              │  Location{B,   │                │              │           │
    │              │   ssd_loc, sz, │                │              │           │
    │              │   SSD} +       │                │              │           │
    │              │   peer_addr... │                │              │           │
    │              │◄───────────────│                │              │           │
    │              │                │                │              │           │
    │              │  [GetOrConnectPeer if needed]   │              │           │
    │              │  lock(peer_B.ssd_op_mutex)      │              │           │
    │              │                │                │              │           │
    │              │  --- Phase 1: RPC → load SSD data to DRAM staging ---     │
    │              │  PrepareSsdRead(key,            │              │           │
    │              │    ssd_loc, sz) ────────────────►              │           │
    │              │                │                │  read SSD    │           │
    │              │                │                │  (PosixFile) │           │
    │              │                │                │  → staging   │           │
    │              │  {staging_off} ◄─────────────────│              │           │
    │              │                │                │              │           │
    │              │  --- Phase 2: RDMA read from remote DRAM staging ---      │
    │              │  Read(staging, 0,               │              │           │
    │              │    remote_dram, staging_off, sz)│              │           │
    │              │───────────────────────────────────────────────►│           │
    │              │                │                │              │  RDMA RD  │
    │              │                │                │              │──────────►│
    │              │  OK ◄──────────────────────────────────────────│           │
    │              │                │                │              │           │
    │              │  memcpy(dst, staging, sz)       │              │           │
    │              │  unlock(peer_B.ssd_op_mutex)    │              │           │
    │  true        │                │                │              │           │
    │◄─────────────│                │                │              │           │
```

### 8.7 Remove

`PoolClient` caches the `Location` returned by each successful Put in a local
`unordered_map<string, Location>`. Remove looks up this cache directly — no
Lookup RPC needed.

```
  Caller          PoolClient         Master
    │                 │                 │
    │  Remove(key)    │                 │
    │────────────────►│                 │
    │                 │  location =     │
    │                 │  location_cache_│
    │                 │    [key]        │
    │                 │                 │
    │                 │  Unregister(key,│
    │                 │    location)    │
    │                 │────────────────►│
    │                 │                 │  remove from BlockIndex
    │                 │                 │  allocators[tier].Deallocate(
    │                 │                 │    offset, size)
    │                 │                 │  // DRAM: reclaim offset + capacity
    │                 │                 │  // SSD: reclaim capacity only
    │                 │  OK ◄───────────│
    │                 │                 │
    │                 │  erase from     │
    │                 │  location_cache_│
    │  true           │                 │
    │◄────────────────│                 │
```

### 8.8 First-Time Peer Connection

When `PoolClient` first encounters a remote node (via RoutePut or RouteGet), it
establishes the IO Engine connection:

```
  PoolClient                             IOEngine
      │                                      │
      │  [RoutePut/RouteGet returned         │
      │   engine_desc + dram_memory_desc     │
      │   from Master]                       │
      │                                      │
      │  unpack EngineDesc from bytes        │
      │  RegisterRemoteEngine(engine_desc)   │
      │─────────────────────────────────────►│
      │                                      │
      │  unpack MemoryDesc from bytes        │
      │  cache in PeerConnection             │
      │                                      │
      │  [subsequent transfers use cached    │
      │   EngineDesc + MemoryDesc directly]  │
```

Since the Master carries `engine_desc` and `dram_memory_desc` in RoutePut/RouteGet
responses, the common path (DRAM) does not require calling `PeerService::GetPeerInfo`.
`GetPeerInfo` is available as a fallback or for SSD-specific information.

---

## 9. PeerService Implementation

### 9.1 PeerServiceServer

```cpp
namespace mori::umbp {

class PeerServiceServer {
 public:
  PeerServiceServer(void* dram_buffer, size_t dram_size,
                    const mori::io::MemoryDesc& dram_mem_desc,
                    const mori::io::EngineDesc& engine_desc,
                    const std::string& ssd_dir, size_t ssd_capacity);

  void Start(uint16_t port);
  void Stop();

 private:
  // SSD staging buffer: a simple fixed region within the RDMA-registered DRAM.
  // Not managed by Master's PoolAllocator — just a local mutex-protected buffer.
  // SSD operations are serialized by ssd_mutex_, so a single staging slot suffices.
  void* ssd_staging_base_;
  size_t ssd_staging_size_;

  // SSD: PosixFile I/O
  std::string ssd_dir_;
  size_t ssd_capacity_;
  size_t ssd_used_ = 0;
  std::mutex ssd_mutex_;  // serializes all SSD operations + staging access

  // IO Engine descriptors (returned by GetPeerInfo)
  mori::io::EngineDesc engine_desc_;
  mori::io::MemoryDesc dram_mem_desc_;
};

}  // namespace mori::umbp
```

SSD operations use standard POSIX file I/O:
- Write: `open(O_WRONLY | O_CREAT | O_TRUNC)` → `write` → `fsync` → `close`
- Read: `open(O_RDONLY)` → `pread` → `close`
- File layout: one file per block at `{ssd_dir}/{key}.bin`

### 9.2 DRAM Layout

A node's DRAM buffer is logically split into two regions:

```
  ┌──────────────────────────────────────────────────────┐
  │              RDMA-registered DRAM Buffer              │
  │                                                      │
  │  ┌────────────────────┐  ┌────────────────────────┐  │
  │  │  Main region       │  │  SSD staging region    │  │
  │  │  (Master-managed   │  │  (PeerService-managed  │  │
  │  │   offset alloc)    │  │   for SSD read/write)  │  │
  │  └────────────────────┘  └────────────────────────┘  │
  └──────────────────────────────────────────────────────┘
```

- **Main region**: Allocations managed by the Master's `PoolAllocator` (DRAM mode). Used for
  DRAM-tier block storage. Addresses are returned in `RoutePut`.
- **SSD staging region**: Managed locally by PeerService (`ssd_mutex_`-protected
  fixed buffer). Temporary staging area for SSD read/write operations. Fixed reusable
  buffer — no allocation/deallocation needed.

Both regions are part of the same RDMA-registered `MemoryDesc`, but at non-overlapping
offset ranges.

---

## 10. Files to Add/Modify

**New files:**
- `include/umbp/pool_client.h` — PoolClient header
- `src/pool_client.cpp` — PoolClient implementation
- `include/umbp/peer_service.h` — PeerServiceServer header
- `src/peer_service.cpp` — PeerServiceServer implementation
- `proto/umbp_peer.proto` — PeerService proto definition
- `include/umbp/pool_allocator.h` — PoolAllocator header

**Modified files:**
- `proto/umbp.proto` — add `peer_address`, `engine_desc`, `dram_memory_desc`,
  `allocated_offset` fields
- `include/umbp/types.h` — extend `ClientRecord` with peer info + per-tier `PoolAllocator`
- `include/umbp/client.h` — `MasterClient::RegisterSelf` gains peer/engine params
- `src/client.cpp` — implement extended `RegisterSelf`, parse new response fields
- `src/client_registry.cpp` — store new fields in `ClientRecord`
- `src/master_server.cpp` — populate new response fields; call `PoolAllocator` in
  RoutePut; call `Deallocate` in Unregister (both DRAM and SSD tiers)
- `include/umbp/route_put_strategy.h` — extend `RoutePutResult`
- `CMakeLists.txt` — new sources, new proto, link `mori::io`

---

## 11. Concurrency Design

### 11.1 PoolAllocator Thread Safety

`PoolAllocator` lives inside `ClientRecord` on the Master. Both `RoutePut`
(`Allocate`) and `Unregister` (`Deallocate`) modify allocator state from
concurrent gRPC handler threads.

**Approach**: `PoolAllocator` is **not internally locked**. It is protected by
`ClientRegistry`'s `shared_mutex`:

- **Problem**: The current `RoutePut` path only calls `GetAliveClients()` (read
  operation, shared lock). Adding `PoolAllocator::Allocate` makes it a write.
- **Solution**: `Router::RoutePut` executes in two steps internally:
  1. Shared lock to get alive clients (`GetAliveClients()`)
  2. After strategy selects a node, **exclusive lock** via
     `registry_.AllocateForPut()` to call `allocators[tier].Allocate(block_size)`
     and sync `tier_capacities`
  3. If allocation fails, remove the failed (node, tier) from the local candidate
     list and retry — all within `Router::RoutePut`

Pseudocode:

```
Router::RoutePut(key, node_id, block_size):
    candidates = registry_.GetAliveClients()         // shared lock
    loop:
        result = strategy.Select(candidates, block_size)
        if !result: return nullopt

        alloc = registry_.AllocateForPut(...)        // exclusive lock
        if alloc:
            // merge alloc result into RoutePutResult, return
        candidates[result.node_id].tier_capacities.erase(result.tier)  // retry
```

- `AllocateForPut` and `DeallocateForUnregister` both update
  `tier_capacities[tier].available_bytes` from `PoolAllocator::AvailableBytes()`,
  keeping the routing strategy's capacity view accurate.
- `Unregister` already runs under exclusive lock (existing design), so
  `Deallocate` is called within it directly.
- `PoolAllocator` itself has no lock, avoiding nested-lock issues.

### 11.2 PoolClient Internal Thread Safety

| Resource | Protection |
|----------|-----------|
| `staging_buffer_` / `staging_mem_` | `staging_mutex_` — serializes RDMA transfers |
| `location_cache_` | `cache_mutex_` |
| `peers_` | `peers_mutex_` — lazy init + cached lookup |
| `MasterClient` | Internally thread-safe (gRPC channel is thread-safe) |
| `PeerServiceServer` | gRPC server is thread-safe; `ssd_mutex_` serializes SSD ops |

---

## 12. MasterClient API Changes

### 12.1 MasterClientConfig

No changes. `peer_address` and IO engine info are passed by `PoolClient` when
calling `RegisterSelf`, not baked into the config.

### 12.2 RegisterSelf Signature Change

```cpp
// Existing signature
grpc::Status RegisterSelf(const std::map<TierType, TierCapacity>& tier_capacities);

// New signature: adds data-plane info for PoolClient integration
grpc::Status RegisterSelf(
    const std::map<TierType, TierCapacity>& tier_capacities,
    const std::string& peer_address,                    // PeerService gRPC address
    const std::vector<uint8_t>& engine_desc_bytes,      // packed EngineDesc
    const std::vector<uint8_t>& dram_memory_desc_bytes  // packed MemoryDesc (DRAM)
);
```

Implementation: populate `RegisterClientRequest` fields 4-6 with the new params.

### 12.3 RoutePut Return Value

`RoutePutResult` is already extended in Section 6. `MasterClient::RoutePut`
signature is unchanged:

```cpp
grpc::Status RoutePut(const std::string& key, uint64_t block_size,
                      std::optional<RoutePutResult>* out_result);
```

Implementation: parse `RoutePutResponse` fields 5-8 into `RoutePutResult`'s
`peer_address`, `engine_desc_bytes`, `dram_memory_desc_bytes`, `allocated_offset`.

### 12.4 RouteGet Return Value

A new result struct carries data-plane info alongside the Location:

```cpp
struct RouteGetResult {
    Location location;                                // existing: block location
    std::string peer_address;                         // new
    std::vector<uint8_t> engine_desc_bytes;           // new
    std::vector<uint8_t> dram_memory_desc_bytes;      // new
};
```

Signature change:

```cpp
// Existing signature
grpc::Status RouteGet(const std::string& key,
                      std::optional<Location>* out_location);

// New signature
grpc::Status RouteGet(const std::string& key,
                      std::optional<RouteGetResult>* out_result);
```

Implementation: parse `RouteGetResponse` fields 3-5.

### 12.5 Unregister

Signature unchanged. Implementation change on the `MasterServer` side: the
`Unregister` handler parses tier and offset from the `Location`, then calls
`allocators[tier].Deallocate(offset, size)` to reclaim capacity.

---

## 13. Testing Strategy

### 13.1 PoolAllocator Unit Tests

- Basic allocate: `AvailableBytes` decreases correctly
- Allocate beyond capacity: returns `nullopt`
- Deallocate reclaim: `AvailableBytes` restored
- Allocate + Deallocate + re-Allocate: free list reuses released blocks
- Fragmentation: allocate A, B, C → free A and C → allocate D (D > A or C alone) → fails
- SSD mode (no offset_tracker): `Allocate` returns 0, only deducts capacity
- Concurrent Allocate/Deallocate stress test

### 13.2 PeerService Unit Tests

- GetPeerInfo: returns correct EngineDesc and MemoryDesc
- CommitSsdWrite: staging data correctly written to `{ssd_dir}/{key}.bin`
- PrepareSsdRead: SSD file correctly loaded into staging, returns staging_offset
- CommitSsdWrite over capacity: returns `success=false`
- PrepareSsdRead file not found: returns `success=false`

### 13.3 PoolClient Four-Path Integration Tests

Each path tests Put → Get → verify data → Remove → verify cleanup:

**13.3.1 Local DRAM**

```
1. Configure RoutePut strategy to return local node + tier=DRAM
2. Put(key, src, size) → success
3. Get(key, dst, size) → success, dst == src
4. Remove(key) → success
5. Get(key, ...) → fail (key not found)
```

**13.3.2 Local SSD**

```
1. Configure RoutePut strategy to return local node + tier=SSD
2. Put(key, src, size) → success
3. Verify {ssd_dir}/{key}.bin exists
4. Get(key, dst, size) → success, dst == src
5. Remove(key) → success
```

**13.3.3 Remote DRAM**

```
1. Start MasterServer + two PoolClients (node A and node B)
2. Both RegisterSelf with DRAM capacity
3. Node A: Put(key, src, size)
   → RoutePut returns node B + DRAM + offset
   → RDMA write to node B
   → Register metadata
4. Node A: Get(key, dst, size)
   → RouteGet returns node B's Location
   → RDMA read from node B
   → dst == src
5. Node A: Remove(key)
   → Unregister + Deallocate reclaims offset
```

**13.3.4 Remote SSD**

```
1. Start MasterServer + two PoolClients (node A and node B, both with peer_service_port > 0)
2. Node A: Put(key, src, size)
   → RoutePut returns node B + SSD
   → RDMA write to node B's DRAM staging
   → CommitSsdWrite → node B persists to SSD file
   → Register metadata
3. Node A: Get(key, dst, size)
   → RouteGet returns node B + SSD
   → PrepareSsdRead → node B loads SSD to staging
   → RDMA read from staging
   → dst == src
5. Node A: Remove(key)
```

### 13.4 MasterClient Extension Tests

- RegisterSelf with peer_address / engine_desc / dram_memory_desc
- RoutePut returns extended fields (peer_address, engine_desc, memory_desc, allocated_offset)
- RouteGet returns extended fields
- RoutePut correctly deducts PoolAllocator capacity
- Unregister correctly restores PoolAllocator capacity
- Multiple RoutePuts allocate non-overlapping offsets
- Full node → RoutePut routes to a different node

### 13.5 Edge Case & Error Tests

- Put when all nodes have no available space → RoutePut returns `found=false` → Put returns false
- RDMA transfer failure → Put returns false, allocated offset needs rollback (or GC cleanup)
- PeerService unreachable → SSD path Put/Get returns false
- PoolClient Init failure (cannot connect to Master) → Init returns false
- Remove non-existent key → location_cache_ miss → returns false
- Concurrent Put/Get/Remove stress test

---

## 14. Open Questions

- **Staging buffer concurrency**: Multiple concurrent Put/Get operations contend on
  the local staging buffer. Initial version uses a mutex; later can be upgraded to
  a ring buffer or arena allocator.
- **Large block transfer**: If `block_size > staging_buffer_size`, the transfer must
  be split into chunks. Initial version can reject oversized blocks.
- **DRAM fragmentation**: The bump + free list allocator may fragment over time.
  Adjacent free blocks should be coalesced on `Deallocate`. If fragmentation becomes
  a problem, upgrade to a slab allocator.
- **SSD staging concurrency scaling**: The current per-peer `ssd_op_mutex` serializes
  SSD operations. For higher concurrency, split the staging area into multiple slots
  to allow concurrent SSD operations to the same remote node.
- **PeerService connection pooling**: Under high concurrency, gRPC channel reuse and
  connection pooling may be needed.
