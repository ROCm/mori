# Local / Distributed Unification Design

> Status: Ready for Implementation (Rev 8)
> Reviewed: 6 rounds, all blocking issues resolved
> Date: 2026-03-27
> Context: `local/` and `distributed/` were developed independently by two contributors and recently merged into one repository. This document identifies architectural issues and proposes a unification plan.

---

## 1. Current Architecture Overview

```
            User API
               │
               ▼
        ┌──────────────┐
        │  UMBPClient   │  (local/)
        │               │
        │  LocalBlock   │
        │  Index        │
        │               │
        │  LocalStorage │─── DRAMTier (slab allocator, segment log)
        │  Manager      │─── SSDTier  (segment log, CRC, io_uring)
        │               │─── CopyPipeline
        │               │
        │  PoolClient?  │  (distributed/, optional)
        └──────┬───────┘
               │ owns (composition)
               ▼
        ┌──────────────┐
        │  PoolClient   │  (distributed/)
        │               │
        │  MasterClient │─── gRPC control plane
        │  IOEngine     │─── RDMA data plane
        │  PeerService  │─── gRPC peer SSD coordination
        │               │
        │  PutLocalDram │─── direct memcpy into DRAM buffer
        │  PutLocalSsd  │─── POSIX open/write/fsync
        │  GetLocalDram │
        │  GetLocalSsd  │
        └──────────────┘
```

### Two Sets of APIs in PoolClient

PoolClient exposes two distinct groups of methods:

**Group A — Standalone full-path methods** (used by `pool_client_main.cpp`):
- `Put(key, src, size)` — routes via Master, handles all 4 paths (local DRAM/SSD + remote DRAM/SSD)
- `Get(key, dst, size)` — routes via Master, dispatches to local or remote read
- `Remove(key)` — unregisters from Master

**Group B — Integration methods for UMBPClient** (header comment: "Phase 2: DRAM-only"):
- `RegisterWithMaster(key, size, location_id, tier)` — register locally-written block
- `UnregisterFromMaster(key)` — unregister on eviction
- `GetRemote(key, dst, size)` — fetch from remote DRAM only
- `PutRemote(key, src, size)` — write to remote DRAM only
- `ExistsRemote(key)` — check remote existence

UMBPClient currently only uses Group B. Group A is used by standalone test binaries.

---

## 2. Identified Problems

### 2.1 Two Independent SSD Implementations

| Aspect | local/ SSDTier | distributed/ PoolClient + PeerService |
|--------|---------------|---------------------------------------|
| Storage format | Segment log (`RecordHeader` + payload) | One `.bin` file per key |
| I/O backend | io_uring / PThread (configurable) | POSIX `open/write/read` |
| Index | `segment::Index` + `segment::Meta` | None (file name = key) |
| Capacity mgmt | Watermark-based eviction | Simple `used` counter (PeerService only) |
| Concurrency | Two-phase locking (`mu_` + `io_mu_`) | `ssd_mutex_` (PeerService only) |
| Data integrity | CRC32 per record | None |

**Impact**: Data written via PoolClient/PeerService is invisible to SSDTier, and vice versa. If both write to the same SSD directory, they produce incompatible file formats that interfere with each other.

### 2.2 DRAM Buffer Dual Management

UMBPClient exports `DRAMTier::GetBasePtr()` to PoolClient at construction time:

```cpp
// umbp_client.cpp constructor
pc_config.dram_buffers.push_back({dram->GetBasePtr(), total});
pc_config.tier_capacities[TierType::DRAM] = {total, total};  // full capacity reported
```

This creates dual ownership:
- **DRAMTier's slab allocator** assigns slots for local writes
- **Master's PoolAllocator** can assign offsets for remote writes into the same buffer

If Master routes a remote node's write to this node's DRAM, `PoolClient::PutLocalDram` performs a raw `memcpy` at the Master-assigned offset — bypassing the slab allocator entirely. This can **silently overwrite data** managed by DRAMTier.

The design doc mentions "Option B: report DRAM capacity as 0 to Master" to avoid this, but the current code reports full capacity.

### 2.3 Eviction Callback Only Covers DRAM

```cpp
// umbp_client.cpp
storage_.SetOnTierChange([this](..., StorageTier from, ...) {
    if (from == StorageTier::CPU_DRAM && pool_client_) {
        pool_client_->UnregisterFromMaster(key);  // only DRAM → out
    }
});
```

When a block is demoted from DRAM to SSD:
1. Master is told the key no longer exists (unregistered)
2. The data still exists on local SSD
3. Remote nodes cannot discover or access this SSD-resident data

This is expected for the current "Phase 2: DRAM-only" scope, but means **SSD-resident data is invisible to the cluster indefinitely**.

### 2.4 Conflicting Routing Philosophies

| | UMBPClient (local-first) | PoolClient (Master-directed) |
|---|---|---|
| Write strategy | Write locally, then register globally | Ask Master where to write, then execute |
| Who decides placement | Local node | Master (centralized) |
| SSD trigger | Eviction: DRAM full → demote to SSD | Master routes directly to SSD tier |
| Local storage role | Primary | Passive recipient of Master allocations |

When Master returns `is_local && tier == SSD`, PoolClient's `PutLocalSsd` writes a raw `.bin` file — completely bypassing `LocalStorageManager` and `SSDTier`. The two philosophies produce incompatible local state.

### 2.5 PeerService Bypasses Local Storage Stack

`PeerServiceServer::CommitSsdWrite` (handling remote SSD writes on this node) uses raw POSIX I/O:

```cpp
// peer_service.cpp
int fd = ::open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
::write(fd, src, request->size());
::fsync(fd);
```

Data written by PeerService is not tracked by `LocalBlockIndex`, not managed by `SSDTier`, and not subject to eviction. The local `UMBPClient` on the same node cannot discover these blocks.

### 2.6 Heartbeat Overwrites Master-Side Capacity Accounting

The Master maintains server-side capacity state via `AllocateForPut` / `DeallocateForUnregister`, which update `record.tier_capacities[tier].available_bytes` to stay in sync with `PoolAllocator`. However, the heartbeat path **overwrites this state unconditionally**:

```cpp
// client_registry.cpp:161-176  — Heartbeat handler
it->second.tier_capacities = tier_capacities;  // wholesale overwrite
```

Meanwhile, the client's heartbeat sends `current_capacities_`, which is **only set once at registration time** (`master_client.cpp:114`) and never updated by subsequent `AllocateForPut`/`DeallocateForUnregister` calls on the server side:

```cpp
// master_client.cpp:346-368  — HeartbeatLoop
for (const auto& [tier, cap] : current_capacities_) {  // stale snapshot
    tc->set_available_capacity_bytes(cap.available_bytes);
}
```

**Impact**: After a `RoutePut` allocates space (Master decrements `available_bytes`), the next heartbeat from the client restores the old pre-allocation value, causing the Master to believe more space is available than actually is. This can lead to double-allocation of the same DRAM offsets and silent data corruption. The bug affects both DRAM and SSD capacity accounting.

**Required fix**: Either (a) have the Master ignore capacity fields in heartbeat (use heartbeat only for liveness), or (b) have the client track its own allocations and send accurate capacity in heartbeats.

### 2.7 Promote() Does Not Fire on_tier_change_

`MaybeAutoPromote()` → `Promote()` moves data from SSD to DRAM (`local_storage_manager.cpp:801-855`), but `Promote()` calls `from->Evict()` + `UpsertIndexTier()` directly — it does **not** go through `MoveKey()` and therefore **does not fire `on_tier_change_`**. This means:

1. A local read hits SSD → auto-promote copies to DRAM → `LocalBlockIndex` updated to DRAM
2. But `on_tier_change_` never fires → Master still thinks the block is on SSD
3. Remote nodes that `RouteGet` this key receive the old SSD location, which is now stale

This is an existing bug (independent of the unification work) that becomes critical once SSD-tier cluster visibility is enabled.

### 2.8 RoutePut Allocation Lacks Failure Rollback

`Router::RoutePut()` calls `AllocateForPut()` on the Master side, which decrements `available_bytes` and reserves a DRAM offset or SSD capacity slot (`router.cpp:71`, `client_registry.cpp:242`). This allocation is returned to the client.

If the subsequent data-plane operation fails (RDMA write fails, or `CommitSsdWrite` is rejected), or the client crashes between `RoutePut` and `Register`, the allocated capacity is **permanently leaked** — there is no `AbortPut` / `RollbackAllocation` RPC. The existing `Unregister` path only calls `DeallocateForUnregister` if the key **exists in `GlobalBlockIndex`** (`master_server.cpp:171`), so an allocation that was never registered cannot be reclaimed.

For DRAM this is especially bad: the offset reservation in `PoolAllocator` is permanent, making that memory range unavailable for future writes.

---

## 3. Proposed Architecture

### 3.1 Core Principles

> 1. All local storage operations MUST go through `LocalStorageManager`.
> 2. The networking layer (renamed from PoolClient) handles ONLY cluster coordination and data transport.
> 3. This design targets a **single SSD store per node** (store_index = 0). Multi-store support is deferred; any store_index plumbing retained for forward compatibility must use the `"0:<opaque>"` format consistently.

### 3.2 Target Architecture

```
                         UMBPClient
                  (sole public API, owns all local state)
                             │
              ┌──────────────┼──────────────────┐
              ▼              ▼                  ▼
      LocalStorage    ClusterCoord       PeerServiceServer
      Manager         (network only)     (gRPC inbound)
      ┌──────────┐    ┌─────────────┐    ┌──────────────────┐
      │ DRAMTier │    │ MasterClient│    │ staging_buf*     │
      │ SSDTier  │    │ IOEngine    │    │ &storage_        │
      │ Eviction │    │ (RDMA)      │    │ &index_          │
      │ Index    │    │             │    │ &coordinator_    │
      └──────────┘    └─────────────┘    └────────┬─────────┘
                            │                     │
                     RDMA registers       write/read/register
                     staging_buf at       via injected refs
                     init time (once)
```

**Key properties:**
- `ClusterCoordinator` has no local storage logic — only Master gRPC + RDMA transport
- `PeerServiceServer` receives injected references to `LocalStorageManager`, `LocalBlockIndex`, and `ClusterCoordinator` to perform write/read/register operations. Note: despite the diagram showing these as siblings, **PeerService depends on ClusterCoordinator at runtime** (calls `coordinator_.FinalizeAllocation()`). The "no runtime dependency" claim from earlier revisions was incorrect.
- `UMBPClient` is the sole orchestrator of local-vs-remote decisions

**Dependency direction** (no cycles):
```
PeerServiceServer → {LocalStorageManager, LocalBlockIndex, ClusterCoordinator}
UMBPClient → {LocalStorageManager, LocalBlockIndex, ClusterCoordinator, PeerServiceServer}
ClusterCoordinator → {MasterClient, IOEngine}  (no reverse deps)
```

### 3.3 ClusterCoordinator (Refactored from PoolClient)

**Kept** (network-only responsibilities):
```cpp
class ClusterCoordinator {
public:
    bool Init();
    void Shutdown();

    // --- Identity ---
    const std::string& NodeId() const;

    // --- Control plane (Master interaction) ---
    bool UnregisterFromMaster(const std::string& key);
    bool ExistsRemote(const std::string& key);

    // Local cluster-location map query (no Master RPC).
    bool IsRegistered(const std::string& key) const;

    // Allocation lifecycle (§3.11, §3.13)
    //
    // FinalizeAllocation: for RoutePut-initiated writes. Clears the pending
    //   lease identified by allocation_id. No capacity deduction (already done
    //   at RoutePut time).
    // PublishLocalBlock: for local demotion (DRAM→SSD), auto-promote (SSD→DRAM),
    //   or local-first writes. Adds to GlobalBlockIndex. For SSD tier, also
    //   deducts capacity from ssd_allocators. For DRAM tier, skips allocator
    //   deduction (DRAM allocators not created when available=0, §3.13). (§3.13)
    // AbortAllocation: reclaims capacity on explicit failure.
    //   allocation_id: echoed from RoutePut response to identify the lease.
    //   node_id: target node (may differ from caller for remote writes).
    bool FinalizeAllocation(const std::string& key, size_t size,
                            const std::string& location_id, TierType tier,
                            const std::string& allocation_id);
    bool PublishLocalBlock(const std::string& key, size_t size,
                           const std::string& location_id, TierType tier);
    bool AbortAllocation(const std::string& node_id, TierType tier,
                         const std::string& allocation_id, uint64_t size);

    // --- Data plane (RDMA transport) ---
    bool RemoteDramRead(const std::string& key, void* dst, size_t size);
    bool RemoteDramWrite(const std::string& key, const void* src, size_t size);
    bool RemoteSsdRead(const std::string& key, void* dst, size_t size);
    bool RemoteSsdWrite(const std::string& key, const void* src, size_t size);

    // --- RDMA memory management ---
    bool RegisterMemory(void* ptr, size_t size);
    void DeregisterMemory(void* ptr);

    // SSD staging buffer accessors (buffer owned by this object, §3.10)
    void* SsdStagingPtr() const;
    size_t SsdStagingSize() const;
    const std::vector<uint8_t>& SsdStagingMemDescBytes() const;

    MasterClient& Master();
};
```

**Removed:**
- `PutLocalDram` / `GetLocalDram` — local DRAM access belongs to `LocalStorageManager`
- `PutLocalSsd` / `GetLocalSsd` — local SSD access belongs to `LocalStorageManager`
- `Put(key, src, size)` / `Get(key, dst, size)` / `Remove(key)` — routing orchestration belongs to `UMBPClient`

**`location_cache_` — requires careful handling** (see §3.8).

### 3.4 PeerServiceServer (Write Path)

**Problem with delegating to `UMBPClient::Put()`**: `UMBPClient::Put()` always writes to DRAM first, not SSD. SSD persistence happens only asynchronously via `CopyPipeline` and only when `role == SharedSSDLeader`. In normal distributed mode, `client_.Put()` would write to DRAM, register the block with Master as DRAM-resident, and may never copy to SSD — completely breaking the `RoutePut(SSD)` contract.

**Solution: PeerService inlines the write-register sequence using injected references**, not `UMBPClient::Put()`:

```cpp
// PeerServiceServer receives injected references at construction:
//   LocalStorageManager& storage_
//   LocalBlockIndex& index_
//   ClusterCoordinator& coordinator_

grpc::Status CommitSsdWrite(...) {
    const std::string& key = request->key();
    const size_t size = request->size();
    const uint32_t store_index = request->store_index();

    // Single-store enforcement (§3.1)
    if (store_index != 0) {
        MORI_UMBP_ERROR("[PeerService] CommitSsdWrite: store_index {} != 0, rejected",
                        store_index);
        response->set_success(false);
        return grpc::Status::OK;
    }

    // Idempotency: if key already exists locally but is NOT yet registered
    // with Master, fall through to the registration step. If fully committed
    // (exists in index AND cluster_locations_), short-circuit.
    auto existing = index_.Lookup(key);
    if (existing && coordinator_.IsRegistered(key)) {
        response->set_success(true);
        return grpc::Status::OK;
    }

    const void* src = staging_base_ + request->staging_offset();

    // Step 1: Write to SSD (skip if already landed from a previous attempt)
    if (!existing) {
        bool ok = storage_.Write(key, src, size, StorageTier::LOCAL_SSD);
        if (!ok) { response->set_success(false); return grpc::Status::OK; }
        index_.Insert(key, {StorageTier::LOCAL_SSD, 0, size});
    }

    // Step 2: Obtain SSD location — must succeed for registration
    auto* ssd = storage_.GetTier(StorageTier::LOCAL_SSD);
    auto loc_id = ssd ? ssd->GetLocationId(key) : std::nullopt;
    if (!loc_id) {
        // SSD write succeeded but location query failed — data is orphaned.
        // Roll back: evict local data and report failure so writer can
        // AbortAllocation on the Master side.
        storage_.Evict(key);
        index_.Remove(key);
        MORI_UMBP_ERROR("[PeerService] CommitSsdWrite: GetLocationId failed for '{}'", key);
        response->set_success(false);
        return grpc::Status::OK;
    }
    std::string location_id = "0:" + *loc_id;

    // Step 3: Finalize with Master — must succeed for the commit to be valid.
    //   Uses FinalizeAllocation (not PublishLocalBlock) because this write was
    //   initiated by a RoutePut that already reserved capacity (§3.13).
    //   allocation_id is echoed from the CommitSsdWriteRequest (originated from
    //   RoutePut response → writer → CommitSsdWrite RPC).
    bool registered = coordinator_.FinalizeAllocation(
        key, size, location_id, TierType::SSD,
        request->allocation_id());
    if (!registered) {
        // Local data persists but is not cluster-visible.
        // Roll back local state so a retry doesn't hit the idempotency gate.
        storage_.Evict(key);
        index_.Remove(key);
        MORI_UMBP_ERROR("[PeerService] CommitSsdWrite: FinalizeAllocation failed for '{}'", key);
        response->set_success(false);
        return grpc::Status::OK;
    }

    response->set_success(true);
    return grpc::Status::OK;
}
```

**Why not `UMBPClient&`**: Injecting `UMBPClient&` into PeerService creates a wiring problem — PeerService is currently constructed inside `PoolClient::Init()`, which is called during `UMBPClient`'s constructor. Passing `this` (an incomplete `UMBPClient`) would risk use-before-construction. Using individual references (`storage_`, `index_`, `coordinator_`) avoids this and is compatible with two-phase init (see §3.10).

**Transactional semantics**: The operation is "all-or-nothing at the cluster level". Local SSD write, index update, and Master finalization must ALL succeed for the commit to return success. If GetLocationId or FinalizeAllocation fails after local write, the local state is rolled back (`Evict` + `index_.Remove`) so that:
- The writer receives failure and can call `AbortAllocation` to reclaim Master-side capacity
- A subsequent retry will NOT be short-circuited by the idempotency check
- No orphaned local data accumulates

**Idempotency on retry**: The idempotency gate checks both `index_.Lookup()` AND `coordinator_.IsRegistered()`. A key that was written locally but not registered (from a rolled-back attempt) will not match the gate, so the retry re-executes the full write+register sequence. A fully committed key (index + registered) short-circuits correctly. `IsRegistered()` is a local check against the `cluster_locations_` map — no Master RPC needed.

### 3.5 PeerServiceServer (Read Path)

`LocalStorageManager::ReadIntoPtr()` triggers `MaybeAutoPromote()` when `auto_promote_on_read = true` (the default, `config.h:91`). A remote SSD read through this path would silently move data from SSD to DRAM, and since `Promote()` does not fire `on_tier_change_` (§2.7), Master metadata would become stale.

**Solution: Add a no-promote read path to LocalStorageManager:**

```cpp
// New API: read without triggering auto-promotion
bool LocalStorageManager::ReadIntoPtrNoPromote(
    const std::string& key, uintptr_t dst, size_t size);
```

Implementation: identical to `ReadIntoPtr()` but omits the `MaybeAutoPromote()` call. PeerService uses this path since the caller is remote and promotion creates unnecessary DRAM pressure:

```cpp
grpc::Status PrepareSsdRead(...) {
    void* dst = staging_base_ + read_offset;
    bool ok = storage_.ReadIntoPtrNoPromote(
        request->key(), reinterpret_cast<uintptr_t>(dst), request->size());
    // ...
}
```

### 3.6 UMBPClient Orchestration (Unified Routing)

UMBPClient absorbs the local-vs-remote dispatch logic currently in PoolClient's `Put/Get`. Note that `storage_.Write()` does NOT update `LocalBlockIndex` — `UMBPClient` must do this explicitly, as it does today:

```cpp
bool UMBPClient::Put(const std::string& key, const void* data, size_t size) {
    if (index_.MayExist(key)) return true;  // dedup

    // Write to local DRAM (storage_.Write defaults to CPU_DRAM tier)
    if (!storage_.Write(key, data, size)) return false;

    // Index update is UMBPClient's responsibility (not done by LocalStorageManager)
    index_.Insert(key, {StorageTier::CPU_DRAM, 0, size});

    // Register with cluster if distributed
    if (coordinator_) {
        MaybePublishLocal(key, size);  // → coordinator_->PublishLocalBlock()
    }
    copy_pipeline_->MaybeCopyToSharedSSD(key);
    return true;
}

bool UMBPClient::GetIntoPtr(const std::string& key, uintptr_t dst, size_t size) {
    // 1. Try local storage (DRAM + SSD)
    //    Note: ReadIntoPtr may trigger MaybeAutoPromote (SSD→DRAM).
    //    After §2.7 is fixed, on_tier_change_ handles Master re-registration.
    if (storage_.ReadIntoPtr(key, dst, size)) return true;

    // 2. Try remote DRAM
    if (coordinator_) {
        if (coordinator_->RemoteDramRead(key, (void*)dst, size)) {
            if (config_.distributed->cache_remote_fetches) {
                // Cache locally: write to DRAM + update index + register with Master
                storage_.WriteFromPtr(key, dst, size);
                index_.Insert(key, {StorageTier::CPU_DRAM, 0, size});
                MaybePublishLocal(key, size);  // → coordinator_->PublishLocalBlock()
            }
            return true;
        }
        // 3. Try remote SSD (future)
        if (coordinator_->RemoteSsdRead(key, (void*)dst, size)) {
            return true;
        }
    }
    return false;
}
```

### 3.7 Eviction Callback and TierBackend Location Query

The `on_tier_change_` callback has an explicit contract: **must NOT call back into `LocalStorageManager`** (`local_storage_manager.h:43`). This means we cannot call a hypothetical `storage_.GetSsdLocation(key)` from within the callback.

**Solution**: Pre-compute the location information BEFORE the callback fires, and pass it through the callback signature.

**Prerequisite — TierBackend location query**: Currently `TierBackend::Write()` returns only `bool`. To populate `TierLocationInfo`, `MoveKey()` needs to know where the data landed in the destination tier:

```cpp
virtual std::optional<std::string> GetLocationId(const std::string& key) const;
```

`MoveKey()` calls `GetLocationId()` on the destination tier after `Write()` succeeds but before firing the callback. This does not change `Write()`'s signature.

**location_id format**: Must encode `store_index` for Master's `DeallocateForUnregister` to work correctly. The Master parses `location_id` as `"<store_index>:<value>"` (`master_server.cpp:174-183`), using `store_index` to select the correct `ssd_allocator` or `dram_allocator`. Under the single-SSD-store model (§3.1), `GetLocationId()` returns the opaque tier-specific handle, and callers prepend `"0:"`:

```
DRAM: GetLocationId() returns "4096"    → location_id = "0:4096"
SSD:  GetLocationId() returns "seg3:80" → location_id = "0:seg3:80"
```

**Extended callback signature:**
```cpp
using TierChangeCallback = std::function<void(
    const std::string& key,
    StorageTier from_tier,
    std::optional<StorageTier> to_tier,
    std::optional<TierLocationInfo> new_location  // pre-computed by MoveKey/Promote
)>;

struct TierLocationInfo {
    std::string location_id;  // full "store_index:opaque" format, ready for Master
    size_t size;
};
```

**Callback usage in UMBPClient:**
```cpp
storage_.SetOnTierChange(
    [this](const std::string& key, StorageTier from,
           std::optional<StorageTier> to,
           std::optional<TierLocationInfo> new_loc) {
        if (!coordinator_) return;

        // Always unregister old location (restores allocator capacity on Master)
        coordinator_->UnregisterFromMaster(key);

        if (to == StorageTier::LOCAL_SSD && new_loc) {
            // DRAM → SSD: publish at SSD location.
            // Uses PublishLocalBlock (§3.13) because this is a local demotion
            // with no prior RoutePut — Master must deduct SSD capacity.
            coordinator_->PublishLocalBlock(
                key, new_loc->size, new_loc->location_id, TierType::SSD);
        } else if (to == StorageTier::CPU_DRAM && new_loc) {
            // SSD → DRAM (auto-promote): publish at DRAM location.
            coordinator_->PublishLocalBlock(
                key, new_loc->size, new_loc->location_id, TierType::DRAM);
        }
        // to == nullopt: fully evicted, just unregister (already done above)
    });
```

**Critical fix required**: `Promote()` (`local_storage_manager.cpp:801`) must also fire `on_tier_change_`. Currently it calls `from->Evict()` + `UpsertIndexTier()` directly. Fix: call `on_tier_change_(key, from_id, to_id, new_loc)` at the end of `Promote()`, same as `MoveKey()` does.

### 3.8 Location Cache and LocalBlockIndex Data Model Gap

PoolClient's `location_cache_` stores full `Location {node_id, location_id, size, tier}`, which is required by `MasterClient::Unregister()`. `LocalBlockIndex` currently only stores `LocalLocation {tier, offset, size}` — it lacks `node_id` and `location_id`.

**This means `location_cache_` cannot simply be deleted.** Two options:

**Option A — Keep a cluster-location map in ClusterCoordinator**: Retain a mapping from `key → Location` specifically for Unregister operations. This is essentially what `location_cache_` already does, but scoped to keys that THIS node registered (not all keys).

**Option B — Extend LocalBlockIndex**: Add `std::optional<ClusterLocation>` to `LocalLocation`:

```cpp
struct ClusterLocation {
    std::string location_id;  // e.g. "0:4096" for DRAM, "0:seg3:80" for SSD
    TierType cluster_tier;    // TierType::DRAM or TierType::SSD
};

struct LocalLocation {
    StorageTier tier;
    uint64_t offset;
    size_t size;
    std::optional<ClusterLocation> cluster;  // NEW: present when registered with Master
};
```

**Recommended: Option A for now.** Extending `LocalBlockIndex` is cleaner long-term but increases blast radius. The cluster-location map in ClusterCoordinator is functionally identical to the existing `location_cache_` — just with a clearer name and ownership.

### 3.9 Peer-Written Block Ownership

When a remote node routes a write to this node (via PeerService), the data is now persisted locally. But the **Master registration was done by the writing node**, not this node. This creates an ownership gap:

- The writing node's `location_cache_` (or cluster-location map) holds the `Location`
- This node (the storage owner) has no record that the block exists in Master's index
- If this node evicts the block (SSD full, or DRAM demotion), it cannot call `UnregisterFromMaster` because it doesn't have the `Location`

**Problem with naive dual registration**: `GlobalBlockIndex::BatchRegister` deduplicates by `Location` equality (`operator==` compares `{node_id, location_id, size, tier}`). If both the writing node and the storage-owner node register with the **same** `Location` (same `node_id` = storage-owner), the second registration is a no-op — there's only one entry. Then whichever node calls `Unregister` first deletes the sole entry, leaving the other node with a stale cluster-location map.

**Solution — Storage-owner-only registration**: Instead of dual registration, make the **storage-owner node the sole registrant**. The writing node does NOT register — it relies on PeerService to handle registration on its behalf (see §3.4 step 4):

1. Writing node: `RoutePut → RDMA write → CommitSsdWrite RPC`
2. PeerService (storage-owner): `Write to SSD → Update local index → FinalizeAllocation(key, size, loc, SSD, allocation_id)`
3. Writing node: receives success, does NOT call `Register` itself (no `cluster_locations_` entry)

This is clean because:
- Only one `Location` exists in Master's index, owned by the storage node
- The storage node's eviction callback can unregister it
- The writing node treats the operation as fire-and-forget
- `RoutePut` response told the writing node WHERE to write; PeerService confirms it landed

**Trade-off**: The writing node loses the ability to `Remove()` the block. If that's needed, add a `ReleaseSsdBlock` RPC to PeerService that triggers eviction + unregister on the storage-owner side.

### 3.10 PeerService Ownership and Two-Phase Init

**Current wiring**: PeerServiceServer is created inside `PoolClient::Init()`, which is called during `UMBPClient`'s constructor. The target architecture (§3.2) shows PeerService as a sibling of ClusterCoordinator under UMBPClient, but the construction order creates a circular dependency if PeerService needs references to `storage_`, `index_`, and `coordinator_`.

**Solution — Two-phase initialization**:

```cpp
UMBPClient::UMBPClient(const UMBPConfig& config)
    : storage_(config, &index_), ... {

    // Phase 1: Create ClusterCoordinator (replaces PoolClient)
    //   ClusterCoordinator allocates and RDMA-registers the SSD staging buffer.
    if (config_.distributed.has_value()) {
        coordinator_ = std::make_unique<ClusterCoordinator>(...);
        coordinator_->Init();  // registers with Master, starts heartbeat
    }

    // Phase 2: Create PeerServiceServer with fully-constructed references
    //   Receives staging buffer pointer from ClusterCoordinator (which owns the buffer
    //   and its RDMA registration). PeerService only reads/writes the buffer contents.
    if (coordinator_ && config_.distributed->peer_service_port > 0) {
        peer_service_ = std::make_unique<PeerServiceServer>(
            coordinator_->SsdStagingPtr(),   // owned by coordinator
            coordinator_->SsdStagingSize(),  // owned by coordinator
            coordinator_->SsdStagingMemDescBytes(),  // for GetPeerInfo response
            storage_,       // fully constructed
            index_,         // fully constructed
            *coordinator_   // fully constructed
        );
        peer_service_->Start(config_.distributed->peer_service_port);
    }
}
```

**SSD staging buffer ownership**: The staging buffer and its RDMA memory descriptor remain allocated and registered by `ClusterCoordinator` (as PoolClient does today, `pool_client.cpp:151-161`). `ClusterCoordinator` exposes the buffer pointer, size, and serialized `MemoryDesc` via accessor methods. PeerServiceServer receives these at construction and uses them for `GetPeerInfo` responses and staging I/O, but does **not** own or deallocate the buffer.

**Ownership hierarchy:**

```
UMBPClient owns:
  ├── LocalStorageManager  (constructed first)
  ├── LocalBlockIndex      (constructed first)
  ├── ClusterCoordinator   (Phase 1: owns staging buffers + RDMA registration)
  │     └── ssd_staging_buffer_ (allocated here, RDMA-registered here)
  └── PeerServiceServer    (Phase 2: receives refs to all above + staging ptr)
```

**Shutdown order**: Reverse of construction — `peer_service_->Stop()` first, then `coordinator_->Shutdown()` (deregisters RDMA, frees staging buffer), then storage/index destruction.

### 3.11 RoutePut Allocation Rollback

`Router::RoutePut()` calls `AllocateForPut()` which reserves capacity on the Master side. If the subsequent data-plane write or `FinalizeAllocation` fails, this reservation is leaked.

There are two failure modes:
1. **Explicit failure**: The writing node detects the failure (RDMA timeout, CommitSsdWrite rejected) and can actively clean up.
2. **Crash failure**: The writing node crashes between `RoutePut` and `Register`/`AbortAllocation`, leaving a permanently leaked reservation.

#### 3.11.1 Allocation ID

The current `AllocateForPut` returns `{buffer_index, allocated_offset}`, but SSD allocators have no offset tracker — they always return offset=0 (`pool_allocator.h:47-49`). Multiple concurrent SSD writes to the same node produce identical `(node_id, tier=SSD, buffer_index=0, offset=0, size)` tuples, making it impossible to match a `FinalizeAllocation` or `AbortAllocation` to the correct pending lease.

**Solution — Master-generated allocation_id**: `AllocateForPut` returns a unique `allocation_id` string (e.g., UUID or monotonic counter) that serves as the sole lease identifier. All subsequent operations (`FinalizeAllocation`, `AbortAllocation`, TTL reaper) match by this ID, not by `(buffer_index, offset)`.

```cpp
struct AllocateResult {
    std::string allocation_id;        // NEW: unique lease identifier
    std::string peer_address;
    std::vector<uint8_t> engine_desc_bytes;
    std::vector<uint8_t> dram_memory_desc_bytes;
    uint64_t allocated_offset = 0;    // DRAM only (still needed for RDMA addressing)
    uint32_t buffer_index = 0;        // which DRAM buffer or SSD store
};
```

The `allocation_id` is echoed back through `RoutePutResult` → client → `CommitSsdWriteRequest` → PeerService → `FinalizeAllocation`.

#### 3.11.2 Explicit Failure — AbortAllocation RPC

```protobuf
rpc AbortAllocation(AbortAllocationRequest) returns (AbortAllocationResponse);

message AbortAllocationRequest {
    string node_id = 1;           // target node whose allocation to roll back
    string allocation_id = 2;     // from RoutePut response
    uint64 size = 3;
}
```

Master handler: find the `PendingAllocation` by `allocation_id`, call `DeallocateForUnregister(node_id, tier, buffer_index, offset, size)` using the stored fields, then remove the pending entry.

**Client-side usage:**

```cpp
bool ClusterCoordinator::RemoteSsdWrite(...) {
    auto result = master_client_->RoutePut(key, size, &route);
    // ... RDMA write + CommitSsdWrite ...
    if (!ok) {
        master_client_->AbortAllocation(
            route->node_id, route->allocation_id, size);
        return false;
    }
    // Storage-owner finalization happens inside PeerService (§3.4)
}
```

#### 3.11.3 Crash Failure — Allocation Lease with TTL

If the client crashes between `RoutePut` and `FinalizeAllocation`/`AbortAllocation`, the pending allocation is orphaned. The reaper (which currently only cleans up dead clients) is extended to also expire stale leases on **alive** nodes:

```cpp
struct PendingAllocation {
    std::string allocation_id;    // unique identifier
    std::string node_id;
    TierType tier;
    uint32_t buffer_index;
    uint64_t offset;
    uint64_t size;
    std::chrono::steady_clock::time_point allocated_at;
};

// In ReaperLoop (extended):
for (auto it = pending_allocations_.begin(); it != pending_allocations_.end(); ) {
    if (now - it->allocated_at > allocation_ttl_) {
        DeallocateForUnregister(it->node_id, it->tier,
                                it->buffer_index, it->offset, it->size);
        MORI_UMBP_WARN("[Reaper] Expired pending allocation: id={}", it->allocation_id);
        it = pending_allocations_.erase(it);
    } else {
        ++it;
    }
}
```

**Important**: This runs regardless of whether the allocating client is still alive — the lease expires on its own. `FinalizeAllocation` and `AbortAllocation` both remove the pending entry by `allocation_id`, preventing the reaper from double-deallocating.

**TTL value**: Should be long enough for a normal `RoutePut → RDMA → CommitSsdWrite → FinalizeAllocation` round-trip (e.g., 30 seconds), but short enough to avoid prolonged capacity leaks. Configurable via `ClientRegistryConfig::allocation_ttl`.

### 3.12 FinalizeAllocation / PublishLocalBlock Timeout Ambiguity

`FinalizeAllocation` and `PublishLocalBlock` are gRPC calls. If the RPC succeeds on the Master but the client receives a timeout/disconnect, the client treats it as failure and rolls back locally (§3.4). This creates a split-brain: Master has the registration, but the local node has evicted the data and the writer has called `AbortAllocation`.

**Consequences of a stale Master entry:**

The current reaper only cleans up when a node's heartbeat expires — it does **not** detect "key registered to an alive node but data no longer exists locally." So the stale `Location` persists indefinitely while the node is alive. When a remote node calls `RouteGet` for this key, `Router::RouteGet` returns the stale location. The current read path (`PoolClient::GetRemote`, `pool_client.cpp:531`) makes a single attempt — if the RDMA read or PeerService read fails (because the data is gone), **the read fails entirely** rather than retrying another replica or falling back. This is not a benign "phantom hit" — it is a hard read failure.

**This is a user-visible failure mode**: a `RouteGet` that hits the stale entry will fail the read, not silently fall back. Callers must handle this as a cache miss.

**For the initial unification this is an acceptable risk** because:
- The frequency of "Master registered OK + client gRPC timeout" is low in practice
- The data was just written and is recoverable (the writer can retry the whole operation)
- Fixing this properly requires either idempotent RPCs or a reconciliation protocol, which is out of scope

**Long-term mitigation options (TODO):**

1. **Idempotent FinalizeAllocation with allocation_id dedup**: Since `allocation_id` is already unique (§3.11.1), the Master can deduplicate: if the `allocation_id` has already been finalized, return success. The client retries with the same `allocation_id` on timeout — no ambiguity.

2. **Read-path retry**: If `RouteGet` returns a location but the data read fails, retry with the next replica or re-route. This is independently useful and mitigates stale entries from any cause.

3. **Periodic reconciliation**: The node periodically scans its `cluster_locations_` against actual local storage; entries pointing to evicted data trigger `UnregisterFromMaster`.

### 3.13 Two Registration Paths: FinalizeAllocation vs PublishLocalBlock

The current `Register` RPC only writes to `GlobalBlockIndex` — it does **not** deduct from allocators (`master_server.cpp:129-151`). This is correct for RoutePut-initiated writes because `AllocateForPut` already deducted capacity at routing time. But it breaks for local demotion:

```
RoutePut path:   AllocateForPut (deducts) → data write → Register (index only) ✓
Local demotion:  (no allocation)          → MoveKey    → Register (index only) ✗
                                                         ↑ SSD capacity not deducted!
```

If local DRAM→SSD demotion calls `RegisterWithMaster(key, size, loc, SSD)`, the Master adds the key to `GlobalBlockIndex` with tier=SSD, but `ssd_allocators` still thinks that capacity is free. Subsequent `RoutePut` calls may over-allocate SSD capacity.

Additionally, the lease system (§3.11.2) expects `Register` to clear the matching `PendingAllocation` entry. But a "local publish" Register has no associated lease — the Master cannot distinguish which pending entry to clear.

**Solution — Split into two Master RPCs:**

**`FinalizeAllocation`** — for RoutePut-initiated writes:
```protobuf
rpc FinalizeAllocation(FinalizeRequest) returns (FinalizeResponse);
message FinalizeRequest {
    string node_id = 1;
    string key = 2;
    umbp.Location location = 3;
    string allocation_id = 4;   // from RoutePut response (§3.11.1)
}
```
Master handler: find PendingAllocation by `allocation_id`, add to GlobalBlockIndex, clear the pending entry. Capacity was already deducted by `AllocateForPut` — no further deduction needed.

**`PublishLocalBlock`** — for local demotion / CopyToSSD / local-first writes:
```protobuf
rpc PublishLocalBlock(PublishRequest) returns (PublishResponse);
message PublishRequest {
    string node_id = 1;
    string key = 2;
    umbp.Location location = 3;
}
```
Master handler: add to GlobalBlockIndex + deduct `location.size` from the appropriate `ssd_allocators[buffer_index]` (SSD only; DRAM skips deduction — see below). No pending lease involved.

**`CommitSsdWriteRequest` proto update** — must carry `allocation_id` from writer:
```protobuf
message CommitSsdWriteRequest {
    string key = 1;
    uint64 staging_offset = 2;
    uint64 size = 3;
    uint32 store_index = 4;
    string allocation_id = 5;  // NEW: echoed from RoutePut → writer → PeerService
}
```

**Client-side mapping:**

| Scenario | Who calls | Which RPC |
|----------|-----------|-----------|
| Remote SSD write via PeerService (§3.4) | PeerService (storage-owner) | `FinalizeAllocation` (echoes `allocation_id` from CommitSsdWriteRequest) |
| Local DRAM→SSD demotion (§3.7) | UMBPClient (via callback) | `PublishLocalBlock` |
| Local Put (§3.6) | UMBPClient | `PublishLocalBlock` (no prior RoutePut for local-first writes) |
| SSD→DRAM auto-promote (§3.7) | UMBPClient (via callback) | `PublishLocalBlock` (DRAM, no allocator deduction) |

**DRAM special case**: Since §3.6 specifies `available_bytes = 0` for DRAM, all local DRAM writes are "unprompted" — no RoutePut precedes them. `MaybeRegisterWithMaster()` in `UMBPClient::Put()` should use `PublishLocalBlock`. However, `PublishLocalBlock` must **NOT deduct from DRAM allocators** — if it did, `Deallocate(offset, size)` would subtract from `used_size`, increasing `AvailableBytes()` above 0, and `RoutePut` would start routing remote DRAM writes back to this node (re-introducing §2.2).

**Implementation**: `PublishLocalBlock` on the Master side checks tier:
- **SSD**: deduct from `ssd_allocators[buffer_index]` (capacity-only)
- **DRAM**: skip allocator deduction entirely; only add to `GlobalBlockIndex`

This is safe because DRAM capacity for local-first nodes is managed exclusively by the local `DRAMTier` slab allocator; the Master only needs to know the block exists for `RouteGet` discovery, not for capacity accounting. Conversely, `UnregisterFromMaster` for DRAM blocks must also skip the `DeallocateForUnregister` path for DRAM to avoid underflow. The simplest approach: **do not create DRAM allocators at all when `available_bytes = 0`** at registration time:

```cpp
// In ClientRegistry::RegisterClient, DRAM allocator setup:
if (!dram_buffer_sizes.empty() && dram_available > 0) {
    // Only create allocators if this node accepts remote DRAM writes
    for (size_t i = 0; i < dram_buffer_sizes.size(); ++i) {
        PoolAllocator alloc;
        alloc.total_size = dram_buffer_sizes[i];
        alloc.offset_tracker = PoolAllocator::OffsetTracker{};
        record.dram_allocators.push_back(std::move(alloc));
    }
}
// If available=0, dram_allocators is empty → AllocateForPut skips DRAM,
// DeallocateForUnregister is a no-op (buffer_index out of range)
```

**PeerService CommitSsdWrite update**: See §3.4 for the full transactional pseudocode. The key change from the proto perspective is that `CommitSsdWriteRequest` now carries `allocation_id` (echoed from `RoutePutResult` through the writer), which PeerService passes to `FinalizeAllocation` to clear the correct pending lease.

---

## 4. Migration Path

### Phase A: Fix Heartbeat Capacity Bug (Low Risk, High Impact)

1. Change `ClientRegistry::Heartbeat()` to NOT overwrite `tier_capacities`
2. Heartbeat becomes liveness-only; Master-side allocator state is authoritative
3. Test: `RoutePut` → allocate → heartbeat → verify `available_bytes` unchanged

### Phase B: Eliminate DRAM Dual-Allocation Risk (Low Risk)

1. Change DRAM capacity reporting to Master: `available_bytes = 0`
2. Verify that Master never routes remote writes to this node's DRAM
3. No code deletion — just a config change in UMBPClient constructor

### Phase C: Split Register into FinalizeAllocation / PublishLocalBlock + AbortAllocation (Medium Risk)

1. Add `FinalizeAllocation`, `PublishLocalBlock`, `AbortAllocation` to Master proto
2. `FinalizeAllocation` handler: add to GlobalBlockIndex + clear matching PendingAllocation (no capacity change)
3. `PublishLocalBlock` handler: add to GlobalBlockIndex + deduct from allocators (capacity change)
4. `AbortAllocation` handler: DeallocateForUnregister directly + clear matching PendingAllocation
5. Add `PendingAllocation` tracking to `ClientRegistry::AllocateForPut`
6. Extend `ReaperLoop` to reclaim expired pending allocations
7. Update `MasterClient` and `ClusterCoordinator` with new APIs
8. Wire into existing `PoolClient::Put()` failure paths as immediate fix
9. Test: `RoutePut` → RDMA write fails → `AbortAllocation(target_node_id)` → capacity restored
10. Test: `RoutePut` → client crash (simulate) → TTL expires → reaper reclaims capacity
11. Test: local DRAM→SSD demotion → `PublishLocalBlock` → SSD allocator capacity decremented
12. Test: RoutePut-initiated SSD write → `FinalizeAllocation` → pending lease cleared, no double-deduction

### Phase D: Fix Promote() to Fire on_tier_change_ (Low Risk)

1. Add `on_tier_change_` call at the end of `Promote()`, matching `MoveKey()`'s pattern
2. Add `TierBackend::GetLocationId(key)` virtual method; implement in `DRAMTier` and `SSDTier`
3. Extend `TierChangeCallback` signature to include `TierLocationInfo`
4. Update `MoveKey()` and `Promote()` to call `GetLocationId()` and pass result to callback
5. Ensure `GetLocationId()` returns value compatible with `"<store_index>:<value>"` format
6. Test: auto-promote on local read → `on_tier_change_` fires with correct from/to/location

### Phase E: Add ReadIntoPtrNoPromote (Low Risk)

1. Add `ReadIntoPtrNoPromote()` (or `ReadOptions` parameter) that suppresses `MaybeAutoPromote`
2. PeerService's `PrepareSsdRead` uses this path
3. Test: remote SSD read does NOT promote to DRAM; block stays on SSD

### Phase F: Move PeerService Ownership to UMBPClient (Medium Risk)

1. Move `PeerServiceServer` creation from `PoolClient::Init()` to `UMBPClient` constructor (two-phase init, §3.10)
2. `ClusterCoordinator` exposes staging buffer accessors (`SsdStagingPtr()`, `SsdStagingSize()`, `SsdStagingMemDescBytes()`)
3. PeerService receives `storage_`, `index_`, `coordinator_` references + staging buffer pointer
4. Shutdown order: PeerService → ClusterCoordinator → storage
5. Test: PeerService starts after all members constructed; no use-before-init
6. Test: `GetPeerInfo` returns correct `ssd_staging_mem_desc` (sourced from ClusterCoordinator)

### Phase G: PeerService Delegates to Local Storage (Medium Risk)

1. Replace `CommitSsdWrite` with transactional write-finalize sequence (§3.4): SSD write → index insert → GetLocationId → FinalizeAllocation, with rollback on any failure
2. Add `ClusterCoordinator::IsRegistered(key)` for idempotency gate
3. Add `store_index != 0` rejection at top of `CommitSsdWrite`
4. Replace `PrepareSsdRead` raw file I/O with `storage_.ReadIntoPtrNoPromote()`
5. Writing node stops calling `Register` for peer-written blocks (storage-owner-only registration, §3.9)
6. Writing node calls `AbortAllocation(target_node_id, ...)` on `CommitSsdWrite` failure
7. Remove `SsdStore` struct and `used` tracking (handled by SSDTier)
8. Test: remote SSD write → local `UMBPClient::GetIntoPtr` can read the data
9. Test: remote SSD write → target node eviction → Master unregister succeeds
10. Test: `CommitSsdWrite` retry after timeout → no duplicate SSD entries
11. Test: `CommitSsdWrite` with SSD write OK but FinalizeAllocation fail → local state rolled back, writer receives failure
12. Test: `CommitSsdWrite` with `store_index = 1` → rejected

### Phase H: Refactor PoolClient to ClusterCoordinator (Medium Risk)

1. Remove `PutLocalDram`, `GetLocalDram`, `PutLocalSsd`, `GetLocalSsd`
2. Remove standalone `Put/Get/Remove` (Group A methods)
3. Rename `location_cache_` to `cluster_locations_` with clear ownership semantics
4. Rename class: `PoolClient` → `ClusterCoordinator`
5. Update `pool_client_main.cpp` to use `UMBPClient` with distributed config

### Phase I: Extend Eviction Callback for SSD (Medium Risk)

1. Wire up the extended `on_tier_change_` callback in `UMBPClient` (§3.7)
2. Verify no reentrancy: callback does NOT call back into `LocalStorageManager`
3. Test: write to DRAM → eviction demotes to SSD → remote node can still fetch via PeerService
4. Test: auto-promote on read → Master metadata updated from SSD to DRAM

---

## 5. Summary of Changes

| Component | Current State | Target State |
|-----------|--------------|--------------|
| PoolClient | Full-path client with local + remote storage | ClusterCoordinator: network-only (Master + RDMA); exposes staging buffer accessors |
| PeerServiceServer | Raw POSIX file I/O, owned by PoolClient | Delegates to LSM+index+coordinator; owned by UMBPClient; two-phase init; idempotent writes |
| UMBPClient | Uses PoolClient for DRAM-only remote ops | Orchestrates all local/remote paths; explicit index updates; owns PeerService |
| DRAM buffer | Dual-managed (slab allocator + Master PoolAllocator) | Single-managed (slab allocator only) |
| SSD data | Two formats (segment log vs raw .bin) | Single format (segment log via SSDTier); single-store model (store_index = 0) |
| Promote() | Does not fire on_tier_change_ | Fires on_tier_change_ with TierLocationInfo |
| TierBackend | Write returns bool only | + `GetLocationId()` for post-write location query |
| Eviction → cluster | DRAM eviction unregisters from Master | Extended callback with pre-computed location; DRAM↔SSD re-registers |
| Peer-written blocks | Writing node registers; storage-owner unaware | Storage-owner-only registration; writing node fire-and-forget |
| Heartbeat capacity | Client sends stale snapshot, overwrites Master state | Liveness-only; Master-side allocator is authoritative |
| Master Register RPC | Single path, no capacity deduction, no lease awareness | Split: `FinalizeAllocation` (clears lease) vs `PublishLocalBlock` (deducts capacity) |
| RoutePut failure | Allocated capacity leaked permanently | `AbortAllocation` RPC (explicit) + allocation lease TTL (crash) |
| CommitSsdWrite | Non-transactional, no idempotency | Transactional with rollback; idempotent via index + IsRegistered gate |
| store_index | Silently accepted for any value | Rejected at PeerService if != 0 (single-store enforcement) |
| Finalize/Publish timeout | RPC failure treated as definitive; stale entry causes hard read failure | Acknowledged risk (low frequency); TODO: allocation_id-based dedup + read-path retry (§3.12) |
| location_cache_ | Implicit, mixed ownership | Renamed to cluster_locations_ with clear scope |
| SSD staging buffer | Owned by PoolClient | Owned by ClusterCoordinator; pointer shared to PeerService |

---

## 6. Recommended Tests

**Priority**: Tests marked P0 cover the new allocation protocol (FinalizeAllocation / AbortAllocation / lease TTL) — implement these first as they are the most regression-prone area.

| Pri | Test | Validates |
|-----|------|-----------|
| P0 | `RoutePut` → RDMA write fails → `AbortAllocation(target_node_id)` → capacity restored | §2.8 / §3.11.2 explicit rollback |
| P0 | `RoutePut` → client crash (simulate) → allocation TTL expires → reaper reclaims | §3.11.3 lease-based crash recovery |
| P0 | RoutePut SSD write → `FinalizeAllocation` → pending lease cleared, no double-deduction | §3.13 finalize vs publish separation |
| P0 | Two concurrent SSD `RoutePut` to same node → distinct `allocation_id` → no lease collision | §3.11.1 allocation_id uniqueness |
| P0 | `CommitSsdWrite` with SSD write OK but FinalizeAllocation fail → local state rolled back | §3.4 transactional rollback |
| P1 | Peer write → target node `Exists` / `GetIntoPtr` → target eviction → Master unregister | §3.9 storage-owner registration + eviction cleanup |
| P1 | Remote SSD read with `auto_promote_on_read=true` → verify no promote via NoPromote path | §3.5 no-promote read path |
| P1 | Local SSD read with `auto_promote_on_read=true` → verify `on_tier_change_` fires | §2.7 / §3.7 Promote() event fix |
| P1 | `RoutePut` → allocate → heartbeat → verify Master `available_bytes` unchanged | §2.6 heartbeat fix |
| P1 | Local DRAM→SSD demotion → `PublishLocalBlock` → Master SSD allocator capacity decremented | §3.13 local-publish capacity accounting |
| P1 | DRAM write → eviction → SSD demotion → remote node `GetRemote` via PeerService | §3.7 eviction callback re-registration |
| P1 | Two concurrent `RoutePut` to same node → verify no DRAM offset collision | §2.2 / §3.6 single-allocator guarantee |
| P2 | `CommitSsdWrite` → data visible via local `SSDTier` and `LocalBlockIndex` | §3.4 unified storage path |
| P2 | `CommitSsdWrite` retry after first attempt fully committed → idempotent success | §3.4 idempotency (index + IsRegistered) |
| P2 | `CommitSsdWrite` retry after first attempt rolled back → re-executes full sequence | §3.4 retry after rollback |
| P2 | `CommitSsdWrite` with `store_index = 1` → rejected | §3.1 / §3.4 single-store enforcement |
| P2 | UMBPClient construction → PeerService start → `GetPeerInfo` returns valid staging desc | §3.10 two-phase init + staging buffer wiring |
