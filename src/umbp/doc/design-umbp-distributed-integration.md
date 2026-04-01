# UMBPClient ← PoolClient Integration Design

**Scope:** Embed `PoolClient` as an optional member of `UMBPClient` to enable distributed KV cache sharing
**Depends on:** `design-master-control-plane.md`, `design-pool-client.md`

---

## 1. Overview

`UMBPClient` is the **sole public API** for UMBP. It operates in one of two modes,
selected at runtime via `UMBPConfig`:

- **Local Mode** (default): `pool_client_` is `nullptr`. DRAM + SSD only, no network.
- **Distributed Mode**: `config.distributed` is set. `UMBPClient` creates a `PoolClient`
  and optionally a `PeerServiceServer` as internal members.

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    UMBPClient  (public API)                   │
│  Put / Get / BatchPut / BatchGet / Exists / Remove / Clear   │
├──────────────────────────────────────────────────────────────┤
│  LocalBlockIndex       index_          (key → tier/offset)   │
│  LocalStorageManager   storage_        (tiered write/read)   │
│    ├─ DramTier   (mmap slab, LRU, RDMA-registered)          │
│    └─ SsdTier    (segmented log, io_uring/posix)             │
│  CopyPipeline          copy_pipeline_  (async DRAM→SSD)      │
│                                                              │
│  PoolClient*           pool_client_    (optional)            │
│    ├─ MasterClient     (gRPC control plane)                  │
│    ├─ IOEngine         (RDMA data plane)                     │
│    └─ PeerConnections  (lazy RDMA to remote nodes)           │
│                                                              │
│  PeerServiceServer*    peer_service_   (optional, gRPC)      │
│    └─ Receives injected refs: storage_, index_, *pool_client_│
│       Handles remote SSD read/write via staging slots        │
└──────────────────────────────────────────────────────────────┘
```

**Key properties:**
- `PeerServiceServer` is owned by `UMBPClient` (not PoolClient), constructed via
  two-phase init after PoolClient is ready.
- PeerService accesses local SSD via injected `LocalStorageManager&` reference
  (calls `storage_.Write()` and `storage_.ReadIntoPtrNoPromote()`), not through
  UMBPClient callbacks or raw POSIX I/O.
- DramTier's mmap'd buffer is RDMA-registered. Remote DRAM reads go directly
  to this buffer. SSD reads go through PeerService staging slots.

---

## 3. Data Flow

### 3.1 Put (local-first, then register)

```
UMBPClient::Put(key, data, size)
  ├─ dedup: index_.MayExist(key)? → return true
  ├─ storage_.Write(key, data, size) → DRAM
  ├─ index_.Insert(key, {CPU_DRAM, 0, size})
  ├─ [distributed] MaybePublishLocal(key, size) → PublishLocalBlock(key, DRAM)
  └─ copy_pipeline_->MaybeCopyToSharedSSD(key)
```

### 3.2 Get (local hit → remote DRAM → remote SSD)

```
UMBPClient::GetIntoPtr(key, dst, size)
  ├─ storage_.ReadIntoPtr(key, dst, size) → true? return true
  │
  └─ [distributed] pool_client_->GetRemote(key, dst, size)
      ├─ RouteGet(key) → Location{node, tier, location_id}
      │
      ├─ tier==DRAM: direct RDMA read from remote DRAM buffer
      │
      └─ tier==SSD: slot-based staging
          ├─ PrepareSsdRead → server allocates slot, reads SSD (lock-free)
          ├─ RDMA read from staging slot
          └─ ReleaseSsdLease (fire-and-forget with retry)
```

`Exists()` also queries remote via `pool_client_->ExistsRemote()` when distributed.

### 3.3 Eviction / Tier Change Callback

When a block moves between tiers (DRAM ↔ SSD) or is fully evicted:

```
on_tier_change_(key, from_tier, to_tier, new_location)
  ├─ Always: pool_client_->UnregisterFromMaster(key)
  ├─ to == LOCAL_SSD: pool_client_->PublishLocalBlock(key, size, loc, SSD)
  ├─ to == CPU_DRAM:  pool_client_->PublishLocalBlock(key, size, loc, DRAM)
  └─ to == nullopt:   (fully evicted, just unregister)
```

This ensures blocks remain discoverable after DRAM→SSD demotion.

### 3.4 Remove

```
UMBPClient::Remove(key)
  ├─ index_.Remove(key)
  ├─ storage_.Evict(key)
  └─ [distributed] pool_client_->UnregisterFromMaster(key)
```

---

## 4. SSD Staging: Slot-Based Lease Protocol

Remote SSD operations use a staging buffer divided into fixed-size slots to
prevent concurrent data corruption.

### Buffer Layout

```
┌──────────────────────┬──────────────────────┐
│  Write slots (N_W)   │  Read slots (N_R)    │
│  [0, size/2)         │  [size/2, size)      │
└──────────────────────┴──────────────────────┘
```

Default: 64MB total → 8 write slots + 8 read slots → 4MB each.

### Read Path

```
Client                              Server (PeerService)
  ├─ PrepareSsdRead(key, size) ───► AllocateSlot → lease_id
  │                                 SSD → staging[slot] (lock-free I/O)
  ◄─ (offset, lease_id, ttl_ms) ──┘
  ├─ RDMA Read(staging[slot])
  └─ ReleaseSsdLease(lease_id) ──► free slot
```

### Write Path

```
Client                              Server (PeerService)
  ├─ AllocateWriteSlot(size) ────► AllocateSlot → lease_id
  ◄─ (offset, lease_id, ttl_ms) ─┘
  ├─ RDMA Write → staging[slot]
  └─ CommitSsdWrite(key, lease_id) ► validate lease + size + offset
                                     staging → SSD (lock-free)
                                     FinalizeAllocation
                                     free slot
```

### Concurrency Design

- `read_slots_mutex_` and `write_slots_mutex_` are separate — reads and writes
  don't block each other.
- Mutexes only protect slot metadata (allocation/release), never held during I/O.
- `next_lease_id_` is `std::atomic<uint64_t>` — safe across both mutexes.
- All slot release paths use `ReleaseSlotByLeaseId(lease_id)` to prevent
  accidentally freeing a slot that was TTL-reclaimed and reassigned.
- TTL default: 10 seconds. Lazy GC scans on every allocation attempt.
- `StagingMetrics` tracks: `expired_reclaims`, `invalid_lease_rejects`, `slot_full_rejects`.

### Validation in CommitSsdWrite

1. `lease_id` → find slot (reject if expired/invalid)
2. `store_index == 0` (single-store enforcement)
3. `request.size() <= slot.allocated_size`
4. `request.staging_offset() == slot_offset`

All failures release the slot immediately.

---

## 5. DRAM Offset Conflict Resolution

Two allocators manage the same DramTier mmap region:
- **DramTier slab allocator** — local writes
- **Master PoolAllocator** — remote RDMA writes

**Current approach (Option B):** Report DRAM `available_bytes = 0` to Master.
Master never routes remote writes to this node's DRAM. Local-first writes only.

---

## 6. Thread Safety

| Interaction | Direction | Safety |
|-------------|-----------|--------|
| `UMBPClient` → `PoolClient` methods | Forward | Sequential in Put/Get paths |
| `on_tier_change_` callback → `PoolClient` | Forward | Callback fires under tier locks; must NOT call `storage_` (deadlock) |
| `PeerService` → `storage_.ReadIntoPtrNoPromote` | Reverse (injected ref) | PeerService holds no UMBPClient/LSM locks; acquires tier locks independently |
| `PeerService` → `pool_client_.FinalizeAllocation` | Reverse (injected ref) | Goes to gRPC (network), no local lock contention |

Lock order: `UMBPClient → PoolClient → Master (network)`. No cycles.

---

## 7. Testing

### Existing Tests

| Test File | Coverage |
|-----------|----------|
| `test_peer_service.cpp` (19 cases) | PeerService gRPC: slot alloc/release/exhaust, TTL reclaim, lease/size/offset validation, slot isolation |
| `test_client_registry.cpp` | ClientRegistry: register/unregister/heartbeat/reaper (memory-level, no gRPC) |
| `test_router.cpp` | RouteGet/RoutePut strategies (memory-level) |
| `test_umbp_integration.sh` | E2E: SGLang + UMBP distributed mode, GSM8K accuracy benchmark |

### Test Gaps (TODO)

- Master + MasterClient gRPC integration test (RegisterSelf → PublishLocalBlock → RouteGet round-trip)
- `GetRemote` SSD dispatch E2E (requires RDMA environment)
- Concurrent slot TTL-reclaim-during-RDMA stress test

---

## 8. Resolved Architectural Issues

These issues were identified during the local/distributed unification effort.
All have been addressed in the current implementation.

| Issue | Resolution |
|-------|------------|
| Two independent SSD implementations (segment log vs raw `.bin`) | PeerService now uses `LocalStorageManager` (segment log via SSDTier) |
| DRAM buffer dual management (slab allocator vs Master PoolAllocator) | DRAM `available_bytes = 0` reported to Master (Option B, local-first) |
| Eviction callback only covers DRAM | `on_tier_change_` handles all tier transitions: Unregister + PublishLocalBlock |
| PeerService bypasses local storage stack | PeerService uses injected `storage_` ref (`Write`, `ReadIntoPtrNoPromote`) |
| Heartbeat overwrites Master capacity accounting | Heartbeat ignores capacity fields (liveness-only) |
| `Promote()` does not fire `on_tier_change_` | Fixed: all Promote paths now call `on_tier_change_` |
| RoutePut allocation lacks failure rollback | `AbortAllocation` RPC + allocation lease TTL reaper |
| Conflicting routing philosophies | Local-first writes via UMBPClient; PoolClient handles only cluster coordination |

---

## 9. Open Questions

1. **`cache_remote_fetches`**: Config field exists but not implemented in `GetIntoPtr`.
   After successful `GetRemote`, data is returned but not cached locally.

2. **`BatchGetIntoPtr` parallel remote**: Currently serial per key. Batch RDMA or
   batch `PrepareSsdRead` could improve throughput.

3. **`Clear()` and Master**: `Clear()` only resets local state. Master entries
   become stale until heartbeat timeout triggers reaper cleanup.
