# UMBPClient ← PoolClient Integration Design Document

**Author:** Dev3
**Status:** Draft
**Scope:** Embed `PoolClient` as an optional member of `UMBPClient` to enable distributed KV cache sharing
**Depends on:** `design-master-control-plane.md`, `design-pool-client.md`

---

## 1. Overview

`UMBPClient` is the **sole public API** for UMBP. It operates in one of two modes,
selected at runtime via `UMBPConfig`:

### Local Mode (default)

Only `UMBPClient` is used. It manages local storage (DRAM + SSD) with no network
dependencies. `pool_client_` is `nullptr`. No Master, no RDMA, no gRPC — the binary
still links against distributed code but none of it is activated.

### Distributed Mode

Enabled by setting `config.distributed` to a valid `UMBPDistributedConfig`.
`UMBPClient` creates a `PoolClient` as an internal member (`pool_client_`), which
handles all cluster interactions:

1. **Register** locally-written blocks with the Master so remote nodes can discover them.
2. **Fetch** blocks from remote nodes on local cache miss via RDMA.
3. **Cache** remotely-fetched blocks into local tiers for future reads.
4. **Serve** local data to remote nodes via RDMA (the DramTier's mmap region IS the RDMA buffer).

The mode is determined purely at runtime: `config_.distributed.has_value()` →
distributed, otherwise → local. There is no compile-time flag.

---

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
│  PoolClient*           pool_client_    ← NEW (optional)      │
│    ├─ MasterClient     (gRPC control plane)                  │
│    │   ├─ RegisterSelf / UnregisterSelf / Heartbeat          │
│    │   ├─ Register / Unregister (block index)                │
│    │   └─ RoutePut / RouteGet  (routing)                     │
│    ├─ IOEngine         (RDMA data plane)                     │
│    │   └─ RegisterMemory(DramTier::GetBasePtr(), capacity)   │
│    ├─ PeerService      (gRPC SSD staging for remote callers) │
│    └─ PeerConnections  (lazy RDMA to remote nodes)           │
└──────────────────────────────────────────────────────────────┘
```

**Key insight:** DramTier's mmap'd buffer is registered with the IOEngine for RDMA.
Remote nodes read/write directly into this buffer at offsets managed by the Master's
`PoolAllocator`. Local writes use DramTier's own slab allocator. These do NOT conflict
because local writes go through `LocalStorageManager` and the Master only assigns offsets
within the RDMA-registered region for *remote-originated* writes.

---

## 3. Data Flow

### 3.1 Put

```
Caller → UMBPClient::Put(key, data, size)
  │
  ├─ index_.MayExist(key)? → return true  (content-addressed dedup)
  │
  ├─ storage_.Write(key, data, size)       → DramTier allocates at offset X
  │   └─ If DRAM full: SelectVictim → Demote to SSD (triggers on_tier_change_ callback)
  │
  ├─ index_.Insert(key, {CPU_DRAM, 0, size})
  │
  ├─ copy_pipeline_->MaybeCopyToSharedSSD(key)  (SharedSSDLeader only)
  │
  └─ [Distributed only] pool_client_->RegisterWithMaster(key, size, "0:<X>", DRAM)
       └─ master_client_->Register(key, {node_id, "0:X", size, DRAM})
          → block is now discoverable cluster-wide
```

**Local mode:** The last step is skipped (`pool_client_` is `nullptr`).

### 3.2 Get (Local Hit)

```
Caller → UMBPClient::GetIntoPtr(key, dst, size)
  │
  ├─ index_.MayExist(key)? → yes
  ├─ storage_.ReadIntoPtr(key, dst, size) → true
  └─ return true
```

**Identical in both modes.** Local hits never touch the network.

### 3.3 Get (Local Miss)

```
Caller → UMBPClient::GetIntoPtr(key, dst, size)
  │
  ├─ index_.MayExist(key)? → no
  ├─ storage_.ReadIntoPtr() → false (or skipped)
  │
  ├─ [Local mode] return false  ← miss is final
  │
  └─ [Distributed mode] pool_client_->FetchRemote(key, dst, size)
      ├─ master_client_->RouteGet(key) → {node_B, "0:1024", DRAM, peer_addr}
      ├─ GetOrConnectPeer(node_B) → RDMA connection
      └─ RemoteDramRead(peer, buf=0, dst, size, offset=1024)
          └─ io_engine_->Read(staging, 0, remote_mem, 1024, size)
      │
      └─ Cache locally (if distributed.cache_remote_fetches):
          ├─ storage_.WriteFromPtr(key, dst, size)
          ├─ index_.Insert(key, {CPU_DRAM, 0, size})
          └─ pool_client_->RegisterWithMaster(key, size, "0:<Y>", DRAM)
             → block is now cached locally AND registered with Master
```

### 3.4 Remove

```
Caller → UMBPClient::Remove(key)
  │
  ├─ index_.Remove(key)
  ├─ storage_.Evict(key)
  └─ [Distributed only] pool_client_->UnregisterFromMaster(key)
       └─ master_client_->Unregister(key, location)
```

**Local mode:** Only the first two steps execute.

### 3.5 Eviction / Demotion (DRAM → SSD tier change)

```
LocalStorageManager::DemoteLRUForSpace(dram_tier)
  │
  ├─ SelectVictim(dram_tier) → victim_key
  ├─ MoveKey(victim_key, dram → ssd)
  └─ on_tier_change_(victim_key, CPU_DRAM, LOCAL_SSD)
       └─ [Distributed only] UMBPClient callback:
          └─ pool_client_->UpdateMasterRegistration(key, new_location_id, SSD)
             └─ Unregister old DRAM location, Register new SSD location
```

**Local mode:** `on_tier_change_` callback is not set, so no Master update occurs.

---

## 4. Files to Modify

### Phase 1: Foundation (PoolClient member + heartbeat)

| File | Change |
|------|--------|
| `include/umbp/common/config.h` | Add `UMBPDistributedConfig` struct with fields: `master_address`, `node_id`, `node_address`, `auto_heartbeat`, `io_engine_host`, `io_engine_port`, `staging_buffer_size`, `peer_service_port`, `cache_remote_fetches`. Add `std::optional<UMBPDistributedConfig> distributed` to `UMBPConfig` (nullopt = local mode). Extend `Validate()`. |
| `include/umbp/local/umbp_client.h` | `#include "umbp/distributed/pool_client.h"`. Add `std::unique_ptr<mori::umbp::PoolClient> pool_client_` member. Add `bool IsDistributed() const { return pool_client_ != nullptr; }` helper. |
| `local/umbp_client.cpp` | When `config.distributed` is set, construct `PoolClient` with master config + network config and call `Init()`. PoolClient connects to Master, calls `RegisterSelf()`, and starts heartbeat. No delegation pointers, no DRAM/SSD export, no changes to Put/Get/Remove yet. On init failure, log error and reset `pool_client_` to fall back to local mode. |
| `CMakeLists.txt` | Link `umbp_core` → `umbp_common` unconditionally. |

**After Phase 1:** UMBPClient in distributed mode connects to the Master, registers
itself as a node, and sends periodic heartbeats. All Put/Get/Remove behavior is
unchanged (local-only). The Master can see the node is alive.

### Phase 2: PoolClient Refactor + DramTier Accessors

| File | Change |
|------|--------|
| `include/umbp/local/storage/dram_tier.h` | Add `void* GetBasePtr() const` (returns `base_ptr_`) and `std::optional<size_t> GetSlotOffset(const std::string& key) const` (looks up `slots_[key].offset` under lock, returns nullopt if key not found). |
| `local/storage/dram_tier.cpp` | Implement `GetSlotOffset()`. |
| `include/umbp/distributed/pool_client.h` | Add delegation pointers to `PoolClientConfig`: `local_storage`, `local_index`, `dram_base_ptr`, `dram_capacity`, `ssd_dir`, `ssd_capacity`. Add new methods: `RegisterWithMaster(key, size, location_id, tier)`, `FetchRemote(key, dst, size)`, `UnregisterFromMaster(key)`. |
| `distributed/pool_client.cpp` | Implement new methods. In `Init()`, when `dram_base_ptr` is set, register it as the RDMA-exportable buffer instead of iterating `dram_buffers`. When `local_storage` delegate is set, redirect `PutLocalDram/GetLocalDram/PutLocalSsd/GetLocalSsd` to delegate (for remote-originated writes routed to this node). |

### Phase 3: Wire UMBPClient → PoolClient

| File | Change |
|------|--------|
| `local/umbp_client.cpp` | Extend constructor to set delegation pointers (`local_storage`, `local_index`), export DramTier buffer for RDMA (`dram_base_ptr`, `dram_capacity`), export SSD config (`ssd_dir`, `ssd_capacity`), and populate `tier_capacities`. Install `on_tier_change_` callback that unregisters old location and registers new location on tier change. Modify `Put`, `PutFromPtr`, `BatchPutFromPtr`, `BatchPutFromPtrWithDepth`: after local write, if `pool_client_` is non-null, get slot offset from DramTier and call `RegisterWithMaster`. Modify `GetIntoPtr`, `BatchGetIntoPtr`: on local miss, if `pool_client_` is non-null, call `FetchRemote`; on success, optionally cache locally and register with Master. Modify `Remove`: if `pool_client_` is non-null, call `UnregisterFromMaster`. `Clear`: let Master reap stale entries via heartbeat timeout. |
| `include/umbp/local/storage/local_storage_manager.h` | Add `TierChangeCallback` typedef and `SetOnTierChange()` method. |
| `local/storage/local_storage_manager.cpp` | Call `on_tier_change_` in `MoveKey()` and `Evict()` after successful tier change. |

### Phase 4: Build System + Python Bindings

| File | Change |
|------|--------|
| `src/pybind/pybind_umbp.cpp` | Bind `UMBPDistributedConfig` class with all fields. Add `.def_readwrite("distributed", ...)` to `UMBPConfig`. |
| `src/pybind/CMakeLists.txt` | Link pybinds to `umbp_common`. |

### Phase 5: Python Consumer Adaptation

| File | Change |
|------|--------|
| `sglang/.../umbp_store.py` | Read distributed config from `extra_config`. When `distributed_enabled` is set, construct `UMBPDistributedConfig`, populate fields from extra config, and assign to `cfg.distributed`. No other changes to `UMBPStore` — Put/Get/Remove are transparently enhanced by `pool_client_` inside `UMBPClient`. |

---

## 5. DRAM Offset Conflict Resolution

### Problem

Two separate allocators manage offsets in the same DramTier mmap region:

- **DramTier's slab allocator** — used by local writes via `LocalStorageManager`
- **Master's PoolAllocator** — assigns offsets for remote RDMA writes to this node

If both allocate independently, they could assign overlapping offsets.

### Solution Options

**Option A: Split the DRAM region** (if remote-write-to-peer is needed)

Partition DramTier's buffer into two zones:
- `[0, local_capacity)` — managed by DramTier's slab allocator for local writes
- `[local_capacity, total_capacity)` — managed by Master's PoolAllocator for remote writes

**Option B: No remote writes to self** (simpler, recommended for initial integration)

Tell the Master this node's DRAM capacity is 0 for `RoutePut` (never assign remote
writes here). Only call `RegisterWithMaster` after local writes to make blocks
discoverable. Remote nodes read via RDMA at the offset reported in `location_id`.

This avoids the split entirely. The Master never assigns DRAM space on this node for
remote writers — all writes are local-first. Works well when each node writes its own
data and only reads from others.

**Recommended:** Option B for initial integration. Option A if remote-write-to-peer
is needed later.

---

## 6. Thread Safety

| Interaction | Direction | Safety |
|-------------|-----------|--------|
| `UMBPClient` → `PoolClient::RegisterWithMaster` | After local write | Sequential in Put path; PoolClient has its own `cache_mutex_` |
| `UMBPClient` → `PoolClient::FetchRemote` | On local miss in Get | Sequential in Get path; RDMA and staging have own mutexes |
| `LocalStorageManager` → `on_tier_change_` callback | During eviction | Callback must NOT call back into `LocalStorageManager` (deadlock risk) |
| `PoolClient` → `config_.local_storage` (delegation) | For remote-originated local writes | `LocalStorageManager` tier backends have their own locks |

**No circular lock dependency**: UMBPClient → PoolClient → Master (gRPC, network).
PoolClient → LocalStorageManager → TierBackend. These are one-directional.

The `on_tier_change_` callback fires inside `LocalStorageManager::MoveKey()` which
may hold tier locks. The callback must only call `PoolClient` methods (which go to
gRPC/network, no local locks). This is safe as long as the callback does NOT call
`storage_.Write/Read/Evict`.

---

## 7. Testing Strategy

### Unit Tests

| Test | Description |
|------|-------------|
| `test_standalone_unchanged` | `UMBPClient` with no `distributed` → identical to current behavior |
| `test_config_validation` | `distributed` with empty `master_address` → `Validate()` fails |
| `test_dram_get_base_ptr` | `DramTier::GetBasePtr()` returns non-null after construction |
| `test_dram_get_slot_offset` | Write key, verify offset returned. Evict, verify nullopt. |
| `test_pool_client_register` | `RegisterWithMaster` → Master sees the block |
| `test_pool_client_fetch_remote` | `FetchRemote` retrieves data from remote node |
| `test_pool_client_unregister` | `UnregisterFromMaster` → Master no longer has the block |

### Integration Tests

| Test | Description |
|------|-------------|
| `test_two_node_put_get` | Node A puts, Node B gets via RDMA. Verify data integrity. |
| `test_remote_cache_local` | After remote fetch, verify `Exists()` returns true on fetcher. |
| `test_remove_propagation` | Put on A, Remove on A. Get from B fails. |
| `test_eviction_tier_update` | Fill DRAM past watermark. Verify Master registration updates. |

### E2E Test

Extend existing `test_umbp_integration.sh`:
- Launch Master server
- Launch SGLang with `distributed_enabled=true` in extra_config
- Run GSM8K benchmark, verify accuracy >= 0.95

---

## 8. Implementation Sequence

```
Phase 1  ──────────────────────────────────────────────────  PR #1
  ├─ config.h: UMBPDistributedConfig + optional field
  ├─ umbp_client.h: #include pool_client.h, pool_client_ member
  ├─ umbp_client.cpp: construct PoolClient when config.distributed is set
  ├─ CMakeLists.txt: link umbp_core → umbp_common unconditionally
  ├─ After this PR: UMBPClient connects to Master and sends heartbeats
  └─ All existing tests pass, Put/Get/Remove unchanged

Phase 2  ──────────────────────────────────────────────────  PR #2
  ├─ dram_tier.h/cpp: GetBasePtr(), GetSlotOffset()
  ├─ pool_client.h/cpp: delegation pointers, new methods
  ├─ pool_client_main.cpp: adapt for new config shape
  └─ Existing distributed tests pass

Phase 3  ──────────────────────────────────────────────────  PR #3
  ├─ umbp_client.cpp: wire DRAM/SSD export, delegation, tier_capacities
  ├─ umbp_client.cpp: wire Put/Get/Remove with runtime pool_client_ checks
  ├─ local_storage_manager.h/cpp: on_tier_change_ callback
  └─ New distributed unit + integration tests

Phase 4  ──────────────────────────────────────────────────  PR #4
  ├─ pybind_umbp.cpp: bind UMBPDistributedConfig
  ├─ umbp_store.py: read distributed config from extra_config
  └─ E2E test with SGLang
```

---

## 9. Open Questions

1. **Should `Exists()` query remote?** Currently `Exists()` is local-only. For distributed
   mode, should it also ask the Master? This adds a gRPC round-trip per call. Recommendation:
   keep `Exists()` local-only; callers who need remote existence checking should use `GetIntoPtr`.

2. **Should `BatchGetIntoPtr` fetch remote in parallel?** Current `FetchRemote` is serial per
   key. For batch misses, launching parallel RDMA reads would improve throughput. This can be
   a follow-up optimization.

3. **Eviction vs. Master registration**: When a block is fully evicted (removed from both DRAM
   and SSD), should we unregister from Master? Yes — otherwise remote nodes will try to read
   stale data. The `on_tier_change_` callback with `new_tier = EVICTED` (or a separate
   `on_evict_` callback) handles this.

4. **`Clear()` behavior**: Should `Clear()` unregister all blocks from Master? For bulk reset
   this could be expensive. Alternative: let the Master reap stale entries via heartbeat
   timeout after the node re-registers with fresh state.
