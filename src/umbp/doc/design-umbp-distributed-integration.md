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
  └─ [Distributed mode] pool_client_->GetRemote(key, dst, size)
      ├─ master_client_->RouteGet(key) → {node_B, "0:1024", DRAM, peer_addr}
      ├─ GetOrConnectPeer(node_B) → RDMA connection
      └─ RemoteDramRead(peer, buf=0, dst, size, offset=1024)
          └─ io_engine_->Read(staging, 0, remote_mem, 1024, size)
      │
      └─ Cache locally (if distributed.cache_remote_fetches):
          ├─ storage_.WriteFromPtr(key, dst, size)     ← UMBPClient calls directly
          ├─ index_.Insert(key, {CPU_DRAM, 0, size})   ← UMBPClient calls directly
          └─ pool_client_->RegisterWithMaster(key, size, "0:<Y>", DRAM)
             → block is now cached locally AND registered with Master
```

**Phase 3 (DRAM-only):** `FetchRemote` only supports RDMA reads from remote DRAM.
If the remote block is on SSD, the fetch fails and returns false.

**Phase 6 (SSD support):** `FetchRemote` for SSD-resident remote blocks goes through
PeerService gRPC staging (see Phase 6 data flow diagram).

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

**Phase 3 (DRAM-only):**
```
LocalStorageManager::DemoteLRUForSpace(dram_tier)
  │
  ├─ SelectVictim(dram_tier) → victim_key
  ├─ MoveKey(victim_key, dram → ssd)
  └─ on_tier_change_(victim_key, CPU_DRAM, LOCAL_SSD)
       └─ [Distributed only] UMBPClient callback:
          └─ pool_client_->UnregisterFromMaster(key)
             → block is no longer remotely accessible (SSD not served yet)
```

**Phase 6 (SSD support):**
```
  └─ on_tier_change_(victim_key, CPU_DRAM, LOCAL_SSD)
       └─ [Distributed only] UMBPClient callback:
          └─ pool_client_->UpdateMasterRegistration(key, new_location_id, SSD)
             → Unregister old DRAM location, Register new SSD location
             → block remains remotely accessible via PeerService staging
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

### Phase 2: PoolClient Refactor + DramTier Accessors (DRAM-only)

| File | Change |
|------|--------|
| `include/umbp/local/storage/dram_tier.h` | Add `void* GetBasePtr() const` (returns `base_ptr_`) and `std::optional<size_t> GetSlotOffset(const std::string& key) const` (looks up `slots_[key].offset` under lock, returns nullopt if key not found). |
| `local/storage/dram_tier.cpp` | Implement `GetSlotOffset()`. |
| `include/umbp/distributed/pool_client.h` | No delegation pointers, no new config fields. Add new methods: `RegisterWithMaster(key, size, location_id, tier)`, `GetRemote(key, dst, size)`, `PutRemote(key, src, size)`, `UnregisterFromMaster(key)`. Keep `PutLocalDram/GetLocalDram/PutLocalSsd/GetLocalSsd` as-is — they are bypassed by the new methods but retained for existing tests. |
| `distributed/pool_client.cpp` | Implement new methods. DRAM buffer registration uses the existing `RegisterMemory(ptr, size)` method — UMBPClient calls it in Phase 3 with `DramTier::GetBasePtr()`. No delegation redirects — PoolClient only handles Master registration and RDMA. |

**Design principle:** No delegation functions. UMBPClient owns both `storage_` and
`pool_client_` and orchestrates all flows top-down. PoolClient never touches local
storage directly — it only handles cluster interactions (Master gRPC + RDMA).

**New methods are the remote-only subsets of the current `Put`/`Get`:**

| Current method | New method | What it extracts |
|----------------|------------|------------------|
| `Put` (lines 294–377) | `RegisterWithMaster` | The `master_client_->Register()` call at the end (line 364). UMBPClient already wrote locally; this just makes the block discoverable. |
| `Put` (lines 294–377) | `PutRemote` | The remote DRAM branch (lines 344–350): `RoutePut` → `GetOrConnectPeer` → `RemoteDramWrite`. Only the `!is_local && DRAM` case. |
| `Get` (lines 379–422) | `GetRemote` | The remote DRAM branch (lines 406–411): `RouteGet` → `GetOrConnectPeer` → `RemoteDramRead`. Only the `!is_local && DRAM` case. |
| `Put`/`Get` | (removed) | Local DRAM/SSD branches → handled by UMBPClient via `storage_` directly. |
| `Put`/`Get` | (Phase 6) | Remote SSD branches (`RemoteSsdWrite`/`RemoteSsdRead`) → deferred. |

**DRAM-only scope:** `GetRemote` and `PutRemote` only support RDMA reads/writes
to remote DRAM. If the target block is on a remote node's SSD, the operation fails.
SSD support is deferred to Phase 6.

### Phase 3: Wire UMBPClient → PoolClient (DRAM-only)

| File | Change |
|------|--------|
| `local/umbp_client.cpp` | Extend constructor to export DramTier buffer for RDMA via `pool_client_->RegisterMemory(dram_tier.GetBasePtr(), dram_capacity)`. No new config fields needed, no delegation pointers, no SSD export. Install `on_tier_change_` callback: on DRAM→SSD demotion, **unregister** the block from Master (block is no longer remotely accessible until Phase 6 adds SSD serving). On full eviction, also unregister. Modify `Put`, `PutFromPtr`, `BatchPutFromPtr`, `BatchPutFromPtrWithDepth`: after local write, if `pool_client_` is non-null, get slot offset from DramTier and call `RegisterWithMaster` (DRAM tier only). Modify `GetIntoPtr`, `BatchGetIntoPtr`: on local miss, if `pool_client_` is non-null, call `GetRemote` (DRAM-only RDMA); on success, optionally cache locally and register with Master. Modify `Remove`: if `pool_client_` is non-null, call `UnregisterFromMaster`. `Clear`: let Master reap stale entries via heartbeat timeout. |
| `include/umbp/local/storage/local_storage_manager.h` | Add `TierChangeCallback` typedef and `SetOnTierChange()` method. |
| `local/storage/local_storage_manager.cpp` | Call `on_tier_change_` in `MoveKey()` and `Evict()` after successful tier change. |

**DRAM-only implication:** Only blocks in DRAM are visible to the cluster. When a block
is demoted to SSD, it becomes "local-only" again until Phase 6. This is acceptable
because hot blocks stay in DRAM, and nodes primarily read recent data from peers.

### Phase 4: Build System + Python Bindings

| File | Change |
|------|--------|
| `src/pybind/pybind_umbp.cpp` | Bind `UMBPDistributedConfig` class with all fields. Add `.def_readwrite("distributed", ...)` to `UMBPConfig`. |
| `src/pybind/CMakeLists.txt` | Link pybinds to `umbp_common`. |

### Phase 5: Python Consumer Adaptation

| File | Change |
|------|--------|
| `sglang/.../umbp_store.py` | Read distributed config from `extra_config`. When `distributed_enabled` is set, construct `UMBPDistributedConfig`, populate fields from extra config, and assign to `cfg.distributed`. No other changes to `UMBPStore` — Put/Get/Remove are transparently enhanced by `pool_client_` inside `UMBPClient`. |

### Phase 6: Remote SSD Support

| File | Change |
|------|--------|
| `include/umbp/local/umbp_client.h` | Add `bool ReadForRemotePeer(const std::string& key, void* dst, size_t size)` — public method for PeerService to request data that may reside on SSD. UMBPClient reads from `storage_` (which handles DRAM/SSD transparently), stages into `dst`, and returns success. This keeps UMBPClient as the **sole entry point** to local storage. |
| `local/umbp_client.cpp` | Implement `ReadForRemotePeer`: read from `storage_.ReadIntoPtr()`. If the block is on SSD, this triggers SSD I/O transparently. No index mutation, no Master registration — this is a read-only serving path. |
| `include/umbp/distributed/pool_client.h` | Add `SetRemoteReadCallback(std::function<bool(key, dst, size)>)` to `PoolClient`. PeerService uses this callback when a remote node requests a block that is on SSD (not directly RDMA-readable). |
| `distributed/pool_client.cpp` | Wire PeerService to use the callback for SSD-resident block requests. When a remote `RouteGet` returns a SSD location on this node, PeerService calls the callback to stage data into RDMA-exportable memory, then completes the RDMA transfer. |
| `distributed/peer_service.cpp` | On incoming remote read for a SSD-resident block: call `remote_read_callback_(key, staging_buf, size)` instead of directly accessing SSD. Stage the result into RDMA buffer and complete the transfer. |
| `local/umbp_client.cpp` | In constructor (distributed mode): call `pool_client_->SetRemoteReadCallback(...)` with a lambda that calls `this->ReadForRemotePeer(...)`. Update `on_tier_change_` callback: on DRAM→SSD demotion, **re-register** with Master at new SSD location (instead of unregistering). Block remains remotely accessible via PeerService staging. |

**Design principle:** PeerService never touches `LocalStorageManager` or `SsdTier`
directly. Instead it calls back through UMBPClient via a `std::function` callback.
This ensures:
- **Single entry point:** All local storage access flows through UMBPClient
- **Thread safety:** UMBPClient controls locking order — no risk of PeerService
  acquiring locks in wrong order
- **No circular header dependency:** PeerService holds a `std::function`, not a
  `UMBPClient*`

**Data flow for remote SSD read:**
```
Remote Node B                          This Node A (block on SSD)
    │                                       │
    ├─ RouteGet(key) → Master ──────────► {node_A, ssd_location}
    │                                       │
    ├─ gRPC to PeerService(node_A) ────►  PeerService receives request
    │                                       ├─ remote_read_callback_(key, staging, size)
    │                                       │   └─ UMBPClient::ReadForRemotePeer()
    │                                       │       └─ storage_.ReadIntoPtr() (SSD I/O)
    │                                       ├─ Data now in staging buffer (RDMA-registered)
    │                                       └─ RDMA write staging → Node B
    ◄──────────────────────────────────── Data arrives at Node B
```

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

### Phase 3 (DRAM-only, no delegation)

| Interaction | Direction | Safety |
|-------------|-----------|--------|
| `UMBPClient` → `PoolClient::RegisterWithMaster` | After local write | Sequential in Put path; PoolClient has its own `cache_mutex_` |
| `UMBPClient` → `PoolClient::GetRemote` | On local miss in Get | Sequential in Get path; RDMA and staging have own mutexes |
| `LocalStorageManager` → `on_tier_change_` callback | During eviction | Callback must NOT call back into `LocalStorageManager` (deadlock risk) |

**No delegation = no reverse calls.** Lock order is strictly one-directional:
UMBPClient → PoolClient → Master (gRPC, network). PoolClient never calls back
into UMBPClient or LocalStorageManager.

The `on_tier_change_` callback fires inside `LocalStorageManager::MoveKey()` which
may hold tier locks. The callback must only call `PoolClient` methods (which go to
gRPC/network, no local locks). This is safe as long as the callback does NOT call
`storage_.Write/Read/Evict`.

### Phase 6 (SSD support adds one reverse path)

| Interaction | Direction | Safety |
|-------------|-----------|--------|
| `PeerService` → `remote_read_callback_` → `UMBPClient::ReadForRemotePeer` | Remote read request | PeerService holds no UMBPClient locks; `ReadForRemotePeer` acquires `storage_` locks independently |

**New reverse path:** PeerService → UMBPClient → `storage_`. This is safe because:
- PeerService uses a `std::function` callback, not a direct `UMBPClient*` (no header dependency)
- PeerService holds no locks from UMBPClient or LocalStorageManager when calling the callback
- `ReadForRemotePeer` is read-only — it does not mutate `index_` or call `PoolClient`
- Lock order: PeerService (gRPC thread) → UMBPClient::ReadForRemotePeer → storage_ tier locks. No overlap with the forward path (UMBPClient → PoolClient → PeerService)

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
| `test_pool_client_get_remote` | `GetRemote` retrieves data from remote node's DRAM via RDMA |
| `test_pool_client_put_remote` | `PutRemote` writes data to remote node's DRAM via RDMA |
| `test_pool_client_unregister` | `UnregisterFromMaster` → Master no longer has the block |

### Integration Tests (Phase 3, DRAM-only)

| Test | Description |
|------|-------------|
| `test_two_node_put_get` | Node A puts, Node B gets via RDMA. Verify data integrity. |
| `test_remote_cache_local` | After remote fetch, verify `Exists()` returns true on fetcher. |
| `test_remove_propagation` | Put on A, Remove on A. Get from B fails. |
| `test_eviction_unregisters` | Fill DRAM past watermark. Verify demoted block is **unregistered** from Master. Remote Get fails. |

### Integration Tests (Phase 6, SSD support)

| Test | Description |
|------|-------------|
| `test_remote_ssd_read` | Node A puts, block demoted to SSD. Node B gets via PeerService staging. Verify data integrity. |
| `test_eviction_tier_update` | Fill DRAM past watermark. Verify Master registration **updates** to SSD location (not unregistered). Remote Get succeeds via PeerService. |
| `test_peer_service_callback` | Verify PeerService calls `ReadForRemotePeer` (not SsdTier directly) for SSD-resident blocks. |

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
  ├─ pool_client.h/cpp: no delegates, DRAM-only methods
  │   ├─ RegisterWithMaster, GetRemote, PutRemote, UnregisterFromMaster
  │   └─ Keep PutLocalDram/GetLocalDram/PutLocalSsd/GetLocalSsd (existing tests)
  ├─ pool_client_main.cpp: adapt for new config shape
  └─ Existing distributed tests pass

Phase 3  ──────────────────────────────────────────────────  PR #3
  ├─ umbp_client.cpp: call RegisterMemory(GetBasePtr(), capacity) (no SSD export)
  ├─ umbp_client.cpp: wire Put/Get/Remove with runtime pool_client_ checks
  ├─ on_tier_change_: DRAM→SSD = unregister from Master (DRAM-only)
  ├─ local_storage_manager.h/cpp: on_tier_change_ callback
  └─ New distributed unit + integration tests (DRAM-only)

Phase 4  ──────────────────────────────────────────────────  PR #4
  ├─ pybind_umbp.cpp: bind UMBPDistributedConfig
  ├─ umbp_store.py: read distributed config from extra_config
  └─ E2E test with SGLang

Phase 5  ──────────────────────────────────────────────────  PR #5
  └─ (Python consumer adaptation, same as before)

Phase 6  ──────────────────────────────────────────────────  PR #6
  ├─ umbp_client.h/cpp: ReadForRemotePeer() method
  ├─ pool_client.h/cpp: SetRemoteReadCallback(), wire PeerService
  ├─ peer_service.cpp: use callback for SSD-resident block requests
  ├─ umbp_client.cpp: update on_tier_change_ to re-register SSD location
  │   (instead of unregistering — block stays remotely accessible)
  └─ Tests: remote SSD read, tier demotion with continued remote access
```

---

## 9. Open Questions

1. **Should `Exists()` query remote?** Currently `Exists()` is local-only. For distributed
   mode, should it also ask the Master? This adds a gRPC round-trip per call. Recommendation:
   keep `Exists()` local-only; callers who need remote existence checking should use `GetIntoPtr`.

2. **Should `BatchGetIntoPtr` fetch remote in parallel?** Current `FetchRemote` is serial per
   key. For batch misses, launching parallel RDMA reads would improve throughput. This can be
   a follow-up optimization.

3. **Eviction vs. Master registration**: Resolved by phased approach. Phase 3 (DRAM-only):
   any demotion out of DRAM unregisters from Master. Phase 6 (SSD support): DRAM→SSD
   re-registers at SSD location; full eviction (removed from both tiers) unregisters.
   The `on_tier_change_` callback handles both cases.

4. **`Clear()` behavior**: Should `Clear()` unregister all blocks from Master? For bulk reset
   this could be expensive. Alternative: let the Master reap stale entries via heartbeat
   timeout after the node re-registers with fresh state.
