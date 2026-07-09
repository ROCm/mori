# UMBP Standalone Process Mode — Design (Draft v0.5)

**Scope:** A new deployment mode for UMBP in which the DRAM/SSD tiers,
`LocalBlockIndex`, and `CopyPipeline` run inside a dedicated, long-lived
OS process on the same host, while one or more SGLang/vLLM worker
processes talk to it through `IUMBPClient` as if it were still
in-process. This is *not* the existing `distributed/` mode (separate
master + multi-node peers over gRPC + RDMA) — it targets the
**single-host, multi-worker-process** case that today forces every
worker to duplicate its own DRAM tier and lose the ability to share a
KV cache across sibling processes (e.g. TP ranks, or a restarted
worker resuming a warm cache).

**Status:** design proposal, not yet implemented. Everything under
"Open questions" needs a decision before implementation starts.

**Revision history:** this draft has gone through two source-grounded
code review passes so far. Each found concrete implementation gaps in
the prior draft — inline "(v0.2 fix)" / "(v0.3 fix)" markers throughout
mark exactly what changed and why, so the reasoning stays visible
rather than silently overwritten.

- **v0.2** (§3.1, §3.2, §4.3, §5, §6, §7, §8): the `is_distributed()`
  gate that would silently skip `register_memory` in this mode (§5);
  the missing mechanism for getting a `memfd` out of `HostBufferHandle`
  and into `StandaloneProcessClient::RegisterMemory` (§3.2, §5); an
  internal inconsistency between "same UDS" and "second UDS" for fd
  handoff (§3.2, §8); an incompatible TCP-fallback claim for an
  `SCM_RIGHTS`-based data plane (§3.1, §6, §8); an overreach in
  claiming the server's own `DRAMTier` shared-memory backing needs no
  new code (§3.2, §7); plus a missing `CopyPipeline::Drain()` for
  clean shutdown (§4.3).
- **v0.3** (§3.1, §4.1, §5, §6, §10): the still-unclosed activation
  path for `cfg.standalone_process` — nothing in v0.2 specified who
  actually sets it from Python, the same way `umbp_store.py` already
  does for `cfg.distributed` (§5, §10); a leftover default-path
  inconsistency between `<node_id>.sock` and `<node_id>.grpc.sock`
  plus an unspecified fd-socket derivation rule (§3.1, §4.1, §6); an
  fd-ownership contradiction between the registry design and
  "`DeregisterMemory` closes the worker's fd" (§3.2); a missing `Ping`
  RPC that §4.1's readiness probe depends on, plus explicit
  client-local-vs-RPC method classification (§3.1); and small
  corrections (a stale `batch_resolder_codec.h` typo, a table cell
  claiming the server "owns the shared-memory DRAM region" that
  contradicted v0.2's own fix).
- **v0.4** (§4.2, §5, §6): `UMBPStandaloneProcessConfig`'s
  field/parsing ownership was still ambiguous — §6's table described
  `fd_socket`/`transport`/`auto_start` with routing language that
  conflicted with both the config struct definition (only
  `address`/`auto_start`/`startup_timeout_ms`) and the §5 Python
  snippet (which only ever set `address`). Resolved by picking one of
  the two existing, mutually-inconsistent precedents in this codebase
  (`cfg.distributed`: all fields Python-set, vs.
  `spdk_proxy_auto_start`: read directly by C++
  `FromEnvironment()`) and applying it consistently: `address`,
  `auto_start`, and `startup_timeout_ms` are now all Python-set
  (mirroring `distributed`), while `fd_socket` and `transport` are
  removed from the config surface entirely — `fd_socket` is always
  mechanically derived from `address` (no knob needed), and
  `transport` has no v0.1 config surface since `tcp_staging` isn't
  built yet (§8 item 6b).
- **v0.5** (§5, §10): two blockers found by tracing SGLang's actual
  construction order and `register_mem_pool_host`'s actual error
  handling, both confirmed and resolved with the team rather than
  picked unilaterally, since both are scope/policy calls, not
  implementation details: (1) `extra_config["standalone_address"]`
  cannot reliably activate standalone-process mode — verified, the
  host memory pool is constructed and allocates its buffer in
  `HiRadixCache.__init__` (`hiradix_cache.py:91,106`,
  `pool_host/base.py:97,139`) *before* `extra_config` is parsed and
  handed to `UMBPStore` (`hiradix_cache.py:175,182`), so
  `umbp_host_allocator.py` can only ever see environment variables at
  the point it must decide a buffer's backing. **Decision: v0.1
  activation is `UMBP_STANDALONE_ADDRESS`-only;
  `extra_config["standalone_address"]` without the env var raises.**
  (2) `register_mem_pool_host`'s existing exception/`False` handling
  (`umbp_store.py:862-929`) silently downgrades to a warning — correct
  for `distributed`, which has a real staging-buffer fallback, but
  wrong for standalone-process mode, which has none in v0.1: a
  swallowed registration failure there means every later `Put`/`Get`
  fails with no correlated error. **Decision: under
  `get_deployment_mode() == StandaloneProcess`, all three cases
  (`disable_zero_copy_register` set, `register_memory` exception,
  `register_memory() == False`) raise instead of warn;
  `distributed`'s behavior is unchanged.**

**Primary references (source-verified, see §9):**
- UMBP's own SPDK-proxy daemon (`storage/spdk/proxy/`,
  `local/tiers/local_storage_manager.cpp:SpawnProxyDaemon/EnsureProxyDaemon`)
  — the only existing "library spawns a separate process, backed by a
  shared-memory ring protocol" precedent in this codebase.
- **Mooncake's Real Client / DummyClient split**
  (`mooncake-store/src/real_client.cpp`, `real_client_main.cpp`,
  `shm_helper.cpp`, `dummy_client.cpp`) — explicitly the primary model
  for this design per team guidance ("Mooncake 是我们的老师").
- LMCache's `lmcache/v1/multiprocess/` — reviewed and **not** used as
  the primary template (its zero-copy path is CUDA-IPC/GPU-only and
  has no host-memory equivalent; see §8.4).

---

## 1. Why not just reuse the existing `distributed/` mode?

UMBP already runs multi-process today (`umbp_master` + peers over
gRPC + RDMA). That mode solves a different problem: **cross-node**
sharing with RDMA as the bulk-transfer fabric, master-mediated routing,
and lease/eviction protocols built around remote peers that may be on
a different machine. Standalone-process mode is strictly **same-host**:
its entire job is to let N worker processes that already sit on one
NUMA machine share one DRAM tier without RDMA, without a master, and
without inventing new eviction protocols. Reusing `distributed/`
wholesale means dragging in a master, RDMA registration, and the
peer/lease machinery for a problem that doesn't need any of it — this
design intentionally sits *underneath* the existing `local/`
(`StandaloneClient`) code, not beside `distributed/`.

Concretely: **the standalone-process server is `StandaloneClient`'s own
`LocalStorageManager`/`LocalBlockIndex`/`CopyPipeline`, unchanged, moved
into its own process and fronted by an RPC+shm shell.** No new tier
logic, no new eviction policy, no new durability logic — those are
100% reused from `local/`.

---

## 2. Component overview

Two new roles, named after Mooncake's split for consistency:

| UMBP name | Mooncake analog | Lives in | Role |
|---|---|---|---|
| `umbp_standalone_server` | `mooncake_client` (Real Client) | its own process | Owns `LocalStorageManager` (DRAM/SSD tiers, private anon/hugetlb memory by default — see §3.2/§7, **not** a shared DRAM region), `LocalBlockIndex`, `CopyPipeline`. Exposes `IUMBPClient` semantics over gRPC. `mmap`s each worker's registered host-buffer segment (via fd handoff) for zero-copy Put/Get access. |
| `StandaloneProcessClient` | `DummyClient` | inside the SGLang/vLLM worker process, behind `IUMBPClient` | A new `IUMBPClient` implementation. Forwards every call to the server over gRPC. Registers the worker's host KV buffer with the server once via fd-passing, then references it by offset on every hot-path call. |

```
 ┌───────────────────────────┐        ┌───────────────────────────┐
 │ SGLang worker process A    │        │ SGLang worker process B    │
 │                             │        │                             │
 │  UMBPStore (Python)         │        │  UMBPStore (Python)         │
 │       │ pybind11            │        │       │ pybind11            │
 │       ▼                     │        │       ▼                     │
 │  StandaloneProcessClient    │        │  StandaloneProcessClient    │
 │   (IUMBPClient impl)        │        │   (IUMBPClient impl)        │
 │    │ gRPC            │ raw UDS  │        │    │ gRPC            │ raw UDS │
 │    │ (<addr>.grpc.sock)│(<addr>.fd.sock)│        │    │ (<addr>.grpc.sock)│(<addr>.fd.sock)│
 └────┼─────────────────┼──────--┘        └────┼─────────────────┼──────--┘
      │                 │                       │                 │
      │                 │  one-time fd handoff  │                 │
      │                 │  (SCM_RIGHTS), at     │                 │
      │                 │  RegisterMemory() time │                 │
      ▼                 ▼                       ▼                 ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                     umbp_standalone_server                      │
 │                                                                   │
 │  UMBPStandaloneService (gRPC, control plane, own UDS socket)     │
 │    Put/Get/BatchPut/BatchGet/Exists/... — same names as          │
 │    IUMBPClient, arguments are (key, client_id, shm_offset, size) │
 │    instead of raw pointers                                       │
 │                                                                   │
 │  Fd-handoff listener (raw AF_UNIX, SEPARATE socket path from     │
 │    the gRPC UDS — see §3.2/§8.3; gRPC cannot carry an fd)        │
 │                                                                   │
 │  LocalStorageManager (REUSED, unmodified)                         │
 │    DRAMTier (own private anon/hugetlb memory, NOT shared — see   │
 │    §3.2 note) ── SsdTier                                          │
 │  LocalBlockIndex (REUSED, unmodified)                             │
 │  CopyPipeline (REUSED, unmodified; gains Drain(), see §4.3)       │
 │                                                                   │
 │  Per-worker shm registry: client_id → {fd, base, size, offset}   │
 └─────────────────────────────────────────────────────────────────┘
```

Two logical channels, two socket paths, deliberately — see §3.2/§8.3
for why gRPC cannot double as the fd-handoff channel.

---

## 3. IPC mechanism

### 3.1 Control plane: gRPC over a Unix domain socket

**Decision: reuse UMBP's existing gRPC stack, not a new RPC framework,
and not Mooncake's `coro_rpc`.**

Mooncake uses `ylt::coro_rpc` for its control plane. We explicitly do
**not** copy that choice: UMBP's `umbp_common` already depends on
gRPC/protobuf for the `distributed/` master/peer path
(`umbp/CMakeLists.txt:305-328,347-349`), and `pybind_umbp.cpp` already
has a proven, working convention for calling blocking C++ RPC methods
from Python with the GIL released
(`py::call_guard<py::gil_scoped_release>()`,
`pybind_umbp.cpp:248-279`). Adopting `coro_rpc` would require building
a new coroutine-to-pybind11 bridge with **no existing precedent in
either UMBP or the reports on Mooncake's own Python integration** —
pure added risk for zero benefit, since gRPC already does everything
Mooncake uses `coro_rpc` for (typed request/response, async server
loop, TCP or UDS transport).

Design:

- New proto service `UMBPStandalone` (new file,
  `distributed/proto/umbp_standalone.proto`). **(v0.3 fix — this list
  is now explicit about what is an RPC, what is proto-but-not-`IUMBPClient`,
  and what stays entirely client-local, to avoid an implementer
  guessing.)**
  - RPCs mirroring `IUMBPClient` 1:1: `Put`, `Get`, `BatchPut`,
    `BatchPutWithDepth`, `BatchGet`, `Exists`, `BatchExists`,
    `BatchExistsConsecutive`, `Clear`, `Flush`, `RegisterMemory`,
    `DeregisterMemory`, and the full external-KV set —
    `ReportExternalKvBlocks`, `RevokeExternalKvBlocks`,
    `RevokeAllExternalKvBlocksAtTier`, `MatchExternalKv`, **and
    `GetExternalKvHitCounts`** (`umbp_client.h:166-170`; named
    explicitly here since it's easy to drop — it has a default no-op
    body in the interface, unlike its siblings, which makes it look
    optional when it isn't for a real implementation).
  - `Ping(Empty) -> PingResponse{ready: bool}`: new RPC, **no**
    `IUMBPClient` analog, exists solely for the §4.1 readiness probe.
  - **Not RPCs, stay entirely client-local** in
    `StandaloneProcessClient`: `Close()` (tears down the local gRPC
    channel + fd-handoff socket + registry state; no server-side call
    needed beyond whatever `DeregisterMemory` already covers) and the
    new `get_deployment_mode()` accessor from §5 (pure local constant,
    never needs a round trip).
  Reuse the existing struct-of-arrays batch codec approach from
  `batch_resolve_codec.h`
  (`distributed/peer/batch_resolve_codec.h:24-28`) for `BatchPut`/
  `BatchGet` so per-key protobuf submessage overhead doesn't reappear
  here — this is exactly the pattern UMBP already validated for the
  peer `BatchResolveKeys` RPC.
- **Default transport: gRPC over a Unix domain socket**
  (`unix:///run/umbp/standalone/<node_id>.grpc.sock`), not TCP. This is a
  deliberate deviation from Mooncake (TCP, port 50052) and LMCache
  (TCP, port 5555): both of those need to be reachable from a
  different host or container in the general case, but standalone
  mode by definition never leaves the host, so a UDS avoids picking a
  port, avoids the network stack, and gets free filesystem-permission
  based access control (§8.5). gRPC's UDS support
  (`unix:` target scheme) is mature and already linked in via the same
  gRPC dependency `umbp_common` uses today.
  **(v0.2 fix) TCP is not a drop-in fallback for this design and must
  not be presented as one.** §3.2's data plane depends on `SCM_RIGHTS`
  fd passing, which is an `AF_UNIX`-only mechanism — there is no
  TCP-socket equivalent for handing a `memfd` to another process, and
  because the design deliberately uses an anonymous `memfd_create`
  (not a named `shm_open` segment, precisely to avoid `/dev/shm` name
  collisions, see §3.2), there is also no name the TCP-connected side
  could use to reopen the segment out-of-band. Concretely, for v0.1:
  the fd-handoff channel (§3.2, §8.3) **must** be a UDS reachable by
  both processes (i.e. they must share the same mount/PID namespace
  for `/dev/shm`-adjacent sockets, or a bind-mounted socket directory
  across containers). If a deployment genuinely cannot provide that
  (fully separate network namespaces with no shared filesystem), it
  cannot use the zero-copy data plane at all; the only option is a
  non-zero-copy fallback where `Put`/`Get` payload bytes travel inside
  the gRPC message itself (i.e. the same "no RDMA, no shm" degraded
  path `distributed/` already has via `staging_buffer_size`-capped
  copies when `UMBP_DISABLE_ZERO_COPY_REGISTER=1`), not a variant of
  the shm data plane over TCP. The gRPC *control* channel could in
  principle still run over TCP in that scenario, but the two are no
  longer both UDS by construction — this is a materially different,
  slower deployment mode conceptually named `tcp_staging` to
  distinguish it from the default `uds` path. **(v0.4 fix)** it has no
  config/env surface in v0.1 at all, precisely because it isn't built
  in v0.1 (§8 item 6b, §10 step 13) — see §6 for why a `transport`
  knob is deliberately not added to `UMBPStandaloneProcessConfig`
  ahead of the feature it would select.
- Argument shape: control messages never carry KV bytes. `Put`/`Get`
  take `(key, client_id, shm_offset, size)`; the *raw pointer* that
  `IUMBPClient::Put(key, uintptr_t src, size_t size)` takes today is
  translated to `(client_id, shm_offset)` **inside**
  `StandaloneProcessClient` before the RPC is issued (see §3.2) — this
  is exactly Mooncake's `map_dummy_buffer_range_to_real` /
  `shm_addr_offset` translation
  (`real_client.cpp:1510-1583,3296-3318`), just carried over gRPC
  instead of coro_rpc.

### 3.2 Data plane: shared memory, modeled directly on Mooncake's `ShmHelper`

This is the part of the design that should track Mooncake most
closely, because UMBP and Mooncake independently converged on the same
shape (control message carries a handle, bytes live in shared memory)
— UMBP's own SPDK-proxy already does this too
(`spdk_proxy_protocol.h:RingSlot.data_offset/data_size`). The novelty
here is *who allocates the shared segment*.

**(v0.2 fix — this paragraph overreached in v0.1.)** `DRAMTier` does
already support POSIX shared memory — `use_shared_memory`/`shm_name`
in `UMBPDramConfig` (`common/config.h:57-58`) are wired through
`LocalStorageManager` (`local_storage_manager.cpp:443-444`) into
`DRAMTier`'s constructor, which does `shm_open`+`ftruncate`+`mmap`+
`shm_unlink` (`dram_tier.cpp:104-175`). But two things in the v0.1
draft were wrong about what this buys us:

- The server's own DRAM cache **does not need to be visible to any
  other process** — it is read/written exclusively by
  `umbp_standalone_server` itself; no worker ever `mmap`s it directly
  (workers only ever see their own registered host KV buffer, per
  below). So `umbp_standalone_server` should default to the same
  `kAnonymous`/`kAnonymousHugetlb` private backing
  `StandaloneClient` already uses today — **not**
  `use_shared_memory=true`. There is no reason to pay for a named
  POSIX shm segment (and its cleanup/lifecycle concerns) for memory
  nothing else needs to touch.
- Even where a future variant *did* want `use_shared_memory=true` (for
  example, a warm-restart scenario that reattaches to a still-live
  segment after the server process is replaced), the existing
  `shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666)` call
  (`dram_tier.cpp:114`) creates the segment world-read-write. That is
  a pre-existing gap in `DRAMTier` itself (not introduced by this
  design), and this design must not casually inherit it by pointing
  more traffic at that code path while claiming "zero new C++ code" —
  tightening `dram_tier.cpp`'s permissions is a separate, pre-existing
  fix, out of scope here unless/until a future revision actually
  chooses to share the server's DRAM tier across processes.

The actual sharing requirement in this design is one-directional and
much narrower than "the whole DRAM tier": only the **worker's host KV
buffer** needs to become visible to the server process. That is the
problem the rest of this section solves.

The problem this design actually has to solve is the **other** side:
today, `Put(key, src, size)`/`Get(key, dst, size)` take a pointer that
is only valid in the *caller's* address space
(`umbp_client.h:47-171`) — this is fine when `StandaloneClient` runs
in-process, and it is fine in `distributed/` mode because RDMA reads
directly from the registered remote memory without needing a shared
mapping. It is **not** fine when the server is a separate process on
the same host with no RDMA involved: the server cannot dereference a
pointer that only exists in the worker's page tables.

Adopt Mooncake's answer exactly:

1. **The worker process (not the server) owns the buffer**, exactly
   like Mooncake's `DummyClient` owns its local buffer via
   `setup_dummy()` → `shm_helper_->allocate(...)`
   (`dummy_client.cpp:449-451`). Concretely: `UMBPHostMemAllocator`
   (`host_mem_allocator.h/.cpp`, already used by
   `umbp_host_allocator.py` to back SGLang's host KV pool) gains a new
   backing kind, `kAnonymousShm`, using `memfd_create` +
   `ftruncate` + `mmap(MAP_SHARED)` — the same primitives Mooncake's
   `ShmHelper::allocate()` uses (`shm_helper.cpp:102-123`), chosen over
   UMBP's own named-`shm_open` DRAM-tier path specifically to avoid
   `/dev/shm` name collisions across many SGLang instances on one
   host (Mooncake's rationale for `memfd_create` over named
   `shm_open` — anonymous, refcounted purely by open fds, nothing to
   leak into `/dev/shm` if a process is killed uncleanly).
2. **One-time registration handshake**, triggered by the existing
   `RegisterMemory(ptr, size)` call
   (`umbp_store.py:912` → `register_mem_pool_host`).

   **(v0.2 fix — this step was underspecified in v0.1.)**
   `IUMBPClient::RegisterMemory(uintptr_t ptr, size_t size)`
   (`umbp_client.h:119`) and the pybind binding
   `register_memory(ptr, size)` only carry a pointer and a size — there
   is no fd parameter anywhere on this path, and `HostBufferHandle`
   (`host_mem_allocator.h:41`) has no fd field either. The Python side
   never sees a file descriptor (`kv_buffer.data_ptr()` is just an
   integer, `umbp_store.py:896`), so it has nothing to pass through
   pybind even if we wanted to add an argument. The fd has to be
   recovered entirely on the C++ side of the process boundary that
   already exists (`StandaloneProcessClient` and the allocator both
   live in the same worker process), not threaded through Python:

   - `HostMemAllocator::Alloc` gains an internal, process-local
     registry (a mutex-guarded `std::map<void* /*base*/, AllocRecord>`
     where `AllocRecord` holds `{fd, size}`), populated only for
     `kAnonymousShm` allocations at alloc time and erased on `Free`.
   - `StandaloneProcessClient::RegisterMemory(ptr, size)` looks up
     `ptr` in that registry (a range lookup, since `ptr` may be
     `kv_buffer.data_ptr()` — the tensor's base address, which for the
     host KV pool is normally exactly the allocation's base, but a
     range/floor lookup is safer than requiring exact match) and, on a
     hit, sends the recovered `fd` to the server via `SCM_RIGHTS`
     ancillary data over the **separate raw-UDS fd-handoff socket**
     (`<addr>.fd.sock`, distinct from the gRPC UDS — see §8.3),
     mirroring `ipc_send_fd`/`ipc_recv_fd` (`shm_helper.cpp:181-230`)
     and Mooncake's `IPC_SHM_REGISTER` handler
     (`real_client.cpp:4046-4058`). On a miss (the pointer isn't
     backed by a `kAnonymousShm` allocation — e.g. standalone-process
     mode was configured but the host allocator wasn't switched to the
     shm backing), `RegisterMemory` must fail loudly rather than
     silently falling back to a broken pointer-based `Put`/`Get`.
   - The server `mmap`s the received fd read-write into its own
     address space and stores `client_id → {base, size}` in a registry
     keyed by `client_id` (mirrors `shm_contexts_`,
     `real_client.cpp:4074-4082`).
3. **Every subsequent Put/Get/BatchPut/BatchGet RPC carries only
   `(client_id, shm_offset, size)`** — `StandaloneProcessClient`
   computes `shm_offset = ptr - registered_base` locally before
   issuing the RPC; the server computes `real_addr = base + shm_offset`
   and reads/writes there directly
   (mirrors `map_dummy_buffer_range_to_real`,
   `real_client.cpp:1510-1583,3317`). No KV bytes ever cross the gRPC
   channel; the gRPC message is a handful of integers per key.
4. `DeregisterMemory` is the inverse — **(v0.3 fix: fd ownership
   clarified, since the v0.2 wording here conflicted with the §3.2
   step 2 registry design).** The allocator's ptr→fd registry (§3.2
   step 2), not `RegisterMemory`/`DeregisterMemory`, owns the fd:
   `HostMemAllocator` opened it at `Alloc` time and is the only thing
   that closes it, at `Free` time. `RegisterMemory` merely *borrows*
   that fd to send over `SCM_RIGHTS`; sending a fd over `SCM_RIGHTS`
   duplicates it into the receiving process, so the sender's copy is
   unaffected by anything the server does with its own copy.
   `DeregisterMemory`, therefore, only (a) tells the server to
   `munmap` its copy and drop the `client_id` registry entry, and (b)
   clears `StandaloneProcessClient`'s own bookkeeping of "this range
   is registered" — it must **not** close the allocator-owned fd.
   Closing happens exactly once, in `HostMemAllocator::Free`. Getting
   this wrong either double-closes an fd number the kernel may have
   already recycled for something else, or leaves a re-`RegisterMemory`
   call after a `DeregisterMemory` trying to send an already-closed fd
   — both are the kind of bug that only shows up under a
   register/deregister/re-register cycle, so the unit test in §10
   step 11 must specifically exercise that cycle, not just a single
   register→use→deregister pass.

Net effect: the hot path becomes *exactly* the same shape as UMBP's
existing RDMA-registered `distributed/` path (`RegisterMemory` once,
then reference-by-handle on every call) — this is not a new interface
concept for UMBP, just a new transport underneath a pattern
`IUMBPClient` already exposes.

### 3.3 Why not a busy-polled ring buffer (SPDK-proxy's own approach)?

UMBP's own precedent for "separate process, shared-memory data plane"
(`spdk_proxy_protocol.h`) is a fixed-slot ring buffer with busy-polling
client-side (`spdk_proxy_tier.cpp:240,354`, `sleep_for(ProxyPollInterval())`)
and no RPC framework at all — every control operation (attach, admin
shutdown, stats) is also squeezed into the same 256-slot ring. This
works well for SPDK-proxy because it's low cardinality (few tenants,
few request types) and latency-critical at the microsecond level for
raw block I/O. It is a poor fit for the standalone-process control
plane because:

- it has no natural way to express variable-shape requests (batch
  registration, external-KV admission with variable-length hash
  lists) without hand-rolling a second serialization format inside
  the slot payload — gRPC already solves this,
- it offers no async multiplexing for slow control operations
  without blocking the ring for the next request,
- neither Mooncake nor LMCache use a pure ring buffer for control —
  both back it with a real RPC/messaging library and reserve raw
  shared memory strictly for bulk bytes, which is the strategy this
  design also adopts.

The *data plane* still ends up looking like SPDK-proxy's idea (handle
+ offset/size, bytes in shm) — we're only rejecting the ring buffer
for the **control** plane, not the "shared memory for bytes" idea
itself.

---

## 4. Lifecycle management

This is the one dimension where UMBP's own precedent (SPDK-proxy:
library-spawned, idle-self-exit) and the Mooncake/LMCache precedent
(externally launched, long-lived, no self-exit) genuinely disagree,
and a specific choice has to be made for this codebase.

**Decision: default to Mooncake's model (externally launched,
long-lived), with UMBP's SPDK-proxy auto-start pattern available as an
opt-in convenience, and idle-self-exit *disabled by default*.**

Rationale: unlike the SPDK-proxy (a stateless passthrough to a shared
NVMe device — nothing is lost if it exits and respawns), the standalone
server **holds the DRAM cache tier's actual data**. Self-exiting on
idle silently discards the cache, defeating the purpose of the whole
feature. Neither Mooncake's Real Client nor LMCache's multiprocess
server self-terminate in the reports we reviewed — that absence is
itself evidence, not just a gap.

### 4.1 Primary path: externally launched

- Launched the same way `run_umbp_single_node_hicache.sh` already
  launches `umbp_master` today (background process, log redirected,
  `trap cleanup EXIT`) — no new orchestration concept, just a new
  binary target.
- Discovery: `UMBP_STANDALONE_ADDRESS` (default
  `unix:///run/umbp/standalone/<node_id>.grpc.sock`), following the exact
  naming convention of `UMBP_MASTER_ADDRESS`
  (`runtime-env-vars.md:146-152`) for consistency.
- **fd-socket path derivation, made explicit (v0.3), confirmed to have
  no separate config/env knob (v0.4 — see §6).** The fd-handoff socket
  path is derived from `cfg.standalone_process.address` mechanically,
  by both processes independently: strip the `unix://` scheme prefix
  to get a filesystem path (raw `AF_UNIX` `connect()`/`bind()` need a
  path, not a URI), then replace a trailing `.grpc.sock` with
  `.fd.sock`; if the configured address doesn't end in `.grpc.sock`
  (a custom path), append `.fd.sock` to it instead of doing a
  substring replace. E.g.
  `unix:///run/umbp/standalone/node0.grpc.sock` →
  `/run/umbp/standalone/node0.fd.sock`. This derivation must live in
  one shared place (`StandaloneProcessClient`'s ctor and
  `umbp_standalone_server`'s startup both call it) rather than being
  reimplemented on each side, to guarantee they agree — and precisely
  because it's a pure function of `address`, it deliberately has no
  independent `UMBP_STANDALONE_FD_SOCKET` env var or config field to
  keep in sync (§6).
- **(v0.3 fix) `Ping` must be added to the `UMBPStandalone` proto
  service.** The method list in §3.1 mirrors `IUMBPClient` 1:1 and
  `IUMBPClient` has no `Ping` method, so a reader implementing §3.1's
  proto as written would have nothing for this readiness probe to
  call. `Ping(Empty) -> PingResponse{ready: bool}` is a new,
  UMBP-standalone-specific addition to the proto with no `IUMBPClient`
  analog — call it out explicitly in §3.1's method list, not just
  here in §4.1.
- Readiness probe: connect to the UDS and call `Ping`
  (equivalent to `ProxyShmRegion::ProbeExisting`'s role for
  SPDK-proxy, but over gRPC instead of a shm header flag, since there
  is no shm header to probe until a client registers memory).

### 4.2 Convenience path: auto-start (opt-in)

- **(v0.4 fix — routing corrected to match §6's decision.)** When
  `cfg.standalone_process.auto_start` is `true` (Python-set from
  `UMBP_STANDALONE_AUTO_START`/`extra["standalone_auto_start"]`, §5,
  §6 — `CreateUMBPClient` itself never reads that env var; by the time
  it runs, the bool is already sitting on the config it was handed):
  the first worker process on a host to reach `CreateUMBPClient`
  (leader election reuses the *existing*
  `UMBP_ROLE`/`LOCAL_RANK`/`OMPI_COMM_WORLD_LOCAL_RANK` rank-0 logic
  already in `UMBPConfig::FromEnvironment`
  (`config.h:410-428`) — no new election protocol) probes the UDS at
  `cfg.standalone_process.address`; if absent, it does the same
  `fork()` + `setsid()` + `execlp()` sequence
  `LocalStorageManager::SpawnProxyDaemon` already uses
  (`local_storage_manager.cpp:319-372`), passing the spawned server's
  own config down via env vars (this is a different hop than the one
  above — see `UMBP_STANDALONE_BIN` in §6, which *is* read directly by
  this C++ auto-start code, matching its deployment-only status), and
  waits for readiness bounded by
  `cfg.standalone_process.startup_timeout_ms` (also Python-set, §6 —
  mirrors `spdk_proxy_startup_timeout_ms`'s *role* only; unlike its
  SPDK-proxy analog it is not itself read by
  `UMBPConfig::FromEnvironment`, per §6's Precedent A decision). A
  bootstrap lock (reuse the existing `ScopedBootstrapLock` pattern,
  `local_storage_manager.cpp:274-283`) prevents a thundering herd of
  workers all trying to spawn the server simultaneously.
- **Not** the default, because auto-start from inside a worker process
  means the worker's `fork()`'d child inherits CUDA/HIP context, open
  gRPC channels, and GPU device state at the moment of fork — exactly
  the fork-safety hazard `SpawnProxyDaemon` already has to be careful
  about (`local_storage_manager.cpp:328` calls `setsid()` immediately,
  and the child does nothing but `execlp()` before replacing itself —
  **this rule must not be relaxed** for the standalone-server spawn
  path either; no CUDA/HIP/gRPC call may happen between `fork()` and
  `execlp()`).

### 4.3 Shutdown

- `SIGTERM`/`SIGINT` → drain in-flight RPCs (bounded deadline,
  mirroring `UMBP_GRPC_SHUTDOWN_DEADLINE_SEC` already used by
  `MasterServer::Shutdown()`, `master_server.cpp:809-816`) → flush
  `CopyPipeline` (best-effort persist DRAM-tier dirty pages to SSD
  before exit, if the SSD tier is enabled and
  `force_ssd_copy_on_write` is set) → `munmap` all registered client
  shm segments → exit.

  **(v0.2 fix — no API exists today for the "flush CopyPipeline"
  step above.)** `CopyPipeline` only drains its queue in its
  destructor (`copy_pipeline.cpp:40-52`, `stop_copy_worker_` +
  worker-thread join), and `StandaloneClient::Flush()`
  (`standalone_client.cpp:300`) only calls `storage_.Flush()` — it
  never waits for the async SSD-copy queue to empty. A clean shutdown
  that wants "no dirty DRAM pages lost" needs the queue drained
  *before* the object is destroyed, while the server can still respond
  to health checks. This requires a small, genuinely new piece of
  work, not a reuse of an existing hook: add
  `CopyPipeline::Drain(std::chrono::milliseconds timeout)` (blocks
  until `copy_queue_` is empty and in-flight copies complete, or the
  timeout elapses) and have both `umbp_standalone_server`'s shutdown
  path and the `Flush` RPC call it. See §10 implementation order —
  this is listed as its own task, not folded into "reuse
  `LocalStorageManager` unmodified" from §1, because it genuinely
  isn't unmodified once this exists.
- Client-side: if the RPC channel drops (server crashed or was
  killed), `StandaloneProcessClient` must **not** silently return
  stale success — every in-flight call fails, and the recommended
  worker-side behavior is to treat this the same as "cache miss /
  cache unavailable" rather than crashing the inference process
  (SGLang's `HiCacheStorage` abstraction already tolerates backend
  failures returning `False`/`None`; this needs to be threaded through
  explicitly — see Open Question in §8.2, since neither Mooncake nor
  LMCache's reviewed code shows a supervised-restart or
  reconnect-on-crash path to copy).

### 4.4 Multi-tenant workers on one host

- Multiple SGLang worker processes (e.g. one per TP/DP rank) may
  attach to the same standalone server. Each gets a `client_id`
  (UUID, assigned at first `RegisterMemory` call) and its own shm
  registration entry — mirrors Mooncake's per-client `shm_contexts_`
  map and UMBP's own SPDK-proxy per-tenant quota concept
  (`TenantInfo.quota_bytes`, `spdk_proxy_protocol.h:227-240`).
  Capacity accounting inside `LocalStorageManager`/`DRAMTier` is
  already global (not per-caller) today — **this is an open question**,
  not solved by this design (see §8.3): without per-tenant quotas, one
  worker can starve another's share of the shared DRAM tier.

---

## 5. Python / pybind interface adaptation

**(v0.2 fix — the v0.1 headline claim "zero call-site changes in
`umbp_store.py`" is false and is retracted here.)** One call site does
need to change, and the reason is a real bug the v0.1 draft
introduced by combining two of its own recommendations
inconsistently: §5 (v0.1) said `is_distributed()` should keep meaning
"true cross-node distributed" and *not* be overloaded for
standalone-process mode, while `register_mem_pool_host`
(`umbp_store.py:870-881`) gates the entire `register_memory` call on
`if not is_distributed: return` (`umbp_store.py:880`). Taken together,
a `StandaloneProcessClient` — which correctly reports
`is_distributed() == False` per the v0.1 recommendation — would never
have its host KV buffer registered, and therefore never trigger the
fd-handoff handshake in §3.2 at all. The whole standalone-process data
plane silently never activates. This is not a minor wording issue; it
is a functional gap that must be fixed as part of this design, not
left as an implementation detail:

- Add `get_deployment_mode()` (enum: `Local`/`StandaloneProcess`/
  `Distributed`) to `IUMBPClient`/pybind, as already proposed below.
- Change the gate at `umbp_store.py:880` from `if not is_distributed:
  return` to also allow `StandaloneProcess`, e.g.
  `if deployment_mode not in (Distributed, StandaloneProcess): return`.
  This *is* a call-site change in `umbp_store.py`, and needs to be
  listed explicitly in the implementation order (§10), not implied.

With that correction, the remaining adaptation is small:

- `CreateUMBPClient(const UMBPConfig&)` (`umbp_client_factory.cpp:28-33`)
  gains a third branch: `config.standalone_process.has_value()` (new
  optional field on `UMBPConfig`, alongside the existing `distributed`
  optional) → construct `StandaloneProcessClient`. Precedence, if both
  happen to be set, should be `distributed` first (cross-node case is
  strictly more general) — but in practice these are mutually
  exclusive deployment choices and `UMBPConfig::Validate()` should
  reject setting both.
- **(v0.3 fix, superseded by v0.5 below — kept for history.)** Nothing
  in v0.2 said *who sets* `cfg.standalone_process`; v0.3 filled that in
  by mirroring how `umbp_store.py` populates `cfg.distributed` from
  either `extra_config["master_address"]` or `UMBP_MASTER_ADDRESS`
  (`umbp_store.py:458-465`).
- **(v0.5 fix — the v0.3 answer above is wrong for this specific field,
  for a reason that doesn't apply to `master_address`.)** Mirroring
  `master_address`'s two-source pattern (`extra_config` *or* env)
  assumes both sources are read at the same point in the SGLang
  startup sequence. They are not, and the difference matters here
  specifically because of a new constraint standalone-process mode
  introduces that distributed mode never had: **verified**, in
  `HiRadixCache.__init__`, the host memory pool
  (`MHATokenToKVPoolHost`, via `pool_host/base.py`) is constructed with
  `allocator_type=server_args.hicache_storage_backend`
  (`hiradix_cache.py:91,106`) and immediately allocates `kv_buffer` in
  its own constructor (`pool_host/base.py:97` `get_allocator_from_storage`,
  `pool_host/base.py:139` `self.kv_buffer = self.init_kv_buffer()`) —
  this happens *before* `storage_backend_extra_config` is even parsed
  and handed to `CacheController`/`UMBPStore` (`hiradix_cache.py:175,182`,
  well after the pool already exists). `UMBPHostTensorAllocator` (used
  by that pool) therefore can only see **process environment
  variables** at the moment it decides a buffer's backing — never
  `extra_config`, which doesn't exist yet at that point in the process.
  This asymmetry is new to standalone-process mode: `distributed`
  mode's `RegisterMemory` pins/registers *whatever pointer it's given*
  for RDMA regardless of how it was allocated, so `master_address`
  arriving late via `extra_config` was always fine — nothing upstream
  needed to know in advance. Standalone-process mode's
  `RegisterMemory` instead needs the buffer to already be a
  `kAnonymousShm` allocation *before* `register_mem_pool_host` ever
  runs, and that decision is made by `umbp_host_allocator.py` strictly
  earlier than `UMBPStore.__init__` gets to see `extra_config` at all.
  **Decision (confirmed with the team): v0.1 supports activation via
  `UMBP_STANDALONE_ADDRESS` only.** `extra_config["standalone_address"]`
  is not a valid activation path in v0.1 and `UMBPStore.__init__` must
  reject it outright with an actionable error rather than silently
  accepting a config that the host allocator already missed:
  ```python
  if extra.get("standalone_address") and not _optional_env_str(
      "UMBP_STANDALONE_ADDRESS"
  ):
      raise ValueError(
          "standalone_address in hicache-storage-backend-extra-config is "
          "not supported: the host memory pool allocator decides its "
          "backing (Anonymous vs. AnonymousShm) before extra_config is "
          "parsed, so extra_config alone cannot activate standalone-process "
          "mode reliably. Set the UMBP_STANDALONE_ADDRESS environment "
          "variable instead (see design-standalone-process-mode.md §5)."
      )
  standalone_address = _optional_env_str("UMBP_STANDALONE_ADDRESS")
  if standalone_address and master_address:
      raise ValueError(
          "master_address and UMBP_STANDALONE_ADDRESS are mutually "
          "exclusive (distributed vs. standalone-process mode)"
      )
  if standalone_address and UMBPStandaloneProcessConfig is not None:
      standalone_cfg = UMBPStandaloneProcessConfig()
      standalone_cfg.address = str(standalone_address)
      # auto_start/startup_timeout_ms may still come from extra_config --
      # unlike `address`, they carry no allocator-timing constraint,
      # since they only affect what CreateUMBPClient does once cfg is
      # already fully built. Only `address` (the activation switch) is
      # env-only; these two may keep the extra/env fallback pattern.
      standalone_cfg.auto_start = _strict_bool(
          extra.get("standalone_auto_start", _optional_env_str("UMBP_STANDALONE_AUTO_START")),
          "standalone_auto_start",
      ) if extra.get("standalone_auto_start") or _optional_env_str("UMBP_STANDALONE_AUTO_START") else False
      standalone_cfg.startup_timeout_ms = int(
          extra.get(
              "standalone_startup_timeout_ms",
              _optional_env_str("UMBP_STANDALONE_STARTUP_TIMEOUT_MS") or 30000,
          )
      )
      cfg.standalone_process = standalone_cfg
  ```
  This must run, and `cfg.standalone_process` must be populated,
  **before** `UMBPClient(cfg)` is constructed (`umbp_store.py:790`) —
  listed as its own step in §10 rather than folded silently into "wire
  into `umbp_client_factory.cpp`", since that step is C++-only and
  does not by itself make Python ever populate the field. The rejected
  alternative — teaching `hiradix_cache.py`/`kv_cache_builder.py`/
  `memory_pool_host.py` to see `extra_config` before constructing the
  host pool, so `extra_config` could activate standalone-process mode
  too — was considered and explicitly deferred: it touches SGLang's
  core pool-construction sequence outside this feature's natural
  boundary, for a convenience (config-file-only activation) that
  `UMBP_STANDALONE_ADDRESS` already covers. Revisit only if an env-var-only
  activation switch turns out to be a real deployment blocker in
  practice.
- **(v0.2 fix)** `pybind_umbp.cpp` is **not** change-free the way v0.1
  claimed. The `IUMBPClient` method binding table itself
  (`put_from_ptr`/`get_into_ptr`/etc., `pybind_umbp.cpp:248-279`)
  indeed needs no changes, since it dispatches through the vtable and
  `StandaloneProcessClient` implements the same interface — that part
  of the v0.1 claim was correct. But at least two *other* bindings in
  the same file are missing and block this design from working at
  all:
  - `HostBufferBacking`/`UMBPHostBufferBacking`
    (`pybind_umbp.cpp:42-45`) currently only exports `Anonymous` and
    `AnonymousHugetlb` — it needs a third value,
    `AnonymousShm`/`kAnonymousShm`, or `umbp_host_allocator.py` has no
    way to ask for the new backing added in §3.2.
  - `UMBPConfig`'s pybind class (`pybind_umbp.cpp:230-239`) only
    exposes `dram`/`ssd`/`eviction`/`copy_pipeline`/`role`/
    `follower_mode`/`force_ssd_copy_on_write`/`distributed` — it needs
    a new `standalone_process` property (and a new
    `UMBPStandaloneProcessConfig` pybind class, mirroring how
    `UMBPDistributedConfig` is bound at `pybind_umbp.cpp:218-228`)
    before Python can construct or detect this config at all.
  The GIL-release `call_guard` convention on the existing methods is
  still correct as-is, since `StandaloneProcessClient`'s methods block
  on a gRPC call and never call back into Python — same justification
  as the existing comment (`pybind_umbp.cpp:245-246`, "block on RDMA,
  SSD, or gRPC").
- **(v0.5 fix — must read the raw env var directly, not `UMBPConfig`.)**
  `umbp_host_allocator.py` needs one change: `UMBPHostTensorAllocator.__init__`
  reads `os.environ["UMBP_STANDALONE_ADDRESS"]` directly (same style as
  its existing `SGLANG_HICACHE_HOST_*` env reads,
  `umbp_host_allocator.py:42-47`) — **not** anything derived from
  `UMBPConfig` or `extra_config`, since neither exists yet at this
  point in the process per the v0.5 fix above. If set, request the new
  `AnonymousShm` backing from `UMBPHostMemAllocator.alloc(...)` instead
  of the default anonymous/hugetlb backing. This is a small branch in
  `UMBPHostTensorAllocator.allocate()` (`umbp_host_allocator.py:50`) —
  the returned `handle.ptr` is used identically afterward (same
  `ctypes.c_byte.from_address` + `torch.frombuffer` zero-copy wrap,
  `umbp_host_allocator.py:86-87`), gated on the new `AnonymousShm`
  pybind enum value existing (see above).
- **(v0.5 fix — the existing exception/false handling in
  `register_mem_pool_host` was written for `distributed`'s optional,
  gracefully-degradable RDMA registration and must not apply as-is to
  standalone-process mode, which has no staging/inline fallback to
  degrade to in v0.1.)** Verified: `register_mem_pool_host`
  (`umbp_store.py:862-929`) currently (a) treats
  `disable_zero_copy_register` as "log info, skip registration, stay
  on staging path" (`umbp_store.py:~886-892`), (b) wraps the
  `register_memory` call in `except Exception as exc: logger.warning(...);
  return` (`umbp_store.py:~913`), and (c) treats a `False` return as
  `logger.warning(...)` only, no raise (`umbp_store.py:~929`). All
  three are correct for `distributed` — a failed/declined RDMA
  registration there really does have a working fallback
  (`distributed.staging_buffer_size`-capped copies). For
  standalone-process mode none of these three can be a silent
  downgrade, because there is nothing to downgrade *to*: a registry
  miss means every subsequent `Put`/`Get` will fail against the
  server's offset translation, just later and with a less
  correlated-looking error. **Decision (confirmed with the team): when
  `get_deployment_mode() == StandaloneProcess`, all three cases raise
  instead of warn-and-return; `distributed`'s existing warn-and-fallback
  behavior is unchanged.** Concretely, `register_mem_pool_host` branches
  on `get_deployment_mode()` before each of the three points above:
  ```python
  is_standalone_process = (
      self.client.get_deployment_mode() == UMBPDeploymentMode.StandaloneProcess
  )
  if is_standalone_process and getattr(self, "_disable_zero_copy_register", False):
      raise RuntimeError(
          "disable_zero_copy_register is not supported in standalone-process "
          "mode: there is no staging-buffer fallback path to degrade to (v0.1)."
      )
  ...
  try:
      ok = bool(self.client.register_memory(host_ptr, host_size))
  except Exception as exc:
      if is_standalone_process:
          raise RuntimeError(
              f"register_memory failed in standalone-process mode: {exc}. "
              "This is fatal, not a degradable condition -- see design-"
              "standalone-process-mode.md §5."
          ) from exc
      logger.warning(...)  # distributed: unchanged, falls back to staging
      return
  if not ok:
      if is_standalone_process:
          raise RuntimeError(
              "register_memory returned false in standalone-process mode "
              "(fatal -- no fallback path exists in v0.1)."
          )
      logger.warning(...)  # distributed: unchanged
  ```
  `register_mem_pool_host`/`register_memory` call sites themselves
  (`umbp_store.py:896-912`) still pass `(host_ptr, host_size)`
  unchanged — the fd-passing handshake described in §3.2 happens
  entirely inside `StandaloneProcessClient::RegisterMemory` in C++ via
  the process-local fd registry (§3.2 step 2), invisible to Python.
  Only the **gate** above it (`umbp_store.py:880`) and the
  **error-handling branches** just described change, both listed as
  their own items in §10.
- `is_distributed()` remains a pure bool getter, unchanged, and
  continues to mean "true cross-node distributed" for the existing
  external-KV branch points that already depend on it. The new
  `get_deployment_mode()` accessor is not optional polish here — §5's
  own fix above depends on it existing.

---

## 6. New / reused configuration surface

**(v0.4 fix — this section previously left it ambiguous whether each
knob is a `UMBPStandaloneProcessConfig` field set by Python or an env
var read directly by C++, and used both descriptions inconsistently
for the same knobs. Resolved below by picking one routing rule per
knob and sticking to it, using the two conflicting precedents already
in this codebase as the tie-breaker.)**

Two existing precedents disagree on how "which deployment mode, and
with what parameters" config should flow, and this design has to pick
one per knob rather than blend them:

- **Precedent A — `cfg.distributed`:** every sub-field
  (`master_address`, `node_address`, `node_id`, ...) is read and
  assigned **entirely in Python** (`umbp_store.py:458-495`), each via
  the same `extra.get(key, _optional_env_str(ENV_VAR))` pattern.
  `UMBPConfig::FromEnvironment()` never touches any of them.
- **Precedent B — `UMBPSsdConfig::spdk_proxy_auto_start`/
  `spdk_proxy_startup_timeout_ms`:** read **directly by
  `UMBPConfig::FromEnvironment()`** in C++ (`config.h:399-402`, `UMBP_SPDK_PROXY_AUTO_START`/`UMBP_SPDK_PROXY_TIMEOUT_MS`), with no
  Python involvement at all.

**Decision: `standalone_process` follows Precedent A end to end** —
every field Python can set, it does, exactly like `distributed`
already works, rather than splitting the same config object across
two different parsing owners. Rationale: `standalone_process` is an
*activation* choice (like `distributed`), not a leaf tuning knob on an
already-active tier (like `spdk_proxy_auto_start` is, one level below
an already-enabled SSD tier) — it belongs with the precedent for the
thing it resembles, not the deepest-nested one that happened to be
read directly by C++.

| Var | Consumed by | Purpose | Default |
|---|---|---|---|
| `UMBP_STANDALONE_ADDRESS` | **Python** (`UMBPStore.__init__`, §5) into `cfg.standalone_process.address` | UDS path (`unix:///run/umbp/standalone/<node_id>.grpc.sock`) of the standalone server's **gRPC** socket. Presence enables standalone-process mode, mirroring how `UMBP_MASTER_ADDRESS` presence enables distributed mode today. | unset (disabled) |
| `UMBP_STANDALONE_AUTO_START` | **Python** into `cfg.standalone_process.auto_start` | If `true` and no server found at `address`, `CreateUMBPClient` forks+execs `umbp_standalone_server` (rank-0-local only, §4.2). | `false` |
| `UMBP_STANDALONE_STARTUP_TIMEOUT_MS` | **Python** into `cfg.standalone_process.startup_timeout_ms` | Bound on waiting for readiness after spawn, mirrors `spdk_proxy_startup_timeout_ms`'s role (but is itself Python-set, per the decision above — it is not read the same way its SPDK-proxy analog is). | `30000` |
| `UMBP_STANDALONE_BIN` | **Deployment-only, not a config field** — read directly by the auto-start code path in C++ at spawn time, same status as `UMBP_MASTER_BIN`/`spdk_proxy_bin` per `runtime-env-vars.md`'s existing "deployment/launcher knobs, not parsed by `UMBPConfig`" category. | Path override for the `umbp_standalone_server` binary. | resolved via `PATH`/build dir |
| `UMBP_STANDALONE_IDLE_EXIT_TIMEOUT_MS` | `umbp_standalone_server`'s own env read at startup (server-side only; never reaches a worker's `UMBPConfig` at all, so the Precedent A/B question doesn't apply to it) | `0` = never self-exit (default; see §4 rationale). Non-zero opts back into SPDK-proxy-style idle exit, only safe if SSD tier durability is also enabled. | `0` |
| `UMBP_STANDALONE_GRPC_SHUTDOWN_DEADLINE_SEC` | `umbp_standalone_server`'s own env read (server-side only) | Reuses the `MasterServer` shutdown-deadline convention. | same default as `UMBP_GRPC_SHUTDOWN_DEADLINE_SEC` |
| `UMBP_STANDALONE_SHM_DIR` | Both sides' own env read (deployment-only, not a config field — used only for the bootstrap-lock file path, mirrors `ScopedBootstrapLock`'s convention) | Directory for the bootstrap-lock file, not the shm segment itself (which is anonymous via `memfd_create`, never named on the filesystem — see §3.2). | `/tmp` |

**(v0.4 fix) `fd_socket` and `transport` are explicitly *not*
`UMBPStandaloneProcessConfig` fields, and there is no
`UMBP_STANDALONE_FD_SOCKET`/`UMBP_STANDALONE_TRANSPORT` env var in
v0.1** — removing the ambiguity the v0.3 draft left open:

- The fd-handoff socket path is **always mechanically derived** from
  `cfg.standalone_process.address` using the deterministic rule
  already given in §4.1 (strip `unix://`, replace trailing
  `.grpc.sock` with `.fd.sock`, else append `.fd.sock`) — computed
  identically inside `StandaloneProcessClient`'s ctor and
  `umbp_standalone_server`'s startup, from the one `address` value
  they already both have. No separate config surface, no env var, and
  therefore nothing that can drift out of sync between the two
  processes. If a real deployment later needs a non-derived override
  (e.g. the fd socket must live on a different mount than the gRPC
  one), add an explicit `fd_socket_override` field to
  `UMBPStandaloneProcessConfig` *then*, as a v2 change — do not
  pre-add it speculatively now.
- `transport` (`uds` vs. the deferred `tcp_staging`, §8 item 6b, §10
  step 13) has no config surface in v0.1 for the same reason
  `tcp_staging` itself isn't built in v0.1: there is nothing yet for a
  `transport` field to select between. When `tcp_staging` is actually
  implemented in a v2 revision, add `cfg.standalone_process.transport`
  (Python-set, Precedent A, for consistency with everything else in
  this table) at that time — not before.

`UMBPConfig` (C++) gains: `std::optional<UMBPStandaloneProcessConfig> standalone_process;`
with exactly three fields — `address`, `auto_start`,
`startup_timeout_ms` — all three Python-set per the decision above,
same shape as the existing `UMBPDistributedConfig`
(`common/config.h:212-244`), so `ResolveRole()`/`Validate()` extend
naturally rather than needing a parallel code path.

---

## 7. Compatibility with existing modes

- `local/` (in-process `StandaloneClient`) is **unchanged** and remains
  the default when neither `distributed` nor `standalone_process` is
  set.
- `distributed/` (cross-node master+peers+RDMA) is **unchanged**;
  standalone-process mode does not touch any file under
  `distributed/master/`, `distributed/peer/`, or `distributed/routing/`.
- The shared code touched: `common/config.h` (new optional field),
  `umbp_client_factory.cpp` (new branch), `local/host_mem_allocator.*`
  (new `kAnonymousShm` backing plus the fd registry, additive),
  `local/tiers/copy_pipeline.*` (new `Drain()`, see §4.3),
  `pybind_umbp.cpp` (new `AnonymousShm` enum value + new
  `standalone_process`/`UMBPStandaloneProcessConfig` bindings, see §5
  — **not** change-free, correcting the v0.1 claim), and
  `umbp_store.py`'s `register_mem_pool_host` gate (one conditional,
  see §5 — **not** call-site-free, correcting the v0.1 claim).
- A future extension (not in scope here, flagged for later): the
  standalone server could itself become a `distributed/` peer/leader
  (i.e. a host runs one standalone server that both serves local
  workers over UDS *and* participates in the cross-node distributed
  pool) — `LocalStorageManager`'s existing `SharedSSDLeader`/
  `SharedSSDFollower` roles already hint at this kind of layering, but
  wiring it up is out of scope for v0.1.

---

## 8. Open questions / risks

1. **Crash / restart semantics are unsolved by every precedent
   reviewed.** Neither LMCache's server, Mooncake's Real Client, nor
   UMBP's own SPDK-proxy daemon has a supervised-restart or
   reconnect-with-cache-rehydration story in the code we read. If the
   standalone server dies, every attached worker loses its DRAM cache
   simultaneously. Minimum bar for v0.1: workers must treat a broken
   RPC channel as "cache unavailable," not a fatal error (falling back
   to no-cache behavior, same as any other `HiCacheStorage` backend
   failure) — this needs explicit handling in `StandaloneProcessClient`
   and probably a small state machine (`CONNECTED` → `DISCONNECTED` →
   optionally `RECONNECTING`), which does not exist in any of the
   three reference systems today.
2. **Per-tenant DRAM quota is not implemented in `LocalStorageManager`
   today.** Without it, multiple workers sharing one standalone
   server's DRAM tier can starve each other. SPDK-proxy's
   `TenantInfo.quota_bytes` concept (`spdk_proxy_protocol.h:227-240`)
   is the natural template but would need to be added to
   `DRAMTier`/`LocalBlockIndex`, which currently have no notion of
   "owner" per key.
3. **(decided, not open as of v0.2, keeping the entry for traceability)
   fd-passing handshake requires a second, separate socket path from
   the gRPC UDS** (`<addr>.grpc.sock` vs `<addr>.fd.sock`, §2 diagram,
   §6) since gRPC cannot itself carry a file descriptor. The v0.1 draft
   said this but its own diagram contradicted it ("over the same
   UDS") — that inconsistency is fixed in §2/§6/§9. What remains
   genuinely open: whether the fd-handoff listener should be a
   long-lived socket the server keeps open for the whole process
   lifetime (simplest, matches Mooncake's `real_client_main.cpp:104`)
   or a short-lived one opened only during an active registration
   window (marginally more defensible against unauthorized local
   connection attempts, at the cost of a startup race the client must
   retry against). Proposal: long-lived, protected by filesystem
   permissions (item 4 below) — simplicity over a marginal hardening
   gain, revisit if a security review disagrees.
4. **Security/isolation**: the shm segment (memfd, `O_CLOEXEC`
   recommended) and the UDS socket file must both be created with
   `0600`/owner-only permissions, or any other local user can attach
   to another tenant's KV cache. Neither Mooncake's nor UMBP's SPDK-
   proxy code reviewed explicitly sets restrictive permissions on
   their sockets/shm — this needs to be added deliberately, not copied.
5. **gRPC-over-UDS batch overhead vs. SPDK-proxy's ring buffer**: even
   with struct-of-arrays batching, a gRPC call still costs a syscall +
   protobuf encode/decode per `BatchGet`/`BatchPut`, which a busy-polled
   shm ring buffer avoids entirely for the hottest path. If profiling
   later shows this is a bottleneck for very small, very frequent Get
   calls (e.g. single-block prefix-cache lookups), a v2 iteration could
   add an optional shm-ring **fast path** for single-key Get/Put
   alongside the gRPC path for everything else — deferred, not part of
   v0.1, to avoid combining two IPC mechanisms before the simpler one
   is proven to be insufficient.
6a. **(new in v0.2) The process-local ptr→fd registry (§3.2 step 2)
   is new state with its own lifetime/thread-safety questions no
   existing code needs to answer today.** `HostMemAllocator::Alloc`/
   `Free` (`host_mem_allocator.h:41-66`) are currently stateless
   beyond the returned handle — adding a shared registry means: (a) it
   must be safe for `Free` to run concurrently with a
   `RegisterMemory` lookup from a different thread (mutex, as noted in
   §3.2), (b) `DeregisterMemory` and `Free` ordering must be defined —
   if a caller frees the underlying buffer before calling
   `DeregisterMemory`, the registry entry and the server's `mmap`
   would dangle; the design should make `Free` refuse (or at least
   warn loudly) if an active registration still references that
   pointer, rather than silently leaving the server with a stale
   mapping.
6b. **(new in v0.2) The `tcp_staging` transport (§3.1, §6) is a real
   second code path, not a config toggle on the same code.** Once
   `Put`/`Get` payload bytes can travel two different ways (shm-offset
   reference vs. inline bytes in the RPC message), `UMBPStandalone`'s
   proto and `StandaloneProcessClient`'s implementation both need an
   explicit branch, and testing needs to cover both — this roughly
   doubles the data-plane test matrix for standalone-process mode
   versus what v0.1 implied. Recommendation: **do not build
   `tcp_staging` in v0.1** unless a concrete deployment already needs
   it; ship UDS-only first and treat the inline-payload fallback as a
   v2 addition once there's a real caller for it, rather than building
   two data planes up front for a use case that may never materialize.
7. **Version/ABI skew**: the standalone server binary and the
   `mori_pybinds`-linked proto definitions in each worker process must
   agree on the `UMBPStandalone` proto schema. This is the same
   constraint `distributed/` already has between `umbp_master` and
   worker processes, not a new problem — but standalone-process mode
   is more likely to be deployed as a long-lived sidecar that outlives
   several SGLang worker restarts/upgrades, which makes schema drift
   more likely in practice than it is for `umbp_master` today. No
   solution proposed here beyond "same binary/version pinning
   discipline as the existing distributed mode."

---

## 9. Source references used in this design

All claims above trace to source read directly in this repo checkout,
not to general knowledge:

- UMBP: `mori/src/umbp/umbp_client_factory.cpp`,
  `include/umbp/umbp_client.h`, `include/umbp/common/config.h`,
  `local/standalone_client.{h,cpp}`, `local/host_mem_allocator.{h,cpp}`,
  `local/tiers/dram_tier.cpp`, `local/tiers/local_storage_manager.cpp`
  (`SpawnProxyDaemon`/`EnsureProxyDaemon`),
  `local/tiers/copy_pipeline.{h,cpp}`,
  `storage/spdk/proxy/spdk_proxy_protocol.h`,
  `storage/spdk/proxy/spdk_proxy_shm.cpp`,
  `distributed/peer/batch_resolve_codec.h`,
  `distributed/master/master_server.cpp`, `distributed/master/master_client.cpp`,
  `src/pybind/pybind_umbp.cpp`, `src/pybind/CMakeLists.txt`,
  `doc/runtime-env-vars.md`, `scripts/run_umbp_single_node_hicache.sh`.
- SGLang: `sglang/python/sglang/srt/mem_cache/storage/umbp/umbp_store.py`,
  `umbp_host_allocator.py`.
- Mooncake (`/apps/nima/KVManager/Mooncake`, local checkout):
  `mooncake-store/src/real_client.cpp`, `real_client_main.cpp`,
  `shm_helper.cpp`, `dummy_client.cpp`, `client_buffer.cpp`,
  `docs/source/design/mooncake-store.md`,
  `docs/source/getting_started/examples/sglang-integration/hicache-integration-v1.md`.
- LMCache (`/apps/nima/KVManager/LMCache`, local checkout, reviewed
  for comparison only): `lmcache/v1/multiprocess/{mq,server,protocol,
  custom_types,futures,config}.py`, `lmcache/v1/mp_observability/`.

## 10. Suggested implementation order (for step 3, pending approval)

**(Reordered/expanded across v0.2 and v0.3 to make each review-found
gap a concrete, individually-checkable task rather than an implicit
sub-step.)**

1. `common/config.h`: add `UMBPStandaloneProcessConfig` +
   `UMBPConfig::standalone_process`; extend `Validate()`/`ResolveRole()`.
2. `local/host_mem_allocator.{h,cpp}`: add `kAnonymousShm` backing
   (`memfd_create`-based) **plus** the process-local ptr→fd registry
   (§3.2 step 2, §8 item 6a) that `RegisterMemory` will query — these
   belong in the same change since the registry is populated at
   `Alloc`/`Free` time.
3. `local/tiers/copy_pipeline.{h,cpp}`: add `Drain(timeout)` (§4.3,
   §8 item 7 fix) — small, independent of everything else, can land
   first.
4. `src/pybind/pybind_umbp.cpp`: add `AnonymousShm` to
   `UMBPHostBufferBacking`; bind `UMBPStandaloneProcessConfig` and
   `UMBPConfig::standalone_process`; add `get_deployment_mode()` to
   the `IUMBPClient` binding (§5). Do this **before** step 6 below —
   the Python-side change depends on these bindings existing.
5. New proto `distributed/proto/umbp_standalone.proto` +
   generated-code wiring in `CMakeLists.txt`; struct-of-arrays batch
   codec reuse for `BatchPut`/`BatchGet` (§3.1).
6. New `umbp_standalone_server` binary (own `main.cpp`, thin wrapper
   around a new `StandaloneServer` class that owns
   `LocalStorageManager`/`LocalBlockIndex`/`CopyPipeline` — reused, not
   reimplemented, with `DRAMTier` defaulting to private
   anon/hugetlb per §3.2/§7, not `use_shared_memory=true`) — plus the
   gRPC service on `<addr>.grpc.sock` and the **separate** raw-UDS
   fd-handoff listener on `<addr>.fd.sock` (§2, §8 item 3).
7. New `StandaloneProcessClient : IUMBPClient` (worker-side): gRPC
   stub, offset translation, the `SCM_RIGHTS` send side of the
   fd-handoff handshake, wired into `umbp_client_factory.cpp`.
   `RegisterMemory` must fail loudly (not silently degrade) when the
   registry lookup misses (§3.2 step 2).
8. **(v0.3 addition, corrected in v0.5 — this step did not exist
   before and is the activation path itself.)** `umbp_store.py`: in
   `UMBPStore.__init__`, next to the existing `master_address` block
   (`umbp_store.py:458-465`): **first**, raise if
   `extra_config["standalone_address"]` is set without
   `UMBP_STANDALONE_ADDRESS` also being set (v0.5 — extra_config alone
   cannot activate this mode, see §5); **then** read
   `UMBP_STANDALONE_ADDRESS`, construct `UMBPStandaloneProcessConfig`,
   assign `cfg.standalone_process`, and raise if both `master_address`
   and the standalone address are set (§5). Depends on step 4 (pybind
   bindings for `UMBPStandaloneProcessConfig` must exist first).
   Without this step, every other step in this list can be implemented
   correctly and `UMBPStore` will still silently construct a plain
   `StandaloneClient` forever — this is not optional polish, it is the
   switch that turns the feature on.
9. **(v0.5 fix — must read the env var directly, at allocator
   construction time, before `extra_config` exists — see §5.)**
   `umbp_host_allocator.py`: `UMBPHostTensorAllocator` reads
   `UMBP_STANDALONE_ADDRESS` directly from `os.environ` (not via
   `UMBPConfig`/`extra_config`) and branches to request `AnonymousShm`
   backing when set (depends on step 4 for the pybind enum value).
   This step and step 8 both gate on the *same* env var by
   construction, precisely so they can never disagree about whether
   standalone-process mode is active.
10. **`umbp_store.py`: fix the `register_mem_pool_host` gate at line
    880** from `is_distributed()`-only to also accept
    `get_deployment_mode() == StandaloneProcess` (§5) — called out as
    its own task because it is a functional bug an implementation
    could ship with even after step 8 above: the client would be
    correctly constructed as `StandaloneProcessClient`, but still
    never register its host buffer, so every `Put`/`Get` would fail at
    the server's offset-translation step with no obvious cause.
10a. **(v0.5 addition.)** `umbp_store.py`: in the same method, make
    `disable_zero_copy_register`-set / `register_memory` exception /
    `register_memory() == False` all `raise` instead of
    warn-and-return when `get_deployment_mode() == StandaloneProcess`
    (§5); leave `distributed`'s existing warn-and-fallback behavior
    untouched. Land this together with step 10 — both touch the same
    function and the same review found both gaps together.
11. Lifecycle: auto-start path reusing `SpawnProxyDaemon`'s fork/exec
    pattern; shell script analogous to `run_umbp_single_node_hicache.sh`
    for the externally-launched path.
12. Tests: unit tests for the fd-handoff handshake and offset
    translation, the ptr→fd registry's `Free`/`DeregisterMemory`
    ordering including a register→deregister→re-register cycle (§3.2
    step 4, §8 item 6a), `CopyPipeline::Drain()` under load; an
    integration test with 2+ worker processes sharing one standalone
    server on one host, including a kill-the-server test that asserts
    workers degrade to cache-unavailable rather than crashing (§8
    item 1), and an end-to-end test that `UMBP_STANDALONE_ADDRESS`
    alone (via `UMBPStore` config, no other code changes) is sufficient
    to activate the mode (guards against step 8 regressing silently).
    **(v0.5 additions)** a Python test asserting
    `extra_config["standalone_address"]` without the env var raises at
    `UMBPStore.__init__` (guards step 8's rejection path); and Python
    tests for each of the three `register_mem_pool_host` cases (§5,
    step 10a) asserting they raise under `StandaloneProcess` and still
    only warn under `Distributed` (guards against the two modes'
    branches getting merged back together in a future edit).
13. **Deferred to v2, not v0.1** (§8 item 6b): `tcp_staging` transport
    with inline RPC payload bytes. Build only if a concrete deployment
    needs it.
