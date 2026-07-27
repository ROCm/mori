# UMBP Standalone Process Mode — Design

## 1. Background

UMBP (mori's tiered KV-cache system) currently supports two deployment
shapes:

- **`local/`** — an in-process `StandaloneClient` that owns a
  `LocalStorageManager` (`DRAMTier` + `SSDTier`), a `LocalBlockIndex`,
  and a `CopyPipeline`, all living inside the calling process.
- **`distributed/`** — a cross-node deployment (`umbp_master` + peers
  over gRPC + RDMA) for sharing a KV cache across machines, with
  master-mediated routing and lease/eviction protocols for remote
  peers.

Neither shape addresses a common single-host deployment pattern: an
8-GPU server running SGLang with tensor-parallel degree 8 spawns 8
worker processes on one host, and today each process runs its own
`local/`-mode UMBP instance. This duplicates the DRAM tier 8x and gives
sibling processes (TP ranks, or a restarted worker resuming a warm
cache) no way to share a cache that physically fits on the same
machine.

Mooncake solves the same problem with a **Real Client / DummyClient**
split: a long-lived Real Client process owns the actual distributed
storage client, and lightweight DummyClient shims inside each
application process forward calls to it over IPC. Mooncake is the
primary architectural reference for this design; LMCache's
`lmcache/v1/multiprocess/` was reviewed for comparison but not used as
a template (its zero-copy path is CUDA-IPC/GPU-only, with no
host-memory equivalent).

A second, initially underappreciated fact about Mooncake's design
matters here: its "standalone" Real Client is not a same-host-only
cache-sharing shim — it is a **full distributed client** (Master + RDMA
transfer engine) that happens to run in its own OS process. A cache
miss on one host can still be served from a remote node over RDMA.
This makes Mooncake's standalone mode cross-node-capable by
construction, not just same-host-capable.

## 2. Goals

1. **Same-host sharing**: let N SGLang/vLLM worker processes on one
   host share a single DRAM/SSD cache tier through a dedicated,
   long-lived OS process, without each process duplicating its own
   tier.
2. **Mooncake parity**: match Mooncake's Real Client capability,
   including cross-node sharing — a standalone-process deployment
   should be able to serve a cache miss from a remote node over RDMA,
   not only from local storage.
3. **Preserve existing UMBP capabilities** that are not present in
   Mooncake but are part of UMBP's baseline `distributed/` mode,
   specifically per-worker external-KV routing precision (used by a
   custom SGLang router to make placement decisions).
4. **No changes to `distributed/`'s cross-node machinery** — master,
   RDMA registration, routing, and eviction are reused unmodified.

### Non-goals (v1)

- Per-tenant DRAM quota enforcement across workers sharing one server.
- A `tcp_staging` (non-UDS, inline-payload) transport — the zero-copy
  shared-memory data plane requires `AF_UNIX`/`SCM_RIGHTS`; no
  drop-in TCP equivalent exists. This is deferred until a concrete
  deployment needs it.
- Worker crash/disconnect detection and cache rehydration after a
  server restart. Tracked as a known limitation (§8).

## 3. Architecture Overview

Two new roles, named after Mooncake's split for consistency:

| UMBP name | Mooncake analog | Lives in | Role |
|---|---|---|---|
| `umbp_standalone_server` | Real Client | its own process | Owns the storage/transport backend (see §3.1) and exposes `IUMBPClient` semantics over gRPC. `mmap`s each worker's registered host-buffer segment (via fd handoff) for zero-copy Put/Get access. |
| `StandaloneProcessClient` | DummyClient | inside the SGLang/vLLM worker process, behind `IUMBPClient` | An `IUMBPClient` implementation. Forwards every call to the server over gRPC. Registers the worker's host KV buffer with the server once via fd-passing, then references it by offset on every hot-path call. |

```
 ┌────────────────────────────┐       ┌────────────────────────────┐
 │ SGLang worker process A    │       │ SGLang worker process B    │
 │                             │       │                             │
 │  UMBPStore (Python)         │       │  UMBPStore (Python)         │
 │       │ pybind11            │       │       │ pybind11            │
 │       ▼                     │       │       ▼                     │
 │  StandaloneProcessClient    │       │  StandaloneProcessClient    │
 │   (IUMBPClient impl)        │       │   (IUMBPClient impl)        │
 │    │ gRPC        │ raw UDS  │       │    │ gRPC        │ raw UDS  │
 │    │ (.grpc.sock)│(.fd.sock)│       │    │ (.grpc.sock)│(.fd.sock)│
 └────┼─────────────┼─────────┘       └────┼─────────────┼─────────┘
      │             │                       │             │
      │             │  one-time fd handoff  │             │
      │             │  (SCM_RIGHTS), at      │             │
      │             │  RegisterMemory() time │             │
      ▼             ▼                       ▼             ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                     umbp_standalone_server                      │
 │                                                                   │
 │  UMBPStandaloneService (gRPC, control plane, own UDS socket)     │
 │    Put/Get/BatchPut/BatchGet/Exists/... — same names as          │
 │    IUMBPClient, arguments are (key, client_id, shm_offset, size) │
 │    instead of raw pointers                                       │
 │                                                                   │
 │  Fd-handoff listener (raw AF_UNIX, SEPARATE socket path from     │
 │    the gRPC UDS — gRPC cannot carry an fd)                       │
 │                                                                   │
 │  client_ : IUMBPClient — either                                  │
 │    StandaloneClient   (local-backend, §3.1)                      │
 │    DistributedClient  (distributed-backend, §6)                  │
 │                                                                   │
 │  Per-worker shm registry: client_id → {fd, base, size, offset}   │
 └─────────────────────────────────────────────────────────────────┘
```

Two logical channels, two socket paths, deliberately: gRPC cannot
carry a file descriptor, so the one-time fd handoff (§4.1) must use a
separate raw `AF_UNIX` socket alongside the gRPC control-plane UDS.

### 3.1 Deployment shapes

`umbp_standalone_server`'s `client_` member is `IUMBPClient`-typed and
constructed through the existing `CreateUMBPClient(config)` factory
(`umbp_client_factory.cpp`). This yields three deployment shapes:

| Shape | `client_` backend | Cross-node | Server's own storage |
|---|---|---|---|
| **local-backend** (same-host, fallback/dev-convenience) | `StandaloneClient` | no | `LocalStorageManager` (DRAM+SSD tiers, private anonymous/hugetlb memory — reused unmodified from `local/`) |
| **distributed-backend, backend enabled** (Mooncake-parity shape) | `DistributedClient` | yes, via Master + RDMA | `DistributedClient`'s own DRAM pool (a distinct concept from `LocalStorageManager` — see §6.5) |
| **distributed-backend, backend disabled/unconfigured** | `StandaloneClient` | no | same as local-backend |

**distributed-backend is the primary target this feature builds
toward**, since it is the shape that actually matches Mooncake parity
(goal 2). **local-backend is a same-host fallback / dev-convenience
shape**, not a co-equal alternative — it remains fully supported and
is the default when the server has no distributed-backend
configuration.

Existing data-path RPC handlers (Put/Get/BatchPut/BatchGet/Exists/
Clear/Flush) require no logic changes across these shapes: they
already resolve `(client_id, shm_offset)` to a plain pointer before
calling into `client_` (§4.2), so `client_`'s concrete type was never
load-bearing for them.

## 4. IPC Mechanism

### 4.1 Control plane: gRPC over a Unix domain socket

UMBP reuses its existing gRPC stack for the control plane rather than
adopting a new RPC framework or Mooncake's `coro_rpc`: `umbp_common`
already depends on gRPC/protobuf for the `distributed/` master/peer
path, and `pybind_umbp.cpp` already has a working convention for
calling blocking C++ RPC methods from Python with the GIL released.
Adopting `coro_rpc` would add a new coroutine-to-pybind11 bridge with
no precedent in this codebase, for no benefit gRPC doesn't already
provide (typed request/response, an async server loop, UDS transport).

Design:

- A new proto service `UMBPStandalone`
  (`distributed/proto/umbp_standalone.proto`) with:
  - RPCs mirroring `IUMBPClient` 1:1: `Put`, `Get`, `BatchPut`,
    `BatchPutWithDepth`, `BatchGet`, `Exists`, `BatchExists`,
    `BatchExistsConsecutive`, `Clear`, `Flush`, `RegisterMemory`,
    `DeregisterMemory`, and the full external-KV set —
    `ReportExternalKvBlocks`, `RevokeExternalKvBlocks`,
    `RevokeAllExternalKvBlocksAtTier`, `MatchExternalKv`,
    `GetExternalKvHitCounts`.
  - `Ping(Empty) -> PingResponse{ready: bool, deployment_mode: LOCAL|DISTRIBUTED}`:
    a new RPC with no `IUMBPClient` analog, used for readiness probing
    and deployment-mode signaling (§6.4).
  - Methods that stay entirely client-local in `StandaloneProcessClient`
    (no RPC): `Close()` (tears down the local gRPC channel + fd-handoff
    socket + registry state) and `get_deployment_mode()` (a pure local
    constant).
  - `BatchPut`/`BatchGet` reuse the existing struct-of-arrays batch
    codec (`distributed/peer/batch_resolve_codec.h`) so per-key
    protobuf submessage overhead doesn't reappear here — the same
    pattern already validated for the peer `BatchResolveKeys` RPC.
- **Transport: gRPC over a Unix domain socket**
  (`unix:///run/umbp/standalone/<node_id>.grpc.sock`), not TCP. This
  is a deliberate deviation from Mooncake (TCP) and LMCache (TCP):
  standalone mode never leaves the host, so a UDS avoids port
  allocation, avoids the network stack, and gets filesystem-permission
  based access control for free. TCP is not a drop-in fallback for
  this design: the data plane (§4.2) depends on `SCM_RIGHTS` fd
  passing, an `AF_UNIX`-only mechanism with no TCP equivalent. A
  deployment that cannot share a UDS-reachable filesystem between
  processes cannot use the zero-copy data plane at all; the only
  fallback would be a non-zero-copy path where `Put`/`Get` payload
  bytes travel inside the gRPC message itself. This mode
  (`tcp_staging`) is not built in v1 (§8).
- **Argument shape**: control messages never carry KV bytes. `Put`/
  `Get` take `(key, client_id, shm_offset, size)` — the raw pointer
  that `IUMBPClient::Put(key, uintptr_t src, size_t size)` takes is
  translated to `(client_id, shm_offset)` inside
  `StandaloneProcessClient` before the RPC is issued. This mirrors
  Mooncake's `map_dummy_buffer_range_to_real`/`shm_addr_offset`
  translation.

### 4.2 Data plane: shared memory

UMBP and Mooncake independently converged on the same shape for the
data plane: control messages carry a handle, KV bytes live in shared
memory (UMBP's own SPDK-proxy daemon uses the same idea for its ring
protocol). The design point that needs to be specified precisely is
**who allocates the shared segment**, and the answer is taken directly
from Mooncake:

1. **The worker process (not the server) owns the buffer**, exactly
   like Mooncake's `DummyClient` owns its local buffer via
   `setup_dummy()` → `ShmHelper::allocate()`. Concretely,
   `HostMemAllocator` (already used by `umbp_host_allocator.py` to
   back SGLang's host KV pool) gains a new backing kind,
   `kAnonymousShm`, using `memfd_create` + `ftruncate` +
   `mmap(MAP_SHARED)` — the same primitives Mooncake's
   `ShmHelper::allocate()` uses. `memfd_create` (anonymous, refcounted
   purely by open fds) is chosen over UMBP's existing named-`shm_open`
   `DRAMTier` path specifically to avoid `/dev/shm` name collisions
   across many SGLang instances on one host, and to avoid leaking a
   named segment into `/dev/shm` if a process is killed uncleanly.
2. **One-time registration handshake**, triggered by the existing
   `RegisterMemory(ptr, size)` call. `IUMBPClient::RegisterMemory`
   only carries a pointer and a size; the file descriptor is recovered
   entirely on the C++ side of the worker process:
   - `HostMemAllocator::Alloc` maintains a process-local,
     mutex-guarded `ptr → {fd, size}` registry, populated only for
     `kAnonymousShm` allocations at alloc time and erased on `Free`.
   - `StandaloneProcessClient::RegisterMemory(ptr, size)` looks up
     `ptr` in that registry (a range/floor lookup, since `ptr` may be
     the tensor's base address rather than the allocation's exact
     base) and, on a hit, sends the recovered fd to the server via
     `SCM_RIGHTS` over the **separate raw-UDS fd-handoff socket**
     (`<addr>.fd.sock`, distinct from the gRPC UDS) — mirroring
     Mooncake's `UdsConnection::sendFd`/`recvFd` and
     `RealClient::handle_ipc_shm_register`. On a registry miss (the
     pointer isn't backed by a `kAnonymousShm` allocation),
     `RegisterMemory` fails loudly rather than silently falling back
     to a broken pointer-based `Put`/`Get`.
   - The server `mmap`s the received fd read-write into its own
     address space and stores `client_id → {base, size}` in a registry
     keyed by `client_id` — mirroring Mooncake's `shm_contexts_`.
3. **Every subsequent Put/Get/BatchPut/BatchGet RPC carries only
   `(client_id, shm_offset, size)`** — `StandaloneProcessClient`
   computes `shm_offset = ptr - registered_base` locally before
   issuing the RPC; the server computes `real_addr = base +
   shm_offset` and reads/writes there directly. No KV bytes ever cross
   the gRPC channel.
4. **fd ownership**: the allocator's ptr→fd registry, not
   `RegisterMemory`/`DeregisterMemory`, owns the fd — `HostMemAllocator`
   opens it at `Alloc` time and is the only thing that closes it, at
   `Free` time. `RegisterMemory` only *borrows* the fd to send over
   `SCM_RIGHTS` (which duplicates it into the receiving process, so
   the sender's copy is unaffected by anything the server does with
   its own copy). `DeregisterMemory` therefore only (a) tells the
   server to `munmap` its copy and drop the `client_id` registry
   entry, and (b) clears `StandaloneProcessClient`'s own bookkeeping —
   it must **not** close the allocator-owned fd. Closing happens
   exactly once, in `HostMemAllocator::Free`.

The server's own DRAM cache tier (local-backend shape) does not need
to be visible to any other process — no worker ever `mmap`s it
directly — so it defaults to the same private
`kAnonymous`/`kAnonymousHugetlb` backing `StandaloneClient` already
uses today, not a named `use_shared_memory=true` POSIX shm segment.

Net effect: the hot path becomes the same shape as UMBP's existing
RDMA-registered `distributed/` path (register once, then
reference-by-handle on every call) — not a new interface concept for
UMBP, just a new transport underneath a pattern `IUMBPClient` already
exposes.

**Rejected alternative — a busy-polled ring buffer** (the SPDK-proxy
daemon's own approach): well suited to SPDK-proxy's low-cardinality,
microsecond-latency block I/O, but a poor fit for a control plane that
needs to express variable-shape requests (batch registration,
variable-length external-KV hash lists) and async multiplexing for
slow operations. Neither Mooncake nor LMCache use a pure ring buffer
for control; both back it with a real RPC/messaging library and
reserve raw shared memory for bulk bytes, which this design also does.
The data plane still uses the same "handle + offset/size, bytes in
shm" idea as SPDK-proxy — only the ring buffer itself is rejected, for
the control plane specifically.

## 5. Lifecycle Management

The standalone server holds the DRAM cache tier's actual data, so
unlike UMBP's own SPDK-proxy precedent (a stateless passthrough to a
shared NVMe device, safe to self-exit and respawn), it defaults to
Mooncake/LMCache's model: **externally launched, long-lived, no
self-exit**, with idle-self-exit available as an explicit opt-in.

### 5.1 Primary path: externally launched

- Launched the same way `run_umbp_single_node_hicache.sh` already
  launches `umbp_master` (background process, log redirected, `trap
  cleanup EXIT`) — no new orchestration concept, just a new binary
  target.
- Discovery: `UMBP_STANDALONE_ADDRESS` (default
  `unix:///run/umbp/standalone/<node_id>.grpc.sock`), following the
  naming convention of `UMBP_MASTER_ADDRESS`.
- The fd-handoff socket path is always **mechanically derived** from
  `cfg.standalone_process.address` — by both processes independently:
  strip the `unix://` scheme prefix, then replace a trailing
  `.grpc.sock` with `.fd.sock` (or append `.fd.sock` for a custom
  path that doesn't end in `.grpc.sock`). This derivation lives in one
  shared place, computed identically on both sides, and deliberately
  has no independent config field or env var, so it cannot drift out
  of sync between the two processes.
- Readiness probe: connect to the UDS and call `Ping`.

### 5.2 Convenience path: auto-start (opt-in, local-backend only)

When `cfg.standalone_process.auto_start` is `true`: the first worker
process on a host to reach `CreateUMBPClient` (leader election reuses
the existing `UMBP_ROLE`/`LOCAL_RANK`/`OMPI_COMM_WORLD_LOCAL_RANK`
rank-0 logic in `UMBPConfig::FromEnvironment` — no new election
protocol) probes the UDS; if absent, it forks+execs the server binary
using the same `fork()` + `setsid()` + `execlp()` sequence
`LocalStorageManager::SpawnProxyDaemon` already uses, and waits for
readiness bounded by `startup_timeout_ms`. A bootstrap lock (the
existing `ScopedBootstrapLock` pattern) prevents a thundering herd of
workers all trying to spawn the server simultaneously. No CUDA/HIP/
gRPC call may happen between `fork()` and `execlp()`, matching the
fork-safety discipline `SpawnProxyDaemon` already requires.

Auto-start is opt-in, not default, because a worker's `fork()`'d child
inherits CUDA/HIP context and GPU device state at the moment of fork.

**A distributed-backend server does not support auto-start — external
launch only.** This matches Mooncake, where the application side
(`DummyClient`) never forks/execs the Real Client under any
circumstance; the Real Client is always started by something outside
the application. It also avoids a real conflict: the env vars needed
to configure a distributed backend (`UMBP_MASTER_ADDRESS`, etc.) would
most naturally live in the worker's own environment if the worker
forked the child, but a worker's own config parsing treats
`UMBP_STANDALONE_ADDRESS` and `UMBP_MASTER_ADDRESS` both being set as
a hard error (mutual exclusivity between standalone-process and
distributed worker modes). Local-backend auto-start is unaffected by
this restriction.

### 5.3 Shutdown

`SIGTERM`/`SIGINT` triggers, in order:

1. Drain in-flight RPCs (bounded deadline, mirroring
   `UMBP_GRPC_SHUTDOWN_DEADLINE_SEC`/`MasterServer::Shutdown()`).
2. **(distributed-backend only)** Unregister all live
   `ExternalKvIdentityClient` instances (§7) — before the shared
   backend is closed, so the Master's `ClientRegistry` doesn't briefly
   carry stale `ALIVE` entries for a process that is already exiting.
3. **(local-backend)** Flush `CopyPipeline` (best-effort persist
   DRAM-tier dirty pages to SSD, if the SSD tier is enabled and
   `force_ssd_copy_on_write` is set) — `CopyPipeline::Drain(timeout)`
   blocks until the async copy queue is empty or the timeout elapses;
   both the shutdown path and the `Flush` RPC call it.
   **(distributed-backend)** Close the `DistributedClient` backend
   (`client_->Close()`).
4. `munmap` all registered client shm segments.
5. Exit.

Steps 2 and 3/4 for a given registered client must run in the order
"deregister from `client_` → `munmap`" (not the reverse), and the
whole backend-deregistration sequence must complete before
`client_->Close()` — deregistering after `Close()` would call into a
backend that has already torn down its RDMA/IO engine state.

Client-side: if the RPC channel drops (server crashed or was killed),
`StandaloneProcessClient` must not silently return stale success —
every in-flight call fails, and the worker-side behavior is to treat
this as "cache miss / cache unavailable" (SGLang's `HiCacheStorage`
abstraction already tolerates backend failures returning
`False`/`None`), not to crash the inference process. There is
currently no supervised-restart or reconnect path (§8, item 1).

### 5.4 Multi-tenant workers on one host

Multiple SGLang worker processes (e.g. one per TP/DP rank) attach to
the same standalone server. Each gets a `client_id` (UUID, assigned at
first `RegisterMemory` call) and its own shm registration entry,
mirroring Mooncake's per-client `shm_contexts_` map. Capacity
accounting inside `LocalStorageManager`/`DRAMTier` is global, not
per-caller — per-tenant DRAM quota is not implemented (§8, item 2).

### 5.5 Duplicate writes from sibling workers

`local/` mode provides a Shared-SSD Leader/Follower model
(`UMBPRole::SharedSSDLeader`/`SharedSSDFollower`, `common/config.h`)
for the case where N sibling worker processes each own their own
`LocalStorageManager` but point at one shared on-disk SSD directory:
only the leader writes to SSD, followers open it read-only
(`standalone_client.cpp`, `local_storage_manager.cpp`,
`ssd_tier.cpp`), so sibling ranks do not issue duplicate or concurrent
writes to the shared segmented-log. Role assignment uses rank-0-elects-
leader logic in `UMBPConfig::FromEnvironment()`.

Standalone-process mode addresses the same concern by construction
rather than by role coordination, because its topology is different:
there is a single backend instance (one `client_`, one
`LocalStorageManager` or one `DistributedClient`) shared by all workers
over IPC, and all data-path RPCs are serialized on the server's
`client_mu_` (`standalone_server.cpp`). There is therefore a single
writer to the backing storage, so the multi-writer-to-shared-storage
coordination the Leader/Follower model exists for does not arise. The
server accordingly constructs its backend with `role =
UMBPRole::Standalone` (`follower_mode = false`,
`standalone_server_main.cpp`); the Leader/Follower code path is not
exercised in this mode.

Duplicate writes of the same key from different workers are collapsed
by the backend's existing content-addressed index:
`StandaloneClient::Put`/`BatchPut` return early for a key already
present (`index_.MayExist`, `standalone_client.cpp`), so a repeat write
of an already-stored key is a no-op rather than a second copy into the
tier. This is first-writer-wins with no content comparison: if two
workers write different bytes under the same key, writes after the
first are dropped (reported successful, not stored). Whether sibling
workers actually issue same-key writes is determined by the caller
(SGLang), which is outside this design's scope; this section only
describes the server's behavior when they do.

## 6. Cross-Node Extension: `DistributedClient`-backed Server

### 6.1 Two independent `UMBPConfig` instances

There are two separate configuration surfaces, and they stay separate:

1. **Worker-facing** (`standalone_process` field): what a
   `StandaloneProcessClient` inside an SGLang worker uses to find and
   talk to the server over UDS. A worker never knows or cares whether
   the server it's talking to is itself distributed-backed.
2. **Server's own internal config** (`distributed` field): what
   `umbp_standalone_server` itself uses to decide whether its
   `client_` backend should be a `StandaloneClient` or a
   `DistributedClient`. This is a second, independent `UMBPConfig`
   instance, built by `standalone_server_main.cpp` from the server's
   own process environment (§6.6) — never the same object as #1.

`UMBPConfig::Validate()` rejects `distributed.has_value() &&
standalone_process.has_value()` *within one config instance* — it does
not constrain two different `UMBPConfig` objects existing in the same
process, so this split requires no change to that check.

```
 Worker process                    umbp_standalone_server process
 ┌────────────────────────┐        ┌───────────────────────────────────┐
 │ UMBPConfig #1           │  UDS + │ UMBPConfig #1 (mirror of the       │
 │  standalone_process =   │─SCM_──▶│  worker's, address/fd-socket)      │
 │   {address}             │ RIGHTS │  -> used only to open sockets      │
 │  distributed = null     │        │                                     │
 └────────────────────────┘        │ UMBPConfig #2 (server's own)       │
                                    │  distributed = {master_address,    │
                                    │    node_id, node_address,          │
                                    │    io_engine, ...}  (if enabled)   │
                                    │  standalone_process = null         │
                                    │       │                             │
                                    │       ▼                             │
                                    │  client_ = CreateUMBPClient(#2)    │
                                    │   -> DistributedClient             │
                                    │      (Master + RDMA)               │
                                    │      -- or, if #2.distributed      │
                                    │         is unset --                │
                                    │   -> StandaloneClient              │
                                    │      (local-backend, unchanged)    │
                                    └───────────────────────────────────┘
                                                  │  RDMA (if DistributedClient)
                                                  ▼
                                    ┌───────────────────────────────────┐
                                    │  umbp_master + other nodes'        │
                                    │  peers (existing distributed/      │
                                    │  cluster, entirely unmodified)     │
                                    └───────────────────────────────────┘
```

### 6.2 Zero-copy RDMA registration of a shared mapping

RDMA memory registration (`DistributedClient::RegisterMemory` →
`PoolClient::RegisterMemory`) operates purely on `(ptr, size)` in the
calling process's own virtual address space (`io_engine_->RegisterMemory
(ptr, size, ...)`), caching `{base, size, mem_desc}` keyed by that
local VA. Nothing checks who allocated the memory or whether another
process also maps the same physical pages.

This means `umbp_standalone_server`, after `mmap`-ing the worker's
`memfd`-backed region via the fd-handoff path (§4.2), can call
`client_->RegisterMemory(server_local_va, size)` on its own local VA
for that mapping. Since the server's VA and the worker's VA back the
same physical pages (`MAP_SHARED` on the same `memfd`), registering
the server's VA for RDMA registers those physical pages for RDMA,
regardless of which process's VA was used to register them.

`Get()` zero-copy is conditional on this registration: the RDMA read
path (`PoolClient`'s remote-get handling) only skips the memcpy-out
when the destination pointer was pre-registered and found in the
registered-memory index — in that branch, the RDMA read is posted
directly into the registered memory region with no memcpy afterward.
Registering the server's local mapping once, at fd-handoff time,
therefore gives every subsequent `Get` targeting that region true
zero-copy RDMA: the remote write lands in memory the worker can
already see, with no copy on the server side and no RDMA involvement
on the worker side at all — the worker process never touches RDMA/IB
verbs directly, only the server does.

Mooncake's Real Client does the identical dual-mapping pattern in
production: the application (`DummyClient`) allocates the shared
region and performs the first `mmap` via `ShmHelper::allocate()`, then
sends the fd to the Real Client, which performs a second, independent
`mmap` of the same physical pages (`RealClient::map_shm_internal_with_device`)
and registers that server-local mapping with its own Transfer Engine.
This is the same allocation/registration direction
`umbp_standalone_server` already implements (worker allocates and owns
the buffer; server receives the fd and independently mmaps it) — a
shipped, production pattern, not a novel one this design introduces.

**Memory-visibility contract**: the gRPC `Get` response is the
happens-before edge. `umbp_standalone_server` sends the gRPC response
for a `Get` only after the RDMA read completion is observed
server-side; the worker must not read the target offset until it has
received that response.

### 6.3 Backend registration lifecycle

`RegisterMemory`/`DeregisterMemory` currently only perform fd/mmap
bookkeeping and never touch `client_`. They are extended to also
register/deregister the mapped region with the backend:

- `client_->RegisterMemory(server_local_ptr, size)` is safe to call
  unconditionally regardless of backend — `IUMBPClient::RegisterMemory`'s
  default (used by `StandaloneClient`, which never overrides it) is
  already a no-op returning `true` ("standalone mode needs no
  registration — CPU-local memcpy").
- **Registration**: the call is placed inside the existing fd
  registration path, immediately after `mmap` succeeds and before the
  registry entry is written. If it fails, the just-created mapping is
  `munmap`'d and the call returns failure without writing the registry
  entry — the same single-phase-commit shape the registration path
  already has for an `mmap` failure.
- **Deregistration**: `client_->DeregisterMemory(server_local_ptr)` is
  called before `munmap`-ing, so the backend's view of the memory is
  torn down before the memory itself disappears.
- **Shutdown**: every registered mapping is deregistered from `client_`
  before being `munmap`'d, and this whole sequence runs before
  `client_->Close()` (§5.3).

Out of scope, tracked as a pre-existing characteristic orthogonal to
backend choice: if a worker sends its fd via the fd-handoff socket and
then crashes before issuing the confirming gRPC `RegisterMemory` call,
the mapping stays live indefinitely — there is no timeout/reaper for
this today (§8, item 9).

`Flush()`/`Clear()` have backend-dependent meaning, by design: under
`StandaloneClient` they drain `CopyPipeline` to SSD; under
`DistributedClient` they operate on the pool client's own state
(heartbeat/registration bookkeeping — there is no local SSD tier in
this backend, §6.5). The server passes both calls through unchanged;
this is the correct, existing behavior of each `IUMBPClient`
implementation, not something this design needs to reconcile.

### 6.4 Deployment-mode signaling

`PingResponse` includes a `deployment_mode` field (`LOCAL`/
`DISTRIBUTED`), populated from `client_->GetDeploymentMode()`. A
single `UMBP_STANDALONE_ADDRESS` can resolve to either a local-backed
or a distributed-backed server, and the failure mode this guards
against is silent: a distributed-backend deployment whose server
ends up local-backed anyway (e.g. an operator's launch script forgot
`UMBP_MASTER_ADDRESS`) would otherwise have every RPC succeed while
the feature's entire purpose — cross-node capability — silently does
not exist. Launch/health-check/integration-test tooling for the
distributed-backend shape must assert `deployment_mode == DISTRIBUTED`
after startup; a distributed-backend server that comes up as `LOCAL`
is a failed deployment, not a degraded-but-working one.

### 6.5 The server's own storage, under each backend

When `client_` is a `DistributedClient`, the server's
`LocalStorageManager`/`DRAMTier`/`SSDTier` concept does not apply at
all — `DistributedClient` doesn't use `LocalStorageManager`; it has
its own separately-owned DRAM pool (built via `HostMemAllocator`
inside `PoolClient`/`DistributedClient`'s own construction), used as
its local slice of the distributed pool, not as a passthrough cache
for registered worker buffers. The registered worker buffers (§6.3)
exist purely so RDMA can target them directly; `DistributedClient`
does not manage them as cache storage. See the deployment-shapes table
in §3.1.

### 6.6 Server-side distributed configuration

`standalone_server_main.cpp` builds the second, independent
`UMBPConfig` (§6.1) for the backend from its own process environment —
never from the worker's environment. A `BuildDistributedBackendConfigFromEnv()`
helper mirrors `umbp_store.py`'s env-var-driven construction of
`UMBPDistributedConfig`:

- **Required** (absence means "distributed backend not requested" —
  falls back to local-backend; some-but-not-all set is a
  misconfiguration and hard-fails): `UMBP_MASTER_ADDRESS`,
  `UMBP_NODE_ADDRESS`, `UMBP_NODE_ID`, `UMBP_IO_ENGINE_HOST`.
- **Optional**, defaulting the same way `UMBPDistributedConfig`
  already does when unset: `UMBP_IO_ENGINE_PORT`,
  `UMBP_PEER_SERVICE_PORT` (reusing the worker-side env-var names,
  since those refer to the same concept in both places), and five
  server-backend-only names with a `UMBP_DISTRIBUTED_` prefix (chosen
  because these five have no existing worker-side env-var form to
  collide with — worker-side they are `extra_config`-only):
  `UMBP_DISTRIBUTED_STAGING_BUFFER_SIZE`,
  `UMBP_DISTRIBUTED_SSD_STAGING_BUFFER_SIZE`,
  `UMBP_DISTRIBUTED_SSD_STAGING_BUFFER_SLOTS`,
  `UMBP_DISTRIBUTED_CACHE_REMOTE_FETCHES`,
  `UMBP_DISTRIBUTED_DRAM_PAGE_SIZE`.
- `dram_page_size` is normally auto-derived by probing a worker's own
  `mem_pool_host`; this does not apply server-side, since the server's
  backend isn't attached to any one worker's tensor layout and may
  serve workers with different layouts. The server-side builder only
  supports the explicit-override env var above, defaulting to `0`
  (delegating to the master's own default) when unset.
- `auto_heartbeat` has no dedicated env var; the server-side helper
  keeps the `UMBPMasterClientConfig` default (`true`) unconditionally.

If `UMBP_MASTER_ADDRESS` is set but the IO engine cannot actually be
stood up (e.g. `UMBP_IO_ENGINE_HOST` empty), **server startup hard-fails**
rather than silently falling back to a `StandaloneClient` backend — a
misconfigured distributed-backend server must refuse to start, not
quietly become a smaller feature.

Distributed-backend supports external launch only (§5.2) — no
auto-start for any deployment shape.

## 7. External-KV Per-Worker Identity (`ExternalKvIdentityClient`)

### 7.1 Layering

This is a UMBP-compatibility extension, not part of Mooncake-parity
core. The parity core (§6.1–§6.4) alone is what makes this design
match Mooncake's Real Client shape; `ExternalKvIdentityClient` exists
only to avoid regressing a capability the `distributed/` baseline
already has (per-worker external-KV routing precision). It has no
Mooncake analog, is not required for declaring Mooncake parity
achieved, and can be scheduled, implemented, and tested as a separable
unit of work — a distributed-backend server without it is a legitimate
intermediate state (external-KV simply stays unavailable, exactly as
the `StandaloneClient`-backed case already handles it).

### 7.2 What external-KV is

A pure query/registry service, not a data-transfer mechanism.
`Report`/`RevokeExternalKvBlocks` record advisory "node N holds hash H
at tier T" facts in the Master's `GlobalBlockIndex`; `MatchExternalKv`
answers "which node(s) hold these hashes" for a caller (typically a
custom SGLang router) making a routing decision — the caller sends a
new request to whichever node the match points at, and that node's own
already-resident local cache serves it. UMBP itself never moves the
reported bytes for this feature, in `distributed/` mode or here.

### 7.3 Identity granularity

What matters is whether the reported/matched identity (`node_id`)
reflects the actual deployment topology at the granularity a caller
needs. The `distributed/` baseline's `node_id` is built from
per-process rank coordinates
(`f"{node_address}:dp{dp_rank}:pp{pp_rank}:tp{local_rank}"`) with no
awareness of logical TP/DP group boundaries — this means the baseline
already supports, for free, multiple independent replicas sharing one
physical host (e.g. two TP=4 groups on one 8-GPU box), since each
process gets its own distinct `node_id` by rank coordinate regardless
of group membership. For this project's stated deployment shape
(8-GPU hosts running one SGLang TP=8 replica per host) a single shared
per-host identity would be semantically adequate on its own, but would
silently regress the multi-replica-per-host case the baseline already
supports.

**Decision: the server maintains one independent distributed
sub-identity per connected worker (`client_id`)**, implemented by a
new, purpose-built `ExternalKvIdentityClient` class rather than N
`MasterClient` instances, for two reasons:

- **Routing/eviction isolation.** A naive `MasterClient`-based
  sub-identity with nonzero `tier_capacities` would become eligible
  for ordinary `RoutePut`/`EvictionManager` target selection
  (`ClientRegistry::GetAliveClients()`), risking real KV blocks being
  routed onto a node that isn't a real storage participant.
  `ExternalKvIdentityClient` has no `tier_capacities`-reporting code
  path at all (always empty) and never publishes `EventBundle`s on
  heartbeat — both `RoutePut`'s `CollectEligibleOnTier` and
  `EvictionManager::RunOnce` already skip any client with no entry for
  the tier being considered, so this closes the issue structurally,
  with zero changes to the routing/eviction algorithms themselves.
- **Avoiding a known `MasterClient` bug.** `MasterClient`'s existing
  re-register path (triggered by `CLIENT_STATUS_UNKNOWN`) omits
  `peer_address`/`engine_desc`, silently breaking `MatchExternalKv`
  responses for that identity after a Master restart or reaper expiry.
  `ExternalKvIdentityClient` is new code with a single
  `BuildRegisterRequest()` helper shared by the initial `Register()`
  call and any re-register path, so both fields are always included by
  construction.

All N sub-identities (N ≤ 8 for the stated deployment shape) share one
common `peer_address` — the server's own single physical `PeerService`
endpoint. Neither `ClientRegistry` nor `MasterServer` assumes a 1:1
`node_id` ↔ `peer_address` mapping (`ClientRegistry` stores a plain
`node_id → peer_address` map; `MasterServer::GetOrCreateStub` keys its
gRPC stub cache by `node_id`), so this is not blocked by any existing
invariant. Each of the five external-KV RPC handlers dispatches to the
calling `client_id`'s own `ExternalKvIdentityClient` instance. The
bulk Put/Get/RegisterMemory RDMA data path is unaffected and continues
to share the server's one `DistributedClient` backend — RDMA
registration is inherently per-VA and identity-agnostic, so only the
identity-bound external-KV surface needs per-worker handling.

**Rejected alternative**: one shared identity + `tenant_id`-style key
namespacing under one shared identity — the pattern Mooncake itself
uses for its own multi-`DummyClient`-per-`RealClient` case. Rejected
because it would lose the `distributed/` baseline's existing
per-worker routing precision, and because Mooncake has no feature
analogous to external-KV that would validate the pattern for this use
case. This is genuinely new engineering for UMBP with no precedent in
either UMBP's own `distributed/` code or in Mooncake.

### 7.4 Wire schema

The worker conveys `worker_node_id`, `worker_node_address`, and
optionally `tags` — not raw rank components — added to the existing
`RegisterMemoryRequest` message rather than a new RPC:

```protobuf
message RegisterMemoryRequest {
  string client_id = 1;
  uint64 worker_base = 2;
  uint64 size = 3;
  string worker_node_id = 4;       // new
  string worker_node_address = 5;  // new
  repeated string tags = 6;        // new, opaque key=value labels
}
```

`RegisterMemoryRequest` is already the gRPC call that completes a
worker's registration lifecycle (the confirmation step after the raw
fd handoff), so piggybacking on it means a worker's memory
registration and its external-KV identity registration start and end
together, without a second round trip or a second lifecycle to keep in
sync. `worker_node_id`/`worker_node_address` are plain (not
`optional`) strings; a worker that never sets them (e.g. talking to a
`StandaloneClient`-backed server) leaves them empty, and the server
only constructs an `ExternalKvIdentityClient` for a `client_id` whose
`RegisterMemoryRequest` carried a non-empty `worker_node_id`. Python
(`umbp_store.py`'s existing node_id/node_address derivation) remains
the single source of truth for these strings; the server never
re-derives rank semantics.

### 7.5 Interface and lifecycle

`ExternalKvIdentityClient` is a client-side C++ class inside
`umbp_standalone_server`, using the existing `UMBPMaster` gRPC service
(the same one `MasterClient` uses) — not a proto service of its own:

- `Register(worker_node_id, worker_node_address, peer_address, engine_desc, tags)`
  — sends `RegisterClientRequest` with empty `tier_capacities` and the
  given `peer_address`/`engine_desc`. Constructed the moment a
  worker's `RegisterMemoryRequest` carries a non-empty
  `worker_node_id`.
- `Heartbeat()` — periodic, liveness-only. Sends `HeartbeatRequest`
  with empty `tier_capacities` and empty `bundles` on every call; its
  only job is keeping the `ClientRegistry` entry from expiring so
  `MatchExternalKv` continues to resolve this `node_id`.
- `Unregister()` — sends `UnregisterClientRequest`. Called on (a) the
  corresponding worker's `DeregisterMemory` RPC, and (b) server
  shutdown.
- `ReportExternalKvBlocks`/`RevokeExternalKvBlocks`/
  `RevokeAllExternalKvBlocksAtTier`/`MatchExternalKv`/
  `GetExternalKvHitCounts` — thin forwarding wrappers to the
  corresponding `UMBPMaster` RPCs, using this sub-identity's own
  `node_id`.

Access to a per-`client_id` `ExternalKvIdentityClient` is guarded by
the same locking discipline used elsewhere in `standalone_server.cpp`,
since each sub-identity is a stateful object with its own background
heartbeat thread.

Under a `StandaloneClient` backend, external-KV behavior is unchanged
from the local-backend shape: hardcoded stubs, matching
`StandaloneClient`'s own no-op external-KV methods.

### 7.6 Known limitations

**No crash/disconnect detection.** The only paths that tear down an
`ExternalKvIdentityClient` are `DeregisterMemory` (explicit,
worker-initiated) and full server shutdown. If a worker process
crashes or is killed without calling `DeregisterMemory`, its
`ExternalKvIdentityClient`'s heartbeat thread (which lives in the
server process, unaffected by the worker's death) keeps running
indefinitely — the Master's `ClientRegistry` entry for that `node_id`
never expires, and `MatchExternalKv`/`GetExternalKvHitCounts` keep
returning it for KV blocks that may no longer exist. This is not
cleaned up until the entire `umbp_standalone_server` process is
restarted. Fixing this needs a general connection-liveness/crash-detection
mechanism — the same kind of work needed for
`StandaloneProcessClient`'s own reconnect handling (§8, item 1) — and
is a deliberately deferred, tracked limitation.

**Lifecycle lock serializes register/deregister across workers.**
Register, deregister, and shutdown bookkeeping for
`ExternalKvIdentityClient` are serialized by one server-wide lifecycle
lock, closing register-vs-deregister races for the same `client_id`.
This lock can be held across blocking Master RPCs
(`RegisterClient`/`UnregisterClient`, bounded by the RPC deadline), so
when the Master is slow or unavailable, otherwise-unrelated workers
can queue behind each other during `RegisterMemory`/`DeregisterMemory`
— worst case, startup/shutdown delay approaches `N * rpc_deadline` for
the stated 8-worker shape. This is an accepted trade-off at the
current scale (a lifecycle path, not the Put/Get hot path). If future
deployments need more workers, more frequent reconnects, or hit
startup stalls from slow Master RPCs, revisit with per-`client_id`
lifecycle locks or by decoupling external-KV identity registration
from core memory registration via an async/retry path.

## 8. Known Limitations and Open Items

1. **Crash/restart semantics.** No supervised-restart or
   reconnect-with-cache-rehydration story exists — matches every
   reference system reviewed (Mooncake, LMCache, UMBP's own SPDK-proxy
   daemon). If the standalone server dies, every attached worker loses
   its DRAM cache simultaneously. Workers must treat a broken RPC
   channel as "cache unavailable," not fatal; this needs a connection
   state machine (`CONNECTED` → `DISCONNECTED` → optionally
   `RECONNECTING`) in `StandaloneProcessClient` that does not exist
   today.
2. **Per-tenant DRAM quota** is not implemented in
   `LocalStorageManager`/`DRAMTier` — no notion of "owner" per key
   exists, so multiple workers sharing one server's DRAM tier can
   starve each other.
3. **fd-handoff listener lifetime**: long-lived for the server
   process's whole lifetime (matches Mooncake), chosen over a
   short-lived per-registration-window socket for simplicity, relying
   on filesystem permissions (item 4) rather than a narrower open
   window for hardening.
4. **Security/isolation**: the shm segment (`memfd`, `O_CLOEXEC`
   recommended) and both UDS socket files must be created with
   `0600`/owner-only permissions, or another local user could attach
   to a different tenant's KV cache. Neither Mooncake's nor UMBP's
   SPDK-proxy code sets restrictive permissions on their sockets/shm
   by default — this must be added deliberately.
5. **gRPC-over-UDS batch overhead**: a gRPC call costs a syscall +
   protobuf encode/decode per `BatchGet`/`BatchPut` even with
   struct-of-arrays batching, unlike a busy-polled shm ring. Acceptable
   for v1; if profiling later shows this is a bottleneck for very
   small, very frequent `Get` calls, a future iteration could add an
   optional shm-ring fast path for single-key Get/Put alongside gRPC.
6. **Process-local ptr→fd registry** (`HostMemAllocator::Alloc`/`Free`)
   must be safe for `Free` to run concurrently with a `RegisterMemory`
   lookup from a different thread, and `Free`/`DeregisterMemory`
   ordering must be defined — `Free` should refuse (or at least warn
   loudly) if an active registration still references that pointer,
   rather than silently leaving the server with a stale mapping.
7. **`tcp_staging` transport** is deferred; not built unless a
   concrete deployment needs it (§2).
8. **Version/ABI skew**: the standalone server binary and the
   `mori_pybinds`-linked proto definitions in each worker process must
   agree on the `UMBPStandalone` proto schema — the same constraint
   `distributed/` already has between `umbp_master` and workers, but
   more likely in practice here since a standalone server is typically
   a longer-lived sidecar that outlives several worker restarts/
   upgrades. No solution beyond the existing binary/version-pinning
   discipline.
9. **No fd-handoff registration reaper**: if a worker sends its fd via
   the raw fd-handoff socket and then crashes before issuing the
   confirming gRPC `RegisterMemory` RPC, the mapping stays live in the
   server's registry indefinitely — pre-existing, orthogonal to which
   backend `client_` is.
10. **Distributed-backend test coverage**: the existing
    `test_standalone_shm_ipc.cpp` suite only exercises the
    `StandaloneClient`-backed path. A `DistributedClient`-backed test
    needs a running `umbp_master` plus the RDMA/IO-engine stack, likely
    gated separately (e.g. only where RDMA hardware/loopback is
    available). Should also cover per-worker external-KV sub-identity
    registration/dispatch with 2+ connected workers, and the
    `RegisterMemory`/`DeregisterMemory` rollback/shutdown-ordering
    paths (§6.3).
11. **Naming convention**: whether to formally distinguish
    `standalone-process/distributed-backend` vs
    `standalone-process/local-backend` in docs, config, and log output
    (recommended, not yet adopted in code/env-var naming).

## 9. Compatibility with Existing Modes

- `local/` (in-process `StandaloneClient`) is unchanged and remains
  the default when neither `distributed` nor `standalone_process` is
  set.
- `distributed/` (cross-node master+peers+RDMA) is unchanged for
  ordinary workers; this design does not touch any file under
  `distributed/master/`, `distributed/peer/`, or `distributed/routing/`.
- Shared code touched: `common/config.h` (new `standalone_process`
  optional field on `UMBPConfig`), `umbp_client_factory.cpp` (new
  branch), `local/host_mem_allocator.*` (new `kAnonymousShm` backing
  plus the fd registry, additive), `local/tiers/copy_pipeline.*` (new
  `Drain()`), `pybind_umbp.cpp` (new `AnonymousShm` enum value and
  `standalone_process`/`UMBPStandaloneProcessConfig` bindings), and
  `umbp_store.py`'s `register_mem_pool_host` gate and error-handling
  branches (§10).
- Future extension, out of scope here: the standalone server could
  itself become a `distributed/` peer/leader (a host running one
  standalone server that both serves local workers over UDS and
  participates in the cross-node distributed pool directly) —
  `LocalStorageManager`'s existing `SharedSSDLeader`/`SharedSSDFollower`
  roles hint at this kind of layering.

## 10. Python / Pybind Interface

- `get_deployment_mode()` (enum `Local`/`StandaloneProcess`/
  `Distributed`) is added to `IUMBPClient`/pybind. `is_distributed()`
  remains a pure bool getter, unchanged, continuing to mean "true
  cross-node distributed" for existing external-KV branch points.
- `CreateUMBPClient(const UMBPConfig&)` gains a third branch:
  `config.standalone_process.has_value()` (new optional field on
  `UMBPConfig`, alongside the existing `distributed`) constructs
  `StandaloneProcessClient`. `distributed` and `standalone_process`
  are mutually exclusive within one `UMBPConfig`;
  `UMBPConfig::Validate()` rejects setting both.
- **Activation is `UMBP_STANDALONE_ADDRESS`-only.**
  `extra_config["standalone_address"]` without the env var raises at
  `UMBPStore.__init__`. This is required because the host memory pool
  (`MHATokenToKVPoolHost`) is constructed and allocates its `kv_buffer`
  inside `HiRadixCache.__init__`, before `storage_backend_extra_config`
  is parsed and handed to `UMBPStore` — `UMBPHostTensorAllocator` can
  only see process environment variables at the point it must decide a
  buffer's backing, so `extra_config` alone cannot reliably activate
  this mode. `auto_start`/`startup_timeout_ms` (which only affect what
  `CreateUMBPClient` does once the config is already built) may still
  come from either `extra_config` or their env vars, since they carry
  no allocator-timing constraint.
- `umbp_store.py`'s `register_mem_pool_host` gate is changed from `if
  not is_distributed(): return` to also allow `StandaloneProcess` —
  otherwise a correctly-constructed `StandaloneProcessClient` (which
  reports `is_distributed() == False`) would never have its host KV
  buffer registered, and the fd-handoff handshake (§4.2) would never
  trigger.
- Under `get_deployment_mode() == StandaloneProcess`, all three of
  `register_mem_pool_host`'s existing failure paths — a set
  `disable_zero_copy_register`, a `register_memory` exception, and
  `register_memory() == False` — raise instead of warn-and-return.
  This differs from `distributed`'s behavior (unchanged: warn and fall
  back to the staging-buffer path), because standalone-process mode
  has no staging/inline fallback to degrade to: a swallowed
  registration failure would otherwise mean every later `Put`/`Get`
  fails later, with a less correlated-looking error.
- `pybind_umbp.cpp` additions: `AnonymousShm`/`kAnonymousShm` value on
  `UMBPHostBufferBacking`; a new `standalone_process` property and
  `UMBPStandaloneProcessConfig` pybind class on `UMBPConfig`.
- `umbp_host_allocator.py`: `UMBPHostTensorAllocator.__init__` reads
  `UMBP_STANDALONE_ADDRESS` directly from `os.environ` (not via
  `UMBPConfig`/`extra_config`, neither of which exists yet at this
  point in the process) and requests the `AnonymousShm` backing when
  set.

## 11. Configuration Surface

`UMBPConfig` gains `std::optional<UMBPStandaloneProcessConfig>
standalone_process`, with exactly three fields, all Python-set
(mirroring `UMBPDistributedConfig`'s existing convention): `address`,
`auto_start`, `startup_timeout_ms`.

| Var | Consumed by | Purpose | Default |
|---|---|---|---|
| `UMBP_STANDALONE_ADDRESS` | Python, into `cfg.standalone_process.address` | UDS path of the standalone server's gRPC socket. Presence enables standalone-process mode. | unset (disabled) |
| `UMBP_STANDALONE_AUTO_START` | Python, into `cfg.standalone_process.auto_start` | If `true` and no server found at `address`, `CreateUMBPClient` forks+execs `umbp_standalone_server` (rank-0-local only, local-backend only, §5.2). | `false` |
| `UMBP_STANDALONE_STARTUP_TIMEOUT_MS` | Python, into `cfg.standalone_process.startup_timeout_ms` | Bound on waiting for readiness after spawn. | `30000` |
| `UMBP_STANDALONE_BIN` | Deployment-only, read directly by the auto-start code at spawn time | Path override for the `umbp_standalone_server` binary. | resolved via `PATH`/build dir |
| `UMBP_STANDALONE_IDLE_EXIT_TIMEOUT_MS` | `umbp_standalone_server`'s own env read at startup | `0` = never self-exit (default). Non-zero opts into SPDK-proxy-style idle exit. | `0` |
| `UMBP_STANDALONE_GRPC_SHUTDOWN_DEADLINE_SEC` | `umbp_standalone_server`'s own env read | Reuses the `MasterServer` shutdown-deadline convention. | same as `UMBP_GRPC_SHUTDOWN_DEADLINE_SEC` |
| `UMBP_STANDALONE_SHM_DIR` | Both sides' own env read (deployment-only) | Directory for the bootstrap-lock file (not the shm segment itself, which is anonymous via `memfd_create`). | `/tmp` |

Server-side distributed-backend variables are listed in §6.6.

`fd_socket` and `transport` are deliberately not config fields: the
fd-handoff socket path is always derived from `address` (§5.1), and
`transport` (`uds` vs. the deferred `tcp_staging`) has no v1 surface
since `tcp_staging` isn't built.

### 11.1 DRAM tier capacity: ownership moves from per-rank to the server

The UMBP DRAM cache tier is sized by `UMBP_DRAM_CAPACITY`
(`UMBPDramConfig::capacity_bytes`, read by `UMBPConfig::FromEnvironment()`
in `common/config.h`; also settable in Python via
`extra_config["dram_capacity_bytes"]`). This tier is memory the backend
allocates for itself, distinct from SGLang's host KV pool
(the `umbp_host_allocator`-backed buffer that workers register for
zero-copy transfer, §4.2): `Put` copies bytes *from* the registered
host buffer *into* this tier.

Where this tier lives, and therefore which process's `UMBP_DRAM_CAPACITY`
sizes it, differs by mode:

- **`local/` (in-process)**: each worker constructs its own
  `LocalStorageManager` and its own `DRAMTier`, each sized by that
  worker process's `UMBP_DRAM_CAPACITY`. N workers on a host produce N
  independent tiers.
- **Standalone-process**: `StandaloneProcessClient` allocates no tier —
  it only forwards RPCs (`standalone_process_client.{h,cpp}`). The
  single DRAM tier lives in `umbp_standalone_server` and is shared by
  all connected workers (§5.4, §5.5). It is sized by the **server
  process's** `UMBP_DRAM_CAPACITY`
  (`standalone_server_main.cpp` builds the backend config via
  `UMBPConfig::FromEnvironment()`). A worker's own `UMBP_DRAM_CAPACITY`
  no longer sizes a local tier.

The two launch paths (§5.1, §5.2) determine where the server reads that
value from:

- **Externally launched (§5.1; also the only supported path for the
  distributed backend)**: the server reads `UMBP_DRAM_CAPACITY` from its
  own launch environment. Worker-side values are not consulted for tier
  sizing.
- **Auto-start (§5.2; local-backend only)**: the bootstrap-winning
  worker (local rank 0) forks the server and exports its own
  `cfg.dram.capacity_bytes` into the child's environment as
  `UMBP_DRAM_CAPACITY` (`standalone_process_client.cpp`,
  `ExportServerEnv`). The server's tier is therefore sized from that one
  worker's value, not an aggregate of all workers'.

For the distributed backend, `UMBP_DRAM_CAPACITY` sizes the
`DistributedClient`/`PoolClient` DRAM pool
(`distributed/distributed_client.cpp`) rather than a
`LocalStorageManager` tier; the staging buffer is sized separately by
`UMBP_DISTRIBUTED_STAGING_BUFFER_SIZE` (§6.6).

## 12. Source References

- UMBP: `umbp_client_factory.cpp`, `include/umbp/umbp_client.h`,
  `include/umbp/common/config.h`, `local/standalone_client.{h,cpp}`,
  `local/host_mem_allocator.{h,cpp}`, `local/tiers/dram_tier.cpp`,
  `local/tiers/local_storage_manager.cpp`, `local/tiers/copy_pipeline.{h,cpp}`,
  `storage/spdk/proxy/spdk_proxy_protocol.h`, `storage/spdk/proxy/spdk_proxy_shm.cpp`,
  `distributed/peer/batch_resolve_codec.h`,
  `distributed/master/master_server.cpp`, `distributed/master/master_client.cpp`,
  `distributed/master/client_registry.cpp`, `distributed/master/eviction_manager.cpp`,
  `distributed/routing/router.cpp`, `distributed/routing/route_put_strategy.cpp`,
  `distributed/distributed_client.{h,cpp}`, `distributed/pool_client.{h,cpp}`,
  `distributed/proto/umbp.proto`, `distributed/proto/umbp_standalone.proto`,
  `standalone/standalone_server.{h,cpp}`, `standalone/bin/standalone_server_main.cpp`,
  `src/pybind/pybind_umbp.cpp`, `src/pybind/CMakeLists.txt`,
  `doc/runtime-env-vars.md`, `scripts/run_umbp_single_node_hicache.sh`.
- SGLang: `sglang/python/sglang/srt/mem_cache/storage/umbp/umbp_store.py`,
  `umbp_host_allocator.py`.
- Mooncake (`/apps/nima/KVManager/Mooncake`): `mooncake-store/src/real_client.cpp`,
  `real_client_main.cpp`, `shm_helper.cpp`, `dummy_client.cpp`, `client_buffer.cpp`,
  `uds_transport.cpp`, `docs/source/design/mooncake-store.md`,
  `docs/source/getting_started/examples/sglang-integration/hicache-integration-v1.md`.
- LMCache (`/apps/nima/KVManager/LMCache`, reviewed for comparison only):
  `lmcache/v1/multiprocess/{mq,server,protocol,custom_types,futures,config}.py`,
  `lmcache/v1/mp_observability/`.
