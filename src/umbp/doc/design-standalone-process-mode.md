# UMBP Standalone Process Mode — Design (Draft v1.4)

**v1.0 revision note:** §11.4.4 (external-KV) went through two rounds
of direct clarification and was substantially corrected, not just
touched up — worth reading in full even if earlier revisions of this
section were already reviewed. Summary of the arc: v0.7 treated
external-KV as a data-plane feature and concluded the server can't
physically serve HBM-tier reads, so it decided to disable external-KV
entirely for a distributed-backed server. That premise was wrong:
external-KV is a pure query/registry service (`Report`/
`MatchExternalKv` just record and answer "who holds hash H," a caller
— typically a custom SGLang router — uses the answer to route a *new*
request elsewhere; UMBP itself never moves the reported bytes, in
`distributed/` mode or here). With that corrected, the real question
became identity granularity, and a second clarification established
that the project's actual deployment shape is 8-GPU hosts running
SGLang TP=8, **and** that the *baseline* non-standalone implementation
already supports multiple independent replicas sharing one host for
free (`node_id` is built from per-process rank coordinates with no
awareness of logical TP/DP group boundaries, `umbp_store.py:512-519`).
Per explicit instruction to match the baseline rather than settle for
a narrower capability, §11.4.4 now specifies **per-worker distributed
sub-identities** (the server runs one independent, external-KV-only
identity per connected worker, sharing one physical `peer_address` —
verified this pass that `peer_address` uniqueness is not required
anywhere in `ClientRegistry`/`MasterServer` — see §11.4.4 for
citations) rather than either disabling the feature or collapsing all
workers into one shared identity. **(Corrected in v1.2: this is
implemented by a new, purpose-built `ExternalKvIdentityClient` class —
not `MasterClient` — specifically so each sub-identity reports empty
`tier_capacities`, publishes no owned-KV events, and is therefore
structurally ineligible for ordinary routing/eviction; see §11.4.4 and
§11.4.7 for the full reasoning and interface. The paragraph above
originally said "`MasterClient` registration," which is no longer
accurate — kept here only so the revision history stays honest about
what changed and why.)** This is real, non-trivial, correctly-scoped
implementation work (up to N `ExternalKvIdentityClient`
registrations/heartbeats inside one process), not a simple flag flip —
it was **not** in the v0.7/v0.9 implementation-cost accounting and
should be weighed accordingly when this proposal is scheduled.

**v0.9 revision note:** a fourth review pass raised two further
points, both accepted as valid (not over-engineering): (1)
`PingResponse.deployment_mode` was wrongly filed as a low-stakes,
deferrable nice-to-have — it is actually the only guard against the
single most dangerous silent-failure mode this whole section exists to
prevent (a parity-intended deployment silently ending up local-backed,
with every RPC still succeeding); promoted to required, with a new
assertion requirement for distributed-backend launch/test tooling. (2)
§11.4.6's env-var list for the server's own distributed backend config
was incomplete (listed 6 vars, the real `umbp_store.py` surface has ~12
including SSD staging and cache-admission knobs) and, separately,
missed that `dram_page_size`'s normal auto-derivation depends on a
`mem_pool_host` the server never has — both fixed, with a proposed
`BuildDistributedBackendConfigFromEnv()` helper shape and an explicit
required/optional split. See inline `(v0.9 fix)` markers in §11.4.5 and
§11.4.6.

**v0.8 correction pass (citations only, no design changes):** every
Mooncake source citation in this document was originally verified
against `/apps/theresa/Mooncake` (commit `e1d6d6f6`, 2026-04-22). That
was the wrong path — the correct, current Mooncake checkout for this
project is `/apps/nima/KVManager/Mooncake` (commit `c9896684`,
2026-07-08, 75 commits ahead on the relevant files). All citations
were re-verified against the correct path this pass. Outcomes:
- Most citations were **directionally correct but line-stale**
  (functions renamed/refactored, e.g. `map_shm_internal` is now a thin
  wrapper around `map_shm_internal_with_device`; `ipc_send_fd`/
  `ipc_recv_fd` were renamed to `UdsConnection::sendFd`/`recvFd` and
  moved into `uds_transport.cpp`) — corrected in place, marked
  `(v0.7 citation fix)` inline (the fixes were made while the doc was
  still at v0.7; the version bump to v0.8 here just reflects the
  re-verification pass being complete).
- **One substantive error was found and corrected, not just a stale
  line number**: this document had described Mooncake's primary shm
  buffer as "the Real Client allocates it and hands the fd to the
  application." The correct direction, confirmed against the current
  checkout, is the opposite for the primary buffer: the **application**
  (`DummyClient`) allocates and owns it, and sends the fd to the Real
  Client, which is the passive second mapper. (A secondary, read-only
  "hot cache" segment does go the other direction, but that is not
  the buffer this design's citations were describing.) Reassuringly,
  **UMBP's own already-shipped implementation never had this error** —
  §3.2 already correctly has the worker (application) allocate and own
  the buffer, matching Mooncake's real direction; only this document's
  *prose describing* Mooncake's mechanism had drifted from the
  primary-source direction. No code changes result from this
  correction, only doc text (§11.3's "closed open item" section).
- The auto-start precedent claim ("zero matches" for fork/exec/spawn
  patterns across `mooncake-integration/` and `mooncake-wheel/`) was
  also corrected: `mooncake-integration/` (the actual application-
  binding code) is still zero matches, but `mooncake-wheel/` (the
  separate CLI/packaging layer) contains one real match — an explicit,
  human-invoked CLI passthrough, not an implicit auto-start. The
  underlying design decision (§11.4.6: distributed-backed
  `umbp_standalone_server` supports external launch only, no
  auto-start) is unaffected; only the supporting claim's precision was
  corrected.

**v1.1 revision note:** confirmed §11.4.4's per-worker `MasterClient`
sub-identity design (Option A) as final, after clarifying it involves
no worker→server→Master relay (each sub-identity heartbeats Master
directly from inside the server) and after explicitly weighing it
against a simpler Mooncake-aligned alternative (Option B: one shared
identity + `tenant_id`-style key namespacing, the pattern Mooncake
actually uses for its own multi-`DummyClient`-per-`RealClient` case).
Option B was rejected: preserving the `distributed/` baseline's
existing per-worker routing precision (verified as a real, already-
existing capability, not a nice-to-have) was judged more valuable than
the simplicity Option A gives up — accepted as genuinely new
engineering with no direct precedent in either UMBP or Mooncake, not a
known-safe pattern being reused. See the `(v1.1 —` marker in §11.4.4.

**v1.2 revision note:** the v1.1 per-worker sub-identity design was
found to have three confirmed blockers, all fixed by replacing "N
independent `MasterClient` registrations" with a new, purpose-built
`ExternalKvIdentityClient` class: (A) a naive `MasterClient`-based
sub-identity with nonzero `tier_capacities` would have been eligible
for ordinary `RoutePut` target selection, silently polluting real KV
placement with fake storage nodes — closed structurally by giving the
new class no `tier_capacities`-reporting code path at all, rather than
relying on every future call site remembering to pass empty capacities
into `MasterClient`. (B) `MasterClient`'s existing re-register path has
a confirmed bug — it omits `peer_address`/`engine_desc` on re-register,
silently breaking `MatchExternalKv` responses after a Master
restart/reaper-expiry — closed by not reusing that code path at all;
the new class always includes both fields, by construction, rather
than patching a class shared by every real distributed peer. (C) the
registration-handshake field list was under-specified ("just
`node_id`") when the proto actually requires `node_address` too —
decided now: `worker_node_id` + `worker_node_address` + optional
`tags`, Python remains the single source of truth for deriving them,
no raw rank components cross the boundary. `ExternalKvIdentityClient`
itself is flagged as real, net-new implementation work not previously
accounted for in this proposal's cost estimate (§11.7 item 7).

**v1.3 revision note:** a documentation-consistency and finalization
pass, no architecture changes from v1.2. Six items closed: (1) the
v1.0 revision note at the top of this document still said "`MasterClient`
registration per connected worker," contradicting §11.4.4's v1.2
`ExternalKvIdentityClient` fix — corrected in place, kept visible
rather than silently rewritten. (2) `ExternalKvIdentityClient` was
still filed as an open item needing "a design pass" even though
§11.4.4 already made it a required component — it now has a full
interface/lifecycle spec in new §11.4.7 (Register/Heartbeat/Unregister/
the five external-KV RPCs, the re-registration hard requirement, and
shutdown ordering). (3) confirmed via `eviction_manager.cpp:83-91`
(checked directly, not assumed) that the same empty-`tier_capacities`
invariant that hides a sub-identity from `RoutePut` also hides it from
`EvictionManager` — no separate mechanism needed, but stated
explicitly now rather than left to be inferred; also explicitly
decided `ExternalKvIdentityClient` heartbeats must never carry
`EventBundle`s, a separate `HeartbeatRequest` field from
`tier_capacities`. (4) the worker→server identity handshake now has a
concrete wire location: `worker_node_id`/`worker_node_address`/`tags`
are added to the existing `RegisterMemoryRequest` message
(`umbp_standalone.proto:87-91`), not a new RPC. (5) §11.4.6's five
new server-backend env vars are finalized as
`UMBP_DISTRIBUTED_STAGING_BUFFER_SIZE` and siblings (the
`UMBP_DISTRIBUTED_` prefix specifically because these five have no
existing worker-side env-var form to collide with, unlike the ones
reusing worker-side names verbatim) — no longer "proposed." (6) added
an explicit layering statement (new §11.4.7 lead-in): distributed-
backed server + UDS/shm fd handoff is the Mooncake-parity core;
`ExternalKvIdentityClient`/per-worker external-KV identity is a
separate UMBP-compatibility extension preserving a `distributed/`-
baseline capability, not required for declaring parity achieved, and
schedulable/testable independently.

**v1.4 revision note:** two updates from the first implementation +
code-review round on `ExternalKvIdentityClient`. (1) A concurrency bug
found by review — `EnsureExternalIdentity`/`RemoveExternalIdentity` had
no serialization across their read-check/stop-old/construct-and-start-
new/insert sequence, so a `DeregisterMemory` racing an in-flight
`RegisterMemory` for the same `client_id` could silently fail to
remove an identity the register path was about to (re-)insert (a
permanent leak until full server restart), and two concurrent
`RegisterMemory` calls for the same `client_id` could briefly
double-register — was fixed with a single server-wide
`external_identity_lifecycle_mu_` serializing all three call sites;
§11.4.7 now documents this as an accepted trade-off (serializes
otherwise-unrelated workers' register/deregister calls when the lock
covers a slow/unavailable Master RPC, worst case approaching
`N * rpc_deadline` for the stated 8-worker deployment shape) rather
than a free fix, with an explicit note on when to revisit it
(per-`client_id` sharding, or decoupling external-KV registration from
the core memory-registration path). (2) §11.4.7's `Unregister()`
trigger list previously claimed worker crash/disconnect detection was
"already" handled by mirroring a `StandaloneProcessClient` state
machine — but §8 item 1 itself explicitly says that state machine
doesn't exist. This was an internal contradiction, not just a missing
callout: corrected to state plainly that no crash detection exists for
`ExternalKvIdentityClient` today, what the concrete consequence is (a
crashed worker's identity heartbeats forever, `ClientRegistry` never
expires it, only a full server restart clears it), and that this is a
deliberately deferred, tracked limitation rather than a silent gap.

**Status (still true as of v1.4): §11 below is a NEW, UNAPPROVED
proposal. Nothing in §11 is implemented. Do not start coding against
it until it has been discussed — this revision exists to make the
proposal concrete enough to discuss, per explicit instruction not to
touch code first. v0.7 closed 4 of the 5 blockers a second review
pass found in the v0.6 draft (config/auto-start conflict,
external-KV routing correctness, backend registration
transaction/rollback + shutdown ordering, and the cross-mapping
memory-visibility open item); one naming/scope-framing question
remains genuinely open (§11.7 item 1).** §1-§10
describe what is already implemented and shipped (v0.1-v0.5); §11
revises the framing in §1 (see the callout there) to close a scope gap
identified after implementation: v0.1-v0.5 built a same-host-only
subsystem, whereas the stated goal was Mooncake-parity, and Mooncake's
"standalone" Real Client is cross-node-capable (it is a full
distributed client that happens to run in its own process, not a
narrower local-only subsystem). §11 proposes closing that gap without
discarding the v0.1-v0.5 implementation.

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

**(v0.6 note — this section's framing is revised by §11, not replaced.
Read this section for why v0.1-v0.5 built what they built; read §11
for why that scope turned out to be narrower than "Mooncake parity"
requires, and how the two are reconciled without a rewrite.)**

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
  `shm_addr_offset` translation **(v0.7 citation fix — re-verified
  against the correct checkout; function confirmed still present,
  only line numbers moved from a stale reference)**:
  `real_client.cpp:369-379` (`map_dummy_buffer_range_to_real`), called
  from the batch RPC handlers `get_into_range_shm_helper`/
  `get_into_ranges_shm_helper` (`real_client.cpp:4291`/`:4316`), just
  carried over gRPC instead of coro_rpc.

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
   `setup_dummy()` → `shm_helper_->allocate(...)` **(v0.7 citation fix
   — re-verified against the correct Mooncake checkout,
   `/apps/nima/KVManager/Mooncake`, not the stale one earlier research
   in this doc used; line numbers below are corrected accordingly, the
   underlying claim is unchanged and was directionally correct all
   along)**: `dummy_client.cpp:469`, inside `setup_dummy()`
   (`dummy_client.cpp:430-509`). Concretely: `UMBPHostMemAllocator`
   (`host_mem_allocator.h/.cpp`, already used by
   `umbp_host_allocator.py` to back SGLang's host KV pool) gains a new
   backing kind, `kAnonymousShm`, using `memfd_create` +
   `ftruncate` + `mmap(MAP_SHARED)` — the same primitives Mooncake's
   `ShmHelper::allocate()` uses (`shm_helper.cpp:74-140`:
   `memfd_create_wrapper` at `:26-32`, `memfd_create` at `:108`,
   `ftruncate` at `:117`, `mmap(MAP_SHARED|MAP_POPULATE)` at
   `:123-124`), chosen over
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
     mirroring Mooncake's fd-passing and registration handshake
     **(v0.7 citation fix — the function names `ipc_send_fd`/
     `ipc_recv_fd` no longer exist in the current Mooncake checkout;
     the mechanism was renamed/relocated, not removed)**: fd send/recv
     is now `UdsConnection::sendFd`/`UdsConnection::recvFd`
     (`uds_transport.cpp:167`/`:196`), and the registration entry point
     is `RealClient::handle_ipc_shm_register`
     (`real_client.cpp:5311`, dispatched from the `IPC_SHM_REGISTER`
     case at `real_client.cpp:5286-5287`). On a miss (the pointer isn't
     backed by a `kAnonymousShm` allocation — e.g. standalone-process
     mode was configured but the host allocator wasn't switched to the
     shm backing), `RegisterMemory` must fail loudly rather than
     silently falling back to a broken pointer-based `Put`/`Get`.
   - The server `mmap`s the received fd read-write into its own
     address space (`real_client.cpp:2136-2138`, inside
     `map_shm_internal_with_device`) and stores `client_id → {base,
     size}` in a registry keyed by `client_id` (mirrors
     `shm_contexts_[client_id]`, e.g. `real_client.cpp:2158` — the map
     itself is unchanged, only specific usage line numbers moved
     across the file's many call sites).
3. **Every subsequent Put/Get/BatchPut/BatchGet RPC carries only
   `(client_id, shm_offset, size)`** — `StandaloneProcessClient`
   computes `shm_offset = ptr - registered_base` locally before
   issuing the RPC; the server computes `real_addr = base + shm_offset`
   and reads/writes there directly (mirrors
   `RealClient::map_dummy_buffer_range_to_real`, now at
   `real_client.cpp:369-379`, called from the batch RPC surface at
   `get_into_range_shm_helper`/`get_into_ranges_shm_helper`,
   `real_client.cpp:4291`/`:4316` — **(v0.7 citation fix: function
   still exists exactly as described, only line numbers moved; the
   RPC-handler line numbers were re-confirmed this pass, unlike some
   other citations in this section which were only spot-checked)**).
   No KV bytes ever cross the gRPC channel; the gRPC message is a
   handful of integers per key.
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

---

## 11. Cross-Node Extension: making `umbp_standalone_server` optionally `DistributedClient`-backed (v0.7 proposal, UNAPPROVED)

**v0.7 revision note:** a second review pass found 5 concrete gaps in
the v0.6 draft of this section — a config conflict with the already-
shipped v0.5 activation rule, an external-KV routing-correctness hole,
missing transaction/rollback semantics for backend memory
registration, an open correctness question left open instead of
closed, and an overstated "no logic change" claim. All five are
addressed below with inline `(v0.7 fix)` markers. **Guiding principle
adopted for this revision and going forward: where a design question
has a direct Mooncake precedent, resolve it by matching Mooncake,
rather than inventing a new answer** — per explicit direction ("我们与
Mooncake 保持对标就好,类似的问题都一个思路"). Two of the five fixes
below were resolved this way, with the precedent re-verified against
the local Mooncake checkout (not recalled from memory):

- **No *implicit* auto-start anywhere in Mooncake's application-side
  binding code.** **(v0.7 citation fix, re-checked a second time
  against the correct checkout `/apps/nima/KVManager/Mooncake` after
  discovering the first verification pass had used a stale checkout at
  a different path — the precise claim below is narrower than what was
  first written, because the first, broader "zero matches across both
  directories" claim turned out to be literally false.)** Re-checked
  directly: `grep -rln "fork(|execvp|execve|posix_spawn|subprocess\.|Popen" mooncake-integration/`
  returns zero matches — the `DummyClient` application-binding code
  never spawns anything. `mooncake-wheel/` (the packaging/CLI layer,
  not the application-binding layer) does contain one real match,
  `mooncake-wheel/mooncake/cli_client.py:24`
  (`subprocess.call([bin_path] + sys.argv[1:])`), but this is the
  pass-through implementation of the explicitly-invoked `mooncake_client`
  console-script command — functionally equivalent to a human running
  the `mooncake_client` binary directly from a shell, not an implicit
  auto-start triggered by normal `DummyClient`/integration-API usage.
  (The other `mooncake-wheel/` hits — `cli.py`, `cli_bench.py`,
  `setup.py`, `tests/test_cli.py` — wrap unrelated binaries, are
  build-time only, or are an explicit human-run test harness,
  respectively.) The Real Client is always started by something
  outside the application (deployment system, operator, systemd unit,
  or an explicit CLI invocation) — never implicitly, by any code path
  a normal application integration would exercise. §11.4.6 below
  adopts this exactly.
- **Mooncake's Real Client already does the "mmap the app's shm, then
  register that same mapping with the distributed transport" pattern
  this section proposes** **(v0.7 citation fix — line numbers below
  corrected against the correct checkout; the function has been
  renamed/refactored since the stale citation was written, but the
  mechanism is unchanged)** — `real_client.cpp:2136-2137`
  (the `mmap(nullptr, shm_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0)`
  call) and `real_client.cpp:2161-2162`
  (`client_->RegisterLocalMemory(shm.shm_buffer, shm_size, ...)`,
  munmap'd on failure), both inside
  `RealClient::map_shm_internal_with_device`
  (`real_client.cpp:2097-2186`), which the now-thin
  `RealClient::map_shm_internal` (`real_client.cpp:2089-2095`)
  delegates to. This is production code, not
  a prototype — it directly validates §11.3's core mechanism (register
  the *server's* local view of a shared mapping with the distributed
  transport) as an already-shipped pattern, not a novel one this
  design is the first to attempt. This substantially de-risks the
  "cross-mapping visibility" open item in §11.3 (see the fix there).

**Scope reframing (v0.7, requested explicitly):** §1's original prose
("Standalone-process mode is strictly same-host") describes the
history of what v0.1-v0.5 built, not the target end-state. Given the
project's actual goal is Mooncake parity, and Mooncake's "standalone"
Real Client is cross-node-capable by construction, this section's
distributed-backed shape — not the local-only shape §1 describes — is
the one that actually satisfies that goal. Going forward:
**`standalone-process/distributed-backend` (cross-node, Mooncake-parity
shape) is the primary target this feature is building toward;
`standalone-process/local-backend` (§1-§10, already shipped) is a
same-host fallback/dev-convenience shape, not a co-equal alternative.**
§1's prose is not rewritten in place (it remains an accurate
description of what v0.1-v0.5 actually built and why, and that
reasoning was sound for the narrower problem it solved), but it should
no longer be read as this feature's final scope statement — this
paragraph supersedes it for that purpose. See §11.7 item 1 for the
naming-convention decision that follows from this.

### 11.0 Why this section exists

After v0.1-v0.5 shipped, we compared what we built against what
Mooncake's Real Client actually does (source-verified earlier in this
project, see the LMCache/Mooncake research this design already cites)
and found a real divergence, not a rounding error:

- **Mooncake's Real Client**: a full distributed `Client` (Master +
  etcd + Transfer Engine/RDMA), just hosted in its own OS process.
  **(v0.7 citation fix — corrected against the correct checkout;
  construction pattern unchanged, only line numbers moved, and a
  second, structurally identical call site was added since the stale
  citation was written)** `real_client.cpp:672-675` (and duplicated at
  `real_client.cpp:708-711`, an auto-port-binding-with-retry variant of
  the same construction) constructs it via the exact same
  `Client::Create(..., master_server_addr, transfer_engine,
  {"client_mode":"real"})` an embedded client would use — "standalone"
  only changes which process the library lives in, not which protocol
  it speaks to the rest of the cluster. A cache miss can still be
  served from a remote node over RDMA.
- **What v0.1-v0.5 built**: `umbp_standalone_server` wraps a concrete
  `StandaloneClient` (`standalone_server.cpp:587`,
  `client_(config)` at `standalone_server.cpp:125`), i.e. `local/`'s
  `LocalStorageManager`/`LocalBlockIndex`/`CopyPipeline` moved into its
  own process. It has **no** connection to `distributed/` — no master,
  no RDMA, no peer protocol. A cache miss is just a miss; there is no
  "somewhere else" to check.

The UDS + `SCM_RIGHTS`-fd-handoff IPC layer between the worker and the
server (§2, §3) is *not* the divergence — that part already correctly
mirrors Mooncake's own app-process↔Real-Client IPC. The divergence is
narrower and specific: **what backs the server side of that IPC**,
local storage vs. a full distributed client.

### 11.1 Proposal, in one sentence

Make the concrete `StandaloneClient client_;` member in
`StandaloneServer::Impl` into an `IUMBPClient`-typed backend built
through the **already-existing** `CreateUMBPClient(config)` factory
(`umbp_client_factory.cpp:29-37`), and give `standalone_server_main.cpp`
a way to populate that config's `distributed` field (today it
unconditionally clears it — `standalone_server_main.cpp:43`,
`config.distributed.reset()`). Same server binary, same worker-facing
IPC protocol, same `umbp_standalone_server` deployment story — the
only thing that changes is which `IUMBPClient` implementation answers
Put/Get/etc, decided entirely by how the server operator starts it.

This is deliberately **not** a rewrite: §11.5 below shows the existing
data-path RPC handlers (Put/Get/BatchPut/BatchGet/Exists/Clear/Flush)
need no logic changes at all, because they already resolve
`(client_id, shm_offset)` to a plain pointer before ever touching
`client_` (`standalone_server.cpp:541-568`) — `client_`'s concrete
type was never load-bearing for those handlers in the first place.

### 11.2 Two independent `UMBPConfig` instances — this is the key structural point

There are, and always were, two separate configuration surfaces in
play, and they must stay separate:

1. **Worker-facing** (`standalone_process` field): what a
   `StandaloneProcessClient` inside an SGLang worker uses to find and
   talk to the server over UDS. **Unchanged by this proposal** — a
   worker never knows or cares whether the server it's talking to is
   itself distributed-backed.
2. **Server's own internal config** (`distributed` field, new):
   what `umbp_standalone_server` itself uses to decide whether *its*
   `client_` backend should be a `StandaloneClient` (today's behavior)
   or a `DistributedClient` (this proposal). This is a **second,
   independent `UMBPConfig` instance** — never the same object as #1.

**Verified this is not blocked by the existing mutual-exclusivity
check.** `UMBPConfig::Validate()` (`config.h:345-349`) rejects
`distributed.has_value() && standalone_process.has_value()` *within
one config instance* — it says nothing about two different `UMBPConfig`
objects existing in the same process. `standalone_server_main.cpp`
already builds one `UMBPConfig` for the worker-facing side (with only
`standalone_process` set); this proposal adds a second, separate
`UMBPConfig` for the server's own `client_` backend (with only
`distributed` set, when requested). Neither instance ever has both
fields set, so `Validate()`'s existing rule is satisfied unchanged —
no change needed to that check.

```
 Worker process                         umbp_standalone_server process
 ┌───────────────────────┐              ┌──────────────────────────────────┐
 │ UMBPConfig #1          │   UDS +      │ UMBPConfig #1 (mirror of the      │
 │  standalone_process =  │───SCM_RIGHTS─│  worker's, address/fd-socket)     │
 │   {address}            │              │  -> used only to open sockets     │
 │  distributed = null    │              │                                    │
 └───────────────────────┘              │ UMBPConfig #2 (server's OWN,      │
                                          │  new in this proposal)            │
                                          │  distributed = {master_address,   │
                                          │    node_id, node_address,         │
                                          │    io_engine, ...}  (if enabled)  │
                                          │  standalone_process = null        │
                                          │       │                          │
                                          │       ▼                          │
                                          │  client_ = CreateUMBPClient(#2)  │
                                          │   -> DistributedClient           │
                                          │      (Master + RDMA)             │
                                          │      ── or, if #2.distributed    │
                                          │         is unset ──              │
                                          │   -> StandaloneClient            │
                                          │      (today's local-only         │
                                          │       behavior, unchanged)        │
                                          └──────────────────────────────────┘
                                                       │  RDMA (if DistributedClient)
                                                       ▼
                                          ┌──────────────────────────────────┐
                                          │  umbp_master + other nodes'      │
                                          │  peers (the existing distributed/│
                                          │  cluster, entirely unmodified)   │
                                          └──────────────────────────────────┘
```

### 11.3 Why the zero-copy story still holds with an RDMA backend

This was the load-bearing technical question and it checks out against
the actual `PoolClient`/`DistributedClient` code (not assumed):

- RDMA memory registration (`DistributedClient::RegisterMemory` →
  `PoolClient::RegisterMemory`, `pool_client.cpp:635-651`) operates
  purely on `(ptr, size)` **in the calling process's own virtual
  address space** — it calls
  `io_engine_->RegisterMemory(ptr, size, -1, MemoryLocationType::CPU)`
  (`pool_client.cpp:648`) and caches `{base, size, mem_desc}` keyed by
  that local VA. **Nothing checks who allocated the memory or whether
  another process also maps the same physical pages**
  (`pool_client.cpp:635-675`, no ownership check anywhere).
- This means: `umbp_standalone_server`, after `mmap`-ing the worker's
  `memfd`-backed shared region via the existing fd-handoff path (§3.2,
  already built and working), can call
  `client_->RegisterMemory(server_local_va, size)` on its **own**
  local VA for that mapping. Since the server's VA and the worker's VA
  back the *same physical pages* (`MAP_SHARED` on the same `memfd`),
  registering the server's VA for RDMA registers those physical pages
  for RDMA — full stop, regardless of which process's VA was used to
  register them.
- **Get() zero-copy is conditional on this registration, and confirmed
  in the actual RDMA read path**: `WaitRemoteBatchGet`
  (`pool_client.cpp:1812-1853`) only skips the memcpy-out when the
  destination pointer was pre-registered and `FindRegisteredMemory`
  matches it (`pool_client.cpp:1661-1664`) — in that branch, the RDMA
  read from the remote peer is posted directly into that registered
  MR (`pool_client.cpp:1806-1808`) with **no memcpy** afterward. If
  the pointer wasn't registered, RDMA lands in a shared staging buffer
  and is then `memcpy`'d to the destination (`pool_client.cpp:1847`) —
  the ordinary, already-existing non-zero-copy path.
- **Net effect**: register the server's local mapping once (at
  fd-handoff time, alongside the existing mmap), and every subsequent
  `Get` that targets an offset inside that region gets true zero-copy
  RDMA — the remote write lands in memory the worker can already see,
  with no copy on the server side and no RDMA involvement on the
  worker side at all. This is *better* than Mooncake's shape in one
  respect: the worker process never touches RDMA/IB verbs directly,
  only the server does, entirely consistent with the "worker only
  knows about UDS + shm" boundary the v0.1-v0.5 design already
  established.

**(v0.7 fix — closed, was left open in v0.6.)** Cross-mapping memory
visibility/ordering: does an RDMA-completed write into the physical
pages via the server's VA become visible to a subsequent read through
the worker's *independent* VA mapping of the same `memfd`? v0.6 left
this as an unresolved research question. It is now closed by two
independent legs, not a single hand-wave:

1. **Mooncake precedent, re-verified against the correct checkout this
   pass (an earlier pass had used a stale checkout and, separately,
   mis-described the allocation direction — both corrected here).**
   Mooncake's Real Client does the *identical* dual-mapping shape in
   shipping production code, with this exact allocation/registration
   direction: the **application** (`DummyClient`) is the one that
   originally allocates the shared region, via
   `ShmHelper::allocate()` (`dummy_client.cpp:469`, inside
   `setup_dummy()`) — that allocation call itself performs the first
   `mmap` (`shm_helper.cpp:123-124`), giving `DummyClient` its own
   independent VA mapping of the physical pages from the moment of
   allocation, not by "receiving and mapping a fd" the way the server
   does. `DummyClient` then sends that fd to the Real Client
   (`sendFd`, `dummy_client.cpp:408`). The **Real Client** receives it
   (`recvFd` via `handle_ipc_shm_register`, `real_client.cpp:5311`)
   and performs the *second*, independent `mmap` of the same physical
   pages into its own address space (`real_client.cpp:2136-2137`),
   then registers that server-local mapping with its own Transfer
   Engine (`real_client.cpp:2161-2162`). This is the same shape
   `umbp_standalone_server` already implements today (worker allocates
   and owns the buffer, §3.2 step 1; server receives the fd and
   independently mmaps it, §3.2 step 2) — UMBP's already-shipped
   direction matches Mooncake's actual direction exactly; only this
   design doc's *prose describing* Mooncake's mechanism had briefly
   drifted from that (now corrected). Two independent processes, two
   independent VAs, one physical region, the *receiving* side's VA
   registered with a distributed transport, exactly this design's
   shape — running in production today. This is strong evidence the
   pattern is sound, not merely "should be fine" reasoning from first
   principles.
2. **Explicit, adopted assumption for the implementation (decision,
   not left implicit):** the gRPC `Get` response is defined as the
   happens-before edge. `umbp_standalone_server` must only send the
   gRPC response for a `Get` **after** the RDMA read completion is
   observed server-side (already the natural control flow — there is
   no reason to respond before the transfer completes). The worker
   must not read the target offset until it has received that
   response. This is not a new constraint on the implementation beyond
   what it would already naturally do; it is being stated explicitly
   here so it is a documented contract, not an accidental property of
   the current code shape.

Residual, non-blocking follow-up: add one integration test that
specifically exercises this path (RDMA `Get` into a `DistributedClient`-
backed server's mapped region, immediately followed by the worker
reading the result through its own independent mapping) once the
distributed-backend test infrastructure from §11.7 item 5 exists — this
is a good regression guard to have, not a prerequisite for the design
being considered closed.

### 11.4 What changes, concretely, mapped against the current code

Grounded in the actual current implementation (all citations verified
against source in this pass, not the earlier v0.1-v0.5 design intent):

1. **`StandaloneServer::Impl`** (`standalone_server.h/.cpp`):
   `StandaloneClient client_;` → `std::unique_ptr<IUMBPClient> client_;`,
   constructed via `CreateUMBPClient(backend_config)` instead of
   `client_(config)` directly. `backend_config` is `UMBPConfig #2`
   from §11.2, built by `standalone_server_main.cpp` (see #6 below).
2. **Put/Get/BatchPut/BatchPutWithDepth/BatchGet/Exists/BatchExists/
   BatchExistsConsecutive/Clear/Flush RPC handlers**: mechanically,
   swapping `client_` from a value to a `unique_ptr` is a
   `client_.` → `client_->` change and nothing more — they already
   call `client_.Put(...)` etc. on plain resolved pointers
   (`standalone_server.cpp:199-354`). **(v0.7 fix — "no logic change"
   overstated in v0.6; three things must be documented even though no
   code changes are required for them):**
   - **Registration gating is already correct, verified this pass, no
     new work needed**: `ResolveRange` (`standalone_server.cpp:541-549`)
     already fails (`return false`) if `client_id` isn't present in
     the `memory_` map, so `Put`/`Get`/etc. already cannot reach
     `client_` at all before `RegisterMemory` has succeeded, for
     either backend. The reviewer's concern here is already satisfied
     by existing code — noted so a future reader doesn't think it's a
     gap.
   - **Failure-policy ambiguity (backend-unavailable vs. genuine
     cache-miss both surface as `false`) is a pre-existing
     characteristic of `IUMBPClient`, not something this proposal
     introduces or worsens.** `DistributedClient::Get` already returns
     `false` uniformly for "not found" and "RPC/transport failure"
     today, for ordinary distributed-mode workers, independent of
     whether it runs inside a standalone server. This proposal
     inherits that behavior unchanged; fixing it (if ever needed) is
     an `IUMBPClient` interface question orthogonal to this section
     and explicitly out of scope here.
   - **`Flush()`/`Clear()` mean different things per backend, and this
     is an accepted semantic difference, not a bug to fix**: under
     `StandaloneClient`, `Flush()` drains `CopyPipeline` to SSD
     (§4.3); under `DistributedClient`, `Flush()`/`Clear()` operate on
     the pool client's own state (heartbeat/registration bookkeeping,
     not a local SSD tier — there is no local SSD tier in this backend
     per §11.6). Both are the correct, existing behavior of the
     respective `IUMBPClient` implementation for their backend; the
     server passing the call through unchanged is correct, and this
     should be documented as an accepted, backend-dependent semantic
     rather than left for a future reader to wonder whether it's an
     oversight.
3. **`RegisterMemory`/`DeregisterMemory` RPC handlers**
   (`standalone_server.cpp:356-375`): **currently never touch `client_`
   at all** — pure fd/mmap bookkeeping against the `memory_` map. Must
   add, after a successful `mmap`:
   `client_->RegisterMemory(server_local_ptr, size)`, and on
   deregistration: `client_->DeregisterMemory(server_local_ptr)`. This
   call is **safe to make unconditionally regardless of backend** —
   `IUMBPClient::RegisterMemory`'s default (used by `StandaloneClient`,
   which never overrides it) is already a no-op returning `true`
   (`umbp_client.h`, documented as "Standalone mode needs no
   registration (CPU-local memcpy)"). So no `IsDistributed()` branch
   is needed here — the interface was already designed for exactly
   this kind of backend-agnostic call.

   **(v0.7 fix — v0.6 said "add the call" but didn't specify the
   transaction/rollback semantics around it; this is a genuine
   correctness gap, not an implementation detail to leave for later.)**

   - **Rollback on backend registration failure.** `RegisterFd`
     (`standalone_server.cpp:504-523`, confirmed this pass) is a
     single-phase commit: on a successful `mmap`, it immediately writes
     the entry into `memory_` and returns success — there is no
     existing "pending" intermediate state to roll back from. The new
     `client_->RegisterMemory(server_local_ptr, size)` call must be
     placed **inside `RegisterFd`, immediately after `mmap` succeeds
     and before the entry is written into `memory_`**. If it fails,
     `RegisterFd` must `munmap` the just-created mapping and return
     failure **without** writing to `memory_` — i.e. backend
     registration failure is treated exactly like an `mmap` failure
     already is, using the exact same single-phase-commit shape the
     function already has. This requires no new state machine, only
     ordering the existing calls correctly.
   - **Deregistration ordering**: `DeregisterMemory`
     (`standalone_server.cpp:370-375`) currently only calls
     `UnmapClient()` (munmap + erase). Must call
     `client_->DeregisterMemory(server_local_ptr)` **before**
     `munmap`-ing (deregister the transport's view of the memory before
     the memory itself disappears — the reverse order risks the
     backend still holding a registered MR pointing at unmapped
     memory, however briefly).
   - **Shutdown ordering — `UnmapAll()` must deregister from the
     backend before unmapping, and this must happen before
     `client_->Close()`.** `UnmapAll()` (`standalone_server.cpp:533-538`)
     currently only `munmap`s every entry in `memory_`. Once `client_`
     can be a `DistributedClient` holding live RDMA registrations, this
     must become: for each registered mapping, call
     `client_->DeregisterMemory(base)`, *then* `munmap`. And this whole
     sequence must run **before** `client_->Close()` in `Shutdown()`
     (`standalone_server.cpp:154-169`, the ordering already fixed in
     the v0.1-v0.5 review pass — see the earlier "shutdown ordering"
     fix in this document's history) — deregistering after `Close()`
     would call into a backend that has already torn down its RDMA/IO
     engine state.
   - **Explicitly out of scope for this proposal, tracked separately
     (not a new problem introduced here):** if a worker sends its fd
     via the raw fd-handoff socket and then crashes before ever issuing
     the confirming gRPC `RegisterMemory` RPC, the mapping stays live in
     `memory_` indefinitely (no timeout/reaper exists for this today).
     This is a pre-existing characteristic of the v0.1-v0.5 fd-handoff
     protocol, orthogonal to which backend `client_` is — it existed
     before this proposal and isn't made worse by it. Worth a future
     hardening pass (e.g. a liveness check tied to the worker's gRPC
     channel), but not a blocker for this section.
4. **The five external-KV RPC handlers**
   (`ReportExternalKvBlocks`/`RevokeExternalKvBlocks`/
   `RevokeAllExternalKvBlocksAtTier`/`MatchExternalKv`/
   `GetExternalKvHitCounts`, `standalone_server.cpp:377-408`):
   currently hardcoded stubs (three always return `true`, two return
   empty responses) that never touch `client_` at all.

   **(v1.0 fix — superseding both the v0.7 "disable entirely" decision
   and the intermediate "naive single shared identity" idea; this is
   the third and final revision of this item, corrected after two
   rounds of clarification about what external-KV actually is and what
   the baseline non-standalone implementation actually already
   guarantees.)**

   **What external-KV actually is, clarified:** it is a pure
   query/registry service, not a data-transfer mechanism. `Report/
   RevokeExternalKvBlocks` record advisory "node N holds hash H at tier
   T" facts in the Master's `GlobalBlockIndex`; `MatchExternalKv`
   answers "which node(s) hold these hashes" for a caller (typically a
   custom SGLang router) making a *routing* decision — it sends a NEW
   request to whichever node the match points at, and that node's own
   already-resident local cache (GPU HBM or otherwise) serves it
   locally. **UMBP itself never moves the reported bytes for this
   feature, in `distributed/` mode or here** — so the earlier v0.7
   concern "the server can't physically reach the worker's HBM memory"
   does not apply: nothing ever needs to read that memory through
   UMBP for this feature. That concern was based on treating
   external-KV as a data-plane feature; it is not one.

   **What genuinely matters instead: does the reported/matched
   *identity* (`node_id`) correctly reflect the actual deployment
   topology, at the granularity a caller needs?** This is where the
   real design work is, and it is grounded in what the *baseline*,
   non-standalone implementation already does — per this project's
   standing principle of matching existing capability rather than
   inventing a narrower one:

   - **Verified against `umbp_store.py:512-519`**: a normal distributed
     worker's `node_id` is built as
     `f"{node_address}:dp{dp_rank}:pp{pp_rank}:tp{local_rank}"` — pure
     per-process rank coordinates, with **zero awareness of which
     logical TP/DP replica group the process belongs to**. This means
     the baseline already, for free, supports multiple independent
     replicas sharing one physical host (e.g. two TP=4 groups on one
     8-GPU box) — each of the 8 processes gets its own distinct
     `node_id` regardless of logical grouping, because disambiguation
     is by rank coordinate, not by group membership. This is a real,
     already-existing capability of the system this proposal must not
     silently drop.
   - **This project's stated deployment shape is 8-GPU hosts running
     SGLang TP=8 (one replica, 8 ranks, per host).** For that specific
     shape, a single shared identity per host would actually have been
     *semantically adequate* on its own (all 8 ranks in one TP group
     process every request in lock-step, so "this host has hash H
     cached" is a node-level fact, not a per-rank one) — but adopting a
     single-shared-identity design would silently regress the
     multi-replica-per-host case the baseline already supports, for no
     reason other than convenience. Per explicit instruction, this
     proposal should support that case too rather than quietly drop it.

   **Decision: the server maintains one independent distributed
   sub-identity per connected worker (`client_id`), not one shared
   identity for the whole host.**

   **(v1.2 fix — this used to say "N independent `MasterClient`-level
   registrations." That was wrong in a way that would have broken
   ordinary distributed routing, not just been imprecise. Corrected to
   a purpose-built `ExternalKvIdentityClient` instead, per three
   blockers found on review, all confirmed against source, not
   speculative.)**

   **Blocker A — a naive `MasterClient`-based sub-identity would pollute
   normal Put/Get routing.** Once `RegisterClient` succeeds, that
   identity becomes eligible for `ClientRegistry::GetAliveClients()`
   (`client_registry.cpp:50`), which ordinary `RoutePut` placement
   selects targets from
   (`route_put_strategy.cpp`'s `CollectEligibleOnTier`, iterating
   *all* alive clients and matching on `tier_capacities`). If a
   sub-identity reported any nonzero `tier_capacities`, real KV blocks
   from unrelated Puts could get routed onto it — a node that isn't a
   real storage participant at all. **Verified the fix works with the
   existing code, no routing-code change needed**: `CollectEligibleOnTier`
   already skips any candidate with no entry for the tier being routed
   (`if (it == client.tier_capacities.end()) continue;`), and
   `MasterServer::RegisterClient` does not reject empty/absent
   `tier_capacities` (`master_server.cpp:218-240`) — so a sub-identity
   registered with **empty `tier_capacities`** is already, structurally,
   invisible to `RoutePut`/`RouteGet` selection, with zero changes to
   the routing algorithm itself.

   **(v1.2 addition — the same invariant also covers `EvictionManager`,
   checked directly rather than assumed to be covered by the RoutePut
   fix alone.)** `EvictionManager::RunOnce` (`eviction_manager.cpp:83-91`)
   also calls `registry_.GetAliveClients()` and iterates each client's
   `tier_capacities` to detect overloaded node-tiers
   (`for (const auto& [tier, cap] : client.tier_capacities) { ... }`);
   a client with an empty `tier_capacities` map contributes zero
   iterations and can never be flagged as overloaded or selected as an
   eviction target — the same empty-capacities invariant that hides a
   sub-identity from `RoutePut` also makes it structurally invisible to
   `EvictionManager`, with no separate mechanism needed. **This must
   hold by construction, not by convention that a future edit could
   quietly break** (e.g. someone "helpfully" wiring up real capacity
   reporting on `ExternalKvIdentityClient` to make monitoring output
   look more complete) — which is exactly why Blocker B's fix is a
   dedicated class with no `tier_capacities`-reporting code path at
   all, rather than a `MasterClient` configured to behave this way by
   convention. **Additionally decided: `ExternalKvIdentityClient`'s
   heartbeat must never publish `EventBundle`s (owned-KV-block delta/
   full-sync events) — it has no owned index to report in the first
   place, since it is not a real storage participant; this is stated
   explicitly here, not left to be inferred from "empty capacities,"
   because the two are different fields in `HeartbeatRequest`
   (`tier_capacities` vs. `bundles`, `distributed/proto/umbp.proto:103-109`)
   and both must independently stay empty.**

   **Blocker B — reusing `MasterClient` directly is the wrong vehicle,
   for two compounding reasons, so this is now a dedicated class:**
   `MasterClient` is a full-featured class built for real distributed
   peers — capacity heartbeats, KV-event-bundle sequencing, metrics
   reporting, and (separately) a real, **confirmed** bug in its
   re-register path: on receiving `CLIENT_STATUS_UNKNOWN`, the
   re-register request sets only `node_id`/`node_address`/
   `tier_capacities`/`tags` (`master_client.cpp:~612`) and **omits
   `peer_address`/`engine_desc`** — unlike the initial `RegisterSelf`
   (`master_client.cpp:137-141`), which sets both. Since
   `ClientRegistry::RegisterClient` unconditionally overwrites
   `record.peer_address`/`record.engine_desc_bytes` with whatever the
   request carried (empty, in the re-register case), any identity that
   re-registers after Master forgets it (restart, reaper expiry) loses
   its `peer_address` — and `MatchExternalKv`'s response, sourced from
   `ClientRegistry::GetAlivePeerView()` (`master_server.cpp:590`),
   would then return an empty `peer_address` for that node. This is a
   pre-existing bug in `MasterClient` itself (it would affect real
   distributed peers too, not just this proposal), but relying on it
   un-fixed for a *new* use case is not acceptable, and patching the
   shared `MasterClient` class carries real regression risk across
   every existing distributed peer for the sake of a narrow new
   feature.

   **Decision: introduce a new, purpose-built `ExternalKvIdentityClient`
   class instead of instantiating N `MasterClient`s.** It implements
   only `Register`/`Heartbeat`/`Unregister` plus the five external-KV
   RPCs — no capacity heartbeats, no KV-event-bundle logic, no
   `PeerDramAllocator`/`PeerSsdManager`/`PeerServiceServer` coupling.
   This resolves both blockers structurally rather than by convention:
   - **Blocker A is closed by construction, not configuration**: the
     class has no `tier_capacities`-reporting code path at all, so
     there is no field a future edit could accidentally populate with
     real capacity — contrast with "reuse `MasterClient` but always
     pass empty capacities," which depends on every future call site
     remembering to keep passing empty.
   - **Blocker B is closed by not repeating the bug**: since this is
     new code, its `Register`/re-register path always includes
     `peer_address`/`engine_desc` on every call, from the start — no
     fix to shared `MasterClient` code required, no regression surface
     on real distributed peers.
   - Heartbeat's only job for this class is liveness (keep the
     `ClientRegistry` entry from expiring) — it must **not** publish
     owned-KV events or capacity snapshots, since it represents no real
     storage.
   - All N instances (N ≤ 8 for the stated deployment shape) **share
     one common `peer_address`** (the server's own single physical
     `PeerService` endpoint) — **verified this pass, not assumed**:
     `peer_address` uniqueness is not required anywhere in the
     registry/dispatch path — `ClientRegistry` stores a plain
     `node_id → peer_address` map (`client_registry.cpp:217`),
     `MasterServer::GetOrCreateStub` keys its gRPC stub cache by
     `node_id` (`master_server.cpp:158-166`, tolerating multiple
     `node_id`s pointing at the same `peer_address`), and the
     `MatchExternalKv` response path returns `(node_id, peer_address)`
     pairs per match (`master_server.cpp:595`) — nothing here assumes
     or requires a 1:1 `node_id`↔`peer_address` mapping.
   - Each of the five external-KV RPC handlers, on receiving a call
     from a given `client_id`, dispatches it to *that worker's*
     `ExternalKvIdentityClient` instance (`client_id →
     ExternalKvIdentityClient` lookup, mirroring the existing
     `client_id → {base, size}` lookup already used for offset
     resolution, §3.2) rather than to any single, server-wide identity.
   - The bulk Put/Get/RegisterMemory RDMA data path is **unaffected**
     and continues to share the server's one real `DistributedClient`
     backend (built via `CreateUMBPClient`, per §11.1/§11.4.1) for
     efficiency — per §11.3/§11.4.3, RDMA registration is inherently
     per-VA and identity-agnostic, so there is no reason to split it
     per worker; only the identity-bound external-KV surface needs
     per-worker handling, and now has a class whose only job is that.

   **Blocker C — the registration-handshake field list, decided now,
   not left to implementation time.** `RegisterClientRequest` requires
   both `node_id` **and** `node_address`
   (`distributed/proto/umbp.proto:54-60`) — a v1.1 draft of this
   section under-specified this as "just `node_id`." **Decided:** the
   worker conveys `worker_node_id`, `worker_node_address`, and
   optionally `tags` — **not raw rank components** (`dp_rank`/`pp_rank`/
   `tp_rank`). Python (`umbp_store.py`'s existing baseline node_id/
   node_address derivation logic, `umbp_store.py:496-519`) remains the
   single source of truth for how these strings are computed; the
   server never re-derives rank semantics itself. This avoids two
   independent, potentially-diverging implementations of the same
   derivation rule ever existing at once.

   **(v1.2 — wire-schema placement decided, not left as prose without
   a concrete message.)** These fields are added to the **existing**
   `RegisterMemoryRequest` message (`distributed/proto/umbp_standalone.proto:87-91`,
   currently `{client_id, worker_base, size}`), not a new RPC:

   ```protobuf
   message RegisterMemoryRequest {
     string client_id = 1;
     uint64 worker_base = 2;
     uint64 size = 3;
     string worker_node_id = 4;       // new
     string worker_node_address = 5;  // new
     repeated string tags = 6;        // new, opaque key=value labels,
                                       // mirrors RegisterClientRequest.tags
   }
   ```

   Rationale for extending this message rather than adding a separate
   `RegisterWorkerRequest`: `RegisterMemoryRequest` is already the gRPC
   call that completes a worker's registration lifecycle (the
   "confirm" step after the raw-UDS fd handoff, §3.2/§11.4.3) — a
   worker's memory registration and its external-KV identity
   registration should start and end together (the sub-identity should
   not exist before the worker has a registered buffer, and should be
   torn down at `DeregisterMemory`, per §11.4.7's shutdown/lifecycle
   spec below), so piggybacking on the existing message avoids both a
   new round trip and a second, independent lifecycle to keep in sync
   with the first. `worker_node_id`/`worker_node_address` are plain
   `string` fields (not `optional`) — a worker that never sets them
   (e.g. a `StandaloneClient`-backed server has no use for them at all)
   simply leaves them empty, and the server only constructs an
   `ExternalKvIdentityClient` for a `client_id` whose
   `RegisterMemoryRequest` carried a non-empty `worker_node_id`.

   **Real, bounded implementation cost, not hidden:** this means the
   server runs up to N independent `ExternalKvIdentityClient`
   registrations and heartbeat threads concurrently (N ≤ 8 for the
   stated deployment
   shape) — the same aggregate cost the baseline already pays today
   (8 independent processes each running one), just consolidated into
   one process. This is not free, but it is not new cost relative to
   the baseline either; it is the honest price of preserving a
   capability the baseline already has, not a discretionary addition.

   **(v1.1 — confirmed as the final decision, alternatives considered
   and explicitly rejected, not left as an open trade-off.)** Two
   points raised in review, both resolved:
   - **No relay/double-hop exists in this design.** Each per-worker
     `ExternalKvIdentityClient` sub-instance runs entirely inside
     `umbp_standalone_server` and heartbeats `MasterServer` directly —
     the worker is never in this path after the one-time `node_id`
     handoff at registration; there is no "worker → server → Master"
     forwarding of heartbeat traffic anywhere in this design. What *is*
     real is N independent direct connections/threads originating from
     the same host, not a relay chain.
   - **Batching N identities onto one heartbeat connection is not
     possible without a Master-protocol change, and is out of scope
     here.** Verified: `RegisterClientRequest`/`HeartbeatRequest`
     (`distributed/proto/umbp.proto:54-115`) both carry a single
     `node_id` field, not `repeated` — the protocol has no batched-
     identity shape today, and both `MasterClient` and the new
     `ExternalKvIdentityClient` are 1-instance-per-`node_id` classes.
     Adding batched multi-identity heartbeats would be
     a `distributed/` master/protocol change affecting far more than
     `umbp_standalone_server`, and is explicitly not attempted here.
   - **No Mooncake precedent exists for this specific problem, checked
     directly rather than assumed.** `real_client.cpp` constructs
     exactly one `Client::Create(...)` for the Real Client's entire
     lifetime (confirmed: exactly 2 call sites, both single-instance
     construction — one primary, one port-retry variant — never one
     per `DummyClient`). Multiple `DummyClient`s sharing one Real
     Client are never given independent Master-facing identities in
     Mooncake; multi-tenancy there is handled purely as **storage-key
     namespacing** via `tenant_id` (`MakeTenantScopedStorageKey`) under
     one shared identity — a fundamentally lighter-weight answer than
     per-worker sub-identities, but one that only works because
     Mooncake has no feature analogous to external-KV requiring
     per-original-owner routing precision. This was raised explicitly
     as an alternative (call it Option B: one shared identity + `client_id`-
     based key/tenant namespacing, matching Mooncake's actual pattern,
     at the cost of losing the multi-replica-per-host routing precision
     §11.4.4 exists to preserve) and **explicitly rejected in favor of
     the N-sub-identity design (Option A, this section)** — decision
     confirmed: preserving parity with the `distributed/` baseline's
     existing per-worker-identity capability outweighs the simplicity
     Mooncake's narrower pattern would have offered here. This is
     genuinely new engineering for UMBP with no precedent (in either
     UMBP's own `distributed/` code or in Mooncake) to lean on for the
     "N identities behind one process" mechanics specifically — flagged
     honestly as such, not presented as a known-safe pattern being
     reused.

   Independent of the above, add the same `client_mu_`-style locking
   discipline these five handlers currently skip
   (`standalone_server.cpp:206-213` pattern) around whichever
   per-worker sub-identity they dispatch to, since each sub-identity
   is now a stateful object with its own background threads.

   Under a `StandaloneClient` backend (no distributed identity at all),
   behavior is unchanged from today: hardcoded stubs, matching
   `StandaloneClient`'s own no-op external-KV methods
   (`standalone_client.h:67-83`).
5. **`Ping` / capability signaling** (`standalone_server.cpp:193-197`,
   `umbp_standalone.proto:33-35`): today `PingResponse` has only
   `bool ready`, and nothing anywhere lets a worker learn whether the
   server it's connected to is distributed-backed.

   **(v0.9 fix — upgraded from "open, low stakes" to a required part of
   this proposal's implementation, not deferred.)** The reasoning in
   the v0.7 draft — "the data plane doesn't care, only humans debugging
   need it" — missed the actual failure mode this exists to catch: a
   single `UMBP_STANDALONE_ADDRESS` can now resolve to *either* a
   local-backed or a distributed-backed server (§11.6), and if the
   whole point of this section is Mooncake parity (cross-node), the
   most dangerous failure mode is exactly the quiet one — a worker (or
   the deployment/test tooling that stood up the server) connects
   successfully, every RPC succeeds, and the server has silently ended
   up local-backed instead of distributed-backed (e.g. an operator's
   deployment script forgot to set `UMBP_MASTER_ADDRESS` on the
   *server's* side — a legitimate, non-erroring config per §11.2, not
   the already-hard-failing misconfiguration case in item 6 below).
   Nothing observes this today; the feature's entire purpose (cross-
   node capability) can silently not exist while everything appears to
   work. This is the same class of failure this document has rejected
   everywhere else (§5, §8, item 6 below) — it should not be waved
   through here just because the data path itself doesn't strictly
   need it.

   **Decision: `PingResponse.deployment_mode` (new proto field,
   enum `LOCAL`/`DISTRIBUTED`, additive/backward-compatible,
   populated from `client_->GetDeploymentMode()`) is a required part
   of this proposal's implementation, not an optional nice-to-have.**
   Additionally: **any launch/health-check/integration-test tooling for
   the distributed-backed shape must assert
   `deployment_mode == DISTRIBUTED` after startup** — a distributed-
   backed server that comes up as `LOCAL` (for whatever reason) must be
   treated as a failed deployment, not a degraded-but-working one. The
   local-backed shape itself remains a fully legitimate, intentional
   deployment choice (§11.6) when that's what was actually configured
   — this assertion only guards against the *parity* shape silently
   collapsing into the *fallback* shape unnoticed.
6. **`standalone_server_main.cpp`**: currently unconditionally does
   `config.distributed.reset()` (`standalone_server_main.cpp:43`) and
   only ever builds `config.standalone_process`. Needs a **second,
   independent** config-building path for `backend_config` (§11.2).

   **(v0.9 fix — the v0.7 env-var list was incomplete, not just
   abbreviated; re-checked against the actual current
   `umbp_store.py` code, not recalled from memory.)** The v0.7 draft
   said "reuse the exact set `umbp_store.py` already reads" and then
   listed only 6 variables. Re-reading `umbp_store.py:458-620` in full,
   a real distributed worker's config surface is materially larger:
   `master_address`, `node_address`, `node_id`, `auto_heartbeat`
   (`umbp_store.py:524-527` — **note: `extra_config`-only, no env-var
   fallback exists for this one today**), `io_engine.host`,
   `io_engine.port`, `staging_buffer_size`, `ssd_staging_buffer_size`,
   `ssd_staging_buffer_slots`, `peer_service_port`,
   `cache_remote_fetches`, and `dram_page_size`. An implementer who
   copied only the original 6-variable list would build a server-side
   `DistributedClient` that silently behaves differently from an
   ordinary distributed worker — e.g. no SSD staging buffer sizing, no
   `cache_remote_fetches` admission control tuning — a real behavior
   drift, not a cosmetic gap.

   **One field needs special handling, not a direct copy:**
   `dram_page_size` in `umbp_store.py` (`umbp_store.py:598-620`) is
   normally *auto-derived by probing `mem_pool_host`* (the worker's own
   host KV tensor buffer) when not explicitly set — this derivation is
   meaningless for `standalone_server_main.cpp`, which has no
   `mem_pool_host` of its own (the server's `client_` backend isn't
   attached to any particular worker's tensor layout — it may serve
   many workers with potentially different layouts, per §4.4). The
   server-side config builder must **only** support the explicit-
   override path (an env var, e.g. `UMBP_DRAM_PAGE_SIZE`, mirroring
   `extra_config["dram_page_size"]`'s role as "operator-controlled
   escape hatch" per the comment at `umbp_store.py:590-598`) and leave
   it at the `UMBPDistributedConfig` default (`0`, delegating to the
   master's own `ClientRegistryConfig.default_dram_page_size`) when
   unset — it must not attempt to replicate the `mem_pool_host`-probing
   auto-derivation, which does not apply here.

   **Proposed shape for the implementation**: a single
   `BuildDistributedBackendConfigFromEnv()` C++ helper (new,
   `standalone_server_main.cpp` or a shared helper if reused elsewhere)
   that mirrors `umbp_store.py`'s env-var-driven construction of
   `UMBPDistributedConfig`, split explicitly into:
   - **Required** (missing any of these means "distributed backend not
     requested" — server falls back to local-backed, or hard-fails per
     the decision later in this same item if some but not all are set,
     which is itself a misconfiguration signal): `UMBP_MASTER_ADDRESS`,
     `UMBP_NODE_ADDRESS`, `UMBP_NODE_ID`, `UMBP_IO_ENGINE_HOST`.
   - **Optional, with the same defaults `UMBPDistributedConfig` already
     has when unset**: `UMBP_IO_ENGINE_PORT`, `UMBP_PEER_SERVICE_PORT`
     (both reuse the exact worker-side names, since those two already
     have an established env-var convention — `umbp_store.py:526-543` —
     to mirror), plus five genuinely new names, **finalized now, not
     left as "proposed"**: `UMBP_DISTRIBUTED_STAGING_BUFFER_SIZE`,
     `UMBP_DISTRIBUTED_SSD_STAGING_BUFFER_SIZE`,
     `UMBP_DISTRIBUTED_SSD_STAGING_BUFFER_SLOTS`,
     `UMBP_DISTRIBUTED_CACHE_REMOTE_FETCHES`,
     `UMBP_DISTRIBUTED_DRAM_PAGE_SIZE` (per the special handling above).
     **(v1.2 — decided, no longer "proposed.")** These five get the
     `UMBP_DISTRIBUTED_` prefix specifically because, unlike
     `UMBP_MASTER_ADDRESS`/`UMBP_NODE_ADDRESS`/`UMBP_NODE_ID`/
     `UMBP_IO_ENGINE_HOST`/`UMBP_IO_ENGINE_PORT`/`UMBP_PEER_SERVICE_PORT`
     (which reuse worker-side names verbatim, referring to the exact
     same concept in both places), these five have **no existing
     env-var form at all today** — worker-side, they are
     `extra_config`-only (`umbp_store.py:555-598`) — so there is no
     established bare name to collide with, but there is a real future
     risk of a bare `UMBP_STAGING_BUFFER_SIZE` etc. later being added
     as a worker-side env var with different semantics (e.g. a
     per-worker override rather than a server-backend-wide default);
     the `_DISTRIBUTED_` infix makes clear these five specifically
     configure the *server's own* `UMBPDistributedConfig`, not
     anything worker-facing, closing off that ambiguity before it can
     arise. `auto_heartbeat` has no worker-side env var today either
     (only `extra_config`); **decided: the server-side helper keeps the
     `UMBPMasterClientConfig` default (`true`) unconditionally** rather
     than adding a dedicated env var for a field with no operational
     reason to ever be `false` for these sub-identities-plus-shared-
     backend registrations — simpler than inventing
     `UMBP_DISTRIBUTED_AUTO_HEARTBEAT` for a knob nobody has asked to
     turn off.

   **(v0.7 fix — decided, was flagged as a v0.6 blocker.) A
   distributed-backed `umbp_standalone_server` supports external
   launch ONLY — it does not support the §4.2 auto-start (worker
   fork+exec) path. Local-backed auto-start (§4.2, already shipped in
   v0.1-v0.5) is completely unaffected by this and continues to work
   exactly as today.**

   The conflict this closes: v0.5 (§5) made
   `UMBPStore.__init__` raise if a worker's own environment has both
   `UMBP_STANDALONE_ADDRESS` and `UMBP_MASTER_ADDRESS` set (mutual
   exclusivity, by design — a worker is either distributed or
   standalone-process, never both). If a worker's own auto-start path
   were to fork+exec a distributed-backed server, the env vars needed
   to configure that child (`UMBP_MASTER_ADDRESS` etc.) would most
   naturally live in the *parent's* (the worker's) environment too
   (fork inherits environment by default) — which is exactly the
   combination v0.5 already made illegal, on purpose, for the worker's
   own config parsing. Two ways to resolve this were considered
   (support auto-start via a separate `UMBP_STANDALONE_BACKEND_*` env
   namespace the worker never reads, vs. drop auto-start for this mode
   entirely) — **resolved by matching Mooncake rather than inventing
   either option**: re-checked directly against the local Mooncake
   checkout this pass (`grep -rln "fork\(|execvp|execve|posix_spawn|subprocess\.|Popen" mooncake-integration/ mooncake-wheel/`
   → zero matches), Mooncake's application side (`DummyClient`) never
   forks/execs the Real Client under any circumstance — the Real
   Client is always started by something outside the application
   (deployment system, operator, systemd unit, etc.), full stop, no
   exceptions for any deployment shape. Adopting the same rule here is
   simpler than a second env namespace, has zero conflict with the
   already-shipped v0.5 rule, and matches the explicit goal of this
   whole section (Mooncake parity) more directly than inventing a
   UMBP-specific auto-start variant Mooncake itself doesn't have.

   In practice: `standalone_server_main.cpp` reads
   `UMBP_MASTER_ADDRESS` etc. **from its own process environment**,
   set by whatever external launcher starts it (a shell script,
   Kubernetes sidecar spec, systemd unit — the same category of
   launcher `run_umbp_single_node_hicache.sh` already represents for
   `umbp_master` today). Workers never see or set these variables;
   they only ever need `UMBP_STANDALONE_ADDRESS`, unchanged from
   today.

   **Decision (also confirmed, unchanged from v0.6): if
   `UMBP_MASTER_ADDRESS` is set but `PoolClient::Init` can't actually
   stand up the RDMA IO engine** (e.g. `UMBP_IO_ENGINE_HOST` empty —
   `PoolClient::Init` silently skips building the IO engine entirely
   in that case, `pool_client.cpp:385-415`, rather than failing),
   **server startup must hard-fail**, not fall back to a
   `StandaloneClient` backend with a warning. Silently downgrading a
   server an operator explicitly asked to be distributed-backed into a
   local-only one is exactly the kind of silent-degradation failure
   mode this whole design doc has repeatedly flagged as unacceptable
   elsewhere (§5, §8). A misconfigured distributed-backed server
   should refuse to start, not quietly become a different, smaller
   feature.

### 11.4.7 `ExternalKvIdentityClient`: interface, lifecycle, and layering (v1.2, new)

**(This subsection exists because §11.7 previously filed
`ExternalKvIdentityClient` as an open item needing "its own design
pass" — that framing was too weak once §11.4.4 made it a required
component of the final design, not an optional extra. This is that
design pass, not a placeholder for a future one.)**

**Layering — this is a UMBP compatibility extension, not part of
Mooncake-parity core.** Worth stating explicitly so it is never
mistaken for required minimum-viable scope: the Mooncake-parity core
of this whole section (§11) is `distributed-backed umbp_standalone_server
+ UDS/shm fd handoff` (§11.1-§11.3) — that alone is what makes this
design match Mooncake's Real Client shape. `ExternalKvIdentityClient`
exists purely to avoid *regressing* a capability the `distributed/`
baseline already has (per-worker external-KV routing precision, §11.4.4's
"why this matters" discussion) — it has no Mooncake analog and is not
required for Mooncake parity itself. Concretely, this means:
- It can be scheduled, implemented, and tested as its own, separable
  unit of work after the parity core is working — a distributed-backed
  server with `ExternalKvIdentityClient` not yet implemented is a
  legitimate intermediate state (external-KV simply stays unavailable
  for that server, exactly like the `StandaloneClient`-backed case
  already handles it today), not a broken or non-conforming one.
- It must never be treated as sitting on the critical path for
  declaring "Mooncake parity achieved" — conflating the two would
  under-scope the parity-core milestone by tying it to a strictly
  UMBP-specific feature Mooncake itself has no equivalent of.

**Interface** (methods on the new class; not a proto service of its
own — it is a client-side C++ class inside `umbp_standalone_server`,
using the *existing* `UMBPMaster` gRPC service just like `MasterClient`
does):

- `Register(worker_node_id, worker_node_address, peer_address, engine_desc, tags)`
  — sends `RegisterClientRequest` with **empty `tier_capacities`**
  (§11.4.4's Blocker A invariant) and the given `peer_address`
  (shared across all sub-identities on this server, §11.4.4) /
  `engine_desc` (may be empty/dummy — nothing ever RDMA-targets a
  sub-identity's "engine," since external-KV never moves bytes through
  UMBP, §11.4.4's opening clarification). Constructed the moment a
  worker's `RegisterMemoryRequest` carries a non-empty
  `worker_node_id` (§11.4.4 Blocker C's wire-schema decision).
- `Heartbeat()` — periodic, liveness-only. Sends `HeartbeatRequest`
  with **empty `tier_capacities` and empty `bundles`** on every call
  (§11.4.4's v1.2 addition) — never derives or forwards real capacity
  or owned-KV-event data. Its only job is keeping the `ClientRegistry`
  entry from expiring so `MatchExternalKv` continues to resolve this
  `node_id`.
- `Unregister()` — sends `UnregisterClientRequest`. Called on: (a) the
  corresponding worker's `DeregisterMemory` RPC (worker-initiated
  teardown), and (b) server shutdown (see below).

  **(v1.4 fix — this bullet previously claimed a third trigger,
  "worker disconnect/crash detection (mirrors however
  `StandaloneProcessClient` disconnect is already detected... §8 item
  1's connection state machine)," as if it already existed. That was
  false, and worse than simply undocumented: §8 item 1 itself states
  plainly that this exact mechanism is unsolved — "Crash / restart
  semantics are unsolved by every precedent reviewed... this needs
  explicit handling in `StandaloneProcessClient` and probably a small
  state machine (`CONNECTED` → `DISCONNECTED` → optionally
  `RECONNECTING`), which does not exist in any of the three reference
  systems today." §11.4.7 was citing, as its basis for "already
  detected," a mechanism §8 item 1 explicitly says does not exist —
  an internal contradiction a reader of this subsection alone would
  never catch.)**

  **Confirmed gap, stated plainly instead of implied as solved: there
  is currently no worker crash/disconnect detection anywhere in
  `standalone_server.cpp`.** The only two paths that ever tear down an
  `ExternalKvIdentityClient` are `DeregisterMemory` (explicit,
  worker-initiated) and full server `Shutdown()`
  (`UnregisterAllExternalIdentities`, see below). **Consequence, stated
  explicitly:** if a worker process crashes or is killed without ever
  calling `DeregisterMemory` (SIGKILL, OOM-kill, or any other
  non-graceful exit), its `ExternalKvIdentityClient` keeps running
  inside `umbp_standalone_server` indefinitely — its heartbeat thread
  (which lives in the *server* process, not the worker's, so the
  worker's death has no effect on it at all) keeps sending
  liveness-only heartbeats forever, so the Master's `ClientRegistry`
  entry for that `node_id` never expires and stays `ALIVE`.
  `MatchExternalKv`/`GetExternalKvHitCounts` will keep returning this
  node_id/peer_address to callers for KV blocks that may no longer
  exist (the crashed worker's shm segment is gone), and this is **not
  cleaned up until the entire `umbp_standalone_server` process is
  restarted** — there is no timeout, reaper, or liveness poll for this
  case anywhere in the reviewed code. This is a known, deliberately
  deferred limitation (fixing it properly needs a real
  liveness/crash-detection mechanism — most plausibly something built
  on top of the connection-state-machine work §8 item 1 already
  flags as needed for `StandaloneProcessClient`'s own reconnect
  handling, not a small patch specific to `ExternalKvIdentityClient`
  alone), tracked here and in §8 item 1, not silently absent.
- `ReportExternalKvBlocks`/`RevokeExternalKvBlocks`/
  `RevokeAllExternalKvBlocksAtTier`/`MatchExternalKv`/
  `GetExternalKvHitCounts` — thin forwarding wrappers to the
  corresponding `UMBPMaster` RPCs, using this sub-identity's own
  `node_id`. These are what the five `standalone_server.cpp` external-KV
  RPC handlers (§11.4.4, item 4) dispatch into once they've resolved
  the calling `client_id` to its `ExternalKvIdentityClient` instance.

**Re-registration must not repeat `MasterClient`'s bug (§11.4.4
Blocker B) — stated as a hard requirement, not left implicit in "this
is new code so it won't have the bug":** every `RegisterClientRequest`
this class ever sends — the very first one and any sent after a
`CLIENT_STATUS_UNKNOWN` heartbeat response — **must** include
`peer_address` and `engine_desc`. Implementation-wise, the simplest way
to guarantee this is to store `peer_address_`/`engine_desc_bytes_` as
member fields set once at construction and have exactly one internal
`BuildRegisterRequest()` helper used for both the initial `Register()`
call and any re-register path, rather than two independently-written
request-construction code paths (which is how `MasterClient`'s bug was
introduced in the first place — the re-register path was written
separately from `RegisterSelf` and simply forgot two fields).

**Lifecycle serialization trade-off (accepted v1.4 implementation
simplification):** the current implementation may use one server-wide
lifecycle mutex to serialize `ExternalKvIdentityClient` register,
deregister, and shutdown bookkeeping. This deliberately closes
register-vs-deregister races for the same `client_id`, but it also
means the lock can cover blocking Master RPCs (`RegisterClient` /
`UnregisterClient`, bounded by the RPC shutdown deadline). Therefore,
when the Master is slow or unavailable, otherwise unrelated workers can
queue behind each other during `RegisterMemory`/`DeregisterMemory`; in
the stated TP=8 deployment shape, the worst-case startup/shutdown delay
can be amplified toward `N * rpc_deadline` rather than one parallel
deadline. This is an accepted simplification because these are
lifecycle operations, not the Put/Get hot path, and because the current
target scale is small. If future deployments support many more workers,
frequent worker reconnects, or observe startup stalls caused by slow
Master RPCs, this lock should be the first place to revisit: either
replace it with per-`client_id` lifecycle locks or decouple
external-KV identity registration from core memory registration with an
async/retry path.

**Shutdown ordering** (extends §4.3/§11.4.3's existing shutdown
sequence, does not replace it): `umbp_standalone_server`'s `Shutdown()`
must `Unregister()` all live `ExternalKvIdentityClient` instances
**before** tearing down the shared `DistributedClient` backend
(`client_->Close()`) or exiting — unregistering first ensures the
Master's `ClientRegistry` doesn't briefly carry stale `ALIVE` entries
for identities whose process is already gone, which would otherwise
sit around until heartbeat-timeout-driven reaping (§4, the same
"absent liveness" concern already motivated for the primary
`DistributedClient` identity). Order within `Shutdown()`, appended to
the sequence already specified in §4.3/§11.4.3: drain in-flight RPCs →
**unregister all `ExternalKvIdentityClient` instances** → flush
`CopyPipeline`/close the shared `DistributedClient` backend → unmap →
exit.

### 11.5 What does *not* change

- The worker-facing IPC layer: UDS control-plane socket, the separate
  raw-UDS `.fd.sock` `SCM_RIGHTS` channel, offset translation,
  `StandaloneProcessClient`'s locking discipline, the fd-ownership
  registry in `HostMemAllocator` — **none of this is touched**. All of
  it was built, reviewed, and bug-fixed independent of what backs the
  server side, and stays exactly as-is.
- `local/` and today's `distributed/`-for-workers path — **both
  unmodified**, per §7's existing compatibility statement, which this
  proposal does not change.
- The `UMBPStandaloneProcessConfig` shape workers use (§6) —
  unmodified. Workers still only ever set `address` (env-var-only
  activation, per the v0.5 decision) and never need to know or care
  what the server's own backend is.

### 11.6 Consequence for "what does the local DRAM tier even mean now"

Worth stating explicitly since it's a natural point of confusion: when
`client_` is a `DistributedClient`, the server's `LocalStorageManager`/
`DRAMTier`/`SSDTier` concept (§3.2's earlier "server's own DRAM tier
should default to private anon/hugetlb" discussion) **does not apply
at all** — `DistributedClient` doesn't use `LocalStorageManager`, it
has its own separately-owned DRAM pool built via `HostMemAllocator`
inside `PoolClient`/`DistributedClient`'s own construction
(`distributed_client.cpp:41-53`), used as its local slice of the
distributed pool, not as a passthrough cache for the registered worker
buffers. The registered worker buffers (via `RegisterMemory` in
§11.4.3) are a *separate* thing: they exist purely so RDMA can target
them directly, not so `DistributedClient` manages them as cache
storage. **There are now three, not two, deployment shapes for
`umbp_standalone_server`**, and the design should name them distinctly
to avoid ambiguity in docs/logs/config:

| Shape | `client_` backend | Cross-node | Server's own storage |
|---|---|---|---|
| existing v0.1-v0.5 (unchanged) | `StandaloneClient` | no | `LocalStorageManager` (DRAM+SSD tiers, private memory) |
| this proposal, backend enabled | `DistributedClient` | **yes**, via Master+RDMA | `DistributedClient`'s own DRAM pool (distinct concept, not `LocalStorageManager`) |
| this proposal, backend disabled | `StandaloneClient` | no | same as row 1 — this is the fallback/default, identical to today |

### 11.7 Open questions for discussion (before any code)

**(v0.7: items 2 and 4 from the v0.6 draft are now resolved — see
§11.4.6 and §11.3 respectively — and are kept here only as a record of
what was decided and why. Items 1, 3, 5 remain genuinely open.)**

1. **Naming/scope framing — still open, needs a decision.** The
   "Scope reframing" paragraph near the top of §11 (added this pass)
   already states that
   distributed-backed is the shape that actually matches the stated
   goal, and local-backed is a fallback/dev-convenience shape, not a
   co-equal alternative. Given that, should the *document* (and
   eventually config/logs) use a name like
   `standalone-process/distributed-backend` (or "real-client mode") vs
   `standalone-process/local-backend` per §11.6's table, rather than
   presenting them as two unlabeled options? Recommend adopting this
   secondary naming now, in the doc, ahead of implementation — it's a
   documentation-only decision but should be settled before code
   introduces its own inconsistent naming (env var prefixes, log
   messages, etc).
2. ~~The hard-fail-on-misconfiguration decision in §11.4.6~~ —
   **resolved**: hard-fail, confirmed in §11.4.6.
3. ~~Whether `PingResponse.deployment_mode` is worth the proto change
   now~~ — **resolved (v0.9): required, not deferred.** See §11.4.5 —
   this is now a mandatory guard against the parity shape silently
   collapsing into the local-only fallback shape, not an optional
   debugging aid.
4. ~~The cross-mapping memory-visibility question in §11.3~~ —
   **resolved**: closed via the Mooncake production-precedent citation
   plus the explicit gRPC-response-as-happens-before-edge contract, see
   §11.3.
5. Testing implications, not yet scoped: the existing
   `test_standalone_shm_ipc.cpp` suite only exercises the
   `StandaloneClient`-backed path. A `DistributedClient`-backed test
   needs at minimum a running `umbp_master` plus the RDMA/IO-engine
   stack available in the test environment — likely a heavier
   integration test than what exists today, possibly gated separately
   (e.g. only run where RDMA hardware/loopback is available), which
   needs its own test-infrastructure decision before it can be added
   to `§10`'s implementation-order list. Should also cover: per-worker
   external-KV sub-identity registration/dispatch under a
   distributed-backed server with 2+ connected workers (§11.4.4 — this
   is the item's own core mechanism now, not an edge case), and the
   `RegisterMemory`/`DeregisterMemory` rollback/shutdown-ordering paths
   added in §11.4.3.
6. ~~The exact registration-handshake field(s) for per-worker identity
   conveyance~~ — **resolved (v1.2, Blocker C): `worker_node_id` +
   `worker_node_address` + optional `tags`, not raw rank components.**
   See §11.4.4.
7. ~~`ExternalKvIdentityClient` needs a design/implementation pass of
   its own~~ — **resolved (v1.2): see §11.4.7** for the full interface
   (`Register`/`Heartbeat`/`Unregister`/the five external-KV RPCs),
   the re-registration hard requirement, and shutdown ordering. It is
   still real, net-new implementation work — it doesn't exist anywhere
   in the codebase today, unlike `MasterClient`, which at least could
   have been (incorrectly) reused — so it still needs to be factored
   into scheduling, not just the N-heartbeat-threads cost already noted
   above; what changed is that its shape is now fully specified rather
   than deferred.

### 11.8 Source references for this section

- `/apps/nima/KVManager/mori/src/umbp/standalone/standalone_server.{h,cpp}`
- `/apps/nima/KVManager/mori/src/umbp/standalone/bin/standalone_server_main.cpp`
- `/apps/nima/KVManager/mori/src/umbp/distributed/distributed_client.{h,cpp}`
- `/apps/nima/KVManager/mori/src/umbp/distributed/pool_client.{h,cpp}`
- `/apps/nima/KVManager/mori/src/umbp/include/umbp/distributed/master/master_client.h`
- `/apps/nima/KVManager/mori/src/umbp/include/umbp/common/config.h`
  (`UMBPConfig::Validate()`, `UMBPDistributedConfig`,
  `UMBPMasterClientConfig`, `UMBPIoEngineConfig`)
- `/apps/nima/KVManager/mori/src/umbp/umbp_client_factory.cpp`
- `/apps/nima/KVManager/mori/src/umbp/distributed/proto/umbp_standalone.proto`
- `/apps/nima/KVManager/mori/src/umbp/include/umbp/umbp_client.h`
  (`IUMBPClient::RegisterMemory`'s default no-op-returns-true contract)
