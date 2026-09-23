# Standalone-process `client_mu_` startup stall

Status: **fixed**. The mechanism and the fix are below; the reproduction recipe
is kept so the fix can be confirmed against the workload that found it.

## Symptom

Under real concurrent agentic load (Kimi-K3, TP8, DCP8, conc=24-48,
external-cache-linker / standalone-process mode), the first minute or
two of warmup can show every scheduler rank's main thread hung with zero
progress for anywhere from ~1 to ~5 minutes. `aiperf`'s `returned` counter
sits flat; `#running-req`/`in_flight` do not move. The run is not
permanently dead — it eventually clears on its own and proceeds
normally — but the wait is unbounded in principle and its length is not
predictable from run to run (observed: transient and self-clearing in
some repros, a hard scheduler death — `FATAL: server gone`, process no
longer present — in at least one other repro on a heavily-loaded shared
node, though that specific crash was not confirmed to share this root
cause).

Reproduction: `k3-dcp8/shared/scripts/ci_dcp8.sh ARM=umbp CONC=24`
(or 32/48) against image `aiat-yutongwu-pr39511-tp8-umbp-pr656:20260917`,
sglang-k3 branch `k3-hybrid-mamba-linker-rebased`. No special config
needed beyond what that harness already sets; `UMBP_DRAM_READ_LEASE_MS`
at its default (500ms) reproduces it, and so does 30000ms — that knob
turned out to be unrelated (see below).

## The lock

`py-spy dump --native --pid <scheduler_TP*>` during the stall, on every
one of the 8 ranks simultaneously:

```
epoll_wait (libc.so.6)
grpc_pollset_work (libgrpc.so.29.0.0)
grpc::internal::BlockingUnaryCallImpl<...>::BlockingUnaryCallImpl (mori/libmori_pybinds.so)
_exists_chunk (sglang: storage/umbp/umbp_direct_linker.py:466)
lookup -> match -> match_prefix -> scheduler main loop
```

The scheduler's main thread is blocked inside a genuine gRPC unary call
(`BatchExists`) waiting for the standalone server's response. `BatchExists`
itself is a shared-mutex-protected hash-map lookup (`PageBackend::BatchContains`)
plus, for keys this node does not hold, one master `BatchLookup` — sub-millisecond
work. What it was waiting on was `standalone_server.cpp`'s `client_mu_`.

**`client_mu_` is one mutex per NODE, not per rank.** The server address is
`unix:///run/umbp/standalone/${UMBP_NODE_ID}.grpc.sock`
(`ipc.cpp: DefaultStandaloneAddress`), and `StandaloneProcessClient` takes a
bootstrap lock so only one server process is ever spawned. All 8 TP ranks of a
node share one server object and therefore one `client_mu_`. (An earlier
revision of this document claimed it was per-rank because `client_id` is; that
was wrong, and it understated the contention eightfold.)

Two paths held it **exclusively** across operations that are not short:

1. **Memory registration — the dominant cause of the startup window.**
   `RegisterFd` → `RegisterBackendMemory` wrapped `client_->RegisterMemory` in
   `unique_lock<shared_mutex>(client_mu_)`, and that call ends in
   `IOEngine::RegisterMemory`, i.e. an RDMA MR pin over the worker's whole host
   KV pool. `standalone_process_client.cpp` records the measured cost in the
   comment on `RegisterMemoryRpcTimeoutMs`: *"can legitimately take 90-120+
   seconds (observed: `[DRAMTier] host memory registered for GPU access:
   1187840 MiB in 599.6 s`), plus sequential per-GPU IPC handle registration
   each well over a minute"*. Ranks register at staggered times during warmup,
   so for minutes at a stretch the whole node's data plane — every rank's
   `BatchExists` included — sat behind one rank's pin. This matches the
   observation that the stall is confined to the startup window, lasts 70-260s,
   clears by itself, and varies run to run.

2. **Bulk writes.** `BatchPutRanges` and friends held the same lock exclusively
   until `client_->BatchPutRanges` returned, covering the data copy *and* the
   `BatchRoutePut` round trip to the master. Steady-state contention rather than
   a startup cliff, but it serialized all 8 ranks' offload threads against each
   other.

Underneath, the same shape appeared once more: `PoolClient::RegisterMemory`
called the slow `transfer_engine_->RegisterMemory` while holding
`registered_mem_mutex_` exclusively — the same lock `FindRegisteredMemory` takes
once per range on every transfer.

## Why the lock could not simply be narrowed

Holding `client_mu_` across the backend call was also, incidentally, the
lifetime barrier for resolved host/GPU mappings. `ReleaseRegisteredMemory` took
the same lock exclusively, so a copy in flight could not have its buffer
`munmap`'d — nor, more importantly, have its RDMA MR torn down by the inner
`DeregisterMemory`, since `PoolClient::FindRegisteredMemory` hands out *copies*
of a region's `TransferRef` and the inner client has no pin of its own.

`StandaloneShmIpcTest.DeregistrationCannotUnmapAnInFlightDataOperation` guards
that invariant, and confirms it is load-bearing: removing the replacement
mechanism below makes it segfault on every run.

## The fix

**`standalone_server.cpp`** — a per-region pin replaces the lock as the
lifetime barrier.

- `memory_` holds `shared_ptr<Region>`, where `Region` is an immutable
  `RegisteredMemory` plus an atomic pin count. Reaching regions only through
  `shared_ptr` is what lets a handler keep a resolved pointer across a backend
  call while `InsertOrReplaceRegion` or `UnmapClient` runs underneath.
- `ResolveRange`/`ResolveRanges` — the single choke point every data path goes
  through — pin every region they resolve into, **while still holding
  `memory_mu_`**. A region reachable through `memory_` has not begun releasing,
  which is what makes the acquisition race-free. `RegionPins` releases on scope
  exit.
- `ReleaseRegisteredMemory` waits for the pin count to reach zero before
  deregistering or unmapping. Its precondition — the region is already
  unreachable through `memory_`, so no new pin can appear — holds at all four
  call sites. It waits holding neither `client_mu_` nor `memory_mu_`, because an
  in-flight operation needs both to finish and drop its pin.
- Data handlers and `RegisterBackendMemory` now take `client_mu_` **shared**.
  Writes use the same `ConditionalDataLock` as reads, so the SSD medium
  (`shared_reads_ == false`) keeps serializing exactly as before — only DRAM
  gains concurrency. `Clear` and shutdown remain exclusive.

After this, no data-plane or registration path takes `client_mu_` exclusively on
a DRAM medium, so neither cause above can reproduce.

**`pool_client.cpp`** — a new `registration_mutex_` serializes
`RegisterMemory`/`DeregisterMemory` against each other, held across the engine
call. That is the mutual exclusion `IOEngine` actually needs (its `memPool` and
`backends` carry no lock of their own — `include/mori/io/engine.hpp`); it is
moved, not relaxed. `registered_mem_mutex_` is now taken only for the table
insert/erase, so `FindRegisteredMemory` no longer waits minutes behind a pin.

**Why the inner client tolerates the new concurrency.** `DistributedClient`
already had exactly this discipline — every data operation and
register/deregister takes `op_mutex_` shared, only `Clear`/`Close` take it
exclusively. `PoolClient` serializes just its shared staging arenas
(`ranged_put_scratch_mutex_` / `ranged_get_scratch_mutex_`), `PeerPool` keeps
backend calls outside `operation_mutex_`, and each transfer engine carries its
own lock or thread pool. Pinning until the call returns is sufficient because
the inner data-plane calls are synchronous with respect to caller pointers:
`SubmitRemoteBatchPut`/`SubmitRemoteBatchGet` post without waiting, but the
in-flight handles are function-local and always waited before return, with the
handle's destructor draining on exceptional exit.

## Regression coverage

`tests/cpp/umbp/distributed/test_standalone_shm_ipc.cpp`:

- `DeregistrationCannotUnmapAnInFlightDataOperation` (pre-existing) — the
  single-region lifetime invariant.
- `DeregistrationWaitsForAnInFlightRangedOperationAcrossRegions` — the
  multi-region one. Its ranges are listed last-region-first on purpose: a
  teardown releases a client's regions in registration order, so a pin covering
  only the first region resolved would still be shielded by that ordering.
  Resolving in the opposite order removes the accident. Verified by mutation —
  pinning only the first region segfaults this test 5 runs out of 5 while every
  other test still passes.
- `ConcurrentPutsToDistinctKeysAllSucceed` — writers now run through the inner
  client side by side; every value must read back as its writer wrote it.
- `RepeatedRegistrationChurnUnderLoadNeverWedges` — registration/deregistration
  churn against live traffic, under a deadline. A leaked pin or a lost wakeup in
  `WaitForPinsZero` hangs rather than asserts, which is what the deadline is for.
- `ExistsStaysResponsiveUnderConcurrentWriteLoad` — the symptom, bounded. Honest
  limits: a unit test's copies and registrations are milliseconds, not the
  minutes a real KV pool takes, so this does not by itself prove the production
  stall is gone. It locks down the lock *shape*.

`tests/cpp/umbp/distributed/test_pool_client_batch_put.cpp`:
`ConcurrentRegistrationsAndLookupsStayConsistent` — registrations racing each
other and racing table lookups.

Known environment dependency: `PoolClientRangesTest.RemoteRoundTrip*`,
`StaleSelfLocationIsExcludedBeforeRemoteFetch`, `BatchPutWarnTest.
StagingFallbackSucceedsWithoutWarn` and `RegisteredSrcsNoWarn` need a working
RDMA device and fail identically with and without this change on a host without
one.

## Two things that are NOT the fix

- **`UMBP_DRAM_READ_LEASE_MS`** (peer-side DRAM read lease,
  `page_backend.cpp`, default 500ms) — raising this to 30000ms was
  originally believed to fix an earlier UMBP hang. Re-investigated: the stall
  reproduced identically at both 500ms and 30000ms.
  `PageBackend::MaybeEvictToLowWatermark()` is explicitly non-blocking
  (skips lease-protected keys, "must not spin waiting for a reader"),
  so the lease value was never on the path that blocked here.
  Leave it alone; it is not implicated.

- **PR #678** (`fix/umbp-standalone-rpc-deadline`) arms every standalone-process
  data-plane RPC with a client-side deadline (`UMBP_DATA_PLANE_RPC_TIMEOUT_MS`)
  and logs the failure instead of silently degrading to a miss. Worth keeping as
  a bound on a genuinely wedged server, but it treats the symptom. Note its
  default is **300000ms**, not the 10000ms an earlier revision of this document
  quoted — the "10s would fire routinely during a normal warmup burst" concern
  recorded there no longer applies.

## Reproduction recipe (to confirm the fix end to end)

```
CONC=24 ARM=umbp DURATION=900 GPU_WAIT=120 \
  bash k3-dcp8/shared/scripts/ci_dcp8.sh
```

Watch `aiperf.log`'s warmup progress line; a stall shows as `returned`
and `in_flight` frozen for well over a minute right after
`server ready`. While it's frozen:

```
CTR=$(docker ps --format '{{.Names}}' | grep pr39511 | head -1)
for p in $(docker exec "$CTR" pgrep -f scheduler_TP); do
  docker exec "$CTR" py-spy dump --pid "$p" --native
done
```

(`--native` needs the ptrace capability the harness's `docker run`
already grants; drop `--nonblocking` or py-spy refuses native stacks.)

To attribute a stall that still occurs, line the server log's
`[StandaloneServer] registered shm client_id=...` /
`registered GPU IPC client_id=...` timestamps up against the frozen window. If
they cover it, registration is still serializing something.

This reliably reproduced on `crsuse2-m2m-v2-010` and `crsuse2-m2m-v2-015`.
Cold-page-cache weight loading on a fresh node adds 15-25 minutes before the
server is even ready to reach the reproducible window; a node that has already
loaded this model once in the current session gets there in ~5-10 minutes.
