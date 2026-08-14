# Porting the Tree-Connector Work onto the Backend-Agnostic Branch — Plan

**Goal:** bring the GPU-buffer and ranged-multi-buffer-I/O work from
`origin/feat/umbp-tree-connector` onto the backend-agnostic branch, so a
GPU-resident sglang HiCache connector can drive UMBP without a host bounce.

**Provenance.** Upstream branch `feat/umbp-tree-connector`, originally at
`1ab7e751` with nine UMBP commits on the common base `ae9635f7`. Our side is
`integration/umbp-ssd-verify` at `ee4a5861`, ten UMBP commits on the same base
(the Phase 0–6 backend-agnostic refactor, the HBM/SSD `MediumBackend`s, one
medium per node, and the segment CRC work). This port lands on
`feat/umbp-tree-connector-port`, branched from `ee4a5861`.

Upstream has since been rebased onto a newer main and grown three more UMBP
commits; it is now at `875d4ac3` with twelve. The rebase left the original nine
byte-identical (`git range-diff 9d5e30f5..1ab7e751 f185c5f0..362cd22c` reports
`=` for all nine), so the second round of porting is exactly those three
commits and nothing else — see §8.

**Round three.** Upstream is now at `c0044f87` with thirteen, and was *not*
rebased this time (`git range-diff f185c5f0..362cd22c 7f6effc5..c0044f87`
reports `=` for all twelve carried commits), so the third round is exactly one
commit — see §12. This branch was at the same time rebased off `ee4a5861` onto
`c12c68e8`, the backend-agnostic branch's current head, which added the
pure-SSD/multi-drive work; see §12.1 for what that cost.

**Method:** not a merge, and not a series of cherry-picks. The two branches
rewrote almost disjoint halves of `src/umbp`, so most upstream files can be
taken verbatim; the small shared set is hand-merged, and the files where the
two designs actually collide are *reimplemented* rather than ported.

**Scope:** `src/umbp/` (local tier stack, standalone server/client, distributed
client, common helpers, proto), `src/pybind/`, `tests/`.

---

## 1. Why a merge is the wrong tool

A trial merge of the upstream branch into ours reports exactly one conflicted
file. That number is misleading in both directions.

`src/umbp/distributed/pool_client.cpp` conflicts, and it is not resolvable by
picking hunks: upstream's five edit sites are `LocalPutPages`, `LocalGetPages`,
`RemoteDramScatterWrite`, `RemoteDramScatterRead` and `RemoteSsdReadOnce`, all
deleted by Phase 6. `FindRegisteredMemory` survives but now returns
`TransferRef`, not `mori::io::MemoryDesc`.

Three files auto-merge *cleanly but wrongly*. The worst is
`standalone/standalone_process_client.cpp`: our `RegisterMemory` throws when
`loc != CPU`, and upstream's `7af295f6` implements exactly that case over GPU
IPC. Git keeps both, the throw wins, and the feature is dead with no conflict
marker to notice.

So the merge would be silently wrong in the places that matter most.

## 2. What the port does not need to do

Three pieces of upstream work are already satisfied by the backend-agnostic
refactor. Porting them would duplicate the abstraction they generalize.

**GPU endpoint dispatch.** `HbmCopyEngine::CanHandle` claims any pair with at
least one GPU endpoint, derives the `hipMemcpyKind` from the pair, and selects
the device the copy runs on (`hbm_copy_engine.cpp:68–81, 111–118`). That is what
upstream's `DeviceCopy` + `ScopedHipDevice` do at each call site, except ours is
selected by `CompositeTransferEngine` instead of branched on inline. We still
take `common/device_copy.{h,cpp}` — the *local* tier stack needs it, and
`DetectPointerLocation` is needed at the API boundary (§4.1) — but no call site
in `pool_client.cpp` gains a manual `hipMemcpy`.

**Run coalescing.** Both local engines merge adjacent segments while building
the plan (`local_copy_engine.cpp:177–186`, `hbm_copy_engine.cpp:160–163`). This
is upstream's `BuildTierPageRuns` merge generalized: per endpoint pair, across
keys, on host and GPU paths alike, and with no `UMBP_LOCAL_PAGE_MERGE` escape
hatch because it is the only path. The distributed half of `500e186f` is
therefore dropped; its local half is taken.

**Pointer location in the API.** `IUMBPClient::RegisterMemory` already takes
`loc` and `device` and plumbs them to `transfer_engine_->RegisterMemory`
(`umbp_client.h:131–140`, `pool_client.cpp:509–526`). Upstream had to infer
location from the pointer because the base signature could not express it. We
keep our explicit parameters *and* add inference, because a connector written
against upstream calls `register_memory(ptr, size)` with no location — see §4.1.

## 3. File classification

The rule: for any file where `git diff ae9635f7 HEAD -- <file>` is empty, our
branch never touched it, so upstream's final state *is* the correct merge
result. Take it verbatim. Everything else is hand-merged.

**Take verbatim (25 files).** Local tier stack and its new source-private
headers (`dram_tier.{h,cpp}`, `tier_backend.{h,cpp}`, `local_storage_manager.{h,cpp}`,
`device_copy_run.h`, `device_gather.{h,hip}`, `host_registration.{h,cpp}`),
`local/standalone_client.{h,cpp}`, `standalone/standalone_server.cpp`,
`common/device_copy.{h,cpp}`, `common/range_utils.h`,
`proto/umbp_standalone.proto`, and the tests
(`test_dram_tier_{ranges,gather,concurrency}.cpp`, `test_umbp_client.cpp`,
`test_standalone_shm_ipc.cpp`, `test_umbp_client_ptr.py`).

`test_standalone_shm_ipc.cpp` already exists at the base and is already wired in
`tests/cpp/umbp/distributed/CMakeLists.txt`; our branch rewrote that CMakeLists
but kept the entry, so no build wiring is needed for it.

**Hand-merge (9 files).**

| File | Reconciliation |
|---|---|
| `include/umbp/umbp_client.h` | Add upstream's two pure virtuals; keep our four-argument `RegisterMemory`. |
| `pybind/pybind_umbp.cpp` | Add the two ranged bindings next to ours. |
| `src/umbp/CMakeLists.txt` | Add `common/device_copy.cpp`, `local/tiers/host_registration.cpp`, `local/tiers/device_gather.hip` to `UMBP_CORE_SRCS`; add `hip::host` as a PUBLIC link on `umbp_core`. |
| `distributed/distributed_client.{h,cpp}` | Add the two warn-once stubs (see §5). |
| `standalone/standalone_process_client.{h,cpp}` | Take upstream's GPU-IPC registration and ranged forwarding; delete our `loc != CPU` throw. |
| `distributed/pool_client.cpp` | **Reimplemented, not ported** — §4. |
| `tests/.../test_pool_client_batch_put.cpp` | Append upstream's two multi-page byte-exact cases; they drive the public API and exercise our coalescing. |
| `tests/cpp/umbp/local/CMakeLists.txt` | Union of their three dram-tier test targets and our `test_segment_crc`. |

## 4. The four gaps — the whole distributed-side port

Upstream's 200-line `pool_client.cpp` diff exists to close these. In our
architecture they close much smaller, because the transfer layer already carries
pointer location and offsets.

### 4.1 An unregistered device pointer is labelled host memory

`PoolClient::UserBufferRef` (`pool_client.cpp:552–556`) falls back to
`TransferRef::HostBytes(ptr, size)`, which stamps `loc = CPU`. An unregistered
device pointer therefore routes to `LocalCopyEngine`, which `memcpy`s from
device memory.

Fix: classify the pointer in the fallback with `DetectPointerLocation` and build
a GPU `TransferRef` when it is device memory, so `HbmCopyEngine` claims the
pair. This is one function, and it is what makes an unmodified upstream-era
connector — `register_memory(ptr, size)` with no location — correct on our
branch instead of merely accepted.

### 4.2 The remote bounce path stages GPU memory through a host buffer

`MoriIoEngine` decides bounce eligibility on `local.HasHostPtr()` alone
(`mori_io_engine.cpp:403`, `:462`) and then `std::memcpy`s in and out of the
host staging region (`:592`, `:608`). A GPU local endpoint that misses zero-copy
is therefore memcpy'd from device memory.

Fix: require `local.loc == mori::io::MemoryLocationType::CPU` in both
predicates, so a GPU endpoint that cannot be zero-copied is *rejected by Plan*
and surfaces as a per-key failure. One predicate change at each of two sites is
the single-point equivalent of upstream's five scattered "refusing host staging
fallback" guards — and it is strictly better placed: the transfer layer is the
only layer that knows whether staging is involved.

### 4.3 `RegisterMemory` lacks upstream's hardening

Ours returns `true` when a pointer is already registered regardless of the new
size, does not validate the descriptor it got back, and does not catch
(`pool_client.cpp:509–526`).

Fix: port upstream's logic against `TransferRef` — reject a re-registration that
grows the region, validate base/size/loc/device on the returned ref and
deregister on mismatch, and convert throws into `false`.

### 4.4 Standalone-process mode throws on the case upstream implements

`StandaloneProcessClient::RegisterMemory` raises "only CPU (AnonymousShm-backed
host) buffers are supported". Replace the throw with upstream's
`RegisterDeviceMemory` path, keeping our explicit `loc`/`device` parameters and
inferring them when the caller leaves them at the default.

**The server-side half, which is not visible in the client diff:**
`standalone_server.cpp:757` registers the mapped worker region with the
two-argument `client_->RegisterMemory(base, size)`, which on our branch defaults
to `loc = CPU, device = -1`. On the GPU-IPC path the server maps a *device*
handle there, so that call must pass `loc = GPU` and the device ordinal —
otherwise the region is registered as host memory and lands straight back in
§4.1 and §4.2.

## 5. Deliberate non-goals

**Distributed ranged I/O stayed a stub — in round one only.** ~~Upstream ships
`DistributedClient::BatchGetRanges` / `BatchPutRanges` as warn-once all-false,
and this port keeps that.~~ **Superseded by §10.3:** upstream implemented it in
`875d4ac3` and this branch has now ported it. The reasoning below is kept
because it was the design note the implementation was written against, and it
held up — the prediction that "generalizing it is turning a constant into a
parameter" is what the ported code actually does.

The original note: implementing it is design work (object-range → page-range
mapping, and whether the master's metadata needs to describe sub-object
extents), and doing it inside the first round of the port would make both
unreviewable.

Worth recording for whoever picks it up: on our branch this is *much* closer
than upstream's shape suggests. `MediumBackend` has no data-movement virtual at
all — allocate, commit, resolve, evict, publish `TransferRef`s — and every byte
moves as a `TransferItem`, which already carries `src_offset`, `dst_offset` and
`size`. A `TransferItem` *is* a range. So no engine, backend or wire-format
change is needed; what is needed is the mapping in `PoolClient`, and the
whole-object path already computes it with the range pinned to `[0, size)`
(`LogicalPageBytes`, consumed by `BuildLocalPageTransfers`). Generalizing it is
turning a constant into a parameter.

Note also that the gating factor is the *backend*, not the client mode. The
standalone server forwards ranged RPCs straight to its inner client, so
standalone-process over a Local backend works while standalone-process over a
Distributed backend inherited the stub. Implementing this lit up both
distributed rows at once, as predicted.

**Whole-object I/O is not collapsed into ranged I/O.** Upstream added ranged ops
as a parallel path at every layer — `ReadBatchRangesIntoPtr` beside
`ReadBatchIntoPtr`, `BatchWriteRanges` beside `BatchWrite`, a second RPC pair, a
second client method pair. Whole-object I/O *is* the degenerate single-range
case, so those pairs could collapse into one primitive plus a thin adapter,
halving the surface instead of doubling it.

This port keeps upstream's parallel structure. Two reasons: a port that also
redesigns is not reviewable against its source, and the collapse has a real
cost — the whole-object path takes shortcuts a general ranged path cannot (the
DRAM tier's parallel non-temporal write of one contiguous payload per key, and
coalescing that assumes dense user offsets), which become "detect the
single-full-range case and take the fast branch". Recommended as a follow-up,
deliberately out of scope here.

**The three unrelated upstream commits are not taken.** `01a5eaa6`, `3552d318`
and `9d5e30f5` are EPv2 and build fixes that account for every `pyproject.toml`
and `python/mori/ops/` line in the upstream branch diff. They come from main on
their own schedule.

## 6. Stages — as landed

| Commit | Contents |
|---|---|
| `doc(umbp): plan the tree-connector port…` | this document |
| `feat(umbp): take the tree-connector local and standalone half verbatim` | the 24-file take-verbatim set (§3); touches nothing the refactor modified, so it cannot regress the distributed data plane |
| `feat(umbp): wire the ranged I/O interface and GPU-capable standalone registration` | `CMakeLists.txt`, `umbp_client.h`, `pybind_umbp.cpp`, `DistributedClient` stubs, local test CMake, and §4.4 both halves |
| `feat(umbp): accept GPU user buffers in the distributed data plane` | §4.1, §4.2, §4.3 |
| `test(umbp): cover multi-page runs and GPU user buffers through PoolClient` | upstream's two multi-page cases plus two GPU cases neither branch had |
| `test(umbp): assert why the GPU remote-path put fails…` | strengthens the rejection case after measuring the fixture's real routing |
| `doc(umbp): record what the port actually built, ran and found` | §7, §8 |
| `doc(umbp): record the mode support matrix and the local SSD device-buffer hole` | §9 |

§4.4 was folded into the interface commit rather than kept separate: the ranged
pure virtuals and the `RegisterMemory` signature live in the same two files, so
splitting them would have produced a commit that does not compile.

**Round two** (§10), one commit per upstream commit, in upstream order:

| Commit | Contents |
|---|---|
| `refactor(umbp): share host registration and gather helpers` | §10.1 — upstream `93f2998a` |
| `fix(umbp): bound heartbeat event batches` | §10.2 — upstream `1d3c859e` |
| `feat(umbp): support ranged I/O in the distributed data plane` | §10.3 — upstream `875d4ac3`, plus the `HbmCopyEngine` gather path it needs |

## 7. Verification — what was actually run

Built and run in `rocm/mori-dev:…-rdmapush` with `/apps/ditian12` bind-mounted at
the same path, so the existing `build_check/` cache stays valid. The image has
`ENTRYPOINT ["bash"]`, so `--entrypoint bash` is required or the command is
passed to bash as a filename.

- **Build:** `umbp_core` (including `device_gather.hip`, compiled for gfx950 and
  host), `umbp_client`, `umbp_standalone_server`, and every touched test target.
  Clean apart from the pre-existing spdlog `-Wdeprecated-literal-operator`
  warnings.
- **pybind:** there is no cmake target for `src/pybind/pybind_umbp.cpp` — it is
  built by `pip install`, so a plain `cmake --build` never compiles it and a
  typo in a binding would surface only at install time. Checked directly with
  `hipcc -fsyntax-only -std=c++17 -I . -I src/umbp/include -I include -I
  3rdparty/spdlog/include -I build_check/src/umbp/proto_gen -I <pybind11> -I
  <python>`; note `-I .` and the generated `proto_gen` directory are both
  required.
- **ctest:** 22 umbp tests pass (`-E 'cross_node|e2e|medium_selection'`, which
  need a second node). This host has `/dev/kfd` and `/dev/dri` but **no**
  `/dev/infiniband`; the live `PoolClient` tests nonetheless pass here, so the
  known infiniband constraint applies to the multi-node suites rather than to
  these.
- **The gather suite runs twice by design** — `umbp_dram_tier_gather` and
  `umbp_dram_tier_gather_kernel_off` — because the kernel and the copy engine
  must be indistinguishable in their results. Both pass.
- **The four new `PoolClient` cases were confirmed to run, not skip.** The GPU
  cases skip when no HIP device is present, and a skip is not a pass, so they
  were re-run under `--gtest_filter` and observed as `[ OK ]`.
- **The rejection case was measured, not assumed.** Its first version allowed
  either disposition per key; instrumenting it showed all four keys rejected,
  because the fixture pins `caller_` to a single page to force remote routing.
  It now asserts that, plus the `transfer unplannable` WARN (which pins the
  reason to the transfer layer refusing the GPU endpoint) and that no key was
  published. `UmbpLogCapture` forces the module to WARN, which is why that
  message is observable in the test and invisible in a default-level run.
- **Python:** `BUILD_UMBP=ON pip3 install --no-build-isolation .` from a copy of
  the tree, then `pytest tests/python/umbp/test_umbp_client_ptr.py` — 5 passed.
  This is the only check that exercises the two new pybind bindings end to end,
  and `test_batch_put_get_ranges_round_trip_and_shape_failure` is the closest
  thing to a contract test for what the connector calls: a 12-byte object
  assembled from three scattered 4-byte buffers in shuffled offset order, read
  back as two out-of-order sub-ranges, plus a ragged inner vector rejected as a
  whole-call shape error. It needs a full install, so it is not part of the
  ctest run above.

**Not verified here.** The multi-node suites (`cross_node`, `e2e`,
`medium_selection`) need a second node and were excluded. Nothing exercises
GPU-IPC registration in standalone-process mode against a live server — §4.4 is
covered by construction and by review, not by a test — and no measurement was
taken of the GPU path's throughput.

## 8. Follow-ups this port deliberately leaves open

1. Implement `BatchGetRanges`/`BatchPutRanges` for distributed (§5).
2. Collapse whole-object I/O into the degenerate ranged case (§5).
3. `HbmCopyEngine` issues blocking `hipMemcpy` on the null stream. Upstream's
   `549c14d5` found that sharing one stream per device serialized every batch in
   the DRAM tier (1.7 → 4.9 GB/s once split); our engine has the same coupling
   in a different shape. Measure once GPU buffers are live end-to-end.
4. SSD ranged reads will work through the staging buffer as soon as §5 lands,
   but without disk-side savings until `SsdCopyPipeline` learns sub-extent
   reads.
5. **GPU buffers are not supported against the local SSD tier.** Inherited from
   upstream, not introduced here. `SSDTier::ReadIntoPtr` passes the caller's
   pointer straight into `ReadRecordLocked` (`ssd_tier.cpp:301`), which preads
   at that address; neither `ssd_tier.cpp` nor `local_storage_manager.cpp` has
   any device awareness, and `LocalStorageManager::ReadIntoPtrNoPromote` reads
   from the owning tier directly rather than promoting to DRAM first. So in
   Local mode (and standalone-process over a Local backend) a device
   destination works for a DRAM hit and not for an SSD hit. It should fail
   rather than corrupt, but that is untested and the behaviour may differ under
   large-BAR host visibility.

   Note the asymmetry, because it is the refactor paying off rather than an
   accident: distributed mode has no equivalent hole. There every byte moves
   through the transfer layer, so an SSD-resident object reaches a device buffer
   via `SsdBackend`'s staging refs and `HbmCopyEngine` — device-awareness is one
   property of `TransferRef` instead of a per-tier concern. Fixing the local
   side means teaching `SSDTier` to stage, or promoting to DRAM before serving a
   device destination.

## 9. Mode support matrix

Both features, as this branch leaves them (updated after §12).

| Client | Backend | GPU buffers | Ranged I/O |
|---|---|---|---|
| `StandaloneClient` | — | DRAM tier only (§8.5) | DRAM tier only |
| `StandaloneProcessClient` | Local | DRAM tier only (§8.5) | DRAM tier only |
| `StandaloneProcessClient` | Distributed | yes, all media | opt-in (§12), except SSD (§10.3) |
| `DistributedClient` | — | yes, all media | opt-in (§12), except SSD (§10.3) |

Ranged I/O is DRAM-only within Local mode: `TierBackend`'s default ranged
methods return all-false (`tier_backend.cpp:52`, `:69`) and only `DRAMTier`
overrides them.

In distributed mode two independent conditions must both hold, and
`SupportsRangedIO()` tests both, so a client reports the truth rather than the
caller discovering it as a failed batch. The medium must be able to serve it —
ranged access maps object ranges onto pages a backend publishes as *in-process*
endpoints, and `SsdBackend` publishes storage refs — and the deployment must
have opted in by configuring a ranged scratch arena (§12), which defaults to
zero.

A client can now ask instead of assuming: `GetBackendMode()` reports what is
behind a forwarding client, and `SupportsRangedIO()` reports the capability.
Both are exposed to Python (`get_backend_mode`, `supports_ranged_io`).

---

## 10. Round two — upstream's three later commits

Ported after upstream's rebase. In upstream order:

### 10.1 `93f2998a` — share host registration and gather helpers

Moves `HostTierRegistration` and `LaunchDeviceGather` from DRAM-tier-private
headers into `src/umbp/include/umbp/common/`, leaving forwarding shims at the
old paths. Our branch had never touched those four files, so this cherry-picked
clean.

One deliberate deviation: upstream's move rewrote both headers' comments down to
a few lines. This branch keeps the original rationale — the measured
per-fragment costs that justify a kernel over `hipMemcpy` / `hipMemcpy2DAsync`,
the ~128 KiB crossover, and why registration is single-shot and all-or-nothing.
Those numbers are why the code has the shape it has, and §10.3 depends on the
crossover being written down.

### 10.2 `1d3c859e` — bound heartbeat event batches

A real bug fix, not a feature: a large offload burst shipped every unacked
bundle in each heartbeat, so once a request exceeded the RPC deadline nothing
was acked and the next heartbeat resent the backlog plus new events. Upstream
saw 921 `DEADLINE_EXCEEDED` failures on an 8-rank run.

`master_client.cpp` took upstream's change as-is; the only reconciliation is
that the drain reads `DrainAllBackends(Backends())` rather than the deleted
`DrainAllSources(owned_sources_)`.

The **test** is the interesting half, and a good example of the §1 hazard.
Upstream's fake implements `OwnedLocationSource` and attaches via
`AddOwnedLocationSource` — both deleted by Phase 3. The cherry-pick auto-merges
without a conflict marker and does not compile. It was rebuilt to drive a
`MockBackend` (a full `MediumBackend` already in the tree) through a
`BackendRegistry`, which exercises the same `DrainAllBackends` path production
uses. The registry is attached *after* the commits that build the backlog, since
`SetBackendRegistry` installs the auto-flush hook that would otherwise drain it
before it accumulates.

### 10.3 `875d4ac3` — ranged I/O in the distributed data plane

The one that retires §5. §5's prediction held: the object-range → page-range
mapping is one function, `BuildLocalRangeTransfers`, and no backend, engine or
wire-format change was needed.

What upstream's 780-line `pool_client.cpp` diff spends its length on, and why
none of it is here: a `RangeCopyRun` planner with adjacency merging, a
per-endpoint `hipMemcpyKind`, `ScopedHipDevice` at each call site, and a manual
gather/`hipMemcpyAsync` fork. Every one of those is a thing the transfer layer
already does — §2's three "already ours" items, applied to a second feature.

Four differences worth reviewing against the source:

1. **A locally-routed ranged put needs no assembly buffer.** Upstream assembles
   scattered ranges into the scratch arena and writes the object; here the
   ranges are written straight into the slot's pages, because a `TransferItem`
   carries offsets. The arena is needed only for the remote direction, where the
   object must be contiguous on the wire.
2. **A remote object is served from the arena, and the local install is a
   caching side effect.** Upstream installs into the tier first and serves
   through it, so an install failure loses a fetch it already paid for. The
   install-failure metric is kept and now counts exactly what its name says: a
   missed caching opportunity, not a failed read.
3. **No DRAM/HBM tier filter on remote ranged reads.** Upstream needs one
   because its remote path is a hand-written DRAM RDMA read. Ours fetches the
   whole object through `ExecuteBatchGetPlan`, which reaches any medium the
   owning peer publishes.
4. **Per-key results instead of all-or-nothing.** Items are tagged with the
   caller's key index, so the engine's failed tags map back per key.

**Where the gather kernel went, and why.** Upstream calls `LaunchDeviceGather`
from `pool_client.cpp`. Here it lives in `HbmCopyEngine`, for the §4.2 reason:
the engine is the only layer that sees the segment shape a copy decomposes
into, so it is the only one that can decide when a kernel beats `hipMemcpy`.

The load-bearing detail is that bucketing is **per device across the whole
batch, not per plan**. A plan is one `(src base, dst base)` pair, so a caller
reading three ranges into three separate GPU allocations produces three
single-segment plans — nothing to gather within any of them, and together
exactly the scattered small-fragment batch the kernel wins on. Getting this
wrong is not a correctness bug, which is why it is recorded: the first version
bucketed per plan, produced correct bytes, and silently never launched a kernel.
The same insight is why the local ranged put batches across keys — one kernel
for the batch rather than one per key.

Consequence beyond ranged I/O: `PoolClient` declares the live medium's host
buffers and the scratch arena as gather regions, so *every* GPU transfer in the
tree gets the fast path, not just ranged ones.

**The test** is upstream's `test_pool_client_ranges.cpp` rebuilt against this
architecture: the fixture drives `BackendRegistry` / `MediumBackend` instead of
the deleted `PeerDramAllocator`, and the backend self-allocates its pool
(Phase 2b) so the test passes a size rather than a buffer. All four cases pass,
including the two that assert the gather kernel actually ran — which is what
caught the per-plan bucketing mistake.

---

## 11. Verification — round two

Same container and constraints as §7.

- **Build:** whole tree clean, plus the `pybind_umbp.cpp` syntax check (still
  not covered by any cmake target — see §7 for the exact invocation).
- **ctest:** `-R umbp -E 'cross_node|e2e|medium_selection'`, 22 of 23 pass.
  `umbp_pool_client_ranges` is new and passes all four cases; this host needs
  `--privileged --device=/dev/infiniband` for it, unlike the suites §7 ran.
- **One failure, pre-existing and unrelated:** `umbp_local_client` aborts on
  `test_gpu_put_get`'s `batch_a == host_a` when `umbp_host_mem_allocator` has
  run first in the same ctest invocation. It passes standalone, and it fails
  identically at `b1176b61` — the commit *before* any round-two data-plane work
  — so it is not a regression from this round. It is a cross-test interaction
  through shared host state, and it is worth chasing separately: the two tests
  are separate processes, so the coupling has to be the filesystem or hugepages
  rather than anything in the library.

---

## 12. Round three — rebase onto `c12c68e8`, and upstream's one later commit

Two independent movements, done together because the second lands on top of the
first.

### 12.1 The rebase — `ee4a5861` → `c12c68e8`

The backend-agnostic branch grew one commit, `c12c68e8` ("port the pure-SSD /
multi-drive work onto the backend-agnostic SSD path", #554): `ShardedSSDTier`
and its factory, O_DIRECT with the 4 KiB-aligned record format v3, hardware
CRC-32C, `PeerSsdManager` single-flight, and a `parallel_for` helper.

Replaying the twelve port commits over it produced **one conflict, in one file,
of one line**: `pool_client.cpp`'s include block, where `c12c68e8` added
`umbp/common/parallel_for.h` and the ranged commit added
`umbp/common/range_utils.h`. Both were kept. `git range-diff` over the twelve
shows only context shifts — no hunk was rewritten and no resolution changed
anyone's semantics.

That is worth recording rather than just being lucky, because it is §1's
argument holding a second time from the other side: the SSD work is a
`MediumBackend` and a local tier, the tree-connector work is the client-side
data plane and the DRAM tier, and the two barely intersect.

Two interactions were checked rather than assumed:

- **`ShardedSSDTier` inherits the ranged stubs, not the DRAM overrides.** It
  derives from `TierBackend`, whose default ranged methods return all-false
  (§9), so the new sharded tier reports exactly what the single-drive tier
  reports. §8.5's local-SSD device-buffer hole is unchanged by it — wider, in
  that it now covers N drives instead of one, but not different.
- **The new SSD tests build and pass under the port's CMake wiring.**
  `test_sharded_ssd_tier`, `test_ssd_direct_io`, `test_segment_crc` and
  `test_peer_ssd_single_flight` were the risk, because the port rewrote both
  test `CMakeLists.txt` files. See §13.

### 12.2 `c0044f87` — make the distributed ranged scratch opt-in

The arena defaulted to 256 MiB, so every distributed client allocated and
RDMA-registered it whether or not it ever issued a ranged operation. Zero is now
the default and means "this deployment does not do ranged I/O": no allocation,
no registration, `SupportsRangedIO()` false. `UMBPConfig::Validate` no longer
rejects zero, and neither does `UMBP_DISTRIBUTED_RANGED_SCRATCH_BYTES`.

Two deviations from upstream, both because our side already differs at the site:

1. **`SupportsRangedIO()` gains the arena test rather than being replaced by
   it.** Upstream's is now exactly `ranged_scratch_size_ > 0`; ours keeps the
   medium test as well (§9). The conditions are independent and neither implies
   the other — the arena is what the *remote* direction lands in, and the medium
   is what decides whether object ranges can map onto published in-process
   endpoints at all. Upstream has no medium selector, so it cannot express the
   second half.

2. **`StandaloneServer`'s `shared_reads_` keeps our predicate.** Upstream now
   gates concurrent standalone reads on `SupportsRangedIO()`. Taken verbatim
   here that would serialize *every* read of a distributed SSD-medium
   deployment — the pure-SSD mode §12.1 just merged in — because our
   `SupportsRangedIO()` also excludes that medium. The gate buys nothing
   either: the RPCs under that lock are whole-object reads, and the only two
   paths that touch the arena (`BatchGetRanges` / `BatchPutRanges` in
   `pool_client.cpp`) already serialize on `ranged_scratch_mutex_` and bail out
   cleanly when it is absent.

Upstream's cleanup paths null the DRAM pool alongside the arena and guard the
frees with `if (ranged_scratch_)`. Neither carries over: the DRAM backend has
owned its own pool since Phase 2b, and `HostMemAllocator::Free` is already a
no-op on an unallocated handle (`host_mem_allocator.cpp:452`), so `Close()` and
the constructor's `release_scratch` are correct unchanged.

The test is upstream's, adapted to drive a real `DistributedClient` through
`CreateUMBPClient` rather than the file's bare `PoolClient` fixture — the opt-in
lives in the constructor and in `SupportsRangedIO()`, neither of which
`PoolClient` sees.

**Consequence for a connector.** Ranged I/O is now off by default in
distributed mode. A GPU-resident HiCache connector that wants it must set
`distributed.ranged_scratch_size` (Python: `ranged_scratch_size` on
`UMBPDistributedConfig`; `UMBP_DISTRIBUTED_RANGED_SCRATCH_BYTES` for the
standalone server) and should read `supports_ranged_io()` back rather than
assume it took effect.

---

## 13. Verification — round three

Same container as §7 (`rocm/mori-dev:…-rdmapush`, `--entrypoint bash`,
`/apps/ditian12` bind-mounted at the same path), on a host with `/dev/kfd`,
`/dev/dri` and `/dev/infiniband`.

- **Build:** whole tree clean, incremental over the existing `build_check/`.
- **pybind:** `pybind_umbp.cpp` still has no cmake target, so it was
  syntax-checked directly again — see §7 for the invocation. Clean apart from
  the pre-existing spdlog `-Wdeprecated-literal-operator` warnings.
- **ctest: the whole suite this time, not the `-R umbp` subset.** That filter,
  used in §7 and §11, silently skips 18 targets whose names begin `test_` —
  including every SSD target `c12c68e8` added, which are exactly the ones this
  round needed to see. With `-E 'cross_node|e2e|medium_selection'` only,
  **46 of 47 pass**: `test_sharded_ssd_tier`, `test_ssd_direct_io`,
  `test_segment_crc` and `test_peer_ssd_single_flight` among them, and
  `umbp_pool_client_ranges` passing all five cases including the new opt-in one.

**The one failure is `umbp_local_client`, and §11's account of it was wrong.**
§11 recorded it as an ordering artefact that "passes standalone". Measured
properly — 12 runs of the binary from a neutral working directory — it aborts on
`test_gpu_put_get`'s `batch_a == host_a` in 4 of 12 runs at HEAD, with
`UMBP_DRAM_GATHER_KERNEL=0` as well as with the kernel on. So it is **flaky, not
order-dependent, and not the gather kernel**; the single-item `Put`/`Get`
immediately above it never fails, only the batched device path does.

It is not a regression from this round. The same measurement against
`99fa37d5` — the pre-rebase branch head, built from its own worktree — gives 4
of 12 on the identical assertion. Same test, same rate, before any of this
round's work.

(That comparison build also fails `test_batch_get_follower` in the runs that get
past the GPU test, which HEAD's `build_check` never does. That is a property of
the throwaway build directory, not of the commit: follower mode depends on
on-disk state the fresh tree does not have. It is noted only so the raw counts
above are not mistaken for a second finding.)

Chasing the `test_gpu_put_get` flake belongs with the round-one verbatim import
that introduced it, and wants a look at the DRAM tier's batched device path
rather than at anything ranged.
