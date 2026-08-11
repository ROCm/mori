# Porting the Tree-Connector Work onto the Backend-Agnostic Branch — Plan

**Goal:** bring the GPU-buffer and ranged-multi-buffer-I/O work from
`origin/feat/umbp-tree-connector` onto the backend-agnostic branch, so a
GPU-resident sglang HiCache connector can drive UMBP without a host bounce.

**Provenance.** Upstream branch `feat/umbp-tree-connector` at `1ab7e751`, nine
UMBP commits on top of the common base `ae9635f7`. Our side is
`integration/umbp-ssd-verify` at `ee4a5861`, ten UMBP commits on the same base
(the Phase 0–6 backend-agnostic refactor, the HBM/SSD `MediumBackend`s, one
medium per node, and the segment CRC work). This port lands on
`feat/umbp-tree-connector-port`, branched from `ee4a5861`.

**Method:** not a merge, and not nine cherry-picks. The two branches rewrote
almost disjoint halves of `src/umbp`, so most upstream files can be taken
verbatim; the small shared set is hand-merged, and the one file where the two
designs actually collide is *reimplemented* rather than ported.

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

**Distributed ranged I/O stays a stub.** Upstream ships
`DistributedClient::BatchGetRanges` / `BatchPutRanges` as warn-once all-false,
and this port keeps that. It is not a regression and not laziness: implementing
it is design work (object-range → page-range mapping, and whether the master's
metadata needs to describe sub-object extents), and doing it inside a port would
make both unreviewable.

Worth recording for whoever picks it up: on our branch this is *much* closer
than upstream's shape suggests. `MediumBackend` has no data-movement virtual at
all — allocate, commit, resolve, evict, publish `TransferRef`s — and every byte
moves as a `TransferItem`, which already carries `src_offset`, `dst_offset` and
`size`. A `TransferItem` *is* a range. So no engine, backend or wire-format
change is needed; what is needed is the mapping in `PoolClient`, and the
whole-object path already computes it with the range pinned to `[0, size)`
(`LogicalPageBytes` at `pool_client.cpp:597`, consumed by
`BuildLocalPageTransfers`). Generalizing it is turning a constant into a
parameter.

Note also that the gating factor is the *backend*, not the client mode. The
standalone server forwards ranged RPCs straight to its inner client, so
standalone-process over a Local backend works while standalone-process over a
Distributed backend inherits the stub. Implementing §5 lights up both
distributed rows at once.

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

§4.4 was folded into the interface commit rather than kept separate: the ranged
pure virtuals and the `RegisterMemory` signature live in the same two files, so
splitting them would have produced a commit that does not compile.

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

Both features, as this branch leaves them.

| Client | Backend | GPU buffers | Ranged I/O |
|---|---|---|---|
| `StandaloneClient` | — | DRAM tier only (§8.5) | DRAM tier only |
| `StandaloneProcessClient` | Local | DRAM tier only (§8.5) | DRAM tier only |
| `StandaloneProcessClient` | Distributed | yes, all media | no — inherits the §5 stub |
| `DistributedClient` | — | yes, all media | no — §5 stub |

Ranged I/O is DRAM-only even within Local mode: `TierBackend`'s default ranged
methods return all-false (`tier_backend.cpp:52`, `:69`) and only `DRAMTier`
overrides them.
