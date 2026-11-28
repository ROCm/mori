---
name: review-mori-pr
description: Review a MORI pull request with deep expertise in GPU-initiated RDMA (IBGDA on mlx5, bnxt, and ionic NICs), XGMI P2P and SDMA, MORI-SHMEM, MORI-EP dispatch/combine, MORI-IO, CCO/CCL collectives, the JIT, and UMBP. Use when asked to review a MORI PR or check a MORI change.
argument-hint: [PR-number]
context: fork
agent: general-purpose
allowed-tools: Bash(gh pr view *), Bash(gh pr diff *), Bash(git log *), Bash(git show *), Bash(git diff *), Bash(git blame *), Bash(head *), Bash(grep *), Read, Grep, Glob
---

# MORI PR Review

## IMPORTANT: Do NOT post comments

This skill is read-only. Do NOT post any comments, reviews, or reactions to GitHub.
Do NOT use `gh pr comment`, `gh pr review`, `gh api` with POST/PUT/PATCH/DELETE,
or any other write operation. Only return your findings as text output — the caller
is responsible for posting them.

## Fetching Pull Request Context

Before reviewing, fetch the PR context by running these commands:
- `gh pr view $ARGUMENTS` — PR title, description, author
- `gh pr diff $ARGUMENTS --name-only` — list of changed files
- `gh pr diff $ARGUMENTS` — full diff
- `gh pr view $ARGUMENTS --comments 2>/dev/null || echo "(no comments yet)"` — existing review comments

MORI has no `AGENTS.md`. Its contracts are written next to the code they govern — comments such as "Append-only", "Keep in sync with", and "must", and `static_assert`s on sizes and field counts — so read the code around every changed hunk, not only the hunk. If the PR touches an area that has a skill in `.claude/skills/`, read that skill too: `cco-sdma-api` (the `ccoSdma` device API), `umbp-add-backend` and `umbp-add-transfer-engine` (UMBP extension contracts), and `known-issues` (environment problems that look like MORI bugs; do not attribute those to the PR).

## Your Task

Review the PR above thoroughly using the checklist below. Be specific: cite file paths and line numbers from the diff. Describe each finding clearly and explain why it matters. Do NOT assign severity labels (no BLOCKER/WARNING/NIT) — let the human reviewer judge importance.

**CRITICAL SCOPE RULE**: Only flag issues that exist in the PR diff itself — lines added or modified by this PR. Do NOT flag pre-existing code that the PR did not touch, even if that code is in the same files. If a pre-existing problem is worth noting, mention it briefly in a separate "Pre-existing issues (out of scope)" section at the end, never as a BLOCKER or WARNING against this PR.

**PRACTICALITY RULE**: Every finding is one of three kinds, and each kind has its own bar. Every fix adds code that someone has to read and maintain, so a finding has to be worth more than the code it asks for.

**1. Behavioral findings** — bugs, hangs, wrong results, missing handling, performance regressions, and compatibility breaks. This covers sections 1–10. Report such a problem only if it can happen with the code as it stands after this PR — through an existing caller, a supported configuration, or inputs that real workloads produce — and only if the problem outweighs the size and complexity of its fix.

- **Check reachability before reporting.** Use Grep / Read to find the callers of the changed code and the NICs, GPU architectures, kernel types, dtypes, and environment settings that actually reach it. If you cannot name a concrete path that triggers the problem today, drop the finding.
- **No future-proofing requests.** Do not ask for handling of cases no caller produces, defensive checks for invariants that upstream code already guarantees, generalization to dtypes / kernel types / NICs the change does not target, or extension points nothing uses yet. "This would break if someone later..." is not a finding. Two exceptions. Every NIC in the support matrix (mlx5 ConnectX, bnxt Thor2, ionic Pollara AINIC) and every GPU architecture the wheels are built for (`MORI_GPU_ARCHS='gfx942;gfx950;gfx1250'` in `nightly.yml`) is a supported configuration, so a change that lands for one provider or architecture without its counterparts is reachable today, even though no CI runner has a bnxt NIC. And the downstream users in section 10 are existing callers.
- **Weigh the fix against the risk.** A silent wrong result or a hang on a reachable path is worth reporting even if the fix is large. A rare failure that is loud and easy to diagnose is not worth a multi-line guard. If the fix would add more code or conceptual weight than the realistic risk justifies, drop the finding.

**2. Convention findings** — the items in section 12. These have no triggering path by nature, so the reachability test does not apply. Report them whenever they appear in lines the PR adds or modifies.

**3. Test-coverage findings** (section 11). Tests are code too, so the same economy applies.

- **Report a missing test when the main behavior the PR adds or changes is not tested at all, or is tested only by a file no CI job runs.** Examples: a new kernel variant with no test, a bug fix with no regression test, or a test file added but not wired into a workflow. One test that pins down the change is enough.
- **Ask for a corner-case test only when the PR adds code specifically to handle that corner case.** Do not ask for tests of inputs the code cannot receive, for every combination of NICs / dtypes / EP sizes, or for more tests of behavior that is already covered.

**Fewer, real findings beat many plausible ones.** An empty Findings list is a valid outcome for a sound PR. Do not move dropped findings into another section. Items in "Pre-existing issues (out of scope)" must pass the behavioral bar too, and pre-existing convention violations are not listed at all.

The checklist below lists what to look at, not what to demand. An item becomes a finding only when it applies to this PR and clears the bar for its kind.

---

## MORI Review Checklist

### 1. Device-Side RDMA (IBGDA) and NIC Provider Parity
- A new or changed provider template declared in `include/mori/core/transport/rdma/device_primitives.hpp` (`PostSend`, `PostReadWrite`, `PostWriteInline`, `PostAtomic`, `RingDoorbell`, `UpdateSendDbrRecord`, `UpdateDbrAndRingDbSend/Recv`, `PollCq*`) needs its specialization in all three of `providers/mlx5/mlx5_device_primitives.hpp`, `providers/bnxt/bnxt_device_primitives.hpp`, and `providers/ionic/ionic_device_primitives.hpp` (`ProviderType::PSD`). A missing or divergent one fails only when that NIC is selected.
- Doorbell order in every provider's `UpdateDbrAndRingDbSend/Recv`: take the lock, update the doorbell record, `__threadfence_system()`, then `RingDoorbell`. Dropping or reordering the fence lets the NIC fetch a stale WQE.
- `postIdx` / `dbTouchIdx` / `doneIdx` are free-running `uint32_t` serials that wrap at 2^32 (comment at the top of `device_primitives.hpp`, Issue #626). Compare them by forward difference, never by magnitude, never mix them with `uint64_t`, and use `AtomicMaxSerial` instead of `__hip_atomic_fetch_max`. Narrower hardware counters (ionic's 24-bit MSN, bnxt's 16-bit CQE index) must be truncated to their width, not stored raw.
- `quiet` and fence must not return before the caller's own work requests complete: they wait until `doneIdx` reaches the `postIdx` the caller observed on the QP it posted to, not for any completion.
- WQE and CQE fields are big-endian and go through `HTOBE32` / `HTOBE64` / `BE32TOH` (`core_device_types.hpp`); a plain store into a WQE field is a bug. Segment layouts carry `static_assert`s on their sizes, which must still hold.
- `core::WcStatus` mirrors `ibv_wc_status`, checked value by value with `MORI_WC_STATUS_PARITY` in `src/application/transport/rdma/rdma.cpp`.
- Device code picks the provider at compile time: `MORI_DEVICE_NIC_BNXT`, `MORI_DEVICE_NIC_IONIC`, and mlx5 in the `#else` (`include/mori/shmem/shmem_ibgda_kernels.hpp`; `CCO_GDA_BUILD_PROVIDER` in `include/mori/cco/cco_scale_out.hpp`). `IONIC_CCQE` selects a different `PollCq<ProviderType::PSD>`. Every branch must still compile. CMake gates them with `USE_BNXT` / `USE_IONIC` (default OFF) and `MORI_IONIC_CCQE` (`cmake/MoriDetectDevice.cmake`); the JIT detects them in `python/mori/jit/config.py` and `python/mori/jit/core.py`. A kernel built without `-DIONIC_CCQE` against a CCQE-enabled library polls a collapsed CQ as a ring and hangs, so host and device must agree.

### 2. SHMEM Host Runtime and Transport Selection
- `MORI_ENABLE_SDMA`, `MORI_DISABLE_P2P`, `MORI_ENABLE_HOST_PROXY`, and rail-only mode are read once in the `Context` constructor (`src/application/context/context.cpp`). Later code uses `Context::IsSdmaEnabled()` / `IsP2PDisabled()` / `IsProxyEnabled()` / `IsRailOnly()` or `ShmemSdmaEnabled()`, never `getenv`; otherwise transport selection and per-allocation decisions can disagree, which hangs. Per-peer transport precedence is in `Context::DefaultPolicyResolve`.
- `include/mori/shmem/shmem_api.hpp` includes only `application_device_types.hpp` when compiled by hipcc. A header that device code includes must not pull in `application.hpp` or other host RDMA headers (`verbs.h`, `mlx5dv.h`).
- `ShmemSymmetricRegister(ptr, size, rdmaRegister=false)` skips the RDMA memory region (ionic's single-MR limit). It is only valid for buffers no RDMA path ever targets.
- GPU memory registration uses one mechanism per buffer — dma-buf or peer memory, never both (see `MORI_MLX5_DMABUF`) — and a sub-allocated buffer registers its allocation base plus offset. A wrong dma-buf offset shows up as local protection errors on ionic.
- The host-proxy transport exists only under `MORI_PROXY_ENABLED` (`DISPATCH_TRANSPORT_TYPE` in `include/mori/shmem/shmem_device_api.hpp`); the build without it must still compile.

### 3. Device Bitcode ABI and JIT Caches
- A device API that Triton or FlyDSL kernels call needs three matching pieces: the C++ API, an `extern "C" mori_shmem_*` wrapper in `include/mori/shmem/shmem_device_api_wrapper.hpp` and `src/shmem/shmem_device_api_wrapper.cpp`, and an entry in `MORI_DEVICE_FUNCTIONS` (`python/mori/ir/ops.py`), from which the Triton and FlyDSL wrappers are generated. `SIGNAL_SET` / `SIGNAL_ADD` in `ops.py` must equal `AMO_SIGNAL_SET` / `AMO_SIGNAL_ADD` in `core_device_types.hpp`. The CCO equivalents are the wrappers in `src/cco/device/cco_device_wrapper.cpp` and the symbol tags in `python/mori/cco/device/flydsl/_bindings.py`.
- A new `-D` that changes emitted device code must come from `_tunable_defines()` (`python/mori/jit/core.py`). `get_cache_dir()` (`python/mori/jit/cache.py`) hashes that list into the cache key; a flag passed any other way makes one configuration load another's cached binary (the comment in `cache.py` describes the incident).
- Device bitcode is cached under a content hash of fixed `source_paths`: `python/mori/ir/bitcode.py` hashes `src/shmem/shmem_device_api_wrapper.cpp`, `include/mori/shmem`, and `include/mori/core`; `python/mori/cco/device/bitcode.py` hashes the CCO wrapper, `include/mori/cco`, and `include/mori/core`. Those sources also include `include/mori/application/application_device_types.hpp`, `include/mori/utils/*.hpp`, and `include/mori/hip_compat.hpp`, which are not hashed. If the PR changes device-visible code in an unhashed header, or adds an include from outside the hashed paths, users' `~/.mori/jit` keeps serving the old bitcode unless `source_paths` grows to cover it.
- JIT v2 (`src/jit/v2/compiler.cpp`) keys on the kernel name, toolchain signature, NIC, flags, `IncludeTreeHash` of the source dependencies, and the rendered source text. A codegen input that reaches neither the rendered text nor the dependency list reuses a stale `.hsaco`.
- Everything the JIT compiles at run time must ship in the wheel: copied by `_copy_jit_sources()` in `setup.py` and matched by a `_jit-sources/...` pattern in `mori_package_data`. A file missed there works from a source checkout and fails only from a wheel.

### 4. MORI-EP (Dispatch / Combine)
- `EpDispatchCombineConfig` (`include/mori/ops/dispatch_combine/dispatch_combine.hpp`) is mirrored in the pybind constructor and `def_readwrite` list (`src/pybind/pybind_ops.cpp`) and in `ToPackedI32Array` / `FromPackedI32Array` (`src/ops/dispatch_combine/dispatch_combine.cpp`). A new field needs `kPackedI32Len` bumped, both pack bodies updated, and the pybind argument; a change to the packed layout bumps `EP_CONFIG_I32_VERSION`.
- `EpDispatchCombineArgsRaw` and `EpDispatchCombineArgs<T>` must keep identical layouts (`static_assert` in `dispatch_combine.hpp`): a field goes into both at the same position, including the fields under `ENABLE_PROFILER` and `ENABLE_STANDARD_MOE_ADAPT`.
- The v2 schemas in `include/mori/ops/dispatch_combine_v2/ep_cfg.hpp` are count-locked: `kEpArgsFieldCount` with the `MORI_EP_ARGS_FIELDS` list that the ctypes binding is built from, and `MORI_JIT_ASSERT_FIELD_COUNT(EpCfg, ...)` with `VisitFields(EpCfg)`. `EpDType` values must equal `plan_api.DTYPES`.
- Enum values cross languages: `KernelType` / `QuantType` ↔ the pybind enums `EpDispatchCombineKernelType` / `EpDispatchCombineQuantType` ↔ `_QUANT_TYPE_MAP` (`python/mori/ops/dispatch_combine.py`) ↔ `_QUANT_ENUM_TO_CONFIG_STR` (`python/mori/ops/tuning_config.py`). A quant type missing from the tuning map silently falls back to untuned geometry.
- Kernel symbol names are assembled from strings in `src/ops/dispatch_combine/launch.cpp` and again in `python/mori/ops/dispatch_combine.py` ("Keep in sync with the Python launch path"). A new variant, dtype, or codec needs both, and the symbol must exist for every kernel type that can reach it.
- Logic is duplicated per kernel type (`IntraNode`, `InterNode`, `InterNodeV1`, `InterNodeV1LL`, `AsyncLL`, `IntraNodeLL`; mostly one `.hip` each in `src/ops/kernels/`) and again in v2 (`src/ops/dispatch_combine_v2/`, with FlyDSL and HIP backends). A fix to one variant usually applies to its siblings; check them.
- Limits: `MAX_EXPERTS_PER_TOKEN`, and in v2 `EpCfgIsValid` (receive capacity covering `worldSize * maxTokPerRank`, `numExpertPerToken < waveSize`, 16-byte token alignment, and `EpXdbFlagSlots`, which caps the combine block count). New block or warp geometry must stay inside them.
- Work split across blocks takes its slot from the chunk it is processing (e.g. `bid % numRecvBlock`), not from a fixed block id; otherwise block counts that are not a multiple of the chunk count process tokens twice.
- Two things break HIP graph capture without showing up in eager runs (`tests/python/ops/dispatch_combine_v2/test_graph_capture.py`): a host-device sync inside the op, and a cross-device barrier epoch that does not advance on device (dispatch ends with `atomicAdd(xdbFlag, 1)`); capture freezes a host-side value, and every replay after the first spins or passes vacuously. Grid barriers are reset on device by the last arriving block. Launches use the raw current stream (`_current_stream()` in `dispatch_combine.py`); under capture, torch's packed stream id launches on a garbage stream.
- Tuning configs are `python/mori/ops/tuning_configs/{arch}_{model}_{Kernel}_ep{n}_{dispatch|combine}.json`. The loader in `tuning_config.py` checks `"version"` and the required rule fields `_DISPATCH_RULE_REQUIRED` / `_COMBINE_RULE_REQUIRED`.

### 5. MORI-IO
- `EngineDesc` and `MemoryDesc` (`include/mori/io/common.hpp`) and the control-plane messages in `src/io/rdma/protocol.hpp` are msgpack wire formats. They are append-only ("new fields MUST go at the end"), with defaults that let a message from an older peer decode. A new `MemoryDesc` field also goes into `operator==`, which deliberately leaves out `ipcHandle` and `fabricHandle`.
- `BackendType` values (`include/mori/io/enum.hpp`) are on the wire and are `BackendBitmap` bit positions, so they are never renumbered. A new backend appends a value and needs the pybind enum (`src/pybind/pybind_io.cpp`), `CreateBackend`, and `SelectBackend` (`src/io/engine.cpp`).
- `Backend` virtuals (`include/mori/io/backend.hpp`) are implemented in `src/io/{rdma,xgmi,ofi,fabric,sim}`. `IOEngine` fans `RegisterMemory`, `DeregisterMemory`, and `RegisterRemoteEngine` out to every backend, so a backend that silently no-ops one of them breaks routing.
- `RdmaBackendConfig` fields are mirrored in the pybind constructor and `def_readwrite` list and in `python/mori/io/engine.py`.
- A transfer reports a terminal status only after both its data work requests and its notification settle (`CqCallbackMeta::awaitsNotification`, `src/io/rdma/common.hpp`); reporting earlier hands the caller a buffer the NIC still reads or writes. The `wr_id` zones (notification RECV `[0, notifPerQp)`, ledger records `[notifPerQp, 2^63)`, notification SEND with bit 63 set) must not overlap. `TransferStatus::Update` keeps the first error.
- An RC QP that took an error completion stays in the error state: session and endpoint lookups go through `GetLiveEndpoints` / `Alive()` rather than reusing a cached session.
- Per-vendor QP limits (`src/io/rdma/backend_impl.cpp`): ionic caps `max_qp_wr` at 32767 and `max_sge` at 2; bnxt rejects `IBV_SEND_INLINE` unless the QP reserved inline space; mlx5 sizes every WQE for its largest operation, so the default 8192 send WRs exceed its send-queue cap (CI sets `MORI_IO_QP_MAX_SEND_WR=4096` on MI300X_MLX). A new default WR count, inline size, or SGE count must fit all three, and large or multi-SGE transfers are chunked to the device's `max_sge` and message size.
- A map keyed by QP number includes the device context (`(ibv_context*, qpn)`): QP numbers repeat across NICs.
- Listening sockets bind port 0 and read the bound port back. Probing with `bind(0)` + `close` and binding again races with other processes (EADDRINUSE).
- XGMI IPC handles must not depend on the exporting process's HIP ordinal, and a sub-allocated region carries its offset (`MemoryDesc::ipcOffset`).
- Fault catalog (`tests/cpp/io/fault_catalog/*.yaml`): entries with `known_bug` / `fails` are expected failures, and CI fails when one unexpectedly passes. A PR that fixes a known bug deletes that entry's `known_bug` and `fails` lines; a new entry that cannot pass yet needs them. The build with `ENABLE_IO_FAULT_INJECTION=OFF` must still compile.

### 6. CCO, CCL, and SDMA
- `BUILD_CCO_SDMA` must agree across `src/cco/CMakeLists.txt` (a PUBLIC define), the `#else` branch of `ccoSdmaSetupCommQueues` (`src/cco/cco_init.cpp`), the gated block in `src/cco/device/cco_device_wrapper.cpp`, the `test_sdma_` skip in `tests/cpp/cco/CMakeLists.txt`, and the `_build_flags.py` that `setup.py` writes and `python/mori/cco/device/bitcode.py` reads. Only the `build` job's `BUILD_CCO_SDMA=OFF pip install .` step compiles the OFF path.
- `python/mori/cco/cco.pxd` restates `ccoDevComm`, `ccoDevCommRequirements`, `ccoSdmaContext`, and the other device structs field by field. Changing one of them in `include/mori/cco/cco.hpp` without the `.pxd` silently corrupts the Python side. The defaults in `cco.pyx` track `CCO_DEV_COMM_REQUIREMENTS_INITIALIZER`, and a change to `ccoDevCommRequirements` bumps `CCO_API_VERSION` (checked with `CCO_API_MAGIC` in `ccoDevCommCreate`).
- `ccoProviderType` ↔ `core::ProviderType` is pinned with `static_assert`s in `cco_init.cpp`; a new provider updates both enums and `CCO_GDA_BUILD_PROVIDER`.
- An SDMA ring is never filled completely: `wptr == rptr` reads as empty and every waiter hangs (`CanWriteUpto` in `cco.hpp`). Signal slots are indexed by the sender's LSA rank, not its world rank. `ccoSdmaThreadSameQueue` requires every lane to share one (peer, queue). A commit takes ownership through the `CCO_SDMA_RINGING` compare-and-swap and must not wait on another lane.
- SDMA queues and engines are identified by KFD node, not by HIP ordinal.
- A HIP graph records kernel arguments once, so anything that must change per call (generation counters, flags) advances on device, and host stream or device syncs are illegal during capture. When a host sync is removed for capture, the kernel has to restore cross-GPU visibility itself (system-scope loads, `__threadfence_system()`), as the `capturing` path of `AllgatherSdma` does.
- VMM windows (`src/cco/cco_init.cpp`): the granularity query and `hipMemCreate` use the same allocation type; `hipMemAllocationTypeUncached` stays behind its `HIP_VERSION` guard; a failed probe clears the sticky `hipGetLastError()`; ranks are LSA-major contiguous with a uniform `lsaSize` (a fatal contract). Known ROCm 7.0–7.2.x VMM problems are described in `.claude/skills/known-issues/SKILL.md`.

### 7. UMBP
- Protos in `src/umbp/distributed/proto/` stay mixed-version compatible: never renumber or reuse a field number (respect `reserved`), new fields and enum values default to the old behavior, and new behavior is gated on the `supports_*` capability flags in `RegisterClientResponse` (`umbp.proto`).
- A new storage medium or transfer engine follows `.claude/skills/umbp-add-backend/SKILL.md` or `.claude/skills/umbp-add-transfer-engine/SKILL.md`; a missed step fails at neither build nor init. Transfer-engine `CanHandle` claims must be disjoint.
- Heartbeat events from all backends go into one bundle under one monotonic sequence number, never one per medium, and `SnapshotOwnedKeysForFullSync()` drains the outbox in the same critical section (`src/umbp/include/umbp/distributed/peer/backend/medium_backend.h`).
- A staged medium that cannot serve a key right now returns `ResolveOutcome::kBusy`, not `kMissing`: the client excludes a node that answered missing even though it holds the key.
- Buffers are addressed by `(backend_id, buffer_index)`. Every backend numbers its buffers from 0, so an index alone reads the wrong memory on a node with several media.
- Locking (`src/umbp/include/umbp/distributed/pool/peer_pool.h`): backend batch commit and resolve run with `operation_mutex_` released, protected by `migrating_keys_` / `draining_sources_`. No backend I/O runs under `operation_mutex_` or a staging-arena lock, and staging spans a reader still holds are not returned to the arena.
- Eviction frees real bytes (the SSD log is append-only; capacity counts bytes on disk, including dead records) and never drops the only copy of a key because its demotion failed transiently.
- The SSD record format (`src/umbp/include/umbp/local/tiers/segment/segment_format.h`: `kRecordMagic`, `kRecordVersion`, `kRecordAlign`, a 32-byte `RecordHeader`) is persisted on disk; a layout change bumps `kRecordVersion`.
- SGLang constructs `UMBPConfig` directly rather than through `FromEnvironment()`, so a config field it needs must also be exposed in `src/pybind/pybind_umbp.cpp` (see the `single_flight_reads` comment in `src/umbp/include/umbp/common/config.h`).
- Metric names stay generic (`MORI_UMBP_METRIC_BACKEND_*` / `MORI_UMBP_METRIC_MEDIUM_*` in `component_metrics.h`); the medium goes in a label.

### 8. Python Bindings, Packaging, and Build
- A new shared library needs `install(TARGETS ...)` in `CMakeLists.txt`, an entry in the `setup.py` copy list, and an entry in `mori_package_data`. Non-`.py` package files (tuning JSON, `.bc`, YAML) need `package_data` entries.
- Optional native pieces are skipped with only a warning (`mori.cco.cco` without Cython, UMBP without gRPC/protobuf). A new extension the package cannot work without goes into `_REQUIRED_EXTENSIONS` in `tools/verify_install.py`, which CI runs instead of a bare `import mori`.
- A new build dependency goes into `requirements-build.txt` (read by `--no-build-isolation` builds), `pyproject.toml` `[build-system].requires`, and `docker/Dockerfile.dev`. Libraries outside UMBP must not link gRPC/protobuf (`verify_install.py` checks every non-UMBP `.so`).
- ROCm 6.4 and later are supported: newer HIP APIs go behind `HIP_VERSION` checks or the shims in `include/mori/utils/hip_compat.hpp`. Wheels are built for Python 3.10 on ROCm 7.2.4 and Python 3.12 on ROCm 7.14 (`nightly.yml`).
- `CMakeLists.txt` and `setup.py` set some defaults differently (`BUILD_UMBP` is OFF in CMake and ON in `setup.py`). A new option must behave the same through `pip install .` and a plain CMake build.

### 9. General C++/HIP Correctness
- GPUs, SDMA engines, and IPC handles are identified by KFD node or PCI bus id. HIP ordinals differ between ranks that set different `HIP_VISIBLE_DEVICES`.
- A new early return or condition inside a cross-rank barrier, signal, or quiet handshake must keep arrive and wait counts equal on every rank; one skipped arrive hangs the peers.
- Data that a NIC or a peer GPU reads is published with system-scope ordering (`__threadfence_system()` or system-scope atomics) before the signal; flags polled across devices live in uncached / fine-grained memory.
- Registered memory regions, QPs, and exported IPC handles outlive the transfers that use them; check teardown order against in-flight work.
- `std::vector<bool>` packs bits, so threads writing different elements of one race; use `std::vector<char>` for results filled in parallel.
- Each module has its own error channel: MORI-IO reports through `TransferStatus` / `StatusCode`, UMBP through outcome enums and `IoStatus`, CCO host functions return `int` (0 or negative) that Cython raises as `RuntimeError`, and the EP launch path throws `std::runtime_error`. `HIP_RUNTIME_CHECK` (`include/mori/application/utils/check.hpp`) exits the process, so it belongs in initialization, not on a path whose caller expects an error back.
- A boolean environment variable read with `getenv(...) != nullptr` treats `=0` as enabled; `mori::env::IsEnvVarEnabled` (`include/mori/utils/env_utils.hpp`) treats `0`, `false`, `off`, and `no` as disabled.

### 10. Compatibility Surfaces
Downstream projects are existing callers, so a break here is reachable even if nothing in this repository calls the changed code.
- Python APIs: `mori.ops` (SGLang, vLLM, RTP-LLM), `mori.io` (SGLang and vLLM disaggregation connectors), `mori.umbp` (SGLang HiCache), `mori.ccl` (`AllgatherSdma` for DeepSpeed, `HierAllGather.supports_param_contiguous_output()` / `enqueue_param_contiguous()` for FSDP2), and the `mori_shmem_*` bitcode symbols (Triton-distributed). Renaming, reordering positional arguments, or changing a default breaks them; the pybind signatures and defaults in `src/pybind/*.cpp` are part of this surface.
- Wire and on-disk formats: the MORI-IO msgpack structs, the UMBP protos, the SSD record format, and the 128-byte unique id (`MORI_SHMEM_UNIQUE_ID_BYTES`) that ranks bootstrap with. A deployment that mixes versions must keep working.
- Environment variable names (`MORI_*`, `MORI_IO_*`, `MORI_EP_*`, `UMBP_*`) are documented, and users and CI set them. A rename keeps accepting the old name; a changed default is a behavior change the PR description should state.

### 11. Testing and CI Coverage
- A new test must run somewhere. Only `tests/python/io/` is collected as a directory (`ci.yml`); every other Python test file is named explicitly in a step of `ci.yml`, `ci_cco.yml`, or `nightly.yml`, so a new file elsewhere runs nowhere until a step names it.
- C++ tests: `tests/cpp/CMakeLists.txt` registers each executable explicitly, and CI builds and runs named targets. `tests/cpp/cco` globs `test_*.cpp`, but the SDMA tests CI executes are a hard-coded list in `ci_cco.yml`. UMBP tests are registered with `add_test` and run only if their name matches `ctest -R '^(umbp_|test_dummy_ssd_tier|test_ssd_tier)'` in the `umbp-unit-test` job of `ci.yml`. JIT v2 tests must match `^jit_` (`ci_cco.yml`).
- Hardware coverage: per-PR CI runs MI355X with AINIC (ionic, gfx950) and MI300X with ConnectX-7 (mlx5, gfx942). There is no bnxt runner, internode tests run only in the two-node jobs, and EP v2, FlyDSL, and CCO tests run per PR only in `ci_cco.yml`. When the main change sits on a path no CI job exercises, one finding asking how it was validated is appropriate.
- Corner cases that deserve a test when the PR adds code for them: serial wraparound, block counts that are not a multiple of the chunk count, EP sizes spanning nodes, `BUILD_CCO_SDMA=OFF`, more than two HIP graph replays, and decoding a message from an older peer.

### 12. MORI Conventions
- Do not report what `pre-commit.yml` already enforces on the PR diff: clang-format (Google-based, 100 columns, `PointerAlignment: Left`), black, ruff (`tools/codestyle/pyproject.toml`), the license header from `LICENSE`, trailing whitespace, tabs, and end-of-file newlines.
- Library code logs with the module macros from `include/mori/utils/mori_log.hpp` — `MORI_{APP,CORE,SHMEM,OPS,IO,UMBP,METRICS}_{TRACE,DEBUG,INFO,WARN,ERROR,CRITICAL}` (CCO logs through the SHMEM macros) — not `printf` or `std::cout`. A new module name goes into `kKnownModules` so its `MORI_<MODULE>_LOG_LEVEL` works. Device code prints with `MORI_PRINTF`, which compiles away unless debug printf is enabled.
- A new environment variable uses its module's prefix and reader: `mori::env` (`include/mori/utils/env_utils.hpp`), or `env::Override` in MORI-IO (`include/mori/io/env.hpp`).
- When a PR changes a documented environment variable, API, or default, the matching guide (`docs/MORI-EP-GUIDE.md`, `MORI-IO-GUIDE.md`, `MORI-SHMEM-GUIDE.md`, `MORI-CCO-GUIDE.md`, `MORI-IR-GUIDE.md`, the UMBP guides) or `README.md` changes with it. A new page under `docs/` must be listed in a `docs/index.rst` toctree, because Sphinx runs with `-n -W`.

---

## Output Format

Begin directly with the structured review below. Do not include reasoning, analysis steps, or thinking-out-loud before it.

```
## Summary
<2–3 sentence overview of what the PR does>

## Findings
- [file:line] <the problem and why it matters; for behavioral findings, also the concrete path that triggers it today>
```
