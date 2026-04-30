# SDMA AllReduce Perf History (per R9/R10)

This ledger is the authoritative source of measured perf data for SDMA
AllReduce optimization work. Every entry must include: date, commit SHA,
scenario, numbers, mechanism, conclusion. See `.cursor/rules/relentless-perf.mdc`
§R9/R10 for ledger discipline.

**Primary goal**: `AllreduceSdma(copy_output_to_user=True)(input, output, stream)`
multi-stage overlap wall < RCCL `dist.all_reduce(tensor)` wall, measured as
median over ≥100 iters at 256MB/stage, 4 stages, 8-rank MI355X, same
benchmark schedule (`test_allreduce.py --num-stages 4 --elems 67108864`).

**Final success criterion is COPY VS RCCL only**:
- ✅ Only `copy_output_to_user=True` / `SDMA copy` wall < RCCL wall counts as success.
- ❌ `SDMA no-copy` beating RCCL does not count; it is diagnostic / upper-bound data.
- ❌ Solutions requiring users to read transit buffers, reuse output buffers, explicitly
  register user output, or change tensor/lifetime semantics are not drop-in and do not
  satisfy the goal.

**Target gap as of last measurement**: ~0.2 ms (SDMA copy ~7.76 vs RCCL ~7.55
median, 3-run spread RCCL ±0.12 ms).

---

## Entry 1 — Baseline measured (pre-optimization)
- **Date**: 2026-04-21
- **Commit**: `b1694d11` (baseline copy-path timing instrumented)
- **Scenario**: `MORI_PIPELINE_CU=220 python3 tests/python/ccl/test_allreduce.py --num-stages 4 --elems 67108864 --timeline --iterations 100 --warmup 20`
- **Hardware**: MI355X × 8, ROCm + HIP 7.1
- **Numbers (single run)**:
  - SDMA copy wall: **7.746 ms** (median)
  - SDMA no-copy wall: **7.354 ms**
  - RCCL wall: **7.425 ms** (volatile, see Entry 2)
  - copy exposure (copy − no-copy): **0.392 ms**
- **Per-stage AR dur (no-copy)**:
  - AR[0]=1.926, AR[1]=1.919, AR[2]=1.196, AR[3]=1.195
- **Per-stage AR dur (copy)**:
  - AR[0]=2.186, AR[1]=1.835, AR[2]=1.296, AR[3]=1.297
- **AR[0] kernel phase breakdown (no-copy)**:
  ```
  total                                 1.887 ms
  entry→scatter_done                    0.203
  c0:scatter→compute-wait               0.418
  c0:compute-wait→barrier               0.022
  c0:barrier→AG-submit                  0.215
  c1:scatter→compute-wait               0.099
  c1:compute-wait→barrier               0.095
  c1:barrier→AG-submit                  0.214
  AG-submit→AG-wait-done                0.614
  AG-wait-done→exit                     0.007
  ```
- **Copy-path (single hipMemcpyAsync in copy_output_to_user)**:
  - host-side call latency: **3.20 us**
  - gpu-side CU blit wall: **0.259 ms** (256MB linear D2D)
- **Conclusion**: baseline established. SDMA copy is slower than RCCL by
  ~2% (single run). no-copy is faster than RCCL but requires user API
  change (reading `get_output_transit_buffer`), not allowed per R7.

---

## Entry 2 — RCCL noise characterization (3 independent runs, 200 iter)
- **Date**: 2026-04-21
- **Commit**: same baseline (`b1694d11`)
- **Scenario**: 3× `MORI_PIPELINE_CU=220 ... --iterations 200 --warmup 30`
- **Numbers**:
  | run | SDMA copy | SDMA no-copy | RCCL |
  |---|---|---|---|
  | 1 | 7.765 | 7.376 | 7.431 |
  | 2 | 7.753 | 7.397 | 7.582 |
  | 3 | 7.757 | 7.370 | 7.672 |
  | mean | **7.758** | **7.381** | **7.562** |
  | stdev | 0.006 | 0.014 | **0.122** |
- **Conclusion**: SDMA measurements are extremely stable (stdev ≤0.015 ms).
  RCCL noise is ~0.12 ms cross-run. True gap = **copy mean − RCCL mean
  = 0.196 ms (+2.6%)**, stable across runs, not explainable by noise.

---

## Entry 3 — CU sweep (MORI_PIPELINE_CU 80→220)
- **Date**: 2026-04-21
- **Commit**: `b1694d11`
- **Scenario**: `for CU in 80 120 140 160 180 200 220; do MORI_PIPELINE_CU=$CU ...; done`
- **Numbers**:
  | CU | copy (ms) | no-copy (ms) | copy exposure |
  |---|---|---|---|
  | 80  | 8.113 | 7.491 | 0.622 |
  | 120 | 7.797 | 7.454 | 0.343 |
  | 140 | 7.807 | 7.427 | 0.380 |
  | 160 | 7.782 | 7.390 | 0.392 |
  | 180 | 7.764 | 7.407 | 0.357 |
  | 200 | 7.776 | 7.375 | 0.401 |
  | 220 | **7.744** | **7.357** | 0.387 |
- **Per-stage AR[0] dur by CU**: 2.704 (CU=80) → 2.161 (CU=220), monotonic
- **Conclusion**: CU=220 is near-optimal (no-copy wall min, copy wall min).
  Copy exposure is **invariant to CU in ≥120 range** — falsifies H1
  (copy uses idle CU during AG wait). CU tuning alone cannot close the gap.
- **Rule application (R8)**: CU sweep gain ≤0.05 ms, gap is 0.2 ms → CU
  tuning as standalone direction is ruled out.

---

## Entry 4 — AR[0] vs AR[2] phase comparison (attribution of cold-path cost)
- **Date**: 2026-04-21
- **Commit**: `b4e6d4e8` (added `MORI_PHASE_TARGET_STAGE` to instrument any AR[N])
- **Scenario**: two separate runs with `MORI_PHASE_TARGET_STAGE=0` and `=2`
- **Numbers (no-copy mode, so AR kernel is not contaminated by copy)**:
  | Phase | AR[0] | AR[2] | Δ (AR[0] − AR[2]) |
  |---|---|---|---|
  | entry→scatter_done | 0.203 | 0.002 | +0.201 |
  | c0:scatter→compute-wait | 0.418 | 0.076 | +0.342 |
  | c0:barrier→AG-submit | 0.215 | 0.013 | +0.202 |
  | c1:barrier→AG-submit | 0.214 | 0.009 | +0.205 |
  | AG-submit→AG-wait-done | 0.614 | **1.029** | **−0.415 (AR[2] slower!)** |
  | total | **1.887** | **1.189** | **+0.698** |
- **Mechanism**:
  - AR[0] slow across **every** phase, not concentrated anywhere single
  - AR[0] entry slow → cold-path CU allocation blocked by GEMM[1]
  - AR[0] barriers slow → all peers in cold path, barrier waits for slowest
  - AR[2] AG wait longer than AR[0]'s → SDMA queue backlog (prior AR's
    scatter/AG still draining); this is fundamental SDMA transfer time floor
- **Conclusion**: AR[0]'s 0.7 ms overhead is the largest reclaimable slice
  (~3 separate sources: cold entry, CU contention with GEMM[1], cross-PE
  barrier cold-path sync). AR[2]'s 1.03 ms AG wait is **physical lower
  bound** of the SDMA transfer itself — cannot be shortened.

---

## Entry 5 — Stream priority test (ruled out)
- **Date**: 2026-04-21
- **Commit**: `4170b412` (added `--ar-priority/--gemm-priority` knobs)
- **Scenario**: `--ar-priority=-1 --gemm-priority=0` vs default `0/0`
- **Numbers**:
  | metric | prio 0/0 | prio -1/0 | Δ |
  |---|---|---|---|
  | SDMA copy wall | 7.762 | 7.771 | +0.009 |
  | AR[0] dur | 2.167 | 2.198 | +0.031 |
- **Conclusion**: HIP stream priority has **no measurable effect** on
  GEMM-AR CU contention on MI355X/ROCm. Direction **A1 (priority-based
  CU reassignment) ruled out**.

---

## Entry 6 — Stage 1 (compute blocks spin-wait, no copy) — acceptable cost
- **Date**: 2026-04-21
- **Commit**: `02375192` (compute blocks wait post_ag_flag before exit, no copy)
- **Scenario**: `MORI_POST_AG_WAIT=1 MORI_PIPELINE_CU=220 ...` vs OFF
- **Numbers**:
  | metric | OFF | Stage 1 ON | Δ |
  |---|---|---|---|
  | SDMA copy wall | 7.746 | 7.805 | **+0.059** |
  | SDMA no-copy wall | 7.354 | 7.408 | +0.054 |
  | GEMM[1-3] dur | 1.05–1.12 | 1.07–1.14 | +0.02 each (~+0.07 total) |
- **Conclusion**: compute blocks staying alive during AG wait (~1.35 ms
  extra CU occupancy) costs only +0.06 ms wall. **Cost is acceptable**,
  leaves runway for Stage 2.
- **Key insight**: GEMM wall grows (0.07 ms total) but stream_gemm is not
  critical path, so does not hit wall.

---

## Entry 7 — Stage 2b-0 (per-chunk AG wait only, no copy) — also acceptable
- **Date**: 2026-04-22
- **Commit**: `3a738682` (block 0 waits AG chunk-by-chunk instead of one-shot)
- **Scenario**: same as Entry 6
- **Numbers**:
  | metric | Stage 1 OFF baseline | Stage 2b-0 | Δ |
  |---|---|---|---|
  | SDMA copy wall | 7.746 | 7.765 | +0.019 |
  | SDMA no-copy wall | 7.354 | 7.361 | +0.007 |
  | AR[0] total (no-copy) | 1.887 | 1.889 | +0.002 |
  | AG-submit→AG-wait-done (AR[0] no-copy) | 0.614 | 0.597 | **−0.017 (slightly faster)** |
- **Conclusion**: per-chunk AG wait does **not** reproduce the Path B
  failure mode (path B was a host-side `hipStreamWaitValue32 +
  hipMemcpy2DAsync` change, not per-chunk kernel wait). Kernel-side
  per-chunk wait is free. Foundation for Stage 2b-1 validated.

---

## Entry 8 — Stage 2b-1 (per-chunk in-kernel copy) — CATASTROPHIC FAIL, REVERTED
- **Date**: 2026-04-22
- **Commit**: `dfe56fee` (kernel) + `b8ee2b8a` (Test 1b)
- **Reverted**: `ba4b4b41` (revert of 2b-1)
- **Scenario**: `MORI_POST_AG_WAIT=1 MORI_PIPELINE_CU=220 ...`
- **Correctness**: Test 1b PASSED (output written correctly in-kernel) ✅
- **Numbers**:
  | metric | OFF | Stage 2b-1 ON | Δ |
  |---|---|---|---|
  | SDMA copy wall | 7.761 | **8.940** | **+1.179 ms (catastrophic)** |
  | AR[0] dur | 2.186 | 3.508 | +1.322 |
  | AR[1-3] dur | 1.30ish | 1.41ish | +0.11 each |
- **AR[0] Phase (Stage 2b-1 ON) vs no-copy baseline**:
  - entry→scatter_done: +0.138
  - c0:scatter→compute-wait: +0.333
  - barrier→AG-submit (×2): +0.419
  - AG-submit→AG-wait-done: +0.539
  - block 0 total: +1.643 (every phase slowed, not concentrated)
  - compute block `c1-reduce-done→cb-exit`: **0.007 → 7.764 ms** (scaling anomaly,
    block 0 exit cycle much earlier than compute block exit, cy_to_ms mis-scaling)
- **AR[2] Phase (Stage 2b-1 ON) vs no-copy baseline**:
  - total: 1.189 → **1.412 ms** (+0.218, copy barely hidden)
  - AG-submit→AG-wait-done: 1.029 → 1.140 (+0.111, slightly throttled)
  - compute block `c1-reduce-done→cb-exit`: 0.001 → **1.655 ms**
    (compute block does spin+copy here; ~0.5 ms per-chunk copy × 2 + sync)
  - Block 1 total: 0.136 → 1.831 (+1.695)
- **Critical comparison AR[0] vs AR[2] in ON mode**:
  - AR[0] cb-exit phase: 7.764 ms
  - AR[2] cb-exit phase: 1.655 ms
  - delta **6.1 ms** purely explained by GEMM[1] CU contention with
    AR[0]'s compute blocks doing HBM-heavy in-kernel copy
- **Mechanism**:
  - Compute blocks doing simultaneous reduce-HBM-read + copy-HBM-read+write
    saturates HBM bandwidth within the kernel
  - Block 0's SDMA submit/poll is also HBM-serving → gets throttled
  - When AR[s] also has to share CU with GEMM[s+1] (AR[0]/AR[1] case),
    the effect is **compounded** — CU-contention × HBM-contention ×
    cold-path all pile up
  - HBM saturation **propagates to every phase of the AR kernel**, not just copy
- **Conclusion**: **In-kernel per-chunk copy is architecturally infeasible
  on MI355X** when same kernel does HBM-heavy reduce AND HBM-heavy copy.
  Any future in-kernel copy direction must either:
  - (a) ensure copy does not coincide with reduce HBM usage (e.g., copy only
    after reduce is fully idle, which equals just serializing → no gain), or
  - (b) use an engine other than CU for the local copy (SDMA-local is 50× slower
    per Entry 9; no other DMA engine currently available)
  Direction **E' (in-kernel copy) closed** on this hardware.

---

## Entry 9 — Failed directions prior to this session (inherited from transcripts)
Summarized here for quick lookup; exact commits in git log:

| Direction | Commit (both tried and reverted) | Fail mechanism |
|---|---|---|
| In-kernel local SDMA copy (qId=2) | `1b66f5c5` → revert `4c3229ba` | Local HBM-HBM via SDMA engine is ~50× slower than CU blit; SDMA not designed for local traffic |
| Custom low-CU blit kernel (16 blocks) | `13efef1f` → revert `c3b54546` | Fewer blocks cannot saturate HBM bandwidth; standalone kernel slower than `__amd_rocclr_copyBuffer` |
| Dedicated copy_stream + main-stream wait copy_done | `4d49ce01` → revert `0592319a` | Main stream waiting on copy_done puts copy back on critical path; wall not improved (copy still serialized) |
| Path B chunk-level host-dispatch (hipStreamWaitValue32 + hipMemcpy2DAsync) | `5168d801` → revert `66629395` | Host-side per-chunk wait+memcpy has ≥0.5 ms launch/sync overhead that negates the chunk overlap benefit |

---

## Entry 10 — D' Step 1 measured: ShmemSymmetricRegister host us cost
- **Date**: 2026-04-22
- **Commits**: `df3b1ec0` (API + LRU cache), `9b77c0b4` (cost test),
  `aa857af9` (test dist init fix)
- **Scenario**: `python3 tests/python/ccl/test_register_cost.py` (8-rank, measures
  first-register cost vs cache-hit cost for 5 sizes 1→256 MB)
- **Numbers** (C++ chrono around `shmem::ShmemSymmetricRegister`):

  | size (MB) | MISS (us) | HIT (us) |
  |---|---|---|
  | 1   | 1068.1 | 0.0 |
  | 16  |  940.1 | 0.0 |
  | 64  | 1037.6 | 0.0 |
  | 128 | 1264.3 | 0.0 |
  | 256 | 1276.1 | 0.0 |
  | avg | **~1120** | **0** |

- **Mechanism observations**:
  - Register cost is **essentially size-independent** (940→1276 us across
    256× size range). Dominated by the 3× `bootNet.Allgather` collective
    overhead (pointers, IPC handles, rkeys) + `hipIpcOpenMemHandle` × N peers.
  - Cache hit is **truly free** (0 us C++ chrono; 1.7 us python-level
    wall = Python dispatch overhead only).

- **Decision gates (a priori, from Entry 10 preamble)**:
  - <500 us → cheap. **Missed this gate** (actual ~1100 us).
  - 500-2000 us → benchmark warmup can absorb. **Hit this gate** ✅.
  - >2000 us → may not fit. **Not triggered**.

- **Gain/gap analysis (per R8, referencing R9 ledger)**:
  - Target gap = 0.196 ms (Entry 2: 3-run RCCL stable gap)
  - Copy exposure reclaimable = 0.392 ms (Entry 1: copy wall − no-copy wall)
  - Break-even N_iters = register_cost_ms / copy_exposure_ms
    = 1.12 / 0.392 ≈ **3 iters** to pay back one register
  - benchmark's `--warmup 20` easily absorbs this in warmup phase
  - Steady state (post-warmup): cache hit, zero register overhead, gain ≈ 0.4 ms

- **D' projected wall** (steady state, referencing Entry 2):
  - stream_ar runs AR kernel directly writing user output (no transit copy)
  - wall ≈ no-copy wall = **7.381 ms**
  - vs RCCL median 7.562 ms = **超 0.181 ms (-2.4%)** ✅ meets primary goal

- **Conclusion**: Step 1 validates D' is viable for benchmark-style
  workloads (stable output ptr + warmup). **Decision: proceed to Step 2**
  (change AR kernel destination from transit to user_output_symm when
  cache hit; host skips `copy_output_to_user`).

- **Known risk for real user workloads**:
  - If user allocates a fresh output tensor every call AND cache evicts it,
    cost is +1.1 ms per AR call, which exceeds the saved 0.4 ms → net loss.
  - Mitigation: (a) cache size 4 handles most pytorch-caching-allocator
    reuse; (b) Step 2 implementation will include fallback (if register
    fails OR cache miss + would-exceed-cap, fall through to baseline
    transit+copy path, no regression vs current).

### Update (2026-04-22) — Step 2 partial rollout

- Commit `fa1565a1` (C++ Step 2 path selection) + `e5c70de9` (wrapper methods)
- Additional data from `test_allreduce.py` benchmark setup (4 MB, single call
  via `_bench_overlap_one_size` `setup()`): **register MISS = 2398 us**
  — 2× the isolated test_register_cost measurement above.
- Divergence root cause (hypothesis, unverified): benchmark runs set up
  AR+register inside `setup()`, concurrent with other PE-local tensor
  allocation; `ShmemSymmetricRegister`'s internal `bootNet.Allgather` may
  suffer contention with `torch.distributed` (Gloo) initialization or other
  host activity in the first benchmark seconds.
- Impact on break-even: 2.4ms / 0.39ms ≈ **6 iters** to pay back (still well
  within warmup=20 budget). Still **viable** for benchmark; concerning for
  very-short-lived user sessions.
- Action item: **always pre-register in the setup phase, not the hot loop**.
  benchmark's `make_setup()` already does this.

---

---

## Entry 11 — D' Step 2 (fast path) CORRECTNESS FAIL, REVERTED
- **Date**: 2026-04-22
- **Commits**: `fa1565a1` (Step 2 fast path, BUGGY) → reverted `6a544229`
- **Test 1b** (copy_output_to_user=True, fast path ON) **all 8 PE FAIL**:
  ```
  PE 0 first mismatch at idx 0: got 1000 (expected 36000)
  PE 1..7 first mismatch at idx 0: got 0 (expected 36000)
  ```
  - PE 0 gets only its own scattered shard (self-send doesn't go through SDMA)
  - PE 1..7 get literally zero data — remote PE's SDMA puts went to
    wrong addresses, target `output_tensor` never received them

- **Root cause (mechanism-level, not "I guess")**:
  - `ShmemSymmetricRegister(ptr, size)` internally calls `hipIpcGetMemHandle`
    on `ptr` and `hipIpcOpenMemHandle` on peer handles
  - `hipIpcOpenMemHandle` always returns the **allocation base** pointer
    on the remote side, NOT the per-tensor data_ptr
  - PyTorch caching allocator: `tensor.data_ptr()` is an **offset within**
    a larger allocation block. `data_ptr() != allocation_base`
  - So when mori stores `peer_ptrs[i] = p2p_peer_pointer_from_ipc`, it
    actually has peer's ALLOCATION BASE, not peer's tensor start
  - AR kernel's SDMA puts `write to peer's data_ptr + offset` but the
    peer sees that address as pointing into OTHER data (or out-of-bounds)

- **Evidence that the bug is in ptr mapping, not ptr validity**:
  - `register_user_output` returned `ok=True` (register succeeded)
  - Every PE entered fast path (kernel got ar_dst_obj = user_output symm)
  - Scatter SDMA submissions happened (no HIP launch error)
  - But output tensor on peers is unchanged → writes landed elsewhere

- **Why `shmem::ShmemMalloc` works but `hipMalloc`/PyTorch tensor doesn't**:
  - `ShmemMalloc` allocates inside a known shmem heap whose base is tracked
  - The assumption `ptr = allocation_base` holds for shmem-heap allocs
  - PyTorch tensor breaks this assumption; caching allocator subdivides
    blocks allocated via `hipMalloc`

- **What a fix would require (NOT attempted in this session)**:
  - Every rank allgathers (allocation_base, offset_from_base) instead of
    just ptr. After IPC open, compute `peer_ptr = peer_base + peer_offset`.
  - On PyTorch side: need `hipDrvMemGetAddressRange(ptr)` or equivalent to
    get the allocation base. Must handle aliasing, pinned allocations, etc.
  - This is a `mori/shmem` library-level refactor, out of scope for this
    session's goal.

- **Conclusion**: **D' direction is not viable under PyTorch's default
  caching allocator without a shmem-library IPC refactor**. Revert the
  Step 2 kernel launch change; keep the Python `enable_register_user_output`
  API as a no-op placeholder for future offset-aware register.
- **Direction D' closed** for now. Pinning hopes on a correct implementation
  requires rewriting `mori/shmem/application/memory/symmetric_memory.cpp`
  to be offset-aware.

---

## Entry 13 — ρ' (increase chunk density) FAILED (2026-04-22)
- **Scenario**: `MORI_PIPELINE_CHUNKS=2/4/8` env var, 4-stage 256MB overlap
- **Results**:
  | chunks | SDMA copy | SDMA no-copy |
  |---|---|---|
  | 2 (baseline) | 7.783 | 7.359 |
  | 4 | 7.783 | 7.427 |
  | 8 | 7.838 | 7.479 |
- **Conclusion**: Chunk density increase has **zero gain on copy wall** and
  slightly worsens no-copy wall (more submit overhead + more per-chunk barriers).
- **Hypothesis refuted**: "SDMA engine stays hot with denser submission" is
  wrong. Packet density alone doesn't keep engine hot.
- **Refined mechanism hypothesis (needs further validation)**:
  - Multi-stage AR[0] (0.614ms AG) vs Single-stage AR[0] (1.086ms AG) 差异
    根源**不是** stream_ar 内部 submission 密度，而是 **cross-stream HBM 流量**
  - Multi-stage 下 `stream_gemm` 上 GEMM[1..3] 在 AR[0] 期间持续产生 HBM 流量，
    让 **memory controller / SDMA engine 保持 hot**
  - Single-stage 或 AR[last]（multi-stage）下无 cross-stream 活动 → engine 进入
    low-power wait → next packet dispatch latency +0.5 ms
  - **ρ' 只在 stream_ar 内加密度没用，需要 cross-stream activity**

---

## Entry 12.1 — Direction θ (multi-qId AG) FAILED IN IMPLEMENTATION (2026-04-22)
- **Commits**: `1df5a9b1` (impl) + `c11c4870` (test) → reverted `219647a5`, `c3eccaa0`
- **Correctness**: ✅ PASSED (Test 1b with MORI_AG_MULTI_Q=1)
- **Perf (3-run clean env)**:
  - OFF: SDMA copy = 7.770 ms
  - ON θ: SDMA copy = 7.794 ms
  - **Δ ≈ +0.024 ms (noise, no gain)**
- **Direct measurement shows θ mechanism works but gain is absorbed**:
  - OFF AR[0] (c0/c1 use same qId=1, FIFO serial):
    - c0 submit-to-done: 0.643 ms
    - c1 submit-to-done: 0.700 ms
    - c1 global done = c0 submit + 1.219 ms (last to finish)
  - ON θ (c0→qId=1, c1→qId=2, parallel):
    - c0 submit-to-done: **1.193 ms** (+0.55 ms vs OFF!)
    - c1 submit-to-done: 0.745 ms
    - max done global = c0 submit + 1.193 ms
  - **Total AG window shrinks only from 1.219 → 1.193 = 0.026 ms**
- **Why Entry 12's gain prediction (0.5 ms) was wrong**:
  - Entry 12 measured AR[2] warm path where c0/c1 submits were only 0.078 ms apart
  - AR[0] cold path has **c0↔c1 submit interval of 0.52 ms** (scatter, barrier, etc
    happen in between chunks' submits), which naturally overlaps with c0 AG transfer
  - "Serial FIFO" in single-qId mode isn't actually bottlenecking because the
    submit spacing already pipelines transfers
  - Adding a second qId **doubles per-chunk transfer time** (HBM port contention
    or SDMA engine cross-queue interference) — exactly cancelling the parallel gain
- **Machine-level conclusion**:
  - MI355X SDMA engine is **effectively bandwidth-saturated per-link** once the
    chunk submit interval is ≥ chunk transfer time
  - Multi-qId only helps when submit interval ≪ transfer time (e.g., AR[2] warm
    where intervals are tens of us); but those cases are short-duration and contribute
    little to wall time
  - **Direction θ (multi-qId AG) closed** for the current AR structure

### Update 2 — confirmed via `examples/sdma/sdma_bw.cpp` calibration (2026-04-22)
- Ran `sdma_bw` (shader-initiated SDMA, same mechanism as `core::SdmaPutThread`
  used in AR kernel — packet built in shader, transfer executed by SDMA engine;
  NOT CU blit):
  - 1 src GPU → 7 dest GPUs concurrent, single queue per dest, size 1 KB → 1 GB
  - 256 MB: **427.7 GB/s total** (= **61 GB/s per link**)
  - 1 GB:   428.4 GB/s (saturated)
- AR AG phase BW utilization (using this SDMA-engine peak):
  - AR[0] (cold): 224 MB ingress / 0.614 ms = **365 GB/s (85% of peak)**
  - AR[2] (warm, backlog):   224 MB / 1.029 ms = **218 GB/s (51% of peak)**
- This empirically confirms the theoretical reasoning: **AR[0] AG phase is
  already 85% peak**. Multi-qId (θ) cannot help — the SDMA engine's per-link
  throughput is the binding constraint, not queue count or parallelism.
- AR[2]'s 51% utilization (low vs AR[0]'s 85%) is explained by **queue backlog**
  from prior AR calls' lingering SDMA submissions. Not a BW issue.

### Update 3 — backlog hypothesis REFUTED by single-stage test (2026-04-22)
- Ran `--num-stages 1 --elems 67108864 --iterations 20` to test AR[0] when
  there's no prior or subsequent AR (truly "clean" SDMA queue, no backlog):
  - single-stage AR[0] AG-submit→AG-wait-done: **1.086 ms** (slower than
    multi-stage AR[0] 0.614 ms despite having zero backlog)
  - c0 per-chunk AG done: 0.583 ms, c1 per-chunk AG done: 1.086 ms
- **This refutes the "AR[2] slow because of queue backlog" reading**: AR[0]
  with NO work upstream is actually even slower (1.086 ms) than AR[0] with
  3 subsequent AR kernels queued up (0.614 ms).
- **Correct mechanism (new hypothesis, needs validation)**: **SDMA engine
  latency degrades when submission is sparse**. When multi-stage has
  AR[1..3] kernels already queued on stream_ar, the SDMA queue stays
  "hot" (additional packets keep flowing through the engine). When a
  single AR[0] runs alone (single-stage, or AR[last] in multi-stage), the
  SDMA engine processes the packet, drains, and enters low-power wait for
  the next packet; next-packet dispatch latency adds ~0.5 ms per chunk.
- **Correct gap decomposition**:
  - AR[0] multi-stage = 0.614 ms ≈ 85% SDMA engine peak (HOT queue)
  - AR[2]/AR[3]/single-stage AR[0] = 1.029/1.086 ms ≈ 50% peak (COLD engine dispatch overhead)
  - Single-chunk physical transfer time is only ~0.26 ms; the remaining
    0.5-0.8 ms is engine wake-up/dispatch between chunk packets
- **New directions suggested by this mechanism**:
  - ρ'（增加 chunk 密度）: reduce chunk_elems (numChunks=4 or 8 instead of 2)
    → submissions denser → SDMA engine stays hot → AR[2]/AR[3] AG could drop
    from 1.03 ms → close to 0.6 ms = save **0.4 ms per AR × 2 AR = 0.8 ms wall**
  - σ（keep SDMA queue hot in kernel）: at kernel end, insert a small
    dummy SDMA op so the engine doesn't enter low-power state before next
    AR's submit. Measure if subsequent AR gets 0.6 ms AG-wait instead of 1.0 ms.
- **Rule lesson (R6 reflection)**: Entry 12's ONE datapoint (AR[2] warm) led to a
  gain extrapolation that didn't hold for the AR[0] cold path where it actually
  mattered. Future R10 references must use the relevant stage's data, not the
  most-convenient datapoint.

---

## Entry 12 — Per-chunk AG timing proves single SDMA queue is strictly FIFO (not pipelined)
- **Date**: 2026-04-22
- **Commit**: `d56bbe29` (per-chunk AG-done display), data from AR[2] warm path
- **Scenario**: `MORI_PIPELINE_CU=220 MORI_PHASE_TARGET_STAGE=2 --num-stages 4 --elems 67108864 --ar-phase-timing`
- **Numbers** (AR[2], SDMA copy mode, warm / no GEMM contention):
  - `c0:AG-submit→AG-done (per-chunk)`:  **0.591 ms**
  - `c1:AG-submit→AG-done (per-chunk)`:  **1.098 ms**
  - `AG-submit→AG-wait-done` (total):    **1.098 ms**
  - Submit delta (c0 → c1 submit): only 0.078 ms apart
- **Analysis**:
  - c1 completion is 0.507 ms later than c0 completion
  - 0.507 ms ≈ 1 chunk's physical transfer time — exactly what single-queue
    FIFO serial execution predicts
  - If a single queue **saturated HBM bandwidth with pipelined transfers**,
    c0 and c1 would complete ~simultaneously (c1_done ≈ c0_done ≈ 0.6ms).
    Data **decisively refutes** that hypothesis.
  - SDMA hardware architecture: each queue = one DMA channel; within a
    queue ops are strictly FIFO (no per-channel pipelining); to parallelize
    chunks one must use **multiple queues** (i.e. multiple channels).
- **Conclusion**: **Direction θ (multi-qId AG) has real gain potential**.
  - Expected per-chunk AG time going from 0.591+0.591=1.098 ms
    serialized → max(0.591, 0.591)=0.591 ms parallel (two queues)
  - Save ≈ 0.5 ms per AR kernel's AG wait phase
  - Per Entry 4, AR[0] AG wait is 0.614 ms (cold path); splitting two
    chunks over two queues would roughly halve that too (0.3 ms per chunk)
  - Gap=0.196 ms. Gain (conservative 0.3 ms / optimistic 0.8 ms) / gap =
    1.5× – 4×. **Satisfies R8**.
- **Implementation plan (Step 2)**:
  - MULTI_CHUNK AG: chunk c uses `qId = 1 + (c % (numQ - 1))`
    (so c0→qId=1, c1→qId=2; numQ ≥ 3 required)
  - AG wait: per-chunk wait on peer's `signalPtrs[peer*numQ + qId_of_chunk]`
  - scatter stays on qId=0 (unchanged)
  - Verify `dstMemObj->sdmaNumQueue >= 3` (default 8 per `anvil::GetSdmaNumChannels`)

---

## Entry 17 — E'' in-kernel copy FAILED (all 3 variants), fundamental HBM contention
- **Date**: 2026-04-22
- **Commits**: `c00eedcd` (E'' v1, 20 blocks/peer), `efba6f6b` (E'' v2, 1 block/peer), `53eb011b` (E'' v3, 8 blocks/peer); reverted at this commit.
- **User-proposed direction**: AG data already goes to local transit
  (shmem); move the transit→user_output copy INTO the AR kernel, have
  compute blocks do it during block 0's AG wait. No shmem register of
  user_output needed, avoiding D' failure.
- **Data (AR[3] phase, 256 MB, multi-stage overlap)**:
  | variant | blocks/peer | active copy blocks | AR[3] AG wait | AR[3] cb-exit | Wall Δ |
  |---|---|---|---|---|---|
  | BASELINE | — | — | 1.080 ms | 0.001 ms | — |
  | E'' v1 | 20 | 160 | 1.127 (+0.05) | 0.245 ms | **+0.16 ms** |
  | E'' v2 | 1 | 8 | **2.639 (+1.56)** | **9.774 ms** | **+7.54 ms (×2)** |
  | E'' v3 | 8 | 64 | 1.258 (+0.18) | 0.632 ms | **+0.67 ms** |
- **Mechanism analysis**:
  - v1 (20 blocks/peer, 160 copy-workers): **HBM contention with SDMA AG**.
    160 blocks × 256 threads = 40K in-flight HBM requests compete with
    the SDMA engine's inbound AG writes (~224 MB into local HBM over
    1 ms window). AG wait is slowed by 0.05 ms — small per-AR but adds
    up across 4 ARs, exceeds the 0.095 ms per-AR external copy saving.
  - v2 (1 block/peer, 8 copy-workers): **HBM bandwidth starvation**.
    A single block with 256 threads cannot saturate HBM (needs ~4K
    in-flight requests on CDNA). Each load stalls waiting for HBM
    queue, giving ~2.5 MB/s effective. 32 MB / block = ~13 ms per
    copy. Catastrophic; AG also slows by +1.56 ms (mechanism unclear —
    likely the compute blocks' long spin holds CU resources and creates
    some systemic bottleneck).
  - v3 (8 blocks/peer, 64 copy-workers): **intermediate fails too**.
    Better than v1+v2 per-call but still +0.18 ms AG slowdown. Net
    +0.67 ms wall regression (4 ARs × ~0.17 ms).
  - **Fundamental physics**: on MI355X, SDMA engine and CU both access
    the same HBM controller. ANY substantial CU HBM activity during
    SDMA AG will slow AG by a proportional amount. The external
    hipMemcpyAsync (CU blit, 0.095-0.259 ms per AR) runs AFTER AG done,
    so it doesn't contend; moving it INTO AG wait window trades one
    hidden cost for another that partially competes with AG.
  - **Net accounting** (v3, best of bad options):
    - External copy saved: 4 × ~0.15 ms = 0.60 ms
    - AG slowdown due to in-kernel copy: 4 × ~0.17 ms = 0.68 ms
    - Net: −0.08 ms save in theory, but other timing shifts (AR[0] /
      AR[2] / AR[3] each +0.2 ms) make wall +0.67 ms worse in practice.
- **Conclusion**:
  - E'' direction **closed** at all 3 variants. Reverted fully.
  - **In-kernel copy during AG wait is not viable** on this architecture
    without a way to prevent the copy from competing with SDMA for HBM
    bandwidth. (E.g., if CU→HBM and SDMA→HBM used physically separate
    channels, this would work. They don't on MI355X.)
- **Remaining candidates** (all previously evaluated or blocked):
  - **D'' (revised D')**: user-explicit `ar.register_output(tensor)`
    API; no allocator interplay issue. Breaks full drop-in, but only
    adds one optional API call. Expected -0.37 ms if cache hits.
  - **π' (copy on separate SDMA stream with double-buffered transit)**:
    2× transit memory; cross-stream event sync; still SDMA+SDMA HBM
    sharing but they can multi-queue. Complex.
  - **Status quo**: accept the 0.37 ms (5%) gap; SDMA AR is already
    competitive with RCCL on large sizes and faster on no-copy path.

---

## Entry 17-original — E'' in-kernel copy (user-proposed) — hypothesis
- **Date**: 2026-04-22
- **Commit**: this commit
- **User proposal**: AG destination stays as local transit (shmem, unchanged).
  But the `transit → user_output` step is moved INTO the AR kernel: compute
  blocks (idle after reduce) copy per-peer shards during block 0's AG wait.
  The external `hipMemcpyAsync` is removed.
- **Why it's different from prior failed directions**:
  - D' (closed, Entry 11): required registering user_output as shmem symm
    memory (peer AG write-target); failed on PyTorch allocator +
    `hipIpcOpenMemHandle` offset semantics. E'' **does not require symm
    registration of user_output** — the compute-block copy is purely
    local HBM→HBM and uses whatever pointer the AR call received.
  - E' Stage 2b-1 (closed, Entry 8): per-chunk copy interleaved WITH
    reduce → reduce+copy both hit HBM/CU at the same time → +1.18 ms
    regression. E'' does copy **strictly AFTER all-chunks reduce is
    done**, so there is no reduce/copy overlap on the same HBM; it only
    overlaps with block 0's AG wait (SDMA write to peer HBM, not local
    HBM read).
- **Design**:
  - Kernel param `void* user_output_for_copy` added; `nullptr` → legacy path.
  - Each compute block (blockIdx.x >= 1) is statically assigned 1 peer by
    `peer = cb_id * npes / compBlocks` (round-robin). Blocks assigned to
    the same peer split that peer's shard bytes evenly. With compBlocks=160
    and npes=8, each peer gets 20 blocks.
  - If peer == myPe: transit[myPe_slot] already has this PE's reduce
    result; copy can start as soon as reduce completes (no wait).
  - If peer != myPe: thread 0 spins on `signalPtrs[peer*numQ+1] >=
    agBase+numChunks`; once observed, all threads in the block do a
    vectorized (ulonglong2 = 16B) copy of their assigned slice of the
    peer's 32 MB shard.
- **Expected numbers** (256 MB, 4 ARs, CU=160):
  - AR[N-1] kernel: AG wait 1.08 ms, copy ~15 us (per peer) × 7 peers
    copying in parallel (different blocks) → all 256 MB copied within
    ~20 us starting from respective peer signal arrivals → fully hidden
    inside 1.08 ms AG wait.
  - Per-AR saving vs baseline: 0.35 ms (external hipMemcpyAsync removed).
  - 4 ARs × 0.35 = **1.4 ms total wall saving**; wall 8.32 → ~6.92 ms.
  - **6.92 < RCCL 7.43** → target met (≥ 0.5 ms headroom).
- **Risks to verify**:
  - HBM bandwidth: AG inbound (224 MB written to local HBM over AG wait)
    + compute-block copy (256 MB read + 256 MB write). Total ~740 MB /
    1.08 ms ≈ 700 GB/s on local HBM, well within MI355X HBM peak.
  - Alignment: user_output assumed ≥16 B aligned (PyTorch allocator is
    ≥256 B, Torch tensor data_ptr aligned).
  - Correctness: compute-block copy writes to user_output only after the
    relevant peer's AG signal is observed. Same signal semantics as the
    existing per-chunk AG wait (which is already verified correct on
    this codepath). Block 0 exits when its AG wait ends; compute blocks
    exit after all copies done. Kernel total = max(block-0 AG wait,
    slowest block copy end).

---

## Entry 16 — τ'' (in-kernel HBM noise) FAILED, refocus on copy cost
- **Date**: 2026-04-22
- **Commit**: `0f6f4920` (τ'' v1), `4c57e2c0` (τ'' v2 — heavy noise + per-call gating)
- **Scenario**: `tools/bench_tau_xi.sh` with `MORI_POST_AG_WAIT=1 MORI_HBM_NOISE=1`
- **Numbers (median wall ms, copy mode)** v1 (blanket enable, 1 byte/iter noise):
  | label | copy | RCCL | AR[3] AG |
  |---|---|---|---|
  | BASELINE | 7.787 | 7.418 | 1.083 ms |
  | TAU2 v1  | 8.212 | 7.446 | 1.021 ms (-0.06) |
  Wall **+0.425 ms worse** (AR[0/1/2] got longer because HBM noise during
  their AG contends with parallel GEMM[s+1] reading HBM for matmul).
- **Numbers (v2 — heavy noise 512B/iter/thread, gated to AR[N-1] only)**:
  | label | copy | RCCL | AR[3] AG |
  |---|---|---|---|
  | BASELINE | 8.319 | 7.433 | 1.078 ms |
  | TAU2 v2  | 8.378 | 7.429 | 1.107 ms (+0.03) |
  AR[3] AG **unchanged** (noise amount & per-call gating didn't help).
  **Mechanism refuted** at the data level: HBM memory-controller idle state
  is NOT the cause of AR[N-1] AG being 2× slower than AR[0..N-2] AG.
- **Reverted** at this commit (all τ'' kernel / class / pybind / Python changes).
- **True bottleneck identified (from same run's no-copy numbers)**:
  | Mode | wall | Δ vs RCCL |
  |---|---|---|
  | SDMA **no-copy** | 6.933 | **-0.50 ms (WINS)** |
  | RCCL | 7.433 | — |
  | SDMA copy | 8.319 | +0.89 ms (loses) |
  The SDMA collective itself (sans copy) is already **faster than RCCL
  by 0.5 ms**. The entire ~1.4 ms "SDMA vs RCCL" gap is the **4 ×
  `hipMemcpyAsync` output copies** (~0.35 ms each, all on stream_ar,
  all serialize with the next AR). The AG-speed chase is a dead end;
  the copy overhead is the only remaining lever.
- **New direction candidates**:
  - **π'**: move `hipMemcpyAsync` to a dedicated SDMA copy queue (not CU
    blit, not stream_ar) so AR[N+1] kernel launches immediately after
    AR[N] kernel exits, with copy running in parallel. Expected gain:
    (#copies - 1) × 0.35 ≈ 1.05 ms if all but last hide perfectly.
    Historical attempt in this session used CU-blit on separate stream
    (failed due to CU contention); a pure-SDMA implementation has not
    been tried.
  - **D'' (fresh attempt)**: only pre-register output for identical
    (ptr,size) pairs seen at hot-loop time. Previous D' attempt hit
    PyTorch caching allocator offset mismatch with `hipIpcOpenMemHandle`;
    a registry keyed on `(ptr, size, pid)` with an explicit collective
    setup called from user code could bypass that.
  - **μ': in-kernel direct-to-user-output AG**. Similar to D' but the AR
    kernel does the final write itself rather than via `hipMemcpyAsync`.
    Also requires symm memory for user output.

## Entry 15 — ν (skip iter sync) FAILED, reveals true mechanism
- **Date**: 2026-04-22
- **Commit**: `623f61a4` (ν impl), `b56918ba` (baseline)
- **Scenario**: `tools/bench_tau_xi.sh` with `MORI_SKIP_ITER_SYNC=1`
- **Numbers (median wall ms, copy mode)**:
  | label | copy | RCCL | vs RCCL | AR[0] total | AR[3] total |
  |---|---|---|---|---|---|
  | BASELINE | 7.797 | 7.637 | +2.1% | 2.264 ms | 1.314 ms |
  | NU       | 5.247 | 5.684 | -7.7% *(artifact)* | 2.266 ms | 1.314 ms |
- **Mechanism analysis — ν FAILED as measurement artifact**:
  1. Wall diff 2.55ms is **fake gain**: ν mode has `ov_s.record(stream_ar)`
     capturing only stream_ar (no sync → stream_gemm runs decoupled ahead,
     timing shows GEMM start at -342ms relative to ov_s).
  2. RCCL wall (same measurement infra) **also** drops (7.637 → 5.684),
     confirming the drop is measurement-methodology-induced, not real gain.
  3. **AR[0] and AR[3] phase breakdowns are byte-for-byte identical**
     between BASELINE and NU → SDMA engine state unchanged.
     `torch.cuda.synchronize()` between iters is NOT the cause of AR[0]
     cold-path overhead.
  4. **Reverted** at commit `<this>`.
- **Revised mechanism hypothesis (from AR[0] vs AR[3] phase data)**:
  The gap between AR[0] cold and AR[3] warm is dominated by:
  1. **AR[0] c?:scatter→compute-wait** 0.46 + 0.36 = **0.82 ms**
     (vs AR[3] 0.11 + 0.09 = 0.20 ms). This is compute-block reduce
     blocked by CU contention from parallel GEMM[1]. AR[0] runs during
     GEMM[1], GEMM[1] steals CUs from AR[0]'s reduce. AR[3] has no parallel
     GEMM → fast reduce.
  2. **AR[3] AG-submit→AG-wait-done** = **1.082 ms** (vs AR[0] 0.566 ms).
     AR[3] AG is 2× slower than AR[0/1/2]. Surprising: warm-path AR is
     SLOWER than cold-path AR in this phase. Mechanism: during AR[0..N-2]
     AG wait there is a parallel GEMM generating HBM traffic, which keeps
     memory controller active, SDMA sees full bandwidth. During AR[N-1]
     AG wait there is NO parallel activity, HBM is quiet, memory
     controller drops into a low-power / idle state, SDMA throughput
     halves. Per-chunk AG sat bandwidth: 128MB / 0.55ms ≈ 230 GB/s in
     "active" state; 128MB / 1.08ms ≈ 118 GB/s in "idle" state. 2× gap.
- **τ'' proposal (next direction, see kernel commit)**:
  - In-kernel HBM noise during AG wait. Compute blocks (which already
    exist as `post_ag_wait` spin-wait in Stage 1 E') read from `input`
    buffer in their spin-wait loop, generating HBM traffic from within
    the AR kernel itself. No external GEMM needed.
  - Expected gain: AR[N-1] AG 1.082 → 0.55ms, save ~0.5ms on wall.
    Gap to RCCL is 0.16ms; gain/gap = 3×. **Satisfies R8**.
  - Requires `MORI_POST_AG_WAIT=1 MORI_HBM_NOISE=1`. Implementation is
    one spin-loop variant added to kernel; no correctness risk because
    the noise loop only reads (no writes) from `input` which is read-only
    from AR's semantic perspective.

---

## Entry 14 — τ (keep-HBM-hot) + ξ (AR warmup) both FAILED
- **Date**: 2026-04-22
- **Commit**: `95c663f8` (τ + ξ benchmark impl), measurement at `b56918ba`
- **Scenario**: `tools/bench_tau_xi.sh` (MORI_PIPELINE_CU=160, 256MB, 4 stages, 100 iters)
- **Numbers (median wall ms, copy mode)**:
  | label | copy | no-copy | RCCL | Δ copy vs OFF |
  |---|---|---|---|---|
  | OFF     | 7.789 | 7.413 | 7.582 | — |
  | TAU     | 8.424 | 8.072 | 8.468 | **+0.635** |
  | XI      | 7.789 | 7.417 | 7.666 | 0 |
  | TAU_XI  | 8.419 | 8.070 | 8.471 | +0.630 |
- **AR[3] duration (key target for τ)**:
  - OFF baseline not re-captured but Entry 4 shows AR[2] AG ~1.1 ms
  - TAU timeline shows AR[3] = 1.303 ms (τ did NOT shrink AG phase)
- **Mechanism analysis**:
  - **τ fail root cause**: the 2 extra GEMMs on stream_gemm forced
    `stream_gemm.synchronize()` at ov_e.record() to wait for them,
    extending wall by ~0.63 ms. Worse, AR[3] AG phase was NOT
    accelerated — τ hypothesis "cross-stream HBM activity keeps SDMA
    engine hot" is **refuted** at the direct AR[3] level.
  - **ξ fail root cause**: benchmark already has 10-iter warmup loop
    that equally warms AR[0] cold-path. An extra single AR call before
    the measurement loop adds nothing. AR[0] `entry→scatter_done`
    (~0.22 ms cold-path overhead) is NOT due to first-call staleness
    but to **per-measurement-iter prep_ar + torch.cuda.synchronize**
    which drains and cools SDMA queues between iters.
  - **Refined understanding**: The cold-path in AR[0] is an
    **intra-iter** phenomenon (driven by `torch.cuda.synchronize()`
    between iters draining SDMA queues), not an inter-run phenomenon.
    Warming before the loop starts doesn't help because each iter
    re-cools. True fix would be: don't synchronize between iters
    (breaks isolation), OR reduce the submit→doorbell latency inside
    the kernel when queue is cold.
- **Conclusion**:
  - τ direction **closed**: extra GEMM on stream_gemm strictly extends
    wall; no hidden gain elsewhere.
  - ξ direction **closed**: warmup before loop is equivalent to the
    existing warmup iters; no new gain available at host level.
  - **Reverted** at commit: Python `_bench_overlap_one_size`
    `MORI_KEEP_HBM_HOT` / `MORI_AR_WARMUP` paths removed.
- **New evidence for next direction** (Entry 4 revisited with TAU_XI
  AR[0] data):
  - AR[0] `c0:barrier→AG-submit` = **0.247 ms** per chunk (×2 chunks
    = 0.494 ms) vs AR[2/3] warm = **0.021 ms** per chunk (×2 = 0.042
    ms). Delta = **0.452 ms**.
  - This is the dominant AR[0] cold-path cost, NOT entry overhead.
  - Mechanism: SDMA ring doorbell / packet submission from kernel-land
    is slow when queue is cold (~250 us), fast when queue has recent
    traffic (~20 us). Likely an SDMA engine kickoff latency, not a
    ring fullness or CU contention issue.
  - This **elevates σ direction** (keep SDMA queue warm **between**
    AR iters, not just once before loop) to top priority.

---

## Running gap-to-target tally

| date | best SDMA copy wall | RCCL median | gap | notes |
|---|---|---|---|---|
| 2026-04-21 | 7.746 | 7.425 (noisy) | +2.2% | CU=160 baseline |
| 2026-04-21 | 7.758 (3-run) | 7.562 (3-run) | **+2.6%** | stable gap after 3-run characterization |
| 2026-04-22 | 7.744 | 7.672 | +0.9% | CU=220 (optimal); gap dominated by RCCL run noise |
| 2026-04-22 | 8.940 | 7.424 | **+20.4% (WORSE)** | Stage 2b-1 catastrophic, reverted |
| 2026-04-22 | 7.765 | 7.433 | +4.5% | Stage 2b-0 clean, baseline restored |
| 2026-04-22 | _projected 7.381_ | _7.562_ | **_-2.4% (超 RCCL)_** | D' steady state projection (Entry 10) — **invalidated by Entry 11 correctness fail** |
| 2026-04-22 | — | — | — | D' Step 2 **REVERTED** (Entry 11): PyTorch caching allocator incompatible with `hipIpcOpenMemHandle` semantics; direction closed pending shmem refactor |
| 2026-04-22 | 7.789 | 7.582 | **+2.7%** | Entry 14 baseline (τ=ξ=off). Both τ and ξ failed; reverted |
| 2026-04-22 | 7.797 | 7.637 | **+2.1%** | Entry 15 baseline (ν fail confirmed); τ'' proposal pending |
| 2026-04-22 | 8.319 | 7.433 | **+11.9%** | Entry 16: τ'' FAILED in both v1 & v2; no-copy mode already beats RCCL by 0.5ms → copy cost is the only lever left |
| 2026-04-22 | 7.806 | 7.424 | **+5.1%** | Entry 17: E'' user-proposed, 3 variants ALL FAILED. In-kernel copy during AG wait hits fundamental HBM contention with SDMA engine on MI355X |
| 2026-04-23 | 7.798 | 7.422 | **+5.1%** | Entry 18 baseline (Plan A/B 对照组, iter=100) |
| 2026-04-23 | **8.955** | 7.489 | **+19.6% (WORSE)** | Entry 18: Plan A/B FAILED, wall +1.16ms 退化. CU AG + GEMM 抢 CU. 代码未 revert(user 指示暂保留,debug β partition 比例)|

**Target**: SDMA copy ≤ RCCL (gap ≤ 0). Currently outstanding.

---

## Entry 20 — Plan A FAILED (+1.58 ms vs RCCL); root cause is CU-as-AG architectural mismatch with GEMM overlap
- **Date**: 2026-04-28
- **Plan A code SHA**: `611c2f2c` (post-fix; chain `e7050bc6` Plan A swap + `bb8911f4` test syntax fix + `9db1d420` stuck instrumentation + `611c2f2c` phase_ts capacity 32→128)
- **Bench command**: `bash tools/bench_plan_a.sh` — env `MORI_PIPELINE_CU=160 MORI_DIRECT_OUTPUT=1`, `--num-stages 4 --elems 67108864 --iterations 50 --warmup 10` (wall) + `--iterations 3 --warmup 2 --ar-phase-timing` (phase)
- **All Test 1/1b/2/3/4/5/6 PASSED correctness on Plan A path**

### Wall data (256 MB / stage × 4 stages, MI355X 8-rank, today)

| variant | overlap wall | seq_ar | seq_gemm | overlap/seq_gemm | vs RCCL |
|---|---|---|---|---|---|
| **Plan A** (SDMA copy + Plan A active) | **9.084 ms** | 5.845 | 4.129 | **2.200** | **+21.1% (+1.581 ms WORSE)** |
| SDMA no-copy (baseline path, no Plan A) | 7.355 ms | 4.801 | 3.991 | 1.843 | -2.0% |
| RCCL today | 7.503 ms | 5.133 | 3.977 | 1.887 | — |

vs Entry 19 BASELINE (no Plan A) 7.802 ms: **Plan A 慢 1.28 ms (+16.4%)**.

### Root cause analysis (mechanism, not just numbers)

**The fundamental architectural mismatch**: Plan A moves the AllGather phase from the SDMA engine (an independent hardware unit) to the CU (which GEMM also needs). In the multi-stage GEMM+AR overlap test (Test 6), AR and GEMM run concurrently — any algorithm whose AR phase fights GEMM for CU loses overlap.

**Mechanism 1: CU AG breaks chunk-level pipeline (seq_ar +0.66 ms)**

BASELINE chunk pipeline (depth 2):
```
CU:    [reduce c0] [reduce c1] [reduce c2] ...    ← 一直占着 CU 做 reduce
SDMA:              [AG c0    ] [AG c1    ] ...    ← 独立 SDMA engine 做 AG
```
SDMA AG runs on a separate hardware engine, so chunk c+1 reduce on CU runs in parallel with chunk c SDMA AG. Each chunk's wall ≈ max(reduce, SDMA AG).

Plan A chunk timeline (depth 1):
```
CU:    [reduce c0][barrier][AG_pull c0][reduce c1][barrier][AG_pull c1]...
```
CU XGMI pull AG occupies the same CU hardware, so the next chunk's reduce cannot start until the current chunk's AG pull completes. Strict serial. Per-chunk wall ≈ reduce + barrier + AG_pull instead of max(...).

Per-chunk phase data (block 0 c1..c7 deltas, c0 was garbage from slot-1 unwritten): **0.30 ms/chunk** in Plan A. 8 chunks × 0.30 = 2.4 ms upper bound for one AR. seq_ar 5.845 / 4 = 1.46 ms/AR matches the per-chunk depth-1 estimate (some overlap still happens between block 0 barrier and CB AG pull).

**Mechanism 2: CU占用密度 with GEMM overlap (overlap +1.28 ms / 4 AR = +0.32 ms/AR)**

CU occupation per AR by algorithm:

| algorithm | per-AR CU work | overlap/seq_gemm ratio (measured) |
|---|---|---|
| RCCL ring | sparse: 14 steps × ~0.05 ms reduce, idle in between | **1.89** (today) |
| BASELINE SDMA AG | dense reduce only (~0.30 ms) | **2.12** (Entry 19) |
| Plan A CU pull AG + write user_output | reduce + barrier + pull + local write (~1.0 ms) | **2.20** (today) |
| Plan B CU push AG + CU copy | reduce + push + copy (~0.95 ms) | **2.44** (Entry 18) |

**Monotone relationship: more CU occupancy in AR phase → larger GEMM extension in overlap → larger wall**. This is observed across 4 algorithms with different CU usage patterns. Plan A 2.20 is the third-worst point on this curve. RCCL wins because its ring is intrinsically pipelined and CU usage is sparse.

**Why RCCL ring is fast**: each of 14 steps does receive 32MB → reduce 32MB → send 32MB. Reduce step takes ~0.05 ms of CU. Between steps the CU is idle (XGMI traffic moving). Sparse CU occupancy → low GEMM contention → overlap/seq_gemm 1.89 (the lowest).

**Conclusion**: Plan A failure is **architectural, not implementation**. No vectorize/partition/CU-count tuning can recover 1.58 ms (Mechanism 2 alone is 1.28 ms wall regression, fundamentally tied to CU sharing with GEMM). Plan B (Entry 18) failed via the same mechanism (overlap/seq_gemm 2.44, even worse). **Any CU-based AG approach loses to BASELINE/RCCL in the overlap scenario** because BASELINE's SDMA AG is essentially "free CU time" (the SDMA engine is otherwise idle and not contended by GEMM).

### Lessons (R6 reflection)

- **R10 historical reference miss**: Entry 19 predicted Plan A wall 6.7-6.9 ms with "0.3-0.5 ms CU-GEMM contention upper bound from Plan B's 0.81 regression". Actual contention 1.28 ms (3x prediction). Reason: Plan A's CU AG pull is **3x more CU-intensive than Plan B's CU push** (Plan A also does the local write to user_output that Plan B left for a separate C-group). Should have enumerated the CU work breakdown, not extrapolated from one prior data point.
- **R6 reflection on architecture choice**: any future scheme that moves AG/copy onto CU during GEMM overlap should be rejected up-front by mechanism reasoning, not benchmarked. The "CU-occupancy ↔ overlap-ratio" data series across BASELINE/Plan A/Plan B is now solid enough to predict any new candidate.
- **R0 / phase timing OOB bug fixed in this commit chain**: kArPhaseTsCapacity 32 → 128 with bound check on helpers (`611c2f2c`). slot 11+3*numChunks=35 OOB at numChunks=8 corrupted CrossPeBarrier sync state and deadlocked the next launch. Was the trigger for the original 30-min NCCL watchdog timeouts seen in early Plan A runs; root cause not Plan A logic itself.

### Closed direction

**Plan A — PipelinedXGMIPullKernel**: failed +1.58 ms vs RCCL. Mechanism is architectural CU contention with GEMM. Code retained at HEAD pending user decision on revert vs. extracting useful instrumentation pieces.

### Remaining viable directions (must keep SDMA AG to avoid CU contention)

| # | direction | mechanism | predicted gain | blocker |
|---|---|---|---|---|
| 1 | **D''** | SDMA AG writes directly to user_output (register user_output as symm memory), eliminate transit→user_output copy entirely | 1.4 ms (4 × 0.35 ms hipMemcpyAsync removed) → wall ~6.93 ms (-0.5 ms vs RCCL) | user must call `register_user_output(ptr,size)` once before hot loop; D' (Entry 11) failed on PyTorch caching allocator + IPC offset, but D'' lets user control lifetime explicitly |
| 2 | **π'** | move transit→user_output copy onto a separate SDMA queue with double-buffered transit; next AR doesn't wait for prior copy | (N-1) × 0.35 = 1.05 ms (3 of 4 copies hidden, last tail exposed) → wall ~6.75 ms | needs a 3rd SDMA queue; +256 MB transit memory |

Both keep SDMA AG → no CU contention → consistent with the mechanism above.

---

## Entry 21 — Plan A v2 implementation: split K2 into R-group reduce and A-group XGMI-pull pipeline
- **Date**: 2026-04-29
- **Commit**: `f744d41a`
- **Why this exists**: Entry 20 correctly measured the first Plan A implementation
  as **9.084 ms** (worse than RCCL 7.503 ms), but the implementation was not
  the user-intended "CU pipelined" Plan A. The kernel serialized:
  `reduce(c) -> barrier(c) -> XGMI_pull(c) -> reduce(c+1)`. That loses the
  intended overlap between pulling chunk `c` and reducing chunk `c+1`.

### Implementation delta

K1 remains unchanged:
- `ScatterSdmaOnlyKernel`: SDMA push input shards to peer transit.

K2 `PipelinedXGMIPullKernel` is changed from all-CB serial loop to two compute
groups:

| group | blocks | work |
|---|---|---|
| Block 0 | 1 | wait `chunks_complete == (c+1)*nR`, cross-PE reduce_complete barrier, signal `ag_sync` |
| R-group | `nR` | wait scatter signal, reduce chunk `c` into local `transit[myPe slot]`, fence, `fetch_add(chunks_complete)` |
| A-group | `nA = compBlocks - nR` | wait `ag_sync >= c+1`, XGMI pull `peer[p].transit[p slot][chunk c]` to `user_output[p slot][chunk c]` |

This restores the intended pipeline:
```
R-group: reduce0 -> reduce1 -> reduce2 -> ...
A-group:           pull0   -> pull1   -> ...
```

### CU pressure controls

User pointed out 161 blocks/CUs likely starves GEMM. Host now uses separate
Plan A knobs:
- `MORI_PLAN_A_CU` (default **96**) caps total Plan A compute blocks
  (not counting block 0). This leaves more CUs for GEMM than the previous
  160-block default.
- `MORI_PLAN_A_NR` (default `MORI_PLAN_A_CU/3`) sets R-group size; A-group
  gets the remaining blocks. Default R:A ~= 1:2 because AG pull reads+writes
  more bytes than reduce writes.
- Host clamps `nA >= npes` so every output slot has at least one A block.

### Instrumentation

- Existing R-group slots (`10`, `11+3c+{0,1,2}`, `11+3*numChunks`) still report
  first R-block reduce phases.
- New A-group slots:
  - `49`: first A-block entry
  - `50+3c+0`: chunk c AG wait start
  - `50+3c+1`: chunk c `ag_sync` observed
  - `50+3c+2`: chunk c pull done
  - `50+3*numChunks`: first A-block exit
- `test_allreduce.py --ar-phase-timing` now prints a "Plan A A-group" section
  when those slots are populated.

### Test plan

Build:
```bash
BUILD_EXAMPLES=ON BUILD_TESTS=ON pip3 install .
```

First run correctness + wall at default 96/32 split:
```bash
MORI_PIPELINE_CU=160 MORI_DIRECT_OUTPUT=1 python3 tests/python/ccl/test_allreduce.py \
  --num-stages 4 --elems 67108864 --iterations 50 --warmup 10
```

Then sweep CU split using same command:
```bash
MORI_PLAN_A_CU=80  MORI_PLAN_A_NR=24 ...
MORI_PLAN_A_CU=96  MORI_PLAN_A_NR=32 ...
MORI_PLAN_A_CU=128 MORI_PLAN_A_NR=48 ...
```

Measure AR[0] phase with:
```bash
MORI_PIPELINE_CU=160 MORI_DIRECT_OUTPUT=1 MORI_PHASE_TARGET_STAGE=0 \
  python3 tests/python/ccl/test_allreduce.py --num-stages 4 --elems 67108864 \
  --iterations 3 --warmup 2 --ar-phase-timing
```

Success target remains: Plan A wall < RCCL wall (~7.50 ms today, 7.42 ms
Entry 18 best).

---

## Entry 22 — Plan A v2 default 96/32 result: seq_ar fixed, gap now entirely GEMM overlap contention
- **Date**: 2026-04-29
- **Commit under test**: `f744d41a` / docs follow-up `5a6c0b6d`
- **Config**: default Plan A v2 split, `MORI_PLAN_A_CU=96`, `MORI_PLAN_A_NR=32`
  (log should show `blocks=97, nR=32, nA=64`)
- **Command**: `MORI_PIPELINE_CU=160 MORI_DIRECT_OUTPUT=1 python3 tests/python/ccl/test_allreduce.py --num-stages 4 --elems 67108864 --iterations 50 --warmup 10`

| variant | overlap wall | seq_ar | seq_gemm | overlap/seq_gemm | vs RCCL |
|---|---|---|---|---|---|
| **Plan A v2 default** | **8.570 ms** | **5.105** | 3.674 | **2.333** | **+14.9% (+1.111 ms WORSE)** |
| SDMA no-copy | 7.418 ms | 4.775 | 3.673 | 2.020 | +0.5% |
| RCCL | 7.459 ms | 5.129 | 3.675 | 2.029 | — |

### Mechanism update

Plan A v2 fixed the main v1 implementation bug:
- v1 seq_ar = 5.845 ms
- v2 seq_ar = **5.105 ms**
- improvement = **0.740 ms** over 4 ARs (= 0.185 ms/AR)
- v2 seq_ar is now slightly **faster than RCCL** (5.105 vs 5.129)

The algorithmic AR work is no longer the blocker. The remaining gap is overlap:
- Plan A v2 overlap wall = 8.570 ms
- RCCL overlap wall = 7.459 ms
- gap = **1.111 ms**
- Plan A v2 overlap/seq_gemm = **2.333**
- RCCL overlap/seq_gemm = **2.029**

Root cause now: Plan A still uses too much CU during AR, starving GEMM. This is
now tunable via `MORI_PLAN_A_CU` and `MORI_PLAN_A_NR`; v2 default used 96 compute
blocks and still over-contended GEMM.

### Next experiment

Added `tools/bench_plan_a_sweep.sh` to sweep CU/R split with one command.
Default variants:
- `80:24`
- `96:32`
- `112:40`
- `128:48`

Run:
```bash
cd /home/fizhang/test/mori && git pull origin sdma-test
SKIP_BUILD=1 bash tools/bench_plan_a_sweep.sh
```

Decision rule:
- If lower CU improves wall while seq_ar remains near RCCL (~5.13 ms), keep reducing CU.
- If seq_ar grows faster than GEMM contention shrinks, use the best wall split.

---

## Entry 23 — Plan A v2 CU/R split sweep: best 128/48 still +0.87 ms worse than RCCL
- **Date**: 2026-04-29
- **Commit under test**: `4de2c44a` (sweep script + Plan A v2)
- **Command**: `SKIP_BUILD=1 bash tools/bench_plan_a_sweep.sh`
- **Scenario**: 256MB / stage × 4 stages, 8-rank MI355X, `ITERATIONS=50`, `WARMUP=10`
- **All four variants correctness PASSED**

| variant | Plan A active | wall | seq_ar | GEMM slowdown | RCCL wall (same run) | gap vs RCCL |
|---|---|---:|---:|---:|---:|---:|
| `CU80_NR24` | blocks=81, nR=24, nA=56 | 8.757 | 5.202 | 2.383 | 7.596 | +1.161 |
| `CU96_NR32` | blocks=97, nR=32, nA=64 | 8.568 | 5.104 | 2.331 | 7.432 | +1.136 |
| `CU112_NR40` | blocks=113, nR=40, nA=72 | 8.488 | 5.113 | 2.310 | 7.462 | +1.026 |
| **`CU128_NR48`** | blocks=129, nR=48, nA=80 | **8.467** | 5.137 | **2.304** | 7.594 | **+0.873** |

### Observations

1. Lower CU did **not** improve overlap. It worsened both wall and GEMM slowdown:
   - CU80 wall 8.757, slowdown 2.383
   - CU128 wall 8.467, slowdown 2.304
   This refutes the hypothesis that simply lowering Plan A CU occupancy fixes
   the GEMM contention problem.

2. `seq_ar` is no longer the blocker:
   - All variants are near RCCL seq_ar (~5.12 ms)
   - Best wall variant CU128 has seq_ar 5.137 vs RCCL 5.125 (same run),
     only +0.012 ms.

3. The remaining gap is entirely overlap quality:
   - Best Plan A slowdown 2.304 vs RCCL 2.067 (same CU128 run)
   - wall gap +0.873 ms even when sequential AR is equal to RCCL.

4. Increasing CU from 80→128 improves wall, but slope is too small to close
   the remaining gap:
   - wall improves 0.290 ms over +48 CU
   - remaining gap at CU128 is still 0.873 ms
   - linear extrapolation would need >140 additional CU, impossible on this
     setup and likely to worsen GEMM contention.

### Mechanism

Plan A v2 fixed the chunk-serialization bug, but the A-group still performs
large CU XGMI pull + local write during the same window where GEMM needs CU.
The AR algorithm itself is competitive (`seq_ar ~= RCCL`), but the overlap
schedule is not: Plan A occupies CU too continuously, while RCCL ring has
sparser CU bursts between communication steps.

### Conclusion

Plan A v2 remains **not sufficient** for the hard target
`copy_output_to_user=True wall < RCCL wall`. Best measured wall is 8.467 ms
vs RCCL 7.594 ms (+0.873 ms worse). Further CU/R split tuning is unlikely to
recover enough gap; the next viable directions are still:
- **D''**: explicit `register_user_output(ptr,size)` so SDMA AG directly writes
  user_output (eliminate copy without adding CU work)
- **π'**: keep SDMA AG and move transit→user_output copy to independent SDMA
  queue with double-buffered transit

---

## Entry 24 — Plan A v3 chunk-density sweep (16 chunks) FAILED
- **Date**: 2026-04-29
- **Commit under test**: `4de2c44a` / Plan A v2 code path
- **Command**: `SKIP_BUILD=1 VARIANTS="96:32 128:48" MORI_DIRECT_CHUNKS=16 bash tools/bench_plan_a_sweep.sh`
- **Scenario**: 256MB / stage × 4 stages, 8-rank MI355X, `ITERATIONS=50`, `WARMUP=10`
- **All variants correctness PASSED**

| variant | Plan A active | wall | seq_ar | GEMM slowdown | RCCL wall (same run) | gap vs RCCL |
|---|---|---:|---:|---:|---:|---:|
| `CU96_NR32`, chunks=16 | blocks=97, nR=32, nA=64, chunk_elems=4194304 | 8.728 | 5.273 | 2.377 | 7.442 | +1.286 |
| `CU128_NR48`, chunks=16 | blocks=129, nR=48, nA=80, chunk_elems=4194304 | 8.701 | 5.339 | 2.369 | 7.577 | +1.124 |

### Comparison vs chunks=8 (Entry 23)

| split | chunks=8 wall | chunks=16 wall | chunks=8 seq_ar | chunks=16 seq_ar | chunks=8 slowdown | chunks=16 slowdown |
|---|---:|---:|---:|---:|---:|---:|
| 96/32 | 8.568 | **8.728** | 5.104 | **5.273** | 2.331 | **2.377** |
| 128/48 | 8.467 | **8.701** | 5.137 | **5.339** | 2.304 | **2.369** |

### Mechanism

The "shorter CU burst" hypothesis is refuted. Increasing chunk count from 8
to 16 makes both sides worse:
- `seq_ar` worsens by +0.17 to +0.20 ms (more per-chunk barrier/sync overhead)
- GEMM slowdown worsens by +0.046 to +0.065 (more frequent CU wake/dispatch and
  synchronization, not less contention)
- wall worsens by +0.16 to +0.23 ms

Plan A's bottleneck is not just burst granularity; it is the fact that the AG
path performs substantial CU XGMI pull + local user_output write during the
GEMM overlap window. Smaller chunks add overhead without making that CU work
sparse enough to match RCCL.

### Conclusion

Plan A v3 chunk-density tuning is closed. Best Plan A remains Entry 23
`CU128_NR48, chunks=8`: 8.467 ms, still +0.873 ms worse than RCCL. Do not spend
more iterations on Plan A CU scheduling unless a new mechanism appears.

Because user ruled out explicit `register_user_output` (D'') due to non-reused
buffers, the next candidate is **π' / local SDMA copy path**:
- keep current SDMA scatter + reduce + SDMA AG (no extra CU work)
- replace external `hipMemcpyAsync` CU blit transit→user_output with an internal
  local SDMA copy path, or run that copy on a separate SDMA queue with double
  buffering so later AR kernels do not wait on previous copy
- does not require user buffer reuse or API change

---

## Entry 25 — Add Test 6 continuous-iters mode to model real continuous workload
- **Date**: 2026-04-29
- **Commit**: `2ff3e94f`
- **User correction**: real workload is continuous (e.g. 100 consecutive
  requests/micro-batches), not isolated finite 4-stage benchmark iterations.
  Current Test 6 finite mode has `torch.cuda.synchronize()` at the start/end
  of every measured iteration, so the last AR/copy tail is exposed every
  iteration. In continuous serving, that tail can overlap the next iteration's
  GEMM[0].

### Implementation

Added CLI flag:
```bash
--continuous-iters N
```

Only affects Test 6 overlap timing:
- `seq_ar` and `seq_gemm` are still measured with the existing finite method
  for decomposition.
- overlap timing becomes one long measured window:
  - untimed warmup: `warmup` multi-stage pipelines back-to-back
  - measured: `N` multi-stage pipelines back-to-back
  - no `torch.cuda.synchronize()` between those N pipelines
  - reported overlap wall = total measured wall / N

This directly tests whether current finite-mode SDMA copy loses only because
the per-iteration tail copy is repeatedly exposed.

### Usage

```bash
cd /home/fizhang/test/mori && git pull origin sdma-test
SKIP_BUILD=1 CONTINUOUS_ITERS=100 ITERATIONS=1 WARMUP=1 bash tools/bench_plan_a.sh
```

Manual baseline:
```bash
MORI_PIPELINE_CU=160 python3 tests/python/ccl/test_allreduce.py \
  --num-stages 4 --elems 67108864 --iterations 1 --warmup 1 \
  --continuous-iters 100
```

### Expected decision

If continuous-mode **SDMA copy wall < RCCL wall**, then no kernel change is
needed for the real workload; the finite Test 6 target was overly pessimistic.
If continuous-mode SDMA copy is still slower, the remaining gap is not just
tail-copy exposure and requires algorithm/copy-path work.

---

## Entry 26 — Continuous 100x measurement: finite-tail hypothesis FAILED; RCCL still much faster
- **Date**: 2026-04-29
- **Commit under test**: `2ff3e94f` / script before follow-up fix
- **Command**: `CONTINUOUS_ITERS=100 ITERATIONS=1 WARMUP=1 bash tools/bench_plan_a.sh`
- **Scenario**: Test 6 overlap is reported as wall per iteration across 100
  consecutive 4-stage pipelines with no inter-iteration sync.
- **All correctness checks PASSED**

### Continuous wall data

From `BASELINE` block:

| mode | continuous wall/iter | seq_ar | seq_gemm | slowdown |
|---|---:|---:|---:|---:|
| **SDMA copy (baseline)** | **7.143 ms** | 5.197 | 4.307 | 1.658 |
| SDMA no-copy | 6.571 ms | 4.778 | 4.136 | 1.589 |
| RCCL | **5.836 ms** | 5.127 | 4.213 | 1.385 |

From `PLAN_A` block:

| mode | continuous wall/iter | seq_ar | seq_gemm | slowdown |
|---|---:|---:|---:|---:|
| **Plan A copy** | **7.565 ms** | 5.111 | 4.328 | 1.748 |
| Plan A no-copy | 6.571 ms | 4.772 | 4.154 | 1.582 |
| RCCL | **5.834 ms** | 5.126 | 4.121 | 1.416 |

### Interpretation

The "finite 4-stage benchmark repeatedly exposes tail copy" hypothesis is
refuted for the current implementation:
- Baseline SDMA copy is still **+1.307 ms slower** than RCCL in continuous mode
  (7.143 vs 5.836).
- Even SDMA no-copy is **+0.735 ms slower** than RCCL (6.571 vs 5.836).
- Plan A copy is worse: 7.565 vs RCCL 5.834 (+1.731 ms).

Therefore the finite-mode gap is not just final-tail copy exposure. In steady
state, RCCL's pipeline throughput is substantially better. The problem moved
from "hide copy tail" to "SDMA two-shot steady-state algorithm throughput and
overlap are worse than RCCL".

### Caveat / follow-up fixes

`tools/bench_plan_a.sh` incorrectly passed `CONTINUOUS_ITERS=100` into the
`--ar-phase-timing` diagnostic runs, producing meaningless all-zero/gibberish
phase tables. A follow-up commit changes `run_variant_extra()` to force
`--continuous-iters 0` for phase/timeline diagnostics. The Table 1-4 continuous
wall data above are still valid.

Second caveat found after comparing finite timeline: the first implementation
called `prep_ar()` only once before the 100 measured continuous pipelines. Real
continuous workload has fresh input per request. A follow-up changes continuous
mode to enqueue `prep_ar()` once per logical iteration on `stream_ar`, without a
global sync. The original Entry 26 wall data should be treated as invalid until
re-run with that fix.

### Next direction

Plan A / copy-tail hiding should be deprioritized. The next useful investigation
is why continuous SDMA no-copy (6.571 ms) is slower than RCCL (5.836 ms) despite
the finite no-copy data previously looking competitive. That points to the
two-shot SDMA algorithm's steady-state scheduling/throughput, not the final
copy.

---

## Entry 27 — Corrected continuous 100x (prep per logical iter): SDMA still loses, but gap shrinks
- **Date**: 2026-04-29
- **Commit under test**: `0965c2e1` (continuous mode fixed to enqueue
  `prep_ar()` once per logical iteration on `stream_ar`, without global sync)
- **Command**: `SKIP_BUILD=1 CONTINUOUS_ITERS=100 ITERATIONS=1 WARMUP=1 bash tools/bench_plan_a.sh`
- **All correctness checks PASSED**

### Corrected continuous wall data

From `BASELINE` block:

| mode | continuous wall/iter | seq_ar | seq_gemm | slowdown | gap vs RCCL |
|---|---:|---:|---:|---:|---:|
| **SDMA copy** | **7.251 ms** | 5.196 | 4.328 | 1.675 | **+0.999 ms** |
| **SDMA no-copy** | **6.725 ms** | 4.786 | 4.166 | 1.614 | **+0.473 ms** |
| RCCL | **6.252 ms** | 5.145 | 4.137 | 1.511 | — |

From `PLAN_A` block:

| mode | continuous wall/iter | seq_ar | seq_gemm | slowdown | gap vs RCCL |
|---|---:|---:|---:|---:|---:|
| Plan A copy | 7.642 ms | 5.120 | 4.331 | 1.765 | +1.398 ms |
| Plan A no-copy | 6.734 ms | 4.774 | 4.196 | 1.605 | +0.490 ms |
| RCCL | 6.244 ms | 5.131 | 4.109 | 1.520 | — |

### Interpretation

Fixing continuous prep changed the numbers materially vs Entry 26:
- Baseline SDMA copy gap improved from +1.307 ms → **+0.999 ms**
- SDMA no-copy gap improved from +0.735 ms → **+0.473 ms**
- Plan A still worse than baseline copy (+1.398 ms vs RCCL)

The corrected continuous measurement still shows:
1. **Copy is not the only gap**:
   - copy - no-copy = 7.251 - 6.725 = **0.526 ms**
   - no-copy still trails RCCL by **0.473 ms**
2. **No-copy AR algorithm is faster sequentially, but worse in overlap**:
   - no-copy seq_ar 4.786 ms vs RCCL 5.145 ms (SDMA no-copy is **0.359 ms faster**)
   - no-copy slowdown 1.614 vs RCCL 1.511 (SDMA no-copy has worse overlap quality)
3. **Remaining problem after removing copy is overlap scheduling**, not AR
   compute itself. Approx overlap-quality penalty:
   `4.166 * (1.614 - 1.511) ≈ 0.43 ms`, matching the no-copy gap 0.473 ms.

### Next target

To beat RCCL in continuous real workload, two things are needed:
- remove/hide about **0.526 ms** copy gap (copy path)
- reduce no-copy overlap penalty by about **0.47 ms** (scheduling/overlap)

Plan A does not solve either in current form. The next useful experiment is to
measure continuous no-copy timeline/phase cleanly (finite phase is not enough)
or design an AR scheduling change that makes SDMA no-copy overlap more like
RCCL while keeping the faster seq_ar.

### Immediate hypothesis to test

The corrected continuous mode includes `prep_ar()` inside the measured window
once per logical iteration. In this synthetic benchmark `prep_ar()` is a 256MB
`fill_` on `stream_ar`; finite `seq_ar/seq_gemm` do **not** include that cost.
The no-copy continuous gap vs RCCL is ~0.47ms, which is the right order for an
extra 256MB HBM fill/copy kernel. A follow-up test flag
`MORI_CONTINUOUS_PREP=0` skips this per-iteration prep to isolate whether the
gap is a benchmark-prep artifact or true AR/GEMM overlap loss.

### Superseded

Entry 27 still has a continuous-mode event-reuse bug: it reuses the same
`ev_g_e_list[s]` event across all 100 logical iterations. Because the host
enqueues all iterations quickly, the event can be re-recorded before an earlier
`stream_ar.wait_event(event)` executes, so AR[i,s] may wait on a later GEMM
event instead of GEMM[i,s]. Treat Entry 27 numbers as invalid pending Entry 28.

---

## Entry 28 — Fix continuous event reuse bug; allocate per-iteration GEMM-done events
- **Date**: 2026-04-29
- **Commit**: `60b8ba56`
- **Bug**: `--continuous-iters` reused `ev_g_e_list[s]` across all logical
  iterations. In continuous mode there is no per-iteration sync, so the CPU
  enqueue loop can re-record the same event before earlier stream waits execute.
  This means `stream_ar.wait_event(ev_g_e_list[s])` does not necessarily wait
  for the matching GEMM of the same logical iteration/stage.
- **Fix**: allocate `cont_g_done[iter][stage]` unique events in continuous mode
  and make AR[i,s] wait on `cont_g_done[i][s]`.
- **Re-run required**:

```bash
cd /home/fizhang/test/mori && git pull origin sdma-test
SKIP_BUILD=1 MORI_CONTINUOUS_PREP=0 CONTINUOUS_ITERS=100 ITERATIONS=1 WARMUP=1 \
  bash tools/bench_plan_a.sh
```

Then run with prep enabled:
```bash
SKIP_BUILD=1 CONTINUOUS_ITERS=100 ITERATIONS=1 WARMUP=1 bash tools/bench_plan_a.sh
```

---

## Entry 29 — Continuous 100x after event fix: no-copy still loses; overlap quality is root gap
- **Date**: 2026-04-29
- **Commit under test**: `60b8ba56` / docs follow-up `04c77185`
- **Commands**:
  - no-prep: `MORI_CONTINUOUS_PREP=0 SKIP_BUILD=1 CONTINUOUS_ITERS=100 ITERATIONS=1 WARMUP=1 bash tools/bench_plan_a.sh`
  - prep: `SKIP_BUILD=1 CONTINUOUS_ITERS=100 ITERATIONS=1 WARMUP=1 bash tools/bench_plan_a.sh`
- **All correctness checks PASSED**

### No-prep continuous (isolates AR/GEMM scheduling; no synthetic input fill)

| block | mode | wall/iter | slowdown | seq_ar | seq_gemm | gap vs RCCL |
|---|---|---:|---:|---:|---:|---:|
| BASELINE | SDMA copy | 7.118 | 1.656 | 5.198 | 4.298 | +1.327 |
| BASELINE | **SDMA no-copy** | **6.567** | **1.587** | **4.774** | 4.137 | **+0.776** |
| BASELINE | RCCL | 5.791 | 1.396 | 5.130 | 4.150 | — |
| PLAN_A | Plan A copy | 7.563 | 1.742 | 5.120 | 4.342 | +1.757 |
| PLAN_A | Plan A no-copy | 6.552 | 1.568 | 4.776 | 4.177 | +0.746 |
| PLAN_A | RCCL | 5.806 | 1.402 | 5.132 | 4.141 | — |

### Prep-enabled continuous (fresh input per logical iter)

| block | mode | wall/iter | slowdown | seq_ar | seq_gemm | gap vs RCCL |
|---|---|---:|---:|---:|---:|---:|
| BASELINE | SDMA copy | 7.236 | 1.672 | 5.194 | 4.327 | +1.009 |
| BASELINE | **SDMA no-copy** | **6.726** | **1.605** | **4.776** | 4.191 | **+0.499** |
| BASELINE | RCCL | 6.227 | 1.507 | 5.125 | 4.131 | — |
| PLAN_A | Plan A copy | 7.640 | 1.765 | 5.106 | 4.328 | +1.405 |
| PLAN_A | Plan A no-copy | 6.736 | 1.611 | 4.791 | 4.183 | +0.501 |
| PLAN_A | RCCL | 6.235 | 1.506 | 5.138 | 4.140 | — |

### Mechanism

The measurement bug is fixed (unique GEMM-done events per iter/stage) and prep
is isolated. Both no-prep and prep-enabled runs show the same structural fact:

1. **SDMA no-copy is faster sequentially than RCCL**:
   - no-prep: 4.774 vs 5.130 (SDMA faster by 0.356 ms)
   - prep: 4.776 vs 5.125 (SDMA faster by 0.349 ms)

2. **SDMA no-copy overlaps worse with GEMM**:
   - no-prep slowdown: 1.587 vs RCCL 1.396
   - prep slowdown: 1.605 vs RCCL 1.507

3. The overlap-quality penalty dominates the sequential AR advantage:
   - no-prep: `4.137 * (1.587 - 1.396) = 0.790 ms`, matching the measured
     no-copy gap +0.776 ms.
   - prep: `4.191 * (1.605 - 1.507) = 0.411 ms`, close to measured +0.499 ms.

Therefore the current bottleneck is **not copy tail** and **not sequential AR
speed**. It is that SDMA no-copy degrades GEMM overlap more than RCCL does.
RCCL's sequential AR is slower, but its overlap schedule is better.

### Next target

Need a continuous-mode timeline (not finite phase) that reports per-iteration
and per-stage GEMM/AR start/end for a small sample window, using unique events
per iter/stage, to see whether:
- SDMA no-copy AR occupies a longer continuous interval and blocks future GEMMs
- SDMA engine/HBM traffic slows GEMM kernels directly
- stream ordering in continuous mode creates larger bubbles for SDMA than RCCL

---

## Entry 30 — Add continuous timeline samples for rate/root-cause localization
- **Date**: 2026-04-29
- **Commit**: `061dd561`
- **Purpose**: localize why continuous SDMA no-copy has faster `seq_ar` than
  RCCL but worse overlap wall. Need per-iteration/stage GEMM and AR start/end
  times in continuous mode, not finite per-iteration timeline.

### Implementation

Added CLI:
```bash
--continuous-timeline-samples N
```

When `--continuous-iters > 0`, the first `N` continuous iterations record:
- GEMM start/end per stage
- AR start/end per stage
- `AR-GEMM gap = AR_start - GEMM_end`

Uses the same unique per-iteration/stage event model from Entry 28, so the
timeline preserves correct dependencies.

### Usage

No-prep rate localization:
```bash
MORI_CONTINUOUS_PREP=0 MORI_PIPELINE_CU=160 python3 tests/python/ccl/test_allreduce.py \
  --num-stages 4 --elems 67108864 --iterations 1 --warmup 1 \
  --continuous-iters 20 --continuous-timeline-samples 6
```

With prep enabled:
```bash
MORI_PIPELINE_CU=160 python3 tests/python/ccl/test_allreduce.py \
  --num-stages 4 --elems 67108864 --iterations 1 --warmup 1 \
  --continuous-iters 20 --continuous-timeline-samples 6
```

Compare SDMA no-copy and RCCL sections:
- If SDMA no-copy GEMM durations are longer than RCCL, SDMA/HBM traffic slows GEMM.
- If AR-GEMM gaps / AR queueing are larger for SDMA, stream scheduling/order is the issue.
- If both are similar but wall differs, inspect tail/edge intervals beyond sampled rows.

---

## Entry 31 — Continuous timeline root cause: SDMA no-copy AR service time slows under GEMM, causing stream_ar backlog
- **Date**: 2026-04-29
- **Commit under test**: `061dd561` / docs follow-up `aab5370b`
- **Command**: `MORI_CONTINUOUS_PREP=0 ... --continuous-iters 20 --continuous-timeline-samples 6`
- **Scenario**: 256MB / stage × 4 stages, continuous mode, no synthetic prep.

### Summary tables from run

| mode | continuous wall | seq_ar | seq_gemm | slowdown |
|---|---:|---:|---:|---:|
| SDMA copy | 7.060 | 5.204 | 4.346 | 1.625 |
| **SDMA no-copy** | **6.474** | **4.783** | 4.146 | **1.561** |
| RCCL | **5.868** | 5.135 | 4.113 | **1.427** |

### Timeline evidence

For SDMA no-copy, GEMM remains fast (~1.0 ms/stage), but each AR stage takes
~2.0 ms in the sampled continuous overlap window and the AR-GEMM gap grows
linearly:

| iter/stage | GEMM dur | AR dur | AR-GEMM gap |
|---|---:|---:|---:|
| 0/0 | 1.021 | 1.781 | 0.011 |
| 0/1 | 1.008 | 2.088 | 0.790 |
| 0/2 | 1.050 | 2.117 | 1.820 |
| 0/3 | 1.083 | 2.070 | 2.859 |
| 1/0 | 1.039 | 2.029 | 3.895 |
| 5/3 | 0.966 | 2.138 | 24.568 |

For RCCL, after the first two transient stages, GEMM and AR both stabilize
around 1.4-1.5 ms/stage and AR-GEMM gap stays bounded (~0.5-0.6 ms):

| iter/stage | GEMM dur | AR dur | AR-GEMM gap |
|---|---:|---:|---:|
| 1/0 | 1.435 | 1.418 | 0.596 |
| 1/1 | 1.490 | 1.489 | 0.529 |
| 2/0 | 1.411 | 1.389 | 0.609 |
| 5/3 | 1.466 | 1.465 | 0.544 |

### Mechanism

The bottleneck is **not** that SDMA no-copy slows GEMM. In fact, GEMM remains
~1.0 ms/stage under SDMA no-copy, faster than under RCCL (~1.45 ms/stage).

The bottleneck is that **SDMA no-copy AR service time becomes ~2.0 ms/stage
when overlapped with GEMM**, much slower than its isolated sequential rate
(`seq_ar/4 ≈ 1.20 ms/stage`). Since GEMM produces a stage every ~1.0 ms and
stream_ar consumes one stage every ~2.0 ms, the AR queue backlog grows
linearly. RCCL instead balances both streams at ~1.4-1.5 ms/stage, so no
unbounded backlog appears.

This means the next optimization must improve SDMA AR service time under GEMM
or intentionally shift resources/priority so GEMM slows slightly and AR
accelerates, aiming for a balanced stage rate around RCCL's ~1.45 ms/stage.

### Next experiments

1. Sweep `MORI_PIPELINE_CU` upward in continuous no-copy/copy baseline (e.g.
   160, 192, 224) to see if more reduce blocks reduce AR service time enough
   to shrink backlog. This is a new mechanism vs prior CU sweep because prior
   measurements were finite/copy-exposure oriented, not corrected continuous
   timeline.
2. Test `--ar-priority -1` in corrected continuous mode. Prior priority tests
   had little effect in finite mode, but the current mechanism is specifically
   stream_ar starvation under continuous GEMM production.

---

## Entry 32 — Continuous no-copy vs RCCL 5-run stability: SDMA gap is stable ~0.75 ms, RCCL has one outlier
- **Date**: 2026-04-29
- **Commit under test**: `061dd561` / docs follow-ups
- **Command**: five independent runs of
  `MORI_CONTINUOUS_PREP=0 MORI_PIPELINE_CU=160 python3 tests/python/ccl/test_allreduce.py --num-stages 4 --elems 67108864 --iterations 1 --warmup 1 --continuous-iters 100`
- **Scenario**: corrected continuous no-prep, 256MB × 4 stages, 8 ranks.

| run | SDMA no-copy wall | SDMA seq_ar | SDMA seq_gemm | RCCL wall | RCCL seq_ar | RCCL seq_gemm | gap (SDMA-RCCL) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.558 | 4.785 | 4.180 | 5.808 | 5.140 | 4.151 | +0.750 |
| 2 | 6.564 | 4.776 | 4.201 | 5.798 | 5.124 | 4.165 | +0.766 |
| 3 | 6.559 | 4.781 | 4.200 | 5.809 | 5.121 | 4.135 | +0.750 |
| 4 | 6.580 | 4.779 | 4.164 | 5.790 | 5.140 | 4.125 | +0.790 |
| 5 | 6.562 | 4.788 | 4.151 | **13.314 outlier** | 5.132 | 4.112 | n/a |

### Interpretation

Ignoring the single RCCL outlier in run 5, the corrected continuous no-prep gap
is stable:
- SDMA no-copy wall: 6.558–6.580 ms
- RCCL wall: 5.790–5.809 ms
- gap: **+0.75 to +0.79 ms**

This confirms Entry 31's mechanism is stable, not a one-off artifact: SDMA
no-copy has faster sequential AR (~4.78 ms vs RCCL ~5.13 ms), but its continuous
overlap wall is worse because AR service time under concurrent GEMM is worse
and backlog grows.

### Next experiment

Test stream priority in this exact corrected continuous setup:
```bash
MORI_CONTINUOUS_PREP=0 MORI_PIPELINE_CU=160 python3 tests/python/ccl/test_allreduce.py \
  --num-stages 4 --elems 67108864 --iterations 1 --warmup 1 \
  --continuous-iters 100 --ar-priority -1
```

Success signature:
- SDMA no-copy wall drops toward RCCL (~5.8 ms)
- SDMA no-copy seq_ar stays similar or improves
- GEMM seq time may increase slightly; that is acceptable if total wall drops.

If priority has no effect, sweep `MORI_PIPELINE_CU=192/224` in the same
corrected continuous no-prep setup.

---

## Entry 33 — `--ar-priority -1` continuous no-prep: no measurable effect
- **Date**: 2026-04-29
- **Command**: `MORI_CONTINUOUS_PREP=0 MORI_PIPELINE_CU=160 python3 tests/python/ccl/test_allreduce.py --num-stages 4 --elems 67108864 --iterations 1 --warmup 1 --continuous-iters 100 --ar-priority -1`
- **Scenario**: corrected continuous no-prep, 256MB × 4 stages, 8 ranks.
- **All correctness checks PASSED**

| mode | wall | seq_ar | seq_gemm | slowdown | gap vs RCCL |
|---|---:|---:|---:|---:|---:|
| SDMA copy | 7.145 | 5.202 | 4.355 | 1.641 | +1.334 |
| **SDMA no-copy** | **6.567** | **4.776** | 4.167 | **1.576** | **+0.756** |
| RCCL | 5.811 | 5.139 | 4.139 | 1.404 | — |

### Comparison to Entry 32 baseline priority

Entry 32 (normal priority) no-copy wall was stable at 6.558–6.580 ms; RCCL
normal wall was 5.790–5.809 ms (excluding outlier). Entry 33 with
`--ar-priority -1` gives no-copy 6.567 and RCCL 5.811 — within the same range.

### Conclusion

Stream priority does **not** fix the continuous backlog mechanism. Likely reason:
priority affects scheduling of pending kernels but does not preempt already
running GEMM kernels enough to change the SDMA AR service time. Next experiment:
sweep `MORI_PIPELINE_CU=192/224` to see whether more reduce blocks can reduce
SDMA AR service time under GEMM.

---

## Entry 34 — Continuous no-copy with `MORI_PIPELINE_CHUNKS=4`: small improvement, still +0.58 ms vs RCCL
- **Date**: 2026-04-29
- **Command**: `MORI_CONTINUOUS_PREP=0 MORI_PIPELINE_CU=224 MORI_PIPELINE_CHUNKS=4 python3 tests/python/ccl/test_allreduce.py --num-stages 4 --elems 67108864 --iterations 1 --warmup 1 --continuous-iters 100`
- **Scenario**: corrected continuous no-prep, 256MB × 4 stages, 8 ranks.
- **All correctness checks PASSED**

| mode | continuous wall | seq_ar | seq_gemm | slowdown | gap vs RCCL |
|---|---:|---:|---:|---:|---:|
| SDMA copy | 7.389 | 5.227 | 4.334 | 1.705 | +1.580 |
| **SDMA no-copy** | **6.390** | **4.811** | 4.168 | **1.533** | **+0.581** |
| RCCL | **5.809** | 5.139 | 4.128 | 1.407 | — |

### Interpretation

`MORI_PIPELINE_CHUNKS=4` at high CU gives a small improvement over the stable
2-chunk no-copy baseline:
- Entry 32 normal no-copy: 6.558–6.580 ms
- Entry 34 chunks=4 no-copy: **6.390 ms**
- improvement: ~0.17 ms

The remaining gap is still **+0.581 ms**. Increasing chunk density from 2 to 4
appears to improve AR service cadence under GEMM, but not enough. Copy mode
worsens because the external copy remains and more chunk/barrier overhead is
added.

### Next experiment

Sweep chunk count around this setting:
```bash
for ch in 6 8 12; do
  MORI_CONTINUOUS_PREP=0 MORI_PIPELINE_CU=224 MORI_PIPELINE_CHUNKS=$ch \
    python3 tests/python/ccl/test_allreduce.py --num-stages 4 \
    --elems 67108864 --iterations 1 --warmup 1 --continuous-iters 100
done
```

If chunks=6/8 improves no-copy further, continue until barrier overhead turns
it around. If all regress, best known setting is `PIPELINE_CU=224,
PIPELINE_CHUNKS=4`.

---

## Entry 35 — Continuous chunk sweep 6/8/12: chunks=4 remains best; chunk tuning closed
- **Date**: 2026-04-29
- **Command**:
  `for ch in 6 8 12; MORI_CONTINUOUS_PREP=0 MORI_PIPELINE_CU=224 MORI_PIPELINE_CHUNKS=$ch ... --continuous-iters 100`
- **Scenario**: corrected continuous no-prep, 256MB × 4 stages, 8 ranks.

| chunks | SDMA no-copy wall | SDMA seq_ar | SDMA seq_gemm | RCCL wall | RCCL seq_ar | RCCL seq_gemm | gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 (Entry 34) | **6.390** | 4.811 | 4.168 | 5.809 | 5.139 | 4.128 | +0.581 |
| 6 | 8.980 | 5.496 | 4.593 | **13.278 outlier** | 5.763 | 4.565 | n/a |
| 8 | 6.649 | 4.813 | 4.189 | 5.799 | 5.128 | 4.143 | +0.850 |
| 12 | 6.653 | 4.818 | 4.180 | 5.790 | 5.126 | 4.133 | +0.863 |

### Interpretation

`chunks=8/12` both regress vs `chunks=4`:
- no-copy wall 6.649/6.653 vs 6.390
- seq_ar stays similar (~4.81), but overlap wall worsens

`chunks=6` run is not usable for gap because RCCL wall 13.278 is an outlier,
but SDMA no-copy also regressed badly to 8.980, so it is not a candidate.

### Conclusion

Chunk tuning is closed. Best known continuous no-copy setting:
`MORI_PIPELINE_CU=224`, `MORI_PIPELINE_CHUNKS=4`, wall **6.390 ms**, still
**+0.581 ms** slower than RCCL.

The remaining gap is not solved by:
- stream priority (`--ar-priority -1`, Entry 33)
- more CU alone (Entries 32/34 context)
- chunk count beyond 4 (Entry 35)

Next viable direction: change the continuous scheduling model so stream_ar
does not accumulate backlog, e.g. intentionally throttle GEMM / insert
cross-stream pacing so GEMM production rate matches SDMA AR service rate, or
move from two-shot to a ring-like algorithm with more balanced per-stage AR
service time.

---

## Entry 36 — Implement `MORI_BATCH_REDUCE_BARRIER=1` to test removing per-chunk reduce_complete barriers
- **Date**: 2026-04-29
- **Commit**: `18b26810`
- **Motivation**: Continuous phase timing for `PIPELINE_CU=224, CHUNKS=4`
  identified substantial per-chunk control overhead (cross-PE reduce_complete
  barrier + AG submit/signaling + final AG wait). Chunk count >4 regressed
  (Entry 35), so the next test is to reduce control overhead rather than
  increasing chunk density.

### Implementation

Add env:
```bash
MORI_BATCH_REDUCE_BARRIER=1
```

Only affects baseline `PipelinedAllReduceSdmaKernel<T,0,true*>` multi-chunk
path (Plan A kernel unchanged). Default path is unchanged.

With the env enabled:
1. compute blocks reduce all chunks and increment `chunks_complete` per chunk
   as before
2. block 0 waits once for `numChunks * compBlocks`
3. block 0 does one cross-PE reduce_complete barrier by adding `numChunks` to
   the flag and waiting peers to reach `reduceCompleteBase + numChunks`
4. block 0 submits all chunks' SDMA AG packets back-to-back
5. block 0 waits all AG signals as before

This removes per-chunk reduce_complete barriers, at the cost of losing
`reduce(c+1)` vs `AG(c)` overlap. Test decides which effect dominates in
corrected continuous steady-state.

### Test command

```bash
MORI_CONTINUOUS_PREP=0 MORI_PIPELINE_CU=224 MORI_PIPELINE_CHUNKS=4 \
MORI_BATCH_REDUCE_BARRIER=1 python3 tests/python/ccl/test_allreduce.py \
  --num-stages 4 --elems 67108864 --iterations 1 --warmup 1 \
  --continuous-iters 100
```

Compare SDMA no-copy wall against Entry 34 best (6.390 ms) and RCCL (~5.809 ms).

### Result / revert

User run:
```text
SDMA no-copy 49.456 ms (ar 8.605, gemm 7.295)
RCCL 59.237 ms outlier (ar 51.927, gemm 12.892)
```

Even ignoring the RCCL outlier, SDMA no-copy itself is catastrophically worse
than Entry 34 best (49.456 vs 6.390 ms). The experiment destroyed SDMA
pipeline behavior by losing reduce(c+1)/AG(c) overlap and batching AG after all
reduces. This path is **reverted in the next commit**; keep only the phase slot
layout fix.

---

## Entry 37 — `MORI_COPY_KERNEL=1` vectorized copy kernel: copy wall improves but total wall only -0.057 ms
- **Date**: 2026-04-29
- **Commit under test**: `62ec03f6`
- **Command**: baseline vs `MORI_COPY_KERNEL=1`, corrected continuous no-prep,
  `MORI_PIPELINE_CU=224`, `MORI_PIPELINE_CHUNKS=4`, `--continuous-iters 100`,
  `--ar-phase-timing`.
- **All correctness checks PASSED**

| variant | gpu-side copy wall | SDMA copy wall | SDMA no-copy wall | RCCL wall | copy gap vs RCCL |
|---|---:|---:|---:|---:|---:|
| baseline hipMemcpyAsync | 0.388 ms | 6.988 | 6.388 | 5.793 | +1.195 |
| `MORI_COPY_KERNEL=1` | **0.285 ms** | **6.931** | 6.387 | 5.797 | +1.134 |

### Interpretation

The custom vectorized D2D copy kernel is correct and reduces measured copy wall
by **0.103 ms** (0.388 → 0.285 ms), but total copy-mode wall improves only
**0.057 ms** (6.988 → 6.931 ms). This is useful but far from enough; the
remaining copy-mode gap vs RCCL is still **+1.13 ms**.

The no-copy wall is unchanged (6.388/6.387), confirming the copy kernel only
affects `copy_output_to_user=True`.

### Next experiment

Sweep copy-kernel geometry:
- `MORI_COPY_KERNEL_BLOCKS={512,1024,2048,4096}`
- `MORI_COPY_KERNEL_THREADS={128,256,512}`

Success target for this path would need copy wall < ~0.15 ms and a much larger
total wall reduction. If geometry cannot beat 0.285 ms materially, the copy
kernel path is insufficient.

---

## Entry 38 — `MORI_COPY_KERNEL` geometry sweep: best 4096×512 gives 0.180 ms copy wall
- **Date**: 2026-04-29
- **Commit under test**: `62ec03f6`
- **Command**: sweep `MORI_COPY_KERNEL_BLOCKS={512,1024,2048,4096}` and
  `MORI_COPY_KERNEL_THREADS={128,256,512}` with corrected continuous no-prep,
  `MORI_PIPELINE_CU=224`, `MORI_PIPELINE_CHUNKS=4`, `--continuous-iters 100`,
  `--ar-phase-timing`.
- **All variants correctness PASSED**

| blocks | threads | gpu-side copy wall |
|---:|---:|---:|
| 512 | 128 | 0.732 |
| 512 | 256 | 0.340 |
| 512 | 512 | 0.300 |
| 1024 | 128 | 0.343 |
| 1024 | 256 | 0.338 |
| 1024 | 512 | 0.228 |
| 2048 | 128 | 0.321 |
| 2048 | 256 | 0.234 |
| 2048 | 512 | 0.196 |
| 4096 | 128 | 0.272 |
| 4096 | 256 | 0.200 |
| **4096** | **512** | **0.180** |

### Interpretation

Geometry matters. Best observed copy wall is **0.180 ms** at 4096 blocks × 512
threads, vs:
- baseline `hipMemcpyAsync` from Entry 37: 0.388 ms
- default copy kernel 1024×256 from Entry 37: 0.285 ms

This is a **0.208 ms/AR** improvement in the copy timing measurement relative
to baseline hipMemcpyAsync. Need a full Table 1-4 run with the best geometry to
see how much of this copy improvement translates to continuous end-to-end wall.

### Next command

```bash
MORI_CONTINUOUS_PREP=0 MORI_PIPELINE_CU=224 MORI_PIPELINE_CHUNKS=4 \
MORI_COPY_KERNEL=1 MORI_COPY_KERNEL_BLOCKS=4096 MORI_COPY_KERNEL_THREADS=512 \
python3 tests/python/ccl/test_allreduce.py --num-stages 4 --elems 67108864 \
  --iterations 1 --warmup 1 --continuous-iters 100 --ar-phase-timing
```

Compare SDMA copy wall against Entry 34/37 baseline. If SDMA copy wall drops by
~0.2–0.4 ms, keep this geometry as the default for `MORI_COPY_KERNEL=1`.

---

## Entry 39 — Best copy kernel geometry full run: copy wall 0.179 ms, total wall still +1.41 ms vs RCCL
- **Date**: 2026-04-29
- **Commit under test**: `62ec03f6` + docs follow-ups
- **Command**: corrected continuous no-prep, `MORI_PIPELINE_CU=224`,
  `MORI_PIPELINE_CHUNKS=4`, `MORI_COPY_KERNEL=1`,
  `MORI_COPY_KERNEL_BLOCKS=4096`, `MORI_COPY_KERNEL_THREADS=512`,
  `--continuous-iters 100 --ar-phase-timing`.
- **Correctness PASSED**

| mode | wall | seq_ar | seq_gemm | slowdown | gap vs RCCL |
|---|---:|---:|---:|---:|---:|
| **SDMA copy + best copy kernel** | **7.216** | 5.204 | 4.373 | 1.650 | **+1.408** |
| SDMA no-copy | 6.376 | 4.807 | 4.237 | 1.505 | +0.568 |
| RCCL | 5.808 | 5.130 | 4.199 | 1.383 | — |

Copy timing:
- best copy kernel GPU wall: **0.179 ms**
- baseline hipMemcpyAsync from Entry 37: 0.388 ms
- improvement in copy timing: **0.209 ms**

### Interpretation

The copy kernel now makes the isolated copy operation significantly faster,
but end-to-end copy-mode wall is still far from RCCL:
- SDMA copy wall = 7.216 ms
- RCCL wall = 5.808 ms
- gap = **+1.408 ms**

The copy-mode wall is also **0.840 ms slower than no-copy** (7.216 vs 6.376),
far larger than the measured copy kernel wall 0.179 ms. This means the copy
kernel's CU work / stream occupancy hurts overlap beyond its own measured
duration. It improves copy timing but does not solve the dominant overlap
problem.

### Conclusion

The vectorized copy-kernel path is insufficient as a final solution. Even with
the best observed geometry (4096×512), it does not bring copy mode close to
RCCL. The remaining work is not just making the local copy faster; it is
avoiding additional CU occupancy / preserving overlap quality while still
writing user_output.

---

## Entry 48 — Final current best COPY vs RCCL check: SDMA copy still +1.02 ms worse
- **Date**: 2026-04-30
- **Commit under test**: `62ec03f6` copy-kernel code + current branch docs/fixes
- **Command**: corrected continuous no-prep, `MORI_PIPELINE_CU=224`,
  `MORI_PIPELINE_CHUNKS=4`, `MORI_COPY_KERNEL=1`,
  `MORI_COPY_KERNEL_BLOCKS=4096`, `MORI_COPY_KERNEL_THREADS=512`,
  `--continuous-iters 100 --ar-phase-timing`
- **All correctness checks PASSED**

| mode | wall | seq_ar | seq_gemm | slowdown | gap vs RCCL |
|---|---:|---:|---:|---:|---:|
| **SDMA copy (best current)** | **6.817** | 5.202 | 4.360 | 1.563 | **+1.020** |
| SDMA no-copy | 6.410 | 4.810 | 4.156 | 1.543 | +0.613 |
| RCCL | **5.797** | 5.127 | 4.151 | 1.396 | — |

Copy timing:
```text
gpu-side copy kernel wall: 0.180 ms
```

### Interpretation

This is the current best known copy-mode stack:
- CU=224
- chunks=4
- best vectorized copy kernel geometry 4096×512

It still fails the final success criterion (**COPY VS RCCL**):
```text
SDMA copy 6.817 ms vs RCCL 5.797 ms => +1.020 ms worse
```

Even no-copy remains +0.613 ms worse, so the final gap is not just output copy.
Copy is now only ~0.407 ms above no-copy in wall (6.817 - 6.410), but no-copy
itself trails RCCL because SDMA AR overlap quality is worse.

### Conclusion

All incremental two-shot / existing-fullmesh knobs are insufficient for the
drop-in copy target:
- no-copy best remains slower than RCCL in corrected continuous mode
- copy kernel improves local copy but cannot close copy-mode gap
- fullmesh alias, multi-q AG, separate AG buffer, priority, chunk sweeps, and
  barrier variants do not reach target

Next required work: implement a new fullmesh channelized algorithm that produces
`user_output` without a separate large copy while keeping AR/GEMM service rates
balanced. Current two-shot structure is exhausted for the final COPY VS RCCL
goal.

---

## Entry 49 — Implement integer accumulator fast path for uint32/int32 pipeline reduce
- **Date**: 2026-04-30
- **Commit**: _this commit_
- **Motivation**: Continuous phase timing shows the pipeline reduce phase is a
  major part of AR service time under GEMM. For the current benchmark dtype
  (`uint32`), the baseline reduce path still used `packed_t<T>::A = float`,
  converting integer inputs to float, accumulating in float, and converting
  back. This is unnecessary for uint32/int32 and adds CU work in the exact
  phase that is limiting stream_ar service rate.

### Implementation

In `PipelinedAllReduceSdmaKernel` compute-block reduce:
- if `P::type` is `uint32_t` or `int32_t`, accumulate directly in packed integer
  registers (`P`) with `packed_assign_add`
- otherwise keep the existing float-accumulator path

Default semantics remain unchanged for other dtypes.

### Test command

```bash
MORI_CONTINUOUS_PREP=0 MORI_PIPELINE_CU=224 MORI_PIPELINE_CHUNKS=4 \
python3 tests/python/ccl/test_allreduce.py --num-stages 4 --elems 67108864 \
  --iterations 1 --warmup 1 --continuous-iters 100 \
  --continuous-phase-iter 5 --continuous-phase-stage 0
```

Success signature:
- correctness passes
- `SDMA no-copy` wall improves below Entry 34 best 6.390 ms
- phase `R-block reduce` times drop from ~0.39/0.40 ms for first chunks

---

## Entry 50 — Integer reduce fast path: phase reduce improves, end-to-end wall unchanged
- **Date**: 2026-04-30
- **Commit under test**: `9ae490e4`
- **Command**: `MORI_CONTINUOUS_PREP=0 MORI_PIPELINE_CU=224 MORI_PIPELINE_CHUNKS=4 ... --continuous-iters 100 --continuous-phase-iter 5 --continuous-phase-stage 0`
- **All correctness checks PASSED**

| mode | wall | seq_ar | seq_gemm | slowdown | gap vs RCCL |
|---|---:|---:|---:|---:|---:|
| SDMA copy | 7.034 | 5.227 | 4.350 | 1.617 | +1.224 |
| **SDMA no-copy** | **6.388** | **4.816** | 4.242 | **1.506** | **+0.578** |
| RCCL | **5.810** | 5.133 | 4.141 | 1.403 | — |

### Phase evidence (no-copy, iter=5 stage=0)

```text
event = 1.917 ms
c0: compute-wait 0.340, barrier 0.264, submit/signal 0.113
c1: compute-wait 0.001, barrier 0.181, submit/signal 0.159
c2: compute-wait 0.015, barrier 0.031, submit/signal 0.009
c3: compute-wait 0.015, barrier 0.168, submit/signal 0.153
ag_wait 0.347
R-block reduce: c0 0.152, c1 0.142, c2 0.159, c3 0.141
```

### Interpretation

The integer accumulator fast path works at the phase level:
- prior no-copy reduce chunks were around ~0.38–0.40 ms for early chunks
- now all four reduce chunks are around **0.14–0.16 ms**

But end-to-end no-copy wall is essentially unchanged:
- Entry 34 best no-copy: 6.390 ms
- Entry 50 no-copy: **6.388 ms**

The bottleneck shifted/was already dominated by barrier + submit/control +
AG wait + stream backlog, not pure integer reduce ALU. The change is still
worth keeping (less CU work, cleaner uint32 path), but it does not close the
COPY VS RCCL target.

### Next implication

Further optimizing reduce arithmetic is unlikely to recover the remaining
~0.58 ms no-copy gap. The remaining gap is control/scheduling/cadence. A new
fullmesh channelized algorithm must reduce barrier/submit cadence overhead and
avoid long AR service bursts, while still producing `user_output` for copy mode.

---

## Entry 51 — Implement correctness-first `MORI_FULLMESH_CHAN=1` fullmesh channelized MVP
- **Date**: 2026-04-30
- **Commit**: _this commit_
- **Goal**: start a genuinely new fullmesh channelized path for the final
  drop-in copy target. This is not the old `MORI_FULLMESH_PIPE=1` alias; it is
  a new kernel that writes `user_output` inside the allreduce path.

### Implementation

Add env:
```bash
MORI_FULLMESH_CHAN=1
```

Active only for `copy_output_to_user=True`, `scatter_mode=0`, multi-chunk calls.
Default path unchanged.

New kernel `FullMeshChannelizedAllReduceKernel` loops over chunks:
1. block0 fullmesh SDMA scatters chunk `c` into `output_transit_buffer_`
2. compute blocks reduce chunk `c` into `output_transit_buffer_[myPe]`
3. block0 fullmesh SDMA AGs chunk `c` into separate internal symmetric AG buffer
4. compute blocks locally copy chunk `c` from AG buffer to `user_output`

Remote writes never target `user_output` and never overwrite the scatter/reduce
input transit. The path skips the external `copy_output_to_user()` because the
kernel writes user output directly.

### Test command

```bash
cd /home/fizhang/test/mori && git pull origin sdma-test
BUILD_EXAMPLES=ON BUILD_TESTS=ON pip3 install .
MORI_CONTINUOUS_PREP=0 MORI_FULLMESH_CHAN=1 MORI_PIPELINE_CU=224 \
MORI_PIPELINE_CHUNKS=4 python3 tests/python/ccl/test_allreduce.py \
  --num-stages 4 --elems 67108864 --iterations 1 --warmup 1 \
  --continuous-iters 100
```

Success criteria:
- `SDMA copy` correctness passes
- `SDMA copy` wall improves over current best copy (Entry 48: 6.817 ms)
- eventually beat RCCL (~5.8 ms)

---

## Entry 40 — Implement experimental multi-q SDMA AG (`MORI_MULTI_Q_AG=1`)
- **Date**: 2026-04-29
- **Commit**: _this commit_
- **Motivation**: Corrected continuous phase timing with
  `PIPELINE_CU=224, CHUNKS=4` shows SDMA no-copy steady-state AR still spends
  ~1 ms in AG wait. The current baseline submits every chunk's AG on qId=1,
  so the SDMA queue serializes chunks. Earlier Entry 12 already proved a single
  SDMA queue is FIFO and multi-q AG has real gain potential.

### Implementation

Add env:
```bash
MORI_MULTI_Q_AG=1
```

When enabled in `PipelinedAllReduceSdmaKernel`:
- scatter stays on qId=0
- AG chunk `c` uses qId `1 + c % min(numQ-1, 15)`
- wait for chunk `c` reads `signalPtrs[sender*numQ + qId]`
- host maintains per-q AG generation counters in
  `pipeline_ag_gen_by_q_[16]` and copies them to device before launch, avoiding
  unsafe per-launch signal resets

Default path remains unchanged.

### Test command

```bash
MORI_CONTINUOUS_PREP=0 MORI_PIPELINE_CU=224 MORI_PIPELINE_CHUNKS=4 \
MORI_MULTI_Q_AG=1 python3 tests/python/ccl/test_allreduce.py \
  --num-stages 4 --elems 67108864 --iterations 1 --warmup 1 \
  --continuous-iters 100 --continuous-phase-iter 5 --continuous-phase-stage 0
```

Success signature:
- `SDMA no-copy` wall drops below Entry 34 best 6.390 ms
- phase `ag_wait` drops below ~1.0 ms
- correctness passes

---

## Entry 41 — `MORI_MULTI_Q_AG=1` result: correctness passes but wall regresses vs chunks=4 best
- **Date**: 2026-04-30
- **Commit under test**: `3e379594`
- **Command**: `MORI_CONTINUOUS_PREP=0 MORI_PIPELINE_CU=224 MORI_PIPELINE_CHUNKS=4 MORI_MULTI_Q_AG=1 ... --continuous-iters 100 --continuous-phase-iter 5 --continuous-phase-stage 0`
- **All correctness checks PASSED**

| mode | wall | seq_ar | seq_gemm | slowdown | gap vs RCCL |
|---|---:|---:|---:|---:|---:|
| SDMA copy + multi-q AG | 7.070 | 5.302 | 4.344 | 1.628 | +1.276 |
| **SDMA no-copy + multi-q AG** | **6.469** | **4.894** | 4.166 | **1.553** | **+0.675** |
| RCCL | 5.794 | 5.148 | 4.160 | 1.393 | — |

Compare best prior no-copy setting:
- Entry 34 `CU=224, chunks=4`, no multi-q: **6.390 ms**
- Entry 41 multi-q: **6.469 ms**
- regression: **+0.079 ms**

### Phase evidence (no-copy, continuous iter=5 stage=0)

```text
event = 2.122 ms
c0: compute-wait 0.545, barrier 0.040, submit/signal 0.218
c1: compute-wait 0.056, barrier 0.110, submit/signal 0.167
c2: compute-wait 0.008, barrier 0.182, submit/signal 0.167
c3: compute-wait 0.009, barrier 0.023, submit/signal 0.104
ag_wait 0.364
R-block reduce: c0 0.432, c1 0.333, c2 0.094, c3 0.158
```

`ag_wait` is not the dominant bottleneck after chunks=4; reducing queue
serialization does not improve wall. The remaining per-stage time is spread
across compute-wait/reduce, cross-PE barrier, and SDMA submit/control. Multi-q
adds complexity / generation tracking and slightly worsens wall.

### Conclusion

Close `MORI_MULTI_Q_AG` as ineffective for the current best setting. Keep it
off by default; do not pursue unless a new phase profile shows AG wait again
dominates.

---

## Entry 42 — `MORI_GROUPED_REDUCE_BARRIER=2` FAILED correctness; reverted immediately
- **Date**: 2026-04-30
- **Implementation commit**: `4f74e663`
- **Revert commit**: `2dc20476`
- **Command**: `MORI_CONTINUOUS_PREP=0 MORI_PIPELINE_CU=224 MORI_PIPELINE_CHUNKS=4 MORI_GROUPED_REDUCE_BARRIER=2 ... --continuous-iters 100 --continuous-phase-iter 5 --continuous-phase-stage 0`

### Result

Correctness failed before any perf conclusion:
```text
PE 0 [outplace/uint32]: FAILED! Expected 36000, got 34816 mismatches.
First mismatch at idx 194560: 253000
PE 0 [inplace2-transit/uint32]: FAILED! Expected 36000, got 53504 mismatches.
First mismatch at idx 2048: 93000
PE 0 [inplace2/uint32]: FAILED! Expected 36000, got 53504 mismatches.
```

Perf numbers from that invalid run were also unusable:
```text
SDMA no-copy 11.268 ms
RCCL 13.815 ms
```

### Mechanism

Grouped barrier was intended as a middle ground:
```text
reduce c0,c1 -> one barrier -> AG c0,c1
reduce c2,c3 -> one barrier -> AG c2,c3
```

But this violates the original per-chunk safety invariant. Peers may begin
reducing chunk `c1` while another PE has already AG-written chunk `c0` into
the peer's transit slot in the same group, and the shared transit / ordering
assumptions are no longer protected per chunk. The observed stale/incorrect
transit values match this class of race.

### Conclusion

`MORI_GROUPED_REDUCE_BARRIER` is closed. It was reverted immediately (R5).
Do not retry grouped barriers without a new correctness proof and explicit
per-chunk ownership separation.

---

## Entry 43 — Expose existing single-kernel fullmesh micro-pipeline as `MORI_FULLMESH_PIPE=1`
- **Date**: 2026-04-30
- **Commit**: _this commit_
- **Why**: User chose fullmesh over ring. Current code already has a closer
  fullmesh micro-pipeline path: `PipelinedAllReduceSdmaKernel<T,0,true,false>`
  (`external_scatter=false`) where one kernel does fullmesh scatter submission,
  per-chunk reduce, per-chunk cross-PE barrier, and per-chunk SDMA AG. This is
  the first fullmesh experiment before writing a completely new kernel.

### Implementation

`AllreduceSdma::operator()` now treats:
```bash
MORI_FULLMESH_PIPE=1
```
as an alias for the old experimental `MORI_PIPELINE_FUSED=1`, forcing:
```text
pipelined(..., external_scatter=false)
```

Default remains unchanged.

### Test command

```bash
cd /home/fizhang/test/mori && git pull origin sdma-test
BUILD_EXAMPLES=ON BUILD_TESTS=ON pip3 install .
MORI_CONTINUOUS_PREP=0 MORI_FULLMESH_PIPE=1 MORI_PIPELINE_CU=224 \
MORI_PIPELINE_CHUNKS=4 python3 tests/python/ccl/test_allreduce.py \
  --num-stages 4 --elems 67108864 --iterations 1 --warmup 1 \
  --continuous-iters 100 --continuous-phase-iter 5 --continuous-phase-stage 0
```

Compare against best known two-kernel fullmesh:
`MORI_PIPELINE_CU=224, MORI_PIPELINE_CHUNKS=4`, no-copy 6.390 ms.

---

## Entry 44 — `MORI_FULLMESH_PIPE=1` result: correctness passes, performance matches best two-kernel but does not improve
- **Date**: 2026-04-30
- **Commit under test**: `3a6bc7bc`
- **Command**: `MORI_CONTINUOUS_PREP=0 MORI_FULLMESH_PIPE=1 MORI_PIPELINE_CU=224 MORI_PIPELINE_CHUNKS=4 ... --continuous-iters 100 --continuous-phase-iter 5 --continuous-phase-stage 0`
- **All correctness checks PASSED**

| mode | wall | seq_ar | seq_gemm | slowdown | gap vs RCCL |
|---|---:|---:|---:|---:|---:|
| SDMA copy + fullmesh pipe | 7.027 | 5.245 | 4.339 | 1.620 | +1.218 |
| **SDMA no-copy + fullmesh pipe** | **6.406** | **4.812** | 4.166 | **1.538** | **+0.597** |
| RCCL | **5.809** | 5.373 | 4.202 | 1.383 | — |

Compare best prior two-kernel setting:
- Entry 34: `PIPELINE_CU=224, CHUNKS=4`, no-copy **6.390 ms**
- Entry 44: `MORI_FULLMESH_PIPE=1`, no-copy **6.406 ms**
- essentially equal (within noise), no breakthrough.

### Phase evidence (no-copy, iter=5 stage=0)

```text
event = 2.019 ms
c0: compute-wait 0.563, barrier 0.052, submit/signal 0.149
c1: compute-wait 0.173, barrier 0.052, submit/signal 0.148
c2: compute-wait 0.049, barrier 0.031, submit/signal 0.142
c3: compute-wait 0.014, barrier 0.046, submit/signal 0.169
ag_wait 0.275
R-block reduce: c0 0.387, c1 0.397, c2 0.402, c3 0.072
```

### Interpretation

The existing single-kernel fullmesh micro-pipeline is correct and no worse than
the current best two-kernel setup, but it does not fix the fundamental cadence
gap. The steady-state AR event is still ~2.0 ms. The largest component is
compute/reduce/compute-wait, not AG wait. This confirms that merely moving K1
scatter into the same kernel is insufficient; a new algorithm must change the
reduce/communication cadence more substantially.

---

## Entry 45 — Implement `MORI_SEPARATE_AG_BUFFER=1` to remove per-chunk cross-PE reduce barrier safely
- **Date**: 2026-04-30
- **Commit**: _this commit_
- **Motivation**: Grouped/batch reduce barriers failed because AG writes reused
  the same transit slots that peer reduce may still read. A safe way to remove
  the cross-PE reduce_complete wait is to make AG write into a separate internal
  symmetric buffer, not the scatter/reduce transit buffer.

### Implementation

Add env:
```bash
MORI_SEPARATE_AG_BUFFER=1
```

When enabled for `scatter_mode=0` baseline:
- allocate/register a second internal symmetric buffer using `input_transit_buffer_`
  as AG/output staging
- reduce still reads/writes the original `output_transit_buffer_`
- SDMA AG writes to the separate AG buffer
- block0 still increments `reduce_complete` generation counters for monotonic
  protocol compatibility, but skips waiting on peers' reduce_complete flags
  because AG no longer overwrites the peer's reduce input
- `copy_output_to_user()` copies from the separate AG buffer
- `get_output_transit_buffer()` returns the separate AG buffer so no-copy tests
  remain correct

Default path remains unchanged.

### Test command

```bash
cd /home/fizhang/test/mori && git pull origin sdma-test
BUILD_EXAMPLES=ON BUILD_TESTS=ON pip3 install .
MORI_CONTINUOUS_PREP=0 MORI_PIPELINE_CU=224 MORI_PIPELINE_CHUNKS=4 \
MORI_SEPARATE_AG_BUFFER=1 python3 tests/python/ccl/test_allreduce.py \
  --num-stages 4 --elems 67108864 --iterations 1 --warmup 1 \
  --continuous-iters 100 --continuous-phase-iter 5 --continuous-phase-stage 0
```

Success signature:
- correctness passes
- no-copy wall below Entry 34 best 6.390 ms
- phase barrier time drops substantially without grouped-barrier correctness
  failure

---

## Entry 46 — `MORI_SEPARATE_AG_BUFFER=1` result: correctness passes but no performance gain
- **Date**: 2026-04-30
- **Commit under test**: `18b26810` / docs follow-up `64b811b7`
- **Command**: `MORI_CONTINUOUS_PREP=0 MORI_PIPELINE_CU=224 MORI_PIPELINE_CHUNKS=4 MORI_SEPARATE_AG_BUFFER=1 ... --continuous-iters 100 --continuous-phase-iter 5 --continuous-phase-stage 0`
- **All correctness checks PASSED**

| mode | wall | seq_ar | seq_gemm | slowdown | gap vs RCCL |
|---|---:|---:|---:|---:|---:|
| SDMA copy + separate AG | 7.020 | 5.258 | 4.340 | 1.617 | +1.226 |
| **SDMA no-copy + separate AG** | **6.407** | **4.816** | 4.151 | **1.543** | **+0.613** |
| RCCL | **5.794** | 5.134 | 4.099 | 1.414 | — |

Compare best known:
- Entry 34 no-copy (`CU=224, chunks=4`): **6.390 ms**
- Entry 46 separate AG no-copy: **6.407 ms**
- essentially equal / slight regression.

### Phase evidence (no-copy, iter=5 stage=0)

```text
event = 2.022 ms
c0: compute-wait 0.574, barrier 0.110, submit/signal 0.083
c1: compute-wait 0.053, barrier 0.247, submit/signal 0.159
c2: compute-wait 0.001, barrier 0.080, submit/signal 0.159
c3: compute-wait 0.014, barrier 0.046, submit/signal 0.082
ag_wait 0.342
R-block reduce: c0 0.449, c1 0.363, c2 0.134, c3 0.160
```

### Interpretation

Separate AG buffer removes the correctness reason for waiting on peers before
AG, but it does not improve wall. The bottleneck remains a mix of local reduce
service time, barrier/control, and AG wait; simply moving AG writes to another
internal buffer is insufficient.

### Conclusion

Close `MORI_SEPARATE_AG_BUFFER` as ineffective for performance. It may remain
useful as a building block for future algorithms, but it is not a standalone
solution.

---

## Entry 47 — Max-CU sweep (224/240/255): tiny gain only; existing two-shot/fullmesh kernel cannot close gap
- **Date**: 2026-04-30
- **Command**: corrected continuous no-prep, `MORI_PIPELINE_CHUNKS=4`,
  `MORI_PIPELINE_CU={224,240,255}`, `--continuous-iters 100`
- **All correctness checks PASSED**

| CU | SDMA no-copy wall | SDMA seq_ar | SDMA seq_gemm | RCCL wall | RCCL seq_ar | RCCL seq_gemm | gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 224 | 6.415 | 4.820 | 4.174 | 5.793 | 5.127 | 4.138 | +0.622 |
| 240 | 6.386 | 4.816 | 4.193 | 5.797 | 5.132 | 4.198 | +0.589 |
| 255 | **6.369** | 4.836 | 4.172 | 5.809 | 5.132 | 4.119 | **+0.560** |

### Interpretation

Raising CU from 224 to 255 gives only **0.046 ms** wall improvement
(6.415 → 6.369) while the remaining gap is still **+0.56 ms**. The slope is too
small to close the gap by resource scaling. Existing two-shot/fullmesh kernels
are now at their practical CU limit and still cannot match RCCL steady-state
overlap.

### Conclusion

Parameter tuning of the existing kernels is exhausted:
- priority: no effect (Entry 33)
- chunks: best at 4, larger regresses (Entries 34/35)
- CU: max 255 still +0.56 ms (Entry 47)
- multi-q AG: regresses (Entry 41)
- barrier batching/grouping: failed/regressed correctness/perf (Entries 36/42)
- copy kernel: improves copy wall but not enough (Entries 37-39)
- existing fullmesh fused path: no breakthrough (Entry 44)

Next required step is a genuinely new fullmesh channelized algorithm (not just
parameter tuning of the two-shot pipeline).

---

## Entry 19 — Plan A (PipelinedXGMIPullKernel) baseline reference + kernel swap
- **Date**: 2026-04-24
- **Baseline reference SHA**: `5f0072e7` (code HEAD at measurement time) /
  `d975cc60` (docs + rule HEAD)
- **Plan A commit SHA**: `e7050bc6` (replaces Plan B's
  `PipelinedCuReduceAgCopyKernel` with new `PipelinedXGMIPullKernel`;
  host layer `use_plan_a` branch switched to Plan A kernel + ag_sync reset)
- **Context**: Entry 18 established Plan B (CU push AG + CU copy) failed
  by +1.16 ms wall. User-specified fix: switch to **strict Plan A per
  transcript `c0921e01` L1055** — K1 SDMA scatter (免 CU), K2 CU reduce +
  cross-PE barrier + **CU XGMI pull** peer transits + direct write
  user_output (no external hipMemcpyAsync, no push AG, no β partition).
  Uses **方式 2** (单写 transit + unified AG loop): reduce writes only
  local `transit[myPe slot]`; AG loop is unified over all slots including
  self (peerPtrs[myPe] == localPtr → local-to-local copy for self).

### Baseline reference data (SDMA copy path, HEAD `5f0072e7`, iter=100)
- Command: `MORI_PIPELINE_CU=160 python3 tests/python/ccl/test_allreduce.py
  --num-stages 4 --elems 67108864 --iterations 100 --warmup 20`
- Scenario: 256 MB / stage × 4 stages, 8-rank MI355X, multi-stage overlap

| variant | overlap (ms) | seq_ar | seq_gemm | vs RCCL |
|---|---|---|---|---|
| SDMA copy | **7.802** | 5.190 | 3.672 | **+0.237 ms (+3.1%)** |
| SDMA no-copy | 7.402 | 4.774 | 3.675 | -2.2% |
| RCCL | **7.565** | 5.133 | 3.676 | — |

**Note**: Entry 18 RCCL = 7.422 ms (yesterday); today RCCL = 7.565 ms.
Gap against SDMA copy drops from +0.376 → +0.237 solely because RCCL
ran slower. RCCL has run-to-run noise (Entry 15 also saw jumps). Plan A
target: **wall ≤ min(7.422, 7.565) = 7.422 ms** (保险) to claim success
regardless of today's RCCL variance.

### Phase timing reference (SDMA copy AR[0], iter=3 warmup=2)
- Command: `MORI_PIPELINE_CU=160 MORI_PHASE_TARGET_STAGE=0 ...
  --iterations 3 --warmup 2 --ar-phase-timing`

| Phase (AR[0], mean of 2 iters) | ms | 性质 |
|---|---|---|
| entry → scatter_done (K2 wait scatter) | 0.20 | EXTERNAL_SCATTER wait |
| c0: scatter → compute-wait | 0.45 | block 0 observes CB reduce done c0 |
| c0: compute-wait → barrier | 0.14 | cross-PE barrier c0 |
| c0: barrier → AG-submit | 0.11 | SDMA AG packet dispatch |
| c1: scatter → compute-wait | 0.31 | reduce c1 on CB |
| c1: compute-wait → barrier | 0.16 | |
| c1: barrier → AG-submit | 0.21 | |
| **AG-submit → AG-wait-done** | **0.54** | SDMA AG total wait (→ replaced) |
| c0 AG per-chunk done | 0.72 | |
| c1 AG per-chunk done | 0.53 | |
| CB1 reduce c0 (sct-poll→reduce-done) | 0.29 | |
| CB1 reduce c1 | 0.42 | |

### Plan A expected vs baseline

| Phase | BASELINE | Plan A 预期 | Δ |
|---|---|---|---|
| Reduce c0/c1 | 0.29 / 0.42 | ≈ same (single store vs baseline) | ~0 |
| Cross-PE barrier (c0+c1) | 0.30 total | same mechanism | 0 |
| AG phase | **SDMA 0.54 ms** | **CU XGMI pull 256MB + local write** | new measurement |
| External hipMemcpyAsync/AR | ~0.35 ms (Entry 16) | **removed** | **-0.35 ms/AR × 4 = -1.4 ms wall** |
| CU-GEMM contention | 0 (SDMA免CU) | new CU occupancy during AG | +0.3-0.5 ms (< Plan B's 0.81) |

### Theoretical gain (R10 historical refs)
- Entry 16: "4 AR × 0.35 ms hipMemcpyAsync = 1.4 ms" → eliminated
- Entry 18: Plan B overlap regression 0.81 ms upper bound on CU-GEMM contention
- Plan A wall = 7.802 − 1.4 + 0.3-0.5 = **6.7 – 6.9 ms**
- Expected gap vs RCCL (7.422 or 7.565) = **-0.5 to -0.9 ms (SUPER)** if CU
  XGMI pull BW at 256 MB/8 PE ≈ Entry 18 micro-bench 370 GB/s

### Code changes in this commit
- `include/mori/collective/allreduce/pipelined_allreduce_sdma_kernel.hpp`:
  - Removed `PipelinedCuReduceAgCopyKernel` (Plan B, β partition)
  - Added `PipelinedXGMIPullKernel` (Plan A, unified all-CB flow,
    方式 2 single-write + unified AG pull)
- `src/collective/core/twoshot_allreduce_sdma_class.cpp`:
  - `use_plan_a` branch: switched K2 launch to `PipelinedXGMIPullKernel`
  - Added `hipMemsetAsync(&barrierPtr_->ag_sync, 0, sizeof(uint32_t))`
    before launch (Plan A uses ag_sync for block 0 → compute blocks
    AG-ready signaling; zero init required)
  - Updated Plan A branch announce log (from "Plan B" to "Plan A")
  - Updated comments documenting Plan A design

### Next step (Step 2)
User runs in Linux container (`smci355-ccs-aus-m01-33`):

```bash
cd /home/fizhang/test/mori && bash tools/bench_plan_a.sh
```

`tools/bench_plan_a.sh` (added in this commit) handles:
- preflight (hipcc / cmake / python3 / torch.cuda)
- `git pull origin sdma-test`
- **build**: `BUILD_EXAMPLES=ON BUILD_TESTS=ON pip3 install .` (correct
  form; pybind11 is a build-time dep from `pyproject.toml` and gets
  auto-installed by pip's build-isolation, which `--no-build-isolation`
  wrongly bypasses — that caused a failed build in the first attempt)
- Runs BASELINE + PLAN_A wall (iter=100) + AR[0] phase timing both
- Prints comparison table at end

Compare `overlap` to BASELINE 7.802 / RCCL 7.565 / RCCL Entry 18 7.422.
All Test 6 variants (copy/no-copy/RCCL) must correctness-pass.

---

## Entry 18 — Plan A/B (K1=SDMA scatter + K2=CU reduce+AG+copy) FAILED
- **Date**: 2026-04-23
- **Commits**: `ec612289` (A v1), `c652a66c` (A fix), `392165ef` (A v2 2-kernel), `d5143321` (B v3 β partition), `5f0072e7` (B min-chunk fix) — **代码仍在 HEAD, 未 revert**
- **Context**: 承接 Entry 16/17 → "SDMA no-copy 已超 RCCL 0.5ms, 唯一 gap 是 4 × hipMemcpyAsync copy". Entry 17 E'' (CU 在 AG wait 时 copy) 因 HBM 争用失败. 本次试图让 K2 全 CU 做 reduce+AG+copy, pipeline 隐藏 copy, 不再有外部 hipMemcpyAsync.
- **Design (user-specified plan)**:
  - K1 = `ScatterSdmaOnlyKernel` (SDMA scatter, 1 block 512 threads)
  - K2 = `PipelinedCuReduceAgCopyKernel` (blocks=161, threads=512):
    - block 0: 仅打 phase_ts 然后退出 (浪费 1 block)
    - R-group (blocks 1..nR): reduce chunk c + CU XGMI **push** AG chunk c to all peers
    - C-group (blocks nR+1..nR+nC): copy chunk c-1 from local transit → user_output
  - 默认 numChunks=8, β partition = 7:1 (nR=140, nC=20)
  - `copy_output_to_user=True` 时跳过外部 hipMemcpyAsync (skip_external_copy=true)
- **Measured data (256MB, 4 stages, iter=100)**:
  | 变体 | overlap wall | seq_ar | seq_gemm | overlap/seq_gemm | vs RCCL |
  |---|---|---|---|---|---|
  | BASELINE | 7.798 | 5.188 | 3.674 | 2.122 | **+5.1%** (差 0.376ms) |
  | **DIRECT (Plan B)** | **8.955** | 5.538 | 3.676 | **2.436** | **+19.6%** (差 1.466ms, **+1.16ms WORSE**) |
  | RCCL | 7.422 / 7.489 | 5.129 / 5.134 | 3.678 / 3.680 | 2.018 / 2.035 | — |
  - All Tests PASSED correctness (Test 1/1b/2/3/4/6 全过)
- **Mechanism analysis — 双重失败**:
  1. **seq_ar 层面就慢了 0.088 ms/AR** (5.538 vs 5.188, 4 stages → 0.35ms). 说明**即使不和 GEMM 争**, CU AG push + CU copy 本身也比 SDMA AG + hipMemcpyAsync 慢. CU XGMI push 没有硬件 DMA 优化, 28MB/chunk/180GB/s ≈ 155μs vs SDMA AG ~130μs/chunk (warm).
  2. **overlap 额外退化 0.81 ms**: CU 做 AG+copy 时占满 compute unit (Plan B 用 140 blocks × 512 threads = 71680 CU threads), GEMM overlap 打不满. overlap/seq_gemm 2.12 → 2.44.
- **Root cause**: micro-bench 决策 (cu_xgmi_bench 370 GB/s) **测的是 CU XGMI read (pull), 不是 Plan B 实际用的 CU XGMI write (push)**. 方向不匹配. CU push BW 从未测过. **违反 R0 + R6**.
- **已确认可做的 instrumentation bug (R0 违规)**:
  - `PipelinedCuReduceAgCopyKernel` 的 phase timing: copy done slot `11+3c+2` 由 `ar_write_phase_ts_cb1` 写入, 但该 helper 只让 blockIdx==1 (R-group 第一个 block) 写, 而 R-block 从不进入 C-group 代码路径. **copy phase timing 永远是 0, 开发全程从未测过 copy wall**.
- **User 指出的设计质疑 (未解决)**:
  1. AG 方向: **push 正确** (pull 时机难确定, 需额外 per-chunk signal 等待)
  2. **β partition 7:1 (140:20) 不合理, 且无数据支撑**. 理论估算 (BW: 本地 HBM 400 GB/s, XGMI push ~180 GB/s):
     - reduce = 36MB / (nR × 400/160 GB/s) ≈ 90μs × 160/nR
     - AG push = 28MB / (nR × 180/160) ≈ 155μs × 160/nR
     - copy = 64MB / (nC × 400/160) ≈ 160μs × 160/nC
     - 同步条件 (R-group reduce+AG = C-group copy): nR:nC ≈ **60:40** (而非 87.5:12.5)
- **Code status**: 代码在 `5f0072e7` HEAD **未 revert** (user 指示保留 Plan B 代码继续调试分区).
- **Next step options (待 user 决定)**:
  1. **调 Plan B**: 修 phase timing bug → 改 nR 为可配置 (默认 96) → 补 CU XGMI push micro-bench → 跑 phase sweep → 找最优 nR → 对比 wall. 理论最优 Plan B 单次 AR = 1.24ms vs baseline 1.30ms → 每 AR 省 ~0.06ms, 4 AR 省 0.24ms → overlap wall 7.56ms (优于 RCCL 7.42 的 0.14ms, 勉强达标).
  2. **Revert Plan B → π' 方向**: copy 移到独立 SDMA queue + 双缓冲 transit, 下次 AR 不等上次 copy. Entry 16 提出, 未实施. 预期收益 1.05ms (3 个 copy 完全 hide, 1 个 tail 暴露 0.35ms).
  3. **Revert Plan B → D'' 方向**: 显式 `register_user_output(ptr, size)` API, user 显式管理生命周期, 消掉整个 copy 步骤. Entry 11 D' 失败因 PyTorch allocator auto-release; D'' 由 user 控制避开.
- **Rule violation reflection (R0/R6)**:
  - cu_xgmi_bench 测错方向 (read 而非 write) → micro-bench 必须匹配实际使用方向
  - copy phase 从未打点却已开发 5 个 commits (Plan A/B 全系列) → 违反 "先打点后改"
  - β partition 7:1 基于错误 data volume comment ("AG 是 copy 7×"), 实际 copy 64MB > AG 28MB
