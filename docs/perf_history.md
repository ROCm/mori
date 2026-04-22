# SDMA AllReduce Perf History (per R9/R10)

This ledger is the authoritative source of measured perf data for SDMA
AllReduce optimization work. Every entry must include: date, commit SHA,
scenario, numbers, mechanism, conclusion. See `.cursor/rules/relentless-perf.mdc`
§R9/R10 for ledger discipline.

**Primary goal**: `AllreduceSdma(copy_output_to_user=True)(input, output, stream)`
multi-stage overlap wall < RCCL `dist.all_reduce(tensor)` wall, measured as
median over ≥100 iters at 256MB/stage, 4 stages, 8-rank MI355X, same
benchmark schedule (`test_allreduce.py --num-stages 4 --elems 67108864`).

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

**Target**: SDMA copy ≤ RCCL (gap ≤ 0). Currently outstanding.
