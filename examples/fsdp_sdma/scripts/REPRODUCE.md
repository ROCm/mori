# Fixed reproduction recipes (UT + E2E)

Two canonical scripts. Edit only the `MASTER/WORKER/MASTER_IP` CONFIG block at the
top of each; everything else (env vars, NIC list, per-world config, sizes, model
config) is fixed so every run is identical.

- `run_ut.sh`  — standalone AllGather bandwidth (`tests/python/ccl/bench_sweep.py`)
- `run_e2e.sh` — FSDP2 training step (`examples/fsdp_sdma/bench.py`, Qwen-7B vocab=32000)

## Common environment (both scripts, set internally)

| var | value | why |
|-----|-------|-----|
| `GLOO/NCCL/MORI_SOCKET_IFNAME` | `eth0` | front-end control socket |
| `NCCL_IB_HCA` / `MORI_RDMA_DEVICES` | `mlx5_0,2,3,4,5,7,8,9` | the 8 RoCEv2 data NICs (mlx5_1/6 are front-end, excluded) |
| `NCCL_IB_GID_INDEX` | `3` | RoCE v2 GID index |
| `MORI_ENABLE_SDMA` | `1` | enable the SDMA intra-node path |
| `MORI_SHMEM_HEAP_SIZE` | `34359738368` (32 GiB) | fits 512MB × 16-rank output buffers (UT) |
| `PYTHONPATH` | `<repo>/python` (+ `examples/fsdp_sdma` for E2E) | built mori + adapter |

## UT — `run_ut.sh <w8|w16> [overlap]`

Per-world config (the exact env the adapter selects per topology):
- **w8** (4 GPU/node, `HIP_VISIBLE_DEVICES=0,1,2,3`, nproc=4):
  `MORI_HIER_FUSE_LOCAL=1 MORI_HIER_FUSE_REMOTE=1 MORI_HIER_LOCAL_PUSHONLY=1 MORI_HIER_DEEP_PIPE=auto MORI_SDMA_NUM_CHANNELS=8 MORI_HIER_NIC_NUMA_LOCAL=1`
  sizes 4-512 MB.
- **w16** (8 GPU/node, `HIP_VISIBLE_DEVICES=0..7`, nproc=8):
  `MORI_HIER_CROWN=1 MORI_HIER_DEEP_PIPE=auto MORI_HIER_NIC_NUMA_LOCAL=1`
  sizes 8-512 MB. (Do NOT add FUSE_*/NC=8 to w16 — it degrades the crown schedule.)
- `overlap` variant runs `bench_ring_full.py` (AG isolated vs under a concurrent GEMM).

```bash
bash run_ut.sh w8            # -> sweep_w8.csv
bash run_ut.sh w16           # -> sweep_w16.csv
bash run_ut.sh w8 overlap    # -> ut_w8_overlap_m.log (per_rank= lines)
```

## E2E — `run_e2e.sh <w8|w16> <native|mori>`

- Qwen-7B config with `vocab_size=32000` (`/home/mingzliu/sdma/qwen32k/config.json`),
  seq-len 2048, 20 steps, warmup 6, bf16. `--model-name-or-path` points at that dir.
- **native**: framework-default all-gather (RCCL). **mori**: `MORI_ENABLE_SDMA=1` +
  a self-configuring drop-in backend (no extra env — the script/adapter pick it).
- w8: nproc=4 devs 0-3; w16: nproc=8 devs 0-7.

```bash
bash run_e2e.sh w16 native   # -> e2e_w16_native_m.log (avg_tflops_per_gpu, last_loss)
bash run_e2e.sh w16 mori     # -> e2e_w16_mori_m.log
bash run_e2e.sh w8  mori     # -> e2e_w8_mori_m.log
```

### mori backend routing (per world) — the fix

`bench.py --mode mori` selects the FSDP2 all-gather backend from env
(`_apply_fsdp2`), and the two worlds need DIFFERENT backends:

| world | GPU/node | backend | env the script sets |
|-------|----------|---------|---------------------|
| w8  | 4 | `HierAllGather` **host-proxy** (`HostProxyHierAllGather`, CPU-posted) | `MORI_ENABLE_SDMA=1 MORI_FSDP_ENABLE_HIER=1 MORI_FSDP_HOST_PROXY=1 MORI_FSDP_HOSTPROXY_CAP_MB=512 MORI_SHMEM_HEAP_SIZE=17179869184` |
| w16 | 8 | `MoriAllGather` / device IBGDA `HierAllGather` | `MORI_ENABLE_SDMA=1 MORI_FSDP_ENABLE_HIER=1` |

Root cause (Team A): the **shipped default** (`MORI_ENABLE_SDMA` only) routes
`--mode mori` to `MoriSdmaAllGather` (oneshot SDMA), which **null-derefs
cross-node at BOTH w8 and w16** (`Memory access fault … address (nil)` right
after `Registered output buffer`) — the original w16 "Slow wait / hang".
`run_e2e.sh` therefore always adds `MORI_FSDP_ENABLE_HIER=1` for mori, routing
to `HierAllGather`. Then:

- **w16 (rpn=8)**: the device IBGDA `HierAllGather` works; the adapter
  (`examples/fsdp_sdma/mori_allgather.py`) self-configures the **rpn≥8 bit-exact
  base** with zero extra env (below).
- **w8 (rpn=4)**: the device `HierAllGather` **SIGSEGVs at the first AG** (an
  rpn=4 kernel bug that would need a `.so` rebuild to fix). The reliably
  completing, bit-exact path is the **CPU-posted host-proxy transport**
  (`MORI_FSDP_HOST_PROXY=1`, the campaign's proven bit-exact E2E backend), which
  `run_e2e.sh` selects only for w8.

The adapter's rpn≥8 (w16) auto-config:

| adapter auto-set (rpn≥8) | why |
|--------------------------|-----|
| `MORI_FSDP_NO_ZERO_COPY=1` | the zero-copy param-contiguous scatter produces **uniform-garbage** output at 8 ranks/node (loss stuck at `ln(vocab)`≈10.37); route to the rank-major copy-out path |
| `MORI_HIER_DEBUG_SYNC=1` | full per-op **host stream.synchronize()** landing fence — the only bit-exact cross-PE RDMA/SDMA drain at 8 ranks/node; also disables the HIP-graph capture below |
| `MORI_HIER_CUDA_GRAPH=0` | the copy-out `__call__`'s HIP-graph capture **poisons the HIP context** at w16 → `Shmem state is not initialized` SIGABRT; redundant once DEBUG_SYNC is on, but explicit |

w8 (rpn==4) uses the host-proxy transport, so the device-hier branch above does
not apply to it.

### Verified results (Qwen-7B vocab=32000, seq2048, 20 steps / warmup 6, bf16)

| run | last_loss | avg_tflops/gpu | verdict |
|-----|-----------|----------------|---------|
| w16 native (GT) | `11.058271408081055` | ~123.67 | reference (deterministic) |
| w16 mori | `11.058271408081055` | ~119–125 | **BIT-EXACT** (identical loss every window), ratio ≈ **0.96–1.01×** native. Reproduced 4×. |
| w8 native (GT) | `11.048338890075684` | ~145–150 | reference — **non-deterministic run-to-run** (RCCL backward reduce-scatter atomics); a second native draw gave `11.050827980041504` |
| w8 mori | `11.048338890075684` | ~115–124 | **BIT-EXACT** to a native draw (all 4 windows: 11.070929 / 11.130214 / 11.105951 / 11.048339), ratio ≈ **0.80×** native. Host-proxy is deterministic (reproduced 2×) and trades throughput for the bit-exact host-completion fence. |

The mori run prints its JSON summary (`avg_tflops_per_gpu`, `last_loss`) and then
emits a benign `Shmem state is not initialized` SIGABRT during `shmem_finalize`
teardown — **after** the summary, so the result is complete. `grep last_loss`
on `e2e_w16_mori_m.log` is the source of truth.

The script warns `Slow wait detected` if the mori inter-node ring stalls (cross-node
RDMA hang). Note: the vocab=152064 default of the built-in bench.py config produces a
~2 GB embed AG that can trip a cross-node slow-wait at w16; the vocab=32000 config
above matches the campaign's bit-exact E2E recipe.

## overlap UT
`run_ut.sh` also runs the GEMM-overlap contention UT (AllGather under a concurrent CU-saturating GEMM, world=8 & 16) via tests/python/ccl/bench_gemm_overlap.py — the no-CU-contention dividend.
