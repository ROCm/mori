# cco-LSA intranode MoE dispatch / combine (ops v2, FlyDSL)

Intranode (single-node, EP8) MoE dispatch + combine built on **mori-cco LSA**
(intra-node P2P over the flat symmetric VA) and **FlyDSL** device kernels. A
mori-parity reimplementation that swaps the mori-shmem provider for cco-LSA:
peer addresses are computed in-kernel via `cco.Window(arena).lsa_ptr(pe, off)`,
no host P2P tables. Reference = ROCm/FlyDSL PR #522
(`dispatch_combine_intranode_{kernel,op}.py`).

Supported: bf16 + f32 token dtype; gather (UseP2PRead) **and** scatter
(`_nop2p`) combine; weighted combine (`out_weights`); StdMoE
(ConvertDispatchOutput / ConvertCombineInput, standalone + wired into the op);
fp8_direct_cast **and fp8_blockwise quant** (scatter); per-token scales
forwarding; `max_total_recv_tokens` cap; mori-parity host op-layer + tuning
table. fp4 accum branch present but gfx950-only (these cvt intrinsics don't
exist on gfx942). Note: gfx942 fp8 is e4m3**fnuz** (max 240). Not done:
`skip_stage1` (FlyDSL-only).

## Layout

| file | role |
|---|---|
| `flydsl_prims.py` | device primitives: system atomics / ordered stores / fences / volatile-spin waits |
| `intranode_kernels.py` | all FlyDSL intranode kernel factories: `make_dispatch` (+scales/replay), `make_combine` (gather) / `make_combine_scatter` (`_nop2p`, bf16/f32/fp8/fp4), `make_convert_dispatch_output` / `make_convert_combine_input` (StdMoE), `make_local_expert_count` |
| `dispatch_combine_op.py` | `SymmArena` + `EpDispatchCombineOp` / `EpDispatchCombineConfig` (+`.tuned()`) / `EpDispatchRoutingHandle` — mori-parity host op-layer (scales, scatter/quant combine, StdMoE, recv cap, LEC, reset, replay) |
| `tuning_configs.py` | per-(world,hidden,topk) block/warp lookup |

Tests/bench live under `tests/python/ops/dispatch_combine_v2/`:

| file | role |
|---|---|
| `dist_common.py` | torchrun bootstrap (gloo only carries the cco unique-id) |
| `bench_dispatch_combine.py` | eager + CUDA-graph perf bench + e2e correctness. Envs: `DTYPE=bf16\|f32`, `COMBINE=gather\|scatter`, `QUANT=none\|fp8_direct_cast\|fp8_blockwise`, `STDMOE=1`, `SCALE_DIM` |
| `test_op.py` | EP8 op-layer test (gather/scatter, quant, StdMoE, recv-cap, scales, LEC, reset, replay) |
| `run_bench.sh` | bench launcher (runs `bench_dispatch_combine.py` in the container) |

## Run (inside the gfx942 container)

```bash
cd tests/python/ops/dispatch_combine_v2
unset PYTHONPATH LD_LIBRARY_PATH MORI_CCO_BC
export MORI_SOCKET_IFNAME=enp159s0np0 MORI_CCO_GDA_CONN=full

torchrun --standalone --nproc_per_node=8 bench_dispatch_combine.py   # correctness + perf
torchrun --standalone --nproc_per_node=8 test_op.py                  # op-layer correctness
```

Config via env: `HIDDEN`, `TOPK`, `EPR`, `SWEEP`, `DISP_BLOCK`/`COMB_BLOCK`, `WARP_NUM`,
`MODE=eager|graph|both`.

## Design notes

- **dispatch**: per (token, k) dedup same-dest-PE via ballot; lane0 remote
  `atomic_add` allocates a recv slot; publish origin id + idx/wts + 16B dual-issue
  token copy to the peer; grid barrier; per-peer count signal; collect `total_recv`.
- **combine** (gather, = mori `UseP2PRead`): cross-device entry barrier, then each
  local token gathers its k expert outputs **remotely** from `peer.out_tok[dest_tok_id]`
  and reduces in f32. Register-light i32 reads (2 bf16 / `v2f32` accumulate) + 2-way
  unroll keep VGPRs low so 16 warps/block run at high occupancy to hide xGMI read
  latency; remote reads are latency-bound so combine needs ~128 blocks, while
  dispatch's posted writes saturate at ~64 blocks (half the CUs).
- Self-written volatile/atomic spin-waits (`flydsl_prims.spin_until_*`) — mori-shmem's
  `wait_until_*` assert on a cco-only stack. Counters self-reset in-kernel → CUDAGraph-safe.

## Perf (EP8, hidden=7168, top-k=8, 256 experts; dispatch 64blk / combine 128blk × 16warp, CUDA-graph)

Per-rank bandwidth (`recv_tok * hidden * 2 / time`); saturates near MI300X xGMI
(mori reference on the same node: dispatch 304, combine 333 GB/s):

| tok/rank | dispatch | combine |
|---:|---:|---:|
| 512  | 268 GB/s | 213 GB/s |
| 2048 | 306 GB/s | 294 GB/s |
| 8192 | 314 GB/s | 323 GB/s |
