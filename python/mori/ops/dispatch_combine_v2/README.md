# cco-LSA intranode MoE dispatch / combine (ops v2, FlyDSL)

Intranode (single-node, EP8) MoE dispatch + combine built on **mori-cco LSA**
(intra-node P2P over the flat symmetric VA) and **FlyDSL** device kernels. A
mori-parity reimplementation that swaps the mori-shmem provider for cco-LSA:
peer addresses are computed in-kernel via `cco.Window(arena).lsa_ptr(pe, off)`,
no host P2P tables. Reference = ROCm/FlyDSL PR #522
(`dispatch_combine_intranode_{kernel,op}.py`).

Supported: bf16 + f32 token dtype, weighted combine (`out_weights`), StdMoE
(ConvertDispatchOutput / ConvertCombineInput), and a mori-parity host op-layer.
Not yet: fp8/fp4 wire dtype (combine accum scaffolding present, dispatch-side
TODO), scatter/`skip_stage1` combine variants, scales.

## Layout

| file | role |
|---|---|
| `symm_arena.py` | one cco symmetric window carved into named sub-regions (`SymmArena`) |
| `flydsl_prims.py` | device primitives: system atomics / ordered stores / fences / volatile-spin waits |
| `dispatch_kernel.py` | `make_dispatch` — P2P-scatter tokens to their experts' ranks (3 phases) |
| `combine_kernel.py` | `make_combine` — P2P-read (gather) each token's k expert outputs back & reduce in f32; weighted `out_weights`; bf16/f32/fp8 accum |
| `stdmoe_kernel.py` | `make_convert_dispatch_output` / `make_convert_combine_input` — StdMoE per-expert repack + weighted inverse (local kernels) |
| `dispatch_combine_op.py` | `EpDispatchCombineOp` / `EpDispatchCombineConfig` / `EpDispatchRoutingHandle` — mori-parity host op-layer |
| `dist_common.py` | torchrun bootstrap (gloo only carries the cco unique-id) |
| `bench_dispatch_combine.py` | eager + CUDA-graph perf bench; end-to-end correctness during warmup. `STDMOE=1` runs the StdMoE pipeline; `DTYPE=bf16\|f32` |
| `test_op.py` | EP8 correctness test for the host op-layer (bf16 + f32) |

## Run (inside the gfx942 container)

```bash
cd python/mori/ops/dispatch_combine_v2
unset PYTHONPATH LD_LIBRARY_PATH MORI_CCO_BC
export MORI_SOCKET_IFNAME=enp159s0np0 MORI_CCO_GDA_CONN=full

torchrun --standalone --nproc_per_node=8 bench_dispatch_combine.py   # correctness + perf
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
