# cco MoE dispatch / combine (ops v2)

MoE dispatch + combine built on **mori-cco**: intranode (single-node, EP8) over
the flat symmetric LSA VA, and internode (multi-node, e.g. 2×8) over CCO/GDA
RDMA. One op class, `EpDispatchCombineOp`, behind two interchangeable kernel
backends:

- **flydsl** (default): FlyDSL device kernels, the full feature set (gather +
  scatter combine, fp8/fp4, quant, StdMoE, per-token scales, routing replay).
  Intranode only.
- **hip**: C++/HIP kernels JIT-compiled by the v2 JIT framework. Gather combine
  only, in bf16/fp32; the dispatch leg also carries fp8 and fp4 (transport only —
  it moves an already-quantized payload). A dedicated gfx125x TDM body is selected
  by arch. **The only backend with an internode path** (bf16/f32/fp8, no fp4, no
  quant/StdMoE/scatter). Works on a machine with no FlyDSL. See
  `docs/MORI_JIT_V2_DESIGN.md`.

Select with `cfg.kernel_backend` or `MORI_V2_KERNEL_BACKEND`. Same-node peer
addresses are computed in-kernel over the flat LSA VA, no host P2P tables; a peer
on another node is reached with GDA RDMA. Reference =
ROCm/FlyDSL PR #522 (`dispatch_combine_intranode_{kernel,op}.py`).

Supported token dtypes: **bf16**, **f32**, **fp8** (gather-only; OCP e4m3 on
gfx950, e4m3**fnuz** max 240 on gfx942) and **fp4** (e2m1, gather-only,
**gfx950-only** — the `cvt_scalef32_*_fp4` intrinsics don't exist on gfx942).
Combine: gather (UseP2PRead) **and** scatter (`_nop2p`); weighted combine
(`out_weights`); StdMoE (ConvertDispatchOutput / ConvertCombineInput, standalone
+ wired into the op); fp8 combine-wire **quant** (`fp8_direct_cast` **and**
`fp8_blockwise`, scatter-only — distinct from the plain fp8 token dtype, which
keeps a bf16 external payload); per-token scales forwarding;
`max_total_recv_tokens` cap; mori-parity host op-layer + per-device,
dtype-aware tuning table. Not done: `skip_stage1` (FlyDSL-only).

This is a real package: `from mori.ops.dispatch_combine_v2 import
EpDispatchCombineConfig, EpDispatchCombineOp`. Importing it pulls in **no** kernel
backend — the base and config live in `dispatch_combine_op.py`; each backend is imported
lazily, only when selected, so the package imports without FlyDSL installed.

## Layout

| file | role |
|---|---|
| `dispatch_combine_op.py` | backend-agnostic base + entry: `EpDispatchCombineConfig` (+`.tuned()`), `EpDispatchCombineOp` (backend selector + shared dispatch/combine/reset/lifecycle), `EpDispatchRoutingHandle`, `KernelSet` |
| `flydsl_backend.py` | **flydsl** backend subclass (`EpDispatchCombineOpFlyDSL`): arena layout + FlyDSL kernel binding for the full feature set |
| `hip_backend.py` | **hip** backend subclass (`EpDispatchCombineOpHip`): arena layout + C++/JIT plan binding, gather only; rejects unsupported configs at construction. Holds the internode path too — device communicator, per-geometry plan sets for the `copystaging`/`dispatch{,_ll}` and `combinesync`/`combinesyncbarrier`/`combine{,_ll}`/`combineall` pass sequences, and its own unsupported-config gate |
| `ep_plans.py` | EP-specific shim: loads `libmori_ops_v2.so` and exposes `EpDispatchPlan`/`EpCombinePlan`. The generic ctypes binding it calls lives in `mori.jit.v2.plan_api` (the plan_api C ABI), not here |
| `symm_arena.py` | `SymmArena`: one cco-LSA window carved into named regions |
| `internode_regions.py` | `internode_regions(cfg)`: the internode op's `SymmArena` region list — 17 named `(name, nbytes)` regions (16 plus `out_scales` when scales are on), in v1's allocation order and transcribed from what `EpDispatchCombineHandle` allocated |
| `flydsl_prims.py` | FlyDSL device primitives: system atomics / ordered stores / fences / volatile-spin waits |
| `intranode_kernels.py` | FlyDSL kernel factories: `make_dispatch` (+scales/replay), `make_combine` (gather) / `make_combine_scatter` (`_nop2p`, bf16/f32/fp8/fp4), `make_convert_dispatch_output` / `make_convert_combine_input` (StdMoE), `make_local_expert_count` |
| `tuning_configs.py` | **flydsl** kernel geometry: per-(world,hidden,topk) block/warp lookup |
| `hip_tuning_configs.py` | **hip** kernel geometry, separate table (never borrows flydsl's); same `lookup` contract. Independent dispatch/combine tables, keyed by device, shape, topk and (dispatch only) dtype; an unswept shape gets a single-shot default |
| `internode_tuning_configs.py` | **hip** internode kernel geometry, a third table: token-count buckets keyed by device, shape, topk and dispatch dtype, carrying `(block_num, rdma_block_num, warp_num)` **per phase** — dispatch and combine are tuned to different values over one shared arena. The internode plans are compiled per geometry, so the backend walks the whole table at build time and `lookup` only picks a prebuilt bucket; `block_num` is clamped to the CU count |

## Internode config

There is no kernel-type enum: the internode path is selected by
`gpu_per_node < world_size` (`cfg.is_internode`, i.e. `cfg.nodes > 1`), and only
the **hip** backend implements it.

| field | default | role |
|---|---|---|
| `gpu_per_node` | `None` → `world_size`, i.e. one node | GPUs per physical node; `world_size` must be a positive multiple of it. Setting it smaller is what selects the internode path. It is EP's own idea of a node and is checked against the communicator's LSA team (`lsa_size`, `lsa_rank`) at construction |
| `internode_kernel` | `"auto"` | Which internode kernel family runs. `"v2"` = the general path (chunked, deduplicating, sized for wide tokens); `"v2_ll"` = low latency (no dedup across expert slots, one entry per node per token). They are separate JIT modules, so naming one compiles only that one and it cannot fall back; `"auto"` compiles both and chooses per launch |
| `internode_auto_ll_max_tokens` | `512` | The `"auto"` crossover, compared against **this call's** token count (`input.shape[0]` for dispatch, `routing.cur_rank_num_token` for combine): `<=` runs `v2_ll`, `>` runs `v2`. One op therefore alternates as the batch changes. Unrelated to `max_num_inp_token_per_rank`, which is the capacity. Read only when `internode_kernel == "auto"`; must be >= 0 |
| `num_qp_per_pe` | `2` | QPs per peer on the RDMA leg. Only the internode path reads it, and 1 starves it (~1.5x), so the default is what that path wants; the intranode path ignores it. Must be >= 1 |

Two further internode-only rules `__post_init__` applies: `quant_type` must be
`"none"` (the internode combine's fp8 staging path is incomplete and returns
wrong tokens), and `max_num_inp_token_per_rank` is rounded up to a multiple of
the wavefront width, because send slots are handed out in whole wavefronts.

Tests/bench live under `tests/python/ops/dispatch_combine_v2/`:

| file | role |
|---|---|
| `test_dispatch_combine_v2_intranode.py` | pytest wrapper: runs `test_op.py` under torchrun for the representative modes and asserts every line PASS |
| `test_dispatch_combine_v2_internode.py` | the internode entry: a torchrun script (not a pytest wrapper) for correctness, bench and tuning over CCO/GDA. `--cmd test\|bench\|tuning\|stress`, `--max-tokens`, `--hidden-dim`, `--topk`, `--dtype`/`--combine-dtype`, `--num-qp` (default 1), `--kernel-type auto\|v2\|v2_ll`, `--auto-ll-max-tokens`, `--rounds`, `--spawn`. **Needs two nodes**: the op refuses a config whose node grouping disagrees with the communicator's LSA team, so one host cannot emulate it |
| `test_internode_regions.py` | pure-Python invariants of `internode_regions()`: the name contract with the backend and the capacity bounds the kernel's indexing implies. No GPU, no process group |
| `test_op_lifecycle.py` | arena-leak regression: rebuilding the op on one long-lived `Communicator`; `close()` must free and untrack the window. `torchrun --standalone --nproc_per_node=2` |
| `test_op.py` | EP8 op-layer test (gather/scatter, quant, StdMoE, recv-cap, scales, LEC, reset, replay). `MORI_V2_KERNEL_BACKEND=hip` runs it against the HIP kernels |
| `test_ep_backend_parity.py` | runs both backends in one process on the same input and compares element for element |
| `test_jit_binding.py` | JIT plan binding: schemas, request/args round-trip, cache behaviour. No GPU peers needed |
| `test_graph_capture.py` | captures dispatch → identity expert → combine as one HIP graph and replays it |
| `test_asym_dtype.py` | asymmetric dtype legs (fp8/fp4 dispatch + bf16 combine) |
| `bench_ep.py` | the perf bench, for every backend. Alternating dispatch/combine pairs, eager + CUDA graph, each point gated on an identity-expert check and non-zero exit on failure. Envs: `BACKENDS=flydsl,hip`, `MODES=eager,graph`, `SWEEP`, `ITERS`, `DISP=bf16\|fp8\|fp4`, `COMBINE_IN=inplace\|staged`, `CHECK=0`, `DBN`/`DWPB`/`CBN`/`CWPB` to pin geometry, `HIDDEN`/`TOPK`/`EPR` |

(Each script inlines a tiny torchrun/gloo `Dist` bootstrap — gloo only carries the cco unique-id and pass/fail counts.)

## Run (inside the container, 8 GPUs)

`torchrun --standalone` uses a localhost rendezvous, so no socket-iface env is
needed. These are the intranode paths (no GDA/RDMA).

```bash
cd tests/python/ops/dispatch_combine_v2

pytest test_dispatch_combine_v2_intranode.py -v                       # EP8 correctness (all modes)
pytest test_internode_regions.py -v                                   # arena layout, no GPU
torchrun --standalone --nproc_per_node=8 test_op.py                   # op-layer correctness (env-driven)
BACKENDS=flydsl,hip torchrun --standalone --nproc_per_node=8 bench_ep.py   # perf, both backends
```

Config via env: `HIDDEN`, `TOPK`, `EPR`, `SWEEP`, `DISP`, `COMBINE`, `QUANT`,
`BACKENDS`, `MODES`, `ITERS`, `DBN`/`DWPB`/`CBN`/`CWPB`.

## Run internode (two nodes, 8 GPUs each)

The internode entry is a plain torchrun script, one launch per node; `--cmd`
selects correctness, bench or the geometry sweep.

```bash
# rank 0 of 2, from the repo root. --spawn defaults to 8, so torchrun runs ONE
# process per node and the script builds the 8 ranks; pass --spawn 0 to use
# --nproc_per_node=8 instead (the two topologies together are refused).
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr=<ip> --master_port=<port> \
    tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py \
    --cmd test --max-tokens 128 --num-qp 2

# or drive one rank through the shared runner (--entry defaults to the v1 harness)
tools/run_internode_test.sh --rank 0 --master-addr <ip> --ifname <nic> \
    --cmd test --max-tokens 128 \
    --entry tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py
```

Set `MORI_RDMA_TC` / `MORI_RDMA_SL` to match the fabric's RoCE class; unset, every
transfer goes out on SL 1 / DSCP 0. `tools/env_setup.sh` derives both from
`ROCE_DSCP`/`ROCE_PRIO`.

## Design notes (intranode kernels)

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

## Perf (EP8, hidden=7168, top-k=8, 256 experts; dispatch 64blk / combine 128blk × 16warp, CUDA-graph, bf16)

Per-rank bandwidth = `recv_tok * per_token_bytes / time` (the bench sizes the
payload per dtype, `hidden*2` for bf16). Indicative bf16 numbers on **MI308X
(gfx942)** xGMI:

| tok/rank | dispatch | combine |
|---:|---:|---:|
| 512  | 268 GB/s | 213 GB/s |
| 2048 | 306 GB/s | 294 GB/s |
| 8192 | 314 GB/s | 323 GB/s |

Cross-impl (v2 vs mori v1) latency tables for fp8/fp4 are in PR ROCm/mori#448.
