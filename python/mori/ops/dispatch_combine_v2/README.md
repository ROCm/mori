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
  by arch. **The only backend with an internode path** (bf16/f32/fp8, plus packed
  fp4 dispatch with bf16/fp32 combine; no quant/StdMoE/scatter). Works on a machine
  with no FlyDSL. See `docs/MORI_JIT_V2_DESIGN.md`.

Select with `cfg.kernel_backend` or `MORI_V2_KERNEL_BACKEND`. Same-node peer
addresses are computed in-kernel over the flat LSA VA, no host P2P tables; a peer
on another node is reached with GDA RDMA. Reference =
ROCm/FlyDSL PR #522 (`dispatch_combine_intranode_{kernel,op}.py`).

Supported token dtypes: **bf16**, **f32**, **fp8** (gather-only; OCP e4m3 on
gfx950, e4m3**fnuz** max 240 on gfx942) and **fp4** (e2m1, gather-only,
**gfx950 and gfx1250** — the conversion intrinsics don't exist on gfx942. The
two differ: gfx950 uses the pair-at-a-time `cvt_scalef32_pk_*_fp4` family,
gfx1250 a single pack-of-8 per i32, so `_accum_funcs` branches on `WAVE`).
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
| `hip_tuning_configs.py` | **hip** configuration selection: `resolve_schedule(cfg)` returns `ScheduleEntry` rows for both intra and inter. Each phase has a `KernelConfig(family, block_num, warp_per_block, rdma_block_num)`. Inter reads JSON; intra reads `IntraNodeV2` JSON first and falls back to Python tables per phase, whose dispatch half is keyed by `dispatch_dtype_str` (bf16/fp8/fp4) and combine half by `quant_type`. File names come from `v2_table_filename()`; rule validation and merge-on-save reuse the shared `mori.ops.tuning_config` |
| `tuning_configs/*.json` | V2 rules in the shared rule format, one file per kernel, EP size, total experts, top-k and phase: `{arch}_{model}_{kernel}_ep{n}_expert{experts}_top{k}_{phase}.json`, e.g. `gfx950_mi355x_InterNodeV2_ep16_expert256_top6_dispatch.json`. Kernels are `InterNodeV2`, `InterNodeV2LL` and `IntraNodeV2`; generated by `--cmd tuning --tuning-save` |

## gfx125x dispatch: slot reservation

Where the slot allocator word lives in the cco window, which is the default, the
blocks of a rank reserve their slots together: each block adds its token count for
every destination, this rank included, into the rank's total for it, and the last
block to arrive reserves the total with one RMW on that destination's word and
publishes the first slot to the others. In the window the RMWs on one word
serialize, so a rank makes one per destination per call instead of one per block.
The blocks meet in a few hundred bytes of a `__device__` array in the dispatch
kernel's own module, so neither the arena nor the caller allocates anything for it.

`MORI_EP_TOKOFF_EXT=1` (default off) moves the word of a single-host EP into a separate
allocation (`TokOffExt` in `hip_backend.py`), and there every block reserves its
own slots. An EP spanning hosts always keeps the word in the window, since its
ranks cannot open each other's hipIpc handles.

A call small enough to give every block at most 8 warp work units (512 tokens on
the default 64 × 16 geometry at top-k 6) is spread over all the blocks instead of
filling the low half: block b takes units [8b, 8b + 8) on its first 8 warps. Where
the payload pass walks the token map, the warps left without a token then send all
of the metadata, so the token warps go straight to the payload. Geometries with at
most 8 warps per block, such as EP4's tuned 64 × 8, keep the plain map.

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

Internode FP4 dispatch accepts real `torch.float4_e2m1fn_x2` tensors of shape
`[tokens, hidden_dim / 2]`: each byte stores two E2M1 values, with the even
logical element in the low nibble. `hidden_dim` remains the logical width in
the config and must be a multiple of 32 for the 16-byte payload alignment.
This is transport of already packed data. The expert stage must provide an
unpacked `[received_tokens, hidden_dim]` BF16 or FP32 tensor to combine;
FP4 combine and FP4 dispatch with FP8 combine are unsupported.

For ordinary `internode_kernel="v2"`, gfx950 combine uses 4-byte-per-lane
load-first peer gathers when the declared capacity is at most 256 and the row
and staging strides are 4-byte aligned. The kernel calls the existing
`WarpAccumLF<T, 4, 2>` primitive directly, including its vector and scalar-tail
handling; other cases use `WarpAccum<T, 4>`. This selection adds no dtype,
top-k, topology, NIC-provider or QP-count restrictions beyond the operation's
existing validity checks. Architectures other than gfx950 use the ordinary
gather. These are ordinary-v2 rules; default `"auto"` still selects `v2_ll` for
calls of up to 512 tokens.

### Internode geometry: tuned by default

Leave `dispatch_block_num` / `combine_block_num` / `warp_num_per_block` /
`combine_warp_num_per_block` / `dispatch_rdma_block_num` /
`combine_rdma_block_num` at `None` and the backend resolves the swept geometry
for this GPU, shape and dtype out of `tuning_configs/`, one bucket per
token range, with plans prepared up front. A pinned config field overrides only that
field; the remaining fields stay tuned. The sweep hooks `MORI_EP_DISP_GEOM` /
`MORI_EP_COMB_GEOM` (`"block,rdma,warp"`) bypass table selection for both phases
when either is set. Clear both when checking automatic selection.

`MORI_EP_V2_TUNING_DIR` selects an alternate internode JSON directory. The shared
loader in `mori.ops.tuning_config` handles filenames, loading, lookup and
merge-on-save; `hip_tuning_configs.py` adds the HIP schedule and CU limits.

- Each phase matches **its own dtype**: fp8 dispatch + bf16 combine reads
  `fp8_e4m3` dispatch rules and the same `bf16` combine rules as bf16 dispatch.
- `num_tokens` is an inclusive ceiling; the largest matching rule also serves
  larger token counts. Dispatch/combine bucket edges are combined into a schedule,
  while a phase without matching rules keeps its default geometry.
- Total experts (`ep_size * experts_per_rank`) and top-k pick the file: at EP16,
  256/8 (`experts_per_rank=16, topk=8`) reads `expert256_top8`, while 384/6
  (`experts_per_rank=24, topk=6`) reads `expert384_top6`. Unknown combinations use
  the default geometry. Rules record `topk`, and a rule without it is not a
  wildcard in V2; the expert count is fixed by the file and not repeated per rule.
  Saving needs both values to pick the file. V1's legacy wildcard lookup is
  unchanged.
- `InterNodeV2` and `InterNodeV2LL` have separate files. In `auto` mode the family
  crossover is also a bucket edge, so each launch uses its own family's rules.

### HIP V2 configuration interface

Both intra and inter follow the same path:

1. `hip_tuning_configs.resolve_schedule(cfg)` returns a schedule with named
   `dispatch` and `combine` configurations. Intra adapts the Python-table or
   explicit `cfg.schedule`; inter resolves JSON/overrides and the V2/LL crossover.
2. `hip_backend._build_kernels()` prepares one callable per selectable
   `KernelConfig`: one plan for intra, a fixed pass group for inter.
3. `op.kernel_config(num_tokens)` selects an inclusive token bucket. Dispatch
   and combine use these same keys; the bound callables do not select again.

For configuration reporting, use the same interface on either path:

```python
selected = op.kernel_config(num_tokens)
print(selected.dispatch.family, selected.dispatch.block_num,
      selected.dispatch.warp_per_block, selected.dispatch.rdma_block_num)
print(selected.combine)
```

`family` is `"intra"`, `"v2"` or `"v2_ll"`; intra has `rdma_block_num=None`.
`EpDispatchCombineConfig.tuned(kernel_backend="hip", ...)` uses the same HIP
resolution as the normal constructor. FlyDSL keeps its existing entry points.
Intra manual geometry retains its existing fallback behavior; inter supports
per-field tuning overrides as described above.

Intra B/W participate in compilation. Inter B/R/W are plan launch settings:
different geometries can share a compiled module while retaining separate plans
for their grid, block size and bound RDMA split. Inter shape/dtype still specialize
the compiled kernel.

Run a sweep on both nodes through the existing wrapper, using rank 0 on one
node and rank 1 on the other, with the same rendezvous and tuning arguments:

```bash
# Match the current MI355X tuning campaign with ionic CCQE disabled.
MORI_DISABLE_IONIC_CCQE=1 tools/run_internode_test.sh \
    --rank <0-or-1> --master-addr <ip> --ifname <nic> \
    --entry tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py \
    --cmd tuning --kernel-type v2 --max-tokens 4096 --num-qp 2 \
    --dtype bf16 --combine-dtype bf16 --hidden-dim 7168 --topk 8 \
    -- --experts-per-rank 16 --tuning-phase dispatch --tuning-save \
       --tuning-config-dir <directory> --tuning-note "machine and comparison details"
```

Repeat for each token ceiling and phase being tuned. `--tuning-save` merges the
selected rule into the JSON directory; without it the sweep only reports results.
`--tuning-candidate block,rdma,warp` restricts the sweep to one candidate. Selection
defaults to the dispatch+combine pair's total latency (`--tuning-metric total`),
with the other phase held at its selected geometry. A bf16 combine rule is shared
by bf16, fp8 and fp4 dispatch, so validate the affected input dtypes before
replacing it, or use a separate tuning directory for a dtype-specific experiment.
Experiment results and bandwidth records live in [dispatch.md](../../../../dispatch.md) and
[bw.md](../../../../bw.md).

Tests/bench live under `tests/python/ops/dispatch_combine_v2/`:

| file | role |
|---|---|
| `test_dispatch_combine_v2_intranode.py` | pytest wrapper: runs `test_op.py` under torchrun for the representative modes and asserts every line PASS |
| `test_dispatch_combine_v2_internode.py` | the internode entry: a torchrun script (not a pytest wrapper) for correctness, bench and tuning over CCO/GDA. `--cmd test\|bench\|tuning\|stress`, `--max-tokens`, `--hidden-dim`, `--topk`, `--dtype`/`--combine-dtype`, `--num-qp` (default 1), `--kernel-type auto\|v2\|v2_ll`, `--auto-ll-max-tokens`, `--rounds`, `--spawn`. `--cmd tuning` sweeps one phase (`--tuning-phase`) against the other's shipped value and, with `--tuning-save`, merges the winner into `tuning_configs/` (`--tuning-config-dir`, `--tuning-note`). **Needs two nodes**: the op refuses a config whose node grouping disagrees with the communicator's LSA team, so one host cannot emulate it |
| `test_internode_regions.py` | pure-Python invariants of `internode_regions()`: the name contract with the backend and the capacity bounds the kernel's indexing implies. No GPU, no process group |
| `test_op_lifecycle.py` | arena-leak regression: rebuilding the op on one long-lived `Communicator`; `close()` must free and untrack the window. `torchrun --standalone --nproc_per_node=2` |
| `test_op.py` | EP8 op-layer test (gather/scatter, quant, StdMoE, recv-cap, scales, LEC, reset, replay). `MORI_V2_KERNEL_BACKEND=hip` runs it against the HIP kernels. `RAGGED=1` gives the ranks different token counts, two of them none; `NOSELF=1` routes no token to its own rank |
| `test_ep_backend_parity.py` | runs both backends in one process on the same input and compares element for element |
| `test_jit_binding.py` | JIT plan binding: schemas, request/args round-trip, cache behaviour. No GPU peers needed |
| `test_graph_capture.py` | captures dispatch → identity expert → combine as one HIP graph and replays it |
| `test_asym_dtype.py` | asymmetric dtype legs (fp8/fp4 dispatch + bf16 combine) |
| `bench_ep.py` | the perf bench, for every backend. Alternating dispatch/combine pairs, eager + CUDA graph, each point gated on an identity-expert check and non-zero exit on failure. Envs: `BACKENDS=flydsl,hip`, `MODES=eager,graph`, `SWEEP`, `ITERS`, `DISP=bf16\|fp8\|fp4`, `COMB_MODE=push\|pull` (default `push`, the fp4 combine: hip backend on gfx125x only), `COMBINE_IN=inplace\|staged`, `CHECK=0`, `DBN`/`DWPB`/`CBN`/`CWPB` to pin geometry, `HIDDEN`/`TOPK`/`EPR`, `ROUTE=rand\|ring\|noself`, `LATE_US` (rank 0 reaches every timed dispatch that many µs after the others) |

(Each script inlines a tiny torchrun/gloo `Dist` bootstrap — gloo only carries the cco unique-id and pass/fail counts.)

## Run (inside the container, 8 GPUs)

`torchrun --standalone` uses a localhost rendezvous, so no socket-iface env is
needed. These are the intranode paths (no GDA/RDMA).

```bash
cd tests/python/ops/dispatch_combine_v2

pytest test_dispatch_combine_v2_intranode.py -v                       # EP8 correctness (all modes)
pytest test_internode_regions.py -v                                   # arena layout, no GPU
torchrun --standalone --nproc_per_node=8 test_op.py                   # op-layer correctness (env-driven)
BACKENDS=flydsl,hip COMB_MODE=pull torchrun --standalone --nproc_per_node=8 bench_ep.py   # perf, both backends
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
