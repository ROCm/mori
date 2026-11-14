# HIP V2 tuning configurations

These tables set the launch geometry for the HIP backend of
`EpDispatchCombineOp` (V2). `hip_tuning_configs.py` reads them. FlyDSL has its
own tables in `tuning_configs.py`.

## File layout

There is one CSV per GPU model, kernel family and EP size. Each file holds both
phases.

```text
{gpu_arch}_{gpu_model}_{kernel_type}_ep{ep}.csv
e.g. gfx942_mi308x_InterNodeV2_ep16.csv
```

| `kernel_type` | Path | `internode_kernel` |
| --- | --- | --- |
| `IntraNodeV2` | single node (EP8) | — |
| `InterNodeV2` | two nodes (EP16), ordinary kernels | `v2` |
| `InterNodeV2LL` | two nodes (EP16), low-latency kernels | `v2_ll` |

Each row is one rule: the launch geometry for one shape, phase and token
ceiling.

```csv
num_experts,topk,phase,dtype,hidden_dim,num_tokens,block_num,rdma_block_num,warp_per_block,active_qps,combine_load_first,avg_latency_us,bandwidth_gbps,avg_rdma_bandwidth_gbps,avg_xgmi_bandwidth_gbps,avg_ll_bandwidth_gbps,bandwidth_metric
256,8,dispatch,bf16,7168,4,64,42,4,2,,75.12,1.52,1.52,5.1,5.63,grand_mean
256,8,combine,bf16,7168,4,16,10,4,1,1,84.85,1.35,1.35,4.5,4.97,grand_mean
```

- The first six columns select the rule.
- `num_experts` is the total expert count: `ep * experts_per_rank`.
- There is no zero-copy or quant-type column. V2 has no zero-copy mode, and
  its tables serve only `quant_type="none"`.
- `block_num`, `rdma_block_num` and `warp_per_block` are the geometry.
  Intra-node rules use `rdma_block_num` 0.
- `active_qps` and `combine_load_first` are optional flags, described below.
  An empty cell means the default. Booleans are written 0 or 1.
- The metric columns are for reference. Lookup never reads them.
- Rows are sorted by shape (`num_experts`, `topk`, `hidden_dim`). Within a
  shape, dispatch rows come first, then combine rows. Each phase is sorted by
  dtype, then by `num_tokens`.
- The tuning save rewrites the whole file in this order, so a diff shows only
  the rules that changed.
- Each table is checked when it loads. These problems raise an error instead
  of being skipped:
  - an unknown or missing column;
  - a cell that does not parse;
  - a field the kernel family does not read;
  - two rows with the same selector.

### `active_qps`

Inter-node rules (both families, both phases) may set `active_qps`. It is the
number of the op's allocated QPs that the bucket's kernel sends on.

- The allocation is `num_qp_per_pe`, 8 by default.
- An empty cell means `min(2, num_qp_per_pe)`.
- A value above the allocation is clamped to it.
- `cfg.active_qps`, when set, overrides every rule.

Mixing counts in one table has a cost:

- The op's dispatch drains the **largest** count among the rules its token
  capacity can reach. This includes both phases, plus any count compiled for
  `op.set_active_qps()`. Every reachable bucket pays for that drain.
- For example, on MI308X with a largest count of 4, a bucket that sends on 2
  pays about 1–2 µs per dispatch at T ≤ 64.
- The dispatch also drains completions that its bucket's combine left on QPs
  the dispatch did not use. So a dispatch count below the combine's count can
  be slower than either count alone. Choose the two counts as a pair.

Current settings:

- MI308X: every inter-node rule sets a count (1, 2 or 4) per bucket and phase,
  except the FP8 combine rows.
- MI300X: every inter-node rule sets a count (1, 2 or 4).
- MI355X: a rule sets a count (1, 2 or 4) only where the whole-table A/B kept a
  change. That is 121 of the 233 InterNodeV2 rules and 45 of the 233
  InterNodeV2LL rules. Over all token counts, the largest count for one shape
  is 4, or 2 for InterNodeV2LL 272/9 and 896/16. Under `auto` with its default
  crossover of 512, every rule up to T512 comes from InterNodeV2LL and sets at
  most 2, so an op whose capacity is at most 512 drains at most 2, unless
  `cfg.active_qps` or `cfg.active_qp_counts` names a larger count.

### `combine_load_first`

`InterNodeV2` combine rules may set `combine_load_first` to 1. An empty cell
means 0.

- With 1, that bucket's combine pass is compiled with the load-first peer
  gather (`WarpAccumLF`).
- It is set per bucket. The declared capacity does not choose it.
- Other tables may not set it.
- It has no effect at a top-k that the load-first gather has no case for, such
  as 16. Both gathers then fall back to the same `WarpAccumDynamic`. So no
  896/16 rule sets it.

Where it is on today (it is off everywhere else):

- MI355X: chosen per bucket, together with the active QP counts. A bucket that
  the whole-table A/B did not change keeps its earlier setting, from the old
  "capacity ≤ 256" rule.
  - 256/8 bf16 H7168 and H6144, and 272/9 bf16 H7168: T4–256.
  - 384/6 bf16 H5120: T4–256 and T4096.
  - 256/6 bf16 H7168 and 384/6 bf16 H7168: T64–256 and T4096.
- MI308X: chosen per bucket, together with the active QP counts. Kept only
  where a whole-table A/B against the previous tables confirmed it.
  - 256/8 bf16 H7168: T4–512, T2048 and T4096.
  - 256/8 bf16 H6144: T4–128 and T2048.
  - 256/6 bf16 H7168: T64–128.
  - 384/6 bf16 H7168: T64–256.
  - 384/6 bf16 H5120, re-tuned together with its geometry: T4–256, T2048 and
    T4096.
  - 272/9 bf16 H7168, re-tuned on the kernel with the topk-9/16 combine gather
    fixed: T4–256, T1024 and T2048.
- MI300X: chosen per bucket, together with the active QP counts. Kept only
  where a whole-table A/B against the previous tables confirmed it.
  - 256/8 bf16 H7168 and H6144, and 384/6 bf16 H5120: T4–256.
  - 272/9 bf16 H7168: T4–128.
  - 256/6 and 384/6 bf16 H7168: T64–256.

## Lookup

- **Only the current model's own file is read.** The model comes from the PCI
  device ID. MI350X reads MI355X's file (`_MODEL_ALIAS`), because it has the
  same die and CU count. A gfx950 GPU whose device ID mori does not list is
  also treated as MI355X.
- A model without a file, such as MI325X today, does not borrow another
  model's CSV. It falls back to the default geometry:
  - inter-node: the config defaults, clamped to the CU count;
  - intra-node: the Python tables in `hip_tuning_configs.py`, then
    `_hip_default()`.
- `num_experts`, `topk`, `dtype` and `hidden_dim` must match exactly.
- Dispatch and combine are looked up separately, each with its own dtype.
- FP4 dispatch rows store the packed width: `hidden_dim` 3584 means a logical
  hidden size of 7168.
- `num_tokens` is an inclusive ceiling. A batch uses the first rule whose
  ceiling is ≥ its token count. A batch above the last ceiling uses the last
  rule.
- Geometry from a table is clamped to the GPU's CU count, and `rdma_block_num`
  stays below `block_num`.
- On the inter-node path, a geometry field set on the config replaces that
  field in every bucket, and the fields left unset stay tuned. A set field is
  not clamped to the CU count, but `rdma_block_num` is still kept below
  `block_num` whenever `block_num` > 1.
- On the intra-node path, setting any block or warp field, or `schedule`,
  turns the table off for the op. A set `schedule` runs as given. Otherwise
  the fields left unset take fixed defaults, not `_hip_default()`'s: dispatch
  64 blocks × 16 warps, combine 80 × 4. Nothing on this path is clamped to the
  CU count.
- `MORI_EP_V2_TUNING_DIR` makes lookup read another directory.
- `MORI_EP_DISP_GEOM` and `MORI_EP_COMB_GEOM` (`B,R,W[,active_qps]`) replace one
  inter-node phase's geometry in every bucket. They are meant for sweeps. The
  other phase keeps its rules.
  - Without the fourth field, the phase sends on `cfg.active_qps`, else
    `min(2, num_qp_per_pe)`, not the rule's count.
  - A combine override also turns load-first off in every bucket.
  - Intra-node ops do not read them.

## Coverage

| GPU (arch, CUs) | Shapes (experts / top-k) | Hidden | Dispatch dtypes |
| --- | --- | --- | --- |
| MI355X (gfx950, 256) | 256/8, 256/6, 384/6; 264/9 intra; 272/9, 896/16 inter | 7168, plus 6144 for 256/8 and 5120 for 384/6; only 3584 for 896/16 | bf16, fp8_e4m3; fp4 for inter 256/8 at hidden 7168 |
| MI308X (gfx942, 80) | 256/8, 256/6, 384/6; 264/9 intra; 272/9, 896/16 inter | 7168, plus 6144 for 256/8 and 5120 for 384/6; only 3584 for 896/16 | bf16, fp8_e4m3_fnuz; fp4 for inter 256/8 at hidden 7168 |
| MI300X (gfx942, 304) | 256/8, 256/6, 384/6; 264/9 intra; 272/9 inter; 896/16 intra and inter | 7168, plus 6144 for 256/8 and 5120 for 384/6; only 3584 for 896/16 | bf16, fp8_e4m3_fnuz; fp4 for inter 256/8 at hidden 7168 |

- Each GPU model has three tables: IntraNodeV2 (EP8), InterNodeV2 (EP16) and
  InterNodeV2LL (EP16).
- Every shape in a table has both phases.
- Not every shape is in every table, and not every dtype is in every shape.
  See the files.
- Token ceilings are 64–4096. MI308X, MI355X and MI300X also have ceilings of
  4–32 for inter-node 256/8, 272/9, 896/16, and 384/6 at hidden 5120. The
  MI300X IntraNodeV2 table also has ceilings of 16, 32, 8192 and 16384 for every
  shape.

The model behind each shape:

| Model | Experts / top-k | Hidden the dispatch carries | Tables |
| --- | --- | --- | --- |
| DeepSeek-V3/R1 | 256/8 | 7168 | MI355X, MI308X, MI300X: EP8 and EP16; FP4 dispatch (stored as 3584, packed) on MI308X, MI355X and MI300X EP16 |
| DeepSeek-V3/R1, shared expert fused in | 264/9 at EP8, 272/9 at EP16 | 7168 | MI355X, MI308X, MI300X: EP8 and EP16 |
| GLM-5, GLM-5.3 | 256/8 | 6144 | MI355X, MI308X, MI300X: EP8 and EP16 |
| DeepSeek-V4-Pro | 384/6 | 7168 | MI355X, MI308X, MI300X: EP8 and EP16 |
| DeepSeek-V4.1-Flash | 384/6 | 5120 | MI355X, MI308X, MI300X: EP8 and EP16 |
| Kimi K3 | 896/16 | 3584 | MI308X, MI355X: EP16; MI300X: EP8 and EP16 |
| none known | 256/6 | 7168 | MI355X, MI308X, MI300X: EP8 and EP16; bf16 dispatch only, except MI300X EP8, which also has fp8_e4m3_fnuz |

A shape without rows runs on the default geometry. For example:

- DeepSeek-V4-Flash (256/6 at hidden 4096);
- the DSpark draft layers of DeepSeek-V4.1-Flash (128/3 at hidden 5120);
- GLM-5.3 with its shared expert fused in (264/9 and 272/9 at hidden 6144);
- any shape on a GPU or EP size whose table does not have it.

264/9 and 272/9 are DeepSeek-V3/R1 with the shared expert fused in as a routed
expert. The layout follows ATOM's `SharedExpertMode.LOCAL_REPLICA`:

- Every rank holds its 256/EP routed experts, plus one replica of the shared
  expert in its last slot.
- Each token's ninth expert is the replica on the token's own rank.
- The expert count depends on the EP size. So 264/9 (33 per rank) is only in
  the EP8 table, and 272/9 (17 per rank) is only in the EP16 tables.

896/16 is Kimi K3. Its routed experts run on a 3584-wide latent: the model
projects the 7168-wide hidden state down before the dispatch, and back up after
the combine. So the dispatch carries 3584.

FP8 uses the arch's native encoding: `fp8_e4m3_fnuz` on gfx942 and `fp8_e4m3`
(OCP) on gfx950.

Combine rows are BF16. The exception is inter-node 256/8 on MI308X, MI355X and
MI300X, which also has FP8 combine rows for symmetric FP8 runs.

## Measurement conditions

Metrics are only comparable between rows measured under the same conditions.
This section describes how the shipped rows were measured.

The shipped rows were measured by campaign drivers built on the harness's
timing and metrics. `--cmd tuning` does only the timing and a single-bucket
sweep. The drivers also did the acceptance checks, the whole-table A/B and the
pinning. When you tune by hand, these are separate steps; see
[Adding or updating rules](#adding-or-updating-rules).

**QPs**

- Inter-node rows use an allocation of 8 (`num_qp_per_pe=8`) and each rule's
  own active count.
- Inter-node latency depends heavily on both. For example, on MI308X an
  allocation of 64 adds about 270 µs per phase at T ≤ 512. A run with another
  allocation does not reproduce the rows.
- Two groups keep older conditions and are not comparable with the rest:
  - the MI308X FP8 combine rows (two QPs);
  - the MI355X rows that its last re-tune left unchanged (an allocation of 2,
    `scale_dim` 32).

**Workload**

- Uniform routing, `scale_dim=0`, capacity equal to the token count, and
  `quant_type="none"`.
- The 264/9 and 272/9 rows use replica routing (`--routing replica`): eight
  distinct routed experts drawn uniformly, plus the token's own rank's replica.
- FP8 and FP4 dispatch rows are measured with the bucket's BF16 combine rule.
- FP8 combine rows are measured in a symmetric FP8 op, with the bucket's FP8
  dispatch rule.
- For FP4, combine reduces the decoded payload. The decode runs between the
  two timed phases, where FP8 would be cast to BF16.
- The harness's `--dtype fp4` reproduces the FP4 rows:
  - it rounds its BF16 random draw to e2m1 and packs it, so the routing matches
    the BF16 run;
  - it combines in BF16 unless `--combine-dtype` says otherwise;
  - `--tuning-save` writes the packed width.

**Timing**

- 3 passes × (20 warmup + 30 timed rounds). The first timed round of each pass
  is dropped.
- These are the harness's defaults: `--tuning-reps 3 --warmup 20 --rounds 30
  --drop-rounds 1`.
- Warm the JIT cache first. On MI300X, a cell timed right after its kernels
  compile starts with idle-dropped clocks and reads up to 40% slow at T64.
- The MI300X rows were measured with each rank pinned to four physical cores on
  its GPU's NUMA node (`os.sched_setaffinity` in each spawned rank). Without
  pinning, each run of a T ≤ 128 cell there settles into one regime, which can
  be up to 1.5× slower.
- The harness does not pin.

**Acceptance**

A measurement counts only if:

- the correctness checks pass;
- each phase has mean/median ≤ 1.10 and max/median ≤ 5;
- the per-pass D+C means differ by at most 1.20×.

`--cmd tuning --tuning-save` checks none of this. It measures the selected rule
and writes the metrics.

**Selection**

How the shipped rows were chosen:

1. Screen the geometry per family and phase, against the current tables, or
   against the defaults for a shape without rows.
2. Then screen the active QP counts and the combine load-first.
3. Run a whole-table A/B against the previous tables. A bucket takes the new
   rules only if they are ≥1% faster in every A/B pair (D+C summed over the
   dispatch dtypes).
4. Repeat the whole-table A/B to confirm.

What `--cmd tuning` does on its own:

- It sweeps one bucket and one phase against the shipped rule, which stays
  fixed. With `--tuning-greedy`, each win becomes the incumbent for the
  candidates after it, and the last win is selected.
- A geometry's tail is the median over reps of its slowest round in each rep,
  under the selection metric: by default the slowest dispatch round plus the
  slowest combine round, each across ranks.
- A candidate wins when both hold:
  - its paired median improvement is larger than both 1.5 µs and 2%
    (`--tuning-margin-us`, `--tuning-margin-frac`);
  - its tail is slower than the shipped rule's tail by no more than 10% of the
    shipped median (`--tuning-tail-frac`).
- Without `--tuning-greedy`, when no candidate wins, a candidate whose paired
  median is within that margin and whose tail is better than the shipped
  rule's by more than 10% of the shipped median is reported as `TAIL` and
  selected instead. `--tuning-save` writes it as it would write a win.
- Its combine reads the dispatch output, so the kernel first copies it into
  the staging rows. A caller that writes the expert output into
  `combine_in_view()`, as `bench_ep.py` does by default, skips that copy. The
  two can rank combine geometries differently.

**Metrics**

- `avg_latency_us` is the mean of all retained samples.
- Bandwidths are grand means of per-rank, per-round `bytes / time`. They are
  not `bytes / mean latency`.
- `bandwidth_gbps` means a different thing in each family:
  - intra-node rows: XGMI;
  - `InterNodeV2`: RDMA;
  - `InterNodeV2LL`: the LL metric.
- So these metrics are not comparable between families.
- Some MI300X IntraNodeV2 rows were timed inside one HIP graph, without host
  launch overhead, so their latencies read lower than the other rows'.

## Adding or updating rules

Tune with
`tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py`
(`--cmd tuning`). Inter-node rules are tuned on both nodes. Start from the
example command in [../README.md](../README.md). It sets what the defaults get
wrong:

- `--num-qp 8` (the wrapper defaults to 2 QPs);
- `--scale-dim 0` (the entry defaults to 32);
- a long `MORI_INTERNODE_TIMEOUT` (the wrapper kills the run after 120 s).

IntraNodeV2 rules are tuned on one node, where the entry builds the intra-node
op. The wrapper always starts two nodes, so launch the entry with
`torchrun --standalone --nproc_per_node=1` (it spawns the eight ranks), still
with `--scale-dim 0`.

The 264/9 and 272/9 rows also need `--routing replica`.

`--tuning-save` writes through `hip_tuning_configs.save_internode_result()`.
It replaces the rule with the same selector in the model's table for that
kernel family and EP size, or adds a new rule.

`--tuning-save` does not check the winner. Before you keep one:

1. **Confirm it head-to-head.** Run with
   `--tuning-candidate block,rdma,warp[,active_qps]` and a larger
   `--tuning-reps` than the sweep used (the default is 3). This measures only
   that candidate against the shipped rule, in `--tuning-reps` paired passes.
2. **Check correctness.** Run `--cmd test` at that token count, with
   `MORI_EP_V2_TUNING_DIR` pointing at the directory with the new rule.
3. **Confirm the gain on the whole table.** `--cmd bench` needs two nodes, so
   this step applies to inter-node rules. Run it over the affected token counts
   and dtypes. Alternate between the old table and the new one with
   `MORI_EP_V2_TUNING_DIR`, for several pairs. Keep the rule only if the D+C
   total is ≥1% lower in every pair. This matters most at large token counts:
   on MI308X at T1024, the sweep reported wins of 100–175 µs that tied with the
   old rule in alternating runs.

Also:

- Record the allocation (`num_qp_per_pe`) with the results.
- A per-bucket `active_qps` changes the drain of every bucket in the op.
  Accept a new count only from a whole-table A/B, not from the single-bucket
  sweep.
- Never copy rows from another GPU model or another shape. A missing row is
  safer than a borrowed one. On MI308X, before the topk-9 combine fix, 256/8
  rules copied to 264/9 and 272/9 were up to a third slower than the defaults
  in some buckets (InterNodeV2 T64 and T512, IntraNodeV2 T4096).
- Store the geometry that actually ran: keep `block_num` within the CU count.
- `--cmd tuning` measures with load-first off, both for the candidates and for
  the shipped rule it compares them with. It also rewrites the whole rule, so
  re-saving a combine rule clears `combine_load_first`. Before you set the flag
  again, re-measure with it on: A/B the table with and without it.
