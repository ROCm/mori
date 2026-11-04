# HIP V2 tuning configurations

Launch-geometry tables for the HIP backend of `EpDispatchCombineOp` (V2). They
are read by `hip_tuning_configs.py`; FlyDSL has its own tables in
`tuning_configs.py`.

## File layout

One CSV per GPU model, kernel family and EP size, holding both phases:

```text
{gpu_arch}_{gpu_model}_{kernel_type}_ep{ep}.csv
e.g. gfx942_mi308x_InterNodeV2_ep16.csv
```

| `kernel_type` | Path | `internode_kernel` |
| --- | --- | --- |
| `IntraNodeV2` | single node (EP8) | — |
| `InterNodeV2` | two nodes (EP16), ordinary kernels | `v2` |
| `InterNodeV2LL` | two nodes (EP16), low-latency kernels | `v2_ll` |

Each row is one rule, a launch geometry for one shape, phase and token ceiling:

```csv
num_experts,topk,phase,dtype,hidden_dim,zero_copy,quant_type,num_tokens,block_num,rdma_block_num,warp_per_block,active_qps,combine_load_first,avg_latency_us,bandwidth_gbps,avg_rdma_bandwidth_gbps,avg_xgmi_bandwidth_gbps,avg_ll_bandwidth_gbps,bandwidth_metric
256,8,dispatch,bf16,7168,,,4,64,42,4,2,,75.12,1.52,1.52,5.1,5.63,grand_mean
256,8,combine,bf16,7168,0,none,4,16,10,4,1,1,84.85,1.35,1.35,4.5,4.97,grand_mean
```

- The first eight columns are the selector. `num_experts` is the total expert
  count, `ep * experts_per_rank`; `zero_copy` and `quant_type` are combine-only
  and empty on dispatch rows.
- `block_num`, `rdma_block_num` and `warp_per_block` are the geometry.
  Intra-node rules use `rdma_block_num` 0.
- `active_qps` and `combine_load_first` are the optional per-bucket flags below;
  empty means the default. Booleans are written 0 or 1.
- The metric columns are for reference only; lookup never reads them.
- Rows are grouped by shape (`num_experts`, `topk`, `hidden_dim`). Within a
  shape, dispatch rows come before combine rows, then by dtype, the combine
  fields and `num_tokens`. The tuning save rewrites the file in that order, so a
  diff shows only the rules that changed.
- A table is checked when it is loaded: an unknown or missing column, a cell
  that does not parse, a field the kernel family does not read, or two rows
  with the same selector raise instead of being skipped.

Internode rules (both families, both phases) may set `active_qps`: how
many of the op's allocated QPs (`num_qp_per_pe`, default 8) that bucket's kernel
sends on. Absent means `min(2, num_qp_per_pe)`; a value above the allocation is
clamped to it. A configured `cfg.active_qps` overrides every rule.
The op's dispatch drains the **largest** count among all its rules, so mixing
counts in one table makes every bucket pay the largest one's drain. On MI308X
with a largest count of 4, a bucket sending on 2 pays about 1–2 µs per
dispatch at T ≤ 64. The dispatch also drains the completions its own bucket's
combine left on QPs it did not send on, so a dispatch count below the combine's
can be slower than either count alone: the two are chosen as a pair.
MI308X inter-node rules set the count per bucket and phase (1, 2 or 4), except
the FP8 combine rows. Every MI300X inter-node rule sets one (1, 2 or 4). MI355X
rules set one (1, 2 or 4) per bucket and phase where the whole-table A/B kept a
change: 119 of the 233 InterNodeV2 rules and 42 of the 233 InterNodeV2LL ones.
The largest count in one MI355X op is 4, and 2 for InterNodeV2LL 272/9 and
896/16.

`InterNodeV2` combine rules may set `combine_load_first` to 1 (empty means 0).
That bucket's combine pass is then compiled with the load-first peer gather
(`WarpAccumLF`). It is chosen per bucket, not from the declared capacity, and
other tables may not set it. Current settings:

- MI355X: chosen per bucket together with the active QP counts; a bucket the
  whole-table A/B did not change keeps its previous setting (the earlier
  capacity ≤ 256 rule). On for 256/8 bf16 H7168 and H6144 and 272/9 bf16
  H7168 at T4–256; 384/6 bf16 H5120 at T4–256 and T4096; 256/6 bf16 H7168 and
  384/6 bf16 H7168 at T64–256 and T4096; 896/16 bf16 H3584 at T8, T32 and
  T4096. Off elsewhere.
- MI308X: chosen per bucket together with the active QP counts, and kept only
  where a whole-table A/B against the previous tables confirmed it: on for
  256/8 bf16 H7168 at T4–512, T2048 and T4096; 256/8 bf16 H6144 at T4–128 and
  T2048; 256/6 bf16 H7168 at T64–128; 384/6 bf16 H7168 at T64–256; 384/6
  bf16 H5120, re-tuned with its geometry, at T4–256, T2048 and T4096.
  272/9 and 896/16, re-tuned on the kernel with the topk-9/16 combine gather
  fixed: on for 272/9 bf16 H7168 at T4–256, T1024 and T2048, and for 896/16
  bf16 H3584 at T16, T32, T256 and T512.
  Off elsewhere.
- MI300X: chosen per bucket together with the active QP counts, and kept only
  where a whole-table A/B against the previous tables confirmed it: on for
  256/8 bf16 H7168 and H6144 and 384/6 bf16 H5120 at T4–256; 272/9 bf16 H7168
  at T4–128; 256/6 and 384/6 bf16 H7168 at T64–256; 896/16 bf16 H3584 at T16,
  T32 and T256. Off elsewhere.

## Lookup

- **Only the current model's own file is read.** A model without a file, such
  as MI350X or MI325X today, does not borrow another model's CSV; it falls
  back to the default geometry (inter-node: the config defaults clamped to the
  CU count; intra-node: the Python tables in `hip_tuning_configs.py`, where
  MI350X shares MI355X's entries, then `_hip_default()`).
- `num_experts`, `topk`, `dtype` and `hidden_dim` must match exactly. Dispatch
  and combine are matched independently, each with its own dtype.
- FP4 dispatch rows store the packed width: `hidden_dim` 3584 means a logical
  hidden size of 7168.
- `num_tokens` is an inclusive ceiling. A batch uses the first rule whose
  ceiling is ≥ its token count; batches above the last ceiling use the last rule.
- Geometry from a table is clamped to the GPU's CU count, keeping
  `rdma_block_num < block_num`. Fields set explicitly on the config override
  the table, and are not clamped.
- `MORI_EP_V2_TUNING_DIR` points lookup at another directory.
  `MORI_EP_DISP_GEOM` / `MORI_EP_COMB_GEOM` (`B,R,W[,active_qps]`) bypass the
  tables entirely, for sweeps.

## Coverage

| GPU (arch, CUs) | Shapes (experts / top-k) | Hidden | Dispatch dtypes |
| --- | --- | --- | --- |
| MI355X (gfx950, 256) | 256/8, 256/6, 384/6; 272/9, 896/16 inter | 7168; 6144 for inter 256/8; 5120 for 384/6; 3584 for 896/16 | bf16, fp8_e4m3; fp4 for inter 256/8 |
| MI308X (gfx942, 80) | 256/8, 256/6, 384/6; 264/9 intra; 272/9, 896/16 inter | 7168; 6144 for 256/8; 5120 for 384/6; 3584 for 896/16 | bf16, fp8_e4m3_fnuz; fp4 for inter 256/8 |
| MI300X (gfx942, 304) | 256/8, 256/6, 384/6; 264/9 intra; 272/9, 896/16 inter | 7168; 6144 for 256/8; 5120 for 384/6; 3584 for 896/16 | bf16, fp8_e4m3_fnuz; fp4 for inter 256/8 |

Each GPU model has IntraNodeV2 (EP8), InterNodeV2 (EP16) and InterNodeV2LL
(EP16) tables, and every shape in a table has both phases. Not every shape is
in every table, and not every dtype appears in every shape; see the files.
Token ceilings are 64–4096, plus 4–32 for MI308X, MI355X and MI300X
inter-node 256/8, 272/9, 896/16 and 384/6 at hidden 5120.

The model behind each shape:

| Model | Experts / top-k | Hidden the dispatch carries | Tables |
| --- | --- | --- | --- |
| DeepSeek-V3/R1 | 256/8 | 7168 | MI355X, MI308X, MI300X: EP8 and EP16; FP4 dispatch (stored as 3584, packed) on MI308X, MI355X and MI300X EP16 |
| DeepSeek-V3/R1, shared expert fused in | 264/9 at EP8, 272/9 at EP16 | 7168 | MI308X, MI300X; MI355X: EP16 |
| GLM-5, GLM-5.3 | 256/8 | 6144 | MI308X, MI300X: EP8 and EP16; MI355X: EP16 |
| DeepSeek-V4-Pro | 384/6 | 7168 | MI355X, MI308X, MI300X: EP8 and EP16 |
| DeepSeek-V4.1-Flash | 384/6 | 5120 | MI355X, MI308X, MI300X: EP8 and EP16 |
| Kimi K3 | 896/16 | 3584 | MI308X, MI355X, MI300X: EP16 |
| none known | 256/6 | 7168 | MI355X, MI308X, MI300X: EP8 and EP16; bf16 dispatch only |

A shape without rows runs on the default geometry: for example DeepSeek-V4-Flash
(256/6 at hidden 4096), the DSpark draft layers of DeepSeek-V4.1-Flash (128/3 at
hidden 5120), GLM-5.3 with its shared expert fused in (264/9 and 272/9 at
hidden 6144), and every shape on a GPU or EP size its table lacks.

The 264/9 and 272/9 rows are DeepSeek-V3/R1 with the shared expert fused in
as a routed one, laid out as ATOM's
`SharedExpertMode.LOCAL_REPLICA` does: every rank holds its 256/EP routed
experts plus one replica of the shared expert in its last slot, and each
token's ninth expert is its own rank's replica. The expert count depends on the
EP size, so 264/9 (33 a rank) is only in the EP8 table and 272/9 (17 a rank)
only in the EP16 tables.

896/16 is Kimi K3. Its routed experts run on a 3584-wide latent (the
model projects the 7168-wide hidden state down before the dispatch and back
up after the combine), so 3584 is the width the dispatch carries.

FP8 uses the arch's native encoding: `fp8_e4m3_fnuz` on gfx942 and
`fp8_e4m3` (OCP) on gfx950. Combine rows are BF16, except inter-node 256/8 on
MI308X, MI355X and MI300X, which also has FP8 combine rows for symmetric FP8
runs.

## Measurement conditions

Metrics are only comparable between rows measured under the same conditions.

- **QP count.** Inter-node latency depends heavily on the active count
  (`active_qps`, above) and on the allocation (`num_qp_per_pe`) on MI308X:
  QP64 adds about 270 µs per phase at T ≤ 512. MI308X inter-node rows were
  measured with an allocation of 8 and each rule's own count, except FP8
  combine rows, which keep earlier two-QP measurements. MI300X inter-node
  rows likewise, FP8 combine rows included, with no count above 4. MI355X
  inter-node rows changed on 2026-11-02/03 were measured the same way; the ones
  it left unchanged keep the earlier harness-sweep measurements (an allocation
  of 2, `scale_dim` 32), whose metrics are not comparable.
- **Protocol.** Uniform routing, `scale_dim=0`, capacity equal to the token
  count, `quant_type="none"`. The 264/9 and 272/9 rows use replica routing
  (`--routing replica`): eight distinct routed experts drawn uniformly, plus
  the token's own rank's replica. FP8 and FP4 dispatch rows are measured with
  the bucket's BF16 combine rule; MI300X FP8 combine rows in a symmetric FP8
  op, with the bucket's FP8 dispatch rule. For FP4 the combine reduces the
  decoded payload, decoded between the two timed phases where FP8 is cast to
  BF16.
  Timing uses 3 passes × (20 warmup + 30 timed rounds), dropping the first
  timed round of each pass. A measurement is accepted only after correctness
  checks, and only if each phase has
  mean/median ≤ 1.10 and max/median ≤ 5, and the per-pass D+C means vary by
  ≤ 1.20×.
- **MI300X EP8.** Every rule was re-tuned and measured with the protocol above
  on one MI300X, each rank pinned to four physical cores on its GPU's NUMA
  node. Unpinned, the T ≤ 128 cells settle per run into a regime up to 1.5×
  slower, which more rounds do not average out. Each phase was screened over
  blocks 8–304 × warps 4, 8 and 16 against the previous table, two rounds,
  and a bucket took the new rules only where they were ≥1% faster (D+C summed
  over the dispatch dtypes) in every round of a whole-table A/B and again in
  a confirming one. The JIT cache must be warm before timing: a cell measured
  right after its kernels compile starts from idle-dropped clocks and reads up
  to 40% slow at T64.
- **MI300X EP16.** Every rule was re-tuned and measured with the protocol
  above on two MI300X nodes (ConnectX-7, 400 Gb/s a rail), ranks pinned as
  for EP8. Geometry was screened per family and phase in two rounds against
  the previous tables, or the defaults for a shape without rows; the active
  QP counts and the V2 combine load-first then in one round of whole-table
  variants. A bucket took the new rules only where they were ≥1% faster (D+C
  summed over the dispatch dtypes) in every round of a whole-table A/B against
  the previous tables, and again in a confirming one. The FP4 dispatch and FP8
  combine rows were added later, geometry and active QP count screened
  together against the defaults, with the same rounds and gates.
- **MI355X EP16.** Re-tuned on 2026-11-02/03 on two MI355X nodes (ionic
  NICs, `MORI_DISABLE_IONIC_CCQE=1`) with the protocol above, on the kernel
  with the topk-9/16 combine gather and copy tail fixed. Per bucket: geometry
  screened over 37 candidates per phase in two rounds, then the active QP
  counts and load-first, then a whole-table A/B against the previous tables
  and a confirming one; a bucket took the new rules only where they were ≥1%
  faster (D+C summed over the dispatch dtypes) in every round. Rows for a
  bucket that had none and was not changed hold the configuration it ran
  before (the next bucket's rule or the default), so the new rules of its
  neighbours do not move it. On 896/16 InterNodeV2LL dispatch, 256 blocks ×
  16 warps (256/128/16, 256/170/16) gave wrong results on 1 rank in 16 in 4
  of 54 runs; those geometries are excluded there, and its T2048 and T4096
  buckets keep the default until they are re-measured.
- **Kernel.** The MI308X 264/9, 272/9 InterNodeV2, 896/16 and FP4 dispatch
  rows, and every MI300X row, were re-tuned and measured on the kernel
  with the topk-9/16 combine gather and the 3584 B copy tail fixed
  (`device_primitives.hpp`). The 272/9 InterNodeV2LL rows predate the fix,
  which moves them by under 0.5%.
- **Metrics.** `avg_latency_us` is the mean over all retained samples.
  Bandwidths are grand means of per-rank, per-round `bytes / time`, not
  `bytes / mean latency`. `bandwidth_gbps` is XGMI for intra-node rows, RDMA for
  `InterNodeV2` and the LL metric for `InterNodeV2LL`. These metrics are not
  interchangeable between families.

## Adding or updating rules

Tune with `tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py`
(`--cmd tuning --tuning-save`). Saving goes through
`hip_tuning_configs.save_internode_result()`, which replaces the rule with the
same selector in the model's table for that kernel family and EP size, or adds
it.

- Record the allocation (`num_qp_per_pe`) with the results. A per-bucket
  `active_qps` changes the drain of every bucket in the op, so accept a new
  count only from a whole-table A/B, not from the single-bucket sweep.
- Never copy rows from another GPU model or another shape. A missing row is
  safer than a borrowed one: on MI308X, before the topk-9 combine fix, the
  256/8 rules copied whole to 264/9 and 272/9 were up to a third slower than
  the defaults in some buckets (InterNodeV2 T64 and T512, IntraNodeV2 T4096).
- Store the geometry that actually ran: keep `block_num` within the CU count.
- `--cmd tuning` measures candidates with load-first off and rewrites the whole
  rule, so re-saving a combine rule clears `combine_load_first`. Re-measure with
  load-first on (A/B the table with and without the flag) before setting it again.
