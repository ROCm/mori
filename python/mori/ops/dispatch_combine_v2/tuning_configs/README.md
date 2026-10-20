# HIP V2 tuning configurations

Launch-geometry tables for the HIP backend of `EpDispatchCombineOp` (V2). They
are read by `hip_tuning_configs.py`; FlyDSL has its own tables in
`tuning_configs.py`.

## File layout

One file per GPU model, kernel family, EP size, total expert count, top-k and
phase:

```text
{gpu_arch}_{gpu_model}_{kernel_type}_ep{ep}_expert{experts}_top{topk}_{phase}.json
e.g. gfx942_mi308x_InterNodeV2_ep16_expert384_top6_dispatch.json
```

| `kernel_type` | Path | `internode_kernel` |
| --- | --- | --- |
| `IntraNodeV2` | single node (EP8) | — |
| `InterNodeV2` | two nodes (EP16), ordinary kernels | `v2` |
| `InterNodeV2LL` | two nodes (EP16), low-latency kernels | `v2_ll` |

`experts` is the total expert count, `ep * experts_per_rank`.

Each rule holds one launch geometry for a dtype, hidden dimension and token
ceiling:

```json
{
  "dtype": "bf16", "num_tokens": 512, "hidden_dim": 7168, "topk": 6,
  "block_num": 80, "rdma_block_num": 64, "warp_per_block": 8,
  "bandwidth_gbps": 36.05, "avg_rdma_bandwidth_gbps": 36.05,
  "avg_xgmi_bandwidth_gbps": 94.53, "avg_ll_bandwidth_gbps": 110.8,
  "avg_latency_us": 412.26, "bandwidth_metric": "grand_mean"
}
```

Combine rules also carry `"zero_copy": false` and `"quant_type": "none"`.
Intra-node rules use `rdma_block_num: 0`. The metric fields are for reference
only; lookup reads just the selector, the geometry and the optional flag below.

`InterNodeV2` combine rules may set `"combine_load_first": true` (absent means
false). That bucket's combine pass is then compiled with the load-first peer
gather (`WarpAccumLF`). It is chosen per bucket, not from the declared capacity,
and is ignored by `InterNodeV2LL` and intra-node tables. Current settings:

- MI355X: on for T ≤ 256 in every ordinary-V2 combine table (the previous
  capacity ≤ 256 rule, measured with capacity equal to the token count).
- MI308X: on for 256/8 bf16 H7168 at T64–512, T2048 and T4096, and 256/6 bf16
  H7168 at T64, where an interleaved A/B measured it faster (QP2). Elsewhere it
  measured neutral or slower (256/6 T256 and T1024, 256/8 T1024) or is unmeasured.

## Lookup

- **Only the current model's own file is read.** A model without a file, such
  as MI350X or MI325X today, does not borrow another model's table; it falls
  back to the default geometry (inter-node: the config defaults clamped to the
  CU count; intra-node: the Python tables in `hip_tuning_configs.py`, then
  `_hip_default()`).
- `dtype`, `hidden_dim` and `topk` must match exactly. Dispatch and combine are
  matched independently, each with its own dtype.
- FP4 dispatch rows store the packed width: `hidden_dim` 3584 means a logical
  hidden size of 7168.
- `num_tokens` is an inclusive ceiling. A batch uses the first rule whose
  ceiling is ≥ its token count; batches above the last ceiling use the last rule.
- Geometry from a table is clamped to the GPU's CU count, keeping
  `rdma_block_num < block_num`. Fields set explicitly on the config override
  the table, and are not clamped.
- `MORI_EP_V2_TUNING_DIR` points lookup at another directory.
  `MORI_EP_DISP_GEOM` / `MORI_EP_COMB_GEOM` (`B,R,W`) bypass the tables
  entirely, for sweeps.

## Coverage

| GPU (arch, CUs) | Shapes (experts / top-k) | Hidden | Dispatch dtypes |
| --- | --- | --- | --- |
| MI355X (gfx950, 256) | 256/8, 256/6, 384/6 | 7168; 5120 for 384/6 | bf16, fp8_e4m3 |
| MI308X (gfx942, 80) | 256/8, 256/6, 384/6 | 7168; 6144 for inter 256/8; 5120 for 384/6 | bf16, fp8_e4m3_fnuz; fp4 for inter 256/8 |
| MI300X (gfx942, 304) | 256/8, 384/6 | 7168; 6144 for 256/8 | bf16, fp8_e4m3_fnuz |

Each shape has IntraNodeV2 (EP8), InterNodeV2 (EP16) and InterNodeV2LL (EP16)
tables for both phases. Not every dtype appears in every shape; see the files.
Token ceilings are 64–4096, plus 4–32 for MI308X inter-node 256/8.

FP8 uses the arch's native encoding: `fp8_e4m3_fnuz` on gfx942 and
`fp8_e4m3` (OCP) on gfx950. Combine rows are BF16, except MI308X inter-node
256/8, which also has FP8 combine rows for symmetric FP8 runs.

Known gap: MI308X InterNodeV2LL 384/6 at hidden 5120 has no T4096 rows, so
batches above 2048 tokens use the T2048 rule.

## Measurement conditions

Metrics are only comparable between rows measured under the same conditions.

- **QP count.** The tables have no QP selector, but inter-node latency depends
  heavily on it on MI308X. QP64 adds about 270 µs per phase at T ≤ 512. QP2
  (the default `num_qp_per_pe`) shows a heavy long tail for ordinary V2 at
  384/6, T4096. MI308X inter-node top-6 rows were measured with QP2 up to
  T2048 and with QP64 at T4096. MI308X inter-node 256/8 rows reproduce under
  QP2.
- **Protocol.** Uniform routing, `scale_dim=0`, capacity equal to the token
  count, `quant_type="none"`. Timing uses 3 passes × (20 warmup + 30 timed
  rounds), dropping the first timed round of each pass. A measurement is
  accepted only after correctness checks, and only if each phase has
  mean/median ≤ 1.10 and max/median ≤ 5, and the per-pass D+C means vary by
  ≤ 1.20×.
- **Metrics.** `avg_latency_us` is the mean over all retained samples.
  Bandwidths are grand means of per-rank, per-round `bytes / time`, not
  `bytes / mean latency`. `bandwidth_gbps` is XGMI for intra-node rows, RDMA for
  `InterNodeV2` and the LL metric for `InterNodeV2LL`. These metrics are not
  interchangeable between families.

## Adding or updating rules

Tune with `tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py`
(`--cmd tuning --tuning-save`). Saving goes through
`hip_tuning_configs.save_internode_result()`, which writes to the file named by
the shape and merges rules by selector.

- Record the QP count with the results, and don't mix QP counts within a token
  range of one table.
- Never copy rows from another GPU model. A missing row is safer than a
  borrowed one.
- Store the geometry that actually ran: keep `block_num` within the CU count.
- `--cmd tuning` measures candidates with load-first off and rewrites the whole
  rule, so re-saving a combine rule clears `combine_load_first`. Re-measure with
  load-first on (A/B the table with and without the flag) before setting it again.
