# gemm_ar_v2: BF16 GEMM + all-reduce over cco SDMA

`GemmArV2` computes `y = sum over ranks r of a_r @ b_r.T` (a row-parallel linear
layer followed by an all-reduce): bf16 in, bf16 on the wire, bf16 out, every rank
gets the whole `[M, N]`. It was written for the `gemm_ar_bf16` evaluation case
(`benchmark/cco/harness/cases/gemm_ar_bf16/`) and implements that case's
`interface.py`.

## Using it

```python
vmm = 2 * GemmArV2.window_bytes_for(world, m_max=M_MAX, n=N, k=K) + (256 << 20)
with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
    op = GemmArV2(comm, n=N, k=K, m_max=M_MAX)
    op.set_weight(b)          # [N, K] bf16, row-major, once
    op.self_test()            # collective; raises if the puts move nothing
    y = op(a)                 # [M, K] bf16, any 1 <= M <= m_max -> [M, N] bf16
    op.close()
```

- Needs `MORI_ENABLE_SDMA=1` and mori built with `BUILD_CCO_SDMA=ON`; 8 ranks in
  one node; gfx950. `N` a multiple of 256, `K` a multiple of 64 (at least 128).
- The result aliases the op's window and the next call overwrites it; whatever
  reads it must be ordered before this rank's next call (same stream, or an event).
- Graph-capturable; no host synchronisation inside a call. Every rank calls with
  the same M, in the same order.
- `fused_path_ran` is `True` after every call of the default mode.
- `debug_preset_counters(v)` sets every epoch, arrival slot and counter to `v`
  (collective), for wrap tests; `debug_dump()` returns them, for hang reports.

## How it works

Reduce-scatter by row bands, then an all-gather, both over the copy engines. M is
padded to `Mp`, a multiple of `8 * 256`; band `d` = rows `[d*Mp/8, (d+1)*Mp/8)`,
reduced by rank `d`.

1. The GEMM (8-wave quad-subtile bf16, 256x256x64) runs as `chunks` launches on
   the caller's stream. Chunk `c` is row tile `c` of every destination's band, so
   each chunk feeds every peer; it writes this rank's partial C into the window,
   write-through.
2. On a side stream, after each chunk's event, a one-block kernel posts one put
   per peer (lane = peer, 3.5 MiB at the default chunking) into the peer's receive
   slot; then a drain: quiet, fence, publish this call's epoch to every peer, wait
   for theirs.
3. Back on the caller's stream: the reduce of band `rank` (fp32 sum in rank order,
   one bf16 rounding, in place), then the gather: push the reduced band to every
   peer, drain, barrier.

The two barriers per call also guard buffer reuse (no credit region). Epochs are
running u32 values advanced by the kernels and compared wrap-safely.

`chunks` default: the largest divisor of `Mp/8/256` up to 8, i.e. 8 / 4 / 2 at
M = 16384 / 8192 / 4096. M that is not a multiple of 2048 (including M=1) is
padded with a zero-filled copy of `a` and runs the same pipeline.

## Switches

Constructor keywords, or `MORI_GARV2_<NAME>` environment variables; every switch
is in the kernel names.

| Switch | Default | Meaning |
|---|---|---|
| `mode` | `stream` | `stream` (above); `split` (one GEMM launch, then one put per peer; reports `fused_path_ran=False`); `fused` (one GEMM launch whose epilogue posts each chunk from the block that completes it, one SDMA queue per chunk) |
| `chunks` | 0 (rule above) | chunks per destination |
| `ctr_order` | `acq_rel` | the fused epilogue's counter atomic ordering |
| `emit_put` | on | off: the fused epilogue counts and elects but posts nothing |
| `wait` | on | off: the drains skip the flag wait (diagnosis only) |
| `stop_after` | none | `gemm`, `scatter`, `reduce`: end the call there (diagnosis; the result is then wrong) |
| `swap_store` | on | off: 8-byte bf16 C stores instead of the permlane16-paired 16-byte ones |

`op.py` also defines one class per diagnostic variant (`GemmArV2Split`,
`GemmArV2Fused`, `GemmArV2StopGemm`, ...) for `op_harness.py --impl`.

## Dispatch rule

The default mode runs at every M. It was measured at M = 4096, 8192 and 16384,
where it is faster than the split path. Below 4096 it runs only so that every M is
served correctly by the fused pipeline; its speed there against the split path
has not been measured.

## Measured

`crsuse2-m2m-v3-054`, 8x MI355X, 2026-11-16, ROCm 10.0.0, torch 2.11.0+rocm10.0.0,
FlyDSL 0.4.2, mori `1422e4b88`; `[M, 7168]`, K=2048; `op_harness.py --mode time`
(one call per CUDA graph, 51 replays, median per rank, maximum over ranks) through
`run.py --arms`, 3 interleaved rounds, median of rounds; development runs, A/A band
0.06-0.23%. Records: `gemm_ar_bf16-20261116-1/runs/ep2/`.

| M | `GemmArV2` (stream) | existing `GemmAllReduceOp(quant="bf16")`, same launch | `mode="split"` | `mode="fused"` |
|---:|---:|---:|---:|---:|
| 16384 | 1164.5 | 1165.7 | 1428.7 | 1208.2 (existing 1164.4 in its launch) |
| 8192 | 624.3 | 626.7 | 744.6 | 653.8 |
| 4096 | 349.3 | 358.0 | 391.4 | 375.2 |

rel_l2 against the fp32 sum over ranks: at most 2.38e-3 (`op_harness.py --mode
check`, M = 16384, 1, 10657, 6309, eager and graph, wrap covered).

At M=16384 the eight chunk launches take 410.2 µs against 383.3 for one launch;
the call up to the end of the scatter drain takes 603.1 µs.

### Measured, no gain

- **`mode="fused"`** (last-finisher epilogue, a counter per (destination, chunk),
  the completing block posts the put, 8 queues per peer): 1208.2 µs at M=16384
  against 1164.4 for the existing op in the same launch and 1164.5 for the stream
  path. Its call up to the scatter drain takes 648.7 µs against the stream path's
  603.1; the same GEMM with the epilogue but no puts, plus the drain, 411.6 µs.
  The K loop is identical to the communication-off build (`isa_check.py`), so the
  loss is in the epilogue or in posting from GEMM blocks; not decomposed.

## Tests

`tests/python/cco/test_gemm_ar_v2.py` (8 GPUs; skips with fewer or without SDMA).
