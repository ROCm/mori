# ccl: hierarchical cross-node AllGather (intra-node SDMA + inter-node RDMA)

## Summary

Adds a hierarchical AllGather to MORI-CCL (`mori.ccl.HierAllGather`, an
`all_gather_into_tensor`-compatible collective) that keeps intra-node traffic on
the GPU **SDMA copy engines** (XGMI) and moves inter-node traffic over **RDMA**
(NIC).

Motivation is **compute/communication overlap (通算并行)**: the collective runs
on the dedicated SDMA copy engines instead of the compute units, so an AllGather
issued concurrently with a GEMM does not steal CUs from the GEMM — parity with
the native (non-SDMA) path standalone, and a strict win when overlapped with
compute.

## Design

- Intra-node phase: SDMA sub-group gather over XGMI (no CU usage, no NIC).
- Inter-node phase: RDMA ring exchange of node-blocks over the NIC.
- Fused `ring || local-gather` kernel: the inter-node RDMA ring and the
  ring-independent local node-block SDMA gather run concurrently in one grid,
  stream-ordered, direct-to-output (no staging copy).
- Correctness: **bit-exact** vs `torch.distributed.all_gather_into_tensor`
  (zero tolerance) for `{bf16, fp16, fp32, int32}`, all tested sizes.

## API

```python
from mori.ccl import HierAllGather

ag = HierAllGather(
    my_pe=rank, npes=world_size, ranks_per_node=local_world_size,
    input_buffer_size=per_rank_bytes,
    output_buffer_size=per_rank_bytes * world_size,
    copy_output_to_user=True,
)
ag(input_tensor, output_tensor, numel, stream)   # intra=SDMA, inter=RDMA
```

## Results (2 nodes × 4 GPUs = 8 ranks, MI355X, fp32)

**Standalone AllGather — SDMA ≥ native:**

| size | SDMA GB/s | native GB/s | ratio |
|-----:|----------:|------------:|------:|
| 4 MB   | 57.5  | 72.8  | 0.79 |
| 8 MB   | 147.2 | 120.1 | 1.23 |
| 16 MB  | 174.4 | 130.8 | 1.33 |
| 32 MB  | 191.6 | 156.8 | 1.22 |
| 64 MB  | 202.3 | 149.8 | 1.35 |
| 128 MB | 205.9 | 153.0 | 1.35 |
| 256 MB | 202.5 | 165.4 | 1.22 |
| 512 MB | 203.5 | 171.0 | 1.19 |

SDMA ≥ native for every size ≥ 8 MB (1.19–1.35×); 4 MB is latency-bound.

![standalone](benchmarks/allgather_results/chart_standalone.png)

**Under concurrent GEMM (total time, lower is better) — SDMA strictly faster:**

| size | GEMM + native AG (ms) | GEMM + SDMA AG (ms) | SDMA advantage |
|-----:|----------------------:|--------------------:|---------------:|
| 16 MB  | 4.27  | 4.26  | faster |
| 32 MB  | 4.59  | 4.55  | faster |
| 64 MB  | 5.19  | 5.03  | ~3% |
| 128 MB | 7.52  | 6.27  | ~17% |
| 256 MB | 14.15 | 11.38 | ~20% |
| 512 MB | 26.24 | 21.97 | ~16% |

Because SDMA uses copy engines while the native path consumes CUs the GEMM needs,
the SDMA AllGather overlaps with compute far better — 16–20% lower total time at
large sizes.

![gemm overlap](benchmarks/allgather_results/chart_gemm_overlap.png)

Raw data: `benchmarks/allgather_results/sweep_standalone.csv`,
`benchmarks/allgather_results/sweep_gemm_overlap.csv`.

## Test plan

- [x] Bit-exact vs `torch.distributed.all_gather_into_tensor` for
      `{bf16, fp16, fp32, int32}` on every tested size (true 2-node, world=8).
- [x] Standalone bandwidth size sweep 4 MB–512 MB (SDMA ≥ native for ≥ 8 MB).
- [x] GEMM-overlap size sweep (SDMA strictly faster; 16–20% at 128–512 MB).
- Reproduce:
  ```bash
  python3 setup.py build_ext --inplace
  export PYTHONPATH=$PWD:$PWD/python:$PYTHONPATH MORI_ENABLE_SDMA=1
  torchrun --nnodes=2 --nproc_per_node=4 --master_addr=<ip> --master_port=29500 \
    tests/python/ccl/test_hier_allgather.py
  torchrun --nnodes=2 --nproc_per_node=4 ... tests/python/ccl/bench_sweep.py
  torchrun --nnodes=2 --nproc_per_node=4 ... tests/python/ccl/bench_gemm_overlap.py
  ```
