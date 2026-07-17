# ccl: hierarchical cross-node AllGather (intra-node SDMA + inter-node RDMA)

## Summary

Adds a hierarchical AllGather to MORI-CCL (`mori.ccl.HierAllGather`, an
`all_gather_into_tensor`-compatible collective) that keeps intra-node traffic on
the GPU **SDMA copy engines** (XGMI) and moves inter-node traffic over **RDMA**
(NIC).

Motivation is **compute/communication overlap**: the collective runs
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

## Results (MI300X, mlx5 RoCEv2 — w16 = 2 node × 8 GPU; all bit-exact vs RCCL)

Raw logs, CSVs, and the plot scripts live under
[`bench/`](bench/); every figure below is
regenerated from that data. See **Reproduce** for the exact commands.

### 1. Standalone AllGather bandwidth vs RCCL (E2E-stable config)

The shipped E2E construction (`MORI_HIER_UT_FAST=0`, device `ibgda_sdma`) is
bit-exact and tracks RCCL closely, converging as the message grows. A single
AllGather is **not** where the win is — the collective is network-bound and
GPU-light, so standalone bandwidth is near-parity, not a beat:

| per-rank size | ibgda_sdma / RCCL |
|--------------:|:-----------------:|
| 64 MB  | 0.90× |
| 128 MB | 0.94× |
| 256 MB | 0.95× |
| 512 MB | 0.96× |

![standalone AllGather bandwidth](bench/results/mi300x_mlx5/ag_perf_e2e_stable_w16.png)

### 2. GEMM time under concurrent AllGather (the no-CU-contention dividend)

Where the win actually comes from: RCCL's CU-resident AllGather kernels steal the
GPU from concurrent GEMMs; `hp_sdma` keeps the cross-node leg on the CPU and the
intra-node leg on the SDMA copy engine, so the GEMMs finish faster while 50
AllGathers run concurrently (lower = better, bit-exact vs RCCL):

| 50 AGs ‖ 50 GEMMs | RCCL | hp_sdma | speedup |
|---|--:|--:|:--:|
| 8 MB,  GEMM n=2048 | 3.5 ms  | 2.6 ms  | 1.33× |
| 8 MB,  GEMM n=4096 | 17.5 ms | 15.8 ms | 1.11× |
| 16 MB, GEMM n=4096 | 19.1 ms | 16.0 ms | 1.20× |

![GEMM time under concurrent AllGather](bench/results/mi300x_mlx5/overlap_w16_gemm_time.png)

### 3. End-to-end FSDP2 (Qwen-7B, seq 2048, w16) — drop-in MoriAllGather

Training loss is bit-identical to the native run over the whole curve while step
throughput beats the framework default. Three mori variants (intra × inter leg):

| variant | inter-node leg | intra-node leg | throughput vs RCCL |
|---|---|---|--:|
| `hp_sdma`    | host-proxy (CPU-posted RDMA) | SDMA (XGMI, CU-free) | ~1.20× |
| `hp_cu`      | host-proxy                   | NCCL (CU)            | ~1.10× |
| `ibgda_sdma` | device IBGDA (GPU-posted RDMA) | SDMA               | ~1.07× |

![E2E loss](bench/results/mi300x_mlx5/e2e_all_w16_loss.png)
![E2E throughput](bench/results/mi300x_mlx5/e2e_all_w16_tflops.png)

## Reproduce

All launchers live in `bench/scripts/`. Each one drives the 2-node run itself
(ssh into master + worker, clear stale procs, start `torchrun`); the UT sources
are in `tests/python/ccl/`. The node pair / NIC list is at the top of each script
(env-overridable: `MASTER`/`WORKER`/`IFACE`/`MORI_RDMA_DEVICES`/…). Raw logs +
figures land under `bench/results/mi300x_mlx5/`.

```bash
cd examples/fsdp_sdma/bench/scripts

# 1) Standalone AllGather bandwidth UT (device ibgda_sdma vs RCCL)
#    -> ../results/mi300x_mlx5/ag_perf_e2e_stable_w16.png
bash run_ut_ag_perf.sh e2e  64 128 256 512   # E2E-stable (shipped) config
bash run_ut_ag_perf.sh perf 64 128 256 512   # pure-perf (standalone_fast, NOT E2E-legal), for context
python ../results/mi300x_mlx5/plot_ag_perf.py   # (re)draw the figure from the run (or committed CSV)

# 2) Compute/comm overlap UT (GEMM time under 50 concurrent AGs, hp_sdma vs RCCL)
#    -> ../results/mi300x_mlx5/overlap_w16_gemm_time.png   (args: gemm_n size_mb nops)
bash run_ut_overlap.sh 2048  8 50
bash run_ut_overlap.sh 4096  8 50
bash run_ut_overlap.sh 4096 16 50

# 3) End-to-end FSDP2 (Qwen-7B): RCCL baseline + one mori variant, bit-exact loss + tflops
bash run_e2e.sh              # RCCL + hp_sdma (default)
bash run_e2e.sh hp_cu
bash run_e2e.sh ibgda_sdma
WORLD=w8 bash run_e2e.sh     # world=8 (default w16)
```

## Test plan

- [x] Bit-exact vs `torch.distributed.all_gather_into_tensor` for
      `{bf16, fp16, fp32, int32}` on every tested size (true 2-node, world=8 & 16).
- [x] Standalone AllGather bandwidth sweep, E2E-stable config (near-parity, bit-exact).
- [x] Compute/comm overlap UT — GEMM finishes 1.1–1.3× faster under 50 concurrent AGs.
- [x] End-to-end FSDP2 training, loss bit-identical to native at world=8 & 16.
