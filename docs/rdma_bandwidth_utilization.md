# RDMA Bandwidth Utilization in Dispatch/Combine Kernels

## Overview

This document explains how RDMA bandwidth is used in the `internode_v1` (and `v1_ll`) dispatch/combine kernels, and how to measure or verify that the RDMA link is being driven at full capacity.

---

## Kernel Execution Path

Each dispatch+combine iteration consists of the following phases (as seen in the profiler trace):

```
ep_dispatch_copy_to_staging          [serial, separate kernel]
    ↓
EpDispatchInterNodeV1Kernel          [one kernel launch, two block groups]
  ├─ RDMA blocks (blockId < rdmaBlockNum)
  │     dispatch_inter_node_send     → fire RDMA puts to other nodes
  │     dispatch_inter_node_recv     → poll flags, copy received tokens into XGMI mem
  └─ XGMI blocks (blockId >= rdmaBlockNum)
        dispatch_intra               → direct XGMI writes to intra-node peers
  [all blocks] dispatch_sync         → barrier, signal recv counts, shmem_quiet
    ↓
EpCombineSync                        [serial, separate kernel]
    combine_sync                     → copy local expert outputs into combineInp staging;
                                       accumulate routing weights; reset dispatch counters
    ↓
EpCombineSyncBarrier                 [serial, 1 block × 1 warp]
    ep_combine_sync_barrier          → cross-device barrier via RDMA AMO to every peer node;
                                       spins until all nodes acknowledge before combine starts
    ↓
EpCombineInterNodeV1Kernel           [one kernel launch, two block groups]
  ├─ RDMA blocks
  │     combine_inter_node           → poll flags, accumulate, RDMA-send results back
  └─ XGMI blocks
        combine_intra_node           → accumulate intra-node contributions
    ↓
EpCombineAll                         [serial, separate kernel]
    ep_combine_all                   → final weighted accumulation across all nodes into output
```

The low-latency variant (`v1_ll`) has the same structure but uses
`dispatch_inter_node_ll_send` / `_ll_recv` / `combine_inter_node_ll` / `combine_intra_node_ll`.

---

## What RDMA Actually Transfers

### Dispatch send (`DispatchInterNodeSend` / `DispatchInterNodeLLSend`)

Each RDMA block iterates over chunks of `warpSize` (64) local tokens and fires one
`ShmemPutMemNbiSignalThread` call per chunk per remote node:

```
bytes per put = tokenNum * xferBytes
xferBytes     = hiddenBytes + indexBytes + weightBytes + scaleBytes + sizeof(index_t)
```

The put is a 1-sided non-blocking RDMA write + remote atomic signal. QP selection rotates
across `numQpPerPe` queue pairs:

```cpp
int qpId = (tokenId / warpSize) % config.numQpPerPe;
```

Total bytes injected per rank per iteration (dispatch direction):
```
dispatch_rdma_bytes = curRankNumToken × xferBytes × (nNodes - 1)
```

### Combine send (`CombineInterNodeTyped` / `CombineInterNodeLLTyped`)

After accumulating the expert outputs, the combine RDMA blocks fire one
`ShmemPutTypeNbiWarp` per completed chunk back to the originating node:

```
bytes per put = thisChunkTokenNum * tokCombXferBytes
tokCombXferBytes = hiddenBytes  (or fp8HiddenBytes for FP8 paths)
                 + weightBytes  (if weightsBuf != nullptr)
```

Total bytes per rank per iteration (combine direction):
```
combine_rdma_bytes = curRankNumToken × tokCombXferBytes × (nNodes - 1)
```

---

## Theoretical Peak vs. Observed Throughput

### Computing theoretical peak

Given a profiler trace (times in microseconds):

```
dispatch_send_duration_us   = wall-clock span of dispatch_inter_node_send
combine_inter_duration_us   = wall-clock span of combine_inter_node
```

Theoretical RDMA throughput check:

```python
# Example calculation
hidden_dim     = 7168            # model hidden dim
dtype_bytes    = 2               # bf16
index_bytes    = topk * 4       # int32 per expert slot
weight_bytes   = topk * 4       # float32
scale_bytes    = scale_dim * scale_type_size  # optional

xfer_bytes_tok = hidden_dim * dtype_bytes + index_bytes + weight_bytes + scale_bytes + 4

total_send_bytes = cur_rank_num_token * xfer_bytes_tok * (n_nodes - 1)

# From trace
send_dur_sec = dispatch_send_duration_us * 1e-6

observed_bw_GBps = total_send_bytes / send_dur_sec / 1e9

# Compare to link rating, e.g. 400 Gb/s = 50 GB/s per port
# With numQpPerPe QPs the effective ceiling is still the NIC BW
print(f"Observed RDMA send BW: {observed_bw_GBps:.1f} GB/s")
```

### Understanding the receive-side timing

The `dispatch_inter_node_recv` / `combine_inter_node` spans include both:
1. **Waiting (spinning)** on `chunkFlag` or `nodeRecvTokenNum` flags until the remote data arrives.
2. **Copying** from the RDMA staging buffer into XGMI-accessible memory.

A long `recv` span relative to the `send` span indicates the receiver is
**blocked waiting for data** — the NIC/fabric is the bottleneck.
A short `recv` span means the data arrived before or soon after the receiver started polling.

---

## How to Check RDMA Bandwidth from the Trace

### Step 1: Collect a profiler trace

Enable the kernel profiler (requires `ENABLE_PROFILER` build flag) and run:

```bash
python analyze_trace_internode.py <trace_rank_0_XXXXXX.json> --out timeline.png
```

The trace gives wall-clock spans for each kernel on each warp, merged into a
single timeline per iteration.

### Step 2: Read key durations from the text summary

The script prints:
```
Kernel                             start      end      dur
dispatch_inter_node_send          0.00    120.00   120.00   ← RDMA send phase
dispatch_inter_node_recv         10.00    350.00   340.00   ← poll + copy received tokens
dispatch_intra                    0.00    180.00   180.00   ← concurrent XGMI
dispatch_sync                   350.00    360.00    10.00   ← barrier + shmem_quiet
combine_sync                    360.00    400.00    40.00   ← copy outputs into staging
ep_combine_sync_barrier         400.00    420.00    20.00   ← cross-node AMO barrier
combine_inter_node              420.00    760.00   340.00   ← RDMA combine (concurrent)
combine_intra_node              420.00    560.00   140.00   ← XGMI combine (concurrent)
ep_combine_all                  760.00    810.00    50.00   ← final accumulation
```

### Step 3: Compute observed RDMA bandwidth

```python
# Plug in values from your config and trace
send_dur_us        = 120.0        # dispatch_inter_node_send wall-clock span
combine_dur_us     = 340.0        # combine_inter_node wall-clock span
cur_rank_num_token = 2048
n_nodes            = 4
hidden_dim         = 7168
dtype_bytes        = 2            # bf16
topk               = 4
xfer_bytes_tok     = hidden_dim * dtype_bytes + topk * 4 + topk * 4 + 4

dispatch_bytes = cur_rank_num_token * xfer_bytes_tok * (n_nodes - 1)
combine_bytes  = cur_rank_num_token * hidden_dim * dtype_bytes * (n_nodes - 1)

dispatch_bw_GBps = dispatch_bytes / (send_dur_us * 1e-6) / 1e9
combine_bw_GBps  = combine_bytes  / (combine_dur_us * 1e-6) / 1e9

print(f"Dispatch RDMA BW: {dispatch_bw_GBps:.1f} GB/s")
print(f"Combine  RDMA BW: {combine_bw_GBps:.1f} GB/s")
```

### Step 4: Interpret results

| Observation | Likely cause |
|---|---|
| `send_dur` ≈ `dispatch_bytes / NIC_BW` | Send side is BW-saturated (good) |
| `recv_dur` >> `send_dur` | Receiver spins waiting; fabric or sender is the bottleneck |
| `recv_dur` << `send_dur` | Receiver catches up quickly; may be underloaded (small batch) |
| `combine_inter_dur` >> `combine_rdma_bytes / NIC_BW` | Poll loop stall or NIC contention |
| Iteration time dominated by `dispatch_sync` or `combine_sync` | Barrier overhead, not BW |

---

## Key Tuning Parameters That Affect RDMA Utilization

| Parameter | Effect |
|---|---|
| `rdmaBlockNum` | More RDMA blocks → more warps driving sends concurrently |
| `numQpPerPe` | More QPs → more in-flight RDMA operations, reduces head-of-line blocking |
| `curRankNumToken` (batch size) | Larger batches → longer RDMA transfers → easier to saturate |
| `xferBytes` (hidden dim, topk) | Larger transfer size per token → higher bandwidth per put |
| `warpSize`-aligned chunks | Chunk size = 64 tokens; larger chunks = fewer puts with less overhead |

### v1 vs. v1_ll (low-latency) tradeoff

- **v1**: Each RDMA block handles a contiguous range of chunks per remote node. The recv polling is simple but may stall if chunks are out of order across nodes.
- **v1_ll**: Work is distributed per `(expert, token, node)` triplet. Each warp picks up work as soon as any chunk flag signals. Better latency at small batch sizes; similar peak BW at large batches.

---

## Sanity Checks from Hardware Counters (optional)

If you have access to `rocprof` or equivalent:

```bash
rocprof --stats --hip-trace \
  --counter "TCC_EA_RDREQ_sum,TCC_EA_WRREQ_sum" \
  python run_dispatch_combine.py
```

Cross-check the `TCC_EA_WRREQ` write request count against
`total_sends = curRankNumToken / warpSize * nQpPerPe * (nNodes - 1)`.

For RDMA-specific counters (NIC-side), use the vendor tool (e.g., `perfquery` for InfiniBand or
the AMD NIC perf tools) to observe port-level `PortXmitData` and `PortRcvData` during the run.
If `PortXmitData / iteration_time` ≈ rated NIC bandwidth, the link is saturated.

---

## Summary

To determine if RDMA is fully utilized:

1. **Collect a trace** with the built-in kernel profiler.
2. **Read `dispatch_inter_node_send` duration** from the text summary.
3. **Compute theoretical bytes** sent (tokens × xferBytes × (nNodes − 1)).
4. **Divide** to get observed GB/s; compare to the NIC's rated bandwidth.
5. If observed < rated by >20%, look for: too few QPs, too small batch, or the recv side
   blocking the send side (check `recv` span vs. `send` span ratio).
