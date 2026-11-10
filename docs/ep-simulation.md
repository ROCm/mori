# EP dispatch/combine simulation

With `MORI_EP_SIM_ENABLE=1`, `EpDispatchCombineOp` runs without a GPU: no kernel is launched,
no shmem heap is created and no token moves. Each dispatch and combine sleeps the calling
thread for a modelled duration, in C++ (`mori::moe::sim::EpSim`, `src/ops/dispatch_combine/sim.cpp`).
The Python API, shapes and return values stay the same; output tensors are allocated once
and never written, except `total_recv` and `local_expert_count`, which carry real counts.

## Enable

```bash
export MORI_EP_SIM_ENABLE=1 MORI_EP_SIM_GPU=mi355x
```

Ranks find each other through `torch.distributed` (rank 0 publishes a rendezvous name in
the default store), so an existing launcher needs no change. Without `torch.distributed`,
set `MORI_EP_SIM_GROUP` to a name that is the same on every rank and unique per job.
All ranks of one EP group must share a host and `/dev/shm` (one container, or `--ipc=host`).

| Variable | Meaning |
|---|---|
| `MORI_EP_SIM_ENABLE` | `1` to simulate. |
| `MORI_EP_SIM_GPU` | The GPU whose catalog curves time the ops, e.g. `mi355x`. Unset: the analytic model. |
| `MORI_EP_SIM_CATALOG` | Catalog file(s), comma-separated; a later curve replaces an equal one. Default: the shipped `mori/ops/ep_sim_catalog.yaml`. |
| `MORI_EP_SIM_ROUTING` | `indices` (default): read the real top-k ids. `uniform`: ignore them and use the expected load of uniform routing, for callers whose routing is not real (a faked forward). |
| `MORI_EP_SIM_SYNC` | `0`: no rendezvous; each rank times its own load and assumes incoming traffic mirrors outgoing. |
| `MORI_EP_SIM_GROUP` | Rendezvous name when `torch.distributed` is not initialized. |
| `MORI_EP_SIM_TIME_SCALE` | Divide every delay by this factor (same meaning as `MORI_IO_SIM_TIME_SCALE`). |
| `MORI_EP_SIM_LATENCY_US`, `MORI_EP_SIM_XGMI_GBPS`, `MORI_EP_SIM_RDMA_GBPS` | Analytic model: fixed cost and per-GPU bandwidths (defaults 25 us intra-node / 60 us inter-node, 350 GB/s, 45 GB/s). |
| `MORI_EP_SIM_TRACE` | JSONL file, one line per op per rank; `{rank}` is replaced, otherwise `.rank<N>` is appended. |
| `MORI_EP_SIM_SPIN_US` | Spin this long at the end of each sleep for wake-up accuracy (default 20). |
| `MORI_EP_SIM_TIMEOUT_S` | Fail a rendezvous that waits longer than this (default 300): a rank skipped or reordered an op. |

`op.sim_last_stats()` returns the last op's `model_us`, `elapsed_us`, `load_tokens`,
`recv_tokens` and the catalog `source`. `mori.cpp.ep_sim_estimate(config, phase, dtype,
hidden_dim, num_tokens)` prices one op under uniform routing without a rendezvous.

## Model

**Duration of one rank.** The catalog holds latency against tokens per rank, measured by
`bench_dispatch_combine.py` with uniform random routing, per GPU, kernel type, EP size,
phase, dtype, hidden size, top-k, combine quantization and zero copy. A call's load is
counted from its routing in (token, destination rank) pairs, separately for intra-node and
inter-node traffic, as the larger of what the rank sends and what it receives. That load
is converted to the token count that loads a rank equally under the curve's own routing,
and the latency is read there (linear between points, the last slope beyond them). The
closest curve wins when none matches exactly; its token axis is then scaled by bytes per
token, and the substitution is logged once. Without a curve for the GPU, kernel and EP
size, the analytic model is `latency + max(intra bytes / xgmi, inter bytes / rdma)`.

**Coupling between ranks.** EP ops end on a cross-device barrier, so a rank that arrives
early waits for the last one. Each op, every rank publishes its arrival time and per-rank
load in shared memory; all ranks then sleep until `max over ranks of (arrival + duration)`.
A rank that arrives 20 ms late therefore costs every other rank 20 ms, as on hardware.

**AsyncLL.** `dispatch()`/`combine()` only post; `dispatch_recv()`/`combine_recv()` sleep
until the deadline, so work placed between the two hides the transfer.

## Catalog

`python/mori/ops/ep_sim_catalog.yaml` is generated from the latencies recorded in
`python/mori/ops/tuning_configs`; regenerate it after retuning:

```bash
python3 tools/ep_sim_catalog.py > python/mori/ops/ep_sim_catalog.yaml
```

To add hardware or a shape, write another file in the same format and list it after the
default in `MORI_EP_SIM_CATALOG`.

## Not simulated

Routing handles (`return_routing=`, `routing=`), the standard-MoE conversions,
`get_dispatch_src_token_pos*`, CUDA graph capture (the sleep happens on the host at call
time, so a captured graph replays without it), and ranks spread over several hosts.
The C++ launch API (`LaunchDispatch`, `LaunchCombine`, their `Recv` halves) is simulated
for timing; its outputs are not available, and indices must be host-readable.
