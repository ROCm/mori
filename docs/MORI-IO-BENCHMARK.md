# MORI-IO Benchmark

## Table of Contents

- [Which benchmark to use](#which-benchmark-to-use)
- [C++ Benchmark (default, nixlbench-matching)](#c-benchmark-default-nixlbench-matching)
  - [Build](#build)
  - [Run (two nodes)](#run-two-nodes)
  - [Sweep modes](#sweep-modes)
  - [Differences from the Python flags](#differences-from-the-python-flags)
- [Python Benchmark](#python-benchmark)
  - [Fabric (cross-node scale-up UALink super-node)](#fabric-cross-node-scale-up-ualink-super-node)
- [Benchmark Arguments](#benchmark-arguments)
- [Results: Thor2 RDMA Read](#results-thor2-rdma-read)
- [Results: Thor2 RDMA Write](#results-thor2-rdma-write)
  - [Message Size Sweep](#message-size-sweep)
  - [Batch Size Sweep](#batch-size-sweep)
- [Results: CX7 RDMA (Batch Size = 1)](#results-cx7-rdma-batch-size--1)
  - [Write](#write)
  - [Read](#read)
- [Benchmark Tuning Guide](#benchmark-tuning-guide)
  - [Recommended starting config](#recommended-starting-config)
  - [Parameters that help](#parameters-that-help)
  - [The merged-WR size limit](#the-merged-wr-size-limit)
  - [Parameters that usually do not help (or can hurt)](#parameters-that-usually-do-not-help-or-can-hurt)
  - [Host (CPU) vs GPU memory](#host-cpu-vs-gpu-memory)

## Which benchmark to use

MORI-IO ships two benchmarks that measure the same transfer paths:

- **C++ (`tests/cpp/io/bench_engine`) — the default.** No Python interpreter in
  the measurement loop, and its timing matches
  [nixlbench](https://github.com/ai-dynamo/nixl) so MORI-IO and NIXL numbers are
  directly comparable. This is what `tools/run_internode_io_benchmark.sh` runs by
  default and what CI reports. Start here.
- **Python (`tests/python/io/benchmark.py`) — kept for parity.** Uses `torchrun`
  for rendezvous; convenient when you are already in a torch/torchrun workflow.
  Select it with `run_internode_io_benchmark.sh --engine python`.

Both accept the same workload flags (op, sweep, batch, QP, session); the C++ tool
also accepts the Python spellings as aliases. See
[Differences from the Python flags](#differences-from-the-python-flags).

The two-node runner picks the engine with `--engine`:

```bash
# C++ engine (default)
tools/run_internode_io_benchmark.sh --rank 0 --master-addr <ip> --ifname <nic> \
    -- --op write --all --enable-sess --enable-batch-transfer

# Python engine
tools/run_internode_io_benchmark.sh --engine python --rank 0 --master-addr <ip> \
    --ifname <nic> -- --op-type write --all --enable-sess --enable-batch-transfer
```

## C++ Benchmark (default, nixlbench-matching)

The **C++** benchmark, `tests/cpp/io/bench_engine.cpp`, is the default. Its
measurement loop is written to **match [nixlbench](https://github.com/ai-dynamo/nixl)**
(NVIDIA NIXL's `xferbench`) exactly, so MORI-IO and NIXL RDMA numbers are
apples-to-apples on the same fabric. Use it when you want a head-to-head
comparison against NIXL, or a benchmark with no Python-interpreter overhead in the
measurement loop (which the Python bench pays per iteration, so it shows up most
at small message sizes).

What "nixl-matching" means here:

- **One whole-loop timer** around the entire iteration loop (not a per-iteration
  sum), identical to nixlbench's `total_timer`.
- **Latency is reported per single transfer**: `total_us / (iters × batch)`,
  matching nixl's `avg_latency = total_duration / (per_thread_iter × batch_size)`.
- **Bandwidth** `= msg × batch × iters / 1e9 / (total_us / 1e6)` (GB = 10⁹), same
  units as nixl.
- **Completion is always an inline spin-poll** in the benchmark thread (mirroring
  nixl's `getXferStatus` scan); there is no cv-blocking wait that would add wakeup
  latency and make numbers non-comparable.
- **Strict stop-and-wait** (one request outstanding at a time, == nixl
  `--pipeline_depth 1`): post one request, spin until it completes, then post the
  next.
- **Warmup** iterations are excluded from timing (`--warmup-iters`, nixl
  `--warmup_iter`).

### Build

`bench_engine` builds with the C++ tests (both options default `ON`):

```bash
cd /path/to/mori
cmake -B build -DBUILD_IO=ON -DBUILD_TESTS=ON
cmake --build build --target bench_engine -j
# binary: build/tests/cpp/bench_engine
```

`bench_engine --help` prints the full flag list, grouped by rendezvous,
workload, sweep, memory, and backend tuning.

The MORI shared libraries must be discoverable at runtime — from the build tree
above, or from `python/mori` after an editable install:

```bash
export LD_LIBRARY_PATH=$PWD/build/src/io:$PWD/build/src/application:$LD_LIBRARY_PATH
# or, after an editable install:
export LD_LIBRARY_PATH=/path/to/mori/python/mori:$LD_LIBRARY_PATH
```

### Run (two nodes)

Unlike the Python benchmark (which uses `torchrun` for the out-of-band
rendezvous), the C++ benchmark brings the two processes together over MORI's own
socket bootstrap — the same layer that starts multi-node CCO and SHMEM jobs, so
connect and accept are retried over the `MORI_BOOTSTRAP_TIMEOUT` budget (seconds,
default `300`). **Rank 0 is the initiator** and drives every transfer; **rank 1
is the target** and only keeps its memory registered. Start the target first as a
habit — rank 0 is the rendezvous root, so either order works and the two retry at
each other, but the second process has to arrive within the budget.
`--master-ip` is rank 0's IP and is passed identically on both ranks; `--self-ip`
is the address each rank advertises to its peer.

```bash
# On the TARGET node (rank 1) — start this FIRST:
MORI_DISABLE_AUTO_XGMI=1 build/tests/cpp/bench_engine \
    --rank 1 --master-ip 10.190.162.0 --self-ip 10.190.162.1 --port 18515 \
    --op write --all --sweep-start 1048576 --sweep-max 33554432 --sweep-step 1048576 \
    --iters 500 --warmup-iters 50 --num-qp-per-transfer 4 --enable-sess

# On the INITIATOR node (rank 0) — start this SECOND (prints the results table):
MORI_DISABLE_AUTO_XGMI=1 build/tests/cpp/bench_engine \
    --rank 0 --master-ip 10.190.162.0 --self-ip 10.190.162.0 --port 18515 \
    --op write --all --sweep-start 1048576 --sweep-max 33554432 --sweep-step 1048576 \
    --iters 500 --warmup-iters 50 --num-qp-per-transfer 4 --enable-sess
```

Both ranks must be given the **same** benchmark args (op, sweep, batch, etc.).
Only rank 0 (initiator) prints the results table:

```
MsgSize(B)  Batch  Iters  AvgBW(GB/s)  AvgLat(us)  TotalDur(us)
```

`--port` is not the only TCP port in play, which matters behind a firewall:

| Endpoint | Bound by | Address |
|----------|----------|---------|
| Bootstrap rendezvous | rank 0 only | `--master-ip` : `--port` (`18515` above) |
| Engine control plane | both ranks | `--self-ip` : `--port + 1 + rank` (`18516` / `18517` above) |
| Bootstrap ring / allgather | both ranks | chosen interface, kernel-assigned ephemeral port |

The engine control plane is a plain TCP channel: the initiator dials the target's
advertised `EngineDesc` address to run the QP handshake and to look up the remote
memory region. No RDMA payload crosses it.

### Sweep modes

- `--all` — sweep message size. Geometric ×2 by default, or linear when
  `--sweep-step > 0`, from `--sweep-start` to `--sweep-max`; batch stays at
  `--transfer-batch-size`.
- `--all-batch` — fix message size at `--buffer-size` and sweep batch
  `1,2,4,…,32768`.
- neither — a single `(--buffer-size, --transfer-batch-size)` point.

### Differences from the Python flags

Most flags share names with the Python benchmark (`--op-type`/`--op`,
`--num-qp-per-transfer`, `--num-worker-threads`, `--transfer-batch-size`,
`--enable-sess`, `--batch-contiguous`, `--disable-chunking`, `--chunk-bytes`,
`--poll_cq_mode`, `--mem-type`, `--iters`, `--sweep-step`, `--all`,
`--all-batch`), but note:

| Python | C++ (nixl-style) | Notes |
|--------|------------------|-------|
| `torchrun ... --node_rank` | `--rank 0` / `--rank 1` | C++ rendezvous runs over MORI's socket bootstrap; rank 0 is the root, and the second process must arrive within `MORI_BOOTSTRAP_TIMEOUT`. |
| `--host` | `--master-ip` + `--self-ip` | `--host` is accepted as a spelling of `--master-ip`. `--master-ip` = rank 0's IP, same on both ranks; `--self-ip` = the control-plane IP each rank advertises to the peer, which Python does not need because torchrun supplies it. |
| `--src-gpu` / `--dst-gpu` | `--gpu` / `--target-dev-offset` | Python's names are accepted: `--src-gpu` is `--gpu`, and `--dst-gpu N` sets `--target-dev-offset` to `N - gpu` (passing both a conflicting `--dst-gpu` and `--target-dev-offset` is rejected). Under `--xgmi-single-process` they are the two local GPUs directly. |
| `--num-initiator-dev` / `--num-target-dev` | same | One process per GPU on each side, forked before any HIP call; initiator local *i* pairs with target local *i*, driving GPU `--gpu + i`. The two counts must be equal, as Python asserts. Each pair prints its own `[gpu N]`-prefixed row and rank 0 adds an `[AGGREGATE]` line summing bandwidth across pairs — Python prints per-rank tables with no aggregate. |
| `--enable-batch-transfer` (default OFF) | `--enable-batch-transfer` / `--disable-batch-transfer` (C++ default **ON**) | Same meaning as Python: **ON** = one N-descriptor batch request (the nixl-equivalent; nixl always batches); **OFF** = `--transfer-batch-size` N *individual* single-transfer submissions per iteration (Python `run_single_once`). The C++ tool defaults **ON** so the out-of-the-box run and `--all-batch` sweep are nixl-comparable; pass `--disable-batch-transfer` for the Python single-submission path. |
| `--iters` (default `128`) | `--iters` (default `500`), plus `--warmup-iters` | Only the warmup flag is new: it runs untimed iterations before the timed region, matching nixl's `--warmup_iter`. |
| `--sweep-start-size` / `--sweep-max-size` | `--sweep-start` / `--sweep-max` (long names also accepted) | Names only; `--sweep-step` (`0` = geometric ×2, `>0` = linear) behaves the same in both. |
| `--backend rdma\|xgmi\|fabric` | same | `--num-streams` / `--num-events` size the XGMI **and** FABRIC HIP pools, as in Python. `fabric` (UALink scale-up, GPU memory only) needs a host whose GPUs expose `/sys/bus/pci/devices/*/ualink`; on RoCE-only hardware the engine reports `No available backend found` at the first transfer. |
| `--xgmi-multiprocess` (default OFF) | `--xgmi-single-process` (default OFF, i.e. multiprocess) | The defaults are **opposite**. The C++ tool's two-rank path is already Python's `--xgmi-multiprocess`: two processes with distinct engine keys, so the backend takes its `hipIpc*` route. `--xgmi-single-process` adds Python's *default* instead — one process owning both GPUs, no rendezvous and no IPC. `--xgmi-multiprocess` is accepted to document intent but is a no-op. |
| *(always on)* | `--skip-validate` | Both tools verify transferred data as part of the run. Python compares the target's buffer byte-for-byte over gloo; the C++ tool seeds a rank- and offset-dependent pattern and exchanges one checksum per transferred slot after the sweep (so the check is O(batch) on the wire but still covers every byte), printing `validation: OK` or naming the first mismatching slot and exiting non-zero. `--skip-validate` opts out; passing it to only one rank is safe — that rank contributes nothing and the run reports `validation: skipped` rather than hanging. |

> `--self-ip` must be an address the peer can reach over TCP, since it becomes the
> `EngineDesc` host the peer dials for the QP handshake and remote-MR lookup.
> `0.0.0.0` does not work: the peer ends up dialing its own loopback and fails with
> `connect(...): Connection refused`. It does **not** choose the RDMA device that
> carries the data — that stays rail/NUMA-aware (override with
> `MORI_RDMA_DEVICES`) — so the control IP need not be the RoCE interface's. The
> bootstrap picks its local interface independently of both, so set
> `MORI_SOCKET_IFNAME` when that choice has no route to the peer.
>
> `MORI_DISABLE_AUTO_XGMI` gates only the RDMA→XGMI *fallback* for hosts with no
> active RDMA device, and that fallback is off unless the variable is literally
> `0`. The `=1` above is therefore belt-and-braces, worth keeping only where
> something in the environment sets it to `0`.
>
> For `--backend xgmi`, run both ranks on the **same** host with different `--gpu`
> (XGMI is intra-node) and the same `--master-ip`/`--self-ip`; their engine control
> ports still differ because those are derived from the rank. The backend is
> created explicitly, so no env var is needed. That two-rank form measures the
> *cross-process* XGMI path (distinct engine keys ⇒ `hipIpc*`). To measure what the
> Python benchmark measures by default, pass `--xgmi-single-process` with
> `--src-gpu`/`--dst-gpu`: one process owns both GPUs, there is no rendezvous and no
> IPC, so `--rank`/`--master-ip` are not needed and small-message numbers become
> directly comparable to Python's.
>
> For multi-GPU runs (`--num-initiator-dev N --num-target-dev N`) the bootstrap
> world is `2N` ranks, laid out initiators first. Engine control ports are derived
> from the **global** rank, so `--port P` reserves `P` for the bootstrap and
> `P+1 … P+2N` for the engines — leave that range free. Each side forks `N-1`
> children before touching HIP, so a failure in one pair still reaps the others
> rather than leaving them holding ports.

## Python Benchmark

Kept for parity with the C++ default (select it in the runner with
`--engine python`). It uses `torchrun` for the out-of-band rendezvous instead of
MORI's socket bootstrap.

```bash
cd /path/to/mori
export PYTHONPATH=/path/to/mori:$PYTHONPATH
export GLOO_SOCKET_IFNAME=ens14np0  # Set to your NIC interface

# Run on two nodes (replace node_rank and master_addr)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr="10.194.129.65" --master_port=1234 \
    tests/python/io/benchmark.py --host="10.194.129.65"
```

### Fabric (cross-node scale-up UALink super-node)

The `fabric` backend transfers between two nodes that share the same scale-up
fabric domain (**same vPOD** — verify with `amd-smi fabric` that `PPOD_ID` and
`VPOD_ID` match and `ACCEL_STATE` is `READY` on both). Buffers are allocated as
fabric-exportable VMM memory internally, so no `--host`/QP options are needed;
OOB metadata is still exchanged over gloo (torchrun).

```bash
# node A (initiator)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr="<nodeA_ip>" --master_port=1234 \
    tests/python/io/benchmark.py --backend fabric \
    --all --enable-sess --enable-batch-transfer --transfer-batch-size 1 \
    --sweep-start-size 1048576 --sweep-max-size 268435456 --op-type read

# node B (target): same command with --node_rank=1
```

> Requires ROCm ≥ 7.15 (HIP fabric VMM APIs). Only GPU memory is supported
> (`--mem-type gpu`). If the two nodes are not in the same vPOD, session creation
> fails fast with a clear error. The C++ benchmark also supports `--backend fabric`.

## Benchmark Arguments

| Argument | Description |
|----------|-------------|
| `--backend` | `rdma` (cross-node), `xgmi` (intra-node), or `fabric` (cross-node UALink super-node, same vPOD) |
| `--num-streams` / `--num-events` | HIP stream/event pool size for `xgmi`/`fabric` backends (default `64`) |
| `--buffer-size` | Message size per transfer (bytes) |
| `--all` | Sweep message size from 8B to 1MB |
| `--sweep-start-size` | Starting message size when using `--all` sweep |
| `--sweep-max-size` | Maximum message size when using `--all` sweep |
| `--all-batch` | Sweep batch size from 1 to 32768 |
| `--transfer-batch-size` | Number of consecutive transfers |
| `--enable-batch-transfer` | Enable batch transfer mode |
| `--enable-sess` | Enable session transfer (lower latency) |
| `--num-initiator-dev` | Number of initiator devices |
| `--num-target-dev` | Number of target devices |
| `--num-qp-per-transfer` | Number of queue pairs used (default `4`) |
| `--op-type` | Operation type: `read` or `write` |
| `--poll_cq_mode` | CQ polling mode: `polling` or `event` |
| `--num-worker-threads` | Number of worker threads (default `1`; ignored when chunking/multi-NIC is on — posting is single-thread inline) |
| `--disable-chunking` | Disable single-transfer chunking (chunking is **on by default**) |
| `--chunk-bytes` | Chunk size in bytes when chunking is on (default `65536` = 64 KB) |
| `--max-chunks` | Max chunks per transfer (default `64`) |
| `--log-level` | Log level (e.g., `info`) |

> Multi-NIC striping is controlled by the env var `MORI_IO_NUM_NICS_PER_TRANSFER` (default `1`). For host memory, set it to the number of NUMA-local NICs and keep `--num-qp-per-transfer ≥ 2 × NICs` so each NIC gets ≥2 QPs. GPU memory should stay single-NIC (PCIe-bound).

## Results: Thor2 RDMA Read

**Config:** 8 initiator + 8 target devices, session enabled, batch transfer, 2 QPs per transfer

```bash
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
    --master_addr="10.235.192.57" --master_port=1234 \
    tests/python/io/benchmark.py --host="10.235.192.60" \
    --all --enable-sess --enable-batch-transfer \
    --num-qp-per-transfer 2 --num-target-dev 8 --num-initiator-dev 8
```

```
+--------------------------------------------------------------------------------------------+
|                                      Initiator Rank 7                                      |
+-------------+----------------+---------------+---------------+--------------+--------------+
| MsgSize (B) | TotalSize (MB) | Max BW (GB/s) | Avg Bw (GB/s) | Min Lat (us) | Avg Lat (us) |
+-------------+----------------+---------------+---------------+--------------+--------------+
|      8      |      0.00      |      0.02     |      0.02     |    113.73    |    120.65    |
|      16     |      0.00      |      0.04     |      0.03     |    113.96    |    118.57    |
|      32     |      0.01      |      0.07     |      0.07     |    113.49    |    118.69    |
|      64     |      0.02      |      0.14     |      0.14     |    114.44    |    118.73    |
|     128     |      0.03      |      0.29     |      0.27     |    114.68    |    119.34    |
|     256     |      0.07      |      0.57     |      0.55     |    114.44    |    118.67    |
|     512     |      0.13      |      1.15     |      1.11     |    113.49    |    118.18    |
|     1024    |      0.26      |      2.30     |      2.23     |    114.20    |    117.78    |
|     2048    |      0.52      |      4.47     |      4.31     |    117.30    |    121.52    |
|     4096    |      1.05      |      8.31     |      8.01     |    126.12    |    130.95    |
|     8192    |      2.10      |     14.19     |     13.77     |    147.82    |    152.34    |
|    16384    |      4.19      |     22.16     |     21.56     |    189.30    |    194.58    |
|    32768    |      8.39      |     30.44     |     29.84     |    275.61    |    281.09    |
|    65536    |     16.78      |     37.51     |     36.55     |    447.27    |    458.96    |
|    131072   |     33.55      |     42.93     |     41.65     |    781.54    |    805.60    |
|    262144   |     67.11      |     45.66     |     45.08     |   1469.85    |   1488.69    |
|    524288   |     134.22     |     46.99     |     46.81     |   2856.02    |   2867.27    |
|   1048576   |     268.44     |     47.88     |     47.75     |   5605.94    |   5622.15    |
+-------------+----------------+---------------+---------------+--------------+--------------+
```

## Results: Thor2 RDMA Write

### Message Size Sweep

**Config:** 1 initiator + 1 target device, session enabled, batch transfer (128), 4 QPs, polling mode

```bash
numactl --cpunodebind=0 --membind=0 --physcpubind=0-47,96-143 \
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr="10.235.192.60" --master_port=1234 \
    tests/python/io/benchmark.py --host="10.235.192.60" \
    --enable-batch-transfer --enable-sess --buffer-size 1024 \
    --transfer-batch-size 128 --num-initiator-dev 1 --num-target-dev 1 \
    --num-qp-per-transfer 4 --all --num-worker-threads 1 \
    --log-level info --op-type write --poll_cq_mode polling
```

```
+--------------------------------------------------------------------------------------------------------+
|                                            Initiator Rank 0                                            |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
| MsgSize (B) | BatchSize | TotalSize (MB) | Max BW (GB/s) | Avg Bw (GB/s) | Min Lat (us) | Avg Lat (us) |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
|      8      |    128    |      0.00      |      0.03     |      0.03     |    33.38     |    36.33     |
|      16     |    128    |      0.00      |      0.06     |      0.06     |    34.09     |    36.35     |
|      32     |    128    |      0.00      |      0.12     |      0.11     |    34.57     |    36.33     |
|      64     |    128    |      0.01      |      0.24     |      0.23     |    33.62     |    36.33     |
|     128     |    128    |      0.02      |      0.49     |      0.45     |    33.62     |    36.49     |
|     256     |    128    |      0.03      |      0.94     |      0.89     |    34.81     |    36.99     |
|     512     |    128    |      0.07      |      1.86     |      1.77     |    35.29     |    37.01     |
|     1024    |    128    |      0.13      |      3.84     |      3.53     |    34.09     |    37.09     |
|     2048    |    128    |      0.26      |      7.33     |      6.96     |    35.76     |    37.65     |
|     4096    |    128    |      0.52      |     12.94     |     12.46     |    40.53     |    42.09     |
|     8192    |    128    |      1.05      |     20.75     |     20.12     |    50.54     |    52.11     |
|    16384    |    128    |      2.10      |     29.03     |     28.33     |    72.24     |    74.02     |
|    32768    |    128    |      4.19      |     36.50     |     35.91     |    114.92    |    116.81    |
|    65536    |    128    |      8.39      |     41.74     |     41.39     |    200.99    |    202.70    |
|    131072   |    128    |     16.78      |     45.14     |     44.85     |    371.69    |    374.10    |
|    262144   |    128    |     33.55      |     46.93     |     46.76     |    715.02    |    717.56    |
|    524288   |    128    |     67.11      |     47.94     |     47.81     |   1399.99    |   1403.64    |
|   1048576   |    128    |     134.22     |     48.44     |     48.32     |   2770.90    |   2777.76    |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
```

### Batch Size Sweep

**Config:** 1 initiator + 1 target device, 1024B messages, session enabled, 16 QPs, polling mode

```bash
numactl --cpunodebind=0 --membind=0 --physcpubind=0-47,96-143 \
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr="10.235.192.60" --master_port=1234 \
    tests/python/io/benchmark.py --host="10.235.192.60" \
    --enable-batch-transfer --enable-sess --buffer-size 1024 \
    --transfer-batch-size 128 --num-initiator-dev 1 --num-target-dev 1 \
    --num-qp-per-transfer 16 --all-batch --num-worker-threads 1 \
    --log-level info --op-type write --poll_cq_mode polling
```

```
+--------------------------------------------------------------------------------------------------------+
|                                            Initiator Rank 0                                            |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
| MsgSize (B) | BatchSize | TotalSize (MB) | Max BW (GB/s) | Avg Bw (GB/s) | Min Lat (us) | Avg Lat (us) |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
|     1024    |     1     |      0.00      |      0.10     |      0.09     |    10.25     |    11.10     |
|     1024    |     2     |      0.00      |      0.19     |      0.18     |    10.73     |    11.33     |
|     1024    |     4     |      0.00      |      0.34     |      0.33     |    12.16     |    12.60     |
|     1024    |     8     |      0.01      |      0.58     |      0.54     |    14.07     |    15.07     |
|     1024    |     16    |      0.02      |      0.83     |      0.72     |    19.79     |    22.87     |
|     1024    |     32    |      0.03      |      1.48     |      1.30     |    22.17     |    25.19     |
|     1024    |     64    |      0.07      |      2.45     |      2.14     |    26.70     |    30.61     |
|     1024    |    128    |      0.13      |      3.39     |      3.06     |    38.62     |    42.87     |
|     1024    |    256    |      0.26      |      4.43     |      4.19     |    59.13     |    62.53     |
|     1024    |    512    |      0.52      |      5.08     |      4.84     |    103.24    |    108.37    |
|     1024    |    1024   |      1.05      |      5.79     |      5.43     |    181.20    |    192.99    |
|     1024    |    2048   |      2.10      |      6.47     |      6.24     |    324.01    |    336.00    |
|     1024    |    4096   |      4.19      |      7.00     |      6.85     |    599.15    |    611.94    |
|     1024    |    8192   |      8.39      |      7.15     |      6.76     |   1173.02    |   1241.83    |
|     1024    |   16384   |     16.78      |      7.29     |      7.02     |   2301.69    |   2390.62    |
|     1024    |   32768   |     33.55      |      7.32     |      7.27     |   4585.50    |   4617.83    |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
```

## Results: CX7 RDMA (Batch Size = 1)

### Write

**Config:** 1 initiator + 1 target device, single transfer, session enabled, message sweep 1KB-64MB

```bash
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr="10.194.132.29" --master_port=1234 \
    tests/python/io/benchmark.py --host="10.194.132.29" \
    --transfer-batch-size 1 --all --sweep-start-size=1024 \
    --sweep-max-size=67108864 --op-type write \
    --enable-sess --enable-batch-transfer
```

```
+--------------------------------------------------------------------------------------------------------+
|                                            Initiator Rank 0                                            |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
| MsgSize (B) | BatchSize | TotalSize (MB) | Max BW (GB/s) | Avg Bw (GB/s) | Min Lat (us) | Avg Lat (us) |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
|     1024    |     1     |      0.00      |      0.20     |      0.17     |     5.25     |     5.87     |
|     2048    |     1     |      0.00      |      0.37     |      0.35     |     5.48     |     5.81     |
|     4096    |     1     |      0.00      |      0.75     |      0.70     |     5.48     |     5.88     |
|     8192    |     1     |      0.01      |      1.49     |      1.37     |     5.48     |     5.96     |
|    16384    |     1     |      0.02      |      2.86     |      2.68     |     5.72     |     6.11     |
|    32768    |     1     |      0.03      |      5.29     |      4.94     |     6.20     |     6.64     |
|    65536    |     1     |      0.07      |     10.18     |      9.11     |     6.44     |     7.20     |
|    131072   |     1     |      0.13      |     16.17     |     15.44     |     8.11     |     8.49     |
|    262144   |     1     |      0.26      |     24.43     |     23.77     |    10.73     |    11.03     |
|    524288   |     1     |      0.52      |     32.82     |     31.97     |    15.97     |    16.40     |
|   1048576   |     1     |      1.05      |     39.62     |     38.91     |    26.46     |    26.95     |
|   2097152   |     1     |      2.10      |     43.98     |     43.56     |    47.68     |    48.15     |
|   4194304   |     1     |      4.19      |     46.54     |     46.37     |    90.12     |    90.44     |
|   8388608   |     1     |      8.39      |     48.00     |     47.89     |    174.76    |    175.18    |
|   16777216  |     1     |     16.78      |     48.77     |     48.71     |    344.04    |    344.46    |
|   33554432  |     1     |     33.55      |     49.17     |     49.13     |    682.35    |    683.02    |
|   67108864  |     1     |     67.11      |     49.36     |     49.34     |   1359.46    |   1360.09    |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
```

### Read

**Config:** Same as write, but with `--op-type read`

```bash
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr="10.194.132.29" --master_port=1234 \
    tests/python/io/benchmark.py --host="10.194.132.29" \
    --transfer-batch-size 1 --all --sweep-start-size=1024 \
    --sweep-max-size=67108864 --op-type read \
    --enable-sess --enable-batch-transfer
```

```
+--------------------------------------------------------------------------------------------------------+
|                                            Initiator Rank 0                                            |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
| MsgSize (B) | BatchSize | TotalSize (MB) | Max BW (GB/s) | Avg Bw (GB/s) | Min Lat (us) | Avg Lat (us) |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
|     1024    |     1     |      0.00      |      0.17     |      0.15     |     5.96     |     6.74     |
|     2048    |     1     |      0.00      |      0.34     |      0.31     |     5.96     |     6.63     |
|     4096    |     1     |      0.00      |      0.69     |      0.63     |     5.96     |     6.49     |
|     8192    |     1     |      0.01      |      1.32     |      1.23     |     6.20     |     6.66     |
|    16384    |     1     |      0.02      |      2.55     |      2.35     |     6.44     |     6.96     |
|    32768    |     1     |      0.03      |      4.58     |      4.40     |     7.15     |     7.44     |
|    65536    |     1     |      0.07      |      8.33     |      7.96     |     7.87     |     8.23     |
|    131072   |     1     |      0.13      |     14.47     |     13.57     |     9.06     |     9.66     |
|    262144   |     1     |      0.26      |     21.56     |     20.97     |    12.16     |    12.50     |
|    524288   |     1     |      0.52      |     29.32     |     28.63     |    17.88     |    18.31     |
|   1048576   |     1     |      1.05      |     35.47     |     34.95     |    29.56     |    30.00     |
|   2097152   |     1     |      2.10      |     39.62     |     39.38     |    52.93     |    53.25     |
|   4194304   |     1     |      4.19      |     42.09     |     41.95     |    99.66     |    99.99     |
|   8388608   |     1     |      8.39      |     43.49     |     43.38     |    192.88    |    193.39    |
|   16777216  |     1     |     16.78      |     44.23     |     44.17     |    379.32    |    379.83    |
|   33554432  |     1     |     33.55      |     44.59     |     44.57     |    752.45    |    752.90    |
|   67108864  |     1     |     67.11      |     44.81     |     44.78     |   1497.51    |   1498.58    |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
```

## Benchmark Tuning Guide

This section is a practical starting point for maximizing **single-NIC internode
RDMA write bandwidth** with large messages and batched transfers. Latency-bound
or single-transfer workloads may prefer different settings.

### Recommended starting config

These workload flags apply to both engines (the C++ default and Python); only the
launch wrapper differs. Shown against the C++ binary:

```bash
tests/cpp/io/bench_engine --op write \
    --enable-sess --enable-batch-transfer --batch-contiguous \
    --disable-chunking \
    --num-qp-per-transfer 8 --num-worker-threads 8 \
    --transfer-batch-size 128 --num-initiator-dev 1 --num-target-dev 1
    # (plus --rank/--master-ip/--self-ip/--port for the two-node rendezvous)
```

On a 400G-class NIC this typically reaches ~95%+ of link rate across the
1 MiB–32 MiB message range and removes the mid-size bandwidth dip seen with the
defaults. Tune from here.

### Parameters that help

| Knob | Effect |
|------|--------|
| `--enable-batch-transfer --enable-sess` | Submit the whole batch through a persistent session, cutting per-transfer overhead. |
| `--batch-contiguous` | Contiguous per-transfer offsets let consecutive transfers merge into fewer, larger RDMA work requests (WRs). This is usually the biggest single win at large sizes and flattens the message-size curve. Do **not** use it when you intend to stress the send queue / keep one WR per transfer. |
| `--num-qp-per-transfer` | More QPs pipeline a large transfer across the NIC. Bandwidth generally scales up to ~8 QPs and then plateaus; 16+ gives no further gain. Start at 8. |
| `--disable-chunking` | Chunking splits each transfer into `--chunk-bytes` pieces; for large messages this multiplies WRs and can saturate the send queue, collapsing bandwidth at the largest sizes. Turning it off avoids that — but read the WR-size limit below. |
| `--num-worker-threads` | With chunking **off** + `--batch-contiguous`, use several workers (e.g., 8). Besides parallel posting, this splits a large batch across workers so each merged WR stays under the NIC message-size limit. With chunking on, this knob has no bandwidth effect (posting is single-thread inline). |

### The merged-WR size limit

A single RDMA WR cannot exceed the NIC port's reported `max_msg_sz` (`ibv_port_attr`,
**2 GiB** on the usual mlx5 and `bnxt_re` parts), taken as the minimum across the
endpoints a transfer uses.
With `--batch-contiguous --disable-chunking` and a **single** worker, an entire
batch can merge into one WR of `message_size × transfer_batch_size`. If that
exceeds `max_msg_sz` the run aborts with:

```
merged RDMA WR ... exceeds local max_msg_sz ...; enable RDMA transfer chunking or reduce batch size
```

Any one of these avoids it:

- Use multiple `--num-worker-threads` (splits the batch so each merged WR is smaller) — recommended.
- Keep `message_size × transfer_batch_size ≤ max_msg_sz`.
- Re-enable chunking (drop `--disable-chunking`) so oversized transfers are split automatically.

### Parameters that usually do not help (or can hurt)

- **`MORI_IO_QP_MAX_SEND_WR` / `--max-send-wr`**: no benefit once chunking is off (merged WRs are few); leave at default.
- **`--max-msg-sge`**: raising it can *reduce* bandwidth; leave at default.
- **`MORI_RDMA_DEVICES` (manual NIC pinning)**: the backend's automatic NIC selection is rail/NUMA-aware. Overriding it can pin traffic to a non-local or cross-rail NIC and sharply reduce bandwidth. Only override if you have verified the topology and the auto-choice is wrong.
- **Lossless traffic class / service level (`MORI_IO_TC` / `MORI_IO_SL`)**: only helps if the fabric is actually configured for PFC on that priority. On a fabric that is not, setting these can sharply reduce bandwidth. Leave unset unless PFC is configured end-to-end.

### Host (CPU) vs GPU memory

- GPU device memory (registered via dmabuf) is unaffected by the notes below.
- For **host (CPU) memory**, on NICs with a small memory-translation (PTE) cache,
  a large 4 KiB-paged buffer used as the **RDMA-write source** can lose bandwidth
  past a few tens of MiB per transfer, and very large regions may fail memory
  registration outright. Back host buffers with **hugepages** (explicit hugetlb
  or THP) in that case. The read direction and the passive (target) side are
  less sensitive.
- Multi-NIC striping (`MORI_IO_NUM_NICS_PER_TRANSFER`) is intended for host
  memory; keep GPU memory on a single NIC (it is PCIe-bound).
