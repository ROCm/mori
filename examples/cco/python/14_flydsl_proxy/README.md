# CCO Example 14 — FlyDSL ccoProxy (host-proxy RDMA)

Device-initiated RDMA from a FlyDSL `@flyc.kernel` with no GDA QP: the kernel
pushes requests into D2H queues through the `Proxy` handle of
`mori.cco.device.flydsl`, and a host thread posts them to the NIC.

## What it shows
- A proxy-only DevComm: `proxy_only = 1`, FULL connection, one SPSC queue per
  block (`proxy_spsc_queue_count`), one shared MPSC queue
  (`proxy_mpsc_queue_count`), signal slots from `gda_signal_count`.
- `cco.DevComm(dev_comm).proxy(pool=...)` and its ops, each taking the queue at
  run time (`queue=` the block index for SPSC, 0 for the MPSC queue):
  - `put` with `SignalOp.SET` (an RDMA write, no NIC atomic) and `SignalOp.INC`,
    three rounds of put, `flush`, `wait_signal` per block; the INC launch
    continues the SPSC queues the SET launch left off;
  - the same with `CoopScope.WARP` on one-wavefront blocks;
  - every block on the one MPSC queue;
  - `put_value`, plain and with a SET signal;
  - a bare ADD `signal`, `flush_timeout` and `failed()`;
  - an INC nobody waits for, then `read_signal` / `reset_signal` through both
    pools.
- Both ranks send to each other; the host checks the received data and the
  values the kernels stored, and prints one `PASS` line per check.

## Prerequisites
- A wave64 gfx942 / gfx950 GPU per rank and `mori` built with the ccoProxy
  host engine.
- FlyDSL and `mpi4py`; RDMA NICs the GPUs reach.

## Run
```
MORI_RDMA_DEVICES=mlx5_0,mlx5_1 MORI_SOCKET_IFNAME=lo \
  timeout 300 mpirun -np 2 python3 examples/cco/python/14_flydsl_proxy/main.py
```
Each rank uses its own local NIC from `MORI_RDMA_DEVICES`; keep the topology
detection on. Rank 0 prints `All FlyDSL ccoProxy tests PASSED` at the end. The
signal waits have no timeout, so a dead peer hangs the run; `timeout` bounds it.
`tests/python/cco/test_flydsl_proxy.py` runs the same command.
