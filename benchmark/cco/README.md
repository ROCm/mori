# CCO point-to-point benchmarks

Four two-rank benchmarks that move data between PE 0 and PE 1 over one of the
CCO transports:

| binary | measures |
|---|---|
| `cco_p2p_put_bw` | write bandwidth, many ops per timed iteration |
| `cco_p2p_get_bw` | read bandwidth |
| `cco_p2p_put_latency` | write latency, one op + completion per iteration |
| `cco_p2p_get_latency` | read latency |

Transports are selected with `-t`:

- `lsa` — intra-node flat-VA loads/stores from the kernel. No copy engine.
- `sdma` — intra-node copy engine, driven from the kernel (`ccoSdma`).
- `ibgda` — cross-node one-sided RDMA.
- `proxy` — the same one-sided RDMA through `ccoProxy`: kernels push descriptors
  into a KIWI queue and host proxy threads post the ibverbs work requests. This
  needs a `BUILD_CCO_PROXY=ON` build and supports `put_bw` and `put_latency`
  only (see below).

Run `<binary> -h` for the full option list.

## Build

```bash
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARK=ON
ninja -C build cco_p2p_put_bw cco_p2p_get_bw cco_p2p_put_latency cco_p2p_get_latency
```

`BUILD_BENCHMARK=ON` implies `BUILD_CCO_SDMA=ON`; the binaries land in
`build/benchmark/`. `pip install .` with `BUILD_BENCHMARK=ON` also works and
additionally installs them under `python/mori/benchmarks/cco/`.

## Run

Always two ranks:

```bash
# SDMA bandwidth, 1MB..1GB
mpirun -np 2 ./build/benchmark/cco_p2p_put_bw -t sdma -b 1M -e 1G -f 4 -n 20 -w 5

# SDMA latency, small messages, engine-drain completion
mpirun -np 2 ./build/benchmark/cco_p2p_put_latency -t sdma -C quiet -b 8 -e 64K -n 100 -w 20

# Same link with the kernel doing the copy instead, for comparison
mpirun -np 2 ./build/benchmark/cco_p2p_put_bw -t lsa -b 1M -e 1G -f 4 -n 20 -w 5
```

## ccoProxy vs ccoGda

`-t proxy` runs the ibgda kernels' shape with `ccoProxy`. ibgda's `flush` maps
to ccoProxy's `flush`, which waits until the proxy has seen every earlier put
of the queue complete. Its DevComm has proxy QPs to the peers ibgda connects
(`proxyConnectionType`) and no GDA QPs.

- `cco_p2p_put_latency -t proxy`: one put + flush per iteration.
  - `-P` switches ibgda or proxy to a ping-pong: put + SignalInc, PE 1 echoes,
    and the reported value is round trip / 2.
  - `-S` is the same ping-pong with bare signals.
- `cco_p2p_put_bw -t proxy`: ibgda's split. Each unit (block, wavefront with
  `-s warp`, or thread with `-s thread`, `-s thread_agg` or `-s lane`) puts its
  slice every iteration, then does one flush per launch, as ibgda does with
  `flush`.
  - `-s thread` uses `ccoCoopThread` and `-s thread_agg` adds
    `ccoGdaThreadAggregate`; both need `-q mpsc`.
  - `-s lane` uses `ccoCoopLaneMasked`: each wavefront pushes its 64 lanes'
    puts in one push, on SPSC or MPSC queues.
  - `-q spsc` (default) gives every block (`-s block`) or wavefront its own
    SPSC queue; `-q mpsc` shares one MPSC queue per block.
  - `-Q` sets the number of host proxy threads (default 4), `-K` the RC QPs
    per (proxy thread, peer) (default 4).
- Message rate: run small sizes with `-s warp`, `-s lane` or `-s thread` and
  many wavefronts. The `Mpps` column gives the per-message issue rate.

```bash
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARK=ON -DBUILD_CCO_PROXY=ON
ninja -C build cco_p2p_put_bw cco_p2p_put_latency test_proxy_put

# put + flush latency (proxy vs ibgda)
mpirun -np 2 ./build/benchmark/cco_p2p_put_latency -t proxy -s block -b 8 -e 1M -n 1000 -w 50
mpirun -np 2 ./build/benchmark/cco_p2p_put_latency -t ibgda -s block -b 8 -e 1M -n 1000 -w 50
# put + signal ping-pong, and signal-only ping-pong
mpirun -np 2 ./build/benchmark/cco_p2p_put_latency -t proxy -P -s warp -b 8 -e 1M -n 1000 -w 50
mpirun -np 2 ./build/benchmark/cco_p2p_put_latency -t proxy -S -s warp -n 1000 -w 50
# bandwidth: 32 blocks, 4 proxy threads
mpirun -np 2 ./build/benchmark/cco_p2p_put_bw -t proxy -Q 4 -c 32 -T 256 -s block -b 64K -e 256M -n 200
# message rate: 32 blocks x 4 wavefronts = 128 SPSC queues, 8 B..4 KiB per message
mpirun -np 2 ./build/benchmark/cco_p2p_put_bw -t proxy -Q 4 -c 32 -T 256 -s warp -b 1K -e 512K -n 2000
mpirun -np 2 ./build/benchmark/cco_p2p_put_bw -t proxy -q mpsc -Q 4 -c 32 -T 256 -s warp -b 1K -e 512K -n 2000
# per-thread puts, 8 B..1 KiB per message: lane-masked (64 per wavefront push), then ccoCoopThread
mpirun -np 2 ./build/benchmark/cco_p2p_put_bw -t proxy -Q 4 -c 32 -T 256 -s lane -b 64K -e 8M -n 200
mpirun -np 2 ./build/benchmark/cco_p2p_put_bw -t proxy -q mpsc -Q 4 -c 32 -T 256 -s thread -b 64K -e 8M -n 200
```

## Triton benchmark

The Triton implementation mirrors the C++ two-window, rank-0-only and
completion boundaries. It reports one `RESULT_JSON` record per size plus a
human-readable table.

```bash
# LSA put latency: 8B..64KB
torchrun --standalone --nproc_per_node=2 \
  benchmark/cco/triton/bench_p2p.py \
  --transport lsa --op put --metric latency

# SDMA get bandwidth: 64KB..8MB
MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=2 \
  benchmark/cco/triton/bench_p2p.py \
  --transport sdma --op get --metric bandwidth

# IBGDA on one AINIC rail
MORI_DEVICE_NIC=ionic MORI_DISABLE_TOPO=1 MORI_RDMA_DEVICES=rocep9s0 \
  torchrun --standalone --nproc_per_node=2 \
  benchmark/cco/triton/bench_p2p.py \
  --transport ibgda --op put --metric bandwidth
```

Common options are `--min-size`, `--max-size`, `--step-factor`, `--iters`,
`--warmup`, `--grid`, `--threads`, and `--json-out`. The current fair-comparison
preset uses block scope: latency and SDMA bandwidth use one program, while
LSA/IBGDA bandwidth use 32 programs × 256 threads.

Triton JIT compilation and warmup occur before the timed event. Each timed
launch loops `iters` times inside the kernel, so Python launch overhead is not
reported as device latency.

## Automated Triton vs C++ comparison

Build the four C++ binaries, then run:

```bash
BUILD_BENCHMARK=ON BUILD_CCO_SDMA=ON BUILD_UMBP=OFF pip install -e .

MORI_CCO_TRITON_GDA_DEVICE=rocep9s0 \
  python benchmark/cco/compare_triton_cpp.py \
  --cpp-dir build/benchmark \
  --json-out /tmp/cco_compare.json \
  --markdown-out /tmp/cco_compare.md
```

The runner executes LSA, SDMA and IBGDA put/get latency and bandwidth with
identical sizes, block scope, geometry, iterations and warmup. `ratio` is
`Triton/C++`: greater than one is better for bandwidth and worse for latency.

The IBGDA run pins all ranks to one HCA because this MI355X setup cannot route
local GDA traffic across AINIC rails. This is a correctness configuration, not
a multi-rail peak-bandwidth measurement.

## SDMA specifics

`MORI_SDMA_NUM_CHANNELS` sets the queues per GPU pair — default 2, max 8. It is
not a benchmark flag but it bounds how much of the link one run can use, so set
it before comparing configurations:

```bash
export MORI_SDMA_NUM_CHANNELS=8
```

`-c` (grid) and `-T` (threads per block) select how many *issue units* split the
transfer. A unit is a thread, a wavefront or a block, per `-s`. Queues are
round-robined over the units, so only `min(units, MORI_SDMA_NUM_CHANNELS)` are
actually used, and beyond that units share a queue. Omit both flags to get the
historical geometry: one block, and one unit per queue for `-s thread`.

Bandwidth is governed mostly by the size of each individual copy, i.e.
`message / units`. Splitting a transfer over more units than the link needs
makes each op smaller and costs bandwidth; it also serialises on the queues once
units exceed the channel count.

## Expected magnitudes

Rough intra-node put figures, to tell a healthy run from a broken one:

| | gfx950 (MI355X) | gfx1250 |
|---|---|---|
| SDMA peak, `*_bw` | ~60 GB/s | ~1600 GB/s |
| SDMA 1KB round trip, `*_latency` | ~6 us | ~8 us |
| SDMA 1KB issue interval, `size / *_bw` | ~3 us | ~4 us |

The two latency rows differ because of where the completion wait sits. The
latency benchmarks quiet after every put, so they report a full round trip. The
bandwidth benchmarks issue the whole iteration count and quiet once at the end,
so dividing the message size by their result gives the pipelined issue interval
instead. Both are useful; quote which one you mean.

Bandwidth differs by more than an order of magnitude between the two because of
the interconnect, not the software. On gfx950 a single queue saturates the link
at a few MB; on gfx1250 the link is fast enough that a single queue needs a much
larger message, and several queues help in the middle of the range.

Scope barely moves latency -- warp and block issue from a leader, so the extra
threads do not shorten the path.

## Measurement notes

- Run one benchmark at a time. Concurrent runs contend for the same GPUs and the
  numbers are meaningless.
- Do a throwaway run first. The first timed run after a cold link reads low.
- Repeat and take a median. Run-to-run spread is well under 1% for large
  messages but reaches double digits in the steep part of the curve.
- Check the machine is otherwise idle (`rocm-smi --showpids`).
- C++ LSA bandwidth uses a HIP system fence and cross-block sense barrier.
  Triton mirrors both through a CCO fence wrapper and a small fully-resident
  atomic barrier, but generated instruction scheduling is not identical.
- The comparison intentionally fixes scope to `block`; a Triton scalar extern
  call is executed by every hardware lane and is not equivalent to CCO's
  one-thread benchmark mode.
