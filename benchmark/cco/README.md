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
