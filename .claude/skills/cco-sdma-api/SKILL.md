---
name: cco-sdma-api
description: >-
  Write GPU kernels that move data with the CCO SDMA device API (ccoSdma:
  put/get/quiet/waitSignal/commit). Covers the build flags and env vars needed to
  turn the path on, which template arguments and coop scope to pick, how
  completion and signals work, and the failure modes. Use when the user is
  writing or debugging a kernel that calls ccoSdma, asks how to do intra-node
  copies with the copy engine, or hits a hang or wrong data on the SDMA path.
---

# Using the CCO SDMA device API

`ccoSdma` drives the GPU's SDMA copy engines from inside a kernel: intra-node
peer-to-peer copies that do not occupy CUs. It is a device-side API in
`include/mori/cco/cco.hpp` — one header, no separate library call per op.

Only for peers on the same node (LSA). Cross-node goes through `ccoGda`.

Every number quoted below was measured on MI355X (gfx950), two ranks over
intra-node xGMI. Treat them as shape, not as a spec.

## Turning it on

Build: SDMA is compiled out unless both are set.

```
cmake -B build -DBUILD_CCO=ON -DBUILD_CCO_SDMA=ON -DGPU_TARGETS=gfx950
```

Kernel TUs that include `cco.hpp` need `-DBUILD_CCO_SDMA=1` too, or `ccoSdma`
will not be declared.

Runtime:

| var | meaning |
|---|---|
| `MORI_ENABLE_SDMA=1` | **required** — without it the comm has no SDMA queues and every call is a no-op |
| `MORI_SOCKET_IFNAME` | interface for the bootstrap rendezvous |
| `MORI_CCO_SDMA_DEBUG` | compile-time (`-DMORI_CCO_SDMA_DEBUG`): turns the misuse checks below into traps. Off in release, zero cost |

Queue count is a property of the comm, not the environment: set
`reqs.sdmaQueueCount` before `ccoDevCommCreate`. `MORI_SDMA_NUM_CHANNELS` is only
the fallback when it is left at 0. Default 2, hardware cap 8 (clamped with a
warning above that).

The count is fixed by the **first** `ccoDevCommCreate` on a comm; later ones with
a different value warn and keep the original.

## Host setup

```cpp
ccoCommCreate(uid, nRanks, rank, vmmSize, &comm);
ccoMemAlloc(comm, bytes, &buf);              // both src and dst
ccoWindowRegister(comm, buf, bytes, &win);   // both src and dst

ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;   // SDMA needs no GDA
reqs.gdaContextCount = reqs.gdaSignalCount = reqs.gdaCounterCount = 0;
reqs.sdmaQueueCount = 8;                            // 0 = env / default 2
ccoDevComm devComm{};
ccoDevCommCreate(comm, &reqs, &devComm);

if (devComm.sdma.sdmaNumQueue == 0) { /* SDMA unavailable — MORI_ENABLE_SDMA? */ }
```

Pass `devComm` to the kernel by value; construct `ccoSdma sdma{devComm}` inside.

## The API

Everything hangs off `ccoSdma sdma{devComm}`, constructed per kernel.

| call | what it does |
|---|---|
| `put(peer, dstWin, dstOff, srcWin, srcOff, bytes, queueId)` | copy local → peer |
| `get(peer, dstWin, dstOff, srcWin, srcOff, bytes, queueId)` | copy peer → local |
| `commit<Coop>(peer, queueId)` | ring the doorbell for `Aggregate`-posted ops |
| `quietQueue<Coop>(peer, queueId)` | wait until one queue has drained |
| `quiet<Coop>(peer)` | wait until every queue toward `peer` has drained |
| `waitSignal(srcRank, queueId, expected)` | wait for a signal counter to reach `expected` |

`put`/`get` are non-blocking: they place a packet and (unless `Aggregate`) ring
the doorbell. Nothing has happened when they return — you must drain.

All indices are **LSA ranks** (`devComm.lsaRank`, `devComm.lsaSize`), never world
ranks. `queueId` must be `< devComm.sdma.sdmaNumQueue`; out of range is a silent
no-op.

## put / get

```cpp
template <typename Coop            = ccoCoopThread,
          bool localSignal         = false,
          bool remoteSignal        = false,
          uint32_t optFlags        = ccoSdmaOptFlagsDefault,
          ccoSdmaThreadMode mode   = ccoSdmaThreadIndependent>
void put(int peer, ccoWindow_t dstWin, size_t dstOffset,
         ccoWindow_t srcWin, size_t srcOffset, size_t bytes, int queueId = 0);
```

`peer` is an **LSA rank** (`devComm.lsaRank` is yours, `devComm.lsaSize` the
count) — not the world rank. They only coincide on a single node.

`get` is the same with the direction reversed.

### Coop scope

`ccoCoopThread` (default), `ccoCoopWarp`, `ccoCoopBlock`. Warp and block are
leader-only: lane 0 does the whole copy, the rest return. They generate
identical code to thread scope and measure the same — pick whichever matches how
your kernel is structured, not for speed.

Use warp/block scope when the surrounding code is warp- or block-collective and
you want one op per group without writing the `if (lane == 0)` yourself.

### ccoSdmaThreadMode

Only meaningful for `ccoCoopThread` (static_assert enforces it), and only when
several lanes of one wave call `put` at once.

| mode | when | cost |
|---|---|---|
| `ccoSdmaThreadSameQueue` | **every active lane has the same `peer` and `queueId`** — prefer this | +5 SGPR; cost barely moves from 1 to 64 lanes |
| `ccoSdmaThreadIndependent` (default) | lanes target different peers, or you do not know | detects the shape at runtime; +14 SGPR |

**Reach for `SameQueue` first.** Lanes sharing a queue post as one group — one
reservation, one doorbell — so a whole wave's worth of packets costs about what
one costs. Measured with one warp, one packet per lane:

| lanes | spread over 8 queues | one queue, `SameQueue` |
|---|---|---|
| 1 | 2330 | 2315 |
| 2 | 3009 | **2592** |
| 4 | 3379 | **2629** |
| 8 | 3377 | **2612** |

Spreading lanes over queues is *worse*, not better: distinct queues avoid the
commit chain but each lane still issues its own uncached reservation and
doorbell, and those queue up in the memory pipeline. Queues buy parallelism
between *warps*, not between lanes of one warp.

Use `Independent` when the shape forces it — one lane per peer in an alltoall,
say. Lanes on distinct queues then post in parallel, and any that do share a
queue are grouped for you.

Breaking the `SameQueue` promise makes lanes write slots reserved on another
lane's queue. Silent corruption in release; traps under `MORI_CCO_SDMA_DEBUG`.

### optFlags

`ccoSdmaOptFlagsDefault` rings the doorbell per put.
`ccoSdmaOptFlagsAggregate` posts without ringing — issue N puts, then one
`commit(peer, queueId)` rings the batch. Amortises the doorbell; sweet spot
around N = 8–16.

Must be a template argument (it is one). Note the two unrelated "aggregate"
names: `ccoSdmaOptFlagsAggregate` batches doorbells, `ccoSdmaThreadSameQueue`
groups lanes.

## Completion

Three ways to know an op is done. All are **sender-side**: they tell you the
engine consumed the command, not that a receiver observed the data.

```cpp
sdma.put(peer, dst, 0, src, 0, bytes, q);
sdma.quietQueue(peer, q);          // our send is out; src is reusable
```

`quiet`/`quietQueue` take a Coop like `put` does, and it should match the scope
the surrounding code runs at — at warp/block scope only the leader polls, and the
call syncs the group before returning, so the completion is visible to everyone.

| | use when | notes |
|---|---|---|
| `quietQueue<Coop>(peer, q)` | you used one queue | cheapest drain; warp/block poll from the leader only |
| `quiet<Coop>(peer)` | you used several queues | one queue per lane at warp/block scope; each idle queue still costs one uncached read |
| `localSignal` + `waitSignal` | you want the last ~0.1–0.6 µs, or partial completion | you maintain `expected` |

`waitSignal` is slightly faster than `quiet` at every size (0.24 µs at 8B, 0.62
at 64KB) because the signal slot is ordinary device memory while the
queue read pointer is not. `quiet` needs no bookkeeping — prefer it unless the
difference matters.

Do not reuse a source buffer before the op completes.

`commit` only matters with `ccoSdmaOptFlagsAggregate`: it rings what was posted
but not yet rung. `quiet` drains to the last *rung* packet, so an aggregate batch
that was never committed makes `quiet` return with nothing sent.

```cpp
for (int i = 0; i < 8; i++)
  sdma.put<ccoCoopThread, false, false, ccoSdmaOptFlagsAggregate>(
      peer, dst, i * chunk, src, i * chunk, chunk, q);
sdma.commit(peer, q);       // one doorbell for all eight
sdma.quietQueue(peer, q);
```

## Signals

A put can fire trailing atomics in addition to the copy:

- `localSignal` — increments **your own** `signalBuf[myLsaRank * n + q]`
- `remoteSignal` — increments the **peer's** `signalBuf[myLsaRank * n + q]`

Slots are indexed by the **sender**, so your own local signals and a peer's
remote signals land in different slots and never collide.

```cpp
// sender: notify the peer that the data landed
sdma.put<ccoCoopThread, false, true>(peer, dstWin, off, srcWin, off, bytes, q);

// receiver: no barrier needed — the signal alone proves the data arrived
sdma.waitSignal(senderLsaRank, q, expected);
```

The ATOMIC follows the COPY on the same queue over the same link, so seeing the
signal means the data is there.

`waitSignal` rules:

- `expected` is **yours to maintain**, monotonic, per `(srcRank, queueId)` pair.
  Slots are never reset. One running total per pair — a single counter across
  queues will wait forever.
- `srcRank` is an LSA rank. To wait on your own `localSignal`, pass
  `comm.lsaRank`, **not** `comm.rank`.
- The comparison is `>=`, so you cannot miss an increment.

## Queues and issuers

The queue is identified by `(peer, queueId)` — different peers are different
queues even at the same `queueId`.

**Keep concurrent issuing *warps* per queue at or below one.** Several warps
posting to the same queue is correct but serialises linearly on the in-order
commit chain (8 warps on one queue cost ~3.2x the same 8 spread over 8 queues).
Raise `reqs.sdmaQueueCount` so every issuing warp gets its own queue.

This is about warps, not lanes: lanes of one warp should share a queue and use
`SameQueue`. The two combine — a warp aggregates its lanes into one reservation,
and each warp owns a queue.

Queues do not make a single transfer faster: one peer is one xGMI link, so
splitting a copy across queues gains nothing. They exist to let independent
issuers proceed in parallel.

## Transfer size

Bandwidth is a function of **bytes per op** and nothing else. Scope, direction
and queue count do not enter into it:

| bytes per op | GB/s |
|---|---|
| 64 KB | 17 |
| 128 KB | 26 |
| 256 KB | 36 |
| 512 KB | 46 |
| 1 MB | 52 |
| 2 MB | 56 |
| 4 MB | 59 |
| 8 MB | 60 |

`put` and `get` land within 1% of each other, and `thread`/`warp`/`block` are
indistinguishable at equal bytes per op. Half of the achievable bandwidth is
already gone at 512 KB, and below ~256 KB the fixed ~6 µs dispatch cost
dominates whatever the copy itself does.

So the only lever for bandwidth is **not splitting the transfer**. Two ops of
4 MB reach 57 GB/s where one op of 8 MB reaches 60, even on separate queues —
splitting always costs, in proportion to how small the pieces get. Issue the
largest contiguous copy you have and let one engine drain it.

Below ~256 KB you are latency-bound, not bandwidth-bound; that is the regime
where warp aggregation and `SameQueue` matter and transfer size does not.

## Failure modes

| symptom | cause |
|---|---|
| every call silently does nothing | `MORI_ENABLE_SDMA` not set, or `sdmaNumQueue == 0` |
| `ccoSdma` undeclared | kernel TU missing `-DBUILD_CCO_SDMA=1` |
| `waitSignal` never returns | `expected` shared across queues, or `comm.rank` passed where `comm.lsaRank` was meant |
| `quiet` returns but data is stale | aggregate puts never `commit`ed — `quiet` drains to the last *rung* packet. Traps under `MORI_CCO_SDMA_DEBUG` |
| wrong data with several lanes per wave | `ccoSdmaThreadSameQueue` set while lanes target different queues |
| receiver sees a signal but stale data | signal fired on a different queue than the copy |

Build with `-DMORI_CCO_SDMA_DEBUG` when any of these appear: it turns the
misuse checks into traps at the offending call instead of a hang or corruption.

## Worked example

Each lane of the warp ships its own chunk to the same peer — the shape
`SameQueue` is for.

```cpp
__global__ void ship(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin, size_t chunk,
                     ccoDevComm devComm, int peer) {
  ccoSdma sdma{devComm};
  const int lane = threadIdx.x;                 // one queue per warp
  const int q = (threadIdx.x / 64) % devComm.sdma.sdmaNumQueue;

  sdma.put<ccoCoopThread, false, false, ccoSdmaOptFlagsDefault, ccoSdmaThreadSameQueue>(
      peer, reinterpret_cast<ccoWindow_t>(recvWin), lane * chunk,
      reinterpret_cast<ccoWindow_t>(sendWin), lane * chunk, chunk, q);

  sdma.quietQueue(peer, q);
}
```

Every lane of the warp has the same `peer` and `q`, so the promise holds: one
reservation for the whole warp, one doorbell.

### When the shape forces Independent

One lane per peer, one signal to the peer, no barrier on the receive side.

```cpp
__global__ void alltoall(ccoWindowDevice* sendWin, ccoWindowDevice* recvWin,
                         size_t bytes, ccoDevComm devComm, uint64_t expected) {
  ccoSdma sdma{devComm};
  const int me = devComm.lsaRank, n = devComm.lsaSize;
  const int p = threadIdx.x;
  if (p >= n || p == me) return;

  // lanes target distinct peers -> distinct queues -> posted in parallel
  sdma.put<ccoCoopThread, false, /*remoteSignal=*/true>(
      p, reinterpret_cast<ccoWindow_t>(recvWin), me * bytes,
      reinterpret_cast<ccoWindow_t>(sendWin), p * bytes, bytes, /*queueId=*/0);

  sdma.quiet(p);                      // our sends are out
  sdma.waitSignal(p, 0, expected);    // peer p's data has landed
}
```
