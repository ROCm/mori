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
| `putBroadcast(peer, dstWin, dstOff, peer2, dstWin2, dstOff2, srcWin, srcOff, bytes, queueId)` | one local source to two destinations in a single packet |
| `postWaitMem(peer, queueId, win, off, value, cmp, mask)` | park a queue until a local flag holds — the engine polls, not a CU |
| `postWaitSignal(peer, queueId, srcRank, sigQueueId, expected)` | `postWaitMem` on a signal slot |
| `postWaitMemPutPtr<l, r>(peer, queueId, win, off, value, cmp, mask, dst, src, bytes)` | `postWaitMem` + `putPtr` in one reservation and one doorbell |

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
reservation, one doorbell — so the issue cost grows very slowly with the number
of lanes. Trigger cycles for one warp, one packet per lane:

| lanes | one queue, `SameQueue` | per packet | spread over 8 queues |
|---|---|---|---|
| 1 | 2244 | 2244 | 2330 |
| 2 | 2508 | 1254 | 3009 |
| 4 | 2624 | 656 | 3379 |
| 8 | 2652 | 332 | 3377 |
| 16 | 2796 | 175 | — |
| 32 | 2916 | 91 | — |
| 64 | **3220** | **50** | — |

64 packets cost 43% more to issue than one, so per packet it is 45x cheaper.

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

A bitmask, and a template argument. Two bits:

`ccoSdmaOptFlagsAggregate` posts without ringing the doorbell — issue N puts,
then one `commit(peer, queueId)` rings the batch. Amortises the doorbell; sweet
spot around N = 8–16. A batch must stay below the 256 KB ring: at most
256 KB − 4 B on a queue before `commit` (9362 single-copy puts of 28 B, without
signals). The put that would fill the ring waits for the engine, which an
uncommitted batch never starts, so it hangs.

`ccoSdmaOptFlagsSignalPerCopy` gives every copy its own signal packet. **You
rarely want this.** By default a group of lanes signals once, after all of its
copies, so `expected` advances by one per `put()` call whatever the scope and
however many lanes joined. Per-copy signals make it advance by the lane count
instead, and cost a 32 B ATOMIC packet each, which the engine retires like any
other packet. Set it only if a
receiver has to observe copies landing one at a time. No effect at warp or block
scope, which post a single copy either way.

Note the two unrelated "aggregate" names: `ccoSdmaOptFlagsAggregate` batches
doorbells, `ccoSdmaThreadSameQueue` groups lanes.

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
- One `put()` call advances the counter by **one**, no matter how many lanes
  joined the group, unless the sender set `ccoSdmaOptFlagsSignalPerCopy`.
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

Back-to-back puts, `cco_p2p_put_bw -t sdma -s thread`, 2 ranks, one link:

| bytes per op | GB/s |
|---|---|
| 32 KB | 35 |
| 64 KB | 50 |
| 128 KB | 53 |
| 256 KB | 54 |
| 512 KB | 55 |
| 1 MB | 56 |
| 2 MB | 57 |
| 4 MB | 59 |

The link is close to saturated from about 64 KB per op. `put` and `get` land
within 1% of each other, and `thread`/`warp`/`block` are indistinguishable at
equal bytes per op. A single put, launch to drain, takes about 5.1–5.4 µs up to
16 KB (quiet) and 5.7–6.1 µs with a signal; that latency is what small
standalone transfers pay.

### Many small copies

The engine retires a packet in about **0.1 µs** plus its data time: on MI355X,
one queue runs about 10.6M 64 B packets/s, and 7 KB packets reach 54–58 GB/s per
link, the same as large copies. With 7 peers each on its own queue, 7 KB packets
moved about 370–380 GB/s per rank.

So scattering is cheap: one packet per row is fine even for rows of a few KB,
and there is no need to pack non-contiguous data first for bandwidth. Issuing
is cheap too: a wave posts a 64-packet `SameQueue` group at about 30 ns per
packet.

Earlier versions of this file quoted ~2 µs per packet. That was not the
engine: cco used to pad every COPY to 32 B with a NOP dword, and each NOP costs
the engine about as much as a small COPY. Since 2026-11 the ring holds packets
back to back with no padding.

## Engine-side wait and broadcast

Two more packets, with the gfx9 layouts. Validated on gfx950 and gfx1250.

Unlike `put`, these post from a **single issuer** and do no lane grouping: thread
scope posts one active lane at a time (several lanes are serialized, so they may
share a queue), warp/block scope lets the leader post. Only
`ccoSdmaOptFlagsAggregate` is accepted.

### postWaitMem / postWaitSignal

```cpp
// park queue (next, q) until the local flag reads 1; the put behind it then runs
sdma.postWaitMem(next, q, flagWin, 0, 1u);                  // cmp defaults to ccoSdmaCmpEq
sdma.put<ccoCoopThread, false, true>(next, dstWin, 0, srcWin, 0, bytes, q);
// kernel can return here: no CU is polling or has to ring anything later
```

The engine runs the queue in order, so everything posted after the wait runs only
once it holds. `win`/`offset` address memory on **this** GPU (a flag a peer writes
with `put`, a CU store, a host write). `cmp` is `ccoSdmaCmp{Always,Lt,Le,Eq,Ne,Ge,Gt}`
on `(*addr & mask)`; only 32 bits are compared.

`postWaitSignal(peer, q, srcRank, sigQueueId, expected)` waits on a signal slot, so
one engine can forward what another delivered with no CU in the path:

```cpp
sdma.postWaitSignal(next, relayQ, prev, hopQ, 1);   // prev's hop landed
sdma.put<ccoCoopThread, false, true>(next, fwdWin, 0, recvWin, 0, bytes, relayQ);
```

The engine compares 32 bits, so `expected` must stay below 2^32. A larger value is
truncated and releases early: `2^32 + 5` waits for `low32 >= 5`, which a counter of
100 already meets. Traps under `MORI_CCO_SDMA_DEBUG`.

Pre-posting everything and ringing once is `postWaitMem<.., Aggregate>` +
`put<.., Aggregate>` + `commit`.

Measured on gfx950, flag set to copy complete, median of 100: a CU polling the flag
and then issuing a put costs 7.1 µs at 4 KB, 8.5 µs at 64 KB; pre-placing the
packets and having the CU only ring costs 5.9 / 7.3 µs; `postWaitMem` costs
4.4 / 5.8 µs. The gap is a fixed ~1.5–3 µs and is gone by 16 MB. What it really buys
is that no CU waits.

Sharp edges: the queue is blocked until the condition holds, so anything else
posted to it waits too, and a flag that never arrives parks it for good (retry is
unbounded). A full ring spins the poster until the wait clears. Give a waiting queue
its own `queueId`.

### postWaitMemPutPtr

```cpp
// lane p: copy my block to peer p once p has stored this call's epoch in my ready[p];
// the engine waits, the kernel exits right after posting
sdma.postWaitMemPutPtr<false, true>(p, q, ctrlWin, READY_OFF + 4 * p, epoch, ccoSdmaCmpEq,
                                    0xFFFFFFFFu, dstOnPeer, src, bytes);
```

`postWaitMem<Aggregate>` + `putPtr<Aggregate>` + `commit` in one reservation and
one doorbell: cheaper to post, same flag-to-copy latency. Thread scope; lanes on
distinct queues post in parallel. `dst` is the peer's memory as mapped here
(`lsa_ptr`), `win`/`offset` the local flag; `bytes == 0` posts wait + signals,
and nothing without a signal. Over `CCO_SDMA_MAX_COPY_BYTES` (~1 GiB) it posts
wait + chunked put + commit, one lane at a time, so on a shared queue each lane's
copy waits only for its own flag and those of lanes posted before it.
Use it when the flag is usually not there yet; a wait that already holds still
costs ~1.6 µs over a plain `put`, so check the flag on the CU first if it usually
is: read it once, plain `put` if it already holds, this post if not (example 12). Use `Eq` on a per-call epoch: the engine compares unsigned, so `Ge` on a
running total misbehaves across the u32 wrap. The postWaitMem caveats apply.

### putBroadcast

```cpp
sdma.putBroadcast(next, recvWin, 0,  /*peer2=*/me, localCopyWin, 0,  sendWin, 0, bytes, q);
```

One COPY packet, two destinations; `peer`/`dstWin` is the first and selects the
queue, `peer2` may be your own rank. `remoteSignal` bumps each distinct destination
peer other than yourself once; `localSignal` is one atomic for the call.

The two destination addresses must be congruent mod 32 (`(dst1 ^ dst2) & 31 == 0`),
as in ROCr's broadcast packet. If not, the second destination receives the data
shifted by the offset difference: the whole range is wrong and up to 31 bytes next
to it are overwritten. Nothing checks this unless you build with
`MORI_CCO_SDMA_DEBUG`, which traps.

A signal slot counts per (source rank, `queueId`), not per physical queue. The
broadcast is posted on queue `(peer, q)` but its remote signal bumps `peer2`'s
`[you, q]` slot, which `put` only advances from queue `(peer2, q)`. A waiter that reads
the count as "the Nth packet on `(peer2, q)` has landed" can be released by a
broadcast's signal while an earlier `put` on `(peer2, q)` is still in flight; only
the cumulative total is reliable. If you wait on per-queue completion, give
broadcasts their own `queueId`.

**Do not reach for it to save time.** One engine does not overlap the two writes:
broadcast to two remote GPUs ran at 38 GB/s against 61 GB/s for two `put`s on one
queue (122 GB/s for two `put`s on two queues), and local + remote was the same 43 vs
61. It was faster only when both destinations are local (68 vs 61 GB/s). For
fan-out to peers, issue puts on separate queues.

The packet is 36 B and sits back to back with the packets around it, like every
other packet; it may straddle the ring end.

## Failure modes

| symptom | cause |
|---|---|
| every call silently does nothing | `MORI_ENABLE_SDMA` not set, or `sdmaNumQueue == 0` |
| `ccoSdma` undeclared | kernel TU missing `-DBUILD_CCO_SDMA=1` |
| `waitSignal` never returns | `expected` shared across queues, `comm.rank` passed where `comm.lsaRank` was meant, or `expected` counted per lane when the sender signals once per group |
| `quiet` returns but data is stale | aggregate puts never `commit`ed — `quiet` drains to the last *rung* packet. Traps under `MORI_CCO_SDMA_DEBUG` |
| a `put` inside an `Aggregate` batch never returns | 256 KB or more posted on one queue without `commit` — commit sooner. Traps under `MORI_CCO_SDMA_DEBUG` |
| wrong data with several lanes per wave | `ccoSdmaThreadSameQueue` set while lanes target different queues |
| receiver sees a signal but stale data | signal fired on a different queue than the copy |
| a queue stops moving after `postWaitMem` | the flag never reached the value, or it points at another GPU's memory — `win`/`offset` must be local |
| `postWaitSignal` releases too early | `expected` is 2^32 or more and was truncated to 32 bits |

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
