# EPv2 gfx1250 dispatch: removing the peer slot-allocator atomic

Measured on 4x gfx1250, `EP4 / hidden 7168 / topk 6 / 384 experts / 512 tokens-rank`,
fp4 dispatch with bf16 combine. All timings are graph mode, `ITERS=100`.

Everything here is an **experiment on a branch**. Nothing is wired into the op or the
build; `src/` is untouched. The kernel variant is a standalone header selected at run
time through `MORI_SOURCE_ROOT`, and the compaction kernels are launched from the test
drivers via `ctypes`.

## The finding

An in-kernel `wall_clock64()` breakdown of `EpDispatch1250xBody` puts the largest single
cost not in the payload transfer but in the **slot reservation** -- a SYSTEM-scope
`atomic_fetch_add` on *peer* memory (`ep_intranode_1250x.hpp`, the `args.offTokOff` RMW):

| stage (critical path, first-arriver block) | us |
|---|---:|
| precompute | 12.58 |
| &nbsp;&nbsp;of which **slot reserve** | **6.40** (population mean 10.87, p95 21.6) |
| transfer (payload TDM) | 5.86 |
| fence + signal | 0.41 |
| barrier | 13.27 |

That cost is **contention, not the barrier**, established three ways:

1. **Block-count sweep.** The floor is flat at 3.8-4.1us from 8 to 64 blocks while the
   tail grows ~5.6us per doubling. A barrier cannot depend on how many *other* blocks
   exist; the number of atomics landing on the shared word can. (`evidence/blksweep.sh`)
2. **Natural control.** At 128 blocks with 512 tokens, blocks 64-127 get no tokens, so
   `if (n > 0)` skips the atomic. Those blocks run the same region in **0.72us** -- that
   is the whole `__syncthreads()` contribution. (`evidence/idlecheck.py`)
3. **Timestamp shape.** The 64 blocks *start* the region within 1.0us of each other but
   *finish* spread over 9.7-15.3us, median gap between consecutive completions 0.12us.
   Clustered starts, fanned ends: a draining queue. (`evidence/slotshape.py`)

`offTokOff` is one `index_t` per rank, so every block of every rank RMWs the same word:
64 x 4 = 256 remote RMWs per rank onto 4 words.

**Grid tuning cannot fix it.** Holding total warps at 512 and trading blocks for warps
per block, halving the atomics made dispatch *49% slower* -- block count also sets CU
spread, and the payload TDM needs that far more than it needs fewer atomics
(`evidence/geom.sh`):

| blocks x warps | atomics | dispatch us |
|---|---:|---:|
| **64 x 8** (the tuned default) | 256 | **35.0** |
| 32 x 16 | 128 | 52.3 |
| 16 x 32 | 64 | 48.0 |
| 128 x 4 | 512 | 40.5 |

## Variant A: static per-source partition

`EpMaxRecv == worldSize * maxTokPerRank` already reserves one full quota per source, so
give source `s` the slot range `[s*stride, (s+1)*stride)` in every peer. No two sources
can name the same slot, so no peer has to arbitrate and the base becomes a local
computation. The AGENT-scope `atomicAdd` on `destPeTokenCounter` that the stock path
*already performs* returns exactly the in-segment offset, so the remote RMW is deleted
and nothing replaces it.

`variantA.patch` (+46/-4, mostly comment) against `src/ops/dispatch_combine_v2/
ep_intranode_1250x.hpp`. Three changes: the reservation, a per-source count export, and
a staging-copy bound in combine.

Result -- one stage moves, nothing else does:

| stage | stock | variant A |
|---|---:|---:|
| SlotReserve | 10.87 (p95 21.6) | **0.74** (p95 0.96) |
| every other stage | -- | within +-0.2us |

0.74us is exactly the barrier-only floor from the natural control above.

## The cost: the landing zone is no longer contiguous

| | valid slots |
|---|---|
| stock | one dense run, `[0 .. total_recv)` |
| variant A | `[0..425] [512..932] [1024..1451] [1536..1961]`, gaps between |

`total_recv` is still correct as a *count*, but it no longer describes a range. Any
consumer doing `recv_tokens()[:total_recv]` reads garbage from the gaps and misses the
last source. Two ways out, both provided:

- **Segment-aware consumer** -- walk the four `(start, count)` pairs. `start` is
  `s * (recv_cap / world)`; the counts are what dispatch now exports.
- **Compaction** -- `epcompact.hip` (`ep_compact` / `ep_expand`) restores a dense run,
  so the consumer needs no change at all. Both directions are required: combine gathers
  through the *sender's* `disp_dest_tok_id_map`, so the expert's output must be
  scattered back before combine.

## Numbers

Measured **in flow**, inside `bench_ep.py`'s own in-graph GEV window, with `EPCOMPACT=1`
running the production order `dispatch -> compact -> expert -> expand -> combine`.
compact is charged to the dispatch leg, expand to the combine leg -- nothing downstream
can start until compact lands, and combine cannot start until expand does. `EPREMAP=1`
deletes expand; `EPFUSE=1` also folds the two index passes into the payload kernel.

| fp4 | dispatch leg | combine leg | **pair** | vs stock | consumer |
|---|---:|---:|---:|---:|---|
| stock | 35.28 | 40.83 | **76.11us** | -- | unchanged |
| variant A + compaction, **expand** | 33.35 | 44.59 | 77.95us | +1.9% | unchanged |
| variant A + compaction, **remap** | 34.81 | 40.73 | 75.53us | -1.3% | unchanged |
| variant A + compaction, **remap, fused** | 31.87 | 40.77 | 72.64us | -5.1% | unchanged |
| **+ no per-iteration reverse-map clone** | **30.59** | 41.05 | **71.65us** | **-5.9%** | unchanged |
| variant A, segment-aware consumer | 26.32 | 41.21 | **67.53us** | **-11.8%** | must change |

| bf16 | dispatch leg | combine leg | **pair** | vs stock |
|---|---:|---:|---:|---:|
| stock | 51.65 | 41.32 | **92.97us** | -- |
| variant A + compaction, expand | 51.80 | 45.00 | 96.80us | +4.3% |
| variant A + compaction, remap, fused | 50.34 | 40.80 | 91.13us | -1.8% |
| **+ no per-iteration reverse-map clone** | **48.91** | 41.23 | **90.14us** | **-3.0%** |

A rocprofv3 kernel trace found a `__amd_rocclr_copyBuffer` running once per iteration
between dispatch and the compaction, which stock does not have (8 dispatches in a whole
stock run against 3350 here). It was the bench's own doing:
`routing.disp_tok_id_to_src_tok_id_local` is a property that CLONES off the arena on
first access, and dispatch hands back a fresh routing handle every call, so reading it
once per iteration bought a device copy per iteration. The op makes that clone lazy
exactly so the common combine path never pays it. The compaction kernel wants the source
pointer, not a private copy, so it now binds the live arena view (`_reverse_src_view`).
Copies per run: 3350 -> 10, and the steady-state sequence is finally just
`DISPATCH -> FUSED -> COMBINE`.

That is worth **1.3us at fp4 and 1.4us at bf16**, and it is what moves bf16 from a wash
to a real win. Both figures below are with it removed.

Both pass with the bench's own unmodified layout-unaware identity expert. The combine leg
comes back to exactly stock (40.77 against 40.75) -- expand leaves no residue.

Getting there took two steps past the naive version:

**1. Delete expand (-2.4us).** Combine gathers wherever `dispDestTokIdMap` points, so
rewriting that map from segmented slots to dense rows lets the expert leave its output
dense. 12 KB of int32 replaces a 24 MB read plus 24 MB write.

**2. Fuse the three dispatch-side passes into one launch (-2.9us).** The payload move is
6 MB and the two index moves are 20 KB, yet separately they cost 1.58us and 1.77us --
launch overhead, not work. They share the prefix sum, and an fp4 row is 224 uint4 against
256 threads, so the index work rides lanes the payload loop leaves idle.

| | dispatch leg |
|---|---:|
| variant A, no compaction | 26.32 |
| + payload compact | 31.40 |
| + reverse-map compact (2nd launch) | 32.98 |
| + dest-map remap (3rd launch) | 34.81 |
| **all three fused into one launch** | **31.87** |

What compaction now costs in flow is 5.55us on the dispatch leg and nothing on the
combine leg, against variant A's 9.4us saving.

That 5.55us is this arm at this iteration count. Measured in isolation, one token count
per process, the same pass is 4.1us at 512 tokens and ranges 4.0 to 12.25us across the
shapes in "The win is shape-dependent" above. Quote a compaction cost with the shape it
was measured at; there is no single number.

### The front rendezvous: what actually shipped

Everything above -- the segmented landing zone, the compaction pass, the dest-map
remap, the tail prefix publish -- exists because variant A reserved slots before it
knew what the other senders were sending. Exchange the send-count matrix FIRST and
none of it is needed:

    S[a][b] = tokens rank a sends to rank b        (a's own row; it already has it)
    base(a -> p) = sum_{i<a} S[i][p]               (a's first row in p's buffer)

Every sender can then write straight into its final packed row. No segments, so no
compaction; no slide, so no remap; and the sender knows its own base, so no prefix
publish. The cost is one grid rendezvous plus one cross-device round trip, which
`evidence/xdev_barrier.hip` prices at 3.81us at EP4 against the 7.0us it removes.

`MORI_EP_VARIANT_A=1` selects it. `=seg` still selects the old segmented path.

**Drop-in test.** The only difference between the two arms is that one env var:
no `EPCOMPACT`, no `EPREMAP`, no `EPFUSE`, no `MORI_JIT_EXTRA_FLAGS`, no
`EPNOSCALES` -- per-token scales ON -- and `CHECK=1` throughout.

| arm | dispatch | combine | |
|---|---:|---:|---|
| stock fp4 ct=512 | 34.29 | 40.91 | PASS |
| **VARIANT_A=1** fp4 ct=512 | **30.65** | 40.96 | PASS, **-10.6%** |
| stock fp4 ct=2048 | 65.38 | 118.21 | PASS |
| **VARIANT_A=1** fp4 ct=2048 | **55.61** | 117.95 | PASS, **-14.9%** |
| stock bf16 ct=512 | 51.08 | 40.86 | PASS |
| **VARIANT_A=1** bf16 ct=512 | **45.77** | 40.90 | PASS, **-10.4%** |
| stock bf16 ct=2048 | 122.65 | 118.22 | PASS |
| **VARIANT_A=1** bf16 ct=2048 | **118.59** | 117.77 | PASS, **-3.3%** |

Combine is untouched in every arm -- the packed layout leaves no residue.

**It holds across shapes, which the compaction path did not.** One token count per
process:

| tok/rank | stock | old (compact) | FRONT | FRONT vs stock |
|---:|---:|---:|---:|---:|
| 256 | 29.86 | 32.80 | 29.69 | -0.6% |
| 512 | 35.97 | 34.01 | 30.62 | **-14.9%** |
| 1024 | 58.95 | 57.68 | 52.20 | **-11.4%** |
| 2048 | 65.43 | 62.50 | 55.49 | **-15.2%** |
| 4096 | 67.43 | 70.59 | 58.47 | **-13.3%** |

The compaction path went NEGATIVE by 4096 because its cost grew with the payload
(4.0 -> 12.25us) while variant A's contention win saturated. FRONT has no payload
-sized pass at all, so the curve does not cross. At 256 it is a wash: the rendezvous
is a fixed ~3.8us and there is less contention to win back.

**Verified:** fp4 and bf16; ct 128/512/1024/2048/4096; graph and eager; scales on;
300-iteration soak runs repeated, with `EPCOMPACT_LIB` pointed at a nonexistent file
as a positive control that nothing reaches for the compaction library any more.
Zero byte mismatches and zero injectivity failures throughout.

**Not verified, and a reader should not assume it:** worldSize other than 4 (the
code is generic and the loops are strided, but only EP4 was measured); internode
(this is `ep_intranode_1250x.hpp` -- the flag is inert on the internode path);
non-gfx1250; the flydsl backend; and any real MoE consumer.

**Things a reviewer should push back on:**

- The count matrix rides a resized `dense_prefix` arena region instead of its own
  `EpArgs` field. Functional, deliberate -- it kept the experiment free of an ABI
  bump -- but it wants a real field before this merges.
- `hip_backend.py` now binds `xdb_flag` on the DISPATCH path too. FRONT uses it as
  a generation tag, and it was previously combine-only.
- Dispatch now advances `xdbFlag`, which the combine barrier also uses as its phase.
  The two are coupled through that array; changing either means checking both.
- **The rank-to-rank wait moved to the front of the kernel.** Stock waits only at the
  end. Any path where ranks do not call dispatch the same number of times now hangs
  earlier than it used to. Same class of hazard as stock, different position.
- Row ORDER differs from stock. Both produce a dense run of `totalRecvTokenNum` rows,
  but FRONT groups by source rank where stock's order falls out of the remote atomic.
  Combine does not care -- it follows `dispDestTokIdMap` -- and neither does the
  bench's layout-unaware expert. A consumer that depends on row order rather than on
  the reverse map would see the difference.

### The prefix exchange: measured, and it costs 2.7 to 3.1us

`prefix[pe] = sum_{j<rank} counts_pe[j]` is a column sum over what OTHER senders put on
pe, so the sender cannot compute it locally. The bench builds it once during priming
with an `all_gather` of `world` int32, which is legitimate only because the routing is
fixed across iterations; a real implementation needs it per batch. Variant A therefore
publishes it from the kernel: the receiver holds the whole count row, scans it in one
wavefront and stores one int into each sender's slot. That is `world^2` ints, 64 B at
EP4.

Full breakdown at ct=512, two runs each, spread <= 0.46, CHECK=0. `-DMORI_EP_VA_NOPREFIX`
compiles the publish out:

| | dispatch | delta |
|---|---:|---|
| variant A dispatch only, publish OFF | 27.16 | -- |
| variant A dispatch only, publish ON | 29.85 | **+2.69 prefix** |
| variant A + compact + remap, publish OFF | 31.02 | +3.86 compact |
| variant A + compact + remap, publish ON (**shipped**) | 34.16 | **+3.14 prefix** |
| stock (needs no compaction) | 34.88 | |

So variant A wins **7.7us** on the dispatch kernel itself (34.88 against 27.16), and
then spends **7.0us** of it making the result usable: 3.1us for a per-batch prefix and
3.9us to compact the segments back into a dense run. Net **-0.7us, about -2%**.

That is the honest number for the API-compatible path. The -14.2% in the shape table
above is the same shape with the payload left segmented and the prefix supplied by the
host -- neither of which a caller can use as-is.

**Keep `MORI_EP_VA_NOPREFIX`.** An earlier cleanup deleted it, and a later session then
built "publish off" arms with a flag that no longer existed. Every arm was the same
binary, two identical numbers were read as "the publish is free", and that wrong
conclusion reached a written report before the next measurement caught it. A guard that
only exists to make a cost measurable is still load-bearing.

**Do not try to hide the publish in the compaction kernel.** Publishing from compact's
entry instead -- where ~5us of payload copying still lies ahead, and where the scan is
already computed for free as `pre[]` -- was built, validated (PASS, 1/1) and measured.
Two runs each, against the 31.02 "publish off" baseline:

| | dispatch | delta |
|---|---:|---|
| no publish, remap restricted to block 0 | 36.43 | +5.41 restructure |
| publish in compact, no wait | 38.86 | +2.43 publish |
| publish in compact, publish + wait | 39.72 | +0.86 wait |

5.6us worse than shipped. Two separate reasons, and only one of them was anticipated:

- **The wait is genuinely cheap** -- 0.86us, covered by the copy exactly as intended.
- **The publish is not hidden**, costing 2.43us in compact against 2.69-3.14us in
  dispatch, i.e. no better. Same-stream kernels do not overlap: a kernel cannot retire
  until its stores drain, so issuing them earlier inside the kernel buys nothing. Only
  real work *behind* them in the same kernel would, and there is none at either site.
- **Consuming it in compact forces the remap onto one block** (so it can clear the
  slots without racing a sibling), which costs 5.41us on its own -- far more than the
  wait it enables.

Two things an isolated benchmark got wrong, both found only by measuring in flow:

- **expand moves 4x the bytes compact does.** dispatch lands at the wire dtype (fp4,
  3584 B) but combine's input is always bf16 (14336 B). An earlier revision of this file
  priced both legs at 3584 B and reported compaction at 4.39us. The bf16 width was
  already measured at 3.67us in isolation and matches the 3.66us seen here.
- **The reverse map is segmented too, and nobody was moving it.**
  `disp_tok_id_to_src_tok_id_local` is sized at the recv capacity and carries the same
  gaps, so a consumer reading dense rows needs it in dense order. It is 4 B/row against
  3584, yet it costs 1.58us because it is a separate launch -- fusing it into the payload
  kernel is the obvious fix and is not done.

Payload compaction also costs 5.22us in flow against 2.18us measured in a tight local
loop (`evidence/compact.hip`). The isolated loop re-reads the same buffers 50 times with everything resident;
in flow, compact reads 6 MB that dispatch has only just written, behind dispatch's
barrier tail. **Isolated kernel timings did not compose here** -- that is the main
methodological lesson of this file.

### The win is shape-dependent, and it is gone by 4096 tokens

Everything above is the 512 token-per-rank target shape. The win does not hold across
shapes. Dispatch leg, one token count per process, two runs each, spread <= 0.4us:

| tokens/rank | stock | variant A alone | + compact (shipped) | net |
|---:|---:|---:|---:|---:|
| 256 | 30.5 | 25.5 | 29.5 | -3.3% |
| 512 | 35.65 | 26.5 | **30.6** | **-14.2%** |
| 1024 | 57.5 | 48.7 | 54.0 | -6.1% |
| 2048 | 64.2 | 51.3 | 58.4 | -9.0% |
| 4096 | 67.8 | 55.2 | 67.45 | **-0.5%** |

The 512 row above is the segmented payload with a host-supplied prefix. For the
API-compatible path -- packed payload, prefix published per batch by the kernel --
re-measured 2026-09-21, two runs each: stock **34.88**, shipped **34.16**, a net of
**-2%**, not -14.2%. Full breakdown in "The prefix exchange" below.

Two curves in opposite directions. Variant A's own gain saturates -- 5.0, 9.2, 8.8, 12.9,
12.6us -- because it removes contention and barrier queueing, a per-round cost that does
not scale with payload. Compaction's cost grows with the data: 4.0, 4.1, 5.3, 7.1,
12.25us. At 4096 they are 12.6 against 12.25 and cancel. Past 4096 the shipped path will
be SLOWER than stock, because the gain is capped and the cost is not. Anyone taking this
to a larger shape needs the "Eliminating the passes" work first, not this as-is.

**Measure one token count per process.** A multi-point sweep inflates its later points,
unequally between arms: 4096 reads 76-83us as the tail of a sweep and 67.8 +/- 0.4 on its
own. An earlier revision of this file's conclusions came from sweeps and reported a win
at 4096 that does not exist. If a result disagrees with an isolated re-run, trust the
isolated one.

## The win does not depend on the measurement method

All three of the bench's timing methods, stock against optimized, fp4, one session,
every arm correctness-gated:

| method | stock pair | optimized pair | delta |
|---|---:|---:|---:|
| graph, events INSIDE the graph (GEV) | 76.53 | **71.25** | **-6.9%** |
| eager, events around the CALLS | 79.20 | **74.40** | **-6.1%** |
| graph, events around the REPLAYS (`hd`/`hc`) | 83.52 | **77.58** | **-7.1%** |

| method | stock dispatch | optimized dispatch | delta |
|---|---:|---:|---:|
| GEV | 35.52 | **30.40** | -14.4% |
| eager | 37.20 | **32.10** | -13.7% |
| replays | 39.68 | **33.45** | -15.7% |

The three methods disagree with each other by up to 7us on absolute time -- eager costs
about 3us over GEV, and timing graph REPLAYS from outside costs about 6.5us, since that
pays a replay launch and an event pair per iteration. But the overhead is the same for
both arms, so the delta is stable at -5 to -6us on dispatch whichever instrument is used.
GEV is the one to quote because it is the tightest, not because it flatters the result.

combine lands within 40.6-44.1us in all six runs, confirming it is untouched.

## Can compact be made faster?

Two experiments below are negative and worth recording. They do NOT add up to
"nothing left" -- see the limits at the end of this section, which an earlier
revision of this file got wrong.

**Block size is already optimal.** An fp4 row is 3584 B = 224 uint4, so at 256 threads
each thread moves a single 16 B chunk with 32 lanes idle -- it looks starved of work
next to bf16's 3.5 chunks per thread. Giving threads more work makes it *worse*: the
kernel is parallelism-bound, not ILP-bound.

| threads/block | dispatch leg | combine leg | pair |
|---:|---:|---:|---:|
| 64 | 34.36 | 49.03 | 83.39 |
| 128 | 33.56 | 45.90 | 79.46 |
| **256** (default) | **32.84** | **44.87** | **77.71** |
| 512 | 34.01 | 45.30 | 79.31 |
| 1024 | 35.22 | 46.71 | 81.93 |

**The permutation is free.** `EPCOMPACT_STRAIGHT=1` drops the segment search, the LDS
prefix sum and the `__syncthreads()`, copying row r to row r -- same volume, same launch,
no gather:

| | dispatch leg |
|---|---:|
| variant A, no compaction | 26.64 |
| + payload compact, permuted | 31.40 |
| + payload compact, **straight copy** | 31.32 |

0.08us apart. The addressing costs nothing -- and note this also rules the per-block
prologue out as the cost, since `STRAIGHT` drops the segment search, the LDS prefix sum
and the `__syncthreads()` for 0.08us.

**What these two experiments do not show.** Both were run at one token count, and they
vary threads-per-block and addressing -- not the grid, and not the fixed/marginal split.
Measured per token count, one count per process, compact's achieved bandwidth is not flat:

| tokens/rank | recv rows | MB moved (r+w) | compact us | achieved GB/s |
|---:|---:|---:|---:|---:|
| 256 | 844 | 6.05 | 4.0 | 1512 |
| 512 | 1688 | 12.10 | 4.1 | 2951 |
| 1024 | 3358 | 24.07 | 5.3 | 4541 |
| 2048 | 6743 | 48.33 | 7.1 | 6808 |
| 4096 | 13513 | 96.86 | 12.25 | **7907** |

Doubling the data from 256 to 512 tokens costs 2.5% more time, so most of the 4.1us at
the target shape is fixed, not bandwidth. And 7907 GB/s at 4096 is far above the 3496
GB/s uncached-gather figure the "at the floor" claim rested on, so that figure was not
the right ceiling for this kernel.

Two things remain untested, and a colleague should treat them as open:

- **The grid is never swept.** `epcompact.hip` hardcodes `grid = min(maxRows, 4096)`, and
  `maxRows` is the worst case (`world * maxTokPerRank`) while dedup delivers ~82% of it.
  So every block gets about one row until the grid saturates at 4096 -- which is exactly
  the point where achieved bandwidth peaks. There is no env knob for it yet.
- **The fixed cost is not split.** How much of the ~4us is the extra graph node versus
  work inside the kernel is unmeasured. If it is mostly the node, only fusing or
  eliminating the pass helps; if not, the grid is worth a sweep. Isolate it by running
  compact with the payload move compiled out.

## Eliminating the passes

`dispDestTokIdMap` is written by the SENDER at dispatch (line 446 of the variant header):

```cpp
index_t j = atomicAdd(&s_run[myDestPe], 1);
myDestTokId = s_base[myDestPe] + j;              // segmented slot under variant A
args.dispDestTokIdMap[tok * topk + _eLane] = EpFlatIndex<kCfg>(myDestPe, myDestTokId);
```

and combine reads it back and gathers straight from the peer (line 1197):

```cpp
index_t destTokId = args.dispDestTokIdMap[tokenId * topk + j];
srcPtrs[j] = EpPeer<TokT>(win, destPe, args.offOutTok) + destLocalTokId * hiddenDim + ...;
```

Combine never assumes a layout -- it goes exactly where the map points. So the layout is
decided by one value, `s_base[p]`, and both passes are bookkeeping, not data:

- **expand dies if the map holds dense indices.** `dense_idx = prefix_p[me] + off`, where
  `prefix_p[me] = sum_{j<me} n[j][p]`. Rewriting the map is 3072 int32 = 12 KB against
  expand's 24 MB read + 24 MB write.
- **compact dies too if `s_base[p]` is that dense base in the first place.** Then the
  payload lands dense, the layout is byte-identical to stock, and no consumer changes.

`prefix_p[me]` depends only on `n[j][p]`, the send counts -- which every sender knows for
its own row before any payload moves. So it needs a counts-only exchange at the head of
dispatch: `world^2` ints, 64 B at EP4, and a rendezvous before slot assignment. This is
what DeepEP's notify_dispatch does.

**Expand is now deleted** (`EPREMAP=1`, `ep_remap` / the fused kernel): the map rewrite
is real and validated, and the numbers are in the table above. Combine returns to exactly
its stock time.

**Compact is deliberately kept.** Making `s_base[p]` the dense base would delete it too,
but only by moving the prefix ahead of the payload write -- and the expert needs dense
rows regardless, so the pass has to happen somewhere. Folding it into dispatch's tail is
the remaining idea; it needs the counts BEFORE the transfer, which means a front
rendezvous, and today dispatch waits only at the end. Not attempted.

## Correctness

| check | covers | result |
|---|---|---|
| `tests/seg_check.py` | fp4 dispatch bytes + segment ownership, 1701 slots | PASS, 0 mismatches, 0 wrong-segment |
| `tests/seg_e2e.py` | bf16 end-to-end, segment-aware expert | PASS |
| `tests/seg_e2e.py` (control) | bf16, *dense* expert on variant A | **FAIL** as expected |
| `tests/flow_check.py` | bf16 end-to-end, compaction, **stock layout-unaware expert** | PASS |
| `tests/flow_check.py` (control) | same, compaction off | **FAIL** as expected |
| `tests/fp4_compact_check.py` | fp4 dispatch + compaction, byte-exact | PASS, 0 mismatches, 0 out-of-order |
| `bench_ep.py EPCOMPACT=1` fp4 | in-flow compaction, dispatch bytes via the **compacted** map | PASS, 1/1 |
| `bench_ep.py EPCOMPACT=1` bf16 | in-flow compaction, **full identity-expert value check** | PASS, 1/1 |
| `bench_ep.py EPCOMPACT=0` bf16 on variant A (control) | same, compaction off | **FAIL** as expected |
| `bench_ep.py EPREMAP=1` fp4 / bf16 | expand deleted, combine gathers dense rows | PASS, 1/1 both |
| `bench_ep.py EPFUSE=1` fp4 / bf16 | all three dispatch-side passes in one kernel | PASS, 1/1 both |

The end-to-end criterion is `bench_ep.py`'s: with an identity expert,
`combine[t] == U[t] * input[t]`, where `U[t]` is the number of distinct ranks token `t`
routed to. It is a statement about values and never mentions slots, so it is valid under
either layout. The negative controls matter -- they show the checks still discriminate
once the expert is allowed to follow the layout.

Note `bench_ep.py`'s own bf16 check **fails** under variant A without compaction. That is
correct: its identity expert is hard-coded to the dense run. It is a consumer contract
change, not a kernel bug.

With `EPCOMPACT=1` that same check **passes**, which is the strongest statement available
here: the bench's expert is untouched and layout-unaware, so compaction has genuinely
restored the stock contract, payload and reverse map both. `EPCOMPACT=0` on the same
kernel still fails, so the gate has not simply gone blind.

## Not done

- **The prefix exchange is modelled, not built.** It is computed once during priming
  with an `all_gather`, which only works because the routing is fixed here. The argument
  that it rides the signal path for free is untested, and it is the one thing standing
  between this and a real integration.
- **Hiding the prefix publish inside the compaction kernel was tried and it
  regresses**, by 5.6us. The wait hides fine; the publish does not get cheaper for
  being moved, and consuming it there forces the remap onto one block. See "The
  prefix exchange" above.
- **Folding compaction into dispatch's tail was tried and it regresses.** The epilogue
  version is correct but measures 37.3us against 30.5us for dispatch plus a separate
  compact kernel. A diagnostic build with the payload copy removed put 9.1 of the 11.6us
  down to holding every block resident through the peer wait; the copy itself was 2.5us,
  as predicted. The pass is cheap, staying resident to do it is not. Do not re-try this
  without first solving the front rendezvous -- the counts are needed before the
  transfer, and the kernel currently waits only at the end.
- The compaction kernels launch from Python via `ctypes`, not from inside the op.
- The per-source counts ride on `args.scalesBuf`, which is only safe while
  `kCfg.scaleBytes == 0`. A real integration wants its own `EpArgs` field -- an ABI
  change, so a rebuild.
- fp4 combine output is never value-compared anywhere (torch has no fp4 cast); the fp4
  path is verified at the byte level instead.

## Profiler port (`profiler/`)

`profiler/ep_intranode_1250x.profiler.hpp` ports EPv1's `mori::core::profiler` to the v2
1250x dispatch body: same device profiler, same on-wire `(timestamp, meta)` format, so
`python/mori/kernel_profiler`'s `export_to_perfetto` is used **unmodified**. Enabled with
`MORI_JIT_EXTRA_FLAGS=-DENABLE_PROFILER`; with the flag off the macros compile to
nothing and dispatch measures 35.4us against 35.3us stock, i.e. free.

Two deviations, both forced by v2 being JIT-compiled: the buffer arrives via the dead
`scalesBuf` rather than a `profilerConfig` field, and the slot enum is declared in the
header rather than generated by `tools/profiler/generate_profiler_bindings.py` (a CMake
step). `export_to_perfetto` takes `slot_map=` so the names come from Python.

Covers all 512 warps (64 blocks x 8 warps -- the real geometry, which
`hip_tuning_configs.py` selects for this shape; note it is 8 warps, not the
`ep_spec.cpp` default of 16). Wall clock calibrated at **99.845 MHz** against the host.

### Profiling variant A (`ep_intranode_1250x.profilerA.hpp`)

`ep_intranode_1250x.profiler.hpp` instruments the STOCK body, so it answers "why is
stock slow", not "what does variant A look like". `ep_intranode_1250x.profilerA.hpp` is
the same instrumentation over the variant A body: `variantA.patch` hunks 1, 2 and 4.

Hunk 3 is deliberately left out. Variant A's per-source count export and the profiler's
trace buffer both ride on the dead `EpArgs.scalesBuf`, so they cannot coexist; the
profiler keeps it, since nothing in a profiling run reads those counts. The cost is that
this header cannot drive the compaction path -- to trace dispatch and compact in one
timeline, move the count export to `outWeightsBuf` (also dead in dispatch) first.

Run either header with `profiler/ept_prof.py`, which allocates the ring, drains it and
calls `export_to_perfetto`; `profiler/trace_stats2.py` reduces the drained `.pt` to the
per-slot tables below, split by whether a warp was the first arriver:

```sh
cp -a <jit tree> /tmp/jit_profA
cp benchmark/ep_dispatch_atomic/profiler/ep_intranode_1250x.profilerA.hpp \
   /tmp/jit_profA/src/ops/dispatch_combine_v2/ep_intranode_1250x.hpp
MORI_SOURCE_ROOT=/tmp/jit_profA MORI_JIT_EXTRA_FLAGS=-DENABLE_PROFILER ENABLE_PROFILER=1 \
  M=512 PAIRS=5 WARMUP=20 torchrun --standalone --nproc_per_node=4 profiler/ept_prof.py
```

ENABLE_PROFILER kernels are cached under a separate JIT key, so `~/.mori` needs no wipe.

Stock against variant A, identical routing (recv 1701/1697/1663/1693), mean us per warp:

| slot | stock | variant A | stock p95 | variant A p95 |
|---|---:|---:|---:|---:|
| SlotReserve | 11.29 | **0.74** | 20.35 | **0.96** |
| PayloadTdm | 9.11 | 9.23 | 13.70 | 13.82 |
| MetaTdm | 2.79 | 2.92 | 8.37 | 8.05 |
| MetaStage | 1.27 | 1.06 | 1.64 | 1.36 |
| Setup | 1.07 | 1.13 | 1.56 | 1.68 |
| Routing | 0.71 | 0.69 | 0.92 | 0.92 |

Everything else moves by 0.15us or less. On the first-arriver warp, which owns the waits,
`SlotReserve` goes 8.40 -> 0.70 and `InboundWait` 12.01 -> 4.69: peers reach the barrier
sooner once they are not queueing on the allocator.

Traced spans are for attribution only. They cost about 4us (traced stock 39.62us against
35.65us in the bench) and cost more on the contended path, so the traced delta overstates
the bench delta. Use the trace to apportion, the bench for totals. Keep the launch
geometry at the 64x8 default: `PROFILER_WARPS_PER_RANK` backs 4096 warps, and a larger
grid is dropped silently rather than reported.

## Validated from a clean tree

Final check of the shipped artifacts: `_jit-sources` copied fresh, this directory's
`ep_intranode_1250x.variantA.hpp` dropped in, `epcompact.hip` rebuilt from this
directory, four arms run one per process at 512 tokens/rank, fp4, ITERS=50.

| arm | dispatch | combine | correctness |
|---|---:|---:|---|
| stock | 35.87 | 40.97 | 1/1 verified |
| variant A alone | 26.5 | 40.9 | not checkable -- see note |
| variant A + compact | 31.5 | 44.8 | 1/1 verified |
| **variant A + fused + remap (ship)** | **30.28** | 41.45 | **1/1 verified** |

Dispatch leg -5.59us, -15.6%. Pair 76.84 -> 71.73us, -6.7%. Zero byte mismatches and
zero injectivity failures on every checkable arm.

The dispatch spread also collapses: stock ranges 32.73-38.84us across 20 replays, the
shipped path 29.77-30.98us. That is the same contention removal the profiler sees as
`SlotReserve` p95 falling 20.35 -> 0.96us.

`variant A alone` cannot be correctness-checked: with `EPCOMPACT=0` the checker reads
`[:total]` of a landing zone that is now segmented, so it walks the gaps and reports
false mismatches. It is a timing arm only, which is why it runs with `CHECK=0`.

Confirm the `[GEV]` line shows `mode=graph ... R=20 N=20` and that no line says
`periter`; the host-event fallback reports dispatch ~8us high and would invalidate the
comparison. Note the `[E2E]` line that prints `src=` is suppressed when `E2E_R=0`, so
checking for `src=gev` there is vacuous in this configuration -- grep `[GEV]` instead.

## Reproducing

Needs a container with mori built and 4x gfx1250. Copy `_jit-sources` to a scratch tree,
drop in the variant header, and point `MORI_SOURCE_ROOT` at it:

```sh
cp -r python/mori/_jit-sources /tmp/jit_variantA
cp benchmark/ep_dispatch_atomic/ep_intranode_1250x.variantA.hpp \
   /tmp/jit_variantA/src/ops/dispatch_combine_v2/ep_intranode_1250x.hpp
hipcc -O3 -shared -fPIC --offload-arch=gfx1250 \
   benchmark/ep_dispatch_atomic/epcompact.hip -o /tmp/libepcompact.so
```

The JIT hashes its include tree into the cache key, so swapping the header recompiles
automatically -- no stale `.hsaco`, no `.so` rebuild.

`bench_ep.py` runs the whole flow itself under `EPCOMPACT=1`; the three arms behind the
tables above are:

```sh
cd tests/python/ops/dispatch_combine_v2
export HIDDEN=7168 TOPK=6 EPR=96 SWEEP=512 ITERS=100 MODES=graph DISP=fp4 CHECK=1
R="torchrun --standalone --nproc_per_node=4 bench_ep.py"

V=/tmp/jit_variantA
EPCOMPACT=0 $R                                              # stock
MORI_SOURCE_ROOT=$V CHECK=0 EPCOMPACT=0 $R                  # variant A alone
MORI_SOURCE_ROOT=$V EPCOMPACT=1 $R                          # + compaction, expand
MORI_SOURCE_ROOT=$V EPCOMPACT=1 EPREMAP=1 EPFUSE=1 $R       # + remap, fused (best)
```

`EPCOMPACT=0` is the stock path byte for byte -- it was re-run against the unmodified
kernel and reproduced 35.5us, so the hooks cost nothing when off. `EPCOMPACT_MAP=0` drops
the reverse-map move; it is a timing probe only and needs `CHECK=0`.

**`LD_LIBRARY_PATH` must include a directory with a plain `libamdhip64.so`.** Only the
versioned `.so.7` is on the loader path in this image, and without the unversioned name
the bench silently falls back from in-graph GEV events to per-iteration host events,
which reports dispatch at 43.6us instead of 35.5us. The `[E2E]` line prints `src=gev` or
`src=periter` -- check it.
