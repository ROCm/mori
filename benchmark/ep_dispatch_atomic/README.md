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
| stock | 35.78 | 40.75 | **76.53us** | -- | unchanged |
| variant A + compaction, **expand** | 33.35 | 44.59 | 77.95us | +1.9% | unchanged |
| variant A + compaction, **remap** | 34.81 | 40.73 | 75.53us | -1.3% | unchanged |
| variant A + compaction, **remap, fused** | 31.87 | 40.77 | **72.64us** | **-5.1%** | unchanged |
| variant A, segment-aware consumer | 26.32 | 41.21 | **67.53us** | **-11.8%** | must change |

| bf16 | dispatch leg | combine leg | **pair** | vs stock |
|---|---:|---:|---:|---:|
| stock | 51.68 | 41.13 | **92.81us** | -- |
| variant A + compaction, expand | 51.80 | 45.00 | 96.80us | +4.3% |
| variant A + compaction, **remap, fused** | 50.34 | 40.80 | **91.13us** | **-1.8%** |

Repeated in a second matched session: fp4 stock 75.96 against fused 72.63 (**-4.4%**),
bf16 stock 91.59 against fused 91.06 (-0.6%).

So **fp4 is a consistent -4 to -5% win** and **bf16 is a wash** -- its two margins, -1.8%
and -0.6%, straddle the run-to-run spread of the stock arm itself (92.81 then 91.59), so
nothing there should be claimed. bf16 dispatch is 51us of payload against fp4's 26, and
compaction's 5.55us is the same either way, so the proportional win has to be smaller.

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

**Not measured: the prefix exchange.** `prefix[pe] = sum_{j<rank} counts_pe[j]` is a
column sum over what OTHER senders put on pe, so the sender cannot compute it locally.
Here it is built once during priming with an `all_gather` of `world` int32, which is
legitimate only because the routing is fixed across iterations. A real implementation
needs it per batch -- but not a collective: every sender knows its own send-count row
before any payload moves, so the row can ride the existing signal path and land well
inside dispatch's ~13us inbound wait. That is `world^2` ints, 64 B at EP4. Believed
free, **not demonstrated**.

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
loop. The isolated loop re-reads the same buffers 50 times with everything resident;
in flow, compact reads 6 MB that dispatch has only just written, behind dispatch's
barrier tail. **Isolated kernel timings did not compose here** -- that is the main
methodological lesson of this file.

## Can compact be made faster? No -- it is already at the floor

Two experiments, both negative, both worth recording:

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

0.08us apart. The addressing costs nothing; the entire 4.7us is moving 12 MB right after
dispatch has written it. **There is no kernel optimization left.** The only way to make
compaction cheaper is to not do it.

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
- **Compaction still costs 5.55us.** Folding the payload move into dispatch's tail would
  remove it, but that needs the counts before the transfer -- a front rendezvous against
  a kernel that currently waits only at the end.
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
