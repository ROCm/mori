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
can start until compact lands, and combine cannot start until expand does.

| fp4 | dispatch leg | combine leg | **pair** | vs stock | consumer |
|---|---:|---:|---:|---:|---|
| stock | 35.69 | 40.83 | **76.52us** | -- | unchanged |
| variant A, segment-aware consumer | 26.32 | 41.21 | **67.53us** | **-11.8%** | must change |
| variant A + compaction (drop-in) | 33.12 | 44.86 | **77.98us** | **+1.9%** | unchanged |

| bf16 | dispatch leg | combine leg | **pair** | vs stock |
|---|---:|---:|---:|---:|
| stock | 51.23 | 41.00 | **92.22us** | -- |
| variant A + compaction (drop-in) | 50.98 | 44.81 | **95.79us** | **+3.9%** |

**Compaction does not pay for itself.** Variant A saves 9.37us on the fp4 dispatch leg
and compaction gives back 10.46us, so the drop-in path is a net *regression*. The win is
real but it only exists for a consumer that can walk the segments.

Cost of compaction in flow, by difference against variant A alone:

| step | leg | fp4 | note |
|---|---|---:|---|
| compact, payload | dispatch | 5.22 | 3584 B rows |
| compact, reverse map | dispatch | 1.58 | 8 KB, a second launch -- almost all overhead |
| expand, payload | combine | 3.66 | **14336 B rows: combine input is bf16, not fp4** |
| | | **10.46us** | |

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

- **The map move is a separate launch.** `ep_compact_idx` moves 8 KB and costs 1.58us,
  nearly all of it launch and latency. It is the same permutation as the payload move
  and belongs in the same kernel.
- **Compaction is not worth shipping as-is.** At 10.46us against a 9.37us saving it is a
  net loss on both dtypes. Either fuse it into the dispatch kernel's tail -- where the
  rows are already in registers and the barrier has already been paid -- or give the
  consumer the segment table and skip it entirely.
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

EPCOMPACT=0 $R                                              # stock
MORI_SOURCE_ROOT=/tmp/jit_variantA CHECK=0 EPCOMPACT=0 $R   # variant A alone
MORI_SOURCE_ROOT=/tmp/jit_variantA EPCOMPACT=1 $R           # variant A + compaction
```

`EPCOMPACT=0` is the stock path byte for byte -- it was re-run against the unmodified
kernel and reproduced 35.5us, so the hooks cost nothing when off. `EPCOMPACT_MAP=0` drops
the reverse-map move; it is a timing probe only and needs `CHECK=0`.

**`LD_LIBRARY_PATH` must include a directory with a plain `libamdhip64.so`.** Only the
versioned `.so.7` is on the loader path in this image, and without the unversioned name
the bench silently falls back from in-graph GEV events to per-iteration host events,
which reports dispatch at 43.6us instead of 35.5us. The `[E2E]` line prints `src=gev` or
`src=periter` -- check it.
