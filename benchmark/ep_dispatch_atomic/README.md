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

| variant | dispatch | compaction | effective | vs stock | consumer |
|---|---:|---:|---:|---:|---|
| stock | 35.7 | -- | **35.7us** | -- | unchanged |
| variant A + segment-aware expert | 26.6 | 0 | **26.6us** | **-25.5%** | must change |
| variant A + compaction | 26.6 | 4.39 | **31.0us** | **-13.2%** | unchanged |

bf16 for contrast: 50.8 -> 42.1 dispatch, but compaction costs 7.31us there, so the
drop-in path nets only +1.4us. Use the segment-aware expert for bf16.

Compaction cost is **graph-captured**. A plain `ctypes` loop reports 5.2-5.7us because
several us of Python per call starves the GPU -- that measures the host, not the kernel.

| row width | compact | expand | both |
|---|---:|---:|---:|
| fp4 (3584 B) | 2.18 | 2.20 | **4.39us** |
| bf16 (14336 B) | 3.64 | 3.67 | 7.31us |

## Correctness

| check | covers | result |
|---|---|---|
| `tests/seg_check.py` | fp4 dispatch bytes + segment ownership, 1701 slots | PASS, 0 mismatches, 0 wrong-segment |
| `tests/seg_e2e.py` | bf16 end-to-end, segment-aware expert | PASS |
| `tests/seg_e2e.py` (control) | bf16, *dense* expert on variant A | **FAIL** as expected |
| `tests/flow_check.py` | bf16 end-to-end, compaction, **stock layout-unaware expert** | PASS |
| `tests/flow_check.py` (control) | same, compaction off | **FAIL** as expected |
| `tests/fp4_compact_check.py` | fp4 dispatch + compaction, byte-exact | PASS, 0 mismatches, 0 out-of-order |

The end-to-end criterion is `bench_ep.py`'s: with an identity expert,
`combine[t] == U[t] * input[t]`, where `U[t]` is the number of distinct ranks token `t`
routed to. It is a statement about values and never mentions slots, so it is valid under
either layout. The negative controls matter -- they show the checks still discriminate
once the expert is allowed to follow the layout.

Note `bench_ep.py`'s own bf16 check **fails** under variant A without compaction. That is
correct: its identity expert is hard-coded to the dense run. It is a consumer contract
change, not a kernel bug.

## Not done

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

cd tests/python/ops/dispatch_combine_v2
MORI_SOURCE_ROOT=/tmp/jit_variantA COMPACT=1 \
  torchrun --standalone --nproc_per_node=4 ../../../../benchmark/ep_dispatch_atomic/tests/flow_check.py
```

The JIT hashes its include tree into the cache key, so swapping the header recompiles
automatically -- no stale `.hsaco`, no `.so` rebuild.
