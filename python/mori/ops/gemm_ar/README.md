# Fused GEMM + all-reduce (`mori.ops.gemm_ar`)

Fuses a `RowParallelLinear`'s fp8 GEMM with the all-reduce that follows it, over
cco's SDMA copy engines. Built for DeepSeek-V4-Pro's `wo_b` and measured there:
**1146.2us against the model's 1419.5us, -19.4%**, at `[16384, 7168] K=2048` on
8x MI355X.

The public API is `GemmAllReduceOp` in `op.py`. `kernels_fused.py`,
`kernels_sdma.py` and `kernels_lsa.py` hold the kernels, `layout.py` every
window offset and count, and `_gemm_a8w8_8wave.py` / `_shuffle.py` the two
pieces vendored from aiter so mori does not depend on it.

## What it does

aiter's 8-wave fp8 GEMM with the all-reduce's scatter fused into its epilogue.

The target is DeepSeek-V4-Pro's ``wo_b``: a RowParallelLinear whose per-rank GEMM
is ``[M,1024] x [7168,1024]`` fp8 -> ``[M,7168]`` bf16, immediately all-reduced.
Split, that is ``gemm(); all_reduce()``. Fused, the GEMM's C lands directly in a
registered cco window and each destination's slice is pushed by the copy engine
as soon as its last tile is written, so the reduce-scatter transfer overlaps the
rest of the GEMM instead of following it.

Only the *scatter* half of the all-reduce is absorbed. The reduce and all-gather
phases still run as their own kernels, reused verbatim from ``ar.kernels_sdma``
(``build_sdma_phases``), with ``scatter`` swapped for its drain-only twin.

**Nothing here is a production kernel.** aiter's tuned CSV picks a ck/asm/cktile
backend for (N=7168, K=1024), not this one, so the fused-vs-split comparison is
internally valid but is not a claim about DSV4 as shipped.

## Using it

```python
from mori.ops.gemm_ar import GemmAllReduceOp, preshuffle_b

op = GemmAllReduceOp(comm, n=7168, k=2048, m_max=16384)
b_shuffled = preshuffle_b(b_fp8)              # [N, K] -> MFMA preshuffled
out = op(a_fp8, b_shuffled, a_scale, b_scale)  # [M, N] bf16, already reduced
```

`a_scale` is the A 1x128 block scale, physically `[K/128, M]` fp32 (i.e. `[M,
K/128]` column-major -- what `aiter_per1x128_quant(transpose_scale=True)`
emits). `b_scale` is `[N/128, K/128]` fp32 row-major.

`M` must be a multiple of `world_size * block_m`; `op.padded_m(m, world_size)`
rounds up and `op.pad_rows` zero-extends to it. A padded row produces a zero
output row, so the caller slices the result back to `m`.

## fp8 on the wire

`gather_dtype="fp8"` sends the all-gather leg as e4m3 with one fp32 scale per
row, halving its bytes. That leg is ~40% of a fused layer and already runs at
the xGMI ceiling (470 GB/s over 7 links), so halving the bytes halves the time.

Measured, fused-sdma at `[16384, 7168]` K=2048 on 8x MI355X:

| gather wire | us | relL2 |
|---|---:|---:|
| bf16 | 1151.5 | 2.35e-3 |
| fp8 | **1033.9** (-10.2%) | **2.49e-2** |

### Who moves the fp8 gather

`gather_transport="lsa"` (the default for fp8) pulls each peer's slice over
xGMI into registers and widens it on the way to memory. `"sdma"` pushes with
the copy engines and widens in a second kernel -- a copy engine has no ALU, so
for SDMA those cannot be one step, and the second kernel has to read the landed
fp8 back out of local HBM (98 MiB a layer).

| gather | us |
|---|---:|
| bf16 / sdma | 1150.7 |
| fp8 / sdma | 1018.9 |
| **fp8 / lsa** | **957.3** |

The pull's grid is the whole story and is not obvious: these are xGMI reads, so
the grid throttles outstanding remote requests rather than covering HBM latency,
and it wants roughly a tenth of what the local conversion kernels want.

| blocks | 16 | 24 | 32 | 48 | 64 | 80 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| us | 1261 | 1091 | 1006 | 962 | **959** | 963 | 1018 | 1138 | 1184 |

The first version launched 512 -- the quantize grid -- and lost to SDMA by 17%.

### Folding the narrowing into the reduce: measured, and it loses

`fuse_quantize=True` makes the reduce write its bf16 and narrow to fp8 from the
same accumulators, saving a 28 MiB re-read and a launch. It is **off**, because
it costs 20 us rather than saving 12:

| | us |
|---|---:|
| split reduce + quantize | 957.4 |
| fused, row stashed in registers | 977.1 |
| fused, row re-read | 982.0 |

Not register pressure -- the re-reading variant keeps no stash and is no better.
It is the thread map: a per-row amax cannot be taken by a block holding only
part of a row, so fusing forces one-wave-per-row, where `sdma_reduce` walks
packs with a flat grid stride and streams a block through all 8 source slices at
once. Confining a wave to 14 KiB at a time costs the reduce more than the
re-read saves.

It is not free and the cost is not a tuning problem. e4m3 carries 3 mantissa
bits, so one rounding costs ~2.1e-2 on a normal payload whatever the scale
granularity -- per-row measures 2.65e-2 and per-32 measures 2.40e-2, 9% better
for 200x the scale bytes. Going from 2.35e-3 to 2.49e-2 is the price of the
10%, and whether that is payable is a model-level question, not a kernel one.

It also only pays at large M. The two conversion kernels are a fixed cost
against a transfer that shrinks with M, so on the standalone all-reduce it is
+2.4% at M=4096 and -11.2% at M=16384. Per phase at M=16384: quantize 12.4us,
dequantize 88.2us, against ~218us saved on the push.

The scatter leg stays bf16. It carries partial sums that are then added across
every rank, so its fp8 error compounds rather than being a single rounding, and
it is already mostly hidden behind the GEMM -- its 1140 GB/s is not a bandwidth,
it is the tell that the pushes went out from the epilogue and the drain is only
waiting for the tail. `scatter_dtype="fp8"` sizes its regions but raises
`NotImplementedError`.

## Benchmarks and tests

```bash
# numerics
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  pytest tests/python/cco/test_gemm_ar.py tests/python/cco/test_flydsl_ar.py

# the full mode comparison at the model's shape
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo python -m torch.distributed.run \
  --standalone --nproc_per_node=8 benchmark/cco/flydsl/gemm_ar/bench_gemm_ar.py \
  --mode fused-sdma --quant blockscale -m 16384 -n 7168 -k 2048
```

Requires a mori built with `BUILD_CCO_SDMA=ON`. Setting `MORI_ENABLE_SDMA` in
the environment only rebuilds the device bitcode -- a host library built without
the flag has no queues, so every put silently does nothing and the all-reduce
quietly produces zeros.

## A measured negative result, for the record

An earlier revision carried `kernels_preshuffle4w.py`, a port of CK's 4-wave
B-out-of-LDS shape, chasing a 22% gap between this GEMM and CK's at the same
shape. It reached CK's instruction mix and not its speed, and it was deleted
rather than carried (it needed ~1000 lines of further aiter vendoring to serve
a kernel nothing calls). What the investigation ruled out, since the same
ground should not be walked twice:

Ten hypotheses were falsified by measurement -- promote arithmetic, register
spill, store width, VALU scheduling groups, MFMA batch size, scale loads,
address hoisting, load-to-use distance, occupancy, and dependency structure.
Hardware counters (`rocprofv3 --pmc`) show *identical* `SQ_INSTS_MFMA`
(7,340,032) and `SQ_VALU_MFMA_BUSY_CYCLES` (234,881,024), VALU within 1%, and
`MemUnitStalled` at approximately zero -- but `SQ_WAIT_ANY` at 156.0M against
CK's 120.4M. A K-sweep puts the whole difference per-iteration: our fixed cost
is *lower* (51.6us against 66.0us), while each K-block costs 20.9us against
14.5us. The gap is wait, not work, and it is not in any of the places listed
above. See commits `cc696762`, `54fef960`, `9f990637`, `1dead3fc` for the
traces.

## Why the SDMA transport needs no epilogue change

``C`` is an ordinary tensor argument, so pointing it at a cco window is a host-side
change; ``StoreC`` is untouched. That is the whole reason SDMA is the cheap
transport to fuse. The LSA alternative -- having the epilogue store straight into
a peer -- would first require coalescing ``StoreC._store_bf16``, which writes one
bf16 at a time through ``BufferCopy16b``; gcnasm measured that lane-scatter at
0.26x when pushed to a peer. See ``LSA_NOTE`` at the bottom.

## Completion protocol

One monotonic counter per (destination, chunk) in the window (``cfg.counter_off``).
Every block, after its four ``store_c.store`` calls:

    s_waitcnt vmcnt(0) ; s_barrier       -- this block's C tile has retired
    __threadfence_system()               -- ...and is visible to the copy engine
    thread 0: prev = atomic_add(counter[dest][chunk], 1)
    if (prev + 1) % tiles_per_chunk == 0 -- I am the last tile of that chunk
        sdma.put(dest, ...)              -- fire and forget, no quiet

Counters are never reset, so the modulo test works on every launch and the kernel
stays CUDA-graph-safe, exactly like the barrier flags in ``ar.kernels_lsa``.

``quiet`` is deliberately *not* called here: gcnasm measured fusing it into the
GEMM as a 1.8-10.7% regression, so the drain kernel does it afterwards. No
per-destination spin lock either -- gcnasm needed one because several CTAs could
submit for one destination, whereas the modulo test elects exactly one.

Tiles are walked (chunk, destination, n) with the destination rotated by rank,
rather than in aiter's linear order. Linear order finishes destination 0's whole
slice, then 1's, and so on, so the last destination's link only starts at the end
of the GEMM and nothing overlaps. The rotation is gcnasm's
``opus_direct_stripe_tile`` idea. It costs nothing: 49.72us against 49.80us for
the GEMM alone.

## C store: three stages off gcnasm, 6.7% off the GEMM, bit-exact

``StoreC`` emits 128 ``buffer_store_short`` per block -- one bf16 at a time --
because of the MFMA accumulator layout, not a missed vectorization. Lane ``l``
holds ``D[4*(l/16)+i][l%16]``: four consecutive *rows*, stride ``c_cols``, so its
four values are 14336 bytes apart in a row-major C, and the eight bf16 that would
make a 16-byte store live in eight different lanes.

    stage                              GEMM kernel   store instrs   store cycles
    (aiter as-is)                        38.30us       128 short         14,608
    --swap-ab                            slower         32 dwordx2       32,720
    --swap-ab --permlane                 37.04us        16 dwordx4       14,604
    --swap-ab --permlane --lane-transpose 35.74us       16 dwordx4        3,852

Median of 42 dispatches, 8 ranks, [4096,7168] K=1024. Bit-identical to aiter at
[512,512,256], [1024,768,512] and wo_b (``test_swap_ab_is_bitwise_identical``),
and registers are unchanged throughout (VGPR 128 / SGPR 112 / 0 scratch).

**1. ``--swap-ab``** -- ``mfma_adaptor_swap_ab`` (opus.hpp:2064, literally
``base::operator()(b, a, c)`` with ``dim_c()`` redefined). Computing
``B^T A^T = (A B)^T`` in the accumulator's own layout moves a lane to
``D[l%16][4*(l/16)+k]``: four consecutive *columns*, 8 contiguous bytes. Alone it
is *slower* -- a row still only gets 32 bytes, so the same 16 transactions are
squeezed onto a quarter as many instructions and per-instruction address fan-out
quadruples (1022 cycles each against 114).

**2. ``--permlane``** -- two ``v_permlane16_swap_b32`` per M-tile. Semantics
measured rather than assumed: ``vdst' = [X.r0, Y.r0, X.r2, Y.r2]``,
``vsrc' = [X.r1, Y.r1, X.r3, Y.r3]``, so::

    (A, B) = permlane16_swap(tile0.d0, tile1.d0)
    (C, D) = permlane16_swap(tile0.d1, tile1.d1)
    lane group g stores (A, C, B, D)

lands g = 0,1,2,3 on columns 0-7, 16-23, 8-15, 24-31 -- together columns 0..31
contiguously, 16 bytes per lane and 64 per row. No ``ds_bpermute``, no LDS; the
column permutation is absorbed into the address.

This does **not** speed the store up (14,608 -> 14,604). What it pays for is
everything a 2-byte store drags along: 128 stores need 128 addresses, 128 bounds
predicates and 128 scalar multiplies, 16 need 16, and the swapped layout makes
B's scale a vec4 so the scaling packs into ``v_pk_mul_f32``::

    v_mul_f32_e32   257 ->   1     v_lshlrev_b32_e32  159 -> 45
    v_pk_mul_f32      0 -> 128     v_add_u32_e32      133 -> 21
    v_cvt_pk_bf16_f32 128 -> 64    v_cndmask_b32_e64   96 ->  8

**3. ``--lane-transpose``** -- gcnasm's second stage
(kernel_template.hpp:491-510), one ``ds_bpermute`` per dword. After the permlane
stage a row's four 8-column chunks sit in lanes 16 apart, so a 16-lane group
touches 16 rows at 16 bytes each. Transposing the lane index -- lane
``l' = 4r'+q'`` pulls from the lane holding ``(row r', chunk q')``, with
``g = q``'s two bits swapped, 0,1,2,3 -> 0,2,1,3 -- puts adjacent lanes on one
row, so lanes 0-3 write 64 contiguous bytes and a group covers 4 rows.

Same instructions, same 64 addresses, only which lane holds which. The store's
own latency falls **14,604 -> 3,852**, which is the coalescer being sensitive to
lane adjacency and not just to the address set -- exactly what gcnasm's
"pair-coalesced" comment is about. The 64 ``ds_bpermute`` cost 9,604 cycles.

### ``--hoist-scales``: a real redundancy that does not pay to remove

The epilogue calls ``store_c.store`` four times, and those calls share base rows
pairwise and base columns pairwise, so every scale is fetched twice. The
source-attributed thread trace counts it exactly: 16 ``buffer_load_dwordx4`` at
``gemm_a8w8_8wave.py:191`` where only 8 addresses are distinct, and 8
``buffer_load_dword`` at :199 where only 4 are -- 12 of 24 loads redundant, plus
12 redundant address computations, 4,276 cycles or 0.8% of the kernel. The
compiler cannot merge them because all four calls write the same ``reg_f32_*``
register buffer, which makes them a chain of overwrites rather than pure loads.

``store_all`` loads each scale once. It removes exactly the predicted loads --
A-scale 16 -> 8, B-scale 8 -> 4 -- and is slightly **slower**:

    variant                VGPR  scratch  instrs  dwordx4  dword    gemm
    --permlane --lane-transpose  254    0B     1581      72     16   33.32us
      + --hoist-scales           256    0B     1595      68      8   33.96us

Keeping both scale sets live across all four stores costs more in register
moves and re-materialised addresses than the twelve loads were worth, and it
spends the last 2 VGPRs of headroom (254 -> 256). Bit-exact either way.

Kept as a switch rather than deleted: the redundant fraction grows with
``N_TILES_B``, so a larger ``BLOCK_N`` would change the arithmetic, and if
register pressure ever loosens this flips sign. It is also cheaper to re-measure
a flag than to re-derive why it was rejected.

Two limits. **It is worth nothing on ``fused-lsa``**, where the store goes to a
peer over xGMI rather than to local memory::

                        gemm    barrier  reduce  gather     sum
    fused-lsa          177.4us     5.4    11.1   135.3    329.1
    + all three        175.6us     7.8    11.1   135.0    329.4

against 39.0 -> 35.7us for the same three stages on the split path's local
store. Stage 3 buys the local memory pipeline's sensitivity to lane adjacency;
a peer store crosses the fabric instead, and that phase is limited by ~44 GB/s
of link bandwidth, which no amount of transaction shaping changes. (This was
first "measured" with only stages 1 and 2 wired into the direct path -- which
are exactly the two that do *not* speed the store up -- and the conclusion was
right by accident. The direct-LSA branch for stage 3 exists now, and the
argument-validation above is there so a missing branch fails loudly instead of
silently running a weaker variant.)

And 64 bytes is one wave's ceiling here, since a wave owns
``N_TILES_B * 16 = 32`` columns; gcnasm reaches a full 128-byte line only by
having four ``wave_id_n`` waves tile adjacent 16-column runs.

All three stages are candidates to push back into aiter's ``StoreC``: bit-exact,
no register cost, and the 6.7% is on the GEMM itself, independent of any
all-reduce.

## Persistent tiles: tried, and there is no register budget for it

gcnasm builds this kernel family both ways (``PERSISTENT=1|0``) and its README
has a tail-balance sweep: the win is entirely a function of the remainder after
whole 256-CTA batches -- +9.97% at remainder 8, +8.41% at 32, and only +0.77% at
192. wo_b is 16 x 28 = 448 tiles on 256 CUs, i.e. remainder **192**, the benign
end of that curve.

The tail itself is real here, and large. Sweeping N at M=4096, K=1024
(gemm-only, best of 3, 8 ranks):

    N       tiles  remainder   time
    4096      256      0      32.64us
    4352      272     16      45.88us     <- 6% more work, 40% more time
    4608      288     32      44.68
    5120      320     64      45.24
    6144      384    128      47.40
    7168      448    192      49.80

So it was implemented (one workgroup per CU, striding over tiles) and made
bit-exact. It is a ~2x regression, and the reason is a hard resource wall rather
than anything to do with scheduling. From the kernel metadata in the final ISA:

    variant                 VGPR  AGPR  SGPR  V-spill  scratch  instrs   gemm
    as committed             256     0    50        0       0B    2052  38.54us
      + 3-stage C store      254     0    51        0       0B    1573  35.74us
    body inside an scf.for   256     0    68       38     156B    2260  74.72us
    persistent, grid 256     256     0    67       38     156B    2264  72.22us

**The baseline already uses all 256 VGPRs with zero spill.** Wrapping the
pipeline in a runtime loop asks for 38 more -- the K pipeline's accumulators and
operand fragments die at the end of a tile in the flat version, but a loop makes
the compiler assume they may be live across the back-edge -- and there is nowhere
to put them. ``--amdgpu-num-vgpr 256/512`` changes nothing, because the cap was
never the constraint. Persistent and non-persistent spill identically, which
confirms the cost is ``scf.for`` itself and not the scheduling idea.

Note ``fly-promote-regmem-to-vectorssa`` is **not** the problem, contrary to what
an earlier version of this note claimed. It handles ``scf::ForOp``, it promoted
all 451 register allocas here (zero left afterwards, zero ``llvm.alloca`` in the
final IR), and the emitted loop carries no ``iter_args`` at all. The 156 bytes
are ordinary register-allocator spill: ``.vgpr_spill_count: 38``, 38
``scratch_store_dword`` / 38 ``scratch_load_dword``.

The pass does have one real inefficiency, just not one we hit: an alloca declared
*outside* a loop is carried as an ``iter_arg`` unconditionally, even when it is
fully overwritten every iteration, because ``collectTouchedRegAllocaInRegion``
records on any load *or* store with no liveness test. Allocas declared *inside*
the loop are correctly materialised as ``ub.poison`` and not carried, and ours
are all inside.

Two bugs the attempt surfaced, worth knowing if anyone loops this pipeline:

* **The half-wave barrier does not survive a loop.** The prologue's
  ``if wave_m == 1: rocdl.s_barrier()`` deliberately runs the two half-waves one
  barrier out of phase. Fine once; in a loop the offset accumulates by one per
  tile, so from the second tile on the halves rendezvous at mismatched program
  points and the LDS double-buffering races -- silently, as partial corruption
  inside otherwise-correct tiles. A compensating ``if wave_m == 0: s_barrier()``
  at the end of each tile fixes it exactly.
* **The LDS handles must be rebound per tile.** The pipeline swaps those Python
  bindings as it advances, so hoisting them makes eight shared-address-space
  pointers loop-carried, which fails to legalize.

And a FlyDSL gotcha: ``range(...)`` must appear literally in the ``for``
statement or the AST rewriter does not see it and Python evaluates it eagerly
("dynamic 'ArithValue' has no Python integer representation"). Assigning it to a
variable first does not work, so a kernel cannot cheaply offer both a looped and
a flat form from one body.

Reviving this needs 38 VGPRs from somewhere. The 3-stage C store is the only
change measured to *reduce* pressure (256 -> 254, and 2052 -> 1573 instructions,
since 128 two-byte stores need 128 addresses and 128 predicates live at once),
and it is nowhere near enough. Halving BLOCK_M would free roughly 16 by halving
the accumulator count, but moves the tile count to 896 -- remainder 128, where
gcnasm measured +0.41%. There is no version of this that pays.

## Direct LSA (``--mode fused-lsa``): correct now, still slower than split

gcnasm's best mode has the GEMM epilogue store *straight into the destination
rank's window*: no staging buffer, no copy engine, nothing to publish mid-kernel.
Ported here, the structural half is emphatic -- the scatter collapses from
136.8us to 10.1us, i.e. the transfer is entirely absorbed, which fused-sdma never
managed. And the un-coalesced store is nowhere near as bad as feared: 7.34MB per
link in 166.6us is ~44 GB/s, 82% of the SDMA scatter's 53.7, against the 0.26x
gcnasm measured for a lane-scatter pushed to a peer.

It was wrong for a long time, and the fix is one line in the right place. The LSA
2-stage all-reduce publishes from the kernel that produced the data --
``ar/kernels_lsa.py`` fences right after its ``tmp`` stores, in every block.
Direct LSA's producer is the GEMM, and the only fence was in the *separate*
barrier kernel: one block, therefore one XCD's L2 out of eight. The other seven
kept the peer-homed lines dirty, and the LSA flag, being a system-scope atomic,
overtook them. Moving the fence into the GEMM fixes it: **10/10 runs bit-correct**
where it had been 2-3 in 6.

``--direct-fence leader`` (thread 0 only, the default) rather than every lane is
worth 80us, 434 -> 355. It is legal only because the half-wave barrier pair is
now closed -- ``wait_barrier(0)`` really does mean every wave's stores have
retired into this CU's L2, so one wave writing it back covers all eight. Measured
as incorrect before that fix, which is what made a per-wave release look
mandatory.

Two things that did *not* work, and are worth not re-trying:

* **Uncached (sc0|sc1) peer stores**, so there would be nothing to publish. Wrong
  consistently, ~1.7e-2, at every store width. The first time this was tried the
  store was one bf16 and the explanation looked like partial-line writes losing
  updates over the fabric; with ``--permlane --lane-transpose`` making it 16
  bytes per lane and 64 contiguous per row it is *still* wrong, so that
  explanation was not it and the real one is unknown.
* An uncached recv load in the reduce. No effect, which is what rules out a stale
  read on the consumer side.

It does not beat the split path. Per-kernel (rocprofv3, 8 ranks, [4096,7168]
K=1024, with all three C-store stages):

    fused-lsa   gemm 189.4  barrier 10.1  reduce 10.9  gather 135.6   sum 346.0
    split-lsa   gemm  35.7  + one LSA all-reduce kernel                sum ~310

    end to end, median of 4:   split-lsa 318.8us     fused-lsa 359.0us

The 41us gap is structural, not tuning. LSA's 2-stage does read + reduce + write
in a single pass over the wire, so its reduce is free; Direct LSA writes to the
peer and then pays a separate 10.9us local reduce pass, its store rate is ~14%
under LSA's read rate, and it adds the publishing fence LSA gets from being
local. Absorbing the scatter completely still does not cover that.

## What gcnasm does differently, and why its GEMM+a2a wins

``/workspace/gcnasm/opus_gemm_dist/opus_gemm_a2a_lsa`` fuses a GEMM with an
all-to-all and gets 16-27% out of it. Reading it explains most of why this does
not, and one of its lessons was worth ~90us here.

1. **Its collective is one phase; this one is three.** An a2a scatters the GEMM
   output once. An all-reduce is scatter + reduce + all-gather, and fusion only
   touches the scatter -- reduce + gather is 150us of the 327us baseline, 46%,
   untouchable by construction.
2. **Its ratio is inverted.** M=2048 N=18432 K=8192 gives ~518us of GEMM against
   ~200us of comm, 2.6:1. wo_b is 39us against 137us per phase, 0.29:1. Overlap
   can hide at most the smaller of the two, so theirs hides most of the comm and
   this hides at most one GEMM.
3. **Its best mode has no producer/consumer handoff at all.** "Direct LSA" has
   the GEMM epilogue store *straight into the destination rank's buffer*. No
   staging, no copy engine, nothing to publish mid-kernel -- the only sync is the
   barrier at the end. The publication problem this file spends all its time on
   simply does not exist there.
4. **Its fused-SDMA path uses no cache fence.** ``opus_chunk_sdma_submit`` is
   ``s_waitcnt vmcnt(0)``, ``s_barrier``, an ``__ATOMIC_ACQ_REL`` counter, and a
   per-destination submit spin lock -- no ``__threadfence_system``, no
   ``buffer_wbl2``. That is the lesson that transferred: the acq_rel counter *is*
   the release, and the explicit fence added here was 90us of pure waste
   (fused 440us -> 350us on removing it). Its ISA shows the atomic already emits
   its own ``buffer_wbl2``/``buffer_inv`` pair, on thread 0 only.

## The chunks race: it was the GEMM, and it is gone

``--chunks`` > 1 produced wrong output about 3 runs in 10 and was pinned to 1
for that reason. The cause was not the chunk protocol at all: it was aiter's
8-wave GEMM under-counting one ``s_waitcnt`` in its main loop, which corrupted
output non-deterministically on large grids whatever the epilogue did (see the
comment at that ``wait_barrier`` below). Since that fix, at [16384, 7168]
K=2048 on 8 ranks:

* chunks 1/2/4/8, 10 runs each -- 40/40 correct, no hang;
* chunks=8 alone, 25 more runs -- 25/25 correct, no hang.

Two hangs were seen at 8 ranks while the sweep was still being set up and never
reproduced in the 100+ runs after. The submit lock is a plain test-and-set spin
(``_acquire_peer_lock``), so a hang is not impossible; use ``timeout`` when
sweeping and treat one as a finding rather than a flake.

Two things in this file survived that misdiagnosis and are worth keeping
straight:

* The half-wave barrier pairing (``if wave_m == 0: s_barrier()`` before
  ``store_c``) is still needed and still right -- gcnasm does it
  unconditionally at kernel_template.hpp:693. It took ``--chunks 2`` from
  1-in-3 failing to 3-in-10, which at the time read as "better but not fixed";
  the residual 3-in-10 was the GEMM.
* Two *other* diagnoses were wrong and are recorded so they are not retried.
  **A shared SDMA queue**: gcnasm's per-destination submit lock was ported and
  ISA-verified; it fixed nothing, and is kept only because cco's
  one-issuing-warp-per-queue rule still applies. **The counter atomic's
  ordering**: ``acq_rel`` appeared to beat ``monotonic``, which is how the
  acquire half got justified; against a 1-in-3 intermittent failure that
  comparison was noise. ``acq_rel`` stays because release/acquire is right for
  a producer handing tiles to a consumer, not because it was measured -- and at
  chunks=1 it demonstrably orders nothing, since 20 runs of ``monotonic`` pass
  too.

Repeated runs are still the only way to judge any of this, and the gate has to
be tight: the corruption landed at 4-9e-3 against an fp8 floor of 2.35e-3, so a
5e-3 threshold reported a corrupt run as validated (it did, at 3.97e-3). The
bench gates at 3e-3, and ``test_fused_is_stable_across_repeats`` requires three
runs to be *identical* rather than each small.

## Result: fused-sdma wins, once the chunks are unblocked

8 ranks, [16384, 7168] K=2048 -- the real prefill shape -- graph replay, all
three C-store stages on (now the benchmark default), median of 31, max over
ranks:

    mode                        time
    fused-sdma  chunks=8      1114.1us   <- best
    split-sdma                1261.7
    split-lsa                 1262.4
    fused-lsa                 1571.4
    gemm-only                  228.9

Per-kernel, the overlap is visible directly:

    chunks   GEMM   drain  reduce  gather   total
         1  257.5   493.8    44.0   496.4  1291.7
         2  266.5   379.4    44.2   496.4  1186.6
         8  265.0   303.7    43.8   496.6  1109.1

The drain falls 38% while the GEMM grows 7.5us for the extra counter atomics
and the lock. Going finer is worse: at chunks=8 each PUT is 3.5 MiB, and 16
(via ``--block-m 128``) halves that to 1.75 MiB, under the knee in the SDMA
bandwidth curve, for 1130.8us.

fused-lsa still loses, and for a reason that is not going away: it spends 730us
of its GEMM pushing C over xGMI where the copy engines move the same bytes in
499, and ATT shows 99% of that store time is *stall*, so coalescing the stores
(the three C-store stages) buys it nothing -- 954.5 -> 952.1us.

The remaining floor is the tail: reduce plus all-gather is 540us of the 1114,
and neither is touched by fusing the GEMM.
