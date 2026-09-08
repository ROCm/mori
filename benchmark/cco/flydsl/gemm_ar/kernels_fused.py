# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""aiter's 8-wave fp8 GEMM with the all-reduce's scatter fused into its epilogue.

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

## Direct LSA (``--mode fused-lsa``): the structure works, the store does not

gcnasm's best mode has the GEMM epilogue store *straight into the destination
rank's window*, so there is no staging buffer, no copy engine and nothing to
publish mid-kernel. Ported here, and the structural half is emphatic
(rocprofv3 --kernel-trace, us):

    config                gemm   scatter/barrier  reduce  gather    sum
    split-sdma            39.1        136.8         13.5   137.3   326.8
    fused-sdma c=1        62.7        129.0         13.0   136.2   341.0
    fused-lsa (Direct)   166.6         13.3         11.1   135.5   326.6

The scatter collapses from 136.8us to 13.3us -- the transfer is *entirely*
absorbed, which is what fused-sdma never managed. And the un-coalesced store is
nowhere near as bad as expected: 7.34MB per link in 166.6us is ~44 GB/s, 82% of
the SDMA scatter's 53.7, against the 0.26x gcnasm measured for a lane-scatter
pushed to a peer.

It is nonetheless **not correct**, and the reason makes the ``StoreC`` coalescing
rewrite a correctness prerequisite rather than an optimisation:

* **Cached peer stores validate 2-3 runs in 6.** The lines are homed in the
  peer's memory but sit dirty in whichever XCD's L2 the block ran on. Nothing
  downstream can clean that up -- the barrier kernel is one block, and
  ``buffer_wbl2`` writes back only the L2 of the XCD it runs on, one of eight.
* **Uncached (sc0|sc1) peer stores are consistently wrong** (relL2 ~3e-2, never
  right) **and slower** (426us against 341). ``_store_bf16`` writes one bf16, so
  bypassing L2 makes every store a 2-byte partial-line write; concurrent sub-32B
  writes into the same sector across the fabric lose updates. Adding an uncached
  recv load in the reduce does not change it, so this is the store, not a stale
  read.

So L2 merging is what makes the narrow store work at all, and L2 residency is
what makes it invisible. The two are only separable by widening the store --
gcnasm's wave-local ``ds_bpermute`` pair-coalesced C store (their "C-store mode
2"; staging C through LDS regressed 1-3%). That is why they needed it.

Even coalesced, this does not obviously win. The fused sum is 326.6us against
split-lsa's ~324: LSA's 2-stage does read + reduce + write in a single pass over
the wire, so its reduce is free, whereas Direct LSA writes to the peer and then
pays a separate local reduce pass -- and its store rate is ~18% under LSA's read
rate. The prize for coalescing is correctness plus maybe a few percent, not a
step change.

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

## The race that pinned chunks=1: a barrier phase, not a queue

The prologue's ``if wave_m == 1: rocdl.s_barrier()`` gives waves 4-7 one *extra*
barrier. ``s_barrier`` is a counting rendezvous, so from then on every arrival
pairs waves 4-7's k-th barrier with waves 0-3's (k+1)-th, and waves 0-3 run one
phase ahead for the rest of the kernel. A one-shot GEMM does not care: the
trailing unmatched barrier is released when the other half exits, and the offset
is the point -- it staggers the two halves of the M dimension.

The fused epilogue does care. ``wait_barrier(0)`` after ``store_c`` is supposed
to mean "every wave's C tile has retired". Under the offset it instead
rendezvouses waves 0-3 -- which contain thread 0, hence the counter and the
transfer -- with waves 4-7 sitting at the *previous* barrier, before their
stores. Thread 0 then counts the tile complete and can push it while half of it
is still unwritten.

One ``if wave_m == 0: rocdl.s_barrier()`` before ``store_c`` re-balances the
counts and fixes it: ``--chunks 2`` goes from failing about one run in three to
12 of 12, and it is the configuration that actually overlaps.

Two things were blamed for this before it was found, and both were wrong:

* **A shared SDMA queue.** With two chunks per destination the modulo test elects
  two winners that both post to queue ``dest``, so gcnasm's per-destination
  submit lock was ported (ISA-verified: test-and-set with ``s_sleep`` backoff).
  It fixed nothing. It is kept because cco's rule of at most one issuing warp per
  queue still applies, but it was not the bug.
* **The counter atomic's ordering.** ``acq_rel`` appeared to lower the failure
  rate against ``monotonic``, which is how the acquire half got justified. With a
  one-in-three intermittent failure that comparison was noise. ``acq_rel`` stays
  because release/acquire on the counter is right for a producer handing tiles to
  a consumer, not because it was measured.

Repeated runs remain the only way to judge any of this;
``test_fused_is_stable_across_repeats`` requires three runs to be *identical*,
not merely each small, because every individual relL2 here looks plausible.

## Result: fused-sdma reaches parity, and does not beat LSA

8 ranks, [4096, 7168], K=1024, graph replay, median of 3-4 runs:

                             default C-store   + 3-stage C-store
    split-lsa                      326.8us            321.8us
    split-sdma                     334.6              330.9
    fused-sdma  chunks=2           326.2              323.0
    fused-lsa                      341.0    racy, see above

Per-kernel for the SDMA paths (rocprofv3 --kernel-trace, us):

    config                gemm   scatter/drain  reduce  gather    sum
    split-sdma            39.1        136.8      13.5   137.3   326.8
    fused-sdma chunks=2   60.9        126.9      13.1   135.1   336.1

So the fusion is no longer a loss -- it beats split-sdma by ~2.5% and ties
split-lsa -- but it does not win. The overlap it buys (scatter 136.8 -> 126.9)
is roughly cancelled by what the epilogue costs the GEMM (39.1 -> 60.9), and the
reduce and all-gather, 46% of the pipeline, are untouched by construction.

An ATT thread trace puts the epilogue at 26% of the kernel's latency, almost all
of it one instruction waiting on the release fence's writeback -- 13.9% of the
kernel, more than any main-loop stall. The fence discussion under ``--fence``
covers why that is quadratic and what did and did not help.

## Pinned copy

``_fused_kernel_body`` is a copy of ``compile_fp8_gemm_8w``'s ``kernel_gemm``
(aiter ``aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py``) with the epilogue tail
added. It is a copy rather than a hook because the original's main loop contains
a runtime ``if wave_m == 1``, and FlyDSL only rewrites ``if`` inside the
``@flyc.kernel`` function's own AST -- factoring the body into a shared helper
makes that line raise at trace time. Every reusable piece (``G2SLoader``,
``S2RLoader``, ``StoreC``, ``Mfma16x16x128``, the swizzles) is imported rather
than copied, so the duplication is the ~90-line pipeline only.

``tests/python/cco/test_gemm_ar.py`` asserts this kernel's C output is
bit-identical to aiter's unfused kernel on the same inputs, which is what stops
the copy from rotting silently.
"""

# NOTE: no `from __future__ import annotations` here, deliberately, and aiter's
# gemm_a8w8_8wave.py omits it for the same reason: `fx.struct` reads the
# `SharedStorage` field annotations as live objects, and PEP 563 would hand it
# the string "fx.Array[fx.Float8E4M3FN, a_lds_size, 16]" -- whose size operands
# are locals of this factory and so cannot be resolved from the module globals.

import os
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int64
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    StoreC,
    _xcd_swizzle_any,
    ceildiv,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    wait_barrier,
)
from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import split_row_major_2d

import mori.cco.device.flydsl as cco
from mori.cco.device.flydsl import _bindings as raw_cco

_AR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ar")
sys.path.insert(0, _AR_DIR)
from flydsl._mlir import ir  # noqa: E402
from flydsl._mlir.dialects import llvm as _llvm_d  # noqa: E402
from flydsl._mlir.dialects import scf  # noqa: E402

from _compat import (  # noqa: E402
    CM_SC0_SC1,
    atomic_add_u32,
    atomic_store_u32,
    atomic_xchg_u32,
    buffer_store,
    create_buffer_resource_from_addr,
    i32_type,
    wave_uniform_i64,
    buffer_store,
    create_buffer_resource_from_addr,
    release_fence,
    signal_ptr,
)


def _acquire_peer_lock(lock_ptr):
    """Test-and-set spin on one destination's submit lock.

    Transferred from gcnasm ``opus_chunk_sdma_submit``: with more than one chunk
    per destination the tile counter elects one block per *chunk*, so two of them
    can reach the SDMA submit for the same queue at once. Serialising only the
    submit -- not the GEMM work -- is the point.

    Termination: the contending blocks are elected in chunk order, and chunk c's
    tiles are dispatched before chunk c+1's, so a spinning block is always
    waiting on one that has already run.
    """
    i32 = i32_type()
    first = atomic_xchg_u32(lock_ptr, 1)
    loop = scf.WhileOp([i32], [first])
    cond = ir.Block.create_at_start(loop.before, [i32])
    body = ir.Block.create_at_start(loop.after, [i32])
    with ir.InsertionPoint(cond):
        held = fx.Uint32(fx.Int32(cond.arguments[0])) > fx.Uint32(fx.Int32(0))
        scf.ConditionOp(held.ir_value(), [cond.arguments[0]])
    with ir.InsertionPoint(body):
        fx.rocdl.s_sleep(1)
        nxt = atomic_xchg_u32(lock_ptr, 1)
        scf.YieldOp([nxt])


class _RawWriteThroughStoreC(StoreC):
    """``StoreC`` whose C stores really do bypass L1 *and* L2.

    ``StoreC`` stores through a copy atom whose ``cache_modifier`` is a two-value
    enum (0=cached, 2=nt), so ``sc0|sc1`` is not expressible there -- asking for
    it emits a plain ``sc0`` and leaves the line dirty in L2, which is why the
    earlier ``--fence writethrough`` probe measured nothing. This goes around the
    atom to ``raw_ptr_buffer_store``, whose ``aux`` operand is the real cache
    policy, so the data reaches memory at the store and the per-block
    ``buffer_wbl2`` release becomes unnecessary.

    The descriptor is built from the window rather than from the ``C`` argument
    because in the fused kernel they are the same bytes: C *is* the all-reduce's
    input region. ``num_records`` still bounds it, so ``store``'s out-of-bounds
    index for masked columns is dropped exactly as before.

    Deliberately *not* coalesced: the point of this variant is to find out
    whether the coalescing rewrite is needed before paying for it.
    """

    def __init__(self, *args, c_rsrc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._c_rsrc = c_rsrc

    def _store_bf16(self, value_bf16, c_index):
        buffer_store(value_bf16, self._c_rsrc, c_index, cache_modifier=CM_SC0_SC1)


class _SwappedMfma:
    """``Mfma16x16x128`` with the A/B operands exchanged at every call.

    Thin on purpose: the accumulator registers, the atom and ``zero_value`` are
    the wrapped object's, only the operand order and the ``idx`` argument order
    change. ``idx(ti, tj)`` here forwards to the wrapped ``idx(tj, ti)`` so the
    store can keep indexing in (M-tile, N-tile) order.
    """

    def __init__(self, inner):
        self._inner = inner
        self.zero_value = inner.zero_value

    def idx(self, i, j):
        return self._inner.idx(j, i)

    def call(self, a, b, c, *, set_prio=True):
        return self._inner.call(b, a, c, set_prio=set_prio)


class _SwapABStoreC(StoreC):
    """C store for an A/B-swapped MFMA: 4 consecutive N per lane -> 64-bit store.

    ``fx.gemm(atom, c, b, a, c)`` computes ``B^T A^T = (A B)^T`` in the
    accumulator's native layout, so lane ``l`` value ``k`` moves from
    ``D[4*(l/16)+k][l%16]`` (4 consecutive *rows*, stride c_cols) to
    ``D[l%16][4*(l/16)+k]`` (4 consecutive *columns*). In a row-major C that is
    8 contiguous bytes, which is the whole point -- the unswapped layout can only
    ever emit ``buffer_store_short``. This is gcnasm's ``mfma_adaptor_swap_ab``,
    which does literally ``base::operator()(b, a, c)`` and redefines ``dim_c()``
    to match (opus.hpp:2064).

    The scale loads swap with it, and stay equally well vectorized: A's scale is
    indexed by row, which is now ``lane % 16`` -- one row, so a scalar load --
    while B's is indexed by column, now 4 consecutive, so a vec4.
    """

    def __init__(self, *args, peer_rsrc=None, elem_base=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
        self.reg_bf16_4 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
        self.out_atom_8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        self.reg_bf16_8 = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
        self._peer_rsrc = peer_rsrc
        self._elem_base = elem_base

    def _load_a_scale_scalar(self, row):
        fx.copy(
            self.scale_atom_1,
            fx.slice(self.sa_div, (None, fx.Int32(row))),
            self.reg_f32_1,
        )
        return Vec(fx.memref_load_vec(self.reg_f32_1))[0]

    def _load_b_scale_vec4(self, col):
        fx.copy(
            self.scale_atom_4,
            fx.slice(self.sb_div, (None, fx.Int32(col))),
            self.reg_f32_4,
        )
        return Vec(fx.memref_load_vec(self.reg_f32_4))

    def _store_bf16x4(self, values, c_index):
        fx.memref_store_vec(Vec.from_elements(values, fx.BFloat16), self.reg_bf16_4)
        fx.copy(
            self.out_atom_4,
            self.reg_bf16_4,
            fx.slice(self.c_div, (None, fx.Int32(c_index))),
        )

    def store(self, c_frag, base_row, base_col):
        lane = self.lane_id
        a_scales = [
            self._load_a_scale_scalar(base_row + ti * 16 + lane % 16)
            for ti in range_constexpr(self.n_tiles_a)
        ]
        b_scales = [
            self._load_b_scale_vec4(base_col + tj * 16 + (lane // 16) * 4)
            for tj in range_constexpr(self.n_tiles_b)
        ]
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + lane % 16
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + (lane // 16) * 4
                # BLOCK_N divides N in every shape this builds for, so the four
                # columns are either all in range or all out; one predicate.
                col_valid = col + 3 < self.c_cols
                oob = fx.Int32(self.c_rows * self.c_cols)
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                vals = [
                    (vec_f32[k] * (a_scales[ti] * b_scales[tj][k])).to(fx.BFloat16)
                    for k in range_constexpr(4)
                ]
                c_index = row * self.c_cols + col
                self._store_bf16x4(vals, arith.select(col_valid, c_index, oob))


def _permlane16_swap(x, y):
    """gfx950 ``v_permlane16_swap_b32``: exchange 16-lane rows between two VGPRs.

    Verified on hardware rather than assumed::

        vdst_new = [X.r0, Y.r0, X.r2, Y.r2]
        vsrc_new = [X.r1, Y.r1, X.r3, Y.r3]

    Position *within* a row is preserved, so a lane keeps its ``lane % 16`` and
    therefore its C row -- which is what lets the A-scale stay correct across the
    shuffle.
    """
    st = ir.Type.parse("!llvm.struct<(i32, i32)>")
    raw = lambda v: v.ir_value() if hasattr(v, "ir_value") else v
    res = fx.rocdl.permlane16_swap(st, raw(x), raw(y), False, True)
    i32 = ir.IntegerType.get_signless(32)
    return (
        fx.Int32(_llvm_d.ExtractValueOp(i32, res, ir.DenseI64ArrayAttr.get([0])).result),
        fx.Int32(_llvm_d.ExtractValueOp(i32, res, ir.DenseI64ArrayAttr.get([1])).result),
    )


def _ds_bpermute(value, src_lane):
    """Pull ``value`` from ``src_lane``. Byte-addressed, hence the <<2."""
    i32 = ir.IntegerType.get_signless(32)
    raw = lambda v: v.ir_value() if hasattr(v, "ir_value") else v
    return fx.Int32(
        fx.rocdl.ds_bpermute(i32, raw(fx.Int32(src_lane) * fx.Int32(4)), raw(value))
    )


class _PermlaneStoreC(_SwapABStoreC):
    """A/B-swapped MFMA + one cross-lane step, giving 16B per lane and 64B rows.

    After the swap a lane owns 4 consecutive columns of one N-tile, i.e. 8 bytes,
    and the two N-tiles it holds are 16 columns apart -- so a row still only gets
    32 bytes per instruction. Two ``permlane16_swap`` fix that exactly, with no
    ``ds_bpermute`` and no LDS:

        (A, B) = permlane16_swap(tile0.d0, tile1.d0)
        (C, D) = permlane16_swap(tile0.d1, tile1.d1)
        lane group g stores (A, C, B, D)

    which lands g = 0,1,2,3 on columns 0-7, 16-23, 8-15, 24-31. The four groups
    together cover columns 0..31 contiguously: 64 bytes per row, from 16 bytes per
    lane. The column permutation is absorbed into the address, so it is free.

    Scaling happens before the shuffle, in the lane that owns the value; the
    shuffle only moves whole 16-lane rows, so every lane keeps its own C row.
    """

    _adjacent_probe = False
    _lane_transpose = False

    def store(self, c_frag, base_row, base_col):
        assert self.n_tiles_b == 2, (
            "the permlane mapping pairs exactly two N-tiles (BLOCK_N == 256); "
            f"got n_tiles_b={self.n_tiles_b}"
        )
        lane = self.lane_id
        grp = lane // 16
        a_scales = [
            self._load_a_scale_scalar(base_row + ti * 16 + lane % 16)
            for ti in range_constexpr(self.n_tiles_a)
        ]
        b_scales = [
            self._load_b_scale_vec4(base_col + tj * 16 + grp * 4)
            for tj in range_constexpr(self.n_tiles_b)
        ]
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + lane % 16
            dwords = []
            for tj in range_constexpr(self.n_tiles_b):
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                packed = Vec.from_elements(
                    [
                        (vec_f32[k] * (a_scales[ti] * b_scales[tj][k])).to(fx.BFloat16)
                        for k in range_constexpr(4)
                    ],
                    fx.BFloat16,
                ).bitcast(fx.Int32)
                dwords.append((packed[0], packed[1]))
            a, b = _permlane16_swap(dwords[0][0], dwords[1][0])
            c, d = _permlane16_swap(dwords[0][1], dwords[1][1])
            out8 = Vec.from_elements([a, c, b, d], fx.Int32).bitcast(fx.BFloat16)
            if const_expr(self._lane_transpose):
                # gcnasm's second stage (kernel_template.hpp:491-510), as one
                # ds_bpermute per dword. The permlane stage leaves a row's four
                # 8-column chunks in lanes 16 apart, so a 16-lane group touches
                # 16 rows at 16 bytes each. Transposing the lane index -- lane
                # l' = 4r'+q' pulls from the lane holding (row r', chunk q') --
                # puts adjacent lanes on one row, so lanes 0-3 write 64
                # contiguous bytes and a group covers 4 rows instead of 16.
                #
                # chunk q' lives at column q'*8, and the permlane stage put
                # column (g%2)*16 + (g//2)*8 in group g, so g = swap of q's two
                # bits: 0,1,2,3 -> 0,2,1,3.
                q = lane % 4
                src_lane = ((q % 2) * 2 + q // 2) * 16 + lane // 4
                a, c, b, d = (_ds_bpermute(v, src_lane) for v in (a, c, b, d))
                out8 = Vec.from_elements([a, c, b, d], fx.Int32).bitcast(fx.BFloat16)
                row = base_row + ti * 16 + lane // 4
                col = base_col + q * 8
            elif const_expr(self._adjacent_probe):
                # PERF PROBE, output wrong by construction. Same 64 addresses,
                # reassigned so *adjacent* lanes cover one row (lanes 0-3 -> row 0,
                # 64 contiguous bytes) instead of lanes 16 apart. Prices gcnasm's
                # ds_bpermute lane transpose before writing it: if the address
                # coalescer works per 16-lane group rather than per wave, this
                # takes a group from 16 sectors to 4.
                row = base_row + ti * 16 + lane // 4
                col = base_col + (lane % 4) * 8
            else:
                col = base_col + (grp % 2) * 16 + (grp // 2) * 8
            oob = fx.Int32(self.c_rows * self.c_cols)
            idx = arith.select(col + 7 < self.c_cols, row * self.c_cols + col, oob)
            if self._peer_rsrc is not None:
                buffer_store(out8, self._peer_rsrc, fx.Int32(idx) - self._elem_base)
            else:
                fx.memref_store_vec(out8, self.reg_bf16_8)
                fx.copy(
                    self.out_atom_8, self.reg_bf16_8,
                    fx.slice(self.c_div, (None, fx.Int32(idx))),
                )


class _LaneTransposeStoreC(_PermlaneStoreC):
    """``_PermlaneStoreC`` plus gcnasm's ds_bpermute lane transpose."""

    _lane_transpose = True


class _AdjacentLaneProbeC(_PermlaneStoreC):
    """PERF PROBE ONLY -- see the note in ``_PermlaneStoreC.store``."""

    _adjacent_probe = True


class _WideStoreProbeC(_SwapABStoreC):
    """PERF PROBE ONLY -- the output is wrong on purpose.

    Answers "is 16B per lane / 64B per row actually faster?" before anyone builds
    the cross-lane shuffle that would make it correct. It emits exactly the access
    pattern the shuffled version would have -- one 128-bit store per lane at
    ``col = (lane//16)*8``, so 4 lanes cover a row's 32 columns as 64 contiguous
    bytes -- but feeds it the two N-tiles' values concatenated, which is not what
    belongs at those addresses.

    Correct would need lane ``(a, r)`` to gather cols ``8a..8a+7``, which live in
    two *other* lanes' registers (``a'=2a`` and ``2a+1`` of one tile). That is the
    ``permlane16_swap`` step; this probe skips it and keeps only its cost profile.
    """



    def store(self, c_frag, base_row, base_col):
        lane = self.lane_id
        a_scales = [
            self._load_a_scale_scalar(base_row + ti * 16 + lane % 16)
            for ti in range_constexpr(self.n_tiles_a)
        ]
        b_scales = [
            self._load_b_scale_vec4(base_col + tj * 16 + (lane // 16) * 4)
            for tj in range_constexpr(self.n_tiles_b)
        ]
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + lane % 16
            vals = []
            for tj in range_constexpr(self.n_tiles_b):
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                vals += [
                    (vec_f32[k] * (a_scales[ti] * b_scales[tj][k])).to(fx.BFloat16)
                    for k in range_constexpr(4)
                ]
            col = base_col + (lane // 16) * 8
            oob = fx.Int32(self.c_rows * self.c_cols)
            c_index = row * self.c_cols + col
            idx = arith.select(col + 7 < self.c_cols, c_index, oob)
            if self._peer_rsrc is not None:
                buffer_store(
                    Vec.from_elements(vals, fx.BFloat16),
                    self._peer_rsrc,
                    fx.Int32(idx) - self._elem_base,
                )
            else:
                fx.memref_store_vec(
                    Vec.from_elements(vals, fx.BFloat16), self.reg_bf16_8
                )
                fx.copy(
                    self.out_atom_8,
                    self.reg_bf16_8,
                    fx.slice(self.c_div, (None, fx.Int32(idx))),
                )


class _PeerDirectStoreC(StoreC):
    """``StoreC`` that writes its tile straight into the owning peer's window.

    This is gcnasm's "Direct LSA" shape (``opus_gemm_a2a_lsa``), and its whole
    point is that there is no producer/consumer handoff inside the kernel: the
    bytes go to the destination as they are produced, so nothing has to be
    published mid-kernel, there is no completion counter, no submit lock and no
    release fence. The only synchronisation left is the cross-rank barrier after
    the kernel, where a kernel-end release has already happened for free.

    A block's rows all belong to one destination (``block_m // m_tiles_per_peer``),
    so the peer descriptor is uniform per block and is built once.

    ``elem_base`` rebases the global row index onto the destination's slice.
    ``StoreC.store`` masks out-of-range columns by redirecting them to
    ``c_rows * c_cols``, and that stays out of range after rebasing: the largest
    ``elem_base`` is ``(world-1) * slice_rows * c_cols``, which leaves exactly
    ``slice_rows * c_cols`` elements -- one past the end of the descriptor.
    """

    def __init__(self, *args, peer_rsrc=None, elem_base=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._peer_rsrc = peer_rsrc
        self._elem_base = elem_base

    def _store_bf16(self, value_bf16, c_index):
        # Cached, and that is the unresolved half of this variant. See the
        # "Direct LSA" note in the module docstring: uncached (sc0|sc1) is
        # *consistently* wrong here because a 2-byte store bypassing L2 is a
        # partial-line write, and cached is only intermittently right because the
        # dirty peer-homed lines cannot be flushed out of eight XCDs' L2s
        # afterwards. Coalescing the store first resolves both.
        buffer_store(value_bf16, self._peer_rsrc, c_index - self._elem_base)


class _NonTemporalStoreC(StoreC):
    """``StoreC`` with non-temporal C stores (the copy atom's only other mode).

    Probes whether ``buffer_wbl2`` gets cheaper when the lines it is asked to
    flush have already been evicted. The atom's ``cache_modifier`` is a two-value
    enum, 0=cached / 2=nt -- it is *not* the aux bitmask that
    ``raw_ptr_buffer_store`` takes, so sc0|sc1 cannot be requested this way.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(2), fx.BFloat16)


class _WriteThroughStoreC(StoreC):
    """``StoreC`` whose C stores bypass L1 and L2 instead of being written back.

    The only change is the copy atom's cache modifier. It exists because a
    per-block release fence is quadratic: ``buffer_wbl2`` flushes the *whole* L2,
    so block k redundantly writes back every tile blocks 1..k-1 already wrote,
    448 times over. Writing C through in the first place makes the release free --
    ``s_waitcnt vmcnt(0)`` is then all the copy engine needs -- and adds no
    traffic, since C has to reach memory regardless.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_atom_1 = fx.make_copy_atom(
            fx.rocdl.BufferCopy16b(CM_SC0_SC1), fx.BFloat16
        )


BLOCK_K = 128


def compile_fused_gemm_scatter(
    cfg,
    rank: int,
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    b_preshuffled: bool = True,
    waves_per_eu: int = 2,
    xcd_swizzle: int = 0,
    fuse: bool = True,
    rotated: bool | None = None,
    swap_ab: bool = False,
    store_probe: bool = False,
    permlane: bool = False,
    lane_transpose: bool = False,
    transport: str = "sdma",
    fence: str = "none",
    emit_put: bool = True,
    atomic_order: str = "acq_rel",
):
    """Compile the GEMM, with (``fuse=True``) or without the scatter epilogue.

    ``fuse=False`` emits the identical kernel minus the epilogue tail, so the
    split baseline and the fused kernel differ in exactly one thing. That is why
    the baseline is built here rather than called through aiter's wrapper, which
    would also change the launch path and the C tensor.

    ``rotated`` selects the destination-rotated chunk-major tile order; it
    defaults to ``fuse``, since it only matters when something is watching tiles
    complete. It is separately settable so the benchmark can charge the split
    baseline the same tile order and show that the ordering itself is neutral.

    Returns ``launch(A, B_T, C, A_scale, B_scale, c_m, c_n, dev_comm, win,
    stream=...)``.
    """
    cfg.validate()
    ws = cfg.world_size
    M, N = cfg.m, cfg.n
    if transport not in ("sdma", "lsa"):
        raise ValueError(f"transport must be sdma or lsa, got {transport!r}")
    # The C-store variant is picked by a chain of const_expr branches, so an
    # unimplemented combination does not fail -- it quietly falls through to a
    # weaker one. That already cost a wrong conclusion once: --lane-transpose had
    # no direct-LSA branch, so `--mode fused-lsa --lane-transpose` silently ran
    # stages 1+2 and was measured as "no benefit from stage 3".
    if permlane and not swap_ab:
        raise ValueError("--permlane requires --swap-ab")
    if lane_transpose and not permlane:
        raise ValueError("--lane-transpose requires --permlane")
    if store_probe and not swap_ab:
        raise ValueError("--store-probe requires --swap-ab")
    if store_probe and lane_transpose:
        raise ValueError(
            "--store-probe and --lane-transpose are alternatives: the probe emits "
            "the adjacent-lane access pattern without the shuffle that makes it "
            "correct, which is what --lane-transpose then implements"
        )
    if transport == "lsa" and store_probe and permlane:
        raise ValueError(
            "no peer-direct variant of the adjacent-lane probe; use "
            "--lane-transpose for the real thing on --mode fused-lsa"
        )
    direct_lsa = fuse and transport == "lsa"
    rotated = fuse if rotated is None else rotated

    assert BLOCK_M >= 128 and BLOCK_N >= 256
    assert BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0
    if N % BLOCK_N:
        raise ValueError(
            f"N={N} must be a multiple of BLOCK_N={BLOCK_N}: the tile count per "
            "destination has to be a compile-time constant for the modulo test"
        )
    slice_rows = M // ws
    if rotated and slice_rows % BLOCK_M:
        raise ValueError(
            f"each peer's row slice ({slice_rows}) must be a whole number of "
            f"BLOCK_M={BLOCK_M} tiles, so M must be a multiple of "
            f"world_size*BLOCK_M={ws * BLOCK_M} (got M={M}). Decode shapes are "
            "out of scope for this host anyway -- it pads M below 128."
        )

    K_ITERS = K // BLOCK_K
    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    # Compile-time tile accounting for the completion counters.
    n_blocks_const = N // BLOCK_N
    m_tiles_per_peer = slice_rows // BLOCK_M if slice_rows % BLOCK_M == 0 else 0
    tiles_per_peer = m_tiles_per_peer * n_blocks_const

    # Push granularity. A destination's slice is `m_tiles_per_peer` row-bands of
    # BLOCK_M rows; pushing each band as it completes is what lets every link
    # start streaming early instead of waiting for its whole slice.
    chunks = cfg.counter_chunks if fuse else 1
    if fuse and m_tiles_per_peer % chunks:
        raise ValueError(
            f"counter_chunks={chunks} must divide the {m_tiles_per_peer} "
            f"BLOCK_M={BLOCK_M} row-bands in each peer's slice"
        )
    m_tiles_per_chunk = max(1, m_tiles_per_peer // chunks)
    tiles_per_chunk = m_tiles_per_chunk * n_blocks_const
    chunk_bytes = cfg.slice_bytes // chunks

    if fence not in (
        "all", "agent", "agent-leader", "nt-agent", "leader", "none",
        "writethrough", "wt-agent", "raw-wt", "raw-wt-agent", "raw-wt-leader",
    ):
        raise ValueError(
            f"fence must be all/agent/leader/none/writethrough/wt-agent, got {fence!r}"
        )
    _kname_tag = f"{'S' if swap_ab else ''}{'P' if permlane else ''}{'T' if lane_transpose else ''}{'W' if store_probe else ''}c{chunks}{'r' if rotated else 'l'}{fence[0]}"
    counter_off = cfg.counter_off
    lock_off = cfg.lock_off
    in_off = cfg.input_off
    my_recv_slot = cfg.recv_slot_off(rank) if cfg.recv_slots else 0
    slice_bytes = cfg.slice_bytes

    _kname = (
        f"mori_fused_{(transport if fuse else 'split')}_8w_"
        f"{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_k{K}_{_kname_tag}"
        f"{'p' if emit_put else 'x'}{atomic_order[0]}_r{rank}"
    )

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(name=_kname, known_block_size=[512, 1, 1])
    def kernel_gemm_scatter(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        dev_comm: Int64,
        win: Int64,
    ):
        # ---- begin pinned copy of aiter kernel_gemm ----
        F8_IR_t = fx.Float8E4M3FN.ir_type

        n_blocks = ceildiv(c_n, BLOCK_N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        if const_expr(rotated):
            # Destination-rotated, chunk-major tile order. Linear order finishes
            # destination 0's whole slice, then 1's, ...  -- so the last
            # destination's link only starts at the end of the GEMM and the
            # transfer cannot overlap anything. Walking (chunk, dest, n) instead
            # gets every link streaming within the first 1/(chunks*world) of the
            # GEMM. The `+ rank` rotation keeps all 8 ranks from pushing at the
            # same destination simultaneously (gcnasm's opus_direct_stripe_tile).
            # split_row_major_2d(i, n) -> (i // n, i % n)
            rest, bn = split_row_major_2d(fx.block_idx.x, n_blocks)
            tile_i, dest_seq = split_row_major_2d(rest, ws)
            dest_i = (dest_seq + fx.Int32(rank)) % fx.Int32(ws)
            block_m = dest_i * m_tiles_per_peer + tile_i
            block_n = bn
        elif const_expr(xcd_swizzle > 0):
            block_m, block_n = _xcd_swizzle_any(
                ceildiv(c_m, BLOCK_M), n_blocks, wgm=xcd_swizzle
            )
        else:
            block_m, block_n = split_row_major_2d(fx.block_idx.x, n_blocks)

        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B_K_STEP = (2 * 1024) if b_preshuffled else BLOCK_K
        B0_gl_offset = (block_n * BLOCK_N) * K
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(
            lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False
        )
        gl_off_b = compute_global_swizzle(
            lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=b_preshuffled
        )

        if const_expr(swap_ab):
            # Tile counts swap with the operands so Mfma's own asserts and its
            # idx() line up; the store then addresses the accumulator as
            # idx(tj, ti).
            mfma_raw = Mfma16x16x128(N_TILES_B, N_TILES_A)
            mfma = _SwappedMfma(mfma_raw)
        else:
            mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        w_pre = cco.Window(win)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        if const_expr(swap_ab and permlane and lane_transpose and not direct_lsa):
            store_c = _LaneTransposeStoreC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )
        elif const_expr(swap_ab and permlane and store_probe and not direct_lsa):
            store_c = _AdjacentLaneProbeC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )
        elif const_expr(swap_ab and permlane and not direct_lsa):
            store_c = _PermlaneStoreC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )
        elif const_expr(swap_ab and store_probe and not direct_lsa):
            store_c = _WideStoreProbeC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )
        elif const_expr(swap_ab and not direct_lsa):
            store_c = _SwapABStoreC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )
        elif const_expr(direct_lsa and swap_ab and permlane and lane_transpose):
            dest_blk = block_m // fx.Int32(m_tiles_per_peer)
            store_c = _LaneTransposeStoreC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B,
                peer_rsrc=create_buffer_resource_from_addr(
                    wave_uniform_i64(w_pre.lsa_ptr(dest_blk, my_recv_slot)),
                    num_records_bytes=cfg.slice_bytes,
                ),
                elem_base=dest_blk * fx.Int32(slice_rows) * c_n,
            )
        elif const_expr(direct_lsa and swap_ab and permlane):
            dest_blk = block_m // fx.Int32(m_tiles_per_peer)
            store_c = _PermlaneStoreC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B,
                peer_rsrc=create_buffer_resource_from_addr(
                    wave_uniform_i64(w_pre.lsa_ptr(dest_blk, my_recv_slot)),
                    num_records_bytes=cfg.slice_bytes,
                ),
                elem_base=dest_blk * fx.Int32(slice_rows) * c_n,
            )
        elif const_expr(direct_lsa and swap_ab and store_probe):
            dest_blk = block_m // fx.Int32(m_tiles_per_peer)
            store_c = _WideStoreProbeC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B,
                peer_rsrc=create_buffer_resource_from_addr(
                    wave_uniform_i64(w_pre.lsa_ptr(dest_blk, my_recv_slot)),
                    num_records_bytes=cfg.slice_bytes,
                ),
                elem_base=dest_blk * fx.Int32(slice_rows) * c_n,
            )
        elif const_expr(direct_lsa):
            # dest is uniform across the block; readfirstlane keeps the peer
            # descriptor scalar and avoids a waterfall around every store.
            dest_blk = block_m // fx.Int32(m_tiles_per_peer)
            store_c = _PeerDirectStoreC(
                A_scale,
                B_scale,
                C,
                c_m,
                c_n,
                mfma.idx,
                N_TILES_A,
                N_TILES_B,
                peer_rsrc=create_buffer_resource_from_addr(
                    wave_uniform_i64(w_pre.lsa_ptr(dest_blk, my_recv_slot)),
                    num_records_bytes=cfg.slice_bytes,
                ),
                elem_base=dest_blk * fx.Int32(slice_rows) * c_n,
            )
        elif const_expr(fence in ("writethrough", "wt-agent")):
            store_c = _WriteThroughStoreC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )
        elif const_expr(fence in ("raw-wt", "raw-wt-agent", "raw-wt-leader")):
            store_c = _RawWriteThroughStoreC(
                A_scale,
                B_scale,
                C,
                c_m,
                c_n,
                mfma.idx,
                N_TILES_A,
                N_TILES_B,
                c_rsrc=create_buffer_resource_from_addr(
                    fx.Int64(cco.Window(win).lsa_ptr(rank, in_off)),
                    num_records_bytes=cfg.nbytes,
                ),
            )
        elif const_expr(fence == "nt-agent"):
            store_c = _NonTemporalStoreC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )
        else:
            store_c = StoreC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        b_g2s.load(b_cur0, B0_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * B_K_STEP)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * B_K_STEP)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        for k in range_constexpr(K_ITERS - 2):
            b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

            b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
            rocdl.s_barrier()

            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)

        a1_frag = a_s2r.load(a_cur1)
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
        rocdl.s_barrier()

        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)

        b0_frag = b_s2r.load(b_next0, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 1
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)

        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, set_prio=False)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, set_prio=False)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        if const_expr(fuse):
            # Re-balance the half-wave barrier counts before the epilogue.
            # The prologue's `if wave_m == 1: s_barrier()` gives waves 4-7 one
            # extra barrier, so every later rendezvous pairs w1's k-th barrier
            # with w0's (k+1)-th and waves 0-3 run one phase ahead. A one-shot
            # GEMM does not care -- the trailing barrier is released when the
            # other half exits. The fused epilogue does: `wait_barrier(0)` after
            # store_c is supposed to mean "every wave's C tile has retired", and
            # under the offset it instead rendezvouses waves 0-3 (which hold
            # thread 0, hence the counter and the put) with waves 4-7 sitting at
            # the *previous* barrier -- before their stores. Thread 0 then counts
            # the tile and can issue the transfer while half the tile is unwritten.
            if wave_m == 0:
                rocdl.s_barrier()

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)
        # ---- end pinned copy ----

        if const_expr(fuse and not direct_lsa):
            # Retire every lane's C stores, then agree block-wide that they are
            # retired. `wait_barrier` is aiter's own `s_waitcnt vmcnt(0);
            # s_barrier` pair, reused so the tail matches the main loop's idiom.
            wait_barrier(0)
            # Then publish them. This fence is not optional: dropping it -- on the
            # theory that the copy engine reads through our own L2 -- still
            # validates on 2 ranks but takes relL2 from 2.35e-3 to 3.76e-3 on 8,
            # i.e. the engine reads some tiles stale. Every lane fences, which is
            # the form cco_device_wrapper.cpp:105-111 documents for producer
            # lanes; leaderOnly orders thread 0 only and is explicitly not a
            # substitute for a release issued by every producer.
            if const_expr(fence == "all"):
                raw_cco.cco_system_fence(fx.Int32(0))
            elif const_expr(fence == "leader"):
                raw_cco.cco_system_fence(fx.Int32(1))
            elif const_expr(fence in ("agent", "wt-agent", "nt-agent", "raw-wt-agent")):
                release_fence("agent")
            # "writethrough": nothing to do -- the stores already went to memory,
            # and wait_barrier(0) above retired them.

            w = cco.Window(win)
            sdma = cco.DevComm(dev_comm).sdma()
            if fx.thread_idx.x == 0:
                # One release per *block* instead of per wave. cco's leaderOnly
                # form was already shown incorrect, but that went through an
                # extern wrapper; emitting the fence inline rules out the
                # compiler having sunk it past the atomic.
                if const_expr(fence in ("agent-leader", "raw-wt-leader")):
                    release_fence("agent")
                dest = block_m // fx.Int32(m_tiles_per_peer)
                chunk = (block_m % fx.Int32(m_tiles_per_peer)) // fx.Int32(
                    m_tiles_per_chunk
                )
                slot = dest * fx.Int32(chunks) + chunk
                ctr = signal_ptr(
                    fx.Int64(w.lsa_ptr(rank, counter_off))
                    + fx.Int64(slot) * fx.Int64(4)
                )
                seq = fx.Int32(
                    atomic_add_u32(ctr, 1, ordering=atomic_order)
                ) + fx.Int32(1)
                # Monotonic: the counters are never reset, so "last tile of this
                # chunk, this epoch" is a modulo test rather than a compare. Same
                # reason the barrier flags in ar/kernels_lsa are never reset -- it
                # is what makes graph replay behave like a fresh launch.
                if seq % fx.Int32(tiles_per_chunk) == fx.Int32(0):
                    if const_expr(emit_put) and dest != fx.Int32(rank):
                        # One queue per destination. Two chunks of the same
                        # destination can be elected at nearly the same moment and
                        # will then post to the same queue; that serialises the
                        # short issue only, and they share one xGMI link either
                        # way, so a queue each would buy nothing.
                        off = fx.Int64(chunk) * fx.Int64(chunk_bytes)
                        lock = signal_ptr(
                            fx.Int64(w.lsa_ptr(rank, lock_off))
                            + fx.Int64(dest) * fx.Int64(4)
                        )
                        if const_expr(chunks > 1):
                            _acquire_peer_lock(lock)
                        sdma.put(
                            dest,
                            win,
                            fx.Int64(my_recv_slot) + off,
                            win,
                            fx.Int64(in_off)
                            + fx.Int64(dest) * fx.Int64(slice_bytes)
                            + off,
                            fx.Int64(chunk_bytes),
                            dest,
                            coop=cco.CoopScope.THREAD,
                            signal=False,
                        )
                        if const_expr(chunks > 1):
                            atomic_store_u32(lock, 0)
            # gcnasm closes its ChunkFused epilogue with a barrier here; its
            # README lists removing it as a rejected experiment that deadlocked.
            fgpu.barrier()

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        dev_comm: Int64,
        win: Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_gemm_scatter(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            dev_comm,
            win,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch


LSA_NOTE = """\
The LSA transport is not implemented, and the reason is a measurement rather than
a preference.

Fusing LSA means the epilogue stores into a peer instead of into local C. But
`StoreC._store_bf16` writes **one bf16 at a time** through `BufferCopy16b`, at
`c_index = (row + i) * c_cols + col` with `row = base_row + ti*16 + lane//16*4 + i`
and `col = base_col + tj*16 + lane%16` -- a lane-scatter. gcnasm measured that
pattern at 0.26x when the destination is a peer, so the variant is worthless
until `StoreC` is coalesced (their winning form was a wave-local `ds_bpermute`
pair-coalesced store; staging C through LDS regressed 1-3%).

That rewrite is the expensive half of this step, and the ceiling it is competing
for is small: at [4096, 7168] the GEMM is ~40us against an all-reduce of ~290us
(LSA) or ~340us (SDMA), so even perfect overlap removes at most the GEMM, ~12%.
Fusing the cheap transport first establishes whether the completion-counter
structure works at all; if the answer is no, the `StoreC` rewrite is wasted
regardless of transport.
"""

__all__ = ["compile_fused_gemm_scatter", "LSA_NOTE", "BLOCK_K"]
