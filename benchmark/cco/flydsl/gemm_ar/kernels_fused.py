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

## The chunking race, unexplained

Overlap requires more than one chunk per destination: with one, the counter only
fires when the whole slice is done, which under the rotated tile order is the end
of the GEMM, so nothing overlaps (measured: scatter 129.0us against split's
136.8 -- a 7.8us gain, i.e. none).

``--chunks 2`` is worth it on paper -- 328us, which would beat split-sdma's 338
and tie split-lsa's 324 -- and it is **intermittently wrong**, in roughly 1 run in
3. Ruled out, each by measurement rather than argument:

* the per-destination submit lock (gcnasm's, ISA-verified: test-and-set,
  ``s_sleep`` backoff, correct spin) -- still 2/6 wrong;
* the post-submit barrier gcnasm keeps -- still 4/6 wrong;
* the release fence, at both agent and system scope -- still 3/6 wrong;
* the counter atomic's ordering, ``acq_rel`` vs relaxed -- helps, does not fix.

The one factor that separates working from broken is *when* the put is issued. At
chunks=1 it lands at the very end of the GEMM; at chunks=2 the first one is
issued mid-kernel and the engine reads C while the rest of the GEMM is still
running. gcnasm's ChunkFused path does the same thing successfully, but its GEMM
is 13x longer, so its chunk boundaries are far apart in time. Whether chunks=1
here is *correct* or merely always-lucky is not established -- the mechanism
suggests the latter, so treat the fused path as unproven either way.

## Result: the fusion does not pay

Correct-as-measured configurations only, 8 ranks, [4096, 7168], K=1024:

    split-lsa                            324.4us
    split-sdma                           338.0
    fused-sdma  chunks=1, no fence       348.5   (stable 6/6)
    fused-sdma  chunks=1, agent fence    440.0   (the fence is 90us of waste)

Per-kernel (rocprofv3 --kernel-trace, us):

    config                     gemm   scatter  reduce  gather    sum
    split-sdma                 39.1     136.8    13.5   137.3   326.8
    fused c=1, no fence        62.7     129.0    13.0   136.2   341.0
    fused c=1, agent fence    152.4     132.7    13.6   135.4   434.1

So with the fence removed the whole remaining deficit is +23.6us inside the GEMM
-- the block-wide barrier, the acq_rel counter, and the copy engine reading C
while the GEMM writes it -- against a 7.8us overlap gain. The structure is sound;
the shape is wrong for it.

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
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int64

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
from flydsl._mlir.dialects import scf  # noqa: E402

from _compat import (  # noqa: E402
    CM_SC0_SC1,
    atomic_add_u32,
    atomic_store_u32,
    atomic_xchg_u32,
    i32_type,
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
    _kname_tag = f"c{chunks}{'r' if rotated else 'l'}{fence[0]}"
    counter_off = cfg.counter_off
    lock_off = cfg.lock_off
    in_off = cfg.input_off
    my_recv_slot = cfg.recv_slot_off(rank) if cfg.recv_slots else 0
    slice_bytes = cfg.slice_bytes

    _kname = (
        f"mori_fused_{'sdma' if fuse else 'split'}_8w_"
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

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        if const_expr(fence in ("writethrough", "wt-agent")):
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

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)
        # ---- end pinned copy ----

        if const_expr(fuse):
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
