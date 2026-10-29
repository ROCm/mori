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
"""The fp8 GEMM with the all-gather absorbed into its epilogue.

The GEMM is ``gemm_ar``'s. This operation differs in *where C goes*, not in how
it is computed, so the pipeline, the MFMA schedule, the C-store ladder and the
block-scale handling are imported from there rather than duplicated.

Two C destinations exist for this op:

===============  ==========================================  ==================
destination      address of element ``(row, col)``            used by
===============  ==========================================  ==================
``local``        ``row*N + col``, a plain tensor              ``gemm-only``
``window``       ``row*N + col`` in my own ``recv`` slot      ``split-sdma``,
                                                              ``fused-sdma``
===============  ==========================================  ==================

The transport is SDMA only. The LSA transports -- a copy kernel storing into
or loading from the peers, and the epilogue storing every tile into every
peer -- were measured and removed; see the README.

## Why this is a smaller file than gemm_a2a's

Because the index map never changes. ``gemm_a2a``'s peer store has to write
``row*shard_n + (col - dest*shard_n)`` while the per-channel B scale is still
indexed by the *global* column, so it needs a ``store()`` override taking two
different columns and a rebuilt B-scale descriptor at the full width -- a trap
that only showed up on ``--quant ptpc``. None of that arises here: the
destination slab *is* ``[M, N]``, so the store column is the global column and
``_PermlaneStoreC`` is usable unmodified, with ``c_cols = N``, ``elem_base = 0``
and its ``oob = c_rows * c_cols`` already landing exactly one past the slab.

What replaces it is the one thing an all-gather has that the other two do not:
the collective is a **broadcast**, so a single tile goes to every peer -- and
the copy engine pushes the rank's own slab unchanged, so the epilogue writes it
once, into this rank's own ``recv`` slot.

## Chunking, and why the counter has one index

``gemm_a2a`` elects on ``(dest, chunk)``: a chunk of destination 0 completing
says nothing about destination 1, because they hold different columns. Here
every destination receives the *same* bytes, so a chunk completing arms all
``world-1`` pushes at once and the counter is indexed by chunk alone.

The tile order needs no rotation for the same reason. ``gemm_a2a`` walks
destinations round-robin so the last one's link does not sit idle until the end
of the GEMM; a tile here belongs to everyone, and the default
``split_row_major_2d(block_idx, n_blocks)`` order -- ``block_m`` outer,
``block_n`` inner -- already finishes chunk 0's rows before starting chunk 1's,
which is exactly what chunked pushing wants.

## Publication

The copy engine reads this rank's own memory, so the epilogue only has to make
a chunk's stores visible before electing its put: ``wait_barrier(0)`` and then
the ``fence`` selected below.
"""

# NOTE: no `from __future__ import annotations` here, deliberately, and
# _gemm_a8w8_8wave.py omits it for the same reason: `fx.struct` reads the
# `SharedStorage` field annotations as live objects, and PEP 563 would hand it
# the string "fx.Array[fx.Float8E4M3FN, a_lds_size, 16]" -- whose size operands
# are locals of this factory and so cannot be resolved from the module globals.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int64

import mori.cco.device.flydsl as cco
from flydsl._mlir.dialects import llvm as _llvm
from mori.cco.device.flydsl import _bindings as raw_cco

from ..gemm_ar._compat import (
    atomic_add_u32,
    atomic_store_u32,
    create_buffer_resource_from_addr,
    signal_ptr,
    wave_uniform_i64,
)
from ..gemm_ar._gemm_a8w8_8wave import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    _xcd_swizzle_any,
    ceildiv,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    read_done_barrier,
    split_row_major_2d,
    wait_barrier,
)

# The C-store ladder and the block-scale mainloop are gemm_ar's. They are
# private, and importing across ops is a cost -- but the alternative is a second
# copy of ~700 lines whose every future fix would have to be applied twice, and
# which would silently diverge the moment one of them was not. The names used
# here are the two the a2a path builds on; if either changes shape, this file
# fails at import rather than at runtime.
from ..gemm_ar.kernels_fused import (
    _acquire_peer_lock,
    _BlockScaleK,
    _Mxfp8ScaleK,
    _PermlaneStoreC,
    _SwappedMfma,
)
from .layout import (  # noqa: F401
    DEFAULT_BLOCK_M,
    DEFAULT_BLOCK_N,
    MXFP8_BLOCK,
    MXFP8_BLOCK_M,
)

BLOCK_K = 128


class _UncachedPeerStoreC(_PermlaneStoreC):
    """``_PermlaneStoreC`` with ``sc0|sc1`` on the peer store.

    gemm_ar has ``_LaneTransposeUncachedPeerStoreC`` for the same bit, but on
    top of the ds_bpermute lane transpose, which this op does not use. The flag
    is a class attribute rather than an argument, so selecting it means naming a
    class.
    """

    _peer_uncached = True


def compile_gemm_local(
    cfg,
    rank: int,
    *,
    K: int,
    BLOCK_M: int = DEFAULT_BLOCK_M,
    BLOCK_N: int = DEFAULT_BLOCK_N,
    b_preshuffled: bool = True,
    quant: str = "ptpc",
    waves_per_eu: int = 2,
    xcd_swizzle: int = 0,
    swap_ab: bool = True,
    permlane: bool = True,
    lane_transpose: bool = False,
    hoist_scales: bool = False,
):
    """The GEMM alone, writing a plain ``[M, N]`` row-major C.

    This is ``gemm_ar.compile_fused_gemm_scatter`` with ``fuse=False``: that
    compiles its whole epilogue tail out (``if const_expr(fuse and not
    direct_lsa)``) and leaves exactly the unmodified 8-wave fp8 GEMM. Calling it
    rather than re-deriving it is the point -- ``gemm-only`` here and
    ``gemm-only`` there have to be the *same* kernel or neither number means
    anything next to the other.

    Returns ``launch(A, B_T, C, A_scale, B_scale, c_m, c_n, dev_comm, win,
    stream=...)``. ``dev_comm`` and ``win`` are unused on this path but stay in
    the signature because they are the shared kernel's.

    The ``ArConfig`` built below is a **placeholder**. With ``fuse=False`` the
    only things read from it are folded into code that is then discarded, so it
    does not have to describe the window that is actually allocated -- and it
    cannot, because an all-gather's window has a different shape.
    """
    from ..gemm_ar import ArConfig
    from ..gemm_ar.kernels_fused import compile_fused_gemm_scatter

    placeholder = ArConfig(world_size=cfg.world_size, m=cfg.m, n=cfg.n)
    return compile_fused_gemm_scatter(
        placeholder,
        rank,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        b_preshuffled=b_preshuffled,
        quant=quant,
        waves_per_eu=waves_per_eu,
        xcd_swizzle=xcd_swizzle,
        fuse=False,
        emit_put=False,
        swap_ab=swap_ab,
        permlane=permlane,
        lane_transpose=lane_transpose,
        hoist_scales=hoist_scales,
    )


def compile_fused_gemm_ag(
    cfg,
    rank: int,
    *,
    K: int,
    BLOCK_M: int = DEFAULT_BLOCK_M,
    BLOCK_N: int = DEFAULT_BLOCK_N,
    b_preshuffled: bool = True,
    quant: str = "ptpc",
    waves_per_eu: int = 2,
    xcd_swizzle: int = 0,
    fuse: bool = True,
    chunks: int = 1,
    sdma_queues: int = 1,
    post: str = "lanes",
    peer_uncached: bool = False,
    emit_put: bool = True,
    fence: str = "leader",
):
    """The GEMM with its C stored into this rank's own window slot.

    Returns ``launch(A, B_T, C, A_scale, B_scale, c_m, c_n, dev_comm, win,
    stream=...)``. ``C`` is not written on either path -- it is still passed
    because the shared ``StoreC`` builds a descriptor from it -- and a cross-rank
    barrier still has to follow the launch before ``recv`` may be read.

    Each tile is stored once, into this rank's own ``recv`` slot, and the
    epilogue posts a copy-engine put per peer as each chunk of rows completes.
    With ``fuse=False`` it stores and does nothing else, which is the GEMM half
    of ``split-sdma`` -- and unlike ``gemm_a2a``'s unfused staging GEMM it is
    *not* a different address map from ``gemm-only``, only a different base
    address, because nothing is re-laid-out for the copy engine here.

    ``emit_put=False`` runs the whole epilogue -- the release fence, the
    completion counter, the submit lock -- and simply does not post. The result
    is wrong by construction and it exists only to price the bookkeeping
    separately from the transfer. ``gemm_a2a`` needed exactly this to find that
    its fused path was 26% *slower* than its own split baseline with the PUT
    itself worth 0.8us, i.e. that it was paying for bookkeeping and not for
    overlap.

    ``post`` selects how the fused-SDMA epilogue issues a chunk's ``world-1``
    packets. ``"lanes"`` gives one to each of lanes ``0..world-2``; ``"serial"``
    has thread 0 issue them back to back, which is what ``gemm_a2a``'s epilogue
    shape becomes when it is transplanted onto a broadcast. The serial form is
    kept only so the difference stays a measurement.

    ``peer_uncached`` sends the epilogue's C stores with ``sc0|sc1``. The store
    is local and the copy engine reads it, so this is a publication choice
    rather than a correctness one: with ``fence="release"`` it is the measured
    ``sc_release`` configuration of the fused path.

    There is no ``rotated``/``n_stripe`` pair. See the module docstring: those
    exist in ``gemm_a2a`` to keep every destination's link busy when a tile
    belongs to exactly one of them, and a broadcast has no such thing.
    """
    cfg.validate()
    if post not in ("lanes", "serial"):
        raise ValueError(f"post must be lanes or serial, got {post!r}")
    if post == "lanes" and chunks > 1 and sdma_queues < chunks:
        raise ValueError(
            f"post='lanes' with chunks={chunks} needs sdma_queues >= {chunks}, "
            f"got {sdma_queues}: the lane-parallel path has no submit lock "
            f"(world-1 lanes retire in lockstep, so a lock set they hold "
            f"between them cannot be released), so two chunks of one "
            f"destination have to be separated by queue instead"
        )
    if fence not in ("none", "leader", "all", "release"):
        raise ValueError(f"invalid producer fence: {fence!r}")
    if quant not in ("ptpc", "blockscale", "mxfp8"):
        raise ValueError(f"quant must be ptpc, blockscale or mxfp8, got {quant!r}")
    if (BLOCK_M, BLOCK_N) != (cfg.block_m, cfg.block_n):
        raise ValueError(
            f"tile ({BLOCK_M}, {BLOCK_N}) disagrees with the config's "
            f"({cfg.block_m}, {cfg.block_n}); the destination map is derived from "
            f"the config's, so the two must be the same"
        )
    assert K % BLOCK_K == 0
    blockscale = quant == "blockscale"
    mxfp8 = quant == "mxfp8"
    if mxfp8:
        # Not a preference: the packed A scale puts a lane's four M tiles in one
        # dword and picks the byte with the MFMA's opsel, which is four tiles
        # only at BLOCK_M//64 == 4. gemm_ar states the same constant.
        if BLOCK_M != MXFP8_BLOCK_M:
            raise ValueError(
                f"quant='mxfp8' requires BLOCK_M={MXFP8_BLOCK_M}, got {BLOCK_M}"
            )
        if K % 128:
            raise ValueError(f"mxfp8 needs K % 128 == 0, got K={K}")

    ws = cfg.world_size
    N = cfg.n
    if mxfp8 and N % MXFP8_BLOCK:
        # N by the 32-column scale group. The layout already forces N to a
        # multiple of block_n = 256, so this can only fire on a config that
        # bypassed ag_config -- but it is the operand format's own rule and
        # belongs next to the others.
        raise ValueError(f"mxfp8 needs N % {MXFP8_BLOCK} == 0, got N={N}")
    n_blocks_total = cfg.n_blocks
    my_recv_slot = cfg.recv_slot_off(rank)
    slab_bytes = cfg.slab_bytes
    counter_off, lock_off = cfg.counter_off, cfg.lock_off
    _StoreC = _UncachedPeerStoreC if peer_uncached else _PermlaneStoreC
    if cfg.counter_chunks != chunks:
        raise ValueError(
            f"chunks={chunks} disagrees with the config's counter_chunks="
            f"{cfg.counter_chunks}; the counter region is sized from the "
            f"config's, so an epilogue electing on a different count would index "
            f"past it"
        )
    # A chunk is a run of row tiles, so it stays contiguous inside [M, N] -- and
    # inside every peer's copy of it, since they are all the same slab. Every N
    # tile of those rows counts toward it, where gemm_a2a counts only the ones
    # in the destination's column shard.
    m_tiles_per_chunk = cfg.m_tiles // chunks
    tiles_per_chunk = m_tiles_per_chunk * n_blocks_total
    chunk_bytes = m_tiles_per_chunk * BLOCK_M * N * 2

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

    _kname = (
        f"mori_ag_{'sdma' if fuse else 'win'}_8w_"
        f"{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_k{K}_"
        f"{'B' if blockscale else ('X' if mxfp8 else 'P')}"
        f"x{xcd_swizzle}c{chunks}q{sdma_queues}{post[0]}"
        f"{'U' if peer_uncached else 'C'}"
        f"{fence[0]}{'p' if emit_put else 'x'}_r{rank}"
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
    def kernel_gemm_ag(
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
        # ---- begin pinned copy of _gemm_a8w8_8wave kernel_gemm ----
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

        if const_expr(xcd_swizzle > 0):
            block_m, block_n = _xcd_swizzle_any(
                ceildiv(c_m, BLOCK_M), n_blocks, wgm=xcd_swizzle
            )
        else:
            # Row-major over (m_tiles, n_blocks): block_m outer, block_n inner.
            # That is already chunk-major -- every N tile of chunk 0's rows is
            # dispatched before any of chunk 1's -- so chunked pushing gets what
            # it needs from the default order and there is nothing to rotate.
            #
            # gemm_a2a and gemm_ar both need a rotation here because a tile
            # belongs to one destination and linear order would leave the last
            # one's link idle until the end of the GEMM. In a broadcast a tile
            # belongs to all of them, so every link starts with chunk 0.
            #
            # xcd_swizzle is left reachable for the unfused path, where the tile
            # order is a pure GEMM-locality question and the collective is not
            # watching. With fuse=True it breaks the chunk-major property, which
            # is why the benchmark leaves it at 0 there.
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

        # opsel_b_per_tile goes with the packed A scale: _Mxfp8ScaleK(packed_a)
        # puts four B-tile ue8m0 bytes in one dword, and the byte the
        # instruction reads is an atom-time attribute, not a shift. Leaving it
        # off makes all four tiles read byte 0 -- which still validates the GEMM
        # alone at relL2 1.66e-3 because gemm-only takes gemm_ar's own kernel,
        # and gives 0.58 here.
        mfma = _SwappedMfma(Mfma16x16x128(N_TILES_B, N_TILES_A, opsel_b_per_tile=mxfp8))
        w_pre = cco.CachedWindow(win)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)

        # Where C goes. The destination slab is the full [M, N], so the store
        # index is `row*N + col` unmodified and StoreC's own
        # `oob = c_rows * c_cols` lands exactly one past it. gemm_a2a needs a
        # store subclass here precisely because its slab is [M, shard_n] and
        # the index has to be rebased while the B scale is not; nothing is
        # rebased here, so `_PermlaneStoreC` is used as it comes.
        #
        # It goes to this rank's own recv slot, and the copy engine pushes that
        # slab unchanged. (`peer_rsrc` is a misnomer here: it is a local
        # resource, reached through `lsa_ptr(rank, ...)`. Renaming it in gemm_ar
        # to suit this op would be the tail wagging the dog.)
        own_slot = create_buffer_resource_from_addr(
            wave_uniform_i64(w_pre.lsa_ptr(rank, my_recv_slot)),
            num_records_bytes=slab_bytes,
        )
        store_c = _StoreC(
            A_scale,
            B_scale,
            C,
            c_m,
            c_n,
            mfma.idx,
            N_TILES_A,
            N_TILES_B,
            peer_rsrc=own_slot,
            elem_base=fx.Int32(0),
        )

        # Bound unconditionally: the kernel body is re-parsed by FlyDSL's AST
        # rewriter, and names that only exist inside a branch are not reliably
        # visible to a nested def afterwards.
        bsk = nb0 = base_row_pre = None
        msk = base_col_pre = None
        if mxfp8:
            # The scales are MFMA *operands*, not epilogue arithmetic: ue8m0 is
            # exponent-only so the instruction dequantises losslessly and there
            # is no second accumulator and no running rescale. The epilogue must
            # not apply them again.
            store_c._preapplied = True
            msk = _Mxfp8ScaleK(
                A_scale,
                B_scale,
                c_m,
                N,
                K,
                n_tiles_a=N_TILES_A,
                n_tiles_b=N_TILES_B,
                row_major=False,
                packed_a=True,
            )
            base_row_pre = block_m * BLOCK_M + wave_m * (N_TILES_A * 16)
            base_col_pre = block_n * BLOCK_N + wave_n * (N_TILES_B * 16)
        if blockscale:
            store_c._preapplied = True
            bsk = _BlockScaleK(
                A_scale, B_scale, c_m, N, K, swap_ab=True, n_tiles_a=N_TILES_A
            )
            nb0 = block_n * fx.Int32(BLOCK_N // 128)
            base_row_pre = block_m * BLOCK_M + wave_m * (N_TILES_A * 16)
        prev_scales = None

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
            if blockscale:
                (c00_frag, c01_frag, c10_frag, c11_frag), prev_scales = bsk.rescale_for(
                    k,
                    (c00_frag, c01_frag, c10_frag, c11_frag),
                    prev_scales,
                    base_row=base_row_pre,
                    nb0=nb0,
                    idx_fn=mfma.idx,
                    n_tiles_b=N_TILES_B,
                    lds_block_m=LDS_BLOCK_M,
                )
            msa0 = msa1 = msb0 = msb1 = None
            if mxfp8:
                msa0, msa1, msb0, msb1 = msk.step(
                    base_row_pre, base_col_pre, k, LDS_BLOCK_M, LDS_BLOCK_N
                )
            b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            # The leading wave group refills the B0 slot just read one stage
            # after the barrier the lagging group's S0 pairs with; make the
            # reads complete before advertising it (gemm_ar, #785).
            read_done_barrier()

            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, scale_a=msa0, scale_b=msb0)

            b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
            rocdl.s_barrier()

            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, scale_a=msa0, scale_b=msb1)

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, scale_a=msa1, scale_b=msb0)

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
            # The original count, as gemm_ar since #785. A+B-1 used to be the
            # fix for large-grid corruption, but it only made the peeled tail's
            # tiles land sooner; the tail now drains them itself (below).
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, scale_a=msa1, scale_b=msb1)

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        if blockscale:
            (c00_frag, c01_frag, c10_frag, c11_frag), prev_scales = bsk.rescale_for(
                K_ITERS - 2,
                (c00_frag, c01_frag, c10_frag, c11_frag),
                prev_scales,
                base_row=base_row_pre,
                nb0=nb0,
                idx_fn=mfma.idx,
                n_tiles_b=N_TILES_B,
                lds_block_m=LDS_BLOCK_M,
            )
        msa0 = msa1 = msb0 = msb1 = None
        if mxfp8:
            msa0, msa1, msb0, msb1 = msk.step(
                base_row_pre, base_col_pre, K_ITERS - 2, LDS_BLOCK_M, LDS_BLOCK_N
            )
        b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)
        a0_frag = a_s2r.load(a_cur0)
        # The peeled steps pre-read step K_ITERS - 1's B0 (end of this step)
        # and A0 (top of the next), both before the last step's
        # wait_barrier(0). Drain their producers here (gemm_ar, #785). At
        # K=256 the main loop runs zero times, so no count inside it can cover
        # this. The last step's wait stays: it still covers the final A1.
        wait_barrier(0)

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, scale_a=msa0, scale_b=msb0)

        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, scale_a=msa0, scale_b=msb1)

        a1_frag = a_s2r.load(a_cur1)
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
        rocdl.s_barrier()

        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, scale_a=msa1, scale_b=msb0)

        b0_frag = b_s2r.load(b_next0, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, scale_a=msa1, scale_b=msb1)

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 1
        if blockscale:
            (c00_frag, c01_frag, c10_frag, c11_frag), prev_scales = bsk.rescale_for(
                K_ITERS - 1,
                (c00_frag, c01_frag, c10_frag, c11_frag),
                prev_scales,
                base_row=base_row_pre,
                nb0=nb0,
                idx_fn=mfma.idx,
                n_tiles_b=N_TILES_B,
                lds_block_m=LDS_BLOCK_M,
            )
        msa0 = msa1 = msb0 = msb1 = None
        if mxfp8:
            msa0, msa1, msb0, msb1 = msk.step(
                base_row_pre, base_col_pre, K_ITERS - 1, LDS_BLOCK_M, LDS_BLOCK_N
            )
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, scale_a=msa0, scale_b=msb0)

        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, scale_a=msa0, scale_b=msb1)

        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(
            a1_frag, b0_frag, c10_frag, set_prio=False, scale_a=msa1, scale_b=msb0
        )
        c11_frag = mfma.call(
            a1_frag, b1_frag, c11_frag, set_prio=False, scale_a=msa1, scale_b=msb1
        )
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        if blockscale:
            c00_frag, c01_frag, c10_frag, c11_frag = bsk.final_scale(
                (c00_frag, c01_frag, c10_frag, c11_frag),
                prev_scales,
                idx_fn=mfma.idx,
                n_tiles_b=N_TILES_B,
            )

        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset
        # Close the half-wave barrier the prologue opened. Its
        # `if wave_m == 1: s_barrier()` gives waves 4-7 one extra barrier, and
        # s_barrier is a counting rendezvous, so waves 0-3 would otherwise run a
        # phase ahead of the stores that follow. gemm_ar's copy of this tail
        # carries the measurement: leaving it out made --chunks 2 fail one run
        # in three.
        if wave_m == 0:
            rocdl.s_barrier()

        # Three arguments, not gemm_a2a's four: there the store row and the
        # A-scale row differ (the destination is folded into one of them and
        # not the other), and here they are the same global row.
        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)
        # ---- end pinned copy ----

        # Retire this block's stores into the window slot, agree block-wide
        # that they are retired, then publish them -- but only when this
        # kernel is the thing that publishes.
        #
        # The fence is a *whole-L2* writeback issued by every lane of every
        # block, so block k redundantly writes back every tile blocks
        # 1..k-1 already wrote. At 1152 blocks that is quadratic and it is
        # the single most expensive line in this epilogue: gating it on
        # `fuse` is what took the unfused staging GEMM from 458 to ~320us,
        # level with the plain one.
        #
        # With `fuse=False` nothing is published here at all -- the separate
        # scatter kernel issues the puts and does its own ordering -- so the
        # fence has no one to release to. With `fuse=True` it is not
        # optional: gemm_ar measured dropping it as still validating on 2
        # ranks while taking relL2 from 2.35e-3 to 3.76e-3 on 8, i.e. the
        # copy engine reads some tiles stale.
        wait_barrier(0)
        if const_expr(fuse and fence == "all"):
            raw_cco.cco_system_fence(fx.Int32(0))
        elif const_expr(fuse and fence == "leader"):
            # One wave per block, not every lane. Every lane costs a whole-L2
            # writeback each, and at M=16384 there are 9216 blocks: the
            # epilogue's bookkeeping measured 2358us against a 2798us GEMM,
            # with the PUT itself worth 0.8us of that. Same mistake as the
            # unfused path's unconditional fence; this is the second place it
            # was written.
            #
            # Legal because the half-wave barrier pair is closed above, so
            # wait_barrier(0) really does mean every wave's stores have
            # retired into this CU's L2 and one wave writing it back covers
            # all eight.
            raw_cco.cco_system_fence(fx.Int32(1))
        elif const_expr(fuse and fence == "release"):
            if fx.thread_idx.x == fx.Int32(0):
                _llvm.fence(_llvm.AtomicOrdering.release, syncscope="")
        # "none": gemm_ar's own default here. Its stores go through the
        # same L2 the copy engine reads, and it measured the fence as
        # unnecessary on that path -- but it also measured dropping it as
        # taking relL2 from 2.35e-3 to 3.76e-3 on 8 ranks, so it stays
        # selectable rather than assumed.

        # Both bases are rank-local and constant-offset, so they are the same
        # for every thread. Taking them here rather than inside the `if` also
        # keeps the window out of the branch's captured state, which has to
        # be single MLIR values -- see CachedWindow.
        ctr_base = fx.Int64(w_pre.lsa_ptr(rank, counter_off))
        lock_base = fx.Int64(w_pre.lsa_ptr(rank, lock_off))
        sdma = cco.DevComm(dev_comm).sdma()
        # Block-uniform, so every lane can compute it.
        chunk = block_m // fx.Int32(m_tiles_per_chunk)
        off = fx.Int64(chunk) * fx.Int64(chunk_bytes)
        # Lane j owns destination (rank+1+j) and counter slot (chunk, j).
        # `post="serial"` uses one lane and one slot.
        #
        # This is the shape of gemm_a2a's counter, and adopting it is what
        # makes the posting concurrent. The obvious layout for a broadcast
        # is *one* counter per chunk -- a chunk completing arms every
        # destination at once, so why count world-1 times? Because the
        # thread that wins a single counter then has to tell world-2 other
        # lanes that it won, and a broadcast through LDS costs a barrier
        # and a copy atom for one dword. With a counter per (chunk, lane),
        # every block increments all world-1 of them in one instruction,
        # they cross the threshold together, and each lane learns from
        # *its own* atomic that its block won. No broadcast, no barrier.
        #
        # Cost of the change: world-1 atomics per block on world-1 distinct
        # addresses, which is one instruction's worth of lanes rather than
        # world-1 instructions, and world*chunks counter dwords in the
        # window instead of chunks.
        lanes = ws - 1 if post == "lanes" else 1
        if fx.thread_idx.x < fx.Int32(lanes) and const_expr(fuse):
            j = fx.thread_idx.x
            slot = chunk * fx.Int32(ws) + j
            ctr = signal_ptr(ctr_base + fx.Int64(slot) * fx.Int64(4))
            # acq_rel, matching gemm_ar's default rather than
            # _compat's "monotonic": the election has to order
            # against the stores the fence above published.
            seq = fx.Int32(atomic_add_u32(ctr, 1, ordering="acq_rel")) + fx.Int32(1)
            # Monotonic: the counters are never reset, so "last tile of this
            # chunk, this epoch" is a modulo test rather than a compare --
            # the same property that makes graph replay behave like a fresh
            # launch.
            if seq % fx.Int32(tiles_per_chunk) == fx.Int32(0):
                if const_expr(fuse and emit_put):
                    if const_expr(post == "lanes"):
                        # world-1 packets issued at once instead of one
                        # thread issuing them back to back, which measured
                        # 39-55us at 8 ranks and --chunks 4: 28 packets at
                        # SDMA's ~2us each, a constant gemm_ar's layout
                        # states and which this reproduced independently by
                        # being flat across payloads differing 8x.
                        #
                        # Self is excluded by construction: lane j maps to
                        # rank+1+j, which never lands on rank. The +rank
                        # rotation keeps the ranks from all posting to the
                        # same peer first.
                        d = (fx.Int32(rank + 1) + j) % fx.Int32(ws)
                        sdma.put(
                            d,
                            win,
                            fx.Int64(my_recv_slot) + off,
                            win,
                            # Source and destination are the same offset:
                            # recv is indexed by source, and a broadcast has
                            # one source. gemm_a2a reads out of staging
                            # here, at an offset that depends on `dest`.
                            fx.Int64(my_recv_slot) + off,
                            fx.Int64(chunk_bytes),
                            # Keyed by **chunk**, not by destination. Queues
                            # are per (source, destination) pair, so every
                            # peer already has its own and this index only
                            # disambiguates among queues *to one peer* --
                            # which is exactly the collision that remains:
                            # two chunks of the same destination elected at
                            # nearly the same moment. Separating them by
                            # queue is what lets this path drop the submit
                            # lock, and dropping it is not optional here:
                            # world-1 lanes retire in lockstep, so two waves
                            # each holding part of a lock set and spinning
                            # for the rest could never reach their release.
                            # Hence the sdma_queues >= chunks check above.
                            chunk % fx.Int32(sdma_queues),
                            coop=cco.CoopScope.THREAD,
                            signal=False,
                        )
                    else:
                        # The original: one thread issues all world-1
                        # packets back to back, under one submit lock. Kept
                        # selectable so the change above stays a measured
                        # difference rather than an asserted one. Safe with
                        # a single lock because a single thread takes it.
                        lock = signal_ptr(lock_base)
                        if const_expr(chunks > 1):
                            _acquire_peer_lock(lock)
                        for jj in range_constexpr(ws - 1):
                            dd = (rank + 1 + jj) % ws
                            sdma.put(
                                fx.Int32(dd),
                                win,
                                fx.Int64(my_recv_slot) + off,
                                win,
                                fx.Int64(my_recv_slot) + off,
                                fx.Int64(chunk_bytes),
                                fx.Int32(dd % sdma_queues),
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
        kernel_gemm_ag(
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


__all__ = ["compile_fused_gemm_ag", "compile_gemm_local"]
