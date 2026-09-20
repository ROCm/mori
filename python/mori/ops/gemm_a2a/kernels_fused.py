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
"""The fp8 GEMM with the all-to-all absorbed into its epilogue.

The GEMM is ``gemm_ar``'s. This operation differs in *where C goes*, not in how
it is computed, so the pipeline, the MFMA schedule, the C-store ladder and the
block-scale handling are imported from there rather than duplicated.

Three C destinations exist for this op:

===============  ===================================  =========================
destination      address of element ``(row, col)``     used by
===============  ===================================  =========================
``local``        ``row*N + col``                       ``gemm-only``, ``split-*``
``peer``         ``row*shard_n + (col - dest*shard_n)``  ``fused-lsa``
                 in peer ``dest``'s window
``staging``      same index, in a local slab            ``fused-sdma``
===============  ===================================  =========================

where ``dest = col // shard_n``, uniform across a GEMM tile because ``layout``
requires ``N % (world*BLOCK_N) == 0``.

## Why this is a small file

``_PermlaneStoreC._emit`` -- the default store variant -- already computes its
own flat index and already takes the scales as separate arguments::

    idx = select(col + 7 < self.c_cols, row * self.c_cols + col, oob)

So constructing the store with ``c_cols = shard_n`` and handing ``_emit`` a
**destination-local** base column makes that one line produce the a2a address,
the bounds test correct, and ``oob`` land exactly one past the slab. The B scale
still wants the global column, and ``store()`` passes the scales separately, so
the whole address-map difference is a five-line override.

There is one trap in that, and it is why ``_A2aPeerStoreC`` is not three lines:
``StoreC.__init__`` sizes the **B-scale** buffer descriptor from the same
``c_cols``. Shrinking ``c_cols`` to ``shard_n`` would silently clamp every
per-channel scale load past column ``shard_n`` to zero. The descriptor is
rebuilt below at the global width. Block-scale is unaffected -- ``_BlockScaleK``
takes ``N`` explicitly and the epilogue's own loads are switched off -- so this
only ever showed up on ``--quant ptpc``.

## Publication

The peer stores are made by every block, so every block must publish them. The
split path gets this for free from the copy kernel's own fence; here the
producer is the GEMM and the barrier kernel that follows is one block, hence one
XCD's L2 out of eight. ``gemm_ar``'s Direct LSA path found that out the hard way
and the tail below is the same remedy: ``wait_barrier(0)`` then a system fence
from every lane.
"""

# NOTE: no `from __future__ import annotations` here, deliberately, and
# _gemm_a8w8_8wave.py omits it for the same reason: `fx.struct` reads the
# `SharedStorage` field annotations as live objects, and PEP 563 would hand it
# the string "fx.Array[fx.Float8E4M3FN, a_lds_size, 16]" -- whose size operands
# are locals of this factory and so cannot be resolved from the module globals.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int64

import mori.cco.device.flydsl as cco
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
    _PermlaneStoreC,
    _SwappedMfma,
)
from .layout import DEFAULT_BLOCK_M, DEFAULT_BLOCK_N  # noqa: F401

BLOCK_K = 128


class _A2aPeerStoreC(_PermlaneStoreC):
    """``_PermlaneStoreC`` writing into the destination's compact slab.

    The store index has to be ``row*shard_n + local_col`` while the B scale is
    still indexed by the global column. ``_emit`` and the scale loads are already
    separate arguments of ``store``, so both are satisfied by passing different
    columns to each.

    ``elem_base`` is 0, unlike ``gemm_ar``'s peer store: there the peer
    descriptor covers the whole ``[M, N]`` and the base rebases onto the
    destination's row slice, whereas here the descriptor *is* the slab.
    """

    def __init__(self, *args, dest_col_base=None, b_scale=None, n=None, **kwargs):
        super().__init__(*args, elem_base=fx.Int32(0), **kwargs)
        self._dest_col_base = dest_col_base
        # See the module docstring: the inherited descriptor was sized from
        # c_cols, which is shard_n here, and B's per-channel scale is indexed by
        # the global column. Rebuild it at the full width. Only ptpc reads it.
        if b_scale is not None:
            self.sb_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(
                    b_scale, max_size=False, num_records_bytes=n * 4
                ),
                fx.make_layout(1, 1),
            )

    def store(self, c_frag, base_row, base_col):
        self._emit(
            c_frag,
            base_row,
            base_col - self._dest_col_base,
            self._a_scales(base_row),
            self._b_scales(base_col),
        )


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
    cannot, because an all-to-all's window has a different shape.
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


def compile_fused_gemm_a2a(
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
    rotated: bool = True,
    n_stripe: int = 1,
    transport: str = "lsa",
    fuse: bool = True,
    chunks: int = 1,
    sdma_queues: int = 1,
):
    """The GEMM with its C stored straight into the destinations' windows.

    Returns ``launch(A, B_T, C, A_scale, B_scale, c_m, c_n, dev_comm, win,
    stream=...)``. ``C`` is not written on this path -- it is still passed
    because the shared ``StoreC`` builds a descriptor from it -- and a cross-rank
    barrier still has to follow the launch before ``recv`` may be read.

    ``rotated`` walks the destinations round-robin, offset by ``rank``, instead
    of finishing destination 0's columns before starting 1's. Linear order makes
    the last destination's link idle until the end of the GEMM; gcnasm measured
    the rotation worth 17-27% at 8 ranks for M>=8192 and reported it *weaker* at
    4 ranks with a partial-N shard, so it is a knob to sweep rather than a
    setting to assume transfers.
    """
    cfg.validate()
    if transport not in ("lsa", "sdma"):
        raise ValueError(f"transport must be lsa or sdma, got {transport!r}")
    if transport == "sdma" and not cfg.staged:
        raise ValueError(
            "transport='sdma' needs a config built with staged=True: the copy "
            "engine reads one contiguous range, so C has to land in the "
            "[dst][M][shard_n] slab rather than in the peer"
        )
    if transport == "lsa" and not fuse:
        raise ValueError(
            "fuse=False is only meaningful with transport='sdma', where it gives "
            "the staging GEMM for the split path; an unfused LSA GEMM is just "
            "compile_gemm_local"
        )
    if quant not in ("ptpc", "blockscale"):
        raise ValueError(f"quant must be ptpc or blockscale, got {quant!r}")
    if (BLOCK_M, BLOCK_N) != (cfg.block_m, cfg.block_n):
        raise ValueError(
            f"tile ({BLOCK_M}, {BLOCK_N}) disagrees with the config's "
            f"({cfg.block_m}, {cfg.block_n}); the destination map is derived from "
            f"the config's, so the two must be the same"
        )
    assert K % BLOCK_K == 0
    blockscale = quant == "blockscale"

    ws = cfg.world_size
    N = cfg.n
    shard_n = cfg.shard_n
    n_blocks_per_peer = cfg.n_blocks_per_peer
    my_recv_slot = cfg.recv_slot_off(rank)
    slab_bytes = cfg.slab_bytes
    cap_slab_bytes = cfg.cap_slab_bytes
    staging_off = cfg.staging_off if cfg.staged else 0
    counter_off, lock_off = cfg.counter_off, cfg.lock_off
    direct_lsa = transport == "lsa"
    if not direct_lsa and cfg.counter_chunks != chunks:
        raise ValueError(
            f"chunks={chunks} disagrees with the config's counter_chunks="
            f"{cfg.counter_chunks}; the counter region is sized from the "
            f"config's, so an epilogue electing on a different count would index "
            f"past it"
        )
    # A chunk is a run of row tiles, so it stays contiguous inside the slab.
    m_tiles_per_chunk = cfg.m_tiles // chunks
    tiles_per_chunk = m_tiles_per_chunk * n_blocks_per_peer
    chunk_bytes = m_tiles_per_chunk * BLOCK_M * shard_n * 2
    if rotated and n_blocks_per_peer % n_stripe:
        raise ValueError(
            f"n_stripe={n_stripe} must divide the {n_blocks_per_peer} N tiles in a "
            f"destination's shard, or the rotation would not cover them evenly"
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

    _kname = (
        f"mori_a2a_{transport if fuse else 'stage'}_8w_"
        f"{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_k{K}_"
        f"{'B' if blockscale else 'P'}{'r' if rotated else 'l'}s{n_stripe}"
        f"c{chunks}q{sdma_queues}_r{rank}"
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
    def kernel_gemm_a2a(
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

        if const_expr(rotated):
            # Destination-rotated tile order. Linear order finishes destination
            # 0's whole column block, then 1's, so the last destination's link
            # only starts at the end of the GEMM and nothing overlaps. Walking
            # (m, dest, n-within-shard) gets every link streaming within the
            # first 1/world of the GEMM, and the `+ rank` rotation keeps all
            # ranks from pushing at the same destination at once.
            #
            # This is the column-sharded twin of gemm_ar's, which rotates over
            # *row* bands; the axes swap because the sharded dimension does.
            if const_expr(n_stripe >= n_blocks_per_peer):
                rest, n_in_shard = split_row_major_2d(fx.block_idx.x, n_blocks_per_peer)
                block_m, dest_seq = split_row_major_2d(rest, ws)
            else:
                rest, n_in = split_row_major_2d(fx.block_idx.x, n_stripe)
                rest, dest_seq = split_row_major_2d(rest, ws)
                block_m, n_grp = split_row_major_2d(rest, n_blocks_per_peer // n_stripe)
                n_in_shard = n_grp * fx.Int32(n_stripe) + n_in
            dest_i = (dest_seq + fx.Int32(rank)) % fx.Int32(ws)
            block_n = dest_i * fx.Int32(n_blocks_per_peer) + n_in_shard
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

        mfma = _SwappedMfma(Mfma16x16x128(N_TILES_B, N_TILES_A))
        w_pre = cco.CachedWindow(win)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)

        # Which destination this block's columns belong to, and its slab. The
        # descriptor is the slab itself, so the store index needs no rebasing
        # and StoreC's own `oob = c_rows * c_cols` lands exactly one past it.
        #
        # LSA writes the destination's slab in the *destination's* window; SDMA
        # writes the same slab shape in this rank's own staging region, which the
        # copy engine then pushes. Only the base address differs -- the index
        # math, the bounds test and the oob sentinel are identical, which is why
        # one store class serves both. (`peer_rsrc` is then a misnomer on the
        # SDMA path: it is a local resource. Renaming it in gemm_ar to suit this
        # op would be the tail wagging the dog.)
        dest_blk = block_n // fx.Int32(n_blocks_per_peer)
        if const_expr(direct_lsa):
            store_base = wave_uniform_i64(w_pre.lsa_ptr(dest_blk, my_recv_slot))
        else:
            # `dest == rank` goes straight into my own recv slot rather than
            # into staging. Nothing ever pushes that slab -- the put loop skips
            # self, because a copy engine round trip to our own memory would be
            # pure cost -- so staging it would strand it. Leaving it out is not
            # a visible failure either: every *remote* slab still arrives
            # correct, so the transport looks fine and only one eighth of the
            # answer is missing. That is what it did.
            win_base = fx.Int64(w_pre.lsa_ptr(rank, 0))
            staged_at = fx.Int64(staging_off) + fx.Int64(dest_blk) * fx.Int64(
                cap_slab_bytes
            )
            store_base = wave_uniform_i64(
                win_base
                + arith.select(
                    dest_blk == fx.Int32(rank),
                    fx.Int64(my_recv_slot),
                    staged_at,
                )
            )
        store_c = _A2aPeerStoreC(
            A_scale,
            B_scale,
            C,
            c_m,
            fx.Int32(shard_n),
            mfma.idx,
            N_TILES_A,
            N_TILES_B,
            peer_rsrc=create_buffer_resource_from_addr(
                store_base, num_records_bytes=slab_bytes
            ),
            dest_col_base=dest_blk * fx.Int32(shard_n),
            b_scale=B_scale,
            n=N,
        )

        # Bound unconditionally: the kernel body is re-parsed by FlyDSL's AST
        # rewriter, and names that only exist inside a branch are not reliably
        # visible to a nested def afterwards.
        bsk = nb0 = base_row_pre = None
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
            # N_LDS_STEPS_A + N_LDS_STEPS_B - 1, not aiter's
            # 2*N_LDS_STEPS_A + N_LDS_STEPS_B. The upstream count lets too many
            # global->LDS prefetches stay in flight across the barrier and
            # corrupts the output non-deterministically on large grids. See the
            # long comment at the same line in gemm_ar/kernels_fused.py for the
            # measured boundary; do not "restore" it.
            wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B - 1)

            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

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

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)
        # ---- end pinned copy ----

        if const_expr(direct_lsa):
            # Publish the peer stores from the blocks that made them. The barrier
            # kernel that follows is one block, hence one XCD's L2 out of eight;
            # the other seven would keep their peer-homed lines dirty and the
            # barrier's system-scope atomic would overtake them.
            wait_barrier(0)
            raw_cco.cco_system_fence(fx.Int32(0))
        else:
            # Retire this block's stores into the staging slab, agree block-wide
            # that they are retired, then publish them to the copy engine. The
            # fence is not optional even though the engine reads memory this CU
            # wrote: gemm_ar measured dropping it as still validating on 2 ranks
            # while taking relL2 from 2.35e-3 to 3.76e-3 on 8, i.e. the engine
            # reads some tiles stale.
            wait_barrier(0)
            raw_cco.cco_system_fence(fx.Int32(0))

            # Both bases are rank-local and constant-offset, so they are the same
            # for every thread. Taking them here rather than inside the `if` also
            # keeps the window out of the branch's captured state, which has to
            # be single MLIR values -- see CachedWindow.
            ctr_base = fx.Int64(w_pre.lsa_ptr(rank, counter_off))
            lock_base = fx.Int64(w_pre.lsa_ptr(rank, lock_off))
            sdma = cco.DevComm(dev_comm).sdma()
            if fx.thread_idx.x == 0:
                # dest is the *column* block's owner and chunk is a run of row
                # tiles, which is the transpose of gemm_ar's (dest from rows,
                # chunk from... rows as well). Chunking along M is what keeps a
                # chunk contiguous inside [dst][M][shard_n].
                dest = block_n // fx.Int32(n_blocks_per_peer)
                chunk = block_m // fx.Int32(m_tiles_per_chunk)
                slot = dest * fx.Int32(chunks) + chunk
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
                    if const_expr(fuse) and dest != fx.Int32(rank):
                        off = fx.Int64(chunk) * fx.Int64(chunk_bytes)
                        lock = signal_ptr(lock_base + fx.Int64(dest) * fx.Int64(4))
                        if const_expr(chunks > 1):
                            # Two chunks of one destination can be elected at
                            # nearly the same moment and would then post to the
                            # same queue concurrently.
                            _acquire_peer_lock(lock)
                        sdma.put(
                            dest,
                            win,
                            fx.Int64(my_recv_slot) + off,
                            win,
                            fx.Int64(staging_off)
                            + fx.Int64(dest) * fx.Int64(cap_slab_bytes)
                            + off,
                            fx.Int64(chunk_bytes),
                            dest % fx.Int32(sdma_queues),
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
        kernel_gemm_a2a(
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


__all__ = ["compile_fused_gemm_a2a", "compile_gemm_local"]
