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
"""The 8-wave fp8 GEMM with the all-gather that feeds it absorbed into it.

Three launchers, all over the window ``layout.AgGemmConfig`` describes:

``compile_ag_gemm(fuse=True)``
    One kernel. Block 0 doubles as the *poster*: lanes ``0..world-1`` each own
    one peer and push this rank's rows to it band by band -- put, ``quiet``,
    then a system-scope store of the band's arrival flag into the peer. Every
    block then computes one tile. Tiles over this rank's own rows come first and
    never wait; the rest are ordered (chunk, source, row tile, column tile), with
    the source rotated by rank, so the first wave of remote tiles needs only each
    peer's band 0. A remote tile's thread 0 spins on that band's flag before the
    prologue issues a single A load.

``compile_ag_gemm(fuse=False)``
    The same GEMM, linear tile order, no waits: the split path's compute, run
    after ``push`` and ``barrier(0)`` have gathered everything.

``build_phases``
    ``push`` (the poster as its own one-block kernel) and the two cross-rank
    barriers. ``barrier(1, bump=True)`` closes every call: nobody may refill a
    peer's slot while that peer still reads it, which the all-gather has no
    gather-side kernel to guarantee otherwise.

Why the poster lives in the GEMM rather than in a kernel on a second stream:
remote tiles spin, and a grid whose resident blocks all spin waiting for a push
kernel that has not been dispatched yet never finishes. Block 0 is dispatched
first, so as part of the GEMM the push cannot be starved. It costs that block's
own tile the push time -- a local tile, which nothing waits on.
"""

# NOTE: no `from __future__ import annotations`, as in gemm_ar/kernels_fused.py:
# `fx.struct` reads the SharedStorage annotations as live objects.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int64

import mori.cco.device.flydsl as cco
from mori.cco.device.flydsl import _bindings as raw_cco

from ..gemm_ar._compat import (
    local_load_u32,
    local_store_u32,
    signal_ptr,
    signal_store_u32,
)
from ..gemm_ar._gemm_a8w8_8wave import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    split_row_major_2d,
    wait_barrier,
)
from ..gemm_ar.kernels_fused import _LaneTransposeStoreC, _SwappedMfma
from ..gemm_ar.kernels_lsa import _spin_until
from .layout import MAX_CHUNKS, MAX_WORLD

BLOCK_K = 128
#: Wave width; the push kernel and the barrier use one wave, a lane per peer.
PUSH_THREADS = 64


def _lsa(base, stride, peer, offset):
    """``CachedWindow.lsa_ptr`` from its two fields.

    The window object itself cannot be captured by a runtime ``if`` -- FlyDSL
    requires every value a dynamic branch reads to be a single MLIR value, and
    CachedWindow is two (see its docstring) -- so kernels take ``base`` and
    ``stride`` out before branching and build peer addresses with this.
    """
    return base + fx.Int64(peer) * stride + fx.Int64(offset)


def _epoch_now(base, stride, rank, cfg):
    """``(this call's epoch, rsrc of the local counter)``: ``counter + 1``."""
    rsrc = signal_ptr(_lsa(base, stride, rank, cfg.epoch_off))
    return fx.Int32(local_load_u32(rsrc)) + fx.Int32(1), rsrc


def _post_rows(sdma, base, stride, win, rank, cfg, peer, ep, queue):
    """Push this rank's rows to ``peer``, band by band, flagging each band.

    One lane per peer, so the peers are served in parallel, each on its own
    (source, destination) queue. Within a peer the bands are sequential and each
    ends in ``quiet``: the flag is a promise that the band has *landed*, and
    ``quiet`` is the only completion the copy engine reports. The scales ride on
    the same queue as their rows, so they cannot overtake them.
    """
    flag_base = _lsa(base, stride, peer, cfg.flag_off)
    for c in range_constexpr(cfg.chunks):
        row_off = cfg.recv_slot_off(rank) + c * cfg.chunk_bytes
        sca_off = cfg.scale_slot_off(rank) + c * cfg.chunk_scale_bytes
        sdma.put(
            peer,
            win,
            fx.Int64(row_off),
            win,
            fx.Int64(row_off),
            fx.Int64(cfg.chunk_bytes),
            queue,
            coop=cco.CoopScope.THREAD,
            signal=False,
        )
        sdma.put(
            peer,
            win,
            fx.Int64(sca_off),
            win,
            fx.Int64(sca_off),
            fx.Int64(cfg.chunk_scale_bytes),
            queue,
            coop=cco.CoopScope.THREAD,
            signal=False,
        )
        sdma.quiet_queue(peer, queue)
        # Keep the flag store from being hoisted above the drain: the bytes were
        # moved by the copy engine, so there is nothing of ours to flush, only
        # the order the receiver depends on (same as gemm_ar's push kernel).
        raw_cco.cco_system_fence(fx.Int32(0))
        flag = signal_ptr(flag_base + fx.Int64((rank * MAX_CHUNKS + c) * 4))
        signal_store_u32(flag, ep)


def compile_ag_gemm(
    cfg,
    rank: int,
    *,
    N: int,
    BLOCK_N: int = 256,
    fuse: bool = True,
    waves_per_eu: int = 2,
    sdma_queues: int = 1,
):
    """Compile the all-gather + GEMM (``fuse=True``) or the plain GEMM over the
    already-gathered window (``fuse=False``).

    ptpc only: the A scale is one fp32 per row, so it travels with its row.
    blockscale and mxfp8 lay the A scale out K-block major, where a row band is
    strided, and need a different push.

    Returns ``launch(A, B_T, C, A_scale, B_scale, dev_comm, win, stream=...)``.
    ``A`` and ``A_scale`` are the window's ``recv`` and ``recv_scale`` regions,
    ``[world*m_local, K]`` fp8 and ``[world*m_local]`` fp32; ``B_T`` is this
    rank's ``[N, K]`` weight through ``preshuffle_b``; ``C`` is ``[world*m_local,
    N]`` bf16.
    """
    cfg.validate()
    ws = cfg.world_size
    K = cfg.k
    BLOCK_M = cfg.block_m
    M = cfg.m
    if N % BLOCK_N:
        raise ValueError(f"N={N} must be a multiple of BLOCK_N={BLOCK_N}")
    if BLOCK_N != 256:
        raise ValueError(
            "the lane-transpose store pairs exactly two N-tiles: BLOCK_N=256"
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

    nb = N // BLOCK_N
    mt_local = cfg.m_tiles_local
    mtpc = cfg.m_tiles_per_chunk
    local_tiles = mt_local * nb
    grid = (M // BLOCK_M) * nb

    _kname = (
        f"mori_ag_gemm_{'fused' if fuse else 'plain'}_8w_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}"
        f"_k{K}_n{N}_ml{cfg.m_local}_w{ws}_c{cfg.chunks}_r{rank}"
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
    def kernel_ag_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        dev_comm: Int64,
        win: Int64,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type
        tid = fx.thread_idx.x
        t = fx.block_idx.x
        w_pre = cco.CachedWindow(win)
        wbase, wstride = w_pre.base, w_pre.stride

        if const_expr(fuse):
            ep, _ = _epoch_now(wbase, wstride, rank, cfg)
            # Built outside the branches: a runtime `if` captures its free
            # values, and those have to be single MLIR values (gemm_ar does
            # the same with its window bases).
            sdma = cco.DevComm(dev_comm).sdma()

            # ---- the poster: block 0, one lane per peer ----
            if t == fx.Int32(0):
                if tid < ws:
                    if tid != rank:
                        # The caller wrote this rank's rows with an ordinary
                        # kernel; make them visible to the copy engine reading
                        # memory before the first put of them.
                        raw_cco.cco_system_fence(fx.Int32(0))
                        _post_rows(
                            sdma,
                            wbase,
                            wstride,
                            win,
                            rank,
                            cfg,
                            tid,
                            ep,
                            tid % fx.Int32(sdma_queues),
                        )

            # ---- tile order: own rows, then (chunk, source, row tile, column tile) ----
            is_local = t < fx.Int32(local_tiles)
            lm, lbn = split_row_major_2d(t, nb)
            r = arith.select(is_local, fx.Int32(0), t - fx.Int32(local_tiles))
            rest, rbn = split_row_major_2d(r, nb)
            rest, mm = split_row_major_2d(rest, mtpc)
            chunk, j = split_row_major_2d(rest, ws - 1)
            src = (fx.Int32(rank + 1) + j) % fx.Int32(ws)
            block_m = arith.select(
                is_local,
                fx.Int32(rank * mt_local) + lm,
                src * fx.Int32(mt_local) + chunk * fx.Int32(mtpc) + mm,
            )
            block_n = arith.select(is_local, lbn, rbn)

            # ---- a remote tile waits for its band before touching A ----
            # Taken out here so the branch captures one Int64, not the window.
            my_flags = _lsa(wbase, wstride, rank, cfg.flag_off)
            flag_addr = my_flags + fx.Int64(
                src * fx.Int32(MAX_CHUNKS) + chunk
            ) * fx.Int64(4)
            if t >= fx.Int32(local_tiles):
                if tid == fx.Int32(0):
                    _spin_until(signal_ptr(flag_addr), ep)
                    # Acquire: the rows were written by a peer's copy engine,
                    # and this CU may still hold the previous call's lines.
                    raw_cco.cco_system_fence(fx.Int32(0))
            fgpu.barrier()
        else:
            block_m, block_n = split_row_major_2d(t, nb)

        # ---- the 8-wave fp8 GEMM, as in gemm_ar/kernels_fused.py (ptpc) ----
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = tid % 64
        wave_id = tid // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4

        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B_K_STEP = 2 * 1024
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
            lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=True
        )

        mfma = _SwappedMfma(Mfma16x16x128(N_TILES_B, N_TILES_A))
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        store_c = _LaneTransposeStoreC(
            A_scale,
            B_scale,
            C,
            fx.Int32(M),
            fx.Int32(N),
            mfma.idx,
            N_TILES_A,
            N_TILES_B,
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
            b0_frag = b_s2r.load(b_cur0, preshuffled=True)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

            b1_frag = b_s2r.load(b_cur1, preshuffled=True)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
            rocdl.s_barrier()

            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
            # gemm_ar's count, not aiter's: see the long comment at the same
            # line in gemm_ar/kernels_fused.py. Do not "restore" it.
            wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B - 1)

            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        b0_frag = b_s2r.load(b_cur0, preshuffled=True)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        b1_frag = b_s2r.load(b_cur1, preshuffled=True)
        rocdl.s_barrier()
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        a1_frag = a_s2r.load(a_cur1)
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
        rocdl.s_barrier()
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        b0_frag = b_s2r.load(b_next0, preshuffled=True)
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
        b1_frag = b_s2r.load(b_cur1, preshuffled=True)
        rocdl.s_barrier()
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, set_prio=False)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, set_prio=False)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        base_row = block_m * BLOCK_M + wave_m * (N_TILES_A * 16)
        base_col = block_n * BLOCK_N + wave_n * (N_TILES_B * 16)
        # Close the half-wave barrier pair the prologue opened (gemm_ar's
        # reasoning applies: leave it open and anything after the stores runs a
        # phase skewed between the two half-blocks).
        if wave_m == 0:
            rocdl.s_barrier()
        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        dev_comm: Int64,
        win: Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel_ag_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            dev_comm,
            win,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(grid=(grid, 1, 1), block=(512, 1, 1), stream=stream)

    return launch


def build_phases(cfg, rank: int, *, sdma_queues: int = 1):
    """``{"push", "barrier0", "barrier1"}`` launchers, ``(dev_comm, win, stream=)``.

    ``push`` is the fused kernel's poster as a kernel of its own, for the split
    path; it returns once every band it sent has landed. ``barrier0`` is the
    split path's "everything landed here" point; ``barrier1`` closes every call
    on both paths and advances the epoch.
    """
    cfg.validate()
    ws = cfg.world_size

    @flyc.kernel(known_block_size=[PUSH_THREADS, 1, 1])
    def push(dev_comm: Int64, win: Int64):
        tid = fx.thread_idx.x
        w = cco.CachedWindow(win)
        wbase, wstride = w.base, w.stride
        ep, _ = _epoch_now(wbase, wstride, rank, cfg)
        sdma = cco.DevComm(dev_comm).sdma()
        if tid < ws:
            if tid != rank:
                raw_cco.cco_system_fence(fx.Int32(0))
                _post_rows(
                    sdma,
                    wbase,
                    wstride,
                    win,
                    rank,
                    cfg,
                    tid,
                    ep,
                    tid % fx.Int32(sdma_queues),
                )

    def _barrier(which: int, bump: bool):
        @flyc.kernel(known_block_size=[PUSH_THREADS, 1, 1])
        def barrier(dev_comm: Int64, win: Int64):
            tid = fx.thread_idx.x
            w = cco.CachedWindow(win)
            wbase, wstride = w.base, w.stride
            ep, epoch_rsrc = _epoch_now(wbase, wstride, rank, cfg)
            my_bar = _lsa(wbase, wstride, rank, cfg.bar_off + which * MAX_WORLD * 4)
            if tid < ws:
                if tid != rank:
                    raw_cco.cco_system_fence(fx.Int32(0))
                    peer_slot = _lsa(wbase, wstride, tid, cfg.bar_slot_off(which, rank))
                    signal_store_u32(signal_ptr(peer_slot), ep)
                    _spin_until(signal_ptr(my_bar + fx.Int64(tid) * fx.Int64(4)), ep)
            fgpu.barrier()
            if const_expr(bump):
                if tid == fx.Int32(0):
                    local_store_u32(epoch_rsrc, ep)

        return barrier

    barrier0 = _barrier(0, False)
    barrier1 = _barrier(1, True)

    def _phase(kern):
        @flyc.jit
        def go(dev_comm: Int64, win: Int64, stream=fx.Stream(None)):
            kern(dev_comm, win).launch(
                grid=(1, 1, 1), block=[PUSH_THREADS, 1, 1], stream=stream
            )

        return go

    return {
        "push": _phase(push),
        "barrier0": _phase(barrier0),
        "barrier1": _phase(barrier1),
    }


__all__ = ["compile_ag_gemm", "build_phases", "BLOCK_K"]
