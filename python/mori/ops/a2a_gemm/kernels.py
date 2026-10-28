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
"""The 8-wave fp8 GEMM with the all-to-all that feeds it absorbed into it.

Three launchers, all over the window ``layout.A2aGemmConfig`` describes:

``compile_a2a_gemm(fuse=True)``
    One kernel. Block 0 doubles as the *poster*: lanes ``0..world-1`` each own
    one peer and push it its row block, ``send[peer]``, one put per band, back
    to back. Each put carries a trailing copy-engine atomic that bumps the
    peer's signal counter for this rank: same queue, after the bytes, so seeing
    the count is seeing the band. Every block then computes one tile; tiles go
    band-major, so the first wave needs only band 0 of every source. A tile
    walks K one source segment at a time starting with this rank's own, which
    never travels, then the others rank-rotated, and waits for a remote band
    either per segment or once up front (``WAIT_MODES``). Block 0 takes the last
    tile, which waits for the last band anyway, so posting costs it nothing.

``compile_a2a_gemm(fuse=False)``
    The same GEMM, no waits: the split path's compute, run after ``push`` and
    ``barrier0`` (or RCCL) have filled ``recv``. ``self_from_send=False`` reads
    this rank's own segment from ``recv`` too, for RCCL, which copies it there.

``build_phases``
    ``arm`` (snapshot the signal counters into ``sig_base``: once, before the
    first call), ``push`` (the poster as its own one-block kernel) and the two
    cross-rank barriers. ``barrier1`` (bump) closes every call: nobody may
    refill a peer's ``recv`` while that peer still reads it.

Waiting per K segment rather than per tile is what lets a tile start before
anything has arrived: its own segment is ``1/world`` of its K. The wait sits
between two prefetches, so its flag load drains the wave's outstanding tile
loads with it (``vmcnt`` retires in order); that costs one pipeline bubble per
remote segment and keeps every later ``s_waitcnt`` count conservative.
"""

# NOTE: no `from __future__ import annotations`, as in gemm_ar/kernels_fused.py:
# `fx.struct` reads the SharedStorage annotations as live objects.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int64

import mori.cco.device.flydsl as cco
from mori.cco.device.flydsl import _bindings as raw_cco

from ..gemm_ar._compat import (
    local_load_u32,
    local_store_u32,
    signal_load_u32,
    signal_ptr,
    signal_store_u32,
)
from ..gemm_ar._gemm_a8w8_8wave import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    read_done_barrier,
    split_row_major_2d,
    wait_barrier,
)
from ..gemm_ar.kernels_fused import _LaneTransposeStoreC, _SwappedMfma
from ..gemm_ar.kernels_lsa import _spin_until
from .layout import BLOCK_K, MAX_CHUNKS, MAX_WORLD

#: Wave width; the push kernel and the barriers use one wave, a lane per peer.
PUSH_THREADS = 64
#: Bytes between consecutive K steps of a ``preshuffle_b`` weight.
B_K_STEP = 2 * 1024


def _lsa(base, stride, peer, offset):
    """``CachedWindow.lsa_ptr`` from its two fields (see ag_gemm: a runtime
    ``if`` can only capture single MLIR values, and the window is two)."""
    return base + fx.Int64(peer) * stride + fx.Int64(offset)


def _epoch_now(base, stride, rank, cfg):
    """``(this call's epoch, rsrc of the local counter)``: ``counter + 1``."""
    rsrc = signal_ptr(_lsa(base, stride, rank, cfg.epoch_off))
    return fx.Int32(local_load_u32(rsrc)) + fx.Int32(1), rsrc


#: How a tile waits for remote bytes. ``segment``: every wave spins on a
#: segment's band just before its first A load of it, so the tile's own segment
#: overlaps the transfer. ``tile``: thread 0 waits for the band from every
#: source before the prologue, one fence for the tile -- no in-loop waits.
WAIT_MODES = ("segment", "tile")
#: Diagnostic only: post but never wait. The result is not validated (tiles read
#: whatever recv holds); the time separates what the poster costs the GEMM from
#: what the waits cost.
_DIAG_WAIT = "none"


#: What a tile does after seeing a band land, before reading it. ``inv``: a
#: system-scope acquire, which invalidates the caches (``buffer_inv sc0 sc1``)
#: so no line from the previous call survives. ``none``: nothing beyond the
#: spin itself, whose loads the A loads cannot pass. Correct only while the
#: window is uncached -- cco's default (``CCO_UNCACHED_WINDOW``), where A's
#: loads never hit a cached line to begin with.
ACQUIRE_MODES = ("inv", "none")


def _acquire(mode):
    """``cco_system_fence`` would be a full ``__threadfence_system``, whose
    release half writes back L2 -- dirty with this kernel's own C stores --
    which a reader of copy-engine data has no use for."""
    if const_expr(mode == "inv"):
        _llvm_d.fence(_llvm_d.AtomicOrdering.acquire)


def _post_slab(sdma, win, rank, cfg, peer, queue, *, quiet=True):
    """Push ``send[peer]`` into ``peer``'s ``recv[rank]``, one put per band.

    One lane per peer, so the peers are served in parallel, each on its own
    (source, destination) queue. The bands go back to back: each put's trailing
    atomic tells the peer that band has landed, so nothing here waits between
    them. The closing ``quiet`` only keeps this rank from reusing ``send``, or
    returning from ``push``, while its bytes are still on the wire; the fused
    kernel defers it past its own tile (``quiet=False``, then ``_drain``).
    """
    src_off = fx.Int64(cfg.send_off) + fx.Int64(peer) * fx.Int64(cfg.slab_bytes)
    dst_off = cfg.recv_slot_off(rank)
    for c in range_constexpr(cfg.chunks):
        sdma.put(
            peer,
            win,
            fx.Int64(dst_off + c * cfg.chunk_bytes),
            win,
            src_off + fx.Int64(c * cfg.chunk_bytes),
            fx.Int64(cfg.chunk_bytes),
            queue,
            coop=cco.CoopScope.THREAD,
            remote_signal=True,
        )
    if quiet:
        sdma.quiet_queue(peer, queue)


def _band_target(base, stride, rank, cfg, src, ep, chunk):
    """The count of ``src``'s signal counter that means ``chunk`` has landed."""
    sig_base = signal_ptr(_lsa(base, stride, rank, cfg.sig_base_slot_off(src)))
    done = (ep - fx.Int32(1)) * fx.Int32(cfg.chunks) + chunk + fx.Int32(1)
    return fx.Int32(local_load_u32(sig_base)) + done


def _realtime():
    """``s_memrealtime``: a 64-bit 100 MHz clock, constant across DVFS."""
    i64 = ir.IntegerType.get_signless(64)
    return fx.Int64(
        _llvm_d.call_intrinsic(i64, "llvm.amdgcn.s.memrealtime", [], [], [])
    )


def _stamp(base_i64, slot):
    """Diagnostic: now, at u64 slot ``slot`` (a runtime or compile-time index)."""
    addr = fx.Int64(base_i64) + fx.Int64(slot) * fx.Int64(8)
    _llvm_d.store(_realtime().ir_value(), signal_ptr(addr), alignment=8, volatile_=True)


def _observe_bands(sdma, base, stride, rank, cfg, peer, ep, qid, stamps):
    """Diagnostic: lane ``peer`` stamps when each of ``peer``'s bands lands here."""
    sig_base = signal_ptr(
        _lsa(base, stride, rank, cfg.sig_base_off) + fx.Int64(peer) * fx.Int64(4)
    )
    first = fx.Int32(local_load_u32(sig_base)) + (ep - fx.Int32(1)) * fx.Int32(
        cfg.chunks
    )
    slot = signal_ptr(sdma.signal_slot(peer, qid))
    for c in range_constexpr(cfg.chunks):
        _spin_until(slot, first + fx.Int32(c + 1))
        _stamp(stamps, fx.Int32(3) + peer * fx.Int32(MAX_CHUNKS) + fx.Int32(c))


def _wait_band(sdma, base, stride, rank, cfg, src, ep, chunk, qid):
    """Spin until ``src``'s band ``chunk`` of this call has landed here."""
    target = _band_target(base, stride, rank, cfg, src, ep, chunk)
    _spin_until(signal_ptr(sdma.signal_slot(fx.Int32(src), qid)), target)


def compile_a2a_gemm(
    cfg,
    rank: int,
    *,
    N: int,
    BLOCK_N: int = 256,
    fuse: bool = True,
    self_from_send: bool = True,
    waves_per_eu: int = 2,
    sdma_queues: int = 1,
    wait: str = "segment",
    acquire: str = "inv",
    _post: bool = True,
    _timeline: bool = False,
    _grid: int = 0,
):
    """Compile the all-to-all + GEMM (``fuse=True``) or the plain GEMM over an
    already-filled ``recv`` (``fuse=False``).

    ptpc: the A scale is one fp32 per token over all of K, ``[m_local]``, local.

    Returns ``launch(A, B_T, C, A_scale, B_scale, dev_comm, win, stream=...)``.
    ``A`` is the window from ``send_off`` for ``operand_bytes`` bytes; ``B_T``
    the ``[N, K]`` weight through ``preshuffle_b``; ``C`` ``[m_local, N]`` bf16.
    """
    cfg.validate()
    if wait not in WAIT_MODES + (_DIAG_WAIT,):
        raise ValueError(f"wait must be one of {WAIT_MODES}, got {wait!r}")
    if acquire not in ACQUIRE_MODES:
        raise ValueError(f"acquire must be one of {ACQUIRE_MODES}, got {acquire!r}")
    if fuse and not self_from_send:
        raise ValueError(
            "the fused path pushes from send, so it reads its own from there"
        )
    if _timeline and not (fuse and cfg.timeline):
        raise ValueError("_timeline needs fuse=True and a cfg with timeline=True")
    ws = cfg.world_size
    K = cfg.k
    KS = cfg.k_steps_per_shard
    K_SHARD = cfg.k_shard
    BLOCK_M = cfg.block_m
    M = cfg.m_local
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
    mtpc = cfg.m_tiles_per_chunk
    tiles = cfg.m_tiles * nb

    # K step j -> (source, A offset, B offset), all compile-time: this rank's own
    # segment first, then the others rank-rotated.
    def _src(j):
        return (rank + j // KS) % ws

    def a_off(j):
        seg = cfg.segment_off(_src(j), rank, self_from_send=self_from_send)
        return seg + (j % KS) * BLOCK_K

    def b_off(j):
        return (_src(j) * KS + j % KS) * B_K_STEP

    def waits_before(j):
        return fuse and wait == "segment" and j % KS == 0 and _src(j) != rank

    _kname = (
        f"mori_a2a_gemm_{'fused' if fuse else 'plain'}"
        f"{'' if self_from_send else '_rv'}_8w_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}"
        f"_k{K}_n{N}_ml{M}_w{ws}_c{cfg.chunks}{'_wt' if wait == 'tile' else ('_wn' if wait == _DIAG_WAIT else '')}{'_an' if acquire == 'none' else ''}{'' if _post else '_np'}{'_tl' if _timeline else ''}"
        f"_r{rank}"
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
    def kernel_a2a_gemm(
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
            # Built outside the branch: a runtime `if` captures its free values,
            # and those have to be single MLIR values.
            sdma = cco.DevComm(dev_comm).sdma()

            # Every source pushes to this rank on its queue rank % sdma_queues.
            qid = fx.Int32(rank % sdma_queues)
            stamps = _lsa(wbase, wstride, rank, cfg.stamp_off)
            if const_expr(_timeline):
                if t == fx.Int32(0):
                    if tid == fx.Int32(0):
                        _stamp(stamps, 0)

            # ---- the poster: block 0, one lane per peer ----
            # (_post=False is diagnostic: no block index matches -1.)
            if t == fx.Int32(0 if _post else -1):
                if tid < ws:
                    if tid != rank:
                        # The caller wrote send with an ordinary kernel; make it
                        # visible to the copy engine before the first put of it.
                        raw_cco.cco_system_fence(fx.Int32(0))
                        _post_slab(
                            sdma,
                            win,
                            rank,
                            cfg,
                            tid,
                            tid % fx.Int32(sdma_queues),
                            quiet=False,
                        )
                        if const_expr(_timeline):
                            if tid == fx.Int32((rank + 1) % ws):
                                _stamp(stamps, 1)
                            _observe_bands(
                                sdma, wbase, wstride, rank, cfg, tid, ep, qid, stamps
                            )
            # Block 0 takes the last tile, which waits for the last band
            # regardless. Its quiet waits until after that tile: draining first
            # put the whole transfer *and then* a full tile on the critical path
            # (+51us at P=8, K=N=8192, whatever M).
            tile = (t + fx.Int32(tiles - 1)) % fx.Int32(tiles)
            block_m, block_n = split_row_major_2d(tile, nb)
            chunk = block_m // fx.Int32(mtpc)
            tile_slot = fx.Int32(cfg.STAMP_FIXED) + tile * fx.Int32(2)
            if const_expr(_timeline):
                if tid == fx.Int32(0):
                    _stamp(stamps, tile_slot)
            if const_expr(wait == "tile"):
                if tid == fx.Int32(0):
                    for j in range_constexpr(1, ws):
                        _wait_band(
                            sdma,
                            wbase,
                            wstride,
                            rank,
                            cfg,
                            (rank + j) % ws,
                            ep,
                            chunk,
                            qid,
                        )
                    _acquire(acquire)
                fgpu.barrier()
        else:
            block_m, block_n = split_row_major_2d(t, nb)

        def wait_segment(j):
            """Every wave, before its first A load of step ``j``'s segment."""
            if const_expr(waits_before(j)):
                _wait_band(sdma, wbase, wstride, rank, cfg, _src(j), ep, chunk, qid)
                # Acquire: the bytes were written by a peer's copy engine, and
                # this CU may still hold the previous call's lines.
                _acquire(acquire)

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

        # A is per-source [m_local][k_shard]: the row stride is a segment's.
        A0_gl_offset = (block_m * BLOCK_M) * K_SHARD
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K_SHARD
        B0_gl_offset = (block_n * BLOCK_N) * K
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
        gl_off_a = compute_global_swizzle(
            lane_id, wave_id, K_SHARD, N_LDS_ROUNDS, preshuffled=False
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

        b_g2s.load(b_cur0, B0_gl_offset + b_off(0))
        a_g2s.load(a_cur0, A0_gl_offset + a_off(0))
        b_g2s.load(b_cur1, B1_gl_offset + b_off(0))
        a_g2s.load(a_cur1, A1_gl_offset + a_off(0))

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + b_off(1))
        wait_segment(1)
        a_g2s.load(a_next0, A0_gl_offset + a_off(1))
        b_g2s.load(b_next1, B1_gl_offset + b_off(1))

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        for k in range_constexpr(K_ITERS - 2):
            b0_frag = b_s2r.load(b_cur0, preshuffled=True)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + a_off(k + 1))
            # LDS reads done before the leading wave group refills B0 (#785).
            read_done_barrier()

            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

            b1_frag = b_s2r.load(b_cur1, preshuffled=True)
            b_g2s.load(b_cur0, B0_gl_offset + b_off(k + 2))
            rocdl.s_barrier()

            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)

            a1_frag = a_s2r.load(a_cur1)
            wait_segment(k + 2)
            a_g2s.load(a_cur0, A0_gl_offset + a_off(k + 2))
            rocdl.s_barrier()

            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)

            b_g2s.load(b_cur1, B1_gl_offset + b_off(k + 2))
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        b0_frag = b_s2r.load(b_cur0, preshuffled=True)
        a0_frag = a_s2r.load(a_cur0)
        # Drain the producers of step K_ITERS - 1's B0 and A0, which the peeled
        # steps read before the last wait_barrier(0) (#785).
        wait_barrier(0)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        b1_frag = b_s2r.load(b_cur1, preshuffled=True)
        rocdl.s_barrier()
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        a1_frag = a_s2r.load(a_cur1)
        a_g2s.load(a_next1, A1_gl_offset + a_off(K_ITERS - 1))
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

        if const_expr(fuse):
            if const_expr(_timeline):
                if tid == fx.Int32(0):
                    _stamp(stamps, tile_slot + fx.Int32(1))
            if t == fx.Int32(0 if _post else -1):
                if tid < ws:
                    if tid != rank:
                        sdma.quiet_queue(tid, tid % fx.Int32(sdma_queues))
                        if const_expr(_timeline):
                            if tid == fx.Int32((rank + 1) % ws):
                                _stamp(stamps, 2)

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
        kernel_a2a_gemm(
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
        ).launch(grid=(_grid or tiles, 1, 1), block=(512, 1, 1), stream=stream)

    return launch


def build_phases(cfg, rank: int, *, sdma_queues: int = 1):
    """``{"arm", "push", "barrier0", "barrier1"}`` launchers, ``(dev_comm, win,
    stream=)``.

    ``arm`` snapshots the copy engine's signal counters into ``sig_base``; run it
    once on every rank, with the epoch at zero and no push in flight, then a
    host barrier, before the first call -- and never again while the epoch
    counts on from it.

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
        sdma = cco.DevComm(dev_comm).sdma()
        if tid < ws:
            if tid != rank:
                raw_cco.cco_system_fence(fx.Int32(0))
                _post_slab(sdma, win, rank, cfg, tid, tid % fx.Int32(sdma_queues))

    @flyc.kernel(known_block_size=[PUSH_THREADS, 1, 1])
    def arm(dev_comm: Int64, win: Int64):
        tid = fx.thread_idx.x
        w = cco.CachedWindow(win)
        wbase, wstride = w.base, w.stride
        sdma = cco.DevComm(dev_comm).sdma()
        if tid < ws:
            if tid != rank:
                qid = fx.Int32(rank % sdma_queues)
                now = signal_load_u32(signal_ptr(sdma.signal_slot(tid, qid)))
                slot = _lsa(wbase, wstride, rank, cfg.sig_base_off) + fx.Int64(
                    tid
                ) * fx.Int64(4)
                local_store_u32(signal_ptr(slot), now)

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
        "arm": _phase(arm),
        "push": _phase(push),
        "barrier0": _phase(barrier0),
        "barrier1": _phase(barrier1),
    }


__all__ = ["compile_a2a_gemm", "build_phases"]
