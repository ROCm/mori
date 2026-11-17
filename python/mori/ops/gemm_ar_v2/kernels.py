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
"""Kernels of ``gemm_ar_v2``: the bf16 GEMM into the window, and the tail.

GEMM: the 8-wave quad-subtile bf16 pipeline, its main loop as in
``gemm_ag/_gemm_a16w16_8wave.py``'s ``compile_bf16_gemm_ag`` (copied), writing
this rank's partial C into ``Y`` write-through, with tiles ordered chunk-major
(see ``layout``). ``grid="chunk"`` launches one chunk (``v2-stream``),
``grid="all"`` the whole matrix (``split`` and ``fused``). ``fused=True`` adds the
last-finisher epilogue: one counter per (destination, chunk), the block whose
increment completes it posts that chunk's put.

Tail, one 64-thread block each unless noted: ``put_chunk`` (one put per peer,
lane = peer), ``barrier`` (optionally push the reduced band to every peer, drain,
publish this call's epoch to every peer, wait for every peer's), and
``reduce`` (all CUs: band ``rank`` = sum over sources in rank order, fp32, one
bf16 rounding). Epochs are running u32 values, compared as
``(int32)(cur - e) < 0`` so they survive 2^32.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int64
from flydsl.expr.typing import Vector as Vec

import mori.cco.device.flydsl as cco
from mori.cco.device.flydsl import _bindings as raw_cco
from mori.cco.device.flydsl.buffer import (
    CM_CACHED,
    CM_SC0_SC1,
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from mori.cco.device.flydsl.signals import (
    i32_type,
    local_load_u32,
    local_store_u32,
    signal_load_u32,
    signal_ptr,
    signal_store_u32,
)

from mori.ops.gemm_ar_v2 import layout as L
from mori.ops.gemm_ar_v2._blocks import (
    BLOCK_K,
    AgStoreC,
    G2SLoader,
    MfmaBf16,
    S2RLoaderBf16,
    SwappedMfma,
    _wait_lds_reads,
    atomic_add_u32,
    compute_global_swizzle_bf16,
    make_bf16_buffer_tensor,
    tile_constants,
    wait_barrier,
    wave_uniform_i64,
)

THREADS = 512
TAIL_THREADS = 64
#: 16-byte vectors per thread in the reduce kernel
REDUCE_UNROLL = 4


def _named(fn, name):
    """Give a ``@flyc.jit`` launcher a unique name per configuration."""
    fn.__name__ = name
    fn.__qualname__ = name
    return flyc.jit(fn)


def spin_until(ptr, target):
    """Block until the u32 at ``ptr`` reaches ``target``, wrap-safe.

    ``(int32)(cur - target) < 0`` means not yet, so the compare holds across
    2^32 (the construction of ``gemm_ag/kernels_sdma.py::_spin_until``).
    """
    i32 = i32_type()
    first = signal_load_u32(ptr)
    first_v = first.ir_value() if hasattr(first, "ir_value") else first
    loop = scf.WhileOp([i32], [first_v])
    cond = ir.Block.create_at_start(loop.before, [i32])
    body = ir.Block.create_at_start(loop.after, [i32])
    with ir.InsertionPoint(cond):
        cur = fx.Int32(cond.arguments[0])
        should_wait = (cur - fx.Int32(target)) < fx.Int32(0)
        scf.ConditionOp(should_wait.ir_value(), [cond.arguments[0]])
    with ir.InsertionPoint(body):
        nxt = signal_load_u32(ptr)
        scf.YieldOp([nxt.ir_value() if hasattr(nxt, "ir_value") else nxt])


# ---------------------------------------------------------------------------
# GEMM
# ---------------------------------------------------------------------------


def compile_gemm(
    shape: L.Shape,
    lay: L.Layout,
    rank: int,
    *,
    K: int,
    grid: str = "chunk",
    fused: bool = False,
    queues: int = 1,
    ctr_order: str = "acq_rel",
    emit_put: bool = True,
    swap_store: bool = True,
    waves_per_eu: int = 2,
):
    """Returns ``launch(A, B_T, chunk, dev_comm, win, stream=...)``.

    ``A`` / ``B_T`` are flat int16 views of bf16 ``[Mp, K]`` / ``[N, K]``.
    ``chunk`` selects the chunk for ``grid="chunk"`` and is ignored otherwise.
    """
    shape.validate()
    assert grid in ("chunk", "all")
    assert ctr_order in ("acq_rel", "release", "monotonic")
    if fused:
        assert grid == "all"
        assert queues >= shape.chunks or queues == 1
        if queues == 1:
            assert shape.chunks == 1, "queues=1 with chunks>1 needs a lock (not built)"
    W = shape.world
    N = shape.n
    NB = shape.n_blocks
    TPU = shape.tiles_per_unit
    PER_CHUNK = shape.tiles_per_chunk
    MPC = shape.mpc
    BAND_TILES = shape.band_tiles
    MP = shape.mp
    BM, BN = L.BLOCK_M, L.BLOCK_N
    Y_OFF = lay.y_off
    Y_BYTES = MP * N * 2
    R_MINE = lay.r_off + rank * lay.slot_bytes
    BAND_BYTES = shape.band_bytes
    CHUNK_BYTES = shape.chunk_bytes
    CTR_OFF, BCTR_OFF = L.CTR_OFF, L.BCTR_OFF
    MAXC = L.MAX_CHUNKS
    GRID = PER_CHUNK if grid == "chunk" else shape.chunks * PER_CHUNK
    ROT = rank + 1

    tc = tile_constants(BM, BN, "bf16")
    N_TILES_A, N_TILES_B = tc["N_TILES_A"], tc["N_TILES_B"]
    N_ACCUMS = tc["N_ACCUMS"]
    LDS_BLOCK_M, LDS_BLOCK_N = tc["LDS_BLOCK_M"], tc["LDS_BLOCK_N"]
    N_LDS_STEPS_A, N_LDS_STEPS_B = tc["N_LDS_STEPS_A"], tc["N_LDS_STEPS_B"]
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    a_lds_size, b_lds_size = tc["a_lds_size"], tc["b_lds_size"]
    assert K % BLOCK_K == 0
    K_ITERS = K // BLOCK_K
    assert K_ITERS >= 2
    # gemm_ag's "tuned" policy: min(2A+B, A+2B) outstanding G2S loads.
    MAIN_WAIT = min(2 * N_LDS_STEPS_A + N_LDS_STEPS_B, N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

    kname = (
        f"mori_garv2_gemm_{grid}_{'F' if fused else 'P'}"
        f"{'p' if emit_put else 'x'}q{queues}{ctr_order[0:3]}"
        f"{'s' if swap_store else 'b'}_mp{MP}_n{N}_k{K}_c{shape.chunks}"
        f"_w{W}_r{rank}"
    )

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.BFloat16, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.BFloat16, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.BFloat16, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.BFloat16, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.BFloat16, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.BFloat16, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.BFloat16, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.BFloat16, b_lds_size, 16]

    @flyc.kernel(name=kname, known_block_size=[THREADS, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        chunk: fx.Int32,
        dev_comm: Int64,
        win: Int64,
    ):
        w = cco.CachedWindow(win)
        wbase = w.base

        # ---- tile order: chunk-major, destinations rotated by rank ----
        bid = fx.block_idx.x
        if const_expr(grid == "all"):
            c = bid // PER_CHUNK
            r = bid % PER_CHUNK
        else:
            c = chunk
            r = bid
        j = r // TPU
        rr = r % TPU
        i = rr // NB
        block_n = rr % NB
        d = (j + ROT) % W
        block_m = d * BAND_TILES + c * MPC + i

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

        A0_gl_offset = (block_m * BM) * K
        A1_gl_offset = (block_m * BM + LDS_BLOCK_M) * K
        B_K_STEP = BLOCK_K
        B0_gl_offset = (block_n * BN) * K
        B1_gl_offset = (block_n * BN + LDS_BLOCK_N) * K

        gA = make_bf16_buffer_tensor(A)
        gB = make_bf16_buffer_tensor(B_T)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
        gl_off_a = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS)
        gl_off_b = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS)

        mfma = SwappedMfma(MfmaBf16(N_TILES_B, N_TILES_A))

        BF16_IR_t = fx.BFloat16.ir_type
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, BF16_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, BF16_IR_t, wave_id)
        a_s2r = S2RLoaderBf16(wave_m, N_TILES_A)
        b_s2r = S2RLoaderBf16(wave_n, N_TILES_B)

        y_rsrc = create_buffer_resource_from_addr(
            wave_uniform_i64(fx.Int64(wbase) + fx.Int64(rank) * w.stride + fx.Int64(Y_OFF)),
            num_records_bytes=Y_BYTES,
        )
        store_c = AgStoreC(
            None,
            MP,
            N,
            mfma.idx,
            N_TILES_A,
            N_TILES_B,
            out_dtype="bf16",
            peer_rsrc=y_rsrc,
            peer_uncached=True,
            swap=swap_store,
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
            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            _wait_lds_reads()
            rocdl.s_barrier()

            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
            rocdl.s_barrier()

            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
            wait_barrier(MAIN_WAIT)

            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)

        a1_frag = a_s2r.load(a_cur1)
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
        wait_barrier(0)

        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)

        b0_frag = b_s2r.load(b_next0)
        rocdl.s_barrier()

        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 1
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

        b1_frag = b_s2r.load(b_cur1)
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
        base_row = block_m * BM + wave_m_offset
        base_col = block_n * BN + wave_n_offset

        # Close the half-wave barrier the prologue opened (waves 4-7 took one
        # extra s_barrier there).
        if wave_m == 0:
            rocdl.s_barrier()

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

        # ---- last finisher: count the tile, the completing block posts ----
        if const_expr(fused):
            wait_barrier(0)
            ctr_addr = (
                fx.Int64(wbase)
                + fx.Int64(rank) * w.stride
                + fx.Int64(CTR_OFF)
                + fx.Int64(d * MAXC + c) * fx.Int64(4)
            )
            bctr_addr = fx.Int64(wbase) + fx.Int64(rank) * w.stride + fx.Int64(BCTR_OFF)
            ctr_base = fx.Int32(local_load_u32(signal_ptr(bctr_addr)))
            put_dst = fx.Int64(R_MINE) + fx.Int64(c) * fx.Int64(CHUNK_BYTES)
            put_src = (
                fx.Int64(Y_OFF)
                + fx.Int64(d) * fx.Int64(BAND_BYTES)
                + fx.Int64(c) * fx.Int64(CHUNK_BYTES)
            )
            qid = c % queues
            sdma = cco.DevComm(dev_comm).sdma()
            if fx.thread_idx.x == fx.Int32(0):
                if d != fx.Int32(rank):
                    ctr = signal_ptr(ctr_addr)
                    # The counter atomic is the release (G950-SYNC-01).
                    seq = fx.Int32(atomic_add_u32(ctr, 1, ordering=ctr_order)) + 1
                    if seq - ctr_base == fx.Int32(TPU):
                        # last arrival: back to the base (G950-CTR-01)
                        atomic_add_u32(ctr, fx.Int32(0) - fx.Int32(TPU))
                        if const_expr(emit_put):
                            sdma.put(
                                d,
                                win,
                                put_dst,
                                win,
                                put_src,
                                fx.Int64(CHUNK_BYTES),
                                qid,
                                coop=cco.CoopScope.THREAD,
                                signal=False,
                            )
            fgpu.barrier()

    def launch(
        A: fx.Tensor,
        B_T: fx.Tensor,
        chunk: fx.Int32,
        dev_comm: Int64,
        win: Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel_gemm(
            A,
            B_T,
            chunk,
            dev_comm,
            win,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(grid=(GRID, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return _named(launch, f"launch_{kname}")


# ---------------------------------------------------------------------------
# Tail
# ---------------------------------------------------------------------------


def compile_put_chunk(shape: L.Shape, lay: L.Layout, rank: int):
    """Lane ``p``: chunk ``chunk`` of band ``p`` -> peer ``p``'s ``R[rank]``, queue 0."""
    W = shape.world
    Y_OFF = lay.y_off
    R_MINE = lay.r_off + rank * lay.slot_bytes
    BB, CB = shape.band_bytes, shape.chunk_bytes
    kname = f"mori_garv2_put_mp{shape.mp}_c{shape.chunks}_w{W}_r{rank}"

    @flyc.kernel(name=kname, known_block_size=[TAIL_THREADS, 1, 1])
    def put_chunk(dev_comm: Int64, win: Int64, chunk: fx.Int32):
        tid = fx.thread_idx.x
        sdma = cco.DevComm(dev_comm).sdma()
        dst = fx.Int64(R_MINE) + fx.Int64(chunk) * fx.Int64(CB)
        src = fx.Int64(Y_OFF) + fx.Int64(tid) * fx.Int64(BB) + fx.Int64(chunk) * fx.Int64(CB)
        if tid < fx.Int32(W):
            if tid != fx.Int32(rank):
                sdma.put(
                    tid,
                    win,
                    dst,
                    win,
                    src,
                    fx.Int64(CB),
                    fx.Int32(0),
                    coop=cco.CoopScope.THREAD,
                    signal=False,
                )

    def launch(dev_comm: Int64, win: Int64, chunk: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        put_chunk(dev_comm, win, chunk).launch(
            grid=(1, 1, 1), block=(TAIL_THREADS, 1, 1), stream=stream
        )

    return _named(launch, f"launch_{kname}")


def compile_barrier(
    shape: L.Shape,
    lay: L.Layout,
    rank: int,
    *,
    kind: str,
    quiet_all: bool = False,
    wait: bool = True,
):
    """``kind="scatter"``: drain the scatter puts, then the S barrier.
    ``kind="gather"``: push band ``rank`` of ``Y`` to every peer, drain, then the
    G barrier. Each barrier has its own epoch word, advanced here only."""
    assert kind in ("scatter", "gather")
    W = shape.world
    push = kind == "gather"
    EP_OFF = L.FS_OFF if kind == "scatter" else L.FG_OFF
    SLOT_OFF = L.S_OFF if kind == "scatter" else L.G_OFF
    BAND_SRC = lay.y_off + rank * shape.band_bytes
    BB = shape.band_bytes
    kname = (
        f"mori_garv2_{kind}_mp{shape.mp}_c{shape.chunks}"
        f"{'A' if quiet_all else 'Q'}{'w' if wait else 'n'}_w{W}_r{rank}"
    )

    @flyc.kernel(name=kname, known_block_size=[TAIL_THREADS, 1, 1])
    def bar(dev_comm: Int64, win: Int64):
        tid = fx.thread_idx.x
        w = cco.CachedWindow(win)
        wbase, wstride = w.base, w.stride
        sdma = cco.DevComm(dev_comm).sdma()
        ep_ptr_addr = fx.Int64(wbase) + fx.Int64(rank) * wstride + fx.Int64(EP_OFF)
        e = fx.Int32(local_load_u32(signal_ptr(ep_ptr_addr))) + 1
        peer_slot = fx.Int64(wbase) + fx.Int64(tid) * wstride + fx.Int64(SLOT_OFF + rank * 4)
        my_slot = (
            fx.Int64(wbase)
            + fx.Int64(rank) * wstride
            + fx.Int64(SLOT_OFF)
            + fx.Int64(tid) * fx.Int64(4)
        )
        if tid < fx.Int32(W):
            if tid != fx.Int32(rank):
                if const_expr(push):
                    sdma.put(
                        tid,
                        win,
                        fx.Int64(BAND_SRC),
                        win,
                        fx.Int64(BAND_SRC),
                        fx.Int64(BB),
                        fx.Int32(0),
                        coop=cco.CoopScope.THREAD,
                        signal=False,
                    )
                if const_expr(quiet_all):
                    sdma.quiet(tid, coop=cco.CoopScope.THREAD)
                else:
                    sdma.quiet_queue(tid, fx.Int32(0))
            # Keep the arrival store after the drain.
            raw_cco.cco_system_fence(fx.Int32(0))
            if const_expr(wait):
                signal_store_u32(signal_ptr(peer_slot), e)
                spin_until(signal_ptr(my_slot), e)
        fgpu.barrier()
        if tid == fx.Int32(0):
            local_store_u32(signal_ptr(ep_ptr_addr), e)

    def launch(dev_comm: Int64, win: Int64, stream: fx.Stream = fx.Stream(None)):
        bar(dev_comm, win).launch(grid=(1, 1, 1), block=(TAIL_THREADS, 1, 1), stream=stream)

    return _named(launch, f"launch_{kname}")


def compile_reduce(shape: L.Shape, lay: L.Layout, rank: int):
    """Band ``rank`` of ``Y`` = sum over sources ``s`` in rank order of their
    partial (``Y`` itself for ``s == rank``, ``R[s]`` otherwise), fp32, one bf16
    rounding, stored write-through. Loads of ``R`` bypass L2: a peer's copy
    engine wrote them (``mori.cco.device.flydsl.buffer``, ``CM_SC1``)."""
    W = shape.world
    BB = shape.band_bytes
    vecs = BB // 16
    per_block = THREADS * REDUCE_UNROLL
    GRID = -(-vecs // per_block)
    Y_BAND = lay.y_off + rank * BB
    R_OFF, SLOT = lay.r_off, lay.slot_bytes
    kname = f"mori_garv2_reduce_mp{shape.mp}_u{REDUCE_UNROLL}_w{W}_r{rank}"

    @flyc.kernel(name=kname, known_block_size=[THREADS, 1, 1])
    def reduce(win: Int64):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        w = cco.CachedWindow(win)
        mine = fx.Int64(w.base) + fx.Int64(rank) * w.stride
        rs = []
        for s in range_constexpr(W):
            off = Y_BAND if s == rank else R_OFF + s * SLOT
            rs.append(
                create_buffer_resource_from_addr(
                    wave_uniform_i64(mine + fx.Int64(off)), num_records_bytes=BB
                )
            )
        for u in range_constexpr(REDUCE_UNROLL):
            v = (bid * REDUCE_UNROLL + u) * THREADS + tid
            off = v * 4  # in i32 elements
            acc = None
            for s in range_constexpr(W):
                raw = buffer_load(
                    rs[s],
                    off,
                    vec_width=4,
                    dtype=fx.Int32,
                    cache_modifier=CM_CACHED if s == rank else CM_SC0_SC1,
                )
                x = Vec(raw).bitcast(fx.BFloat16)
                xs = [x[k].to(fx.Float32) for k in range_constexpr(8)]
                acc = xs if acc is None else [acc[k] + xs[k] for k in range_constexpr(8)]
            out = Vec.from_elements(
                [acc[k].to(fx.BFloat16) for k in range_constexpr(8)], fx.BFloat16
            ).bitcast(fx.Int32)
            buffer_store(out, rs[rank], off, cache_modifier=CM_SC0_SC1)

    def launch(win: Int64, stream: fx.Stream = fx.Stream(None)):
        reduce(win).launch(grid=(GRID, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return _named(launch, f"launch_{kname}")
