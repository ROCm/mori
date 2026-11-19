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
"""``gemm_ag``'s BF16 GEMM: the factory and the tile rules.

The pipeline's building blocks -- LDS swizzle, S2R loader, MFMA, the C store --
live in ``gemm_ar/_gemm_a16w16_8wave.py``, shared with ``gemm_ar``'s BF16 GEMM +
all-reduce, and its docstring is the account of the port. What is here is
``compile_bf16_gemm_ag`` -- the GEMM with the all-gather's epilogue fused in --
and ``pick_tile`` / ``pick_chunks``, the measured per-shape tile and chunk
rules.
"""

# NOTE: no `from __future__ import annotations`, deliberately, for the reason
# _gemm_a8w8_8wave.py states: `fx.struct` reads the `SharedStorage` field
# annotations as live objects, and PEP 563 would hand it a string whose size
# operands are locals of this factory.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
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

# G2SLoader, wait_barrier, ceildiv, split_row_major_2d and _xcd_swizzle_any are
# gemm_ar's fp8 pipeline's, taken directly: all of them are byte-indexed or
# shape-only, so none carries an fp8 assumption.
from ..gemm_ar._gemm_a8w8_8wave import (
    G2SLoader,
    _xcd_swizzle_any,
    ceildiv,
    split_row_major_2d,
    wait_barrier,
)
from ..gemm_ar._gemm_a16w16_8wave import (
    BLOCK_K,
    E_K,
    VEC,
    AgStoreC,
    MfmaBf16,
    S2RLoaderBf16,
    _wait_lds_reads,
    compute_global_swizzle_bf16,
    make_bf16_buffer_tensor,
    swizzle_row128b,
    tile_constants,
)
from ..gemm_ar.kernels_fused import (
    _acquire_peer_lock,
    _SwappedMfma,
)
from ._credit import call_epoch, lsa, post_credit, wait_credit
from .kernels_sdma import _spin_until


def compile_bf16_gemm_ag(
    cfg,
    rank: int,
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    out_dtype: str = "bf16",
    waves_per_eu: int = 2,
    xcd_swizzle: int = 0,
    wait_policy: str = "tuned",
    fuse: bool = False,
    chunks: int = 1,
    sdma_queues: int = 1,
    post: str = "lanes",
    peer_uncached: bool = False,
    emit_put: bool = True,
    fence: str = "leader",
    war_credit: bool = True,
    split_k: int = 1,
):
    """The bf16 GEMM, with or without the all-gather fused into its epilogue.

    Returns ``launch(A, B_T, C, c_m, c_n, dev_comm, win, stream=...)``. ``A``
    and ``B_T`` are flat **int16** views of bf16 ``[M, K]`` and ``[N, K]`` --
    see ``make_bf16_buffer_tensor`` for why int16 rather than int8. There are no
    scale arguments, unlike ``compile_fp8_gemm_8w``: nothing to dequantise.

    ``fuse=False`` emits the identical kernel minus the epilogue tail and writes
    C wherever the caller points it -- a plain tensor for ``gemm-only``, this
    rank's ``recv`` slot for ``gemm-to-window`` and ``split-sdma``. That
    is the same discipline ``gemm_ar.compile_fused_gemm_scatter`` uses and for
    the same reason: the split baseline and the fused kernel then differ in
    exactly one thing. ``dev_comm`` and ``win`` are unread on that path but stay
    in the signature so the launch path does not change either.

    ``split_k=2/4/8`` partitions K across the grid's second dimension and
    writes FP32 partials to contiguous ``[split_k, M, N]`` storage. A and B
    retain their original row stride K. The caller reduces these partials
    before consuming or communicating C; split-K requires local, unfused
    FP32 output.

    ``fence="release"`` retains system-scope publication while omitting the
    acquire half of the fused SDMA producer fence. Together with
    ``peer_uncached=True`` it selects the measured publication optimization.

    **The fused epilogue below is the twin of the fp8 one in ``kernels_fused.py``
    and the two must be changed together.** They are not shared because FlyDSL's
    rewriter only lowers ``if`` inside a ``@flyc.kernel`` function's own AST, so
    the election and the barriers cannot move into a helper. What differs
    between them is the mainloop and the store; the transport logic -- release
    fence, per-(chunk, lane) counter, lane-parallel put, recv-reuse credit
    (``war_credit``, see ``_credit``) -- is meant to be identical, and any fix
    to one belongs in both.

    Validate ``fuse=False`` before trusting anything fused: a wrong LDS swizzle
    or a wrong ``E_K`` unroll produces a wrong C, and every fused mode validates
    what *arrived*, so without that control a GEMM bug and a transport bug are
    indistinguishable.
    """
    assert K % BLOCK_K == 0, f"K={K} must be a multiple of BLOCK_K={BLOCK_K}"
    if split_k not in (1, 2, 4, 8):
        raise ValueError("split_k must be 1, 2, 4 or 8")
    if split_k > 1 and (fuse or peer_uncached or out_dtype != "fp32"):
        raise ValueError("split-K requires local, unfused FP32 partial output")
    if K % (split_k * BLOCK_K):
        raise ValueError("K must be divisible by split_k * BLOCK_K")
    if wait_policy not in ("tuned", "safe", "conservative"):
        raise ValueError(
            f"wait_policy must be tuned, safe or conservative, got {wait_policy!r}"
        )
    if peer_uncached and not fuse:
        raise ValueError(
            "peer_uncached selects the fused path's publication; the unfused "
            "GEMM's C is read by nothing that needs it"
        )
    if post not in ("lanes", "serial"):
        raise ValueError(f"post must be lanes or serial, got {post!r}")
    if fence not in ("none", "leader", "all", "release"):
        raise ValueError(f"invalid producer fence: {fence!r}")
    if fuse and post == "lanes" and chunks > 1 and sdma_queues < chunks:
        raise ValueError(
            f"post='lanes' with chunks={chunks} needs sdma_queues >= {chunks}, "
            f"got {sdma_queues}: the lane-parallel path has no submit lock, so "
            f"two chunks of one destination are separated by queue instead"
        )
    if cfg.elem_bytes != (2 if out_dtype == "bf16" else 4):
        raise ValueError(
            f"out_dtype={out_dtype!r} needs cfg.elem_bytes="
            f"{2 if out_dtype == 'bf16' else 4}, got {cfg.elem_bytes}: the "
            f"window offsets and the C store have to agree on the element width"
        )
    cfg.validate()
    if fuse and (BLOCK_M, BLOCK_N) != (cfg.block_m, cfg.block_n):
        # tiles_per_chunk counts this GEMM's N tiles through cfg.n_blocks, so a
        # config describing a different tile makes the completion counter a
        # number the epilogue never reaches -- a hang, not a wrong answer.
        raise ValueError(
            f"tile ({BLOCK_M}, {BLOCK_N}) disagrees with the config's "
            f"({cfg.block_m}, {cfg.block_n}); the fused epilogue's completion "
            f"counter is derived from the config's"
        )
    ws = cfg.world_size
    N = cfg.n
    my_recv_slot = cfg.recv_slot_off(rank)
    slab_bytes = cfg.slab_bytes
    counter_off, lock_off = cfg.counter_off, cfg.lock_off
    credit_off, flag_off = cfg.credit_off, cfg.flag_off
    # The fused path pushes into the peers' recv from here, so it carries the
    # recv-reuse credit (see _credit); the unfused GEMM pushes nothing.
    credit = war_credit and fuse
    if fuse and cfg.counter_chunks != chunks:
        raise ValueError(
            f"chunks={chunks} disagrees with the config's counter_chunks="
            f"{cfg.counter_chunks}"
        )
    m_tiles_per_chunk = cfg.m_tiles // chunks
    tiles_per_chunk = m_tiles_per_chunk * cfg.n_blocks
    chunk_bytes = m_tiles_per_chunk * BLOCK_M * N * cfg.elem_bytes
    tc = tile_constants(BLOCK_M, BLOCK_N, out_dtype)
    N_TILES_A, N_TILES_B = tc["N_TILES_A"], tc["N_TILES_B"]
    N_ACCUMS = tc["N_ACCUMS"]
    LDS_BLOCK_M, LDS_BLOCK_N = tc["LDS_BLOCK_M"], tc["LDS_BLOCK_N"]
    N_LDS_STEPS_A, N_LDS_STEPS_B = tc["N_LDS_STEPS_A"], tc["N_LDS_STEPS_B"]
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    a_lds_size, b_lds_size = tc["a_lds_size"], tc["b_lds_size"]
    K_ITERS = K // (split_k * BLOCK_K)
    assert K_ITERS >= 2, "the pipeline peels two K steps off the tail"

    # G2S uses vmcnt; S2R uses lgkmcnt. With A/B load instructions per
    # subtile, the prefetches issued AFTER A1@k+1 are B0, A0, B1@k+2:
    # A+2B instructions may remain when the next iteration starts. Preserve
    # the tighter inherited threshold where it is safe, but cap it for the
    # asymmetric 256x128 tile (2A+B=5 would leave A1 unfinished; A+2B=4).
    # These counts only protect read-after-write. The staggered waves also
    # need _wait_lds_reads() below to protect B0 from an early overwrite.
    # "conservative" remains a diagnostic that drains G2S at every wait.
    def _wb(tuned):
        if wait_policy == "conservative":
            return wait_barrier(0)
        return wait_barrier(tuned)

    _MAIN_WAIT = (
        N_LDS_STEPS_A + 2 * N_LDS_STEPS_B
        if wait_policy in ("safe", "conservative")
        else min(
            2 * N_LDS_STEPS_A + N_LDS_STEPS_B,
            N_LDS_STEPS_A + 2 * N_LDS_STEPS_B,
        )
    )

    _kname = (
        f"mori_ag_bf16_{'sdma' if fuse else 'plain'}_8w_sync2_pub2_"
        f"{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_"
        f"{'F32' if out_dtype == 'fp32' else 'B16'}_{waves_per_eu}x{xcd_swizzle}"
        f"_c{chunks}q{sdma_queues}{post[0]}{'U' if peer_uncached else 'C'}"
        f"w{wait_policy[0]}"
        f"{fence[0]}{'p' if emit_put else 'x'}_k{K}_sk{split_k}"
        f"{'' if credit or not fuse else '_nc'}_r{rank}"
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

    @flyc.kernel(name=_kname, known_block_size=[512, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        dev_comm: Int64,
        win: Int64,
    ):
        n_blocks = ceildiv(c_n, BLOCK_N)
        w_pre = cco.CachedWindow(win)
        # Bases out here: the credit helpers run inside runtime `if`s.
        wbase, wstride = w_pre.base, w_pre.stride
        call_e = my_credits = None
        if const_expr(credit):
            call_e = call_epoch(wbase, wstride, rank, flag_off)
            my_credits = lsa(wbase, wstride, rank, credit_off)
            # Block 0's first world lanes, before any GEMM work: block 0 is in
            # the first wave, so a peer waiting on this never waits on a block
            # that has not been scheduled.
            if fx.block_idx.x == 0:
                ctid = fx.thread_idx.x
                if ctid < fx.Int32(ws) and ctid != fx.Int32(rank):
                    post_credit(wbase, wstride, rank, ctid, credit_off, call_e)

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
            block_m, block_n = split_row_major_2d(fx.block_idx.x, n_blocks)

        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        # No preshuffle, so B advances by BLOCK_K like A. gemm_ar's fp8 path
        # has a 2*1024 stride here because its B is preshuffled.
        B_K_STEP = BLOCK_K
        B0_gl_offset = (block_n * BLOCK_N) * K
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K
        if const_expr(split_k > 1):
            split_offset = fx.block_idx.y * fx.Int32(K // split_k)
            A0_gl_offset += split_offset
            A1_gl_offset += split_offset
            B0_gl_offset += split_offset
            B1_gl_offset += split_offset

        gA = make_bf16_buffer_tensor(A)
        gB = make_bf16_buffer_tensor(B_T)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS)
        gl_off_b = compute_global_swizzle_bf16(lane_id, wave_id, K, N_LDS_ROUNDS)

        mfma = _SwappedMfma(MfmaBf16(N_TILES_B, N_TILES_A))

        BF16_IR_t = fx.BFloat16.ir_type
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, BF16_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, BF16_IR_t, wave_id)
        a_s2r = S2RLoaderBf16(wave_m, N_TILES_A)
        b_s2r = S2RLoaderBf16(wave_n, N_TILES_B)
        # Where C goes. The fused output is anchored to this rank's registered
        # recv slot, which SDMA reads independently of the caller's C view;
        # everywhere else the store goes through C.
        if const_expr(fuse):
            own_slot = create_buffer_resource_from_addr(
                wave_uniform_i64(w_pre.lsa_ptr(rank, my_recv_slot)),
                num_records_bytes=slab_bytes,
            )
        else:
            own_slot = None
        store_c = AgStoreC(
            C,
            c_m * split_k,
            c_n,
            mfma.idx,
            N_TILES_A,
            N_TILES_B,
            out_dtype=out_dtype,
            peer_rsrc=own_slot,
            peer_uncached=peer_uncached,
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

        _wb(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * B_K_STEP)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * B_K_STEP)

        _wb(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        for k in range_constexpr(K_ITERS - 2):
            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            # Waves 0-3 are one barrier ahead of waves 4-7. When the latter
            # reach this barrier, the former can already issue B0@k+2 below.
            # Waiting only inside mfma.call is too late: finish B0's LDS reads
            # BEFORE releasing that rendezvous. Drain A0 too so this does not
            # depend on LLVM's ordering of the two sets of ds_read instructions.
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
            _wb(_MAIN_WAIT)

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
        # The mainloop prefetches a_next1 one step behind; issue the final
        # K_ITERS - 1 tile here or c10 / c11 read stale A1. gemm_ar carries the
        # same line with the same note.
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
        # No mainloop wait follows this last prefetch. Retire the final tile
        # before either wave group reads b_next0 / a_next0 below. The barrier
        # after c10 pairs with this wait in the lagging group. Waiting after
        # the final a0 S2R load cannot repair a value already read from LDS.
        wait_barrier(0)

        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)

        b0_frag = b_s2r.load(b_next0)
        rocdl.s_barrier()

        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 1; the prefetches were drained above.
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
        base_row = block_m * BLOCK_M + wave_m_offset
        if const_expr(split_k > 1):
            base_row += fx.block_idx.y * c_m
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

        # ---- producer publication and the all-gather epilogue ----
        if const_expr(fuse):
            # Retire this block's stores into its own recv slot, then publish
            # them for the copy engine to read.
            wait_barrier(0)
            if const_expr(fence == "all"):
                raw_cco.cco_system_fence(fx.Int32(0))
            elif const_expr(fence == "leader"):
                # One wave per block, not every lane: every lane costs a
                # whole-L2 writeback each. Legal because the half-wave barrier
                # pair is closed above, so wait_barrier(0) really does mean
                # every wave's stores have retired into this CU's L2.
                raw_cco.cco_system_fence(fx.Int32(1))

            elif const_expr(fence == "release"):
                if fx.thread_idx.x == fx.Int32(0):
                    _llvm.fence(_llvm.AtomicOrdering.release, syncscope="")

            ctr_base = fx.Int64(w_pre.lsa_ptr(rank, counter_off))
            lock_base = fx.Int64(w_pre.lsa_ptr(rank, lock_off))
            sdma = cco.DevComm(dev_comm).sdma()
            chunk = block_m // fx.Int32(m_tiles_per_chunk)
            off = fx.Int64(chunk) * fx.Int64(chunk_bytes)
            # Lane j owns destination (rank+1+j) and counter slot (chunk, j), so
            # each lane learns from its own atomic that its block won and no
            # broadcast is needed. See layout.counter_set_bytes.
            lanes = ws - 1 if post == "lanes" else 1
            if fx.thread_idx.x < fx.Int32(lanes):
                j = fx.thread_idx.x
                slot = chunk * fx.Int32(ws) + j
                ctr = signal_ptr(ctr_base + fx.Int64(slot) * fx.Int64(4))
                seq = fx.Int32(atomic_add_u32(ctr, 1, ordering="acq_rel")) + fx.Int32(1)
                # The chunk's last tile this call; it puts the counter back to
                # zero, as in kernels_fused.py.
                if seq == fx.Int32(tiles_per_chunk):
                    atomic_add_u32(ctr, fx.Int32(0) - fx.Int32(tiles_per_chunk))
                    if const_expr(emit_put):
                        if const_expr(post == "lanes"):
                            d = (fx.Int32(rank + 1) + j) % fx.Int32(ws)
                            if const_expr(credit):
                                wait_credit(my_credits, d, call_e, _spin_until)
                            sdma.put(
                                d,
                                win,
                                fx.Int64(my_recv_slot) + off,
                                win,
                                fx.Int64(my_recv_slot) + off,
                                fx.Int64(chunk_bytes),
                                chunk % fx.Int32(sdma_queues),
                                coop=cco.CoopScope.THREAD,
                                signal=False,
                            )
                        else:
                            if const_expr(credit):
                                # Before the lock, so a wait never holds it.
                                for jj in range_constexpr(ws - 1):
                                    wait_credit(
                                        my_credits,
                                        fx.Int32((rank + 1 + jj) % ws),
                                        call_e,
                                        _spin_until,
                                    )
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
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        dev_comm: Int64,
        win: Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_gemm(
            A,
            B_T,
            C,
            c_m,
            c_n,
            dev_comm,
            win,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(grid=(grid_x, split_k, 1), block=(512, 1, 1), stream=stream)

    return launch_gemm


#: Largest first; at equal area 128x256 before 256x128, which measured the
#: faster GEMM (Pro M=4096: 112.5 against 116.4 us) and the faster fused
#: path (Flash M=16384: 1265.7 against 1298.1 us). The smaller tiles were held
#: back until the two missing LDS ordering edges behind their old intermittent
#: failures were fixed (now covered by the LDS model in test_gemm_ag.py) and they
#: had broader coverage: every tile is validated by test_bf16_gemm_pipeline_numerics
#: and, on 8 ranks, by test_gemm_ag_op.py's bf16_tiles case.
TILE_CANDIDATES = ((256, 256), (128, 256), (256, 128), (128, 128))

#: CUs on the part this was tuned on (MI355X). The grid heuristic wants the
#: real number, so it is a parameter of ``pick_tile`` rather than a constant.
DEFAULT_CUS = 256

#: Workgroup rounds a fused tile needs. A chunk can only overlap its transfer
#: with GEMM work still to come, and a grid that runs in one round completes
#: every chunk at once.
FUSED_MIN_ROUNDS = 2


def pick_tile(m, n, out_dtype="bf16", cus=DEFAULT_CUS, *, fused=False):
    """Choose ``(BLOCK_M, BLOCK_N)`` for a shape. Measured, not guessed.

    **Unfused** (``gemm-only`` and the split paths): the largest tile whose
    grid still covers the CUs, or the largest grid available if none does. The
    template's ``256x256`` launches ``ceil(M/256) * ceil(N/256)`` workgroups --
    64 at ``M=N=2048`` against an MI355X's 256 CUs -- so at small M it idles
    three quarters of the GPU, while at large M it is the fastest tile. GEMM
    alone, BF16 in / FP32 out, 8x MI355X, 2026-10-29 (us)::

        shape           M      torch  128/128  128/256  256/128  256/256
        N=2048 K=7168   2048    71.0   *73.3     91.0     93.0    132.0
                        4096   146.0   120.9   *112.5    116.4    143.4
                        8192   181.2   223.0    211.4    212.2   *185.0
                       16384   365.1   527.1    455.9    465.5   *364.6
        N=1024 K=5120   2048    43.9   *54.0     65.7     68.6     99.2
                       16384   140.0   212.2    156.4    209.8   *138.0

    The rule picks the starred tile at every shape. It is **not** "maximise the
    grid": once there is a workgroup per CU, the larger tile amortises better.

    **Fused** (``fused=True``): the largest tile *other than* ``256x256`` whose
    grid runs at least ``FUSED_MIN_ROUNDS`` rounds, or the largest grid
    available if none does. Fusion overlaps a chunk's transfer with the GEMM
    rounds still to come, so a tile that finishes the whole grid in one round
    leaves nothing to overlap -- ``256x256`` at Pro M=8192 runs 1.0% *behind*
    split at the same tile. And with a chunk per round (``pick_chunks``) the
    first chunk is ready after about GEMM / rounds: ``256x256`` at Pro M=16384
    is the faster GEMM (364.6 against 455.9 us) but two rounds to
    ``128x256``'s four, and measured 1.9% behind it fused. It was the best
    fused tile at none of the three shapes.
    Fused SDMA, ``sc_release``, best chunk count per tile (us)::

        shape           M      split*   128/128  128/256  256/128  256/256
        N=2048 K=7168   8192   1341.6   1314.2  *1289.6   1291.0   1354.9
                       16384   2676.2   2525.8  *2479.2   2479.0   2527.1
        N=1024 K=5120  16384   1294.2   1310.0  *1265.7   1298.1   1307.4

    ``split*`` is split SDMA at the unfused rule's tile. See ``pick_chunks``
    for the chunk count that goes with the fused tile.
    """
    cands = [
        (bm, bn)
        for bm, bn in TILE_CANDIDATES
        if m % bm == 0 and n % bn == 0 and (bn == 256 or out_dtype == "fp32")
    ]
    if not cands:
        raise ValueError(
            f"no validated tile divides m={m} n={n} for out_dtype={out_dtype!r}; "
            f"candidates are {TILE_CANDIDATES} and BLOCK_N=128 is fp32-only"
        )
    if fused and len(cands) > 1:
        cands = [t for t in cands if t != (256, 256)]
    need = cus * (FUSED_MIN_ROUNDS if fused else 1)
    covering = [t for t in cands if tile_grid(m, n, t) >= need]
    if covering:
        # Largest area among those that fill the machine; TILE_CANDIDATES is
        # ordered largest-first so the first match is it.
        return covering[0]
    return max(cands, key=lambda t: tile_grid(m, n, t))


def pick_chunks(m, n, tile, cus=DEFAULT_CUS, max_chunks=8):
    """Fused chunk count for ``tile``: one chunk per workgroup round.

    Measured best at every fused shape in ``pick_tile``'s table: two rounds
    take two chunks (Pro M=8192, Flash M=16384), four take four (Pro
    M=16384, 2479.2 us against 2520.7 at eight). Rounded down to a divisor of
    the row tiles, which the completion counter needs.
    """
    rounds = max(1, tile_grid(m, n, tile) // cus)
    m_tiles = m // tile[0]
    for c in range(min(rounds, max_chunks, m_tiles), 0, -1):
        if m_tiles % c == 0:
            return c
    return 1


def tile_grid(m, n, tile):
    """Workgroups a tile launches. Exposed because it is the whole story."""
    return (-(-m // tile[0])) * (-(-n // tile[1]))


__all__ = [
    "AgStoreC",
    "BLOCK_K",
    "compile_bf16_gemm_ag",
    "pick_chunks",
    "pick_tile",
    "TILE_CANDIDATES",
    "E_K",
    "MfmaBf16",
    "S2RLoaderBf16",
    "VEC",
    "_SwappedMfma",
    "compute_global_swizzle_bf16",
    "make_bf16_buffer_tensor",
    "swizzle_row128b",
    "tile_constants",
]
