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
"""The BF16 GEMM with the all-reduce's scatter fused into its epilogue.

``kernels_fused.py``'s fused GEMM is fp8 in: every quant mode it compiles
(ptpc, blockscale, mxfp8) assumes the weight came with fp8 scales. A layer the
checkpoint keeps in BF16 -- Kimi-K3 quantises only its routed experts, so its
attention ``o_proj`` (``[M, 1536] x [7168, 1536]`` at TP8) and shared-expert
``down_proj`` (K/rank 768) are BF16 -- has nothing for it to compile against,
and quantising those weights to reach it is a model-accuracy question rather
than a kernel one. This builds the same operator at the checkpoint's precision.

Two halves, each reused rather than rewritten:

* **The mainloop** is ``_gemm_a16w16_8wave``'s: the same opus quad-subtile
  template as the fp8 kernel, with ``v_mfma_f32_16x16x32_bf16`` (``E_K = 2``),
  ``BLOCK_K = 64`` and no scales. Its staggered pipeline and the two LDS
  ordering edges it needs are carried over verbatim from the kernel those
  blocks were validated in.
* **The tile order and the epilogue tail** are ``kernels_fused.py``'s: the
  destination-rotated chunk-major walk, one monotonic counter per
  ``(dest, chunk)``, the ``acq_rel`` atomic as the release, and the elected
  block's locked ``sdma.put`` from this rank's input region into the peer's
  recv slot. Nothing there depends on the GEMM's precision -- C is BF16 either
  way -- so the reduce and gather phases (``build_sdma_phases``) are unchanged.

Only the default fused-SDMA shape is supported: no ``chunk_bands``, no
``relaxed_rows``, no FP8 scatter, no LSA transport. Those are fp8-path
experiments; port them when a BF16 caller needs them.

As in ``kernels_fused.py``, the kernel body is a pinned copy rather than a
shared helper: FlyDSL only rewrites ``if`` inside the ``@flyc.kernel``
function's own AST, so the runtime ``if wave_m == 1`` and the election cannot
move into one. A fix to the mainloop belongs in ``_gemm_a16w16_8wave``'s users
too, and a fix to the tail in ``kernels_fused.py``.
"""

# NOTE: no `from __future__ import annotations`, for the reason
# _gemm_a8w8_8wave.py states: `fx.struct` reads field annotations as live objects.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int64

import mori.cco.device.flydsl as cco

from ._compat import atomic_add_u32, atomic_store_u32, signal_ptr
from ._gemm_a8w8_8wave import (
    G2SLoader,
    _xcd_swizzle_any,
    ceildiv,
    split_row_major_2d,
    wait_barrier,
)
from ._gemm_a16w16_8wave import (
    BLOCK_K,
    AgStoreC,
    MfmaBf16,
    S2RLoaderBf16,
    _wait_lds_reads,
    compute_global_swizzle_bf16,
    make_bf16_buffer_tensor,
    tile_constants,
)
from .kernels_fused import _acquire_peer_lock, _SwappedMfma


def compile_fused_gemm_scatter_bf16(
    cfg,
    rank: int,
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    sdma_queues: int = 1,
    waves_per_eu: int = 2,
    xcd_swizzle: int = 0,
    fuse: bool = True,
    rotated: bool | None = None,
    n_stripe: int | None = None,
    emit_put: bool = True,
    atomic_order: str = "acq_rel",
    wait_policy: str = "tuned",
):
    """Compile the BF16 GEMM, with (``fuse=True``) or without the scatter epilogue.

    Returns ``launch(A, B_T, C, A_scale, B_scale, c_m, c_n, dev_comm, win,
    stream=...)`` -- the fp8 kernel's signature, so a caller switches precision
    without changing its launch. ``A`` and ``B_T`` are flat **int16** views of
    BF16 ``[M, K]`` and ``[N, K]`` (B row-major, not preshuffled); ``A_scale``
    and ``B_scale`` are accepted and unread.

    ``fuse=False`` is the identical kernel minus the epilogue tail, which is
    the split baseline and the GEMM-only control: validate it before trusting a
    fused number, since a GEMM bug and a transport bug otherwise look the same.
    """
    if wait_policy not in ("tuned", "safe", "conservative"):
        raise ValueError(f"wait_policy must be tuned, safe or conservative, got {wait_policy!r}")
    if atomic_order not in ("acq_rel", "acquire", "release", "monotonic"):
        raise ValueError(f"invalid atomic_order {atomic_order!r}")
    if K % BLOCK_K:
        raise ValueError(f"K={K} must be a multiple of BLOCK_K={BLOCK_K}")
    if cfg.fp8_scatter or getattr(cfg, "compact_recv", False):
        raise ValueError("the BF16 fused GEMM supports only the plain BF16 scatter")
    if cfg.elem_bytes != 2:
        raise ValueError(f"the BF16 GEMM stores BF16 C; cfg.elem_bytes={cfg.elem_bytes}")
    ws = cfg.world_size
    M, N = cfg.m, cfg.n
    if N % BLOCK_N:
        raise ValueError(f"N={N} must be a multiple of BLOCK_N={BLOCK_N}")
    tc = tile_constants(BLOCK_M, BLOCK_N, "bf16")
    rotated = fuse if rotated is None else rotated
    if n_stripe is None or n_stripe == 0:
        n_stripe = N // BLOCK_N
    if n_stripe < 1 or (N // BLOCK_N) % n_stripe:
        raise ValueError(f"n_stripe={n_stripe} must divide the {N // BLOCK_N} N-tiles")
    if n_stripe < N // BLOCK_N and not rotated:
        raise ValueError("a striped tile order needs rotated=True")
    slice_rows = M // ws
    if rotated and slice_rows % BLOCK_M:
        raise ValueError(
            f"each peer's row slice ({slice_rows}) must be a whole number of "
            f"BLOCK_M={BLOCK_M} tiles (M must be a multiple of {ws * BLOCK_M})"
        )

    N_TILES_A, N_TILES_B = tc["N_TILES_A"], tc["N_TILES_B"]
    N_ACCUMS = tc["N_ACCUMS"]
    LDS_BLOCK_M, LDS_BLOCK_N = tc["LDS_BLOCK_M"], tc["LDS_BLOCK_N"]
    N_LDS_STEPS_A, N_LDS_STEPS_B = tc["N_LDS_STEPS_A"], tc["N_LDS_STEPS_B"]
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    a_lds_size, b_lds_size = tc["a_lds_size"], tc["b_lds_size"]
    K_ITERS = K // BLOCK_K
    if K_ITERS < 2:
        raise ValueError("the pipeline peels two K steps off the tail; K >= 2*BLOCK_K")

    # Compile-time tile accounting for the completion counters (kernels_fused.py).
    n_blocks_const = N // BLOCK_N
    m_tiles_per_peer = slice_rows // BLOCK_M if slice_rows % BLOCK_M == 0 else 0
    chunks = cfg.counter_chunks if fuse else 1
    if fuse and (not m_tiles_per_peer or m_tiles_per_peer % chunks):
        raise ValueError(
            f"counter_chunks={chunks} must divide the {m_tiles_per_peer} "
            f"BLOCK_M={BLOCK_M} row-bands in each peer's slice"
        )
    m_tiles_per_chunk = max(1, m_tiles_per_peer // chunks) if fuse else 1
    tiles_per_chunk = m_tiles_per_chunk * n_blocks_const
    chunk_bytes = m_tiles_per_chunk * BLOCK_M * N * 2
    counter_off, lock_off = cfg.counter_off, cfg.lock_off
    in_off = cfg.input_off
    my_recv_slot = cfg.recv_slot_off(rank) if cfg.recv_slots else 0
    slice_bytes = cfg.scatter_slice_bytes

    # The bf16 mainloop's waits (_gemm_a16w16_8wave's users carry the derivation).
    def _wb(tuned):
        if wait_policy == "conservative":
            return wait_barrier(0)
        return wait_barrier(tuned)

    _MAIN_WAIT = (
        N_LDS_STEPS_A + 2 * N_LDS_STEPS_B
        if wait_policy in ("safe", "conservative")
        else min(2 * N_LDS_STEPS_A + N_LDS_STEPS_B, N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)
    )

    _kname = (
        f"mori_fused_{'sdma' if fuse else 'split'}_bf16_8w_"
        f"{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_k{K}_c{chunks}{'r' if rotated else 'l'}"
        f"q{sdma_queues}{'' if n_stripe == n_blocks_const else f's{n_stripe}'}"
        f"x{xcd_swizzle}w{wait_policy[0]}{'p' if emit_put else 'x'}{atomic_order[0]}_r{rank}"
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
    def kernel_gemm_scatter_bf16(
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

        # ---- tile order: kernels_fused.py's ----
        if const_expr(rotated):
            # Destination-rotated, chunk-major: walk (chunk, dest, n) so every
            # link starts streaming in the first 1/(chunks*world) of the GEMM.
            if const_expr(n_stripe >= n_blocks_const):
                rest, bn = split_row_major_2d(fx.block_idx.x, n_blocks)
                tile_i, dest_seq = split_row_major_2d(rest, ws)
            else:
                rest, n_in = split_row_major_2d(fx.block_idx.x, n_stripe)
                rest, dest_seq = split_row_major_2d(rest, ws)
                tile_i, n_grp = split_row_major_2d(rest, n_blocks // n_stripe)
                bn = n_grp * fx.Int32(n_stripe) + n_in
            dest_i = (dest_seq + fx.Int32(rank)) % fx.Int32(ws)
            block_m = dest_i * m_tiles_per_peer + tile_i
            block_n = bn
        elif const_expr(xcd_swizzle > 0):
            block_m, block_n = _xcd_swizzle_any(
                ceildiv(c_m, BLOCK_M), n_blocks, wgm=xcd_swizzle
            )
        else:
            block_m, block_n = split_row_major_2d(fx.block_idx.x, n_blocks)

        # ---- mainloop: the bf16 quad-subtile pipeline ----
        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B0_gl_offset = (block_n * BLOCK_N) * K
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K

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
        # C is this rank's input region of the window (the host points it
        # there), which is what the elected put reads from.
        store_c = AgStoreC(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, out_dtype="bf16")

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()

        _wb(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K)

        _wb(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        for k in range_constexpr(K_ITERS - 2):
            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            # Finish B0/A0's LDS reads before releasing the rendezvous the
            # leading wave group refills B0 after (see _gemm_a16w16_8wave).
            _wait_lds_reads()
            rocdl.s_barrier()

            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K)
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
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
        # Retire the final tile before either wave group reads it below.
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

        base_row = block_m * BLOCK_M + wave_m * (N_TILES_A * 16)
        base_col = block_n * BLOCK_N + wave_n * (N_TILES_B * 16)

        # Close the half-wave barrier pair the prologue opened, so wait_barrier
        # below means every wave's C tile has retired (kernels_fused.py).
        if wave_m == 0:
            rocdl.s_barrier()

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

        # ---- epilogue tail: kernels_fused.py's, BF16 scatter only ----
        if const_expr(fuse):
            # Retire every lane's stores and agree block-wide; the acq_rel
            # counter atomic below is the release (no fence: measured there).
            wait_barrier(0)
            ctr_base = fx.Int64(w_pre.lsa_ptr(rank, counter_off))
            lock_base = fx.Int64(w_pre.lsa_ptr(rank, lock_off))
            sdma = cco.DevComm(dev_comm).sdma()
            if fx.thread_idx.x == 0:
                dest = block_m // fx.Int32(m_tiles_per_peer)
                chunk = (block_m % fx.Int32(m_tiles_per_peer)) // fx.Int32(m_tiles_per_chunk)
                slot = dest * fx.Int32(chunks) + chunk
                ctr = signal_ptr(ctr_base + fx.Int64(slot) * fx.Int64(4))
                seq = fx.Int32(atomic_add_u32(ctr, 1, ordering=atomic_order)) + fx.Int32(1)
                # Counters are never reset: "last tile of this chunk, this
                # epoch" is a modulo test, which keeps graph replay correct.
                if seq % fx.Int32(tiles_per_chunk) == fx.Int32(0):
                    if const_expr(emit_put) and dest != fx.Int32(rank):
                        off = fx.Int64(chunk) * fx.Int64(chunk_bytes)
                        lock = signal_ptr(lock_base + fx.Int64(dest) * fx.Int64(4))
                        if const_expr(chunks > 1):
                            _acquire_peer_lock(lock)
                        sdma.put(
                            dest,
                            win,
                            fx.Int64(my_recv_slot) + off,
                            win,
                            fx.Int64(in_off) + fx.Int64(dest) * fx.Int64(slice_bytes) + off,
                            fx.Int64(chunk_bytes),
                            dest % fx.Int32(sdma_queues),
                            coop=cco.CoopScope.THREAD,
                            signal=False,
                        )
                        if const_expr(chunks > 1):
                            atomic_store_u32(lock, 0)
            # gcnasm closes its ChunkFused epilogue with a barrier here; removing
            # it deadlocked there.
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
        kernel_gemm_scatter_bf16(
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
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch


__all__ = ["compile_fused_gemm_scatter_bf16"]
