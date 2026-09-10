# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Preshuffle GEMM (layout API): f16/bf16/fp8/int8, ping-pong scf.for loop with scheduler hints.

Pinned from ``aiter.ops.flydsl.kernels.preshuffle_gemm`` and extended with
``quant="blockscale"`` -- the A 1x128 / B 128x128 fp32 quantisation DSV4 runs,
which upstream does not have (its fp8 scales are per-row x per-col, applied once
in the epilogue).

Why this kernel and not ``kernels_fused``'s 8-wave one: it is already the shape
CK's fastest blockscale instance uses. 4 waves, 256 threads, and **B never
touches LDS** -- ``thr_g2r_B``/``frag_B_stages`` load it global->VGPR
double-buffered, the way CK's ``blockscale_b_preshuffle_v1`` pipeline does with
``b_thread_bufs``. Only A is staged, so LDS is ``tile_m * tile_k`` x 2 stages
(16 KB at 64x128) against the 8-wave kernel's 96 KB, which caps that one at one
block per CU. B is already preshuffled to ``N0 K0 KLane NLane KPack``, so every
lane pulls its own MFMA operand with one 128-bit load and there is nothing for
LDS to amortise.

Measured at ``[16384, 7168] K=2048`` on MI355X, warm cache, tile 64x256x128 --
CK's own winning shape:

=============================  =======  ========
stage                                us   vs CK
=============================  =======  ========
whole-fragment rescale (divide)  760.2    +153%
per-tile promotion               459.5     +53%
  + explicit fma, scale hoisted  391.7     +30%
  + MFMAs batched per M-row      385.9     +28%
this kernel, scale removed       303.1      +1%
CK 64x256 Intrawave v1           300.4         .
=============================  =======  ========

So the GEMM itself is aligned: strip the scale and this lands on CK. What is
left is the scale, +82.6us against CK's ~0, and the loop body says why. Per
128 accumulator elements:

    CK             128 v_fmac_f32,  8 v_mul_f32      -> VALU/MFMA 5.81
    this           128 v_(pk_)fma, 128 v_(pk_)mul    -> VALU/MFMA 5.53

Our instruction *count* is already under CK's; the shape is not. CK needs one
FMA per element because ``c_scale_thread`` is a single scalar for a whole MFMA
tile, where we need a multiply per element first: A's scale is per row
(ScaleBlockM=1) and an MFMA 16x16 lane holds four different rows, so four
different A scales. CK's four values must instead be four *columns* of one row,
which is what ``kernels_fused``'s ``--swap-ab`` arranges. Closing the last 28%
means adopting that output layout, not more arithmetic.

``swap_ab`` exchanges the MFMA operands so a lane owns four consecutive N of
one row instead of four M. It does exactly what it is supposed to on both
counts CK benefits from -- A's scale collapses to one scalar per tile, so the
promote is a single FMA like ``c_scale_thread``, and the C store goes from 64
``buffer_store_short`` to 16 ``buffer_store_dwordx2``:

============================  =========  ==========  ============
                              VALU/MFMA   C stores    total (us)
============================  =========  ==========  ============
off                                5.53   64 short          385.9
on                                 4.47   16 dwordx2        506.9
============================  =========  ==========  ============

and it is 26% slower anyway, on a register budget. ``--waves-per-eu 2`` caps
VGPRs at 256, and compiling the three parts of the swap separately says where
they go:

===========================  ======  =========
variant                        VGPR   scratch
===========================  ======  =========
off (this kernel's default)      256          8
  exchange MFMA operands         260          8
  scalar A scale                 240          0
  explicit C store               284          8
**operands + scalar scale**  **248**    **0**
all three (swap_ab=True)         256          8
===========================  ======  =========

The scale change *saves* 16 registers and the operand exchange costs 4; both
together are the only variant here that does not spill, and it spills less than
the default does. **The whole cost is the C write-out.** Not its registers
(one shared landing register measures the same as sixteen), not ``frag_C_out``
left allocated, not the conversion order, and not the addressing -- rebuilding
the 16 runtime indices as one base plus compile-time strides took it 284 -> 270
unconstrained and 506.9 -> 485.1us, but under the cap it still spills.

So the missing piece is a store for the transposed accumulator that costs no
registers, i.e. the tiled-copy path (``thr_r2g_C``/``pC_g``) rebuilt against a
transposed C partitioning, rather than written out by hand. Then the measured
248/0 applies and swap_ab is strictly better than the default, which spills.
Left in behind ``swap_ab=False`` until that exists.

Two structural notes that came out of the port. B never touches LDS here --
``thr_g2r_B``/``frag_B_stages`` load it global->VGPR double-buffered, as CK's
``blockscale_b_preshuffle_v1`` does with ``b_thread_bufs`` -- and B is already
preshuffled to ``N0 K0 KLane NLane KPack``, the same layout CK's
``preShuffleBuffer`` emits, so every lane pulls its own MFMA operand with one
128-bit load. And driving MFMAs per accumulator tile beats one ``fx.gemm`` over
the whole fragment even before any scaling: 303.1us against 556.0 at this tile.

For scale, the 8-wave kernel in ``kernels_fused`` is 224.0us unscaled and 370.5
scaled at its own best tile.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, math, range_constexpr, rocdl
from flydsl.expr.typing import (
    BFloat16,
    Float8E4M3FN,
    Float8E4M3FNUZ,
    Float16,
    Float32,
    Int8,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels import buffer_ops, vector
from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import xcd_remap_bx_by
from aiter.ops.flydsl.kernels.splitk_epilogue import (
    CPOL_COHERENT,
    splitk_reduce_epilogue,
)

# (dsrd_preload, dvmem_preload) per (tile_m, tile_n, tile_k).
_TILE_PRELOAD_TABLE = {
    # ── tile_m = 16 ──
    (16, 64, 256): (2, 2),
    (16, 64, 512): (4, 4),
    (16, 128, 256): (2, 2),
    (16, 128, 512): (2, 2),
    (16, 192, 256): (2, 2),
    (16, 256, 256): (2, 2),
    (16, 256, 512): (2, 2),
    (16, 512, 256): (2, 2),
    # ── tile_m = 32 ──
    (32, 64, 128): (6, 6),
    (32, 64, 256): (6, 6),
    (32, 64, 512): (2, 2),
    (32, 128, 128): (6, 6),
    (32, 128, 256): (6, 6),
    (32, 192, 128): (6, 6),
    (32, 192, 256): (6, 6),
    (32, 256, 128): (6, 6),
    (32, 256, 256): (6, 6),
    # ── tile_m = 48 ──
    (48, 64, 128): (8, 8),
    (48, 64, 256): (2, 2),
    (48, 128, 256): (6, 6),
    (48, 192, 256): (6, 6),
    (48, 256, 256): (6, 6),
    # ── tile_m = 64 ──
    (64, 64, 128): (4, 4),
    (64, 64, 256): (4, 4),
    (64, 128, 128): (8, 8),
    (64, 128, 256): (8, 8),
    (64, 192, 128): (8, 8),
    (64, 192, 256): (8, 8),
    (64, 256, 64): (8, 8),
    (64, 256, 128): (8, 8),
    (64, 256, 256): (8, 8),
    # ── tile_m = 80 ──
    (80, 64, 256): (4, 4),
    (80, 128, 256): (8, 8),
    (80, 192, 256): (8, 8),
    (80, 256, 256): (8, 8),
    # ── tile_m = 96 ──
    (96, 64, 128): (6, 6),
    (96, 64, 256): (6, 6),
    (96, 128, 128): (8, 8),
    (96, 128, 256): (6, 6),
    (96, 192, 128): (8, 8),
    (96, 192, 256): (8, 8),
    (96, 256, 128): (8, 8),
    (96, 256, 256): (8, 8),
    # ── tile_m = 112 ──
    (112, 64, 256): (8, 8),
    (112, 128, 256): (4, 4),
    (112, 192, 256): (8, 8),
    (112, 256, 256): (8, 8),
    # ── tile_m = 128 ──
    (128, 64, 128): (6, 6),
    (128, 64, 256): (8, 8),
    (128, 128, 64): (4, 4),
    (128, 128, 128): (8, 8),
    (128, 128, 256): (4, 4),
    (128, 192, 128): (8, 8),
    (128, 192, 256): (8, 8),
    (128, 256, 128): (6, 6),
    (128, 256, 256): (4, 4),
    # ── tile_m = 160 ──
    (160, 192, 128): (8, 8),
    # ── tile_m = 192 ──
    (192, 64, 128): (6, 6),
    (192, 128, 128): (6, 6),
    # ── tile_m = 224 ──
    (224, 64, 128): (4, 4),
    (224, 128, 128): (6, 6),
    (224, 192, 128): (6, 6),
    # ── tile_m = 256 ──
    (256, 64, 128): (4, 4),
    (256, 128, 128): (6, 6),
    (256, 192, 128): (6, 6),
    (256, 256, 128): (4, 4),
}

_TILE_PRELOAD_DEFAULT = (0, 0)


def _get_preload(tile_m, tile_n, tile_k):
    """Look up (dsrd_preload, dvmem_preload) from the tile table."""
    return _TILE_PRELOAD_TABLE.get(
        (int(tile_m), int(tile_n), int(tile_k)), _TILE_PRELOAD_DEFAULT
    )


# Row bound of the A/C layout views; gemm_kernels mirrors it to guard on the host.
PRESHUFFLE_M_MAX = 65536


@functools.lru_cache(maxsize=1024)
def compile_preshuffle_gemm(
    *,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    in_dtype: str = "fp8",
    out_dtype: str = "bf16",
    quant: str = "ptpc",  # "ptpc" (upstream) or "blockscale" (A 1x128, B 128x128)
    swap_ab: bool = False,  # blockscale only; see the module docstring
    epilogue: str = "none",  # "none", "bias", "bias_relu", "bias_silu", "bias_gelu"
    waves_per_eu: int | None = None,
    enable_scheduler: bool = True,
    use_async_copy: bool = False,
    xcd_swizzle: int = 0,
    lds_stage: int = 2,
    split_k: int = 1,
):
    """Compile preshuffle GEMM (fp8/int8/fp16/bf16).
    Signature: fn(C, out, semaphore, A, B, scale_a, scale_b, bias, M, N, stream).
    For split_k > 1, C is an fp32 partial workspace and the last arriving split
    performs the reduction and final conversion inside the same kernel launch.
    """
    if in_dtype not in ("fp8", "int8", "fp16", "bf16"):
        raise ValueError(f"in_dtype must be fp8/int8/fp16/bf16, got {in_dtype!r}")
    if split_k < 1 or K % split_k != 0:
        raise ValueError(f"split_k must divide K; got split_k={split_k}, K={K}")
    split_k_extent = K // split_k
    if tile_k <= 0 or split_k_extent % tile_k != 0:
        raise ValueError(
            "tile_k must be a positive divisor of K/split_k; "
            f"got tile_k={tile_k}, K={K}, split_k={split_k}"
        )
    if epilogue not in ("none", "bias", "bias_relu", "bias_silu", "bias_gelu"):
        raise ValueError(
            f"epilogue must be none/bias/bias_relu/bias_silu/bias_gelu, got {epilogue!r}"
        )
    if lds_stage not in (1, 2):
        raise ValueError(f"lds_stage must be 1 or 2, got {lds_stage}")
    _has_epilogue = epilogue != "none"
    _has_bias = epilogue in ("bias", "bias_relu", "bias_silu", "bias_gelu")
    _has_relu = epilogue == "bias_relu"
    _has_silu = epilogue == "bias_silu"
    _has_gelu = epilogue == "bias_gelu"
    if split_k > 1 and _has_epilogue:
        raise ValueError("split_k > 1 does not support fused bias or activation")

    if quant not in ("ptpc", "blockscale"):
        raise ValueError(f"quant must be ptpc or blockscale, got {quant!r}")
    is_blockscale = quant == "blockscale"
    swap_ab = swap_ab and is_blockscale
    if is_blockscale:
        # tile_k == 128 makes one main-loop tile exactly one K-scale block, so
        # the rescale lands on a loop boundary that already exists.
        if in_dtype != "fp8":
            raise ValueError("blockscale is fp8 only")
        if tile_k % 128:
            raise ValueError(f"blockscale needs tile_k a multiple of 128, got {tile_k}")
        if K % 128 or N % 128:
            raise ValueError(f"blockscale needs K and N multiples of 128, got K={K} N={N}")
        if split_k != 1 or _has_epilogue:
            raise ValueError("blockscale does not support split_k or a fused epilogue")
        # Only the 2-stage pipeline threads the tile index down to mma_kloop,
        # which is what the rescale is indexed by.
        if lds_stage != 2:
            raise ValueError(f"blockscale needs lds_stage=2, got {lds_stage}")

    is_fp8 = in_dtype == "fp8"
    is_int8 = in_dtype == "int8"
    is_f16 = in_dtype == "fp16"
    is_bf16 = in_dtype == "bf16"
    is_f16_or_bf16 = is_f16 or is_bf16
    is_8bit = is_fp8 or is_int8
    elem_bytes = 1 if is_8bit else 2

    # The async gmem->LDS DMA (buffer_load_lds 128b) only lowers for 8-bit inputs.
    if use_async_copy and not is_8bit:
        raise ValueError("use_async_copy is only supported for 8-bit inputs (fp8/int8)")

    gpu_arch = get_rocm_arch()
    is_gfx942 = str(gpu_arch).startswith("gfx942")
    is_gfx950 = str(gpu_arch).startswith("gfx950")
    use_mfma_scale_128 = is_fp8 and is_gfx950 and (tile_k % 128 == 0)
    use_mfma_k32 = is_f16_or_bf16 and is_gfx950

    if is_f16_or_bf16:
        layout_elem = Float16 if is_f16 else BFloat16
    elif is_int8:
        layout_elem = Int8
    else:
        layout_elem = Float8E4M3FN if is_gfx950 else Float8E4M3FNUZ
    final_out_elem_cls = BFloat16 if out_dtype == "bf16" else Float16
    out_elem_cls = Float32 if split_k > 1 else final_out_elem_cls
    out_elem_bytes = 4 if split_k > 1 else 2

    # Tile geometry (tile_K_perm = K-elements grouped per MMA k-step)
    tile_K_perm = 128 if use_mfma_scale_128 else (64 if is_8bit else 32)
    k_iters = tile_k // tile_K_perm
    num_tiles = split_k_extent // tile_k
    #: K-scale blocks; with the tile_k=128 blockscale requires, one per tile.
    kb_count = K // 128 if is_blockscale else 0
    m_repeat = tile_m // 16
    num_waves = 4
    n_per_wave = tile_n // num_waves
    num_acc_n = n_per_wave // 16
    acc_size = m_repeat * num_acc_n * 4

    total_threads = 256
    a_load_bytes = 16
    bytes_per_thread_a = (tile_m * tile_k * elem_bytes) // total_threads
    num_a_loads = bytes_per_thread_a // a_load_bytes
    num_b_loads = (tile_n * tile_k * elem_bytes) // total_threads // 16
    num_ds_load = (tile_m * tile_k * elem_bytes) // 64 // 16  # A LDS reads per wave
    num_gmem_loads = num_a_loads + num_b_loads
    if is_8bit and is_gfx950:
        dsrd_preload, dvmem_preload = _get_preload(tile_m, tile_n, tile_k)
    else:
        dsrd_preload, dvmem_preload = (0, 0)

    a_lds_elems = tile_m * tile_k

    @fx.struct
    class SharedStorage:
        a0: fx.Array[layout_elem, a_lds_elems, 16]
        if lds_stage == 2:
            a1: fx.Array[layout_elem, a_lds_elems, 16]
        if split_k > 1:
            split_flag: fx.Array[Int32, 1, 4]

    # ── Kernel ────────────────────────────────────────────────────────
    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Tensor,
        arg_out: fx.Tensor,
        arg_semaphore: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        tiled_mma_arg: fx.TiledMma,
        tiled_copy_g2s: fx.TiledCopy,
    ):
        tid = fx.thread_idx.x
        bid_x, bid_y, bid_z = fx.block_idx
        k_off = fx.Int32(bid_z) * num_tiles

        if const_expr(xcd_swizzle > 0):
            _bx, _by = xcd_remap_bx_by(
                gpu.block_id("x"),
                gpu.block_id("y"),
                fx.Index(i32_m),
                tile_m=tile_m,
                tile_n=tile_n,
                N=N,
                xcd_swizzle=xcd_swizzle,
            )
            bid_x, bid_y = Int32(_bx), Int32(_by)

        mma_atom = None
        if const_expr(use_mfma_scale_128):
            _scale_atom = fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, layout_elem)
            )
            mma_atom = _scale_atom
            tiled_mma = fx.make_tiled_mma(
                _scale_atom,
                fx.make_layout((1, 4, 1), (0, 1, 0)),
                fx.make_tile(None, None, fx.make_layout((32, 4), (1, 32))),
            )
        else:
            tiled_mma = tiled_mma_arg

        # Bound A (read) and C (store) to the actual M extent so blocks covering
        # rows past M (ragged M) drop their OOB loads/stores at the descriptor
        # instead of faulting / writing past the allocation. B and scale_b/bias
        # are per-N (exact multiple) and stay max_size; scale_a is per-row (M) and
        # is bounded the same way below (see its epilogue load).
        gA = fx.rocdl.make_buffer_tensor(
            arg_a,
            max_size=False,
            num_records_bytes=fx.Int64(i32_m) * fx.Int64(K) * fx.Int64(elem_bytes),
        )
        gB = fx.rocdl.make_buffer_tensor(arg_b)
        c_tensor = arg_c
        if const_expr(split_k > 1):
            c_split_offset = fx.Int64(bid_z) * fx.Int64(i32_m) * fx.Int64(N)
            c_tensor = fx.Tensor(
                fx.make_view(
                    fx.add_offset(fx.get_iter(arg_c), c_split_offset),
                    fx.make_layout((PRESHUFFLE_M_MAX, N), (N, 1)),
                )
            )
        gC = fx.rocdl.make_buffer_tensor(
            c_tensor,
            max_size=False,
            num_records_bytes=fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(out_elem_bytes),
        )

        tA = fx.flat_divide(gA, fx.make_tile(tile_m, tile_k))[None, None, bid_x, None]
        tB = fx.flat_divide(gB, fx.make_tile(tile_n, tile_k))[None, None, bid_y, None]
        tC = fx.flat_divide(gC, fx.make_tile(tile_m, tile_n))[None, None, bid_x, bid_y]

        buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), layout_elem)
        uni_copy = fx.make_copy_atom(fx.UniversalCopy128b(), layout_elem)

        # Per-thread slices
        thr_mma = tiled_mma.thr_slice(tid)
        thr_g2s = tiled_copy_g2s.get_slice(tid)
        thr_s2r = fx.make_tiled_copy_A(buf_copy, tiled_mma).get_slice(tid)
        thr_g2r_B = fx.make_tiled_copy_B(buf_copy, tiled_mma).get_slice(tid)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        if const_expr(is_8bit):
            k_blocks16 = (tile_k * elem_bytes) // 16
            if k_blocks16 <= 0 or (k_blocks16 & (k_blocks16 - 1)) != 0:
                raise ValueError(
                    f"Unsupported tile_k for 8-bit LDS swizzle: tile_k={tile_k}, elem_bytes={elem_bytes} (k_blocks16={k_blocks16}); "
                    "expected tile_k*elem_bytes to be a positive multiple of 16 with (tile_k*elem_bytes/16) a power of two."
                )
            swz_bits = k_blocks16.bit_length() - 1  # log2
            swz = fx.SwizzleType.get(swz_bits, 4, swz_bits)
        else:
            swz = fx.SwizzleType.get(3, 3, 3)

        def _make_sA(arr):
            return fx.make_view(
                arr.ptr,
                fx.make_composed_layout(
                    fx.static(swz),
                    fx.make_ordered_layout((tile_m, tile_k), (1, 0)),
                ),
            )

        if const_expr(lds_stage == 1):
            sA_stages = [_make_sA(lds.a0)]
        else:
            sA_stages = [_make_sA(lds.a0), _make_sA(lds.a1)]

        # Partitions
        pA_g = thr_g2s.partition_S(tA)
        pA_s_stages = [thr_g2s.partition_D(s) for s in sA_stages]
        pA_s2r_stages = [thr_s2r.partition_S(s) for s in sA_stages]
        pB_g = thr_g2r_B.partition_S(tB)

        # Fragments — 2 separate B fragments (split double buffer for VGPR lifetime)
        frag_copy_A = fx.make_fragment_like(pA_s_stages[0][None, None, None])
        frag_A = thr_mma.make_fragment_A(sA_stages[0])
        frag_B_single_layout = thr_mma.partition_B(tB).layout(None, None, None, 0)
        frag_B_stages = [
            fx.make_fragment_like(frag_B_single_layout, layout_elem.ir_type)
            for _ in range(2)
        ]
        frag_C = thr_mma.make_fragment_C(tC)
        frag_A_retile = thr_s2r.retile(frag_A)
        frag_B_retile_stages = [thr_g2r_B.retile(b) for b in frag_B_stages]
        out_cpol = CPOL_COHERENT if split_k > 1 else 0
        copy_op = fx.rocdl.BufferCopy32b if split_k > 1 else fx.rocdl.BufferCopy16b
        buf_copy_out = fx.make_copy_atom(copy_op(out_cpol), out_elem_cls)
        # Only built for the tiled write-out. Under the swapped layout the
        # explicit store replaces it, and leaving frag_C_out allocated costs
        # acc_size out-elements of register space for nothing.
        if const_expr(not swap_ab):
            thr_r2g_C = fx.make_tiled_copy_C(buf_copy_out, tiled_mma).get_slice(tid)
            pC_g = thr_r2g_C.partition_S(tC)
            frag_C_out = fx.make_fragment_like(frag_C, out_elem_cls.ir_type)
            frag_C_retile = thr_r2g_C.retile(frag_C_out)

        # ── Async gmem->LDS DMA (buffer_load_lds) for the A tile ──
        if const_expr(use_async_copy):
            dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
            # Bound to the real M extent (like the sync gA) so ragged-M blocks DMA-read
            # OOB rows as 0 instead of faulting past the allocation.
            gA_flat = fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(
                        fx.get_iter(arg_a), fx.make_layout(PRESHUFFLE_M_MAX * K, 1)
                    )
                ),
                max_size=False,
                num_records_bytes=fx.Int64(i32_m) * fx.Int64(K) * fx.Int64(elem_bytes),
            )
            gA_div = fx.logical_divide(gA_flat, fx.make_layout(1, 1))
            sA_i8_ptr = [fx.recast_iter(Int8, lds.a0.ptr)]
            if const_expr(lds_stage == 2):
                sA_i8_ptr.append(fx.recast_iter(Int8, lds.a1.ptr))
            bx_m = bid_x * tile_m
            wave_id = tid // 64
            step_bytes = total_threads * a_load_bytes
            wave_stride_bytes = 64 * a_load_bytes
            k_blocks16_dma = (tile_k * elem_bytes) // 16
            elems_per_16b = 16 // elem_bytes

            def dma_a_to_lds(k_tile_val, stage):
                wave_off = rocdl.readfirstlane(
                    fx.Int32.ir_type, wave_id * wave_stride_bytes
                )
                lds_ptr = fx.add_offset(sA_i8_ptr[stage], wave_off)
                base_k = (k_off + k_tile_val) * tile_k
                for i in range_constexpr(num_a_loads):
                    if const_expr(i > 0):
                        lds_ptr = fx.add_offset(lds_ptr, step_bytes)
                    pos_bytes = i * total_threads * a_load_bytes + tid * a_load_bytes
                    elem_idx = pos_bytes // elem_bytes
                    m = elem_idx // tile_k
                    k = elem_idx % tile_k
                    k_swz = k ^ ((m % k_blocks16_dma) * elems_per_16b)
                    gmem_byte = ((bx_m + m) * K + base_k + k_swz) * elem_bytes
                    dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                    src = fx.slice(gA_div, (None, fx.Int32(gmem_byte)))
                    fx.copy(dma_atom, src, dst)

        # ── Scheduling hints (ported from old pipeline) ───────────
        def build_scheduler(numer: int, denom: int):
            if const_expr(denom <= 0):
                return []
            if const_expr(numer <= 0):
                return [0] * denom
            out = []
            prev = 0
            for i in range_constexpr(denom):
                cur = ((i + 1) * numer + (denom - 1)) // denom
                out.append(cur - prev)
                prev = cur
            return out

        def hot_loop_scheduler():
            mfma_group = num_acc_n

            if const_expr(is_gfx942):
                mfma_total = (k_iters * 2) * m_repeat * mfma_group
                mfma_per_iter = 2 * mfma_group
                sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)

                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(1)
                if const_expr(tile_m == 16):
                    rocdl.sched_vmem(1)
                rocdl.sched_mfma(1)
                if const_expr(tile_m == 16):
                    rocdl.sched_vmem(1)

                if const_expr(num_acc_n < 4):
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16):
                        rocdl.sched_vmem(1)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16):
                        rocdl.sched_vmem(1)
                    rocdl.sched_mfma(1)

                dswr_tail = num_a_loads
                dstr_advance = 2
                if const_expr(dswr_tail > sche_iters):
                    dswr_tail = sche_iters
                dswr_start = max(sche_iters - dswr_tail - dstr_advance, 0)

                for sche_i in range_constexpr(sche_iters):
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(mfma_group)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(mfma_group)
                    if const_expr(sche_i >= dswr_start - 1):
                        rocdl.sched_dswr(1)
            else:
                if const_expr(use_mfma_scale_128):
                    element_k_per_mfma = 128
                else:
                    element_k_per_mfma = 32
                num_mfma_per_tile_k = tile_k // element_k_per_mfma
                mfma_total = num_mfma_per_tile_k * m_repeat * mfma_group
                dswr_tail = num_a_loads
                dstr_advance = 2
                if const_expr(dswr_tail > mfma_total):
                    dswr_tail = mfma_total
                dsrd_preload_eff = min(int(dsrd_preload), num_ds_load)
                dvmem_preload_eff = min(int(dvmem_preload), num_gmem_loads)
                vmem_remaining = num_gmem_loads - dvmem_preload_eff
                dsrd_remaining = num_ds_load - dsrd_preload_eff
                if const_expr(vmem_remaining > 0 and vmem_remaining < mfma_total):
                    vmem_schedule = build_scheduler(vmem_remaining, vmem_remaining) + [
                        0
                    ] * (mfma_total - vmem_remaining)
                else:
                    vmem_schedule = build_scheduler(vmem_remaining, mfma_total)
                dsrd_schedule = build_scheduler(dsrd_remaining, mfma_total)
                dswr_start = max(mfma_total - dswr_tail - dstr_advance, 0)
                last_dsrd_mfma_idx = -1
                for sched_idx in range_constexpr(mfma_total):
                    if const_expr(dsrd_schedule[sched_idx]):
                        last_dsrd_mfma_idx = sched_idx
                dswr_start = max(dswr_start, last_dsrd_mfma_idx + 1)
                idx_ds_read = dsrd_preload_eff
                idx_gmem_load = dvmem_preload_eff
                idx_ds_write = 0
                if const_expr(dvmem_preload_eff):
                    rocdl.sched_vmem(dvmem_preload_eff)
                if const_expr(dsrd_preload_eff):
                    rocdl.sched_dsrd(dsrd_preload_eff)
                for mfma_idx in range_constexpr(mfma_total):
                    rocdl.sched_mfma(1)
                    n_dsrd = dsrd_schedule[mfma_idx]
                    if const_expr(n_dsrd and (idx_ds_read < num_ds_load)):
                        if const_expr(idx_ds_read + n_dsrd > num_ds_load):
                            n_dsrd = num_ds_load - idx_ds_read
                        if const_expr(n_dsrd):
                            rocdl.sched_dsrd(n_dsrd)
                            idx_ds_read += n_dsrd
                    n_vmem = vmem_schedule[mfma_idx]
                    if const_expr(n_vmem and (idx_gmem_load < num_gmem_loads)):
                        if const_expr(idx_gmem_load + n_vmem > num_gmem_loads):
                            n_vmem = num_gmem_loads - idx_gmem_load
                        if const_expr(n_vmem):
                            rocdl.sched_vmem(n_vmem)
                            idx_gmem_load += n_vmem
                    if const_expr(
                        (not use_async_copy)
                        and (idx_ds_write < dswr_tail)
                        and (mfma_idx >= dswr_start)
                    ):
                        rocdl.sched_dswr(1)
                        idx_ds_write += 1
                if const_expr((not use_async_copy) and idx_ds_write < num_a_loads):
                    rocdl.sched_dswr(num_a_loads - idx_ds_write)

            rocdl.sched_barrier(0)

        # Accumulator addressing, shared by the block-scale rescale and the
        # epilogue: element p of frag_C is row bx_m + mi*16 + lane_div_16*4 + ii,
        # column by_n + (ni*num_waves + wave_id)*16 + lane_mod_16.
        bx_m = bid_x * tile_m
        by_n = bid_y * tile_n
        wave_id = gpu.thread_id("x") // 64
        lane_id = gpu.thread_id("x") % 64
        lane_div_16 = lane_id // 16
        lane_mod_16 = lane_id % 16

        # Scale operands go through the layout API rather than aiter's
        # buffer_ops: on this FlyDSL, RawPtrBufferLoadOp wants `aux` as an
        # operand, and buffer_ops passes None when no cache modifier is set, so
        # every buffer_ops.buffer_load in this file raises. That breaks the
        # upstream ptpc epilogue too; blockscale simply does not use it.
        if const_expr(is_blockscale):
            gSA = fx.rocdl.make_buffer_tensor(
                arg_scale_a,
                max_size=False,
                num_records_bytes=fx.Int64(i32_m) * fx.Int64(kb_count) * fx.Int64(4),
            )
            gSB = fx.rocdl.make_buffer_tensor(arg_scale_b)
            sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
            sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))
            # C write-out for the swapped accumulator. The tiled_mma-derived
            # path (thr_r2g_C / pC_g) partitions C for the *unswapped* layout,
            # so it would scatter these four values down four rows. They are
            # four consecutive columns of one row instead, which in a row-major
            # C is 8 contiguous bytes -- one store, where the unswapped layout
            # can only ever emit buffer_store_short.
            gC_flat = fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(
                        fx.get_iter(c_tensor),
                        fx.make_layout(PRESHUFFLE_M_MAX * N, 1),
                    )
                ),
                max_size=False,
                num_records_bytes=fx.Int64(i32_m)
                * fx.Int64(N)
                * fx.Int64(out_elem_bytes),
            )
            c_div = fx.logical_divide(gC_flat, fx.make_layout(1, 1))
            c_atom_64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), out_elem_cls)
            sc_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            sc_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
            sc_reg_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
            sa_reg_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
            sc_reg_4 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)

        def _load_sa(mi, kb):
            """This lane's A scale for M-tile ``mi`` at K-block ``kb``.

            ``arg_scale_a`` is physically [K/128, M] (column-major [M, K/128]),
            the layout aiter_per1x128_quant(transpose_scale=True) emits. With
            the operands swapped a lane owns a single row -- ``lane % 16`` --
            so this is one scalar, and combined with B's it gives the one
            per-tile ``c_scale_thread`` CK promotes with.
            """
            if const_expr(swap_ab):
                idx = fx.Int32(kb) * fx.Int32(i32_m) + bx_m + mi * 16 + lane_mod_16
                fx.copy(sc_atom_1, fx.slice(sa_div, (None, fx.Int32(idx))), sa_reg_1)
                return Vec(fx.memref_load_vec(sa_reg_1))[0]
            idx = fx.Int32(kb) * fx.Int32(i32_m) + bx_m + mi * 16 + lane_div_16 * 4
            fx.copy(sc_atom_4, fx.slice(sa_div, (None, fx.Int32(idx))), sc_reg_4)
            return Vec(fx.memref_load_vec(sc_reg_4))

        def _load_sb(ni, kb):
            """The single B scale covering N-tile ``ni`` at K-block ``kb``.

            ``arg_scale_b`` is [N/128, K/128] row-major, and a 16-column group
            lies wholly inside one 128-wide block, so this is one scalar.
            """
            nblk = (by_n + (ni * num_waves + wave_id) * 16) // 128
            idx = nblk * fx.Int32(kb_count) + fx.Int32(kb)
            fx.copy(sc_atom_1, fx.slice(sb_div, (None, fx.Int32(idx))), sc_reg_1)
            return Vec(fx.memref_load_vec(sc_reg_1))[0]

        def mma_promote(a_stage, cur_frag_B, ki, kb):
            """One scale block's MFMAs, promoted a tile at a time.

            CK's shape (blockwise_gemm_pipeline_xdlops_blockscale_b_preshuffle
            _v1.hpp): the loop is (m0, n0, kscale) with k innermost and the
            partial lives in ``c_thread_buf_per_scale``, a single
            ``GetRegSizePerXdlops()``-wide buffer. Only one tile's 4 VGPRs are
            live at a time, which is why CK pays almost nothing for the block
            scale where both whole-fragment forms are expensive here: a running
            rescale needs a divide per element per block (+660us) and a
            whole-fragment promotion needs a duplicate accumulator and spills
            (2753us).

            frag_C's modes are ((4,1), m_repeat, num_acc_n) and frag_A /
            frag_B's are (elems, m_repeat|num_acc_n, k_iters), so one tile is a
            plain slice of each; ``fx.gemm`` is handed the bare atom rather than
            tiled_mma, exactly as Mfma16x16x128._do_mma does.
            """
            sb = [_load_sb(ni, kb) for ni in range_constexpr(num_acc_n)]
            # Operands are materialised into bare rmem tensors, the way
            # Mfma16x16x128._make_operand_frag does: fx.gemm on the atom does
            # not take a sliced view, it leaves a ub.poison the rmem-to-SSA
            # promotion pass then rejects. Each is built once per k-step, not
            # once per (mi, ni) pair.
            b_ops = []
            for ni in range_constexpr(num_acc_n):
                b_op = fx.make_rmem_tensor(8, fx.Int32)
                b_op.store(Vec(cur_frag_B[None, ni, ki].load()).bitcast(fx.Int32))
                b_ops.append(b_op)
            acc = Vec(frag_C.load())
            vals = [acc[p] for p in range_constexpr(acc_size)]
            for mi in range_constexpr(m_repeat):
                sa = _load_sa(mi, kb)
                a_op = fx.make_rmem_tensor(8, fx.Int32)
                a_op.store(Vec(frag_A[None, mi, ki].load()).bitcast(fx.Int32))
                # All of this M-row's MFMAs are issued before any of their
                # results is read. With tile_k=128 a scale block is a single
                # MFMA, so promoting each one where it is issued exposes the
                # full MFMA latency; CK hides it by keeping k innermost, which
                # is not available here. num_acc_n partials cost 4*num_acc_n
                # VGPRs.
                parts = []
                for ni in range_constexpr(num_acc_n):
                    part = fx.make_rmem_tensor(4, fx.Float32)
                    part.store(Vec.filled(4, 0.0, fx.Float32))
                    # Operands exchanged: fx.gemm(atom, c, b, a, c) computes
                    # (A B)^T in the accumulator's native layout, so a lane's
                    # four values become four consecutive *columns* of one row
                    # instead of four rows. gcnasm's mfma_adaptor_swap_ab.
                    if const_expr(swap_ab):
                        fx.gemm(mma_atom, part, b_ops[ni], a_op, part)
                    else:
                        fx.gemm(mma_atom, part, a_op, b_ops[ni], part)
                    parts.append(part)
                for ni in range_constexpr(num_acc_n):
                    sab = (
                        [sa * sb[ni]] * 4
                        if const_expr(swap_ab)
                        else [sa[ii] * sb[ni] for ii in range_constexpr(4)]
                    )
                    pv = Vec(parts[ni].load())
                    # Explicit fma: written as ``acc + pv * (sa * sb)`` the
                    # compiler emits v_pk_mul_f32 + v_pk_add_f32, three ops per
                    # element against CK's one v_fmac_f32. Hoisting the scale
                    # product and contracting the rest gets it to two.
                    for ii in range_constexpr(4):
                        # frag_C.load() yields logical order: ii fastest, then
                        # m_repeat, then num_acc_n.
                        idx = ni * (m_repeat * 4) + mi * 4 + ii
                        vals[idx] = fx.math.fma(pv[ii], sab[ii], vals[idx])
            frag_C.store(vector.from_elements(T.vec(acc_size, T.f32), vals))

        # ── Pipeline stage (double-buffered B via split fragments) ─
        def mma_kloop(a_stage, cur_frag_B, tile_idx=None):
            for ki in range_constexpr(k_iters):
                fx.copy(
                    uni_copy,
                    pA_s2r_stages[a_stage][None, None, ki],
                    frag_A_retile[None, None, ki],
                )
                if const_expr(is_blockscale):
                    # One MMA k-step is one 128-wide scale block.
                    mma_promote(
                        a_stage,
                        cur_frag_B,
                        ki,
                        fx.Int32(tile_idx) * fx.Int32(k_iters) + fx.Int32(ki),
                    )
                    continue
                k_coord = ki if (use_mfma_scale_128 or use_mfma_k32) else (None, ki)
                fx.gemm(
                    tiled_mma,
                    frag_C,
                    frag_A[None, None, k_coord],
                    cur_frag_B[None, None, k_coord],
                    frag_C,
                )

        def pipeline_2stage(read_stage, next_k_val=None, read_next=True, cur_k_val=None):
            write_stage = read_stage ^ 1
            a_read = read_stage
            a_write = write_stage
            cur_frag_B = frag_B_stages[read_stage]
            do_next = read_next and next_k_val is not None
            if const_expr(use_async_copy):
                if const_expr(do_next):
                    dma_a_to_lds(next_k_val, a_write)
                    fx.copy(
                        buf_copy,
                        pB_g[None, None, None, k_off + next_k_val],
                        frag_B_retile_stages[write_stage],
                    )
                mma_kloop(a_read, cur_frag_B, cur_k_val)
                if const_expr(enable_scheduler):
                    hot_loop_scheduler()
                if const_expr(do_next):
                    rocdl.s_waitcnt(num_b_loads)
                gpu.barrier()
                return
            if const_expr(do_next):
                global_k_next = k_off + next_k_val
                fx.copy(buf_copy, pA_g[None, None, None, global_k_next], frag_copy_A)
                fx.copy(
                    buf_copy,
                    pB_g[None, None, None, global_k_next],
                    frag_B_retile_stages[write_stage],
                )
            mma_kloop(a_read, cur_frag_B, cur_k_val)
            if const_expr(do_next):
                fx.copy(uni_copy, frag_copy_A, pA_s_stages[a_write][None, None, None])
            if const_expr(enable_scheduler):
                hot_loop_scheduler()
            if const_expr(do_next):
                gpu.barrier()

        # ── Prologue ──────────────────────────────────────────────
        acc_zero = (
            Vec.filled(acc_size, 0, Int32)
            if const_expr(is_int8)
            else Vec.filled(acc_size, 0.0, Float32)
        )
        if const_expr(use_async_copy):
            dma_a_to_lds(fx.Int32(0), 0)
            fx.copy(buf_copy, pB_g[None, None, None, k_off], frag_B_retile_stages[0])
            frag_C.store(acc_zero)
            rocdl.s_waitcnt(num_b_loads)
            gpu.barrier()
        else:
            fx.copy(buf_copy, pA_g[None, None, None, k_off], frag_copy_A)
            fx.copy(buf_copy, pB_g[None, None, None, k_off], frag_B_retile_stages[0])
            frag_C.store(acc_zero)
            fx.copy(uni_copy, frag_copy_A, pA_s_stages[0][None, None, None])
            gpu.barrier()
        rocdl.sched_barrier(0)

        # ── Main tile loop ────────────────────────────────────────────
        if const_expr(lds_stage == 1 and num_tiles > 1):
            frag_Bc = frag_B_stages[0]
            frag_Bc_retile = frag_B_retile_stages[0]
            for iv, state in range(0, num_tiles - 1, 1, init=[frag_C.load()]):
                frag_C.store(state[0])
                k_next = fx.Int32(iv + 1)
                mma_kloop(0, frag_Bc)
                if const_expr(not use_async_copy):
                    fx.copy(
                        buf_copy, pA_g[None, None, None, k_off + k_next], frag_copy_A
                    )
                fx.copy(
                    buf_copy,
                    pB_g[None, None, None, k_off + k_next],
                    frag_Bc_retile,
                )
                gpu.barrier()  # single buffer: all reads done before overwrite
                if const_expr(use_async_copy):
                    dma_a_to_lds(k_next, 0)
                else:
                    fx.copy(uni_copy, frag_copy_A, pA_s_stages[0][None, None, None])
                if const_expr(enable_scheduler):
                    hot_loop_scheduler()
                if const_expr(use_async_copy):
                    rocdl.s_waitcnt(num_b_loads)
                gpu.barrier()
                results = yield [frag_C.load()]
            frag_C.store(results)
        elif const_expr(lds_stage == 2):
            # 2-tile/iter ping-pong: middle loop runs 2 tiles/iter
            is_odd_tiles = (num_tiles % 2) == 1
            tail = 1 if is_odd_tiles else 2
            loop_end = (num_tiles - tail) // 2

            def two_tiles(k_base):
                pipeline_2stage(
                    read_stage=0, next_k_val=k_base + fx.Int32(1), cur_k_val=k_base
                )
                pipeline_2stage(
                    read_stage=1,
                    next_k_val=k_base + fx.Int32(2),
                    cur_k_val=k_base + fx.Int32(1),
                )

            if const_expr(loop_end > 0 and use_async_copy):
                for iv in range_constexpr(loop_end):
                    two_tiles(fx.Int32(iv * 2))
            elif const_expr(loop_end > 0):
                for iv, state in range(0, loop_end, 1, init=[frag_C.load()]):
                    frag_C.store(state[0])
                    two_tiles(fx.Int32(iv * 2))
                    results = yield [frag_C.load()]
                frag_C.store(results)
            k_tail0 = num_tiles - tail  # first tile handled by the peeled tail
            for j in range_constexpr(tail - 1):
                pipeline_2stage(
                    read_stage=(k_tail0 + j) % 2,
                    next_k_val=fx.Int32(k_tail0 + j + 1),
                    cur_k_val=fx.Int32(k_tail0 + j),
                )

        # ── Epilogue-operand preloads (scale_a / scale_b / bias) ─────────
        def load_epi_operands():
            s_a = s_b = bias = None
            if const_expr(is_blockscale):
                # Nothing to preload: mma_promote() folds every K-block's
                # scales into frag_C as it goes, so the epilogue below sees an
                # already-scaled accumulator.
                pass
            elif const_expr(is_8bit):
                # Per-row(scale_a) × per-col(scale_b) scaling, applied in the epilogue.
                scale_b_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_b, max_size=True
                )
                s_b = [
                    buffer_ops.buffer_load(
                        scale_b_rsrc,
                        fx.Int32(by_n + (ni * num_waves + wave_id) * 16 + lane_mod_16),
                        vec_width=1,
                        dtype=T.f32,
                    )
                    for ni in range_constexpr(num_acc_n)
                ]
                scale_a_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_a,
                    max_size=False,
                    num_records_bytes=fx.Int64(i32_m) * fx.Int64(4),
                )
                s_a = [
                    Vec(
                        buffer_ops.buffer_load(
                            scale_a_rsrc,
                            fx.Int32(bx_m + mi * 16 + lane_div_16 * 4),
                            vec_width=4,
                            dtype=T.f32,
                        )
                    ).bitcast(fx.Float32)
                    for mi in range_constexpr(m_repeat)
                ]
            if const_expr(_has_bias):
                # Per-column bias (out_dtype), one scalar per N-block, shared across rows.
                bias_rsrc = buffer_ops.create_buffer_resource(arg_bias, max_size=True)
                bias_elem_ty = T.bf16 if out_dtype == "bf16" else T.f16
                bias = [
                    fx.Float32(
                        buffer_ops.buffer_load(
                            bias_rsrc,
                            fx.Int32(
                                by_n + (ni * num_waves + wave_id) * 16 + lane_mod_16
                            ),
                            vec_width=1,
                            dtype=bias_elem_ty,
                        )
                    )
                    for ni in range_constexpr(num_acc_n)
                ]
            return s_a, s_b, bias

        overlap_epi_load = (
            acc_size <= 64
        )  # small enough accumulator to keep operands live over the MMA
        s_a_vals = s_b_vals = bias_vals = None
        if const_expr(overlap_epi_load):
            s_a_vals, s_b_vals, bias_vals = load_epi_operands()

        # Final MMA stage — overlaps the epilogue-operand loads when issued above.
        if const_expr(lds_stage == 1):
            mma_kloop(0, frag_B_stages[0])
        else:
            pipeline_2stage(
                read_stage=(num_tiles - 1) % 2,
                read_next=False,
                cur_k_val=fx.Int32(num_tiles - 1),
            )

        # ── Epilogue ─────────────────────────────────────────────
        if const_expr(swap_ab):
            # Allocated here, not at kernel scope: held across the main loop
            # these push VGPRs over the 2-waves/SIMD line and the loop spills
            # to scratch, which cost more than the wider store saved.
            c_reg_4 = [
                fx.make_rmem_tensor(fx.make_layout(4, 1), out_elem_cls)
                for _ in range(m_repeat * num_acc_n)
            ]
            # Convert first, store second: interleaving them keeps the f32
            # accumulator alive alongside the growing set of converted values.
            # One base address plus compile-time strides, not 16 runtime
            # indices: mi steps 16 rows and ni steps num_waves*16 columns, both
            # static, so the stores use the buffer's immediate offset instead of
            # a VGPR each.
            base = (
                (bx_m + lane_mod_16) * fx.Int32(N)
                + by_n
                + wave_id * 16
                + lane_div_16 * 4
            )
            cview = fx.make_view(
                fx.add_offset(fx.get_iter(gC_flat), fx.Int32(base)),
                fx.make_layout(
                    (4, m_repeat, num_acc_n), (1, 16 * N, num_waves * 16)
                ),
            )
            acc_vec = Vec(frag_C.load())
            out_elems = [
                acc_vec[p].to(out_elem_cls) for p in range_constexpr(acc_size)
            ]
            for mi in range_constexpr(m_repeat):
                for ni in range_constexpr(num_acc_n):
                    creg = c_reg_4[mi * num_acc_n + ni]
                    creg.store(
                        vector.from_elements(
                            T.vec(4, out_elem_cls.ir_type),
                            [
                                out_elems[ni * (m_repeat * 4) + mi * 4 + ii]
                                for ii in range_constexpr(4)
                            ],
                        )
                    )
                    fx.copy(c_atom_64, creg, cview[None, mi, ni])
        elif const_expr((not is_8bit or is_blockscale) and not _has_epilogue):
            frag_C_out.store(Vec(frag_C.load()).to(out_elem_cls))
        else:
            if const_expr(not overlap_epi_load):
                s_a_vals, s_b_vals, bias_vals = load_epi_operands()

            def apply_activation(val_s):
                # ReLU/SiLU/GeLU : maximumf for relu; exp+rcp for silu;
                # tanh-approx gelu expanded through a non-positive exponent (no overflow).
                if const_expr(_has_relu):
                    return fx.Float32(val_s).maximumf(fx.Float32(0.0))
                if const_expr(_has_silu):
                    exp_neg = math.exp(val_s * fx.Float32(-1.0))
                    return val_s * (fx.Float32(1.0) / (fx.Float32(1.0) + exp_neg))
                if const_expr(_has_gelu):
                    half_f32 = fx.Float32(0.5)
                    one_f32 = fx.Float32(1.0)
                    zero_f32 = fx.Float32(0.0)
                    two_f32 = fx.Float32(2.0)
                    x3 = val_s * val_s * val_s
                    y = fx.Float32(0.7978845608) * (val_s + fx.Float32(0.044715) * x3)
                    abs_y = fx.Float32(y).maximumf(zero_f32 - y)
                    e_neg2abs = math.exp(fx.Float32(-2.0) * abs_y)
                    denom = one_f32 + e_neg2abs
                    numerator = (y > zero_f32).select(two_f32, two_f32 * e_neg2abs)
                    return half_f32 * val_s * (numerator * (one_f32 / denom))
                return val_s

            acc_vec = Vec(frag_C.load())
            out_elems = []
            for p in range_constexpr(acc_size):
                ni = p // (m_repeat * 4)
                mi = (p // 4) % m_repeat
                ii = p % 4
                val = acc_vec[p]
                if const_expr(is_int8):
                    val = val.to(Float32)
                if const_expr(is_8bit and not is_blockscale):
                    val_s = (val * s_a_vals[mi][ii]) * s_b_vals[ni]
                else:
                    val_s = val
                if const_expr(_has_bias):
                    val_s = val_s + bias_vals[ni]
                val_s = apply_activation(val_s)
                out_elems.append(val_s.to(out_elem_cls))

            out_vec = vector.from_elements(
                T.vec(acc_size, out_elem_cls.ir_type), out_elems
            )
            frag_C_out.store(out_vec)

        if const_expr(not swap_ab):
            fx.copy(buf_copy_out, frag_C_retile, pC_g)
        if const_expr(split_k > 1):
            # The store carries sc0|sc1, so waiting on it is the whole release.
            rocdl.s_waitcnt(0)
            gpu.barrier()
            splitk_reduce_epilogue(
                arg_c,
                fx.Tensor(
                    fx.make_view(
                        fx.get_iter(arg_out),
                        fx.make_layout((PRESHUFFLE_M_MAX, N), (N, 1)),
                    )
                ),
                arg_semaphore,
                lds.split_flag.ptr,
                tile_m,
                tile_n,
                total_threads,
                final_out_elem_cls,
                tid,
                bid_x,
                bid_y,
                i32_m,
                N,
                split_k,
            )

    # ── Host launcher ─────────────────────────────────────────────
    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_out: fx.Tensor,
        arg_semaphore: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()

        # MMA atom — layout_elem carries the dtype (Float16/BFloat16/Float8E4M3FN/etc)
        if const_expr(use_mfma_k32):
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, layout_elem))
            k_perm = fx.make_layout((8, 4), (1, 8))
        elif const_expr(is_f16_or_bf16):
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, layout_elem))
            k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
        elif const_expr(is_int8):
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, layout_elem, Int32))
            k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))
        else:
            # fp8: narrow atom here; the scale (16x16x128) tiled_mma is rebuilt in-kernel
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, layout_elem))
            k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))

        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((1, 4, 1), (0, 1, 0)),
            fx.make_tile(None, None, k_perm),
        )

        # G2S tiled copy
        val_per_thr = a_load_bytes // elem_bytes
        thrs_k = tile_k // val_per_thr
        thrs_m = total_threads // thrs_k
        tiled_copy_g2s = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy128b(), layout_elem),
            fx.make_layout(
                ((thrs_k, thrs_m), (1, val_per_thr)),
                ((thrs_m * val_per_thr, 1), (1, thrs_m)),
            ),
            fx.make_tile(thrs_m, tile_k),
        )

        # Preshuffle B layout (2D hierarchical)
        kp_bytes = 16
        kp_elems = kp_bytes if elem_bytes == 1 else kp_bytes // elem_bytes
        k_bytes_b = K * elem_bytes
        n0 = N // 16
        k0 = k_bytes_b // 64
        s_nlane = kp_elems
        s_klane = 16 * s_nlane
        s_k0 = 4 * s_klane
        s_n0 = k0 * s_k0
        preshuffle_B = fx.Tensor(
            fx.make_view(
                fx.get_iter(arg_b),
                fx.make_layout(
                    ((16, n0), (kp_elems, 4, k0)), ((s_nlane, s_n0), (1, s_klane, s_k0))
                ),
            )
        )

        # Reshape A and C to 2D
        M_max = PRESHUFFLE_M_MAX
        arg_a_2d = fx.Tensor(
            fx.make_view(fx.get_iter(arg_a), fx.make_layout((M_max, K), (K, 1)))
        )
        arg_c_2d = fx.Tensor(
            fx.make_view(fx.get_iter(arg_c), fx.make_layout((M_max, N), (N, 1)))
        )

        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = i32_n // tile_n

        kernel_gemm(
            arg_c_2d,
            arg_out,
            arg_semaphore,
            arg_a_2d,
            preshuffle_B,
            arg_scale_a,
            arg_scale_b,
            arg_bias,
            i32_m,
            i32_n,
            tiled_mma,
            tiled_copy_g2s,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu},
        ).launch(
            grid=(gx, gy, split_k),
            block=(256, 1, 1),
            stream=stream,
        )

    if const_expr(is_f16_or_bf16 and num_acc_n <= 2):
        launch_gemm.compile_hints["llvm_options"] = {"enable-post-misched": False}

    return launch_gemm
