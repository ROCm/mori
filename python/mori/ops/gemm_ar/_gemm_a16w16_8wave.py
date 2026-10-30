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
"""Building blocks of the 8-wave quad-subtile BF16 GEMM, with a BF16 or FP32 C store.

``kernels_fused_bf16.py`` builds gemm_ar's BF16 GEMM + all-reduce from these:
layers a checkpoint keeps in BF16 (Kimi-K3's attention ``o_proj`` and shared
experts) have no fp8 contract for ``kernels_fused.py`` to compile against. They
are the pieces ``gemm_ag``'s BF16 GEMM (#807) is built from, copied here
unchanged; only the kernel factories differ.

## What this is a port of

``gcnasm/opus_gemm_dist/opus_gemm_a2a_lsa/gemm_a16w16_quad_subtile_kernel_template.hpp``,
instantiated there as ``opus_gemm_traits<512, 256, 256, 64, bf16, bf16, bf16,
float>``. The port is small because **``gemm_ar/_gemm_a8w8_8wave.py`` is already
a FlyDSL rendering of that same template**, with fp8 operands:

======================================  ====================================
template                                ``_gemm_a8w8_8wave.py``
======================================  ====================================
``T_M=2, T_N=4``, 8 waves               ``wave_m = wave_id // 4``, ``wave_n``
``v_c[2][2]``, four sub-MMAs            ``c00/c01/c10/c11_frag``
``HALF_B_M/N``                          ``LDS_BLOCK_M/N``
``mfma_adaptor_swap_ab``                ``_SwappedMfma``
``permlane16_swap`` store               ``_PermlaneStoreC``
``E_M = HALF_B_M / 32``                 ``N_TILES_A = BLOCK_M // 64``
``E_N = HALF_B_N / 64``                 ``N_TILES_B = BLOCK_N // 128``
======================================  ====================================

The pipeline retains the staggered waves, double-buffered tic/toc, and two
peeled K steps. Its synchronization is checked separately from the template:
the combined B0/A0 read needs an explicit LDS completion wait, and the tail
must drain prefetches before reading the final tile. The precision changes are:

1. **The MFMA** is ``v_mfma_f32_16x16x32_bf16`` instead of the scaled fp8
   ``16x16x128``. An operand fragment is 8 bf16 = 16 B = four dwords, against
   fp8's 32 B / eight.
2. **``E_K = B_K / W_K = 64 / 32 = 2``**, against fp8's ``128 / 128 = 1``. Two
   MFMAs per accumulator per K block, so ``S2RLoaderBf16`` returns ``E_K``
   fragments per tile and ``MfmaBf16.call`` loops over them. ``pack_i32x4_i32x8``
   is dropped rather than adapted -- it exists to build fp8's 32 B operand out
   of two 16 B loads, and bf16's operand *is* 16 B.
3. **No scales at all.** bf16 is not a quantised format, so there is no ptpc /
   blockscale / mxfp8 axis, no ``_BlockScaleK``, no ``_Mxfp8ScaleK``, and no
   scale arguments in the kernel signature.
4. **``VEC = 8``** (16 B / 2 B) against fp8's 16, which changes only the ``col``
   term of the global-load map and the LDS swizzle's element width.

B is **not** preshuffled here. The template reads a plain row-major ``[N, K]``
(``make_layout_gb``), and the preshuffle in ``gemm_ar`` exists for the fp8
kernel's operand layout.

## The LDS swizzle

``gemm_ar``'s ``swizzle_128`` is element-indexed at 128 elements per row, which
for fp8 *is* 128 bytes per row. ``swizzle_row128b`` below is the same function
written in bytes, so it reduces to ``swizzle_128`` exactly at
``elem_bytes == 1`` and covers bf16's 64-element rows at the same 128-byte
geometry. The XOR granule is 16 B and 2 divides 16, so the rotation can never
split a bf16 element.

The template avoids bank conflicts with padding (``smem_linear_wave +
smem_padding``) rather than an XOR. Both work; keeping mori's XOR means the rest
of the pipeline transfers unchanged, which is the point of porting onto this
file rather than away from it.

## A debugging trap, recorded because it cost a morning

FlyDSL caches compiled kernels under ``~/.flydsl/cache``, keyed on the
``@flyc.jit`` **wrapper**'s name (``launch_gemm_<hash>``) rather than on the
``@flyc.kernel`` body. Editing a kernel and re-running can therefore silently
execute the *previous* build. While porting this file that made a fixed bug
look unfixed, and -- because only the ``K`` values compiled before the fix were
stale -- produced a completely convincing false signal: odd ``K_ITERS`` passed
bit-exact and even ``K_ITERS`` failed, which looks exactly like a
double-buffer parity bug and is not one.

If a change to a kernel body appears to have no effect, move
``~/.flydsl/cache/launch_*`` aside before concluding anything.

## The C store, and why FP32 is the cheaper one

``_SwapABStoreC``'s docstring records that the A/B swap moves lane ``l`` value
``k`` to ``D[l % 16][4 * (l / 16) + k]`` -- **four consecutive columns**. At two
bytes that is 8 B, too narrow for a good store, which is the entire reason the
bf16 path pays for a ``permlane16_swap``: it pairs two N-tiles to reach 16 B.

At four bytes those same four columns are already 16 contiguous bytes. So the
FP32 store needs **no shuffle and no convert** -- it is a
``buffer_store_dwordx4`` of the accumulator as it stands. Lane group
``g = lane // 16`` takes ``col = base_col + g * 4``, so the four groups still
cover 64 contiguous bytes of one row, which is the coalescing shape the bf16
path reaches only after the shuffle.
"""

# NOTE: no `from __future__ import annotations`, deliberately, for the reason
# _gemm_a8w8_8wave.py states: `fx.struct` reads the `SharedStorage` field
# annotations as live objects, and PEP 563 would hand it a string whose size
# operands are locals of this factory.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int64
from flydsl.expr.typing import Vector as Vec

import mori.cco.device.flydsl as cco
from mori.cco.device.flydsl import _bindings as raw_cco

from ._compat import (
    CM_CACHED,
    CM_SC0_SC1,
    atomic_add_u32,
    atomic_store_u32,
    buffer_store,
    create_buffer_resource_from_addr,
    signal_ptr,
    wave_uniform_i64,
)

# The A/B swap is gemm_ar's and applies unchanged -- it is a property of the
# MFMA's output layout, not of the operand precision. The kernel that uses this
# module takes G2SLoader, wait_barrier, ceildiv, split_row_major_2d and
# _xcd_swizzle_any from gemm_ar directly: all of them are byte-indexed or
# shape-only, so none carries an fp8 assumption, and re-exporting them here
# would only add a layer to trace through.
from ._gemm_a8w8_8wave import (
    G2SLoader,
    _xcd_swizzle_any,
    ceildiv,
    split_row_major_2d,
    wait_barrier,
)
from .kernels_fused import (
    _acquire_peer_lock,
    _permlane16_swap,
    _SwappedMfma,
)

#: The template's ``B_K``. Not a tuning knob: it sets ``E_K = B_K / W_K``, and
#: the mainloop below issues exactly ``E_K`` MFMAs per accumulator per step.
BLOCK_K = 64

#: ``W_M x W_N x W_K`` of ``v_mfma_f32_16x16x32_bf16``.
W_K = 32
E_K = BLOCK_K // W_K

#: Elements per 16-byte access. ``VEC_A == VEC_B == VEC_C == 8`` in the
#: template's traits, and 8 bf16 is also exactly one MFMA operand fragment.
VEC = 8

BF16_BYTES = 2
PACK_BYTES = 16


def _wait_lds_reads():
    """Finish S2R reads before the other half of the block can reuse LDS."""
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="s_waitcnt lgkmcnt(0)",
        constraints="~{memory}",
        has_side_effects=True,
    )


def swizzle_row128b(row, col, elem_bytes):
    """``gemm_ar.swizzle_128`` written in bytes instead of fp8 elements.

    The original reads::

        offset  = row * 128 + col
        swizzle = ((offset % (16 * 128)) >> 8) << 4
        offset ^= swizzle

    which is a 16-byte rotation selected by the 256-byte group, over a block of
    16 rows of 128 bytes. Spelling ``col`` in bytes makes that geometry explicit
    and dtype-independent; at ``elem_bytes == 1`` this is the original function
    value for value, which is checked in ``tests/python/cco/test_gemm_ag.py``.

    Returns ``(row, col)`` back in elements. The rotation is a multiple of 16
    bytes and ``elem_bytes`` divides 16, so it never lands mid-element.
    """
    byte_off = row * 128 + col * elem_bytes
    swizzle = ((byte_off % (16 * 128)) >> 8) << 4
    swizzled = byte_off ^ swizzle
    return swizzled // 128, (swizzled % 128) // elem_bytes


def compute_global_swizzle_bf16(lane_id, wave_id, K, n_rounds):
    """Global ``[*, K]`` offsets for one thread's share of a G2S round.

    ``gemm_ar.compute_global_swizzle``'s non-preshuffled branch, with ``col``
    scaled by this precision's ``VEC``. The swizzle is applied to the *global*
    address rather than to the LDS write, so the LDS store stays linear
    (``G2SLoader._lds_dst_at`` is a flat per-wave stride) and the data lands
    swizzled -- which is what ``S2RLoaderBf16`` then undoes.

    ``threads_k = BLOCK_K / VEC = 8`` here, the same 8 as fp8's ``128 / 16``, so
    the row map is unchanged and a wave still covers 8 rows per round.
    """
    threads_k = BLOCK_K // VEC
    offsets = []
    n_waves = fx.block_dim.x // 64
    rows_per_wave = 64 // threads_k
    for rnd in range_constexpr(n_rounds):
        row = (
            lane_id // threads_k
            + wave_id * rows_per_wave
            + rnd * (n_waves * rows_per_wave)
        )
        col = (lane_id % threads_k) * VEC
        r, c = swizzle_row128b(row, col, BF16_BYTES)
        offsets.append(r * K + c)
    return offsets


def make_bf16_buffer_tensor(arg_i16):
    """Flat int16 kernel argument -> a bf16 buffer tensor.

    Same recast as ``gemm_ar.make_fp8_buffer_tensor``, against an **int16**
    view rather than int8. That is load-bearing: the recast reuses the incoming
    tensor's layout, and a layout counted in bytes would describe twice as many
    bf16 elements as exist. Taking int16 in makes element counts agree on both
    sides of the recast.
    """
    bf16_ir_t = fx.BFloat16.ir_type
    t_i16 = fx.rocdl.make_buffer_tensor(arg_i16, max_size=False)
    iter_i16 = fx.get_iter(t_i16)
    bf16_ptr_ty = fx.PointerType.get(
        elem_ty=bf16_ir_t,
        address_space=fx.PointerType(iter_i16.type).address_space,
        alignment=fx.PointerType(iter_i16.type).alignment,
    )
    iter_bf16 = fx.recast_iter(bf16_ptr_ty, iter_i16)
    return fx.Tensor(fx.make_view(iter_bf16, fx.get_layout(t_i16)))


class S2RLoaderBf16:
    """LDS -> registers, ``E_K`` operand fragments per 16-row tile.

    ``gemm_ar``'s ``S2RLoader`` builds one 32-byte fp8 operand out of two 16-byte
    loads (``pack_i32x4_i32x8``). A bf16 ``16x16x32`` operand is 16 bytes, so
    there is nothing to pack -- instead there are ``E_K`` of them per tile,
    because ``B_K`` spans two of the instruction's K steps.

    Operand layout for ``v_mfma_f32_16x16x32_bf16``: lane ``l`` supplies
    ``A[l % 16][k_step * 32 + (l / 16) * 8 + 0..7]``.
    """

    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles

    def _vec_load_16b(self, lds_src, elem_offset):
        """16 bytes at ``elem_offset`` bf16 elements into ``lds_src``.

        ``elem_offset`` is in **elements, not bytes**. ``fx.add_offset``
        advances a pointer by its pointee type, and ``lds_src.ptr`` is a
        ``BFloat16*`` -- so scaling by the element size here double-counts.
        gemm_ar's fp8 version passes an element offset too; the bug is
        invisible there because an fp8 element *is* a byte, which is exactly
        why it was easy to introduce when porting. Doubling the offset walks
        every read a row and a half past where the G2S wrote, and because A and
        B span different row ranges it desynchronises their K contraction
        rather than merely shifting it -- the symptom was C cells matching no
        single A element under an identity B.
        """
        off_tup = fx.make_int_tuple(elem_offset)
        ptr_off = fx.add_offset(lds_src.ptr, off_tup)
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        view = fx.make_view(i8_iter, fx.make_layout(PACK_BYTES, 1))
        return view.load()

    def load(self, lds_src):
        """``[tile][k_step]`` fragments, each an i32x4."""
        frag = []
        for i in range_constexpr(self.n_tiles):
            row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane_id % 16
            steps = []
            for s in range_constexpr(E_K):
                col = s * W_K + (self.lane_id // 16) * VEC
                row_swz, col_swz = swizzle_row128b(row, col, BF16_BYTES)
                offset = row_swz * BLOCK_K + col_swz
                steps.append(self._vec_load_16b(lds_src, offset).bitcast(fx.Int32))
            frag.append(steps)
        return frag


class MfmaBf16:
    """``v_mfma_f32_16x16x32_bf16``, A/B swapped, ``E_K`` steps per call.

    The swap is ``_SwappedMfma``'s and is reused unchanged: ``fx.gemm(atom, c,
    b, a, c)`` computes ``B^T A^T = (A B)^T``, which puts four consecutive
    output *columns* in a lane instead of four rows. That is what makes any
    vectorised C store possible at all, and it is what the FP32 store below
    exploits to skip the shuffle entirely.
    """

    def __init__(self, n_tiles_a, n_tiles_b):
        self.atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, W_K, fx.BFloat16))
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _make_operand_frag(self, value):
        # Four dwords: 8 bf16, one MFMA operand. fp8's is eight.
        frag = fx.make_rmem_tensor(4, fx.Int32)
        frag.store(Vec(value))
        return frag

    def _make_accum_frag(self, value):
        frag = fx.make_rmem_tensor(4, fx.Float32)
        frag.store(Vec(value))
        return frag

    def call(self, a, b, c, *, set_prio=True, scale_a=None, scale_b=None):
        """``a``/``b`` are ``[tile][k_step]``; ``c`` is flat over ``idx(i, j)``.

        ``scale_a``/``scale_b`` are accepted and must be ``None``. They exist
        only because ``_SwappedMfma.call`` -- which is gemm_ar's, merged, and
        reused here unchanged -- forwards them unconditionally. bf16 is not a
        quantised format, so a non-None scale is a caller bug rather than an
        unsupported feature, and saying so here is better than the wrapper
        growing a precision-dependent branch.
        """
        assert scale_a is None and scale_b is None, "the bf16 GEMM has no scales"
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b

        if const_expr(set_prio):
            rocdl.s_setprio(1)
        # One K step per pass, chaining through the *returned values* rather
        # than issuing E_K gemms against one accumulator memref. Two
        # back-to-back ``fx.gemm(atom, cf, ..., cf)`` calls do not accumulate:
        # the second overwrites, which measured as exactly half the expected
        # value on a K-sweep probe (256 against 512 at K=512 -- one step per
        # block surviving, times eight blocks). Rebuilding the fragment from
        # the previous step's loaded value is what gemm_ar's single-step call
        # does implicitly, and doing it explicitly here is what makes E_K > 1
        # correct.
        out = list(c)
        for s in range_constexpr(E_K):
            out = self._call_step(a, b, out, s)
        if const_expr(set_prio):
            rocdl.s_setprio(0)
            rocdl.s_barrier()
        return out

    def _call_step(self, a, b, c, s):
        """One MFMA per (M-tile, N-tile) over K step ``s``."""
        a_frags = [
            self._make_operand_frag(a[i][s]) for i in range_constexpr(self.n_tiles_a)
        ]
        b_frags = [
            self._make_operand_frag(b[j][s]) for j in range_constexpr(self.n_tiles_b)
        ]
        c_frags = [
            self._make_accum_frag(c[idx])
            for idx in range_constexpr(self.n_tiles_a * self.n_tiles_b)
        ]
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                cf = c_frags[self.idx(i, j)]
                fx.gemm(self.atom, cf, a_frags[i], b_frags[j], cf)
        return [
            c_frags[idx].load().ir_value()
            for idx in range_constexpr(self.n_tiles_a * self.n_tiles_b)
        ]


class AgStoreC:
    """C store for the bf16 GEMM, in bf16 or fp32, local or broadcast.

    Deliberately **not** a subclass of ``gemm_ar``'s ``StoreC``, for two reasons
    that are both correctness rather than taste:

    * ``StoreC.__init__`` sizes its C buffer descriptor as
      ``c_rows * c_cols * 2  # BFloat16``. Under an fp32 store that clamps the
      descriptor at half the real extent and silently drops everything past it
      -- the same shape of bug ``gemm_a2a`` records for its B-scale descriptor,
      which also only appeared on one configuration.
    * It builds A-scale and B-scale descriptors unconditionally, and a bf16
      GEMM has no scales to describe.

    ``peer_rsrc`` is the fused path's destination, this rank's ``recv`` slot,
    stored through a raw buffer descriptor so the cache policy can be set;
    without it the store goes through the C tensor. ``elem_base`` rebases the
    flat index when the descriptor is a slab rather than the whole ``[M, N]``.
    """

    def __init__(
        self,
        C,
        c_rows,
        c_cols,
        c_idx_fn,
        n_tiles_a,
        n_tiles_b,
        *,
        out_dtype="bf16",
        peer_rsrc=None,
        elem_base=None,
        peer_uncached=False,
    ):
        if out_dtype not in ("bf16", "fp32"):
            raise ValueError(f"out_dtype must be bf16 or fp32, got {out_dtype!r}")
        self.out_dtype = out_dtype
        self.elem_bytes = 2 if out_dtype == "bf16" else 4
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.lane_id = fx.thread_idx.x % 64
        self.peer_rsrc = peer_rsrc
        self.elem_base = fx.Int32(0) if elem_base is None else elem_base
        self.cache_modifier = CM_SC0_SC1 if peer_uncached else CM_CACHED

        if self.peer_rsrc is None:
            # Local path: a copy atom through the C tensor, as gemm_ar does.
            # Sized from the *real* element width, which is the thing StoreC
            # gets wrong for fp32.
            c_nbytes = c_rows * c_cols * self.elem_bytes
            gC = fx.rocdl.make_buffer_tensor(
                C, max_size=False, num_records_bytes=c_nbytes
            )
            self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
            if out_dtype == "bf16":
                self.out_atom = fx.make_copy_atom(
                    fx.rocdl.BufferCopy128b(), fx.BFloat16
                )
                self.reg_out = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
            else:
                self.out_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
                self.reg_out = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)

    def _emit(self, value, idx):
        """One 16-byte store of ``value`` at flat element index ``idx``."""
        if self.peer_rsrc is not None:
            buffer_store(
                value,
                self.peer_rsrc,
                fx.Int32(idx) - self.elem_base,
                cache_modifier=self.cache_modifier,
            )
        else:
            fx.memref_store_vec(value, self.reg_out)
            fx.copy(
                self.out_atom,
                self.reg_out,
                fx.slice(self.c_div, (None, fx.Int32(idx))),
            )

    def store(self, c_frag, base_row, base_col):
        if const_expr(self.out_dtype == "fp32"):
            self._store_fp32(c_frag, base_row, base_col)
        else:
            self._store_bf16(c_frag, base_row, base_col)

    def _store_fp32(self, c_frag, base_row, base_col):
        """No shuffle, no convert: the accumulator is already the payload.

        Lane ``l`` holds four consecutive columns starting at ``4 * (l / 16)``
        of its N-tile, and at four bytes each that is one ``dwordx4``. The two
        N-tiles are 16 columns apart and become two independent stores, where
        the bf16 path pairs them into one.
        """
        lane = self.lane_id
        grp = lane // 16
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + lane % 16
            for tj in range_constexpr(self.n_tiles_b):
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                col = base_col + tj * 16 + grp * 4
                oob = fx.Int32(self.c_rows * self.c_cols)
                idx = arith.select(col + 3 < self.c_cols, row * self.c_cols + col, oob)
                self._emit(vec_f32, idx)

    def _store_bf16(self, c_frag, base_row, base_col):
        """``_PermlaneStoreC._emit``'s shuffle, without the scale multiply.

        Four bf16 per lane is 8 bytes; two ``permlane16_swap`` exchange 16-lane
        rows between the two N-tiles so a lane ends up with 16 bytes covering 8
        columns, and lane group ``g`` lands on columns ``(g%2)*16 + (g//2)*8``.
        The four groups together cover 32 columns contiguously.
        """
        assert self.n_tiles_b == 2, (
            "the permlane mapping pairs exactly two N-tiles (BLOCK_N == 256); "
            f"got n_tiles_b={self.n_tiles_b}"
        )
        lane = self.lane_id
        grp = lane // 16
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + lane % 16
            dwords = []
            for tj in range_constexpr(self.n_tiles_b):
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                packed = Vec.from_elements(
                    [vec_f32[k].to(fx.BFloat16) for k in range_constexpr(4)],
                    fx.BFloat16,
                ).bitcast(fx.Int32)
                dwords.append((packed[0], packed[1]))
            a, b = _permlane16_swap(dwords[0][0], dwords[1][0])
            c, d = _permlane16_swap(dwords[0][1], dwords[1][1])
            out8 = Vec.from_elements([a, c, b, d], fx.Int32).bitcast(fx.BFloat16)
            col = base_col + (grp % 2) * 16 + (grp // 2) * 8
            oob = fx.Int32(self.c_rows * self.c_cols)
            idx = arith.select(col + 7 < self.c_cols, row * self.c_cols + col, oob)
            self._emit(out8, idx)


def tile_constants(BLOCK_M, BLOCK_N, out_dtype="bf16"):
    """The derived tile geometry, shared by every caller of this GEMM.

    Every formula here is unchanged from ``_gemm_a8w8_8wave.py`` including
    ``N_LDS_STEPS_*``, which looks like it should depend on precision and does
    not: it is ``LDS_BLOCK * BLOCK_K / (512 * VEC)``, and bf16 halves ``BLOCK_K``
    and ``VEC`` together.

    **The tile is the dominant tuning knob at these shapes, and not because of
    the inner loop.** The template's ``256x256`` launches
    ``ceil(M/256) * ceil(N/256)`` workgroups, which at the ``wkv_gate`` shape
    (M=2048, N=2048) is 64 -- against 256 CUs on an MI355X, so three quarters of
    the GPU is idle no matter how good the mainloop is. Halving a tile dimension
    doubles the grid.

    ``BLOCK_N=128`` is available **only for an fp32 C**, and the reason is the
    store rather than the GEMM: the bf16 store reaches a 16-byte access by
    pairing two N-tiles through ``permlane16_swap``, so it needs
    ``n_tiles_b == 2``. An fp32 lane already holds 16 bytes of one tile and
    stores each independently, so it does not care.
    """
    if BLOCK_M not in (128, 256):
        raise ValueError(
            f"BLOCK_M must be 128 or 256, got {BLOCK_M}: 256 is the template's "
            f"instantiation and 128 doubles the grid; other values have no "
            f"validated pipeline"
        )
    if BLOCK_N not in (128, 256):
        raise ValueError(f"BLOCK_N must be 128 or 256, got {BLOCK_N}")
    if BLOCK_N == 128 and out_dtype != "fp32":
        raise ValueError(
            f"BLOCK_N=128 needs out_dtype='fp32', got {out_dtype!r}: the bf16 "
            f"store pairs two N tiles with permlane16_swap to reach a 16-byte "
            f"access, and there is only one N tile at this width"
        )
    n_tiles_a = BLOCK_M // 64
    n_tiles_b = BLOCK_N // 128
    lds_block_m = BLOCK_M // 2
    lds_block_n = BLOCK_N // 2
    return dict(
        N_TILES_A=n_tiles_a,
        N_TILES_B=n_tiles_b,
        N_ACCUMS=n_tiles_a * n_tiles_b,
        LDS_BLOCK_M=lds_block_m,
        LDS_BLOCK_N=lds_block_n,
        N_LDS_STEPS_A=lds_block_m // 64,
        N_LDS_STEPS_B=lds_block_n // 64,
        a_lds_size=lds_block_m * BLOCK_K,
        b_lds_size=lds_block_n * BLOCK_K,
    )
