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
"""The 8-wave quad-subtile BF16 GEMM, with a BF16 or FP32 C store.

This is the precision the model actually uses. ``gemm_ar``'s GEMM -- which
``gemm_ag`` was first built on -- is fp8 in, bf16 out; DeepSeek V4-Pro's
``wkv_gate`` goes through ``linear_bf16_fp32``, so an operator meant to replace
that layer has to be bf16 in and fp32 out, and its numbers only mean something
next to an fp32 baseline.

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

So the pipeline -- the ``s_barrier`` ladder, the ``wait_barrier`` counts, the
double-buffered tic/toc, the tail's two unrolled K steps -- is *not* re-derived
here. It is the same schedule with four things changed:

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
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int64
from flydsl.expr.typing import Vector as Vec

import mori.cco.device.flydsl as cco
from mori.cco.device.flydsl import _bindings as raw_cco

from ..gemm_ar._compat import (
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
from ..gemm_ar._gemm_a8w8_8wave import (
    G2SLoader,
    _xcd_swizzle_any,
    ceildiv,
    split_row_major_2d,
    wait_barrier,
)
from ..gemm_ar.kernels_fused import (
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

    ``peer_rsrcs`` is the broadcast list ``fused-lsa`` needs: one tile goes to
    every rank, and doing that through ``world`` separate store objects would
    repeat the convert and the shuffle with it. ``elem_base`` rebases the flat
    index when the descriptor is a slab rather than the whole ``[M, N]``.
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
        peer_rsrcs=None,
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
        self.peer_rsrcs = () if peer_rsrcs is None else tuple(peer_rsrcs)
        self.elem_base = fx.Int32(0) if elem_base is None else elem_base
        self.cache_modifier = CM_SC0_SC1 if peer_uncached else CM_CACHED

        if not self.peer_rsrcs:
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
                self.out_atom = fx.make_copy_atom(
                    fx.rocdl.BufferCopy128b(), fx.Float32
                )
                self.reg_out = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)

    def _emit(self, value, idx):
        """One 16-byte store of ``value`` at flat element index ``idx``."""
        if self.peer_rsrcs:
            # A single resource is one iteration and emits what an unlooped
            # store would; the plural exists for fused-lsa's broadcast.
            for rsrc in self.peer_rsrcs:
                buffer_store(
                    value,
                    rsrc,
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
                idx = arith.select(
                    col + 3 < self.c_cols, row * self.c_cols + col, oob
                )
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
    transport: str = "sdma",
    wait_policy: str = "tuned",
    fuse: bool = False,
    chunks: int = 1,
    sdma_queues: int = 1,
    post: str = "lanes",
    peer_uncached: bool = False,
    direct_fence: str = "leader",
    emit_put: bool = True,
    fence: str = "leader",
):
    """The bf16 GEMM, with or without the all-gather fused into its epilogue.

    Returns ``launch(A, B_T, C, c_m, c_n, dev_comm, win, stream=...)``. ``A``
    and ``B_T`` are flat **int16** views of bf16 ``[M, K]`` and ``[N, K]`` --
    see ``make_bf16_buffer_tensor`` for why int16 rather than int8. There are no
    scale arguments, unlike ``compile_fp8_gemm_8w``: nothing to dequantise.

    ``fuse=False`` emits the identical kernel minus the epilogue tail and writes
    C wherever the caller points it -- a plain tensor for ``gemm-only``, this
    rank's ``recv`` slot for ``gemm-to-window`` and the split transports. That
    is the same discipline ``gemm_ar.compile_fused_gemm_scatter`` uses and for
    the same reason: the split baseline and the fused kernel then differ in
    exactly one thing. ``dev_comm`` and ``win`` are unread on that path but stay
    in the signature so the launch path does not change either.

    **The epilogue below is the twin of the fp8 one in ``kernels_fused.py`` and
    the two must be changed together.** They are not shared because FlyDSL's
    rewriter only lowers ``if`` inside a ``@flyc.kernel`` function's own AST, so
    the election and the barriers cannot move into a helper. What differs
    between them is the mainloop and the store; the transport logic -- release
    fence, per-(chunk, lane) counter, lane-parallel put, ``peer_rsrcs``
    broadcast -- is meant to be identical, and any fix to one belongs in both.

    Validate ``fuse=False`` before trusting anything fused: a wrong LDS swizzle
    or a wrong ``E_K`` unroll produces a wrong C, and every fused mode validates
    what *arrived*, so without that control a GEMM bug and a transport bug are
    indistinguishable.
    """
    assert K % BLOCK_K == 0, f"K={K} must be a multiple of BLOCK_K={BLOCK_K}"
    if wait_policy not in ("tuned", "safe", "conservative"):
        raise ValueError(
            f"wait_policy must be tuned, safe or conservative, got {wait_policy!r}"
        )
    if transport not in ("lsa", "sdma"):
        raise ValueError(f"transport must be lsa or sdma, got {transport!r}")
    if post not in ("lanes", "serial"):
        raise ValueError(f"post must be lanes or serial, got {post!r}")
    if fence not in ("none", "leader", "all"):
        raise ValueError(f"fence must be none, leader or all, got {fence!r}")
    if direct_fence not in ("leader", "all"):
        raise ValueError(f"direct_fence must be leader or all, got {direct_fence!r}")
    if transport == "lsa" and not fuse:
        raise ValueError(
            "fuse=False is only meaningful with transport='sdma', where it gives "
            "the window-writing GEMM for the split paths; an unfused LSA GEMM "
            "writes nowhere in particular"
        )
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
    direct_lsa = transport == "lsa"
    if fuse and not direct_lsa and cfg.counter_chunks != chunks:
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
    K_ITERS = K // BLOCK_K
    assert K_ITERS >= 2, "the pipeline peels two K steps off the tail"

    # How permissive the mainloop's `s_waitcnt vmcnt(N)` is allowed to be.
    #
    # Each G2SLoader.load() issues exactly one buffer_load_lds per step -- A =
    # N_LDS_STEPS_A of them for an A tile, B for a B tile -- and vmcnt retires
    # in issue order. Walking the mainloop: entering an iteration A+2B are
    # outstanding, the iteration issues A1@k+1, B0@k+2, A0@k+2, B1@k+2, so 3A+4B
    # are in flight at the wait. The next iteration reads a_cur1, which is
    # A1@k+1 -- the 4th item -- so retirement has to reach 2A+2B and the wait
    # must therefore be **at most A + 2B**.
    #
    # The inherited count is `2A + B`, and the two are equal only when A == B:
    #
    #     tile      A  B   2A+B   A+2B
    #     256/256   2  2      6      6   exactly tight
    #     128/256   1  2      4      5   has margin
    #     256/128   2  1      5      4   INSUFFICIENT
    #     128/128   1  1      3      3   exactly tight
    #
    # "Exactly tight" means zero slack against the compiler reordering the four
    # loads among themselves, which it is free to do -- `wait_barrier` is a
    # scheduling barrier but does not fix the order of what precedes it.
    #
    # "safe" uses the derived A+2B. "conservative" waits for everything, which
    # is slow and exists to answer "is it the waits?" without having to trust
    # the derivation.
    #
    # It is **not** the waits. The generated ISA says so directly
    # (GPU_DUMP_CODE_OBJECT=1, llvm-objdump --mcpu=gfx950), at M=2048 N=2048
    # K=7168 fp32:
    #
    #   * the S2R loads are `ds_read_b128` -- lgkmcnt, not vmcnt -- so they were
    #     never in this accounting to begin with: 16 per iteration at BM=128,
    #     24 at BM=256;
    #   * the G2S loads are `buffer_load_dwordx4`, exactly A+B per load call, 6
    #     per iteration at BM=128 and 8 at BM=256;
    #   * `vmcnt(4)` / `vmcnt(6)` each retire precisely one iteration's worth,
    #     leaving in flight exactly the k+2 prefetches, which is correct;
    #   * the two kernels' instruction skeletons are structurally identical
    #     (`W BB D..D G B L B D..D GG B L B D..D G B L B GG`), with 8 barriers
    #     per iteration in both and `s_waitcnt lgkmcnt(0)` correctly between the
    #     barrier pair that guards the write-after-read on each LDS buffer;
    #   * neither spills (0 VGPR, 0 SGPR), and both get one workgroup per CU.
    #
    # The one asymmetry is 152 VGPRs and 96 KiB LDS at BM=128 against 248 and
    # 128 KiB at BM=256, which is just the tile. `gemm_ar.wait_barrier` also
    # puts `s_waitcnt` and `s_barrier` in one inline-asm string with
    # `constraints=""` -- no memory clobber, so the compiler is not told there
    # is a barrier in there and could legally move an LDS access across it.
    # Emitting the wait as asm and the barrier as `rocdl.s_barrier()` instead
    # produces **byte-identical** code here, so it does not, at least not at
    # this register pressure.
    #
    # So the instruction stream gives no evidence of a bug at either tile, and
    # `conservative` fixing the corruption is timing perturbation rather than a
    # corrected count. Together with 0/64 clean on a single idle GPU, that
    # points away from this kernel and toward the 8-process harness or the
    # environment -- which is where the next person should start, not here.
    def _wb(tuned):
        if wait_policy == "conservative":
            return wait_barrier(0)
        return wait_barrier(tuned)

    _MAIN_WAIT = (
        N_LDS_STEPS_A + 2 * N_LDS_STEPS_B
        if wait_policy in ("safe", "conservative")
        else 2 * N_LDS_STEPS_A + N_LDS_STEPS_B
    )

    _kname = (
        f"mori_ag_bf16_{transport if fuse else 'plain'}_8w_"
        f"{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_"
        f"{'F32' if out_dtype == 'fp32' else 'B16'}_{waves_per_eu}x{xcd_swizzle}"
        f"_c{chunks}q{sdma_queues}{post[0]}{'U' if peer_uncached else 'C'}"
        f"w{wait_policy[0]}"
        f"{direct_fence[0]}{fence[0]}{'p' if emit_put else 'x'}_k{K}_r{rank}"
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
        # Where C goes. On the fused-LSA path the collective is a *broadcast*,
        # so the store is handed `world` descriptors and does the convert and
        # (for bf16) the shuffle once for all of them. Everywhere else it is a
        # single destination and the store takes the plain C tensor.
        if const_expr(fuse and direct_lsa):
            # Broadcast: `world` descriptors, so the convert and (for bf16) the
            # shuffle happen once and only the 16-byte store repeats.
            rsrcs = [
                create_buffer_resource_from_addr(
                    wave_uniform_i64(w_pre.lsa_ptr((rank + j) % ws, my_recv_slot)),
                    num_records_bytes=slab_bytes,
                )
                for j in range(ws)
            ]
        elif const_expr(fuse):
            # One descriptor at this rank's own recv slot, reached through
            # lsa_ptr(rank, ...) -- structurally what gemm_ar's fused-SDMA path
            # does. Routing the fused-SDMA store through the C *tensor* instead
            # measured as a wrong answer at M=2048 fp32 (relL2 6.9e-3 against
            # 9.9e-7) and was clean through the buffer resource, so the two
            # paths are not interchangeable even though they address the same
            # bytes.
            #
            # `peer_uncached` is deliberately **not** honoured on the unfused
            # path, and that is a known soft spot rather than a decision:
            # sending the split pull baseline's C store with sc0|sc1 through
            # this resource made it *worse*, not better -- relL2 5.9e-3 at
            # BLOCK_M 128 and 256 alike, where the plain cached copy-atom store
            # validates. So the bf16 pull baseline currently has no explicit
            # system-scope publish and relies on the end-of-kernel release,
            # which is not the discipline kernels_lsa's docstring asks for.
            # See the benchmark doc; it is unresolved.
            rsrcs = [
                create_buffer_resource_from_addr(
                    wave_uniform_i64(w_pre.lsa_ptr(rank, my_recv_slot)),
                    num_records_bytes=slab_bytes,
                )
            ]
        else:
            rsrcs = None
        store_c = AgStoreC(
            C,
            c_m,
            c_n,
            mfma.idx,
            N_TILES_A,
            N_TILES_B,
            out_dtype=out_dtype,
            peer_rsrcs=rsrcs,
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
        rocdl.s_barrier()

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
        wait_barrier(0)

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

        # ---- the all-gather epilogue; twin of kernels_fused.py's fp8 one ----
        if const_expr(fuse and direct_lsa):
            # Publish the peer stores from the blocks that made them. The
            # barrier kernel that follows is one block, hence one XCD's L2 out
            # of eight; the other seven would keep their peer-homed lines dirty
            # and the barrier's system-scope atomic would overtake them.
            wait_barrier(0)
            raw_cco.cco_system_fence(fx.Int32(1 if direct_fence == "leader" else 0))
        elif const_expr(fuse):
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
                seq = fx.Int32(atomic_add_u32(ctr, 1, ordering="acq_rel")) + fx.Int32(
                    1
                )
                if seq % fx.Int32(tiles_per_chunk) == fx.Int32(0):
                    if const_expr(emit_put):
                        if const_expr(post == "lanes"):
                            d = (fx.Int32(rank + 1) + j) % fx.Int32(ws)
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
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_gemm


#: Tiles ``pick_tile`` will choose from, largest first.
#:
#: ``BLOCK_N=128`` is **deliberately absent**, and not because it is slow -- it
#: is the fastest thing measured (116.5us at M=2048 against ``128/256``'s
#: 134.2). It **races**: at ``BLOCK_M=128, BLOCK_N=128`` an fp32 C came out
#: wrong once in 48 single-GPU runs (relL2 1.4e-2 against the usual 9.9e-7), and
#: under an 8-rank bench roughly half the ranks failed per launch, a different
#: half each time. ``BLOCK_N=256`` is 48/48 clean on the same probe.
#:
#: The suspected cause is the ``wait_barrier`` vmcnt thresholds: they are
#: written as ``N_LDS_STEPS_A + N_LDS_STEPS_B`` and friends, which is only ever
#: exercised at the step counts ``BLOCK_N=256`` produces. At ``BLOCK_N=128``
#: both counts fall to 1 and the thresholds may admit one prefetch too many --
#: which would be invisible until something perturbs the timing, exactly as
#: observed. Unconfirmed; ``tile_constants`` still accepts the tile so the
#: measurement stays reproducible behind an explicit ``--block-n 128``.
#:
#: ``BLOCK_M=128`` is **also absent, and that is the unhappy part of this
#: file.** It is 24% faster at the target shape and it is not sound either: at
#: ``M=2048`` fp32 it validated 4/4 on ``gemm-only`` and ``gemm-to-window`` and
#: 3/3 on ``split-rccl`` and ``split-sdma``, then on a later pass the same
#: ``gemm-to-window`` and ``split-rccl`` cells came back 1/2, and ``fused-lsa``
#: is 0/4 throughout. ``BLOCK_M=256`` has not failed once across every repeat
#: run here. So the failure rate at 128 is around a third of launches and
#: depends on something not yet identified -- it is **not** the release fence,
#: which was the obvious candidate and which ``--fence all`` and
#: ``--direct-fence all`` both fail to fix (0/3 each).
#:
#: Shipping a wrong answer a third of the time to buy 24% is not a trade, so
#: the default is the tile that validates. Everything measured is in
#: ``pick_tile`` and both faster tiles stay reachable through an explicit
#: ``--block-m`` / ``--block-n``, so the work is reproducible rather than lost.
TILE_CANDIDATES = ((256, 256),)

#: CUs on the part this was tuned on (MI355X). The grid heuristic wants the
#: real number, so it is a parameter of ``pick_tile`` rather than a constant.
DEFAULT_CUS = 256


def pick_tile(m, n, out_dtype="bf16", cus=DEFAULT_CUS):
    """Choose ``(BLOCK_M, BLOCK_N)`` for a shape. Measured, not guessed.

    The template instantiates ``256x256``, and at the shapes this operator runs
    that is the **single worst choice on the table** -- it launches
    ``ceil(M/256) * ceil(N/256)`` workgroups, which at ``M=N=2048`` is 64
    against an MI355X's 256 CUs. Three quarters of the GPU idles regardless of
    how good the mainloop is, and no amount of ``waves_per_eu`` or
    ``xcd_swizzle`` touches it (both measured inert, within 1%).

    Measured at ``M=2048 N=2048 K=7168``, fp32 out, across every tile::

        tile      grid      us    TF/s
        256/256     64   176.0     342
        256/128    128   139.6     431
        128/256    128   134.2     448
        128/128    256   116.5     516   (races -- see TILE_CANDIDATES)

    and per shape at the best *sound* tile (``128/256``), against
    ``torch.matmul``::

          M   grid     us    TF/s   torch    ratio
        512     32  128.4     117    39.0    3.29x
       1024     64  128.4     234    53.1    2.42x
       2048    128  134.2     448    84.8    1.58x
       4096    256  155.2     775   122.9    1.26x

    The rule that fits those measurements is **not** "maximise the grid": at
    ``M=4096``, ``128/128`` gives 512 workgroups and measures 162.8us against
    ``128/256``'s 256 workgroups at 155.2, so once there is at least one
    workgroup per CU the larger tile amortises better. Hence **the largest tile
    whose grid still covers the CUs, or the largest grid available if none
    does**, which reproduces the measured best at every shape.

    **That rule currently has one candidate to choose from.** Both tiles
    smaller than the template's are measurably faster and neither validates
    reliably -- see ``TILE_CANDIDATES`` -- so this returns ``(256, 256)`` today
    and the logic is kept for when one of them is fixed. The measurements are
    the deliverable here rather than the selection: they say the tile is worth
    24-34%, and that ``waves_per_eu`` and ``xcd_swizzle`` are worth nothing,
    which is where the next attempt should and should not look.
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
    grid = lambda t: (m // t[0]) * (n // t[1])  # noqa: E731
    covering = [t for t in cands if grid(t) >= cus]
    if covering:
        # Largest area among those that fill the machine; TILE_CANDIDATES is
        # ordered largest-first so the first match is it.
        return covering[0]
    return max(cands, key=grid)


def tile_grid(m, n, tile):
    """Workgroups a tile launches. Exposed because it is the whole story."""
    return (-(-m // tile[0])) * (-(-n // tile[1]))


__all__ = [
    "AgStoreC",
    "BLOCK_K",
    "compile_bf16_gemm_ag",
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
