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
"""Building blocks of the 8-wave quad-subtile BF16 GEMM, for ``gemm_ar_v2``.

Copied, not imported: importing anything under ``mori.ops.gemm_ar`` (or the sibling
operators that build on it) also imports modules this package must not depend on.
Sources, copied with their comments:

* ``mori/ops/gemm_ar/_gemm_a8w8_8wave.py`` (vendored from aiter, Apache-2.0,
  Copyright (c) 2025 FlyDSL Project Contributors, Modifications Copyright (C) 2026
  Advanced Micro Devices, Inc.): ``split_row_major_2d``, ``ceildiv``, ``_min``,
  ``wait_barrier``, ``read_done_barrier``, ``G2SLoader``, ``_xcd_swizzle_any``.
* ``mori/ops/gemm_ar/_gemm_a16w16_8wave.py``: ``BLOCK_K`` .. ``tile_constants``
  (the bf16 loaders, MFMA, C store), unchanged except the C store's lane swap.
* ``mori/ops/gemm_ar/_compat.py``: ``wave_uniform_i64``, ``atomic_add_u32``.

Written here: ``permlane16_swap`` (ROCDL ``permlane16.swap``), ``SwappedMfma``.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_d
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from mori.cco.device.flydsl.buffer import CM_CACHED, CM_SC0_SC1, buffer_store
from mori.cco.device.flydsl.signals import _AGENT_SCOPE, i32_type

def split_row_major_2d(index, minor_extent):
    """Split a linear row-major index into (major, minor).

    Inlined from aiter ``mfma_preshuffle_pipeline.split_row_major_2d``.
    """
    return index // minor_extent, index % minor_extent


def ceildiv(numer, denom):
    """Ceiling division; works on Python ints and on DSL scalars alike.

    Unlike an int-only ``cdiv``, this one is also applied to runtime
    ``fx.Int32`` values, through operator overloading.
    """
    return (numer + denom - 1) // denom


def _min(a, b):
    return arith.select(a < b, a, b)


def wait_barrier(count):
    """``s_waitcnt vmcnt(count)`` then ``s_barrier``.

    The VM wait constrains only the requests *this* wave issued; the barrier is
    the workgroup rendezvous. A bare ``s_barrier`` is neither -- it drains
    nothing, so it cannot stand in for a completion wait.
    """
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt vmcnt({count})\ns_barrier",
        constraints="",
        has_side_effects=True,
    )


def read_done_barrier():
    """``s_waitcnt lgkmcnt(0)`` then ``s_barrier``: LDS reads done before refill.

    LDS reads retire under LGKM, which the VM waits in this pipeline say nothing
    about. The memory clobber keeps the compiler from sinking a DS read past
    the barrier.
    """
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
        constraints="~{memory}",
        has_side_effects=True,
    )


class G2SLoader:
    def __init__(self, gl_src, gl_offsets, n_load_steps, lds_dtype, wave_id):
        self.g2lds_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.LdsPtr_t = fx.PointerType.get(lds_dtype, 2, 512)
        self.gl_src = gl_src
        self.gl_offsets = gl_offsets
        self.n_load_steps = n_load_steps
        self.wave_id = wave_id
        self.n_waves = fx.block_dim.x // 64

    def _lds_dst_at(self, lds_dst, step):
        step_off = self.wave_id * 1024 + step * (self.n_waves * 1024)
        base_i32 = fx.Int32(fx.ptrtoint(lds_dst.ptr))
        sum_i32 = base_i32 + fx.Int32(step_off)
        lds_ptr = fx.inttoptr(self.LdsPtr_t, sum_i32)
        return fx.make_view(lds_ptr, fx.make_layout(1, 1))

    def load(self, lds_dst, k_offset):
        for step in range_constexpr(self.n_load_steps):
            src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
            dst = self._lds_dst_at(lds_dst, step)
            fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(k_offset))

    def load_one(self, lds_dst, k_offset, step):
        src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
        dst = self._lds_dst_at(lds_dst, step)
        fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(k_offset))


def _xcd_swizzle_any(num_pid_m, num_pid_n, wgm):
    """XCD-aware tile remap that works for any workgroup count. This function is used to swizzle the tile indices to improve the L2 reuse."""
    NUM_XCDS = 8

    wgid = fx.block_idx.x
    num_wg = num_pid_m * num_pid_n

    # Hardware dispatches workgroup i to XCD (i % NUM_XCDS). Invert that so
    # logically adjacent tiles share an XCD, and thus its L2.
    xcd = wgid % NUM_XCDS
    intra_xcd = wgid // NUM_XCDS
    base = num_wg // NUM_XCDS
    extra = num_wg % NUM_XCDS
    wgid_remap = xcd * base + _min(xcd, extra) + intra_xcd

    # Group-of-WGM row swizzle for L2 reuse within one XCD.
    num_wgid_in_group = wgm * num_pid_n
    group_id, intra_group = split_row_major_2d(wgid_remap, num_wgid_in_group)
    first_pid_m = group_id * wgm
    group_size_m = _min(num_pid_m - first_pid_m, wgm)
    pid_n, intra_group_m = split_row_major_2d(intra_group, group_size_m)
    return first_pid_m + intra_group_m, pid_n

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
        swap=True,
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
        # gemm_ar_v2: swap=False stores each N tile's 4 bf16 as one 8-byte store
        # instead of pairing two tiles through permlane16_swap (fallback only).
        self.swap = swap

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
        if const_expr(not self.swap):
            for ti in range_constexpr(self.n_tiles_a):
                row = base_row + ti * 16 + lane % 16
                for tj in range_constexpr(self.n_tiles_b):
                    vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                    out4 = Vec.from_elements(
                        [vec_f32[k].to(fx.BFloat16) for k in range_constexpr(4)],
                        fx.BFloat16,
                    )
                    col = base_col + tj * 16 + grp * 4
                    oob = fx.Int32(self.c_rows * self.c_cols)
                    idx = arith.select(
                        col + 3 < self.c_cols, row * self.c_cols + col, oob
                    )
                    self._emit(out4, idx)
            return
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
            a, b = permlane16_swap(dwords[0][0], dwords[1][0])
            c, d = permlane16_swap(dwords[0][1], dwords[1][1])
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


# ---- written for gemm_ar_v2 ---------------------------------------------------


def wave_uniform_i64(addr):
    """Force a 64-bit address into scalar registers (from ``gemm_ar/_compat.py``):
    a buffer descriptor built from a value the compiler believes divergent is
    wrapped in a readfirstlane waterfall loop around every access."""
    v = fx.Uint64(addr)
    lo = rocdl.readfirstlane(i32_type(), fx.Uint32(v))
    hi = rocdl.readfirstlane(i32_type(), fx.Uint32(v >> 32))
    return (fx.Uint64(hi) << 32) | fx.Uint64(lo)


def atomic_add_u32(ptr, value, *, ordering="monotonic"):
    """Agent-scope ``fetch_add`` (from ``gemm_ar/_compat.py``); returns the old value."""
    return _llvm.atomicrmw(
        _llvm.AtomicBinOp.add,
        ptr,
        fx.Int32(value).ir_value(),
        getattr(_llvm.AtomicOrdering, ordering),
        syncscope=_AGENT_SCOPE,
        alignment=4,
    )


def _iv(x):
    return x.ir_value() if hasattr(x, "ir_value") else x


def permlane16_swap(x, y):
    """gfx950 ``v_permlane16_swap_b32`` on two i32 values.

    A wave is four 16-lane rows. The instruction exchanges the odd rows of its
    first operand with the even rows of its second (row 1 of ``x`` with row 0 of
    ``y``, row 3 of ``x`` with row 2 of ``y``) and returns both, so with
    ``g = lane // 16``::

        x' = [x.r0, y.r0, x.r2, y.r2]      y' = [x.r1, y.r1, x.r3, y.r3]

    which is what ``AgStoreC._store_bf16`` needs to pair two N tiles into 16
    contiguous bytes per lane (lane group g -> columns (g%2)*16 + (g//2)*8).
    ``probe.py`` checks this mapping on the device.
    """
    i32 = ir.IntegerType.get_signless(32)
    xv, yv = _iv(fx.Int32(x)), _iv(fx.Int32(y))
    pair_t = ir.Type.parse("!llvm.struct<(i32, i32)>")
    pair = _rocdl_d.permlane16_swap(pair_t, xv, yv, False, False)
    a = _llvm.extractvalue(i32, pair, [0])
    b = _llvm.extractvalue(i32, pair, [1])
    return fx.Int32(a), fx.Int32(b)


class SwappedMfma:
    """``MfmaBf16`` with A and B exchanged: the instruction computes ``B A^T``,
    the transpose of the tile, so lane ``l`` holds four consecutive output
    *columns* ``4*(l//16) .. +3`` of row ``l % 16`` (what ``AgStoreC`` expects).

    ``call(a, b, c)`` takes ``a`` over the A tiles and ``b`` over the B tiles, as
    the plain atom would; ``idx(i, j)`` is the accumulator of (A tile i, B tile j).
    """

    def __init__(self, inner):
        self.inner = inner  # MfmaBf16(n_tiles_b, n_tiles_a)
        self.zero_value = inner.zero_value

    def idx(self, i, j):
        return self.inner.idx(j, i)

    def call(self, a, b, c, *, set_prio=True):
        return self.inner.call(b, a, c, set_prio=set_prio)
