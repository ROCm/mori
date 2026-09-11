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
"""The 8-wave fp8 GEMM with the all-reduce's scatter fused into its epilogue.

Built for DeepSeek-V4-Pro's ``wo_b``: a RowParallelLinear whose per-rank GEMM is
``[M,K] x [N,K]`` fp8 -> ``[M,N]`` bf16, immediately all-reduced. Split, that is
``gemm(); all_reduce()``. Fused, the GEMM's C lands directly in a registered cco
window and each destination's slice is pushed by the copy engine as soon as its
last tile is written, so the reduce-scatter transfer overlaps the rest of the
GEMM instead of following it.

Only the *scatter* half of the all-reduce is absorbed. The reduce and all-gather
phases still run as their own kernels, reused verbatim from ``kernels_sdma``
(``build_sdma_phases``), with ``scatter`` swapped for its drain-only twin.

``README.md`` carries the design narrative and the measurements: why SDMA needs
no epilogue change, the completion protocol, the three C-store stages, and what
was tried and rejected. This docstring covers only what a caller must know.

## Completion protocol

One monotonic counter per (destination, chunk) in the window
(``cfg.counter_off``). Every block, after its four ``store_c.store`` calls::

    s_waitcnt vmcnt(0) ; s_barrier       -- this block's C tile has retired
    __threadfence_system()               -- ...and is visible to the copy engine
    thread 0: prev = atomic_add(counter[dest][chunk], 1)
    if (prev + 1) % tiles_per_chunk == 0 -- I am the last tile of that chunk
        sdma.put(dest, ...)              -- fire and forget, no quiet

Counters are never reset, so the modulo test works on every launch and the
kernel stays CUDA-graph-safe, exactly like the barrier flags in
``kernels_lsa``.

## Relationship to the vendored pipeline

``_fused_kernel_body`` is a copy of ``compile_fp8_gemm_8w``'s ``kernel_gemm``
from ``_gemm_a8w8_8wave.py`` with the epilogue tail added. It is a copy rather
than a hook because the original's main loop contains a runtime
``if wave_m == 1``, and FlyDSL only rewrites ``if`` inside the
``@flyc.kernel`` function's own AST -- factoring the body into a shared helper
makes that line raise at trace time. Every reusable piece (``G2SLoader``,
``S2RLoader``, ``StoreC``, ``Mfma16x16x128``, the swizzles) is imported rather
than copied, so the duplication is the ~90-line pipeline only.

The copy carries **one deliberate divergence**, marked at its site in the main
loop: the final ``wait_barrier`` count. The upstream count is too permissive and
corrupts the output non-deterministically on large grids. Do not "restore" it --
see the comment at that line. The bug is still live in aiter.
"""

# NOTE: no `from __future__ import annotations` here, deliberately, and
# _gemm_a8w8_8wave.py omits it for the same reason: `fx.struct` reads the
# `SharedStorage` field annotations as live objects, and PEP 563 would hand it
# the string "fx.Array[fx.Float8E4M3FN, a_lds_size, 16]" -- whose size operands
# are locals of this factory and so cannot be resolved from the module globals.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl._mlir.dialects import scf
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
    atomic_xchg_u32,
    buffer_store,
    create_buffer_resource_from_addr,
    i32_type,
    release_fence,
    signal_ptr,
    wave_uniform_i64,
)
from ._gemm_a8w8_8wave import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    StoreC,
    _xcd_swizzle_any,
    ceildiv,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    split_row_major_2d,
    wait_barrier,
)


def _acquire_peer_lock(lock_ptr):
    """Test-and-set spin on one destination's submit lock.

    Transferred from gcnasm ``opus_chunk_sdma_submit``: with more than one chunk
    per destination the tile counter elects one block per *chunk*, so two of them
    can reach the SDMA submit for the same queue at once. Serialising only the
    submit -- not the GEMM work -- is the point.

    Termination: the contending blocks are elected in chunk order, and chunk c's
    tiles are dispatched before chunk c+1's, so a spinning block is always
    waiting on one that has already run.
    """
    i32 = i32_type()
    first = atomic_xchg_u32(lock_ptr, 1)
    loop = scf.WhileOp([i32], [first])
    cond = ir.Block.create_at_start(loop.before, [i32])
    body = ir.Block.create_at_start(loop.after, [i32])
    with ir.InsertionPoint(cond):
        held = fx.Uint32(fx.Int32(cond.arguments[0])) > fx.Uint32(fx.Int32(0))
        scf.ConditionOp(held.ir_value(), [cond.arguments[0]])
    with ir.InsertionPoint(body):
        fx.rocdl.s_sleep(1)
        nxt = atomic_xchg_u32(lock_ptr, 1)
        scf.YieldOp([nxt])


class _RawWriteThroughStoreC(StoreC):
    """``StoreC`` whose C stores really do bypass L1 *and* L2.

    ``StoreC`` stores through a copy atom whose ``cache_modifier`` is a two-value
    enum (0=cached, 2=nt), so ``sc0|sc1`` is not expressible there -- asking for
    it emits a plain ``sc0`` and leaves the line dirty in L2, which is why the
    earlier ``--fence writethrough`` probe measured nothing. This goes around the
    atom to ``raw_ptr_buffer_store``, whose ``aux`` operand is the real cache
    policy, so the data reaches memory at the store and the per-block
    ``buffer_wbl2`` release becomes unnecessary.

    The descriptor is built from the window rather than from the ``C`` argument
    because in the fused kernel they are the same bytes: C *is* the all-reduce's
    input region. ``num_records`` still bounds it, so ``store``'s out-of-bounds
    index for masked columns is dropped exactly as before.

    Deliberately *not* coalesced: the point of this variant is to find out
    whether the coalescing rewrite is needed before paying for it.
    """

    def __init__(self, *args, c_rsrc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._c_rsrc = c_rsrc

    def _store_bf16(self, value_bf16, c_index):
        buffer_store(value_bf16, self._c_rsrc, c_index, cache_modifier=CM_SC0_SC1)


class _SwappedMfma:
    """``Mfma16x16x128`` with the A/B operands exchanged at every call.

    Thin on purpose: the accumulator registers, the atom and ``zero_value`` are
    the wrapped object's, only the operand order and the ``idx`` argument order
    change. ``idx(ti, tj)`` here forwards to the wrapped ``idx(tj, ti)`` so the
    store can keep indexing in (M-tile, N-tile) order.
    """

    def __init__(self, inner):
        self._inner = inner
        self.zero_value = inner.zero_value

    def idx(self, i, j):
        return self._inner.idx(j, i)

    def call(self, a, b, c, *, set_prio=True):
        return self._inner.call(b, a, c, set_prio=set_prio)


class _BlockScaleK:
    """The model's 1x128 / 128x128 fp8 block scales, applied per K-block.

    ``sum_k s_a[m,k] * s_b[nb,k] * d_k`` does not factor, so unlike the
    per-token/per-channel form these scales cannot wait for the epilogue.
    ``BLOCK_K`` is already 128, so a mainloop iteration *is* a scale block and
    there is no sub-block bookkeeping.

    The obvious form -- promote each K-block into a second fp32 accumulator and
    zero the MFMA one -- does not work here, and the reason is worth recording.
    Zeroing removes the only dependency between consecutive K-blocks, and the K
    loop is ``range_constexpr`` (fully unrolled), so the scheduler hoists every
    block's MFMAs above every promotion: measured at K=512, all 64 MFMAs land in
    the first 400 instructions and all the promotion arithmetic after, keeping
    ``K_ITERS`` accumulator sets live at once -- 256 VGPR with **81 spilled**,
    and the GEMM 10x slower (4052us against a 374us target).
    ``rocdl.sched_barrier(0)`` between them does not stop it.

    So instead the accumulator carries the running sum *rescaled*, which keeps a
    real data dependency the scheduler cannot break and needs no second
    accumulator at all::

        t_0   = d_0
        t_k   = t_{k-1} * (s_{k-1} / s_k) + d_k
        out   = t_{K-1} * s_{K-1}

    with the invariant ``t_k = (sum_{j<=k} s_j d_j) / s_k``. The divisions cost
    ~1e-7 relative each and there are ``K/128`` of them, three orders of
    magnitude under the fp8 floor of 1.7e-3. It does assume no scale is exactly
    zero, which a real quantiser never emits.

    Two properties of the tile geometry make the loads cheap, both checked
    against ``_PermlaneStoreC._emit``'s indexing:

    * every accumulator set lies inside a single 128-column block
      (``wave_n*32 + tj*16 + 15 <= 127``), so the B scale is **one scalar for
      the whole set**, uniform across the wave;
    * ``sa`` is column-major -- the layout
      ``gemm_a8w8_blockscale_bpreshuffle`` consumes, via sglang's
      ``materialize_bpreshuffle_fp8_scale`` -- so element ``(row, kb)`` sits at
      ``kb * M + row`` and a lane's rows are contiguous. With ``swap_ab`` a lane
      owns one row per M-tile (a scalar load); without it, four consecutive rows
      (one vec4).
    """

    def __init__(self, A_scale, B_scale, m, n, k, *, swap_ab, n_tiles_a):
        self.kb_count = k // 128
        self.m = m
        self.swap_ab = swap_ab
        self.n_tiles_a = n_tiles_a
        self.lane = fx.thread_idx.x % 64
        gSA = fx.rocdl.make_buffer_tensor(
            A_scale, max_size=False, num_records_bytes=m * self.kb_count * 4
        )
        gSB = fx.rocdl.make_buffer_tensor(
            B_scale, max_size=False, num_records_bytes=(n // 128) * self.kb_count * 4
        )
        self.sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
        self.sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))
        self.atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        self.atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        self.reg_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        self.reg_4 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)

    def _load1(self, div, index):
        fx.copy(self.atom_1, fx.slice(div, (None, fx.Int32(index))), self.reg_1)
        return Vec(fx.memref_load_vec(self.reg_1))[0]

    def _load4(self, div, index):
        fx.copy(self.atom_4, fx.slice(div, (None, fx.Int32(index))), self.reg_4)
        return Vec(fx.memref_load_vec(self.reg_4))

    def a_scales(self, base_row, kb):
        """Per M-tile A scale for this lane's rows at K-block ``kb``."""
        col = fx.Int32(kb) * fx.Int32(self.m)
        lane = self.lane
        if const_expr(self.swap_ab):
            return [
                self._load1(self.sa_div, col + base_row + ti * 16 + lane % 16)
                for ti in range_constexpr(self.n_tiles_a)
            ]
        return [
            self._load4(self.sa_div, col + base_row + ti * 16 + (lane // 16) * 4)
            for ti in range_constexpr(self.n_tiles_a)
        ]

    def b_scale(self, n_block, kb):
        """The single B scale of a 128-column block at K-block ``kb``."""
        return self._load1(self.sb_div, n_block * fx.Int32(self.kb_count) + fx.Int32(kb))

    def scale_acc(self, acc, a_sc, b_sc, idx_fn, n_tiles_b):
        """``acc *= s_a * s_b`` elementwise; returns the new accumulator.

        Used both for the running rescale between K-blocks and for the final
        multiply, which are the same operation with different scale pairs.
        """
        res = list(acc)
        for ti in range_constexpr(self.n_tiles_a):
            for tj in range_constexpr(n_tiles_b):
                i = idx_fn(ti, tj)
                v = Vec(res[i])
                if const_expr(self.swap_ab):
                    s = a_sc[ti] * b_sc
                    vals = [v[e] * s for e in range_constexpr(4)]
                else:
                    vals = [v[e] * (a_sc[ti][e] * b_sc) for e in range_constexpr(4)]
                res[i] = Vec.from_elements(vals, fx.Float32)
        return res

    @staticmethod
    def ratio(prev, cur):
        """``prev / cur`` for a scale pair, elementwise over the M-tile list."""
        if isinstance(prev, list):
            return [p / c for p, c in zip(prev, cur)]
        return prev / cur


    def rescale_for(self, kb, accs, prev, *, base_row, nb0, idx_fn, n_tiles_b,
                    lds_block_m):
        """Rebase the four accumulators from K-block ``kb-1``'s scales to kb's.

        A method rather than a closure inside the kernel: FlyDSL rewrites the
        AST of every ``def`` nested in a ``@flyc.kernel`` function, and the
        rewrite of an ``if`` inside such a nested def turned the captured
        ``_BlockScaleK`` instance into a local, so reading it raised
        UnboundLocalError. Module-level methods are ordinary Python at trace
        time and are left alone.
        """
        c00, c01, c10, c11 = accs
        cur = (
            self.a_scales(base_row + 0 * lds_block_m, kb),
            self.a_scales(base_row + 1 * lds_block_m, kb),
            self.b_scale(nb0 + fx.Int32(0), kb),
            self.b_scale(nb0 + fx.Int32(1), kb),
        )
        if prev is not None:
            pa0, pa1, pb0, pb1 = prev
            ca0, ca1, cb0, cb1 = cur
            ra0, ra1 = self.ratio(pa0, ca0), self.ratio(pa1, ca1)
            rb0, rb1 = self.ratio(pb0, cb0), self.ratio(pb1, cb1)
            c00 = self.scale_acc(c00, ra0, rb0, idx_fn, n_tiles_b)
            c01 = self.scale_acc(c01, ra0, rb1, idx_fn, n_tiles_b)
            c10 = self.scale_acc(c10, ra1, rb0, idx_fn, n_tiles_b)
            c11 = self.scale_acc(c11, ra1, rb1, idx_fn, n_tiles_b)
        return (c00, c01, c10, c11), cur

    def final_scale(self, accs, prev, *, idx_fn, n_tiles_b):
        """Undo the invariant: multiply by the last K-block's scales."""
        fa0, fa1, fb0, fb1 = prev
        c00, c01, c10, c11 = accs
        return (
            self.scale_acc(c00, fa0, fb0, idx_fn, n_tiles_b),
            self.scale_acc(c01, fa0, fb1, idx_fn, n_tiles_b),
            self.scale_acc(c10, fa1, fb0, idx_fn, n_tiles_b),
            self.scale_acc(c11, fa1, fb1, idx_fn, n_tiles_b),
        )


class _SwapABStoreC(StoreC):
    """C store for an A/B-swapped MFMA: 4 consecutive N per lane -> 64-bit store.

    ``fx.gemm(atom, c, b, a, c)`` computes ``B^T A^T = (A B)^T`` in the
    accumulator's native layout, so lane ``l`` value ``k`` moves from
    ``D[4*(l/16)+k][l%16]`` (4 consecutive *rows*, stride c_cols) to
    ``D[l%16][4*(l/16)+k]`` (4 consecutive *columns*). In a row-major C that is
    8 contiguous bytes, which is the whole point -- the unswapped layout can only
    ever emit ``buffer_store_short``. This is gcnasm's ``mfma_adaptor_swap_ab``,
    which does literally ``base::operator()(b, a, c)`` and redefines ``dim_c()``
    to match (opus.hpp:2064).

    The scale loads swap with it, and stay equally well vectorized: A's scale is
    indexed by row, which is now ``lane % 16`` -- one row, so a scalar load --
    while B's is indexed by column, now 4 consecutive, so a vec4.
    """

    def __init__(self, *args, peer_rsrc=None, elem_base=None,
                 scales_preapplied=False, **kwargs):
        super().__init__(*args, **kwargs)
        # blockscale applies the scales per K-block in the mainloop, so by the
        # time C reaches here it is already scaled and the epilogue is a plain
        # fp32 -> bf16 convert. Skipping the loads rather than multiplying by a
        # constant 1.0 keeps them out of the IR instead of trusting a fold.
        self._preapplied = scales_preapplied
        self.out_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
        self.reg_bf16_4 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
        self.out_atom_8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        self.reg_bf16_8 = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
        self._peer_rsrc = peer_rsrc
        self._elem_base = elem_base

    def _scaled(self, v, a, b):
        """``v * a * b``, or ``v`` when the mainloop already applied them."""
        if const_expr(self._preapplied):
            return v
        return v * (a * b)

    def _load_a_scale_scalar(self, row):
        if const_expr(self._preapplied):
            return None
        fx.copy(
            self.scale_atom_1,
            fx.slice(self.sa_div, (None, fx.Int32(row))),
            self.reg_f32_1,
        )
        return Vec(fx.memref_load_vec(self.reg_f32_1))[0]

    def _load_b_scale_vec4(self, col):
        if const_expr(self._preapplied):
            return [None] * 4
        fx.copy(
            self.scale_atom_4,
            fx.slice(self.sb_div, (None, fx.Int32(col))),
            self.reg_f32_4,
        )
        return Vec(fx.memref_load_vec(self.reg_f32_4))

    def _store_bf16x4(self, values, c_index):
        fx.memref_store_vec(Vec.from_elements(values, fx.BFloat16), self.reg_bf16_4)
        fx.copy(
            self.out_atom_4,
            self.reg_bf16_4,
            fx.slice(self.c_div, (None, fx.Int32(c_index))),
        )

    def store(self, c_frag, base_row, base_col):
        lane = self.lane_id
        a_scales = [
            self._load_a_scale_scalar(base_row + ti * 16 + lane % 16)
            for ti in range_constexpr(self.n_tiles_a)
        ]
        b_scales = [
            self._load_b_scale_vec4(base_col + tj * 16 + (lane // 16) * 4)
            for tj in range_constexpr(self.n_tiles_b)
        ]
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + lane % 16
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + (lane // 16) * 4
                # BLOCK_N divides N in every shape this builds for, so the four
                # columns are either all in range or all out; one predicate.
                col_valid = col + 3 < self.c_cols
                oob = fx.Int32(self.c_rows * self.c_cols)
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                vals = [
                    self._scaled(vec_f32[k], a_scales[ti], b_scales[tj][k]).to(fx.BFloat16)
                    for k in range_constexpr(4)
                ]
                c_index = row * self.c_cols + col
                self._store_bf16x4(vals, arith.select(col_valid, c_index, oob))


def _permlane16_swap(x, y):
    """gfx950 ``v_permlane16_swap_b32``: exchange 16-lane rows between two VGPRs.

    Verified on hardware rather than assumed::

        vdst_new = [X.r0, Y.r0, X.r2, Y.r2]
        vsrc_new = [X.r1, Y.r1, X.r3, Y.r3]

    Position *within* a row is preserved, so a lane keeps its ``lane % 16`` and
    therefore its C row -- which is what lets the A-scale stay correct across the
    shuffle.
    """
    st = ir.Type.parse("!llvm.struct<(i32, i32)>")
    raw = lambda v: v.ir_value() if hasattr(v, "ir_value") else v
    res = fx.rocdl.permlane16_swap(st, raw(x), raw(y), False, True)
    i32 = ir.IntegerType.get_signless(32)
    return (
        fx.Int32(_llvm_d.ExtractValueOp(i32, res, ir.DenseI64ArrayAttr.get([0])).result),
        fx.Int32(_llvm_d.ExtractValueOp(i32, res, ir.DenseI64ArrayAttr.get([1])).result),
    )


def _ds_bpermute(value, src_lane):
    """Pull ``value`` from ``src_lane``. Byte-addressed, hence the <<2."""
    i32 = ir.IntegerType.get_signless(32)
    raw = lambda v: v.ir_value() if hasattr(v, "ir_value") else v
    return fx.Int32(
        fx.rocdl.ds_bpermute(i32, raw(fx.Int32(src_lane) * fx.Int32(4)), raw(value))
    )


class _PermlaneStoreC(_SwapABStoreC):
    """A/B-swapped MFMA + one cross-lane step, giving 16B per lane and 64B rows.

    After the swap a lane owns 4 consecutive columns of one N-tile, i.e. 8 bytes,
    and the two N-tiles it holds are 16 columns apart -- so a row still only gets
    32 bytes per instruction. Two ``permlane16_swap`` fix that exactly, with no
    ``ds_bpermute`` and no LDS:

        (A, B) = permlane16_swap(tile0.d0, tile1.d0)
        (C, D) = permlane16_swap(tile0.d1, tile1.d1)
        lane group g stores (A, C, B, D)

    which lands g = 0,1,2,3 on columns 0-7, 16-23, 8-15, 24-31. The four groups
    together cover columns 0..31 contiguously: 64 bytes per row, from 16 bytes per
    lane. The column permutation is absorbed into the address, so it is free.

    Scaling happens before the shuffle, in the lane that owns the value; the
    shuffle only moves whole 16-lane rows, so every lane keeps its own C row.
    """

    _adjacent_probe = False
    _lane_transpose = False
    _peer_uncached = False

    def store(self, c_frag, base_row, base_col):
        self._emit(c_frag, base_row, base_col,
                   self._a_scales(base_row), self._b_scales(base_col))

    def store_all(self, frags, base_row, base_col, row_step, col_step):
        """All four half-tiles, loading each scale once instead of twice.

        The four ``store`` calls the epilogue makes share base rows pairwise and
        base columns pairwise, so calling them individually issues every scale
        load twice -- 16 ``buffer_load_dwordx4`` for A where 8 addresses are
        distinct, 8 ``buffer_load_dword`` for B where 4 are. The compiler cannot
        merge them because they all write the same ``reg_f32_*`` register buffer,
        which turns them into a chain of overwrites rather than pure loads.
        """
        a = [self._a_scales(base_row + r * row_step) for r in range_constexpr(2)]
        b = [self._b_scales(base_col + c * col_step) for c in range_constexpr(2)]
        for frag, r, c in frags:
            self._emit(frag, base_row + r * row_step, base_col + c * col_step,
                       a[r], b[c])

    def _a_scales(self, base_row):
        lane = self.lane_id
        return [
            self._load_a_scale_scalar(base_row + ti * 16 + lane % 16)
            for ti in range_constexpr(self.n_tiles_a)
        ]

    def _b_scales(self, base_col):
        grp = self.lane_id // 16
        return [
            self._load_b_scale_vec4(base_col + tj * 16 + grp * 4)
            for tj in range_constexpr(self.n_tiles_b)
        ]

    def _emit(self, c_frag, base_row, base_col, a_scales, b_scales):
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
                    [
                        self._scaled(vec_f32[k], a_scales[ti], b_scales[tj][k]).to(fx.BFloat16)
                        for k in range_constexpr(4)
                    ],
                    fx.BFloat16,
                ).bitcast(fx.Int32)
                dwords.append((packed[0], packed[1]))
            a, b = _permlane16_swap(dwords[0][0], dwords[1][0])
            c, d = _permlane16_swap(dwords[0][1], dwords[1][1])
            out8 = Vec.from_elements([a, c, b, d], fx.Int32).bitcast(fx.BFloat16)
            if const_expr(self._lane_transpose):
                # gcnasm's second stage (kernel_template.hpp:491-510), as one
                # ds_bpermute per dword. The permlane stage leaves a row's four
                # 8-column chunks in lanes 16 apart, so a 16-lane group touches
                # 16 rows at 16 bytes each. Transposing the lane index -- lane
                # l' = 4r'+q' pulls from the lane holding (row r', chunk q') --
                # puts adjacent lanes on one row, so lanes 0-3 write 64
                # contiguous bytes and a group covers 4 rows instead of 16.
                #
                # chunk q' lives at column q'*8, and the permlane stage put
                # column (g%2)*16 + (g//2)*8 in group g, so g = swap of q's two
                # bits: 0,1,2,3 -> 0,2,1,3.
                q = lane % 4
                src_lane = ((q % 2) * 2 + q // 2) * 16 + lane // 4
                a, c, b, d = (_ds_bpermute(v, src_lane) for v in (a, c, b, d))
                out8 = Vec.from_elements([a, c, b, d], fx.Int32).bitcast(fx.BFloat16)
                row = base_row + ti * 16 + lane // 4
                col = base_col + q * 8
            elif const_expr(self._adjacent_probe):
                # PERF PROBE, output wrong by construction. Same 64 addresses,
                # reassigned so *adjacent* lanes cover one row (lanes 0-3 -> row 0,
                # 64 contiguous bytes) instead of lanes 16 apart. Prices gcnasm's
                # ds_bpermute lane transpose before writing it: if the address
                # coalescer works per 16-lane group rather than per wave, this
                # takes a group from 16 sectors to 4.
                row = base_row + ti * 16 + lane // 4
                col = base_col + (lane % 4) * 8
            else:
                col = base_col + (grp % 2) * 16 + (grp // 2) * 8
            oob = fx.Int32(self.c_rows * self.c_cols)
            idx = arith.select(col + 7 < self.c_cols, row * self.c_cols + col, oob)
            if self._peer_rsrc is not None:
                # The store goes to a *peer*, so a cached one leaves the line
                # dirty in whichever XCD's L2 this block ran on, where nothing
                # downstream can reach it -- the barrier kernel is one block and
                # buffer_wbl2 writes back only its own XCD's L2, one of eight.
                # sc0|sc1 skips both. That was not usable while the store was one
                # bf16 (a 2-byte partial-line write over the fabric loses
                # updates), but the permlane stage makes it 16 bytes per lane and
                # 64 contiguous per row, which is what makes it viable now.
                buffer_store(
                    out8,
                    self._peer_rsrc,
                    fx.Int32(idx) - self._elem_base,
                    cache_modifier=CM_SC0_SC1 if self._peer_uncached else CM_CACHED,
                )
            else:
                fx.memref_store_vec(out8, self.reg_bf16_8)
                fx.copy(
                    self.out_atom_8, self.reg_bf16_8,
                    fx.slice(self.c_div, (None, fx.Int32(idx))),
                )


class _LaneTransposeStoreC(_PermlaneStoreC):
    """``_PermlaneStoreC`` plus gcnasm's ds_bpermute lane transpose."""

    _lane_transpose = True


class _LaneTransposeUncachedPeerStoreC(_LaneTransposeStoreC):
    """Same, but the peer store bypasses L1 and L2 (Direct LSA only)."""

    _peer_uncached = True


class _AdjacentLaneProbeC(_PermlaneStoreC):
    """PERF PROBE ONLY -- see the note in ``_PermlaneStoreC.store``."""

    _adjacent_probe = True


class _WideStoreProbeC(_SwapABStoreC):
    """PERF PROBE ONLY -- the output is wrong on purpose.

    Answers "is 16B per lane / 64B per row actually faster?" before anyone builds
    the cross-lane shuffle that would make it correct. It emits exactly the access
    pattern the shuffled version would have -- one 128-bit store per lane at
    ``col = (lane//16)*8``, so 4 lanes cover a row's 32 columns as 64 contiguous
    bytes -- but feeds it the two N-tiles' values concatenated, which is not what
    belongs at those addresses.

    Correct would need lane ``(a, r)`` to gather cols ``8a..8a+7``, which live in
    two *other* lanes' registers (``a'=2a`` and ``2a+1`` of one tile). That is the
    ``permlane16_swap`` step; this probe skips it and keeps only its cost profile.
    """



    def store(self, c_frag, base_row, base_col):
        lane = self.lane_id
        a_scales = [
            self._load_a_scale_scalar(base_row + ti * 16 + lane % 16)
            for ti in range_constexpr(self.n_tiles_a)
        ]
        b_scales = [
            self._load_b_scale_vec4(base_col + tj * 16 + (lane // 16) * 4)
            for tj in range_constexpr(self.n_tiles_b)
        ]
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + lane % 16
            vals = []
            for tj in range_constexpr(self.n_tiles_b):
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                vals += [
                    self._scaled(vec_f32[k], a_scales[ti], b_scales[tj][k]).to(fx.BFloat16)
                    for k in range_constexpr(4)
                ]
            col = base_col + (lane // 16) * 8
            oob = fx.Int32(self.c_rows * self.c_cols)
            c_index = row * self.c_cols + col
            idx = arith.select(col + 7 < self.c_cols, c_index, oob)
            if self._peer_rsrc is not None:
                buffer_store(
                    Vec.from_elements(vals, fx.BFloat16),
                    self._peer_rsrc,
                    fx.Int32(idx) - self._elem_base,
                )
            else:
                fx.memref_store_vec(
                    Vec.from_elements(vals, fx.BFloat16), self.reg_bf16_8
                )
                fx.copy(
                    self.out_atom_8,
                    self.reg_bf16_8,
                    fx.slice(self.c_div, (None, fx.Int32(idx))),
                )


class _PeerDirectStoreC(StoreC):
    """``StoreC`` that writes its tile straight into the owning peer's window.

    This is gcnasm's "Direct LSA" shape (``opus_gemm_a2a_lsa``), and its whole
    point is that there is no producer/consumer handoff inside the kernel: the
    bytes go to the destination as they are produced, so nothing has to be
    published mid-kernel, there is no completion counter, no submit lock and no
    release fence. The only synchronisation left is the cross-rank barrier after
    the kernel, where a kernel-end release has already happened for free.

    A block's rows all belong to one destination (``block_m // m_tiles_per_peer``),
    so the peer descriptor is uniform per block and is built once.

    ``elem_base`` rebases the global row index onto the destination's slice.
    ``StoreC.store`` masks out-of-range columns by redirecting them to
    ``c_rows * c_cols``, and that stays out of range after rebasing: the largest
    ``elem_base`` is ``(world-1) * slice_rows * c_cols``, which leaves exactly
    ``slice_rows * c_cols`` elements -- one past the end of the descriptor.
    """

    def __init__(self, *args, peer_rsrc=None, elem_base=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._peer_rsrc = peer_rsrc
        self._elem_base = elem_base

    def _store_bf16(self, value_bf16, c_index):
        # Cached, and that is the unresolved half of this variant. See the
        # "Direct LSA" note in the module docstring: uncached (sc0|sc1) is
        # *consistently* wrong here because a 2-byte store bypassing L2 is a
        # partial-line write, and cached is only intermittently right because the
        # dirty peer-homed lines cannot be flushed out of eight XCDs' L2s
        # afterwards. Coalescing the store first resolves both.
        buffer_store(value_bf16, self._peer_rsrc, c_index - self._elem_base)


class _NonTemporalStoreC(StoreC):
    """``StoreC`` with non-temporal C stores (the copy atom's only other mode).

    Probes whether ``buffer_wbl2`` gets cheaper when the lines it is asked to
    flush have already been evicted. The atom's ``cache_modifier`` is a two-value
    enum, 0=cached / 2=nt -- it is *not* the aux bitmask that
    ``raw_ptr_buffer_store`` takes, so sc0|sc1 cannot be requested this way.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(2), fx.BFloat16)


class _WriteThroughStoreC(StoreC):
    """``StoreC`` whose C stores bypass L1 and L2 instead of being written back.

    The only change is the copy atom's cache modifier. It exists because a
    per-block release fence is quadratic: ``buffer_wbl2`` flushes the *whole* L2,
    so block k redundantly writes back every tile blocks 1..k-1 already wrote,
    448 times over. Writing C through in the first place makes the release free --
    ``s_waitcnt vmcnt(0)`` is then all the copy engine needs -- and adds no
    traffic, since C has to reach memory regardless.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_atom_1 = fx.make_copy_atom(
            fx.rocdl.BufferCopy16b(CM_SC0_SC1), fx.BFloat16
        )


BLOCK_K = 128


def compile_fused_gemm_scatter(
    cfg,
    rank: int,
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    b_preshuffled: bool = True,
    waves_per_eu: int = 2,
    xcd_swizzle: int = 0,
    fuse: bool = True,
    rotated: bool | None = None,
    n_stripe: int | None = None,
    quant: str = "ptpc",
    sdma_queues: int = 8,
    swap_ab: bool = False,
    store_probe: bool = False,
    permlane: bool = False,
    lane_transpose: bool = False,
    hoist_scales: bool = False,
    peer_uncached: bool = False,
    direct_fence: str = "leader",
    transport: str = "sdma",
    fence: str = "none",
    emit_put: bool = True,
    atomic_order: str = "acq_rel",
):
    """Compile the GEMM, with (``fuse=True``) or without the scatter epilogue.

    ``fuse=False`` emits the identical kernel minus the epilogue tail, so the
    split baseline and the fused kernel differ in exactly one thing. That is why
    the baseline is built here rather than called through a separate wrapper, which
    would also change the launch path and the C tensor.

    ``rotated`` selects the destination-rotated chunk-major tile order; it
    defaults to ``fuse``, since it only matters when something is watching tiles
    complete. It is separately settable so the benchmark can charge the split
    baseline the same tile order and show that the ordering itself is neutral.

    ``n_stripe`` is how many N-tiles a block walks before the destination
    rotates -- gcnasm's ``opus_direct_stripe_tile``, transposed (its
    destination is on N and ours is on M, so its ``m_stripe`` is our
    ``n_stripe``). It spans the whole range: 1 rotates on every block, and
    ``N // BLOCK_N`` is chunk-major, where a destination's whole chunk is
    finished before moving on. ``None`` picks 2 for ``fused-lsa`` and
    chunk-major elsewhere.

    Swept at [16384, 7168] K=2048 on 8 ranks (28 N-tiles, so the divisors are
    1, 2, 4, 7, 14, 28):

        n_stripe     1      2      4      7     14     28
        fused-lsa  1370.9 1362.5 1404.9 1402.9 1437.0 1569.0
        fused-sdma 1117.9 1117.1 1114.8 1124.0 1117.5 1115.7

    ``fused-lsa`` is monotone from 28 down to 2 and then flat -- 1 and 2 are the
    same within noise -- for 13% end to end. ``fused-sdma`` is flat across the
    whole range: its bytes leave from a staging buffer on the copy engines, so
    only the moment a chunk *completes* matters, and with 256 resident blocks
    out of 1792 the first chunk completes inside the first wave of blocks
    whatever the order.

    Two hypotheses this sweep kills. It is not XCD locality: with
    ``dest = (block / n_stripe) % 8`` and blocks going round-robin over the 8
    XCDs, a destination is fed by exactly ``min(n_stripe, 8)`` XCDs, so 14 and
    28 spread over all eight and 2 over only two -- and 2 is the fast one. And
    it is not chunk-completion timing, or ``fused-sdma`` would prefer
    chunk-major, which it does not.

    Returns ``launch(A, B_T, C, A_scale, B_scale, c_m, c_n, dev_comm, win,
    stream=...)``.
    """
    cfg.validate()
    ws = cfg.world_size
    M, N = cfg.m, cfg.n
    if transport not in ("sdma", "lsa"):
        raise ValueError(f"transport must be sdma or lsa, got {transport!r}")
    # The C-store variant is picked by a chain of const_expr branches, so an
    # unimplemented combination does not fail -- it quietly falls through to a
    # weaker one. That already cost a wrong conclusion once: --lane-transpose had
    # no direct-LSA branch, so `--mode fused-lsa --lane-transpose` silently ran
    # stages 1+2 and was measured as "no benefit from stage 3".
    if permlane and not swap_ab:
        raise ValueError(
            "--permlane requires --swap-ab. Both are on by default in "
            "bench_gemm_ar.py, so pass --no-permlane alongside --no-swap-ab"
        )
    if lane_transpose and not permlane:
        raise ValueError(
            "--lane-transpose requires --permlane. Both are on by default in "
            "bench_gemm_ar.py, so pass --no-lane-transpose alongside "
            "--no-permlane"
        )
    if store_probe and not swap_ab:
        raise ValueError("--store-probe requires --swap-ab (on by default)")
    if store_probe and lane_transpose:
        raise ValueError(
            "--store-probe and --lane-transpose are alternatives: the probe emits "
            "the adjacent-lane access pattern without the shuffle that makes it "
            "correct, which is what --lane-transpose then implements"
        )
    if transport == "lsa" and store_probe and permlane:
        raise ValueError(
            "no peer-direct variant of the adjacent-lane probe; use "
            "--lane-transpose for the real thing on --mode fused-lsa"
        )
    direct_lsa = fuse and transport == "lsa"
    if quant not in ("ptpc", "blockscale"):
        raise ValueError(f"quant must be ptpc or blockscale, got {quant!r}")
    blockscale = quant == "blockscale"
    if blockscale:
        if not swap_ab:
            raise ValueError(
                "--quant blockscale needs --swap-ab: the unswapped epilogue "
                "applies its scales inside the stock StoreC, which this copy does "
                "not override"
            )
        if K % 128:
            raise ValueError(f"blockscale needs K % 128 == 0, got K={K}")
        if N % 128:
            raise ValueError(f"blockscale needs N % 128 == 0, got N={N}")
    rotated = fuse if rotated is None else rotated
    # n_stripe spans 1..N//BLOCK_N; the top of the range *is* chunk-major, so
    # 0 ("per-mode default") resolves to it rather than to a separate branch.
    if n_stripe is None:
        n_stripe = 2 if direct_lsa else 0
    if n_stripe == 0:
        n_stripe = N // BLOCK_N
    if n_stripe < 1 or n_stripe > N // BLOCK_N:
        raise ValueError(
            f"n_stripe must be in [1, {N // BLOCK_N}], got {n_stripe}"
        )
    if (N // BLOCK_N) % n_stripe:
        raise ValueError(
            f"n_stripe={n_stripe} must divide the {N // BLOCK_N} N-tiles"
        )
    # Only a *real* stripe needs the rotated order. Chunk-major is the
    # unstriped case and is what the linear order already does per destination,
    # so leaving it on a non-rotated build is not an error -- the split
    # baseline compiles that way.
    if n_stripe < N // BLOCK_N and not rotated:
        raise ValueError("a striped tile order needs --tile-order rotated")

    assert BLOCK_M >= 128 and BLOCK_N >= 256
    assert BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0
    if N % BLOCK_N:
        raise ValueError(
            f"N={N} must be a multiple of BLOCK_N={BLOCK_N}: the tile count per "
            "destination has to be a compile-time constant for the modulo test"
        )
    slice_rows = M // ws
    if rotated and slice_rows % BLOCK_M:
        raise ValueError(
            f"each peer's row slice ({slice_rows}) must be a whole number of "
            f"BLOCK_M={BLOCK_M} tiles, so M must be a multiple of "
            f"world_size*BLOCK_M={ws * BLOCK_M} (got M={M}). Decode shapes are "
            "out of scope for this host anyway -- it pads M below 128."
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

    # Compile-time tile accounting for the completion counters.
    n_blocks_const = N // BLOCK_N
    m_tiles_per_peer = slice_rows // BLOCK_M if slice_rows % BLOCK_M == 0 else 0
    tiles_per_peer = m_tiles_per_peer * n_blocks_const

    # Push granularity. A destination's slice is `m_tiles_per_peer` row-bands of
    # BLOCK_M rows; pushing each band as it completes is what lets every link
    # start streaming early instead of waiting for its whole slice.
    chunks = cfg.counter_chunks if fuse else 1
    if fuse and m_tiles_per_peer % chunks:
        raise ValueError(
            f"counter_chunks={chunks} must divide the {m_tiles_per_peer} "
            f"BLOCK_M={BLOCK_M} row-bands in each peer's slice"
        )
    m_tiles_per_chunk = max(1, m_tiles_per_peer // chunks)
    tiles_per_chunk = m_tiles_per_chunk * n_blocks_const
    chunk_bytes = cfg.slice_bytes // chunks

    if fence not in (
        "all", "agent", "agent-leader", "nt-agent", "leader", "none",
        "writethrough", "wt-agent", "raw-wt", "raw-wt-agent", "raw-wt-leader",
    ):
        raise ValueError(
            f"fence must be all/agent/leader/none/writethrough/wt-agent, got {fence!r}"
        )
    _kname_tag = f"{'B' if blockscale else ''}{'S' if swap_ab else ''}{'P' if permlane else ''}{'T' if lane_transpose else ''}{'H' if hoist_scales else ''}{'U' if peer_uncached else ''}{direct_fence[0]}{'W' if store_probe else ''}c{chunks}{'r' if rotated else 'l'}q{sdma_queues}{'' if n_stripe == 1 else f's{n_stripe}'}{fence[0]}"
    counter_off = cfg.counter_off
    lock_off = cfg.lock_off
    in_off = cfg.input_off
    my_recv_slot = cfg.recv_slot_off(rank) if cfg.recv_slots else 0
    slice_bytes = cfg.slice_bytes

    _kname = (
        f"mori_fused_{(transport if fuse else 'split')}_8w_"
        f"{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_k{K}_{_kname_tag}"
        f"{'p' if emit_put else 'x'}{atomic_order[0]}_r{rank}"
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
    def kernel_gemm_scatter(
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
            # Destination-rotated, chunk-major tile order. Linear order finishes
            # destination 0's whole slice, then 1's, ...  -- so the last
            # destination's link only starts at the end of the GEMM and the
            # transfer cannot overlap anything. Walking (chunk, dest, n) instead
            # gets every link streaming within the first 1/(chunks*world) of the
            # GEMM. The `+ rank` rotation keeps all 8 ranks from pushing at the
            # same destination simultaneously (gcnasm's opus_direct_stripe_tile).
            # split_row_major_2d(i, n) -> (i // n, i % n)
            if const_expr(n_stripe >= n_blocks_const):
                rest, bn = split_row_major_2d(fx.block_idx.x, n_blocks)
                tile_i, dest_seq = split_row_major_2d(rest, ws)
            else:
                # gcnasm's opus_direct_stripe_tile: rotate the destination
                # every `n_stripe` tiles of the other axis rather than after a
                # whole chunk, so the peer stores of the resident blocks are
                # spread over all links instead of arriving in same-destination
                # bursts. It delays chunk completion by n_groups, which is why
                # gcnasm keeps it out of its chunked path and so do we by
                # leaving the default at 1.
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

        if const_expr(swap_ab):
            # Tile counts swap with the operands so Mfma's own asserts and its
            # idx() line up; the store then addresses the accumulator as
            # idx(tj, ti).
            mfma_raw = Mfma16x16x128(N_TILES_B, N_TILES_A)
            mfma = _SwappedMfma(mfma_raw)
        else:
            mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        w_pre = cco.Window(win)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        if const_expr(swap_ab and permlane and lane_transpose and not direct_lsa):
            store_c = _LaneTransposeStoreC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )
        elif const_expr(swap_ab and permlane and store_probe and not direct_lsa):
            store_c = _AdjacentLaneProbeC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )
        elif const_expr(swap_ab and permlane and not direct_lsa):
            store_c = _PermlaneStoreC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )
        elif const_expr(swap_ab and store_probe and not direct_lsa):
            store_c = _WideStoreProbeC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )
        elif const_expr(swap_ab and not direct_lsa):
            store_c = _SwapABStoreC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )
        elif const_expr(direct_lsa and swap_ab and permlane and lane_transpose):
            dest_blk = block_m // fx.Int32(m_tiles_per_peer)
            _cls = (
                _LaneTransposeUncachedPeerStoreC
                if const_expr(peer_uncached)
                else _LaneTransposeStoreC
            )
            store_c = _cls(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B,
                peer_rsrc=create_buffer_resource_from_addr(
                    wave_uniform_i64(w_pre.lsa_ptr(dest_blk, my_recv_slot)),
                    num_records_bytes=cfg.slice_bytes,
                ),
                elem_base=dest_blk * fx.Int32(slice_rows) * c_n,
            )
        elif const_expr(direct_lsa and swap_ab and permlane):
            dest_blk = block_m // fx.Int32(m_tiles_per_peer)
            store_c = _PermlaneStoreC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B,
                peer_rsrc=create_buffer_resource_from_addr(
                    wave_uniform_i64(w_pre.lsa_ptr(dest_blk, my_recv_slot)),
                    num_records_bytes=cfg.slice_bytes,
                ),
                elem_base=dest_blk * fx.Int32(slice_rows) * c_n,
            )
        elif const_expr(direct_lsa and swap_ab and store_probe):
            dest_blk = block_m // fx.Int32(m_tiles_per_peer)
            store_c = _WideStoreProbeC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B,
                peer_rsrc=create_buffer_resource_from_addr(
                    wave_uniform_i64(w_pre.lsa_ptr(dest_blk, my_recv_slot)),
                    num_records_bytes=cfg.slice_bytes,
                ),
                elem_base=dest_blk * fx.Int32(slice_rows) * c_n,
            )
        elif const_expr(direct_lsa):
            # dest is uniform across the block; readfirstlane keeps the peer
            # descriptor scalar and avoids a waterfall around every store.
            dest_blk = block_m // fx.Int32(m_tiles_per_peer)
            store_c = _PeerDirectStoreC(
                A_scale,
                B_scale,
                C,
                c_m,
                c_n,
                mfma.idx,
                N_TILES_A,
                N_TILES_B,
                peer_rsrc=create_buffer_resource_from_addr(
                    wave_uniform_i64(w_pre.lsa_ptr(dest_blk, my_recv_slot)),
                    num_records_bytes=cfg.slice_bytes,
                ),
                elem_base=dest_blk * fx.Int32(slice_rows) * c_n,
            )
        elif const_expr(fence in ("writethrough", "wt-agent")):
            store_c = _WriteThroughStoreC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )
        elif const_expr(fence in ("raw-wt", "raw-wt-agent", "raw-wt-leader")):
            store_c = _RawWriteThroughStoreC(
                A_scale,
                B_scale,
                C,
                c_m,
                c_n,
                mfma.idx,
                N_TILES_A,
                N_TILES_B,
                c_rsrc=create_buffer_resource_from_addr(
                    fx.Int64(cco.Window(win).lsa_ptr(rank, in_off)),
                    num_records_bytes=cfg.nbytes,
                ),
            )
        elif const_expr(fence == "nt-agent"):
            store_c = _NonTemporalStoreC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )
        else:
            store_c = StoreC(
                A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B
            )
        # Bound unconditionally: the kernel body is re-parsed by FlyDSL's AST
        # rewriter, and names that only exist inside a branch are not reliably
        # visible to a nested def afterwards (a `nonlocal` on one raised "no
        # binding" and reading one raised UnboundLocalError).
        bsk = nb0 = base_row_pre = None
        if blockscale:
            # Set after construction rather than threading a kwarg through
            # thirteen call sites: it is a trace-time Python bool read by
            # ``const_expr`` and nothing looks at it before the first store.
            store_c._preapplied = True
            bsk = _BlockScaleK(
                A_scale, B_scale, c_m, N, K, swap_ab=swap_ab, n_tiles_a=N_TILES_A
            )
            # The B scale is one scalar per 128-column block, and each
            # accumulator set sits inside exactly one: c*0 is block
            # ``block_n * (BLOCK_N // 128)``, c*1 the next.
            nb0 = block_n * fx.Int32(BLOCK_N // 128)
            # Same expression the epilogue rebuilds at ``base_row`` below; the
            # rescale needs it one loop earlier.
            base_row_pre = block_m * BLOCK_M + wave_m * (N_TILES_A * 16)
        # No second accumulator: the running sum lives in the MFMA registers,
        # rescaled between K-blocks (see _BlockScaleK).
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
                (c00_frag, c01_frag, c10_frag, c11_frag), prev_scales = (
                    bsk.rescale_for(
                        k,
                        (c00_frag, c01_frag, c10_frag, c11_frag),
                        prev_scales,
                        base_row=base_row_pre, nb0=nb0,
                        idx_fn=mfma.idx, n_tiles_b=N_TILES_B,
                        lds_block_m=LDS_BLOCK_M,
                    )
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
            # aiter has `2 * N_LDS_STEPS_A + N_LDS_STEPS_B` here, which lets
            # too many global->LDS prefetches stay in flight across the barrier:
            # a wave passes it and refills an LDS buffer that another wave has
            # not finished consuming. The result is a silent, non-deterministic
            # partial corruption -- always a whole (wave_m sub-tile x all four
            # wave_n) region, which is the signature of a shared A-side LDS
            # buffer -- that only shows up on large grids. Measured boundary at
            # [16384, 7168] K=512 on 8x MI355X, 4-5 runs each: at 256x256 the
            # counts 6 (aiter's), 5 and 4 all corrupt while 3, 2 and 0 are
            # clean; at 128x256 the counts 4 (aiter's) and 3 corrupt while 2, 1
            # and 0 are clean. Both thresholds are N_LDS_STEPS_A +
            # N_LDS_STEPS_B - 1. Costs nothing: 241/240/245 us against
            # 240/246/237 us for aiter's count at [16384, 7168] K=2048.
            #
            # Not a complete theory -- 128x512 does not reproduce at all, so the
            # count is not the only variable, and the same under-wait may exist
            # at the other sync points (forcing *any* of the 14 in this loop to
            # vmcnt(0) also fixes it). This is the smallest change that is
            # measurably correct here, not a proven-general formula.
            wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B - 1)

            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        if blockscale:
            (c00_frag, c01_frag, c10_frag, c11_frag), prev_scales = (
                    bsk.rescale_for(
                        K_ITERS - 2,
                        (c00_frag, c01_frag, c10_frag, c11_frag),
                        prev_scales,
                        base_row=base_row_pre, nb0=nb0,
                        idx_fn=mfma.idx, n_tiles_b=N_TILES_B,
                        lds_block_m=LDS_BLOCK_M,
                    )
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
            (c00_frag, c01_frag, c10_frag, c11_frag), prev_scales = (
                    bsk.rescale_for(
                        K_ITERS - 1,
                        (c00_frag, c01_frag, c10_frag, c11_frag),
                        prev_scales,
                        base_row=base_row_pre, nb0=nb0,
                        idx_fn=mfma.idx, n_tiles_b=N_TILES_B,
                        lds_block_m=LDS_BLOCK_M,
                    )
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
            # Undo the rescaling invariant: the accumulator holds the true sum
            # divided by the last K-block's scale. store_c._preapplied makes the
            # epilogue skip its own scale loads.
            c00_frag, c01_frag, c10_frag, c11_frag = bsk.final_scale(
                (c00_frag, c01_frag, c10_frag, c11_frag), prev_scales,
                idx_fn=mfma.idx, n_tiles_b=N_TILES_B,
            )

        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        # Close the half-wave barrier pairing the prologue opened. Its
        # `if wave_m == 1: s_barrier()` gives waves 4-7 one extra barrier, and
        # s_barrier is a counting rendezvous, so from then on waves 0-3 run one
        # phase ahead -- which is the intended stagger for the double-buffered
        # main loop, and wrong for anything that comes after it. gcnasm closes
        # the pair here too and does it unconditionally
        # (opus_gemm_a2a_lsa/gemm_a16w16_quad_subtile_kernel_template.hpp:693,
        # outside its `if constexpr (ChunkFused)`); aiter's 8-wave kernel opens
        # the stagger at gemm_a8w8_8wave.py:449 and never closes it.
        #
        # Not gated on `fuse`: with the offset live, `wait_barrier(0)` after
        # store_c does not mean "every wave's C tile has retired" -- it
        # rendezvouses waves 0-3, which hold thread 0 and therefore the counter
        # and the transfer, with waves 4-7 still sitting at the previous barrier,
        # before their stores. That is what made --chunks 2 fail one run in
        # three. The split path has nothing after store_c so it cannot observe
        # the imbalance today, but leaving the counts unbalanced makes the next
        # thing added after the epilogue silently wrong, which is exactly how
        # this bug got here.
        if wave_m == 0:
            rocdl.s_barrier()

        if const_expr(hoist_scales) and hasattr(store_c, "store_all"):
            store_c.store_all(
                [(c00_frag, 0, 0), (c01_frag, 0, 1),
                 (c10_frag, 1, 0), (c11_frag, 1, 1)],
                base_row, base_col, LDS_BLOCK_M, LDS_BLOCK_N,
            )
        else:
            store_c.store(c00_frag, base_row + 0, base_col + 0)
            store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
            store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
            store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)
        # ---- end pinned copy ----

        if const_expr(direct_lsa):
            # Publish the peer stores from the blocks that made them. This is the
            # one thing Direct LSA cannot inherit from the split path: in the LSA
            # 2-stage all-reduce the producing kernel fences (ar/kernels_lsa.py,
            # every lane of every block, right after the tmp stores), whereas
            # here the producer is the GEMM and the only fence was in the
            # separate barrier kernel -- one block, hence one XCD's L2 out of
            # eight. The other seven XCDs kept the peer-homed lines dirty and the
            # LSA flag, being a system-scope atomic, overtook them.
            wait_barrier(0)
            # leaderOnly is legal here only because the barrier pair above is
            # now closed: wait_barrier(0) really does mean every wave's stores
            # have retired into this CU's L2, so one wave writing it back covers
            # all eight. It was measured as incorrect before that fix, which is
            # what made it look like a per-wave release was mandatory.
            raw_cco.cco_system_fence(fx.Int32(1 if direct_fence == "leader" else 0))

        if const_expr(fuse and not direct_lsa):
            # Retire every lane's C stores, then agree block-wide that they are
            # retired. `wait_barrier` is aiter's own `s_waitcnt vmcnt(0);
            # s_barrier` pair, reused so the tail matches the main loop's idiom.
            wait_barrier(0)
            # Then publish them. This fence is not optional: dropping it -- on the
            # theory that the copy engine reads through our own L2 -- still
            # validates on 2 ranks but takes relL2 from 2.35e-3 to 3.76e-3 on 8,
            # i.e. the engine reads some tiles stale. Every lane fences, which is
            # the form cco_device_wrapper.cpp:105-111 documents for producer
            # lanes; leaderOnly orders thread 0 only and is explicitly not a
            # substitute for a release issued by every producer.
            if const_expr(fence == "all"):
                raw_cco.cco_system_fence(fx.Int32(0))
            elif const_expr(fence == "leader"):
                raw_cco.cco_system_fence(fx.Int32(1))
            elif const_expr(fence in ("agent", "wt-agent", "nt-agent", "raw-wt-agent")):
                release_fence("agent")
            # "writethrough": nothing to do -- the stores already went to memory,
            # and wait_barrier(0) above retired them.

            w = cco.Window(win)
            sdma = cco.DevComm(dev_comm).sdma()
            if fx.thread_idx.x == 0:
                # One release per *block* instead of per wave. cco's leaderOnly
                # form was already shown incorrect, but that went through an
                # extern wrapper; emitting the fence inline rules out the
                # compiler having sunk it past the atomic.
                if const_expr(fence in ("agent-leader", "raw-wt-leader")):
                    release_fence("agent")
                dest = block_m // fx.Int32(m_tiles_per_peer)
                chunk = (block_m % fx.Int32(m_tiles_per_peer)) // fx.Int32(
                    m_tiles_per_chunk
                )
                slot = dest * fx.Int32(chunks) + chunk
                ctr = signal_ptr(
                    fx.Int64(w.lsa_ptr(rank, counter_off))
                    + fx.Int64(slot) * fx.Int64(4)
                )
                seq = fx.Int32(
                    atomic_add_u32(ctr, 1, ordering=atomic_order)
                ) + fx.Int32(1)
                # Monotonic: the counters are never reset, so "last tile of this
                # chunk, this epoch" is a modulo test rather than a compare. Same
                # reason the barrier flags in ar/kernels_lsa are never reset -- it
                # is what makes graph replay behave like a fresh launch.
                if seq % fx.Int32(tiles_per_chunk) == fx.Int32(0):
                    if const_expr(emit_put) and dest != fx.Int32(rank):
                        # One queue per destination. Two chunks of the same
                        # destination can be elected at nearly the same moment and
                        # will then post to the same queue; that serialises the
                        # short issue only, and they share one xGMI link either
                        # way, so a queue each would buy nothing.
                        off = fx.Int64(chunk) * fx.Int64(chunk_bytes)
                        lock = signal_ptr(
                            fx.Int64(w.lsa_ptr(rank, lock_off))
                            + fx.Int64(dest) * fx.Int64(4)
                        )
                        if const_expr(chunks > 1):
                            _acquire_peer_lock(lock)
                        sdma.put(
                            dest,
                            win,
                            fx.Int64(my_recv_slot) + off,
                            win,
                            fx.Int64(in_off)
                            + fx.Int64(dest) * fx.Int64(slice_bytes)
                            + off,
                            fx.Int64(chunk_bytes),
                            # Queues are per (source, destination) pair, so
                            # every peer gets its own even at sdma_queues=1;
                            # allocating one per peer *per pair* only wastes
                            # hardware queues. See build_sdma_phases.
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
        kernel_gemm_scatter(
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


LSA_NOTE = """\
The LSA transport is not implemented, and the reason is a measurement rather than
a preference.

Fusing LSA means the epilogue stores into a peer instead of into local C. But
`StoreC._store_bf16` writes **one bf16 at a time** through `BufferCopy16b`, at
`c_index = (row + i) * c_cols + col` with `row = base_row + ti*16 + lane//16*4 + i`
and `col = base_col + tj*16 + lane%16` -- a lane-scatter. gcnasm measured that
pattern at 0.26x when the destination is a peer, so the variant is worthless
until `StoreC` is coalesced (their winning form was a wave-local `ds_bpermute`
pair-coalesced store; staging C through LDS regressed 1-3%).

That rewrite is the expensive half of this step, and the ceiling it is competing
for is small: at [4096, 7168] the GEMM is ~40us against an all-reduce of ~290us
(LSA) or ~340us (SDMA), so even perfect overlap removes at most the GEMM, ~12%.
Fusing the cheap transport first establishes whether the completion-counter
structure works at all; if the answer is no, the `StoreC` rewrite is wasted
regardless of transport.
"""

__all__ = ["compile_fused_gemm_scatter", "LSA_NOTE", "BLOCK_K"]
