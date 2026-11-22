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
# SPDX-License-Identifier: Apache-2.0
"""aiter's GEMM1 ``do_tile`` (aiter 849f711d3, ``mega_moe/gemm1.py``) cut into phases, for
``tile_opt`` (``hoist_prologue`` and the K-loop changes) and for the per-tile stamps of a trace build. Same
arithmetic, same MFMA order.

Only the configuration GEMM1 uses here: ``pipe_weights``, ``async_a_copy``, ``mfma_amajor``,
indirect input, no ``pair_k``, fp8 output (``evec`` 8).

* ``tile_main``: from "A k-step 0 and the A scales are in LDS (or on their way)" to the last MFMA --
  the barrier that closes the prologue, the first B / scale loads, the K loop, the final step. Returns the
  accumulators. ``hook`` runs once, after the first loads are issued and before the K loop.
* ``KPIPE`` (0 = aiter's K loop): 1 = the next A k-step goes global -> registers at the step start and to
  LDS (``ds_write``) at its end, instead of ``buffer_load ... lds``. LLVM's waitcnt pass makes the step's
  first A ``ds_read`` wait for the outstanding LDS DMA (it cannot prove the other A buffer disjoint), so
  with the DMA every step waits out the A load latency up front.
* ``FINOFF`` (any ``KPIPE``; same results): the final k-step's A buffer parity goes through an empty inline
  asm (the offset is that times ``a_lds_i32``, so its alignment stays known). Otherwise
  ``(last & 1) * a_lds_i32`` is a constant: LLVM folds it into the final step's A ``ds_read`` addresses
  (32 at sort_block_m 128), hoists them out of the persistent tile loop and, short of VGPRs, spills them,
  so every tile reloads them from scratch.
* ``KUNROLL`` (with ``KPIPE`` 1; 1 = off): 2 = the K loop body is two k-steps, so the B registers alternate
  between two sets (even step reads set X and loads set Y, odd step the reverse) and the loop carries the
  set it loaded last: no copy of the next-step B registers at the loop head and no ``vmcnt(0)`` before
  it. An odd number of loop steps puts the last one after the loop. Same MFMAs in the same order. A
  ``sched_barrier`` keeps the step-end A ``ds_write`` at the step end (with the parity of each step
  known, LLVM would otherwise hoist it to the step start, where it waits out the A global load).
* ``ADBUF`` (with ``KPIPE`` 1; 0 = off; same results): the A operand is double-buffered in registers
  (``_pipe_am_dbuf``): MFMA group g + 1's A ``ds_read`` pair is issued before group g's 4 MFMAs, with a
  ``sched_barrier`` on both sides of the MFMAs so LLVM keeps that order, so the pair's LDS latency runs
  under the 4 MFMAs (otherwise every group waits ``lgkmcnt(0)`` for its own pair, issued just before).
  Same MFMAs in the same order; the final k-step too.
* ``do_tile_split``: aiter's ``do_tile`` = prologue + ``tile_main`` + epilogue (trace builds, and
  ``fin_off`` without ``hoist_prologue``).
* the prologue in pieces, so it can run inside the previous tile's epilogue: ``rows_raw`` (the
  rowmap words of the tile's rows), ``stage_issue`` (A k-step 0 and the A scales, global ->
  registers), ``stage_store`` (both to LDS, ``do_tile``'s layout). A k-step 0 goes through
  registers, not straight to LDS (``buffer_load ... lds``): LLVM's waitcnt pass makes every later
  LDS access wait for an outstanding LDS DMA it cannot prove disjoint, so a DMA issued inside the
  epilogue would hold the epilogue's C-tile reads. Through registers the bytes land in LDS where
  and how the DMA puts them (logical 16-byte chunk ``c`` of row ``r`` at chunk ``c ^ (r & 15)``).
  ``sa_pack_*`` (sort_block_m 128): the A scales instead go to LDS packed, one 16-byte record per
  (k-step, lane), read with one ``ds_read_b128`` per k-step.
* the epilogue in two pieces around its barrier: ``epi_stage_c`` (SiLU, C tile to LDS, barrier),
  then aiter's ``SiluQuantEpilogue._store_quant_vec8`` (quantise, store), or ``store_quant_remat``
  (the same, its lane constants recomputed per tile: ``epi_remat``).

Trace stamps (thread 0, 100 MHz) at ``tbuf[tslot + 1]`` (K loop starts) and ``tbuf[tslot + 3]``
(K loop done). Keep in step with aiter's ``do_tile``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.mega_moe.gemm_util import (
    _PACK,
    ATileLoader,
    _e8m0_scale,
    _group_absmax,
    wait_lds_barrier,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)

from sdma_dispatch import memrealtime


class HoistATileLoader(ATileLoader):
    """``ATileLoader`` (indexed input) whose ``for_tile`` is split in two: ``rows_raw`` loads the
    rowmap words, ``set_rows`` does the rest. Each lane loads its own row's word, which is the word
    ``for_tile`` loads on the subgroup's first lane and broadcasts (``_load_row_map_subgroup``).
    """

    def rows_raw(self, tile_row_base_i32):
        assert self._indexed_input and self._async_copy
        chunks_per_row = self._k_step_bytes // 16
        total_chunks = self._sort_block_m * chunks_per_row
        out = []
        for c in range_constexpr(0, total_chunks, self._total_threads):
            lin = fx.Int32(c) + fx.Int32(self._tx)
            row = lin // fx.Int32(chunks_per_row)
            out.append(fx.Int32(self._row_map_rsrc[tile_row_base_i32 + row]))
        return out

    def set_rows(self, raws):
        """``for_tile`` (indexed input) with the rowmap words already loaded."""
        input_iter = fx.get_iter(self._x_tensor)
        input_bytes = self._source_rows * self._row_bytes
        input_view = fx.Tensor(fx.make_view(input_iter, fx.make_layout(input_bytes, 1)))
        self._tile_rsrc = ptr_buf_tensor(
            fx.get_iter(input_view),
            fx.Int32,
            unit_elems=4,
            num_records_bytes=input_bytes,
        )
        tile_buffer = fx.rocdl.make_buffer_tensor(
            input_view, max_size=False, num_records_bytes=input_bytes
        )
        self._tile_dma = fx.logical_divide(tile_buffer, fx.make_layout(1, 1))
        chunks_per_row = self._k_step_bytes // 16
        row_stride_i32 = self._k_step_bytes // 4
        total_chunks = self._sort_block_m * chunks_per_row
        self._chunks = []
        for k, c in enumerate(range(0, total_chunks, self._total_threads)):
            lin = fx.Int32(c) + fx.Int32(self._tx)
            row = lin // fx.Int32(chunks_per_row)
            chunk = lin % fx.Int32(chunks_per_row)
            source_row = raws[k] & fx.Int32(0xFFFFFF)
            row_byte = source_row * fx.Int32(self._row_bytes)
            if const_expr(self._swizzle):
                col_i32 = chunk * 4
                swizzle_mask = (
                    (row & 14) << 1 if const_expr(self._is_fp4) else (row & 15) << 2
                )
                swz = row * fx.Int32(row_stride_i32) + (col_i32 ^ swizzle_mask)
                lds_byte = swz * fx.Int32(4)
            else:
                lds_byte = lin * 16
            self._chunks.append((lds_byte, row_byte + chunk * 16))


def opaque_lane():
    """``thread_idx.x`` through an empty inline asm (a VGPR tied to itself, side effects), so LLVM
    cannot hoist it, or the lane constants built on it, out of the tile loop: they are recomputed
    where used (a few VALU ops) instead of being computed once, spilled, and reloaded from scratch
    every tile with a ``vmcnt`` wait that also waits for every load and store in flight
    (``epi_remat``)."""
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm as _llvm

    v = fx.Int32(fx.thread_idx.x)
    r = _llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [v.ir_value()],
        "; lane $0",
        "=v,0",
        has_side_effects=True,
    )
    return fx.Int32(r)


def sa_pack_rows(a_gather, tile_row_base_i32, remat=False):
    """``SA_PACK``: the rowmap words of this lane's four A-scale rows (``sa_pack_issue``)."""
    lane = opaque_lane() if remat else fx.Int32(fx.thread_idx.x)
    lrow = (lane // fx.Int32(16)) % fx.Int32(16)
    gp = lane // fx.Int32(256)
    return [
        fx.Int32(
            a_gather._row_map_rsrc[
                tile_row_base_i32 + gp * fx.Int32(64) + fx.Int32(r) + lrow
            ]
        )
        for r in (0, 16, 32, 48)
    ]


def sa_pack_issue(a_scale, sa_raws, remat=False):
    """``SA_PACK`` (sort_block_m 128, 512 threads): the A scales of the tile, global -> registers,
    for ``sa_pack_store``. Lane t takes column group q = t % 16 (16 scales, k-steps 2q and 2q + 1),
    lane row l = (t // 16) % 16 and scale groups g = 2 (t // 256) + {0, 1}: rows g * 32 + {0, 16} + l,
    the four rows whose bytes make two of ``AScaleLoader.load_step``'s packed words."""
    groups_per_row = a_scale._n_scale // 16
    q = (opaque_lane() if remat else fx.Int32(a_scale._tx)) % fx.Int32(16)
    active = q < fx.Int32(groups_per_row)
    vs = []
    for k in range_constexpr(4):
        source_row = sa_raws[k] & fx.Int32(0xFFFFFF)
        vs.append(
            buf_copy_load(
                a_scale._rsrc,
                source_row * fx.Int32(a_scale._row_groups)
                + fx.Int32(a_scale._first_group)
                + active.select(q, fx.Int32(0)),
                fx.Int32,
                unit_elems=4,
            )
        )
    return active, vs


@flyc.jit
def _scale_store64(lds_ascale, lin2, active, w0, w1):
    if active:
        dst = fx.make_view(
            fx.add_offset(fx.recast_iter(fx.Int32, lds_ascale.ptr), lin2 * fx.Int32(2)),
            fx.make_layout(2, 1),
        )
        fragment = fx.make_rmem_tensor(2, fx.Int32)
        fragment.store(Vec.from_elements([w0, w1], fx.Int32))
        fx.copy(fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32), fragment, dst)


def sa_pack_store(a_scale_lds, issued, remat=False):
    """``sa_pack_issue``'s registers to LDS, packed: the 16-byte record of (k-step, lane // 16, lane
    % 16) holds ``AScaleLoader.load_step``'s packed word of scale groups 0..3, so a k-step's A scales
    are one ``ds_read_b128`` (``PackedWindowAScaleLoader``). Lane t writes 8 bytes (groups 2 (t //
    256) + {0, 1}) of 8 records: k-steps 2q + s, lane // 16 = j."""
    active, vs = issued
    lane = opaque_lane() if remat else fx.Int32(fx.thread_idx.x)
    q = lane % fx.Int32(16)
    lrow = (lane // fx.Int32(16)) % fx.Int32(16)
    gp = lane // fx.Int32(256)
    d = [
        [Vec(v) for v in vs[2 * gg : 2 * gg + 2]] for gg in range(2)
    ]  # [g'][half], 4 x i32 each

    def byte(x, j):
        return (x >> fx.Int32(8 * j)) & fx.Int32(0xFF)

    for s in range_constexpr(2):
        for j in range_constexpr(4):
            w = []
            for gg in range_constexpr(2):
                lo, hi = d[gg][0], d[gg][1]
                w.append(
                    byte(fx.Int32(lo[2 * s]), j)
                    | (byte(fx.Int32(hi[2 * s]), j) << fx.Int32(8))
                    | (byte(fx.Int32(lo[2 * s + 1]), j) << fx.Int32(16))
                    | (byte(fx.Int32(hi[2 * s + 1]), j) << fx.Int32(24))
                )
            rec = (
                (q * fx.Int32(2) + fx.Int32(s)) * fx.Int32(4) + fx.Int32(j)
            ) * fx.Int32(16) + lrow
            _scale_store64(a_scale_lds, rec * fx.Int32(2) + gp, active, w[0], w[1])


def stage_issue(a_gather, a_scale, raws, scales=True):
    """The prologue's loads for the tile whose rowmap words are ``raws``, into registers: A k-step
    0 (``ATileLoader.load_regs``) and the A scales (``WindowAScaleLoader.stage``'s, one 16-byte
    group per lane and round; a lane whose group is past the row's scales loads group 0 of the row,
    in bounds, and stores nothing). For ``stage_store``."""
    a_gather.set_rows(raws)
    a_regs = a_gather.load_regs(fx.Int32(0))
    if not scales:
        return a_regs, []
    groups_per_row = a_scale._n_scale // 16
    padded = max(16, groups_per_row)
    out = []
    for k, c in enumerate(
        range(0, a_scale._sort_block_m * padded, a_scale._total_threads)
    ):
        slot = fx.Int32(c) + fx.Int32(a_scale._tx)
        row = slot // fx.Int32(padded)
        group = slot % fx.Int32(padded)
        active = group < fx.Int32(groups_per_row)
        source_row = raws[k] & fx.Int32(0xFFFFFF)
        v = buf_copy_load(
            a_scale._rsrc,
            source_row * fx.Int32(a_scale._row_groups)
            + fx.Int32(a_scale._first_group)
            + active.select(group, fx.Int32(0)),
            fx.Int32,
            unit_elems=4,
        )
        out.append((row * fx.Int32(groups_per_row) + group, active, v))
    return a_regs, out


@flyc.jit
def _scale_store(lds_ascale, lin, active, v):
    if active:
        dst = fx.make_view(
            fx.add_offset(fx.recast_iter(fx.Int32, lds_ascale.ptr), lin * fx.Int32(4)),
            fx.make_layout(4, 1),
        )
        fragment = fx.make_rmem_tensor(4, fx.Int32)
        fragment.store(v)
        fx.copy(fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32), fragment, dst)


def stage_store(a_gather, a_buf, a_scale_lds, issued):
    """``stage_issue``'s registers to LDS: A k-step 0 at ``a_buf`` offset 0 (the pool, which the
    previous tile's C tile no longer uses once its epilogue's closing barrier has passed), the A
    scales in ``WindowAScaleLoader.stage``'s layout."""
    a_regs, scales = issued
    a_gather.store(a_buf, a_regs, 0)
    for lin, active, v in scales:
        _scale_store(a_scale_lds, lin, active, v)


def epi_stage_c(epi, acc):
    """First half of ``SiluQuantEpilogue.store`` (fp8 out, ``evec`` 8): act(gate) * up of the
    accumulators to the C tile in LDS, then the barrier. Returns what ``_store_quant_vec8`` takes.
    """
    assert epi._evec == 8 and epi._out_tensor is not None
    combined = epi._combine(acc)
    n_per = len(combined) // epi._m_repeat
    tx = fx.thread_idx.x
    wave = tx // fx.Int32(64)
    lane = tx % fx.Int32(64)
    l16 = lane % fx.Int32(16)
    ld4 = (lane // fx.Int32(16)) * fx.Int32(4)
    cs_tile_n = epi._tile_n // 2
    nwo = cs_tile_n // epi._num_waves
    cbase = wave * fx.Int32(nwo)
    cptr = epi._lds_out.ptr
    for mi in range_constexpr(epi._m_repeat):
        for nj in range_constexpr(n_per):
            v4 = Vec(combined[mi * n_per + nj])
            col = cbase + fx.Int32(nj * 16) + l16
            for ii in range_constexpr(4):
                row = fx.Int32(mi * 16) + ld4 + fx.Int32(ii)
                idx = row * fx.Int32(cs_tile_n) + col
                ptr = fx.add_offset(cptr, fx.make_int_tuple(idx))
                fx.ptr_store(Vec.from_elements([v4[ii]], fx.Float32), ptr)
    gpu.barrier()
    return cbase, cptr, cs_tile_n


def store_quant_remat(
    epi, tile_i32, tile_row_base_i32, n_tile_base_i32, cbase, cptr, cs_tile_n
):
    """aiter's ``SiluQuantEpilogue._store_quant_vec8`` (``epi_remat``) with the lane id from ``opaque_lane``:
    same values, same stores in the same order, but the lane constants are recomputed per tile instead of
    hoisted out of the tile loop, spilled and reloaded."""
    EVEC = 8
    NLANE = cs_tile_n // EVEC
    assert 64 % NLANE == 0 and NLANE * EVEC == cs_tile_n
    tx = opaque_lane()
    c64 = fx.Int32(64)
    n32 = fx.Int32(epi._sorted_scale_cols_i32 * 32)
    rows_per_iter = epi._num_waves * 64 // NLANE
    mlane = tx // fx.Int32(NLANE)
    nlane = tx % fx.Int32(NLANE)
    m_reps = epi._sort_block_m // rows_per_iter
    assert m_reps * rows_per_iter == epi._sort_block_m
    out_tile_base = n_tile_base_i32 // fx.Int32(2) - cbase
    out_vec = epi._tile_out(tile_i32, fx.Int32, unit_elems=2)
    col0 = nlane * fx.Int32(EVEC)
    gcol = out_tile_base + col0
    is_writer = (nlane & fx.Int32(3)) == fx.Int32(0)
    col_s = gcol >> fx.Int32(5)
    d3 = col_s >> fx.Int32(3)
    d4 = (col_s >> fx.Int32(2)) & fx.Int32(1)
    d5 = col_s & fx.Int32(3)
    for mr in range_constexpr(m_reps):
        row = fx.Int32(mr * rows_per_iter) + mlane
        row_g, valid, out_row_base = epi._row_target(tile_i32, tile_row_base_i32, row)
        idx = row * fx.Int32(cs_tile_n) + col0
        f32_iter = fx.recast_iter(
            fx.Float32, fx.add_offset(cptr, fx.make_int_tuple(idx))
        )
        frag = fx.make_view(f32_iter, fx.make_layout(EVEC, 1)).load()
        vals = [frag[i] for i in range_constexpr(EVEC)]
        m = _group_absmax(vals, (1, 2), c64)
        e8m0_v, quant_scale = _e8m0_scale(m)
        words = []
        for w in range_constexpr(2):
            sc = [vals[w * 4 + i] * quant_scale for i in range_constexpr(4)]
            packed = rocdl.cvt_pk_fp8_f32(T.i32, sc[0], sc[1], fx.Int32(0), 0)
            packed = rocdl.cvt_pk_fp8_f32(T.i32, sc[2], sc[3], packed, 1)
            words.append(fx.Int32(packed))
        d0 = row_g >> fx.Int32(5)
        d1 = (row_g >> fx.Int32(4)) & fx.Int32(1)
        d2 = row_g & fx.Int32(15)
        byte_off = (
            d0 * n32
            + d3 * fx.Int32(256)
            + d5 * fx.Int32(64)
            + d2 * fx.Int32(4)
            + d4 * fx.Int32(2)
            + d1
        )
        byte_off = is_writer.select(byte_off, fx.Int32(0x40000000))
        out_byte = out_row_base + gcol
        out_byte = valid.select(out_byte, fx.Int32(0x40000000))
        buf_copy_store(
            out_vec,
            out_byte // fx.Int32(8),
            Vec.from_elements(words, fx.Int32),
            fx.Int32,
            unit_elems=2,
        )
        epi._out_scale_rsrc[byte_off] = e8m0_v.to(fx.Int8)
    wait_lds_barrier()


def _pipe_am_dbuf(mfma, a_load, b_prev, acc, sa, sb, load_next=None):
    """``MfmaScaleGU.call_pipe_am`` (``call_pipe_am_final`` with ``load_next`` None) with the A
    operand double-buffered: group g + 1's A ``ds_read`` pair (and group g's next-B loads) are
    issued before group g's MFMAs, a ``sched_barrier`` on both sides of the MFMAs. Same MFMAs in the
    same order; the next-B loads go at the same groups."""
    nn = mfma._num_acc_n
    order = [(mi, ks) for mi in range(mfma._m_repeat) for ks in range(_PACK)]
    ngrp = len(order)
    b_next = [None] * nn
    nb = 0
    a_cur = a_load(*order[0])
    for g in range(ngrp):
        mi, ksub = order[g]
        a_nxt = a_load(*order[g + 1]) if g + 1 < ngrp else None
        if load_next is not None:
            tgt = ((g + 1) * nn) // ngrp
            while nb < tgt:
                b_next[nb] = load_next(nb)
                nb += 1
        rocdl.sched_barrier(0)
        for ni in range_constexpr(nn):
            aidx = mfma.idx(mi, ni)
            acc[aidx] = mfma._mfma(
                a_cur,
                b_prev[ni][ksub],
                acc[aidx],
                sa[mi // _PACK],
                sb[ni // _PACK],
                ksub,
                mi % _PACK,
                ni % _PACK,
            )
        rocdl.sched_barrier(0)
        a_cur = a_nxt
    if load_next is None:
        return acc
    while nb < nn:
        b_next[nb] = load_next(nb)
        nb += 1
    return acc, b_next


def _opaque_i32(v):
    """``v`` through an empty inline asm (an SGPR tied to itself): same value, unknown to LLVM."""
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm as _llvm

    r = _llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [v.ir_value() if hasattr(v, "ir_value") else v],
        "; fin_off $0",
        "=s,0",
        has_side_effects=True,
    )
    return fx.Int32(r)


# fmt: off
@flyc.jit
def tile_main(b_row, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, a_buf, a_scale_lds, a_lds_i32,
              K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, front_vmcnt, tbuf, tslot, ttrace, hook,
              KPIPE, KUNROLL, FINOFF, ADBUF):
# fmt: on
    N_ACC = M_REPEAT * NUM_ACC_N
    NUM_B_SCALE = NUM_ACC_N // _PACK
    NUM_A_SCALE = M_REPEAT // _PACK
    B_STATE_END = N_ACC + NUM_ACC_N * _PACK
    SB_STATE_END = B_STATE_END + NUM_B_SCALE
    last = fx.Int32(K_ITERS - 1)
    # closes the prologue: ``do_tile`` waits for its A DMA (vmcnt 0); a front written with ds_write
    # (``stage_store``) needs only the LDS counter (63 = no VMEM wait)
    wait_lds_barrier(front_vmcnt)
    b0 = b_loader.load_step(b_row, fx.Int32(0))
    init = [mfma.zero_value for _ in range(N_ACC)]
    init += [h for ni_list in b0 for h in ni_list]
    init += b_scale.load_step(b_row, fx.Int32(0))
    init += a_scale.load_step(a_scale_lds, fx.Int32(0))
    if const_expr(hook is not None):
        hook()
    if const_expr(ttrace):
        if fx.thread_idx.x == fx.Int32(0):
            tbuf[tslot + fx.Int32(1)] = memrealtime()

    def _kstep(sp_i, state, par=None):
        # ``par``: this step's parity when it is known at trace time (``KUNROLL`` 2)
        sp = fx.Int32(sp_i)
        acc = [Vec(a) for a in state[:N_ACC]]
        b_prev = [[Vec(state[N_ACC + ni * _PACK + ks]) for ks in range(_PACK)] for ni in range(NUM_ACC_N)]
        if const_expr(par is None):
            cur_off = (sp & 1) * fx.Int32(a_lds_i32)
            nxt_off = ((sp + 1) & 1) * fx.Int32(a_lds_i32)
        else:
            cur_off = fx.Int32(par * a_lds_i32)
            nxt_off = fx.Int32((1 - par) * a_lds_i32)
        spn = sp + 1
        sb = [fx.Int32(state[B_STATE_END + group]) for group in range_constexpr(NUM_B_SCALE)]
        sa = [fx.Int32(state[SB_STATE_END + group]) for group in range_constexpr(NUM_A_SCALE)]

        def a_load(mi, ks, _base=cur_off):
            return a_s2r.load_operand(a_buf, mi, ks, _base)

        rocdl.sched_barrier(0)
        if const_expr(KPIPE):
            # A k-step sp + 1 global -> registers; to LDS (ds_write) at the end of the step
            a_next = a_gather.load_regs(spn * fx.Int32(A_K_STEP_BYTES))
        else:
            a_gather.prefetch_to_lds(spn * fx.Int32(A_K_STEP_BYTES), a_buf, nxt_off)
        rocdl.sched_barrier(0)
        sb_next = b_scale.load_step(b_row, spn)
        sa_next = a_scale.load_step(a_scale_lds, spn)

        def load_next(ni, _kn=spn):
            return b_loader.load_ni(b_row, ni, _kn)

        if const_expr(ADBUF):
            acc, b_next = _pipe_am_dbuf(mfma, a_load, b_prev, acc, sa, sb, load_next)
        else:
            acc, b_next = mfma.call_pipe_am(a_load, b_prev, acc, sa, sb, load_next)
        if const_expr(KPIPE):
            # the other A buffer was last read in step sp - 1, before that step's closing barrier
            if const_expr(KUNROLL == 2):
                rocdl.sched_barrier(0)
            a_gather.store(a_buf, a_next, nxt_off)
            wait_lds_barrier()
        else:
            wait_lds_barrier(NUM_ACC_N * _PACK + NUM_B_SCALE)
        yv = list(acc) + [h for ni_list in b_next for h in ni_list]
        yv += sb_next
        yv += sa_next
        return yv

    if const_expr(KUNROLL == 2):
        n_pair = (K_ITERS - 1) // 2
        for sp_i, state in range(0, 2 * n_pair, 2, init=init):
            yv = _kstep(sp_i, state, 0)
            yv = _kstep(fx.Int32(sp_i) + fx.Int32(1), yv, 1)
            state = yield yv
        if const_expr((K_ITERS - 1) % 2):
            state = _kstep(K_ITERS - 2, state, (K_ITERS - 2) % 2)
    else:
        for sp_i, state in range(0, K_ITERS - 1, 1, init=init):
            yv = _kstep(sp_i, state)
            state = yield yv
    acc = [Vec(r) for r in state[:N_ACC]]
    b_prev = [[Vec(state[N_ACC + ni * _PACK + ks]) for ks in range(_PACK)] for ni in range(NUM_ACC_N)]
    final_par = last & 1
    if const_expr(FINOFF):
        # the parity, not the offset: the product stays a known multiple of a_lds_i32, so the
        # A ds_reads keep their 16-byte alignment (an opaque offset splits each ds_read_b128 in two
        # ds_read2_b32)
        final_par = _opaque_i32(final_par)
    final_off = final_par * fx.Int32(a_lds_i32)

    def final_a_load(mi, ks, _base=final_off):
        return a_s2r.load_operand(a_buf, mi, ks, _base)

    sb = [fx.Int32(state[B_STATE_END + group]) for group in range_constexpr(NUM_B_SCALE)]
    sa = [fx.Int32(state[SB_STATE_END + group]) for group in range_constexpr(NUM_A_SCALE)]
    if const_expr(ADBUF):
        acc = _pipe_am_dbuf(mfma, final_a_load, b_prev, acc, sa, sb)
    else:
        acc = mfma.call_pipe_am_final(final_a_load, b_prev, acc, sa, sb)
    if const_expr(ttrace):
        if fx.thread_idx.x == fx.Int32(0):
            tbuf[tslot + fx.Int32(3)] = memrealtime()
    return acc


# fmt: off
def do_tile_split(m_tile, n_tile_base, expert, sched, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi,
                  a_buf, a_scale_lds, a_lds_i32, K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, trb_rsrc,
                  tib_rsrc, tbuf, tslot, ttrace=True, fin_off=0):
# fmt: on
    """aiter's ``do_tile`` (the configuration above) as prologue + ``tile_main`` + epilogue, with
    the K-loop stamps (``ttrace``): the trace build's tile, so a tile splits into prologue / K loop /
    epilogue. ``fin_off``: ``tile_main``'s ``FINOFF``."""
    tile_row_base = trb_rsrc[m_tile]
    tile_input_base = tib_rsrc[m_tile]
    b_row = sched.gate_base_row(expert) + n_tile_base
    a_gather.for_tile(tile_input_base)
    a_gather.prefetch_to_lds(fx.Int32(0), a_buf, fx.Int32(0))
    a_scale.stage(a_scale_lds, tile_input_base)
    acc = tile_main(
        b_row, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, a_buf, a_scale_lds, a_lds_i32,
        K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, 0, tbuf, tslot, ttrace, None, 0, 1, fin_off, 0,
    )
    # The epilogue aliases A_buf as cshuffle LDS after every wave finishes its final A ds_read.
    wait_lds_barrier()
    epi.store(acc, m_tile, tile_row_base, n_tile_base)
