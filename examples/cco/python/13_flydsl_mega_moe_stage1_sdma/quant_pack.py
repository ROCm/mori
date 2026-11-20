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
"""Sender side of the fused SDMA Stage1 with the quantisation folded in (``fuse_quant``).

``pack.build_pack`` copies rows that aiter's prepare has already quantised into a separate
buffer, so every routed row is read once more. With ``fuse_quant`` the prepare only plans
(its quant workgroups are not launched) and two kernels take pack's place:

  route  (``build_route``) the production producer's rows, walked as pack walks them (one
         block per dispatch segment, one lane per row), without moving payload: for each row
         ``inv[token * topk + j] = out_row * 2 + own`` (``j`` the route's top-k slot, the
         lowest member slot for a fanout-group row; ``own``: this rank's row, which goes to
         the landing buffer instead of the send slab), and the route header (srcmap word,
         weight) to the destination over P2P, exactly as pack writes it.
  qpack  (``build_quant_pack``) one block per token: quantise the bf16 row once with aiter's
         per-1x32 MX fp8 arithmetic (``emit_per_1x32_mx_fp8_group``, so the bytes are
         identical), stage it in LDS, store it to each of the token's rows in ``inv`` with 16
         contiguous bytes per lane, and reset those entries to -1 for the next call.

Each bf16 row is read once and each routed row written once. At 8192 tokens/rank (v4_pro,
8x MI355X) route + qpack take 0.033 + 0.083 ms where pack took 0.188 ms. Storing the row
straight from registers (two 16-byte pieces 32 bytes apart per lane) instead of through LDS
made qpack 0.26 ms: the cco window is uncached, and strided partial lines write slowly.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int8, ReductionOp, T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.mega_moe.dispatch import DispatchSlot, _load_fanout_pair
from aiter.ops.flydsl.kernels.mega_moe.quant import (
    _FP8_E4M3_INV_MAX_POS_BITS,
    _load_i32x4,
    _store_i32x4,
)
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import lds_typed_ptr, lds_vec_load
from aiter.ops.flydsl.kernels.tensor_shim import ptr_buf_tensor

ROUTE_THREADS = 256
QP_THREADS = 256


def build_route(*, npes, epr, topk, mtpr, rank):
    TE = npes * epr
    SEGS = TE + npes

    @flyc.kernel(
        name=f"sdma_s1_route_p{npes}_e{epr}_k{topk}_m{mtpr}_r{rank}",
        known_block_size=[ROUTE_THREADS, 1, 1],
    )
    def route(
        addr_disp: fx.Int64,
        wts: fx.Int64,
        ids: fx.Int64,
        parity_p: fx.Int64,
        inv_p: fx.Int64,
    ):
        seg = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        disp = ptr_buf_tensor(addr_disp, fx.Int64)

        def dp(slot):
            return disp[fx.Int32(int(slot))]

        local_hist = ptr_buf_tensor(dp(DispatchSlot.LOCAL_HIST), fx.Int32)
        pair_base = ptr_buf_tensor(dp(DispatchSlot.PAIR_BASE), fx.Int32)
        task_base = ptr_buf_tensor(dp(DispatchSlot.TASK_ROW_BASE), fx.Int32)
        group_base = ptr_buf_tensor(dp(DispatchSlot.GROUP_TASK_BASE), fx.Int32)
        pair_order = ptr_buf_tensor(dp(DispatchSlot.PAIR_ORDER), fx.Int32)
        srcmap_table = ptr_buf_tensor(dp(DispatchSlot.P2P_SRCMAP), fx.Int64)
        weight_table = ptr_buf_tensor(dp(DispatchSlot.P2P_WEIGHT), fx.Int64)
        parity = fx.Int32(comm_ops.load_i32_system(parity_p, fx.Int32(0)))
        group = seg >= fx.Int32(TE)
        dest = group.select(seg - fx.Int32(TE), seg // fx.Int32(epr))
        _, pair_a, _, _, _ = _load_fanout_pair(
            dp(DispatchSlot.FANOUT_PAIR_CONFIG), dest, parity, npes=npes
        )
        count = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(local_hist[seg])))
        src_base = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(pair_base[seg])))
        dst_base = fx.Int32(
            rocdl.readfirstlane(
                T.i32,
                group.select(
                    fx.Int32(group_base[dest * fx.Int32(epr) + pair_a]),
                    fx.Int32(task_base[seg]),
                ),
            )
        )
        own = dest == fx.Int32(rank)
        own_bit = own.select(fx.Int32(1), fx.Int32(0))
        wts_p = ptr_buf_tensor(wts, fx.Float32)
        idx_p = ptr_buf_tensor(ids, fx.Int32)
        inv = ptr_buf_tensor(inv_p, fx.Int32)
        srcmap_remote = ptr_buf_tensor(srcmap_table[dest], fx.Int32)
        weight_remote = ptr_buf_tensor(weight_table[dest], fx.Int32)
        for row in range(tid, count, fx.Int32(ROUTE_THREADS)):
            wk = fx.Int32(pair_order[src_base + row])
            token = group.select(wk & fx.Int32(0xFFFFFF), wk // fx.Int32(topk))
            out_row = own.select(dst_base + row, src_base + row)
            source_key = fx.Int32(rank * mtpr) + token
            if group:
                members = (wk >> fx.Int32(24)) & fx.Int32(0xFF)
                # inv slot: the lowest member (its route is in no expert segment).
                low = fx.Int32(topk - 1)
                for slot in range_constexpr(topk - 1, -1, -1):
                    hit = ((members >> fx.Int32(slot)) & fx.Int32(1)) != fx.Int32(0)
                    low = hit.select(fx.Int32(slot), low)
                inv[token * fx.Int32(topk) + low] = out_row * fx.Int32(2) + own_bit
                for slot in range_constexpr(topk):
                    if ((members >> fx.Int32(slot)) & fx.Int32(1)) != fx.Int32(0):
                        route_i = token * fx.Int32(topk) + fx.Int32(slot)
                        member_ge = fx.Int32(idx_p[route_i])
                        hdr_row = fx.Int32(group_base[member_ge]) + row
                        srcmap_remote[hdr_row] = source_key | (
                            fx.Int32(slot) << fx.Int32(24)
                        )
                        weight_remote[hdr_row] = fx.Float32(wts_p[route_i]).bitcast(
                            fx.Int32
                        )
            else:
                slot = wk % fx.Int32(topk)
                inv[wk] = out_row * fx.Int32(2) + own_bit
                srcmap_remote[dst_base + row] = source_key | (slot << fx.Int32(24))
                weight_remote[dst_base + row] = fx.Float32(wts_p[wk]).bitcast(fx.Int32)
        comm_ops.fence_system_release()

    @flyc.jit
    def launch(
        addr_disp: fx.Int64,
        wts: fx.Int64,
        ids: fx.Int64,
        parity_p: fx.Int64,
        inv_p: fx.Int64,
        stream: fx.Stream,
    ):
        route(addr_disp, wts, ids, parity_p, inv_p).launch(
            grid=(SEGS, 1, 1), block=(ROUTE_THREADS, 1, 1), stream=stream
        )

    return launch


def build_quant_pack(*, K, topk, rank, tokens):
    """One block per token (``tokens`` is the grid): the token's ``inv`` entries are read once,
    up front, and many tokens' loads are in flight per CU."""
    G = K // 32  # 1x32 groups per row
    assert G <= QP_THREADS and G % 4 == 0 and topk <= 64
    SC_I32 = G // 4

    @fx.struct
    class QpShared:
        sc: fx.Array[Int8, QP_THREADS * 4, 16]
        dst: fx.Array[Int8, 64 * 4, 16]
        row: fx.Array[Int8, K, 16]

    @flyc.kernel(
        name=f"sdma_s1_qpack1_k{K}_t{topk}_r{rank}_n{tokens}",
        known_block_size=[QP_THREADS, 1, 1],
    )
    def qpack(
        x_bf16: fx.Int64,
        inv_p: fx.Int64,
        slab_x: fx.Int64,
        slab_sc: fx.Int64,
        land_x: fx.Int64,
        land_sc: fx.Int64,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        t = fx.Int32(fx.block_idx.x)
        x_in = ptr_buf_tensor(x_bf16, fx.Int32, unit_elems=4)
        slab_xb = ptr_buf_tensor(slab_x, fx.Int32, unit_elems=4)
        land_xb = ptr_buf_tensor(land_x, fx.Int32, unit_elems=4)
        slab_scb = ptr_buf_tensor(slab_sc, fx.Int32)
        land_scb = ptr_buf_tensor(land_sc, fx.Int32)
        inv = ptr_buf_tensor(inv_p, fx.Int32)
        lds = fx.SharedAllocator().allocate(QpShared).peek()
        sc_word = lds_typed_ptr(fx.Int32(fx.ptrtoint(lds.sc.ptr)), T.i32, align=4)
        dst_word = lds_typed_ptr(fx.Int32(fx.ptrtoint(lds.dst.ptr)), T.i32, align=4)
        # The token's destinations, then reset for the next call (only this block reads them).
        if tid < fx.Int32(topk):
            dst_word[tid] = fx.Int32(inv[t * fx.Int32(topk) + tid])
            inv[t * fx.Int32(topk) + tid] = fx.Int32(-1)
        g = tid
        active = g < fx.Int32(G)
        gs = active.select(g, fx.Int32(0))
        # aiter's emit_per_1x32_mx_fp8_group, for group g of token t.
        act = []
        local_max = fx.Float32(1e-10)
        for chunk in range_constexpr(4):
            raw = _load_i32x4(
                x_in, t * fx.Int32(K // 8) + gs * fx.Int32(4) + fx.Int32(chunk)
            )
            values = fx.Vector(raw).bitcast(fx.BFloat16).to(fx.Float32)
            local_max = local_max.maximumf(fmath.absf(values).reduce(ReductionOp.MAX))
            for elem in range_constexpr(8):
                act.append(values[elem])
        working = (
            local_max * fx.Int32(_FP8_E4M3_INV_MAX_POS_BITS).bitcast(fx.Float32)
        ).bitcast(fx.Int32)
        mantissa = working & fx.Int32(0x7FFFFF)
        biased_exp = (working >> fx.Int32(23)) & fx.Int32(0xFF)
        e8m0 = (mantissa != fx.Int32(0)).select(biased_exp + fx.Int32(1), biased_exp)
        e8m0 = (e8m0 > fx.Int32(255)).select(fx.Int32(255), e8m0)
        quant_scale = ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(fx.Float32)
        scaled = [act[k] * quant_scale for k in range_constexpr(32)]
        halves = []
        for half in range_constexpr(2):
            words = []
            for word in range_constexpr(4):
                base = (half * 4 + word) * 4
                packed = rocdl.cvt_pk_fp8_f32(
                    T.i32, scaled[base], scaled[base + 1], fx.Int32(0), 0
                )
                packed = rocdl.cvt_pk_fp8_f32(
                    T.i32, scaled[base + 2], scaled[base + 3], packed, 1
                )
                words.append(packed)
            halves.append(fx.Vector.from_elements(words, fx.Int32))
        sc_word[g] = e8m0
        # The fp8 row into LDS, so each store below is 16 contiguous bytes per lane (the
        # window is uncached: strided partial lines write slowly).
        row_lds = fx.Int32(fx.ptrtoint(lds.row.ptr))
        if active:
            row_w = lds_typed_ptr(row_lds, T.i32, align=4)
            for half in range_constexpr(2):
                for w in range_constexpr(4):
                    row_w[g * fx.Int32(8) + fx.Int32(half * 4 + w)] = fx.Int32(
                        halves[half][w]
                    )
        fx.barrier()
        UNITS = K // 16
        ROUNDS = (UNITS + QP_THREADS - 1) // QP_THREADS
        units = []
        for r in range_constexpr(ROUNDS):
            u = tid + fx.Int32(r * QP_THREADS)
            u_ok = u < fx.Int32(UNITS)
            us = u_ok.select(u, fx.Int32(0))
            units.append(
                (
                    u_ok,
                    us,
                    fx.Vector(
                        lds_vec_load(
                            row_lds,
                            us * fx.Int32(16),
                            fx.Vector.make_type(4, fx.Int32),
                            fx.Int32,
                            align=16,
                        )
                    ),
                )
            )
        # Four groups' e8m0 bytes make one scale dword (little endian).
        q_ok = tid < fx.Int32(SC_I32)
        qs = q_ok.select(tid, fx.Int32(0)) * fx.Int32(4)
        sc4 = (
            fx.Int32(sc_word[qs])
            | (fx.Int32(sc_word[qs + fx.Int32(1)]) << fx.Int32(8))
            | (fx.Int32(sc_word[qs + fx.Int32(2)]) << fx.Int32(16))
            | (fx.Int32(sc_word[qs + fx.Int32(3)]) << fx.Int32(24))
        )
        for j in range_constexpr(topk):
            d = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(dst_word[fx.Int32(j)])))
            if d >= fx.Int32(0):
                row = d >> fx.Int32(1)
                x_off = row * fx.Int32(K // 16)
                sc_off = row * fx.Int32(SC_I32) + tid
                if (d & fx.Int32(1)) != fx.Int32(0):
                    for r in range_constexpr(ROUNDS):
                        if units[r][0]:
                            _store_i32x4(land_xb, x_off + units[r][1], units[r][2])
                    if q_ok:
                        land_scb[sc_off] = sc4
                else:
                    for r in range_constexpr(ROUNDS):
                        if units[r][0]:
                            _store_i32x4(slab_xb, x_off + units[r][1], units[r][2])
                    if q_ok:
                        slab_scb[sc_off] = sc4

    @flyc.jit
    def launch(
        x_bf16: fx.Int64,
        inv_p: fx.Int64,
        slab_x: fx.Int64,
        slab_sc: fx.Int64,
        land_x: fx.Int64,
        land_sc: fx.Int64,
        stream: fx.Stream,
    ):
        qpack(x_bf16, inv_p, slab_x, slab_sc, land_x, land_sc).launch(
            grid=(tokens, 1, 1), block=(QP_THREADS, 1, 1), stream=stream
        )

    return launch
