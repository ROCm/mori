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
"""Sender side of the fused SDMA Stage1: the production compact dispatch's row loop, local.

One block per dispatch segment (``npes * epr`` expert segments, then one fanout-group
segment per destination), one wave per row, exactly the production producer's rows:
``wk = PAIR_ORDER[PAIR_BASE[seg] + row]`` and destination row ``base + row`` with
``base = TASK_ROW_BASE[ge]`` (or ``GROUP_TASK_BASE`` of the pair's canonical expert).

  * the fp8 row (K bytes) and its e8m0 (K/32 bytes) go to the local send slab at
    ``PAIR_BASE[seg] + row`` -- rows for the same (destination, segment) are therefore
    contiguous, one SDMA put each -- or, for this rank's own rows, straight to the local
    landing buffers at the destination row;
  * the route header (srcmap word, routing weight) goes to the destination's
    ``srcmap_em`` / ``wts_em`` over P2P, as the production producer writes it (for a group
    row, once per pair member at that member's group section).

Launched on the Stage1 stream right before the fused kernel (``gemm1_8wave_sdma``), whose
block 0 then ships the slab; that kernel's final ``quiet`` makes sure every put has read the
slab before the next call's pack rewrites it. The pointers are the cco window's slab and
landing areas (``MegaMoEStage1Sdma``).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels.mega_moe.dispatch import DispatchSlot, _load_fanout_pair
from aiter.ops.flydsl.kernels.tensor_shim import ptr_buf_tensor

WAVES = 4
PACK_B = 4  # rows per step of a wave


def _u64(x):
    return fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(x).ir_value()))


def build_pack(*, K, npes, epr, topk, mtpr, rank):
    TE = npes * epr
    SEGS = TE + npes
    UNITS = K // 16
    PER = (UNITS + 63) // 64
    SC_I32 = K // 128  # e8m0 dwords per row
    assert SC_I32 <= 64
    assert UNITS % 64 == 0, "the row copy moves 16 bytes per lane, whole waves"

    @flyc.kernel(
        name=f"sdma_s1_pack_k{K}_p{npes}_e{epr}_k{topk}_m{mtpr}_r{rank}",
        known_block_size=[64 * WAVES, 1, 1],
    )
    def pack(
        addr_disp: fx.Int64,
        x_q: fx.Int64,
        x_sc: fx.Int64,
        wts: fx.Int64,
        ids: fx.Int64,
        parity_p: fx.Int64,
        slab_x: fx.Int64,
        slab_sc: fx.Int64,
        land_x: fx.Int64,
        land_sc: fx.Int64,
    ):
        seg = fx.Int32(fx.block_idx.x)
        lane = fx.thread_idx.x % fx.Int32(64)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, fx.thread_idx.x // fx.Int32(64)))
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
        rs_x = create_buffer_resource_from_addr(_u64(x_q))
        rs_sc = create_buffer_resource_from_addr(_u64(x_sc))
        out_x = create_buffer_resource_from_addr(
            _u64(own.select(fx.Int64(land_x), fx.Int64(slab_x)))
        )
        out_sc = create_buffer_resource_from_addr(
            _u64(own.select(fx.Int64(land_sc), fx.Int64(slab_sc)))
        )
        wts_p = ptr_buf_tensor(wts, fx.Float32)
        idx_p = ptr_buf_tensor(ids, fx.Int32)  # top-k global expert ids [tokens, topk]
        srcmap_remote = ptr_buf_tensor(srcmap_table[dest], fx.Int32)
        weight_remote = ptr_buf_tensor(weight_table[dest], fx.Int32)
        # This wave's rows (wave + WAVES * i), 64 at a time: the route headers lane-parallel
        # (lane i = row i), then PACK_B rows per step with every load of the step before its
        # stores -- vmcnt retires loads and stores in order, so one store-ack wait (and the
        # headers' remote acks) covers PACK_B rows instead of one. Steps past the last row
        # repeat it (same bytes, same place).
        n_wave = (count - wave + fx.Int32(WAVES - 1)) // fx.Int32(WAVES)
        for g in range(fx.Int32(0), n_wave, fx.Int32(64)):
            n_g = n_wave - g
            n_g = (n_g < fx.Int32(64)).select(n_g, fx.Int32(64))
            valid = lane < n_g
            row_l = wave + (g + valid.select(lane, fx.Int32(0))) * fx.Int32(WAVES)
            wk_l = fx.Int32(pair_order[src_base + row_l])
            token_l = group.select(wk_l & fx.Int32(0xFFFFFF), wk_l // fx.Int32(topk))
            # Route headers at the destination, as the production producer writes them.
            if valid:
                source_key = fx.Int32(rank * mtpr) + token_l
                if group:
                    members = (wk_l >> fx.Int32(24)) & fx.Int32(0xFF)
                    for slot in range_constexpr(topk):
                        if ((members >> fx.Int32(slot)) & fx.Int32(1)) != fx.Int32(0):
                            route = token_l * fx.Int32(topk) + fx.Int32(slot)
                            member_ge = fx.Int32(idx_p[route])
                            hdr_row = fx.Int32(group_base[member_ge]) + row_l
                            srcmap_remote[hdr_row] = source_key | (
                                fx.Int32(slot) << fx.Int32(24)
                            )
                            weight_remote[hdr_row] = fx.Float32(wts_p[route]).bitcast(
                                fx.Int32
                            )
                else:
                    slot = wk_l % fx.Int32(topk)
                    srcmap_remote[dst_base + row_l] = source_key | (
                        slot << fx.Int32(24)
                    )
                    weight_remote[dst_base + row_l] = fx.Float32(wts_p[wk_l]).bitcast(
                        fx.Int32
                    )
            for b0 in range(fx.Int32(0), n_g, fx.Int32(PACK_B)):
                toks = []
                out_rows = []
                for j in range_constexpr(PACK_B):
                    idx = b0 + fx.Int32(j)
                    idx = (idx < n_g).select(idx, n_g - fx.Int32(1))
                    toks.append(fx.Int32(rocdl.readlane(T.i32, token_l, idx)))
                    row = wave + (g + idx) * fx.Int32(WAVES)
                    out_rows.append(own.select(dst_base + row, src_base + row))
                vss = []
                scs = []
                for j in range_constexpr(PACK_B):
                    vss.append(
                        [
                            buffer_load(
                                rs_x,
                                toks[j] * fx.Int32(K // 4)
                                + (lane + fx.Int32(64 * i)) * fx.Int32(4),
                                vec_width=4,
                                dtype=T.i32,
                            )
                            for i in range_constexpr(PER)
                        ]
                    )
                    scs.append(
                        fx.Int32(
                            buffer_load(
                                rs_sc,
                                toks[j] * fx.Int32(SC_I32)
                                + (lane < fx.Int32(SC_I32)).select(
                                    lane, fx.Int32(SC_I32 - 1)
                                ),
                                vec_width=1,
                                dtype=T.i32,
                            )
                        )
                    )
                for j in range_constexpr(PACK_B):
                    for i in range_constexpr(PER):
                        buffer_store(
                            vss[j][i],
                            out_x,
                            out_rows[j] * fx.Int32(K // 4)
                            + (lane + fx.Int32(64 * i)) * fx.Int32(4),
                        )
                    # Lanes past the e8m0 row repeat its last dword.
                    buffer_store(
                        scs[j],
                        out_sc,
                        out_rows[j] * fx.Int32(SC_I32)
                        + (lane < fx.Int32(SC_I32)).select(lane, fx.Int32(SC_I32 - 1)),
                    )
        comm_ops.fence_system_release()

    @flyc.jit
    def launch(
        addr_disp: fx.Int64,
        x_q: fx.Int64,
        x_sc: fx.Int64,
        wts: fx.Int64,
        ids: fx.Int64,
        parity_p: fx.Int64,
        slab_x: fx.Int64,
        slab_sc: fx.Int64,
        land_x: fx.Int64,
        land_sc: fx.Int64,
        stream: fx.Stream,
    ):
        pack(
            addr_disp, x_q, x_sc, wts, ids, parity_p, slab_x, slab_sc, land_x, land_sc
        ).launch(grid=(SEGS, 1, 1), block=(64 * WAVES, 1, 1), stream=stream)

    return launch
