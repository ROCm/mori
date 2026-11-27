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
"""The in-kernel dispatch of the persistent kernel: the owner's plan (count matrix exchange,
sorted row layout, per-source base rows), the producers' grouping of routes and the P2P copy
of the token rows, and the dispatch table's slot numbers.

Copied from ROCm/aiter#6881 (``mega_moe/mega_moe_large_m.py``); each definition carries a
``# from`` comment naming its source module.
"""

from enum import IntEnum
from flydsl.expr import const_expr
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
import flydsl.compiler as flyc
import flydsl.expr as fx
import flyutil as comm_ops

from flyutil import (  # noqa: F401
    AddrBuffer,
    _wave_inclusive_scan_i32,
    _wave_reduce_max_i32,
    load_at,
    store_at,
)


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m
class DispatchSlot(IntEnum):
    PAIR_BASE = 0
    P2P_TOKEN = 1
    P2P_SCALE = 2
    P2P_WEIGHT = 3
    P2P_SRCMAP = 4
    SORTED_EXPERT = 5
    TILE_ROW_BASE = 6
    NUM_VALID = 7
    SRCMAP = 8
    LOCAL_HIST = 9
    COUNT_MATRIX = 10
    P2P_COUNT_MATRIX = 11
    COUNT_DONE = 12
    P2P_COUNT_DONE = 13
    TASK_ROW_BASE = 14
    LOCAL_CURSOR = 15
    P2P_PAYLOAD_READY = 16
    PAIR_ORDER = 17
    P2P_TASK_ROW_BASE = 18
    P2P_PLAN_READY = 19
    PLAN_READY = 20
    PAIR_READY = 21
    ENTRY_COUNT = 22
    EPOCH_GATE = 23
    PAIR_ORDER_READY = 24
    WORK_HEAD = 25
    WORK_TAIL = 26
    EXPERT_TILE_END = 27
    GROUP_DONE = 28
    LAUNCH_READY = 29
    P2P_LAUNCH_READY = 30
    MAX_EXPERT_TILES = 31
    PAYLOAD_CHUNK_DONE = 32
    TILE_READY = 33
    P2P_TILE_READY = 34
    TILE_EXPECTED = 35
    ACTIVE_PAYLOAD_BLOCKS = 36
    PAYLOAD_READY_ROWS = 37
    P2P_PAYLOAD_READY_ROWS = 38
    PAYLOAD_BLOCKS_PER_DESTINATION = 39
    PAYLOAD_CHUNKS_PER_DESTINATION = 40
    P2P_TILE_EXPECTED = 41
    READY_TILE_QUEUE = 42
    P2P_READY_TILE_QUEUE = 43
    READY_TILE_EPOCH = 44
    P2P_READY_TILE_EPOCH = 45
    READY_TILE_TAIL = 46
    P2P_READY_TILE_TAIL = 47
    TILE_EXPECTED_SHARED = 48


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m
DISPATCH_TABLE_SIZE = max(DispatchSlot) + 1


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m
@flyc.jit
def _configure_payload_geometry(
    addr_local_hist,
    addr_chunk_counts,
    addr_block_counts,
    addr_active_blocks,
    lane,
    *,
    fz_npes,
    fz_epr,
    fz_rank,
    payload_chunk_rows,
    dispatch_blocks,
):
    local_hist = AddrBuffer(addr_local_hist)
    chunk_counts = AddrBuffer(addr_chunk_counts)
    block_counts = AddrBuffer(addr_block_counts)
    active_payload_blocks = fx.Int32(0)
    max_blocks = fx.Int32(dispatch_blocks // fz_npes)
    max_source_counts = []
    destination_rows = []
    total_rows = fx.Int32(0)
    for destination in range_constexpr(fz_npes):
        max_source_count = fx.Int32(0)
        rows = fx.Int32(0)
        for local_expert in range(lane, fz_epr, 64):
            ge = fx.Int32(destination * fz_epr) + local_expert
            source_count = load_at(local_hist, ge, fx.Int32, 1)
            max_source_count = (source_count > max_source_count).select(
                source_count, max_source_count
            )
            rows = rows + source_count
        max_source_counts.append(_wave_reduce_max_i32(max_source_count, lane))
        rows = fx.Int32(
            fx.rocdl.readlane(T.i32, _wave_inclusive_scan_i32(rows, lane), 63)
        )
        destination_rows.append(rows)
        total_rows = total_rows + rows
    for destination in range_constexpr(fz_npes):
        if lane == fx.Int32(0):
            chunks = (
                max_source_counts[destination] + fx.Int32(payload_chunk_rows - 1)
            ) // fx.Int32(payload_chunk_rows)
            chunks = (chunks > fx.Int32(0)).select(chunks, fx.Int32(1))
            chunks = (chunks > fx.Int32(4)).select(chunks, fx.Int32(4))
            store_at(chunks, chunk_counts, fx.Int32(destination))
            active_blocks = chunks
            if const_expr(destination != fz_rank):
                # A remote destination with at least twice the average rows (skewed routing) gets
                # proportionally more producers, so its link is not limited by four CTAs.
                scaled_rows = destination_rows[destination] * fx.Int32(fz_npes)
                skewed = scaled_rows >= total_rows * fx.Int32(2)
                boost = (scaled_rows * fx.Int32(4) + total_rows - fx.Int32(1)) // (
                    (total_rows > fx.Int32(0)).select(total_rows, fx.Int32(1))
                )
                active_blocks = (skewed & (boost > active_blocks)).select(
                    boost, active_blocks
                )
            active_blocks = (active_blocks < max_blocks).select(
                active_blocks, max_blocks
            )
            store_at(active_blocks, block_counts, fx.Int32(destination))
            active_payload_blocks = active_payload_blocks + active_blocks
    if lane == fx.Int32(0):
        store_at(active_payload_blocks, AddrBuffer(addr_active_blocks), fx.Int32(0))


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m
@flyc.jit
def _store_expert_metadata(
    addr_sorted_expert,
    addr_tile_row_base,
    addr_srcmap,
    ge,
    local_row_base,
    total_count,
    num_tiles,
    padded_rows,
    *,
    fz_tile_m,
    invalid_source,
):
    sorted_expert = AddrBuffer(addr_sorted_expert)
    tile_row_base = AddrBuffer(addr_tile_row_base)
    srcmap = AddrBuffer(addr_srcmap)
    base_tile = local_row_base // fx.Int32(fz_tile_m)
    for tile in range(fx.Int32(0), num_tiles, 1):
        metadata_index = base_tile + tile
        store_at(ge, sorted_expert, metadata_index)
        store_at(
            local_row_base + tile * fx.Int32(fz_tile_m), tile_row_base, metadata_index
        )
    padding = padded_rows - total_count
    for pad in range(fx.Int32(0), padding, 1):
        store_at(fx.Int32(invalid_source), srcmap, local_row_base + total_count + pad)


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m
@flyc.jit
def _increment_i32(rsrc, index):
    value = load_at(rsrc, index, fx.Int32, 1)
    store_at(value + fx.Int32(1), rsrc, index)


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m
@flyc.jit
def emit_dispatch_plan(
    *,
    num_waves,
    fz_npes,
    fz_epr,
    fz_mtpr,
    fz_rank,
    fz_tile_m,
    fz_total_experts,
    addr_disp,
    parity,
    expected,
    dispatch_blocks,
    payload_chunk_rows,
):
    # fmt: on
    """Build a destination-owned compact plan in one producer-only CTA.

    The producer CTAs count the routes into the local histogram (emit_dispatch_group) and group them after the
    plan publishes the pair prefix.
    """
    rdisp = AddrBuffer(addr_disp)

    def dp(i):
        return load_at(rdisp, fx.Int32(i), fx.Int64, 1)

    a_pair_base = dp(DispatchSlot.PAIR_BASE)
    a_se = dp(DispatchSlot.SORTED_EXPERT)
    a_trb = dp(DispatchSlot.TILE_ROW_BASE)
    a_nv = dp(DispatchSlot.NUM_VALID)
    a_sm = dp(DispatchSlot.SRCMAP)
    a_lh = dp(DispatchSlot.LOCAL_HIST)
    a_bc = dp(DispatchSlot.COUNT_MATRIX)
    p_bc = dp(DispatchSlot.P2P_COUNT_MATRIX)
    a_cd = dp(DispatchSlot.COUNT_DONE)
    p_cd = dp(DispatchSlot.P2P_COUNT_DONE)
    a_lc = dp(DispatchSlot.LOCAL_CURSOR)
    p_mb = dp(DispatchSlot.P2P_TASK_ROW_BASE)
    p_plan_ready = dp(DispatchSlot.P2P_PLAN_READY)
    a_pair_ready = dp(DispatchSlot.PAIR_READY)
    a_pair_order_ready = dp(DispatchSlot.PAIR_ORDER_READY)
    a_expert_tile_end = dp(DispatchSlot.EXPERT_TILE_END)
    a_group_done = dp(DispatchSlot.GROUP_DONE)
    a_max_expert_tiles = dp(DispatchSlot.MAX_EXPERT_TILES)
    a_tile_ready = dp(DispatchSlot.TILE_READY)
    a_tile_expected = dp(DispatchSlot.TILE_EXPECTED)
    a_ready_tail = dp(DispatchSlot.READY_TILE_TAIL)
    a_tile_expected_shared = dp(DispatchSlot.TILE_EXPECTED_SHARED)
    a_active_payload_blocks = dp(DispatchSlot.ACTIVE_PAYLOAD_BLOCKS)
    a_payload_blocks_per_destination = dp(DispatchSlot.PAYLOAD_BLOCKS_PER_DESTINATION)
    a_payload_chunks_per_destination = dp(DispatchSlot.PAYLOAD_CHUNKS_PER_DESTINATION)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    block_threads = num_waves * 64

    gtid = tid
    gnt = fx.Int32(block_threads)
    r_lh = AddrBuffer(a_lh)
    r_bc = AddrBuffer(a_bc)
    r_pair_base = AddrBuffer(a_pair_base)
    r_lc = AddrBuffer(a_lc)
    if tid == fx.Int32(0):
        comm_ops.wait_i32_until_equals(a_group_done, fx.Int32(dispatch_blocks))
        comm_ops.fence_agent_acquire()
        store_at(fx.Int32(0), AddrBuffer(a_group_done), fx.Int32(0))
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        comm_ops.fence_agent_release()
    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
    fx.barrier()
    comm_ops.fence_agent_acquire()

    if const_expr(dispatch_blocks > 32):
        if warp == fx.Int32(0):
            _configure_payload_geometry(
                a_lh,
                a_payload_chunks_per_destination,
                a_payload_blocks_per_destination,
                a_active_payload_blocks,
                lane,
                fz_npes=fz_npes,
                fz_epr=fz_epr,
                fz_rank=fz_rank,
                payload_chunk_rows=payload_chunk_rows,
                dispatch_blocks=dispatch_blocks,
            )
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        fx.barrier()
        comm_ops.fence_agent_release()

    # Transpose the source histogram into each destination's count matrix.
    for ge in range(gtid, fz_total_experts, gnt):
        destination = ge // fx.Int32(fz_epr)
        local_expert = ge - destination * fx.Int32(fz_epr)
        remote_bigcnt = load_at(AddrBuffer(p_bc), destination, fx.Int64, 1)
        count = load_at(r_lh, ge, fx.Int32, 1)
        store_at(
            count, AddrBuffer(remote_bigcnt), fx.Int32(fz_rank * fz_epr) + local_expert
        )
    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
    fx.barrier()

    # Warp 0 plans local experts after all source matrices arrive.
    if warp == fx.Int32(0):
        comm_ops.fence_system_release()
        for peer in range(lane, fz_npes, 64):
            remote_done = load_at(AddrBuffer(p_cd), peer, fx.Int64, 1)
            done_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
            comm_ops.store_i32_system(remote_done, done_index, expected)
        for source in range(lane, fz_npes, 64):
            done_index = parity * fx.Int32(fz_npes) + source
            comm_ops.wait_i32_until_equals(
                a_cd + fx.Int64(done_index) * fx.Int64(4), expected
            )
        comm_ops.fence_system_acquire()

        r_nv = AddrBuffer(a_nv)
        row_carry = fx.Int32(0)
        max_expert_tiles = fx.Int32(0)
        for expert_chunk in range_constexpr((fz_epr + 63) // 64):
            local_expert = fx.Int32(expert_chunk * 64) + lane
            valid_expert = local_expert < fx.Int32(fz_epr)
            safe_expert = valid_expert.select(local_expert, fx.Int32(0))
            ge = fx.Int32(fz_rank * fz_epr + local_expert)
            source_counts = []
            total_count = fx.Int32(0)
            for source in range_constexpr(fz_npes):
                source_count = load_at(
                    r_bc, fx.Int32(source * fz_epr) + safe_expert, fx.Int32, 1
                )
                source_count = valid_expert.select(source_count, fx.Int32(0))
                source_counts.append(source_count)
                total_count = total_count + source_count
            num_tiles = (total_count + fx.Int32(fz_tile_m - 1)) // fx.Int32(fz_tile_m)
            chunk_max = _wave_reduce_max_i32(num_tiles, lane)
            max_expert_tiles = (chunk_max > max_expert_tiles).select(
                chunk_max, max_expert_tiles
            )
            padded_rows = num_tiles * fx.Int32(fz_tile_m)
            inclusive_rows = _wave_inclusive_scan_i32(padded_rows, lane)
            local_row_base = row_carry + inclusive_rows - padded_rows

            sender_prefix = fx.Int32(0)
            for source in range_constexpr(fz_npes):
                if valid_expert:
                    remote_my_base = load_at(
                        AddrBuffer(p_mb), fx.Int32(source), fx.Int64, 1
                    )
                    store_at(
                        local_row_base + sender_prefix, AddrBuffer(remote_my_base), ge
                    )
                sender_prefix = sender_prefix + source_counts[source]

            if valid_expert:
                base_tile = local_row_base // fx.Int32(fz_tile_m)
                for tile in range(fx.Int32(0), num_tiles, 1):
                    tile_index = base_tile + tile
                    store_at(fx.Int32(0), AddrBuffer(a_tile_ready), tile_index)
                    store_at(fx.Int32(1), AddrBuffer(a_tile_expected), tile_index)
                sender_prefix = fx.Int32(0)
                for source in range_constexpr(fz_npes):
                    source_count = source_counts[source]
                    source_active = source_count > fx.Int32(0)
                    source_boundary = source_active & (sender_prefix > fx.Int32(0))
                    source_boundary = source_boundary & (
                        sender_prefix % fx.Int32(fz_tile_m) != fx.Int32(0)
                    )
                    if source_boundary:
                        tile_index = base_tile + sender_prefix // fx.Int32(fz_tile_m)
                        _increment_i32(AddrBuffer(a_tile_expected), tile_index)
                    for chunk_offset in range(
                        fx.Int32(payload_chunk_rows), source_count, payload_chunk_rows
                    ):
                        boundary = sender_prefix + chunk_offset
                        boundary_unaligned = boundary % fx.Int32(fz_tile_m) != fx.Int32(
                            0
                        )
                        if boundary_unaligned:
                            tile_index = base_tile + boundary // fx.Int32(fz_tile_m)
                            _increment_i32(AddrBuffer(a_tile_expected), tile_index)
                    sender_prefix = sender_prefix + source_count
                # Peers read the expected counts for the ready queue from the symmetric copy; this rank's GEMM1
                # waits read the local one.
                for tile in range(fx.Int32(0), num_tiles, 1):
                    tile_index = base_tile + tile
                    store_at(
                        load_at(AddrBuffer(a_tile_expected), tile_index, fx.Int32, 1),
                        AddrBuffer(a_tile_expected_shared),
                        tile_index,
                    )
                store_at(
                    (local_row_base + padded_rows) // fx.Int32(fz_tile_m),
                    AddrBuffer(a_expert_tile_end),
                    local_expert,
                )
                _store_expert_metadata(
                    a_se,
                    a_trb,
                    a_sm,
                    ge,
                    local_row_base,
                    total_count,
                    num_tiles,
                    padded_rows,
                    fz_tile_m=fz_tile_m,
                    invalid_source=fz_npes * fz_mtpr,
                )

            last_lane = min(63, fz_epr - expert_chunk * 64 - 1)
            row_carry = row_carry + fx.Int32(
                fx.rocdl.readlane(T.i32, inclusive_rows, last_lane)
            )

        if lane == fx.Int32(0):
            store_at(row_carry, r_nv, fx.Int32(0))
            store_at(max_expert_tiles, AddrBuffer(a_max_expert_tiles), fx.Int32(0))
            # Skewed routing (one expert holds >= 1/8 of the m-tiles): GEMM1 takes m-tiles in payload
            # completion order from the ready queue, so no CTA waits on a late hot-expert tile while
            # others are ready. [tail, enabled]; no source reads them before it sees this plan.
            ready_order = (
                max_expert_tiles * fx.Int32(8 * fz_tile_m) >= row_carry
            ).select(fx.Int32(1), fx.Int32(0))
            store_at(fx.Int32(0), AddrBuffer(a_ready_tail), fx.Int32(0))
            store_at(ready_order, AddrBuffer(a_ready_tail), fx.Int32(1))
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        comm_ops.fence_system_release()
        for source in range(lane, fz_npes, 64):
            remote_ready = load_at(AddrBuffer(p_plan_ready), source, fx.Int64, 1)
            ready_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
            comm_ops.store_i32_system(remote_ready, ready_index, expected)
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
    elif warp == fx.Int32(1):
        # Build the global-expert exclusive prefix cooperatively.
        pairs_per_lane = (fz_total_experts + 63) // 64
        lane_base = lane * fx.Int32(pairs_per_lane)
        lane_total = fx.Int32(0)
        lane_counts = []
        for item in range_constexpr(pairs_per_lane):
            ge = lane_base + fx.Int32(item)
            valid_ge = ge < fx.Int32(fz_total_experts)
            safe_ge = valid_ge.select(ge, fx.Int32(0))
            source_count = load_at(r_lh, safe_ge, fx.Int32, 1)
            source_count = valid_ge.select(source_count, fx.Int32(0))
            lane_counts.append(source_count)
            lane_total = lane_total + source_count
        lane_prefix = _wave_inclusive_scan_i32(lane_total, lane) - lane_total
        source_prefix = lane_prefix
        for item in range_constexpr(pairs_per_lane):
            ge = lane_base + fx.Int32(item)
            valid_ge = ge < fx.Int32(fz_total_experts)
            if valid_ge:
                store_at(source_prefix, r_pair_base, ge)
                store_at(source_prefix, r_lc, ge)
            source_prefix = source_prefix + lane_counts[item]
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        comm_ops.fence_agent_release()
        if lane == fx.Int32(0):
            comm_ops.store_i32_system(a_pair_ready, parity, expected)

    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
    fx.barrier()
    if tid == fx.Int32(0):
        active_group_blocks = fx.Int32(dispatch_blocks)
        if const_expr(dispatch_blocks > 32):
            active_group_blocks = load_at(
                AddrBuffer(a_active_payload_blocks), fx.Int32(0), fx.Int32, 1
            )
        comm_ops.wait_i32_until_equals(a_group_done, active_group_blocks)
        comm_ops.fence_agent_acquire()
        comm_ops.fence_agent_release()
        comm_ops.store_i32_system(a_pair_order_ready, parity, expected)


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m
@flyc.jit
def emit_dispatch_group(
    *,
    num_waves,
    fz_k,
    fz_total_experts,
    addr_disp,
    i32_cur_tok,
    addr_in_idx,
    dispatch_blocks,
    producer_slot,
    parity,
    expected,
):
    # fmt: on
    """Count and group disjoint route spans across payload producer CTAs."""
    rdisp = AddrBuffer(addr_disp)

    def dp(i):
        return load_at(rdisp, fx.Int32(int(i)), fx.Int64, 1)

    a_pair_ready = dp(DispatchSlot.PAIR_READY)
    a_pair_order_ready = dp(DispatchSlot.PAIR_ORDER_READY)
    a_local_hist = dp(DispatchSlot.LOCAL_HIST)
    a_local_cursor = dp(DispatchSlot.LOCAL_CURSOR)
    a_pair_order = dp(DispatchSlot.PAIR_ORDER)
    a_group_done = dp(DispatchSlot.GROUP_DONE)
    a_active_payload_blocks = dp(DispatchSlot.ACTIVE_PAYLOAD_BLOCKS)
    r_idx = AddrBuffer(addr_in_idx)
    r_pair = AddrBuffer(a_pair_order)
    tid = fx.thread_idx.x
    block_threads = fx.Int32(num_waves * 64)
    group_tid = producer_slot * block_threads + tid
    group_threads = fx.Int32(dispatch_blocks) * block_threads
    route_limit = i32_cur_tok * fx.Int32(fz_k)

    for route in range(group_tid, route_limit, group_threads):
        expert = load_at(r_idx, route, fx.Int32, 1)
        valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(fz_total_experts))
        if valid:
            comm_ops.atomic_add_agent(
                a_local_hist + fx.Int64(expert) * fx.Int64(4), fx.Int32(1)
            )
    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
    fx.barrier()
    if tid == fx.Int32(0):
        comm_ops.fence_agent_release()
        comm_ops.atomic_add_agent(a_group_done, fx.Int32(1))
    if tid == fx.Int32(0):
        comm_ops.wait_i32_until_equals(
            a_pair_ready + fx.Int64(parity) * fx.Int64(4), expected
        )
        comm_ops.fence_agent_acquire()
    fx.barrier()

    active_group_blocks = fx.Int32(dispatch_blocks)
    if const_expr(dispatch_blocks > 32):
        active_group_blocks = load_at(
            AddrBuffer(a_active_payload_blocks), fx.Int32(0), fx.Int32, 1
        )
    group_active = producer_slot < active_group_blocks
    if group_active:
        active_group_tid = producer_slot * block_threads + tid
        active_group_threads = active_group_blocks * block_threads
        for route in range(active_group_tid, route_limit, active_group_threads):
            expert = load_at(r_idx, route, fx.Int32, 1)
            valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(fz_total_experts))
            if valid:
                position = fx.Int32(
                    comm_ops.atomic_add_agent(
                        a_local_cursor + fx.Int64(expert) * fx.Int64(4), fx.Int32(1)
                    )
                )
                store_at(route, r_pair, position)
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.fence_agent_release()
            comm_ops.atomic_add_agent(a_group_done, fx.Int32(1))
    # Payload-active producers are chosen per destination, so some of them sit
    # out adaptive grouping; all must still see this launch's pair order.
    if tid == fx.Int32(0):
        comm_ops.wait_i32_until_equals(
            a_pair_order_ready + fx.Int64(parity) * fx.Int64(4), expected
        )
        comm_ops.fence_agent_acquire()
    fx.barrier()


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m
@flyc.jit
def _copy_token_row(source_rsrc, destination_rsrc, lane, *, fz_safe_end_i32, fz_n_i32):
    lane_offset = lane * fx.Int32(4)
    if const_expr(fz_safe_end_i32 > 0):
        for column in range(lane_offset, fz_safe_end_i32, 512):
            value0 = load_at(source_rsrc, column, fx.Int32, 4)
            value1 = load_at(source_rsrc, column + fx.Int32(256), fx.Int32, 4)
            store_at(value0, destination_rsrc, column)
            store_at(value1, destination_rsrc, column + fx.Int32(256))
    if const_expr(fz_safe_end_i32 < fz_n_i32):
        for column in range(lane_offset + fz_safe_end_i32, fz_n_i32, 256):
            value = load_at(source_rsrc, column, fx.Int32, 4)
            store_at(value, destination_rsrc, column)


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m
@flyc.jit
def _publish_tile_range(
    p_tile_ready,
    p_tile_expected,
    p_ready_queue,
    p_ready_epoch,
    p_ready_tail,
    destination,
    destination_base,
    row_begin,
    row_end,
    rows_per_tile,
    payload_epoch,
):
    """Count this chunk into the destination tiles it covers. When the destination's plan enabled ready
    order, the chunk that completes a tile appends the tile to its ready queue (completion order).
    """
    if row_end > row_begin:
        comm_ops.fence_system_release()
        remote_tile_ready = load_at(AddrBuffer(p_tile_ready), destination, fx.Int64, 1)
        remote_tail = load_at(AddrBuffer(p_ready_tail), destination, fx.Int64, 1)
        ready_order = load_at(AddrBuffer(remote_tail), fx.Int32(1), fx.Int32, 1)
        first_tile = (destination_base + row_begin) // rows_per_tile
        last_tile = (destination_base + row_end - fx.Int32(1)) // rows_per_tile
        for tile in range(first_tile, last_tile + fx.Int32(1), 1):
            previous = fx.Int32(
                comm_ops.atomic_add_system(
                    remote_tile_ready + fx.Int64(tile) * fx.Int64(4), fx.Int32(1)
                )
            )
            if ready_order != fx.Int32(0):
                remote_tile_expected = load_at(
                    AddrBuffer(p_tile_expected), destination, fx.Int64, 1
                )
                if previous + fx.Int32(1) == load_at(
                    AddrBuffer(remote_tile_expected), tile, fx.Int32, 1
                ):
                    # The final RMW observes the release sequence of every chunk of the tile.
                    comm_ops.fence_system_acquire()
                    ready_slot = fx.Int32(
                        comm_ops.atomic_add_system(remote_tail, fx.Int32(1))
                    )
                    remote_queue = load_at(
                        AddrBuffer(p_ready_queue), destination, fx.Int64, 1
                    )
                    store_at(tile, AddrBuffer(remote_queue), ready_slot)
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    comm_ops.fence_system_release()
                    remote_epoch = load_at(
                        AddrBuffer(p_ready_epoch), destination, fx.Int64, 1
                    )
                    comm_ops.store_i32_system(remote_epoch, ready_slot, payload_epoch)


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m
@flyc.jit
def emit_dispatch_payload(
    *,
    num_waves,
    fz_epr,
    fz_k,
    fz_mtpr,
    fz_rank,
    fz_total_experts,
    fz_nbytes,
    fz_n_i32,
    fz_safe_end_i32,
    fz_scale_n_i32,
    fz_enable_scales,
    addr_disp,
    addr_in_tok,
    addr_in_wts,
    addr_in_sc,
    dispatch_blocks,
    producer_slot,
    parity,
    expected,
    producers_per_destination,
    chunks_per_destination,
    payload_chunk_rows,
    pack=None,
):
    # fmt: on
    """Produce independently publishable expert payloads from a compact plan.

    Each (expert, chunk of payload_chunk_rows source rows) is one task; a chunk publishes the destination tiles
    it completes, and the last chunk of an expert publishes the expert.
    """
    rdisp = AddrBuffer(addr_disp)

    def dp(i):
        return load_at(rdisp, fx.Int32(i), fx.Int64, 1)

    p_rx = dp(DispatchSlot.P2P_TOKEN)
    p_sc = dp(DispatchSlot.P2P_SCALE)
    p_wts = dp(DispatchSlot.P2P_WEIGHT)
    p_sm = dp(DispatchSlot.P2P_SRCMAP)
    a_pair_base = dp(DispatchSlot.PAIR_BASE)
    a_lh = dp(DispatchSlot.LOCAL_HIST)
    a_mb = dp(DispatchSlot.TASK_ROW_BASE)
    p_payload_ready = dp(DispatchSlot.P2P_PAYLOAD_READY)
    a_pair_order = dp(DispatchSlot.PAIR_ORDER)
    a_plan_ready = dp(DispatchSlot.PLAN_READY)
    a_chunk_done = dp(DispatchSlot.PAYLOAD_CHUNK_DONE)
    p_tile_ready = dp(DispatchSlot.P2P_TILE_READY)
    p_tile_expected = dp(DispatchSlot.P2P_TILE_EXPECTED)
    p_ready_queue = dp(DispatchSlot.P2P_READY_TILE_QUEUE)
    p_ready_epoch = dp(DispatchSlot.P2P_READY_TILE_EPOCH)
    p_ready_tail = dp(DispatchSlot.P2P_READY_TILE_TAIL)
    p_payload_ready_rows = dp(DispatchSlot.P2P_PAYLOAD_READY_ROWS)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    r_pair_base = AddrBuffer(a_pair_base)
    r_lh = AddrBuffer(a_lh)
    r_mb = AddrBuffer(a_mb)
    r_pair = AddrBuffer(a_pair_order)
    r_wts = AddrBuffer(addr_in_wts)
    r_chunk_done = AddrBuffer(a_chunk_done)
    row0 = warp
    row_stride = fx.Int32(num_waves)
    PACK_B = (
        4  # rows per step of the pack path: their loads all go out before their stores
    )

    def _pack_rows(
        row_begin,
        row_end,
        source_base,
        destination_base,
        payload_base,
        token_dst,
        scale_dst,
        wts_dst,
        srcmap_dst,
        cm,
    ):
        """Pack path: this wave's rows (row_begin + warp + 8 i) of the chunk, 64 at a time. The
        route headers go lane-parallel (lane i = row i); then PACK_B rows per step, every load
        of the step before its stores, so one store-ack wait covers PACK_B rows (vmcnt retires
        loads and stores in order). Steps past the last row repeat it (same bytes, same place).
        """
        n_wave = (row_end - row_begin - warp + row_stride - fx.Int32(1)) // row_stride
        scale_off = lane
        if const_expr(fz_enable_scales):
            scale_off = (lane < fx.Int32(fz_scale_n_i32 // 4)).select(
                lane, fx.Int32(fz_scale_n_i32 // 4 - 1)
            ) * fx.Int32(4)
        for g in range(fx.Int32(0), n_wave, fx.Int32(64)):
            n_g = n_wave - g
            n_g = (n_g < fx.Int32(64)).select(n_g, fx.Int32(64))
            valid = lane < n_g
            row_l = (
                row_begin + warp + (g + valid.select(lane, fx.Int32(0))) * row_stride
            )
            wk_l = load_at(r_pair, source_base + row_l, fx.Int32, 1)
            tok_l = wk_l // fx.Int32(fz_k)
            weight_l = load_at(r_wts, wk_l, fx.Float32, 1)
            enc_l = (fx.Int32(fz_rank * fz_mtpr) + tok_l) | (
                (wk_l % fx.Int32(fz_k)) << fx.Int32(24)
            )
            wbits_l = fx.Vector.from_elements([weight_l], fx.Float32).bitcast(fx.Int32)
            wbits_l = wbits_l[0]
            if valid:
                store_at(wbits_l, wts_dst, destination_base + row_l)
                store_at(enc_l, srcmap_dst, destination_base + row_l)
            for b0 in range(fx.Int32(0), n_g, fx.Int32(PACK_B)):
                toks = []
                prows = []
                for j in range_constexpr(PACK_B):
                    idx = b0 + fx.Int32(j)
                    idx = (idx < n_g).select(idx, n_g - fx.Int32(1))
                    toks.append(fx.Int32(fx.rocdl.readlane(T.i32, tok_l, idx)))
                    prows.append(
                        payload_base + row_begin + warp + (g + idx) * row_stride
                    )
                svals = []
                tvals = []
                for j in range_constexpr(PACK_B):
                    if const_expr(fz_enable_scales):
                        svals.append(
                            load_at(
                                AddrBuffer(addr_in_sc),
                                toks[j] * fx.Int32(fz_scale_n_i32) + scale_off,
                                fx.Int32,
                                4,
                            )
                        )
                    src = AddrBuffer(
                        addr_in_tok + fx.Int64(toks[j]) * fx.Int64(fz_nbytes)
                    )
                    tvals.append(
                        [
                            load_at(
                                src, lane * fx.Int32(4) + fx.Int32(c * 256), fx.Int32, 4
                            )
                            for c in range_constexpr(fz_n_i32 // 256)
                        ]
                    )
                for j in range_constexpr(PACK_B):
                    if const_expr(fz_enable_scales):
                        store_at(
                            svals[j],
                            scale_dst,
                            prows[j] * fx.Int32(fz_scale_n_i32) + scale_off,
                            cache_modifier=cm,
                        )
                    dst = AddrBuffer(
                        token_dst + fx.Int64(prows[j]) * fx.Int64(fz_nbytes)
                    )
                    for c in range_constexpr(fz_n_i32 // 256):
                        store_at(
                            tvals[j][c],
                            dst,
                            lane * fx.Int32(4) + fx.Int32(c * 256),
                            cache_modifier=cm,
                        )

    def _publish_task(destination, local_expert, ge):
        comm_ops.fence_system_release()
        ready_remote = load_at(AddrBuffer(p_payload_ready), destination, fx.Int64, 1)
        ready_index = parity * fx.Int32(fz_epr) + local_expert
        comm_ops.atomic_add_system(
            ready_remote + fx.Int64(ready_index) * fx.Int64(4), fx.Int32(1)
        )
        store_at(fx.Int32(0), r_lh, ge)

    def _finish_task(destination, local_expert, ge, num_chunks):
        comm_ops.fence_system_release()
        completed = fx.Int32(
            comm_ops.atomic_add_agent(
                a_chunk_done + fx.Int64(ge) * fx.Int64(4), fx.Int32(1)
            )
        )
        if completed == num_chunks - fx.Int32(1):
            comm_ops.fence_agent_acquire()
            store_at(fx.Int32(0), r_chunk_done, ge)
            _publish_task(destination, local_expert, ge)

    num_destinations = fz_total_experts // fz_epr
    assert payload_chunk_rows > 0 and dispatch_blocks % num_destinations == 0
    # Equal on every rank: all ranks run the same launches.
    payload_epoch = (expected // fx.Int32(num_destinations)) * fx.Int32(2) - parity
    task_limit = fx.Int32(fz_epr) * chunks_per_destination
    task0 = producer_slot // fx.Int32(num_destinations)
    task_stride = fx.Int32(producers_per_destination)
    producer_destination = producer_slot % fx.Int32(num_destinations)
    ready_index = parity * fx.Int32(num_destinations) + producer_destination
    if tid == fx.Int32(0):
        comm_ops.wait_i32_until_equals(
            a_plan_ready + fx.Int64(ready_index) * fx.Int64(4), expected
        )
        comm_ops.fence_system_acquire()
    destination_ready_rows = fx.Int32(0)
    if tid == fx.Int32(0):
        remote_ready_rows = load_at(
            AddrBuffer(p_payload_ready_rows), producer_destination, fx.Int64, 1
        )
        destination_ready_rows = load_at(
            AddrBuffer(remote_ready_rows), fx.Int32(0), fx.Int32, 1
        )
    fx.barrier()
    for task_index in range(task0, task_limit, task_stride):
        chunk_id = task_index // fx.Int32(fz_epr)
        rotated_expert = task_index - chunk_id * fx.Int32(fz_epr)
        rotation = (chunk_id * fx.Int32(17)) % fx.Int32(fz_epr)
        local_expert = (rotated_expert + fx.Int32(fz_epr) - rotation) % fx.Int32(fz_epr)
        destination = producer_destination
        ge = destination * fx.Int32(fz_epr) + local_expert
        source_count_lane = fx.Int32(0)
        source_base_lane = fx.Int32(0)
        destination_base_lane = fx.Int32(0)
        if lane == fx.Int32(0):
            source_count_lane = load_at(r_lh, ge, fx.Int32, 1)
            source_base_lane = load_at(r_pair_base, ge, fx.Int32, 1)
            destination_base_lane = load_at(r_mb, ge, fx.Int32, 1)
        source_count = fx.Int32(fx.rocdl.readfirstlane(T.i32, source_count_lane))
        source_base = fx.Int32(fx.rocdl.readfirstlane(T.i32, source_base_lane))
        destination_base = fx.Int32(
            fx.rocdl.readfirstlane(T.i32, destination_base_lane)
        )
        num_chunks = (source_count + fx.Int32(payload_chunk_rows - 1)) // fx.Int32(
            payload_chunk_rows
        )
        num_chunks = (num_chunks > fx.Int32(0)).select(num_chunks, fx.Int32(1))
        chunk_active = chunk_id < num_chunks
        chunk_begin = chunk_id * fx.Int32(payload_chunk_rows)
        chunk_limit = chunk_begin + fx.Int32(payload_chunk_rows)
        chunk_end = (source_count < chunk_limit).select(source_count, chunk_limit)
        row_begin = chunk_active.select(chunk_begin, fx.Int32(0))
        row_end = chunk_active.select(chunk_end, fx.Int32(0))
        wts_remote_rsrc = AddrBuffer(
            load_at(AddrBuffer(p_wts), destination, fx.Int64, 1)
        )
        srcmap_remote_rsrc = AddrBuffer(
            load_at(AddrBuffer(p_sm), destination, fx.Int64, 1)
        )
        if const_expr(pack is not None):
            # SDMA dispatch: rows for a peer go to this rank's send slab at their pair row, this
            # rank's own rows to its landing buffer at their sorted row, written through (the
            # copy engine reads memory; other XCDs read the landing rows); the route headers
            # still go to the destination over P2P.
            p_slab_x, p_slab_sc, p_land_x, p_land_sc = pack
            own = destination == fx.Int32(fz_rank)
            _pack_rows(
                row_begin,
                row_end,
                source_base,
                destination_base,
                own.select(destination_base, source_base),
                own.select(fx.Int64(p_land_x), fx.Int64(p_slab_x)),
                (
                    AddrBuffer(own.select(fx.Int64(p_land_sc), fx.Int64(p_slab_sc)))
                    if fz_enable_scales
                    else None
                ),
                wts_remote_rsrc,
                srcmap_remote_rsrc,
                1 | 16,
            )
        else:
            token_remote = load_at(AddrBuffer(p_rx), destination, fx.Int64, 1)
            if const_expr(fz_enable_scales):
                scale_remote_rsrc = AddrBuffer(
                    load_at(AddrBuffer(p_sc), destination, fx.Int64, 1)
                )
            for row in range(row_begin + row0, row_end, row_stride):
                wk_lane = fx.Int32(0)
                if lane == fx.Int32(0):
                    wk_lane = load_at(r_pair, source_base + row, fx.Int32, 1)
                wk = fx.Int32(fx.rocdl.readfirstlane(T.i32, wk_lane))
                source_token = wk // fx.Int32(fz_k)
                topk_slot = wk % fx.Int32(fz_k)
                destination_row = destination_base + row

                def _copy_route_header():
                    weight = load_at(r_wts, wk, fx.Float32, 1)
                    source_encoding = (fx.Int32(fz_rank * fz_mtpr) + source_token) | (
                        topk_slot << fx.Int32(24)
                    )
                    weight_bits = fx.Vector.from_elements([weight], fx.Float32).bitcast(
                        fx.Int32
                    )[0]
                    store_at(weight_bits, wts_remote_rsrc, destination_row)
                    store_at(source_encoding, srcmap_remote_rsrc, destination_row)

                if lane == fx.Int32(0):
                    _copy_route_header()

                if const_expr(fz_enable_scales):
                    scale_lane = lane
                    if const_expr(fz_scale_n_i32 % 4 == 0):
                        scale_offset = scale_lane * fx.Int32(4)
                        if scale_offset < fx.Int32(fz_scale_n_i32):
                            scale = load_at(
                                AddrBuffer(addr_in_sc),
                                source_token * fx.Int32(fz_scale_n_i32) + scale_offset,
                                fx.Int32,
                                4,
                            )
                            store_at(
                                scale,
                                scale_remote_rsrc,
                                destination_row * fx.Int32(fz_scale_n_i32)
                                + scale_offset,
                            )
                    elif scale_lane < fx.Int32(fz_scale_n_i32):
                        scale = load_at(
                            AddrBuffer(addr_in_sc),
                            source_token * fx.Int32(fz_scale_n_i32) + scale_lane,
                            fx.Int32,
                            1,
                        )
                        store_at(
                            scale,
                            scale_remote_rsrc,
                            destination_row * fx.Int32(fz_scale_n_i32) + scale_lane,
                        )
                source_rsrc = AddrBuffer(
                    addr_in_tok + fx.Int64(source_token) * fx.Int64(fz_nbytes)
                )
                destination_rsrc = AddrBuffer(
                    token_remote + fx.Int64(destination_row) * fx.Int64(fz_nbytes)
                )
                _copy_token_row(
                    source_rsrc,
                    destination_rsrc,
                    lane,
                    fz_safe_end_i32=fz_safe_end_i32,
                    fz_n_i32=fz_n_i32,
                )

        if chunk_active:
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            fx.barrier()
            if tid == fx.Int32(0):
                if const_expr(pack is not None):
                    # The chunk's rows are in memory (written through, retired): count it
                    # toward its segment for the SDMA put (or, for this rank's own rows,
                    # toward the GEMM1 items of the expert).
                    comm_ops.fence_agent_release()
                    comm_ops.atomic_add_agent(
                        a_chunk_done + fx.Int64(ge) * fx.Int64(4), fx.Int32(1)
                    )
                else:
                    _publish_tile_range(
                        p_tile_ready,
                        p_tile_expected,
                        p_ready_queue,
                        p_ready_epoch,
                        p_ready_tail,
                        destination,
                        destination_base,
                        row_begin,
                        row_end,
                        destination_ready_rows,
                        payload_epoch,
                    )
                    _finish_task(destination, local_expert, ge, num_chunks)
            fx.barrier()


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m
def _wave_uniform_i64(addr):
    # Hoist a wave-uniform 64-bit address into SGPRs so buffer resources avoid a waterfall.
    v = fx.Uint64(addr)
    lo = fx.rocdl.readfirstlane(T.i32, fx.Uint32(v))
    hi = fx.rocdl.readfirstlane(T.i32, fx.Uint32(v >> 32))
    return (fx.Uint64(hi) << 32) | fx.Uint64(lo)
