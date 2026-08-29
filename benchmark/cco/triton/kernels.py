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
"""Triton kernels mirroring ``benchmark/cco/p2p_*.cpp``.

The benchmark currently fixes CCO cooperative scope to ``block``.  This makes
the C++ and Triton launch geometry/completion rules directly comparable and
avoids treating a Triton scalar extern call as a single CCO thread operation.
"""

import triton
import triton.language as tl

from mori.ir.triton import cco

ELEMENT_BYTES = tl.constexpr(8)

OP_PUT = 0
OP_GET = 1
METRIC_BANDWIDTH = 0
METRIC_LATENCY = 1


@triton.jit
def _grid_barrier(counter, iteration, nprograms: tl.constexpr):
    """Sense barrier for the small, fully-resident LSA bandwidth grid."""

    ticket = tl.atomic_add(counter, 1, sem="acq_rel")
    phase = iteration + 1
    if ticket == nprograms - 1:
        tl.store(counter, 0)
        tl.atomic_xchg(counter + 1, phase, sem="release")
    else:
        observed = tl.load(counter + 1, cache_modifier=".cv")
        while observed < phase:
            observed = tl.load(counter + 1, cache_modifier=".cv")


@triton.jit
def lsa_p2p_kernel(
    dev_comm,
    send_win,
    recv_win,
    peer_lsa_rank,
    size_bytes,
    iterations,
    sync_counter,
    op: tl.constexpr,
    metric: tl.constexpr,
    copy_block: tl.constexpr,
    nprograms: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = size_bytes // ELEMENT_BYTES
    elements_per_program = total_elements // nprograms
    base_element = pid * elements_per_program
    byte_offset = base_element * ELEMENT_BYTES
    local_lsa_rank = cco.devcomm_lsa_rank(dev_comm)

    if op == 0:
        src_addr = cco.lsa_ptr(send_win, local_lsa_rank, byte_offset)
        dst_addr = cco.lsa_ptr(recv_win, peer_lsa_rank, byte_offset)
    else:
        src_addr = cco.lsa_ptr(send_win, peer_lsa_rank, byte_offset)
        dst_addr = cco.lsa_ptr(recv_win, local_lsa_rank, byte_offset)

    src = src_addr.to(tl.pointer_type(tl.uint64), bitcast=True)
    dst = dst_addr.to(tl.pointer_type(tl.uint64), bitcast=True)
    lane_offsets = tl.arange(0, copy_block)

    for iteration in tl.range(0, iterations, loop_unroll_factor=1):
        for start in tl.range(
            0, elements_per_program, copy_block, loop_unroll_factor=1
        ):
            offsets = start + lane_offsets
            mask = offsets < elements_per_program
            values = tl.load(src + offsets, mask=mask)
            tl.store(dst + offsets, values, mask=mask)

        tl.debug_barrier()
        if op == 0 or metric == 0:
            cco.system_fence(metric == 1)
        tl.debug_barrier()

        if metric == 0 and nprograms > 1:
            _grid_barrier(sync_counter, iteration, nprograms)


@triton.jit
def sdma_p2p_kernel(
    dev_comm,
    send_win,
    recv_win,
    peer_lsa_rank,
    size_bytes,
    iterations,
    num_queues,
    op: tl.constexpr,
    metric: tl.constexpr,
    aggregate: tl.constexpr,
    agg_depth: tl.constexpr,
    nprograms: tl.constexpr,
):
    pid = tl.program_id(0)
    per_program = (size_bytes // nprograms) & -8
    base = pid * per_program
    bytes_to_copy = per_program
    if pid == nprograms - 1:
        bytes_to_copy = size_bytes - base
    qid = pid % num_queues
    flags = 1 if aggregate else 0

    for _ in tl.range(0, iterations, loop_unroll_factor=1):
        sub_bytes = bytes_to_copy // agg_depth
        for part in tl.static_range(agg_depth):
            part_offset = part * sub_bytes
            part_bytes = sub_bytes
            if part == agg_depth - 1:
                part_bytes = bytes_to_copy - part_offset
            if op == 0:
                cco.sdma_put_block(
                    dev_comm,
                    peer_lsa_rank,
                    recv_win,
                    base + part_offset,
                    send_win,
                    base + part_offset,
                    part_bytes,
                    qid,
                    flags,
                )
            else:
                cco.sdma_get_block(
                    dev_comm,
                    peer_lsa_rank,
                    recv_win,
                    base + part_offset,
                    send_win,
                    base + part_offset,
                    part_bytes,
                    qid,
                    flags,
                )
        if aggregate:
            cco.sdma_commit_block(dev_comm, peer_lsa_rank, qid)
        if metric == 1:
            cco.sdma_quiet_queue(dev_comm, peer_lsa_rank, qid)

    if metric == 0:
        cco.sdma_quiet_queue(dev_comm, peer_lsa_rank, qid)


@triton.jit
def gda_p2p_kernel(
    dev_comm,
    send_win,
    recv_win,
    peer_world_rank,
    size_bytes,
    iterations,
    op: tl.constexpr,
    metric: tl.constexpr,
    nprograms: tl.constexpr,
):
    pid = tl.program_id(0)
    bytes_per_program = size_bytes // nprograms
    offset = pid * bytes_per_program

    for _ in tl.range(0, iterations, loop_unroll_factor=1):
        if op == 0:
            cco.gda_put_ib_none(
                dev_comm,
                pid,
                peer_world_rank,
                recv_win,
                offset,
                send_win,
                offset,
                bytes_per_program,
                0,
                0,
            )
        else:
            cco.gda_get_ib(
                dev_comm,
                pid,
                peer_world_rank,
                send_win,
                offset,
                recv_win,
                offset,
                bytes_per_program,
            )
        if metric == 1:
            cco.gda_flush_warp(dev_comm, pid)

    if metric == 0:
        cco.gda_flush_block(dev_comm, pid)


__all__ = [
    "ELEMENT_BYTES",
    "OP_PUT",
    "OP_GET",
    "METRIC_BANDWIDTH",
    "METRIC_LATENCY",
    "lsa_p2p_kernel",
    "sdma_p2p_kernel",
    "gda_p2p_kernel",
]
