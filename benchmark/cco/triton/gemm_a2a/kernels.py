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

import triton
import triton.language as tl

from mori.ir.triton import cco


@triton.jit
def gemm_to_staging_kernel(
    a,
    b,
    staging_addr,
    tail,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    shard_n: tl.constexpr,
    scatter_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    LOOP_UNROLL: tl.constexpr,
    TILE_ORDER: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_m = M // BLOCK_M
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if TILE_ORDER == 2:
        pid_m = tile_id % num_pid_m
        n_sequence = tile_id // num_pid_m
        scatter_tiles = scatter_n // BLOCK_N
        tiles_per_peer = shard_n // BLOCK_N
        num_peers = scatter_n // shard_n
        if n_sequence < scatter_tiles:
            inner = n_sequence // num_peers
            peer = n_sequence - inner * num_peers
            pid_n = peer * tiles_per_peer + inner
        else:
            pid_n = n_sequence
    else:
        group_width = 1 if TILE_ORDER == 0 else GROUP_M
        num_pid_in_group = group_width * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * group_width
        group_size_m = tl.minimum(num_pid_m - first_pid_m, group_width)
        pid_in_group = tile_id % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(
        0,
        K,
        BLOCK_K,
        num_stages=PIPELINE_STAGES,
        loop_unroll_factor=LOOP_UNROLL,
    ):
        a_ptrs = a + offs_m[:, None] * K + (k_start + offs_k[None, :])
        b_ptrs = b + offs_n[None, :] * K + (k_start + offs_k[:, None])
        a_tile = tl.load(a_ptrs)
        b_tile = tl.load(b_ptrs)
        accumulator += tl.dot(a_tile, b_tile)

    output = accumulator.to(tl.bfloat16)
    rows = offs_m[:, None]
    if pid_n < scatter_n // BLOCK_N:
        tiles_per_peer = shard_n // BLOCK_N
        dst_rank = pid_n // tiles_per_peer
        tile_in_peer = pid_n - dst_rank * tiles_per_peer
        local_cols = tile_in_peer * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        staging = staging_addr.to(tl.pointer_type(tl.bfloat16), bitcast=True)
        staging_offsets = (
            dst_rank * (M * shard_n) + rows * shard_n + local_cols
        )
        tl.store(staging + staging_offsets, output)
    else:
        cols = offs_n[None, :]
        tail_offsets = rows * N + cols
        tl.store(tail + tail_offsets, output)


@triton.jit
def lsa_a2a_copy_kernel(
    dev_comm,
    staging_addr,
    recv_win,
    M,
    shard_n,
    world_size: tl.constexpr,
    blocks_per_dst: tl.constexpr,
    COPY_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    dst_rank = pid % world_size
    block_in_dst = pid // world_size
    my_rank = cco.devcomm_lsa_rank(dev_comm)

    slab_bytes = M * shard_n * 2
    slab_words = slab_bytes // 8
    words_per_block = tl.cdiv(slab_words, blocks_per_dst)
    word_start = block_in_dst * words_per_block
    word_end = tl.minimum(word_start + words_per_block, slab_words)

    src_addr = staging_addr + dst_rank * slab_bytes + word_start * 8
    dst_addr = cco.lsa_ptr(
        recv_win,
        dst_rank,
        my_rank * slab_bytes + word_start * 8,
    )
    src = src_addr.to(tl.pointer_type(tl.uint64), bitcast=True)
    dst = dst_addr.to(tl.pointer_type(tl.uint64), bitcast=True)
    lane_offsets = tl.arange(0, COPY_BLOCK)
    words = word_end - word_start

    for start in tl.range(0, words, COPY_BLOCK, loop_unroll_factor=1):
        offsets = start + lane_offsets
        mask = offsets < words
        values = tl.load(src + offsets, mask=mask)
        tl.store(dst + offsets, values, mask=mask)

    tl.debug_barrier()
    cco.system_fence(0)


__all__ = ["gemm_to_staging_kernel", "lsa_a2a_copy_kernel"]
