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
"""The internode op's symmetric arena: 16 named regions and their sizes, 17 with
the scale transport on (``out_scales`` is appended only when there are scales).

Every size here is transcribed from what ``EpDispatchCombineHandle`` allocates on
the CCO path (``src/ops/dispatch_combine/dispatch_combine.cpp``,
``InitializeShmemBuf`` / ``InitializeTokenNumSignalBuf`` / ``InitializeOrderMapBuf``
/ ``InitializeBarrier``), because that is what the kernel was written against and
what it is still validated against. Transcribed, not re-derived: a formula that
merely resembles the original is the one bug class this file cannot afford.

Two things about the v1 allocator are worth carrying over as intent rather than
as code:

* It already carves every buffer out of ONE registered CCO window with 256 B
  alignment -- its own comment says it mirrors FlyDSL's SymmArena. So this is not
  a new memory model, it is the same one with names.
* It sizes that window as ``6 * MaxNumTokensToRecv * hiddenDim * maxTokenTypeSize
  + 256 MB``. The 256 MB is slack, and slack is what has been absorbing any
  discrepancy between these formulas and what the kernel actually indexes. Sizing
  the arena exactly, as SymmArena does, removes that cushion -- which is the point
  (an overrun should fail loudly) but also means a wrong formula here shows up as
  corruption rather than as nothing at all.

`interNodeChunkFlagCombine` is deliberately NOT here. Despite the name it is not
the local half of ``inter_node_chunk_flag``: the symmetric one is written by
dispatch and read+cleared by combine (combine enumerates its work from dispatch's
leftover flags), while that one counts completions and is plain device memory.
They have different types and different lifetimes; pairing them by name gives
combine an all-zero work list and a hang.
"""

from __future__ import annotations

_I32 = 4
_U64 = 8
_F32 = 4


def internode_regions(cfg):
    """``[(name, nbytes)]`` for :class:`SymmArena`, in v1's allocation order.

    Order matters only for reproducing v1's offsets when comparing the two side
    by side; the kernel addresses everything by name.

    ``cfg`` needs: world_size, gpu_per_node, hidden_dim, max_num_inp_token_per_rank,
    num_experts_per_rank, num_experts_per_token, max_token_type_size, scale_dim,
    scale_type_size, num_qp_per_pe, max_total_recv_tokens.
    """
    # Local names are the snake_case of the EpInterNodeDeviceCfg helper each one
    # transcribes (ep_internode_args.hpp), so the two can be read side by side.
    world_size = cfg.world_size
    gpu_per_node = cfg.gpu_per_node
    if gpu_per_node <= 0 or world_size % gpu_per_node:
        raise ValueError(
            f"world_size {world_size} must be a multiple of "
            f"gpu_per_node {gpu_per_node}"
        )
    n_nodes = world_size // gpu_per_node

    max_tokens_to_send_per_rank = cfg.max_num_inp_token_per_rank
    topk = cfg.num_experts_per_token
    max_token_type_size = cfg.max_token_type_size
    hidden_dim = cfg.hidden_dim

    # MaxNumTokensToRecvPerRank clamps by max_total_recv_tokens when set.
    if cfg.max_total_recv_tokens > 0:
        max_tokens_to_recv_per_rank = min(
            (cfg.max_total_recv_tokens + world_size - 1) // world_size,
            max_tokens_to_send_per_rank,
        )
    else:
        max_tokens_to_recv_per_rank = max_tokens_to_send_per_rank
    max_tokens_to_recv = world_size * max_tokens_to_recv_per_rank

    # XferBytesPerToken(maxTokenTypeSize): hidden + index + weight + srcTokenId + scale.
    scale_bytes = cfg.scale_dim * cfg.scale_type_size
    xfer_bytes = (
        hidden_dim * max_token_type_size
        + topk * _I32
        + topk * _F32
        + _I32
        + scale_bytes
    )

    # maxNumOutToken, the stride of the order maps.
    max_out_token = world_size * max_tokens_to_send_per_rank * cfg.num_experts_per_rank
    barrier_bytes = world_size * _I32

    regions = [
        # --- ShmemBufsInterNodeV1, in InitializeShmemBuf's order --------------
        ("inter_dispatch_inp", n_nodes * max_tokens_to_send_per_rank * xfer_bytes),
        ("inter_combine_inp", max_tokens_to_recv * xfer_bytes),  # v1's maxStagingSize
        ("inter_staging", 2 * n_nodes * max_tokens_to_send_per_rank * xfer_bytes),
        (
            "inter_dispatch_out",
            max_tokens_to_recv * hidden_dim * max_token_type_size,
        ),
        (
            "inter_combine_out",
            max_tokens_to_send_per_rank * hidden_dim * max_token_type_size,
        ),
        ("inter_dispatch_staging", max_tokens_to_send_per_rank * xfer_bytes),
        # --- weights ---------------------------------------------------------
        ("inp_weights", max_tokens_to_recv * topk * _F32),
        ("dispatch_out_weights", max_tokens_to_recv * topk * _F32),
        # Sized by the SEND count, not the recv one: EpCombineAll indexes this by
        # the LOCAL token id, the same index that sizes inter_combine_out.
        # max_tokens_to_recv drops below the send count once max_total_recv_tokens
        # clamps.
        ("combine_out_weights", max_tokens_to_send_per_rank * topk * _F32),
        # --- indices ---------------------------------------------------------
        ("out_indices", max_tokens_to_recv * topk * _I32),
        # --- token-count signals ---------------------------------------------
        ("recv_token_num", world_size * _I32 * 2 * cfg.num_qp_per_pe),
        ("node_recv_token_num", n_nodes * _U64),
        # --- order maps ------------------------------------------------------
        ("disp_tok_offset", _I32),
        ("disp_tok_id_to_src_tok_id", max_out_token * _I32),
        # --- barriers and flags ----------------------------------------------
        ("cross_device_barrier", barrier_bytes * 2 * _U64),
        ("inter_node_chunk_flag", n_nodes * max_tokens_to_send_per_rank * _U64),
    ]

    # Scales are absent, not zero-sized, when the transport is off: the kernel
    # gates on the region being unbound exactly as v1 gated on an invalid
    # SymmMemObjPtr.
    if scale_bytes:
        regions.append(("out_scales", max_tokens_to_recv * scale_bytes))

    return regions
