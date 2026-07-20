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
"""
Hierarchical cross-node AllGather.

Within a node use the SDMA copy-engine path (``AllgatherSdma`` over XGMI);
across nodes use the RDMA ring. The final per-rank output matches
``torch.distributed.all_gather_into_tensor`` (rank-major: rank0, rank1, ...
rank(N*G-1)) with zero numerical tolerance.

For ``num_nodes == 1`` the hierarchical operation degenerates to a single
intra-node SDMA AllGather over all ``G`` local ranks. For ``num_nodes >= 2``
the node-blocks are exchanged over the inter-node RDMA ring.

The SDMA AllGather moves bytes in uint32 lanes regardless of the logical
dtype, so as long as each rank's contribution is a multiple of 4 bytes the
byte layout is identical to torch's concatenation -- hence bit-exact for
bf16/fp16/fp32/int32.

Default shipped path (no ``MORI_HIER_*`` env set)
-------------------------------------------------
A caller that constructs ``HierAllGather`` with no env overrides gets the
sliced 2-D path (``slice_inter``) with fused Phase-B (``slice_fused``),
stream-ordered ring/intra barriers (``stream_ring``/``stream_intra``) and
deferred finish fences (``slice_defer_fin``/``slice_defer_inter_fin``), the
serial ``slice_direct`` reassembly gather, and CU-domain copy-out
(``_py_cu_copyout``). At ``ranks_per_node >= 8`` the two cross-PE finish fences
are forced ON for bit-exactness (see ``_apply_dense_node_defaults``). The fused
``ring || local-gather`` kernel (``fuse_local``) is OFF by default (E2E-unstable;
standalone-only). Every ``MORI_HIER_*`` flag is an opt-in A/B lever with an env
override; see ``examples/fsdp_sdma/README.md`` for the shipped-vs-experimental
table. This module docstring and that README are the single source of truth for
the default path.
"""

import os
import socket
import sys
from typing import List, Optional, Sequence

import torch

# HARDWARE SDMA CHANNEL CAP (measured on MI300X): the per-GPU SDMA
# queue-slot count caps MORI_SDMA_NUM_CHANNELS at 8 -- requesting 12 or 16
# CRASHES at SDMA queue creation (anvil.cpp:228 hsaKmtCreateQueueExt "Failed",
# queue-slot exhaustion, NOT an engine-id error). anvil reads this env at shmem
# init, BEFORE any HierAllGather is constructed, so the only place a Python guard
# can prevent the crash is at IMPORT time (this runs on `import mori.ccl...`,
# ahead of shmem.init()). Bit-exact by construction: it only rewrites a value
# that is >8, i.e. one that would otherwise abort the process; every real config
# (E2E giant-AG nq=2, UT standalone_fast nq=8) is left untouched.
MORI_SDMA_CH_HW_MAX = 8

# HIERARCHICAL BARRIER SLOT CAPACITY (fail-closed topology guard bounds).
# The shipped C++ hier barrier ``ShmemInternalBarrierHierBlock``
# (include/mori/shmem/shmem_device_api.hpp) packs its rendezvous flags into the
# fixed 128-uint64 internalSync region with this DISJOINT layout:
#   HIER_PEER_BASE [96 .. 96+num_nodes)          coordinator inter-node inbox
#   HIER_LOCAL_BASE[112 .. 112+ranks_per_node)   local PE -> coordinator inbox
#   HIER_REL_SLOT  [120], HIER_GEN_SLOT [126]
# So the local inbox must fit within [112, 120) => ranks_per_node <= 8, and the
# coordinator inbox must fit within [96, 112) => num_nodes <= 16. Exceeding
# either bound silently overwrites an adjacent slot (the REL/GEN slots or the
# local inbox), corrupting the barrier. The Python guard fails CLOSED with an
# actionable error before any such topology reaches the kernel. The shipped
# config (num_nodes=2, ranks_per_node=8) sits exactly at the ranks_per_node
# bound and well within the num_nodes bound, so the guard never trips it.
MORI_HIER_MAX_RANKS_PER_NODE = 8
MORI_HIER_MAX_NUM_NODES = 16


def _clamp_sdma_channels():
    _v = os.environ.get("MORI_SDMA_NUM_CHANNELS")
    if _v is None:
        return
    try:
        _n = int(_v)
    except ValueError:
        return
    if _n > MORI_SDMA_CH_HW_MAX:
        os.environ["MORI_SDMA_NUM_CHANNELS"] = str(MORI_SDMA_CH_HW_MAX)
        print(
            "[hier_fill] MORI_SDMA_NUM_CHANNELS=%s exceeds the MI300X SDMA "
            "queue-slot cap (%d); clamping to %d to avoid the anvil.cpp:228 "
            "queue-creation crash." % (_v, MORI_SDMA_CH_HW_MAX, MORI_SDMA_CH_HW_MAX),
            flush=True,
        )


_clamp_sdma_channels()


# Falsy-string set shared by every MORI_HIER_* boolean env flag. A value is
# "true" iff it is NOT one of these. Kept as a module constant so the parse is
# identical everywhere (the tuple is a membership test, so order is irrelevant).
_ENV_FALSE = ("0", "", "false", "False")


def _env_true(key: str, default: str = "0") -> bool:
    """Behavior-identical replacement for the inline
    ``os.environ.get(key, default) not in ("0", "", "false", "False")`` idiom."""
    return os.environ.get(key, default) not in _ENV_FALSE


def _env_int(key: str, default: str) -> int:
    """Behavior-identical replacement for the inline
    ``int(os.environ.get(key, default))`` idiom used by MORI_HIER_* int flags."""
    return int(os.environ.get(key, default))


def _auto_ranks_per_node(my_pe: int, npes: int) -> int:
    """Detect how many ranks are co-located on this physical node so callers can
    use the same signature as the flat ``AllgatherSdma`` (no ``ranks_per_node``).

    Resolution order:
      1. launcher-provided local world size (``LOCAL_WORLD_SIZE`` from torchrun,
         or the MPI equivalents),
      2. group ranks by hostname over the initialized process group,
      3. fall back to ``npes`` (treat everything as a single node).

    Always returns a positive integer that divides ``npes``.
    """
    for key in (
        "LOCAL_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_SIZE",
        "MV2_COMM_WORLD_LOCAL_SIZE",
    ):
        v = os.environ.get(key)
        if v and v.isdigit():
            g = int(v)
            if g > 0 and npes % g == 0:
                return g
    try:
        import torch.distributed as dist

        if (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() == npes
        ):
            host = socket.gethostname()
            hosts = [None] * npes
            dist.all_gather_object(hosts, host)
            g = sum(1 for h in hosts if h == host)
            if g > 0 and npes % g == 0:
                return g
    except Exception:
        pass
    return npes


# NOTE: ``AllgatherSdma`` (and hence the compiled C++ .so) is imported lazily
# inside ``HierAllGather.__init__`` so that the pure-Python executable specs
# (``hier_allgather_reference`` / ``inter_node_ring_reference``) can be
# imported and unit-tested on a CPU-only / no-.so environment.


def hier_allgather_reference(
    shards: Sequence["torch.Tensor"],
    num_nodes: int,
    ranks_per_node: int,
) -> List["torch.Tensor"]:
    """Executable spec of the hierarchical AllGather data movement (CPU).

    This mirrors *exactly* the byte/element-offset arithmetic the GPU path
    must perform, so that the offset math can be validated bit-exactly on CPU
    -- with NO GPU, NO SDMA and NO RDMA. It is the algorithmic contract for
    the real implementation.

    Three phases:

      1. **Intra-node gather (SDMA on device):** for node ``n`` whose ranks are
         the contiguous block ``[n*G, n*G+G)``, every rank in the node ends up
         holding ``node_block[n] = concat(shard[n*G], ..., shard[n*G+G-1])``.
      2. **Inter-node gather (RDMA ring on device):** the ``N`` node-blocks are
         all-gathered across nodes; every node ends up holding all ``N`` blocks.
      3. **Placement:** each rank's output is the node-blocks concatenated in
         node order ``concat(node_block[0], ..., node_block[N-1])``.

    Because node ``n`` owns ranks ``[n*G, n*G+G)`` and blocks are concatenated
    in node order, the result equals ``concat(shard[0], ..., shard[W-1])`` --
    i.e. rank-major order, identical to
    ``torch.distributed.all_gather_into_tensor``.

    Parameters
    ----------
    shards:
        ``world = num_nodes * ranks_per_node`` tensors, each the per-rank input
        shard (same shape/dtype). Indexed by global rank.
    num_nodes, ranks_per_node:
        ``N`` and ``G``; ``world == N * G``.

    Returns
    -------
    list of length ``world``; entry ``r`` is rank ``r``'s full output tensor.
    """
    G = ranks_per_node
    N = num_nodes
    world = N * G
    if len(shards) != world:
        raise ValueError(
            f"expected {world} shards (num_nodes*ranks_per_node), got {len(shards)}"
        )
    count = shards[0].numel()
    dtype = shards[0].dtype
    for r, s in enumerate(shards):
        if s.numel() != count or s.dtype != dtype:
            raise ValueError(f"shard {r} has mismatched numel/dtype")

    # Phase 1: build each node's contiguous G-shard block via per-rank offsets.
    node_blocks: List[torch.Tensor] = []
    for n in range(N):
        block = torch.empty(count * G, dtype=dtype)
        for g in range(G):
            src = shards[n * G + g].reshape(-1)
            block[g * count : (g + 1) * count] = src
        node_blocks.append(block)

    # Phases 2+3: every rank lays the N node-blocks down in node order.
    block_elems = count * G
    outputs: List[torch.Tensor] = []
    for _ in range(world):
        out = torch.empty(count * world, dtype=dtype)
        for n in range(N):
            out[n * block_elems : (n + 1) * block_elems] = node_blocks[n]
        outputs.append(out)
    return outputs


def inter_node_ring_reference(
    node_blocks: Sequence["torch.Tensor"],
) -> List["torch.Tensor"]:
    """Executable spec of the inter-node ring AllGather (CPU, no RDMA).

    This mirrors *exactly* the schedule of the device kernel
    ``AllGatherRingKernel`` (``include/mori/collective/inter_node/kernels/
    all_gather.hpp``) that the M2 inter-node phase will launch over the
    ``N`` node-leaders. Each leader contributes one node-block (the ``G``
    local shards already gathered intra-node by the SDMA path). The kernel
    treats a contiguous ``N``-chunk buffer where chunk ``k`` lives at offset
    ``k * block_elems``; leader ``n`` starts with only its own chunk ``n``
    filled, then runs ``N-1`` rounds of:

        nextPeer     = (myPe + 1) % N
        sendDataRank = (myPe - i + N) % N   # chunk pushed to nextPeer
        recvDataRank = (myPe - i - 1 + N) % N  # chunk received from prev

    After ``N-1`` rounds every leader holds all ``N`` chunks in node order,
    i.e. ``concat(node_block[0], ..., node_block[N-1])``. The per-round
    clone models the flag/quiet barrier the kernel uses between rounds.

    Returns one buffer per node-leader (all identical after the ring); the
    intra-node placement phase then broadcasts leader ``n``'s buffer to the
    ``G`` ranks of node ``n``.
    """
    N = len(node_blocks)
    if N == 0:
        return []
    block_elems = node_blocks[0].numel()
    dtype = node_blocks[0].dtype
    for b in node_blocks:
        if b.numel() != block_elems or b.dtype != dtype:
            raise ValueError("node_blocks must share numel and dtype")

    bufs: List[torch.Tensor] = []
    for n in range(N):
        buf = torch.zeros(block_elems * N, dtype=dtype)
        buf[n * block_elems : (n + 1) * block_elems] = node_blocks[n]
        bufs.append(buf)

    for i in range(N - 1):
        # All sends in a round happen against the start-of-round state.
        snapshot = [b.clone() for b in bufs]
        for my_pe in range(N):
            next_peer = (my_pe + 1) % N
            send_rank = (my_pe - i + N) % N
            lo = send_rank * block_elems
            hi = lo + block_elems
            bufs[next_peer][lo:hi] = snapshot[my_pe][lo:hi]
    return bufs


class HierAllGather:
    """Hierarchical AllGather: intra-node SDMA + inter-node RDMA ring.

    Parameters
    ----------
    my_pe, npes:
        Global rank and world size (``npes == num_nodes * ranks_per_node``).
    ranks_per_node:
        Number of ranks (GPUs) co-located on one node, i.e. ``G``. Optional
        and keyword-only: when omitted it is auto-detected (``LOCAL_WORLD_SIZE``
        from the launcher, else grouping ranks by hostname, else ``npes`` for a
        single node), so callers can use the same signature as the flat
        ``AllgatherSdma``.
    transit_buffer_size:
        Optional single combined transit size (flat ``AllgatherSdma``
        compatibility); split into input/output when those are not given.
    input_buffer_size, output_buffer_size:
        Per-rank input byte capacity and total output byte capacity to
        pre-allocate inside the SDMA transit buffers. Sized for the largest
        ``count * dtype`` that will be passed to ``__call__``.
    copy_output_to_user:
        When True the gathered result is copied into the user ``output``
        tensor (required for cached PyTorch allocations).
    """

    def __init__(
        self,
        my_pe: int,
        npes: int,
        input_buffer_size: Optional[int] = None,
        output_buffer_size: Optional[int] = None,
        transit_buffer_size: Optional[int] = None,
        copy_output_to_user: bool = True,
        *,
        ranks_per_node: Optional[int] = None,
        inter_num_qp: Optional[int] = None,
        leader_only: Optional[bool] = None,
        gather_in_place: Optional[bool] = None,
        out_in_place: Optional[bool] = None,
        inter_num_blocks: Optional[int] = None,
        fuse_barrier: Optional[bool] = None,
        slice_inter: Optional[bool] = None,
        slice_fused: Optional[bool] = None,
        slice_oop: Optional[bool] = None,
        slice_min_bytes: Optional[int] = None,
        slice_overlap: Optional[bool] = None,
        slice_fuse_ib: Optional[bool] = None,
        slice_pipe: Optional[bool] = None,
        slice_pipe_chunks: Optional[int] = None,
        slice_pipe_overlap: Optional[bool] = None,
        slice_direct: Optional[bool] = None,
        standalone_fast: bool = False,
    ):
        # M4: fan the inter-node RDMA ring put across this many QPs to
        # better fill the NIC (RCCL drives many channels; the ring used 1 of the
        # transport's MORI_NUM_QP_PER_PE provisioned QPs). Defaults to the
        # provisioned count (env MORI_NUM_QP_PER_PE, default 4). The kernel only
        # fans out for true cross-node (RDMA) neighbours, so single-node runs are
        # unaffected (stay single-warp).
        if inter_num_qp is None:
            inter_num_qp = _env_int("MORI_NUM_QP_PER_PE", "4")
        self.inter_num_qp = max(1, inter_num_qp)
        # Opt-in multi-block ("channels") inter-node ring: num_blocks>1 launches
        # the ring as that many CTAs, each driving a disjoint chunk sub-range on
        # its own QP (the RCCL channel model). Engaged only for true RDMA
        # neighbours; single-node sims fall back to one working block. Default 1
        # (single-block ring) -- multi-block was measured neutral/negative (the
        # per-NIC RDMA throughput is already saturated at numQp>=4, so spreading
        # the same QPs across CTAs adds scheduling overhead without NIC bandwidth).
        # Kept opt-in via env MORI_HIER_RING_BLOCKS for A/B.
        if inter_num_blocks is None:
            inter_num_blocks = _env_int("MORI_HIER_RING_BLOCKS", "1")
        self.inter_num_blocks = max(1, inter_num_blocks)
        # M4: opt-in leader-only pipeline (DESIGN's primary design).
        # Default every-rank-direct (proven correct since ). Toggle via
        # env MORI_HIER_LEADER_ONLY=1 or the explicit arg. See the N>=2 branch.
        if leader_only is None:
            leader_only = _env_true("MORI_HIER_LEADER_ONLY", "0")
        self.leader_only = bool(leader_only)
        # M4: opt-in "gather-in-place" -- have the intra-node SDMA
        # gather write its node-block DIRECTLY into the inter-node ring slot,
        # eliminating the prepare_sync copy-IN (a full node-block D2D copy) and
        # the node_block intermediate. Default OFF: the proven staged path (intra
        # -> node_block -> ring slot, since ) stays the default.
        if gather_in_place is None:
            gather_in_place = _env_true("MORI_HIER_GATHER_IN_PLACE", "0")
        self.gather_in_place = bool(gather_in_place)
        # Opt-in "out-in-place": leave the gathered result in the inter-node ring
        # buffer and read it via ``result_tensor`` instead of copying it to a user
        # output. Implies gather_in_place (the gather writes straight into the ring
        # slot) so there is zero staging on either side. Default OFF: the staged
        # path (writes the user output) stays the default -- out-in-place changes
        # the result-delivery contract (read ``result_tensor``) and was measured a
        # tiny (~+2.6%) win only, so it is opt-in via MORI_HIER_OUT_IN_PLACE.
        if out_in_place is None:
            out_in_place = _env_true("MORI_HIER_OUT_IN_PLACE", "0")
        self.out_in_place = bool(out_in_place)
        # "fuse-barrier": drop the intra-node SDMA gather's finish ShmemBarrierAll
        # in the every-rank-direct N>=2 path (removes 1 of 4 global barriers/op).
        # CORRECTNESS INVARIANT: the dropped intra finish barrier is redundant --
        # the PUSH gather's in-kernel flag-wait already makes this PE's node-block
        # complete on kernel return, and the inter ring's prepare_sync
        # ShmemBarrierAll immediately follows to synchronize all PEs before the
        # ring's cross-PE atomics; flags are monotonic per-call (no reset) so there
        # is no cross-call flag hazard. Crash-safe via the _prev_op_completed guard
        # (keeps the entry barrier on first op / after any mid-pipeline exception).
        # Default ON (bit-exact, cuts ~40% of the small/mid per-op floor); applies
        # only to the every-rank-direct path (not leader-only). Set
        # MORI_HIER_FUSE_BARRIER=0 to restore the pre-fuse baseline for A/B.
        if fuse_barrier is None:
            fuse_barrier = _env_true("MORI_HIER_FUSE_BARRIER", "1")
        self.fuse_barrier = bool(fuse_barrier)
        # M5: opt-in SLICED 2-D AllGather -- the real bandwidth lever
        # ( NEXT). The default every-rank-direct path has each of the G
        # local ranks push its FULL node-block (G*count) to its same-index peer,
        # so node n's block crosses the boundary G times => per-NIC inter bytes =
        # G*count. RCCL pushes only ~count/NIC. We close that G x gap WITHOUT the
        # single-NIC funnel of leader-only ( negative):
        #   1. Inter ring over same-local-index peers {g, g+G, ...} but each rank
        #      contributes only its OWN shard (count, NOT the G*count node-block).
        #      Because slice_g(B_n) == shard[n*G+g] == this rank's own input, the
        #      ring yields C_g = [slice_g(B_0), ..., slice_g(B_{N-1})] in node
        #      order. Per-NIC inter bytes drop to (N-1)*count -- a G x cut, spread
        #      across ALL G NICs (no funnel).
        #   2. N intra-node SDMA gathers (one per node-block m) reassemble full
        #      B_m = concat_g slice_g(B_m) into output[m*block:(m+1)*block]. The
        #      SDMA gather concatenates by group_pos=local_rank, so the result is
        #      exactly rank-major concat(B_0..B_{N-1}) == torch all_gather.
        # The extra intra gather rides fast XGMI while the inter phase (the ~80%
        # bottleneck) shrinks ~G x. Default ON (the proven-best bandwidth path,
        # bit-exact and >= the non-sliced fuse-barrier path at every tested size).
        # It owns its own inter+intra data path, so it is incompatible with
        # leader_only / out_in_place / gather_in_place; if any of those is
        # explicitly enabled, slice defaults OFF so those levers still work. Set
        # MORI_HIER_SLICE=0 to force the pre-slice baseline for A/B.
        _slice_conflict = self.leader_only or self.out_in_place or self.gather_in_place
        if slice_inter is None:
            slice_inter = _env_true("MORI_HIER_SLICE", "0" if _slice_conflict else "1")
        self.slice_inter = bool(slice_inter)
        # FUSED sliced Phase B: fold the N intra reassembly gathers into ONE batch.
        # CORRECTNESS INVARIANT: the fused path stacks the N gathers into DISJOINT
        # regions of one enlarged transit (dst_base_offset = m*block) so they never
        # overlap; it keeps only the m==0 entry barrier + one bulk copy-OUT + one
        # exit barrier, and flags stay monotonic per-call, so there is no
        # cross-gather race and the output is byte-identical to the unfused sliced
        # path. Default ON (proven-best variant); only meaningful with slice_inter.
        # Set MORI_HIER_SLICE_FUSED=0 for the unfused sliced path.
        if slice_fused is None:
            slice_fused = _env_true("MORI_HIER_SLICE_FUSED", "1")
        self.slice_fused = bool(slice_fused)
        # M5: opt-in "slice out-of-place elimination" -- run the sliced
        # Phase A inter ring in out_in_place mode and have Phase B read its input
        # (the collection C_g = [slice_g(B_0)..slice_g(B_{N-1})]) DIRECTLY from
        # the ring buffer (full_tensor) instead of the finish_sync copy-OUT into
        # a separate ``_slice_scratch``. This is the sliced analog of lever (b):
        # the inter ring's finish copy-OUT moves N*count bytes (the whole
        # collection) every op; reading the ring buffer in place drops it. The
        # ring buffer is symmetric/uncached so the N Phase-B gather copy-INs now
        # read from uncached HBM -- the same offset that made gather_in_place
        # NEUTRAL on the non-sliced path (-24); whether it nets out
        # positive on the sliced (smaller, count-sized) reads is what the A/B
        # measures. Default OFF; only meaningful with slice_inter.
        if slice_oop is None:
            slice_oop = _env_true("MORI_HIER_SLICE_OOP", "0")
        self.slice_oop = bool(slice_oop)
        # Per-call SIZE THRESHOLD for the sliced path. The sliced 2-D path wins big
        # at large per-rank payloads but loses at small/mid (its extra kernel
        # launches + N reassembly gathers cost more than the saved inter bytes), so
        # engage slice ONLY when the per-rank payload is >= this many bytes; below
        # it, fall through to the non-sliced fuse-barrier path. Default 8 MiB. Set
        # MORI_HIER_SLICE_MIN_BYTES=0 to force slice at all sizes (A/B / tests).
        if slice_min_bytes is None:
            slice_min_bytes = _env_int(
                "MORI_HIER_SLICE_MIN_BYTES", str(8 * 1024 * 1024)
            )
        self.slice_min_bytes = max(0, slice_min_bytes)
        # MID/SMALL-SIZE BAND -> stream pipe-overlap path. For per-rank payloads
        # BELOW slice_min_bytes, route [pipe_band_min, slice_min) to the
        # stream-ordered chunked-ring pipeline overlap path (chunked side-stream
        # gathers hide under the ring, all barriers on-device), which is faster than
        # the non-sliced path across the sub-threshold band. Default ON; disable
        # with MORI_HIER_PIPE_BAND=0. At/above slice_min the slice_direct path wins.
        self.pipe_band = _env_true("MORI_HIER_PIPE_BAND", "1")
        self.pipe_band_min_bytes = _env_int("MORI_HIER_PIPE_BAND_MIN_BYTES", "0")
        # Opt-in: OVERLAP Phase-A (inter RDMA ring) with the LOCAL node-block's
        # Phase-B reassembly gather. The gather for block m=node_id needs only this
        # rank's OWN input (== collection[node_id]) and does NOT depend on the ring,
        # so it runs on a SIDE stream concurrently with the inter ring, hiding ~1/N
        # of the intra phase. Deadlock-safe: every ShmemBarrierAll is host-blocking
        # (deterministic program order) and hipStreamSynchronize only targets its
        # own (main) stream. Only meaningful with slice_inter + slice_fused. Default
        # OFF; toggle MORI_HIER_SLICE_OVERLAP=1.
        if slice_overlap is None:
            slice_overlap = _env_true("MORI_HIER_SLICE_OVERLAP", "0")
        self.slice_overlap = bool(slice_overlap)
        # Drop the REDUNDANT Phase-B entry barrier in the sliced+fused NON-overlap
        # path. CORRECTNESS INVARIANT: that path runs the inter ring first; its
        # finish_sync issues a global ShmemBarrierAll immediately before the
        # Phase-B m==0 gather's own entry barrier -- two back-to-back all-PE
        # barriers with no remote memory op between them, so the second is
        # redundant (every PE has passed the inter finish barrier before any PE
        # starts Phase-B, so every peer's out_ transit from the previous op is
        # already free). Byte-identical output (pure host-sync removal). Does NOT
        # apply to the overlap path (its local gather runs concurrently on a side
        # stream and must keep its own entry barrier). Default ON; set
        # MORI_HIER_SLICE_FUSE_IB=0 to restore the entry barrier for A/B.
        if slice_fuse_ib is None:
            slice_fuse_ib = _env_true("MORI_HIER_SLICE_FUSE_IB", "1")
        self.slice_fuse_ib = bool(slice_fuse_ib)
        # Opt-in CHUNKED (strided) Phase-B reassembly: split each node-block's
        # reassembly gather into ``slice_pipe_chunks`` element-range chunks, each
        # writing ck elements per peer at slot stride = count via gather_kernel
        # dst_slot_stride, so chunk j of peer g lands at m*block + g*count + j*ck --
        # byte-identical to the unchunked gather (the strided-write foundation for
        # the chunked inter/intra pipeline). Only meaningful with slice_inter +
        # slice_fused (non-overlap, non-oop). Default OFF; toggle
        # MORI_HIER_SLICE_PIPE=1.
        if slice_pipe is None:
            slice_pipe = _env_true("MORI_HIER_SLICE_PIPE", "0")
        self.slice_pipe = bool(slice_pipe)
        if slice_pipe_chunks is None:
            slice_pipe_chunks = _env_int("MORI_HIER_SLICE_PIPE_CHUNKS", "2")
        self.slice_pipe_chunks = max(1, slice_pipe_chunks)
        # Uneven front/tail chunk split for the K==2 SLICE_PIPE_OVERLAP pipeline.
        # The exposed serial cost is
        #   ring(chunk_0)                         [front: nothing to hide under]
        #   + max(phaseB(chunk_0), ring(chunk_1)) [overlapped middle]
        #   + phaseB(chunk_{K-1})                 [tail: no successor ring]
        # When the inter-node RDMA ring is slower per byte than the intra-node
        # XGMI Phase-B gather, the front ring dominates while the tail gather is
        # cheap; shrinking chunk_0 shrinks the exposed front ring but grows the
        # exposed tail gather, so the optimum is a per-fabric balance -- expose it
        # as a lever. ``split`` = fraction of ``count`` given to the FIRST chunk
        # when K==2 (rest to the second); None or K!=2 => even split. Bit-exact by
        # construction: the strided reassembly writes each element to the same
        # final block slot regardless of chunk boundary (dst_slot_stride=count;
        # disjoint per-(k,m) regions), so only the order/size of the pipeline
        # stages changes, never the output bytes.
        sps = os.environ.get("MORI_HIER_SLICE_PIPE_SPLIT")
        self.slice_pipe_split = float(sps) if sps not in (None, "") else None
        # Strided-gather enabler:
        # CHUNK THE INTER RING into K pipeline stages and OVERLAP each chunk's
        # Phase-B reassembly gather (on a side SDMA stream) with the NEXT chunk's
        # inter RDMA ring (on the main stream). Because the ring's prepare/finish
        # ShmemBarrierAll are host-blocking, after self._inter(chunk k) returns
        # chunk k's N slices are physically in scratch; we launch chunk k's gather
        # on the side stream (no barrier) and immediately call self._inter(chunk
        # k+1) on the main stream -- the side SDMA gather runs concurrently with
        # the main RDMA ring (distinct engines, disjoint scratch regions). Only the
        # LAST chunk's gather (~1/K of the data) is serial after the final ring, so
        # the ~1.5ms serial Phase-B tail collapses to ~tail/K. Strided write
        # (dst_slot_stride=count) lands each chunk in its final block slot.
        # Requires slice_inter+slice_fused, non-oop, non-slice_overlap. Default OFF
        # (toggle MORI_HIER_SLICE_PIPE_OVERLAP=1). Cost: +2(K-1) ring barriers; net
        # win only if hidden gather tail > added barrier overhead (A/B decides).
        if slice_pipe_overlap is None:
            slice_pipe_overlap = _env_true("MORI_HIER_SLICE_PIPE_OVERLAP", "0")
        self.slice_pipe_overlap = bool(slice_pipe_overlap)
        # Per-chunk landing fence for SLICE_PIPE_OVERLAP.
        # The deferred overlap loop (defer_inter_fin=sr + gather
        # prepare_barrier=False) has no cross-PE fence guaranteeing a peer's
        # chunk-k inter-ring RDMA landing is globally visible before the side
        # gather reads that peer's ``region`` over XGMI -- a per-chunk landing
        # race that drifts under contention. A full global ShmemBarrierOnStream
        # per chunk closes it but serializes the K-pipe and eats the overlap.
        # Since the dependency is purely intra-node (the side gather reads only
        # same-node peers' ``region`` over XGMI), the inter-node half of the
        # global barrier is unnecessary. The default "intra" fence keeps
        # defer_inter_fin=True and instead arms the first Phase-B gather of each
        # chunk with prepare_barrier=True -- the intra-node subgroup entry
        # ShmemBarrier (G ranks, XGMI-scope, no NIC quiet-drain, no inter-node
        # rendezvous), the same primitive fuse_local uses as its landing fence.
        # It orders all G local peers past their chunk-k ring copy-OUT and
        # threadfence before any reads a peer region, keeping the overlap.
        # Modes: "intra"/"1" (default, cheap intra-node barrier), "global" (full
        # ShmemBarrierOnStream), "0"/off (deferred, drifting).
        _spf = os.environ.get("MORI_HIER_SLICE_PIPE_FENCE", "1").strip().lower()
        if _spf in ("0", "", "false", "off"):
            self.slice_pipe_fence = "off"
        elif _spf in ("global", "barrier", "full"):
            self.slice_pipe_fence = "global"
        else:
            self.slice_pipe_fence = "intra"
        # STREAM-ORDERED inter ring. Replaces the inter ring's
        # host-blocking prepare/finish (hipStreamSynchronize + host bootNet
        # ShmemBarrierAll) with the on-device ShmemBarrierOnStream prepare/finish,
        # removing 2 CPU<->GPU round-trips per inter ring op. This is the lever
        # this work  measured at +6-7% standalone; cross-read per COORD
        # "combine levers". Stacks on the slice path (the 64MiB winner). Default
        # Default ON (: +10-12% @64MiB fp32 xnode, bit-exact, more stable;
        # only affects the slice path, which gates the 64MiB acceptance number).
        # Set MORI_HIER_STREAM_RING=0 / --no-stream-ring to restore host-sync.
        self.stream_ring = _env_true("MORI_HIER_STREAM_RING", "1")
        # STREAM-ORDERED Phase-B finish_batch. The fused sliced
        # Phase-B still ends with finish_batch = bulk copy-OUT + host
        # hipStreamSynchronize + host ShmemBarrierAll -- the LAST host CPU<->GPU
        # round-trip in the op. Replace it with finish_batch_stream
        # (ShmemBarrierOnStream) so, paired with the stream_ring, the
        # whole op (inter ring + Phase-B gathers + copy-OUT) is fully on-stream
        # with NO host stall. Only the default fused non-overlap, non-pipe slice
        # path uses it. Default ON; set MORI_HIER_STREAM_INTRA=0 /
        # --no-stream-intra to restore the host-synced finish_batch for A/B.
        self.stream_intra = _env_true("MORI_HIER_STREAM_INTRA", "1")
        # DEFER the Phase-B finish_batch_stream fence. The
        # default fused-stream slice op issues 3 on-stream global ShmemBarrierOn
        # Stream fences/op: inter prepare (#1), inter finish (#2), Phase-B finish
        # (#3). #3 (end of op i) is back-to-back -- across the op boundary -- with
        # the NEXT op's #1 (inter prepare), with no remote memory op between them
        # on the stream. #1 already globally fences (all PEs) AFTER op i's
        # copy-OUT and BEFORE any peer reuses the shared transit/ring buffers, so
        # #3 is redundant for every op that is followed by another hier op.
        # Dropping it removes 1 of 3 on-stream fences/op. SAFE because: (a) the
        # copy-OUT is stream-ordered so THIS PE's output is correct without #3;
        # (b) cross-PE buffer REUSE is covered by the successor op's #1 (slice
        # path) or its forced intra entry barrier (non-slice path: the size-
        # dispatcher resets _prev_op_completed on a path switch -> entry barrier
        # fires); (c) the LAST op (no successor) needs no reuse fence and its
        # output is already stream-correct. Only the default fused non-overlap,
        # non-pipe, non-oop slice path (which uses finish_batch_stream) defers.
        # Default ON; set MORI_HIER_SLICE_DEFER_FIN=0 to restore the fence (A/B).
        self.slice_defer_fin = _env_true("MORI_HIER_SLICE_DEFER_FIN", "1")
        # defer the INTER ring's finish_stream fence (the
        # stream-ordered ShmemBarrierOnStream guarding cross-PE ring-buffer reuse)
        # to the NEXT slice op's prepare_stream barrier. The ring buffer is reused
        # ONLY by another op through this same _inter handle, and prepare_stream
        # ALWAYS fences (global, on-stream) before its ring kernel issues the peer
        # RDMA puts -> the successor's prepare fence already provides the required
        # ordering, so this finish fence is redundant for any op with a slice
        # successor. The copy-OUT into the scratch collection stays stream-ordered
        # (Phase B reads a correct collection regardless); only the cross-PE reuse
        # fence is deferred. Mirrors slice_defer_fin (Phase-B, ). Only on
        # the non-oop slice path (stream_ring). DEFAULT ON
        # (validated +0.85-1.0%, bit-exact) -- same safety class as slice_defer_fin
        # (already default ON since ): the successor's prepare_stream fence
        # guards cross-PE ring reuse; the last op (no successor) reuses nothing and
        # its copy-OUT is stream-ordered; the size-dispatcher's path switch resets
        # _prev_op_completed forcing an entry barrier. Set
        # MORI_HIER_SLICE_DEFER_INTER_FIN=0 (--no-slice-defer-inter-fin) to restore
        # the fence for A/B.
        self.slice_defer_inter_fin = _env_true("MORI_HIER_SLICE_DEFER_INTER_FIN", "1")
        # PHASE-B ENTRY BARRIER (accuracy). Force a full cross-PE
        # ShmemBarrierOnStream on the FIRST Phase-B reassembly gather even when
        # slice_fuse_ib would otherwise drop it. The Phase-B intra SDMA gathers
        # read PEER ranks' `collection` (the inter ring's per-node-block output)
        # over XGMI; slice_fuse_ib=1 relies on the ring's deferred/own finish for
        # cross-PE visibility, but under FSDP tight back-to-back overlap the SDMA
        # read can observe a peer's collection before that peer's ring finish is
        # globally visible -> the residual completion race (host-sync-recoverable
        # loss drift). A full entry barrier here strictly orders every peer's ring
        # finish BEFORE any Phase-B gather reads it, on-device (no host sync).
        # Default OFF (preserves the perf path); set MORI_HIER_PHASEB_ENTRY_BARRIER=1
        # to A/B the accuracy fix and measure its one-barrier/op perf cost.
        self.phaseb_entry_barrier = _env_true("MORI_HIER_PHASEB_ENTRY_BARRIER", "0")
        # DIRECT-PATH LOCAL-BLOCK OVERLAP. In the shipped
        # slice_direct path the dominant cost is now Phase B (the XGMI reassembly
        # gathers ~2.5ms), not Phase A (the sliced RDMA ring ~1.6ms) -- see the
        # measured phase split. The reassembly gather for the LOCAL
        # node-block (m == node_id) builds B_{node_id} = concat_g shard[node_id*G+g]
        # entirely from the G local ranks' OWN inputs (slice_g(B_node_id) ==
        # this rank's input == collection[node_id]); it has ZERO dependency on the
        # inter ring. So run it on a SIDE stream CONCURRENTLY with the ring kernel,
        # hiding ~1/N of Phase B (~1.25ms for N=2) under Phase A (~1.6ms).
        #
        # SAFETY (distinct from the pairwise-barrier race): the shipped
        # non-overlap direct path's SOLE global entry barrier is ALREADY the ring's
        # prepare_stream ShmemBarrierOnStream (its finish is deferred via
        # slice_defer_inter_fin and the direct gathers skip their entry barrier via
        # slice_fuse_ib). We keep that exact barrier model: split the ring into
        # prepare_stream_only (the global entry barrier, on main) + the kernel/
        # finish, and launch the local-block gather barrier-free on the side stream
        # AFTER side.wait_stream(main) (so it observes the entry barrier) and
        # BEFORE the ring kernel. Only ONE global on-stream fence is ever in flight
        # (no concurrent-barrier aliasing). Write targets are disjoint (side ->
        # output block node_id; ring -> collection scratch; main gathers -> output
        # blocks m != node_id) and the SDMA gather / ring use distinct flag
        # buffers. Default OFF; toggle MORI_HIER_SLICE_DIRECT_OVERLAP=1.
        #
        # Validated neutral on true cross-node RDMA (N=2 G=4 fp32
        # 64MiB/rank, both bit-exact + dispatch-span):
        #   overlap ON:  mori 3.802ms 141.2 GB/s | rccl 148.2 | 1.05x
        #   overlap OFF: mori 3.768ms 142.5 GB/s | rccl 155.8 | 1.09x
        # => -0.9% mori-side (NEUTRAL/slightly worse; ratio delta is RCCL-draw
        # noise). The local-block reassembly gather is ALREADY hidden in the
        # shipped single-stream pipeline (the GPU overlaps the SDMA gather with the
        # RDMA ring without an explicit side stream -- measurements show the
        # serial path already overlaps ~0.32ms); forcing it onto a side stream only
        # adds the side.wait_stream / main.wait_stream merge overhead, which offsets
        # the recovered overlap. This CLOSES the "overlap Phase A with the local
        # Phase-B block" lever from the DIRECT-path angle (/7 closed it on the
        # old copy-OUT path). Kept opt-in so it is not re-litigated; default stays
        # the shipped serial direct path (~142 GB/s, 1.06x).
        self.slice_direct_overlap = _env_true("MORI_HIER_SLICE_DIRECT_OVERLAP", "0")
        # MULTI-STREAM Phase-B reassembly.
        # The N disjoint slice_fused reassembly gathers currently run SERIALLY on
        # one stream, so at most one OneShotAllGatherSdmaSubGroupKernel drives the
        # SDMA copy engines at a time. Each gather writes a DISJOINT node-block
        # region of the (registered) user output (dst_block_offset = m*block_count)
        # and reads a disjoint source slice (collection[m] / own input), so the N
        # gathers have NO data dependency on each other -> distributing them
        # round-robin across a pool of side streams lets the runtime run them
        # CONCURRENTLY without changing any byte written. Bit-exact BY CONSTRUCTION
        # (identical launches, identical args, disjoint outputs; only the stream
        # assignment changes). Ordering is preserved by fences: the entry barrier /
        # ring finish stays on the MAIN stream and every side stream does
        # side.wait_stream(main) before its gather; the MAIN stream does
        # main.wait_stream(side) for every side before finish_direct_stream, so the
        # completion fence still strictly follows ALL gathers. Default 1 ==
        # single-stream shipped path byte-identical (pool never allocated).
        # NOTE at N=2 with fuse_local the Phase-B loop is only N-1=1 remote gather
        # (local block folded into the ring), so this lever is a NO-OP there; it
        # engages on the plain fused-direct path (all N gathers in Phase B) and at
        # N>2. Measures whether concurrent gathers beat the ~305 GB/s XGMI
        # reassembly wall (engine-fan WITHIN a gather was fabric-bound; this
        # tests whether MULTIPLE gathers in flight lift utilization).
        try:
            self.reasm_streams = _env_int("MORI_HIER_REASM_STREAMS", "1")
        except ValueError:
            self.reasm_streams = 1
        if self.reasm_streams < 1:
            self.reasm_streams = 1
        self._reasm_stream_pool = None
        # FUSED ring || local-block gather (the RCCL-parity
        # lever this work proved out, ported. The slice_direct_overlap
        # path above recovers the NIC-ring || XGMI-local-gather overlap by running
        # the local block on a SIDE stream, but pays a side.wait_stream +
        # main.wait_stream host merge that offsets the win (, ~neutral).
        # This lever instead runs BOTH halves in ONE kernel launch
        # (FusedRingLocalGatherKernel_u32: blocks [0,num_blocks) = RDMA ring, last
        # block = local-block SDMA gather), so the overlap is intrinsic to the
        # grid with NO host merge and one fewer kernel launch. Engaged only on the
        # default fused stream-ordered slice_direct path (same prereqs as
        # slice_direct_overlap). Default ON (MORI_HIER_FUSE_LOCAL=0 /
        # --no-fuse-local to A/B the prior slice_direct path); when ON it takes
        # precedence over slice_direct_overlap.
        # SHIPPED default ON after a clean-window A/B at 64 MiB
        # fp32 (5 reps, bit-exact green) showed the fused ring||local-gather
        # kernel hits 200.8/200.3 GB/s vs RCCL 145.9/155.4 (ratio 0.73/0.78x --
        # BEATS RCCL) vs the prior slice_direct default 142.6/142.2 (1.04x) ==
        # +41% mori-side and parity EXCEEDED. Engages ONLY on the >=8 MiB sliced
        # path (use_slice gate); small sizes stay on the safe non-slice path so
        # the fused kernel's small-size constraint never triggers.
        # CORRECTNESS: default flipped to OFF. In-situ FSDP AGVERIFY
        # (2-node world=8, copy-out, VOCAB=32000 LAYERS=28) shows the FUSED
        # ring||local-gather kernel produces STALE remote-half AG output on ~48%
        # of per-layer all-gathers (184/384) under FSDP's tight back-to-back
        # overlap -- the RDMA-ring buffer is read out (finish_ring_stream copy +
        # remote-block direct gathers) before the concurrently-launched ring
        # CTA's remote puts are globally visible to the subsequent readers. The
        # SERIAL monolithic ring path (fuse_local OFF) drops this to ~2-3%
        # (8-11/384) at the same config, and DEBUG_SYNC is 0/384 -- i.e. the
        # fused concurrency is the dominant offender, NOT the ring flag/data QP
        # ordering (numQp=1 and numQp=4 both still 184/384). The standalone-only
        # fused bandwidth win (+41% @64MiB) is not worth a wrong training loss,
        # so the shipped default is the serial direct path until the fused
        # kernel's ring-completion visibility to the finish readers is fixed
        # on-device. Opt back in with MORI_HIER_FUSE_LOCAL=1 for standalone A/B.
        # In the standalone benchmark (world=8 N=2 G=4, bit-exact),
        # MORI_HIER_FUSE_LOCAL=1 beats RCCL at every size >=32MiB in both
        # dtypes (fp32 1.148-1.295x, bf16 1.152-1.291x); the default serial
        # path (this OFF) is ~0.815/0.885x. So the standalone UT
        # requirement (ratio>=1.0x, bit-exact) is MET by this path; the ONLY reason
        # it is not the shipped default is the E2E copy-engine-finish stale-remote
        # race above. Note:
        # MORI_SHMEM_HEAP_TYPE=normal (cached ring buffer) is identical to
        # uncached => RDMA DMA is unaffected by the ring cache attribute. Also moot:
        # the "leader-only ring + XGMI broadcast" redundancy killer -- the default
        # slice path already sends 1 shard/NIC ((N-1)*count, the minimum), so no
        # inter-node redundancy remains for it to eliminate.
        # The serial slice_direct path (fuse_local OFF) hits a large-buffer floor
        # (~124 GB/s). Enabling fuse_local lifts the standalone w8 path to
        # 0.80-0.94 of RCCL (fp32/bf16, all sizes) bit-exact -- the floor is the
        # serial fan-out, not a fabric wall. fuse_local is not the global default
        # because of the E2E FSDP tight-overlap stale-remote race (~48% of AGs)
        # documented above; the standalone AllGather (no back-to-back FSDP
        # overlap) never triggers that race and is bit-exact with fuse_local. So a
        # standalone caller may opt into the fast fan-out via standalone_fast=True
        # without touching the E2E default: the env still overrides either way,
        # and any caller that omits the flag (all FSDP/E2E paths) keeps the
        # byte-identical serial default.
        if "MORI_HIER_FUSE_LOCAL" in os.environ:
            self.fuse_local = _env_true("MORI_HIER_FUSE_LOCAL")
        else:
            self.fuse_local = bool(standalone_fast)
        # Remember the standalone gate so the standalone fast path can auto-engage
        # its bit-exact-safe fill/overlap levers below without touching any
        # FSDP/E2E caller (none pass standalone_fast).
        self._standalone_fast = bool(standalone_fast)
        # Standalone fast-path fill. fuse_local (above) clears the serial floor but
        # tops out at ~0.86-0.94 of RCCL (a per-NIC fill shelf). The bit-exact-safe
        # lever is more per-peer SDMA fill: default MORI_SDMA_NUM_CHANNELS to 8 (vs
        # the library default 2), a deterministic reassembly widening.
        #
        # The full fuse_remote + deep_pipe + large max-bytes config is NOT bit-exact
        # on every fabric -- with an uncaged coherence window each temporal
        # sub-chunk can exceed the mlx5 NIC-DMA->HBM ~32MB coherence window and
        # expose a flag-beats-data race (the put-signal AMO landing before the WRITE
        # DMA is globally visible) at >=128MB. So the fill knob (channels) is the
        # only lever defaulted unconditionally; fuse_remote/deep_pipe are engaged
        # only within the coherence window (see below).
        #
        # The SDMA copy-channel count is hardware-capped at 8 on MI300X: requesting
        # 12 or 16 crashes at SDMA queue creation (per-GPU queue-slot exhaustion).
        # 8 is the fill ceiling on the bit-exact path, not a tunable. Other fill
        # directions regress: MORI_NUM_QP_PER_PE=8 adds per-QP quiet overhead with
        # no fill gain, and a 1MB RDMA put chunk is already bandwidth-bound. The >8
        # crash-guard lives at module import (see _clamp_sdma_channels) because the
        # physical SDMA queue count is fixed by anvil at shmem init, before this
        # __init__ runs.
        self._apply_standalone_fast_defaults(standalone_fast)
        self._init_fused_ring_state()
        # DIRECT-TO-OUTPUT Phase B. The default fused sliced
        # path SDMA-gathers the N node-blocks into an internal symmetric transit
        # (_intra.out_) and then a finish_batch copies the WHOLE output
        # (N*block = full AllGather result, ~512 MiB @64 MiB/rank) D2D into the
        # user output -- pure HBM traffic on the critical path. With slice_direct
        # the gathers PUSH each member's slice straight into the (registered) user
        # output, eliminating that copy entirely (the only remaining serial
        # Phase-B cost after the stream-ordered barriers). The user output is
        # registered once (collective ShmemSymmetricRegister, cached) on first
        # sight; the cost amortizes across calls that reuse the same output (the
        # benchmark + steady-state inference both do). Only engaged on the
        # default fused, non-overlap, non-pipe, non-oop, stream-ordered slice
        # path (the shipped path). kept OPT-IN (default OFF).
        # The direct path registers the USER output as a symmetric buffer
        # (ShmemSymmetricRegister); over RDMA (true xnode) this succeeds, but
        # under single-node IPC (hipIpcGetMemHandle on an arbitrary torch
        # allocation) it HARD-FAILS ("invalid argument") and aborts the process
        # -- so default-ON would crash single-process multi-GPU users. It is a
        # validated true-xnode lever (+5.4% @64 MiB, 133.7->141.2 GB/s); enable
        # with MORI_HIER_SLICE_DIRECT=1 / --slice-direct on a real RDMA setup.
        #
        # now that slice_direct is robustly correct under
        # varying output pointers ( exact-base + stale-evict fix) and the
        # teardown crash is fixed, promote it to DEFAULT ON whenever we
        # are on a true multi-node (RDMA) setup (num_nodes >= 2), where
        # ShmemSymmetricRegister succeeds. It stays OFF on single-node (num_nodes
        # == 1, IPC sim) where hipIpcGetMemHandle hard-aborts. An explicit
        # arg or MORI_HIER_SLICE_DIRECT env override still wins. The default
        # decision is deferred to after num_nodes is known (see below).
        if slice_direct is None:
            if "MORI_HIER_SLICE_DIRECT" not in os.environ:
                # Sentinel: decide from num_nodes after it is computed.
                slice_direct = None
            else:
                slice_direct = _env_true("MORI_HIER_SLICE_DIRECT")
        self.slice_direct = None if slice_direct is None else bool(slice_direct)
        if self.slice_inter and (
            self.leader_only or self.out_in_place or self.gather_in_place
        ):
            raise ValueError(
                "slice_inter is incompatible with leader_only/out_in_place/"
                "gather_in_place (it owns the inter+intra data path)"
            )
        if self.out_in_place:
            # out-in-place subsumes gather-in-place (no copy-IN either) and is
            # incompatible with the leader-only broadcast pipeline (its result is
            # produced by the SDMA broadcast, not the ring buffer).
            if self.leader_only:
                raise ValueError(
                    "out_in_place is incompatible with leader_only (the leader-only "
                    "result comes from the SDMA broadcast, not the ring buffer)"
                )
            self.gather_in_place = True
        # Interface compatibility with the flat AllgatherSdma: accept a single
        # combined transit size and split it into input/output when the caller
        # did not size them explicitly.
        if transit_buffer_size is not None:
            if input_buffer_size is None:
                input_buffer_size = transit_buffer_size
            if output_buffer_size is None:
                output_buffer_size = transit_buffer_size * npes
        # Topology is auto-detected so callers use the same signature as the
        # flat AllgatherSdma (no ranks_per_node needed). Single node -> the
        # operation degenerates to a pure intra-node SDMA AllGather.
        if ranks_per_node is None:
            ranks_per_node = _auto_ranks_per_node(my_pe, npes)
        if ranks_per_node < 1 or npes % ranks_per_node != 0:
            raise ValueError(
                f"npes ({npes}) must be a positive multiple of ranks_per_node "
                f"({ranks_per_node})"
            )
        # Fail-closed topology guard against the fixed hier-barrier slot layout
        # (see MORI_HIER_MAX_* above). Exceeding either bound would silently
        # corrupt the barrier's rendezvous slots on the device.
        if ranks_per_node > MORI_HIER_MAX_RANKS_PER_NODE:
            raise ValueError(
                f"ranks_per_node ({ranks_per_node}) exceeds the supported "
                f"maximum ({MORI_HIER_MAX_RANKS_PER_NODE}); the hier barrier's "
                f"local-PE inbox has only {MORI_HIER_MAX_RANKS_PER_NODE} slots "
                f"(see ShmemInternalBarrierHierBlock)."
            )
        _num_nodes = npes // ranks_per_node
        if _num_nodes > MORI_HIER_MAX_NUM_NODES:
            raise ValueError(
                f"num_nodes ({_num_nodes} = npes {npes} / ranks_per_node "
                f"{ranks_per_node}) exceeds the supported maximum "
                f"({MORI_HIER_MAX_NUM_NODES}); the hier barrier's coordinator "
                f"inter-node inbox has only {MORI_HIER_MAX_NUM_NODES} slots "
                f"(see ShmemInternalBarrierHierBlock)."
            )

        self.my_pe = my_pe
        self.npes = npes
        self.ranks_per_node = ranks_per_node
        self.num_nodes = npes // ranks_per_node
        self.node_id = my_pe // ranks_per_node
        self.local_rank = my_pe % ranks_per_node
        self._apply_dense_node_defaults()
        self.copy_output_to_user = copy_output_to_user
        self._init_sync_drain_state()

        # the deferred slice_direct default (None sentinel) is
        # resolved LATER, after the inter-node ring is built, by probing the
        # actual transport (shmem_ptr_p2p to a cross-node peer: 0 => RDMA =>
        # ShmemSymmetricRegister of the user output works => default ON; non-zero
        # => P2P/IPC, incl. the single-node spawn sim that fakes num_nodes>=2 over
        # IPC => keep OFF to avoid the hipIpcGetMemHandle hard-abort). num_nodes
        # alone is NOT a safe signal (the sim runs num_nodes>=2 over IPC).
        if self.num_nodes == 1 and self.slice_direct is None:
            # The single-node path never uses the direct Phase-B gather.
            self.slice_direct = False

        self._build_subcollectives(
            my_pe,
            npes,
            input_buffer_size,
            output_buffer_size,
            copy_output_to_user,
        )

    def _apply_standalone_fast_defaults(self, standalone_fast):
        """Standalone fast-path env defaults (extracted verbatim from __init__).

        Reads only the ``standalone_fast`` arg + os.environ; writes
        self._standalone_fast / self._standalone_defer_fin and (only when
        standalone_fast is set) the MORI_SDMA_NUM_CHANNELS / FUSE_REMOTE /
        DEEP_PIPE / DISSEM_BARRIER env defaults. No FSDP/E2E caller passes
        standalone_fast, so the shipped path is byte-identical."""
        self._standalone_fast = bool(standalone_fast)
        if standalone_fast:
            if os.environ.get("MORI_SDMA_NUM_CHANNELS") is None:
                os.environ["MORI_SDMA_NUM_CHANNELS"] = str(MORI_SDMA_CH_HW_MAX)
            # Bit-exact-safe fuse_remote reassembly overlap: engage FUSE_REMOTE +
            # DEEP_PIPE=auto so the fused ring||reassembly kernel runs with the
            # auto-quiet send-CQ landing fence. Do NOT set
            # MORI_HIER_DEEP_PIPE_MAXBYTES -- the default 32MB window is the
            # coherence cage that keeps it bit-exact; uncaging it re-exposes the
            # flag-beats-data race.
            #
            # This config is the measured optimum. DEEP_PIPE depth is inert on the
            # fill (auto/2/4 all land the same BW; a finer 8MB sub-chunk regresses
            # via write fragmentation), and a deeper device send-queue (WQE_DEPTH>1)
            # is neutral-to-negative on mlx5 -- per-QP RC write is already
            # bandwidth-bound at wqeDepth=1. The residual at large sizes is a fixed
            # per-op cost (3 landing-fence barriers + 3 kernel launches) that only
            # amortises toward parity at 256MB, not a per-NIC fill deficit: mori's
            # marginal per-NIC fill BW already beats RCCL, so time ~= F + bytes/B
            # with a nonzero fixed F that RCCL does not pay. Cutting F needs a
            # kernel fusion collapsing the 3 launches into 1 without losing the
            # async ring||reassembly overlap; graph-capture collapses launches but
            # serializes the pipe and regresses bulk (see the CUDA_GRAPH size-gate).
            if os.environ.get("MORI_HIER_FUSE_REMOTE") is None:
                os.environ["MORI_HIER_FUSE_REMOTE"] = "1"
            if os.environ.get("MORI_HIER_DEEP_PIPE") is None:
                os.environ["MORI_HIER_DEEP_PIPE"] = "auto"
            # Route the standalone path's cross-PE fences through the O(log n)
            # dissemination barrier. It has identical global all-PE rendezvous
            # semantics (bit-exact; byte image and NIC-landing->reassembly-consume
            # ordering unchanged) but a ceil(log2 n) parallel critical path instead
            # of the PE0 funnel's ~2(n-1) serial hops -- a strictly cheaper barrier
            # topology. The gain is within noise (the barrier is a small part of the
            # fixed cost) but it cannot regress. Default ON for the standalone_fast
            # path only; no FSDP/E2E caller passes standalone_fast, so the E2E paths
            # stay byte-identical (dissem default OFF there). Env overrides either
            # way.
            if os.environ.get("MORI_HIER_DISSEM_BARRIER") is None:
                os.environ["MORI_HIER_DISSEM_BARRIER"] = "1"
        # Standalone finish-barrier deferral: drop the per-op finish
        # ShmemBarrierOnStream, leaning on the successor op's entry barrier for
        # cross-PE ring reuse (see the finish site). Strictly cheaper topology,
        # identical semantics, bit-exact and non-regressing at every size, with the
        # largest gain at small sizes where the per-op fixed barrier cost dominates.
        # Default ON for the standalone_fast path only (same class as
        # slice_defer_fin/dissem_barrier). The w8 path already runs
        # slice_defer_fin=True so this is byte-identical there; only w16 (which
        # forces slice_defer_fin=False for the E2E drift guard) newly benefits. No
        # FSDP/E2E caller passes standalone_fast, so the w16 E2E finish fences stay
        # ON; the E2E host-drain reference loss is unchanged. Set
        # MORI_HIER_STANDALONE_DEFER_FIN=0 to restore the finish barrier.
        self._standalone_defer_fin = bool(standalone_fast) and _env_true(
            "MORI_HIER_STANDALONE_DEFER_FIN", "1"
        )

    def _init_fused_ring_state(self):
        """Init fused ring / persistent-kernel / flag-token / reasm-deep-SQ / host-proxy-inter state (moved verbatim from __init__)."""
        # PHASE 4: pipeline the inter-node RDMA ring with the REMOTE-block XGMI
        # reassembly (the 143->168 GB/s lever). When ON (and on the fuse_local
        # slice_direct path) the fused FusedRingRemoteGatherKernel runs the ring
        # AND, per landed sub-range, the remote-block SDMA push straight from this
        # PE's ring buffer into the registered output -- NO ring copy-OUT, NO
        # whole-phase finish barrier, remote gather overlaps the still-in-flight
        # NIC ring. Only valid at num_nodes==2 (single ring round); default OFF
        # until the standalone bit-exact sweep gate passes.
        self.fuse_remote = _env_true("MORI_HIER_FUSE_REMOTE", "0")
        # PERSISTENT-KERNEL PORT: fold the host hipMemcpyAsync
        # copy-IN of this PE's input into its ring slot INTO the fused kernel (each
        # ring channel stages its own send sub-range before the put). Drops one GPU
        # op per AG; combined with MORI_HIER_GEN_RING (no entry barrier) +
        # slice_defer_fin (deferred finish) the whole AG collapses to a SINGLE host
        # kernel launch -- the aggregate-collapse hypothesis for the fixed per-op
        # floor. Only on the fuse_remote path (the crown UT config). Default OFF
        # => the host copy-IN runs, byte-identical shipped path.
        # The single-launch aggregate-collapse shares the fatal tradeoff of the
        # HIP-graph launch-collapse (below): collapsing the per-op multi-launch to
        # one launch forfeits the CPU-driven async ring||reassembly overlap that
        # bulk BW depends on. Graph capture wins at small sizes but regresses bulk.
        # The three host-op levers that would let this path collapse -- GEN_RING (no
        # entry barrier), FLAG_TOKEN, fuse_copyin -- each regress or break: GEN_RING
        # is E2E-racy and makes graph capture silently fall back to eager;
        # FLAG_TOKEN regresses the eager >48MB path; the copy-IN fold is neutral on
        # SDMA and much slower if forced onto CUs (which also violates the CU
        # red-line). Single-launch and bulk ring||reassembly overlap are mutually
        # exclusive on this fabric, so the fixed per-op floor is irreducible on the
        # bit-exact path without forfeiting the overlap that gives mori its winning
        # marginal fill.
        self.fuse_copyin = _env_true("MORI_HIER_FUSE_COPYIN", "0")
        self._chunk_ready_flags = None
        # Cross-size carryover guard: the exact DEEP_PIPE flag layout
        # (slots, per-PE count, pipe depth) the persistent buffer was last sized
        # for. A layout CHANGE forces a fresh zeroed buffer so stale per-sub-chunk
        # landing state can't leak into the next distinct size (see the fuse_remote
        # DEEP_PIPE block below).
        self._chunk_ready_flags_layout = None
        # Gen-token chunkReadyFlags (MORI_HIER_FLAG_TOKEN): drop the per-op host
        # flags.zero_() hipMemset launch (part of the fixed per-op launch floor
        # that dominates the small-buffer gap) by publishing a strictly-increasing
        # per-op token into chunkReadyFlags and waiting `< opGen` in the reassembly
        # worker -- the same reset-free pattern the classic ring (opGen) and the
        # reassembly-completion flags (gFlagVal) already use. Default OFF
        # (op_gen=0 -> kernel writes 1 / waits <1 / host zeroes -> byte-identical).
        self._flag_token = _env_true("MORI_HIER_FLAG_TOKEN", "0")
        self._flag_opgen = 0
        # Device flag-token (MORI_HIER_FLAG_TOKEN_DEV, requires GEN_RING_DBL): like
        # FLAG_TOKEN but the per-op generation is derived device-side from the
        # graph-safe parity counter (HierFlagTokenDevOn), so it advances on every
        # HIP-graph replay -- the host FLAG_TOKEN counter freezes at capture. Here
        # we only skip the host flags.zero_() so the flags accumulate; the device
        # supplies the strictly-increasing token.
        self._flag_token_dev = _env_true("MORI_HIER_FLAG_TOKEN_DEV", "0")
        # Intra reassembly deep-SQ (MORI_HIER_REASM_DEEPSQ): submit all owned
        # reassembly channels' SDMA copies back-to-back (SQ continuously fed) then a
        # single drain covers them all plus deferred flags, instead of submit+drain
        # per channel. Bit-exact by construction (the landing wait stays in pass 0
        # before any submit; the output flag fires only after the pass-1 drain, so
        # it never precedes its bytes -- see FusedRemoteReassembleWorker). Default
        # ON only on the fused path (fuse_local/fuse_remote), where a single reasm
        # CTA processes the deep-pipe sub-chunks serially and the per-queue
        # drain-per-sub-chunk cap holds it below the plain default; feeding the SQ
        # continuously lifts the fused-path bandwidth. The plain (non-fused) N=2
        # path has a single remote reassembly gather, so this is a no-op there
        # (nPass collapses to the single-shot submit+drain) and the shipped default
        # path stays byte-identical. Env still overrides either way.
        if "MORI_HIER_REASM_DEEPSQ" in os.environ:
            self._reasm_deep_sq = 1 if _env_true("MORI_HIER_REASM_DEEPSQ") else 0
        else:
            self._reasm_deep_sq = 1 if (self.fuse_local or self.fuse_remote) else 0
        # HOST-PROXY INTER producer (MORI_HIER_HOSTPROXY_REASM): lazily built
        # against the inter ring buffer; owns the inter leg + publishes flags.
        self._hp_inter = None
        self._hp_src_ev = None

    def _init_sync_drain_state(self):
        # isolation probe: force full stream completion at op return.
        self._debug_sync = _env_true("MORI_HIER_DEBUG_SYNC", "0")
        # Dense-node device-landing drain (opt-in, default OFF -- a documented
        # negative kept as an escape hatch).
        # The bit-exact dense-node base needs the full per-op host drain
        # (MORI_HIER_DEBUG_SYNC=1 -> s.synchronize() on every AG) because at 8
        # ranks/node the residual RDMA remote-landing race is not confined to the
        # big embed/lm_head AGs -- every small per-layer AG also races. The host
        # synchronize is bit-exact but its CPU stall kills cross-op run-ahead.
        # DEVDRAIN attempts to replace the per-op host synchronize with the
        # on-device equivalent enqueued on the same comm stream (no CPU round-trip):
        # shmem_barrier_on_stream (cross-PE rendezvous ordering every peer's
        # remote-half RDMA writes plus the intra SDMA gather) then
        # launch_device_landing_gate (a live CQ-drain plus __threadfence_system
        # spinning the CQ until every posted inter-node WQE has physically landed in
        # HBM) -- the host drain's two jobs done device-side, with no CU payload
        # copy. It does NOT achieve bit-exactness: the on-device CQ drain plus
        # cross-PE barrier is insufficient, and only the host s.synchronize reaches
        # the reference loss. This mirrors the w8 finding that device transport
        # fences are not sufficient and only a host CPU RDMA-progress round-trip is
        # bit-exact. The real dense-node bandwidth lever is a deferred host drain
        # (hide the CPU stall behind FSDP prefetch), not a device-fence replacement.
        self._w16_devdrain = _env_true("MORI_HIER_W16_DEVDRAIN", "0")
        # SYNC_BIG: targeted remote-completion fence on ONLY the big cross-node
        # all-gathers. The residual
        # fast-path stale reads (~184/384 calls) concentrate in the LARGE
        # embed/lm_head cross-node AGs (per-rank bytes >> a regular layer); the
        # many small per-layer AGs converge fine. So instead of a full-op host
        # sync on EVERY call (DEBUG_SYNC, which forfeits all overlap), host-sync
        # ONLY when the per-rank payload is >= SYNC_BIG_BYTES. This closes the
        # remote-landing race exactly where it bites while keeping the ring<->
        # gather overlap on the numerous small AGs (perf-preserving convergence
        # fix). Default OFF; threshold 8 MiB/rank cleanly separates embed/lm_head
        # from the regular transformer-block params for Qwen-class models.
        self._sync_big = _env_true("MORI_HIER_SYNC_BIG", "0")
        self._sync_big_bytes = _env_int(
            "MORI_HIER_SYNC_BIG_BYTES", str(8 * 1024 * 1024)
        )
        # SYNC_BIG mode: "host" (default) = host stream.synchronize() on the big
        # AGs (proven bit-exact but stalls the CPU->GPU pipeline ~23%); "barrier"
        # = a DEVICE-side cross-PE ShmemBarrierOnStream enqueued on the SAME
        # stream right after the big AG. The barrier kernel quiesces every PE
        # (drains the RDMA send-queues => RC remote landing) and system-fences
        # before releasing, so the FSDP consumer (stream-ordered AFTER it) can
        # only read the AG output once every peer's remote-half bytes have
        # physically landed -- the same guarantee host-sync gives, but WITHOUT a
        # host stall (keeps the ring<->gather + AG<->backward overlap). Targets
        # ONLY the big embed/lm_head cross-node AGs where the residual
        # remote-landing race lives.
        self._sync_big_mode = os.environ.get("MORI_HIER_SYNC_BIG_MODE", "host")
        # SYNC_BIG mode "throttle": a BOUNDED CPU run-ahead throttle. The
        # residual fast-path loss drift is a
        # race that ONLY a host stream-drain masks -- yet it is numQp-independent
        # and every device-side transport fence failed, so it is NOT
        # an unlanded-RDMA race. That signature = the CPU running arbitrarily far
        # ahead of the GPU, so the big embed/lm_head AG's true completion drifts
        # relative to when its consumer is actually enqueued/observed. Full
        # SYNC_BIG (host-sync the CURRENT big AG) fixes it but stalls the whole
        # CPU->GPU pipeline (~24%). "throttle" instead records an event AFTER each
        # big AG and, on the NEXT big AG, host-waits on the PREVIOUS event. That
        # bounds CPU run-ahead to ~1 step while the CURRENT big AG still overlaps
        # with backward compute -- so if BUG B is pure run-ahead, this recovers
        # ground truth at far lower cost than SYNC_BIG. Default OFF (host mode).
        self._sync_big_prev_event = None
        # DEFERRED host-drain state (SYNC_BIG_MODE=deferhost): the completion
        # event of the last big AG, host-drained by the harness Work.wait() at
        # the consume point instead of at issue.
        self._deferred_drain_event = None
        self._deferred_drain_pending = False
        # DEFERBWD: the in-tree counterpart of the harness-only
        # MORI_FSDP_DEFER_HOSTSYNC. On this MI300X/mlx5 pair the only reliably
        # bit-exact landing fence is a host stream round-trip (device fences are
        # insufficient), and only the backward re-unshard of the big embed/lm_head
        # AG races its consumer GEMM (the forward big AGs do not). Rather than
        # host-draining that AG inline (which stalls CPU enqueue mid-step) or
        # per-AG in the harness (which records an event on every big AG, including
        # the forward ones), this mode records one completion event on the backward
        # big AG after its kernel and does not sync here -- the deferred consumer
        # boundary calls drain_deferbwd() at copy-out, so the host wait overlaps the
        # backward GEMM and is paid at most once per step. Landing guarantee
        # identical to an inline host wait (it completes only after the AG kernel
        # plus copy-engine/NIC work land), so bit-exact by construction. Default OFF
        # (mode!="deferbwd"); reached via MORI_HIER_SYNC_BIG_MODE=deferbwd.
        self._deferbwd_event = None
        # Auto-compose SLICE_PIPE_OVERLAP on the deferbwd correct path. Chunking
        # the giant embed/lm_head backward AG's inter ring and overlapping chunk
        # k's Phase-B SDMA reassembly with chunk k+1's inter RDMA ring collapses
        # the serial Phase-B tail. The optimal chunk count is pair-specific (fewer
        # chunks means less per-chunk barrier overhead), and a forward prefetch
        # depth >1 drifts, so it stays clamped to 1. K=2 is a reasonable default.
        # Bit-exact by construction (strided dst_slot_stride=count writes identical
        # bytes; the per-chunk landing fence is unchanged). Only fires in deferbwd
        # mode and only when the user has not pinned the flags (explicit env wins),
        # so non-deferbwd paths stay byte-identical.
        if self._sync_big_mode == "deferbwd":
            if os.environ.get("MORI_HIER_SLICE_PIPE") is None:
                self.slice_pipe = True
            if os.environ.get("MORI_HIER_SLICE_PIPE_OVERLAP") is None:
                self.slice_pipe_overlap = True
            if os.environ.get("MORI_HIER_SLICE_PIPE_CHUNKS") is None:
                self.slice_pipe_chunks = 2  # fabric-dependent; tune per node pair
            # The front-load split was measured neutral-to-worse and is moot here
            # (the fence, not the split, gates the pipe), so auto-compose keeps the
            # even split (slice_pipe_split=None); the cheap intra-node landing fence
            # (self.slice_pipe_fence, default "intra") is what recovers the overlap
            # a global barrier would eat. Env still overrides.
        # FSDP copy-out coherence fix (CU-domain copy-out). The nodirect
        # Phase-B copy-OUT is a copy-ENGINE hipMemcpyAsync (out_ -> output); the
        # FSDP consumer (backward GEMM) reads ``output`` from a COMPUTE UNIT. On
        # this GPU a copy-engine write is not made coherent with a later CU read
        # by HIP stream-ordering alone (proven: only a host stream.synchronize
        # gave loss==native, on-device barriers/system-scope flags did not) ->
        # occasional stale bytes -> loss drifts ~0.15% high, run-to-run jitter.
        # When set, the C++ finish copies into a persistent scratch and a torch
        # ELEMENTWISE (CU) kernel writes scratch -> output, so the producer of
        # the consumed buffer is a CU op (CU/L2-coherent with the GEMM) WITHOUT a
        # host stall (preserving the AG<->backward overlap). Default ON.
        self._py_cu_copyout = _env_true("MORI_HIER_PY_CU_COPYOUT", "1")
        self._cu_copyout_scratch = None

    def _apply_dense_node_defaults(self):
        # Dense-node (8 ranks/node) landing-fence fix.
        # At 8 ranks/node the shipped default drops the Phase-B entry barrier
        # (slice_fuse_ib) and defers the inter/intra finish fences, relying on the
        # ring's own deferred finish for cross-PE visibility. The Phase-B intra
        # SDMA gathers read peer ranks' ``collection`` (the inter-ring node-block
        # output) over XGMI; with 8 local ranks the SDMA read can observe a peer's
        # collection before that peer's ring finish is globally visible -- a
        # cross-PE visibility race that a host stream.synchronize cannot fix (it is
        # peer-side, not local completion). The race scales with local fan-out, so
        # w8 (4 ranks/node) is unaffected while w16 (8 ranks/node) drifts. Forcing
        # the two cross-PE finish fences ON restores bit-exactness.
        # Fix: at ranks_per_node >= 8 default the two cross-PE finish fences ON
        # (explicit env always wins). At ranks_per_node == 4 the gate never fires,
        # so the w8 path is byte-identical.
        #
        # The Phase-B entry barrier is redundant once the two finish fences run
        # inline (slice_defer_fin=0 restores the Phase-B finish ShmemBarrierAll
        # immediately before the m==0 gather -- see the slice_fuse_ib comment
        # above: two back-to-back global barriers with no remote op between make the
        # entry one redundant). It is dropped from the gate default (one fewer
        # global barrier/op, still bit-exact); set
        # MORI_HIER_PHASEB_ENTRY_BARRIER=1 to restore it.
        if self.ranks_per_node >= 8:
            if "MORI_HIER_SLICE_DEFER_FIN" not in os.environ:
                self.slice_defer_fin = False
            if "MORI_HIER_SLICE_DEFER_INTER_FIN" not in os.environ:
                self.slice_defer_inter_fin = False
            # Dense-node signal-pipe default.
            # At 8 ranks/node the inter-node ring's temporal DEEP_PIPE sub-chunks
            # land via the auto-engaged DEEP_PIPE_QUIET send-CQ drain fence (quiet
            # auto-on whenever deepPipe>1 && !deepPipeImm). That serial
            # per-sub-chunk QP quiet-drain is the exposed per-round completion
            # latency. The fused put-with-signal landing path (deepPipeQuiet=0)
            # instead rides the completion AMO on the same QP as its data (RC
            # in-order, so the flag never precedes its bytes) with no extra drain --
            # the copy-engine completion model reimplemented natively, which is
            # markedly faster. Bit-exact here because the 32MB DEEP_PIPE window gate
            # (_dp_sub_bytes>=32MB => depth 1) caps engagement to sub-chunk<32MB,
            # under the NIC->HBM coherence window where the put-signal AMO could
            # outrun its own data. Any AG whose sub-chunk would reach >=32MB (e.g.
            # the giant embed/lm_head E2E AG) is caged to depth 1 and never runs the
            # signal path. Explicit MORI_HIER_DEEP_PIPE_QUIET always wins;
            # ranks_per_node==4 never enters this gate.
            if "MORI_HIER_DEEP_PIPE_QUIET" not in os.environ:
                os.environ["MORI_HIER_DEEP_PIPE_QUIET"] = "0"
            # Deep-pipe depth axis. Once the quiet-drain is gone, the temporal pipe
            # depth (1 vs 2 vs 4) is neutral within noise -- bandwidth is
            # depth-invariant, with a flat steady fill deficit plus a small fixed
            # per-op cost (the ratio ramp with size is the fixed cost amortising,
            # not depth helping). DEEP_PIPE=1 is never worse and strictly simpler
            # (no put-signal AMO that can outrun data, so no coherence-window
            # dependency and fewer landing flags). MORI_HIER_W16_DP1=1 forces it on
            # the dense-node gate; default OFF keeps the depth-2 signal default
            # byte-identical.
            if (
                _env_true("MORI_HIER_W16_DP1", "0")
                and "MORI_HIER_DEEP_PIPE" not in os.environ
            ):
                os.environ["MORI_HIER_DEEP_PIPE"] = "1"

    def _build_subcollectives(
        self,
        my_pe,
        npes,
        input_buffer_size,
        output_buffer_size,
        copy_output_to_user,
    ):
        """Construct the intra/inter/bcast sub-collectives for this handle.

        Pure structural extraction of the sub-collective construction tail of
        ``__init__`` (single-node AllgatherSdma vs the multi-node intra-gather
        + inter-ring [+ leader-only broadcast] pipeline, plus the deferred
        slice_direct transport probe). Reads only the listed ctor args and the
        already-set ``self.*`` topology/flag attrs; writes only ``self.*``."""
        if self.num_nodes == 1:
            # M1: single node -> a plain intra-node SDMA AllGather over all
            # local ranks is exactly the full AllGather.
            from .collective import AllgatherSdma

            self._intra = AllgatherSdma(
                my_pe,
                npes,
                input_buffer_size=input_buffer_size,
                output_buffer_size=output_buffer_size,
                copy_output_to_user=copy_output_to_user,
            )
        else:
            # M2b: hierarchical pipeline. Every rank runs two sub-group
            # collectives and ends with the full rank-major output -- no
            # separate broadcast phase (the "every-rank direct" decomposition):
            #
            #   1. Intra-node SDMA gather over my node's G local ranks
            #      {node*G, ..., node*G+G-1} -> my node-block (G shards in
            #      local-rank order). DESIGN: intra-node == SDMA copy engines.
            #   2. Inter-node RDMA ring over my same-local-index peers across
            #      nodes {local, local+G, ..., local+(N-1)*G} -> all N
            #      node-blocks in node order = concat(shard[0..W-1]), the
            #      rank-major all_gather result. DESIGN: inter-node == RDMA.
            #
            # Because node n owns ranks [n*G, n*G+G) and the ring lays blocks
            # down in node order, the result is bit-exact vs
            # torch.distributed.all_gather_into_tensor.
            #
            # PERF NOTE (M4, ) -- the dominant remaining cost:
            # This "every-rank direct" decomposition is simple (no broadcast
            # phase) but it sends each node-block over the NIC G times. All G
            # local ranks hold the same node-block after phase 1 and each one
            # independently rings its same-local-index peer, so node n's block
            # crosses the NIC once per local rank -- Gx redundant inter-node
            # traffic. The ring is bandwidth-limited, not QP/warp-limited (raising
            # the QP count gives no gain), so the Gx redundancy is the bottleneck.
            # The alternative is a leader-only inter-node ring (local_rank==0 over
            # the node-leaders {0,G,2G,...}) into a symmetric staging buffer, then
            # an intra-node SDMA broadcast of the full N*G output to the G local
            # ranks: it cuts NIC traffic ~Gx (1 block/node instead of G) at the
            # cost of one extra XGMI hop.
            from .collective import (
                IntraNodeSubGroupAllgatherSdma,
                InterNodeRingAllgather,
            )

            G = self.ranks_per_node
            N = self.num_nodes
            # input_buffer_size is sized per-rank shard; the intra gather output
            # is the node-block (G shards), so the intra transit must hold G*.
            # output_buffer_size is the full N*G-shard output, which is exactly
            # what the inter-node ring buffer must hold.
            intra_bytes = (
                G * input_buffer_size
                if input_buffer_size is not None
                else 512 * 1024 * 1024
            )
            inter_bytes = (
                output_buffer_size
                if output_buffer_size is not None
                else 512 * 1024 * 1024
            )
            # M5: the fused sliced Phase B stacks all N reassembly gathers
            # into ONE transit, so it must hold the full N*G-shard output (== the
            # inter ring buffer size), not just a single G-shard node-block.
            if self.slice_inter and self.slice_fused:
                intra_bytes = max(intra_bytes, inter_bytes)
            # remember the inter ring buffer size for the host-proxy GDR
            # registration (MORI_HIER_HOSTPROXY_REASM).
            self._inter_ring_bytes = inter_bytes

            # Phase 1 (both paths): intra-node SDMA gather over my node's G ranks.
            self._intra = IntraNodeSubGroupAllgatherSdma(
                my_pe=my_pe,
                npes=npes,
                out_buffer_bytes=intra_bytes,
                group_size=G,
                group_pos=self.local_rank,
                pe_base=self.node_id * G,
                pe_stride=1,
            )

            if not self.leader_only:
                # Every-rank-direct (default): every rank rings its same-local-
                # index peers across nodes; no broadcast phase. Sends each node-
                # block over the NIC G times ( bottleneck) but is simple
                # and proven bit-exact since .
                self._inter = InterNodeRingAllgather(
                    my_pe=my_pe,
                    npes=npes,
                    ring_buffer_bytes=inter_bytes,
                    ring_size=N,
                    ring_pos=self.node_id,
                    pe_base=self.local_rank,
                    pe_stride=G,
                    num_qp=self.inter_num_qp,
                    num_blocks=self.inter_num_blocks,
                )
            else:
                # Leader-only (M4, DESIGN's primary design): only local_rank==0
                # (the node-leader) rings over the node-leaders {0,G,2G,...} into
                # a staging buffer, then SDMA-broadcasts the full N*G output to
                # its G local ranks over XGMI. Cuts inter-node NIC traffic ~G x
                # (1 node-block/node instead of G).
                #
                # Validated negative on true cross-node RDMA
                # (N=2 G=4, fp32 64MiB/rank,
                # both bit-exact): leader-only 29.8 GB/s vs
                # every-rank-direct 63.8 GB/s -> 2.1x SLOWER. Reason: these MI355X
                # nodes have ONE ionic NIC PER GPU (8/node). The "G x
                # redundant NIC traffic" framing is misleading -- the per-NIC
                # byte load is IDENTICAL for both designs (each ring member, leader
                # or not, pushes (N-1) chunks of G*count over ITS OWN NIC).
                # every-rank-direct runs G rings on G distinct NICs in parallel
                # (the extra bytes ride extra NICs, so per-NIC time is unchanged),
                # whereas leader-only funnels everything through the leader's
                # SINGLE NIC and then pays an extra serial XGMI broadcast hop ->
                # strictly worse. So the bottleneck on this topology is per-NIC
                # BW x NIC-count, not aggregate fabric bytes; leader-only helps
                # only on topologies with fewer NICs than GPUs/node. Kept opt-in
                # (default every-rank-direct) for those topologies; do NOT make it
                # the default here. The ~2.4x gap vs RCCL (63.8 vs ~152) is NOT
                # closed by leader-only.
                #
                # ShmemMalloc (handle ctor) and ShmemBarrierAll (prepare/finish)
                # are COLLECTIVE over ALL PEs, so every PE must construct a ring
                # handle (same ring_buffer_bytes -> symmetric) and call its
                # prepare/finish to keep the barriers balanced. Non-leaders use a
                # degenerate singleton ring (ringSize=1, no kernel launch) whose
                # only purpose is to participate in those two barriers; the real
                # ring runs only among leaders, which never target non-leader
                # buffers (nextPeer stays within {0,G,2G,...}).
                from .collective import IntraNodeSubGroupBroadcastSdma

                if self.local_rank == 0:
                    self._inter = InterNodeRingAllgather(
                        my_pe=my_pe,
                        npes=npes,
                        ring_buffer_bytes=inter_bytes,
                        ring_size=N,
                        ring_pos=self.node_id,
                        pe_base=0,
                        pe_stride=G,
                        num_qp=self.inter_num_qp,
                    )
                else:
                    self._inter = InterNodeRingAllgather(
                        my_pe=my_pe,
                        npes=npes,
                        ring_buffer_bytes=inter_bytes,
                        ring_size=1,
                        ring_pos=0,
                        pe_base=my_pe,
                        pe_stride=1,
                        num_qp=1,
                    )
                # Phase 3: SDMA broadcast root=local_rank 0 -> the G local ranks.
                self._bcast = IntraNodeSubGroupBroadcastSdma(
                    my_pe=my_pe,
                    npes=npes,
                    out_buffer_bytes=inter_bytes,
                    group_size=G,
                    group_pos=self.local_rank,
                    pe_base=self.node_id * G,
                    pe_stride=1,
                )
                self._ring_scratch = None
            self._node_block = None
            # M5: scratch for the sliced path -- holds this rank's
            # collection C_g = [slice_g(B_0)..slice_g(B_{N-1})] (N*count) gathered
            # by the inter ring before the N intra reassembly gathers.
            self._slice_scratch = None
            # dbufstream: DEDICATED double-buffered collection scratch for the big
            # backward AGs. The consecutive big embed/lm_head AGs alternate output
            # buffers; on the fast contiguous copy-out path they SHARE the single
            # _slice_scratch with a DEFERRED finish fence, so AG#2's ring copy-IN
            # can clobber _slice_scratch before AG#1's copy-OUT drains it (the
            # scratch-reuse staleness). Two dedicated buffers, alternated per big
            # AG, break that reuse without touching the fast per-op path.
            self._big_scratch = [None, None]
            self._big_scratch_parity = 0
            # Toggle set by the dbufstream wrapper so the slice scratch selection
            # picks _big_scratch[parity] for the one big AG instead of the shared.
            self._big_dbuf_active = False
            # M5: lazy side stream for the slice-overlap lever (c). The
            # local node-block gather runs here concurrently with the inter ring.
            self._overlap_stream = None
            # M4 (/32): guard for the fuse-barrier entry-barrier skip. The
            # intra-gather ENTRY barrier may be skipped only when the PRIOR op ran
            # to COMPLETION (through its inter-finish ShmemBarrierAll, which is what
            # guarantees every peer's out_ transit is free before the next gather).
            # A plain call counter is NOT sufficient ( review): if a prior
            # op raised mid-pipeline -- after the intra-gather dirtied out_ but
            # before the inter-finish barrier -- a counter would still be >0 and the
            # next op would wrongly skip the entry barrier with a dirty buffer. So
            # we track explicit clean-completion: set False at entry, True only
            # after a full successful op. First call (and any post-crash call)
            # therefore keeps the barrier. Steady-state behavior is identical to the
            # old counter (every op completes), so the happy path stays bit-exact.
            self._prev_op_completed = False
            # M5: which path the PREVIOUS op took (sliced vs non-sliced),
            # so the size-threshold dispatcher can force-keep the entry barrier on
            # a path switch (the fuse-barrier entry-skip assumes the prior op's
            # barriers freed the SAME buffers this path will reuse). None = no
            # prior op.
            self._last_use_slice = None
            # SINGLE-registration tracking for the DIRECT-TO-
            # OUTPUT Phase-B path (slice_direct). The output buffer must be
            # collectively registered (ShmemSymmetricRegister all-gathers peer
            # pointers + opens IPC handles), so the register/deregister decision
            # MUST be identical on every PE. The old guard
            # (``if not is_output_registered: register``) made a PER-RANK
            # decision that drove a COLLECTIVE: when torch's caching allocator
            # placed a new output so that its range overlapped a prior (freed)
            # registration DIFFERENTLY across ranks, the C++ overlap-eviction set
            # diverged -> mismatched #collective calls -> the peer-pointer
            # all-gather mis-aligned -> SDMA read the wrong peer's window (a
            # bit-exact failure, reproducible with a single large --numels). Fix:
            # track exactly ONE live registration here and, only on an EXACT
            # (ptr,size) change, deregister the old + register the new. Exact
            # same-size buffer reuse IS SPMD-consistent across ranks (the steady-
            # state bench reuses one output for thousands of ops, rock-stable),
            # so this decision is lockstep-uniform without any extra collective.
            self._direct_reg_ptr = None
            self._direct_reg_size = None
            # MULTI-ENTRY LRU registration cache. The single-entry
            # tracker above deregisters the old buffer on EVERY (ptr,size) change,
            # so two ALTERNATING output buffers (e.g. the embed-grad and
            # lm_head-grad big backward AGs, which use different unsharded param
            # buffers) each pay dereg(old)+reg(new) = TWO cross-node collectives
            # per call -- the dominant cost of the de-fused big-AG path
            # (serialfast/olapfast). Holding K>=2 registrations lets both stay
            # resident so steady state pays ZERO register collectives. Keyed by
            # exact (ptr,size); eviction is deterministic (oldest-first) and the
            # (ptr,size) sequence is identical on every PE (FSDP issues the same
            # AGs in the same order), so the register/deregister collectives stay
            # lockstep-uniform -- the SPMD invariant that motivated the single-
            # entry design is preserved. Exact-match hits only (no Python-side
            # overlap logic) so two resident entries are always distinct live
            # buffers. Insertion-ordered dict = the LRU. Cap via env (default 4;
            # 0/1 restores single-entry behavior). Value = size (for bookkeeping).
            self._reg_cache_cap = _env_int("MORI_HIER_REG_CACHE", "4")
            self._direct_reg_lru = {}

            # resolve the deferred slice_direct default by
            # PROBING the real transport to a cross-node ring peer. slice_direct
            # registers the user output via ShmemSymmetricRegister; that path is
            # only safe over RDMA (true xnode). The single-node spawn sim fakes
            # num_nodes>=2 but wires peers over IPC, where hipIpcGetMemHandle on
            # an arbitrary torch alloc HARD-ABORTS. shmem_ptr_p2p returns 0 for an
            # RDMA-connected peer (different physical node) and non-zero for a
            # P2P/IPC peer (same host) -- the exact RDMA-vs-IPC signal. We probe
            # the symmetric ring buffer (already allocated) against a cross-node
            # ring member. Only the every-rank-direct slice path supports direct;
            # leader_only keeps the copy-OUT default.
            if self.slice_direct is None:
                self.slice_direct = self._probe_rdma_transport()

    def _get_hostproxy_inter(self):
        """Lazily build the persistent host-proxy inter-node producer against
        this PE's inter ring buffer (MORI_HIER_HOSTPROXY_REASM)."""
        if self._hp_inter is None:
            from .hostproxy_inter import HostProxyInterProducer

            ring_ptr = self._inter._handle.buf_ptr()
            # ring buffer holds ring_size chunks; size it generously (the handle
            # was allocated with inter_bytes >= the full output). Use the full
            # allocated region so any chunk offset is registered.
            ring_bytes = self._inter_ring_bytes
            self._hp_inter = HostProxyInterProducer(
                my_pe=self.my_pe,
                npes=self.npes,
                ranks_per_node=self.ranks_per_node,
                ring_buf_ptr=ring_ptr,
                ring_buf_bytes=ring_bytes,
            )
        return self._hp_inter

    def drain_hostproxy(self):
        """Join any in-flight async host-proxy inter worker so ALL chunkReadyFlags
        for the AGs issued so far are published+landed. Called by the deferred FSDP
        consumer fence (async completion-ordering fix). No-op unless the
        async host-proxy producer is live."""
        hp = getattr(self, "_hp_inter", None)
        if hp is not None:
            return bool(hp.drain())
        return False

    def drain_deferbwd(self):
        """Host-wait on the pending backward big-AG landing event, if any.

        The committed-source counterpart of the harness deferred host fence:
        MORI_HIER_SYNC_BIG_MODE=deferbwd records a completion event on the
        backward big embed/lm_head AG (after its kernel) WITHOUT syncing inline.
        The deferred FSDP consumer boundary (copy-out wait) calls this so the
        required host landing round-trip overlaps the backward GEMM issued
        between the AG and copy-out, and is paid at most once per step. Consumes
        the event (one-shot). No-op unless a backward big AG is pending."""
        ev = self._deferbwd_event
        if ev is not None:
            ev.synchronize()
            self._deferbwd_event = None

    def _ensure_output_registered(self, output_data):
        """Register output_data for direct-to-output SDMA push, LRU-cached.

        Lockstep across PEs: the (ptr,size) sequence and eviction order are
        identical on every rank (FSDP issues the same AGs in order), so the
        register/deregister collectives stay uniform. Exact-match only. On a hit
        no collective is issued (steady state = free).
        """
        out_ptr = output_data.data_ptr()
        out_size = output_data.numel() * output_data.element_size()
        key = (out_ptr, out_size)
        cap = self._reg_cache_cap
        if cap <= 1:
            # single-entry behavior (original path)
            if key != (self._direct_reg_ptr, self._direct_reg_size):
                if self._direct_reg_ptr is not None:
                    self._intra.deregister_output_buffer_ptr(self._direct_reg_ptr)
                self._intra.register_output_buffer(output_data)
                self._direct_reg_ptr = out_ptr
                self._direct_reg_size = out_size
            return
        lru = self._direct_reg_lru
        if key in lru:
            # hit: refresh recency, no collective. Safe ONLY because the LRU
            # mirrors C++'s overlap-eviction (below), so a hit is guaranteed
            # still-registered on the C++ side.
            lru.pop(key)
            lru[key] = out_size
            return
        # miss. The C++ register_output_buffer evicts ANY prior registration
        # whose range overlaps [ptr,ptr+size) (torch's caching allocator carves a
        # fresh output inside a freed segment). We MUST drop those same entries
        # from the Python LRU here, or a later cache-hit would skip re-register
        # for a ptr C++ has already evicted -> find_exact fails / "exceeds output"
        # at the direct gather. Bookkeeping only: C++ does the actual (symmetric)
        # deregister inside register_output_buffer.
        for k in [
            k for k in lru if (k[0] < out_ptr + out_size) and (out_ptr < k[0] + k[1])
        ]:
            lru.pop(k)
        # capacity eviction (deterministic oldest-first, lockstep collective).
        while len(lru) >= cap:
            old_key = next(iter(lru))
            lru.pop(old_key)
            self._intra.deregister_output_buffer_ptr(old_key[0])
        self._intra.register_output_buffer(output_data)
        lru[key] = out_size

    def _probe_rdma_transport(self) -> bool:
        """Return True iff a cross-node ring peer is reached over RDMA (not IPC).

        Used to default slice_direct ON only where ShmemSymmetricRegister of the
        user output is safe. Conservative: any error or P2P/IPC peer -> False.
        """
        if self.leader_only or self.num_nodes < 2:
            return False
        try:
            from ..shmem import shmem_ptr_p2p

            # Every-rank-direct ring: members {local_rank + G*j}; pick a peer on a
            # different node (different ring_pos) so the connection is inter-node.
            G = self.ranks_per_node
            peer_pe = self.local_rank + G * ((self.node_id + 1) % self.num_nodes)
            buf_ptr = self._inter._handle.buf_ptr()
            p2p = shmem_ptr_p2p(buf_ptr, self.my_pe, peer_pe)
            # 0 => RDMA transport (different nodes) => direct-to-output is safe.
            return p2p == 0
        except Exception:
            return False

    def all_gather(self, tensor_list, tensor, stream=None) -> bool:
        """Traditional list-based AllGather (matches ``torch.distributed.all_gather``).

        Gathers ``tensor`` from every rank into ``tensor_list`` -- a list of
        ``npes`` tensors, each shaped like ``tensor``. Uses the same
        hierarchical intra-node SDMA / inter-node RDMA path as the contiguous
        ``__call__`` (``all_gather_into_tensor`` style); the gathered rank-major
        output is scattered into the list entries.
        """
        if len(tensor_list) != self.npes:
            raise ValueError(
                f"tensor_list must have npes={self.npes} entries, got {len(tensor_list)}"
            )
        count = tensor.numel()
        flat = torch.empty(count * self.npes, dtype=tensor.dtype, device=tensor.device)
        if not self.__call__(tensor, flat, count, stream):
            return False
        # The scatter copies read ``flat``, so make the gather visible first.
        if stream is not None and hasattr(stream, "synchronize"):
            stream.synchronize()
        else:
            torch.cuda.synchronize()
        view = (
            flat.view(self.npes, *tensor.shape)
            if tensor.dim() > 0
            else flat.view(self.npes)
        )
        for i in range(self.npes):
            tensor_list[i].copy_(view[i])
        return True

    def _cu_copyout_finish(self, output_data, total_count_elems, stream):
        """CU-domain copy-OUT for the nodirect Phase-B (root-cause fix).

        The Phase-B gathers stack the reassembled result into the intra transit
        ``out_`` via raw SDMA; the receiver ``__threadfence_system`` makes those
        bytes coherently visible to a CU read but NOT to the copy engine, whose
        ``hipMemcpyAsync(out_ -> output)`` read is unfenced against the SDMA
        writes -> occasional stale bytes -> loss drifts ~0.15% high with
        run-to-run jitter (only a host ``stream.synchronize`` masked it, killing
        overlap). Here we instead do the copy-OUT as a SINGLE torch ELEMENTWISE
        (CU) kernel: it reads ``out_`` as a tensor (fenced/coherent CU read) and
        writes ``output`` in the CU/L2 domain the consumer GEMM reads. No copy
        engine, no host stall -> deterministic AND overlap preserved. ``add`` by
        0 is bit-exact for bf16/fp16/fp32 (x rounds to itself) and int dtypes.

        Cross-PE ``out_`` reuse is still fenced by ``finish_direct_stream``'s
        ShmemBarrierOnStream (deferrable, same as the copy-engine path).
        """
        # View the internal transit as the output dtype, rank-major length.
        transit = self._intra.get_output_transit_buffer(
            dtype=output_data.dtype, device=output_data.device
        )[:total_count_elems]
        out_flat = output_data.view(-1)[:total_count_elems]
        if stream is not None:
            with torch.cuda.stream(stream):
                torch.add(transit, 0, out=out_flat)
        else:
            torch.add(transit, 0, out=out_flat)
        # Cross-PE reuse fence (no copy-OUT): reuse the direct-path stream fence.
        self._intra.finish_direct_stream(
            stream=stream, barrier=not self.slice_defer_fin
        )

    def _get_reasm_pool(self, device):
        # Lazily allocate the side-stream pool for MULTI-STREAM Phase-B
        # reassembly (MORI_HIER_REASM_STREAMS). Pool size = reasm_streams-1 side
        # streams (the main stream is the first "lane"). Cached on the instance.
        n_side = self.reasm_streams - 1
        if n_side < 1:
            return []
        if self._reasm_stream_pool is None or len(self._reasm_stream_pool) < n_side:
            self._reasm_stream_pool = [
                torch.cuda.Stream(device=device) for _ in range(n_side)
            ]
        return self._reasm_stream_pool[:n_side]

    def _multistream_gathers(self, gathers, main, device):
        # Run a list of Phase-B reassembly gathers concurrently across the
        # main stream + a side-stream pool. ``gathers`` is a list of zero-arg
        # callables, each issuing exactly one gather_kernel_direct on the stream it
        # is given (bound below). Ordering: the FIRST gather runs on ``main`` (it
        # may carry the entry barrier); every side stream waits on ``main`` before
        # its gathers, and ``main`` waits on every side stream afterwards, so the
        # caller's finish_direct_stream (on main) still follows ALL gathers.
        pool = self._get_reasm_pool(device)
        if not pool or len(gathers) <= 1:
            for g in gathers:
                g(main)
            return
        lanes = [main] + list(pool)
        # First gather (entry barrier, if any) on main.
        gathers[0](main)
        # Side lanes observe main's entry barrier before issuing their gathers.
        used_side = set()
        for i, g in enumerate(gathers[1:], start=1):
            lane = lanes[i % len(lanes)]
            if lane is not main and lane not in used_side:
                lane.wait_stream(main)
                used_side.add(lane)
            g(lane)
        # Merge every used side lane back into main before the completion fence.
        for lane in used_side:
            main.wait_stream(lane)

    def __call__(self, input_data, output_data, count: int, stream=None) -> bool:
        """Gather ``count`` elements/rank into ``output_data`` (rank-major).

        Thin wrapper over ``_call_impl`` that optionally forces full completion
        before returning. Set ``MORI_HIER_DEBUG_SYNC=1`` to host-block on the
        caller's stream at op return -- an isolation switch for the FSDP
        copy-out loss-nondeterminism probe: if forcing full completion makes the
        loss deterministic==native, the residual bug is an async completion
        fence (the op returns before the SDMA/ring work the recorded event is
        supposed to capture is actually visible); if it stays nondeterministic
        the bug is a genuine data/layout race in the kernel.
        """
        # DEFERBWD safety-net drain: the deferbwd mode records one host landing
        # event on the backward big AG and relies on the FSDP copy-out hook
        # (drain_deferbwd()) to host-wait on it before the consumer GEMM. If a
        # still-pending deferred event survives into the next AG call (hook absent,
        # or an unexpected AG issued before copy-out drained), that next AG could
        # reuse the same buffer before the prior big AG's remote bytes land.
        # Draining here before issuing the next op closes that window without
        # depending on an external drain caller. When the copy-out hook already
        # drained (the shipped path), _deferbwd_event is None so this is a no-op
        # with no host stall. Bit-exact by construction (same host wait, never
        # later than the next AG's buffer reuse).
        if self._deferbwd_event is not None:
            self.drain_deferbwd()
        # CROSS-STREAM lifetime guard: this AG may run on a dedicated comm
        # stream (FSDP2) while the INPUT was produced on -- and is freed/recycled
        # by reshard on -- the compute stream. The caching allocator only tracks
        # the input's original stream, so it can hand the input storage to a
        # later compute-stream op while this AG is still reading it on `stream`
        # -> the big embed/lm_head AGs read a partially-overwritten input and
        # diverge (reproduced by the rapid-fire XSTREAM+FREE_INPUT probe).
        # record_stream marks the buffers in-use on the AG stream so the
        # allocator defers reuse until this AG completes. Sync-free (no host
        # stall) -- the standard non-default-stream collective safety contract.
        if stream is not None and os.environ.get(
            "MORI_HIER_NO_RECORD_STREAM", "0"
        ) not in ("1", "true", "True"):
            if hasattr(input_data, "record_stream"):
                input_data.record_stream(stream)
            if hasattr(output_data, "record_stream"):
                output_data.record_stream(stream)
        # Launch-collapse via HIP graph replay.
        # The residual large-buffer gap is the fixed per-op HIP-launch ramp: even
        # the fuse_remote path issues several host launches per AG (flags memset ->
        # prepare_stream copy-IN memcpy -> fused ring+gather kernel -> finish
        # fence), vs a single resident kernel. Replaying the whole op sequence from
        # a captured HIP graph pays one launch. Bit-exact by construction (identical
        # kernels, identical order, same buffers) with copy still on the SDMA
        # engine. _graph_replay returns False on any capture failure, so it falls
        # back cleanly to the normal fused path.
        # The collapse only wins while the fixed launch ramp is a large fraction of
        # the op (small buffers). On bulk buffers the static graph replay serializes
        # the CPU-driven inter/intra multi-launch overlap (it pays one launch but
        # loses the async ring||reassembly pipelining), so cap graph engagement to
        # buffers at/below MORI_HIER_CUDA_GRAPH_MAX_MB (default 48MB); bulk sizes
        # fall through to the normal fused fast path. Default ON (gated <=48MB); set
        # MORI_HIER_CUDA_GRAPH=0 to force the non-captured path.
        # Capture-poison guard: MORI_HIER_HOSTPROXY_REASM drives the inter leg from
        # a persistent CPU proxy (host ibverbs post + dist.barrier + flag publish)
        # inside the op body. Those host ops cannot be captured, so a
        # torch.cuda.graph attempt raises a capture-invalidated error -- and unlike
        # a clean Python-op capture miss, the failed device-side capture poisons the
        # HIP context so every subsequent eager launch dies, crashing the process.
        # Skip the capture attempt entirely when a known host-op mode is active so
        # the path degrades to a clean eager fallback. Byte-identical on every
        # default/E2E path (none set HOSTPROXY_REASM); env still overrides.
        _hp_host_ops = _env_true("MORI_HIER_HOSTPROXY_REASM", "0")
        if (
            os.environ.get("MORI_HIER_CUDA_GRAPH", "1").strip().lower()
            in ("1", "true", "yes", "on")
            and not self._debug_sync
            and not _hp_host_ops
        ):
            _cg_max_mb = float(os.environ.get("MORI_HIER_CUDA_GRAPH_MAX_MB", "48"))
            # Path-aware gate widening. The 48MB cap exists because the multi-chunk
            # CPU-pipelined path (side-stream ring||reassembly) regresses under
            # static graph replay (the graph serializes the CPU-driven pipe). But
            # the standalone fused crown (standalone_fast + fuse_remote/fuse_local)
            # runs ring||reassembly inside one grid (device-side pipeline, no CPU
            # side-stream to serialize), so capture is lossless at bulk sizes: it is
            # never worse and marginally better (the fixed cost is dominated by the
            # device barrier executions and pipe fill/drain, not host-launch
            # dispatch). So on this single-grid crown only, drop the size cap; every
            # other path keeps the 48MB gate, and slice_pipe/overlap stay capped
            # (their CPU pipe does regress under replay). Bit-exact by construction
            # (identical kernels/order/buffers, copy on SDMA never CU). E2E
            # byte-identical (no E2E caller sets standalone_fast). Env MAX_MB
            # overrides.
            if (
                self._standalone_fast
                and (self.fuse_remote or self.fuse_local)
                and not (self.slice_pipe and self.slice_pipe_chunks > 1)
                and not self.slice_overlap
                and os.environ.get("MORI_HIER_CUDA_GRAPH_MAX_MB") is None
            ):
                _cg_max_mb = 0.0  # uncapped: single-grid crown is lossless-capturable
            _cg_bytes = int(count) * int(input_data.element_size())
            if _cg_max_mb <= 0 or _cg_bytes <= _cg_max_mb * 1024 * 1024:
                # Graph-vs-launch-reduction footgun guard. The small-buffer win is
                # entirely the HIP-graph launch-collapse. The per-op host
                # launch-reduction levers (GEN_RING, FLAG_TOKEN, NO_ENTRY_BARRIER)
                # each add a graph-incompatible host op inside the op body, so
                # capture silently fails (cache[key]=False -> eager forever) and the
                # win is lost -- they do not collapse the launch ramp, they destroy
                # the mechanism that already does (and FLAG_TOKEN additionally
                # regresses the eager >48MB path). So on the graph-eligible band
                # these flags are a net-negative footgun; force them off for the
                # captured op so a stray env cannot silently forfeit the graph win.
                # The flags remain live on the >gate eager path where the graph
                # never runs. Bit-exact: the shipped default has all three OFF.
                if self._flag_token:
                    self._flag_token = False
                    print(
                        "[hier_graph] MORI_HIER_FLAG_TOKEN disabled on the "
                        "graph-eligible (<=%.0fMB) band: it breaks capture and "
                        "collapses the launch-collapse win." % _cg_max_mb,
                        flush=True,
                    )
                if self._graph_replay(input_data, output_data, count, stream):
                    return True
        do_sync = self._debug_sync
        big_ag = False
        if self._sync_big and self.num_nodes > 1:
            # Only the big cross-node AGs get a targeted completion fence.
            if count * input_data.element_size() >= self._sync_big_bytes:
                big_ag = True
        # TARGETED RING/INTRA HOST-FINISH on the BACKWARD big AG only.
        # The residual fast-path loss drift is in the BACKWARD re-unshard of the
        # big embed/lm_head cross-node AG: a backward-only host stream.synchronize()
        # makes grads bit-exact; forward is exonerated. Every device-side transport
        # fence failed, so the repair mechanism is a HOST CPU RDMA-progress
        # round-trip (STREAM_RING=0 host finish, which does hipStreamSynchronize +
        # host ShmemBarrierAll on the ring/intra finish). Full stream.synchronize()
        # is bit-exact but -22% because it also blocks CPU enqueue of the whole
        # rest of the step. This mode runs ONLY the backward big AG through the
        # host-synced ring+intra finish (stream_ring/stream_intra off for that ONE
        # call) -- draining just that op's stream, not the caller's whole compute
        # stream -- while every other AG stays fully on-stream/overlapped. The
        # host-sync path always issues full global ShmemBarrierAll (prepare+finish),
        # so cross-PE buffer reuse stays correct across the mixed on-stream/host
        # ops. Gated to autograd backward (grad disabled).
        _ring_hostfin = (
            self._sync_big_mode == "ringhostfin"
            and big_ag
            and not self._debug_sync
            and not torch.is_grad_enabled()
        )
        _saved_sr = _saved_si = None
        if _ring_hostfin:
            _saved_sr, _saved_si = self.stream_ring, self.stream_intra
            self.stream_ring = False
            self.stream_intra = False
        # Device-side isolation of the concurrent fused ring||local-gather
        # kernel (one grid, NIC ring blocks || XGMI gather block). This mode
        # forces the SERIAL non-fused, non-overlap slice_direct path (explicit
        # per-block gathers + a global finish_direct_stream barrier) for the
        # backward big AG, fully ON-STREAM (stream_ring/intra stay True, NO host
        # sync), to isolate whether the concurrent grid is the race.
        # Gated to autograd backward big AG.
        _serial_big = (
            self._sync_big_mode == "serial"
            and big_ag
            and not self._debug_sync
            and not torch.is_grad_enabled()
        )
        # "serialfast" -- de-fuse the backward big AG WITHOUT flipping the
        # global stream flags. The concurrent FusedRingLocalGatherKernel
        # (ring||XGMI gather in one grid) is the corruptor: serializing that
        # ONE big backward AG onto the two-launch on-stream slice_direct direct-
        # gather path (per-block gather_kernel_direct + finish_direct_stream) gives
        # device-side, sync-free, bit-exact grads. serialfast flips
        # stream_intra/ring True (and fuse_local/overlap False) for
        # ONLY this one big backward AG, so it takes the bit-exact on-stream direct
        # path while every other AG (forward + small) keeps the shipped fast
        # stream_intra=0 baseline untouched. ~1-2 de-fused AGs/step -> the overlap
        # loss is bounded to those calls, not the whole step.
        _serialfast_big = (
            self._sync_big_mode == "serialfast"
            and big_ag
            and not self._debug_sync
            and not torch.is_grad_enabled()
        )
        # "olapfast" -- de-fuse ONLY the single-grid concurrency, keep the
        # side-stream ring<->local-gather OVERLAP. serialfast is bit-exact but
        # -30% (the two-launch SERIAL direct path forgoes overlap AND re-registers
        # the output buffer per alternating embed/lm_head call). The corruptor is
        # the CONCURRENT FusedRingLocalGatherKernel (ring||XGMI gather in ONE grid);
        # its predecessor slice_direct_overlap uses TWO SEPARATE launches (ring on
        # main || local-block gather on a side stream, merged by
        # main.wait_stream(side)) -- overlapped but NOT one concurrent grid, so this
        # path is bit-exact AND keeps the overlap.
        # Gated to the backward big AG only; every other AG stays on the shipped
        # fast stream_intra=0 baseline.
        _olapfast_big = (
            self._sync_big_mode == "olapfast"
            and big_ag
            and not self._debug_sync
            and not torch.is_grad_enabled()
        )
        # "olapfast2": olapfast (de-fuse the single concurrent grid, keep the
        # two-launch side-stream ring<->local-gather overlap) plus defer the
        # de-fused big AG's inter-ring finish fence. The direct-to-output cost here
        # is the per-op global inter-ring ShmemBarrier that the env forces
        # (SLICE_DEFER_INTER_FIN=0) on every big AG; the direct-overlap path also
        # ping-pongs a side stream, so the exposed global fence serializes harder.
        # Defer it (barrier=False) for only this de-fused big AG so the successor
        # op's prepare barrier covers ring-buffer reuse -- same safety class as
        # slice_defer_fin, which is already deferred on this path. Gated to the
        # autograd backward big AG.
        _olapfast2_big = (
            self._sync_big_mode == "olapfast2"
            and big_ag
            and not self._debug_sync
            and not torch.is_grad_enabled()
        )
        # "devgate" -- run the backward big AG
        # on the SAME sync-free on-stream olapfast overlap path (two-launch
        # side-stream ring<->local-gather, stream_ring/intra on) BUT append a
        # DEVICE landing gate kernel on the caller's stream after finish. The gate
        # (ShmemQuietThread<RDMA> live-drain of every mlx5 CQ/QP + threadfence_sys)
        # is the on-device equivalent of the host stream.synchronize that is the
        # ONLY bit-exact fence on this HW -- it waits the actual RDMA
        # hardware landing so the FSDP-recorded event coincides with it, WITHOUT
        # the host round-trip that costs olapfast/bwdbig ~-22%. Gated to backward
        # big AG only; every other AG unchanged.
        _devgate_big = (
            self._sync_big_mode == "devgate"
            and big_ag
            and not self._debug_sync
            and not torch.is_grad_enabled()
        )
        # "devtouch" -- COMPOSE the two best near-miss
        # fences, each of which closes a DIFFERENT half of the backward big-AG
        # residual and neither of which alone is bit-exact on this MI300X/mlx5:
        #   - devgate (device landing gate, ShmemQuietThread RDMA CQ live-drain +
        #     threadfence_system): the strongest TEMPORAL landing fence on-device
        #     -- waits the actual RDMA hardware landing.
        #   - coretouch (L2CoherentRetouch, volatile glc L2-bypass load + CU
        #     re-publish): closes the CACHE-COHERENCE half (SDMA-copy-engine-written
        #     output not L2-coherent to the consumer GEMM under FSDP buffer reuse).
        # Run the gate FIRST (drain RDMA landing so the bytes have physically
        # landed) THEN the coherent re-touch (pull the freshly-landed HBM into the
        # CU/L2 domain the GEMM reads). Both on the caller stream, no host stall.
        _devtouch_big = (
            self._sync_big_mode == "devtouch"
            and big_ag
            and not self._debug_sync
            and not torch.is_grad_enabled()
        )
        # "dbufstream" -- keep the big backward AG on the FAST contiguous
        # copy-out STRUCTURE (gather all node-blocks into a contiguous scratch,
        # then ONE bulk finish_batch_stream copy-OUT -- the 222-TFLOPS shipped
        # path's shape), NOT the ~21%-slower direct-to-output strided writes that
        # serialfast/olapfast take. To do that force slice_direct=False for this
        # ONE AG (routes it to the else copy-out branch at ~1922) while turning ON
        # the stream-ordered ring+copy-out (stream_ring/stream_intra=True), and
        # DE-FUSE (fuse_local/slice_direct_overlap=False, no concurrent grid). The
        # copy-out branch reuses the SHARED _slice_scratch with a deferred finish
        # fence, which is exactly the scratch-reuse staleness for consecutive
        # big AGs -- so give this AG a DEDICATED double-buffered scratch (see
        # _big_dbuf_active below). Gated to autograd backward big AG only.
        _dbufstream_big = (
            self._sync_big_mode == "dbufstream"
            and big_ag
            and not self._debug_sync
            and not torch.is_grad_enabled()
        )
        # "bwdbigff" -- backward big AG host-drained
        # (bit-exact, same as bwdbig) BUT the FORWARD big AG (forward [FP]
        # bit-exact, only backward [GFP] races)
        # takes the FAST fused-fill kernel (fuse_local) instead of the default
        # de-fused slice overlap. Forward carries ~2 big embed/lm_head AGs/step;
        # filling them faster (fuse_local ~0.9x vs slice ~0.71x on this fabric)
        # shaves the exposed forward comm while the backward host-drain keeps the
        # landing->consume ordering the racy backward re-unshard requires. Every
        # small AG + all backward stays on the correct path unchanged.
        _bwdbigff_fwd = (
            self._sync_big_mode == "bwdbigff"
            and big_ag
            and not self._debug_sync
            and torch.is_grad_enabled()
        )
        _saved_fl_ff = None
        if _bwdbigff_fwd:
            _saved_fl_ff = self.fuse_local
            self.fuse_local = True
        _saved_fl = _saved_sdo = None
        _saved_sr2 = _saved_si2 = None
        _saved_dif = None
        if _serial_big:
            _saved_fl, _saved_sdo = self.fuse_local, self.slice_direct_overlap
            self.fuse_local = False
            self.slice_direct_overlap = False
        if _serialfast_big:
            _saved_fl, _saved_sdo = self.fuse_local, self.slice_direct_overlap
            _saved_sr2, _saved_si2 = self.stream_ring, self.stream_intra
            # on-stream, sync-free: force the two-launch direct-gather serial path
            self.fuse_local = False
            self.slice_direct_overlap = False
            self.stream_ring = True
            self.stream_intra = True
        if _olapfast_big or _olapfast2_big or _devgate_big or _devtouch_big:
            _saved_fl, _saved_sdo = self.fuse_local, self.slice_direct_overlap
            _saved_sr2, _saved_si2 = self.stream_ring, self.stream_intra
            # on-stream, sync-free: two-launch side-stream OVERLAP direct path
            # (no single concurrent grid), keeps ring<->local-gather overlap.
            self.fuse_local = False
            self.slice_direct_overlap = True
            self.stream_ring = True
            self.stream_intra = True
        if _olapfast2_big:
            # defer the exposed global inter-ring finish fence for this big AG.
            _saved_dif = self.slice_defer_inter_fin
            self.slice_defer_inter_fin = True
        _saved_sd = None
        if _dbufstream_big:
            _saved_fl, _saved_sdo = self.fuse_local, self.slice_direct_overlap
            _saved_sr2, _saved_si2 = self.stream_ring, self.stream_intra
            _saved_sd = self.slice_direct
            # fast contiguous copy-out shape, de-fused, dedicated double buffer.
            self.fuse_local = False
            self.slice_direct_overlap = False
            self.stream_ring = True
            self.stream_intra = True
            self.slice_direct = False
            self._big_dbuf_active = True
            self._big_scratch_parity ^= 1
        ret = self._call_impl(input_data, output_data, count, stream)
        if _bwdbigff_fwd:
            self.fuse_local = _saved_fl_ff
        if _serial_big:
            self.fuse_local, self.slice_direct_overlap = _saved_fl, _saved_sdo
        if (
            _serialfast_big
            or _olapfast_big
            or _olapfast2_big
            or _devgate_big
            or _devtouch_big
        ):
            self.fuse_local, self.slice_direct_overlap = _saved_fl, _saved_sdo
            self.stream_ring, self.stream_intra = _saved_sr2, _saved_si2
        if _devgate_big:
            # Device landing gate on the caller's stream: drain every posted
            # inter-node RDMA WQE (mlx5 CQ) on-device before the FSDP event, the
            # sync-free equivalent of the host stream.synchronize.
            from .collective import launch_device_landing_gate
            from .collective import _stream_to_int

            launch_device_landing_gate(_stream_to_int(stream))
            return ret
        if _devtouch_big:
            # Compose: device RDMA landing gate (temporal landing fence) FIRST,
            # THEN the L2-coherent re-touch (cache-coherence fence). Both on the
            # caller stream (stream-ordered, no host stall). Targets the residual
            # each fence alone leaves: gate=Δ-0.0042, coretouch=Δ-0.0093.
            from .collective import (
                launch_device_landing_gate,
                launch_l2_coherent_retouch,
                _stream_to_int,
            )

            cs = (
                torch.cuda.current_stream(input_data.device)
                if stream is None
                else stream
            )
            si = _stream_to_int(cs)
            launch_device_landing_gate(si)
            n = count * self.ranks_per_node * self.num_nodes
            u32_count = (n * output_data.element_size() + 3) // 4
            launch_l2_coherent_retouch(output_data.data_ptr(), u32_count, si)
            return ret
        if _olapfast2_big:
            self.slice_defer_inter_fin = _saved_dif
        if _dbufstream_big:
            self.fuse_local, self.slice_direct_overlap = _saved_fl, _saved_sdo
            self.stream_ring, self.stream_intra = _saved_sr2, _saved_si2
            self.slice_direct = _saved_sd
            self._big_dbuf_active = False
        if _ring_hostfin:
            self.stream_ring, self.stream_intra = _saved_sr, _saved_si
            return ret
        # ALL-COHERENT (world=16): at 8-GPU/node the intra-node gather
        # spans G=8 local peers via the SDMA COPY ENGINE -- a separate hw agent whose
        # writes a per-call stream.synchronize (DEBUG_SYNC) does NOT bring into CU/L2
        # coherence with the consumer GEMM. big_ag only fences the big embed/lm_head
        # cross-node AGs (>= sync_big_bytes); every SMALL per-layer AG runs UNFENCED.
        # At w8 (G=4) that was bit-exact; at w16 (G=8) the wider intra gather leaves
        # the small-AG output partially-landed/stale -> E2E loss drift under BOTH the
        # host-drain discriminator and K=1 (T9: 11.119320 / 11.124185 vs GT
        # 11.0912446975708). Fix: apply the PROVEN big-AG fence (barriercutouch) to
        # EVERY cross-node AG -- shmem_barrier_on_stream orders the RDMA cross-PE
        # landing + the intra SDMA gather, THEN an L2-coherent CU re-touch makes the
        # consumed buffer's LAST writer a CU op (CU/L2-coherent with the GEMM). Both
        # on the caller stream => stream-ordered, NO host stall (keeps overlap). The
        # re-touch (add-by-0 / coherent rewrite) is bit-exact for fp16/bf16/fp32/int.
        # Default OFF; when set it supersedes the big_ag-only fence for all sizes.
        _all_coherent = (
            self.num_nodes > 1
            and not self._debug_sync
            and _env_true("MORI_HIER_ALL_COHERENT", "0")
        )
        if _all_coherent:
            from ..shmem import shmem_barrier_on_stream
            from .collective import launch_l2_coherent_retouch, _stream_to_int

            cs = (
                torch.cuda.current_stream(input_data.device)
                if stream is None
                else stream
            )
            shmem_barrier_on_stream(cs)
            n = count * self.ranks_per_node * self.num_nodes
            u32_count = (n * output_data.element_size() + 3) // 4
            launch_l2_coherent_retouch(
                output_data.data_ptr(), u32_count, _stream_to_int(cs)
            )
            return ret
        # W16 DEVICE-LANDING DRAIN: device-side equivalent of the
        # per-op host synchronize for EVERY cross-node AG -- cross-PE rendezvous +
        # on-device mlx5 CQ drain, no host stall, no CU copy. Supersedes big_ag /
        # host-drain when set. See __init__ note.
        if self._w16_devdrain and self.num_nodes > 1:
            from ..shmem import shmem_barrier_on_stream
            from .collective import launch_device_landing_gate, _stream_to_int

            cs = (
                torch.cuda.current_stream(input_data.device)
                if stream is None
                else stream
            )
            shmem_barrier_on_stream(cs)  # rendezvous: order peer RDMA + SDMA gather
            launch_device_landing_gate(
                _stream_to_int(cs)
            )  # drain mlx5 CQ (bytes landed)
            return ret
        # DEBUG_SYNC always host-syncs; SYNC_BIG (barrier mode) prefers a
        # device-side cross-PE barrier on the big AGs (no host stall).
        if big_ag and not self._debug_sync and self._sync_big_mode == "coretouch":
            # barrier (orders RDMA cross-PE landing + intra SDMA
            # gather) FIRST, THEN an L2-COHERENT re-touch of the consumed output.
            # This is barriercutouch's compose EXCEPT the re-touch is the
            # system-scope acquire/release L2CoherentRetouchKernel_u32 (bypasses the
            # stale L2 line that barriercutouch's torch.add(out,0) read through) --
            # the direct fix for the residual dL 0.0023 barriercutouch left. Both
            # ops on the caller stream => stream-ordered, sync-free (keeps overlap).
            from ..shmem import shmem_barrier_on_stream
            from .collective import launch_l2_coherent_retouch, _stream_to_int

            cs = (
                torch.cuda.current_stream(input_data.device)
                if stream is None
                else stream
            )
            shmem_barrier_on_stream(cs)
            n = count * self.ranks_per_node * self.num_nodes
            u32_count = (n * output_data.element_size() + 3) // 4
            launch_l2_coherent_retouch(
                output_data.data_ptr(), u32_count, _stream_to_int(cs)
            )
        elif (
            big_ag and not self._debug_sync and self._sync_big_mode == "barriercutouch"
        ):
            # On mlx5: the device barrier and the CU
            # re-touch each close a DIFFERENT half of the big-AG completion race
            # and neither alone is bit-exact on this MI300X/mlx5 pair:
            #  - "barrier" alone: cut the olapfast drift ~24x (Δ0.054 -> Δ0.0023)
            #    at ~139 TFLOPS. The RDMA quiet+rendezvous orders the NIC remote
            #    landing, but the slice_direct direct-to-output write is done by
            #    the intra SDMA COPY ENGINE (a separate hw agent) whose bytes are
            #    not guaranteed CU/L2-coherent to the consumer GEMM just by the
            #    barrier -> residual Δ0.0023.
            #  - "cutouch" alone: makes the consumed buffer's LAST writer a CU op
            #    (CU/L2-coherent with the GEMM) but does NOT order the cross-node
            #    RDMA landing -> a CU re-touch can read still-unlanded remote bytes.
            # Compose them: barrier FIRST (quiet the NIC + cross-PE rendezvous so
            # every peer's remote-half bytes have physically landed and the intra
            # SDMA gather has run), THEN a CU re-touch (fenced CU read+rewrite of
            # the landed bytes into the CU/L2 domain the GEMM reads). Both enqueued
            # on the SAME caller stream => stream-ordered (barrier kernel completes
            # before the re-touch reads), sync-free (no host stall, keeps overlap).
            # add-by-0 is bit-exact for bf16/fp16/fp32/int. Targets ONLY the big
            # embed/lm_head cross-node AGs where the residual race lives.
            from ..shmem import shmem_barrier_on_stream

            cs = (
                torch.cuda.current_stream(input_data.device)
                if stream is None
                else stream
            )
            shmem_barrier_on_stream(cs)
            n = count * self.ranks_per_node * self.num_nodes
            out_flat = output_data.view(-1)[:n]
            with torch.cuda.stream(cs):
                torch.add(out_flat, 0, out=out_flat)
        elif big_ag and not self._debug_sync and self._sync_big_mode == "devcutouch":
            # THREE fences, each closing a distinct part
            # of the backward big-AG residual that no single/double compose closed:
            #   1. shmem_barrier_on_stream -- CROSS-PE RENDEZVOUS: the device gate
            #      alone drains only THIS PE's outgoing send CQ (proves my writes
            #      LEFT), it has NO guarantee the NEIGHBOR's RDMA writes have LANDED
            #      in my output. The barrier is the cross-PE order that guarantees
            #      every peer's remote-half bytes are in flight/ordered.
            #   2. device landing gate (ShmemQuietThread RDMA CQ live-drain +
            #      threadfence_system) -- TEMPORAL HW LANDING: spins the mlx5 CQ
            #      until every posted WQE completed, so bytes have PHYSICALLY landed
            #      at HBM (barrier orders but does not drain the hw send queues).
            #   3. torch.add(out,0) CU re-touch -- L2 REPUBLISH: makes the consumed
            #      buffer's LAST writer a CU op (CU/L2-coherent with the GEMM). The
            #      plain add-by-0 (barriercutouch's Δ0.0023 mechanism) beat the
            #      volatile-glc L2CoherentRetouch (coretouch Δ-0.0093), so use it.
            # barriercutouch (barrier+retouch) reached Δ0.0023 and devgate
            # (gate) reached Δ-0.0042; each leaves a DIFFERENT residual, so drain
            # the hw landing (gate) BETWEEN the rendezvous (barrier) and the CU
            # republish (retouch). All on the caller stream, no host stall.
            from ..shmem import shmem_barrier_on_stream
            from .collective import launch_device_landing_gate, _stream_to_int

            cs = (
                torch.cuda.current_stream(input_data.device)
                if stream is None
                else stream
            )
            shmem_barrier_on_stream(cs)
            launch_device_landing_gate(_stream_to_int(cs))
            n = count * self.ranks_per_node * self.num_nodes
            out_flat = output_data.view(-1)[:n]
            with torch.cuda.stream(cs):
                torch.add(out_flat, 0, out=out_flat)
        elif (
            big_ag
            and not self._debug_sync
            and self._sync_big_mode in ("gateinv", "bargateinv")
        ):
            # The CONSUME-SIDE LANDING GATE = device WAIT-on-landing THEN L2 INVALIDATE,
            # gating the big embed/lm_head AG consumer. The host-drain audit showed
            # only a HOST CQ-drain closes the fuse_remote consumer race and NO device
            # fence alone does; L2INV ALONE is the
            # WRONG HALF (it invalidates an UN-landed line so the GEMM re-fetches STALE
            # HBM => drift). The missing half is the WAIT. Compose them on-device, no
            # host stall:
            #   1. (bargateinv only) shmem_barrier_on_stream -- cross-PE rendezvous so
            #      every peer's remote-half RDMA writes are ordered/in-flight.
            #   2. device landing gate (ShmemQuietThread<RDMA> live CQ-drain +
            #      __threadfence_system) -- the WAIT: spins the mlx5 CQ until every
            #      posted WQE has COMPLETED, i.e. every inter-node write has physically
            #      LANDED in HBM. This is the device equivalent of the host CQ-drain
            #      that is the ONLY bit-exact mechanism -- but with no CPU
            #      round-trip. (devgate alone lands but the GEMM
            #      still reads a stale L2 line for the reused FSDP output buffer.)
            #   3. L2InvOnly buffer_inv -- the INVALIDATE: drops the stale device-L2
            #      lines AFTER the landing wait, so the stream-ordered consumer GEMM
            #      MISSES L2 and re-fetches the freshly-LANDED HBM bytes. Because it is
            #      stream-ordered strictly behind step 2 it invalidates a LANDED line
            #      NOT an un-landed one (plain L2INV_ONLY's bug).
            # All three enqueued on the SAME caller stream => stream-ordered, sync-free.
            from .collective import (
                launch_device_landing_gate,
                launch_l2_inv_only,
                _stream_to_int,
            )

            cs = (
                torch.cuda.current_stream(input_data.device)
                if stream is None
                else stream
            )
            si = _stream_to_int(cs)
            if self._sync_big_mode == "bargateinv":
                from ..shmem import shmem_barrier_on_stream

                shmem_barrier_on_stream(cs)
            launch_device_landing_gate(si)  # WAIT: device CQ-drain (bytes landed)
            launch_l2_inv_only(si)  # INVALIDATE: buffer_inv (drop stale L2)
        elif big_ag and not self._debug_sync and self._sync_big_mode == "barrier":
            from ..shmem import shmem_barrier_on_stream

            bs = (
                torch.cuda.current_stream(input_data.device)
                if stream is None
                else stream
            )
            shmem_barrier_on_stream(bs)
        elif (
            big_ag and not self._debug_sync and self._sync_big_mode == "barrierthrottle"
        ):
            # On mlx5: device barrier (NIC-landing +
            # cross-PE order) COMPOSED WITH a bounded CPU run-ahead throttle.
            # RATIONALE: on this MI300X/mlx5 pair "barrier" alone cuts the
            # olapfast completion-race drift ~24x (Δ0.054 -> Δ0.0023) at fast-path
            # speed (139 TFLOPS), but leaves a SIGN-FLIPPING residual (the
            # threshold-8MB run gives +0.0023, threshold-1MB gives -0.018) --
            # i.e. a NON-DETERMINISTIC residual that ACCUMULATES across steps into
            # the last_loss drift. Composing the CU re-touch (barriercutouch) made
            # it WORSE (the re-touch re-reads the window). Instead bound the CPU
            # run-ahead: after the device barrier orders THIS big AG's landing,
            # host-wait on the PREVIOUS big AG's completion event (bounds the CPU
            # to ~1 big-AG ahead of the GPU) and record this AG's event for the
            # next call. This does NOT re-read the current AG (no reopened window,
            # unlike barriercutouch) -- it only prevents the tiny per-AG residual
            # from compounding step-over-step, at far less cost than a full
            # host-drain of the current AG (which is the 112-TFLOPS bit-exact
            # floor). If the last_loss drift is an ACCUMULATION of the barrier's
            # damped residual, this lands bit-exact well above 114 TFLOPS.
            from ..shmem import shmem_barrier_on_stream

            bts = (
                torch.cuda.current_stream(input_data.device)
                if stream is None
                else stream
            )
            shmem_barrier_on_stream(bts)
            if self._sync_big_prev_event is not None:
                self._sync_big_prev_event.synchronize()
            ev = torch.cuda.Event()
            ev.record(bts)
            self._sync_big_prev_event = ev
        elif big_ag and not self._debug_sync and self._sync_big_mode == "cutouch":
            # CU re-touch on the big AGs: the slice_direct direct-to-output path
            # writes ``output`` with the intra SDMA gather (copy-engine domain)
            # and has NO CU re-touch (unlike the nodirect copy-out path, which
            # already routes through _cu_copyout_finish). The
            # residual stale-read race is exactly these big
            # embed/lm_head cross-node AGs. Force the LAST writer of the consumed
            # buffer to be a CU op on the CALLER stream via an in-place
            # elementwise re-touch: it does a fenced/coherent CU READ of the
            # SDMA-written bytes and re-writes them in the CU/L2 domain the
            # consumer GEMM reads, stream-ordered so FSDP's recorded event
            # captures it -- the same coherence _cu_copyout_finish gives the
            # copy-out path, but WITHOUT a host stall (keeps overlap). add-by-0
            # is bit-exact for bf16/fp16/fp32 (x rounds to itself) and ints.
            n = count * self.ranks_per_node * self.num_nodes
            out_flat = output_data.view(-1)[:n]
            cs = (
                torch.cuda.current_stream(input_data.device)
                if stream is None
                else stream
            )
            with torch.cuda.stream(cs):
                torch.add(out_flat, 0, out=out_flat)
        elif (
            big_ag
            and not self._debug_sync
            and self._sync_big_mode in ("bwdbig", "bwdbigff")
        ):
            # Selective host-drain, TIGHTER than SYNC_BIG=host. Ground-truth
            # bisection (fwd [FP] bit-exact, bwd [GFP] diverges) exonerates the
            # FORWARD big embed/lm_head AGs and every small per-layer AG -- only
            # the BACKWARD re-unshard of the big AG races the consumer GEMM. So
            # host-drain ONLY that call (FSDP runs the backward re-unshard with
            # grad DISABLED => not is_grad_enabled()), and let the forward big AG
            # + all small AGs keep the fast overlap. Same landing->consume
            # ordering the backward AG needs, half the big-AG host stalls of
            # SYNC_BIG=host (drops the ~2 exonerated forward big-AG syncs/step).
            # bwdbigff additionally fast-fills the forward big AG via fuse_local
            # (toggled above); the backward host-drain is identical.
            if not torch.is_grad_enabled():
                s = (
                    torch.cuda.current_stream(input_data.device)
                    if stream is None
                    else stream
                )
                s.synchronize()
        elif big_ag and not self._debug_sync and self._sync_big_mode == "throttle":
            # Bounded CPU run-ahead throttle: host-wait on the PREVIOUS big AG's
            # completion event (bounds run-ahead to ~1 step), then record this
            # big AG's completion for the next call. The current big AG still
            # overlaps with compute -- only the CPU is prevented from getting
            # more than one big-AG ahead of the GPU.
            s = (
                torch.cuda.current_stream(input_data.device)
                if stream is None
                else stream
            )
            if self._sync_big_prev_event is not None:
                self._sync_big_prev_event.synchronize()
            ev = torch.cuda.Event()
            ev.record(s)
            self._sync_big_prev_event = ev
        elif big_ag and not self._debug_sync and self._sync_big_mode == "deferhost":
            # DEFERRED host-drain. The
            # audit showed the ONLY E2E-bit-exact big-AG completion fence
            # on this MI300X/mlx5 pair is a HOST CQ-drain (stream.synchronize);
            # every device landing gate drifts. But draining AT ISSUE (SYNC_BIG=
            # host, line below) stalls the CPU->GPU pipeline the moment the big
            # embed/lm_head AG is enqueued -> the 0.71-0.73x floor. FSDP2 PREFETCHES
            # the big AG far ahead of its consumer GEMM (it issues the all_gather on
            # the comm stream, runs intervening compute, then calls Work.wait right
            # before the copy-out+consume). So DEFER the one fence that works: record
            # this big AG's completion event here (NO host stall at issue) and let the
            # harness _HierWork.wait() host-drain that event at the natural consume
            # point. Bit-exact BY CONSTRUCTION (the drain still completes strictly
            # before the consumer reads the output), but the CPU stall is hidden
            # behind the FSDP prefetch distance -> lifts the proven bit-exact base
            # above the 0.73x at-issue floor. The event is stashed for the harness;
            # if no harness Work waits on it, the next big AG's at-issue path (or a
            # final drain) still orders it, so correctness never depends on the Work.
            s = (
                torch.cuda.current_stream(input_data.device)
                if stream is None
                else stream
            )
            ev = torch.cuda.Event()
            ev.record(s)
            self._deferred_drain_event = ev
            self._deferred_drain_pending = True
        elif big_ag and not self._debug_sync and self._sync_big_mode == "deferbwd":
            # DEFERBWD (committed-source deferhost landing fix): on the BACKWARD
            # big AG (grad disabled), record ONE completion event AFTER the kernel
            # and DO NOT sync here. The deferred consumer boundary drains it via
            # drain_deferbwd() at copy-out, so the required host round-trip
            # overlaps the backward GEMM and is paid at most once per step. The
            # forward big AGs (exonerated) and all small AGs stay fully
            # overlapped -- no event, no drain. Bit-exact by construction: the
            # consumer host-waits on an event that completes only after this AG's
            # kernel + copy-engine/NIC work land (same guarantee as hostbwd's
            # inline s.synchronize(), just deferred + overlapped).
            if not torch.is_grad_enabled():
                s = (
                    torch.cuda.current_stream(input_data.device)
                    if stream is None
                    else stream
                )
                ev = torch.cuda.Event()
                ev.record(s)
                self._deferbwd_event = ev
        elif big_ag and not self._debug_sync and self._sync_big_mode == "hostbwd":
            # host-sync ONLY the BACKWARD big AGs. The residual landing->consume
            # race lives on the
            # BACKWARD re-unshard of the big embed/lm_head cross-node AG: a
            # backward-only host stream.synchronize() gave BIT-EXACT grads while
            # forward was exonerated. SYNC_BIG mode=host host-syncs EVERY big AG
            # incl. forward, paying a host stall on forward big AGs that don't
            # need it. This mode drains only when grad is disabled (autograd
            # backward), so forward big AGs stay overlapped -- same determinism if
            # the forward exoneration holds, strictly less host stall.
            if not torch.is_grad_enabled():
                s = (
                    torch.cuda.current_stream(input_data.device)
                    if stream is None
                    else stream
                )
                s.synchronize()
        elif do_sync or big_ag:
            s = (
                torch.cuda.current_stream(input_data.device)
                if stream is None
                else stream
            )
            s.synchronize()
        return ret

    def _graph_replay(self, input_data, output_data, count, stream) -> bool:
        """Launch-collapse: capture the op once into a HIP graph, then replay.

        Returns True if the op was serviced here (via warm/capture or replay),
        False to fall through to the normal eager path (capture was impossible
        for this buffer key -- e.g. a host barrier is still inside the op).
        Keyed by (in_ptr, out_ptr, count, dtype): FSDP reuses the same param
        buffers step-to-step and the UT reruns one size, so steady state is a
        pure replay -- ONE launch instead of the multi-launch ramp.
        """
        cache = getattr(self, "_graph_cache", None)
        if cache is None:
            cache = {}
            self._graph_cache = cache
        key = (
            input_data.data_ptr(),
            output_data.data_ptr(),
            int(count),
            str(input_data.dtype),
        )
        entry = cache.get(key)
        cs = torch.cuda.current_stream(input_data.device) if stream is None else stream
        if entry is None:
            # Warm the eager op so registration / scratch alloc / completion
            # state fully settle, THEN capture one replayable graph. The warm
            # runs also produce a correct output, so the caller's first (pre-
            # timing) bit-exact check passes even if capture later fails.
            for _ in range(3):
                self._call_impl(input_data, output_data, count, cs)
            cs.synchronize()
            try:
                graph = torch.cuda.CUDAGraph()
                cap = torch.cuda.Stream(device=input_data.device)
                cap.wait_stream(cs)
                with torch.cuda.graph(graph, stream=cap):
                    self._call_impl(input_data, output_data, count, cap)
                cs.wait_stream(cap)
            except Exception as e:  # noqa: BLE001
                # Capture blocked by a host-side op inside the op body; record
                # the miss (fall through to eager next time) and name the
                # blocker -- that host op is the remaining launch-collapse gate.
                cache[key] = False
                print(
                    f"[hier_graph] capture FAILED count={count} "
                    f"dtype={input_data.dtype}: {type(e).__name__}: {e}",
                    flush=True,
                )
                return True  # warm runs above already produced a valid output
            cache[key] = graph
            print(
                f"[hier_graph] captured count={count} " f"dtype={input_data.dtype}",
                flush=True,
            )
            return True
        if entry is False:
            return False  # capture impossible for this key -> eager path
        entry.replay()
        return True

    def _call_impl(self, input_data, output_data, count: int, stream=None) -> bool:
        if self.num_nodes == 1:
            return self._intra(input_data, output_data, count, stream)

        # Phase 1 (intra, SDMA): gather the G local shards into my node-block.
        G = self.ranks_per_node
        N = self.num_nodes
        block_count = count * G

        # M5: size-threshold dispatch. Engage the sliced 2-D path only
        # for large per-rank payloads (where it wins); below the threshold the
        # non-sliced fuse-barrier path is faster. On a path switch, conservatively
        # keep the entry barrier (clear the clean-completion guard) since the two
        # paths reuse the shared _intra/_inter buffers differently.
        byte_count = count * input_data.element_size()
        use_slice = self.slice_inter and (byte_count >= self.slice_min_bytes)
        # mid/small band (below slice_min) routes to the stream
        # pipe-overlap path (needs the sliced fused, non-oop, non-local-overlap
        # path with K>1 chunks -- same prerequisites as slice_pipe_overlap).
        use_pipe_band = (
            (not use_slice)
            and self.pipe_band
            and self.slice_inter
            and self.slice_fused
            and not self.slice_oop
            and not self.slice_overlap
            and self.slice_pipe_chunks > 1
            and byte_count >= self.pipe_band_min_bytes
        )
        # 3-way path key (None=non-slice, "pipe"=pipe-band, "slice"=slice path):
        # any switch reuses the shared _intra/_inter buffers differently, so
        # conservatively clear the clean-completion guard (forces an entry fence).
        path_key = "slice" if use_slice else ("pipe" if use_pipe_band else None)
        if path_key != self._last_use_slice:
            self._prev_op_completed = False
        self._last_use_slice = path_key

        # Path diagnostic (default-inert, bit-exact): prints the chosen path plus
        # fuse/slice/pipe state per call when MORI_HIER_PATH_LOG is set; zero effect
        # on the shipped path otherwise. An apparent "bimodal" large-buffer result
        # (fast fan-out vs a serial floor) is just fuse_local ON vs OFF: the floor
        # is the serial fuse_local=OFF slice_direct path. standalone_fast already
        # engages fuse_local, so path_key is deterministic per size (not a per-op
        # selection race), and the residual is the fixed per-op startup cost rather
        # than a per-NIC fill deficit.
        if _env_true("MORI_HIER_PATH_LOG", "0"):
            print(
                "[hier_path] bytes=%d path=%s slice_direct=%s fuse_local=%s "
                "slice_fused=%s slice_oop=%s slice_overlap=%s pipe_chunks=%d "
                "num_blocks=%d numQp=%d"
                % (
                    byte_count,
                    path_key,
                    getattr(self, "slice_direct", None),
                    getattr(self, "fuse_local", None),
                    self.slice_fused,
                    self.slice_oop,
                    self.slice_overlap,
                    self.slice_pipe_chunks,
                    self.inter_num_blocks,
                    self.inter_num_qp,
                ),
                flush=True,
            )

        if use_slice or use_pipe_band:
            # M5: SLICED 2-D AllGather (the bandwidth lever; see __init__).
            # Phase A (inter, RDMA ring): every rank rings ONLY its own shard
            # (count) across its same-local-index peers {g, g+G, ...}. The ring
            # gathers N chunks in node order into C_g; because slice_g(B_n) ==
            # shard[n*G+g] == this rank's own input, C_g == [slice_g(B_0)..
            # slice_g(B_{N-1})]. Per-NIC inter bytes = (N-1)*count (a G x cut vs
            # the default G*count), spread across all G NICs (no leader funnel).
            slice_total = count * N
            if (
                (self.slice_pipe_overlap or use_pipe_band)
                and self.slice_fused
                and not self.slice_oop
                and not self.slice_overlap
                and self.slice_pipe_chunks > 1
            ):
                # M5: CHUNKED-RING PIPELINE OVERLAP (rule#1 payoff).
                # Split count into K element-range chunks. For each chunk k run the
                # inter ring (main stream) into a DISJOINT region of the scratch,
                # then launch chunk k's N reassembly gathers on a side SDMA stream
                # (strided, no barrier). The side gather of chunk k overlaps the
                # main-stream ring of chunk k+1 -> only the last chunk's gather is
                # serial after the final ring. Scratch holds the full collection
                # (count*N); chunk k's N slices (N*ck) live at [N*off, N*off+N*ck)
                # contiguously (exactly what the ring finish_sync produces), so the
                # strided gather reads region[m*ck:(m+1)*ck] and writes block m at
                # element offset off with slot stride = count. All writes (across
                # k,m) are disjoint -> one final finish_batch copies them all out.
                K = self.slice_pipe_chunks
                base_ck = count // K
                # Optional uneven split for K==2 (front-load lever). Build an
                # explicit per-chunk size list; the even split reproduces the
                # base_ck bytes exactly. Bit-exact regardless of boundary.
                chunk_sizes = None
                if K == 2 and self.slice_pipe_split is not None and count > 1:
                    c0 = int(round(count * self.slice_pipe_split))
                    c0 = max(1, min(count - 1, c0))
                    chunk_sizes = [c0, count - c0]
                if (
                    self._slice_scratch is None
                    or self._slice_scratch.numel() < slice_total
                    or self._slice_scratch.dtype != input_data.dtype
                    or self._slice_scratch.device != input_data.device
                ):
                    self._slice_scratch = torch.empty(
                        slice_total, dtype=input_data.dtype, device=input_data.device
                    )
                collection = self._slice_scratch[:slice_total]
                if self._overlap_stream is None:
                    self._overlap_stream = torch.cuda.Stream(device=input_data.device)
                side = self._overlap_stream
                main = (
                    torch.cuda.current_stream(input_data.device)
                    if stream is None
                    else stream
                )
                # Side stream must observe the producer of input_data.
                side.wait_stream(main)
                off = 0
                # use the STREAM-ORDERED inter ring
                # (stream_ring, ) + deferred finish fence (defer_inter_fin,
                # ) here.  measured this chunked-ring overlap at -13%,
                # but that was with the HOST-blocking ShmemBarrierAll (2(K-1) CPU
                # round-trips); those wins did not exist yet. With on-device
                # barriers each chunk's prepare fence is ~0.03-0.05ms (not a host
                # stall), so the per-chunk barrier cost that killed  is now
                # small enough that the overlap (side gather of chunk k hidden
                # under the main-stream ring of chunk k+1) can net positive. Safety
                # mirrors : only ONE global on-stream fence is ever in
                # flight (the main-stream ring prepare); the side gathers run
                # barrier-free (prepare_barrier=False), so this is NOT the
                # concurrent-global-barrier race. Cross-chunk ring-buffer reuse is
                # ordered by chunk k+1's prepare_stream global fence (defer is safe
                # exactly as in the shipped non-chunked path).
                sr = self.stream_ring
                for k in range(K):
                    if chunk_sizes is not None:
                        ck = chunk_sizes[k]
                    else:
                        ck = base_ck if k < K - 1 else count - base_ck * (K - 1)
                    if ck == 0:
                        continue
                    region = collection[N * off : N * off + N * ck]
                    # Inter ring of chunk k on the MAIN stream. With stream_ring the
                    # finish copy-OUT into ``region`` is stream-ordered. The
                    # per-chunk landing fence picks how the peer's chunk-k landing
                    # is made globally visible before the side gather reads peer
                    # ``region`` over XGMI:
                    #  - "global": non-deferred cross-PE ShmemBarrierOnStream in
                    #    _inter (correct but serializes the K-pipe).
                    #  - "intra"/"off": DEFER the global barrier; "intra" instead
                    #    arms the first gather with the cheap intra-node subgroup
                    #    barrier (below); "off" leaves it unfenced (drifts).
                    _fence_global = self.slice_pipe_fence == "global"
                    _defer = sr and not _fence_global
                    self._inter(
                        input_data[off : off + ck],
                        region,
                        ck,
                        stream,
                        stream_ring=sr,
                        defer_inter_fin=_defer,
                    )
                    # Make the side stream observe the ring's copy-OUT (+ the
                    # cross-PE finish barrier when "global") into ``region``.
                    side.wait_stream(main)
                    # CHEAP intra-node landing fence: arm ONLY the first gather of
                    # this chunk with the intra-node SUBGROUP entry ShmemBarrier
                    # (G ranks, XGMI-scope, NO NIC quiet-drain / inter-node
                    # rendezvous). It rendezvouses all G local peers PAST their
                    # chunk-k ring copy-OUT + threadfence before ANY reads a peer
                    # ``region`` -- closing the exact intra-node landing race the
                    # global barrier closes, WITHOUT the inter-node all-to-all that
                    # eats the overlap. Same primitive as fuse_local's
                    # MORI_HIER_FUSE_LOCAL_INTRA_BAR (~3534). Once the first read is
                    # fenced, the remaining N-1 reads of this chunk are safe.
                    _chunk_intra_bar = self.slice_pipe_fence == "intra"
                    for m in range(N):
                        self._intra.gather_kernel(
                            region[m * ck : (m + 1) * ck],
                            ck,
                            dst_base_offset=m * block_count + off,
                            stream=side,
                            prepare_barrier=(_chunk_intra_bar and m == 0),
                            dst_slot_stride=count,
                        )
                    off += ck
                # All gathers (incl. the last chunk's, serial after the final ring)
                # must land before the bulk copy-OUT reads them.
                main.wait_stream(side)
                if sr and self.stream_intra:
                    self._intra.finish_batch_stream(
                        output_data, N * block_count, stream=stream, barrier=True
                    )
                else:
                    self._intra.finish_batch(
                        output_data, N * block_count, stream=stream, barrier=True
                    )
                self._prev_op_completed = True
                return True
            if self.slice_overlap and self.slice_fused and not self.slice_oop:
                # M5: OVERLAP lever (c). Run the LOCAL node-block gather
                # (m=node_id, reads this rank's own input == collection[node_id])
                # on a side stream CONCURRENTLY with the inter ring (Phase A).
                node = self.node_id
                if self._overlap_stream is None:
                    self._overlap_stream = torch.cuda.Stream(device=input_data.device)
                side = self._overlap_stream
                # Make the side stream observe any work already queued on the
                # caller's stream (e.g. the producer of input_data).
                main = (
                    torch.cuda.current_stream(input_data.device)
                    if stream is None
                    else stream
                )
                side.wait_stream(main)
                # Local-block gather on the side stream. Keep its entry barrier
                # (prepare_barrier=True) -- the single global ShmemBarrierAll that
                # frees out_ before any peer pushes; host-blocking so it stays
                # ordered ahead of the ring's barriers. Writes block node_id.
                self._intra.gather_kernel(
                    input_data,
                    count,
                    dst_base_offset=node * block_count,
                    stream=side,
                    prepare_barrier=True,
                )
                # Phase A inter ring on the MAIN stream (overlaps the side gather).
                if (
                    self._slice_scratch is None
                    or self._slice_scratch.numel() < slice_total
                    or self._slice_scratch.dtype != input_data.dtype
                    or self._slice_scratch.device != input_data.device
                ):
                    self._slice_scratch = torch.empty(
                        slice_total, dtype=input_data.dtype, device=input_data.device
                    )
                collection = self._slice_scratch[:slice_total]
                self._inter(input_data, collection, count, stream)
                # Remaining gathers (m != node_id) read the ring collection; their
                # out_ blocks are disjoint from the side gather's and freed by the
                # ring's finish barrier, so prepare_barrier=False is safe.
                for m in range(N):
                    if m == node:
                        continue
                    self._intra.gather_kernel(
                        collection[m * count : (m + 1) * count],
                        count,
                        dst_base_offset=m * block_count,
                        stream=stream,
                        prepare_barrier=False,
                    )
                # The bulk copy-OUT reads block node_id too, so the side gather
                # must be visible on the main stream first.
                main = (
                    torch.cuda.current_stream(input_data.device)
                    if stream is None
                    else stream
                )
                main.wait_stream(side)
                self._intra.finish_batch(
                    output_data, N * block_count, stream=stream, barrier=True
                )
                self._prev_op_completed = True
                return True
            if self.slice_oop:
                # M5: run the ring out-in-place and read the collection
                # straight from the (persistent) ring buffer -- no finish copy-OUT
                # into a separate scratch. ``output_data`` is ignored in this mode.
                self._inter(
                    input_data,
                    input_data,
                    count,
                    stream,
                    out_in_place=True,
                    stream_ring=self.stream_ring,
                )
                collection = self._inter.full_tensor(
                    count, input_data.dtype, input_data.device
                )
            else:
                if self._big_dbuf_active:
                    # dbufstream: dedicated double-buffered scratch for the big
                    # backward AG. Alternating buffers (parity toggled per big AG)
                    # ensure consecutive big embed/lm_head AGs never share the same
                    # collection, so a deferred finish fence on AG#1 cannot be
                    # clobbered by AG#2's ring copy-IN (the scratch-reuse staleness).
                    p = self._big_scratch_parity
                    buf = self._big_scratch[p]
                    if (
                        buf is None
                        or buf.numel() < slice_total
                        or buf.dtype != input_data.dtype
                        or buf.device != input_data.device
                    ):
                        buf = torch.empty(
                            slice_total,
                            dtype=input_data.dtype,
                            device=input_data.device,
                        )
                        self._big_scratch[p] = buf
                    collection = buf[:slice_total]
                else:
                    if (
                        self._slice_scratch is None
                        or self._slice_scratch.numel() < slice_total
                        or self._slice_scratch.dtype != input_data.dtype
                        or self._slice_scratch.device != input_data.device
                    ):
                        self._slice_scratch = torch.empty(
                            slice_total,
                            dtype=input_data.dtype,
                            device=input_data.device,
                        )
                    collection = self._slice_scratch[:slice_total]
                # when the direct-path local-block overlap is
                # active, the ring is run INTERLEAVED inside Phase B (split into
                # prepare_stream_only + kernel/finish so the local-block gather
                # can overlap the ring kernel on a side stream). Skip the
                # monolithic ring call here in that case.
                # the FUSED path (fuse_local) likewise runs the
                # ring inside Phase B (as part of the fused kernel launch), so it
                # must ALSO skip the monolithic ring call here.
                overlap_active = (
                    (self.slice_direct_overlap or self.fuse_local)
                    and self.slice_direct
                    and self.slice_fused
                    and self.stream_intra
                    and self.stream_ring
                    and self.slice_fuse_ib
                    and not (self.slice_pipe and self.slice_pipe_chunks > 1)
                )
                if not overlap_active:
                    self._inter(
                        input_data,
                        collection,
                        count,
                        stream,
                        stream_ring=self.stream_ring,
                        defer_inter_fin=self.slice_defer_inter_fin,
                    )
            # Phase B (intra, SDMA): reassemble each node-block B_m from the G
            # local ranks' m-th slices. Gather m writes the full block into
            # output[m*block:(m+1)*block]; the SDMA gather concatenates by
            # group_pos=local_rank => output == concat_m B_m == rank-major.
            if self.slice_fused:
                # M5: fold the N gathers into ONE batch -- stack each into
                # a DISJOINT region [m*block, (m+1)*block) of the enlarged transit
                # (dst_base_offset = m*block_count), drop the per-gather finish
                # barrier/copy, then ONE bulk copy-OUT. Keep only the m==0 entry
                # barrier and the final exit barrier (2 barriers vs 2N).
                # M5: the inter ring's finish barrier (run just above)
                # already synchronizes all PEs, so the m==0 entry barrier is
                # redundant -- drop it when slice_fuse_ib (default). Keep it only
                # if explicitly disabled (A/B / safety fallback).
                entry_barrier = (not self.slice_fuse_ib) or self.phaseb_entry_barrier
                if self.slice_pipe and self.slice_pipe_chunks > 1:
                    # M5: CHUNKED (strided) Phase-B. Split each block's
                    # reassembly gather into K element-range chunks; chunk j of
                    # peer g lands at m*block + g*count + j*ck via slot stride =
                    # count (the full slice). Byte-identical to the unchunked
                    # gather (this turn: correctness only, no ring overlap yet).
                    K = self.slice_pipe_chunks
                    base_ck = count // K
                    first = True
                    for m in range(N):
                        off = 0
                        for j in range(K):
                            ck = base_ck if j < K - 1 else count - base_ck * (K - 1)
                            if ck == 0:
                                continue
                            self._intra.gather_kernel(
                                collection[m * count + off : m * count + off + ck],
                                ck,
                                dst_base_offset=m * block_count + off,
                                stream=stream,
                                prepare_barrier=(entry_barrier and first),
                                dst_slot_stride=count,
                            )
                            off += ck
                            first = False
                    self._intra.finish_batch(
                        output_data, N * block_count, stream=stream, barrier=True
                    )
                elif self.slice_direct and self.stream_intra and self.stream_ring:
                    # DIRECT-TO-OUTPUT Phase B. Register the
                    # user output once (collective, cached) then PUSH each
                    # node-block's slices straight into output[m*block:] -- no
                    # internal transit, no full-output copy-OUT. Only a single
                    # global fence completes the op (deferrable like the
                    # copy-OUT path).
                    # LOCKSTEP single-registration. Register the
                    # user output collectively ONLY on an exact (ptr,size) change,
                    # deregistering the previous one first so the C++ map holds at
                    # most one entry and never runs its (potentially per-rank
                    # divergent) overlap-eviction. Steady state (same output reused
                    # op-to-op) skips the collective entirely; a buffer change runs
                    # exactly {Dereg(old) if old; Reg(new)} uniformly on every PE.
                    self._ensure_output_registered(output_data)
                    if (
                        self.fuse_remote
                        and self.num_nodes == 2
                        and not entry_barrier
                        and not self.slice_oop
                    ):
                        # PHASE 4 PIPELINE: one fused launch runs the ring AND,
                        # per landed sub-range, the remote-block SDMA reassembly
                        # straight from this PE's ring buffer into the registered
                        # output (no ring copy-OUT, no whole-phase finish barrier).
                        # The remote gather of sub-range j overlaps ring channel
                        # j+1 still crossing the NIC -- fuses the two serial phases.
                        from .collective import launch_fused_ring_remote_gather

                        node = self.node_id
                        rb = self._inter.num_blocks
                        # Persistent chunk-landing flag buffer. DEEP_PIPE splits the
                        # single ring channel into P temporal sub-chunks, each with
                        # its own landing flag, so the buffer must hold >= P slots
                        # (only engages at rb==1). Otherwise >= ring_blocks u64.
                        # Adaptive depth: the winning pipe depth is size-dependent --
                        # small AGs want a shallow pipe (landing-flag handshakes
                        # dominate) and big AGs a deep one (more overlap of the inter
                        # ring under the intra reassembly). The optimum tracks a
                        # roughly fixed per-PE sub-chunk size, so a single fixed depth
                        # loses at one end. MORI_HIER_DEEP_PIPE=auto picks depth =
                        # round(perPE_bytes / SUBBYTES) so every size rides its own
                        # optimum. An explicit integer keeps the exact prior behavior.
                        # Default depth 2 matches the C++ HierDeepPipe() default; the
                        # 32MB per-sub-chunk gate below self-cages the giant AG to the
                        # crown fence so depth 2 stays E2E bit-exact with no explicit
                        # env.
                        _dp_raw = os.environ.get("MORI_HIER_DEEP_PIPE", "2").strip()
                        if _dp_raw.lower() == "auto":
                            # 16MiB sub-chunk target (must match the C++
                            # HierDeepPipeSubBytes default): count*elsz is this PE's
                            # chunk, the same per-PE quantity the kernel gates on, so
                            # both selectors compute the identical depth and the flag
                            # buffer sized here holds exactly the kernel's sub-chunk
                            # count. 16MiB beats 8MiB on the mid-buffer path and stays
                            # under the 32MB coherence window. Only reached on
                            # DEEP_PIPE=auto.
                            _dp_sub_target = int(
                                os.environ.get(
                                    "MORI_HIER_DEEP_PIPE_SUBBYTES",
                                    str(16 * 1024 * 1024),
                                )
                            )
                            _dp_cb = int(count) * int(input_data.element_size())
                            _deep_pipe = int(
                                (_dp_cb + _dp_sub_target // 2) // _dp_sub_target
                            )
                        else:
                            _deep_pipe = int(_dp_raw)
                        if _deep_pipe < 1:
                            _deep_pipe = 1
                        if _deep_pipe > 16:
                            _deep_pipe = 16
                        # Size gate + self-safe default: the per-sub-chunk device
                        # landing signal is bit-exact and faster than native only
                        # while each temporal sub-chunk stays inside the MI300X/mlx5
                        # NIC-DMA->HBM coherence window (32MB per-PE chunk). Beyond it
                        # the signal can mismatch or crash on the giant embed/lm_head
                        # AG. With DEEP_PIPE>1 and no explicit
                        # MORI_HIER_DEEP_PIPE_MAXBYTES, default the gate to 32MB so
                        # DEEP_PIPE=2 alone stays E2E bit-exact (giant AG falls to the
                        # crown fence). DEEP_PIPE=1 => whole block inert.
                        # Gate on the PER-SUB-CHUNK coherence window (chunkBytes/P),
                        # not the total chunk: DEEP_PIPE=4 stays engaged on the
                        # 34-67MB steady-state decoder AGs (sub-chunks 8.5-16.75MB,
                        # under the 32MB window) while the 466MB giant AG (116MB
                        # sub-chunk) falls to the crown fence. Strict '<' cages a
                        # 32MB sub-chunk (64MB@P2). Mirrors the C++ HierDeepPipe gate.
                        _dp_chunk_bytes = int(count) * int(input_data.element_size())
                        # Sub-chunk NIC-fill floor: DEEP_PIPE>1 splits the per-PE ring
                        # shard into P temporal sub-chunks; at small per-PE sizes a
                        # deep P makes each sub-chunk too small to fill the mlx5 NIC
                        # DMA, so the ratio tracks sub-chunk size (sub-chunks >=8MiB
                        # win, smaller ones under-fill the NIC). Cap the effective
                        # depth so each temporal sub-chunk stays >= MINBYTES: large
                        # buffers keep their winning depth (sub already >= floor) while
                        # small buffers auto-drop to the deeper-filling depth. This is
                        # a per-QP NIC-fill lever, not a concurrency/overlap axis.
                        # Bit-exact by construction: a smaller depth pipelines the same
                        # bytes in the same RC order. Default 0=OFF; the shipped path
                        # (DEEP_PIPE=1) never enters this block. Note this floor helps
                        # only where the residual is sub-chunk-fill bound, not where it
                        # is fixed-cost bound.
                        _dp_floor = int(
                            os.environ.get("MORI_HIER_DEEP_PIPE_MINBYTES", "0")
                        )
                        if _deep_pipe > 1 and _dp_floor > 0:
                            _dp_max_depth = max(1, _dp_chunk_bytes // _dp_floor)
                            if _deep_pipe > _dp_max_depth:
                                _deep_pipe = _dp_max_depth
                        # MID-BUFFER PIPE-ENGAGE FLOOR (world=16):
                        # at world=16 the per-PE ring shard is total/16, so for total
                        # 32/64/128MB the per-PE chunk is only 2/4/8MB. The auto
                        # subtarget (8MB) then rounds the pipe depth to 1 => the whole
                        # DEEP_PIPE block is INERT and the inter NIC fill runs strictly
                        # SERIAL before the intra XGMI reassembly (no overlap partner),
                        # which is exactly the measured w16 mid-buffer ratio floor
                        # (32/64/128MB = 0.75/0.79/0.81x) vs 256/512MB (which DO
                        # pipeline) at ~0.88x. MORI_HIER_DP_MIN_DEPTH forces a minimum
                        # temporal depth so those mid buffers pipeline: while a
                        # sub-chunk crosses the NIC the prior sub-chunk's already-landed
                        # bytes reassemble over XGMI. Capped so each sub-chunk stays
                        # >= 16B (aligned split) and <= the 16 clamp; the MAXBYTES
                        # coherence gate below still fires (raising depth only SHRINKS
                        # the sub-chunk, moving it further inside the landing window).
                        # Bit-exact BY CONSTRUCTION: a deeper valid depth pipelines the
                        # SAME bytes in the SAME per-peer RC order, all sub-chunks
                        # drained before the completion flag (identical argument to the
                        # MINBYTES floor cap above, and B's per-size DP2/4/8/16 are each
                        # independently bit-exact). Default 1 => OFF => byte-identical
                        # shipped path (block already skipped at DEEP_PIPE=1).
                        _dp_min_depth = int(
                            os.environ.get("MORI_HIER_DP_MIN_DEPTH", "1")
                        )
                        if _dp_min_depth > 1 and _dp_chunk_bytes >= 32:
                            _dp_split_cap = max(1, _dp_chunk_bytes // 16)
                            _dp_tgt = min(_dp_min_depth, _dp_split_cap, 16)
                            if _dp_tgt > _deep_pipe:
                                _deep_pipe = _dp_tgt
                        _dp_window = int(
                            os.environ.get(
                                "MORI_HIER_DEEP_PIPE_MAXBYTES", str(32 * 1024 * 1024)
                            )
                        )
                        _dp_sub_bytes = (
                            _dp_chunk_bytes // _deep_pipe
                            if _deep_pipe > 1
                            else _dp_chunk_bytes
                        )
                        if (
                            _deep_pipe > 1
                            and _dp_window > 0
                            and _dp_sub_bytes >= _dp_window
                        ):
                            _deep_pipe = 1
                        # PER-PE TOTAL FLOOR: mirror the C++
                        # HierDeepPipeMinBytes gate (MORI_HIER_DEEP_PIPE_MIN_MB,
                        # commit 83368b34) in the Python landing-flag selector. The
                        # kernel drops deepPipe->1 for any PER-PE chunk BELOW the
                        # floor -- that floor is the LOW edge of the w16 [MIN,MAX]
                        # window that cages the 32/64MB small-buffer deep-pipe HANG
                        # (T9) and pins the pipeline to the mid-buffer band where it
                        # wins. But Python sized _flag_slots from the PRE-floor depth
                        # (auto/min_depth), so a caged small chunk still allocated the
                        # deeper sub-chunk landing budget the kernel never publishes to
                        # -- a stale-slot layout the carryover fix has to churn
                        # over on every distinct size. Snap the Python depth to the
                        # SAME per-PE floor so _flag_slots == the kernel's gated depth
                        # exactly across the whole [MIN,MAX] window. Down-only (floor
                        # can only cage to depth 1, the plain path) => bit-exact BY
                        # CONSTRUCTION (a caged chunk takes the identical deepPipe<=1
                        # code path the kernel takes). Default 0 => no floor =>
                        # byte-identical shipped path (matches the C++ default).
                        _dp_floor_mb = int(
                            os.environ.get("MORI_HIER_DEEP_PIPE_MIN_MB", "0")
                        )
                        if (
                            _deep_pipe > 1
                            and _dp_floor_mb > 0
                            and _dp_chunk_bytes < _dp_floor_mb * 1024 * 1024
                        ):
                            _deep_pipe = 1
                        # DP-DEBUG: one-time-per-distinct-size print of the
                        # RESOLVED deep-pipe depth so we can confirm whether the giant
                        # root embed/lm_head AG actually pipelines (depth>1) or falls to
                        # the crown whole-chunk fence (depth==1). Diagnostic only; env-
                        # gated, no effect on the shipped path.
                        if _env_true("MORI_HIER_DP_DEBUG", ""):
                            _dbg = getattr(self, "_dp_dbg_seen", None)
                            if _dbg is None:
                                _dbg = self._dp_dbg_seen = set()
                            _dbg_key = (int(_dp_chunk_bytes), int(_deep_pipe))
                            if _dbg_key not in _dbg:
                                _dbg.add(_dbg_key)
                                _sub = (
                                    _dp_chunk_bytes // _deep_pipe
                                    if _deep_pipe > 1
                                    else _dp_chunk_bytes
                                )
                                print(
                                    f"[dp-debug] perPE_chunk_MB="
                                    f"{_dp_chunk_bytes/1048576:.2f} depth={_deep_pipe} "
                                    f"sub_MB={_sub/1048576:.2f}",
                                    flush=True,
                                )
                        _flag_slots = max(rb, _deep_pipe if rb == 1 else 1, 1)
                        # PER-QP FINE-GRAIN INTER-ARRIVAL DRAIN (MORI_HIER_SHARD_DRAIN):
                        # the producer publishes one landing flag per QP shard (sw =
                        # min(numQp, 8)), so chunkReadyFlags needs numQp slots (>2 the
                        # deep_pipe default). Bump the buffer accordingly; default OFF
                        # leaves _flag_slots byte-identical.
                        if _env_true("MORI_HIER_SHARD_DRAIN", "0") and rb == 1:
                            _sw = min(int(self.inter_num_qp), 8)
                            if _sw > _flag_slots:
                                _flag_slots = _sw
                        # CROSS-SIZE CARRYOVER FIX: the
                        # persistent chunk-landing flag buffer must be reallocated
                        # on a LAYOUT change (per-PE count / slots / pipe depth),
                        # not only when it needs to GROW. Reusing one buffer across
                        # two DIFFERENT DEEP_PIPE sizes in a single process (the UT
                        # sweep drives all sizes through ONE handle) left stale
                        # per-sub-chunk landing state that made the 2nd distinct
                        # size bit-exact MISMATCH (32MB ok -> 64MB fail) even though
                        # every size ISOLATED is clean and DEEP_PIPE=1 sweeps clean
                        # across sizes -- localizing the carryover to exactly this
                        # DP flag path. A fresh zeroed buffer per distinct layout
                        # (tiny, <=16 u64) removes it; SAME-size steady state (E2E
                        # decoder AGs, bench reps) still reuses + zeros with no
                        # per-call alloc, and DEEP_PIPE=1 (shipped default) never
                        # enters this block so the default path stays byte-identical.
                        _dp_layout = (_flag_slots, int(count), _deep_pipe)
                        if (
                            self._chunk_ready_flags is None
                            or self._chunk_ready_flags.numel() < _flag_slots
                            or self._chunk_ready_flags_layout != _dp_layout
                        ):
                            self._chunk_ready_flags = torch.zeros(
                                _flag_slots, dtype=torch.int64, device=input_data.device
                            )
                            self._chunk_ready_flags_layout = _dp_layout
                        flags = self._chunk_ready_flags
                        # GEN-TOKEN: in token mode advance the per-op
                        # generation and SKIP the host reset (the higher token
                        # supersedes stale slots); a fresh re-alloc above is still
                        # zeroed so gen starts clean. Legacy mode zeroes every op.
                        if self._flag_token_dev:
                            # DEVICE flag-token: the crown overrides opGen with the
                            # device parity counter (graph-safe); host only skips the
                            # reset so the flags accumulate. Pass op_gen=0 (ignored by
                            # the kernel once flagGenDev fires).
                            _op_gen = 0
                        elif self._flag_token:
                            self._flag_opgen += 1
                            _op_gen = self._flag_opgen
                        else:
                            _op_gen = 0
                            flags.zero_()
                        # HOST-PROXY INTER (MORI_HIER_HOSTPROXY_REASM=1): the
                        # device ring-send CTAs skip the RDMA send; a persistent
                        # CPU proxy owns the inter leg and publishes
                        # chunkReadyFlags[f] after its send-CQ drains. Build the
                        # producer lazily against THIS PE's ring buffer.
                        _hp_reasm = _env_true("MORI_HIER_HOSTPROXY_REASM", "0")
                        _hp_prod = None
                        if _hp_reasm:
                            _hp_prod = self._get_hostproxy_inter()
                        # Ring prepare = global entry barrier + copy-IN (no launch).
                        # fuse_copyin: skip the host copy-IN; the fused kernel stages
                        # each channel's send sub-range in-kernel (single-launch collapse).
                        if self.fuse_copyin:
                            ring_args, u32c, s_main = (
                                self._inter.prepare_stream_only_no_copyin(
                                    input_data, count, stream
                                )
                            )
                        else:
                            ring_args, u32c, s_main = self._inter.prepare_stream_only(
                                input_data, count, stream
                            )
                        if _hp_prod is not None:
                            # record the copy-IN so the host RDMA read sees it.
                            if self._hp_src_ev is None:
                                self._hp_src_ev = torch.cuda.Event()
                            self._hp_src_ev.record(
                                torch.cuda.current_stream(input_data.device)
                                if stream is None
                                else stream
                            )
                        # Local-block direct-gather jit_args (m == node_id): reads
                        # this rank's own input, no ring dependency.
                        gather_args = self._intra.prepare_direct_only(
                            input_data,
                            output_data,
                            count,
                            dst_block_offset=node * block_count,
                            stream=stream,
                            prepare_barrier=False,
                        )
                        # PHASE 4 (deadlock-free push-only reassembly): the remote
                        # blocks are now PUSH-ONLY (each rank writes its own column
                        # into disjoint output slots, no cross-rank wait) with a
                        # single completion reader in the local-block CTA, so
                        # reasm>1 no longer dead-locks. Each reassembly block j uses
                        # SDMA queue qId=j, so to avoid racing the per-queue signal
                        # counter reasm MUST be <= the peer's SDMA queue count
                        # (sdmaNumQueue, default 2). Clamp accordingly. Flags use
                        # slots [0,G) (local) + [G, G+reasm*(N-1)*G) (reassembly),
                        # covered by the enlarged intra flags buffer.
                        # Reassembly blocks use SDMA queues [1, nq); queue 0 is
                        # taken by the concurrent local-block CTA. So max safe
                        # concurrent reassembly blocks == sdmaNumQueue-1. The ACTUAL
                        # per-peer SDMA queue count is MORI_SDMA_NUM_CHANNELS
                        # (anvil::GetSdmaNumChannels, default 2); MORI_HIER_SDMA_NQ
                        # overrides the clamp if set. Use the real channel count as
                        # the default so the SPATIAL reassembly split (rb==1,
                        # REASSEM_BLOCKS>1) actually lands on distinct queues/engines
                        # instead of wrapping onto queue 0 (per-queue counter race).
                        _sdma_nq = int(
                            os.environ.get(
                                "MORI_HIER_SDMA_NQ",
                                os.environ.get("MORI_SDMA_NUM_CHANNELS", "2"),
                            )
                        )
                        # NOTE: the fused-remote reassembly worker drives
                        # SDMA queue q = (j+1) % nq_physical (kernel, ccl_kernels.hip
                        # qId=j+1) while the local-block CTA owns queue 0. nq>=2 is
                        # REQUIRED so the reasm worker lands on queue 1, not queue 0
                        # (at physical nq==1 the modulo aliases onto queue 0 -> the
                        # two share one per-queue signal counter -> an intermittent
                        # liveness HANG, reproduced this turn via a 32MB->64MB size
                        # transition). The physical queue count is fixed at shmem/
                        # anvil init (BEFORE this op runs and before HierAllGather
                        # __init__), so it can only be corrected by setting
                        # MORI_SDMA_NUM_CHANNELS>=2 up front; mori's library default
                        # (anvil::GetSdmaNumChannels) is already 2 (E2E/FSDP is safe).
                        # bench_sweep formerly forced 1 -> fixed this turn.
                        # T3 (A): AUTO-SCALE the reassembly tail to the available SDMA
                        # engines. The reassembly TAIL (SDMA drain after the last inter
                        # RDMA land, HIERPROF ~40-45% of the giant-AG wall) runs on
                        # queues [1, reasm]; with the historical default reasm=1 it is
                        # SINGLE-ENGINE (~50 GB/s) -- the documented 0.90x/128MB device
                        # wall. Defaulting reasm to (nq-1) makes raising
                        # MORI_SDMA_NUM_CHANNELS automatically fan the tail across ALL
                        # spare engines via the existing SPATIAL split (rb==1,DP<=1,
                        # effReasm>1), the on-thesis large-buffer BW lever. Byte-
                        # identical at the default nq=2 (reasm still clamps to 1); the
                        # E2E giant-AG path runs nq=2 so it is unaffected. Explicit
                        # MORI_HIER_REASSEM_BLOCKS still overrides.
                        reasm = int(
                            os.environ.get(
                                "MORI_HIER_REASSEM_BLOCKS", str(max(1, _sdma_nq - 1))
                            )
                        )
                        if reasm > _sdma_nq - 1:
                            reasm = _sdma_nq - 1
                        if reasm < 1:
                            reasm = 1
                        # Double-buffer: the parity counter ptr is 0 unless
                        # MORI_HIER_GEN_RING_DBL is engaged (requires GEN_RING_DEV);
                        # when nonzero the launcher fires the captured parity bump on
                        # s_main before the fused kernel so op N+1 lands in the other
                        # ring half than op N reassembles, closing the reuse race.
                        _parity_ptr = 0
                        _pcp = getattr(self._inter, "parity_counter_ptr", None)
                        if _pcp is not None:
                            _parity_ptr = _pcp()
                        launch_fused_ring_remote_gather(
                            ring_args,
                            gather_args,
                            rb,
                            flags.data_ptr(),
                            N,
                            node,
                            s_main,
                            reassembly_blocks=reasm,
                            op_gen=_op_gen,
                            reasm_deep_sq=self._reasm_deep_sq,
                            parity_ptr=_parity_ptr,
                        )
                        if _hp_prod is not None:
                            # The kernel is now live: ring CTAs no-op (host owns
                            # inter), reassembly workers spin on chunkReadyFlags.
                            # The host proxy RDMA-writes this PE's ring chunk into
                            # the partner's ring buffer, drains its send-CQ (the
                            # proven landing fence), rail-pair barriers, then
                            # publishes chunkReadyFlags[f] so the reassembly runs.
                            chunk_bytes = count * input_data.element_size()
                            _hp_async = _env_true("MORI_HIER_HOSTPROXY_ASYNC", "0")
                            if _hp_async:
                                # non-blocking: worker owns the inter round-trip
                                # so the caller keeps issuing / computing while it
                                # runs; the kernel + deferred fence gate consume.
                                _hp_prod.fill_async(
                                    chunk_bytes,
                                    _deep_pipe,
                                    flags,
                                    src_ready_event=self._hp_src_ev,
                                    stream=stream,
                                )
                            else:
                                _hp_prod.fill(
                                    chunk_bytes,
                                    _deep_pipe,
                                    flags,
                                    src_ready_event=self._hp_src_ev,
                                    stream=stream,
                                )
                        # Single completion fence (no copy-OUT: gathers already
                        # pushed straight into the user output).
                        # Standalone finish-barrier deferral: at ranks_per_node>=8 the
                        # fused path is forced slice_defer_fin=False (dense-node E2E
                        # drifts without the two finish fences), so it issues a global
                        # cross-node ShmemBarrierOnStream every op -- one of two per-op
                        # barriers (the other is prepare_stream_only's entry fence). The
                        # device completion reader (the bx==rb CTA) has already spun
                        # until every remote push landed in this PE's output plus
                        # __threadfence_system, so this PE's output is stream-correct
                        # without the finish barrier. The finish barrier only adds
                        # cross-PE ring-buffer reuse safety for the next op, and the
                        # successor op's prepare_stream entry barrier (always global,
                        # before any copy-IN / peer RDMA put) already provides that
                        # ordering. The last op has no successor and reuses nothing.
                        # So deferring this finish barrier drops the per-op barrier
                        # count from 2 to 1 with byte-identical output. Gated on
                        # standalone_fast only; no FSDP/E2E caller passes it, so the
                        # dense-node E2E finish fences are untouched. Default OFF; set
                        # MORI_HIER_STANDALONE_DEFER_FIN=1 to defer.
                        _crown_fin_barrier = not self.slice_defer_fin
                        if self._standalone_defer_fin:
                            _crown_fin_barrier = False
                        self._intra.finish_direct_stream(
                            stream=stream, barrier=_crown_fin_barrier
                        )
                        # CU-COHERENT COPY-OUT (MORI_HIER_FUSE_REMOTE_RETOUCH):
                        # the fused kernel lands every peer's SDMA push into the
                        # output (per-queue SdmaQueitThread + threadfence + flag,
                        # waited by the completion reader) -- fabric-coherent in HBM
                        # but the consumer GEMM may read a STALE L2 line for the
                        # reused FSDP output buffer (the +0.018 E2E drift). Republish
                        # with a FULL-GRID volatile-glc re-touch (bypass stale L2,
                        # fetch fresh HBM, store back) AFTER the completion fence, so
                        # it is stream-ordered behind the landing and runs on all
                        # CUs (fast, not the thread-starved in-kernel pass). No host
                        # stall. Default OFF (byte-identical shipped path).
                        if os.environ.get(
                            "MORI_HIER_FUSE_REMOTE_RETOUCH", "0"
                        ).strip().lower() in ("1", "true", "yes", "on"):
                            from .collective import (
                                launch_l2_coherent_retouch,
                                _stream_to_int,
                            )

                            _u32 = (
                                output_data.numel() * output_data.element_size()
                            ) // 4
                            launch_l2_coherent_retouch(
                                output_data.data_ptr(), _u32, _stream_to_int(stream)
                            )
                    elif self.fuse_local and not entry_barrier and not self.slice_oop:
                        # FUSED ring || local-block gather in ONE
                        # kernel launch (NIC ring blocks [0,num_blocks) || XGMI
                        # local-block SDMA gather in the last block). Replaces the
                        # slice_direct_overlap path's two launches + side-stream
                        # wait_stream merge with a single concurrent grid -- the
                        # RCCL-parity lever this work proved (>= RCCL @>=32MiB),
                        # adopted. The ring's prepare_stream barrier is
                        # the sole global entry fence; the local gather runs
                        # barrier-free (prepare_barrier=False), reading only this
                        # rank's own input (no ring dependency) and pushing block
                        # node_id straight into the registered output. The REMOTE
                        # blocks (which DO depend on the ring) still follow as
                        # separate direct gathers after the ring copy-OUT.
                        from .collective import launch_fused_ring_local_gather

                        node = self.node_id
                        main = (
                            torch.cuda.current_stream(input_data.device)
                            if stream is None
                            else stream
                        )
                        # Ring prepare = global entry barrier + copy-IN (no kernel
                        # launch yet); returns the ring jit_args ptr.
                        # When MORI_HIER_FUSE_COPYIN is on, skip the host copy-IN --
                        # the fused kernel stages this PE's send sub-range in-kernel
                        # (fuseCopyIn) before the RDMA put, dropping one GPU op while
                        # keeping the entry rendezvous.
                        if self.fuse_copyin:
                            ring_args, u32c, s_main = (
                                self._inter.prepare_stream_only_no_copyin(
                                    input_data, count, stream
                                )
                            )
                        else:
                            ring_args, u32c, s_main = self._inter.prepare_stream_only(
                                input_data, count, stream
                            )
                        # Local-block direct-gather jit_args (no launch); writes
                        # block node_id straight into the registered output.
                        gather_args = self._intra.prepare_direct_only(
                            input_data,
                            output_data,
                            count,
                            dst_block_offset=node * block_count,
                            stream=stream,
                            prepare_barrier=False,
                        )
                        # ONE fused launch: ring (num_blocks CTAs) || local gather
                        # (1 CTA), concurrent on the same stream after the entry
                        # barrier. No host wait_stream merge.
                        launch_fused_ring_local_gather(
                            ring_args, gather_args, self._inter.num_blocks, s_main
                        )
                        # Ring finish copy-OUT into the collection scratch (the
                        # ring kernel already ran inside the fused launch -- do NOT
                        # relaunch it). RED-LINE: force the COPY-ENGINE finish (not
                        # the CU RingFinishCopyKernel) so the fuse_local win path
                        # never moves bulk all-gather bytes on CUs -- the fused
                        # overlap ALONE beats RCCL bit-exact with the copy-engine
                        # finish, so the CU copy is pure red-line
                        # cost with ~1% BW upside. MORI_HIER_RING_CU_COPYOUT can
                        # still force CU for an explicit A/B, but default stays SDMA.
                        _fl_cu = os.environ.get(
                            "MORI_HIER_RING_CU_COPYOUT", "0"
                        ).strip().lower() in ("1", "true", "yes", "on")
                        # PHASE 4 E2E-COHERENCE lever (MORI_HIER_FUSE_LOCAL_NOCOPY):
                        # the finish_ring_stream copy-OUT is a COPY-ENGINE (SDMA)
                        # D2D read of the ring buffer. This
                        # copy-engine read is the fuse_local E2E stale-remote race:
                        # the ring CTA lands the remote half via NIC RDMA and does a
                        # CU-scope __threadfence_system, but the copy engine is a
                        # SEPARATE hw agent NOT ordered by that fence, so its D2D
                        # can drain STALE remote-half ring bytes into ``collection``,
                        # which the reassembly then propagates. This lever DROPS the
                        # copy-OUT entirely and points the remote reassembly SDMA
                        # read straight at the ring buffer (full_tensor view) -- one
                        # fewer cross-agent hop, no copy-engine D2D of bulk bytes,
                        # AND a saved ~(N-1)/N copy-out (BW upside). Byte-identical by
                        # construction: ring slot m == collection[m]. Default OFF for
                        # A/B; keeps bulk bytes on SDMA (red-line safe).
                        # COPY-OUT ELIMINATION on the standalone crown:
                        # the finish_ring_stream copy-OUT is a copy-engine D2D of
                        # (N-1)/N of a node-block PLUS its own kernel launch --
                        # part of the ~0.28ms FIXED per-op cost that is the
                        # SOLE remaining UT ratio residual (marginal fill already
                        # == RCCL). NOCOPY drops that copy-OUT and points the remote
                        # reassembly SDMA read straight at the ring buffer (ring slot
                        # m == collection[m], byte-identical by construction); the
                        # trailing finish_direct_stream + the intra entry barrier on
                        # the first remote gather (default ON) KEEP the landing fence
                        # (same-stream ordering completes the fused ring CTA before the
                        # remote gather reads, __threadfence_system makes the NIC half
                        # visible). This is the mission "copy-out elimination that
                        # KEEPS the landing fence" lever. Auto-ON for standalone_fast
                        # ONLY (the UT gate, no cross-PE tight reuse => the E2E
                        # stale-remote race documented above never triggers); every
                        # FSDP/E2E caller omits standalone_fast => byte-identical,
                        # default OFF there. Env still overrides either way.
                        _fl_nocopy_env = os.environ.get("MORI_HIER_FUSE_LOCAL_NOCOPY")
                        if _fl_nocopy_env is not None:
                            _fl_nocopy = _fl_nocopy_env.strip().lower() in (
                                "1",
                                "true",
                                "yes",
                                "on",
                            )
                        else:
                            _fl_nocopy = bool(getattr(self, "_standalone_fast", False))
                        if _fl_nocopy:
                            # DROP the copy-OUT: read the ring buffer directly. The
                            # trailing finish_direct_stream fence + the intra entry
                            # barrier on the first remote gather (below) order the
                            # ring landing before any reassembly read and complete
                            # the op, so no separate ring copy-OUT fence is needed.
                            reasm_src = self._inter.full_tensor(
                                count, input_data.dtype, input_data.device
                            )
                        else:
                            reasm_src = collection
                            self._inter.finish_ring_stream(
                                collection,
                                count,
                                stream,
                                barrier=not self.slice_defer_inter_fin,
                                cu_copyout=_fl_cu,
                            )
                        # Remaining (remote) blocks read the ring collection.
                        # PHASE 4 E2E-SAFETY: the remote reassembly is an INTRA-node
                        # subgroup gather -- each local rank SDMA-pushes its OWN
                        # collection[m] slice to the G local peers over XGMI. That
                        # slice was produced by THIS rank's finish_ring_stream copy-
                        # OUT of the ring buffer. Under FSDP tight back-to-back reuse
                        # a peer rank's gather can read our collection over XGMI
                        # BEFORE our copy-OUT (and the concurrently-launched fused
                        # ring CTA's remote-half landing) is globally visible -> the
                        # documented ~48% stale-remote-half loss race that keeps
                        # fuse_local default-OFF. The general fix (phaseb_entry_barrier)
                        # is a FULL global ShmemBarrierOnStream (NIC quiet-drain +
                        # inter-node all-to-all) and is coded mutually-exclusive with
                        # fuse_local. But the dependency here is purely INTRA-node
                        # (Phase-B reads only same-node peers' collection), so a
                        # single INTRA-node subgroup entry barrier on the first remote
                        # gather strictly orders every local rank's ring copy-OUT
                        # BEFORE any peer reads it -- on-device, XGMI-scope only, NO
                        # NIC quiet-drain, NO inter-node barrier. Cheap (G ranks) vs
                        # the global fence, so fuse_local keeps its ~200 GB/s while
                        # becoming E2E bit-exact. Set MORI_HIER_FUSE_LOCAL_INTRA_BAR=0
                        # to A/B (restores the racy no-barrier path).
                        _fl_intra_bar = _env_true("MORI_HIER_FUSE_LOCAL_INTRA_BAR", "1")
                        remotes = [m for m in range(N) if m != node]
                        if self.reasm_streams > 1 and len(remotes) > 1:
                            # MULTI-STREAM the N-1 remote reassembly gathers
                            # (disjoint output blocks; entry barrier on the first,
                            # which runs on main so side lanes observe it).
                            main = (
                                torch.cuda.current_stream(input_data.device)
                                if stream is None
                                else stream
                            )

                            def _mkr(m, first, lane):
                                def _run(s):
                                    self._intra.gather_kernel_direct(
                                        reasm_src[m * count : (m + 1) * count],
                                        output_data,
                                        count,
                                        dst_block_offset=m * block_count,
                                        stream=s,
                                        prepare_barrier=(_fl_intra_bar and first),
                                        # disjoint flag-slot region per lane
                                        # so concurrent gathers never race.
                                        flag_slot_base=lane * self.ranks_per_node,
                                    )

                                return _run

                            self._multistream_gathers(
                                [_mkr(m, i == 0, i) for i, m in enumerate(remotes)],
                                main,
                                input_data.device,
                            )
                        else:
                            _first_remote = True
                            for m in remotes:
                                self._intra.gather_kernel_direct(
                                    reasm_src[m * count : (m + 1) * count],
                                    output_data,
                                    count,
                                    dst_block_offset=m * block_count,
                                    stream=stream,
                                    prepare_barrier=(_fl_intra_bar and _first_remote),
                                )
                                _first_remote = False
                        self._intra.finish_direct_stream(
                            stream=stream, barrier=not self.slice_defer_fin
                        )
                    elif (
                        self.slice_direct_overlap
                        and not entry_barrier
                        and not self.slice_oop
                    ):
                        # overlap the LOCAL node-block (m=node_id)
                        # reassembly gather (reads only this rank's own input, no
                        # ring dependency) on a side stream concurrently with the
                        # inter ring kernel. The ring's prepare_stream barrier is
                        # the sole global entry fence (see __init__).
                        node = self.node_id
                        if self._overlap_stream is None:
                            self._overlap_stream = torch.cuda.Stream(
                                device=input_data.device
                            )
                        side = self._overlap_stream
                        main = (
                            torch.cuda.current_stream(input_data.device)
                            if stream is None
                            else stream
                        )
                        # Ring prepare = global entry barrier + copy-IN (main).
                        args, u32c, s_main = self._inter.prepare_stream_only(
                            input_data, count, stream
                        )
                        # Side stream observes the entry barrier, then runs the
                        # local-block gather barrier-free, concurrent with the ring.
                        side.wait_stream(main)
                        self._intra.gather_kernel_direct(
                            input_data,
                            output_data,
                            count,
                            dst_block_offset=node * block_count,
                            stream=side,
                            prepare_barrier=False,
                        )
                        # Ring kernel + finish on main (overlaps the side gather).
                        self._inter.launch_finish_stream(
                            args,
                            collection,
                            u32c,
                            s_main,
                            barrier=not self.slice_defer_inter_fin,
                        )
                        # Remaining (remote) blocks read the ring collection (main).
                        for m in range(N):
                            if m == node:
                                continue
                            self._intra.gather_kernel_direct(
                                collection[m * count : (m + 1) * count],
                                output_data,
                                count,
                                dst_block_offset=m * block_count,
                                stream=stream,
                                prepare_barrier=False,
                            )
                        # Merge the side local-block gather before the op fence.
                        main.wait_stream(side)
                        self._intra.finish_direct_stream(
                            stream=stream, barrier=not self.slice_defer_fin
                        )
                    elif self.reasm_streams > 1 and N > 1:
                        # MULTI-STREAM the N disjoint reassembly gathers.
                        main = (
                            torch.cuda.current_stream(input_data.device)
                            if stream is None
                            else stream
                        )

                        def _mk(m, lane):
                            def _run(s):
                                self._intra.gather_kernel_direct(
                                    collection[m * count : (m + 1) * count],
                                    output_data,
                                    count,
                                    dst_block_offset=m * block_count,
                                    stream=s,
                                    prepare_barrier=(entry_barrier and m == 0),
                                    # disjoint flag-slot region per lane so
                                    # concurrent gathers never race on flag slots.
                                    flag_slot_base=lane * self.ranks_per_node,
                                )

                            return _run

                        self._multistream_gathers(
                            [_mk(m, i) for i, m in enumerate(range(N))],
                            main,
                            input_data.device,
                        )
                        self._intra.finish_direct_stream(
                            stream=stream, barrier=not self.slice_defer_fin
                        )
                    else:
                        for m in range(N):
                            self._intra.gather_kernel_direct(
                                collection[m * count : (m + 1) * count],
                                output_data,
                                count,
                                dst_block_offset=m * block_count,
                                stream=stream,
                                prepare_barrier=(entry_barrier and m == 0),
                            )
                        self._intra.finish_direct_stream(
                            stream=stream, barrier=not self.slice_defer_fin
                        )
                else:
                    for m in range(N):
                        self._intra.gather_kernel(
                            collection[m * count : (m + 1) * count],
                            count,
                            dst_base_offset=m * block_count,
                            stream=stream,
                            prepare_barrier=(entry_barrier and m == 0),
                        )
                    # stream-ordered copy-OUT (no host
                    # round-trip) when paired with the stream_ring inter ring.
                    # defer this fence to the next op's inter
                    # prepare barrier (slice_defer_fin) -- the copy-OUT stays
                    # stream-ordered so output is correct; only cross-PE reuse
                    # needs the fence, which the successor op provides.
                    if self._py_cu_copyout:
                        self._cu_copyout_finish(output_data, N * block_count, stream)
                    elif self.stream_intra and self.stream_ring:
                        self._intra.finish_batch_stream(
                            output_data,
                            N * block_count,
                            stream=stream,
                            barrier=not self.slice_defer_fin,
                        )
                    else:
                        self._intra.finish_batch(
                            output_data, N * block_count, stream=stream, barrier=True
                        )
            else:
                for m in range(N):
                    self._intra(
                        collection[m * count : (m + 1) * count],
                        output_data[m * block_count : (m + 1) * block_count],
                        count,
                        stream,
                    )
            self._prev_op_completed = True
            return True

        # M4 (/32): fuse-barrier also drops the intra-gather ENTRY barrier
        # on the every-rank-direct path, but ONLY when the PRIOR op completed
        # cleanly (its inter-finish barrier freed every peer's out_). The first op,
        # and any op following a mid-pipeline crash, keep the barrier (see __init__).
        prev_op_completed = self._prev_op_completed
        # Cleared until THIS op finishes; any exception below leaves it False so the
        # next op conservatively keeps the entry barrier.
        self._prev_op_completed = False
        intra_prepare_barrier = not (
            self.fuse_barrier and not self.leader_only and prev_op_completed
        )

        if not self.leader_only and self.gather_in_place:
            # M4 (, opt-in): write the intra-gather node-block DIRECTLY
            # into this PE's ring slot, then run the ring with chunk_in_place=True.
            # This removes the prepare_sync copy-IN (a full node-block D2D copy)
            # AND the node_block intermediate -- the gather's own finish_sync now
            # lands straight in the ring buffer.
            #
            # VALIDATED-NEUTRAL (, single-node world=8 N=2,G=4 fp32
            # 64MiB/rank, >=3 reps, A/B same binary): gather_in_place 84.7 GB/s
            # vs staged (default) 83-85 GB/s -- within noise. The eliminated
            # copy is NOT free here: the ring slot lives in the UNCACHED
            # symmetric heap, so the gather's finish_sync now writes 256MiB into
            # uncached memory (~slow) instead of into normal-HBM node_block
            # (~fast) -- the saved copy is offset by the slower write. Kept
            # opt-in; default stays the proven staged path. The copy-IN can only
            # be made cheaper by also moving the intra transit into the same
            # symmetric region, a larger change.
            #
            # Confirmed neutral on true cross-node RDMA (N=2, G=4, fp32
            # 64MiB/rank, both bit-exact vs torch.all_gather_into_tensor):
            # gather_in_place and staged are within noise (~60 GB/s), inter
            # phase identical either way (the copy-IN saving never materializes -- same
            # uncached-heap offset on the real RDMA transport, not a single-node
            # P2P artifact). So copy-IN elimination is a dead lever on this
            # topology; the remaining staging fish is the finish_sync copy-OUT
            # (~2.7ms @512MiB), which needs RDMA-into-user-output (register the
            # output as symmetric) to avoid the same uncached read penalty.
            node_block = self._inter.slot_tensor(
                block_count, input_data.dtype, input_data.device
            )
            # fuse_barrier: the inter ring's prepare_sync_in_place
            # ShmemBarrierAll follows immediately, covering the dropped barrier.
            #: also drop the entry barrier from the 2nd op onward.
            self._intra(
                input_data,
                node_block,
                count,
                stream,
                barrier=not self.fuse_barrier,
                prepare_barrier=intra_prepare_barrier,
            )
            # Phase 2 (inter, RDMA ring): all-gather the N node-blocks across
            # nodes, laid down in node order -> the full rank-major output.
            if self.out_in_place:
                # M4: leave the result in the ring buffer (read it via
                # result_tensor); skip the finish_sync copy-OUT. ZERO staging on
                # either side (copy-IN already dropped by chunk_in_place).
                self._inter(
                    node_block,
                    output_data,
                    block_count,
                    stream,
                    chunk_in_place=True,
                    out_in_place=True,
                )
            else:
                self._inter(
                    node_block, output_data, block_count, stream, chunk_in_place=True
                )
            self._prev_op_completed = True
            return True

        if (
            self._node_block is None
            or self._node_block.numel() < block_count
            or self._node_block.dtype != input_data.dtype
            or self._node_block.device != input_data.device
        ):
            self._node_block = torch.empty(
                block_count, dtype=input_data.dtype, device=input_data.device
            )
        node_block = self._node_block[:block_count]
        # fuse_barrier: drop the intra finish barrier only on the
        # every-rank-direct path, where the inter ring's prepare_sync barrier
        # follows immediately. Leader-only keeps the barrier (unchanged).
        intra_barrier = not (self.fuse_barrier and not self.leader_only)
        self._intra(
            input_data,
            node_block,
            count,
            stream,
            barrier=intra_barrier,
            prepare_barrier=intra_prepare_barrier,
        )

        if not self.leader_only:
            # Phase 2 (inter, RDMA ring): staged path (default) -- prepare_sync
            # copies node_block into the ring slot, then the ring all-gathers the
            # N node-blocks across nodes in node order -> full rank-major output.
            self._inter(node_block, output_data, block_count, stream)
            self._prev_op_completed = True
            return True

        # Leader-only: phase 2 ring (leaders only) -> phase 3 SDMA broadcast.
        full_count = count * self.npes
        if self.local_rank == 0:
            # Leader rings the N node-blocks across nodes into output_data
            # (the full rank-major result).
            self._inter(node_block, output_data, block_count, stream)
        else:
            # Non-leader: degenerate singleton ring on scratch only to take part
            # in the two collective ShmemBarrierAll calls (no real data move).
            if (
                self._ring_scratch is None
                or self._ring_scratch.numel() < block_count
                or self._ring_scratch.dtype != input_data.dtype
                or self._ring_scratch.device != input_data.device
            ):
                self._ring_scratch = torch.empty(
                    block_count, dtype=input_data.dtype, device=input_data.device
                )
            self._inter(
                node_block, self._ring_scratch[:block_count], block_count, stream
            )

        # Phase 3 (intra, SDMA broadcast): leader (root, group_pos 0) fans its
        # full N*G output to the G local ranks over XGMI. Non-root members get
        # the result here; the root's output is overwritten with identical data.
        self._bcast(output_data, output_data, full_count, stream)
        self._prev_op_completed = True
        return True

    def supports_param_contiguous_output(self) -> bool:
        """True when the direct-to-output PARAM-CONTIGUOUS zero-copy path is
        available for this instance (cross-node, slice_direct over RDMA). The
        FSDP adapter probes this to decide whether it can skip its copy-OUT.
        """
        return bool(
            self.num_nodes >= 2
            and self.slice_inter
            and self.slice_direct
            and self.stream_intra
            and self.stream_ring
        )

    def enqueue_param_contiguous(
        self,
        input_data,
        output_data,
        count: int,
        split_sizes,
        split_offsets,
        stream=None,
    ) -> bool:
        """PARAM-CONTIGUOUS zero-copy AllGather (kills the FSDP copy-OUT).

        Motivation: cross-node FSDP2 loses to RCCL only because the copy-out
        HierAllGather forces the backend to reshuffle rank-major -> param-
        contiguous on every per-layer gather. This writes the gathered result
        straight into ``output_data`` in PARAM-CONTIGUOUS layout: for global
        rank ``r`` and param ``s`` (per-rank elems ``E_s`` at input offset
        ``O_s``), rank ``r``'s slice lands at ``O_s*W + r*E_s`` -- exactly what
        FSDP's packed all-gather expects, so no copy-OUT is needed.

        ``split_sizes[s]`` / ``split_offsets[s]`` are in INPUT-DTYPE elements
        (``E_s`` and ``O_s``); their byte extents must be 4-byte aligned (SDMA).
        Returns False (caller must fall back to copy-OUT ``__call__``) when the
        direct param-contiguous path is unavailable for this instance.

        Implementation reuses the proven slice_direct primitives with NO new
        C++ kernel: Phase A rings each rank's own shard into ``collection``;
        Phase B PUSHES, per (node-block m, param s), the E_s-element sub-slice
        via the existing per-slot ``gather_kernel_direct`` with
        ``dst_block_offset = O_s*W + m*G*E_s`` and ``dst_slot_stride = E_s`` so
        local member g (global rank r = m*G+g) writes to ``O_s*W + r*E_s``.
        """
        if not self.supports_param_contiguous_output():
            return False

        W = self.npes
        N = self.num_nodes

        ss = split_sizes.tolist() if torch.is_tensor(split_sizes) else list(split_sizes)
        so = (
            split_offsets.tolist()
            if torch.is_tensor(split_offsets)
            else list(split_offsets)
        )
        if len(ss) != len(so):
            raise ValueError("split_sizes and split_offsets must have equal length")
        elem = input_data.element_size()
        # SDMA needs 4-byte-aligned byte extents; the adapter pads params to
        # honor this, but guard here so a bad layout falls back to copy-OUT
        # rather than corrupting output.
        for E, off in zip(ss, so):
            if (E * elem) % 4 != 0 or (off * elem) % 4 != 0:
                return False

        slice_total = count * N
        if (
            self._slice_scratch is None
            or self._slice_scratch.numel() < slice_total
            or self._slice_scratch.dtype != input_data.dtype
            or self._slice_scratch.device != input_data.device
        ):
            self._slice_scratch = torch.empty(
                slice_total, dtype=input_data.dtype, device=input_data.device
            )
        collection = self._slice_scratch[:slice_total]

        # Register the user output for the direct SDMA push (lockstep, LRU-cached).
        out_ptr = output_data.data_ptr()
        out_size = output_data.numel() * output_data.element_size()
        if self._reg_cache_cap <= 1:
            _reg_changed = (out_ptr, out_size) != (
                self._direct_reg_ptr,
                self._direct_reg_size,
            )
        else:
            _reg_changed = (out_ptr, out_size) not in self._direct_reg_lru
        self._ensure_output_registered(output_data)
        # DIAGNOSTIC (MORI_HIER_REG_STATS=1): count how often the output ptr
        # CHANGES across per-layer AG calls. Each change is a cross-node COLLECTIVE
        # (deregister+register ShmemSymmetric*) that RCCL never pays and that cannot
        # overlap -> a candidate for the in-FSDP per-AG inflation. If steady state
        # shows ~0 changes/call, registration churn is NOT the bottleneck.
        if os.environ.get("MORI_HIER_REG_STATS", "0") in ("1", "true", "True"):
            self._reg_calls = getattr(self, "_reg_calls", 0) + 1
            if _reg_changed:
                self._reg_changes = getattr(self, "_reg_changes", 0) + 1
            if self._reg_calls % 100 == 0:
                sys.stderr.write(
                    "[MORI_HIER_REG_STATS] calls=%d reg_changes=%d (%.1f%% of calls "
                    "trigger a cross-node register collective)\n"
                    % (
                        self._reg_calls,
                        getattr(self, "_reg_changes", 0),
                        100.0 * getattr(self, "_reg_changes", 0) / self._reg_calls,
                    )
                )
                sys.stderr.flush()

        # Split geometry in u32 lanes (SDMA byte move); the 4-byte alignment guard
        # above makes these conversions exact. CACHE the u32 GPU tensors keyed by
        # the split geometry: rebuilding them (torch.tensor + H2D) on every
        # per-layer all-gather added host overhead per call. FSDP reuses the same
        # split geometry across a param group, so steady state hits the cache.
        u32 = 4
        blk_stride_u32 = (count * elem) // u32
        _u32_key = (tuple(ss), tuple(so), elem, str(input_data.device))
        if getattr(self, "_pc_u32_key", None) != _u32_key:
            self._pc_u32_ss = torch.tensor(
                [(E * elem) // u32 for E in ss],
                dtype=torch.int64,
                device=input_data.device,
            )
            self._pc_u32_so = torch.tensor(
                [(off * elem) // u32 for off in so],
                dtype=torch.int64,
                device=input_data.device,
            )
            self._pc_u32_key = _u32_key
        split_sizes_u32 = self._pc_u32_ss
        split_offsets_u32 = self._pc_u32_so
        entry_barrier = not self.slice_fuse_ib

        # OVERLAPPED param-contiguous zero-copy (the lever to beat RCCL): the
        # LOCAL node-block (m == node_id) scatter reads only THIS rank's own input
        # (no ring dependency) so it runs on a SIDE stream concurrently with the
        # inter-node RDMA ring -- exactly the ring||gather overlap the copy-OUT
        # __call__ path uses, but writing PARAM-CONTIGUOUS straight into the user
        # output (no copy-OUT). The serial Phase-A-then-scatter path forwent this
        # overlap and lost to RCCL (99.7 vs 127 TFLOPS); this recovers it.
        # OPT-IN (default OFF): the overlap path is bit-exact in the standalone
        # 2-node test but currently triggers an HSA memory-exception under FSDP's
        # repeated-call / buffer-reuse pattern (side-stream local scatter). Ship
        # the proven non-overlap fused scatter as the default zero-copy path;
        # enable overlap with MORI_HIER_PC_OVERLAP=1 to iterate on the fault.
        overlap = (
            self.stream_intra
            and self.stream_ring
            and self.slice_direct
            and N >= 2
            and not entry_barrier
            and os.environ.get("MORI_HIER_PC_OVERLAP", "0") in ("1", "true", "True")
        )
        if overlap:
            node = self.node_id
            if self._overlap_stream is None:
                self._overlap_stream = torch.cuda.Stream(device=input_data.device)
            side = self._overlap_stream
            main = (
                torch.cuda.current_stream(input_data.device)
                if stream is None
                else stream
            )
            # Ring prepare = global entry barrier + copy-IN of this rank's shard.
            args, u32c, s_main = self._inter.prepare_stream_only(
                input_data, count, stream
            )
            # Side stream observes the entry barrier, then scatters the LOCAL
            # block (r = node*G+g) barrier-free, concurrent with the ring. Source
            # is this rank's own input (one block); first_block=node maps it to
            # global ranks node*G..node*G+G-1.
            side.wait_stream(main)
            self._intra.gather_kernel_direct_param_contiguous(
                input_data,
                output_data,
                blk_stride_u32,
                1,
                W,
                split_sizes_u32,
                split_offsets_u32,
                stream=side,
                prepare_barrier=False,
                first_block=node,
            )
            # INPUT/OUTPUT LIFETIME across the side stream (the loss-drift race):
            # input_data and output_data are produced/freed by
            # FSDP on the MAIN stream, but the local-block scatter above READS
            # input_data and WRITES output_data on the SIDE stream. The torch
            # caching allocator only tracks the free-stream (main); without
            # record_stream it may recycle these blocks while the side kernel is
            # still draining -> a per-call nondeterministic corruption that shows
            # up as run-to-run loss drift (12.617/12.566). main.wait_stream(side)
            # below orders the KERNELS but does NOT inform the allocator about the
            # side-stream use. Record it so the block is not reused until the side
            # scatter completes.
            if hasattr(input_data, "record_stream"):
                input_data.record_stream(side)
                output_data.record_stream(side)
            # Ring kernel + finish copy-OUT into collection (main), overlapping
            # the side local-block scatter.
            self._inter.launch_finish_stream(
                args,
                collection,
                u32c,
                s_main,
                barrier=not self.slice_defer_inter_fin,
            )
            # Merge the side local-block scatter BEFORE issuing the remote-block
            # scatters. The local (side) and remote (main) scatters BOTH call
            # gather_kernel_direct_param_contiguous, which shares one per-groupPos
            # flag slot + seq token on this handle; running them concurrently made
            # a receiver observe the other scatter's flag bump -> premature
            # completion / spin-deadlock under FSDP (observed hang). Serialize the
            # two scatter phases here. The KEY overlap (side local scatter || the
            # inter-node RDMA ring) is PRESERVED: the ring's launch_finish_stream
            # was already enqueued on main above and runs concurrently with the
            # side scatter; only the ring-DEPENDENT remote scatters wait.
            main.wait_stream(side)
            # Remote node-blocks read the ring collection; scatter each into the
            # param-contiguous output (r = m*G+g).
            for m in range(N):
                if m == node:
                    continue
                self._intra.gather_kernel_direct_param_contiguous(
                    collection[m * count : (m + 1) * count],
                    output_data,
                    blk_stride_u32,
                    1,
                    W,
                    split_sizes_u32,
                    split_offsets_u32,
                    stream=stream,
                    prepare_barrier=False,
                    first_block=m,
                )
            self._intra.finish_direct_stream(
                stream=stream, barrier=not self.slice_defer_fin
            )
            self._prev_op_completed = True
            return True

        # Non-overlapped fallback: serial Phase A ring then ONE fused scatter.
        self._inter(
            input_data,
            collection,
            count,
            stream,
            stream_ring=self.stream_ring,
            defer_inter_fin=self.slice_defer_inter_fin,
        )
        self._intra.gather_kernel_direct_param_contiguous(
            collection,
            output_data,
            blk_stride_u32,
            N,
            W,
            split_sizes_u32,
            split_offsets_u32,
            stream=stream,
            prepare_barrier=entry_barrier,
        )
        self._intra.finish_direct_stream(
            stream=stream, barrier=not self.slice_defer_fin
        )
        self._prev_op_completed = True
        return True

    def result_tensor(self, count: int, dtype, device=None):
        """Torch view of the gathered result when ``out_in_place`` is enabled.

        In out-in-place mode ``__call__`` leaves the full rank-major result in
        the inter-node ring buffer instead of copying it to a user output
        (eliminating the finish_sync copy-OUT). Read it from here. ``count`` is
        the per-rank element count; the returned view has ``count * npes``
        elements. Only valid for the every-rank-direct N>=2 path with
        ``out_in_place=True``.
        """
        if not self.out_in_place:
            raise RuntimeError("result_tensor is only valid when out_in_place=True")
        if self.num_nodes < 2:
            raise RuntimeError(
                "result_tensor is only valid for the N>=2 hierarchical path"
            )
        block_count = count * self.ranks_per_node
        return self._inter.full_tensor(block_count, dtype, device)

    def get_output_transit_buffer(self, dtype=None, device=None):
        if self.num_nodes == 1:
            return self._intra.get_output_transit_buffer(dtype=dtype, device=device)
        raise NotImplementedError(
            "HierAllGather inter-node path writes directly to the user output; "
            "no transit-buffer view is exposed."
        )
