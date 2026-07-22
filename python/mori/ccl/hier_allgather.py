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

Default path (no ``MORI_HIER_*`` env set)
-----------------------------------------
With no env overrides ``HierAllGather`` runs the sliced 2-D path
(``slice_inter``) with fused Phase-B (``slice_fused``), stream-ordered
ring/intra barriers (``stream_ring``/``stream_intra``), deferred finish fences
(``slice_defer_fin``/``slice_defer_inter_fin``), the serial ``slice_direct``
reassembly gather, and CU-domain copy-out. At
``ranks_per_node >= 8`` the two cross-PE finish fences are forced ON for
bit-exactness (see ``_apply_dense_node_defaults``). The fused
``ring || local-gather`` kernel (``fuse_local``) is off by default: under
back-to-back FSDP overlap it can read stale remote halves, so the serial path
is the supported one. Each ``MORI_HIER_*`` flag is an opt-in override; see
``examples/fsdp_sdma/README.md`` for the flag table.
"""

import os
import socket
from typing import List, Optional, Sequence

import torch

# MI300X SDMA channel cap: the per-GPU SDMA queue-slot count caps
# MORI_SDMA_NUM_CHANNELS at 8; a larger value aborts at SDMA queue creation
# (hsaKmtCreateQueueExt, queue-slot exhaustion). anvil reads this env at shmem
# init, before any HierAllGather is constructed, so the guard must run at import
# time (on `import mori.ccl...`, ahead of shmem.init()). It only rewrites values
# >8 (which would abort the process); every valid config is left untouched.
MORI_SDMA_CH_HW_MAX = 8

# Hier-barrier slot capacity (fail-closed topology guard bounds).
# The C++ hier barrier ``ShmemInternalBarrierHierBlock``
# (include/mori/shmem/shmem_device_api.hpp) packs its rendezvous flags into the
# fixed 128-uint64 internalSync region with this disjoint layout:
#   HIER_PEER_BASE [96 .. 96+num_nodes)          coordinator inter-node inbox
#   HIER_LOCAL_BASE[112 .. 112+ranks_per_node)   local PE -> coordinator inbox
#   HIER_REL_SLOT  [120], HIER_GEN_SLOT [126]
# The local inbox must fit within [112, 120) => ranks_per_node <= 8, and the
# coordinator inbox within [96, 112) => num_nodes <= 16. Exceeding either bound
# silently overwrites an adjacent slot (the REL/GEN slots or the local inbox),
# corrupting the barrier. The guard fails closed with an actionable error before
# such a topology reaches the kernel.
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


# Falsy-string set for every MORI_HIER_* boolean env flag: a value is "true"
# iff it is not one of these. A module constant so the parse is identical
# everywhere.
_ENV_FALSE = ("0", "", "false", "False")


def _env_true(key: str, default: str = "0") -> bool:
    """Return True unless the env flag is one of the falsy strings in _ENV_FALSE."""
    return os.environ.get(key, default) not in _ENV_FALSE


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
    all_gather.hpp``) launched over the ``N`` node-leaders. Each leader
    contributes one node-block (the ``G``
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
        slice_min_bytes: Optional[int] = None,
        slice_fuse_ib: Optional[bool] = None,
        slice_pipe_chunks: Optional[int] = None,
        slice_direct: Optional[bool] = None,
        standalone_fast: bool = False,
    ):
        # Fan the inter-node RDMA ring put across this many QPs to fill the NIC
        # (default 4 = provisioned count). The kernel fans out only for true
        # cross-node (RDMA) neighbours, so single-node runs stay single-warp.
        inter_num_qp = inter_num_qp if inter_num_qp is not None else 4
        self.inter_num_qp = max(1, inter_num_qp)
        # Single-block inter-node ring: per-NIC RDMA throughput is saturated at
        # numQp>=4, so one working block per RDMA neighbour.
        inter_num_blocks = inter_num_blocks if inter_num_blocks is not None else 1
        self.inter_num_blocks = max(1, inter_num_blocks)
        # Opt-in leader-only pipeline; default every-rank-direct. See the N>=2 branch.
        self.leader_only = False if leader_only is None else bool(leader_only)
        # Opt-in "gather-in-place": the intra-node SDMA gather writes its
        # node-block directly into the inter-node ring slot, eliminating the
        # prepare_sync copy-IN (a full node-block D2D copy) and the node_block
        # intermediate. Default OFF: the staged path (intra -> node_block -> ring
        # slot) stays the default.
        self.gather_in_place = (
            False if gather_in_place is None else bool(gather_in_place)
        )
        # Opt-in "out-in-place": leave the gathered result in the inter-node ring
        # buffer and read it via ``result_tensor`` instead of copying it to a user
        # output. Implies gather_in_place (the gather writes straight into the ring
        # slot) so there is zero staging on either side. Default OFF: the staged
        # path (writes the user output) stays the default -- out-in-place changes
        # the result-delivery contract (read ``result_tensor``).
        self.out_in_place = False if out_in_place is None else bool(out_in_place)
        # "fuse-barrier": drop the intra-node SDMA gather's finish ShmemBarrierAll
        # in the every-rank-direct N>=2 path (removes 1 of 4 global barriers/op).
        # Correctness invariant: the dropped intra finish barrier is redundant --
        # the PUSH gather's in-kernel flag-wait already makes this PE's node-block
        # complete on kernel return, and the inter ring's prepare_sync
        # ShmemBarrierAll immediately follows to synchronize all PEs before the
        # ring's cross-PE atomics; flags are monotonic per-call (no reset) so there
        # is no cross-call flag hazard. Crash-safe via the _prev_op_completed guard
        # (keeps the entry barrier on first op / after any mid-pipeline exception).
        # On (bit-exact); applies only to the every-rank-direct path (not leader-only).
        self.fuse_barrier = True if fuse_barrier is None else bool(fuse_barrier)
        # Sliced 2-D AllGather (the primary bandwidth path). The non-sliced
        # every-rank-direct path has each of the G local ranks push its full
        # node-block (G*count) to its same-index peer, so node n's block crosses
        # the boundary G times => per-NIC inter bytes = G*count. The sliced path
        # closes that G x gap without the single-NIC funnel of leader-only:
        #   1. Inter ring over same-local-index peers {g, g+G, ...} where each rank
        #      contributes only its OWN shard (count, not the G*count node-block).
        #      Because slice_g(B_n) == shard[n*G+g] == this rank's own input, the
        #      ring yields C_g = [slice_g(B_0), ..., slice_g(B_{N-1})] in node
        #      order. Per-NIC inter bytes drop to (N-1)*count -- a G x cut, spread
        #      across all G NICs (no funnel).
        #   2. N intra-node SDMA gathers (one per node-block m) reassemble full
        #      B_m = concat_g slice_g(B_m) into output[m*block:(m+1)*block]. The
        #      SDMA gather concatenates by group_pos=local_rank, so the result is
        #      exactly rank-major concat(B_0..B_{N-1}) == torch all_gather.
        # The extra intra gather rides fast XGMI while the inter phase shrinks ~G x.
        # Default ON. It owns its own inter+intra data path, so it is incompatible
        # with leader_only / out_in_place / gather_in_place; if any of those is
        # explicitly enabled, slice defaults OFF so those levers still work.
        _slice_conflict = self.leader_only or self.out_in_place or self.gather_in_place
        self.slice_inter = (
            bool(slice_inter)
            if slice_inter is not None
            else not _slice_conflict
        )
        # Fused sliced Phase B: fold the N intra reassembly gathers into one batch.
        # Correctness invariant: the fused path stacks the N gathers into disjoint
        # regions of one enlarged transit (dst_base_offset = m*block) so they never
        # overlap; it keeps only the m==0 entry barrier + one bulk copy-OUT + one
        # exit barrier, and flags stay monotonic per-call, so there is no
        # cross-gather race and the output is byte-identical to the unfused sliced
        # path. Only meaningful with slice_inter.
        self.slice_fused = bool(slice_fused) if slice_fused is not None else True
        # Per-call size threshold for the sliced path. The sliced 2-D path wins at
        # large per-rank payloads but loses at small/mid (its extra kernel launches
        # + N reassembly gathers cost more than the saved inter bytes), so engage
        # slice only when the per-rank payload is >= this many bytes; below it, fall
        # through to the non-sliced path. Default 8 MiB.
        slice_min_bytes = (
            slice_min_bytes if slice_min_bytes is not None else 8 * 1024 * 1024
        )
        self.slice_min_bytes = max(0, slice_min_bytes)
        # MID/SMALL-SIZE BAND -> stream pipe-overlap path. Per-rank payloads below
        # slice_min_bytes route to the stream-ordered chunked-ring pipeline overlap
        # path (chunked side-stream gathers hidden under the ring, all barriers
        # on-device), faster than the non-sliced path across the sub-threshold band.
        # At/above slice_min the slice_direct path wins.
        # Drop the redundant Phase-B entry barrier in the sliced+fused non-overlap
        # path. Correctness invariant: that path runs the inter ring first; its
        # finish_sync issues a global ShmemBarrierAll immediately before the
        # Phase-B m==0 gather's own entry barrier -- two back-to-back all-PE
        # barriers with no remote memory op between them, so the second is
        # redundant (every PE has passed the inter finish barrier before any PE
        # starts Phase-B, so every peer's out_ transit from the previous op is
        # already free). Byte-identical output. Does not apply to the overlap path
        # (its local gather runs concurrently on a side stream and must keep its own
        # entry barrier).
        self.slice_fuse_ib = bool(slice_fuse_ib) if slice_fuse_ib is not None else True
        # Number of element-range chunks K for the chunked-ring pipe-band path:
        # each node-block's Phase-B reassembly gather is split into K chunks
        # (dst_slot_stride=count), byte-identical to the unchunked gather. Feeds
        # the mid-band pipe-overlap path (see use_pipe_band). Default 2.
        slice_pipe_chunks = slice_pipe_chunks if slice_pipe_chunks is not None else 2
        self.slice_pipe_chunks = max(1, slice_pipe_chunks)
        # Stream-ordered inter ring. Replaces the inter ring's host-blocking
        # prepare/finish (hipStreamSynchronize + host bootNet ShmemBarrierAll)
        # with the on-device ShmemBarrierOnStream prepare/finish, removing 2
        # CPU<->GPU round-trips per inter ring op. Only affects the slice path.
        self.stream_ring = True
        # Stream-ordered Phase-B finish_batch. Replaces the fused sliced Phase-B's
        # finish_batch (bulk copy-OUT + host hipStreamSynchronize + host
        # ShmemBarrierAll) with finish_batch_stream (ShmemBarrierOnStream) so,
        # paired with stream_ring, the whole op (inter ring + Phase-B gathers +
        # copy-OUT) runs fully on-stream with no host stall.
        self.stream_intra = True
        # Defer the Phase-B finish_batch_stream fence. The default fused-stream
        # slice op issues 3 on-stream global ShmemBarrierOnStream fences/op: inter
        # prepare (#1), inter finish (#2), Phase-B finish (#3). #3 (end of op i) is
        # back-to-back -- across the op boundary -- with the next op's #1 (inter
        # prepare), with no remote memory op between them on the stream. #1 already
        # globally fences (all PEs) after op i's copy-OUT and before any peer reuses
        # the shared transit/ring buffers, so #3 is redundant for every op followed
        # by another hier op. Safe because: (a) the copy-OUT is stream-ordered so
        # this PE's output is correct without #3; (b) cross-PE buffer reuse is
        # covered by the successor op's #1 (slice path) or its forced intra entry
        # barrier (non-slice path: the size-dispatcher resets _prev_op_completed on
        # a path switch -> entry barrier fires); (c) the last op (no successor)
        # needs no reuse fence and its output is already stream-correct. Only the
        # default fused non-overlap, non-pipe, non-oop slice path defers. On by
        # default (dropped at ranks_per_node>=8, see _apply_dense_node_defaults).
        self.slice_defer_fin = True
        # Defer the inter ring's finish_stream fence (the stream-ordered
        # ShmemBarrierOnStream guarding cross-PE ring-buffer reuse) to the next
        # slice op's prepare_stream barrier. Safe because: the ring buffer is
        # reused only by another op through this same _inter handle, and
        # prepare_stream always fences (global, on-stream) before its ring kernel
        # issues the peer RDMA puts, so the successor's prepare fence already orders
        # cross-PE reuse; the copy-OUT into the scratch collection stays
        # stream-ordered (Phase B reads a correct collection regardless); the last
        # op (no successor) reuses nothing; the size-dispatcher's path switch resets
        # _prev_op_completed, forcing an entry barrier. Same safety class as
        # slice_defer_fin (Phase-B). Only on the non-oop slice path (stream_ring).
        # On by default (dropped at ranks_per_node>=8, see _apply_dense_node_defaults).
        self.slice_defer_inter_fin = True
        # Fused ring || local-block gather. Runs the RDMA ring and the local-block
        # SDMA gather in one kernel launch (FusedRingLocalGatherKernel_u32: blocks
        # [0,num_blocks) = ring, last block = local gather), so the overlap is
        # intrinsic to the grid with no host merge. Engaged only on the default
        # fused stream-ordered slice_direct path.
        #
        # OFF by default: under FSDP's tight back-to-back overlap the fused kernel
        # produces stale remote-half output -- the RDMA-ring buffer is read out
        # (finish_ring_stream copy + remote-block direct gathers) before the
        # concurrently-launched ring CTA's remote puts are globally visible to
        # those readers. The serial monolithic ring path (fuse_local OFF) does not
        # have this race, so it is the default until the fused kernel's
        # ring-completion visibility to the finish readers is fixed on-device.
        #
        # The standalone AllGather (no back-to-back FSDP overlap) does not trigger
        # the race and is bit-exact with fuse_local ON. A standalone caller opts in
        # via standalone_fast=True without touching the E2E default; explicit env
        # always overrides.
        if "MORI_HIER_FUSE_LOCAL" in os.environ:
            self.fuse_local = _env_true("MORI_HIER_FUSE_LOCAL")
        else:
            self.fuse_local = bool(standalone_fast)
        # Remember the standalone gate so the standalone fast path can auto-engage
        # its bit-exact-safe fill/overlap levers below without touching any
        # FSDP/E2E caller (none pass standalone_fast).
        self._standalone_fast = bool(standalone_fast)
        # Standalone fast-path fill (see _apply_standalone_fast_defaults). The only
        # unconditionally-defaulted lever is more per-peer SDMA fill (default
        # MORI_SDMA_NUM_CHANNELS=8, a bit-exact-safe reassembly widening);
        # fuse_remote/deep_pipe engage only within the mlx5 NIC-DMA->HBM coherence
        # window, because an uncaged window exposes a flag-beats-data race at large
        # sizes. The SDMA channel count is hardware-capped at 8 on MI300X; the >8
        # crash-guard lives at module import (_clamp_sdma_channels) because the
        # physical queue count is fixed by anvil at shmem init, before this __init__.
        self._apply_standalone_fast_defaults(standalone_fast)
        self._init_fused_ring_state()
        # DIRECT-TO-OUTPUT Phase B. With slice_direct the Phase-B gathers PUSH each
        # member's slice straight into the (registered) user output, eliminating
        # the finish_batch D2D copy of the whole result. Only engaged on the
        # default fused, non-overlap, non-pipe, non-oop, stream-ordered slice path.
        # CRASH-SAFETY INVARIANT: the direct path registers the USER output as a
        # symmetric buffer (ShmemSymmetricRegister). Over RDMA (true xnode) this
        # succeeds, but under single-node IPC (hipIpcGetMemHandle on an arbitrary
        # torch allocation) it HARD-ABORTS the process -- so it must stay OFF on
        # single-node (num_nodes == 1, IPC sim). Default: ON for true multi-node
        # (num_nodes >= 2, RDMA), OFF for single-node; an explicit arg wins. The
        # None sentinel defers the decision to the transport probe below.
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
        """Apply the standalone fast-path env defaults.

        Reads only the ``standalone_fast`` arg + os.environ; writes
        self._standalone_fast / self._standalone_defer_fin and (only when
        standalone_fast is set) the MORI_SDMA_NUM_CHANNELS / FUSE_REMOTE /
        DEEP_PIPE / DISSEM_BARRIER env defaults. No FSDP/E2E caller passes
        standalone_fast, so those paths are unaffected."""
        self._standalone_fast = bool(standalone_fast)
        if standalone_fast:
            if os.environ.get("MORI_SDMA_NUM_CHANNELS") is None:
                os.environ["MORI_SDMA_NUM_CHANNELS"] = str(MORI_SDMA_CH_HW_MAX)
            # Bit-exact-safe fuse_remote reassembly overlap: engage FUSE_REMOTE +
            # DEEP_PIPE=auto so the fused ring||reassembly kernel runs with the
            # auto-quiet send-CQ landing fence. The fixed 32MB deep-pipe window is
            # the coherence cage that keeps it bit-exact.
            if os.environ.get("MORI_HIER_FUSE_REMOTE") is None:
                os.environ["MORI_HIER_FUSE_REMOTE"] = "1"
            if os.environ.get("MORI_HIER_DEEP_PIPE") is None:
                os.environ["MORI_HIER_DEEP_PIPE"] = "auto"
            # Route the standalone path's cross-PE fences through the O(log n)
            # dissemination barrier. Identical global all-PE rendezvous semantics
            # (bit-exact; byte image and NIC-landing->reassembly-consume ordering
            # unchanged) but a ceil(log2 n) parallel critical path instead of the
            # PE0 funnel's ~2(n-1) serial hops. Default ON for the standalone_fast
            # path only; E2E paths keep dissem OFF. Env overrides either way.
            if os.environ.get("MORI_HIER_DISSEM_BARRIER") is None:
                os.environ["MORI_HIER_DISSEM_BARRIER"] = "1"
        # Standalone finish-barrier deferral: drop the per-op finish
        # ShmemBarrierOnStream, leaning on the successor op's entry barrier for
        # cross-PE ring reuse (bit-exact). standalone_fast path only; the E2E
        # finish fences stay ON.
        self._standalone_defer_fin = bool(standalone_fast)

    def _init_fused_ring_state(self):
        """Init fused ring / flag-token / reasm-deep-SQ / host-proxy-inter state."""
        # fuse_remote: pipeline the inter-node RDMA
        # ring with the remote-block XGMI reassembly. When ON (on the fuse_local
        # slice_direct path) the fused FusedRingRemoteGatherKernel runs the ring
        # and, per landed sub-range, pushes the remote-block SDMA straight from this
        # PE's ring buffer into the registered output -- no ring copy-OUT, no
        # whole-phase finish barrier, remote gather overlaps the still-in-flight
        # NIC ring. Invariant: only valid at num_nodes==2 (single ring round).
        self.fuse_remote = _env_true("MORI_HIER_FUSE_REMOTE", "0")
        self._chunk_ready_flags = None
        # Cross-size carryover guard: the exact DEEP_PIPE flag layout (slots, per-PE
        # count, pipe depth) the persistent buffer was last sized for. A layout
        # change forces a fresh zeroed buffer so stale per-sub-chunk landing state
        # can't leak into the next distinct size.
        self._chunk_ready_flags_layout = None
        # Intra reassembly deep-SQ: feed all owned reassembly channels' SDMA copies
        # back-to-back then drain once. Bit-exact; ON only on the fused path
        # (no-op on the plain N=2 single-gather path).
        self._reasm_deep_sq = 1 if (self.fuse_local or self.fuse_remote) else 0

    def _init_sync_drain_state(self):
        # Isolation probe: force full stream completion at op return.
        self._debug_sync = _env_true("MORI_HIER_DEBUG_SYNC", "0")
        # Deferred backward big-AG landing event; unset on the shipped path
        # (drain_deferbwd is a no-op stub unless an event is pending).
        self._deferbwd_event = None
        # CU-domain copy-out: a copy-engine hipMemcpyAsync write to ``output`` is
        # not coherent with the FSDP backward GEMM's later CU read under HIP
        # stream-ordering alone. The finish copies into a persistent scratch and a
        # torch elementwise (CU) kernel writes scratch -> output, so the producer
        # is a CU op (L2-coherent with the GEMM) without a host stall.
        self._cu_copyout_scratch = None

    def _apply_dense_node_defaults(self):
        # Dense-node (>=8 ranks/node) landing-fence fix. The Phase-B intra SDMA
        # gather reads a peer's ring-output ``collection`` over XGMI; with 8-way
        # local fan-out that read can observe the peer's block before its ring
        # finish is globally visible -- a cross-PE visibility race a host sync
        # cannot fix (peer-side, not local completion). At ranks_per_node >= 8
        # force the two cross-PE finish fences ON for bit-exactness; the gate
        # never fires at ==4 (byte-identical). The Phase-B entry barrier stays
        # dropped -- redundant once the finish fences run inline.
        if self.ranks_per_node >= 8:
            self.slice_defer_fin = False
            self.slice_defer_inter_fin = False
            # Dense-node signal-pipe: default DEEP_PIPE_QUIET=0 lands each sub-chunk
            # via the completion AMO on the data QP (RC in-order, flag never precedes
            # its bytes) instead of a per-sub-chunk send-CQ quiet drain. Bit-exact:
            # the 32MB DEEP_PIPE window caps engagement below the NIC->HBM coherence
            # window. Explicit MORI_HIER_DEEP_PIPE_QUIET wins.
            if "MORI_HIER_DEEP_PIPE_QUIET" not in os.environ:
                os.environ["MORI_HIER_DEEP_PIPE_QUIET"] = "0"

    def _build_subcollectives(
        self,
        my_pe,
        npes,
        input_buffer_size,
        output_buffer_size,
        copy_output_to_user,
    ):
        """Construct the intra/inter/bcast sub-collectives for this handle.

        Single-node AllgatherSdma vs the multi-node intra-gather + inter-ring
        [+ leader-only broadcast] pipeline, plus the deferred slice_direct
        transport probe. Reads only the listed ctor args and the already-set
        ``self.*`` topology/flag attrs; writes only ``self.*``."""
        if self.num_nodes == 1:
            # Single node -> a plain intra-node SDMA AllGather over all local ranks
            # is exactly the full AllGather.
            from .collective import AllgatherSdma

            self._intra = AllgatherSdma(
                my_pe,
                npes,
                input_buffer_size=input_buffer_size,
                output_buffer_size=output_buffer_size,
                copy_output_to_user=copy_output_to_user,
            )
        else:
            # Hierarchical pipeline (every-rank-direct decomposition). Every rank
            # runs two sub-group collectives and ends with the full rank-major
            # output -- no separate broadcast phase:
            #
            #   1. Intra-node SDMA gather over my node's G local ranks
            #      {node*G, ..., node*G+G-1} -> my node-block (G shards in
            #      local-rank order).
            #   2. Inter-node RDMA ring over my same-local-index peers across
            #      nodes {local, local+G, ..., local+(N-1)*G} -> all N
            #      node-blocks in node order = concat(shard[0..W-1]), the
            #      rank-major all_gather result.
            #
            # Because node n owns ranks [n*G, n*G+G) and the ring lays blocks
            # down in node order, the result is bit-exact vs
            # torch.distributed.all_gather_into_tensor.
            #
            # Each node-block crosses the NIC once per local rank (Gx redundant
            # inter-node traffic); the sliced 2-D path (slice_inter) removes it.
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
            # The fused sliced Phase B stacks all N reassembly gathers into one
            # transit, so it must hold the full N*G-shard output (== the inter ring
            # buffer size), not just a single G-shard node-block.
            if self.slice_inter and self.slice_fused:
                intra_bytes = max(intra_bytes, inter_bytes)

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
                # block over the NIC G times, but is simple and bit-exact.
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
                # Leader-only (opt-in): local_rank==0 rings over node-leaders
                # {0,G,2G,...} into a staging buffer, then SDMA-broadcasts the full
                # N*G output to its G local ranks over XGMI. Funnels all inter-node
                # traffic through the leader's single NIC plus a serial XGMI hop, so
                # it only helps when NICs < GPUs/node; default is every-rank-direct.
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
            # Scratch for the sliced path -- holds this rank's collection
            # C_g = [slice_g(B_0)..slice_g(B_{N-1})] (N*count) gathered by the
            # inter ring before the N intra reassembly gathers.
            self._slice_scratch = None
            # Lazy side stream for the slice-overlap lever. The local node-block
            # gather runs here concurrently with the inter ring.
            self._overlap_stream = None
            # Guard for the fuse-barrier entry-barrier skip. The intra-gather entry
            # barrier may be skipped only when the prior op ran to completion
            # (through its inter-finish ShmemBarrierAll, which guarantees every
            # peer's out_ transit is free before the next gather). A plain call
            # counter is not sufficient: if a prior op raised mid-pipeline -- after
            # the intra-gather dirtied out_ but before the inter-finish barrier --
            # a counter would still be >0 and the next op would wrongly skip the
            # entry barrier with a dirty buffer. So we track explicit
            # clean-completion: set False at entry, True only after a full
            # successful op. First call (and any post-crash call) therefore keeps
            # the barrier.
            self._prev_op_completed = False
            # Which path the previous op took (sliced vs non-sliced), so the
            # size-threshold dispatcher can force-keep the entry barrier on a path
            # switch (the fuse-barrier entry-skip assumes the prior op's barriers
            # freed the same buffers this path will reuse). None = no prior op.
            self._last_use_slice = None
            # Single-registration tracking for the direct-to-output Phase-B path
            # (slice_direct). The output buffer must be collectively registered
            # (ShmemSymmetricRegister all-gathers peer pointers + opens IPC
            # handles), so the register/deregister decision must be identical on
            # every PE. A per-rank "register if not already registered" decision
            # would drive a divergent collective: when torch's caching allocator
            # placed a new output whose range overlapped a prior (freed)
            # registration differently across ranks, the C++ overlap-eviction set
            # diverged -> mismatched #collective calls -> the peer-pointer
            # all-gather mis-aligned -> SDMA read the wrong peer's window. Instead
            # track exactly one live registration here and, only on an exact
            # (ptr,size) change, deregister the old + register the new. Exact
            # same-size buffer reuse is SPMD-consistent across ranks, so this
            # decision is lockstep-uniform without any extra collective.
            self._direct_reg_ptr = None
            self._direct_reg_size = None
            # Multi-entry LRU registration cache. The single-entry tracker above
            # deregisters the old buffer on every (ptr,size) change, so two
            # alternating output buffers (e.g. the embed-grad and lm_head-grad big
            # backward AGs, which use different unsharded param buffers) each pay
            # dereg(old)+reg(new) = two cross-node collectives per call. Holding
            # K>=2 registrations lets both stay resident so steady state pays zero
            # register collectives. Keyed by exact (ptr,size); eviction is
            # deterministic (oldest-first) and the (ptr,size) sequence is identical
            # on every PE (FSDP issues the same AGs in the same order), so the
            # register/deregister collectives stay lockstep-uniform. Exact-match
            # hits only, so two resident entries are always distinct live buffers.
            # Insertion-ordered dict = the LRU (cap 4). Value = size (bookkeeping).
            self._reg_cache_cap = 4
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
            elif self.slice_direct:
                # Fail-closed buffer-registration-mode guard. An EXPLICIT
                # slice_direct=True (ctor arg) registers the user output as a
                # symmetric buffer
                # (ShmemSymmetricRegister) for the direct-to-output SDMA push.
                # That registration mode is only valid over RDMA: over a
                # P2P/IPC transport hipIpcGetMemHandle on an arbitrary torch
                # allocation HARD-ABORTS in C++ (see _probe_rdma_transport /
                # the slice_direct default comment above). The auto path
                # (slice_direct=None) already probes and defaults OFF for IPC;
                # when the caller forces it ON we validate the same
                # precondition and raise a clear, actionable error instead of
                # letting the opaque device abort fire. The default config
                # leaves slice_direct unset (auto), so this never trips it.
                if self.leader_only:
                    raise ValueError(
                        "slice_direct=True is incompatible with leader_only: "
                        "the direct-to-output SDMA push registers the user "
                        "output, but leader_only produces its result via the "
                        "copy-OUT ring path. Leave slice_direct unset (auto) "
                        "or disable leader_only."
                    )
                if not self._probe_rdma_transport():
                    raise ValueError(
                        "slice_direct=True requires an RDMA transport: it "
                        "registers the user "
                        "output via ShmemSymmetricRegister for direct-to-"
                        "output SDMA push, which hard-aborts "
                        "(hipIpcGetMemHandle) over a P2P/IPC transport such as "
                        "the single-node spawn simulation. Leave slice_direct "
                        "unset to auto-select the safe mode, or disable it for "
                        "IPC/single-node topologies."
                    )

    def drain_hostproxy(self):
        """Join any in-flight async host-proxy inter worker so ALL chunkReadyFlags
        for the AGs issued so far are published+landed. Called by the deferred FSDP
        consumer fence (async completion-ordering fix). No-op unless the
        async host-proxy producer is live. Shipped path has no async inter
        producer, so this is a no-op."""
        return False

    def drain_deferbwd(self):
        """Host-wait on the pending backward big-AG landing event, if any.

        Consumes the event (one-shot). No-op unless an event is pending; on the
        shipped path no event is recorded, so this never stalls."""
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
        """CU-domain copy-OUT for the nodirect Phase-B.

        Phase-B stacks the reassembled result into intra transit ``out_`` via
        SDMA; the receiver ``__threadfence_system`` makes those bytes coherent to
        a CU read but NOT to the copy engine, so an ``hipMemcpyAsync(out_ ->
        output)`` copy-out would race the SDMA writes. Do the copy-OUT as a single
        torch elementwise (CU) kernel instead: it reads ``out_`` with a coherent
        CU read and writes ``output`` in the CU/L2 domain the consumer GEMM reads
        -- no copy engine, no host stall. ``add`` by 0 is bit-exact for
        bf16/fp16/fp32 (x rounds to itself) and int dtypes.

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
        # Safety-net drain: if a deferred backward big-AG landing event survives
        # into the next AG call (external drain caller absent), drain it before
        # issuing the next op so the buffer is not reused before the prior AG's
        # remote bytes land. No event on the shipped path -> no-op.
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
        if stream is not None:
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
        # the op (small buffers). On bulk buffers static graph replay serializes the
        # CPU-driven inter/intra multi-launch overlap (one launch, but loses the
        # async ring||reassembly pipelining), so cap graph engagement to <=48MB
        # buffers; bulk sizes fall through to the normal fused fast path. Default ON;
        # set MORI_HIER_CUDA_GRAPH=0 to force the non-captured path.
        if (
            os.environ.get("MORI_HIER_CUDA_GRAPH", "1").strip().lower()
            in ("1", "true", "yes", "on")
            and not self._debug_sync
        ):
            _cg_max_mb = 48.0
            # Path-aware gate widening: the standalone fused path (standalone_fast +
            # fuse_remote/fuse_local) runs ring||reassembly inside one grid (no CPU
            # side-stream to serialize under replay), so capture is lossless at bulk
            # sizes -> drop the size cap; every other path keeps the 48MB gate.
            # Bit-exact by construction (identical kernels/order/buffers, copy on SDMA).
            if self._standalone_fast and (self.fuse_remote or self.fuse_local):
                _cg_max_mb = 0.0  # uncapped: single-grid path is lossless-capturable
            _cg_bytes = int(count) * int(input_data.element_size())
            if _cg_max_mb <= 0 or _cg_bytes <= _cg_max_mb * 1024 * 1024:
                if self._graph_replay(input_data, output_data, count, stream):
                    return True
        do_sync = self._debug_sync
        ret = self._call_impl(input_data, output_data, count, stream)
        if do_sync:
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

        # Size-threshold dispatch. Engage the sliced 2-D path only for large
        # per-rank payloads (where it wins); below the threshold the non-sliced
        # fuse-barrier path is faster. On a path switch, conservatively keep the
        # entry barrier (clear the clean-completion guard) since the two paths reuse
        # the shared _intra/_inter buffers differently.
        byte_count = count * input_data.element_size()
        use_slice = self.slice_inter and (byte_count >= self.slice_min_bytes)
        # mid/small band (below slice_min) routes to the chunked-ring
        # pipe-overlap path (sliced fused, non-oop, K>1 chunks).
        use_pipe_band = (
            (not use_slice)
            and self.slice_inter
            and self.slice_fused
            and self.slice_pipe_chunks > 1
        )
        # 3-way path key (None=non-slice, "pipe"=pipe-band, "slice"=slice path):
        # any switch reuses the shared _intra/_inter buffers differently, so
        # conservatively clear the clean-completion guard (forces an entry fence).
        path_key = "slice" if use_slice else ("pipe" if use_pipe_band else None)
        if path_key != self._last_use_slice:
            self._prev_op_completed = False
        self._last_use_slice = path_key

        if use_slice or use_pipe_band:
            # Sliced 2-D AllGather (see __init__). Phase A (inter, RDMA ring):
            # every rank rings only its own shard (count) across its
            # same-local-index peers {g, g+G, ...}. The ring gathers N chunks in
            # node order into C_g; because slice_g(B_n) == shard[n*G+g] == this
            # rank's own input, C_g == [slice_g(B_0)..slice_g(B_{N-1})]. Per-NIC
            # inter bytes = (N-1)*count (a G x cut vs the default G*count), spread
            # across all G NICs (no leader funnel).
            slice_total = count * N
            if (
                use_pipe_band
                and self.slice_fused
                and self.slice_pipe_chunks > 1
            ):
                # Chunked-ring pipeline overlap.
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
                # Uses the STREAM-ORDERED inter ring (stream_ring) + deferred
                # finish fence (defer_inter_fin): side gather of chunk k is hidden
                # under the main-stream ring of chunk k+1.
                # Correctness invariant: only ONE global on-stream fence is ever in
                # flight (the main-stream ring prepare); the side gathers run
                # barrier-free (prepare_barrier=False), so this is NOT the
                # concurrent-global-barrier race. Cross-chunk ring-buffer reuse is
                # ordered by chunk k+1's prepare_stream global fence (defer is safe
                # exactly as in the default non-chunked path).
                sr = self.stream_ring
                for k in range(K):
                    ck = base_ck if k < K - 1 else count - base_ck * (K - 1)
                    if ck == 0:
                        continue
                    region = collection[N * off : N * off + N * ck]
                    # Inter ring of chunk k on the MAIN stream. With stream_ring the
                    # finish copy-OUT into ``region`` is stream-ordered. The peer's
                    # chunk-k landing is fenced by the intra-node subgroup barrier
                    # on the first Phase-B gather (below), so the inter-ring finish
                    # is deferred under stream_ring.
                    _defer = sr
                    self._inter(
                        input_data[off : off + ck],
                        region,
                        ck,
                        stream,
                        stream_ring=sr,
                        defer_inter_fin=_defer,
                    )
                    # Make the side stream observe the ring's copy-OUT into ``region``.
                    side.wait_stream(main)
                    # CHEAP intra-node landing fence: arm ONLY the first gather of
                    # this chunk with the intra-node SUBGROUP entry ShmemBarrier
                    # (G ranks, XGMI-scope, no NIC quiet-drain / inter-node
                    # rendezvous). It orders all G local peers past their chunk-k
                    # ring copy-OUT + threadfence before any reads a peer ``region``;
                    # the remaining N-1 reads of the chunk are then safe.
                    for m in range(N):
                        self._intra.gather_kernel(
                            region[m * ck : (m + 1) * ck],
                            ck,
                            dst_base_offset=m * block_count + off,
                            stream=side,
                            prepare_barrier=(m == 0),
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
            # When the direct-path local-block overlap is active, the ring runs
            # interleaved inside Phase B (split into prepare_stream_only +
            # kernel/finish so the local-block gather can overlap the ring
            # kernel on a side stream), so skip the monolithic ring call here.
            # The fused path (fuse_local) likewise runs the ring inside Phase B
            # (as part of the fused kernel launch), so it must also skip it.
            overlap_active = (
                self.fuse_local
                and self.slice_direct
                and self.slice_fused
                and self.stream_intra
                and self.stream_ring
                and self.slice_fuse_ib
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
                # Fold the N gathers into one batch -- stack each into a disjoint
                # region [m*block, (m+1)*block) of the enlarged transit
                # (dst_base_offset = m*block_count), drop the per-gather finish
                # barrier/copy, then one bulk copy-OUT. Keep only the m==0 entry
                # barrier and the final exit barrier (2 barriers vs 2N). The inter
                # ring's finish barrier (run just above) already synchronizes all
                # PEs, so the m==0 entry barrier is redundant -- drop it when
                # slice_fuse_ib (default); keep it only if explicitly disabled.
                entry_barrier = not self.slice_fuse_ib
                if self.slice_direct and self.stream_intra and self.stream_ring:
                    # Direct-to-output Phase B. Register the user output once
                    # (collective, cached) then push each node-block's slices
                    # straight into output[m*block:] -- no internal transit, no
                    # full-output copy-OUT. Only a single global fence completes the
                    # op (deferrable like the copy-OUT path).
                    # Lockstep single-registration: register the user output
                    # collectively only on an exact (ptr,size) change, deregistering
                    # the previous one first so the C++ map holds at
                    # most one entry and never runs its (potentially per-rank
                    # divergent) overlap-eviction. Steady state (same output reused
                    # op-to-op) skips the collective entirely; a buffer change runs
                    # exactly {Dereg(old) if old; Reg(new)} uniformly on every PE.
                    self._ensure_output_registered(output_data)
                    if (
                        self.fuse_remote
                        and self.num_nodes == 2
                        and not entry_barrier
                    ):
                        # Fused-remote pipeline: one launch runs the ring and, per
                        # landed sub-range, the remote-block SDMA reassembly straight
                        # from this PE's ring buffer into the registered output (no
                        # ring copy-OUT, no whole-phase finish barrier). The remote
                        # gather of sub-range j overlaps ring channel j+1 still
                        # crossing the NIC, fusing the two serial phases.
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
                        # round(perPE_bytes / SUBBYTES); an explicit integer keeps the
                        # exact prior behavior. Default depth 2 matches the C++
                        # HierDeepPipe() default; the 32MB per-sub-chunk gate below
                        # self-cages the giant AG to the depth-1 fence so depth 2 stays
                        # E2E bit-exact with no explicit env.
                        _dp_raw = os.environ.get("MORI_HIER_DEEP_PIPE", "2").strip()
                        if _dp_raw.lower() == "auto":
                            # 16MiB sub-chunk target, matching the C++ auto selector:
                            # count*elsz is this PE's chunk, the same per-PE quantity
                            # the kernel gates on, so both compute the identical depth
                            # and the flag buffer sized here holds exactly the kernel's
                            # sub-chunk count. Stays under the 32MB coherence window.
                            _dp_sub_target = 16 * 1024 * 1024
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
                        # Per-sub-chunk coherence gate (32MB per-PE window): the device
                        # landing signal is bit-exact only while each temporal sub-chunk
                        # (chunkBytes/P) stays inside the MI300X/mlx5 NIC-DMA->HBM window;
                        # larger sub-chunks fall to the depth-1 fence. DEEP_PIPE=1 => inert.
                        # Mirrors the C++ HierDeepPipe gate.
                        _dp_chunk_bytes = int(count) * int(input_data.element_size())
                        _dp_window = 32 * 1024 * 1024
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
                        _flag_slots = max(rb, _deep_pipe if rb == 1 else 1, 1)
                        # Cross-size carryover fix: the persistent chunk-landing flag
                        # buffer must be reallocated on a layout change (per-PE count
                        # / slots / pipe depth), not only when it needs to grow.
                        # Reusing one buffer across two different DEEP_PIPE sizes in a
                        # single process (the UT sweep drives all sizes through one
                        # handle) leaves stale per-sub-chunk landing state that makes
                        # the 2nd distinct size bit-exact mismatch, even though every
                        # size isolated is clean. A fresh zeroed buffer per distinct
                        # layout (tiny, <=16 u64) removes it; same-size steady state
                        # still reuses + zeros with no per-call alloc, and DEEP_PIPE=1
                        # (default) never enters this block so the default path stays
                        # byte-identical.
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
                        flags.zero_()
                        # Ring prepare = global entry barrier + copy-IN (no launch).
                        ring_args, u32c, s_main = self._inter.prepare_stream_only(
                            input_data, count, stream
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
                        # Deadlock-free push-only reassembly: the remote blocks are
                        # push-only (each rank writes its own column into disjoint
                        # output slots, no cross-rank wait) with a single completion
                        # reader in the local-block CTA, so reasm>1 does not deadlock.
                        # Each reassembly block j uses
                        # SDMA queue qId=j, so to avoid racing the per-queue signal
                        # counter reasm MUST be <= the peer's SDMA queue count
                        # (sdmaNumQueue, default 2). Clamp accordingly. Flags use
                        # slots [0,G) (local) + [G, G+reasm*(N-1)*G) (reassembly),
                        # covered by the enlarged intra flags buffer.
                        # Reassembly blocks use SDMA queues [1, nq); queue 0 is
                        # taken by the concurrent local-block CTA. So max safe
                        # concurrent reassembly blocks == sdmaNumQueue-1. The
                        # per-peer SDMA queue count is MORI_SDMA_NUM_CHANNELS
                        # (anvil::GetSdmaNumChannels, default 2); use it so the
                        # SPATIAL reassembly split (rb==1, reasm>1) lands on
                        # distinct queues/engines instead of wrapping onto queue 0.
                        _sdma_nq = int(os.environ.get("MORI_SDMA_NUM_CHANNELS", "2"))
                        # INVARIANT (nq>=2 required): the fused-remote reassembly
                        # worker drives SDMA queue q = (j+1) % nq_physical (kernel
                        # ccl_kernels.hip qId=j+1) while the local-block CTA owns
                        # queue 0. At physical nq==1 the modulo aliases onto queue 0,
                        # so both share one per-queue signal counter -> intermittent
                        # liveness HANG. The physical queue count is fixed at shmem/
                        # anvil init (before this op and before HierAllGather
                        # __init__), so it can only be corrected by setting
                        # MORI_SDMA_NUM_CHANNELS>=2 up front; mori's library default
                        # (anvil::GetSdmaNumChannels) is already 2 (E2E/FSDP is safe).
                        # Default reasm to (nq-1) so the reassembly tail (queues
                        # [1, reasm]) fans across the spare SDMA engines via the
                        # existing SPATIAL split (rb==1, DP<=1, effReasm>1) when
                        # MORI_SDMA_NUM_CHANNELS is raised. Byte-identical at the
                        # default nq=2 (reasm clamps to 1); the E2E giant-AG path runs
                        # nq=2 so it is unaffected.
                        reasm = max(1, _sdma_nq - 1)
                        if reasm > _sdma_nq - 1:
                            reasm = _sdma_nq - 1
                        if reasm < 1:
                            reasm = 1
                        launch_fused_ring_remote_gather(
                            ring_args,
                            gather_args,
                            rb,
                            flags.data_ptr(),
                            N,
                            node,
                            s_main,
                            reassembly_blocks=reasm,
                            reasm_deep_sq=self._reasm_deep_sq,
                        )
                        # Single completion fence (no copy-OUT: gathers already
                        # pushed straight into the user output).
                        # Standalone finish-barrier deferral: at ranks_per_node>=8 the
                        # fused path is forced slice_defer_fin=False (dense-node E2E
                        # drifts without the two finish fences), so it issues a global
                        # cross-node ShmemBarrierOnStream every op. The device
                        # completion reader (the bx==rb CTA) has already spun until
                        # every remote push landed in this PE's output plus
                        # __threadfence_system, so this PE's output is stream-correct
                        # without the finish barrier. The finish barrier only adds
                        # cross-PE ring-buffer reuse safety for the next op, and the
                        # successor op's prepare_stream entry barrier (always global,
                        # before any copy-IN / peer RDMA put) already provides that
                        # ordering; the last op has no successor and reuses nothing. So
                        # deferring drops the per-op barrier count from 2 to 1 with
                        # byte-identical output. standalone_fast path only.
                        _crown_fin_barrier = not self.slice_defer_fin
                        if self._standalone_defer_fin:
                            _crown_fin_barrier = False
                        self._intra.finish_direct_stream(
                            stream=stream, barrier=_crown_fin_barrier
                        )
                    elif self.fuse_local and not entry_barrier:
                        # Fused ring || local-block gather in one kernel launch (NIC
                        # ring blocks [0,num_blocks) || XGMI local-block SDMA gather
                        # in the last block), a single concurrent grid with no host
                        # merge. The ring's prepare_stream barrier is the
                        # sole global entry fence; the local gather runs barrier-free
                        # (prepare_barrier=False), reading only this rank's own input
                        # (no ring dependency) and pushing block node_id straight into
                        # the registered output. The remote blocks (which depend on
                        # the ring) still follow as separate direct gathers after the
                        # ring copy-OUT.
                        from .collective import launch_fused_ring_local_gather

                        node = self.node_id
                        main = (
                            torch.cuda.current_stream(input_data.device)
                            if stream is None
                            else stream
                        )
                        # Ring prepare = global entry barrier + copy-IN (no kernel
                        # launch yet); returns the ring jit_args ptr.
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
                        # Ring finish copy-OUT into the collection scratch (the ring
                        # kernel already ran inside the fused launch -- do not
                        # relaunch it). Copy-engine finish (not the CU
                        # RingFinishCopyKernel) so the fuse_local path never moves
                        # bulk all-gather bytes on CUs.
                        # Copy-out elimination (standalone_fast): the
                        # finish_ring_stream copy-OUT is a copy-engine (SDMA) D2D read
                        # of the ring buffer. That copy-engine read is the fuse_local
                        # E2E stale-remote race: the ring CTA lands the remote half via
                        # NIC RDMA and does a CU-scope __threadfence_system, but the
                        # copy engine is a separate hw agent not ordered by that fence,
                        # so its D2D can drain stale remote-half ring bytes into
                        # ``collection``, which the reassembly then propagates. This
                        # lever drops the copy-OUT and points the remote reassembly
                        # SDMA read straight at the ring buffer (full_tensor view) --
                        # one fewer cross-agent hop and a saved ~(N-1)/N copy-out.
                        # Byte-identical by construction: ring slot m == collection[m].
                        # The trailing finish_direct_stream + the intra entry barrier
                        # on the first remote gather (default ON) keep the landing
                        # fence. ON for standalone_fast only (no cross-PE tight
                        # reuse, so the stale-remote race never triggers); every
                        # FSDP/E2E caller omits standalone_fast => OFF there.
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
                                cu_copyout=False,
                            )
                        # Remaining (remote) blocks read the ring collection.
                        # The remote reassembly is an intra-node subgroup gather --
                        # each local rank SDMA-pushes its own collection[m] slice to
                        # the G local peers over XGMI. That slice was produced by this
                        # rank's finish_ring_stream copy-OUT of the ring buffer. Under
                        # FSDP tight back-to-back reuse a peer rank's gather can read
                        # our collection over XGMI before our copy-OUT (and the
                        # concurrently-launched fused ring CTA's remote-half landing)
                        # is globally visible -- the stale-remote-half race that keeps
                        # fuse_local default-OFF. A full global cross-PE barrier is
                        # mutually exclusive with fuse_local,
                        # but the dependency here is purely intra-node (Phase-B reads
                        # only same-node peers' collection), so a single intra-node
                        # subgroup entry barrier on the first remote gather strictly
                        # orders every local rank's ring copy-OUT before any peer reads
                        # it -- on-device, XGMI-scope only, no NIC quiet-drain, no
                        # inter-node barrier.
                        remotes = [m for m in range(N) if m != node]
                        _first_remote = True
                        for m in remotes:
                            self._intra.gather_kernel_direct(
                                reasm_src[m * count : (m + 1) * count],
                                output_data,
                                count,
                                dst_block_offset=m * block_count,
                                stream=stream,
                                prepare_barrier=_first_remote,
                            )
                            _first_remote = False
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
                    self._cu_copyout_finish(output_data, N * block_count, stream)
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

        # fuse-barrier also drops the intra-gather entry barrier on the
        # every-rank-direct path, but only when the prior op completed cleanly (its
        # inter-finish barrier freed every peer's out_). The first op, and any op
        # following a mid-pipeline crash, keep the barrier (see __init__).
        prev_op_completed = self._prev_op_completed
        # Cleared until THIS op finishes; any exception below leaves it False so the
        # next op conservatively keeps the entry barrier.
        self._prev_op_completed = False
        intra_prepare_barrier = not (
            self.fuse_barrier and not self.leader_only and prev_op_completed
        )

        if not self.leader_only and self.gather_in_place:
            # gather_in_place (opt-in): write the intra-gather node-block directly
            # into this PE's inter-ring slot and run the ring with
            # chunk_in_place=True, removing the prepare_sync copy-IN and the
            # separate node_block intermediate. Perf-neutral vs the staged default
            # (the ring slot lives in the uncached symmetric heap, so the gather's
            # finish_sync write into it offsets the saved copy), so the default
            # stays the staged path; correctness is identical either way.
            node_block = self._inter.slot_tensor(
                block_count, input_data.dtype, input_data.device
            )
            # fuse_barrier: the inter ring's prepare_sync_in_place ShmemBarrierAll
            # follows immediately, covering the dropped barrier; also drop the entry
            # barrier from the 2nd op onward.
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
                # Leave the result in the ring buffer (read it via result_tensor);
                # skip the finish_sync copy-OUT. Zero staging on either side (copy-IN
                # already dropped by chunk_in_place).
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

        Implementation reuses the slice_direct primitives with NO new
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
        self._ensure_output_registered(output_data)

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

        # Serial Phase A ring then ONE fused param-contiguous scatter.
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
