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
Hierarchical cross-node AllGather: intra-node SDMA (XGMI) + inter-node RDMA ring.

Output matches ``torch.distributed.all_gather_into_tensor`` (rank-major) with
zero tolerance. ``num_nodes == 1`` degenerates to a pure intra-node SDMA
AllGather. Bit-exact because the SDMA path moves uint32 lanes regardless of
dtype: a 4-byte-multiple per-rank shard has the same byte layout as torch's
concatenation (bf16/fp16/fp32/int32).

Default path (no ``MORI_HIER_*`` env): sliced 2-D (``slice_inter``) + fused
Phase-B + stream-ordered ring/intra + deferred finish fences + serial
``slice_direct`` + CU-domain copy-out. At ``ranks_per_node >= 8`` the two
cross-PE finish fences are forced ON (see ``_apply_dense_node_defaults``). The
fused ``ring || local-gather`` kernel (``fuse_local``) is OFF: under
back-to-back FSDP overlap it can read stale remote halves. Each ``MORI_HIER_*``
flag is an opt-in override; flag table in ``examples/fsdp_sdma/README.md``.
"""

import os
import socket
from typing import List, Optional, Sequence

import torch

# MI300X SDMA queue-slot cap: MORI_SDMA_NUM_CHANNELS>8 aborts at queue creation.
# anvil reads this env at shmem init (before any HierAllGather), so clamp at
# import time, ahead of shmem.init().
MORI_SDMA_CH_HW_MAX = 8


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


# Falsy strings for every MORI_HIER_* boolean env flag (true iff not in this set).
_ENV_FALSE = ("0", "", "false", "False")


def _env_true(key: str, default: str = "0") -> bool:
    """Return True unless the env flag is one of the falsy strings in _ENV_FALSE."""
    return os.environ.get(key, default) not in _ENV_FALSE


def _auto_ranks_per_node(my_pe: int, npes: int) -> int:
    """Ranks co-located on this node (a positive divisor of ``npes``).

    Resolution: launcher local world size (LOCAL_WORLD_SIZE / MPI), else group
    ranks by hostname, else ``npes`` (single node).
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


# AllgatherSdma (and the compiled .so) is imported lazily inside __init__ so the
# pure-Python reference specs below stay importable/testable on CPU-only hosts.


def hier_allgather_reference(
    shards: Sequence["torch.Tensor"],
    num_nodes: int,
    ranks_per_node: int,
) -> List["torch.Tensor"]:
    """CPU executable spec of the hier AllGather offset arithmetic (bit-exact
    contract for the GPU path; no GPU/SDMA/RDMA).

    ``shards``: ``world = N*G`` per-rank inputs (same shape/dtype), indexed by
    global rank. Returns ``world`` outputs; entry ``r`` is rank ``r``'s full
    result. Result == rank-major ``concat(shard[0..world-1])`` because node ``n``
    owns ranks ``[n*G, n*G+G)`` and node-blocks are laid down in node order.
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

    node_blocks: List[torch.Tensor] = []
    for n in range(N):
        block = torch.empty(count * G, dtype=dtype)
        for g in range(G):
            src = shards[n * G + g].reshape(-1)
            block[g * count : (g + 1) * count] = src
        node_blocks.append(block)

    # Every rank lays the N node-blocks down in node order.
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
    """CPU executable spec of the inter-node ring AllGather (no RDMA), mirroring
    the ``AllGatherRingKernel`` schedule
    (include/mori/collective/inter_node/kernels/all_gather.hpp): ``N-1`` rounds
    of send ``(myPe-i)%N`` to ``(myPe+1)%N``. After the ring every leader holds
    all ``N`` chunks in node order. Per-round clone models the between-round
    flag/quiet barrier. Returns one (identical) buffer per node-leader.
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
        snapshot = [b.clone() for b in bufs]  # round reads start-of-round state
        for my_pe in range(N):
            next_peer = (my_pe + 1) % N
            send_rank = (my_pe - i + N) % N
            lo = send_rank * block_elems
            hi = lo + block_elems
            bufs[next_peer][lo:hi] = snapshot[my_pe][lo:hi]
    return bufs


class HierAllGather:
    """Hierarchical AllGather: intra-node SDMA + inter-node RDMA ring.

    ``npes == num_nodes * ranks_per_node``; ``ranks_per_node`` (``G``) is
    keyword-only and auto-detected when omitted (see ``_auto_ranks_per_node``).
    ``input_buffer_size``/``output_buffer_size`` (or a combined
    ``transit_buffer_size``) pre-allocate the SDMA transit for the largest
    ``count * dtype`` passed to ``__call__``. ``copy_output_to_user`` copies the
    result into the user output (required for cached PyTorch allocations).
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
        inter_num_blocks: Optional[int] = None,
        fuse_barrier: Optional[bool] = None,
        slice_inter: Optional[bool] = None,
        slice_fused: Optional[bool] = None,
        slice_min_bytes: Optional[int] = None,
        slice_fuse_ib: Optional[bool] = None,
        slice_pipe_chunks: Optional[int] = None,
        slice_direct: Optional[bool] = None,
    ):
        # Fan the inter-node RDMA ring put across this many QPs to fill the NIC
        # (default 4 = provisioned). Fans out only for true cross-node neighbours.
        inter_num_qp = inter_num_qp if inter_num_qp is not None else 4
        self.inter_num_qp = max(1, inter_num_qp)
        # One working block per RDMA neighbour: per-NIC throughput saturates at numQp>=4.
        inter_num_blocks = inter_num_blocks if inter_num_blocks is not None else 1
        self.inter_num_blocks = max(1, inter_num_blocks)
        # Opt-in leader-only pipeline; default every-rank-direct (see the N>=2 branch).
        self.leader_only = False if leader_only is None else bool(leader_only)
        # Opt-in gather-in-place: intra gather writes into the ring slot, dropping
        # the prepare_sync copy-IN. Default OFF (staged).
        self.gather_in_place = (
            False if gather_in_place is None else bool(gather_in_place)
        )
        # fuse-barrier: drop the intra gather's finish barrier on the every-rank-
        # direct N>=2 path. Bit-exact: the PUSH gather's flag-wait completes the
        # node-block on return and the ring's prepare_sync barrier follows; flags
        # monotonic per-call. Crash-safe via _prev_op_completed. Not for leader-only.
        self.fuse_barrier = True if fuse_barrier is None else bool(fuse_barrier)
        # Sliced 2-D AllGather (primary bandwidth path): inter ring contributes only
        # each rank's own shard (per-NIC inter (N-1)*count vs non-sliced G*count),
        # then N intra SDMA gathers reassemble rank-major. Default ON; owns its own
        # data path -> incompatible with leader_only/gather_in_place.
        _slice_conflict = self.leader_only or self.gather_in_place
        self.slice_inter = (
            bool(slice_inter) if slice_inter is not None else not _slice_conflict
        )
        # Fused sliced Phase B: fold the N reassembly gathers into one batch stacked
        # into disjoint transit regions (dst_base_offset=m*block), keeping only the
        # m==0 entry + exit barriers. Bit-exact. Only with slice_inter.
        self.slice_fused = bool(slice_fused) if slice_fused is not None else True
        # Per-rank byte threshold to engage the sliced path (below it its extra
        # launches cost more than the saved inter bytes).
        slice_min_bytes = (
            slice_min_bytes if slice_min_bytes is not None else 8 * 1024 * 1024
        )
        self.slice_min_bytes = max(0, slice_min_bytes)
        # Drop the redundant Phase-B m==0 entry barrier on the sliced+fused non-
        # overlap path (the inter ring's finish_sync barrier immediately precedes it
        # with no remote op between). Byte-identical. Not for the overlap path.
        self.slice_fuse_ib = bool(slice_fuse_ib) if slice_fuse_ib is not None else True
        # K element-range chunks for the chunked-ring pipe-band path (byte-identical).
        slice_pipe_chunks = slice_pipe_chunks if slice_pipe_chunks is not None else 2
        self.slice_pipe_chunks = max(1, slice_pipe_chunks)
        # Stream-ordered inter ring: on-device ShmemBarrierOnStream prepare/finish
        # instead of host-blocking sync (removes 2 CPU<->GPU round-trips/op).
        self.stream_ring = True
        # Stream-ordered Phase-B finish_batch: paired with stream_ring the whole op
        # runs on-stream with no host stall.
        self.stream_intra = True
        # Defer the Phase-B finish fence to the next op's inter-prepare barrier.
        # Safe: copy-OUT is stream-ordered (output correct); cross-PE reuse is
        # covered by the successor's prepare/entry barrier; the last op reuses
        # nothing. Dropped at ranks_per_node>=8 (_apply_dense_node_defaults).
        self.slice_defer_fin = True
        # Defer the inter ring's finish fence (cross-PE ring reuse) to the next slice
        # op's prepare_stream barrier. Same safety class as slice_defer_fin; non-oop
        # slice path only. Dropped at ranks_per_node>=8.
        self.slice_defer_inter_fin = True
        # Fused ring || local-block gather in one launch. OFF by default: under FSDP
        # tight back-to-back overlap the ring buffer is read out before the ring
        # CTA's remote puts are globally visible -> stale remote-half. Env overrides.
        if "MORI_HIER_FUSE_LOCAL" in os.environ:
            self.fuse_local = _env_true("MORI_HIER_FUSE_LOCAL")
        else:
            self.fuse_local = False
        self._init_fused_ring_state()
        # Direct-to-output Phase B: gathers PUSH each slice into the registered user
        # output, dropping the finish_batch D2D copy. Registration is safe only over
        # RDMA (single-node IPC hard-aborts), so default ON for multi-node, OFF for
        # single-node; None defers to the transport probe below.
        self.slice_direct = None if slice_direct is None else bool(slice_direct)
        if self.slice_inter and (self.leader_only or self.gather_in_place):
            raise ValueError(
                "slice_inter is incompatible with leader_only/gather_in_place "
                "(it owns the inter+intra data path)"
            )
        # Flat AllgatherSdma compatibility: split a combined transit size into
        # input/output when not sized explicitly.
        if transit_buffer_size is not None:
            if input_buffer_size is None:
                input_buffer_size = transit_buffer_size
            if output_buffer_size is None:
                output_buffer_size = transit_buffer_size * npes
        # Auto-detect topology (single node -> pure intra-node SDMA AllGather).
        if ranks_per_node is None:
            ranks_per_node = _auto_ranks_per_node(my_pe, npes)
        if ranks_per_node < 1 or npes % ranks_per_node != 0:
            raise ValueError(
                f"npes ({npes}) must be a positive multiple of ranks_per_node "
                f"({ranks_per_node})"
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

        # slice_direct (None) is resolved after the ring is built, by a transport
        # probe (see _probe_rdma_transport). num_nodes alone is not a safe signal
        # (the single-node spawn sim runs num_nodes>=2 over IPC).
        if self.num_nodes == 1 and self.slice_direct is None:
            self.slice_direct = False

        self._build_subcollectives(
            my_pe,
            npes,
            input_buffer_size,
            output_buffer_size,
            copy_output_to_user,
        )

    def _init_fused_ring_state(self):
        """Init fused ring / flag-token / reasm-deep-SQ / host-proxy-inter state."""
        # fuse_remote: FusedRingRemoteGatherKernel runs the ring and, per landed
        # sub-range, pushes the remote-block SDMA from the ring buffer into the
        # registered output (no ring copy-OUT / finish barrier). Only valid at
        # num_nodes==2 (single ring round).
        self.fuse_remote = _env_true("MORI_HIER_FUSE_REMOTE", "0")
        self._chunk_ready_flags = None
        # DEEP_PIPE flag layout (slots, per-PE count, depth) the persistent buffer
        # was last sized for; a layout change forces a fresh zeroed buffer so stale
        # per-sub-chunk landing state can't leak into the next size.
        self._chunk_ready_flags_layout = None
        # Intra reassembly deep-SQ: feed all reassembly SDMA copies back-to-back
        # then drain once. Bit-exact; fused path only.
        self._reasm_deep_sq = 1 if (self.fuse_local or self.fuse_remote) else 0

    def _init_sync_drain_state(self):
        # Isolation probe: force full stream completion at op return.
        self._debug_sync = _env_true("MORI_HIER_DEBUG_SYNC", "0")
        # CU-domain copy-out scratch: a copy-engine write to ``output`` is not
        # coherent with the FSDP backward GEMM's later CU read, so a torch
        # elementwise (CU) kernel does scratch -> output.
        self._cu_copyout_scratch = None

    def _apply_dense_node_defaults(self):
        # Dense-node (>=8 ranks/node) landing-fence fix: the Phase-B intra gather can
        # read a peer's ring-output over XGMI before that peer's ring finish is
        # globally visible (cross-PE race). Force the two finish fences ON (byte-
        # identical at ==4, where it never fires).
        if self.ranks_per_node >= 8:
            self.slice_defer_fin = False
            self.slice_defer_inter_fin = False
            # DEEP_PIPE_QUIET=0: land each sub-chunk via the completion AMO on the
            # data QP (RC in-order, flag never precedes bytes) instead of a send-CQ
            # quiet drain. Bit-exact. Explicit env wins.
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
        """Construct the intra/inter/bcast sub-collectives (single-node
        AllgatherSdma vs the multi-node intra-gather + inter-ring [+ leader-only
        broadcast] pipeline) and run the deferred slice_direct transport probe."""
        if self.num_nodes == 1:
            # Single node -> a plain intra-node SDMA AllGather is the full AllGather.
            from .collective import AllgatherSdma

            self._intra = AllgatherSdma(
                my_pe,
                npes,
                input_buffer_size=input_buffer_size,
                output_buffer_size=output_buffer_size,
                copy_output_to_user=copy_output_to_user,
            )
        else:
            # Hierarchical every-rank-direct pipeline (no broadcast phase):
            #   1. Intra-node SDMA gather over my node's G ranks -> node-block.
            #   2. Inter-node RDMA ring over same-local-index peers across nodes ->
            #      all N node-blocks in node order = rank-major all_gather result.
            # Bit-exact vs torch. Each node-block crosses the NIC G times (Gx
            # redundant); the sliced 2-D path (slice_inter) removes it.
            from .collective import (
                IntraNodeSubGroupAllgatherSdma,
                InterNodeRingAllgather,
            )

            G = self.ranks_per_node
            N = self.num_nodes
            # Intra transit holds the node-block (G shards); inter ring buffer holds
            # the full N*G-shard output.
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
            # Fused sliced Phase B stacks all N gathers into one transit, so it must
            # hold the full N*G output, not a single node-block.
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
                # Every-rank-direct (default): rings same-local-index peers; no
                # broadcast. Sends each node-block G times but is simple/bit-exact.
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
                # Leader-only (opt-in): local_rank==0 rings over node-leaders then
                # SDMA-broadcasts the full N*G output to its G local ranks (funnels
                # inter-node traffic through the leader's single NIC). ShmemMalloc/
                # ShmemBarrierAll are COLLECTIVE over ALL PEs, so non-leaders build a
                # degenerate singleton ring (ringSize=1, no launch) solely to keep
                # those barriers balanced.
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
            # Sliced-path scratch: this rank's collection C_g (N*count) from the
            # inter ring, consumed by the N intra reassembly gathers.
            self._slice_scratch = None
            # Lazy side stream for the slice-overlap lever (local gather concurrent
            # with the inter ring).
            self._overlap_stream = None
            # Clean-completion guard for the fuse-barrier entry-skip: True only after
            # a full successful op freed every peer's out_. A plain counter would
            # wrongly skip the entry barrier after a mid-pipeline crash.
            self._prev_op_completed = False
            # Previous op's path (slice vs non-slice); a switch force-keeps the entry
            # barrier since the paths reuse the shared buffers differently. None=none.
            self._last_use_slice = None
            # slice_direct output registration is a collective, so the register/
            # deregister decision MUST be lockstep on every PE (keyed by exact
            # (ptr,size); a per-rank decision mis-aligns the peer-pointer all-gather).
            # Single-entry tracker + LRU (cap 4) so alternating buffers stay resident.
            self._direct_reg_ptr = None
            self._direct_reg_size = None
            self._reg_cache_cap = 4
            self._direct_reg_lru = {}

            # Resolve deferred slice_direct by probing the transport: p2p==0 => RDMA
            # (register safe) => ON; non-zero => P2P/IPC => OFF (see
            # _probe_rdma_transport). Every-rank-direct slice only.
            if self.slice_direct is None:
                self.slice_direct = self._probe_rdma_transport()
            elif self.slice_direct:
                # Fail-closed guard: explicit slice_direct=True is valid only over
                # RDMA (P2P/IPC register hard-aborts); raise instead of the opaque
                # device abort.
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

    def _ensure_output_registered(self, output_data):
        """Register output_data for the direct-to-output SDMA push, LRU-cached.
        Lockstep across PEs, exact-match only; a hit issues no collective.
        """
        out_ptr = output_data.data_ptr()
        out_size = output_data.numel() * output_data.element_size()
        key = (out_ptr, out_size)
        cap = self._reg_cache_cap
        if cap <= 1:
            if key != (self._direct_reg_ptr, self._direct_reg_size):
                if self._direct_reg_ptr is not None:
                    self._intra.deregister_output_buffer_ptr(self._direct_reg_ptr)
                self._intra.register_output_buffer(output_data)
                self._direct_reg_ptr = out_ptr
                self._direct_reg_size = out_size
            return
        lru = self._direct_reg_lru
        if key in lru:
            # hit: refresh recency, no collective (LRU mirrors C++ overlap-eviction).
            lru.pop(key)
            lru[key] = out_size
            return
        # miss: C++ register_output_buffer evicts any registration overlapping
        # [ptr,ptr+size), so drop those from the Python LRU too (else a later hit
        # skips re-register for a ptr C++ already evicted).
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
        """True iff a cross-node ring peer is reached over RDMA (not IPC), where
        ShmemSymmetricRegister is safe. Conservative: any error/IPC peer -> False.
        """
        if self.leader_only or self.num_nodes < 2:
            return False
        try:
            from ..shmem import shmem_ptr_p2p

            # Pick a peer on a different node so the connection is inter-node.
            G = self.ranks_per_node
            peer_pe = self.local_rank + G * ((self.node_id + 1) % self.num_nodes)
            buf_ptr = self._inter._handle.buf_ptr()
            p2p = shmem_ptr_p2p(buf_ptr, self.my_pe, peer_pe)
            return p2p == 0  # 0 => RDMA => direct-to-output safe
        except Exception:
            return False

    def _cu_copyout_finish(self, output_data, total_count_elems, stream):
        """CU-domain copy-OUT for the nodirect Phase-B.

        The SDMA-written transit ``out_`` is coherent to a CU read but NOT to the
        copy engine, so copy-OUT via a torch elementwise (CU) kernel, not
        hipMemcpyAsync, to avoid racing the SDMA writes. ``add`` by 0 is bit-exact.
        Cross-PE ``out_`` reuse is fenced by ``finish_direct_stream`` (deferrable).
        """
        transit = self._intra.get_output_transit_buffer(
            dtype=output_data.dtype, device=output_data.device
        )[:total_count_elems]
        out_flat = output_data.view(-1)[:total_count_elems]
        if stream is not None:
            with torch.cuda.stream(stream):
                torch.add(transit, 0, out=out_flat)
        else:
            torch.add(transit, 0, out=out_flat)
        self._intra.finish_direct_stream(
            stream=stream, barrier=not self.slice_defer_fin
        )

    def __call__(self, input_data, output_data, count: int, stream=None) -> bool:
        """Gather ``count`` elements/rank into ``output_data`` (rank-major).

        Thin wrapper over ``_call_impl``. ``MORI_HIER_DEBUG_SYNC=1`` host-blocks on
        the caller's stream at op return (FSDP nondeterminism isolation probe).
        """
        # Cross-stream lifetime guard: on a comm stream (FSDP2) the input may be
        # freed/recycled by reshard on the compute stream, so record_stream defers
        # reuse until this AG completes (else it reads a partially-overwritten input).
        if stream is not None:
            if hasattr(input_data, "record_stream"):
                input_data.record_stream(stream)
            if hasattr(output_data, "record_stream"):
                output_data.record_stream(stream)
        # Launch-collapse via HIP graph replay: one launch instead of the per-op
        # multi-launch ramp. Bit-exact; falls back cleanly on capture failure. Capped
        # at <=48MB -- bulk sizes fall through to the fused path (static replay
        # serializes the ring||reassembly overlap). Default ON (CUDA_GRAPH=0 off).
        if (
            os.environ.get("MORI_HIER_CUDA_GRAPH", "1").strip().lower()
            in ("1", "true", "yes", "on")
            and not self._debug_sync
        ):
            _cg_max_mb = 48.0
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
        """Capture the op once into a HIP graph, then replay.

        Returns True if serviced here (warm/capture or replay), False to fall
        through to eager (capture impossible for this key, e.g. a host barrier is
        still inside the op). Keyed by (in_ptr, out_ptr, count, dtype); FSDP reuses
        the same buffers so steady state is a pure one-launch replay.
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
            # Warm the eager op so registration/scratch/completion state settles,
            # THEN capture. Warm runs also produce a correct output.
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
                # Capture blocked by a host-side op inside the op body; record the
                # miss (eager next time) and name the blocker.
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

        # Size-threshold dispatch: sliced 2-D only for large payloads; below the
        # threshold the non-sliced fuse-barrier path is faster.
        byte_count = count * input_data.element_size()
        use_slice = self.slice_inter and (byte_count >= self.slice_min_bytes)
        # Mid/small band -> chunked-ring pipe-overlap path (sliced fused, K>1).
        use_pipe_band = (
            (not use_slice)
            and self.slice_inter
            and self.slice_fused
            and self.slice_pipe_chunks > 1
        )
        # 3-way path key; a switch clears the clean-completion guard (entry fence).
        path_key = "slice" if use_slice else ("pipe" if use_pipe_band else None)
        if path_key != self._last_use_slice:
            self._prev_op_completed = False
        self._last_use_slice = path_key

        if use_slice or use_pipe_band:
            # Sliced 2-D AllGather (see __init__). Phase A (inter ring): each rank
            # rings only its own shard into C_g in node order.
            slice_total = count * N
            if use_pipe_band and self.slice_fused and self.slice_pipe_chunks > 1:
                # Chunked-ring pipeline overlap: per element-range chunk, run the
                # inter ring (main stream) into a disjoint scratch region then launch
                # its N reassembly gathers on a side SDMA stream, so chunk k's gather
                # overlaps chunk k+1's ring (only the last serial); one final copy-OUT.
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
                side.wait_stream(main)
                off = 0
                # Invariant: only ONE global on-stream fence in flight (the ring
                # prepare); side gathers are barrier-free, so no concurrent-global-
                # barrier race. Cross-chunk ring reuse is ordered by chunk k+1's
                # prepare_stream fence.
                sr = self.stream_ring
                for k in range(K):
                    ck = base_ck if k < K - 1 else count - base_ck * (K - 1)
                    if ck == 0:
                        continue
                    region = collection[N * off : N * off + N * ck]
                    # Inter ring of chunk k on the main stream (stream-ordered
                    # copy-OUT into ``region``); its finish is deferred because the
                    # peer's chunk-k landing is fenced by the intra subgroup barrier
                    # on the first Phase-B gather below.
                    _defer = sr
                    self._inter(
                        input_data[off : off + ck],
                        region,
                        ck,
                        stream,
                        stream_ring=sr,
                        defer_inter_fin=_defer,
                    )
                    side.wait_stream(main)
                    # Cheap intra landing fence: arm only the first gather with the
                    # intra subgroup entry barrier (G ranks, XGMI-scope), ordering all
                    # G peers past their chunk-k ring copy-OUT before any peer read.
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
                # All gathers must land before the bulk copy-OUT reads them.
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
            # The direct-path overlap and the fused path both run the ring inside
            # Phase B, so skip the monolithic ring call here.
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
            # Phase B (intra SDMA): reassemble each B_m into output[m*block:]
            # (concatenated by group_pos=local_rank => rank-major).
            if self.slice_fused:
                # Fold the N gathers into one batch stacked into disjoint transit
                # regions + one bulk copy-OUT, keeping only entry+exit barriers. The
                # inter ring's finish barrier makes the m==0 entry redundant -> drop
                # under slice_fuse_ib (default).
                entry_barrier = not self.slice_fuse_ib
                if self.slice_direct and self.stream_intra and self.stream_ring:
                    # Direct-to-output Phase B: push each node-block's slices straight
                    # into output[m*block:] -- no transit, no full-output copy-OUT;
                    # one deferrable global fence completes the op. See
                    # _ensure_output_registered for the lockstep registration.
                    self._ensure_output_registered(output_data)
                    if self.fuse_remote and self.num_nodes == 2 and not entry_barrier:
                        # Fused-remote pipeline: one launch runs the ring and, per
                        # landed sub-range, pushes the remote-block SDMA from the ring
                        # buffer into the output (no ring copy-OUT / finish barrier).
                        from .collective import launch_fused_ring_remote_gather

                        node = self.node_id
                        rb = self._inter.num_blocks
                        # Chunk-landing flag buffer: >= P slots when DEEP_PIPE splits
                        # the ring channel into P sub-chunks (rb==1), else ring_blocks
                        # u64. Depth: auto=round(perPE_bytes/SUBBYTES) else explicit;
                        # default 2, caged to depth 1 by the 32MB gate below (bit-exact).
                        _dp_raw = os.environ.get("MORI_HIER_DEEP_PIPE", "2").strip()
                        if _dp_raw.lower() == "auto":
                            # 16MiB sub-chunk target, matching the C++ auto selector.
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
                        # Coherence gate: landing signal is bit-exact only while each
                        # sub-chunk stays inside the 32MB NIC-DMA->HBM window; larger
                        # falls to depth 1. Mirrors the C++ HierDeepPipe gate.
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
                        # Reallocate the flag buffer on any layout change (not only to
                        # grow): reusing it across two DEEP_PIPE sizes leaves stale
                        # landing state -> mismatch. DEEP_PIPE=1 never enters here.
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
                        ring_args, _u32c, s_main = self._inter.prepare_stream_only(
                            input_data, count, stream
                        )
                        # Local-block direct-gather jit_args (own input, no ring dep).
                        gather_args = self._intra.prepare_direct_only(
                            input_data,
                            output_data,
                            count,
                            dst_block_offset=node * block_count,
                            stream=stream,
                            prepare_barrier=False,
                        )
                        # Reassembly block j drives SDMA queue (j+1)%nq (local-block
                        # CTA owns queue 0); nq>=2 REQUIRED, else the modulo aliases
                        # onto queue 0 -> shared signal counter -> liveness HANG. nq
                        # fixed at anvil init (default 2, E2E-safe); reasm=nq-1.
                        _sdma_nq = int(os.environ.get("MORI_SDMA_NUM_CHANNELS", "2"))
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
                        # Single completion fence (gathers already pushed to output).
                        # Standalone deferral: the device completion reader spins until
                        # every remote push landed + threadfence, so output is stream-
                        # correct without it; reuse covered by the successor's prepare.
                        _fin_barrier = not self.slice_defer_fin
                        self._intra.finish_direct_stream(
                            stream=stream, barrier=_fin_barrier
                        )
                    elif self.fuse_local and not entry_barrier:
                        # Fused ring || local-block gather in one launch. The ring's
                        # prepare_stream barrier is the sole entry fence; the local
                        # gather runs barrier-free (own input) and pushes block node_id
                        # into the output. Remote blocks follow after the copy-OUT.
                        from .collective import launch_fused_ring_local_gather

                        node = self.node_id
                        main = (
                            torch.cuda.current_stream(input_data.device)
                            if stream is None
                            else stream
                        )
                        # Ring prepare = global entry barrier + copy-IN (no launch).
                        ring_args, _u32c, s_main = self._inter.prepare_stream_only(
                            input_data, count, stream
                        )
                        # Local-block direct-gather jit_args (no launch).
                        gather_args = self._intra.prepare_direct_only(
                            input_data,
                            output_data,
                            count,
                            dst_block_offset=node * block_count,
                            stream=stream,
                            prepare_barrier=False,
                        )
                        # One fused launch: ring || local gather, concurrent after the
                        # entry barrier (no host wait_stream merge).
                        launch_fused_ring_local_gather(
                            ring_args, gather_args, self._inter.num_blocks, s_main
                        )
                        # Ring finish copy-OUT (ring kernel already ran in the fused
                        # launch); copy-engine finish so fuse_local never moves bulk
                        # bytes on CUs.
                        reasm_src = collection
                        self._inter.finish_ring_stream(
                            collection,
                            count,
                            stream,
                            barrier=not self.slice_defer_inter_fin,
                            cu_copyout=False,
                        )
                        # Remote blocks read the ring collection via an intra-node
                        # subgroup gather. A single intra entry barrier on the first
                        # remote gather orders every rank's ring copy-OUT before any
                        # peer read (XGMI-scope) -- avoids the stale-remote-half race
                        # without a global barrier (dependency is purely intra-node).
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
                    # Stream-ordered copy-OUT; its fence is deferrable
                    # (slice_defer_fin) since only cross-PE reuse needs it.
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
            # gather_in_place (opt-in): intra gather writes its node-block straight
            # into this PE's ring slot (chunk_in_place=True), dropping the
            # prepare_sync copy-IN. Perf-neutral vs the staged default (correctness
            # identical).
            node_block = self._inter.slot_tensor(
                block_count, input_data.dtype, input_data.device
            )
            # fuse_barrier: the ring's prepare_sync_in_place barrier follows
            # immediately, covering the dropped barrier.
            self._intra(
                input_data,
                node_block,
                count,
                stream,
                barrier=not self.fuse_barrier,
                prepare_barrier=intra_prepare_barrier,
            )
            # Phase 2 (inter ring): all-gather the N node-blocks in node order.
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
        # fuse_barrier: drop the intra finish barrier on the every-rank-direct path
        # (the inter ring's prepare_sync barrier follows immediately); leader-only
        # keeps it.
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
            # Phase 2 (inter ring, staged default): all-gather the N node-blocks in
            # node order -> full rank-major output.
            self._inter(node_block, output_data, block_count, stream)
            self._prev_op_completed = True
            return True

        # Leader-only: phase 2 ring (leaders only) -> phase 3 SDMA broadcast.
        full_count = count * self.npes
        if self.local_rank == 0:
            self._inter(node_block, output_data, block_count, stream)
        else:
            # Non-leader: degenerate singleton ring on scratch, only to join the two
            # collective ShmemBarrierAll calls (no data move).
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

        # Phase 3 (intra SDMA broadcast): leader (root) fans its full N*G output to
        # the G local ranks over XGMI.
        self._bcast(output_data, output_data, full_count, stream)
        self._prev_op_completed = True
        return True

    def supports_param_contiguous_output(self) -> bool:
        """True when the direct-to-output param-contiguous zero-copy path is
        available (cross-node, slice_direct over RDMA); the FSDP adapter probes it
        to decide whether to skip its copy-OUT.
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
        """Param-contiguous zero-copy AllGather (kills the FSDP copy-OUT).

        Writes the result straight into ``output_data`` in param-contiguous layout:
        for rank ``r`` and param ``s`` (per-rank elems ``E_s`` at input offset
        ``O_s``) rank ``r``'s slice lands at ``O_s*W + r*E_s`` -- what FSDP's packed
        all-gather expects. ``split_sizes``/``split_offsets`` are in input-dtype
        elements; their byte extents must be 4-byte aligned (SDMA). Returns False
        (caller falls back to copy-OUT ``__call__``) when unavailable.
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

        # Split geometry in u32 lanes (SDMA byte move; exact given the alignment
        # guard above). Cache the u32 GPU tensors keyed by geometry -- FSDP reuses
        # the same geometry across a param group, so steady state avoids the H2D.
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

    def get_output_transit_buffer(self, dtype=None, device=None):
        if self.num_nodes == 1:
            return self._intra.get_output_transit_buffer(dtype=dtype, device=device)
        raise NotImplementedError(
            "HierAllGather inter-node path writes directly to the user output; "
            "no transit-buffer view is exposed."
        )
