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
#
# Hierarchical host-proxy all-gather (persistent, reusable collective).
#
# The GPU-initiated device RDMA-WRITE inter-node leg is limited by the per-QP
# posting wall: a single CTA cannot keep a deep enough send queue in flight. A
# persistent CPU proxy posts a deep multi-WQE SQ pipeline instead, and fans the
# one cross-node node-block across every data NIC concurrently (rail-paired, no
# G x redundancy).
#
# Exposes the same callable contract as the device HierAllGather --
# handle(inp, out, numel, stream) -> bool -- so it backs both the standalone
# sweep and the FSDP custom all-gather. Only the node-block exchange rides the
# CPU-posted host-ibverbs transport; the intra-node legs ride XGMI (NCCL).
#
# Layout assumption (matches HierAllGather / torchrun node-major ranks):
#   pe p -> node p // ranks_per_node, local index p % ranks_per_node.
# Rail partner of pe p = same local index on the other node. Two nodes only;
# num_nodes==1 degenerates to a pure intra-node all-gather.
#
# Landing fence: the host completion (wait_all) plus the post-exchange barrier
# guarantee the received remote shard is physically in HBM before any consumer
# (the step-3 intra gather / copy-out) reads it.

import os
import socket

import torch
import torch.distributed as dist


def _local_ip(peer_ip):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((peer_ip, 80))
        return s.getsockname()[0]
    finally:
        s.close()


class HostProxyHierAllGather:
    """Persistent hierarchical CPU-posted all-gather.

    Constructed ONCE per rank (engine + rail-partner session + a registered
    max-size staging buffer live for the object lifetime); ``__call__`` is the
    per-op hot path and allocates nothing on the fabric side.
    """

    def __init__(
        self,
        my_pe,
        npes,
        ranks_per_node,
        output_buffer_size,
        device=None,
        qp_per_transfer=None,
        num_worker_threads=None,
        chunk_bytes=None,
    ):
        from mori.io import (
            IOEngine,
            IOEngineConfig,
            RdmaBackendConfig,
            BackendType,
            EngineDesc,
            MemoryDesc,
            PollCqMode,
            StatusCode,
            set_log_level,
        )

        self._StatusCode = StatusCode
        set_log_level("error")

        self.my_pe = my_pe
        self.npes = npes
        self.ranks_per_node = ranks_per_node
        self.num_nodes = npes // ranks_per_node
        self.node_id = my_pe // ranks_per_node
        self.local_rank = my_pe % ranks_per_node
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self._max_bytes = int(output_buffer_size)

        # RDMA backend tunables (deep-SQ regime validated on this fabric).
        qp = qp_per_transfer or 4
        wt = num_worker_threads or 1
        chunk = chunk_bytes or (64 * 1024)
        self._timeout_ms = 60000
        # When set, the two intra all-gather legs ride the SDMA copy engine over
        # XGMI (mori's IntraNodeSubGroupAllgatherSdma, a PUSH gather) instead of
        # dist.all_gather (NCCL, which drives bytes on CU/SM), keeping the CUs
        # free for the backward GEMM. Default OFF keeps the NCCL legs.
        self._sdma_intra = os.environ.get("MORI_HOSTPROXY_SDMA_INTRA", "0") not in (
            "0",
            "",
            "false",
            "False",
        )
        # Twin transit: step-3 (in _complete) drives its OWN
        # IntraNodeSubGroupAllgatherSdma handle (disjoint out_) so it never
        # contends step-1's transit -> no intra-AG WAR, a single node-local
        # barrier per AG frees both transits. Costs one extra one-node-block
        # ShmemMalloc; only the 2-node SDMA-intra path builds it.

        # ASYNC double-buffered receive staging (MORI_HOSTPROXY_ASYNC_RING,
        # default 1 = OFF = byte-identical). With a single staging slot the async
        # path has a read/write race: after the rail-pair barrier, _complete(N)'s
        # step-3 GPU gather READS my recv staging slot, but the partner is then
        # free to run op(N+1)'s _post which RDMA-WRITES the SAME staging bytes;
        # under deferral the read can be overtaken -> torn recv -> NaN. RING>=2
        # lands op N and op N+1 in DISJOINT byte regions so the partner's next
        # write cannot clobber this op's read. The heap (cap*world bytes) already
        # dwarfs the 2 shards an op uses, so the ring regions are carved from the
        # existing allocation (no extra memory, no host fence, bulk bytes stay on
        # RDMA/SDMA). Both ranks derive the slot from the same monotone op counter
        # (FSDP issues AGs in identical order per rank).
        self._async_ring = int(os.environ.get("MORI_HOSTPROXY_ASYNC_RING", "1") or "1")
        if self._async_ring < 1:
            self._async_ring = 1
        # Per-rank staging slot size (one shard cap). _max_bytes == cap*npes.
        self._cap_stage_bytes = self._max_bytes // npes
        if self._async_ring > 1 and (1 + self._async_ring) > npes:
            raise RuntimeError(
                f"MORI_HOSTPROXY_ASYNC_RING={self._async_ring} needs "
                f"{1 + self._async_ring} staging slots but only {npes} exist"
            )
        self._op_ctr = 0

        # Persistent byte-addressable staging heap (registered on the NIC once).
        self._stage = torch.zeros(self._max_bytes, dtype=torch.uint8, device=device)

        # Intra-node NCCL subgroups (XGMI). Every rank must build every group.
        self._node_locals = [
            list(range(n * ranks_per_node, (n + 1) * ranks_per_node))
            for n in range(self.num_nodes)
        ]
        self._intra_groups = [
            dist.new_group(ranks=self._node_locals[n]) for n in range(self.num_nodes)
        ]
        self._intra_group = self._intra_groups[self.node_id]

        # SDMA (XGMI copy-engine) intra all-gather over THIS node's local
        # sub-group -- the on-thesis replacement for the NCCL intra legs. One
        # persistent handle drives both step-1 (own shard) and step-3 (received
        # remote shard) gathers: the sub-group + group-position are identical
        # (same local ranks, gathered in local-index order), only the input and
        # the destination node-block region differ per call. Requires
        # MORI_ENABLE_SDMA=1 (the harness sets it) + shmem initialized.
        self._sdma = None
        self._sdma3 = None  # twin transit for step-3
        if self._sdma_intra:
            from mori.ccl import IntraNodeSubGroupAllgatherSdma

            self._sdma = IntraNodeSubGroupAllgatherSdma(
                my_pe=self.my_pe,
                npes=self.npes,
                out_buffer_bytes=self._max_bytes,
                group_size=self.ranks_per_node,
                group_pos=self.local_rank,
                pe_base=self.node_id * self.ranks_per_node,
                pe_stride=1,
            )
            # Twin transit: a SECOND handle (disjoint out_) drives step-3 so it
            # never contends step-1's transit. Every rank builds it in lockstep
            # (ShmemMalloc is collective-symmetric). Sized to exactly one node-block
            # (_max_bytes // num_nodes) so the 2nd malloc fits the heap alongside the
            # first handle (which over-allocates full output).
            if self.num_nodes == 2:
                twin_bytes = self._max_bytes // self.num_nodes
                self._sdma3 = IntraNodeSubGroupAllgatherSdma(
                    my_pe=self.my_pe,
                    npes=self.npes,
                    out_buffer_bytes=twin_bytes,
                    group_size=self.ranks_per_node,
                    group_pos=self.local_rank,
                    pe_base=self.node_id * self.ranks_per_node,
                    pe_stride=1,
                )

        if self.num_nodes == 1:
            # Degenerate: no fabric transport needed.
            self._session = None
            return
        if self.num_nodes != 2:
            raise NotImplementedError(
                "HostProxyHierAllGather rail-paired exchange supports exactly "
                f"2 nodes (got num_nodes={self.num_nodes})"
            )

        other_node = 1 - self.node_id
        self._partner = self._node_locals[other_node][self.local_rank]

        # Rail-pair subgroups: the landing fence only needs MY partner to have
        # finished ITS write to me, so a 2-rank barrier replaces the world one.
        # Every rank must build every group.
        self._pair_barrier = None
        for i in range(ranks_per_node):
            pair = sorted([self._node_locals[0][i], self._node_locals[1][i]])
            g = dist.new_group(ranks=pair)
            if my_pe in pair:
                self._pair_barrier = g
        self._copy_ev = torch.cuda.Event()

        master_ip = os.environ["MASTER_ADDR"]
        my_ip = _local_ip(master_ip)
        base_port = 31500
        port = base_port + my_pe

        cfg = IOEngineConfig(host=my_ip, port=port)
        self._engine = IOEngine(key=f"hpag-{my_pe}", config=cfg)
        rcfg = RdmaBackendConfig(
            qp_per_transfer=qp,
            post_batch_size=-1,
            num_worker_threads=wt,
            poll_cq_mode=PollCqMode.POLLING,
            enable_transfer_chunking=True,
            chunk_bytes=chunk,
        )
        rcfg.max_send_wr = 512
        rcfg.max_cqe_num = 2048
        rcfg.max_msg_sge = 1
        self._engine.create_backend(BackendType.RDMA, rcfg)

        my_edesc = self._engine.get_engine_desc().pack()
        all_edesc = [None] * npes
        dist.all_gather_object(all_edesc, my_edesc)

        self._local_mem = self._engine.register_torch_tensor(self._stage)
        my_mdesc = self._local_mem.pack()
        all_mdesc = [None] * npes
        dist.all_gather_object(all_mdesc, my_mdesc)

        dist.barrier()
        self._engine.register_remote_engine(EngineDesc.unpack(all_edesc[self._partner]))
        remote_mem = MemoryDesc.unpack(all_mdesc[self._partner])
        self._session = self._engine.create_session(self._local_mem, remote_mem)
        dist.barrier()

    # -- helpers -----------------------------------------------------------
    def _stage_view(self, dtype, nelems):
        nbytes = nelems * torch.tensor([], dtype=dtype).element_size()
        return self._stage[:nbytes].view(dtype)

    def _intra_ag(
        self,
        inp_1d,
        out_slots,
        out_block_1d,
        count,
        stream,
        sdma_handle=None,
        barrier_after=True,
    ):
        """Gather ``count``-element shards over this node's local sub-group.

        SDMA path (on-thesis): PUSH-gather over XGMI straight into the
        contiguous node-block ``out_block_1d`` -- bulk bytes ride the copy
        engine, CUs stay free. NCCL fallback: dist.all_gather into the per-slot
        views ``out_slots``. Both leave the node-block laid out in local-index
        order, bit-exact.
        """
        hnd = sdma_handle if sdma_handle is not None else self._sdma
        if hnd is not None:
            # The sub-group gather's default prepare_sync/finish_sync each issue a
            # WORLD host ShmemBarrierAll (the socket-bootstrap TCP barrier). On
            # this per-AG hot path that world barrier is slow and fragile, and its
            # only job is inter-op ordering on the shared transit ``out_`` for THIS
            # node's sub-group. Replace it 1:1 with a NODE-LOCAL barrier on the
            # torch PG (self._intra_group): the exit barrier ensures all peers
            # finished pushing before the next op reuses out_. Node-local
            # invariant, so the data path is byte-for-byte unchanged.
            hnd(
                inp_1d,
                out_block_1d,
                count,
                stream,
                barrier=False,
                prepare_barrier=False,
            )
            if barrier_after:
                dist.barrier(group=self._intra_group)
        else:
            dist.all_gather(out_slots, inp_1d, group=self._intra_group)

    # -- hot path ----------------------------------------------------------
    def __call__(self, inp, out, numel, stream=None):
        """Blocking all-gather. Returns True when ``out`` is fully landed.

        The host thread blocks on wait_all + rail-pair barrier before step 3, so
        on return every consumer-visible byte is in HBM (the bit-exact base).
        Equivalent to ``_post`` immediately followed by ``_complete``.
        """
        h = self._post(inp, out, numel, stream)
        if h is None:
            return True
        self._complete(h)
        return True

    def call_async(self, inp, out, numel, stream=None):
        """Non-blocking all-gather. Returns a handle whose ``_complete`` runs the
        host-blocking landing fence (wait_all + rail-pair barrier + step 3).

        Splitting the op lets the cross-node CPU-posted RDMA round trip + the
        intra XGMI/SDMA gather overlap the CALLER's compute: the host thread is
        free between post() and complete() instead of stalling mid-AG. The caller
        MUST run ``_complete(handle)`` before reading ``out`` and before the NEXT
        all-gather (the staging heap holds a single in-flight op). This is the
        overlap window native RCCL gets by returning a Work; the sync ``__call__``
        path forfeits it. Returns None for the single-node degenerate path
        (already blocking, nothing to defer).
        """
        return self._post(inp, out, numel, stream)

    def _post(self, inp, out, numel, stream=None):
        """Non-blocking half: stage my shard, POST the cross-node RDMA write(s),
        and issue the step-1 intra gather. Returns a completion handle (or None
        for the single-node path). No wait_all / no landing barrier here, so the
        host does NOT stall on the fabric round trip."""
        assert inp.is_cuda and out.is_cuda
        assert out.numel() == numel * self.npes
        e = numel
        world = self.npes
        rpn = self.ranks_per_node
        elsize = inp.element_size()
        shard_bytes = e * elsize
        inp = inp.contiguous()

        base = self.node_id * rpn

        # Run every GPU op (stage copy, intra all_gathers, pair barrier) on the
        # CALLER's stream, not the default stream: a cross-node consumer (FSDP)
        # records its completion event on this stream and gates downstream compute
        # on it, so ops on the default stream would be untracked and could race
        # the gather. No-op when stream == default (standalone UT).
        if stream is None:
            stream = torch.cuda.current_stream(self.device)

        with torch.cuda.stream(stream):
            my_out_slots = [
                out[(base + i) * e : (base + i + 1) * e] for i in range(rpn)
            ]

            if self.num_nodes == 1:
                self._intra_ag(
                    inp,
                    my_out_slots,
                    out[base * e : (base + rpn) * e],
                    e,
                    stream,
                )
                return None

            # Staging layout for the cross-node shard exchange.
            #   send_local_off : byte offset of MY shard I read for the RDMA write.
            #   write_remote_off: byte offset in the PARTNER's heap I write to (==
            #     where the partner reads its recv), so both sides must agree.
            #   recv_slot      : local view where the PARTNER's shard lands (== the
            #     partner's write_remote_off into MY heap, so it matches by symmetry).
            # Default (ring==1): pe-indexed slots (send from sv[my_pe], the partner
            # writes into MY sv[my_pe] and I read sv[partner]) -- byte-identical.
            # RING>1: send region [0,cap), recv region cap*(1+slot) with slot cycled
            # per op, so op N and op N+1 land in DISJOINT bytes (breaks the async
            # read/write race). K forced to 1 (single-chunk ring math).
            if self._async_ring > 1:
                self._op_ctr += 1
                slot = self._op_ctr % self._async_ring
                cap = self._cap_stage_bytes
                send_local_off = 0
                write_remote_off = cap * (1 + slot)
                sv_send = self._stage[
                    send_local_off : send_local_off + shard_bytes
                ].view(inp.dtype)
                sv_send.copy_(inp)
                recv_slot = self._stage[
                    write_remote_off : write_remote_off + shard_bytes
                ].view(inp.dtype)
            else:
                # The staging heap only carries single shards (send from sv[my_pe],
                # receive into sv[partner]); the intra gathers write STRAIGHT into
                # the user's ``out`` so there is no full-output copy-out.
                sv = self._stage_view(inp.dtype, e * world)
                # Stage my shard for the NIC and make it device-visible (GDR read).
                sv[self.my_pe * e : (self.my_pe + 1) * e].copy_(inp)
                send_local_off = self.my_pe * shard_bytes
                write_remote_off = self.my_pe * shard_bytes
                recv_slot = sv[self._partner * e : (self._partner + 1) * e]
            self._copy_ev.record(stream)
            self._copy_ev.synchronize()

            other_node = 1 - self.node_id
            obase = other_node * rpn
            other_out_slots = [
                out[(obase + i) * e : (obase + i + 1) * e] for i in range(rpn)
            ]

            # Whole-shard cross-node exchange: one write, one landing barrier, one
            # step-3 gather (SDMA packs slots contiguously and cannot chunk-with-
            # stride; the ring layout carves single-shard recv regions).
            K = 1
            bounds = [(k * e) // K for k in range(K + 1)]

            # step 2 (inter, CPU-posted RDMA, rail-paired): POST every chunk write
            # up front so all chunks stream concurrently on the persistent workers.
            sts = []
            for k in range(K):
                o0 = bounds[k]
                nb = (bounds[k + 1] - o0) * elsize
                if nb == 0:
                    sts.append(None)
                    continue
                b0_local = send_local_off + o0 * elsize
                b0_remote = write_remote_off + o0 * elsize
                uid = self._engine.allocate_transfer_uid()
                sts.append(self._session.write(b0_local, b0_remote, nb, uid))

            # step 1 (intra XGMI, overlapped): gather my node's block into out
            # while the fabric writes are in flight. TWIN: skip the step-1 exit
            # barrier -- step-3 uses a disjoint transit (self._sdma3) so there is
            # no intra-AG WAR, and _complete's single end barrier frees step-1's
            # transit for the next op.
            self._intra_ag(
                inp,
                my_out_slots,
                out[base * e : (base + rpn) * e],
                e,
                stream,
                barrier_after=(self._sdma3 is None),
            )

        return {
            "stream": stream,
            "out": out,
            "e": e,
            "rpn": rpn,
            "obase": obase,
            "recv_slot": recv_slot,
            "other_out_slots": other_out_slots,
            "sts": sts,
            "bounds": bounds,
            "K": K,
        }

    def _complete(self, h):
        """Blocking half: for each cross-node chunk, drain its host CQE
        (wait_all), rail-pair barrier (partner's write into MY sv landed), then
        run the step-3 intra broadcast into ``out``. On return every
        consumer-visible byte is in HBM (the landing fence)."""
        stream = h["stream"]
        out, e, rpn, obase = h["out"], h["e"], h["rpn"], h["obase"]
        recv_slot, other_out_slots = h["recv_slot"], h["other_out_slots"]
        sts, bounds, K = h["sts"], h["bounds"], h["K"]

        def _bcast_k(k):
            o0, o1 = bounds[k], bounds[k + 1]
            recv_k = recv_slot[o0:o1]
            slots_k = [s[o0:o1] for s in other_out_slots]
            # TWIN: drive step-3 on the disjoint transit (self._sdma3) so it never
            # contends step-1's out_. The trailing barrier stays here -- it now
            # frees BOTH transits for the next op.
            self._intra_ag(
                recv_k,
                slots_k,
                out[obase * e : (obase + rpn) * e],
                o1 - o0,
                stream,
                sdma_handle=self._sdma3,
            )

        with torch.cuda.stream(stream):
            # step 3 (intra XGMI), pipelined: as each cross-node chunk lands,
            # broadcast it into out. The GPU runs step-3 chunk k while the host
            # workers still push chunks >k, hiding the exposed cross-node tail.
            for k in range(K):
                if sts[k] is None:
                    continue
                rc = self._engine.wait_all([sts[k]], self._timeout_ms)
                if rc != self._StatusCode.SUCCESS:
                    raise RuntimeError(f"HostProxy inter-node write rc={rc}")
                # rail-pair barrier: partner's write of chunk k into MY sv[partner]
                # has landed (partner's own wait_all on chunk k returned before it
                # entered this barrier => the bytes are in my HBM).
                dist.barrier(group=self._pair_barrier)
                _bcast_k(k)
        return True
