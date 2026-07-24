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
# A persistent CPU proxy posts the deep-SQ RDMA-WRITE inter-node leg (a single
# CTA cannot keep a deep enough send queue in flight); intra-node legs ride XGMI.
# Same callable contract as device HierAllGather: handle(inp,out,numel,stream)->bool.
#
# Layout (node-major ranks): pe p -> node p//ranks_per_node, local p%ranks_per_node.
# Rail partner = same local index on the other node. 2 nodes only; num_nodes==1
# degenerates to a pure intra-node all-gather.
#
# Landing fence: host wait_all + post-exchange barrier guarantee the remote shard
# is in HBM before any consumer (step-3 gather / copy-out) reads it.

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
        # When set, intra legs ride SDMA/XGMI (PUSH gather) instead of dist.all_gather
        # (native all-gather on CU/SM), freeing CUs for the backward GEMM. Default OFF = native legs.
        self._sdma_intra = os.environ.get("MORI_HOSTPROXY_SDMA_INTRA", "0") not in (
            "0",
            "",
            "false",
            "False",
        )
        # Twin transit: step-3 drives its OWN SDMA handle (disjoint out_) so it never
        # contends step-1's transit -> no intra-AG WAR; one node-local barrier frees both.

        # ASYNC double-buffered recv staging (MORI_HOSTPROXY_ASYNC_RING, default 1 =
        # OFF = byte-identical). Single-slot async path races: _complete(N) step-3 reads
        # my recv slot while the partner's op(N+1) _post RDMA-WRITES the same bytes ->
        # torn recv. RING>=2 lands op N and N+1 in DISJOINT regions (carved from the
        # existing heap, no extra memory). Both ranks derive the slot from the same
        # monotone op counter (FSDP issues AGs in identical order per rank).
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

        # Intra-node subgroups (XGMI). Every rank must build every group.
        self._node_locals = [
            list(range(n * ranks_per_node, (n + 1) * ranks_per_node))
            for n in range(self.num_nodes)
        ]
        self._intra_groups = [
            dist.new_group(ranks=self._node_locals[n]) for n in range(self.num_nodes)
        ]
        self._intra_group = self._intra_groups[self.node_id]

        # SDMA (XGMI copy-engine) intra all-gather over this node's sub-group. One
        # persistent handle drives both step-1 and step-3 gathers (same sub-group +
        # group-position; only input and dest node-block differ). Requires
        # MORI_ENABLE_SDMA=1 + shmem initialized.
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
            # Twin transit for step-3 (disjoint out_). Every rank builds it in lockstep
            # (ShmemMalloc is collective-symmetric); sized to one node-block.
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

        SDMA path: PUSH-gather over XGMI into the contiguous node-block
        ``out_block_1d``. Native fallback: dist.all_gather into the per-slot views
        ``out_slots``. Both leave the node-block in local-index order, bit-exact.
        """
        hnd = sdma_handle if sdma_handle is not None else self._sdma
        if hnd is not None:
            # The sub-group gather's default sync issues a WORLD host ShmemBarrierAll;
            # its only job is inter-op ordering on the shared transit out_ for THIS
            # node's sub-group. Replace 1:1 with a node-local barrier (self._intra_group):
            # the exit barrier ensures all peers finished pushing before the next op
            # reuses out_. Node-local invariant, data path byte-for-byte unchanged.
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

        Equivalent to ``_post`` immediately followed by ``_complete``; on return
        every consumer-visible byte is in HBM (the landing fence).
        """
        h = self._post(inp, out, numel, stream)
        if h is None:
            return True
        self._complete(h)
        return True

    def call_async(self, inp, out, numel, stream=None):
        """Non-blocking all-gather. Returns a handle whose ``_complete`` runs the
        landing fence (wait_all + rail-pair barrier + step 3).

        Caller MUST run ``_complete(handle)`` before reading ``out`` and before the
        NEXT all-gather (staging heap holds a single in-flight op). Returns None for
        the single-node degenerate path.
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

        # Run every GPU op on the CALLER's stream, not the default: a consumer (FSDP)
        # records its completion event on this stream to gate downstream compute; ops on
        # the default stream would be untracked and could race the gather.
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

            # Staging layout for the cross-node shard exchange. write_remote_off must
            # match the partner's recv offset (symmetric), recv_slot the partner's write
            # into MY heap. Default (ring==1): pe-indexed slots, byte-identical. RING>1:
            # send [0,cap), recv cap*(1+slot) cycled per op so op N and N+1 land in
            # DISJOINT bytes. K forced to 1 (single-chunk ring math).
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
                # Staging heap carries single shards; intra gathers write straight into
                # out (no full-output copy-out).
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

            # Whole-shard exchange: one write, one landing barrier, one step-3 gather
            # (SDMA cannot chunk-with-stride; K forced to 1).
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

            # step 1 (intra XGMI, overlapped): gather my node's block into out while the
            # fabric writes are in flight. TWIN: skip step-1's exit barrier (step-3 uses a
            # disjoint transit, no intra-AG WAR; _complete's end barrier frees it).
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
            # TWIN: drive step-3 on the disjoint transit (self._sdma3); the trailing
            # barrier frees BOTH transits for the next op.
            self._intra_ag(
                recv_k,
                slots_k,
                out[obase * e : (obase + rpn) * e],
                o1 - o0,
                stream,
                sdma_handle=self._sdma3,
            )

        with torch.cuda.stream(stream):
            # step 3 (intra XGMI): as each cross-node chunk lands, broadcast it into out.
            for k in range(K):
                if sts[k] is None:
                    continue
                rc = self._engine.wait_all([sts[k]], self._timeout_ms)
                if rc != self._StatusCode.SUCCESS:
                    raise RuntimeError(f"HostProxy inter-node write rc={rc}")
                # rail-pair barrier: partner's write of chunk k into MY sv[partner] has
                # landed (partner's wait_all returned before it entered this barrier).
                dist.barrier(group=self._pair_barrier)
                _bcast_k(k)
        return True
