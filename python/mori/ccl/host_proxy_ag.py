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
# HIERARCHICAL HOST-PROXY ALL-GATHER (persistent, reusable collective).
#
# Motivation: the GPU-initiated device RDMA-WRITE inter-node leg tops out at
# ~31 GB/s/NIC (the per-QP posting wall on this fabric) because a single CTA
# cannot keep a deep enough send queue in flight. A persistent CPU proxy that
# posts a deep multi-WQE SQ pipeline sustains ~48 GB/s/NIC on the SAME NIC.
# The win only materialises when the ONE cross-node node-block is FANNED across
# every data NIC concurrently (rail-paired, no G x redundancy), so aggregate
# cross-node fill climbs 48 -> ~4x48 on a 4-GPU node.
#
# This module provides that transport as a drop-in collective with the SAME
# callable contract the device HierAllGather uses --  handle(inp, out, numel,
# stream) -> bool -- so it can back both the standalone sweep (gate 1) and the
# FSDP custom all-gather (gate 2). Only the node-block exchange rides the
# CPU-posted host-ibverbs transport; the intra-node legs ride XGMI (NCCL).
#
# Layout assumption (matches HierAllGather / torchrun node-major ranks):
#   pe p -> node p // ranks_per_node, local index p % ranks_per_node.
# Rail partner of pe p = same local index on the other node. Two nodes only;
# num_nodes==1 degenerates to a pure intra-node all-gather.
#
# Bit-exact by construction (same bytes, same slots as RCCL all_gather). The
# host completion (wait_all) plus the post-exchange barrier are the landing
# fence: the received remote shard is physically in HBM before any consumer
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

        # Tunables (env overridable) -- the deep-SQ regime validated on this fabric.
        qp = qp_per_transfer or int(os.environ.get("MORI_HOSTPROXY_QP", "4"))
        wt = num_worker_threads or int(os.environ.get("MORI_HOSTPROXY_WT", "1"))
        chunk = chunk_bytes or int(
            os.environ.get("MORI_HOSTPROXY_CHUNK", str(64 * 1024))
        )
        self._timeout_ms = int(os.environ.get("MORI_HOSTPROXY_TIMEOUT_MS", "60000"))
        # THESIS: the intra-node gather MUST ride the SDMA copy engine over XGMI
        # (no CU / no NCCL) so the CUs stay free for the backward GEMM. When this
        # is set the two intra all-gather legs go through mori's
        # IntraNodeSubGroupAllgatherSdma (SDMA PUSH gather over XGMI) instead of
        # dist.all_gather (NCCL, which drives bytes on CU/SM). Default OFF keeps
        # the NCCL legs so the two paths can be A/B'd bit-exact.
        self._sdma_intra = os.environ.get("MORI_HOSTPROXY_SDMA_INTRA", "0") not in (
            "0",
            "",
            "false",
            "False",
        )
        # Cross-node/step-3 pipeline depth. K=1 reproduces the serial path
        # (one write, one landing barrier, one step-3 gather). K>1 splits the
        # node-block exchange into K chunks so the step-3 intra broadcast of
        # chunk k overlaps the still-in-flight cross-node writes of chunks >k
        # (fabric ~48 GB/s/NIC is the exposed long pole vs XGMI ~200 GB/s, so
        # the cross-node tail beyond step-1 is what step-3 can hide).
        self._pipe_chunks = int(os.environ.get("MORI_HOSTPROXY_PIPE_CHUNKS", "1"))
        # DIRECT-to-output stream-ordered SDMA intra gather. When set (and the
        # SDMA intra path is active) the step-1/step-3 gathers PUSH straight into
        # the user output with an on-stream ShmemBarrierOnStream completion --
        # removing the finish_sync host hipStreamSynchronize + host ShmemBarrierAll
        # + transit->output copy-OUT (2 host stalls/AG) that pin the async
        # overlap ceiling at ~0.78x native. On-thesis (bulk bytes stay on SDMA)
        # and matches the shipped device-path coherence contract. Default OFF.
        self._sdma_direct = os.environ.get("MORI_HOSTPROXY_SDMA_DIRECT", "0") not in (
            "0",
            "",
            "false",
            "False",
        )
        # TWIN-TRANSIT (MORI_HOSTPROXY_SDMA_TWIN=1, default ON). In the shared-handle
        # path step-1 (own node block, in _post) and step-3 (remote node block, in
        # _complete) both drive the SAME IntraNodeSubGroupAllgatherSdma handle -> the
        # SAME internal transit ``out_``. Reusing one transit forces TWO node-local
        # dist.barrier per AG: the step-1 EXIT barrier must free out_ (all peers
        # finished the step-1 copy-OUT that reads out_) before step-3 SDMA-pushes into
        # it, and the step-3 EXIT barrier frees out_ for the NEXT op. Giving step-3 its
        # OWN handle (disjoint transit) removes the intra-AG WAR entirely: step-1 needs
        # no exit barrier because step-3 no longer touches step-1's transit, and a
        # SINGLE barrier at the end of _complete frees BOTH transits for the next op.
        # Net: 2->1 node-local barrier per AG. Costs one extra transit ShmemMalloc
        # (sized to one node-block). Only the non-direct 2-node SDMA-intra path
        # benefits; single-node / direct keep the single-handle contract. Copy-OUT
        # unchanged so bytes are identical. Measured ~+1% E2E, bit-exact, no
        # regression on the SDMA_INTRA=0 path. Set =0 to disable.
        self._sdma_twin = os.environ.get("MORI_HOSTPROXY_SDMA_TWIN", "1") not in (
            "0",
            "",
            "false",
            "False",
        )
        # Triage: direct PUSH (copy-out eliminated) but with a per-call HOST
        # completion fence restored, to isolate copy-out elimination from the
        # stream-barrier weakening (the direct/stream path drifts/NaNs E2E).
        self._sdma_direct_hostsync = os.environ.get(
            "MORI_HOSTPROXY_SDMA_DIRECT_HOSTSYNC", "0"
        ) not in ("0", "", "false", "False")
        # BARRIER-FREE per-sub-chunk landing flag (MORI_HOSTPROXY_PIPE_FLAG=1).
        # The K-way pipelined _complete's cross-rank landing point is a per-chunk
        # dist.barrier(pair) -- a collective that is where the standalone pipeline
        # perf dies. Replace it with a tiny point-to-point RDMA flag: after
        # sub-chunk k's DATA send-CQ drains (my write of chunk k landed in the
        # partner's staging), I
        # RDMA a generation-stamped flag into the partner's flag buffer; the
        # partner spins on that flag (cheap pinned-host poll) before consuming
        # chunk k -- NO collective barrier. A GENERATION counter (monotone, never
        # reset) makes it correctness-safe across the many E2E AGs with no per-op
        # reset barrier: the receiver waits flag[k] >= gen (this op's stamp), so a
        # stale prior-op value can never satisfy the wait. FSDP issues AGs in a
        # deterministic order on every rank, so gen stays synchronized rank-to-
        # rank. This also LETS the SDMA-intra path run K>1 (the per-chunk collective
        # barrier that made SDMA K>1 lose is gone). Default OFF => byte-identical.
        self._pipe_flag = os.environ.get("MORI_HOSTPROXY_PIPE_FLAG", "0") not in (
            "0",
            "",
            "false",
            "False",
        )
        self._flag_gen = 0

        # ASYNC completion-ordering diagnostic (MORI_HOSTPROXY_ASYNC_DIAG=1,
        # default OFF => byte-identical). Counts _post vs _complete calls. The
        # ASYNC path (MORI_HOSTPROXY_ASYNC=1) hit 267 TFLOPS (>native) but NaN;
        # the leading hypothesis is that FSDP never calls Work.wait() on some AGs,
        # so their _complete (step-3 remote-half broadcast) never runs and the
        # remote node-block of ``out`` stays garbage -> NaN. If n_complete <
        # n_post at teardown/dump, that hypothesis is confirmed and the fix is a
        # self-healing drain of un-waited ops before their output is consumed. If
        # the counts stay equal, the NaN is instead a staging-heap reuse race.
        self._async_diag = os.environ.get("MORI_HOSTPROXY_ASYNC_DIAG", "0") not in (
            "0",
            "",
            "false",
            "False",
        )
        self._n_post = 0
        self._n_complete = 0

        # ASYNC double-buffered receive staging (MORI_HOSTPROXY_ASYNC_RING, default
        # 1 = OFF = byte-identical). The ASYNC path's NaN is a staging read/write
        # race (diag: FSDP DOES wait() every AG, so no completion is skipped): after
        # the rail-pair barrier, _complete(N)'s step-3 GPU gather READS my recv
        # staging slot, but the partner is then free to run op(N+1)'s _post which
        # RDMA-WRITES the SAME staging bytes; under deferral the read can be
        # overtaken -> torn recv -> NaN. RING>=2 lands op N and op N+1 in DISJOINT
        # byte regions so the partner's next write cannot clobber this op's read.
        # The heap (cap*world bytes) already dwarfs the 2 shards an op uses, so the
        # ring regions are carved from the existing allocation (no extra memory, no
        # host fence, bulk bytes stay on RDMA/SDMA). Both ranks derive the slot from
        # the same monotone op counter (FSDP issues AGs in identical order per rank).
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
        self._sdma3 = None  # twin transit for step-3 (see _sdma_twin)
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
            # never contends step-1's transit. Only for the non-direct 2-node path
            # (direct pushes into the user output, no transit to double). Every rank
            # builds it in lockstep (ShmemMalloc is collective-symmetric). Sized to
            # exactly one node-block (_max_bytes // num_nodes) so the 2nd malloc fits
            # the heap alongside the first handle (which over-allocates full output).
            if self._sdma_twin and not self._sdma_direct and self.num_nodes == 2:
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
        base_port = int(os.environ.get("MORI_HOSTPROXY_BASE_PORT", "31500"))
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

        # BARRIER-FREE per-sub-chunk landing-flag session (MORI_HOSTPROXY_PIPE_FLAG).
        # flag_recv: partner RDMA-writes my per-chunk landing generation here (pinned
        # host so the consume-side spin is a cheap host read). flag_send: the source
        # buffer I stamp with the current generation and RDMA into the partner's
        # flag_recv. Sized to the max pipeline depth. Only built for the 2-node path.
        self._flag_session = None
        if self._pipe_flag:
            kmax = max(1, self._pipe_chunks)
            self._flag_recv = torch.zeros(
                kmax, dtype=torch.int64, device="cpu"
            ).pin_memory()
            self._flag_send = torch.zeros(
                kmax, dtype=torch.int64, device="cpu"
            ).pin_memory()
            self._flag_slots = kmax
            flag_recv_mem = self._engine.register_torch_tensor(self._flag_recv)
            flag_send_mem = self._engine.register_torch_tensor(self._flag_send)
            my_frdesc = flag_recv_mem.pack()
            all_frdesc = [None] * npes
            dist.all_gather_object(all_frdesc, my_frdesc)
            dist.barrier()
            partner_frecv = MemoryDesc.unpack(all_frdesc[self._partner])
            self._flag_session = self._engine.create_session(
                flag_send_mem, partner_frecv
            )
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
        out_full=None,
        block_off_elems=0,
        sdma_handle=None,
        barrier_after=True,
    ):
        """Gather ``count``-element shards over this node's local sub-group.

        SDMA path (on-thesis): PUSH-gather over XGMI straight into the
        contiguous node-block ``out_block_1d`` -- bulk bytes ride the copy
        engine, CUs stay free. When ``self._sdma_direct`` is set and the full
        output tensor ``out_full`` (+ node-block element offset
        ``block_off_elems``) is supplied, the gather PUSHes STRAIGHT into
        ``out_full`` with an on-stream completion fence -- no transit copy-OUT
        and no host stall (the async-overlap lever). NCCL fallback:
        dist.all_gather into the per-slot views ``out_slots``. All variants
        leave the node-block laid out in local-index order, bit-exact.
        """
        hnd = sdma_handle if sdma_handle is not None else self._sdma
        if hnd is not None:
            if self._sdma_direct and out_full is not None:
                hnd.call_direct(
                    inp_1d,
                    out_full,
                    count,
                    block_off_elems,
                    stream,
                    host_sync=self._sdma_direct_hostsync,
                )
            else:
                # The non-direct sub-group gather's default prepare_sync/finish_sync
                # each issue a WORLD host ShmemBarrierAll (runtime.cpp:218 -> the mori
                # socket-bootstrap TCP recursive-doubling barrier). Driven per-AG on
                # this hot path -- thousands of times per training step -- that host
                # barrier is both slow (~84 vs ~160 tflops/gpu for the NCCL leg) and
                # FRAGILE: it reproducibly fails ("Barrier operation failed") at ~step
                # 15 of the E2E, the second defect behind the SDMA-intra hang (the NCCL
                # leg at line below uses dist.all_gather with no such barrier and runs
                # 60 steps clean). The barrier's only job is inter-op ordering on the
                # shared transit ``out_``, and the gather involves ONLY this node's
                # sub-group -- a WORLD barrier is both wrong-scope and needlessly on the
                # fragile bootstrap channel. Replace it 1:1 with a NODE-LOCAL barrier on
                # the robust torch PG (self._intra_group, the 8 same-node ranks): the
                # entry barrier frees every peer's ``out_`` from the previous op, the
                # exit barrier ensures all peers finished pushing before copy-out. Both
                # are node-local invariants, so the data path is byte-for-byte unchanged
                # (prepare_sync(barrier=False) only skips the ShmemBarrierAll; the
                # monotonic flag token, arg setup and copy-out are untouched).
                #
                # ONE barrier per AG suffices: this EXIT barrier fires after finish_sync
                # (which stream-syncs this rank's copy-out of out_), so once all 8 peers
                # pass it every rank has copied out this op -> the NEXT op's pushes into
                # out_ are safe with NO separate entry barrier (the first op needs none:
                # out_ is freshly allocated). Halving the per-AG host-barrier count keeps
                # the SDMA leg off the critical path (perf, rule 2) while preserving the
                # inter-op ordering invariant.
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

        Splitting the op lets the ~1.5ms cross-node CPU-posted RDMA round trip +
        the intra XGMI/SDMA gather overlap the CALLER's compute: the host thread
        is free between post() and complete() instead of stalling mid-AG. The
        caller MUST run ``_complete(handle)`` before reading ``out`` and before
        the NEXT all-gather (the staging heap holds a single in-flight op). This
        is the overlap window native RCCL gets by returning a Work; the sync
        ``__call__`` path forfeits it. Returns None for the single-node
        degenerate path (already blocking, nothing to defer).
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

        # STREAM CORRECTNESS: run every GPU op (stage copy, intra all_gathers,
        # pair barrier) on the CALLER's stream, not the default stream. A
        # cross-node consumer (FSDP) records its completion event on THIS stream
        # and gates the downstream compute on it; if the intra gathers ran on the
        # default stream that event would not track them and the consumer would
        # race the gather -- observed as an E2E loss drift (+0.021) even though
        # the host completion fence guarantees LANDING. For the standalone UT
        # (stream == default) this is a no-op, so bit-exact BW is unchanged.
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
                    out_full=out,
                    block_off_elems=base * e,
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

            # Element-space chunk boundaries for the K-way cross-node/step-3 pipeline.
            # The SDMA intra path uses the serial K=1 form (chunk-pipelining was
            # refuted at the Python level; the SDMA gather has its own global
            # fence per call, so per-chunk barriers would only add cost).
            # SDMA-intra path K selection. The NON-DIRECT SDMA gather packs the
            # rpn slots CONTIGUOUSLY (slot i at out_block[i*count:]) with slot
            # stride == count, so it can only place a WHOLE shard (K=1); a chunked
            # gather (count<e) would pack chunks instead of writing them at the
            # e-strided per-slot offset -> wrong output.
            # Only the DIRECT path (call_direct with out_full + block_off_elems)
            # writes at the correct e-stride+offset, so SDMA K>1 REQUIRES direct.
            # The NCCL fallback gathers into per-slot offset VIEWS (slots_k), so it
            # handles K>1 fine. The barrier-free pipe-flag removes the per-chunk
            # collective barrier but does NOT change these layout constraints.
            if self._async_ring > 1:
                K = 1  # ring layout carves single-shard recv regions (no chunking)
            elif self._sdma is not None and not self._sdma_direct:
                K = 1  # non-direct SDMA cannot chunk-with-stride; force whole shard
            else:
                K = max(1, self._pipe_chunks)
            if self._flag_session is not None and K > self._flag_slots:
                K = self._flag_slots
            bounds = [(k * e) // K for k in range(K + 1)]
            # Per-op landing-flag generation stamp (monotone; receiver waits >= gen).
            self._flag_gen += 1
            flag_gen = self._flag_gen

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
                out_full=out,
                block_off_elems=base * e,
                barrier_after=(self._sdma3 is None),
            )

        if self._async_diag:
            self._n_post += 1
            if self._n_post % 200 == 0:
                self._diag_dump("post")

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
            "gen": flag_gen,
        }

    def _diag_dump(self, where):
        import sys

        sys.stderr.write(
            f"HPASYNCDIAG pe={self.my_pe} where={where} "
            f"n_post={self._n_post} n_complete={self._n_complete} "
            f"inflight={self._n_post - self._n_complete}\n"
        )
        sys.stderr.flush()

    def _complete(self, h):
        """Blocking half: for each cross-node chunk, drain its host CQE
        (wait_all), rail-pair barrier (partner's write into MY sv landed), then
        run the step-3 intra broadcast into ``out``. On return every
        consumer-visible byte is in HBM (the landing fence)."""
        if self._async_diag:
            self._n_complete += 1
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
                out_full=out,
                block_off_elems=obase * e + o0,
                sdma_handle=self._sdma3,
            )

        with torch.cuda.stream(stream):
            if self._flag_session is not None:
                # BARRIER-FREE landing: for each chunk k, drain MY data send-CQ
                # then RDMA a generation-stamped flag into the partner. To CONSUME
                # chunk k I wait for the PARTNER's flag[k] >= gen (its write of
                # chunk k into MY staging has landed), then issue the intra
                # broadcast on the stream. No collective barrier; the point-to-
                # point flag is the exact cross-rank order point. Broadcasts queue
                # on the stream, so broadcast k runs on the GPU while the host
                # spins for flag k+1 -- the pipeline the collective barrier killed.
                gen = h["gen"]
                self._flag_send.fill_(gen)
                for k in range(K):
                    if sts[k] is None:
                        continue
                    rc = self._engine.wait_all([sts[k]], self._timeout_ms)
                    if rc != self._StatusCode.SUCCESS:
                        raise RuntimeError(f"HostProxy inter-node write rc={rc}")
                    # my chunk k landed remotely -> signal partner (and pump the
                    # tiny flag write so POLLING mode drives it to the wire; without
                    # this both ranks spin on each other's flag = symmetric hang).
                    uid = self._engine.allocate_transfer_uid()
                    fst = self._flag_session.write(k * 8, k * 8, 8, uid)
                    self._engine.wait_all([fst], self._timeout_ms)
                import time as _time

                for k in range(K):
                    if sts[k] is None:
                        continue
                    deadline = _time.perf_counter() + self._timeout_ms / 1000.0
                    while self._flag_recv[k].item() < gen:
                        if _time.perf_counter() > deadline:
                            raise RuntimeError(
                                f"HostProxy pipe-flag timeout k={k} gen={gen}"
                            )
                    _bcast_k(k)
                return True
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
