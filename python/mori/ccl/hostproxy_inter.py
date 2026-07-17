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
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# HOST-PROXY INTER-NODE PRODUCER for the fused-ring reassembly consumer.
#
# The device fused-ring giant-AG kernel (FusedRingRemoteGatherKernel_u32) splits
# into three concurrent CTA groups: ring channels (inter-node RDMA fill), the
# local-block SDMA gather, and the reassembly workers that SDMA-push each landed
# ring slot into the output. On this mlx5 provider the device per-sub-chunk
# landing signal is not usable (WRITE_WITH_IMM HW-faults; put-signal/quiet races
# mismatch or crash at large sizes), so the reassembly cannot safely overlap the
# inter fill on-device -- the two run serial.
#
# This producer moves the inter-node leg off the device: with
# MORI_HIER_HOSTPROXY_REASM=1 the device ring-send CTAs skip the RDMA send and the
# reassembly workers spin on host-published chunkReadyFlags[f]. A persistent CPU
# proxy RDMA-writes each ring chunk into the partner's device ring buffer, drains
# its send-CQ (the host-drain landing fence, bit-exact at the giant-AG size where
# the device signal dies), rail-pair barriers so the partner's write into this
# rank's ring buffer has landed, then publishes chunkReadyFlags[f] device-visibly.
# The device reassembly worker then SDMA-pushes chunk f the instant its host flag
# lands, overlapping the still-in-flight later chunks -- the same pipeline the
# device signal could not make bit-exact.
#
# Flat crown ring (rb==1, ring_size==N==2): PE(node,L) rail-exchanges its single
# chunk with PE(1-node,L). ring buffer slot m == node m's chunk (ring order); this
# PE owns slot node_id (its own input, copied in by prepare_stream_only) and must
# RECEIVE slot (1-node) from its partner. One landing flag (f==0) for the single
# remote chunk.
#
# Bit-exact by construction: same ring-buffer bytes as the device RDMA send would
# have produced (slot m*chunkBytes, matching all_gather.hpp ringBase[m*sliceElems]);
# the send-CQ drain + rail-pair barrier order the landing before the flag; the flag
# is published only after the barrier so the device reassembly reads landed bytes.

import os
import socket
import threading
import queue as _queue

import torch
import torch.distributed as dist


def _local_ip(peer_ip):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((peer_ip, 80))
        return s.getsockname()[0]
    finally:
        s.close()


class HostProxyInterProducer:
    """Persistent CPU proxy that owns the crown fused-ring inter-node leg.

    Constructed once (engine + rail-partner session registered against the DEVICE
    ring buffer live for the object lifetime). ``fill(chunk_bytes, deep_pipe,
    flags_tensor, stream)`` is the per-giant-AG hot path: RDMA-writes this PE's
    ring chunk into the partner's ring slot, drains its send-CQ, rail-pair
    barriers, then publishes the reassembly landing flags for the received chunk.
    """

    # Process-wide SHARED engine (composition IOEngine reuse).
    # FSDP builds a SEPARATE HierAllGather per param group (root embed+lm_head AND
    # every decoder layer), so with MORI_HIER_HOSTPROXY_REASM=1 a process would
    # otherwise stand up ~29 DISTINCT host ibverbs IOEngines, each with its own
    # RDMA backend + QPs + bootstrap listener, ALL coexisting with crown's device
    # IBGDA shmem stack on the same 8 mlx5 NICs. That dual-stack fan-out is the
    # documented nondeterministic bootstrap hang: create_backend
    # handshakes race, QP/NIC resources contend. Fix: ONE shared engine+backend
    # per process; each producer instance only registers ITS ring buffer as a new
    # MemoryDesc + creates a session against the partner's matching ring mem. The
    # engine + partner remote-engine registration happen exactly once. This
    # collapses ~29 coexisting engines -> 1 (director 22:26Z: "producer must REUSE
    # crown's IOEngine, not open a coexisting one").
    _shared_engine = None  # the one process-wide IOEngine
    _partner_registered = False  # partner's engine desc registered once
    _pair_barriers = None  # cached rail-pair group (built once)
    _instance_seq = 0

    @classmethod
    def _ensure_shared_engine(cls, my_pe, npes, ranks_per_node, partner, dbglog):
        """Build the ONE process-wide host ibverbs engine + backend and register
        the rail partner's engine, exactly once. Collective (every rank calls in
        lockstep). Serialized bring-up avoids concurrent create_backend races."""
        from mori.io import (
            IOEngine,
            IOEngineConfig,
            RdmaBackendConfig,
            BackendType,
            EngineDesc,
            PollCqMode,
            set_log_level,
        )

        if cls._shared_engine is not None:
            return
        set_log_level("error")
        qp = int(os.environ.get("MORI_HOSTPROXY_QP", "4"))
        wt = int(os.environ.get("MORI_HOSTPROXY_WT", "1"))
        chunk = int(os.environ.get("MORI_HOSTPROXY_CHUNK", str(64 * 1024)))
        master_ip = os.environ["MASTER_ADDR"]
        my_ip = _local_ip(master_ip)
        base_port = int(os.environ.get("MORI_HOSTPROXY_INTER_BASE_PORT", "32600"))
        # single engine per process -> single port block per rank (my_pe offset).
        port = base_port + my_pe

        rcfg = RdmaBackendConfig(
            qp_per_transfer=qp,
            post_batch_size=-1,
            num_worker_threads=wt,
            poll_cq_mode=PollCqMode.POLLING,
            enable_transfer_chunking=True,
            chunk_bytes=chunk,
            num_nics_per_transfer=1,
        )
        rcfg.max_send_wr = 512
        rcfg.max_cqe_num = 2048
        rcfg.max_msg_sge = 1
        # Serialized bring-up: bind ONE engine+backend per rank, one rank at a
        # time behind a rank-ordered barrier so no two create_backend control-
        # plane handshakes overlap (tcp.cpp:166 EADDRINUSE was an ACTIVE bind
        # race, not TIME_WAIT). Retry with a fresh port on residual EADDRINUSE.
        max_try = int(os.environ.get("MORI_HOSTPROXY_INTER_BRINGUP_RETRY", "4"))
        eng = None
        for _r in range(npes):
            if my_pe == _r:
                last_exc = None
                for _t in range(max_try):
                    try:
                        cur_port = port + _t * npes
                        cfg = IOEngineConfig(host=my_ip, port=cur_port)
                        eng = IOEngine(key=f"hpinter-{my_pe}", config=cfg)
                        dbglog(
                            f"shared create_backend begin (port={cur_port} try={_t})"
                        )
                        eng.create_backend(BackendType.RDMA, rcfg)
                        dbglog("shared create_backend done")
                        last_exc = None
                        break
                    except Exception as e:  # noqa: BLE001
                        last_exc = e
                        dbglog(f"shared bring-up try={_t} failed: {e}")
                        eng = None
                if last_exc is not None:
                    raise last_exc
            dist.barrier()
        # exchange engine descs ONCE, register the rail partner's engine ONCE.
        my_edesc = eng.get_engine_desc().pack()
        all_edesc = [None] * npes
        dist.all_gather_object(all_edesc, my_edesc)
        dist.barrier()
        eng.register_remote_engine(EngineDesc.unpack(all_edesc[partner]))
        dist.barrier()
        cls._shared_engine = eng
        cls._partner_registered = True
        dbglog("shared engine ready")

    @classmethod
    def _ensure_pair_barriers(cls, my_pe, ranks_per_node):
        """Build the rail-pair process groups ONCE (dist.new_group is collective +
        creates a comm; building 29x per param group is wasteful)."""
        if cls._pair_barriers is not None:
            return cls._pair_barriers
        mine = None
        for i in range(ranks_per_node):
            pair = sorted([i, ranks_per_node + i])
            g = dist.new_group(ranks=pair)
            if my_pe in pair:
                mine = g
        cls._pair_barriers = mine
        return mine

    def __init__(self, my_pe, npes, ranks_per_node, ring_buf_ptr, ring_buf_bytes):
        from mori.io import (
            MemoryDesc,
            StatusCode,
            MemoryLocationType,
        )

        self._StatusCode = StatusCode
        self._dbg = os.environ.get("MORI_HOSTPROXY_DEBUG", "0") not in (
            "0",
            "",
            "false",
        )

        def _dbg(msg):
            if self._dbg:
                import sys as _s

                _s.stderr.write(f"[hpinter pe{my_pe}] {msg}\n")
                _s.stderr.flush()

        self._dbglog = _dbg
        _dbg("ctor begin")

        self.my_pe = my_pe
        self.npes = npes
        self.ranks_per_node = ranks_per_node
        self.num_nodes = npes // ranks_per_node
        self.node_id = my_pe // ranks_per_node
        self.local_rank = my_pe % ranks_per_node
        self._ring_ptr = int(ring_buf_ptr)
        self._ring_bytes = int(ring_buf_bytes)
        self._timeout_ms = int(os.environ.get("MORI_HOSTPROXY_TIMEOUT_MS", "60000"))

        if self.num_nodes != 2:
            raise NotImplementedError(
                "HostProxyInterProducer supports exactly 2 nodes "
                f"(got num_nodes={self.num_nodes})"
            )

        other_node = 1 - self.node_id
        self._partner = other_node * ranks_per_node + self.local_rank
        _dbg(f"partner={self._partner} node={self.node_id} L={self.local_rank}")

        # rail-pair barrier (cached process-wide) + side stream for flag publish.
        self._pair_barrier = self._ensure_pair_barriers(my_pe, ranks_per_node)
        self._flag_stream = None

        # Build the ONE shared engine + backend + partner registration (once).
        _inst = HostProxyInterProducer._instance_seq
        HostProxyInterProducer._instance_seq += 1
        self._ensure_shared_engine(my_pe, npes, ranks_per_node, self._partner, _dbg)
        self._engine = HostProxyInterProducer._shared_engine

        # Per-instance: register THIS ring buffer + create a session against the
        # partner's matching ring mem. Only the memory descs are exchanged per
        # producer; the engine/backend/partner-engine are shared.
        dev_id = torch.cuda.current_device()
        self._local_mem = self._engine.register_memory(
            self._ring_ptr, self._ring_bytes, dev_id, MemoryLocationType.GPU
        )
        my_mdesc = self._local_mem.pack()
        all_mdesc = [None] * npes
        dist.all_gather_object(all_mdesc, my_mdesc)
        dist.barrier()
        remote_mem = MemoryDesc.unpack(all_mdesc[self._partner])
        self._session = self._engine.create_session(self._local_mem, remote_mem)
        dist.barrier()

        # BARRIER-FREE point-to-point landing flag (MORI_HIER_HOSTPROXY_INTER_FLAG):
        # the default fill() proves the PARTNER's write into MY ring slot has
        # landed via a per-AG COLLECTIVE dist.barrier(pair) -- a collective on the
        # hot path (fires on every fuse_remote AG) that both serializes the two
        # rail peers AND, being a c10d collective, cannot run off the main Python
        # thread (blocks any async-overlap). Replace it with a generation-stamped
        # point-to-point RDMA flag (the mechanism validated barrier-free at
        # 256/466MB): after MY data send-CQ drains, RDMA a monotone gen stamp into
        # the partner's pinned-host flag_recv; I spin on MY flag_recv >= gen (the
        # partner's write of its chunk into MY ring slot has landed, since it only
        # posts the flag AFTER draining its own data CQ). No collective => lower
        # per-AG latency AND thread-safe for async fill. gen stays rank-synced
        # because E2E AGs fire in deterministic order. Default OFF => collective.
        self._inter_flag = os.environ.get(
            "MORI_HIER_HOSTPROXY_INTER_FLAG", "0"
        ) not in ("0", "", "false", "False")
        self._flag_gen = 0
        self._flag_session = None
        # async worker (MORI_HIER_HOSTPROXY_ASYNC), lazily started in fill_async.
        self._worker = None
        self._worker_q = None
        self._worker_exc = None
        if self._inter_flag:
            self._flag_recv = torch.zeros(
                1, dtype=torch.int64, device="cpu"
            ).pin_memory()
            self._flag_send = torch.zeros(
                1, dtype=torch.int64, device="cpu"
            ).pin_memory()
            frecv_mem = self._engine.register_torch_tensor(self._flag_recv)
            fsend_mem = self._engine.register_torch_tensor(self._flag_send)
            my_fr = frecv_mem.pack()
            all_fr = [None] * npes
            dist.all_gather_object(all_fr, my_fr)
            dist.barrier()
            partner_fr = MemoryDesc.unpack(all_fr[self._partner])
            self._flag_session = self._engine.create_session(fsend_mem, partner_fr)
            dist.barrier()
            _dbg("barrier-free inter flag session ready")
        _dbg(f"ctor done (inst={_inst}, shared engine reused)")

    def fill_async(
        self, chunk_bytes, deep_pipe, flags, src_ready_event=None, stream=None
    ):
        """ASYNC overlap (MORI_HIER_HOSTPROXY_ASYNC): submit the (now
        collective-free, barrier-free-flag) inter leg to a persistent background
        worker and return IMMEDIATELY so the main Python thread keeps issuing the
        rest of the step -- the host RDMA write + send-CQ drain + p2p landing flag
        + device flag publish then overlap the caller's compute. Correctness is
        unchanged: the live kernel's reassembly workers spin on chunkReadyFlags,
        which the worker publishes only AFTER the remote landing, and the caller's
        stream-ordered finish/deferred fence blocks the CONSUMER until the kernel
        (hence the flags) completes. Requires the barrier-free flag (no collective
        can run off the main thread). One serial worker preserves AG order and
        avoids CQ contention; backpressure comes from the GPU fence naturally.
        """
        if self._flag_session is None:
            raise RuntimeError(
                "HOSTPROXY_ASYNC requires MORI_HIER_HOSTPROXY_INTER_FLAG=1 "
                "(the collective barrier cannot run off the main thread)"
            )
        if self._worker is None:
            self._worker_q = _queue.Queue()
            self._worker_exc = None
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()
        # surface any prior async failure on the issuing thread.
        if self._worker_exc is not None:
            e = self._worker_exc
            self._worker_exc = None
            raise e
        self._worker_q.put((chunk_bytes, deep_pipe, flags, src_ready_event, stream))

    def _worker_loop(self):
        while True:
            item = self._worker_q.get()
            if item is None:
                return
            try:
                self.fill(*item)
            except Exception as e:  # noqa: BLE001
                self._worker_exc = e
            finally:
                self._worker_q.task_done()

    def drain(self):
        """Block until EVERY submitted async fill() has fully completed -- i.e.
        the background worker has posted+drained the inter RDMA leg AND published
        (fs.synchronize) ALL chunkReadyFlags for every AG queued so far.

        This is the async-composition completion-ordering fence. In async mode
        fill_async() returns immediately, so the deferred consumer fence
        (_DeviceDeferredHostSyncWork.wait -> stream.synchronize) could fire at
        copy-out while the off-thread worker is still publishing a later AG's
        chunkReadyFlags -- the reassembly then reads unpublished (stale) flags and
        the loss drifts. Joining the worker queue here forces the flags to be
        published and landed before the caller-stream synchronize gates the
        consumer, restoring the sync path's ordering while keeping the overlap the
        worker gained during compute.
        """
        drained = self._worker_q is not None
        if drained:
            self._worker_q.join()
        if self._worker_exc is not None:
            e = self._worker_exc
            self._worker_exc = None
            raise e
        return drained

    def fill(self, chunk_bytes, deep_pipe, flags, src_ready_event=None, stream=None):
        """Own the inter-node leg for ONE giant AG.

        ``chunk_bytes`` = per-chunk (per-node-block-shard) byte count == the
        device ring's ``chunkBytes`` (== count*elemsize for the flat rb==1 ring).
        ``deep_pipe`` = P temporal sub-chunks (>=1). ``flags`` = the device int64
        chunkReadyFlags tensor (already zeroed by the caller). Publishes flags
        [0, deep_pipe) for the single received remote chunk.

        MUST be called AFTER the caller has copied this PE's input into the ring
        slot (prepare_stream_only) and issued that copy on ``stream``, so the
        source bytes for the RDMA read are device-visible.
        """
        P = deep_pipe if deep_pipe and deep_pipe >= 1 else 1
        self._dbglog(f"fill begin chunk_bytes={chunk_bytes} P={P}")
        # the RDMA read source is the ring slot the caller just copied our input
        # into (prepare_stream_only). Make that copy device-visible before posting.
        if src_ready_event is not None:
            src_ready_event.synchronize()
        self._dbglog("fill src_ready synced")
        # this PE's own chunk sits in ring slot node_id; the partner receives it
        # into ITS ring slot node_id (same slot index -- ring order is by sender
        # node). We WRITE our slot node_id -> partner slot node_id, and RECEIVE
        # our partner's slot (1-node) written by the partner into OUR slot (1-node).
        my_slot_boff = self.node_id * chunk_bytes
        # 16B-aligned sub-chunk tiling matches all_gather.hpp unitsPerChan.
        kAlign = 16
        nUnits = (chunk_bytes + kAlign - 1) // kAlign
        unitsPerP = (nUnits + P - 1) // P

        # POST P temporal sub-chunk writes on the deep SQ (full NIC BW), collect
        # per-sub-chunk send-CQ handles so each landing can be signalled as soon
        # as it drains.
        sts = []
        for p in range(P):
            su = p * unitsPerP
            eu = min(su + unitsPerP, nUnits)
            if su >= eu:
                sts.append(None)
                continue
            off = my_slot_boff + su * kAlign
            nb = min(eu * kAlign, chunk_bytes) - su * kAlign
            uid = self._engine.allocate_transfer_uid()
            sts.append(self._session.write(off, off, nb, uid))
        self._dbglog(f"fill posted {P} writes (slot_boff={my_slot_boff})")

        # Drain each sub-chunk's send-CQ IN ORDER == that sub-range landed
        # remotely (proven host-drain fence). The received remote chunk lands in
        # OUR ring slot (1-node); rail-pair barrier guarantees the partner has
        # finished its write into us before we publish the reassembly flags.
        for p in range(P):
            if sts[p] is None:
                continue
            rc = self._engine.wait_all([sts[p]], self._timeout_ms)
            if rc != self._StatusCode.SUCCESS:
                raise RuntimeError(f"HostProxyInter write p={p} rc={rc}")
        self._dbglog("fill writes drained (landed remote)")

        if self._flag_session is not None:
            # BARRIER-FREE landing: my data landed remotely (drained above), so
            # stamp+RDMA my generation into the partner, then spin until the
            # partner's stamp reaches ME (its write into my ring slot landed).
            # No collective; point-to-point flag is the exact cross-rank order.
            self._flag_gen += 1
            gen = self._flag_gen
            self._flag_send.fill_(gen)
            uid = self._engine.allocate_transfer_uid()
            fst = self._flag_session.write(0, 0, 8, uid)
            self._engine.wait_all([fst], self._timeout_ms)
            import time as _time

            deadline = _time.perf_counter() + self._timeout_ms / 1000.0
            while self._flag_recv[0].item() < gen:
                if _time.perf_counter() > deadline:
                    raise RuntimeError(f"HostProxyInter flag timeout gen={gen}")
            self._dbglog("fill p2p flag landed")
        else:
            # single rail-pair barrier: partner's writes into MY ring slot landed.
            dist.barrier(group=self._pair_barrier)
            self._dbglog("fill pair_barrier passed")

        # publish the reassembly landing flags for the received chunk. The device
        # reassembly worker spins on chunkReadyFlags[f] < 1; write 1 to slots
        # [0, P). Use a fill on a side stream so it is device-visible to the
        # concurrently-running kernel (system-scope AtomicLoadSeqCstSystem reads
        # HBM). A plain torch fill_ on the caller stream would be ordered AFTER
        # the kernel that is spinning -> deadlock; publish on a separate stream.
        if self._flag_stream is None:
            self._flag_stream = torch.cuda.Stream(device=flags.device)
        fs = self._flag_stream
        with torch.cuda.stream(fs):
            flags[:P].fill_(1)
        # ensure the flag stores reach HBM (the kernel polls system-scope).
        fs.synchronize()
        self._dbglog("fill flags published")
