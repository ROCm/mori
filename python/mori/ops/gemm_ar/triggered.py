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
"""GEMM + two-shot all-reduce on host-armed copy-engine chains.

The GEMM's epilogue only releases gates; copy engines move every byte, on chains the
host armed before the GEMM was launched::

    host    arm:  per peer d, for each band b that d owns
                    [POLL rs_gate[b]] [COPY my rows of b -> d's recv slot] [ATOMIC d's rs_arrive]
                  then for each band b that I own
                    [POLL ag_gate[b]] [COPY my reduced b -> d's output] [ATOMIC d's ag_arrive]
                  then [POLL last ag_gate] [ATOMIC my sent[d]]
    GEMM    each work-group stores its tile and adds 1 to its band's rs_gate
    reduce  per band I own, in order: wait for my tiles and every peer's copy, sum,
            store, add to ag_gate
    wait    until every peer's reduced bands have landed in my output and my
            all-gather has finished reading mine

Band ``b`` (``mt1`` rows) belongs to rank ``b % W``, so every link starts early. The
GEMM is the caller's: it writes the partial ``[m, n]`` (bf16, row-major) into
:meth:`GemmTriggeredAllReduceOp.input_view` and releases each band through
:attr:`~GemmTriggeredAllReduceOp.gate_view` (``GateTileBase = 0``)::

    op.arm(m)                                               # host: this call's chains
    op.begin(m)                                             # concurrent mode: reducer
    gemm(..., out=op.input_view(m), gate_view=op.gate_view)  # current stream
    out = op.complete(m)                                    # (reduce +) wait, current stream

``reduce_mode="concurrent"`` launches the reducer in :meth:`begin`, on a
high-priority stream, so bands are reduced while the GEMM runs.

Calls are host-synchronous: arm call k+1 only after call k's :meth:`complete` was
issued, and finish with ``out`` before the next GEMM (concurrent mode: before the
next :meth:`begin`). :meth:`arm` raises if the previous call's device waits timed
out; :meth:`check` also verifies every gate. :meth:`close` is collective. The
window must be cco's default uncached memory.
"""

from __future__ import annotations

import os

import flydsl.expr as fx
import numpy as np
import torch

from mori.cco import TriggerBatch, TriggerQueue, peer_ptr
from mori.tensor_utils import from_gpu_ptr

from .gates import GATE_STRIDE, GateProducerSpec, GateTable, gate_view
from .kernels_trigger import build_trigger_kernels
from .op import _PinnedLaunch

__all__ = ["GemmTriggeredAllReduceOp", "TriggeredArLayout"]

_ALIGN = 4096
_ELEM_BYTES = 2  # bf16
_UNIT_BYTES = 32  # ring bytes per SDMA packet
_CHAIN_BYTES = 3 * _UNIT_BYTES  # POLL, COPY, ATOMIC
_MARK_BYTES = 2 * _UNIT_BYTES  # POLL, ATOMIC
_DEFAULT_RING_BYTES = 256 << 10


def _align(x: int) -> int:
    return (x + _ALIGN - 1) // _ALIGN * _ALIGN


class TriggeredArLayout:
    """Byte layout of one rank's window. Every region starts 4 KiB aligned.

    Control words (gates, arrivals, completion marks, call counts, the error word)
    come first, each on its own 64-byte line, then the partial (``input``), the
    receive slots (one per source, holding that source's rows of the bands I own,
    in band order) and the result (``output``).
    """

    def __init__(self, world_size: int, n: int, m_max: int, mt1: int, publishers: int):
        self.world_size, self.n, self.m_max, self.mt1 = world_size, n, m_max, mt1
        self.publishers = publishers
        self.gate_stride = GATE_STRIDE
        self.band_bytes = mt1 * n * _ELEM_BYTES
        self.max_bands = m_max // mt1
        self.max_bands_per_rank = self.max_bands // world_size
        ws, kmax, gs = world_size, self.max_bands_per_rank, self.gate_stride
        sizes = [
            ("rs_gate_off", self.max_bands * gs),
            ("ag_gate_off", kmax * gs),
            ("rs_arrive_off", ws * kmax * gs),
            ("ag_arrive_off", ws * kmax * gs),
            ("sent_off", ws * gs),
            ("calls_off", self.max_bands * 4),
            ("sent_calls_off", gs),
            ("err_off", gs),
            ("input_off", self.max_bands * self.band_bytes),
            ("recv_off", ws * kmax * self.band_bytes),
            ("output_off", self.max_bands * self.band_bytes),
        ]
        at = 0
        for name, size in sizes:
            setattr(self, name, at)
            at = _align(at + size)
        self.window_bytes = at
        self.control_bytes = self.input_off

    def bands(self, m: int) -> int:
        return m // self.mt1

    def offsets(self) -> tuple:
        return tuple(sorted((k, v) for k, v in vars(self).items()))

    def recv_slot_off(self, src: int) -> int:
        """Slot holding source ``src``'s rows of the bands I own, in band order."""
        return self.recv_off + src * self.max_bands_per_rank * self.band_bytes


class GemmTriggeredAllReduceOp:
    """GEMM + two-shot all-reduce (reduce-scatter and all-gather on copy engines).

    One instance serves one feature count ``n`` and any token count ``m <= m_max``
    with ``m / producer.mt1`` bands divisible by the world size. bf16 only.
    ``reducer_blocks`` work-groups reduce the bands in order, after the GEMM or
    beside it (``reduce_mode``). ``ag_engine_offset``
    ``None`` (default) chains the all-gather behind the reduce-scatter on the link's
    engine; an offset gives it its own queues on another engine (gfx942 only).
    ``wait_polls`` bounds every device wait. Construction is collective.
    """

    #: Engine targets above this need a re-base: the engine compares 32 bits.
    REBASE_AT = 1 << 31

    def __init__(
        self,
        comm,
        *,
        n: int,
        m_max: int,
        producer: GateProducerSpec,
        reducer_blocks: int = 32,
        reduce_mode: str = "after",
        ag_engine_offset: int | None = None,
        ring_bytes: int = 0,
        wait_polls: int = 1 << 24,
        quiesce_timeout_s: float = 30.0,
    ):
        self._closed = True
        self._queues: list[TriggerQueue] = []
        self.mem = self.win = None
        ws = comm.nranks
        bands = m_max // producer.mt1
        if m_max % producer.mt1 or bands % ws or bands < ws:
            raise ValueError(
                f"m_max={m_max} must be a positive multiple of mt1={producer.mt1} "
                f"with a band count divisible by the world size {ws}"
            )
        if (n * _ELEM_BYTES) % 16:
            raise ValueError(f"n={n}: rows must be whole 16-byte packs")
        if reducer_blocks < 1:
            raise ValueError("reducer_blocks must be >= 1")
        if reduce_mode not in ("after", "concurrent"):
            raise ValueError(f"reduce_mode={reduce_mode!r}: 'after' or 'concurrent'")
        if reduce_mode == "concurrent":
            cus = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            if reducer_blocks > cus // 4:
                raise ValueError(
                    f"reducer_blocks={reducer_blocks}: concurrent mode allows at most "
                    f"a quarter of the {cus} CUs"
                )
        if os.environ.get("CCO_UNCACHED_WINDOW", "1") == "0":
            raise ValueError(
                "needs cco's default uncached window (CCO_UNCACHED_WINDOW)"
            )
        ring = ring_bytes or _DEFAULT_RING_BYTES
        per_queue = bands // ws * (2 if ag_engine_offset is None else 1)
        if per_queue * _CHAIN_BYTES + _MARK_BYTES >= ring:
            raise ValueError(
                f"one call needs {per_queue} chains per queue; a {ring}-byte ring "
                f"cannot hold them"
            )
        self.comm, self.rank, self.world_size = comm, comm.rank, ws
        self.n, self.producer = n, producer
        self.reducer_blocks, self.wait_polls = reducer_blocks, wait_polls
        self.reduce_mode = reduce_mode
        self.quiesce_timeout_s = quiesce_timeout_s
        self.rebase_at = self.REBASE_AT  # lowered by tests to force a re-base
        self.layout = TriggeredArLayout(
            ws, n, m_max, producer.mt1, producer.publishers(n)
        )
        lay = self.layout
        self.m_max = m_max
        self._kernels = {}
        self._batches = {}
        self._faulted = None
        self._armed_m = None
        self._begun = False
        self._reduce_stream = None
        if reduce_mode == "concurrent":
            self._reduce_stream = torch.cuda.Stream(
                priority=torch.cuda.Stream.priority_range()[1]
            )
            self._reduce_stream_fx = fx.Stream(self._reduce_stream)
            self._ready = torch.cuda.Event()
            self._reduced = torch.cuda.Event()
        #: Collective re-bases so far.
        self.rebases = 0
        self._err_host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self._err_ready = None
        try:
            self.mem = comm.alloc_mem(lay.window_bytes)
            self.win = comm.register_window(self.mem.ptr, self.mem.size)
            self.base = self.win.local_ptr
            from_gpu_ptr(self.base, (lay.window_bytes,), torch.uint8).zero_()
            torch.cuda.synchronize()
            self.peers = [p for p in range(ws) if p != self.rank]
            self._peer_base = {p: peer_ptr(comm, self.base, p) for p in self.peers}
            # One target array for both tables: the batch reads its slots.
            self._targets = np.zeros(lay.max_bands + lay.max_bands_per_rank, np.uint64)
            self.rs_gates = GateTable(
                self.base + lay.rs_gate_off,
                lay.max_bands,
                expected=self._targets[: lay.max_bands],
            )
            self.ag_gates = GateTable(
                self.base + lay.ag_gate_off,
                lay.max_bands_per_rank,
                expected=self._targets[lay.max_bands :],
            )
            self._view = gate_view(
                self.base,
                [self.rs_gates.addr(b) for b in range(lay.max_bands)],
                publishers=lay.publishers,
                rank=self.rank,
            )
            self.q_rs, self.q_ag = {}, {}
            for p in self.peers:
                self.q_rs[p] = TriggerQueue(comm, p, ring_bytes=ring_bytes)
                self._queues.append(self.q_rs[p])
                if ag_engine_offset is None:
                    self.q_ag[p] = self.q_rs[p]
                else:
                    self.q_ag[p] = TriggerQueue(
                        comm, p, engine_offset=ag_engine_offset, ring_bytes=ring_bytes
                    )
                    self._queues.append(self.q_ag[p])
            comm.barrier()
        except Exception:
            self._release()
            raise
        self._closed = False

    # ── producer interface ────────────────────────────────────────────────────

    @property
    def gate_view(self) -> int:
        """``GateView`` kernel argument (gate-epilogue ABI); ``GateTileBase`` is 0."""
        return self._view.data_ptr()

    def input_view(self, m: int) -> torch.Tensor:
        """``[m, n]`` bf16 view the GEMM writes its partial into."""
        self._check_m(m)
        return from_gpu_ptr(
            self.base + self.layout.input_off, (m, self.n), torch.bfloat16
        )

    def output_view(self, m: int) -> torch.Tensor:
        """``[m, n]`` bf16 view holding the all-reduced result after :meth:`complete`."""
        self._check_m(m)
        return from_gpu_ptr(
            self.base + self.layout.output_off, (m, self.n), torch.bfloat16
        )

    def _check_m(self, m: int) -> None:
        lay = self.layout
        if m < 1 or m > self.m_max or m % lay.mt1 or lay.bands(m) % self.world_size:
            raise ValueError(
                f"m={m}: need a multiple of {lay.mt1} up to {self.m_max} whose band "
                f"count divides by {self.world_size}"
            )

    # ── one call ──────────────────────────────────────────────────────────────

    def _quiesce_all(self) -> TriggerQueue | None:
        """Drain every queue; returns the first that timed out, if any."""
        for q in self._queues:
            if not q.quiesce(timeout_s=self.quiesce_timeout_s):
                return q
        return None

    def arm(self, m: int) -> None:
        """Arm this call's chains on every queue. Host only; launches nothing."""
        self._check_open()
        self._check_m(m)
        if self._armed_m is not None:
            raise RuntimeError(
                f"call with m={self._armed_m} is armed but not completed"
            )
        if self._err_ready is not None:
            self._err_ready.synchronize()
            self._err_ready = None
            if int(self._err_host.item()):
                self._fault("a device wait of the previous call timed out")
        lay = self.layout
        bands = lay.bands(m)
        if (
            self.rs_gates.high() + lay.publishers >= self.rebase_at
            or self.ag_gates.high() + self.reducer_blocks >= self.rebase_at
        ):
            self._rebase()
        batch = self._batch(m)
        snapshot = self._targets.copy()
        status = None
        try:
            self.rs_gates.expect_first(bands, lay.publishers)
            self.ag_gates.expect_first(bands // self.world_size, self.reducer_blocks)
            status = batch.arm(self._targets, self.quiesce_timeout_s)
        except Exception as e:
            self._targets[:] = snapshot
            self._fault(f"arming failed: {e!r}", release=False)
        if status != TriggerBatch.OK:
            # Nothing of this call was appended; a queue stuck on the previous call has
            # its chains released (the restored targets are exactly that call's).
            self._targets[:] = snapshot
            stuck = status == TriggerBatch.DRAIN_TIMEOUT
            why = "did not drain the previous call" if stuck else "has no room"
            self._fault(f"a queue {why}", release=stuck)
        self._armed_m = m
        self._begun = False

    def _batch(self, m: int) -> TriggerBatch:
        """This call's chains (built once per m): per peer, reduce-scatter, all-gather, mark."""
        batch = self._batches.get(m)
        if batch is not None:
            return batch
        lay, ws, rank = self.layout, self.world_size, self.rank
        nk = lay.bands(m) // ws
        bb, gs, kmax = lay.band_bytes, lay.gate_stride, lay.max_bands_per_rank
        ag_slot = lay.max_bands
        inp, out = self.base + lay.input_off, lay.output_off
        batch = TriggerBatch()
        for d in self.peers:
            pb = self._peer_base[d]
            slot = pb + lay.recv_slot_off(rank)
            rs_arr = pb + lay.rs_arrive_off + rank * kmax * gs
            ag_arr = pb + lay.ag_arrive_off + rank * kmax * gs
            for k in range(nk):
                b = k * ws + d
                batch.add(
                    self.q_rs[d],
                    self.rs_gates.addr(b),
                    b,
                    [(inp + b * bb, slot + k * bb, bb)],
                    rs_arr + k * gs,
                )
            for k in range(nk):
                b = k * ws + rank
                batch.add(
                    self.q_ag[d],
                    self.ag_gates.addr(k),
                    ag_slot + k,
                    [(self.base + out + b * bb, pb + out + b * bb, bb)],
                    ag_arr + k * gs,
                )
            # Completion mark: runs after the queue's last read of my output.
            batch.add(
                self.q_ag[d],
                self.ag_gates.addr(nk - 1),
                ag_slot + nk - 1,
                [],
                self.base + lay.sent_off + d * gs,
            )
        self._batches[m] = batch
        return batch

    def begin(self, m: int) -> None:
        """Concurrent mode: launches the reducer on the op's stream, after the work
        queued so far on the current stream. Call between :meth:`arm` and the GEMM.

        Does nothing in "after" mode. Call :meth:`complete` even if the GEMM is not
        launched: the reducer's waits then time out and the next :meth:`arm`
        faults.
        """
        self._check_open()
        self._check_m(m)
        if self._armed_m != m:
            raise RuntimeError(
                f"begin(m={m}) does not match the armed call ({self._armed_m})"
            )
        if self._begun:
            raise RuntimeError(f"begin(m={m}) was already called for this call")
        if self._reduce_stream is not None:
            reduce, _ = self._compiled(m)
            self._ready.record(torch.cuda.current_stream())
            self._reduce_stream.wait_event(self._ready)
            reduce(self.base, self.wait_polls, stream=self._reduce_stream_fx)
            self._reduced.record(self._reduce_stream)
        self._begun = True

    def complete(self, m: int) -> torch.Tensor:
        """Launches the final wait on the current stream, after the reduce ("after"
        mode) or after joining the op's stream (concurrent mode).

        Returns the output view, valid once the stream reaches this point.
        """
        self._check_open()
        self._check_m(m)
        if self._armed_m != m:
            raise RuntimeError(
                f"complete(m={m}) does not match the armed call ({self._armed_m})"
            )
        if self._reduce_stream is not None and not self._begun:
            raise RuntimeError(f"complete(m={m}): concurrent mode needs begin(m)")
        self._armed_m = None
        self._begun = False
        reduce, wait = self._compiled(m)
        torch_stream = torch.cuda.current_stream()
        stream = fx.Stream(torch_stream)
        if self._reduce_stream is None:
            reduce(self.base, self.wait_polls, stream=stream)
        else:
            torch_stream.wait_event(self._reduced)
        wait(self.base, self.wait_polls, stream=stream)
        self._err_host.copy_(self._err_view(), non_blocking=True)
        self._err_ready = torch.cuda.Event()
        self._err_ready.record(torch_stream)
        return self.output_view(m)

    def _compiled(self, m: int):
        k = self._kernels.get(m)
        if k is None:
            reduce, wait = build_trigger_kernels(
                self.layout, self.rank, m, reducer_blocks=self.reducer_blocks
            )
            k = (_PinnedLaunch(reduce), _PinnedLaunch(wait))
            self._kernels[m] = k
        return k

    def _err_view(self) -> torch.Tensor:
        return from_gpu_ptr(self.base + self.layout.err_off, (1,), torch.int32)

    def _gate_words(self, off: int, count: int) -> torch.Tensor:
        """One u64 per gate line, as a strided view of the window."""
        lay = self.layout
        step = lay.gate_stride // 8
        return from_gpu_ptr(self.base + off, (count * step,), torch.int64)[::step]

    def error(self) -> int:
        """Synchronises; the error word, nonzero once a device wait gave up."""
        torch.cuda.synchronize()
        return int(self._err_view().item())

    def check(self) -> None:
        """Synchronises; faults if a device wait gave up or a gate is off its target."""
        if self.error():
            self._fault("a device wait timed out")
        lay = self.layout
        rs = self._gate_words(lay.rs_gate_off, lay.max_bands).cpu().tolist()
        ag = self._gate_words(lay.ag_gate_off, lay.max_bands_per_rank).cpu().tolist()
        if (
            rs != self.rs_gates.expected.tolist()
            or ag != self.ag_gates.expected.tolist()
        ):
            self._fault(
                "a gate is not at its target (producer added too little or too much)"
            )

    # ── re-base, fault, teardown ──────────────────────────────────────────────

    def _rebase(self) -> None:
        """Collective: zeroes every control word and target. Every rank reaches it on the
        same call; the barriers keep peers from zeroing or arming while one still runs.
        """
        if self.error():
            self._fault("a device wait timed out (found at re-base)")
        stuck = self._quiesce_all()
        if stuck is not None:
            self._fault(
                f"the queue to rank {stuck.peer} did not drain before a re-base"
            )
        self.comm.barrier()
        from_gpu_ptr(self.base, (self.layout.control_bytes,), torch.uint8).zero_()
        torch.cuda.synchronize()
        self.rs_gates.reset()
        self.ag_gates.reset()
        self.rebases += 1
        self.comm.barrier()

    def _fault(self, why: str, *, release: bool = True) -> None:
        """Sticky: flag every peer, release this rank's parked chains (unless the
        call never got its targets), drain, and refuse further calls."""
        self._faulted = why
        self._armed_m = None
        self._begun = False
        torch.cuda.synchronize()
        lay = self.layout
        # Peers must not trust anything a released chain delivers from here on.
        for p in self.peers:
            from_gpu_ptr(self._peer_base[p] + lay.err_off, (1,), torch.int32).fill_(1)
        if release:
            self._gate_words(lay.rs_gate_off, lay.max_bands).copy_(
                torch.from_numpy(self.rs_gates.expected.astype(np.int64))
            )
            self._gate_words(lay.ag_gate_off, lay.max_bands_per_rank).copy_(
                torch.from_numpy(self.ag_gates.expected.astype(np.int64))
            )
        torch.cuda.synchronize()
        if release:
            self._quiesce_all()
        raise RuntimeError(f"GemmTriggeredAllReduceOp faulted: {why}; re-create the op")

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("GemmTriggeredAllReduceOp is closed")
        if self._faulted is not None:
            raise RuntimeError(
                f"GemmTriggeredAllReduceOp faulted earlier: {self._faulted}"
            )

    def self_test(self, m: int | None = None) -> None:
        """Collective: one call with the gates opened from the host instead of a
        GEMM. Fails fast if a queue never moves (unarmed or dead engine)."""
        m = m if m is not None else self.layout.mt1 * self.world_size
        self.input_view(m).fill_(0.0)
        self.output_view(m).fill_(float("nan"))
        torch.cuda.synchronize()
        self.comm.barrier()
        self.arm(m)
        self.begin(m)
        nb = self.layout.bands(m)
        self._gate_words(self.layout.rs_gate_off, nb).copy_(
            torch.from_numpy(self.rs_gates.expected[:nb].astype(np.int64))
        )
        out = self.complete(m)
        if self.error():
            self._fault("self-test: a device wait timed out")
        if not bool((out == 0).all()):
            self._fault("self-test: output is not the sum of zeros")

    def close(self) -> bool:
        """Collective teardown; returns False if a queue still held unconsumed work."""
        if self._closed:
            return True
        torch.cuda.synchronize()
        self._quiesce_all()
        # No peer may destroy its queues or free its window while one of mine can
        # still be copying into it.
        self.comm.barrier()
        return self._release()

    def _release(self) -> bool:
        self._closed = True
        clean = True
        for q in self._queues:
            clean = q.close() and clean
        self._queues = []
        self._batches = {}
        if self.win is not None:
            self.win.close()
            self.win = None
        if self.mem is not None:
            self.mem.close()
            self.mem = None
        return clean
