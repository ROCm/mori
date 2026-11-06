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
"""``SdmaAllGatherOp``: an in-place all-gather over cco SDMA, split into ``start`` and
``wait`` so compute queued between them keeps every CU, with the region chosen per call.

Contract: construction is collective; every rank calls ``start`` / ``wait`` in the same
order with the same region, one call in flight; a rank's block is left alone from
``start`` to ``wait`` and the region is read before the next ``start`` in stream order;
one node, and one SDMA user per Communicator (its SDMA queues are comm-scoped). Epochs
advance on the device, so a ``start`` / ``wait`` pair can be captured in a graph.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import torch

from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements
from mori.tensor_utils import from_gpu_ptr

#: Polls of a peer's signal before a wait gives up, about 120 ns each on MI355X:
#: the Int32 maximum is about 4 minutes, the self-test's about 1 second.
WAIT_SPIN_LIMIT = (1 << 31) - 1
SELF_TEST_SPIN_LIMIT = 1 << 23
ALIGN = 16
#: Control area at the head of the window. READY: u32 slot ``p`` is written by rank
#: ``p``. EPOCH: [0] start, [1] wait. FAULT: u32, sticky. EPOCH and FAULT are private.
READY_OFF, EPOCH_OFF, FAULT_OFF, CTRL_BYTES = 0, 256, 512, 1024


class SdmaAllGatherOp:
    """Owns one window ``[ctrl | capacity]`` on ``comm``; the dev-comm is left to it.

    ``agree``, when given, is a collective vote (e.g. a MIN all-reduce over a CPU group)
    called after each setup step with whether it succeeded here; it returns whether it
    succeeded on every rank, so a failure that returns on one rank makes every rank
    raise at the same step. A failure inside a collective step still hangs the peers.

    A wait that gives up is local: it releases my held put, which may land in a late
    peer's region while that peer still reads it. That call's region is invalid on
    every rank, and every rank must stop using the op.
    """

    def __init__(
        self,
        comm,
        capacity: int,
        *,
        agree: Callable[[bool], bool] | None = None,
    ):
        self.comm = comm
        self.rank, self.world_size = comm.rank, comm.nranks
        self.capacity = -(-capacity // ALIGN) * ALIGN
        self.mem = self.win = self.dev_comm = None
        self._inflight = False
        self._agree = agree
        try:
            # One collective a step, so a rank failing after it returns meets the vote.
            for step in (
                self._alloc,
                self._register_window,
                self._create_dev_comm,
                self._load_kernels,
            ):
                self._step(step)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def per_rank_vmm(capacity: int) -> int:
        """A ``per_rank_vmm`` for ``Communicator.init`` that fits this op's window."""
        # Measured on MI355X: two windows' worth plus 64 MiB of cco's own.
        need = 2 * (CTRL_BYTES + capacity) + (64 << 20)
        return -(-need // (2 << 20)) * (2 << 20)

    def _step(self, fn) -> None:
        err = None
        try:
            fn()
        except Exception as e:
            if self._agree is None:
                raise
            err = e
        if self._agree is not None and not self._agree(err is None):
            raise err or RuntimeError("SdmaAllGatherOp setup failed on another rank")

    def _alloc(self) -> None:
        self.mem = self.comm.alloc_mem(CTRL_BYTES + self.capacity)
        from_gpu_ptr(self.mem.ptr, (CTRL_BYTES,), torch.uint8).zero_()
        self.buffer = from_gpu_ptr(
            self.mem.ptr + CTRL_BYTES, (self.capacity,), torch.uint8
        )
        self._fault = from_gpu_ptr(self.mem.ptr + FAULT_OFF, (1,), torch.int32)
        # Lands before the next step's collective, so before any peer's READY store.
        torch.cuda.synchronize()

    def _register_window(self) -> None:
        self.win = self.comm.register_window(self.mem.ptr, self.mem.size)

    def _create_dev_comm(self) -> None:
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 1
        self.dev_comm = self.comm.create_dev_comm(reqs)

    def _load_kernels(self) -> None:
        import flydsl.compiler as flyc
        import flydsl.expr as fx

        from .kernels import build_kernels

        if self.dev_comm.sdma_num_queue < 1:
            raise RuntimeError(
                "SdmaAllGatherOp needs SDMA queues: build mori with BUILD_CCO_SDMA=ON "
                "and set MORI_ENABLE_SDMA=1"
            )
        if self.dev_comm.lsa_size != self.world_size:
            raise RuntimeError(
                f"SdmaAllGatherOp needs one node; lsa_size={self.dev_comm.lsa_size}, "
                f"world_size={self.world_size}"
            )
        start, wait = build_kernels(self.rank, self.world_size)
        stream = fx.Stream(torch.cuda.current_stream())
        # flyc.compile specializations skip the @jit dispatch: ~18 us host per pair.
        self._start = flyc.compile(
            start, self.dev_comm.ptr, self.win.handle, 0, ALIGN, 0, stream
        )
        self._wait = flyc.compile(
            wait, self.dev_comm.ptr, self.win.handle, 0, 0, stream
        )
        # live=0: a launch that touches no peer, so a load failure surfaces here.
        self._start(self.dev_comm.ptr, self.win.handle, 0, ALIGN, 0, stream)
        self._wait(self.dev_comm.ptr, self.win.handle, 0, 0, stream)
        torch.cuda.synchronize()

    def region(self, block_bytes: int, base: int = 0) -> torch.Tensor:
        """The uint8 view of a call's region: rank r's block at ``r * block_bytes``."""
        return self.buffer[base : base + self.world_size * block_bytes]

    def start(self, block_bytes: int, base: int = 0, stream=None) -> None:
        import flydsl.expr as fx

        if self._inflight:
            raise RuntimeError("SdmaAllGatherOp: start while a gather is in flight")
        if (
            block_bytes <= 0
            or block_bytes % ALIGN
            or base % ALIGN
            or base + self.world_size * block_bytes > self.capacity
        ):
            raise ValueError(
                f"region base={base} block_bytes={block_bytes} x {self.world_size}: "
                f"needs {ALIGN}-byte alignment and to fit in {self.capacity} bytes"
            )
        stream = fx.Stream(stream or torch.cuda.current_stream())
        self._start(self.dev_comm.ptr, self.win.handle, base, block_bytes, 1, stream)
        self._inflight = True

    def wait(self, spin_limit: int = WAIT_SPIN_LIMIT, stream=None) -> None:
        """Gives up on a peer after ``spin_limit`` polls and sets :attr:`faulted`."""
        import flydsl.expr as fx

        if not self._inflight:
            raise RuntimeError("SdmaAllGatherOp: wait without start")
        if not 0 <= spin_limit <= WAIT_SPIN_LIMIT:
            raise ValueError(f"spin_limit={spin_limit} not in [0, {WAIT_SPIN_LIMIT}]")
        stream = fx.Stream(stream or torch.cuda.current_stream())
        self._wait(self.dev_comm.ptr, self.win.handle, 1, spin_limit, stream)
        self._inflight = False

    @property
    def faulted(self) -> bool:
        """Whether a wait gave up on a peer; sticky. Syncs the device, so check it off
        the hot path (setup, a drain)."""
        torch.cuda.synchronize()
        return bool(self._fault.item())

    def drain(self) -> None:
        """Waits for a gather started and never waited for (an exception in between),
        so the start and wait epochs stay equal."""
        if self._inflight:
            self.wait()

    def self_test(self, spin_limit: int = SELF_TEST_SPIN_LIMIT) -> None:
        """Collective: gathers each rank's id through the region at 0 with a bounded
        wait; raises unless every block arrived."""
        rows = 4
        region = self.region(rows * ALIGN).view(torch.int32).view(self.world_size, -1)
        region[self.rank].fill_(self.rank + 1)
        self.start(rows * ALIGN)
        self.wait(spin_limit=spin_limit)
        want = torch.arange(1, self.world_size + 1, device=region.device)
        if self.faulted or not bool((region == want[:, None]).all()):
            raise RuntimeError(
                f"SdmaAllGatherOp self-test failed (faulted={self.faulted}): "
                f"{region[:, 0].tolist()}"
            )

    def close(self) -> None:
        """Drains (bounded) and frees the window. Idempotent."""
        if self._inflight:
            with contextlib.suppress(Exception):  # best effort; still free below
                self.wait(spin_limit=SELF_TEST_SPIN_LIMIT)
        self._inflight = False
        if self.mem is not None:
            torch.cuda.synchronize()  # queued puts still address the window
        # The dev-comm is left to the communicator, as in the other ops: its SDMA queues
        # are comm-scoped, and comm.destroy() releases it.
        for handle in (self.win, self.mem):
            if handle is not None:
                handle.close()
        self.win = self.mem = None
