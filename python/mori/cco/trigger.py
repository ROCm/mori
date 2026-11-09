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
"""Host-armed SDMA trigger queues (``ccoTrigger*`` in ``cco.hpp``).

The host appends chains ``[POLL gate >= target] [COPY]... [ATOMIC arrival += 1]`` to a
private SDMA queue and kicks it before the producing kernel runs; the engine waits at
each POLL until a kernel bumps the gate. One host thread per queue; kick only a queue
that is drained or parked on a closed gate. Gates are monotonic, so targets are
cumulative.
"""

from __future__ import annotations

import mori.cco.cco as _cco
from mori.cco.device.ops import SdmaCmp

from .communicator import CCOResource, Communicator

__all__ = [
    "TriggerBatch",
    "TriggerQueue",
    "peer_ptr",
]


def peer_ptr(comm: Communicator, local_ptr: int, peer: int) -> int:
    """Peer ``peer``'s address of a symmetric pointer on ``comm``."""
    return _cco.peer_ptr(comm._raw, int(local_ptr), int(peer))


class TriggerQueue(CCOResource):
    """Private SDMA queue to ``peer`` (a world rank in this LSA team).

    ``engine_offset`` (gfx942 only) moves it off the link's engine; ``ring_bytes`` 0
    selects 256 KiB.
    """

    close_first = True

    def __init__(
        self,
        comm: Communicator,
        peer: int,
        *,
        engine_offset: int = 0,
        ring_bytes: int = 0,
    ) -> None:
        super().__init__(comm)
        self._clean = True
        self._q = _cco._TriggerQueue(
            comm._raw, int(peer), int(engine_offset), int(ring_bytes)
        )
        comm._resources.append(self)

    def _deallocate(self) -> None:
        if self._q is not None:
            self._clean = self._q.close()
            self._q = None

    def close(self) -> bool:
        """Destroys the queue; returns False if kicked work was not yet consumed."""
        super().close()
        return self._clean

    def append(self, gate: int, target: int, copies, arrival: int, cmp=SdmaCmp.GE):
        """Appends one chain of ``(src, dst, nbytes)`` copies without kicking.

        Returns the ring index past the chain, or None if the ring is full.
        """
        self._check_valid()
        if not 0 <= int(target) < 1 << 32:
            raise ValueError(
                f"target {target} does not fit the engine's 32-bit compare"
            )
        return self._q.append(int(gate), int(target), copies, int(arrival), cmp)

    def kick(self) -> None:
        self._check_valid()
        self._q.kick()

    def quiesce(self, index: int | None = None, timeout_s: float = 30.0) -> bool:
        """Waits until the engine has consumed up to ``index`` (default: all kicked).

        Returns False on timeout.
        """
        self._check_valid()
        return self._q.quiesce(index, int(timeout_s * 1e9))

    def state(self) -> tuple[int, int]:
        """``(read_index, write_index)`` in monotonic ring bytes."""
        self._check_valid()
        return self._q.state()

    def read_ring(self, index: int, nbytes: int) -> bytes:
        self._check_valid()
        return self._q.read_ring(index, nbytes)

    @property
    def peer(self) -> int:
        self._check_valid()
        return self._q.peer

    @property
    def engine_id(self) -> int:
        self._check_valid()
        return self._q.engine_id

    @property
    def ring_bytes(self) -> int:
        self._check_valid()
        return self._q.ring_bytes

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<TriggerQueue: closed>"
        r, w = self.state()
        return f"<TriggerQueue: peer={self.peer}, engine={self.engine_id}, read={r}, write={w}>"


class TriggerBatch:
    """Chains over several :class:`TriggerQueue` objects, built once and armed in one call.

    Each :meth:`arm` drains every queue, appends every chain with its target from
    ``targets[slot]`` and kicks all queues behind one HDP flush. At most two copies per
    chain, ``>=`` compares only. The queues must outlive the batch.
    """

    OK = _cco.TRIGGER_BATCH_OK
    NO_ROOM = _cco.TRIGGER_BATCH_NO_ROOM
    DRAIN_TIMEOUT = _cco.TRIGGER_BATCH_DRAIN_TIMEOUT

    def __init__(self) -> None:
        self._b = _cco._TriggerBatch()
        self._queues: list[TriggerQueue] = []

    def add(
        self, queue: TriggerQueue, gate: int, slot: int, copies, arrival: int
    ) -> None:
        queue._check_valid()
        self._b.add(queue._q, int(gate), int(slot), copies, int(arrival))
        if all(q is not queue for q in self._queues):
            self._queues.append(queue)

    def __len__(self) -> int:
        return len(self._b)

    def arm(self, targets, timeout_s: float = 30.0) -> int:
        """Returns :attr:`OK`, or :attr:`NO_ROOM` / :attr:`DRAIN_TIMEOUT` with nothing appended."""
        for q in self._queues:
            q._check_valid()
        return self._b.arm(targets, int(timeout_s * 1e9))
