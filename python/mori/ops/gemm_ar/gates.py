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
"""Gates for triggered copy-engine ops, and the view a GEMM epilogue releases them through.

A gate is a u64 counter that producer work-groups bump with a system-scope release add
once their tile is stored; a copy-engine chain armed on the host waits for it (low 32
bits, ``>=``). Gates are never reset, so targets are cumulative: each call adds its
expected increments. :func:`gate_view` lays the gates out in the gate-epilogue kernel
ABI: band ``b``'s work-groups add 1 at ``heap_base + record[GateTileBase + b].ready_offset``.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "GATE_ABI_VERSION",
    "GATE_STRIDE",
    "GateProducerSpec",
    "GateTable",
    "gate_view",
]

#: Version of the gate-epilogue kernel ABI that :func:`gate_view` writes.
GATE_ABI_VERSION = 2
#: Header and record sizes of that ABI, in bytes.
GATE_VIEW_HEADER_BYTES = 48
GATE_VIEW_RECORD_BYTES = 48
#: Record ``role_flags`` bit: this rank's producer releases the band.
GATE_ROLE_PUBLISH = 0x1
#: Bytes between consecutive gates: one per 64-byte line.
GATE_STRIDE = 64

# abi, size, record size, count, rank, reserved, epoch, heap base, records
_HEADER = struct.Struct("<IIIIIIQQQ")
# id, channel, role, publishers, ready, arrival, completion, error
_RECORD = struct.Struct("<IIIIQQQQ")
assert _HEADER.size == GATE_VIEW_HEADER_BYTES and _RECORD.size == GATE_VIEW_RECORD_BYTES


@dataclass(frozen=True)
class GateProducerSpec:
    """Tile shape of a producer that releases gates from its epilogue.

    The output is ``[M, N]`` row-major (tokens by features), i.e. the column-major
    ``D`` of a GEMM whose contiguous dimension is the features. A **band** is
    ``mt1`` consecutive rows; every one of its ``ceil(N / mt0)`` work-groups adds 1
    to the band's gate once per call.
    """

    mt0: int  #: tile extent along the features (N)
    mt1: int  #: tile extent along the tokens (M): the band height

    def __post_init__(self):
        if self.mt0 < 1 or self.mt1 < 1:
            raise ValueError(f"tile extents must be positive, got {self}")

    def publishers(self, n: int) -> int:
        """Gate increments one band receives per call."""
        return math.ceil(n / self.mt0)


class GateTable:
    """``count`` monotonic u64 gates at ``base + i * stride`` and their cumulative
    targets in ``expected`` (a uint64 array, optionally a view of a larger one)."""

    def __init__(
        self,
        base: int,
        count: int,
        stride: int = GATE_STRIDE,
        expected: np.ndarray | None = None,
    ):
        if base % 8 or stride % 8:
            raise ValueError("gates must be 8-byte aligned")
        self.base, self.count, self.stride = base, count, stride
        if expected is None:
            expected = np.zeros(count, dtype=np.uint64)
        if expected.dtype != np.uint64 or expected.shape != (count,):
            raise ValueError(f"expected must be a uint64 array of {count}")
        self.expected = expected

    def addr(self, i: int) -> int:
        return self.base + i * self.stride

    def expect(self, i: int, increments: int) -> int:
        """Account for ``increments`` more adds on gate ``i``; returns its new target."""
        self.expected[i] += np.uint64(increments)
        return int(self.expected[i])

    def expect_first(self, n: int, increments: int) -> None:
        """Account for ``increments`` more adds on each of gates ``0 .. n-1``."""
        self.expected[:n] += np.uint64(increments)

    def high(self) -> int:
        """Largest target, to decide when the 32-bit engine compare needs a re-base."""
        return int(self.expected.max()) if self.count else 0

    def reset(self) -> None:
        """Forget all targets; only valid once the gates themselves are zeroed."""
        self.expected[:] = 0


def gate_view(
    heap_base: int,
    gate_addrs: list[int],
    *,
    publishers: int,
    rank: int,
    device: torch.device | int | None = None,
) -> torch.Tensor:
    """Builds the view in device memory: record ``i`` releases ``gate_addrs[i]``.

    Pass ``GateTileBase = 0``. Only the ready offsets are used; the other offsets are 0.
    The tensor's ``data_ptr()`` is the ``GateView`` argument; it must outlive every launch.
    """
    count = len(gate_addrs)
    view = torch.empty(
        GATE_VIEW_HEADER_BYTES + count * GATE_VIEW_RECORD_BYTES,
        dtype=torch.uint8,
        device=device if device is not None else torch.cuda.current_device(),
    )
    records = view.data_ptr() + GATE_VIEW_HEADER_BYTES
    host = bytearray(view.numel())
    _HEADER.pack_into(
        host,
        0,
        GATE_ABI_VERSION,
        GATE_VIEW_HEADER_BYTES,
        GATE_VIEW_RECORD_BYTES,
        count,
        rank,
        0,  # reserved
        0,  # epoch: unused by the epilogue
        heap_base,
        records,
    )
    for i, addr in enumerate(gate_addrs):
        if addr < heap_base:
            raise ValueError(
                f"gate {i} at {addr:#x} lies below heap_base {heap_base:#x}"
            )
        _RECORD.pack_into(
            host,
            GATE_VIEW_HEADER_BYTES + i * GATE_VIEW_RECORD_BYTES,
            i,  # tile id: must equal the record index
            0,  # channel
            GATE_ROLE_PUBLISH,
            publishers,
            addr - heap_base,
            0,
            0,
            0,
        )
    view.copy_(torch.frombuffer(host, dtype=torch.uint8))
    return view
