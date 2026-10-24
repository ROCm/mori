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
"""Device timestamps for the fused-sdma path, for attributing its tail.

Debug instrumentation, off unless a kernel is compiled with a stamp offset.
Every stamp is ``s_memrealtime`` (100 MHz) written by one thread into a u64
slot of this rank's own window. The clock is per GPU: compare stamps within a
rank, never across ranks.

Slot map, in u64 slots from the stamp base::

    [0, G)                    block start, per GEMM block
    [G, 2G)                   block end: stores retired and released
    [2G, 2G + 3*W*C)          per (dest, chunk): elected, lock held, put returned
    [D, D + 2 + 2*W)          drain: start, end, quiet done[W], peer flag seen[W]

with ``G`` the GEMM grid, ``W`` the world size, ``C`` the chunk count and
``D = 2G + 3*W*C``.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d

from ..gemm_ar._compat import signal_ptr

#: s_memrealtime ticks per microsecond.
TICKS_PER_US = 100.0


def chunk_slot(grid: int) -> int:
    return 2 * grid


def drain_slot(grid: int, world: int, chunks: int) -> int:
    return 2 * grid + 3 * world * chunks


def stamp_slots(grid: int, world: int, chunks: int) -> int:
    return drain_slot(grid, world, chunks) + 2 + 2 * world


def realtime():
    """``s_memrealtime``: a 64-bit 100 MHz clock, constant across DVFS."""
    i64 = ir.IntegerType.get_signless(64)
    return fx.Int64(
        _llvm_d.call_intrinsic(i64, "llvm.amdgcn.s.memrealtime", [], [], [])
    )


def stamp(base_i64, slot, value=None):
    """Store ``value`` (default: now) at u64 slot ``slot`` of the stamp region."""
    addr = fx.Int64(base_i64) + fx.Int64(slot) * fx.Int64(8)
    v = realtime() if value is None else fx.Int64(value)
    _llvm_d.store(v.ir_value(), signal_ptr(addr), alignment=8, volatile_=True)


def decode(raw, *, grid: int, world: int, chunks: int, rank: int, chunk_bytes):
    """One rank's raw u64 slots -> microseconds relative to its first block.

    ``chunk_bytes[c]`` is the payload of chunk ``c`` for one destination, so
    the report can set each put's drain time against what the bytes alone
    would take.
    """
    raw = [int(x) for x in raw]
    starts = [t for t in raw[:grid] if t]
    ends = [t for t in raw[grid : 2 * grid] if t]
    t0 = min(starts)

    def us(t):
        return (t - t0) / TICKS_PER_US if t else None

    cs = chunk_slot(grid)
    puts = []
    for d in range(world):
        for c in range(chunks):
            e, lk, p = raw[cs + 3 * (d * chunks + c) : cs + 3 * (d * chunks + c) + 3]
            if d == rank or not e:
                continue
            puts.append(
                {
                    "dest": d,
                    "chunk": c,
                    "bytes": chunk_bytes[c],
                    "elect": us(e),
                    "lock": us(lk),
                    "posted": us(p),
                }
            )
    ds = drain_slot(grid, world, chunks)
    drain = {
        "start": us(raw[ds]),
        "end": us(raw[ds + 1]),
        "quiet": {p: us(raw[ds + 2 + p]) for p in range(world) if p != rank},
        "flag": {p: us(raw[ds + 2 + world + p]) for p in range(world) if p != rank},
    }
    return {
        "rank": rank,
        "block_start": sorted(us(t) for t in starts),
        "block_end": sorted(us(t) for t in ends),
        "gemm_end": us(max(ends)),
        "puts": puts,
        "drain": drain,
    }


__all__ = [
    "TICKS_PER_US",
    "chunk_slot",
    "decode",
    "drain_slot",
    "realtime",
    "stamp",
    "stamp_slots",
]
