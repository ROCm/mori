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
"""Device side of the triggered GEMM + all-reduce: the reduce and the final wait.

    reduce  ``R`` work-groups walk my bands in order: wait for my tiles and every peer's
            copy, sum slice ``block_idx`` of the ``W`` partials in fp32 (peer order
            ``(rank + j) % W``), store bf16, add 1 to the band's all-gather gate.
    wait    one work-group: wait until every peer's reduced bands and my all-gather's
            reads are done, then advance the call counts.

Targets are device-resident: ``calls[b]`` counts completed calls covering band ``b``,
``sent_calls`` counts calls; the host keeps the same totals for the engine's POLLs.
Every wait is bounded; one that gives up sets the error word, after which the
remaining waits give up at once and nothing more is reduced or released.

Ordering: arrivals land after their data on the same queue; waits poll with relaxed
system-scope loads and issue one acquire fence once they pass; each wave drains its
stores (``s_waitcnt vmcnt(0)``) before the barrier that precedes a release add.
"""

from __future__ import annotations

import hashlib

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl._mlir.dialects import scf
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int64

from ._compat import (
    CM_CACHED,
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
    i32_type,
    local_load_u32,
    local_store_u32,
    signal_load_u32,
    signal_ptr,
    signal_store_u32,
    wave_uniform_i64,
)
from ._gemm_a8w8_8wave import wait_barrier

__all__ = ["build_trigger_kernels", "TRIGGER_THREADS"]

#: Threads per work-group of both kernels.
TRIGGER_THREADS = 256
#: Bytes one lane moves per load: a 16-byte pack of 8 bf16.
PACK_BYTES = 16


def _i64():
    return ir.IntegerType.get_signless(64)


def _raw(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _load_u64(addr):
    """Relaxed system-scope u64 load; callers fence once their wait has passed."""
    return _llvm_d.load(
        _i64(),
        signal_ptr(addr),
        alignment=8,
        ordering=_llvm_d.AtomicOrdering.monotonic,
        syncscope="",
    )


def _wait_ge_u64(addr, target, limit):
    """Polls until ``*addr >= target`` (unsigned) or ``limit`` polls; returns success."""
    i64 = _i64()
    first = _load_u64(addr)
    loop = scf.WhileOp([i64, i64], [first, _raw(fx.Int64(0))])
    cond = ir.Block.create_at_start(loop.before, [i64, i64])
    body = ir.Block.create_at_start(loop.after, [i64, i64])
    with ir.InsertionPoint(cond):
        cur = fx.Uint64(fx.Int64(cond.arguments[0]))
        polls = fx.Int64(cond.arguments[1])
        keep = (cur < fx.Uint64(target)) & (polls < limit)
        scf.ConditionOp(_raw(keep), [cond.arguments[0], cond.arguments[1]])
    with ir.InsertionPoint(body):
        fx.rocdl.s_sleep(1)
        nxt = _load_u64(addr)
        scf.YieldOp([nxt, _raw(fx.Int64(body.arguments[1]) + fx.Int64(1))])
    return fx.Uint64(fx.Int64(loop.results[0])) >= fx.Uint64(target)


def _release_add_u64(addr, value):
    """System-scope release ``fetch_add`` on the u64 at a raw i64 address."""
    _llvm_d.atomicrmw(
        _llvm_d.AtomicBinOp.add,
        signal_ptr(addr),
        _raw(fx.Int64(value)),
        _llvm_d.AtomicOrdering.release,
        syncscope="",
        alignment=8,
    )


def _acquire_fence():
    _llvm_d.fence(_llvm_d.AtomicOrdering.acquire, syncscope="")


def build_trigger_kernels(layout, rank: int, m: int, *, reducer_blocks: int):
    """Compile the reduce and wait launchers for one ``m``.

    ``layout`` is the op's window layout. Returns ``(reduce, wait)``, each a
    launcher taking ``(base, polls, stream=...)``: this rank's window address and
    the number of polls after which a device wait gives up.
    """
    ws = layout.world_size
    bands = layout.bands(m)
    if bands % ws:
        raise ValueError(
            f"{bands} bands of {layout.mt1} rows do not split over {ws} ranks"
        )
    my_bands = bands // ws
    band_bytes = layout.band_bytes
    if band_bytes % PACK_BYTES:
        raise ValueError(f"a band of {band_bytes} bytes is not whole 16-byte packs")
    packs = band_bytes // PACK_BYTES
    R = reducer_blocks
    threads = TRIGGER_THREADS
    pubs = layout.publishers
    kmax = layout.max_bands_per_rank
    off = layout
    spec = hashlib.sha256(
        repr((off.offsets(), ws, rank, m, R, pubs, band_bytes)).encode()
    ).hexdigest()[:16]

    @flyc.kernel(name=f"mori_trigger_reduce_{spec}", known_block_size=[threads, 1, 1])
    def trigger_reduce(base: Int64, polls: Int64):
        tid = fx.thread_idx.x
        part = fx.block_idx.x
        limit = fx.Int64(polls)
        win = fx.Int64(base)
        err = signal_ptr(win + fx.Int64(off.err_off))

        for k in range_constexpr(my_bands):
            band = k * ws + rank
            if tid == 0:
                if fx.Int32(signal_load_u32(err)) == 0:
                    c = fx.Int64(
                        fx.Int32(
                            local_load_u32(
                                signal_ptr(
                                    win
                                    + fx.Int64(off.calls_off)
                                    + fx.Int64(band) * fx.Int64(4)
                                )
                            )
                        )
                    ) + fx.Int64(1)
                    ok = _wait_ge_u64(
                        win
                        + fx.Int64(off.rs_gate_off)
                        + fx.Int64(band) * fx.Int64(off.gate_stride),
                        c * fx.Int64(pubs),
                        limit,
                    )
                    for s in range_constexpr(ws):
                        if const_expr(s != rank):
                            # After a failed wait the rest give up at once.
                            ok = ok & _wait_ge_u64(
                                win
                                + fx.Int64(off.rs_arrive_off)
                                + fx.Int64(s * kmax + k) * fx.Int64(off.gate_stride),
                                c,
                                ok.select(limit, fx.Int64(0)),
                            )
                    if ok.select(fx.Int32(0), fx.Int32(1)) != 0:
                        signal_store_u32(err, fx.Int32(1))
            fgpu.barrier()
            _acquire_fence()

            band_off = fx.Int64(band * band_bytes)
            mine = fx.Int64(k * band_bytes)
            srcs = [
                create_buffer_resource_from_addr(
                    wave_uniform_i64(win + fx.Int64(off.input_off) + band_off)
                )
            ] + [
                create_buffer_resource_from_addr(
                    wave_uniform_i64(
                        win + fx.Int64(off.recv_slot_off((rank + j) % ws)) + mine
                    )
                )
                for j in range(1, ws)
            ]
            out = create_buffer_resource_from_addr(
                wave_uniform_i64(win + fx.Int64(off.output_off) + band_off)
            )
            # A faulted call leaves the output alone.
            if fx.Int32(signal_load_u32(err)) == 0:
                for pk in range(part * threads + tid, packs, R * threads):
                    i32_off = pk * 4
                    acc = None
                    for j in range_constexpr(ws):
                        raw = fx.Vector(
                            buffer_load(
                                srcs[j],
                                i32_off,
                                vec_width=4,
                                dtype=i32_type(),
                                cache_modifier=CM_CACHED,
                            )
                        )
                        v = raw.bitcast(fx.BFloat16).to(fx.Float32)
                        acc = v if acc is None else acc + v
                    buffer_store(
                        acc.to(fx.BFloat16).bitcast(fx.Int32),
                        out,
                        i32_off,
                        cache_modifier=CM_CACHED,
                    )

            wait_barrier(0)
            if tid == 0:
                if fx.Int32(signal_load_u32(err)) == 0:
                    _release_add_u64(
                        win + fx.Int64(off.ag_gate_off + k * off.gate_stride),
                        1,
                    )

    @flyc.kernel(name=f"mori_trigger_wait_{spec}", known_block_size=[threads, 1, 1])
    def trigger_wait(base: Int64, polls: Int64):
        tid = fx.thread_idx.x
        limit = fx.Int64(polls)
        win = fx.Int64(base)
        calls = win + fx.Int64(off.calls_off)
        err = signal_ptr(win + fx.Int64(off.err_off))
        for idx in range(tid, ws * my_bands, threads):
            s = idx // my_bands
            k = idx % my_bands
            if s != rank:
                band = k * ws + s
                c = fx.Int64(
                    fx.Int32(
                        local_load_u32(signal_ptr(calls + fx.Int64(band) * fx.Int64(4)))
                    )
                ) + fx.Int64(1)
                ok = _wait_ge_u64(
                    win
                    + fx.Int64(off.ag_arrive_off)
                    + (fx.Int64(s) * fx.Int64(kmax) + fx.Int64(k))
                    * fx.Int64(off.gate_stride),
                    c,
                    limit,
                )
                if ok.select(fx.Int32(0), fx.Int32(1)) != 0:
                    signal_store_u32(err, fx.Int32(1))
        # My own all-gather queues: each bumps sent[d] after its last read of my
        # output, so the caller may overwrite the output once this kernel is done.
        sent_calls = signal_ptr(win + fx.Int64(off.sent_calls_off))
        if tid < ws:
            if tid != rank:
                c = fx.Int64(fx.Int32(local_load_u32(sent_calls))) + fx.Int64(1)
                ok = _wait_ge_u64(
                    win
                    + fx.Int64(off.sent_off)
                    + fx.Int64(tid) * fx.Int64(off.gate_stride),
                    c,
                    limit,
                )
                if ok.select(fx.Int32(0), fx.Int32(1)) != 0:
                    signal_store_u32(err, fx.Int32(1))
        fgpu.barrier()
        _acquire_fence()
        for b in range(tid, bands, threads):
            p = signal_ptr(calls + fx.Int64(b) * fx.Int64(4))
            local_store_u32(p, fx.Int32(local_load_u32(p)) + fx.Int32(1))
        if tid == 0:
            local_store_u32(
                sent_calls, fx.Int32(local_load_u32(sent_calls)) + fx.Int32(1)
            )

    def _launcher(kern, grid):
        name = f"{spec}:{grid}"

        @flyc.jit
        def go(base: Int64, polls: Int64, stream=fx.Stream(None)):
            # The kernel object is not a scalar closure capture, so it would not
            # reach FlyDSL's cache key; referencing `name` keeps this kernel's full
            # specialization in the persisted JIT identity.
            assert const_expr(len(name) > 0)
            kern(base, polls).launch(
                grid=(grid, 1, 1), block=[threads, 1, 1], stream=stream
            )

        return go

    return _launcher(trigger_reduce, R), _launcher(trigger_wait, 1)
