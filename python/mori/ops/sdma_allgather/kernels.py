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
"""Start / wait kernels of :class:`SdmaAllGatherOp`: example 12's engine-wait all-gather
with a bounded wait. One block of ``world`` threads rounded up to whole waves; thread
``p`` serves peer ``p`` on SDMA queue ``(p, queue)``.
"""

# No `from __future__ import annotations`: FlyDSL resolves the kernel parameter
# annotations at trace time, and string annotations pass 64-bit pointers truncated.
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm, scf
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int32, Int64

import mori.cco.device.flydsl as cco
from mori.cco.device.flydsl.signals import (
    i32_type,
    local_load_u32,
    local_store_u32,
    signal_ptr,
    signal_store_u32,
)
from mori.jit.config import detect_wave_size

from .op import CTRL_BYTES, EPOCH_OFF, FAULT_OFF, READY_OFF


def _ir(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _load_u32_system(ptr):
    """System scope, as ``waitSignal`` reads the engine-written counter."""
    return llvm.load(
        i32_type(),
        ptr,
        alignment=4,
        ordering=llvm.AtomicOrdering.monotonic,
        syncscope="",
    )


def _spin_until(ptr, flag, limit):
    """Spins until the u32 at ``ptr`` reaches ``flag`` (wrap-safe signed difference) or
    for ``limit`` polls; returns the last value read."""
    i32 = i32_type()
    loop = scf.WhileOp([i32, i32], [_ir(_load_u32_system(ptr)), _ir(fx.Int32(0))])
    cond = ir.Block.create_at_start(loop.before, [i32, i32])
    body = ir.Block.create_at_start(loop.after, [i32, i32])
    with ir.InsertionPoint(cond):
        cur, polls = fx.Int32(cond.arguments[0]), fx.Int32(cond.arguments[1])
        behind = _ir((cur - fx.Int32(flag)) < fx.Int32(0))
        more = arith.AndIOp(behind, _ir(polls < fx.Int32(limit))).result
        scf.ConditionOp(more, [cond.arguments[0], cond.arguments[1]])
    with ir.InsertionPoint(body):
        polls = _ir(fx.Int32(body.arguments[1]) + fx.Int32(1))
        scf.YieldOp([_ir(_load_u32_system(ptr)), polls])
    return fx.Int32(loop.results[0])


def _next_epoch(w, rank, slot):
    ptr = signal_ptr(fx.Int64(w.lsa_ptr(rank, EPOCH_OFF + 4 * slot)))
    return fx.Int32(local_load_u32(ptr)) + fx.Int32(1), ptr


def build_kernels(rank: int, world: int, queue: int):
    """``(start, wait)`` launchers for ``rank``. ``live == 0`` makes either a no-op (a
    load check that touches no peer and no epoch)."""
    wave = detect_wave_size()
    block = -(-world // wave) * wave
    # The FlyDSL disk cache does not hash kernel bodies: everything baked in is named.
    tag = f"r{rank}_w{world}_q{queue}_b{block}"

    @flyc.kernel(name=f"mori_sdma_ag_start_{tag}", known_block_size=[block, 1, 1])
    def start_kernel(dev_comm: Int64, win: Int64, base: Int64, blk: Int64, live: Int32):
        tid = fx.thread_idx.x
        w = cco.CachedWindow(win)
        sdma = cco.DevComm(dev_comm).sdma()
        epoch, epoch_ptr = _next_epoch(w, rank, 0)
        # Built before the branches: a CachedWindow cannot be captured into an `if`.
        off = fx.Int64(CTRL_BYTES) + base + blk * fx.Int64(rank)
        src = fx.Int64(w.lsa_ptr(rank, off))
        dst = fx.Int64(w.lsa_ptr(tid, off))
        peer_ready = fx.Int64(w.lsa_ptr(tid, READY_OFF)) + fx.Int64(rank * 4)
        my_ready_off = fx.Int64(READY_OFF) + fx.Int64(tid) * fx.Int64(4)
        my_ready = fx.Int64(w.lsa_ptr(rank, 0)) + my_ready_off
        fault = fx.Int32(
            local_load_u32(signal_ptr(fx.Int64(w.lsa_ptr(rank, FAULT_OFF))))
        )
        # Once faulted, post nothing: no held put is left to release or quiet.
        post = live * (fx.Int32(1) - fault)
        if post == 1:
            if tid < world:
                if tid != rank:
                    signal_store_u32(signal_ptr(peer_ready), epoch)
                    # The peer cannot pass this epoch before its wait for this put, so a
                    # stale read only sees an older epoch and takes the engine wait.
                    ready = fx.Int32(_load_u32_system(signal_ptr(my_ready)))
                    if ready == epoch:
                        sdma.put_ptr(tid, dst, src, blk, queue, remote_signal=True)
                    if ready != epoch:
                        sdma.post_wait_mem_put_ptr(
                            tid,
                            queue,
                            win,
                            my_ready_off,
                            epoch,
                            dst,
                            src,
                            blk,
                            remote_signal=True,
                        )
        fgpu.barrier()
        if live == 1:
            if tid == 0:
                local_store_u32(epoch_ptr, epoch)

    @flyc.jit
    def start(
        dev_comm: Int64,
        win: Int64,
        base: Int64,
        blk: Int64,
        live: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        start_kernel(dev_comm, win, base, blk, live).launch(
            grid=(1, 1, 1), block=[block, 1, 1], stream=stream
        )

    @flyc.kernel(name=f"mori_sdma_ag_wait_{tag}", known_block_size=[block, 1, 1])
    def wait_kernel(dev_comm: Int64, win: Int64, live: Int32, spin_limit: Int32):
        tid = fx.thread_idx.x
        w = cco.CachedWindow(win)
        sdma = cco.DevComm(dev_comm).sdma()
        epoch, epoch_ptr = _next_epoch(w, rank, 1)
        slot = signal_ptr(fx.Int64(sdma.signal_slot(tid, queue)))
        fault_ptr = signal_ptr(fx.Int64(w.lsa_ptr(rank, FAULT_OFF)))
        my_ready = fx.Int64(w.lsa_ptr(rank, READY_OFF)) + fx.Int64(tid) * fx.Int64(4)
        limit = spin_limit * (fx.Int32(1) - fx.Int32(local_load_u32(fault_ptr)))
        if live == 1:
            if tid < world:
                if tid != rank:
                    last = _spin_until(slot, epoch, limit)
                    if (last - epoch) < fx.Int32(0):
                        local_store_u32(fault_ptr, fx.Int32(1))
                        # Release my engine-held put, else it blocks my queue.
                        signal_store_u32(signal_ptr(my_ready), epoch)
                    sdma.quiet_queue(tid, queue)
        fgpu.barrier()
        if live == 1:
            if tid == 0:
                local_store_u32(epoch_ptr, epoch)

    @flyc.jit
    def wait(
        dev_comm: Int64,
        win: Int64,
        live: Int32,
        spin_limit: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        wait_kernel(dev_comm, win, live, spin_limit).launch(
            grid=(1, 1, 1), block=[block, 1, 1], stream=stream
        )

    return start, wait
