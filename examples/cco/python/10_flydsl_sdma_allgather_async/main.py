#!/usr/bin/env python3
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
"""CCO Example 10 — FlyDSL SDMA all-gather with a start / wait split (async)

An all-gather on the copy engines, split in two so compute can run in between --
on the *same* stream, no side stream needed:

    ag.start_async(kv)      # post the pushes and return; the DMA engines take over
    out = gemm(a, b)        # any compute: runs while the bytes move
    ag.wait_async()         # waits for the data; then read ag.output

This is the shape context-parallel attention wants: gather K/V while the Q-side
projections run. The overlap works on one stream because ``start`` ends as soon
as the copies are posted, so kernels queued behind it run concurrently with them.

Each rank owns one window laid out as

    [ ctrl | input (nbytes) | output (W * nbytes) ]

and SDMA-``put``s its ``input`` into slot ``rank`` of every peer's ``output``,
using the put's *remote signal*: once the copy has landed, the copy engine itself
adds 1 to a counter in the peer's SDMA signal pool (``signalBuf[src * nq + q]``).
Two tiny kernels, one wave each:

  * **start** -- ready handshake, then one signalled ``put`` per peer (lane p ->
    peer p). The handshake is what makes back-to-back calls safe: a peer's
    ``output`` is still being read by that peer until *it* reaches its own
    ``start``, so each rank first tells every peer "my output is free" and waits
    to hear the same, and only then pushes. Nothing waits for the copies.
  * **wait** -- ``quiet`` my queues (my pushes have left, so ``input`` is
    reusable), then poll my signal slot for each peer until it reaches this
    epoch. The signal's ATOMIC follows the COPY on the same queue, so seeing it
    means the peer's bytes are in my ``output``.

Counters are monotonic and never reset; the per-rank epoch is read from, and
written back to, the window by the kernels, so nothing is baked into the launch
and the two launches are CUDA-graph friendly.

Contract:
  * ``input`` must not change between ``start_async`` and the end of
    ``wait_async`` (the engines are still reading it).
  * ``output`` is overwritten by the *next* ``start_async``. Finish reading it
    first, in stream order on the stream you call ``start_async`` on.
  * Every rank calls both, the same number of times, in the same order.
  * The whole thing is intra-node (LSA ranks) and needs SDMA enabled.

Single node, ``world_size`` ranks (one GPU each):

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 main.py
"""

import argparse
import os
import statistics
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
import torch
import torch.distributed as dist
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int64
from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements, Communicator, UniqueId

# flydsl moved its buffer/atomic helpers between 0.2 and 0.3; this module is the
# one place that papers over it, so the example runs on either.
from mori.ops.gemm_ar._compat import (
    i32_type,
    local_load_u32,
    local_store_u32,
    signal_load_u32,
    signal_ptr,
    signal_store_u32,
)
from mori.tensor_utils import from_gpu_ptr

WAVE = 64  # one wave: lane p serves peer p
SDMA_QUEUE = 0  # one queue per (src, dst) pair is all a single issuer needs

# Control region. Slot r of READY is written by *rank r* into my window.
READY_OFF = 0
EPOCH_OFF = 256  # [0] = start epoch, [1] = wait epoch; private to this rank
CTRL_BYTES = 512


def _load_u32_system(ptr):
    """System-scope load, as ``waitSignal`` does for the engine-written counter."""
    return llvm.load(
        i32_type(),
        ptr,
        alignment=4,
        ordering=llvm.AtomicOrdering.monotonic,
        syncscope="",
    )


def _spin_until(ptr, flag, load=signal_load_u32):
    """Block until the u32 at ``ptr`` has reached ``flag``.

    The test is a signed difference, so a counter that wraps past 2**32 still
    compares correctly, as long as the two stay less than 2**31 apart. A plain
    unsigned ``<`` would let a wait pass at the wrap: 0xFFFFFFFF < 0 is false.
    """
    first = load(ptr)
    first_v = first.ir_value() if hasattr(first, "ir_value") else first
    i32 = i32_type()
    loop = scf.WhileOp([i32], [first_v])
    cond = ir.Block.create_at_start(loop.before, [i32])
    body = ir.Block.create_at_start(loop.after, [i32])
    with ir.InsertionPoint(cond):
        cur = fx.Int32(cond.arguments[0])
        scf.ConditionOp(
            ((cur - fx.Int32(flag)) < fx.Int32(0)).ir_value(), [cond.arguments[0]]
        )
    with ir.InsertionPoint(body):
        nxt = load(ptr)
        scf.YieldOp([nxt.ir_value() if hasattr(nxt, "ir_value") else nxt])


def _next_epoch(w, rank, slot):
    """``epoch[slot] + 1`` and the pointer to store it back through."""
    ptr = signal_ptr(fx.Int64(w.lsa_ptr(rank, EPOCH_OFF + 4 * slot)))
    return fx.Int32(local_load_u32(ptr)) + fx.Int32(1), ptr


def _handshake(w, rank, tid, epoch):
    """Tell peer ``tid`` my output is free, then wait for it to tell me the same."""
    peer_slot = fx.Int64(w.lsa_ptr(tid, READY_OFF)) + fx.Int64(rank * 4)
    signal_store_u32(signal_ptr(peer_slot), epoch)
    my_slot = fx.Int64(w.lsa_ptr(rank, READY_OFF)) + fx.Int64(tid) * fx.Int64(4)
    _spin_until(signal_ptr(my_slot), epoch)


def make_start_kernel(rank, world, nbytes, in_off, out_off):
    my_slot_in_peer = out_off + rank * nbytes

    @flyc.kernel(known_block_size=[WAVE, 1, 1])
    def start_kernel(dev_comm: Int64, win: Int64):
        tid = fx.thread_idx.x
        w = cco.CachedWindow(win)
        sdma = cco.DevComm(dev_comm).sdma()
        epoch, epoch_ptr = _next_epoch(w, rank, 0)
        # Addresses are plain arithmetic, so build them before the branch: a
        # CachedWindow cannot be captured into a dynamic `if`.
        dst = fx.Int64(w.lsa_ptr(tid, my_slot_in_peer))
        src = fx.Int64(w.lsa_ptr(rank, in_off))
        if tid < world:
            if tid != rank:
                _handshake(w, rank, tid, epoch)
                # remoteSignal: the engine bumps the peer's signalBuf[rank*nq+q]
                # after the copy lands.
                sdma.put_ptr(
                    tid, dst, src, fx.Int64(nbytes), SDMA_QUEUE, remote_signal=True
                )
        fgpu.barrier()
        if tid == 0:
            local_store_u32(epoch_ptr, epoch)

    @flyc.jit
    def run(dev_comm: Int64, win: Int64, stream: fx.Stream = fx.Stream(None)):
        start_kernel(dev_comm, win).launch(
            grid=(1, 1, 1), block=[WAVE, 1, 1], stream=stream
        )

    return run


def make_wait_kernel(rank, world):
    @flyc.kernel(known_block_size=[WAVE, 1, 1])
    def wait_kernel(dev_comm: Int64, win: Int64):
        tid = fx.thread_idx.x
        w = cco.CachedWindow(win)
        sdma = cco.DevComm(dev_comm).sdma()
        epoch, epoch_ptr = _next_epoch(w, rank, 1)
        if tid < world:
            if tid != rank:
                # My pushes to this peer have left: ``input`` is reusable.
                sdma.quiet_queue(tid, SDMA_QUEUE)
                # The peer's push into my output: its put bumped this slot.
                slot = signal_ptr(fx.Int64(sdma.signal_slot(tid, SDMA_QUEUE)))
                _spin_until(slot, epoch, load=_load_u32_system)
        fgpu.barrier()
        if tid == 0:
            local_store_u32(epoch_ptr, epoch)

    @flyc.jit
    def run(dev_comm: Int64, win: Int64, stream: fx.Stream = fx.Stream(None)):
        wait_kernel(dev_comm, win).launch(
            grid=(1, 1, 1), block=[WAVE, 1, 1], stream=stream
        )

    return run


class AllGatherAsync:
    """SDMA all-gather over a cco window, split into ``start_async``/``wait_async``.

    ``shape``/``dtype`` describe one rank's contribution; :attr:`input` is
    ``[*shape]`` and :attr:`output` is ``[world, *shape]``, both views of the
    window (so they alias it -- clone to keep).
    """

    def __init__(self, comm, shape, dtype):
        self.comm = comm
        self.rank, self.world = comm.rank, comm.nranks
        self.shape, self.dtype = tuple(shape), dtype
        numel = 1
        for s in self.shape:
            numel *= s
        self.nbytes = numel * torch.empty((), dtype=dtype).element_size()
        if self.nbytes % 16:
            raise ValueError(
                f"per-rank payload must be a multiple of 16 B, got {self.nbytes}"
            )

        self.in_off = CTRL_BYTES
        self.out_off = self.in_off + self.nbytes
        self.window_bytes = self.out_off + self.world * self.nbytes

        self.mem = self.win = None
        self.mem = comm.alloc_mem(self.window_bytes)
        self.win = comm.register_window(self.mem.ptr, self.mem.size)
        # Flags and epochs are read before they are first written.
        from_gpu_ptr(self.mem.ptr, (self.window_bytes,), torch.uint8).zero_()

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 1
        self.dev_comm = comm.create_dev_comm(reqs)

        self.input = from_gpu_ptr(self.mem.ptr + self.in_off, self.shape, dtype)
        self.output = from_gpu_ptr(
            self.mem.ptr + self.out_off, (self.world, *self.shape), dtype
        )
        self._start = make_start_kernel(
            self.rank, self.world, self.nbytes, self.in_off, self.out_off
        )
        self._wait = make_wait_kernel(self.rank, self.world)

    @staticmethod
    def window_bytes_for(world, nbytes):
        return CTRL_BYTES + (1 + world) * nbytes

    def start_async(self, x=None, stream=None):
        """Post the pushes and return. ``x`` is copied into the window if given;
        otherwise the caller has already written :attr:`input`."""
        stream = stream or torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            if x is not None:
                self.input.copy_(x.reshape(self.shape))
            self._start(self.dev_comm.ptr, self.win.handle, stream=fx.Stream(stream))

    def wait_async(self, stream=None):
        """Enqueue the wait on ``stream``: kernels queued after it see ``output``."""
        stream = stream or torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            # My own slice never crosses a link.
            self.output[self.rank].copy_(self.input)
            self._wait(self.dev_comm.ptr, self.win.handle, stream=fx.Stream(stream))
        return self.output

    def __call__(self, x, stream=None):
        self.start_async(x, stream)
        return self.wait_async(stream)

    def close(self):
        # Dev comm first: destroying it frees this instance's signal pool and peer
        # IPC mappings and leaves the communicator-scoped SDMA queues alone. Then
        # deregister the window, then free its memory.
        for h in (self.dev_comm, self.win, self.mem):
            if h is not None:
                h.close()
        self.dev_comm = self.win = self.mem = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _pattern(rank, it, shape, device):
    """Rank- and iteration-dependent payload, cheap to regenerate for checking."""
    n = 1
    for s in shape:
        n *= s
    base = torch.arange(n, device=device, dtype=torch.float32)
    return ((base % 251) + rank * 1000 + it * 7).reshape(shape).to(torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=16.0, help="MiB contributed per rank")
    ap.add_argument("--iters", type=int, default=5, help="correctness rounds")
    ap.add_argument("--gemms", type=int, default=8, help="4096^3 GEMMs to overlap with")
    ap.add_argument("--timing-iters", type=int, default=30)
    args = ap.parse_args()

    if os.environ.get("MORI_ENABLE_SDMA") not in ("1", "true", "TRUE", "on", "ON"):
        print("Needs SDMA: MORI_ENABLE_SDMA=1 torchrun ... main.py", flush=True)
        return 1

    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dist.init_process_group(backend="cpu:gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local)

    uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)

    cols = 4096
    rows = max(1, int(args.mb * (1 << 20)) // (cols * 2))
    shape = (rows, cols)
    nbytes = rows * cols * 2
    vmm = 2 * AllGatherAsync.window_bytes_for(world, nbytes) + (64 << 20)

    errors = 0
    with (
        Communicator.init(
            world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=vmm
        ) as comm,
        AllGatherAsync(comm, shape, torch.bfloat16) as ag,
    ):
        a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
        b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)

        def gemm():
            for _ in range(args.gemms):
                a @ b

        # ---- correctness: several calls back to back, fresh data every time ----
        for it in range(args.iters):
            ag.start_async(_pattern(rank, it, shape, device))
            gemm()  # overlaps with the transfer
            got = ag.wait_async().clone()  # consume before the next start overwrites
            torch.cuda.synchronize()
            for r in range(world):
                if not torch.equal(got[r], _pattern(r, it, shape, device)):
                    print(f"[rank {rank}] iter {it}: slice {r} MISMATCH", flush=True)
                    errors += 1

        # ---- overlap: the four cases are interleaved so clock drift hits all
        # of them equally; median over iterations, slowest rank reported. ----
        x = _pattern(rank, 0, shape, device)

        def comm_only():
            ag(x)

        def sequential():
            ag(x)
            gemm()

        def overlapped():
            ag.start_async(x)
            gemm()
            ag.wait_async()

        cases = {
            "allgather": comm_only,
            "gemm": gemm,
            "sequential": sequential,
            "overlapped": overlapped,
        }
        samples = {k: [] for k in cases}
        for fn in cases.values():
            for _ in range(3):
                fn()
        torch.cuda.synchronize()
        for _ in range(args.timing_iters):
            for name, fn in cases.items():
                dist.barrier()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                fn()
                e.record()
                torch.cuda.synchronize()
                samples[name].append(s.elapsed_time(e) * 1e3)
        mine = {k: statistics.median(v) for k, v in samples.items()}
        every = [None] * world
        dist.all_gather_object(every, mine)
        if rank == 0:
            t = {k: max(m[k] for m in every) for k in cases}
            print(
                f"{world} ranks, {nbytes / (1 << 20):.1f} MiB per rank, "
                f"{args.gemms} x 4096^3 bf16 GEMM, single stream (us):\n"
                + "  ".join(f"{k} {v:.0f}" for k, v in t.items())
                + f"\nhidden {t['sequential'] - t['overlapped']:.0f} us of "
                f"{t['allgather']:.0f} us all-gather "
                f"(ideal overlapped = max(allgather, gemm) = "
                f"{max(t['allgather'], t['gemm']):.0f} us)",
                flush=True,
            )

    all_err = torch.tensor([errors])
    dist.all_reduce(all_err)
    if rank == 0:
        print("SUCCESS" if all_err.item() == 0 else "FAILED", flush=True)
    dist.destroy_process_group()
    return 0 if all_err.item() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
