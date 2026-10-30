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
"""CCO Example 12 — in-place all-gather whose start kernel never waits for a peer

The KV-cache-sharding shape: each call names a region of one buffer at run time,
rank ``r`` owns block ``r`` of it, and the gather fills in the rest in place.

    ag.start(base, blk)   # post the pushes and return; no CU waits for peers
    out = gemm(a, b)      # compute queued behind start runs right away
    ag.wait()             # every block of [base, base + world * blk) is here

A push may land only once the peer has read the previous call's region, which
it signals by storing this call's epoch into its READY slot in my window.
Example 10 spins a CU on READY, so one late rank stalls every other rank's
stream. Here lane ``p`` of the start kernel stores the epoch into ``p``'s READY
slot for me, then reads my READY slot for ``p`` once (the recommended use of
``Sdma.post_wait_mem_put_ptr``):

* already this call's epoch: a plain ``put_ptr`` (no engine wait to pay for);
* not yet: ``post_wait_mem_put_ptr``, so the copy engine polls my READY slot for
  ``p``, then copies my block to ``p`` and bumps ``p``'s signal slot.

Either way the start kernel never waits for a peer. ``wait`` drains my queues
and spins until every peer's signal slot has counted this call; it holds a wave
while it spins, so enqueue it after the compute it overlaps. ``--mode cu``
spins on the CU instead, for comparison.

Each run prints, per rank, the start-kernel GPU time and the end-to-end
all-gather time (start of the start kernel to the end of the wait kernel). A
GPU-side barrier lines the ranks up first and the last rank's GPU then idles
``--skew-us``; launch latency is kept out. With a skew, ``fused`` start stays at
a few us while ``cu`` start takes the skew plus a few us on the early ranks, and
the early ranks' all-gather time is the skew plus the gather.

Contract: the constructor is collective; all ranks call ``start``/``wait`` in
the same order with the same region, one call outstanding; write your block
before ``start`` and leave it until ``wait``; read the region before the next
``start`` (stream order); intra-node only. The epochs live in the window and
the kernels advance them, so a ``start``/``wait`` pair can be captured in a CUDA
graph (a replay gathers the captured region); the constructor runs one gather to
compile the kernels, so capture works from the first call.

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 main.py --skew-us 200
"""

import argparse
import os
import statistics
import sys
import time

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
import torch
import torch.distributed as dist
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int64
from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements, Communicator, UniqueId
from mori.cco.device.flydsl.signals import (
    local_store_u32,
    signal_ptr,
    signal_store_u32,
)
from mori.tensor_utils import from_gpu_ptr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cco_example_common import load_u32_system, next_epoch, spin_until  # noqa: E402

WAVE = 64  # one wave: lane p serves peer p
SDMA_QUEUE = 0  # one queue per (me, peer); the READY wait parks only that one
READY_OFF = 0  # u32 slot p is written by rank p
EPOCH_OFF = 256  # [0] start, [1] wait, [2] timing barrier; private to this rank
BAR_OFF = 512  # u32 slot p is written by rank p (timing barrier only)
CTRL_BYTES = 1024
CAPACITY = 8 << 20
MODES = ("fused", "cu")


def make_kernels(rank, world, mode):
    fused = mode == "fused"

    @flyc.kernel(known_block_size=[WAVE, 1, 1])
    def start_kernel(dev_comm: Int64, win: Int64, base: Int64, blk: Int64):
        tid = fx.thread_idx.x
        w = cco.CachedWindow(win)
        sdma = cco.DevComm(dev_comm).sdma()
        epoch, epoch_ptr = next_epoch(w, rank, EPOCH_OFF, 0)
        # Built before the branch: a CachedWindow cannot be captured into an `if`.
        off = fx.Int64(CTRL_BYTES) + base + blk * fx.Int64(rank)
        src = fx.Int64(w.lsa_ptr(rank, off))
        dst = fx.Int64(w.lsa_ptr(tid, off))
        peer_ready = fx.Int64(w.lsa_ptr(tid, READY_OFF)) + fx.Int64(rank * 4)
        my_ready_off = fx.Int64(READY_OFF) + fx.Int64(tid) * fx.Int64(4)
        my_ready = fx.Int64(w.lsa_ptr(rank, 0)) + my_ready_off
        if tid < world:
            if tid != rank:
                # My region is free: peer `tid` may push into it.
                signal_store_u32(signal_ptr(peer_ready), epoch)
                if fused:
                    # Read my READY slot for this peer once: when it is
                    # already this call's epoch, a plain put (an engine wait
                    # that already holds still costs more than this read).
                    # Otherwise the engine waits: EQ on this call's epoch (the
                    # engine compares unsigned); the peer cannot move past it
                    # before its wait for this very put, so a stale read can
                    # only see an older epoch and takes the engine wait.
                    ready = fx.Int32(load_u32_system(signal_ptr(my_ready)))
                    if ready == epoch:
                        sdma.put_ptr(tid, dst, src, blk, SDMA_QUEUE, remote_signal=True)
                    if ready != epoch:
                        sdma.post_wait_mem_put_ptr(
                            tid,
                            SDMA_QUEUE,
                            win,
                            my_ready_off,
                            epoch,
                            dst,
                            src,
                            blk,
                            remote_signal=True,
                        )
                else:
                    spin_until(signal_ptr(my_ready), epoch)
                    sdma.put_ptr(tid, dst, src, blk, SDMA_QUEUE, remote_signal=True)
        fgpu.barrier()
        if tid == 0:
            local_store_u32(epoch_ptr, epoch)

    @flyc.jit
    def start(
        dev_comm: Int64,
        win: Int64,
        base: Int64,
        blk: Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        start_kernel(dev_comm, win, base, blk).launch(
            grid=(1, 1, 1), block=[WAVE, 1, 1], stream=stream
        )

    @flyc.kernel(known_block_size=[WAVE, 1, 1])
    def wait_kernel(dev_comm: Int64, win: Int64):
        tid = fx.thread_idx.x
        w = cco.CachedWindow(win)
        sdma = cco.DevComm(dev_comm).sdma()
        epoch, epoch_ptr = next_epoch(w, rank, EPOCH_OFF, 1)
        slot = signal_ptr(fx.Int64(sdma.signal_slot(tid, SDMA_QUEUE)))
        if tid < world:
            if tid != rank:
                # My push to this peer has read my block; the peer's push into
                # my buffer bumped this slot after its copy landed.
                sdma.quiet_queue(tid, SDMA_QUEUE)
                spin_until(slot, epoch, load=load_u32_system)
        fgpu.barrier()
        if tid == 0:
            local_store_u32(epoch_ptr, epoch)

    @flyc.jit
    def wait(dev_comm: Int64, win: Int64, stream: fx.Stream = fx.Stream(None)):
        wait_kernel(dev_comm, win).launch(
            grid=(1, 1, 1), block=[WAVE, 1, 1], stream=stream
        )

    @flyc.kernel(known_block_size=[WAVE, 1, 1])
    def barrier_kernel(win: Int64):
        """Timing only: every rank's GPU leaves within a store's latency."""
        tid = fx.thread_idx.x
        w = cco.CachedWindow(win)
        epoch, epoch_ptr = next_epoch(w, rank, EPOCH_OFF, 2)
        peer_bar = fx.Int64(w.lsa_ptr(tid, BAR_OFF)) + fx.Int64(rank * 4)
        my_bar = fx.Int64(w.lsa_ptr(rank, BAR_OFF)) + fx.Int64(tid) * fx.Int64(4)
        if tid < world:
            if tid != rank:
                signal_store_u32(signal_ptr(peer_bar), epoch)
                spin_until(signal_ptr(my_bar), epoch)
        fgpu.barrier()
        if tid == 0:
            local_store_u32(epoch_ptr, epoch)

    @flyc.jit
    def gpu_barrier(win: Int64, stream: fx.Stream = fx.Stream(None)):
        barrier_kernel(win).launch(grid=(1, 1, 1), block=[WAVE, 1, 1], stream=stream)

    return start, wait, gpu_barrier


class InPlaceAllGather:
    """In-place all-gather over one cco window ``[ctrl | buffer (capacity)]``.
    Collective: every rank constructs it with the same capacity."""

    def __init__(self, comm, capacity, mode="fused"):
        self.rank, self.world = comm.rank, comm.nranks
        self.capacity = capacity
        self.mem = self.win = self.dev_comm = None
        self.mem = comm.alloc_mem(CTRL_BYTES + capacity)
        self.win = comm.register_window(self.mem.ptr, self.mem.size)
        # READY slots and epochs are read before they are first written.
        from_gpu_ptr(self.mem.ptr, (CTRL_BYTES,), torch.uint8).zero_()

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 1
        self.dev_comm = comm.create_dev_comm(reqs)
        self.buffer = from_gpu_ptr(self.mem.ptr + CTRL_BYTES, (capacity,), torch.uint8)
        self._start, self._wait, self._barrier = make_kernels(
            self.rank, self.world, mode
        )
        # Every rank's zeroing must land before any peer's first start stores
        # into our READY slots: finish it, then line up across ranks.
        torch.cuda.synchronize()
        comm.barrier()
        # One gather to compile the kernels (FlyDSL JITs on first call, which
        # cannot happen inside a CUDA-graph capture). It advances the epochs
        # like any call.
        self.start(0, 16)
        self.wait()
        torch.cuda.synchronize()

    def start(self, base, blk, stream=None):
        """Push my block ``[base + rank * blk, +blk)`` to every peer and return."""
        if blk <= 0 or blk % 16 or base % 16 or base + self.world * blk > self.capacity:
            raise ValueError(f"bad region base={base} blk={blk}")
        stream = stream or torch.cuda.current_stream()
        self._start(
            self.dev_comm.ptr, self.win.handle, base, blk, stream=fx.Stream(stream)
        )

    def wait(self, stream=None):
        """Enqueue the wait: kernels queued after it see every block."""
        stream = stream or torch.cuda.current_stream()
        self._wait(self.dev_comm.ptr, self.win.handle, stream=fx.Stream(stream))

    def gpu_barrier(self, stream=None):
        """Enqueue a cross-rank barrier on the GPU (used to line up timing)."""
        stream = stream or torch.cuda.current_stream()
        self._barrier(self.win.handle, stream=fx.Stream(stream))

    def close(self):
        torch.cuda.synchronize()  # the queues outlive the dev comm: drain first
        for h in (self.dev_comm, self.win, self.mem):
            if h is not None:
                h.close()
        self.dev_comm = self.win = self.mem = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _region(it, world):
    """A run-time region, the same on every rank: (base, blk), 16 B aligned."""
    g = torch.Generator().manual_seed(1234 + it)
    blk = 16 * int(torch.randint(1, CAPACITY // world // 16 + 1, (1,), generator=g))
    slack = (CAPACITY - world * blk) // 16
    return 16 * int(torch.randint(0, slack + 1, (1,), generator=g)), blk


def _pattern(rank, it, blk, device):
    x = torch.arange(blk // 4, device=device, dtype=torch.int32) * 7
    return (x + rank * 1000003 + it * 7919).view(torch.uint8)


def _block(buf, base, blk, r):
    return buf[base + r * blk : base + (r + 1) * blk]


def _gpu_delay():
    """``delay(us)`` spins the GPU so the host queues a timed call before the GPU
    reaches it: the timing holds no launch latency."""
    e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda._sleep(1 << 20)  # warm up
    e0.record()
    torch.cuda._sleep(1 << 20)
    e1.record()
    torch.cuda.synchronize()
    per_us = (1 << 20) / (e0.elapsed_time(e1) * 1e3)
    return lambda us: torch.cuda._sleep(int(us * per_us))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, default="fused")
    ap.add_argument("--iters", type=int, default=20, help="checked calls")
    ap.add_argument("--graph-iters", type=int, default=12, help="checked replays")
    ap.add_argument("--skew-us", type=float, default=0.0, help="last rank enters late")
    ap.add_argument("--blk-kb", type=int, default=64, help="block per rank, timing")
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
    late = world - 1

    def line_up():
        """Line the ranks up on the host; the last one enters late."""
        torch.cuda.synchronize()
        dist.barrier()
        if rank == late and args.skew_us:
            t = time.perf_counter() + args.skew_us * 1e-6
            while time.perf_counter() < t:
                pass

    uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    vmm = 2 * (CTRL_BYTES + CAPACITY) + (64 << 20)

    errors = 0
    with (
        Communicator.init(
            world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=vmm
        ) as comm,
        InPlaceAllGather(comm, CAPACITY, args.mode) as ag,
    ):
        buf = ag.buffer

        def call(it, base, blk):
            _block(buf, base, blk, rank).copy_(_pattern(rank, it, blk, device))
            ag.start(base, blk)
            ag.wait()

        def check(it, base, blk, what):
            nonlocal errors
            torch.cuda.synchronize()
            for r in range(world):
                if not torch.equal(
                    _block(buf, base, blk, r), _pattern(r, it, blk, device)
                ):
                    print(f"[rank {rank}] {what}: block {r} MISMATCH", flush=True)
                    errors += 1

        # Correctness: a fresh region and fresh data every call, one rank late.
        for it in range(args.iters):
            base, blk = _region(it, world)
            line_up()
            call(it, base, blk)
            check(it, base, blk, f"call {it}")

        # CUDA graph: replay one captured start/wait pair, with an eager call
        # every third step; the epochs advance on the device, so both stay in step.
        # The constructor's warm-up compiled the kernels, so capture works even
        # with --iters 0.
        g_base, g_blk = _region(args.iters, world)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            ag.start(g_base, g_blk)
            ag.wait()
        for i in range(args.graph_iters):
            it = args.iters + 1 + i
            line_up()
            if i % 3 == 2:
                base, blk = _region(it, world)
                call(it, base, blk)
                check(it, base, blk, f"eager call {it} between replays")
                continue
            _block(buf, g_base, g_blk, rank).copy_(_pattern(rank, it, g_blk, device))
            graph.replay()
            check(it, g_base, g_blk, f"graph replay {i}")
        # Every call, eager or replayed (and the constructor's warm-up), advanced
        # both epochs, and every peer announced the latest one in my READY slots.
        calls = 1 + args.iters + args.graph_iters
        ctrl = from_gpu_ptr(ag.mem.ptr, (CTRL_BYTES // 4,), torch.int32).tolist()
        seen = ctrl[EPOCH_OFF // 4 : EPOCH_OFF // 4 + 2]
        seen += [ctrl[READY_OFF // 4 + r] for r in range(world) if r != rank]
        if any(v != calls for v in seen):
            print(f"[rank {rank}] epochs {seen}, expected {calls}", flush=True)
            errors += 1

        # Timing, per rank: the start kernel, and the whole all-gather (start of
        # the start kernel to the end of the wait kernel). A GPU barrier lines
        # the ranks up, then the last rank's GPU idles --skew-us.
        blk = (args.blk_kb << 10) // 16 * 16
        if world * blk > CAPACITY:
            raise ValueError(f"--blk-kb: {world} blocks exceed {CAPACITY >> 20} MiB")
        delay = _gpu_delay()
        times = {"start": [], "all-gather": []}
        for i in range(3 + args.timing_iters):
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            torch.cuda.synchronize()
            dist.barrier()
            delay(300.0)  # the host queues all of the call before the GPU gets here
            ag.gpu_barrier()
            if rank == late and args.skew_us:
                delay(args.skew_us)
            ev[0].record()
            ag.start(0, blk)
            ev[1].record()
            ag.wait()
            ev[2].record()
            torch.cuda.synchronize()
            if i >= 3:
                times["start"].append(ev[0].elapsed_time(ev[1]) * 1e3)
                times["all-gather"].append(ev[0].elapsed_time(ev[2]) * 1e3)
        mine = {k: round(statistics.median(v), 1) for k, v in times.items()}
        every = [None] * world
        dist.all_gather_object(every, mine)
        if rank == 0:
            print(
                f"{world} ranks, mode {args.mode}, {blk >> 10} KiB blocks, rank {late}"
                f" late by {args.skew_us:.0f} us; median us per rank:",
                flush=True,
            )
            for k in times:
                print(f"  {k:<10} {[m[k] for m in every]}", flush=True)

    all_err = torch.tensor([errors])
    dist.all_reduce(all_err)
    if rank == 0:
        print("SUCCESS" if all_err.item() == 0 else "FAILED", flush=True)
    dist.destroy_process_group()
    return 0 if all_err.item() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
