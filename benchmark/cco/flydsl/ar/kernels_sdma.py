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
"""SDMA all-reduce in FlyDSL: reduce-scatter push, local reduce, all-gather push.

Same algorithm and the same window as ``kernels_lsa``'s 2-stage, and the same
``2(P-1)/P`` bytes on the wire. The only difference is who moves them: here the
copy engines do, so the CUs are free for compute during the transfers. That is
the property the GEMM fusion in step 3 is after -- raw bandwidth is not expected
to beat LSA.

Three kernels rather than one, because both transfers need a *device-wide*
ordering point (every block must see the landed data), and a kernel boundary is
the only one that is free and graph-capturable:

    1. scatter  lane p pushes my rows of peer p's slice into p's recv slot,
                drains queue p, then bumps p's `start[0][rank]` flag
    2. reduce   fp32-sum my own slice plus the P-1 landed slices into
                `output + rank*slice`
    3. gather   lane p pushes my reduced slice into peer p's output at the same
                offset, drains, then bumps p's `end[0][rank]` flag

Rank-to-rank arrival is signalled over LSA, not SDMA, because FlyDSL's ``Sdma``
exposes no ``wait_signal`` and its ``put`` wrapper hard-codes ``remoteSignal =
false`` (``src/cco/device/cco_device_wrapper.cpp:140``), so a peer cannot observe
an SDMA signal at all. This is the ``opus_sdma_a2a_quiet_notify_kernel`` pattern
from gcnasm: no per-PUT signal, ``quiet_queue`` on the sender, then an LSA atomic
into the peer's flag. The flags, their monotonic epoch and the spin loop are all
reused verbatim from ``kernels_lsa`` -- only row 0 of the signal array is used,
since the pushes are issued by one warp.

Transfer sizing follows the copy engine's one real lever: bytes per op. Each peer
gets exactly one ``put`` of ``slice_bytes``, never split across queues -- one peer
is one xGMI link, so splitting only pays the ~2us per-packet cost again. At
[4096, 7168] bf16 a slice is 7.3MB (~59 GB/s territory); at [64, 7168] it is
114KB, below the ~256KB knee, so decode sizes are expected to lose to LSA.
cco's own chunking is not a factor: ``CCO_SDMA_MAX_COPY_BYTES`` is 1GB, so every
slice we push is a single packet.

Measured on 8x MI355X, [M, 7168] bf16, against the LSA kernel (full table in
``bench_ar.py``): ~1.20x slower across M >= 128, of which ~12us is a fixed floor
(two transfer phases x the engine's ~6us dispatch cost) and the rest is ~18%
lower sustained bandwidth than CU-issued loads over the same links. Both are
expected. This backend is not here to be faster -- it is here because the
transfers occupy one warp of one block instead of the whole grid, which is what
makes an in-GEMM epilogue possible in step 3.
"""

from __future__ import annotations

import os
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu as fgpu
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int64

import mori.cco.device.flydsl as cco
from mori.cco.device.flydsl import _bindings as raw_cco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _compat import (  # noqa: E402
    CM_CACHED,
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
    i32_type,
    local_load_u32,
    local_store_u32,
    signal_ptr,
    signal_store_u32,
    wave_uniform_i64,
)
from kernels_lsa import _spin_until  # noqa: E402
from layout import MAX_WORLD  # noqa: E402

#: The push warp. One lane per peer, each on its own queue, so the lanes post in
#: parallel (cco's ccoSdmaThreadIndependent shape). 64 is the wave size; only the
#: first `world_size` lanes do anything.
PUSH_THREADS = 64


def build_sdma_ar(cfg, rank: int, *, queues: int = 8, signal: bool = False):
    """Compile an SDMA all-reduce launcher for ``cfg`` on ``rank``.

    Returns ``(run, stage)``; ``stage`` is always 2 -- there is no 1-stage SDMA
    variant, since a copy engine gains nothing from broadcasting the whole
    payload to every peer.

    ``queues`` must match ``reqs.sdma_queue_count``; one queue per peer keeps
    concurrently-issuing warps per queue at 1, which is cco's stated rule.

    ``signal`` selects ``put``'s trailing local ATOMIC. It is off by default:
    ``quiet``/``quietQueue`` drain the queue's read pointer and are documented as
    "independent of the signals" (``include/mori/cco/cco.hpp:1707``), and nothing
    here polls a signal, so the packet would be pure overhead. Exposed because
    FlyDSL's own ``Sdma._xfer`` docstring claims the opposite -- that a no-signal
    put "cannot be drained by quiet" -- and that claim is worth being able to
    disprove by measurement rather than by reading.
    """
    cfg.validate()
    ws = cfg.world_size
    if cfg.recv_slots < ws:
        raise ValueError(
            f"SDMA needs a landing slot per peer: build the ArConfig with "
            f"recv_slots={ws} (got {cfg.recv_slots})"
        )
    if queues < ws:
        raise ValueError(
            f"need one SDMA queue per peer: queues={queues} < world_size={ws}"
        )

    threads, blocks = cfg.threads, cfg.blocks
    elem_dtype = fx.BFloat16 if cfg.elem_bytes == 2 else fx.Float32
    stride_packs = blocks * threads
    part = cfg.packs_per_rank
    slice_bytes = cfg.slice_bytes

    start_off, end_off, flag_off = cfg.start_off, cfg.end_off, cfg.flag_off
    in_off, out_off = cfg.input_off, cfg.output_off
    # Constant-folded because `rank` is a Python int: the slice I own, and the
    # slot I occupy in every peer's recv region.
    my_slice_off = rank * slice_bytes
    my_recv_slot = cfg.recv_slot_off(rank)

    I32_PER_PACK = 4  # a 16B pack always moves as 4 x i32, whatever the payload

    def _next_flag(w, bid):
        """``_flag[block] + 1`` -- monotonic, never reset (graph-replay safe)."""
        base = fx.Int64(w.lsa_ptr(rank, flag_off)) + fx.Int64(bid) * fx.Int64(4)
        rsrc = signal_ptr(base)
        return fx.Int32(local_load_u32(rsrc)) + fx.Int32(1), rsrc

    def _signal_and_wait(w, arr_off, flag, tid):
        """Row-0 barrier: publish to peer ``tid``, then wait on ``tid``'s slot."""
        peer_arr = fx.Int64(w.lsa_ptr(tid, arr_off))
        signal_store_u32(signal_ptr(peer_arr + fx.Int64(rank * 4)), flag)
        self_arr = fx.Int64(w.lsa_ptr(rank, arr_off))
        _spin_until(signal_ptr(self_arr + fx.Int64(tid) * fx.Int64(4)), flag)

    def _push_kernel(arr_off, dst_off_of_peer, src_off_expr):
        """A push phase: one put per peer, drain, then the cross-rank barrier.

        ``dst_off_of_peer`` is a constant byte offset *in the destination's*
        window; ``src_off_expr(tid)`` builds the offset in mine, which may depend
        on which peer the lane is serving.
        """

        @flyc.kernel(known_block_size=[PUSH_THREADS, 1, 1])
        def push(dev_comm: Int64, win: Int64):
            tid = fx.thread_idx.x
            w = cco.Window(win)
            sdma = cco.DevComm(dev_comm).sdma()

            flag, flag_rsrc = _next_flag(w, 0)
            if tid < ws:
                if tid != rank:
                    # Lane `tid` owns peer `tid` and queue `tid`: distinct queues
                    # per lane, so the posts do not serialise on one commit chain.
                    sdma.put(
                        tid,
                        win,
                        fx.Int64(dst_off_of_peer),
                        win,
                        src_off_expr(tid),
                        fx.Int64(slice_bytes),
                        tid,
                        coop=cco.CoopScope.THREAD,
                        signal=signal,
                    )
                    sdma.quiet_queue(tid, tid)
                # Release before publishing arrival. The bytes were moved by the
                # copy engine rather than by this CU, so there is nothing of ours
                # to flush; the fence is here to keep the flag store from being
                # hoisted above the drain, which is the one ordering the receiver
                # depends on.
                raw_cco.cco_system_fence(fx.Int32(0))
                _signal_and_wait(w, arr_off, flag, tid)
            fgpu.barrier()
            if tid == 0:
                local_store_u32(flag_rsrc, flag)

        return push

    # Scatter. Peer p owns slice p, so what p needs from me is *p's* slice range
    # of my input -- the source offset moves with the lane. It lands in the slot p
    # reserves for me, which is at the same byte offset in every rank's window.
    scatter = _push_kernel(
        start_off,
        dst_off_of_peer=my_recv_slot,
        src_off_expr=lambda tid: fx.Int64(in_off) + fx.Int64(tid) * fx.Int64(slice_bytes),
    )

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def sdma_reduce(dev_comm: Int64, win: Int64):
        """Sum my own slice plus the P-1 landed slices into my part of output.

        Peer order is ``(rank + j) % ws`` to match ``kernels_lsa``'s 2-stage, so
        the two backends are bitwise identical and the test can assert
        ``rel_l2 == 0`` rather than a tolerance.
        """
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        w = cco.Window(win)

        srcs = [
            create_buffer_resource_from_addr(
                wave_uniform_i64(w.lsa_ptr(rank, in_off + my_slice_off))
            )
        ] + [
            create_buffer_resource_from_addr(
                wave_uniform_i64(w.lsa_ptr(rank, cfg.recv_slot_off((rank + j) % ws)))
            )
            for j in range(1, ws)
        ]
        out = create_buffer_resource_from_addr(
            wave_uniform_i64(w.lsa_ptr(rank, out_off + my_slice_off))
        )

        gtid = bid * threads + tid
        for pk in range(gtid, part, stride_packs):
            i32_off = pk * I32_PER_PACK
            acc = None
            for j in range_constexpr(ws):
                raw = fx.Vector(
                    buffer_load(srcs[j], i32_off, vec_width=4, dtype=i32_type())
                )
                v = (
                    raw.bitcast(fx.Float32)
                    if elem_dtype is fx.Float32
                    else raw.bitcast(elem_dtype).to(fx.Float32)
                )
                acc = v if acc is None else acc + v
            packed = (
                acc.bitcast(fx.Int32)
                if elem_dtype is fx.Float32
                else acc.to(elem_dtype).bitcast(fx.Int32)
            )
            buffer_store(packed, out, i32_off, cache_modifier=CM_CACHED)

    # Gather. My reduced slice goes to the same offset in every peer's output.
    gather = _push_kernel(
        end_off,
        dst_off_of_peer=out_off + my_slice_off,
        src_off_expr=lambda tid: fx.Int64(out_off + my_slice_off),
    )

    @flyc.jit
    def run(dev_comm: Int64, win: Int64, stream=fx.Stream(None)):
        scatter(dev_comm, win).launch(
            grid=(1, 1, 1), block=[PUSH_THREADS, 1, 1], stream=stream
        )
        sdma_reduce(dev_comm, win).launch(
            grid=(blocks, 1, 1), block=[threads, 1, 1], stream=stream
        )
        gather(dev_comm, win).launch(
            grid=(1, 1, 1), block=[PUSH_THREADS, 1, 1], stream=stream
        )

    return run, 2


__all__ = ["build_sdma_ar", "PUSH_THREADS"]
