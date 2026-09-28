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
"""The all-gather over SDMA: one contiguous push per peer, all of them the same.

The copy engines cannot gather, and this is the one op in the family where that
costs nothing. ``gemm_a2a`` has to build a ``[dst][M][shard_n]`` staging slab in
its GEMM epilogue purely so the engine has a contiguous source range per
destination; ``gemm_ar`` shards its reduce-scatter along ``m`` for the same
reason. All-gather's payload is the rank's whole ``[M, N]``, already contiguous
and already identical for every destination, so the transfer is:

    lane d, queue d:  my recv slot  ->  peer d's recv slot for me

with **the same byte offset on both sides**, and that offset a compile-time
constant folded from the Python ``rank``. There is no source-side index at all.

## Two phases, and why ``drain`` exists

``gather`` issues the puts, drains, and barriers -- that is ``split-sdma``.
``drain`` does everything except issue: the fused GEMM already posted the puts
from its epilogue as each chunk completed, so all that is left is to wait for
the queues and agree with the peers. Compiling them from one builder is
deliberate: the two must use the same queue map and the same barrier slots, and
a second copy of that would be a second thing to keep in step.

Protocol is gcnasm's, ``gemm_ar``'s and ``gemm_a2a``'s: **no per-PUT signal**,
drain on the sender, then an LSA atomic to publish arrival. A trailing
signal packet costs about what a copy packet does and nothing here polls it.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int64

import mori.cco.device.flydsl as cco
from mori.cco.device.flydsl import _bindings as raw_cco

from ..gemm_ar._compat import (
    local_load_u32,
    local_store_u32,
    signal_ptr,
    signal_store_u32,
)
from .kernels_lsa import _spin_until

#: One wave is enough: the kernel issues at most ``world_size`` puts and then
#: polls. gemm_ar and gemm_a2a use the same number for the same reason.
PUSH_THREADS = 64


def build_sdma_phases(cfg, rank: int, *, queues: int = 1, signal: bool = False):
    """Compile the push phases for ``cfg`` on ``rank``.

    Returns ``{"gather", "drain"}``, each a launcher taking
    ``(dev_comm, win, stream=...)``.

    * ``gather`` pushes this rank's slab to every peer *and* drains -- the split
      path.
    * ``drain`` only drains and barriers, for the fused path where the pushes
      were already issued from inside the GEMM epilogue.

    ``queues`` must match ``reqs.sdma_queue_count``. Queue ids are taken modulo
    it, and the hardware queues are per **(source, destination) pair**, so even
    ``queues=1`` gives every peer its own queue -- which is all this pipeline
    needs, since one destination is one xGMI link and cco's concurrency rule is
    about concurrently-issuing *warps* per queue, of which there is one here.
    Asking for ``world_size`` would create ``world_size`` queues per peer and
    touch one of them; inside a process that already holds SDMA engines,
    ``hsaKmtCreateQueueExt`` then starts failing (``anvil.cpp:237``).

    The fused lane-parallel producer assigns chunks to ``chunk % queues``.
    Its drain must therefore wait for every queue to a peer before publishing
    arrival; completion on one queue does not order transfers on another.

    ``signal`` selects ``put``'s trailing local ATOMIC, off by default:
    ``quiet_queue`` drains the queue's read pointer independently of signals and
    nothing here polls one, so the packet would be pure overhead.

    Note what is *not* a parameter, unlike ``gemm_a2a``'s builder: there is no
    ``cfg.staged`` check, because there is nothing to stage.
    """
    cfg.validate()
    if queues < 1:
        raise ValueError(f"queues must be >= 1, got {queues}")

    ws = cfg.world_size
    start_off, flag_off = cfg.start_off, cfg.flag_off
    slab_bytes = cfg.slab_bytes
    # My slot in *every* window, my own included -- the source and the
    # destination are the same offset. That is what indexing ``recv`` by source
    # rather than by destination buys, and in a broadcast it buys both ends.
    my_recv_slot = cfg.recv_slot_off(rank)

    def _next_flag(w):
        """``_flag[0] + 1`` -- monotonic, never reset (graph-replay safe)."""
        rsrc = signal_ptr(fx.Int64(w.lsa_ptr(rank, flag_off)))
        return fx.Int32(local_load_u32(rsrc)) + fx.Int32(1), rsrc

    def _signal_and_wait(w, flag, tid):
        """Publish to peer ``tid``, then wait on ``tid``'s slot.

        A closure taking ``w`` rather than inlined into the ``if`` in the kernel:
        a ``CachedWindow`` is three MLIR values and FlyDSL's scf.if state capture
        wants single ones. Inlining it raises ``Cannot extract IR values from
        CachedWindow``.

        Slot map is ``rank*4`` within ``start_off``, which is block 0's row of
        the ``(bid*MAX_WORLD + rank)*4`` map the LSA kernels use -- these kernels
        are one block, so the two agree.
        """
        peer_arr = fx.Int64(w.lsa_ptr(tid, start_off))
        signal_store_u32(signal_ptr(peer_arr + fx.Int64(rank * 4)), flag)
        self_arr = fx.Int64(w.lsa_ptr(rank, start_off))
        _spin_until(signal_ptr(self_arr + fx.Int64(tid) * fx.Int64(4)), flag)

    def _push_kernel(pushes: bool, name: str):
        @flyc.kernel(name=name, known_block_size=[PUSH_THREADS, 1, 1])
        def push(dev_comm: Int64, win: Int64):
            tid = fx.thread_idx.x
            w = cco.CachedWindow(win)
            sdma = cco.DevComm(dev_comm).sdma()

            flag, flag_rsrc = _next_flag(w)
            if tid < ws:
                if tid != rank:
                    if const_expr(pushes):
                        # Lane `tid` owns peer `tid` and queue `tid`: distinct
                        # queues per lane, so the posts do not serialise on one
                        # commit chain.
                        sdma.put(
                            tid,
                            win,
                            fx.Int64(my_recv_slot),
                            win,
                            fx.Int64(my_recv_slot),
                            fx.Int64(slab_bytes),
                            tid % fx.Int32(queues),
                            coop=cco.CoopScope.THREAD,
                            signal=signal,
                        )
                    if const_expr(pushes or queues == 1):
                        # The split kernel submitted only this queue. The
                        # single-queue fused path has the same completion set.
                        sdma.quiet_queue(tid, tid % fx.Int32(queues))
                    else:
                        # Different chunks may finish in any order and use
                        # different queues. Cover every producer submission.
                        sdma.quiet(tid, coop=cco.CoopScope.THREAD)
                # Release before publishing arrival. The bytes were moved by the
                # copy engine rather than by this CU, so there is nothing of ours
                # to flush; the fence keeps the flag store from being hoisted
                # above the drain, which is the one ordering the receiver relies
                # on.
                raw_cco.cco_system_fence(fx.Int32(0))
                _signal_and_wait(w, flag, tid)
            fgpu.barrier()
            if tid == 0:
                local_store_u32(flag_rsrc, flag)

        @flyc.jit
        def run(dev_comm: Int64, win: Int64, stream: fx.Stream = fx.Stream(None)):
            push(dev_comm, win).launch(
                grid=(1, 1, 1), block=[PUSH_THREADS, 1, 1], stream=stream
            )

        return run

    return {
        "gather": _push_kernel(True, f"mori_ag_sdma_gather_r{rank}_q{queues}"),
        "drain": _push_kernel(False, f"mori_ag_sdma_drain_allq_r{rank}_q{queues}"),
    }


__all__ = ["build_sdma_phases", "PUSH_THREADS"]
