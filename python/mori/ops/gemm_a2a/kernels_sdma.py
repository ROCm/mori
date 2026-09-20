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
"""The all-to-all over SDMA: one contiguous push per destination.

The copy engines cannot gather, so this path needs the payload laid out as one
contiguous range per destination before anything is pushed. That is the
``[dst][M][shard_n]`` staging slab, and producing it is the GEMM's job -- see
``kernels_fused``. Once it exists, the transfer is the simplest possible shape:

    lane d, queue d:  staging[d]  ->  peer d's recv slot for me

Both offsets are constants. The destination offset is the *same* in every peer's
window, which is what indexing ``recv`` by source rather than by destination
buys; the source offset is just ``d`` slabs into staging.

## Two phases, and why ``drain`` exists

``scatter`` issues the puts, drains, and barriers -- that is ``split-sdma``.
``drain`` does everything except issue: the fused GEMM already posted the puts
from its epilogue as each chunk completed, so all that is left is to wait for
the queues and agree with the peers. Compiling them from one builder is
deliberate: the two must use the same queue map and the same barrier slots, and
a second copy of that would be a second thing to keep in step.

Protocol is gcnasm's and ``gemm_ar``'s: **no per-PUT signal**, ``quiet_queue``
on the sender, then an LSA atomic to publish arrival. A trailing signal packet
costs about what a copy packet does and nothing here polls it.
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
#: polls. gemm_ar uses the same number for the same reason.
PUSH_THREADS = 64


def build_sdma_phases(cfg, rank: int, *, queues: int = 1, signal: bool = False):
    """Compile the push phases for ``cfg`` on ``rank``.

    Returns ``{"scatter", "drain"}``, each a launcher taking
    ``(dev_comm, win, stream=...)``.

    * ``scatter`` pushes every destination's slab *and* drains -- the split path.
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

    ``signal`` selects ``put``'s trailing local ATOMIC, off by default:
    ``quiet_queue`` drains the queue's read pointer independently of signals and
    nothing here polls one, so the packet would be pure overhead.
    """
    cfg.validate()
    if not cfg.staged:
        raise ValueError(
            "the SDMA path needs a config built with staged=True: a copy engine "
            "reads one contiguous source range and cannot gather a column block "
            "out of a row-major [M, N]"
        )
    if queues < 1:
        raise ValueError(f"queues must be >= 1, got {queues}")

    ws = cfg.world_size
    start_off, flag_off = cfg.start_off, cfg.flag_off
    staging_off = cfg.staging_off
    slab_bytes = cfg.slab_bytes
    cap_slab_bytes = cfg.cap_slab_bytes
    # My slot in *every* peer's window -- the same byte offset on all of them.
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
                            fx.Int64(staging_off)
                            + fx.Int64(tid) * fx.Int64(cap_slab_bytes),
                            fx.Int64(slab_bytes),
                            tid % fx.Int32(queues),
                            coop=cco.CoopScope.THREAD,
                            signal=signal,
                        )
                    sdma.quiet_queue(tid, tid % fx.Int32(queues))
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
        "scatter": _push_kernel(True, f"mori_a2a_sdma_scatter_r{rank}_q{queues}"),
        "drain": _push_kernel(False, f"mori_a2a_sdma_drain_r{rank}_q{queues}"),
    }


__all__ = ["build_sdma_phases", "PUSH_THREADS"]
