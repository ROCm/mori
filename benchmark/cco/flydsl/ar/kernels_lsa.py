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
"""LSA all-reduce in FlyDSL: 1-stage and 2-stage, mirroring aiter's kernels.

Transport is plain load/store against peer pointers obtained from cco's flat
symmetric VA (``Window.lsa_ptr``) -- the same mechanism aiter's custom all-reduce
uses via HIP IPC handles, so this is a substrate swap, not an algorithm change.

Protocol parity with ``aiter/csrc/include/custom_all_reduce.cuh``:

* per-(block, peer) signal slots and a **monotonic** ``_flag[block]`` that is
  never reset, which is what makes CUDA-graph replay safe (``start_sync``
  :160-201, ``end_sync`` :204-240);
* the signal store is uncached (SC0|SC1, i.e. system-visible) and the spin load
  bypasses L2 (SC1) -- aiter uses ``__MEMORY_SCOPE_SYSTEM`` / ``_DEVICE`` for the
  same reason;
* accumulation order matches aiter so results are bitwise comparable: 1-stage
  walks peers 0..P-1 unrotated (its source comment: "we don't reorder the
  address so the accumulation order is the same for all ranks"), 2-stage walks
  ``(rank + i) % P`` since each element is reduced exactly once by its owner;
* stage 2 reuses stage 1's thread->index mapping, because cross-device
  visibility is only guaranteed between threads with the same tid
  (``custom_all_reduce.cuh:556-560``).

Deliberate divergence: aiter's 2-stage stages every peer's pack through LDS and
reduces on ``warp_id == 0`` only, costing two ``__syncthreads()`` per iteration
at 1/8 occupancy. Here each thread reads all peers straight into registers. The
benchmark A/Bs the two.

Each kernel is specialised on the Python ``rank`` so every window offset folds
to a constant, as ``examples/cco/python/07_flydsl_sdma`` does.
"""

from __future__ import annotations

import os
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import gpu as fgpu
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int64

import mori.cco.device.flydsl as cco
from mori.cco.device.flydsl import _bindings as raw_cco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _compat import (  # noqa: E402
    CM_CACHED,
    CM_SC0_SC1,
    CM_SC1,
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
    i32_type,
    signal_load_u32,
    local_load_u32,
    local_store_u32,
    signal_ptr,
    wave_uniform_i64,
    signal_store_u32,
)
from layout import MAX_WORLD  # noqa: E402


def _spin_until(rsrc, flag):
    """Block until the u32 at ``rsrc`` reaches ``flag`` (unsigned compare).

    Unsigned so a wrapped counter still terminates; ``>=`` so an increment can
    never be missed, matching aiter's ``while (load < flag);``.
    """
    i32 = i32_type()
    first = signal_load_u32(rsrc)
    first_v = first.ir_value() if hasattr(first, "ir_value") else first
    loop = scf.WhileOp([i32], [first_v])
    cond = ir.Block.create_at_start(loop.before, [i32])
    body = ir.Block.create_at_start(loop.after, [i32])
    with ir.InsertionPoint(cond):
        cur = fx.Int32(cond.arguments[0])
        should_wait = fx.Uint32(cur) < fx.Uint32(flag)
        scf.ConditionOp(should_wait.ir_value(), [cond.arguments[0]])
    with ir.InsertionPoint(body):
        nxt = signal_load_u32(rsrc)
        scf.YieldOp([nxt.ir_value() if hasattr(nxt, "ir_value") else nxt])


def build_lsa_ar(cfg, rank: int, *, force_stage: int | None = None):
    """Compile an LSA all-reduce launcher for ``cfg`` on ``rank``.

    Returns ``(run, stage)`` where ``run(dev_comm, win, stream=...)`` performs one
    in-place all-reduce of the window's input region into its output region.
    """
    cfg.validate()
    ws = cfg.world_size
    stage = force_stage if force_stage is not None else cfg.stage
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage}")

    threads = cfg.threads
    blocks = cfg.blocks
    elems_per_pack = cfg.elems_per_pack
    elem_dtype = fx.BFloat16 if cfg.elem_bytes == 2 else fx.Float32
    num_packs = cfg.num_packs
    stride_packs = blocks * threads

    start_off, end_off, flag_off = cfg.start_off, cfg.end_off, cfg.flag_off
    in_off, out_off, tmp_off = cfg.input_off, cfg.output_off, cfg.tmp_off
    own_start, own_end = cfg.owner_pack_range(rank)
    own_packs = own_end - own_start
    part = cfg.packs_per_rank

    def _signal_and_wait(w, bid, arr_off, flag, tid):
        """Barrier body for the ``tid < ws`` lanes: publish, then wait.

        Branch-free on purpose. FlyDSL only rewrites ``if`` inside the
        ``@flyc.kernel`` function's own AST, so a runtime ``if`` in a helper is
        evaluated as a Python bool and raises during tracing; the guard has to
        stay in the kernel body.
        """
        peer_arr = fx.Int64(w.lsa_ptr(tid, arr_off))
        mine = fx.Int64((bid * MAX_WORLD + rank) * 4)
        signal_store_u32(signal_ptr(peer_arr + mine), flag)

        self_arr = fx.Int64(w.lsa_ptr(rank, arr_off))
        theirs = fx.Int64(bid * MAX_WORLD * 4) + fx.Int64(tid) * fx.Int64(4)
        _spin_until(signal_ptr(self_arr + theirs), flag)

    def _next_flag(w, bid):
        """``_flag[block] + 1`` -- monotonic, never reset (graph-replay safe)."""
        base = fx.Int64(w.lsa_ptr(rank, flag_off)) + fx.Int64(bid) * fx.Int64(4)
        rsrc = signal_ptr(base)
        return fx.Int32(local_load_u32(rsrc)) + fx.Int32(1), rsrc

    # A 16B pack is always moved as 4 x i32, so buffer offsets are in i32 units
    # regardless of the payload dtype -- for bf16 that pack holds 8 values, and
    # indexing it in payload elements would stride twice as far as intended.
    I32_PER_PACK = 4

    def _unpack(rsrc, i32_off):
        """One 16B pack -> a float32 vector of ``elems_per_pack`` lanes."""
        raw = fx.Vector(buffer_load(rsrc, i32_off, vec_width=4, dtype=i32_type()))
        if elem_dtype is fx.Float32:
            return raw.bitcast(fx.Float32)
        return raw.bitcast(elem_dtype).to(fx.Float32)

    def _pack(acc):
        """float32 vector -> the 4 x i32 that the payload dtype occupies."""
        if elem_dtype is fx.Float32:
            return acc.bitcast(fx.Int32)
        return acc.to(elem_dtype).bitcast(fx.Int32)

    def _acc_peers(rsrcs, pack_idx):
        """fp32 sum of one 16B pack across every peer, in ``rsrcs`` order."""
        i32_off = pack_idx * I32_PER_PACK
        acc = None
        for i in range_constexpr(len(rsrcs)):
            v = _unpack(rsrcs[i], i32_off)
            acc = v if acc is None else acc + v
        return acc

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def ar_1stage(dev_comm: Int64, win: Int64):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        w = cco.Window(win)

        flag, flag_rsrc = _next_flag(w, bid)
        if tid < ws:
            _signal_and_wait(w, bid, start_off, flag, tid)
        fgpu.barrier()
        if tid == 0:
            local_store_u32(flag_rsrc, flag)

        # Unrotated peer order: every rank accumulates identically, so all ranks
        # produce bitwise identical output (aiter's 1-stage does the same).
        ins = [
            create_buffer_resource_from_addr(wave_uniform_i64(w.lsa_ptr(p, in_off)))
            for p in range(ws)
        ]
        out = create_buffer_resource_from_addr(wave_uniform_i64(w.lsa_ptr(rank, out_off)))

        gtid = bid * threads + tid
        for pk in range(gtid, num_packs, stride_packs):
            acc = _acc_peers(ins, pk)
            buffer_store(_pack(acc), out, pk * I32_PER_PACK, cache_modifier=CM_CACHED)

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def ar_2stage(dev_comm: Int64, win: Int64):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        w = cco.Window(win)

        flag_a, flag_rsrc = _next_flag(w, bid)
        if tid < ws:
            _signal_and_wait(w, bid, start_off, flag_a, tid)
        fgpu.barrier()
        if tid == 0:
            local_store_u32(flag_rsrc, flag_a)

        # --- stage 1: reduce-scatter. I own packs [own_start, own_end) and read
        # that slice from every peer, rotated so the 8 links are not all pointed
        # at rank 0 at once. Rotation is safe for bitwise agreement because each
        # pack has exactly one owner.
        ins = [
            create_buffer_resource_from_addr(
                wave_uniform_i64(w.lsa_ptr((rank + i) % ws, in_off))
            )
            for i in range(ws)
        ]
        tmp = create_buffer_resource_from_addr(wave_uniform_i64(w.lsa_ptr(rank, tmp_off)))

        gtid = bid * threads + tid
        for pk in range(gtid, own_packs, stride_packs):
            acc = _acc_peers(ins, own_start + pk)
            buffer_store(_pack(acc), tmp, pk * I32_PER_PACK, cache_modifier=CM_CACHED)

        # Release barrier. Three things must happen in this order, and each was
        # a real bug when missing:
        #   1. block-wide barrier so every lane's tmp store is issued, AND so
        #      thread 0's `_flag` update from the first barrier is visible --
        #      without it the other lanes re-read the stale flag, compute the
        #      same value again, and the second barrier is already satisfied
        #      (aiter's end_sync opens with __syncthreads() for exactly this);
        #   2. a system fence, the release that flushes those cached tmp stores
        #      out where peers reading over xGMI can see them (aiter gets this
        #      from __ATOMIC_RELEASE at __MEMORY_SCOPE_SYSTEM on the signal);
        #   3. only then publish the signal.
        # leaderOnly=0: every lane fences. The cco ABI notes leaderOnly=1 fences
        # thread 0 only and is not a substitute for a per-lane release.
        fgpu.barrier()
        raw_cco.cco_system_fence(fx.Int32(0))
        flag_b, flag_rsrc_b = _next_flag(w, bid)
        if tid < ws:
            _signal_and_wait(w, bid, end_off, flag_b, tid)
        # Acquire half of the pair: invalidate lines this rank may still hold
        # for a peer's tmp. The barrier's atomic load is agent-scope monotonic,
        # which orders but does not invalidate, so without this the gather can
        # read stale L2. aiter gets it from __ATOMIC_ACQUIRE on its spin load.
        raw_cco.cco_system_fence(fx.Int32(0))
        fgpu.barrier()
        if tid == 0:
            local_store_u32(flag_rsrc_b, flag_b)

        # --- stage 2: all-gather. Thread `gtid` reads exactly the tmp indices it
        # wrote in stage 1 -- cross-device visibility only holds between threads
        # with the same tid.
        out = create_buffer_resource_from_addr(wave_uniform_i64(w.lsa_ptr(rank, out_off)))
        for j in range_constexpr(ws):
            src_rank = (rank + j) % ws
            src = create_buffer_resource_from_addr(
                wave_uniform_i64(w.lsa_ptr(src_rank, tmp_off))
            )
            dst_base = src_rank * part
            n_here = cfg.owner_pack_range(src_rank)[1] - dst_base
            for pk in range(gtid, n_here, stride_packs):
                v = buffer_load(
                    src, pk * I32_PER_PACK, vec_width=4, dtype=i32_type()
                )
                buffer_store(
                    v, out, (dst_base + pk) * I32_PER_PACK, cache_modifier=CM_CACHED
                )

    kernel = ar_1stage if stage == 1 else ar_2stage

    @flyc.jit
    def run(dev_comm: Int64, win: Int64, stream=fx.Stream(None)):
        kernel(dev_comm, win).launch(
            grid=(blocks, 1, 1), block=[threads, 1, 1], stream=stream
        )

    return run, stage


__all__ = ["build_lsa_ar"]
