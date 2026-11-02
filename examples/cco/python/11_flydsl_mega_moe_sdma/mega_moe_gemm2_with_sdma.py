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
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: B023, I001
"""MegaMoE stage2 + combine for gfx950 with the transfer done by cco SDMA: the host op ``MegaMoESdma`` and
its three kernels (the plan kernel, GEMM2 with comm waves, and the combine kernel).

Stage2 normally scatters every GEMM2 row straight into the origin rank's
``comb_inp``. Here the rows go to a local send buffer ``S`` grouped by origin
rank, so one contiguous run per peer can be moved by a copy engine::

    S = [ origin rank 0: rows... | origin rank 1: rows... | ... ]   (exact counts, packed)

Within a rank's run rows are in sorted-row order, i.e. expert-major, so a prefix
of every run is finished as soon as the first experts are. That is what lets the
GEMM2 kernel ship chunks while it is still computing.

Three kernels on one stream (the P2P-scatter baseline has two):

    1. rank_sort_plan   persistent grid; count -> prefix -> assign with one rendezvous:
                        out_row per sorted row, idx_send, tile_base, cursor, and ``inv``
                        for the rows that stay on this rank; zeroes cdone
    2. gemm2_with_sdma  writes S[out_row]; extra comm waves (one per peer) fill that peer's
                        ``inv``, then put finished chunks with SDMA while the GEMM tiles
                        are still computing
    3. combine          wait for the peers' flags -> top-k reduce -> publish "consumed"

Everything lives in ONE cco window, so ``put`` names the same window for source
and destination and only the byte offsets differ::

    [ S | R | inv | sig | flags | done | counter ]

``inv[slot]`` is the landing row of the row for token slot ``slot`` (``origin_lid*topk
+ k``), or -1. The SENDER writes it -- the plan kernel for rows that stay here, the
sender's comm waves (direct stores into the peer's window) for the rest -- so no index
travels with the data and the receiver builds no table. ``sig`` is scratch for the
4-byte put that carries the completion signal.

``flags[src]`` says "src's data for call n has landed".
``done[dst]`` says "dst has finished reading call n": a sender waits for it before
overwriting that peer's ``R`` and ``inv`` with call n+1, which keeps a fast rank
from running into a slow one. ``counter`` is the call number; only the combine
kernel advances it, after every reader of it has run.

A row's *landing row* is its row index counted from the start of ``S``: ``S`` rows
are ``[0, send_rows)`` and ``R`` rows ``[send_rows, send_rows + recv_rows)``. My own run
never travels; it is read from ``S`` in place.

The layout is exact. aiter's prepare all-gathers every rank's per-expert route counts
(``bigcnt``), so the plan kernel knows how many rows every rank sends every other
before anything moves: my run for ``d`` starts at ``send_base[d]`` (the rows I send
lower-numbered ranks) and lands in ``d``'s ``R`` at ``land_off[d]`` (the rows
lower-numbered ranks other than ``d`` send ``d``) -- no per-pair reservation and no
handshake. ``R`` holds what one rank gets back, at most ``max_tok * topk`` rows, so it
never overflows. ``S`` holds what one rank sends -- everything it received in the
dispatch, which skewed routing can make several times the even share -- and is sized
``send_slack`` times ``max_tok * topk`` (capped by the buffer-resource range). The plan
kernel never places a row past it.

Rows are transported as fp8 with one e8m0 scale per 32 values, laid out as
aiter's P2P scatter lays them out -- payload then scales, inline -- so one
contiguous put still carries a whole row. bf16 doubles the bytes and the
transfer stops fitting under the GEMM.

The plan kernel rendezvouses across its grid, so that grid must be fully resident
(one block per CU is). The combine has no grid-wide rendezvous and is not resident.
"""
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cvt_pk_f32_fp8
from flydsl.expr.typing import Int8, T
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.compiler.llvm_options import llvm_options
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops

# A device-side helper defined outside a kernel body is not rewritten by
# @flyc.kernel, so a data-dependent ``if`` inside it would stay a plain Python
# branch on a traced value. flyc.jit turns such a body into scf ops with
# ASTRewriter.transform; apply the same thing by hand on aiter builds that do
# not export comm_ops.traced yet.
if not hasattr(comm_ops, "traced"):
    comm_ops.traced = ASTRewriter.transform
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import (
    _fabs_f32 as fabs_f32,
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)

from aiter.ops.flydsl.kernels.mega_moe.gemm2 import (
    _resolve_g2_knobs,
    _spart_output_tile_index,
    gemm2_compute_v2,
    issue_a_load_lds_dt,
    kStages,
)
from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage2 import (
    _BUFFER_OFFSET_ABI_BYTES,
    _fp8_scale_for_leader,
    _stage2_lds_bytes,
)


#: i32 slots between consecutive chunk-completion counters. Every unit of a
#: chunk bumps its counter, so packing them adjacent puts all of a launch's
#: atomics on one or two cache lines: 128 B apart instead is worth ~8% of the
#: GEMM2 kernel at bs=4096. Spreading a chunk's count over several counters on
#: top of this buys nothing -- the atomic itself is not the cost.
CDONE_STRIDE = 32

#: Rows per lane per loop trip when a comm wave pushes inv entries to a peer.
PUSH_U = 8

#: ``out_row`` of a valid row past the end of ``S``, under the P2P fallback:
#: GEMM2 writes it straight into the origin rank's ``comb_inp[slot]``.
OUT_ROW_P2P = -2
#: ``inv`` of a slot whose row is in this rank's ``comb_inp`` (the P2P fallback).
INV_P2P = -2


BLOCK = 256
WAVE = 64
BAR_STRIDE = 32  # ints per tree_barrier counter: one 128 B cache line
BAR_GROUP = 32  # blocks per tree_barrier group
MIN_SBM = 16


# --- The cco window and the sender side of the SDMA protocol ---------------------------

_ALIGN = 128


def _align(n):
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


@dataclass(frozen=True)
class ArenaLayout:
    """Byte offsets inside the cco window; identical on every rank."""

    npes: int
    send_rows: int
    recv_rows: int
    row_bytes: int
    slot_cap: int

    @property
    def off_s(self):
        return 0

    @property
    def off_r(self):
        return self.off_s + self.send_rows * self.row_bytes

    @property
    def off_inv(self):
        return self.off_r + self.recv_rows * self.row_bytes

    @property
    def off_sig(self):
        return self.off_inv + _align(self.slot_cap * 4)

    @property
    def off_flags(self):
        return self.off_sig + _align(self.npes * 4)

    @property
    def off_done(self):
        return self.off_flags + _align(self.npes * 4)

    @property
    def off_counter(self):
        return self.off_done + _align(self.npes * 4)

    @property
    def total_bytes(self):
        return self.off_counter + _ALIGN

    def check(self):
        # The reduce addresses S and R through one 32-bit buffer-resource offset.
        if self.off_inv >= 1 << 32:
            raise ValueError("S+R exceed the 4 GiB buffer-resource range")


def sdma_wait_consumed(w, peer, phase, *, layout, rank):
    """Block until ``peer`` has finished reading the previous call's data from my sends."""
    comm_ops.spin_until_ge_i32_system(
        fx.Int64(w.lsa_ptr(rank, layout.off_done)) + fx.Int64(peer) * fx.Int64(4),
        phase - fx.Int32(1),
        acquire=True,
    )


def sdma_put_run(
    sdma,
    win,
    peer,
    queue,
    *,
    layout,
    send_base,
    land_off,
    lo,
    hi,
    cco,
    remote_signal=False,
):
    """Ship rows ``[lo, hi)`` of my run for ``peer`` (``S`` row ``send_base`` on) to
    ``peer``'s ``R`` row ``land_off`` on.

    No local signal: a put that names none keeps cco's old default, a trailing
    ATOMIC into our own signal pool, and nothing here waits on it -- completion
    is the peer's remote signal or ``quiet_queue``. Dropping it is one packet
    less per put (-3.6% stage2+combine at 2048 tokens/rank, -0.5% at 8192).
    ``remote_signal`` hangs the completion signal on this put instead (see
    ``sdma_finish_peer``).

    One put for the whole run. Puts are asynchronous -- each reserves a slot,
    writes the copy descriptor and rings the doorbell -- but a queue is a ring
    its engine drains in order, so splitting a run across channels only buys
    parallelism if those channels sit on different engines. KFD recommends a
    single SDMA engine per peer link on this topology (one bit per link in
    ``recommended_sdma_engine_id_mask``), so they do not: splitting a 1.44 MB run
    in two measured 3% slower, rotating whole runs across two channels 2%.
    """
    rb = layout.row_bytes
    sdma.put(
        peer,
        win,
        fx.Int64(layout.off_r) + (fx.Int64(land_off) + fx.Int64(lo)) * fx.Int64(rb),
        win,
        fx.Int64(layout.off_s) + (fx.Int64(send_base) + fx.Int64(lo)) * fx.Int64(rb),
        fx.Int64(hi - lo) * fx.Int64(rb),
        queue,
        coop=cco.CoopScope.THREAD,
        local_signal=False,
        remote_signal=remote_signal,
    )


@comm_ops.traced
def sdma_finish_peer(
    sdma,
    w,
    win,
    peer,
    queue,
    phase,
    *,
    layout,
    rank,
    cco,
    signal_mode="remote",
    signalled=False,
):
    """Tell ``peer`` its rows have landed.

    Two ways to say "landed":

    ``quiet``   drain our queue here, then store the flag. The drain waits for
                the last chunk to reach the peer, and that wait sits at the tail
                of the GEMM2 kernel, after the last tile -- 42us of a 772us
                kernel at bs=4096, CT=16.
    ``remote``  hang the copy engine's own trailing ATOMIC on this put, so the
                engine bumps a counter on the peer once the bytes are there.
                Nothing is drained: the comm wave issues and leaves, and the
                peer's combine -- which was going to wait for us anyway -- polls
                the counter. Worth 6-12us end to end.

    ``remote`` needs this put to be the LAST op on the channel the data went out
    on. It is: everything for ``peer`` uses channel ``peer % queues``, and a
    queue is a FIFO its engine drains in order, so the trailing ATOMIC fires
    only once every chunk ahead of it has landed.

    Either way the flag carries the call number. Under ``remote`` it says nothing
    about the data -- the signal does that.

    ``signalled``: the last data put for ``peer`` already carried the signal, so
    the 4-byte put is skipped. Only a peer whose last chunk was empty needs it.
    """
    if signal_mode not in ("quiet", "remote"):
        raise ValueError(f"signal_mode must be quiet or remote, got {signal_mode!r}")
    remote = signal_mode == "remote"
    # Under `remote` this put carries the signal: 4 bytes the peer never reads.
    if const_expr(remote and not signalled):
        sdma.put(
            peer,
            win,
            fx.Int64(layout.off_sig + rank * 4),
            win,
            fx.Int64(layout.off_sig + peer * 4),
            fx.Int64(4),
            queue,
            coop=cco.CoopScope.THREAD,
            remote_signal=True,
        )
    if const_expr(not remote):
        sdma.quiet_queue(peer, queue)
        # The bytes moved by the copy engine, not by this CU, so nothing of ours
        # has to be written back -- this only keeps the flag store from being
        # hoisted above the drain.
        rocdl.s_waitcnt(vmcnt=0)
    comm_ops.store_i32_global_system_release(
        fx.Int64(w.lsa_ptr(peer, layout.off_flags)) + fx.Int64(rank * 4), phase
    )


def sdma_signal_peer(sdma, win, peer, queue, *, layout, rank, cco):
    """A 4-byte put that only carries the completion signal (``remote`` mode, for a
    peer whose last chunk was empty)."""
    sdma.put(
        peer,
        win,
        fx.Int64(layout.off_sig + rank * 4),
        win,
        fx.Int64(layout.off_sig + peer * 4),
        fx.Int64(4),
        queue,
        coop=cco.CoopScope.THREAD,
        remote_signal=True,
    )


def publish_flag(w, peer, phase, *, layout, rank):
    """``flags[rank]`` on ``peer`` = call number (``remote`` mode: the data itself is
    covered by the SDMA signal)."""
    comm_ops.store_i32_global_system_release(
        fx.Int64(w.lsa_ptr(peer, layout.off_flags)) + fx.Int64(rank * 4), phase
    )


# --- Kernel 1: rank sort plan --------------------------------------------------


def _uniform_i64(x):
    """A wave-uniform i64 (SGPR) copy of ``x``, as a buffer-resource base needs."""
    return fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(x).ir_value()))


def _decode(p, *, log2_max_tok, recv_cap, topk, npes):
    """Packed srcmap word -> (valid, dest_pe, dest_lid, slot_k), as the stage2 epilogue does."""
    t = p & fx.Int32(0x00FFFFFF)
    s = p.shrui(fx.Int32(24)) & fx.Int32(0xFF)
    dest_pe = t >> fx.Int32(log2_max_tok)
    dest_lid = t & fx.Int32((1 << log2_max_tok) - 1)
    valid = (t < fx.Int32(recv_cap)) & (s < fx.Int32(topk)) & (dest_pe < fx.Int32(npes))
    return valid, dest_pe, dest_lid, s


@comm_ops.traced
def tree_barrier(bar_addr, nblocks, tid, bid):
    """All blocks of a resident grid rendezvous with two levels of small atomics.

    Blocks are grouped ``BAR_GROUP`` to a counter; the last block of a group bumps a
    top counter; everyone polls only the top one. No address sees more than
    ``BAR_GROUP`` atomics, and a round of polling is one load per block, not
    ``nblocks`` of them (which, spun on by every thread of every block, slowed the
    very stores it was waiting for). Counters are never reset: a group's counter
    advances by its size per launch, so the launch number is ``old // size``.
    Each counter sits on its own cache line.
    """
    group = bid // fx.Int32(BAR_GROUP)
    n_groups = (nblocks + fx.Int32(BAR_GROUP - 1)) // fx.Int32(BAR_GROUP)
    size = nblocks - group * fx.Int32(BAR_GROUP)
    size = (size < fx.Int32(BAR_GROUP)).select(size, fx.Int32(BAR_GROUP))
    rocdl.s_waitcnt(0)
    fx.barrier()
    if tid == fx.Int32(0):
        comm_ops.fence_agent_release()
        old = fx.Int32(
            comm_ops.atomic_add_agent(
                bar_addr + fx.Int64(group) * fx.Int64(BAR_STRIDE * 4), fx.Int32(1)
            )
        )
        launch = old // size
        if old - launch * size == size - fx.Int32(1):
            comm_ops.atomic_add_agent(
                bar_addr + fx.Int64(n_groups) * fx.Int64(BAR_STRIDE * 4), fx.Int32(1)
            )
        comm_ops.spin_until_ge_i32_agent(
            bar_addr + fx.Int64(n_groups) * fx.Int64(BAR_STRIDE * 4),
            (launch + fx.Int32(1)) * n_groups,
        )
        comm_ops.fence_agent_acquire()
    fx.barrier()


def build_rank_sort_plan(
    *,
    SBM,
    topk,
    log2_max_tok,
    recv_cap,
    npes,
    rank,
    epr,
    total_segments,
    send_rows,
    p2p_fallback=False,
):
    """Kernel 1: give every valid sorted row a stable index in its rank's run.

    * offsets: from aiter's gathered route counts ``bigcnt[src][segment]`` (per expert,
              then one group column per rank for the dispatch's fanout pair), per pe:
              ``n_send`` (rows I send it), ``send_base`` (rows I send lower pes),
              ``land_off`` (rows lower ranks other than pe send it) and ``avail``
              (rows of its run that fit in ``S``)
    * count:  ``tile_cnt[tile][pe]`` = rows of the sort tile bound for ``pe``
    * prefix: ``tile_base[tile][pe]`` = rows for ``pe`` in earlier tiles (so the
              rows of a tile range are contiguous in the run), ``cursor[pe]`` = total
    * assign: ``out_row = send_base[pe] + tile_base + (earlier rows of this tile for pe)``,
              and ``idx_send[out_row] = origin_lid*topk + k``. Rows bound for this rank
              also get their ``inv`` entry here; the other ranks' entries are pushed by
              the GEMM2 kernel's comm waves, off this kernel's critical path

    One rendezvous, between count and prefix; the prefix needs every earlier tile's
    count. It is a ``tree_barrier`` (no address takes more than a group's worth of atomics). After it, a block
    does not scan the whole ``tile_cnt`` array: for each tile it owns it sums the
    counts of the earlier tiles with one pass of the block's threads (thread ``t``
    handles pe ``t % npes``) and a short LDS reduction -- a parallel sum, not a
    log-depth scan with a barrier per step. The sum runs to the tile's own index,
    so nothing here is sized by a bound on the tile count: a rank that receives
    far more rows than an even share (skewed routing) is handled the same way.

    Before it writes anything the kernel waits until every peer has consumed the
    previous call: that is what guarantees this rank's own outbound puts have
    finished reading ``S`` (rewritten by the GEMM2 that follows).
    ``tile_base`` is clamped to ``avail``, so a chunk never ships past the end of ``S``.
    ``cursor`` = [rows in the run | send_base | land_off | avail] per pe, then a counter
    of pes whose counted rows differ from the gathered ones (must stay 0).

    Count and assign each run one block per tile with one thread per row: a row
    is decoded once, a wave ballot per pe gives the rows of the wave bound for it,
    and the per-wave counts go through a few LDS words. A row's rank among the
    earlier rows for its pe is then a popcount, not a walk over the tile.

    A row past ``avail`` is counted in ``arg_over``. Without ``p2p_fallback`` it is
    dropped (the host must check the counter); with it, it gets ``OUT_ROW_P2P`` and
    GEMM2 writes it into the origin rank's ``comb_inp`` instead.
    """
    dec = dict(log2_max_tok=log2_max_tok, recv_cap=recv_cap, topk=topk, npes=npes)

    if SBM > BLOCK:
        raise ValueError(
            f"plan needs one thread per row of a sort tile: SBM={SBM} > {BLOCK}"
        )
    if BLOCK % npes:
        raise ValueError(f"plan needs npes to divide {BLOCK}, got {npes}")
    GROUPS = BLOCK // npes
    WAVES = BLOCK // WAVE
    # LDS words: per-wave per-pe counts | per-thread partial sums | the tile's prefix.
    OFF_PART = WAVES * npes
    OFF_PREF = OFF_PART + BLOCK
    # Per pe, for the whole kernel: rows I send it, my run's base in S, rows that fit.
    OFF_NSEND = OFF_PREF + npes
    OFF_SBASE = OFF_NSEND + npes
    OFF_AVAIL = OFF_SBASE + npes
    OFF_LAND = OFF_AVAIL + npes
    lds_bytes = (OFF_LAND + npes) * 4

    @fx.struct
    class PlanShared:
        buf: fx.Array[Int8, lds_bytes, 16]

    # fmt: off
    @flyc.kernel(name=f"mega_rank_sort_plan_sbm{SBM}_k{topk}_p{npes}_s{send_rows}"
                      f"_e{epr}x{total_segments}_f{int(p2p_fallback)}_r{rank}",
                 known_block_size=[BLOCK, 1, 1])
    def plan(arg_stids: fx.Int64, arg_trb: fx.Int64, arg_num_valid: fx.Int64,
        arg_tile_cnt: fx.Int64, arg_tile_base: fx.Int64, arg_cursor: fx.Int64,
        arg_out_row: fx.Int64, arg_idx: fx.Int64, arg_inv: fx.Int64, arg_cdone: fx.Int64,
        arg_bar: fx.Int64, arg_over: fx.Int64, arg_done: fx.Int64, arg_counter: fx.Int64,
        arg_bigcnt: fx.Int64, arg_pair_cfg: fx.Int64, arg_parity: fx.Int64,
        i32_nblocks: fx.Int32, i32_cdone_n: fx.Int32):
    # fmt: on
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        gid = bid * fx.Int32(BLOCK) + tid
        stride = i32_nblocks * fx.Int32(BLOCK)
        num_valid = global_typed_ptr(arg_num_valid, T.i32)[0]
        tiles = (num_valid + fx.Int32(SBM - 1)) // fx.Int32(SBM)
        trb = ptr_buf_tensor(arg_trb, fx.Int32)
        stids = ptr_buf_tensor(arg_stids, fx.Int32)
        tile_cnt = ptr_buf_tensor(arg_tile_cnt, fx.Int32)
        tile_base = ptr_buf_tensor(arg_tile_base, fx.Int32)
        out_buf = ptr_buf_tensor(arg_out_row, fx.Int32)
        cdone = ptr_buf_tensor(arg_cdone, fx.Int32)

        lds = fx.SharedAllocator().allocate(PlanShared).peek()
        lds_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))

        def lw(i):
            """LDS i32 word ``i`` of the block's scratch."""
            return lds_typed_ptr(lds_i32 + i * fx.Int32(4), T.i32, align=4)

        # S and the peers' inv are rewritten from here on. A peer's done flag says it
        # has read everything I sent it last call, and with the copy engine's FIFO that
        # means my puts have finished reading S and it has reset its inv.
        consumed = global_typed_ptr(arg_counter, T.i32)[0]
        if tid < fx.Int32(npes):
            if tid != fx.Int32(rank):
                comm_ops.spin_until_ge_i32_system(
                    fx.Int64(arg_done) + fx.Int64(tid) * fx.Int64(4),
                    consumed,
                    acquire=True,
                )
        fx.barrier()

        # The GEMM2 kernel's per-chunk completion counters start from zero every call.
        for g in range(gid, i32_cdone_n, stride):
            cdone[g] = fx.Int32(0)

        # Exact offsets from the gathered route counts. Thread pe reads bigcnt row pe
        # (the routes pe dispatched) and, per rank s, the rows pe sent s: s's expert
        # counts plus, for the fanout pair the dispatch picked at s this call, one row per
        # pair member for every token sent once as a group (bigcnt column TE + s).
        # Rows I send pe = the rows pe sent me; they land in pe's R after the rows of
        # the ranks below me other than pe itself.
        cursor_buf = ptr_buf_tensor(arg_cursor, fx.Int32)
        bc = ptr_buf_tensor(arg_bigcnt, fx.Int32)
        parity = fx.Int32(comm_ops.load_i32_system(arg_parity, fx.Int32(0)))
        if tid < fx.Int32(npes):
            row = tid * fx.Int32(total_segments)
            n_send = fx.Int32(0)
            below = fx.Int32(0)
            for dst in range_constexpr(npes):
                if const_expr(dst <= rank):
                    rows = fx.Int32(0)
                    for e in range(fx.Int32(dst * epr), fx.Int32(dst * epr + epr), fx.Int32(1)):
                        rows = rows + fx.Int32(bc[row + e])
                    packed = fx.Int32(
                        comm_ops.load_i32_system(arg_pair_cfg, parity * fx.Int32(npes) + fx.Int32(dst))
                    )
                    enabled = (packed & fx.Int32(1 << 16)) != fx.Int32(0)
                    same = (packed & fx.Int32(0xFF)) == ((packed >> fx.Int32(8)) & fx.Int32(0xFF))
                    members = enabled.select(same.select(fx.Int32(1), fx.Int32(2)), fx.Int32(0))
                    rows = rows + members * fx.Int32(bc[row + fx.Int32(total_segments - npes + dst)])
                    if const_expr(dst == rank):
                        n_send = rows
                    else:
                        below = below + (tid != fx.Int32(dst)).select(rows, fx.Int32(0))
            fx.ptr_store(n_send, lw(fx.Int32(OFF_NSEND) + tid))
            fx.ptr_store(below, lw(fx.Int32(OFF_LAND) + tid))
        fx.barrier()
        if tid == fx.Int32(0):
            sbase = fx.Int32(0)
            for p in range_constexpr(npes):
                n_p = fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_NSEND + p))))
                room = fx.Int32(send_rows) - sbase
                room = (room > fx.Int32(0)).select(room, fx.Int32(0))
                fx.ptr_store(sbase, lw(fx.Int32(OFF_SBASE + p)))
                fx.ptr_store((n_p < room).select(n_p, room), lw(fx.Int32(OFF_AVAIL + p)))
                sbase = sbase + n_p
        fx.barrier()
        if (bid == fx.Int32(0)) & (tid < fx.Int32(npes)):
            cursor_buf[fx.Int32(npes) + tid] = fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_SBASE) + tid)))
            cursor_buf[fx.Int32(2 * npes) + tid] = fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_LAND) + tid)))
            cursor_buf[fx.Int32(3 * npes) + tid] = fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_AVAIL) + tid)))

        lane = tid % fx.Int32(WAVE)
        wave = tid // fx.Int32(WAVE)
        row_ok = tid < fx.Int32(SBM)

        def tile_masks(tile):
            """One thread per row of ``tile``: decode it once, then a ballot per pe
            gives the wave's rows bound for that pe. Returns the row's decode and
            those masks; the per-wave counts are published to LDS and made
            visible (barrier) before returning."""
            base = trb[tile]
            valid, dest_pe, dest_lid, s = _decode(stids[base + row_ok.select(tid, fx.Int32(0))], **dec)
            valid = valid & row_ok
            masks = [
                fx.Int64(rocdl.ballot(T.i64, valid & (dest_pe == fx.Int32(p))))
                for p in range_constexpr(npes)
            ]
            if lane == fx.Int32(0):
                for p in range_constexpr(npes):
                    fx.ptr_store(fx.Int32(fx.math.ctpop(masks[p])), lw(wave * fx.Int32(npes) + fx.Int32(p)))
            fx.barrier()
            return valid, dest_pe, dest_lid, s, masks

        # One block per tile; no thread walks a tile's rows.
        for tile in range(bid, tiles, i32_nblocks):
            tile_masks(tile)
            if tid < fx.Int32(npes):
                total = fx.Int32(0)
                for w in range_constexpr(WAVES):
                    total = total + fx.Int32(fx.ptr_load(lw(fx.Int32(w * npes) + tid)))
                tile_cnt[tile * fx.Int32(npes) + tid] = total
            fx.barrier()
        tree_barrier(arg_bar, i32_nblocks, tid, bid)

        # No rows at all: the loop below never runs, so say so explicitly.
        if (tiles == fx.Int32(0)) & (bid == fx.Int32(0)) & (tid < fx.Int32(npes)):
            tile_base[tid] = fx.Int32(0)
            cursor_buf[tid] = fx.Int32(0)
            if fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_NSEND) + tid))) != fx.Int32(0):
                comm_ops.atomic_add_agent(fx.Int64(arg_cursor) + fx.Int64(16 * npes), fx.Int32(1))

        idx_buf = ptr_buf_tensor(arg_idx, fx.Int32)
        inv_buf = ptr_buf_tensor(arg_inv, fx.Int32)
        lane_lt = (fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1)
        for tile in range(bid, tiles, i32_nblocks):
            # Rows for pe in the tiles before this one. Element e of tile_cnt is
            # (tile e // npes, pe e % npes), and BLOCK % npes == 0, so each thread
            # always sums one pe.
            part = fx.Int32(0)
            for e in range(tid, tile * fx.Int32(npes), fx.Int32(BLOCK)):
                part = part + tile_cnt[e]
            fx.ptr_store(part, lw(fx.Int32(OFF_PART) + tid))
            fx.barrier()
            if tid < fx.Int32(npes):
                pre = fx.Int32(0)
                for k in range_constexpr(GROUPS):
                    pre = pre + fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_PART + k * npes) + tid)))
                fx.ptr_store(pre, lw(fx.Int32(OFF_PREF) + tid))
                avail = fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_AVAIL) + tid)))
                tile_base[tile * fx.Int32(npes) + tid] = (pre < avail).select(pre, avail)
                # The extra row holds the totals the comm waves read as a chunk's end.
                if tile == tiles - fx.Int32(1):
                    total = pre + tile_cnt[tile * fx.Int32(npes) + tid]
                    capped = (total < avail).select(total, avail)
                    tile_base[tiles * fx.Int32(npes) + tid] = capped
                    cursor_buf[tid] = capped
                    # The rows counted here must be the ones the gathered counts promised
                    # (the peers placed my rows in their R from those).
                    if total != fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_NSEND) + tid))):
                        comm_ops.atomic_add_agent(
                            fx.Int64(arg_cursor) + fx.Int64(16 * npes), fx.Int32(1)
                        )
            fx.barrier()

            valid, dest_pe, dest_lid, s, masks = tile_masks(tile)
            g = tile * fx.Int32(SBM) + tid
            if row_ok:
                out_buf[g] = fx.Int32(-1)
            if valid:
                # Rank among this wave's rows for my pe, plus the rows of the
                # earlier waves of the tile for it.
                mine = masks[0]
                for p in range_constexpr(1, npes):
                    mine = (dest_pe == fx.Int32(p)).select(masks[p], mine)
                within = fx.Int32(fx.math.ctpop(mine & lane_lt))
                for w in range_constexpr(WAVES):
                    within = within + (fx.Int32(w) < wave).select(
                        fx.Int32(fx.ptr_load(lw(fx.Int32(w * npes) + dest_pe))), fx.Int32(0)
                    )
                excl = fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_PREF) + dest_pe)))
                place = excl + within
                # Past the end of S there is no room for this row. Never write into
                # the next peer's run: count it, and either drop it (the host asserts
                # on the count) or hand it to GEMM2's direct P2P path.
                if place >= fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_AVAIL) + dest_pe))):
                    if const_expr(p2p_fallback):
                        out_buf[g] = fx.Int32(OUT_ROW_P2P)
                    comm_ops.atomic_add_agent(arg_over, fx.Int32(1))
                else:
                    out_row = fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_SBASE) + dest_pe))) + place
                    out_buf[g] = out_row
                    slot = dest_lid * fx.Int32(topk) + s
                    idx_buf[out_row] = slot
                    if dest_pe == fx.Int32(rank):
                        inv_buf[slot] = out_row
            fx.barrier()

    # fmt: off
    @flyc.jit
    def launch(arg_stids: fx.Int64, arg_trb: fx.Int64, arg_num_valid: fx.Int64,
        arg_tile_cnt: fx.Int64, arg_tile_base: fx.Int64, arg_cursor: fx.Int64,
        arg_out_row: fx.Int64, arg_idx: fx.Int64, arg_inv: fx.Int64, arg_cdone: fx.Int64,
        arg_bar: fx.Int64, arg_over: fx.Int64, arg_done: fx.Int64, arg_counter: fx.Int64,
        arg_bigcnt: fx.Int64, arg_pair_cfg: fx.Int64, arg_parity: fx.Int64,
        i32_nblocks: fx.Int32, i32_cdone_n: fx.Int32, stream: fx.Stream):
    # fmt: on
        plan(arg_stids, arg_trb, arg_num_valid, arg_tile_cnt, arg_tile_base, arg_cursor,
             arg_out_row, arg_idx, arg_inv, arg_cdone, arg_bar, arg_over, arg_done, arg_counter,
             arg_bigcnt, arg_pair_cfg, arg_parity, i32_nblocks, i32_cdone_n).launch(
            grid=(i32_nblocks, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    return launch


# --- Kernel 2: GEMM2 with comm waves ----------------------------------------
#
# GEMM2 whose epilogue writes weighted bf16 rows into a local, rank-sorted send buffer.
#
# Same GEMM2 mainloop and input layout as ``mega_moe_stage2``; only the output side
# differs. Instead of scattering each row into the origin rank's ``comb_inp`` it
# writes the row to ``send[out_row[row]]``, where ``out_row`` was assigned once per
# row beforehand (``build_rank_sort_plan``) so every N
# tile of a row agrees on it. A negative ``out_row`` drops the row.
#
# With ``comm`` the kernel also carries ``comm.comm_wgs`` extra leading workgroups that
# ship finished chunks of the send buffer to the peers with cco SDMA while the GEMM
# workgroups are still computing. A GEMM unit signals ``cdone[chunk]`` after its stores.
# Each comm WAVE owns its own peers (peer ``p`` belongs to wave ``p % n_comm_waves``): it
# spins on the chunk counter itself, then puts that chunk's rows for its peers on its own
# SDMA queue, so peers neither wait for each other's spin nor share a wave's issue slot.
#
# Two things about the publish are load-bearing and were each worth more than a
# factor on this kernel. The epilogue stores the rows write-through (sc0|sc1) so
# that retiring them is the release a copy engine needs; releasing at system scope
# instead is a whole-L2 write-back per unit. And the per-chunk counters sit a
# cache line apart, because every unit of a chunk bumps one.


@dataclass(frozen=True)
class CommSpec:
    """Compile-time description of the fused SDMA comm workgroups.

    ``comm_wgs`` workgroups of 4 waves each; one wave per peer needs
    ``comm_wgs * 4 >= npes``. ``queues`` is the SDMA channels reserved per
    (src, dst) pair; peer ``p`` uses channel ``p % queues`` for everything it
    sends, so a chunk is never split across channels -- splitting measured
    slower, and on this topology KFD recommends one engine per peer link
    anyway, which makes extra channels pure overhead.
    """

    layout: ArenaLayout
    rank: int
    queues: int
    chunk_tiles: int
    comm_wgs: int = 1
    #: How a sender tells a peer its rows have landed; see
    #: ``sdma_finish_peer``. "remote" needs the signal to ride
    #: the channel the data went out on, which ``queues`` already guarantees.
    signal_mode: str = "remote"
    #: Rows past the end of S go to the origin rank's ``comb_inp`` (fp8 rows,
    #: aiter's P2P layout) instead of being dropped; needs the fields below.
    p2p_fallback: bool = False
    topk: int = 0
    max_tok: int = 0
    #: Bytes of one rank's ``comb_inp``: the bound of the direct stores.
    comb_bytes: int = 0


@comm_ops.traced
def _direct_row(
    fb,
    out_row,
    row_byte_off,
    payload,
    e8m0,
    active,
    is_leader,
    lane,
    col,
    n_block_idx,
    *,
    N_OUT,
    BN,
    row_nbytes,
    group_lanes,
    store_cm,
):
    """P2P fallback: if ``out_row`` is ``OUT_ROW_P2P``, store this lane's part of the row
    into ``comb_inp[slot]`` of its origin rank, as aiter's Stage2 does, and point that
    rank's ``inv[slot]`` at it. ``out_row`` is wave-uniform, so the branch is too."""
    import mori.cco.device.flydsl as cco

    if fx.Int32(out_row) == fx.Int32(OUT_ROW_P2P):
        comm = fb["comm"]
        lay, rank = comm.layout, comm.rank
        p = fx.Int32(
            rocdl.readfirstlane(
                T.i32,
                fx.ptr_load(
                    lds_typed_ptr(
                        fx.Int32(fb["lds_src_off"]) + row_byte_off, T.i32, align=4
                    )
                ).ir_value(),
            )
        )
        t = p & fx.Int32(0x00FFFFFF)
        s = p.shrui(fx.Int32(24)) & fx.Int32(0xFF)
        dest_pe = t >> fx.Int32(comm.max_tok.bit_length() - 1)
        slot = (t & fx.Int32(comm.max_tok - 1)) * fx.Int32(comm.topk) + s
        peer = fx.Int64(
            rocdl.readfirstlane(
                T.i64, global_typed_ptr(fb["arg_p2p"], T.i64, align=8)[dest_pe]
            )
        )
        w = cco.Window(fb["arg_win"])
        # Like the comm waves: not before the origin rank has read the last call's
        # comb_inp and inv (its combine publishes done[] to this rank).
        if dest_pe != fx.Int32(rank):
            rsrc_counter = create_buffer_resource_from_addr(
                fx.Int64(w.lsa_ptr(rank, lay.off_counter))
            )
            phase = fx.Int32(
                buffer_load(rsrc_counter, 0, vec_width=1, dtype=T.i32)
            ) + fx.Int32(1)
            sdma_wait_consumed(w, dest_pe, phase, layout=lay, rank=rank)
        # Written through like S, so the unit's s_waitcnt before it publishes its chunk
        # is the release.
        bound = fx.Int32(comm.comb_bytes)
        buf = create_buffer_resource_from_addr(peer, num_records_bytes=comm.comb_bytes)
        base = slot * fx.Int32(row_nbytes)
        off = active.select(base + n_block_idx * fx.Int32(BN) + col, bound)
        buffer_store(payload, buf, off, cache_modifier=store_cm, offset_is_bytes=True)
        s_off = (active & is_leader).select(
            base
            + fx.Int32(N_OUT)
            + n_block_idx * fx.Int32(BN // 32)
            + lane // fx.Int32(group_lanes),
            bound,
        )
        buffer_store(
            e8m0.to(fx.Int8), buf, s_off, cache_modifier=store_cm, offset_is_bytes=True
        )
        if (n_block_idx == fx.Int32(0)) & (lane == fx.Int32(0)):
            comm_ops.store_i32_global_system_monotonic(
                fx.Int64(w.lsa_ptr(dest_pe, lay.off_inv))
                + fx.Int64(slot) * fx.Int64(4),
                fx.Int32(INV_P2P),
            )


# fmt: off
def local_sort_epilog(lds_acc_base, accm, n_block_idx, wave, lane, send_base, *, N_OUT, BM, BN,
    send_nbytes, lds_outrow_off, lds_weight_off, g2_bf16_lds=False, scatter_vec=8, store_cm=0,
    quant="none", fallback=None):
# fmt: on
    """CShuffle one GEMM2 tile into weighted BF16 rows and store them at their send-buffer index.

    ``fallback`` (fp8 only): the P2P fallback's operands; a row whose ``out_row`` is
    ``OUT_ROW_P2P`` is stored into its origin rank's ``comb_inp`` (``_direct_row``).
    """
    kMChunks = BM // 16
    numAccN = (BN // 4) // 16
    wave_n = BN // 4
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)
    lds_base_bf16 = (
        lds_typed_ptr(lds_acc_base, T.bf16, align=2)
        if const_expr(g2_bf16_lds)
        else None
    )
    quant_fp8 = quant == "fp8_blockwise_1x32"
    if quant not in ("none", "fp8_blockwise_1x32"):
        raise ValueError(f"unsupported quant={quant!r}")
    if quant_fp8 and g2_bf16_lds:
        raise ValueError("fp8_blockwise_1x32 needs the f32 CShuffle (g2_bf16_lds=False)")
    # Row layout matches aiter's P2P scatter: N_OUT payload bytes followed by the
    # N_OUT/32 e8m0 block scales, so one contiguous SDMA put ships both.
    row_nbytes = N_OUT + N_OUT // 32 if quant_fp8 else N_OUT * 2

    fx.barrier()

    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * 4
        if const_expr(g2_bf16_lds):
            w_row = [
                fx.ptr_load(
                    lds_typed_ptr(
                        fx.Int32(lds_weight_off) + (row_base + v) * fx.Int32(4),
                        T.f32,
                        align=4,
                    )
                )
                for v in range_constexpr(4)
            ]
        for J in range_constexpr(numAccN):
            col = wave * fx.Int32(wave_n) + J * 16 + lane_mod_16
            vec = fx.Vector(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + v) * BN + col
                if const_expr(g2_bf16_lds):
                    lds_base_bf16[idx] = fx.BFloat16(fx.Float32(vec[v]) * fx.Float32(w_row[v]))
                else:
                    lds_base_fptr[idx] = fx.Float32(vec[v])

    fx.barrier()

    # ``buf_copy_store``'s cache_modifier is a BufferCopy enum (0=cached, 2=nt),
    # so sc0|sc1 cannot be reached through it -- flydsl's own type stub says so,
    # and so does mori/ops/gemm_ar/_compat.py. ``buffer_store`` passes the value
    # through as the instruction's aux field, so the write-through path uses it.
    if const_expr(store_cm or quant_fp8):
        output_buf = create_buffer_resource_from_addr(
            send_base, num_records_bytes=send_nbytes
        )
    else:
        output_buf = ptr_buf_tensor(
            send_base,
            fx.BFloat16,
            unit_elems=scatter_vec,
            num_records_bytes=send_nbytes,
        )
    # Inactive lanes read safe LDS and issue a bounded OOB store.
    active = lane < fx.Int32(BN // scatter_vec)
    col = active.select(lane * fx.Int32(scatter_vec), fx.Int32(0))

    for row_iter in range_constexpr(BM // 4):
        row = wave + fx.Int32(row_iter * 4)
        row_byte_off = row * fx.Int32(4)
        out_row = fx.ptr_load(
            lds_typed_ptr(fx.Int32(lds_outrow_off) + row_byte_off, T.i32, align=4)
        )
        out_row = rocdl.readfirstlane(T.i32, out_row.ir_value())
        idx0 = row * fx.Int32(BN) + col
        if const_expr(g2_bf16_lds):
            pk = fx.Vector(
                lds_vec_load(
                    lds_acc_base,
                    idx0 * fx.Int32(2),
                    fx.Vector.make_type(scatter_vec, fx.BFloat16),
                    fx.BFloat16,
                    align=16,
                )
            )
        else:
            weight = fx.ptr_load(
                lds_typed_ptr(fx.Int32(lds_weight_off) + row_byte_off, T.f32, align=4)
            )
            weight = rocdl.readfirstlane(T.f32, weight.ir_value())
            v8 = fx.Vector(
                lds_vec_load(
                    lds_acc_base,
                    idx0 * fx.Int32(4),
                    fx.Vector.make_type(scatter_vec, fx.Float32),
                    fx.Float32,
                    align=16,
                )
            )
            weighted = fx.Vector.from_elements(
                [v8[i] * weight for i in range_constexpr(scatter_vec)],
                fx.Float32,
            )
            if const_expr(not quant_fp8):
                pk = weighted.to(fx.BFloat16)
        row_base = out_row * fx.Int32(row_nbytes)
        valid = (out_row >= fx.Int32(0)) & active
        if const_expr(quant_fp8):
            # One e8m0 per 32 contiguous values, i.e. per 32//scatter_vec lanes.
            vals = [fx.Float32(weighted[i]) for i in range_constexpr(scatter_vec)]
            local_max = fabs_f32(vals[0])
            for q in range_constexpr(1, scatter_vec):
                local_max = local_max.maximumf(fabs_f32(vals[q]))
            max_bits = local_max.bitcast(fx.Int32)
            for xor_lane in (1, 2):
                if xor_lane < 32 // scatter_vec:
                    remote = rocdl.ds_bpermute(
                        T.i32, (lane ^ fx.Int32(xor_lane)) * fx.Int32(4), max_bits
                    )
                    local_max = local_max.maximumf(fx.Int32(remote).bitcast(fx.Float32))
                    max_bits = local_max.bitcast(fx.Int32)
            group_lanes = 32 // scatter_vec
            leader_lane = lane & fx.Int32(~(group_lanes - 1))
            is_leader = (lane & fx.Int32(group_lanes - 1)) == fx.Int32(0)
            e8m0 = fx.Int32(
                rocdl.ds_bpermute(
                    T.i32,
                    leader_lane * fx.Int32(4),
                    _fp8_scale_for_leader(is_leader, local_max),
                )
            )
            block_scale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
            pk_ty = T.vec(2, T.i16)
            words = []
            for word in range_constexpr(scatter_vec // 4):
                packed = fx.Vector.filled(2, 0, fx.Int16).ir_value()
                for pair in range_constexpr(2):
                    e = word * 4 + pair * 2
                    packed = rocdl.cvt_scalef32_pk_fp8_f32(
                        pk_ty, packed, vals[e].ir_value(), vals[e + 1].ir_value(),
                        block_scale.ir_value(), pair,
                    )
                words.append(fx.Vector(packed).bitcast(fx.Int32)[0])
            payload = fx.Vector.from_elements(words, fx.Int32)
            off = valid.select(
                row_base + n_block_idx * fx.Int32(BN) + col, fx.Int32(send_nbytes)
            )
            buffer_store(
                payload, output_buf, off,
                cache_modifier=store_cm, offset_is_bytes=True,
            )
            # Only the lane that owns a 32-value block writes its scale byte.
            scale_off = (valid & is_leader).select(
                row_base + fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN // 32)
                + lane // fx.Int32(group_lanes),
                fx.Int32(send_nbytes),
            )
            buffer_store(
                e8m0.to(fx.Int8), output_buf, scale_off,
                cache_modifier=store_cm, offset_is_bytes=True,
            )
            if fallback is not None:
                _direct_row(fallback, out_row, row_byte_off, payload, e8m0, active, is_leader,
                            lane, col, n_block_idx, N_OUT=N_OUT, BN=BN, row_nbytes=row_nbytes,
                            group_lanes=group_lanes, store_cm=store_cm)
        else:
            off = valid.select(
                row_base + n_block_idx * fx.Int32(BN * 2) + col * fx.Int32(2),
                fx.Int32(send_nbytes),
            )
            if const_expr(store_cm):
                buffer_store(pk, output_buf, off, cache_modifier=store_cm, offset_is_bytes=True)
            else:
                buf_copy_store(
                    output_buf,
                    off // fx.Int32(scatter_vec * 2),
                    pk,
                    fx.BFloat16,
                    unit_elems=scatter_vec,
                    cache_modifier=0,
                )


# fmt: off
def compile_moe_gemm2_with_sdma(*, model_dim: int, inter_dim: int, experts: int, rank: int,
    send_nbytes: int, BM: int = 32, BN: int = 256, BK: int = 256, use_nt: bool = True,
    HIDDEN_MAX: int = 8192, INTER_MAX: int = 8192, a_dtype: str = "fp8", SBM: int | None = None,
    persist: bool = False, cu_num: int = 0, has_pad: bool = False, g2_bhoist=None,
    g2_ascale_pf=None, g2_spart=None, persist_strided: bool = False, g2_bf16_lds: bool = False,
    skew_cu: int = 0, scatter_vec: int = 8, comm=None, quant: str = "none"):
# fmt: on
    """Compile GEMM2 with the rank-sorted local-output epilogue (optionally fused with SDMA)."""
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx95"):
        raise RuntimeError(f"MegaMoE v2 stage2 requires CDNA4 (gfx95x), got {arch or 'unknown'}")
    assert model_dim % BN == 0 and HIDDEN_MAX % BN == 0
    assert INTER_MAX % BK == 0, f"INTER_MAX must be a multiple of {BK}"
    if BM not in (16, 32, 64, 128):
        raise AssertionError(f"BM must be in {{16,32,64,128}}, got {BM}")
    SBM = BM if SBM is None else int(SBM)
    if SBM % BM != 0:
        raise AssertionError(f"SBM ({SBM}) must be a multiple of BM ({BM})")
    if scatter_vec not in (8, 16):
        raise ValueError("Stage2 scatter_vec must be 8 or 16")
    if BN % scatter_vec:
        raise ValueError("Stage2 scatter_vec must divide BN")
    if a_dtype not in ("fp4", "fp8"):
        raise AssertionError(f"a_dtype must be 'fp4' or 'fp8', got {a_dtype!r}")
    if persist and cu_num <= 0:
        raise AssertionError(f"persist=True requires cu_num>0, got {cu_num}")
    if skew_cu and (not persist or not 0 < skew_cu <= cu_num):
        raise AssertionError(f"skew_cu={skew_cu} requires persist=True and 0<skew_cu<=cu_num={cu_num}")
    if not 0 < send_nbytes < _BUFFER_OFFSET_ABI_BYTES:
        raise ValueError("MegaMoE v2 stage2 send buffer exceeds the 32-bit buffer-resource ABI")
    fuse = comm is not None
    NC = comm.comm_wgs if fuse else 0
    # comm_wgs == 0 is a diagnostic build: the GEMM units still publish each chunk
    # (s_waitcnt + barrier + atomic), but no comm wave is traced and nothing ships.
    if fuse and (comm.comm_wgs < 1 or comm.chunk_tiles < 1):
        raise ValueError("fused comm needs comm_wgs >= 1 and chunk_tiles >= 1")
    # The rows are stored sc0|sc1 -- past L1 (GLC) and past L2 (SCC) -- so they
    # are in memory when the store retires and the ``s_waitcnt vmcnt(0)`` in
    # front of the publish is the release the copy engine needs. Leaving them in
    # L2 and issuing a system-scope release instead costs a whole-L2 write-back
    # per unit (``buffer_wbl2`` on gfx950): measured 2.4x on the GEMM2 kernel.
    # See mori/ops/gemm_ar/_compat.py for the encoding, and why sc1 is 16 not 4.
    STORE_CM = 1 | 16 if fuse else 0
    fallback = fuse and comm.p2p_fallback
    if fallback and quant != "fp8_blockwise_1x32":
        raise ValueError("the P2P fallback writes aiter's fp8 comb_inp rows: needs quant=fp8_blockwise_1x32")
    N_OUT = model_dim
    g2_bhoist, g2_ascale_pf, g2_spart, g2_group_num, g2_m01, _g2_bf16_lds = _resolve_g2_knobs(
        g2_bhoist, g2_ascale_pf, g2_spart, False, False
    )
    is_f8 = a_dtype == "fp8"
    aStages = kStages + 1
    KH_TILE_A = BK // (1 if is_f8 else 2)
    compute_lds_bytes = _stage2_lds_bytes(BM, BN, BK, a_dtype, aStages, g2_bf16_lds)
    lds_outrow_off = compute_lds_bytes
    lds_weight_off = lds_outrow_off + BM * 4
    # The P2P fallback also needs each row's srcmap word (origin rank and slot).
    lds_src_off = lds_weight_off + BM * 4
    lds_bytes = lds_src_off + (BM * 4 if fallback else 0)
    if lds_bytes > 160 * 1024:
        raise ValueError(f"Stage2 LDS use {lds_bytes} exceeds 160 KiB")
    _expert_offset = rank * experts

    @fx.struct
    class SharedStorage:
        buf: fx.Array[Int8, lds_bytes, 16]

    kernel_name = (
        f"megamoe_stage2_with_sdma_t{BM}x{BN}x{BK}"
        f"_sbm{SBM}_{a_dtype}_nt{int(use_nt)}"
        f"_p{int(persist)}cu{cu_num}s{int(persist_strided)}_pad{int(has_pad)}"
        f"_sk{skew_cu}"
        f"_bh{int(g2_bhoist)}apf{int(g2_ascale_pf)}sp{g2_group_num}x{g2_m01}"
        f"_bf16lds{int(g2_bf16_lds)}_sv{scatter_vec}_{quant}"
        + (
            f"_fuse{comm.chunk_tiles}q{comm.queues}w{comm.comm_wgs}"
            f"sm{comm.signal_mode[0]}r{comm.rank}f{int(comm.p2p_fallback)}"
            if fuse
            else ""
        )
    )

    # fmt: off
    @flyc.kernel(name=kernel_name, known_block_size=[256, 1, 1])
    def kernel_with_sdma(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_max_expert_tiles: fx.Int64,
        arg_sweights: fx.Int64, arg_trb: fx.Int64, arg_send: fx.Int64, arg_out_row: fx.Int64,
        arg_dev_comm: fx.Int64, arg_win: fx.Int64, arg_cdone: fx.Int64, arg_tile_base: fx.Int64,
        arg_cursor: fx.Int64, arg_idx: fx.Int64, arg_stids: fx.Int64, arg_p2p: fx.Int64,
        i32_max_m_blocks: fx.Int32, i32_inter: fx.Int32, i32_hidden: fx.Int32, i32_kpad: fx.Int32,
        i32_npad: fx.Int32):
    # fmt: on
        tx_i32 = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))

        num_n_blocks = fx.Int32(i32_hidden) // fx.Int32(BN)
        k_bytes = fx.Int32(i32_inter) // fx.Int32(1 if is_f8 else 2)
        trb_buf = ptr_buf_tensor(arg_trb, fx.Int32)
        sweights_buf = ptr_buf_tensor(arg_sweights, fx.Float32)
        outrow_buf = ptr_buf_tensor(arg_out_row, fx.Int32)
        stids_buf = ptr_buf_tensor(arg_stids, fx.Int32)
        fb = (
            dict(comm=comm, lds_src_off=lds_src_off, arg_p2p=arg_p2p, arg_win=arg_win)
            if fallback
            else None
        )

        def issue_all_a_loads(m_row0):
            for slot in range_constexpr(kStages):
                issue_a_load_lds_dt(arg_aq, lds_base_i32, slot, slot, m_row0, wave, lane,
                    is_f8, KH_TILE_A, k_bytes, BM=BM)

        def run_unit(unit_bx, m_block_idx):
            # Map each Stage2 BM sub-tile to its Stage1 SBM metadata row.
            m_row = m_block_idx * fx.Int32(BM)
            sort_block_idx = m_row // fx.Int32(SBM)
            row_in_sort_block = m_row - sort_block_idx * fx.Int32(SBM)
            srcmap_row_base = trb_buf[sort_block_idx] + row_in_sort_block
            if tx_i32 < fx.Int32(BM):
                sorted_pos = srcmap_row_base + tx_i32
                fx.ptr_store(
                    outrow_buf[m_row + tx_i32],
                    lds_typed_ptr(
                        fx.Int32(lds_outrow_off) + tx_i32 * fx.Int32(4),
                        T.i32,
                        align=4,
                    ),
                )
                fx.ptr_store(
                    sweights_buf[sorted_pos],
                    lds_typed_ptr(
                        fx.Int32(lds_weight_off) + tx_i32 * fx.Int32(4),
                        T.f32,
                        align=4,
                    ),
                )
                if const_expr(fallback):
                    fx.ptr_store(
                        stids_buf[sorted_pos],
                        lds_typed_ptr(
                            fx.Int32(lds_src_off) + tx_i32 * fx.Int32(4),
                            T.i32,
                            align=4,
                        ),
                    )
            # fmt: off
            accm_vecs, m_row, n_block_idx, _n_out_rt = gemm2_compute_v2(lds_base_i32, arg_ascale, arg_bq,
                arg_bscale, arg_eids, arg_aq, i32_max_m_blocks, unit_bx, lane, wave, i32_inter, i32_hidden,
                i32_kpad, i32_npad, BM=BM, BN=BN, BK=BK, use_nt=use_nt, INTER_MAX=INTER_MAX, aStages=aStages,
                a_dtype=a_dtype, has_pad=has_pad, SBM=SBM, g2_bhoist=g2_bhoist, g2_ascale_pf=g2_ascale_pf,
                expert_offset=_expert_offset)
            send_base = rocdl.readfirstlane(T.i64, fx.Int64(arg_send).ir_value())
            local_sort_epilog(lds_base_i32, accm_vecs, n_block_idx, wave, lane, send_base,
                N_OUT=N_OUT, BM=BM, BN=BN, send_nbytes=send_nbytes, lds_outrow_off=lds_outrow_off,
                lds_weight_off=lds_weight_off, g2_bf16_lds=g2_bf16_lds, scatter_vec=scatter_vec,
                store_cm=STORE_CM, quant=quant, fallback=fb)
            # fmt: on

        def run_one(unit_bx, m_block_idx):
            issue_all_a_loads(m_block_idx * fx.Int32(BM))
            rocdl.sched_barrier(0)
            run_unit(unit_bx, m_block_idx)
            if const_expr(fuse):
                # The rows went past L2 on the way out, so retiring the stores
                # is all the release the copy engine needs; the barrier makes it
                # the whole workgroup's stores, not just this wave's.
                rocdl.s_waitcnt(vmcnt=0)
                fx.barrier()
                if tx_i32 == fx.Int32(0):
                    chunk = (m_block_idx * fx.Int32(BM)) // fx.Int32(SBM * comm.chunk_tiles)
                    comm_ops.atomic_add_agent(
                        fx.Int64(arg_cdone) + fx.Int64(chunk) * fx.Int64(CDONE_STRIDE * 4),
                        fx.Int32(1),
                    )

        cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
        total_m_blocks = (cumsum0 + fx.Int32(BM - 1)) // fx.Int32(BM)

        def gemm_role(bx_i32):

            if const_expr(not persist and g2_spart <= 0):
                bound = total_m_blocks * fx.Int32(num_n_blocks)
                if fx.Int32(bx_i32) < bound:
                    run_one(bx_i32, bx_i32 // num_n_blocks)
            elif const_expr(not persist):
                bound = total_m_blocks * fx.Int32(num_n_blocks)
                if fx.Int32(bx_i32) < bound:
                    m_block_idx, n_block_idx = _spart_output_tile_index(
                        bx_i32, total_m_blocks, num_n_blocks, g2_group_num, g2_m01
                    )
                    unit_bx = m_block_idx * fx.Int32(num_n_blocks) + n_block_idx
                    run_one(unit_bx, m_block_idx)
            elif const_expr(skew_cu > 0):
                total_stage1_tiles = (cumsum0 + fx.Int32(SBM - 1)) // fx.Int32(SBM)
                max_expert_tiles = global_typed_ptr(arg_max_expert_tiles, T.i32)[0]
                skewed = max_expert_tiles * fx.Int32(4) > total_stage1_tiles
                skew_n_block = bx_i32 // fx.Int32(cu_num)
                skew_m_slot = bx_i32 - skew_n_block * fx.Int32(cu_num)
                normal_m_slot = bx_i32 // fx.Int32(num_n_blocks)
                normal_n_block = bx_i32 - normal_m_slot * fx.Int32(num_n_blocks)
                n_block = skewed.select(skew_n_block, normal_n_block)
                m_slot = skewed.select(skew_m_slot, normal_m_slot)
                active_cu = skewed.select(fx.Int32(skew_cu), fx.Int32(cu_num))
                strided_diff = total_m_blocks - m_slot
                strided_rem = (strided_diff > fx.Int32(0)).select(strided_diff, fx.Int32(0))
                strided_iters = (strided_rem + active_cu - fx.Int32(1)) // active_cu
                tiles_per_slot = (
                    total_m_blocks + active_cu - fx.Int32(1)
                ) // active_cu
                m_tile0 = m_slot * tiles_per_slot
                contiguous_diff = total_m_blocks - m_tile0
                contiguous_rem = (contiguous_diff > fx.Int32(0)).select(
                    contiguous_diff, fx.Int32(0)
                )
                contiguous_iters = (contiguous_rem < tiles_per_slot).select(
                    contiguous_rem, tiles_per_slot
                )
                n_iters = skewed.select(strided_iters, contiguous_iters)
                active = m_slot < active_cu
                for _it in range(fx.Int32(0), n_iters, fx.Int32(1)):
                    strided_m = m_slot + fx.Int32(_it) * active_cu
                    contiguous_m = m_tile0 + fx.Int32(_it)
                    m_block = skewed.select(strided_m, contiguous_m)
                    if active:
                        unit_bx = m_block * fx.Int32(num_n_blocks) + n_block
                        fx.barrier()
                        if m_block < total_m_blocks:
                            run_one(unit_bx, m_block)
            else:
                m_slot = bx_i32 // fx.Int32(num_n_blocks)
                n_block = bx_i32 - m_slot * fx.Int32(num_n_blocks)
                if const_expr(persist_strided):
                    diff = total_m_blocks - m_slot
                    rem = (diff > fx.Int32(0)).select(diff, fx.Int32(0))
                    n_iters = (rem + fx.Int32(cu_num - 1)) // fx.Int32(cu_num)
                else:
                    tiles_per_slot = (
                        total_m_blocks + fx.Int32(cu_num - 1)
                    ) // fx.Int32(cu_num)
                    m_tile0 = m_slot * tiles_per_slot
                    diff = total_m_blocks - m_tile0
                    rem = (diff > fx.Int32(0)).select(diff, fx.Int32(0))
                    n_iters = (rem < tiles_per_slot).select(rem, tiles_per_slot)
                for _it in range(fx.Int32(0), n_iters, fx.Int32(1)):
                    if const_expr(persist_strided):
                        m_block = m_slot + fx.Int32(_it) * fx.Int32(cu_num)
                    else:
                        m_block = m_tile0 + fx.Int32(_it)
                    unit_bx = m_block * fx.Int32(num_n_blocks) + n_block
                    fx.barrier()  # separate prev-iter epilog LDS reads from this iter's A-load into the LDS union
                    if fx.Int32(m_block) < total_m_blocks:
                        run_one(unit_bx, m_block)

        def comm_role(cw):
            import mori.cco.device.flydsl as cco

            lay, rank, npes, CT = comm.layout, comm.rank, comm.layout.npes, comm.chunk_tiles
            n_waves = NC * 4
            n_mine = (npes + n_waves - 1) // n_waves
            mb_per_tile = SBM // BM
            # This wave owns peers owner, owner + n_waves, ...; a wave past the last
            # peer has nothing to do.
            owner = cw * fx.Int32(4) + wave
            is_lane0 = lane == fx.Int32(0)
            my_peers = [owner + fx.Int32(j * n_waves) for j in range_constexpr(n_mine)]

            def sends_to(peer):
                return (peer < fx.Int32(npes)) & (peer != fx.Int32(rank))

            def queue_of(peer):
                return peer % fx.Int32(comm.queues)

            if owner < fx.Int32(npes):
                w = cco.Window(arg_win)
                sdma = cco.DevComm(arg_dev_comm).sdma()
                rsrc_counter = create_buffer_resource_from_addr(
                    fx.Int64(w.lsa_ptr(rank, lay.off_counter))
                )
                # Read-only here: the combine kernel advances the counter after the call,
                # so every comm wave sees the same phase whenever it starts.
                phase = fx.Int32(buffer_load(rsrc_counter, 0, vec_width=1, dtype=T.i32)) + fx.Int32(1)
                cursor_buf = ptr_buf_tensor(arg_cursor, fx.Int32)
                tile_base_buf = ptr_buf_tensor(arg_tile_base, fx.Int32)
                idx_buf = ptr_buf_tensor(arg_idx, fx.Int32)

                def run_offsets(peer):
                    """The plan's exact offsets: my run's base in S and where it lands in
                    ``peer``'s R (wave-uniform)."""
                    run0 = fx.Int32(rocdl.readfirstlane(T.i32, cursor_buf[fx.Int32(npes) + peer]))
                    land_run0 = fx.Int32(
                        rocdl.readfirstlane(T.i32, cursor_buf[fx.Int32(2 * npes) + peer])
                    )
                    return run0, land_run0

                def push_inv(peer):
                    """Fill ``peer``'s inv for the rows I send it: slot -> where the row lands
                    there (its R rows follow its S rows). All 64 lanes work, PUSH_U rows each
                    per trip, so that many idx loads are in flight before their stores."""
                    cnt = cursor_buf[peer]
                    run0, land_run0 = run_offsets(peer)
                    inv_peer = fx.Int64(w.lsa_ptr(peer, lay.off_inv))
                    land0 = fx.Int32(lay.send_rows) + land_run0
                    for base in range(fx.Int32(0), cnt, fx.Int32(64 * PUSH_U)):
                        ok, js, slots = [], [], []
                        for u in range_constexpr(PUSH_U):
                            j = base + fx.Int32(u * 64) + lane
                            in_run = j < cnt
                            ok.append(in_run)
                            js.append(j)
                            slots.append(
                                idx_buf[run0 + in_run.select(j, fx.Int32(0))]
                            )
                        for u in range_constexpr(PUSH_U):
                            if ok[u]:
                                comm_ops.store_i32_global_system_monotonic(
                                    inv_peer + fx.Int64(slots[u]) * fx.Int64(4), land0 + js[u]
                                )

                # Do not overwrite a peer's landing buffer or inv until it has read the last call's.
                for peer in my_peers:
                    if sends_to(peer):
                        if is_lane0:
                            sdma_wait_consumed(w, peer, phase, layout=lay, rank=rank)
                # The inv pushes run while the GEMM computes its first chunk. They have to be
                # performed before any put or signal of mine, hence the wait on vmcnt.
                for peer in my_peers:
                    if sends_to(peer):
                        push_inv(peer)
                rocdl.s_waitcnt(vmcnt=0)

                # Read once, outside every runtime branch (a peer past npes reads pe 0's).
                run_offs = [
                    run_offsets((peer < fx.Int32(npes)).select(peer, fx.Int32(0)))
                    for peer in my_peers
                ]
                total_tiles = (cumsum0 + fx.Int32(SBM - 1)) // fx.Int32(SBM)
                n_chunks = (total_tiles + fx.Int32(CT - 1)) // fx.Int32(CT)
                last = n_chunks - fx.Int32(1)
                # Under "remote" the last chunk's put carries the completion signal.
                sig_on_put = comm.signal_mode == "remote"
                for c in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                    ci = fx.Int32(c)
                    m_lo = ci * fx.Int32(CT * mb_per_tile)
                    m_hi = m_lo + fx.Int32(CT * mb_per_tile)
                    m_hi = (m_hi < total_m_blocks).select(m_hi, total_m_blocks)
                    # Every (m_block, n_block) unit of this chunk has stored.
                    if is_lane0:
                        comm_ops.spin_until_ge_i32_agent(
                            fx.Int64(arg_cdone) + fx.Int64(ci) * fx.Int64(CDONE_STRIDE * 4),
                            (m_hi - m_lo) * num_n_blocks,
                        )
                    comm_ops.fence_agent_acquire()
                    t0 = ci * fx.Int32(CT)
                    t1 = t0 + fx.Int32(CT)
                    t1 = (t1 < total_tiles).select(t1, total_tiles)
                    for peer, (run0, land_run0) in zip(my_peers, run_offs):
                        if sends_to(peer):
                            lo = tile_base_buf[t0 * fx.Int32(npes) + peer]
                            hi = tile_base_buf[t1 * fx.Int32(npes) + peer]
                            if hi > lo:
                                if is_lane0:
                                    if const_expr(sig_on_put):
                                        if ci == last:
                                            sdma_put_run(sdma, arg_win, peer, queue_of(peer),
                                                         layout=lay, send_base=run0, land_off=land_run0, lo=lo, hi=hi,
                                                         cco=cco, remote_signal=True)
                                        else:
                                            sdma_put_run(sdma, arg_win, peer, queue_of(peer),
                                                         layout=lay, send_base=run0, land_off=land_run0, lo=lo, hi=hi,
                                                         cco=cco)
                                    else:
                                        sdma_put_run(sdma, arg_win, peer, queue_of(peer),
                                                     layout=lay, send_base=run0, land_off=land_run0, lo=lo, hi=hi, cco=cco)
                for peer in my_peers:
                    if sends_to(peer):
                        if is_lane0:
                            if const_expr(sig_on_put):
                                # Did the last chunk's put carry the signal? Not if that
                                # chunk was empty for this peer, or there were no chunks.
                                t_last = last * fx.Int32(CT)
                                t_last = (t_last > fx.Int32(0)).select(t_last, fx.Int32(0))
                                lo_l = tile_base_buf[t_last * fx.Int32(npes) + peer]
                                hi_l = tile_base_buf[total_tiles * fx.Int32(npes) + peer]
                                if (n_chunks > fx.Int32(0)) & (hi_l > lo_l):
                                    sdma_finish_peer(sdma, w, arg_win, peer, queue_of(peer),
                                                     phase, layout=lay, rank=rank, cco=cco,
                                                     signal_mode=comm.signal_mode,
                                                     signalled=True)
                                else:
                                    sdma_finish_peer(sdma, w, arg_win, peer, queue_of(peer),
                                                     phase, layout=lay, rank=rank, cco=cco,
                                                     signal_mode=comm.signal_mode)
                            else:
                                sdma_finish_peer(sdma, w, arg_win, peer, queue_of(peer), phase,
                                                 layout=lay, rank=rank, cco=cco,
                                                 signal_mode=comm.signal_mode)

        if const_expr(fuse):
            if bid < fx.Int32(NC):
                comm_role(bid)
            else:
                gemm_role(bid - fx.Int32(NC))
        else:
            gemm_role(bid)

    # fmt: off
    @flyc.jit
    def launch(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_max_expert_tiles: fx.Int64,
        arg_sweights: fx.Int64, arg_trb: fx.Int64, arg_send: fx.Int64, arg_out_row: fx.Int64,
        arg_dev_comm: fx.Int64, arg_win: fx.Int64, arg_cdone: fx.Int64, arg_tile_base: fx.Int64,
        arg_cursor: fx.Int64, arg_idx: fx.Int64, arg_stids: fx.Int64, arg_p2p: fx.Int64,
        i32_max_m_blocks: fx.Int32, i32_grid_blocks: fx.Int32,
        i32_inter: fx.Int32, i32_hidden: fx.Int32, i32_kpad: fx.Int32, i32_npad: fx.Int32,
        stream: fx.Stream):
    # fmt: on
        num_n_blocks = fx.Int32(i32_hidden) // fx.Int32(BN)
        grid_x = i32_grid_blocks * num_n_blocks + fx.Int32(NC)
        kernel_with_sdma(
            arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum, arg_max_expert_tiles,
            arg_sweights, arg_trb, arg_send, arg_out_row, arg_dev_comm, arg_win, arg_cdone,
            arg_tile_base, arg_cursor, arg_idx, arg_stids, arg_p2p, i32_max_m_blocks, i32_inter, i32_hidden,
            i32_kpad, i32_npad,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


_LAUNCH_CACHE = {}


# fmt: off
def run_moe_gemm2_with_sdma(arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum,
    arg_max_expert_tiles, arg_sweights, arg_trb, arg_send, arg_out_row, arg_dev_comm, arg_win,
    arg_cdone, arg_tile_base, arg_cursor, arg_idx, arg_stids, arg_p2p, row_capacity, i32_inter,
    i32_hidden, stream, *,
    model_dim, inter_dim, experts, rank, send_nbytes, BM, SBM, HIDDEN_MAX, INTER_MAX, cu_num,
    BN=256, BK=256, use_nt=True, g2_bhoist=True, g2_ascale_pf=True, g2_spart=402, persist=False,
    persist_cu=0, persist_strided=False, g2_bf16_lds=False, skew_cu=0, scatter_vec=8, comm=None,
    quant="none"):
    # fmt: on
    """Compile or reuse one configuration and launch it."""
    if comm is not None and persist and 0 < persist_cu < cu_num:
        # A comm workgroup holds one of its CU's two workgroup slots for the whole
        # kernel: ktrace shows those CUs run 8 GEMM workgroups where the rest run
        # 15-18. The displaced work has to go somewhere, and at the bucket's tuned
        # persist_cu=192 one workgroup is ~13% of the kernel span, so the tail
        # quantises badly -- the last CU finishes 3.3% late while every workgroup
        # runs at its normal speed. More, shorter workgroups absorb it: the fused
        # kernel's cost over plain GEMM2 drops a third at 16384 tokens per rank
        # (V4.1-Flash 0.200 -> 0.133 ms, V4-Pro 0.274 -> 0.184 ms).
        persist_cu = cu_num
    launch_cu_num = min(cu_num, persist_cu) if persist and persist_cu > 0 else cu_num
    compile_kw = dict(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, rank=rank,
        send_nbytes=send_nbytes, BM=BM, BN=BN, BK=BK, use_nt=use_nt, HIDDEN_MAX=HIDDEN_MAX,
        INTER_MAX=INTER_MAX, SBM=SBM, persist=persist, cu_num=launch_cu_num,
        g2_bhoist=g2_bhoist, g2_ascale_pf=g2_ascale_pf, g2_spart=g2_spart,
        persist_strided=persist_strided, g2_bf16_lds=g2_bf16_lds, skew_cu=skew_cu,
        scatter_vec=scatter_vec, comm=comm, quant=quant,
    )
    key = tuple(sorted(compile_kw.items()))
    launch = _LAUNCH_CACHE.get(key)
    max_m_blocks = (row_capacity + BM - 1) // BM
    grid_blocks = launch_cu_num if persist else max_m_blocks
    run_args = (
        arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum,
        arg_max_expert_tiles, arg_sweights, arg_trb, arg_send, arg_out_row, arg_dev_comm, arg_win,
        arg_cdone, arg_tile_base, arg_cursor, arg_idx, arg_stids, arg_p2p, fx.Int32(max_m_blocks),
        fx.Int32(grid_blocks),
        fx.Int32(i32_inter), fx.Int32(i32_hidden), fx.Int32(0), fx.Int32(0), stream,
    )
    if launch is None:
        # The comm role's live ranges lift this function's register pressure over the
        # threshold that turns on the AMDGPU UnclusteredHighRPReschedule stage, and
        # that stage then breaks the GEMM main loop's clustering: the MFMAs move 23
        # instructions earlier and the mean load -> s_waitcnt distance falls from
        # 20.9 to 13.6, so the loop hides less latency. Same instruction multiset,
        # worse order, and the loop pays it on every trip -- 8.5% of GEMM2 at 16384
        # tokens per rank, present even when the comm role never runs. The register
        # allocation is fine either way: 13 SGPR spills, all in the prologue, all
        # filling s_nop slots.
        #
        # FlyDSL emits code on the first launch, not in the call below, so the option
        # has to stay set across that launch as well.
        with llvm_options({"amdgpu-disable-unclustered-high-rp-reschedule": True}):
            launch = _LAUNCH_CACHE[key] = compile_moe_gemm2_with_sdma(**compile_kw)
            _run_compiled(launch, *run_args)
        return
    _run_compiled(launch, *run_args)


# --- Kernel 3: combine --------------------------------------------------------

def build_combine(
    *, layout, rank, hidden, topk, quant="none", signal_mode="remote", p2p_fallback=False
):
    """Kernel 3: finish the transfer and reduce.

    One launch, in phases:

      A. wait for every peer's flag for this call (acquire);
      B. ``out[t] = sum_k row[inv[t*topk+k]]`` in f32, one block per token in turn. A slot's
         ``inv`` entry is reset to -1 once read, so it starts clean next call and nothing
         has to refill it; slots no route filled stay -1 and contribute zero;
      C. the last block to finish publishes ``done`` to every peer and advances ``counter``.

    ``p2p_fallback``: a slot whose ``inv`` is ``INV_P2P`` is read from this rank's
    ``comb_inp[slot]`` (``arg_comb_inp``), where GEMM2 put a row past the end of ``S``.
    """
    import mori.cco.device.flydsl as cco

    if hidden % 8:
        raise ValueError(f"hidden must be a multiple of 8, got {hidden}")
    quant_fp8 = quant == "fp8_blockwise_1x32"
    if quant not in ("none", "fp8_blockwise_1x32"):
        raise ValueError(f"unsupported quant={quant!r}")
    if quant_fp8 and hidden % 32:
        raise ValueError(f"fp8_blockwise_1x32 needs hidden % 32 == 0, got {hidden}")
    if p2p_fallback and not quant_fp8:
        raise ValueError("the P2P fallback reads aiter's fp8 comb_inp rows: needs fp8_blockwise_1x32")
    npes = layout.npes
    units = hidden // 8
    chunks = (units + BLOCK - 1) // BLOCK
    row_bytes = layout.row_bytes

    # fmt: off
    @flyc.kernel(name=f"mega_combine_p{npes}_s{layout.send_rows}r{layout.recv_rows}_h{hidden}_k{topk}_{quant}_{signal_mode}"
                 f"_f{int(p2p_fallback)}_r{rank}", known_block_size=[BLOCK, 1, 1])
    def combine(win: fx.Int64, arg_inv: fx.Int64, arg_out: fx.Int64, arg_comb_inp: fx.Int64,
        arg_fin: fx.Int64, arg_dev_comm: fx.Int64,
        i32_nblocks: fx.Int32, i32_tokens: fx.Int32):
    # fmt: on
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        w = cco.Window(win)
        rsrc_counter = create_buffer_resource_from_addr(
            _uniform_i64(w.lsa_ptr(rank, layout.off_counter))
        )
        # Advanced only by the last block below, after every block has read it.
        phase = fx.Int32(buffer_load(rsrc_counter, 0, vec_width=1, dtype=T.i32)) + fx.Int32(1)
        flags_base = fx.Int64(w.lsa_ptr(rank, layout.off_flags))

        # A. the peers' data has landed
        if tid < fx.Int32(npes):
            if tid != fx.Int32(rank):
                # The flag is the call number (and, under "quiet", data-readiness too).
                comm_ops.spin_until_ge_i32_system(
                    flags_base + fx.Int64(tid) * fx.Int64(4), phase, acquire=True
                )
                if const_expr(signal_mode == "remote"):
                    # The sender did not drain its queue; its copy engine bumps
                    # this counter once per call when the last put lands. One
                    # signal per (src, queue) per call, so the running total is
                    # the call number.
                    cco.DevComm(arg_dev_comm).sdma().wait_signal(
                        tid, 0, fx.Int64(phase)
                    )
                    comm_ops.fence_system_acquire()
        fx.barrier()

        inv_buf = ptr_buf_tensor(arg_inv, fx.Int32)

        # B. top-k reduce
        s_base = _uniform_i64(w.lsa_ptr(rank, layout.off_s))
        if const_expr(quant_fp8):
            # Payload is 8 fp8 bytes (2 i32) per thread; the 32-value block they
            # fall in is u // 4, so each thread reads its own scale byte rather
            # than electing a leader and broadcasting.
            rows = ptr_buf_tensor(s_base, fx.Int32, unit_elems=2)
            scales = ptr_buf_tensor(s_base, fx.Int8)
            if const_expr(p2p_fallback):
                c_base = _uniform_i64(arg_comb_inp)
                c_rows = ptr_buf_tensor(c_base, fx.Int32, unit_elems=2)
                c_scales = ptr_buf_tensor(c_base, fx.Int8)
        else:
            rows = ptr_buf_tensor(s_base, fx.BFloat16, unit_elems=8)
        out = ptr_buf_tensor(arg_out, fx.BFloat16, unit_elems=8)
        v2f32 = T.vec(2, T.f32)

        def fp8_row(rows_t, scales_t, row0, u):
            """8 dequantised values of unit ``u`` of the fp8 row at byte ``row0``, and
            their block scale."""
            raw = fx.Vector(
                buf_copy_load(
                    rows_t, (row0 + u * fx.Int32(8)) // fx.Int32(8),
                    fx.Int32, unit_elems=2,
                )
            )
            sc = scales_t[row0 + fx.Int32(hidden) + u // fx.Int32(4)]
            scale = (
                fx.Uint8(sc).to(fx.Uint32).bitcast(fx.Int32)
                << fx.Int32(23)
            ).bitcast(fx.Float32)
            parts = []
            for word in range_constexpr(2):
                lo = cvt_pk_f32_fp8(res=v2f32, src=raw[word], word_sel=False)
                hi = cvt_pk_f32_fp8(res=v2f32, src=raw[word], word_sel=True)
                parts.append(lo.shuffle(hi, [0, 1, 2, 3]))
            vals = fx.Vector(parts[0]).shuffle(
                fx.Vector(parts[1]), [0, 1, 2, 3, 4, 5, 6, 7]
            )
            return fx.Vector(vals), scale
        for tok in range(bid, i32_tokens, i32_nblocks):
            slots = []
            for k in range_constexpr(topk):
                slots.append(inv_buf[tok * fx.Int32(topk) + fx.Int32(k)])
            for chunk in range_constexpr(chunks):
                u = fx.Int32(chunk * BLOCK) + tid
                if u < fx.Int32(units):
                    acc = fx.Vector.filled(8, 0.0, fx.Float32)
                    for k in range_constexpr(topk):
                        ok = slots[k] >= fx.Int32(0)
                        safe = ok.select(slots[k], fx.Int32(0))
                        keep = ok.select(fx.Float32(1.0), fx.Float32(0.0))
                        if const_expr(quant_fp8):
                            vals, scale = fp8_row(rows, scales, safe * fx.Int32(row_bytes), u)
                            acc = acc + vals * (scale * keep)
                        else:
                            raw = buf_copy_load(
                                rows, safe * fx.Int32(units) + u, fx.BFloat16, unit_elems=8
                            )
                            acc = acc + fx.Vector(raw).to(fx.Float32) * keep
                    if const_expr(p2p_fallback):
                        # Rare: rows past the end of S, in comb_inp[slot]. The
                        # slots are the same for the whole block, so this branch is uniform.
                        direct = [slots[k] == fx.Int32(INV_P2P) for k in range_constexpr(topk)]
                        any_direct = direct[0]
                        for k in range_constexpr(1, topk):
                            any_direct = any_direct | direct[k]
                        if any_direct:
                            for k in range_constexpr(topk):
                                slot = direct[k].select(
                                    tok * fx.Int32(topk) + fx.Int32(k), fx.Int32(0)
                                )
                                pvals, pscale = fp8_row(c_rows, c_scales, slot * fx.Int32(row_bytes), u)
                                keep = direct[k].select(fx.Float32(1.0), fx.Float32(0.0))
                                acc = acc + pvals * (pscale * keep)
                    buf_copy_store(
                        out, tok * fx.Int32(units) + u, acc.to(fx.BFloat16), fx.BFloat16, unit_elems=8
                    )
            # Every thread has the slots in registers; only then may they be reset.
            fx.barrier()
            if tid < fx.Int32(topk):
                inv_buf[tok * fx.Int32(topk) + tid] = fx.Int32(-1)

        # C. the last block out tells the peers this call's data has been consumed
        rocdl.s_waitcnt(0)
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.fence_agent_release()
            old = fx.Int32(comm_ops.atomic_add_agent(arg_fin, fx.Int32(1)))
            if (old + fx.Int32(1)) % i32_nblocks == fx.Int32(0):
                comm_ops.fence_agent_acquire()
                buffer_store(phase, rsrc_counter, 0)
                comm_ops.fence_system_release()
                for peer in range_constexpr(npes):
                    if peer != rank:
                        comm_ops.store_i32_global_system_release(
                            fx.Int64(w.lsa_ptr(peer, layout.off_done)) + fx.Int64(rank * 4), phase
                        )

    # fmt: off
    @flyc.jit
    def launch(win: fx.Int64, arg_inv: fx.Int64, arg_out: fx.Int64, arg_comb_inp: fx.Int64,
        arg_fin: fx.Int64, arg_dev_comm: fx.Int64,
        i32_nblocks: fx.Int32, i32_tokens: fx.Int32, stream: fx.Stream):
    # fmt: on
        combine(win, arg_inv, arg_out, arg_comb_inp, arg_fin, arg_dev_comm,
                i32_nblocks, i32_tokens).launch(
            grid=(i32_nblocks, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    return launch


# --- The host op --------------------------------------------------------------

def _torch_view(ptr, nbytes, dtype):
    import torch

    class _View:
        def __init__(self):
            self.__cuda_array_interface__ = {
                "data": (int(ptr), False),
                "shape": (int(nbytes),),
                "strides": None,
                "typestr": "|u1",
                "version": 3,
            }

    return torch.as_tensor(_View(), device=f"cuda:{torch.cuda.current_device()}").view(
        dtype
    )


class MegaMoESdma:
    """MegaMoE stage2 + combine with the transfer done by SDMA from inside the GEMM2 kernel.

    Reuses a ``MegaMoEV2``'s stage1/dispatch state (and reads its private attributes),
    so it can run right after ``moe(...)`` and be compared with that call's own
    combine output.

    ``chunk_tiles`` is how many sort tiles make one shipped chunk; ``comm_wgs`` the number
    of comm workgroups (4 waves each, one wave per peer by default).

    ``gemm2`` picks the GEMM2 kernel: ``"aiter"`` is aiter's GEMM2 main loop with
    ``comm_wgs`` leading comm workgroups; ``"8wave"`` is the 8-wave 256x256
    persistent GEMM2 of ``gemm2_8wave.py`` (workgroup 0 is the comm workgroup), which
    needs fp8 transport, ``remote`` signalling and one queue. Switch with ``set_gemm2``.

    The send/landing layout is exact (from aiter's gathered route counts); ``S`` is
    ``send_slack`` times ``max_tok * topk`` rows, capped by the 32-bit buffer-resource
    range. With fp8 transport (and aiter's comb_inp in the same fp8 format) a row past
    the end of ``S`` takes the P2P fallback -- GEMM2 writes it into the origin rank's
    ``comb_inp`` -- instead of being dropped; ``take_overflow_rows`` reports how many
    did. With bf16 transport such rows are dropped and ``check_overflow`` raises.

    Needs ``MORI_ENABLE_SDMA=1`` and a mori built with SDMA.
    """

    def __init__(
        self,
        moe,
        queues=8,
        chunk_tiles=4,
        comm_wgs=None,
        send_slack=4.0,
        signal_mode="quiet",
        quant="none",
        gemm2="aiter",
    ):
        import torch
        import torch.distributed as dist
        from mori.cco import (
            GDA_CONNECTION_NONE,
            CCODevCommRequirements,
            Communicator,
        )

        cfg = moe.comb_cfg
        self.moe = moe
        self.rank, self.npes = int(cfg.rank), int(cfg.world_size)
        self.hidden = int(cfg.hidden_dim)
        self.topk = int(cfg.num_experts_per_token)
        self.max_tok = int(cfg.max_num_inp_token_per_rank)
        # Token-slot space: one entry per (token, k). This is what idx values and
        # ``inv`` are indexed by.
        self.slot_cap = self.max_tok * self.topk
        # Row space. R holds what one rank gets back -- at most its own routes, so
        # slot_cap rows and never more. S holds what it sends -- everything it
        # received, several times the even share under skewed routing -- so it is
        # send_slack * slot_cap rows, capped so S + R fit a 32-bit buffer offset.
        self.send_slack = send_slack
        if signal_mode not in ("quiet", "remote"):
            raise ValueError(
                f"signal_mode must be quiet or remote, got {signal_mode!r}"
            )
        if signal_mode == "remote" and queues != 1:
            raise ValueError(
                "signal_mode='remote' hangs the completion signal on the last put "
                f"of one channel; queues={queues} spreads the data over channels "
                "it would not cover"
            )
        self.signal_mode = signal_mode
        self.quant = quant
        # fp8 rows carry their e8m0 block scales inline, so one contiguous SDMA
        # put still ships a whole row.
        row_bytes = (
            self.hidden + self.hidden // 32
            if quant == "fp8_blockwise_1x32"
            else self.hidden * 2
        )
        recv_rows = _align(self.slot_cap)
        fit = ((1 << 31) - 1 - recv_rows * row_bytes) // row_bytes // _ALIGN * _ALIGN
        send_rows = min(_align(int(self.slot_cap * send_slack)), fit)
        if send_rows < _ALIGN:
            raise ValueError(f"max_tok={self.max_tok} leaves no room for the send buffer")
        self.layout = ArenaLayout(self.npes, send_rows, recv_rows, row_bytes, self.slot_cap)
        self.layout.check()
        # The exact layout reads aiter's gathered route counts and fanout pairs.
        self.epr = moe.epr
        ws = moe._s1_dispatch_workspace
        self.bigcnt = ws["bigcnt"]
        self.pair_cfg = ws["fanout_pair_config"]
        self.total_segments = self.bigcnt.numel() // self.npes
        # The fallback writes rows in aiter's comb_inp layout, which is the fp8 row
        # only if every bucket of the operator scatters fp8 rows.
        self.p2p_fallback = quant == "fp8_blockwise_1x32" and all(
            e.config.p2p_quant == "fp8_blockwise_1x32" for e in moe._bundle_plan.entries
        )
        self.comb_bytes = self.slot_cap * row_bytes
        dev = torch.device("cuda", self.rank)
        lay = self.layout

        uid = Communicator.get_unique_id() if self.rank == 0 else None
        holder = [uid]
        dist.broadcast_object_list(holder, src=0)
        self.comm = Communicator.init(
            self.npes,
            self.rank,
            holder[0],
            per_rank_vmm=max(Communicator.DEFAULT_PER_RANK_VMM, _align(lay.total_bytes) + (256 << 20)),
        )
        self.mem = self.comm.alloc_mem(lay.total_bytes)
        self.win = self.comm.register_window(self.mem.ptr, self.mem.size)
        _torch_view(self.mem.ptr, lay.total_bytes, torch.uint8).zero_()
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = queues
        self.dev_comm = self.comm.create_dev_comm(reqs)

        # The sort block size can change with the token count, so size the per-tile arrays
        # for the smallest one and build the plan kernel on first use of each size.
        nvm = int(moe._s1_nvm)
        max_tiles = nvm // MIN_SBM + 2
        self.max_tiles = max_tiles
        i32 = dict(dtype=torch.int32, device=dev)
        self.out_row = torch.full((nvm + 4 * 128,), -1, **i32)
        self.tile_cnt = torch.zeros(max_tiles * self.npes, **i32)
        self.tile_base = torch.zeros((max_tiles + 1) * self.npes, **i32)
        # [run rows | send_base | land_off | avail] per pe, then the count-mismatch counter.
        self.cursor = torch.zeros(4 * self.npes + 1, **i32)
        # One completion counter per chunk, each on its own cache line; the plan
        # kernel zeroes the prefix a launch can touch.
        self.cdone_stride = CDONE_STRIDE
        self.cdone = torch.zeros((max_tiles + 1) * CDONE_STRIDE, **i32)
        # Never-reset counters: the plan kernel's barrier, and the combine's last-block count.
        self.plan_bar = torch.zeros(
            torch.cuda.get_device_properties(dev).multi_processor_count * BAR_STRIDE,
            **i32,
        )
        # Rows past the end of S (sent through the fallback, or dropped without it).
        self.overflow = torch.zeros(1, **i32)
        self.combine_fin = torch.zeros(1, **i32)
        # In the window, so a sender can fill it: -1 means "no row for this slot"; the
        # combine kernel puts it back after reading. Set before the barrier below, so
        # no peer's first plan can write into it first.
        self.inv = _torch_view(
            self.mem.ptr + lay.off_inv, self.slot_cap * 4, torch.int32
        )
        self.inv.fill_(-1)
        # The plan's per-row slot, read back by the comm waves that push the peers' inv.
        self.idx_send = torch.zeros(lay.send_rows, **i32)
        self.out = torch.empty(
            self.max_tok * self.hidden, dtype=torch.bfloat16, device=dev
        )
        # The plan kernel barriers across its grid, so keep it resident: one block per CU.
        self.nblocks = torch.cuda.get_device_properties(dev).multi_processor_count
        # The combine has no grid-wide rendezvous, so it need not be resident. Two blocks
        # per CU keep more row loads in flight for the reduce: 1.5-2x measured 1-2% faster
        # stage2+combine at 2048-8192 tokens per rank, 4x and up was already slower.
        self.comb_blocks = 2 * self.nblocks
        torch.cuda.synchronize()
        self.comm.barrier()

        # One comm wave per peer (4 waves per workgroup).
        comm_wgs = max(1, (self.npes + 3) // 4) if comm_wgs is None else comm_wgs
        self.comm_spec = CommSpec(
            layout=lay,
            rank=self.rank,
            queues=queues,
            chunk_tiles=chunk_tiles,
            comm_wgs=comm_wgs,
            signal_mode=signal_mode,
            p2p_fallback=self.p2p_fallback,
            topk=self.topk,
            max_tok=self.max_tok,
            comb_bytes=self.comb_bytes,
        )
        self._plans = {}
        self._gemm2_8w = {}
        self._combine = build_combine(
            layout=lay,
            rank=self.rank,
            hidden=self.hidden,
            topk=self.topk,
            quant=quant,
            signal_mode=signal_mode,
            p2p_fallback=self.p2p_fallback,
        )
        self.set_gemm2(gemm2)

    def _plan_for(self, sbm):
        if sbm < MIN_SBM:
            raise ValueError(
                f"sort_block_m={sbm} is below the supported minimum {MIN_SBM}"
            )
        if sbm not in self._plans:
            self._plans[sbm] = build_rank_sort_plan(
                SBM=sbm,
                topk=self.topk,
                log2_max_tok=self.max_tok.bit_length() - 1,
                recv_cap=int(self.moe.max_recv),
                npes=self.npes,
                rank=self.rank,
                epr=self.epr,
                total_segments=self.total_segments,
                send_rows=self.layout.send_rows,
                p2p_fallback=self.p2p_fallback,
            )
        return self._plans[sbm]

    def set_gemm2(self, gemm2):
        """Pick the GEMM2 kernel, ``"aiter"`` or ``"8wave"`` (see the class docstring)."""
        if gemm2 not in ("aiter", "8wave"):
            raise ValueError(f"gemm2 must be aiter or 8wave, got {gemm2!r}")
        if gemm2 == "8wave":
            spec, lay = self.comm_spec, self.layout
            why = []
            if self.quant != "fp8_blockwise_1x32":
                why.append("fp8_blockwise_1x32 transport")
            if spec.signal_mode != "remote" or spec.queues != 1:
                why.append("signal_mode='remote' with queues=1")
            if not self.p2p_fallback:
                why.append("aiter's comb_inp in fp8 rows (the P2P fallback)")
            if self.moe.model_dim % 256 or self.moe.inter_dim % 256:
                why.append("model_dim and inter_dim multiples of 256")
            if lay.off_inv >= 1 << 31:
                why.append("S+R under 2 GiB (lower send_slack)")
            if why:
                raise ValueError("gemm2='8wave' needs " + ", ".join(why))
        self.gemm2 = gemm2

    def set_chunk_tiles(self, chunk_tiles):
        """Change the shipped chunk size; only the gemm2 kernel is (re)compiled."""
        from dataclasses import replace

        self.comm_spec = replace(self.comm_spec, chunk_tiles=chunk_tiles)

    def _cdone_used(self):
        """Prefix of cdone a launch at this config can touch; the rest stays zero."""
        spec = self.comm_spec
        ct = 1 if spec is None else spec.chunk_tiles
        n_chunks = (self.max_tiles + ct - 1) // ct + 1
        return min(self.cdone.numel(), n_chunks * self.cdone_stride)

    def _launch_plan(self, config, s_fx):
        import flydsl.expr as fx
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        op, lay = self.moe._s1_op, self.layout
        _run_compiled(
            self._plan_for(int(config.stage1.sort_block_m)),
            fx.Int64(op.srcmap_em.data_ptr()),
            fx.Int64(op.tile_row_base.data_ptr()),
            fx.Int64(op.num_valid.data_ptr()),
            fx.Int64(self.tile_cnt.data_ptr()),
            fx.Int64(self.tile_base.data_ptr()),
            fx.Int64(self.cursor.data_ptr()),
            fx.Int64(self.out_row.data_ptr()),
            fx.Int64(self.idx_send.data_ptr()),
            fx.Int64(self.inv.data_ptr()),
            fx.Int64(self.cdone.data_ptr()),
            fx.Int64(self.plan_bar.data_ptr()),
            fx.Int64(self.overflow.data_ptr()),
            fx.Int64(self.mem.ptr + lay.off_done),
            fx.Int64(self.mem.ptr + lay.off_counter),
            fx.Int64(self.bigcnt.data_ptr()),
            fx.Int64(self.pair_cfg.data_ptr()),
            fx.Int64(self.moe._s1_epoch_parity.data_ptr()),
            fx.Int32(self.nblocks),
            fx.Int32(self._cdone_used()),
            s_fx,
        )

    def _launch_gemm2(self, config, s_fx):
        import flydsl.expr as fx

        moe, op, lay = self.moe, self.moe._s1_op, self.layout
        ws, stage2 = moe._s1_dispatch_workspace, config.stage2
        inv = moe._g2_invariants_by_quant["none"]
        run_moe_gemm2_with_sdma(
            fx.Int64(moe._s1_out.view(-1).data_ptr()),
            fx.Int64(moe._s1_osd.data_ptr()),
            fx.Int64(moe.w2.data_ptr()),
            fx.Int64(moe.w2_scale.data_ptr()),
            fx.Int64(op.sorted_expert_ids.data_ptr()),
            fx.Int64(op.num_valid.data_ptr()),
            fx.Int64(ws["max_expert_tiles"].data_ptr()),
            fx.Int64(op.wts_em.data_ptr()),
            fx.Int64(op.tile_row_base.data_ptr()),
            fx.Int64(self.mem.ptr + lay.off_s),
            fx.Int64(self.out_row.data_ptr()),
            fx.Int64(self.dev_comm.ptr),
            fx.Int64(self.win.handle),
            fx.Int64(self.cdone.data_ptr()),
            fx.Int64(self.tile_base.data_ptr()),
            fx.Int64(self.cursor.data_ptr()),
            fx.Int64(self.idx_send.data_ptr()),
            fx.Int64(op.srcmap_em.data_ptr()),
            moe.comb_op._fx_p2p_comb_inp,
            moe._s1_nvm,
            moe._g2v2_inter,
            moe._g2v2_hidden,
            s_fx,
            model_dim=inv["model_dim"],
            inter_dim=inv["inter_dim"],
            experts=inv["experts"],
            rank=inv["rank"],
            send_nbytes=int(lay.send_rows * lay.row_bytes),
            BM=stage2.block_m,
            SBM=config.stage1.sort_block_m,
            HIDDEN_MAX=inv["HIDDEN_MAX"],
            INTER_MAX=inv["INTER_MAX"],
            cu_num=inv["cu_num"],
            BN=stage2.block_n,
            BK=stage2.block_k,
            use_nt=stage2.use_nt,
            g2_bhoist=stage2.b_hoist,
            g2_ascale_pf=stage2.ascale_prefetch,
            g2_spart=stage2.spatial_partition,
            persist=stage2.persist,
            # Pipeline depth: each workgroup runs ceil(total_m_blocks/persist_cu)
            # iterations, and that -- not chunk_tiles -- is how many batches the
            # GEMM hands the comm waves. Strided order also decides *which*
            # m_blocks finish first: strided completes them roughly in order, so
            # a low chunk is whole early; contiguous spreads each slot over its
            # own range, so no chunk is whole until near the end.
            persist_cu=stage2.persist_cu,
            persist_strided=stage2.persist_strided,
            skew_cu=stage2.skew_cu,
            g2_bf16_lds=False if self.quant != "none" else stage2.bf16_lds,
            comm=self.comm_spec,
            quant=self.quant,
        )

    def _launch_gemm2_8wave(self, config, s_fx):
        import flydsl.expr as fx
        import torch
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
        from gemm2_8wave import compile_gemm2_8wave_sdma

        moe, op, lay = self.moe, self.moe._s1_op, self.layout
        sbm = int(config.stage1.sort_block_m)
        ct = self.comm_spec.chunk_tiles
        key = (sbm, ct)
        if key not in self._gemm2_8w:
            self._gemm2_8w[key] = compile_gemm2_8wave_sdma(
                K=moe.inter_dim,
                N=moe.model_dim,
                experts=moe.epr,
                topk=self.topk,
                npes=self.npes,
                max_tok=self.max_tok,
                recv_cap=int(moe.max_recv),
                SBM=sbm,
                rank=self.rank,
                layout=lay,
                chunk_tiles=ct,
                grid_ctas=self.nblocks,
                comb_bytes=self.comb_bytes,
            )
        nvm = int(moe._s1_nvm)
        _run_compiled(
            self._gemm2_8w[key],
            moe._s1_out.view(torch.uint8).view(-1),
            moe.w2.view(torch.uint8).view(-1),
            moe._s1_osd.view(-1),
            moe.w2_scale.view(torch.uint8).view(-1),
            (nvm + 255) // 256 * 256,
            fx.Int64(moe._s1_dispatch_workspace["expert_tile_end"].data_ptr()),
            fx.Int64(op.srcmap_em.data_ptr()),
            fx.Int64(op.wts_em.data_ptr()),
            fx.Int64(op.tile_row_base.data_ptr()),
            moe.comb_op._fx_p2p_comb_inp,
            fx.Int64(op.num_valid.data_ptr()),
            fx.Int64(self.out_row.data_ptr()),
            fx.Int64(self.mem.ptr + lay.off_s),
            fx.Int64(self.dev_comm.ptr),
            fx.Int64(self.win.handle),
            fx.Int64(self.cdone.data_ptr()),
            fx.Int64(self.tile_base.data_ptr()),
            fx.Int64(self.cursor.data_ptr()),
            fx.Int64(self.idx_send.data_ptr()),
            s_fx,
        )

    def run(self, run_tokens, config=None, profile=None):
        """Three kernel launches. ``profile``: a list that receives ``(stage, cuda_event)`` marks."""
        import flydsl.expr as fx
        import torch
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        moe = self.moe
        config = moe._active_config if config is None else config
        s_fx = fx.Stream(torch.cuda.current_stream().cuda_stream)

        def mark(stage):
            if profile is not None:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                profile.append((stage, ev))

        mark("start")
        self._launch_plan(config, s_fx)
        mark("plan")
        if self.gemm2 == "8wave":
            self._launch_gemm2_8wave(config, s_fx)
        else:
            self._launch_gemm2(config, s_fx)
        mark("gemm2+sdma")
        _run_compiled(
            self._combine,
            fx.Int64(self.win.handle),
            fx.Int64(self.inv.data_ptr()),
            fx.Int64(self.out.data_ptr()),
            fx.Int64(moe.comb_op.shmem_comb_inp_tok.data_ptr()),
            fx.Int64(self.combine_fin.data_ptr()),
            fx.Int64(self.dev_comm.ptr),
            fx.Int32(self.comb_blocks),
            fx.Int32(run_tokens),
            s_fx,
        )
        mark("combine")
        return self.out[: run_tokens * self.hidden].view(run_tokens, self.hidden)

    def take_overflow_rows(self):
        """Rows past the end of ``S`` since the last call (syncs the device): sent through
        the P2P fallback, or dropped without it. Raises if a plan found its row counts
        differ from aiter's gathered ones (the exact layout would then be wrong)."""
        import torch

        torch.cuda.synchronize()
        rows = int(self.overflow.item())
        self.overflow.zero_()
        bad = int(self.cursor[4 * self.npes].item())
        if bad:
            raise RuntimeError(
                f"rank {self.rank}: {bad} per-rank row counts differ from the gathered route counts"
            )
        return rows

    def check_overflow(self):
        """Raise if the plan dropped rows since the last check (syncs the device).

        Without the P2P fallback (bf16 transport) a row past the end of ``S`` is
        dropped, so the output is silently wrong unless the host looks at this
        counter. With it nothing is dropped and this only resets the counter.
        """
        dropped = self.take_overflow_rows()
        if dropped and not self.p2p_fallback:
            raise RuntimeError(
                f"rank {self.rank}: {dropped} rows did not fit the send buffer "
                f"({self.layout.send_rows} rows, send_slack={self.send_slack}); raise send_slack"
            )

    def _wait_peers_consumed(self, timeout_s=30.0):
        """Wait until every peer has read what this rank last sent it.

        Under ``remote`` signalling a rank's last call can return while its own
        puts are still in flight; freeing the window then would pull it out from
        under the copy engine.
        """
        import time

        import torch

        torch.cuda.synchronize()
        lay = self.layout
        counter = _torch_view(self.mem.ptr + lay.off_counter, 4, torch.int32)
        done = _torch_view(self.mem.ptr + lay.off_done, self.npes * 4, torch.int32)
        calls = int(counter.item())
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            flags = done.tolist()
            if all(f >= calls for r, f in enumerate(flags) if r != self.rank):
                return
            time.sleep(0.001)

    def close(self):
        self._wait_peers_consumed()
        self.comm.close_all_resources()
