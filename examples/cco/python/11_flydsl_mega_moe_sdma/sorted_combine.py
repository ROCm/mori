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
"""Rank-sorted send buffer for gfx950 MegaMoE stage2 + combine, shipped by cco SDMA.

Stage2 normally scatters every GEMM2 row straight into the origin rank's
``comb_inp``. Here the rows go to a local send buffer ``S`` grouped by origin
rank, so one contiguous run per peer can be moved by a copy engine::

    S = [ origin rank 0: rows... | origin rank 1: rows... | ... ]   (cap rows each)

Within a rank's run rows are in sorted-row order, i.e. expert-major, so a prefix
of every run is finished as soon as the first experts are. That is what lets the
GEMM2 kernel ship chunks while it is still computing.

Three kernels on one stream (the P2P-scatter baseline has two):

    1. rank_sort_plan   persistent grid; count -> scan -> assign with grid barriers:
                        out_row per sorted row, idx_send, tile_base, cursor; zeroes cdone
    2. gemm2_localsort  writes S[out_row]; extra comm waves (one per peer) put finished
                        chunks with SDMA while the GEMM tiles are still computing
    3. combine          persistent grid; wait for the peers' flags -> build slot->row
                        table -> grid barrier -> top-k reduce -> publish "consumed"

Everything lives in ONE cco window, so ``put`` names the same window for source
and destination and only the byte offsets differ::

    [ S | R | idx_send | idx_land | flags | done | counter ]

``flags[src]`` says "src's data for call n has landed" (and how many rows).
``done[dst]`` says "dst has finished reading call n": a sender waits for it before
overwriting that peer's ``R``/``idx_land`` with call n+1, which keeps a fast rank
from running into a slow one. ``counter`` is the call number; only the combine
kernel advances it, after every reader of it has run.

A row's *landing row* is its row index counted from the start of ``S``: ``S`` rows
are ``[0, npes*cap)`` and ``R`` rows ``[npes*cap, 2*npes*cap)``. My own run never
travels; it is read from ``S`` in place.

``cap`` is the rows reserved for ONE (src, dst) pair, not the whole token-slot
space: a sender writes into the peer's ``R`` at a statically known offset with
no handshake, so the reservation has to be per pair. Sizing it at the worst case
(every token of a peer routing all of top-k here) costs ``npes`` times the
window and put bs=8192 over the 4 GiB buffer-resource range; the real counts sit
at ``slot_cap/npes``, so ``row_slack`` reserves that share with headroom and the
plan kernel refuses to place a row past it.

Rows are transported as fp8 with one e8m0 scale per 32 values, laid out as
aiter's P2P scatter lays them out -- payload then scales, inline -- so one
contiguous put still carries a whole row. bf16 doubles the bytes and the
transfer stops fitting under the GEMM.

Both persistent kernels use grid-wide barriers, so their grid must be fully
resident (one block per CU is).
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cvt_pk_f32_fp8
from flydsl.expr.typing import Int8, T
from flydsl.compiler.ast_rewriter import ASTRewriter

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
    global_typed_ptr,
    lds_typed_ptr,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)

BLOCK = 256
WAVE = 64
BAR_STRIDE = 32  # ints per flag_barrier slot: one 128 B cache line
MIN_SBM = 16
_ALIGN = 128


def _align(n):
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


@dataclass(frozen=True)
class ArenaLayout:
    """Byte offsets inside the cco window; identical on every rank."""

    npes: int
    cap: int
    row_bytes: int

    @property
    def n_rows(self):
        return self.npes * self.cap

    @property
    def off_s(self):
        return 0

    @property
    def off_r(self):
        return self.off_s + self.n_rows * self.row_bytes

    @property
    def off_idx_send(self):
        return self.off_r + self.n_rows * self.row_bytes

    @property
    def off_idx_land(self):
        return self.off_idx_send + _align(self.n_rows * 4)

    @property
    def off_flags(self):
        return self.off_idx_land + _align(self.n_rows * 4)

    @property
    def off_done(self):
        return self.off_flags + _align(self.npes * 8)

    @property
    def off_counter(self):
        return self.off_done + _align(self.npes * 4)

    @property
    def total_bytes(self):
        return self.off_counter + _ALIGN

    def check(self):
        # The reduce addresses S and R through one 32-bit buffer-resource offset.
        if 2 * self.n_rows * self.row_bytes >= 1 << 32:
            raise ValueError("S+R exceed the 4 GiB buffer-resource range")


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
def grid_barrier(bar_addr, nblocks, tid):
    """All blocks of a resident grid rendezvous. ``bar_addr`` is a never-reset i32 counter.

    Barrier k of a launch sees arrivals ``(k-1)*nblocks .. k*nblocks-1``, so the target
    ``(old // nblocks + 1) * nblocks`` works for any number of barriers without a reset.
    """
    rocdl.s_waitcnt(0)
    fx.barrier()
    if tid == fx.Int32(0):
        comm_ops.fence_agent_release()
        old = fx.Int32(comm_ops.atomic_add_agent(bar_addr, fx.Int32(1)))
        target = (old // nblocks + fx.Int32(1)) * nblocks
        comm_ops.spin_until_ge_i32_agent(bar_addr, target)
        comm_ops.fence_agent_acquire()
    fx.barrier()


@comm_ops.traced
def flag_barrier(bar_addr, nblocks, tid, bid):
    """All blocks of a resident grid rendezvous without a contended atomic.

    Block ``i`` bumps its own cache-line-sized slot ``bar[i]`` and every block polls
    all slots, so arrival costs one uncontended atomic instead of ``nblocks`` of them
    queueing on one address. Slots are never reset: a block's target is its own slot's
    value at entry plus one, and every block bumps once per launch.
    """
    target = fx.Int32(
        comm_ops.load_i32_global_agent(
            bar_addr + fx.Int64(bid) * fx.Int64(BAR_STRIDE * 4)
        )
    ) + fx.Int32(1)
    rocdl.s_waitcnt(0)
    fx.barrier()
    if tid == fx.Int32(0):
        comm_ops.fence_agent_release()
        comm_ops.atomic_add_agent(
            bar_addr + fx.Int64(bid) * fx.Int64(BAR_STRIDE * 4), fx.Int32(1)
        )
    for i in range(tid, nblocks, fx.Int32(BLOCK)):
        comm_ops.spin_until_ge_i32_agent(
            bar_addr + fx.Int64(i) * fx.Int64(BAR_STRIDE * 4), target
        )
    comm_ops.fence_agent_acquire()
    fx.barrier()


def build_rank_sort_plan(*, SBM, topk, log2_max_tok, recv_cap, npes, cap, rank):
    """Kernel 1: give every valid sorted row a stable index in its rank's run.

    * count:  ``tile_cnt[tile][pe]`` = rows of the sort tile bound for ``pe``
    * prefix: ``tile_base[tile][pe]`` = rows for ``pe`` in earlier tiles (so the
              rows of a tile range are contiguous in the run), ``cursor[pe]`` = total
    * assign: ``out_row = pe*cap + tile_base + (earlier rows of this tile for pe)``,
              and ``idx_send[out_row] = origin_lid*topk + k``

    One rendezvous, between count and prefix; the prefix needs every earlier tile's
    count. It is a ``flag_barrier`` (no single contended atomic). After it, a block
    does not scan the whole ``tile_cnt`` array: for each tile it owns it sums the
    counts of the earlier tiles with one pass of the block's threads (thread ``t``
    handles pe ``t % npes``) and a short LDS reduction -- a parallel sum, not a
    log-depth scan with a barrier per step. The sum runs to the tile's own index,
    so nothing here is sized by a bound on the tile count: a rank that receives
    far more rows than an even share (skewed routing) is handled the same way.

    Before it writes anything the kernel waits until every peer has consumed the
    previous call: that is what guarantees this rank's own outbound puts have
    finished reading ``S`` and ``idx_send``, which this call is about to rewrite.
    ``tile_base`` is clamped to ``cap``, so a chunk that crosses the per-pair
    reservation is cut at it instead of shipping the next peer's rows.

    Count and assign each run one block per tile with one thread per row: a row
    is decoded once, a wave ballot per pe gives the rows of the wave bound for it,
    and the per-wave counts go through a few LDS words. A row's rank among the
    earlier rows for its pe is then a popcount, not a walk over the tile.
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
    lds_bytes = (OFF_PREF + npes) * 4

    @fx.struct
    class PlanShared:
        buf: fx.Array[Int8, lds_bytes, 16]

    # fmt: off
    @flyc.kernel(name=f"mega_rank_sort_plan_sbm{SBM}_k{topk}_p{npes}_c{cap}",
                 known_block_size=[BLOCK, 1, 1])
    def plan(arg_stids: fx.Int64, arg_trb: fx.Int64, arg_num_valid: fx.Int64,
        arg_tile_cnt: fx.Int64, arg_tile_base: fx.Int64, arg_cursor: fx.Int64,
        arg_out_row: fx.Int64, arg_idx: fx.Int64, arg_cdone: fx.Int64, arg_bar: fx.Int64,
        arg_over: fx.Int64, arg_done: fx.Int64, arg_counter: fx.Int64,
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

        # S and idx_send are rewritten from here on. A peer's done flag says it has
        # read everything I sent it last call, and with the copy engine's FIFO that
        # means my puts have finished reading them.
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
        flag_barrier(arg_bar, i32_nblocks, tid, bid)

        # No rows at all: the loop below never runs, so say so explicitly.
        if (tiles == fx.Int32(0)) & (bid == fx.Int32(0)) & (tid < fx.Int32(npes)):
            tile_base[tid] = fx.Int32(0)
            ptr_buf_tensor(arg_cursor, fx.Int32)[tid] = fx.Int32(0)

        idx_buf = ptr_buf_tensor(arg_idx, fx.Int32)
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
                tile_base[tile * fx.Int32(npes) + tid] = (pre < fx.Int32(cap)).select(
                    pre, fx.Int32(cap)
                )
                # The extra row holds the totals the comm waves read as a chunk's end.
                if tile == tiles - fx.Int32(1):
                    total = pre + tile_cnt[tile * fx.Int32(npes) + tid]
                    capped = (total < fx.Int32(cap)).select(total, fx.Int32(cap))
                    tile_base[tiles * fx.Int32(npes) + tid] = capped
                    ptr_buf_tensor(arg_cursor, fx.Int32)[tid] = capped
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
                # Past the per-pair reservation there is nowhere to put this
                # row. Drop it rather than write into the next peer's run,
                # and leave a mark the host can assert on.
                if place >= fx.Int32(cap):
                    comm_ops.atomic_add_agent(arg_over, fx.Int32(1))
                else:
                    out_row = dest_pe * fx.Int32(cap) + place
                    out_buf[g] = out_row
                    idx_buf[out_row] = dest_lid * fx.Int32(topk) + s
            fx.barrier()

    # fmt: off
    @flyc.jit
    def launch(arg_stids: fx.Int64, arg_trb: fx.Int64, arg_num_valid: fx.Int64,
        arg_tile_cnt: fx.Int64, arg_tile_base: fx.Int64, arg_cursor: fx.Int64,
        arg_out_row: fx.Int64, arg_idx: fx.Int64, arg_cdone: fx.Int64, arg_bar: fx.Int64,
        arg_over: fx.Int64, arg_done: fx.Int64, arg_counter: fx.Int64,
        i32_nblocks: fx.Int32, i32_cdone_n: fx.Int32, stream: fx.Stream):
    # fmt: on
        plan(arg_stids, arg_trb, arg_num_valid, arg_tile_cnt, arg_tile_base, arg_cursor,
             arg_out_row, arg_idx, arg_cdone, arg_bar, arg_over, arg_done, arg_counter,
             i32_nblocks, i32_cdone_n).launch(
            grid=(i32_nblocks, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    return launch


def sdma_wait_consumed(w, peer, phase, *, layout, rank):
    """Block until ``peer`` has finished reading the previous call's data from my sends."""
    comm_ops.spin_until_ge_i32_system(
        fx.Int64(w.lsa_ptr(rank, layout.off_done)) + fx.Int64(peer) * fx.Int64(4),
        phase - fx.Int32(1),
        acquire=True,
    )


def sdma_put_run(sdma, win, peer, queue, *, layout, rank, lo, hi, cco):
    """Ship rows ``[lo, hi)`` of my run for ``peer`` (data only; idx goes at the end).

    One put for the whole run. Puts are asynchronous -- each reserves a slot,
    writes the copy descriptor and rings the doorbell -- but a queue is a ring
    its engine drains in order, so splitting a run across channels only buys
    parallelism if those channels sit on different engines. KFD recommends a
    single SDMA engine per peer link on this topology (one bit per link in
    ``recommended_sdma_engine_id_mask``), so they do not: splitting a 1.44 MB run
    in two measured 3% slower, rotating whole runs across two channels 2%.
    """
    rb, cap = layout.row_bytes, layout.cap
    sdma.put(
        peer,
        win,
        fx.Int64(layout.off_r + rank * cap * rb) + fx.Int64(lo) * fx.Int64(rb),
        win,
        fx.Int64(layout.off_s)
        + (fx.Int64(peer) * fx.Int64(cap) + fx.Int64(lo)) * fx.Int64(rb),
        fx.Int64(hi - lo) * fx.Int64(rb),
        queue,
        coop=cco.CoopScope.THREAD,
    )


@comm_ops.traced
def sdma_finish_peer(
    sdma, w, win, peer, queue, cnt, phase, *, layout, rank, cco, signal_mode="remote"
):
    """Ship idx, tell ``peer`` its rows have landed, and publish ``cnt``.

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

    The flag carries ``cnt`` either way. Under ``remote`` it no longer says
    anything about the data, so it needs no release -- the signal does that.
    """
    if signal_mode not in ("quiet", "remote"):
        raise ValueError(f"signal_mode must be quiet or remote, got {signal_mode!r}")
    remote = signal_mode == "remote"
    # Under `remote` this put carries the signal, so it is issued even when
    # there is no idx to ship -- 4 bytes the peer never reads.
    if const_expr(remote) or cnt > fx.Int32(0):
        sdma.put(
            peer,
            win,
            fx.Int64(layout.off_idx_land + rank * layout.cap * 4),
            win,
            fx.Int64(layout.off_idx_send) + fx.Int64(peer) * fx.Int64(layout.cap * 4),
            fx.Int64((cnt > fx.Int32(0)).select(cnt, fx.Int32(1))) * fx.Int64(4),
            queue,
            coop=cco.CoopScope.THREAD,
            remote_signal=True if remote else None,
        )
    if const_expr(not remote):
        sdma.quiet_queue(peer, queue)
        # The bytes moved by the copy engine, not by this CU, so nothing of ours
        # has to be written back -- this only keeps the flag store from being
        # hoisted above the drain.
        rocdl.s_waitcnt(vmcnt=0)
    comm_ops.store_i64_global_system(
        fx.Int64(w.lsa_ptr(peer, layout.off_flags)) + fx.Int64(rank * 8),
        (fx.Int64(cnt) << fx.Int64(32)) | fx.Int64(phase),
    )


def build_combine(*, layout, rank, hidden, topk, quant="none", signal_mode="remote"):
    """Kernel 3: finish the transfer and reduce.

    One persistent launch (grid resident), in phases:

      A. wait for every peer's flag for this call (acquire), which also says how many
         rows each sent;
      B. ``inv[idx[src][j]] = landing row`` for every landed row, spread over the grid;
      -- grid barrier --
      C. ``out[t] = sum_k row[inv[t*topk+k]]`` in f32, one block per token in turn. A slot's
         ``inv`` entry is reset to -1 once read, so it starts clean next call and nothing
         has to refill it; slots no route filled stay -1 and contribute zero;
      D. the last block to finish publishes ``done`` to every peer and advances ``counter``.
    """
    import mori.cco.device.flydsl as cco

    if hidden % 8:
        raise ValueError(f"hidden must be a multiple of 8, got {hidden}")
    quant_fp8 = quant == "fp8_blockwise_1x32"
    if quant not in ("none", "fp8_blockwise_1x32"):
        raise ValueError(f"unsupported quant={quant!r}")
    if quant_fp8 and hidden % 32:
        raise ValueError(f"fp8_blockwise_1x32 needs hidden % 32 == 0, got {hidden}")
    npes, cap = layout.npes, layout.cap
    units = hidden // 8
    chunks = (units + BLOCK - 1) // BLOCK
    row_bytes = layout.row_bytes

    # fmt: off
    @flyc.kernel(name=f"mega_combine_p{npes}_c{cap}_h{hidden}_k{topk}_{quant}_{signal_mode}"
                 f"_r{rank}", known_block_size=[BLOCK, 1, 1])
    def combine(win: fx.Int64, arg_cursor: fx.Int64, arg_inv: fx.Int64, arg_out: fx.Int64,
        arg_bar: fx.Int64, arg_fin: fx.Int64, arg_dev_comm: fx.Int64,
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
                # The flag carries cnt (and, under "quiet", data-readiness too).
                comm_ops.spin_until_ge_i32_system(
                    flags_base + fx.Int64(tid) * fx.Int64(8), phase, acquire=True
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

        # B. slot -> landing row
        idx_send = fx.Int64(w.lsa_ptr(rank, layout.off_idx_send))
        idx_land = fx.Int64(w.lsa_ptr(rank, layout.off_idx_land))
        inv_buf = ptr_buf_tensor(arg_inv, fx.Int32)
        cursor = ptr_buf_tensor(arg_cursor, fx.Int32)
        gid = bid * fx.Int32(BLOCK) + tid
        stride = i32_nblocks * fx.Int32(BLOCK)
        for g in range(gid, fx.Int32(npes * cap), stride):
            src = g // fx.Int32(cap)
            is_self = src == fx.Int32(rank)
            theirs = fx.Int32(
                comm_ops.load_i32_global_system(flags_base + fx.Int64(src) * fx.Int64(8) + fx.Int64(4))
            )
            cnt = is_self.select(cursor[fx.Int32(rank)], theirs)
            if (g - src * fx.Int32(cap)) < cnt:
                base = is_self.select(idx_send, idx_land)
                slot = global_typed_ptr(base + fx.Int64(g) * fx.Int64(4), T.i32)[0]
                inv_buf[slot] = is_self.select(g, g + fx.Int32(npes * cap))
        grid_barrier(arg_bar, i32_nblocks, tid)

        # C. top-k reduce
        s_base = _uniform_i64(w.lsa_ptr(rank, layout.off_s))
        if const_expr(quant_fp8):
            # Payload is 8 fp8 bytes (2 i32) per thread; the 32-value block they
            # fall in is u // 4, so each thread reads its own scale byte rather
            # than electing a leader and broadcasting.
            rows = ptr_buf_tensor(s_base, fx.Int32, unit_elems=2)
            scales = ptr_buf_tensor(s_base, fx.Int8)
        else:
            rows = ptr_buf_tensor(s_base, fx.BFloat16, unit_elems=8)
        out = ptr_buf_tensor(arg_out, fx.BFloat16, unit_elems=8)
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
                            row0 = safe * fx.Int32(row_bytes)
                            raw = fx.Vector(
                                buf_copy_load(
                                    rows, (row0 + u * fx.Int32(8)) // fx.Int32(8),
                                    fx.Int32, unit_elems=2,
                                )
                            )
                            sc = scales[row0 + fx.Int32(hidden) + u // fx.Int32(4)]
                            scale = (
                                fx.Uint8(sc).to(fx.Uint32).bitcast(fx.Int32)
                                << fx.Int32(23)
                            ).bitcast(fx.Float32)
                            v2f32 = T.vec(2, T.f32)
                            parts = []
                            for word in range_constexpr(2):
                                lo = cvt_pk_f32_fp8(res=v2f32, src=raw[word], word_sel=False)
                                hi = cvt_pk_f32_fp8(res=v2f32, src=raw[word], word_sel=True)
                                parts.append(lo.shuffle(hi, [0, 1, 2, 3]))
                            vals = fx.Vector(parts[0]).shuffle(
                                fx.Vector(parts[1]), [0, 1, 2, 3, 4, 5, 6, 7]
                            )
                            acc = acc + fx.Vector(vals) * (scale * keep)
                        else:
                            raw = buf_copy_load(
                                rows, safe * fx.Int32(units) + u, fx.BFloat16, unit_elems=8
                            )
                            acc = acc + fx.Vector(raw).to(fx.Float32) * keep
                    buf_copy_store(
                        out, tok * fx.Int32(units) + u, acc.to(fx.BFloat16), fx.BFloat16, unit_elems=8
                    )
            # Every thread has the slots in registers; only then may they be reset.
            fx.barrier()
            if tid < fx.Int32(topk):
                inv_buf[tok * fx.Int32(topk) + tid] = fx.Int32(-1)

        # D. the last block out tells the peers this call's data has been consumed
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
    def launch(win: fx.Int64, arg_cursor: fx.Int64, arg_inv: fx.Int64, arg_out: fx.Int64,
        arg_bar: fx.Int64, arg_fin: fx.Int64, arg_dev_comm: fx.Int64,
        i32_nblocks: fx.Int32, i32_tokens: fx.Int32, stream: fx.Stream):
    # fmt: on
        combine(win, arg_cursor, arg_inv, arg_out, arg_bar, arg_fin, arg_dev_comm,
                i32_nblocks, i32_tokens).launch(
            grid=(i32_nblocks, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    return launch


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


class SortedCombine:
    """GEMM2 with a rank-sorted local output, shipped by SDMA from inside the GEMM2 kernel.

    Reuses a ``MegaMoEV2``'s stage1/dispatch state, so it can run right after
    ``moe(...)`` and be compared with that call's own combine output.

    ``chunk_tiles`` is how many sort tiles make one shipped chunk; ``comm_wgs`` the number
    of comm workgroups (4 waves each, one wave per peer by default).

    Needs ``MORI_ENABLE_SDMA=1`` and a mori built with SDMA.
    """

    def __init__(
        self,
        moe,
        queues=8,
        chunk_tiles=4,
        comm_wgs=None,
        row_slack=1.5,
        signal_mode="quiet",
        quant="none",
    ):
        import torch
        import torch.distributed as dist
        from mori.cco import (
            GDA_CONNECTION_NONE,
            CCODevCommRequirements,
            Communicator,
        )

        from moe_gemm2_localsort import CommSpec

        cfg = moe.comb_cfg
        self.moe = moe
        self.rank, self.npes = int(cfg.rank), int(cfg.world_size)
        self.hidden = int(cfg.hidden_dim)
        self.topk = int(cfg.num_experts_per_token)
        self.max_tok = int(cfg.max_num_inp_token_per_rank)
        # Token-slot space: one entry per (token, k). This is what idx values and
        # ``inv`` are indexed by.
        self.slot_cap = self.max_tok * self.topk
        # Row space: how many rows one rank reserves for ONE peer, in both the
        # send buffer and the peer's landing buffer. A sender writes into the
        # peer's landing buffer at a statically known offset, with no handshake,
        # so this has to be reserved per (src, dst) pair rather than packed. The
        # worst case is slot_cap (every token of the peer routing all of top-k
        # here), but the real counts sit at slot_cap/npes -- 4012..4221 against
        # an even share of 4096, measured at bs=4096 -- so reserve that share
        # times ``row_slack`` and have the plan kernel refuse to write past it.
        # Reserving the worst case instead costs npes times the window and is
        # what put bs=8192 over the 4 GiB buffer-resource range.
        self.row_slack = row_slack
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
        share = -(-self.slot_cap // self.npes)
        self.cap = -(-int(share * row_slack) // 128) * 128
        self.quant = quant
        # fp8 rows carry their e8m0 block scales inline, so one contiguous SDMA
        # put still ships a whole row.
        row_bytes = (
            self.hidden + self.hidden // 32
            if quant == "fp8_blockwise_1x32"
            else self.hidden * 2
        )
        self.layout = ArenaLayout(self.npes, self.cap, row_bytes)
        self.layout.check()
        dev = torch.device("cuda", self.rank)
        lay = self.layout

        uid = Communicator.get_unique_id() if self.rank == 0 else None
        holder = [uid]
        dist.broadcast_object_list(holder, src=0)
        self.comm = Communicator.init(self.npes, self.rank, holder[0])
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
        self.cursor = torch.zeros(self.npes, **i32)
        # One completion counter per chunk, each on its own cache line; the plan
        # kernel zeroes the prefix a launch can touch.
        from moe_gemm2_localsort import CDONE_STRIDE

        self.cdone_stride = CDONE_STRIDE
        self.cdone = torch.zeros((max_tiles + 1) * CDONE_STRIDE, **i32)
        # Never-reset counters for the two grid barriers and the combine's last-block count.
        self.plan_bar = torch.zeros(
            torch.cuda.get_device_properties(dev).multi_processor_count * BAR_STRIDE,
            **i32,
        )
        # Rows the per-pair reservation had no room for; must stay 0.
        self.overflow = torch.zeros(1, **i32)
        self.combine_bar = torch.zeros(1, **i32)
        self.combine_fin = torch.zeros(1, **i32)
        # -1 means "no row for this slot"; the combine kernel puts it back after reading.
        self.inv = torch.full((self.slot_cap,), -1, **i32)
        self.out = torch.empty(
            self.max_tok * self.hidden, dtype=torch.bfloat16, device=dev
        )
        # Both persistent kernels barrier across the grid, so keep it resident: one block per CU.
        self.nblocks = torch.cuda.get_device_properties(dev).multi_processor_count
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
        )
        self._plans = {}
        self._combine = build_combine(
            layout=lay,
            rank=self.rank,
            hidden=self.hidden,
            topk=self.topk,
            quant=quant,
            signal_mode=signal_mode,
        )

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
                cap=self.cap,
                rank=self.rank,
            )
        return self._plans[sbm]

    def set_chunk_tiles(self, chunk_tiles):
        """Change the shipped chunk size; only the gemm2 kernel is (re)compiled."""
        from dataclasses import replace

        self.comm_spec = replace(self.comm_spec, chunk_tiles=chunk_tiles)

    def _cdone_used(self, config):
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
            fx.Int64(self.mem.ptr + lay.off_idx_send),
            fx.Int64(self.cdone.data_ptr()),
            fx.Int64(self.plan_bar.data_ptr()),
            fx.Int64(self.overflow.data_ptr()),
            fx.Int64(self.mem.ptr + lay.off_done),
            fx.Int64(self.mem.ptr + lay.off_counter),
            fx.Int32(self.nblocks),
            fx.Int32(self._cdone_used(config)),
            s_fx,
        )

    def _launch_gemm2(self, config, s_fx):
        import flydsl.expr as fx
        from moe_gemm2_localsort import run_moe_gemm2_localsort

        moe, op, lay = self.moe, self.moe._s1_op, self.layout
        ws, stage2 = moe._s1_dispatch_workspace, config.stage2
        inv = moe._g2_invariants_by_quant["none"]
        run_moe_gemm2_localsort(
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
            moe._s1_nvm,
            moe._g2v2_inter,
            moe._g2v2_hidden,
            s_fx,
            model_dim=inv["model_dim"],
            inter_dim=inv["inter_dim"],
            experts=inv["experts"],
            rank=inv["rank"],
            send_nbytes=int(lay.n_rows * lay.row_bytes),
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
        self._launch_gemm2(config, s_fx)
        mark("gemm2+sdma")
        _run_compiled(
            self._combine,
            fx.Int64(self.win.handle),
            fx.Int64(self.cursor.data_ptr()),
            fx.Int64(self.inv.data_ptr()),
            fx.Int64(self.out.data_ptr()),
            fx.Int64(self.combine_bar.data_ptr()),
            fx.Int64(self.combine_fin.data_ptr()),
            fx.Int64(self.dev_comm.ptr),
            fx.Int32(self.nblocks),
            fx.Int32(run_tokens),
            s_fx,
        )
        mark("combine")
        return self.out[: run_tokens * self.hidden].view(run_tokens, self.hidden)

    def check_overflow(self):
        """Raise if the plan dropped rows since the last check (syncs the device).

        A row that does not fit its peer's per-pair reservation is dropped, so
        the output is silently wrong unless the host looks at this counter.
        """
        import torch

        torch.cuda.synchronize()
        dropped = int(self.overflow.item())
        self.overflow.zero_()
        if dropped:
            raise RuntimeError(
                f"rank {self.rank}: {dropped} rows did not fit the per-pair reservation "
                f"(cap={self.cap} rows, row_slack={self.row_slack}); raise row_slack"
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
