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
"""MegaMoE stage1 dispatch on cco SDMA (gfx950): the new prepare.

Stage1 dispatch used to copy every routed row into the destination rank with P2P
stores from producer CUs. Here the sender writes the rows into a local send buffer
``S`` grouped by destination rank and ships each rank's run with SDMA ``put``.

A token goes to a destination rank once, whatever the number of its experts there.
Its row in the run of rank ``d`` sits at the position given by its FIRST expert on
``d`` (the smallest local expert id among the token's top-k)::

    run(d) = [ rows with first expert 0 | rows with first expert 1 | ... ]

So the rows an expert ``e`` needs on ``d`` all lie in the prefix of the run that ends
with the bucket of ``e``: a prefix of the run that has landed is a prefix of experts
that can start. The order inside a bucket is not defined (atomics); the receiver
reads rows through an index table, so it does not matter.

The whole prepare is ONE kernel (``build_prepare``) and the receiver's wait is another
(``build_wait``), or ``gemm1_sdma`` waits by itself.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int8, ReductionOp, T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels.tensor_shim import ptr_buf_tensor
from aiter.ops.flydsl.kernels.mega_moe.quant import (
    _FP8_E4M3_INV_MAX_POS_BITS,
    GROUP,
)

PREP_BLOCK = 512
PREP_WAVES = PREP_BLOCK // 64
#: i32 slots between two chunk-completion counters: one 128 B cache line each, because
#: every quantised row of a chunk bumps its counter.
CDONE_STRIDE = 32
#: Row stores go through to memory (sc0|sc1), so retiring them is the release the copy
#: engine needs; see ``gemm2_8wave.SEND_CM`` in example 11.
SEND_CM = 1 | 16
BAR_STRIDE = 32  # ints per tree_barrier counter: one 128 B cache line
BAR_GROUP = 32  # blocks per tree_barrier group


def _align16(n):
    return (n + 15) // 16 * 16


def _align128(n):
    return (n + 127) // 128 * 128


@dataclass(frozen=True)
class Arena:
    """Byte offsets inside the cco window ``[ S | R ]``; identical on every rank.

    A row is ``K`` fp8 bytes then ``K / 32`` e8m0 scales, so one contiguous put carries
    whole rows. ``S`` holds the rows I send, run ``d`` at ``S[d * cap]``; ``R`` the rows I
    receive, run ``s`` at ``R[s * cap]``, where ``cap`` rows are reserved for each pair:
    a token goes to a rank once, so ``cap = max_tok_per_rank`` is the worst case and a
    sender can write at a static offset with no handshake.
    """

    npes: int
    cap: int
    hidden: int
    chunk_rows: int
    topk: int
    experts_per_rank: int
    #: chunks of each run that ``prepare`` puts itself; the GEMM1 comm workgroup puts the rest
    early_chunks: int = 1

    @property
    def row_bytes(self):
        return _align16(self.hidden + self.hidden // GROUP)

    @property
    def n_rows(self):
        return self.npes * self.cap

    @property
    def off_s(self):
        return 0

    @property
    def off_r(self):
        return self.n_rows * self.row_bytes

    @property
    def max_routes(self):
        """Expert-major rows a rank can receive, at most (one per (source token, k))."""
        return self.n_rows * self.topk

    @property
    def off_rowmap(self):
        """``rowmap[r]``: the row that expert-major row ``r`` reads, counted from the start of
        ``S``: ``S`` rows are ``[0, n_rows)`` and ``R`` rows ``[n_rows, 2 * n_rows)``. My own
        run is read from ``S`` in place. Filled by the sender."""
        return self.off_r + self.n_rows * self.row_bytes

    @property
    def off_wts(self):
        """``wts[r]``: the routing weight of row ``r`` (f32 bits)."""
        return self.off_rowmap + _align128(self.max_routes * 4)

    @property
    def off_srcmap(self):
        """``srcmap[r]``: ``(source rank * max_tok + token) | k << 24`` of row ``r``."""
        return self.off_wts + _align128(self.max_routes * 4)

    @property
    def off_need(self):
        """``need[s * epr + e]``: chunks of source ``s``'s run for me that have to have landed
        before my expert ``e`` can start (its prefix ``cum`` in chunks). Filled by the sender.
        """
        return self.off_srcmap + _align128(self.max_routes * 4)

    @property
    def off_flags(self):
        """``flags[s]``: how many of source ``s``'s experts-of-mine have pushed their rows."""
        return self.off_need + _align128(self.npes * self.experts_per_rank * 4)

    @property
    def total_bytes(self):
        return self.off_flags + _align128(self.npes * 4)

    @property
    def cdone_words(self):
        """i32 slots of the counters of the early chunks of ONE run (one cache line each)."""
        return self.early_chunks * CDONE_STRIDE

    def check(self):
        # The quant kernel addresses S through one 32-bit buffer-resource offset.
        if self.n_rows * self.row_bytes >= 1 << 32:
            raise ValueError("S exceeds the 4 GiB buffer-resource range")
        if self.hidden % 256 or self.chunk_rows <= 0 or self.early_chunks < 1:
            raise ValueError(
                "hidden must be a multiple of 256, chunk_rows and early_chunks positive"
            )
        # A chunk has to start on a 128 B cache line (rows are a multiple of 16 B apart): then
        # the line at its first row holds no byte of the chunk before it, so a row that has
        # landed can be read while the next chunk has not -- no stale line of that one is
        # pulled into L2 on the way.
        if self.chunk_rows * self.row_bytes % 128:
            raise ValueError(f"chunk_rows={self.chunk_rows} must be a multiple of 8")


@comm_ops.traced
def tree_barrier(bar_addr, nblocks, tid, bid):
    """All blocks of a resident grid rendezvous with two levels of small atomics.

    Blocks are grouped ``BAR_GROUP`` to a counter; the last block of a group bumps a top
    counter; everyone polls only the top one. Counters are never reset: a group's counter
    advances by its size per launch, so the launch number is ``old // size``. Each counter
    sits on its own cache line. (``tree_barrier`` of example 11's plan kernel.)
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


def _quant_group(x_rsrc, in_dword, s_rsrc, out_byte, scale_byte):
    """One 32-value group of a bf16 row -> 32 fp8 bytes and one e8m0 scale in ``S``.

    The numbers of ``quant.emit_per_1x32_mx_fp8_group``; only the stores differ: they
    are written through (``SEND_CM``) at byte offsets of the send buffer.
    """
    act = []
    local_max = fx.Float32(1e-10)
    for chunk in range_constexpr(GROUP // 8):
        raw = buffer_load(
            x_rsrc, in_dword + fx.Int32(chunk * 4), vec_width=4, dtype=T.i32
        )
        values = fx.Vector(raw).bitcast(fx.BFloat16).to(fx.Float32)
        local_max = local_max.maximumf(fmath.absf(values).reduce(ReductionOp.MAX))
        for elem in range_constexpr(8):
            act.append(values[elem])
    working = (
        local_max * fx.Int32(_FP8_E4M3_INV_MAX_POS_BITS).bitcast(fx.Float32)
    ).bitcast(fx.Int32)
    mantissa = working & fx.Int32(0x7FFFFF)
    biased_exp = (working >> fx.Int32(23)) & fx.Int32(0xFF)
    e8m0 = (mantissa != fx.Int32(0)).select(biased_exp + fx.Int32(1), biased_exp)
    e8m0 = (e8m0 > fx.Int32(255)).select(fx.Int32(255), e8m0)
    buffer_store(
        e8m0.to(fx.Int8),
        s_rsrc,
        scale_byte,
        cache_modifier=SEND_CM,
        offset_is_bytes=True,
    )
    quant_scale = ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(fx.Float32)
    scaled = [act[k] * quant_scale for k in range_constexpr(GROUP)]
    for half in range_constexpr(2):
        words = []
        for word in range_constexpr(4):
            base = (half * 4 + word) * 4
            packed = rocdl.cvt_pk_fp8_f32(
                T.i32, scaled[base], scaled[base + 1], fx.Int32(0), 0
            )
            packed = rocdl.cvt_pk_fp8_f32(
                T.i32, scaled[base + 2], scaled[base + 3], packed, 1
            )
            words.append(packed)
        buffer_store(
            fx.Vector.from_elements(words, fx.Int32),
            s_rsrc,
            out_byte + fx.Int32(half * 16),
            cache_modifier=SEND_CM,
            offset_is_bytes=True,
        )


def put_chunk(sdma, win, peer, c, n, *, rank, arena, cco):
    """Put chunk ``c`` of my run for ``peer`` (``n`` rows in all) into the same place of its
    ``R``, with a remote signal: the peer's copy engine bumps a counter on the peer when the
    bytes are there."""
    rb, cap, CR = arena.row_bytes, arena.cap, arena.chunk_rows
    lo = c * fx.Int32(CR)
    hi = (lo + fx.Int32(CR) < n).select(lo + fx.Int32(CR), n)
    sdma.put(
        peer,
        win,
        fx.Int64(arena.off_r + rank * cap * rb) + fx.Int64(lo) * fx.Int64(rb),
        win,
        fx.Int64(arena.off_s)
        + (fx.Int64(peer) * fx.Int64(cap) + fx.Int64(lo)) * fx.Int64(rb),
        fx.Int64(hi - lo) * fx.Int64(rb),
        0,
        coop=cco.CoopScope.THREAD,
        local_signal=False,
        remote_signal=True,
    )


@comm_ops.traced
def post_rest_chunks(sdma, win, wave, lane, cnt, *, rank, arena, cco):
    """Comm wave ``wave``, which serves peer ``wave``: put the chunks of my run for it that
    ``prepare`` did not (every one but the first ``early_chunks``). All rows are in ``S`` already, because
    ``prepare`` has finished, so nothing is waited for: the puts are posted back to back and
    the copy engine moves them while the GEMM computes. Used by ``build_post_rest`` and by the
    comm workgroup of ``gemm1_sdma``."""
    if (wave < fx.Int32(arena.npes)) & (wave != fx.Int32(rank)):
        n = cnt[wave]
        n_chunks = (n + fx.Int32(arena.chunk_rows - 1)) // fx.Int32(arena.chunk_rows)
        for c in range(fx.Int32(arena.early_chunks), n_chunks, fx.Int32(1)):
            if lane == fx.Int32(0):
                put_chunk(
                    sdma, win, wave, fx.Int32(c), n, rank=rank, arena=arena, cco=cco
                )


def build_prepare(*, rank, arena, epr, quant_blocks, route_push):
    """The new prepare, one resident kernel: plan, route push, quantise into ``S``, put.

    Workgroup 0 is the comm workgroup, the other ``quant_blocks`` quantise; every
    workgroup first takes part in the plan, and the grid is resident (one block per CU), so
    the phases are separated by grid barriers (``tree_barrier``).

    **Plan** (all blocks, tokens spread over all threads). A token's position in the run of
    destination ``d`` is set by its first expert there, so the run is a counting sort by
    ``(d, first expert)`` = by one of the ``E`` global experts ``g``:

    1. each block counts its tokens per ``g`` in LDS and adds them to ``hist[g]``;  barrier
    2. every block turns ``hist`` into per-``g`` bases (block 0 also stores ``cum``, the
       rows with first expert ``<= e``, and ``cnt[d]``), and reserves its share of each ``g``
       with one atomic on ``cursor[g]``;
    3. each token takes a slot of its block's share with an LDS atomic and stores
       ``rowpos[t, d]`` and ``row_tok[d, pos]``;  barrier.

    The order inside a ``g`` is the atomics' (not defined). Block 0 then zeroes ``hist`` and
    ``cursor`` for the next call.

    **Route push** (``route_push``; quant blocks, one expert at a time): tell every
    destination where each of my rows goes and where it reads it. The expert-major row numbers come from the
    existing compact plan (``pair_order``: my routes sorted by expert, ``pair_base`` /
    ``local_hist``: where an expert's routes start and how many there are, ``my_base``: the
    destination row of my first route of that expert). For route ``(token, k)`` to expert
    ``ge`` on rank ``dst`` as destination row ``r`` this stores, straight into ``dst``'s
    window::

        rowmap[r] = rowpos[token, dst] + (rank * cap if dst == rank else n_rows + rank * cap)
        wts[r]    = weight[token, k]
        srcmap[r] = (rank * max_tok + token) | k << 24   (what stage2's combine reads)

    and once per expert ``need[rank * epr + e]``, the chunks of my run for ``dst`` that expert
    ``e`` of ``dst`` has to wait for (``cum``, in chunks). When it has stored its rows the
    block bumps ``flags[rank]`` on ``dst``: a destination has all of mine once it has seen
    ``epr`` bumps per call.

    **Quantise and put.** A quant WAVE takes one row at a time (its 64 lanes share the row's
    ``hidden / 32`` groups). The rows are taken chunk-major over the peers -- chunk 0 of
    every peer, then chunk 1 -- and the rows for this rank last, so the first chunks are the
    first to be complete::

        item i < rounds * (npes - 1) * chunk_rows : (chunk c, peer rank+1+dd, row r)
        item i >= that                            : row i - that of my own run

    (Only the early chunks have to come first; the rest of the order does not matter.)

    A row of one of the first ``early_chunks`` chunks, once its stores have retired (they are
    written through), bumps the counter ``cdone[d, c]`` of its chunk. Comm wave ``p`` (one per
    peer, ``npes <= 8``) waits for each early chunk of run ``p`` to be complete, zeroes its
    counter for the next call, and posts its put with a remote signal (``put_chunk``).
    **Those are the only puts prepare posts**: posting is not free (about 50 us of the kernel
    when it posted every chunk), and the other chunks are put by the GEMM1 kernel's comm
    workgroup (``post_rest_chunks``), off prepare's critical path. The put names a fixed place in the peer's ``R``; nothing is waited for on
    the receiver.

    Counters ``hist``, ``cursor``, ``cdone`` start at zero and are left at zero; the barrier
    counters ``bar`` are never reset.
    """
    import mori.cco.device.flydsl as cco

    npes, cap, hidden, CR, topk = (
        arena.npes,
        arena.cap,
        arena.hidden,
        arena.chunk_rows,
        arena.topk,
    )
    arena.check()
    if npes > PREP_WAVES:
        raise ValueError(f"one comm wave per peer needs npes <= {PREP_WAVES}")
    rb = arena.row_bytes
    groups = hidden // GROUP
    per_round = (npes - 1) * CR
    E = npes * epr
    nblocks = quant_blocks + 1
    nthreads = nblocks * PREP_BLOCK
    n_bar_groups = (nblocks + BAR_GROUP - 1) // BAR_GROUP
    bar2_off = (
        (n_bar_groups + 1) * BAR_STRIDE * 4
    )  # the second barrier's counters, in bytes
    lds_bytes = 3 * E * 4  # per-block counts | per-expert bases | per-block cursors

    @fx.struct
    class PrepShared:
        buf: fx.Array[Int8, lds_bytes, 16]

    # fmt: off
    @flyc.kernel(name=f"stage1_sdma_prepare_p{npes}_r{rank}_e{epr}_k{topk}_h{hidden}_c{cap}_cr{CR}"
                      f"_q{quant_blocks}_rp{int(route_push)}", known_block_size=[PREP_BLOCK, 1, 1])
    def prepare(arg_ids: fx.Int64, arg_x: fx.Int64, arg_row_tok: fx.Int64, arg_rowpos: fx.Int64,
                arg_cnt: fx.Int64, arg_cum: fx.Int64, arg_cdone: fx.Int64, arg_hist: fx.Int64,
                arg_cursor: fx.Int64, arg_bar: fx.Int64, arg_s: fx.Int64, arg_dev_comm: fx.Int64,
                arg_win: fx.Int64, arg_pair_order: fx.Int64, arg_pair_base: fx.Int64,
                arg_local_hist: fx.Int64, arg_my_base: fx.Int64, arg_wts: fx.Int64,
                tokens: fx.Int32):
    # fmt: on
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        wave = tid // fx.Int32(64)
        lane = tid % fx.Int32(64)
        gtid = bid * fx.Int32(PREP_BLOCK) + tid
        ids = ptr_buf_tensor(arg_ids, fx.Int32)
        row_tok = ptr_buf_tensor(arg_row_tok, fx.Int32)
        rowpos = ptr_buf_tensor(arg_rowpos, fx.Int32)
        cnt = ptr_buf_tensor(arg_cnt, fx.Int32)
        cum = ptr_buf_tensor(arg_cum, fx.Int32)
        hist = ptr_buf_tensor(arg_hist, fx.Int32)
        cursor = ptr_buf_tensor(arg_cursor, fx.Int32)
        cdone_addr = fx.Int64(arg_cdone)
        hist_addr = fx.Int64(arg_hist)
        cursor_addr = fx.Int64(arg_cursor)
        lds = fx.SharedAllocator().allocate(PrepShared).peek()
        lds_hist = fx.Int64(fx.ptrtoint(lds.buf.ptr))
        lds_base = lds_hist + fx.Int64(E * 4)
        lds_cursor = lds_base + fx.Int64(E * 4)

        def firsts(t):
            """``f[d]``: the smallest local expert of ``d`` among the top-k of token ``t``,
            ``epr`` if it has none there (an id of -1 counts for no rank)."""
            f = [fx.Int32(epr) for _ in range_constexpr(npes)]
            for j in range_constexpr(topk):
                g = ids[t * fx.Int32(topk) + fx.Int32(j)]
                d = g // fx.Int32(epr)
                local = g - d * fx.Int32(epr)
                for dd in range_constexpr(npes):
                    hit = (g >= fx.Int32(0)) & (d == fx.Int32(dd)) & (local < f[dd])
                    f[dd] = hit.select(local, f[dd])
            return f

        # ---- plan 1: count
        for g in range(tid, fx.Int32(E), fx.Int32(PREP_BLOCK)):
            comm_ops.store_i32_lds(lds_hist + fx.Int64(g) * fx.Int64(4), fx.Int32(0))
            comm_ops.store_i32_lds(lds_cursor + fx.Int64(g) * fx.Int64(4), fx.Int32(0))
        fx.barrier()
        for t in range(gtid, tokens, fx.Int32(nthreads)):
            f = firsts(t)
            for dd in range_constexpr(npes):
                if f[dd] < fx.Int32(epr):
                    comm_ops.atomic_add_lds(
                        lds_hist + fx.Int64(fx.Int32(dd * epr) + f[dd]) * fx.Int64(4), fx.Int32(1)
                    )
        fx.barrier()
        for g in range(tid, fx.Int32(E), fx.Int32(PREP_BLOCK)):
            v = fx.Int32(comm_ops.load_i32_lds(lds_hist + fx.Int64(g) * fx.Int64(4)))
            if v > fx.Int32(0):
                comm_ops.atomic_add_agent(hist_addr + fx.Int64(g) * fx.Int64(4), v)
        tree_barrier(fx.Int64(arg_bar), fx.Int32(nblocks), tid, bid)

        # ---- plan 2: bases (and, in block 0, cum and cnt)
        if tid < fx.Int32(npes):
            run = fx.Int32(0)
            for e in range_constexpr(epr):
                h = hist[tid * fx.Int32(epr) + fx.Int32(e)]
                comm_ops.store_i32_lds(lds_base + fx.Int64(tid * fx.Int32(epr) + fx.Int32(e)) * fx.Int64(4), run)
                run = run + h
                if bid == fx.Int32(0):
                    cum[tid * fx.Int32(epr) + fx.Int32(e)] = run
            if bid == fx.Int32(0):
                cnt[tid] = run
        fx.barrier()
        # This block's tokens for ``g`` take the rows ``base[g] + (what earlier blocks took) ...``:
        # one atomic per (block, g) reserves them, instead of one per token.
        for g in range(tid, fx.Int32(E), fx.Int32(PREP_BLOCK)):
            v = fx.Int32(comm_ops.load_i32_lds(lds_hist + fx.Int64(g) * fx.Int64(4)))
            if v > fx.Int32(0):
                taken = fx.Int32(comm_ops.atomic_add_agent(cursor_addr + fx.Int64(g) * fx.Int64(4), v))
                comm_ops.store_i32_lds(
                    lds_base + fx.Int64(g) * fx.Int64(4),
                    fx.Int32(comm_ops.load_i32_lds(lds_base + fx.Int64(g) * fx.Int64(4))) + taken,
                )
        fx.barrier()

        # ---- plan 3: place every token
        for t in range(gtid, tokens, fx.Int32(nthreads)):
            f = firsts(t)
            for dd in range_constexpr(npes):
                pos = fx.Int32(-1)
                if f[dd] < fx.Int32(epr):
                    g = fx.Int32(dd * epr) + f[dd]
                    slot = fx.Int32(comm_ops.atomic_add_lds(lds_cursor + fx.Int64(g) * fx.Int64(4), fx.Int32(1)))
                    pos = fx.Int32(comm_ops.load_i32_lds(lds_base + fx.Int64(g) * fx.Int64(4))) + slot
                    row_tok[fx.Int32(dd * cap) + pos] = t
                rowpos[t * fx.Int32(npes) + fx.Int32(dd)] = pos
        tree_barrier(fx.Int64(arg_bar) + fx.Int64(bar2_off), fx.Int32(nblocks), tid, bid)
        if bid == fx.Int32(0):
            # Nobody reads hist or cursor any more: leave them zero for the next call.
            for g in range(tid, fx.Int32(E), fx.Int32(PREP_BLOCK)):
                hist[g] = fx.Int32(0)
                cursor[g] = fx.Int32(0)

        def cdone_slot(d, c):
            return cdone_addr + fx.Int64((d * fx.Int32(arena.early_chunks) + c) * fx.Int32(CDONE_STRIDE * 4))

        if bid == fx.Int32(0):
            peer = wave
            if (peer < fx.Int32(npes)) & (peer != fx.Int32(rank)):
                n = cnt[peer]
                n_chunks = (n + fx.Int32(CR - 1)) // fx.Int32(CR)
                early = (n_chunks < fx.Int32(arena.early_chunks)).select(n_chunks, fx.Int32(arena.early_chunks))
                for c in range(fx.Int32(0), early, fx.Int32(1)):
                    c = fx.Int32(c)
                    rows = (n - c * fx.Int32(CR) < fx.Int32(CR)).select(n - c * fx.Int32(CR), fx.Int32(CR))
                    if lane == fx.Int32(0):
                        comm_ops.spin_until_ge_i32_agent(cdone_slot(peer, c), rows)
                        # Every row of the chunk has counted: leave the counter zero for the next call.
                        comm_ops.store_i32_global_agent_release(cdone_slot(peer, c), fx.Int32(0))
                    comm_ops.fence_agent_acquire()
                    if lane == fx.Int32(0):
                        put_chunk(
                            cco.DevComm(arg_dev_comm).sdma(), arg_win, peer, c, n,
                            rank=rank, arena=arena, cco=cco,
                        )
        else:
            if const_expr(route_push):
                pair_order = ptr_buf_tensor(arg_pair_order, fx.Int32)
                weights = ptr_buf_tensor(arg_wts, fx.Float32)
                w = cco.Window(arg_win)
                for ge in range(bid - fx.Int32(1), fx.Int32(E), fx.Int32(quant_blocks)):
                    ge = fx.Int32(ge)
                    dst = ge // fx.Int32(epr)
                    n = ptr_buf_tensor(arg_local_hist, fx.Int32)[ge]
                    base = ptr_buf_tensor(arg_pair_base, fx.Int32)[ge]
                    row0 = ptr_buf_tensor(arg_my_base, fx.Int32)[ge]
                    rowmap = fx.Int64(w.lsa_ptr(dst, arena.off_rowmap))
                    wts_p = fx.Int64(w.lsa_ptr(dst, arena.off_wts))
                    srcmap = fx.Int64(w.lsa_ptr(dst, arena.off_srcmap))
                    flag = fx.Int64(w.lsa_ptr(dst, arena.off_flags)) + fx.Int64(rank * 4)
                    # A row for myself is read from my own S, one for another rank from its R.
                    landing = (dst == fx.Int32(rank)).select(
                        fx.Int32(rank * cap), fx.Int32(arena.n_rows + rank * cap)
                    )
                    for row in range(tid, n, fx.Int32(PREP_BLOCK)):
                        wk = pair_order[base + row]
                        token = wk // fx.Int32(topk)
                        slot = wk - token * fx.Int32(topk)
                        at = fx.Int64(row0 + row) * fx.Int64(4)
                        weight_bits = fx.Vector.from_elements([weights[wk]], fx.Float32).bitcast(fx.Int32)[0]
                        comm_ops.store_i32_global_system_monotonic(
                            rowmap + at, landing + rowpos[token * fx.Int32(npes) + dst]
                        )
                        comm_ops.store_i32_global_system_monotonic(wts_p + at, weight_bits)
                        comm_ops.store_i32_global_system_monotonic(
                            srcmap + at, (fx.Int32(rank * cap) + token) | (slot << fx.Int32(24))
                        )
                    if tid == fx.Int32(0):
                        chunks = (cum[ge] + fx.Int32(CR - 1)) // fx.Int32(CR)
                        comm_ops.store_i32_global_system_monotonic(
                            fx.Int64(w.lsa_ptr(dst, arena.off_need))
                            + fx.Int64(rank * epr + (ge - dst * fx.Int32(epr))) * fx.Int64(4),
                            chunks,
                        )
                    rocdl.s_waitcnt(vmcnt=0)
                    fx.barrier()
                    if tid == fx.Int32(0):
                        comm_ops.fence_system_release()
                        comm_ops.atomic_add_system(flag, fx.Int32(1))

            x_rsrc = create_buffer_resource_from_addr(fx.Int64(arg_x))
            s_rsrc = create_buffer_resource_from_addr(
                fx.Int64(arg_s), num_records_bytes=arena.n_rows * rb
            )
            max_remote = fx.Int32(0)
            for d in range_constexpr(npes):
                if d != rank:
                    max_remote = (cnt[fx.Int32(d)] > max_remote).select(cnt[fx.Int32(d)], max_remote)
            rounds = (max_remote + fx.Int32(CR - 1)) // fx.Int32(CR)
            remote_items = rounds * fx.Int32(per_round)
            total = remote_items + cnt[fx.Int32(rank)]
            waves = fx.Int32(quant_blocks * PREP_WAVES)
            first = (bid - fx.Int32(1)) * fx.Int32(PREP_WAVES) + wave

            for i in range(first, total, waves):
                i = fx.Int32(i)
                is_local = i >= remote_items
                c = i // fx.Int32(per_round)
                rem = i - c * fx.Int32(per_round)
                dd = rem // fx.Int32(CR)
                d = is_local.select(fx.Int32(rank), (fx.Int32(rank + 1) + dd) % fx.Int32(npes))
                row = is_local.select(i - remote_items, c * fx.Int32(CR) + (rem - dd * fx.Int32(CR)))
                if row < cnt[d]:
                    t = row_tok[d * fx.Int32(cap) + row]
                    s_row = (d * fx.Int32(cap) + row) * fx.Int32(rb)
                    for g in range(lane, fx.Int32(groups), fx.Int32(64)):
                        g = fx.Int32(g)
                        _quant_group(
                            x_rsrc,
                            t * fx.Int32(hidden // 2) + g * fx.Int32(GROUP // 2),
                            s_rsrc,
                            s_row + g * fx.Int32(GROUP),
                            s_row + fx.Int32(hidden) + g,
                        )
                    # Written through: once every lane's stores have retired the row is in
                    # memory, which is the release the copy engine needs.
                    rocdl.s_waitcnt(vmcnt=0)
                    if (lane == fx.Int32(0)) & (is_local == fx.Int32(0)) & (row < fx.Int32(arena.early_chunks * CR)):
                        comm_ops.atomic_add_agent(cdone_slot(d, row // fx.Int32(CR)), fx.Int32(1))

    # fmt: off
    @flyc.jit
    def launch(arg_ids: fx.Int64, arg_x: fx.Int64, arg_row_tok: fx.Int64, arg_rowpos: fx.Int64,
               arg_cnt: fx.Int64, arg_cum: fx.Int64, arg_cdone: fx.Int64, arg_hist: fx.Int64,
               arg_cursor: fx.Int64, arg_bar: fx.Int64, arg_s: fx.Int64, arg_dev_comm: fx.Int64,
               arg_win: fx.Int64, arg_pair_order: fx.Int64, arg_pair_base: fx.Int64,
               arg_local_hist: fx.Int64, arg_my_base: fx.Int64, arg_wts: fx.Int64,
               tokens: fx.Int32, stream: fx.Stream):
    # fmt: on
        prepare(arg_ids, arg_x, arg_row_tok, arg_rowpos, arg_cnt, arg_cum, arg_cdone, arg_hist,
                arg_cursor, arg_bar, arg_s, arg_dev_comm, arg_win, arg_pair_order, arg_pair_base,
                arg_local_hist, arg_my_base, arg_wts, tokens).launch(
            grid=(nblocks, 1, 1), block=(PREP_BLOCK, 1, 1), stream=stream
        )

    return launch


def build_post_rest(*, rank, arena):
    """Kernel 2: put the chunks ``prepare`` did not, for a caller that has no GEMM1 comm
    workgroup (the tests; ``gemm1_sdma`` does this itself). One workgroup, one wave per peer.
    """
    import mori.cco.device.flydsl as cco

    # fmt: off
    @flyc.kernel(name=f"stage1_sdma_post_rest_p{arena.npes}_r{rank}", known_block_size=[PREP_BLOCK, 1, 1])
    def post(arg_cnt: fx.Int64, arg_dev_comm: fx.Int64, arg_win: fx.Int64):
    # fmt: on
        tid = fx.thread_idx.x
        post_rest_chunks(
            cco.DevComm(arg_dev_comm).sdma(), arg_win, tid // fx.Int32(64), tid % fx.Int32(64),
            ptr_buf_tensor(arg_cnt, fx.Int32), rank=rank, arena=arena, cco=cco,
        )

    @flyc.jit
    def launch(arg_cnt: fx.Int64, arg_dev_comm: fx.Int64, arg_win: fx.Int64, stream: fx.Stream):
        post(arg_cnt, arg_dev_comm, arg_win).launch(grid=(1, 1, 1), block=(PREP_BLOCK, 1, 1), stream=stream)

    return launch


def build_wait(*, rank, arena):
    """Kernel 4: wait until every peer's puts and route rows for this call have landed.

    ``expected[s]`` is the running total of signals source ``s`` has sent here: one per
    chunk of every call so far; ``expected[npes + s]`` the running total of the flag of
    ``push_routes``. Neither counter is ever reset. Then drain my own queues, so the next
    call may rewrite ``S`` and the window may be closed.
    """
    import mori.cco.device.flydsl as cco

    npes = arena.npes

    # fmt: off
    @flyc.kernel(name=f"stage1_sdma_wait_p{npes}_r{rank}", known_block_size=[64, 1, 1])
    def wait(arg_dev_comm: fx.Int64, arg_win: fx.Int64, arg_expected: fx.Int64):
    # fmt: on
        lane = fx.thread_idx.x
        expected = ptr_buf_tensor(arg_expected, fx.Int32)
        if lane < fx.Int32(npes):
            w = cco.Window(arg_win)
            sdma = cco.DevComm(arg_dev_comm).sdma()
            comm_ops.spin_until_ge_i32_system(
                fx.Int64(w.lsa_ptr(rank, arena.off_flags)) + fx.Int64(lane) * fx.Int64(4),
                expected[fx.Int32(npes) + lane],
                acquire=True,
            )
            if lane != fx.Int32(rank):
                sdma.quiet_queue(lane, 0)
                sdma.wait_signal(lane, 0, fx.Int64(expected[lane]))
                comm_ops.fence_system_acquire()

    @flyc.jit
    def launch(arg_dev_comm: fx.Int64, arg_win: fx.Int64, arg_expected: fx.Int64, stream: fx.Stream):
        wait(arg_dev_comm, arg_win, arg_expected).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


class Stage1SdmaPrepare:
    """The new prepare for one rank: plan, route push, quantise into ``S``, put the first chunks of each run.

    Collective constructor (it creates the cco window). ``max_tok`` is the most tokens a
    rank sends per call; ``chunk_rows`` the rows of one put.
    """

    def __init__(
        self,
        comm,
        *,
        experts,
        topk,
        hidden,
        max_tok,
        chunk_rows,
        early_chunks=1,
        quant_blocks=None,
    ):
        import torch
        from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements
        from mori.tensor_utils import from_gpu_ptr

        self.rank, self.npes = comm.rank, comm.nranks
        if experts % self.npes:
            raise ValueError("experts must divide evenly over the ranks")
        self.epr, self.topk, self.hidden, self.max_tok = (
            experts // self.npes,
            topk,
            hidden,
            max_tok,
        )
        self.arena = a = Arena(
            self.npes, max_tok, hidden, chunk_rows, topk, self.epr, early_chunks
        )
        a.check()
        dev = torch.device("cuda", torch.cuda.current_device())
        self.mem = comm.alloc_mem(a.total_bytes)
        self.win = comm.register_window(self.mem.ptr, self.mem.size)
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 1
        self.dev_comm = comm.create_dev_comm(reqs)

        i32 = dict(dtype=torch.int32, device=dev)
        self.rowpos = torch.empty(max_tok * self.npes, **i32)
        self.row_tok = torch.empty(self.npes * max_tok, **i32)
        self.cnt = torch.empty(self.npes, **i32)
        self.cum = torch.empty(self.npes * self.epr, **i32)
        # Left at zero by every call, so zero from the start: chunk counters, the plan's
        # counters; and the grid barrier's, which are never reset.
        self.cdone = torch.zeros(self.npes * a.cdone_words, **i32)
        self.hist = torch.zeros(experts, **i32)
        self.cursor = torch.zeros(experts, **i32)
        # One block per CU for the quant waves, less the comm workgroup.
        n_cu = torch.cuda.get_device_properties(dev).multi_processor_count
        self.quant_blocks = n_cu - 1 if quant_blocks is None else quant_blocks
        n_groups = (self.quant_blocks + 1 + BAR_GROUP - 1) // BAR_GROUP
        self.bar = torch.zeros(2 * (n_groups + 1) * BAR_STRIDE, **i32)
        # Running totals the wait kernel waits for: signals per source, then route flags.
        self.expected = torch.zeros(2 * self.npes, **i32)
        # The signal totals as they were before this call: its chunks count from here.
        self.sig_base = torch.zeros(self.npes, **i32)
        # Zero flags: peers bump them as soon as they run.
        from_gpu_ptr(self.mem.ptr + a.off_flags, (self.npes,), torch.int32).zero_()
        self.rowmap = from_gpu_ptr(
            self.mem.ptr + a.off_rowmap, (a.max_routes,), torch.int32
        )
        self.wts = from_gpu_ptr(self.mem.ptr + a.off_wts, (a.max_routes,), torch.int32)
        self.srcmap = from_gpu_ptr(
            self.mem.ptr + a.off_srcmap, (a.max_routes,), torch.int32
        )
        self._prepares = {}
        self._post_rest = build_post_rest(rank=self.rank, arena=a)
        self._wait = build_wait(rank=self.rank, arena=a)
        torch.cuda.synchronize()
        comm.barrier()

    def run(self, x, ids, wts=None, plan=None, stream=None):
        """Plan, push the routes, quantise, and put the first ``early_chunks`` of every run, for ``x`` bf16
        ``[T, hidden]`` and ``ids`` int32 ``[T, topk]``. The other chunks follow from
        ``post_rest`` or the GEMM1 kernel.

        ``plan`` = ``(pair_order, pair_base, local_hist, my_base)``, the compact plan's tables
        (``MegaMoEV2._s1_dispatch_workspace``), and ``wts`` f32 ``[T, topk]`` give the route
        table that is pushed to the destinations; without them there is none.
        """
        import torch
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        tokens = int(x.shape[0])
        if tokens > self.max_tok:
            raise ValueError(f"tokens={tokens} > max_tok={self.max_tok}")
        route_push = plan is not None
        if route_push and wts is None:
            raise ValueError("the route table needs the routing weights")
        if route_push not in self._prepares:
            self._prepares[route_push] = build_prepare(
                rank=self.rank,
                arena=self.arena,
                epr=self.epr,
                quant_blocks=self.quant_blocks,
                route_push=route_push,
            )
        s = fx.Stream((stream or torch.cuda.current_stream()).cuda_stream)
        a = self.arena
        tables = [t.data_ptr() for t in plan] if route_push else [0] * 4
        _run_compiled(
            self._prepares[route_push],
            fx.Int64(ids.data_ptr()),
            fx.Int64(x.data_ptr()),
            *(
                fx.Int64(t.data_ptr())
                for t in (
                    self.row_tok,
                    self.rowpos,
                    self.cnt,
                    self.cum,
                    self.cdone,
                    self.hist,
                    self.cursor,
                    self.bar,
                )
            ),
            fx.Int64(self.mem.ptr + a.off_s),
            fx.Int64(self.dev_comm.ptr),
            fx.Int64(self.win.handle),
            *(fx.Int64(p) for p in tables),
            fx.Int64(wts.data_ptr() if route_push else 0),
            fx.Int32(tokens),
            s,
        )

    def post_rest(self, stream=None):
        """Put the chunks ``run`` left (all but the first ``early_chunks``). Not needed when the GEMM1 kernel's
        comm workgroup does it (``run_gemm1_sdma`` with ``prep``)."""
        import torch
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        s = fx.Stream((stream or torch.cuda.current_stream()).cuda_stream)
        _run_compiled(
            self._post_rest,
            fx.Int64(self.cnt.data_ptr()),
            fx.Int64(self.dev_comm.ptr),
            fx.Int64(self.win.handle),
            s,
        )

    def arm(self, chunks_from, routes=True):
        """Bump the running totals for one call without waiting: ``chunks_from[s]`` = chunks
        source ``s`` sends me. A kernel that waits on the device (``gemm1_sdma`` with ``prep``)
        reads ``sig_base`` (where this call's signals start) and ``expected``."""
        import torch

        self.sig_base.copy_(self.expected[: self.npes])
        self.expected[: self.npes] += torch.as_tensor(
            chunks_from, dtype=torch.int32, device=self.expected.device
        )
        if routes:
            self.expected[self.npes :] += self.epr

    def wait(self, chunks_from, routes=True, stream=None):
        """Wait for the puts (and, with ``routes``, the route rows) of one call; ``chunks_from[s]`` = chunks source ``s`` sends me."""
        import torch
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        self.arm(chunks_from, routes)
        s = fx.Stream((stream or torch.cuda.current_stream()).cuda_stream)
        _run_compiled(
            self._wait,
            fx.Int64(self.dev_comm.ptr),
            fx.Int64(self.win.handle),
            fx.Int64(self.expected.data_ptr()),
            s,
        )

    def close(self):
        import torch

        torch.cuda.synchronize()  # the queues outlive the dev comm: drain first
        for h in (self.dev_comm, self.win, self.mem):
            if h is not None:
                h.close()
        self.dev_comm = self.win = self.mem = None
