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
"""Quantise-and-dispatch as one op, with the transfer on the cco SDMA copy engines.

The MoE dispatch of an EP layer, as a standalone op so it can be put next to mori's EPv2
(``EpDispatchCombineOp``): take bf16 tokens and a routing, deliver every token to each rank
that routes to it, quantised to fp8 with one e8m0 per 32 values.

What is different from a CU dispatch is WHO moves the bytes. EPv2 quantises with the CUs and
then copies with the CUs, two passes over the data that cannot overlap. Here the CUs only
quantise -- into a send buffer ``S`` grouped by destination rank -- and the copy engines ship
each run while the CUs are still quantising the rest of it. The cost is therefore about
``max(quant, transfer)`` instead of ``quant + transfer``, and the CUs are free for the next
thing in the layer as soon as the quant is done.

Four things make that pipeline possible:

* **A run is ordered by block, not by expert.** Block ``b`` owns a contiguous slice of the
  tokens and its rows sit together in every run, so quantising the tokens in order fills a
  PREFIX of every run and a chunk of it can be put while the rest is still being written.
* **The payload moves by copy engine, the routing by store.** The ids, the weights and the
  source id are ~1% of the bytes and are known as soon as the plan is, so they go straight
  into the destination as write-through stores, spread over the transfer rather than added
  to it. The CUs that write them are a different agent from the copy engines, so they get
  their own marker -- a remote atomic, issued once those stores have retired. The receiver
  waits on both.
* **The landing place is dense.** A sender takes ``ceil(count / chunk_rows)`` chunks off the
  destination's allocator with one atomic, so every source's rows sit together from row 0 and
  the consumer reads ``nrecv`` rows, not the worst case. A ``cons`` handshake at the top of a
  call keeps a rank that is one call ahead from writing over rows the owner has not read.
* **Nothing lives on the host.** The signal totals and the call parity are device-side and the
  kernel advances them itself, so ``dispatch()`` is one launch with fixed arguments: capture
  it in a CUDA graph and replay it.

``dispatch()`` returns once every peer's rows have landed -- the wait is fused into the same
kernel, since by then the quant blocks have exited and there is nothing left to overlap it
with. The receive arrays are laid out for a fused-MoE GEMM1 to read directly. Needs
``MORI_ENABLE_SDMA=1`` and a mori built with SDMA.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int8, ReductionOp, T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels.buffer_ops import buffer_load
from aiter.ops.flydsl.kernels.mega_moe.quant import (
    _FP8_E4M3_INV_MAX_POS_BITS,
    GROUP,
)
from aiter.ops.flydsl.kernels.tensor_shim import ptr_buf_tensor


def _align(n, a):
    return (n + a - 1) // a * a


#: Threads of a block, and waves in it: one wave per peer in the comm workgroup.
PREP_BLOCK = 512
PREP_WAVES = PREP_BLOCK // 64
#: Row stores go through to memory (sc0|sc1), so retiring them is the release a copy engine
#: needs, and a remote store is at the destination once it has retired.
SEND_CM = 1 | 16
BAR_STRIDE = 32  # ints per tree_barrier counter: one 128 B cache line
BAR_GROUP = 32  # blocks per tree_barrier group
#: i32 slots between two chunk counters: one 128 B cache line each, because every row of a
#: chunk bumps its counter.
CDONE_STRIDE = 32


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


def _quantize_group(x_rsrc, in_dword):
    """One 32-value group of a bf16 row -> its e8m0 scale and its 32 fp8 values (8 i32 words).

    The numbers of ``quant.emit_per_1x32_mx_fp8_group``. The row is quantised ONCE and then
    stored to every rank that gets it (``_store_group``).
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
    quant_scale = ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(fx.Float32)
    scaled = [act[k] * quant_scale for k in range_constexpr(GROUP)]
    words = []
    for word in range_constexpr(8):
        base = word * 4
        packed = rocdl.cvt_pk_fp8_f32(
            T.i32, scaled[base], scaled[base + 1], fx.Int32(0), 0
        )
        packed = rocdl.cvt_pk_fp8_f32(
            T.i32, scaled[base + 2], scaled[base + 3], packed, 1
        )
        words.append(packed)
    return e8m0, words


@dataclass(frozen=True)
class Arena:
    """Byte offsets inside the cco window, identical on every rank::

        [ S_payload | S_scale |
          R_payload | R_scale | R_ids | R_wts | R_src | meta | cons | alloc | nrecv ]

    The receive side is laid out the way a fused-MoE GEMM1 wants to read it: the payload is
    ``[rows, K]`` fp8 with a row stride of exactly ``K``, the scales ``[rows, K/32]``, and the
    routing ``[rows, topk]``, each its own contiguous array. That is why payload and scale are
    separate regions rather than one packed row -- a packed row would need the consumer to
    stride over the scales.

    The routing goes to the destination as plain write-through stores, not through the copy
    engines: it is ~1% of the bytes and known from the plan, so spreading it over the transfer
    costs nothing. A remote ATOMIC per peer per call is what orders it -- issued once every
    row's stores have retired, and from the same agent, so the destination cannot see it
    first. The SDMA signal orders the payload; the receiver waits on both.

    Nothing stops a rank that is a hair ahead from writing the next call's rows into a
    destination that has not read this call's yet -- and with a dense layout those rows land
    somewhere else, so it corrupts rather than repeats. ``cons`` closes that: at the top of a
    call every rank publishes its call number to its peers and waits for theirs, so no rank
    writes into a buffer whose owner has not started the same call.

    ``R`` is packed at CHUNK granularity: a sender takes ``ceil(count / chunk_rows)`` chunks
    off the destination's ``alloc`` counter with one atomic, so the rows of every source sit
    together from row 0 and the consumer sees ``nrecv`` rows instead of the worst case. The
    tail of a source's last chunk is padding; its ``ids`` are set to -1 so the consumer's
    sorting skips it. ``alloc`` is two counters picked by the call parity, which is one more
    call of slack than the dispatch/wait pairing can drift by.
    """

    npes: int
    cap: int
    hidden: int
    topk: int
    chunk_rows: int

    @property
    def scale_bytes(self):
        return self.hidden // GROUP

    @property
    def send_rows(self):
        return self.npes * self.cap

    @property
    def recv_rows(self):
        """Worst case: every source fills its cap, rounded up to whole chunks."""
        per = (self.cap + self.chunk_rows - 1) // self.chunk_rows * self.chunk_rows
        return self.npes * per

    @property
    def off_s_pay(self):
        return 0

    @property
    def off_s_sc(self):
        return self.off_s_pay + self.send_rows * self.hidden

    @property
    def off_r_pay(self):
        return self.off_s_sc + self.send_rows * self.scale_bytes

    @property
    def off_r_sc(self):
        return self.off_r_pay + self.recv_rows * self.hidden

    @property
    def off_r_ids(self):
        return self.off_r_sc + self.recv_rows * self.scale_bytes

    @property
    def off_r_wts(self):
        return self.off_r_ids + self.recv_rows * self.topk * 4

    @property
    def off_r_src(self):
        """``src_rank * cap + token`` of each row: what the combine sends back along."""
        return self.off_r_wts + self.recv_rows * self.topk * 4

    @property
    def off_meta(self):
        """``meta[s]`` = the call whose routing source ``s`` has finished storing here. The
        routing is written by ``s``'s CUs, the payload by its copy engines; one agent's write
        says nothing about the other's, so the routing gets its own marker -- a remote atomic,
        which the destination cannot see before that CU's own stores have retired."""
        return self.off_r_src + self.recv_rows * 4

    @property
    def off_cons(self):
        """``cons[s]`` = the call number source ``s`` has started, so it has finished reading
        the call before it. A sender waits for this before it writes anything of mine.
        """
        return self.off_meta + _align(self.npes * 4, 128)

    @property
    def off_alloc(self):
        return self.off_cons + _align(self.npes * 4, 128)

    @property
    def off_nrecv(self):
        return self.off_alloc + 128

    @property
    def total_bytes(self):
        return self.off_nrecv + 128

    @property
    def max_chunks(self):
        return (self.cap + self.chunk_rows - 1) // self.chunk_rows

    def check(self):
        if self.hidden % 256:
            raise ValueError(f"hidden must be a multiple of 256, got {self.hidden}")
        if self.chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        # A chunk must start on a 128 B line in both regions, so that a landed chunk can be
        # read while the next one is still arriving without pulling a stale line in with it.
        for n, w in (("payload", self.hidden), ("scale", self.scale_bytes)):
            if self.chunk_rows * w % 128:
                raise ValueError(
                    f"chunk_rows={self.chunk_rows} leaves the {n} chunks unaligned "
                    f"({self.chunk_rows} * {w} is not a multiple of 128)"
                )
        if self.off_nrecv >= 1 << 32:
            raise ValueError("the window exceeds the 4 GiB buffer-resource range")


def _store_split(pay, sca, e8m0, words, out_byte, scale_byte):
    """One quantised group into the two send regions. Written through (``SEND_CM``), so
    retiring the stores is the release the copy engine needs."""
    buffer_store(
        e8m0.to(fx.Int8), sca, scale_byte, cache_modifier=SEND_CM, offset_is_bytes=True
    )
    for half in range_constexpr(2):
        buffer_store(
            fx.Vector.from_elements(words[half * 4 : half * 4 + 4], fx.Int32),
            pay,
            out_byte + fx.Int32(half * 16),
            cache_modifier=SEND_CM,
            offset_is_bytes=True,
        )


def _put_chunk(sdma, win, peer, dst_row, lo, hi, *, arena, cco, signal):
    """Rows ``[lo, hi)`` of my run for ``peer`` -- payload and scales -- into the rows starting
    at ``dst_row`` of its receive arrays. Only the last chunk of a call signals: a queue is a
    FIFO its engine drains in order, so that one signal covers everything sent this call.
    """
    n = fx.Int64(hi - lo)
    sdma.put(
        peer,
        win,
        fx.Int64(arena.off_r_pay) + fx.Int64(dst_row) * fx.Int64(arena.hidden),
        win,
        fx.Int64(arena.off_s_pay)
        + (fx.Int64(peer) * fx.Int64(arena.cap) + fx.Int64(lo))
        * fx.Int64(arena.hidden),
        n * fx.Int64(arena.hidden),
        0,
        coop=cco.CoopScope.THREAD,
        local_signal=False,
        remote_signal=False,
    )
    sdma.put(
        peer,
        win,
        fx.Int64(arena.off_r_sc) + fx.Int64(dst_row) * fx.Int64(arena.scale_bytes),
        win,
        fx.Int64(arena.off_s_sc)
        + (fx.Int64(peer) * fx.Int64(arena.cap) + fx.Int64(lo))
        * fx.Int64(arena.scale_bytes),
        n * fx.Int64(arena.scale_bytes),
        0,
        coop=cco.CoopScope.THREAD,
        local_signal=False,
        remote_signal=signal,
    )


def build_dispatch(*, rank, arena, epr, quant_blocks):
    """One resident kernel: plan, claim room at the destinations, quantise, put, and wait.

    **Plan** (every block). Block ``b`` owns the token slice ``[b*T/nb, (b+1)*T/nb)``, counts
    how many of its tokens go to each destination into ``block_cnt[b]``, and after a grid
    rendezvous sums the blocks before it to get its base in every run. Its tokens then take
    consecutive rows from that base, so a run ends up in block order and quantising the tokens
    in order fills a PREFIX of it -- which is what lets a chunk go out while the rest is still
    being written.

    **Claim** (block 0, between the plan's two rendezvous, so it costs no extra barrier). Wave
    ``p`` takes ``ceil(cnt[p] / chunk_rows)`` chunks off peer ``p``'s ``alloc`` counter with one
    remote atomic and keeps the base in ``peer_base[p]``. The second rendezvous publishes it.

    **Quantise** (blocks 1..n). A wave takes one TOKEN at a time: its bf16 row is read and
    quantised once and written to that token's row of ``S_payload`` / ``S_scale`` for every run
    it is in. The routing goes straight to the destination as plain write-through stores --
    it is known from the plan, it is ~1% of the bytes, and spreading it over the transfer is
    cheaper than two more puts per peer. Each row then bumps its chunk's counter.

    **Put and wait** (block 0). Wave ``p`` walks its chunks in order, waits for the counter,
    and posts the payload and the scales of the chunk; the last one carries the remote signal.
    Then it waits for peer ``p``'s own signal and drains its queue. The wait lives here because
    there is nothing to overlap it with, and the quant blocks have already exited by then, so
    the CUs are free while the copy engines finish.
    """
    import mori.cco.device.flydsl as cco

    npes, cap, hidden, topk, CR = (
        arena.npes,
        arena.cap,
        arena.hidden,
        arena.topk,
        arena.chunk_rows,
    )
    arena.check()
    if npes > PREP_WAVES:
        raise ValueError(f"one comm wave per peer needs npes <= {PREP_WAVES}")
    sc_b = arena.scale_bytes
    groups = hidden // GROUP
    nblocks = quant_blocks + 1
    n_bar_groups = (nblocks + BAR_GROUP - 1) // BAR_GROUP
    bar2_off = (n_bar_groups + 1) * BAR_STRIDE * 4
    group_rounds = (groups + 63) // 64
    maxc = arena.max_chunks

    @fx.struct
    class Shared:
        buf: fx.Array[Int8, _align(npes * 4, 16), 16]

    # fmt: off
    @flyc.kernel(name=f"sdma_dispatch_p{npes}_r{rank}_h{hidden}_k{topk}_c{cap}_cr{CR}_q{quant_blocks}",
                 known_block_size=[PREP_BLOCK, 1, 1])
    def kernel(arg_x: fx.Int64, arg_ids: fx.Int64, arg_wts: fx.Int64, arg_pos: fx.Int64,
               arg_block_cnt: fx.Int64, arg_cnt: fx.Int64, arg_cdone: fx.Int64,
               arg_bar: fx.Int64, arg_base: fx.Int64, arg_parity: fx.Int64,
               arg_dev_comm: fx.Int64, arg_win: fx.Int64, arg_local: fx.Int64,
               tokens: fx.Int32):
    # fmt: on
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        wave = tid // fx.Int32(64)
        lane = tid % fx.Int32(64)
        ids = ptr_buf_tensor(arg_ids, fx.Int32)
        wts = ptr_buf_tensor(arg_wts, fx.Int32)  # f32 bits, copied verbatim
        pos = ptr_buf_tensor(arg_pos, fx.Int32)
        block_cnt = ptr_buf_tensor(arg_block_cnt, fx.Int32)
        cnt = ptr_buf_tensor(arg_cnt, fx.Int32)
        peer_base = ptr_buf_tensor(arg_base, fx.Int32)
        cdone_addr = fx.Int64(arg_cdone)
        lds = fx.SharedAllocator().allocate(Shared).peek()
        lds_base = fx.Int64(fx.ptrtoint(lds.buf.ptr))
        w = cco.Window(arg_win)
        # This call's number; the slot it claims in is its parity, so the call after next
        # reuses a counter this call has already zeroed.
        callno = ptr_buf_tensor(arg_parity, fx.Int32)[fx.Int32(0)] + fx.Int32(1)
        parity = callno & fx.Int32(1)

        per = (tokens + fx.Int32(nblocks - 1)) // fx.Int32(nblocks)
        t0 = bid * per
        t1 = ((t0 + per) < tokens).select(t0 + per, tokens)

        def goes_to(t, d):
            """Does token ``t`` route to rank ``d``? (any of its top-k experts live there)"""
            hit = fx.Int32(0) == fx.Int32(1)
            for j in range_constexpr(topk):
                g = ids[t * fx.Int32(topk) + fx.Int32(j)]
                hit = hit | ((g >= fx.Int32(0)) & (g // fx.Int32(epr) == d))
            return hit

        # ---- plan 1: this block's count per destination
        for d in range(tid, fx.Int32(npes), fx.Int32(PREP_BLOCK)):
            comm_ops.store_i32_lds(lds_base + fx.Int64(d) * fx.Int64(4), fx.Int32(0))
        fx.barrier()
        for t in range(t0 + tid, t1, fx.Int32(PREP_BLOCK)):
            for dd in range_constexpr(npes):
                if goes_to(fx.Int32(t), fx.Int32(dd)):
                    comm_ops.atomic_add_lds(lds_base + fx.Int64(dd * 4), fx.Int32(1))
        fx.barrier()
        if tid < fx.Int32(npes):
            block_cnt[bid * fx.Int32(npes) + tid] = fx.Int32(
                comm_ops.load_i32_lds(lds_base + fx.Int64(tid) * fx.Int64(4))
            )
        tree_barrier(fx.Int64(arg_bar), fx.Int32(nblocks), tid, bid)

        # ---- plan 2: this block's base in every run, the run totals, and the claim
        if tid < fx.Int32(npes):
            base = fx.Int32(0)
            total = fx.Int32(0)
            for b in range(fx.Int32(0), fx.Int32(nblocks), fx.Int32(1)):
                v = block_cnt[fx.Int32(b) * fx.Int32(npes) + tid]
                if fx.Int32(b) < bid:
                    base = base + v
                total = total + v
            comm_ops.store_i32_lds(lds_base + fx.Int64(tid) * fx.Int64(4), base)
            if bid == fx.Int32(0):
                cnt[tid] = total
                # Say I have started this call (so I am done reading the last one), then wait
                # for every destination to say the same: writing into one that has not would
                # land on rows it has not read.
                comm_ops.store_i32_global_system_release(
                    fx.Int64(w.lsa_ptr(tid, arena.off_cons)) + fx.Int64(rank * 4), callno
                )
                comm_ops.spin_until_ge_i32_system(
                    fx.Int64(w.lsa_ptr(rank, arena.off_cons)) + fx.Int64(tid) * fx.Int64(4),
                    callno,
                    acquire=True,
                )
                # One remote atomic per destination takes whole chunks of its receive
                # arrays, so the sources pack together from row 0.
                nc = (total + fx.Int32(CR - 1)) // fx.Int32(CR)
                got = fx.Int32(
                    comm_ops.atomic_add_system(
                        fx.Int64(w.lsa_ptr(tid, arena.off_alloc))
                        + fx.Int64(parity) * fx.Int64(4),
                        nc,
                    )
                )
                peer_base[tid] = got * fx.Int32(CR)
        fx.barrier()

        # ---- plan 3: place this block's tokens, in token order within the block
        for t in range(t0, t1, fx.Int32(1)):
            t = fx.Int32(t)
            if tid < fx.Int32(npes):
                p = fx.Int32(-1)
                if goes_to(t, tid):
                    p = fx.Int32(comm_ops.load_i32_lds(lds_base + fx.Int64(tid) * fx.Int64(4)))
                    comm_ops.store_i32_lds(
                        lds_base + fx.Int64(tid) * fx.Int64(4), p + fx.Int32(1)
                    )
                pos[t * fx.Int32(npes) + tid] = p
            fx.barrier()
        tree_barrier(fx.Int64(arg_bar) + fx.Int64(bar2_off), fx.Int32(nblocks), tid, bid)

        def cdone_slot(d, c):
            return cdone_addr + fx.Int64((d * fx.Int32(maxc) + c) * fx.Int32(CDONE_STRIDE * 4))

        if bid == fx.Int32(0):
            peer = wave
            sdma = cco.DevComm(arg_dev_comm).sdma()
            if peer < fx.Int32(npes):
                n = cnt[peer]
                n_chunks = (n + fx.Int32(CR - 1)) // fx.Int32(CR)
                dst0 = peer_base[peer]
                # The tail of my last chunk is padding at the destination: -1 says there is
                # no route there. Written where it is read, like the rest of the routing.
                d_ids = create_buffer_resource_from_addr(
                    fx.Int64(w.lsa_ptr(peer, arena.off_r_ids)),
                    num_records_bytes=arena.recv_rows * topk * 4,
                )
                d_src = create_buffer_resource_from_addr(
                    fx.Int64(w.lsa_ptr(peer, arena.off_r_src)),
                    num_records_bytes=arena.recv_rows * 4,
                )
                for r in range(n + lane, n_chunks * fx.Int32(CR), fx.Int32(64)):
                    r = fx.Int32(r)
                    for j in range_constexpr(topk):
                        buffer_store(
                            fx.Int32(-1), d_ids,
                            ((dst0 + r) * fx.Int32(topk) + fx.Int32(j)) * fx.Int32(4),
                            cache_modifier=SEND_CM, offset_is_bytes=True,
                        )
                    buffer_store(fx.Int32(-1), d_src, (dst0 + r) * fx.Int32(4),
                                 cache_modifier=SEND_CM, offset_is_bytes=True)
                # This wave's own stores have retired; the quant waves' have too, because
                # each drains its own before it counts its row into the chunk below.
                rocdl.s_waitcnt(vmcnt=0)
                if peer != fx.Int32(rank):
                    if lane == fx.Int32(0):
                        for c in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                            c = fx.Int32(c)
                            lo = c * fx.Int32(CR)
                            hi = (lo + fx.Int32(CR) < n).select(lo + fx.Int32(CR), n)
                            comm_ops.spin_until_ge_i32_agent(cdone_slot(peer, c), hi - lo)
                            comm_ops.store_i32_global_agent_release(
                                cdone_slot(peer, c), fx.Int32(0)
                            )
                            comm_ops.fence_agent_acquire()
                            if c == n_chunks - fx.Int32(1):
                                _put_chunk(sdma, arg_win, peer, dst0 + lo, lo, hi,
                                           arena=arena, cco=cco, signal=True)
                            else:
                                _put_chunk(sdma, arg_win, peer, dst0 + lo, lo, hi,
                                           arena=arena, cco=cco, signal=False)
                        # Every row's routing is stored and retired by now (the last chunk's
                        # counter says so), so this atomic cannot reach the peer before them.
                        comm_ops.atomic_add_system(
                            fx.Int64(w.lsa_ptr(peer, arena.off_meta)) + fx.Int64(rank * 4),
                            fx.Int32(1),
                        )
                    # Nothing overlaps the wait, so it lives here: the quant blocks have
                    # retired and the CUs are free while the engines finish.
                    if lane == fx.Int32(0):
                        e = ptr_buf_tensor(arg_local, fx.Int32)[peer] + fx.Int32(1)
                        ptr_buf_tensor(arg_local, fx.Int32)[peer] = e
                        sdma.quiet_queue(peer, 0)
                        sdma.wait_signal(peer, 0, fx.Int64(e))  # the payload
                        comm_ops.spin_until_ge_i32_system(  # and the routing
                            fx.Int64(w.lsa_ptr(rank, arena.off_meta))
                            + fx.Int64(peer) * fx.Int64(4),
                            callno,
                            acquire=True,
                        )
            fx.barrier()
            comm_ops.fence_system_acquire()
            if tid == fx.Int32(0):
                # How many rows landed, and the counters set up for the call after next.
                alloc = fx.Int64(w.lsa_ptr(rank, arena.off_alloc)) + fx.Int64(parity) * fx.Int64(4)
                rows = fx.Int32(comm_ops.load_i32_global_system(alloc)) * fx.Int32(CR)
                comm_ops.store_i32_global_system_monotonic(
                    fx.Int64(w.lsa_ptr(rank, arena.off_nrecv)), rows
                )
                comm_ops.store_i32_global_system_monotonic(alloc, fx.Int32(0))
                ptr_buf_tensor(arg_parity, fx.Int32)[fx.Int32(0)] = callno
        else:
            x_rsrc = create_buffer_resource_from_addr(fx.Int64(arg_x))
            pay = create_buffer_resource_from_addr(
                fx.Int64(w.lsa_ptr(rank, arena.off_s_pay)),
                num_records_bytes=arena.send_rows * hidden,
            )
            sca = create_buffer_resource_from_addr(
                fx.Int64(w.lsa_ptr(rank, arena.off_s_sc)),
                num_records_bytes=arena.send_rows * sc_b,
            )
            # My own rows are never put, so they go straight into the receive arrays and the
            # consumer sees one layout over every source.
            r_pay = create_buffer_resource_from_addr(
                fx.Int64(w.lsa_ptr(rank, arena.off_r_pay)),
                num_records_bytes=arena.recv_rows * hidden,
            )
            r_sca = create_buffer_resource_from_addr(
                fx.Int64(w.lsa_ptr(rank, arena.off_r_sc)),
                num_records_bytes=arena.recv_rows * sc_b,
            )
            d_ids = [
                create_buffer_resource_from_addr(
                    fx.Int64(w.lsa_ptr(fx.Int32(dd), arena.off_r_ids)),
                    num_records_bytes=arena.recv_rows * topk * 4,
                )
                for dd in range_constexpr(npes)
            ]
            d_wts = [
                create_buffer_resource_from_addr(
                    fx.Int64(w.lsa_ptr(fx.Int32(dd), arena.off_r_wts)),
                    num_records_bytes=arena.recv_rows * topk * 4,
                )
                for dd in range_constexpr(npes)
            ]
            d_src = [
                create_buffer_resource_from_addr(
                    fx.Int64(w.lsa_ptr(fx.Int32(dd), arena.off_r_src)),
                    num_records_bytes=arena.recv_rows * 4,
                )
                for dd in range_constexpr(npes)
            ]
            waves = fx.Int32(quant_blocks * PREP_WAVES)
            first = (bid - fx.Int32(1)) * fx.Int32(PREP_WAVES) + wave
            for t in range(first, tokens, waves):
                t = fx.Int32(t)
                ps = [pos[t * fx.Int32(npes) + fx.Int32(dd)] for dd in range_constexpr(npes)]
                any_row = fx.Int32(0) == fx.Int32(1)
                for dd in range_constexpr(npes):
                    any_row = any_row | (ps[dd] >= fx.Int32(0))
                if any_row:
                    q = []
                    for k in range_constexpr(group_rounds):
                        g = lane + fx.Int32(64 * k)
                        gc = (g < fx.Int32(groups)).select(g, fx.Int32(0))
                        q.append(
                            _quantize_group(
                                x_rsrc, t * fx.Int32(hidden // 2) + gc * fx.Int32(GROUP // 2)
                            )
                        )
                    for dd in range_constexpr(npes):
                        if ps[dd] >= fx.Int32(0):
                            if const_expr(dd == rank):
                                dst_p, dst_s = r_pay, r_sca
                                row = peer_base[fx.Int32(rank)] + ps[dd]
                            else:
                                dst_p, dst_s = pay, sca
                                row = fx.Int32(dd * cap) + ps[dd]
                            for k in range_constexpr(group_rounds):
                                g = lane + fx.Int32(64 * k)
                                if g < fx.Int32(groups):
                                    _store_split(
                                        dst_p, dst_s, q[k][0], q[k][1],
                                        row * fx.Int32(hidden) + g * fx.Int32(GROUP),
                                        row * fx.Int32(sc_b) + g,
                                    )
                            # The routing, straight to the destination: it is ~1% of the
                            # bytes and the atomic below is what orders it.
                            mrow = peer_base[fx.Int32(dd)] + ps[dd]
                            m_i, m_w, m_s = d_ids[dd], d_wts[dd], d_src[dd]
                            # Written through like the payload: a copy engine reads memory,
                            # so a cached store could leave it reading the last call's row.
                            if lane < fx.Int32(topk):
                                at = (mrow * fx.Int32(topk) + lane) * fx.Int32(4)
                                buffer_store(ids[t * fx.Int32(topk) + lane], m_i, at,
                                             cache_modifier=SEND_CM, offset_is_bytes=True)
                                buffer_store(wts[t * fx.Int32(topk) + lane], m_w, at,
                                             cache_modifier=SEND_CM, offset_is_bytes=True)
                            if lane == fx.Int32(0):
                                buffer_store(fx.Int32(rank * cap) + t, m_s,
                                             mrow * fx.Int32(4),
                                             cache_modifier=SEND_CM, offset_is_bytes=True)
                    rocdl.s_waitcnt(vmcnt=0)
                    if lane < fx.Int32(npes):
                        p = pos[t * fx.Int32(npes) + lane]
                        if (lane != fx.Int32(rank)) & (p >= fx.Int32(0)):
                            comm_ops.atomic_add_agent(
                                cdone_slot(lane, p // fx.Int32(CR)), fx.Int32(1)
                            )

    # fmt: off
    @flyc.jit
    def launch(arg_x: fx.Int64, arg_ids: fx.Int64, arg_wts: fx.Int64, arg_pos: fx.Int64,
               arg_block_cnt: fx.Int64, arg_cnt: fx.Int64, arg_cdone: fx.Int64,
               arg_bar: fx.Int64, arg_base: fx.Int64, arg_parity: fx.Int64,
               arg_dev_comm: fx.Int64, arg_win: fx.Int64, arg_local: fx.Int64,
               tokens: fx.Int32, stream: fx.Stream):
    # fmt: on
        kernel(arg_x, arg_ids, arg_wts, arg_pos, arg_block_cnt, arg_cnt, arg_cdone, arg_bar,
               arg_base, arg_parity, arg_dev_comm, arg_win, arg_local, tokens).launch(
            grid=(nblocks, 1, 1), block=(PREP_BLOCK, 1, 1), stream=stream
        )

    return launch


class SdmaDispatch:
    """Quantise-and-dispatch over SDMA, as one kernel. Collective constructor (it makes the
    cco window).

    ``dispatch(x, ids, wts)`` returns once every peer's rows are in this rank's receive
    arrays. It is a single kernel launch with fixed arguments and no host-side state, so it
    can be captured in a CUDA graph and replayed: the per-source signal totals and the call
    parity live on the device and the kernel advances them itself.

    The receive side is what a fused-MoE GEMM1 reads, each array contiguous and packed from
    row 0 over all sources: ``payload()`` ``[rows, K]`` fp8, ``scales()`` ``[rows, K/32]``
    e8m0, ``ids()`` / ``weights()`` ``[rows, topk]``, and ``nrecv()`` rows of them. A padding
    row (a source's last chunk is not full) has ``ids`` -1.
    """

    def __init__(
        self, comm, *, experts, topk, hidden, max_tok, chunk_rows, quant_blocks=None
    ):
        import torch
        from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements
        from mori.tensor_utils import from_gpu_ptr

        self.rank, self.npes = comm.rank, comm.nranks
        if experts % self.npes:
            raise ValueError("experts must divide evenly over the ranks")
        self.epr = experts // self.npes
        self.topk, self.hidden, self.max_tok = topk, hidden, max_tok
        self.arena = a = Arena(self.npes, max_tok, hidden, topk, chunk_rows)
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
        n_cu = torch.cuda.get_device_properties(dev).multi_processor_count
        self.quant_blocks = n_cu - 1 if quant_blocks is None else quant_blocks
        nb = self.quant_blocks + 1
        self.pos = torch.empty(max_tok * self.npes, **i32)
        self.block_cnt = torch.zeros(nb * self.npes, **i32)
        self.cnt = torch.zeros(self.npes, **i32)
        self.peer_base = torch.zeros(self.npes, **i32)
        # left at zero by every call; the barrier counters are never reset
        self.cdone = torch.zeros(self.npes * a.max_chunks * CDONE_STRIDE, **i32)
        n_groups = (nb + BAR_GROUP - 1) // BAR_GROUP
        self.bar = torch.zeros(2 * (n_groups + 1) * BAR_STRIDE, **i32)
        # device-side so a graph replay advances them
        self.sig = torch.zeros(self.npes, **i32)
        self.callno = torch.zeros(1, **i32)
        from_gpu_ptr(self.mem.ptr + a.off_meta, (self.npes,), torch.int32).zero_()
        from_gpu_ptr(self.mem.ptr + a.off_cons, (self.npes,), torch.int32).zero_()
        from_gpu_ptr(self.mem.ptr + a.off_alloc, (64,), torch.int32).zero_()
        rows = a.recv_rows
        self._pay = from_gpu_ptr(
            self.mem.ptr + a.off_r_pay, (rows, hidden), torch.float8_e4m3fn
        )
        self._sc = from_gpu_ptr(
            self.mem.ptr + a.off_r_sc, (rows, a.scale_bytes), torch.uint8
        )
        self._ids = from_gpu_ptr(self.mem.ptr + a.off_r_ids, (rows, topk), torch.int32)
        self._wts = from_gpu_ptr(
            self.mem.ptr + a.off_r_wts, (rows, topk), torch.float32
        )
        self._src = from_gpu_ptr(self.mem.ptr + a.off_r_src, (rows,), torch.int32)
        self._nrecv = from_gpu_ptr(self.mem.ptr + a.off_nrecv, (1,), torch.int32)
        self._kernel = build_dispatch(
            rank=self.rank, arena=a, epr=self.epr, quant_blocks=self.quant_blocks
        )
        torch.cuda.synchronize()
        comm.barrier()

    def dispatch(self, x, ids, wts, stream=None):
        """``x`` bf16 ``[T, hidden]``, ``ids`` int32 ``[T, topk]``, ``wts`` f32 ``[T, topk]``.
        Returns once every peer's rows have landed here."""
        import torch
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        tokens = int(x.shape[0])
        if tokens > self.max_tok:
            raise ValueError(f"tokens={tokens} > max_tok={self.max_tok}")
        s = fx.Stream((stream or torch.cuda.current_stream()).cuda_stream)
        _run_compiled(
            self._kernel,
            *(
                fx.Int64(t.data_ptr())
                for t in (
                    x,
                    ids,
                    wts,
                    self.pos,
                    self.block_cnt,
                    self.cnt,
                    self.cdone,
                    self.bar,
                    self.peer_base,
                    self.callno,
                )
            ),
            fx.Int64(self.dev_comm.ptr),
            fx.Int64(self.win.handle),
            fx.Int64(self.sig.data_ptr()),
            fx.Int32(tokens),
            s,
        )

    # --- what landed, laid out for a fused-MoE GEMM1 ---
    def payload(self):
        return self._pay

    def scales(self):
        return self._sc

    def ids(self):
        return self._ids

    def weights(self):
        return self._wts

    def src(self):
        """``src_rank * cap + token`` per row, -1 for a padding row."""
        return self._src

    def nrecv(self):
        """Rows of the arrays above that this call filled (device tensor)."""
        return self._nrecv

    def close(self):
        import torch

        torch.cuda.synchronize()  # the queues outlive the dev comm: drain first
        for h in (self.dev_comm, self.win, self.mem):
            if h is not None:
                h.close()
        self.dev_comm = self.win = self.mem = None
