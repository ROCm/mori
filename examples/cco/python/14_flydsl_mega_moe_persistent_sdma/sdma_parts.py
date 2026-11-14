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
"""The Stage2 SDMA protocol: the cco window layout (send buffer ``S``, landing buffer ``R``,
``inv``, flags), the sender's put / signal / flag helpers, and the combine kernel.

Copied from ROCm/aiter#7196 (``mega_moe/stage2_sdma.py``, itself ported from mori example
11); each definition carries a ``# from`` comment naming its source module.
"""

from flydsl.expr import range_constexpr
from flydsl.expr import rocdl
from flydsl.expr.rocdl import cvt_pk_f32_fp8
from flydsl.expr.typing import T
import flydsl.compiler as flyc
import flydsl.expr as fx
import flyutil as comm_ops

from flyutil import (  # noqa: F401
    buf_copy_load,
    buf_copy_store,
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
    ptr_buf_tensor,
)


# from aiter.ops.flydsl.kernels.mega_moe.stage2_sdma
CDONE_STRIDE = 32


# from aiter.ops.flydsl.kernels.mega_moe.stage2_sdma
OUT_ROW_P2P = -2


# from aiter.ops.flydsl.kernels.mega_moe.stage2_sdma
INV_P2P = -2


# from aiter.ops.flydsl.kernels.mega_moe.stage2_sdma
_ALIGN = 128


# from aiter.ops.flydsl.kernels.mega_moe.stage2_sdma
def _align(n):
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


# from aiter.ops.flydsl.kernels.mega_moe.stage2_sdma
class ArenaLayout:
    """Byte offsets inside the cco window; identical on every rank."""

    def __init__(self, npes, cap, row_bytes, slot_cap):
        self.npes, self.cap, self.row_bytes, self.slot_cap = (
            npes,
            cap,
            row_bytes,
            slot_cap,
        )
        self.n_rows = npes * cap
        self.off_s = 0
        self.off_r = self.n_rows * row_bytes
        self.off_inv = self.off_r + self.n_rows * row_bytes
        self.off_sig = self.off_inv + _align(slot_cap * 4)
        self.off_flags = self.off_sig + _align(npes * 4)
        self.off_done = self.off_flags + _align(npes * 4)
        self.off_counter = self.off_done + _align(npes * 4)
        self.total_bytes = self.off_counter + _ALIGN
        # The combine addresses S and R through one 32-bit buffer-resource offset.
        if 2 * self.n_rows * row_bytes >= 1 << 31:
            raise ValueError(
                f"SDMA Stage2 send+landing buffers ({2 * self.n_rows * row_bytes} B) exceed "
                "the 32-bit buffer-resource range; lower row_slack or max_tok_per_rank"
            )

    def key(self):
        return (self.npes, self.cap, self.row_bytes, self.slot_cap)


# from aiter.ops.flydsl.kernels.mega_moe.stage2_sdma
def _decode(p, *, log2_max_tok, recv_cap, topk, npes):
    """Packed srcmap word -> (valid, dest_pe, dest_lid, slot_k), as the Stage2 epilogue does."""
    t = p & fx.Int32(0x00FFFFFF)
    s = p.shrui(fx.Int32(24)) & fx.Int32(0xFF)
    dest_pe = t >> fx.Int32(log2_max_tok)
    dest_lid = t & fx.Int32((1 << log2_max_tok) - 1)
    valid = (t < fx.Int32(recv_cap)) & (s < fx.Int32(topk)) & (dest_pe < fx.Int32(npes))
    return valid, dest_pe, dest_lid, s


# from aiter.ops.flydsl.kernels.mega_moe.stage2_sdma
def sdma_put_run(
    sdma, win, peer, queue, *, layout, rank, lo, hi, cco, remote_signal=False
):
    """Ship rows ``[lo, hi)`` of my run for ``peer`` with one put.

    No local signal (nothing waits on it). ``remote_signal`` hangs the completion
    signal on this put: the copy engine bumps the peer's counter once the bytes
    are there, and the peer's combine polls it. A queue is a FIFO its engine
    drains in order, so a signal on the last put covers every put before it.
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
        local_signal=False,
        remote_signal=remote_signal,
    )


# from aiter.ops.flydsl.kernels.mega_moe.stage2_sdma
def sdma_signal_peer(sdma, win, peer, queue, *, layout, rank, cco):
    """A 4-byte put that only carries the completion signal (a peer whose last chunk was empty)."""
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


# from aiter.ops.flydsl.kernels.mega_moe.stage2_sdma
def sdma_wait_consumed(w, peer, phase, *, layout, rank):
    """Block until ``peer`` has finished reading the previous call's data from my sends."""
    comm_ops.spin_until_ge_i32_system(
        fx.Int64(w.lsa_ptr(rank, layout.off_done)) + fx.Int64(peer) * fx.Int64(4),
        phase - fx.Int32(1),
        acquire=True,
    )


# from aiter.ops.flydsl.kernels.mega_moe.stage2_sdma
def publish_flag(w, peer, phase, *, layout, rank):
    """``flags[rank]`` on ``peer`` = call number (the data itself is covered by the SDMA signal)."""
    comm_ops.store_i32_global_system_release(
        fx.Int64(w.lsa_ptr(peer, layout.off_flags)) + fx.Int64(rank * 4), phase
    )


# from aiter.ops.flydsl.kernels.mega_moe.stage2_sdma
BLOCK = 256


# from aiter.ops.flydsl.kernels.mega_moe.stage2_sdma
def _uniform_i64(x):
    """A wave-uniform i64 (SGPR) copy of ``x``, as a buffer-resource base needs."""
    return fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(x).ir_value()))


# from aiter.ops.flydsl.kernels.mega_moe.stage2_sdma
def build_combine(*, layout, rank, hidden, topk):
    """Kernel 3: finish the transfer and reduce.

    A. wait for every peer's flag and SDMA completion signal for this call;
    B. ``out[t] = sum_k row(inv[t*topk+k])`` in f32, one block per token in turn: a
       landing row of ``S``/``R``, or ``comb_inp[slot]`` for ``INV_P2P``. A slot's
       ``inv`` is reset to -1 once read; slots no route filled stay -1 and add zero;
    C. the last block publishes ``done`` to every peer and advances ``counter``.
    """
    import mori.cco.device.flydsl as cco

    if hidden % 32:
        raise ValueError(f"fp8 1x32 rows need hidden % 32 == 0, got {hidden}")
    npes = layout.npes
    units = hidden // 8
    chunks = (units + BLOCK - 1) // BLOCK
    row_bytes = layout.row_bytes

    # fmt: off
    @flyc.kernel(name=f"mega_moe_sdma_combine_p{npes}_c{layout.cap}_h{hidden}_k{topk}_r{rank}",
                 known_block_size=[BLOCK, 1, 1])
    def combine(win: fx.Int64, arg_inv: fx.Int64, arg_out: fx.Int64, arg_comb_inp: fx.Int64,
        arg_fin: fx.Int64, arg_dev_comm: fx.Int64, i32_nblocks: fx.Int32, i32_tokens: fx.Int32):
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
        if (tid < fx.Int32(npes)) & (tid != fx.Int32(rank)):
            comm_ops.spin_until_ge_i32_system(
                flags_base + fx.Int64(tid) * fx.Int64(4), phase, acquire=True
            )
            # The sender's copy engine bumps this once per call when its last put lands.
            cco.DevComm(arg_dev_comm).sdma().wait_signal(tid, 0, fx.Int64(phase))
            comm_ops.fence_system_acquire()
        fx.barrier()

        inv_buf = ptr_buf_tensor(arg_inv, fx.Int32)

        # B. top-k reduce. Payload is 8 fp8 bytes (2 i32) per thread; its 32-value
        # block is u // 4, so each thread reads its own scale byte.
        bases = [_uniform_i64(w.lsa_ptr(rank, layout.off_s)), _uniform_i64(arg_comb_inp)]
        rows = [ptr_buf_tensor(b, fx.Int32, unit_elems=2) for b in bases]
        scales = [ptr_buf_tensor(b, fx.Int8) for b in bases]
        out = ptr_buf_tensor(arg_out, fx.BFloat16, unit_elems=8)
        v2f32 = T.vec(2, T.f32)

        def row_vals(src, row0, u):
            raw = fx.Vector(
                buf_copy_load(rows[src], (row0 + u * fx.Int32(8)) // fx.Int32(8), fx.Int32, unit_elems=2)
            )
            sc = scales[src][row0 + fx.Int32(hidden) + u // fx.Int32(4)]
            scale = (fx.Uint8(sc).to(fx.Uint32).bitcast(fx.Int32) << fx.Int32(23)).bitcast(fx.Float32)
            parts = []
            for word in range_constexpr(2):
                lo = cvt_pk_f32_fp8(res=v2f32, src=raw[word], word_sel=False)
                hi = cvt_pk_f32_fp8(res=v2f32, src=raw[word], word_sel=True)
                parts.append(lo.shuffle(hi, [0, 1, 2, 3]))
            vals = fx.Vector(parts[0]).shuffle(fx.Vector(parts[1]), [0, 1, 2, 3, 4, 5, 6, 7])
            return fx.Vector(vals), scale

        for tok in range(bid, i32_tokens, i32_nblocks):
            slots = []
            for k in range_constexpr(topk):
                slots.append(inv_buf[tok * fx.Int32(topk) + fx.Int32(k)])
            for chunk in range_constexpr(chunks):
                u = fx.Int32(chunk * BLOCK) + tid
                if u < fx.Int32(units):
                    # Branch-free over k, so all top-k row loads are in flight together.
                    acc = fx.Vector.filled(8, 0.0, fx.Float32)
                    for k in range_constexpr(topk):
                        landed = slots[k] >= fx.Int32(0)
                        keep = landed.select(fx.Float32(1.0), fx.Float32(0.0))
                        vals, scale = row_vals(
                            0, landed.select(slots[k], fx.Int32(0)) * fx.Int32(row_bytes), u
                        )
                        acc = acc + vals * (scale * keep)
                    # Rare: rows past the per-pair reservation, in comb_inp[slot]. The
                    # slots are the same for the whole block, so this branch is uniform.
                    direct = [slots[k] == fx.Int32(INV_P2P) for k in range_constexpr(topk)]
                    any_direct = direct[0]
                    for k in range_constexpr(1, topk):
                        any_direct = any_direct | direct[k]
                    if any_direct:
                        for k in range_constexpr(topk):
                            slot = direct[k].select(tok * fx.Int32(topk) + fx.Int32(k), fx.Int32(0))
                            pvals, pscale = row_vals(1, slot * fx.Int32(row_bytes), u)
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
        arg_fin: fx.Int64, arg_dev_comm: fx.Int64, i32_nblocks: fx.Int32, i32_tokens: fx.Int32,
        stream: fx.Stream):
    # fmt: on
        combine(win, arg_inv, arg_out, arg_comb_inp, arg_fin, arg_dev_comm,
                i32_nblocks, i32_tokens).launch(
            grid=(i32_nblocks, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    return launch
