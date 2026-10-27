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

Both persistent kernels use grid-wide barriers, so their grid must be fully
resident (one block per CU is).
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import global_typed_ptr
from aiter.ops.flydsl.kernels.tensor_shim import (
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)

BLOCK = 256
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


def build_rank_sort_plan(*, SBM, topk, log2_max_tok, recv_cap, npes, cap):
    """Kernel 1: give every valid sorted row a stable index in its rank's run.

    * count:  ``tile_cnt[tile][pe]`` = rows of the sort tile bound for ``pe``
    * scan:   ``tile_base[tile][pe]`` = rows for ``pe`` in earlier tiles (so the
              rows of a tile range are contiguous in the run), ``cursor[pe]`` = total
    * assign: ``out_row = pe*cap + tile_base + (earlier rows of this tile for pe)``,
              and ``idx_send[out_row] = origin_lid*topk + k``

    The phases are separated by grid barriers inside one persistent launch. Row
    loops read the tile's SBM rows from global memory instead of using LDS scans;
    SBM is small and the rows are L2-resident.
    """
    dec = dict(log2_max_tok=log2_max_tok, recv_cap=recv_cap, topk=topk, npes=npes)

    # fmt: off
    @flyc.kernel(name=f"mega_rank_sort_plan_sbm{SBM}_k{topk}_p{npes}", known_block_size=[BLOCK, 1, 1])
    def plan(arg_stids: fx.Int64, arg_trb: fx.Int64, arg_num_valid: fx.Int64,
        arg_tile_cnt: fx.Int64, arg_tile_base: fx.Int64, arg_cursor: fx.Int64,
        arg_out_row: fx.Int64, arg_idx: fx.Int64, arg_cdone: fx.Int64, arg_bar: fx.Int64,
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

        # The GEMM2 kernel's per-chunk completion counters start from zero every call.
        for g in range(gid, i32_cdone_n, stride):
            cdone[g] = fx.Int32(0)

        for g in range(gid, tiles * fx.Int32(npes), stride):
            tile = g // fx.Int32(npes)
            pe = g - tile * fx.Int32(npes)
            base = trb[tile]
            cnt = fx.Int32(0)
            for r in range_constexpr(SBM):
                valid, dest_pe, _, _ = _decode(stids[base + fx.Int32(r)], **dec)
                cnt = cnt + (valid & (dest_pe == pe)).select(fx.Int32(1), fx.Int32(0))
            tile_cnt[g] = cnt
        grid_barrier(arg_bar, i32_nblocks, tid)

        if bid == fx.Int32(0):
            if tid < fx.Int32(npes):
                run = arith.constant(0)
                for t in range(fx.Int32(0), tiles, fx.Int32(1)):
                    tile_base[t * fx.Int32(npes) + tid] = run
                    run = run + tile_cnt[t * fx.Int32(npes) + tid]
                tile_base[tiles * fx.Int32(npes) + tid] = run
                ptr_buf_tensor(arg_cursor, fx.Int32)[tid] = run
        grid_barrier(arg_bar, i32_nblocks, tid)

        idx_buf = ptr_buf_tensor(arg_idx, fx.Int32)
        for g in range(gid, tiles * fx.Int32(SBM), stride):
            tile = g // fx.Int32(SBM)
            row = g - tile * fx.Int32(SBM)
            base = trb[tile]
            valid, dest_pe, dest_lid, s = _decode(stids[base + row], **dec)
            out_buf[g] = fx.Int32(-1)
            if valid:
                within = fx.Int32(0)
                for rp in range_constexpr(SBM):
                    v2, pe2, _, _ = _decode(stids[base + fx.Int32(rp)], **dec)
                    earlier = (fx.Int32(rp) < row) & v2 & (pe2 == dest_pe)
                    within = within + earlier.select(fx.Int32(1), fx.Int32(0))
                out_row = dest_pe * fx.Int32(cap) + tile_base[tile * fx.Int32(npes) + dest_pe] + within
                out_buf[g] = out_row
                idx_buf[out_row] = dest_lid * fx.Int32(topk) + s

    # fmt: off
    @flyc.jit
    def launch(arg_stids: fx.Int64, arg_trb: fx.Int64, arg_num_valid: fx.Int64,
        arg_tile_cnt: fx.Int64, arg_tile_base: fx.Int64, arg_cursor: fx.Int64,
        arg_out_row: fx.Int64, arg_idx: fx.Int64, arg_cdone: fx.Int64, arg_bar: fx.Int64,
        i32_nblocks: fx.Int32, i32_cdone_n: fx.Int32, stream: fx.Stream):
    # fmt: on
        plan(arg_stids, arg_trb, arg_num_valid, arg_tile_cnt, arg_tile_base, arg_cursor,
             arg_out_row, arg_idx, arg_cdone, arg_bar, i32_nblocks, i32_cdone_n).launch(
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
    """Ship rows ``[lo, hi)`` of my run for ``peer`` (data only; idx goes once at the end)."""
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
def sdma_finish_peer(sdma, w, win, peer, queue, cnt, phase, *, layout, rank, cco):
    """After the last data put: ship idx, drain the queue, publish ``(cnt << 32) | phase``."""
    if cnt > fx.Int32(0):
        sdma.put(
            peer,
            win,
            fx.Int64(layout.off_idx_land + rank * layout.cap * 4),
            win,
            fx.Int64(layout.off_idx_send) + fx.Int64(peer) * fx.Int64(layout.cap * 4),
            fx.Int64(cnt) * fx.Int64(4),
            queue,
            coop=cco.CoopScope.THREAD,
        )
    sdma.quiet_queue(peer, queue)
    # The bytes were moved by the copy engine, not by this CU; the fence only keeps
    # the flag store from being hoisted above the drain, which is the ordering the
    # receiver relies on.
    comm_ops.fence_system_release()
    comm_ops.store_i64_global_system(
        fx.Int64(w.lsa_ptr(peer, layout.off_flags)) + fx.Int64(rank * 8),
        (fx.Int64(cnt) << fx.Int64(32)) | fx.Int64(phase),
    )


def build_combine(*, layout, rank, hidden, topk):
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
    npes, cap = layout.npes, layout.cap
    units = hidden // 8
    chunks = (units + BLOCK - 1) // BLOCK

    # fmt: off
    @flyc.kernel(name=f"mega_combine_p{npes}_c{cap}_h{hidden}_k{topk}_r{rank}", known_block_size=[BLOCK, 1, 1])
    def combine(win: fx.Int64, arg_cursor: fx.Int64, arg_inv: fx.Int64, arg_out: fx.Int64,
        arg_bar: fx.Int64, arg_fin: fx.Int64, i32_nblocks: fx.Int32, i32_tokens: fx.Int32):
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
                comm_ops.spin_until_ge_i32_system(
                    flags_base + fx.Int64(tid) * fx.Int64(8), phase, acquire=True
                )
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
        rows = ptr_buf_tensor(_uniform_i64(w.lsa_ptr(rank, layout.off_s)), fx.BFloat16, unit_elems=8)
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
                        raw = buf_copy_load(rows, safe * fx.Int32(units) + u, fx.BFloat16, unit_elems=8)
                        keep = ok.select(fx.Float32(1.0), fx.Float32(0.0))
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
        arg_bar: fx.Int64, arg_fin: fx.Int64, i32_nblocks: fx.Int32, i32_tokens: fx.Int32,
        stream: fx.Stream):
    # fmt: on
        combine(win, arg_cursor, arg_inv, arg_out, arg_bar, arg_fin, i32_nblocks, i32_tokens).launch(
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

    def __init__(self, moe, queues=8, chunk_tiles=4, comm_wgs=None):
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
        self.cap = self.max_tok * self.topk
        self.layout = ArenaLayout(self.npes, self.cap, self.hidden * 2)
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
        self.cdone = torch.zeros(max_tiles + 1, **i32)
        # Never-reset counters for the two grid barriers and the combine's last-block count.
        self.plan_bar = torch.zeros(1, **i32)
        self.combine_bar = torch.zeros(1, **i32)
        self.combine_fin = torch.zeros(1, **i32)
        # -1 means "no row for this slot"; the combine kernel puts it back after reading.
        self.inv = torch.full((self.cap,), -1, **i32)
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
        )
        self._plans = {}
        self._combine = build_combine(
            layout=lay, rank=self.rank, hidden=self.hidden, topk=self.topk
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
            )
        return self._plans[sbm]

    def set_chunk_tiles(self, chunk_tiles):
        """Change the shipped chunk size; only the gemm2 kernel is (re)compiled."""
        from dataclasses import replace

        self.comm_spec = replace(self.comm_spec, chunk_tiles=chunk_tiles)

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
            persist_cu=stage2.persist_cu,
            persist_strided=stage2.persist_strided,
            skew_cu=stage2.skew_cu,
            g2_bf16_lds=stage2.bf16_lds,
            comm=self.comm_spec,
        )

    def run(self, run_tokens, config=None, profile=None):
        """Three kernel launches. ``profile``: a list that receives ``(stage, cuda_event)`` marks."""
        import flydsl.expr as fx
        import torch
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        moe, lay = self.moe, self.layout
        config = moe._active_config if config is None else config
        s_fx = fx.Stream(torch.cuda.current_stream().cuda_stream)
        op = moe._s1_op

        def mark(stage):
            if profile is not None:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                profile.append((stage, ev))

        mark("start")
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
            fx.Int32(self.nblocks),
            fx.Int32(self.cdone.numel()),
            s_fx,
        )
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
            fx.Int32(self.nblocks),
            fx.Int32(run_tokens),
            s_fx,
        )
        mark("combine")
        return self.out[: run_tokens * self.hidden].view(run_tokens, self.hidden)

    def close(self):
        self.comm.close_all_resources()
