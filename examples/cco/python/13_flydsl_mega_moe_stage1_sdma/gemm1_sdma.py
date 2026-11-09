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
"""MegaMoEV2 GEMM1 whose A rows come from the cco window ``[S | R]`` through ``rowmap``.

Same compute as aiter's ``compile_gemm1`` (``do_tile``, the epilogue, the B loaders are
aiter's own, so the output is bit for bit the same); only where A comes from differs.
aiter's dispatch copies every routed row into an expert-major ``rx``. Here the rows are
where ``prepare`` and the peers' puts left them, once per (token, rank), and a tile finds
its rows through ``rowmap`` (``stage1_sdma.Arena``)::

    row r of a tile (an expert-major row)  ->  rowmap[r]  ->  row of the window

A row of the window is ``K`` fp8 bytes then ``K / 32`` e8m0 scales, ``row_bytes`` apart, so
one index gives both the A row and its scale.

Phase 1c: the gated kernel always builds the receive layout itself (no separate layout kernel,
no start gate of its own); local_first, sched_layout, layout_blocks, arrival_order,
sched_cache_need, pair_k and the ``S1_NO_URP`` switch are gone.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.mega_moe.gemm1 import _LdsF32View, do_tile
from aiter.ops.flydsl.kernels.mega_moe.gemm_util import (
    AS2RLoader,
    AScaleLoader,
    ATileLoader,
    BScaleLoader,
    BWeightLoader,
    MfmaScaleGU,
    SiluQuantEpilogue,
    TileScheduler,
    _load_row_map_subgroup,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    buf_copy_load,
    ptr_buf_tensor,
)

from stage1_sdma import (
    PHASES_VER,
    SCH_HEAD,
    SCH_LDONE,
    SCH_RQ,
    SCH_TAIL,
    Arena,
    layout_lds_words,
    layout_phase,
    make_prepare_phase,
    memrealtime,
    memtime,
    post_rest_chunks,
    sched_lds_words,
    scheduler_phase,
)

#: ``trace``: int64 slots per workgroup, and work items recorded per workgroup (8 + 3 * items).
TRACE_STRIDE = 2048
TRACE_SLOTS = 5  # per item: gate start, gate passed, done (100 MHz); passed, done (s_memtime)
TRACE_ITEMS = (TRACE_STRIDE - 8) // TRACE_SLOTS
RF_TILE, RF_NEXT = 0, 1  # ``ready_first``: words of the ``rf`` LDS array


class WindowAScaleLoader(AScaleLoader):
    """``AScaleLoader`` for scales that sit at the end of a window row.

    aiter's loader finds the scales of source row ``s`` at ``s * n_scale``; here they are at
    ``s * row_bytes + K``. Always gathers through ``rowmap`` (aiter's does so only from
    ``sort_block_m`` 128 up).
    """

    def __init__(self, *, row_bytes, **kw):
        super().__init__(indexed_input=True, **kw)
        self._row_groups = row_bytes // 16  # window row stride, in 16 B units
        self._first_group = (
            self._n_scale * 32
        ) // 16  # the scales start after the K payload bytes

    def stage(self, lds_ascale, tile_row_base_i32):
        groups_per_row = self._n_scale // 16
        padded_groups_per_row = max(16, groups_per_row)
        slots = self._sort_block_m * padded_groups_per_row

        @flyc.jit
        def copy_slot(slot: fx.Int32):
            if slot < fx.Int32(slots):
                row = slot // fx.Int32(padded_groups_per_row)
                group = slot % fx.Int32(padded_groups_per_row)
                if group < fx.Int32(groups_per_row):
                    lane = self._tx & fx.Int32(63)
                    packed = _load_row_map_subgroup(
                        self._row_map_rsrc, tile_row_base_i32 + row, lane,
                        padded_groups_per_row,
                    )
                    source_row = packed & fx.Int32(0xFFFFFF)
                    v = buf_copy_load(
                        self._rsrc,
                        source_row * fx.Int32(self._row_groups)
                        + fx.Int32(self._first_group)
                        + group,
                        fx.Int32,
                        unit_elems=4,
                    )
                    lin = row * fx.Int32(groups_per_row) + group
                    dst = fx.make_view(
                        fx.add_offset(
                            fx.recast_iter(fx.Int32, lds_ascale.ptr), lin * fx.Int32(4)
                        ),
                        fx.make_layout(4, 1),
                    )
                    fragment = fx.make_rmem_tensor(4, fx.Int32)
                    fragment.store(v)
                    fx.copy(
                        fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32),
                        fragment,
                        dst,
                    )

        for c in range_constexpr(0, slots, self._total_threads):
            copy_slot(fx.Int32(c) + fx.Int32(self._tx))


# fmt: off
@functools.cache
def compile_gemm1_sdma(
    *, model_dim: int, inter_dim: int, row_bytes: int, source_rows: int, expert_offset: int = 0,
    sort_block_m: int = 128, tile_n: int = 512, tile_k: int = 256, num_waves: int = 8,
    pipe_weights: bool = True, mfma_amajor: bool = True, swizzle_a: bool = True,
    async_a_copy: bool = True, use_tile_resource: bool = True, waves_per_eu_hint: int = 2,
    b_cache_modifier: int = 0, swiglu_limit: float = 0.0, num_cu: int = 256,
    gate_arena: Arena | None = None, gate_rank: int = 0, trace: bool = False,
    fuse_prepare: bool = False, prep_trace: bool = False, prep_split: int = 3,
    prep_unroll: int = 1, ready_first: bool = False, no_plan: bool = False,
):
    # fmt: on
    """Persistent GEMM1, one workgroup per CU, flat (m tile, n tile) work item per step.

    With ``gate_arena`` (the ``Arena`` of the prepare) the kernel does the receiving side
    itself instead of leaving it to a ``wait`` kernel:

    * workgroup 0 is the comm workgroup: it puts the chunks of my runs that ``prepare`` did
      not (``post_rest_chunks``; ``prepare`` put the first ``early_chunks``), and the other
      workgroups compute;
    * the compute workgroups first build the receiver's layout (``layout_phase``): it waits
      for every source's route flag, does the one acquire, and ends in a rendezvous that
      releases the tables it wrote; only then is ``num_valid`` read to size the tile loop;
    * then, before each tile, a workgroup waits for the chunks of every peer's run that the
      tile's expert needs (``need``, and the SDMA signal counters, which count from
      ``sig_base``). A tile whose chunks have landed starts at once, so the GEMM runs while
      later chunks are in flight. There is no fence per tile: see the comment there.

    Without ``gate_arena`` it is the plain static loop over tables built beforehand.

    ``fuse_prepare`` (with ``gate_arena``): the prepare (``prepare_phase``: plan, route staging,
    quant, the early puts) runs first in the same kernel, so a call is ONE launch. Workgroup 0
    puts the early chunks inside it, then waits for every worker to have quantised its rows
    (``qdone``, one-way arrivals) before it puts the rest; the workers go on to the layout.
    The prepare's LDS (plan + quant store swizzle) is lent out of the pool.

    ``ready_first`` (with ``gate_arena``): the per-tile SDMA gate is gone. One wave of the comm
    workgroup is a TILE SCHEDULER (``scheduler_phase``): it watches the SDMA signal levels and
    appends m-tiles to a ready queue (``Stage1SdmaPrepare.sched``) in the order they become
    ready. A worker takes a ticket (``atomic_add`` on ``head``), q = ticket // n_tiles, waits
    for ``rq_tail > q`` (one lane, ``s_sleep``), reads ``rq[q]`` = the m-tile and runs its
    (n_tile = ticket % n_tiles) work item. It exits when its ticket is past the last m-tile.
    No worker reads a signal or ``need``. ``arm`` must zero ``sched`` (``sched=True``).

    ``no_plan`` (the prepare's): no ``need`` exists, so a tile (or the scheduler) waits for ALL
    of every source's chunks of this call (``expected``).

    ``trace`` (with ``gate_arena``): thread 0 of each workgroup writes the clock (100 MHz) to
    ``trace[bid * TRACE_STRIDE + ...]``: ``[0]`` kernel start, ``[1]`` layout done, ``[2]``
    end, ``[3]`` its number of work items, and from ``[8]`` per item ``(gate start, gate
    passed, item done)``, up to ``TRACE_ITEMS`` items. Workgroup 0 (comm): start and end only
    (and, with ``ready_first``, the scheduler's ``SCH_TR_*`` slots).
    """
    # FlyDSL's cache key holds a closure's scalars only, not ``gate_arena``: its repr has to be
    # one, or a kernel compiled for another chunk_rows / early_chunks is reused.
    GATE_KEY = repr(gate_arena)
    gate_npes = gate_arena.npes if gate_arena else 0
    gate_epr = gate_arena.experts_per_rank if gate_arena else 0
    assert tile_k == 256 and model_dim % tile_k == 0 and tile_n % num_waves == 0
    assert (2 * inter_dim) % tile_n == 0
    n_per_wave = tile_n // num_waves
    n_tiles = (2 * inter_dim) // tile_n
    m_repeat = sort_block_m // 16
    num_acc_n = n_per_wave // 16
    assert num_acc_n % 2 == 0 and m_repeat % 2 == 0
    a_k_step_bytes = tile_k
    k_iters = model_dim // tile_k
    total_threads = num_waves * 64
    a_lds_size = sort_block_m * a_k_step_bytes
    a_lds_i32 = a_lds_size // 4
    lds_pool_bytes = max(2 * a_lds_size, sort_block_m * (tile_n // 2) * 4)
    n_scale_bytes = sort_block_m * (model_dim // 32)
    if source_rows * row_bytes >= 1 << 32:
        raise ValueError("the window exceeds the 32-bit buffer-resource range")
    if ready_first and not gate_arena:
        raise ValueError("ready_first needs gate_arena")
    NWQ = num_cu - 1   # workers: every one owns tokens and quantises (``fuse_prepare``)
    sched_words = 0
    if ready_first:
        sched_words = sched_lds_words((gate_arena.max_routes + sort_block_m - 1) // sort_block_m)
        if sched_words * 4 > n_scale_bytes:
            raise ValueError(
                f"ready_first: the scheduler's per-tile table ({sched_words} words) does not fit "
                f"the A-scale LDS ({n_scale_bytes // 4} words) at sort_block_m={sort_block_m}")
        if gate_npes > 8 or gate_epr >= 65536:
            raise ValueError("ready_first: npes <= 8 and fewer than 65536 experts per rank")
    if gate_arena:
        if lds_pool_bytes < layout_lds_words(gate_arena) * 4:
            raise ValueError("the LDS pool is too small to lend the layout its words")
    prepare_phase = None
    if fuse_prepare:
        if not gate_arena:
            raise ValueError("fuse_prepare needs gate_arena")
        prepare_phase, pcfg = make_prepare_phase(
            rank=gate_rank, arena=gate_arena, epr=gate_epr, quant_blocks=num_cu - 1,
            trace=prep_trace, split=prep_split, unroll=prep_unroll, prep_block=total_threads,
            fused=True, no_plan=no_plan,
        )
        if pcfg.nblocks != num_cu or pcfg.PB != total_threads:
            raise ValueError("the prepare's grid must be the GEMM1's grid")
        if lds_pool_bytes < pcfg.lds_bytes:
            # the quant store swizzle adds QLS_STRIDE (2304 B) per wave to the plan's LDS: 8
            # waves put pcfg.lds_bytes at ~6 KB + 18 KB. The pool is sort_block_m * tile_n / 2 *
            # 4 bytes (128 KB at 128x512), so there is room; a small tile_n / sort_block_m could
            # run out.
            raise ValueError(
                f"the LDS pool ({lds_pool_bytes} B) is too small to lend the prepare its "
                f"{pcfg.lds_bytes} bytes (quant store staging included)")

    if ready_first:
        @fx.struct
        class SharedStorage:
            pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
            A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]
            # survives ``do_tile`` (which owns the pool): the chosen m-tile and the next ticket
            rf: fx.Array[fx.Int8, 32, 16]
    else:
        @fx.struct
        class SharedStorage:
            pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
            A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    # fmt: off
    @flyc.kernel(name=f"stage1_sdma_gemm1_t{sort_block_m}x{tile_n}_rb{row_bytes}_r{expert_offset}_tr{int(trace)}"
                      f"_g{int(bool(gate_npes))}_fp{int(fuse_prepare)}_ps{prep_split}u{prep_unroll}"
                      f"pt{int(prep_trace)}_rf{int(ready_first)}np{int(no_plan)}_pv{PHASES_VER}",
                 known_block_size=[total_threads, 1, 1])
    def kernel(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_w: fx.Tensor, tile_row_base: fx.Tensor,
        tile_input_base: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor,
        out_scale: fx.Tensor, rowmap: fx.Tensor, nvm: fx.Int32, arg_dev_comm: fx.Int64,
        arg_win: fx.Int64, arg_sig_base: fx.Int64, arg_expected: fx.Int64, arg_win_handle: fx.Int64,
        arg_cnt: fx.Int64, arg_trace: fx.Int64,
        arg_hist: fx.Int64, arg_cursor: fx.Int64, arg_lbar: fx.Int64, arg_srcmap: fx.Int64,
        arg_wts: fx.Int64, arg_eids: fx.Int64, arg_trb: fx.Int64, arg_tib: fx.Int64,
        arg_nv: fx.Int64, arg_met: fx.Int64, arg_ete: fx.Int64, arg_sched: fx.Int64,
        p_ids: fx.Int64, p_x: fx.Int64, p_row_tok: fx.Int64, p_rowpos: fx.Int64, p_cum: fx.Int64,
        p_hist: fx.Int64, p_cursor: fx.Int64, p_bar: fx.Int64, p_wts: fx.Int64,
        p_trace: fx.Int64, p_done: fx.Int64, p_qdone: fx.Int64, p_tokens: fx.Int32,
    ):
    # fmt: on
        _ = GATE_KEY
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))

        w_rsrc = ptr_buf_tensor(fx.get_iter(w), fx.Int32, unit_elems=4)
        sx_rsrc = ptr_buf_tensor(fx.get_iter(x), fx.Int32, unit_elems=4)
        sw_rsrc = ptr_buf_tensor(fx.get_iter(scale_w), fx.Int32)
        trb_rsrc = ptr_buf_tensor(fx.get_iter(tile_row_base), fx.Int32)
        tib_rsrc = ptr_buf_tensor(fx.get_iter(tile_input_base), fx.Int32)
        expert_rsrc = ptr_buf_tensor(fx.get_iter(expert_ids), fx.Int32)
        nv_rsrc = ptr_buf_tensor(fx.get_iter(num_valid_ids), fx.Int32)
        row_map_rsrc = ptr_buf_tensor(fx.get_iter(rowmap), fx.Int32)
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        os_rsrc = ptr_buf_tensor(
            fx.get_iter(out_scale), fx.Int8, num_records_bytes=nvm * fx.Int32(scale_cols) + fx.Int32(8192)
        )
        wave_id = fx.thread_idx.x // 64

        sched = TileScheduler(expert_rsrc=expert_rsrc, inter_dim=inter_dim, expert_offset=expert_offset)
        n_wave_base = wave_id * fx.Int32(n_per_wave)
        a_gather = ATileLoader(
            row_bytes=row_bytes, sort_block_m=sort_block_m, k_step_bytes=a_k_step_bytes,
            total_threads=total_threads, swizzle=swizzle_a, x_tensor=x, async_copy=async_a_copy,
            indexed_input=True, row_map_rsrc=row_map_rsrc, source_rows=source_rows,
        )
        a_s2r = AS2RLoader(k_step_bytes=a_k_step_bytes, swizzle=swizzle_a)
        b_loader = BWeightLoader(
            w_rsrc=w_rsrc, num_acc_n=num_acc_n, model_dim=model_dim, cache_modifier=b_cache_modifier
        )
        b_scale = BScaleLoader(scale_rsrc=sw_rsrc, num_acc_n=num_acc_n, model_dim=model_dim)
        a_scale = WindowAScaleLoader(
            row_bytes=row_bytes, scale_rsrc=sx_rsrc, m_repeat=m_repeat, model_dim=model_dim,
            sort_block_m=sort_block_m, total_threads=total_threads, row_map_rsrc=row_map_rsrc,
        )
        mfma = MfmaScaleGU(m_repeat=m_repeat, num_acc_n=num_acc_n)
        epi = SiluQuantEpilogue(
            out_rsrc=None, out_scale_rsrc=os_rsrc, sorted_rsrc=trb_rsrc, tokens=0, inter_dim=inter_dim,
            m_repeat=m_repeat, num_acc_n=num_acc_n, sort_block_m=sort_block_m, tile_n=tile_n,
            num_waves=num_waves, lds_out=c_tile, swiglu_limit=swiglu_limit, always_valid=True,
            out_tensor=out, evec=8,
        )

        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        def work_items():
            """Tile loop size, read when it is needed: with ``gate_arena`` it is written by
            the layout at the front of this very kernel."""
            num_valid = nv_rsrc[fx.Int32(0)]
            return ((num_valid + fx.Int32(sort_block_m - 1)) // fx.Int32(sort_block_m)) * fx.Int32(n_tiles)

        trace_buf = ptr_buf_tensor(arg_trace, fx.Int64)

        def stamp(buf, slot):
            """``buf[bid * TRACE_STRIDE + slot]`` = the clock, written by thread 0."""
            if const_expr(trace and gate_npes):
                if tid == fx.Int32(0):
                    buf[bid * fx.Int32(TRACE_STRIDE) + slot] = memrealtime()

        def stamp_mt(buf, slot):
            """``buf[bid * TRACE_STRIDE + slot]`` = the shader clock (thread 0)."""
            if const_expr(trace and gate_npes):
                if tid == fx.Int32(0):
                    buf[bid * fx.Int32(TRACE_STRIDE) + slot] = memtime()

        def stamp_at(buf, idx):
            """``buf[idx]`` = the 100 MHz clock, by whichever lane runs this."""
            buf[idx] = memrealtime()

        def stamp_i64(buf, slot, v):
            """``buf[bid * TRACE_STRIDE + slot]`` = ``v`` (thread 0); a count, not a time."""
            if const_expr(trace and gate_npes):
                if tid == fx.Int32(0):
                    buf[bid * fx.Int32(TRACE_STRIDE) + slot] = v

        stamp(trace_buf, fx.Int32(0))
        if const_expr(gate_npes):
            import mori.cco.device.flydsl as cco

            sig_base = ptr_buf_tensor(arg_sig_base, fx.Int32)
            expected = ptr_buf_tensor(arg_expected, fx.Int32)

        def run_tile_rf(m_tile, n_tile, item):
            """``ready_first``: m-tile ``m_tile`` (from the ready queue) is ready, nothing to
            wait for. The trace records the item's start twice (no gate)."""
            slot0 = (item < fx.Int32(TRACE_ITEMS)).select(fx.Int32(8) + item * fx.Int32(TRACE_SLOTS), fx.Int32(TRACE_STRIDE - TRACE_SLOTS))
            stamp(trace_buf, slot0)
            stamp(trace_buf, slot0 + fx.Int32(1))
            stamp_mt(trace_buf, slot0 + fx.Int32(3))
            do_tile(
                m_tile, n_wave_base + n_tile * fx.Int32(tile_n), sched.expert_of(m_tile), sched,
                a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf, a_scale_lds,
                a_lds_i32, k_iters, m_repeat, num_acc_n, a_k_step_bytes, pipe_weights,
                mfma_amajor, async_a_copy, trb_rsrc, tib_rsrc, True, False,  # indirect, pair_k
            )
            stamp(trace_buf, slot0 + fx.Int32(2))
            stamp_mt(trace_buf, slot0 + fx.Int32(4))

        def run_tile(flat):
            m_tile = flat // fx.Int32(n_tiles)
            n_tile = flat - m_tile * fx.Int32(n_tiles)
            # item number of this work item in this workgroup (the loop below is a static stride)
            item = (flat - (bid - fx.Int32(1))) // fx.Int32(num_cu - 1)
            slot0 = (item < fx.Int32(TRACE_ITEMS)).select(fx.Int32(8) + item * fx.Int32(TRACE_SLOTS), fx.Int32(TRACE_STRIDE - TRACE_SLOTS))
            if const_expr(gate_npes):
                stamp(trace_buf, slot0)
                e_local = sched.expert_of(m_tile)
                if (tid < fx.Int32(gate_npes)) & (tid != fx.Int32(gate_rank)):
                    # RELAXED on purpose: an acquire load emits ``buffer_inv sc0 sc1`` -- a
                    # whole-L2 invalidate -- and this runs once per TILE, with the weights in
                    # that L2. ``need`` is published by the source before it bumps its flag,
                    # and the kernel already did one system acquire after waiting for the
                    # flags, so every ``need`` is visible by the time any tile runs. The load
                    # itself is system-scope (``sc0 sc1``), so it does not hit a stale line.
                    if const_expr(no_plan):
                        # the prepare wrote no ``need`` (no counting sort): an expert's rows can be
                        # anywhere in a source's run, so wait for all of its chunks of this call
                        cco.DevComm(arg_dev_comm).sdma().wait_signal(
                            tid, 0, fx.Int64(expected[tid])
                        )
                    else:
                        need = fx.Int32(
                            comm_ops.load_i32_global_system(
                                fx.Int64(arg_win) + fx.Int64(gate_arena.off_need)
                                + fx.Int64(tid * fx.Int32(gate_epr) + e_local) * fx.Int64(4),
                            )
                        )
                        cco.DevComm(arg_dev_comm).sdma().wait_signal(
                            tid, 0, fx.Int64(sig_base[tid] + need)
                        )
                fx.barrier()
                stamp(trace_buf, slot0 + fx.Int32(1))
                stamp_mt(trace_buf, slot0 + fx.Int32(3))
            do_tile(
                m_tile, n_wave_base + n_tile * fx.Int32(tile_n), sched.expert_of(m_tile), sched,
                a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf, a_scale_lds,
                a_lds_i32, k_iters, m_repeat, num_acc_n, a_k_step_bytes, pipe_weights,
                mfma_amajor, async_a_copy, trb_rsrc, tib_rsrc, True, False,  # indirect, pair_k
            )
            if const_expr(gate_npes):
                stamp(trace_buf, slot0 + fx.Int32(2))
                stamp_mt(trace_buf, slot0 + fx.Int32(4))

        q0 = fx.Int32(0)
        if const_expr(fuse_prepare):
            q0 = prepare_phase(
                tid=tid, bid=bid, tokens=p_tokens, arg_ids=p_ids, arg_x=p_x,
                arg_row_tok=p_row_tok, arg_rowpos=p_rowpos, arg_cnt=arg_cnt, arg_cum=p_cum,
                arg_hist=p_hist, arg_cursor=p_cursor, arg_bar=p_bar,
                arg_s=fx.Int64(arg_win) + fx.Int64(gate_arena.off_s),
                arg_dev_comm=arg_dev_comm, arg_win=arg_win_handle, arg_wts=p_wts,
                arg_trace=p_trace, arg_done=p_done, arg_qdone=p_qdone,
                L=fx.Int64(fx.ptrtoint(lds.pool.ptr)),
            )
            stamp(trace_buf, fx.Int32(4))   # trace: this block's prepare_phase is done

        if const_expr(gate_npes):
            if bid == fx.Int32(0):
                if const_expr(fuse_prepare):
                    # every worker has quantised its rows (they are in memory): the rest of the
                    # chunks can go. The workers do not wait for this.
                    comm_ops.spin_until_ge_i32_agent(
                        fx.Int64(p_qdone), q0 + fx.Int32(NWQ)
                    )
                    stamp(trace_buf, fx.Int32(5))   # trace: qdone reached (every quant row retired)
                post_rest_chunks(
                    cco.DevComm(arg_dev_comm).sdma(), arg_win_handle, tid // fx.Int32(64),
                    tid % fx.Int32(64), ptr_buf_tensor(arg_cnt, fx.Int32),
                    rank=gate_rank, arena=gate_arena, cco=cco,
                )
                stamp(trace_buf, fx.Int32(2))
                if const_expr(trace):
                    # trace: every comm wave's own "rest chunks posted" time, at ``trace[16 + wave]``
                    if tid % fx.Int32(64) == fx.Int32(0):
                        stamp_at(trace_buf, fx.Int32(16) + tid // fx.Int32(64))
                if const_expr(trace and not ready_first):
                    # trace only: when the last chunk from each peer has landed (WG0 is idle
                    # here): ``trace[8 + peer]``. Not with ``ready_first``: this wait would hold
                    # the scheduler wave (wave 0) until EVERY chunk has landed.
                    if (tid < fx.Int32(gate_npes)) & (tid != fx.Int32(gate_rank)):
                        cco.DevComm(arg_dev_comm).sdma().wait_signal(tid, 0, fx.Int64(expected[tid]))
                        stamp_at(trace_buf, fx.Int32(8) + tid)
                if const_expr(ready_first):
                    # wave 0 of the comm workgroup is the tile scheduler (it has put its peer's
                    # chunks above). Its tile table lives in the ``A_scale`` LDS of this block:
                    # only ``do_tile`` uses that, and workgroup 0 runs no tile.
                    if tid < fx.Int32(64):
                        scheduler_phase(
                            rank=gate_rank, arena=gate_arena, sbm=sort_block_m, nb=num_cu - 1,
                            max_tiles=(gate_arena.max_routes + sort_block_m - 1) // sort_block_m,
                            lane=tid, L=fx.Int64(fx.ptrtoint(lds.A_scale.ptr)),
                            win=fx.Int64(arg_win),
                            sdma=cco.DevComm(arg_dev_comm).sdma(), arg_sched=arg_sched,
                            nv_rsrc=nv_rsrc, expert_of=sched.expert_of, sig_base=sig_base,
                            trace=trace, trace_buf=trace_buf,
                            trace_base=bid * fx.Int32(TRACE_STRIDE),
                            full_wait=no_plan, expected=expected,
                        )
            else:
                # the front of the workers IS the layout: it waits for the route flags and does
                # the one acquire, and its last rendezvous releases the tables the tiles below
                # read from other workgroups (and, ready_first, tells the scheduler). A tile does
                # no cache invalidation after it has seen its chunks land (the SDMA signal is a
                # relaxed load): SDMA writes memory without touching this GPU's L2, lines of
                # earlier calls are gone by now, and a line of this call is only read after its
                # chunk has landed, because a chunk starts on a 128 B line (``Arena.check``).
                # Measured: an acquire per tile cost ~90 us of 0.96 ms (it invalidates L2, with
                # the weights in it).
                layout_phase(
                    rank=gate_rank, arena=gate_arena, sbm=sort_block_m, max_tok=gate_arena.cap,
                    nb=num_cu - 1, tid=tid, lbid=bid - fx.Int32(1), tpb=total_threads,
                    L=fx.Int64(fx.ptrtoint(lds.pool.ptr)), win=fx.Int64(arg_win),
                    arg_expected=arg_expected, arg_hist=arg_hist, arg_cursor=arg_cursor,
                    arg_bar=arg_lbar, arg_srcmap=arg_srcmap, arg_wts=arg_wts,
                    arg_eids=arg_eids, arg_trb=arg_trb, arg_tib=arg_tib, arg_nv=arg_nv,
                    arg_met=arg_met, arg_ete=arg_ete, final_release=True,
                    signal_done=ready_first,
                    arg_ldone=fx.Int64(arg_sched) + fx.Int64(SCH_LDONE * 4),
                )
                stamp(trace_buf, fx.Int32(1))
                total_work = work_items()
                if const_expr(ready_first):
                    # tickets over work items (q = ticket // n_tiles is the ready-queue index,
                    # so a m-tile's n-tiles are consecutive); the m-tile comes from the queue
                    rf_lds = fx.Int64(fx.ptrtoint(lds.rf.ptr))
                    head_addr = fx.Int64(arg_sched) + fx.Int64(SCH_HEAD * 4)
                    tail_addr = fx.Int64(arg_sched) + fx.Int64(SCH_TAIL * 4)
                    rq_addr = fx.Int64(arg_sched) + fx.Int64(SCH_RQ * 4)

                    def rfw(i):
                        return rf_lds + fx.Int64(i) * fx.Int64(4)

                    def claim():
                        return fx.Int32(comm_ops.atomic_add_agent(head_addr, fx.Int32(1)))

                    if tid == fx.Int32(0):
                        comm_ops.store_i32_lds(rfw(RF_NEXT), claim())
                    fx.barrier()
                    tk = fx.Int32(comm_ops.load_i32_lds(rfw(RF_NEXT)))
                    item = fx.Int32(0)
                    wacc = fx.Int64(0)
                    wfirst = fx.Int64(0)
                    tacc = fx.Int64(0)
                    while tk < total_work:
                        q = tk // fx.Int32(n_tiles)
                        n_tile = tk - q * fx.Int32(n_tiles)
                        nxt = fx.Int32(0)
                        if tid == fx.Int32(0):
                            # ONE lane waits for the scheduler to have published slot q. The
                            # entry was stored (write-through) before the tail that covers
                            # it; both are read with system-scope loads.
                            if const_expr(trace):
                                tw0 = memrealtime()
                            tl = fx.Int32(comm_ops.load_i32_global_system(tail_addr))
                            while tl <= q:
                                rocdl.s_sleep(1)
                                tl = fx.Int32(comm_ops.load_i32_global_system(tail_addr))
                            comm_ops.store_i32_lds(rfw(RF_TILE), fx.Int32(
                                comm_ops.load_i32_global_system(rq_addr + fx.Int64(q) * fx.Int64(4))))
                            # the next ticket is claimed now, its latency hidden by the tile
                            nxt = claim()
                            if const_expr(trace):
                                dw = memrealtime() - tw0
                                wacc = wacc + (item > fx.Int32(0)).select(dw, fx.Int64(0))
                                wfirst = (item == fx.Int32(0)).select(dw, wfirst)
                        fx.barrier()
                        m_tile = fx.Int32(comm_ops.load_i32_lds(rfw(RF_TILE)))
                        if const_expr(trace):
                            ta = memrealtime()
                        run_tile_rf(m_tile, n_tile, item)
                        if const_expr(trace):
                            tacc = tacc + (memrealtime() - ta)
                        if tid == fx.Int32(0):
                            comm_ops.store_i32_lds(rfw(RF_NEXT), nxt)
                        fx.barrier()
                        tk = fx.Int32(comm_ops.load_i32_lds(rfw(RF_NEXT)))
                        item = item + fx.Int32(1)
                    stamp(trace_buf, fx.Int32(2))
                    stamp_i64(trace_buf, fx.Int32(5), wacc)
                    stamp_i64(trace_buf, fx.Int32(6), wfirst)
                    stamp_i64(trace_buf, fx.Int32(7), tacc)
                    if const_expr(trace):
                        if tid == fx.Int32(0):
                            trace_buf[bid * fx.Int32(TRACE_STRIDE) + fx.Int32(3)] = fx.Int64(item)
                else:
                    for flat in range(bid - fx.Int32(1), total_work, fx.Int32(num_cu - 1)):
                        run_tile(flat)
                    stamp(trace_buf, fx.Int32(2))
                    if const_expr(trace):
                        if tid == fx.Int32(0):
                            n_items = (total_work - (bid - fx.Int32(1)) + fx.Int32(num_cu - 2)) // fx.Int32(num_cu - 1)
                            trace_buf[bid * fx.Int32(TRACE_STRIDE) + fx.Int32(3)] = fx.Int64(n_items)
        else:
            total_work = work_items()
            for flat in range(bid, total_work, fx.Int32(num_cu)):
                run_tile(flat)

    # fmt: off
    @flyc.jit
    def launch(out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_w: fx.Tensor,
               tile_row_base: fx.Tensor, tile_input_base: fx.Tensor, expert_ids: fx.Tensor,
               num_valid_ids: fx.Tensor, out_scale: fx.Tensor, rowmap: fx.Tensor, nvm: fx.Int32,
               arg_dev_comm: fx.Int64, arg_win: fx.Int64, arg_sig_base: fx.Int64,
               arg_expected: fx.Int64, arg_win_handle: fx.Int64, arg_cnt: fx.Int64,
               arg_trace: fx.Int64, arg_hist: fx.Int64, arg_cursor: fx.Int64, arg_lbar: fx.Int64,
               arg_srcmap: fx.Int64, arg_wts: fx.Int64, arg_eids: fx.Int64, arg_trb: fx.Int64,
               arg_tib: fx.Int64, arg_nv: fx.Int64, arg_met: fx.Int64, arg_ete: fx.Int64,
               arg_sched: fx.Int64,
               p_ids: fx.Int64, p_x: fx.Int64, p_row_tok: fx.Int64, p_rowpos: fx.Int64,
               p_cum: fx.Int64, p_hist: fx.Int64, p_cursor: fx.Int64, p_bar: fx.Int64,
               p_wts: fx.Int64, p_trace: fx.Int64, p_done: fx.Int64, p_qdone: fx.Int64,
               p_tokens: fx.Int32, stream: fx.Stream):
    # fmt: on
        kernel(out, x, w, scale_w, tile_row_base, tile_input_base, expert_ids, num_valid_ids,
               out_scale, rowmap, nvm, arg_dev_comm, arg_win, arg_sig_base, arg_expected,
               arg_win_handle, arg_cnt, arg_trace, arg_hist, arg_cursor, arg_lbar, arg_srcmap,
               arg_wts, arg_eids, arg_trb, arg_tib, arg_nv, arg_met, arg_ete, arg_sched,
               p_ids, p_x, p_row_tok, p_rowpos, p_cum, p_hist, p_cursor, p_bar, p_wts, p_trace,
               p_done, p_qdone, p_tokens,
               value_attrs={
                   "rocdl.waves_per_eu": waves_per_eu_hint,
                   "rocdl.flat_work_group_size": f"{total_threads},{total_threads}",
               }).launch(grid=(num_cu, 1, 1), block=(total_threads, 1, 1), stream=stream)

    return launch


def run_gemm1_sdma(out, window, w, scale_w, tile_row_base, tile_input_base, expert_ids,
                   num_valid, out_scale, rowmap, nvm, stream, prep=None, trace=None, fuse=None,
                   **kwargs):
    """``window`` is ``[S | R]`` as fp8 ``[2 * n_rows, row_bytes]``; the others are the
    tensors of aiter's stage1 (``MegaMoEV2._s1_*`` and its dispatch op).

    With ``prep`` (a ``Stage1SdmaPrepare`` that has ``arm``-ed this call) the kernel waits for
    the data AND builds the receive layout itself: ``fuse`` must be
    ``prep.fused_layout_args(moe)`` (plus, for ``fuse_prepare``, ``fuse["prepare"] =
    prep.fused_prepare_args(x, ids, wts)``), and the separate layout kernel must NOT be run for
    this call. Without ``prep`` it is the plain static loop over tables built beforehand
    (``prep.layout``)."""
    gate = {}
    dev_comm = win = sig_base = expected = win_handle = cnt = 0
    if prep is not None:
        if fuse is None:
            raise ValueError("the gated GEMM1 builds the layout itself: pass "
                             "fuse=prep.fused_layout_args(moe)")
        gate = dict(gate_arena=prep.arena, gate_rank=prep.rank, no_plan=bool(prep.no_plan))
        dev_comm, win, win_handle = prep.dev_comm.ptr, prep.mem.ptr, prep.win.handle
        sig_base, expected, cnt = prep.sig_base.data_ptr(), prep.expected.data_ptr(), prep.cnt.data_ptr()
    if fuse is not None and prep is None:
        raise ValueError("fuse needs prep")
    if prep is None:
        # no gate, no scheduler: the plain static loop whatever the flags say
        kwargs.pop("ready_first", None)
    elif kwargs.get("ready_first") and not prep.sched_on:
        raise ValueError("ready_first needs Stage1SdmaPrepare(sched=True) (arm zeroes its counters)")
    fz = dict(fuse or {})
    if prep is not None:
        fz.setdefault("sched", prep.sched.data_ptr())
    pz = fz.get("prepare")
    pre = {}
    if pz is not None:
        pre = dict(fuse_prepare=True, prep_trace=bool(prep.trace), prep_split=int(prep.split),
                   prep_unroll=int(prep.unroll))
    launch = compile_gemm1_sdma(
        row_bytes=int(window.shape[1]), source_rows=int(window.shape[0]), trace=trace is not None,
        **pre, **gate, **kwargs,
    )
    fuse_ptrs = [fx.Int64(fz.get(k, 0)) for k in (
        "hist", "cursor", "lbar", "srcmap", "wts", "eids", "trb", "tib", "nv", "met", "ete",
        "sched")]
    pz = pz or {}
    fuse_ptrs += [fx.Int64(pz.get(k, 0)) for k in (
        "ids", "x", "row_tok", "rowpos", "cum", "hist", "cursor", "bar", "wts", "trace", "done",
        "qdone")]
    fuse_ptrs.append(fx.Int32(int(pz.get("tokens", 0))))
    _run_compiled(
        launch, out, window, w, scale_w, tile_row_base, tile_input_base, expert_ids, num_valid,
        out_scale, rowmap, fx.Int32(nvm), fx.Int64(dev_comm), fx.Int64(win), fx.Int64(sig_base),
        fx.Int64(expected), fx.Int64(win_handle), fx.Int64(cnt),
        fx.Int64(trace.data_ptr() if trace is not None else 0), *fuse_ptrs, stream,
    )


def gemm1_kwargs(moe, config):
    """``compile_gemm1_sdma`` / ``run_gemm1_sdma`` keyword arguments for aiter's stage1 shape and
    tuning on ``moe`` (a ``MegaMoEV2``): the same tile as aiter's own GEMM1."""
    s1 = config.stage1
    return dict(
        model_dim=moe.model_dim,
        inter_dim=moe.inter_dim,
        expert_offset=moe.rank * moe.epr,
        sort_block_m=s1.sort_block_m,
        tile_n=s1.tile_n,
        tile_k=s1.tile_k,
        num_waves=s1.num_waves,
        pipe_weights=s1.pipe_weights,
        mfma_amajor=s1.mfma_amajor,
        swizzle_a=s1.swizzle_a,
        async_a_copy=s1.async_a_copy,
        use_tile_resource=s1.use_tile_resource,
        waves_per_eu_hint=s1.waves_per_eu_hint,
        b_cache_modifier=max(int(s1.b_nt), 0),
        swiglu_limit=moe.swiglu_limit,
        num_cu=moe._s1_num_cu,
    )
