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
    SCH_ARR,
    SCH_HEAD,
    SCH_LDONE,
    SCH_RQ,
    SCH_TOT,
    SCH_TAIL,
    Arena,
    layout_lds_words,
    layout_phase,
    make_prepare_phase,
    memrealtime,
    memtime,
    post_rest_chunks,
    sched_layout_phase,
    sched_lds_words,
    scheduler_phase,
)


import os as _os

if _os.environ.get("S1_NO_URP") == "1":
    # A/B switch: the AMDGPU UnclusteredHighRPReschedule stage re-orders the GEMM K loop when the
    # kernel's register pressure is high (see the GEMM2 fix of 2026-10-29); the role code fused
    # into this kernel raises it. Set before any kernel compiles.
    from flydsl._mlir._mlir_libs._mlirDialectsFly import set_llvm_option_bool as _set_llvm_bool

    _set_llvm_bool("amdgpu-disable-unclustered-high-rp-reschedule", True)

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
    fuse_layout: bool = False, fuse_prepare: bool = False, prep_trace: bool = False,
    prep_split: int = 1, prep_unroll: int = 1, prep_stage_first: bool = False,
    prep_own_tokens: bool = False, ready_first: bool = False, local_first: bool = False,
    sched_layout: bool = False, prep_quant_skip: bool = False, layout_blocks: int = 0,
    sched_cache_need: bool = False, arrival_order: bool = False,
    pair_k: bool = False, prep_quant_lds_store: bool = False, prep_plan_blocks: int = 0,
):
    # fmt: on
    """Persistent GEMM1, one workgroup per CU, flat (m tile, n tile) work item per step.

    With ``gate_arena`` (the ``Arena`` of the prepare) the kernel does the receiving side
    itself instead of leaving it to a ``wait`` kernel:

    * workgroup 0 is the comm workgroup: it puts the chunks of my runs that ``prepare`` did
      not (``post_rest_chunks``; ``prepare`` put chunk 0), and the other workgroups compute;
    * a compute workgroup waits once at the start for the peers' route rows (``flags``), then
      before each tile for the chunks of every peer's run that the tile's expert needs
      (``need``, and the SDMA signal counters, which count from ``sig_base``). A tile whose
      chunks have landed starts at once, so the GEMM runs while later chunks are in flight.
      Beyond one cache invalidation at the start there is no fence: see the comment there.

    ``fuse_layout`` (with ``gate_arena``): the receiver's layout (``layout_phase``, what the
    separate layout kernel does) runs at the front of this kernel, in the compute workgroups,
    instead of as a kernel of its own. It replaces the start gate (it waits for the same
    flags), ends in a rendezvous that releases the tables it wrote, and only then is
    ``num_valid`` read to size the tile loop.

    ``fuse_prepare`` (needs ``fuse_layout``): the prepare (``prepare_phase``: plan, quant, route
    staging, the early puts) runs first in the same kernel, so a call is ONE launch. Workgroup
    0 puts the early chunks inside it, then waits for every worker to have quantised its rows
    (``qdone``, one-way arrivals) before it puts the rest; the workers go on to the layout.

    ``ready_first`` (with ``gate_arena``): the per-tile SDMA gate is gone. One wave of the comm
    workgroup is a TILE SCHEDULER (``scheduler_phase``): it watches the SDMA signal levels and
    appends m-tiles to a ready queue (``Stage1SdmaPrepare.sched``) in the order they become
    ready. A worker takes a ticket (``atomic_add`` on ``head``), q = ticket // n_tiles, waits
    for ``rq_tail > q`` (one lane, ``s_sleep``), reads ``rq[q]`` = the m-tile and runs its
    (n_tile = ticket % n_tiles) work item. It exits when its ticket is past the last m-tile.
    No worker reads a signal or ``need``. ``arm`` must zero ``sched`` (``sched=True``).

    ``local_first`` (needs ``ready_first``): ``layout_phase`` orders an expert's rows by source
    in rotated order (local rank first) and writes ``tsrc[tile]``; the scheduler then lets a
    tile start once the sources up to ``tsrc[tile]`` have landed (a tile of local rows only
    waits for nothing). The kernel must run with the layout of the same ``local_first``.

    ``sched_layout`` (needs ``ready_first`` and ``fuse_layout``): block 1 is layout + scheduler in
    one (``sched_layout_phase``: block barriers only, tables written through, then wave 0 runs
    the scheduler loop on them directly); blocks 2.. are workers and never run the layout; the
    comm block no longer hosts the scheduler. The grid is unchanged (``num_cu`` blocks).

    ``prep_quant_skip`` (``fuse_prepare``): the prepare's quant passes only load and quantise the
    tokens they store (see ``make_prepare_phase``).

    ``prep_quant_lds_store`` (``fuse_prepare``): the prepare's quantised-row stores go through an
    LDS swizzle so each lane stores 16 CONTIGUOUS bytes (the wave 1024 per instruction) instead
    of two half-line pieces 32 bytes apart; its staging is lent out of the same LDS pool
    (see ``make_prepare_phase``).

    ``prep_plan_blocks`` ``P`` (``fuse_prepare``, needs ``prep_own_tokens``): only the first ``P``
    worker blocks run the prepare's plan, which brings back the prepare's grid barrier B2
    (see ``make_prepare_phase``).

    ``layout_blocks`` ``LB`` (needs ``fuse_prepare``, ``ready_first``, ``prep_own_tokens``): the
    last ``LB`` workers own no tokens; right after the prepare's rendezvous B3 they run
    ``layout_phase`` among themselves (``nb = LB``, so its two rendezvous are small) while the
    other ``NW = num_cu - 1 - LB`` workers quantise. The scheduler waits for ``LB`` layout
    arrivals. EVERY worker then runs the ticket loop that ends on the scheduler's m-tile count
    (``SCH_TOT``), not on ``num_valid``: a quant worker is not past the layout's rendezvous.
    Why a tile never reads a row that is still being quantised: the scheduler wave sits behind
    comm's wait for ``qdone == q0 + NW`` (every quant block's S rows are retired, write-through)
    and publishes nothing before it, so a published tile, local rows included, has all its rows.

    ``trace`` (with ``gate_arena``): thread 0 of each workgroup writes the clock (100 MHz) to
    ``trace[bid * TRACE_STRIDE + ...]``: ``[0]`` kernel start, ``[1]`` start gate passed (flags),
    ``[2]`` end, ``[3]`` its number of work items, and from ``[8]`` per item ``(gate start, gate
    passed, item done)``, up to ``TRACE_ITEMS`` items. Workgroup 0 (comm): start and end only.
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
    if local_first and not ready_first:
        raise ValueError("local_first needs ready_first")
    if sched_layout and not (ready_first and fuse_layout):
        raise ValueError("sched_layout needs ready_first and fuse_layout")
    if sched_layout and lds_pool_bytes < layout_lds_words(gate_arena, True) * 4:
        raise ValueError("the LDS pool is too small for sched_layout_phase")
    if arrival_order and not (local_first and ready_first and fuse_layout and not sched_layout):
        raise ValueError("arrival_order needs local_first, ready_first, fuse_layout (not sched_layout)")
    if layout_blocks:
        if not (fuse_prepare and fuse_layout and ready_first and prep_own_tokens):
            raise ValueError("layout_blocks needs fuse_prepare, fuse_layout, ready_first and own_tokens")
        if sched_layout:
            raise ValueError("layout_blocks and sched_layout are exclusive")
        if not 0 < layout_blocks < num_cu - 1:
            raise ValueError(f"layout_blocks={layout_blocks} must be in (0, {num_cu - 1})")
    if (prep_quant_lds_store or prep_plan_blocks) and not fuse_prepare:
        raise ValueError("prep_quant_lds_store / prep_plan_blocks need fuse_prepare")
    if prep_plan_blocks and not prep_own_tokens:
        raise ValueError("prep_plan_blocks needs prep_own_tokens")
    NWQ = num_cu - 1 - layout_blocks   # workers that own tokens and quantise
    sched_words = 0
    if ready_first:
        sched_words = sched_lds_words((gate_arena.max_routes + sort_block_m - 1) // sort_block_m,
                                      (gate_npes * gate_epr if sched_cache_need else 0)
                                      + (gate_npes if arrival_order else 0))
        if sched_words * 4 > n_scale_bytes:
            raise ValueError(
                f"ready_first: the scheduler's per-tile table ({sched_words} words) does not fit "
                f"the A-scale LDS ({n_scale_bytes // 4} words) at sort_block_m={sort_block_m}")
        if gate_npes > 8 or gate_epr >= 65536:
            raise ValueError("ready_first: npes <= 8 and fewer than 65536 experts per rank")
    if fuse_layout:
        if not gate_arena:
            raise ValueError("fuse_layout needs gate_arena")
        if lds_pool_bytes < layout_lds_words(gate_arena, local_first) * 4:
            raise ValueError("the LDS pool is too small to lend the layout its words")
    prepare_phase = None
    if fuse_prepare:
        if not fuse_layout:
            raise ValueError("fuse_prepare needs fuse_layout")
        prepare_phase, pcfg = make_prepare_phase(
            rank=gate_rank, arena=gate_arena, epr=gate_epr, quant_blocks=num_cu - 1,
            route_push=False, trace=prep_trace, recv_layout=True, split=prep_split,
            unroll=prep_unroll, prep_block=total_threads, stage_first=prep_stage_first,
            fused=True, own_tokens=prep_own_tokens, quant_skip=prep_quant_skip,
            layout_blocks=layout_blocks, quant_lds_store=prep_quant_lds_store,
            plan_blocks=prep_plan_blocks,
        )
        if pcfg.nblocks != num_cu or pcfg.PB != total_threads:
            raise ValueError("the prepare's grid must be the GEMM1's grid")
        if lds_pool_bytes < pcfg.lds_bytes:
            # ``prep_quant_lds_store`` adds QLS_STRIDE (2304 B) per wave to this: 8 waves put
            # pcfg.lds_bytes at ~6 KB + 18 KB. The pool is sort_block_m * tile_n / 2 * 4 bytes
            # (128 KB at 128x512), so there is room; a small tile_n / sort_block_m could run out.
            raise ValueError(
                f"the LDS pool ({lds_pool_bytes} B) is too small to lend the prepare its "
                f"{pcfg.lds_bytes} bytes"
                + (" (prep_quant_lds_store staging included)" if prep_quant_lds_store else ""))

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
                      f"_g{int(bool(gate_npes))}_fl{int(fuse_layout)}_fp{int(fuse_prepare)}_ps{prep_split}u{prep_unroll}"
                      f"pk{int(pair_k)}sf{int(prep_stage_first)}pt{int(prep_trace)}ow{int(prep_own_tokens)}"
                      f"_rf{int(ready_first)}lf{int(local_first)}sl{int(sched_layout)}"
                      f"_cn{int(sched_cache_need)}ao{int(arrival_order)}_qs{int(prep_quant_skip)}lb{int(layout_blocks)}"
                      f"_ql{int(prep_quant_lds_store)}pb{int(prep_plan_blocks)}_pv{PHASES_VER}",
                 known_block_size=[total_threads, 1, 1])
    def kernel(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_w: fx.Tensor, tile_row_base: fx.Tensor,
        tile_input_base: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor,
        out_scale: fx.Tensor, rowmap: fx.Tensor, nvm: fx.Int32, arg_dev_comm: fx.Int64,
        arg_win: fx.Int64, arg_sig_base: fx.Int64, arg_expected: fx.Int64, arg_win_handle: fx.Int64,
        arg_cnt: fx.Int64, arg_trace: fx.Int64,
        arg_hist: fx.Int64, arg_cursor: fx.Int64, arg_lbar: fx.Int64, arg_srcmap: fx.Int64,
        arg_wts: fx.Int64, arg_eids: fx.Int64, arg_trb: fx.Int64, arg_tib: fx.Int64,
        arg_nv: fx.Int64, arg_met: fx.Int64, arg_ete: fx.Int64,
        arg_tsrc: fx.Int64, arg_sched: fx.Int64,
        p_ids: fx.Int64, p_x: fx.Int64, p_row_tok: fx.Int64, p_rowpos: fx.Int64, p_cum: fx.Int64,
        p_hist: fx.Int64, p_cursor: fx.Int64, p_bar: fx.Int64, p_pair_order: fx.Int64,
        p_pair_base: fx.Int64, p_local_hist: fx.Int64, p_my_base: fx.Int64, p_wts: fx.Int64,
        p_trace: fx.Int64, p_done: fx.Int64, p_stream: fx.Int64, p_qdone: fx.Int64,
        p_tokens: fx.Int32,
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
            """Tile loop size, read when it is needed: with ``fuse_layout`` it is written by
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
                mfma_amajor, async_a_copy, trb_rsrc, tib_rsrc, True, pair_k,
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
                mfma_amajor, async_a_copy, trb_rsrc, tib_rsrc, True, pair_k,
            )
            if const_expr(gate_npes):
                stamp(trace_buf, slot0 + fx.Int32(2))
                stamp_mt(trace_buf, slot0 + fx.Int32(4))

        def tot_ticket_loop():
            """The ticket loop of a worker that does not read ``num_valid``: ticket -> q = ticket //
            n_tiles; ONE lane polls ``rq_tail > q`` (the scheduler stores the m-tile count + 1 in
            ``SCH_TOT`` BEFORE its first tail, so ``q >= count`` means the ticket is past the last
            m-tile and the worker leaves), reads ``rq[q]`` only after it saw the tail, and the
            next ticket is claimed at tile start. One ``fence_system_acquire`` before the first
            tile: every table was written (and released) before the first tail, and the
            acquire drops whatever line of them -- or of last call's rows -- this XCD's L2 holds."""
            rf_lds = fx.Int64(fx.ptrtoint(lds.rf.ptr))
            head_addr = fx.Int64(arg_sched) + fx.Int64(SCH_HEAD * 4)
            tail_addr = fx.Int64(arg_sched) + fx.Int64(SCH_TAIL * 4)
            tot_addr = fx.Int64(arg_sched) + fx.Int64(SCH_TOT * 4)
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
            active = fx.Int32(1)
            wacc = fx.Int64(0)      # trace: queue wait of tiles 1.. (thread 0), 100 MHz ticks
            wfirst = fx.Int64(0)    # trace: the first wait
            tacc = fx.Int64(0)      # trace: do_tile time
            while active == fx.Int32(1):
                q = tk // fx.Int32(n_tiles)
                n_tile = tk - q * fx.Int32(n_tiles)
                nxt = fx.Int32(0)
                if tid == fx.Int32(0):
                    # st 1: slot q is published; 2: q is past the last m-tile (the
                    # scheduler stores the count + 1 BEFORE its first tail); 0: poll
                    if const_expr(trace):
                        tw0 = memrealtime()
                    st = fx.Int32(0)
                    while st == fx.Int32(0):
                        tl = fx.Int32(comm_ops.load_i32_global_system(tail_addr))
                        tt = fx.Int32(comm_ops.load_i32_global_system(tot_addr))
                        st = (tl > q).select(
                            fx.Int32(1),
                            ((tt > fx.Int32(0)) & (q >= tt - fx.Int32(1))).select(
                                fx.Int32(2), fx.Int32(0)))
                        if st == fx.Int32(0):
                            rocdl.s_sleep(1)
                    sel = fx.Int32(-1)
                    if st == fx.Int32(1):
                        # read only after the tail was seen: entry stored first
                        sel = fx.Int32(comm_ops.load_i32_global_system(
                            rq_addr + fx.Int64(q) * fx.Int64(4)))
                    comm_ops.store_i32_lds(rfw(RF_TILE), sel)
                    nxt = claim()
                    if const_expr(trace):
                        dw = memrealtime() - tw0
                        wacc = wacc + (item > fx.Int32(0)).select(dw, fx.Int64(0))
                        wfirst = (item == fx.Int32(0)).select(dw, wfirst)
                fx.barrier()
                m_tile = fx.Int32(comm_ops.load_i32_lds(rfw(RF_TILE)))
                if m_tile >= fx.Int32(0):
                    if item == fx.Int32(0):
                        # the ONE invalidate: every table was written before the
                        # first tail, so no line of them is cached after this
                        comm_ops.fence_system_acquire()
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
                active = (m_tile >= fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
            stamp(trace_buf, fx.Int32(2))
            stamp_i64(trace_buf, fx.Int32(3), fx.Int64(item - fx.Int32(1)))   # tiles this block ran
            stamp_i64(trace_buf, fx.Int32(5), wacc)      # trace: queue wait, tiles 1..
            stamp_i64(trace_buf, fx.Int32(6), wfirst)    # trace: first queue wait
            stamp_i64(trace_buf, fx.Int32(7), tacc)      # trace: sum of tile times

        q0 = fx.Int32(0)
        if const_expr(fuse_prepare):
            q0 = prepare_phase(
                tid=tid, bid=bid, tokens=p_tokens, arg_ids=p_ids, arg_x=p_x,
                arg_row_tok=p_row_tok, arg_rowpos=p_rowpos, arg_cnt=arg_cnt, arg_cum=p_cum,
                arg_hist=p_hist, arg_cursor=p_cursor, arg_bar=p_bar,
                arg_s=fx.Int64(arg_win) + fx.Int64(gate_arena.off_s),
                arg_dev_comm=arg_dev_comm, arg_win=arg_win_handle,
                arg_pair_order=p_pair_order, arg_pair_base=p_pair_base,
                arg_local_hist=p_local_hist, arg_my_base=p_my_base, arg_wts=p_wts,
                arg_trace=p_trace, arg_done=p_done, arg_stream=p_stream, arg_qdone=p_qdone,
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
                if const_expr(ready_first and not sched_layout):
                    # wave 0 of the comm workgroup is the tile scheduler (it has put its peer's
                    # chunks above). Its tile table lives in the ``A_scale`` LDS of this block:
                    # only ``do_tile`` uses that, and workgroup 0 runs no tile.
                    if tid < fx.Int32(64):
                        scheduler_phase(
                            rank=gate_rank, arena=gate_arena, sbm=sort_block_m,
                            nb=layout_blocks if layout_blocks else num_cu - 1,
                            wait_layout=fuse_layout, local_first=local_first,
                            max_tiles=(gate_arena.max_routes + sort_block_m - 1) // sort_block_m,
                            lane=tid, L=fx.Int64(fx.ptrtoint(lds.A_scale.ptr)),
                            win=fx.Int64(arg_win),
                            sdma=cco.DevComm(arg_dev_comm).sdma(), arg_sched=arg_sched,
                            nv_rsrc=nv_rsrc, expert_of=sched.expert_of,
                            tsrc_rsrc=ptr_buf_tensor(arg_tsrc, fx.Int32), sig_base=sig_base,
                            direct=False, a_nv=0, a_eids=0, a_tsrc=0,
                            publish_tot=layout_blocks > 0, trace=trace, trace_buf=trace_buf,
                            trace_base=bid * fx.Int32(TRACE_STRIDE), cache_need=sched_cache_need, arrival=arrival_order,
                            arg_arr=fx.Int64(arg_sched) + fx.Int64(SCH_ARR * 4),
                        )
            else:
                if const_expr(sched_layout):
                    # ---- sched_layout: block 1 = layout + scheduler (block barriers only),
                    # blocks 2.. = workers (prepare, then tickets; they never touch the tables
                    # before the ready queue says a tile is published)
                    if bid == fx.Int32(1):
                        sched_layout_phase(
                            rank=gate_rank, arena=gate_arena, sbm=sort_block_m,
                            max_tok=gate_arena.cap, nb=1, tid=tid, lbid=fx.Int32(0),
                            tpb=total_threads, L=fx.Int64(fx.ptrtoint(lds.pool.ptr)),
                            win=fx.Int64(arg_win), arg_expected=arg_expected,
                            arg_srcmap=arg_srcmap, arg_wts=arg_wts, arg_eids=arg_eids,
                            arg_trb=arg_trb, arg_tib=arg_tib, arg_nv=arg_nv, arg_met=arg_met,
                            arg_ete=arg_ete, local_first=local_first, arg_tsrc=arg_tsrc,
                        )
                        # the tables are in memory (write-through, barrier): wave 0 schedules,
                        # the other waves are done
                        if tid < fx.Int32(64):
                            scheduler_phase(
                                rank=gate_rank, arena=gate_arena, sbm=sort_block_m,
                                nb=num_cu - 1, wait_layout=False, local_first=local_first,
                                max_tiles=(gate_arena.max_routes + sort_block_m - 1) // sort_block_m,
                                lane=tid, L=fx.Int64(fx.ptrtoint(lds.A_scale.ptr)),
                                win=fx.Int64(arg_win),
                                sdma=cco.DevComm(arg_dev_comm).sdma(), arg_sched=arg_sched,
                                nv_rsrc=nv_rsrc, expert_of=sched.expert_of,
                                tsrc_rsrc=ptr_buf_tensor(arg_tsrc, fx.Int32), sig_base=sig_base,
                                direct=True, a_nv=arg_nv, a_eids=arg_eids, a_tsrc=arg_tsrc,
                                publish_tot=True, trace=trace, trace_buf=trace_buf,
                                trace_base=bid * fx.Int32(TRACE_STRIDE), cache_need=sched_cache_need, arrival=arrival_order,
                            arg_arr=fx.Int64(arg_sched) + fx.Int64(SCH_ARR * 4),
                            )
                        stamp(trace_buf, fx.Int32(2))
                    if bid > fx.Int32(1):
                        tot_ticket_loop()
                if const_expr(layout_blocks > 0):
                    # ---- layout_blocks: the last ``layout_blocks`` workers own no token. The
                    # prepare returned for them right after its rendezvous B3: they run the
                    # layout among themselves (their two rendezvous are small) while the other
                    # workers quantise. Nobody else touches the tables before the scheduler's
                    # tail says a tile is published, so no quant worker takes part in them.
                    if bid > fx.Int32(NWQ):
                        layout_phase(
                            rank=gate_rank, arena=gate_arena, sbm=sort_block_m,
                            max_tok=gate_arena.cap, nb=layout_blocks, tid=tid,
                            lbid=bid - fx.Int32(1 + NWQ), tpb=total_threads,
                            L=fx.Int64(fx.ptrtoint(lds.pool.ptr)), win=fx.Int64(arg_win),
                            arg_expected=arg_expected, arg_hist=arg_hist, arg_cursor=arg_cursor,
                            arg_bar=arg_lbar, arg_srcmap=arg_srcmap, arg_wts=arg_wts,
                            arg_eids=arg_eids, arg_trb=arg_trb, arg_tib=arg_tib, arg_nv=arg_nv,
                            arg_met=arg_met, arg_ete=arg_ete, final_release=True,
                            local_first=local_first, arg_tsrc=arg_tsrc, signal_done=True,
                            arg_ldone=fx.Int64(arg_sched) + fx.Int64(SCH_LDONE * 4),
                            arrival=arrival_order, arg_arr=fx.Int64(arg_sched) + fx.Int64(SCH_ARR * 4),
                        )
                    stamp(trace_buf, fx.Int32(1))
                    tot_ticket_loop()
                if const_expr(not (sched_layout or layout_blocks > 0)):
                    if const_expr(fuse_layout):
                        # the front of this kernel IS the layout: it waits for the same flags and
                        # does the one acquire, and its last rendezvous releases the tables the
                        # tiles below read from other workgroups
                        layout_phase(
                            rank=gate_rank, arena=gate_arena, sbm=sort_block_m, max_tok=gate_arena.cap,
                            nb=num_cu - 1, tid=tid, lbid=bid - fx.Int32(1), tpb=total_threads,
                            L=fx.Int64(fx.ptrtoint(lds.pool.ptr)), win=fx.Int64(arg_win),
                            arg_expected=arg_expected, arg_hist=arg_hist, arg_cursor=arg_cursor,
                            arg_bar=arg_lbar, arg_srcmap=arg_srcmap, arg_wts=arg_wts,
                            arg_eids=arg_eids, arg_trb=arg_trb, arg_tib=arg_tib, arg_nv=arg_nv,
                            arg_met=arg_met, arg_ete=arg_ete, final_release=True,
                            local_first=local_first, arg_tsrc=arg_tsrc, signal_done=ready_first,
                            arg_ldone=fx.Int64(arg_sched) + fx.Int64(SCH_LDONE * 4),
                            arrival=arrival_order, arg_arr=fx.Int64(arg_sched) + fx.Int64(SCH_ARR * 4),
                        )
                    else:
                        if tid < fx.Int32(gate_npes):
                            # also relaxed: the poll is a system-scope load, and the single
                            # ``fence_system_acquire`` below is what makes the data it guards visible.
                            # With ``acquire=True`` every poll iteration invalidated the whole L2.
                            comm_ops.spin_until_ge_i32_system(
                                fx.Int64(arg_win) + fx.Int64(gate_arena.off_flags) + fx.Int64(tid) * fx.Int64(4),
                                expected[fx.Int32(gate_npes) + tid],
                            )
                        fx.barrier()
                        # The only cache invalidation of the kernel. A tile does none after it has
                        # seen its chunks land (the SDMA signal is a relaxed load, and none was
                        # needed): SDMA writes memory without touching this GPU's L2, so a stale
                        # line would be one that this L2 already holds. Lines from earlier calls
                        # are gone by now; and a line of this call is only read after its chunk
                        # has landed, because a chunk starts on a 128 B line (``Arena.check``), so
                        # no line is pulled in early by a neighbour's read. Measured: an acquire
                        # per tile, of any scope, cost ~90 us of 0.96 ms (it invalidates L2, with
                        # the weights in it). With ``fuse_layout`` the layout's own acquire (at its
                        # start and in each of its rendezvous) plays this part.
                        comm_ops.fence_system_acquire()
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
               arg_tsrc: fx.Int64, arg_sched: fx.Int64,
               p_ids: fx.Int64, p_x: fx.Int64, p_row_tok: fx.Int64, p_rowpos: fx.Int64,
               p_cum: fx.Int64, p_hist: fx.Int64, p_cursor: fx.Int64, p_bar: fx.Int64,
               p_pair_order: fx.Int64, p_pair_base: fx.Int64, p_local_hist: fx.Int64,
               p_my_base: fx.Int64, p_wts: fx.Int64, p_trace: fx.Int64, p_done: fx.Int64,
               p_stream: fx.Int64, p_qdone: fx.Int64, p_tokens: fx.Int32, stream: fx.Stream):
    # fmt: on
        kernel(out, x, w, scale_w, tile_row_base, tile_input_base, expert_ids, num_valid_ids,
               out_scale, rowmap, nvm, arg_dev_comm, arg_win, arg_sig_base, arg_expected,
               arg_win_handle, arg_cnt, arg_trace, arg_hist, arg_cursor, arg_lbar, arg_srcmap,
               arg_wts, arg_eids, arg_trb, arg_tib, arg_nv, arg_met, arg_ete,
               arg_tsrc, arg_sched,
               p_ids, p_x, p_row_tok, p_rowpos, p_cum, p_hist, p_cursor, p_bar, p_pair_order,
               p_pair_base, p_local_hist, p_my_base, p_wts, p_trace, p_done, p_stream, p_qdone,
               p_tokens,
               value_attrs={
                   "rocdl.waves_per_eu": waves_per_eu_hint,
                   "rocdl.flat_work_group_size": f"{total_threads},{total_threads}",
               }).launch(grid=(num_cu, 1, 1), block=(total_threads, 1, 1), stream=stream)

    return launch


def run_gemm1_sdma(out, window, w, scale_w, tile_row_base, tile_input_base, expert_ids,
                   num_valid, out_scale, rowmap, nvm, stream, prep=None, trace=None, fuse=None,
                   **kwargs):
    """``window`` is ``[S | R]`` as fp8 ``[2 * n_rows, row_bytes]``; the others are the
    tensors of aiter's stage1 (``MegaMoEV2._s1_*`` and its dispatch op). With ``prep`` (a
    ``Stage1SdmaPrepare`` that has ``arm``-ed this call) the kernel waits for the data itself.
    ``fuse``: ``prep.fused_layout_args(moe)``; the kernel then also builds the receive layout
    (``fuse_layout``) and the separate layout kernel must NOT be run for this call."""
    gate = {}
    dev_comm = win = sig_base = expected = win_handle = cnt = 0
    if prep is not None:
        gate = dict(gate_arena=prep.arena, gate_rank=prep.rank)
        dev_comm, win, win_handle = prep.dev_comm.ptr, prep.mem.ptr, prep.win.handle
        sig_base, expected, cnt = prep.sig_base.data_ptr(), prep.expected.data_ptr(), prep.cnt.data_ptr()
    if fuse is not None and prep is None:
        raise ValueError("fuse needs prep")
    if prep is None:
        # no gate, no scheduler: the plain static loop whatever the flags say
        kwargs.pop("ready_first", None)
        kwargs.pop("local_first", None)
        kwargs.pop("sched_layout", None)
        kwargs.pop("layout_blocks", None)
        kwargs.pop("sched_cache_need", None)
        kwargs.pop("arrival_order", None)
    elif kwargs.get("ready_first") and not prep.sched_on:
        raise ValueError("ready_first needs Stage1SdmaPrepare(sched=True) (arm zeroes its counters)")
    fz = dict(fuse or {})
    if prep is not None:
        fz.setdefault("tsrc", prep.tsrc.data_ptr())
        fz.setdefault("sched", prep.sched.data_ptr())
    pz = fz.get("prepare")
    pre = {}
    if pz is not None:
        pre = dict(fuse_prepare=True, prep_trace=bool(prep.trace), prep_split=int(prep.split),
                   prep_unroll=int(prep.unroll), prep_stage_first=bool(prep.stage_first),
                   prep_own_tokens=bool(prep.own_tokens), prep_quant_skip=bool(prep.quant_skip),
                   prep_quant_lds_store=bool(prep.quant_lds_store),
                   prep_plan_blocks=int(prep.plan_blocks))
    launch = compile_gemm1_sdma(
        row_bytes=int(window.shape[1]), source_rows=int(window.shape[0]), trace=trace is not None,
        fuse_layout=fuse is not None, **pre, **gate, **kwargs,
    )
    fuse_ptrs = [fx.Int64(fz.get(k, 0)) for k in (
        "hist", "cursor", "lbar", "srcmap", "wts", "eids", "trb", "tib", "nv", "met", "ete",
        "tsrc", "sched")]
    pz = pz or {}
    fuse_ptrs += [fx.Int64(pz.get(k, 0)) for k in (
        "ids", "x", "row_tok", "rowpos", "cum", "hist", "cursor", "bar", "pair_order",
        "pair_base", "local_hist", "my_base", "wts", "trace", "done", "stream", "qdone")]
    fuse_ptrs.append(fx.Int32(int(pz.get("tokens", 0))))
    _run_compiled(
        launch, out, window, w, scale_w, tile_row_base, tile_input_base, expert_ids, num_valid,
        out_scale, rowmap, fx.Int32(nvm), fx.Int64(dev_comm), fx.Int64(win), fx.Int64(sig_base),
        fx.Int64(expected), fx.Int64(win_handle), fx.Int64(cnt),
        fx.Int64(trace.data_ptr() if trace is not None else 0), *fuse_ptrs, stream,
    )
