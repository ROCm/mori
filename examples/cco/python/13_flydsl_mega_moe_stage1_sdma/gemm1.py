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
its rows through ``rowmap`` (``sdma_dispatch.Arena``)::

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
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

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
    wait_lds_barrier,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    buf_copy_load,
    ptr_buf_tensor,
)

from sdma_dispatch import (
    PHASES_VER,
    SCH_HEAD,
    SCH_LDONE,
    SCH_RQ,
    SCH_TAIL,
    SCH_XHEAD,
    SCH_XLISTS,
    SCH_XTAIL,
    Arena,
    layout_lds_words,
    layout_phase,
    make_prepare_phase,
    memrealtime,
    memtime,
    post_rest_chunks,
    sched_lds_words,
    sched_xcd_base,
    sched_xcd_cap,
    scheduler_phase,
)
from tile_hoist import (
    HoistATileLoader,
    do_tile_split,
    epi_stage_c,
    sa_pack_issue,
    sa_pack_rows,
    sa_pack_store,
    stage_issue,
    stage_store,
    store_quant_remat,
    tile_main,
)

#: ``trace``: int64 slots per workgroup, and work items recorded per workgroup (8 + 3 * items).
TRACE_STRIDE = 2048
TRACE_SLOTS = (
    5  # per item: gate start, gate passed, done (100 MHz); passed, done (s_memtime)
)
TRACE_ITEMS = (TRACE_STRIDE - 8) // TRACE_SLOTS
RF_TILE, RF_NEXT = 0, 1  # ``ready_first``: words of the ``rf`` LDS array
#: ``hoist_prologue``: words of ``rf``. HA_*: the NEXT tile (ticket, m-tile, its table entries),
#: written by thread 0, read by all after a barrier. HB_*: the ticket after it, its m-tile and
#: whether its ready-queue slot was read (thread 0 only).
HA_T, HA_M, HA_TRB, HA_TIB, HA_EXP, HB_T, HB_M, HB_OK = range(8)
#: ``xcd_order``: word of ``rf`` = the XCD list this workgroup claims from (thread 0)
HX_Q = 8
#: ``xcd_steal`` 2: word of ``rf`` = 1 once thread 0 has seen the whole ready queue published
HX_F = 9
#: in the kernel name and closure of a ``hoist_prologue`` build: bump when ``tile_hoist`` changes
HOIST_VER = 9


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
                        self._row_map_rsrc,
                        tile_row_base_i32 + row,
                        lane,
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


class PackedWindowAScaleLoader(WindowAScaleLoader):
    """``sa_pack``: the A scales in LDS as ``tile_hoist.sa_pack_store`` writes them, one 16-byte
    record per (k-step, lane // 16, lane % 16) holding the packed word of each scale group, so
    ``load_step`` is one ``ds_read_b128`` instead of 16 ``ds_read_u8`` (same words)."""

    def load_step(self, lds_ascale, kstep_i32):
        rec = (kstep_i32 * fx.Int32(4) + fx.Int32(self._lane // 16)) * fx.Int32(
            16
        ) + fx.Int32(self._lane % 16)
        ptr = fx.add_offset(lds_ascale.ptr, fx.make_int_tuple(rec * fx.Int32(16)))
        v = Vec(
            fx.make_view(fx.recast_iter(fx.Uint8, ptr), fx.make_layout(16, 1)).load()
        ).bitcast(fx.Int32)
        return [fx.Int32(v[g]) for g in range(self._n_groups)]


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
    hoist_prologue: bool = False, kpipe: int = 0, kunroll: int = 1, xcd_order: bool = False,
    xcd_steal: int = 1, fin_off: bool = False, sa_pack: bool = False, a_dbuf: int = 0,
    epi_remat: int = 0,
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

    ``tile_opt`` (``run_stage1.py --tile-opt``) is the set of switches below that pays at each
    sort_block_m: all but ``kunroll`` 2 at 128, all but ``sa_pack`` / ``epi_remat`` at 64.

    ``hoist_prologue`` (with ``ready_first``): the next tile's ready-slot read and prologue run inside
    the current tile (``tile_hoist``), each step issued one window before its result is used (windows:
    the K loop, the two halves of the epilogue). Thread 0: after the K loop it claims the ticket after
    next and reads the tail; between the epilogue halves it reads that ticket's ready-queue slot; at the
    next K-loop start (after its first loads) it reads the next tile's table entries (tile_row_base,
    tile_input_base, expert). All lanes: after the K loop the next tile's rowmap words; between the
    epilogue halves A k-step 0 and the A scales, global -> registers; after the epilogue's closing
    barrier both to LDS (the same bytes and places as ``do_tile``). The next tile then starts with an
    LDS barrier only. A ready-queue slot not yet published when it is read is read the old way,
    blocking, after the next K loop. Same LDS.

    ``kpipe`` (with ``hoist_prologue``; 0 = off): 1 = the K loop of ``tile_main`` loads the next A
    k-step through registers instead of ``buffer_load ... lds`` (``tile_hoist``'s ``KPIPE``).

    ``kunroll`` (with ``kpipe`` 1; 1 = off): 2 = the K loop body is two k-steps with two B register sets
    that alternate (``tile_hoist``), so no loop-carried B copies. Same arithmetic.

    ``xcd_order`` (with ``hoist_prologue``): XCD-affine tile order. The scheduler also appends each
    published m-tile's work items to 8 per-XCD lists (n-tile n of expert e to list (e * N + n) mod 8,
    ``scheduler_phase``), and workgroup ``bid`` (on XCD ``bid % 8``) claims ONE work item at a time
    from list ``bid & 7``, so slice (e, n) always runs on one XCD and its weights are fetched into
    that XCD's L2 only. A claim is a ticket ``j * 8 + list``; it resolves when the list's tail passes
    j (the ready-queue semantics: only published m-tiles are listed). When the whole queue is
    published and the list is exhausted, thread 0 steals from another list.

    ``xcd_steal`` (with ``xcd_order``): 1 = once the whole queue is published, the next lists in turn
    (y + 1, y + 2, ...: one round trip each), resolved after the next K loop, in front of that tile's
    epilogue. 2 = one round trip reads all 8 heads and tails and claims from the list with the most
    left (``xsteal``); the claim after the K loop also reads the ready-queue tail, so a ticket found
    dead (the queue published, its list exhausted) is stolen for (or found to be the end) after the
    epilogue it was read in, not after the next K loop. A ticket still unresolved after the next K
    loop (rare: the queue was published in between) takes the path of 1. (``xsteal`` at that second
    place too costs 7 SGPRs, which reschedule the K loop.)

    ``fin_off`` (with ``ready_first``; same results, off = False): the final k-step's A buffer parity
    is opaque to LLVM (``tile_hoist.tile_main``'s ``FINOFF``), so its A ``ds_read`` addresses are not
    folded into constants hoisted out of the tile loop (at sort_block_m 128 they spill and are reloaded
    every tile). With ``hoist_prologue`` in ``tile_main``; without, the ready-first tile loop runs
    ``tile_hoist.do_tile_split`` (aiter's ``do_tile`` in pieces) with it instead of aiter's ``do_tile``.

    ``sa_pack`` (with ``hoist_prologue``, sort_block_m 128; same results, off = False): the A scales go
    to LDS packed (``tile_hoist.sa_pack_store``), one 16-byte record per (k-step, lane) holding the
    packed word of each scale group, so a k-step reads its A scales with one ``ds_read_b128``
    (``PackedWindowAScaleLoader``) instead of 16 ``ds_read_u8`` with a wait after each pair.

    ``a_dbuf`` (with ``kpipe`` 1; same results, off = 0): 1 = the K loop's A operand is
    double-buffered in registers: MFMA group g + 1's A ``ds_read`` pair is issued before group g's
    MFMAs, which are fenced with ``sched_barrier`` (``tile_hoist._pipe_am_dbuf``).

    ``epi_remat`` (with ``hoist_prologue``; same results, off = 0): 1 = the lane constants of the
    sa-pack writer (with ``sa_pack``: rowmap words, A-scale loads, packed LDS records) and of the store
    pass come from ``tile_hoist.opaque_lane``, so they are recomputed in the epilogue. Otherwise LLVM
    hoists them out of the tile loop, spills them, and reloads them each tile: each reload's ``vmcnt``
    wait also waits for the hoisted loads and the output stores in flight (8 serialised waits in the
    sa-pack writer).

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
        sched_words = sched_lds_words((gate_arena.max_routes + sort_block_m - 1) // sort_block_m,
                                      xcd_order)
        if sched_words * 4 > n_scale_bytes:
            raise ValueError(
                f"ready_first: the scheduler's per-tile table ({sched_words} words) does not fit "
                f"the A-scale LDS ({n_scale_bytes // 4} words) at sort_block_m={sort_block_m}")
        if gate_npes > 8 or gate_epr >= 65536:
            raise ValueError("ready_first: npes <= 8 and fewer than 65536 experts per rank")
    n_rounds = sort_block_m * (a_k_step_bytes // 16) // total_threads   # A k-step copy rounds
    if hoist_prologue:
        if not ready_first:
            raise ValueError("hoist_prologue needs ready_first")
        if not (pipe_weights and async_a_copy and mfma_amajor):
            raise ValueError("hoist_prologue: only pipe_weights + async_a_copy + mfma_amajor")
        if sort_block_m * (a_k_step_bytes // 16) % total_threads or not 1 <= n_rounds <= 4:
            raise ValueError("hoist_prologue: the A k-step must be 1..4 whole copy rounds")
        if max(16, model_dim // 32 // 16) != a_k_step_bytes // 16:
            raise ValueError("hoist_prologue: the A-scale rounds must be the A copy rounds")
    if kpipe and not hoist_prologue:
        raise ValueError("kpipe needs hoist_prologue")
    if xcd_order and not hoist_prologue:
        raise ValueError("xcd_order needs hoist_prologue")
    if xcd_steal not in (1, 2) or (xcd_steal != 1 and not xcd_order):
        raise ValueError("xcd_steal: 1, or 2 with xcd_order")
    if fin_off and not ready_first:
        raise ValueError("fin_off needs ready_first")
    if sa_pack and not (hoist_prologue and sort_block_m == 128 and num_waves * 64 == 512
                        and model_dim % 512 == 0 and model_dim <= 8192):
        raise ValueError("sa_pack: hoist_prologue, sort_block_m 128, 512 threads, model_dim a multiple "
                         "of 512 up to 8192")
    if fin_off and not (pipe_weights and async_a_copy and mfma_amajor):
        raise ValueError("fin_off: only pipe_weights + async_a_copy + mfma_amajor")
    if xcd_order and n_tiles > 32:
        raise ValueError("xcd_order: the XCD lists hold up to 4 n-tiles of an m-tile (N <= 32)")
    if epi_remat not in (0, 1) or (epi_remat and not hoist_prologue):
        raise ValueError("epi_remat: 0 / 1, with hoist_prologue")
    if kpipe not in (0, 1):
        raise ValueError("kpipe: 0 / 1")
    if a_dbuf not in (0, 1) or (a_dbuf and kpipe != 1):
        raise ValueError("a_dbuf: 0, or 1 with kpipe 1")
    if kunroll not in (1, 2) or (kunroll != 1 and kpipe != 1):
        raise ValueError("kunroll: 1, or 2 with kpipe 1")
    # the tile split (``do_tile_split``) only covers this config; aiter turns it off at small batches
    split_ok = bool(pipe_weights and async_a_copy and mfma_amajor)
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

    ALoader = HoistATileLoader if hoist_prologue else ATileLoader
    HOIST_KEY = HOIST_VER  # a closure scalar, so the JIT cache key sees a tile_hoist change
    KPIPE_KEY = kpipe
    KUNROLL_KEY = kunroll
    XCD_KEY = int(xcd_order)
    XS2 = bool(xcd_order and xcd_steal == 2)
    XSTEAL_KEY = int(XS2)
    FINOFF_KEY = int(fin_off)
    SAPACK_KEY = int(sa_pack)
    ADBUF_KEY = int(a_dbuf)
    ER_KEY = int(epi_remat)
    XB = sched_xcd_base(gate_arena) if xcd_order else 0    # words of ``sched``
    XCAP = sched_xcd_cap(gate_arena) if xcd_order else 0
    rf_bytes = 48 if xcd_order else 32
    if ready_first:
        @fx.struct
        class SharedStorage:
            pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
            A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]
            # survives ``do_tile`` (which owns the pool): the chosen m-tile and the next ticket
            rf: fx.Array[fx.Int8, rf_bytes, 16]
    else:
        @fx.struct
        class SharedStorage:
            pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
            A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    # fmt: off
    @flyc.kernel(name=f"stage1_sdma_gemm1_t{sort_block_m}x{tile_n}_rb{row_bytes}_r{expert_offset}_tr{int(trace)}"
                      f"_g{int(bool(gate_npes))}_fp{int(fuse_prepare)}_ps{prep_split}u{prep_unroll}"
                      f"pt{int(prep_trace)}_rf{int(ready_first)}np{int(no_plan)}_pv{PHASES_VER}"
                      + (f"_hp{HOIST_VER}" if hoist_prologue else "")
                      + (f"_kp{kpipe}" if kpipe else "")
                      + (f"_ku{kunroll}" if kunroll != 1 else "")
                      + ("_xo1" if xcd_order else "")
                      + ("_xs2" if XS2 else "")
                      + ("_fo1" if fin_off else "")
                      + ("_sp1" if sa_pack else "")
                      + (f"_ad{a_dbuf}" if a_dbuf else "")
                      + (f"_er{epi_remat}" if epi_remat else ""),
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
        _ = FINOFF_KEY
        _ = SAPACK_KEY
        _ = ADBUF_KEY
        _ = ER_KEY
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        if const_expr(hoist_prologue):
            _ = HOIST_KEY
            _ = KPIPE_KEY
            _ = KUNROLL_KEY
            _ = XCD_KEY
            _ = XSTEAL_KEY
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        pool_addr = fx.Int64(fx.ptrtoint(lds.pool.ptr))
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
        a_gather = ALoader(
            row_bytes=row_bytes, sort_block_m=sort_block_m, k_step_bytes=a_k_step_bytes,
            total_threads=total_threads, swizzle=swizzle_a, x_tensor=x, async_copy=async_a_copy,
            indexed_input=True, row_map_rsrc=row_map_rsrc, source_rows=source_rows,
        )
        a_s2r = AS2RLoader(k_step_bytes=a_k_step_bytes, swizzle=swizzle_a)
        b_loader = BWeightLoader(
            w_rsrc=w_rsrc, num_acc_n=num_acc_n, model_dim=model_dim, cache_modifier=b_cache_modifier
        )
        b_scale = BScaleLoader(scale_rsrc=sw_rsrc, num_acc_n=num_acc_n, model_dim=model_dim)
        a_scale = (PackedWindowAScaleLoader if sa_pack else WindowAScaleLoader)(
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
            wait for. The trace build runs the same tile split (``tile_hoist.do_tile_split``):
            per item ``[0]`` start, ``[1]`` K loop starts, ``[3]`` K loop done, ``[2]`` done.
            A config the split does not cover (``split_ok`` false) traces the item's start twice
            around aiter's ``do_tile``."""
            slot0 = (item < fx.Int32(TRACE_ITEMS)).select(fx.Int32(8) + item * fx.Int32(TRACE_SLOTS), fx.Int32(TRACE_STRIDE - TRACE_SLOTS))
            stamp(trace_buf, slot0)
            if const_expr(trace and split_ok):
                do_tile_split(
                    m_tile, n_wave_base + n_tile * fx.Int32(tile_n), sched.expert_of(m_tile), sched,
                    a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf, a_scale_lds,
                    a_lds_i32, k_iters, m_repeat, num_acc_n, a_k_step_bytes, trb_rsrc, tib_rsrc,
                    trace_buf, bid * fx.Int32(TRACE_STRIDE) + slot0, True, FINOFF_KEY,
                )
            elif const_expr(fin_off):
                # aiter's do_tile in pieces, the final k-step's A offset opaque (no trace stamps)
                do_tile_split(
                    m_tile, n_wave_base + n_tile * fx.Int32(tile_n), sched.expert_of(m_tile), sched,
                    a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf, a_scale_lds,
                    a_lds_i32, k_iters, m_repeat, num_acc_n, a_k_step_bytes, trb_rsrc, tib_rsrc,
                    trace_buf, fx.Int32(0), False, 1,
                )
            else:
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

        # ``hoist_prologue``: every call that touches a Python object (loader, epilogue, scheduler)
        # lives here at kernel scope. FlyDSL carries an object whose method is called inside a
        # dynamic if / while as that region's state, and a loader or scheduler cannot be an IR value.
        def hp_rows(tib):
            return a_gather.rows_raw(tib)

        def hp_sa_rows(tib):
            # ``sa_pack``: the rowmap words of this lane's A-scale rows (none without it)
            if const_expr(sa_pack):
                return sa_pack_rows(a_gather, tib, bool(epi_remat))
            return []

        def hp_issue(raw, sar):
            if const_expr(sa_pack):
                a_regs, _ = stage_issue(a_gather, a_scale, raw, scales=False)
                return a_regs, sa_pack_issue(a_scale, sar, bool(epi_remat))
            return stage_issue(a_gather, a_scale, raw)

        def hp_stage_store(sv):
            if const_expr(sa_pack):
                a_gather.store(a_buf, sv[0], 0)
                sa_pack_store(a_scale_lds, sv[1], bool(epi_remat))
            else:
                stage_store(a_gather, a_buf, a_scale_lds, sv)

        def hp_expert(m):
            return fx.Int32(sched.expert_of(m))

        def hp_b_row(e, n_base):
            return sched.gate_base_row(e) + n_base

        def hp_tile(b_row, r0, r1, r2, r3, slot0, hook):
            a_gather.set_rows([r0, r1, r2, r3][:n_rounds])
            return tile_main(
                b_row, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, a_buf, a_scale_lds,
                a_lds_i32, k_iters, m_repeat, num_acc_n, a_k_step_bytes, 63, trace_buf,
                bid * fx.Int32(TRACE_STRIDE) + slot0, trace, hook, kpipe, kunroll, FINOFF_KEY, ADBUF_KEY,
            )

        def hp_epi_c(acc):
            return epi_stage_c(epi, acc)

        def hp_store_quant(hm, trb, n_base, ce):
            if const_expr(epi_remat):
                store_quant_remat(epi, hm, trb, n_base, ce[0], ce[1], ce[2])
            else:
                epi._store_quant_vec8(hm, trb, n_base, ce[0], ce[1], ce[2])

        q0 = fx.Int32(0)
        if const_expr(fuse_prepare):
            q0 = prepare_phase(
                tid=tid, bid=bid, tokens=p_tokens, arg_ids=p_ids, arg_x=p_x,
                arg_row_tok=p_row_tok, arg_rowpos=p_rowpos, arg_cnt=arg_cnt, arg_cum=p_cum,
                arg_hist=p_hist, arg_cursor=p_cursor, arg_bar=p_bar,
                arg_s=fx.Int64(arg_win) + fx.Int64(gate_arena.off_s),
                arg_dev_comm=arg_dev_comm, arg_win=arg_win_handle, arg_wts=p_wts,
                arg_trace=p_trace, arg_done=p_done, arg_qdone=p_qdone,
                L=pool_addr,
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
                            xcd_n=n_tiles if xcd_order else 0, xcd_base=XB, xcd_cap=XCAP,
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
                    L=pool_addr, win=fx.Int64(arg_win),
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

                    if const_expr(not hoist_prologue):
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
                        # ``tile_hoist``: each step of the next tile's front runs one window ahead of its use
                        # (docstring). Thread 0 keeps the ticket state in ``rf`` (HA_*, HB_*).
                        def rd(i):
                            return fx.Int32(comm_ops.load_i32_lds(rfw(i)))

                        def wr(i, v):
                            comm_ops.store_i32_lds(rfw(i), v)

                        def resolve(t):
                            """thread 0: ticket ``t``'s m-tile, after waiting for its ready-queue slot
                            (as the plain loop does); 0 past the last work item."""
                            m = fx.Int32(0)
                            if t < total_work:
                                q = t // fx.Int32(n_tiles)
                                tl = fx.Int32(comm_ops.load_i32_global_system(tail_addr))
                                while tl <= q:
                                    rocdl.s_sleep(1)
                                    tl = fx.Int32(comm_ops.load_i32_global_system(tail_addr))
                                m = fx.Int32(comm_ops.load_i32_global_system(rq_addr + fx.Int64(q) * fx.Int64(4)))
                            return m

                        # ``xcd_order``: the 8 lists' words (``sched_xcd_base``). The offset is 32-bit math on
                        # a dynamic y, so no ``arg_sched + constant`` pair is hoisted into SGPRs live across
                        # the K loop (the kp1 kernel is at 95 SGPRs: 3 more pairs spill and change the K loop).
                        def xhead(y):
                            return fx.Int64(arg_sched) + fx.Int64((fx.Int32(XB + SCH_XHEAD) + y * fx.Int32(32)) * fx.Int32(4))

                        def xtail(y):
                            return fx.Int64(arg_sched) + fx.Int64((fx.Int32(XB + SCH_XTAIL) + y * fx.Int32(32)) * fx.Int32(4))

                        def xentry(y, j):
                            return fx.Int64(arg_sched) + fx.Int64((fx.Int32(XB + SCH_XLISTS) + y * fx.Int32(XCAP) + j) * fx.Int32(4))

                        def xclaim():
                            """thread 0: a ticket j * 8 + y from list y = ``rf[HX_Q]``"""
                            y = rd(HX_Q)
                            return fx.Int32(comm_ops.atomic_add_agent(xhead(y), fx.Int32(1))) * fx.Int32(8) + y

                        def xldm(base, words):
                            """thread 0 (``xcd_steal`` 2): a monotonic, NOT volatile load of word ``words``
                            past ``base``: a volatile load is waited for before the next one is issued, these
                            are in flight together"""
                            return fx.Int32(fx.generic_load(
                                comm_ops._to_ptr_global(base + fx.Int64(words * 4)), volatile=False,
                                memory_order=fx.AtomicOrdering.Monotonic, syncscope=fx.rocdl.SyncScope.OneAs))

                        def xdz(seed):
                            """thread 0: 0 as a value the compiler must treat as divergent (mbcnt_lo(seed, 0) =
                            the set bits of ``seed`` below this lane: none for lane 0), so the addresses, loads
                            and selection built on it stay in VGPRs; ``seed`` (a value of this tile) keeps it
                            from being hoisted out of the tile loop, live across the K loop."""
                            return fx.Int32(rocdl.mbcnt_lo(T.i32, fx.Int32(seed).ir_value(), fx.Int32(0).ir_value()))

                        def xbase(seed):
                            """``sched`` + 4 * the lists' base word, divergent (``xdz``): the loads and the
                            selection on them stay in VGPRs"""
                            return fx.Int64(arg_sched) + fx.Int64(fx.Int32(XB * 4) + xdz(seed))

                        def xprobe(y):
                            """thread 0 (``xcd_steal`` 2, the whole queue published: the tails are final): ONE
                            round trip reads the 8 heads and tails (one base, immediate offsets) and list y's
                            tail. Returns key = max over lists i of ((tail - head) << 3 | i), so the list with the
                            most items left is key & 7 and has one iff key >= 8; and list y's tail. (A max, no
                            selects: a select chain kept 16 compare masks in SGPR pairs, and they spilled.)"""
                            base = xbase(y)
                            hs = [xldm(base, SCH_XHEAD + i * 32) for i in range_constexpr(8)]
                            ts = [xldm(base, SCH_XTAIL + i * 32) for i in range_constexpr(8)]
                            ty = xldm(base + fx.Int64(y * fx.Int32(128)), SCH_XTAIL)
                            key = fx.Int32(0)
                            for i in range_constexpr(8):
                                ki = (ts[i] - hs[i]) * fx.Int32(8) + fx.Int32(i)
                                key = (ki > key).select(ki, key)   # a max (v_max_i32)
                            return key, ty

                        def xsteal(raw):
                            """thread 0 (``xcd_steal`` 2, the whole queue published): ticket ``raw`` (j * 8 + y)'s
                            work item if list y's final tail passes j; else one claimed from the list with the
                            most left (``xprobe``; the claim and that list's tail in one round trip; probe again
                            if the claim loses a race), which becomes this workgroup's list (``rf[HX_Q]``);
                            ``total_work`` when no list has one left (one round trip to learn it)."""
                            y = raw % fx.Int32(8)
                            j = raw // fx.Int32(8)
                            w = total_work
                            go = fx.Int32(1)
                            key = fx.Int32(0)
                            ty = fx.Int32(0)
                            while go != fx.Int32(0):
                                key, ty = xprobe(y)
                                if ty > j:
                                    w = fx.Int32(comm_ops.load_i32_global_system(xentry(y, j)))
                                    go = fx.Int32(0)
                                elif key < fx.Int32(8):
                                    go = fx.Int32(0)
                                else:
                                    b = key & fx.Int32(7)
                                    jb = fx.Int32(comm_ops.atomic_add_agent(xhead(b), fx.Int32(1)))
                                    bt = fx.Int32(comm_ops.load_i32_global_system(xtail(b)))
                                    if jb < bt:
                                        w = fx.Int32(comm_ops.load_i32_global_system(xentry(b, jb)))
                                        wr(HX_Q, b)
                                        go = fx.Int32(0)
                            return w

                        def xresolve(raw):
                            """thread 0: ticket ``raw``'s work item (m_tile * N + n), waiting for the list's
                            tail; ``total_work`` once every list is exhausted. Once the whole ready queue is
                            published (``fin``: the list tails are final) and list y is exhausted, steal: the
                            next list (y + 1, y + 2, ...; an atomic and its tail per step, in parallel), and go
                            on claiming from it (``rf[HX_Q]``); back at this XCD's own list = done. (A scan of
                            the 8 lists for the fullest costs 10 SGPRs, spills and changes the K loop.)"""
                            total_q = total_work // fx.Int32(n_tiles)
                            y = raw % fx.Int32(8)
                            j = raw // fx.Int32(8)
                            w = total_work
                            go = fx.Int32(1)
                            fin = fx.Int32(0)
                            if const_expr(XS2):
                                fin = rd(HX_F)
                            tl = fx.Int32(comm_ops.load_i32_global_system(xtail(y)))
                            while go != fx.Int32(0):
                                if tl > j:
                                    w = fx.Int32(comm_ops.load_i32_global_system(xentry(y, j)))
                                    go = fx.Int32(0)
                                elif fin == fx.Int32(0):
                                    gt = fx.Int32(comm_ops.load_i32_global_system(tail_addr))
                                    tl = fx.Int32(comm_ops.load_i32_global_system(xtail(y)))
                                    if gt >= total_q:
                                        # tl may predate gt: read it again (issued after gt returned)
                                        fin = fx.Int32(1)
                                        tl = fx.Int32(comm_ops.load_i32_global_system(xtail(y)))
                                        if const_expr(XS2):
                                            wr(HX_F, fx.Int32(1))
                                    else:
                                        rocdl.s_sleep(1)
                                else:
                                    y = (y + fx.Int32(1)) & fx.Int32(7)
                                    if y == (bid & fx.Int32(7)):
                                        go = fx.Int32(0)
                                    else:
                                        j = fx.Int32(comm_ops.atomic_add_agent(xhead(y), fx.Int32(1)))
                                        tl = fx.Int32(comm_ops.load_i32_global_system(xtail(y)))
                                        wr(HX_Q, y)
                            return w

                        def publish(t, m):
                            """thread 0: ticket ``t`` (m-tile ``m``) is the next tile: HA_* = it and its
                            table entries."""
                            wr(HA_T, t)
                            wr(HA_M, m)
                            wr(HA_TRB, fx.Int32(trb_rsrc[m]))
                            wr(HA_TIB, fx.Int32(tib_rsrc[m]))
                            wr(HA_EXP, hp_expert(m))

                        def take(t):
                            """thread 0: ticket ``t`` is the next tile (blocking)"""
                            if const_expr(xcd_order):
                                w = xresolve(t)
                                publish(w, (w < total_work).select(w // fx.Int32(n_tiles), fx.Int32(0)))
                            else:
                                publish(t, resolve(t))

                        def claim_t():
                            if const_expr(xcd_order):
                                return xclaim()
                            return claim()

                        def hook():
                            """In ``tile_main``, after the K loop's first loads: the next tile's table
                            entries, if its ready-queue slot was read in the last epilogue."""
                            if tid == fx.Int32(0):
                                if rd(HB_OK) != fx.Int32(0):
                                    publish(rd(HB_T), rd(HB_M))

                        # first tile: ticket, slot and front blocking, as the plain loop; the ticket after it
                        # is claimed now and its slot read after the first K loop
                        if tid == fx.Int32(0):
                            if const_expr(xcd_order):
                                wr(HX_Q, bid & fx.Int32(7))
                            if const_expr(XS2):
                                wr(HX_F, fx.Int32(0))
                            take(claim_t())
                            wr(HB_T, claim_t())
                            wr(HB_M, fx.Int32(0))
                            wr(HB_OK, fx.Int32(0))
                        fx.barrier()
                        tk = rd(HA_T)
                        r0 = fx.Int32(0)
                        r1 = fx.Int32(0)
                        r2 = fx.Int32(0)
                        r3 = fx.Int32(0)
                        # a dead first ticket (``tk >= total_work``): ``resolve()`` returned the m-tile 0
                        # placeholder without waiting for the ready queue, so m-tile 0's window rows may
                        # not have landed (or may not exist). The loop below does not run: gather nothing.
                        if tk < total_work:
                            raw0 = hp_rows(rd(HA_TIB))
                            hp_stage_store(hp_issue(raw0, hp_sa_rows(rd(HA_TIB))))
                            raw0 = raw0 + [fx.Int32(0)] * (4 - n_rounds)
                            r0 = raw0[0]
                            r1 = raw0[1]
                            r2 = raw0[2]
                            r3 = raw0[3]
                        item = fx.Int32(0)
                        tacc = fx.Int64(0)
                        while tk < total_work:
                            # this tile: its ticket tk, the HA_* words (read before tile_main's first barrier,
                            # thread 0 rewrites them after it), its rowmap words r0..r3; A k-step 0 and its A
                            # scales are in LDS (written in the last epilogue)
                            hm = rd(HA_M)
                            trb = rd(HA_TRB)
                            hn = tk - (tk // fx.Int32(n_tiles)) * fx.Int32(n_tiles)
                            n_base = n_wave_base + hn * fx.Int32(tile_n)
                            b_row = hp_b_row(rd(HA_EXP), n_base)
                            slot0 = (item < fx.Int32(TRACE_ITEMS)).select(fx.Int32(8) + item * fx.Int32(TRACE_SLOTS), fx.Int32(TRACE_STRIDE - TRACE_SLOTS))
                            if const_expr(trace):
                                # no running tile-time sum here (``tacc``, slot 7 stays 0): its four
                                # SGPRs, live across the K loop, spilled an SGPR into a VGPR lane and
                                # re-scheduled the K loop (an A ds_read pair serialised)
                                stamp(trace_buf, slot0)
                            acc = hp_tile(b_row, r0, r1, r2, r3, slot0, hook)
                            # K loop done. Thread 0: the next tile's slot if the hook could not read it
                            # (blocking, the plain loop's wait); then claim the ticket after next and read
                            # the tail, both landing during the first epilogue half.
                            ct = fx.Int32(0)
                            tl2 = fx.Int32(0)
                            gt2 = fx.Int32(0)
                            if tid == fx.Int32(0):
                                if rd(HB_OK) == fx.Int32(0):
                                    take(rd(HB_T))
                                ct = claim_t()
                                if const_expr(XS2):
                                    # the ready-queue tail too (not volatile: in flight with the two others)
                                    gt2 = xldm(fx.Int64(arg_sched) + fx.Int64(xdz(ct)), SCH_TAIL)
                                if const_expr(xcd_order):
                                    tl2 = fx.Int32(comm_ops.load_i32_global_system(xtail(rd(HX_Q))))
                                else:
                                    tl2 = fx.Int32(comm_ops.load_i32_global_system(tail_addr))
                            # every wave is past its last A read (``do_tile``'s barrier); HA_* = the next tile
                            wait_lds_barrier()
                            tk_n = rd(HA_T)
                            raw_n = hp_rows(rd(HA_TIB))
                            sar_n = hp_sa_rows(rd(HA_TIB))
                            ce = hp_epi_c(acc)
                            # between the epilogue halves. Thread 0: the slot of the ticket after next if the
                            # tail already covers it. All: the next tile's A k-step 0 and A scales -> registers.
                            rqv = fx.Int32(0)
                            ok = fx.Int32(1)
                            if tid == fx.Int32(0):
                                if const_expr(xcd_order):
                                    # ct = j * 8 + y; a resolved ticket becomes its work item
                                    if tl2 > ct // fx.Int32(8):
                                        rqv = fx.Int32(comm_ops.load_i32_global_system(
                                            xentry(ct % fx.Int32(8), ct // fx.Int32(8))))
                                    else:
                                        ok = fx.Int32(0)
                                else:
                                    q2 = ct // fx.Int32(n_tiles)
                                    if ct < total_work:
                                        if tl2 > q2:
                                            rqv = fx.Int32(comm_ops.load_i32_global_system(rq_addr + fx.Int64(q2) * fx.Int64(4)))
                                        else:
                                            ok = fx.Int32(0)
                            sv = hp_issue(raw_n, sar_n)
                            hp_store_quant(hm, trb, n_base, ce)
                            # after the epilogue's closing barrier (the C tile is read): the next tile's A
                            # k-step 0 and A scales to LDS; thread 0 keeps the ticket after next for the hook
                            hp_stage_store(sv)
                            if tid == fx.Int32(0):
                                if const_expr(XS2):
                                    # not resolved and the whole queue published: the ticket's list is (or
                                    # will be) exhausted, steal now (or learn it is the end), not after the
                                    # next K loop. Here, after the epilogue, few registers are live.
                                    if ok == fx.Int32(0):
                                        if gt2 >= total_work // fx.Int32(n_tiles):
                                            wr(HX_F, fx.Int32(1))
                                            rqv = xsteal(ct)
                                            ok = fx.Int32(1)
                                    wr(HB_T, (ok != fx.Int32(0)).select(rqv, ct))
                                    wr(HB_M, (rqv < total_work).select(rqv // fx.Int32(n_tiles), fx.Int32(0)))
                                elif const_expr(xcd_order):
                                    # resolved: HB_T = the work item, HB_M = its m-tile; else the raw ticket
                                    wr(HB_T, (ok != fx.Int32(0)).select(rqv, ct))
                                    wr(HB_M, rqv // fx.Int32(n_tiles))
                                else:
                                    wr(HB_T, ct)
                                    wr(HB_M, rqv)
                                wr(HB_OK, ok)
                            if const_expr(trace):
                                stamp(trace_buf, slot0 + fx.Int32(2))
                                stamp_mt(trace_buf, slot0 + fx.Int32(4))
                            raw_n = raw_n + [fx.Int32(0)] * (4 - n_rounds)
                            r0 = raw_n[0]
                            r1 = raw_n[1]
                            r2 = raw_n[2]
                            r3 = raw_n[3]
                            tk = tk_n
                            item = item + fx.Int32(1)
                        stamp(trace_buf, fx.Int32(2))
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
        kwargs.pop("hoist_prologue", None)
        kwargs.pop("kpipe", None)
        kwargs.pop("kunroll", None)
        kwargs.pop("xcd_order", None)
        kwargs.pop("xcd_steal", None)
        kwargs.pop("fin_off", None)
        kwargs.pop("sa_pack", None)
        kwargs.pop("a_dbuf", None)
        kwargs.pop("epi_remat", None)
    elif kwargs.get("ready_first") and not prep.sched_on:
        raise ValueError("ready_first needs Stage1SdmaPrepare(sched=True) (arm zeroes its counters)")
    elif kwargs.get("xcd_order") and not prep.xcd_on:
        raise ValueError("xcd_order needs Stage1SdmaPrepare(sched=True, xcd_order=True) (the XCD lists)")
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
