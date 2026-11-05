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
from flydsl.expr import const_expr, range_constexpr

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

from stage1_sdma import Arena, memrealtime, memtime, post_rest_chunks


#: ``trace``: int64 slots per workgroup, and work items recorded per workgroup (8 + 3 * items).
TRACE_STRIDE = 2048
TRACE_SLOTS = 5  # per item: gate start, gate passed, done (100 MHz); passed, done (s_memtime)
TRACE_ITEMS = (TRACE_STRIDE - 8) // TRACE_SLOTS


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

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    # fmt: off
    @flyc.kernel(name=f"stage1_sdma_gemm1_t{sort_block_m}x{tile_n}_rb{row_bytes}_r{expert_offset}_tr{int(trace)}",
                 known_block_size=[total_threads, 1, 1])
    def kernel(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_w: fx.Tensor, tile_row_base: fx.Tensor,
        tile_input_base: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor,
        out_scale: fx.Tensor, rowmap: fx.Tensor, nvm: fx.Int32, arg_dev_comm: fx.Int64,
        arg_win: fx.Int64, arg_sig_base: fx.Int64, arg_expected: fx.Int64, arg_win_handle: fx.Int64,
        arg_cnt: fx.Int64, arg_trace: fx.Int64,
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

        num_valid = nv_rsrc[fx.Int32(0)]
        total_work = ((num_valid + fx.Int32(sort_block_m - 1)) // fx.Int32(sort_block_m)) * fx.Int32(n_tiles)
        tid = fx.thread_idx.x
        bid = fx.block_idx.x

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

        stamp(trace_buf, fx.Int32(0))
        if const_expr(gate_npes):
            import mori.cco.device.flydsl as cco

            sig_base = ptr_buf_tensor(arg_sig_base, fx.Int32)
            expected = ptr_buf_tensor(arg_expected, fx.Int32)

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
                mfma_amajor, async_a_copy, trb_rsrc, tib_rsrc, True, False,
            )
            if const_expr(gate_npes):
                stamp(trace_buf, slot0 + fx.Int32(2))
                stamp_mt(trace_buf, slot0 + fx.Int32(4))

        if const_expr(gate_npes):
            if bid == fx.Int32(0):
                post_rest_chunks(
                    cco.DevComm(arg_dev_comm).sdma(), arg_win_handle, tid // fx.Int32(64),
                    tid % fx.Int32(64), ptr_buf_tensor(arg_cnt, fx.Int32),
                    rank=gate_rank, arena=gate_arena, cco=cco,
                )
                stamp(trace_buf, fx.Int32(2))
                if const_expr(trace):
                    # trace only: when the last chunk from each peer has landed (WG0 is idle
                    # here): ``trace[8 + peer]``
                    if (tid < fx.Int32(gate_npes)) & (tid != fx.Int32(gate_rank)):
                        cco.DevComm(arg_dev_comm).sdma().wait_signal(tid, 0, fx.Int64(expected[tid]))
                        stamp_at(trace_buf, fx.Int32(8) + tid)
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
                # The only cache invalidation of the kernel. A tile does none after it has seen
                # its chunks land (the SDMA signal is a relaxed load, and none was needed):
                # SDMA writes memory without touching this GPU's L2, so a stale line would be
                # one that this L2 already holds. Lines from earlier calls are gone by now;
                # and a line of this call is only read after its chunk has landed, because a
                # chunk starts on a 128 B line (``Arena.check``), so no line is pulled in early
                # by a neighbour's read. Measured: an acquire per tile, of any scope, cost
                # ~90 us of 0.96 ms (it invalidates L2, with the weights in it).
                comm_ops.fence_system_acquire()
                stamp(trace_buf, fx.Int32(1))
                for flat in range(bid - fx.Int32(1), total_work, fx.Int32(num_cu - 1)):
                    run_tile(flat)
                stamp(trace_buf, fx.Int32(2))
                if const_expr(trace):
                    if tid == fx.Int32(0):
                        n_items = (total_work - (bid - fx.Int32(1)) + fx.Int32(num_cu - 2)) // fx.Int32(num_cu - 1)
                        trace_buf[bid * fx.Int32(TRACE_STRIDE) + fx.Int32(3)] = fx.Int64(n_items)
        else:
            for flat in range(bid, total_work, fx.Int32(num_cu)):
                run_tile(flat)

    # fmt: off
    @flyc.jit
    def launch(out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_w: fx.Tensor,
               tile_row_base: fx.Tensor, tile_input_base: fx.Tensor, expert_ids: fx.Tensor,
               num_valid_ids: fx.Tensor, out_scale: fx.Tensor, rowmap: fx.Tensor, nvm: fx.Int32,
               arg_dev_comm: fx.Int64, arg_win: fx.Int64, arg_sig_base: fx.Int64,
               arg_expected: fx.Int64, arg_win_handle: fx.Int64, arg_cnt: fx.Int64,
               arg_trace: fx.Int64, stream: fx.Stream):
    # fmt: on
        kernel(out, x, w, scale_w, tile_row_base, tile_input_base, expert_ids, num_valid_ids,
               out_scale, rowmap, nvm, arg_dev_comm, arg_win, arg_sig_base, arg_expected,
               arg_win_handle, arg_cnt, arg_trace,
               value_attrs={
                   "rocdl.waves_per_eu": waves_per_eu_hint,
                   "rocdl.flat_work_group_size": f"{total_threads},{total_threads}",
               }).launch(grid=(num_cu, 1, 1), block=(total_threads, 1, 1), stream=stream)

    return launch


def run_gemm1_sdma(out, window, w, scale_w, tile_row_base, tile_input_base, expert_ids,
                   num_valid, out_scale, rowmap, nvm, stream, prep=None, trace=None, **kwargs):
    """``window`` is ``[S | R]`` as fp8 ``[2 * n_rows, row_bytes]``; the others are the
    tensors of aiter's stage1 (``MegaMoEV2._s1_*`` and its dispatch op). With ``prep`` (a
    ``Stage1SdmaPrepare`` that has ``arm``-ed this call) the kernel waits for the data itself."""
    gate = {}
    dev_comm = win = sig_base = expected = win_handle = cnt = 0
    if prep is not None:
        gate = dict(gate_arena=prep.arena, gate_rank=prep.rank)
        dev_comm, win, win_handle = prep.dev_comm.ptr, prep.mem.ptr, prep.win.handle
        sig_base, expected, cnt = prep.sig_base.data_ptr(), prep.expected.data_ptr(), prep.cnt.data_ptr()
    launch = compile_gemm1_sdma(
        row_bytes=int(window.shape[1]), source_rows=int(window.shape[0]), trace=trace is not None,
        **gate, **kwargs,
    )
    _run_compiled(
        launch, out, window, w, scale_w, tile_row_base, tile_input_base, expert_ids, num_valid,
        out_scale, rowmap, fx.Int32(nvm), fx.Int64(dev_comm), fx.Int64(win), fx.Int64(sig_base),
        fx.Int64(expected), fx.Int64(win_handle), fx.Int64(cnt),
        fx.Int64(trace.data_ptr() if trace is not None else 0), stream,
    )
