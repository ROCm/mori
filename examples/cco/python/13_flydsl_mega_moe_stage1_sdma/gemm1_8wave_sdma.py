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
"""The fused SDMA Stage1 kernel: MegaMoEV2's dispatch on the copy engines, GEMM1 on an
8-wave 256x256 A8W4 main loop, in one persistent launch.

Runs after aiter's compact prepare (routing plan, ``TILE_INPUT_BASE``, ``expert_tile_end``)
and after ``pack.build_pack``, which put this rank's routed rows into a local send slab,
grouped by (destination rank, segment). Everything the copy engine touches lives in ONE cco
window, at the same offsets on every rank (compile-time constants here)::

    [ slab x | slab e8m0 | landing x | landing e8m0 | spare | shuffled e8m0 | flags ]

``slab``    rows this rank sends, at ``PAIR_BASE[seg] + row``: fp8 ``[rows, K]`` and raw
            e8m0 ``[rows, K/32]``; a segment's rows are contiguous, one put each.
``landing`` rows this rank computes, at the production payload row numbers (``TASK_ROW_BASE``
            / ``GROUP_TASK_BASE``), so an SBM sort tile's rows are contiguous from
            ``TILE_INPUT_BASE[tile]`` exactly as in aiter's route-major payload. This rank's
            own rows are written there by the pack kernel; the peers' rows arrive by SDMA.
``spare``   4 KB scratch: source and destination of the 4-byte put of an empty segment.
``shuffled e8m0`` (``scale_mode="shuffler"``) the A scales in aiter's ``shuffle_scale`` layout
            at the sorted row, i.e. the contiguous GEMM1's scale operand.
``flags``   (``scale_mode="shuffler"``) one i64 per shuffler block at ``8 * sid``.

Roles, by workgroup:

* block 0 -- ``sdma_comm_role``: wave ``p`` (p != rank), lane 0, ships this rank's rows for
  peer ``p`` with one data put and one e8m0 put per segment, in the order the GEMM consumes
  them: the fanout-group segment first, then experts 0..epr-1. Every e8m0 put carries the
  remote signal; an empty segment still sends two 4-byte puts within the spare area, so
  peer ``p``'s signal count from this rank is "segments done". Then ``quiet``: every put
  has read the slab before the next call's pack rewrites it.
* blocks 1..``shufflers`` (``scale_mode="shuffler"``) -- ``shuffler_role``: per expert, in
  order, wait until every peer's signal reached the expert's target, then this block's
  256-row chunks of the expert (round robin over the shuffler blocks): load the raw e8m0 rows,
  shuffle them in LDS, copy the chunk -- a contiguous range of the ``shuffle_scale`` layout --
  to the shuffled e8m0 area; release, raise this block's flag to the expert's target.
* the other blocks -- GEMM1 work items (M-tile, N-block) off an atomic counter, in expert
  order. A rows from the landing buffer through ``TILE_INPUT_BASE``. An item of expert ``e``
  waits until its data is there: the peers' signal counts (``ldsw``) or the shuffler flags
  (``shuffler``) reach ``sig_base - (epr + 1) + e + 2`` (``sig_base`` is advanced by the host
  by ``epr + 1`` per call; the group segment and experts 0..e are ``e + 2`` segments).
  Epilogue: MegaMoEV2's SwiGLU, fp8 1x32 rows and e8m0 to the Stage1 output in sorted order.

Scale modes. The indexed GEMM pays twice over the contiguous one, measured at bs 8192 (v4_pro,
8x MI355X, kernel alone, no puts): staging the A scales from the raw landing rows into LDS
(``ScaleLdsWordS2R``) per work item ~0.16 ms, and the per-item spin on 7 peer signals ~0.09 ms.

* ``"ldsw"``: per work item, wait for the 7 peer signals, then stage the M-tile's scales into
  LDS in the shuffled word layout; the global loads are issued before the prologue's A/B loads
  and written to LDS after them, so the two latencies overlap.
* ``"shuffler"``: the scales are shuffled once per M-tile (not once per N-block) by the
  shuffler blocks, and the GEMM reads them like the contiguous kernel
  (``ScalePreshuffledS2R``); an item waits only on the shuffler flags. Fused kernel 1.83 ms vs
  a 1.79 ms lower bound (no waits, pre-shuffled scales). It delays the first M-tile by the
  shuffle of expert 0, which costs more than it saves at small batches (see the op).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Float4E2M1FN, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import (
    G2SLoader,
    S2RLoader,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    wait_barrier,
)
from aiter.ops.flydsl.kernels.kernels_common import set_slp_vectorize_once_used
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import (
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
)
from aiter.ops.flydsl.kernels.tensor_shim import ptr_buf_tensor
from aiter.ops.flydsl.kernels.mega_moe.dispatch import (
    DispatchSlot,
    _load_fanout_pair,
    _wave_inclusive_scan_i32,
)

__all__ = ["compile_gemm1_8wave_sdma", "SCALE_MODES"]

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
LDS_LIMIT_BYTES = 160 * 1024
SCALE_MODES = ("shuffler", "ldsw")


class ScalePreshuffledS2R:
    """Coalesced reader for ``shuffle_scale``-packed E8M0 -- no LDS staging."""

    def __init__(self, scale_arg, rows, K, n_tiles, lane):
        assert n_tiles % 2 == 0, "the shuffled scale layout pairs tiles two at a time"
        self.n_pairs = n_tiles // 2
        self.k1_stride = K // 256  # i32 groups of 64 per 32-row super-row
        self.lane = lane
        t_i8 = fx.rocdl.make_buffer_tensor(
            scale_arg,
            max_size=False,
            num_records_bytes=fx.Int64(rows) * fx.Int64(K // 32),
        )
        i32_ptr = fx.PointerType.get(
            elem_ty=fx.Int32.ir_type,
            address_space=rocdl.TargetAddressSpace.BufferDesc,
            alignment=4,
        )
        iter_i32 = fx.recast_iter(i32_ptr, fx.get_iter(t_i8))
        n_i32 = fx.Int32(rows) * fx.Int32(K // 128)
        self.g_div = fx.logical_divide(
            fx.Tensor(fx.make_view(iter_i32, fx.make_layout(n_i32, 1))),
            fx.make_layout(1, 1),
        )
        self.atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        # Two register sets: the next K-pair is prefetched while the current one
        # still feeds MFMAs.
        self.regs = [
            [
                fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
                for _ in range_constexpr(self.n_pairs)
            ]
            for _ in range_constexpr(2)
        ]

    def read(self, row_base16, k):
        """``n_tiles`` scale operands for K-step ``k``; tiles of a pair share one word."""
        k1 = k // 2  # compile-time; k % 2 is the k_pack the opsel encodes
        regs = self.regs[k1 % 2]
        words = []
        for p in range_constexpr(self.n_pairs):
            # row_base16 is even (wave offsets are multiples of 32 rows), so the
            # 32-row super-row is row_base16 // 2 + p and the tile parity picks the byte.
            n1 = row_base16 // 2 + p
            base = rocdl.readfirstlane(
                fx.Int32.ir_type, (n1 * self.k1_stride + k1) * 64
            )
            fx.copy(
                self.atom,
                fx.slice(self.g_div, (None, fx.Int32(base) + self.lane)),
                regs[p],
            )
            w = fx.Int32(regs[p].load()[0])
            words += [w, w]
        return words


class ScaleLdsWordS2R:
    """Route-major payload, A scales staged in LDS already in the ``shuffle_scale`` word
    layout: per work item all 512 threads load the M-tile's raw ``[row, K/32]`` e8m0
    dwords (rows from TILE_INPUT_BASE, one SGPR base per sort tile) and scatter their
    bytes to the shuffled positions of a 256-row local copy. The main loop then reads
    one ds_read_b32 per MFMA scale word -- the contiguous version's addressing, from LDS.

    ``load_wide`` / ``write_wide`` are split so the global loads can be issued early and
    written to LDS later; the shuffler blocks use the same two halves."""

    def __init__(self, n_tiles, lane, K):
        self.n_pairs = n_tiles // 2
        self.lane = lane
        self.SC = K // 32
        self.W = self.SC // 4  # scale dwords per row
        self.k1_stride = K // 256  # 64-dword groups per 32-row super-row

    def load_wide(self, raw_i64, tib_bases, SBM, n_rows, tid):
        """One unit = rows (r, r + 16) of a 32-row group x dword pair (2j, 2j + 1): two 8-byte
        loads, then the four LDS words those 16 bytes fill in the shuffled layout -- word k
        (at byte ``64k``) is ``[a_k, c_k, b_k, d_k]`` for a, b = row r's dwords 2j, 2j + 1 and
        c, d = row r + 16's. ``tib_bases``: the payload row of each sort tile of the M-tile
        (uniform)."""
        W, SC = self.W, self.SC
        J = W // 2
        units = 8 * 16 * J
        assert W % 2 == 0 and units % 512 == 0
        rsrc = create_buffer_resource_from_addr(
            fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(raw_i64).ir_value()))
        )
        # glc: a K/32-byte scale row can share a cache line with the next sort tile's
        # first row, which may not have landed yet -- never let L1 keep it.
        cm = 1

        def src_row(r):
            base = tib_bases[0]
            for i in range_constexpr(1, len(tib_bases)):
                base = (r >= fx.Int32(i * SBM)).select(tib_bases[i], base)
            return base + r % fx.Int32(SBM)

        loads, offs = [], []
        for step in range_constexpr(units // 512):
            u = fx.Int32(step * 512) + tid
            j = u % fx.Int32(J)
            rr = u // fx.Int32(J)
            r0 = (rr >> fx.Int32(4)) * fx.Int32(32) + (rr & fx.Int32(15))
            pair = []
            for half in range_constexpr(2):
                r = r0 + fx.Int32(16 * half)
                ok = r < n_rows
                v = fx.Vector(
                    buffer_load(
                        rsrc,
                        ok.select(src_row(r), fx.Int32(0)) * fx.Int32(W)
                        + j * fx.Int32(2),
                        vec_width=2,
                        dtype=T.i32,
                        cache_modifier=cm,
                    )
                )
                pair.append(
                    [
                        ok.select(fx.Int32(v[0]), fx.Int32(0)),
                        ok.select(fx.Int32(v[1]), fx.Int32(0)),
                    ]
                )
            loads.append(pair)
            offs.append(
                (
                    (rr >> fx.Int32(4)) * fx.Int32(32 * SC)
                    + j * fx.Int32(256)
                    + (rr & fx.Int32(15)) * fx.Int32(4)
                )
                // fx.Int32(4)
            )
        return loads, offs

    def write_wide(self, lds_base, staged):
        loads, offs = staged
        ptr = lds_typed_ptr(lds_base, T.i32, align=4)
        m8 = fx.Int32(0xFF)
        for step in range_constexpr(len(loads)):
            (a, b), (c, d) = loads[step]
            for k in range_constexpr(4):
                sh = fx.Int32(8 * k)
                w = (
                    (a.shrui(sh) & m8)
                    | ((c.shrui(sh) & m8) << fx.Int32(8))
                    | ((b.shrui(sh) & m8) << fx.Int32(16))
                    | ((d.shrui(sh) & m8) << fx.Int32(24))
                )
                ptr[offs[step] + fx.Int32(16 * k)] = w

    def read(self, row_base16, k, lds_base):
        """``row_base16`` is the wave's 16-row tile index *within the M-tile*."""
        k1 = k // 2
        ptr = lds_typed_ptr(lds_base, T.i32, align=4)
        words = []
        for p in range_constexpr(self.n_pairs):
            n1 = row_base16 // 2 + p
            base = fx.Int32(rocdl.readfirstlane(T.i32, (n1 * self.k1_stride + k1) * 64))
            w = fx.Int32(ptr[base + self.lane])
            words += [w, w]
        return words


class S2RF4G:
    """fp4 B operands from the ``shuffle_weight_a16w4`` layout staged 1:1 in LDS.

    LDS holds 1 KB per 16-row group, ``[k_lane 4][n 16][16 B]``; lane (g, n) of tile
    ``i`` reads its 16 bytes (32 contiguous K, the fp4 MFMA operand) conflict-free.
    """

    def __init__(self, wave_idx, n_tiles, lane):
        self.lane_id = lane
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles

    def load(self, lds_src):
        frag = []
        for i in range_constexpr(self.n_tiles):
            group = self.wave_idx * self.n_tiles + i
            offset = (
                group * 1024 + (self.lane_id // 16) * 256 + (self.lane_id % 16) * 16
            )
            ptr_off = fx.add_offset(lds_src.ptr, fx.make_int_tuple(offset))
            view = fx.make_view(
                fx.recast_iter(fx.Uint8, ptr_off), fx.make_layout(16, 1)
            )
            frag.append(view.load().bitcast(fx.Int32))
        return frag


class MxMfma:
    """n_tiles_a x n_tiles_b scaled 16x16x128 fp8 x fp4 MFMAs; a scale word's byte is
    ``k_pack * 2 + tile parity``."""

    def __init__(self, n_tiles_a, n_tiles_b):
        self.atoms = {
            (osa, osb): fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(
                    16, 16, 128, fx.Float8E4M3FN, Float4E2M1FN, opsel_a=osa, opsel_b=osb
                )
            )
            for osa in range(4)
            for osb in range(4)
        }
        self.zero_value = fx.full(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _frag(self, value, n, dtype):
        frag = fx.make_rmem_tensor(n, dtype)
        frag.store(Vec(value))
        return frag

    def call(self, a, b, c, sa, sb, *, k_pack, set_prio=True):
        a_frags = [self._frag(v, 8, fx.Int32) for v in a]
        b_frags = [self._frag(v, 4, fx.Int32) for v in b]
        c_frags = [self._frag(v, 4, fx.Float32) for v in c]
        if const_expr(set_prio):
            rocdl.s_setprio(1)
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                cf = c_frags[self.idx(i, j)]
                fx.gemm(
                    self.atoms[(k_pack * 2 + i % 2, k_pack * 2 + j % 2)],
                    cf,
                    a_frags[i],
                    b_frags[j],
                    cf,
                    scale_a=sa[i],
                    scale_b=sb[j],
                )
        if const_expr(set_prio):
            rocdl.s_setprio(0)
            rocdl.s_barrier()
        return [cf.load().ir_value() for cf in c_frags]


# --- 256-row tiles of the expert segments --------------------------------------------


def tile_table(tile_end, lane_id, *, SBM, experts):
    """Per-lane 256-row tile counts of the local experts (lane e: expert e), from
    Prepare's cumulative SBM-tile ends. Returns the per-lane values and the tile total.
    """
    ends = global_typed_ptr(tile_end, T.i32)
    ok = lane_id < fx.Int32(experts)
    li = ok.select(lane_id, fx.Int32(0))
    end_l = ends[li]
    prev_l = (ok & (lane_id > fx.Int32(0))).select(
        ends[(li > fx.Int32(0)).select(li - fx.Int32(1), fx.Int32(0))], fx.Int32(0)
    )
    seg_l = ok.select((end_l - prev_l) * fx.Int32(SBM), fx.Int32(0))
    tiles_l = (seg_l + fx.Int32(255)) >> fx.Int32(8)
    cum_l = _wave_inclusive_scan_i32(tiles_l, lane_id)
    total = fx.Int32(
        rocdl.readfirstlane(
            T.i32, fx.Int32(rocdl.ds_bpermute(T.i32, fx.Int32(63 * 4), cum_l))
        )
    )
    return (prev_l, seg_l, tiles_l, cum_l), total


def tile_table_to_lds(values, tab_lds, lane_id):
    """Park the per-lane table in LDS (every wave writes the same values), so the
    persistent loop does not keep it in VGPRs."""
    p = lds_typed_ptr(tab_lds, T.i32)
    for i, v in enumerate(values):
        p[fx.Int32(i * 64) + lane_id] = v


def tile_find(tab_lds, j, lane_id, *, SBM, experts):
    """(row_base, expert, valid_rows) of 256-row tile ``j`` -- LDS and wave ops only."""
    p = lds_typed_ptr(tab_lds, T.i32)
    prev_l, seg_l, tiles_l, cum_l = [
        fx.Int32(p[fx.Int32(i * 64) + lane_id]) for i in range(4)
    ]
    ok = lane_id < fx.Int32(experts)
    # The cumulative ends rise with the lane, so the experts that end at or before ``j``
    # are a lane prefix: their count is the expert of ``j``. One ballot and a popcount
    # instead of a butterfly max (six dependent LDS permutes per work item), and
    # ``v_readlane`` instead of a permute to read the expert's row of the table.
    before = fx.Int32(fx.math.ctpop(fx.Int64(rocdl.ballot(T.i64, ok & (cum_l <= j)))))
    e = fx.Int32(rocdl.readfirstlane(T.i32, before))
    e = (e < fx.Int32(experts)).select(e, fx.Int32(experts - 1))

    def from_lane(v):
        return fx.Int32(rocdl.readlane(T.i32, v, e))

    t_in = j - from_lane(cum_l - tiles_l)
    row_base = from_lane(prev_l * fx.Int32(SBM)) + t_in * fx.Int32(256)
    left = from_lane(seg_l) - t_in * fx.Int32(256)
    valid = (left > fx.Int32(256)).select(
        fx.Int32(256), (left > fx.Int32(0)).select(left, fx.Int32(0))
    )
    return row_base, e, valid


# --- Small helpers -------------------------------------------------------------------


def opaque_i32(v):
    """``v`` through an identity the compiler cannot see through.

    In the persistent loop, lane-derived addresses would otherwise be hoisted out of
    it as loop invariants and spilled (the main loop leaves no VGPRs for them), and a
    scratch reload counts in vmcnt like the epilogue's stores.
    """
    return fx.Int32(
        _llvm.inline_asm(
            res=T.i32,
            operands_=[fx.Int32(v).ir_value()],
            asm_string="v_mov_b32 $0, $1",
            constraints="=v,v",
            has_side_effects=True,
        )
    )


def lds_barrier():
    """Workgroup barrier for LDS only: unlike ``fx.barrier`` it never waits on vmcnt."""
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
        constraints="",
        has_side_effects=True,
    )


# --- The kernel ------------------------------------------------------------------------


def compile_gemm1_8wave_sdma(
    *,
    K,
    N,
    experts,
    SBM,
    grid_ctas,
    out_rows,
    scale_bytes,
    rank,
    npes,
    off_slab_x,
    off_slab_sc,
    off_land_x,
    off_land_sc,
    off_spare,
    scale_mode="shuffler",
    shufflers=4,
    swiglu_limit=0.0,
    waves_per_eu=2,
):
    """The fused SDMA Stage1 (see the module docstring).

    ``K`` model_dim; ``N`` = 2 * inter_dim (gate/up weight rows interleaved per 16);
    ``experts`` local experts (epr); ``SBM`` the active Stage1 sort_block_m; ``grid_ctas``
    one workgroup per CU; ``out_rows`` / ``scale_bytes`` the Stage1 output rows and the
    size of its shuffled e8m0 (out-of-range stores are dropped); ``off_*`` the cco window
    layout (bytes); ``scale_mode`` "shuffler" (``shufflers`` blocks) or "ldsw".

    Epilogue: MegaMoEV2's GEMM1 SwiGLU -- ``silu(min(g, limit)) * clamp(u, +-limit)`` --
    fp8 1x32 with the Stage1 e8m0 rule, rows to ``s1_out`` [rows, N/2] and scales to the
    sorted shuffled layout ``s1_out_scale``.
    """
    assert K % 256 == 0 and N % BLOCK_N == 0, f"K={K} needs % 256, N={N} % {BLOCK_N}"
    assert 256 % SBM == 0, f"SBM={SBM} must divide 256"
    assert scale_mode in SCALE_MODES, scale_mode
    INTER = N // 2
    SC_COLS32 = (
        (INTER // 32 + 7) // 8 * 8 * 32
    )  # bytes per 32-row super-row of out_scale
    K_ITERS = K // BLOCK_K
    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    B_ROW_BYTES = BLOCK_K // 2  # fp4: one K step of a B row is 64 bytes
    N_LDS_STEPS_B = LDS_BLOCK_N * B_ROW_BYTES // (8 * 1024)  # 16 B per lane, 8 waves
    assert N_LDS_STEPS_B == 1
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * B_ROW_BYTES
    A_GRP_ROWS = N_TILES_A * 16  # rows one wave_m owns inside one LDS half
    B_GRP_ROWS = N_TILES_B * 16
    B_K_STEP = 1024  # one 16-row group x 128 K of the a16w4 layout

    ldsw = scale_mode == "ldsw"
    N_SHUF = 0 if ldsw else int(shufflers)  # shuffler blocks 1..N_SHUF (LDS-staged)
    assert ldsw or N_SHUF > 0, "the shuffler mode needs at least one shuffler block"
    assert grid_ctas > 1 + N_SHUF
    # One M-tile's raw A scales, 256 rows x K/32 bytes (ldsw: per work item; shuffler: per chunk).
    SC_LDS_BYTES = 256 * (K // 32)
    E_STAGE_BYTES = max(128 * 256, SC_LDS_BYTES)
    E_TAB_BYTES = 4 * 64 * 4 + 16
    assert (
        4 * (a_lds_size + b_lds_size) + E_STAGE_BYTES + E_TAB_BYTES <= LDS_LIMIT_BYTES
    )

    kname = (
        f"mega_moe_s1_sdma_8w_n{N}_k{K}_e{experts}_s{SBM}_g{grid_ctas}_l{swiglu_limit}"
        f"_r{rank}p{npes}_o{out_rows}b{scale_bytes}_{scale_mode}{N_SHUF}_w{waves_per_eu}"
        f"_win{off_slab_x}_{off_slab_sc}_{off_land_x}_{off_land_sc}_{off_spare}"
    )

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        # Epilogue staging, never aliased with the main loop's buffers; also the tile's
        # raw A scales during the main loop (ldsw) or a shuffler block's chunk, never live
        # at the same time.
        E_stage: fx.Array[fx.Float8E4M3FN, E_STAGE_BYTES, 16]
        E_tab: fx.Array[
            fx.Float8E4M3FN, E_TAB_BYTES, 16
        ]  # the tile table (+ the dyn work-item word)

    @flyc.kernel(name=kname, known_block_size=[512, 1, 1])
    def kernel(
        A: fx.Tensor,
        B_T: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        tile_end: fx.Int64,
        land_sc: fx.Int64,
        tib: fx.Int64,
        parity_p: fx.Int64,
        s1_out: fx.Int64,
        plan_ready: fx.Int64,
        expected_p: fx.Int64,
        shuf_flags: fx.Int64,
        s1_out_scale: fx.Int64,
        cursor: fx.Int64,
        shuf_scale: fx.Int64,
        addr_disp: fx.Int64,
        win: fx.Int64,
        dev_comm: fx.Int64,
        sig_base: fx.Int64,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type
        n_blocks = fx.Int32(N // BLOCK_N)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        def lds_addr(arr):
            return fx.Int32(fx.ptrtoint(arr.ptr))

        def lane_ids():
            # Recomputed per work item and per epilogue (see opaque_i32).
            tid = opaque_i32(fx.thread_idx.x)
            wave_id = tid // fx.Int32(64)
            return (
                tid,
                tid % fx.Int32(64),
                wave_id,
                wave_id // fx.Int32(4),
                wave_id % fx.Int32(4),
            )

        def signal_slots():
            """(address of peer 0's queue-0 signal count, stride to the next peer's), wave
            uniform: the counts are one array indexed by the sender, so two lookups give
            all of them, once per workgroup."""
            import mori.cco.device.flydsl as cco

            sd = cco.DevComm(dev_comm).sdma()
            s0 = fx.Int64(
                rocdl.readfirstlane(T.i64, fx.Int64(sd.signal_slot(0, 0)).ir_value())
            )
            s1 = fx.Int64(
                rocdl.readfirstlane(T.i64, fx.Int64(sd.signal_slot(1, 0)).ir_value())
            )
            return s0, s1 - s0

        def wait_signals(slot0, slot_stride, need):
            """Every peer's queue-0 signal count reaches ``need``: all of them in one round
            trip, then a spin only on a short one."""
            addrs = [
                slot0 + slot_stride * fx.Int64(src)
                for src in range(npes)
                if src != rank
            ]
            sigs = [fx.Int64(comm_ops.load_i64_acquire(a)) for a in addrs]
            short = sigs[0] < need
            for v in sigs[1:]:
                short = short | (v < need)
            if short:
                for a in addrs:
                    comm_ops.spin_until_ge_i64(a, need)

        def process_tile(block_m, block_n, tab_lds, sig_slot0, sig_stride):
            _, lane_id, wave_id, wave_m, wave_n = lane_ids()
            a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
            a_next0, a_next1 = lds.A_lds_next_0, lds.A_lds_next_1
            b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
            b_next0, b_next1 = lds.B_lds_next_0, lds.B_lds_next_1
            row_base, expert, n_valid = tile_find(
                tab_lds, block_m, lane_id, SBM=SBM, experts=experts
            )
            b_row_block = expert * n_blocks + block_n
            tid0 = fx.Int32(fx.thread_idx.x) == fx.Int32(0)
            # The expert's rows have landed once every peer has signalled the group segment
            # and experts 0..expert of this call.
            need = (
                fx.Int64(global_typed_ptr(sig_base, T.i64, align=8)[0])
                - fx.Int64(experts + 1)
                + fx.Int64(expert + fx.Int32(2))
            )
            if const_expr(not ldsw):
                # The shuffler blocks waited for the expert's data and wrote its scales into
                # A_scale; ``shuf_flags`` holds their per-expert flags. All flags in one round
                # trip; spin only if one is short.
                if tid0:
                    flag_addrs = [
                        fx.Int64(shuf_flags) + fx.Int64(8 * si) for si in range(N_SHUF)
                    ]
                    flags = [fx.Int64(comm_ops.load_i64_acquire(a)) for a in flag_addrs]
                    short = flags[0] < need
                    for v in flags[1:]:
                        short = short | (v < need)
                    if short:
                        for a in flag_addrs:
                            comm_ops.spin_until_ge_i64(a, need)
                    # A short M-tile also reads the next expert's scale rows, which this
                    # CU's L1 may hold from before the shuffler wrote them: drop L1 (only).
                    _llvm.inline_asm(
                        res=None,
                        operands_=[],
                        asm_string="buffer_inv sc0",
                        constraints="",
                        has_side_effects=True,
                    )
                fx.barrier()
            else:
                if tid0:
                    wait_signals(sig_slot0, sig_stride, need)
                fx.barrier()
            tib_p = global_typed_ptr(tib, T.i32)

            B0_gl_offset = (b_row_block * BLOCK_N) * (K // 2)
            B1_gl_offset = (b_row_block * BLOCK_N + LDS_BLOCK_N) * (K // 2)
            a_div = fx.logical_divide(
                make_fp8_buffer_tensor(A, F8_IR_t), fx.make_layout(1, 1)
            )
            b_div = fx.logical_divide(
                make_fp8_buffer_tensor(B_T, F8_IR_t), fx.make_layout(1, 1)
            )
            gl_off_a = compute_global_swizzle(
                lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False
            )
            # a16w4 layout: wave w copies 16-row group w's contiguous 1 KB of a K step.
            gl_off_b = [wave_id * (16 * (K // 2)) + lane_id * 16]

            mfma = MxMfma(N_TILES_A, N_TILES_B)
            # Route-major payload: an SBM tile's rows are contiguous, so a lane's offset is
            # the contiguous one (gl_off_a) and the gather is a uniform per (half, step)
            # base in soffset -- no extra VGPRs.
            a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            a_soff = []
            for h in range_constexpr(2):
                per_step = []
                for step in range_constexpr(N_LDS_STEPS_A):
                    g0 = row_base + fx.Int32(h * LDS_BLOCK_M + step * 64)
                    st = g0 // fx.Int32(SBM)
                    rel = row_base + fx.Int32(h * LDS_BLOCK_M) - st * fx.Int32(SBM)
                    per_step.append(
                        fx.Int32(
                            rocdl.readfirstlane(
                                T.i32, (fx.Int32(tib_p[st]) + rel) * fx.Int32(K)
                            )
                        )
                    )
                a_soff.append(per_step)

            def load_a(h, lds_dst, k_byte):
                for step in range_constexpr(N_LDS_STEPS_A):
                    a_g2s.load_one(lds_dst, a_soff[h][step] + fx.Int32(k_byte), step)

            b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
            a_s2r = S2RLoader(wave_m, N_TILES_A)
            a_s2r.lane_id = lane_id
            b_s2r = S2RF4G(wave_n, N_TILES_B, lane_id)
            if const_expr(ldsw):
                a_sc = ScaleLdsWordS2R(N_TILES_A, lane_id, K)
            else:
                a_sc = ScalePreshuffledS2R(A_scale, c_m, K, N_TILES_A, lane_id)
            b_sc = ScalePreshuffledS2R(B_scale, experts * N, K, N_TILES_B, lane_id)
            # 16-row tile index of each wave's first row, per LDS half.
            a_base16 = [
                (row_base + h * LDS_BLOCK_M + wave_m * A_GRP_ROWS) // 16
                for h in range_constexpr(2)
            ]
            b_base16 = [
                (b_row_block * BLOCK_N + h * LDS_BLOCK_N + wave_n * B_GRP_ROWS) // 16
                for h in range_constexpr(2)
            ]
            sc_lds = lds_addr(lds.E_stage)
            if const_expr(ldsw):
                # The M-tile's scales into LDS in the shuffled word layout; the waves then
                # address it with tile-local 16-row indices.
                tib_bases = [
                    fx.Int32(
                        rocdl.readfirstlane(
                            T.i32,
                            fx.Int32(tib_p[row_base // fx.Int32(SBM) + fx.Int32(i)]),
                        )
                    )
                    for i in range_constexpr(256 // SBM)
                ]
                # Loads only; written to LDS after the prologue's A/B loads are issued.
                sc_staged = a_sc.load_wide(
                    land_sc, tib_bases, SBM, n_valid, wave_id * fx.Int32(64) + lane_id
                )
                a_base16 = [
                    a_base16[h] - row_base // fx.Int32(16) for h in range_constexpr(2)
                ]
            # The wait_barrier counts below assume 6 scale loads per K-pair.
            assert 2 * (N_TILES_A // 2) + 2 * (N_TILES_B // 2) == 6

            def scale_prefetch(sc, k0):
                if const_expr(k0 >= K_ITERS or k0 // 2 in sc):
                    return sc
                words = {
                    w: (
                        (
                            a_sc.read(a_base16[int(w[1])], k0, sc_lds)
                            if ldsw
                            else a_sc.read(a_base16[int(w[1])], k0)
                        )
                        if w[0] == "a"
                        else b_sc.read(b_base16[int(w[1])], k0)
                    )
                    for w in ("a0", "a1", "b0", "b1")
                }
                return {**sc, k0 // 2: words}

            def scale_read(sc, k, which):
                return sc[k // 2][which]

            c00_frag = [mfma.zero_value] * N_ACCUMS
            c01_frag = [mfma.zero_value] * N_ACCUMS
            c10_frag = [mfma.zero_value] * N_ACCUMS
            c11_frag = [mfma.zero_value] * N_ACCUMS

            if const_expr(ldsw):
                b_g2s.load(b_cur0, B0_gl_offset)
                load_a(0, a_cur0, 0)
                b_g2s.load(b_cur1, B1_gl_offset)
                load_a(1, a_cur1, 0)
                a_sc.write_wide(sc_lds, sc_staged)
                lds_barrier()
                sc_pf = scale_prefetch({}, 0)
            else:
                sc_pf = scale_prefetch({}, 0)
                b_g2s.load(b_cur0, B0_gl_offset)
                load_a(0, a_cur0, 0)
                b_g2s.load(b_cur1, B1_gl_offset)
                load_a(1, a_cur1, 0)
            # The wave_m == 1 group runs one barrier behind the other from here on.
            if wave_m == 1:
                rocdl.s_barrier()
            wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)
            b_g2s.load(b_next0, B0_gl_offset + B_K_STEP)
            load_a(0, a_next0, BLOCK_K)
            b_g2s.load(b_next1, B1_gl_offset + B_K_STEP)
            wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

            for k in range_constexpr(K_ITERS - 2):
                if const_expr(k % 2 == 1):
                    sc_pf = scale_prefetch(sc_pf, k + 1)
                sa0 = scale_read(sc_pf, k, "a0")
                sb0 = scale_read(sc_pf, k, "b0")
                b0_frag = b_s2r.load(b_cur0)
                a0_frag = a_s2r.load(a_cur0)
                load_a(1, a_next1, (k + 1) * BLOCK_K)
                rocdl.s_barrier()

                c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)

                sb1 = scale_read(sc_pf, k, "b1")
                b1_frag = b_s2r.load(b_cur1)
                b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
                rocdl.s_barrier()

                c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)

                sa1 = scale_read(sc_pf, k, "a1")
                a1_frag = a_s2r.load(a_cur1)
                load_a(0, a_cur0, (k + 2) * BLOCK_K)
                # This barrier of the lagging group pairs with the other group's
                # end-of-step wait, after which that group reads step k + 1's
                # b_cur0 / a_cur0, loaded in step k - 1: drain everything but this
                # step's three loads and step k - 1's b_cur1.
                wait_barrier(2 * N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

                c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2)

                b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
                wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

                c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2)

                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1

            # Step K_ITERS - 2: no loads left to overlap but the last A1 tile.
            k = K_ITERS - 2
            sa0 = scale_read(sc_pf, k, "a0")
            sb0 = scale_read(sc_pf, k, "b0")
            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            # The lagging group loaded step K_ITERS - 1's tiles in step K_ITERS - 3:
            # drain them here, the barrier the other group's next reads pair with.
            wait_barrier(0)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)
            sb1 = scale_read(sc_pf, k, "b1")
            b1_frag = b_s2r.load(b_cur1)
            rocdl.s_barrier()
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)
            sa1 = scale_read(sc_pf, k, "a1")
            a1_frag = a_s2r.load(a_cur1)
            load_a(1, a_next1, (K_ITERS - 1) * BLOCK_K)
            rocdl.s_barrier()
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2)
            b0_frag = b_s2r.load(b_next0)
            rocdl.s_barrier()
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2)
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

            # Step K_ITERS - 1
            k = K_ITERS - 1
            sa0 = scale_read(sc_pf, k, "a0")
            sb0 = scale_read(sc_pf, k, "b0")
            a0_frag = a_s2r.load(a_cur0)
            wait_barrier(0)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)
            sb1 = scale_read(sc_pf, k, "b1")
            b1_frag = b_s2r.load(b_cur1)
            rocdl.s_barrier()
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)
            sa1 = scale_read(sc_pf, k, "a1")
            a1_frag = a_s2r.load(a_cur1)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c10_frag = mfma.call(
                a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2, set_prio=False
            )
            c11_frag = mfma.call(
                a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2, set_prio=False
            )
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            # Even the two groups' barrier counts again before the epilogue's barriers.
            if wave_m == 0:
                rocdl.s_barrier()

            epilogue_swiglu(
                (c00_frag, c01_frag),
                (c10_frag, c11_frag),
                row_base,
                n_valid,
                block_n,
                mfma.idx,
            )

        def act_mul(g, u):
            """MegaMoEV2 SwiGLU, as SiluQuantEpilogue._act_mul / _silu compute it."""

            def silu(x):
                emu = (x * fx.Float32(-1.4426950408889634)).exp2()
                return x * (fx.Float32(1.0) / (fx.Float32(1.0) + emu))

            if const_expr(swiglu_limit <= 0):
                return silu(g) * u
            limit = fx.Float32(swiglu_limit)
            return silu(-(-g).maximumf(-limit)) * fx.clampf(u, -limit, limit)

        def epilogue_swiglu(half0, half1, row_base, n_valid, block_n, mfma_idx):
            """SwiGLU in registers, then per 64-row quarter (one wave_m group's rows) as f32
            through LDS [64][128]; every lane reads 8 contiguous columns of a row, the 1x32
            |max| is 4 lanes (2 xor shuffles per 8 values, as the Stage1 vec8 store) and the
            8 fp8 bytes and the group's e8m0 go straight to global."""
            tid, lane_id, wave_id, wave_m, wave_n = lane_ids()
            stage_lds = lds_addr(lds.E_stage)
            f_p = lds_typed_ptr(stage_lds, T.f32)
            out_rsrc = create_buffer_resource_from_addr(
                fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(s1_out).ir_value())),
                num_records_bytes=out_rows * INTER,
            )
            osc_rsrc = create_buffer_resource_from_addr(
                fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(s1_out_scale).ir_value())),
                num_records_bytes=scale_bytes,
            )
            l16 = lane_id % fx.Int32(16)
            c64 = fx.Int32(64)
            # Read side: thread t takes row t // 16 (+32 per pass) and columns (t % 16) * 8.
            rd_row = tid // fx.Int32(16)
            rd_col = (tid % fx.Int32(16)) * fx.Int32(8)
            col_s = block_n * fx.Int32(4) + (tid % fx.Int32(16)) // fx.Int32(4)
            writer = (tid % fx.Int32(4)) == fx.Int32(0)
            d3 = col_s >> fx.Int32(3)
            d4 = (col_s >> fx.Int32(2)) & fx.Int32(1)
            d5 = col_s & fx.Int32(3)
            for h_m in range_constexpr(2):
                frags = half1 if h_m else half0
                ys = []
                for ti in range_constexpr(N_TILES_A):
                    for i in range_constexpr(4):
                        rq = (
                            ti * 16 + (lane_id // 16) * 4 + i
                        )  # row within the wave_m quarter
                        for h_n in range_constexpr(2):
                            g = Vec(frags[h_n][mfma_idx(ti, 0)])[i]
                            u = Vec(frags[h_n][mfma_idx(ti, 1)])[i]
                            ys.append(
                                (rq * 128 + h_n * 64 + wave_n * 16 + l16, act_mul(g, u))
                            )
                for q in range_constexpr(2):
                    lds_barrier()
                    if wave_m == fx.Int32(q):
                        for idx, y in ys:
                            f_p[idx] = y
                    lds_barrier()
                    for ps in range_constexpr(2):
                        row_q = rd_row + fx.Int32(ps * 32)
                        v = Vec(
                            lds_vec_load(
                                stage_lds,
                                (row_q * 128 + rd_col) * 4,
                                Vec.make_type(8, fx.Float32),
                                fx.Float32,
                                align=16,
                            )
                        )
                        vals = [fx.Float32(v[k]) for k in range_constexpr(8)]
                        m = vals[0].maximumf(fx.Float32(0.0) - vals[0])
                        for k in range_constexpr(1, 8):
                            m = m.maximumf(vals[k].maximumf(fx.Float32(0.0) - vals[k]))
                        for off in (1, 2):
                            m = m.maximumf(m.shuffle_xor(fx.Int32(off), c64))
                        max_rounded = (
                            m.bitcast(fx.Int32) + fx.Int32(0x400000)
                        ) & fx.Int32(0xFF800000)
                        e = (max_rounded >> fx.Int32(23)) - fx.Int32(8)
                        e8m0 = (e > fx.Int32(0)).select(e, fx.Int32(0))
                        qs = ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(
                            fx.Float32
                        )
                        words = []
                        for w in range_constexpr(2):
                            sc = [vals[w * 4 + k] * qs for k in range_constexpr(4)]
                            pk = rocdl.cvt_pk_fp8_f32(
                                T.i32, sc[0], sc[1], fx.Int32(0), 0
                            )
                            pk = rocdl.cvt_pk_fp8_f32(T.i32, sc[2], sc[3], pk, 1)
                            words.append(fx.Int32(pk))
                        row_t = fx.Int32(h_m * 128 + q * 64) + row_q
                        row_g = row_base + row_t
                        ok = row_t < n_valid
                        off = ok.select(
                            row_g * fx.Int32(INTER) + block_n * fx.Int32(128) + rd_col,
                            fx.Int32(out_rows * INTER),
                        )
                        buffer_store(
                            Vec.from_elements(words, fx.Int32),
                            out_rsrc,
                            off,
                            offset_is_bytes=True,
                        )
                        d0 = row_g >> fx.Int32(5)
                        d1 = (row_g >> fx.Int32(4)) & fx.Int32(1)
                        d2 = row_g & fx.Int32(15)
                        b_off = (
                            d0 * fx.Int32(SC_COLS32)
                            + d3 * fx.Int32(256)
                            + d5 * fx.Int32(64)
                            + d2 * fx.Int32(4)
                            + d4 * fx.Int32(2)
                            + d1
                        )
                        b_off = (ok & writer).select(b_off, fx.Int32(scale_bytes))
                        buffer_store(
                            e8m0.to(fx.Int8), osc_rsrc, b_off, offset_is_bytes=True
                        )

        # Wait for this rank's Stage1 plan of this call (parity flipped by prepare):
        # plan_ready[parity*npes + rank] == expected[parity], as the Stage1 consumers do.
        parity = fx.Int32(comm_ops.load_i32_system(parity_p, fx.Int32(0)))
        expected = fx.Int32(comm_ops.load_i32_system(expected_p, parity))
        if fx.Int32(fx.thread_idx.x) == fx.Int32(0):
            comm_ops.wait_i32_until_equals(
                fx.Int64(plan_ready)
                + fx.Int64(parity * fx.Int32(npes) + fx.Int32(rank)) * fx.Int64(4),
                expected,
            )
        fx.barrier()
        comm_ops.fence_system_acquire()

        def gemm_role():
            tab_lds = lds_addr(lds.E_tab)
            lane = fx.thread_idx.x % fx.Int32(64)
            values, n_tiles = tile_table(tile_end, lane, SBM=SBM, experts=experts)
            tile_table_to_lds(values, tab_lds, lane)
            lds_barrier()
            # Work items from an atomic counter (``cursor``, zeroed before the launch), so
            # they are taken in expert order -- the order the data lands in.
            # One atomic per work item; LDS broadcasts it (the tile table's 4th row
            # of 64 words is the last one tile_find reads, so word 4*64 is free).
            bcast_addr = tab_lds + fx.Int32(4 * 64 * 4)

            def fetch():
                if fx.Int32(fx.thread_idx.x) == fx.Int32(0):
                    fx.ptr_store(
                        fx.Int32(comm_ops.atomic_add_agent(cursor, fx.Int32(1))),
                        lds_typed_ptr(bcast_addr, T.i32, align=4),
                    )
                lds_barrier()
                v = fx.Int32(
                    rocdl.readfirstlane(
                        T.i32,
                        fx.ptr_load(
                            lds_typed_ptr(bcast_addr, T.i32, align=4)
                        ).ir_value(),
                    )
                )
                fx.barrier()
                return v

            g_slot0, g_stride = fx.Int64(0), fx.Int64(0)
            if const_expr(ldsw):
                g_slot0, g_stride = signal_slots()
            w = fetch()
            while w < n_tiles * n_blocks:
                process_tile(w // n_blocks, w % n_blocks, tab_lds, g_slot0, g_stride)
                fx.barrier()
                w = fetch()

        def sdma_comm_role():
            """Wave p (p != rank), lane 0: this rank's rows for peer p, one put per segment."""
            import mori.cco.device.flydsl as cco

            npes_, epr_, rank_ = npes, experts, rank
            te = npes_ * epr_
            lane = fx.thread_idx.x % fx.Int32(64)
            peer = fx.Int32(rocdl.readfirstlane(T.i32, fx.thread_idx.x // fx.Int32(64)))
            if (
                (peer < fx.Int32(npes_))
                & (peer != fx.Int32(rank_))
                & (lane == fx.Int32(0))
            ):
                disp = ptr_buf_tensor(addr_disp, fx.Int64)
                local_hist = ptr_buf_tensor(
                    disp[fx.Int32(int(DispatchSlot.LOCAL_HIST))], fx.Int32
                )
                pair_base = ptr_buf_tensor(
                    disp[fx.Int32(int(DispatchSlot.PAIR_BASE))], fx.Int32
                )
                task_base = ptr_buf_tensor(
                    disp[fx.Int32(int(DispatchSlot.TASK_ROW_BASE))], fx.Int32
                )
                group_base = ptr_buf_tensor(
                    disp[fx.Int32(int(DispatchSlot.GROUP_TASK_BASE))], fx.Int32
                )
                c_parity = fx.Int32(comm_ops.load_i32_system(parity_p, fx.Int32(0)))
                _, pair_a, _, _, _ = _load_fanout_pair(
                    disp[fx.Int32(int(DispatchSlot.FANOUT_PAIR_CONFIG))],
                    peer,
                    c_parity,
                    npes=npes_,
                )
                sd = cco.DevComm(dev_comm).sdma()
                q = fx.Int32(0)
                spare = fx.Int64(off_spare)
                for j in range(fx.Int32(0), fx.Int32(epr_ + 1), fx.Int32(1)):
                    grp = j == fx.Int32(0)
                    seg = grp.select(
                        fx.Int32(te) + peer, peer * fx.Int32(epr_) + j - fx.Int32(1)
                    )
                    cnt = fx.Int32(local_hist[seg])
                    src_row = fx.Int64(fx.Int32(pair_base[seg]))
                    dst_row = fx.Int64(
                        grp.select(
                            fx.Int32(group_base[peer * fx.Int32(epr_) + pair_a]),
                            fx.Int32(task_base[seg]),
                        )
                    )
                    has = cnt > fx.Int32(0)
                    # Empty segment: 4-byte puts within the spare area keep one signal per segment.
                    nx = fx.Int64(has.select(cnt * fx.Int32(K), fx.Int32(4)))
                    nsc = fx.Int64(has.select(cnt * fx.Int32(K // 32), fx.Int32(4)))
                    sx = has.select(fx.Int64(off_slab_x) + src_row * fx.Int64(K), spare)
                    dx = has.select(fx.Int64(off_land_x) + dst_row * fx.Int64(K), spare)
                    ss = has.select(
                        fx.Int64(off_slab_sc) + src_row * fx.Int64(K // 32), spare
                    )
                    ds = has.select(
                        fx.Int64(off_land_sc) + dst_row * fx.Int64(K // 32), spare
                    )
                    sd.put(
                        peer,
                        win,
                        dx,
                        win,
                        sx,
                        nx,
                        q,
                        coop=cco.CoopScope.THREAD,
                        local_signal=False,
                        remote_signal=False,
                    )
                    sd.put(
                        peer,
                        win,
                        ds,
                        win,
                        ss,
                        nsc,
                        q,
                        coop=cco.CoopScope.THREAD,
                        local_signal=False,
                        remote_signal=True,
                    )
                # Every put has read the slab before the next call's pack rewrites it.
                sd.quiet_queue(peer, q)

        def shuffler_role(sid):
            """Blocks 1..N_SHUF (``sid`` = 0..N_SHUF-1): per expert, in order, wait for its rows on
            every source, then for this block's 256-row chunks of the expert (round robin): load
            the raw e8m0 rows, shuffle them into LDS, copy the chunk's contiguous ``shuffle_scale``
            bytes to ``A_scale`` (``shuf_scale``) at the sorted rows. Release, then raise flag
            ``shuf_flags[sid]`` to the expert's signal target."""
            tid = fx.Int32(fx.thread_idx.x)
            tid0 = tid == fx.Int32(0)
            ends = global_typed_ptr(tile_end, T.i32)
            tib_s = global_typed_ptr(tib, T.i32)
            out_rsrc = create_buffer_resource_from_addr(
                fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(shuf_scale).ir_value()))
            )
            stage_lds = lds_addr(lds.E_stage)
            sig0 = fx.Int64(global_typed_ptr(sig_base, T.i64, align=8)[0]) - fx.Int64(
                experts + 1
            )
            STEP = 256 // SBM
            CH_I32 = 256 * (K // 32) // 4
            sh_slot0, sh_stride = signal_slots()
            for e in range(fx.Int32(0), fx.Int32(experts), fx.Int32(1)):
                need_e = sig0 + fx.Int64(e + fx.Int32(2))
                if tid0:
                    wait_signals(sh_slot0, sh_stride, need_e)
                fx.barrier()
                t_lo = fx.Int32(
                    rocdl.readfirstlane(
                        T.i32,
                        (e > fx.Int32(0)).select(
                            fx.Int32(
                                ends[
                                    (e > fx.Int32(0)).select(
                                        e - fx.Int32(1), fx.Int32(0)
                                    )
                                ]
                            ),
                            fx.Int32(0),
                        ),
                    )
                )
                t_hi = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(ends[e])))
                for t in range(
                    t_lo + sid * fx.Int32(STEP), t_hi, fx.Int32(N_SHUF * STEP)
                ):
                    n_rows = (t_hi - t) * fx.Int32(SBM)
                    n_rows = (n_rows < fx.Int32(256)).select(n_rows, fx.Int32(256))
                    tib_bases = [
                        fx.Int32(
                            rocdl.readfirstlane(
                                T.i32,
                                fx.Int32(
                                    tib_s[
                                        (t + fx.Int32(i) < t_hi).select(
                                            t + fx.Int32(i), t
                                        )
                                    ]
                                ),
                            )
                        )
                        for i in range_constexpr(STEP)
                    ]
                    shf = ScaleLdsWordS2R(N_TILES_A, tid % fx.Int32(64), K)
                    shf.write_wide(
                        stage_lds, shf.load_wide(land_sc, tib_bases, SBM, n_rows, tid)
                    )
                    lds_barrier()
                    # The chunk's rows [0, n_rows) are the first n_rows * K/32 bytes of its
                    # 256-row shuffled block (whole 32-row groups): a straight coalesced copy.
                    n_i32 = n_rows * fx.Int32(K // 128)
                    base = t * fx.Int32(SBM * (K // 128))
                    lp = lds_typed_ptr(stage_lds, T.i32, align=4)
                    for i in range_constexpr(CH_I32 // 512):
                        w_i = fx.Int32(i * 512) + tid
                        if w_i < n_i32:
                            buffer_store(fx.Int32(lp[w_i]), out_rsrc, base + w_i)
                    fx.barrier()
                comm_ops.fence_system_release()
                fx.barrier()
                if tid0:
                    comm_ops.store_i64_global_system(
                        fx.Int64(shuf_flags) + fx.Int64(sid * fx.Int32(8)), need_e
                    )

        if fx.Int32(fx.block_idx.x) == fx.Int32(0):
            sdma_comm_role()
        elif const_expr(N_SHUF > 0):
            bid = fx.Int32(fx.block_idx.x)
            if bid <= fx.Int32(N_SHUF):
                shuffler_role(bid - fx.Int32(1))
            else:
                gemm_role()
        else:
            gemm_role()

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B_T: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        tile_end: fx.Int64,
        land_sc: fx.Int64,
        tib: fx.Int64,
        parity_p: fx.Int64,
        s1_out: fx.Int64,
        plan_ready: fx.Int64,
        expected_p: fx.Int64,
        shuf_flags: fx.Int64,
        s1_out_scale: fx.Int64,
        cursor: fx.Int64,
        shuf_scale: fx.Int64,
        addr_disp: fx.Int64,
        win: fx.Int64,
        dev_comm: fx.Int64,
        sig_base: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            A,
            B_T,
            A_scale,
            B_scale,
            c_m,
            tile_end,
            land_sc,
            tib,
            parity_p,
            s1_out,
            plan_ready,
            expected_p,
            shuf_flags,
            s1_out_scale,
            cursor,
            shuf_scale,
            addr_disp,
            win,
            dev_comm,
            sig_base,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(grid=(grid_ctas, 1, 1), block=(512, 1, 1), stream=stream)

    return set_slp_vectorize_once_used(launch, False)
