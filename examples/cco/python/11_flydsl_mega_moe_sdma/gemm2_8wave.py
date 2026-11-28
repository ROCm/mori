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
"""MegaMoEV2 GEMM2 on an 8-wave 256x256 A8W4 main loop, for the SDMA Stage2.

``MegaMoESdma(gemm2="8wave")``: an alternative to the aiter-mainloop GEMM2 of
``mega_moe_gemm2_with_sdma`` with the same plan, window and combine; the same
kernel as aiter's opt-in ``MegaMoEV2(stage2_sdma=True)`` path (ROCm/aiter#7196).

``out = act(A) @ w2[expert].T`` for the Stage1 output rows, weighted by the routing
weight and quantised to fp8 1x32, then written to the local send buffer ``S`` at
the plan kernel's ``out_row``; workgroup 0 ships finished chunks of ``S`` to the
peers with SDMA (see ``mega_moe_gemm2_with_sdma``).

Inputs are MegaMoEV2's Stage2 inputs as they are:

* A    Stage1 output, fp8 ``[rows, K]`` in sorted-row (expert-major) order.
* A-s  Stage1 output scale ``_s1_osd``: ``shuffle_scale`` layout over rows padded to 256.
* B    w2, ``shuffle_weight_a16w4(w2, 16, False)``: per 16-row group and 128-K step a
       contiguous 1 KB block ``[k_lane 4][n 16][16 B]``, copied to LDS as is and read
       conflict-free -- lane (g, n) at ``g*256 + n*16``, 32 contiguous K of fp4.
* B-s  w2 scale, ``shuffle_scale_a16w4``: the same super-row layout as A-s.

Main loop: the 8-wave mxfp8 pipeline of ``mxfp8_gemm_8wave`` (2 x 4 waves, BK=128,
A and B through LDS with cur/next double buffers), grouped by expert and with an
fp4 B operand. A 256-row tile never straddles experts: each expert's SBM-padded
segment is split into 256-row tiles, found in-kernel from ``expert_tile_end``.

Epilogue: a wave's 32 columns of an N-half are exactly one fp8 1x32 group, so rows
are weighted and quantised in registers (group max over 16 lanes x 2 tiles) -- the
bytes the default Stage2 writes. fp8 rows and e8m0 go to a dedicated LDS stage,
then two rows per store (lanes 0-31 / 32-63) are written through (sc0|sc1) to
``S``; a row the plan marked ``OUT_ROW_P2P`` goes to the origin rank's
``comb_inp[slot]`` instead. Each finished work item counts once per sort tile
toward its chunk's completion counter.

Persistent: one workgroup per CU; workgroup 0 is the comm workgroup, the others
walk the (tile, N-block) work items.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Float4E2M1FN, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import (
    G2SLoader,
    S2RLoader,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    wait_barrier,
)
from aiter.ops.flydsl.kernels.kernels_common import set_slp_vectorize_once_used
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import _fabs_f32 as fabs_f32
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import (
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
)
from aiter.ops.flydsl.kernels.tensor_shim import ptr_buf_tensor
from aiter.ops.flydsl.kernels.mega_moe.dispatch import (
    _wave_inclusive_scan_i32,
    _wave_reduce_max_i32,
)
from mega_moe_gemm2_with_sdma import (
    CDONE_STRIDE,
    INV_P2P,
    OUT_ROW_P2P,
    PUSH_U,
    publish_flag,
    sdma_put_run,
    sdma_signal_peer,
    sdma_wait_consumed,
)

__all__ = ["compile_gemm2_8wave_sdma"]

#: Plan-free section table in LDS (``plan_free=True``), int32 words: ``REC[expert][pe]`` =
#: {rows for pe in all sections before the expert's normal section, exclusive pe-prefix of the
#: normal section's counts, the normal section's count for pe, -}; ``PE[pe]`` = {group rows from
#: pe, exclusive pe-prefix of them, avail, send_base, n_send, -, -, -}; ``MISC`` = {pair expert a,
#: pair expert b, pair enabled, SBM-padded group section, members, -, -, -}.
PF_REC_W = 4
PF_PE_W = 8
PF_MISC_W = 8


@flyc.jit
def _spin_ge_from(addr_i64, cur0, val):
    """Spin until the system-scope i32 at ``addr_i64`` reaches ``val``; ``cur0`` is a first
    reading already loaded. Relaxed loads, no acquire (an acquire is a whole-L2 invalidate).
    """
    cur = fx.Int32(cur0)
    while cur < fx.Int32(val):
        rocdl.s_sleep(1)
        cur = fx.Int32(comm_ops.load_i32_global_system(addr_i64))
    return cur


def _st_i32(addr_i64, idx, v):
    """``((int32*)addr_i64)[idx] = v``. Stores go through these helpers: a subscript store whose target
    names a kernel closure variable inside control flow makes FlyDSL's AST rewriter treat that name as
    a local of the enclosing function."""
    global_typed_ptr(addr_i64, T.i32)[idx] = fx.Int32(v)


def _st_lds_i32(base_i32, idx, v):
    lds_typed_ptr(base_i32, T.i32)[idx] = fx.Int32(v)


def opaque_i64(v):
    """``v`` (wave-uniform i64) through an identity the compiler cannot see through, so an
    address derived from a kernel argument is formed where it is used instead of being hoisted
    out of the persistent loop (and kept live across the K loop)."""
    return fx.Int64(
        _llvm.inline_asm(
            res=T.i64,
            operands_=[fx.Int64(v).ir_value()],
            asm_string="s_mov_b64 $0, $1",
            constraints="=s,s",
            has_side_effects=True,
        )
    )


BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
LDS_LIMIT_BYTES = 160 * 1024
#: Send-buffer stores: sc0|sc1, write-through to memory, so retiring them is the
#: release the copy engine needs (a system-scope release per item would instead
#: write back the whole L2).
SEND_CM = 1 | 16


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
    before = _wave_reduce_max_i32(
        (ok & (cum_l <= j)).select(lane_id + fx.Int32(1), fx.Int32(0)), lane_id
    )
    e = fx.Int32(rocdl.readfirstlane(T.i32, before))
    e = (e < fx.Int32(experts)).select(e, fx.Int32(experts - 1))

    def from_lane(v):
        return fx.Int32(
            rocdl.readfirstlane(
                T.i32, fx.Int32(rocdl.ds_bpermute(T.i32, e * fx.Int32(4), v))
            )
        )

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


def e8m0_of(amax):
    """The fp8 1x32 block exponent of a group's absolute max (the default Stage2's rule)."""
    working_bits = (amax * fx.Int32(0x3B124925).bitcast(fx.Float32)).bitcast(fx.Int32)
    mantissa = working_bits & fx.Int32(0x7FFFFF)
    biased_exp = (working_bits >> fx.Int32(23)) & fx.Int32(0xFF)
    e8m0 = (mantissa != fx.Int32(0)).select(biased_exp + fx.Int32(1), biased_exp)
    e8m0 = (e8m0 > fx.Int32(0xFF)).select(fx.Int32(0xFF), e8m0)
    return (amax.bitcast(fx.Int32) == fx.Int32(0)).select(fx.Int32(0), e8m0)


def epi_quant_half(
    frags_n0,
    frags_n1,
    *,
    h_m,
    wave_m,
    wave_n,
    lane_id,
    mfma_idx,
    n_tiles_a,
    wts_lds,
    stage_lds,
    scale_lds,
):
    """Weight and fp8 1x32-quantise one 128-row half in registers; bytes and e8m0 to LDS.

    A wave's 32 columns of an N-half are exactly one quant group: per row the group
    max spans the 16 lanes of a lane-quarter and its two 16-column tiles.
    """
    wts_p = lds_typed_ptr(wts_lds, T.f32)
    st_p = lds_typed_ptr(stage_lds, T.i8, align=1)
    sc_p = lds_typed_ptr(scale_lds, T.i8, align=1)
    pk_ty = T.vec(2, T.i16)
    for ti in range_constexpr(n_tiles_a):
        for i in range_constexpr(4):
            r = (
                wave_m * (n_tiles_a * 16) + ti * 16 + (lane_id // 16) * 4 + i
            )  # row in the half
            w = fx.Float32(wts_p[fx.Int32(h_m * 128) + r])
            for h_n in range_constexpr(2):
                frags = frags_n1 if h_n else frags_n0
                v0 = Vec(frags[mfma_idx(ti, 0)])[i] * w
                v1 = Vec(frags[mfma_idx(ti, 1)])[i] * w
                amax = fabs_f32(v0).maximumf(fabs_f32(v1))
                for sh in (1, 2, 4, 8):
                    amax = amax.maximumf(amax.shuffle_xor(fx.Int32(sh), fx.Int32(64)))
                e8m0 = e8m0_of(amax)
                block_scale = (e8m0 << 23).bitcast(fx.Float32)
                pkw = rocdl.cvt_scalef32_pk_fp8_f32(
                    pk_ty,
                    fx.full(2, 0, fx.Int16).ir_value(),
                    v0.ir_value(),
                    v1.ir_value(),
                    block_scale.ir_value(),
                    0,
                )
                word = fx.Vector(pkw).bitcast(fx.Int32)[0]
                col = h_n * 128 + wave_n * 32 + lane_id % 16
                st_p[r * 256 + col] = (word & fx.Int32(0xFF)).to(fx.Int8)
                st_p[r * 256 + col + 16] = ((word >> fx.Int32(8)) & fx.Int32(0xFF)).to(
                    fx.Int8
                )
                sc_p[r * 8 + h_n * 4 + wave_n] = e8m0.to(fx.Int8)


# --- The kernel ------------------------------------------------------------------------


def compile_gemm2_8wave_sdma(
    *,
    K: int,
    N: int,
    experts: int,
    topk: int,
    npes: int,
    max_tok: int,
    recv_cap: int,
    SBM: int,
    rank: int,
    layout,
    chunk_tiles: int,
    grid_ctas: int,
    comb_bytes: int,
    waves_per_eu: int = 2,
    plan_free: bool = False,
    total_segments: int = 0,
    inv_pipe: bool = False,
):
    """Build the launcher of the 8-wave GEMM2 + SDMA comm kernel.

    ``layout``: the ``mega_moe_gemm2_with_sdma.ArenaLayout`` of the cco window; ``chunk_tiles``: sort
    tiles per shipped chunk; ``grid_ctas``: one per CU, workgroup 0 being the comm
    workgroup; ``comb_bytes``: size of a rank's ``comb_inp`` (rows past the end of ``S``).

    ``plan_free``: no plan kernel runs before this one. Every workgroup builds the section table
    of aiter's compact layout (``total_segments`` = bigcnt's row length) in LDS in its prologue;
    the epilogue computes ``out_row`` in closed form, and the comm workgroup the offsets, the
    chunk bounds, the inverse map for the inv push, this rank's own inv entries, the overflow
    count and the cdone reset. The plan's outputs are not read, so their argument slots carry
    the plan-free inputs: ``out_row`` = bigcnt, ``tile_base`` = fanout_pair_config, ``cursor`` =
    stage1 epoch parity, ``idx_send`` = int32 counters [overflow rows, layout mismatches].

    ``inv_pipe`` (plan-free only): when a peer's comm wave pushes its inv entries. ``False``: all of
    them before the first put. ``True``: chunk c+1's entries after chunk c's put, chunk 0's before
    the first chunk is ready. Either way every entry is complete before the signal-carrying put (and
    before the flag), and none is written before the peer has consumed the last call.
    """
    assert K % 256 == 0 and N % BLOCK_N == 0, f"K={K} needs % 256, N={N} % {BLOCK_N}"
    assert 256 % SBM == 0, f"SBM={SBM} must divide 256"
    assert npes <= 8, "the comm workgroup has one wave per peer"
    assert layout.row_bytes == N + N // 32
    plan_free = bool(plan_free)
    TE = npes * experts
    TS = total_segments
    if plan_free:
        # A comm wave spins on every chunk before the comm workgroup resets cdone.
        assert npes >= 2, "plan_free needs a peer"
        assert experts <= 64, "the section table holds one expert per lane"
        assert TS == TE + npes, f"bigcnt rows hold {TS} segments, expected {TE + npes}"
    inv_pipe = bool(inv_pipe)
    assert plan_free or not inv_pipe, "inv_pipe needs plan_free"
    PF_REC = 0
    PF_PE = experts * npes * PF_REC_W
    PF_MISC = PF_PE + npes * PF_PE_W
    pf_bytes = (PF_MISC + PF_MISC_W) * 4 if plan_free else 0
    send_rows = layout.send_rows
    # Window offsets from the start of S (the ``send_base`` argument is my window's S).
    rel_done = layout.off_done - layout.off_s
    rel_counter = layout.off_counter - layout.off_s
    rel_inv = layout.off_inv - layout.off_s

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
    send_nbytes = layout.send_rows * layout.row_bytes
    row_bytes = layout.row_bytes
    gemm_ctas = grid_ctas - 1
    log2_max_tok = max_tok.bit_length() - 1

    epi_bytes = 128 * 256 + 128 * 8 + 3 * 256 * 4 + 4 * 64 * 4
    assert 4 * (a_lds_size + b_lds_size) + epi_bytes + pf_bytes <= LDS_LIMIT_BYTES
    if plan_free:
        # WG 0 keeps one start-of-expert array per wave in a main-loop buffer it never uses.
        assert 8 * 64 * 4 <= a_lds_size

    kname = (
        f"mega_moe_gemm2_8wave_sdma_n{N}_k{K}_e{experts}_t{topk}_p{npes}_m{max_tok}_s{SBM}"
        f"_s{layout.send_rows}r{layout.recv_rows}_ct{chunk_tiles}_g{grid_ctas}_r{rank}"
        f"{'_pf' if plan_free else ''}{'_ip' if inv_pipe else ''}"
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
        # Epilogue, never aliased with the main loop's buffers.
        E_stage: fx.Array[
            fx.Float8E4M3FN, 128 * 256, 16
        ]  # fp8 rows of one 128-row half
        E_scale: fx.Array[fx.Float8E4M3FN, 128 * 8, 16]  # their e8m0, 8 per row
        E_orow: fx.Array[fx.Float8E4M3FN, 256 * 4, 16]  # out_row per tile row
        E_src: fx.Array[fx.Float8E4M3FN, 256 * 4, 16]  # packed srcmap word per tile row
        E_wts: fx.Array[fx.Float8E4M3FN, 256 * 4, 16]  # routing weight per tile row
        E_tab: fx.Array[fx.Float8E4M3FN, 4 * 64 * 4, 16]  # the tile table
        if plan_free:
            E_pf: fx.Array[fx.Float8E4M3FN, pf_bytes, 16]  # the section table

    @flyc.kernel(name=kname, known_block_size=[512, 1, 1])
    def kernel(
        A: fx.Tensor,
        B_T: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        tile_end: fx.Int64,
        stids: fx.Int64,
        sweights: fx.Int64,
        trb: fx.Int64,
        p2p: fx.Int64,
        num_valid: fx.Int64,
        out_row: fx.Int64,
        send_base: fx.Int64,
        dev_comm: fx.Int64,
        win: fx.Int64,
        cdone: fx.Int64,
        tile_base: fx.Int64,
        cursor: fx.Int64,
        idx_send: fx.Int64,
    ):
        import mori.cco.device.flydsl as cco

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

        # --- Plan-free: the section table of aiter's compact layout ---------------------
        # Per local expert j: [group section G_j if j is a member of my rank's fanout pair |
        # normal section N_j], each ordered by source pe and padded to SBM. For pe:
        # cnt(G_j) = bigcnt[pe][TE + rank], cnt(N_j) = bigcnt[pe][rank*E + j].

        def pf_ld(idx):
            return fx.Int32(lds_typed_ptr(lds_addr(lds.E_pf), T.i32)[idx])

        def pf_st(idx, v):
            _st_lds_i32(lds_addr(lds.E_pf), idx, v)

        def pf_vec4(word):
            """Four int32 of the section table from word ``word`` (16-byte aligned)."""
            return Vec(
                lds_vec_load(
                    lds_addr(lds.E_pf),
                    word * fx.Int32(4),
                    Vec.make_type(4, fx.Int32),
                    fx.Int32,
                    align=16,
                )
            )

        def pf_issue():
            """The table's loads, issued together: one bigcnt count per thread (wave = pe,
            lane = local expert), the pe's group count, the pair entries of both parities and
            the parity (no dependent load)."""
            tid = fx.Int32(fx.thread_idx.x)
            wave = tid // fx.Int32(64)
            lane = tid % fx.Int32(64)
            okp = wave < fx.Int32(npes)
            okj = lane < fx.Int32(experts)
            pe = okp.select(wave, fx.Int32(0))
            j = okj.select(lane, fx.Int32(0))
            bc = global_typed_ptr(out_row, T.i32)  # plan-free: bigcnt
            cnt = fx.Int32(bc[pe * fx.Int32(TS) + fx.Int32(rank * experts) + j])
            gcol = fx.Int32(bc[pe * fx.Int32(TS) + fx.Int32(TE + rank)])
            par = fx.Int32(comm_ops.load_i32_system(cursor, fx.Int32(0)))  # parity
            pc0 = fx.Int32(
                comm_ops.load_i32_system(tile_base, fx.Int32(rank))
            )  # pair_cfg
            pc1 = fx.Int32(comm_ops.load_i32_system(tile_base, fx.Int32(npes + rank)))
            return okp, okj, wave, lane, cnt, gcol, par, pc0, pc1

        def pair_of(par, pc0, pc1):
            """(enabled, a, b, members) of a fanout_pair_config entry at the stage1 parity (0/1)."""
            packed = (par == fx.Int32(0)).select(pc0, pc1)
            enabled = (packed & fx.Int32(1 << 16)) != fx.Int32(0)
            a = packed & fx.Int32(0xFF)
            b = (packed >> fx.Int32(8)) & fx.Int32(0xFF)
            nmem = enabled.select(
                (a == b).select(fx.Int32(1), fx.Int32(2)), fx.Int32(0)
            )
            return enabled, a, b, nmem

        def pf_write(ld):
            """Pass 1 (wave = pe, lane = expert): rows for pe before each normal section, the
            counts, n_send, the pair. Needs an LDS barrier before ``pf_prefix``."""
            okp, okj, wave, lane, cnt, gcol, par, pc0, pc1 = ld
            cnt = (okp & okj).select(cnt, fx.Int32(0))
            gcol = okp.select(gcol, fx.Int32(0))
            enabled, a, b, nmem = pair_of(par, pc0, pc1)
            # members m <= j: G_m (m <= j) all come before N_j
            le = (a <= lane).select(fx.Int32(1), fx.Int32(0)) + (
                (b <= lane) & (b != a)
            ).select(fx.Int32(1), fx.Int32(0))
            nle = enabled.select(le, fx.Int32(0))
            incl = _wave_inclusive_scan_i32(cnt, lane)
            cum_n = incl - cnt + nle * gcol
            tot = fx.Int32(
                rocdl.readfirstlane(
                    T.i32, fx.Int32(rocdl.ds_bpermute(T.i32, fx.Int32(63 * 4), incl))
                )
            )
            rec = fx.Int32(PF_REC) + (lane * fx.Int32(npes) + wave) * fx.Int32(PF_REC_W)
            if okp & okj:
                pf_st(rec, cum_n)
                pf_st(rec + fx.Int32(2), cnt)
            if okp & (lane == fx.Int32(0)):
                pe_w = fx.Int32(PF_PE) + wave * fx.Int32(PF_PE_W)
                pf_st(pe_w, gcol)
                pf_st(pe_w + fx.Int32(4), tot + nmem * gcol)
            if (wave == fx.Int32(0)) & (lane == fx.Int32(0)):
                pf_st(fx.Int32(PF_MISC), a)
                pf_st(fx.Int32(PF_MISC + 1), b)
                pf_st(fx.Int32(PF_MISC + 2), enabled.select(fx.Int32(1), fx.Int32(0)))
                pf_st(fx.Int32(PF_MISC + 4), nmem)

        def pf_prefix():
            """Pass 2: exclusive prefixes over pe (normal sections, group section, n_send ->
            send_base), avail, the padded group section. Needs an LDS barrier after."""
            tid = fx.Int32(fx.thread_idx.x)
            if tid < fx.Int32(experts):
                run = fx.Int32(0)
                for q in range_constexpr(npes):
                    i = fx.Int32(PF_REC) + (
                        tid * fx.Int32(npes) + fx.Int32(q)
                    ) * fx.Int32(PF_REC_W)
                    c = pf_ld(i + fx.Int32(2))
                    pf_st(i + fx.Int32(1), run)
                    run = run + c
            if tid == fx.Int32(64):
                g = fx.Int32(0)
                s = fx.Int32(0)
                for q in range_constexpr(npes):
                    base = PF_PE + q * PF_PE_W
                    gc = pf_ld(fx.Int32(base))
                    ns = pf_ld(fx.Int32(base + 4))
                    room = fx.Int32(send_rows) - s
                    room = (room > fx.Int32(0)).select(room, fx.Int32(0))
                    pf_st(fx.Int32(base + 1), g)
                    pf_st(fx.Int32(base + 2), (ns < room).select(ns, room))
                    pf_st(fx.Int32(base + 3), s)
                    g = g + gc
                    s = s + ns
                pf_st(
                    fx.Int32(PF_MISC + 3),
                    (g + fx.Int32(SBM - 1)) // fx.Int32(SBM) * fx.Int32(SBM),
                )

        def pf_misc():
            """(enabled, a, b, gpad), wave-uniform."""
            m = pf_vec4(fx.Int32(PF_MISC))
            return (
                fx.Int32(m[2]) != fx.Int32(0),
                fx.Int32(m[0]),
                fx.Int32(m[1]),
                fx.Int32(m[3]),
            )

        def pf_out_row(packed, grow, in_tile, row_base, lane_id):
            """out_row of the row at sorted position ``grow`` with srcmap word ``packed``, in
            closed form; and 1 for a valid row outside its predicted run (else 0)."""
            # The tile's expert: the segment that holds row_base (one ballot over the tile table).
            tabp = lds_typed_ptr(lds_addr(lds.E_tab), T.i32)
            start_l = fx.Int32(tabp[lane_id]) * fx.Int32(SBM)
            seg_l = fx.Int32(tabp[fx.Int32(64) + lane_id])
            hit = (
                (lane_id < fx.Int32(experts))
                & (start_l <= row_base)
                & (row_base < start_l + seg_l)
            )
            mask = fx.Int64(rocdl.ballot(T.i64, hit))
            e = fx.Int32(fx.math.ctpop(mask - fx.Int64(1)))
            e = (e < fx.Int32(experts)).select(e, fx.Int32(0))
            seg0 = fx.Int32(
                rocdl.readfirstlane(
                    T.i32, fx.Int32(rocdl.ds_bpermute(T.i32, e * fx.Int32(4), start_l))
                )
            )
            enabled, a, b, gpad = pf_misc()
            mem = enabled & ((e == a) | (e == b))
            # The row: section, offset in its (section, pe) run, place in my run for pe.
            t = packed & fx.Int32(0x00FFFFFF)
            s = packed.shrui(fx.Int32(24)) & fx.Int32(0xFF)
            pe = t >> fx.Int32(log2_max_tok)
            valid = (
                in_tile
                & (t < fx.Int32(recv_cap))
                & (s < fx.Int32(topk))
                & (pe < fx.Int32(npes))
            )
            pes = valid.select(pe, fx.Int32(0))
            rec = pf_vec4(
                fx.Int32(PF_REC) + (e * fx.Int32(npes) + pes) * fx.Int32(PF_REC_W)
            )
            per = pf_vec4(fx.Int32(PF_PE) + pes * fx.Int32(PF_PE_W))
            cum_n, pfx_n, cnt_n = fx.Int32(rec[0]), fx.Int32(rec[1]), fx.Int32(rec[2])
            gcol, pfx_g = fx.Int32(per[0]), fx.Int32(per[1])
            avail, sbase = fx.Int32(per[2]), fx.Int32(per[3])
            rel = grow - seg0
            is_g = mem & (rel < gpad)
            off = is_g.select(rel - pfx_g, rel - mem.select(gpad, fx.Int32(0)) - pfx_n)
            cnt = is_g.select(gcol, cnt_n)
            cum = is_g.select(cum_n - gcol, cum_n)
            inside = valid & (off >= fx.Int32(0)) & (off < cnt)
            place = cum + off
            fits = inside & (place < avail)
            orow = fits.select(
                sbase + place, inside.select(fx.Int32(OUT_ROW_P2P), fx.Int32(-1))
            )
            bad = valid.select(inside.select(fx.Int32(0), fx.Int32(1)), fx.Int32(0))
            return orow, bad

        def pf_peer_vals(p):
            """Wave-uniform per-pe values: (gcol, pfx_g, avail, send_base, n_send)."""
            v = pf_vec4(fx.Int32(PF_PE) + p * fx.Int32(PF_PE_W))
            return (
                fx.Int32(v[0]),
                fx.Int32(v[1]),
                fx.Int32(v[2]),
                fx.Int32(v[3]),
                pf_ld(fx.Int32(PF_PE + 4) + p * fx.Int32(PF_PE_W)),
            )

        def pf_tile_base(t, p, total_tiles, pv):
            """tile_base[t][p] of the plan in closed form: rows for p in sort tiles < t, capped
            by avail (per lane; ``t`` may differ per lane)."""
            gcol_p, pfx_g_p, avail_p, _, nsend_p = pv
            tabp = lds_typed_ptr(lds_addr(lds.E_tab), T.i32)
            # The expert whose segment holds tile t: the last one starting at or before it.
            lo = fx.Int32(0)
            for step in (32, 16, 8, 4, 2, 1):
                if const_expr(step < experts):
                    cand = lo + fx.Int32(step)
                    okc = cand < fx.Int32(experts)
                    v = fx.Int32(tabp[okc.select(cand, fx.Int32(0))])
                    lo = (okc & (v <= t)).select(cand, lo)
            e = lo
            rel = t * fx.Int32(SBM) - fx.Int32(tabp[e]) * fx.Int32(SBM)
            enabled, a, b, gpad = pf_misc()
            mem = enabled & ((e == a) | (e == b))
            rec = pf_vec4(
                fx.Int32(PF_REC) + (e * fx.Int32(npes) + p) * fx.Int32(PF_REC_W)
            )
            cum_n, pfx_n, cnt_n = fx.Int32(rec[0]), fx.Int32(rec[1]), fx.Int32(rec[2])

            def clamp(x, hi):
                x = (x > fx.Int32(0)).select(x, fx.Int32(0))
                return (x < hi).select(x, hi)

            tb_g = cum_n - gcol_p + clamp(rel - pfx_g_p, gcol_p)
            tb_n = cum_n + clamp(rel - mem.select(gpad, fx.Int32(0)) - pfx_n, cnt_n)
            tb = (mem & (rel < gpad)).select(tb_g, tb_n)
            tb = (tb < avail_p).select(tb, avail_p)
            capped = (nsend_p < avail_p).select(nsend_p, avail_p)
            return (t >= total_tiles).select(capped, tb)

        def pf_row_of_place(j, p, st_p, pv):
            """Sorted position of the row at place ``j`` of my run for ``p`` (per lane): the
            inverse of the closed form. ``st_p``: LDS array of each expert's first place for p.
            """
            gcol_p, pfx_g_p = pv[0], pv[1]
            stp = lds_typed_ptr(st_p, T.i32)
            lo = fx.Int32(0)
            for step in (32, 16, 8, 4, 2, 1):
                if const_expr(step < experts):
                    cand = lo + fx.Int32(step)
                    okc = cand < fx.Int32(experts)
                    v = fx.Int32(stp[okc.select(cand, fx.Int32(0))])
                    lo = (okc & (v <= j)).select(cand, lo)
            e = lo
            seg0 = fx.Int32(lds_typed_ptr(lds_addr(lds.E_tab), T.i32)[e]) * fx.Int32(
                SBM
            )
            enabled, a, b, gpad = pf_misc()
            mem = enabled & ((e == a) | (e == b))
            rec = pf_vec4(
                fx.Int32(PF_REC) + (e * fx.Int32(npes) + p) * fx.Int32(PF_REC_W)
            )
            cum_n, pfx_n = fx.Int32(rec[0]), fx.Int32(rec[1])
            g_grp = seg0 + pfx_g_p + j - (cum_n - gcol_p)
            g_nrm = seg0 + mem.select(gpad, fx.Int32(0)) + pfx_n + j - cum_n
            return (mem & (j < cum_n)).select(g_grp, g_nrm)

        def process_tile(block_m, block_n, tab_lds):
            _, lane_id, wave_id, wave_m, wave_n = lane_ids()
            a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
            a_next0, a_next1 = lds.A_lds_next_0, lds.A_lds_next_1
            b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
            b_next0, b_next1 = lds.B_lds_next_0, lds.B_lds_next_1
            row_base, expert, n_valid = tile_find(
                tab_lds, block_m, lane_id, SBM=SBM, experts=experts
            )
            b_row_block = expert * n_blocks + block_n

            A0_gl_offset = row_base * K
            A1_gl_offset = (row_base + LDS_BLOCK_M) * K
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
            a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
            a_s2r = S2RLoader(wave_m, N_TILES_A)
            a_s2r.lane_id = lane_id
            b_s2r = S2RF4G(wave_n, N_TILES_B, lane_id)
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
            # The wait_barrier counts below assume 6 scale loads per K-pair.
            assert 2 * (N_TILES_A // 2) + 2 * (N_TILES_B // 2) == 6

            def scale_prefetch(sc, k0):
                if const_expr(k0 >= K_ITERS or k0 // 2 in sc):
                    return sc
                words = {
                    w: (a_sc if w[0] == "a" else b_sc).read(
                        (a_base16 if w[0] == "a" else b_base16)[int(w[1])], k0
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

            sc_pf = scale_prefetch({}, 0)
            b_g2s.load(b_cur0, B0_gl_offset)
            a_g2s.load(a_cur0, A0_gl_offset)
            b_g2s.load(b_cur1, B1_gl_offset)
            a_g2s.load(a_cur1, A1_gl_offset)
            # The wave_m == 1 group runs one barrier behind the other from here on.
            if wave_m == 1:
                rocdl.s_barrier()
            wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)
            b_g2s.load(b_next0, B0_gl_offset + B_K_STEP)
            a_g2s.load(a_next0, A0_gl_offset + BLOCK_K)
            b_g2s.load(b_next1, B1_gl_offset + B_K_STEP)
            wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

            for k in range_constexpr(K_ITERS - 2):
                if const_expr(k % 2 == 1):
                    sc_pf = scale_prefetch(sc_pf, k + 1)
                sa0 = scale_read(sc_pf, k, "a0")
                sb0 = scale_read(sc_pf, k, "b0")
                b0_frag = b_s2r.load(b_cur0)
                a0_frag = a_s2r.load(a_cur0)
                a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
                rocdl.s_barrier()

                c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)

                sb1 = scale_read(sc_pf, k, "b1")
                b1_frag = b_s2r.load(b_cur1)
                b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
                rocdl.s_barrier()

                c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)

                sa1 = scale_read(sc_pf, k, "a1")
                a1_frag = a_s2r.load(a_cur1)
                a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
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
            a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
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

            epilogue(
                (c00_frag, c01_frag),
                (c10_frag, c11_frag),
                row_base,
                n_valid,
                block_n,
                mfma.idx,
            )

        def epilogue(half0, half1, row_base, n_valid, block_n, mfma_idx):
            tid, lane_id, wave_id, wave_m, wave_n = lane_ids()
            orow_lds, src_lds, wts_lds = (
                lds_addr(lds.E_orow),
                lds_addr(lds.E_src),
                lds_addr(lds.E_wts),
            )
            stage_lds, scale_lds = lds_addr(lds.E_stage), lds_addr(lds.E_scale)
            if const_expr(plan_free):
                # The S guard (the plan's spin, moved here): S is rewritten below, so every
                # peer must have consumed the last call (done[pe] >= my counter), which with
                # the copy engine's FIFO means my last call's puts have finished reading S.
                # Relaxed system-scope loads of my own window, in flight with the row
                # descriptors' loads; later epilogues pass at once.
                g_lane = (tid < fx.Int32(npes)) & (tid != fx.Int32(rank))
                win_s = fx.Int64(send_base)
                done_a = (
                    win_s
                    + opaque_i64(fx.Int64(rel_done))
                    + fx.Int64(g_lane.select(tid, fx.Int32(0))) * fx.Int64(4)
                )
                done_v = fx.Int32(comm_ops.load_i32_global_system(done_a))
                consumed = fx.Int32(
                    comm_ops.load_i32_global_system(
                        win_s + opaque_i64(fx.Int64(rel_counter))
                    )
                )

            # Row descriptors of the tile: out_row, srcmap word, weight.
            row = tid % fx.Int32(256)
            in_tile = (row < n_valid) & (tid < fx.Int32(256))
            grow = row_base + in_tile.select(row, fx.Int32(0))
            sm_idx = global_typed_ptr(trb, T.i32)[
                grow // fx.Int32(SBM)
            ] + grow % fx.Int32(SBM)
            if const_expr(plan_free):
                packed = fx.Int32(global_typed_ptr(stids, T.i32)[sm_idx])
                weight = fx.Float32(global_typed_ptr(sweights, T.f32)[sm_idx])
                orow, bad = pf_out_row(packed, grow, in_tile, row_base, lane_id)
                # Impossible under aiter's compact layout; the host raises on the count.
                if (bad != fx.Int32(0)) & (block_n == fx.Int32(0)):
                    comm_ops.atomic_add_agent(
                        fx.Int64(idx_send) + fx.Int64(4), fx.Int32(1)
                    )
            else:
                orow = in_tile.select(
                    fx.Int32(global_typed_ptr(out_row, T.i32)[grow]), fx.Int32(-1)
                )
                packed = fx.Int32(global_typed_ptr(stids, T.i32)[sm_idx])
                weight = fx.Float32(global_typed_ptr(sweights, T.f32)[sm_idx])
            if tid < fx.Int32(256):
                lds_typed_ptr(orow_lds, T.i32)[row] = orow
                lds_typed_ptr(src_lds, T.i32)[row] = packed
                lds_typed_ptr(wts_lds, T.f32)[row] = weight
            if const_expr(plan_free):
                if g_lane:
                    _spin_ge_from(done_a, done_v, consumed)
            lds_barrier()

            send = create_buffer_resource_from_addr(
                fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(send_base).ir_value())),
                num_records_bytes=send_nbytes,
            )
            orow_p = lds_typed_ptr(orow_lds, T.i32)
            src_p = lds_typed_ptr(src_lds, T.i32)
            half = lane_id // fx.Int32(32)
            col = (lane_id % fx.Int32(32)) * fx.Int32(8)
            row_lead = (lane_id % fx.Int32(32)) == fx.Int32(0)
            for h_m in range_constexpr(2):
                if const_expr(h_m > 0):
                    lds_barrier()
                frags = half1 if h_m else half0
                epi_quant_half(
                    frags[0],
                    frags[1],
                    h_m=h_m,
                    wave_m=wave_m,
                    wave_n=wave_n,
                    lane_id=lane_id,
                    mfma_idx=mfma_idx,
                    n_tiles_a=N_TILES_A,
                    wts_lds=wts_lds,
                    stage_lds=stage_lds,
                    scale_lds=scale_lds,
                )
                lds_barrier()
                # Each wave copies 16 rows of the half, two per store (lanes 0-31 / 32-63).
                for it in range_constexpr(8):
                    r = wave_id * fx.Int32(16) + fx.Int32(2 * it) + half
                    o = fx.Int32(orow_p[fx.Int32(h_m * 128) + r])
                    payload = Vec(
                        lds_vec_load(
                            stage_lds,
                            r * 256 + col,
                            Vec.make_type(2, fx.Int32),
                            fx.Int32,
                            align=8,
                        )
                    )
                    scales = Vec(
                        lds_vec_load(
                            scale_lds,
                            r * 8,
                            Vec.make_type(2, fx.Int32),
                            fx.Int32,
                            align=8,
                        )
                    )
                    base = o * fx.Int32(row_bytes)
                    ok = o >= fx.Int32(0)
                    p_off = ok.select(
                        base + block_n * fx.Int32(256) + col, fx.Int32(send_nbytes)
                    )
                    buffer_store(
                        payload,
                        send,
                        p_off,
                        cache_modifier=SEND_CM,
                        offset_is_bytes=True,
                    )
                    s_off = (ok & row_lead).select(
                        base + fx.Int32(N) + block_n * fx.Int32(8),
                        fx.Int32(send_nbytes),
                    )
                    buffer_store(
                        scales,
                        send,
                        s_off,
                        cache_modifier=SEND_CM,
                        offset_is_bytes=True,
                    )
                    # Rows past the end of S go straight to the origin rank.
                    for h in range_constexpr(2):
                        rh = (
                            wave_id * fx.Int32(16)
                            + fx.Int32(2 * it + h)
                            + fx.Int32(h_m * 128)
                        )
                        if fx.Int32(rocdl.readfirstlane(T.i32, orow_p[rh])) == fx.Int32(
                            OUT_ROW_P2P
                        ):
                            direct_row(
                                fx.Int32(rocdl.readfirstlane(T.i32, src_p[rh])),
                                payload,
                                scales,
                                half == fx.Int32(h),
                                row_lead,
                                col,
                                block_n,
                            )
            publish(row_base, n_valid)

        def direct_row(p, payload, scales, mine, row_lead, col, block_n):
            """Store one row's 256 columns into ``comb_inp[slot]`` of its origin rank, as
            the default Stage2 does, and point that rank's ``inv[slot]`` at it."""
            t = p & fx.Int32(0x00FFFFFF)
            s = p.shrui(fx.Int32(24)) & fx.Int32(0xFF)
            dest_pe = t >> fx.Int32(log2_max_tok)
            slot = (t & fx.Int32(max_tok - 1)) * fx.Int32(topk) + s
            peer = fx.Int64(
                rocdl.readfirstlane(
                    T.i64, global_typed_ptr(p2p, T.i64, align=8)[dest_pe]
                )
            )
            w = cco.Window(win)
            # Like the comm waves: not before the origin rank has read the last call's
            # comb_inp and inv (its combine publishes done[] to this rank).
            if dest_pe != fx.Int32(rank):
                rsrc_counter = create_buffer_resource_from_addr(
                    fx.Int64(w.lsa_ptr(rank, layout.off_counter))
                )
                phase = fx.Int32(
                    buffer_load(rsrc_counter, 0, vec_width=1, dtype=T.i32)
                ) + fx.Int32(1)
                sdma_wait_consumed(w, dest_pe, phase, layout=layout, rank=rank)
            # Written through like S: retired stores are in the peer's memory before the
            # item publishes, with no system-scope release.
            buf = create_buffer_resource_from_addr(peer, num_records_bytes=comb_bytes)
            base = slot * fx.Int32(row_bytes)
            p_off = mine.select(
                base + block_n * fx.Int32(256) + col, fx.Int32(comb_bytes)
            )
            buffer_store(
                payload, buf, p_off, cache_modifier=SEND_CM, offset_is_bytes=True
            )
            s_off = (mine & row_lead).select(
                base + fx.Int32(N) + block_n * fx.Int32(8), fx.Int32(comb_bytes)
            )
            buffer_store(
                scales, buf, s_off, cache_modifier=SEND_CM, offset_is_bytes=True
            )
            if (block_n == fx.Int32(0)) & mine & row_lead:
                comm_ops.store_i32_global_system_monotonic(
                    fx.Int64(w.lsa_ptr(dest_pe, layout.off_inv))
                    + fx.Int64(slot) * fx.Int64(4),
                    fx.Int32(INV_P2P),
                )

        def publish(row_base, n_valid):
            """Every wave's rows have retired (written through); count each sort tile
            of the item once toward its chunk."""
            rocdl.s_waitcnt(vmcnt=0)
            fx.barrier()
            if fx.thread_idx.x == fx.Int32(0):
                st0 = row_base // fx.Int32(SBM)
                n_st = n_valid // fx.Int32(SBM)
                for i in range_constexpr(256 // SBM):
                    if fx.Int32(i) < n_st:
                        comm_ops.atomic_add_agent(
                            fx.Int64(cdone)
                            + fx.Int64((st0 + fx.Int32(i)) // fx.Int32(chunk_tiles))
                            * fx.Int64(CDONE_STRIDE * 4),
                            fx.Int32(1),
                        )

        def comm_role():
            """One wave per peer: fill the peer's ``inv`` for the rows I send it, put
            each finished chunk of my run for it, then signal it."""
            lane = fx.thread_idx.x % fx.Int32(64)
            peer = fx.Int32(rocdl.readfirstlane(T.i32, fx.thread_idx.x // fx.Int32(64)))
            is_lane0 = lane == fx.Int32(0)
            queue = fx.Int32(0)  # one SDMA channel per peer
            lay = layout
            if (peer < fx.Int32(npes)) & (peer != fx.Int32(rank)):
                w = cco.Window(win)
                sdma = cco.DevComm(dev_comm).sdma()
                rsrc_counter = create_buffer_resource_from_addr(
                    fx.Int64(w.lsa_ptr(rank, lay.off_counter))
                )
                # Read-only here: the combine advances it after the call.
                phase = fx.Int32(
                    buffer_load(rsrc_counter, 0, vec_width=1, dtype=T.i32)
                ) + fx.Int32(1)
                tile_base_buf = ptr_buf_tensor(tile_base, fx.Int32)
                idx_buf = ptr_buf_tensor(idx_send, fx.Int32)
                # Do not overwrite the peer's landing rows or inv before it has read the last call's.
                if is_lane0:
                    sdma_wait_consumed(w, peer, phase, layout=lay, rank=rank)
                # inv entries go out while the GEMM computes its first chunk; they must be
                # visible before any put or signal of mine, hence the vmcnt wait.
                # The plan's exact offsets: rows in my run for peer, the run's base in S,
                # and where it lands in peer's R.
                cursor_buf = ptr_buf_tensor(cursor, fx.Int32)
                cnt = cursor_buf[peer]
                run0 = fx.Int32(
                    rocdl.readfirstlane(T.i32, cursor_buf[fx.Int32(npes) + peer])
                )
                land_run0 = fx.Int32(
                    rocdl.readfirstlane(T.i32, cursor_buf[fx.Int32(2 * npes) + peer])
                )
                inv_peer = fx.Int64(w.lsa_ptr(peer, lay.off_inv))
                land0 = fx.Int32(lay.send_rows) + land_run0
                for base in range(fx.Int32(0), cnt, fx.Int32(64 * PUSH_U)):
                    ok, js, slots = [], [], []
                    for u in range_constexpr(PUSH_U):
                        j = base + fx.Int32(u * 64) + lane
                        in_run = j < cnt
                        ok.append(in_run)
                        js.append(j)
                        slots.append(idx_buf[run0 + in_run.select(j, fx.Int32(0))])
                    for u in range_constexpr(PUSH_U):
                        if ok[u]:
                            comm_ops.store_i32_global_system_monotonic(
                                inv_peer + fx.Int64(slots[u]) * fx.Int64(4),
                                land0 + js[u],
                            )
                rocdl.s_waitcnt(vmcnt=0)

                total_tiles = (
                    global_typed_ptr(num_valid, T.i32)[0] + fx.Int32(SBM - 1)
                ) // fx.Int32(SBM)
                n_chunks = (total_tiles + fx.Int32(chunk_tiles - 1)) // fx.Int32(
                    chunk_tiles
                )
                last = n_chunks - fx.Int32(1)
                for c in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                    ci = fx.Int32(c)
                    t0 = ci * fx.Int32(chunk_tiles)
                    t1 = t0 + fx.Int32(chunk_tiles)
                    t1 = (t1 < total_tiles).select(t1, total_tiles)
                    # Every (sort tile, N-block) of this chunk has stored.
                    if is_lane0:
                        comm_ops.spin_until_ge_i32_agent(
                            fx.Int64(cdone) + fx.Int64(ci) * fx.Int64(CDONE_STRIDE * 4),
                            (t1 - t0) * n_blocks,
                        )
                    comm_ops.fence_agent_acquire()
                    lo = tile_base_buf[t0 * fx.Int32(npes) + peer]
                    hi = tile_base_buf[t1 * fx.Int32(npes) + peer]
                    if (hi > lo) & is_lane0:
                        # The last chunk's put carries the completion signal.
                        if ci == last:
                            sdma_put_run(
                                sdma,
                                win,
                                peer,
                                queue,
                                layout=lay,
                                send_base=run0,
                                land_off=land_run0,
                                lo=lo,
                                hi=hi,
                                cco=cco,
                                remote_signal=True,
                            )
                        else:
                            sdma_put_run(
                                sdma,
                                win,
                                peer,
                                queue,
                                layout=lay,
                                send_base=run0,
                                land_off=land_run0,
                                lo=lo,
                                hi=hi,
                                cco=cco,
                            )
                if is_lane0:
                    t_last = last * fx.Int32(chunk_tiles)
                    t_last = (t_last > fx.Int32(0)).select(t_last, fx.Int32(0))
                    lo_l = tile_base_buf[t_last * fx.Int32(npes) + peer]
                    hi_l = tile_base_buf[total_tiles * fx.Int32(npes) + peer]
                    # A peer whose last chunk was empty (or with no chunks) still needs the signal.
                    if (n_chunks <= fx.Int32(0)) | (hi_l <= lo_l):
                        sdma_signal_peer(
                            sdma, win, peer, queue, layout=lay, rank=rank, cco=cco
                        )
                    publish_flag(w, peer, phase, layout=lay, rank=rank)

        def pf_land_off(p, lane, par):
            """Rows the ranks below me other than ``p`` send ``p`` (where my run lands in its R),
            as the plan's offsets: lane = expert, dst-major loads, one wave sum."""
            okj = lane < fx.Int32(experts)
            j = okj.select(lane, fx.Int32(0))
            bc = global_typed_ptr(out_row, T.i32)  # plan-free: bigcnt
            acc = fx.Int32(0)
            for dst in range_constexpr(rank):
                v = fx.Int32(bc[p * fx.Int32(TS) + fx.Int32(dst * experts) + j])
                acc = acc + (okj & (p != fx.Int32(dst))).select(v, fx.Int32(0))
            # dst's fanout-pair rows: one per member for every group token p sent dst
            okd = (lane < fx.Int32(rank)) & (lane != p)
            d = okd.select(lane, fx.Int32(0))
            pc0 = fx.Int32(comm_ops.load_i32_system(tile_base, d))
            pc1 = fx.Int32(comm_ops.load_i32_system(tile_base, fx.Int32(npes) + d))
            _, _, _, mem_d = pair_of(par, pc0, pc1)
            gv = fx.Int32(bc[p * fx.Int32(TS) + fx.Int32(TE) + d])
            acc = acc + okd.select(mem_d * gv, fx.Int32(0))
            incl = _wave_inclusive_scan_i32(acc, lane)
            return fx.Int32(
                rocdl.readfirstlane(
                    T.i32, fx.Int32(rocdl.ds_bpermute(T.i32, fx.Int32(63 * 4), incl))
                )
            )

        def pf_push(p, lo, hi, pv, st_p, inv_base, value0, sbase_p):
            """inv[slot] = value0 + place for every place in [lo, hi) of my run for ``p``, the
            slot read from the srcmap word of the row at that place (the inverse map).
            """
            lane = fx.Int32(fx.thread_idx.x) % fx.Int32(64)
            for base in range(lo, hi, fx.Int32(64 * PUSH_U)):
                ok, js, gs = [], [], []
                for u in range_constexpr(PUSH_U):
                    j = base + fx.Int32(u * 64) + lane
                    in_run = j < hi
                    jj = in_run.select(j, fx.Int32(0))
                    ok.append(in_run)
                    js.append(jj)
                    gs.append(pf_row_of_place(jj, p, st_p, pv))
                sms = [
                    fx.Int32(global_typed_ptr(trb, T.i32)[g // fx.Int32(SBM)])
                    + g % fx.Int32(SBM)
                    for g in gs
                ]
                words = [fx.Int32(global_typed_ptr(stids, T.i32)[sm]) for sm in sms]
                for u in range_constexpr(PUSH_U):
                    wd = words[u]
                    slot = (wd & fx.Int32(max_tok - 1)) * fx.Int32(topk) + (
                        wd.shrui(fx.Int32(24)) & fx.Int32(0xFF)
                    )
                    if ok[u]:
                        comm_ops.store_i32_global_system_monotonic(
                            inv_base + fx.Int64(slot) * fx.Int64(4), value0 + js[u]
                        )

        def comm_role_pf():
            """Plan-free comm workgroup: the plan's offsets, chunk bounds and inverse map in
            closed form from the section table; wave p < npes serves peer p (wave ``rank``
            fills this rank's own inv entries); then the overflow count and the cdone reset.
            """
            tid = fx.Int32(fx.thread_idx.x)
            lane = tid % fx.Int32(64)
            wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // fx.Int32(64)))
            is_lane0 = lane == fx.Int32(0)
            queue = fx.Int32(0)  # one SDMA channel per peer
            lay = layout
            tab_lds = lds_addr(lds.E_tab)
            ld = pf_issue()
            values, _ = tile_table(tile_end, lane, SBM=SBM, experts=experts)
            tile_table_to_lds(values, tab_lds, lane)
            pf_write(ld)
            lds_barrier()
            pf_prefix()
            lds_barrier()
            par = ld[6]
            total_tiles = (
                global_typed_ptr(num_valid, T.i32)[0] + fx.Int32(SBM - 1)
            ) // fx.Int32(SBM)
            n_chunks = (total_tiles + fx.Int32(chunk_tiles - 1)) // fx.Int32(
                chunk_tiles
            )
            # Rows past the end of S go through the P2P fallback; count them as the plan did.
            if tid == fx.Int32(0):
                over = fx.Int32(0)
                for q in range_constexpr(npes):
                    _, _, av, _, ns = pf_peer_vals(fx.Int32(q))
                    d = ns - av
                    over = over + (d > fx.Int32(0)).select(d, fx.Int32(0))
                if over > fx.Int32(0):
                    comm_ops.atomic_add_agent(fx.Int64(idx_send), over)
            if wave < fx.Int32(npes):
                p = wave
                pv = pf_peer_vals(p)
                gcol_p, _, avail_p, sbase_p, nsend_p = pv
                cnt_p = (nsend_p < avail_p).select(nsend_p, avail_p)
                land_p = pf_land_off(p, lane, par)
                # Each expert's first place for p (the inverse search), in a main-loop
                # buffer this workgroup never uses, one array per wave.
                st_p = lds_addr(lds.A_lds_cur_0) + wave * fx.Int32(64 * 4)
                enabled, a, b, _ = pf_misc()
                okj = lane < fx.Int32(experts)
                cum_l = pf_ld(
                    fx.Int32(PF_REC)
                    + (okj.select(lane, fx.Int32(0)) * fx.Int32(npes) + p)
                    * fx.Int32(PF_REC_W)
                )
                mem_l = enabled & ((lane == a) | (lane == b))
                _st_lds_i32(
                    st_p,
                    lane,
                    okj.select(
                        cum_l - mem_l.select(gcol_p, fx.Int32(0)), fx.Int32(0x7FFFFFFF)
                    ),
                )
                if p != fx.Int32(rank):
                    w = cco.Window(win)
                    sdma = cco.DevComm(dev_comm).sdma()
                    rsrc_counter = create_buffer_resource_from_addr(
                        fx.Int64(w.lsa_ptr(rank, lay.off_counter))
                    )
                    # Read-only here: the combine advances it after the call.
                    phase = fx.Int32(
                        buffer_load(rsrc_counter, 0, vec_width=1, dtype=T.i32)
                    ) + fx.Int32(1)
                    # Do not overwrite the peer's landing rows or inv before it has read the last call's.
                    if is_lane0:
                        sdma_wait_consumed(w, p, phase, layout=lay, rank=rank)
                    if const_expr(not inv_pipe):
                        # inv entries go out while the GEMM computes its first chunk; they must be
                        # visible before any put or signal of mine, hence the vmcnt wait.
                        pf_push(
                            p,
                            fx.Int32(0),
                            cnt_p,
                            pv,
                            st_p,
                            fx.Int64(w.lsa_ptr(p, lay.off_inv)),
                            fx.Int32(send_rows) + land_p,
                            sbase_p,
                        )
                        rocdl.s_waitcnt(vmcnt=0)
                    else:
                        # The entries go out between the puts instead. The peer reads them only
                        # after my flag and my signal, and every entry is complete before both:
                        # the vmcnt wait before the signal-carrying put (the put also waits
                        # before its doorbell) and the flag's release.
                        inv_p = fx.Int64(w.lsa_ptr(p, lay.off_inv))
                        land0 = fx.Int32(send_rows) + land_p
                        # Chunk 0's entries while the GEMM computes chunk 0.
                        t1_0 = (total_tiles < fx.Int32(chunk_tiles)).select(
                            total_tiles, fx.Int32(chunk_tiles)
                        )
                        pf_push(
                            p,
                            fx.Int32(0),
                            pf_tile_base(t1_0, p, total_tiles, pv),
                            pv,
                            st_p,
                            inv_p,
                            land0,
                            sbase_p,
                        )
                    last = n_chunks - fx.Int32(1)
                    for c in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                        ci = fx.Int32(c)
                        t0 = ci * fx.Int32(chunk_tiles)
                        t1 = t0 + fx.Int32(chunk_tiles)
                        t1 = (t1 < total_tiles).select(t1, total_tiles)
                        # The chunk's bounds come from LDS: ready before its spin ends.
                        lo = pf_tile_base(t0, p, total_tiles, pv)
                        hi = pf_tile_base(t1, p, total_tiles, pv)
                        # Every (sort tile, N-block) of this chunk has stored.
                        if is_lane0:
                            comm_ops.spin_until_ge_i32_agent(
                                fx.Int64(cdone)
                                + fx.Int64(ci) * fx.Int64(CDONE_STRIDE * 4),
                                (t1 - t0) * n_blocks,
                            )
                        comm_ops.fence_agent_acquire()
                        if const_expr(inv_pipe):
                            is_last = ci == last
                            if is_last:
                                # Every inv entry of this call is complete.
                                rocdl.s_waitcnt(vmcnt=0)
                        if (hi > lo) & is_lane0:
                            # The last chunk's put carries the completion signal.
                            if ci == last:
                                sdma_put_run(
                                    sdma,
                                    win,
                                    p,
                                    queue,
                                    layout=lay,
                                    send_base=sbase_p,
                                    land_off=land_p,
                                    lo=lo,
                                    hi=hi,
                                    cco=cco,
                                    remote_signal=True,
                                )
                            else:
                                sdma_put_run(
                                    sdma,
                                    win,
                                    p,
                                    queue,
                                    layout=lay,
                                    send_base=sbase_p,
                                    land_off=land_p,
                                    lo=lo,
                                    hi=hi,
                                    cco=cco,
                                )
                        if const_expr(inv_pipe):
                            # The next chunk's entries while the copy engine moves this one
                            # (none after the last chunk).
                            t2 = t1 + fx.Int32(chunk_tiles)
                            t2 = (t2 < total_tiles).select(t2, total_tiles)
                            pf_push(
                                p,
                                hi,
                                pf_tile_base(t2, p, total_tiles, pv),
                                pv,
                                st_p,
                                inv_p,
                                land0,
                                sbase_p,
                            )
                    if const_expr(inv_pipe):
                        # Before the signal-only put below (no chunk put carried the signal).
                        rocdl.s_waitcnt(vmcnt=0)
                    if is_lane0:
                        t_last = last * fx.Int32(chunk_tiles)
                        t_last = (t_last > fx.Int32(0)).select(t_last, fx.Int32(0))
                        lo_l = pf_tile_base(t_last, p, total_tiles, pv)
                        hi_l = pf_tile_base(total_tiles, p, total_tiles, pv)
                        # A peer whose last chunk was empty (or with no chunks) still needs the signal.
                        if (n_chunks <= fx.Int32(0)) | (hi_l <= lo_l):
                            sdma_signal_peer(
                                sdma, win, p, queue, layout=lay, rank=rank, cco=cco
                            )
                        publish_flag(w, p, phase, layout=lay, rank=rank)
                else:
                    # My own run stays in S; the combine finds its rows through inv (what the
                    # plan wrote). Its last readers ran before this kernel (stream order).
                    pf_push(
                        p,
                        fx.Int32(0),
                        cnt_p,
                        pv,
                        st_p,
                        fx.Int64(send_base) + fx.Int64(rel_inv),
                        sbase_p,
                        sbase_p,
                    )
            # Every comm wave has passed every chunk's spin, so every increment of this call
            # has landed: zero this call's counters for the next one (the plan zeroed them at
            # the start of each call; the kernel boundary orders these stores before them).
            fx.barrier()
            for c in range(tid, n_chunks, fx.Int32(512)):
                _st_i32(
                    fx.Int64(cdone), fx.Int32(c) * fx.Int32(CDONE_STRIDE), fx.Int32(0)
                )

        def gemm_role():
            tab_lds = lds_addr(lds.E_tab)
            lane = fx.thread_idx.x % fx.Int32(64)
            if const_expr(plan_free):
                ld = pf_issue()
            values, n_tiles = tile_table(tile_end, lane, SBM=SBM, experts=experts)
            tile_table_to_lds(values, tab_lds, lane)
            if const_expr(plan_free):
                pf_write(ld)
            lds_barrier()
            if const_expr(plan_free):
                pf_prefix()
                lds_barrier()
            for w in range(
                fx.Int32(fx.block_idx.x) - fx.Int32(1),
                n_tiles * n_blocks,
                fx.Int32(gemm_ctas),
            ):
                fx.barrier()
                process_tile(w // n_blocks, w % n_blocks, tab_lds)

        if fx.Int32(fx.block_idx.x) == fx.Int32(0):
            if const_expr(plan_free):
                comm_role_pf()
            else:
                comm_role()
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
        stids: fx.Int64,
        sweights: fx.Int64,
        trb: fx.Int64,
        p2p: fx.Int64,
        num_valid: fx.Int64,
        out_row: fx.Int64,
        send_base: fx.Int64,
        dev_comm: fx.Int64,
        win: fx.Int64,
        cdone: fx.Int64,
        tile_base: fx.Int64,
        cursor: fx.Int64,
        idx_send: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            A,
            B_T,
            A_scale,
            B_scale,
            c_m,
            tile_end,
            stids,
            sweights,
            trb,
            p2p,
            num_valid,
            out_row,
            send_base,
            dev_comm,
            win,
            cdone,
            tile_base,
            cursor,
            idx_send,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(grid=(grid_ctas, 1, 1), block=(512, 1, 1), stream=stream)

    return set_slp_vectorize_once_used(launch, False)
