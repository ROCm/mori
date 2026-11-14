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
"""Pieces of the 8-wave 256x256 A8W4 GEMM (MFMA scaled fp8 x fp4, the e8m0 scale readers,
the per-expert tile table) and GEMM2's epilogue quantisation to fp8 1x32 rows.

Copied from this work's GEMM1 (mori example 13, ROCm/mori#904) and ROCm/aiter#7196's 8-wave
GEMM2; each definition carries a ``# from`` comment naming its source module.
"""

from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr
from flydsl.expr import range_constexpr
from flydsl.expr import rocdl
from flydsl.expr.typing import Float4E2M1FN
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
import flydsl.expr as fx

from flyutil import (  # noqa: F401
    _fabs_f32,
    _wave_inclusive_scan_i32,
    _wave_reduce_max_i32,
    buffer_load,
    create_buffer_resource_from_addr,
    global_typed_ptr,
    lds_typed_ptr,
)


# from aiter.ops.flydsl.kernels.mega_moe.g1_8w_parts
BLOCK_M = 256


# from aiter.ops.flydsl.kernels.mega_moe.g1_8w_parts
BLOCK_N = 256


# from aiter.ops.flydsl.kernels.mega_moe.g1_8w_parts
BLOCK_K = 128


# from aiter.ops.flydsl.kernels.mega_moe.g1_8w_parts
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


# from aiter.ops.flydsl.kernels.mega_moe.g1_8w_parts
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


# from aiter.ops.flydsl.kernels.mega_moe.g1_8w_parts
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


# from aiter.ops.flydsl.kernels.mega_moe.g1_8w_parts
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


# from aiter.ops.flydsl.kernels.mega_moe.g1_8w_parts
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


# from aiter.ops.flydsl.kernels.mega_moe.g1_8w_parts
def tile_table_to_lds(values, tab_lds, lane_id):
    """Park the per-lane table in LDS (every wave writes the same values), so the
    persistent loop does not keep it in VGPRs."""
    p = lds_typed_ptr(tab_lds, T.i32)
    for i, v in enumerate(values):
        p[fx.Int32(i * 64) + lane_id] = v


# from aiter.ops.flydsl.kernels.mega_moe.g1_8w_parts (expert search reworked)
def tile_find(tab_lds, j, lane_id, *, SBM, experts):
    """(row_base, expert, valid_rows) of 256-row tile ``j`` -- LDS and wave ops only."""
    p = lds_typed_ptr(tab_lds, T.i32)
    prev_l, seg_l, tiles_l, cum_l = [
        fx.Int32(p[fx.Int32(i * 64) + lane_id]) for i in range(4)
    ]
    ok = lane_id < fx.Int32(experts)
    # The cumulative ends rise with the lane, so the experts that end at or before ``j``
    # are a lane prefix: their count is the expert of ``j``. One ballot and a popcount,
    # where a butterfly max takes six LDS permutes and six lane-address VGPRs (which the
    # persistent loop hoisted and spilled, then reloaded one by one per item).
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


# from aiter.ops.flydsl.kernels.mega_moe.g1_8w_parts
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


# from aiter.ops.flydsl.kernels.mega_moe.g1_8w_parts
def lds_barrier():
    """Workgroup barrier for LDS only: unlike ``fx.barrier`` it never waits on vmcnt."""
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
        constraints="",
        has_side_effects=True,
    )


# from aiter.ops.flydsl.kernels.mega_moe.gemm2_8wave
def e8m0_of(amax):
    """The fp8 1x32 block exponent of a group's absolute max (the default Stage2's rule)."""
    working_bits = (amax * fx.Int32(0x3B124925).bitcast(fx.Float32)).bitcast(fx.Int32)
    mantissa = working_bits & fx.Int32(0x7FFFFF)
    biased_exp = (working_bits >> fx.Int32(23)) & fx.Int32(0xFF)
    e8m0 = (mantissa != fx.Int32(0)).select(biased_exp + fx.Int32(1), biased_exp)
    e8m0 = (e8m0 > fx.Int32(0xFF)).select(fx.Int32(0xFF), e8m0)
    return (amax.bitcast(fx.Int32) == fx.Int32(0)).select(fx.Int32(0), e8m0)


# from aiter.ops.flydsl.kernels.mega_moe.gemm2_8wave
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


# from aiter.ops.flydsl.kernels.mega_moe.gemm2_8wave
SEND_CM = 1 | 16

fabs_f32 = _fabs_f32
