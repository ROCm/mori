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
"""The mxfp8 matmul for a handful of tokens -- decode's shape, not prefill's.

`kernels_fused.py`'s GEMM is built for a full prefill batch and stops paying at
small M for two structural reasons, neither of which a tile-size knob reaches.
Its smallest tile is 256 rows by 128 columns, so at M=64 on `N=8192` the grid is
`ceildiv(64,256) * ceildiv(8192,128) = 64` workgroups on a 256-CU part -- most
of the machine idle -- and of the 256 rows each workgroup computes, 64 are real.
And it stages both operands through LDS, which buys nothing when A is 64 rows
and fits in L1 outright.

So this is a different kernel rather than a tuning of that one, and it inverts
the same instruction: **the weight is the MFMA's A operand and the tokens are
its B operand**, because the weight is what there is a lot of. A wave owns
`ROWS` columns of the output and all `TOKENS` of them, streams the weight from
global straight into registers, and never touches LDS except to reduce. The
memory traffic is then exactly the weight, once, which is the floor for this
shape -- at M=1 the arithmetic is 2 FLOP per weight byte.

Two ways to fill the machine, both compile-time (`ksplit`):

- off: one workgroup covers `waves * ROWS` output columns, each wave the whole
  of K. No reduction, no LDS, but the grid is `N / (16*AT*waves)`, which on
  `wq_b` with four waves is 128 workgroups -- still half the part.
- on: one workgroup covers `ROWS` columns and its waves split K between them,
  reducing through LDS. The grid multiplies by `waves` and each wave does
  `1/waves` of the work.

Operands are the **natural** layouts, not the GEMM's. The weight goes through
`preshuffle_b` (shared with the GEMM, so a server shuffles once), but both
scales are read as the checkpoint stores them -- `[N/32, K/32]` and
`[M, K/32]`, row-major ue8m0 bytes. The GEMM needs `preshuffle_a_scale` because
its sixteen lanes want sixteen *rows* of one K block; here the sixteen lanes of
a block group are sixteen *tokens*, M is at most 32, and the whole A scale is
under a kilobyte, so the layout cannot pay for a pass over it.

Lane mapping of `v_mfma_scale_f32_16x16x128_f8f6f4` is the one measured in
sglang's `mxfp8_gemv_gfx95.cuh` and already relied on by `_Mxfp8ScaleK`:
lane ``16*s + r`` supplies the ue8m0 scale of 32-block ``s`` of row ``r``, and
``acc[r]`` of lane ``l`` is ``D[4*(l//16) + r][l % 16]``.
"""

# NOTE: no `from __future__ import annotations` here, deliberately, for the
# reason kernels_fused.py states at the same spot: `fx.struct` reads the
# `SharedStorage` field annotations as live objects, and PEP 563 would hand it
# the string "fx.Array[fx.Float32, RED_FLOATS, 16]", whose size operand is a
# local of this factory and so cannot be resolved from the module globals.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from ._gemm_a8w8_8wave import Mfma16x16x128, ceildiv, pack_i32x4_i32x8

#: K per MFMA, and so per step.
STEP_K = 128
#: ue8m0 block size along K, on both operands.
MXFP8_BLOCK = 32
#: Rows of one MFMA tile, on both operands.
TILE = 16
#: `preshuffle_b` emits 16 rows x 64 K per 1024-byte block -- see its docstring.
SHUF_BLOCK_K = 64


class _Gemv:
    """One wave's share of the work, as loads and MFMAs over a step range.

    Split out of the kernel body only so the two `ksplit` shapes share it; it
    holds the buffer views and the lane's fixed offsets, which are every
    address in the mainloop bar the step.
    """

    def __init__(self, W, WS, X, XS, *, n, k, at, bt, m_max):
        self.k, self.at, self.bt = k, at, bt
        self.nsteps = k // STEP_K
        #: Scale rows are K/32 bytes; four consecutive blocks are one dword, and
        #: one MFMA step consumes exactly those four. So a step is one dword and
        #: the byte select is the lane's block index.
        self.k_dwords = k // STEP_K
        self.lane = fx.thread_idx.x % 64
        self.row = self.lane % TILE  # output column within a tile / token
        self.g = self.lane // TILE  # this lane's 32-block within the step

        w_bytes = n * k
        x_bytes = m_max * k
        ws_bytes = (n // MXFP8_BLOCK) * (k // MXFP8_BLOCK)
        xs_bytes = m_max * (k // MXFP8_BLOCK)
        self.w = self._div(W, w_bytes)
        self.x = self._div(X, x_bytes)
        self.ws = self._div(WS, ws_bytes)
        self.xs = self._div(XS, xs_bytes)
        #: A dword index at or past this reads outside the buffer descriptor's
        #: records, and `buffer_load` returns zero. That is how a masked step
        #: costs a `v_cndmask` instead of a branch: zero data times any scale
        #: adds zero to the accumulator.
        self.w_oob = w_bytes // 4
        self.ws_oob = ws_bytes // 4
        self.x_oob = x_bytes // 4

        self.atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        self.atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.reg_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
        self.reg_4 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        self.byte_shift = self.g * fx.Int32(8)

    @staticmethod
    def _div(t, nbytes):
        gt = fx.rocdl.make_buffer_tensor(t, max_size=False, num_records_bytes=nbytes)
        return fx.logical_divide(gt, fx.make_layout(1, 1))

    def _load1(self, div, index):
        fx.copy(self.atom_1, fx.slice(div, (None, fx.Int32(index))), self.reg_1)
        return Vec(fx.memref_load_vec(self.reg_1))[0]

    def _load4(self, div, index):
        fx.copy(self.atom_4, fx.slice(div, (None, fx.Int32(index))), self.reg_4)
        return Vec(fx.memref_load_vec(self.reg_4))

    def _byte(self, div, index):
        """One ue8m0 scale out of a row-major dword of four."""
        v = self._load1(div, index)
        return arith.andi(arith.shrui(v, self.byte_shift), fx.Int32(0xFF))

    def w_frag(self, tile, t, step_dw):
        """The 32 weight bytes of tile ``tile + t`` this lane feeds one MFMA.

        `preshuffle_b` lays a 16x64 block out as ``KLane(4) NLane(16) KPack(16)``
        so within a 1024-byte block this lane's 16 bytes sit at
        ``g*256 + row*16`` -- sixteen lanes of a block group covering 256
        contiguous bytes, which is the coalescing the layout exists for. A
        128-wide MFMA step is two such blocks, hence the second load 1024 bytes
        on.
        """
        base = (tile + t) * fx.Int32(self.k // SHUF_BLOCK_K * 256) + step_dw
        off = self.g * fx.Int32(64) + self.row * fx.Int32(4)
        lo = self._load4(self.w, base + off)
        hi = self._load4(self.w, base + fx.Int32(256) + off)
        return pack_i32x4_i32x8(lo, hi)

    def x_frag(self, tok, step_dw):
        """The 32 activation bytes of token ``tok`` for this step.

        The MFMA wants K ``[32*(g//2) + 16*(g%2), +16)`` and the same 64 on:
        both simplify to ``g*16``, so the fp8 activation is read where it lies,
        row-major, no shuffle.
        """
        base = tok * fx.Int32(self.k // 4) + step_dw + self.g * fx.Int32(4)
        lo = self._load4(self.x, base)
        hi = self._load4(self.x, base + fx.Int32(16))
        return pack_i32x4_i32x8(lo, hi)

    def w_scale_base(self, tile, t):
        """Dword index of this lane's weight-scale row. Loop-invariant."""
        n_group = ((tile + t) * fx.Int32(TILE) + self.row) // fx.Int32(MXFP8_BLOCK)
        return n_group * fx.Int32(self.k_dwords)

    def x_scale_base(self, tok):
        """Dword index of this token's activation-scale row. Loop-invariant."""
        return tok * fx.Int32(self.k_dwords)

    def scale_at(self, div, base, step):
        """The ue8m0 scale of this lane's 32-block at ``step``.

        Four consecutive blocks are one dword and one MFMA step consumes exactly
        those four, so the step *is* the dword index and the byte select is the
        lane's block within the step.
        """
        return self._byte(div, base + step)


def compile_mxfp8_gemv(
    *,
    n: int,
    k: int,
    m_max: int = 32,
    waves: int = 4,
    steps: int = 2,
    rows: int = 16,
    tokens: int = 32,
    ksplit: bool = True,
):
    """A skinny mxfp8 matmul: ``out[M, N] = X[M, K] @ W[N, K].T``, M <= ``m_max``.

    ``rows`` is the output columns one wave owns and ``tokens`` the token rows;
    both are 16 or 32, being whole MFMA tiles. ``steps`` is how many 128-wide K
    steps are loaded before any of them is multiplied, which is the only knob
    that trades registers for latency hiding.
    """
    if k % STEP_K:
        raise ValueError(f"K={k} must be a multiple of {STEP_K}")
    if n % TILE:
        raise ValueError(f"N={n} must be a multiple of {TILE}")
    if n % MXFP8_BLOCK:
        raise ValueError(f"N={n} must be a multiple of {MXFP8_BLOCK} (the B scale group)")
    if rows not in (16, 32) or tokens not in (16, 32):
        raise ValueError(f"rows/tokens must be 16 or 32, got {rows}/{tokens}")
    if tokens < m_max:
        raise ValueError(f"tokens={tokens} cannot cover m_max={m_max}")
    if waves not in (4, 8, 16):
        raise ValueError(f"waves must be 4, 8 or 16, got {waves}")

    AT = rows // TILE
    BT = tokens // TILE
    NSTEPS = k // STEP_K
    #: Steps one wave runs. With ksplit the waves divide K; without, each runs
    #: all of it and they divide N instead.
    CHUNK = ceildiv(NSTEPS, waves) if ksplit else NSTEPS
    N_ITER = ceildiv(CHUNK, steps)
    #: Whether every emitted step is a real one. It is whenever `waves` divides
    #: `nsteps` -- `wo_b` at K=2048 is 16 steps and all three wave counts divide
    #: it -- and then the out-of-range test is a compile-time constant and the
    #: mainloop carries no masking at all. `wq_b` at K=1280 is 10 steps, which
    #: none of 4, 8, 16 divides, so it pays one select a step.
    EXACT = (waves * CHUNK == NSTEPS) if ksplit else True
    #: Output columns one workgroup covers.
    WG_ROWS = rows if ksplit else rows * waves
    RED_WAVES = waves if ksplit else 1
    RED_FLOATS = RED_WAVES * AT * BT * TILE * TILE
    BLOCK = waves * 64

    _kname = (
        f"mori_mxfp8_gemv_n{n}k{k}_w{waves}s{steps}r{rows}t{tokens}"
        f"{'_ks' if ksplit else ''}"
    )

    @fx.struct
    class SharedStorage:
        red: fx.Array[fx.Float32, RED_FLOATS, 16]

    @flyc.kernel(name=_kname, known_block_size=[BLOCK, 1, 1])
    def kernel_gemv(
        W: fx.Tensor,
        WS: fx.Tensor,
        X: fx.Tensor,
        XS: fx.Tensor,
        C: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        wave = fx.thread_idx.x // 64
        gv = _Gemv(W, WS, X, XS, n=n, k=k, at=AT, bt=BT, m_max=m_max)
        lane_row, g = gv.row, gv.g

        if const_expr(ksplit):
            tile = fx.block_idx.x * fx.Int32(AT)
            step0 = wave * fx.Int32(CHUNK)
        else:
            tile = (fx.block_idx.x * fx.Int32(waves) + wave) * fx.Int32(AT)
            step0 = fx.Int32(0)

        # A token row past M reads token 0 and is simply never stored. Clamping
        # rather than branching keeps the wave converged, and the extra MFMA is
        # already being issued for the full 16-token tile whatever M is.
        toks = [
            arith.select(
                lane_row + fx.Int32(TILE * b) < c_m,
                lane_row + fx.Int32(TILE * b),
                fx.Int32(0),
            )
            for b in range_constexpr(BT)
        ]

        mfma = Mfma16x16x128(AT, BT)
        acc = [mfma.zero_value] * (AT * BT)

        # Scale rows are fixed for the whole mainloop -- only the step moves --
        # so the row arithmetic is hoisted here rather than redone each step.
        w_sc_base = [gv.w_scale_base(tile, t) for t in range_constexpr(AT)]
        x_sc_base = [gv.x_scale_base(toks[b]) for b in range_constexpr(BT)]

        for it in range_constexpr(N_ITER):
            w_frags, x_frags, w_sc, x_sc = [], [], [], []
            for s in range_constexpr(steps):
                # Two separate ways a step can be out of range, and they are not
                # the same kind of condition. Past this wave's `CHUNK` share is
                # known at compile time (the share is a constant and the loop is
                # unrolled), so the step is simply not emitted -- waves must
                # *partition* K, and one that runs a step belonging to its
                # neighbour double-counts it into the reduction. Past K itself
                # is runtime and only arises when `waves * CHUNK` overshoots
                # `nsteps`, which `EXACT` decides at compile time.
                local = it * steps + s
                if const_expr(local >= CHUNK):
                    continue
                step = step0 + fx.Int32(local)
                sc_step = step
                if const_expr(EXACT):
                    step_dw = step * fx.Int32(512)
                    x_step_dw = step * fx.Int32(STEP_K // 4)
                else:
                    # **Every** address this step forms has to be dealt with,
                    # not just the ones that would give a wrong number. Past K
                    # an index walks off the end of its row, and the operand
                    # buffers are sized for `m_max` tokens while the caller
                    # passes M of them -- so the read can be inside the
                    # descriptor's records and outside the allocation. At M=1
                    # there is no next row at all. Whatever comes back then is
                    # not merely wrong, it is *poisonous*: 0xFF is NaN in both
                    # ue8m0 and e4m3, and NaN times the zeroed weight is still
                    # NaN. Leaving the two operand addresses unmasked because
                    # "the weight is zero anyway" made every `wq_b` output at
                    # M=1 NaN, and left M=2, 3, 7 and 17 depending on what
                    # happened to be in memory past the tensor.
                    in_k = step < fx.Int32(NSTEPS)
                    step_dw = arith.select(
                        in_k, step * fx.Int32(512), fx.Int32(gv.w_oob)
                    )
                    x_step_dw = arith.select(
                        in_k, step * fx.Int32(STEP_K // 4), fx.Int32(gv.x_oob)
                    )
                    # The scales are clamped rather than pushed out, because one
                    # `v_min` covers the step where a select would be per tile
                    # -- and a real scale from the last step, against a zero
                    # weight, is just as harmless as a zero one.
                    sc_step = arith.select(in_k, step, fx.Int32(NSTEPS - 1))
                w_frags.append(
                    [gv.w_frag(tile, t, step_dw) for t in range_constexpr(AT)]
                )
                x_frags.append(
                    [gv.x_frag(toks[b], x_step_dw) for b in range_constexpr(BT)]
                )
                w_sc.append(
                    [gv.scale_at(gv.ws, w_sc_base[t], sc_step) for t in range_constexpr(AT)]
                )
                x_sc.append(
                    [gv.scale_at(gv.xs, x_sc_base[b], sc_step) for b in range_constexpr(BT)]
                )
            for s in range_constexpr(len(w_frags)):
                acc = mfma.call(
                    w_frags[s],
                    x_frags[s],
                    acc,
                    set_prio=False,
                    scale_a=w_sc[s],
                    scale_b=x_sc[s],
                )

        oob = fx.Int32(0x7FFFFFF0)
        out_atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        out_reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.BFloat16)
        gC = fx.rocdl.make_buffer_tensor(
            C, max_size=False, num_records_bytes=m_max * n * 2
        )
        c_div = fx.logical_divide(gC, fx.make_layout(1, 1))

        def store(value, tok, col):
            in_range = arith.andi(tok < c_m, col < c_n)
            idx = arith.select(in_range, tok * c_n + col, oob)
            fx.memref_store_vec(Vec.filled(1, value, fx.BFloat16), out_reg)
            fx.copy(out_atom, out_reg, fx.slice(c_div, (None, fx.Int32(idx))))

        if const_expr(ksplit):
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            for t in range_constexpr(AT):
                for b in range_constexpr(BT):
                    vec = Vec(acc[mfma.idx(t, b)])
                    for r in range_constexpr(4):
                        slot = (
                            wave * fx.Int32((AT * BT) * 256)
                            + fx.Int32((t * BT + b) * 256)
                            + (g * fx.Int32(4) + fx.Int32(r)) * fx.Int32(TILE)
                            + lane_row
                        )
                        fx.ptr_store(
                            vec[r].ir_value(), fx.add_offset(lds.red.ptr, slot)
                        )
            fx.barrier()
            # A fixed wave order, so repeated calls sum identically and a row
            # stays batch-invariant.
            # The workgroup can be larger than the reduction -- a 16-wave config
            # on a 32x16 tile is 1024 threads over 512 accumulators -- so clamp
            # the LDS index and drop the surplus threads at the store. Without
            # the clamp they read past `red` and write a live output element
            # with whatever came back.
            slots = AT * BT * 256
            per_thread = ceildiv(slots, BLOCK)
            for e in range_constexpr(per_thread):
                flat = fx.thread_idx.x + fx.Int32(e * BLOCK)
                in_lds = flat < fx.Int32(slots)
                idx = arith.select(in_lds, flat, fx.Int32(0))
                total = None
                for w in range_constexpr(RED_WAVES):
                    v = fx.ptr_load(
                        fx.add_offset(lds.red.ptr, idx + fx.Int32(w * slots))
                    )
                    v = fx.Float32(v) if not hasattr(v, "to") else v
                    total = v if total is None else total + v
                tb = flat // fx.Int32(256)
                t_i = tb // fx.Int32(BT)
                b_i = tb % fx.Int32(BT)
                i = (flat % fx.Int32(256)) // fx.Int32(TILE)
                j = flat % fx.Int32(TILE)
                col = arith.select(
                    in_lds, (tile + t_i) * fx.Int32(TILE) + i, fx.Int32(0x7FFFFFF0)
                )
                tok = j + b_i * fx.Int32(TILE)
                store(total.to(fx.BFloat16), tok, col)
        else:
            for t in range_constexpr(AT):
                col_base = (tile + fx.Int32(t)) * fx.Int32(TILE) + g * fx.Int32(4)
                for b in range_constexpr(BT):
                    vec = Vec(acc[mfma.idx(t, b)])
                    tok = lane_row + fx.Int32(TILE * b)
                    for r in range_constexpr(4):
                        store(vec[r].to(fx.BFloat16), tok, col_base + fx.Int32(r))

    @flyc.jit
    def launch_gemv(
        W: fx.Tensor,
        WS: fx.Tensor,
        X: fx.Tensor,
        XS: fx.Tensor,
        C: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_n, WG_ROWS)
        kernel_gemv(
            W,
            WS,
            X,
            XS,
            C,
            c_m,
            c_n,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": f"{BLOCK},{BLOCK}",
            },
        ).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    return launch_gemv
