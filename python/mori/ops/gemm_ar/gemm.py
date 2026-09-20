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
"""The mxfp8 GEMM on its own, with no collective attached.

`GemmAllReduceOp` compiles this same kernel and then fuses a scatter into its
epilogue. Plenty of callers want only the GEMM: a `ColumnParallelLinear` has no
all-reduce to overlap with at all, and a `RowParallelLinear` below the fusion's
profitability floor still wants the multiply. Until now the only way to reach it
was `compile_fused_gemm_scatter(..., fuse=False)`, whose signature asks for an
`ArConfig`, a rank and two window handles -- every one of them an all-reduce
concept that a plain GEMM has no answer for.

Measured against the two routes SGLang has for the same operands, bf16 in and
bf16 out including the quantisation, at `N=5120 K=2048 M=16384` on MI355X:

    bf16 (fake_quant + hipBLASLt)   281.3 us
    sglang mxfp8 (tl.dot_scaled)    274.9
    this                            196.0     -30.3%

**M only has to be a multiple of 64 here**, not of `BLOCK_M`: the grid is
`ceildiv(M, BLOCK_M)` and the tail block masks its stores. 64 is the packed
scale's group -- a lane's four M tiles are four 16-row tiles -- and
`preshuffle_a_scale` enforces it. Anything else is zero-extended internally,
which costs at most 63 rows of GEMM. That is a far looser constraint than the
fused op's `world_size * BLOCK_M`, where a whole row band has to belong to one
destination.
"""

from __future__ import annotations

import flydsl.expr as fx
import torch

from .kernels_fused import BLOCK_K, compile_fused_gemm_scatter
from .layout import ArConfig
from .op import (
    DEFAULT_BLOCK_N,
    FP8_DTYPES,
    MIN_K,
    MXFP8_BLOCK,
    MXFP8_BLOCK_M,
    _flatten_mxfp8_a_scale,
    _PinnedLaunch,
    _tile_constraints,
)

#: Rows one packed A-scale group spans: four M tiles of sixteen rows. M is
#: zero-extended to a multiple of this, not to BLOCK_M.
SCALE_GROUP_M = 64

#: Workgroups the 256-wide tile's grid needs before it beats the 128-wide one.
#:
#: The narrow tile exists to double the grid when the wide one is launch-starved,
#: so what decides between them is the wide grid's *size*, not M -- and the grid
#: is `ceildiv(M, BLOCK_M) * (N / BLOCK_N)`, which depends on N as much as on M.
#: An earlier version of this was a bare `M < 2048`, measured on an M grid that
#: jumped 1024 -> 2048 and so never looked between them. It is wrong on both
#: shapes. Cold, GEMM only:
#:
#:     M      wq_b 8192x1280        wo_b 5120x2048
#:            BN=256  BN=128        BN=256  BN=128
#:     1024    21.8    20.9 narrow   27.6    25.2 narrow
#:     1280    22.4    33.2 WIDE     28.1    25.7 narrow
#:     1536    23.4    33.8 WIDE     28.4    26.8 narrow
#:     1792    24.1    34.7 WIDE     28.5    44.0 WIDE
#:     2048    25.2    36.2 WIDE     29.9    44.7 WIDE
#:
#: wq_b turns over between 1024 and 1280, wo_b between 1536 and 1792 -- one
#: constant cannot express that, and at M=1280 the old one cost wq_b 48%. In
#: wide-grid workgroups those four transitions are 128 -> 160 and 120 -> 140, so
#: this threshold sits at 140 and fits all ten rows. Fitted to two shapes; treat
#: it as measured rather than derived, and re-measure if a third shape appears.
_WIDE_TILE_MIN_GRID = 140

#: Kept for callers that ask "is this M small": the largest M at which *some*
#: shape still prefers the narrow tile.
#:
#: The wide tile is launch-starved at small M. At M=64 with N=8192 its grid is
#: ceildiv(64,256) * ceildiv(8192,256) = 32 workgroups on a 256-CU part, so most
#: of the GPU is idle and the time is flat from M=64 to M=512 -- it is doing one
#: tile-row of work either way. Halving BLOCK_N doubles the grid. GEMM only,
#: both of V4.1-Flash's attention shapes, cold (weights rotated past the LLC,
#: which is what a forward pass does) over hot:
#:
#:     M      wq_b 8192x1280        wo_b 5120x2048
#:            BN=256  BN=128        BN=256  BN=128
#:       64    19.3    16.4 -15.1%   25.6    21.7 -15.5%
#:      256    20.3    18.9  -6.8%   26.1    24.1  -7.7%
#:     1024    21.8    20.9  -3.9%   27.6    25.2  -8.7%
#:     2048    25.2    36.2 +43.6%   29.9    44.7 +49.6%
#:    16384   167.2   239.1 +43.0%  148.6   212.0 +42.7%
#:
#: An earlier table here read 23.7 / 19.9 at M=64 and was measured with one call
#: per CUDA-graph capture, which on this box has a 13.4us replay floor -- so the
#: small-M rows were mostly harness. See
#: `benchmark/cco/flydsl/gemm_ar/timing.py`.
#:
#: Above the crossover the wide tile wins by more than the narrow one ever wins
#: below, because it is also the one that keeps `lane_transpose` -- that store
#: pairs exactly two N-tiles, so it needs BLOCK_N=256 and the narrow tile gives
#: it up.
NARROW_N_BELOW_M = 2048
#: What the narrow tile costs to get: its store cannot use the permlane lane
#: transpose, so this is only worth it where the grid gain is bigger.
NARROW_BLOCK_N = 128


def _gemm_shape_constraint(n: int, k: int, block_n: int) -> str | None:
    """Why this shape cannot be compiled, or None."""
    if k % BLOCK_K:
        return f"K={k} must be a multiple of {BLOCK_K} (the scaled MFMA's K step)"
    if k < MIN_K:
        return (
            f"K={k} is below the minimum {MIN_K}: the mainloop prefetches a "
            f"second K block and runs two tail steps, so K/{BLOCK_K} must be >= 2"
        )
    if n % block_n:
        return f"N={n} must be a multiple of block_n={block_n}"
    if n % MXFP8_BLOCK:
        return f"N={n} must be a multiple of {MXFP8_BLOCK} (the B scale group)"
    return None


def supports_gemm(n: int, k: int, *, block_n: int = DEFAULT_BLOCK_N) -> bool:
    """Whether this shape is *expressible*, which is not whether it is faster.

    M is deliberately not an argument: any M is servable, because one below a
    multiple of 64 is zero-extended. Profitability is the caller's call -- at
    small M a GEMV-shaped kernel will win, and this says nothing about that.
    """
    return (
        _tile_constraints(MXFP8_BLOCK_M, block_n) is None
        and _gemm_shape_constraint(n, k, block_n) is None
        # Small M runs the narrow tile, so N has to divide by that too.
        and _gemm_shape_constraint(n, k, NARROW_BLOCK_N) is None
    )


class Mxfp8GemmOp:
    """A compiled mxfp8 GEMM for one ``(N, K)``, serving every M.

    ::

        op = Mxfp8GemmOp(n=8192, k=1280)
        out = op(a_fp8, b_preshuffled, a_scale, b_scale)   # [M, N] bf16

    ``b_preshuffled`` is a weight through :func:`~mori.ops.gemm_ar.preshuffle_b`
    and ``a_scale`` is through :func:`~mori.ops.gemm_ar.preshuffle_a_scale`;
    ``b_scale`` is the ``[N/32, K/32]`` ue8m0 bytes K-block major and widened to
    int32, flat. Those are the same operands ``GemmAllReduceOp`` takes, and for
    the same reasons -- see its docstring.

    No communicator and no symmetric window: the output is an ordinary tensor,
    allocated here or supplied by the caller.
    """

    def __init__(
        self,
        *,
        n: int,
        k: int,
        block_n: int = DEFAULT_BLOCK_N,
    ):
        why = _tile_constraints(MXFP8_BLOCK_M, block_n)
        if why is not None:
            raise ValueError(f"unsupported tile: {why}")
        why = _gemm_shape_constraint(n, k, block_n)
        if why is not None:
            raise ValueError(f"unsupported shape: {why}")
        self.n, self.k = n, k
        self.block_m, self.block_n = MXFP8_BLOCK_M, block_n
        self._launch: dict[bool, _PinnedLaunch] = {}
        self._pad_in: torch.Tensor | None = None

    def padded_m(self, m: int) -> int:
        """``m`` rounded up to a whole packed A-scale group."""
        return (m + SCALE_GROUP_M - 1) // SCALE_GROUP_M * SCALE_GROUP_M

    def wants_narrow_n(self, m: int) -> bool:
        """Whether this M runs the 128-wide N tile. See `_WIDE_TILE_MIN_GRID`."""
        wide_grid = -(-m // self.block_m) * (self.n // self.block_n)
        return wide_grid < _WIDE_TILE_MIN_GRID

    def _compiled(self, m: int) -> _PinnedLaunch:
        """The kernel for this M -- the narrow-N tile below the crossover.

        **Not a per-M cache**, unlike `GemmAllReduceOp`: there are exactly two
        kernels ever, chosen by `NARROW_N_BELOW_M`. `c_m` is a runtime argument,
        the launch computes its own grid as
        `ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)`, and the tail block
        masks -- verified by compiling at M=4096 and calling at 64, 1024, 7040,
        14080 and 16384, all exact.

        That is not a micro-optimisation. A per-M cache looks harmless until a
        server drives it: every prefill batch has a different token count, so
        each one paid a fresh multi-second FlyDSL compile, and a bounded cache
        then made the path decline for good once it filled.
        """
        narrow = self.wants_narrow_n(m)
        hit = self._launch.get(narrow)
        if hit is not None:
            return hit
        block_n = NARROW_BLOCK_N if narrow else self.block_n
        # permlane's store pairs exactly two N-tiles, so it needs BLOCK_N=256.
        permlane = not narrow
        # A throwaway ArConfig purely to satisfy compile_fused_gemm_scatter's
        # signature. With fuse=False the emitted kernel never touches the
        # window -- no counters, no puts, no peer addresses -- and nothing
        # downstream reads cfg.m, which is why any M compiles the same kernel.
        # world_size=2 is the smallest ArConfig.validate accepts; nothing here
        # is distributed, and the window handles go in as 0 for the same reason.
        cfg = ArConfig(world_size=2, m=self.block_m, n=self.n)
        hit = _PinnedLaunch(
            compile_fused_gemm_scatter(
                cfg,
                0,
                K=self.k,
                BLOCK_M=self.block_m,
                BLOCK_N=block_n,
                b_preshuffled=True,
                fuse=False,
                quant="mxfp8",
                swap_ab=True,
                permlane=permlane,
                lane_transpose=permlane,
            )
        )
        self._launch[narrow] = hit
        return hit

    def pad_rows(self, x: torch.Tensor, m_pad: int) -> torch.Tensor:
        """Zero-extend ``x`` to ``m_pad`` rows, in a buffer reused across calls.

        A GEMM row depends only on the same input row, so the added rows produce
        zeros the caller slices off. Pad *before* quantising, so the A scale is
        built over the padded M and its packed layout needs no repair.

        The result views a buffer this op reuses; a later call overwrites it.
        """
        if x.dim() != 2:
            raise ValueError(f"expected a 2-D [M, K] tensor, got {tuple(x.shape)}")
        m, k = x.shape
        if m_pad < m:
            raise ValueError(f"m_pad={m_pad} is smaller than the input's M={m}")
        buf = self._pad_in
        if (
            buf is None
            or buf.shape[1] != k
            or buf.shape[0] < m_pad
            or buf.dtype != x.dtype
            or buf.device != x.device
        ):
            buf = torch.empty((m_pad, k), dtype=x.dtype, device=x.device)
            self._pad_in = buf
        out = buf[:m_pad]
        out[:m].copy_(x)
        out[m:].zero_()
        return out

    def _check_operands(self, a_fp8, b_preshuffled, a_scale, b_scale, m: int) -> None:
        """Reject what the launch boundary would otherwise reinterpret as bytes.

        ``a_fp8``/``b_preshuffled`` go in as int8 and the scales as dwords, so a
        tensor of the right shape in the wrong dtype reaches the kernel and comes
        back finite and wrong. All metadata, no device work.
        """
        for name, t, shape in (
            ("a_fp8", a_fp8, (m, self.k)),
            ("b_preshuffled", b_preshuffled, (self.n, self.k)),
        ):
            if t.dim() != 2 or tuple(t.shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(t.shape)}")
            if t.dtype not in FP8_DTYPES:
                raise ValueError(
                    f"{name} must be one of {[str(d) for d in FP8_DTYPES]}, got "
                    f"{t.dtype}; the launch reinterprets it as raw bytes, so a "
                    f"wider dtype runs and returns nonsense rather than failing"
                )
            if t.device.type != "cuda":
                raise ValueError(f"{name} must be on a GPU, got {t.device}")
        for name, t, numel in (
            ("a_scale", a_scale, m * self.k // (4 * MXFP8_BLOCK)),
            ("b_scale", b_scale, (self.n // MXFP8_BLOCK) * (self.k // MXFP8_BLOCK)),
        ):
            if t.dtype != torch.int32:
                raise ValueError(f"{name} must be torch.int32, got {t.dtype}")
            if t.numel() != numel:
                raise ValueError(f"{name} must have {numel} elements, got {t.numel()}")
            if t.device.type != "cuda":
                raise ValueError(f"{name} must be on a GPU, got {t.device}")

    def __call__(
        self,
        a_fp8: torch.Tensor,
        b_preshuffled: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``a_fp8 @ b_preshuffled.T`` with 32-wide ue8m0 scales, into bf16.

        ``M`` must already be a multiple of 64 and the operands must agree with
        it -- this does not pad, because the A scale is built over the padded M
        and repairing its packed layout afterwards is not a reshape. Use
        :meth:`padded_m` and :meth:`pad_rows` on the *unquantised* input.
        """
        m = a_fp8.shape[0]
        if m % SCALE_GROUP_M:
            raise ValueError(
                f"M={m} must be a multiple of {SCALE_GROUP_M} (the packed A "
                f"scale's group); pad the bf16 input with pad_rows() before "
                f"quantising, not the fp8 afterwards"
            )
        self._check_operands(a_fp8, b_preshuffled, a_scale, b_scale, m)
        if out is None:
            out = torch.empty(
                (m, self.n), dtype=torch.bfloat16, device=a_fp8.device
            )
        elif tuple(out.shape) != (m, self.n) or out.dtype != torch.bfloat16:
            raise ValueError(
                f"out must be {(m, self.n)} bfloat16, got {tuple(out.shape)} "
                f"{out.dtype}"
            )
        self._compiled(m)(
            a_fp8.contiguous().view(torch.int8).view(-1),
            b_preshuffled.contiguous().view(torch.int8).view(-1),
            out.view(-1),
            _flatten_mxfp8_a_scale(a_scale, m, self.k),
            b_scale.reshape(-1),
            m,
            self.n,
            0,  # dev_comm: unused with fuse=False
            0,  # win
            stream=fx.Stream(torch.cuda.current_stream()),
        )
        return out
