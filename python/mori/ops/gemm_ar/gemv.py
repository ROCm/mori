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
"""The public entry to the skinny mxfp8 matmul -- decode's M, not prefill's.

Same relationship to `kernels_gemv.py` as `Mxfp8GemmOp` has to
`kernels_fused.py`: pick the compile-time configuration, hold the compiled
launchers, and check the operands before the launch boundary reinterprets them
as bytes.

The handover between this and `Mxfp8GemmOp` is by M and is the caller's to make.
The GEMM's floor is where its 256-row tile stops being launch-starved; this
kernel's ceiling is 32 tokens, because a token is an MFMA row and two tiles of
them is where the register file runs out. They do not meet: M in (32, 1024) is
served well by neither, which is what `mxfp8_native_blockscaled_linear`'s
`dot_scaled` route is for.
"""

from __future__ import annotations

import flydsl.expr as fx
import torch

from .kernels_gemv import MXFP8_BLOCK, STEP_K, TILE, compile_mxfp8_gemv
from .op import FP8_DTYPES, _PinnedLaunch

#: Most tokens a config can carry: two 16-row MFMA tiles on the B operand.
MAX_TOKENS = 32

#: Token counts a config is tuned at, each covering M up to its own value.
M_BUCKETS = (1, 2, 4, 8, 16, 32)

#: Tuned per (N, K, M bucket) by `benchmark/cco/flydsl/gemm_ar/sweep.py gemv-tune`,
#: on the cold column -- a decode step reads each layer's weight once, so a
#: hot-loop number is measuring the LLC rather than the kernel.
#:
#: Keys are `(N, K, bucket)`; anything absent falls back to `_HEURISTIC`.
#:
#: Against sglang's `mxfp8_gemv` on the same fp8 operands -- the two agree bit
#: for bit, so this is the same arithmetic done faster. Cold us, MI355X:
#:
#:     M     wq_b 8192x1280        wo_b 5120x2048
#:           sglang  mori          sglang  mori
#:      1     4.19   3.91  -6.8%    4.29   3.93  -8.4%
#:      2     4.14   3.92  -5.4%    4.52   3.99 -11.8%
#:      4     4.20   4.07  -3.0%    4.54   4.17  -8.1%
#:      8     4.29   4.11  -4.1%    4.76   4.27 -10.1%
#:     16     4.57   4.54  -0.7%    5.28   4.94  -6.4%
#:     32     5.58   5.27  -5.7%    6.97   6.32  -9.3%
#:
#: `wq_b` is the thinner margin and it is the shape, not the tuning: K=1280 is
#: ten 128-wide steps and none of 4, 8, 16 waves divides ten, so a K-split wave
#: always issues a step it masks off and throws away. `wo_b`'s K=2048 is sixteen
#: and every wave count divides it -- `EXACT` in `kernels_gemv.py` -- which is
#: where its flat 6-12% comes from. M=16 on `wq_b` is a tie within run-to-run
#: noise and is reported as one.
#:
#: **Both rows are tuned shapes, and the margin does not survive without that.**
#: Measured across all twelve of the checkpoint's shapes, everything that falls
#: back to `_HEURISTIC` lands inside +-2% -- run-to-run noise -- except `wo_a`,
#: which loses 12-16% at M >= 8 on both TP degrees, and `wq_b` at TP8, which
#: loses 14% at M=32. `wo_a` is the only shape here with N <= 2048 *and*
#: K >= 4096, so the heuristic's 4-wave 16x16 tile has both few N tiles to
#: spread over and a long K to walk. Sweep a shape that matters
#: (`sweep.py gemv-tune`, about two minutes a bucket) rather than assuming it
#: inherits this.
#:
#: This is also the fp8-input comparison. sglang quantises a bf16 activation
#: inside its kernel where mori needs a separate ~2us pass, so on these two
#: shapes a bf16 caller is better off with sglang; on shapes where sglang's
#: fusion has to redo that work per workgroup, it is not. The caller decides --
#: see `mori_mxfp8_gemm.py`'s `_GEMV_MAX_M`.
_TUNED: dict[tuple[int, int, int], dict] = {}


def _tune(n: int, k: int, **by_bucket: str) -> None:
    import re

    for bucket, key in by_bucket.items():
        g = re.fullmatch(r"w(\d+)s(\d+)r(\d+)t(\d+)([kn])", key)
        _TUNED[(n, k, int(bucket[1:]))] = {
            "waves": int(g[1]),
            "steps": int(g[2]),
            "rows": int(g[3]),
            "tokens": int(g[4]),
            "ksplit": g[5] == "k",
        }


# V4.1-Flash's two attention shapes, per rank at TP4.
_tune(
    8192,
    1280,  # wq_b, ColumnParallel
    m1="w4s4r16t16k",
    m2="w4s4r16t16k",
    m4="w4s4r16t16k",
    m8="w4s4r16t16k",
    m16="w4s4r32t16k",
    m32="w16s2r32t32k",
)
_tune(
    5120,
    2048,  # wo_b, RowParallel
    m1="w16s1r32t16k",
    m2="w16s2r32t16k",
    m4="w16s4r32t16k",
    m8="w16s4r32t16k",
    m16="w16s2r32t16k",
    m32="w16s4r32t32k",
)

#: What to run for an untuned shape. Four waves splitting K, one 16x16 tile per
#: wave: the sweep's best at every bucket it has covered so far, and the only
#: shape of config that does not either starve the grid (no ksplit) or carry
#: token tiles nothing fills (tokens=32 below M=17).
_HEURISTIC = {"waves": 4, "steps": 2, "rows": 16, "tokens": 16, "ksplit": True}


def m_bucket(m: int) -> int:
    """The smallest tuned bucket that covers ``m``."""
    for b in M_BUCKETS:
        if m <= b:
            return b
    raise ValueError(f"M={m} exceeds the skinny kernel's {MAX_TOKENS} tokens")


def supports_gemv(n: int, k: int) -> bool:
    """Whether this shape is expressible. Profitability is the caller's call."""
    return k % STEP_K == 0 and n % MXFP8_BLOCK == 0 and n % TILE == 0


def select_config(n: int, k: int, m: int) -> dict:
    cfg = dict(_TUNED.get((n, k, m_bucket(m)), _HEURISTIC))
    # A config's token tile is also the most tokens it can serve, so a bucket
    # above 16 has to widen it whatever the table says.
    if cfg["tokens"] < m:
        cfg["tokens"] = MAX_TOKENS
    return cfg


class Mxfp8GemvOp:
    """A compiled skinny mxfp8 matmul for one ``(N, K)``, M up to 32.

    ::

        op = Mxfp8GemvOp(n=8192, k=1280)
        out = op(x_fp8, w_preshuffled, x_scale, w_scale)   # [M, N] bf16

    ``w_preshuffled`` is a weight through :func:`~mori.ops.gemm_ar.preshuffle_b`,
    shared with :class:`~mori.ops.gemm_ar.Mxfp8GemmOp` so a server shuffles once
    and both paths read it. **The scales are not shared**: both go in as the
    checkpoint stores them, row-major ue8m0 bytes, ``[M, K/32]`` and
    ``[N/32, K/32]``. The GEMM's `preshuffle_a_scale` exists to coalesce sixteen
    lanes reading sixteen *rows* of one K block; here those lanes are sixteen
    tokens, M is at most 32, and the whole A scale is well under a kilobyte.

    One compiled kernel per M bucket, at most six. Unlike the GEMM, M is *not*
    a pure runtime argument -- the token tile is a compile-time tile width -- but
    the buckets are a fixed ladder, so the compiles are bounded and a server
    cannot drive an unbounded cache with its batch size.
    """

    def __init__(self, *, n: int, k: int):
        if not supports_gemv(n, k):
            raise ValueError(
                f"unsupported shape N={n} K={k}: K must be a multiple of "
                f"{STEP_K} and N of {MXFP8_BLOCK}"
            )
        self.n, self.k = n, k
        self._launch: dict[int, _PinnedLaunch] = {}

    def _compiled(self, m: int) -> _PinnedLaunch:
        bucket = m_bucket(m)
        hit = self._launch.get(bucket)
        if hit is not None:
            return hit
        cfg = select_config(self.n, self.k, m)
        hit = _PinnedLaunch(
            compile_mxfp8_gemv(n=self.n, k=self.k, m_max=cfg["tokens"], **cfg)
        )
        self._launch[bucket] = hit
        return hit

    def _check(self, x_fp8, w, x_scale, w_scale, m: int) -> None:
        """Reject what the launch would otherwise reinterpret as raw bytes."""
        for name, t, shape in (
            ("x_fp8", x_fp8, (m, self.k)),
            ("w_preshuffled", w, (self.n, self.k)),
        ):
            if t.dim() != 2 or tuple(t.shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(t.shape)}")
            if t.dtype not in FP8_DTYPES:
                raise ValueError(
                    f"{name} must be one of {[str(d) for d in FP8_DTYPES]}, got "
                    f"{t.dtype}; the launch reads it as bytes, so a wider dtype "
                    f"runs and returns nonsense rather than failing"
                )
            if t.device.type != "cuda":
                raise ValueError(f"{name} must be on a GPU, got {t.device}")
        for name, t, shape in (
            ("x_scale", x_scale, (m, self.k // MXFP8_BLOCK)),
            ("w_scale", w_scale, (self.n // MXFP8_BLOCK, self.k // MXFP8_BLOCK)),
        ):
            if t.dtype != torch.uint8:
                raise ValueError(f"{name} must be torch.uint8 ue8m0, got {t.dtype}")
            if t.dim() != 2 or tuple(t.shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(t.shape)}")
            if t.device.type != "cuda":
                raise ValueError(f"{name} must be on a GPU, got {t.device}")

    def __call__(
        self,
        x_fp8: torch.Tensor,
        w_preshuffled: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x_fp8 @ w_preshuffled.T`` with 32-wide ue8m0 scales, into bf16.

        No padding and none needed: M is a runtime argument, the token tile
        masks its own stores, and a token row past M is clamped to row 0 rather
        than read out of bounds.
        """
        m = x_fp8.shape[0]
        if m > MAX_TOKENS:
            raise ValueError(
                f"M={m} exceeds the skinny kernel's {MAX_TOKENS} tokens; use "
                f"Mxfp8GemmOp above it"
            )
        self._check(x_fp8, w_preshuffled, x_scale, w_scale, m)
        if out is None:
            out = torch.empty((m, self.n), dtype=torch.bfloat16, device=x_fp8.device)
        elif tuple(out.shape) != (m, self.n) or out.dtype != torch.bfloat16:
            raise ValueError(
                f"out must be {(m, self.n)} bfloat16, got {tuple(out.shape)} "
                f"{out.dtype}"
            )
        self._compiled(m)(
            w_preshuffled.contiguous().view(torch.int32).view(-1),
            w_scale.contiguous().view(torch.int32).view(-1),
            x_fp8.contiguous().view(torch.int32).view(-1),
            x_scale.contiguous().view(torch.int32).view(-1),
            out.view(-1),
            m,
            self.n,
            stream=fx.Stream(torch.cuda.current_stream()),
        )
        return out
