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
"""The mxfp8 GEMM's trigger gate epilogue, ``compile_fused_gemm_scatter(gate_tail=True)``.

One launch into zeroed gates: every row band must receive exactly ``N / 256``
adds, no word outside the gates may change, and the output must be the exact
product (+-1 operands, unit ue8m0 scales, so every value is a small integer).
"""

import flydsl.expr as fx
import pytest
import torch

from mori.ops.gemm_ar import preshuffle_a_scale, preshuffle_b
from mori.ops.gemm_ar.gates import GATE_STRIDE
from mori.ops.gemm_ar.kernels_fused import compile_fused_gemm_scatter
from mori.ops.gemm_ar.layout import ArConfig
from mori.ops.gemm_ar.op import _PinnedLaunch

BLOCK = 256  # the mxfp8 GEMM's tile, both ways
NNZ = 32  # +-1 entries per row of A
UNIT_EXP = 127  # ue8m0 byte for 2^0
N, K = 7168, 2048


def _is_gfx950() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_properties(
        0
    ).gcnArchName.startswith("gfx950")


pytestmark = pytest.mark.skipif(not _is_gfx950(), reason="the mxfp8 GEMM needs gfx950")


def _compile(*, fuse=False, gate_tail=True, gate_scope="system", m=BLOCK):
    # With fuse=False the ArConfig only satisfies the signature (see Mxfp8GemmOp).
    cfg = ArConfig(world_size=2, m=m, n=N)
    return compile_fused_gemm_scatter(
        cfg,
        0,
        K=K,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        b_preshuffled=True,
        fuse=fuse,
        quant="mxfp8",
        swap_ab=True,
        permlane=True,
        lane_transpose=True,
        gate_tail=gate_tail,
        gate_scope=gate_scope,
    )


def _operands(m: int):
    g = torch.Generator(device="cuda").manual_seed(1234 + m)
    stride = K // NNZ
    rows = torch.arange(m, device="cuda")[:, None]
    cols = rows % stride + stride * torch.arange(NNZ, device="cuda")[None, :]
    signs = torch.randint(0, 2, (m, NNZ), generator=g, device="cuda").float() * 2 - 1
    a = (
        torch.zeros(m, K, device="cuda")
        .scatter_(1, cols, signs)
        .to(torch.float8_e4m3fn)
    )
    w = torch.randint(0, 2, (N, K), generator=g, device="cuda").float() * 2 - 1
    w = w.to(torch.float8_e4m3fn)
    sa = torch.full((m, K // 32), UNIT_EXP, dtype=torch.uint8, device="cuda")
    sb = torch.full((N // 32, K // 32), UNIT_EXP, dtype=torch.uint8, device="cuda")
    want = (a.float() @ w.float().t()).to(torch.bfloat16)
    return (
        a.contiguous().view(torch.int8).view(-1),
        preshuffle_b(w).contiguous().view(torch.int8).view(-1),
        preshuffle_a_scale(sa),
        sb.to(torch.int32).t().reshape(-1).contiguous(),
        want,
    )


@pytest.mark.parametrize("gate_scope", ["system", "agent"])
@pytest.mark.parametrize("m", [2048, 2304])  # 8 bands, and 9 (not a multiple of 8)
def test_gate_tail_counts_every_tile_once(m, gate_scope):
    a_i8, b_i8, sa, sb, want = _operands(m)
    bands, step = m // BLOCK, GATE_STRIDE // 8
    # A margin past the last gate, so a stray add there is caught too.
    gates = torch.zeros(bands * step + 64, dtype=torch.int64, device="cuda")
    out = torch.full((m, N), float("nan"), dtype=torch.bfloat16, device="cuda")
    launch = _PinnedLaunch(_compile(gate_scope=gate_scope))
    launch(
        a_i8,
        b_i8,
        out.view(-1),
        sa,
        sb,
        m,
        N,
        0,
        gates.data_ptr(),
        stream=fx.Stream(torch.cuda.current_stream()),
    )
    torch.cuda.synchronize()

    assert torch.equal(out, want)
    assert gates[: bands * step : step].tolist() == [N // BLOCK] * bands
    stray = torch.ones_like(gates, dtype=torch.bool)
    stray[: bands * step : step] = False
    assert int(gates[stray].count_nonzero()) == 0


def test_gate_tail_options_are_checked():
    with pytest.raises(ValueError, match="fuse=False"):
        # m = world * BLOCK, so the fused path's own row-slice check passes first.
        _compile(fuse=True, m=2 * BLOCK)
    with pytest.raises(ValueError, match="gate_scope"):
        _compile(gate_scope="device")
