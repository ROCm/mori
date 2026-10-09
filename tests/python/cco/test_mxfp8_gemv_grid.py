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
"""The skinny GEMM across its compile-time configuration space.

``test_gemm_ar_op.py`` covers ``Mxfp8GemvOp``, which runs whatever the tuned
table selects -- so it exercises six configurations out of seventy-two and says
nothing about the rest. This is the grid, and it exists because the kernel is a
different program per configuration: ``ksplit`` changes how K is partitioned and
whether the reduction runs at all, and ``waves`` x ``steps`` decides how many
masked steps a wave issues when the wave count does not divide K.

That is where the bugs were. Every masked address has to be pushed out of its
buffer's records or clamped, not just the ones that would give a wrong number:
0xFF is NaN in both ue8m0 and e4m3, and NaN times a zeroed weight is still NaN.
Leaving two operand addresses unmasked made every ``wq_b`` output at M=1 NaN and
left M=2, 3, 7 and 17 depending on what happened to sit past the tensor.

**Each configuration runs in its own process.** A FlyDSL compile failure takes
the interpreter with it, so a shared one would lose the rest of the grid.

This replaced a shell script that printed ``fail=N`` and exited 0 -- verified
with a probe rigged to fail every case, the script still reported success. A
correctness sweep that cannot fail its caller is not a check.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import pytest

#: DeepSeek-V4.1-Flash's two attention shapes, per rank at TP4. K=1280 is ten
#: 128-wide steps and K=2048 is sixteen, which is the distinction that matters:
#: no wave count divides ten, so `wq_b` exercises the masked-step path on every
#: ksplit configuration and `wo_b` exercises none of it.
SHAPES = {"wq_b": (8192, 1280), "wo_b": (5120, 2048)}

#: (waves, steps, rows, tokens). A representative slice of the 72-entry space:
#: both `rows` and both `tokens`, every `waves`, and `steps` 1/2/4 so the
#: leftover-step arithmetic is covered at each unroll.
CONFIGS = [
    (4, 1, 16, 16),
    (4, 2, 16, 32),
    (4, 4, 32, 32),
    (8, 2, 32, 32),
    (8, 4, 16, 32),
    (16, 1, 32, 16),
]

#: M values that are not tile widths. 3 and 17 are the point: rows past M read a
#: clamped token and must not reach the output, and 17 crosses from a 16-token
#: configuration to a 32-token one.
M_VALUES = [1, 2, 3, 8, 16, 17, 32]

FP8_FLOOR = 2.4e-3

requires_gpu = pytest.mark.skipif(
    not os.environ.get("MORI_TEST_GPU", "1") == "1",
    reason="needs a gfx950 GPU",
)


def _cases():
    for shape in SHAPES:
        for waves, steps, rows, tokens in CONFIGS:
            for m in M_VALUES:
                if m > tokens:
                    continue  # the token tile is also the largest M it can serve
                for ksplit in (0, 1):
                    yield shape, m, waves, steps, rows, tokens, ksplit


ALL_CASES = list(_cases())
#: The full grid is ~150 subprocess compiles. Default to every configuration at
#: the M values that broke before, and take the rest under MORI_TEST_GEMV_FULL.
QUICK = [c for c in ALL_CASES if c[1] in (1, 3, 17, 32)]
CASES = ALL_CASES if os.environ.get("MORI_TEST_GEMV_FULL") == "1" else QUICK


def _run_case(shape, m, waves, steps, rows, tokens, ksplit):
    """One configuration, in its own interpreter. Returns (rc, payload|text)."""
    p = subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            "--shape",
            shape,
            "-m",
            str(m),
            "--waves",
            str(waves),
            "--steps",
            str(steps),
            "--rows",
            str(rows),
            "--tokens",
            str(tokens),
            "--ksplit",
            str(ksplit),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON"):
            return p.returncode, json.loads(line.split(" ", 1)[1])
    return p.returncode, (p.stdout + p.stderr)[-2000:]


@requires_gpu
@pytest.mark.parametrize(
    "shape,m,waves,steps,rows,tokens,ksplit",
    CASES,
    ids=[
        f"{s}-m{m}-w{w}s{st}r{r}t{t}{'k' if ks else 'n'}"
        for s, m, w, st, r, t, ks in CASES
    ],
)
def test_gemv_config(shape, m, waves, steps, rows, tokens, ksplit):
    """Every configuration must agree with an fp32 reference, not just the tuned one."""
    rc, got = _run_case(shape, m, waves, steps, rows, tokens, ksplit)
    assert rc == 0, f"worker exited {rc}:\n{got}"
    assert isinstance(got, dict), f"worker produced no result:\n{got}"
    assert got["finite"], f"non-finite output at {got}"
    assert got["rel_l2"] < FP8_FLOOR, got


# --------------------------------------------------------------------------
# worker: one configuration, compiled and checked against fp32
# --------------------------------------------------------------------------


def _worker(args) -> int:
    import torch
    from mori.ops.gemm_ar import preshuffle_b
    from mori.ops.gemm_ar.kernels_gemv import compile_mxfp8_gemv
    import flydsl.expr as fx

    n, k = SHAPES[args.shape]
    m, bk = args.m, 32
    g = torch.Generator(device="cuda").manual_seed(1234)
    # Allocate exactly M rows, never the config's token tile: the operand
    # buffers are sized for `m_max` and a masked step that walks off the last
    # row must not reach past the allocation. At M=1 there is no next row.
    x = (torch.randn(m, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    w = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    ex = torch.randint(
        120, 123, (m, k // bk), generator=g, device="cuda", dtype=torch.int32
    )
    ew = torch.randint(
        120, 123, (n // bk, k // bk), generator=g, device="cuda", dtype=torch.int32
    )

    gemv = compile_mxfp8_gemv(
        n=n,
        k=k,
        m_max=args.tokens,
        waves=args.waves,
        steps=args.steps,
        rows=args.rows,
        tokens=args.tokens,
        ksplit=bool(args.ksplit),
    )
    out = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
    gemv(
        preshuffle_b(w).contiguous().view(torch.int32).view(-1),
        ew.to(torch.uint8).contiguous().view(torch.int32).view(-1),
        x.contiguous().view(torch.int32).view(-1),
        ex.to(torch.uint8).contiguous().view(torch.int32).view(-1),
        out.view(-1),
        m,
        n,
        stream=fx.Stream(torch.cuda.current_stream()),
    )
    torch.cuda.synchronize()

    sx, sw = torch.exp2(ex.float() - 127.0), torch.exp2(ew.float() - 127.0)
    xf, wf = x.float(), w.float()
    ref = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    for i in range(k // bk):
        ks = slice(i * bk, (i + 1) * bk)
        ref += (
            (xf[:, ks] @ wf[:, ks].T)
            * sx[:, i][:, None]
            * sw[:, i].repeat_interleave(bk)[None, :]
        )
    got = out.float()
    rel = (torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref)).item()
    print(
        "RESULT_JSON "
        + json.dumps(
            {
                "shape": args.shape,
                "n": n,
                "k": k,
                "m": m,
                "config": f"w{args.waves}s{args.steps}r{args.rows}t{args.tokens}"
                f"{'k' if args.ksplit else 'n'}",
                "rel_l2": rel,
                "finite": bool(got.isfinite().all()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--shape", choices=sorted(SHAPES))
    p.add_argument("-m", type=int)
    p.add_argument("--waves", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--rows", type=int)
    p.add_argument("--tokens", type=int)
    p.add_argument("--ksplit", type=int)
    raise SystemExit(_worker(p.parse_args()))
