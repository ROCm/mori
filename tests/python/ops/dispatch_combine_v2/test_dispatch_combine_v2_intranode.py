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
"""Pytest wrapper for the ops-v2 (FlyDSL) EP8 dispatch/combine correctness test.

The v2 op-layer test is a torchrun standalone script (``test_op.py``) driven by
env vars, so this wrapper just launches it under torchrun on 8 GPUs for a few
representative modes and asserts every ``# OP-... PASS/FAIL`` line reports PASS.
It is the pytest-collectable counterpart to the v1 ``test_dispatch_combine_intranode.py``.
"""
import os
import subprocess
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEST_OP = os.path.join(_HERE, "test_op.py")
_NPROC = 8
_SWEEP = "8,128,512"  # small token counts keep the correctness run fast

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < _NPROC,
    reason=f"v2 EP8 intranode test needs {_NPROC} GPUs",
)


def _arch():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    return ""


_IS_GFX950 = _arch().startswith("gfx95")  # fp4 (e2m1) is gfx950-only


def _run(extra_env, timeout=600):
    """Launch test_op.py under torchrun; return (returncode, merged_output)."""
    env = os.environ.copy()
    env.update({k: str(v) for k, v in extra_env.items()})
    env.setdefault("SWEEP", _SWEEP)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={_NPROC}",
        _TEST_OP,
    ]
    proc = subprocess.run(
        cmd,
        cwd=_HERE,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout


def _assert_all_pass(rc, out):
    results = [
        ln for ln in out.splitlines() if "ct=" in ln and ("PASS" in ln or "FAIL" in ln)
    ]
    assert results, f"no OP result lines found (rc={rc}); output:\n{out}"
    fails = [ln for ln in results if "FAIL" in ln]
    assert not fails, "FAIL(s):\n" + "\n".join(fails) + f"\n\nfull output:\n{out}"
    assert rc == 0, f"torchrun exited {rc}; output:\n{out}"


_skip_fp4 = pytest.mark.skipif(not _IS_GFX950, reason="fp4 is gfx950-only")

# This is a functional test, not a perf test: hidden_dim is fixed at 7168 and the
# token-count "shape" is covered cheaply by SWEEP inside a single torchrun launch.
# The parametrized matrix covers the functional axes (dtype, combine, quant, scale
# forwarding, topk, and the feature paths).


def _cases():
    # dtype x combine (non-quant)
    yield pytest.param({"DTYPE": "bf16", "COMBINE": "gather"}, id="bf16-gather")
    yield pytest.param({"DTYPE": "bf16", "COMBINE": "scatter"}, id="bf16-scatter")
    yield pytest.param({"DTYPE": "f32", "COMBINE": "gather"}, id="f32-gather")
    yield pytest.param({"DTYPE": "f32", "COMBINE": "scatter"}, id="f32-scatter")
    yield pytest.param(  # fp4 gather (tuned schedule), gfx950-only
        {"DTYPE": "fp4", "COMBINE": "gather", "TUNED": 1},
        id="fp4-gather",
        marks=_skip_fp4,
    )
    # plain fp8 token dtype (gather-only), any arch (fnuz on gfx942, OCP on gfx950)
    yield pytest.param({"DTYPE": "fp8", "COMBINE": "gather"}, id="fp8-gather")
    yield pytest.param(
        {"DTYPE": "fp8", "COMBINE": "gather", "TUNED": 1}, id="fp8-gather-tuned"
    )
    # quant paths (scatter-only, compress-on-write)
    yield pytest.param(
        {"DTYPE": "bf16", "COMBINE": "scatter", "QUANT": "fp8_direct_cast"},
        id="bf16-scatter-fp8direct",
    )
    yield pytest.param(
        {"DTYPE": "bf16", "COMBINE": "scatter", "QUANT": "fp8_blockwise"},
        id="bf16-scatter-fp8blockwise",
    )
    # blockwise with INSCALE>fp8_max so amax exceeds the fp8 range -> exercises the
    # per-block scaling branch (the plain case above never triggers it).
    yield pytest.param(
        {
            "DTYPE": "bf16",
            "COMBINE": "scatter",
            "QUANT": "fp8_blockwise",
            "INSCALE": 200,
        },
        id="bf16-scatter-fp8blockwise-scaled",
    )
    # per-token scale forwarding (v1 uses scale_dim=32), on both combine modes
    yield pytest.param(
        {"DTYPE": "bf16", "COMBINE": "gather", "SCALE_DIM": 32}, id="bf16-gather-scales"
    )
    yield pytest.param(
        {"DTYPE": "bf16", "COMBINE": "scatter", "SCALE_DIM": 32},
        id="bf16-scatter-scales",
    )
    # topk variations (topk=9 is the shared-experts-fusion / AccumNum=9 path)
    for k in (4, 9):
        yield pytest.param(
            {"DTYPE": "bf16", "COMBINE": "gather", "TOPK": k}, id=f"bf16-gather-topk{k}"
        )
    # feature paths
    yield pytest.param(
        {"DTYPE": "bf16", "COMBINE": "gather", "TUNED": 1}, id="bf16-gather-tuned"
    )
    yield pytest.param(
        {"DTYPE": "bf16", "COMBINE": "gather", "REPLAY": 1}, id="bf16-gather-replay"
    )
    yield pytest.param(
        {"DTYPE": "bf16", "COMBINE": "gather", "STDMOE": 1}, id="bf16-stdmoe"
    )


@pytest.mark.parametrize("env", list(_cases()))
def test_dispatch_combine_v2_intranode(env):
    rc, out = _run(env)
    _assert_all_pass(rc, out)
