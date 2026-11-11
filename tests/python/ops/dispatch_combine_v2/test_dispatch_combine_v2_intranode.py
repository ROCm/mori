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
env vars, so this wrapper launches it under torchrun on 8 GPUs for a few
representative modes and asserts every ``# OP-... PASS/FAIL`` line reports PASS.
All the modes run in one launch (``test_op.py``'s CASES), one test per mode.
It is the pytest-collectable counterpart to the v1 ``test_dispatch_combine_intranode.py``.
"""
import json
import os
import re
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
    """(id, env) of every mode."""
    # dtype x combine (non-quant)
    yield "bf16-gather", {"DTYPE": "bf16", "COMBINE": "gather"}
    yield "bf16-scatter", {"DTYPE": "bf16", "COMBINE": "scatter"}
    yield "f32-gather", {"DTYPE": "f32", "COMBINE": "gather"}
    yield "f32-scatter", {"DTYPE": "f32", "COMBINE": "scatter"}
    # fp4 gather (tuned schedule), gfx950-only
    yield "fp4-gather", {"DTYPE": "fp4", "COMBINE": "gather", "TUNED": 1}
    # plain fp8 token dtype (gather-only), any arch (fnuz on gfx942, OCP on gfx950)
    yield "fp8-gather", {"DTYPE": "fp8", "COMBINE": "gather"}
    yield "fp8-gather-tuned", {"DTYPE": "fp8", "COMBINE": "gather", "TUNED": 1}
    # quant paths (scatter-only, compress-on-write)
    yield "bf16-scatter-fp8direct", {
        "DTYPE": "bf16",
        "COMBINE": "scatter",
        "QUANT": "fp8_direct_cast",
    }
    yield "bf16-scatter-fp8blockwise", {
        "DTYPE": "bf16",
        "COMBINE": "scatter",
        "QUANT": "fp8_blockwise",
    }
    # blockwise with INSCALE>fp8_max so amax exceeds the fp8 range -> exercises the
    # per-block scaling branch (the plain case above never triggers it).
    yield "bf16-scatter-fp8blockwise-scaled", {
        "DTYPE": "bf16",
        "COMBINE": "scatter",
        "QUANT": "fp8_blockwise",
        "INSCALE": 200,
    }
    # per-token scale forwarding (v1 uses scale_dim=32), on both combine modes
    yield "bf16-gather-scales", {"DTYPE": "bf16", "COMBINE": "gather", "SCALE_DIM": 32}
    yield "bf16-scatter-scales", {
        "DTYPE": "bf16",
        "COMBINE": "scatter",
        "SCALE_DIM": 32,
    }
    # topk variations (topk=9 is the shared-experts-fusion / AccumNum=9 path)
    for k in (4, 9):
        yield f"bf16-gather-topk{k}", {"DTYPE": "bf16", "COMBINE": "gather", "TOPK": k}
    # feature paths
    yield "bf16-gather-tuned", {"DTYPE": "bf16", "COMBINE": "gather", "TUNED": 1}
    yield "bf16-gather-replay", {"DTYPE": "bf16", "COMBINE": "gather", "REPLAY": 1}
    yield "bf16-stdmoe", {"DTYPE": "bf16", "COMBINE": "gather", "STDMOE": 1}


_CASES = dict(_cases())
_GFX950_ONLY = {"fp4-gather"}


@pytest.fixture(scope="module")
def case_output():
    """{id: (output, finished)} for every mode this arch runs, from one launch."""
    ids = [c for c in _CASES if _IS_GFX950 or c not in _GFX950_ONLY]
    rc, out = _run({"CASES": json.dumps([_CASES[c] for c in ids])}, timeout=1800)
    parts = {c: ([], False) for c in ids}
    current = None
    for ln in out.splitlines():
        m = re.search(r"# CASE (\d+)( DONE)?", ln)
        if m:
            current = ids[int(m.group(1))]
            if m.group(2):
                parts[current] = (parts[current][0], True)
            continue
        if current is not None:
            parts[current][0].append(ln)
    # A crash is the first unfinished mode's; if every mode finished, the last's.
    unfinished = [c for c in ids if not parts[c][1]]
    crashed = unfinished[0] if unfinished else (ids[-1] if rc != 0 else None)
    return rc, out, crashed, {c: ("\n".join(t), done) for c, (t, done) in parts.items()}


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, marks=_skip_fp4) if c in _GFX950_ONLY else c for c in _CASES],
)
def test_dispatch_combine_v2_intranode(case, case_output):
    rc, out, crashed, parts = case_output
    assert case != crashed, f"the launch failed in {case} (rc={rc}); output:\n{out}"
    text, finished = parts[case]
    assert finished, f"{case} did not run: the launch failed in {crashed} first"
    _assert_all_pass(0, text)


def test_op_lifecycle():
    """Repeated op build/close on one comm must not leak (see test_op_lifecycle.py)."""
    script = os.path.join(_HERE, "test_op_lifecycle.py")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        script,
    ]
    proc = subprocess.run(
        cmd,
        cwd=_HERE,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
    )
    assert "# LIFECYCLE: PASS" in proc.stdout, f"rc={proc.returncode}\n{proc.stdout}"
