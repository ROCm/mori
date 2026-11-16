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
"""``mori.ops.gemm_ar_v2.GemmArV2`` (BF16 GEMM + all-reduce) on 8 ranks.

Layout arithmetic runs on the CPU. Everything else comes from one torchrun spawn
of ``gemm_ar_v2_worker.py``: ``self_test``, correctness against the fp32 sum
over ranks at M = 1, 1000 (not a multiple of 256), 4096 and 6309, eager and
graph-replayed, a counter and epoch wrap through ``debug_preset_counters``, that
the fused path ran, and a same-process ordering test of the fused path against
the in-module split path. Absolute times are recorded, not asserted.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import pytest
import torch

from mori.ops.gemm_ar_v2 import layout as L

from . import kfd_settle

WORKER = Path(__file__).with_name("gemm_ar_v2_worker.py")
TOL = 6e-3
_QUEUE_EXHAUSTED = "anvil.cpp"


# --- CPU: layout ------------------------------------------------------------


@pytest.mark.parametrize(
    "m,mp,chunks",
    [(1, 2048, 1), (1000, 2048, 1), (4096, 4096, 2), (6309, 8192, 4), (10657, 12288, 6), (16384, 16384, 8)],
)
def test_padding_and_chunk_rule(m, mp, chunks):
    assert L.padded_m(m, 8) == mp
    sh = L.Shape(8, 7168, mp, L.default_chunks(mp // 8 // L.BLOCK_M))
    sh.validate()
    assert sh.chunks == chunks
    # one 256-row tile per (destination, chunk): 3.5 MiB puts
    assert sh.chunk_bytes == 256 * 7168 * 2


def test_window_fits_buffer_offsets():
    lay = L.Layout(8, 7168, L.padded_m(16384, 8))
    assert lay.window_bytes < 2**31  # element offsets are computed in i32
    assert max(L.EPOCH_WORDS + L.COUNTER_WORDS) < L.CTRL_BYTES


# --- 8 ranks ------------------------------------------------------------------


def _spawn(last=False):
    env = os.environ.copy()
    env.setdefault("MORI_SOCKET_IFNAME", "lo")
    env.setdefault("MORI_ENABLE_SDMA", "1")
    result = kfd_settle.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=8", str(WORKER)],
        env=env,
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("GEMM_AR_V2_TEST_TIMEOUT", "540")),
    )
    out = result.stdout + result.stderr
    if result.returncode != 0 and _QUEUE_EXHAUSTED in out and not last:
        time.sleep(60)
        return _spawn(last=True)
    records = [
        json.loads(ln.removeprefix("RESULT_JSON "))
        for ln in out.splitlines()
        if ln.startswith("RESULT_JSON ")
    ]
    assert records, out[-8000:]
    return {r["case"]: r["per_rank"] for r in records}, result.returncode, out


@pytest.fixture(scope="module")
def cases():
    if torch.cuda.device_count() < 8:
        pytest.skip("requires 8 GPUs")
    pytest.importorskip("flydsl")
    got, rc, out = _spawn()
    sdma = got.get("sdma")
    if sdma and not all(r["available"] for r in sdma):
        pytest.skip(f"no SDMA: {sdma[0].get('error')}")
    assert rc == 0, out[-8000:]
    return got


def _rows(cases, case):
    per_rank = cases.get(case)
    assert per_rank is not None, f"case {case} missing"
    for r, mine in enumerate(per_rank):
        assert "error" not in mine, f"rank {r}: {mine['error']}"
    return [row for mine in per_rank for row in mine["rows"]]


@pytest.mark.parametrize("m", [1, 1000, 4096, 6309])
def test_correct_eager_and_graph(cases, m):
    rows = _rows(cases, f"correct_m{m}")
    assert {r["kind"] for r in rows} == {"eager", "graph"}
    worst = max(r["rel_l2"] for r in rows)
    assert worst <= TOL, f"M={m}: rel_l2 {worst:.3e}"


@pytest.mark.parametrize("case", ["correct_m1", "correct_m1000", "correct_m4096", "wrap"])
def test_fused_path_ran(cases, case):
    # M=1 and M=1000 run the padded fused pipeline, not a fallback
    assert all(r["fused"] is True for r in _rows(cases, case))


def test_counters_and_epochs_cross_2_32(cases):
    rows = _rows(cases, "wrap")
    worst = max(r["rel_l2"] for r in rows)
    assert worst <= TOL, f"after the preset: rel_l2 {worst:.3e}"
    # 8 calls (4 eager, 1 warm-up, 3 replays) after a preset of 2^32 - 3 have
    # wrapped the epochs past zero
    for mine in cases["wrap"]:
        assert mine["dump"]["FG"] < 2**31, mine["dump"]


def test_fused_beats_split_in_one_process(cases):
    per_rank = cases["ordering"]
    for r, mine in enumerate(per_rank):
        assert "error" not in mine, f"rank {r}: {mine['error']}"
    rounds = len(per_rank[0]["us"]["fused"])
    # maximum over ranks per round, median over rounds
    fused = statistics.median(max(m["us"]["fused"][i] for m in per_rank) for i in range(rounds))
    split = statistics.median(max(m["us"]["split"][i] for m in per_rank) for i in range(rounds))
    print(f"gemm_ar_v2 M=16384: fused {fused:.1f} us, split {split:.1f} us")
    assert fused < split, (fused, split)
