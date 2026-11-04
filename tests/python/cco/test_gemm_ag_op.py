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
"""``mori.ops.gemm_ag.GemmAgOp`` on several ranks.

The configuration rules run on the CPU. Everything else comes from one torchrun
spawn of ``gemm_ag_op_worker.py``: every precision in both modes, every bf16
tile, chunk counts, back-to-back calls with a held reader with the recv-reuse
credit on and off, alternating M, padding, operand checks, ``self_test`` and its
teeth, ``close``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mori.ops.gemm_ag import GemmAgOp, supports
from mori.ops.gemm_ag.op import shape_constraint

WORKER = Path(__file__).with_name("gemm_ag_op_worker.py")
#: fp8's own floor at these operands, and bf16's rounding for a bf16 C.
FLOOR = 3e-3
#: bf16 in, fp32 accumulate and out: only the accumulation order differs.
FP32_FLOOR = 1e-5
_SETTLE_SECONDS = 20
_QUEUE_EXHAUSTED = "anvil.cpp"


@pytest.mark.parametrize(
    "n,k,kw",
    [
        (2048, 7168, {}),
        (2048, 7168, dict(quant="blockscale")),
        (2048, 7168, dict(quant="mxfp8")),
        (2048, 7168, dict(in_dtype="bf16", out_dtype="fp32")),
        (1024, 5120, dict(in_dtype="bf16")),
        (1024, 128, dict(in_dtype="bf16")),
    ],
)
def test_supports_the_model_shapes(n, k, kw):
    assert supports(n, k, 8, **kw), shape_constraint(n, k, **kw)


@pytest.mark.parametrize(
    "n,k,kw",
    [
        (1000, 7168, {}),  # N not a multiple of 256
        (2048, 128, {}),  # fp8 K below two blocks
        (2048, 7200, {}),  # fp8 K not a multiple of 128
        (2048, 64, dict(in_dtype="bf16")),  # bf16 K below two blocks
        (2048, 7168, dict(out_dtype="fp32")),  # the fp8 epilogue emits bf16
        (2048, 7168, dict(in_dtype="bf16", quant="mxfp8")),  # bf16 is not quantised
        (2048, 7168, dict(in_dtype="fp4")),
        (2048, 7168, dict(quant="int8")),
    ],
)
def test_rejects_what_no_kernel_can_build(n, k, kw):
    assert not supports(n, k, 8, **kw)
    assert shape_constraint(n, k, **kw) is not None


@pytest.mark.parametrize("world", [1, 9])
def test_rejects_world_sizes_outside_a_node(world):
    assert not supports(2048, 7168, world)


@pytest.mark.parametrize(
    "chunks,cap,m,tile,want",
    [
        (1, 4, 1024, (128, 256), 1),
        (2, 4, 1024, (128, 256), 2),
        (4, 4, 1024, (128, 256), 4),
        (8, 4, 1024, (128, 256), 4),  # capped by max_chunks
        (4, 4, 768, (256, 256), 3),  # rounded down to a divisor of 3 tiles
    ],
)
def test_pinned_chunks_divide_the_row_tiles_within_the_cap(chunks, cap, m, tile, want):
    op = SimpleNamespace(tile=tile, chunks=chunks, max_chunks=cap)
    op.fuses_at = lambda _m: True
    op.tile_for = lambda _m: tile
    assert GemmAgOp.chunks_for(op, m) == want


def test_split_mode_pushes_in_one_piece():
    op = SimpleNamespace(tile=None, chunks=4, max_chunks=4)
    op.fuses_at = lambda _m: False
    assert GemmAgOp.chunks_for(op, 4096) == 1


@pytest.mark.parametrize(
    "n,m,cus,want",
    [
        # Every pair in the README's results table, at 256 CUs.
        (2048, 2048, 256, False),
        (2048, 4096, 256, False),
        (2048, 8192, 256, True),
        (2048, 16384, 256, True),
        (1024, 2048, 256, False),
        (1024, 16384, 256, True),
        # The same shape on a part with more CUs needs a longer GEMM.
        (2048, 8192, 304, False),
    ],
)
def test_auto_fuses_where_the_measurements_say_fusion_pays(n, m, cus, want):
    op = SimpleNamespace(mode="auto", n=n, cus=cus)
    assert GemmAgOp.fuses_at(op, m) is want


@pytest.mark.parametrize("mode,want", [("fused", True), ("split", False)])
def test_explicit_modes_ignore_the_rule(mode, want):
    op = SimpleNamespace(mode=mode, n=2048, cus=256)
    assert GemmAgOp.fuses_at(op, 2048) is want and GemmAgOp.fuses_at(op, 16384) is want


def _spawn(last=False):
    time.sleep(_SETTLE_SECONDS)
    env = os.environ.copy()
    env.setdefault("MORI_SOCKET_IFNAME", "lo")
    env.setdefault("MORI_ENABLE_SDMA", "1")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            str(WORKER),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=2400,
    )
    out = result.stdout + result.stderr
    if result.returncode != 0:
        if _QUEUE_EXHAUSTED in out and not last:
            time.sleep(_SETTLE_SECONDS * 3)
            return _spawn(last=True)
        raise AssertionError(out[-8000:])
    records = [
        json.loads(ln.removeprefix("RESULT_JSON "))
        for ln in out.splitlines()
        if ln.startswith("RESULT_JSON ")
    ]
    assert records, out[-8000:]
    return {r.pop("case"): r for r in records}


@pytest.fixture(scope="module")
def cases():
    if torch.cuda.device_count() < 8:
        pytest.skip("requires 8 GPUs")
    pytest.importorskip("flydsl")
    return _spawn()


def test_self_test_passes_and_can_fail(cases):
    r = cases["self_test"]
    assert r["passes"] and r["detects_missing_push"], r


@pytest.mark.parametrize(
    "prec", ["fp8-ptpc", "fp8-blockscale", "fp8-mxfp8", "bf16-bf16", "bf16-fp32"]
)
def test_every_precision_validates_in_both_modes(cases, prec):
    """Two operand sets per op, so a stale read of the first cannot pass for
    the second. Parametrised over the quantisation as well as the mode: a
    since-removed LSA path once validated under ptpc and failed under
    blockscale, whose slower GEMM widens the rank skew any race needs."""
    r = cases["precisions_and_modes"]
    floor = FP32_FLOOR if prec == "bf16-fp32" else FLOOR
    assert r[f"{prec}_fused"] < floor, r
    assert r[f"{prec}_split"] < floor, r
    assert r[f"{prec}_equal"], (prec, r)


def test_every_bf16_tile_validates_fused_and_split(cases):
    r = cases["bf16_tiles"]
    assert len(r) == 8, r
    for key, value in r.items():
        assert value < FP32_FLOOR, (key, r)


def test_the_chunked_epilogue_is_deterministic(cases):
    """A race in the chunked tail is intermittent, not absent: the same bytes at
    every chunk count is what a clean one looks like."""
    r = cases["chunk_counts"]
    assert r["used"] == [1, 2, 4], r
    assert r["equal"], r
    assert r["rel_l2"] < FLOOR, r


def test_back_to_back_calls_with_a_held_reader_stay_correct(cases):
    r = cases["back_to_back"]
    for mode in ("fused", "split"):
        assert len(r[mode]) == 8, r
        assert all(v < FP32_FLOOR for v in r[mode]), (mode, r)


def test_without_the_credit_a_held_reader_reads_the_next_call(cases):
    """The negative control: the same race with the recv-reuse credit off has
    to fail, or the test above proves nothing."""
    r = cases["back_to_back"]
    assert any(v > 0.1 for v in r["fused_no_credit"]), r


def test_one_instance_serves_two_m_values_back_to_back(cases):
    """Alternating M -- and with ``auto``, alternating fused and split -- with
    no host sync and a held reader: recv is laid out per M, and only the
    credit keeps a peer's next push off rows still being read."""
    r = cases["alternating_m"]
    assert r["fused"] == [False, True], r
    assert len(r["rels"]) == 8, r
    assert all(v < FP32_FLOOR for v in r["rels"]), r


def test_alternating_m_without_the_credit_reads_the_next_call(cases):
    """The negative control for the test above."""
    r = cases["alternating_m"]
    assert any(v > 0.1 for v in r["no_credit"]), r


def test_one_instance_serves_any_number_of_m(cases):
    """Ten distinct M through one fused instance, then every election counter
    back at zero -- which is what lets one counter set serve them all."""
    r = cases["many_m"]
    assert len(r["rels"]) == 10, r
    assert all(v < FP32_FLOOR for v in r["rels"]), r
    assert r["counters_left"] == 0, r
    assert len(r["chunks"]) > 1, r


def test_padded_token_count_is_correct_and_its_padding_zero(cases):
    r = cases["padded_rows"]
    assert r["rel_l2"] < FP32_FLOOR and r["padding_is_zero"], r


def test_operands_and_configurations_no_kernel_takes_are_refused(cases):
    r = cases["operand_checks"]
    assert all(r.values()), r


def test_close_releases_and_is_idempotent(cases):
    assert cases["close"]["refused_after_close"], cases["close"]
