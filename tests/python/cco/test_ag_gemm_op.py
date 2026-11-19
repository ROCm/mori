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
"""``mori.ops.ag_gemm.AgGemmOp``: the public wrapper.

The shape rules run on the CPU; the rest comes from one torchrun spawn of
``ag_gemm_op_worker.py``, which runs every GPU case in one process group.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mori.ops.ag_gemm import AgGemmOp, supports
from mori.ops.ag_gemm.op import shape_constraint

from . import kfd_settle

WORKER = Path(__file__).with_name("ag_gemm_op_worker.py")
FP8_FLOOR = 3e-3
_SETTLE_SECONDS = 15
_QUEUE_EXHAUSTED = "anvil.cpp"


@pytest.mark.parametrize(
    "n,k",
    # 6288 and 3648 are Kimi-K3's KDA and MLA shards at TP8: a partial last
    # column tile, which the kernel masks rather than the caller padding.
    [(6400, 7168), (6288, 7168), (3648, 7168), (1040, 256)],
)
def test_supports_takes_any_n_in_whole_weight_bands(n, k):
    assert supports(n, k, 8) and shape_constraint(n, k) is None


@pytest.mark.parametrize(
    "n,k",
    [(6280, 7168), (1024, 128), (1024, 200), (0, 1024)],
)
def test_supports_rejects_what_the_kernel_cannot_build(n, k):
    assert not supports(n, k, 8) and shape_constraint(n, k) is not None


@pytest.mark.parametrize("world", [1, 9])
def test_supports_rejects_world_sizes_outside_a_node(world):
    assert not supports(1024, 1024, world)


@pytest.mark.parametrize(
    "chunks,n,m,want",
    [
        # Pinned: the largest divisor of the row tiles up to it.
        (4, 1024, 768, 3),
        (4, 1024, 1024, 4),
        (8, 1024, 2048, 8),
        (8, 1024, 1536, 6),
        # Default, at every cell of the README's tables (P=8, m_local rows).
        (None, 6400, 512, 2),  # KDA
        (None, 6400, 1024, 4),
        (None, 6400, 2048, 4),
        (None, 6400, 4096, 4),
        (None, 3840, 512, 1),  # MLA: c2 lost to c1 and to split here
        (None, 3840, 1024, 2),
        (None, 3840, 2048, 4),
        (None, 3840, 4096, 4),
        (None, 6288, 1024, 4),  # the unpadded KDA shard: same 25 column tiles
        (None, 1024, 256, 1),  # too few tiles for any band to reach the floor
    ],
)
def test_chunks_are_a_divisor_of_the_row_tiles(chunks, n, m, want):
    assert AgGemmOp.chunks_for(SimpleNamespace(chunks=chunks, n=n), m) == want


requires_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")


def _spawn(last=False):
    env = os.environ.copy()
    env.setdefault("MORI_SOCKET_IFNAME", "lo")
    env.setdefault("MORI_ENABLE_SDMA", "1")
    world = min(8, torch.cuda.device_count())
    result = kfd_settle.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world}",
            str(WORKER),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    out = result.stdout + result.stderr
    if result.returncode != 0:
        if _QUEUE_EXHAUSTED in out and not last:
            time.sleep(_SETTLE_SECONDS * 3)
            return _spawn(last=True)
        raise AssertionError(out[-6000:])
    records = [
        json.loads(ln.removeprefix("RESULT_JSON "))
        for ln in out.splitlines()
        if ln.startswith("RESULT_JSON ")
    ]
    assert records, out[-6000:]
    return {r.pop("case"): r for r in records}


@pytest.fixture(scope="module")
def cases():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    return _spawn()


@requires_gpus
def test_self_test_passes_and_can_fail(cases):
    """The guard against a mori whose puts move nothing: it passes on a working
    stack, and it fails when the push is skipped."""
    r = cases["self_test"]
    assert r["passes"] and r["detects_missing_push"], r


@requires_gpus
def test_padded_token_count_is_correct_and_its_padding_zero(cases):
    r = cases["padded_rows"]
    assert r["rel_l2"] < FP8_FLOOR, r
    assert r["padding_is_zero"], r


@requires_gpus
def test_one_instance_serves_changing_m_back_to_back(cases):
    """The receive slots are laid out for the current M, so switching M moves
    them; the closing barrier is what makes that safe. Four M, no host sync,
    a different rank held before each call."""
    r = cases["alternating_m"]
    assert len(r) == 6, r
    for key, value in r.items():
        assert value < FP8_FLOOR, (key, r)


@requires_gpus
def test_quantising_into_the_window_matches_the_copying_path(cases):
    r = cases["local_rows_and_out"]
    assert r["returned_out"] and r["same_bytes"], r
    assert r["rel_l2"] < FP8_FLOOR, r


@requires_gpus
def test_split_and_fused_write_the_same_bytes(cases):
    r = cases["split_matches_fused"]
    assert r["equal"], r
    assert r["rel_l2"] < FP8_FLOOR, r


@requires_gpus
def test_back_to_back_calls_with_a_held_rank_stay_correct(cases):
    r = cases["back_to_back"]
    assert len(r) == 8, r
    for key, value in r.items():
        assert value < FP8_FLOOR, (key, r)


@requires_gpus
def test_operands_the_launch_would_misread_are_refused(cases):
    r = cases["operand_checks"]
    assert all(r.values()), r


@requires_gpus
def test_close_releases_and_is_idempotent(cases):
    assert cases["close"]["refused_after_close"], cases["close"]


@requires_gpus
def test_a_partial_last_column_tile_is_masked_not_padded(cases):
    """N=1168 is four whole column tiles and 144 columns: the last tile reads
    zeros past N and drops its stores there, in both modes."""
    r = cases["ragged_n"]
    assert r["fused_rel_l2"] < FP8_FLOOR, r
    assert r["split_rel_l2"] < FP8_FLOOR, r
    assert r["equal"], r


@requires_gpus
def test_calls_across_the_epoch_wrap_stay_correct(cases):
    """The band and barrier waits compare ``(int32)(cur - target)``, so the
    call whose epoch wraps to 0 still waits for its flags."""
    r = cases["epoch_wrap"]
    assert len(r) == 8, r
    for key, value in r.items():
        assert value < FP8_FLOOR, (key, r)
