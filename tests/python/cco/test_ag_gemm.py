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
"""``mori.ops.ag_gemm``: the window layout on the CPU, and every mode end to end.

The GPU half drives ``bench_ag_gemm.py``, every run from one torchrun launch
(``bench_runs.py``). Each run validates every
row of C twice with different operands, then ``--back-to-back`` issues calls with
no host sync between them, holding a different rank before each, and validates
every call -- the case where a rank a call ahead refills a slot a held rank is
still reading.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest
import torch

from mori.ops.ag_gemm import MAX_CHUNKS, MAX_WORLD, AgGemmConfig

from . import kfd_settle

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH = REPO_ROOT / "benchmark" / "cco" / "flydsl" / "ag_gemm" / "bench_ag_gemm.py"

#: SDMA queues of a finished process group take seconds to come back; launching
#: the next one straight away fails in hsaKmtCreateQueueExt (anvil.cpp).
_SETTLE_SECONDS = 15
_QUEUE_EXHAUSTED = "anvil.cpp"


def test_control_regions_do_not_overlap_and_payload_is_page_aligned():
    cfg = AgGemmConfig(world_size=8, m_local=1024, k=7168, chunks=4)
    cfg.validate()
    assert cfg.flag_off + MAX_WORLD * MAX_CHUNKS * 4 <= cfg.bar_off
    assert cfg.flag_slot_off(7, 3) + 4 <= cfg.bar_off
    assert cfg.bar_slot_off(1, 7) + 4 <= cfg.control_bytes <= cfg.recv_off
    assert cfg.recv_off % 4096 == 0 and cfg.scale_off % 4096 == 0
    assert cfg.recv_slot_off(7) + cfg.slab_bytes <= cfg.scale_off
    assert cfg.scale_slot_off(7) + cfg.scale_slab_bytes <= cfg.window_bytes


def test_chunks_cut_the_slab_into_whole_row_tiles():
    cfg = AgGemmConfig(world_size=8, m_local=1024, k=7168, block_m=256, chunks=4)
    cfg.validate()
    assert cfg.chunk_rows == 256 and cfg.m_tiles_per_chunk == 1
    assert cfg.chunk_bytes * cfg.chunks == cfg.slab_bytes
    assert cfg.chunk_scale_bytes * cfg.chunks == cfg.scale_slab_bytes


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(world_size=8, m_local=768, k=7168, chunks=2),  # 3 row tiles, 2 chunks
        dict(world_size=8, m_local=512, k=7168, chunks=MAX_CHUNKS + 1),
        dict(world_size=9, m_local=512, k=7168),
        dict(world_size=8, m_local=512, k=200),
        dict(world_size=8, m_local=512, k=7168, block_m=192),
    ],
)
def test_validate_rejects_what_the_kernel_cannot_express(kwargs):
    with pytest.raises(ValueError):
        AgGemmConfig(**kwargs).validate()


def test_remote_bytes_count_rows_and_scales_to_every_peer():
    cfg = AgGemmConfig(world_size=8, m_local=512, k=7168)
    assert cfg.remote_bytes_per_rank == 7 * (512 * 7168 + 512 * 4)


requires_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs at least 2 GPUs"
)


DRIVER = Path(__file__).with_name("bench_runs.py")


def _run_benches(runs, last=False):
    """bench_ag_gemm.py once per argument list in runs, all from one launch;
    the results in order."""
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
            str(DRIVER),
            str(BENCH),
            json.dumps(runs),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    out = result.stdout + result.stderr
    if result.returncode != 0 and _QUEUE_EXHAUSTED in out and not last:
        time.sleep(_SETTLE_SECONDS * 3)
        return _run_benches(runs, last=True)
    hits = [
        json.loads(ln.removeprefix("RESULT_JSON "))
        for ln in out.splitlines()
        if ln.startswith("RESULT_JSON ")
    ]
    assert len(hits) == len(runs), out[-4000:]
    # A failed validation is each test's to report; any other non-zero exit is not.
    assert result.returncode == 0 or not all(h["validated"] for h in hits), out[-4000:]
    return hits


SMALL = ["-m", "512", "-N", "1024", "-k", "1024", "--warmup", "2", "--iters", "3"]
EVERY_MODE = {
    "fused-sdma-1": ["--mode", "fused-sdma", "--chunks", "1"],
    "fused-sdma-2": ["--mode", "fused-sdma", "--chunks", "2"],
    "split-sdma": ["--mode", "split-sdma"],
    "split-rccl": ["--mode", "split-rccl"],
}
PARTIAL_TILE = {
    "fused": ["--mode", "fused-sdma", "--chunks", "2"],
    "split": ["--mode", "split-sdma"],
}


@pytest.fixture(scope="module")
def bench():
    """Every GPU case below, from one launch."""
    runs = {
        f"every/{m}": [*a, *SMALL, "--back-to-back", "8"] for m, a in EVERY_MODE.items()
    }
    runs |= {
        f"partial/{m}": [*a, "-m", "512", "-N", "1168", "-k", "1024"]
        + ["--warmup", "2", "--iters", "3", "--back-to-back", "4"]
        for m, a in PARTIAL_TILE.items()
    }
    runs["gemm-only"] = ["--mode", "gemm-only", *SMALL]
    return dict(zip(runs, _run_benches(list(runs.values()))))


@requires_gpus
@pytest.mark.parametrize("mode", list(EVERY_MODE))
def test_every_mode_is_exact_to_the_fp8_floor_back_to_back(mode, bench):
    r = bench[f"every/{mode}"]
    assert r["validated"], r
    assert r["rel_l2"] < 3e-3, r


@requires_gpus
@pytest.mark.parametrize("mode", list(PARTIAL_TILE))
def test_a_partial_last_column_tile_back_to_back(mode, bench):
    """N=1168: four whole column tiles and 144 columns, masked in the store."""
    r = bench[f"partial/{mode}"]
    assert r["validated"], r
    assert r["rel_l2"] < 3e-3, r


@requires_gpus
def test_gemm_only_matches_the_same_reference(bench):
    """The compute floor the others are measured against has to be right too."""
    r = bench["gemm-only"]
    assert r["validated"], r
