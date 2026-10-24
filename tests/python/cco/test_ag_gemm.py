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

The GPU half drives ``bench_ag_gemm.py`` under torchrun. Each run validates every
row of C twice with different operands, then ``--back-to-back`` issues calls with
no host sync between them, holding a different rank before each, and validates
every call -- the case where a rank a call ahead refills a slot a held rank is
still reading.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from mori.ops.ag_gemm import MAX_CHUNKS, MAX_WORLD, AgGemmConfig

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


def _run_bench(*args, last=False):
    time.sleep(_SETTLE_SECONDS)
    env = os.environ.copy()
    env.setdefault("MORI_SOCKET_IFNAME", "lo")
    env.setdefault("MORI_ENABLE_SDMA", "1")
    world = min(8, torch.cuda.device_count())
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world}",
            str(BENCH),
            *args,
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
        return _run_bench(*args, last=True)
    hit = [ln for ln in out.splitlines() if ln.startswith("RESULT_JSON ")]
    assert result.returncode == 0 and hit, out[-4000:]
    return json.loads(hit[-1].removeprefix("RESULT_JSON "))


SMALL = ["-m", "512", "-N", "1024", "-k", "1024", "--warmup", "2", "--iters", "3"]


@requires_gpus
@pytest.mark.parametrize(
    "mode",
    [
        ["--mode", "fused-sdma", "--chunks", "1"],
        ["--mode", "fused-sdma", "--chunks", "2"],
        ["--mode", "split-sdma"],
        ["--mode", "split-rccl"],
    ],
    ids=lambda m: "-".join(m[1::2]) if len(m) > 2 else m[1],
)
def test_every_mode_is_exact_to_the_fp8_floor_back_to_back(mode):
    r = _run_bench(*mode, *SMALL, "--back-to-back", "8")
    assert r["validated"], r
    assert r["rel_l2"] < 3e-3, r


@requires_gpus
def test_gemm_only_matches_the_same_reference():
    """The compute floor the others are measured against has to be right too."""
    r = _run_bench("--mode", "gemm-only", *SMALL)
    assert r["validated"], r
