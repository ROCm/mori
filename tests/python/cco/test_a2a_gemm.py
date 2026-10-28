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
"""``mori.ops.a2a_gemm``: the window layout on the CPU, and every mode end to end.

The GPU half drives ``bench_a2a_gemm.py`` under torchrun, every mode, chunk
count and wait mode in one process, with ``--acquire none`` -- the variant that
leans on cco's uncached window rather than a cache invalidate. Each case validates every row of C twice with different
operands, then ``--back-to-back`` issues calls with no host sync between them,
holding a different rank before each, and validates every call -- the case
where a rank a call ahead refills a ``recv`` slot a held rank is still reading.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mori.ops.a2a_gemm import BLOCK_K, MAX_CHUNKS, MAX_WORLD, A2aGemmConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH = REPO_ROOT / "benchmark" / "cco" / "flydsl" / "a2a_gemm" / "bench_a2a_gemm.py"


def test_control_send_and_recv_do_not_overlap_and_are_page_aligned():
    cfg = A2aGemmConfig(world_size=8, m_local=1024, k=8192, chunks=4)
    cfg.validate()
    assert cfg.sig_base_slot_off(7) + 4 <= cfg.bar_off
    assert cfg.sig_base_off + MAX_WORLD * 4 <= cfg.bar_off
    assert cfg.bar_slot_off(1, 7) + 4 <= cfg.control_bytes <= cfg.send_off
    assert cfg.send_off % 4096 == 0 and cfg.recv_off % 4096 == 0
    assert cfg.send_slot_off(7) + cfg.slab_bytes <= cfg.recv_off
    assert cfg.recv_slot_off(7) + cfg.slab_bytes <= cfg.window_bytes


def test_a_slab_is_one_destinations_rows_over_one_head_shard():
    cfg = A2aGemmConfig(world_size=4, m_local=2048, k=6144, chunks=4)
    assert cfg.k_shard == 1536 and cfg.k_steps_per_shard == 1536 // BLOCK_K
    assert cfg.slab_bytes == 2048 * 1536
    assert cfg.chunk_bytes * cfg.chunks == cfg.slab_bytes
    assert cfg.remote_bytes_per_rank == 3 * 2048 * 1536


def test_own_segment_reads_send_unless_told_otherwise():
    cfg = A2aGemmConfig(world_size=4, m_local=512, k=2048)
    own = cfg.segment_off(2, 2, self_from_send=True)
    assert own == cfg.send_slot_off(2) - cfg.send_off
    assert (
        cfg.segment_off(2, 2, self_from_send=False)
        == cfg.recv_slot_off(2) - cfg.send_off
    )
    assert (
        cfg.segment_off(1, 2, self_from_send=True)
        == cfg.recv_slot_off(1) - cfg.send_off
    )
    # The A descriptor spans send through recv.
    assert (
        cfg.segment_off(3, 2, self_from_send=True) + cfg.slab_bytes <= cfg.operand_bytes
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(world_size=4, m_local=768, k=6144, chunks=2),  # 3 row tiles, 2 chunks
        dict(world_size=4, m_local=512, k=6144, chunks=MAX_CHUNKS + 1),
        dict(world_size=9, m_local=512, k=9216),
        dict(world_size=8, m_local=512, k=6144 + 512),  # k_shard not whole K steps
        dict(world_size=4, m_local=512, k=6144, block_m=192),
    ],
)
def test_validate_rejects_what_the_kernel_cannot_express(kwargs):
    with pytest.raises(ValueError):
        A2aGemmConfig(**kwargs).validate()


requires_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs at least 2 GPUs"
)


@requires_gpus
def test_every_mode_is_exact_to_the_fp8_floor_back_to_back():
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
            "-m",
            "512",
            "-n",
            "1024",
            "-k",
            str(world * 256),
            "--chunks",
            "1",
            "2",
            "--wait",
            "segment",
            "tile",
            "--acquire",
            "none",
            "--warmup",
            "2",
            "--iters",
            "3",
            "--min-seconds",
            "0",
            "--back-to-back",
            "8",
        ],  # fmt: skip
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    out = result.stdout + result.stderr
    rows = [
        json.loads(ln.removeprefix("RESULT_JSON "))
        for ln in out.splitlines()
        if ln.startswith("RESULT_JSON ")
    ]
    assert result.returncode == 0, out[-4000:]
    seen = {(r["mode"], r["chunks"], r["wait"]) for r in rows}
    assert seen == {
        ("gemm-only", None, None),
        ("split-rccl", None, None),
        ("split-sdma", None, None),
        *{("fused-sdma", c, w) for c in (1, 2) for w in ("segment", "tile")},
    }, seen
    for r in rows:
        assert r["validated"] and r["rel_l2"] < 3e-3, r
