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
"""FlyDSL cco all-reduce: layout unit tests + multi-GPU smoke.

Mirrors ``test_triton_gemm_a2a.py``: the pure-Python half checks the arithmetic
the kernels depend on without a GPU, the GPU half runs ``bench_ar.py`` under
torchrun and parses its ``RESULT_JSON`` line.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from mori.ops.gemm_ar import layout

REPO_ROOT = Path(__file__).resolve().parents[3]
AR_DIR = REPO_ROOT / "benchmark" / "cco" / "flydsl" / "ar"


# --------------------------------------------------------------------------
# layout / dispatch arithmetic (no GPU)
# --------------------------------------------------------------------------


def _cfg(**kw):
    base = dict(world_size=8, m=4096, n=7168)
    base.update(kw)
    return layout.ArConfig(**base)


def test_window_regions_are_disjoint_and_ordered():
    c = _cfg()
    c.validate()
    assert c.input_off >= c.signal_bytes
    assert c.output_off == c.input_off + c.nbytes
    assert c.tmp_off == c.output_off + c.nbytes
    assert c.window_bytes == c.tmp_off + c.tmp_bytes
    # signal sub-regions must not overlap the payload
    assert c.flag_off + c.max_blocks * 4 <= c.input_off


def test_recv_region_is_absent_unless_sdma_asks_for_it():
    """LSA reduces out of the peers' inputs, so it must not pay for landing slots."""
    lsa = _cfg()
    assert lsa.recv_bytes == 0
    assert lsa.window_bytes == lsa.tmp_off + lsa.tmp_bytes
    with pytest.raises(IndexError, match="no landing slot"):
        lsa.recv_slot_off(0)


def test_recv_slots_are_one_contiguous_slice_per_peer():
    c = _cfg(recv_slots=8)
    assert c.recv_off == c.tmp_off + c.tmp_bytes
    assert c.recv_bytes == 8 * c.slice_bytes
    assert c.window_bytes == c.recv_off + c.recv_bytes
    offs = [c.recv_slot_off(p) for p in range(8)]
    assert offs == sorted(offs)
    assert all(b - a == c.slice_bytes for a, b in zip(offs, offs[1:]))
    with pytest.raises(IndexError):
        c.recv_slot_off(8)


def test_validate_rejects_more_recv_slots_than_peers():
    with pytest.raises(ValueError, match="recv_slots"):
        _cfg(recv_slots=9).validate()


def test_signal_slot_is_block_major_over_peers():
    c = _cfg()
    assert c.signal_slot(0, 0) == 0
    assert c.signal_slot(0, 7) == 7
    assert c.signal_slot(1, 0) == layout.MAX_WORLD
    assert c.signal_slot(79, 7) == 79 * layout.MAX_WORLD + 7
    with pytest.raises(IndexError):
        c.signal_slot(layout.K_MAX_BLOCKS, 0)
    with pytest.raises(IndexError):
        c.signal_slot(0, layout.MAX_WORLD)


def test_pack_and_shard_math():
    c = _cfg(m=4096, n=7168)  # 4096*7168*2 = 58,720,256 B
    assert c.nbytes == 4096 * 7168 * 2
    assert c.num_packs == c.nbytes // 16
    assert c.packs_per_rank == c.num_packs // 8
    start, end = c.owner_pack_range(0)
    assert (start, end) == (0, c.packs_per_rank)
    # the last rank absorbs the remainder so no pack is dropped
    assert c.owner_pack_range(7)[1] == c.num_packs
    covered = sum(e - s for s, e in (c.owner_pack_range(r) for r in range(8)))
    assert covered == c.num_packs


def test_slice_is_contiguous_and_large_enough_for_sdma():
    """M-sharding must give one contiguous run per peer, not m strided pieces."""
    c = _cfg(m=4096, n=7168)
    assert c.slice_bytes == (4096 // 8) * 7168 * 2
    # >=1MB is where SDMA leaves the ~6us dispatch floor behind
    assert c.slice_bytes > 1 << 20


@pytest.mark.parametrize(
    "world_size,nbytes,expected",
    [
        (2, 1 << 30, 1),  # world==2 is always 1-stage
        (4, 100 * 1024, 1),  # < 160KB
        (4, 200 * 1024, 2),
        (8, 64 * 1024, 1),  # < 80KB
        (8, 96 * 1024, 2),
        (8, 80 * 1024, 2),  # threshold is strict <
    ],
)
def test_stage_selection_matches_aiter(world_size, nbytes, expected):
    assert layout.select_stage(world_size, nbytes) == expected


def test_decode_shape_picks_two_stage():
    """[M=64, 7168] bf16 = 896KB -> above the 80KB gate, so 2-stage."""
    c = _cfg(m=64)
    assert c.nbytes == 64 * 7168 * 2
    assert c.stage == 2


def test_blocks_are_capped_and_never_zero():
    assert _cfg(m=16384).blocks == layout.LSA_BLOCK_CAP
    assert _cfg(m=1).blocks >= 1
    assert _cfg(m=1).blocks <= layout.LSA_BLOCK_CAP
    # the cap is a tuning knob, the signal array is the correctness bound
    assert layout.LSA_BLOCK_CAP <= layout.K_MAX_BLOCKS


def test_blocks_stride_the_index_space_by_threads_not_by_peer_group():
    """One thread owns one index across all peers, so a block covers `threads`."""
    c = _cfg(m=4096)
    small = layout.ArConfig(world_size=8, m=8, n=7168)
    assert small.packs_per_rank == 896
    assert small.blocks == 2  # ceil(896/512), not aiter's ceil(896/64) = 14
    assert c.blocks == layout.LSA_BLOCK_CAP


def test_force_blocks_must_fit_the_signal_array():
    with pytest.raises(ValueError, match="force_blocks"):
        layout.ArConfig(
            world_size=8, m=4096, n=7168, force_blocks=layout.K_MAX_BLOCKS + 1
        ).validate()


def test_remote_traffic_model():
    c = _cfg(m=4096)
    assert c.stage == 2
    # 2-stage moves 2*(P-1)/P of the payload, i.e. 1.75x at P=8
    assert c.remote_bytes_per_rank == 2 * 7 * c.nbytes // 8


def test_validate_rejects_indivisible_world():
    with pytest.raises(ValueError, match="divisible"):
        _cfg(world_size=6).validate()


def test_validate_rejects_unvectorizable_payload():
    with pytest.raises(ValueError, match="multiple"):
        # 7 elements of bf16 is not a multiple of world*16
        layout.ArConfig(world_size=8, m=1, n=7).validate()


# --------------------------------------------------------------------------
# GPU smoke
# --------------------------------------------------------------------------


def _run_bench(world_size, backend, m, n, extra_env=None):
    env = os.environ.copy()
    env.setdefault("MORI_SOCKET_IFNAME", "lo")
    env.setdefault("MORI_ENABLE_SDMA", "1")
    if extra_env:
        env.update(extra_env)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        str(AR_DIR / "bench_ar.py"),
        "--backend",
        backend,
        "-m",
        str(m),
        "-n",
        str(n),
        "--warmup",
        "1",
        "--iters",
        "3",
    ]
    result = subprocess.run(
        command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    records = [
        json.loads(line.removeprefix("RESULT_JSON "))
        for line in output.splitlines()
        if line.startswith("RESULT_JSON ")
    ]
    assert len(records) == 1, output
    return records[0]


@pytest.mark.parametrize("backend", ["lsa", "sdma"])
@pytest.mark.parametrize("world_size,m,n", [(2, 256, 1024), (4, 256, 1024)])
def test_flydsl_ar_gpu_smoke(backend, world_size, m, n):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    record = _run_bench(world_size, backend, m, n)
    assert record["validated"] is True
    assert record["max_rank_time_ms"] > 0
    assert record["backend"] == backend


def test_flydsl_ar_matches_aiter_bitwise():
    """Every element is reduced once on its owner rank, so the two agree exactly."""
    world_size = 2
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    pytest.importorskip("aiter", reason="aiter baseline not installed in this venv")
    record = _run_bench(world_size, "lsa", 256, 1024, {"AR_CHECK_VS_AITER": "1"})
    assert record["validated"] is True
    assert record.get("rel_l2_vs_aiter") == 0.0
