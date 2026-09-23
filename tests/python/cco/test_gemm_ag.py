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
"""Numerics and layout math for ``mori.ops.gemm_ag``.

The layout half is pure arithmetic and needs no GPU, which is the point: the
all-gather's index map is where a mistake is invisible at runtime -- wrong data
still arrives, in the wrong place, and only a value check finds it.

The multi-rank half spawns the benchmark and checks the modes agree.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from mori.ops.gemm_ag import layout

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH = REPO_ROOT / "benchmark" / "cco" / "flydsl" / "gemm_ag" / "bench_gemm_ag.py"

#: The DeepSeek V4-Pro prefill-CP ``wkv_gate`` shape this op is aimed at:
#: K=7168 hidden, N=2048 for the ratio-4 layers, M = tokens/P.
MODEL = dict(world_size=8, m=2048, n=2048)


# --------------------------------------------------------------------------
# layout (no GPU)
# --------------------------------------------------------------------------


def test_slab_is_the_whole_local_result():
    """No sharding, which is the whole difference from the all-to-all."""
    c = layout.ag_config(**MODEL)
    assert c.slab_elems == 2048 * 2048
    assert c.m_tiles == 16  # 2048 / 128
    assert c.n_blocks == 8  # 2048 / 256
    assert c.tiles_total == 16 * 8
    # Every rank sends world-1 copies of its whole result and keeps one. That
    # is `world` times what gemm_a2a sends at the same [M, N], because there a
    # slab is M*N/world and here it is M*N.
    assert c.remote_bytes_per_rank == 7 * c.slab_bytes
    assert c.slab_bytes == c.nbytes


def test_recv_is_the_output_and_the_input():
    """Slot ``rank`` is where the GEMM writes; there is no staging region.

    The saving is not a copy, it is a *region*: gemm_a2a reserves a second full
    payload's worth of window to hold the re-laid-out slab the copy engine can
    read. This op's window is the output and nothing else.
    """
    c = layout.ag_config(**MODEL)
    assert c.recv_bytes == 8 * c.slab_bytes
    assert c.recv_off == c.signal_bytes  # nothing between the locks and recv
    assert c.window_bytes == c.recv_off + c.recv_bytes
    assert not hasattr(c, "staging_off")


def test_every_region_is_disjoint_and_inside_the_window():
    c = layout.ag_config(**MODEL, counter_chunks=4)
    regions = [
        ("start", c.start_off, c.end_off),
        ("end", c.end_off, c.end_off),
        ("flag", c.flag_off, c.max_blocks * 4),
        ("counters", c.counter_region_off, c.counter_bytes),
        ("locks", c.lock_off, c.lock_bytes),
        ("recv", c.recv_off, c.recv_bytes),
    ]
    regions = [r for r in regions if r[2] > 0]
    regions.sort(key=lambda r: r[1])
    for (an, ao, asz), (bn, bo, _) in zip(regions, regions[1:]):
        assert ao + asz <= bo, f"{an} overruns {bn}"
    last_n, last_o, last_sz = regions[-1]
    assert last_o + last_sz <= c.window_bytes, f"{last_n} past the window"


def test_control_regions_are_aligned():
    c = layout.ag_config(**MODEL, counter_chunks=4)
    for name in ("start_off", "flag_off", "counter_region_off", "lock_off", "recv_off"):
        off = getattr(c, name)
        assert off % layout.SIGNAL_ALIGN == 0, f"{name}={off} unaligned"


def test_recv_slot_offsets_are_source_major_and_stay_in_region():
    """The same offset in every rank's window -- both ends of the transfer.

    In gemm_a2a only the *destination* offset has this property; the source
    comes out of staging and depends on the destination. Here a push reads
    ``recv_slot_off(rank)`` locally and writes ``recv_slot_off(rank)`` remotely,
    which is why its SDMA put has one constant instead of two.
    """
    c = layout.ag_config(**MODEL)
    for src in range(8):
        off = c.recv_slot_off(src)
        assert off == c.recv_off + src * c.cap_slab_bytes
        assert c.recv_off <= off < c.recv_off + c.recv_bytes
        assert off + c.slab_bytes <= c.window_bytes
    with pytest.raises(IndexError):
        c.recv_slot_off(8)


def test_recv_index_is_row_major_within_a_source_slab():
    c = layout.ag_config(**MODEL)
    assert c.recv_index(0, 0, 0) == 0
    assert c.recv_index(0, 1, 0) == c.n
    assert c.recv_index(3, 0, 0) == 3 * c.slab_elems
    # The last element of the last source lands exactly at the end.
    assert c.recv_index(7, c.m - 1, c.n - 1) == 8 * c.slab_elems - 1
    with pytest.raises(IndexError):
        c.recv_index(8, 0, 0)


def test_n_need_only_be_whole_n_tiles():
    """The rule gemm_a2a has and this op does not.

    ``n=1024`` at world_size=8 is rejected by ``a2a_config`` -- a destination's
    column shard would be 128 wide, half a GEMM tile -- and accepted here,
    because no column is sharded. That is not a corner case: it is every
    ratio-128 layer of the wkv_gate chain.
    """
    assert layout.ag_config(world_size=8, m=2048, n=1024).n_blocks == 4
    with pytest.raises(ValueError, match="multiple of block_n"):
        layout.ag_config(world_size=8, m=2048, n=1152)


def test_m_must_be_whole_row_tiles_and_chunks_must_divide_them():
    with pytest.raises(ValueError, match="multiple of block_m"):
        layout.ag_config(world_size=8, m=2048 + 64, n=2048)
    with pytest.raises(ValueError, match="must divide the 16 row tiles"):
        layout.ag_config(**MODEL, counter_chunks=5)


def test_counter_region_is_sized_by_chunks_alone():
    """One counter per chunk, not per (destination, chunk).

    A chunk completing arms every destination's push at once, because they all
    receive the same bytes. gemm_a2a needs ``world`` times as many because a
    chunk of destination 0 says nothing about destination 1.
    """
    c = layout.ag_config(**MODEL, counter_chunks=4)
    assert c.counter_set_bytes == 4 * 4
    a2a_equivalent = 8 * 4 * 4
    assert c.counter_set_bytes * 8 == a2a_equivalent


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(world_size=1, m=2048, n=2048), "world_size must be"),
        (dict(world_size=9, m=2048, n=2048), "world_size must be"),
        (dict(world_size=8, m=0, n=2048), "must be positive"),
        (dict(world_size=8, m=2048, n=2048, block_n=512), "block_n must be exactly"),
        (dict(world_size=8, m=2048, n=2048, block_m=64), "multiple of 128"),
        (dict(world_size=8, m=2048, n=2048, counter_chunks=0), "counter_chunks"),
        (
            dict(world_size=8, m=2048, n=2048, counter_chunks=4, counter_capacity=2),
            "counter_capacity",
        ),
        (dict(world_size=8, m=2048, n=2048, capacity_m=1024), "capacity_m"),
        (dict(world_size=8, m=2048, n=2048, force_blocks=999), "force_blocks"),
    ],
)
def test_validate_rejects(kwargs, match):
    with pytest.raises(ValueError, match=match):
        layout.ag_config(**kwargs)


def test_capacity_m_pins_the_offsets_so_two_shapes_cannot_alias():
    """Two ``m`` sharing a window must share a map, or one aliases the other."""
    big = layout.ag_config(world_size=8, m=4096, n=2048)
    small = layout.ag_config(world_size=8, m=1024, n=2048, capacity_m=4096)
    for src in range(8):
        assert small.recv_slot_off(src) == big.recv_slot_off(src)
    assert small.window_bytes == big.window_bytes
    # Sizes still follow the real m: this moves where things are, not how much
    # is moved.
    assert small.slab_bytes == 1024 * 2048 * 2
    assert small.remote_bytes_per_rank == 7 * small.slab_bytes


def test_counter_shape_slots_are_disjoint():
    """Two shapes sharing a set would elect on each other's leftovers."""
    offs = []
    for i in range(3):
        c = layout.ag_config(
            **MODEL, counter_chunks=4, counter_shape_slots=3, counter_shape_index=i
        )
        offs.append((c.counter_off, c.counter_set_bytes))
        assert c.counter_off + c.counter_set_bytes <= c.lock_off
    for (ao, asz), (bo, _) in zip(offs, offs[1:]):
        assert ao + asz <= bo


@pytest.mark.parametrize(
    "m_tiles, requested, want",
    [(16, 8, 8), (16, 5, 4), (16, 32, 16), (12, 8, 6), (7, 4, 1), (16, 1, 1)],
)
def test_counter_chunks_rounds_down_to_a_divisor(m_tiles, requested, want):
    assert layout.counter_chunks(m_tiles, requested) == want


def test_counter_chunks_reports_the_shape_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="m_tiles must be"):
        layout.counter_chunks(0, 8)
    with pytest.raises(ValueError, match="requested chunks"):
        layout.counter_chunks(16, 0)


# --------------------------------------------------------------------------
# multi-rank: the modes agree
# --------------------------------------------------------------------------


#: Back-to-back distributed launches outrun SDMA queue teardown: the next
#: process then fails in hsaKmtCreateQueueExt (anvil.cpp:237), or aborts in the
#: bootstrap allgather that follows it. gemm_ar's equivalent file was written
#: without this and three of its cases went red on main; do not remove.
_SETTLE_SECONDS = 20
_QUEUE_EXHAUSTED = "anvil.cpp"


def _run_bench(world_size, mode, m, n, k, extra=()):
    """One benchmark process group, retried once if its queues were not free."""
    result = _spawn_bench(world_size, mode, m, n, k, extra)
    if result is None:
        time.sleep(_SETTLE_SECONDS * 3)
        result = _spawn_bench(world_size, mode, m, n, k, extra, last=True)
    return result


def _spawn_bench(world_size, mode, m, n, k, extra=(), last=False):
    """One launch. Returns None if it lost the queue race and may be retried."""
    time.sleep(_SETTLE_SECONDS)
    env = os.environ.copy()
    env.setdefault("MORI_SOCKET_IFNAME", "lo")
    env.setdefault("MORI_ENABLE_SDMA", "1")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        str(BENCH),
        "--mode",
        mode,
        "-m",
        str(m),
        "--out-dim",
        str(n),
        "-k",
        str(k),
        "--warmup",
        "1",
        "--iters",
        "3",
        *extra,
    ]
    result = subprocess.run(
        command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=1200
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        if _QUEUE_EXHAUSTED in output and not last:
            return None
        raise AssertionError(output)
    records = [
        json.loads(line.removeprefix("RESULT_JSON "))
        for line in output.splitlines()
        if line.startswith("RESULT_JSON ")
    ]
    assert len(records) == 1, output
    return records[0]


@pytest.mark.parametrize(
    "mode",
    [
        "gemm-only",
        "gemm-to-window",
        "split-rccl",
        "split-lsa-push",
        "split-lsa-pull",
        "fused-lsa",
        "split-sdma",
        "fused-sdma",
    ],
)
@pytest.mark.parametrize("quant", ["ptpc", "blockscale", "mxfp8"])
def test_every_mode_validates_under_every_quant(mode, quant):
    """All eight paths must land on the same all-gathered result.

    Parametrised over the quantisation as well as the mode, and that is not
    thoroughness for its own sake: ``split-lsa-pull`` shipped without the
    producer's release fence and validated at 1.66e-3 under ptpc while giving
    2.05e-1 under blockscale. The blockscale GEMM is 96us against ptpc's 57, so
    the rank skew is wider and the race window with it. Spot-checking the
    default quantisation would have passed.

    ``mxfp8`` is here for the mirror-image reason: it forces ``BLOCK_M=256`` for
    the packed A scale, and getting the MFMA's ``opsel`` wrong leaves
    ``gemm-only`` correct -- it is gemm_ar's own kernel -- while every fused
    path reads byte 0 for all four tiles.
    """
    if torch.cuda.device_count() < 8:
        pytest.skip("requires 8 GPUs")
    pytest.importorskip("flydsl")
    extra = ("--quant", quant)
    record = _run_bench(8, mode, 2048, 2048, 7168, extra)
    assert record["validated"] is True, record
    assert record["rel_l2"] < 3e-3, record
    assert record["us"] > 0


def test_fused_sdma_is_deterministic_across_chunk_counts():
    """Repeat the chunked epilogue: a race here is intermittent, not absent.

    gemm_ar shipped two real races behind a one-shot check that passed
    repeatedly -- a shared SDMA queue when a destination has more than one
    chunk, and a counter atomic that had lost its acquire half. The same tail is
    reused here, with the election moved from (dest, chunk) to chunk and a
    single submit lock in place of one per destination, so it has to be shown
    clean rather than assumed to be.
    """
    if torch.cuda.device_count() < 8:
        pytest.skip("requires 8 GPUs")
    pytest.importorskip("flydsl")
    seen = []
    for chunks in ("1", "2", "4", "8"):
        record = _run_bench(8, "fused-sdma", 2048, 2048, 7168, ("--chunks", chunks))
        seen.append(record["rel_l2"])
    assert all(v == pytest.approx(seen[0], rel=1e-6) for v in seen), (
        f"fused all-gather is not deterministic across chunk counts: {seen}. "
        "That is an ordering bug in the epilogue, not numerical noise."
    )
    assert all(v < 3e-3 for v in seen), seen
