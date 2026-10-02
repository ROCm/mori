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
"""Numerics and layout math for ``mori.ops.gemm_a2a``.

The layout half is pure arithmetic and needs no GPU, which is the point: the
all-to-all's index map is where a mistake is invisible at runtime -- wrong data
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

from mori.ops.gemm_a2a import layout

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH = REPO_ROOT / "benchmark" / "cco" / "flydsl" / "gemm_a2a" / "bench_gemm_a2a.py"

# The shape gcnasm's opus_gemm_a2a_lsa reports its 8-rank numbers on, and the
# one this op is aimed at. shard_n = 18432/8 = 2304 = 9 * 256.
MODEL = dict(world_size=8, m=2048, n=18432)


# --------------------------------------------------------------------------
# layout (no GPU)
# --------------------------------------------------------------------------


def test_shard_and_slab_sizes_at_the_model_shape():
    c = layout.a2a_config(**MODEL)
    assert c.shard_n == 2304
    assert c.n_blocks_per_peer == 9  # 2304 / 256
    assert c.m_tiles == 16  # 2048 / 128
    assert c.tiles_per_peer == 16 * 9
    assert c.slab_elems == 2048 * 2304
    # Every rank sends world-1 slabs and keeps one.
    assert c.remote_bytes_per_rank == 7 * c.slab_bytes


def test_recv_region_is_the_whole_output_and_staging_is_opt_in():
    direct = layout.a2a_config(**MODEL)
    staged = layout.a2a_config(**MODEL, staged=True)

    assert direct.staging_bytes == 0
    assert staged.staging_bytes == 8 * staged.slab_bytes
    # recv is the output either way, and identical in both.
    assert direct.recv_bytes == staged.recv_bytes == 8 * direct.slab_bytes
    # Staging costs its own copy of the payload, which is why it is opt-in:
    # 75 MiB per rank at this shape.
    assert staged.window_bytes - direct.window_bytes == staged.staging_bytes

    with pytest.raises(ValueError, match="no staging region"):
        direct.staging_slot_off(0)


def test_every_region_is_disjoint_and_inside_the_window():
    for staged in (False, True):
        c = layout.a2a_config(**MODEL, staged=staged, counter_chunks=4)
        regions = [
            ("start", c.start_off, c.end_off),
            ("end", c.end_off, c.end_off),
            ("flag", c.flag_off, c.max_blocks * 4),
            ("counters", c.counter_region_off, c.counter_bytes),
            ("locks", c.lock_off, c.lock_bytes),
            ("staging", c.staging_off, c.staging_bytes),
            ("recv", c.recv_off, c.recv_bytes),
        ]
        regions = [r for r in regions if r[2] > 0]
        regions.sort(key=lambda r: r[1])
        for (an, ao, asz), (bn, bo, _) in zip(regions, regions[1:]):
            assert ao + asz <= bo, f"staged={staged}: {an} overruns {bn}"
        last_n, last_o, last_sz = regions[-1]
        assert last_o + last_sz <= c.window_bytes, f"{last_n} past the window"


def test_staging_and_recv_index_are_the_same_map_from_the_two_ends():
    """The all-to-all in one line: what I write for ``d`` is what ``d`` reads for me.

    Cross-checked against ``benchmark/cco/triton/gemm_a2a/layout.py`` on branch
    ``xiangch/triton_gemm_a2a``, which states the same two formulas.
    """
    c = layout.a2a_config(**MODEL)
    for rank in (0, 3, 7):
        for row, col in ((0, 0), (1, 5), (2047, 2303)):
            assert c.staging_index(rank, row, col) == c.recv_index(rank, row, col)
    # ...and it is dense: the slab is exactly covered.
    small = layout.a2a_config(world_size=2, m=128, n=1024)
    seen = {
        small.staging_index(d, r, lc)
        for d in range(2)
        for r in range(small.m)
        for lc in range(small.shard_n)
    }
    assert seen == set(range(2 * small.slab_elems))


def test_dest_of_column_partitions_n_into_equal_runs():
    c = layout.a2a_config(**MODEL)
    assert c.dest_of_column(0) == 0
    assert c.dest_of_column(2303) == 0
    assert c.dest_of_column(2304) == 1
    assert c.dest_of_column(18431) == 7
    with pytest.raises(IndexError):
        c.dest_of_column(18432)
    # Each destination's run starts on an N-tile boundary, which is the rule
    # that lets a GEMM tile belong to exactly one destination.
    for d in range(8):
        assert (d * c.shard_n) % c.block_n == 0


def test_recv_slot_offsets_are_source_major_and_stay_in_region():
    c = layout.a2a_config(**MODEL)
    for src in range(8):
        off = c.recv_slot_off(src)
        assert off == c.recv_off + src * c.slab_bytes
        assert off + c.slab_bytes <= c.recv_off + c.recv_bytes
    with pytest.raises(IndexError):
        c.recv_slot_off(8)


def test_n_must_split_into_whole_n_tiles_per_destination():
    """``n % world == 0`` is not enough, and this is the case that shows why.

    world=8, n=2048+256: 2304/8 = 288 columns each, which is not a multiple of
    the 256-wide GEMM tile -- so a tile would straddle two destinations and its
    store could not stay contiguous.
    """
    with pytest.raises(ValueError, match="whole number of N tiles"):
        layout.a2a_config(world_size=8, m=128, n=2304).validate()
    # The neighbouring legal shape is accepted.
    layout.a2a_config(world_size=8, m=128, n=2048).validate()


def test_m_must_be_whole_row_tiles_and_chunks_must_divide_them():
    with pytest.raises(ValueError, match="multiple of block_m"):
        layout.a2a_config(world_size=8, m=100, n=2048).validate()
    with pytest.raises(ValueError, match="must divide"):
        # 2048/128 = 16 row tiles; 5 does not divide 16.
        layout.a2a_config(world_size=8, m=2048, n=2048, counter_chunks=5).validate()
    layout.a2a_config(world_size=8, m=2048, n=2048, counter_chunks=8).validate()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(world_size=1, m=128, n=2048), "world_size"),
        (dict(world_size=9, m=128, n=2048), "world_size"),
        (dict(world_size=8, m=0, n=2048), "positive"),
        (dict(world_size=8, m=128, n=2048, block_n=128), "block_n"),
        (dict(world_size=8, m=128, n=2048, block_m=64), "block_m"),
        (dict(world_size=8, m=128, n=2048, counter_chunks=0), "counter_chunks"),
        (
            dict(world_size=8, m=256, n=2048, counter_capacity=1, counter_chunks=2),
            "counter_capacity",
        ),
        (dict(world_size=8, m=128, n=2048, counter_shape_index=1), "shape_index"),
        (dict(world_size=8, m=256, n=2048, capacity_m=128), "capacity_m"),
        (dict(world_size=8, m=128, n=2048, force_blocks=999), "force_blocks"),
    ],
)
def test_validate_rejects(kwargs, match):
    with pytest.raises(ValueError, match=match):
        layout.A2aConfig(**kwargs).validate()


def test_capacity_m_pins_the_offsets_so_two_shapes_cannot_alias():
    """The bug this exists to prevent, in the smallest shape that shows it.

    Without pinning, every payload offset is a running sum of m-sized regions, so
    a smaller m puts *its* recv slot for peer 1 exactly where a larger m puts
    part of its own slab -- and one shape reads the other's payload. Pinning to
    the largest m makes every shape's map identical.
    """
    big = layout.a2a_config(world_size=2, m=1024, n=1024, staged=True)
    small_unpinned = layout.a2a_config(world_size=2, m=512, n=1024, staged=True)
    small_pinned = layout.a2a_config(
        world_size=2, m=512, n=1024, staged=True, capacity_m=1024
    )

    assert small_unpinned.recv_off != big.recv_off  # the trap
    assert small_pinned.recv_off == big.recv_off
    assert small_pinned.recv_slot_off(1) == big.recv_slot_off(1)
    assert small_pinned.staging_slot_off(1) == big.staging_slot_off(1)
    # Sizes still follow the real m -- pinning moves where, not how much.
    assert small_pinned.slab_bytes == big.slab_bytes // 2


def test_counter_shape_slots_are_disjoint():
    a, b = (
        layout.a2a_config(
            **MODEL, counter_chunks=4, counter_shape_slots=2, counter_shape_index=i
        )
        for i in (0, 1)
    )
    assert a.counter_off + a.counter_set_bytes <= b.counter_off
    assert b.counter_off + b.counter_set_bytes <= b.lock_off
    # The control area is the same size whichever slot a config uses, so the
    # payload does not move between them.
    assert a.signal_bytes == b.signal_bytes
    assert a.recv_off == b.recv_off


@pytest.mark.parametrize(
    "m_tiles, requested, want",
    [(16, 8, 8), (16, 5, 4), (16, 32, 16), (9, 8, 3), (1, 8, 1), (7, 4, 1)],
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
        "-n",
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
