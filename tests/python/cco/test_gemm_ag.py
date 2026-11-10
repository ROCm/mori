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
"""Everything about ``mori.ops.gemm_ag`` that needs no process group.

* the window layout and the tile / chunk pickers -- pure arithmetic;
* two models that execute the real kernel bodies: the bf16 pipeline's LDS
  happens-before edges, and the SDMA phases' completion and credit order;
* the benchmark's validation contract;
* the single-GPU GEMM numerics against exact small-integer references;
* the benchmark's two baselines on 8 ranks.

``GemmAgOp`` on several ranks -- every precision, mode, tile and race -- is
``test_gemm_ag_op.py``, from one spawn.
"""

import ast
from dataclasses import dataclass
from functools import lru_cache
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from mori.ops.gemm_ag import layout
from mori.ops.gemm_ag import _gemm_a16w16_8wave as bf16gemm
from mori.ops.gemm_ar import _gemm_a8w8_8wave as fp8gemm

from . import kfd_settle

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


def test_kernels_read_no_config_object():
    """Config a kernel body reads must reach it as scalars.

    FlyDSL keys its compile cache on a kernel's scalar closure values; an
    ``AgConfig`` read inside the body is not part of the key, so two configs
    differing only there -- the window capacity behind every receive-slot
    offset, say -- could share a cached kernel with the wrong offsets. gemm_ar
    pinned the same rule in #748; the since-removed LSA copy kernel broke it.
    """
    import re

    root = REPO_ROOT / "python/mori/ops/gemm_ag"
    for name in (
        "kernels_fused.py",
        "kernels_sdma.py",
        "_gemm_a16w16_8wave.py",
    ):
        src = (root / name).read_text()
        bodies = 0
        for m in re.finditer(r"\n( +)@flyc\.kernel", src):
            end = src.index(f"\n{m.group(1)}@flyc.jit", m.end())
            assert not re.findall(r"\bcfg\.\w+", src[m.end() : end]), name
            bodies += 1
        assert bodies, f"no @flyc.kernel found in {name}"


def test_counter_region_is_one_slot_per_chunk_and_lane():
    """``world`` counters per chunk, sized for the *posting*, not the counting.

    A broadcast chunk arms every destination at once, so one counter per chunk
    would decide correctly -- and then the winning thread would have to tell
    ``world-2`` other lanes, which costs a barrier and an LDS round trip for
    one dword. A counter per (chunk, lane) lets each lane learn its own
    election, which is what makes the ``world-1`` packets issue at once.
    """
    c = layout.ag_config(**MODEL, counter_chunks=4)
    assert c.counter_set_bytes == 8 * 4 * 4
    # Slot `chunk * world + lane` has to stay inside the set.
    last = (4 - 1) * 8 + (8 - 1)
    assert c.counter_off + (last + 1) * 4 <= c.lock_off


@pytest.mark.parametrize("elem_bytes", [2, 4])
def test_layout_holds_at_both_output_widths(elem_bytes):
    """fp32 output doubles every payload offset; nothing may overlap or slip.

    The control regions are fixed-size and the payload is not, so this is where
    an fp32 C would show up as an aliased counter or an unaligned recv slot.
    """
    c = layout.ag_config(**MODEL, elem_bytes=elem_bytes, counter_chunks=4)
    assert c.slab_bytes == 2048 * 2048 * elem_bytes
    assert c.remote_bytes_per_rank == 7 * c.slab_bytes
    for name in ("start_off", "flag_off", "counter_region_off", "lock_off", "recv_off"):
        assert getattr(c, name) % layout.SIGNAL_ALIGN == 0, name
    assert c.lock_off + c.lock_bytes <= c.recv_off
    for src in range(8):
        off = c.recv_slot_off(src)
        assert off % layout.PACK_BYTES == 0
        assert off + c.slab_bytes <= c.window_bytes
    # A slab must be a whole number of 16B packs at either width.
    assert c.slab_bytes % layout.PACK_BYTES == 0


def test_elem_bytes_must_divide_a_pack():
    with pytest.raises(ValueError, match="elem_bytes must be"):
        layout.ag_config(**MODEL, elem_bytes=1)
    with pytest.raises(ValueError, match="elem_bytes must be"):
        layout.ag_config(**MODEL, elem_bytes=8)


def test_byte_swizzle_reduces_to_the_fp8_one():
    """``swizzle_row128b`` is ``swizzle_128`` in bytes, and must prove it.

    The bf16 GEMM reuses gemm_ar's LDS swizzle at a different element width.
    Restating it in bytes is only safe if it is the *same function* at
    ``elem_bytes=1``, where an fp8 element is a byte -- so that is asserted
    rather than assumed.
    """
    for row in range(64):
        for col in range(128):
            assert bf16gemm.swizzle_row128b(row, col, 1) == fp8gemm.swizzle_128(
                row, col
            )


def test_byte_swizzle_is_a_bijection_at_bf16_width():
    """A swizzle that collided would silently drop LDS rows."""
    seen = set()
    for row in range(16):
        for col in range(64):
            r, c = bf16gemm.swizzle_row128b(row, col, 2)
            assert r == row, "the XOR must stay inside its 128-byte row"
            assert 0 <= c < 64
            seen.add((r, c))
    assert len(seen) == 16 * 64


def test_pick_tile_matches_the_measured_best():
    """Each shape's fastest tile from the 2026-10-29 sweep, split and fused.

    The split rule takes the largest tile whose grid covers the 256 CUs; the
    fused one needs at least two workgroup rounds to overlap anything, and
    pairs the tile with one chunk per round. See ``pick_tile``'s tables.
    """
    split = {
        (2048, 2048): (128, 128),
        (4096, 2048): (128, 256),
        (8192, 2048): (256, 256),
        (16384, 2048): (256, 256),
        (2048, 1024): (128, 128),
        (16384, 1024): (256, 256),
    }
    for (m, n), tile in split.items():
        assert bf16gemm.pick_tile(m, n, "fp32") == tile, (m, n)
    fused = {
        (8192, 2048): ((128, 256), 2),
        (16384, 2048): ((128, 256), 4),
        (16384, 1024): ((128, 256), 2),
    }
    for (m, n), (tile, chunks) in fused.items():
        assert bf16gemm.pick_tile(m, n, "fp32", fused=True) == tile, (m, n)
        assert bf16gemm.pick_chunks(m, n, tile) == chunks, (m, n)
    # BLOCK_N=128 stays fp32-only: a bf16 C never gets it.
    for m in (2048, 4096, 8192, 16384):
        for fused_ in (False, True):
            assert bf16gemm.pick_tile(m, 2048, "bf16", fused=fused_)[1] == 256
    assert bf16gemm.tile_grid(8192, 2048, (256, 256)) == 256
    # A chunk count has to divide the row tiles.
    assert bf16gemm.pick_chunks(384, 2048, (128, 128), cus=8) == 3


def test_bf16_tile_constants_match_the_template():
    """``<512, 256, 256, 64>`` from gemm_a16w16_quad_subtile_kernel.cc."""
    tc = bf16gemm.tile_constants(256, 256)
    assert bf16gemm.BLOCK_K == 64
    assert bf16gemm.E_K == 2  # B_K / W_K
    assert bf16gemm.VEC == 8  # 16B / 2B
    assert tc["N_TILES_A"] == 4 and tc["N_TILES_B"] == 2  # E_M, E_N
    assert tc["LDS_BLOCK_M"] == 128 and tc["LDS_BLOCK_N"] == 128  # HALF_B_M/N
    # The finer-chunking variant, which the template does not instantiate.
    tc128 = bf16gemm.tile_constants(128, 256)
    assert tc128["N_TILES_A"] == 2 and tc128["N_LDS_STEPS_A"] == 1
    for bad in (64, 192, 512):
        with pytest.raises(ValueError, match="BLOCK_M must be"):
            bf16gemm.tile_constants(bad, 256)
    with pytest.raises(ValueError, match="BLOCK_N must be"):
        bf16gemm.tile_constants(256, 512)
    # BLOCK_N=128 doubles the grid again but only an fp32 store can take it:
    # the bf16 one pairs two N tiles through permlane16_swap.
    assert bf16gemm.tile_constants(128, 128, "fp32")["N_TILES_B"] == 1
    with pytest.raises(ValueError, match="BLOCK_N=128 needs"):
        bf16gemm.tile_constants(128, 128, "bf16")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(world_size=1, m=2048, n=2048), "world_size must be"),
        (dict(world_size=9, m=2048, n=2048), "world_size must be"),
        (dict(world_size=8, m=0, n=2048), "must be positive"),
        (dict(world_size=8, m=2048, n=2048, block_n=512), "block_n must be 128 or"),
        # 128 is layout-legal but only under an fp32 C.
        (
            dict(world_size=8, m=2048, n=2048, block_n=128),
            "block_n=128 needs elem_bytes=4",
        ),
        (dict(world_size=8, m=2048, n=2048, block_m=64), "multiple of 128"),
        (dict(world_size=8, m=2048, n=2048, counter_chunks=0), "counter_chunks"),
        (
            dict(world_size=8, m=2048, n=2048, counter_chunks=4, counter_capacity=2),
            "counter_capacity",
        ),
        (dict(world_size=8, m=2048, n=2048, capacity_m=1024), "capacity_m"),
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
# the benchmark's own baselines, on 8 ranks
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
    result = kfd_settle.run(
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
    "bm,bn,out_dtype",
    [
        (128, 128, "fp32"),
        (256, 128, "fp32"),
        (128, 256, "fp32"),
        (256, 256, "fp32"),
        (128, 256, "bf16"),
        (256, 256, "bf16"),
    ],
)
@pytest.mark.parametrize("k", [128, 192, 256, 7168])
def test_bf16_gemm_pipeline_numerics(bm, bn, out_dtype, k):
    """Exercise both tail parities, the empty mainloop, and graph replay.

    Small integer operands make FP32 accumulation exact. Build the oracle on
    the CPU before launching any GEMM, and change inputs between replays so a
    stale LDS tile cannot match the previous launch's answer by accident.
    No CCO window, communicator, or GPU reference GEMM is involved.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires a gfx950 GPU")
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950 BF16 MFMA")
    import flydsl.expr as fx

    m, n = 2 * bm, 2 * bn
    dtype = torch.float32 if out_dtype == "fp32" else torch.bfloat16
    cfg = layout.ag_config(
        world_size=8,
        m=m,
        n=n,
        elem_bytes=4 if out_dtype == "fp32" else 2,
        block_m=bm,
        block_n=bn,
    )
    gemm = bf16gemm.compile_bf16_gemm_ag(
        cfg,
        0,
        K=k,
        BLOCK_M=bm,
        BLOCK_N=bn,
        out_dtype=out_dtype,
    )
    a = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.empty((n, k), dtype=torch.bfloat16, device="cuda")
    c = torch.empty((m, n), dtype=dtype, device="cuda")
    a_arg, b_arg, c_arg = (
        a.view(torch.int16).view(-1),
        b.view(torch.int16).view(-1),
        c.view(-1),
    )

    def launch():
        gemm(
            a_arg,
            b_arg,
            c_arg,
            m,
            n,
            0,
            0,
            stream=fx.Stream(torch.cuda.current_stream()),
        )

    generator = torch.Generator().manual_seed(1234)
    graph = None
    for generation in range(4):
        ah = torch.randint(-2, 3, (m, k), generator=generator).float()
        bh = torch.randint(-2, 3, (n, k), generator=generator).float()
        expected = (ah @ bh.T).to(dtype)
        a.copy_(ah)
        b.copy_(bh)
        c.fill_(float("nan"))
        if graph is None:
            launch()
        else:
            graph.replay()
        torch.testing.assert_close(c.cpu(), expected, rtol=0, atol=0)
        if generation == 0:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"split_k": 3}, "split_k must"),
        ({"split_k": 4, "fuse": True}, "local, unfused FP32"),
        ({"split_k": 4, "peer_uncached": True}, "local, unfused FP32"),
        ({"split_k": 4, "out_dtype": "bf16"}, "local, unfused FP32"),
        ({"split_k": 4, "K": 320}, "divisible"),
    ],
)
def test_split_k_rejects_incompatible_output(kwargs, match):
    cfg = layout.ag_config(
        world_size=8, m=256, n=256, elem_bytes=4, block_m=128, block_n=128
    )
    options = dict(K=7168, BLOCK_M=128, BLOCK_N=128, out_dtype="fp32")
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        bf16gemm.compile_bf16_gemm_ag(cfg, 0, **options)


@pytest.mark.parametrize("bm,bn", [(128, 128), (256, 256)])
@pytest.mark.parametrize("splits,k", [(2, 256), (4, 768), (8, 7168)])
def test_split_k_partials_and_changed_input_graph(bm, bn, splits, k):
    """Check every partial plane against a CPU oracle, including row strides."""
    if not torch.cuda.is_available():
        pytest.skip("requires a gfx950 GPU")
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950 BF16 MFMA")
    import flydsl.expr as fx

    m, n = 2 * bm, 2 * bn
    cfg = layout.ag_config(world_size=8, m=m, n=n, elem_bytes=4, block_m=bm, block_n=bn)
    gemm = bf16gemm.compile_bf16_gemm_ag(
        cfg, 0, K=k, BLOCK_M=bm, BLOCK_N=bn, out_dtype="fp32", split_k=splits
    )
    a = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.empty((n, k), dtype=torch.bfloat16, device="cuda")
    partials = torch.empty((splits, m, n), dtype=torch.float32, device="cuda")
    a_arg = a.view(torch.int16).view(-1)
    b_arg = b.view(torch.int16).view(-1)
    c_arg = partials.view(-1)

    def launch():
        gemm(
            a_arg,
            b_arg,
            c_arg,
            m,
            n,
            0,
            0,
            stream=fx.Stream(torch.cuda.current_stream()),
        )

    generator = torch.Generator().manual_seed(9028)
    graph = None
    for _ in range(2):
        ah = torch.randint(-2, 3, (m, k), generator=generator).float()
        bh = torch.randint(-2, 3, (n, k), generator=generator).float()
        step = k // splits
        expected = torch.stack(
            [
                ah[:, i * step : (i + 1) * step] @ bh[:, i * step : (i + 1) * step].T
                for i in range(splits)
            ]
        )
        a.copy_(ah)
        b.copy_(bh)
        partials.fill_(float("nan"))
        torch.cuda.synchronize()
        if graph is None:
            launch()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch()
        else:
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(partials.cpu(), expected)
        assert torch.equal(partials.sum(dim=0).cpu(), ah @ bh.T)


@pytest.mark.parametrize("mode", ["gemm-only", "split-rccl"])
def test_the_benchmark_baselines_validate(mode):
    """The two baselines only the benchmark has -- the GEMM alone, and it
    followed by RCCL's all-gather. Every op mode is covered by
    test_gemm_ag_op.py; this keeps the benchmark itself honest."""
    if torch.cuda.device_count() < 8:
        pytest.skip("requires 8 GPUs")
    pytest.importorskip("flydsl")
    record = _run_bench(8, mode, 2048, 2048, 7168)
    assert record["validated"] is True, record
    assert record["rel_l2"] < 3e-3, record


# --------------------------------------------------------------------------
# bf16 pipeline: LDS happens-before (no GPU)
#
# Check the BF16 pipeline's happens-before edges without a GPU.
#
# Execute the actual kernel's scheduling statements with tracing loaders. Memory
# operations may complete as late as their first required wait. Barrier ordinals,
# not source locations, determine inter-wave ordering: wave_m=1 starts one
# barrier behind wave_m=0. This catches races that a hot numerical run can hide.
# The model assumes FIFO completion of G2S loads, as the vmcnt arithmetic does.
# It does not model instruction latency or replace ISA/numerical validation.
# --------------------------------------------------------------------------


BF16_KERNEL = (
    Path(__file__).resolve().parents[3]
    / "python/mori/ops/gemm_ag/_gemm_a16w16_8wave.py"
)


@dataclass(frozen=True)
class Position:
    wave: int
    phase: int
    order: int

    def before(self, other):
        return self.phase < other.phase or (
            self.wave == other.wave and self.order < other.order
        )


@dataclass
class Access:
    buffer: str
    tile: int
    rows: range
    issued: Position
    done: Position | None = None


class Wave:
    def __init__(self, wave, bm, bn):
        self.wave = wave
        self.bm, self.bn = bm, bn
        self.phase = self.order = 0
        self.writes = []
        self.reads = []
        self.generations = {}

    def position(self):
        self.order += 1
        return Position(self.wave, self.phase, self.order)

    def barrier(self):
        self.phase += 1

    def wait_vm(self, count):
        pos = self.position()
        for write in self.writes[: len(self.writes) - count if count else None]:
            if write.done is None:
                write.done = pos
        self.barrier()

    def wait_lds(self):
        pos = self.position()
        for read in self.reads:
            if read.done is None:
                read.done = pos

    def write(self, buffer, offset):
        rows = (self.bm if buffer[0] == "a" else self.bn) // 2
        for step in range(rows // 64):
            start = self.wave * 8 + step * 64
            self.writes.append(
                Access(buffer, offset // 64, range(start, start + 8), self.position())
            )

    def read(self, buffer):
        if buffer[0] == "a":
            size, part = self.bm // 4, self.wave // 4
        else:
            size, part = self.bn // 8, self.wave % 4
        tile = self.generations.get(buffer, int(buffer[1]))
        self.generations[buffer] = tile + 2
        read = Access(
            buffer, tile, range(part * size, (part + 1) * size), self.position()
        )
        self.reads.append(read)
        return read

    def mma(self, a, b, c, set_prio=True):
        # LLVM must wait for the operand registers before their first use.
        pos = self.position()
        for read in (a, b):
            if read.done is None:
                read.done = pos
        if set_prio:
            self.barrier()
        return c


def trace_pipeline(source, bm, bn, k_iters, policy):
    tree = ast.parse(source)
    factory = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "compile_bf16_gemm_ag"
    )
    kernel = next(
        n
        for n in factory.body
        if isinstance(n, ast.FunctionDef) and n.name == "kernel_gemm"
    )
    # Extract the real prologue, mainloop, tail, and closing half-wave barrier.
    start = next(
        i for i, n in enumerate(kernel.body) if ast.unparse(n).startswith("b_g2s.load(")
    )
    end = next(
        i
        for i, n in enumerate(kernel.body)
        if ast.unparse(n).startswith("store_c.store(")
    )
    code = compile(ast.Module(kernel.body[start:end], []), str(BF16_KERNEL), "exec")
    main_wait = next(
        n
        for n in factory.body
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "_MAIN_WAIT"
    )
    wait_code = compile(ast.Module([main_wait], []), str(BF16_KERNEL), "exec")
    waves = []
    for wave_id in range(8):
        wave = Wave(wave_id, bm, bn)
        env = dict(
            K_ITERS=k_iters,
            split_k=1,
            const_expr=lambda value: value,
            BLOCK_K=64,
            B_K_STEP=64,
            BLOCK_M=bm,
            BLOCK_N=bn,
            N_TILES_A=bm // 64,
            N_TILES_B=bn // 128,
            LDS_BLOCK_M=bm // 2,
            LDS_BLOCK_N=bn // 2,
            N_LDS_STEPS_A=bm // 128,
            N_LDS_STEPS_B=bn // 128,
            A0_gl_offset=0,
            A1_gl_offset=0,
            B0_gl_offset=0,
            B1_gl_offset=0,
            wave_m=wave_id // 4,
            wave_n=wave_id % 4,
            block_m=0,
            block_n=0,
            c00_frag=None,
            c01_frag=None,
            c10_frag=None,
            c11_frag=None,
            range_constexpr=range,
            wait_policy=policy,
            a_cur0="a00",
            a_cur1="a01",
            a_next0="a10",
            a_next1="a11",
            b_cur0="b00",
            b_cur1="b01",
            b_next0="b10",
            b_next1="b11",
            a_g2s=SimpleNamespace(load=wave.write),
            b_g2s=SimpleNamespace(load=wave.write),
            a_s2r=SimpleNamespace(load=wave.read),
            b_s2r=SimpleNamespace(load=wave.read),
            mfma=SimpleNamespace(call=wave.mma),
            rocdl=SimpleNamespace(s_barrier=wave.barrier, s_setprio=lambda _: None),
            _wait_lds_reads=wave.wait_lds,
            wait_barrier=wave.wait_vm,
            _wb=lambda count, wave=wave: wave.wait_vm(
                0 if policy == "conservative" else count
            ),
        )
        exec(wait_code, env)  # noqa: S102 - trace trusted local kernel source
        exec(code, env)  # noqa: S102 - trace trusted local kernel source
        waves.append(wave)
    assert len({w.phase for w in waves}) == 1, "unbalanced half-wave barriers"
    return waves


def hazards(waves):
    writers = {}
    readers = {}
    for wave in waves:
        for access in wave.writes:
            for row in access.rows:
                key = access.buffer, access.tile, row
                assert key not in writers, f"multiple writers: {key}"
                writers[key] = access
        for access in wave.reads:
            for row in access.rows:
                readers.setdefault((access.buffer, access.tile, row), []).append(access)
    errors = set()
    for key, reads in readers.items():
        writer = writers[key]
        overwrite = writers.get((key[0], key[1] + 2, key[2]))
        for read in reads:
            if writer.done is None or not writer.done.before(read.issued):
                errors.add(f"RAW {key[0]} k={key[1]}")
            if overwrite and (
                read.done is None or not read.done.before(overwrite.issued)
            ):
                errors.add(f"WAR {key[0]} k={key[1]}")
    return sorted(errors)


@pytest.mark.parametrize("bm,bn", [(128, 128), (128, 256), (256, 128), (256, 256)])
@pytest.mark.parametrize("k_iters", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("policy", ["tuned", "safe", "conservative"])
def test_bf16_pipeline_orders_lds_accesses(bm, bn, k_iters, policy):
    assert not hazards(trace_pipeline(BF16_KERNEL.read_text(), bm, bn, k_iters, policy))


def test_model_detects_missing_b0_read_completion():
    # Mutation control: removing the wait must expose the original WAR even
    # with correct vmcnt thresholds and all barriers still in place.
    source = BF16_KERNEL.read_text().replace("            _wait_lds_reads()\n", "")
    errors = hazards(trace_pipeline(source, 128, 256, 4, "tuned"))
    assert any(e.startswith("WAR b") for e in errors), errors


def test_model_detects_late_tail_wait():
    # Move the drain back after a0's final S2R, as in the broken tail.
    source = BF16_KERNEL.read_text()
    start = source.index("        # Step k = K_ITERS - 2")
    head, tail = source[:start], source[start:]
    tail = tail.replace("        wait_barrier(0)\n", "        rocdl.s_barrier()\n", 1)
    penultimate, final = tail.split("        # Step k = K_ITERS - 1", 1)
    final = final.replace(
        "        a0_frag = a_s2r.load(a_cur0)\n        rocdl.s_barrier()",
        "        a0_frag = a_s2r.load(a_cur0)\n        wait_barrier(0)",
        1,
    )
    tail = penultimate + "        # Step k = K_ITERS - 1" + final
    errors = hazards(trace_pipeline(head + tail, 128, 256, 4, "tuned"))
    assert any(e.startswith(("RAW b", "RAW a")) for e in errors), errors


def test_model_detects_permissive_asymmetric_wait():
    source = BF16_KERNEL.read_text()
    source = source.replace(
        "else min(\n            2 * N_LDS_STEPS_A + N_LDS_STEPS_B,\n"
        "            N_LDS_STEPS_A + 2 * N_LDS_STEPS_B,\n        )",
        "else 2 * N_LDS_STEPS_A + N_LDS_STEPS_B",
    )
    errors = hazards(trace_pipeline(source, 256, 128, 4, "tuned"))
    assert any(e.startswith("RAW a") for e in errors), errors


# --------------------------------------------------------------------------
# SDMA phases: completion and credit order (no GPU)
#
# Check that an SDMA completion publication covers every submitted queue.
#
# Execute the real kernel body against a queue-state model. This checks the
# publication contract; it does not simulate GPU scheduling or DMA latency.
# --------------------------------------------------------------------------


SDMA_KERNEL = (
    Path(__file__).resolve().parents[3] / "python/mori/ops/gemm_ag/kernels_sdma.py"
)


@lru_cache(maxsize=1)
def drain_body():
    tree = ast.parse(SDMA_KERNEL.read_text())
    factory = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "build_sdma_phases"
    )
    builder = next(
        n
        for n in factory.body
        if isinstance(n, ast.FunctionDef) and n.name == "_push_kernel"
    )
    kernel = next(
        n for n in builder.body if isinstance(n, ast.FunctionDef) and n.name == "push"
    )
    return compile(ast.Module(kernel.body, []), str(SDMA_KERNEL), "exec")


@pytest.mark.parametrize("queues", [1, 2, 4, 8])
@pytest.mark.parametrize("pushes", [False, True])
@pytest.mark.parametrize("rank", [0, 3, 7])
def test_arrival_covers_all_submitted_queues_and_puts_wait_for_credit(
    queues, pushes, rank
):
    world = 8
    # The split kernel submits one queue per peer. A chunked fused producer
    # can leave work in every queue, with no ordering between those queues.
    pending = {
        peer: set() if pushes or peer == rank else set(range(queues))
        for peer in range(world)
    }
    published = set()
    credited = set()

    class Sdma:
        def put(self, peer, src_win, src_off, dst_win, dst_off, size, qid, **kwargs):
            # The recv-reuse credit: a push into a peer's recv only after that
            # peer has started this call.
            assert peer in credited, f"put to peer {peer} before its credit"
            pending[peer].add(qid)

        def quiet_queue(self, peer, qid):
            pending[peer].discard(qid)

        def quiet(self, peer, **kwargs):
            pending[peer].clear()

    sdma = Sdma()

    def publish(window, flag, peer):
        assert not pending[
            peer
        ], f"arrival for peer {peer} preceded queues {sorted(pending[peer])}"
        published.add(peer)

    for tid in range(world):
        env = dict(
            dev_comm=0,
            win=0,
            rank=rank,
            ws=world,
            queues=queues,
            pushes=pushes,
            my_recv_slot=0,
            slab_bytes=16 << 20,
            signal=False,
            fx=SimpleNamespace(thread_idx=SimpleNamespace(x=tid), Int32=int, Int64=int),
            cco=SimpleNamespace(
                CachedWindow=lambda _: SimpleNamespace(base=0, stride=0),
                DevComm=lambda _: SimpleNamespace(sdma=lambda: sdma),
                CoopScope=SimpleNamespace(THREAD=0),
            ),
            const_expr=bool,
            _next_flag=lambda _: (1, None),
            raw_cco=SimpleNamespace(cco_system_fence=lambda _: None),
            _signal_and_wait=publish,
            fgpu=SimpleNamespace(barrier=lambda: None),
            local_store_u32=lambda *args: None,
            war_credit=True,
            credit_off=0,
            lsa=lambda *args: 0,
            post_credit=lambda *args: None,
            wait_credit=lambda base, peer, epoch, spin: credited.add(peer),
            _spin_until=None,
        )
        exec(drain_body(), env)  # noqa: S102 - execute trusted local kernel source
    assert published == set(range(world))


# --------------------------------------------------------------------------
# the benchmark's validation contract (no GPU)
#
# Numerical validation contracts for GEMM + all-gather benchmarks.
# --------------------------------------------------------------------------


@pytest.fixture
def bench(monkeypatch):
    pytest.importorskip("flydsl")
    path = Path(__file__).resolve().parents[3] / "benchmark/cco/flydsl/gemm_ag"
    monkeypatch.syspath_prepend(str(path))
    module = importlib.import_module("bench_gemm_ag")
    monkeypatch.setattr(
        module,
        "_operands",
        lambda rank, args: (torch.eye(2) * (rank + 1), torch.eye(2), None, None),
    )
    monkeypatch.setattr(module, "_reference", lambda a, b, sa, sb, args: a @ b.T)
    return module


def test_recv_validation_rejects_a_nonfinite_peer(bench):
    args = SimpleNamespace(m=2, tolerance=1e-5)
    recv = torch.cat([torch.eye(2), 2 * torch.eye(2)])
    assert bench._validate_recv(recv, args, 0, 2)[1]
    recv[2, 0] = float("nan")
    error, valid = bench._validate_recv(recv, args, 0, 2)
    assert not valid and error == float("inf")


def test_recv_validation_checks_changed_input_sign(bench):
    args = SimpleNamespace(m=2, tolerance=1e-5)
    recv = -torch.cat([torch.eye(2), 2 * torch.eye(2)])
    assert bench._validate_recv(recv, args, 0, 2, sign=-1)[1]
    assert not bench._validate_recv(recv, args, 0, 2, sign=1)[1]
