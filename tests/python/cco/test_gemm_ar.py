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
"""Fused GEMM + all-reduce: anti-rot guard for the pinned kernel copy, plus smoke.

``benchmark/cco/flydsl/gemm_ar/kernels_fused.py`` holds a copy of aiter's
``kernel_gemm`` with an epilogue added. The single most valuable test here is not
that the fused path is correct end to end -- ``bench_gemm_ar.py`` checks that
against a host reference -- but that the *copy* still computes what aiter's
original computes. That one runs on a single GPU and needs no cco at all.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
GEMM_AR_DIR = REPO_ROOT / "benchmark" / "cco" / "flydsl" / "gemm_ar"
AR_DIR = REPO_ROOT / "benchmark" / "cco" / "flydsl" / "ar"
for _d in (str(AR_DIR), str(GEMM_AR_DIR)):
    if _d not in sys.path:
        sys.path.insert(0, _d)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


layout = _load("flydsl_ar_layout_gemm", AR_DIR / "layout.py")


# --------------------------------------------------------------------------
# counter-region layout (no GPU)
# --------------------------------------------------------------------------


def test_counter_region_sizes_with_chunks_and_never_overlaps_payload():
    a = layout.ArConfig(world_size=8, m=4096, n=7168, counter_chunks=1)
    b = layout.ArConfig(world_size=8, m=4096, n=7168, counter_chunks=4)
    a.validate()
    b.validate()
    # own region, past the last _flag slot
    assert a.counter_off >= a.flag_off + a.max_blocks * 4
    assert a.counter_off % layout.SIGNAL_ALIGN == 0
    # grows with chunks, and the payload is pushed out to match
    assert b.counter_bytes >= a.counter_bytes
    assert b.input_off >= a.input_off
    assert a.counter_off + a.counter_bytes <= a.input_off
    assert b.counter_off + b.counter_bytes <= b.input_off


def test_counter_slot_is_dest_major():
    c = layout.ArConfig(world_size=8, m=4096, n=7168, counter_chunks=2)
    assert c.counter_slot(0, 0) == 0
    assert c.counter_slot(0, 1) == 1
    assert c.counter_slot(1, 0) == 2
    assert c.counter_slot(7, 1) == 15
    with pytest.raises(IndexError):
        c.counter_slot(8, 0)
    with pytest.raises(IndexError):
        c.counter_slot(0, 2)


def test_validate_rejects_zero_chunks():
    with pytest.raises(ValueError, match="counter_chunks"):
        layout.ArConfig(world_size=8, m=4096, n=7168, counter_chunks=0).validate()


# --------------------------------------------------------------------------
# the anti-rot guard: our pinned copy must match aiter's original bit-for-bit
# --------------------------------------------------------------------------


@pytest.mark.parametrize("m,n,k", [(512, 512, 256), (1024, 768, 512)])
def test_pinned_copy_matches_aiter_kernel_bitwise(m, n, k):
    """The unfused copy must be indistinguishable from aiter's kernel.

    Bitwise, not approximate: the two are meant to be the same instruction
    sequence over the same inputs, so any drift in the copied pipeline shows up
    here rather than as a mystery in a fused benchmark.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    pytest.importorskip("aiter", reason="aiter not importable (set PYTHONPATH)")
    pytest.importorskip("flydsl")
    from aiter.ops.flydsl.gemm_a8w8_bpreshuffle_8wave import flydsl_8wave_gemm_a8
    from aiter.ops.shuffle import shuffle_weight

    import flydsl.expr as fx

    from kernels_fused import compile_fused_gemm_scatter

    g = torch.Generator(device="cuda").manual_seed(7)
    a = (torch.randn(m, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    sa = torch.rand(m, generator=g, device="cuda", dtype=torch.float32) * 0.01 + 0.01
    sb = torch.rand(n, generator=g, device="cuda", dtype=torch.float32) * 0.01 + 0.01
    b_shuf = shuffle_weight(b, layout=(16, 16))

    ref = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
    flydsl_8wave_gemm_a8(a, b_shuf, sa, sb, ref, 256, 256)

    # world_size=1 is not a legal ArConfig, and it does not need to be: with
    # fuse=False nothing in the epilogue reads the window, so any valid cfg of
    # the right (m, n) picks the same tile shape.
    cfg = layout.ArConfig(world_size=2, m=m, n=n)
    gemm = compile_fused_gemm_scatter(
        cfg, 0, K=k, BLOCK_M=256, BLOCK_N=256, b_preshuffled=True, fuse=False
    )
    got = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
    gemm(
        a.contiguous().view(torch.int8).view(-1),
        b_shuf.contiguous().view(torch.int8).view(-1),
        got.view(-1),
        sa,
        sb,
        m,
        n,
        0,  # dev_comm: unused when fuse=False
        0,  # win: ditto
        stream=fx.Stream(torch.cuda.current_stream()),
    )
    torch.cuda.synchronize()
    assert torch.equal(got, ref), (
        "the pinned copy of aiter's kernel_gemm has drifted from the original; "
        "re-sync benchmark/cco/flydsl/gemm_ar/kernels_fused.py"
    )


@pytest.mark.parametrize("m,n,k", [(512, 512, 256), (4096, 7168, 1024)])
def test_swap_ab_is_bitwise_identical(m, n, k):
    """Exchanging the MFMA operands must not change a single bit.

    ``fx.gemm(atom, c, b, a, c)`` computes ``(A B)^T`` in the accumulator's own
    layout: same products, same fp32 accumulation order along k, only the
    register-to-(row, col) mapping differs. So this is exactly reproducible, and
    anything less than bit-equality means the store's index math drifted rather
    than that the arithmetic changed.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    pytest.importorskip("aiter", reason="aiter not importable (set PYTHONPATH)")
    from aiter.ops.flydsl.gemm_a8w8_bpreshuffle_8wave import flydsl_8wave_gemm_a8
    from aiter.ops.shuffle import shuffle_weight

    import flydsl.expr as fx

    from kernels_fused import compile_fused_gemm_scatter

    g = torch.Generator(device="cuda").manual_seed(7)
    a = (torch.randn(m, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    sa = torch.rand(m, generator=g, device="cuda", dtype=torch.float32) * 0.01 + 0.01
    sb = torch.rand(n, generator=g, device="cuda", dtype=torch.float32) * 0.01 + 0.01
    b_shuf = shuffle_weight(b, layout=(16, 16))

    ref = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
    flydsl_8wave_gemm_a8(a, b_shuf, sa, sb, ref, 256, 256)

    cfg = layout.ArConfig(world_size=2, m=m, n=n)
    gemm = compile_fused_gemm_scatter(
        cfg, 0, K=k, BLOCK_M=256, BLOCK_N=256, b_preshuffled=True,
        fuse=False, swap_ab=True,
    )
    got = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
    gemm(
        a.contiguous().view(torch.int8).view(-1),
        b_shuf.contiguous().view(torch.int8).view(-1),
        got.view(-1), sa, sb, m, n, 0, 0,
        stream=fx.Stream(torch.cuda.current_stream()),
    )
    torch.cuda.synchronize()
    assert torch.equal(got, ref)


def test_rotated_tile_order_is_a_permutation_of_the_linear_one():
    """Reordering tiles must not change which tiles exist -- checked on the host.

    Reproduces the kernel's index math in Python so a mapping bug is caught
    without an 8-GPU run.
    """
    ws, m, n, block_m, block_n = 8, 4096, 7168, 256, 256
    n_blocks = n // block_n
    m_tiles_per_peer = (m // ws) // block_m
    total = (m // block_m) * n_blocks
    for rank in range(ws):
        seen = set()
        for idx in range(total):
            rest, bn = divmod(idx, n_blocks)
            tile_i, dest_seq = divmod(rest, ws)
            dest_i = (dest_seq + rank) % ws
            seen.add((dest_i * m_tiles_per_peer + tile_i, bn))
        assert len(seen) == total
        assert seen == {(bm, bn) for bm in range(m // block_m) for bn in range(n_blocks)}


# --------------------------------------------------------------------------
# GPU smoke: every mode reaches the same answer
# --------------------------------------------------------------------------


def _run_bench(world_size, mode, m, n, k, extra=()):
    env = os.environ.copy()
    env.setdefault("MORI_SOCKET_IFNAME", "lo")
    env.setdefault("MORI_ENABLE_SDMA", "1")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        str(GEMM_AR_DIR / "bench_gemm_ar.py"),
        "--mode", mode,
        "-m", str(m), "-n", str(n), "-k", str(k),
        "--warmup", "1", "--iters", "3",
        *extra,
    ]
    result = subprocess.run(
        command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=1200
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


def test_fused_is_stable_across_repeats():
    """A single passing run proves nothing about the fused path.

    Its epilogue publishes tiles from 448 blocks and elects one of them to issue
    the transfer, so every ordering bug in it is intermittent -- and each
    individual relL2 looks "small" on its own. Two real races were shipped here
    behind exactly that: a shared SDMA queue when a destination has more than one
    chunk, and a counter atomic that had lost its acquire half. Both passed a
    one-shot check repeatedly before failing.

    Needs the full world size: the race needs more than one chunk per
    destination, and at 2 ranks the wo_b shape only has one.
    """
    if torch.cuda.device_count() < 8:
        pytest.skip("the fused race only reproduces at world_size=8")
    pytest.importorskip("aiter", reason="aiter not importable (set PYTHONPATH)")
    seen = []
    for _ in range(3):
        record = _run_bench(8, "fused-sdma", 4096, 7168, 1024)
        seen.append(record["rel_l2"])
    assert all(v == pytest.approx(seen[0], rel=1e-6) for v in seen), (
        f"fused all-reduce is not deterministic across runs: relL2 = {seen}. "
        "That is an ordering bug in the epilogue, not numerical noise."
    )
    assert all(v < 5e-3 for v in seen), seen


@pytest.mark.parametrize("mode", ["split-sdma", "fused-sdma", "split-lsa"])
def test_gemm_ar_modes_agree(mode):
    """All three paths must land on the same all-reduced result.

    The tolerance is fp8 quantization error, not collective error -- the
    collective itself is asserted bit-exact in ``test_flydsl_ar.py``. The point
    here is that the three modes agree with *each other*, which is what caught
    the missing release fence in the fused epilogue (it showed up as 3.8e-3
    against split's 2.35e-3, both individually "small").
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 GPUs")
    pytest.importorskip("aiter", reason="aiter not importable (set PYTHONPATH)")
    record = _run_bench(2, mode, 512, 1024, 256)
    assert record["validated"] is True
    assert record["max_rank_time_us"] > 0
    assert record["rel_l2"] < 5e-3
