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
"""``GemmAllReduceOp``, the public wrapper.

``test_gemm_ar.py`` covers the kernels through ``bench_gemm_ar.py``, which calls
the low-level builders directly -- so it says nothing about this layer, and three
correctness bugs lived here behind a green suite: the A scale flattened in the
wrong order, a control region that moved when one instance served a second M, and
a support predicate that admitted K=128.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flydsl")

from mori.ops.gemm_ar import ArConfig  # noqa: E402
from mori.ops.gemm_ar.op import (  # noqa: E402
    FP8_DTYPES,
    MAX_CHUNKS,
    MIN_K,
    _flatten_a_scale,
    counter_chunks,
    padded_m,
    supports,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
WORKER = pathlib.Path(__file__).with_name("gemm_ar_op_worker.py")

# The fp8 floor for these operands; the kernel-level tests gate on the same value.
FP8_FLOOR = 3e-3


# --- host-only: the predicate and the layout ------------------------------


@pytest.mark.parametrize("k", [128, MIN_K - 128, 64])
def test_supports_rejects_k_below_two_blocks(k):
    """The mainloop prefetches block 1 and runs two tail steps.

    K=128 passed the divisibility check, constructed, ran, and returned finite
    output at relL2 1.41 against 2.4e-3 for the otherwise identical K=256.
    """
    assert supports(512, 1024, k, 2) is False


def test_supports_accepts_the_smallest_real_k():
    assert supports(512, 1024, MIN_K, 2) is True


@pytest.mark.parametrize("world_size", [3, 5, 6, 7])
def test_supports_agrees_with_config_validation_on_world_size(world_size):
    """``supports`` said yes where construction raised.

    512 threads must divide by the world size, which ``ArConfig.validate``
    enforces and the predicate used to ignore.
    """
    assert supports(512, 1024, 512, world_size) is False
    with pytest.raises(ValueError):
        ArConfig(world_size=world_size, m=512, n=1024, recv_slots=world_size).validate()


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_supports_accepts_world_sizes_that_divide_the_block(world_size):
    assert supports(512, 1024, 512, world_size) is True


def test_supports_requires_the_fp8_row_multiple():
    """The fp8 conversion gives one wave a row, so N must be a whole wave-chunk."""
    assert supports(512, 7168, 512, 8, gather_dtype="fp8") is True
    assert supports(512, 1536, 512, 8, gather_dtype="fp8") is False
    assert supports(512, 1536, 512, 8, gather_dtype="bf16") is True


def test_control_region_does_not_move_between_m_values():
    """The bug: counter_chunks changes with M and everything after it shifts.

    At capacity, the payload starts at the same offset whatever M is, which is
    what makes one window safe for several shapes.
    """
    offsets = set()
    for m in (4096, 8192, 16384):
        cfg = ArConfig(
            world_size=8,
            m=m,
            n=7168,
            recv_slots=8,
            counter_chunks=counter_chunks(m, 8),
            counter_capacity=MAX_CHUNKS,
            counter_shape_slots=4,
        )
        cfg.validate()
        # Only the control region has to be stable. output_off sits after the
        # payload and moves with M by construction.
        offsets.add((cfg.counter_region_off, cfg.lock_off, cfg.input_off))
    assert len(offsets) == 1, offsets


def test_each_shape_slot_gets_its_own_counters():
    """Counters are never reset and the election modulus depends on M.

    Two shapes sharing a set elect on the residue the other left, so the sets
    have to be disjoint -- while everything after them stays put.
    """
    cfgs = [
        ArConfig(
            world_size=8,
            m=4096,
            n=7168,
            recv_slots=8,
            counter_chunks=4,
            counter_capacity=MAX_CHUNKS,
            counter_shape_slots=4,
            counter_shape_index=i,
        )
        for i in range(4)
    ]
    starts = [c.counter_off for c in cfgs]
    assert len(set(starts)) == 4
    span = 8 * MAX_CHUNKS * 4
    for a, b in zip(starts, starts[1:]):
        assert b - a == span
    assert starts[-1] + span <= cfgs[0].lock_off
    assert len({c.input_off for c in cfgs}) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("gather_dtype", "FP8"),
        ("gather_dtype", "fp16"),
        ("gather_transport", "sla"),
        ("scatter_dtype", "int8"),
    ],
)
def test_validate_rejects_misspelled_modes(field, value):
    """A typo used to select the default silently: "FP8" sent bf16."""
    cfg = ArConfig(world_size=8, m=4096, n=7168, recv_slots=8, **{field: value})
    with pytest.raises(ValueError, match=field):
        cfg.validate()


# --- host-only: the scale flattening --------------------------------------


def test_flatten_a_scale_preserves_physical_order():
    """The kernel reads (row, kb) at kb*M + row, for the unambiguous spellings."""
    m, kb = 8, 3
    logical = torch.arange(m * kb, dtype=torch.float32).reshape(m, kb)
    want = logical.t().reshape(-1)

    col_major = logical.t().contiguous().t()  # [M, kb], storage already kb-major
    assert col_major.stride() == (1, m)

    assert torch.equal(_flatten_a_scale(col_major, m, kb), want)
    assert torch.equal(_flatten_a_scale(logical.t().contiguous(), m, kb), want)
    assert torch.equal(_flatten_a_scale(want, m, kb), want)


def test_flatten_a_scale_is_not_a_reshape():
    """Guards the review's finding: reshape(-1) on the column-major tensor."""
    m, kb = 8, 3
    col_major = (
        torch.arange(m * kb, dtype=torch.float32).reshape(m, kb).t().contiguous().t()
    )
    assert not torch.equal(_flatten_a_scale(col_major, m, kb), col_major.reshape(-1))


def test_flatten_a_scale_refuses_the_ambiguous_row_major_case():
    """[M, K/128] row-major means two different things and must not be guessed.

    A logical [M, K/128] needs transposing; a K/128-major buffer carrying that
    shape -- which is what aiter_per1x128_quant(transpose_scale=True) returns --
    must be read flat. Shape and stride are identical in both cases, and
    guessing "transpose" took SGLang's perplexity from 3.26 to 862511 while
    every shape-level test here stayed green, because they build the logical
    tensor the guess assumes.
    """
    m, kb = 8, 3
    row_major = torch.arange(m * kb, dtype=torch.float32).reshape(m, kb)
    assert row_major.stride() == (kb, 1)
    with pytest.raises(ValueError, match="ambiguous"):
        _flatten_a_scale(row_major, m, kb)
    # Both meanings stay expressible, and they differ.
    flat = _flatten_a_scale(row_major.reshape(-1), m, kb)
    transposed = _flatten_a_scale(row_major.t(), m, kb)
    assert torch.equal(flat, row_major.reshape(-1))
    assert torch.equal(transposed, row_major.t().reshape(-1))
    assert not torch.equal(flat, transposed)


@pytest.mark.parametrize("shape", [(7, 3), (8, 4), (8,)])
def test_flatten_a_scale_rejects_shapes_that_are_neither(shape):
    with pytest.raises(ValueError):
        _flatten_a_scale(torch.zeros(shape), 8, 3)


def test_padded_m_rounds_to_whole_bands():
    assert padded_m(1, 8, 128) == 1024
    assert padded_m(1024, 8, 128) == 1024
    assert padded_m(1025, 8, 128) == 2048


# --- host-only: the tile and dtype boundaries -----------------------------


@pytest.mark.parametrize("block_m", [64, 1, 192, 0, -128])
def test_supports_rejects_tiles_the_kernel_cannot_build(block_m):
    """The kernel's assert is bare, so the boundary has to answer instead.

    block_m=64 used to return True here, construct, acquire all three
    resources, and only then raise an empty AssertionError from inside the
    mainloop on the first real call. block_m=0 used to raise ZeroDivisionError
    out of the padding arithmetic before reaching any check at all.
    """
    assert supports(4096, 7168, 2048, 8, block_m=block_m) is False


@pytest.mark.parametrize("block_n", [128, 384, 0])
def test_supports_rejects_n_tiles_the_kernel_cannot_build(block_n):
    assert supports(4096, 7168, 2048, 8, block_n=block_n) is False


def test_supports_accepts_the_legal_tile_multiples():
    """256x512 is legal even though it is not the default 128x256."""
    assert supports(4096, 7168, 2048, 8, block_m=256, block_n=512) is True


@pytest.mark.parametrize("world_size", [0, -1])
def test_supports_rejects_nonsense_world_size(world_size):
    """padded_m divides by world_size * block_m, so this has to be caught first."""
    assert supports(4096, 7168, 2048, world_size) is False


def test_fnuz_is_not_an_accepted_dtype():
    """Same width, different exponent bias -- see FP8_DTYPES.

    The MMA atom is fixed to OCP e4m3. Reading FNUZ bytes through it scales
    every A and every B element by 2, so the product comes back 4x too large:
    finite, plausible and wrong. Matching element size is not evidence.
    """
    assert torch.float8_e4m3fn in FP8_DTYPES
    assert torch.float8_e4m3fnuz not in FP8_DTYPES


def test_constructor_rolls_back_cleanly_when_a_resource_fails(monkeypatch):
    """Every acquisition step, failed in turn.

    The rollback calls close(), which clears _cache and _pad_in. Those used to
    be assigned after the try block, so any failure here died with
    AttributeError -- masking the real error and skipping the release loop that
    is the whole point of the rollback.

    Recording fakes, not a real communicator: this is about the ordering of the
    constructor's own state, which needs no GPU. The fake handles carry a null
    pointer, so the window-zeroing write is stubbed out -- left in, it faults
    the context and the fault surfaces in whatever runs next.
    """
    from mori.ops.gemm_ar import op as op_module
    from mori.ops.gemm_ar.op import GemmAllReduceOp

    monkeypatch.setattr(op_module, "from_gpu_ptr", lambda *a, **kw: torch.zeros(1))

    class _Handle:
        def __init__(self, log, name):
            self._log, self._name = log, name
            self.ptr, self.size = 0, 0

        def close(self):
            self._log.append(f"close:{self._name}")

    class _Comm:
        """Fails at `fail_at`; records what was acquired and what was closed."""

        nranks, rank = 8, 0

        def __init__(self, fail_at):
            self.fail_at, self.log = fail_at, []

        def _step(self, name):
            self.log.append(f"acquire:{name}")
            if name == self.fail_at:
                raise RuntimeError(f"injected failure in {name}")
            return _Handle(self.log, name)

        def alloc_mem(self, nbytes):
            return self._step("alloc_mem")

        def register_window(self, ptr, size):
            return self._step("register_window")

        def create_dev_comm(self, reqs):
            return self._step("create_dev_comm")

    for fail_at in ("alloc_mem", "register_window", "create_dev_comm"):
        comm = _Comm(fail_at)
        with pytest.raises(RuntimeError) as excinfo:
            GemmAllReduceOp(comm, n=7168, k=2048, m_max=4096)

        # The injected error propagates, rather than an AttributeError from
        # cleanup with the real one demoted to __context__.
        assert fail_at in str(excinfo.value)

        # Everything acquired before the failure was released. The dev-comm is
        # not in that set by design -- its queues are communicator-scoped and go
        # with the communicator, so it is never the last-but-one here.
        acquired = [x[len("acquire:") :] for x in comm.log if x.startswith("acquire:")]
        closed = {x[len("close:") :] for x in comm.log if x.startswith("close:")}
        assert acquired[-1] == fail_at, f"stopped at {acquired[-1]}, not {fail_at}"
        assert (
            set(acquired[:-1]) == closed
        ), f"failing at {fail_at}: acquired {acquired}, closed {closed}"


# --- multi-rank: the op itself --------------------------------------------


#: Back-to-back 8-rank launches outrun SDMA queue teardown -- the next process
#: then fails in hsaKmtCreateQueueExt (anvil.cpp:237). Each of these cases passes
#: alone and they failed only when spawned in sequence, so let the previous one
#: release before starting the next.
_SETTLE_SECONDS = 20


#: hsaKmtCreateQueueExt failing because the previous run's SDMA queues are not
#: reclaimed yet. It is a resource race, not a defect in what is under test, and
#: it is the only failure worth retrying -- everything else is reported as is.
_QUEUE_EXHAUSTED = "anvil.cpp"


def _run_worker_multi(world_size: int, case: str, *extra: str, timeout: int = 900):
    result = _spawn_worker(world_size, case, *extra, timeout=timeout)
    if result is None:
        # One retry, with a longer settle. Two files' worth of 8-rank runs come
        # before this one, and the queues do not always come back in 20s.
        time.sleep(_SETTLE_SECONDS * 3)
        result = _spawn_worker(world_size, case, *extra, timeout=timeout, last=True)
    return result


def _spawn_worker(
    world_size: int, case: str, *extra: str, timeout: int = 900, last: bool = False
):
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
            f"--nproc_per_node={world_size}",
            str(WORKER),
            "--case",
            case,
            *extra,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
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
    assert records, output
    return records


def _run_worker(world_size: int, case: str, *extra: str, timeout: int = 900):
    """One case, one record."""
    return _run_worker_multi(world_size, case, *extra, timeout=timeout)[0]


requires_two_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs 2 GPUs"
)
requires_gpu = pytest.mark.skipif(torch.cuda.device_count() < 1, reason="needs a GPU")


def _mxfp8_gemm_rel_l2(n: int, k: int, m: int, pad: bool = False) -> float:
    """One standalone GEMM against an fp32 reference built from the same bytes."""
    from mori.ops.gemm_ar import Mxfp8GemmOp, preshuffle_a_scale, preshuffle_b

    g = torch.Generator(device="cuda").manual_seed(11)
    op = Mxfp8GemmOp(n=n, k=k)
    m_pad = op.padded_m(m) if pad else m
    a_bf16 = (torch.randn(m, k, generator=g, device="cuda") / 8).to(torch.bfloat16)
    a_in = op.pad_rows(a_bf16, m_pad) if m_pad != m else a_bf16
    a = a_in.to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    ea = torch.randint(
        120, 123, (m_pad, k // 32), generator=g, device="cuda", dtype=torch.int32
    )
    eb = torch.randint(
        120, 123, (n // 32, k // 32), generator=g, device="cuda", dtype=torch.int32
    )
    got = op(
        a, preshuffle_b(b), preshuffle_a_scale(ea), eb.t().reshape(-1).contiguous()
    )[:m]

    sav, sbv = torch.exp2(ea.float() - 127.0), torch.exp2(eb.float() - 127.0)
    af, bf = a.float(), b.float()
    ref = torch.zeros(m_pad, n, device="cuda", dtype=torch.float32)
    for j in range(k // 32):
        sl = slice(j * 32, (j + 1) * 32)
        ref += (
            (af[:, sl] @ bf[:, sl].t())
            * sav[:, j : j + 1]
            * sbv[:, j].repeat_interleave(32)[None, :n]
        )
    ref = ref[:m]
    return (
        torch.linalg.vector_norm(got.float() - ref) / torch.linalg.vector_norm(ref)
    ).item()


@requires_gpu
@pytest.mark.parametrize("n,k,label", [(8192, 1280, "wq_b"), (5120, 2048, "wo_b")])
@pytest.mark.parametrize("m", [64, 192, 1024])
def test_standalone_mxfp8_gemm(n, k, label, m):
    """``Mxfp8GemmOp`` at DeepSeek-V4.1-Flash's two attention shapes.

    Single process on purpose: this op has no communicator and no symmetric
    window, which is the whole reason it exists -- a ColumnParallelLinear like
    wq_b has no all-reduce to fuse with. M=192 is not a multiple of BLOCK_M and
    still has to be exact: the grid is ceildiv and the tail block masks.
    """
    assert _mxfp8_gemm_rel_l2(n, k, m) < FP8_FLOOR


@requires_gpu
def test_standalone_mxfp8_gemm_pads_ragged_m():
    """A ragged M is zero-extended to the packed A scale's 64-row group.

    The padded rows must not disturb the real ones, and the check is numerical
    because they would not: a GEMM row depends only on its own input row, so
    getting this wrong shifts the *scale* layout rather than raising.
    """
    from mori.ops.gemm_ar import Mxfp8GemmOp

    op = Mxfp8GemmOp(n=5120, k=2048)
    assert op.padded_m(100) == 128
    assert op.padded_m(64) == 64
    assert _mxfp8_gemm_rel_l2(5120, 2048, 100, pad=True) < FP8_FLOOR


@requires_gpu
def test_standalone_mxfp8_gemm_refuses_unpadded_m():
    """M not on the group boundary is refused rather than quietly mis-scaled."""
    from mori.ops.gemm_ar import Mxfp8GemmOp

    op = Mxfp8GemmOp(n=5120, k=2048)
    a = torch.zeros(100, 2048, device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.zeros(5120, 2048, device="cuda", dtype=torch.float8_e4m3fn)
    sa = torch.zeros(100 * 2048 // 128, device="cuda", dtype=torch.int32)
    sb = torch.zeros(160 * 64, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="multiple of 64"):
        op(a, b, sa, sb)


def _mxfp8_gemv_rel_l2(n: int, k: int, m: int) -> float:
    """One skinny GEMM against an fp32 reference built from the same bytes."""
    from mori.ops.gemm_ar import Mxfp8GemvOp, preshuffle_b

    g = torch.Generator(device="cuda").manual_seed(11)
    x = (torch.randn(m, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    w = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    ex = torch.randint(
        120, 123, (m, k // 32), generator=g, device="cuda", dtype=torch.int32
    )
    ew = torch.randint(
        120, 123, (n // 32, k // 32), generator=g, device="cuda", dtype=torch.int32
    )
    got = Mxfp8GemvOp(n=n, k=k)(
        x, preshuffle_b(w), ex.to(torch.uint8), ew.to(torch.uint8)
    )

    sxv, swv = torch.exp2(ex.float() - 127.0), torch.exp2(ew.float() - 127.0)
    xf, wf = x.float(), w.float()
    ref = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    for j in range(k // 32):
        sl = slice(j * 32, (j + 1) * 32)
        ref += (
            (xf[:, sl] @ wf[:, sl].t())
            * sxv[:, j : j + 1]
            * swv[:, j].repeat_interleave(32)[None, :n]
        )
    return (
        torch.linalg.vector_norm(got.float() - ref) / torch.linalg.vector_norm(ref)
    ).item()


@requires_gpu
@pytest.mark.parametrize("n,k,label", [(8192, 1280, "wq_b"), (5120, 2048, "wo_b")])
@pytest.mark.parametrize("m", [1, 3, 16, 17, 32])
def test_standalone_mxfp8_gemv(n, k, label, m):
    """``Mxfp8GemvOp`` at V4.1-Flash's two attention shapes, over the M ladder.

    M=3 and M=17 are the ones that matter: they are not tile widths, so the
    token rows past M read a clamped row and must not reach the output, and
    M=17 additionally crosses from a 16-token config to a 32-token one.
    """
    assert _mxfp8_gemv_rel_l2(n, k, m) < FP8_FLOOR


@requires_gpu
def test_mxfp8_gemv_refuses_m_above_its_tile():
    """Past 32 tokens it declines rather than silently truncating the batch.

    The B operand is two 16-row MFMA tiles and there is no third, so a 33rd
    token has nowhere to go; without this it would compute 32 rows and return a
    tensor whose remaining rows were never written.
    """
    from mori.ops.gemm_ar import Mxfp8GemvOp

    op = Mxfp8GemvOp(n=5120, k=2048)
    x = torch.zeros(33, 2048, device="cuda", dtype=torch.float8_e4m3fn)
    w = torch.zeros(5120, 2048, device="cuda", dtype=torch.float8_e4m3fn)
    sx = torch.zeros(33, 64, device="cuda", dtype=torch.uint8)
    sw = torch.zeros(160, 64, device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="exceeds the skinny kernel"):
        op(x, w, sx, sw)


def test_gemv_config_widens_the_token_tile_for_the_top_bucket():
    """A tuned 16-token config must not be handed an M it cannot hold.

    The table is keyed by bucket and a bucket's config is free to be narrower
    than the bucket; the guard is here rather than in the kernel because the
    kernel's rejection would be a compile error on a server's hot path.
    """
    from mori.ops.gemm_ar.gemv import m_bucket, select_config

    assert m_bucket(1) == 1 and m_bucket(17) == 32
    assert select_config(8192, 1280, 1)["tokens"] == 16
    assert select_config(8192, 1280, 17)["tokens"] == 32


def test_supports_gemm_takes_both_attention_shapes():
    """Neither of V4.1-Flash's attention GEMMs needs a special case."""
    from mori.ops.gemm_ar import supports_gemm

    assert supports_gemm(8192, 1280) is True  # wq_b, ColumnParallel
    assert supports_gemm(5120, 2048) is True  # wo_b, RowParallel
    assert supports_gemm(5120, 64) is False  # K below MIN_K
    assert supports_gemm(5000, 2048) is False  # N not a multiple of BLOCK_N


@pytest.fixture(scope="module")
def worker_results():
    """Every GPU case, from one 2-rank spawn.

    Each case used to get its own launch. Six of them back to back outran the
    SDMA queue reclaim (hsaKmtCreateQueueExt, anvil.cpp:237) and the casualties
    were as often the neighbouring file's tests as this file's, because the cost
    is paid by whoever launches next. One spawn, results keyed by case.
    """
    records = _run_worker_multi(2, "all", "-n", "1024", "-k", "512")
    return records


def _case(records, name):
    for r in records:
        if r.get("case") == name:
            return r
    raise AssertionError(f"worker produced no record for {name!r}: {records}")


@requires_two_gpus
def test_public_op_takes_the_column_major_scale(worker_results):
    """End to end through ``GemmAllReduceOp``, with the scale the model emits.

    Before the fix this returned relL2 0.26: ``reshape(-1)`` on a column-major
    ``[M, K/128]`` walks the logical rows, so every scale met the wrong K block.
    """
    r = _case(worker_results, "scale_order")
    assert r["rel_l2"] < FP8_FLOOR, r
    assert r["flat_matches"], r
    assert r["physical_matches"], r


@requires_two_gpus
@pytest.mark.parametrize("m_small,m_large", [(512, 1024), (2048, 4096)])
def test_one_instance_serves_two_m_values_in_any_order(
    worker_results, m_small, m_large
):
    """Two M values through one instance, alternating, then repeated.

    Two separate defects lived here. The reviewer's: counter_chunks changes with
    M, which moved lock_off and everything after it, so the second shape read the
    first one's payload as control state and hung. Then, once that was fixed,
    this: the payload offsets are a running sum of m-sized regions, so at
    world=2 n=1024 m=1024's output rows 512..767 land exactly where m=512's recv
    slot for peer 1 was. The first call at the second shape came back with that
    peer's previous *scatter* payload in those rows -- 4 runs in 8, always the
    same value, with the repeat correct because by then the bytes were its own.

    Both are fixed by pinning the map: the control region to MAX_CHUNKS, the
    payload to the instance's m_max. Every shape now has the same map, so a
    region only ever aliases itself.
    """
    r = _case(worker_results, f"alternating_m_{m_small}_{m_large}")
    for key in ("first_small", "first_large", "again_small", "again_large"):
        assert r[key] < FP8_FLOOR, (key, r)


@requires_two_gpus
def test_changing_operands_between_calls_stays_correct(worker_results):
    """Five calls at one M, different data each time.

    Every other test and every benchmark feeds identical operands on every
    iteration, which makes a read of a peer's not-yet-landed slice return the
    previous call's bytes -- bit-identical to the right answer. This is the only
    thing in the suite that would notice.
    """
    r = _case(worker_results, "changing_data")
    for key in (f"call{i}" for i in range(5)):
        assert r[key] < FP8_FLOOR, (key, r)


@requires_two_gpus
def test_mxfp8_quant_runs_end_to_end(worker_results):
    """``quant="mxfp8"`` through the public op: V4.1-Flash's quantisation.

    The op hardcoded ``quant="blockscale"`` until this, so the mxfp8 kernel --
    tuned and tested at the kernel layer -- had no way out to a caller. What
    this covers that the kernel tests do not is the operand contract: the A
    scale is ``preshuffle_a_scale``'s packed int32 and the B scale is
    ``[N/32, K/32]`` exponent bytes K-block major, and both are int32 where
    blockscale's are fp32. Getting either wrong returns finite nonsense rather
    than failing, which is why the check is numerical.
    """
    r = _case(worker_results, "mxfp8")
    assert r["rel_l2"] < FP8_FLOOR, r


@requires_two_gpus
def test_self_test_passes_and_can_fail(worker_results):
    """The guard against a mori whose SDMA puts were compiled out.

    That build used to be the default (`BUILD_CCO_SDMA` was OFF unless
    BUILD_BENCHMARK was ON) and it fails silently: every symbol is there, every kernel launches,
    every put returns, and nothing moves. The all-reduce then yields mostly the
    local slice, the model still answers fluently, and the fused path measures
    *faster* than it is. An end-to-end campaign read -6.8% instead of -2.3% that
    way and only perplexity caught it.

    So this asserts both halves: the check passes on a working stack, and it
    fails when the scatter does not happen -- which is what an inert put is.
    """
    r = _case(worker_results, "self_test")
    assert r["passes_on_working_stack"], r
    assert r["detects_missing_scatter"], r


@requires_two_gpus
def test_close_releases_and_is_idempotent(worker_results):
    """The communicator holds the handles, so dropping the op is not enough."""
    r = _case(worker_results, "close")
    assert r["refused_after_close"], r
