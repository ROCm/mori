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
"""The REAL P<->D kernel pair: a normal kernel flipped to/from AsyncLL.

`test_dispatch_combine_role_switch.py` covers IntraNode <-> IntraNodeLL, which
is the right *mechanism* but not the right *pair*. sglang's decode role runs
**AsyncLL**; prefill runs a normal kernel. AsyncLL is the one kernel type whose
selection is not a per-op decision:

    NormalizeConfig (dispatch_combine.cpp:160)
        config.enableSdma = ShmemSdmaEnabled()
    ShmemSdmaEnabled  (shmem/runtime.cpp:187)
        -> commContext->IsSdmaEnabled()
    Context::Context   (application/context/context.cpp:51)
        sdmaEnabled = env::IsEnvVarEnabled("MORI_ENABLE_SDMA")   <-- ONCE

so the flag is a snapshot taken at shmem init, and `SymmMemManager::Malloc`
(memory/symmetric_memory.cpp:84) picks uncached-vs-plain allocation off the same
snapshot. Setting the env after init gives uncached buffers under a transport
that already chose P2P -- a cache/IPC inconsistency HANG rather than an error,
which is why `dispatch_combine.cpp:154-159` explicitly refuses to getenv it
per-op.

That is why this is a separate MODULE and not more parametrizations of the
existing file: the env has to be set before the session-scoped worker pool
spawns, exactly as `test_dispatch_combine_async_ll.py:32` does it. It is also
the constraint Team E must honour in the server launch command (COORD
[M, turn 19] §1) -- a runtime flip into AsyncLL cannot turn SDMA on for itself.
"""
import gc
import os

# BEFORE the pool spawns. See the module docstring -- a late set is worse than
# no set, because it desynchronizes the transport choice from the allocator's.
os.environ.setdefault("MORI_ENABLE_SDMA", "1")

import pytest
import torch

import mori
from tests.python.ops.dispatch_combine_test_utils import (
    EpDispatchCombineTestCase,
    assert_worker_results,
)
from tests.python.ops.test_dispatch_combine_async_ll import (
    AsyncLLDispatchCombineTestCase,
)

PREFILL_TOKENS = 128
DECODE_TOKENS = 8

_WORLD_SIZES = (int(os.environ.get("MORI_TEST_WORLD_SIZE", "8")),)


def _make_config(rank, world_size, max_num_inp_token_per_rank, kernel_type):
    return mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world_size,
        hidden_dim=4096,
        scale_dim=0,
        scale_type_size=1,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=32,
        num_experts_per_token=8,
        max_token_type_size=4,
        block_num=64,
        # AsyncLL's own suite uses 8 warps per block; the intranode kernels use
        # 4. Kept per-kernel-type rather than unified because warp_num_per_block
        # feeds the launch geometry, and a value that merely *works* for one
        # kernel is not evidence for the other.
        warp_num_per_block=8 if kernel_type == "AsyncLL" else 4,
        use_external_inp_buf=True,
        gpu_per_node=min(world_size, 8),
        kernel_type=getattr(mori.ops.EpDispatchCombineKernelType, kernel_type),
        quant_type="none",
    )


def _run_once(op, config):
    """One fully checked dispatch+combine, on whichever protocol this op uses.

    AsyncLL splits dispatch/combine into send+recv halves and cannot be driven
    through the single-call path, so the test case class is selected from the
    kernel type rather than assumed -- which is itself part of what the flip has
    to get right.
    """
    if (
        config.kernel_type.value
        == mori.ops.EpDispatchCombineKernelType.AsyncLL.value
    ):
        test_case = AsyncLLDispatchCombineTestCase(config)
    else:
        test_case = EpDispatchCombineTestCase(config)
    test_data = test_case.gen_test_data(use_max_token_num=True, routing="random")
    test_case.run_test_once(op, test_data, check_results=True)


def _heap_stats():
    gc.collect()
    return mori.shmem.shmem_get_heap_stats()


def _heap_stats_required(rank, what):
    """`_heap_stats()`, but a None FAILS instead of silently voiding the claim.

    Outside static-heap mode `shmem_get_heap_stats()` returns None by design.
    Guarding the leak assertions on that (`if baseline is not None:`) makes the
    entire check vanish and the test report green while proving nothing --
    which is indistinguishable, in a log, from a run that proved something.
    Mirrors the same helper in test_dispatch_combine_role_switch.py.
    """
    stats = _heap_stats()
    assert stats is not None, (
        f"rank {rank}: symmetric-heap stats unavailable ({what}) -- this test "
        f"cannot make its leak claim, so it fails rather than passing "
        f"vacuously. Re-run in static-heap mode."
    )
    return stats


def _assert_capacity(op, expected_tokens, expected_kernel_type):
    assert op.config.max_num_inp_token_per_rank == expected_tokens
    assert op._handle_info["max_num_inp_token_per_rank"] == expected_tokens
    assert op.max_num_tokens_to_send_per_rank() == expected_tokens
    assert op.is_initialized
    assert (
        op.config.kernel_type.value
        == getattr(
            mori.ops.EpDispatchCombineKernelType, expected_kernel_type
        ).value
    ), f"op is on kernel type {op.config.kernel_type}, want {expected_kernel_type}"


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------
def _worker_sdma_snapshot_is_a_launch_constraint(rank, world_size):
    """Prove the SDMA flag is a LAUNCH snapshot, and that this pool has it on.

    Runs first and is deliberately cheap. Two jobs:

      * a guard against a vacuous suite. If `MORI_ENABLE_SDMA` did not reach
        the workers, AsyncLL still constructs and still computes -- it just
        burns CUs for communication (`dispatch_combine.cpp:163-168` warns
        exactly this). Every other test here would then pass while measuring
        the wrong transport, and nothing in the output would say so.
      * it states the constraint E has to honour, as an executable assertion
        rather than a paragraph in COORD: an op created at ANY point in this
        process reports the value that was in the environment at shmem init.

    Mutating the env here and re-reading it through a NEW op is the direct
    demonstration that a flip cannot turn SDMA on for itself.
    """
    assert os.environ.get("MORI_ENABLE_SDMA") == "1", (
        f"rank {rank}: MORI_ENABLE_SDMA did not reach the worker "
        f"(got {os.environ.get('MORI_ENABLE_SDMA')!r}); every AsyncLL result "
        f"from this pool would be measuring the non-SDMA path while claiming "
        f"otherwise"
    )

    config = _make_config(rank, world_size, DECODE_TOKENS, "AsyncLL")
    op = mori.ops.EpDispatchCombineOp(config)
    _assert_capacity(op, DECODE_TOKENS, "AsyncLL")
    _run_once(op, config)

    # Now turn it OFF and build a second op. If enableSdma were a per-op getenv
    # this would change; it must not, because it is the Context snapshot.
    prev = os.environ.get("MORI_ENABLE_SDMA")
    os.environ["MORI_ENABLE_SDMA"] = "0"
    try:
        op2 = mori.ops.EpDispatchCombineOp(
            _make_config(rank, world_size, DECODE_TOKENS, "AsyncLL")
        )
        _run_once(op2, config)
        op2.finalize()
        del op2
    finally:
        if prev is None:
            os.environ.pop("MORI_ENABLE_SDMA", None)
        else:
            os.environ["MORI_ENABLE_SDMA"] = prev

    op.finalize()
    del op


def _worker_asyncll_flip_destroy_recreate(rank, world_size):
    """P(IntraNode) <-> D(AsyncLL) by destroy+recreate. THE acceptance pair.

    Same mechanism as `test_kernel_type_flip_destroy_recreate` in the sibling
    module, on the kernel pair sglang actually flips between. Three things make
    this strictly stronger than the IntraNode<->IntraNodeLL version:

      * the two kernel types map to DIFFERENT hip sources -- `ep_intranode` vs
        `ep_async_ll` (`dispatch_combine.py:141,146`) -- so the recreate really
        does load a different module, and a stale `_hip_module` would be a
        cross-source mismatch rather than a cross-entry-point one.
      * they take DIFFERENT symmetric buffer sets. AsyncLL falls to the
        `ShmemBufsInterNode` branch (`dispatch_combine.cpp:620`, the else) with
        5 token buffers sized off `MaxNumTokensToSend()`, while IntraNode takes
        the 3-buffer `ShmemBufsIntraNode` branch (`:598`). So the closed-loop
        heap equality below is over two genuinely different allocation shapes,
        not the same shape twice.
      * they use different DRIVING protocols (single-call vs send/recv halves),
        which is why `_run_once` selects the test case class off the config.
    """
    from mori.ops.dispatch_combine import warmup_jit_kernels

    # Both roles warmed BEFORE the flip, which is what COORD tells sglang to do
    # pre-serving: these are two different hip sources, so a cold compile inside
    # the flip would land on the critical path of a live role switch.
    for kt in ("IntraNode", "AsyncLL"):
        warmup_jit_kernels(getattr(mori.ops.EpDispatchCombineKernelType, kt))

    baseline = _heap_stats_required(rank, "baseline")

    p_config = _make_config(rank, world_size, PREFILL_TOKENS, "IntraNode")
    d_config = _make_config(rank, world_size, DECODE_TOKENS, "AsyncLL")

    # P
    op = mori.ops.EpDispatchCombineOp(p_config)
    _assert_capacity(op, PREFILL_TOKENS, "IntraNode")
    _run_once(op, p_config)

    built = _heap_stats_required(rank, "post-construct stats")
    assert built["total_free_space"] < baseline["total_free_space"], (
        f"rank {rank}: the P op allocated no symmetric memory "
        f"(baseline={baseline} built={built}) -- this test measures nothing"
    )

    # P -> D
    op.finalize()
    del op
    op = mori.ops.EpDispatchCombineOp(d_config)
    _assert_capacity(op, DECODE_TOKENS, "AsyncLL")
    _run_once(op, d_config)

    # D -> P (the grow direction)
    op.finalize()
    del op
    op = mori.ops.EpDispatchCombineOp(p_config)
    _assert_capacity(op, PREFILL_TOKENS, "IntraNode")
    _run_once(op, p_config)

    op.finalize()
    del op

    after = _heap_stats_required(rank, "after")
    assert after["total_free_space"] == baseline["total_free_space"], (
        f"rank {rank}: an IntraNode->AsyncLL->IntraNode destroy+recreate "
        f"loop did not return the heap to baseline: {baseline} -> {after}"
    )
    assert after["num_mem_objs"] == baseline["num_mem_objs"], (
        f"rank {rank}: SymmMemObj count moved across the closed loop: "
        f"{baseline} -> {after}"
    )
    if rank == 0:
        print(
            f"[asyncll-flip] rank 0: RAN IntraNode@{PREFILL_TOKENS} -> "
            f"AsyncLL@{DECODE_TOKENS} -> IntraNode@{PREFILL_TOKENS} via "
            f"destroy+recreate, heap baseline={baseline} after={after}",
            flush=True,
        )


def _worker_asyncll_reconfigure_resizes(rank, world_size):
    """Within ONE role, AsyncLL must still resize through reconfigure().

    Not redundant with the flip test. sglang does not only change roles: the
    decode role's own capacity moves (`SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK`
    per MOTIVATION_RULES §2), and a same-kernel-type resize is what
    `reconfigure()` is FOR. Every existing reconfigure test in this repo runs
    IntraNode or IntraNodeLL, whose buffers come from the 3-object
    `ShmemBufsIntraNode` branch. AsyncLL takes the 5-object `ShmemBufsInterNode`
    branch (`dispatch_combine.cpp:620`), so `FinalizeShmemBuf`/`InitializeShmemBuf`
    walk a different `std::variant` alternative -- a teardown that misses an
    object there is invisible to the whole existing suite.

    Exact heap equality around the closed loop is what would catch that.
    """
    config = _make_config(rank, world_size, PREFILL_TOKENS, "AsyncLL")
    op = mori.ops.EpDispatchCombineOp(config)
    _assert_capacity(op, PREFILL_TOKENS, "AsyncLL")
    _run_once(op, config)

    after_build = _heap_stats_required(rank, "after_build")

    op.reconfigure(max_num_inp_token_per_rank=DECODE_TOKENS)
    _assert_capacity(op, DECODE_TOKENS, "AsyncLL")
    _run_once(op, _make_config(rank, world_size, DECODE_TOKENS, "AsyncLL"))

    op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
    _assert_capacity(op, PREFILL_TOKENS, "AsyncLL")
    _run_once(op, config)

    after_cycle = _heap_stats_required(rank, "after_cycle")
    assert after_cycle["total_free_space"] == after_build["total_free_space"], (
        f"rank {rank}: an AsyncLL P->D->P resize did not return the heap to "
        f"its starting state: {after_build} -> {after_cycle}"
    )
    assert after_cycle["num_mem_objs"] == after_build["num_mem_objs"], (
        f"rank {rank}: AsyncLL symmetric object count moved across a closed "
        f"resize loop -- a buffer in the InterNode variant is not being "
        f"freed or not being re-registered: {after_build} -> {after_cycle}"
    )

    op.finalize()
    del op


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_sdma_snapshot_is_a_launch_constraint(
    torch_dist_process_manager, world_size
):
    """SDMA is decided at shmem init; a flip cannot turn it on for itself."""
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_sdma_snapshot_is_a_launch_constraint, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_asyncll_flip_destroy_recreate(torch_dist_process_manager, world_size):
    """The acceptance flip on the kernel pair sglang really uses."""
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_asyncll_flip_destroy_recreate, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_asyncll_reconfigure_resizes(torch_dist_process_manager, world_size):
    """A same-kernel-type resize on AsyncLL's own symmetric buffer variant."""
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_asyncll_reconfigure_resizes, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)
