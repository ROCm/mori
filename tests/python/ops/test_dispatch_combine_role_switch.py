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
"""EpDispatchCombineOp buffer teardown / rebuild (`reconfigure`, `finalize`).

Covers the mori side of an sglang PD role switch: a prefill role sizes its
all-to-all dispatch buffer for a large chunked-prefill token count, a decode
role for a small per-step batch. On a flip the buffers must be freed and
re-allocated for the new capacity, in place, with the shmem context intact --
and dispatch/combine must still be numerically correct afterwards.
"""
import os

import pytest
import torch
import torch.distributed as dist

import mori
from tests.python.ops.dispatch_combine_test_utils import (
    EpDispatchCombineTestCase,
    assert_worker_results,
)

# "prefill" (large chunked-prefill token count) vs "decode" (per-step batch).
# Small on purpose: 19 tests share these, several of them 10-20 cycle stress
# loops, and the injection tests need headroom in the symmetric heap. A 16x flip
# exercises every code path a 32x one does.
PREFILL_TOKENS = 128
DECODE_TOKENS = 8

# The capacities sglang ACTUALLY issues, measured by Team E on a live
# DeepSeek EP+a2a server (COORD [E, turn 3]; worktrees/teamE/results/
# buffer_gap_v2.txt): prefill 4096, decode 128. Two orders of magnitude more
# memory than the values above and a 32x rather than 16x ratio, so the paths
# that only fail at scale -- a grow that does not fit, a shrink that fragments --
# are not reachable from the defaults. Used by the real-capacity test below,
# which is separate rather than a re-parametrization of the whole suite: at
# 4096 the leak-stress and injection tests would spend minutes and risk failing
# for heap-size reasons that say nothing about the code under test.
REAL_PREFILL_TOKENS = int(os.environ.get("MORI_TEST_REAL_PREFILL_TOKENS", "4096"))
REAL_DECODE_TOKENS = int(os.environ.get("MORI_TEST_REAL_DECODE_TOKENS", "128"))


# Kernel types reachable with a single node of 8 GPUs. The InterNode* paths
# need real multi-node RDMA and AsyncLL needs MORI_ENABLE_SDMA set before shmem
# init, so both are left to their own coverage (see backlog in RESULTS_M.md).
_KERNEL_TYPES = ("IntraNode", "IntraNodeLL")

# The world size the shared worker pool was started at (conftest reads the same
# env var). Every test parametrizes on this rather than a literal 8 so that a
# run confined to the free GPUs of a contended node -- HIP_VISIBLE_DEVICES=1,2,4,5
# MORI_TEST_WORLD_SIZE=4 -- actually exercises the tests instead of asking a
# 4-rank pool for 8 workers and hanging on the missing four.
_WORLD_SIZES = (int(os.environ.get("MORI_TEST_WORLD_SIZE", "8")),)


def _make_config(
    rank,
    world_size,
    max_num_inp_token_per_rank,
    kernel_type="IntraNode",
    quant_type="none",
):
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
        warp_num_per_block=4,
        use_external_inp_buf=True,
        # Defaults to 8, and NormalizeConfig asserts
        # `IsPowerOf2(gpuPerNode) && worldSize % gpuPerNode == 0`
        # (dispatch_combine.cpp:136). At world_size=4 that assert ABORTS the
        # rank -- SIGABRT, no python traceback, nothing on the result queue --
        # so a sub-8 run has to say how many GPUs it is really using. Every
        # config here is single-node by construction, so it is the world size.
        gpu_per_node=min(world_size, 8),
        kernel_type=getattr(mori.ops.EpDispatchCombineKernelType, kernel_type),
        quant_type=quant_type,
    )


def _run_once(op, config, routing="random"):
    """One dispatch+combine at the op's current capacity, fully checked."""
    test_case = EpDispatchCombineTestCase(config)
    test_data = test_case.gen_test_data(use_max_token_num=True, routing=routing)
    test_case.run_test_once(op, test_data, check_results=True)


def _heap_stats():
    """Symmetric-heap accounting, or None when shmem is not in static-heap mode.

    NOT torch.cuda.mem_get_info: mori hipMallocs the entire static symmetric
    heap once at shmem init and thereafter only carves VA out of it, so device
    free bytes are constant whether or not `ShmemFree` is ever called. These
    counters come from the heap's own VA manager and so actually move when an
    a2a buffer is leaked.
    """
    return mori.shmem.shmem_get_heap_stats()


def _assert_capacity(op, expected_tokens):
    """The rebuilt buffers must actually be sized for the new role."""
    assert op.config.max_num_inp_token_per_rank == expected_tokens
    assert op._handle_info["max_num_inp_token_per_rank"] == expected_tokens
    assert op.max_num_tokens_to_send_per_rank() == expected_tokens
    assert op.max_num_tokens_to_send() == expected_tokens * op.config.world_size
    assert op.is_initialized


# ---------------------------------------------------------------------------
# workers (run one per rank via the torch-dist process manager)
# ---------------------------------------------------------------------------
def _worker_flip_and_flip_back(rank, world_size, kernel_type="IntraNode",
                               quant_type="none"):
    """P -> D -> P: correct numerics at every capacity, buffers really resized."""
    config = _make_config(rank, world_size, PREFILL_TOKENS, kernel_type, quant_type)
    op = mori.ops.EpDispatchCombineOp(config)

    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, config)

    # flip P -> D: buffers shrink
    op.reconfigure(max_num_inp_token_per_rank=DECODE_TOKENS)
    _assert_capacity(op, DECODE_TOKENS)
    _run_once(op, config)

    # flip back D -> P: buffers grow again (the OOM-risk direction)
    op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, config)


def _worker_leak_stress(rank, world_size, cycles):
    """N flip cycles must return the symmetric heap to EXACTLY its baseline.

    This is an exact-equality assertion, not a tolerance: the symmetric heap is
    a deterministic first-fit VA manager, not a caching allocator, so a
    P->D->P round trip that frees everything it took must land on the identical
    block count and free-byte count it started from. Any drift is a leaked a2a
    buffer.

    An earlier version of this test asserted on `torch.cuda.mem_get_info`,
    which cannot fail: the whole 2 GiB static heap is hipMalloc'd once at shmem
    init, so device free bytes are constant even if `ShmemFree` were a complete
    no-op. `shmem_get_heap_stats()` is the counter that can actually observe
    the leak.
    """
    config = _make_config(rank, world_size, PREFILL_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)
    _run_once(op, config)

    baseline = _heap_stats()
    if baseline is None:
        # Return, do NOT pytest.skip(): this runs in a worker process, and
        # `Skipped` is a BaseException, so the worker's `except Exception`
        # would miss it and leave the parent blocked on result_queue.get().
        return

    # Baseline is taken AFTER the first dispatch/combine so that any buffer
    # allocated lazily on first use is already counted; from here every cycle
    # must be a closed loop.
    per_cycle = []
    for _ in range(cycles):
        op.reconfigure(max_num_inp_token_per_rank=DECODE_TOKENS)
        _run_once(op, config)
        op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
        _run_once(op, config)
        per_cycle.append(_heap_stats())

    end = per_cycle[-1]
    detail = (
        f"rank {rank}: baseline={baseline} end={end} "
        f"per_cycle_allocated_blocks={[s['allocated_blocks'] for s in per_cycle]} "
        f"per_cycle_free_space={[s['total_free_space'] for s in per_cycle]}"
    )

    # (1) Every symmetric byte taken by the grown buffers came back.
    assert end["total_free_space"] == baseline["total_free_space"], (
        f"a2a symmetric buffers LEAKED "
        f"{baseline['total_free_space'] - end['total_free_space']} bytes over "
        f"{cycles} flip cycles. {detail}"
    )
    # (2) ...and no allocation record was orphaned. total_free_space alone
    # could be restored while a zero-length or double-counted block lingers.
    assert end["allocated_blocks"] == baseline["allocated_blocks"], (
        f"a2a buffers leaked "
        f"{end['allocated_blocks'] - baseline['allocated_blocks']} VA blocks "
        f"over {cycles} flip cycles. {detail}"
    )
    # (3) The SymmMemObj pool is a second, independent leak channel: a block
    # can be returned to the VA manager while its registration is never
    # deregistered from the pool.
    assert end["num_mem_objs"] == baseline["num_mem_objs"], (
        f"a2a buffers leaked "
        f"{end['num_mem_objs'] - baseline['num_mem_objs']} SymmMemObj "
        f"registrations over {cycles} flip cycles. {detail}"
    )
    # (4) Per-cycle steady state: catches a leak that happens to net out at the
    # end, and pins down that the loop is closed every cycle rather than only
    # in aggregate.
    for i, s in enumerate(per_cycle):
        assert s["total_free_space"] == baseline["total_free_space"], (
            f"heap free space diverged at cycle {i}. {detail}"
        )


def _worker_finalize_returns_everything(rank, world_size):
    """finalize() must return the heap to its pre-construction state.

    reconfigure() is a free+alloc pair, so a leak there can hide behind the
    re-allocation. finalize() frees with nothing following it, which is the
    sharpest possible test of the teardown path -- and it is the path sglang
    takes when it destroys an op rather than resizing it.
    """
    before = _heap_stats()
    if before is None:
        # Return, not pytest.skip() -- see _worker_leak_stress.
        return

    config = _make_config(rank, world_size, PREFILL_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)
    _run_once(op, config)

    during = _heap_stats()
    # Sanity: the op must actually have taken symmetric memory, otherwise the
    # "it all came back" assertion below would be trivially true.
    assert during["total_free_space"] < before["total_free_space"], (
        f"rank {rank}: EpDispatchCombineOp allocated no symmetric memory "
        f"(before={before} during={during}) -- this test is not measuring "
        f"anything"
    )

    op.finalize()
    assert not op.is_initialized

    after = _heap_stats()
    assert after["total_free_space"] == before["total_free_space"], (
        f"rank {rank}: finalize() leaked "
        f"{before['total_free_space'] - after['total_free_space']} symmetric "
        f"bytes (before={before} during={during} after={after})"
    )
    assert after["allocated_blocks"] == before["allocated_blocks"], (
        f"rank {rank}: finalize() leaked "
        f"{after['allocated_blocks'] - before['allocated_blocks']} VA blocks "
        f"(before={before} after={after})"
    )
    assert after["num_mem_objs"] == before["num_mem_objs"], (
        f"rank {rank}: finalize() leaked "
        f"{after['num_mem_objs'] - before['num_mem_objs']} SymmMemObj "
        f"registrations (before={before} after={after})"
    )


def _worker_flip_at_real_capacities(rank, world_size):
    """The flip sglang ACTUALLY issues: 4096 <-> 128, not 128 <-> 8.

    Every other test in this module runs a 16x flip at two orders of magnitude
    less memory. REVIEW_M has carried that gap since review #7 for a concrete
    reason: the D->P grow is the OOM direction sglang sizes on, and "128 -> 4096
    works" is not evidence for it. At these capacities the symmetric buffers are
    real -- MaxNumTokensToSend scales with maxNumInpTokenPerRank and the order
    maps scale with it times numExpertPerRank -- so a grow that does not fit or
    a free that fragments has somewhere to show.

    P -> D -> P with numerics checked at every capacity, plus the heap
    accounting: the closed loop must return the symmetric heap EXACTLY to its
    starting numbers. Exact, not approximate, because mori's static heap is a
    deterministic first-fit VA manager and not a caching allocator, so a closed
    P->D->P loop that does not land on identical counters has leaked or
    fragmented.

    Skips itself rather than failing when the heap cannot hold the large
    capacity: a run that dies because MORI_SHMEM_HEAP_SIZE is too small on a
    busy node says nothing about the code under test, and three turns of this
    campaign were lost to not distinguishing those two.
    """
    config = _make_config(rank, world_size, REAL_PREFILL_TOKENS)
    before = _heap_stats()

    try:
        op = mori.ops.EpDispatchCombineOp(config)
    except Exception as exc:  # heap too small for the real capacity on this node
        if "heap" in str(exc).lower() or "memory" in str(exc).lower():
            # Say so. A silent early return is indistinguishable in the log from
            # a full green run, and "the test passed" would then mean nothing --
            # the exact vacuity that made T1/T2's leak claim worthless.
            print(
                f"[real-capacities] rank {rank}: SKIPPED, cannot build at "
                f"{REAL_PREFILL_TOKENS} on this node: {exc}",
                flush=True,
            )
            return
        raise

    _run_once(op, config)
    _assert_capacity(op, REAL_PREFILL_TOKENS)
    after_build = _heap_stats()

    # D: the shrink sglang performs on a P->D flip.
    op.reconfigure(max_num_inp_token_per_rank=REAL_DECODE_TOKENS)
    _assert_capacity(op, REAL_DECODE_TOKENS)
    _run_once(op, _make_config(rank, world_size, REAL_DECODE_TOKENS))

    # P again: the GROW, which is the direction that can OOM.
    op.reconfigure(max_num_inp_token_per_rank=REAL_PREFILL_TOKENS)
    _assert_capacity(op, REAL_PREFILL_TOKENS)
    _run_once(op, config)

    after_cycle = _heap_stats()
    if after_build is not None and after_cycle is not None:
        assert after_cycle["total_free_space"] == after_build["total_free_space"], (
            f"rank {rank}: a {REAL_PREFILL_TOKENS}->{REAL_DECODE_TOKENS}->"
            f"{REAL_PREFILL_TOKENS} flip did not return the heap to its starting "
            f"state: {after_build} -> {after_cycle}"
        )
        assert after_cycle["num_mem_objs"] == after_build["num_mem_objs"], (
            f"rank {rank}: symmetric object count moved across a closed flip "
            f"loop: {after_build} -> {after_cycle}"
        )

    if rank == 0:
        print(
            f"[real-capacities] rank 0: RAN {REAL_PREFILL_TOKENS} -> "
            f"{REAL_DECODE_TOKENS} -> {REAL_PREFILL_TOKENS}, heap "
            f"after_build={after_build} after_cycle={after_cycle}",
            flush=True,
        )

    # And the teardown returns everything, at the real size.
    op.finalize()
    after_final = _heap_stats()
    if before is not None and after_final is not None:
        assert after_final["total_free_space"] == before["total_free_space"], (
            f"rank {rank}: finalize() at the real capacity leaked "
            f"{before['total_free_space'] - after_final['total_free_space']} bytes"
        )


def _worker_finalized_getters_raise(rank, world_size):
    """EVERY pybind buffer getter must raise on a finalized handle, not hand
    back a freed or null pointer.

    Two distinct failure modes live behind these entry points, and both end at a
    kernel launch:

      * the SymmMemObjPtr getters (`get_dispatch_output_ptrs`, ...) dereference
        a `{cpu=nullptr, gpu=nullptr}` pointer that `ShmemFreeAndInvalidate` left
        behind -- an immediate SIGSEGV inside pybind, which is how a reportable
        "must be rebuilt" became a dead rank in T5d;
      * the plain-device getters (`get_standard_moe_packed_recv_count_ptr`,
        `get_dispatch_sender_token_idx_map`, `set_standard_moe_output_buffers`)
        read members `HipFreeAndNull`'d at teardown, so they return **0** and
        the crash is deferred to whatever kernel is handed that address. That is
        one severity WORSE than the segfault, not better: it is silent.

    Neither is hypothetical. A D->P flip whose grow AND rollback both fail
    finalizes the handle in place, and sglang's `except` unwinds through code
    that re-reads these pointers. `758d97e2` claimed to guard "every pybind
    buffer deref" and guarded 8 of 11; this test is what makes that claim
    checkable instead of a commit message.

    Iterated over a table rather than asserted one by one so that a getter added
    later without a guard shows up here rather than in an inference server.
    """
    config = _make_config(rank, world_size, DECODE_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)
    _run_once(op, config)

    op.finalize()
    assert op.is_initialized is False

    # (callable, name). Each is invoked with only the handle plus whatever
    # scalars it needs; none of them may touch the freed buffers before the
    # guard runs.
    from mori import cpp as mori_cpp

    probes = [
        (
            lambda: mori_cpp.get_dispatch_output_ptrs(op._handle, True),
            "get_dispatch_output_ptrs",
        ),
        (
            lambda: mori_cpp.get_combine_output_ptrs(op._handle, True),
            "get_combine_output_ptrs",
        ),
        (lambda: mori_cpp.build_args(op._handle, rdma_block_num=0), "build_args"),
        (
            lambda: mori_cpp.get_dispatch_sender_token_idx_map(op._handle),
            "get_dispatch_sender_token_idx_map",
        ),
        (
            lambda: mori_cpp.get_dispatch_receiver_token_idx_map(op._handle),
            "get_dispatch_receiver_token_idx_map",
        ),
        (lambda: mori_cpp.get_dispatch_src_token_pos(op._handle), "get_dispatch_src_token_pos"),
    ]
    for name in ("get_standard_moe_packed_recv_count_ptr", "get_combine_input_ptr"):
        # Compiled only under ENABLE_STANDARD_MOE_ADAPT.
        fn = getattr(mori_cpp, name, None)
        if fn is not None:
            probes.append((lambda fn=fn: fn(op._handle), name))

    for probe, name in probes:
        with pytest.raises(RuntimeError) as excinfo:
            probe()
        message = str(excinfo.value)
        assert "holds no buffers" in message, f"rank {rank}: {name} -> {message}"
        assert name in message, (
            f"rank {rank}: {name} raised without naming itself, which is what an "
            f"operator greps for: {message}"
        )

    # And the op is genuinely rebuildable afterwards -- the guards must reject
    # the reads, not poison the handle.
    op.reconfigure(max_num_inp_token_per_rank=DECODE_TOKENS)
    _assert_capacity(op, DECODE_TOKENS)
    _run_once(op, config)


def _worker_reconfigure_rejects_layout_change(rank, world_size):
    """Fields the peers' symmetric layout depends on must not be changeable."""
    config = _make_config(rank, world_size, PREFILL_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)

    for field, bad_value in (
        ("hidden_dim", config.hidden_dim * 2),
        ("num_experts_per_rank", config.num_experts_per_rank + 1),
        ("num_experts_per_token", config.num_experts_per_token + 1),
        ("world_size", config.world_size + 1),
    ):
        good = getattr(op._cpp_config, field)
        setattr(op._cpp_config, field, bad_value)
        try:
            with pytest.raises(RuntimeError, match="must not change"):
                op._handle.reconfigure(op._cpp_config)
        finally:
            setattr(op._cpp_config, field, good)

    # A rejected reconfigure must leave the op fully usable -- validation runs
    # before anything is freed.
    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, config)


def _worker_reconfigure_oom_rolls_back(rank, world_size):
    """A resize that does not fit must leave the op usable at the OLD capacity.

    This is the D->P (decode -> prefill) direction, where the buffers GROW: per
    the E2E team's measurement the decode role runs with the larger
    chunked_prefill_size, so a flip into prefill is the one that can exhaust the
    symmetric heap. Before the rollback existed, a failure here left the handle
    with buffersInitialized=False and half-built buffers -- worse than not
    having flipped at all.
    """
    config = _make_config(rank, world_size, PREFILL_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)
    _run_once(op, config)

    # Far past the default 2 GiB MORI_SHMEM_HEAP_SIZE: the dispatch-out buffer
    # alone would be 2**24 * 8 * 4096 * 4 B. Every rank fails identically, so
    # this exercises the C++ per-rank rollback rather than the cross-rank
    # disagreement path.
    with pytest.raises(RuntimeError) as excinfo:
        op.reconfigure(max_num_inp_token_per_rank=1 << 24)
    assert "could not grow" in str(excinfo.value)

    # The whole point: still alive, still at the old size, still correct.
    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, config)

    # ...and a subsequent legitimate flip still works, i.e. the failed attempt
    # did not poison the heap.
    op.reconfigure(max_num_inp_token_per_rank=DECODE_TOKENS)
    _assert_capacity(op, DECODE_TOKENS)
    _run_once(op, config)


def _worker_plain_device_oom_raises(rank, world_size):
    """A flip that cannot grow PLAIN-DEVICE scratch must raise, not kill us.

    STATUS 2026-07-29: THIS TEST CURRENTLY FAILS -- SIGSEGV on all 8 ranks
    (exitcode -11), on an idle node in a fresh container. See RESULTS_M.md T5d.
    It is left in the tree, failing, rather than skipped or deleted, because it
    is reporting something real: the control in T5e (same HEAD) shows the
    SYMMETRIC-heap OOM rollback passes, so the crash is specific to a throw
    that lands DEEPER in InitializeAll -- group 3 (InitializeOrderMapBuf) with
    groups 1-2 already allocated -- rather than in group 1. Prime suspect is a
    Finalize*() body running over members the failed pass never assigned.
    Do not cite this test as evidence for the plain-device OOM contract until
    it is green; that contract is by-construction today, not measured.

    The symmetric-heap OOM above is only half the story, and the less likely
    half. The order maps / barriers / counters are raw `hipMalloc`, sized by
    `maxNumOutToken = MaxNumTokensToSend() * numExpertPerRank`, so they grow
    with the very capacity a role switch changes -- and plain device memory is
    exactly what sglang's KV cache has already eaten. Under HIP_RUNTIME_CHECK
    a failure there called `exit(-1)`, i.e. a D->P flip that ran out of device
    memory SIGKILLed the whole inference server instead of returning an error.

    Those call sites now throw. But "it raises instead of exiting" cannot be
    established by reading the code -- if it were wrong, the process would
    vanish and the test would look like a hang. It also cannot be provoked by
    asking for a huge capacity: that trips the symmetric heap first, which was
    already covered. Hence MORI_TEST_FAIL_HIPMALLOC, which fails one named
    plain-device allocation.

    The assertion that matters most here is the boring one at the end: this
    worker still reaches its return statement.
    """
    import os

    config = _make_config(rank, world_size, DECODE_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)
    _run_once(op, config)

    before = _heap_stats()

    # `dispSenderIdxMap` is the SECOND plain-device allocation in
    # InitializeOrderMapBuf, and InitializeShmemBuf/TokenNumSignalBuf have both
    # already run by then. So the failure lands mid-InitializeAll with real
    # symmetric-heap allocations outstanding -- which is precisely the state
    # the rollback used to leak, because FinalizeAll() early-returned on a
    # handle whose buffersInitialized flag was not yet set.
    os.environ["MORI_TEST_FAIL_HIPMALLOC"] = "dispSenderIdxMap"
    try:
        with pytest.raises(RuntimeError) as excinfo:
            op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
    finally:
        del os.environ["MORI_TEST_FAIL_HIPMALLOC"]

    # It must be the rollback's error, not the "rollback also failed" one --
    # the op has to still be usable.
    message = str(excinfo.value)
    assert "could not grow" in message, message
    assert "must be rebuilt" not in message, message

    # THE POINT: the process is still alive and serving at the old capacity.
    _assert_capacity(op, DECODE_TOKENS)
    _run_once(op, config)

    # And the failed attempt gave everything back. Without the fix the catch
    # path called FinalizeAll() on a handle with buffersInitialized=False, so
    # it freed nothing and the rollback then overwrote every pointer: the
    # symmetric-heap bites taken by the failed InitializeShmemBuf/
    # TokenNumSignalBuf would be leaked permanently, making each failed D->P
    # flip likelier to fail than the last. Exact equality, because the heap is
    # a deterministic first-fit VA manager and not a caching allocator.
    after = _heap_stats()
    if before is not None and after is not None:
        assert after["total_free_space"] == before["total_free_space"], (
            f"rank {rank}: a FAILED flip leaked symmetric heap: "
            f"total_free_space {before['total_free_space']} -> "
            f"{after['total_free_space']} "
            f"(lost {before['total_free_space'] - after['total_free_space']} bytes)"
        )
        assert after["num_mem_objs"] == before["num_mem_objs"], (
            f"rank {rank}: a FAILED flip leaked symmetric objects: "
            f"num_mem_objs {before['num_mem_objs']} -> {after['num_mem_objs']}"
        )

    # A later legitimate flip must still succeed: a failed flip is retryable,
    # which is the contract sglang's role switch relies on to keep serving.
    op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, config)


def _worker_repeated_failed_flips_do_not_accumulate(rank, world_size):
    """N failed D->P flips in a row must cost nothing cumulatively.

    STATUS 2026-07-29: FAILS for the same reason as
    `_worker_plain_device_oom_raises` above (SIGSEGV on all 8 ranks) -- it uses
    the same injection point. See that docstring and RESULTS_M.md T5d/T5e.

    The leak the previous test catches is per-failure and permanent, so its
    real-world signature is progressive: an sglang instance whose flips keep
    failing slowly loses symmetric heap until even the flips that used to
    succeed cannot. One failed flip barely moves the number; ten make it
    obvious. This is the test that would have caught the bug from a distance.
    """
    import os

    config = _make_config(rank, world_size, DECODE_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)
    _run_once(op, config)

    baseline = _heap_stats()
    if baseline is None:
        return

    cycles = 10
    trace = []
    for _ in range(cycles):
        os.environ["MORI_TEST_FAIL_HIPMALLOC"] = "dispSenderIdxMap"
        try:
            with pytest.raises(RuntimeError):
                op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
        finally:
            del os.environ["MORI_TEST_FAIL_HIPMALLOC"]
        trace.append(_heap_stats()["total_free_space"])

    summary = (
        f"rank {rank}: baseline={baseline['total_free_space']} "
        f"after_each_failed_flip={trace}"
    )
    if rank == 0:
        print("[failed-flip-accumulation] " + summary, flush=True)

    assert trace[-1] == baseline["total_free_space"], (
        f"{cycles} failed flips leaked symmetric heap cumulatively -- an "
        f"instance that keeps failing to flip eventually cannot flip at all. "
        f"{summary}"
    )
    # Still fully functional after 10 failures.
    _assert_capacity(op, DECODE_TOKENS)
    _run_once(op, config)


def _worker_rank_asymmetric_failure(rank, world_size):
    """ONE rank fails to grow while the other seven succeed.

    This is the path REVIEW_M has called the riskiest untested one in the
    feature since review #4, and it is the only one where the ranks do NOT run
    identical allocation sequences. That matters concretely:
    `RegisterStaticHeapSubRegion` computes each peer's address as
    `heapObj->peerPtrs[i] + offset` -- it ASSUMES every rank landed at the same
    VA offset in the symmetric heap, and there is no allgather that would catch
    it if they did not. A first-fit VA manager only guarantees that while every
    rank has performed the same alloc/free history. Here rank 0 does
    Finalize -> Init(new, partial, throws) -> Finalize -> Init(old), while its
    peers do Finalize -> Init(new) -> [agree] -> Finalize -> Init(old). If those
    two histories can land at different offsets, every peer pointer on the
    surviving ranks is silently wrong -- corruption, not a hang, and no
    assertion about capacity would notice.

    So the load-bearing assertion here is the LAST one: after the group has
    given the capacity back, a real dispatch+combine still produces correct
    numerics on every rank. `_run_once(check_results=True)` compares against a
    CPU reference, and it is checked on all 8 ranks, so a peer pointer that
    drifted on any rank shows up as a numeric mismatch.

    Also covers the severity agreement: rank 0's failure must roll back cleanly
    (SEVERITY_ROLLED_BACK, not UNRECOVERABLE), so every rank must raise, every
    rank must end at the OLD capacity, and no rank may be left holding the new
    one. A group where some ranks grew and others did not is corrupt.
    """
    import os

    config = _make_config(rank, world_size, DECODE_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)
    _run_once(op, config)

    before = _heap_stats()

    # Arm the injection on rank 0 ONLY. The hook is one-shot (it unsets the
    # variable as it fires), so rank 0's rollback to DECODE_TOKENS is NOT
    # sabotaged -- it must succeed, leaving rank 0 usable at the old capacity
    # and the group's worst severity at ROLLED_BACK.
    if rank == 0:
        os.environ["MORI_TEST_FAIL_HIPMALLOC"] = "dispSenderIdxMap"
    try:
        with pytest.raises(RuntimeError) as excinfo:
            op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
    finally:
        os.environ.pop("MORI_TEST_FAIL_HIPMALLOC", None)

    message = str(excinfo.value)
    # Every rank raises the SAME message: the seven that succeeded must not
    # silently keep the capacity they grew to, because their peers do not have
    # it, and sglang branches on this string per rank -- so if the wording
    # differed by rank, so would sglang's keep-serving/escalate decision.
    # (Before 45470024 rank 0 raised its own C++ error and the peers raised a
    # "another rank in the EP group ..." variant; the group verdict is now
    # unconditional, so both ranks see the identical rolled-back text.)
    assert "could not grow" in message, f"rank {rank}: {message}"
    assert (
        "still usable at the old capacity" in message
    ), f"rank {rank}: {message}"
    # Nobody may report the group as unusable -- rank 0's rollback succeeded.
    assert "must be rebuilt" not in message, f"rank {rank}: {message}"
    assert "GROUP is not usable" not in message, f"rank {rank}: {message}"

    # All 8 ranks are back at the OLD capacity, including the 7 that grew fine.
    _assert_capacity(op, DECODE_TOKENS)

    # THE ASSERTION THAT MATTERS: correct numerics after the divergence. If the
    # differing alloc/free histories desynchronized peer VA offsets, this is
    # where it surfaces.
    _run_once(op, config)

    # And the divergence cost the heap nothing on any rank.
    after = _heap_stats()
    if before is not None and after is not None:
        assert after["total_free_space"] == before["total_free_space"], (
            f"rank {rank}: rank-asymmetric failed flip leaked symmetric heap: "
            f"{before['total_free_space']} -> {after['total_free_space']}"
        )
        assert after["num_mem_objs"] == before["num_mem_objs"], (
            f"rank {rank}: rank-asymmetric failed flip leaked symmetric objects: "
            f"{before['num_mem_objs']} -> {after['num_mem_objs']}"
        )

    # The group is still flippable: a subsequent real flip must work on all
    # ranks and still compute correctly. This is what sglang relies on to keep
    # serving after an aborted role switch.
    op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, config)


def _worker_rank_asymmetric_unrecoverable(rank, world_size):
    """ONE rank ends UNRECOVERABLE while the other seven roll back cleanly.

    This is the asymmetry `_worker_rank_asymmetric_failure` does NOT cover: there
    rank 0's rollback succeeds, so the group's worst severity is ROLLED_BACK and
    every rank really is usable at the old capacity. Here rank 0's rollback
    fails too (MORI_TEST_FAIL_HIPMALLOC_TIMES=2 fails both the grow and the
    rollback), so rank 0 holds NO a2a buffers while its seven peers hold intact
    ones. That is the state SEVERITY_UNRECOVERABLE, the C++ "must be rebuilt"
    message and pybind's RequireInitialized() guard all exist for, and until the
    fire-count knob landed no test could reach it -- so the SIGSEGV fixed in
    85c34c18 could not be proven gone, and sglang's MoriA2AResizeUnrecoverable,
    which keys on "must be rebuilt", had nothing establishing that string is
    ever emitted.

    Three things are asserted, and each was a real bug:

    1. EVERY rank reports the GROUP's outcome. The seven healthy ranks must NOT
       raise "still usable at the old capacity" -- a group whose symmetric
       buffers exist on 7 of 8 ranks is corrupt, not degraded, and sglang makes
       a per-rank keep-serving/escalate decision off this string. Before
       review #21 item 2's fix `raise local_err` ran first, so co-failing ranks
       raised their own C++ message and the group SPLIT.
    2. Rank 0's own message must survive to python as "must be rebuilt" ACROSS
       PYBIND. Reaching it at all exercises `_restore_mirror` /
       `_refresh_handle_state` over a finalized handle -- the null SymmMemObjPtr
       deref that used to SIGSEGV all 8 ranks (T5d).
    3. The group RECOVERS. sglang is told a resize failure leaves it able to
       keep serving, so a retry at the old capacity must rebuild rank 0's
       buffers and leave all 8 computing correctly. That retry is also the
       deadlock review #21 item 4 found: the healthy ranks see "same capacity,
       already initialized" and would have returned early -- skipping two
       barriers -- while rank 0 (is_initialized False) fell through into them.
    """
    import os

    config = _make_config(rank, world_size, DECODE_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)
    _run_once(op, config)

    if rank == 0:
        os.environ["MORI_TEST_FAIL_HIPMALLOC"] = "dispSenderIdxMap"
        # 2 => the rollback's InitializeAll() re-hits the same site and throws
        # too, so the handle is finalized on purpose.
        os.environ["MORI_TEST_FAIL_HIPMALLOC_TIMES"] = "2"
    try:
        with pytest.raises(RuntimeError) as excinfo:
            op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
    finally:
        os.environ.pop("MORI_TEST_FAIL_HIPMALLOC", None)
        os.environ.pop("MORI_TEST_FAIL_HIPMALLOC_TIMES", None)

    message = str(excinfo.value)
    # (1) + (2): the group verdict is unanimous, on every rank.
    assert "GROUP is not usable" in message, f"rank {rank}: {message}"
    assert (
        "still usable at the old capacity" not in message
    ), f"rank {rank} reported the group usable while a peer holds nothing: {message}"
    if rank == 0:
        assert "must be rebuilt" in message, f"rank 0: {message}"
        assert op.is_initialized is False, "rank 0 should hold no buffers"
    else:
        # A healthy peer is locally fine -- is_initialized is LOCAL state, which
        # is exactly why the group verdict cannot be derived from it.
        assert op.is_initialized is True, f"rank {rank} lost its buffers"

    # (3) The retry at the OLD capacity recovers the group. On the seven healthy
    # ranks this is a same-size free+realloc; on rank 0 it is a full rebuild.
    # Before the no-op agreement this hung: 7 ranks returned early, 1 did not.
    op.reconfigure(max_num_inp_token_per_rank=DECODE_TOKENS)
    assert op.is_initialized is True, f"rank {rank} did not recover"
    _assert_capacity(op, DECODE_TOKENS)
    _run_once(op, config)

    # And the recovered group is still flippable for real.
    op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, config)


def _worker_rank_asymmetric_giveback_fails(rank, world_size):
    """Rank 0 cannot grow; rank 1 grows but then cannot GIVE THE CAPACITY BACK.

    The third asymmetry, and the one the severity code most nearly got wrong.
    In the other two the failing rank is the one the injection hits directly.
    Here the injection hits a rank whose resize SUCCEEDED, on the shrink that
    the group agreement asks it to perform afterwards -- so rank 1 ends holding
    the LARGE buffers while its seven peers hold the small ones. Its symmetric
    heap is a different shape from everyone else's, which is exactly the
    peer-VA class `RegisterStaticHeapSubRegion` cannot detect (it derives peer
    addresses from a shared offset, with no allgather to check).

    Before the fix this reported HEALTHY. The give-back `except` bumped severity
    only `if not self.is_initialized`, and rank 1 IS initialized: C++
    Reconfigure() rolls back to the config it was ENTERED with, and rank 1
    entered the give-back already grown, so the shrink's own rollback restores
    the LARGE buffers. Rank 1 therefore voted SEVERITY_OK, the second
    `_agree_on_severity` saw ROLLED_BACK, and all 8 ranks raised "every rank
    rolled back ... the whole group is still usable at the old capacity" --
    which sglang keys `_MORI_STILL_USABLE_PHRASE` on and treats as recoverable,
    i.e. keep serving on a group with mismatched symmetric buffers. On top of
    that `_restore_mirror()` rewrote rank 1's python mirror to the SMALL
    capacity, so `max_num_tokens_to_recv()` under-reported against buffers that
    were actually large -- a second lie, in the opposite direction.

    Asserted:
      1. all 8 ranks raise the GROUP verdict "GROUP is not usable", and NO rank
         says "still usable at the old capacity" (the split-decision bug);
      2. rank 1's message names BOTH capacities (an operator sizes the retry on
         that number, and "rolled back to 8" would be false here);
      3. rank 1's mirror reports the capacity its buffers ACTUALLY have;
      4. the group still RECOVERS: a retry at the old capacity re-shrinks rank 1
         and all 8 compute correctly afterwards -- which is also the only way to
         show the divergent alloc/free histories did not desynchronize peer VA.
    """
    import os

    config = _make_config(rank, world_size, DECODE_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)
    _run_once(op, config)

    if rank == 0:
        # Rank 0 simply cannot grow. TIMES=1 => its own rollback succeeds, so
        # rank 0 alone would be merely ROLLED_BACK; the group's verdict has to
        # come from rank 1.
        os.environ["MORI_TEST_FAIL_HIPMALLOC"] = "dispSenderIdxMap"
    elif rank == 1:
        # AFTER=1 skips rank 1's grow (allocation 1 of this buffer in this
        # reconfigure) and fires on the give-back shrink (allocation 2). The
        # shrink's own C++ rollback (allocation 3) is past the single fire, so
        # it succeeds -- which is the whole point: rank 1 survives, initialized,
        # at the GROWN capacity.
        os.environ["MORI_TEST_FAIL_HIPMALLOC"] = "dispSenderIdxMap"
        os.environ["MORI_TEST_FAIL_HIPMALLOC_AFTER"] = "1"
    try:
        with pytest.raises(RuntimeError) as excinfo:
            op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
    finally:
        os.environ.pop("MORI_TEST_FAIL_HIPMALLOC", None)
        os.environ.pop("MORI_TEST_FAIL_HIPMALLOC_AFTER", None)

    message = str(excinfo.value)
    # (1) One verdict, on every rank, and it is the honest one.
    assert "GROUP is not usable" in message, f"rank {rank}: {message}"
    assert (
        "still usable at the old capacity" not in message
    ), f"rank {rank} reported the group usable while rank 1 is stranded: {message}"

    if rank == 1:
        # (2) The stranded rank names both capacities rather than claiming a
        # rollback that did not happen.
        assert "stranded at" in message, f"rank 1: {message}"
        assert str(PREFILL_TOKENS) in message and str(DECODE_TOKENS) in message, (
            f"rank 1 must name both capacities: {message}"
        )
        # (3) is_initialized is True (its buffers survived) and the mirror
        # reports the LARGE capacity they actually are, not the small one the
        # give-back intended. A mirror under-reporting here is a silent overrun
        # waiting for the next caller that sizes a tensor off it.
        assert op.is_initialized is True, "rank 1's buffers should have survived"
        _assert_capacity(op, PREFILL_TOKENS)
    else:
        assert op.is_initialized is True, f"rank {rank} lost its buffers"
        _assert_capacity(op, DECODE_TOKENS)

    # (4) The retry at the old capacity must re-shrink rank 1 and leave all 8
    # ranks symmetric and numerically correct. This is the load-bearing
    # assertion: three ranks ran three different alloc/free histories through
    # this resize (rank 0 grow-fail+rollback, rank 1 grow+failed-shrink+rollback,
    # ranks 2-7 grow+shrink), so if that can move a peer VA offset, the
    # check_results=True comparison against the CPU reference is where it shows.
    op.reconfigure(max_num_inp_token_per_rank=DECODE_TOKENS)
    _assert_capacity(op, DECODE_TOKENS)
    _run_once(op, config)

    # And the recovered group is still flippable for real.
    op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, config)


def _worker_rank_asymmetric_finalize_fails(rank, world_size):
    """Rank 0's finalize() RAISES while its N-1 peers finalize normally.

    The teardown counterpart of the reject deadlock, and until 9909de28 it was
    the same bug: python `finalize()` was `barrier / handle.finalize() /
    barrier`, so a rank whose finalize threw unwound out of the middle and
    never entered the trailing barrier. Its peers block there with N-1 of N
    participants -- one rank raises, the rest hang, forever.

    This matters more than its line count suggests. `finalize()` is published
    COLLECTIVE in the COORD contract, and sglang calls it on the
    destroy+recreate fallback that multi-node EP ALWAYS takes ([S, turn 2]),
    so it is the teardown path of the configuration with the least other
    coverage in this campaign.

    The primary assertion is not a string: it is that every rank RETURNS inside
    MORI_TEST_RESULT_TIMEOUT rather than the harness reporting silent ranks.
    A hang here fails as a timeout naming the blocked peers, which is precisely
    the signal the old code could not produce.

    Also asserted:
      1. the raising rank really raises (otherwise the injection did not fire
         and the whole test is vacuous -- the failure mode that made two
         earlier tests in this file assert against nothing);
      2. the peers' finalize SUCCEEDS. Deliberately not a group verdict:
         finalize is destructive by intent, there is no capacity to negotiate
         back, and C++ FinalizeAll() is idempotent and frees what it can, so
         the contract is "barriers stay balanced and the raiser is named", not
         "every rank reports the group's worst outcome" as reconfigure() does.
         Encoding that here keeps 9909de28's deliberate asymmetry from being
         quietly changed later;
      3. every rank -- including the raiser -- can still take part in a
         collective afterwards. A rank that raised past its barrier would leave
         the group off by one, and the very next collective would expose it.
    """
    config = _make_config(rank, world_size, DECODE_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)
    _run_once(op, config)

    if rank == 0:
        # One-shot: the hook disarms itself as it fires, so rank 0's later
        # finalize() (below) is a real one. A permanently-armed hook would make
        # the recovery this test checks impossible.
        os.environ["MORI_TEST_FAIL_FINALIZE"] = "1"

    raised = None
    try:
        try:
            op.finalize()
        except RuntimeError as exc:
            raised = str(exc)
    finally:
        os.environ.pop("MORI_TEST_FAIL_FINALIZE", None)

    if rank == 0:
        # (1) The injection must actually have fired, or nothing below is a test.
        assert raised is not None, (
            "rank 0's finalize() did not raise -- MORI_TEST_FAIL_FINALIZE never "
            "fired, so this test proves nothing about the trailing barrier"
        )
        assert "MORI_TEST_FAIL_FINALIZE" in raised, f"rank 0: {raised}"
    else:
        # (2) Peers are NOT told about rank 0's failure. See the docstring:
        # finalize deliberately has no severity agreement.
        assert raised is None, f"rank {rank}: peers must not inherit a raise: {raised}"
        assert op.is_initialized is False, f"rank {rank} still holds buffers"

    # (3) The group is still collectively usable. If rank 0 had skipped its
    # trailing barrier, it would be one barrier ahead of its peers from here on
    # and this all_reduce would pair rank 0's value with the wrong round.
    # Checked with a value that differs per rank so a mispairing shows up as a
    # wrong sum rather than passing by luck.
    probe = torch.tensor([rank + 1], dtype=torch.int32, device="cpu")
    dist.all_reduce(probe, op=dist.ReduceOp.SUM, group=mori.shmem.shmem_get_process_group())
    expected = world_size * (world_size + 1) // 2
    assert int(probe.item()) == expected, (
        f"rank {rank}: group is off by a collective after a failed finalize "
        f"(got {int(probe.item())}, expected {expected})"
    )

    # And rank 0 -- whose buffers were never released, because the injection
    # throws before any teardown work -- can still finalize for real. That is
    # the difference between "this rank reported a failure" and "this rank is
    # wedged": sglang's fallback path retries the teardown.
    op.finalize()
    assert op.is_initialized is False, f"rank {rank}: retry finalize did not release"


def _worker_rank_asymmetric_reject(rank, world_size):
    """Rank 0 is handed an INVALID capacity; ranks 1-7 are handed a valid one.

    The fourth asymmetry, and until 24b4379e it was a seven-rank DEADLOCK
    rather than a wrong message. The reject fast-path raised AFTER the entry
    `_shmem_barrier` and BEFORE `_agree_on_severity`, so:

      * all 8 ranks vote need_resize=1 and clear the entry barrier;
      * rank 0's config is refused by C++ ValidateReconfigurable and it raised
        out of reconfigure() there and then;
      * ranks 1-7 resized fine, voted SEVERITY_OK, and blocked in
        `_agree_on_severity`'s all_reduce with 7 of 8 participants -- forever.

    One rank raises, seven hang: exactly the group split the severity code was
    introduced to prevent, reintroduced one layer above it. The rationale in
    the code was that making the rejection collective would hang a divergent
    config; the opposite is true, and this test is the demonstration -- a
    divergent config is the ONLY way to reach the branch, and it now completes.

    The test's primary assertion is therefore not a string but the fact that it
    RETURNS: every rank must put a result on the queue inside
    MORI_TEST_RESULT_TIMEOUT rather than the harness reporting silent ranks.

    Also asserted, because a rejection is still not an OOM:
      1. every rank raises;
      2. NO rank's message says "could not grow" (sglang aborts-and-retries on
         that) or "GROUP is not usable" (sglang escalates on that). A config
         that is refused as invalid will be refused identically forever, so
         both of those verdicts send sglang somewhere wrong;
      3. rank 0 gets the C++ detail ("must be > 0"); its peers are told a group
         member refused, since on this bug the peers are the ones that look
         healthy;
      4. every rank is back at the OLD capacity with correct numerics -- the
         peers did resize before learning of the rejection, so the give-back
         has to have run, and the divergent alloc/free histories must not have
         moved a peer VA offset;
      5. a legitimate group-wide flip afterwards still works.
    """
    config = _make_config(rank, world_size, DECODE_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)
    _run_once(op, config)

    # The divergence: rank 0 alone asks for something C++ refuses (> 0 required).
    target = 0 if rank == 0 else PREFILL_TOKENS

    with pytest.raises(RuntimeError) as excinfo:
        op.reconfigure(max_num_inp_token_per_rank=target)

    message = str(excinfo.value)
    assert (
        "could not grow" not in message
    ), f"rank {rank}: a rejection must not wear the OOM wording: {message}"
    assert (
        "GROUP is not usable" not in message
    ), f"rank {rank}: nothing was freed anywhere; do not tell sglang to stop: {message}"

    if rank == 0:
        assert "must be > 0" in message, f"rank 0: {message}"
    else:
        assert "refused the requested config" in message, f"rank {rank}: {message}"

    # Every rank is unchanged, including the seven that DID resize and then gave
    # the capacity back.
    assert op.is_initialized is True, f"rank {rank} lost its buffers"
    _assert_capacity(op, DECODE_TOKENS)
    _run_once(op, config)

    # And a legitimate flip, agreed by the whole group, still works afterwards.
    op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, config)


def _worker_rank_asymmetric_reject_heap_symmetry(rank, world_size):
    """Measure the heap divergence a REJECT leaves behind, WITHOUT dispatching.

    `_worker_rank_asymmetric_reject` hangs on all ranks in the post-combine
    `torch.cuda.synchronize()` of the first `_run_once` after the rejection
    (RESULTS_M T10b, py-spy stack identical on 4/4 ranks). A hang produces no
    diagnostic -- just a queue timeout that misreports live ranks as silent --
    so this variant runs the identical rejection and then reports heap
    accounting instead of driving the kernel that wedges.

    The hypothesis it exists to test: a rejection is the ONE path where ranks
    run different symmetric-heap histories that are never reconciled. Rank 0 is
    refused inside ValidateReconfigurable, i.e. BEFORE anything is freed, so it
    does no heap work at all; its peers pass validation, allocate at PREFILL,
    then give the capacity back -- two full free/alloc round trips. A first-fit
    VA manager fed different histories can land the same logical buffer at a
    different offset per rank, and `RegisterStaticHeapSubRegion`
    (symmetric_memory.cpp:392) computes peers as `peerPtrs[i] + offset` with no
    allgather to check. That is review #4 item 2's warning, and a combine
    spinning on a peer flag that will never be written is exactly a hang in the
    post-combine sync.

    `test_rank_asymmetric_failure` does NOT cover this: there rank 0 attempts
    the allocation and rolls back, so every rank does free+alloc.

    This test asserts only what it can prove. If the heap returns to baseline
    identically on every rank, the hypothesis is REFUTED and the hang is
    somewhere else (kernel state, the give-back's device scratch) -- which is
    just as useful, and is why the assertion is written to fail loudly with the
    per-rank numbers either way.
    """
    config = _make_config(rank, world_size, DECODE_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)
    _run_once(op, config)

    baseline = _heap_stats()
    if baseline is None:
        # Not in static-heap mode, so there is nothing to measure. Return
        # rather than pytest.skip(): Skipped is a BaseException and escaping it
        # here leaves the parent's collective get() waiting forever (be825794).
        return

    target = 0 if rank == 0 else PREFILL_TOKENS
    with pytest.raises(RuntimeError):
        op.reconfigure(max_num_inp_token_per_rank=target)

    after = _heap_stats()

    # 1. Local accounting: a rejection must be heap-neutral on EVERY rank --
    #    rank 0 never allocated, and its peers gave back everything they took.
    #    Exact equality is the right bar: this heap is a deterministic first-fit
    #    VA manager, not a caching allocator, so a closed loop must land on the
    #    same numbers.
    for key in ("total_free_space", "allocated_blocks", "num_mem_objs"):
        assert after[key] == baseline[key], (
            f"rank {rank}: {key} moved across a REJECTED reconfigure: "
            f"{baseline[key]} -> {after[key]}. Nothing should have been "
            f"allocated or freed net on any rank. "
            f"(rank 0 rejects before freeing; peers resize then give back.)"
        )

    # 2. Cross-rank symmetry: the numbers must AGREE across ranks, not merely
    #    return to each rank's own baseline. Rank 0 and its peers ran different
    #    histories to get here, and it is the divergence -- not the drift --
    #    that would move a peer VA offset. Gathered on CPU: the shmem group can
    #    be gloo-only (bf9757e9).
    if dist.is_initialized():
        from mori.shmem import shmem_get_process_group

        group = shmem_get_process_group()
        mine = torch.tensor(
            [
                int(after["total_free_space"]),
                int(after["allocated_blocks"]),
                int(after["num_mem_objs"]),
            ],
            dtype=torch.int64,
            device="cpu",
        )
        gathered = [torch.zeros_like(mine) for _ in range(world_size)]
        dist.all_gather(gathered, mine, group=group)
        rows = [tuple(int(v) for v in g.tolist()) for g in gathered]
        assert len(set(rows)) == 1, (
            f"rank {rank}: symmetric heap DIVERGED across ranks after a "
            f"rejection -- (total_free_space, allocated_blocks, num_mem_objs) "
            f"per rank = {rows}. Ranks that ran different alloc/free histories "
            f"are no longer at the same VA offsets, and "
            f"RegisterStaticHeapSubRegion assumes they are."
        )

    # 3. The op still reports itself usable at the old capacity on every rank.
    assert op.is_initialized is True, f"rank {rank} lost its buffers"
    _assert_capacity(op, DECODE_TOKENS)


def _worker_rank_asymmetric_reject_barrier_state(rank, world_size, do_reject=True):
    """MEASURE what the group disagrees about after a REJECT. No dispatch.

    Two fixes for the reject hang have been aimed at hypotheses inferred from
    outside the process and both were refuted after the fact: the peer-VA
    offset story (T10, `1b407d45`, refuted by the heap-symmetry probe) and the
    barrier-GENERATION story (T11, `a430c6e2`, refuted at world 8 in T12b with
    the binding confirmed present in the shipped .so). This test stops guessing
    and reads the three things a cross-device barrier actually depends on:

      1. the GENERATION each rank holds (`crossDeviceBarrierFlag[0]`) -- every
         barrier atomicAdds it and then spins until its peers publish the same
         value (intranode.hpp:69,73), so the group progresses only while all
         ranks agree;
      2. each rank's LOCAL address for the symmetric barrier buffer;
      3. what each rank believes its PEERS' copies are at.

    (3) is the one nothing has ever checked, and it is not implied by (and
    therefore not refuted by) the heap-symmetry probe: that compares heap
    ACCOUNTING (total_free_space, block counts), and a deterministic first-fit
    VA manager can return identical totals for different OFFSETS.

    The symmetric-heap contract is stronger than "same free space", but it is
    NOT "peer_ptrs[i] as seen by any rank == local_ptr as seen by rank i" --
    that was this test's first formulation and it is wrong (see the long note
    at the assertion; peerPtrs holds viewer-local IPC mappings, so it fails on
    a healthy group). The real contract is that a symmetric allocation lands at
    the SAME OFFSET from the heap base on every rank, because
    `RegisterStaticHeapSubRegion` (application/memory/symmetric_memory.cpp:392)
    derives every peer address as `peerPtrs[i] + offset` using the local rank's
    offset for all i, with no allgather to check it.

    Deliberately does NOT dispatch: `_worker_rank_asymmetric_reject` wedges in
    the post-combine sync, and a wedge yields no diagnostic at all -- just a
    harness timeout naming ranks that are in fact alive, which is how three
    turns of this campaign were spent. This one always reports numbers.

    Assertions are written so that a failure PRINTS the full per-rank table
    for whichever invariant broke. A green here refutes all three mechanisms at
    once and moves the search to kernel-resident state, which is equally
    useful; the point is that either outcome is a measurement.
    """
    config = _make_config(rank, world_size, DECODE_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)
    _run_once(op, config)

    # FAIL, do not return, when the binding is missing. An early return here is
    # indistinguishable from a real green, and that is not hypothetical: T13's
    # first run of this test reported "1 passed" against a .so whose build had
    # failed (BUILD_RC=1), because the wrapper tolerates an older extension.
    # A diagnostic that silently reports success when it measured nothing is
    # worse than no diagnostic. (raise, not pytest.skip: Skipped derives from
    # BaseException and escaping it strands the parent's get() -- be825794.)
    assert op.probe_barrier_state() is not None, (
        f"rank {rank}: this mori extension has no probe_barrier_state binding, "
        f"so NOTHING was measured. The .so is older than the python. Rebuild "
        f"and check BUILD_RC before reading this test's result."
    )

    # BEFORE. A cross-rank comparison at one instant can only find a state that
    # is asymmetric right now; it cannot see a rank whose barrier object MOVED
    # while another rank's cached view of it did not. That delta is the thing a
    # reject could actually break, because a reject is the one outcome where
    # some ranks free and re-allocate the symmetric barrier twice and one rank
    # does no heap work at all. So snapshot first and compare against the
    # snapshot, not only across ranks.
    before = op.probe_barrier_state()

    # The divergence: rank 0 alone asks for something C++ refuses (> 0 required).
    # In the CONTROL arm every rank passes a valid target, so the resize path,
    # the barriers and the probes are identical and only the reject is removed.
    # Without that arm a red here is unattributable: it could be the reject or
    # it could be anything a resize does, and this test has already produced one
    # false positive (T13) by asserting an invariant that never held.
    if do_reject:
        target = 0 if rank == 0 else PREFILL_TOKENS
        with pytest.raises(RuntimeError):
            op.reconfigure(max_num_inp_token_per_rank=target)
    else:
        op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)

    probe = op.probe_barrier_state()
    assert probe["initialized"] is True, f"rank {rank} lost its buffers"

    if not dist.is_initialized():
        return
    from mori.shmem import shmem_get_process_group

    group = shmem_get_process_group()

    # The cross-rank-comparable form of a symmetric address is its OFFSET from
    # this rank's static-heap base, NOT the raw pointer: each rank's heap is
    # hipMalloc'd wherever its own allocator lands it (shmem/init.cpp:226), so
    # raw local_ptr values differ on a perfectly healthy group. heap_base comes
    # from shmem_get_heap_stats() and is None outside static-heap mode.
    _hs = mori.shmem.shmem_get_heap_stats()
    heap_base = int(_hs["heap_base"]) if _hs else 0

    # Gather (generation, seed, heap-relative offset, size) from every rank.
    # CPU tensors: the shmem group can be gloo-only (bf9757e9).
    mine = torch.tensor(
        [
            int(probe["generation"]),
            int(probe["seed"]),
            int(probe["local_ptr"]) - heap_base,
            int(probe["size"]),
        ],
        dtype=torch.int64,
        device="cpu",
    )
    gathered = [torch.zeros_like(mine) for _ in range(world_size)]
    dist.all_gather(gathered, mine, group=group)
    rows = [tuple(int(v) for v in g.tolist()) for g in gathered]
    gens = [r[0] for r in rows]
    local_offsets = [r[2] for r in rows]

    # peer_ptrs are likewise only comparable as heap-relative offsets, and only
    # against the VIEWER's own peer-heap base -- see the assertion below for why
    # comparing them to local_ptrs is wrong.
    peers = torch.tensor(
        [int(p) for p in probe["peer_ptrs"]] or [0] * world_size,
        dtype=torch.int64,
        device="cpu",
    )
    gathered_peers = [torch.zeros_like(peers) for _ in range(world_size)]
    dist.all_gather(gathered_peers, peers, group=group)
    peer_rows = [[int(v) for v in g.tolist()] for g in gathered_peers]

    # (1) GENERATION. If this is what diverges, a430c6e2 did not take effect on
    #     the reject path and the T11 mechanism is back in play.
    assert len(set(gens)) == 1, (
        f"rank {rank}: barrier GENERATION diverged after a REJECT -- per-rank "
        f"generation = {gens} (seeds {[r[1] for r in rows]}). Every barrier "
        f"spins until all ranks publish the SAME generation "
        f"(intranode.hpp:69,73), so the next collective wedges. This is the "
        f"T11/a430c6e2 mechanism: the rejecting rank is refused inside "
        f"ValidateReconfigurable BEFORE FinalizeAll, so it never re-seeds."
    )

    # (2)+(3) ADDRESSES -- as OFFSETS, which is the only comparable form.
    #
    # An earlier revision of this test asserted `peer_rows[viewer] ==
    # local_ptrs`, i.e. that what rank j believes rank i's buffer is at equals
    # the raw pointer rank i reports. **That assertion is invalid and it fired
    # a false positive** (T13, logged in RESULTS_M): it fails on a perfectly
    # healthy group, for two independent reasons read out of the source rather
    # than inferred:
    #
    #   * For every non-RDMA peer, symmetric_memory.cpp:169-177 OVERWRITES
    #     peerPtrs[i] with p2pPeerPtrs[i], the address hipIpcOpenMemHandle
    #     returned *in the viewer's own address space*. So peerPtrs is by
    #     construction a viewer-local mapping, never the peer's local VA.
    #   * Even the pre-overwrite allgathered values (:120) would not match,
    #     because each rank's static heap is hipMalloc'd wherever its own
    #     allocator lands it.
    #
    # The measured data said the same thing: rank 0's peer_ptrs were uniformly
    # strided by exactly the 8 GiB heap size (8 consecutive IPC mappings in one
    # VA space) with peer_ptrs[0] == its own local_ptr, while the true
    # local_ptrs were scattered. That is a healthy P2P mapping table.
    #
    # The invariant RegisterStaticHeapSubRegion (:392) actually assumes -- and
    # the one a resize could really break -- is that the sub-region's OFFSET
    # from the heap base is identical on every rank, since it derives peers as
    # `heapObj->peerPtrs[i] + offset` using the LOCAL rank's offset for all i.
    assert len(set(local_offsets)) == 1, (
        f"rank {rank}: symmetric HEAP OFFSET diverged after a REJECT -- "
        f"per-rank offset of the barrier buffer from its own heap base = "
        f"{local_offsets}. RegisterStaticHeapSubRegion "
        f"(application/memory/symmetric_memory.cpp:392) derives every peer "
        f"address as peerPtrs[i] + offset using the LOCAL offset for all i, "
        f"with no allgather to check, so unequal offsets mean each rank is "
        f"reading and writing its peers at the wrong address. A REJECT is the "
        f"one outcome where a rank does zero heap work while its peers do two "
        f"full free/alloc round trips."
    )

    # Self-consistency of the mapping table: every rank's view of ITSELF must
    # be its own local pointer. This is the one peerPtrs entry that IS directly
    # comparable (symmetric_memory.cpp:127 sets p2pPeerPtrs[rank] = localPtr),
    # and it is cheap insurance that the table was rebuilt at all.
    for viewer, seen in enumerate(peer_rows):
        if not seen or all(v == 0 for v in seen):
            continue  # no peerPtrs on this rank (non-static-heap mode)
        assert len(set(seen)) == len(seen), (
            f"rank {rank}: rank {viewer}'s peer mapping table has DUPLICATE "
            f"addresses after a REJECT: {seen}. Two peers mapped to one "
            f"address means writes to one silently land on the other."
        )

    # Every rank agrees on the buffer size too -- a size mismatch would make
    # the spin read past what its peers publish.
    assert len({r[3] for r in rows}) == 1, (
        f"rank {rank}: barrier buffer SIZE diverged: {[r[3] for r in rows]}"
    )

    # (4) THE DELTA -- the invariant the cross-rank checks above structurally
    #     cannot see, and the one a REJECT is uniquely positioned to break.
    #
    # After a reject: the rejecting rank was refused inside
    # ValidateReconfigurable BEFORE FinalizeAll, so it did zero heap work and
    # its `peer_ptrs` table is the one built at construction. Its peers, by
    # contrast, freed and re-allocated the barrier object TWICE (grow, then
    # give-back). If a peer's barrier landed at a different heap offset on the
    # way back, that peer's own view is self-consistent and the cross-rank
    # offset check at (2) still passes -- because every rank recomputes its
    # offset from its own current state -- while the rejecting rank keeps
    # pointing at where that buffer USED to be. Nobody writes the address it
    # spins on, and the group wedges with no host-visible asymmetry anywhere.
    # That is exactly the observed signature.
    #
    # Comparing each rank's own before/after is what detects it, and it is
    # valid in both arms: peer_ptrs are viewer-local mappings, but they are the
    # SAME viewer's mappings at two instants, so they are directly comparable
    # (unlike the cross-rank form that produced T13's false positive).
    moved = {}
    if int(before["local_ptr"]) != int(probe["local_ptr"]):
        moved["local_ptr"] = (int(before["local_ptr"]), int(probe["local_ptr"]))
    if int(before["size"]) != int(probe["size"]):
        moved["size"] = (int(before["size"]), int(probe["size"]))
    b_peers = [int(p) for p in before["peer_ptrs"]]
    a_peers = [int(p) for p in probe["peer_ptrs"]]
    if b_peers != a_peers:
        moved["peer_ptrs"] = (b_peers, a_peers)

    # Did ANY rank move? Gather the boolean so that a rank which stayed put
    # (the rejecter) reports its peers' movement rather than passing quietly.
    # The rejecter is precisely the rank that cannot see the problem locally.
    my_moved = torch.tensor([1 if moved else 0], dtype=torch.int64, device="cpu")
    gathered_moved = [torch.zeros_like(my_moved) for _ in range(world_size)]
    dist.all_gather(gathered_moved, my_moved, group=group)
    moved_ranks = [i for i, g in enumerate(gathered_moved) if int(g.item())]

    arm = "REJECT" if do_reject else "CONTROL (no reject)"
    assert not moved_ranks, (
        f"rank {rank} [{arm}]: the symmetric barrier object MOVED across the "
        f"resize on rank(s) {moved_ranks}. This rank's own delta = {moved}. "
        f"A rank that did no heap work (the rejecter) keeps a peer_ptrs entry "
        f"addressing the OLD allocation, so its barrier spin waits on memory "
        f"nobody writes -- a wedge with no host-visible cross-rank asymmetry, "
        f"which is why every instantaneous probe so far has come back green. "
        f"If this fires only in the REJECT arm and not in the CONTROL arm, the "
        f"reject is the cause; if it fires in both, any resize does it and the "
        f"reject is merely the case with no re-seed to hide it."
    )


def _worker_reconfigure_rejects_immutable_fields(rank, world_size):
    """numQpPerPe / useExternalInpBuffer are peer-visible; they must be frozen.

    Reconfigure does NOT rebuild the shmem context that owns numQpPerPe, and
    use_external_inp_buf decides whether the dispatch input buffer is allocated
    at all -- letting either change would desynchronize the symmetric layout.
    """
    config = _make_config(rank, world_size, PREFILL_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)

    for field, bad_value in (
        ("use_external_inp_buf", not config.use_external_inp_buf),
        ("num_qp_per_pe", op._cpp_config.num_qp_per_pe + 1),
    ):
        good = getattr(op._cpp_config, field)
        setattr(op._cpp_config, field, bad_value)
        try:
            with pytest.raises(RuntimeError, match="must not change"):
                op._handle.reconfigure(op._cpp_config)
        finally:
            setattr(op._cpp_config, field, good)

    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, config)


def _worker_heap_fragmentation(rank, world_size, cycles):
    """After N grow/shrink cycles the heap must still admit the LARGE size.

    mori's static heap is a first-fit VA manager with coalescing
    (`HeapVAManager`). Repeated free-then-realloc-at-a-different-size is
    exactly the pattern that fragments a first-fit allocator, and a role switch
    does it once per flip. If cycle N could allocate prefill capacity but cycle
    N+1 cannot, the feature has a hard lifetime limit -- that would be a real
    negative result worth reporting, so assert it explicitly rather than
    trusting coalescing.
    """
    config = _make_config(rank, world_size, PREFILL_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)

    start = _heap_stats()

    # Vary the requested capacity so successive allocations are NOT the same
    # size -- same-size realloc trivially reuses the just-freed block and would
    # not exercise fragmentation at all.
    trace = []
    for i in range(cycles):
        op.reconfigure(max_num_inp_token_per_rank=DECODE_TOKENS + i)
        op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS + i)
        if start is not None:
            trace.append(_heap_stats())

    # The real assertion: the largest size still fits after all that churn.
    op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, config)

    if start is None:
        return

    # Quantify the fragmentation instead of only asserting "it still worked".
    # `largest_free_block` collapsing while `total_free_space` holds steady is
    # the signature of a fragmenting first-fit heap, and it would put a hard
    # ceiling on how many times an sglang instance may flip role. Report the
    # trace in the failure message so a regression comes with its own data.
    end = _heap_stats()
    summary = (
        f"rank {rank}: cycles={cycles} "
        f"start(free={start['total_free_space']} largest={start['largest_free_block']} "
        f"blocks={start['total_blocks']}) "
        f"end(free={end['total_free_space']} largest={end['largest_free_block']} "
        f"blocks={end['total_blocks']}) "
        f"largest_free_block_trace={[s['largest_free_block'] for s in trace]} "
        f"total_blocks_trace={[s['total_blocks'] for s in trace]}"
    )
    if rank == 0:
        print("[heap-fragmentation] " + summary, flush=True)

    # The VA manager coalesces adjacent free blocks, so the block count must
    # not grow without bound across flips. A slack of 2x the starting count
    # allows for the churn of a single in-flight resize while still failing on
    # per-cycle accumulation (which at 20 cycles would be an order of magnitude
    # more).
    assert end["total_blocks"] <= max(start["total_blocks"] * 2, start["total_blocks"] + 8), (
        "symmetric heap VA blocks accumulate per flip cycle -- role switch has "
        f"a lifetime limit. {summary}"
    )


def _worker_public_reject_restores_mirror(rank, world_size):
    """A rejected `op.reconfigure(...)` must not leave the python mirror lying.

    The other rejection tests poke `op._cpp_config` and call
    `op._handle.reconfigure()` directly, which bypasses the python wrapper
    entirely -- so the wrapper's own error path had no coverage. The wrapper
    assigns the requested capacity into `_cpp_config` BEFORE calling into C++,
    and if it does not restore it on the throw then `max_num_tokens_to_recv()`
    over-reports against buffers that were never resized: a silent overrun,
    not an exception.

    `max_num_inp_token_per_rank=0` is the reachable-through-the-public-API
    rejection (C++ ValidateReconfigurable requires > 0), and it is rejected
    before anything is freed.
    """
    config = _make_config(rank, world_size, PREFILL_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)
    _run_once(op, config)

    recv_before = op.max_num_tokens_to_recv()
    send_before = op.max_num_tokens_to_send()

    with pytest.raises(RuntimeError, match="must be > 0"):
        op.reconfigure(max_num_inp_token_per_rank=0)

    # The mirror must describe the buffers that ACTUALLY exist.
    assert op.max_num_tokens_to_recv() == recv_before
    assert op.max_num_tokens_to_send() == send_before
    assert op._cpp_config.max_num_inp_token_per_rank == PREFILL_TOKENS
    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, config)

    # ...and a legitimate flip after the rejection still works, i.e. the failed
    # attempt left no residue in either the mirror or the heap.
    op.reconfigure(max_num_inp_token_per_rank=DECODE_TOKENS)
    _assert_capacity(op, DECODE_TOKENS)
    _run_once(op, config)


def _worker_reconfigure_max_total_recv(rank, world_size):
    """`max_total_recv_tokens` is in the M<->S contract, so it must resize too.

    It caps the RECEIVE side independently of the per-rank send capacity:
    MaxNumTokensToRecvPerRank() = min(ceil(cap/world), max_num_inp_token_per_rank).
    Every other test in this file passes only max_num_inp_token_per_rank, so
    this kwarg was contract surface with zero coverage -- either it works or it
    comes out of the contract.

    `spread` routing (one expert per rank per token) makes every rank receive
    every source token, so the recv buffers are driven to their true worst case
    and an under-sized recv buffer would corrupt rather than pass silently.
    """
    # spread routing requires num_experts_per_token == world_size.
    def cfg(tokens, recv_cap):
        c = _make_config(rank, world_size, tokens)
        c.num_experts_per_token = world_size
        c.max_total_recv_tokens = recv_cap
        return c

    # Start at prefill capacity with an exact-worst-case recv cap.
    config = cfg(PREFILL_TOKENS, PREFILL_TOKENS * world_size)
    op = mori.ops.EpDispatchCombineOp(config)
    assert op.max_num_tokens_to_recv() == PREFILL_TOKENS * world_size
    _run_once(op, config, routing="spread")

    # Flip to decode: BOTH capacities shrink together, the way a real role
    # switch would move them.
    decode_cap = DECODE_TOKENS * world_size
    op.reconfigure(
        max_num_inp_token_per_rank=DECODE_TOKENS, max_total_recv_tokens=decode_cap
    )
    _assert_capacity(op, DECODE_TOKENS)
    assert op.max_num_tokens_to_recv() == decode_cap, (
        f"rank {rank}: recv capacity did not follow max_total_recv_tokens: "
        f"got {op.max_num_tokens_to_recv()}, want {decode_cap}"
    )
    assert op.config.max_total_recv_tokens == decode_cap
    _run_once(op, cfg(DECODE_TOKENS, decode_cap), routing="spread")

    # Flip back, and this time move ONLY max_total_recv_tokens on the way --
    # max_num_inp_token_per_rank defaults to "keep current", so this is the
    # kwarg on its own.
    op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
    assert op.config.max_total_recv_tokens == decode_cap
    # recv is clamped by BOTH terms, so with the small cap still in force the
    # recv side must NOT have grown with the send side.
    assert op.max_num_tokens_to_recv() == decode_cap, (
        f"rank {rank}: recv capacity ignored the still-small "
        f"max_total_recv_tokens={decode_cap} after the send side grew: "
        f"got {op.max_num_tokens_to_recv()}"
    )

    op.reconfigure(max_total_recv_tokens=PREFILL_TOKENS * world_size)
    assert op.max_num_tokens_to_recv() == PREFILL_TOKENS * world_size
    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, cfg(PREFILL_TOKENS, PREFILL_TOKENS * world_size), routing="spread")


def _worker_reconfigure_noop_and_finalize(rank, world_size):
    """Same-capacity reconfigure is a no-op; finalize is idempotent."""
    config = _make_config(rank, world_size, PREFILL_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)

    op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, config)

    op.finalize()
    assert not op.is_initialized
    op.finalize()  # idempotent
    assert not op.is_initialized
    # The destructor must not double-free after an explicit finalize.
    del op


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("world_size", _WORLD_SIZES)
@pytest.mark.parametrize("kernel_type", _KERNEL_TYPES)
def test_reconfigure_flip_and_flip_back(
    torch_dist_process_manager, world_size, kernel_type
):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_flip_and_flip_back, [world_size, kernel_type])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


# fp8 paths allocate extra scale buffers off the same capacity fields, so the
# rebuild has to resize those too.
@pytest.mark.parametrize("world_size", _WORLD_SIZES)
@pytest.mark.parametrize("quant_type", ("fp8_direct_cast", "fp8_blockwise"))
def test_reconfigure_flip_and_flip_back_quant(
    torch_dist_process_manager, world_size, quant_type
):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_flip_and_flip_back, [world_size, "IntraNode", quant_type])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
@pytest.mark.parametrize("cycles", (5,))
def test_reconfigure_leak_stress(torch_dist_process_manager, world_size, cycles):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_leak_stress, [world_size, cycles])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_finalize_returns_everything(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_finalize_returns_everything, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_reconfigure_public_reject_restores_mirror(
    torch_dist_process_manager, world_size
):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_public_reject_restores_mirror, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_reconfigure_max_total_recv_tokens(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_reconfigure_max_total_recv, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_reconfigure_rejects_layout_change(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_reconfigure_rejects_layout_change, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_reconfigure_noop_and_finalize(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_reconfigure_noop_and_finalize, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_reconfigure_oom_rolls_back(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_reconfigure_oom_rolls_back, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_plain_device_oom_raises(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_plain_device_oom_raises, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_repeated_failed_flips_do_not_accumulate(
    torch_dist_process_manager, world_size
):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_repeated_failed_flips_do_not_accumulate, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_rank_asymmetric_failure(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_rank_asymmetric_failure, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_rank_asymmetric_unrecoverable(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_rank_asymmetric_unrecoverable, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_flip_at_real_capacities(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_flip_at_real_capacities, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_finalized_getters_raise(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_finalized_getters_raise, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_rank_asymmetric_reject_heap_symmetry(torch_dist_process_manager, world_size):
    """Diagnostic for the T10b hang: does a REJECT leave the heap asymmetric?

    Split out of `test_rank_asymmetric_reject`, which hangs (RESULTS_M T10b) in
    the dispatch/combine AFTER the rejection -- and a hang yields no signal at
    all, just a harness timeout naming ranks that are actually alive. This test
    stops short of that dispatch and reports the numbers instead, so the next
    turn starts from a measurement rather than the hypothesis.
    """
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_rank_asymmetric_reject_heap_symmetry, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_rank_asymmetric_finalize_fails(torch_dist_process_manager, world_size):
    """One rank's finalize() raises; the other N-1 must not hang.

    The test 9909de28 has been owed since review #36. Its primary assertion is
    that all N workers RETURN inside the harness timeout.
    """
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_rank_asymmetric_finalize_fails, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
@pytest.mark.parametrize("do_reject", [False, True], ids=["control", "reject"])
def test_rank_asymmetric_reject_barrier_state(
    torch_dist_process_manager, world_size, do_reject
):
    """Which barrier invariant does a REJECT break -- and does a resize alone?

    The measurement `test_rank_asymmetric_reject`'s hang cannot give: it stops
    before the dispatch that wedges and compares the generation, the local
    barrier addresses and the peer pointers across all ranks, plus each rank's
    own BEFORE/AFTER delta.

    Runs in two arms so a result is ATTRIBUTABLE. `control` gives every rank a
    valid target, so the resize, both barriers and both probes happen exactly
    as in `reject` and only the rejection is removed. Without it, a red in the
    reject arm cannot be told apart from "any resize does this", and a green
    cannot be told apart from "the probe measures nothing" -- this test has
    already produced one false positive (T13) by asserting an invariant that
    never held on a healthy group.
    """
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_rank_asymmetric_reject_barrier_state, [world_size, do_reject])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_rank_asymmetric_reject(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_rank_asymmetric_reject, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_rank_asymmetric_giveback_fails(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_rank_asymmetric_giveback_fails, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_reconfigure_rejects_immutable_fields(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_reconfigure_rejects_immutable_fields, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
@pytest.mark.parametrize("cycles", (20,))
def test_reconfigure_heap_fragmentation(
    torch_dist_process_manager, world_size, cycles
):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_heap_fragmentation, [world_size, cycles])
        )
    assert_worker_results(torch_dist_process_manager, world_size)
