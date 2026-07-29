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
import pytest
import torch

import mori
from tests.python.ops.dispatch_combine_test_utils import (
    EpDispatchCombineTestCase,
    assert_worker_results,
)

# "prefill" (large chunked-prefill token count) vs "decode" (per-step batch).
PREFILL_TOKENS = 128
DECODE_TOKENS = 8


# Kernel types reachable with a single node of 8 GPUs. The InterNode* paths
# need real multi-node RDMA and AsyncLL needs MORI_ENABLE_SDMA set before shmem
# init, so both are left to their own coverage (see backlog in RESULTS_M.md).
_KERNEL_TYPES = ("IntraNode", "IntraNodeLL")


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
    # Every rank raises: the seven that succeeded must not silently keep the
    # capacity they grew to, because their peers do not have it.
    if rank == 0:
        assert "could not grow" in message, f"rank 0: {message}"
    else:
        assert "another rank in the EP group" in message, f"rank {rank}: {message}"
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
@pytest.mark.parametrize("world_size", (8,))
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
@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("quant_type", ("fp8_direct_cast", "fp8_blockwise"))
def test_reconfigure_flip_and_flip_back_quant(
    torch_dist_process_manager, world_size, quant_type
):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_flip_and_flip_back, [world_size, "IntraNode", quant_type])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("cycles", (5,))
def test_reconfigure_leak_stress(torch_dist_process_manager, world_size, cycles):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_leak_stress, [world_size, cycles])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", (8,))
def test_finalize_returns_everything(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_finalize_returns_everything, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", (8,))
def test_reconfigure_public_reject_restores_mirror(
    torch_dist_process_manager, world_size
):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_public_reject_restores_mirror, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", (8,))
def test_reconfigure_max_total_recv_tokens(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_reconfigure_max_total_recv, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", (8,))
def test_reconfigure_rejects_layout_change(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_reconfigure_rejects_layout_change, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", (8,))
def test_reconfigure_noop_and_finalize(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_reconfigure_noop_and_finalize, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", (8,))
def test_reconfigure_oom_rolls_back(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_reconfigure_oom_rolls_back, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", (8,))
def test_plain_device_oom_raises(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_plain_device_oom_raises, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", (8,))
def test_repeated_failed_flips_do_not_accumulate(
    torch_dist_process_manager, world_size
):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_repeated_failed_flips_do_not_accumulate, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", (8,))
def test_rank_asymmetric_failure(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_rank_asymmetric_failure, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", (8,))
def test_reconfigure_rejects_immutable_fields(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_reconfigure_rejects_immutable_fields, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("cycles", (20,))
def test_reconfigure_heap_fragmentation(
    torch_dist_process_manager, world_size, cycles
):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_heap_fragmentation, [world_size, cycles])
        )
    assert_worker_results(torch_dist_process_manager, world_size)
