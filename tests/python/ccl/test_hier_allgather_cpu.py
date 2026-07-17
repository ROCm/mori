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
"""CPU bit-exact spec test for the hierarchical AllGather offset math.

This validates ``hier_allgather_reference`` -- the executable specification of
the 3-phase hierarchical AllGather data movement (intra-node SDMA gather ->
inter-node RDMA gather -> placement) -- WITHOUT any GPU, SDMA or RDMA. It
proves the byte/element offset arithmetic reproduces, bit-exactly, the
rank-major ordering of ``torch.distributed.all_gather_into_tensor`` for the
N>=2 (multi-node) decomposition before the device kernels (M2) are wired up.

The torch reference for AllGather is a pure data move: the output is simply
``concat(shard[0], ..., shard[world-1])`` in rank order. We assert
``torch.equal`` (zero numerical tolerance) for every rank's output.

Run (CPU only, no GPU needed):
    PYTHONPATH=<worktree>:<worktree>/python python3 \
        tests/python/ccl/test_hier_allgather_cpu.py
"""

import sys

import torch

# Import the reference straight from the module file so this test does not
# require the C++ .so (hier_allgather.py's only hard dep is torch for this fn).
from mori.ccl.hier_allgather import (
    HierAllGather,
    hier_allgather_reference,
    inter_node_ring_reference,
)

# (num_nodes N, ranks_per_node G) layouts to exercise. Includes the DESIGN.md
# contract case N=2,G=4 plus a few others to stress the offset math.
LAYOUTS = [
    (1, 4),  # degenerate single-node (M1)
    (2, 4),  # DESIGN.md contract world=8
    (2, 8),  # full node
    (3, 2),  # uneven N
    (4, 1),  # one rank per node (pure inter-node)
]

# Per-rank element counts (must be multiple of 4 bytes; all are). Small-ish so
# the CPU test stays fast while still covering odd/non-power-of-two counts.
COUNTS = [1, 7, 1024, 4099]

DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.int32]


def _make_shard(count: int, dtype: torch.dtype, rank: int) -> torch.Tensor:
    """Deterministic, rank-distinct shard so cross-rank mixups are detectable."""
    if dtype == torch.int32:
        base = torch.arange(count, dtype=torch.int32) + rank * 1_000_003
        return base
    # Floating: distinct per (rank, index); exactly representable small ints
    # scaled so bf16/fp16 round-trip is exact (values are integers < 2^8).
    vals = (torch.arange(count, dtype=torch.float32) % 97) + (rank % 13)
    return vals.to(dtype)


def _torch_reference(shards):
    """Ground-truth AllGather output (same on every rank): rank-major concat."""
    return torch.cat([s.reshape(-1) for s in shards])


def _intra_gather(shards, N, G):
    """Phase 1 (SDMA on device): each node's contiguous G-shard block."""
    count = shards[0].numel()
    dtype = shards[0].dtype
    blocks = []
    for n in range(N):
        block = torch.empty(count * G, dtype=dtype)
        for g in range(G):
            block[g * count : (g + 1) * count] = shards[n * G + g].reshape(-1)
        blocks.append(block)
    return blocks


def run_ring() -> int:
    """Validate the inter-node ring schedule (M2 RDMA phase) bit-exactly.

    Composes the real phases the device path will run: intra-node gather
    (SDMA) -> inter-node ring (RDMA, AllGatherRingKernel schedule) ->
    intra-node placement broadcast, and asserts every rank's output equals
    the rank-major concat (== torch.distributed.all_gather_into_tensor).
    """
    failures = 0
    checks = 0
    for N, G in LAYOUTS:
        world = N * G
        for dtype in DTYPES:
            for count in COUNTS:
                shards = [_make_shard(count, dtype, r) for r in range(world)]
                expected = _torch_reference(shards)

                node_blocks = _intra_gather(shards, N, G)
                leader_bufs = inter_node_ring_reference(node_blocks)

                # Placement: rank r (node r//G) gets its leader's full buffer.
                for r in range(world):
                    checks += 1
                    out = leader_bufs[r // G]
                    if not torch.equal(out, expected):
                        print(
                            f"FAIL ring N={N} G={G} dtype={dtype} count={count} "
                            f"rank={r}: not bit-exact vs rank-major concat"
                        )
                        failures += 1
    if failures:
        print(f"\n{failures} ring FAILED ({checks} rank-checks total)")
        return 1
    print(
        f"PASSED ring — intra-gather + inter-node ring + placement bit-exact "
        f"across {len(LAYOUTS)} layouts x {len(DTYPES)} dtypes x "
        f"{len(COUNTS)} sizes ({checks} rank-checks)."
    )
    return 0


def run() -> int:
    failures = 0
    checks = 0
    for N, G in LAYOUTS:
        world = N * G
        for dtype in DTYPES:
            for count in COUNTS:
                shards = [_make_shard(count, dtype, r) for r in range(world)]
                expected = _torch_reference(shards)
                outputs = hier_allgather_reference(shards, N, G)

                if len(outputs) != world:
                    print(
                        f"FAIL N={N} G={G} dtype={dtype} count={count}: "
                        f"got {len(outputs)} outputs, expected {world}"
                    )
                    failures += 1
                    continue

                for r, out in enumerate(outputs):
                    checks += 1
                    if out.numel() != expected.numel() or out.dtype != dtype:
                        print(
                            f"FAIL N={N} G={G} dtype={dtype} count={count} "
                            f"rank={r}: shape/dtype mismatch"
                        )
                        failures += 1
                        continue
                    if not torch.equal(out, expected):
                        print(
                            f"FAIL N={N} G={G} dtype={dtype} count={count} "
                            f"rank={r}: not bit-exact vs rank-major concat"
                        )
                        failures += 1

    if failures:
        print(f"\n{failures} FAILED ({checks} rank-checks total)")
        return 1
    print(
        f"PASSED — hier_allgather_reference bit-exact vs rank-major AllGather "
        f"across {len(LAYOUTS)} layouts x {len(DTYPES)} dtypes x "
        f"{len(COUNTS)} sizes ({checks} rank-checks)."
    )
    return 0


class _RecordingIntra:
    """Stub intra-gather phase: records the ``prepare_barrier`` arg of each call
    and (optionally) raises once to simulate a mid-pipeline crash."""

    def __init__(self):
        self.prepare_barrier_calls = []
        self.raise_next = False

    def __call__(
        self,
        input_data,
        output_data,
        count,
        stream=None,
        barrier=True,
        prepare_barrier=True,
    ):
        self.prepare_barrier_calls.append(prepare_barrier)
        if self.raise_next:
            self.raise_next = False
            raise RuntimeError("injected mid-pipeline intra-gather failure")


class _StubInter:
    """Stub inter-node ring phase: data movement is irrelevant to this test.

    Provides ``slot_tensor`` so the ``gather_in_place`` return path (which writes
    the intra-gather node-block straight into the ring slot) can be exercised
    without a real symmetric ring buffer. Callable as the ring itself (noop)."""

    def slot_tensor(self, block_count, dtype, device):
        return torch.zeros(block_count, dtype=dtype, device=device)

    def __call__(self, *args, **kwargs):
        return True


def _noop_inter(*args, **kwargs):
    """Stub inter-node ring phase: data movement is irrelevant to this test."""
    return True


def _make_hier_stub(
    fuse_barrier: bool, leader_only: bool = False, gather_in_place: bool = False
):
    """Build a HierAllGather with the phase ops stubbed, bypassing __init__.

    __init__ allocates real C++/shmem handles (collective ShmemMalloc), which
    need the full distributed runtime + GPU. We only want to exercise the pure
    Python ``_prev_op_completed`` state machine in __call__, so we construct the
    object via ``object.__new__`` and set just the attributes that path reads.

    ``gather_in_place`` selects the in-place return site (line ~624 of
    hier_allgather.py) instead of the default staged site (~649); both share the
    crash-recovery guard but are distinct return paths.
    """
    h = object.__new__(HierAllGather)
    h.num_nodes = 2
    h.ranks_per_node = 2
    h.npes = 4
    h.leader_only = leader_only
    h.gather_in_place = gather_in_place
    h.out_in_place = False
    h.fuse_barrier = fuse_barrier
    h._node_block = None
    h._prev_op_completed = False
    h._deferbwd_event = None  # __init__ default; keeps the drain-guard a no-op
    # size-threshold dispatcher state (_call_impl reads these before the guard);
    # both dispatch levers OFF -> the default non-slice fuse-barrier path.
    h.slice_inter = False
    h.pipe_band = False
    h._last_use_slice = None
    h._intra = _RecordingIntra()
    h._inter = _StubInter()
    # leader-only path also touches these:
    h.local_rank = 0
    h._bcast = _noop_inter
    h._ring_scratch = None
    return h


def run_fuse_barrier_guard() -> int:
    """Unit-test the fuse-barrier entry-barrier crash-recovery guard.

    The committed ``_prev_op_completed`` guard decides whether __call__ may skip
    the intra-gather ENTRY ShmemBarrierAll. It must be skipped ONLY when the
    prior op ran to clean completion; the first op AND any op after a
    mid-pipeline crash must KEEP the barrier (a dirty out_ buffer would
    otherwise corrupt the gather). This was flagged in review (/53) as
    having no exception-path test. CPU-only: no GPU/SDMA/RDMA.

    ``prepare_barrier=True`` means the barrier is KEPT; ``False`` means SKIPPED.
    """
    failures = 0
    inp = torch.zeros(8, dtype=torch.float32)
    out = torch.zeros(32, dtype=torch.float32)

    def check(cond, msg):
        nonlocal failures
        if not cond:
            print(f"FAIL guard: {msg}")
            failures += 1

    # 1) fuse_barrier ON: first op keeps the barrier; steady-state ops skip it.
    h = _make_hier_stub(fuse_barrier=True)
    h._call_impl(inp, out, 4)
    check(
        h._intra.prepare_barrier_calls[-1] is True,
        "first op must KEEP entry barrier (no prior clean op)",
    )
    check(h._prev_op_completed is True, "clean op must set _prev_op_completed")
    h._call_impl(inp, out, 4)
    check(
        h._intra.prepare_barrier_calls[-1] is False,
        "2nd op after clean op must SKIP entry barrier",
    )
    h._call_impl(inp, out, 4)
    check(
        h._intra.prepare_barrier_calls[-1] is False,
        "steady-state op must SKIP entry barrier",
    )

    # 2) Mid-pipeline crash: next op must KEEP the barrier (out_ may be dirty).
    h = _make_hier_stub(fuse_barrier=True)
    h._call_impl(inp, out, 4)  # clean -> _prev_op_completed True, would skip next
    h._intra.raise_next = True
    try:
        h._call_impl(inp, out, 4)
        check(False, "injected failure should have propagated")
    except RuntimeError:
        pass
    check(h._prev_op_completed is False, "crash must leave _prev_op_completed False")
    h._call_impl(inp, out, 4)
    check(
        h._intra.prepare_barrier_calls[-1] is True,
        "op after mid-pipeline crash must KEEP entry barrier",
    )

    # 3) fuse_barrier OFF: barrier is always kept (never skipped).
    h = _make_hier_stub(fuse_barrier=False)
    for _ in range(3):
        h._call_impl(inp, out, 4)
    check(
        all(b is True for b in h._intra.prepare_barrier_calls),
        "fuse_barrier=0 must always KEEP entry barrier",
    )

    # 4) leader_only ON: guard never skips (skip requires not leader_only).
    h = _make_hier_stub(fuse_barrier=True, leader_only=True)
    for _ in range(3):
        h._call_impl(inp, out, 4)
    check(
        all(b is True for b in h._intra.prepare_barrier_calls),
        "leader_only must always KEEP entry barrier even with fuse_barrier=1",
    )

    # 5) gather_in_place ON: the in-place return site (distinct from the staged
    #    one in scenarios 1-2) must observe the SAME guard. First op keeps,
    #    steady-state skips, and a mid-pipeline crash makes the next op keep.
    h = _make_hier_stub(fuse_barrier=True, gather_in_place=True)
    h._call_impl(inp, out, 4)
    check(
        h._intra.prepare_barrier_calls[-1] is True,
        "gather_in_place first op must KEEP entry barrier",
    )
    check(
        h._prev_op_completed is True,
        "gather_in_place clean op must set _prev_op_completed",
    )
    h._call_impl(inp, out, 4)
    check(
        h._intra.prepare_barrier_calls[-1] is False,
        "gather_in_place 2nd op after clean op must SKIP entry barrier",
    )
    h._intra.raise_next = True
    try:
        h._call_impl(inp, out, 4)
        check(False, "injected failure should have propagated (gather_in_place)")
    except RuntimeError:
        pass
    check(
        h._prev_op_completed is False,
        "gather_in_place crash must leave _prev_op_completed False",
    )
    h._call_impl(inp, out, 4)
    check(
        h._intra.prepare_barrier_calls[-1] is True,
        "gather_in_place op after crash must KEEP entry barrier",
    )

    if failures:
        print(f"\n{failures} guard checks FAILED")
        return 1
    print(
        "PASSED fuse-barrier guard — entry barrier kept on first op + after "
        "mid-pipeline crash, skipped only after a clean op; covers both the "
        "staged and gather_in_place return sites (5 scenarios)."
    )
    return 0


def test_hier_allgather_cpu():
    assert run() == 0
    assert run_ring() == 0
    assert run_fuse_barrier_guard() == 0


if __name__ == "__main__":
    sys.exit(run() or run_ring() or run_fuse_barrier_guard())
