# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""Timing for kernels small enough that the harness is the measurement.

Two corrections over the obvious `capture one call, replay, take the median`,
both of which matter only at small M and both of which silently inflated an
earlier round of thresholds on this branch.

**A single-call graph replay has a floor.** On this box an empty-ish kernel
replays in 13.40us; amortised over 200 calls in one graph the same kernel is
1.51us. So every number below ~15us produced by single-call capture is mostly
the floor. sglang's mxfp8 GEMV at `wq_b` M=1 measured 13.48us that way and is
2.45us -- it was entirely inside the floor, and so was the margin it was being
compared on. `amortized=True` (the default) captures `reps` calls per graph.

**Repeating a call leaves the weight in LLC.** MI355X has 256MB of it and
`wq_b`'s weight is 10.5MB, so a hot loop reads from cache at 4308 GB/s where
one cold pass gets 2561 GB/s -- a 1.7x overstatement of a memory-bound kernel.
A decode step touches each layer's weight once, so cold is the number that
predicts the server and hot is the ceiling. `cold()` rotates enough copies of
the weight to exceed the cache and reports both.

Both are properties of the measurement, not of the kernel: a compute-bound GEMM
at M=16384 is unaffected by either, which is why they went unnoticed.
"""

from __future__ import annotations

import statistics
import subprocess

import torch

#: Past this the LLC cannot hold the working set. MI355X has 256MB; the margin
#: covers the activations and output sharing it.
LLC_BYTES = 256 << 20
COLD_WORKING_SET = 384 << 20


def _graph(fn, reps: int) -> torch.cuda.CUDAGraph:
    """Capture `reps` calls into one graph, warming on a side stream first."""
    fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    return g


def median_us(fn, reps: int = 200, iters: int = 21) -> float:
    """Median us per call, amortised over `reps` calls inside one graph.

    `reps=1` reproduces the old single-call number, which is useful only for
    showing what the floor was doing to it.
    """
    g = _graph(fn, reps)
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0 / reps)
    return statistics.median(ts)


def replay_floor_us(iters: int = 21) -> float:
    """What one single-call graph replay costs when the kernel does nothing.

    Report this next to any single-call number so the reader can see how much
    of it is the harness.
    """
    x = torch.zeros(1, device="cuda")
    return median_us(lambda: x.add_(1.0), reps=1, iters=iters)


def n_copies_for(weight_bytes: int, target: int = COLD_WORKING_SET) -> int:
    """How many rotating weight copies push the working set past the LLC."""
    return max(1, -(-target // max(weight_bytes, 1)))


class Rotating:
    """`n` copies of a tensor, handed out round-robin.

    Enough of them and consecutive calls cannot hit in LLC, which is what a
    real forward pass looks like: each layer's weight is read once per step.
    """

    def __init__(self, t: torch.Tensor, n: int):
        self.copies = [t] + [t.clone() for _ in range(n - 1)]
        self.i = 0

    def next(self) -> torch.Tensor:
        t = self.copies[self.i]
        self.i = (self.i + 1) % len(self.copies)
        return t

    @property
    def bytes(self) -> int:
        return self.copies[0].numel() * self.copies[0].element_size() * len(self.copies)


def cold_hot_us(make_fn, weights, reps: int = 32, iters: int = 21) -> dict:
    """Both numbers for one kernel: cold (rotating weights) and hot (one copy).

    `weights` is the tensors whose residency is in question, largest first;
    `make_fn(picked)` returns the closure to time, given one tensor per entry.
    Cold rotates them together so a rep never revisits the previous rep's copy.

    **The cold capture must be at least as long as the ring.** The ring advances
    at capture time, not at replay time, so a graph of `reps` calls bakes in
    `reps` pointers and every replay revisits those same ones -- the working set
    is `min(reps, n)` copies, not `n`. At `reps=32` that silently left seven of
    this branch's twelve shapes under the 256MB LLC and therefore not cold at
    all, and the ones that landed *on* it read worst of any: a working set at
    exactly cache capacity thrashes, where one comfortably over it just streams.
    That is why `wo_a` (8MB weight, 256MB at 32 reps) measured +10% against
    SGLang cold and -2% hot.

    **And the ring is sized so that every tensor clears the cache on its own,
    not so their sum does.** A caller may hand over several weights of which the
    kernel reads only one -- SGLang's native linear takes both an fp8 weight and
    a dequantised bf16 one and reads whichever its route picked. Sizing on the
    sum then buys `384MB / (fp8 + bf16)` copies, and the route that reads only
    the 10MB fp8 weight sees 130MB of it: back inside the LLC, hot again. Sizing
    on the largest member costs more VRAM and is correct either way, since a
    kernel that does read all of them gets a working set larger still.
    """
    n = max(n_copies_for(w.numel() * w.element_size()) for w in weights)
    rings = [Rotating(w, n) for w in weights]

    def cold_call():
        return make_fn([r.next() for r in rings])

    hot = median_us(lambda: make_fn(list(weights)), reps=reps, iters=iters)
    cold_reps = max(reps, n)
    cold = median_us(cold_call, reps=cold_reps, iters=iters)
    return {
        "hot_us": hot,
        "cold_us": cold,
        "copies": n,
        "cold_reps": cold_reps,
        "working_set_mb": sum(r.bytes for r in rings) / 2**20,
    }


def vram_used(device: int = 0) -> int | None:
    """Bytes in use per rocm-smi, for the before/after check around a run."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if line.startswith(f"card{device},"):
            parts = line.strip().split(",")
            if len(parts) >= 3 and parts[2].isdigit():
                return int(parts[2])
    return None
