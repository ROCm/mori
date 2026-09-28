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
"""Shared checkout selection and graph timing for GEMM + AG benchmarks."""

from pathlib import Path
import statistics
import sys

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python"))


def positive_int(value):
    import argparse

    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def measure(fn, warmup, iters, rounds=1, *, graph=True):
    """Time complete operations; return duration, rank/round data, and replay.

    Capture and allocation are excluded. The result is the median of the
    per-round maxima across ranks, each drawn from a rank's sample median.
    Keep the returned replay callable for changed-input validation.
    """
    if warmup < 0 or iters <= 0 or rounds <= 0:
        raise ValueError("warmup must be nonnegative; iters/rounds must be positive")
    distributed = dist.is_initialized()

    def barrier():
        if distributed:
            dist.barrier()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    barrier()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            fn()
        replay = captured.replay
    else:
        replay = fn
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    per_round = []
    for _ in range(rounds):
        barrier()
        for _ in range(warmup):
            replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(iters):
            start.record()
            replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
        per_round.append(statistics.median(samples))
    per_rank = [per_round]
    if distributed:
        per_rank = [None] * dist.get_world_size()
        dist.all_gather_object(per_rank, per_round)
    maxima = [max(r[i] for r in per_rank) for i in range(rounds)]
    return (
        statistics.median(maxima),
        dict(per_rank_us=per_rank, round_max_us=maxima),
        replay,
    )
