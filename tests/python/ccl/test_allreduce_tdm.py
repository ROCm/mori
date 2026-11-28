#!/usr/bin/env python3
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
"""AllreduceSdma on gfx125x without SDMA, where it runs on the TDM All2all kernels.

Every call is checked against the sum the ranks' inputs give, with data that changes
from call to call: into a plain output, into an output every rank registered, in place,
and for int32, float32 and uint32. Counts cover the one-shot path, the two-shot path,
both sides of the threshold between them, and counts that do not split into whole
chunks.

  python tests/python/ccl/test_allreduce_tdm.py --bench

also times every power of two from 16 B to 1 GiB a rank: one graph of --iters calls a
size and mode, the slowest rank, the median over the timed replays.
"""
import os
import statistics

import pytest
import torch
import torch.distributed as dist

import mori.shmem as shmem
from tests.python.utils import get_free_port

# Elements of the buffer a rank reduces.
_CHECK_COUNTS = (
    4,
    5,
    64,
    1000,
    4096,
    (1 << 18) - 4,
    1 << 18,  # 1 MiB of int32: the largest one-shot call
    (1 << 18) + 4,  # the smallest two-shot call
    (1 << 18) + 7,  # not a multiple of the ranks
    3 * (1 << 20) + 12,
    1 << 24,  # 64 MiB: the peers write the output directly when it is registered
)
_CHECK_CALLS = 3
_BENCH_BYTES_PER_GPU = tuple(1 << k for k in range(4, 31))
# Sizes above this are checked once instead of _CHECK_CALLS times.
_CHECK_ONCE_BYTES = 32 << 20


def _data(pe, call, count, dtype, device):
    j = torch.arange(count, device=device, dtype=torch.int64)
    return ((pe + 1) * (call * 13 + 1) + j % 251).to(dtype)


def _want(npes, call, count, dtype, device):
    j = torch.arange(count, device=device, dtype=torch.int64)
    total = npes * (npes + 1) // 2 * (call * 13 + 1) + npes * (j % 251)
    return total.to(dtype)


def _fmt_bytes(n):
    if n >= (1 << 20):
        return f"{n >> 20} MiB"
    return f"{n >> 10} KiB" if n >= (1 << 10) else f"{n} B"


def _worker(rank, world_size, port, bench, rounds, iters, warmup):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    # gloo only: the test needs a CPU barrier and the shmem bootstrap, not a GPU collective.
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    shmem.shmem_torch_process_group_init("default")

    from mori.ccl import AllreduceSdma

    my_pe, npes = shmem.shmem_mype(), shmem.shmem_npes()
    assert (my_pe, npes) == (rank, world_size)
    dev = torch.device("cuda", rank)

    bench_counts = [b // 4 for b in _BENCH_BYTES_PER_GPU] if bench else []
    max_count = max(list(_CHECK_COUNTS) + bench_counts)
    nbytes = -(-max_count * 4 // (npes * 4096)) * npes * 4096
    handles = {
        "into output": AllreduceSdma(
            my_pe, npes, input_buffer_size=nbytes, dtype=torch.int32
        ),
        # Its outputs are registered on every rank, so large calls land in them directly.
        "registered": AllreduceSdma(
            my_pe, npes, input_buffer_size=nbytes, dtype=torch.int32
        ),
        "float32": AllreduceSdma(
            my_pe, npes, input_buffer_size=nbytes, dtype=torch.float32
        ),
        "uint32": AllreduceSdma(
            my_pe, npes, input_buffer_size=nbytes, dtype=torch.uint32
        ),
    }
    dtypes = {
        "into output": torch.int32,
        "registered": torch.int32,
        "float32": torch.float32,
        "uint32": torch.int32,
    }
    backends = {m: h._backend for m, h in handles.items()}
    assert set(backends.values()) == {"tdm"}, backends
    stream = torch.cuda.Stream(device=dev)
    calls = {m: 0 for m in handles}
    calls["in place"] = 0
    failed = 0

    def check(mode, count, in_place=False):
        nonlocal failed
        h = handles["into output" if in_place else mode]
        name = "in place" if in_place else mode
        calls[name] += 1
        call = calls[name]
        dtype = dtypes["into output" if in_place else mode]
        inp = _data(my_pe, call, count, dtype, dev)
        if mode == "uint32" and not in_place:
            inp = inp.view(torch.uint32)
        out = inp if in_place else torch.full_like(inp, 0)
        if mode == "registered":
            h.register_output_buffer(out)
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            if in_place:
                h.allreduce_inplace(out, count, stream)
            else:
                h(inp, out, count, stream)
        torch.cuda.synchronize()
        if mode == "registered":
            h.deregister_output_buffer(out)
        got = out.view(torch.int32) if out.dtype == torch.uint32 else out
        want = _want(npes, call, count, dtype, dev)
        wrong = got != want
        n = int(wrong.sum().item())
        if n:
            k = int(wrong.nonzero()[0].item())
            print(
                f"[check] PE {my_pe} {name} count {count} call {call}: {n} of {count} "
                f"wrong, first at {k}: got {got[k].item()} want {want[k].item()}",
                flush=True,
            )
        failed += int(n != 0)

    for mode in handles:
        for count in _CHECK_COUNTS:
            once = count * 4 > _CHECK_ONCE_BYTES
            for _ in range(1 if once else _CHECK_CALLS):
                check(mode, count)
    for count in _CHECK_COUNTS:
        check("into output", count, in_place=True)

    if bench:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        modes = ("into output", "registered")
        times = {m: {b: [] for b in _BENCH_BYTES_PER_GPU} for m in modes}
        for b, count in zip(_BENCH_BYTES_PER_GPU, bench_counts):
            inp = torch.zeros(count, dtype=torch.int32, device=dev)
            out = torch.empty_like(inp)
            for mode in modes:
                if mode == "registered":
                    handles[mode].register_output_buffer(out)
                graph = torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(iters):
                        handles[mode](inp, out, count, stream)
                torch.cuda.synchronize()
                for r in range(warmup + rounds):
                    dist.barrier()
                    with torch.cuda.stream(stream):
                        start.record(stream)
                        graph.replay()
                        end.record(stream)
                    end.synchronize()
                    if r < warmup:
                        continue
                    t = torch.tensor(
                        [start.elapsed_time(end) * 1000.0 / iters], dtype=torch.float64
                    )
                    dist.all_reduce(t, op=dist.ReduceOp.MAX)
                    times[mode][b].append(t.item())
                del graph
                if mode == "registered":
                    handles[mode].deregister_output_buffer(out)
        # The handles must still agree after thousands of back-to-back calls.
        for mode in handles:
            check(mode, _CHECK_COUNTS[-1])
        if rank == 0:
            print(
                f"[bench] npes {npes}, one graph of {iters} calls per size and mode, "
                f"{warmup} warmup replays, {rounds} timed; us per call = slowest rank, "
                "median over the timed replays",
                flush=True,
            )
            for mode in modes:
                for b in _BENCH_BYTES_PER_GPU:
                    us = statistics.median(times[mode][b])
                    alg = b / us / 1000.0
                    print(
                        f"[ar-tdm] {mode:<21} {_fmt_bytes(b):>7} per GPU  {us:8.2f} us "
                        f"({'/'.join(f'{x:.2f}' for x in times[mode][b])})  "
                        f"algBw {alg:7.1f} GB/s  "
                        f"busBw {alg * 2 * (npes - 1) / npes:7.1f} GB/s",
                        flush=True,
                    )

    f = torch.tensor([failed], dtype=torch.int64)
    dist.all_reduce(f, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            f"[check] {'PASS' if f.item() == 0 else 'FAIL'}: "
            + ", ".join(f"{calls[m]} calls {m}" for m in calls)
            + f" on {npes} ranks, {f.item()} rank-calls wrong",
            flush=True,
        )

    torch.cuda.synchronize()
    dist.barrier()
    del handles
    dist.barrier()
    shmem.shmem_finalize()
    dist.destroy_process_group()
    if failed:
        raise AssertionError(
            f"PE {rank}: TDM allreduce output mismatch in {failed} call(s)"
        )


def _gfx125x():
    try:
        from mori.jit.config import detect_gpu_arch

        return detect_gpu_arch().startswith("gfx125")
    except Exception:
        return False


def _run(world_size, bench=False, rounds=5, iters=100, warmup=2):
    # Nothing on this path may use SDMA: no SDMA queues in mori, and no copy
    # engine behind the ROCm runtime's own copies either.
    os.environ["MORI_ENABLE_SDMA"] = "0"
    os.environ["HSA_ENABLE_SDMA"] = "0"
    os.environ.pop("MORI_CCL_ALL2ALL_BACKEND", None)
    os.environ.pop("MORI_CCL_ALLREDUCE_BACKEND", None)
    # Four handles; each holds an input buffer, two halves for the reduce-scatter and two
    # for the all-gather of the largest size, plus words and LL128 areas.
    largest = max(_BENCH_BYTES_PER_GPU) if bench else 2 * _CHECK_ONCE_BYTES
    os.environ.setdefault(
        "MORI_SHMEM_HEAP_SIZE", str(max(4 << 30, 26 * largest + (1 << 30)))
    )
    torch.multiprocessing.spawn(
        _worker,
        args=(world_size, get_free_port(), bench, rounds, iters, warmup),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(not _gfx125x(), reason="the TDM engine exists only on gfx125x")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 GPUs")
def test_allreduce_tdm():
    _run(min(torch.cuda.device_count(), 8))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--rounds", type=int, default=5, help="timed graph replays")
    parser.add_argument("--iters", type=int, default=100, help="calls per graph")
    parser.add_argument("--warmup", type=int, default=2, help="untimed graph replays")
    a = parser.parse_args()
    _run(a.world_size, a.bench, a.rounds, a.iters, a.warmup)
