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
"""All2allSdma on the TDM transport (MORI_CCL_ALL2ALL_BACKEND=tdm, gfx125x only).

Every call sends data that depends on the call number, so a chunk left in a
transit buffer by an earlier call cannot pass for the current one. The sizes
cover whole TDM tiles, a TDM tail of a few rows, tails short enough to be left
to the lanes, and chunk sizes whose offsets are not 128 B aligned, which the
kernel copies with the lanes alone.

--bench also times the kernel alone (no host synchronization, no copy into the
user tensor) at power-of-two sizes per GPU. Back-to-back launches are captured
into one HIP graph and replayed, so the host's launch cost does not cap the
small sizes; the kernel keeps its call count in device memory, which is what
makes it replayable. Bandwidth is reported the way rccl-tests does for
alltoall: algBw = bytes per GPU / time, busBw = algBw * (npes - 1) / npes.
"""

import os
import statistics

import pytest
import torch
import torch.distributed as dist

import mori.shmem as shmem
from tests.python.utils import get_free_port

# Elements per destination. One TDM tile is 4096 elements (16 KiB), one row 256.
_CHECK_COUNTS = (
    1,
    33,
    256,
    512,
    4096,
    4096 + 300,
    3 * 4096 + 5 * 256,
    3 * 4096 + 5 * 256 + 7,
    1 << 19,
)
_CHECK_CALLS = 3
_BENCH_BYTES_PER_GPU = (64 << 10,) + tuple(m << 20 for m in (1, 2, 4, 8, 16, 32))


def _chunk(src_pe, dest_pe, npes, call, count, device):
    j = torch.arange(count, device=device, dtype=torch.int64)
    base = (src_pe * npes + dest_pe) * 1000003 + call * 7919
    return ((base + j) % (1 << 31)).to(torch.int32)


def _fmt_bytes(n):
    return f"{n >> 20} MiB" if n >= (1 << 20) else f"{n >> 10} KiB"


def _worker(rank, world_size, port, bench, rounds, iters, warmup):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    # gloo only: the test needs a CPU barrier and the shmem bootstrap, not a GPU collective.
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    shmem.shmem_torch_process_group_init("default")

    from mori.ccl import All2allSdma

    my_pe, npes = shmem.shmem_mype(), shmem.shmem_npes()
    assert (my_pe, npes) == (rank, world_size)
    dev = torch.device("cuda", rank)

    bench_counts = [b // 4 // npes for b in _BENCH_BYTES_PER_GPU] if bench else []
    max_count = max(list(_CHECK_COUNTS) + bench_counts)
    nbytes = max_count * npes * 4
    a2a = All2allSdma(
        my_pe,
        npes,
        input_buffer_size=nbytes,
        output_buffer_size=nbytes,
        copy_output_to_user=True,
    )
    assert a2a._backend == "tdm", a2a._backend
    stream = torch.cuda.Stream(device=dev)
    call = 0
    failed = 0

    def check(count, use_async=False):
        nonlocal call, failed
        call += 1
        inp = torch.cat([_chunk(my_pe, d, npes, call, count, dev) for d in range(npes)])
        out = torch.full((npes * count,), -1, dtype=torch.int32, device=dev)
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            if use_async:
                a2a.start_async(inp, out, count, stream)
                a2a.wait_async(stream)
            else:
                a2a(inp, out, count, stream)
        torch.cuda.synchronize()
        bad = 0
        for s in range(npes):
            want = _chunk(s, my_pe, npes, call, count, dev)
            got = out[s * count : (s + 1) * count]
            wrong = got != want
            n = int(wrong.sum().item())
            if n:
                k = int(wrong.nonzero()[0].item())
                print(
                    f"[check] PE {my_pe} count {count} call {call} from PE {s}: {n} of "
                    f"{count} wrong, first at {k}: got {int(got[k])} want {int(want[k])}",
                    flush=True,
                )
            bad += n
        failed += int(bad != 0)

    for count in list(_CHECK_COUNTS) + bench_counts:
        for _ in range(_CHECK_CALLS):
            check(count)
    check(8 * 4096, use_async=True)
    checks = call

    if bench:
        s = stream.cuda_stream

        def launch(inp, out, count):
            args = a2a._handle.prepare_sync(inp.data_ptr(), out.data_ptr(), count, s)
            a2a._launch_tdm(args, s)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = {b: [] for b in _BENCH_BYTES_PER_GPU}
        for b, count in zip(_BENCH_BYTES_PER_GPU, bench_counts):
            inp = torch.ones(npes * count, dtype=torch.int32, device=dev)
            out = torch.empty_like(inp)
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(iters):
                    launch(inp, out, count)
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
                t = torch.tensor([start.elapsed_time(end) * 1000.0 / iters], dtype=torch.float64)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                times[b].append(t.item())
            del graph
        # The flags must still agree after thousands of back-to-back calls.
        check(bench_counts[-1])
        checks = call
        if rank == 0:
            print(
                f"[bench] npes {npes}, one graph of {iters} calls per size, {warmup} warmup "
                f"replays, {rounds} timed; us per call = slowest rank, median over the timed "
                "replays",
                flush=True,
            )
            for b in _BENCH_BYTES_PER_GPU:
                us = statistics.median(times[b])
                alg = b / us / 1000.0
                print(
                    f"[a2a-tdm] {_fmt_bytes(b):>7} per GPU  {us:8.2f} us "
                    f"({'/'.join(f'{x:.2f}' for x in times[b])})  "
                    f"algBw {alg:7.1f} GB/s  busBw {alg * (npes - 1) / npes:7.1f} GB/s",
                    flush=True,
                )

    f = torch.tensor([failed], dtype=torch.int64)
    dist.all_reduce(f, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            f"[check] {'PASS' if f.item() == 0 else 'FAIL'}: {checks} calls on {npes} ranks, "
            f"{f.item()} rank-calls wrong",
            flush=True,
        )

    torch.cuda.synchronize()
    dist.barrier()
    del a2a
    dist.barrier()
    shmem.shmem_finalize()
    dist.destroy_process_group()
    if failed:
        raise AssertionError(f"PE {rank}: TDM all2all output mismatch in {failed} call(s)")


def _gfx125x():
    try:
        from mori.jit.config import detect_gpu_arch

        return detect_gpu_arch().startswith("gfx125")
    except Exception:
        return False


def _run(world_size, bench=False, rounds=5, iters=100, warmup=2):
    # The TDM transport never uses an SDMA queue, so the copy engines stay off.
    os.environ["MORI_ENABLE_SDMA"] = "0"
    os.environ["MORI_CCL_ALL2ALL_BACKEND"] = "tdm"
    torch.multiprocessing.spawn(
        _worker,
        args=(world_size, get_free_port(), bench, rounds, iters, warmup),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(not _gfx125x(), reason="the TDM engine exists only on gfx125x")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 GPUs")
def test_all2all_tdm():
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
