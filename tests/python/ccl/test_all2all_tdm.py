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
"""All2allSdma on gfx125x, where it runs on the TDM transport only.

The test runs with SDMA off in mori (MORI_ENABLE_SDMA=0) and in the ROCm
runtime (HSA_ENABLE_SDMA=0), and checks that asking for the SDMA transport is
refused. Both ways of delivering the result are checked: into the caller's
output (copy_output_to_user=True), into an output every rank registered
(register_output_buffer, or by the handle itself with
MORI_CCL_ALL2ALL_AUTO_REGISTER=1), and left in the output transit buffer.

Every call sends data that depends on the call number, so a chunk left in a
transit buffer by an earlier call cannot pass for the current one. The sizes
cover whole TDM tiles, a TDM tail of a few rows, tails short enough to be left
to the lanes, and chunk sizes whose offsets are not 128 B aligned, which the
kernel copies with the lanes alone.

Calls of up to _TDM_A2A_LL128_MAX_BYTES a rank move as LL128 lines and larger
ones as TDM tiles. A third handle, built with MORI_CCL_ALL2ALL_LL128_MAX_BYTES=0,
sends every size through TDM, so both paths see the small sizes. The sizes also
include the largest LL128 call and one element more, and consecutive calls of a
handle switch between the two paths.

--bench times both ways at power-of-two sizes per GPU. Back-to-back calls are
captured into one HIP graph and replayed, so the host's launch cost does not cap
the small sizes; the kernel keeps its call count in device memory, which is what
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
# Every power of two from the smallest call 4 ranks can make (one element a rank) to 1 GiB.
_BENCH_BYTES_PER_GPU = tuple(1 << k for k in range(4, 31))
# Sizes above this are checked once instead of _CHECK_CALLS times.
_CHECK_ONCE_BYTES = 32 << 20


def _chunk(src_pe, dest_pe, npes, call, count, device):
    j = torch.arange(count, device=device, dtype=torch.int64)
    base = (src_pe * npes + dest_pe) * 1000003 + call * 7919
    return ((base + j) % (1 << 31)).to(torch.int32)


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

    from mori.ccl import All2allSdma, AllgatherSdma
    from mori.ccl.collective import (
        _TDM_A2A_COPY_AFTER_PUSH_MIN_BYTES,
        _TDM_A2A_LL128_MAX_BYTES,
    )

    # A2A_TEST_OP=allgather runs the same checks and bench on AllgatherSdma, which on gfx125x
    # is this kernel with every rank sending one chunk to all the others.
    gather = os.environ.get("A2A_TEST_OP", "alltoall") == "allgather"
    Handle = AllgatherSdma if gather else All2allSdma
    my_pe, npes = shmem.shmem_mype(), shmem.shmem_npes()
    assert (my_pe, npes) == (rank, world_size)
    dev = torch.device("cuda", rank)

    # Refused before the constructor allocates anything, so no rank is left in a collective.
    os.environ["MORI_CCL_ALL2ALL_BACKEND"] = "sdma"
    try:
        All2allSdma(my_pe, npes, transit_buffer_size=1 << 20)
        refused = False
    except RuntimeError:
        refused = True
    del os.environ["MORI_CCL_ALL2ALL_BACKEND"]
    assert refused, "MORI_CCL_ALL2ALL_BACKEND=sdma was accepted on gfx125x"

    # The largest call that moves as LL128 lines, and the smallest that does not.
    ll_edge = _TDM_A2A_LL128_MAX_BYTES // (4 * npes)
    # Into output, the smallest call that pushes everything before it copies, the largest that
    # copies while it pushes, and one of the former with a tail of rows and of lanes.
    ap_edge = -(-_TDM_A2A_COPY_AFTER_PUSH_MIN_BYTES // (4 * npes))
    check_counts = list(_CHECK_COUNTS) + [
        ll_edge,
        ll_edge + 1,
        ap_edge - 1,
        ap_edge,
        ap_edge + 5 * 256 + 7,
    ]
    bench_counts = [b // 4 // npes for b in _BENCH_BYTES_PER_GPU] if bench else []
    max_count = max(check_counts + bench_counts)
    # A whole number of pages: the handle rounds a half down to the 128 B TDM needs.
    nbytes = -(-max_count * npes * 4 // 4096) * 4096
    handles = {
        # Calls alternate between the halves of the output transit buffer.
        "into output": Handle(
            my_pe,
            npes,
            input_buffer_size=nbytes,
            output_buffer_size=2 * nbytes,
            copy_output_to_user=True,
        ),
        "in transit": Handle(
            my_pe,
            npes,
            input_buffer_size=nbytes,
            output_buffer_size=nbytes,
            copy_output_to_user=False,
        ),
    }
    ll_env = os.environ.pop("MORI_CCL_ALL2ALL_LL128_MAX_BYTES", None)
    os.environ["MORI_CCL_ALL2ALL_LL128_MAX_BYTES"] = "0"
    handles["TDM only"] = Handle(
        my_pe,
        npes,
        input_buffer_size=nbytes,
        output_buffer_size=2 * nbytes,
        copy_output_to_user=True,
    )
    del os.environ["MORI_CCL_ALL2ALL_LL128_MAX_BYTES"]
    if ll_env is not None:
        os.environ["MORI_CCL_ALL2ALL_LL128_MAX_BYTES"] = ll_env
    # Its outputs are registered on every rank, so large calls land in them directly.
    handles["registered"] = Handle(
        my_pe,
        npes,
        input_buffer_size=nbytes,
        output_buffer_size=2 * nbytes,
        copy_output_to_user=True,
    )
    # Registered as above, but rank 1 behaves as if its outputs could not be exported (it then
    # receives through the transit buffer), and in the second handle the last rank launches as
    # if its output were not registered (it then sends and receives through the transit buffer).
    for mode, env, ranks in (
        ("registered, 1 transit", "MORI_CCL_ALL2ALL_TEST_NO_DIRECT", "1"),
        ("registered, 1 without", "MORI_CCL_ALL2ALL_TEST_NO_REG", str(npes - 1)),
    ):
        os.environ[env] = ranks
        handles[mode] = Handle(
            my_pe,
            npes,
            input_buffer_size=nbytes,
            output_buffer_size=2 * nbytes,
            copy_output_to_user=True,
        )
        del os.environ[env]
    # Registers by itself every output a large call is given.
    os.environ["MORI_CCL_ALL2ALL_AUTO_REGISTER"] = "1"
    handles["auto"] = Handle(
        my_pe,
        npes,
        input_buffer_size=nbytes,
        output_buffer_size=2 * nbytes,
        copy_output_to_user=True,
    )
    del os.environ["MORI_CCL_ALL2ALL_AUTO_REGISTER"]
    backends = {m: h._backend for m, h in handles.items()}
    assert set(backends.values()) == {"tdm"}, backends
    stream = torch.cuda.Stream(device=dev)
    calls = {m: 0 for m in handles}
    failed = 0

    def check(mode, count, use_async=False):
        nonlocal failed
        h = handles[mode]
        calls[mode] += 1
        call = calls[mode]
        if gather:
            inp = _chunk(my_pe, 0, npes, call, count, dev)
        else:
            inp = torch.cat(
                [_chunk(my_pe, d, npes, call, count, dev) for d in range(npes)]
            )
        out = torch.full((npes * count,), -1, dtype=torch.int32, device=dev)
        if mode.startswith("registered"):
            h.register_output_buffer(out)
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            if use_async:
                h.start_async(inp, out, count, stream)
                h.wait_async(stream)
            else:
                h(inp, out, count, stream)
        torch.cuda.synchronize()
        if mode.startswith("registered"):
            h.deregister_output_buffer(out)
        if mode == "in transit":
            out = h.get_output_transit_buffer(dtype=torch.int32)[: npes * count]
        bad = 0
        for s in range(npes):
            want = _chunk(s, 0 if gather else my_pe, npes, call, count, dev)
            got = out[s * count : (s + 1) * count]
            wrong = got != want
            n = int(wrong.sum().item())
            if n:
                k = int(wrong.nonzero()[0].item())
                print(
                    f"[check] PE {my_pe} {mode} count {count} call {call} from PE {s}: {n} of "
                    f"{count} wrong, first at {k}: got {int(got[k])} want {int(want[k])}",
                    flush=True,
                )
            bad += n
        failed += int(bad != 0)

    for mode in handles:
        for count in check_counts + bench_counts:
            once = count * npes * 4 > _CHECK_ONCE_BYTES
            for _ in range(1 if once else _CHECK_CALLS):
                check(mode, count)
        check(mode, 8 * 4096, use_async=True)

    if bench:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = {m: {b: [] for b in _BENCH_BYTES_PER_GPU} for m in handles}
        for b, count in zip(_BENCH_BYTES_PER_GPU, bench_counts):
            inp = torch.ones(npes * count, dtype=torch.int32, device=dev)
            out = torch.empty_like(inp)
            # No loop variable holds a handle: every handle must be gone by shmem_finalize.
            for mode in handles:
                # One handle at a time holds the output: two registrations of the
                # same memory would undo each other.
                if mode.startswith("registered"):
                    handles[mode].register_output_buffer(out)
                if mode == "auto":
                    # Registers on a call outside a capture; the captured calls
                    # then land directly.
                    handles[mode](inp, out, count, stream)
                    torch.cuda.synchronize()
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
                if mode.startswith("registered"):
                    handles[mode].deregister_output_buffer(out)
        # The flags must still agree after thousands of back-to-back calls.
        for mode in handles:
            check(mode, bench_counts[-1])
        if rank == 0:
            print(
                f"[bench] npes {npes}, one graph of {iters} calls per size and mode, {warmup} "
                f"warmup replays, {rounds} timed; us per call = slowest rank, median over the "
                "timed replays",
                flush=True,
            )
            for mode in handles:
                for b in _BENCH_BYTES_PER_GPU:
                    us = statistics.median(times[mode][b])
                    alg = b / us / 1000.0
                    print(
                        f"[a2a-tdm] {mode:<21} {_fmt_bytes(b):>7} per GPU  {us:8.2f} us "
                        f"({'/'.join(f'{x:.2f}' for x in times[mode][b])})  "
                        f"algBw {alg:7.1f} GB/s  busBw {alg * (npes - 1) / npes:7.1f} GB/s",
                        flush=True,
                    )

    f = torch.tensor([failed], dtype=torch.int64)
    dist.all_reduce(f, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            f"[check] {'PASS' if f.item() == 0 else 'FAIL'}: "
            + ", ".join(f"{calls[m]} calls {m}" for m in handles)
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
            f"PE {rank}: TDM all2all output mismatch in {failed} call(s)"
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
    # Seven handles, each with an input buffer and up to two halves of output transit buffer of
    # the largest size, plus their words and LL128 areas.
    largest = max(_BENCH_BYTES_PER_GPU) if bench else 2 * _CHECK_ONCE_BYTES + (1 << 20)
    os.environ.setdefault(
        "MORI_SHMEM_HEAP_SIZE", str(max(4 << 30, 21 * largest + (1 << 30)))
    )
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
