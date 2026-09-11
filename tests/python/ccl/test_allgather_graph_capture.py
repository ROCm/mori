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
"""AllGather SDMA under HIP graph capture and repeated replay.

The interesting case is the *second* replay onward. The completion flags are
monotonic AMO_SET and never reset, so a generation minted on the host and passed
to the kernel by value gets frozen into the graph node: from replay 2 the wait is
already satisfied on entry and returns without the peers' payload having landed.
Nothing crashes, the output is just silently stale.

So each replay writes a fresh value into the (fixed) input buffer and the output
is checked against that value. A stale read shows up as the previous replay's
value, which the failure report prints explicitly.

The three modes cover the three kernels that derive the generation: `sync` the
fused gather, `async` the standalone wait kernel, and `param_contiguous` the
split-aware gather. There is no param-contiguous async mode because that path's
PUT kernel never touches the flags -- it reuses the same wait kernel `async`
already covers.

Detecting it needs injected skew. With every rank leaving the same barrier and
doing the same amount of work, peers finish their puts within microseconds of
each other while the transfer itself takes hundreds, so a wait that returns
immediately still happens to find the data there. `--skew-cycles` holds half the
ranks back with a spin kernel enqueued ahead of the replay, so the early ranks
reach their copy-out long before the late ranks have posted anything. A correct
wait absorbs the skew; a frozen generation copies out stale data.
"""

import os
import numpy as np
import torch
import torch.distributed as dist
import mori.shmem as shmem
from mori.ccl import AllgatherSdma
from tests.python.utils import TorchDistContext, get_free_port


# Distinct per (pe, replay) and never zero, so a stale or untouched buffer is
# always distinguishable from a correct one.
def _expected(pe, replay):
    return (pe + 1) * 1000 + replay * 7919


def _test_graph_capture(rank, world_size, port, elems, replays, mode, skew_cycles):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")

        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        assert my_pe == rank and npes == world_size

        bytes_per_pe = elems * 4
        total_bytes = bytes_per_pe * npes

        if rank == 0:
            print(f"\n{'=' * 68}")
            print(f"AllGather SDMA graph capture test  (mode={mode})")
            print(f"  world size    : {world_size}")
            print(f"  elems per PE  : {elems:,}  ({bytes_per_pe / 2**20:.1f} MB)")
            print(f"  output total  : {total_bytes / 2**20:.1f} MB")
            print(f"  replays       : {replays}")
            print(f"  skew cycles   : {skew_cycles:,} (on odd ranks)")
            print(f"{'=' * 68}\n", flush=True)

        # Hold odd ranks back so the even ranks reach their copy-out while the
        # odd ranks have not posted their contribution yet.
        is_late = skew_cycles > 0 and (my_pe % 2 == 1)

        allgather = AllgatherSdma(
            my_pe,
            npes,
            input_buffer_size=bytes_per_pe,
            output_buffer_size=total_bytes,
            copy_output_to_user=True,
        )

        device = torch.device(f"cuda:{rank}")
        input_tensor = torch.zeros(elems, dtype=torch.uint32, device=device)
        output_tensor = torch.zeros(elems * npes, dtype=torch.uint32, device=device)

        # Param-contiguous scatters each split to `split_offset * npes +
        # pe * split_size`, so what has to be checked is one region per
        # (split, pe) instead of one contiguous chunk per pe.
        split_sizes = split_offsets = None
        if mode == "param_contiguous":
            head = elems // 4
            body = elems // 2
            sizes = [head, body, elems - head - body]
            if min(sizes) <= 0:
                raise ValueError(f"--elems {elems} is too small to split three ways")
            offsets = [0, head, head + body]
            split_sizes = torch.tensor(sizes, dtype=torch.uint64, device=device)
            split_offsets = torch.tensor(offsets, dtype=torch.uint64, device=device)
            regions = [
                (off * npes + pe * size, off * npes + pe * size + size, pe, idx)
                for idx, (size, off) in enumerate(zip(sizes, offsets))
                for pe in range(npes)
            ]
        else:
            regions = [(pe * elems, (pe + 1) * elems, pe, None) for pe in range(npes)]

        def fill_input(replay):
            input_tensor.fill_(_expected(my_pe, replay))

        def enqueue(stream, capturing):
            if mode == "sync":
                allgather(
                    input_tensor, output_tensor, elems, stream, capturing=capturing
                )
            elif mode == "param_contiguous":
                allgather.enqueue_param_contiguous(
                    input_tensor,
                    output_tensor,
                    elems,
                    split_sizes,
                    split_offsets,
                    stream,
                    capturing=capturing,
                )
            else:
                allgather.start_async(input_tensor, output_tensor, elems, stream)
                allgather.wait_async(stream, capturing=capturing)

        # Warm up eagerly first: the kernels are JIT compiled on first use and
        # the transit buffers grow on demand, neither of which is capturable.
        fill_input(0)
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                enqueue(warmup_stream, capturing=False)
        warmup_stream.synchronize()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print("warmup done, capturing graph ...", flush=True)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            enqueue(torch.cuda.current_stream(), capturing=True)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print("capture done, replaying ...\n", flush=True)

        failures = []
        for replay in range(1, replays + 1):
            # Every rank must have its new contribution in place before any rank
            # starts replaying, or a peer legitimately gathers the old value.
            fill_input(replay)
            output_tensor.fill_(0xDEADBEEF)
            torch.cuda.synchronize()
            dist.barrier()

            if is_late:
                # Enqueued on the stream the replay goes to, so it delays the
                # whole collective on this rank rather than just the host.
                torch.cuda._sleep(skew_cycles)
            graph.replay()
            torch.cuda.synchronize()

            got = output_tensor.cpu().numpy()
            bad = []
            for start, end, src_pe, split_idx in regions:
                chunk = got[start:end]
                want = _expected(src_pe, replay)
                if not np.all(chunk == want):
                    uniq = np.unique(chunk)
                    stale = _expected(src_pe, replay - 1)
                    hint = ""
                    if stale in uniq:
                        hint = f"  <-- contains PREVIOUS replay's value {stale}"
                    elif 0xDEADBEEF in uniq:
                        hint = "  <-- still poison, nothing was written"
                    where = f"PE {src_pe}"
                    if split_idx is not None:
                        where += f" split {split_idx}"
                    bad.append(
                        f"    replay {replay} chunk from {where}: "
                        f"want {want}, got {uniq[:6]}{hint}"
                    )
            if bad:
                failures.extend(bad)
                print(f"PE {rank}: replay {replay} MISMATCH", flush=True)
                for line in bad:
                    print(line, flush=True)
            elif rank == 0:
                print(f"  replay {replay}: ok", flush=True)

            dist.barrier()

        ok = torch.tensor([0 if failures else 1], dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.SUM)
        passed = ok.item()

        if rank == 0:
            print(f"\nPEs passed: {passed}/{npes}")
            print(f"=== {'PASSED' if passed == npes else 'FAILED'} (mode={mode}) ===\n")

        torch.cuda.synchronize()
        dist.barrier()
        del graph
        del allgather
        dist.barrier()
        shmem.shmem_finalize()

        if passed != npes:
            raise AssertionError(
                f"PE {rank}: graph replay verification failed "
                f"({len(failures)} bad chunks locally)"
            )


def test_allgather_graph_capture(
    elems=4 * 1024 * 1024,
    world_size=8,
    replays=10,
    mode="sync",
    skew_cycles=400_000_000,
):
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    port = get_free_port()
    torch.multiprocessing.spawn(
        _test_graph_capture,
        args=(world_size, port, elems, replays, mode, skew_cycles),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elems", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument(
        "--mode", choices=["sync", "async", "param_contiguous"], default="sync"
    )
    parser.add_argument(
        "--skew-cycles",
        type=int,
        default=400_000_000,
        help="spin cycles enqueued on odd ranks before each replay (0 disables)",
    )
    parser.add_argument("--enable-sdma", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()
    os.environ["MORI_ENABLE_SDMA"] = str(args.enable_sdma)

    test_allgather_graph_capture(
        args.elems, args.world_size, args.replays, args.mode, args.skew_cycles
    )
