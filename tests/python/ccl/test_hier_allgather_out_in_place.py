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
"""
Bit-exact + A/B bench for the M4 "out-in-place" copy-OUT elimination
(``HierAllGather(out_in_place=True)``).

The default hierarchical path copies the gathered ring buffer to the user
output (the finish_sync copy-OUT, ~2.7ms @512MiB per  phase
attribution -- the single biggest remaining staging cost). out-in-place
leaves the result in the ring buffer and the caller reads it via
``handle.result_tensor(...)``, skipping that copy. This test asserts the
out-in-place result is bit-exact vs ``torch.distributed.all_gather_into_tensor``
(zero tolerance, ``torch.equal``) AND A/B-benches it against the default
staged path so the win (or wash) is measured with >=3 reps + RCCL baseline.

Single node (simulates N>=2 by splitting local GPUs into sub-groups)::

    python3 tests/python/ccl/test_hier_allgather_out_in_place.py \
        --world-size 4 --ranks-per-node 2 --bench
"""

import os
import traceback

import torch
import torch.distributed as dist

import mori.shmem as shmem
from mori.ccl import HierAllGather

from tests.python.utils import TorchDistContext, get_free_port

_DEFAULT_DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.int32]


def _make_input(dtype, numel, rank, device):
    base = (rank + 1) * 17
    ramp = torch.arange(numel, dtype=torch.int32) % 64
    return (ramp + base).to(dtype=dtype).contiguous().to(device=device)


def _check(out, ref, dtype, numel, tag):
    if not torch.equal(out, ref):
        diff = (out != ref).nonzero(as_tuple=False).flatten()[:8].tolist()
        raise AssertionError(
            f"{tag} mismatch dtype={dtype} numel={numel}: positions={diff} "
            f"got={out[diff].tolist()} ref={ref[diff].tolist()}"
        )


def _bench(fn, reps=5, warmup=2):
    ts = []
    for i in range(warmup + reps):
        torch.cuda.synchronize()
        dist.barrier()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize()
        if i >= warmup:
            ts.append(ev0.elapsed_time(ev1))
    return min(ts), sum(ts) / len(ts)


def _worker_body(rank, world_size, ranks_per_node, numels, dtypes, device, bench):
    shmem.shmem_torch_process_group_init("default")
    assert shmem.shmem_mype() == rank and shmem.shmem_npes() == world_size

    max_itemsize = max(torch.tensor([], dtype=d).element_size() for d in dtypes)
    per_rank_bytes = max(numels) * max_itemsize + 4096

    # out-in-place handle (result read from the ring buffer; no copy-OUT).
    h_oip = HierAllGather(
        my_pe=rank, npes=world_size, ranks_per_node=ranks_per_node,
        input_buffer_size=per_rank_bytes,
        output_buffer_size=per_rank_bytes * world_size,
        out_in_place=True,
    )
    # Default staged handle (fills the user output via finish_sync copy-OUT).
    h_def = HierAllGather(
        my_pe=rank, npes=world_size, ranks_per_node=ranks_per_node,
        input_buffer_size=per_rank_bytes,
        output_buffer_size=per_rank_bytes * world_size,
    )
    if rank == 0:
        print(f"out-in-place: world={world_size} ranks_per_node={ranks_per_node} "
              f"num_nodes={h_oip.num_nodes}")
    assert h_oip.num_nodes >= 2, "out-in-place test needs num_nodes>=2"

    stream = torch.cuda.current_stream()
    try:
        for dtype in dtypes:
            for numel in numels:
                if (numel * torch.tensor([], dtype=dtype).element_size()) % 4 != 0:
                    continue
                inp = _make_input(dtype, numel, rank, device)
                out_ref = torch.empty(numel * world_size, dtype=dtype, device=device)
                dist.all_gather_into_tensor(out_ref, inp)

                # out-in-place: __call__ leaves the result in the ring buffer.
                dummy = torch.empty(0, dtype=dtype, device=device)
                assert h_oip(inp, dummy, numel, stream)
                stream.synchronize()
                res = h_oip.result_tensor(numel, dtype, device)
                _check(res, out_ref, dtype, numel, "out_in_place")

                # default staged: fills the user output buffer.
                out_def = torch.empty(numel * world_size, dtype=dtype, device=device)
                assert h_def(inp, out_def, numel, stream)
                stream.synchronize()
                _check(out_def, out_ref, dtype, numel, "default")

                if rank == 0:
                    print(f"  ok dtype={dtype} numel={numel}")
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print("test_hier_allgather_out_in_place: PASSED")

        if bench:
            bdtype, bnumel = torch.float32, max(numels)
            inp = _make_input(bdtype, bnumel, rank, device)
            out_def = torch.empty(bnumel * world_size, dtype=bdtype, device=device)
            dummy = torch.empty(0, dtype=bdtype, device=device)

            def call_def():
                assert h_def(inp, out_def, bnumel, stream)
                stream.synchronize()

            def call_oip():
                assert h_oip(inp, dummy, bnumel, stream)
                stream.synchronize()

            out_ref = torch.empty(bnumel * world_size, dtype=bdtype, device=device)

            d_min, d_avg = _bench(call_def)
            o_min, o_avg = _bench(call_oip)
            r_min, r_avg = _bench(lambda: dist.all_gather_into_tensor(out_ref, inp))
            if rank == 0:
                tot_gb = bnumel * world_size * 4 / 1e9
                print(
                    f"[bench] world={world_size} fp32 numel={bnumel} out={tot_gb:.3f}GB\n"
                    f"  default(copy-OUT)  min={d_min:.3f}ms avg={d_avg:.3f}ms "
                    f"BW={tot_gb/(d_min/1e3):.1f}GB/s\n"
                    f"  out-in-place       min={o_min:.3f}ms avg={o_avg:.3f}ms "
                    f"BW={tot_gb/(o_min/1e3):.1f}GB/s\n"
                    f"  rccl               min={r_min:.3f}ms avg={r_avg:.3f}ms "
                    f"BW={tot_gb/(r_min/1e3):.1f}GB/s"
                )
            dist.barrier()
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        del h_oip, h_def
        dist.barrier()
        shmem.shmem_finalize()


def _spawn_worker(rank, world_size, ranks_per_node, port, numels, dtypes, bench):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        _worker_body(rank, world_size, ranks_per_node, numels, dtypes, device, bench)


def test_hier_allgather_out_in_place(world_size=None, ranks_per_node=None,
                                     numels=None, dtypes=None, bench=False):
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    os.environ.setdefault("MORI_SDMA_NUM_CHANNELS", "1")
    if world_size is None:
        world_size = torch.cuda.device_count()
    assert world_size >= 2
    if ranks_per_node is None:
        # Default: split into 2 simulated nodes so num_nodes>=2 (out-in-place is
        # an N>=2-only path).
        ranks_per_node = world_size // 2
    assert ranks_per_node >= 1 and world_size % ranks_per_node == 0
    assert world_size // ranks_per_node >= 2, "out-in-place needs num_nodes>=2"
    if numels is None:
        numels = [1024, 1024 * 1024, 16 * 1024 * 1024]
    if dtypes is None:
        dtypes = _DEFAULT_DTYPES
    port = get_free_port()
    torch.multiprocessing.spawn(
        _spawn_worker,
        args=(world_size, ranks_per_node, port, numels, dtypes, bench),
        nprocs=world_size,
        join=True,
    )


def _run_torchrun(numels, dtypes, bench=False):
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    ranks_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank,
                            world_size=world_size, device_id=device)
    world_group = torch.distributed.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)
    try:
        _worker_body(rank, world_size, ranks_per_node, numels, dtypes, device, bench)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="out-in-place copy-OUT elimination test")
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--ranks-per-node", type=int, default=None)
    parser.add_argument("--numels", type=int, nargs="+", default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()

    if args.dtype is not None:
        from tests.python.utils import string_to_dtype
        dtypes = [string_to_dtype(args.dtype)]
    else:
        dtypes = _DEFAULT_DTYPES
    numels = args.numels if args.numels is not None else [1024, 1024 * 1024, 16 * 1024 * 1024]

    try:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            _run_torchrun(numels, dtypes, bench=args.bench)
        else:
            test_hier_allgather_out_in_place(
                world_size=args.world_size, ranks_per_node=args.ranks_per_node,
                numels=numels, dtypes=dtypes, bench=args.bench,
            )
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
