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
"""Eight-rank pull publication regression, launched by test_gemm_ag.py."""

import argparse
import os
from pathlib import Path
import runpy
import sys
import time

import flydsl.expr as fx
import torch
import torch.distributed as dist

from mori.cco import CCODevCommRequirements, Communicator, GDA_CONNECTION_NONE, UniqueId
from mori.ops.gemm_ag import ag_config, build_lsa_ag
from mori.ops.gemm_ag._gemm_a16w16_8wave import compile_bf16_gemm_ag
from mori.tensor_utils import from_gpu_ptr


def run_dirty_window_benchmark():
    """Exercise the real benchmark with recycled memory and a slow rank 0."""
    from mori.ops.gemm_ag import _gemm_a16w16_8wave as bf16gemm

    cfg = ag_config(world_size=8, m=2048, n=2048, elem_bytes=4)
    register = Communicator.register_window
    compile_gemm = bf16gemm.compile_bf16_gemm_ag

    def dirty_register(comm, ptr, size):
        win = register(comm, ptr, size)
        # Arrival slots look complete while the local epoch counter is zero.
        # Clearing only recv would let every block bypass its first barrier.
        from_gpu_ptr(ptr, (cfg.recv_off // 4,), torch.int32).fill_(0x7FFFFF01)
        from_gpu_ptr(ptr + cfg.flag_off, (cfg.max_blocks,), torch.int32).zero_()
        torch.cuda.synchronize()
        return win

    def delayed_compile(*args, **kwargs):
        launch = compile_gemm(*args, **kwargs)
        first = True

        def delayed_launch(*launch_args, **launch_kwargs):
            nonlocal first
            if first and int(os.environ["RANK"]) == 0:
                time.sleep(5)
            first = False
            return launch(*launch_args, **launch_kwargs)

        return delayed_launch

    Communicator.register_window = dirty_register
    bf16gemm.compile_bf16_gemm_ag = delayed_compile
    bench = (
        Path(__file__).resolve().parents[3]
        / "benchmark/cco/flydsl/gemm_ag/bench_gemm_ag.py"
    )
    sys.argv = [
        str(bench),
        "--mode",
        "split-lsa-pull",
        "--in-dtype",
        "bf16",
        "--out-dtype",
        "fp32",
        "--block-m",
        "128",
        "--block-n",
        "256",
        "-m",
        "2048",
        "--out-dim",
        "2048",
        "-k",
        "7168",
        "--warmup",
        "0",
        "--iters",
        "1",
        "--no-graph",
    ]
    runpy.run_path(str(bench), run_name="__main__")


def run(bm, out_dtype, replays):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    uid = UniqueId.from_bytes(payload[0])
    m = n = 2048
    k = 7168
    dtype = torch.float32 if out_dtype == "fp32" else torch.bfloat16
    cfg = ag_config(
        world_size=world,
        m=m,
        n=n,
        block_m=bm,
        block_n=256,
        elem_bytes=4 if out_dtype == "fp32" else 2,
    )
    # Small integer operands keep FP32 accumulation exact. Build the oracle
    # on the CPU; rank and generation shift A by a known integer, so each
    # replay has a different answer without another reference GEMM.
    generator = torch.Generator().manual_seed(1234)
    ah = torch.randint(-2, 3, (m, k), generator=generator).float()
    bh = torch.randint(-2, 3, (n, k), generator=generator).float()
    base = ah @ bh.T
    col_sum = bh.sum(dim=1)
    a = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    b = bh.to(device="cuda", dtype=torch.bfloat16)
    a_arg, b_arg = a.view(torch.int16).view(-1), b.view(torch.int16).view(-1)
    gemm = compile_bf16_gemm_ag(
        cfg,
        rank,
        K=k,
        BLOCK_M=bm,
        BLOCK_N=256,
        out_dtype=out_dtype,
        peer_uncached=True,
    )
    pull = build_lsa_ag(cfg, rank, direction="pull")
    with Communicator.init(
        world, rank, uid, per_rank_vmm=2 * cfg.window_bytes + (64 << 20)
    ) as comm:
        mem = comm.alloc_mem(cfg.window_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        from_gpu_ptr(mem.ptr, (cfg.window_bytes,), torch.uint8).zero_()
        recv = from_gpu_ptr(mem.ptr + cfg.recv_off, (world, m, n), dtype)
        c = recv[rank].view(-1)
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = reqs.gda_counter_count = 0
        dc = comm.create_dev_comm(reqs)

        def once():
            stream = fx.Stream(torch.cuda.current_stream())
            gemm(a_arg, b_arg, c, m, n, dc.ptr, win.handle, stream=stream)
            pull(dc.ptr, win.handle, stream=stream)

        graph = None
        for generation in range(replays + 1):
            a.copy_(ah + rank * 8 + generation)
            # Poison every slot on every replay, not just the initial output.
            recv.fill_(float("nan"))
            expected = torch.stack(
                [
                    (base + (peer * 8 + generation) * col_sum[None, :]).to(dtype)
                    for peer in range(world)
                ]
            )
            torch.cuda.synchronize()
            dist.barrier()
            if generation % world == rank:
                # Rotate the slow producer; the pull's entry barrier must
                # wait for its GEMM, including all of its C stores.
                torch.cuda._sleep(200_000)
            if graph is None:
                once()
            else:
                graph.replay()
            actual = recv.cpu()
            local_ok = torch.tensor(int(torch.equal(actual, expected)))
            dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
            if not local_ok.item():
                bad = [
                    peer
                    for peer in range(world)
                    if not torch.equal(actual[peer], expected[peer])
                ]
                raise AssertionError(
                    f"rank={rank} generation={generation} bad peer slabs={bad}"
                )
            if graph is None:
                # Capture only GEMM + pull; input updates remain outside the
                # graph so each replay has to publish a different answer.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    once()
        if rank == 0:
            print(
                f"PULL_OK bm={bm} out={out_dtype} ranks={world} eager=1 replays={replays}",
                flush=True,
            )
        del graph
        comm.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-m", type=int, choices=(128, 256), default=128)
    parser.add_argument("--out-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--replays", type=int, default=16)
    parser.add_argument("--dirty-window-benchmark", action="store_true")
    args = parser.parse_args()
    if args.dirty_window_benchmark:
        run_dirty_window_benchmark()
    else:
        run(args.block_m, args.out_dtype, args.replays)
