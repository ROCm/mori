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
"""Quantise-and-dispatch: ``SdmaDispatch`` against mori's EPv2, FlyDSL backend.

Both legs deliver the same thing -- every token that routes to a rank, once per rank, as fp8
with one e8m0 per 32 values, plus the routing that row needs -- so the window being timed is
the same work:

    EPv2      per_1x32_mx_quant(x)  ->  op.dispatch(x_fp8, wts, scales, ids)
    SDMA      op.dispatch(x, ids, wts)          (quant, transfer and wait are one kernel)

EPv2's dispatch has no quant of its own (``quant_type`` is the combine wire, scatter-only), so
the quantisation is a kernel in front of it and the two run back to back on the CUs. The SDMA
op quantises on the CUs and copies with the copy engines at the same time, so what it costs is
about ``max(quant, transfer)`` rather than their sum -- that is the thing being measured.

Both are deduplicated (EPv2 dedups same-destination routes with a ballot), so the bytes on the
wire are the same; the difference is who moves them.

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 bench_vs_epv2.py \
        --network flash --bs-list 512,1024,2048,4096,8192
"""

import argparse
import os

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "20G")
os.environ.setdefault("MORI_ENABLE_SDMA", "1")

import torch  # noqa: E402  (before anything that pulls in mori.cco)
import torch.distributed as dist  # noqa: E402
from mori.cco import Communicator, UniqueId  # noqa: E402
from mori.ops.dispatch_combine_v2 import (  # noqa: E402
    EpDispatchCombineConfig,
    EpDispatchCombineOp,
)

from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant  # noqa: E402

from sdma_dispatch import SdmaDispatch  # noqa: E402
from test_dispatch_op import NETWORKS, make_inputs  # noqa: E402


def timed(fn, iters, warmup=5):
    """Median ms of one call, and the max over ranks of that median."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    ts = []
    for _ in range(iters):
        e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    med = torch.tensor([sorted(ts)[len(ts) // 2]], dtype=torch.float64)
    dist.all_reduce(med, op=dist.ReduceOp.MAX)
    return float(med)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", choices=NETWORKS, default="flash")
    ap.add_argument("--bs-list", default="512,1024,2048,4096,8192")
    ap.add_argument("--chunk-rows", type=int, default=512)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    net = NETWORKS[args.network]
    hidden, experts, topk = net["hidden"], net["experts"], net["topk"]
    bss = [int(v) for v in args.bs_list.split(",")]

    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", local)
    uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)

    max_tok = 1 << (max(bss) - 1).bit_length()
    scale_dim = hidden // 32
    # one arena for each op, sized for fp8 rows and their scales
    vmm = 6 * world * max_tok * (hidden + scale_dim + 256) + (1 << 30)
    rows = []
    with Communicator.init(
        world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=vmm
    ) as comm:
        sdma_op = SdmaDispatch(
            comm,
            experts=experts,
            topk=topk,
            hidden=hidden,
            max_tok=max_tok,
            chunk_rows=args.chunk_rows,
        )
        ep = EpDispatchCombineOp(
            EpDispatchCombineConfig(
                rank=rank,
                world_size=world,
                hidden_dim=hidden,
                max_num_inp_token_per_rank=max_tok,
                num_experts_per_rank=experts // world,
                num_experts_per_token=topk,
                data_type=torch.bfloat16,
                dispatch_data_type=torch.float8_e4m3fn,
                combine_data_type=torch.bfloat16,
                scale_dim=scale_dim,
                scale_type_size=1,
                kernel_backend="flydsl",
            ),
            comm,
        )

        for bs in bss:
            x, ids, wts = make_inputs(rank, args.seed, bs, hidden, experts, topk, dev)

            def ep_call():
                xq, sc = per_1x32_mx_quant(x, quant_mode="fp8")
                ep.dispatch(xq, wts, sc.view(torch.uint8), ids)

            def sdma_call():
                sdma_op.dispatch(x, ids, wts)

            # and the quant on its own, to show what is being overlapped
            def quant_only():
                per_1x32_mx_quant(x, quant_mode="fp8")

            t_ep = timed(ep_call, args.iters)
            t_sdma = timed(sdma_call, args.iters)
            t_q = timed(quant_only, args.iters)
            rows.append((bs, t_q, t_ep, t_sdma))
            if rank == 0:
                print(
                    f"bs={bs}: quant alone {t_q * 1000:.0f} us | "
                    f"EPv2 quant+dispatch {t_ep * 1000:.0f} us | "
                    f"SDMA {t_sdma * 1000:.0f} us | {t_ep / t_sdma:.2f}x",
                    flush=True,
                )
        sdma_op.close()

    if rank == 0:
        print(
            "\n| tokens/rank | quant alone | EPv2 quant+dispatch | SDMA quant+dispatch | 加速 |"
        )
        print("|---:|---:|---:|---:|---:|")
        for bs, q, e, s in rows:
            print(
                f"| {bs} | {q * 1000:.0f} us | {e * 1000:.0f} us | {s * 1000:.0f} us | {e / s:.2f}x |"
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
