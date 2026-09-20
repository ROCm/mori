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
"""End-to-end dispatch->expert->combine check, in bf16 so combine's OUTPUT is compared.

The expert is the identity, exactly as bench_ep.py models it -- but applied over the
slot set the layout actually uses. That is the whole point: variant A changes where
received rows live, so the consumer must read them there. bench_ep's expert is hard
-coded to the dense run, which is why it fails under seg; this shows the kernel is
correct once the consumer honours the layout.

Success criterion is bench_ep's: with an identity expert, combine[t] == U[t]*input[t],
where U[t] is the number of distinct ranks token t was routed to.
"""
import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, "/app/mori/tests/python/ops/dispatch_combine_v2")
import _data
import mori.cco as cco
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp

HIDDEN, TOPK, EPR, SEED = 7168, 6, 96, 1234
M = int(os.environ.get("M", 512))
LAYOUT = os.environ.get("LAYOUT", "dense")


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    n_experts = world * EPR

    gp = _data.make_generator(_data.seed_for(SEED, rank))
    gr = _data.make_generator(_data.seed_for(SEED, rank, routing=True))
    inp = _data.make_payload((M, HIDDEN), "norm", gp, torch.bfloat16).to(dev)
    wts = torch.rand(M, TOPK, generator=gr, dtype=torch.float32).to(dev)
    idx = (
        torch.stack([torch.randperm(n_experts, generator=gr)[:TOPK] for _ in range(M)])
        .to(torch.int32)
        .to(dev)
    )

    dest = idx.cpu().long() // EPR
    onehot = torch.zeros(M, world, dtype=torch.bool).scatter_(1, dest, True)
    U = onehot.sum(1)  # distinct dest ranks per token
    my_send = onehot.sum(0).to(torch.int64)
    gathered = [torch.zeros_like(my_send) for _ in range(world)]
    dist.all_gather(gathered, my_send)
    cnt_from = [int(gathered[s][rank]) for s in range(world)]

    obj = [cco.Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    comm = cco.Communicator.init(
        world, rank, obj[0], 2 * (world * M * HIDDEN * 2 * 2 + (16 << 20)) + (512 << 20)
    )
    cfg = EpDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=HIDDEN,
        max_num_inp_token_per_rank=M,
        num_experts_per_rank=EPR,
        num_experts_per_token=TOPK,
        data_type=torch.bfloat16,
        kernel_backend="hip",
    )
    op = EpDispatchCombineOp(cfg, comm)

    def lockstep():
        torch.cuda.synchronize()
        dist.barrier()

    bad = 0
    for it in range(3):
        *_, total_t, r = op.dispatch(inp, wts, None, idx, return_routing=True)
        lockstep()
        total = int(total_t.cpu().item())
        stride = op.recv_tokens().shape[0] // world
        if LAYOUT == "seg":
            slots = [
                i
                for s in range(world)
                for i in range(s * stride, s * stride + cnt_from[s])
            ]
        else:
            slots = list(range(total))
        st = torch.tensor(slots, device=dev)

        stage = op.combine_in_view()  # aliases out_tok: no staging copy
        stage.zero_()
        stage[st] = op.recv_tokens()[st].to(
            stage.dtype
        )  # identity expert, layout-aware
        out, _ = op.combine(stage, routing=r)
        lockstep()

        exp = U.view(M, 1).float() * inp.float().cpu()
        if not torch.allclose(out.float().cpu(), exp, atol=2e-2, rtol=2e-2):
            bad += 1
            if rank == 0 and it == 0:
                got = out.float().cpu()
                d = (got - exp).abs()
                print(
                    f"  worst abs err {d.max():.4f} at row {int(d.max(1).values.argmax())}; "
                    f"rows wrong: {int((d.max(1).values > 2e-2).sum())}/{M}",
                    flush=True,
                )

    t = torch.tensor([bad])
    dist.all_reduce(t)
    if rank == 0:
        print(
            f"[{LAYOUT}] combine e2e: {'PASS' if t.item()==0 else 'FAIL'}  "
            f"({int(t.item())} bad iters across {world} ranks) "
            f"total_recv={total} stride={stride} cnt_from={cnt_from} "
            f"U_range=[{int(U.min())},{int(U.max())}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
