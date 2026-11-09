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
import os
import sys

import torch
import torch.distributed as dist

import mori.cco as cco
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp
from mori.ops.dispatch_combine_v2.hip_backend import ll128_layout
from mori.ops.dispatch_combine_v2.hip_tuning_configs import bf16_push

HIDDEN = 7168
TOPK = 6
EPR = int(os.environ.get("EPR", 8))
M = int(os.environ.get("M", 8192))
ROUNDS = int(os.environ.get("ROUNDS", 2))
HOWS = ("plain", "in_view", "weights")


def make_cfg(rank, world):
    return EpDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=HIDDEN,
        max_num_inp_token_per_rank=M,
        num_experts_per_rank=EPR,
        num_experts_per_token=TOPK,
        data_type=torch.bfloat16,
        kernel_backend="hip",
    )


def call_sizes(lim, pull_max):
    edges = [1, lim, lim + 1, pull_max, pull_max + 1, M]
    once = edges + edges[::-1] + [lim + 1, M, pull_max, 1, pull_max + 1, lim]
    return once * ROUNDS


def make_inputs(i, src, n, world):
    g = torch.Generator().manual_seed(7919 * i + src)
    x = torch.randn(n, HIDDEN, generator=g).to(torch.bfloat16)
    idx = torch.stack(
        [torch.randperm(world * EPR, generator=g)[:TOPK] for _ in range(n)]
    ).to(torch.int32)
    w = torch.rand(n, TOPK, generator=g) + 0.5
    return x, idx, w


def pushes(n, how, lim, pull_max):
    if n <= lim or how == "weights":
        return False
    return how == "plain" or n > pull_max


def run(cfg, comm, lim, pull_max, sizes, args):
    rank, world = cfg.rank, cfg.world_size
    op = EpDispatchCombineOp(cfg, comm)
    checks = {}

    def check(name, ok):
        checks[name] = checks.get(name, True) and bool(ok)

    def sync():
        torch.cuda.synchronize()
        comm.barrier()

    def pair(i, how):
        x, idx, w = args[i]
        out, _, _, _, _, h = op.dispatch(x, w, None, idx, return_routing=True)
        if how == "plain":
            o, _ = op.combine(out, routing=h)
        else:
            view = op.combine_in_view()[: out.shape[0]]
            view.copy_(out)
            o, _ = op.combine(view, weights=w if how == "weights" else None, routing=h)
        return o.clone()

    built = bool(op._push_specs)
    check("push_plans_built", built)
    land = op.push_landing() if built else None
    refs = []
    for i, n in enumerate(sizes):
        x, idx, _ = (t.cpu() for t in args[i])
        u = torch.zeros(n, world).scatter_(1, idx.long() // EPR, 1.0)
        want = u.sum(1, keepdim=True) * x.float()
        got = {}
        for how in HOWS:
            if land is not None:
                land.zero_()
            sync()
            got[how] = pair(i, how)
            sync()
            if land is not None:
                check(f"path_{how}", bool(land.any()) == pushes(n, how, lim, pull_max))
            check(
                f"bf16_sum_{how}",
                torch.allclose(got[how].float().cpu(), want, atol=2e-2, rtol=2e-2),
            )
            refs.append(got[how])
        check("plain_equals_in_view", torch.equal(got["plain"], got["in_view"]))
        check("in_view_equals_weights", torch.equal(got["in_view"], got["weights"]))

    got = [pair(i, how) for i in range(len(sizes)) for how in HOWS]
    sync()
    check("eager_back_to_back", all(map(torch.equal, got, refs)))

    op.close()
    failures = 0
    for name, ok in checks.items():
        bad = torch.tensor([0 if ok else 1], dtype=torch.int32)
        dist.all_reduce(bad)
        failures += int(bad.item() != 0)
        if rank == 0:
            verdict = "PASS" if bad.item() == 0 else "FAIL"
            print(f"# BF16PUSH {name}: {verdict}", flush=True)
    return failures


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())

    cfg = make_cfg(rank, world)
    geom = bf16_push(world, HIDDEN, TOPK, experts_per_rank=EPR)
    lay = ll128_layout(cfg)
    if geom is None or lay is None or geom["in_view_pull_max"] >= M:
        if rank == 0:
            print(f"# BF16PUSH: SKIP (EP{world} has no push entry below M={M})")
        dist.destroy_process_group()
        return
    lim, pull_max = lay["max_tok"], geom["in_view_pull_max"]
    sizes = call_sizes(lim, pull_max)
    args = [
        tuple(t.to(dev) for t in make_inputs(i, rank, n, world))
        for i, n in enumerate(sizes)
    ]
    if rank == 0:
        print(
            f"# BF16PUSH EP{world} M={M} ll128={lim} pull_max={pull_max} sizes={sizes}"
        )
    torch.cuda.synchronize()

    obj = [cco.Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    vmm = 2 * (world * M * HIDDEN * 2 * 3 + (16 << 20)) + (1 << 30)
    comm = cco.Communicator.init(world, rank, obj[0], vmm)

    failures = run(cfg, comm, lim, pull_max, sizes, args)

    if rank == 0:
        print(f"# total bf16 push select failures: {failures}", flush=True)
    comm.destroy()
    dist.destroy_process_group()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
