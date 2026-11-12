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
PUSH = os.environ.get("MORI_EP_BF16_PUSH", "1") == "1"
HOWS = ("plain", "in_view", "mixed", "weights", "snapshot")
PULLS = ("weights", "snapshot")


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


def call_sizes(lim, buckets, world):
    edges = [e for e, _, _ in buckets if e is not None and lim < e < M]
    steps = [n for e in edges for n in (e, e + 1)]
    same = [n for n in (1, lim, lim + 1, *steps, M) if n]
    calls = [(n,) * world for n in same + same[::-1]]
    for e in sorted({lim + 1, *edges, M // 4, M // 2}):
        if lim < e < M:
            calls += [tuple(e + r % 2 for r in range(world)), (1,) * world]
    spread = [lim + 1, *(e + 1 for e in edges), M]
    calls.append(tuple(spread[r % len(spread)] for r in range(world)))
    calls.append(tuple(spread[::-1][r % len(spread)] for r in range(world)))
    if lim < 4000 and M >= 4200:
        calls.append((4000,) + (4200,) * (world - 1))
    return calls * ROUNDS


def make_inputs(i, src, n, world):
    g = torch.Generator().manual_seed(7919 * i + src)
    x = torch.randn(n, HIDDEN, generator=g).to(torch.bfloat16)
    idx = torch.stack(
        [torch.randperm(world * EPR, generator=g)[:TOPK] for _ in range(n)]
    ).to(torch.int32)
    w = torch.rand(n, TOPK, generator=g) + 0.5
    return x, idx, w


def pushes(n, how, lim):
    return PUSH and n > lim and how not in PULLS


def run(cfg, comm, lim, sizes, args):
    rank, world = cfg.rank, cfg.world_size
    op = EpDispatchCombineOp(cfg, comm)
    checks = {}

    def check(name, ok):
        checks[name] = checks.get(name, True) and bool(ok)

    def sync():
        torch.cuda.synchronize()
        comm.barrier()

    def hows(i):
        return [h for h in HOWS if h != "snapshot" or min(sizes[i]) > lim]

    def pair(i, how):
        x, idx, w = args[i]
        out, _, _, _, _, h = op.dispatch(
            x,
            w,
            None,
            idx,
            return_routing=True,
            snapshot=how == "snapshot",
            max_num_inp_token_across_ranks=max(sizes[i]),
        )
        if how == "mixed":
            how = "in_view" if (rank + i) % 2 else "plain"
        if how in ("plain", "snapshot"):
            o, _ = op.combine(out, routing=h)
        else:
            view = op.combine_in_view()[: out.shape[0]]
            view.copy_(out)
            o, _ = op.combine(view, weights=w if how == "weights" else None, routing=h)
        return o.clone()

    built = bool(op._push_specs)
    check("push_plans_built" if PUSH else "no_push_plans", built == PUSH)
    land = op.push_landing() if built else None
    refs = []
    for i, ns in enumerate(sizes):
        n = ns[rank]
        x, idx, _ = (t.cpu() for t in args[i])
        u = torch.zeros(n, world).scatter_(1, idx.long() // EPR, 1.0)
        want = u.sum(1, keepdim=True) * x.float()
        got = {}
        for how in hows(i):
            if land is not None:
                land.zero_()
            sync()
            got[how] = pair(i, how)
            sync()
            if land is not None:
                check(f"path_{how}", bool(land.any()) == pushes(max(ns), how, lim))
            check(
                f"bf16_sum_{how}",
                torch.allclose(got[how].float().cpu(), want, atol=2e-2, rtol=2e-2),
            )
            if how != "plain":
                check(f"plain_equals_{how}", torch.equal(got["plain"], got[how]))
            refs.append(got[how])

    got = [pair(i, how) for i in range(len(sizes)) for how in hows(i)]
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
    buckets = bf16_push(world, HIDDEN, TOPK, experts_per_rank=EPR)
    lay = ll128_layout(cfg)
    lim = 0 if lay is None else lay["max_tok"]
    if buckets is None or lim >= M:
        if rank == 0:
            print(f"# BF16PUSH: SKIP (EP{world} M={M} has no bf16 push entry)")
        dist.destroy_process_group()
        return
    sizes = call_sizes(lim, buckets, world)
    args = [
        tuple(t.to(dev) for t in make_inputs(i, rank, ns[rank], world))
        for i, ns in enumerate(sizes)
    ]
    if rank == 0:
        print(
            f"# BF16PUSH EP{world} M={M} ll128={lim} push={int(PUSH)} "
            f"buckets={buckets} calls={sizes[: len(sizes) // ROUNDS]}"
        )
    torch.cuda.synchronize()

    obj = [cco.Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    vmm = 2 * (world * M * HIDDEN * 2 * 3 + (16 << 20)) + (1 << 30)
    comm = cco.Communicator.init(world, rank, obj[0], vmm)

    failures = run(cfg, comm, lim, sizes, args)

    if rank == 0:
        print(f"# total bf16 push select failures: {failures}", flush=True)
    comm.destroy()
    dist.destroy_process_group()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
