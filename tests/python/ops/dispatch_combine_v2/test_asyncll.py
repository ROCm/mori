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
"""AsyncLL (hip backend, ccoSdma transport) correctness on one node.

A torchrun script, not a pytest file -- it needs eight ranks and a cco
communicator, like its neighbours here.

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 test_asyncll.py

Env: HIDDEN, TOPK, EPR, M, LOOP, plus SHAPES to override the swept list.

What is checked, and why these and not more:

  * **arrival count** -- every source token routed to any expert of mine arrives
    exactly once. Top-k agnostic, so it is the check that survives a top-k the
    positional comparison cannot handle, and an orphan or duplicated slot breaks
    it. This is the invariant v1's lane-tiling regression was found with.
  * **origin decode** -- the reverse map must decode to a live (rank, token).
    It catches a slot that was allocated but never written, which the count alone
    does not: an orphan leaves the recorded origin at whatever was there before.
  * **round trip** -- an identity expert must return each token scaled by its
    number of DISTINCT destination ranks. That is the end-to-end statement, and
    it is what fails if dedup, the slot maps or the reduction disagree.

The swept shapes are chosen for the two things that have actually broken:
top-k that does not divide the wavefront (64 % 6 == 4, DeepSeek-V4's shape), and
ranks that are handed no tokens at all.
"""
import os
import sys

import torch
import torch.distributed as dist

from mori.cco.communicator import Communicator
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp

# (hidden, topk, experts_per_rank, max_tokens)
_DEFAULT_SHAPES = [
    (2048, 8, 8, 32),  # small and square
    (7168, 8, 32, 128),  # the shape the bench and the tuning tables use
    (4096, 6, 48, 128),  # 64 % 6 == 4: trailing lanes alias the next warp's token
    (4096, 9, 32, 96),  # 64 % 9 == 1
    (2048, 12, 16, 64),  # 64 % 12 == 4
]


def _shapes():
    raw = os.environ.get("SHAPES")
    if not raw:
        if "HIDDEN" in os.environ or "TOPK" in os.environ:
            return [
                (
                    int(os.environ.get("HIDDEN", 7168)),
                    int(os.environ.get("TOPK", 8)),
                    int(os.environ.get("EPR", 32)),
                    int(os.environ.get("M", 128)),
                )
            ]
        return _DEFAULT_SHAPES
    return [tuple(int(x) for x in row.split(",")) for row in raw.split(";")]


class Dist:
    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    def bcast_uid(self, uid):
        objs = [uid if self.rank == 0 else None]
        dist.broadcast_object_list(objs, src=0)
        return objs[0]

    def allreduce_sum(self, value):
        t = torch.tensor([value], dtype=torch.int64)
        dist.all_reduce(t)
        return int(t.item())


def _routing(rank, world, m, topk, epr, dev, gen, num_tokens):
    """Random top-k over every expert, plus the per-rank token count."""
    total_experts = epr * world
    idx = torch.stack(
        [
            torch.randperm(total_experts, generator=gen, device=dev)[:topk]
            for _ in range(m)
        ]
    ).to(torch.int32)
    return idx[:num_tokens]


def run_shape(d, comm, hidden, topk, epr, m, loops):
    rank, world = d.rank, d.world
    dev = torch.device("cuda", rank)
    cfg = EpDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=hidden,
        max_num_inp_token_per_rank=m,
        num_experts_per_rank=epr,
        num_experts_per_token=topk,
        data_type=torch.bfloat16,
        kernel_backend="hip",
        async_ll=True,
        async_ll_transport=os.environ.get("TRANSPORT", "sdma"),
    )
    failures = 0
    with EpDispatchCombineOp(cfg, comm) as op:
        for it in range(loops):
            # A different token count per rank per iteration, including zero:
            # a rank with no tokens still has to publish its count or every peer
            # waits on it forever, and the per-token divisions must not divide by
            # that zero either.
            counts = [((r + it) * 7) % (m + 1) for r in range(world)]
            n = counts[rank]
            gen = torch.Generator(device=dev).manual_seed(1234 + rank * 17 + it)
            x = torch.randn(m, hidden, generator=gen, device=dev, dtype=torch.bfloat16)
            w = torch.rand(m, topk, generator=gen, device=dev, dtype=torch.float32)
            idx_full = _routing(rank, world, m, topk, epr, dev, gen, m)
            idx, xin, win = idx_full[:n], x[:n], w[:n]

            out, _, _, _, total, routing = op.dispatch_send(
                xin, win, None, idx, return_routing=True
            )
            op.dispatch_recv()
            torch.cuda.synchronize()
            dist.barrier()

            recv = int(total[0].item())

            # -- arrival count ------------------------------------------------
            idx_cpu = idx.cpu()
            gathered = [None] * world
            dist.all_gather_object(gathered, (n, idx_cpu))
            expected = 0
            for src_n, src_idx in gathered:
                for t in range(src_n):
                    if rank in {int(e) // epr for e in src_idx[t].tolist()}:
                        expected += 1
            if recv != expected:
                print(
                    f"[rank {rank}] h={hidden} k={topk} it={it}: recv={recv} "
                    f"expected={expected} -- slot assignment produced "
                    "orphan or duplicate slots",
                    flush=True,
                )
                failures += 1

            # -- origin decode ------------------------------------------------
            tis = routing.disp_tok_id_to_src_tok_id_local[:recv].cpu()
            src_pe, src_tok = (tis // m).tolist(), (tis % m).tolist()
            bad_origin = sum(
                1
                for pe, tok in zip(src_pe, src_tok)
                if not (0 <= pe < world and 0 <= tok < gathered[pe][0])
            )
            if bad_origin:
                print(
                    f"[rank {rank}] h={hidden} k={topk} it={it}: {bad_origin} "
                    "recv slots decode to a source that never sent -- the slot "
                    "was never written",
                    flush=True,
                )
                failures += 1

            # -- round trip ---------------------------------------------------
            cout, _ = op.combine_send(out[:recv].clone(), None, idx, routing=routing)
            op.combine_recv()
            torch.cuda.synchronize()
            dist.barrier()

            if n:
                uniq = torch.tensor(
                    [
                        len({int(e) // epr for e in idx_cpu[t].tolist()})
                        for t in range(n)
                    ],
                    device=dev,
                    dtype=torch.float32,
                )
                want = (xin.float() * uniq[:, None]).to(torch.bfloat16)
                got = cout[:n]
                if not torch.allclose(got.float(), want.float(), atol=1e-2, rtol=1e-2):
                    diff = (got.float() - want.float()).abs()
                    print(
                        f"[rank {rank}] h={hidden} k={topk} it={it}: combine "
                        f"mismatch, max |diff| {diff.max().item():.4f}",
                        flush=True,
                    )
                    failures += 1
    return failures


def main():
    d = Dist()
    shapes = _shapes()
    loops = int(os.environ.get("LOOP", 2))

    failures = 0
    # One communicator PER SHAPE, each with its own unique id, rather than one for
    # the whole run. Each shape's arena is a differently-sized window, and on
    # ROCm 7.0-7.2.x mapping several of those into one reservation trips a clr
    # bug: hipMemSetAccess validates coverage by walking the sub-buffers from
    # index 0 regardless of the pointer it was given, so every window after the
    # first fails with "invalid argument" (known-issues skill, Issue 2). Nothing
    # to do with these kernels -- the same failure reproduces on an untouched
    # tree, in test_ep_backend_parity, which likewise builds two ops on one
    # communicator -- but the shapes are the point of this test, so it is
    # structured to stay out of the bug's way.
    for hidden, topk, epr, m in shapes:
        uid = Communicator.get_unique_id() if d.rank == 0 else None
        uid = d.bcast_uid(uid)
        win_bytes = d.world * hidden * m * 2 * 4 + (1 << 24)
        with Communicator.init(
            d.world, d.rank, uid, per_rank_vmm=6 * win_bytes + (1 << 28)
        ) as comm:
            failures += run_shape(d, comm, hidden, topk, epr, m, loops)
        if d.rank == 0:
            print(
                f"# asyncll hidden={hidden} topk={topk} epr={epr} m={m}: done",
                flush=True,
            )

    total = d.allreduce_sum(failures)
    if d.rank == 0:
        print(
            f"# {'PASS' if total == 0 else 'FAIL'}: {total} failed checks across "
            f"{d.world} ranks, {len(shapes)} shapes x {loops} iterations",
            flush=True,
        )
    dist.destroy_process_group()
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
