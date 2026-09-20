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
"""Validate + time the segmented-reservation dispatch (variant A) against stock.

LAYOUT=dense  -> stock: one packed run of `total` slots, allocated by a remote RMW.
LAYOUT=seg    -> variant A: source s owns slots [s*stride, s*stride + cnt_s).

The check is byte-exact in both cases and uses the same evidence bench_ep.py uses
(the reverse map plus a regeneration of the sender's payload), but over the slot
set the layout actually claims. For seg it additionally asserts that segment s
contains ONLY source s's rows -- which is the whole safety property that lets the
remote allocator go away.
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
ITERS = int(os.environ.get("ITERS", 50))
PROF = int(os.environ.get("PROF", 1))
OFF_WORDS, WORDS_PER_WARP, MAX_WARPS = 4096, 32768, 4096


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    n_experts = world * EPR

    gp = _data.make_generator(_data.seed_for(SEED, rank))
    gr = _data.make_generator(_data.seed_for(SEED, rank, routing=True))
    inp = _data.make_payload((M, HIDDEN), "norm", gp, torch.float4_e2m1fn_x2).to(dev)
    wts = torch.rand(M, TOPK, generator=gr, dtype=torch.float32).to(dev)
    idx = (
        torch.stack([torch.randperm(n_experts, generator=gr)[:TOPK] for _ in range(M)])
        .to(torch.int32)
        .to(dev)
    )

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
        dispatch_data_type=torch.float4_e2m1fn_x2,
        combine_data_type=torch.bfloat16,
        kernel_backend="hip",
    )
    op = EpDispatchCombineOp(cfg, comm)
    prof = (
        torch.zeros(
            OFF_WORDS + MAX_WARPS * WORDS_PER_WARP, dtype=torch.int64, device=dev
        )
        if PROF
        else None
    )

    # how many of source s's tokens are routed to me (dedup: at most one slot per token)
    dest = idx.cpu().long() // EPR
    onehot = torch.zeros(M, world, dtype=torch.bool).scatter_(1, dest, True)
    my_send = onehot.sum(0).to(torch.int64)
    gathered = [torch.zeros_like(my_send) for _ in range(world)]
    dist.all_gather(gathered, my_send)
    cnt_from = [int(gathered[s][rank]) for s in range(world)]

    def lockstep():
        torch.cuda.synchronize()
        dist.barrier()

    *_, total_t, r = op.dispatch(inp, wts, prof, idx, return_routing=True)
    lockstep()
    total = int(total_t.cpu().item())
    stride = op.recv_tokens().shape[0] // world

    if LAYOUT == "seg":
        slots, want_pe = [], []
        for s in range(world):
            slots += list(range(s * stride, s * stride + cnt_from[s]))
            want_pe += [s] * cnt_from[s]
    else:
        slots, want_pe = list(range(total)), None

    tis = r.disp_tok_id_to_src_tok_id_local[torch.tensor(slots, device=dev)].cpu()
    src_pe, src_tok = (tis // M).long(), (tis % M).long()
    # view as bytes BEFORE gathering: torch has no advanced indexing for fp4
    got = op.recv_tokens().view(torch.uint8)[torch.tensor(slots, device=dev)].cpu()
    bad_bytes = 0
    for pe in src_pe.unique().tolist():
        sel = src_pe == pe
        ref = _data.make_payload(
            (M, HIDDEN),
            "norm",
            _data.make_generator(_data.seed_for(SEED, int(pe))),
            torch.float4_e2m1fn_x2,
        ).view(torch.uint8)
        bad_bytes += int((got[sel] != ref[src_tok[sel]]).any(dim=1).sum())
    bad_seg = 0 if want_pe is None else int((src_pe != torch.tensor(want_pe)).sum())

    buf = op.combine_in_view()[:total]
    op.combine(buf, routing=r)
    lockstep()

    for _ in range(20):
        *_, rr = op.dispatch(inp, wts, prof, idx, return_routing=True)
        op.combine(buf, routing=rr)
    lockstep()
    ev = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(ITERS)
    ]
    for i in range(ITERS):
        ev[i][0].record()
        *_, rr = op.dispatch(inp, wts, prof, idx, return_routing=True)
        ev[i][1].record()
        op.combine(buf, routing=rr)
    lockstep()
    us = sum(a.elapsed_time(b) for a, b in ev) / ITERS * 1000

    t = torch.tensor([bad_bytes, bad_seg])
    dist.all_reduce(t)
    if rank == 0:
        ok = "PASS" if t[0].item() == 0 and t[1].item() == 0 else "FAIL"
        print(
            f"[{LAYOUT}] {ok}  slots_checked={len(slots)} total_recv={total} "
            f"stride={stride} cnt_from={cnt_from}",
            flush=True,
        )
        print(
            f"[{LAYOUT}] byte_mismatches={int(t[0])} wrong_segment_rows={int(t[1])} "
            f"dispatch_host_us={us:.1f}",
            flush=True,
        )
    if prof is not None:
        torch.save(
            prof[:OFF_WORDS].view(torch.int32)[:MAX_WARPS].cpu(),
            f"/tmp/segoff_rank{rank}.pt",
        )
        parts = []
        offs = prof[:OFF_WORDS].view(torch.int32)[:MAX_WARPS].cpu()
        for w in range(MAX_WARPS):
            o = int(offs[w])
            if o > 0:
                base = OFF_WORDS + w * WORDS_PER_WARP
                parts.append(prof[base : base + o].cpu())
        torch.save(
            torch.cat(parts) if parts else torch.zeros(0, dtype=torch.int64),
            f"/tmp/segtrace_{LAYOUT}_rank{rank}.pt",
        )


if __name__ == "__main__":
    main()
