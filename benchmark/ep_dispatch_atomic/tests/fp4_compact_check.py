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
"""Byte-exact check of the fp4 compaction path -- the config the perf number is for.

combine's output cannot be value-compared in fp4, so the chain is verified at the
byte level instead: after dispatch + ep_compact, dense row r must equal the SOURCE
row that the reverse map says lives at the segmented slot r came from. That closes
dispatch -> compact without needing a cast fp4 does not have.
"""
import ctypes
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, "/app/mori/tests/python/ops/dispatch_combine_v2")
import _data
import mori.cco as cco
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp

HIDDEN, TOPK, EPR, SEED, M = 7168, 6, 96, 1234, 512
ROWB = HIDDEN // 2  # fp4: two e2m1 per byte -> 3584
lib = ctypes.CDLL("/tmp/libepcompact.so")
lib.ep_compact.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
lib.ep_compact.restype = ctypes.c_int


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    gp = _data.make_generator(_data.seed_for(SEED, rank))
    gr = _data.make_generator(_data.seed_for(SEED, rank, routing=True))
    inp = _data.make_payload((M, HIDDEN), "norm", gp, torch.float4_e2m1fn_x2).to(dev)
    wts = torch.rand(M, TOPK, generator=gr, dtype=torch.float32).to(dev)
    idx = (
        torch.stack(
            [torch.randperm(world * EPR, generator=gr)[:TOPK] for _ in range(M)]
        )
        .to(torch.int32)
        .to(dev)
    )
    obj = [cco.Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    comm = cco.Communicator.init(
        world, rank, obj[0], 2 * (world * M * HIDDEN * 2 * 2 + (16 << 20)) + (512 << 20)
    )
    op = EpDispatchCombineOp(
        EpDispatchCombineConfig(
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
        ),
        comm,
    )
    counts = torch.zeros(world, dtype=torch.int32, device=dev)
    cap = op.recv_tokens().shape[0]
    stride = cap // world
    dense = torch.zeros(cap, ROWB, dtype=torch.uint8, device=dev)

    counts.zero_()
    *_, total_t, r = op.dispatch(inp, wts, counts, idx, return_routing=True)
    torch.cuda.synchronize()
    dist.barrier()
    total = int(total_t.cpu().item())
    dense.zero_()
    lib.ep_compact(
        op.recv_tokens().data_ptr(),
        dense.data_ptr(),
        counts.data_ptr(),
        world,
        stride,
        ROWB,
        total,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    dist.barrier()

    c = counts.cpu().tolist()
    # dense row r came from segmented slot s*stride + (r - prefix_s)
    segslot, acc = [], 0
    for s in range(world):
        segslot += [s * stride + k for k in range(c[s])]
        acc += c[s]
    assert acc == total, (acc, total)
    ss = torch.tensor(segslot, device=dev)
    tis = r.disp_tok_id_to_src_tok_id_local[ss].cpu()
    src_pe, src_tok = (tis // M).long(), (tis % M).long()
    got = dense[:total].cpu()
    bad = 0
    for pe in src_pe.unique().tolist():
        sel = src_pe == pe
        ref = _data.make_payload(
            (M, HIDDEN),
            "norm",
            _data.make_generator(_data.seed_for(SEED, int(pe))),
            torch.float4_e2m1fn_x2,
        ).view(torch.uint8)
        bad += int((got[sel] != ref[src_tok[sel]]).any(dim=1).sum())
    # dense must be source-ordered and gap-free
    order_ok = int((src_pe.diff() < 0).sum())
    t = torch.tensor([bad, order_ok])
    dist.all_reduce(t)
    if rank == 0:
        print(
            f"[fp4+compact] {'PASS' if t[0].item()==0 and t[1].item()==0 else 'FAIL'}  "
            f"rows={total} counts={c} byte_mismatches={int(t[0])} "
            f"out_of_order={int(t[1])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
