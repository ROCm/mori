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
"""Variant A + compaction, wired into the flow.

    dispatch  -> segmented landing zone (no remote atomic)
    ep_compact-> dense run                     <-- new kernel
    expert    -> operates on a DENSE buffer, exactly as it does on stock
    ep_expand -> back to segmented positions   <-- new kernel
    combine   -> gathers via the sender's map, unchanged

The expert here is deliberately LAYOUT-UNAWARE -- it treats rows [0,total) as the
valid set, which is the stock contract. If the end-to-end value check passes with
that expert, the compaction has genuinely restored the stock layout and this is a
drop-in change.
"""
import ctypes
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
COMPACT = int(os.environ.get("COMPACT", 1))
ITERS = int(os.environ.get("ITERS", 50))

lib = ctypes.CDLL("/tmp/libepcompact.so")
for fn in (lib.ep_compact, lib.ep_expand):
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    gp = _data.make_generator(_data.seed_for(SEED, rank))
    gr = _data.make_generator(_data.seed_for(SEED, rank, routing=True))
    inp = _data.make_payload((M, HIDDEN), "norm", gp, torch.bfloat16).to(dev)
    wts = torch.rand(M, TOPK, generator=gr, dtype=torch.float32).to(dev)
    idx = (
        torch.stack(
            [torch.randperm(world * EPR, generator=gr)[:TOPK] for _ in range(M)]
        )
        .to(torch.int32)
        .to(dev)
    )
    U = (
        torch.zeros(M, world, dtype=torch.bool)
        .scatter_(1, idx.cpu().long() // EPR, True)
        .sum(1)
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
            kernel_backend="hip",
        ),
        comm,
    )

    counts = torch.zeros(world, dtype=torch.int32, device=dev)
    cap = op.recv_tokens().shape[0]
    stride = cap // world
    rowB = HIDDEN * 2
    dense = torch.zeros(cap, HIDDEN, dtype=torch.bfloat16, device=dev)

    def lockstep():
        torch.cuda.synchronize()
        dist.barrier()

    def compact(total):
        lib.ep_compact(
            op.recv_tokens().data_ptr(),
            dense.data_ptr(),
            counts.data_ptr(),
            world,
            stride,
            rowB,
            total,
            torch.cuda.current_stream().cuda_stream,
        )

    def expand(total):
        lib.ep_expand(
            op.combine_in_view().data_ptr(),
            dense.data_ptr(),
            counts.data_ptr(),
            world,
            stride,
            rowB,
            total,
            torch.cuda.current_stream().cuda_stream,
        )

    bad = 0
    for _ in range(3):
        counts.zero_()
        *_, total_t, r = op.dispatch(inp, wts, counts, idx, return_routing=True)
        lockstep()
        total = int(total_t.cpu().item())
        stage = op.combine_in_view()
        stage.zero_()
        if COMPACT:
            compact(total)
            # ---- stock, layout-UNAWARE identity expert on a dense buffer ----
            dense[total:].zero_()
            expand(total)
        else:
            stage[:total] = op.recv_tokens()[:total]  # stock expert, no compaction
        out, _ = op.combine(stage, routing=r)
        lockstep()
        exp = U.view(M, 1).float() * inp.float().cpu()
        if not torch.allclose(out.float().cpu(), exp, atol=2e-2, rtol=2e-2):
            bad += 1

    # Time the two local kernels in isolation, back to back after a full
    # rendezvous. Timing them inside the dispatch loop measured rank skew: with no
    # host sync per iteration the ranks drift and dispatch's inbound spin absorbs
    # it, which put "dispatch" at ~490us. compact/expand touch no peer, so a tight
    # local loop is both clean and representative.
    lockstep()

    def tight(fn):
        for _ in range(10):
            fn(total)
        torch.cuda.synchronize()
        a_, b_ = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a_.record()
        for _ in range(ITERS):
            fn(total)
        b_.record()
        torch.cuda.synchronize()
        return a_.elapsed_time(b_) / ITERS * 1000

    def graphed(fn, n=20):
        """Per-launch cost with the launch removed. A ctypes call costs several us
        of Python per invocation, which starves the GPU and makes a plain loop
        measure the host, not the kernel. Capturing N launches and replaying gives
        the figure a C++ caller or a captured graph would actually see."""
        for _ in range(5):
            fn(total)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                fn(total)
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        a_, b_ = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a_.record()
        for _ in range(20):
            g.replay()
        b_.record()
        torch.cuda.synchronize()
        return a_.elapsed_time(b_) / (20 * n) * 1000

    cg_us = graphed(compact) if COMPACT else 0.0
    eg_us = graphed(expand) if COMPACT else 0.0
    # Same kernels at the fp4 row width (3584 B), which is the target config. Only
    # the byte count differs, so this needs no correctness path -- buffers of the
    # right shape are enough to price it.
    f4a = torch.zeros(cap, 3584, dtype=torch.uint8, device=dev)
    f4b = torch.zeros(cap, 3584, dtype=torch.uint8, device=dev)

    def f4_compact(tot):
        lib.ep_compact(
            f4a.data_ptr(),
            f4b.data_ptr(),
            counts.data_ptr(),
            world,
            stride,
            3584,
            tot,
            torch.cuda.current_stream().cuda_stream,
        )

    def f4_expand(tot):
        lib.ep_expand(
            f4a.data_ptr(),
            f4b.data_ptr(),
            counts.data_ptr(),
            world,
            stride,
            3584,
            tot,
            torch.cuda.current_stream().cuda_stream,
        )

    f4c = graphed(f4_compact) if COMPACT else 0.0
    f4e = graphed(f4_expand) if COMPACT else 0.0

    c_us = tight(compact) if COMPACT else 0.0
    e_us = tight(expand) if COMPACT else 0.0

    # Same volume, same kernel, but out of ORDINARY device memory rather than the
    # symmetric arena -- the arena is peer-mapped, and that is the difference we
    # want named rather than folded into the total.
    plain = torch.zeros(cap, HIDDEN, dtype=torch.bfloat16, device=dev)

    def compact_plain(tot):
        lib.ep_compact(
            plain.data_ptr(),
            dense.data_ptr(),
            counts.data_ptr(),
            world,
            stride,
            rowB,
            tot,
            torch.cuda.current_stream().cuda_stream,
        )

    p_us = tight(compact_plain) if COMPACT else 0.0

    t = torch.tensor([bad])
    dist.all_reduce(t)
    if rank == 0:
        tag = "compact" if COMPACT else "no-compact"
        print(
            f"[{tag}] e2e {'PASS' if t.item()==0 else 'FAIL'} (bad={int(t.item())})  "
            f"counts={counts.tolist()} total={total}",
            flush=True,
        )
        print(
            f"[{tag}] GRAPHED (launch removed): compact {cg_us:.2f}us  expand {eg_us:.2f}us  "
            f"both {cg_us+eg_us:.2f}us",
            flush=True,
        )
        print(
            f"[{tag}] GRAPHED fp4 width (3584B): compact {f4c:.2f}us  expand {f4e:.2f}us  "
            f"both {f4c+f4e:.2f}us",
            flush=True,
        )
        print(
            f"[{tag}] ctypes loop (host-bound):  compact {c_us:.2f}us  expand {e_us:.2f}us  "
            f"| plain VRAM {p_us:.2f}us",
            flush=True,
        )


if __name__ == "__main__":
    main()
