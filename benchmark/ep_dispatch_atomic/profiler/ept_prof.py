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
"""epv1-style trace profiling of mori's EPv2 fp4 dispatch, every warp on every CU.

The device side is v1's own mori::core::profiler (per-warp ring of (ts, meta)
int64 pairs). Here we only have to hand it a buffer, drain it, and hand the
drained bytes to v1's export_to_perfetto, which is used unmodified.

The buffer rides in on `scales`: EpArgs.scalesBuf is dereferenced only under
`kCfg.scaleBytes > 0` and this config has no scale row, so the pointer is free.
"""
import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, "/app/mori/tests/python/ops/dispatch_combine_v2")
import _data
import mori.cco as cco
from mori.kernel_profiler import export_to_perfetto
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp

HIDDEN, TOPK, EPR, SEED = 7168, 6, 96, 1234
M = int(os.environ.get("M", 512))
PAIRS = int(os.environ.get("PAIRS", 5))
WARMUP = int(os.environ.get("WARMUP", 20))
MHZ = float(os.environ.get("MHZ", 99.845))  # measured on this node, not assumed

EVENTS_PER_WARP = 16384  # MAX_TRACE_EVENTS_PER_WARP
WORDS_PER_WARP = EVENTS_PER_WARP * 2  # MAX_DEBUG_TIMESTAMP_PER_WARP
OFF_WORDS = 4096  # kEpProfOffWords in the kernel
MAX_WARPS = int(os.environ.get("MAX_WARPS", 4096))  # PROFILER_WARPS_PER_RANK

SLOTS = {
    0: "Setup",
    1: "Routing",
    2: "SlotReserve",
    3: "MetaStage",
    4: "MetaTdm",
    5: "PayloadTdm",
    6: "GridTicket",
    7: "DrainSpin",
    8: "FenceAgent",
    9: "FenceSignal",
    10: "InboundWait",
}


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

    prof = torch.zeros(
        OFF_WORDS + MAX_WARPS * WORDS_PER_WARP, dtype=torch.int64, device=dev
    )
    if rank == 0:
        print(
            f"# profiler buffer {prof.numel()*8/2**30:.2f} GiB, "
            f"{MAX_WARPS} warps x {EVENTS_PER_WARP} events",
            flush=True,
        )

    def lockstep():
        torch.cuda.synchronize()
        dist.barrier()

    # warm up (and JIT-compile) with the profiler buffer attached, then clear it so
    # the exported trace holds only the measured launches.
    *_, total_t, r0 = op.dispatch(inp, wts, prof, idx, return_routing=True)
    lockstep()
    total = int(total_t.cpu().item())
    buf = op.combine_in_view()[:total]
    op.combine(buf, routing=r0)
    lockstep()
    for _ in range(WARMUP):
        *_, r = op.dispatch(inp, wts, prof, idx, return_routing=True)
        op.combine(buf, routing=r)
    lockstep()

    prof.zero_()
    lockstep()
    for _ in range(PAIRS):  # no host sync inside: steady state
        *_, r = op.dispatch(inp, wts, prof, idx, return_routing=True)
        op.combine(buf, routing=r)
    lockstep()

    # drain: each warp's ring holds `off` int64 words starting at its own base
    offs = prof[:OFF_WORDS].view(torch.int32)[:MAX_WARPS].cpu()
    parts = []
    for w in range(MAX_WARPS):
        o = int(offs[w])
        if o <= 0:
            continue
        base = OFF_WORDS + w * WORDS_PER_WARP
        parts.append(prof[base : base + o].cpu())
    live = len(parts)
    drained = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.int64)
    torch.save(drained, f"/tmp/eptrace_rank{rank}.pt")
    export_to_perfetto(
        drained,
        filename=f"/tmp/eptrace_rank{rank}.json",
        slot_map=SLOTS,
        gpu_freq_ghz=MHZ / 1e3,
    )
    print(
        f"[rank {rank}] warps traced={live}  events={drained.numel()//2}  "
        f"recv={total}",
        flush=True,
    )


if __name__ == "__main__":
    main()
