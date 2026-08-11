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
"""Timing for the C++/JIT EP intranode op.

Same methodology as bench_dispatch_combine.py (the FlyDSL one) so the numbers are
comparable: cuda events around ITERS launches, warmup in lock-step.

    torchrun --standalone --nproc_per_node=8 bench_ep_cpp.py
    HIDDEN=7168 TOPK=8 EPR=32 SWEEP=128,512,4096 ITERS=50 torchrun ... bench_ep_cpp.py
"""

import ctypes
import os

import torch
import torch.distributed as dist

import mori.cco as cco
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp

HIDDEN = int(os.environ.get("HIDDEN", 7168))
TOPK = int(os.environ.get("TOPK", 8))
EPR = int(os.environ.get("EPR", 32))
WARMUP = int(os.environ.get("WARMUP", 10))
ITERS = int(os.environ.get("ITERS", 50))
SWEEP = [int(x) for x in os.environ.get("SWEEP", "128,512,4096").split(",")]
# What dispatch transports; combine is always bf16, so anything else here is the
# asymmetric case. fp4 packs 2 e2m1 per byte, half of what fp8 moves.
_DISP_DT = {
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
    "fp4": torch.float4_e2m1fn_x2,
}[os.environ.get("DISP", "bf16")]
_DISP_NBYTES = {torch.bfloat16: 2, torch.float8_e4m3fn: 1}.get(_DISP_DT, 0.5)
# Geometry, same spelling as tools/ep_test.sh. Unset = the backend's tuned default,
# i.e. what ships; setting any one of them pins the geometry for the whole run.
_G = {k: (int(os.environ[k]) if os.environ.get(k) else None)
      for k in ("DBN", "DWPB", "CBN", "CWPB")}


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    ctypes.CDLL("libamdhip64.so").hipSetDevice(ctypes.c_int(rank))
    dev = torch.device("cuda", rank)

    obj = [cco.Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    M = max(SWEEP)
    vmm = 2 * (world * M * HIDDEN * 2 * 2 + (16 << 20)) + (256 << 20)
    comm = cco.Communicator.init(world, rank, obj[0], vmm)

    BACKEND = os.environ.get("MORI_V2_KERNEL_BACKEND", "hip")
    cfg = EpDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=HIDDEN,
        max_num_inp_token_per_rank=M,
        num_experts_per_rank=EPR,
        num_experts_per_token=TOPK,
        data_type=torch.bfloat16,
        dispatch_data_type=None if _DISP_DT is torch.bfloat16 else _DISP_DT,
        combine_data_type=None if _DISP_DT is torch.bfloat16 else torch.bfloat16,
        kernel_backend=BACKEND,
        dispatch_block_num=_G["DBN"],
        warp_num_per_block=_G["DWPB"],
        combine_block_num=_G["CBN"],
        combine_warp_num_per_block=_G["CWPB"],
    )
    op = EpDispatchCombineOp(cfg, comm)

    n_experts = world * EPR
    g = torch.Generator(device="cpu").manual_seed(1234 + rank)
    if _DISP_DT is torch.float4_e2m1fn_x2:  # no torch cast; generate packed bytes
        inp = torch.randint(
            0, 256, (M, HIDDEN // 2), generator=g, dtype=torch.uint8
        ).view(_DISP_DT).to(dev)
    else:
        inp = (
            torch.randn(M, HIDDEN, generator=g, dtype=torch.float32).to(_DISP_DT).to(dev)
        )
    wts = torch.rand(M, TOPK, generator=g, dtype=torch.float32).to(dev)
    idx = torch.stack(
        [torch.randperm(n_experts, generator=g)[:TOPK] for _ in range(M)]
    ).to(torch.int32).to(dev)

    if rank == 0:
        print(
            f"# EP [{op.backend_name}]  disp={_DISP_DT} comb=bf16  "
            f"world={world} hidden={HIDDEN} topk={TOPK} "
            f"epr={EPR} iters={ITERS}  variants "
            f"disp={sorted(op._kernels.dispatch)} comb={sorted(op._kernels.combine)}",
            flush=True,
        )

    def lockstep():
        torch.cuda.synchronize()
        dist.barrier()

    def time_it(fn):
        """Lock-step in WARMUP only, never inside the timed loop -- a
        synchronize+gloo barrier per iteration costs more than the kernel and
        would not be comparable with the FlyDSL bench. Combine's own cross-device
        barrier already keeps the ranks within one iteration of each other, which
        is the drift the epoch word can tolerate."""
        for _ in range(WARMUP):
            fn()
            lockstep()
        lockstep()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(ITERS):
            fn()
        e.record()
        torch.cuda.synchronize()
        dist.barrier()
        return s.elapsed_time(e) / ITERS * 1000

    for ct in SWEEP:
        i_, w_, x_ = inp[:ct], wts[:ct], idx[:ct]

        *_, total_t, routing = op.dispatch(i_, w_, None, x_, return_routing=True)
        torch.cuda.synchronize()
        total = int(total_t.cpu().item())
        # A separate buffer, so the staging copy is actually exercised.
        staged = op.combine_in_view()[:total].clone()
        lockstep()

        d_us = time_it(lambda: op.dispatch(i_, w_, None, x_))
        # Two combine numbers, because they answer different questions:
        #  - staged: the caller handed us a separate buffer, so the kernel copies
        #    it into the arena first. That is the full op.
        #  - inplace: the expert already wrote into combine_in_view, so staging is
        #    skipped. This is what the FlyDSL bench measures (its mock_fmoe writes
        #    straight into the staging view), so only this one is comparable.
        c_us = time_it(lambda: op.combine(staged, routing=routing))
        inplace = op.combine_in_view()[:total]
        c_us_ip = time_it(lambda: op.combine(inplace, routing=routing))

        # Bytes off this rank. The two legs differ whenever dispatch is narrower.
        d_bytes = total * HIDDEN * _DISP_NBYTES
        c_bytes = total * HIDDEN * 2
        d_bw = d_bytes / (1000**3) / (d_us / 1e6)
        c_bw = c_bytes / (1000**3) / (c_us / 1e6)
        ip_bw = c_bytes / (1000**3) / (c_us_ip / 1e6)
        got = torch.tensor([d_us, c_us, c_us_ip, float(total)], dtype=torch.float64)
        dist.all_reduce(got)
        if rank == 0:
            n = world
            print(
                f"  ct={ct:<5d} dispatch {got[0]/n:7.1f} us ({d_bw:6.1f} GB/s)  "
                f"combine {got[1]/n:7.1f} us ({c_bw:6.1f} GB/s)  "
                f"combine-inplace {got[2]/n:7.1f} us ({ip_bw:6.1f} GB/s)  "
                f"recv~{got[3]/n:.0f}",
                flush=True,
            )

    op.close()
    comm.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
