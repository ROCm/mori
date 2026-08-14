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
"""Backend parity: the FlyDSL, C++/JIT, and hybrid kernels must agree.

The same problem is run through all backends in one process, on the same input,
and the outputs are compared directly. That is a stronger check than each backend
matching the analytic reference separately: it catches a difference the reference
tolerates (a routing tie broken the other way, a weight forwarded from the wrong
slot) and it is the property the backend switch actually promises.

    torchrun --standalone --nproc_per_node=8 test_ep_backend_parity.py
"""

import ctypes
import os
import sys

import torch
import torch.distributed as dist

import mori.cco as cco
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp

HIDDEN = int(os.environ.get("HIDDEN", 2048))
TOPK = int(os.environ.get("TOPK", 4))
EPR = int(os.environ.get("EPR", 4))
SWEEP = [int(x) for x in os.environ.get("SWEEP", "8,64,512").split(",")]


def make_op(name, comm, rank, world, M):
    cfg = EpDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=HIDDEN,
        max_num_inp_token_per_rank=M,
        num_experts_per_rank=EPR,
        num_experts_per_token=TOPK,
        data_type=torch.bfloat16,
        kernel_backend=name,
    )
    op = EpDispatchCombineOp(cfg, comm)
    assert op.backend_name == name, f"{op.backend_name} != {name}"
    return op


def run_once(op, comm, ct, inp, wts, idx):
    """One dispatch+combine at `ct` tokens. The op is reused across token counts
    on purpose -- that is what exercises the variant table's _pick."""
    if True:
        *_, total_recv_t, routing = op.dispatch(
            inp[:ct], wts[:ct], None, idx[:ct], return_routing=True
        )
        torch.cuda.synchronize()
        comm.barrier()
        # Read it HERE: it is dispatch's return value. Reading after combine
        # measures each backend's reset policy instead -- flydsl's combine kernel
        # zeroes the op's live counter, the C++ one zeroes only the handle's copy.
        total = int(total_recv_t.cpu().item())
        # Identity expert: hand the received tokens straight back.
        out, out_wts = op.combine(op.combine_in_view(), wts[:ct], routing=routing)
        torch.cuda.synchronize()
        comm.barrier()
        return out.clone(), out_wts.clone(), total


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    ctypes.CDLL("libamdhip64.so").hipSetDevice(ctypes.c_int(rank))
    dev = torch.device("cuda", rank)

    obj = [cco.Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    M = max(SWEEP)

    # Inputs BEFORE the communicator, and not as a style choice: comm_create leaves
    # a HIP error latched, and torch reports whatever is latched on its next GPU
    # call, so a .to(dev) after Communicator.init dies with someone else's error.
    # The bug is cco's and the fix belongs next to whichever tolerated call sets the
    # flag, as src/application/memory/symmetric_memory.cpp already does with
    # (void)hipGetLastError().
    n_experts = world * EPR
    g = torch.Generator(device="cpu").manual_seed(1234 + rank)
    inp = (
        torch.randn(M, HIDDEN, generator=g, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(dev)
    )
    wts = torch.rand(M, TOPK, generator=g, dtype=torch.float32).to(dev)
    idx = (
        torch.stack([torch.randperm(n_experts, generator=g)[:TOPK] for _ in range(M)])
        .to(torch.int32)
        .to(dev)
    )

    # Six ops are built and closed in sequence (2 backends x len(SWEEP)),
    # so the reservation has to survive the repeated alloc/free, not just
    # hold one arena.
    vmm = 4 * (world * M * HIDDEN * 2 * 2 + (16 << 20)) + (2 << 30)
    comm = cco.Communicator.init(world, rank, obj[0], vmm)

    if rank == 0:
        print(f"# backends: {EpDispatchCombineOp.available_backends()}", flush=True)

    # All ops built once and kept alive: rebuilding per token count churns the
    # cco VMM reservation for no benefit, and reuse is what a real caller does.
    backends = ("flydsl", "hip", "hybrid")
    ops = {n: make_op(n, comm, rank, world, M) for n in backends}

    failures = 0
    for ct in SWEEP:
        results = {
            n: run_once(ops[n], comm, ct, inp, wts, idx) for n in backends
        }
        ref_out, ref_w, ref_recv = results["flydsl"]

        for other in ("hip", "hybrid"):
            b_out, b_w, b_recv = results[other]
            ok_recv = ref_recv == b_recv
            ok_out = torch.allclose(
                ref_out.to(torch.float32),
                b_out.to(torch.float32),
                atol=2e-2,
                rtol=2e-2,
            )
            ok_w = torch.allclose(ref_w, b_w, atol=2e-3, rtol=2e-3)
            ok = ok_recv and ok_out and ok_w
            failures += 0 if ok else 1
            if not ok:
                print(
                    f"[rank {rank}] ct={ct}: PARITY FAIL flydsl vs {other} "
                    f"recv={ref_recv}/{b_recv} "
                    f"out_max_diff={(ref_out.to(torch.float32) - b_out.to(torch.float32)).abs().max():.4f} "
                    f"wts_max_diff={(ref_w - b_w).abs().max():.5f}",
                    flush=True,
                )
            elif rank == 0:
                print(
                    f"# PARITY ct={ct}: PASS (recv={ref_recv}, flydsl == {other})",
                    flush=True,
                )

    counts = torch.tensor([failures], dtype=torch.int32)
    dist.all_reduce(counts)
    if rank == 0:
        print(f"# total parity failures across ranks: {int(counts.item())}", flush=True)

    for op in ops.values():
        op.close()
    comm.destroy()
    dist.destroy_process_group()
    sys.exit(1 if int(counts.item()) else 0)


if __name__ == "__main__":
    main()
