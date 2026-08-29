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
"""Backend parity: the FlyDSL and C++/JIT kernels must agree, element for element.

The same problem is run through both backends in one process, on the same input,
and the outputs are compared directly. That is a stronger check than each backend
matching the analytic reference separately: it catches a difference the reference
tolerates (a routing tie broken the other way, a weight forwarded from the wrong
slot) and it is the property the backend switch actually promises.

    torchrun --standalone --nproc_per_node=8 test_ep_backend_parity.py
"""

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
# The one region the backends lay out differently (HIP pads the row, FlyDSL does
# not); recv_scales() is supposed to hide that. 224 is not 128-aligned on purpose,
# so the pitches really differ -- an aligned row would agree by accident.
SCALE_DIM = int(os.environ.get("SCALE_DIM", 224))


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
        scale_dim=SCALE_DIM,
        scale_type_size=1 if SCALE_DIM else 0,
    )
    op = EpDispatchCombineOp(cfg, comm)
    assert op.backend_name == name, f"{op.backend_name} != {name}"
    return op


def run_once(op, comm, ct, inp, wts, idx, scales):
    """One dispatch+combine at `ct` tokens. The op is reused across token counts
    on purpose -- that is what exercises the variant table's _pick."""
    if True:
        _, _, out_s, _, total_recv_t, routing = op.dispatch(
            inp[:ct],
            wts[:ct],
            scales[:ct] if scales is not None else None,
            idx[:ct],
            return_routing=True,
        )
        torch.cuda.synchronize()
        comm.barrier()
        # Read it HERE: it is dispatch's return value. Reading after combine
        # measures each backend's reset policy instead -- flydsl's combine kernel
        # zeroes the op's live counter, the C++ one zeroes only the handle's copy.
        total = int(total_recv_t.cpu().item())
        # Cloned before combine, and only the live rows: this is the view whose
        # PITCH differs between the backends, so it has to be read the way a
        # caller reads it (through the returned tensor) rather than off the arena.
        # With the reverse map: the backends do not agree on WHICH slot a token
        # lands in, so a row-by-row compare would fail on order, not on values.
        recv_s = out_s[:total].clone() if (out_s is not None and total) else None
        tis = (
            routing.disp_tok_id_to_src_tok_id_local[:total].clone()
            if recv_s is not None
            else None
        )
        # Identity expert: hand the received tokens straight back.
        out, out_wts = op.combine(op.combine_in_view(), wts[:ct], routing=routing)
        torch.cuda.synchronize()
        comm.barrier()
        return out.clone(), out_wts.clone(), total, recv_s, tis


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
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

    # One recognisable dword per token: a row read at the wrong pitch, or landed in
    # the wrong slot, then shows up as a mismatch rather than as noise.
    scales = None
    if SCALE_DIM:
        n_i32 = (SCALE_DIM + 3) // 4
        scales = (
            (torch.arange(M, device=dev) + rank * 100003)
            .view(M, 1)
            .expand(M, n_i32)
            .contiguous()
            .to(torch.int32)
            .view(torch.uint8)
        )

    # Six ops are built and closed in sequence (2 backends x len(SWEEP)),
    # so the reservation has to survive the repeated alloc/free, not just
    # hold one arena.
    vmm = 4 * (world * M * HIDDEN * 2 * 2 + (16 << 20)) + (2 << 30)
    comm = cco.Communicator.init(world, rank, obj[0], vmm)

    if rank == 0:
        print(f"# backends: {EpDispatchCombineOp.available_backends()}", flush=True)

    # Both ops built once and kept alive: rebuilding per token count churns the
    # cco VMM reservation for no benefit, and reuse is what a real caller does.
    ops = {n: make_op(n, comm, rank, world, M) for n in ("flydsl", "hip")}

    failures = 0
    for ct in SWEEP:
        a_out, a_w, a_recv, a_s, a_tis = run_once(
            ops["flydsl"], comm, ct, inp, wts, idx, scales
        )
        b_out, b_w, b_recv, b_s, b_tis = run_once(
            ops["hip"], comm, ct, inp, wts, idx, scales
        )

        # Routing is deterministic given the same indices, so the recv counts must
        # match exactly; the payload is the same bf16 sum in the same order.
        ok_recv = a_recv == b_recv
        ok_out = torch.allclose(
            a_out.to(torch.float32), b_out.to(torch.float32), atol=2e-2, rtol=2e-2
        )
        ok_w = torch.allclose(a_w, b_w, atol=2e-3, rtol=2e-3)
        # Byte-exact (opaque payload), in SOURCE-token order via the reverse map --
        # the pitches differ underneath, which is the point of checking at all.
        ok_s = True
        if a_s is not None and b_s is not None:
            ok_s = a_s.shape == b_s.shape
            if ok_s:
                a_ord = torch.argsort(a_tis.cpu(), stable=True)
                b_ord = torch.argsort(b_tis.cpu(), stable=True)
                ok_s = torch.equal(a_tis.cpu()[a_ord], b_tis.cpu()[b_ord]) and (
                    torch.equal(a_s.cpu()[a_ord], b_s.cpu()[b_ord])
                )
        ok = ok_recv and ok_out and ok_w and ok_s
        failures += 0 if ok else 1
        if not ok:
            print(
                f"[rank {rank}] ct={ct}: PARITY FAIL "
                f"recv={a_recv}/{b_recv} "
                f"out_max_diff={(a_out.to(torch.float32) - b_out.to(torch.float32)).abs().max():.4f} "
                f"wts_max_diff={(a_w - b_w).abs().max():.5f} "
                # Which check failed: a scale-only failure has a different cause.
                f"[recv={'ok' if ok_recv else 'BAD'} out={'ok' if ok_out else 'BAD'} "
                f"wts={'ok' if ok_w else 'BAD'} scales={'ok' if ok_s else 'BAD'}]",
                flush=True,
            )
        elif rank == 0:
            sc = (
                f", scales {a_s.shape[1]} dwords byte-exact across pitches"
                if a_s is not None
                else ""
            )
            print(
                f"# PARITY ct={ct}: PASS (recv={a_recv}, flydsl == hip{sc})",
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
