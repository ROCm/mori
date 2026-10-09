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
"""Regression test (#715): dispatch(return_routing=True, snapshot=True) handles
own their routing, so several can be held at once -- one per MoE layer, then the
backward in reverse -- and each still combines (and, on flydsl, replays) correctly
after later dispatches on the same op. Also: a handle combines more than once, and
the default (snapshot=False) handle still aliases the op's buffers.

Identity expert, so combine[t] == U[t] * input[t] (U = distinct dest PEs of t).

    torchrun --standalone --nproc_per_node=2 test_routing_snapshot.py
    BACKENDS=hip torchrun --standalone --nproc_per_node=2 test_routing_snapshot.py
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
M = int(os.environ.get("M", 64))
BACKENDS = os.environ.get("BACKENDS", "flydsl,hip").split(",")


def make_inputs(rank, world, dev):
    g = torch.Generator(device="cpu").manual_seed(4321 + rank)
    n_experts = world * EPR

    def routing():
        return torch.stack(
            [torch.randperm(n_experts, generator=g)[:TOPK] for _ in range(M)]
        ).to(torch.int32)

    def tokens():
        return torch.randn(M, HIDDEN, generator=g).to(torch.bfloat16)

    def n_dest(idx):
        return torch.tensor([len({e // EPR for e in row}) for row in idx.tolist()])

    idx_a, idx_b = routing(), routing()
    x_a, x_b, x_c = tokens(), tokens(), tokens()
    return dict(
        idx_a=idx_a.to(dev),
        idx_b=idx_b.to(dev),
        x_a=x_a.to(dev),
        x_b=x_b.to(dev),
        x_c=x_c.to(dev),
        w=torch.ones(M, TOPK, device=dev),
        u_a=n_dest(idx_a).to(dev)[:, None].float(),
        u_b=n_dest(idx_b).to(dev)[:, None].float(),
    )


def run_backend(name, comm, rank, world, d):
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
    checks = {}

    def sync():
        torch.cuda.synchronize()
        comm.barrier()

    def dispatch(x, idx, **kw):
        out, _, _, _, total, *handle = op.dispatch(x, d["w"], None, idx, **kw)
        sync()
        n = int(total.item())
        recv = out[:n].clone()
        sync()
        return recv, n, (handle[0] if handle else None)

    def combine_ok(recv, handle, x, u):
        out, _ = op.combine(recv, None, None, routing=handle)
        sync()
        ok = torch.allclose(out.float(), u * x.float(), atol=2e-2, rtol=2e-2)
        sync()
        return ok

    try:
        op.dispatch(d["x_a"], d["w"], None, d["idx_a"], snapshot=True)
        checks["snapshot_needs_return_routing"] = False
    except ValueError:
        checks["snapshot_needs_return_routing"] = True

    # Default: the handle aliases the op's buffers (valid until the next
    # return_routing dispatch), and combines correctly within that lifetime.
    recv_0, _, h0 = dispatch(d["x_a"], d["idx_a"], return_routing=True)
    checks["default_aliases_op"] = (
        h0.disp_dest_tok_id_map.data_ptr() == op.routing_dest_map.data_ptr()
        and h0.total_recv_token_num.data_ptr() == op.total_recv.data_ptr()
    )
    checks["default_combine"] = combine_ok(recv_0, h0, d["x_a"], d["u_a"])

    # Two snapshot handles from back-to-back dispatches, B after A.
    recv_a, n_a, ha = dispatch(d["x_a"], d["idx_a"], return_routing=True, snapshot=True)
    recv_b, n_b, hb = dispatch(d["x_b"], d["idx_b"], return_routing=True, snapshot=True)
    ptrs = {
        t.data_ptr()
        for t in (
            ha.disp_dest_tok_id_map,
            hb.disp_dest_tok_id_map,
            op.routing_dest_map,
            op.token_dest_map,
        )
    }
    counts = {
        t.data_ptr()
        for t in (ha.total_recv_token_num, hb.total_recv_token_num, op.total_recv)
    }
    checks["snapshot_owns_buffers"] = len(ptrs) == 4 and len(counts) == 3
    checks["older_handle_combine"] = combine_ok(recv_a, ha, d["x_a"], d["u_a"])
    checks["newer_handle_combine"] = combine_ok(recv_b, hb, d["x_b"], d["u_b"])
    # Again with A, after B's combine restaged the arena: a combine that had
    # consumed A's count would skip restaging A's tokens.
    checks["older_handle_second_combine"] = combine_ok(recv_a, ha, d["x_a"], d["u_a"])
    checks["counts_survive_combine"] = (
        int(ha.total_recv_token_num.item()) == n_a
        and int(hb.total_recv_token_num.item()) == n_b
    )

    if "replay" in op.capabilities:
        # Replay the OLDER handle with a new payload: same routing, new tokens.
        recv_c, n_c, _ = dispatch(d["x_c"], d["idx_a"], routing=ha)
        checks["older_handle_replay_count"] = n_c == n_a
        checks["older_handle_replay_combine"] = combine_ok(
            recv_c, ha, d["x_c"], d["u_a"]
        )

    op.close()
    failures = 0
    for check, ok in checks.items():
        bad = torch.tensor([0 if ok else 1], dtype=torch.int32)
        dist.all_reduce(bad)
        failures += int(bad.item() != 0)
        if rank == 0:
            print(
                f"# SNAPSHOT[{name}] {check}: {'PASS' if bad.item() == 0 else 'FAIL'}",
                flush=True,
            )
    return failures


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())

    # Inputs before the communicator: comm_create leaves a HIP error latched that
    # the next torch GPU call would report (see test_ep_backend_parity.py).
    d = make_inputs(rank, world, dev)
    torch.cuda.synchronize()

    obj = [cco.Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    vmm = 2 * (world * M * HIDDEN * 2 * 2 + (16 << 20)) + (1 << 30)
    comm = cco.Communicator.init(world, rank, obj[0], vmm)

    available = EpDispatchCombineOp.available_backends()
    failures = 0
    for name in BACKENDS:
        if name not in available:
            if rank == 0:
                print(f"# SNAPSHOT[{name}]: SKIP (backend not available)", flush=True)
            continue
        failures += run_backend(name, comm, rank, world, d)

    if rank == 0:
        print(f"# total snapshot failures: {failures}", flush=True)
    comm.destroy()
    dist.destroy_process_group()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
