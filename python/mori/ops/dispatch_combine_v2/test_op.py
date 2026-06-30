#!/usr/bin/env python3
"""EP8 correctness test for the EpDispatchCombineOp host op-layer.

Identity expert (recv tokens are combined unchanged), so combine[t] == U[t] *
input[t] (U = #unique dest PEs) and out_weights[t] == U[t] * wts[t].

    torchrun --standalone --nproc_per_node=8 test_op.py
"""
import os
import sys

import numpy as np
import torch

from mori.cco import Communicator

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "examples", "cco", "python"))
from cco_example_common import set_device, sync  # noqa: E402
from dist_common import Dist  # noqa: E402
from dispatch_combine_op import EpDispatchCombineConfig, EpDispatchCombineOp  # noqa: E402

HIDDEN = int(os.environ.get("HIDDEN", 7168))
K = int(os.environ.get("TOPK", 8))
EPR = int(os.environ.get("EPR", 32))
DTYPE = {"bf16": torch.bfloat16, "f32": torch.float32}[os.environ.get("DTYPE", "bf16")]
SWEEP = [int(x) for x in os.environ.get("SWEEP", "128,512").split(",")]


def main():
    d = Dist()
    rank, npes = d.rank, d.world
    set_device(d.local_rank)
    dev = torch.device("cuda", d.local_rank)
    M = max(SWEEP)
    num_experts = npes * EPR

    g = torch.Generator(device="cpu").manual_seed(1234 + rank)
    inp = torch.randn(M, HIDDEN, generator=g, dtype=torch.float32).to(DTYPE).to(dev)
    idx = torch.randint(0, num_experts, (M, K), generator=g, dtype=torch.int32).to(dev)
    wts = torch.rand(M, K, generator=g, dtype=torch.float32).to(dev)

    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = d.bcast_uid(uid)
    win_bytes = npes * M * HIDDEN * _DT() * 2 + (1 << 24)
    with Communicator.init(npes, rank, uid, per_rank_vmm=2 * win_bytes + (1 << 28)) as comm:
        cfg = EpDispatchCombineConfig(
            rank=rank, world_size=npes, hidden_dim=HIDDEN, max_num_inp_token_per_rank=M,
            num_experts_per_rank=EPR, num_experts_per_token=K, data_type=DTYPE)
        op = EpDispatchCombineOp(cfg, comm)
        comm.barrier()

        for ct in SWEEP:
            recv_x, total_recv, routing = op.dispatch(inp[:ct], wts[:ct], idx[:ct])
            sync(); comm.barrier()
            # identity expert: recv_x already holds the dispatched tokens.
            out, out_w = op.combine(recv_x, routing)
            sync(); comm.barrier()

            idx_c = idx[:ct].cpu().numpy()
            U = np.array([len({int(idx_c[t, j]) // EPR for j in range(K)}) for t in range(ct)])
            exp = (torch.from_numpy(U).view(ct, 1).float() * inp[:ct].float().cpu()).to(DTYPE)
            exp_w = torch.from_numpy(U).view(ct, 1).float() * wts[:ct].float().cpu()
            ok = torch.allclose(out.float().cpu(), exp.float(), atol=2e-2, rtol=2e-2)
            ok_w = torch.allclose(out_w.cpu(), exp_w, atol=2e-3, rtol=2e-3)
            errs = d.allreduce_sum(0 if (ok and ok_w) else 1)
            if rank == 0:
                print(f"# OP ct={ct}: {'PASS' if errs == 0 else 'FAIL'} "
                      f"(hidden={'ok' if ok else 'BAD'} wts={'ok' if ok_w else 'BAD'}; "
                      f"recv={total_recv}, U in [{U.min()},{U.max()}])", flush=True)
    d.shutdown()


def _DT():
    return {torch.bfloat16: 2, torch.float32: 4}[DTYPE]


if __name__ == "__main__":
    main()
