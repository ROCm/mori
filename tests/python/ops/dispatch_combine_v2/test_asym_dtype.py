#!/usr/bin/env python3
"""Asymmetric dtype test: fp8 dispatch + bf16 combine.

Models the fmoe use case where an expert op sits between dispatch and combine and
converts dtype: dispatch moves fp8 tokens, the (mock) expert dequants fp8->bf16
(identity), combine reduces bf16. Verifies dispatch and combine independently:

  * ASYM-DISPATCH(fp8): local-expert-count routing sum + recv tokens are the
    byte-exact source fp8 tokens (checked via the routing reverse map; every
    rank's input is regenerated from its per-rank seed, no collective needed).
  * ASYM-COMBINE(bf16): identity-expert telescoping — out == U * dequant(inp) * wt
    (U = distinct dest PEs per token), weights == U * wt.

    torchrun --nnodes=1 --nproc_per_node=4 --tee 3 ... test_asym_dtype.py
"""
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "python", "mori", "ops", "dispatch_combine_v2"))
sys.path.insert(0, os.path.join(_ROOT, "examples", "cco", "python"))
from cco_example_common import set_device, sync  # noqa: E402
from dispatch_combine_op import EpDispatchCombineConfig, EpDispatchCombineOp  # noqa: E402
from mori.cco import Communicator  # noqa: E402

FP8 = torch.float8_e4m3fn  # gfx1250/gfx950 OCP e4m3
HIDDEN = int(os.environ.get("HIDDEN", 512))  # 16 B aligned for both fp8 and bf16
K = int(os.environ.get("TOPK", 6))
EPR = int(os.environ.get("EPR", 8))
SWEEP = [int(x) for x in os.environ.get("SWEEP", "8,64,512").split(",")]


class Dist:
    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        torch.cuda.set_device(self.local_rank)

    def bcast_uid(self, uid):
        objs = [uid if self.rank == 0 else None]
        dist.broadcast_object_list(objs, src=0)
        return objs[0]

    def allreduce_sum(self, v):
        t = torch.tensor([v], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())

    def shutdown(self):
        if dist.is_initialized():
            dist.destroy_process_group()


def gen_inp_fp8(r, M):
    """Rank r's fp8 input tokens — reproducible from the per-rank seed."""
    g = torch.Generator(device="cpu").manual_seed(1234 + r)
    return torch.randn(M, HIDDEN, generator=g, dtype=torch.float32).to(FP8)


def main():
    d = Dist()
    rank, npes = d.rank, d.world
    set_device(d.local_rank)
    dev = torch.device("cuda", d.local_rank)
    M = max(SWEEP)
    num_experts = npes * EPR

    inp = gen_inp_fp8(rank, M).to(dev)  # fp8 dispatch input
    g2 = torch.Generator(device="cpu").manual_seed(4321 + rank)
    idx = torch.randint(0, num_experts, (M, K), generator=g2, dtype=torch.int32).to(dev)
    wts = torch.rand(M, K, generator=g2, dtype=torch.float32).to(dev)
    # every rank's fp8 input, global token id = r*M + tok (for the recv-value check)
    all_inp = torch.cat([gen_inp_fp8(r, M) for r in range(npes)], dim=0).float()

    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = d.bcast_uid(uid)
    win_bytes = npes * M * HIDDEN * 2 * 2 + (1 << 24)  # sized for bf16 (the larger)
    with Communicator.init(npes, rank, uid, per_rank_vmm=2 * win_bytes + (1 << 28)) as comm:
        cfg = EpDispatchCombineConfig(
            rank=rank,
            world_size=npes,
            hidden_dim=HIDDEN,
            max_num_inp_token_per_rank=M,
            num_experts_per_rank=EPR,
            num_experts_per_token=K,
            dispatch_data_type=FP8,  # fp8 dispatch
            combine_data_type=torch.bfloat16,  # bf16 combine
            combine_mode="gather",
        )
        op = EpDispatchCombineOp(cfg, comm)
        comm.barrier()

        for ct in SWEEP:
            recv_x, _w, _s, _i, total_recv_t, routing = op.dispatch(
                inp[:ct], wts[:ct], None, idx[:ct], return_routing=True
            )
            total_recv = int(total_recv_t.cpu().item())
            sync()
            comm.barrier()

            # ---- dispatch correctness ----
            lec_sum = int(op.local_expert_count().sum().cpu().item())
            sync()
            comm.barrier()
            ok_lec = d.allreduce_sum(lec_sum) == npes * ct * K
            # recv tokens are the byte-exact fp8 source tokens (via reverse map)
            tis = routing.disp_tok_id_to_src_tok_id_local[:total_recv].cpu().long()
            recv_f = recv_x[:total_recv].float().cpu()  # fp8 -> f32 (exact)
            exp_recv = all_inp[tis]  # source token per recv slot
            ok_disp = bool(torch.equal(recv_f, exp_recv))
            errs_disp = d.allreduce_sum(0 if (ok_lec and ok_disp) else 1)

            # ---- mock fmoe: fp8 -> bf16 (identity dequant) ----
            expert_out = recv_x.to(torch.bfloat16)

            # ---- combine ----
            out, out_w = op.combine(expert_out, routing=routing)
            sync()
            comm.barrier()

            # ---- combine correctness (identity-expert telescoping) ----
            idx_c = idx[:ct].cpu().numpy()
            U = np.array(
                [len({int(idx_c[t, j]) // EPR for j in range(K)}) for t in range(ct)]
            )
            exp = (
                torch.from_numpy(U).view(ct, 1).float() * inp[:ct].float().cpu()
            ).to(torch.bfloat16)
            exp_w = torch.from_numpy(U).view(ct, 1).float() * wts[:ct].float().cpu()
            ok_comb = torch.allclose(out.float().cpu(), exp.float(), atol=3e-1, rtol=1e-1)
            ok_w = torch.allclose(out_w.cpu(), exp_w, atol=2e-3, rtol=2e-3)
            # dtype sanity: dispatch recv is fp8, combine out is bf16
            ok_dt = recv_x.dtype == FP8 and out.dtype == torch.bfloat16
            errs_comb = d.allreduce_sum(0 if (ok_comb and ok_w and ok_dt) else 1)

            if rank == 0:
                print(
                    f"# ASYM-DISPATCH(fp8) ct={ct}: {'PASS' if errs_disp == 0 else 'FAIL'} "
                    f"(LEC={'ok' if ok_lec else 'BAD'} recv-values={'ok' if ok_disp else 'BAD'}; "
                    f"recv={total_recv})",
                    flush=True,
                )
                print(
                    f"# ASYM-COMBINE(bf16) ct={ct}: {'PASS' if errs_comb == 0 else 'FAIL'} "
                    f"(hidden={'ok' if ok_comb else 'BAD'} wts={'ok' if ok_w else 'BAD'} "
                    f"dtype={'ok' if ok_dt else 'BAD'})",
                    flush=True,
                )
    d.shutdown()


if __name__ == "__main__":
    main()
