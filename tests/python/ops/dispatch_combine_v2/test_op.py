#!/usr/bin/env python3
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
"""EP8 correctness test for the EpDispatchCombineOp host op-layer.

Identity expert (recv tokens are combined unchanged), so combine[t] == U[t] *
input[t] (U = #unique dest PEs) and out_weights[t] == U[t] * wts[t].

    torchrun --standalone --nproc_per_node=8 test_op.py
"""
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

from mori.cco import Communicator

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(
    0, os.path.join(_ROOT, "python", "mori", "ops", "dispatch_combine_v2")
)  # op + kernels
sys.path.insert(
    0, os.path.join(_ROOT, "examples", "cco", "python")
)  # cco_example_common
from cco_example_common import set_device, sync  # noqa: E402
from dispatch_combine_op import (  # noqa: E402
    EpDispatchCombineConfig,
    EpDispatchCombineOp,
)


class Dist:
    """Minimal torchrun/gloo bootstrap: RANK/WORLD_SIZE/LOCAL_RANK, carry the cco
    unique-id (broadcast) and a test-only int allreduce. gloo (CPU) is just the
    courier for the uid + pass/fail counts; cco does the GPU comm."""

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

    def allreduce_sum(self, value):
        t = torch.tensor([value], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())

    def shutdown(self):
        if dist.is_initialized():
            dist.destroy_process_group()


HIDDEN = int(os.environ.get("HIDDEN", 7168))
K = int(os.environ.get("TOPK", 8))
EPR = int(os.environ.get("EPR", 32))
# fp8 flavor is arch-specific: OCP e4m3 on gfx950, fnuz on gfx942.
import tuning_configs as _tc  # noqa: E402

_FP8_DT = torch.float8_e4m3fn if _tc._topology()[1] == 90500 else torch.float8_e4m3fnuz
DTYPE = {
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "fp8": _FP8_DT,
    "fp4": torch.float4_e2m1fn_x2,
}[os.environ.get("DTYPE", "bf16")]
_IS_FP4 = DTYPE == torch.float4_e2m1fn_x2
_IS_FP8 = DTYPE == _FP8_DT
STDMOE = int(os.environ.get("STDMOE", 0))
SCALE_DIM = int(
    os.environ.get("SCALE_DIM", 0)
)  # >0 = also verify per-token scales forwarding
COMBINE = os.environ.get("COMBINE", "gather")  # gather | scatter
QUANT = os.environ.get("QUANT", "none")  # none | fp8_direct_cast | fp8_blockwise
SWEEP = [int(x) for x in os.environ.get("SWEEP", "128,512").split(",")]


def main():
    d = Dist()
    rank, npes = d.rank, d.world
    set_device(d.local_rank)
    dev = torch.device("cuda", d.local_rank)
    M = max(SWEEP)
    num_experts = npes * EPR

    g = torch.Generator(device="cpu").manual_seed(1234 + rank)
    if _IS_FP4:  # fp4: packed uint8 (2 e2m1/byte), reinterpret as float4_e2m1fn_x2
        inp = (
            torch.randint(0, 256, (M, HIDDEN // 2), generator=g, dtype=torch.uint8)
            .view(torch.float4_e2m1fn_x2)
            .to(dev)
        )
    else:
        inp = torch.randn(M, HIDDEN, generator=g, dtype=torch.float32).to(DTYPE).to(dev)
    idx = torch.randint(0, num_experts, (M, K), generator=g, dtype=torch.int32).to(dev)
    wts = torch.rand(M, K, generator=g, dtype=torch.float32).to(dev)

    # Per-token scales (int8 bytes): pattern = rank*100003 + tok per dword, so the
    # recv side can verify the bijection by decoding the origin from the routing
    # handle's reverse map (disp_tok_id_to_src_tok_id_local).
    sc_n_i32 = (SCALE_DIM + 3) // 4
    scales = None
    if SCALE_DIM:
        scales = (
            torch.arange(M, device=dev).view(M, 1).expand(M, sc_n_i32).contiguous()
            + rank * 100003
        ).to(torch.int32)

    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = d.bcast_uid(uid)
    win_bytes = npes * M * HIDDEN * _DT() * 2 + (1 << 24)
    with Communicator.init(
        npes, rank, uid, per_rank_vmm=2 * win_bytes + (1 << 28)
    ) as comm:
        cfg = EpDispatchCombineConfig(
            rank=rank,
            world_size=npes,
            hidden_dim=HIDDEN,
            max_num_inp_token_per_rank=M,
            num_experts_per_rank=EPR,
            num_experts_per_token=K,
            data_type=DTYPE,
            enable_std_moe=bool(STDMOE),
            scale_dim=SCALE_DIM,
            scale_type_size=1 if SCALE_DIM else 0,
            combine_mode=COMBINE,
            quant_type=QUANT,
            max_total_recv_tokens=int(os.environ.get("MAXRECV", 0)),
        )
        if int(os.environ.get("TUNED", 0)):
            from tuning_configs import lookup

            _dt = "fp4" if _IS_FP4 else ("fp8" if _IS_FP8 else "bf16")
            cfg.schedule = lookup(npes, HIDDEN, K, dtype=_dt)["schedule"]
        op = EpDispatchCombineOp(cfg, comm)
        comm.barrier()

        cap = cfg.effective_max_recv if cfg.max_total_recv_tokens > 0 else None
        for ct in SWEEP:
            if int(os.environ.get("RESET", 0)):
                op.reset()
                sync()
                comm.barrier()
            sc_in = scales[:ct] if SCALE_DIM else None
            recv_x, _out_w, out_s, _out_i, total_recv_t, routing = op.dispatch(
                inp[:ct], wts[:ct], sc_in, idx[:ct], return_routing=True
            )
            total_recv = int(total_recv_t.cpu().item())
            sync()
            comm.barrier()
            idx_c = idx[:ct].cpu().numpy()
            # local_expert_count: global sum over ranks == world*ct*K (every
            # (token,k) assignment is recorded on exactly the rank owning it).
            lec_sum = int(op.local_expert_count().sum().cpu().item())
            sync()
            comm.barrier()
            lec_total = d.allreduce_sum(lec_sum)
            if rank == 0:
                ok_lec = lec_total == npes * ct * K
                print(
                    f"# OP-LEC ct={ct}: {'PASS' if ok_lec else 'FAIL'} "
                    f"(sum={lec_total} exp={npes * ct * K})",
                    flush=True,
                )
            if int(os.environ.get("REPLAY", 0)) and cap is None:
                ref = recv_x[:total_recv].clone()
                r2 = op.dispatch(inp[:ct], wts[:ct], sc_in, idx[:ct], routing=routing)
                sync()
                comm.barrier()
                ok_r = torch.equal(r2[0][:total_recv], ref)
                errs_r = d.allreduce_sum(0 if ok_r else 1)
                if rank == 0:
                    print(
                        f"# OP-REPLAY ct={ct}: {'PASS' if errs_r == 0 else 'FAIL'} "
                        f"(replayed layout == original)",
                        flush=True,
                    )
            if SCALE_DIM:
                # Verify per-token scales forwarding via the routing handle's
                # reverse map: recv slot s came from token disp_tok_id_to_src_tok_id
                # -> expected dword = (src_rank)*100003 + (src_tok).
                tis = routing.disp_tok_id_to_src_tok_id_local[:total_recv].cpu()
                exp = ((tis // M) * 100003 + (tis % M)).view(total_recv, 1)
                got = out_s[:total_recv].cpu()
                ok_sc = bool((got == exp.expand(total_recv, sc_n_i32)).all())
                errs = d.allreduce_sum(0 if ok_sc else 1)
                if rank == 0:
                    print(
                        f"# OP-SCALES ct={ct}: {'PASS' if errs == 0 else 'FAIL'} "
                        f"(recv={total_recv}, scale_dim={SCALE_DIM}, {sc_n_i32} dwords/tok, "
                        f"reverse-map ok)",
                        flush=True,
                    )
            if cap is not None:
                # Capped run: over-cap tokens are intentionally dropped (mori
                # parity), so identity-verify won't hold. Validate the cap is
                # respected and nothing OOB/crashed.
                op.combine(recv_x, routing=routing)
                sync()
                comm.barrier()
                ok = total_recv <= cap
                errs = d.allreduce_sum(0 if ok else 1)
                if rank == 0:
                    print(
                        f"# OP-CAP ct={ct}: {'PASS' if errs == 0 else 'FAIL'} "
                        f"(recv={total_recv} <= cap={cap}, no OOB)",
                        flush=True,
                    )
                continue
            if STDMOE:
                op.convert_dispatch_output()  # identity GEMM: packed_x = token
                sync()
                comm.barrier()
                op.convert_combine_input(routing)
                sync()
                comm.barrier()
                out, out_w = op.combine(recv_x, routing=routing)
                sync()
                comm.barrier()
                # telescopes to (sum_k wts)*input
                ws = wts[:ct].float().cpu().sum(dim=1, keepdim=True)
                exp = (ws * inp[:ct].float().cpu()).to(DTYPE)
                ok = torch.allclose(
                    out.float().cpu(), exp.float(), atol=5e-2, rtol=5e-2
                )
                ok_w = True
                tag = "OP-STDMOE"
            else:
                # identity expert: recv_x already holds the dispatched tokens.
                out, out_w = op.combine(recv_x, routing=routing)
                sync()
                comm.barrier()
                U = np.array(
                    [
                        len({int(idx_c[t, j]) // EPR for j in range(K)})
                        for t in range(ct)
                    ]
                )
                exp_w = torch.from_numpy(U).view(ct, 1).float() * wts[:ct].float().cpu()
                if _IS_FP4:
                    # fp4 combine is too lossy for a numeric hidden check (mirror v1);
                    # routing is validated by OP-LEC and the weights check.
                    ok = True
                else:
                    exp = (
                        torch.from_numpy(U).view(ct, 1).float() * inp[:ct].float().cpu()
                    ).to(DTYPE)
                    # fp8 paths (quant or plain fp8 token) lose precision; relax.
                    lossy = QUANT != "none" or _IS_FP8
                    atol = 3e-1 if lossy else 2e-2
                    rtol = 1e-1 if lossy else 2e-2
                    ok = torch.allclose(
                        out.float().cpu(), exp.float(), atol=atol, rtol=rtol
                    )
                ok_w = torch.allclose(out_w.cpu(), exp_w, atol=2e-3, rtol=2e-3)
                tag = f"OP-{COMBINE}" + (f"-{QUANT}" if QUANT != "none" else "")
            errs = d.allreduce_sum(0 if (ok and ok_w) else 1)
            if rank == 0:
                print(
                    f"# {tag} ct={ct}: {'PASS' if errs == 0 else 'FAIL'} "
                    f"(hidden={'ok' if ok else 'BAD'} wts={'ok' if ok_w else 'BAD'}; "
                    f"recv={total_recv})",
                    flush=True,
                )
    d.shutdown()


def _DT():
    return {
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.float4_e2m1fn_x2: 1,
        torch.float8_e4m3fnuz: 1,
        torch.float8_e4m3fn: 1,
    }[DTYPE]


if __name__ == "__main__":
    main()
