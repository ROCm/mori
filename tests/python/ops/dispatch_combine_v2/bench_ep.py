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
"""Latency of the EPv2 intranode op, at the op API rather than the raw kernels.

Every backend is driven through EpDispatchCombineOp, so one script covers all of
them and BACKENDS=flydsl,hip compares them in a single process on one input.

dispatch and combine ALTERNATE, one pair per iteration, because that is the order
a layer runs them in. Timing N dispatches and then N combines instead leaves
combine's staging copy unmeasured: combine reads totalRecvTokenNum to size that
copy and clears it on the way out (ep_intranode_kernel.hpp:343, and
intranode_kernels.py:1209 for flydsl), so with no dispatch in between only the
first combine of the loop copies anything.

dispatch is called with return_routing=True, the way a serving stack calls it, so
the routing handle is inside the measured window. Each leg gets its own cuda event
pair; the loop does not sync, since a synchronize costs 5-20 us against a 30 us
kernel. Means over ITERS, not percentiles -- at these sizes the tail is the
machine, and a mean over enough iterations is the number that composes.

Every point is correctness-gated first: an identity expert makes combine[t] equal
U[t]*input[t], where U[t] is how many distinct PEs token t routed to. A geometry
that computes garbage never gets a bandwidth number. That check is deliberately
one invariant, not a matrix -- test_op.py owns dtypes, quant, scatter, StdMoE,
scales and recv-cap, across both backends.

    torchrun --standalone --nproc_per_node=8 bench_ep.py
    BACKENDS=flydsl,hip SWEEP=512,4096 ITERS=200 torchrun ... bench_ep.py
"""

import os
import sys

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
# Comma-separated; MORI_V2_KERNEL_BACKEND still works for a single backend.
BACKENDS = [
    b
    for b in os.environ.get(
        "BACKENDS", os.environ.get("MORI_V2_KERNEL_BACKEND", "hip")
    ).split(",")
    if b
]
MODES = os.environ.get("MODES", "eager,graph").split(",")
# "inplace": the expert already wrote into the staging view, so combine elides the
# copy -- what a real pipeline does. "staged": a separate buffer, copy included.
COMBINE_IN = os.environ.get("COMBINE_IN", "inplace")
CHECK = int(os.environ.get("CHECK", 1))
SDMA_TOKEN_COPY = int(os.environ.get("SDMA_TOKEN_COPY", 0))
SDMA_QUEUES = int(os.environ.get("SDMA_QUEUES", 8))
# What dispatch transports; combine is always bf16, so anything else is asymmetric.
_DISP_DT = {
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
    "fp4": torch.float4_e2m1fn_x2,
}[os.environ.get("DISP", "bf16")]
_DISP_NBYTES = {torch.bfloat16: 2, torch.float8_e4m3fn: 1}.get(_DISP_DT, 0.5)
_FP4 = _DISP_DT is torch.float4_e2m1fn_x2
# Geometry, same spelling as tools/ep_test.sh. Unset = the backend's tuned default.
_G = {
    k: (int(os.environ[k]) if os.environ.get(k) else None)
    for k in ("DBN", "DWPB", "CBN", "CWPB")
}


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    n_experts = world * EPR
    M = max(SWEEP)
    g = torch.Generator(device="cpu").manual_seed(1234 + rank)
    # Inputs before the communicator: comm_create leaves a latched HIP error that
    # the next torch call reports as its own.
    if _FP4:  # no float cast path; generate packed bytes and reinterpret
        inp = (
            torch.randint(0, 256, (M, HIDDEN // 2), generator=g, dtype=torch.uint8)
            .view(_DISP_DT)
            .to(dev)
        )
    else:
        inp = (
            torch.randn(M, HIDDEN, generator=g, dtype=torch.float32)
            .to(_DISP_DT)
            .to(dev)
        )
    wts = torch.rand(M, TOPK, generator=g, dtype=torch.float32).to(dev)
    idx = (
        torch.stack([torch.randperm(n_experts, generator=g)[:TOPK] for _ in range(M)])
        .to(torch.int32)
        .to(dev)
    )
    # Unique destination PEs per token: what an identity expert makes combine sum.
    U = (
        torch.zeros(M, world, dtype=torch.bool)
        .scatter_(1, (idx.cpu().long() // EPR), True)
        .sum(1)
    )

    obj = [cco.Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    # Sized for bf16, the widest, and for one arena per backend.
    vmm = len(BACKENDS) * 2 * (world * M * HIDDEN * 2 * 2 + (16 << 20)) + (512 << 20)
    comm = cco.Communicator.init(world, rank, obj[0], vmm)

    def build(backend):
        if SDMA_TOKEN_COPY and backend != "flydsl":
            raise ValueError("SDMA_TOKEN_COPY is supported by the flydsl backend only")
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
            kernel_backend=backend,
            sdma_token_copy=bool(SDMA_TOKEN_COPY),
            sdma_queue_count=SDMA_QUEUES,
            dispatch_block_num=_G["DBN"],
            warp_num_per_block=_G["DWPB"],
            combine_block_num=_G["CBN"],
            combine_warp_num_per_block=_G["CWPB"],
        )
        return EpDispatchCombineOp(cfg, comm)

    ops = {b: build(b) for b in BACKENDS}

    if rank == 0:
        print(
            f"# EP{world} hidden={HIDDEN} topk={TOPK} epr={EPR} "
            f"disp={_DISP_DT} comb=bf16 backends={BACKENDS} modes={MODES} "
            f"iters={ITERS} combine_in={COMBINE_IN} check={CHECK} "
            f"sdma={SDMA_TOKEN_COPY} sdma_queues={SDMA_QUEUES}",
            flush=True,
        )

    def lockstep():
        torch.cuda.synchronize()
        dist.barrier()

    def prime(op, ct, i_, w_, x_):
        """One full pair, untimed. Reads total_recv for the host, builds the buffer
        the timed loop will reuse, and with CHECK verifies the result through that
        same buffer -- so the gate covers exactly what gets timed, staged copy
        included. Must be a PAIR: a bare dispatch would leave total_recv set, and
        dispatch accumulates into it while only combine clears it, so the next
        combine would stage twice the tokens and run past the arena.
        Returns (total_recv, buf, ok, checked)."""
        *_, total_t, r = op.dispatch(i_, w_, None, x_, return_routing=True)
        lockstep()
        total = int(total_t.cpu().item())
        stage = op.combine_in_view()[:total]
        checked = bool(CHECK) and not _FP4  # fp4 combine is too lossy to compare
        if checked:  # identity expert: stage the dispatched tokens unchanged
            stage.copy_(op.recv_tokens()[:total].to(stage.dtype))
        buf = stage.clone() if COMBINE_IN == "staged" else stage
        out, _ = op.combine(buf, routing=r)
        lockstep()
        if not checked:
            return total, buf, True, False
        exp = U[:ct].view(ct, 1).float() * inp[:ct].float().cpu()
        lossy = _DISP_DT is torch.float8_e4m3fn
        atol, rtol = (1.0, 1.5e-1) if lossy else (2e-2, 2e-2)
        bad = torch.tensor(
            [0 if torch.allclose(out.float().cpu(), exp, atol=atol, rtol=rtol) else 1]
        )
        dist.all_reduce(bad)
        if rank == 0 and bad.item():
            print(
                f"  ct={ct:<5d} [{op.backend_name}] CHECK FAIL "
                f"({int(bad.item())}/{world} ranks, identity expert, U in "
                f"[{int(U[:ct].min())},{int(U[:ct].max())}])",
                flush=True,
            )
        return total, buf, bad.item() == 0, True

    def time_pairs(mode, one_pair, capture):
        """ITERS (dispatch, combine) pairs; mean us per leg.

        Warmup is lock-stepped per iteration: at small token counts a rank that
        starts call N+1 before every rank finished N can overwrite an unconsumed
        cross-device barrier flag, and both ranks hang. The timed loop is not --
        the kernels' own barrier keeps the ranks within one iteration."""
        for _ in range(WARMUP):
            one_pair()
            lockstep()
        lockstep()

        if mode == "graph":
            gd, gc = capture()
            for _ in range(WARMUP):
                gd.replay()
                gc.replay()
            lockstep()
            run_d, run_c = gd.replay, gc.replay
        else:
            run_d, run_c = capture()

        ev = [
            [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
            for _ in range(4)
        ]
        for i in range(ITERS):
            ev[0][i].record()
            run_d()
            ev[1][i].record()
            ev[2][i].record()
            run_c()
            ev[3][i].record()
        torch.cuda.synchronize()
        dist.barrier()
        d = sum(ev[0][i].elapsed_time(ev[1][i]) for i in range(ITERS)) / ITERS * 1000
        c = sum(ev[2][i].elapsed_time(ev[3][i]) for i in range(ITERS)) / ITERS * 1000
        return d, c

    failures = checked = points = 0
    for ct in SWEEP:
        i_, w_, x_ = inp[:ct], wts[:ct], idx[:ct]
        for name, op in ops.items():
            points += 1
            total, buf, ok, was_checked = prime(op, ct, i_, w_, x_)
            checked += was_checked
            if not ok:
                failures += 1
                continue  # never report bandwidth for a kernel computing garbage

            def one_pair():
                """A layer's two all2all legs, in order, nothing in between."""
                *_, r = op.dispatch(i_, w_, None, x_, return_routing=True)
                op.combine(buf, routing=r)

            def capture_pair():
                """One graph per leg, so the pair still alternates on replay. The
                dispatch graph rewrites the same dest_map every time and the
                combine graph was captured against that handle."""
                gd = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gd):
                    *_, r_cap = op.dispatch(i_, w_, None, x_, return_routing=True)
                lockstep()
                gc = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gc):
                    op.combine(buf, routing=r_cap)
                lockstep()
                return gd, gc

            held = [None]  # eager's combine needs the handle its dispatch produced

            def eager_d():
                *_, r = op.dispatch(i_, w_, None, x_, return_routing=True)
                held[0] = r

            def eager_legs():
                return eager_d, lambda: op.combine(buf, routing=held[0])

            for mode in MODES:
                d_us, c_us = time_pairs(
                    mode, one_pair, capture_pair if mode == "graph" else eager_legs
                )
                # Bytes off this rank; the legs differ whenever dispatch is narrower.
                d_bw = total * HIDDEN * _DISP_NBYTES / (1000**3) / (d_us / 1e6)
                c_bw = total * HIDDEN * 2 / (1000**3) / (c_us / 1e6)
                got = torch.tensor([d_us, c_us, float(total)], dtype=torch.float64)
                dist.all_reduce(got)
                if rank == 0:
                    n = world
                    print(
                        f"  ct={ct:<5d} [{name}/{mode}] "
                        f"dispatch {got[0]/n:7.1f} us ({d_bw:6.1f} GB/s)  "
                        f"combine {got[1]/n:7.1f} us ({c_bw:6.1f} GB/s)  "
                        f"pair {(got[0]+got[1])/n:7.1f} us  recv~{got[2]/n:.0f}",
                        flush=True,
                    )
                lockstep()

    if rank == 0:
        # Say how many points were verified, not just that none failed -- with
        # CHECK=0 or DISP=fp4 nothing is compared, and a skipped check is not a
        # passing one.
        why = " (fp4 not compared)" if _FP4 else "" if CHECK else " (CHECK=0)"
        print(
            f"# {'FAIL' if failures else 'PASS'}: {failures} failed, "
            f"{checked}/{points} points verified{why}"
        )
    for op in ops.values():
        op.close()
    comm.destroy()
    dist.destroy_process_group()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
