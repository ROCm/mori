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
"""HIP-graph capture of a whole dispatch -> expert -> combine step.

Two things could break capture and neither shows up in eager runs: a host->device
sync inside the op aborts it, and the cross-device barrier epoch has to advance ON
DEVICE (dispatch ends with atomicAdd(xdbFlag, 1)) or capture freezes one value into
the graph and every replay after the first spins or passes vacuously.

Identity expert, so the reference is U[t] * inp[t] (U = distinct dest PEs). What is
verified is that REPLAY keeps reproducing it, not just the first launch.

    torchrun --standalone --nproc_per_node=4 test_graph_capture.py
"""
import os
import sys

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "examples", "cco", "python"))
from cco_example_common import set_device, sync  # noqa: E402
from mori.cco import Communicator  # noqa: E402
from mori.ops.dispatch_combine_v2 import (  # noqa: E402
    EpDispatchCombineConfig,
    EpDispatchCombineOp,
)

HIDDEN = int(os.environ.get("HIDDEN", 2048))
K = int(os.environ.get("TOPK", 8))
EPR = int(os.environ.get("EPR", 16))
CT = int(os.environ.get("CT", 512))  # one token count: a graph is a fixed shape
REPLAYS = int(os.environ.get("REPLAYS", 5))
# LEC=1: local_expert_count() between dispatch and combine, as SGLang's recorder.
LEC = int(os.environ.get("LEC", 0))
# STDMOE=1: convert_dispatch_output -> identity -> convert_combine_input in the step.
STDMOE = int(os.environ.get("STDMOE", 0))


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


def main():
    d = Dist()
    rank, npes = d.rank, d.world
    set_device(d.local_rank)
    dev = torch.device("cuda", d.local_rank)
    num_experts = npes * EPR

    g = torch.Generator(device="cpu").manual_seed(1234 + rank)
    inp = (
        torch.randn(CT, HIDDEN, generator=g, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(dev)
    )
    idx = torch.randint(0, num_experts, (CT, K), generator=g, dtype=torch.int32).to(dev)
    wts = torch.rand(CT, K, generator=g, dtype=torch.float32).to(dev)

    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = d.bcast_uid(uid)
    win = npes * CT * HIDDEN * 2 * 2 + (1 << 24)
    with Communicator.init(npes, rank, uid, per_rank_vmm=2 * win + (1 << 28)) as comm:
        cfg = EpDispatchCombineConfig(
            rank=rank,
            world_size=npes,
            hidden_dim=HIDDEN,
            max_num_inp_token_per_rank=CT,
            num_experts_per_rank=EPR,
            num_experts_per_token=K,
            data_type=torch.bfloat16,
            enable_std_moe=bool(STDMOE),
        )
        op = EpDispatchCombineOp(cfg, comm)
        comm.barrier()

        # Identity expert: hand the dispatched tokens straight to combine. The hip
        # backend stages them into the arena inside the combine kernel, so the whole
        # step is two launches and no host work in between -- which is what makes it
        # capturable as one graph.
        def step():
            recv_x, _w, _s, _i, _total, routing = op.dispatch(
                inp, wts, None, idx, return_routing=True
            )
            lec = op.local_expert_count() if LEC else None
            if STDMOE:
                op.convert_dispatch_output()
                op.convert_combine_input(routing)
            out, _ow = op.combine(recv_x, routing=routing)
            return out, lec

        # Reference, eager.
        out_eager, lec_eager = step()
        out_eager = out_eager.clone()
        lec_eager = lec_eager.clone() if LEC else None
        sync()
        comm.barrier()

        if STDMOE:
            # The identity expert telescopes to (sum_k w) * x.
            scale = wts.float().cpu().sum(dim=1, keepdim=True)
        else:
            U = [len({int(i) // EPR for i in idx[t].cpu().tolist()}) for t in range(CT)]
            scale = torch.tensor(U, dtype=torch.float32).view(CT, 1)
        exp = (scale * inp.float().cpu()).to(torch.bfloat16)
        tol = dict(atol=5e-2, rtol=5e-2) if STDMOE else dict(atol=1e-2, rtol=1e-2)
        ok_eager = torch.allclose(out_eager.float().cpu(), exp.float(), **tol)
        lec_exp = None
        if LEC:
            all_i = [torch.empty_like(idx.cpu()) for _ in range(npes)]
            dist.all_gather(all_i, idx.cpu())
            ids = torch.cat(all_i).flatten().long() - rank * EPR
            ids = ids[(ids >= 0) & (ids < EPR)]
            lec_exp = torch.bincount(ids, minlength=EPR).to(torch.int32)
            ok_eager = ok_eager and torch.equal(lec_eager.cpu(), lec_exp)

        # Warm up on a side stream, the torch-documented prerequisite for capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)
        sync()
        comm.barrier()

        # Capture. Nothing here executes on the GPU: a failed capture raises, and a
        # host sync inside the op would be the thing that raises.
        graph = torch.cuda.CUDAGraph()
        captured = captured_lec = None
        err = ""
        try:
            with torch.cuda.graph(graph):
                captured, captured_lec = step()
        except Exception as e:  # noqa: BLE001 - reported, not swallowed
            err = f"{type(e).__name__}: {e}"
        ok_capture = err == ""
        errs_cap = d.allreduce_sum(0 if ok_capture else 1)
        if errs_cap:
            if rank == 0:
                print(f"# GRAPH-CAPTURE: FAIL ({err})", flush=True)
            d.shutdown()
            return

        # Replay. Every rank replays the same number of times; the kernels' own
        # cross-device barrier keeps them together exactly as it does eagerly.
        # With metrics on: a replay runs no host code, only device counters see it.
        get_stats = getattr(op, "get_traffic_stats", None)
        stats0 = get_stats() if get_stats is not None else None
        stats_first = None
        bad = 0
        for _ in range(REPLAYS):
            comm.barrier()
            graph.replay()
            sync()
            comm.barrier()
            if stats0 is not None and stats_first is None:
                stats_first = get_stats()
            if not torch.allclose(captured.float().cpu(), exp.float(), **tol):
                bad += 1
            elif LEC and not torch.equal(captured_lec.cpu(), lec_exp):
                bad += 1
        errs = d.allreduce_sum(bad) + d.allreduce_sum(0 if ok_eager else 1)

        if stats0 is not None:
            stats1 = get_stats()
            one = int((stats_first["dispatch_sent"] - stats0["dispatch_sent"]).sum())
            total = int((stats1["dispatch_sent"] - stats0["dispatch_sent"]).sum())
            back = int((stats1["combine_recv"] - stats0["combine_recv"]).sum())
            ok_stats = (
                stats1["dispatch_calls"] - stats0["dispatch_calls"] == REPLAYS
                and stats1["combine_calls"] - stats0["combine_calls"] == REPLAYS
                and one > 0
                and total == REPLAYS * one
                and back == total
                and stats1["dispatch_kernel_seconds"] > stats0["dispatch_kernel_seconds"]
                and stats1["combine_kernel_seconds"] > stats0["combine_kernel_seconds"]
            )
            errs_stats = d.allreduce_sum(0 if ok_stats else 1)
            if rank == 0:
                print(
                    f"# GRAPH-STATS: {'PASS' if errs_stats == 0 else 'FAIL'} "
                    f"({REPLAYS} replays counted {total} tokens on rank 0, {one} a replay)",
                    flush=True,
                )

        if rank == 0:
            print(
                f"# GRAPH-CAPTURE: {'PASS' if errs == 0 else 'FAIL'} "
                f"(capture ok, {REPLAYS} replays, ct={CT} hidden={HIDDEN} "
                f"backend={op.backend_name} lec={LEC} stdmoe={STDMOE}; "
                f"eager={'ok' if ok_eager else 'BAD'})",
                flush=True,
            )
    d.shutdown()


if __name__ == "__main__":
    main()
