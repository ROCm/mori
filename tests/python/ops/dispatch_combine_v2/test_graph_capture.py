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

The Prepare/Launch split exists for this: every plan is compiled and its module
loaded at construction, so a launch is only hipModuleLaunchKernel and nothing in
the steady-state path can fork a compiler mid-capture. This test is what makes
that claim checkable, because two things could still break it and neither shows
up in eager runs:

  * a host->device sync inside the op (a `.item()` on a counter) aborts a capture;
  * the cross-device barrier epoch has to advance ON DEVICE. It does --
    dispatch ends with atomicAdd(xdbFlag, 1) -- so a replay bumps it like a fresh
    launch. Had the host computed the epoch, capture would freeze one value into
    the graph and every replay after the first would spin or pass vacuously.

Identity expert, so the reference is U[t] * inp[t] (U = distinct dest PEs), the
same invariant test_op checks; what is verified here is that REPLAY reproduces it
and keeps reproducing it, not just that the first launch works.

    torchrun --standalone --nproc_per_node=4 test_graph_capture.py
"""
import os
import sys

import numpy as np
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
    inp = torch.randn(CT, HIDDEN, generator=g, dtype=torch.float32).to(
        torch.bfloat16
    ).to(dev)
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
            out, _ow = op.combine(recv_x, routing=routing)
            return out

        # Reference, eager.
        out_eager = step().clone()
        sync()
        comm.barrier()

        U = np.array(
            [
                len({int(i) // EPR for i in idx[t].cpu().tolist()})
                for t in range(CT)
            ]
        )
        exp = (torch.from_numpy(U).view(CT, 1).float() * inp.float().cpu()).to(
            torch.bfloat16
        )
        ok_eager = torch.allclose(
            out_eager.float().cpu(), exp.float(), atol=1e-2, rtol=1e-2
        )

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
        captured = None
        err = ""
        try:
            with torch.cuda.graph(graph):
                captured = step()
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
        bad = 0
        for i in range(REPLAYS):
            comm.barrier()
            graph.replay()
            sync()
            comm.barrier()
            if not torch.allclose(
                captured.float().cpu(), exp.float(), atol=1e-2, rtol=1e-2
            ):
                bad += 1
        errs = d.allreduce_sum(bad) + d.allreduce_sum(0 if ok_eager else 1)

        if rank == 0:
            print(
                f"# GRAPH-CAPTURE: {'PASS' if errs == 0 else 'FAIL'} "
                f"(capture ok, {REPLAYS} replays, ct={CT} hidden={HIDDEN} "
                f"backend={op.backend_name}; eager={'ok' if ok_eager else 'BAD'})",
                flush=True,
            )
    d.shutdown()


if __name__ == "__main__":
    main()
