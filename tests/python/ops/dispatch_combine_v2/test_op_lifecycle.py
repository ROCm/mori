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
"""Regression test: building EpDispatchCombineOp repeatedly on one long-lived
Communicator must not leak its symmetric arena. close() (and the context manager)
frees the window and untracks it from comm._resources.

    torchrun --standalone --nproc_per_node=2 test_op_lifecycle.py
"""
import os
import sys

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "python", "mori", "ops", "dispatch_combine_v2"))
sys.path.insert(0, os.path.join(_ROOT, "examples", "cco", "python"))
from cco_example_common import set_device  # noqa: E402
from dispatch_combine_op import (  # noqa: E402
    EpDispatchCombineConfig,
    EpDispatchCombineOp,
)
from mori.cco import Communicator  # noqa: E402

HIDDEN, M, EPR, K, ITERS = 1024, 64, 8, 2, 5


def _cfg(rank, npes):
    return EpDispatchCombineConfig(
        rank=rank,
        world_size=npes,
        hidden_dim=HIDDEN,
        max_num_inp_token_per_rank=M,
        num_experts_per_rank=EPR,
        num_experts_per_token=K,
        data_type=torch.bfloat16,
        combine_mode="gather",
    )


def main():
    rank = int(os.environ["RANK"])
    npes = int(os.environ["WORLD_SIZE"])
    set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    uid = Communicator.get_unique_id() if rank == 0 else None
    obj = [uid]
    dist.broadcast_object_list(obj, src=0)
    uid = obj[0]

    win = M * HIDDEN * 2 * npes + (1 << 24)
    with Communicator.init(npes, rank, uid, per_rank_vmm=2 * win + (1 << 28)) as comm:
        base = len(comm._resources)
        ok = True
        # explicit close(): each op adds mem+win (2), close() returns to base.
        for _ in range(ITERS):
            op = EpDispatchCombineOp(_cfg(rank, npes), comm)
            grew = len(comm._resources) - base
            op.close()
            op.close()  # idempotent
            if grew != 2 or len(comm._resources) != base:
                ok = False
        # context manager exercises __enter__/__exit__.
        with EpDispatchCombineOp(_cfg(rank, npes), comm):
            pass
        if len(comm._resources) != base:
            ok = False
        errs = torch.tensor([0 if ok else 1], device="cuda")
        dist.all_reduce(errs)
        if rank == 0:
            print(
                f"# LIFECYCLE: {'PASS' if errs.item() == 0 else 'FAIL'} "
                f"(base={base}, {ITERS} build/close cycles, no _resources growth)",
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
