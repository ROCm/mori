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
"""The measured process group for the EPv2 VRAM-release test: build one cco
stack, tear it down cleanly, exit.

Nothing is asserted here. The point is what the *parent* sees once every rank is
gone, so this side only has to build what it was asked for, close it the way
production does, and exit 0. The assertions live in
``test_vram_release_on_exit.py``, which outlives this process group.

``STAGE`` picks how far up the stack to build, so a residue can be charged to a
layer instead of to "EPv2":

    comm    Communicator.init -> destroy.  The flat-VA reservation only; no
            allocation, no export, no peer import.
    window  + alloc_mem + register_window -> close.  Exactly the pair
            ``SymmArena.__init__`` makes, which isolates the window
            export/import from everything the op layers on top.
    op      + EpDispatchCombineOp + ROUNDS x dispatch/combine -> close.

``CCO_GPU`` (read by ``set_device``) pins every rank to one GPU, which turns the
``window`` stage into a same-device import: the peer loop still runs, but DRM
hands back the exporter's own GEM object instead of attaching across devices.
The test uses that as a control.

Every stage closes what it opened and leaves through ``with``, so a residue the
parent sees afterwards is one that survived an orderly teardown.

    STAGE=window WIN_MB=256 torchrun --standalone --nproc_per_node=8 vram_leak_worker.py
"""

import json
import os
import sys

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "examples", "cco", "python"))
from cco_example_common import set_device  # noqa: E402

from mori.cco import Communicator  # noqa: E402
from mori.ops.dispatch_combine_v2 import (  # noqa: E402
    EpDispatchCombineConfig,
    EpDispatchCombineOp,
)

STAGE = os.environ.get("STAGE", "window")

# Pinned rather than defaulted: on gfx950 the op would otherwise pick flydsl,
# which carves a different arena, and the report would name a stack that was not
# the one measured.
BACKEND = os.environ.get("BACKEND", "hip")

# The window stage registers a fixed size instead of deriving one from the world
# size, so solo / same-GPU / cross-GPU runs all register the same bytes and their
# deltas are directly comparable.
WIN_BYTES = int(os.environ.get("WIN_MB", "256")) << 20

# The op stage's arena comes from the config. These are sized to land well above
# the test's detection floor while staying far cheaper than a serving shape.
M = int(os.environ.get("NUM_TOKEN", "1024"))
HIDDEN = int(os.environ.get("HIDDEN", "7168"))
TOPK = int(os.environ.get("TOPK", "4"))
EPR = int(os.environ.get("NUM_EXPERT_PER_RANK", "8"))
ROUNDS = int(os.environ.get("ROUNDS", "5"))

PER_RANK_VMM = int(os.environ.get("PER_RANK_VMM_GB", "4")) << 30

_STAGES = ("comm", "window", "op")


def _cfg(rank, world):
    return EpDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=HIDDEN,
        max_num_inp_token_per_rank=M,
        num_experts_per_rank=EPR,
        num_experts_per_token=TOPK,
        data_type=torch.bfloat16,
        kernel_backend=BACKEND,
    )


def _stage_window(comm):
    """alloc_mem + register_window, closed in SymmArena's order (window first)."""
    mem = comm.alloc_mem(WIN_BYTES)
    window = comm.register_window(mem.ptr, WIN_BYTES)
    try:
        return WIN_BYTES
    finally:
        window.close()
        mem.close()


def _stage_op(comm, rank, world):
    """Build the op, run ROUNDS of dispatch/combine, close."""
    dev = torch.device("cuda")
    with EpDispatchCombineOp(_cfg(rank, world), comm) as op:
        x = torch.randn(M, HIDDEN, device=dev, dtype=torch.bfloat16)
        indices = torch.stack(
            [torch.randperm(EPR * world, device=dev)[:TOPK] for _ in range(M)]
        ).to(torch.int32)
        weights = torch.rand(M, TOPK, device=dev)
        staged = op.combine_in_view()
        for _ in range(ROUNDS):
            out, _, _, _, _, routing = op.dispatch(
                x, weights, None, indices, return_routing=True
            )
            staged.copy_(out)
            op.combine(staged, routing=routing)
        torch.cuda.synchronize()
        return op.arena.total_bytes


def main():
    if STAGE not in _STAGES:
        raise SystemExit(f"STAGE must be one of {_STAGES}, got {STAGE!r}")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="gloo")

    objs = [Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(objs, src=0)

    symm_bytes = 0
    with Communicator.init(world, rank, objs[0], per_rank_vmm=PER_RANK_VMM) as comm:
        if STAGE == "window":
            symm_bytes = _stage_window(comm)
        elif STAGE == "op":
            symm_bytes = _stage_op(comm, rank, world)

    # Past this line the communicator is destroyed and every symmetric resource
    # it tracked is closed. Whatever the parent still sees outlived that.
    dist.barrier()
    print(
        "# PROBE "
        + json.dumps(
            {
                "rank": rank,
                "stage": STAGE,
                "backend": BACKEND,
                "gpu": torch.cuda.current_device(),
                # Symmetric bytes this rank actually registered. The number the
                # parent's per-GPU delta is compared against: a delta that lands
                # on it is this window failing to come back.
                "symm_bytes": symm_bytes,
                "per_rank_vmm": PER_RANK_VMM,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    dist.destroy_process_group()
    print(f"[rank{rank}] clean exit", flush=True)


if __name__ == "__main__":
    main()
