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
"""Smallest possible check that cco can stand up a multi-node communicator.

This exists because of one environment bug with an expensive failure mode:
ROCm 7.0-7.2.x ships a clr `hipMemSetAccess` defect that makes cco's window
setup fail CROSS-NODE while passing on a single node (known-issues skill,
Issue 2). Anything built on cco internode -- EP, CCL, UMBP -- then dies at
construction with `ccoMemAlloc: hipMemSetAccess failed ... (invalid argument)`,
and the stack trace points at whatever happened to call it rather than at the
runtime.

Running the full EP internode test to learn this costs a build plus several
minutes per attempt. This costs seconds, and it names the cause.

What it does: build a world-sized cco communicator, register a few windows of
DIFFERENT sizes into it (the bug is triggered by sub-buffer geometry, so one
window is not enough to see it), and take a device communicator with the GDA
resources an internode kernel would ask for.

    srun -N2 --ntasks-per-node=8 ... python3 tools/cco_devcomm_probe.py

Prints one PASS/FAIL line from rank 0. Exit code is the verdict.
"""

from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.distributed as dist

from mori.cco import cco as _cco
from mori.cco.communicator import Communicator, DevCommHandle


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")

    objs = [Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(objs, src=0)

    # Deliberately several windows with DIFFERENT sizes. The clr defect walks the
    # parent reservation's sub-buffers from index 0 regardless of the pointer it
    # was handed, so it only misfires once a second, differently-sized window is
    # mapped -- a single-window probe would come back clean on a broken runtime.
    sizes = [64 << 20, 96 << 20, 32 << 20]
    note = ""
    try:
        with Communicator.init(
            world, rank, objs[0], per_rank_vmm=4 * sum(sizes) + (1 << 28)
        ) as comm:
            held = []
            for i, nbytes in enumerate(sizes):
                note = f"window {i} of {len(sizes)} ({nbytes >> 20} MB)"
                mem = comm.alloc_mem(nbytes)
                win = comm.register_window(mem.ptr, nbytes)
                held.append((mem, win))

            # The GDA resources an internode kernel asks for. CROSSNODE rather
            # than RAIL on purpose: a per-PE protocol such as AsyncLL needs QPs
            # to every cross-node peer, and asking for them here is what makes
            # this probe representative of that path.
            reqs = _cco.DevCommRequirements()
            reqs.gda_connection_type = _cco.GDA_CONNECTION_CROSSNODE
            reqs.gda_context_count = 2
            note = "ccoDevCommCreate"
            handle = DevCommHandle(comm, requirements=reqs)
            note = (
                f"ok (lsa_size={handle.lsa_size} lsa_rank={handle.lsa_rank} "
                f"world={handle.world_size})"
            )
            handle.close()
        ok, detail = True, note
    except BaseException as exc:  # noqa: BLE001 - the verdict is the point
        ok, detail = False, f"failed at {note}: {exc}"
        if rank == 0:
            traceback.print_exc()

    votes = torch.tensor([0 if ok else 1])
    dist.all_reduce(votes)
    failures = int(votes.item())

    if rank == 0:
        nodes = world // int(os.environ.get("LOCAL_WORLD_SIZE", world))
        print(
            f"# cco devcomm probe: {'PASS' if failures == 0 else 'FAIL'} "
            f"(world={world}, nodes~{nodes}, {failures} ranks failed) {detail}",
            flush=True,
        )
        if failures:
            print(
                "# If this says hipMemSetAccess/invalid argument, it is the ROCm "
                "7.0-7.2.x clr defect, not your code -- use ROCm 7.14+ (see the "
                "known-issues skill, Issue 2).",
                flush=True,
            )
    dist.destroy_process_group()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
