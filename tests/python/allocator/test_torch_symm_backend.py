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
import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp
from mori.allocator import flat_layout, handle_type, register_symm_backend

from tests.python.utils import TorchDistContext, get_free_port


def _run(rank, world_size, port):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        group_name = dist.group.WORLD.group_name

        register_symm_backend()
        symm_mem.set_backend("MORI")
        assert symm_mem.get_backend(device) == "MORI"
        symm_mem.enable_symm_mem_for_group(group_name)
        assert handle_type(rank) in ("fabric", "posix_fd")

        n = 1024
        t = symm_mem.empty(n, dtype=torch.float32, device=device)
        t.fill_(float(rank + 1))
        torch.cuda.synchronize()

        hdl = symm_mem.rendezvous(t, group_name)
        assert hdl.world_size == world_size and hdl.rank == rank

        # Peers land in one flat span: buffer_ptrs[r] == flat_base + r*stride.
        base, stride = flat_layout(hdl)
        for r, p in enumerate(hdl.buffer_ptrs):
            assert p == base + r * stride, f"rank {r} not at base + r*stride"

        # Read every peer and check its sentinel.
        for pe in range(world_size):
            peer = hdl.get_buffer(pe, (4,), torch.float32)
            assert abs(peer[0].item() - (pe + 1)) < 1e-6, f"peer {pe} mismatch"

        # torch's own collective, on mori memory.
        expect = float(sum(r + 1 for r in range(world_size)))
        out = torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group_name)
        torch.cuda.synchronize()
        assert abs(out[0].item() - expect) < 1e-3

        dist.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 GPUs")
def test_symm_backend():
    world_size = 2
    port = get_free_port()
    mp.spawn(_run, args=(world_size, port), nprocs=world_size, join=True)
