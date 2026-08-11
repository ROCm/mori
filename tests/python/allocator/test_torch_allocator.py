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
from mori import shmem
from mori.allocator import MoriAllocator, is_available, register_symm_backend

from tests.python.utils import TorchDistContext, get_free_port


def _test_mem_pool(rank, world_size, port):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init(dist.group.WORLD.group_name)
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)

        assert is_available(), "symmetric heap should be usable after shmem init"

        pool = torch.cuda.MemPool(MoriAllocator.get_allocator(device).allocator())

        # Allocation is collective: same order, same sizes on every rank.
        with torch.cuda.use_mem_pool(pool):
            t = torch.zeros(1024, dtype=torch.int32, device=device)

        # The tensor must be usable as an ordinary tensor ...
        t.fill_(rank)
        torch.cuda.synchronize()
        assert int(t[0].item()) == rank
        assert int(t[-1].item()) == rank

        # ... and must live in the symmetric heap, so shmem can resolve it.
        peer_ptr = shmem.shmem_ptr_p2p(t.data_ptr(), rank, rank)
        assert peer_ptr != 0, "tensor did not come from the symmetric heap"

        shmem.shmem_barrier_all()

        # A tensor produced by a compute op inside the pool is symmetric too -- the
        # reason for using a MemPool rather than shmem_malloc directly.
        with torch.cuda.use_mem_pool(pool):
            y = torch.ones(64, 64, device=device) @ torch.ones(64, 64, device=device)
        torch.cuda.synchronize()
        assert shmem.shmem_ptr_p2p(y.data_ptr(), rank, rank) != 0

        shmem.shmem_barrier_all()
        del t, y
        shmem.shmem_finalize()


def _test_symm_backend(rank, world_size, port):
    """Drive the MORI backend through torch's own symm_mem entry points."""
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        group_name = dist.group.WORLD.group_name
        shmem.shmem_torch_process_group_init(group_name)
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)

        register_symm_backend()
        symm_mem.set_backend("MORI")
        assert symm_mem.get_backend(device) == "MORI"
        symm_mem.enable_symm_mem_for_group(group_name)

        n = 1024
        t = symm_mem.empty(n, dtype=torch.float32, device=device)
        t.fill_(float(rank + 1))
        torch.cuda.synchronize()

        hdl = symm_mem.rendezvous(t, group_name)
        assert hdl.world_size == world_size
        assert hdl.rank == rank

        # Every peer is P2P-reachable on one node, so no buffer pointer should be null.
        for pe in range(world_size):
            peer = hdl.get_buffer(pe, (4,), torch.float32)
            assert abs(peer[0].item() - (pe + 1)) < 1e-6, f"peer {pe} readback mismatch"

        # torch's own collective, running on mori memory.
        expect = float(sum(r + 1 for r in range(world_size)))
        out = torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group_name)
        torch.cuda.synchronize()
        assert abs(out[0].item() - expect) < 1e-3

        shmem.shmem_barrier_all()
        del t
        shmem.shmem_finalize()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 GPUs")
def test_symm_backend():
    world_size = 2
    port = get_free_port()
    mp.spawn(_test_symm_backend, args=(world_size, port), nprocs=world_size, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 GPUs")
def test_mem_pool_allocation():
    world_size = 2
    port = get_free_port()
    mp.spawn(_test_mem_pool, args=(world_size, port), nprocs=world_size, join=True)


def test_probe_before_init():
    """is_available() must report False rather than crashing when shmem is not up."""
    assert is_available() is False
