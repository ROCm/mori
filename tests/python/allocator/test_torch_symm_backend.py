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
import gc

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp
from mori.allocator import (  # importing registers "MORI"
    get_cco_window,
    handle_type,
    signal_pad_supported,
)

from tests.python.utils import TorchDistContext, get_free_port


def _run(rank, world_size, port):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        group_name = dist.group.WORLD.group_name

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

        # The public torch view remains a pointer array, but every entry now comes from
        # the same CCO window exposed to MORI-specific LSA kernels.
        ptrs = hdl.buffer_ptrs
        assert len(ptrs) == world_size and all(p for p in ptrs)
        lsa = get_cco_window(t, group_name)
        assert lsa.window_handle
        assert lsa.local_ptr == ptrs[rank]
        assert lsa.rank == rank and lsa.world_size == world_size
        assert lsa.lsa_rank == rank and lsa.lsa_size == world_size
        assert lsa.lsa_start == 0 and lsa.per_rank_size % (4 << 30) == 0
        assert [lsa.peer_ptr(r) for r in range(world_size)] == ptrs

        stride = ptrs[1] - ptrs[0]
        for r, p in enumerate(ptrs):
            assert p == ptrs[0] + r * stride, f"rank {r} not at base + r*stride"

        # Read every peer and check its sentinel.
        for pe in range(world_size):
            peer = hdl.get_buffer(pe, (4,), torch.float32)
            assert abs(peer[0].item() - (pe + 1)) < 1e-6, f"peer {pe} mismatch"

        # torch's own collective, on mori memory. It barriers on the device, so it needs
        # the signal pad; without it the backend must say so rather than corrupt memory.
        expect = float(sum(r + 1 for r in range(world_size)))
        if signal_pad_supported():
            out = torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group_name)
            torch.cuda.synchronize()
            assert abs(out[0].item() - expect) < 1e-3
        else:
            with pytest.raises(RuntimeError, match="MORI_SYMM_SIGNAL_PAD"):
                torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group_name)

        dist.barrier()
        # The owning LSA/torch handles keep the original VMM allocation alive even
        # after tensor storage invokes allocator.free().
        del t
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier()
        assert lsa.local_ptr == hdl.buffer_ptrs[rank]
        assert hdl.get_buffer(rank, (1,), torch.float32).item() != 0


def _run_divergent_release(rank, world_size, port):
    """A later CCO window must work after ranks release an older one differently."""
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        group_name = dist.group.WORLD.group_name
        symm_mem.set_backend("MORI")
        symm_mem.enable_symm_mem_for_group(group_name)

        old = symm_mem.empty(4096, dtype=torch.float32, device=device)
        old.fill_(float(rank + 1))
        old_hdl = symm_mem.rendezvous(old, group_name)
        old_lsa = get_cco_window(old, group_name)
        torch.cuda.synchronize()
        dist.barrier()

        # rank 0 returns its first-fit slot; rank 1 keeps the old window live. The
        # next collective therefore uses different local slot offsets on the ranks.
        if rank == 0:
            del old_lsa, old_hdl, old
            gc.collect()
            torch.cuda.synchronize()
        dist.barrier()

        new = symm_mem.empty(4096, dtype=torch.float32, device=device)
        new.fill_(float(rank + 11))
        torch.cuda.synchronize()
        new_hdl = symm_mem.rendezvous(new, group_name)
        new_lsa = get_cco_window(new, group_name)
        assert new_lsa.local_ptr == new_hdl.buffer_ptrs[rank]
        assert [new_lsa.peer_ptr(r) for r in range(world_size)] == new_hdl.buffer_ptrs
        for pe in range(world_size):
            peer = new_hdl.get_buffer(pe, (1,), torch.float32)
            assert peer.item() == float(pe + 11)

        dist.barrier()
        if rank != 0:
            del old_lsa, old_hdl, old
        del new_lsa, new_hdl, new
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier()


def _run_release(rank, world_size, port):
    """Releasing a rendezvous'd window must both survive and give the memory back.

    Teardown used to segfault, so it was disabled and every window leaked; this pins
    down both halves of that.
    """
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        group_name = dist.group.WORLD.group_name
        symm_mem.set_backend("MORI")
        symm_mem.enable_symm_mem_for_group(group_name)

        mib = 1 << 20
        rounds = 4

        def cycle(rendezvous):
            t = symm_mem.empty(16 * mib // 4, dtype=torch.float32, device=device)
            if rendezvous:
                hdl = symm_mem.rendezvous(t, group_name)
                assert hdl.world_size == world_size
            else:
                hdl = None
            dist.barrier()
            del hdl, t
            gc.collect()
            torch.cuda.synchronize()
            dist.barrier()

        def measure(rendezvous):
            cycle(rendezvous)  # first cycle also pays any one-off context growth
            settled = torch.cuda.mem_get_info(device)[0]
            for _ in range(rounds):
                cycle(rendezvous)
            return settled - torch.cuda.mem_get_info(device)[0]

        # The control isolates who is at fault. Plain alloc/free exercises only
        # hipMemCreate/hipMemRelease; adding rendezvous brings in the shareable-fd
        # export and import, whose fd lifetime rules changed in ROCm 7.14. If both
        # leak, the pairing itself is broken rather than anything about sharing.
        plain = measure(rendezvous=False)
        shared = measure(rendezvous=True)

        # mem_get_info is device-wide, so anything else sharing the GPU moves it too.
        # Leaking would cost rounds * 16 MiB; half of one window is a wide enough margin
        # to stay clear of that noise while still failing loudly on a real leak.
        report = (
            f"[rank {rank}] hip={torch.version.hip} world={world_size} "
            f"lost over {rounds} x 16 MiB cycles: alloc/free only={plain / mib:.1f} MiB, "
            f"with rendezvous={shared / mib:.1f} MiB"
        )
        print(report, flush=True)
        assert plain < 8 * mib, f"plain alloc/free leaks, before any sharing. {report}"
        assert shared < 8 * mib, f"the rendezvous'd window leaks. {report}"


def _run_subgroups(rank, world_size, port):
    """Disjoint process groups get independent CCO communicators and rank maps."""
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        group01 = dist.new_group([0, 1])
        group23 = dist.new_group([2, 3])
        group = group01 if rank < 2 else group23
        group_rank = rank % 2
        group_name = group.group_name

        symm_mem.set_backend("MORI")
        symm_mem.enable_symm_mem_for_group(group_name)
        tensor = symm_mem.empty(1024, dtype=torch.float32, device=device)
        tensor.fill_(float(rank + 1))
        torch.cuda.synchronize()

        hdl = symm_mem.rendezvous(tensor, group_name)
        lsa = get_cco_window(tensor, group)
        assert hdl.rank == group_rank and hdl.world_size == 2
        assert lsa.rank == group_rank and lsa.world_size == 2
        assert lsa.lsa_rank == group_rank and lsa.lsa_size == 2
        assert [lsa.peer_ptr(r) for r in range(2)] == hdl.buffer_ptrs

        first_global_rank = 0 if rank < 2 else 2
        for peer in range(2):
            remote = hdl.get_buffer(peer, (1,), torch.float32)
            assert remote.item() == float(first_global_rank + peer + 1)

        dist.barrier(group=group)
        del lsa, hdl, tensor
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier(group=group)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 GPUs")
def test_symm_backend():
    world_size = 2
    port = get_free_port()
    mp.spawn(_run, args=(world_size, port), nprocs=world_size, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 GPUs")
def test_symm_divergent_release():
    world_size = 2
    port = get_free_port()
    mp.spawn(
        _run_divergent_release,
        args=(world_size, port),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs at least 4 GPUs")
def test_symm_release():
    # 4 ranks on purpose: the teardown crash did not show at 2 on every torch.
    world_size = 4
    port = get_free_port()
    mp.spawn(_run_release, args=(world_size, port), nprocs=world_size, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs at least 4 GPUs")
def test_symm_subgroups():
    world_size = 4
    port = get_free_port()
    mp.spawn(_run_subgroups, args=(world_size, port), nprocs=world_size, join=True)
