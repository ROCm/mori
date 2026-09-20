"""Distributed correctness and fixed-storage checks for shared FSDP outputs."""

import copy
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from mori.ccl.torch_fsdp import MoriSdmaAllGather, MoriSdmaAllGatherPool


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(64, 128)
        self.down = torch.nn.Linear(128, 64)

    def forward(self, x):
        return x + self.down(self.up(x).relu()) * 0.1


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Linear(64, 64)
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(6)])
        self.head = torch.nn.Linear(64, 64)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def _group(module):
    return module._get_fsdp_state()._fsdp_param_group


def _bytes(module, dtype):
    group = _group(module)
    return sum(p.padded_sharded_param_size.numel() for p in group.fsdp_params) * (
        group.mesh_info.shard_process_group.size() * torch.empty((), dtype=dtype).element_size()
    )


def _training_case(rank, world_size, dtype, async_op, reshard, zero_copy=True):
    device = torch.device("cuda", rank)
    torch.manual_seed(2718)
    native = _Model().to(device=device, dtype=dtype)
    shared = copy.deepcopy(native)
    mesh = init_device_mesh("cuda", (world_size,))
    policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)
    for model in (native, shared):
        for block in model.blocks:
            fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard)
        fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
        model._set_unshard_async_op(async_op)
    block_bytes = max(_bytes(block, dtype) for block in shared.blocks)
    pool = MoriSdmaAllGatherPool(
        [block_bytes, block_bytes, _bytes(shared, dtype)], group=dist.group.WORLD,
        device=device,
    )
    modules = list(shared.blocks) + [shared]
    comms = []
    for index, module in enumerate(modules):
        comm = MoriSdmaAllGather(zero_copy_output=zero_copy, output_pool=pool,
                                buffer_index=index % 2 if index < 6 else 2)
        module.set_custom_all_gather(comm)
        comms.append(comm)
    pool.initialize()

    # One rank's consumers finish later; remote writes must respect that event.
    def delayed_consumer(module, args):
        if rank == 0:
            torch.cuda._sleep(100_000)

    handles = [block.register_forward_pre_hook(delayed_consumer) for block in shared.blocks]
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.01) for model in (native, shared)]
    expected_pointers = None
    for p, q in zip(native.parameters(), shared.parameters()):
        torch.testing.assert_close(p.to_local(), q.to_local(), rtol=0, atol=0)
    for step in range(4):
        torch.manual_seed(8192 + step + rank)
        x = torch.randn(8, 64, device=device, dtype=dtype)
        losses = []
        for model, optimizer in zip((native, shared), optimizers):
            optimizer.zero_grad(set_to_none=True)
            loss = model(x).float().square().mean()
            loss.backward()
            losses.append(loss.detach())
        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
        for p, q in zip(native.parameters(), shared.parameters()):
            torch.testing.assert_close(p.grad.to_local(), q.grad.to_local(), rtol=0, atol=0)
        for optimizer in optimizers:
            optimizer.step()
        for p, q in zip(native.parameters(), shared.parameters()):
            torch.testing.assert_close(p.to_local(), q.to_local(), rtol=0, atol=0)
        pointers = []
        for module, comm in zip(modules, comms):
            slot = pool._slots[comm._buffer_index]
            assert slot.owner is None
            for param in _group(module).fsdp_params:
                if not zero_copy:
                    assert not param._keep_all_gather_output_storage
                    continue
                assert param._keep_all_gather_output_storage
                for output in param.all_gather_outputs:
                    assert output.untyped_storage().data_ptr() == pool._buffer.data_ptr()
                    assert slot.buffer.data_ptr() <= output.data_ptr()
                    assert output.data_ptr() + output.numel() * output.element_size() <= (
                        slot.buffer.data_ptr() + slot.buffer.numel()
                    )
                    pointers.append(output.data_ptr())
        if expected_pointers is None:
            expected_pointers = pointers
        assert pointers == expected_pointers
    for handle in handles:
        handle.remove()
    ready_bytes = (world_size * 4 + 15) // 16 * 16
    assert pool.allocated_bytes == 2 * block_bytes + _bytes(shared, dtype) + ready_bytes
    pool.close()


def _stream_reuse_case(rank, world_size):
    device = torch.device("cuda", rank)
    count = 4096
    pool = MoriSdmaAllGatherPool([count * world_size * 4] * 2, group=dist.group.WORLD, device=device)
    comms = [MoriSdmaAllGather(output_pool=pool, buffer_index=i) for i in range(2)]
    pool.initialize()
    gathers = [torch.cuda.Stream(), torch.cuda.Stream()]
    consumer = torch.cuda.Stream()
    results = []
    for step in range(12):
        comm = comms[step % 2]
        gather = gathers[step % 2]
        with torch.cuda.stream(gather):
            output = comm.allocate((count * world_size,), dtype=torch.float32, device=device)
            # Input packing may write the output before the collective starts.
            output.fill_(-1)
            source = torch.full((count,), float(step * 10 + rank), device=device)
            work = comm(output, source, dist.group.WORLD, async_op=True)
        with torch.cuda.stream(consumer):
            work.wait()
            if rank == step % world_size:
                torch.cuda._sleep(1_000_000)
            results.append(output.clone())
            comm.release_output()
    torch.cuda.synchronize()
    for step, result in enumerate(results):
        expected = torch.cat([
            torch.full((count,), float(step * 10 + peer), device=device)
            for peer in range(world_size)
        ])
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    pool.close()


def _persistent_packing_case(rank):
    device = torch.device("cuda", rank)
    packer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    for zero_copy in (False, True):
        comm = MoriSdmaAllGather(zero_copy_output=zero_copy)
        results = []
        for step in range(8):
            with torch.cuda.stream(packer):
                output = comm.allocate((4096,), dtype=torch.float32, device=device)
                output.fill_(step)
                ready = packer.record_event()
            with torch.cuda.stream(consumer):
                consumer.wait_event(ready)
                torch.cuda._sleep(1_000_000)
                results.append(output.clone())
                comm.release_output()
        torch.cuda.synchronize()
        for step, result in enumerate(results):
            torch.testing.assert_close(result, torch.full_like(result, step), rtol=0, atol=0)


def _subgroup_case(rank, world_size):
    if world_size != 4:
        return
    device = torch.device("cuda", rank)
    subgroup = dist.new_group([0, 1])
    other_full_group = dist.new_group(list(range(world_size)))
    pool = MoriSdmaAllGatherPool([4096], group=dist.group.WORLD, device=device)
    comm = MoriSdmaAllGather(output_pool=pool)
    pool.initialize()
    if rank < 2:
        comm.layout.prepare_output([4], 4, 2, torch.float32, device,
                                   [[torch.float32]], [[4]], False)
        output = comm.allocate((8,), dtype=torch.float32, device=device)
        source = torch.full((4,), float(rank), device=device)
        comm(output, source, subgroup, async_op=True).wait()
        torch.testing.assert_close(output, torch.tensor([0.] * 4 + [1.] * 4, device=device))
        assert pool._collective is None
        assert not comm._pool_active
    # Nonmembers never enter the adapter's subgroup call.
    dist.barrier()
    comm.layout.prepare_output([4], 4, world_size, torch.float32, device,
                               [[torch.float32]], [[4]], True)
    output = comm.allocate((4 * world_size,), dtype=torch.float32, device=device)
    try:
        comm(output, torch.ones(4, device=device), other_full_group)
    except ValueError as error:
        assert "bound process group" in str(error)
    else:
        raise AssertionError("a different full process group was accepted")
    assert pool._collective is None
    assert pool._slots[0].owner is None
    output = comm.allocate((4 * world_size,), dtype=torch.float32, device=device)
    comm(output, torch.full((4,), float(rank), device=device), dist.group.WORLD, async_op=True).wait()
    expected = torch.arange(world_size, device=device, dtype=torch.float32).repeat_interleave(4)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    comm.release_output()
    comm.release_output()
    pool.close()
    if rank < 2:
        dist.destroy_process_group(subgroup)
    dist.destroy_process_group(other_full_group)


def _configuration_case(rank):
    device = torch.device("cuda", rank)
    for field in ("capacity", "slot", "mode", "key"):
        sizes = [128, 128]
        if field == "capacity" and rank == 0:
            sizes[0] = 144
        pool = MoriSdmaAllGatherPool(sizes, group=dist.group.WORLD, device=device)
        MoriSdmaAllGather(
            zero_copy_output=not (field == "mode" and rank == 0),
            output_pool=pool, buffer_index=int(field == "slot" and rank == 0),
            group_key="other" if field == "key" and rank == 0 else "block",
        )
        try:
            pool.initialize()
        except ValueError as error:
            assert "differ across ranks" in str(error)
        else:
            raise AssertionError("mismatched static configuration was accepted")
        assert pool._collective is None
        pool.close()


def _worker(rank, world_size, port):
    import mori.shmem as shmem

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
        world_size=world_size, timeout=timedelta(seconds=120),
    )
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    shmem.shmem_torch_process_group_init("default")
    _configuration_case(rank)
    _subgroup_case(rank, world_size)
    _persistent_packing_case(rank)
    _stream_reuse_case(rank, world_size)
    for dtype in (torch.float32, torch.bfloat16):
        for async_op in (False, True):
            _training_case(rank, world_size, dtype, async_op, True)
    if world_size == 4:
        _training_case(rank, world_size, torch.bfloat16, True, 2)
        _training_case(rank, world_size, torch.float32, True, True, zero_copy=False)
    dist.barrier()
    shmem.shmem_finalize()
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_shared_fsdp_outputs(world_size):
    if os.environ.get("MORI_ENABLE_SDMA") != "1" or torch.cuda.device_count() < world_size:
        pytest.skip("requires MORI_ENABLE_SDMA=1 and enough GPUs")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    torch.multiprocessing.spawn(_worker, args=(world_size, port), nprocs=world_size)
