"""MORI training parity on FSDP-owned and backend-owned layout contracts."""

import copy
import gc
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from mori.ccl.torch_fsdp import MoriSdmaAllGather
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Shard


class _CountedAllGather(MoriSdmaAllGather):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.param_contiguous_calls = 0
        self.rank_major_calls = 0

    def _call_full_group(self, output, input, group, async_op):
        if self._can_call_param_contiguous(input):
            self.param_contiguous_calls += 1
        else:
            self.rank_major_calls += 1
        return super()._call_full_group(output, input, group, async_op)


def _case(
    rank,
    world_size,
    dtype,
    async_op,
    zero_copy,
    fsdp_owned,
    shard_dim,
    keep_storage=False,
):
    device = torch.device("cuda", rank)
    torch.manual_seed(2718)
    reference = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32)
    ).to(device=device, dtype=dtype)
    candidate = copy.deepcopy(reference)
    mesh = init_device_mesh("cuda", (world_size,))
    comms, groups = [], []
    for model in (reference, candidate):
        for module in (model[0], model[2]):
            fully_shard(
                module,
                mesh=mesh,
                reshard_after_forward=True,
                shard_placement_fn=lambda p: Shard(shard_dim if p.ndim > 1 else 0),
            )
            if keep_storage:
                module.set_keep_unsharded_storage(True)
            if model is candidate:
                comm = _CountedAllGather(
                    zero_copy_output=zero_copy, fsdp_owned_output=fsdp_owned
                )
                module.set_all_gather_layout(comm.layout, recurse=False)
                comms.append(comm)
                groups.append(module._get_fsdp_state()._fsdp_param_group)
        fully_shard(model, mesh=mesh, reshard_after_forward=False)
        model._set_unshard_async_op(async_op)
        model.set_manual_backward_finalization(True)
    optimizers = [
        torch.optim.SGD(m.parameters(), lr=0.01) for m in (reference, candidate)
    ]
    pointer_history = None
    for cycle in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for model in (reference, candidate):
            model.set_requires_gradient_sync(False)
        for microbatch in range(2):
            torch.manual_seed(8128 + cycle * 10 + microbatch + rank)
            inputs = torch.randn(4, 32, device=device, dtype=dtype)
            expected = reference(inputs).float().square().mean()
            actual = candidate(inputs).float().square().mean()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected.backward()
            actual.backward()
        for model in (reference, candidate):
            model.set_requires_gradient_sync(True)
            model.finalize_backward()
            model.finalize_backward()
        for ref, got in zip(reference.parameters(), candidate.parameters()):
            torch.testing.assert_close(
                got.grad.to_local(), ref.grad.to_local(), rtol=0, atol=0
            )
        for optimizer in optimizers:
            optimizer.step()
        for ref, got in zip(reference.parameters(), candidate.parameters()):
            torch.testing.assert_close(got.to_local(), ref.to_local(), rtol=0, atol=0)
        pointers = []
        for group, comm in zip(groups, comms):
            if fsdp_owned:
                assert not comm._output_active
                assert group._all_gather_buffers is not None
                for buffer in group._all_gather_buffers.tensors:
                    assert (buffer.untyped_storage().size() != 0) == keep_storage
                for param in group.fsdp_params:
                    for output in param.all_gather_outputs:
                        assert (
                            output.untyped_storage()._cdata
                            != comm._output_buffer.untyped_storage()._cdata
                        )
            else:
                assert not comm._output_active
                if zero_copy and shard_dim == 0:
                    for param in group.fsdp_params:
                        assert param._keep_all_gather_output_storage
                        for output in param.all_gather_outputs:
                            assert (
                                output.untyped_storage().data_ptr()
                                == comm._output_buffer.untyped_storage().data_ptr()
                            )
                            pointers.append(output.data_ptr())
        if pointer_history is None:
            pointer_history = pointers
        assert pointer_history == pointers
    if zero_copy and shard_dim == 0:
        assert all(comm.param_contiguous_calls > 0 for comm in comms)
    else:
        assert all(
            comm.param_contiguous_calls == 0 and comm.rank_major_calls > 0
            for comm in comms
        )
    for comm in comms:
        comm._deregister_output_buffer_if_needed()
    if rank == 0:
        print(
            f"PASS layout e2e {dtype=} {async_op=} {zero_copy=} {fsdp_owned=} {shard_dim=} {keep_storage=}",
            flush=True,
        )


def _unaligned_case(rank, world_size):
    device = torch.device("cuda", rank)
    torch.manual_seed(1729)
    reference = torch.nn.Linear(3, 4, bias=False).to(
        device=device, dtype=torch.bfloat16
    )
    candidate = copy.deepcopy(reference)
    mesh = init_device_mesh("cuda", (world_size,))
    comm = _CountedAllGather(fsdp_owned_output=True)
    for model in (reference, candidate):
        fully_shard(model, mesh=mesh, reshard_after_forward=True)
        model._set_unshard_async_op(True)
    candidate.set_all_gather_layout(comm.layout, recurse=False)
    optimizers = [
        torch.optim.SGD(model.parameters(), lr=0.01) for model in (reference, candidate)
    ]
    for step in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(1024 + step + rank)
        inputs = torch.randn(4, 3, device=device, dtype=torch.bfloat16)
        expected = reference(inputs).float().square().mean()
        actual = candidate(inputs).float().square().mean()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.backward()
        actual.backward()
        torch.testing.assert_close(
            candidate.weight.grad.to_local(),
            reference.weight.grad.to_local(),
            rtol=0,
            atol=0,
        )
        for optimizer in optimizers:
            optimizer.step()
        torch.testing.assert_close(
            candidate.weight.to_local(), reference.weight.to_local(), rtol=0, atol=0
        )
        assert not comm._output_active
    assert comm.param_contiguous_calls == 0 and comm.rank_major_calls > 0
    comm._deregister_output_buffer_if_needed()
    if rank == 0:
        print("PASS layout e2e unaligned BF16 native async fallback", flush=True)


def _worker(rank, world_size, port, fsdp_owned):
    from mori import shmem

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    shmem.shmem_torch_process_group_init("default")
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for async_op in (False, True):
                for zero_copy in (False, True):
                    _case(rank, world_size, dtype, async_op, zero_copy, fsdp_owned, 0)
                    gc.collect()
                    dist.barrier()
        _case(rank, world_size, torch.float32, True, True, fsdp_owned, 1)
        if fsdp_owned:
            _case(rank, world_size, torch.bfloat16, True, True, True, 0, True)
            _unaligned_case(rank, world_size)
        dist.barrier()
    finally:
        shmem.shmem_finalize()
        dist.destroy_process_group()


def test_mori_layout_e2e():
    if os.environ.get("MORI_ENABLE_SDMA") != "1" or torch.cuda.device_count() < 4:
        pytest.skip("requires MORI_ENABLE_SDMA=1 and four GPUs")
    fsdp_owned = os.environ.get("MORI_TEST_FSDP_OWNED") == "1"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    torch.multiprocessing.spawn(_worker, args=(4, port, fsdp_owned), nprocs=4)
