#!/usr/bin/env python3
import os

import torch
import torch.distributed as dist
import torch.nn as nn

import mori.shmem as shmem
from fsdp import fully_shard
from fsdp._fully_shard._mori_sdma_allgather import MoriSdmaAllGather
from tests.python.utils import TorchDistContext, get_free_port


def _assert_allgather_layout(output: torch.Tensor, elems: int, world_size: int) -> None:
    for pe in range(world_size):
        chunk = output[pe * elems : (pe + 1) * elems]
        expected = torch.full_like(chunk, pe + 1)
        if not torch.equal(chunk, expected):
            raise AssertionError(f"Allgather output chunk {pe} did not match")


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, 4, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _run_adapter_layout(rank: int, world_size: int, elems: int) -> None:
    device = torch.device("cuda", rank)
    comm = MoriSdmaAllGather()
    input_tensor = torch.full((elems,), rank + 1, dtype=torch.uint32, device=device)
    output_tensor = comm.allocate(
        (elems * world_size,), dtype=input_tensor.dtype, device=device
    )

    work = comm(
        output_tensor=output_tensor,
        input_tensor=input_tensor,
        group=dist.group.WORLD,
        async_op=True,
    )
    assert work is not None
    work.wait()
    torch.cuda.synchronize(device)
    _assert_allgather_layout(output_tensor, elems, world_size)


def _run_fsdp_smoke(rank: int) -> None:
    device = torch.device("cuda", rank)
    torch.manual_seed(0)
    model = _TinyModel().to(device)
    fully_shard(model, reshard_after_forward=True)

    inp = torch.randn(2, 8, device=device)
    out = model(inp)
    loss = out.float().sum()
    loss.backward()
    torch.cuda.synchronize(device)


def _test_mori_sdma_allgather(rank: int, world_size: int, port: int, elems: int) -> None:
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")
        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        assert my_pe == rank, f"PE mismatch: {my_pe} != {rank}"
        assert npes == world_size, f"npes mismatch: {npes} != {world_size}"

        _run_adapter_layout(rank, world_size, elems)
        dist.barrier()
        _run_fsdp_smoke(rank)
        dist.barrier()

        shmem.shmem_finalize()


def test_mori_sdma_allgather(elems: int = 1024, world_size: int = 8) -> None:
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    os.environ.setdefault("MORI_FSDP_ENABLE_SDMA", "1")
    port = get_free_port()
    torch.multiprocessing.spawn(
        _test_mori_sdma_allgather,
        args=(world_size, port, elems),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test FSDP2 MORI SDMA allgather")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--elems", type=int, default=1024)
    args = parser.parse_args()
    test_mori_sdma_allgather(elems=args.elems, world_size=args.world_size)
