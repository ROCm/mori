import pytest
import mori
from tests.python.utils import TorchDistContext, get_free_port
import torch


def _test_torch_init(rank, world_size, port):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port) as ctx:
        mori.shmem.shmem_torch_process_group_init("default")
        assert rank == mori.shmem.shmem_mype()
        assert world_size == mori.shmem.shmem_npes()
        mori.shmem.shmem_finalize()


@pytest.mark.parametrize("world_size", (8,))
def test_torch_init(world_size):
    torch.multiprocessing.spawn(
        _test_torch_init,
        args=(world_size, get_free_port()),
        nprocs=world_size,
        join=True,
    )
