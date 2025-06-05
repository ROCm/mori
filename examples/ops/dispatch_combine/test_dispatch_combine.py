import mori
import os
import torch
import torch.distributed as dist


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    world_group = torch.distributed.group.WORLD
    assert world_group is not None
    torch._C._distributed_c10d._register_process_group("default", world_group)


def cleanup():
    dist.destroy_process_group()


def test_dispatch_combine(rank, world_size):
    setup(rank, world_size)

    mori.torch_mori_shmem.shmem_torch_process_group_init("default")
    print(mori.torch_mori_shmem.shmem_mype())
    print(mori.torch_mori_shmem.shmem_npes())
    mori.torch_mori_shmem.shmem_finalize()

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        test_dispatch_combine, args=(world_size,), nprocs=world_size, join=True
    )
