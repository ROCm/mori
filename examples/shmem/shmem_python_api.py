import mori
import os
import time

import torch
import torch.distributed as dist


def setup(local_rank, num_node, gpu_per_node):
    world_size = num_node * gpu_per_node

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    node_rank = int(os.environ["RANK"])
    global_rank = node_rank * gpu_per_node + local_rank
    print(
        f"before init process group, rank{local_rank}, env rank {os.environ["RANK"]}, world_size{world_size}, env worldsize {os.environ['WORLD_SIZE']}"
        f" global_rank {global_rank}"
    )

    dist.init_process_group(
        backend="cpu:gloo",#,cuda:nccl",
        rank=global_rank,
        world_size=world_size,
        # device_id=device,
    )

    print("init process group done")
    world_group = torch.distributed.group.WORLD
    assert world_group is not None

    print("process group ok")
    torch._C._distributed_c10d._register_process_group("default", world_group)
    print(mori.shmem.shmem_torch_process_group_init("default"))

    print(f"I'm pe {mori.shmem.shmem_mype()} in {mori.shmem.shmem_npes()} pes")


def cleanup():
    mori.shmem.shmem_finalize()
    dist.destroy_process_group()


def test_shmem(rank, num_node, gpu_per_node):
    setup(rank, num_node, gpu_per_node)
    cleanup()


if __name__ == "__main__":
    num_node = 2
    gpu_per_node = 4
    world_size = num_node * gpu_per_node
    torch.multiprocessing.spawn(
        test_shmem,
        args=(
            num_node,
            gpu_per_node,
        ),
        nprocs=gpu_per_node,
        join=True,
    )
    # print(os.environ['RANK'] )
    # test_shmem(num_node, gpu_per_node)
