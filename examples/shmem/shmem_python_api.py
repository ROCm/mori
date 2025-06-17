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

    config = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=global_rank,
        world_size=world_size,
        hidden_dim=7168,
        max_num_inp_token_per_rank=512,
        num_experts_per_rank=32,
        num_experts_per_token=8,
    )
    op = mori.ops.EpDispatchCombineOp(config)
    op.dispatch_internode(
        torch.ones(4, 7168).to(torch.bfloat16).to(device), 
        torch.ones(4, 1).to(torch.float).to(device), 
        torch.ones(4, 8).to(torch.uint32).to(device)
    )
    torch.cuda.synchronize()


def cleanup():
    mori.shmem.shmem_finalize()
    dist.destroy_process_group()


def test_shmem(rank, num_node, gpu_per_node):
    setup(rank, num_node, gpu_per_node)
    cleanup()


if __name__ == "__main__":
    gpu_per_node = os.environ.get('GPU_PER_NODE', None)
    gpu_per_node = int(gpu_per_node) if gpu_per_node is not None else 8
    num_node = int(os.environ['WORLD_SIZE'])

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
