"""torchrun-based bootstrap for the cco dispatch/combine tests (no MPI).

cco itself only needs (nranks, rank, unique_id); MPI was just the courier for
the uid and for test-only allgathers. Here we use torch.distributed (gloo, CPU)
purely as that courier, driven by torchrun's RANK/WORLD_SIZE/LOCAL_RANK env.

    torchrun --standalone --nproc_per_node=8 m2_dispatch_test.py
"""
import os

import torch
import torch.distributed as dist


class Dist:
    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        torch.cuda.set_device(self.local_rank)

    def bcast_uid(self, uid):
        objs = [uid if self.rank == 0 else None]
        dist.broadcast_object_list(objs, src=0)
        return objs[0]

    def allgather(self, obj):
        out = [None] * self.world
        dist.all_gather_object(out, obj)
        return out

    def allreduce_sum(self, value):
        t = torch.tensor([value], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())

    def barrier(self):
        dist.barrier()

    def shutdown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
