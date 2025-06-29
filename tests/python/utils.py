import os
import torch
import torch.distributed as dist
import socket
from multiprocessing import Queue
import sys
from torch.multiprocessing import Pipe
import mori
import traceback


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TorchDistContext:
    def __init__(self, rank, world_size, master_addr="localhost", master_port="12335"):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port

    def __enter__(self):
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)

        torch.cuda.set_device(self.rank)
        device = torch.device("cuda", self.rank)

        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            rank=self.rank,
            world_size=self.world_size,
            device_id=device,
        )

        world_group = torch.distributed.group.WORLD
        assert world_group is not None
        torch._C._distributed_c10d._register_process_group("default", world_group)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


class TorchDistProcessManager:
    def __init__(self, init_mori_shmem=True):
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.processes = []
        self.init_mori_shmem = init_mori_shmem

    def start_workers(self, world_size):
        def worker(rank, world_size, port):
            with TorchDistContext(
                rank=rank, world_size=world_size, master_port=port
            ) as ctx:
                if self.init_mori_shmem:
                    mori.shmem.shmem_torch_process_group_init("default")
                while True:
                    task = self.task_queue.get()
                    if task == "STOP":
                        break
                    func, args = task
                    try:
                        result = func(rank, *args)
                        self.result_queue.put((rank, result))
                    except Exception as e:
                        self.result_queue.put((rank, traceback.format_exc()))

        port = get_free_port()
        self.processes = [
            torch.multiprocessing.Process(target=worker, args=(rank, world_size, port))
            for rank in range(world_size)
        ]
        for p in self.processes:
            p.start()

    def shutdown(self):
        for _ in range(len(self.processes)):
            self.task_queue.put("STOP")
        for p in self.processes:
            p.join()
