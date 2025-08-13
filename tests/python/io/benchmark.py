from tests.python.utils import get_free_port, TorchDistContext
import torch
import torch.distributed as dist
import mori
from mori.io import (
    IOEngineConfig,
    BackendType,
    IOEngine,
    EngineDesc,
    MemoryDesc,
    StatusCode,
    MemoryLocationType,
)
import argparse
from enum import Enum
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MORI-IO")
    parser.add_argument(
        "--host", type=str, help="Host IP for mori io engine OOB communication"
    )
    parser.add_argument(
        "--num-initiator-dev",
        type=int,
        default=1,
        help="Number of devices on initiator side",
    )
    parser.add_argument(
        "--num-target-dev",
        type=int,
        default=1,
        help="Number of devices on target side",
    )

    args = parser.parse_args()
    return args


class EngineRole(Enum):
    INITIATOR = 0
    TARGET = 1


class MoriIoBenchmark:
    def __init__(
        self,
        host: str,
        port: int,
        node_rank: int,
        rank_in_node: int,
        num_initiator_dev: int = 1,
        num_target_dev: int = 1,
    ):
        self.host = host
        self.port = port
        self.node_rank = node_rank
        self.role_rank = rank_in_node
        self.num_initiator_dev = num_initiator_dev
        self.num_target_dev = num_target_dev
        assert self.num_initiator_dev == self.num_target_dev
        self.single_transfer_size = 32768
        self.num_transfer = 64

        self.world_size = self.num_initiator_dev + self.num_target_dev
        if self.node_rank == 0:
            self.global_rank = self.role_rank
            self.role = EngineRole.INITIATOR
        else:
            self.global_rank = self.role_rank + self.num_initiator_dev
            self.role = EngineRole.TARGET
        self.peer_meta = {}

    def send_bytes(self, b: bytes, dst: int):
        t = torch.ByteTensor(list(b))
        length_tensor = torch.IntTensor([t.numel()])
        dist.send(length_tensor, dst=dst)
        dist.send(t, dst=dst)

    def recv_bytes(self, src: int) -> bytes:
        length_tensor = torch.IntTensor([0])
        dist.recv(length_tensor, src=src)
        length = length_tensor.item()
        t = torch.ByteTensor(length)
        dist.recv(t, src=src)
        return bytes(t.tolist())

    def initialize(self):
        config = IOEngineConfig(
            host=self.host,
            port=self.port,
        )
        self.engine = IOEngine(key=f"{self.role.name}-{self.role_rank}", config=config)
        self.engine.create_backend(BackendType.RDMA)

        self.engine_desc = self.engine.get_engine_desc()
        engine_desc_bytes = self.engine_desc.pack()
        print(self.engine_desc, self.engine_desc.key, self.engine_desc.pack())

        if self.role is EngineRole.INITIATOR:
            for i in range(self.num_target_dev):
                self.send_bytes(engine_desc_bytes, self.num_initiator_dev + i)
            for i in range(self.num_target_dev):
                peer_engine_desc_bytes = self.recv_bytes(self.num_initiator_dev + i)
                peer_engine_desc = EngineDesc.unpack(peer_engine_desc_bytes)
                self.engine.register_remote_engine(peer_engine_desc)
                print(f"register remote engine {peer_engine_desc.key}")
        else:
            for i in range(self.num_initiator_dev):
                peer_engine_desc_bytes = self.recv_bytes(i)
                peer_engine_desc = EngineDesc.unpack(peer_engine_desc_bytes)
                self.engine.register_remote_engine(peer_engine_desc)
                print(f"register remote engine {peer_engine_desc.key}")
            for i in range(self.num_initiator_dev):
                self.send_bytes(engine_desc_bytes, i)

        device = torch.device("cuda", self.role_rank)
        tensor = torch.randn(self.single_transfer_size * self.num_transfer).to(
            device, dtype=torch.int8
        )
        self.mem = self.engine.register_torch_tensor(tensor)

        if self.role is EngineRole.TARGET:
            mem_desc = self.mem.pack()
            print(f"{self.engine_desc.key} send mem obj to rank {self.role_rank}")
            self.send_bytes(mem_desc, self.role_rank)
        else:
            print(
                f"{self.engine_desc.key} recv mem obj from rank {self.num_initiator_dev + self.role_rank}"
            )
            target_mem_desc = self.recv_bytes(self.num_initiator_dev + self.role_rank)
            self.target_mem = MemoryDesc.unpack(target_mem_desc)

    def run_once(self, batch_size):
        if self.role is EngineRole.INITIATOR:
            status = []
            transfer_uids = []

            st = time.time()
            for i in range(batch_size):
                transfer_uids.append(self.engine.allocate_transfer_uid())
            print(f"alloc uid duration {time.time()-st}")

            read_st = time.time()
            for i in range(batch_size):
                transfer_status = self.engine.read(
                    self.mem,
                    self.single_transfer_size * (i % self.num_transfer),
                    self.target_mem,
                    self.single_transfer_size * (i % self.num_transfer),
                    self.single_transfer_size,
                    transfer_uids[i],
                )
                status.append(transfer_status)
            print(f"submit duration {time.time()-read_st}")

            while status[-1].Code() == StatusCode.INIT:
                pass

    def run(self):
        with TorchDistContext(
            rank=self.global_rank,
            world_size=self.world_size,
            master_addr=None,
            master_port=None,
            device_id=self.role_rank,
        ) as ctx:
            self.initialize()
            for _ in range(3):
                self.run_once(1)
            round = 1
            dist.barrier()
            st = time.time()
            self.run_once(self.num_transfer * round)
            end = time.time()
            duration = end - st
            total_mem_gb = (
                self.single_transfer_size * self.num_transfer * round / (10**9)
            )
            bw = total_mem_gb / (end - st)
            print(
                f"total duration {duration*1000} ms, bytes {total_mem_gb} GB, bandwidth: {bw} GB/s"
            )


def benchmark_engine(local_rank, node_rank, args):
    bench = MoriIoBenchmark(
        host=args.host,
        port=get_free_port(),
        node_rank=node_rank,
        rank_in_node=local_rank,
        num_initiator_dev=args.num_initiator_dev,
        num_target_dev=args.num_target_dev,
    )
    bench.run()


def benchmark():
    args = parse_args()
    num_node = int(os.environ["WORLD_SIZE"])
    assert num_node == 2
    node_rank = int(os.environ["RANK"])
    nprocs = args.num_initiator_dev if node_rank == 0 else args.num_target_dev
    torch.multiprocessing.spawn(
        benchmark_engine,
        args=(
            node_rank,
            args,
        ),
        nprocs=nprocs,
        join=True,
    )


if __name__ == "__main__":
    benchmark()
