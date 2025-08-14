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

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of batch size",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=1,
        help="Number of block size for each transfer",
    )

    parser.add_argument(
        "--unit",
        type=str,
        default="GiB",
        help="GB|GiB|Gb|MB|MiB|Mb|KB|KiB|Kb",
    )

    parser.add_argument(
        "--role",
        type=str,
        default="target",
        help="target|initiator",
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

        engine_desc = self.engine.get_engine_desc()
        engine_desc_bytes = engine_desc.pack()
        print(engine_desc, engine_desc.key, engine_desc.pack())

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

    def run_once(self):
        pass

    def run(self):
        with TorchDistContext(
            rank=self.global_rank,
            world_size=self.world_size,
            master_addr=None,
            master_port=None,
        ) as ctx:
            self.initialize()


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
