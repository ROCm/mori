# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from tests.python.utils import get_free_port, TorchDistContext
import torch
import torch.distributed as dist
from mori.io import (
    IOEngineConfig,
    BackendType,
    IOEngine,
    EngineDesc,
    MemoryDesc,
    StatusCode,
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
        "--buffer-size",
        type=int,
        default=32768,
        help="Number of element in a single transfer, default: 16384",
    )
    parser.add_argument(
        "--transfer-batch-size",
        type=int,
        default=64,
        help="Number of transfer per round, default: 64",
    )
    parser.add_argument(
        "--enable-batch-transfer",
        action="store_true",
        help="Whether to enable batch APIs, default: False",
    )
    parser.add_argument(
        "--enable-sess",
        action="store_true",
        help="Whether to use session, default: False",
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
        buffer_size: int,
        transfer_batch_size: int,
        enable_batch_transfer: bool = False,
        enable_sess: bool = False,
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
        self.buffer_size = buffer_size
        self.transfer_batch_size = transfer_batch_size
        self.enable_batch_transfer = enable_batch_transfer
        self.enable_sess = enable_sess

        self.world_size = self.num_initiator_dev + self.num_target_dev
        if self.node_rank == 0:
            self.global_rank = self.role_rank
            self.role = EngineRole.INITIATOR
        else:
            self.global_rank = self.role_rank + self.num_initiator_dev
            self.role = EngineRole.TARGET

        self.device = torch.device("cuda", self.role_rank)
        self.tensor = torch.randn(self.buffer_size * self.transfer_batch_size).to(
            self.device, dtype=torch.float8_e4m3fnuz
        )

    def print_config(self):
        print("MORI-IO Benchmark Configurations:")
        print(f"  host: {self.host}")
        print(f"  port: {self.port}")
        print(f"  node_rank: {self.node_rank}")
        print(f"  role: {self.role}")
        print(f"  role_rank: {self.role_rank}")
        print(f"  num_initiator_dev: {self.num_initiator_dev}")
        print(f"  num_target_dev: {self.num_target_dev}")
        print(f"  buffer_size: {self.buffer_size} B")
        print(f"  transfer_batch_size: {self.transfer_batch_size}")
        print(f"  enable_batch_transfer: {self.enable_batch_transfer}")
        print(f"  enable_sess: {self.enable_sess}")
        print()

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

    def validate(self):
        if self.role is EngineRole.INITIATOR:
            int8_tensor = torch.empty(
                self.buffer_size * self.transfer_batch_size,
                device=self.device,
                dtype=torch.int8,
            )
            dist.recv(int8_tensor, src=self.num_initiator_dev + self.role_rank)
            tensor = int8_tensor.view(torch.float8_e4m3fnuz)
            assert torch.equal(self.tensor, tensor)
            print("Validation Pass")
        else:
            int8_view = self.tensor.view(torch.uint8)
            dist.send(int8_view, dst=self.role_rank)

    def initialize(self):
        config = IOEngineConfig(
            host=self.host,
            port=self.port,
        )
        self.engine = IOEngine(key=f"{self.role.name}-{self.role_rank}", config=config)
        self.engine.create_backend(BackendType.RDMA)

        self.engine_desc = self.engine.get_engine_desc()
        engine_desc_bytes = self.engine_desc.pack()

        if self.role is EngineRole.INITIATOR:
            for i in range(self.num_target_dev):
                self.send_bytes(engine_desc_bytes, self.num_initiator_dev + i)
            for i in range(self.num_target_dev):
                peer_engine_desc_bytes = self.recv_bytes(self.num_initiator_dev + i)
                peer_engine_desc = EngineDesc.unpack(peer_engine_desc_bytes)
                self.engine.register_remote_engine(peer_engine_desc)
        else:
            for i in range(self.num_initiator_dev):
                peer_engine_desc_bytes = self.recv_bytes(i)
                peer_engine_desc = EngineDesc.unpack(peer_engine_desc_bytes)
                self.engine.register_remote_engine(peer_engine_desc)
            for i in range(self.num_initiator_dev):
                self.send_bytes(engine_desc_bytes, i)

        self.mem = self.engine.register_torch_tensor(self.tensor)

        if self.role is EngineRole.TARGET:
            mem_desc = self.mem.pack()
            self.send_bytes(mem_desc, self.role_rank)
        else:
            target_mem_desc = self.recv_bytes(self.num_initiator_dev + self.role_rank)
            self.target_mem = MemoryDesc.unpack(target_mem_desc)
            self.sess = self.engine.create_session(self.mem, self.target_mem)

    def run_single_once(self):
        if self.role is EngineRole.INITIATOR:
            status_list = []
            transfer_uids = []

            for i in range(self.transfer_batch_size):
                transfer_uids.append(self.engine.allocate_transfer_uid())

            for i in range(self.transfer_batch_size):
                offset = self.buffer_size * i
                if self.enable_sess:
                    transfer_status = self.sess.read(
                        offset,
                        offset,
                        self.buffer_size,
                        transfer_uids[i],
                    )
                else:
                    transfer_status = self.engine.read(
                        self.mem,
                        offset,
                        self.target_mem,
                        offset,
                        self.buffer_size,
                        transfer_uids[i],
                    )
                status_list.append(transfer_status)

            for i, status in enumerate(status_list):
                while status.Code() == StatusCode.INIT:
                    pass
                assert status.Code() == StatusCode.SUCCESS

    def run_batch_once(self):
        if self.role is EngineRole.INITIATOR:
            offsets = [(i * self.buffer_size) for i in range(self.transfer_batch_size)]
            sizes = [self.buffer_size for _ in range(self.transfer_batch_size)]
            transfer_uid = self.engine.allocate_transfer_uid()
            if self.enable_sess:
                transfer_status = self.sess.batch_read(
                    offsets,
                    offsets,
                    sizes,
                    transfer_uid,
                )
            else:
                transfer_status = self.engine.batch_read(
                    self.mem,
                    offsets,
                    self.target_mem,
                    offsets,
                    sizes,
                    transfer_uid,
                )
            while transfer_status.Code() == StatusCode.INIT:
                pass
            assert transfer_status.Code() == StatusCode.SUCCESS

    def run_once(self):
        if self.enable_batch_transfer:
            self.run_batch_once()
        else:
            self.run_single_once()

    def run(self):
        with TorchDistContext(
            rank=self.global_rank,
            world_size=self.world_size,
            master_addr=None,
            master_port=None,
            device_id=self.role_rank,
        ):
            self.initialize()
            self.run_once()
            self.validate()
            self.run_once()
            dist.barrier()

            round = 100
            latency = []
            for i in range(round):
                st = time.time()
                self.run_once()
                latency.append(time.time() - st)

            total_mem_gb = self.buffer_size * self.transfer_batch_size / (10**6)

            avg_duration = sum(latency) / len(latency)
            min_duration = min(latency)
            avg_duration_us, min_duration_us = avg_duration * (
                10**6
            ), min_duration * (10**6)

            avg_bw = total_mem_gb / (10**3) / avg_duration
            max_bw = total_mem_gb / (10**3) / min_duration

            if self.role is EngineRole.INITIATOR:
                print(
                    f"Duration {min_duration_us:.2f}({avg_duration_us:.2f}) us, "
                    f"bytes {total_mem_gb} MB, bandwidth: {max_bw:.2f}({avg_bw:.2f}) GB/s"
                )


def benchmark_engine(local_rank, node_rank, args):
    bench = MoriIoBenchmark(
        host=args.host,
        port=get_free_port(),
        node_rank=node_rank,
        rank_in_node=local_rank,
        buffer_size=args.buffer_size,
        transfer_batch_size=args.transfer_batch_size,
        enable_batch_transfer=args.enable_batch_transfer,
        enable_sess=args.enable_sess,
        num_initiator_dev=args.num_initiator_dev,
        num_target_dev=args.num_target_dev,
    )
    bench.print_config()
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
