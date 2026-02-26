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
    PollCqMode,
    RdmaBackendConfig,
    XgmiBackendConfig,
    set_log_level,
)
import argparse
from enum import Enum
import os
import time
from prettytable import PrettyTable


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MORI-IO")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["rdma", "xgmi", "tcp"],
        default="rdma",
        help="Backend type: 'rdma' for cross-node, 'tcp' for cross-node fallback, 'xgmi' for intra-node GPU-to-GPU (default: rdma)",
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host IP for mori io engine OOB communication (RDMA/TCP only)",
    )
    parser.add_argument(
        "--src-gpu",
        type=int,
        default=0,
        help="Source GPU device ID for XGMI mode (default: 0)",
    )
    parser.add_argument(
        "--dst-gpu",
        type=int,
        default=1,
        help="Destination GPU device ID for XGMI mode (default: 1)",
    )
    parser.add_argument(
        "--num-streams",
        type=int,
        default=64,
        help="Number of HIP streams per device for XGMI mode (default: 64)",
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=64,
        help="Number of HIP events per device for XGMI mode (default: 64)",
    )
    parser.add_argument(
        "--xgmi-multiprocess",
        action="store_true",
        help="Enable multi-process mode for XGMI backend to test cross-process GPU communication (default: False)",
    )
    parser.add_argument(
        "--op-type",
        type=str,
        choices=["read", "write"],
        default="read",
        help="Type of ops, choices [read, write], default to 'read'",
    )
    # Backward-compatible alias for existing scripts.
    parser.add_argument(
        "--op",
        dest="op_type",
        type=str,
        choices=["read", "write"],
        help="Alias for --op-type",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=32768,
        help="Number of element in a single transfer, default: 16384",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run sizes from 8 till 2^20",
    )
    parser.add_argument(
        "--sweep-start-size",
        type=int,
        default=8,
        help="Starting message size when using --all sweep (default: 8)",
    )
    parser.add_argument(
        "--sweep-max-size",
        type=int,
        default=2**20,
        help="Maximum message size when using --all sweep (default: 2**20)",
    )
    parser.add_argument(
        "--all-batch",
        action="store_true",
        help="Run batch sizes from 8 to 32768",
    )
    parser.add_argument(
        "--transfer-batch-size",
        type=int,
        default=256,
        help="Number of transfer per iteration, default: 64",
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
    parser.add_argument(
        "--num-qp-per-transfer",
        type=int,
        default=1,
        help="Number of QPs for a single transfer",
    )
    parser.add_argument(
        "--num-worker-threads",
        type=int,
        default=1,
        help="Number of threads used for transfer",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=128,
        help="Number of iterations running test",
    )
    parser.add_argument(
        "--poll_cq_mode",
        type=str,
        default="polling",
        choices=["polling", "event"],
        help="Determines how to process CQE, choices ['polling', event]",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["trace", "debug", "info", "warning", "error", "critical"],
        help="Log level options: 'trace', 'debug', 'info', 'warning', 'error', 'critical'",
    )

    args = parser.parse_args()
    return args


class EngineRole(Enum):
    INITIATOR = 0
    TARGET = 1


class MoriIoBenchmark:
    def __init__(
        self,
        op_type: str,
        buffer_size: int,
        transfer_batch_size: int,
        enable_batch_transfer: bool = False,
        enable_sess: bool = False,
        iters: int = 128,
        sweep: bool = False,
        sweep_batch: bool = False,
        sweep_start_size: int = 8,
        sweep_max_size: int = 2**20,
        backend_type: str = "rdma",
        host: str = "",
        port: int = 0,
        node_rank: int = 0,
        rank_in_node: int = 0,
        num_initiator_dev: int = 1,
        num_target_dev: int = 1,
        num_qp_per_transfer: int = 1,
        num_worker_threads: int = 1,
        poll_cq_mode: str = "polling",
        src_gpu: int = 0,
        dst_gpu: int = 1,
        num_streams: int = 64,
        num_events: int = 64,
        xgmi_multiprocess: bool = False,
    ):
        self.op_type = op_type
        self.buffer_size = buffer_size
        self.transfer_batch_size = transfer_batch_size
        self.enable_batch_transfer = enable_batch_transfer
        self.enable_sess = enable_sess
        self.iters = iters
        self.sweep = sweep
        self.sweep_batch = sweep_batch
        self.sweep_start_size = sweep_start_size
        self.sweep_max_size = sweep_max_size
        self.backend_type = backend_type

        self.host = host
        self.port = port
        self.node_rank = node_rank
        self.role_rank = rank_in_node
        self.num_initiator_dev = num_initiator_dev
        self.num_target_dev = num_target_dev
        self.num_qp_per_transfer = num_qp_per_transfer
        self.num_worker_threads = num_worker_threads
        self.poll_cq_mode = (
            PollCqMode.POLLING if poll_cq_mode == "polling" else PollCqMode.EVENT
        )

        self.src_gpu = src_gpu
        self.dst_gpu = dst_gpu
        self.num_streams = num_streams
        self.num_events = num_events
        self.xgmi_multiprocess = xgmi_multiprocess

        if self.sweep:
            if self.sweep_start_size <= 0 or self.sweep_max_size <= 0:
                raise ValueError("Sweep sizes must be positive integers")
            if self.sweep_start_size > self.sweep_max_size:
                raise ValueError(
                    f"start-buffer-size ({self.sweep_start_size}) should not exceed max-buffer-size ({self.sweep_max_size})"
                )

        if self.backend_type == "xgmi":
            self._setup_xgmi()
        else:
            self._setup_rdma()

    def _setup_rdma(self):
        assert self.num_initiator_dev == self.num_target_dev
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

    def _setup_xgmi(self):
        if self.xgmi_multiprocess:
            self.world_size = 2
            if self.node_rank == 0:
                self.global_rank = self.role_rank
                self.role = EngineRole.INITIATOR
                self.device = torch.device("cuda", self.src_gpu)
            else:
                self.global_rank = self.role_rank + 1
                self.role = EngineRole.TARGET
                self.device = torch.device("cuda", self.dst_gpu)

            self.tensor = torch.randn(self.buffer_size * self.transfer_batch_size).to(
                self.device, dtype=torch.float8_e4m3fnuz
            )
        else:
            self.role = EngineRole.INITIATOR
            self.src_device = torch.device("cuda", self.src_gpu)
            self.dst_device = torch.device("cuda", self.dst_gpu)

            self.tensor = torch.randn(self.buffer_size * self.transfer_batch_size).to(
                self.src_device, dtype=torch.float8_e4m3fnuz
            )
            self.target_tensor = torch.zeros(
                self.buffer_size * self.transfer_batch_size
            ).to(self.dst_device, dtype=torch.float8_e4m3fnuz)

    def print_config(self):
        print("MORI-IO Benchmark Configurations:")
        print(f"  backend: {self.backend_type.upper()}")
        print(f"  op_type: {self.op_type}")

        if self.backend_type == "xgmi":
            print(f"  xgmi_multiprocess: {self.xgmi_multiprocess}")
            print(f"  src_gpu: {self.src_gpu}")
            print(f"  dst_gpu: {self.dst_gpu}")
            print(f"  num_streams: {self.num_streams}")
            print(f"  num_events: {self.num_events}")
        else:
            print(f"  host: {self.host}")
            print(f"  port: {self.port}")
            print(f"  node_rank: {self.node_rank}")
            print(f"  role: {self.role}")
            print(f"  role_rank: {self.role_rank}")
            print(f"  num_initiator_dev: {self.num_initiator_dev}")
            print(f"  num_target_dev: {self.num_target_dev}")
            print(f"  num_qp_per_transfer: {self.num_qp_per_transfer}")
            print(f"  num_worker_threads: {self.num_worker_threads}")
            print(f"  poll_cq_mode: {self.poll_cq_mode}")

        print(f"  buffer_size: {self.buffer_size} B")
        print(f"  transfer_batch_size: {self.transfer_batch_size}")
        print(f"  enable_batch_transfer: {self.enable_batch_transfer}")
        print(f"  enable_sess: {self.enable_sess}")
        print(f"  iters: {self.iters}")
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
        if self.backend_type == "xgmi":
            self._validate_xgmi()
        else:
            self._validate_rdma()

    def _validate_rdma(self):
        if self.role is EngineRole.INITIATOR:
            int8_tensor = torch.empty(
                self.buffer_size * self.transfer_batch_size,
                device=self.device,
                dtype=torch.int8,
            )
            dist.recv(int8_tensor, src=self.num_initiator_dev + self.role_rank)
            tensor = int8_tensor.view(torch.float8_e4m3fnuz)
            assert torch.equal(self.tensor, tensor)
        else:
            int8_view = self.tensor.view(torch.uint8)
            dist.send(int8_view, dst=self.role_rank)

    def _validate_xgmi(self):
        if self.xgmi_multiprocess:
            if self.role is EngineRole.INITIATOR:
                transfer_uid = self.engine.allocate_transfer_uid()
                status = self.engine.write(
                    self.mem, 0, self.target_mem, 0, self.buffer_size, transfer_uid
                )
                status.Wait()
                assert status.Succeeded()
            else:
                time.sleep(0.5)
        else:
            transfer_uid = self.engine.allocate_transfer_uid()
            status = self.engine.write(
                self.mem, 0, self.target_mem, 0, self.buffer_size, transfer_uid
            )
            status.Wait()
            assert status.Succeeded()

            src_cpu = self.tensor[: self.buffer_size].cpu()
            dst_cpu = self.target_tensor[: self.buffer_size].cpu()
            assert torch.equal(
                src_cpu.view(torch.uint8), dst_cpu.view(torch.uint8)
            ), "Validation failed: data mismatch"

    def initialize(self):
        if self.backend_type == "xgmi":
            self._initialize_xgmi()
        else:
            self._initialize_rdma()

    def _initialize_rdma(self):
        config = IOEngineConfig(
            host=self.host,
            port=self.port,
        )
        self.engine = IOEngine(key=f"{self.role.name}-{self.role_rank}", config=config)
        if self.backend_type == "tcp":
            self.engine.create_backend(BackendType.TCP)
        else:
            config = RdmaBackendConfig(
                qp_per_transfer=self.num_qp_per_transfer,
                post_batch_size=-1,
                num_worker_threads=self.num_worker_threads,
                poll_cq_mode=self.poll_cq_mode,
            )
            self.engine.create_backend(BackendType.RDMA, config)

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

    def _initialize_xgmi(self):
        config = IOEngineConfig(host="", port=0)

        if self.xgmi_multiprocess:
            engine_key = f"xgmi-{self.role.name}-{self.role_rank}"
        else:
            engine_key = "xgmi-benchmark"

        self.engine = IOEngine(key=engine_key, config=config)

        xgmi_config = XgmiBackendConfig(
            num_streams=self.num_streams,
            num_events=self.num_events,
        )
        self.engine.create_backend(BackendType.XGMI, xgmi_config)

        if self.xgmi_multiprocess:
            self.engine_desc = self.engine.get_engine_desc()
            engine_desc_bytes = self.engine_desc.pack()

            if self.role is EngineRole.INITIATOR:
                target_engine_desc_bytes = self.recv_bytes(src=self.global_rank + 1)
                target_engine_desc = EngineDesc.unpack(target_engine_desc_bytes)
                self.engine.register_remote_engine(target_engine_desc)
                self.send_bytes(engine_desc_bytes, dst=self.global_rank + 1)
            else:
                self.send_bytes(engine_desc_bytes, dst=self.global_rank - 1)
                initiator_engine_desc_bytes = self.recv_bytes(src=self.global_rank - 1)
                initiator_engine_desc = EngineDesc.unpack(initiator_engine_desc_bytes)
                self.engine.register_remote_engine(initiator_engine_desc)

            self.mem = self.engine.register_torch_tensor(self.tensor)

            mem_desc_bytes = self.mem.pack()
            if self.role is EngineRole.INITIATOR:
                target_mem_desc_bytes = self.recv_bytes(src=self.global_rank + 1)
                self.target_mem = MemoryDesc.unpack(target_mem_desc_bytes)
                self.send_bytes(mem_desc_bytes, dst=self.global_rank + 1)
            else:
                self.send_bytes(mem_desc_bytes, dst=self.global_rank - 1)
                initiator_mem_desc_bytes = self.recv_bytes(src=self.global_rank - 1)
                self.target_mem = MemoryDesc.unpack(initiator_mem_desc_bytes)

            if self.enable_sess:
                self.sess = self.engine.create_session(self.mem, self.target_mem)
        else:
            self.mem = self.engine.register_torch_tensor(self.tensor)
            self.target_mem = self.engine.register_torch_tensor(self.target_tensor)

            if self.enable_sess:
                self.sess = self.engine.create_session(self.mem, self.target_mem)

    def run_single_once(self, buffer_size, transfer_batch_size):
        assert buffer_size <= self.buffer_size
        if (
            self.backend_type in ("rdma", "tcp")
            or (self.backend_type == "xgmi" and self.xgmi_multiprocess)
        ) and self.role is EngineRole.TARGET:
            return 0

        status_list = []
        transfer_uids = []

        for i in range(transfer_batch_size):
            transfer_uids.append(self.engine.allocate_transfer_uid())

        func, arg_list = None, []
        for i in range(transfer_batch_size):
            offset = buffer_size * i
            if self.enable_sess:
                func = self.sess.read if self.op_type == "read" else self.sess.write
                arg_list.append(
                    (
                        offset,
                        offset,
                        buffer_size,
                        transfer_uids[i],
                    )
                )
            else:
                func = self.engine.read if self.op_type == "read" else self.engine.write
                arg_list.append(
                    (
                        self.mem,
                        offset,
                        self.target_mem,
                        offset,
                        buffer_size,
                        transfer_uids[i],
                    )
                )

        st = time.time()
        for i in range(transfer_batch_size):
            status = func(*arg_list[i])
            status_list.append(status)
        for status in status_list:
            status.Wait()
        duration = time.time() - st

        for status in status_list:
            assert status.Succeeded(), f"Transfer failed: {status.Message()}"
        return duration

    def run_batch_once(self, buffer_size, transfer_batch_size):
        assert buffer_size <= self.buffer_size
        if (
            self.backend_type in ("rdma", "tcp")
            or (self.backend_type == "xgmi" and self.xgmi_multiprocess)
        ) and self.role is EngineRole.TARGET:
            return 0

        offsets = [(i * buffer_size) for i in range(transfer_batch_size)]
        sizes = [buffer_size for _ in range(transfer_batch_size)]
        transfer_uid = self.engine.allocate_transfer_uid()

        if self.enable_sess:
            func = (
                self.sess.batch_read
                if self.op_type == "read"
                else self.sess.batch_write
            )
            args = (
                offsets,
                offsets,
                sizes,
                transfer_uid,
            )
            st = time.time()
            transfer_status = func(*args)
        else:
            func = (
                self.engine.batch_read
                if self.op_type == "read"
                else self.engine.batch_write
            )
            args = (
                [self.mem],
                [offsets],
                [self.target_mem],
                [offsets],
                [sizes],
                [transfer_uid],
            )
            st = time.time()
            transfer_status = func(*args)[0]

        transfer_status.Wait()
        duration = time.time() - st
        assert (
            transfer_status.Succeeded()
        ), f"Batch transfer failed: {transfer_status.Message()}"
        return duration

    def run_once(self, buffer_size, transfer_batch_size):
        if self.enable_batch_transfer:
            return self.run_batch_once(buffer_size, transfer_batch_size)
        else:
            return self.run_single_once(buffer_size, transfer_batch_size)

    def _run_and_compute(self, buffer_size, transfer_batch_size, iters):
        latency = []
        for _ in range(iters):
            duration = self.run_once(buffer_size, transfer_batch_size)
            latency.append(duration)

        if self.role is EngineRole.TARGET and (
            self.backend_type in ("rdma", "tcp")
            or (self.backend_type == "xgmi" and self.xgmi_multiprocess)
        ):
            return 0, 0, 0, 0, 0

        total_mem_mb = buffer_size * transfer_batch_size / (10**6)
        avg_duration = sum(latency) / len(latency)
        min_duration = min(latency)
        avg_duration_us = avg_duration * (10**6)
        min_duration_us = min_duration * (10**6)
        avg_bw = total_mem_mb / (10**3) / avg_duration
        max_bw = total_mem_mb / (10**3) / min_duration

        return total_mem_mb, avg_duration_us, min_duration_us, avg_bw, max_bw

    def _get_table_title(self):
        if self.backend_type == "xgmi":
            if self.xgmi_multiprocess:
                return f"XGMI Multiprocess Benchmark: Rank {self.role_rank} ({self.role.name})"
            else:
                return f"XGMI Benchmark: GPU{self.src_gpu} -> GPU{self.dst_gpu}"
        elif self.backend_type == "tcp":
            return f"TCP Benchmark: Initiator Rank {self.role_rank}"
        else:
            return f"RDMA Benchmark: Initiator Rank {self.role_rank}"

    def _run_benchmark_loop(self):
        self.run_once(self.buffer_size, self.transfer_batch_size)

        table = PrettyTable(
            field_names=[
                "MsgSize (B)",
                "BatchSize",
                "TotalSize (MB)",
                "Max BW (GB/s)",
                "Avg BW (GB/s)",
                "Min Lat (us)",
                "Avg Lat (us)",
            ],
            title=self._get_table_title(),
        )

        if self.sweep:
            cur_size = self.sweep_start_size
            max_size = self.sweep_max_size
            while cur_size <= max_size:
                if self.backend_type in ("rdma", "tcp") or (
                    self.backend_type == "xgmi" and self.xgmi_multiprocess
                ):
                    dist.barrier()
                total_mem_mb, avg_duration, min_duration, avg_bw, max_bw = (
                    self._run_and_compute(
                        cur_size, self.transfer_batch_size, self.iters
                    )
                )
                table.add_row(
                    [
                        cur_size,
                        self.transfer_batch_size,
                        f"{total_mem_mb:.2f}",
                        f"{max_bw:.2f}",
                        f"{avg_bw:.2f}",
                        f"{min_duration:.2f}",
                        f"{avg_duration:.2f}",
                    ]
                )
                cur_size *= 2
        elif self.sweep_batch:
            cur_transfer_batch_size = 1
            max_transfer_batch_size = 32768
            while cur_transfer_batch_size <= max_transfer_batch_size:
                if self.backend_type in ("rdma", "tcp") or (
                    self.backend_type == "xgmi" and self.xgmi_multiprocess
                ):
                    dist.barrier()
                total_mem_mb, avg_duration, min_duration, avg_bw, max_bw = (
                    self._run_and_compute(
                        self.buffer_size, cur_transfer_batch_size, self.iters
                    )
                )
                table.add_row(
                    [
                        self.buffer_size,
                        cur_transfer_batch_size,
                        f"{total_mem_mb:.2f}",
                        f"{max_bw:.2f}",
                        f"{avg_bw:.2f}",
                        f"{min_duration:.2f}",
                        f"{avg_duration:.2f}",
                    ]
                )
                cur_transfer_batch_size *= 2
        else:
            total_mem_mb, avg_duration, min_duration, avg_bw, max_bw = (
                self._run_and_compute(
                    self.buffer_size, self.transfer_batch_size, self.iters
                )
            )
            table.add_row(
                [
                    self.buffer_size,
                    self.transfer_batch_size,
                    f"{total_mem_mb:.2f}",
                    f"{max_bw:.2f}",
                    f"{avg_bw:.2f}",
                    f"{min_duration:.2f}",
                    f"{avg_duration:.2f}",
                ]
            )

        if (
            self.backend_type == "xgmi" and not self.xgmi_multiprocess
        ) or self.role is EngineRole.INITIATOR:
            print(table)

    def run(self):
        if self.backend_type == "xgmi":
            self._run_xgmi()
        else:
            self._run_rdma()

    def _run_xgmi(self):
        if self.xgmi_multiprocess:
            context_device_id = (
                self.device.index
                if hasattr(self, "device") and self.device.index is not None
                else self.role_rank
            )
            with TorchDistContext(
                rank=self.global_rank,
                world_size=self.world_size,
                master_addr=None,
                master_port=None,
                device_id=context_device_id,
                backend="gloo",
            ):
                self.initialize()
                self.run_once(self.buffer_size, self.transfer_batch_size)
                self.validate()
                self.run_once(self.buffer_size, self.transfer_batch_size)
                dist.barrier()
                self._run_benchmark_loop()
        else:
            self.initialize()
            self.validate()
            self._run_benchmark_loop()

    def _run_rdma(self):
        context_device_id = (
            self.device.index
            if hasattr(self, "device") and self.device.index is not None
            else self.role_rank
        )
        with TorchDistContext(
            rank=self.global_rank,
            world_size=self.world_size,
            master_addr=None,
            master_port=None,
            device_id=context_device_id,
            backend="gloo",
        ):
            self.initialize()
            self.run_once(self.buffer_size, self.transfer_batch_size)
            self.validate()
            self.run_once(self.buffer_size, self.transfer_batch_size)
            dist.barrier()
            self._run_benchmark_loop()


def benchmark_xgmi_worker(local_rank, node_rank, args):
    set_log_level(args.log_level)
    max_buffer_size = args.buffer_size
    if args.all:
        max_buffer_size = max(max_buffer_size, args.sweep_max_size)
    max_transfer_batch_size = args.transfer_batch_size
    if args.all_batch:
        max_transfer_batch_size = max(max_transfer_batch_size, 2**15)

    bench = MoriIoBenchmark(
        op_type=args.op_type,
        buffer_size=max_buffer_size,
        transfer_batch_size=max_transfer_batch_size,
        enable_batch_transfer=args.enable_batch_transfer,
        enable_sess=args.enable_sess,
        iters=args.iters,
        sweep=args.all,
        sweep_batch=args.all_batch,
        sweep_start_size=args.sweep_start_size,
        sweep_max_size=args.sweep_max_size,
        backend_type="xgmi",
        node_rank=node_rank,
        rank_in_node=local_rank,
        src_gpu=args.src_gpu,
        dst_gpu=args.dst_gpu,
        num_streams=args.num_streams,
        num_events=args.num_events,
        xgmi_multiprocess=True,
    )
    bench.print_config()
    bench.run()


def benchmark_engine(local_rank, node_rank, args):
    set_log_level(args.log_level)
    max_buffer_size = args.buffer_size
    if args.all:
        max_buffer_size = max(max_buffer_size, args.sweep_max_size)
    max_transfer_batch_size = args.transfer_batch_size
    if args.all_batch:
        max_transfer_batch_size = max(max_transfer_batch_size, 2**15)

    bench = MoriIoBenchmark(
        op_type=args.op_type,
        buffer_size=max_buffer_size,
        transfer_batch_size=max_transfer_batch_size,
        enable_batch_transfer=args.enable_batch_transfer,
        enable_sess=args.enable_sess,
        iters=args.iters,
        sweep=args.all,
        sweep_batch=args.all_batch,
        sweep_start_size=args.sweep_start_size,
        sweep_max_size=args.sweep_max_size,
        backend_type=args.backend,
        host=args.host,
        port=get_free_port(),
        node_rank=node_rank,
        rank_in_node=local_rank,
        num_initiator_dev=args.num_initiator_dev,
        num_target_dev=args.num_target_dev,
        num_qp_per_transfer=args.num_qp_per_transfer,
        num_worker_threads=args.num_worker_threads,
        poll_cq_mode=args.poll_cq_mode,
    )
    bench.print_config()
    bench.run()


def benchmark_xgmi(args):
    num_gpus = torch.cuda.device_count()
    if args.src_gpu >= num_gpus or args.dst_gpu >= num_gpus:
        raise ValueError(f"Invalid GPU ID. Available GPUs: 0-{num_gpus-1}")

    if args.src_gpu == args.dst_gpu:
        print(
            "Warning: src_gpu and dst_gpu are the same. This will be a device-local transfer."
        )

    if args.xgmi_multiprocess:
        num_node = int(os.environ.get("WORLD_SIZE", "2"))
        if num_node != 2:
            raise ValueError(
                f"XGMI multi-process mode requires WORLD_SIZE=2, got {num_node}"
            )

        node_rank = int(os.environ.get("RANK", "0"))
        nprocs = 1
        torch.multiprocessing.spawn(
            benchmark_xgmi_worker,
            args=(node_rank, args),
            nprocs=nprocs,
            join=True,
        )
    else:
        set_log_level(args.log_level)
        max_buffer_size = args.buffer_size
        if args.all:
            max_buffer_size = max(max_buffer_size, args.sweep_max_size)
        max_transfer_batch_size = args.transfer_batch_size
        if args.all_batch:
            max_transfer_batch_size = max(max_transfer_batch_size, 2**15)

        bench = MoriIoBenchmark(
            op_type=args.op_type,
            buffer_size=max_buffer_size,
            transfer_batch_size=max_transfer_batch_size,
            enable_batch_transfer=args.enable_batch_transfer,
            enable_sess=args.enable_sess,
            iters=args.iters,
            sweep=args.all,
            sweep_batch=args.all_batch,
            sweep_start_size=args.sweep_start_size,
            sweep_max_size=args.sweep_max_size,
            backend_type="xgmi",
            src_gpu=args.src_gpu,
            dst_gpu=args.dst_gpu,
            num_streams=args.num_streams,
            num_events=args.num_events,
            xgmi_multiprocess=False,
        )
        bench.print_config()
        bench.run()


def benchmark_rdma(args):
    if args.all:
        if args.sweep_start_size > args.sweep_max_size:
            raise ValueError(
                f"--start-buffer-size ({args.sweep_start_size}) must be <= --max-buffer-size ({args.sweep_max_size})"
            )
        if args.sweep_start_size <= 0 or args.sweep_max_size <= 0:
            raise ValueError("Sweep sizes must be positive integers")

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


def benchmark():
    args = parse_args()

    if args.backend == "xgmi":
        benchmark_xgmi(args)
    else:
        benchmark_rdma(args)


if __name__ == "__main__":
    benchmark()
