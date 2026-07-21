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
from tests.python.utils import TorchDistContext
import torch
import torch.distributed as dist
from mori.io import (
    IOEngineConfig,
    BackendType,
    IOEngine,
    EngineDesc,
    MemoryDesc,
    MemoryLocationType,
    PollCqMode,
    RdmaBackendConfig,
    XgmiBackendConfig,
    FabricBackendConfig,
    fabric_alloc,
    set_log_level,
)
import argparse
import ctypes
from enum import Enum
import os
import time
from prettytable import PrettyTable


# --- Minimal HIP memcpy helpers (fabric buffers are raw VMM allocations, not
# --- torch tensors, so we fill/verify them directly via the HIP runtime).
_hip_lib = None


def _hip():
    global _hip_lib
    if _hip_lib is None:
        for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
            try:
                _hip_lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if _hip_lib is None:
            raise RuntimeError("Could not load libamdhip64.so for fabric validation")
        _hip_lib.hipMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        _hip_lib.hipMemcpy.restype = ctypes.c_int
        _hip_lib.hipSetDevice.argtypes = [ctypes.c_int]
        _hip_lib.hipSetDevice.restype = ctypes.c_int
        _hip_lib.hipDeviceSynchronize.restype = ctypes.c_int
    return _hip_lib


_HIP_MEMCPY_H2D = 1
_HIP_MEMCPY_D2H = 2


def hip_fill(ptr: int, data: bytes, device: int):
    h = _hip()
    assert h.hipSetDevice(device) == 0
    buf = (ctypes.c_char * len(data)).from_buffer_copy(data)
    assert h.hipMemcpy(ctypes.c_void_p(ptr), buf, len(data), _HIP_MEMCPY_H2D) == 0
    assert h.hipDeviceSynchronize() == 0


def hip_read(ptr: int, nbytes: int, device: int) -> bytes:
    h = _hip()
    assert h.hipSetDevice(device) == 0
    buf = (ctypes.c_char * nbytes)()
    assert h.hipMemcpy(buf, ctypes.c_void_p(ptr), nbytes, _HIP_MEMCPY_D2H) == 0
    assert h.hipDeviceSynchronize() == 0
    return bytes(buf)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MORI-IO")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["rdma", "xgmi", "fabric"],
        default="rdma",
        help="Backend type: 'rdma' for cross-node RDMA, 'xgmi' for intra-node GPU-to-GPU, "
        "'fabric' for cross-node scale-up UALink super-node (same vPOD) (default: rdma)",
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host IP for mori io engine OOB communication (RDMA only)",
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
        "--sweep-step",
        type=int,
        default=0,
        help="Linear step (bytes) for the --all sweep: message size goes "
        "start, start+step, start+2*step, ... up to max. Default 0 = geometric "
        "(double each step). Example: --sweep-start-size 1048576 "
        "--sweep-max-size 33554432 --sweep-step 1048576 sweeps 1MiB..32MiB by 1MiB.",
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
        "--batch-contiguous",
        action="store_true",
        help="Use contiguous offsets so transfers may be merged. Default is non-contiguous (strided offsets, each transfer is a separate WR). Don't enable this when stress SQ / reproduce ENOMEM on notify due to merged wr",
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
        "--target-dev-offset",
        type=int,
        default=0,
        help="Shift each target buffer to GPU (role_rank + offset) %% gpu_count, so the "
        "initiator's GPU i pairs with a different-index target GPU. Used to exercise cross-rail "
        "transfers on rail-only fabrics (e.g. offset 5 makes GPU0 -> GPU5). GPU memory only.",
    )
    parser.add_argument(
        "--num-qp-per-transfer",
        type=int,
        default=4,
        help="Number of QPs for a single transfer (default: 4)",
    )
    parser.add_argument(
        "--num-worker-threads",
        type=int,
        default=1,
        help="Number of threads used for transfer",
    )
    parser.add_argument(
        "--disable-chunking",
        action="store_true",
        help="Disable single-transfer chunking (chunking is enabled by default)",
    )
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=65536,
        help="Chunk size in bytes when chunking is enabled (default: 64KB)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=64,
        help="Max number of chunks per transfer (default: 64)",
    )
    parser.add_argument(
        "--mem-type",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Memory type for transfer buffers: 'gpu' (cuda) or 'cpu' (host) (default: gpu)",
    )
    parser.add_argument(
        "--initiator-mem-type",
        type=str,
        default=None,
        choices=["gpu", "cpu"],
        help="Override buffer memory type on the INITIATOR side (default: --mem-type). "
        "Combine with --target-mem-type for a mixed CPU<->GPU transfer, e.g. "
        "--initiator-mem-type cpu --target-mem-type gpu (RDMA backend only).",
    )
    parser.add_argument(
        "--target-mem-type",
        type=str,
        default=None,
        choices=["gpu", "cpu"],
        help="Override buffer memory type on the TARGET side (default: --mem-type).",
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
        "--max-send-wr",
        type=int,
        default=0,
        help="RDMA max send WRs per QP; 0 = use backend default (default: 0)",
    )
    parser.add_argument(
        "--max-cqe-num",
        type=int,
        default=0,
        help="RDMA max CQEs per CQ; 0 = use backend default (default: 0)",
    )
    parser.add_argument(
        "--max-msg-sge",
        type=int,
        default=0,
        help="RDMA max SGEs per send WR; 0 = use backend default (default: 0)",
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
        batch_contiguous: bool = False,
        enable_sess: bool = False,
        iters: int = 128,
        sweep: bool = False,
        sweep_batch: bool = False,
        sweep_start_size: int = 8,
        sweep_max_size: int = 2**20,
        sweep_step: int = 0,
        backend_type: str = "rdma",
        host: str = "",
        port: int = 0,
        node_rank: int = 0,
        rank_in_node: int = 0,
        num_initiator_dev: int = 1,
        num_target_dev: int = 1,
        target_dev_offset: int = 0,
        num_qp_per_transfer: int = 4,
        num_worker_threads: int = 1,
        poll_cq_mode: str = "polling",
        max_send_wr: int = 0,
        max_cqe_num: int = 0,
        max_msg_sge: int = 0,
        enable_chunking: bool = True,
        chunk_bytes: int = 65536,
        max_chunks: int = 64,
        mem_type: str = "gpu",
        initiator_mem_type: str = None,
        target_mem_type: str = None,
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
        self.batch_contiguous = batch_contiguous
        self.enable_sess = enable_sess
        self.iters = iters
        self.sweep = sweep
        self.sweep_batch = sweep_batch
        self.sweep_start_size = sweep_start_size
        self.sweep_max_size = sweep_max_size
        self.sweep_step = sweep_step
        self.backend_type = backend_type

        self.host = host
        self.port = port
        self.node_rank = node_rank
        self.role_rank = rank_in_node
        self.num_initiator_dev = num_initiator_dev
        self.num_target_dev = num_target_dev
        self.target_dev_offset = target_dev_offset
        self.num_qp_per_transfer = num_qp_per_transfer
        self.num_worker_threads = num_worker_threads
        self.poll_cq_mode = (
            PollCqMode.POLLING if poll_cq_mode == "polling" else PollCqMode.EVENT
        )
        self.max_send_wr = max_send_wr
        self.max_cqe_num = max_cqe_num
        self.max_msg_sge = max_msg_sge
        self.enable_chunking = enable_chunking
        self.chunk_bytes = chunk_bytes
        self.max_chunks = max_chunks
        self.mem_type = mem_type
        self.initiator_mem_type = initiator_mem_type or mem_type
        self.target_mem_type = target_mem_type or mem_type

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
        elif self.backend_type == "fabric":
            self._setup_fabric()
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

        self.mem_type = (
            self.initiator_mem_type
            if self.role is EngineRole.INITIATOR
            else self.target_mem_type
        )

        # When not batch_contiguous, use strided offsets so buffer must fit (buffer_size+1)*transfer_batch_size
        total_elements = (
            (self.buffer_size + 1) * self.transfer_batch_size
            if not self.batch_contiguous
            else self.buffer_size * self.transfer_batch_size
        )
        if self.mem_type == "cpu":
            self.device = torch.device("cpu")
            self.tensor = torch.randint(0, 256, (total_elements,), dtype=torch.uint8)
        else:
            gpu_index = self.role_rank
            if self.role is EngineRole.TARGET and self.target_dev_offset:
                gpu_index = (
                    self.role_rank + self.target_dev_offset
                ) % torch.cuda.device_count()
            self.device = torch.device("cuda", gpu_index)
            self.tensor = torch.randn(total_elements).to(
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

            total_elements = (
                (self.buffer_size + 1) * self.transfer_batch_size
                if not self.batch_contiguous
                else self.buffer_size * self.transfer_batch_size
            )
            self.tensor = torch.randn(total_elements).to(
                self.device, dtype=torch.float8_e4m3fnuz
            )
        else:
            self.role = EngineRole.INITIATOR
            self.src_device = torch.device("cuda", self.src_gpu)
            self.dst_device = torch.device("cuda", self.dst_gpu)

            total_elements = (
                (self.buffer_size + 1) * self.transfer_batch_size
                if not self.batch_contiguous
                else self.buffer_size * self.transfer_batch_size
            )
            self.tensor = torch.randn(total_elements).to(
                self.src_device, dtype=torch.float8_e4m3fnuz
            )
            self.target_tensor = torch.zeros(total_elements).to(
                self.dst_device, dtype=torch.float8_e4m3fnuz
            )

    def _setup_fabric(self):
        # Fabric mirrors the RDMA cross-node role model (2 nodes, gloo OOB), but
        # the transferred buffer MUST be a fabric-exportable VMM allocation
        # (fabric_alloc) rather than a torch tensor.
        if self.mem_type != "gpu":
            raise ValueError("fabric backend supports GPU memory only")
        assert self.num_initiator_dev == self.num_target_dev
        self.world_size = self.num_initiator_dev + self.num_target_dev
        if self.node_rank == 0:
            self.global_rank = self.role_rank
            self.role = EngineRole.INITIATOR
        else:
            self.global_rank = self.role_rank + self.num_initiator_dev
            self.role = EngineRole.TARGET

        # 1 byte per element (matches the fp8/uint8 sizing used by the other paths).
        total_elements = (
            (self.buffer_size + 1) * self.transfer_batch_size
            if not self.batch_contiguous
            else self.buffer_size * self.transfer_batch_size
        )
        gpu_index = self.role_rank
        if self.role is EngineRole.TARGET and self.target_dev_offset:
            ndev = torch.cuda.device_count()
            if ndev <= 0:
                raise RuntimeError(
                    "fabric setup: torch.cuda.device_count()==0 (no visible GPUs); "
                    "cannot apply --target-dev-offset"
                )
            gpu_index = (self.role_rank + self.target_dev_offset) % ndev
        self.device = torch.device("cuda", gpu_index)
        self.fabric_nbytes = total_elements
        self.fabric_ptr = fabric_alloc(self.fabric_nbytes, gpu_index)
        if not self.fabric_ptr:
            raise RuntimeError(
                f"fabric_alloc failed for {self.fabric_nbytes} bytes on gpu {gpu_index}; "
                "the device must be a UALink fabric-capable GPU in a ready vPOD"
            )

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
        elif self.backend_type == "fabric":
            print(f"  node_rank: {self.node_rank}")
            print(f"  role: {self.role}")
            print(f"  role_rank: {self.role_rank}")
            print(f"  num_initiator_dev: {self.num_initiator_dev}")
            print(f"  num_target_dev: {self.num_target_dev}")
            print(f"  target_dev_offset: {self.target_dev_offset}")
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
            print(f"  target_dev_offset: {self.target_dev_offset}")
            print(f"  mem_type: {self.mem_type}")
            print(f"  num_qp_per_transfer: {self.num_qp_per_transfer}")
            print(f"  num_worker_threads: {self.num_worker_threads}")
            print(f"  enable_chunking: {self.enable_chunking}")
            if self.enable_chunking:
                print(f"  chunk_bytes: {self.chunk_bytes}")
                print(f"  max_chunks: {self.max_chunks}")
            print(f"  poll_cq_mode: {self.poll_cq_mode}")
            if self.max_send_wr or self.max_cqe_num or self.max_msg_sge:
                print(
                    f"  max_send_wr: {self.max_send_wr}, max_cqe_num: {self.max_cqe_num}, max_msg_sge: {self.max_msg_sge}"
                )

        print(f"  buffer_size: {self.buffer_size} B")
        print(f"  transfer_batch_size: {self.transfer_batch_size}")
        print(f"  enable_batch_transfer: {self.enable_batch_transfer}")
        print(f"  batch_contiguous: {self.batch_contiguous}")
        print(f"  enable_sess: {self.enable_sess}")
        print(f"  iters: {self.iters}")
        print()

    def _get_transfer_offsets(self, buffer_size, transfer_batch_size, batched):
        if batched and not self.batch_contiguous:
            stride = buffer_size + 1
            return [i * stride for i in range(transfer_batch_size)]
        return [i * buffer_size for i in range(transfer_batch_size)]

    def _pack_tensor_segments(self, tensor, buffer_size, transfer_batch_size, batched):
        offsets = self._get_transfer_offsets(
            buffer_size, transfer_batch_size, batched=batched
        )
        packed = torch.empty(
            buffer_size * transfer_batch_size,
            device=tensor.device,
            dtype=torch.uint8,
        )
        for i, offset in enumerate(offsets):
            end = offset + buffer_size
            packed[i * buffer_size : (i + 1) * buffer_size].copy_(
                tensor[offset:end].view(torch.uint8)
            )
        return packed

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
        elif self.backend_type == "fabric":
            self._validate_fabric()
        else:
            self._validate_rdma()

    def _validate_fabric(self):
        # Correctness spot-check decoupled from the (possibly strided/batched)
        # perf pattern: target fills a known pattern, initiator transfers it over
        # the fabric and both sides verify. Bounded to keep it cheap.
        check = min(self.fabric_nbytes, 1 << 20)
        tile = bytes(range(256))
        pattern = (tile * (check // 256 + 1))[:check]
        zero = bytes(check)
        dev = self.device.index

        if self.op_type == "read":
            if self.role is EngineRole.TARGET:
                hip_fill(self.fabric_ptr, pattern, dev)
                dist.barrier()  # #1 target data ready
                dist.barrier()  # #2 initiator finished
            else:
                hip_fill(self.fabric_ptr, zero, dev)
                dist.barrier()  # #1 wait for target fill
                status = self.engine.read(
                    self.mem,
                    0,
                    self.target_mem,
                    0,
                    check,
                    self.engine.allocate_transfer_uid(),
                )
                status.Wait()
                assert status.Succeeded(), f"fabric read failed: {status.Message()}"
                got = hip_read(self.fabric_ptr, check, dev)
                assert got == pattern, "fabric READ validation: data mismatch"
                dist.barrier()  # #2 done
        else:  # write
            if self.role is EngineRole.INITIATOR:
                hip_fill(self.fabric_ptr, pattern, dev)
                dist.barrier()  # #1 both ready
                status = self.engine.write(
                    self.mem,
                    0,
                    self.target_mem,
                    0,
                    check,
                    self.engine.allocate_transfer_uid(),
                )
                status.Wait()
                assert status.Succeeded(), f"fabric write failed: {status.Message()}"
                dist.barrier()  # #2 signal target to verify
            else:
                hip_fill(self.fabric_ptr, zero, dev)
                dist.barrier()  # #1 both ready
                dist.barrier()  # #2 initiator wrote
                got = hip_read(self.fabric_ptr, check, dev)
                assert got == pattern, "fabric WRITE validation: data mismatch"

    def _validate_rdma(self):
        if self.role is EngineRole.INITIATOR:
            recv_tensor = torch.empty(
                self.buffer_size * self.transfer_batch_size,
                device=self.device,
                dtype=torch.uint8,
            )
            dist.recv(recv_tensor, src=self.num_initiator_dev + self.role_rank)
            if not self.batch_contiguous:
                # Received data is packed (contiguous); compare to packed view of self.tensor
                stride = self.buffer_size + 1
                expected = torch.empty(
                    self.buffer_size * self.transfer_batch_size,
                    device=self.device,
                    dtype=torch.uint8,
                )
                for i in range(self.transfer_batch_size):
                    beg = i * stride
                    end = beg + self.buffer_size
                    expected[i * self.buffer_size : (i + 1) * self.buffer_size].copy_(
                        self.tensor[beg:end].view(torch.uint8)
                    )
                assert torch.equal(recv_tensor, expected)
            else:
                expected = self.tensor.view(torch.uint8)
                assert torch.equal(recv_tensor, expected)
        else:
            # Without batch_contiguous, tensor has (buffer_size+1)*transfer_batch_size
            # elements; Gloo send size must match initiator recv (buffer_size*transfer_batch_size).
            if not self.batch_contiguous:
                stride = self.buffer_size + 1
                packed = torch.empty(
                    self.buffer_size * self.transfer_batch_size,
                    device=self.device,
                    dtype=torch.uint8,
                )
                for i in range(self.transfer_batch_size):
                    beg = i * stride
                    end = beg + self.buffer_size
                    packed[i * self.buffer_size : (i + 1) * self.buffer_size].copy_(
                        self.tensor[beg:end].view(torch.uint8)
                    )
                dist.send(packed, dst=self.role_rank)
            else:
                int8_view = self.tensor.view(torch.uint8)
                dist.send(int8_view, dst=self.role_rank)

    def _validate_xgmi(self):
        if self.xgmi_multiprocess:
            # Target returns from run_once immediately, so sync here before reading
            # back the transferred segments for validation.
            dist.barrier()
            local_packed = self._pack_tensor_segments(
                self.tensor,
                self.buffer_size,
                self.transfer_batch_size,
                batched=self.enable_batch_transfer,
            ).cpu()
            if self.role is EngineRole.INITIATOR:
                peer_packed = torch.empty(
                    self.buffer_size * self.transfer_batch_size,
                    dtype=torch.uint8,
                )
                dist.recv(peer_packed, src=self.global_rank + 1)
                assert torch.equal(
                    local_packed, peer_packed
                ), "Validation failed: data mismatch"
            else:
                dist.send(local_packed, dst=self.global_rank - 1)
        else:
            self.run_once(self.buffer_size, self.transfer_batch_size)
            src_cpu = self._pack_tensor_segments(
                self.tensor,
                self.buffer_size,
                self.transfer_batch_size,
                batched=self.enable_batch_transfer,
            ).cpu()
            dst_cpu = self._pack_tensor_segments(
                self.target_tensor,
                self.buffer_size,
                self.transfer_batch_size,
                batched=self.enable_batch_transfer,
            ).cpu()
            assert torch.equal(src_cpu, dst_cpu), "Validation failed: data mismatch"

    def initialize(self):
        if self.backend_type == "xgmi":
            self._initialize_xgmi()
        elif self.backend_type == "fabric":
            self._initialize_fabric()
        else:
            self._initialize_rdma()

    def _initialize_rdma(self):
        config = IOEngineConfig(
            host=self.host,
            port=self.port,
        )
        self.engine = IOEngine(key=f"{self.role.name}-{self.role_rank}", config=config)
        config = RdmaBackendConfig(
            qp_per_transfer=self.num_qp_per_transfer,
            post_batch_size=-1,
            num_worker_threads=self.num_worker_threads,
            poll_cq_mode=self.poll_cq_mode,
            enable_notification=False,
            enable_transfer_chunking=self.enable_chunking,
            chunk_bytes=self.chunk_bytes,
            max_chunks_per_transfer=self.max_chunks,
        )
        if self.max_send_wr > 0:
            config.max_send_wr = self.max_send_wr
        if self.max_cqe_num > 0:
            config.max_cqe_num = self.max_cqe_num
        if self.max_msg_sge > 0:
            config.max_msg_sge = self.max_msg_sge
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

    def _initialize_fabric(self):
        # Same OOB/role flow as RDMA (gloo desc exchange), but a FABRIC backend
        # and register_memory() on the raw fabric_alloc pointer. The FABRIC
        # backend has no TCP control plane; descriptors are exchanged via gloo.
        config = IOEngineConfig(host="", port=0)
        self.engine = IOEngine(key=f"{self.role.name}-{self.role_rank}", config=config)
        fabric_config = FabricBackendConfig(
            num_streams=self.num_streams,
            num_events=self.num_events,
        )
        self.engine.create_backend(BackendType.FABRIC, fabric_config)

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

        self.mem = self.engine.register_memory(
            self.fabric_ptr,
            self.fabric_nbytes,
            self.device.index,
            MemoryLocationType.GPU,
        )

        if self.role is EngineRole.TARGET:
            mem_desc = self.mem.pack()
            self.send_bytes(mem_desc, self.role_rank)
        else:
            target_mem_desc = self.recv_bytes(self.num_initiator_dev + self.role_rank)
            self.target_mem = MemoryDesc.unpack(target_mem_desc)
            self.sess = self.engine.create_session(self.mem, self.target_mem)
            if self.sess is None:
                raise RuntimeError(
                    "create_session returned None for fabric: peers likely not in the "
                    "same vPOD, or remote memory is not fabric-exportable"
                )

    def run_single_once(self, buffer_size, transfer_batch_size):
        assert buffer_size <= self.buffer_size
        if (
            self.backend_type in ("rdma", "fabric")
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
            self.backend_type in ("rdma", "fabric")
            or (self.backend_type == "xgmi" and self.xgmi_multiprocess)
        ) and self.role is EngineRole.TARGET:
            return 0

        # Strided offsets prevent merging: each transfer becomes a separate WR (to stress SQ / reproduce notify ENOMEM)
        offsets = self._get_transfer_offsets(
            buffer_size, transfer_batch_size, batched=True
        )
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
            self.backend_type in ("rdma", "fabric")
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
        elif self.backend_type == "fabric":
            return f"FABRIC Benchmark: Initiator Rank {self.role_rank}"
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
                if self.backend_type in ("rdma", "fabric") or (
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
                if self.sweep_step > 0:
                    cur_size += self.sweep_step
                else:
                    cur_size *= 2
        elif self.sweep_batch:
            cur_transfer_batch_size = 1
            max_transfer_batch_size = 32768
            while cur_transfer_batch_size <= max_transfer_batch_size:
                if self.backend_type in ("rdma", "fabric") or (
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
            self._run_distributed()

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

    def _run_distributed(self):
        # Shared cross-node driver for RDMA and FABRIC (2-node gloo OOB).
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
            dist.barrier()
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
        batch_contiguous=args.batch_contiguous,
        enable_sess=args.enable_sess,
        iters=args.iters,
        sweep=args.all,
        sweep_batch=args.all_batch,
        sweep_start_size=args.sweep_start_size,
        sweep_max_size=args.sweep_max_size,
        sweep_step=args.sweep_step,
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
        batch_contiguous=args.batch_contiguous,
        enable_sess=args.enable_sess,
        iters=args.iters,
        sweep=args.all,
        sweep_batch=args.all_batch,
        sweep_start_size=args.sweep_start_size,
        sweep_max_size=args.sweep_max_size,
        sweep_step=args.sweep_step,
        backend_type=args.backend,  # "rdma" or "fabric" (both use this driver)
        host=args.host,
        port=0,
        node_rank=node_rank,
        rank_in_node=local_rank,
        num_initiator_dev=args.num_initiator_dev,
        num_target_dev=args.num_target_dev,
        target_dev_offset=args.target_dev_offset,
        num_qp_per_transfer=args.num_qp_per_transfer,
        num_worker_threads=args.num_worker_threads,
        poll_cq_mode=args.poll_cq_mode,
        max_send_wr=args.max_send_wr,
        max_cqe_num=args.max_cqe_num,
        max_msg_sge=args.max_msg_sge,
        enable_chunking=not args.disable_chunking,
        chunk_bytes=args.chunk_bytes,
        max_chunks=args.max_chunks,
        mem_type=args.mem_type,
        initiator_mem_type=args.initiator_mem_type,
        target_mem_type=args.target_mem_type,
        num_streams=args.num_streams,
        num_events=args.num_events,
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
            batch_contiguous=args.batch_contiguous,
            enable_sess=args.enable_sess,
            iters=args.iters,
            sweep=args.all,
            sweep_batch=args.all_batch,
            sweep_start_size=args.sweep_start_size,
            sweep_max_size=args.sweep_max_size,
            sweep_step=args.sweep_step,
            backend_type="xgmi",
            src_gpu=args.src_gpu,
            dst_gpu=args.dst_gpu,
            num_streams=args.num_streams,
            num_events=args.num_events,
            xgmi_multiprocess=False,
        )
        bench.print_config()
        bench.run()


def benchmark_distributed(args):
    # Cross-node driver shared by RDMA and FABRIC backends.
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
        benchmark_distributed(args)


if __name__ == "__main__":
    benchmark()
