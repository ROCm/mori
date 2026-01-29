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
import os
import torch
import torch.distributed as dist
import socket
from multiprocessing import Queue
import mori
import traceback


str_to_dtype = {
    "float32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "half": torch.float16,
    "int8": torch.int8,
    "int16": torch.int16,
    "short": torch.int16,
    "int32": torch.int32,
    "int": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def string_to_dtype(s):
    s = s.lower()
    if s not in str_to_dtype:
        raise ValueError(f"Unknown dtype string: {s}")
    return str_to_dtype[s]


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def data_type_supported(dtype):
    arch = torch.cuda.get_device_capability(0)
    arch_int = int("".join(map(str, arch)))
    if dtype is torch.float8_e4m3fnuz:
        return arch_int == 94
    if dtype is torch.float8_e4m3fn:
        return arch_int >= 95
    return True


class TorchDistContext:
    def __init__(
        self,
        rank,
        world_size,
        master_addr="localhost",
        master_port="12335",
        device_id=None,
        backend="cpu:gloo,cuda:nccl",
    ):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.device_id = device_id if device_id is not None else self.rank
        self.backend = backend

    def __enter__(self):
        if self.master_addr is not None:
            os.environ["MASTER_ADDR"] = self.master_addr
        if self.master_port is not None:
            os.environ["MASTER_PORT"] = str(self.master_port)

        torch.cuda.set_device(self.device_id)
        device = torch.device("cuda", self.device_id)

        dist.init_process_group(
            backend=self.backend,
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

    @staticmethod
    def _worker(rank, world_size, port, init_shmem, task_queue, result_queue):
        with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
            if init_shmem:
                mori.shmem.shmem_torch_process_group_init("default")
            while True:
                task = task_queue.get()
                if task == "STOP":
                    if init_shmem:
                        mori.shmem.shmem_finalize()
                    break
                func, args = task
                try:
                    result = func(rank, *args)
                    result_queue.put((rank, result))
                except Exception:
                    result_queue.put((rank, traceback.format_exc()))

    def start_workers(self, world_size):
        port = get_free_port()
        self.processes = [
            torch.multiprocessing.Process(
                target=self._worker,
                args=(
                    rank,
                    world_size,
                    port,
                    self.init_mori_shmem,
                    self.task_queue,
                    self.result_queue,
                ),
            )
            for rank in range(world_size)
        ]
        for p in self.processes:
            p.start()

    def shutdown(self):
        for _ in range(len(self.processes)):
            self.task_queue.put("STOP")
        for p in self.processes:
            p.join()


# E2M1 lookup table (8 non-negative values)
# 2-bit exponent + 1-bit mantissa, implicit leading 1
# Actual value = 1.mantissa * 2^(exponent - bias)
# bias = 1, so exponent values 0,1,2,3 correspond to actual exponents -1, 0, 1, 2
_E2M1_TO_FP32 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def fp4_x2_to_fp32(x: torch.Tensor) -> torch.Tensor:
    """
    Decode float4_e2m1fn_x2 (packed uint8) into float32.

    Args:
        x: torch.Tensor, dtype torch.float4_e2m1fn_x2 or torch.uint8
           shape: [...]
           Each element contains 2 packed FP4 values.

    Returns:
        torch.Tensor, shape: [...], dtype torch.float32
        The output length is twice the input length
        (each byte is decoded into 2 float values).
    """
    # Ensure the tensor is viewed as uint8
    if x.dtype == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)
    else:
        x = x.to(torch.uint8)

    # Get device information
    device = x.device

    # Flatten for processing
    flat = x.flatten()

    # Split into two nibbles (4-bit values)
    low_nibble = (flat & 0x0F).long()  # lower 4 bits: first FP4 value
    high_nibble = (flat >> 4).long()  # upper 4 bits: second FP4 value

    # Decode sign bit and absolute-value index for each nibble
    # Sign bit is bit 3 (0x8), absolute-value index is the lower 3 bits (0x7)
    low_sign = (low_nibble & 0x8).bool()
    low_abs_idx = low_nibble & 0x7

    high_sign = (high_nibble & 0x8).bool()
    high_abs_idx = high_nibble & 0x7

    # Lookup absolute values from the table (move table to target device)
    _table = _E2M1_TO_FP32.to(device)
    low_val = _table[low_abs_idx]
    high_val = _table[high_abs_idx]

    # Apply sign
    low_val = torch.where(low_sign, -low_val, low_val)
    high_val = torch.where(high_sign, -high_val, high_val)

    # Interleave results:
    # [low_val[0], high_val[0], low_val[1], high_val[1], ...]
    result = torch.stack([low_val, high_val], dim=1).flatten()

    return result


# FP4 E2M1 可表示的 15 个非零值（按大小排序）
# 格式: 1.m * 2^(e-1), 其中 e=指数(2位), m=尾数(1位)
# 特殊值: 0.0 (所有位为0)
_FP4_VALUES = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)

# 对应的 4-bit 编码 (绝对值索引 + 符号位)
# 绝对值索引: 0->0.0(特殊), 1->0.5, 2->1.0, 3->1.5, 4->2.0, 5->3.0, 6->4.0, 7->6.0
# 符号位: 0=正, 1=负 (第 3 位)
_FP4_CODES = torch.tensor(
    [
        0x8 | 0x7,  # -6.0  (1 111)
        0x8 | 0x6,  # -4.0  (1 110)
        0x8 | 0x5,  # -3.0  (1 101)
        0x8 | 0x4,  # -2.0  (1 100)
        0x8 | 0x3,  # -1.5  (1 011)
        0x8 | 0x2,  # -1.0  (1 010)
        0x8 | 0x1,  # -0.5  (1 001)
        0x0,  #  0.0  (0 000)
        0x1,  #  0.5  (0 001)
        0x2,  #  1.0  (0 010)
        0x3,  #  1.5  (0 011)
        0x4,  #  2.0  (0 100)
        0x5,  #  3.0  (0 101)
        0x6,  #  4.0  (0 110)
        0x7,  #  6.0  (0 111)
    ],
    dtype=torch.uint8,
)


def fp32_to_fp4_x2(x: torch.Tensor) -> torch.Tensor:
    """
    将 FP32 一维张量量化为 packed FP4 (float4_e2m1fn_x2)

    Args:
        x: torch.Tensor, dtype=torch.float32, shape: [N]
           注意: N 必须是偶数（因为是 2 个值打包在一起）

    Returns:
        torch.Tensor, dtype=torch.uint8 (可作为 float4_e2m1fn_x2 的存储), shape: [N//2]
    """
    if x.dim() != 1:
        raise ValueError(f"输入必须是一维张量，当前维度: {x.dim()}")

    if x.shape[0] % 2 != 0:
        raise ValueError(f"输入长度必须是偶数，当前长度: {x.shape[0]}")

    device = x.device
    x = x.contiguous().to(torch.float32)
    # N = x.shape[0]

    # 量化到最近的 FP4: 使用 searchsorted 或 abs diff
    # 方法: 计算与每个 FP4 值的绝对差，取最小
    # 为了效率，对于大 tensor 可以用向量化搜索

    # 简单方法：查找表搜索 (适用于任意大小)
    # 扩展维度用于广播: [N, 1] vs [15]
    diff = torch.abs(x.unsqueeze(1) - _FP4_VALUES.to(device))
    nearest_indices = torch.argmin(diff, dim=1)  # [N], 值范围 0-14

    # 获取对应的 4-bit 编码 (0-15)
    codes = _FP4_CODES.to(device)[nearest_indices]  # [N], dtype=uint8

    # 打包: 两个 4-bit 合并为一个 8-bit
    # 输入: [val0, val1, val2, val3, ...]
    # 输出: [(val1<<4)|val0, (val3<<4)|val2, ...]
    even_codes = codes[0::2]  # val0, val2, ...
    odd_codes = codes[1::2]  # val1, val3, ...

    packed = (odd_codes << 4) | even_codes  # [N//2]

    return packed
