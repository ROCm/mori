# /root/mori/python/mori/ccl/collective.py
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

from mori import cpp as mori_cpp
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import torch
import torch.distributed as dist


# 枚举定义
class All2allMode(Enum):
    SYNC = 0
    ASYNC = 1


class CollectiveStatusCode(Enum):
    SUCCESS = 0
    INVALID_ARGUMENT = 1
    MEMORY_ERROR = 2
    COMMUNICATION_ERROR = 3


# 配置类
@dataclass
class All2allConfig:
    rank: int = 0
    world_size: int = 1
    use_async: bool = False
    verify_results: bool = False
    timeout_ms: int = 1000


# 结果类
@dataclass
class All2allResult:
    output_tensor: Optional[torch.Tensor] = None
    duration_seconds: float = 0.0
    bandwidth_gbps: float = 0.0
    status: CollectiveStatusCode = CollectiveStatusCode.SUCCESS
    error_message: str = ""


# All2All操作符类
class All2allSdmaOp:
    """Operator for All2All SDMA operations."""

    def __init__(self, config: Optional[All2allConfig] = None):
        self.config = config or All2allConfig()
        self._stream = None

        # 初始化分布式环境（如果未初始化）
        if not dist.is_initialized():
            try:
                dist.init_process_group(backend='nccl')
            except:
                try:
                    dist.init_process_group(backend='mpi')
                except:
                    dist.init_process_group(backend='gloo')

        # 更新配置中的实际rank和world_size
        self.config.rank = dist.get_rank()
        self.config.world_size = dist.get_world_size()

        if self.config.use_async:
            self._stream = torch.cuda.Stream()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
        stream: Optional[Union[torch.cuda.Stream, int]] = None,
    ) -> All2allResult:
        """Execute All2All operation."""

        # 验证输入
        if not input_tensor.is_cuda:
            return All2allResult(
                status=CollectiveStatusCode.INVALID_ARGUMENT,
                error_message="Input tensor must be on GPU"
            )

        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        local_count = input_tensor.numel()

        # 准备输出张量
        if output_tensor is None:
            total_count = local_count * self.config.world_size
            output_tensor = torch.empty(
                total_count,
                dtype=input_tensor.dtype,
                device=input_tensor.device
            )
        else:
            # 验证输出张量
            expected_count = local_count * self.config.world_size
            if output_tensor.numel() != expected_count:
                return All2allResult(
                    status=CollectiveStatusCode.INVALID_ARGUMENT,
                    error_message=f"Output tensor size mismatch. Expected {expected_count}, got {output_tensor.numel()}"
                )

            if output_tensor.dtype != input_tensor.dtype:
                return All2allResult(
                    status=CollectiveStatusCode.INVALID_ARGUMENT,
                    error_message="Output tensor dtype must match input tensor dtype"
                )

            if not output_tensor.is_contiguous():
                output_tensor = output_tensor.contiguous()

        # 准备流
        if stream is None:
            if self.config.use_async and self._stream is not None:
                stream_obj = self._stream
                hip_stream = stream_obj.cuda_stream
            else:
                stream_obj = torch.cuda.current_stream()
                hip_stream = stream_obj.cuda_stream
        elif isinstance(stream, torch.cuda.Stream):
            stream_obj = stream
            hip_stream = stream.cuda_stream
        else:
            # 假设是HIP流指针
            stream_obj = None
            hip_stream = stream

        # 确保所有进程就绪
        dist.barrier()

        try:
            # 根据数据类型调用相应的C++函数
            dtype = input_tensor.dtype
            duration = 0.0

            if dtype == torch.float32:
                duration = mori_cpp.all2all_sdma_float32(
                    input_tensor.data_ptr(),
                    output_tensor.data_ptr(),
                    local_count,
                    hip_stream
                )
            elif dtype == torch.int32:
                duration = mori_cpp.all2all_sdma_int32(
                    input_tensor.data_ptr(),
                    output_tensor.data_ptr(),
                    local_count,
                    hip_stream
                )
            elif dtype == torch.uint32:
                duration = mori_cpp.all2all_sdma_uint32(
                    input_tensor.data_ptr(),
                    output_tensor.data_ptr(),
                    local_count,
                    hip_stream
                )
            else:
                return All2allResult(
                    status=CollectiveStatusCode.INVALID_ARGUMENT,
                    error_message=f"Unsupported dtype: {dtype}"
                )

            # 等待完成
            if stream_obj is not None:
                stream_obj.synchronize()
            torch.cuda.synchronize()

            # 最终同步
            dist.barrier()

            # 计算带宽
            data_size_bytes = output_tensor.numel() * input_tensor.element_size()
            bandwidth_gbps = data_size_bytes / duration / (1024**3)

            # 验证结果（如果需要）
            if self.config.verify_results:
                self._verify_result(input_tensor, output_tensor, local_count)

            return All2allResult(
                output_tensor=output_tensor,
                duration_seconds=duration,
                bandwidth_gbps=bandwidth_gbps,
                status=CollectiveStatusCode.SUCCESS
            )

        except Exception as e:
            return All2allResult(
                status=CollectiveStatusCode.COMMUNICATION_ERROR,
                error_message=str(e)
            )

    def _verify_result(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        local_count: int
    ):
        """验证All2All结果（简化版本）"""
        torch.cuda.synchronize()

        output_reshaped = output_tensor.view(self.config.world_size, local_count)

        # 简化验证：检查每个块是否非零
        for i in range(self.config.world_size):
            chunk = output_reshaped[i]
            if torch.all(chunk == 0):
                print(f"Warning: Rank {self.config.rank} received zero data from rank {i}")

    def benchmark(
        self,
        local_count: int,
        iterations: int = 10,
        warmup: int = 2
    ) -> dict:
        """运行性能基准测试。"""
        dtype = torch.float32

        # 准备数据
        input_tensor = torch.full(
            (local_count,),
            float(self.config.rank),
            dtype=dtype,
            device='cuda'
        )

        # 预热
        for _ in range(warmup):
            result = self(input_tensor)
            if result.status != CollectiveStatusCode.SUCCESS:
                return {"error": result.error_message}

        # 基准测试
        durations = []
        for i in range(iterations):
            result = self(input_tensor)
            if result.status != CollectiveStatusCode.SUCCESS:
                return {"error": result.error_message}
            durations.append(result.duration_seconds)

        # 计算统计信息
        avg_duration = sum(durations) / len(durations)
        data_size_bytes = local_count * self.config.world_size * 4  # float32 = 4 bytes
        bandwidth_gbps = data_size_bytes / avg_duration / (1024**3)

        return {
            "local_count": local_count,
            "world_size": self.config.world_size,
            "iterations": iterations,
            "avg_duration_seconds": avg_duration,
            "min_duration_seconds": min(durations),
            "max_duration_seconds": max(durations),
            "bandwidth_gbps": bandwidth_gbps
        }


# 便捷函数
def all2all_sdma(
    input_tensor: torch.Tensor,
    output_tensor: Optional[torch.Tensor] = None,
    stream: Union[torch.cuda.Stream, int] = 0,
    config: Optional[All2allConfig] = None
) -> All2allResult:
    """便捷函数：创建操作符并执行All2All。"""
    op = All2allSdmaOp(config)
    return op(input_tensor, output_tensor, stream)


def create_all2all_op(config: Optional[All2allConfig] = None) -> All2allSdmaOp:
    """创建All2All操作符。"""
    return All2allSdmaOp(config)
