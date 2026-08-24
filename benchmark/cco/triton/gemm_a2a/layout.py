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

from dataclasses import dataclass


@dataclass(frozen=True)
class GemmA2AConfig:
    world_size: int
    m: int
    n: int
    k: int
    shard_n: int
    block_m: int = 256
    block_n: int = 128
    block_k: int = 64

    @property
    def scatter_n(self) -> int:
        return self.world_size * self.shard_n

    @property
    def slab_elements(self) -> int:
        return self.m * self.shard_n

    @property
    def staging_elements(self) -> int:
        return self.world_size * self.slab_elements

    @property
    def recv_elements(self) -> int:
        return self.world_size * self.slab_elements

    @property
    def staging_bytes(self) -> int:
        return self.staging_elements * 2

    @property
    def recv_bytes(self) -> int:
        return self.recv_elements * 2

    @property
    def tail_elements(self) -> int:
        return self.m * self.n

    @property
    def remote_bytes_per_rank(self) -> int:
        return (self.world_size - 1) * self.slab_elements * 2

    @property
    def gemm_grid(self) -> int:
        return (self.m // self.block_m) * (self.n // self.block_n)

    def staging_index(self, dst_rank: int, row: int, local_col: int) -> int:
        return dst_rank * self.slab_elements + row * self.shard_n + local_col

    def recv_index(self, src_rank: int, row: int, local_col: int) -> int:
        return src_rank * self.slab_elements + row * self.shard_n + local_col

    def validate(self) -> None:
        if self.world_size < 2:
            raise ValueError("GEMM+A2A requires at least two ranks")
        if min(self.m, self.n, self.k, self.shard_n) <= 0:
            raise ValueError("M, N, K, and shard_n must be positive")
        if self.scatter_n > self.n:
            raise ValueError("world_size * shard_n must not exceed N")
        if self.m % self.block_m:
            raise ValueError("M must be divisible by BLOCK_M")
        if self.n % self.block_n:
            raise ValueError("N must be divisible by BLOCK_N")
        if self.k % self.block_k:
            raise ValueError("K must be divisible by BLOCK_K")
        if self.shard_n % self.block_n:
            raise ValueError("shard_n must be divisible by BLOCK_N")
        for value, name in (
            (self.block_m, "BLOCK_M"),
            (self.block_n, "BLOCK_N"),
            (self.block_k, "BLOCK_K"),
        ):
            if value & (value - 1):
                raise ValueError(f"{name} must be a power of two")


__all__ = ["GemmA2AConfig"]
