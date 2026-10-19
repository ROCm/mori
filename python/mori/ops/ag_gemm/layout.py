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
"""Window layout for the all-gather + GEMM.

Each rank holds ``A_r [m_local, K]`` -- its own tokens, after a sequence-parallel
shard -- and its own weight shard ``B_r [N, K]``. It needs every rank's tokens
times its weight, ``C_r = concat_s(A_s) @ B_r.T`` of shape ``[world*m_local, N]``:
the all-gather that feeds a ColumnParallelLinear, fused with it.

The window holds the gathered operand, so the GEMM reads one ``[world*m_local,
K]`` tensor whatever the source of a row, and only the tile order and a wait
before each remote tile differ from a plain GEMM::

    [ control | recv: [world][m_local][K] fp8 | recv_scale: [world][m_local] fp32 ]

Rank ``r``'s own rows live in ``recv[r]``; the caller writes them there, and the
copy engine pushes the same bytes into slot ``r`` of every peer's window, in
``chunks`` row bands so a peer can start on band 0 while band 1 is still on the
wire.

Control area, all u32 and 128-byte aligned:

* ``epoch`` -- local, bumped once per call by the closing barrier. Every flag
  below is compared against ``epoch + 1``, so nothing is ever reset and a CUDA
  graph replays like a fresh launch.
* ``flag[src][chunk]`` -- written *by* ``src`` into this window when that band
  of its rows has landed here.
* ``bar[2][src]`` -- two cross-rank barriers per call, written by ``src``. The
  split path needs both (before the GEMM: everything landed; after it: nobody
  still reads a slot the next call refills); the fused path only the second.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_WORLD = 8
MAX_CHUNKS = 16
ALIGN = 128
#: The ptpc A scale is one fp32 per row, so it can ride along with the rows.
SCALE_BYTES = 4
FP8_BYTES = 1


def _align(value: int, alignment: int = ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class AgGemmConfig:
    world_size: int
    m_local: int
    k: int
    block_m: int = 256
    chunks: int = 1

    @property
    def m(self) -> int:
        """Rows of the gathered operand, and of C."""
        return self.world_size * self.m_local

    @property
    def m_tiles_local(self) -> int:
        return self.m_local // self.block_m

    @property
    def m_tiles_per_chunk(self) -> int:
        return self.m_tiles_local // self.chunks

    @property
    def chunk_rows(self) -> int:
        return self.m_local // self.chunks

    # --- control ---

    @property
    def epoch_off(self) -> int:
        return 0

    @property
    def flag_off(self) -> int:
        return ALIGN

    def flag_slot_off(self, src: int, chunk: int) -> int:
        if not 0 <= src < self.world_size:
            raise IndexError(f"src {src} outside [0, {self.world_size})")
        if not 0 <= chunk < self.chunks:
            raise IndexError(f"chunk {chunk} outside [0, {self.chunks})")
        return self.flag_off + (src * MAX_CHUNKS + chunk) * 4

    @property
    def bar_off(self) -> int:
        return _align(self.flag_off + MAX_WORLD * MAX_CHUNKS * 4)

    def bar_slot_off(self, which: int, src: int) -> int:
        if which not in (0, 1):
            raise IndexError(f"barrier {which} is not 0 or 1")
        return self.bar_off + (which * MAX_WORLD + src) * 4

    @property
    def control_bytes(self) -> int:
        return _align(self.bar_off + 2 * MAX_WORLD * 4)

    # --- payload ---

    @property
    def slab_bytes(self) -> int:
        """One rank's rows: what it pushes to each peer per call."""
        return self.m_local * self.k * FP8_BYTES

    @property
    def chunk_bytes(self) -> int:
        return self.slab_bytes // self.chunks

    @property
    def recv_off(self) -> int:
        # 4 KiB so the fp8 buffer descriptor and the copy engine both see a
        # page-aligned operand.
        return _align(self.control_bytes, 4096)

    def recv_slot_off(self, src: int) -> int:
        return self.recv_off + src * self.slab_bytes

    @property
    def scale_slab_bytes(self) -> int:
        return self.m_local * SCALE_BYTES

    @property
    def chunk_scale_bytes(self) -> int:
        return self.scale_slab_bytes // self.chunks

    @property
    def scale_off(self) -> int:
        return _align(self.recv_off + self.world_size * self.slab_bytes, 4096)

    def scale_slot_off(self, src: int) -> int:
        return self.scale_off + src * self.scale_slab_bytes

    @property
    def window_bytes(self) -> int:
        return _align(self.scale_off + self.world_size * self.scale_slab_bytes, 4096)

    @property
    def remote_bytes_per_rank(self) -> int:
        """Bytes this rank sends per call: its rows and scales to every peer."""
        return (self.world_size - 1) * (self.slab_bytes + self.scale_slab_bytes)

    def validate(self) -> None:
        if not 2 <= self.world_size <= MAX_WORLD:
            raise ValueError(
                f"world_size must be in [2, {MAX_WORLD}], got {self.world_size}"
            )
        if self.block_m < 128 or self.block_m % 128:
            raise ValueError(
                f"block_m={self.block_m} must be a positive multiple of 128"
            )
        if not 1 <= self.chunks <= MAX_CHUNKS:
            raise ValueError(f"chunks must be in [1, {MAX_CHUNKS}], got {self.chunks}")
        if self.m_local < 1 or self.m_local % (self.block_m * self.chunks):
            raise ValueError(
                f"m_local={self.m_local} must be a positive multiple of "
                f"block_m*chunks={self.block_m * self.chunks}: a chunk is a whole "
                f"number of row tiles, and a remote tile waits on exactly one chunk"
            )
        if self.k < 256 or self.k % 128:
            raise ValueError(f"K={self.k} must be a multiple of 128 and at least 256")


__all__ = ["AgGemmConfig", "MAX_WORLD", "MAX_CHUNKS"]
