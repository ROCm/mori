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
"""Window layout for the all-to-all + GEMM.

Under Ulysses sequence parallelism, attention ends with every rank holding all
``world*m_local`` tokens for its own ``K/world`` columns of heads. The
o-projection needs the opposite: this rank's ``m_local`` tokens over all ``K``
columns. Rank ``r`` therefore sends row block ``d`` of its head shard to rank
``d``, and rank ``d`` computes::

    C_d = concat_s(A_s[d*m_local:(d+1)*m_local]) @ B.T     # [m_local, N]

where the concatenation is along K, source ``s`` owning columns
``[s*k_shard, (s+1)*k_shard)``. ``B [N, K]`` is replicated.

The window keeps what the caller sends and what peers send it side by side, in
the same per-source layout::

    [ control | send: [world][m_local][k_shard] fp8 | recv: [world][m_local][k_shard] fp8 ]

``send[d]`` is the row block bound for rank ``d`` -- a contiguous slab, which is
why the head shard can go out as one put per destination and band. ``recv[s]``
is what source ``s`` sent here. The GEMM reads A one K segment at a time from
whichever of the two holds it: ``send[rank]`` for this rank's own columns, which
never travel, and ``recv[s]`` for every other source. The A scale is one fp32
per token over *all* K, so it is computed from the gathered row maxima before
quantization (as ATOM's fp8 head exchange does) and never needs to travel.

Control area, all u32 and 128-byte aligned:

* ``epoch`` -- local, bumped once per call by the closing barrier. Every flag
  below is compared against ``epoch + 1``, so nothing is ever reset and a CUDA
  graph replays like a fresh launch.
* ``sig_base[src]`` -- the low 32 bits of the copy engine's signal counter for
  ``src`` when the window was armed (``build_phases()["arm"]``). Every band
  ``src`` pushes here bumps that counter by one, after its bytes, on the same
  queue; the counter is never reset, so a band's arrival is "counter reaches
  ``sig_base + (epoch - 1) * chunks + band + 1``".
* ``bar[2][src]`` -- two cross-rank barriers per call, written by ``src``. The
  split path needs both (before the GEMM: everything landed; after it: nobody
  still reads a slot the next call refills); the fused path only the second.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_WORLD = 8
MAX_CHUNKS = 16
ALIGN = 128
FP8_BYTES = 1
#: The GEMM's K step; a source's K segment is a whole number of them.
BLOCK_K = 128


def _align(value: int, alignment: int = ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class A2aGemmConfig:
    world_size: int
    m_local: int
    k: int
    block_m: int = 256
    chunks: int = 1
    #: Diagnostic: reserve a stamp region after ``recv`` (see ``stamp_slots``).
    timeline: bool = False

    @property
    def k_shard(self) -> int:
        """Columns each source owns: one head shard."""
        return self.k // self.world_size

    @property
    def k_steps_per_shard(self) -> int:
        return self.k_shard // BLOCK_K

    @property
    def m_tiles(self) -> int:
        return self.m_local // self.block_m

    @property
    def m_tiles_per_chunk(self) -> int:
        return self.m_tiles // self.chunks

    # --- control ---

    @property
    def epoch_off(self) -> int:
        return 0

    @property
    def sig_base_off(self) -> int:
        return ALIGN

    def sig_base_slot_off(self, src: int) -> int:
        if not 0 <= src < self.world_size:
            raise IndexError(f"src {src} outside [0, {self.world_size})")
        return self.sig_base_off + src * 4

    @property
    def bar_off(self) -> int:
        return _align(self.sig_base_off + MAX_WORLD * 4)

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
        """One (source, destination) block: ``[m_local, k_shard]`` fp8."""
        return self.m_local * self.k_shard * FP8_BYTES

    @property
    def chunk_bytes(self) -> int:
        return self.slab_bytes // self.chunks

    @property
    def send_off(self) -> int:
        # 4 KiB so the fp8 buffer descriptor and the copy engine both see a
        # page-aligned operand.
        return _align(self.control_bytes, 4096)

    def send_slot_off(self, dst: int) -> int:
        return self.send_off + dst * self.slab_bytes

    @property
    def recv_off(self) -> int:
        return _align(self.send_off + self.world_size * self.slab_bytes, 4096)

    def recv_slot_off(self, src: int) -> int:
        return self.recv_off + src * self.slab_bytes

    @property
    def recv_end(self) -> int:
        return self.recv_off + self.world_size * self.slab_bytes

    # --- diagnostic timeline: u64 s_memrealtime stamps ---
    #: [0] block 0 start, [1] its puts issued, [2] its quiet done,
    #: [3 + src*MAX_CHUNKS + c] band c from src observed here,
    #: then a (start, end) pair per tile.
    STAMP_FIXED = 3 + MAX_WORLD * MAX_CHUNKS

    @property
    def stamp_off(self) -> int:
        return _align(self.recv_end, 4096)

    def stamp_slots(self, tiles: int) -> int:
        return self.STAMP_FIXED + 2 * tiles

    @property
    def window_bytes(self) -> int:
        if not self.timeline:
            return _align(self.recv_end, 4096)
        # Room for the tile pairs of any N up to 64 column tiles.
        return _align(self.stamp_off + 8 * self.stamp_slots(self.m_tiles * 64), 4096)

    @property
    def operand_bytes(self) -> int:
        """The span the GEMM's A descriptor covers: ``send`` through ``recv``."""
        return self.recv_end - self.send_off

    def segment_off(self, src: int, rank: int, *, self_from_send: bool) -> int:
        """Where source ``src``'s K segment starts, relative to ``send_off``."""
        if src == rank and self_from_send:
            return self.send_slot_off(rank) - self.send_off
        return self.recv_slot_off(src) - self.send_off

    @property
    def remote_bytes_per_rank(self) -> int:
        """Bytes this rank sends per call: one slab to every peer."""
        return (self.world_size - 1) * self.slab_bytes

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
                f"number of row tiles, and a tile waits on exactly one chunk"
            )
        if self.k % (self.world_size * BLOCK_K):
            raise ValueError(
                f"K={self.k} must be a multiple of world_size*{BLOCK_K}="
                f"{self.world_size * BLOCK_K}: a source's K segment is a whole "
                f"number of K steps"
            )
        if self.k < 2 * BLOCK_K:
            raise ValueError(f"K={self.k} must be at least {2 * BLOCK_K}")
        if self.operand_bytes >= 1 << 31:
            raise ValueError(
                f"send+recv span {self.operand_bytes} bytes; the A buffer "
                f"descriptor takes 32-bit offsets"
            )


__all__ = ["A2aGemmConfig", "BLOCK_K", "MAX_CHUNKS", "MAX_WORLD"]
