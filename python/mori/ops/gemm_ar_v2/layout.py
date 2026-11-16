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
"""Window layout and shape arithmetic for ``gemm_ar_v2``. Pure Python, no GPU.

All-reduce = reduce-scatter by row bands + all-gather. With ``Mp`` the padded M
(a multiple of ``world * BLOCK_M``) and ``Ms = Mp / world``, band ``d`` is rows
``[d*Ms, (d+1)*Ms)`` and rank ``d`` reduces it. Window::

    [ ctrl | Y : [Mp_cap, N] bf16 | R : world slots x [Ms_cap, N] bf16 ]

``Y`` holds this rank's GEMM output (its partial C); the reduce writes band
``rank`` in place and the peers' gathers fill the other bands, so ``Y[:M]`` is
the result. ``R[s]`` receives source ``s``'s partial of band ``rank``.
"""

from __future__ import annotations

from dataclasses import dataclass

BLOCK_M = 256
BLOCK_N = 256
MAX_WORLD = 8
MAX_CHUNKS = 8
CTRL_BYTES = 4096
LINE = 128

# ctrl region, byte offsets
FS_OFF = 0  # scatter-barrier epoch (calls whose scatter drain finished)
FG_OFF = 64  # gather-barrier epoch (calls finished)
S_OFF = 1 * LINE  # S[src]: scatter arrivals, written by the peers
G_OFF = 2 * LINE  # G[src]: gather arrivals, written by the peers
BCTR_OFF = 3 * LINE  # counter base (0 unless debug_preset_counters)
CTR_OFF = 4 * LINE  # C[dest][chunk] tile counters (fused)
RG_CTR_OFF = 7 * LINE  # reduce-chunk counters (rg_fuse)
#: every u32 the kernels keep running (debug_preset_counters sets all of them)
EPOCH_WORDS = (
    [FS_OFF, FG_OFF]
    + [S_OFF + 4 * i for i in range(MAX_WORLD)]
    + [G_OFF + 4 * i for i in range(MAX_WORLD)]
)
COUNTER_WORDS = (
    [BCTR_OFF]
    + [CTR_OFF + 4 * i for i in range(MAX_WORLD * MAX_CHUNKS)]
    + [RG_CTR_OFF + 4 * i for i in range(MAX_CHUNKS)]
)


def _up(x: int, g: int) -> int:
    return (x + g - 1) // g * g


def padded_m(m: int, world: int) -> int:
    """M rounded up so every band is a whole number of row tiles."""
    return _up(max(m, 1), world * BLOCK_M)


def default_chunks(band_tiles: int, cap: int = MAX_CHUNKS) -> int:
    """Largest divisor of the band's row tiles not above ``cap``."""
    for c in range(min(cap, band_tiles), 0, -1):
        if band_tiles % c == 0:
            return c
    return 1


@dataclass(frozen=True)
class Layout:
    world: int
    n: int
    m_cap: int  # padded m_max

    @property
    def y_off(self) -> int:
        return CTRL_BYTES

    @property
    def y_bytes(self) -> int:
        return self.m_cap * self.n * 2

    @property
    def r_off(self) -> int:
        return self.y_off + self.y_bytes

    @property
    def slot_bytes(self) -> int:
        return (self.m_cap // self.world) * self.n * 2

    @property
    def window_bytes(self) -> int:
        return self.r_off + self.world * self.slot_bytes


@dataclass(frozen=True)
class Shape:
    """Per-call geometry for padded ``mp``."""

    world: int
    n: int
    mp: int
    chunks: int

    @property
    def ms(self) -> int:
        return self.mp // self.world

    @property
    def band_tiles(self) -> int:
        return self.ms // BLOCK_M

    @property
    def n_blocks(self) -> int:
        return self.n // BLOCK_N

    @property
    def mpc(self) -> int:
        """Row tiles per (destination, chunk)."""
        return self.band_tiles // self.chunks

    @property
    def band_bytes(self) -> int:
        return self.ms * self.n * 2

    @property
    def chunk_bytes(self) -> int:
        return self.mpc * BLOCK_M * self.n * 2

    @property
    def tiles_per_unit(self) -> int:
        """Tiles of one (destination, chunk): the fused election's target."""
        return self.mpc * self.n_blocks

    @property
    def tiles_per_chunk(self) -> int:
        return self.world * self.tiles_per_unit

    def validate(self) -> None:
        assert self.mp % (self.world * BLOCK_M) == 0, self.mp
        assert self.n % BLOCK_N == 0, self.n
        assert 1 <= self.chunks <= MAX_CHUNKS and self.band_tiles % self.chunks == 0
        assert self.world <= MAX_WORLD
