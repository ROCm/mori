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
"""Window layout and launch geometry for the FlyDSL cco GEMM+all-to-all.

All of the arithmetic the kernels depend on lives here so it can be unit tested
without a GPU: the symmetric-window offsets, the staging and receive index maps,
the completion-counter geometry, and the shape rules.

## What the operation is

Every rank holds its own ``A [M, K]`` and a replicated ``B [N, K]``, computes the
whole ``C = A @ B.T`` as ``[M, N]`` bf16, and sends column block ``j`` --
``cols [j*shard_n, (j+1)*shard_n)`` -- to rank ``j``. Rank ``d`` ends up with

    recv[d] = [world*M, shard_n]   rows [r*M, (r+1)*M) contributed by rank r

including its own contribution at row block ``d``.

``shard_n = N / world_size``. Full-N only: every column is sent somewhere.

## Why the sharded dimension is the interesting part

``gemm_ar``'s ``ArConfig`` says, of the all-reduce:

    ``n`` ... is never sharded -- the reduce-scatter shards along ``m`` so every
    peer slice is one contiguous byte range. Sharding along ``n`` would make each
    slice ``m`` separate ``n/world*2``-byte pieces, which is fatal for SDMA
    (~2us per packet regardless of size).

This operation *does* shard along ``n``, and that warning is exactly the problem
it has to solve. Two things make it tractable:

* The **receiver** layout is compact. A source's contribution occupies
  ``m * shard_n`` consecutive elements of the destination's window, so the
  landing side is one contiguous range per (source, destination) pair.
* The **sender** side is made contiguous by staging. The GEMM writes
  ``[dst][M][shard_n]`` rather than ``[M][N]``, which costs nothing because the
  epilogue is computing a store address either way -- it just computes a
  different one. That turns each destination's payload into one contiguous slab,
  which is what SDMA needs.

## Window layout

    [ signal | counters | locks | staging | recv ]

``staging`` is zero-sized when nothing on the configured path needs it (the
GEMM alone), and ``recv`` is always present -- it is the operation's output.
"""

from __future__ import annotations

from dataclasses import dataclass

# Kept in step with gemm_ar's layout.py: the two ops share a window discipline
# and the same cco primitives, and having the barrier geometry differ between
# them would be a trap for anyone reading both.
K_MAX_BLOCKS = 80  # signal slot rows
THREADS = 512
MAX_WORLD = 8  # the signal array fixes peer fan-out at 8
SIGNAL_ALIGN = 128  # alignas(128) on each control region
PACK_BYTES = 16  # the 16B pack the copy kernels index in

# The fp8 GEMM this op is built on (``gemm_ar/_gemm_a8w8_8wave.py``) is compiled
# for one tile shape. BLOCK_N is not a tuning knob: the epilogue's permlane lane
# transpose is written for 256 wide, and ``gemm_ar/op.py`` rejects anything else
# for the same reason.
DEFAULT_BLOCK_M = 128
DEFAULT_BLOCK_N = 256

#: mxfp8's BLOCK_M, and it is not a preference. The kernel packs a lane's four
#: M tiles into one dword and selects the byte with the MFMA's ``opsel``, which
#: is four tiles only when ``BLOCK_M // 64 == 4``. gemm_ar's op.py states the
#: same constant for the same reason.
#: The M granule the mainloop is written for: it builds BLOCK_M/64 MFMA
#: accumulators and the LDS staging halves the dimension, so a tile that is not
#: a whole multiple of this does not describe a real schedule. DEFAULT_BLOCK_M
#: and MXFP8_BLOCK_M are both multiples of it -- this is the rule, those are
#: choices.
TILE_M_GRANULE = 128

MXFP8_BLOCK_M = 256

#: ptpc's tile when there is enough work for it. 256x256 moves a third less A+B
#: per output element than 128x256 and measured 12-32% faster as a plain GEMM
#: at every model shape here except the one below two waves (8x MI355X, see the
#: README). DEFAULT_BLOCK_M stays the 128 that ``A2aConfig`` defaults to.
LARGE_BLOCK_M = 256

#: Waves of 256x256 tiles below which ptpc falls back to 128x256. At 70B M=2048
#: 256x256 is 1.25 waves, so the second wave runs a quarter of the CUs and the
#: tail costs more than the reuse saves.
LARGE_TILE_MIN_WAVES = 2


def default_block_m(m: int, n: int, quant: str = "ptpc", *, num_cus: int = 256) -> int:
    """BLOCK_M for a shape: 256 for mxfp8, 256 for ptpc/blockscale with enough work.

    blockscale is left at 128: its per-K-block promotion doubles the
    accumulators and 256x256 spills there (``gemm_ar/op.py``).
    """
    if quant == "mxfp8":
        return MXFP8_BLOCK_M
    if quant != "ptpc" or m % LARGE_BLOCK_M:
        return DEFAULT_BLOCK_M
    tiles = (m // LARGE_BLOCK_M) * (n // DEFAULT_BLOCK_N)
    if tiles < LARGE_TILE_MIN_WAVES * num_cus:
        return DEFAULT_BLOCK_M
    return LARGE_BLOCK_M


#: The ue8m0 group, on both operands: A is 1x32, B is 32x32.
MXFP8_BLOCK = 32

#: The most chunks a destination's slab is pushed in, and the smallest chunk
#: worth a put. Below ~1 MiB the per-packet cost starts to dominate (README), so
#: the default chunk count is the largest one at or below MAX_CHUNKS that keeps
#: a chunk at or above MIN_CHUNK_BYTES.
MAX_CHUNKS = 8
MIN_CHUNK_BYTES = 1 << 20


def default_chunks(m: int, shard_n: int, block_m: int) -> int:
    """Chunks per destination for this M: as many as keep each put >= 1 MiB."""
    m_tiles = m // block_m
    best = 1
    for c in range(1, min(MAX_CHUNKS, m_tiles) + 1):
        if m_tiles % c == 0 and (m // c) * shard_n * 2 >= MIN_CHUNK_BYTES:
            best = c
    return best


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class A2aConfig:
    """Shape + world size -> every offset and count the kernels need.

    ``m`` is this rank's row count and ``n`` the full output width, which is
    split ``world_size`` ways. Both are the *logical* bf16 tensor; the wire is
    bf16 too, so there is no separate wire size here (``gemm_ar`` has one
    because its gather leg can be fp8; that idea applies here as well and is
    deliberately left out of the first version).
    """

    world_size: int
    m: int
    n: int
    elem_bytes: int = 2  # bf16 output
    block_m: int = DEFAULT_BLOCK_M
    block_n: int = DEFAULT_BLOCK_N
    threads: int = THREADS
    max_blocks: int = K_MAX_BLOCKS
    #: Whether the window reserves the ``[dst][M][shard_n]`` staging slab. The
    #: SDMA paths need it -- a copy engine reads a contiguous source range and
    #: cannot gather -- and the GEMM alone does not. Sizing it is ``world_size * m * shard_n * 2`` bytes, so on
    #: the model shape that is 75 MiB it is worth not reserving when unused.
    staged: bool = False
    #: Sub-slices each destination's slab is pushed in, for the fused GEMM.
    #: 1 means "one put per destination"; more lets a destination's first rows
    #: leave while the GEMM is still computing its later ones. Chunking is along
    #: **M**, so a chunk stays contiguous inside ``[dst][M][shard_n]``.
    counter_chunks: int = 1
    #: Row tiles in each chunk, in order, when they are not all equal. Empty
    #: means ``counter_chunks`` equal chunks. An uneven plan exists to shrink
    #: the two ends a fused GEMM cannot hide -- the wait before the first put
    #: and the transfer of the last chunk -- while keeping the middle puts large
    #: enough that the per-packet cost does not dominate.
    chunk_plan: tuple[int, ...] = ()
    #: Chunks the counter *region* is sized for, as opposed to the number this
    #: config uses. Same rationale as gemm_ar: ``counter_chunks`` changes with
    #: ``m``, and sizing the region from it would move every later offset, which
    #: silently reinterprets one shape's payload as another's control state when
    #: a single window serves several ``m``. 0 means "same as counter_chunks".
    counter_capacity: int = 0
    #: Independent counter sets the region holds, and which one this config
    #: uses. Every call's elected tile puts its counter back to zero, so sets
    #: could be shared; one per shape is kept so no shape's election depends on
    #: another's calls having completed.
    counter_shape_slots: int = 1
    counter_shape_index: int = 0
    #: Lay the payload regions out for this ``m`` instead of for ``m``. Pinning
    #: them to the largest ``m`` an instance will serve makes every shape's map
    #: identical, so a region only ever aliases *itself*. Sizes stay ``m``-based:
    #: this moves where things are, not how much is moved. 0 means "same as m".
    capacity_m: int = 0

    # --- basic sizes -------------------------------------------------------

    @property
    def shard_n(self) -> int:
        """Columns each destination receives. Full-N: ``n`` is split exactly."""
        return self.n // self.world_size

    @property
    def cap_m(self) -> int:
        return self.capacity_m or self.m

    @property
    def slab_elems(self) -> int:
        """One (source, destination) pair's payload, in elements."""
        return self.m * self.shard_n

    @property
    def slab_bytes(self) -> int:
        return self.slab_elems * self.elem_bytes

    @property
    def cap_slab_bytes(self) -> int:
        """``slab_bytes`` at the capacity shape -- what the offsets are built on."""
        return self.cap_m * self.shard_n * self.elem_bytes

    @property
    def nbytes(self) -> int:
        """The full local ``[M, N]`` result, for the gemm-only path and reporting."""
        return self.m * self.n * self.elem_bytes

    @property
    def m_tiles(self) -> int:
        return self.m // self.block_m

    @property
    def n_blocks(self) -> int:
        return self.n // self.block_n

    @property
    def n_blocks_per_peer(self) -> int:
        """N tiles inside one destination's shard. A tile never straddles two."""
        return self.shard_n // self.block_n

    @property
    def tiles_per_peer(self) -> int:
        return self.m_tiles * self.n_blocks_per_peer

    @property
    def chunk_tiles(self) -> tuple[int, ...]:
        """Row tiles per chunk, in order -- ``chunk_plan`` or the even split."""
        if self.chunk_plan:
            return tuple(self.chunk_plan)
        return (self.m_tiles // self.counter_chunks,) * self.counter_chunks

    @property
    def chunk_starts(self) -> tuple[int, ...]:
        """First row tile of each chunk."""
        starts, at = [], 0
        for t in self.chunk_tiles:
            starts.append(at)
            at += t
        return tuple(starts)

    # --- index maps --------------------------------------------------------
    #
    # Both are element indices, and both are the same formula with a different
    # leading rank -- ``staging`` is indexed by *destination* on the sender and
    # ``recv`` by *source* on the receiver, which is the whole of the all-to-all.

    def staging_index(self, dst_rank: int, row: int, local_col: int) -> int:
        """Where the sender puts the value bound for ``dst_rank``."""
        self._check_rank(dst_rank)
        return dst_rank * self.slab_elems + row * self.shard_n + local_col

    def recv_index(self, src_rank: int, row: int, local_col: int) -> int:
        """Where that value lands in the receiver's window."""
        self._check_rank(src_rank)
        return src_rank * self.slab_elems + row * self.shard_n + local_col

    def dest_of_column(self, col: int) -> int:
        """Which rank owns output column ``col``."""
        if not 0 <= col < self.n:
            raise IndexError(f"column {col} outside [0, {self.n})")
        return col // self.shard_n

    def _check_rank(self, r: int) -> None:
        if not 0 <= r < self.world_size:
            raise IndexError(f"rank {r} outside [0, {self.world_size})")

    # --- symmetric window layout: [ signal | counters | locks | staging | recv ]

    @property
    def start_off(self) -> int:
        return 0

    @property
    def end_off(self) -> int:
        return _align_up(self.max_blocks * MAX_WORLD * 4, SIGNAL_ALIGN)

    @property
    def flag_off(self) -> int:
        return self.end_off * 2

    @property
    def counter_capacity_eff(self) -> int:
        """Chunks per set the region is sized for; ``counter_chunks`` if unset."""
        return self.counter_capacity or self.counter_chunks

    @property
    def counter_region_off(self) -> int:
        """Tile counters for the fused GEMM's election, one per (dest, chunk);
        each call's last tile resets its counter to zero."""
        return _align_up(self.flag_off + self.max_blocks * 4, SIGNAL_ALIGN)

    @property
    def counter_set_bytes(self) -> int:
        return self.world_size * self.counter_capacity_eff * 4

    @property
    def counter_off(self) -> int:
        """This config's counter set. Sized by capacity, so it does not move."""
        return (
            self.counter_region_off + self.counter_shape_index * self.counter_set_bytes
        )

    @property
    def counter_bytes(self) -> int:
        return _align_up(
            self.counter_set_bytes * self.counter_shape_slots, SIGNAL_ALIGN
        )

    @property
    def lock_off(self) -> int:
        """One submit lock per destination, for the fused epilogue.

        Needed as soon as a destination has more than one chunk: the tile counter
        elects one block per *chunk*, so two of them can reach the SDMA submit
        for the same queue at once. Shared across shapes -- a lock is taken and
        released inside one kernel, so it carries nothing between calls.
        """
        return self.counter_region_off + self.counter_bytes

    @property
    def lock_bytes(self) -> int:
        return _align_up(self.world_size * 4, SIGNAL_ALIGN)

    @property
    def credit_off(self) -> int:
        """One u32 per source rank: the call that source has started.

        Source ``s`` writes ``e`` into slot ``s`` of every peer's window when it
        starts call ``e``. Starting a call means its stream has finished
        everything before it, including whatever read call ``e-1``'s recv, so a
        sender may write into ``s``'s recv once slot ``s`` reads ``>= e``. The
        end-of-call barrier alone does not give that: it says every rank has
        *written* call ``e-1``, not that every rank has *read* it.
        """
        return self.lock_off + self.lock_bytes

    @property
    def credit_bytes(self) -> int:
        return _align_up(self.world_size * 4, SIGNAL_ALIGN)

    @property
    def signal_bytes(self) -> int:
        """Everything before the payload: barriers, counters, locks, credits."""
        return self.credit_off + self.credit_bytes

    @property
    def staging_off(self) -> int:
        return self.signal_bytes

    @property
    def staging_bytes(self) -> int:
        """``[dst][M][shard_n]``, or nothing when the path stores direct."""
        if not self.staged:
            return 0
        return self.world_size * self.cap_slab_bytes

    def staging_slot_off(self, dst_rank: int) -> int:
        self._check_rank(dst_rank)
        if not self.staged:
            raise ValueError(
                "this config has no staging region (staged=False); build it "
                "with staged=True for the SDMA paths"
            )
        return self.staging_off + dst_rank * self.cap_slab_bytes

    @property
    def recv_off(self) -> int:
        return self.staging_off + self.staging_bytes

    @property
    def recv_bytes(self) -> int:
        """``[world*M, shard_n]`` -- the operation's output."""
        return self.world_size * self.cap_slab_bytes

    def recv_slot_off(self, src_rank: int) -> int:
        """Byte offset of ``src_rank``'s contribution in *any* rank's window.

        The region starts at the capacity-pinned ``recv_off``, but inside it the
        slots are packed by *this* call's ``m``: the result is then one
        contiguous ``[world*m, shard_n]`` block whatever the capacity, which is
        what a caller can take as a tensor view. Two shapes do overlap inside
        the region, the same way two calls of one shape do -- each call's recv
        replaces the previous one's, and the recv-reuse credit orders that.
        """
        self._check_rank(src_rank)
        return self.recv_off + src_rank * self.slab_bytes

    @property
    def window_bytes(self) -> int:
        return self.recv_off + self.recv_bytes

    # --- traffic model, for reporting alongside measured time ---------------

    @property
    def remote_bytes_per_rank(self) -> int:
        """Bytes a rank sends. Its own slab stays local, so it is world-1 slabs."""
        return (self.world_size - 1) * self.slab_bytes

    # --- validation --------------------------------------------------------

    def validate(self) -> None:
        if self.world_size < 2 or self.world_size > MAX_WORLD:
            raise ValueError(
                f"world_size must be in [2, {MAX_WORLD}], got {self.world_size}"
            )
        if self.m < 1 or self.n < 1:
            raise ValueError(f"m and n must be positive, got m={self.m} n={self.n}")
        for name, value, granule in (
            ("block_m", self.block_m, TILE_M_GRANULE),
            ("block_n", self.block_n, DEFAULT_BLOCK_N),
        ):
            if value < granule or value % granule:
                raise ValueError(
                    f"{name}={value} must be a positive multiple of {granule}: the "
                    f"GEMM's MFMA tiling and LDS staging are written for that "
                    f"granule"
                )
        if self.block_n != DEFAULT_BLOCK_N:
            raise ValueError(
                f"block_n must be exactly {DEFAULT_BLOCK_N}: the epilogue always "
                f"compiles with permlane, whose lane transpose is written for "
                f"that width"
            )
        # The rule this operation turns on. A destination owns a contiguous run
        # of columns; if that run were not a whole number of N tiles, one GEMM
        # tile would span two destinations and its 256-wide store would have to
        # be split -- which is the case the compact receive layout exists to
        # avoid.
        if self.n % (self.world_size * self.block_n):
            raise ValueError(
                f"n={self.n} must be a multiple of world_size*block_n="
                f"{self.world_size * self.block_n} so each destination's shard is "
                f"a whole number of N tiles; with n%world_size==0 alone a tile can "
                f"straddle two destinations and its store cannot stay contiguous"
            )
        if self.m % self.block_m:
            raise ValueError(
                f"m={self.m} must be a multiple of block_m={self.block_m}: the "
                f"completion counters count whole tiles, so a partial row tile "
                f"would never complete its chunk"
            )
        if self.counter_chunks < 1:
            raise ValueError(f"counter_chunks must be >= 1, got {self.counter_chunks}")
        if self.chunk_plan:
            if len(self.chunk_plan) != self.counter_chunks:
                raise ValueError(
                    f"chunk_plan has {len(self.chunk_plan)} chunks but "
                    f"counter_chunks={self.counter_chunks}; the counter region "
                    f"is indexed by chunk, so the two must agree"
                )
            if any(t < 1 for t in self.chunk_plan):
                raise ValueError(
                    f"chunk_plan {self.chunk_plan} has an empty chunk; a chunk "
                    f"with no tiles would never be elected and never pushed"
                )
            if sum(self.chunk_plan) != self.m_tiles:
                raise ValueError(
                    f"chunk_plan {self.chunk_plan} covers {sum(self.chunk_plan)} "
                    f"row tiles, but m={self.m} has {self.m_tiles}"
                )
        elif self.m_tiles % self.counter_chunks:
            raise ValueError(
                f"counter_chunks ({self.counter_chunks}) must divide the "
                f"{self.m_tiles} row tiles: chunking is along M so that a chunk "
                f"stays contiguous inside [dst][M][shard_n]"
            )
        if self.counter_capacity and self.counter_capacity < self.counter_chunks:
            raise ValueError(
                f"counter_capacity ({self.counter_capacity}) must be >= "
                f"counter_chunks ({self.counter_chunks}); the region has to hold "
                f"the set this config indexes into"
            )
        if self.counter_shape_slots < 1:
            raise ValueError(
                f"counter_shape_slots must be >= 1, got {self.counter_shape_slots}"
            )
        if not 0 <= self.counter_shape_index < self.counter_shape_slots:
            raise ValueError(
                f"counter_shape_index must be in [0, {self.counter_shape_slots}), "
                f"got {self.counter_shape_index}"
            )
        if self.capacity_m and self.capacity_m < self.m:
            raise ValueError(
                f"capacity_m ({self.capacity_m}) must be >= m ({self.m}); it is "
                f"the largest shape the offsets are laid out for"
            )
        if self.slab_bytes % PACK_BYTES:
            raise ValueError(
                f"a slab is {self.slab_bytes}B and must be a multiple of "
                f"{PACK_BYTES} so the copy kernels can move whole packs"
            )


def a2a_config(**kwargs) -> A2aConfig:
    """Build and validate in one step, which is what every caller wants."""
    cfg = A2aConfig(**kwargs)
    cfg.validate()
    return cfg


def counter_chunks(m_tiles: int, requested: int) -> int:
    """The largest usable chunk count at or below ``requested``.

    ``counter_chunks`` has to divide the row tiles, and the caller usually has a
    preference rather than a requirement. Raising instead of silently rounding
    would make a benchmark sweep unusable, so this rounds down to a divisor.
    """
    if m_tiles < 1:
        raise ValueError(
            f"m_tiles must be >= 1, got {m_tiles}: m is below one block_m row "
            f"tile, so there is nothing to chunk"
        )
    if requested < 1:
        raise ValueError(f"requested chunks must be >= 1, got {requested}")
    for c in range(min(requested, m_tiles), 0, -1):
        if m_tiles % c == 0:
            return c
    return 1  # unreachable: 1 always divides


__all__ = [
    "A2aConfig",
    "a2a_config",
    "counter_chunks",
    "default_block_m",
    "default_chunks",
    "MAX_CHUNKS",
    "MIN_CHUNK_BYTES",
    "DEFAULT_BLOCK_M",
    "DEFAULT_BLOCK_N",
    "LARGE_BLOCK_M",
    "LARGE_TILE_MIN_WAVES",
    "MXFP8_BLOCK",
    "MXFP8_BLOCK_M",
    "MAX_WORLD",
    "PACK_BYTES",
    "THREADS",
]
