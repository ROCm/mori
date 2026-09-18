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

The LSA paths skip staging entirely: a vector store can be strided, so they
write straight into the peer's ``recv`` region.

## Window layout

    [ signal | counters | locks | staging | recv ]

``staging`` is zero-sized when nothing on the configured path needs it (the LSA
transports), and ``recv`` is always present -- it is the operation's output.
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

# Grid cap for the LSA copy kernel. Same reasoning as gemm_ar's LSA_BLOCK_CAP --
# these stores go over xGMI and over-subscribing the request queues costs time
# rather than saving it -- but the number is deliberately not shared: that one
# was measured on an all-reduce's access pattern, and this one has to be measured
# on this one before it means anything. Until then it is a starting point.
LSA_BLOCK_CAP = 24


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
    block_cap: int = LSA_BLOCK_CAP
    #: Whether the window reserves the ``[dst][M][shard_n]`` staging slab. The
    #: SDMA paths need it -- a copy engine reads a contiguous source range and
    #: cannot gather -- and the LSA paths do not, because they store straight
    #: into the peer. Sizing it is ``world_size * m * shard_n * 2`` bytes, so on
    #: the model shape that is 75 MiB it is worth not reserving when unused.
    staged: bool = False
    #: Sub-slices each destination's slab is pushed in, for the fused GEMM.
    #: 1 means "one put per destination"; more lets a destination's first rows
    #: leave while the GEMM is still computing its later ones. Chunking is along
    #: **M**, so a chunk stays contiguous inside ``[dst][M][shard_n]``.
    counter_chunks: int = 1
    #: Chunks the counter *region* is sized for, as opposed to the number this
    #: config uses. Same rationale as gemm_ar: ``counter_chunks`` changes with
    #: ``m``, and sizing the region from it would move every later offset, which
    #: silently reinterprets one shape's payload as another's control state when
    #: a single window serves several ``m``. 0 means "same as counter_chunks".
    counter_capacity: int = 0
    #: Independent counter sets the region holds, and which one this config
    #: uses. The election test is a residue on a counter that is never reset, so
    #: two shapes sharing a set would elect on each other's leftovers.
    counter_shape_slots: int = 1
    counter_shape_index: int = 0
    #: Lay the payload regions out for this ``m`` instead of for ``m``. Pinning
    #: them to the largest ``m`` an instance will serve makes every shape's map
    #: identical, so a region only ever aliases *itself*. Sizes stay ``m``-based:
    #: this moves where things are, not how much is moved. 0 means "same as m".
    capacity_m: int = 0
    #: Override the computed grid for the copy kernel. Only ``max_blocks`` bounds
    #: it for correctness (the signal array has one row per block), so raising
    #: this past that needs ``max_blocks`` raised too.
    force_blocks: int | None = None

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

    # --- launch geometry ---------------------------------------------------

    @property
    def copy_blocks(self) -> int:
        """Grid for the standalone LSA copy kernel."""
        if self.force_blocks is not None:
            return self.force_blocks
        # One block per (destination, slice) with at least one slice each; the
        # cap is what actually binds on any real shape.
        return max(self.world_size, min(self.block_cap, self.world_size * 4))

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
        """Monotonic tile counters for the fused GEMM, one per (dest, chunk)."""
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
    def signal_bytes(self) -> int:
        """Everything before the payload: barriers, counters, locks."""
        return self.lock_off + self.lock_bytes

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
                "this config has no staging region (staged=False); the LSA "
                "transports store into the peer window directly"
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
        """Byte offset of ``src_rank``'s contribution in *any* rank's window."""
        self._check_rank(src_rank)
        return self.recv_off + src_rank * self.cap_slab_bytes

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
            ("block_m", self.block_m, DEFAULT_BLOCK_M),
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
        if self.m_tiles % self.counter_chunks:
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
        if (
            self.force_blocks is not None
            and not 1 <= self.force_blocks <= self.max_blocks
        ):
            raise ValueError(
                f"force_blocks={self.force_blocks} outside [1, {self.max_blocks}]; "
                f"the signal array has one row per block, so raise max_blocks too"
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
    "DEFAULT_BLOCK_M",
    "DEFAULT_BLOCK_N",
    "LSA_BLOCK_CAP",
    "MAX_WORLD",
    "PACK_BYTES",
    "THREADS",
]
