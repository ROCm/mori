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
"""Window layout and launch geometry for the FlyDSL cco GEMM+all-gather.

All of the arithmetic the kernels depend on lives here so it can be unit tested
without a GPU: the symmetric-window offsets, the receive index map, the
completion-counter geometry, and the shape rules.

## What the operation is

Every rank holds its own ``A [M, K]`` -- its shard of the tokens -- and a
replicated ``B [N, K]``, computes the whole ``C = A @ B.T`` as ``[M, N]`` bf16,
and every rank ends up with all of them concatenated along rows:

    recv = [world*M, N]   rows [r*M, (r+1)*M) contributed by rank r

including its own contribution at row block ``rank``. This is the shape of the
PCP ``wkv_gate`` chain: each context-parallel rank scores its own tokens and
then everyone needs every token's score.

## Why this is the easy one of the three

Compare the sibling ops' constraints, because what is absent here is the design:

* ``gemm_ar``'s reduce-scatter shards along **m**, so each peer slice is one
  contiguous byte range -- deliberately, since sharding along ``n`` would make a
  slice ``m`` separate pieces, which is fatal for SDMA (~2us per packet).
* ``gemm_a2a`` *does* shard along ``n``, and pays for it with a
  ``[dst][M][shard_n]`` **staging** slab that the GEMM epilogue has to write in
  place of a plain ``[M, N]``, purely so the copy engine has one contiguous
  source range per destination.

All-gather shards nothing. A rank's payload is its whole ``C [M, N]``, which is
already one contiguous range, and it is the *same* range for every destination
-- the operation is a broadcast, not a distribution. Three things follow, and
they are the whole of this file's difference from ``gemm_a2a/layout.py``:

1. **There is no staging region.** The GEMM writes straight into this rank's own
   ``recv`` slot, and that slab is pushed unchanged. Nothing is ever re-laid-out.
2. **Counters are per chunk, not per (destination, chunk).** A chunk completing
   arms every destination's push at once, because they all get the same bytes.
3. **The copy is flat on both sides.** Source index and destination index are
   equal. ``gemm_a2a``'s LSA kernel reads strided and writes compact; this one
   does neither.

The price is on the wire: a rank sends ``(world-1) * M * N * 2`` bytes, which is
``world`` times what the all-to-all sends. All-gather is the bandwidth-bound
member of the family, which is also why it is the one where a low-precision wire
would pay the most.

## Window layout

    [ signal | counters | locks | recv ]

``recv`` is the operation's output, and also its *input*: slot ``rank`` is where
this rank's GEMM writes. That overlap is intentional and is what removes the
staging copy.
"""

from __future__ import annotations

from dataclasses import dataclass

# Kept in step with gemm_ar's and gemm_a2a's layout.py: the three ops share a
# window discipline and the same cco primitives, and having the barrier geometry
# differ between them would be a trap for anyone reading two of them.
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

#: The M granule the mainloop is written for: it builds BLOCK_M/64 MFMA
#: accumulators and the LDS staging halves the dimension, so a tile that is not
#: a whole multiple of this does not describe a real schedule. DEFAULT_BLOCK_M
#: and MXFP8_BLOCK_M are both multiples of it -- this is the rule, those are
#: choices.
TILE_M_GRANULE = 128

#: mxfp8's BLOCK_M, and it is not a preference. The kernel packs a lane's four
#: M tiles into one dword and selects the byte with the MFMA's ``opsel``, which
#: is four tiles only when ``BLOCK_M // 64 == 4``. gemm_ar's op.py and
#: gemm_a2a's layout state the same constant for the same reason.
MXFP8_BLOCK_M = 256

#: The ue8m0 group, on both operands: A is 1x32, B is 32x32.
MXFP8_BLOCK = 32

# Grid cap for the LSA copy kernel. Same reasoning as gemm_ar's and gemm_a2a's
# LSA_BLOCK_CAP -- these stores go over xGMI and over-subscribing the request
# queues costs time rather than saving it -- but the number is deliberately not
# shared: each was measured on a different access pattern, and this op's is the
# one pattern where a single load feeds world-1 stores. Until it is measured
# here it is a starting point.
LSA_BLOCK_CAP = 24


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class AgConfig:
    """Shape + world size -> every offset and count the kernels need.

    ``m`` is this rank's row count and ``n`` the full output width. Unlike
    ``gemm_a2a``'s config there is no ``shard_n``: nothing is split.

    Both dimensions describe the *logical* bf16 tensor, and the wire is bf16
    too, so there is no separate wire size here. ``gemm_ar`` has one because its
    gather leg can be fp8 -- and that leg is exactly this operation, so the idea
    applies here more directly than anywhere else. It is deliberately left out
    of the first version: the point of the first version is to establish what
    the bf16 wire costs, which is the baseline a narrowed wire is measured
    against.
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
    #: Sub-slices this rank's slab is pushed in, for the fused GEMM. 1 means
    #: "one put per destination at the end"; more lets the first rows leave
    #: while the GEMM is still computing the later ones. Chunking is along
    #: **M**, so a chunk is a contiguous byte range of ``[M, N]``.
    #:
    #: Note the transpose against gemm_a2a: there a chunk is contiguous inside
    #: one destination's slab and there are ``world * chunks`` counters; here a
    #: chunk is contiguous inside the single slab every destination receives,
    #: and there are ``chunks``.
    counter_chunks: int = 1
    #: Chunks the counter *region* is sized for, as opposed to the number this
    #: config uses. Same rationale as gemm_ar and gemm_a2a: ``counter_chunks``
    #: changes with ``m``, and sizing the region from it would move every later
    #: offset, which silently reinterprets one shape's payload as another's
    #: control state when a single window serves several ``m``. 0 means "same as
    #: counter_chunks".
    counter_capacity: int = 0
    #: Independent counter sets the region holds, and which one this config
    #: uses. The election test is a residue on a counter that is never reset, so
    #: two shapes sharing a set would elect on each other's leftovers.
    counter_shape_slots: int = 1
    counter_shape_index: int = 0
    #: Lay the payload region out for this ``m`` instead of for ``m``. Pinning
    #: it to the largest ``m`` an instance will serve makes every shape's map
    #: identical, so a region only ever aliases *itself*. Sizes stay ``m``-based:
    #: this moves where things are, not how much is moved. 0 means "same as m".
    capacity_m: int = 0
    #: Override the computed grid for the copy kernel. Only ``max_blocks`` bounds
    #: it for correctness (the signal array has one row per block), so raising
    #: this past that needs ``max_blocks`` raised too.
    force_blocks: int | None = None

    # --- basic sizes -------------------------------------------------------

    @property
    def cap_m(self) -> int:
        return self.capacity_m or self.m

    @property
    def slab_elems(self) -> int:
        """One rank's whole contribution, in elements. The full ``[M, N]``."""
        return self.m * self.n

    @property
    def slab_bytes(self) -> int:
        return self.slab_elems * self.elem_bytes

    @property
    def cap_slab_bytes(self) -> int:
        """``slab_bytes`` at the capacity shape -- what the offsets are built on."""
        return self.cap_m * self.n * self.elem_bytes

    @property
    def nbytes(self) -> int:
        """The local ``[M, N]`` result. Same as ``slab_bytes`` here, unlike in
        ``gemm_a2a`` where the local result is ``world`` slabs; kept as its own
        name because the benchmark reports GEMM traffic and wire traffic
        separately and they happen to coincide rather than being the same idea.
        """
        return self.m * self.n * self.elem_bytes

    @property
    def m_tiles(self) -> int:
        return self.m // self.block_m

    @property
    def n_blocks(self) -> int:
        return self.n // self.block_n

    @property
    def tiles_total(self) -> int:
        """Every tile of the GEMM. All of them feed every destination."""
        return self.m_tiles * self.n_blocks

    # --- index map ---------------------------------------------------------

    def recv_index(self, src_rank: int, row: int, col: int) -> int:
        """Where ``src_rank``'s ``[row, col]`` lands, in elements.

        There is only one index map, where ``gemm_a2a`` has two. Its sender map
        (``staging_index``, keyed by destination) and receiver map (``recv_index``,
        keyed by source) are the two halves of an all-to-all; a broadcast has one
        half, and the sender's copy of it is this same function at
        ``src_rank == rank``.
        """
        self._check_rank(src_rank)
        return src_rank * self.slab_elems + row * self.n + col

    def _check_rank(self, r: int) -> None:
        if not 0 <= r < self.world_size:
            raise IndexError(f"rank {r} outside [0, {self.world_size})")

    # --- launch geometry ---------------------------------------------------

    @property
    def copy_blocks(self) -> int:
        """Grid for the standalone LSA copy kernel."""
        if self.force_blocks is not None:
            return self.force_blocks
        # One block per (peer, slice) with at least one slice each; the cap is
        # what actually binds on any real shape.
        return max(self.world_size, min(self.block_cap, self.world_size * 4))

    # --- symmetric window layout: [ signal | counters | locks | recv ] ------

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
        """Monotonic tile counters for the fused GEMM, one per chunk."""
        return _align_up(self.flag_off + self.max_blocks * 4, SIGNAL_ALIGN)

    @property
    def counter_set_bytes(self) -> int:
        """No ``world_size`` factor -- see ``counter_chunks``."""
        return self.counter_capacity_eff * 4

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

        Needed as soon as there is more than one chunk: the tile counter elects
        one block per chunk, and that block posts to *every* destination's
        queue, so two chunks electing at once collide on all of them rather than
        on one. Shared across shapes -- a lock is taken and released inside one
        kernel, so it carries nothing between calls.
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
    def recv_off(self) -> int:
        return self.signal_bytes

    @property
    def recv_bytes(self) -> int:
        """``[world*M, N]`` -- the operation's output, and its input."""
        return self.world_size * self.cap_slab_bytes

    def recv_slot_off(self, src_rank: int) -> int:
        """Byte offset of ``src_rank``'s contribution in *any* rank's window.

        The same offset on every rank, which is what makes the transfer a pair of
        constants: a push reads ``recv_slot_off(rank)`` locally and writes
        ``recv_slot_off(rank)`` remotely, and a pull does the mirror image. In
        ``gemm_a2a`` only the destination offset had this property; the source
        came out of staging and depended on the destination.
        """
        self._check_rank(src_rank)
        return self.recv_off + src_rank * self.cap_slab_bytes

    @property
    def window_bytes(self) -> int:
        return self.recv_off + self.recv_bytes

    # --- traffic model, for reporting alongside measured time ---------------

    @property
    def remote_bytes_per_rank(self) -> int:
        """Bytes a rank sends. Its own slab stays local, so it is world-1 of them.

        ``world`` times ``gemm_a2a``'s, at the same ``[M, N]``: there a slab is
        ``M * N / world``, here it is ``M * N``.
        """
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
        # Note what is *not* here: gemm_a2a requires n % (world_size*block_n) so
        # that a destination's column shard is a whole number of N tiles. No
        # column is sharded here, so a tile cannot straddle two destinations and
        # the rule reduces to the GEMM's own tiling. That is what lets this op
        # run the ratio-128 wkv_gate shape (n=1024) at world_size=8, which the
        # all-to-all's rule would reject.
        if self.n % self.block_n:
            raise ValueError(
                f"n={self.n} must be a multiple of block_n={self.block_n}: the "
                f"GEMM emits whole N tiles and a partial one has no store path"
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
                f"stays contiguous inside [M, N]"
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


def ag_config(**kwargs) -> AgConfig:
    """Build and validate in one step, which is what every caller wants."""
    cfg = AgConfig(**kwargs)
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
    "AgConfig",
    "ag_config",
    "counter_chunks",
    "DEFAULT_BLOCK_M",
    "DEFAULT_BLOCK_N",
    "MXFP8_BLOCK",
    "MXFP8_BLOCK_M",
    "LSA_BLOCK_CAP",
    "MAX_WORLD",
    "PACK_BYTES",
    "THREADS",
]
