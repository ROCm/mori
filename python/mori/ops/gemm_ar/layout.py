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
"""Window layout and launch geometry for the FlyDSL cco all-reduce.

All of the arithmetic that the kernels depend on lives here so it can be unit
tested without a GPU: symmetric-window offsets, the per-block signal slot map,
the 1-stage/2-stage dispatch thresholds, and the launch geometry.

Everything mirrors aiter's ``custom_all_reduce.cuh`` so the two are comparable
kernel-for-kernel:

* ``Signal { start[kMaxBlocks][8]; end[kMaxBlocks][8]; _flag[kMaxBlocks] }``,
  each region 128-byte aligned -- the slots are per (block, peer) so blocks
  synchronise independently (``custom_all_reduce.cuh:43-51``).
* 1-stage below the size thresholds, 2-stage above
  (``custom_all_reduce.cuh:3771-3788``).

The launch geometry deliberately does *not* follow aiter -- see ``LSA_BLOCK_CAP``.

The transport is the only thing this file does not model: LSA and SDMA share the
window layout, they differ only in how bytes move.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- aiter parity constants (custom_all_reduce.cuh) ---
K_MAX_BLOCKS = 80  # kMaxBlocks; also the number of signal slot rows
THREADS = 512  # THREAD_NUM / __launch_bounds__(512, 1)
MAX_WORLD = 8  # Signal::start[..][8] fixes the peer fan-out at 8
PACK_BYTES = 16  # one 16B pack is the atomic transfer unit
SIGNAL_ALIGN = 128  # alignas(128) on each Signal region

# Grid cap for the LSA kernels. NOT aiter's 80: aiter gives each thread one pack
# from one peer (its `warp_id` selects the peer), while ours gives each thread one
# index across all `world` peers -- 8x the in-flight bytes per thread. Matching
# aiter's grid therefore over-subscribes the xGMI request queues, and measurably
# so: at [4096, 7168] bf16 on 8x MI355X the time rises monotonically past ~32
# blocks (286us @28 -> 361 @80 -> 583 @256). Measured plateau is 24-36; 24 is
# chosen because it is also the optimum at [512, 7168], where the curve is
# sharper (64.0us @24 vs 68.5 @32).
#
# The signal array still has K_MAX_BLOCKS rows, so this cap can be raised for
# experiments without touching the window layout.
LSA_BLOCK_CAP = 24

# Grid cap for the SDMA all-reduce's local reduce kernel. Deliberately not
# LSA_BLOCK_CAP: that one throttles outstanding xGMI requests, whereas this
# kernel's loads are all local HBM and want the opposite. See the sweep in
# bench_ar.py.
SDMA_REDUCE_BLOCK_CAP = 256

# 1-stage thresholds, in bytes (custom_all_reduce.cuh:3779).
ONE_STAGE_MAX_BYTES_LE4 = 160 * 1024
ONE_STAGE_MAX_BYTES_LE8 = 80 * 1024


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def select_stage(world_size: int, nbytes: int) -> int:
    """Return 1 or 2 -- aiter's dispatch rule, reproduced exactly.

    world==2 is always 1-stage: with a single peer the reduce-scatter half of
    the 2-stage algorithm saves nothing and costs an extra barrier.
    """
    if world_size == 2:
        return 1
    if world_size <= 4 and nbytes < ONE_STAGE_MAX_BYTES_LE4:
        return 1
    if world_size <= 8 and nbytes < ONE_STAGE_MAX_BYTES_LE8:
        return 1
    return 2


@dataclass(frozen=True)
class ArConfig:
    """Shape + world size -> every offset and count the kernels need.

    ``m``/``n`` describe the logical bf16 tensor being reduced; ``n`` is the row
    length (7168 for DSV4-Pro ``wo_b``) and is never sharded -- the reduce-scatter
    shards along ``m`` so every peer slice is one contiguous byte range. Sharding
    along ``n`` would make each slice ``m`` separate ``n/world*2``-byte pieces,
    which is fatal for SDMA (~2us per packet regardless of size).
    """

    world_size: int
    m: int
    n: int
    elem_bytes: int = 2  # bf16
    threads: int = THREADS
    max_blocks: int = K_MAX_BLOCKS  # signal-array rows, never the launched grid
    block_cap: int = LSA_BLOCK_CAP
    #: SDMA landing slots, one slice per peer. LSA needs none -- it reduces
    #: straight out of the peers' input regions -- but a copy engine has to be
    #: given somewhere to write, so the SDMA backend reserves ``world_size``
    #: slices (slot ``rank`` goes unused; keeping it makes the index be the
    #: peer rank, which is worth one slice of VRAM).
    recv_slots: int = 0
    #: Sub-slices each peer's landing region is pushed in, for the fused GEMM.
    #: 1 means "one put per destination"; more lets a destination's first chunk
    #: leave while the GEMM is still computing its later ones. Only the fused
    #: path reads this; the standalone SDMA all-reduce always pushes whole slices.
    counter_chunks: int = 1
    #: Grid for the SDMA reduce. Independent of ``block_cap`` because that cap
    #: throttles *xGMI* requests, while this kernel reads local HBM and wants as
    #: much in flight as it can get.
    force_reduce_blocks: int = 0
    #: Override the computed grid. Only the signal array bounds the grid for
    #: correctness (``start[max_blocks][8]``), so raising this also needs
    #: ``max_blocks`` raised. Exists so the benchmark can sweep occupancy.
    force_blocks: int | None = None

    # --- basic sizes ---

    @property
    def num_elems(self) -> int:
        return self.m * self.n

    @property
    def nbytes(self) -> int:
        return self.num_elems * self.elem_bytes

    @property
    def elems_per_pack(self) -> int:
        return PACK_BYTES // self.elem_bytes

    @property
    def num_packs(self) -> int:
        """Payload measured in 16B packs -- the unit the kernels index in."""
        return self.nbytes // PACK_BYTES

    @property
    def stage(self) -> int:
        return select_stage(self.world_size, self.nbytes)

    # --- reduce-scatter partition (2-stage), sharded along m ---

    @property
    def packs_per_rank(self) -> int:
        """Packs each rank owns and reduces. Trailing packs go to the last rank."""
        return self.num_packs // self.world_size

    def owner_pack_range(self, rank: int) -> tuple[int, int]:
        start = rank * self.packs_per_rank
        end = (
            self.num_packs
            if rank == self.world_size - 1
            else start + self.packs_per_rank
        )
        return start, end

    @property
    def largest_pack_part(self) -> int:
        """``part + size % ngpus`` -- the all-gather loop bound in aiter."""
        return self.packs_per_rank + self.num_packs % self.world_size

    @property
    def slice_bytes(self) -> int:
        """Contiguous bytes in one peer slice; drives SDMA bandwidth."""
        return self.packs_per_rank * PACK_BYTES

    # --- launch geometry ---

    @property
    def threads_per_peer(self) -> int:
        """``tnum_gpu``: the thread group that services one peer."""
        return self.threads // self.world_size

    @property
    def blocks(self) -> int:
        # One thread owns one pack index and reduces it across every peer, so a
        # block advances the index space by `threads` per iteration -- not by
        # `threads_per_peer` as in aiter, whose threads are split across peers.
        if self.force_blocks is not None:
            return self.force_blocks
        per_block = self.threads
        covered = self.num_packs if self.stage == 1 else self.packs_per_rank
        need = (covered + per_block - 1) // per_block
        return max(1, min(self.block_cap, need))

    # --- symmetric window layout: [ signal | input | output | tmp | recv ] ---
    # ``tmp`` holds each rank's reduced slice during 2-stage; peers all-gather
    # straight out of it, which is why it lives inside the registered window.
    # ``recv`` is SDMA-only and empty unless ``recv_slots`` is set.

    @property
    def signal_bytes(self) -> int:
        return self.lock_off + self.lock_bytes

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
    def reduce_blocks(self) -> int:
        if self.force_reduce_blocks:
            return self.force_reduce_blocks
        need = (self.packs_per_rank + self.threads - 1) // self.threads
        return max(1, min(SDMA_REDUCE_BLOCK_CAP, need))

    @property
    def counter_off(self) -> int:
        """Monotonic tile counters for the fused GEMM, one per (dest, chunk).

        Its own 128-byte-aligned region rather than slack in ``_flag``: the count
        is ``world_size * counter_chunks`` and grows with M, so borrowing the 16
        spare ``_flag`` slots would silently stop fitting at M >= 8192.
        """
        return _align_up(self.flag_off + self.max_blocks * 4, SIGNAL_ALIGN)

    @property
    def counter_bytes(self) -> int:
        return _align_up(self.world_size * self.counter_chunks * 4, SIGNAL_ALIGN)

    @property
    def lock_off(self) -> int:
        """One submit lock per destination, for the fused GEMM's epilogue.

        Needed as soon as a destination has more than one chunk: the tile
        counter elects one block per *chunk*, so two of them can reach the SDMA
        submit for the same queue at once.
        """
        return self.counter_off + self.counter_bytes

    @property
    def lock_bytes(self) -> int:
        return _align_up(self.world_size * 4, SIGNAL_ALIGN)

    def counter_slot(self, dest: int, chunk: int) -> int:
        if not 0 <= dest < self.world_size:
            raise IndexError(f"dest {dest} outside [0, {self.world_size})")
        if not 0 <= chunk < self.counter_chunks:
            raise IndexError(f"chunk {chunk} outside [0, {self.counter_chunks})")
        return dest * self.counter_chunks + chunk

    def signal_slot(self, block: int, peer: int) -> int:
        """Element index into ``start``/``end`` for (block, peer)."""
        if not 0 <= block < self.max_blocks:
            raise IndexError(f"block {block} outside [0, {self.max_blocks})")
        if not 0 <= peer < MAX_WORLD:
            raise IndexError(f"peer {peer} outside [0, {MAX_WORLD})")
        return block * MAX_WORLD + peer

    @property
    def input_off(self) -> int:
        return _align_up(self.signal_bytes, SIGNAL_ALIGN)

    @property
    def output_off(self) -> int:
        return self.input_off + self.nbytes

    @property
    def tmp_off(self) -> int:
        return self.output_off + self.nbytes

    @property
    def tmp_bytes(self) -> int:
        """Room for the largest owned slice, so rank world-1's tail fits too."""
        return self.largest_pack_part * PACK_BYTES

    @property
    def recv_off(self) -> int:
        return self.tmp_off + self.tmp_bytes

    @property
    def recv_bytes(self) -> int:
        return self.recv_slots * self.slice_bytes

    def recv_slot_off(self, peer: int) -> int:
        """Where peer ``peer``'s contribution to *my* slice lands."""
        if not 0 <= peer < self.world_size:
            raise IndexError(f"peer {peer} outside [0, {self.world_size})")
        if peer >= self.recv_slots:
            raise IndexError(
                f"peer {peer} has no landing slot; recv_slots={self.recv_slots}"
            )
        return self.recv_off + peer * self.slice_bytes

    @property
    def window_bytes(self) -> int:
        return self.recv_off + self.recv_bytes

    # --- traffic model, for reporting alongside measured time ---

    @property
    def remote_bytes_per_rank(self) -> int:
        """Bytes crossing xGMI per rank: (P-1)/P*2 for 2-stage, (P-1) for 1-stage."""
        p = self.world_size
        if self.stage == 1:
            return (p - 1) * self.nbytes
        return 2 * (p - 1) * self.nbytes // p

    def validate(self) -> None:
        if self.world_size < 2 or self.world_size > MAX_WORLD:
            raise ValueError(
                f"world_size must be in [2, {MAX_WORLD}], got {self.world_size}"
            )
        if self.threads % self.world_size:
            raise ValueError(
                f"threads ({self.threads}) must be divisible by world_size "
                f"({self.world_size}); aiter falls back to the naive kernel for "
                "world_size=6 for exactly this reason"
            )
        if self.nbytes % (self.world_size * PACK_BYTES):
            raise ValueError(
                f"payload {self.nbytes}B must be a multiple of "
                f"world_size*{PACK_BYTES}={self.world_size * PACK_BYTES} so the "
                "vectorized path applies (aiter: DISPATCH_REDUCE falls back to "
                "_naive otherwise)"
            )
        if (
            self.force_blocks is not None
            and not 1 <= self.force_blocks <= self.max_blocks
        ):
            raise ValueError(
                f"force_blocks={self.force_blocks} outside [1, {self.max_blocks}]; "
                "the signal array has one row per block, so raise max_blocks too"
            )
        if self.counter_chunks < 1:
            raise ValueError(f"counter_chunks must be >= 1, got {self.counter_chunks}")
        if not 0 <= self.recv_slots <= self.world_size:
            raise ValueError(
                f"recv_slots must be in [0, world_size={self.world_size}], "
                f"got {self.recv_slots}"
            )
        if self.elem_bytes not in (2, 4):
            raise ValueError(f"elem_bytes must be 2 or 4, got {self.elem_bytes}")


__all__ = [
    "ArConfig",
    "select_stage",
    "K_MAX_BLOCKS",
    "LSA_BLOCK_CAP",
    "SDMA_REDUCE_BLOCK_CAP",
    "THREADS",
    "MAX_WORLD",
    "PACK_BYTES",
]
