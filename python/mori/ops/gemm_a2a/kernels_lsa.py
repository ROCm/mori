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
"""The standalone all-to-all: a strided read of ``[M, N]`` into compact peers.

This is the split baseline. The GEMM has already written the whole ``[M, N]``
result into the window's ``local`` region; this kernel moves column block ``d``
into rank ``d``'s ``recv`` slot for this rank, over LSA -- plain vector stores
against a peer pointer from cco's flat symmetric VA, the same substrate
``gemm_ar``'s ``kernels_lsa`` uses for its all-reduce.

## The shape of the copy, and why it is not a memcpy

Source and destination disagree about what is contiguous:

    source      C[row, d*shard_n + j]        row stride N
    destination recv[my_rank][row, j]        row stride shard_n

so one destination's payload is ``M`` separate runs of ``shard_n`` elements on
the read side and one run of ``M*shard_n`` on the write side. The kernel is
therefore written around **rows**: each iteration moves one row's worth of one
destination, which is ``shard_n * 2`` bytes -- 4608 B at the model shape, i.e.
288 packs. That is long enough that the strided read costs nothing over a flat
one, and it keeps the peer-side stores fully contiguous, which is the side that
crosses xGMI.

gcnasm's ``opus_lsa_a2a_copy_kernel`` reads a *pre-staged* contiguous slab
instead, because its GEMM already wrote ``[dst][M][shard_n]``. Doing the
re-layout here rather than in the GEMM is what lets the split path reuse an
unmodified GEMM; the fused paths do it in the epilogue, where it is free.

## Completion

The kernel ends with the same monotonic-flag barrier as ``gemm_ar``'s LSA
all-reduce: signal every peer, spin until every peer has signalled. Flags are
never reset, so graph replay is safe. Without it a rank can return while a peer
is still writing into its ``recv``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int64

import mori.cco.device.flydsl as cco

# The GEMM, the buffer primitives and the cache-policy constants are gemm_ar's:
# this op differs in where C goes, not in how any of that works, and a second
# copy of them would be a second thing to keep in step with flydsl's API.
from ..gemm_ar._compat import (
    CM_CACHED,
    CM_SC0_SC1,
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
    i32_type,
    local_load_u32,
    local_store_u32,
    signal_load_u32,
    signal_ptr,
    signal_store_u32,
    wave_uniform_i64,
)
from .layout import MAX_WORLD

#: A 16B pack is always moved as 4 x i32, so buffer offsets are in i32 units
#: whatever the payload dtype. Indexing a bf16 pack in elements would stride
#: four times too far.
I32_PER_PACK = 4
PACK_ELEMS_BF16 = 8


def _spin_until(rsrc, flag):
    """Block until the u32 at ``rsrc`` reaches ``flag`` (unsigned compare).

    Unsigned so a wrapped counter still terminates; ``>=`` so an increment can
    never be missed. Same construction as ``gemm_ar.kernels_lsa._spin_until`` --
    it is duplicated rather than imported because that module is the all-reduce
    and importing a private helper across ops to save nine lines would tie this
    file's correctness to an unrelated one's refactors.
    """
    i32 = i32_type()
    first = signal_load_u32(rsrc)
    first_v = first.ir_value() if hasattr(first, "ir_value") else first
    loop = scf.WhileOp([i32], [first_v])
    cond = ir.Block.create_at_start(loop.before, [i32])
    body = ir.Block.create_at_start(loop.after, [i32])
    with ir.InsertionPoint(cond):
        cur = fx.Int32(cond.arguments[0])
        should_wait = fx.Uint32(cur) < fx.Uint32(flag)
        scf.ConditionOp(should_wait.ir_value(), [cond.arguments[0]])
    with ir.InsertionPoint(body):
        nxt = signal_load_u32(rsrc)
        scf.YieldOp([nxt.ir_value() if hasattr(nxt, "ir_value") else nxt])


def build_lsa_a2a(cfg, rank: int, *, src: str = "local", uncached: bool = True):
    """Compile an LSA all-to-all launcher for ``cfg`` on ``rank``.

    Returns ``run(dev_comm, win, stream=...)``, which moves this rank's
    ``[M, N]`` result out of the window and into every peer's ``recv`` slot for
    this rank, then barriers.

    ``src`` says what the GEMM left behind, and is only about the *read* side:

    * ``"local"`` -- the full ``[M, N]``, row stride ``N``. The kernel does the
      column-block extraction, so the GEMM needs no modification at all.
    * ``"staging"`` -- ``[dst][M][shard_n]`` already compacted. The read is then
      flat. Only useful once a GEMM writes that layout; kept because the SDMA
      path needs exactly this slab and the two should share one kernel.

    The source is passed in as a pointer rather than taken from the window: only
    the *destination* has to be symmetric, and reserving another ``M*N*2`` bytes
    of registered window (75 MiB at the model shape) to hold a buffer no peer
    ever reads would be pure waste.

    ``uncached`` sends the peer stores with ``sc0|sc1`` instead of letting them
    sit in this rank's L2. It is on by default for two reasons, and the
    correctness one came first:

    * A cached store to peer-homed memory leaves the line dirty in whichever
      XCD's L2 the block ran on, and the barrier that follows publishes arrival
      with a *system-scope* atomic, which can overtake it. gemm_ar hit exactly
      this and its note is at the same store. Nothing here fences before
      signalling, so cached stores were relying on luck.
    * It is also what the bandwidth wants: cached peer stores were reaching 58%
      of line rate where the copy engines reach 94%.

    The reason gemm_ar could not always use it is that a 2-byte partial-line
    write over the fabric loses updates. This kernel stores 16 bytes per lane
    and 1024 contiguous bytes per wave, so that objection does not apply.

    Each kernel is specialised on the Python ``rank`` so every window offset
    folds to a constant, as ``examples/cco/python/07_flydsl_sdma`` does.
    """
    cfg.validate()
    if src not in ("local", "staging"):
        raise ValueError(f"src must be local or staging, got {src!r}")
    if src == "staging" and not cfg.staged:
        raise ValueError(
            "src='staging' needs a config built with staged=True; without it the "
            "window has no staging region and the read would land in recv"
        )

    ws = cfg.world_size
    threads = cfg.threads
    blocks = cfg.copy_blocks
    m, n, shard_n = cfg.m, cfg.n, cfg.shard_n

    start_off, flag_off = cfg.start_off, cfg.flag_off
    # Where this rank's contribution goes in *every* peer's window. It is the
    # same offset on all of them -- that symmetry is the whole point of indexing
    # recv by source rather than by destination.
    my_recv_slot = cfg.recv_slot_off(rank)

    # Packs in one row of one destination's shard, and in a whole slab.
    if shard_n % PACK_ELEMS_BF16:
        raise ValueError(
            f"shard_n={shard_n} must be a multiple of {PACK_ELEMS_BF16} so a row "
            f"of a shard is a whole number of 16B packs"
        )
    row_packs = shard_n // PACK_ELEMS_BF16
    n_row_packs = n // PACK_ELEMS_BF16
    slab_packs = m * row_packs

    def _next_flag(w, bid):
        """``_flag[block] + 1`` -- monotonic, never reset (graph-replay safe)."""
        base = fx.Int64(w.lsa_ptr(rank, flag_off)) + fx.Int64(bid) * fx.Int64(4)
        rsrc = signal_ptr(base)
        return fx.Int32(local_load_u32(rsrc)) + fx.Int32(1), rsrc

    def _signal_and_wait(w, bid, flag, tid):
        """Publish to peer ``tid``, then wait for peer ``tid``'s publication.

        Branch-free on purpose: FlyDSL only rewrites ``if`` inside the
        ``@flyc.kernel`` function's own AST, so the ``tid < ws`` guard has to
        stay in the kernel body rather than move in here.
        """
        peer_arr = fx.Int64(w.lsa_ptr(tid, start_off))
        mine = fx.Int64((bid * MAX_WORLD + rank) * 4)
        signal_store_u32(signal_ptr(peer_arr + mine), flag)

        self_arr = fx.Int64(w.lsa_ptr(rank, start_off))
        theirs = fx.Int64(bid * MAX_WORLD * 4) + fx.Int64(tid) * fx.Int64(4)
        _spin_until(signal_ptr(self_arr + theirs), flag)

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def a2a_copy(src_ptr: Int64, dev_comm: Int64, win: Int64):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        w = cco.CachedWindow(win)

        # Read side: a plain local pointer, built once per wave.
        source = create_buffer_resource_from_addr(wave_uniform_i64(src_ptr))
        # Write side: my slot in each peer's recv. Rotated by rank so the eight
        # ranks do not all start pushing at destination 0 -- the same reason
        # gemm_ar's all-gather rotates, and gcnasm's stripe scheduling.
        dests = [
            create_buffer_resource_from_addr(
                wave_uniform_i64(w.lsa_ptr((rank + j) % ws, my_recv_slot))
            )
            for j in range(ws)
        ]

        gtid = bid * threads + tid
        stride = blocks * threads
        # Destination is the *inner*, unrolled loop and the pack index the outer
        # one, so a thread holds all `world` links in flight at once. The other
        # nesting -- a thread finishing destination 0 before starting 1 -- keeps
        # exactly one link busy per thread and measured half the per-link
        # bandwidth of the fused path over the same transport. gemm_ar's
        # all-gather says the same thing at its own loop: "issue every load
        # before any store so the links overlap; a load/store pair per peer
        # would serialise on each s_waitcnt".
        #
        # `pk` walks a destination's slab in pack units, which is the *write*
        # side's natural order because that side is compact. `row_packs` is a
        # build-time constant, so the div/mod are a multiply-shift pair.
        for pk in range(gtid, slab_packs, stride):
            row = pk // row_packs
            col_pack = pk % row_packs
            vals = []
            for j in range_constexpr(ws):
                d = (rank + j) % ws
                if const_expr(src == "staging" and d == rank):
                    # The staging GEMM routes its own destination straight into
                    # recv, so that slab is not in staging and is already where
                    # it belongs. Skipping it also makes this path move exactly
                    # the bytes the SDMA scatter moves, which is the point of
                    # being able to compare them.
                    vals.append(None)
                    continue
                if const_expr(src == "staging"):
                    read_at = (d * slab_packs + pk) * I32_PER_PACK
                else:
                    # [M, N]: skip to this row, then into the destination's block.
                    read_at = (
                        row * n_row_packs + d * row_packs + col_pack
                    ) * I32_PER_PACK
                vals.append(buffer_load(source, read_at, vec_width=4, dtype=i32_type()))
            for j in range_constexpr(ws):
                if const_expr(vals[j] is None):
                    continue
                # Compact on the peer side, so the address is just `pk`. That is
                # the side that crosses xGMI, and it is fully contiguous.
                buffer_store(
                    vals[j],
                    dests[j],
                    pk * I32_PER_PACK,
                    cache_modifier=CM_SC0_SC1 if uncached else CM_CACHED,
                )

        # Every store must be visible to the destination before it is told the
        # data is there, and the flag update must not be reordered before them.
        fgpu.barrier()
        flag, flag_rsrc = _next_flag(w, bid)
        if tid < ws:
            _signal_and_wait(w, bid, flag, tid)
        fgpu.barrier()
        if tid == 0:
            local_store_u32(flag_rsrc, flag)

    @flyc.jit
    def run(
        src_ptr: Int64, dev_comm: Int64, win: Int64, stream: fx.Stream = fx.Stream(None)
    ):
        a2a_copy(src_ptr, dev_comm, win).launch(
            grid=(blocks, 1, 1), block=[threads, 1, 1], stream=stream
        )

    return run


def build_lsa_barrier(cfg, rank: int, *, blocks: int = 1):
    """Compile the cross-rank barrier on its own, for the fused path.

    ``fused-lsa`` has no collective kernel: the GEMM epilogue already wrote into
    every peer. What is still missing is the agreement that it *finished* --
    without it a rank reads its ``recv`` while a peer is mid-epilogue, and the
    result is a partially stale slab that validates on some runs and not others.

    Same monotonic-flag protocol as ``build_lsa_a2a``'s tail, so the two cannot
    disagree about the slot map. One block is enough and is the cheapest: this
    kernel moves no data, and the flag array is indexed by block, so more blocks
    would mean more round trips for no extra parallelism.

    The producer's release is *not* here. It cannot be: this kernel is one
    block, so its fence reaches one XCD's L2 out of eight, and the other seven
    would still hold the peer-homed lines dirty while this kernel's
    system-scope atomic overtook them. The GEMM publishes its own stores -- see
    the tail of ``kernels_fused.compile_fused_gemm_a2a``.
    """
    cfg.validate()
    ws = cfg.world_size
    threads = cfg.threads
    start_off, flag_off = cfg.start_off, cfg.flag_off

    def _next_flag(w, bid):
        base = fx.Int64(w.lsa_ptr(rank, flag_off)) + fx.Int64(bid) * fx.Int64(4)
        rsrc = signal_ptr(base)
        return fx.Int32(local_load_u32(rsrc)) + fx.Int32(1), rsrc

    def _signal_and_wait(w, bid, flag, tid):
        """Publish to peer ``tid``, then wait for peer ``tid``'s publication.

        A closure taking ``w``, not inlined into the ``if`` below. A
        ``CachedWindow`` holds three MLIR values, and FlyDSL's scf.if state
        capture requires single values -- inlining this raised
        ``Cannot extract IR values from CachedWindow``. Passing it as a call
        argument keeps it out of the branch's captured state.
        """
        peer_arr = fx.Int64(w.lsa_ptr(tid, start_off))
        mine = fx.Int64((bid * MAX_WORLD + rank) * 4)
        signal_store_u32(signal_ptr(peer_arr + mine), flag)

        self_arr = fx.Int64(w.lsa_ptr(rank, start_off))
        theirs = fx.Int64(bid * MAX_WORLD * 4) + fx.Int64(tid) * fx.Int64(4)
        _spin_until(signal_ptr(self_arr + theirs), flag)

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def barrier(dev_comm: Int64, win: Int64):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        w = cco.CachedWindow(win)
        flag, flag_rsrc = _next_flag(w, bid)
        if tid < ws:
            _signal_and_wait(w, bid, flag, tid)
        fgpu.barrier()
        if tid == 0:
            local_store_u32(flag_rsrc, flag)

    @flyc.jit
    def run(dev_comm: Int64, win: Int64, stream: fx.Stream = fx.Stream(None)):
        barrier(dev_comm, win).launch(
            grid=(blocks, 1, 1), block=[threads, 1, 1], stream=stream
        )

    return run


__all__ = ["build_lsa_a2a", "build_lsa_barrier"]
