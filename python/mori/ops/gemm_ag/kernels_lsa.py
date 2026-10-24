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
"""The all-gather over LSA, in both directions.

This is the split baseline. The GEMM has already written this rank's whole
``[M, N]`` result into its own ``recv`` slot, and every other rank needs a copy
of it. The transfer is plain vector stores or loads against a peer pointer from
cco's flat symmetric VA -- the same substrate ``gemm_ar``'s and ``gemm_a2a``'s
``kernels_lsa`` use.

## The shape of the copy

A source's slab sits at the *same* byte offset in every rank's window, so:

    push   read  local  recv[rank] at pk   ->  write peer d's recv[rank] at pk
    pull   read  peer d's recv[d]  at pk   ->  write local  recv[d]      at pk

Source index and destination index are equal in both. That is the whole
difference from ``gemm_a2a``'s copy kernel, which has to reconcile a row-strided
read (``C[row, d*shard_n + j]``) with a compact write; there is no re-layout
here at all, in either direction.

## Why both directions exist

All-gather is a broadcast, so it can be driven from either end, and the two are
*not* symmetric in what they cost:

* **push** issues one load and ``world-1`` stores per pack. The stores all
  depend on that one load, so a thread's loop body has its memory latency fully
  exposed unless the packs are unrolled -- hence ``unroll``.
* **pull** issues ``world-1`` loads and ``world-1`` stores per pack. The loads
  are independent of each other, so the peers themselves provide the
  latency hiding and the loop needs no unrolling. This is also the direction
  ``gemm_ar``'s own gather leg takes, where it beat the SDMA push by 5.6-10.5%.

The second reason to keep pull is forward-looking: a low-precision wire wants
the dequantize on the *consumer* side, where the narrow value is already in a
register and widening it is free. ``gemm_ar`` measured exactly that -- an SDMA
push needs a separate 61.0us widen kernel, an LSA pull needs none. Push cannot
have that property, because the producer would have to widen before sending,
which is the thing narrowing was for.

Only push has a fused twin: the epilogue runs on the producer.

## Completion

Both end with the same monotonic-flag barrier as ``gemm_ar``'s and
``gemm_a2a``'s LSA copies: signal every peer, spin until every peer has
signalled. Flags are never reset, so graph replay is safe. Without it a rank can
return while a peer is still writing into its ``recv`` (push), or start reading
a peer that has not finished its GEMM (pull -- see the note at ``build_lsa_ag``).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int64

import mori.cco.device.flydsl as cco
from mori.cco.device.flydsl import _bindings as raw_cco

# The buffer primitives and cache-policy constants are gemm_ar's: this op
# differs in where C goes, not in how any of that works, and a second copy of
# them would be a second thing to keep in step with flydsl's API.
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

#: Bytes in a pack. Elements per pack is ``PACK_BYTES // cfg.elem_bytes`` -- 8
#: for a bf16 output, 4 for fp32 -- and is derived per config rather than fixed,
#: because this operator now carries both. Only the *count* changes: the copy
#: still moves 16 bytes per lane either way, which is what keeps the peer-side
#: store fully contiguous.
PACK_BYTES = 16

#: Packs a push thread keeps in flight. See the module docstring: push has one
#: load feeding world-1 stores, so without this the loop body is a load, a
#: wait, and a burst of stores. Only used when it divides the work evenly --
#: ``_pick_unroll`` falls back to 1 rather than emitting a ragged tail.
DEFAULT_PUSH_UNROLL = 4


def _spin_until(rsrc, flag):
    """Block until the u32 at ``rsrc`` reaches ``flag`` (unsigned compare).

    Unsigned so a wrapped counter still terminates; ``>=`` so an increment can
    never be missed. Same construction as ``gemm_ar.kernels_lsa._spin_until``
    and ``gemm_a2a``'s -- duplicated rather than imported because those modules
    are other collectives and importing a private helper across ops to save nine
    lines would tie this file's correctness to an unrelated one's refactors.
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


def _pick_unroll(slab_packs: int, stride: int, requested: int) -> int:
    """The largest factor at or below ``requested`` that divides the work.

    An unrolled body reads ``base + u*stride``, which keeps a load's lanes
    contiguous; that only stays in bounds for every ``u`` if the per-thread
    iteration count is a whole multiple of the factor. Rounding down rather than
    emitting a bounds check keeps the inner loop branch-free, and rather than
    raising keeps a sweep over shapes usable.
    """
    for u in range(max(1, requested), 0, -1):
        if slab_packs % (stride * u) == 0:
            return u
    return 1


def build_lsa_ag(
    cfg,
    rank: int,
    *,
    direction: str = "push",
    uncached: bool = True,
    unroll: int = DEFAULT_PUSH_UNROLL,
):
    """Compile an LSA all-gather launcher for ``cfg`` on ``rank``.

    Returns ``run(dev_comm, win, stream=...)``. Unlike ``gemm_a2a``'s copy
    kernel this takes no source pointer: both ends of the transfer are the
    symmetric ``recv`` region, because the GEMM wrote its result there rather
    than into a private buffer. That is what removes the staging copy.

    ``direction``:

    * ``"push"`` -- read my own slab once, store it into every peer's slot for
      me. The fused epilogue is this, done tile by tile, which is why this is
      the direction the fused/split pair is compared on.
    * ``"pull"`` -- read every peer's slab out of its own window, store into
      mine. **This needs a barrier before it**, not only after: a push writes
      into a peer that is merely idle, but a pull reads a peer that may still be
      running its GEMM. The launcher therefore barriers on entry; the caller
      does not have to.

      It also imposes a requirement the caller *does* have to meet: **the GEMM
      feeding a pull must publish its C with system scope.** A pull reads a
      peer's HBM over xGMI; an ordinary cached store leaves the line dirty in
      the producer's own L2, which that read never reaches. The barrier cannot
      fix this from inside this kernel -- ``cco_system_fence`` is
      ``__threadfence_system()``, which orders the calling thread's own writes
      and cannot write back lines another kernel left behind.

      The window's control region must also be zero-initialized before the
      first launch. Allocation alone does not do this: a stale positive
      arrival flag satisfies the first ``>=`` comparison and lets a block
      read a peer before its GEMM finishes. Keep these flags monotonic across
      subsequent launches; do not reset them during graph replay.

      Concretely: compile the fp8 GEMM with
      ``compile_fused_gemm_ag(..., transport="sdma", fuse=False,
      peer_uncached=True)``, whose epilogue stores through a buffer descriptor
      with ``sc0|sc1``. Handing this kernel ``compile_gemm_local``'s output
      instead is not a clean failure -- it validated at relL2 1.66e-3 under
      ``--quant ptpc`` and 2.05e-1 under ``--quant blockscale``, because the
      blockscale GEMM is 96us against ptpc's 57 and the wider rank skew widens
      the window with it. That is why the mode matrix is run across all three
      quantisations rather than spot-checked on the default.

      The bf16 counterpart is ``compile_bf16_gemm_ag(..., fuse=False,
      peer_uncached=True)``: it uses ``sc0|sc1`` stores and an explicit
      system fence from every producer lane before returning.

    ``uncached`` sends the fabric-crossing access with ``sc0|sc1`` instead of
    letting it sit in a cache. It is on by default for two reasons, and the
    correctness one came first:

    * On push, a cached store to peer-homed memory leaves the line dirty in
      whichever XCD's L2 the block ran on, and the barrier that follows
      publishes arrival with a *system-scope* atomic, which can overtake it.
      gemm_ar hit exactly this. Nothing here fences before signalling, so cached
      stores were relying on luck.
    * On pull it is the same bit for the mirrored reason -- ``sc1`` is what
      makes a read observe a peer's fresh write rather than a stale L2 line.
    * It is also what the bandwidth wants: gemm_a2a measured cached peer stores
      reaching 58% of line rate where the copy engines reach 94%.

    The reason gemm_ar could not always use it is that a 2-byte partial-line
    write over the fabric loses updates. This kernel moves 16 bytes per lane and
    1024 contiguous bytes per wave, so that objection does not apply.

    Each kernel is specialised on the Python ``rank`` so every window offset
    folds to a constant, as ``examples/cco/python/07_flydsl_sdma`` does.
    """
    cfg.validate()
    if direction not in ("push", "pull"):
        raise ValueError(f"direction must be push or pull, got {direction!r}")

    ws = cfg.world_size
    threads = cfg.threads
    blocks = cfg.copy_blocks
    m, n = cfg.m, cfg.n

    start_off, flag_off = cfg.start_off, cfg.flag_off

    pack_elems = PACK_BYTES // cfg.elem_bytes
    if (m * n) % pack_elems:
        raise ValueError(
            f"m*n={m * n} must be a multiple of {pack_elems} so a slab is a "
            f"whole number of {PACK_BYTES}B packs at elem_bytes={cfg.elem_bytes}"
        )
    slab_packs = m * n // pack_elems
    stride = blocks * threads
    # Only push benefits; pull already has world-1 independent loads in flight.
    unroll = _pick_unroll(slab_packs, stride, unroll if direction == "push" else 1)
    fabric_cm = CM_SC0_SC1 if uncached else CM_CACHED

    # Peer order, rotated by rank so the eight ranks do not all start at the
    # same partner -- the same reason gemm_ar's all-gather rotates, and gcnasm's
    # stripe scheduling. Self is excluded: on push my slab is already in my own
    # recv slot, and on pull it is the slot I would be reading.
    peers = [(rank + j) % ws for j in range(1, ws)]

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

    # A pull reads a peer that may still be mid-GEMM, so it needs agreement on
    # entry as well as on exit. Two flag epochs per launch, one before and one
    # after; the counter is monotonic so they cannot be confused.
    entry_barrier = direction == "pull"
    kname = (
        f"mori_ag_lsa_{direction}_r{rank}_w{ws}_" f"{'u' if uncached else 'c'}x{unroll}"
    )

    @flyc.kernel(name=kname, known_block_size=[threads, 1, 1])
    def ag_copy(dev_comm: Int64, win: Int64):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        w = cco.CachedWindow(win)

        if const_expr(entry_barrier):
            # A pull reads memory a *peer's GEMM* wrote, so it needs agreement
            # that the GEMM finished before the first load, not only after the
            # last store. A push does not: it writes into a peer that is merely
            # idle.
            #
            # The release half of this pair is **not here and cannot be**.
            # ``cco_system_fence`` is ``__threadfence_system()``, which orders
            # the *calling thread's own* prior writes; this kernel's threads
            # wrote nothing, and the lines to publish belong to a different
            # kernel's L2. Publishing them is the producer's job, and the
            # producer is the GEMM -- see ``build_lsa_ag``'s docstring, which
            # says what a caller has to hand this kernel.
            flag_in, flag_in_rsrc = _next_flag(w, bid)
            if tid < ws:
                _signal_and_wait(w, bid, flag_in, tid)
            # The acquire half, which *is* ours: invalidate any line this rank
            # holds for a peer's slab. The barrier's atomic load is agent-scope
            # monotonic, so it orders but does not invalidate. gemm_ar carries
            # the same fence at the same point in its own all-gather stage,
            # with the same note; aiter gets it from __ATOMIC_ACQUIRE on its
            # spin load. leaderOnly=0 -- every lane, since the cco ABI says
            # leaderOnly=1 orders thread 0 only.
            raw_cco.cco_system_fence(fx.Int32(0))
            fgpu.barrier()
            if tid == 0:
                local_store_u32(flag_in_rsrc, flag_in)
            fgpu.barrier()

        # One side of the transfer is local and one is a peer; which is which is
        # the whole of `direction`. `srcs[j]` and `dsts[j]` are paired, and in
        # both cases the *index* into them is the same `pk`.
        if const_expr(direction == "push"):
            local = create_buffer_resource_from_addr(
                wave_uniform_i64(w.lsa_ptr(rank, cfg.recv_slot_off(rank)))
            )
            srcs = [local] * (ws - 1)
            dsts = [
                create_buffer_resource_from_addr(
                    wave_uniform_i64(w.lsa_ptr(d, cfg.recv_slot_off(rank)))
                )
                for d in peers
            ]
        else:
            srcs = [
                create_buffer_resource_from_addr(
                    wave_uniform_i64(w.lsa_ptr(d, cfg.recv_slot_off(d)))
                )
                for d in peers
            ]
            dsts = [
                create_buffer_resource_from_addr(
                    wave_uniform_i64(w.lsa_ptr(rank, cfg.recv_slot_off(d)))
                )
                for d in peers
            ]

        gtid = bid * threads + tid
        # Peer is the *inner*, unrolled loop and the pack index the outer one,
        # so a thread holds all world-1 links in flight at once. The other
        # nesting -- a thread finishing peer 0 before starting 1 -- keeps
        # exactly one link busy per thread and gemm_a2a measured half the
        # per-link bandwidth from it. gemm_ar's all-gather says the same thing
        # at its own loop: "issue every load before any store so the links
        # overlap; a load/store pair per peer would serialise on each
        # s_waitcnt".
        #
        # On push there is only one load to issue, so `unroll` supplies the
        # independent work instead: `u` strides by the whole grid, which keeps
        # each load's lanes contiguous (striding by 1 would give lane l pack
        # l*unroll and shred the coalescing).
        for base in range(gtid, slab_packs, stride * unroll):
            vals = []
            for u in range_constexpr(unroll):
                at = (base + u * stride) * I32_PER_PACK
                for j in range_constexpr(ws - 1):
                    vals.append(
                        buffer_load(
                            srcs[j],
                            at,
                            vec_width=4,
                            dtype=i32_type(),
                            # The read crosses the fabric only on pull.
                            cache_modifier=(
                                fabric_cm
                                if const_expr(direction == "pull")
                                else CM_CACHED
                            ),
                        )
                    )
            for u in range_constexpr(unroll):
                at = (base + u * stride) * I32_PER_PACK
                for j in range_constexpr(ws - 1):
                    buffer_store(
                        vals[u * (ws - 1) + j],
                        dsts[j],
                        at,
                        # ...and the write crosses it only on push.
                        cache_modifier=(
                            fabric_cm if const_expr(direction == "push") else CM_CACHED
                        ),
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
    def run(dev_comm: Int64, win: Int64, stream: fx.Stream = fx.Stream(None)):
        ag_copy(dev_comm, win).launch(
            grid=(blocks, 1, 1), block=[threads, 1, 1], stream=stream
        )

    return run


def build_lsa_barrier(cfg, rank: int, *, blocks: int = 1):
    """Compile the cross-rank barrier on its own, for the fused path.

    ``fused-lsa`` has no collective kernel: the GEMM epilogue already wrote into
    every peer. What is still missing is the agreement that it *finished* --
    without it a rank reads its ``recv`` while a peer is mid-epilogue, and the
    result is a partially stale slab that validates on some runs and not others.

    Same monotonic-flag protocol as ``build_lsa_ag``'s tail, so the two cannot
    disagree about the slot map. One block is enough and is the cheapest: this
    kernel moves no data, and the flag array is indexed by block, so more blocks
    would mean more round trips for no extra parallelism.

    The producer's release is *not* here. It cannot be: this kernel is one
    block, so its fence reaches one XCD's L2 out of eight, and the other seven
    would still hold the peer-homed lines dirty while this kernel's
    system-scope atomic overtook them. The GEMM publishes its own stores -- see
    the tail of ``kernels_fused.compile_fused_gemm_ag``.
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


__all__ = ["build_lsa_ag", "build_lsa_barrier", "DEFAULT_PUSH_UNROLL"]
