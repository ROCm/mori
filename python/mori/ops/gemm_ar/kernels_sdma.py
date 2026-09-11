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
"""SDMA all-reduce in FlyDSL: reduce-scatter push, local reduce, all-gather push.

Same algorithm and the same window as ``kernels_lsa``'s 2-stage, and the same
``2(P-1)/P`` bytes on the wire. The only difference is who moves them: here the
copy engines do, so the CUs are free for compute during the transfers. That is
the property the GEMM fusion in step 3 is after -- raw bandwidth is not expected
to beat LSA.

Three kernels rather than one, because both transfers need a *device-wide*
ordering point (every block must see the landed data), and a kernel boundary is
the only one that is free and graph-capturable:

    1. scatter  lane p pushes my rows of peer p's slice into p's recv slot,
                drains queue p, then bumps p's `start[0][rank]` flag
    2. reduce   fp32-sum my own slice plus the P-1 landed slices into
                `output + rank*slice`
    3. gather   lane p pushes my reduced slice into peer p's output at the same
                offset, drains, then bumps p's `end[0][rank]` flag

Rank-to-rank arrival is signalled over LSA, not SDMA, because FlyDSL's ``Sdma``
exposes no ``wait_signal`` and its ``put`` wrapper hard-codes ``remoteSignal =
false`` (``src/cco/device/cco_device_wrapper.cpp:140``), so a peer cannot observe
an SDMA signal at all. This is the ``opus_sdma_a2a_quiet_notify_kernel`` pattern
from gcnasm: no per-PUT signal, ``quiet_queue`` on the sender, then an LSA atomic
into the peer's flag. The flags, their monotonic epoch and the spin loop are all
reused verbatim from ``kernels_lsa`` -- only row 0 of the signal array is used,
since the pushes are issued by one warp.

Transfer sizing follows the copy engine's one real lever: bytes per op. Each peer
gets exactly one ``put`` of ``slice_bytes``, never split across queues -- one peer
is one xGMI link, so splitting only pays the ~2us per-packet cost again. At
[4096, 7168] bf16 a slice is 7.3MB (~59 GB/s territory); at [64, 7168] it is
114KB, below the ~256KB knee, so decode sizes are expected to lose to LSA.
cco's own chunking is not a factor: ``CCO_SDMA_MAX_COPY_BYTES`` is 1GB, so every
slice we push is a single packet.

Measured on 8x MI355X, [M, 7168] bf16, against the LSA kernel (full table in
``bench_ar.py``): within 5% of LSA for M >= 2048, and faster than aiter at
M = 8192. What is left is mostly a fixed ~40us floor -- two transfer phases times
the engine's ~6us dispatch -- so decode sizes still lose by ~1.5x, as expected
for 114KB slices.

Getting there needed one non-obvious thing: the reduce kernel must **not** share
the LSA all-reduce's grid. Its loads are local HBM, not xGMI, so the cap that
throttles outstanding xGMI requests starves it instead -- 24 blocks left it at
1.11 TB/s on an ~8 TB/s part. Sized independently it runs at 4.9 TB/s, worth 48us
at M=4096 (342 -> 294) and turning a 1.20x deficit against LSA into 1.03x. See
``SDMA_REDUCE_BLOCK_CAP``.

The other reason this backend exists is that its transfers occupy one warp of one
block instead of the whole grid, which is what makes an in-GEMM epilogue possible
in step 3.
"""

from __future__ import annotations


import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int64

import mori.cco.device.flydsl as cco
from mori.cco.device.flydsl import _bindings as raw_cco

from flydsl._mlir import ir

from ._compat import (
    CM_CACHED,
    CM_SC1,
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
    i32_type,
    local_load_u32,
    local_store_u32,
    signal_ptr,
    signal_store_u32,
    wave_uniform_i64,
)
from .kernels_lsa import _spin_until

#: e4m3's largest finite magnitude. ``scale = amax / FP8_E4M3_MAX`` puts a row's
#: biggest element exactly at the top of the range.
FP8_E4M3_MAX = 448.0

#: Wave width, and the width of the amax butterfly below (2**6).
WAVE = 64
WAVE_LOG2 = 6

#: Elements one lane converts per chunk: 8 bf16 is one 16B load, and the 8 fp8
#: it becomes is one 8B store. Both are contiguous across the wave, so both are
#: perfectly coalesced.
CHUNK_ELEMS = 8


def _raw_value(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _bpermute_f32(value, src_lane):
    """``ds_bpermute`` on an f32, through its i32 bit pattern. Byte-addressed."""
    i32 = ir.IntegerType.get_signless(32)
    raw = fx.rocdl.ds_bpermute(
        i32,
        _raw_value(fx.Int32(src_lane) * fx.Int32(4)),
        _raw_value(value.bitcast(fx.Int32)),
    )
    return fx.Int32(raw).bitcast(fx.Float32)


def _pack8_fp8(v8):
    """Eight scaled f32 -> eight e4m3 bytes, as a ``vector<2xi32>``.

    ``v_cvt_pk_fp8_f32`` rather than the one-shot ``pk8`` form: the latter is
    gfx1250, and this has to run on gfx950. Four instructions instead of one,
    which is nothing in a kernel this memory-bound. Note that
    ``arith.truncf`` to fp8 is *not* an option -- it has no LLVM lowering and
    fails in the translation pass, not at trace time.

    ``old`` threads the two halves of a dword through one register: word_sel=0
    writes the low 16 bits, word_sel=1 the high.
    """
    i32 = ir.IntegerType.get_signless(32)
    poison = fx.Int32(0)
    words = []
    for half in range_constexpr(2):
        acc = _raw_value(poison)
        for pair in range_constexpr(2):
            i = half * 4 + pair * 2
            acc = fx.rocdl.cvt_pk_fp8_f32(
                i32,
                _raw_value(v8[i]),
                _raw_value(v8[i + 1]),
                acc,
                pair == 1,
            )
        words.append(fx.Int32(acc))
    return fx.Vector.from_elements(words, fx.Int32)


def _unpack8_fp8(v2i32):
    """Eight e4m3 bytes in a ``vector<2xi32>`` -> eight f32. The inverse."""
    f32x2 = ir.VectorType.get([2], ir.F32Type.get())
    out = []
    for half in range_constexpr(2):
        word = _raw_value(v2i32[half])
        for sel in range_constexpr(2):
            pair = fx.Vector(fx.rocdl.cvt_pk_f32_fp8(f32x2, word, sel == 1))
            out.append(pair[0])
            out.append(pair[1])
    return out


def _wave_amax(value, lane):
    """Max of ``value`` over the wave, as a butterfly -- no LDS, no barrier.

    Every lane ends with the same result, which is what lets each of them scale
    its own chunk without a second broadcast.
    """
    acc = value
    for k in range_constexpr(WAVE_LOG2):
        acc = acc.maximumf(_bpermute_f32(acc, lane ^ fx.Int32(1 << k)))
    return acc


#: The push warp. One lane per peer, each on its own queue, so the lanes post in
#: parallel (cco's ccoSdmaThreadIndependent shape). 64 is the wave size; only the
#: first `world_size` lanes do anything.
PUSH_THREADS = 64


def build_sdma_phases(
    cfg,
    rank: int,
    *,
    queues: int = 8,
    signal: bool = False,
    reduce_blocks=None,
    reduce_self_from_recv: bool = False,
    recv_uncached: bool = False,
):
    """Compile the phases separately, so the fused GEMM can reuse the tail.

    Returns ``{"scatter", "drain", "reduce", "gather"}``, each a launcher taking
    ``(dev_comm, win, stream=...)``. ``scatter`` pushes *and* drains; ``drain``
    only drains and barriers, and exists for the fused path, where the pushes
    were already issued from inside the GEMM epilogue.

    ``queues`` must match ``reqs.sdma_queue_count``. Queue ids are taken modulo
    it, and the hardware queues are per **(source, destination) pair**, so even
    ``queues=1`` still gives every peer its own queue -- which is all this
    pipeline needs, since one destination is one xGMI link and cco's rule is
    about concurrently-issuing *warps* per queue, of which there is one.

    Prefer ``queues=1`` when sharing the process with something else. With
    ``queues=world_size`` a rank creates ``world_size`` hardware queues per peer
    and touches exactly one of them: 56 queues to use 7. Standalone that is
    merely wasteful, but inside a server that already holds SDMA engines for its
    own copies, ``hsaKmtCreateQueueExt`` starts failing
    (``anvil.cpp:237``).

    ``signal`` selects ``put``'s trailing local ATOMIC. It is off by default:
    ``quiet``/``quietQueue`` drain the queue's read pointer and are documented as
    "independent of the signals" (``include/mori/cco/cco.hpp:1707``), and nothing
    here polls a signal, so the packet would be pure overhead. Exposed because
    FlyDSL's own ``Sdma._xfer`` docstring claims the opposite -- that a no-signal
    put "cannot be drained by quiet" -- and that claim is worth being able to
    disprove by measurement rather than by reading.
    """
    cfg.validate()
    ws = cfg.world_size
    if cfg.recv_slots < ws:
        raise ValueError(
            f"SDMA needs a landing slot per peer: build the ArConfig with "
            f"recv_slots={ws} (got {cfg.recv_slots})"
        )
    if queues < 1:
        raise ValueError(f"queues must be >= 1, got {queues}")
    if cfg.fp8_scatter:
        # The regions exist (layout.py sizes them), but nothing writes the
        # quantised payload yet: on the fused path that is the GEMM epilogue's
        # job, and on the standalone path it needs its own kernel. Refuse
        # rather than read an fp8-sized region that still holds bf16, which
        # faults somewhere unrelated.
        raise NotImplementedError(
            "scatter_dtype='fp8' is not wired yet; the gather leg is "
            "(gather_dtype='fp8'), and it is where the time is"
        )

    threads = cfg.threads
    # The reduce is a *local HBM* kernel, so it must not inherit the grid the LSA
    # all-reduce uses. LSA_BLOCK_CAP is small on purpose -- it caps the number of
    # outstanding xGMI requests -- but here every load is local, and 24x512
    # threads x 8 packs is only 1.6MB in flight, about a fifth of what it takes to
    # cover HBM latency. Sized independently, and swept by bench_ar.py.
    red_blocks = reduce_blocks if reduce_blocks else cfg.reduce_blocks
    red_stride = red_blocks * threads
    elem_dtype = fx.BFloat16 if cfg.elem_bytes == 2 else fx.Float32
    part = cfg.packs_per_rank
    slice_bytes = cfg.slice_bytes

    start_off, end_off, flag_off = cfg.start_off, cfg.end_off, cfg.flag_off
    in_off, out_off = cfg.input_off, cfg.output_off
    # Constant-folded because `rank` is a Python int: the slice I own, and the
    # slot I occupy in every peer's recv region.
    my_slice_off = rank * slice_bytes
    my_recv_slot = cfg.recv_slot_off(rank)

    I32_PER_PACK = 4  # a 16B pack always moves as 4 x i32, whatever the payload

    def _next_flag(w, bid):
        """``_flag[block] + 1`` -- monotonic, never reset (graph-replay safe)."""
        base = fx.Int64(w.lsa_ptr(rank, flag_off)) + fx.Int64(bid) * fx.Int64(4)
        rsrc = signal_ptr(base)
        return fx.Int32(local_load_u32(rsrc)) + fx.Int32(1), rsrc

    def _signal_and_wait(w, arr_off, flag, tid):
        """Row-0 barrier: publish to peer ``tid``, then wait on ``tid``'s slot."""
        peer_arr = fx.Int64(w.lsa_ptr(tid, arr_off))
        signal_store_u32(signal_ptr(peer_arr + fx.Int64(rank * 4)), flag)
        self_arr = fx.Int64(w.lsa_ptr(rank, arr_off))
        _spin_until(signal_ptr(self_arr + fx.Int64(tid) * fx.Int64(4)), flag)

    def _push_kernel(
        arr_off,
        dst_off_of_peer=None,
        src_off_expr=None,
        nbytes=None,
        extra=None,
    ):
        """A push phase: one put per peer, drain, then the cross-rank barrier.

        ``dst_off_of_peer`` is a constant byte offset *in the destination's*
        window; ``src_off_expr(tid)`` builds the offset in mine, which may depend
        on which peer the lane is serving. Passing neither emits a drain-only
        kernel -- the pushes are assumed already issued elsewhere.

        ``extra`` is an optional second ``(dst_off, src_off_expr, nbytes)`` put on
        the same queue, for the fp8 wire's scales. It rides the same queue rather
        than its own so it cannot overtake the payload, and it is ~8 KiB against
        14 MiB -- about 2us of packet cost, against the ~210us the halved payload
        saves.
        """
        pushes = dst_off_of_peer is not None
        put_bytes = slice_bytes if nbytes is None else nbytes

        @flyc.kernel(known_block_size=[PUSH_THREADS, 1, 1])
        def push(dev_comm: Int64, win: Int64):
            tid = fx.thread_idx.x
            w = cco.Window(win)
            sdma = cco.DevComm(dev_comm).sdma()

            flag, flag_rsrc = _next_flag(w, 0)
            if tid < ws:
                if tid != rank:
                    if const_expr(pushes):
                        # Lane `tid` owns peer `tid` and queue `tid`: distinct
                        # queues per lane, so the posts do not serialise on one
                        # commit chain.
                        sdma.put(
                            tid,
                            win,
                            fx.Int64(dst_off_of_peer),
                            win,
                            src_off_expr(tid),
                            fx.Int64(put_bytes),
                            tid % fx.Int32(queues),
                            coop=cco.CoopScope.THREAD,
                            signal=signal,
                        )
                        if const_expr(extra is not None):
                            sdma.put(
                                tid,
                                win,
                                fx.Int64(extra[0]),
                                win,
                                extra[1](tid),
                                fx.Int64(extra[2]),
                                tid % fx.Int32(queues),
                                coop=cco.CoopScope.THREAD,
                                signal=signal,
                            )
                    sdma.quiet_queue(tid, tid % fx.Int32(queues))
                # Release before publishing arrival. The bytes were moved by the
                # copy engine rather than by this CU, so there is nothing of ours
                # to flush; the fence is here to keep the flag store from being
                # hoisted above the drain, which is the one ordering the receiver
                # depends on.
                raw_cco.cco_system_fence(fx.Int32(0))
                _signal_and_wait(w, arr_off, flag, tid)
            fgpu.barrier()
            if tid == 0:
                local_store_u32(flag_rsrc, flag)

        return push

    # Scatter. Peer p owns slice p, so what p needs from me is *p's* slice range
    # of my input -- the source offset moves with the lane. It lands in the slot p
    # reserves for me, which is at the same byte offset in every rank's window.
    scatter = _push_kernel(
        start_off,
        dst_off_of_peer=my_recv_slot,
        src_off_expr=lambda tid: fx.Int64(in_off)
        + fx.Int64(tid) * fx.Int64(slice_bytes),
    )

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def sdma_reduce(dev_comm: Int64, win: Int64):
        """Sum my own slice plus the P-1 landed slices into my part of output.

        Peer order is ``(rank + j) % ws`` to match ``kernels_lsa``'s 2-stage, so
        the two backends are bitwise identical and the test can assert
        ``rel_l2 == 0`` rather than a tolerance.
        """
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        w = cco.Window(win)

        # j=0 is my own contribution. Normally it is still sitting in my input
        # region; with the Direct-LSA fused GEMM the epilogue wrote it into my own
        # recv slot instead, along with everyone else's.
        self_off = (
            cfg.recv_slot_off(rank) if reduce_self_from_recv else in_off + my_slice_off
        )
        srcs = [
            create_buffer_resource_from_addr(
                wave_uniform_i64(w.lsa_ptr(rank, self_off))
            )
        ] + [
            create_buffer_resource_from_addr(
                wave_uniform_i64(w.lsa_ptr(rank, cfg.recv_slot_off((rank + j) % ws)))
            )
            for j in range(1, ws)
        ]
        out = create_buffer_resource_from_addr(
            wave_uniform_i64(w.lsa_ptr(rank, out_off + my_slice_off))
        )

        gtid = bid * threads + tid
        for pk in range(gtid, part, red_stride):
            i32_off = pk * I32_PER_PACK
            acc = None
            for j in range_constexpr(ws):
                # recv is filled by a *peer*. When the copy engine wrote it the
                # lines are coherent, but when a peer's CUs stored into it over
                # xGMI our own L2 is never invalidated, so a cached load here can
                # return the previous iteration's bytes. SC1 skips L2 for those.
                raw = fx.Vector(
                    buffer_load(
                        srcs[j],
                        i32_off,
                        vec_width=4,
                        dtype=i32_type(),
                        cache_modifier=CM_SC1 if recv_uncached else CM_CACHED,
                    )
                )
                v = (
                    raw.bitcast(fx.Float32)
                    if elem_dtype is fx.Float32
                    else raw.bitcast(elem_dtype).to(fx.Float32)
                )
                acc = v if acc is None else acc + v
            packed = (
                acc.bitcast(fx.Int32)
                if elem_dtype is fx.Float32
                else acc.to(elem_dtype).bitcast(fx.Int32)
            )
            buffer_store(packed, out, i32_off, cache_modifier=CM_CACHED)

    # --- fp8 wire: the two conversions that bracket the gather ---------------
    #
    # quantise runs after the reduce on my own slice; dequantise runs after the
    # gather on everybody else's. My own rows never travel, so they stay the
    # exact bf16 the reduce produced -- only the 7/8 that cross a link are
    # rounded, and each of them exactly once.

    QUANT_WAVES = 4  # waves per block; one wave owns one row
    QUANT_THREADS = QUANT_WAVES * WAVE

    if cfg.fp8_gather:
        if cfg.n % (CHUNK_ELEMS * WAVE):
            raise ValueError(
                f"the fp8 wire gives one wave one row and converts "
                f"{CHUNK_ELEMS} elements per lane per chunk, so n={cfg.n} must "
                f"be a multiple of {CHUNK_ELEMS * WAVE}"
            )
        chunks_per_row = cfg.n // (CHUNK_ELEMS * WAVE)
        slice_rows = cfg.slice_rows
        quant_blocks = max(1, min(1024, (slice_rows + QUANT_WAVES - 1) // QUANT_WAVES))

        def _row_addr(w, byte_off):
            return create_buffer_resource_from_addr(
                wave_uniform_i64(w.lsa_ptr(rank, byte_off))
            )

        @flyc.kernel(known_block_size=[QUANT_THREADS, 1, 1])
        def quantize_gather(dev_comm: Int64, win: Int64):
            """My reduced slice, bf16 -> fp8 + one fp32 scale per row.

            Reads `output + my_slice` (what the reduce just wrote) and writes
            `gout + my_slice` plus `gout_scale`. Two passes over the row rather than
            holding it in registers: 112 fp32 per lane would not fit, and the row is
            14 KiB and still in L2 from the reduce.
            """
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            w = cco.Window(win)
            lane = tid % fx.Int32(WAVE)
            wave = tid // fx.Int32(WAVE)

            src = _row_addr(w, out_off + my_slice_off)
            dst = _row_addr(w, cfg.gout_off + rank * cfg.slice_rows * cfg.n)
            sca = _row_addr(w, cfg.gout_scale_slice_off(rank))

            for row in range(
                bid * QUANT_WAVES + wave, slice_rows, quant_blocks * QUANT_WAVES
            ):
                # pass 1: this row's amax, over the whole wave
                local = fx.Float32(0.0)
                for c in range_constexpr(chunks_per_row):
                    i32_off = (row * cfg.n + (c * WAVE) * CHUNK_ELEMS) // 2 + lane * 4
                    v = (
                        fx.Vector(
                            buffer_load(
                                src,
                                i32_off,
                                vec_width=4,
                                dtype=i32_type(),
                                cache_modifier=CM_CACHED,
                            )
                        )
                        .bitcast(fx.BFloat16)
                        .to(fx.Float32)
                    )
                    for e in range_constexpr(CHUNK_ELEMS):
                        a = v[e]
                        local = local.maximumf((-a).maximumf(a))
                amax = _wave_amax(local, lane)

                # A row of exact zeros would divide by zero; 1.0 keeps it exact.
                is_zero = amax == fx.Float32(0.0)
                scale = is_zero.select(
                    fx.Float32(1.0), amax * fx.Float32(1.0 / FP8_E4M3_MAX)
                )
                inv = is_zero.select(fx.Float32(1.0), fx.Float32(FP8_E4M3_MAX) / amax)
                if lane == fx.Int32(0):
                    buffer_store(scale, sca, row, cache_modifier=CM_CACHED)

                # pass 2: scale and narrow
                for c in range_constexpr(chunks_per_row):
                    base = row * cfg.n + (c * WAVE) * CHUNK_ELEMS
                    v = (
                        fx.Vector(
                            buffer_load(
                                src,
                                base // 2 + lane * 4,
                                vec_width=4,
                                dtype=i32_type(),
                                cache_modifier=CM_CACHED,
                            )
                        )
                        .bitcast(fx.BFloat16)
                        .to(fx.Float32)
                    )
                    scaled = v * inv
                    q = _pack8_fp8([scaled[e] for e in range_constexpr(CHUNK_ELEMS)])
                    buffer_store(
                        q,
                        dst,
                        (base + lane * CHUNK_ELEMS) // 4,
                        cache_modifier=CM_CACHED,
                    )

        @flyc.kernel(known_block_size=[QUANT_THREADS, 1, 1])
        def dequantize_gather(dev_comm: Int64, win: Int64):
            """Every peer's gathered slice, fp8 * scale -> bf16 into `output`.

            Skips my own slice: it was never quantised and is already in `output`.
            """
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            w = cco.Window(win)
            lane = tid % fx.Int32(WAVE)
            wave = tid // fx.Int32(WAVE)

            for j in range_constexpr(1, ws):
                peer = (rank + j) % ws
                src = _row_addr(w, cfg.gout_off + peer * cfg.slice_rows * cfg.n)
                sca = _row_addr(w, cfg.gout_scale_slice_off(peer))
                dst = _row_addr(w, out_off + peer * cfg.slice_rows * cfg.n * 2)

                for row in range(
                    bid * QUANT_WAVES + wave, slice_rows, quant_blocks * QUANT_WAVES
                ):
                    # Written by a peer's copy engine. SC1 for the same reason the
                    # reduce uses it on `recv`: our L2 is never invalidated by a
                    # remote store, so a cached load can return stale bytes.
                    scale = fx.Float32(
                        buffer_load(
                            sca,
                            row,
                            vec_width=1,
                            dtype=fx.Float32,
                            cache_modifier=CM_SC1,
                        )
                    )
                    for c in range_constexpr(chunks_per_row):
                        base = row * cfg.n + (c * WAVE) * CHUNK_ELEMS
                        packed = fx.Vector(
                            buffer_load(
                                src,
                                (base + lane * CHUNK_ELEMS) // 4,
                                vec_width=2,
                                dtype=i32_type(),
                                cache_modifier=CM_SC1,
                            )
                        )
                        wide = [e * scale for e in _unpack8_fp8(packed)]
                        v = fx.Vector.from_elements(wide, fx.Float32)
                        buffer_store(
                            v.to(fx.BFloat16).bitcast(fx.Int32),
                            dst,
                            base // 2 + lane * 4,
                            cache_modifier=CM_CACHED,
                        )

    # Gather. My reduced slice goes to the same offset in every peer's window.
    # On the bf16 wire that offset is in `output` and there is nothing to stage;
    # on the fp8 wire it is `gout`, which `quantize_gather` filled, and the
    # scales ride along on the same queue.
    if cfg.fp8_gather:
        my_gout_off = cfg.gout_off + rank * cfg.slice_rows * cfg.n
        my_gscale_off = cfg.gout_scale_slice_off(rank)
        gather = _push_kernel(
            end_off,
            dst_off_of_peer=my_gout_off,
            src_off_expr=lambda tid: fx.Int64(my_gout_off),
            nbytes=cfg.gather_slice_bytes,
            extra=(
                my_gscale_off,
                lambda tid: fx.Int64(my_gscale_off),
                cfg.gather_scale_slice_bytes,
            ),
        )
    else:
        gather = _push_kernel(
            end_off,
            dst_off_of_peer=out_off + my_slice_off,
            src_off_expr=lambda tid: fx.Int64(out_off + my_slice_off),
        )

    # Drain-only twin of `scatter`: same barrier, no puts. The fused GEMM issues
    # the puts from its epilogue, but still has to drain the queues and tell the
    # peers their slices landed.
    drain = _push_kernel(start_off)

    def _phase(kern, grid, blk):
        @flyc.jit
        def go(dev_comm: Int64, win: Int64, stream=fx.Stream(None)):
            kern(dev_comm, win).launch(
                grid=(grid, 1, 1), block=[blk, 1, 1], stream=stream
            )

        return go

    phases = {
        "scatter": _phase(scatter, 1, PUSH_THREADS),
        "drain": _phase(drain, 1, PUSH_THREADS),
        "reduce": _phase(sdma_reduce, red_blocks, threads),
        "gather": _phase(gather, 1, PUSH_THREADS),
    }
    if cfg.fp8_gather:
        # Bracket the gather. `quantize` has to follow the reduce and precede the
        # push; `dequantize` has to follow the barrier the push ends with.
        phases["quantize"] = _phase(quantize_gather, quant_blocks, QUANT_THREADS)
        phases["dequantize"] = _phase(dequantize_gather, quant_blocks, QUANT_THREADS)
    # The order a caller must run them in. Carried with the phases rather than
    # left to each caller to hardcode: the fp8 wire adds two, and a caller that
    # kept its own ("scatter", "reduce", "gather") list would silently skip them
    # and all-reduce into zeros.
    phases["order"] = (
        ("scatter", "reduce", "quantize", "gather", "dequantize")
        if cfg.fp8_gather
        else ("scatter", "reduce", "gather")
    )
    return phases


def build_sdma_ar(cfg, rank: int, *, queues: int = 8, signal: bool = False):
    """Compile a whole SDMA all-reduce for ``cfg`` on ``rank``.

    Returns ``(run, stage)``; ``stage`` is always 2 -- there is no 1-stage SDMA
    variant, since a copy engine gains nothing from broadcasting the whole
    payload to every peer.
    """
    parts = build_sdma_phases(cfg, rank, queues=queues, signal=signal)
    # On the fp8 wire this is five kernels, not three: the reduced slice has to
    # be narrowed before the push and widened after it, and each needs its own
    # device-wide ordering point for the same reason the original three do.
    seq = [parts[k] for k in parts["order"]]

    def run(dev_comm, win, stream=fx.Stream(None)):
        for phase in seq:
            phase(dev_comm, win, stream=stream)

    return run, 2


__all__ = ["build_sdma_ar", "build_sdma_phases", "PUSH_THREADS"]
