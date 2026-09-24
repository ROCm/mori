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
"""fp8 GEMM + all-gather on cco, by mode.

    gemm-only       the GEMM alone into a plain tensor, to size the ceiling
    gemm-to-window  the same kernel, writing this rank's recv slot instead
    split-rccl      gemm-only(); then torch's all_gather_into_tensor
    split-lsa-push  gemm-to-window(); then LSA stores into every peer
    split-lsa-pull  gemm-to-window(); then LSA loads from every peer
    fused-lsa       C stored straight into every peer from the epilogue
    split-sdma      gemm-to-window(); then one copy-engine push per peer
    fused-sdma      C into my own slot, pushed per chunk from the epilogue

Every rank holds its own ``A [M, K]`` and a **replicated** ``B [N, K]``,
computes the whole ``[M, N]``, and every rank ends up with all ``world`` of them
concatenated along rows. That B is replicated is the thing to keep in mind when
reading this against ``bench_gemm_ar.py``: there, every rank holds a different
K-shard of both operands and the collective sums them. Here every rank computes
the same function of different rows, and the collective only moves bytes. So B
is seeded identically on every rank and A is not -- get that wrong and each rank
validates happily against its own wrong reference.

``split-rccl`` is the baseline the fusion is claimed against: a plain GEMM into
a normal tensor followed by the collective a model would actually call. The two
``split-lsa-*`` modes and ``split-sdma`` are the same decomposition over mori's
own transports, and the ``fused-*`` pair is what absorbs the transfer into the
epilogue.

Note there is no ``gemm-staging``. ``bench_gemm_a2a`` needs one because its
split path's GEMM writes a different address map from ``gemm-only``'s, so
"split minus gemm-only" would charge the transfer for the difference between two
kernels. Here the split path's GEMM *is* ``gemm-only``'s kernel with a different
C pointer, which is what ``gemm-to-window`` measures -- and that mode validates,
unlike its all-to-all namesake, because writing ``[M, N]`` into this rank's recv
slot is exactly half the answer rather than none of it.

    torchrun --nproc_per_node=8 bench_gemm_ag.py --mode split-lsa-pull \\
      -m 2048 --out-dim 2048 -k 7168 --warmup 10 --iters 30
"""

import argparse
import json
import os
import statistics
import sys

import flydsl.expr as fx
import torch
import torch.distributed as dist

from mori.cco import (
    CCODevCommRequirements,
    Communicator,
    GDA_CONNECTION_NONE,
    UniqueId,
)
from mori.tensor_utils import from_gpu_ptr

from mori.ops.gemm_ar import preshuffle_a_scale
from mori.ops.gemm_ag import (
    ag_config,
    build_lsa_ag,
    build_lsa_barrier,
    build_sdma_phases,
    counter_chunks,
    preshuffle_b,
)
from mori.ops.gemm_ag.layout import DEFAULT_BLOCK_M, MXFP8_BLOCK_M
from mori.ops.gemm_ag.kernels_fused import (
    compile_fused_gemm_ag,
    compile_gemm_local,
)
from mori.ops.gemm_ag._gemm_a16w16_8wave import (
    BLOCK_K as BF16_BLOCK_K,
    compile_bf16_gemm_ag,
    pick_tile,
)

MODES = (
    "gemm-only",
    "gemm-to-window",
    "split-rccl",
    "split-lsa-push",
    "split-lsa-pull",
    "fused-lsa",
    "split-sdma",
    "fused-sdma",
)
NOT_YET = ()

#: Modes whose GEMM writes straight into this rank's recv slot rather than into
#: a private tensor. Everything except the two that hand C to something else:
#: ``gemm-only`` has no collective and ``split-rccl``'s collective is a torch
#: call that would then be aliasing part of its own output.
WINDOW_C = (
    "gemm-to-window",
    "split-lsa-push",
    "split-lsa-pull",
    "split-sdma",
    "fused-sdma",
    "fused-lsa",
)

#: fp8 block-scale group size along K. Fixed by the model's quantiser and by the
#: kernel, whose BLOCK_K is already 128.
SCALE_BK = 128

#: ue8m0 group, on both operands: A per 32 K, B per 32x32.
MXFP8_BK = 32


def _ue8m0_bytes(shape, g, lo=120, hi=123):
    """Random ue8m0 exponent bytes, i.e. scales 2**(byte-127) around 1e-2.

    ue8m0 *is* the exponent: there is no mantissa, so every scale is exactly a
    power of two and applying it is lossless. That is the whole reason the
    scaled MFMA can take it as an operand rather than as epilogue arithmetic.
    """
    e = torch.randint(lo, hi, shape, generator=g, device="cuda", dtype=torch.int32)
    return e.to(torch.uint8)


def _ue8m0_value(e: torch.Tensor) -> torch.Tensor:
    return torch.exp2(e.to(torch.float32) - 127.0)


def _setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # gloo for the object broadcasts the bootstrap needs, nccl (RCCL here)
        # for the collective baseline. Asking for both up front is cheaper than
        # a second group later and costs nothing when the baseline is not run.
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return local_rank, rank, world_size, UniqueId.from_bytes(payload[0])


def make_operands(rank: int, m: int, n: int, k: int, quant: str = "ptpc"):
    """Per-rank A, **replicated** B, and their scales.

    The replication is load-bearing. Every rank computes the same ``B``, so the
    rows a rank contributes mean the same thing on all of them and the receiver
    can concatenate along M. Seeding B per-rank would still validate on each rank
    against its own reference and produce a meaningless collective.

    Values are kept small so the fp8 rounding is the only error source and the
    reference can be rebuilt on the host from the same recipe. Layout follows
    ``bench_gemm_ar.make_operands`` exactly, including that a blockscale ``sa``
    is column-major -- see its docstring for why that specific combination.
    """
    ga = torch.Generator(device="cuda").manual_seed(1234 + rank)
    gb = torch.Generator(device="cuda").manual_seed(9999)  # same on every rank
    a = (torch.randn(m, k, generator=ga, device="cuda") / 8).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, generator=gb, device="cuda") / 8).to(torch.float8_e4m3fn)
    if quant == "ptpc":
        sa = (
            torch.rand(m, generator=ga, device="cuda", dtype=torch.float32) * 0.01
            + 0.01
        )
        sb = (
            torch.rand(n, generator=gb, device="cuda", dtype=torch.float32) * 0.01
            + 0.01
        )
        return a, b, sa, sb
    if quant == "mxfp8":
        # A per-32-K, B per-32x32, both ue8m0. Scales stay as exponent bytes all
        # the way to the MFMA, which is what its scale operand reads. B is
        # replicated like the fp8 weight itself, so it takes the shared seed.
        kb = k // MXFP8_BK
        return (
            a,
            b,
            _ue8m0_bytes((m, kb), ga).contiguous(),
            _ue8m0_bytes((n // MXFP8_BK, kb), gb).contiguous(),
        )
    if quant != "blockscale":
        raise ValueError(f"quant must be ptpc, blockscale or mxfp8, got {quant!r}")
    kb = k // SCALE_BK
    sa = (
        torch.rand(m, kb, generator=ga, device="cuda", dtype=torch.float32) * 0.01
        + 0.01
    )
    sb = (
        torch.rand(n // SCALE_BK, kb, generator=gb, device="cuda", dtype=torch.float32)
        * 0.01
        + 0.01
    )
    return a, b, sa.t().contiguous().t(), sb.contiguous()


def make_operands_bf16(rank: int, m: int, n: int, k: int):
    """Per-rank A, **replicated** B, no scales.

    Same seeding discipline as ``make_operands`` and the replication is
    load-bearing for the same reason -- see its docstring. There is nothing to
    quantise here: bf16 is the model's own input precision, which is the whole
    point of this path.
    """
    ga = torch.Generator(device="cuda").manual_seed(1234 + rank)
    gb = torch.Generator(device="cuda").manual_seed(9999)  # same on every rank
    a = (torch.randn(m, k, generator=ga, device="cuda") / 8).to(torch.bfloat16)
    b = (torch.randn(n, k, generator=gb, device="cuda") / 8).to(torch.bfloat16)
    return a, b, None, None


def reference_gemm_bf16(a, b) -> torch.Tensor:
    """Exact fp32 reference. No quantisation recipe to mirror."""
    return a.float() @ b.float().T


def reference_gemm(a, b, sa, sb, quant: str) -> torch.Tensor:
    """This rank's fp32 ``[M, N]``, matching ``make_operands``' quantisation."""
    af, bf = a.float(), b.float()
    if quant == "ptpc":
        return (af @ bf.T) * sa[:, None] * sb[None, :]
    if quant == "mxfp8":
        sav, sbv = _ue8m0_value(sa), _ue8m0_value(sb)
        out = torch.zeros(a.shape[0], b.shape[0], device=a.device, dtype=torch.float32)
        for i in range(a.shape[1] // MXFP8_BK):
            ks = slice(i * MXFP8_BK, (i + 1) * MXFP8_BK)
            out += (
                (af[:, ks] @ bf[:, ks].T)
                * sav[:, i][:, None]
                * sbv[:, i].repeat_interleave(MXFP8_BK)[None, :]
            )
        return out
    out = torch.zeros(a.shape[0], b.shape[0], device=a.device, dtype=torch.float32)
    for i in range(a.shape[1] // SCALE_BK):
        ks = slice(i * SCALE_BK, (i + 1) * SCALE_BK)
        out += (
            (af[:, ks] @ bf[:, ks].T)
            * sa[:, i][:, None]
            * sb[:, i].repeat_interleave(SCALE_BK)[None, :]
        )
    return out


def _median_us(fn, warmup: int, iters: int, *, graph: bool = True) -> float:
    if graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return _median_us(g.replay, warmup, iters, graph=False)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def _validate_recv(recv, args, rank, world_size) -> tuple[float, bool]:
    """Check what *arrived*, rank by source rank.

    Reads the received buffer, never the local one. A transport that silently
    moves nothing -- a mori built without ``BUILD_CCO_SDMA``, a peer pointer that
    resolved to our own window -- leaves the local slot correct and the other
    seven stale, so any check that looks at this rank's own contribution passes
    while seven eighths of the answer is wrong.

    Takes no operands, unlike ``bench_gemm_a2a``'s: every source's are rebuilt
    from the seed anyway, and this rank's are just ``src == rank``.
    """
    worst = 0.0
    per_src = []
    for src in range(world_size):
        a_src, b_src, sa_src, sb_src = _operands(src, args)
        ref = _reference(a_src, b_src, sa_src, sb_src, args)
        got = recv[src * args.m : (src + 1) * args.m].float()
        denom = ref.norm().item()
        rel = (got - ref).norm().item() / denom if denom else float("inf")
        worst = max(worst, rel)
        per_src.append(rel)
        del a_src, b_src, sa_src, sb_src, ref
    if worst >= args.tolerance:
        # Per source, not just the worst. "Everything is missing" and "only my
        # own slab is missing" both report relL2 1.0 from the maximum, and they
        # are entirely different bugs -- the second means the transport works
        # and the local copy was forgotten.
        marks = " ".join(
            f"{i}{'*' if i == rank else ''}={r:.2e}" for i, r in enumerate(per_src)
        )
        print(f"[rank {rank}] per-source relL2: {marks}", flush=True)
    return worst, worst < args.tolerance


def _operands(rank, args):
    if args.in_dtype == "bf16":
        return make_operands_bf16(rank, args.m, args.n, args.k)
    return make_operands(rank, args.m, args.n, args.k, args.quant)


def _reference(a, b, sa, sb, args):
    if args.in_dtype == "bf16":
        return reference_gemm_bf16(a, b)
    return reference_gemm(a, b, sa, sb, args.quant)


def run(args) -> int:
    local_rank, rank, world_size, uid = _setup_distributed()
    if args.mode in NOT_YET:
        if rank == 0:
            print(f"--mode {args.mode} is not implemented yet", file=sys.stderr)
        return 2

    bf16_in = args.in_dtype == "bf16"
    out_t = torch.bfloat16 if args.out_dtype == "bf16" else torch.float32
    elem_bytes = 2 if args.out_dtype == "bf16" else 4
    if not bf16_in:
        if args.out_dtype != "bf16":
            raise SystemExit(
                "--out-dtype fp32 needs --in-dtype bf16: the fp8 GEMM's "
                "epilogue only emits bf16"
            )
        if args.block_m == 256 and args.quant != "mxfp8":
            pass
    if bf16_in:
        if args.quant != "ptpc":
            raise SystemExit(
                "--quant is meaningless with --in-dtype bf16 and is rejected "
                "rather than ignored: bf16 is not a quantised format"
            )
        if args.k % BF16_BLOCK_K:
            raise SystemExit(f"--in-dtype bf16 needs k % {BF16_BLOCK_K} == 0")
        if args.block_m == 0 or args.block_n == 0:
            # The tile is the dominant knob at these shapes and the template's
            # 256x256 is the worst of them -- 64 workgroups against 256 CUs at
            # M=N=2048. pick_tile's docstring carries the measurements.
            bm, bn = pick_tile(args.m, args.n, args.out_dtype)
            args.block_m = args.block_m or bm
            args.block_n = args.block_n or bn
    if not bf16_in:
        # The fp8 path's tile is not a free choice: its epilogue is written for
        # BLOCK_N=256, so only block_m moves.
        args.block_m = args.block_m or DEFAULT_BLOCK_M
        args.block_n = args.block_n or 256
    if not bf16_in and args.quant == "mxfp8" and args.block_m == DEFAULT_BLOCK_M:
        # mxfp8's BLOCK_M is a property of the packed A scale, not a tuning
        # choice, so take it rather than making every caller pass it.
        args.block_m = MXFP8_BLOCK_M
    needs_sdma = args.mode in ("split-sdma", "fused-sdma")
    if needs_sdma and os.environ.get("MORI_ENABLE_SDMA") != "1":
        # Without it, cco's SDMA put is a no-op that reports success: the run
        # completes, the timing looks plausible because it is the GEMM plus a
        # barrier, and every peer slot stays at whatever it held. That reads as
        # a transport bug rather than a missing environment variable, which is
        # exactly what it did -- split-sdma at 64.8us against a 57.3us
        # gemm-only, relL2 1.0. Validation catches it, but only if it is on.
        raise SystemExit(
            f"--mode {args.mode} needs MORI_ENABLE_SDMA=1 in the environment; "
            f"without it the copy-engine put silently moves nothing"
        )
    # A chunk is a run of row tiles; the count has to divide them, and the
    # request is rounded down rather than rejected so a sweep stays usable.
    chunks = counter_chunks(args.m // args.block_m, args.chunks) if needs_sdma else 1
    if args.mode == "fused-sdma" and args.post == "lanes" and chunks > 1:
        # The lane-parallel epilogue has no submit lock, so two chunks of one
        # destination are kept apart by queue instead. Derived, not a choice --
        # compile_fused_gemm_ag rejects the combination otherwise.
        args.sdma_queues = max(args.sdma_queues, chunks)
    cfg = ag_config(
        world_size=world_size,
        m=args.m,
        n=args.n,
        elem_bytes=elem_bytes,
        block_m=args.block_m,
        block_n=args.block_n,
        counter_chunks=chunks,
        force_blocks=args.copy_blocks or None,
    )
    a, b, sa, sb = _operands(rank, args)
    # The bf16 GEMM reads a plain row-major [N, K]; the preshuffle is the fp8
    # kernel's operand layout and does not apply.
    b_shuf = b if bf16_in else preshuffle_b(b)

    # The window holds only what peers must reach: recv. And recv holds the
    # input too -- this rank's slot is where its GEMM writes -- which is what
    # removes gemm_a2a's separate staging region.
    vmm = 2 * cfg.window_bytes + (64 << 20)
    with Communicator.init(world_size, rank, uid, per_rank_vmm=vmm) as comm:
        mem = comm.alloc_mem(cfg.window_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        from_gpu_ptr(mem.ptr + cfg.recv_off, (cfg.recv_bytes,), torch.uint8).zero_()
        recv = from_gpu_ptr(
            mem.ptr + cfg.recv_off, (world_size * args.m, args.n), out_t
        )
        if args.mode in WINDOW_C:
            # The GEMM's output slab *is* this rank's contribution to the
            # collective, at exactly the shape and stride recv wants, so it can
            # be written in place. No mode needs a re-layout, which is the
            # single biggest difference from the all-to-all.
            c = from_gpu_ptr(
                mem.ptr + cfg.recv_slot_off(rank), (args.m, args.n), out_t
            )
        else:
            c = torch.empty(args.m, args.n, device="cuda", dtype=out_t)
        torch.cuda.synchronize()

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        if needs_sdma:
            reqs.sdma_queue_count = args.sdma_queues
        dc = comm.create_dev_comm(reqs)

        fused = args.mode in ("fused-lsa", "fused-sdma")
        # The pull baseline is not free to pick its GEMM. A pull reads a peer's
        # HBM, and an ordinary cached C store leaves the line dirty in the
        # producer's own L2 where that read never reaches -- so the GEMM has to
        # publish with sc0|sc1. See build_lsa_ag's docstring; the symptom
        # without it is a relL2 that depends on the quantisation.
        pull_gemm = args.mode == "split-lsa-pull"
        if bf16_in:
            # One factory for both, as gemm_ar does: fuse=False emits the same
            # kernel minus the epilogue tail, so the split baseline and the
            # fused kernel differ in exactly one thing.
            gemm = compile_bf16_gemm_ag(
                cfg,
                rank,
                K=args.k,
                BLOCK_M=args.block_m,
                BLOCK_N=args.block_n,
                out_dtype=args.out_dtype,
                waves_per_eu=args.waves_per_eu,
                xcd_swizzle=args.xcd_swizzle if not fused else 0,
                transport="lsa" if args.mode == "fused-lsa" else "sdma",
                fuse=fused,
                chunks=chunks,
                sdma_queues=args.sdma_queues,
                post=args.post,
                peer_uncached=(
                    pull_gemm or (args.peer_uncached and args.mode == "fused-lsa")
                ),
                direct_fence=args.direct_fence,
                emit_put=not args.no_put,
                fence=args.fence,
            )
        elif fused or pull_gemm or (needs_sdma and args.split_gemm == "epilogue"):
            # One builder for the fused path and, optionally, for its split
            # baseline: with fuse=False it runs the same kernel minus the
            # election and the put, so the pair differs in exactly one thing.
            gemm = compile_fused_gemm_ag(
                cfg,
                rank,
                K=args.k,
                BLOCK_M=args.block_m,
                BLOCK_N=args.block_n,
                quant=args.quant,
                waves_per_eu=args.waves_per_eu,
                transport="lsa" if args.mode == "fused-lsa" else "sdma",
                fuse=fused,
                chunks=chunks,
                sdma_queues=args.sdma_queues,
                post=args.post,
                peer_uncached=(
                    pull_gemm or (args.peer_uncached and args.mode == "fused-lsa")
                ),
                direct_fence=args.direct_fence,
                emit_put=not args.no_put,
                fence=args.fence,
            )
        else:
            # Literally gemm-only's kernel. Where it writes is the only thing
            # that changes between `gemm-only`, `gemm-to-window` and the split
            # transports, and `c` above already decided that.
            gemm = compile_gemm_local(
                cfg,
                rank,
                K=args.k,
                BLOCK_M=args.block_m,
                BLOCK_N=args.block_n,
                quant=args.quant,
                waves_per_eu=args.waves_per_eu,
                xcd_swizzle=args.xcd_swizzle,
                swap_ab=args.swap_ab,
                permlane=args.permlane,
            )
        if bf16_in:
            # int16 views, not int8: make_bf16_buffer_tensor reuses the incoming
            # tensor's layout, and a byte-counted one would describe twice as
            # many bf16 elements as exist.
            a_arg = a.contiguous().view(torch.int16).view(-1)
            b_arg = b_shuf.contiguous().view(torch.int16).view(-1)
        else:
            a_arg = a.contiguous().view(torch.int8).view(-1)
            b_arg = b_shuf.contiguous().view(torch.int8).view(-1)
        c_flat = c.reshape(-1)
        # The kernel indexes both scale buffers linearly, so it needs the
        # *physical* element order. sa is logically [M, K/128] but column-major,
        # so its physical order is sa.t().
        sa_arg = sb_arg = None
        if bf16_in:
            pass
        elif args.quant == "blockscale":
            sa_arg = sa.t().reshape(-1).contiguous()
            sb_arg = sb.reshape(-1).contiguous()
        elif args.quant == "mxfp8":
            # Exponent bytes widened to int32 with the byte in the low 8 bits:
            # the MFMA's scale operand is a 32-bit register read at op_sel 0.
            # A goes through gemm_ar's preshuffle_a_scale; B stays K-block major
            # because a 16-column tile never straddles a 32-column group, so its
            # load is already a broadcast.
            sa_arg = preshuffle_a_scale(sa)
            sb_arg = sb.to(torch.int32).t().reshape(-1).contiguous()
        else:
            sa_arg, sb_arg = sa, sb

        # Same kernel, same bytes over the fabric, same barrier. Only which end
        # drives the transfer differs -- that is the one variable this pair
        # isolates, and it is the question all-gather raises and the other two
        # collectives do not.
        copy = (
            build_lsa_ag(
                cfg,
                rank,
                direction="pull" if args.mode == "split-lsa-pull" else "push",
                uncached=args.lsa_uncached,
                unroll=args.push_unroll,
            )
            if args.mode in ("split-lsa-push", "split-lsa-pull")
            else None
        )
        # fused-lsa has no collective kernel -- the epilogue already wrote into
        # the peers -- but it still needs the agreement that every peer
        # *finished*, or a rank reads a slab a peer is still writing.
        barrier = build_lsa_barrier(cfg, rank) if args.mode == "fused-lsa" else None
        parts = (
            build_sdma_phases(cfg, rank, queues=args.sdma_queues)
            if needs_sdma
            else None
        )
        rccl = args.mode == "split-rccl"

        def call_gemm(stream):
            # The two GEMMs differ by two arguments -- the bf16 one has no
            # scales -- so the launch is adapted here rather than by giving the
            # bf16 kernel two ignored tensors.
            if bf16_in:
                gemm(
                    a_arg, b_arg, c_flat, args.m, args.n, dc.ptr, win.handle,
                    stream=stream,
                )
            else:
                gemm(
                    a_arg, b_arg, c_flat, sa_arg, sb_arg, args.m, args.n,
                    dc.ptr, win.handle, stream=stream,
                )

        def once():
            stream = fx.Stream(torch.cuda.current_stream())
            call_gemm(stream)
            if copy is not None:
                copy(dc.ptr, win.handle, stream=stream)
            if barrier is not None:
                barrier(dc.ptr, win.handle, stream=stream)
            if rccl:
                # Same bytes, same GEMM as gemm-only: only the transport
                # differs. all_gather_into_tensor concatenates the inputs along
                # dim 0 in rank order, which is exactly the recv layout -- and C
                # is a private tensor here precisely so this call is not
                # aliasing part of its own output.
                dist.all_gather_into_tensor(recv, c)
            if parts is not None:
                # fused-sdma: the puts left from the epilogue, so only the drain
                # runs. split-sdma: the full gather.
                phase = "drain" if args.mode == "fused-sdma" else "gather"
                parts[phase](dc.ptr, win.handle, stream=stream)

        comm.barrier()
        once()
        torch.cuda.synchronize()
        comm.barrier()

        rel_l2 = float("nan")
        validated = True
        if args.no_put:
            # Nothing was transferred; the received buffer holds this
            # rank's own slab and stale peer slabs. Timing only.
            rel_l2, validated = float("nan"), True
        elif not args.skip_validation:
            if args.mode in ("gemm-only", "gemm-to-window"):
                # The GEMM on its own. Without this control a GEMM bug reads as
                # a transport bug: every other mode validates what arrived, so a
                # corrupt C and a corrupt copy are indistinguishable.
                #
                # gemm-to-window checks the same thing through the window
                # pointer, so it also proves the recv slot map: if
                # recv_slot_off(rank) were wrong, this mode writes the right
                # bytes to the wrong place and every transport mode then fails
                # in a way that looks like a transport bug.
                ref = _reference(a, b, sa, sb, args)
                denom = ref.norm().item()
                rel_l2 = (c.float() - ref).norm().item() / denom if denom else 0.0
                validated = rel_l2 < args.tolerance
                del ref
            else:
                rel_l2, validated = _validate_recv(recv, args, rank, world_size)
            if not validated:
                print(
                    f"[rank {rank}] {args.mode} VALIDATION FAILED relL2={rel_l2:.3e}",
                    flush=True,
                )

        us = _median_us(once, args.warmup, args.iters, graph=not args.no_graph)
        # Same-run phase split, as gcnasm's strict_timing does. The headline
        # `us` stays the whole thing; these two are only for attribution, and
        # they exist because "mode E2E minus a separate gemm-only run" charges
        # the transfer for any difference between two process groups. gemm_a2a
        # measured 324.4us for a GEMM in its own run against 297.3 for the same
        # kernel timed here -- 27us of clock ramp, enough to take split-sdma
        # from a real 81% of line rate to an apparent 94%.
        phase_us = {}
        if args.phase_split and (copy is not None or parts is not None or rccl):

            def _gemm_only():
                call_gemm(fx.Stream(torch.cuda.current_stream()))

            comm.barrier()
            phase_us["compute"] = _median_us(
                _gemm_only, args.warmup, args.iters, graph=not args.no_graph
            )
            comm.barrier()
            phase_us["comm"] = us - phase_us["compute"]
        comm.barrier()

        gathered = [None] * world_size
        dist.all_gather_object(gathered, {"rank": rank, "us": us, "ok": validated})
        if rank == 0:
            result = {
                "mode": args.mode,
                "m": args.m,
                "n": args.n,
                "k": args.k,
                "quant": args.quant if args.in_dtype == "fp8" else None,
                "in_dtype": args.in_dtype,
                "out_dtype": args.out_dtype,
                "elem_bytes": cfg.elem_bytes,
                "world_size": world_size,
                "chunks": chunks,
                "us": max(g["us"] for g in gathered),
                "rel_l2": rel_l2,
                "validated": all(g["ok"] for g in gathered),
                "remote_bytes_per_rank": cfg.remote_bytes_per_rank,
                **{f"phase_{k}": v for k, v in phase_us.items()},
            }
            print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)

        win.close()
        mem.close()
    dist.barrier()
    return 0 if validated else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES + NOT_YET, default="split-lsa-pull")
    p.add_argument("-m", type=int, default=2048)
    # Not `-n`: torchrun takes --nproc_per_node and abbreviation matching makes
    # a bare `-n` ambiguous against it.
    p.add_argument("--out-dim", "-N", dest="n", type=int, default=2048)
    p.add_argument("-k", type=int, default=7168)
    p.add_argument(
        "--in-dtype",
        choices=("fp8", "bf16"),
        default="fp8",
        help="operand precision. fp8 is gemm_ar's 8-wave GEMM (the default, so "
        "every earlier number stays reproducible); bf16 is the quad-subtile "
        "port, which is what the model's wkv_gate actually uses",
    )
    p.add_argument(
        "--out-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="C and wire precision. fp32 needs --in-dtype bf16 and doubles the "
        "bytes on the wire, which is the model's real traffic "
        "(linear_bf16_fp32)",
    )
    p.add_argument("--quant", choices=("ptpc", "blockscale", "mxfp8"), default="ptpc")
    p.add_argument(
        "--block-m",
        type=int,
        default=0,
        help="0 picks the tile from the shape (see pick_tile); the fp8 path "
        "falls back to its fixed 128",
    )
    p.add_argument("--block-n", type=int, default=0)
    p.add_argument("--waves-per-eu", type=int, default=2)
    p.add_argument("--xcd-swizzle", type=int, default=0)
    p.add_argument("--swap-ab", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--permlane", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--chunks", type=int, default=1)
    p.add_argument(
        "--split-gemm",
        choices=("local", "epilogue"),
        default="local",
        help="which GEMM the split-sdma baseline uses: gemm-only's kernel "
        "writing the window (local), or the fused kernel with fuse=False "
        "(epilogue), which makes split-sdma and fused-sdma differ in exactly "
        "one thing at the cost of no longer being gemm-only's kernel",
    )
    p.add_argument("--direct-fence", choices=("leader", "all"), default="leader")
    p.add_argument("--fence", choices=("none", "leader", "all"), default="leader")
    p.add_argument("--copy-blocks", type=int, default=0)
    p.add_argument(
        "--lsa-uncached", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--peer-uncached",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="sc0|sc1 on fused-lsa's peer stores; the split copy kernel's own "
        "default is the opposite (--lsa-uncached), so the pair is worth "
        "sweeping together",
    )
    p.add_argument(
        "--push-unroll",
        type=int,
        default=4,
        help="packs a split-lsa-push thread keeps in flight; ignored on pull, "
        "which has world-1 independent loads already",
    )
    p.add_argument(
        "--post",
        choices=("lanes", "serial"),
        default="lanes",
        help="how fused-sdma issues a chunk's world-1 packets: one per lane, "
        "or all from thread 0 back to back (the original, kept for "
        "attribution)",
    )
    p.add_argument("--sdma-queues", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--phase-split", action="store_true")
    p.add_argument(
        "--no-put",
        action="store_true",
        help="run the fused epilogue without posting; output is wrong "
        "by construction and validation is skipped",
    )
    p.add_argument("--skip-validation", action="store_true")
    #: fp8's own floor at these shapes is ~2e-3; 3e-3 leaves headroom for the
    #: accumulation order differing from the host reference without admitting a
    #: real error, which starts at 1e-2 and rises fast.
    p.add_argument("--tolerance", type=float, default=3e-3)
    return p


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
