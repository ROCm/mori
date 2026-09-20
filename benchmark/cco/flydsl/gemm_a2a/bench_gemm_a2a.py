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
"""fp8 GEMM + all-to-all on cco, by mode.

    gemm-only    the GEMM alone, to size the ceiling
    gemm-staging the staging GEMM alone -- what split-sdma/rccl subtract
    gemm-to-window  gemm-only's kernel, writing the window instead of a tensor
    split-lsa    gemm(); then the standalone LSA all-to-all
    split-sdma   gemm(); then staged SDMA pushes            (not yet)
    fused-lsa    C stored straight into the peers            (not yet)
    fused-sdma   C staged per destination, pushed per chunk  (not yet)

Every rank holds its own ``A [M, K]`` and a **replicated** ``B [N, K]``,
computes the whole ``[M, N]``, and sends column block ``j`` to rank ``j``. That
B is replicated is the thing to keep in mind when reading this against
``bench_gemm_ar.py``: there, every rank holds a different K-shard of both
operands and the collective sums them. Here every rank computes the same
function of different rows, and the collective only moves bytes. So B is seeded
identically on every rank and A is not -- get that wrong and each rank validates
happily against its own wrong reference.

Shape to compare against gcnasm's ``opus_gemm_a2a_lsa``:
``-m 2048 -n 18432 -k 8192`` at 8 ranks, giving ``shard_n = 2304``. Its numbers
are bf16; this is fp8, so only the *fused-vs-split ratio* is comparable between
the two, never the absolute times.

    mpirun/torchrun --nproc_per_node=8 bench_gemm_a2a.py --mode split-lsa \\
      -m 2048 -n 18432 -k 8192 --warmup 10 --iters 30
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

from mori.ops.gemm_a2a import (
    a2a_config,
    build_lsa_a2a,
    build_lsa_barrier,
    build_sdma_phases,
    counter_chunks,
    preshuffle_b,
)
from mori.ops.gemm_a2a.kernels_fused import (
    compile_fused_gemm_a2a,
    compile_gemm_local,
)

#: gemm-staging exists to make the split-sdma subtraction honest. Its GEMM
#: writes [dst][M][shard_n] and is a *different kernel* from gemm-only's
#: [M,N] one, so "split-sdma minus gemm-only" silently charges the transfer
#: for however much the two GEMMs differ. split-lsa has no such problem --
#: its GEMM is gemm-only's, literally.
MODES = (
    "gemm-only",
    "gemm-staging",
    "split-lsa",
    "fused-lsa",
    "split-sdma",
    "fused-sdma",
    "split-rccl",
    "gemm-to-window",
)
NOT_YET = ()

#: fp8 block-scale group size along K. Fixed by the model's quantiser and by the
#: kernel, whose BLOCK_K is already 128.
SCALE_BK = 128


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

    The replication is load-bearing. Every rank computes the same ``B``, so a
    column block means the same thing on all of them and the receiver can
    concatenate contributions along M. Seeding B per-rank would still validate
    on each rank against its own reference and produce a meaningless collective.

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
    if quant != "blockscale":
        raise ValueError(f"quant must be ptpc or blockscale, got {quant!r}")
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


def reference_gemm(a, b, sa, sb, quant: str) -> torch.Tensor:
    """This rank's fp32 ``[M, N]``, matching ``make_operands``' quantisation."""
    af, bf = a.float(), b.float()
    if quant == "ptpc":
        return (af @ bf.T) * sa[:, None] * sb[None, :]
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


def _validate_recv(recv, a, b, sa, sb, args, rank, world_size) -> tuple[float, bool]:
    """Check what *arrived*, rank by source rank.

    Reads the received buffer, never the local one. A transport that silently
    moves nothing -- a mori built without ``BUILD_CCO_SDMA``, a peer pointer that
    resolved to our own window -- leaves the local half correct and the other
    seven slots stale, so any check that looks at this rank's own contribution
    passes while seven eighths of the answer is wrong.
    """
    shard_n = args.n // world_size
    my_cols = slice(rank * shard_n, (rank + 1) * shard_n)
    worst = 0.0
    per_src = []
    for src in range(world_size):
        a_src, b_src, sa_src, sb_src = make_operands(
            src, args.m, args.n, args.k, args.quant
        )
        ref = reference_gemm(a_src, b_src, sa_src, sb_src, args.quant)[:, my_cols]
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


def run(args) -> int:
    local_rank, rank, world_size, uid = _setup_distributed()
    if args.mode in NOT_YET:
        if rank == 0:
            print(f"--mode {args.mode} is not implemented yet", file=sys.stderr)
        return 2

    # All three want the [dst][M][shard_n] slab; only two of them push it
    # with the copy engine.
    needs_staging = args.mode in (
        "split-sdma",
        "fused-sdma",
        "gemm-staging",
        "split-rccl",
    )
    needs_sdma = args.mode in ("split-sdma", "fused-sdma")
    # A chunk is a run of row tiles; the count has to divide them, and the
    # request is rounded down rather than rejected so a sweep stays usable.
    chunks = counter_chunks(args.m // args.block_m, args.chunks) if needs_sdma else 1
    cfg = a2a_config(
        world_size=world_size,
        m=args.m,
        n=args.n,
        block_m=args.block_m,
        block_n=args.block_n,
        staged=needs_staging,
        counter_chunks=chunks,
        force_blocks=args.copy_blocks or None,
    )
    a, b, sa, sb = make_operands(rank, args.m, args.n, args.k, args.quant)
    b_shuf = preshuffle_b(b)

    # The window holds only what peers must reach: recv. C is local.
    vmm = 2 * cfg.window_bytes + (64 << 20)
    with Communicator.init(world_size, rank, uid, per_rank_vmm=vmm) as comm:
        mem = comm.alloc_mem(cfg.window_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        from_gpu_ptr(mem.ptr, (cfg.window_bytes,), torch.uint8).zero_()
        recv = from_gpu_ptr(
            mem.ptr + cfg.recv_off,
            (world_size * args.m, cfg.shard_n),
            torch.bfloat16,
        )
        if args.mode == "gemm-to-window":
            # The control that separates "the staging GEMM's address map is
            # slower" from "writing cco's window is slower". recv is exactly
            # M*N*2 bytes, so gemm-only's kernel can target it unchanged: same
            # instructions, same address map, same descriptor size -- only the
            # memory differs. Its contents are then meaningless, so this mode
            # does not validate.
            c = from_gpu_ptr(mem.ptr + cfg.recv_off, (args.m, args.n), torch.bfloat16)
        else:
            c = torch.empty(args.m, args.n, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        if needs_sdma:
            reqs.sdma_queue_count = args.sdma_queues
        dc = comm.create_dev_comm(reqs)

        if needs_staging:
            # The same kernel for every staging mode, differing only in whether the
            # epilogue posts the puts. That is the point of a split baseline:
            # one changed thing, not two.
            gemm = compile_fused_gemm_a2a(
                cfg,
                rank,
                K=args.k,
                BLOCK_M=args.block_m,
                BLOCK_N=args.block_n,
                quant=args.quant,
                waves_per_eu=args.waves_per_eu,
                xcd_swizzle=args.xcd_swizzle,
                rotated=args.rotated,
                n_stripe=args.n_stripe,
                transport="sdma",
                fuse=args.mode == "fused-sdma",
                chunks=chunks,
                sdma_queues=args.sdma_queues,
                # RCCL takes one input covering every destination,
                # self included, so the self-to-recv shortcut has to
                # be off or it reads a hole.
                self_to_recv=args.mode != "split-rccl",
                staging_store=args.staging_store,
            )
        elif args.mode == "fused-lsa":
            # The epilogue writes into the peers, so there is no separate
            # collective -- only the barrier below.
            gemm = compile_fused_gemm_a2a(
                cfg,
                rank,
                K=args.k,
                BLOCK_M=args.block_m,
                BLOCK_N=args.block_n,
                quant=args.quant,
                waves_per_eu=args.waves_per_eu,
                xcd_swizzle=args.xcd_swizzle,
                rotated=args.rotated,
                n_stripe=args.n_stripe,
            )
        else:
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
        if needs_staging and args.staging_store == "copy":
            # The copy path's descriptor is the staging region, so that is what
            # the kernel's C argument has to be.
            # staging and recv, as one contiguous descriptor: the store
            # selects between them by row offset rather than by base address.
            c_for_gemm = from_gpu_ptr(
                mem.ptr + cfg.staging_off,
                (2 * world_size * args.m, cfg.shard_n),
                torch.bfloat16,
            )
        else:
            c_for_gemm = c
        a_i8 = a.contiguous().view(torch.int8).view(-1)
        b_i8 = b_shuf.contiguous().view(torch.int8).view(-1)
        c_flat = c_for_gemm.reshape(-1)
        # The kernel indexes both scale buffers linearly, so it needs the
        # *physical* element order. sa is logically [M, K/128] but column-major,
        # so its physical order is sa.t().
        if args.quant == "blockscale":
            sa_arg = sa.t().reshape(-1).contiguous()
            sb_arg = sb.reshape(-1).contiguous()
        else:
            sa_arg, sb_arg = sa, sb

        copy = (
            build_lsa_a2a(cfg, rank, uncached=args.lsa_uncached)
            if args.mode == "split-lsa"
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
        staging = (
            from_gpu_ptr(
                mem.ptr + cfg.staging_off,
                (world_size * args.m, cfg.shard_n),
                torch.bfloat16,
            )
            if needs_staging
            else None
        )
        rccl = args.mode == "split-rccl"
        c_ptr = c.data_ptr()

        def once():
            stream = fx.Stream(torch.cuda.current_stream())
            gemm(
                a_i8,
                b_i8,
                c_flat,
                sa_arg,
                sb_arg,
                args.m,
                args.n,
                dc.ptr,
                win.handle,
                stream=stream,
            )
            if copy is not None:
                copy(c_ptr, dc.ptr, win.handle, stream=stream)
            if barrier is not None:
                barrier(dc.ptr, win.handle, stream=stream)
            if rccl:
                # Same bytes, same layout, same GEMM as split-sdma: only the
                # transport differs. all_to_all_single splits the input along
                # dim 0 into world chunks and gives back the concatenation of
                # what each rank sent, which is exactly staging -> recv.
                dist.all_to_all_single(recv, staging)
            if parts is not None:
                # fused-sdma: the puts left from the epilogue, so only the drain
                # runs. split-sdma: the full scatter.
                phase = "drain" if args.mode == "fused-sdma" else "scatter"
                parts[phase](dc.ptr, win.handle, stream=stream)

        comm.barrier()
        once()
        torch.cuda.synchronize()
        comm.barrier()

        rel_l2 = float("nan")
        validated = True
        if not args.skip_validation:
            if args.mode == "gemm-to-window":
                # Writing [M,N] over the recv region does not produce the
                # all-to-all's answer, and is not meant to. Only the time
                # is of interest.
                rel_l2, validated = float("nan"), True
            elif args.mode == "gemm-staging":
                # Nothing is transferred, so recv is meaningless except for this
                # rank's own block, which the GEMM routes there directly. Check
                # that *and* the remote destinations' staging slabs: a timing of
                # a kernel that is not doing the work is worth nothing, and the
                # whole point of this mode is to be subtracted from another.
                ref = reference_gemm(a, b, sa, sb, args.quant)
                worst = 0.0
                for d in range(world_size):
                    cols = slice(d * cfg.shard_n, (d + 1) * cfg.shard_n)
                    want = ref[:, cols]
                    if d == rank:
                        got = recv[rank * args.m : (rank + 1) * args.m]
                    else:
                        got = staging[d * args.m : (d + 1) * args.m]
                    den = want.norm().item()
                    worst = max(
                        worst, (got.float() - want).norm().item() / den if den else 0.0
                    )
                rel_l2, validated = worst, worst < args.tolerance
                del ref
            elif args.mode == "gemm-only":
                # The GEMM on its own. Without this control a GEMM bug reads as
                # a transport bug: every other mode validates what arrived, so a
                # corrupt C and a corrupt copy are indistinguishable.
                ref = reference_gemm(a, b, sa, sb, args.quant)
                denom = ref.norm().item()
                rel_l2 = (c.float() - ref).norm().item() / denom if denom else 0.0
                validated = rel_l2 < args.tolerance
                del ref
            else:
                rel_l2, validated = _validate_recv(
                    recv, a, b, sa, sb, args, rank, world_size
                )
            if not validated:
                print(
                    f"[rank {rank}] {args.mode} VALIDATION FAILED relL2={rel_l2:.3e}",
                    flush=True,
                )

        us = _median_us(once, args.warmup, args.iters, graph=not args.no_graph)
        comm.barrier()

        gathered = [None] * world_size
        dist.all_gather_object(gathered, {"rank": rank, "us": us, "ok": validated})
        if rank == 0:
            result = {
                "mode": args.mode,
                "m": args.m,
                "n": args.n,
                "k": args.k,
                "quant": args.quant,
                "world_size": world_size,
                "shard_n": cfg.shard_n,
                "us": max(g["us"] for g in gathered),
                "rel_l2": rel_l2,
                "validated": all(g["ok"] for g in gathered),
                "remote_bytes_per_rank": cfg.remote_bytes_per_rank,
            }
            print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)

        win.close()
        mem.close()
    dist.barrier()
    return 0 if validated else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES + NOT_YET, default="split-lsa")
    p.add_argument("-m", type=int, default=2048)
    p.add_argument("-n", type=int, default=18432)
    p.add_argument("-k", type=int, default=8192)
    p.add_argument("--quant", choices=("ptpc", "blockscale"), default="ptpc")
    p.add_argument("--block-m", type=int, default=128)
    p.add_argument("--block-n", type=int, default=256)
    p.add_argument("--waves-per-eu", type=int, default=2)
    p.add_argument("--xcd-swizzle", type=int, default=0)
    p.add_argument("--swap-ab", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--permlane", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rotated", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-stripe", type=int, default=1)
    p.add_argument("--chunks", type=int, default=1)
    p.add_argument("--copy-blocks", type=int, default=0)
    p.add_argument(
        "--lsa-uncached", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--staging-store", choices=("buffer", "copy"), default="buffer")
    p.add_argument("--sdma-queues", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--skip-validation", action="store_true")
    #: fp8's own floor at these shapes is ~2e-3; 3e-3 leaves headroom for the
    #: accumulation order differing from the host reference without admitting a
    #: real error, which starts at 1e-2 and rises fast.
    p.add_argument("--tolerance", type=float, default=3e-3)
    return p


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
