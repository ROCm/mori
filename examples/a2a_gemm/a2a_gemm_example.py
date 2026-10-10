"""Standalone: mori-SDMA All-to-All overlapped with a bf16 GEMM (AMD gfx950).

Self-contained example of overlapping an All-to-All collective with the GEMM that
consumes it, using device-initiated SDMA (mori putmem_nbi_signal) so the transfer
runs on the CU-free DMA copy engines while the GEMM runs on the CUs.

Model: a sequence-parallel "A2A -> projection" step. Each of `world` ranks holds
[world*sw, Hp] (its tokens' local feature shard). An All-to-All redistributes it to
chunk-major a_full[world, sw, Hp] (chunk i = source rank i's contribution), then a
split-K projection GEMM computes out[sw, N] = sum_i a_full[i] @ W[i].T (bf16).

Variants compared (all correctness-checked vs the nccl reference):
  unfused_nccl : nccl all_to_all_single, then split-K GEMM (production baseline)
  unfused_mori : mori-SDMA push+signal, wait ALL chunks, then split-K GEMM  (no overlap)
  fused_ingrid : mori-SDMA chunked push + per-source gated_hgemm with the SDMA wait fused
                 INSIDE the GEMM (each M-tile waits its row-block; one kernel, per-tile gate)
  unfused_mori_single : mori-SDMA push, wait all, transpose chunk->seq-major, then ONE big
                 [sw,H]@[H,H] GEMM instead of split-K (wins on GEMM structure, not overlap)
  unfused_nccl_single : nccl all_to_all, transpose, then ONE big GEMM (isolates the single-GEMM
                 lever on the nccl path; the 2x2 comm-path x GEMM-structure with the above)

unfused_mori vs unfused_mori_single isolates split-K vs single-GEMM (same comm, no overlap).
fused_ingrid needs the bundled flydsl_gated_hgemm.py. Requires the flydsl/mori/aiter runtime
and MORI_ENABLE_SDMA=1.

Run (4 GPUs):
  HIP_VISIBLE_DEVICES=0,1,2,3 LD_LIBRARY_PATH=/opt/rocm/lib \
    MORI_ENABLE_SDMA=1 MORI_SHMEM_HEAP_SIZE=16G MORI_SOCKET_IFNAME=lo \
    torchrun --nproc_per_node=4 a2a_gemm_example.py --hidden 8192 --seqs 8192 16384
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori
import mori.shmem
import torch
import torch.distributed as dist
from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm
from aiter.ops.flydsl.kernels.tensor_shim import GTensor
from flydsl.expr import arith, range_constexpr
from flydsl.expr.typing import T
from mori.ir import flydsl as mori_shmem
from mori.ir.flydsl import SIGNAL_ADD

# In-kernel per-tile gate: gated_hgemm is a fork of the tuned FlyDSL HGEMM with the
# SDMA wait fused INSIDE the kernel (each M-tile spins on its row-block flag before the
# A-load), vs the two-kernel chunk_gate + stock GEMM below. Bundled in this folder
# (flydsl_gated_hgemm.py) so the fused_ingrid path is fully self-contained.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flydsl_gated_hgemm import gated_hgemm as _GATED  # noqa: E402

DTYPE = torch.bfloat16
# whitelisted FlyDSL bf16 tiling for all-dims >= 4096 on gfx950
TILE = dict(
    tile_m=256,
    tile_n=256,
    tile_k=64,
    stages=2,
    split_k=1,
    block_m_warps=2,
    block_n_warps=4,
    block_k_warps=1,
    b_to_lds=False,
)


def _p(t):
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


# ---------------------------------------------------------------------------
# Producer: A2A scatter push+signal. grid=(world); block j pushes this rank's
# [sw, Hp] slice for dest peer j into peer j's chunk-major a_full[chunk=my_rank]
# and bumps peer j's flags[my_rank] (+1, monotonic). CU-free SDMA when
# MORI_ENABLE_SDMA=1; the block only *issues* the transfer and returns (nbi).
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _build_a2a_push(world: int, my_rank: int, chunk_bytes: int, flag_bytes: int = 8):
    @flyc.kernel(known_block_size=[256, 1, 1])
    def a2a_push(A_FULL: fx.Pointer, SRC: fx.Pointer, FLAGS: fx.Pointer):
        j = fx.block_idx.x  # dest peer
        AF = GTensor(A_FULL, dtype=T.i32, shape=(-1,))
        SR = GTensor(SRC, dtype=T.i32, shape=(-1,))
        FL = GTensor(FLAGS, dtype=T.i32, shape=(-1,))
        src = SR.get_llvm_ptr(SRC, fx.Index(j) * fx.Index(chunk_bytes))
        dest = AF.get_llvm_ptr(A_FULL, fx.Index(my_rank * chunk_bytes))
        sig = FL.get_llvm_ptr(FLAGS, fx.Index(my_rank * flag_bytes))
        mori_shmem.putmem_nbi_signal_block(
            dest,
            src,
            arith.constant(chunk_bytes, type=T.i64),
            sig,
            arith.constant(1, type=T.i64),
            fx.Int32(SIGNAL_ADD),
            fx.Int32(j),
            fx.Int32(0),
        )

    @flyc.jit
    def launch(
        A_FULL: fx.Pointer,
        SRC: fx.Pointer,
        FLAGS: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),
    ):
        a2a_push(A_FULL, SRC, FLAGS).launch(
            grid=(world, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch


def a2a_push_signal(a_full, src, flags, my_rank, world, stream):
    sw = src.shape[0] // world
    chunk_bytes = sw * src.shape[1] * src.element_size()
    _build_a2a_push(world, my_rank, chunk_bytes)(
        _p(a_full), _p(src), _p(flags), stream=stream
    )


# ---------------------------------------------------------------------------
# Consumer gate: block the stream until flags[slot] >= gen (64-bit monotonic),
# i.e. until source `slot`'s chunk has landed. 1 block; all lanes spin the flag
# (SYSTEM-scope read, since the flag is written by a remote SDMA signal).
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _build_gate(slot: int, flag_bytes: int = 8):
    @flyc.kernel(known_block_size=[64, 1, 1])
    def gate(FLAGS: fx.Pointer, gen: fx.Int64):
        FL = GTensor(FLAGS, dtype=T.i32, shape=(-1,))
        addr = FL.get_llvm_ptr(FLAGS, fx.Index(slot * flag_bytes))
        mori_shmem.uint64_wait_until_greater_than(
            addr, gen - arith.constant(1, type=T.i64)
        )

    @flyc.jit
    def launch(FLAGS: fx.Pointer, gen: fx.Int64, stream: fx.Stream = fx.Stream(None)):
        gate(FLAGS, gen).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def chunk_gate_signal(flags, slot, gen, stream):
    _build_gate(int(slot))(_p(flags), int(gen), stream=stream)


# ---------------------------------------------------------------------------
def mori_init():
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    cpu = dist.new_group(backend="gloo")
    torch._C._distributed_c10d._register_process_group("a2a_gemm", cpu)
    mori.shmem.shmem_torch_process_group_init("a2a_gemm")
    return rank, world, dev


# ---------------------------------------------------------------------------
# Chunked A2A producer for the in-kernel-gated path: push each source's [sw,Hp]
# contribution in BLOCK_M-row chunks, one flag per (source, row-block), so the
# per-source gated GEMM can gate each M-tile on its own row-block (finer than the
# per-source chunk_gate above). Receiver: flags[my_rank*num_rb + m] set when this
# rank's row-block m lands. grid=(world dest peers); in-block row-block loop.
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _build_a2a_chunked_push(
    world, my_rank, sw, row_bytes, block_m, num_rb, flag_bytes=8
):
    rb_bytes = block_m * row_bytes
    chunk_stride = sw * row_bytes

    @flyc.kernel(known_block_size=[256, 1, 1])
    def a2a_chunked(A_FULL: fx.Pointer, SRC: fx.Pointer, FLAGS: fx.Pointer):
        j = fx.block_idx.x  # dest peer
        AF = GTensor(A_FULL, dtype=T.i32, shape=(-1,))
        SR = GTensor(SRC, dtype=T.i32, shape=(-1,))
        FL = GTensor(FLAGS, dtype=T.i32, shape=(-1,))
        for m in range_constexpr(num_rb):
            src = SR.get_llvm_ptr(
                SRC, fx.Index(j) * fx.Index(chunk_stride) + fx.Index(m * rb_bytes)
            )
            dest = AF.get_llvm_ptr(
                A_FULL, fx.Index(my_rank * chunk_stride + m * rb_bytes)
            )
            sig = FL.get_llvm_ptr(FLAGS, fx.Index((my_rank * num_rb + m) * flag_bytes))
            mori_shmem.putmem_nbi_signal_block(
                dest,
                src,
                arith.constant(rb_bytes, type=T.i64),
                sig,
                arith.constant(1, type=T.i64),
                fx.Int32(SIGNAL_ADD),
                fx.Int32(j),
                fx.Int32(0),
            )

    @flyc.jit
    def launch(
        A_FULL: fx.Pointer,
        SRC: fx.Pointer,
        FLAGS: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),
    ):
        a2a_chunked(A_FULL, SRC, FLAGS).launch(
            grid=(world, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch


def a2a_chunked_push_signal(a_full, src, flags, my_rank, world, block_m, stream):
    sw = src.shape[0] // world
    num_rb = sw // block_m
    row_bytes = src.shape[1] * src.element_size()
    _build_a2a_chunked_push(world, my_rank, sw, row_bytes, block_m, num_rb)(
        _p(a_full), _p(src), _p(flags), stream=stream
    )


def bench(fn, dist, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    from mori.shmem import mori_shmem_create_tensor, mori_shmem_free_tensor

    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=8192)  # H
    ap.add_argument("--seqs", type=int, nargs="+", default=[8192, 16384])
    args = ap.parse_args()
    rank, world, dev = mori_init()
    H = args.hidden
    Hp = H // world  # per-rank feature shard
    if rank == 0:
        print(f"\nmori-SDMA A2A + bf16 GEMM  H={H} Hp={Hp} P={world}", flush=True)
        print(
            "variants: unfused_nccl | unfused_mori(SDMA no-overlap) | "
            "fused_ingrid(in-kernel per-tile gate) | "
            "unfused_{mori,nccl}_single(one big GEMM)",
            flush=True,
        )

    BM = TILE["tile_m"]  # in-kernel gate granularity = GEMM M-tile
    for S in args.seqs:
        assert S % world == 0
        sw = S // world
        assert sw % BM == 0
        num_rb = sw // BM
        g = torch.Generator(device="cuda").manual_seed(1 + rank + S)
        src = torch.rand(
            (world * sw, Hp), generator=g, device=dev, dtype=DTYPE
        )  # local shard
        W = [
            (torch.rand((H, Hp), generator=g, device=dev, dtype=DTYPE) - 0.5) * 0.05
            for _ in range(world)
        ]  # per-source W[i]=[N=H,K=Hp]
        Wo_full = torch.cat(
            W, dim=1
        )  # [H, H]: single-GEMM weight (W[i] = its column slice)
        a_full = mori_shmem_create_tensor((world * sw, Hp), DTYPE)
        flags = mori_shmem_create_tensor((world,), torch.int64)
        flags.zero_()
        flags_t = mori_shmem_create_tensor((world * num_rb,), torch.int64)
        flags_t.zero_()
        parts = [torch.empty((sw, H), device=dev, dtype=DTYPE) for _ in range(world)]
        out_seq = torch.empty((sw, H), device=dev, dtype=DTYPE)
        ref = torch.empty((world * sw, Hp), device=dev, dtype=DTYPE)
        mori.shmem.shmem_barrier_all()
        torch.cuda.synchronize()
        gc = [0]
        gt = [0]

        def gemm_all(buf):
            # Split-K O-projection helper: out[sw,H] = sum_i buf[i-chunk] @ W[i].T.
            # `buf` is chunk-major [world*sw, Hp] (chunk i = source i's [sw,Hp]); the K=H
            # reduction is split across the `world` sources, so this runs `world` per-source
            # GEMMs into parts[i] and sums them. Shared by `unfused_nccl` and `unfused_mori`.
            for i in range(world):
                flydsl_hgemm(buf[i * sw : (i + 1) * sw], W[i], out=parts[i], **TILE)
            C = parts[0]
            for i in range(1, world):
                C = C + parts[i]
            return C

        def fused_ingrid():
            # REAL in-kernel-gated fusion: chunked A2A push (one flag per source+row-block)
            # + per-source gated_hgemm(ingrid=True, shard_rows=BM). The SDMA wait is compiled
            # INSIDE the GEMM -- each M-tile spins on its own row-block flag before the A-load,
            # so it is a SINGLE kernel per source with per-tile gating (no separate gate
            # launches). Best design for one big gated GEMM (e.g. AG+QKV); here it is split-K
            # over `world` sources so the per-tile gate is diluted -> ~ties nccl.
            cur = torch.cuda.current_stream()
            dist.barrier()
            gt[0] += 1
            a2a_chunked_push_signal(a_full, src, flags_t, rank, world, BM, cur)
            C = None
            for i in range(world):
                _GATED(
                    a_full[i * sw : (i + 1) * sw],
                    W[i],
                    parts[i],
                    flags_t[i * num_rb :],
                    0,
                    stream=cur,
                    ingrid=True,
                    shard_rows=BM,
                    mono64=True,
                    gen=gt[0],
                    **TILE,
                )
                C = parts[i] if C is None else C + parts[i]
            return C

        def unfused_mori():
            # BASELINE (mori SDMA, no overlap): CU-free SDMA A2A push, then wait for ALL source
            # chunks (per-source gate + barrier), then the split-K GEMM. Same SDMA comm as the
            # fused variants but with comm fully completed before any compute -> isolates the
            # SDMA-path cost (compare vs fused_ingrid for the overlap benefit, vs
            # unfused_mori_single for the split-K-vs-single-GEMM effect).
            cur = torch.cuda.current_stream()
            dist.barrier()
            gc[0] += 1
            a2a_push_signal(a_full, src, flags, rank, world, cur)
            for i in range(world):
                chunk_gate_signal(flags, i, gc[0], cur)
            mori.shmem.shmem_barrier_all()
            return gemm_all(a_full)

        def unfused_nccl():
            # BASELINE (production path): stock nccl all_to_all_single, then the split-K GEMM.
            # No fusion; this is the reference every variant's correctness + speedup compares to.
            dist.all_to_all_single(ref, src)
            return gemm_all(ref)

        def unfused_mori_single():
            # single big GEMM (no overlap): contiguous chunk-major SDMA push, wait all, then
            # transpose chunk->seq-major and do ONE [sw,H]@[H,H] GEMM instead of split-K.
            # mori putmem is contiguous-only, so writing each source into a column slice of
            # a_full[sw,H] needs the transpose (extra HBM pass, no comm/compute overlap).
            # Wins purely on GEMM structure (one big GEMM > world small GEMMs), not overlap.
            cur = torch.cuda.current_stream()
            dist.barrier()
            gc[0] += 1
            a2a_push_signal(a_full, src, flags, rank, world, cur)
            for i in range(world):
                chunk_gate_signal(flags, i, gc[0], cur)
            mori.shmem.shmem_barrier_all()
            a_seq = (
                a_full.view(world, sw, Hp).permute(1, 0, 2).reshape(sw, H).contiguous()
            )
            flydsl_hgemm(a_seq, Wo_full, out=out_seq, **TILE)
            return out_seq

        def unfused_nccl_single():
            # nccl comm + single big GEMM: isolates the single-GEMM lever on the nccl comm path
            # (2x2 with unfused_nccl=nccl+split-K, unfused_mori=mori+split-K,
            # unfused_mori_single=mori+single-GEMM). Same transpose->one-GEMM as
            # unfused_mori_single, but comm via nccl all_to_all instead of mori SDMA.
            dist.all_to_all_single(ref, src)
            a_seq = ref.view(world, sw, Hp).permute(1, 0, 2).reshape(sw, H).contiguous()
            flydsl_hgemm(a_seq, Wo_full, out=out_seq, **TILE)
            return out_seq

        Cu = unfused_nccl()
        torch.cuda.synchronize()
        ms_u = bench(unfused_nccl, dist)
        ms_s = bench(unfused_mori, dist)
        if rank == 0:
            print(
                f"{S:>7} | unfused_nccl {ms_u:>8.4f} ms | unfused_mori(SDMA no-overlap) {ms_s:>8.4f} ms",
                flush=True,
            )
        Cg = fused_ingrid()
        torch.cuda.synchronize()
        okg = torch.allclose(Cg.float(), Cu.float(), atol=1e-2, rtol=1e-2)
        fg = torch.tensor([0 if okg else 1], device=dev)
        dist.all_reduce(fg)
        ms_fi = bench(fused_ingrid, dist)
        if rank == 0:
            print(
                f"{S:>7} |  in-kernel-gated fused_ingrid = {ms_fi:>8.4f} ms | "
                f"ingrid/mori {ms_s / ms_fi:>5.2f}x | ingrid/nccl {ms_u / ms_fi:>5.2f}x | "
                f"correct={fg.item() == 0}",
                flush=True,
            )
        Csq = unfused_mori_single()
        torch.cuda.synchronize()
        oksq = (
            torch.isclose(Csq.float(), Cu.float(), atol=1e-1, rtol=2e-2)
            .float()
            .mean()
            .item()
            > 0.99
        )
        fsq = torch.tensor([0 if oksq else 1], device=dev)
        dist.all_reduce(fsq)
        ms_sq = bench(unfused_mori_single, dist)
        if rank == 0:
            print(
                f"{S:>7} |  unfused_mori_single (1 GEMM) = {ms_sq:>8.4f} ms | "
                f"ss/nccl {ms_u / ms_sq:>5.2f}x | correct={fsq.item() == 0}",
                flush=True,
            )
        Cns = unfused_nccl_single()
        torch.cuda.synchronize()
        okns = (
            torch.isclose(Cns.float(), Cu.float(), atol=1e-1, rtol=2e-2)
            .float()
            .mean()
            .item()
            > 0.99
        )
        fns = torch.tensor([0 if okns else 1], device=dev)
        dist.all_reduce(fns)
        ms_ns = bench(unfused_nccl_single, dist)
        if rank == 0:
            print(
                f"{S:>7} |  unfused_nccl_single (1 GEMM) = {ms_ns:>8.4f} ms | "
                f"ns/nccl {ms_u / ms_ns:>5.2f}x | correct={fns.item() == 0}",
                flush=True,
            )
        mori.shmem.shmem_barrier_all()
        for t in (flags_t, flags, a_full):
            mori_shmem_free_tensor(t)

    mori.shmem.shmem_finalize()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
