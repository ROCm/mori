#!/usr/bin/env python3
"""
Intra-node allreduce (bf16 sum) using Triton + mori.ir.triton.

Each PE reads from all PEs via P2P pointers (Kernel A) or put+signal (Kernel B)
and accumulates locally — every PE gets the same result.

Usage:
    torchrun --nproc_per_node=2 test_triton_allreduce.py
    torchrun --nproc_per_node=8 test_triton_allreduce.py
"""

import os
import sys
import time

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from mori.ir import triton as mori_shmem_device
from mori.ir.triton import get_extern_libs, install_hook
from mori.ir import SIGNAL_ADD as _SIGNAL_ADD

SIGNAL_ADD = tl.constexpr(_SIGNAL_ADD)


# ===================================================================
# 1. Kernel A: device-side shmem_ptr_p2p allreduce
# ===================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=32, num_stages=1),
    ],
    key=["N", "npes"],
)
@triton.jit
def allreduce_p2p_kernel(
    data_ptr,
    result_ptr,
    npes,
    N,
    BLOCK_SIZE: tl.constexpr,
    MAX_PES: tl.constexpr,
):
    """Allreduce via device-side shmem_ptr_p2p: each block resolves remote
    pointers on the fly using mori bitcode, then P2P loads + accumulates."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    mype = mori_shmem_device.my_pe()
    data_ptr_int = data_ptr.to(tl.uint64, bitcast=True)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in tl.static_range(MAX_PES):
        if i < npes:
            if i == mype:
                data = tl.load(data_ptr + offs, mask=mask, other=0.0)
            else:
                remote_int = mori_shmem_device.ptr_p2p(data_ptr_int, mype, i)
                remote_ptr = remote_int.to(tl.pointer_type(tl.bfloat16), bitcast=True)
                data = tl.load(remote_ptr + offs, mask=mask, other=0.0)
            acc += data.to(tl.float32)

    tl.store(result_ptr + offs, acc.to(tl.bfloat16), mask=mask)


# ===================================================================
# 2. Kernel B: multi-block all-to-all put+signal allreduce
# ===================================================================
@triton.jit
def allreduce_put_signal_kernel(
    input_ptr,
    recv_ptr,
    output_ptr,
    signal_ptr,
    mype: int, npes: int, N: int,
    CHUNK_SIZE: tl.constexpr, MAX_PES: tl.constexpr, CHUNKS_PER_PE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bid = tl.program_id(0)
    target_pe = bid // CHUNKS_PER_PE
    chunk_id = bid % CHUNKS_PER_PE

    chunk_offset = chunk_id * CHUNK_SIZE
    chunk_bytes = CHUNK_SIZE * 2

    # Phase 1: each block puts one chunk to one PE (or self-copies)
    if target_pe < npes:
        if target_pe == mype:
            for off in range(0, CHUNK_SIZE, BLOCK_SIZE):
                o = chunk_offset + off + tl.arange(0, BLOCK_SIZE)
                m = o < N
                tl.store(recv_ptr + mype * N + o,
                         tl.load(input_ptr + o, mask=m), mask=m)
            sig_addr = signal_ptr + mype
            tl.atomic_add(sig_addr, 1, sem='release')
        else:
            mori_shmem_device.putmem_nbi_signal_block(
                recv_ptr + mype * N + chunk_offset,
                input_ptr + chunk_offset,
                tl.cast(chunk_bytes, tl.uint64),
                signal_ptr + mype,
                tl.full([], 1, tl.uint64),
                SIGNAL_ADD,
                target_pe,
                0,
            )
            mori_shmem_device.quiet_thread()

    # Phase 2: wait for all signals (all are on local HBM)
    for i in tl.static_range(MAX_PES):
        if i < npes:
            mori_shmem_device.uint64_wait_until_equals(
                signal_ptr + i, tl.cast(CHUNKS_PER_PE, tl.uint64))

    # Phase 3: multi-block accumulate
    total_blocks = MAX_PES * CHUNKS_PER_PE
    for base in range(bid * BLOCK_SIZE, N, total_blocks * BLOCK_SIZE):
        offs = base + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for pe in tl.static_range(MAX_PES):
            if pe < npes:
                acc += tl.load(recv_ptr + pe * N + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(output_ptr + offs, acc.to(tl.bfloat16), mask=mask)


# ===================================================================
# 3. Distributed setup
# ===================================================================
def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo")
    world_group = dist.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)
    import mori.shmem as ms
    ms.shmem_torch_process_group_init("default")
    mype, npes = ms.shmem_mype(), ms.shmem_npes()
    print(f"[PE {mype}/{npes}] initialized")
    return mype, npes


def cleanup():
    import mori.shmem as ms
    ms.shmem_finalize()
    if dist.is_initialized():
        dist.destroy_process_group()


# ===================================================================
# 4. Benchmark helper
# ===================================================================
def bench(label, fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


# ===================================================================
# 5. Test
# ===================================================================
def test_allreduce(mype, npes, extern_libs):
    import mori.shmem as ms
    from mori.shmem import mori_shmem_create_tensor

    M, K = 64, 7168
    N = M * K
    nbytes = N * 2
    MAX_PES = 8
    BS = 8192

    print(f"\n[PE {mype}] === Triton allreduce (bf16, {M}x{K}) ===")

    torch.manual_seed(42 + mype)
    local_data = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    symm_buf = mori_shmem_create_tensor((N,), torch.bfloat16)
    symm_buf.copy_(local_data.view(-1))
    result_a = torch.empty(N, dtype=torch.bfloat16, device="cuda")

    torch.cuda.synchronize()

    local_cpu = local_data.cpu()
    all_data_cpu = [torch.empty_like(local_cpu) for _ in range(npes)]
    dist.all_gather(all_data_cpu, local_cpu)
    expected = torch.stack(all_data_cpu).to(torch.float32).sum(dim=0).to(torch.bfloat16).view(-1).cuda()

    ms.shmem_barrier_all()

    # -- Kernel A --
    print(f"[PE {mype}] --- Kernel A: device-side shmem_ptr_p2p ---")
    grid_a = (triton.cdiv(N, 1024),)
    allreduce_p2p_kernel[grid_a](
        symm_buf, result_a, npes, N, MAX_PES=MAX_PES,
        extern_libs=extern_libs)
    torch.cuda.synchronize()

    best_a = allreduce_p2p_kernel.best_config
    grid_a = (triton.cdiv(N, best_a.kwargs["BLOCK_SIZE"]),)
    allreduce_p2p_kernel[grid_a](
        symm_buf, result_a, npes, N, MAX_PES=MAX_PES,
        extern_libs=extern_libs)
    torch.cuda.synchronize()

    err_a = (result_a.float() - expected.float()).abs().max().item()
    print(f"[PE {mype}] A max_err={err_a:.6f}")
    torch.testing.assert_close(result_a.view(M, K), expected.view(M, K), atol=1e-1, rtol=1e-1)
    print(f"[PE {mype}] A PASS (BLOCK_SIZE={best_a.kwargs['BLOCK_SIZE']}, warps={best_a.num_warps})")

    us_a = bench("A", lambda: allreduce_p2p_kernel[grid_a](
        symm_buf, result_a, npes, N, MAX_PES=MAX_PES, extern_libs=extern_libs))
    bw_a = N * 2 * npes / (us_a * 1e-6) / 1e9
    print(f"[PE {mype}] A: {us_a:.1f} us, {bw_a:.1f} GB/s")

    # -- Kernel B --
    is_rdma = False
    for pe in range(npes):
        if pe != mype:
            if ms.shmem_ptr_p2p(symm_buf.data_ptr(), mype, pe) == 0:
                is_rdma = True
                break

    CPP = 1 if is_rdma else 8
    NW = 16
    transport_name = "RDMA/IBGDA" if is_rdma else "P2P"
    print(f"\n[PE {mype}] --- Kernel B: put+signal ({transport_name}, chunks_per_pe={CPP}) ---")
    ms.shmem_barrier_all()

    recv_b = mori_shmem_create_tensor((npes * N,), torch.bfloat16)
    output_b = torch.empty(N, dtype=torch.bfloat16, device="cuda")
    signal_b = mori_shmem_create_tensor((npes,), torch.int64)

    CHUNK_SZ = triton.cdiv(N, CPP)
    CHUNK_SZ = triton.cdiv(CHUNK_SZ, BS) * BS
    TOTAL_BLOCKS = MAX_PES * CPP

    recv_b.zero_(); signal_b.zero_()
    torch.cuda.synchronize(); ms.shmem_barrier_all()

    allreduce_put_signal_kernel[(TOTAL_BLOCKS,)](
        symm_buf, recv_b, output_b, signal_b,
        mype, npes, N,
        CHUNK_SIZE=CHUNK_SZ, MAX_PES=MAX_PES,
        CHUNKS_PER_PE=CPP, BLOCK_SIZE=BS,
        extern_libs=extern_libs, num_warps=NW)
    torch.cuda.synchronize()

    err_b = (output_b.float() - expected.float()).abs().max().item()
    print(f"[PE {mype}] B max_err={err_b:.6f} (grid={TOTAL_BLOCKS}, {transport_name})")
    torch.testing.assert_close(output_b.view(M, K), expected.view(M, K), atol=1e-1, rtol=1e-1)
    print(f"[PE {mype}] B PASS")

    ms.shmem_barrier_all()
    signal_b.zero_(); torch.cuda.synchronize(); ms.shmem_barrier_all()

    s_b = torch.cuda.Event(enable_timing=True)
    e_b = torch.cuda.Event(enable_timing=True)
    s_b.record()
    allreduce_put_signal_kernel[(TOTAL_BLOCKS,)](
        symm_buf, recv_b, output_b, signal_b,
        mype, npes, N,
        CHUNK_SIZE=CHUNK_SZ, MAX_PES=MAX_PES,
        CHUNKS_PER_PE=CPP, BLOCK_SIZE=BS,
        extern_libs=extern_libs, num_warps=NW)
    e_b.record()
    torch.cuda.synchronize()
    us_b = s_b.elapsed_time(e_b) * 1000
    print(f"[PE {mype}] B: {us_b:.1f} us")

    if mype == 0:
        print(f"\n[PE 0] Summary ({npes} PEs, {M}x{K} bf16 = {nbytes//1024} KB):")
        print(f"  Kernel A (shmem_ptr_p2p):      {us_a:.1f} us, {bw_a:.1f} GB/s")
        print(f"  Kernel B (put+signal, 1 kern): {us_b:.1f} us")


# ===================================================================
# main
# ===================================================================
def main():
    install_hook()
    extern_libs = get_extern_libs()

    mype, npes = setup_distributed()
    try:
        test_allreduce(mype, npes, extern_libs)
        if mype == 0:
            print(f"\n{'=' * 60}")
            print(f"  All allreduce tests PASSED on {npes} PEs")
            print(f"  (Triton + mori shmem device API)")
            print(f"{'=' * 60}")
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
