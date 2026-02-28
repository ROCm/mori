#!/usr/bin/env python3
"""
Intra-node allreduce (bf16 sum) using pure Triton + mori shmem P2P.

Each PE reads from all PEs via P2P pointers
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
from triton.language import core
from triton.language.core import builtin, tensor
from triton import knobs
from typing import List
import builtins


# ===================================================================
# 1. extern_call helpers (from triton.language.core)
# ===================================================================
def _dispatch(func, lib_name, lib_path, args, arg_type_symbol_dict,
              is_pure, _semantic):
    arg_types, arg_list = [], []
    for arg in args:
        if isinstance(arg, tensor):
            arg_types.append(arg.dtype)
            arg_list.append(arg.handle)
        else:
            arg_types.append(type(arg))
            arg_list.append(arg)
    arg_types = tuple(arg_types)
    symbol = arg_type_symbol_dict[arg_types][0]
    ret_types = arg_type_symbol_dict[arg_types][1]
    if not isinstance(ret_types, (List, tuple)):
        ret_types = [ret_types]
    call = func(lib_name, lib_path, symbol, arg_list,
                [rt.to_ir(_semantic.builder) for rt in ret_types], is_pure)
    if len(ret_types) == 0:
        return tensor(call, core.void)
    if len(ret_types) == 1:
        return tensor(call.get_result(0), ret_types[0])
    return tuple(tensor(call.get_result(i), ty) for i, ty in enumerate(ret_types))


@builtin
def extern_call(lib_name, lib_path, args, arg_type_symbol_dict,
                is_pure, _semantic=None):
    dispatch_args = args.copy()
    for i in builtins.range(len(dispatch_args)):
        dispatch_args[i] = _semantic.to_tensor(dispatch_args[i])
    func = _semantic.builder.create_extern_call
    return _dispatch(func, lib_name, lib_path, dispatch_args,
                     arg_type_symbol_dict, is_pure, _semantic)


# ===================================================================
# 2. Mori shmem device function declarations
# ===================================================================
SIGNAL_SET = tl.constexpr(9)

@core.extern
def shmem_my_pe(_semantic=None):
    return extern_call("libmori_shmem_device", "", [],
                       {(): ("mori_shmem_my_pe", (tl.int32))},
                       is_pure=False, _semantic=_semantic)

@core.extern
def shmem_n_pes(_semantic=None):
    return extern_call("libmori_shmem_device", "", [],
                       {(): ("mori_shmem_n_pes", (tl.int32))},
                       is_pure=True, _semantic=_semantic)

@core.extern
def shmem_ptr_p2p(dest_ptr, my_pe, dest_pe, _semantic=None):
    return extern_call(
        "libmori_shmem_device", "",
        [tl.cast(dest_ptr, tl.uint64, _semantic=_semantic),
         tl.cast(my_pe, tl.int32, _semantic=_semantic),
         tl.cast(dest_pe, tl.int32, _semantic=_semantic)],
        {(tl.uint64, tl.int32, tl.int32): ("mori_shmem_ptr_p2p", (tl.uint64,))},
        is_pure=False, _semantic=_semantic)

@core.extern
def shmem_putmem_nbi_signal_block(dest, source, nbytes, sig_addr, sig_val,
                                  sig_op, pe, qp_id, _semantic=None):
    return extern_call(
        "libmori_shmem_device", "",
        [tl.cast(dest, tl.pointer_type(tl.void), _semantic=_semantic),
         tl.cast(source, tl.pointer_type(tl.void), _semantic=_semantic),
         tl.cast(nbytes, tl.uint64, _semantic=_semantic),
         tl.cast(sig_addr, tl.pointer_type(tl.void), _semantic=_semantic),
         tl.cast(sig_val, tl.uint64, _semantic=_semantic),
         tl.cast(sig_op, tl.int32, _semantic=_semantic),
         tl.cast(pe, tl.int32, _semantic=_semantic),
         tl.cast(qp_id, tl.int32, _semantic=_semantic)],
        {(tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64,
          tl.pointer_type(tl.void), tl.uint64, tl.int32, tl.int32, tl.int32):
         ("mori_shmem_putmem_nbi_signal_block", ())},
        is_pure=False, _semantic=_semantic)

@core.extern
def shmem_uint64_wait_until_equals(addr, val, _semantic=None):
    return extern_call(
        "libmori_shmem_device", "",
        [tl.cast(addr, tl.pointer_type(tl.uint64), _semantic=_semantic),
         tl.cast(val, tl.uint64, _semantic=_semantic)],
        {(tl.pointer_type(tl.uint64), tl.uint64):
         ("mori_shmem_uint64_wait_until_equals", ())},
        is_pure=False, _semantic=_semantic)

@core.extern
def shmem_quiet(_semantic=None):
    return extern_call("libmori_shmem_device", "", [],
                       {(): ("mori_shmem_quiet_thread", ())},
                       is_pure=False, _semantic=_semantic)

@core.extern
def shmem_barrier_all_block(_semantic=None):
    return extern_call("libmori_shmem_device", "", [],
                       {(): ("mori_shmem_barrier_all_block", ())},
                       is_pure=False, _semantic=_semantic)


# ===================================================================
# 3a. Kernel A: device-side shmem_ptr_p2p allreduce
#     Each block calls shmem_ptr_p2p() in the kernel to get remote ptrs.
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
    data_ptr,       # symmetric bf16 buffer (same vaddr on all PEs)
    result_ptr,     # output bf16
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

    mype = shmem_my_pe()
    data_ptr_int = data_ptr.to(tl.uint64, bitcast=True)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in tl.static_range(MAX_PES):
        if i < npes:
            if i == mype:
                data = tl.load(data_ptr + offs, mask=mask, other=0.0)
            else:
                remote_int = shmem_ptr_p2p(data_ptr_int, mype, i)
                remote_ptr = remote_int.to(tl.pointer_type(tl.bfloat16), bitcast=True)
                data = tl.load(remote_ptr + offs, mask=mask, other=0.0)
            acc += data.to(tl.float32)

    tl.store(result_ptr + offs, acc.to(tl.bfloat16), mask=mask)


SIGNAL_ADD = tl.constexpr(10)


# ===================================================================
# 3b. Kernel B: multi-block all-to-all put+signal allreduce
#     grid=(TOTAL_BLOCKS,) where TOTAL_BLOCKS = npes * CHUNKS_PER_PE.
#     Multiple blocks put chunks in parallel, all wait signals, all accumulate.
# ===================================================================
@triton.jit
def allreduce_put_signal_kernel(
    input_ptr,      # symmetric bf16: each PE's input (read-only)
    recv_ptr,       # symmetric bf16: recv buffer (npes * N), each PE has its own
    output_ptr,     # bf16: output
    signal_ptr,     # symmetric int64: signal_buf[i] = "data from PE i arrived"
    mype: int, npes: int, N: int,
    CHUNK_SIZE: tl.constexpr, MAX_PES: tl.constexpr, CHUNKS_PER_PE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bid = tl.program_id(0)
    target_pe = bid // CHUNKS_PER_PE       # which PE this block targets
    chunk_id = bid % CHUNKS_PER_PE         # which chunk within that PE's data

    chunk_offset = chunk_id * CHUNK_SIZE   # byte-element offset within data
    chunk_bytes = CHUNK_SIZE * 2           # bf16 = 2 bytes

    # Phase 1: each block puts one chunk to one PE (or self-copies)
    if target_pe < npes:
        if target_pe == mype:
            for off in range(0, CHUNK_SIZE, BLOCK_SIZE):
                o = chunk_offset + off + tl.arange(0, BLOCK_SIZE)
                m = o < N
                tl.store(recv_ptr + mype * N + o,
                         tl.load(input_ptr + o, mask=m), mask=m)
        else:
            # Put this chunk to target_pe's recv_buf[mype] at chunk_offset
            # Use SIGNAL_ADD so all chunks from all blocks to the same PE
            # accumulate into one signal. When signal == CHUNKS_PER_PE, all done.
            shmem_putmem_nbi_signal_block(
                recv_ptr + mype * N + chunk_offset,
                input_ptr + chunk_offset,
                tl.cast(chunk_bytes, tl.uint64),
                signal_ptr + mype,
                tl.full([], 1, tl.uint64),
                SIGNAL_ADD,
                target_pe,
                0,
            )
            shmem_quiet()

    # Phase 2: all blocks wait for all signals
    # Each PE has CHUNKS_PER_PE blocks sending to it, each adding 1.
    # Signal from PE i is complete when signal_buf[i] == CHUNKS_PER_PE.
    for i in tl.static_range(MAX_PES):
        if i < npes and i != mype:
            shmem_uint64_wait_until_equals(
                signal_ptr + i, tl.cast(CHUNKS_PER_PE, tl.uint64))

    # Phase 3: multi-block accumulate across all TOTAL_BLOCKS blocks
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
# 4. Bitcode + hook
# ===================================================================
def _find_mori_shmem_bc():
    candidates = []
    if os.environ.get("MORI_SHMEM_BC"):
        candidates.append(os.environ["MORI_SHMEM_BC"])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mori_root = os.path.abspath(os.path.join(script_dir, "../../.."))
    candidates.append(os.path.join(mori_root, "lib", "libmori_shmem_device.bc"))
    candidates.append(os.path.join(mori_root, "build", "lib",
                                   "libmori_shmem_device.bc"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"libmori_shmem_device.bc not found: {candidates}")


def _install_shmem_hook():
    def hook(*args, **kwargs):
        key = kwargs["key"]
        jit_function = kwargs["fn"].jit_function
        device = kwargs["compile"]["device"]
        kernel = jit_function.device_caches[device][0].get(key)
        if kernel is None:
            return
        kernel._init_handles()
        import mori.shmem as ms
        ms.shmem_module_init(kernel.module)
    knobs.runtime.jit_post_compile_hook = hook


def _get_extern_libs():
    return {"mori_shmem": _find_mori_shmem_bc()}


# ===================================================================
# 5. Distributed setup
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
# 6. Build P2P pointer array
# ===================================================================
def build_p2p_ptrs(data_buf, mype, npes):
    """Compute P2P pointers to each PE's symmetric buffer, return as int64 tensor."""
    import mori.shmem as ms
    ptrs = torch.zeros(npes, dtype=torch.int64, device="cuda")
    local_ptr = data_buf.data_ptr()
    for pe in range(npes):
        if pe == mype:
            ptrs[pe] = local_ptr
        else:
            p2p = ms.shmem_ptr_p2p(local_ptr, mype, pe)
            assert p2p != 0, f"PE {mype}: P2P not available for PE {pe}"
            ptrs[pe] = p2p
    return ptrs


# ===================================================================
# 7. Benchmark helper
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
    return s.elapsed_time(e) / iters * 1000  # us


# ===================================================================
# 8. Test
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

    # Symmetric buffers
    symm_buf = mori_shmem_create_tensor((N,), torch.bfloat16)
    symm_buf.copy_(local_data.view(-1))
    result_a = torch.empty(N, dtype=torch.bfloat16, device="cuda")

    torch.cuda.synchronize()

    # Reference
    local_cpu = local_data.cpu()
    all_data_cpu = [torch.empty_like(local_cpu) for _ in range(npes)]
    dist.all_gather(all_data_cpu, local_cpu)
    expected = torch.stack(all_data_cpu).to(torch.float32).sum(dim=0).to(torch.bfloat16).view(-1).cuda()

    ms.shmem_barrier_all()

    # ── Kernel A: device-side shmem_ptr_p2p ──
    print(f"[PE {mype}] --- Kernel A: device-side shmem_ptr_p2p ---")
    grid_a = (triton.cdiv(N, 1024),)  # trigger autotune
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

    # ── Kernel B: all-to-all put+signal ──
    # Detect transport: if shmem_ptr_p2p returns 0 for any peer → RDMA path
    is_rdma = False
    for pe in range(npes):
        if pe != mype:
            if ms.shmem_ptr_p2p(symm_buf.data_ptr(), mype, pe) == 0:
                is_rdma = True
                break

    # P2P: multi-chunk per PE (8 blocks per PE = 64 total)
    # RDMA/IBGDA: 1 block per PE (avoid QP contention)
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

    # Timed run
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

    # Summary
    if mype == 0:
        print(f"\n[PE 0] Summary ({npes} PEs, {M}x{K} bf16 = {nbytes//1024} KB):")
        print(f"  Kernel A (shmem_ptr_p2p):      {us_a:.1f} us, {bw_a:.1f} GB/s")
        print(f"  Kernel B (put+signal, 1 kern): {us_b:.1f} us")


# ===================================================================
# main
# ===================================================================
def main():
    _install_shmem_hook()
    extern_libs = _get_extern_libs()

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
