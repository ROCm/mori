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


# ===================================================================
# 3. Allreduce kernel (with autotune)
# ===================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=32, num_stages=1),
    ],
    key=["N", "npes"],
)
@triton.jit
def allreduce_sum_kernel(
    pe_ptrs,
    result_ptr,
    npes,
    N,
    BLOCK_SIZE: tl.constexpr,
    MAX_PES: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in tl.static_range(MAX_PES):
        if i < npes:
            ptr_int = tl.load(pe_ptrs + i)
            ptr = ptr_int.to(tl.pointer_type(tl.bfloat16), bitcast=True)
            data = tl.load(ptr + offs, mask=mask, other=0.0)
            acc += data.to(tl.float32)

    tl.store(result_ptr + offs, acc.to(tl.bfloat16), mask=mask)


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
# 7. Test
# ===================================================================
def test_allreduce(mype, npes, extern_libs):
    import mori.shmem as ms
    from mori.shmem import mori_shmem_create_tensor

    M, K = 64, 7168
    N = M * K
    MAX_PES = 8

    print(f"\n[PE {mype}] === Triton allreduce sum (bf16, {M}x{K}, N={N}) ===")

    torch.manual_seed(42 + mype)
    local_data = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    symm_buf = mori_shmem_create_tensor((N,), torch.bfloat16)
    symm_buf.copy_(local_data.view(-1))
    torch.cuda.synchronize()

    result = torch.empty(N, dtype=torch.bfloat16, device="cuda")

    local_cpu = local_data.cpu()
    all_data_cpu = [torch.empty_like(local_cpu) for _ in range(npes)]
    dist.all_gather(all_data_cpu, local_cpu)
    expected = torch.stack(all_data_cpu).to(torch.float32).sum(dim=0).to(torch.bfloat16).view(-1).cuda()

    pe_ptrs = build_p2p_ptrs(symm_buf, mype, npes)
    ms.shmem_barrier_all()

    def launch():
        allreduce_sum_kernel[(triton.cdiv(N, allreduce_sum_kernel.best_config.kwargs.get("BLOCK_SIZE", 4096)),)](
            pe_ptrs, result, npes, N,
            MAX_PES=MAX_PES,
            extern_libs=extern_libs,
        )

    # First launch triggers autotune
    allreduce_sum_kernel[(triton.cdiv(N, 1024),)](
        pe_ptrs, result, npes, N,
        MAX_PES=MAX_PES,
        extern_libs=extern_libs,
    )
    torch.cuda.synchronize()

    # Print best config
    best = allreduce_sum_kernel.best_config
    print(f"[PE {mype}] autotune best: BLOCK_SIZE={best.kwargs['BLOCK_SIZE']}, "
          f"num_warps={best.num_warps}, num_stages={best.num_stages}")

    # Re-launch with best config for verification
    best_bs = best.kwargs["BLOCK_SIZE"]
    grid = (triton.cdiv(N, best_bs),)
    allreduce_sum_kernel[grid](
        pe_ptrs, result, npes, N,
        MAX_PES=MAX_PES,
        extern_libs=extern_libs,
    )
    torch.cuda.synchronize()

    max_err = (result.float() - expected.float()).abs().max().item()
    mean_err = (result.float() - expected.float()).abs().mean().item()
    print(f"[PE {mype}] max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    torch.testing.assert_close(result.view(M, K), expected.view(M, K), atol=1e-1, rtol=1e-1)
    print(f"[PE {mype}] [Triton] allreduce  PASS")

    # Benchmark
    ms.shmem_barrier_all()
    torch.cuda.synchronize()
    warmup, iters = 10, 50
    for _ in range(warmup):
        allreduce_sum_kernel[grid](
            pe_ptrs, result, npes, N, MAX_PES=MAX_PES,
            extern_libs=extern_libs,
        )
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        allreduce_sum_kernel[grid](
            pe_ptrs, result, npes, N, MAX_PES=MAX_PES,
            extern_libs=extern_libs,
        )
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    us = elapsed_ms / iters * 1000
    nbytes = N * 2 * npes
    bw_gb = nbytes / (us * 1e-6) / 1e9
    print(f"[PE {mype}] avg {us:.1f} us/iter, {bw_gb:.1f} GB/s "
          f"({N * 2 / 1024:.0f} KB x {npes} PEs)")


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
            print(f"  Allreduce PASSED on {npes} PEs")
            print(f"  (Triton + mori shmem P2P)")
            print(f"{'=' * 60}")
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
