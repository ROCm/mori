#!/usr/bin/env python3
"""
Mori shmem via Triton — uses vanilla @triton.jit directly.

Uses vanilla @triton.jit with:
  - extern_libs to link libmori_shmem_device.bc
  - @core.extern to declare mori device functions
  - triton.knobs hook for shmem_module_init

Usage:
    torchrun --nproc_per_node=2 test_triton_shmem.py [chip]
"""

import os
import sys
import re
import tempfile
import subprocess

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language import core
from triton import knobs


# ===================================================================
# 1. Mori shmem device function declarations (@core.extern)
# ===================================================================
@core.extern
def shmem_my_pe(_semantic=None):
    return core.extern_elementwise("libmori_shmem_device", "", [],
                                   {(): ("mori_shmem_my_pe", tl.int32)},
                                   is_pure=False, _semantic=_semantic)


@core.extern
def shmem_n_pes(_semantic=None):
    return core.extern_elementwise("libmori_shmem_device", "", [],
                                   {(): ("mori_shmem_n_pes", tl.int32)},
                                   is_pure=True, _semantic=_semantic)


@core.extern
def shmem_int32_p(dest, value, pe, qp_id, _semantic=None):
    return core.extern_elementwise(
        "libmori_shmem_device", "",
        [tl.cast(dest, tl.uint64, _semantic=_semantic),
         tl.cast(value, tl.int32, _semantic=_semantic),
         tl.cast(pe, tl.int32, _semantic=_semantic),
         tl.cast(qp_id, tl.int32, _semantic=_semantic)],
        {(tl.uint64, tl.int32, tl.int32, tl.int32):
         ("mori_shmem_int32_p", tl.int32)},
        is_pure=False, _semantic=_semantic)


@core.extern
def shmem_quiet(_semantic=None):
    return core.extern_elementwise("libmori_shmem_device", "", [],
                                   {(): ("mori_shmem_quiet_thread", tl.int32)},
                                   is_pure=False, _semantic=_semantic)


@core.extern
def shmem_barrier_all_device(_semantic=None):
    return core.extern_elementwise("libmori_shmem_device", "", [],
                                   {(): ("mori_shmem_barrier_all_thread", tl.int32)},
                                   is_pure=False, _semantic=_semantic)


# ===================================================================
# 2. Triton kernels (vanilla @triton.jit)
# ===================================================================
@triton.jit
def shmem_basic_kernel(out_ptr):
    mype = shmem_my_pe()
    npes = shmem_n_pes()
    tl.store(out_ptr, mype)
    tl.store(out_ptr + 1, npes)


@triton.jit
def shmem_put_kernel(symm_buf_ptr, value):
    mype = shmem_my_pe()
    npes = shmem_n_pes()
    dest_pe = (mype + 1) % npes
    shmem_int32_p(symm_buf_ptr, value, dest_pe, 0)
    shmem_quiet()


# ===================================================================
# 3. Bitcode + hook setup
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
    """Post-compile hook: initialize globalGpuStates in each kernel module."""
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


def _strip_lifetime_intrinsics(bc_path):
    """Strip llvm.lifetime intrinsics to avoid LLVM version mismatches."""
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    llvm_dis = os.path.join(rocm, "llvm", "bin", "llvm-dis")
    llvm_link = os.path.join(rocm, "llvm", "bin", "llvm-link")
    pid = os.getpid()
    ll_path = os.path.join(tempfile.gettempdir(), f'shmem_device_{pid}.ll')
    clean_ll = os.path.join(tempfile.gettempdir(), f'shmem_device_clean_{pid}.ll')
    clean_bc = os.path.join(tempfile.gettempdir(), f'shmem_device_clean_{pid}.bc')
    subprocess.check_call([llvm_dis, bc_path, '-o', ll_path])
    with open(ll_path) as f:
        text = f.read()
    text = re.sub(r'^\s*call void @llvm\.lifetime\.[^\n]*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^declare void @llvm\.lifetime\.[^\n]*\n', '', text, flags=re.MULTILINE)
    with open(clean_ll, 'w') as f:
        f.write(text)
    subprocess.check_call([llvm_link, clean_ll, '-o', clean_bc])
    return clean_bc


def _get_extern_libs():
    bc = _find_mori_shmem_bc()
    return {"mori_shmem": _strip_lifetime_intrinsics(bc)}


# ===================================================================
# 4. Distributed setup
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
# 5. Tests
# ===================================================================
def test_basic(mype, npes, extern_libs):
    print(f"\n[PE {mype}] === Triton: shmem_basic_kernel ===")
    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    shmem_basic_kernel[(1,)](out, extern_libs=extern_libs)
    torch.cuda.synchronize()
    expected = torch.tensor([mype, npes], dtype=torch.int32, device="cuda")
    print(f"[PE {mype}] Result: {out.tolist()}, Expected: {expected.tolist()}")
    torch.testing.assert_close(out, expected)
    print(f"[PE {mype}] [Triton] basic  PASS")


def test_put(mype, npes, extern_libs):
    import mori.shmem as ms
    from mori.shmem import mori_shmem_create_tensor

    print(f"\n[PE {mype}] === Triton: shmem_put_kernel ===")
    buf = mori_shmem_create_tensor((1,), torch.int32)
    buf.fill_(-1)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    value = mype * 100 + 42
    shmem_put_kernel[(1,)](buf, value, extern_libs=extern_libs)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    src = (mype - 1 + npes) % npes
    exp = src * 100 + 42
    got = buf.item()
    print(f"[PE {mype}] buf={got}, expected={exp} (from PE {src})")
    assert got == exp, f"PE {mype}: expected {exp}, got {got}"
    print(f"[PE {mype}] [Triton] put    PASS")


# ===================================================================
# main
# ===================================================================
def main():
    _install_shmem_hook()
    extern_libs = _get_extern_libs()

    mype, npes = setup_distributed()
    try:
        test_basic(mype, npes, extern_libs)
        test_put(mype, npes, extern_libs)
        if mype == 0:
            print(f"\n{'=' * 60}")
            print(f"  All tests PASSED on {npes} PEs (Triton + mori shmem)")
            print(f"{'=' * 60}")
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
