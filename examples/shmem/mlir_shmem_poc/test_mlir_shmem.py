#!/usr/bin/env python3
"""
End-to-end test: two paths, both no Triton.

  Path A: MLIR Python API → mlir-translate → LLVM IR → link bc → .hsaco
  Path B: Direct LLVM IR text →                        link bc → .hsaco

Both paths load via HIP, init mori globalGpuStates, launch, and verify.

Usage:
    torchrun --nproc_per_node=2 test_mlir_shmem.py [chip]
"""

import ctypes
import os
import sys
import tempfile

import torch
import torch.distributed as dist

_hip = ctypes.CDLL("libamdhip64.so")


def _check(err, msg=""):
    if err != 0:
        raise RuntimeError(f"HIP error {err}: {msg}")


def hip_module_load(path):
    mod = ctypes.c_void_p()
    _check(_hip.hipModuleLoad(ctypes.byref(mod), path.encode()), f"load({path})")
    return mod


def hip_get_function(mod, name):
    func = ctypes.c_void_p()
    _check(_hip.hipModuleGetFunction(ctypes.byref(func), mod, name.encode()), name)
    return func


def hip_launch(func, args):
    ptrs = (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
        ptrs[i] = ctypes.cast(ctypes.pointer(a), ctypes.c_void_p)
    _check(_hip.hipModuleLaunchKernel(
        func, 1, 1, 1, 1, 1, 1, 0, None, ptrs, None), "launch")


def hip_sync():
    _check(_hip.hipDeviceSynchronize(), "sync")


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo")
    world_group = dist.group.WORLD
    assert world_group is not None
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


# ---------------------------------------------------------------------------
# Generic test runner — works for both MLIR and LLVM IR .hsaco
# ---------------------------------------------------------------------------
def run_basic_test(tag, hsaco_path, mype, npes):
    import mori.shmem as ms
    hip_mod = hip_module_load(hsaco_path)
    ms.shmem_module_init(hip_mod.value)
    func = hip_get_function(hip_mod, "shmem_basic_kernel")
    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    hip_launch(func, [ctypes.c_void_p(out.data_ptr())])
    hip_sync()
    expected = torch.tensor([mype, npes], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(out, expected)
    print(f"[PE {mype}] {tag} basic: {out.tolist()} == {expected.tolist()}  PASS")


def run_put_test(tag, hsaco_path, mype, npes):
    import mori.shmem as ms
    from mori.shmem import mori_shmem_create_tensor
    hip_mod = hip_module_load(hsaco_path)
    ms.shmem_module_init(hip_mod.value)
    func = hip_get_function(hip_mod, "shmem_put_kernel")
    buf = mori_shmem_create_tensor((1,), torch.int32)
    buf.fill_(-1)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    val = mype * 100 + 42
    hip_launch(func, [ctypes.c_void_p(buf.data_ptr()), ctypes.c_int32(val)])
    hip_sync()
    ms.shmem_barrier_all()
    src = (mype - 1 + npes) % npes
    exp = src * 100 + 42
    got = buf.item()
    assert got == exp, f"PE {mype}: expected {exp}, got {got}"
    print(f"[PE {mype}] {tag} put:   buf={got} (from PE {src})  PASS")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    chip = sys.argv[1] if len(sys.argv) > 1 else "gfx942"
    poc_dir = os.path.dirname(os.path.abspath(__file__))
    if poc_dir not in sys.path:
        sys.path.insert(0, poc_dir)

    mype, npes = setup_distributed()
    tmp = tempfile.gettempdir()

    from mlir_shmem_kernel import (
        build_shmem_basic_kernel, build_shmem_put_kernel,
        compile_to_hsaco, _find_mori_shmem_bc,
        compile_ir_to_hsaco, SHMEM_BASIC_KERNEL_IR, SHMEM_PUT_KERNEL_IR,
    )
    mori_bc = _find_mori_shmem_bc()

    try:
        # ============================================================
        # Path A: MLIR
        # ============================================================
        print(f"\n[PE {mype}] ===== Path A: MLIR =====")

        m1 = build_shmem_basic_kernel()
        h1 = compile_to_hsaco(m1, chip, os.path.join(tmp, f"mlir_basic_pe{mype}.hsaco"), mori_bc)
        run_basic_test("[MLIR]", h1, mype, npes)

        m2 = build_shmem_put_kernel()
        h2 = compile_to_hsaco(m2, chip, os.path.join(tmp, f"mlir_put_pe{mype}.hsaco"), mori_bc)
        run_put_test("[MLIR]", h2, mype, npes)

        # ============================================================
        # Path B: Direct LLVM IR
        # ============================================================
        print(f"\n[PE {mype}] ===== Path B: LLVM IR =====")

        h3 = compile_ir_to_hsaco(SHMEM_BASIC_KERNEL_IR, chip,
                                 os.path.join(tmp, f"llvmir_basic_pe{mype}.hsaco"), mori_bc)
        run_basic_test("[LLVM IR]", h3, mype, npes)

        h4 = compile_ir_to_hsaco(SHMEM_PUT_KERNEL_IR, chip,
                                 os.path.join(tmp, f"llvmir_put_pe{mype}.hsaco"), mori_bc)
        run_put_test("[LLVM IR]", h4, mype, npes)

        # ============================================================
        if mype == 0:
            print(f"\n{'=' * 60}")
            print(f"  All 4 tests PASSED on {npes} PEs")
            print(f"    Path A (MLIR):     basic + put  PASS")
            print(f"    Path B (LLVM IR):  basic + put  PASS")
            print(f"{'=' * 60}")

    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
