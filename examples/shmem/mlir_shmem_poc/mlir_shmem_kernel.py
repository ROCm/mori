#!/usr/bin/env python3
"""
MLIR-based mori shmem kernel — no Triton dependency.

Builds GPU kernels programmatically with MLIR Python bindings (llvm + rocdl
dialects), translates to LLVM IR, links with libmori_shmem_device.bc, and
compiles to .hsaco via ROCm clang.

Pipeline:
    Python (mlir.ir)  →  MLIR (llvm+rocdl)  →  LLVM IR  →  link bc  →  .hsaco

Usage:
    python3 mlir_shmem_kernel.py [chip]   # default: gfx942
"""

import os
import shutil
import subprocess
import sys
import tempfile

from mlir.ir import (
    Attribute,
    Context,
    DenseI32ArrayAttr,
    FlatSymbolRefAttr,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    Operation,
    Type,
    TypeAttr,
    UnitAttr,
)
from mlir.dialects import llvm, rocdl

ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
LLVM_LINK = os.path.join(ROCM_PATH, "lib/llvm/bin/llvm-link")
ROCM_CLANG = os.path.join(ROCM_PATH, "llvm/bin/clang")
MLIR_TRANSLATE = (
    shutil.which("mlir-translate-20")
    or shutil.which("mlir-translate")
    or "/tmp/mlir-build/bin/mlir-translate"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_no_bundle = None  # lazily created inside Context


def _get_no_bundle():
    global _no_bundle
    if _no_bundle is None:
        _no_bundle = DenseI32ArrayAttr.get([])
    return _no_bundle


def declare(name: str, sig: str):
    """Declare an external llvm.func (body provided by bitcode)."""
    f = llvm.LLVMFuncOp(name, TypeAttr.get(Type.parse(sig)))
    f.operation.attributes["sym_visibility"] = Attribute.parse('"private"')
    return f


def call(result_type, callee: str, operands=None):
    """Call an extern function that returns a value."""
    if operands is None:
        operands = []
    op = llvm.CallOp(result_type, operands, [], _get_no_bundle(),
                     callee=FlatSymbolRefAttr.get(callee))
    return op.result


def call_void(callee: str, operands=None):
    """Call an extern function that returns void."""
    if operands is None:
        operands = []
    Operation.create(
        "llvm.call",
        results=[],
        operands=operands,
        attributes={
            "callee": FlatSymbolRefAttr.get(callee),
            "operandSegmentSizes": DenseI32ArrayAttr.get([len(operands), 0]),
            "op_bundle_sizes": DenseI32ArrayAttr.get([]),
            "CConv": Attribute.parse("#llvm.cconv<ccc>"),
            "TailCallKind": Attribute.parse("#llvm.tailcallkind<none>"),
            "fastmathFlags": Attribute.parse("#llvm.fastmath<none>"),
        },
    )


def _find_mori_shmem_bc() -> str:
    """Locate libmori_shmem_device.bc.

    Search order:
      1. $MORI_SHMEM_BC  (explicit override)
      2. $MORI_HOME/lib/
      3. <mori_repo>/lib/          (output of tools/build_shmem_bitcode.sh)
      4. <mori_repo>/build/lib/
      5. <triton-dist>/python/triton_dist/tools/compile/  (legacy fallback)
    """
    candidates = []
    if os.environ.get("MORI_SHMEM_BC"):
        candidates.append(os.environ["MORI_SHMEM_BC"])
    if os.environ.get("MORI_HOME"):
        candidates.append(os.path.join(os.environ["MORI_HOME"], "lib",
                                       "libmori_shmem_device.bc"))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mori_root = os.path.abspath(os.path.join(script_dir, "../../.."))
    candidates.append(os.path.join(mori_root, "lib", "libmori_shmem_device.bc"))
    candidates.append(os.path.join(mori_root, "build", "lib",
                                   "libmori_shmem_device.bc"))

    # Legacy: triton-dist workspace
    ws = os.path.abspath(os.path.join(mori_root, "../.."))
    candidates.append(os.path.join(ws, "python/triton_dist/tools/compile",
                                   "libmori_shmem_device.bc"))

    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "libmori_shmem_device.bc not found.\n"
        "Run: bash tools/build_shmem_bitcode.sh\n"
        f"Searched: {candidates}")


# ---------------------------------------------------------------------------
# 1. Build kernel IR with MLIR Python API
# ---------------------------------------------------------------------------
def build_shmem_basic_kernel() -> Module:
    """
    void shmem_basic_kernel(int32_t *out) {
        out[0] = mori_shmem_my_pe();
        out[1] = mori_shmem_n_pes();
    }
    """
    ctx = Context()
    with ctx, Location.unknown():
        module = Module.create()
        ptr = llvm.PointerType.get()
        i32 = IntegerType.get_signless(32)
        i64 = IntegerType.get_signless(64)
        gep_dyn = DenseI32ArrayAttr.get([-2147483648])

        with InsertionPoint(module.body):
            declare("mori_shmem_my_pe", "!llvm.func<i32 ()>")
            declare("mori_shmem_n_pes", "!llvm.func<i32 ()>")

            fn = llvm.LLVMFuncOp(
                "shmem_basic_kernel",
                TypeAttr.get(Type.parse("!llvm.func<void (ptr)>")))
            fn.operation.attributes["rocdl.kernel"] = UnitAttr.get()

            entry = fn.body.blocks.append(ptr)
            with InsertionPoint(entry):
                out_ptr = entry.arguments[0]

                mype = call(i32, "mori_shmem_my_pe")
                npes = call(i32, "mori_shmem_n_pes")

                # out[0] = mype
                llvm.StoreOp(mype, out_ptr)

                # out[1] = npes
                one = llvm.ConstantOp(i64, IntegerAttr.get(i64, 1)).result
                out1 = llvm.GEPOp(ptr, out_ptr, [one], gep_dyn, i32).result
                llvm.StoreOp(npes, out1)

                llvm.ReturnOp()

        assert module.operation.verify()
        return module


def build_shmem_put_kernel() -> Module:
    """
    void shmem_put_kernel(int32_t *symm_buf, int32_t value) {
        int mype = mori_shmem_my_pe();
        int npes = mori_shmem_n_pes();
        int dest_pe = (mype + 1) % npes;
        mori_shmem_int32_p(symm_buf, value, dest_pe, 0);
        mori_shmem_quiet_thread();
    }
    """
    ctx = Context()
    with ctx, Location.unknown():
        module = Module.create()
        ptr = llvm.PointerType.get()
        i32 = IntegerType.get_signless(32)
        no_of = Attribute.parse("#llvm.overflow<none>")

        with InsertionPoint(module.body):
            declare("mori_shmem_my_pe", "!llvm.func<i32 ()>")
            declare("mori_shmem_n_pes", "!llvm.func<i32 ()>")
            declare("mori_shmem_int32_p",
                    "!llvm.func<void (ptr, i32, i32, i32)>")
            declare("mori_shmem_quiet_thread", "!llvm.func<void ()>")

            fn = llvm.LLVMFuncOp(
                "shmem_put_kernel",
                TypeAttr.get(Type.parse("!llvm.func<void (ptr, i32)>")))
            fn.operation.attributes["rocdl.kernel"] = UnitAttr.get()

            entry = fn.body.blocks.append(ptr, i32)
            with InsertionPoint(entry):
                symm_buf, value = entry.arguments

                mype = call(i32, "mori_shmem_my_pe")
                npes = call(i32, "mori_shmem_n_pes")

                # dest_pe = (mype + 1) % npes
                one = llvm.ConstantOp(i32, IntegerAttr.get(i32, 1)).result
                mype1 = llvm.AddOp(mype, one, no_of).result
                dest_pe = llvm.SRemOp(mype1, npes).result

                zero = llvm.ConstantOp(i32, IntegerAttr.get(i32, 0)).result

                call_void("mori_shmem_int32_p",
                          [symm_buf, value, dest_pe, zero])
                call_void("mori_shmem_quiet_thread")

                llvm.ReturnOp()

        assert module.operation.verify()
        return module


# ---------------------------------------------------------------------------
# 2. Compile:  MLIR module → LLVM IR → link bitcode → .hsaco
# ---------------------------------------------------------------------------
def compile_to_hsaco(module: Module, chip: str, out_path: str,
                     mori_bc: str) -> str:
    """Compile MLIR module to .hsaco, linking with mori shmem bitcode."""
    mlir_text = str(module)
    tmpdir = tempfile.mkdtemp(prefix="mori_mlir_")
    try:
        # MLIR → LLVM IR
        r = subprocess.run(
            [MLIR_TRANSLATE, "--mlir-to-llvmir"],
            input=mlir_text, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"mlir-translate failed:\n{r.stderr}")

        kernel_ll = os.path.join(tmpdir, "kernel.ll")
        with open(kernel_ll, "w") as f:
            f.write(r.stdout)

        # Link with mori bitcode
        linked_bc = os.path.join(tmpdir, "linked.bc")
        r = subprocess.run(
            [LLVM_LINK, kernel_ll, mori_bc, "-o", linked_bc],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"llvm-link failed:\n{r.stderr}")

        # Compile to .hsaco
        r = subprocess.run(
            [ROCM_CLANG,
             "-x", "ir", linked_bc,
             "-target", "amdgcn-amd-amdhsa",
             f"-mcpu={chip}",
             "-O3",
             "-o", out_path],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"clang failed:\n{r.stderr}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return out_path


# ---------------------------------------------------------------------------
# 3. Direct LLVM IR path (no MLIR dependency)
# ---------------------------------------------------------------------------
SHMEM_BASIC_KERNEL_IR = """\
declare i32 @mori_shmem_my_pe()
declare i32 @mori_shmem_n_pes()

define amdgpu_kernel void @shmem_basic_kernel(ptr %out) #0 {
entry:
  %mype = call i32 @mori_shmem_my_pe()
  %npes = call i32 @mori_shmem_n_pes()
  store i32 %mype, ptr %out, align 4
  %out1 = getelementptr inbounds i32, ptr %out, i64 1
  store i32 %npes, ptr %out1, align 4
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1" }
"""

SHMEM_PUT_KERNEL_IR = """\
declare i32 @mori_shmem_my_pe()
declare i32 @mori_shmem_n_pes()
declare void @mori_shmem_int32_p(ptr, i32, i32, i32)
declare void @mori_shmem_quiet_thread()

define amdgpu_kernel void @shmem_put_kernel(ptr %symm_buf, i32 %value) #0 {
entry:
  %mype = call i32 @mori_shmem_my_pe()
  %npes = call i32 @mori_shmem_n_pes()
  %mype1 = add i32 %mype, 1
  %dest_pe = srem i32 %mype1, %npes
  call void @mori_shmem_int32_p(ptr %symm_buf, i32 %value, i32 %dest_pe, i32 0)
  call void @mori_shmem_quiet_thread()
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1" }
"""


def compile_ir_to_hsaco(llvm_ir: str, chip: str, out_path: str,
                        mori_bc: str) -> str:
    """Compile raw LLVM IR text to .hsaco (no MLIR needed)."""
    tmpdir = tempfile.mkdtemp(prefix="mori_llvmir_")
    try:
        kernel_ll = os.path.join(tmpdir, "kernel.ll")
        with open(kernel_ll, "w") as f:
            f.write(llvm_ir)

        linked_bc = os.path.join(tmpdir, "linked.bc")
        r = subprocess.run(
            [LLVM_LINK, kernel_ll, mori_bc, "-o", linked_bc],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"llvm-link failed:\n{r.stderr}")

        r = subprocess.run(
            [ROCM_CLANG,
             "-x", "ir", linked_bc,
             "-target", "amdgcn-amd-amdhsa",
             f"-mcpu={chip}",
             "-O3",
             "-o", out_path],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"clang failed:\n{r.stderr}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return out_path


# ---------------------------------------------------------------------------
# main — compile-only test
# ---------------------------------------------------------------------------
def main():
    chip = sys.argv[1] if len(sys.argv) > 1 else "gfx942"
    print(f"[*] Target: {chip}")
    mori_bc = _find_mori_shmem_bc()
    print(f"[*] Bitcode: {mori_bc}")

    tmpdir = tempfile.gettempdir()

    print("\n[1/2] Building shmem_basic_kernel (MLIR path) ...")
    m1 = build_shmem_basic_kernel()
    print(m1)
    h1 = compile_to_hsaco(m1, chip, os.path.join(tmpdir,
                          "shmem_basic_kernel.hsaco"), mori_bc)
    print(f"[OK] {h1} ({os.path.getsize(h1)} bytes)")

    print("\n[2/2] Building shmem_put_kernel (MLIR path) ...")
    m2 = build_shmem_put_kernel()
    print(m2)
    h2 = compile_to_hsaco(m2, chip, os.path.join(tmpdir,
                          "shmem_put_kernel.hsaco"), mori_bc)
    print(f"[OK] {h2} ({os.path.getsize(h2)} bytes)")

    print("\n[*] All kernels compiled successfully!")


if __name__ == "__main__":
    main()
