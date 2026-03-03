# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""MORI JIT compilation framework.

Compiles device bitcode and host libraries on-demand at runtime.
Results are cached to ~/.mori/jit/ for subsequent runs.

To precompile all kernels (avoid first-run latency)::

    MORI_PRECOMPILE=1 python -c "import mori"
"""

import os

from mori.jit.core import compile_genco, ensure_bitcode

__all__ = ["compile_genco", "ensure_bitcode", "precompile"]


def precompile():
    """Precompile all JIT kernels (bitcode + ops .hsaco) into the cache."""
    from mori.jit.config import detect_build_config, detect_nic_type
    cfg = detect_build_config()
    nic = detect_nic_type()
    print(f"[mori-jit] Precompiling all kernels (arch={cfg.arch}, nic={nic}) ...")

    try:
        path = ensure_bitcode()
        print(f"[mori-jit]   shmem bitcode: {path}")
    except Exception as e:
        print(f"[mori-jit]   shmem bitcode: SKIPPED ({e})")

    for name in ["dispatch_combine_kernels", "cast_kernel"]:
        try:
            path = compile_genco(name)
            print(f"[mori-jit]   {name}: {path}")
        except Exception as e:
            print(f"[mori-jit]   {name}: SKIPPED ({e})")

    print("[mori-jit] Precompilation done.")


