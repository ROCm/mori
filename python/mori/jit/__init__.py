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
    """Precompile all JIT kernels (bitcode + ops .hsaco) into the cache.

    Compiles all kernel groups in parallel using threads (each spawns hipcc).
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from mori.jit.config import detect_build_config, detect_nic_type

    cfg = detect_build_config()
    nic = detect_nic_type()
    print(f"[mori-jit] Precompiling all kernels (arch={cfg.arch}, nic={nic}) ...")
    t0 = time.time()

    all_kernels = [
        "shmem_kernels",
        "ep_intranode",
        "ep_internode",
        "ep_internode_v1",
        "ep_internode_v1ll",
        "ep_async_ll",
        "cast_kernel",
    ]

    def _compile_bc():
        return "shmem bitcode", ensure_bitcode()

    def _compile_genco(name):
        return name, compile_genco(name)

    with ThreadPoolExecutor(max_workers=len(all_kernels) + 1) as pool:
        futures = [pool.submit(_compile_bc)]
        for name in all_kernels:
            futures.append(pool.submit(_compile_genco, name))

        for future in as_completed(futures):
            try:
                label, path = future.result()
                print(f"[mori-jit]   {label}: {path}")
            except Exception as e:
                print(f"[mori-jit]   SKIPPED ({e})")

    print(f"[mori-jit] Precompilation done ({time.time() - t0:.1f}s).")
