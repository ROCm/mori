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

    # IO kernels use a different source directory
    io_kernels = [
        ("scatter_gather", "src/io/kernels"),
    ]

    def _compile_bc():
        return "shmem bitcode", ensure_bitcode()

    def _compile_genco(name, source_dir="src/ops/kernels"):
        return name, compile_genco(name, source_dir=source_dir)

    total_tasks = len(all_kernels) + len(io_kernels) + 1
    with ThreadPoolExecutor(max_workers=total_tasks) as pool:
        futures = [pool.submit(_compile_bc)]
        for name in all_kernels:
            futures.append(pool.submit(_compile_genco, name))
        for name, src_dir in io_kernels:
            futures.append(pool.submit(_compile_genco, name, src_dir))

        for future in as_completed(futures):
            try:
                label, path = future.result()
                print(f"[mori-jit]   {label}: {path}")
            except Exception as e:
                print(f"[mori-jit]   SKIPPED ({e})")

    print(f"[mori-jit] Precompilation done ({time.time() - t0:.1f}s).")
