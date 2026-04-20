#!/usr/bin/env python3
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
"""Standalone precompile script — does NOT import mori.__init__ (avoids native .so loading)."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from mori.jit.config import detect_build_config, detect_nic_type
from mori.jit.core import compile_genco, ensure_bitcode

ALL_KERNELS = [
    "ep_intranode",
    "ep_internode",
    "ep_internode_v1",
    "ep_internode_v1ll",
    "ep_async_ll",
    "ep_local_expert_count",
    "cast_kernel",
]

IO_KERNELS = [
    ("scatter_gather", "src/io/kernels"),
]


def main():
    cfg = detect_build_config()
    nic = detect_nic_type()
    print(f"[mori-jit] Pre-compiling all kernels (arch={cfg.arch}, nic={nic}) ...")
    t0 = time.time()

    def _bc():
        return "shmem bitcode", ensure_bitcode()

    def _genco(name, source_dir="src/ops/kernels"):
        return name, compile_genco(name, source_dir=source_dir)

    total = len(ALL_KERNELS) + len(IO_KERNELS) + 1
    with ThreadPoolExecutor(max_workers=total) as pool:
        futures = [pool.submit(_bc)]
        for k in ALL_KERNELS:
            futures.append(pool.submit(_genco, k))
        for name, src_dir in IO_KERNELS:
            futures.append(pool.submit(_genco, name, src_dir))

        for f in as_completed(futures):
            try:
                label, path = f.result()
                print(f"[mori-jit]   {label}: {path}")
            except Exception as e:
                print(f"[mori-jit]   SKIPPED ({e})")

    print(f"[mori-jit] Pre-compilation done ({time.time() - t0:.1f}s).")


if __name__ == "__main__":
    main()
