#!/usr/bin/env python3
"""Standalone precompile script — does NOT import mori.__init__ (avoids native .so loading)."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from mori.jit.config import detect_build_config, detect_nic_type
from mori.jit.core import compile_genco, ensure_bitcode

ALL_KERNELS = [
    "ep_intranode", "ep_internode", "ep_internode_v1",
    "ep_internode_v1ll", "ep_async_ll", "cast_kernel",
]


def main():
    cfg = detect_build_config()
    nic = detect_nic_type()
    print(f"[mori-jit] Pre-compiling all kernels (arch={cfg.arch}, nic={nic}) ...")
    t0 = time.time()

    def _bc():
        return "shmem bitcode", ensure_bitcode()

    def _genco(name):
        return name, compile_genco(name)

    with ThreadPoolExecutor(max_workers=len(ALL_KERNELS) + 1) as pool:
        futures = [pool.submit(_bc)]
        for k in ALL_KERNELS:
            futures.append(pool.submit(_genco, k))

        for f in as_completed(futures):
            try:
                label, path = f.result()
                print(f"[mori-jit]   {label}: {path}")
            except Exception as e:
                print(f"[mori-jit]   SKIPPED ({e})")

    print(f"[mori-jit] Pre-compilation done ({time.time() - t0:.1f}s).")


if __name__ == "__main__":
    main()
