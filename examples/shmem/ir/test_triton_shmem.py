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
"""
Mori shmem via Triton — uses mori.ir.triton integration layer.

Usage:
    torchrun --nproc_per_node=2 test_triton_shmem.py
"""

import os
import sys

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from mori.ir import triton as mori_shmem_device
from mori.ir.triton import get_extern_libs, install_hook


# ===================================================================
# 1. Triton kernels
# ===================================================================
@triton.jit
def shmem_basic_kernel(out_ptr):
    mype = mori_shmem_device.my_pe()
    npes = mori_shmem_device.n_pes()
    tl.store(out_ptr, mype)
    tl.store(out_ptr + 1, npes)


@triton.jit
def shmem_put_kernel(symm_buf_ptr, value):
    mype = mori_shmem_device.my_pe()
    npes = mori_shmem_device.n_pes()
    dest_pe = (mype + 1) % npes
    mori_shmem_device.int32_p(symm_buf_ptr, value, dest_pe, 0)
    mori_shmem_device.quiet_thread()


@triton.jit
def shmem_get_nbi_kernel(local_buf_ptr, remote_buf_ptr, nbytes):
    mype = mori_shmem_device.my_pe()
    npes = mori_shmem_device.n_pes()
    src_pe = (mype + 1) % npes
    mori_shmem_device.getmem_nbi_thread(
        local_buf_ptr, remote_buf_ptr, nbytes, src_pe, 0
    )
    mori_shmem_device.quiet_thread()


@triton.jit
def shmem_get_blocking_kernel(local_buf_ptr, remote_buf_ptr, nbytes):
    mype = mori_shmem_device.my_pe()
    npes = mori_shmem_device.n_pes()
    src_pe = (mype + 1) % npes
    mori_shmem_device.getmem_thread(local_buf_ptr, remote_buf_ptr, nbytes, src_pe, 0)


# ===================================================================
# 2. Distributed setup
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
# 3. Tests
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


def test_get_nbi(mype, npes, extern_libs):
    import mori.shmem as ms
    from mori.shmem import mori_shmem_create_tensor

    print(f"\n[PE {mype}] === Triton: shmem_get_nbi_kernel ===")
    remote_buf = mori_shmem_create_tensor((1,), torch.int32)
    remote_buf.fill_(mype * 100 + 42)
    local_buf = mori_shmem_create_tensor((1,), torch.int32)
    local_buf.fill_(-1)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    nbytes = 4
    shmem_get_nbi_kernel[(1,)](local_buf, remote_buf, nbytes, extern_libs=extern_libs)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    src_pe = (mype + 1) % npes
    exp = src_pe * 100 + 42
    got = local_buf.item()
    print(f"[PE {mype}] local_buf={got}, expected={exp} (GET nbi from PE {src_pe})")
    assert got == exp, f"PE {mype}: expected {exp}, got {got}"
    print(f"[PE {mype}] [Triton] get_nbi      PASS")


def test_get_blocking(mype, npes, extern_libs):
    import mori.shmem as ms
    from mori.shmem import mori_shmem_create_tensor

    print(f"\n[PE {mype}] === Triton: shmem_get_blocking_kernel ===")
    remote_buf = mori_shmem_create_tensor((1,), torch.int32)
    remote_buf.fill_(mype * 200 + 99)
    local_buf = mori_shmem_create_tensor((1,), torch.int32)
    local_buf.fill_(-1)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    nbytes = 4
    shmem_get_blocking_kernel[(1,)](
        local_buf, remote_buf, nbytes, extern_libs=extern_libs
    )
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    src_pe = (mype + 1) % npes
    exp = src_pe * 200 + 99
    got = local_buf.item()
    print(
        f"[PE {mype}] local_buf={got}, expected={exp} (GET blocking from PE {src_pe})"
    )
    assert got == exp, f"PE {mype}: expected {exp}, got {got}"
    print(f"[PE {mype}] [Triton] get_blocking  PASS")


# ===================================================================
# main
# ===================================================================
def main():
    install_hook()
    extern_libs = get_extern_libs()

    mype, npes = setup_distributed()
    try:
        test_basic(mype, npes, extern_libs)
        test_put(mype, npes, extern_libs)
        test_get_nbi(mype, npes, extern_libs)
        test_get_blocking(mype, npes, extern_libs)
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
