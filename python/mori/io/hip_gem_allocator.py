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
"""Torch pluggable allocator that keeps RDMA-registered buffers out of the KFD count.

amdgpu-dkms counts a hipMalloc buffer that a NIC pins twice in the KFD
allocation limit (ROCm/mori#721, known-issues Issue 4). Once a KV cache larger
than the VRAM left after allocating it is registered, every later device
allocation of the process fails (OOM, HIP error 209, a segfault in
``HIPStream::stream()``). Buffers from this allocator are amdgpu GEM BOs
imported into HIP, which are counted once on every driver, and mori registers
them through their own dma-buf.

Scope it to the buffers that get registered, e.g. a KV pool::

    import torch
    from mori.io import hip_gem_mem_pool

    with hip_gem_mem_pool() as pool:        # keep `pool` alive while tensors live
        kv_cache = torch.empty(nbytes, dtype=torch.uint8, device="cuda:0")
    mem = engine.register_torch_tensor(kv_cache)

Needs RDMA NICs that accept dma-buf MRs (each buffer is pinned with a one-page
dma-buf MR when it is allocated; ``MORI_HIP_GEM_PIN_DEVICE`` picks the NIC).
HIP has no IPC handle for these buffers, so they can only move through RDMA.
"""

from __future__ import annotations

import contextlib
import ctypes
import os

_MALLOC_SYMBOL = "mori_hip_gem_malloc"
_FREE_SYMBOL = "mori_hip_gem_free"

_allocator = None
_lib = None


def _mori_application_lib_path() -> str:
    """Locate the libmori_application.so that mori itself loads.

    The allocator's buffer registry lives in this library, so torch must load the
    same file that mori's RDMA registration uses.
    """
    import mori

    pkg_dir = os.path.dirname(os.path.abspath(mori.__file__))
    candidates = [
        os.path.join(pkg_dir, "libmori_application.so"),
        os.path.join(pkg_dir, "..", "libmori_application.so"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        f"libmori_application.so not found next to the mori package (searched {candidates})."
    )


def _library():
    global _lib
    if _lib is None:
        import mori.cpp  # noqa: F401  (loads libmori_application.so the way mori does)

        _lib = ctypes.CDLL(_mori_application_lib_path(), mode=os.RTLD_NOW)
        _lib.mori_hip_gem_lookup.restype = ctypes.c_int
        _lib.mori_hip_gem_lookup.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
    return _lib


def hip_gem_torch_allocator():
    """Return (cached) a torch ``CUDAPluggableAllocator`` backed by amdgpu GEM BOs."""
    global _allocator
    if _allocator is None:
        import torch

        _library()
        _allocator = torch.cuda.memory.CUDAPluggableAllocator(
            _mori_application_lib_path(), _MALLOC_SYMBOL, _FREE_SYMBOL
        )
    return _allocator


def _raw_allocator():
    """Underlying torch._C._cuda_CUDAAllocator handle (for MemPool)."""
    alloc = hip_gem_torch_allocator()
    if hasattr(alloc, "allocator"):
        return alloc.allocator()
    return alloc._allocator  # older torch


def make_hip_gem_mem_pool(device=None):
    """Create a ``torch.cuda.MemPool`` backed by the HIP GEM allocator (torch>=2.5).

    A pool serves the device it is created on (``device``, default the current one);
    allocations on another device inside ``use_mem_pool`` bypass it. Keep the returned
    pool alive for as long as the tensors allocated from it are in use (the pool owns
    the underlying memory).
    """
    import torch

    with torch.cuda.device(torch.cuda.current_device() if device is None else device):
        return torch.cuda.MemPool(_raw_allocator())


@contextlib.contextmanager
def hip_gem_mem_pool(device=None):
    """Context manager: CUDA allocations on ``device`` (default the current one) inside
    the block come from amdgpu GEM BOs; other allocations keep torch's default allocator.

    Yields the ``MemPool``; hold a reference to it (and the tensors) while in use.
    """
    import torch

    device = torch.cuda.current_device() if device is None else device
    pool = make_hip_gem_mem_pool(device)
    with torch.cuda.use_mem_pool(pool, device=device):
        yield pool


def hip_gem_lookup(ptr: int):
    """``(base, size, device)`` of the HIP GEM allocator buffer containing ``ptr``, or None."""
    base = ctypes.c_void_p()
    size = ctypes.c_size_t()
    device = ctypes.c_int()
    rc = _library().mori_hip_gem_lookup(
        ctypes.c_void_p(ptr),
        ctypes.byref(base),
        ctypes.byref(size),
        None,
        ctypes.byref(device),
    )
    if rc != 0:
        return None
    return base.value, size.value, device.value


class HipGemAllocator:
    """Allocator handle in the shape of Mooncake's ``HipGemAllocator`` and SGLang's
    custom memory pools: ``HipGemAllocator.get_allocator(device)`` returns the
    ``CUDAPluggableAllocator`` (one serves every device)."""

    @classmethod
    def get_allocator(cls, device=None):
        return hip_gem_torch_allocator()
