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
"""Torch pluggable allocator backed by fabric-exportable VMM memory.

Plain torch tensors (hipMalloc / default caching allocator) cannot be exported
over a UALink super-node fabric, so the mori-io FabricBackend cannot register
them. This module wires torch's ``CUDAPluggableAllocator`` to mori-io's
``FabricMalloc`` / ``FabricFree`` (VMM allocations created with the fabric handle
type), so tensors allocated under it become fabric-exportable and can be
registered + transferred by the FabricBackend.

Recommended usage (scope fabric memory to just the KV pool, keep the rest of
torch on the fast default allocator):

    import torch
    from mori.io import fabric_mem_pool

    with fabric_mem_pool() as pool:          # keep `pool` alive while tensors live
        kv_cache = torch.empty(nbytes, dtype=torch.uint8, device="cuda:0")
    mem = engine.register_torch_tensor(kv_cache)   # now fabric-exportable

For a dedicated process you may instead switch the process-global allocator
(must be called before any CUDA allocation): ``use_fabric_torch_allocator()``.
"""

from __future__ import annotations

import contextlib
import os

_MALLOC_SYMBOL = "mori_io_fabric_malloc"
_FREE_SYMBOL = "mori_io_fabric_free"

_allocator = None


def _mori_io_lib_path() -> str:
    """Locate libmori_io.so shipped next to the mori package."""
    import mori

    pkg_dir = os.path.dirname(os.path.abspath(mori.__file__))
    candidates = [
        os.path.join(pkg_dir, "libmori_io.so"),
        os.path.join(pkg_dir, "..", "libmori_io.so"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        f"libmori_io.so not found next to the mori package (searched {candidates}). "
        "The fabric torch allocator requires the compiled mori-io shared library."
    )


def fabric_torch_allocator():
    """Return (cached) a torch ``CUDAPluggableAllocator`` backed by fabric memory."""
    global _allocator
    if _allocator is None:
        import torch

        _allocator = torch.cuda.memory.CUDAPluggableAllocator(
            _mori_io_lib_path(), _MALLOC_SYMBOL, _FREE_SYMBOL
        )
    return _allocator


def _raw_allocator():
    """Underlying torch._C._cuda_CUDAAllocator handle (for MemPool)."""
    alloc = fabric_torch_allocator()
    if hasattr(alloc, "allocator"):
        return alloc.allocator()
    return alloc._allocator  # older torch


def use_fabric_torch_allocator() -> None:
    """Switch torch's *process-global* CUDA allocator to the fabric allocator.

    Must be called before any CUDA allocation. Prefer :func:`fabric_mem_pool` to
    scope fabric memory to the KV pool only.
    """
    import torch

    torch.cuda.memory.change_current_allocator(fabric_torch_allocator())


def make_fabric_mem_pool():
    """Create a ``torch.cuda.MemPool`` backed by the fabric allocator (torch>=2.5).

    Keep the returned pool alive for as long as the tensors allocated from it are
    in use (the pool owns the underlying fabric memory).
    """
    import torch

    return torch.cuda.MemPool(_raw_allocator())


@contextlib.contextmanager
def fabric_mem_pool():
    """Context manager: CUDA allocations inside the block use fabric-exportable
    memory; allocations outside keep torch's default (fast) allocator.

    Yields the ``MemPool``; hold a reference to it (and the tensors) while in use.
    """
    import torch

    pool = make_fabric_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        yield pool
