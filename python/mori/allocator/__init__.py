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
"""Torch integration for the mori symmetric heap.

**SymmetricMemory backend** -- registers mori with torch as ``"MORI"``, so torch's own
entry points drive it and ``torch.ops.symm_mem.*`` runs on mori memory::

    mori.shmem.shmem_torch_process_group_init(group_name)
    register_symm_backend()
    symm_mem.set_backend("MORI")

    t   = symm_mem.empty(1024, dtype=torch.bfloat16, device=device)
    hdl = symm_mem.rendezvous(t, group_name)

``hdl.get_buffer_ptrs()[pe]`` is null for a PE reached over RDMA rather than P2P, and
``world_within_direct_access()`` aggregates that -- the scale-up vs scale-out
distinction, straight from ``ShmemPtrP2p``.

**MemPool allocator** -- ``MoriAllocator`` backs a ``torch.cuda.MemPool``, so tensors you
do not allocate by hand (a KV cache, a GEMM output) also come from the symmetric heap::

    pool = torch.cuda.MemPool(MoriAllocator.get_allocator(device).allocator())
    with torch.cuda.use_mem_pool(pool):
        kv_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)

Both paths allocate through ``ShmemMalloc``, which is collective: every rank must
allocate the same sizes in the same order, or the heap will hang or corrupt.
"""

import logging
import os
import threading
from typing import ClassVar, Final

from torch import device as torch_device
from torch.cuda.memory import CUDAPluggableAllocator

logger = logging.getLogger(__name__)

__all__ = [
    "SYMM_BACKEND_NAME",
    "MoriAllocator",
    "get_so_path",
    "is_available",
    "register_symm_backend",
]

SYMM_BACKEND_NAME = "MORI"

_SO_NAME = "libmori_torch_allocator.so"
_ALLOC_FN = "mori_allocator_malloc"
_FREE_FN = "mori_allocator_free"
_PROBE_FN = "mori_allocator_probe"


def get_so_path() -> str:
    """Absolute path to the allocator shared library shipped with the package."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    so_path = os.path.join(here, _SO_NAME)
    if not os.path.exists(so_path):
        raise FileNotFoundError(
            f"{_SO_NAME} not found at {so_path}. Build mori with BUILD_ALLOCATOR=ON "
            "(the default when BUILD_SHMEM=ON)."
        )
    return so_path


def is_available() -> bool:
    """True if the symmetric heap can serve allocations now.

    Allocation happens inside tensor constructors where raising is awkward; check here
    and fall back to the default pool instead.
    """
    import ctypes

    try:
        lib = ctypes.CDLL(get_so_path())
    except (OSError, FileNotFoundError) as exc:
        logger.debug("mori torch allocator unavailable: %s", exc)
        return False

    probe = getattr(lib, _PROBE_FN, None)
    if probe is None:
        return False
    probe.restype = ctypes.c_int
    return probe() == 1


class MoriAllocator:
    """``CUDAPluggableAllocator`` over the mori symmetric heap, one per device.

    Matches the shape inference engines already expect, so it drops in wherever a custom
    memory pool is selected.
    """

    _instances: ClassVar[dict[torch_device, CUDAPluggableAllocator]] = {}
    _lock: Final = threading.Lock()

    @classmethod
    def get_allocator(
        cls, device: torch_device | None = None
    ) -> CUDAPluggableAllocator:
        """Return (and cache) the allocator for ``device``."""
        key = torch_device(device) if device is not None else torch_device("cuda")
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = CUDAPluggableAllocator(
                    get_so_path(), _ALLOC_FN, _FREE_FN
                )
            return cls._instances[key]


def register_symm_backend() -> str:
    """Register the backend with torch and return its name. Idempotent.

    Requires the extension (``BUILD_TORCH_SYMM=ON``, default when torch is importable at
    build time) and shmem to be initialised before the first allocation.
    """
    try:
        from .. import mori_torch_symm
    except ImportError as exc:  # pragma: no cover - depends on build flags
        raise ImportError(
            "mori_torch_symm extension not found. Rebuild mori with BUILD_TORCH_SYMM=ON "
            "(requires torch at build time)."
        ) from exc

    mori_torch_symm.register_backend()
    return SYMM_BACKEND_NAME
