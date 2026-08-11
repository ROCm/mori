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
"""Back a ``torch.cuda.MemPool`` with the mori symmetric heap.

Tensors allocated inside the pool's context come from ``ShmemMalloc`` instead of the
caching allocator, so they are symmetric and directly reachable by peers -- including
tensors an engine never allocates by hand, such as a KV cache or a GEMM output.

    import torch
    import mori
    from mori.allocator import MoriAllocator

    mori.shmem.shmem_torch_process_group_init("default")

    pool = torch.cuda.MemPool(MoriAllocator.get_allocator(device).allocator())
    with torch.cuda.use_mem_pool(pool):
        kv_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)

``ShmemMalloc`` is collective: every rank must enter and leave the pool context in the
same order and allocate the same sizes, exactly as for ``shmem_malloc``. Allocations that
are not symmetric across ranks will hang or corrupt the heap.
"""

import logging
import os
import threading
from typing import ClassVar, Final

from torch import device as torch_device
from torch.cuda.memory import CUDAPluggableAllocator

logger = logging.getLogger(__name__)

__all__ = ["MoriAllocator", "get_so_path", "is_available"]

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
    """True if the symmetric heap is initialised and can serve allocations now.

    Allocation happens inside tensor constructors where raising is awkward, so callers
    are encouraged to check here and fall back to the default pool instead.
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

    Mirrors the shape inference engines already expect from a pluggable allocator, so it
    can be dropped in wherever a custom memory pool is selected.
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
