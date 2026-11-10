"""Torch MemPool backed by UMBP's independently exportable HIP VMM allocations."""

from __future__ import annotations

import threading
from pathlib import Path

from mori.cpp import _load_umbp

_lock = threading.Lock()
_pools = {}


def _binding():
    return _load_umbp().engine


def _library_path() -> str:
    module = _load_umbp()
    return str(Path(module.__file__).resolve())


def kv_mem_pool(device: int | str):
    """Return the one no-split UMBP MemPool for this process/device.

    The pool and its CUDAPluggableAllocator are retained together for the life
    of the process. No HIP allocation is made until torch allocates in the pool.
    """
    import torch

    dev = torch.device(device)
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    with _lock:
        cached = _pools.get(index)
        if cached is None:
            allocator = torch.cuda.memory.CUDAPluggableAllocator(
                _library_path(), "mori_umbp_kv_malloc", "mori_umbp_kv_free"
            )
            raw = allocator.allocator() if hasattr(allocator, "allocator") else allocator._allocator
            with torch.cuda.device(index):
                pool = torch.cuda.MemPool(raw, no_split=True)
            cached = (allocator, pool)
            _pools[index] = cached
        return cached[1]


def allocator_stats() -> dict:
    return dict(_binding().allocator_stats())


def lookup_allocation(ptr: int, length: int = 1):
    return _binding().lookup_allocation(ptr, length)


def export_allocation(ptr: int):
    """Export the allocation containing ``ptr``; caller closes a returned fd."""
    return _binding().export_allocation(ptr)
