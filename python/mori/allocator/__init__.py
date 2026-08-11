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
"""Register mori as a torch SymmetricMemory backend.

Self-contained: plain HIP VMM, no shmem or cco allocator involved, so no mori bootstrap
is needed -- torch's process group is the only rendezvous::

    import torch.distributed._symmetric_memory as symm_mem
    from mori.allocator import register_symm_backend

    register_symm_backend()
    symm_mem.set_backend("MORI")

    t   = symm_mem.empty(1024, dtype=torch.bfloat16, device=device)
    hdl = symm_mem.rendezvous(t, group_name)
    peer = hdl.get_buffer(1, (1024,), torch.bfloat16)

Every rank is mapped into one flat span, so ``hdl.get_buffer_ptrs()[r]`` is
``flat_base + r*stride``. ``flat_layout(t)`` returns that pair for kernels that would
rather do arithmetic than index a pointer array.

The handle type is probed per device: fabric where supported, POSIX fd otherwise. gfx9
(MI300/MI355) has no fabric support -- ``hipMemCreate`` itself reports "operation not
supported" -- so those fall back to fd, which needs no configuration.

Known gap: releasing a rendezvous'd window segfaults at world_size >= 4 (2 ranks are
fine), somewhere in the unmap/release path. Teardown is therefore disabled by default and
the mappings are leaked -- symmetric buffers are few and long-lived, so that is much less
harmful than crashing. Set ``MORI_SYMM_TEARDOWN=1`` to re-enable it while debugging.
"""

import atexit
from typing import Literal

__all__ = [
    "SYMM_BACKEND_NAME",
    "flat_layout",
    "handle_type",
    "register_symm_backend",
]

SYMM_BACKEND_NAME = "MORI"

_atexit_registered = False


def _ext():
    try:
        from .. import mori_torch_symm
    except ImportError as exc:  # pragma: no cover - depends on build flags
        raise ImportError(
            "mori_torch_symm extension not found. Rebuild mori with BUILD_TORCH_SYMM=ON "
            "(requires torch at build time)."
        ) from exc
    return mori_torch_symm


def register_symm_backend() -> str:
    """Register the backend with torch and return its name. Idempotent.

    After this, ``symm_mem.set_backend("MORI")`` routes ``symm_mem.empty`` and
    ``symm_mem.rendezvous`` into mori.
    """
    ext = _ext()
    ext.register_backend()
    global _atexit_registered
    if not _atexit_registered:
        # Allocations still live at interpreter shutdown are torn down too late to touch
        # HIP safely, so release them here instead.
        atexit.register(ext.shutdown)
        _atexit_registered = True
    return SYMM_BACKEND_NAME


def flat_layout(tensor) -> tuple[int, int]:
    """``(flat_base, stride)`` of a rendezvous'd tensor: peer r lives at
    ``flat_base + r*stride``."""
    return _ext().flat_layout(tensor)


def handle_type(device_index: int = 0) -> Literal["fabric", "posix_fd"]:
    """Which shareable handle type this device can export."""
    return _ext().handle_type(device_index)
