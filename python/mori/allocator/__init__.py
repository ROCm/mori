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

    symm_mem.set_backend("MORI")   # importing mori.allocator registers it

    t   = symm_mem.empty(1024, dtype=torch.bfloat16, device=device)
    hdl = symm_mem.rendezvous(t, group_name)
    peer = hdl.get_buffer(1, (1024,), torch.bfloat16)

Peers are exposed the way torch's model expects, as the ``buffer_ptrs`` /
``buffer_ptrs_dev`` array -- one base address per rank, same as every other backend.

Internally the ranks are mapped into one flat span, so those pointers happen to be evenly
strided (``buffer_ptrs[r] == buffer_ptrs[0] + r*stride``). That is deliberately not public
yet: exposing it belongs with mori's cco window, whose ``ccoWindowDevice`` already defines
the layout (``winBase``, 4 GiB-quantised ``stride4G``, LSA-rank indexing). See ROCm/mori#557
for the staged plan.

The handle type is probed per device: fabric where supported, POSIX fd otherwise. gfx9
(MI300/MI355) has no fabric support -- ``hipMemCreate`` itself reports "operation not
supported" -- so those fall back to fd, which needs no configuration.

``barrier``/``put_signal``/``wait_signal`` are not implemented and raise, so no signal
pad is reserved in the window -- torch's 9216-byte pad would cost a whole extra 2 MiB page
on a page-aligned allocation. Build with ``MORI_SYMM_SIGNAL_PAD=ON`` to reserve it, and
check ``signal_pad_supported()`` at run time. torch's own ``symm_mem`` collectives
synchronise through the pad, so they need that build; ``dist.barrier()`` is the stand-in
without it.

Known gap: releasing a rendezvous'd window segfaults at world_size >= 4 (2 ranks are
fine), somewhere in the unmap/release path. Teardown is therefore disabled by default and
the mappings are leaked -- symmetric buffers are few and long-lived, so that is much less
harmful than crashing. Set ``MORI_SYMM_TEARDOWN=1`` to re-enable it while debugging.
"""

import atexit
import logging
from typing import Literal

__all__ = [
    "SYMM_BACKEND_NAME",
    "handle_type",
    "register_symm_backend",
    "signal_pad_supported",
]

logger = logging.getLogger(__name__)

SYMM_BACKEND_NAME = "MORI"

_atexit_registered = False


def _ext():
    # torch must be imported first: the extension links libtorch, and nothing on its
    # RUNPATH resolves those, so they have to already be in the process.
    try:
        import torch
    except ImportError as exc:
        raise ImportError("mori.allocator requires torch") from exc
    try:
        from .. import mori_torch_symm
    except ImportError as exc:  # pragma: no cover - depends on build flags
        raise ImportError(
            "mori_torch_symm extension not found. It is built by torch's cpp_extension, "
            "so torch must be installed when mori is built; reinstall mori with torch "
            f"present. ({exc})"
        ) from exc
    return mori_torch_symm


def register_symm_backend() -> str:
    """Register the backend with torch and return its name. Idempotent.

    Importing this module already does this, so calling it is only needed to force
    a clear error when the extension is missing. Registering makes ``"MORI"``
    *selectable*; ``symm_mem.set_backend("MORI")`` is what makes it active, and that
    stays an explicit choice because other backends may also be available.
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


def handle_type(device_index: int = 0) -> Literal["fabric", "posix_fd"]:
    """Which shareable handle type this device can export."""
    return _ext().handle_type(device_index)


def signal_pad_supported() -> bool:
    """Whether windows carry torch's signal pad.

    False unless built with ``MORI_SYMM_SIGNAL_PAD=ON``. Without the pad, torch's own
    ``symm_mem`` collectives raise -- their kernels synchronise through it -- and
    ``dist.barrier()`` is the stand-in. With it they work, since they use the pad
    directly; this backend's ``barrier``/``put_signal``/``wait_signal`` raise either way.
    """
    return bool(_ext().signal_pad_supported)


def _register_on_import() -> None:
    """Make "MORI" selectable as soon as the module is imported.

    Best effort: if torch or the extension is missing there is nothing to register,
    and the first real call raises with a useful message instead of turning
    ``import mori.allocator`` into an error.
    """
    try:
        register_symm_backend()
    except (
        ImportError,
        RuntimeError,
    ) as exc:  # pragma: no cover - build/runtime dependent
        logger.debug("%s backend not registered: %s", SYMM_BACKEND_NAME, exc)


_register_on_import()
