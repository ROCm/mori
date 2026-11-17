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
"""Register mori as a torch SymmetricMemory backend. Requires **torch >= 2.9**.

torch owns the allocation and its lifetime; CCO owns the peer addressing. ``alloc`` is
plain HIP VMM, and ``rendezvous`` hands that allocation to ``ccoWindowRegister``, which
aliases it into CCO's flat LSA space -- so ``buffer_ptrs`` are ``ccoGetPeerPtr`` results
and agree with the device-side ``Window.lsa_ptr`` by construction. CCO's communicator is
created on the first rendezvous of a process group and its unique id travels over torch's
own Store, so there is no second bootstrap channel to configure::

    import torch.distributed._symmetric_memory as symm_mem
    from mori.allocator import register_symm_backend

    symm_mem.set_backend("MORI")   # importing mori.allocator registers it

    t   = symm_mem.empty(1024, dtype=torch.bfloat16, device=device)
    hdl = symm_mem.rendezvous(t, group_name)
    peer = hdl.get_buffer(1, (1024,), torch.bfloat16)

``enable_symm_mem_for_group()`` is not needed: the group is resolved with
``resolve_process_group()``, the way torch's own CUDA and NCCL backends do it, rather
than through the ``GroupInfo`` registry that only that deprecated call populates.

Known constraint on the POSIX-fd path (gfx9, i.e. MI300/MI355): CCO names its rendezvous
socket after the local slot offset, and freeing a window returns that slot to a per-rank
allocator. Ranks that free symmetric tensors in *different* orders therefore get different
offsets, and a later ``rendezvous()`` fails with a P2P fd exchange timeout. Free order is
unconstrained as far as torch is concerned, so keep it uniform across ranks until that
naming changes. The fabric path is unaffected -- peer addressing is computed from each
rank's own offset.

Peers are exposed the way torch's model expects, as the ``buffer_ptrs`` /
``buffer_ptrs_dev`` array -- one base address per rank, same as every other backend.

Internally the ranks are mapped into one flat span, so those pointers happen to be evenly
strided (``buffer_ptrs[r] == buffer_ptrs[0] + r*stride``). That is deliberately not public
yet: exposing it belongs with mori's cco window, whose ``ccoWindowDevice`` already defines
the layout (``winBase``, 4 GiB-quantised ``stride4G``, LSA-rank indexing). See ROCm/mori#557
for the staged plan.

The handle type is probed per device: fabric where supported, POSIX fd otherwise. gfx9
(MI300/MI355) has no fabric support -- ``hipMemCreate`` itself reports "operation not
supported" -- so those fall back to fd, which needs no configuration. The allocation and
CCO's own export have to agree on that type, since CCO re-exports what it imports.

``MORI_SYMM_PER_RANK_VMM`` sizes the communicator's flat VA reservation (default 4 GiB).
It is a floor rather than a budget -- CCO quantises it up to 4 GiB -- and it bounds live
symmetric memory per rank, not allocations over time, because slots are recycled on free.

The signal pad is always there, as its own CCO allocation and window rather than a tail
appended to the buffer -- the same split NCCL's backend uses. torch's own ``symm_mem``
collectives synchronise through it, so they work without any build flag.

``barrier()`` runs over CCO's host barrier, which means the current stream is drained
first: stronger than the device-side barrier torch asks for, and heavier.
``put_signal``/``wait_signal`` still raise -- point-to-point signalling needs a
``ccoDevComm``, which this backend does not create yet.

``rendezvous()`` takes the tensor's *storage* base, so ``get_offset()`` is 0 by
construction: every ``alloc()`` is its own VMM allocation. Handing it a view
(``rendezvous(t[512:])``) therefore describes the whole storage, and ``get_remote_tensor``
answers for the storage base -- pass ``storage_offset`` to ``get_buffer`` to address a
slice. That is torch's model, not something this backend narrows.

The extension is optional and its ABI tracks the installed torch, so it is a plugin:
``pip install`` prebuilds it when it can (``BUILD_TORCH_SYMM=AUTO``, the default; ``ON``
makes a failure fatal, ``OFF`` skips it). A prebuilt module made for a different torch
build is refused before it can register, and the backend is then compiled on first import
from the sources shipped in ``_jit-sources`` -- against torch's own ROCm, into a cache
directory per torch build. So switching torch does not require reinstalling mori.
``MORI_SYMM_FORCE_JIT=1`` ignores the prebuilt module. If neither works, importing
``mori.allocator`` logs a warning and ``set_backend("MORI")`` reports the backend as
unknown.
"""

import atexit
import logging
import os
from pathlib import Path
from typing import Literal

__all__ = [
    "SYMM_BACKEND_NAME",
    "handle_type",
    "register_symm_backend",
    "signal_pad_supported",
    "window_handle",
    "dev_comm",
]

logger = logging.getLogger(__name__)

SYMM_BACKEND_NAME = "MORI"

# torch 2.9 is where symm_mem.set_backend() and the SymmetricMemory interface this
# implements arrived; on 2.8 there is no pluggable backend to register with at all.
TORCH_MIN_VERSION = (2, 9)

_atexit_registered = False


_ext_cache = None


def _jit_ext(fingerprint: str):
    """Compile the backend against the torch that is actually loaded.

    Built in a directory of its own per torch build, so processes on different torch
    builds sharing TORCH_EXTENSIONS_DIR neither reuse each other's module nor keep
    rebuilding over it."""
    import torch
    from torch.utils import cpp_extension

    from ..jit.config import get_mori_source_root
    from ._build import hip_runtime_library, rocm_home

    # Checked before ninja, not after: on torch < 2.9 the SymmetricMemory interface this
    # implements does not exist, so the compile is a guaranteed minute-long failure whose
    # error says nothing about the version.
    have = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    if have < TORCH_MIN_VERSION:
        raise ImportError(
            f"mori's torch SymmetricMemory backend needs torch >= "
            f"{'.'.join(map(str, TORCH_MIN_VERSION))}, found {torch.__version__}. "
            "torch.distributed._symmetric_memory.set_backend() does not exist before "
            "then, so there is nothing to register the backend with."
        )

    root = get_mori_source_root()
    if root is None:
        raise ImportError(
            "mori_torch_symm is not built and its sources were not found. Reinstall "
            "mori from a source tree or a wheel that ships _jit-sources."
        )
    source = root / "src" / "allocator" / "symm_backend.cpp"
    if not source.is_file():
        raise ImportError(f"mori_torch_symm sources missing at {source}")

    rocm = rocm_home(cpp_extension)
    # By file: records the SONAME, and pip ROCm has no libamdhip64.so for -l.
    hip = hip_runtime_library(rocm)
    hip_link = [hip] if hip else [f"-L{rocm}/lib", "-lamdhip64"]
    # No -std here: load() already passes the standard torch's headers need.
    flags = [
        "-O3",
        "-D__HIP_PLATFORM_AMD__=1",
        "-DUSE_ROCM=1",
        f"-DMORI_TORCH_FINGERPRINT={fingerprint}",
    ]
    name = "mori_torch_symm"
    try:
        build_dir = cpp_extension._get_build_directory(f"{name}_{fingerprint}", False)
    except AttributeError:  # pragma: no cover - private helper, present since torch 1.x
        build_dir = None
    logger.info("compiling %s for torch %s from %s", name, torch.__version__, source)
    # libmori_cco sits in the package directory, wherever mori was installed.
    pkg = str(Path(__file__).resolve().parent.parent)
    return cpp_extension.load(
        name=name,
        sources=[str(source)],
        extra_include_paths=[str(root / "include"), f"{rocm}/include"],
        extra_cflags=flags,
        extra_ldflags=hip_link
        + ["-lc10_hip", f"-L{pkg}", "-lmori_cco", f"-Wl,-rpath,{pkg}"],
        build_directory=build_dir,
    )


def _prebuilt_ext(fingerprint: str):
    """The module ``pip install`` built, if it exists and was built for this torch."""
    import importlib.util

    if (
        importlib.util.find_spec(f"{__package__.rsplit('.', 1)[0]}.mori_torch_symm")
        is None
    ):
        return None, "not built"
    try:
        from .. import mori_torch_symm as ext
    # The .so is there but does not load, e.g. a symbol of another torch is gone.
    except (ImportError, OSError) as exc:  # pragma: no cover - build dependent
        logger.warning("prebuilt mori_torch_symm does not load, rebuilding: %s", exc)
        return None, exc
    built = getattr(ext, "torch_fingerprint", "unknown")
    if built != fingerprint:
        # Loading is harmless (registration is explicit); registering it would not be.
        reason = f"built for another torch build ({built}, running {fingerprint})"
        logger.info("prebuilt mori_torch_symm %s, rebuilding", reason)
        return None, reason
    return ext, None


def _ext():
    # torch must be imported first: the extension links libtorch, and nothing on its
    # RUNPATH resolves those, so they have to already be in the process.
    global _ext_cache
    if _ext_cache is not None:
        return _ext_cache
    try:
        import torch
    except ImportError as exc:
        raise ImportError("mori.allocator requires torch") from exc
    from ._build import torch_fingerprint

    fingerprint = torch_fingerprint(torch)
    prebuilt_error = "ignored (MORI_SYMM_FORCE_JIT)"
    if os.environ.get("MORI_SYMM_FORCE_JIT", "0").strip() not in {"1", "ON", "on"}:
        ext, prebuilt_error = _prebuilt_ext(fingerprint)
        if ext is not None:
            _ext_cache = ext
            return _ext_cache

    try:
        _ext_cache = _jit_ext(fingerprint)
    except Exception as exc:
        raise ImportError(
            f"mori_torch_symm: prebuilt module {prebuilt_error}, and building it for "
            f"torch {torch.__version__} failed ({exc})."
        ) from exc
    return _ext_cache


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

    Always true now: the pad is its own CCO window. Kept because callers written against
    the earlier build-flag behaviour still check it.
    """
    return bool(_ext().signal_pad_supported)


def window_handle(tensor, signal_pad: bool = False) -> int:
    """The cco window backing a rendezvous'd tensor, as an integer handle.

    Backend-specific on purpose: torch's ``SymmetricMemory`` has nowhere to carry a
    window, so NCCL's backend exposes ``get_window()`` on its own class. Python only ever
    sees the base class, so this is a lookup keyed by the tensor instead.

    Takes the tensor rather than a pointer because the window belongs to the *storage*:
    ``rendezvous(t[512:])`` registers the whole storage, so a view has to resolve to the
    same window rather than to "pointer is not a mori allocation".

    Pass the result to a kernel that addresses peers with cco's device API
    (``mori.ir.triton.cco.Window.lsa_ptr``) instead of walking ``buffer_ptrs``.
    """
    return _ext().window_handle(tensor.untyped_storage().data_ptr(), signal_pad)


def dev_comm(group_name: str, key: str = "default", lsa_barrier_count: int = 1) -> int:
    """Device ``ccoDevComm`` pointer for a group, as a kernel argument.

    Created on first use and cached per ``(group_name, key)``. The key exists because a
    DevComm is parameterised by signal counts, QP counts and connection type -- properties
    of the algorithm about to run, not of the group -- so two collectives over one group
    want two of them. torch's NCCL backend keys the same cache on ``__builtin_FUNCTION()``;
    from Python that would be stack inspection, so pass a key explicitly instead.

    Rendezvous a tensor with the group first: that is what creates the communicator this
    hangs off. Intra-node only today -- the connection type is fixed to NONE.
    """
    return _ext().dev_comm(group_name, key, lsa_barrier_count)


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
        logger.warning(
            "%s symmetric memory backend not registered: %s", SYMM_BACKEND_NAME, exc
        )


_register_on_import()
