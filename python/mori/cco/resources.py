# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mori.cco.communicator import Communicator

import mori.cco.cco as _cco


__all__ = [
    "CCOResource",
    "AllocatedMemory",
    "RegisteredWindow",
    "AllocatedWindow",
    "DevCommHandle",
]


# ── Abstract base ─────────────────────────────────────────────────────────────

class CCOResource(ABC):
    """Abstract base for CCO communicator-owned resources.

    Subclasses implement ``_deallocate()``; callers use ``close()`` which
    guarantees idempotency.
    """

    def __init__(self, comm: Communicator) -> None:
        self._comm = comm
        self._closed = False

    @abstractmethod
    def _deallocate(self) -> None: ...

    def close(self) -> None:
        """Release the resource.  Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        self._deallocate()

    def _check_valid(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} has already been closed")

    @property
    def is_valid(self) -> bool:
        """True while the resource has not been closed."""
        return not self._closed


# ── AllocatedMemory ───────────────────────────────────────────────────────────

class AllocatedMemory(CCOResource):
    """Symmetric GPU memory allocated via ``ccoMemAlloc``.

    The allocation lives in the flat VA space; after registering as a window
    it becomes P2P-accessible to all LSA peers.

    Example::

        mem = comm.alloc_mem(4096)
        win = comm.register_window(mem.ptr, mem.size)
        # ... use win ...
        win.close()
        mem.close()
    """

    def __init__(self, comm: Communicator, size: int) -> None:
        super().__init__(comm)
        self._size = size
        self._ptr = _cco.mem_alloc(comm._raw, size)

    def _deallocate(self) -> None:
        if self._ptr:
            _cco.mem_free(self._comm._raw, self._ptr)
            self._ptr = 0

    @property
    def ptr(self) -> int:
        """Device pointer (intptr_t) to the allocated symmetric memory."""
        self._check_valid()
        return self._ptr

    @property
    def size(self) -> int:
        """Allocation size in bytes."""
        return self._size

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<AllocatedMemory: closed>"
        return f"<AllocatedMemory: ptr={self._ptr:#x}, size={self._size}>"


# ── RegisteredWindow ──────────────────────────────────────────────────────────

class RegisteredWindow(CCOResource):
    """Window registered from a caller-allocated ``ccoMemAlloc`` pointer.

    Corresponds to ``ccoWindowRegister`` overload B.  The caller is
    responsible for the lifetime of the underlying memory.  Deregistration
    is local (not collective).

    Example::

        mem = comm.alloc_mem(size)
        win = comm.register_window(mem.ptr, mem.size)
        # kernel uses win.handle
        win.close()
        mem.close()
    """

    def __init__(self, comm: Communicator, ptr: int, size: int) -> None:
        super().__init__(comm)
        self._ptr = ptr
        self._size = size
        self._handle = _cco.window_register_ptr(comm._raw, ptr, size)

    def _deallocate(self) -> None:
        if self._handle:
            _cco.window_deregister(self._comm._raw, self._handle)
            self._handle = 0

    @property
    def handle(self) -> int:
        """intptr_t window handle (``ccoWindow_t``) for kernel arguments."""
        self._check_valid()
        return self._handle

    @property
    def local_ptr(self) -> int:
        """The device pointer that was registered."""
        return self._ptr

    @property
    def size(self) -> int:
        """Window size in bytes."""
        return self._size

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<RegisteredWindow: closed>"
        return f"<RegisteredWindow: handle={self._handle:#x}, size={self._size}>"


# ── AllocatedWindow ───────────────────────────────────────────────────────────

class AllocatedWindow(CCOResource):
    """Window where CCO allocates and registers memory internally.

    Corresponds to ``ccoWindowRegister`` overload A.  CCO owns the backing
    allocation; ``local_ptr`` is the host/device-accessible pointer to this
    rank's slot.  Deregistering frees the allocation.

    This is the simplest path: one call creates both the allocation and the
    window.

    Example::

        win = comm.alloc_window(size)
        # write to win.local_ptr on host, or pass win.handle to a kernel
        win.close()
    """

    def __init__(self, comm: Communicator, size: int) -> None:
        super().__init__(comm)
        self._size = size
        self._handle, self._local_ptr = _cco.window_register(comm._raw, size)

    def _deallocate(self) -> None:
        if self._handle:
            _cco.window_deregister(self._comm._raw, self._handle)
            self._handle = 0
            self._local_ptr = 0

    @property
    def handle(self) -> int:
        """intptr_t window handle (``ccoWindow_t``) for kernel arguments."""
        self._check_valid()
        return self._handle

    @property
    def local_ptr(self) -> int:
        """intptr_t pointer to this rank's slot in the CCO flat VA."""
        self._check_valid()
        return self._local_ptr

    @property
    def size(self) -> int:
        """Window size in bytes."""
        return self._size

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<AllocatedWindow: closed>"
        return (f"<AllocatedWindow: handle={self._handle:#x}, "
                f"local_ptr={self._local_ptr:#x}, size={self._size}>")


# ── DevCommHandle ─────────────────────────────────────────────────────────────

class DevCommHandle(CCOResource):
    """Device communicator resource wrapping ``ccoDevComm``.

    Created by :py:meth:`Communicator.create_dev_comm`.  The underlying
    ``ccoDevComm`` struct is embedded inside the Cython ``DevComm`` object;
    ``ptr`` gives the address of that struct for passing to GPU kernels
    (either as a pointer argument or for ``launch_struct``).

    Example::

        reqs = CcoDevCommRequirements()
        reqs.lsa_barrier_count = 2
        dc = comm.create_dev_comm(reqs)
        kernel.launch(..., dc.ptr)
        dc.close()
    """

    def __init__(
        self,
        comm: Communicator,
        requirements: _cco.DevCommRequirements | None = None,
    ) -> None:
        super().__init__(comm)
        if requirements is None:
            requirements = _cco.DevCommRequirements()
        self._reqs = requirements
        self._dev_comm = _cco.dev_comm_create(comm._raw, requirements)

    def _deallocate(self) -> None:
        if self._dev_comm is not None:
            _cco.dev_comm_destroy(self._comm._raw, self._dev_comm)
            self._dev_comm = None

    @property
    def ptr(self) -> int:
        """intptr_t address of the embedded ``ccoDevComm`` struct.

        Pass this to GPU kernels that take ``ccoDevComm*``.  The struct is
        valid for the lifetime of this handle.
        """
        self._check_valid()
        return self._dev_comm.ptr

    @property
    def rank(self) -> int:
        """This rank's world rank from the device communicator."""
        self._check_valid()
        return self._dev_comm.rank

    @property
    def world_size(self) -> int:
        """World size from the device communicator."""
        self._check_valid()
        return self._dev_comm.world_size

    @property
    def lsa_size(self) -> int:
        """Number of ranks on this node (LSA team size)."""
        self._check_valid()
        return self._dev_comm.lsa_size

    @property
    def lsa_rank(self) -> int:
        """This rank's index within the intra-node LSA team."""
        self._check_valid()
        return self._dev_comm.lsa_rank

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<DevCommHandle: closed>"
        return (f"<DevCommHandle: ptr={self.ptr:#x}, rank={self.rank}, "
                f"world_size={self.world_size}, lsa_size={self.lsa_size}>")
