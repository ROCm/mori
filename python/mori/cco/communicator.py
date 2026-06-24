# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License

"""CCO high-level Communicator and resource classes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import mori.cco.cco as _cco


__all__ = [
    "UniqueId",
    "get_unique_id",
    "CCODevCommRequirements",
    "CCOResource",
    "AllocatedMemory",
    "RegisteredWindow",
    "AllocatedWindow",
    "DevCommHandle",
    "Communicator",
]


# ── UniqueId (pure-Python wrapper with pickle support) ───────────────────────

class UniqueId:
    """CCO unique identifier for communicator initialization.

    Wraps the Cython-level ``cco.UniqueId`` with pickle support so that
    ``mpi4py.MPI.Comm.bcast`` (lowercase ``b``, pickle-based) works
    out of the box.
    """

    def __init__(self, _internal: _cco.UniqueId | None = None) -> None:
        if _internal is None:
            _internal = _cco.UniqueId()
        self._internal: _cco.UniqueId = _internal

    def __bytes__(self) -> bytes:
        return self._internal.tobytes()

    def __getstate__(self) -> bytes:
        return bytes(self)

    def __setstate__(self, state: bytes) -> None:
        self._internal = _cco.UniqueId.frombytes(state)

    @staticmethod
    def from_bytes(b: bytes | bytearray | memoryview) -> UniqueId:
        return UniqueId(_cco.UniqueId.frombytes(bytes(b)))

    @property
    def ptr(self) -> _cco.UniqueId:
        return self._internal

    def __repr__(self) -> str:
        return f"<UniqueId: {bytes(self)[:8].hex()}...>"


def get_unique_id() -> UniqueId:
    """Generate a CCO rendezvous token (call on rank 0, then broadcast)."""
    return UniqueId(_cco.UniqueId.create())


class CCODevCommRequirements(_cco.DevCommRequirements):
    """CCO device communicator requirements.

    Initialised to safe defaults matching
    ``CCO_DEV_COMM_REQUIREMENTS_INITIALIZER``.  Override fields before
    passing to :py:meth:`Communicator.create_dev_comm`.
    """


# ── Resource base ─────────────────────────────────────────────────────────────

class CCOResource(ABC):
    """Abstract base for CCO communicator-owned resources."""

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
        return not self._closed


# ── AllocatedMemory ───────────────────────────────────────────────────────────

class AllocatedMemory(CCOResource):
    """Symmetric GPU memory allocated via ``ccoMemAlloc``."""

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
        self._check_valid()
        return self._ptr

    @property
    def size(self) -> int:
        return self._size

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<AllocatedMemory: closed>"
        return f"<AllocatedMemory: ptr={self._ptr:#x}, size={self._size}>"


# ── RegisteredWindow ──────────────────────────────────────────────────────────

class RegisteredWindow(CCOResource):
    """Window registered from a caller-allocated ``ccoMemAlloc`` pointer."""

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
        self._check_valid()
        return self._handle

    @property
    def local_ptr(self) -> int:
        return self._ptr

    @property
    def size(self) -> int:
        return self._size

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<RegisteredWindow: closed>"
        return f"<RegisteredWindow: handle={self._handle:#x}, size={self._size}>"


# ── AllocatedWindow ───────────────────────────────────────────────────────────

class AllocatedWindow(CCOResource):
    """Window where CCO allocates and registers memory internally."""

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
        self._check_valid()
        return self._handle

    @property
    def local_ptr(self) -> int:
        self._check_valid()
        return self._local_ptr

    @property
    def size(self) -> int:
        return self._size

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<AllocatedWindow: closed>"
        return (f"<AllocatedWindow: handle={self._handle:#x}, "
                f"local_ptr={self._local_ptr:#x}, size={self._size}>")


# ── DevCommHandle ─────────────────────────────────────────────────────────────

class DevCommHandle(CCOResource):
    """Device communicator resource wrapping ``ccoDevComm``."""

    def __init__(
        self,
        comm: Communicator,
        requirements: _cco.DevCommRequirements | None = None,
    ) -> None:
        super().__init__(comm)
        if requirements is None:
            requirements = _cco.DevCommRequirements()
        self._dev_comm = _cco.dev_comm_create(comm._raw, requirements)

    def _deallocate(self) -> None:
        if self._dev_comm is not None:
            _cco.dev_comm_destroy(self._comm._raw, self._dev_comm)
            self._dev_comm = None

    @property
    def ptr(self) -> int:
        self._check_valid()
        return self._dev_comm.ptr

    @property
    def device_ptr(self) -> int:
        self._check_valid()
        return self.dev_comm._device_ptr

    @property
    def rank(self) -> int:
        self._check_valid()
        return self._dev_comm.rank

    @property
    def world_size(self) -> int:
        self._check_valid()
        return self._dev_comm.world_size

    @property
    def lsa_size(self) -> int:
        self._check_valid()
        return self._dev_comm.lsa_size

    @property
    def lsa_rank(self) -> int:
        self._check_valid()
        return self._dev_comm.lsa_rank

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<DevCommHandle: closed>"
        return (f"<DevCommHandle: ptr={self.ptr:#x}, rank={self.rank}, "
                f"world_size={self.world_size}, lsa_size={self.lsa_size}>")


# ── Communicator ──────────────────────────────────────────────────────────────

class Communicator:
    """CCO communicator for intra- and inter-node symmetric memory operations."""

    DEFAULT_PER_RANK_VMM: int = 4 * 1024 * 1024 * 1024

    def __init__(self) -> None:
        self._raw: _cco.Comm | None = None
        self._resources: list[CCOResource] = []
        self._rank: int | None = None
        self._nranks: int | None = None

    @staticmethod
    def get_unique_id() -> UniqueId:
        """Generate a rendezvous token on rank 0."""
        return get_unique_id()

    @classmethod
    def init(
        cls,
        nranks: int,
        rank: int,
        unique_id: UniqueId,
        per_rank_vmm: int = DEFAULT_PER_RANK_VMM,
    ) -> Communicator:
        """Collective: create a CCO communicator."""
        comm = cls()
        uid = unique_id.ptr if isinstance(unique_id, UniqueId) else unique_id
        comm._raw    = _cco.comm_create(uid, nranks, rank, per_rank_vmm)
        comm._rank   = rank
        comm._nranks = nranks
        return comm

    def destroy(self) -> None:
        """Destroy the communicator and free all owned resources."""
        self.close_all_resources()
        if self._raw is not None:
            _cco.comm_destroy(self._raw)
            self._raw = None

    def close_all_resources(self) -> None:
        """Close all tracked resources (best-effort; errors are suppressed)."""
        for r in reversed(self._resources):
            try:
                r.close()
            except Exception:
                pass
        self._resources.clear()

    @property
    def is_valid(self) -> bool:
        return self._raw is not None

    @property
    def rank(self) -> int:
        self._check_valid("get rank")
        return self._rank  # type: ignore[return-value]

    @property
    def nranks(self) -> int:
        self._check_valid("get nranks")
        return self._nranks  # type: ignore[return-value]

    @property
    def ptr(self) -> int:
        self._check_valid("get ptr")
        return self._raw.ptr  # type: ignore[union-attr]

    def _check_valid(self, op: str = "use") -> None:
        if not self.is_valid:
            raise RuntimeError(f"Cannot {op}: Communicator has been destroyed")

    def alloc_mem(self, size: int) -> AllocatedMemory:
        self._check_valid("alloc_mem")
        r = AllocatedMemory(self, size)
        self._resources.append(r)
        return r

    def register_window(self, ptr: int, size: int) -> RegisteredWindow:
        self._check_valid("register_window")
        r = RegisteredWindow(self, ptr, size)
        self._resources.append(r)
        return r

    def alloc_window(self, size: int) -> AllocatedWindow:
        self._check_valid("alloc_window")
        r = AllocatedWindow(self, size)
        self._resources.append(r)
        return r

    def create_dev_comm(
        self,
        requirements: CCODevCommRequirements | None = None,
    ) -> DevCommHandle:
        self._check_valid("create_dev_comm")
        reqs = requirements if requirements is not None else CCODevCommRequirements()
        r = DevCommHandle(self, reqs)
        self._resources.append(r)
        return r

    def barrier(self) -> None:
        self._check_valid("barrier")
        _cco.barrier_all(self._raw)

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<Communicator: destroyed>"
        return (f"<Communicator: rank={self._rank}/{self._nranks}, "
                f"ptr={self._raw.ptr:#x}>")  # type: ignore[union-attr]

    def __enter__(self) -> Communicator:
        return self

    def __exit__(self, *_: object) -> None:
        self.destroy()
