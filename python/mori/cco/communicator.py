# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License

"""CCO high-level Communicator class.

Mirrors the nccl4py ``Communicator`` pattern:

* :py:meth:`Communicator.init` — class-method constructor (collective).
* Resource factory methods (:py:meth:`alloc_mem`, :py:meth:`alloc_window`,
  :py:meth:`register_window`, :py:meth:`create_dev_comm`) return
  :py:mod:`~mori.cco.resources` objects that are owned and tracked by the
  communicator.
* :py:meth:`destroy` closes all tracked resources and frees the communicator.
"""

from __future__ import annotations

from typing import Sequence

import mori.cco.cco as _cco
from mori.cco.resources import (
    AllocatedMemory,
    AllocatedWindow,
    CCOResource,
    DevCommHandle,
    RegisteredWindow,
)


__all__ = ["Communicator", "CCODevCommRequirements", "UniqueId", "get_unique_id"]


# ── UniqueId (pure-Python wrapper with pickle support) ───────────────────────

class UniqueId:
    """CCO unique identifier for communicator initialization.

    Wraps the Cython-level ``cco.UniqueId`` with pickle support so that
    ``mpi4py.MPI.Comm.bcast`` (lowercase ``b``, pickle-based) works
    out of the box.

    Serialization paths:

    * **Bytes**: ``bytes(uid)`` / :py:meth:`from_bytes`.
    * **Pickle**: ``__getstate__`` / ``__setstate__`` — used by MPI bcast.
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

    Example::

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.lsa_barrier_count   = 4
        dc = comm.create_dev_comm(reqs)
    """


# ── Communicator ──────────────────────────────────────────────────────────────

class Communicator:
    """CCO communicator for intra- and inter-node symmetric memory operations.

    A communicator groups ``nranks`` processes, each pinned to one GPU.
    It manages the flat VMM address space used by all CCO symmetric-memory
    operations.

    Typical lifecycle::

        uid  = Communicator.get_unique_id() if rank == 0 else None
        uid  = mpi_comm.bcast(uid, root=0)
        comm = Communicator.init(nranks, rank, uid, per_rank_vmm=4 << 30)

        win  = comm.alloc_window(size)      # allocate + register
        dc   = comm.create_dev_comm()       # build GPU-side DevComm

        # ... launch kernels using win.handle and dc.ptr ...

        comm.barrier()                      # host collective fence
        comm.destroy()                      # releases all resources
    """

    # Default VMM budget: 4 GiB per rank (minimum for stride4G = 1).
    DEFAULT_PER_RANK_VMM: int = 4 * 1024 * 1024 * 1024

    def __init__(self) -> None:
        self._raw: _cco.Comm | None = None
        self._resources: list[CCOResource] = []
        self._rank: int | None = None
        self._nranks: int | None = None
        self._lsa_size: int | None = None
        self._lsa_rank: int | None = None

    # ── Constructors ──────────────────────────────────────────────────────────

    @staticmethod
    def get_unique_id() -> UniqueId:
        """Generate a rendezvous token on rank 0.

        Call only on rank 0, then broadcast the result to all ranks via MPI
        (or any other out-of-band channel) before calling
        :py:meth:`Communicator.init`.

        Returns:
            :py:class:`UniqueId` that serialises to 128 bytes.
        """
        return get_unique_id()

    @classmethod
    def init(
        cls,
        nranks: int,
        rank: int,
        unique_id: UniqueId,
        per_rank_vmm: int = DEFAULT_PER_RANK_VMM,
    ) -> Communicator:
        """Collective: create a CCO communicator.

        All ranks must call this with the same ``nranks`` and ``unique_id``
        (broadcast from rank 0) but different ``rank`` values.

        Args:
            nranks: Total number of ranks.
            rank: This rank's index (0 … nranks-1).
            unique_id: Token generated and broadcast by rank 0.
            per_rank_vmm: Bytes of flat VA reserved per rank for symmetric
                memory.  Must be a multiple of the VMM granularity (typically
                2 MiB on ROCm).  All ``alloc_mem`` / ``alloc_window`` calls
                on this comm must fit within this budget.
                Defaults to 4 GiB (minimum for ``stride4G = 1``).

        Returns:
            Initialised :py:class:`Communicator`.
        """
        comm = cls()
        uid = unique_id.ptr if isinstance(unique_id, UniqueId) else unique_id
        comm._raw    = _cco.comm_create(uid, nranks, rank, per_rank_vmm)
        comm._rank   = rank
        comm._nranks = nranks
        return comm

    # ── Teardown ──────────────────────────────────────────────────────────────

    def destroy(self) -> None:
        """Destroy the communicator and free all owned resources.

        Closes all tracked resources (windows, device communicators, allocated
        memory) in reverse-registration order, then frees the communicator.
        Safe to call multiple times.
        """
        self.close_all_resources()
        if self._raw is not None:
            _cco.comm_destroy(self._raw)
            self._raw = None

    def close_all_resources(self) -> None:
        """Close all tracked resources (best-effort; errors are suppressed).

        Called automatically by :py:meth:`destroy`.
        """
        for r in reversed(self._resources):
            try:
                r.close()
            except Exception:
                pass
        self._resources.clear()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_valid(self) -> bool:
        """True while the communicator has not been destroyed."""
        return self._raw is not None

    @property
    def rank(self) -> int:
        """This process's rank in the communicator."""
        self._check_valid("get rank")
        return self._rank  # type: ignore[return-value]

    @property
    def nranks(self) -> int:
        """Total number of ranks in the communicator."""
        self._check_valid("get nranks")
        return self._nranks  # type: ignore[return-value]

    @property
    def ptr(self) -> int:
        """Raw ``ccoComm*`` pointer (intptr_t) for advanced use."""
        self._check_valid("get ptr")
        return self._raw.ptr  # type: ignore[union-attr]

    def _check_valid(self, op: str = "use") -> None:
        if not self.is_valid:
            raise RuntimeError(f"Cannot {op}: Communicator has been destroyed")

    # ── Resource factories ────────────────────────────────────────────────────

    def alloc_mem(self, size: int) -> AllocatedMemory:
        """Allocate ``size`` bytes of symmetric GPU memory via ``ccoMemAlloc``.

        The allocation lives in the flat VA space shared by all LSA peers.
        To make it P2P-accessible, register it as a window with
        :py:meth:`register_window`.

        Args:
            size: Number of bytes to allocate.

        Returns:
            :py:class:`~mori.cco.resources.AllocatedMemory` tracking the
            allocation.
        """
        self._check_valid("alloc_mem")
        r = AllocatedMemory(self, size)
        self._resources.append(r)
        return r

    def register_window(self, ptr: int, size: int) -> RegisteredWindow:
        """Register a pre-allocated ``ccoMemAlloc`` pointer as a CCO window.

        Collective: all ranks must call in the same order with the same
        ``size``.  The caller retains ownership of the memory; close the
        :py:class:`~mori.cco.resources.AllocatedMemory` only *after* closing
        the window.

        Args:
            ptr: Device pointer (``intptr_t``) from :py:meth:`alloc_mem`.
            size: Size in bytes (must match the allocation).

        Returns:
            :py:class:`~mori.cco.resources.RegisteredWindow`.
        """
        self._check_valid("register_window")
        r = RegisteredWindow(self, ptr, size)
        self._resources.append(r)
        return r

    def alloc_window(self, size: int) -> AllocatedWindow:
        """Allocate *and* register a symmetric window in one collective call.

        CCO handles the underlying memory allocation internally.  This is the
        simplest path when you do not need to separate allocation from
        registration.

        Collective: all ranks must call in the same order with the same
        ``size``.

        Args:
            size: Window size in bytes.

        Returns:
            :py:class:`~mori.cco.resources.AllocatedWindow` with ``handle``
            and ``local_ptr``.
        """
        self._check_valid("alloc_window")
        r = AllocatedWindow(self, size)
        self._resources.append(r)
        return r

    def create_dev_comm(
        self,
        requirements: CCODevCommRequirements | None = None,
    ) -> DevCommHandle:
        """Build a GPU-side ``ccoDevComm`` from this communicator.

        Multiple device communicators can be created from a single host
        communicator (e.g. one per kernel launch type with different signal
        counts).

        Args:
            requirements: Resource configuration.  If ``None``, a default
                :py:class:`CCODevCommRequirements` is used (CROSSNODE GDA,
                16 signals, 16 counters, no barriers).

        Returns:
            :py:class:`~mori.cco.resources.DevCommHandle` whose ``ptr``
            property gives the ``intptr_t`` address of the ``ccoDevComm``
            struct for kernel arguments.
        """
        self._check_valid("create_dev_comm")
        reqs = requirements if requirements is not None else CCODevCommRequirements()
        r = DevCommHandle(self, reqs)
        self._resources.append(r)
        return r

    # ── Collective operations ──────────────────────────────────────────────────

    def barrier(self) -> None:
        """Host-side collective barrier across all ranks.

        Blocks until every rank has called this method.  Use after GPU kernel
        launches (with ``hipStreamSynchronize`` + ``__threadfence_system`` in
        the kernel) to guarantee all remote writes are visible before any rank
        reads them.
        """
        self._check_valid("barrier")
        _cco.barrier_all(self._raw)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if not self.is_valid:
            return "<Communicator: destroyed>"
        return (f"<Communicator: rank={self._rank}/{self._nranks}, "
                f"ptr={self._raw.ptr:#x}>")  # type: ignore[union-attr]

    def __enter__(self) -> Communicator:
        return self

    def __exit__(self, *_: object) -> None:
        self.destroy()
