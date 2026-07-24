# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
# distutils: language = c++

from libc.stdint cimport intptr_t
from libc.stddef cimport size_t
from libc.string cimport memcpy

from .cco cimport (
    ccoComm, ccoUniqueId, ccoDevCommRequirements, ccoDevComm, ccoWindow_t,
    ccoGdaConnectionType,
    CCO_API_MAGIC, CCO_API_VERSION,
    CCO_GDA_CONNECTION_NONE, CCO_GDA_CONNECTION_FULL,
    CCO_GDA_CONNECTION_CROSSNODE, CCO_GDA_CONNECTION_RAIL,
    ccoGetUniqueId, ccoCommCreate, ccoCommDestroy,
    ccoMemAlloc, ccoMemImport, ccoMemFree,
    ccoWindowDeregister, ccoDevCommCreate, ccoDevCommDestroy,
    ccoDevCommCopyToDevice, ccoDevCommFreeDeviceCopy,
    ccoBarrierAll, ccoWindowRegister,
)


###############################################################################
# GdaConnectionType constants (mirrored as Python-level ints)
###############################################################################

GDA_CONNECTION_NONE      = CCO_GDA_CONNECTION_NONE
GDA_CONNECTION_FULL      = CCO_GDA_CONNECTION_FULL
GDA_CONNECTION_CROSSNODE = CCO_GDA_CONNECTION_CROSSNODE
GDA_CONNECTION_RAIL      = CCO_GDA_CONNECTION_RAIL


###############################################################################
# UniqueId
###############################################################################

cdef class UniqueId:
    """128-byte rendezvous identifier produced by rank 0 and broadcast out-of-band."""

    @staticmethod
    def create():
        """Call on rank 0 to generate a rendezvous id."""
        cdef UniqueId obj = UniqueId.__new__(UniqueId)
        cdef int ret
        with nogil:
            ret = ccoGetUniqueId(&obj._uid)
        if ret != 0:
            raise RuntimeError(f"ccoGetUniqueId failed: {ret}")
        return obj

    def tobytes(self):
        """Serialize to 128-byte bytes for out-of-band broadcast."""
        return bytes(self._uid.internal[:128])

    @staticmethod
    def frombytes(data):
        """Reconstruct from 128-byte bytes received from rank 0."""
        if len(data) != 128:
            raise ValueError(f"UniqueId requires exactly 128 bytes, got {len(data)}")
        cdef UniqueId obj = UniqueId.__new__(UniqueId)
        memcpy(obj._uid.internal, <const char*>data, 128)
        return obj

    def __repr__(self):
        return f"UniqueId({self.tobytes().hex()[:16]}…)"


###############################################################################
# Comm
###############################################################################

cdef class Comm:
    """Opaque communicator handle. Owns the underlying ccoComm*."""

    def __dealloc__(self):
        if self._ptr != NULL:
            ccoCommDestroy(self._ptr)
            self._ptr = NULL

    @property
    def ptr(self):
        """intptr_t address of the underlying ccoComm*."""
        return <intptr_t>self._ptr


###############################################################################
# DevCommRequirements
###############################################################################

cdef class DevCommRequirements:
    """
    Parameters for ccoDevCommCreate.  Initialised to safe defaults matching
    CCO_DEV_COMM_REQUIREMENTS_INITIALIZER; override individual fields before
    passing to dev_comm_create().
    """

    def __init__(self):
        self._reqs.size = sizeof(ccoDevCommRequirements)
        self._reqs.magic = CCO_API_MAGIC
        self._reqs.version = CCO_API_VERSION
        self._reqs.resourceRequirementsList = NULL
        self._reqs.gdaConnectionType = CCO_GDA_CONNECTION_CROSSNODE
        self._reqs.gdaContextCount = 4
        self._reqs.gdaSignalCount = 16
        self._reqs.gdaCounterCount = 16
        self._reqs.gdaQueueDepth = 0
        self._reqs.gdaTrafficClass = -1
        self._reqs.lsaBarrierCount = 0
        self._reqs.railGdaBarrierCount = 0
        self._reqs.sdmaQueueCount = 0
        self._reqs.barrierCount = 0

    @property
    def gda_connection_type(self):
        return <int>self._reqs.gdaConnectionType

    @gda_connection_type.setter
    def gda_connection_type(self, int val):
        self._reqs.gdaConnectionType = <ccoGdaConnectionType>val

    @property
    def gda_context_count(self):
        return self._reqs.gdaContextCount

    @gda_context_count.setter
    def gda_context_count(self, int val):
        self._reqs.gdaContextCount = val

    @property
    def gda_signal_count(self):
        return self._reqs.gdaSignalCount

    @gda_signal_count.setter
    def gda_signal_count(self, int val):
        self._reqs.gdaSignalCount = val

    @property
    def gda_counter_count(self):
        return self._reqs.gdaCounterCount

    @gda_counter_count.setter
    def gda_counter_count(self, int val):
        self._reqs.gdaCounterCount = val

    @property
    def gda_queue_depth(self):
        return self._reqs.gdaQueueDepth

    @gda_queue_depth.setter
    def gda_queue_depth(self, int val):
        self._reqs.gdaQueueDepth = val

    @property
    def gda_traffic_class(self):
        return self._reqs.gdaTrafficClass

    @gda_traffic_class.setter
    def gda_traffic_class(self, int val):
        self._reqs.gdaTrafficClass = val

    @property
    def lsa_barrier_count(self):
        return self._reqs.lsaBarrierCount

    @lsa_barrier_count.setter
    def lsa_barrier_count(self, int val):
        self._reqs.lsaBarrierCount = val

    @property
    def rail_gda_barrier_count(self):
        return self._reqs.railGdaBarrierCount

    @rail_gda_barrier_count.setter
    def rail_gda_barrier_count(self, int val):
        self._reqs.railGdaBarrierCount = val

    @property
    def sdma_queue_count(self):
        return self._reqs.sdmaQueueCount

    @sdma_queue_count.setter
    def sdma_queue_count(self, int val):
        self._reqs.sdmaQueueCount = val

    @property
    def barrier_count(self):
        return self._reqs.barrierCount

    @barrier_count.setter
    def barrier_count(self, int val):
        self._reqs.barrierCount = val


###############################################################################
# DevComm
###############################################################################

cdef class DevComm:
    """
    Host-side snapshot of the device communicator.  Pass by value (or pass
    .ptr as an intptr_t) to GPU kernels.  Kept alive by the Comm that owns
    its resources.
    """

    @property
    def ptr(self):
        """intptr_t address of the device-side ccoDevComm copy (for kernel arguments)."""
        return <intptr_t>self._device_ptr

    @property
    def host_ptr(self):
        """intptr_t address of the host-side ccoDevComm struct (for by-value kernel args)."""
        return <intptr_t>&self._dc

    @property
    def rank(self):
        return self._dc.rank

    @property
    def world_size(self):
        return self._dc.worldSize

    @property
    def lsa_size(self):
        return self._dc.lsaSize

    @property
    def lsa_rank(self):
        return self._dc.lsaRank

    @property
    def my_node_start(self):
        return self._dc.myNodeStart

    @property
    def gda_conn_type(self):
        return <int>self._dc.gdaConnType

    @property
    def flat_base(self):
        return <intptr_t>self._dc.flatBase

    @property
    def per_rank_size(self):
        return self._dc.perRankSize

    def __repr__(self):
        return (f"DevComm(rank={self.rank}, world_size={self.world_size}, "
                f"lsa_size={self.lsa_size}, lsa_rank={self.lsa_rank})")


###############################################################################
# Host API — free functions
###############################################################################

def get_unique_id():
    """Rank 0: generate a rendezvous UniqueId."""
    return UniqueId.create()


def comm_create(UniqueId uid, int n_ranks, int rank, size_t per_rank_vmm_size):
    """
    Create a communicator.  All ranks call with the same uid (broadcast from
    rank 0) and their own rank index.

    per_rank_vmm_size: bytes of flat VA reserved per rank for symmetric memory.
    """
    cdef Comm comm = Comm.__new__(Comm)
    cdef ccoComm* out_comm = NULL
    cdef int ret
    with nogil:
        ret = ccoCommCreate(uid._uid, n_ranks, rank, per_rank_vmm_size, &out_comm)
    if ret != 0:
        raise RuntimeError(f"ccoCommCreate failed: {ret}")
    comm._ptr = out_comm
    return comm


def comm_destroy(Comm comm):
    """Destroy and release the communicator (also called by Comm.__dealloc__)."""
    cdef int ret
    if comm._ptr == NULL:
        return
    with nogil:
        ret = ccoCommDestroy(comm._ptr)
    comm._ptr = NULL
    if ret != 0:
        raise RuntimeError(f"ccoCommDestroy failed: {ret}")


def mem_alloc(Comm comm, size_t size):
    """
    Allocate symmetric GPU memory.  Returns the local pointer as intptr_t.
    The allocation is P2P-accessible by all LSA peers after window_register().
    """
    cdef void* ptr = NULL
    cdef int ret
    with nogil:
        ret = ccoMemAlloc(comm._ptr, size, &ptr)
    if ret != 0:
        raise RuntimeError(f"ccoMemAlloc failed: {ret}")
    return <intptr_t>ptr


def mem_import(Comm comm, intptr_t external_ptr, size_t size):
    """
    Import an external HIP VMM allocation (e.g. a torch.symm_mem tensor buffer)
    into this rank's flat-VA slot without allocating new physical memory. Returns
    the local (flat-VA) pointer as intptr_t; it aliases the external buffer and can
    be passed to window_register_ptr() and freed with mem_free().
    """
    cdef void* ptr = NULL
    cdef int ret
    with nogil:
        ret = ccoMemImport(comm._ptr, <void*>external_ptr, size, &ptr)
    if ret != 0:
        raise RuntimeError(f"ccoMemImport failed: {ret}")
    return <intptr_t>ptr


def mem_free(Comm comm, intptr_t ptr):
    """Free symmetric GPU memory previously allocated by mem_alloc()."""
    cdef int ret
    with nogil:
        ret = ccoMemFree(comm._ptr, <void*>ptr)
    if ret != 0:
        raise RuntimeError(f"ccoMemFree failed: {ret}")


def window_register(Comm comm, size_t size):
    """
    Overload A: CCO allocates symmetric memory internally and registers it.
    Collective — all ranks call with the same size in the same order.

    Returns (win: intptr_t, local_ptr: intptr_t).
    """
    cdef ccoWindow_t win = NULL
    cdef void* local_ptr = NULL
    cdef int ret
    with nogil:
        ret = ccoWindowRegister(comm._ptr, size, &win, &local_ptr)
    if ret != 0:
        raise RuntimeError(f"ccoWindowRegister (alloc) failed: {ret}")
    return (<intptr_t>win, <intptr_t>local_ptr)


def window_register_ptr(Comm comm, intptr_t ptr, size_t size):
    """
    Overload B: register a pre-allocated ccoMemAlloc pointer as a window.
    Collective — all ranks call in the same order with the same size.

    Returns win: intptr_t.
    """
    cdef ccoWindow_t win = NULL
    cdef int ret
    with nogil:
        ret = ccoWindowRegister(comm._ptr, <void*>ptr, size, &win)
    if ret != 0:
        raise RuntimeError(f"ccoWindowRegister (ptr) failed: {ret}")
    return <intptr_t>win


def window_register_external(Comm comm, intptr_t external_ptr, size_t size):
    """
    Overload C: import an EXTERNAL HIP VMM allocation (e.g. a torch.symm_mem
    tensor buffer) into the flat LSA space and register it as a window in one call.
    Collective — all ranks call in the same order with the same size.

    Returns (win: intptr_t, local_ptr: intptr_t), where local_ptr is the flat-VA
    alias of the external buffer.
    """
    cdef ccoWindow_t win = NULL
    cdef void* local_ptr = NULL
    cdef int ret
    with nogil:
        ret = ccoWindowRegister(comm._ptr, <void*>external_ptr, size, &win, &local_ptr)
    if ret != 0:
        raise RuntimeError(f"ccoWindowRegister (external) failed: {ret}")
    return (<intptr_t>win, <intptr_t>local_ptr)


def window_deregister(Comm comm, intptr_t win):
    """Deregister and release a window.  Collective."""
    cdef int ret
    with nogil:
        ret = ccoWindowDeregister(comm._ptr, <ccoWindow_t>win)
    if ret != 0:
        raise RuntimeError(f"ccoWindowDeregister failed: {ret}")


def dev_comm_create(Comm comm, DevCommRequirements reqs):
    """
    Build a DevComm from the communicator.  The returned DevComm's .ptr gives
    a device-side intptr_t address suitable for passing as a kernel argument.
    """
    cdef DevComm dc = DevComm.__new__(DevComm)
    dc._comm = comm
    dc._device_ptr = NULL
    cdef int ret
    with nogil:
        ret = ccoDevCommCreate(comm._ptr, &reqs._reqs, &dc._dc)
    if ret != 0:
        raise RuntimeError(f"ccoDevCommCreate failed: {ret}")
    dc._device_ptr = ccoDevCommCopyToDevice(&dc._dc)
    return dc


def dev_comm_destroy(Comm comm, DevComm dev_comm):
    """Release device resources held by a DevComm."""
    ccoDevCommFreeDeviceCopy(dev_comm._device_ptr)
    dev_comm._device_ptr = NULL
    cdef int ret
    with nogil:
        ret = ccoDevCommDestroy(comm._ptr, &dev_comm._dc)
    if ret != 0:
        raise RuntimeError(f"ccoDevCommDestroy failed: {ret}")


def barrier_all(Comm comm):
    """CPU-side collective barrier across all ranks."""
    cdef int ret
    with nogil:
        ret = ccoBarrierAll(comm._ptr)
    if ret != 0:
        raise RuntimeError(f"ccoBarrierAll failed: {ret}")
