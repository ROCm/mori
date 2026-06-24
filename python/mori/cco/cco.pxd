# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
# distutils: language = c++

from libc.stdint cimport intptr_t, uint32_t, uint64_t
from libc.stddef cimport size_t

cdef extern from "mori/cco/cco.hpp" namespace "mori::cco":

    unsigned int CCO_API_MAGIC
    unsigned int CCO_API_VERSION

    ctypedef enum ccoGdaConnectionType:
        CCO_GDA_CONNECTION_NONE
        CCO_GDA_CONNECTION_FULL
        CCO_GDA_CONNECTION_CROSSNODE
        CCO_GDA_CONNECTION_RAIL

    ctypedef enum ccoProviderType:
        CCO_PROVIDER_UNKNOWN
        CCO_PROVIDER_MLX5
        CCO_PROVIDER_BNXT
        CCO_PROVIDER_PSD
        CCO_PROVIDER_IBVERBS

    cdef cppclass ccoComm:
        pass

    cdef cppclass ccoWindowDevice:
        pass

    ctypedef ccoWindowDevice* ccoWindow_t

    cdef struct ccoUniqueId:
        char internal[128]

    cdef struct ccoDevResourceRequirements:
        pass

    cdef struct ccoDevCommRequirements:
        size_t size
        uint32_t magic
        uint32_t version
        ccoDevResourceRequirements* resourceRequirementsList
        ccoGdaConnectionType gdaConnectionType
        int gdaContextCount
        int gdaSignalCount
        int gdaCounterCount
        int gdaQueueDepth
        int gdaTrafficClass
        int lsaBarrierCount
        int railGdaBarrierCount
        int sdmaQueueCount
        int barrierCount

    cdef struct ccoDevComm:
        int rank
        int worldSize
        int lsaSize
        int lsaRank
        int myNodeStart
        ccoGdaConnectionType gdaConnType
        void* flatBase
        size_t perRankSize

    int ccoGetUniqueId(ccoUniqueId* uniqueId) nogil
    int ccoCommCreate(const ccoUniqueId& uniqueId, int nRanks, int rank,
                      size_t perRankVmmSize, ccoComm** outComm) nogil
    int ccoCommDestroy(ccoComm* comm) nogil
    int ccoMemAlloc(ccoComm* comm, size_t size, void** outPtr) nogil
    int ccoMemFree(ccoComm* comm, void* ptr) nogil
    int ccoWindowRegister(ccoComm* comm, size_t size,
                          ccoWindow_t* outWin, void** outLocalPtr) nogil
    int ccoWindowRegister(ccoComm* comm, void* ptr, size_t size,
                          ccoWindow_t* outWin) nogil
    int ccoWindowDeregister(ccoComm* comm, ccoWindow_t win) nogil
    int ccoDevCommCreate(ccoComm* comm, const ccoDevCommRequirements* reqs,
                         ccoDevComm* outDevComm) nogil
    int ccoDevCommDestroy(ccoComm* comm, ccoDevComm* devComm) nogil
    int ccoBarrierAll(ccoComm* comm) nogil


cdef class UniqueId:
    cdef ccoUniqueId _uid

cdef class Comm:
    cdef ccoComm* _ptr

cdef class DevCommRequirements:
    cdef ccoDevCommRequirements _reqs

cdef class DevComm:
    cdef ccoDevComm _dc
    cdef object _comm
