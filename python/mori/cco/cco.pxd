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

    cdef struct ccoIbgdaWin:
        uint32_t* peerRkeys
        uint32_t lkey

    cdef struct ccoWindowDevice:
        char* winBase
        uint32_t stride4G
        int lsaRank
        ccoIbgdaWin ibgdaWin

    ctypedef ccoWindowDevice* ccoWindow_t

    cdef struct ccoWindowTableNode:
        pass

    cdef cppclass RdmaEndpointDevice:
        pass

cdef extern from "mori/cco/cco.hpp" namespace "anvil":
    cdef cppclass SdmaQueueDeviceHandle:
        pass

cdef extern from "mori/cco/cco.hpp" namespace "mori::cco":

    cdef struct ccoUniqueId:
        char internal[128]

    cdef struct ccoDevResourceRequirements:
        pass

    cdef struct ccoIbgdaContext:
        RdmaEndpointDevice* endpoints
        int numQpPerPe
        int signalCount
        uint64_t* signalBuf
        uint64_t* signalShadows
        int counterCount
        uint64_t* counterBuf

    cdef struct ccoLsaBarrierHandle:
        uint32_t bufOffset
        int nBarriers

    cdef struct ccoGdaBarrierHandle:
        uint32_t signal0
        int nBarriers

    cdef struct ccoSdmaContext:
        uint32_t sdmaNumQueue
        SdmaQueueDeviceHandle** deviceHandles
        uint64_t* signalBuf
        uint64_t* expectSignals
        uint64_t** peerSignalPtrs

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
        ccoWindowTableNode* windowTable
        ccoWindowDevice* resourceWindow
        ccoWindowDevice resourceWindow_inlined
        ccoIbgdaContext ibgda
        ccoLsaBarrierHandle lsaBarrier
        ccoGdaBarrierHandle railGdaBarrier
        ccoLsaBarrierHandle hybridLsaBarrier
        ccoGdaBarrierHandle hybridRailGdaBarrier
        ccoSdmaContext sdma

    int ccoGetUniqueId(ccoUniqueId* uniqueId) nogil
    int ccoCommCreate(const ccoUniqueId& uniqueId, int nRanks, int rank,
                      size_t perRankVmmSize, ccoComm** outComm) nogil
    int ccoCommDestroy(ccoComm* comm) nogil
    int ccoMemAlloc(ccoComm* comm, size_t size, void** outPtr) nogil
    int ccoMemImport(ccoComm* comm, void* externalPtr, size_t size, void** outPtr) nogil
    int ccoMemFree(ccoComm* comm, void* ptr) nogil
    int ccoWindowRegister(ccoComm* comm, size_t size,
                          ccoWindow_t* outWin, void** outLocalPtr) nogil
    int ccoWindowRegister(ccoComm* comm, void* ptr, size_t size,
                          ccoWindow_t* outWin) nogil
    int ccoWindowRegister(ccoComm* comm, void* externalPtr, size_t size,
                          ccoWindow_t* outWin, void** outLocalPtr) nogil
    int ccoWindowDeregister(ccoComm* comm, ccoWindow_t win) nogil
    int ccoDevCommCreate(ccoComm* comm, const ccoDevCommRequirements* reqs,
                         ccoDevComm* outDevComm) nogil
    int ccoDevCommDestroy(ccoComm* comm, ccoDevComm* devComm) nogil
    ccoDevComm* ccoDevCommCopyToDevice(const ccoDevComm* host) nogil
    void ccoDevCommFreeDeviceCopy(ccoDevComm* devicePtr) nogil
    int ccoBarrierAll(ccoComm* comm) nogil


cdef class UniqueId:
    cdef ccoUniqueId _uid

cdef class Comm:
    cdef ccoComm* _ptr

cdef class DevCommRequirements:
    cdef ccoDevCommRequirements _reqs

cdef class DevComm:
    cdef ccoDevComm _dc          # host shadow (for DevCommDestroy + property access)
    cdef ccoDevComm* _device_ptr  # device copy (for kernel pointer arguments)
    cdef object _comm
