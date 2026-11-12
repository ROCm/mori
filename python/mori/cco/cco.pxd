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
    void* ccoGetPeerPtr(ccoComm* comm, void* localPtr, int pe) nogil

    cdef struct ccoTriggerQueue:
        pass
    ctypedef ccoTriggerQueue* ccoTriggerQueue_t

    ctypedef enum ccoSdmaCmp:
        ccoSdmaCmpGe
    size_t CCO_SDMA_MAX_COPY_BYTES

    cdef struct ccoTriggerCopy:
        const void* src
        void* dst
        uint64_t bytes

    cdef struct ccoTriggerQueueInfo:
        int peer
        uint32_t engineId
        uint64_t ringBytes

    int ccoTriggerQueueCreate(ccoComm* comm, int pe, int engineOffset, uint64_t ringBytes,
                              ccoTriggerQueue_t* queue) nogil
    int ccoTriggerQueueDestroy(ccoTriggerQueue_t queue) nogil
    int ccoTriggerAppendChain(ccoTriggerQueue_t queue, const void* gate, uint32_t target,
                              ccoSdmaCmp cmp, const ccoTriggerCopy* copies,
                              int numCopies, void* arrival, uint64_t* endIndex) nogil
    int ccoTriggerKick(ccoTriggerQueue_t queue) nogil
    int ccoTriggerKickAll(const ccoTriggerQueue_t* queues, int numQueues) nogil
    int ccoTriggerQuiesce(ccoTriggerQueue_t queue, uint64_t index, uint64_t timeoutNs) nogil
    int ccoTriggerQueueState(ccoTriggerQueue_t queue, uint64_t* readIndex,
                             uint64_t* writeIndex) nogil
    int ccoTriggerQueueGetInfo(ccoTriggerQueue_t queue, ccoTriggerQueueInfo* info) nogil
    int ccoTriggerQueueReadRing(ccoTriggerQueue_t queue, uint64_t index, void* out,
                                uint64_t bytes) nogil


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

cdef class _TriggerQueue:
    cdef ccoTriggerQueue_t _q
    cdef object _comm

cdef struct _BatchChain:
    int queue
    const void* gate
    Py_ssize_t slot
    void* arrival
    int numCopies
    ccoTriggerCopy copies[2]

cdef class _TriggerBatch:
    cdef _BatchChain* _chains
    cdef int _n
    cdef int _cap
    cdef ccoTriggerQueue_t* _qs
    cdef uint64_t* _need
    cdef uint64_t* _ring
    cdef int _nq
    cdef list _queues
