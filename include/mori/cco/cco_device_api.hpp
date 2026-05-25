// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
//
// CCO Device API — skeleton declarations for Phase 1.
// Implementations will be filled in Phase 2.
#pragma once

#include "mori/cco/cco_types.hpp"

namespace mori {
namespace cco {

// ── Window lookup: find window by pointer (like ncclFindWindow) ──
__device__ inline CcoWindow_t CcoFindWindow(CcoDevComm* comm, const void* ptr) {
  uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
  CcoWindowTableNode* node = comm->windowTable;
  while (node) {
    for (int i = 0; i < CCO_WINDOW_TABLE_SIZE; i++) {
      auto& e = node->entries[i];
      if (e.base != 0 && e.size != 0 && e.window != nullptr) {
        if (uptr >= e.base && uptr < e.base + e.size) {
          return e.window;
        }
      }
    }
    node = node->next;
  }
  return nullptr;
}

// ── Address helpers ──
__device__ inline void* CcoGetPeerPtr(CcoWindow_t win, int pe, size_t offset = 0) {
  return win->winBase + ((static_cast<uint64_t>(pe) * win->stride4G) << 32) + offset;
}

__device__ inline void* CcoGetLocalPtr(CcoWindow_t win, size_t offset = 0) {
  return win->winBase + ((static_cast<uint64_t>(win->rank) * win->stride4G) << 32) + offset;
}

// ── P2P: direct GPU store, intra-node xGMI ──
__device__ inline void CcoP2pPutThread(CcoWindow_t dst, size_t dstOff,
                                          CcoWindow_t src, size_t srcOff, size_t bytes,
                                          int pe) {
  (void)dst;
  (void)dstOff;
  (void)src;
  (void)srcOff;
  (void)bytes;
  (void)pe;
  // Phase 2: void* remote = CcoGetPeerPtr(dst, pe, dstOff);
  //          void* local  = CcoGetLocalPtr(src, srcOff);
  //          p2pPutThread(local, remote, bytes);
}

// ── RDMA: ibgda RDMA Write, inter-node ──
__device__ inline void CcoRdmaPutThread(CcoDevComm* comm, CcoWindow_t dst, size_t dstOff,
                                           CcoWindow_t src, size_t srcOff, size_t bytes, int pe,
                                           int qpId = 0) {
  (void)comm;
  (void)dst;
  (void)dstOff;
  (void)src;
  (void)srcOff;
  (void)bytes;
  (void)pe;
  (void)qpId;
  // Phase 2: raddr = dstOff (iova=0), rkey = dst->ibgdaWin.peerRkeys[pe]
  //          laddr = srcOff (iova=0), lkey = src->ibgdaWin.lkey
  //          QP endpoint = comm->ibgda.endpoints[pe * comm->ibgda.numQpPerPe + qpId]
}

__device__ inline void CcoRdmaQuietThread(CcoDevComm* comm, int pe, int qpId = 0) {
  (void)comm;
  (void)pe;
  (void)qpId;
  // Phase 2: poll CQ / drain WQE
}

// ── SDMA: DMA engine packet queue, intra-node ──
__device__ inline void CcoSdmaPutThread(CcoWindow_t dst, size_t dstOff, CcoWindow_t src,
                                           size_t srcOff, size_t bytes, int pe, int qpId = 0) {
  (void)dst;
  (void)dstOff;
  (void)src;
  (void)srcOff;
  (void)bytes;
  (void)pe;
  (void)qpId;
  // Phase 2: dstPtr = CcoGetPeerPtr(dst, pe, dstOff)
  //          srcPtr = CcoGetLocalPtr(src, srcOff)
  //          core::SdmaPutThread(...)
}

__device__ inline void CcoSdmaQuietThread(CcoWindow_t win, int pe, int qpId = 0) {
  (void)win;
  (void)pe;
  (void)qpId;
  // Phase 2: wait on expectSignalsPtr
}

// ── Signal: remote notification (analogous to NCCL ncclGin_SignalInc) ──
// Remote peer's NIC does RDMA atomic +1 to comm->ibgda.signalBuf[signalIndex].
// Signal raddr for peer pe: signalIndex * sizeof(uint64_t)
// Signal rkey for peer pe: comm->ibgda.peerSignalRkeys[pe]

__device__ inline uint64_t CcoReadSignal(CcoDevComm* comm, int signalIndex) {
  // Phase 2: return atomicLoad(&comm->ibgda.signalBuf[signalIndex])
  (void)comm;
  (void)signalIndex;
  return 0;
}

__device__ inline void CcoWaitSignal(CcoDevComm* comm, int signalIndex, uint64_t threshold) {
  // Phase 2: spin until comm->ibgda.signalBuf[signalIndex] >= threshold
  (void)comm;
  (void)signalIndex;
  (void)threshold;
}

// ── Counter: local completion (analogous to NCCL ncclGin_CounterInc) ──
// NIC loopback writes to comm->ibgda.counterBuf after source data fully transmitted.

__device__ inline uint64_t CcoReadCounter(CcoDevComm* comm, int counterIndex) {
  (void)comm;
  (void)counterIndex;
  return 0;
}

__device__ inline void CcoWaitCounter(CcoDevComm* comm, int counterIndex,
                                         uint64_t threshold) {
  (void)comm;
  (void)counterIndex;
  (void)threshold;
}

// ── Barrier ──
__device__ inline void CcoBarrierAllBlock(CcoDevComm* comm) {
  (void)comm;
  // Phase 2: reuse ShmemInternalBarrierBlock logic with comm->internalSyncPtr
}

}  // namespace cco
}  // namespace mori
