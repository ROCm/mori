// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#pragma once

#include "mori/core/transport/rdma/proxy/proxy_device_primitives.hpp"
#include "mori/shmem/shmem_proxy_state.hpp"

#ifdef __HIPCC__

namespace mori {
namespace shmem {
extern __device__ __attribute__((visibility("default"))) ProxyGpuState globalProxyState;
static __device__ ProxyGpuState* GetGlobalProxyStatePtr() { return &globalProxyState; }
}  // namespace shmem
namespace shmem {

inline __device__ volatile core::ProxyRing* ProxyRingForEp(
    ProxyGpuState* ps, uint32_t epIndex) {
  int pe = epIndex / ps->numQpPerPe;
  int peerLocal = pe % ps->numNics;
  int nicIdx = (ps->localGpuIdx > peerLocal ? ps->localGpuIdx : peerLocal) % ps->numNics;
  return static_cast<volatile core::ProxyRing*>(ps->rings[nicIdx]);
}

// Proxy variant of ShmemPutMemNbiThreadKernelImpl — RDMA WRITE via proxy ring

inline __device__ void ShmemPutMemNbiThreadKernelImpl_proxy(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe,
    int qpId) {
  if (bytes == 0) return;
  ProxyGpuState* ps = GetGlobalProxyStatePtr();
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);
  volatile core::ProxyRing* ring = ProxyRingForEp(ps, epIndex);
  uint32_t lkey = source->lkey;
  uintptr_t srcAddr = reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset;
  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uint32_t rkey = dest->peerRkeys[pe];
  core::ProxyPostWrite(ring, epIndex, srcAddr, lkey, raddr, rkey, bytes);
}

// Proxy variant of ShmemPutSizeImmNbiThreadKernelImpl — inline RDMA WRITE

inline __device__ void ShmemPutSizeImmNbiThreadKernelImpl_proxy(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes,
    int pe, int qpId) {
  ProxyGpuState* ps = GetGlobalProxyStatePtr();
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);
  volatile core::ProxyRing* ring = ProxyRingForEp(ps, epIndex);
  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uint32_t rkey = dest->peerRkeys[pe];
  core::ProxyPostWriteInline(ring, epIndex,
                             reinterpret_cast<uint64_t>(val), 0, raddr, rkey, bytes);
}

// Proxy variant of ShmemPutMemNbiSignalThreadKernelImpl — data + signal

template <bool onlyOneSignal = true>
inline __device__ void ShmemPutMemNbiSignalThreadKernelImpl_proxy(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  if (bytes == 0) return;
  ProxyGpuState* ps = GetGlobalProxyStatePtr();
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);
  volatile core::ProxyRing* ring = ProxyRingForEp(ps, epIndex);
  uint32_t lkey = source->lkey;
  uintptr_t srcAddr = reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset;
  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uint32_t rkey = dest->peerRkeys[pe];
  core::ProxyPostWrite(ring, epIndex, srcAddr, lkey, raddr, rkey, bytes);
  uintptr_t sigRaddr = signalDest->peerPtrs[pe] + signalDestOffset;
  uint32_t sigRkey = signalDest->peerRkeys[pe];
  core::IbufHandle& ibuf = gs->rdmaEndpoints[epIndex].atomicIbuf;
  core::ProxyPostSignalWrite(ring, epIndex, sigRaddr, sigRkey, signalValue,
                             ibuf.lkey, ibuf.addr);
}

// Proxy variant of ShmemAtomicSizeNonFetchThreadKernelImpl — fire-and-forget atomic

inline __device__ void ShmemAtomicSizeNonFetchThreadKernelImpl_proxy(
    const application::SymmMemObjPtr dest, size_t destOffset, const void* val,
    size_t bytes, core::atomicType amoType, int pe, int qpId) {
  ProxyGpuState* ps = GetGlobalProxyStatePtr();
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);
  volatile core::ProxyRing* ring = ProxyRingForEp(ps, epIndex);
  uintptr_t raddr;
  uint32_t rkey;
  if (gs->useVMMHeap) {
    uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dest->localPtr) + destOffset;
    VmmLookupRemote(dstAddr, pe, raddr, rkey);
  } else {
    raddr = dest->peerPtrs[pe] + destOffset;
    rkey = dest->peerRkeys[pe];
  }
  core::IbufHandle& ibuf = gs->rdmaEndpoints[epIndex].atomicIbuf;
  uint64_t atomicVal = 0;
  memcpy(&atomicVal, val, bytes <= 8 ? bytes : 8);
  core::ProxyPostAtomicNonFetch(ring, epIndex, raddr, rkey, atomicVal, ibuf.lkey, ibuf.addr);
}

// Proxy variant of ShmemAtomicTypeFetchThreadKernelImpl — fetch atomic (blocks until done)

template <typename T>
inline __device__ T ShmemAtomicTypeFetchThreadKernelImpl_proxy(
    const application::SymmMemObjPtr dest, size_t destOffset, const void* val,
    size_t bytes, core::atomicType amoType, int pe, int qpId) {
  ProxyGpuState* ps = GetGlobalProxyStatePtr();
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);
  volatile core::ProxyRing* ring = ProxyRingForEp(ps, epIndex);
  uintptr_t raddr;
  uint32_t rkey;
  if (gs->useVMMHeap) {
    uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dest->localPtr) + destOffset;
    VmmLookupRemote(dstAddr, pe, raddr, rkey);
  } else {
    raddr = dest->peerPtrs[pe] + destOffset;
    rkey = dest->peerRkeys[pe];
  }
  core::IbufHandle& ibuf = gs->rdmaEndpoints[epIndex].atomicIbuf;
  uint64_t atomicVal = 0;
  memcpy(&atomicVal, val, bytes <= 8 ? bytes : 8);
  uint64_t result = core::ProxyPostAtomicFetch(ring, epIndex, raddr, rkey,
                                                atomicVal, ibuf.lkey, ibuf.addr);
  T retVal;
  memcpy(&retVal, &result, sizeof(T));
  return retVal;
}

// Proxy variant of ShmemQuietThreadKernelPsdImpl — per-NIC targeted quiet
inline __device__ void ShmemQuietThreadKernelPsdImpl_proxy(int pe, int qpId) {
  ProxyGpuState* ps = GetGlobalProxyStatePtr();
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);
  int peerLocal = pe % ps->numNics;
  int nicIdx = (ps->localGpuIdx > peerLocal ? ps->localGpuIdx : peerLocal) % ps->numNics;
  volatile core::ProxyRing* ring = static_cast<volatile core::ProxyRing*>(ps->rings[nicIdx]);
  if (ring) {
    uint32_t head = ring->gpu_head;
    uint32_t lastQuiet = ps->quietHead[nicIdx];
    if (head != lastQuiet) {
      core::ProxyQuiet(ring, lastQuiet, head - lastQuiet);
      ps->quietHead[nicIdx] = head;
    }
  }
}

// Proxy quiet for all rings (used by fence)
inline __device__ void ShmemQuietAllProxy() {
  ProxyGpuState* ps = GetGlobalProxyStatePtr();
  for (int n = 0; n < ps->numRings; n++) {
    volatile core::ProxyRing* ring = static_cast<volatile core::ProxyRing*>(ps->rings[n]);
    if (!ring) continue;
    uint32_t head = ring->gpu_head;
    uint32_t lastQuiet = ps->quietHead[n];
    if (head != lastQuiet) {
      core::ProxyQuiet(ring, lastQuiet, head - lastQuiet);
      ps->quietHead[n] = head;
    }
  }
}

}  // namespace shmem
}  // namespace mori

#endif  // __HIPCC__
