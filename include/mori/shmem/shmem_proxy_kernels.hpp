// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#pragma once

#include "mori/core/transport/rdma/proxy/proxy_device_primitives.hpp"
#include "mori/shmem/internal.hpp"

#if defined(__HIPCC__) && defined(MORI_PROXY_ENABLED)

namespace mori {
namespace shmem {

inline __device__ volatile core::ProxyRing* ProxyRingForEp(
    ProxyGpuStates* ps, uint32_t epIndex) {
  int pe = epIndex / ps->numQpPerPe;
  int peerLocal = pe % ps->numNics;
  int nicIdx = (ps->localGpuIdx > peerLocal ? ps->localGpuIdx : peerLocal) % ps->numNics;
  return static_cast<volatile core::ProxyRing*>(ps->rings[nicIdx]);
}

inline __device__ void ShmemQuietAllProxy() {
  ProxyGpuStates* ps = GetGlobalProxyStatePtr();
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

// ---------------------------------------------------------------------------
// ShmemQuietThreadKernel<PROXY>
// ---------------------------------------------------------------------------
template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::PROXY>() {
  ShmemQuietAllProxy();
}

template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::PROXY>(int pe) {
  ProxyGpuStates* ps = GetGlobalProxyStatePtr();
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe;
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

template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::PROXY>(
    int pe, int qpId) {
  ShmemQuietThreadKernel<application::TransportType::PROXY>(pe);
}

// ---------------------------------------------------------------------------
// ShmemPutMemNbiThreadKernel<PROXY> (SymmMemObjPtr)
// ---------------------------------------------------------------------------
template <>
inline __device__ void ShmemPutMemNbiThreadKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe,
    int qpId) {
  if (bytes == 0) return;
  ProxyGpuStates* ps = GetGlobalProxyStatePtr();
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);
  volatile core::ProxyRing* ring = ProxyRingForEp(ps, epIndex);
  uint32_t lkey = source->lkey;
  uintptr_t srcAddr = reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset;
  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uint32_t rkey = dest->peerRkeys[pe];
  core::ProxyPostWrite(ring, epIndex, srcAddr, lkey, raddr, rkey, bytes);
}

template <>
inline __device__ void ShmemPutMemNbiWarpKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe,
    int qpId) {
  ShmemPutMemNbiThreadKernel<application::TransportType::PROXY>(
      dest, destOffset, source, sourceOffset, bytes, pe, qpId);
}

template <>
inline __device__ void ShmemPutMemNbiBlockKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe,
    int qpId) {
  ShmemPutMemNbiThreadKernel<application::TransportType::PROXY>(
      dest, destOffset, source, sourceOffset, bytes, pe, qpId);
}

// ---------------------------------------------------------------------------
// ShmemPutSizeImmNbiThreadKernel<PROXY> (SymmMemObjPtr)
// ---------------------------------------------------------------------------
template <>
inline __device__ void ShmemPutSizeImmNbiThreadKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes,
    int pe, int qpId) {
  ProxyGpuStates* ps = GetGlobalProxyStatePtr();
  GpuStates* gs = GetGlobalGpuStatesPtr();
  int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);
  volatile core::ProxyRing* ring = ProxyRingForEp(ps, epIndex);
  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uint32_t rkey = dest->peerRkeys[pe];
  core::ProxyPostWriteInline(ring, epIndex,
                             reinterpret_cast<uint64_t>(val), 0, raddr, rkey, bytes);
}

template <>
inline __device__ void ShmemPutSizeImmNbiWarpKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes,
    int pe, int qpId) {
  ShmemPutSizeImmNbiThreadKernel<application::TransportType::PROXY>(
      dest, destOffset, val, bytes, pe, qpId);
}

// ---------------------------------------------------------------------------
// ShmemPutMemNbiSignalThreadKernel<PROXY> (SymmMemObjPtr)
// ---------------------------------------------------------------------------
template <>
inline __device__ void ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, true>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  if (bytes == 0) return;
  ProxyGpuStates* ps = GetGlobalProxyStatePtr();
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

template <>
inline __device__ void ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, false>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, true>(
      dest, destOffset, source, sourceOffset, bytes,
      signalDest, signalDestOffset, signalValue, signalOp, pe, qpId);
}

template <>
inline __device__ void ShmemPutMemNbiSignalWarpKernel<application::TransportType::PROXY, true>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, true>(
      dest, destOffset, source, sourceOffset, bytes,
      signalDest, signalDestOffset, signalValue, signalOp, pe, qpId);
}

template <>
inline __device__ void ShmemPutMemNbiSignalWarpKernel<application::TransportType::PROXY, false>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, true>(
      dest, destOffset, source, sourceOffset, bytes,
      signalDest, signalDestOffset, signalValue, signalOp, pe, qpId);
}

template <>
inline __device__ void ShmemPutMemNbiSignalBlockKernel<application::TransportType::PROXY, true>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, true>(
      dest, destOffset, source, sourceOffset, bytes,
      signalDest, signalDestOffset, signalValue, signalOp, pe, qpId);
}

template <>
inline __device__ void ShmemPutMemNbiSignalBlockKernel<application::TransportType::PROXY, false>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, true>(
      dest, destOffset, source, sourceOffset, bytes,
      signalDest, signalDestOffset, signalValue, signalOp, pe, qpId);
}

// ---------------------------------------------------------------------------
// ShmemAtomicSizeNonFetchThreadKernel<PROXY> (SymmMemObjPtr)
// ---------------------------------------------------------------------------
template <>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes,
    core::atomicType amoType, int pe, int qpId) {
  ProxyGpuStates* ps = GetGlobalProxyStatePtr();
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

template <>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes,
    core::atomicType amoType, int pe, int qpId) {
  ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::PROXY>(
      dest, destOffset, val, bytes, amoType, pe, qpId);
}

// ---------------------------------------------------------------------------
// ShmemAtomicTypeFetchThreadKernel<PROXY, T> (SymmMemObjPtr)
// ---------------------------------------------------------------------------
#define DEFINE_PROXY_ATOMIC_FETCH_THREAD(T)                                                    \
  template <>                                                                                  \
  inline __device__ T                                                                          \
  ShmemAtomicTypeFetchThreadKernel<application::TransportType::PROXY, T>(                      \
      const application::SymmMemObjPtr dest, size_t destOffset, void* val, void* compare,      \
      size_t bytes, core::atomicType amoType, int pe, int qpId) {                               \
    ProxyGpuStates* ps = GetGlobalProxyStatePtr();                                              \
    GpuStates* gs = GetGlobalGpuStatesPtr();                                                   \
    int epIndex = pe * gs->numQpPerPe + (qpId % gs->numQpPerPe);                               \
    volatile core::ProxyRing* ring = ProxyRingForEp(ps, epIndex);                              \
    uintptr_t raddr;                                                                           \
    uint32_t rkey;                                                                             \
    if (gs->useVMMHeap) {                                                                      \
      uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dest->localPtr) + destOffset;            \
      VmmLookupRemote(dstAddr, pe, raddr, rkey);                                               \
    } else {                                                                                   \
      raddr = dest->peerPtrs[pe] + destOffset;                                                 \
      rkey = dest->peerRkeys[pe];                                                              \
    }                                                                                          \
    core::IbufHandle& ibuf = gs->rdmaEndpoints[epIndex].atomicIbuf;                            \
    uint64_t atomicVal = 0;                                                                    \
    memcpy(&atomicVal, val, bytes <= 8 ? bytes : 8);                                           \
    uint64_t result = core::ProxyPostAtomicFetch(ring, epIndex, raddr, rkey,                   \
                                                  atomicVal, ibuf.lkey, ibuf.addr);            \
    T retVal;                                                                                  \
    memcpy(&retVal, &result, sizeof(T));                                                       \
    return retVal;                                                                             \
  }

DEFINE_PROXY_ATOMIC_FETCH_THREAD(uint32_t)
DEFINE_PROXY_ATOMIC_FETCH_THREAD(uint64_t)
DEFINE_PROXY_ATOMIC_FETCH_THREAD(int32_t)
DEFINE_PROXY_ATOMIC_FETCH_THREAD(int64_t)
#undef DEFINE_PROXY_ATOMIC_FETCH_THREAD

#define DEFINE_PROXY_ATOMIC_FETCH_WARP(T)                                                      \
  template <>                                                                                  \
  inline __device__ T                                                                          \
  ShmemAtomicTypeFetchWarpKernel<application::TransportType::PROXY, T>(                        \
      const application::SymmMemObjPtr dest, size_t destOffset, void* val, void* compare,      \
      size_t bytes, core::atomicType amoType, int pe, int qpId) {                               \
    return ShmemAtomicTypeFetchThreadKernel<application::TransportType::PROXY, T>(              \
        dest, destOffset, val, compare, bytes, amoType, pe, qpId);                             \
  }

DEFINE_PROXY_ATOMIC_FETCH_WARP(uint32_t)
DEFINE_PROXY_ATOMIC_FETCH_WARP(uint64_t)
DEFINE_PROXY_ATOMIC_FETCH_WARP(int32_t)
DEFINE_PROXY_ATOMIC_FETCH_WARP(int64_t)
#undef DEFINE_PROXY_ATOMIC_FETCH_WARP

// ---------------------------------------------------------------------------
// ShmemGetMemNbi<PROXY> — not supported (proxy is write-only)
// ---------------------------------------------------------------------------
template <>
inline __device__ void ShmemGetMemNbiThreadKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe,
    int qpId) {
  assert(false);
}

template <>
inline __device__ void ShmemGetMemNbiWarpKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe,
    int qpId) {
  assert(false);
}

template <>
inline __device__ void ShmemGetMemNbiBlockKernel<application::TransportType::PROXY>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe,
    int qpId) {
  assert(false);
}

// ---------------------------------------------------------------------------
// Address-based overloads — stubs (EP uses SymmMemObjPtr APIs, not these)
// ---------------------------------------------------------------------------
template <> inline __device__ void ShmemPutMemNbiThreadKernel<application::TransportType::PROXY>(
    const void* d, const void* s, size_t b, int pe, int q) { assert(false); }
template <> inline __device__ void ShmemPutMemNbiWarpKernel<application::TransportType::PROXY>(
    const void* d, const void* s, size_t b, int pe, int q) { assert(false); }
template <> inline __device__ void ShmemPutMemNbiBlockKernel<application::TransportType::PROXY>(
    const void* d, const void* s, size_t b, int pe, int q) { assert(false); }
template <> inline __device__ void ShmemPutSizeImmNbiThreadKernel<application::TransportType::PROXY>(
    const void* d, void* v, size_t b, int pe, int q) { assert(false); }
template <> inline __device__ void ShmemPutSizeImmNbiWarpKernel<application::TransportType::PROXY>(
    const void* d, void* v, size_t b, int pe, int q) { assert(false); }
template <> inline __device__ void ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::PROXY>(
    const void* d, void* v, size_t b, core::atomicType a, int pe, int q) { assert(false); }
template <> inline __device__ void ShmemAtomicSizeNonFetchWarpKernel<application::TransportType::PROXY>(
    const void* d, void* v, size_t b, core::atomicType a, int pe, int q) { assert(false); }
template <> inline __device__ void ShmemGetMemNbiThreadKernel<application::TransportType::PROXY>(
    void* d, const void* s, size_t b, int pe, int q) { assert(false); }
template <> inline __device__ void ShmemGetMemNbiWarpKernel<application::TransportType::PROXY>(
    void* d, const void* s, size_t b, int pe, int q) { assert(false); }
template <> inline __device__ void ShmemGetMemNbiBlockKernel<application::TransportType::PROXY>(
    void* d, const void* s, size_t b, int pe, int q) { assert(false); }

// Signal address-based stubs
template <> inline __device__ void ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, true>(
    const void* d, const void* s, size_t b, const void* sd, uint64_t sv,
    core::atomicType so, int pe, int q) { assert(false); }
template <> inline __device__ void ShmemPutMemNbiSignalThreadKernel<application::TransportType::PROXY, false>(
    const void* d, const void* s, size_t b, const void* sd, uint64_t sv,
    core::atomicType so, int pe, int q) { assert(false); }
template <> inline __device__ void ShmemPutMemNbiSignalWarpKernel<application::TransportType::PROXY, true>(
    const void* d, const void* s, size_t b, const void* sd, uint64_t sv,
    core::atomicType so, int pe, int q) { assert(false); }
template <> inline __device__ void ShmemPutMemNbiSignalWarpKernel<application::TransportType::PROXY, false>(
    const void* d, const void* s, size_t b, const void* sd, uint64_t sv,
    core::atomicType so, int pe, int q) { assert(false); }
template <> inline __device__ void ShmemPutMemNbiSignalBlockKernel<application::TransportType::PROXY, true>(
    const void* d, const void* s, size_t b, const void* sd, uint64_t sv,
    core::atomicType so, int pe, int q) { assert(false); }
template <> inline __device__ void ShmemPutMemNbiSignalBlockKernel<application::TransportType::PROXY, false>(
    const void* d, const void* s, size_t b, const void* sd, uint64_t sv,
    core::atomicType so, int pe, int q) { assert(false); }

// AtomicFetch address-based stubs
#define DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Scope, T)                                          \
  template <> inline __device__ T                                                              \
  ShmemAtomicTypeFetch##Scope##Kernel<application::TransportType::PROXY, T>(                   \
      const void* d, void* v, void* c, size_t b, core::atomicType a, int pe, int q) {          \
    assert(false); return T{}; }

DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Thread, uint32_t)
DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Thread, uint64_t)
DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Thread, int32_t)
DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Thread, int64_t)
DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Warp, uint32_t)
DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Warp, uint64_t)
DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Warp, int32_t)
DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB(Warp, int64_t)
#undef DEFINE_PROXY_ATOMIC_FETCH_ADDR_STUB

}  // namespace shmem
}  // namespace mori

#endif  // __HIPCC__ && MORI_PROXY_ENABLED
