#pragma once

namespace mori {
namespace core {

/* ---------------------------------------------------------------------------------------------- */
/*                                             Thread                                             */
/* ---------------------------------------------------------------------------------------------- */

inline __device__ int DeviceWarpSize() { return warpSize; }

inline __device__ int FlatBlockSize() { return blockDim.z * blockDim.y * blockDim.x; }

inline __device__ int FlatBlockThreadId() {
  return (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + threadIdx.x;
}

inline __device__ int FlatBlockWarpNum() { return FlatBlockSize() / DeviceWarpSize(); }

inline __device__ int FlatBlockWarpId() { return FlatBlockThreadId() / DeviceWarpSize(); }

inline __device__ int WarpLaneId() { return FlatBlockThreadId() & (DeviceWarpSize() - 1); }

/* ---------------------------------------------------------------------------------------------- */
/*                                        Atomic Operations                                       */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ T AtomicLoadSeqCst(T* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T>
inline __device__ T AtomicLoadSeqCstSystem(T* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
inline __device__ T AtomicLoadRelaxed(T* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T>
inline __device__ T AtomicLoadRelaxedSystem(T* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
inline __device__ void AtomicStoreRelaxed(T* ptr, T val) {
  return __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T>
inline __device__ void AtomicStoreRelaxedSystem(T* ptr, T val) {
  return __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
inline __device__ void AtomicStoreSeqCst(T* ptr, T val) {
  return __hip_atomic_store(ptr, val, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T>
inline __device__ void AtomicStoreSeqCstSystem(T* ptr, T val) {
  return __hip_atomic_store(ptr, val, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
__device__ T AtomicCompareExchange(T* address, T* compare, T val) {
  __hip_atomic_compare_exchange_strong(address, compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return *compare;
}

template <typename T>
__device__ T AtomicCompareExchangeSystem(T* address, T* compare, T val) {
  __hip_atomic_compare_exchange_strong(address, compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  return *compare;
}

/* -------------------------------------------------------------------------- */
/*                                    Match                                   */
/* -------------------------------------------------------------------------- */
template <typename T>
constexpr inline __device__ __host__ T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

/* -------------------------------------------------------------------------- */
/*                                    Lock                                    */
/* -------------------------------------------------------------------------- */
// TODO: Whether to use GPU lock in lock.hpp
__device__ inline void AcquireLock(uint32_t* lockVar) {  
    while (atomicCAS(lockVar, 0, 1) != 0) {  
    }  
}  

__device__ inline void ReleaseLock(uint32_t* lockVar) {  
    atomicExch(lockVar, 0);  
}  

}  // namespace core
}  // namespace mori