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

inline __device__ int FlatThreadId() {
  return FlatBlockThreadId() +
         (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) *
             FlatBlockSize();
}

inline __device__ int FlatBlockWarpNum() { return FlatBlockSize() / DeviceWarpSize(); }

inline __device__ int FlatBlockWarpId() { return FlatBlockThreadId() / DeviceWarpSize(); }

inline __device__ int WarpLaneId() { return FlatBlockThreadId() & (DeviceWarpSize() - 1); }

inline __device__ bool IsThreadZeroInBlock() {
  return (FlatBlockThreadId() % DeviceWarpSize()) == 0;
}

inline __device__ uint64_t GetActiveLaneMask() { return __ballot(true); }

inline __device__ unsigned int GetActiveLaneCount(uint64_t activeLaneMask) {
  return __popcll(activeLaneMask);
}

inline __device__ unsigned int GetActiveLaneCount() {
  return GetActiveLaneCount(GetActiveLaneMask());
}

inline __device__ unsigned int GetActiveLaneNum(uint64_t activeLaneMask) {
  return __popcll(activeLaneMask & __lanemask_lt());
}

inline __device__ unsigned int GetActiveLaneNum() { return GetActiveLaneNum(GetActiveLaneMask()); }

inline __device__ int GetFirstActiveLaneID(uint64_t activeLaneMask) {
  return __ffsll((unsigned long long int)activeLaneMask) - 1;
}

inline __device__ int GetFirstActiveLaneID() { return GetFirstActiveLaneID(GetActiveLaneMask()); }

inline __device__ bool IsFirstActiveLane(uint64_t activeLaneMask) {
  return GetActiveLaneNum(activeLaneMask) == 0;
}

inline __device__ bool IsFirstActiveLane() { return IsFirstActiveLane(GetActiveLaneMask()); }

inline __device__ bool IsLastActiveLane(uint64_t activeLaneMask) {
  return GetActiveLaneNum(activeLaneMask) == GetActiveLaneCount(activeLaneMask) - 1;
}

inline __device__ bool IsLastActiveLane() { return IsLastActiveLane(GetActiveLaneMask()); }

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

template <typename T>
constexpr inline __device__ __host__ T IsPowerOf2(T x) {
  return (x > 0) && ((x & (x - 1)) == 0);
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