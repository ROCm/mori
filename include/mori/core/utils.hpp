#pragma once

namespace mori {
namespace core {

/* -------------------------------------------------------------------------- */
/*                              Atomic Operations                             */
/* -------------------------------------------------------------------------- */
template <typename T>
__device__ T atomicLoadSeqCst(T* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T>
__device__ T atomicLoadRelaxed(T* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T>
__device__ T atomicLoadRelaxedSystem(T* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
__device__ T atomicLoadSeqCstSystem(T* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
__device__ void atomicStoreRelaxed(T* ptr, T val) {
  return __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T>
__device__ void atomicStoreSeqCst(T* ptr, T val) {
  return __hip_atomic_store(ptr, val, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T>
__device__ void atomicStoreSeqCstSystem(T* ptr, T val) {
  return __hip_atomic_store(ptr, val, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
}

/* -------------------------------------------------------------------------- */
/*                                    Match                                   */
/* -------------------------------------------------------------------------- */
template <typename T>
constexpr __device__ T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

}  // namespace core
}  // namespace mori