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

/* -------------------------------------------------------------------------- */
/*                                    Match                                   */
/* -------------------------------------------------------------------------- */
template <typename T>
constexpr __device__ T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

}  // namespace core
}  // namespace mori