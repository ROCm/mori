#pragma once

namespace mori {
namespace core {

class GpuLock {
 public:
  __device__ GpuLock(uint32_t* lockMem) : lock(lockMem) {}
  __device__ ~GpuLock() = default;

  __device__ void Lock() {
    while (!atomicCAS(lock, 0, 1)) {
    }
    __threadfence_system();
  }

  __device__ void Unlock() { atomicCAS(lock, 1, 0); }

 private:
  uint32_t* lock{nullptr};
};

}  // namespace core
}  // namespace mori