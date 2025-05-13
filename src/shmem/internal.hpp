#pragma once

#include <mutex>
#include <vector>

#include "mori/application/application.hpp"
#include "mori/application/bootstrap/bootstrap.hpp"

namespace mori {
namespace shmem {

struct BootStates {
  int rank{0};
  int worldSize{0};
  application::BootstrapNetwork* bootNet{nullptr};
};

using RdmaEndpointList = std::vector<application::RdmaEndpoint>;
using RdmaEndpointHandleList = std::vector<application::RdmaEndpointHandle>;

struct RdmaStates {
  application::RdmaContext* context{nullptr};
  application::RdmaDeviceContext* deviceContext{nullptr};
  RdmaEndpointList localEps;
  application::RdmaEndpoint* localEpsGpu;
  uint32_t* epCqLockMemGpu;
  std::vector<RdmaEndpointHandleList> remoteEpHandles;
};

struct MemoryStates {
  application::SymmMemManager* symmMemMgr{nullptr};
  application::MemoryRegionManager* mrMgr{nullptr};
};

struct ShmemStates {
  BootStates* bootStates{nullptr};
  RdmaStates* rdmaStates{nullptr};
  MemoryStates* memoryStates{nullptr};
};

constexpr int MaxRdmaEndpointNum = 1024;

struct GpuStates {
  int rank;
  int worldSize;
  application::RdmaEndpoint* epsStartAddr;
  uint32_t* epCqLockMemGpu;
};

extern __constant__ GpuStates globalGpuStates;

class ShmemStatesSingleton {
 public:
  ShmemStatesSingleton(const ShmemStatesSingleton& obj) = delete;

  static ShmemStates* GetInstance() {
    static ShmemStates states;
    return &states;
  }
};

}  // namespace shmem
}  // namespace mori
