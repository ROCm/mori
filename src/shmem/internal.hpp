#pragma once

#include <iostream>
#include <memory>
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
  application::Context* commContext{nullptr};
};

struct MemoryStates {
  application::SymmMemManager* symmMemMgr{nullptr};
  application::RdmaMemoryRegionManager* mrMgr{nullptr};
};

enum ShmemStatesStatus {
  New = 0,
  Initialized = 1,
  Finalized = 2,
};

struct ShmemStates {
  ShmemStatesStatus status{ShmemStatesStatus::New};
  BootStates* bootStates{nullptr};
  RdmaStates* rdmaStates{nullptr};
  MemoryStates* memoryStates{nullptr};

  // This is a temporary API for debugging only
  void CheckStatusValid() {
    if (status == ShmemStatesStatus::New) {
      std::cout
          << "Shmem state is not initialized, initialize it by calling ShmemMpiIntialize first."
          << std::endl;
      assert(false);
    }
    if (status == ShmemStatesStatus::Finalized) {
      std::cout << "Shmem state has been finalized." << std::endl;
      assert(false);
    }
  }
};

constexpr int MaxRdmaEndpointNum = 1024;

struct GpuStates {
  int rank{-1};
  int worldSize{-1};
  application::TransportType* transportTypes{nullptr};
  application::RdmaEndpoint* rdmaEndpoints{nullptr};
  uint32_t* endpointLock{nullptr};
};

extern __constant__ GpuStates globalGpuStates;

static __device__ GpuStates* GetGlobalGpuStatesPtr() { return &globalGpuStates; }

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
