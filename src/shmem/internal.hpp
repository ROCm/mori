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

struct RdmaStates {
  application::RdmaContext* context{nullptr};
  application::RdmaDeviceContext* deviceContext{nullptr};
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
