#pragma once

#include "hip/hip_runtime.h"

namespace mori {
namespace application {

struct P2PMemoryRegion {
  uintptr_t addr;
  size_t length;
  hipIpcMemHandle_t ipcHandle;
};

}  // namespace application
}  // namespace mori
