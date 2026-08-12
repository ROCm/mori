// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#pragma once

#include <cstdint>

namespace mori {
namespace shmem {

static constexpr int PROXY_STATE_MAX_NICS = 8;

struct ProxyGpuState {
  bool active{false};
  void* rings[PROXY_STATE_MAX_NICS]{};
  uint32_t quietHead[PROXY_STATE_MAX_NICS]{};
  int numRings{0};
  int numNics{0};
  int localGpuIdx{0};
  int numQpPerPe{4};
};

}  // namespace shmem
}  // namespace mori
