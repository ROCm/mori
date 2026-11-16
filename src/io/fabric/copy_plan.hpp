// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <variant>
#include <vector>

#include "src/io/kernels/fabric_copy_abi.hpp"

namespace mori::io::fabric {
struct CopySegment {
  uint64_t src;
  uint64_t dst;
  uint64_t size;
};
struct CopyDispatch {
  std::variant<FabricCopyArgs<kSingleK>, FabricCopyArgs<kSmallK>, FabricCopyArgs<kLargeK>> args;
  uint32_t grid;
};
struct CopyPlan {
  std::vector<CopySegment> segments;  // sorted, merged, nonzero; also used by memcpy fallback
  std::vector<CopyDispatch> dispatches;
  std::string error;
};
// remoteIsSrc selects the remote-address sort key, preserving read/write merge semantics.
// maxBlocks must be positive. No allocation addresses are dereferenced.
CopyPlan PlanCopy(std::vector<CopySegment> segments, uint32_t maxBlocks, bool kernelLoaded,
                  bool remoteIsSrc = false);
}  // namespace mori::io::fabric
