// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
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
