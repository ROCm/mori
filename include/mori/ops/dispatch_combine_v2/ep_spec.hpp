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
//
// Host side of the EP intranode kernels: what the caller asks for, how that
// becomes an EpCfg, and the two Specs. Two kernels, one Cfg -- they run over
// the same arena and the same shape, and differ only in launch geometry.

#pragma once

#include <string>
#include <vector>

#include "mori/jit/v2/spec.hpp"
#include "mori/ops/dispatch_combine_v2/ep_cfg.hpp"

namespace mori {
namespace ops {
namespace v2 {

// What a caller asks for. Crosses the language boundary as (name, value) pairs
// driven by the VisitFields walk below, so there is no second struct to keep in
// step on the Python side.
struct EpRequest {
  int worldSize = 8;
  int hiddenDim = 7168;
  int maxTokPerRank = 128;
  int numExpertPerRank = 8;
  int numExpertPerToken = 8;
  int maxRecv = 0;
  EpDType dtype = EpDType::Bf16;
  bool useWeights = true;
  int blockNum = 0;      // 0 = arch default
  int warpPerBlock = 0;  // 0 = arch default
};

template <typename Self, typename Visit>
inline void VisitFields(Self& r, const EpRequest& d, Visit&& v) {
#define MORI_FIELD(x) v(#x, r.x, d.x)
  MORI_FIELD(worldSize);
  MORI_FIELD(hiddenDim);
  MORI_FIELD(maxTokPerRank);
  MORI_FIELD(numExpertPerRank);
  MORI_FIELD(numExpertPerToken);
  MORI_FIELD(maxRecv);
  MORI_FIELD(dtype);
  MORI_FIELD(useWeights);
  MORI_FIELD(blockNum);
  MORI_FIELD(warpPerBlock);
#undef MORI_FIELD
}

MORI_JIT_ASSERT_FIELD_COUNT(EpRequest, 10,
                            "added an EpRequest field -- update VisitFields(EpRequest) too");

std::string EpRequestSchema();

// Which kernel the geometry defaults are for: dispatch is copy-bound and wants
// many warps, combine is reduction-bound and wants few (v1's tuned split).
enum class EpKernelKind { Dispatch, Combine };

EpCfg MakeEpCfg(const std::string& arch, const EpRequest& req, EpKernelKind kind);

// ---------------------------------------------------------------------------
// The two Specs. Same Cfg, same Args, different body and different geometry.
// ---------------------------------------------------------------------------
class EpDispatchSpec : public mori::jit::v2::KernelSpec<EpDispatchSpec, EpCfg> {
 public:
  using Args = EpArgs;
  static constexpr const char* kName = "ep_dispatch";
  static std::string RenderSource(const Cfg& cfg);
  static mori::jit::v2::LaunchGeometry Geometry(const Cfg& cfg);
  static const std::vector<std::string>& SourceDeps();
};

class EpCombineSpec : public mori::jit::v2::KernelSpec<EpCombineSpec, EpCfg> {
 public:
  using Args = EpArgs;
  static constexpr const char* kName = "ep_combine";
  static std::string RenderSource(const Cfg& cfg);
  static mori::jit::v2::LaunchGeometry Geometry(const Cfg& cfg);
  static const std::vector<std::string>& SourceDeps();
};

}  // namespace v2
}  // namespace ops
}  // namespace mori
