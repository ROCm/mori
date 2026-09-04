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
// ---------------------------------------------------------------------------
// Host side of the v1 internode kernels on cco: what the caller asks for, how
// that becomes an EpInterNodeCfg, and the Specs.
//
// Same shape as ep_spec.hpp: a Request, a Cfg, and KernelSpec subclasses that
// supply RenderSource/Geometry and inherit hashing, caching, module load and
// launch. Eight entries instead of two, because the v1 dispatch and combine
// sequences are several passes each and every pass is its own module.
//
// Host-only, and the device TU must never include it -- everything shared with
// the kernel lives in ep_internode_cfg.hpp.
// ---------------------------------------------------------------------------
#pragma once

#include <string>
#include <vector>

#include "mori/jit/v2/spec.hpp"
#include "mori/ops/dispatch_combine_v2/ep_internode_cfg.hpp"

namespace mori {
namespace ops {
namespace v2 {

// ---------------------------------------------------------------------------
// The compiled specialisation.
//
// One difference from EpCfg is deliberate. There, launch geometry is part of the
// Cfg *and* reaches the source, through Render(kCfg) and
// __launch_bounds__(EpBlockThreads(kCfg)) -- so a geometry sweep really is a
// hipcc sweep. Here geometry is in the Cfg (KernelSpec::Geometry needs it, and
// the plan reports it) but is NOT rendered: only kernelCfg and dtype reach the
// TU. Two cfgs differing only in geometry therefore render identical text, land
// in the same content-addressed cache entry, and cost no extra compile. v1
// retunes blocks and warps per token count, so that difference is the point.
// ---------------------------------------------------------------------------
struct EpInterNodeCfg {
  // ---- rendered: this is the specialisation ----
  EpInterNodeKernelCfg kernelCfg{};
  EpInterNodeDType dtype{EpInterNodeDType::Bf16};
  // Only meaningful for the LL dispatch/combine pair, which the standard-MoE
  // adapter specialises on; the other six entries ignore it.
  bool enableStdMoE{false};

  // ---- launch geometry (host-derived; see MakeEpInterNodeCfg). NOT rendered. ----
  int blockNum{64};
  int warpPerBlock{8};
  // v1's third grid dimension: the staging/RDMA passes are sized separately.
  int rdmaBlockNum{8};
  // The passes that fan out over the whole device (CopyToStaging, CombineSync,
  // CombineAll) take their grid from this rather than from blockNum. A device
  // property, so the host has to supply it -- Geometry() must stay a pure
  // function of the Cfg to remain correct when cross-compiling.
  int mpCount{64};
  int waveSize{64};
};

template <typename Self, typename Visit>
inline void VisitFields(Self& c, const EpInterNodeCfg& d, Visit&& v) {
#define MORI_FIELD(x) v(#x, c.x, d.x)
  MORI_FIELD(kernelCfg);
  MORI_FIELD(dtype);
  MORI_FIELD(enableStdMoE);
  MORI_FIELD(blockNum);
  MORI_FIELD(warpPerBlock);
  MORI_FIELD(rdmaBlockNum);
  MORI_FIELD(mpCount);
  MORI_FIELD(waveSize);
#undef MORI_FIELD
}

MORI_JIT_ASSERT_FIELD_COUNT(
    EpInterNodeCfg, 8, "added an EpInterNodeCfg field -- update VisitFields(EpInterNodeCfg) too");

// The whole Cfg as text, for the plan's `info` and for logs. NOT the source
// text: RenderSource renders kernelCfg alone, because geometry must not enter
// the cache key. Rendered here anyway so `info` reports what the host actually
// resolved, geometry included.
inline std::string Render(const EpInterNodeCfg& c) {
  const EpInterNodeCfg d{};
  mori::jit::v2::Fields f;
  VisitFields(c, d, [&f](const char* name, const auto& value, const auto& dflt) {
    f.Put(name, value, dflt);
  });
  return mori::jit::v2::BraceInit("::mori::ops::v2::EpInterNodeCfg", f);
}

inline bool operator==(const EpInterNodeCfg& a, const EpInterNodeCfg& b) {
  return Render(a) == Render(b);
}

// What the plan's `info` reports. The shape fields are flattened to the top
// level rather than reported as one kernelCfg= brace initialiser: this is the
// surface a caller reads a resolved plan out of, and EpCfg -- being flat -- puts
// every field there individually. The names do not collide with the geometry
// ones, so no prefix is needed to keep them apart.
inline std::string Describe(const EpInterNodeCfg& c) {
  std::string out;
  auto emit = [&out](const char* name, const auto& value) {
    using mori::jit::v2::RenderValue;  // ADL for the Ep types, jit's for scalars
    out += name;
    out += "=";
    out += RenderValue(value);
    out += "\n";
  };
  VisitFields(c.kernelCfg, c.kernelCfg,
              [&emit](const char* n, const auto& v, const auto&) { emit(n, v); });
  VisitFields(c, c, [&emit](const char* n, const auto& v, const auto&) {
    if (std::string(n) == "kernelCfg") return;  // already flattened above
    emit(n, v);
  });
  return out;
}

// ---------------------------------------------------------------------------
// Shared arithmetic. One definition for the host to size the launch with; these
// are the formulas launch.cpp uses for the AOT symbols, so the JIT path cannot
// under-reserve LDS for a kernel that stages into it.
// ---------------------------------------------------------------------------
constexpr int EpInterNodeBlockThreads(const EpInterNodeCfg& c) {
  return c.warpPerBlock * c.waveSize;
}

// Dispatch's index arrays: per-warp destination counts plus the per-expert
// tallies. Mirrors dispatch_shared_mem() in launch.cpp.
constexpr int EpInterNodeDispatchSharedBytes(const EpInterNodeCfg& c) {
  const int wpb = c.warpPerBlock;
  return (c.kernelCfg.worldSize * wpb + c.kernelCfg.numExpertPerRank * wpb +
          c.kernelCfg.numExpertPerRank) *
         static_cast<int>(sizeof(mori::moe::index_t));
}

// Combine's pointer arrays: one per warp for the topk sources, a second for the
// weights, a third for the scales when the quant type is blockwise. Mirrors
// combine_shared_mem() in launch.cpp with use_weight_ptrs=true.
constexpr int EpInterNodeCombineSharedBytes(const EpInterNodeCfg& c) {
  const bool blockwise = c.kernelCfg.quantType == mori::moe::QuantType::Fp8BlockwiseQuant ||
                         c.kernelCfg.quantType == mori::moe::QuantType::Fp4BlockwiseQuant;
  const int ptrArrays = 2 + (blockwise ? 1 : 0);
  return c.warpPerBlock * c.kernelCfg.numExpertPerToken * ptrArrays * 8;
}

// ---------------------------------------------------------------------------
// What a caller asks for. Crosses the language boundary as (name, value) pairs
// driven by the VisitFields walk below, so there is no second struct to keep in
// step on the Python side.
// ---------------------------------------------------------------------------
struct EpInterNodeRequest {
  // shape -> EpInterNodeKernelCfg
  int worldSize = 8;
  int hiddenDim = 4096;
  int scaleDim = 32;
  int scaleTypeSize = 1;
  int maxTokenTypeSize = 4;
  int maxNumInpTokenPerRank = 128;
  int numExpertPerRank = 1;
  int numExpertPerToken = 2;
  int maxTotalRecvTokens = 0;
  int gpuPerNode = 8;
  int numQpPerPe = 1;
  mori::moe::QuantType quantType = mori::moe::QuantType::None;
  // specialisation
  EpInterNodeDType dtype = EpInterNodeDType::Bf16;
  bool enableStdMoE = false;
  // geometry; 0 = the placeholder MakeEpInterNodeCfg picks
  int blockNum = 0;
  int warpPerBlock = 0;
  int rdmaBlockNum = 0;
  int mpCount = 0;
};

template <typename Self, typename Visit>
inline void VisitFields(Self& r, const EpInterNodeRequest& d, Visit&& v) {
#define MORI_FIELD(x) v(#x, r.x, d.x)
  MORI_FIELD(worldSize);
  MORI_FIELD(hiddenDim);
  MORI_FIELD(scaleDim);
  MORI_FIELD(scaleTypeSize);
  MORI_FIELD(maxTokenTypeSize);
  MORI_FIELD(maxNumInpTokenPerRank);
  MORI_FIELD(numExpertPerRank);
  MORI_FIELD(numExpertPerToken);
  MORI_FIELD(maxTotalRecvTokens);
  MORI_FIELD(gpuPerNode);
  MORI_FIELD(numQpPerPe);
  MORI_FIELD(quantType);
  MORI_FIELD(dtype);
  MORI_FIELD(enableStdMoE);
  MORI_FIELD(blockNum);
  MORI_FIELD(warpPerBlock);
  MORI_FIELD(rdmaBlockNum);
  MORI_FIELD(mpCount);
#undef MORI_FIELD
}

MORI_JIT_ASSERT_FIELD_COUNT(
    EpInterNodeRequest, 18,
    "added an EpInterNodeRequest field -- update VisitFields(EpInterNodeRequest) too");

std::string EpInterNodeRequestSchema();

// Which of the v1 internode entry points. The dispatch and combine sequences are
// several kernels each; every one of them is its own JIT module.
enum class EpInterNodeKernel {
  CopyToStaging,
  Dispatch,
  DispatchLL,
  CombineSync,
  CombineSyncBarrier,
  Combine,
  CombineLL,
  CombineAll,
};

// Geometry rules differ per pass, so the Request -> Cfg step is told which pass
// it is building for -- the same reason MakeEpCfg takes EpKernelKind.
EpInterNodeCfg MakeEpInterNodeCfg(const std::string& arch, const EpInterNodeRequest& req,
                                  EpInterNodeKernel kind);

// Rendering, naming and geometry are shared by all eight Specs. Exposed so a
// test can exercise them per kernel without going through eight class names.
std::string EpInterNodeEntryName(const EpInterNodeCfg& cfg, EpInterNodeKernel kind);
std::string EpInterNodeRenderSource(const EpInterNodeCfg& cfg, EpInterNodeKernel kind);
mori::jit::v2::LaunchGeometry EpInterNodeGeometry(const EpInterNodeCfg& cfg,
                                                  EpInterNodeKernel kind);

// ---------------------------------------------------------------------------
// The eight Specs. Same Cfg, same Args; they differ in which *_body the TU
// instantiates and in launch geometry.
//
// Declared by macro because the eight are identical: KernelSpec needs kName as a
// constexpr string, so one class template over EpInterNodeKernel would need a parallel
// constexpr name table -- more machinery than the lines it would save.
// ---------------------------------------------------------------------------
#define MORI_EP_INTERNODE_DECLARE_SPEC(ClassName, planName)                       \
  class ClassName : public mori::jit::v2::KernelSpec<ClassName, EpInterNodeCfg> { \
   public:                                                                        \
    using Args = EpInterNodeCcoArgs;                                              \
    static constexpr const char* kName = planName;                                \
    static std::string EntryName(const Cfg& cfg);                                 \
    static std::string RenderSource(const Cfg& cfg);                              \
    static mori::jit::v2::LaunchGeometry Geometry(const Cfg& cfg);                \
    static const std::vector<std::string>& SourceDeps();                          \
  }

MORI_EP_INTERNODE_DECLARE_SPEC(EpInterNodeCopyToStagingSpec, "ep_internode_copystaging");
MORI_EP_INTERNODE_DECLARE_SPEC(EpInterNodeDispatchSpec, "ep_internode_dispatch");
MORI_EP_INTERNODE_DECLARE_SPEC(EpInterNodeDispatchLLSpec, "ep_internode_dispatch_ll");
MORI_EP_INTERNODE_DECLARE_SPEC(EpInterNodeCombineSyncSpec, "ep_internode_combinesync");
MORI_EP_INTERNODE_DECLARE_SPEC(EpInterNodeCombineSyncBarrierSpec,
                               "ep_internode_combinesyncbarrier");
MORI_EP_INTERNODE_DECLARE_SPEC(EpInterNodeCombineSpec, "ep_internode_combine");
MORI_EP_INTERNODE_DECLARE_SPEC(EpInterNodeCombineLLSpec, "ep_internode_combine_ll");
MORI_EP_INTERNODE_DECLARE_SPEC(EpInterNodeCombineAllSpec, "ep_internode_combineall");

#undef MORI_EP_INTERNODE_DECLARE_SPEC

}  // namespace v2
}  // namespace ops
}  // namespace mori
