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
// Host-only. Never includes the device kernel header -- that is what keeps this
// target compilable without hipcc.

#include "mori/ops/dispatch_combine_v2/ep_internode_spec.hpp"

#include <stdexcept>
#include <string>

#include "mori/jit/v2/toolchain.hpp"

namespace mori {
namespace ops {
namespace v2 {

namespace {

struct KernelDesc {
  const char* tag;   // goes in the entry name
  const char* body;  // the *_body function in ep_internode_kernel.hpp
  bool takesComm;    // false for the passes with no cross-node traffic
  bool takesStdMoE;  // the LL pair is additionally specialised on it
};

KernelDesc DescFor(EpInterNodeKernel k) {
  switch (k) {
    case EpInterNodeKernel::CopyToStaging:
      return {"copystaging", "EpDispatchCopyToStaging_body", false, false};
    case EpInterNodeKernel::Dispatch:
      return {"dispatch", "EpDispatchInterNodeV1Kernel_body", true, false};
    case EpInterNodeKernel::DispatchLL:
      return {"dispatch_ll", "EpDispatchInterNodeV1KernelLowLatency_body", true, true};
    case EpInterNodeKernel::CombineSync:
      return {"combinesync", "EpCombineSync_body", false, false};
    case EpInterNodeKernel::CombineSyncBarrier:
      return {"combinesyncbarrier", "EpCombineSyncBarrier_body", false, false};
    case EpInterNodeKernel::Combine:
      return {"combine", "EpCombineInterNodeV1Kernel_body", true, false};
    case EpInterNodeKernel::CombineLL:
      return {"combine_ll", "EpCombineInterNodeV1KernelLowLatency_body", true, true};
    case EpInterNodeKernel::CombineAll:
      return {"combineall", "EpCombineAll_body", false, false};
  }
  throw std::runtime_error("mori ep v1 jit: unknown kernel");
}

// Header subtrees whose contents invalidate a compiled module. Coarser than the
// real include graph on purpose (see IncludeTreeHash): over-invalidating costs a
// rebuild, under-invalidating ships stale code. The v1 kernel body still pulls
// common.hpp and convert.hpp out of the v1 tree, so that directory is in the set
// alongside the v2 one.
const std::vector<std::string>& EpInterNodeDeps() {
  static const std::vector<std::string> deps{"include/mori", "src/ops/dispatch_combine_v2",
                                             "src/ops/dispatch_combine", "src/cco"};
  return deps;
}

}  // namespace

// ---------------------------------------------------------------------------
// Request -> Cfg
// ---------------------------------------------------------------------------
EpInterNodeCfg MakeEpInterNodeCfg(const std::string& arch, const EpInterNodeRequest& req,
                                  EpInterNodeKernel kind) {
  EpInterNodeCfg c;
  c.kernelCfg.worldSize = req.worldSize;
  c.kernelCfg.hiddenDim = req.hiddenDim;
  c.kernelCfg.scaleDim = req.scaleDim;
  c.kernelCfg.scaleTypeSize = req.scaleTypeSize;
  c.kernelCfg.maxTokenTypeSize = req.maxTokenTypeSize;
  c.kernelCfg.maxNumInpTokenPerRank = req.maxNumInpTokenPerRank;
  c.kernelCfg.numExpertPerRank = req.numExpertPerRank;
  c.kernelCfg.numExpertPerToken = req.numExpertPerToken;
  c.kernelCfg.maxTotalRecvTokens = req.maxTotalRecvTokens;
  c.kernelCfg.gpuPerNode = req.gpuPerNode;
  c.kernelCfg.numQpPerPe = req.numQpPerPe;
  c.kernelCfg.quantType = req.quantType;

  c.dtype = req.dtype;
  c.enableStdMoE = req.enableStdMoE;

  c.waveSize = mori::jit::v2::WaveSizeForArch(arch);

  // Placeholders for a bare C++ caller only. v1 retunes blocks and warps per
  // token count, so a real caller always passes them; unlike the intranode
  // defaults these are not measured optima, they only have to be launchable.
  c.blockNum = 64;
  c.warpPerBlock = 8;
  c.rdmaBlockNum = 8;
  c.mpCount = 64;

  if (req.blockNum > 0) c.blockNum = req.blockNum;
  if (req.warpPerBlock > 0) c.warpPerBlock = req.warpPerBlock;
  if (req.rdmaBlockNum > 0) c.rdmaBlockNum = req.rdmaBlockNum;
  if (req.mpCount > 0) c.mpCount = req.mpCount;

  if (!EpInterNodeKernelCfgIsValid(c.kernelCfg)) {
    throw std::runtime_error(
        "mori ep v1: unusable config " + Render(c.kernelCfg) +
        "; every divisor must be positive and worldSize must be a multiple of gpuPerNode");
  }

  // Caught here rather than by hipModuleLaunchKernel, which reports it as a
  // generic launch failure with no mention of which knob was too large.
  const int threads = EpInterNodeBlockThreads(c);
  if (threads <= 0 || threads > 1024) {
    throw std::runtime_error("mori ep v1: warpPerBlock " + std::to_string(c.warpPerBlock) +
                             " x waveSize " + std::to_string(c.waveSize) + " = " +
                             std::to_string(threads) + " threads per block, which exceeds 1024");
  }
  (void)kind;
  return c;
}

std::string EpInterNodeRequestSchema() {
  mori::jit::v2::SchemaBuilder sb;
  const EpInterNodeRequest def{};
  VisitFields(def, def, [&sb](const char* n, const auto& val, const auto&) {
    using mori::jit::v2::WireTag;
    using mori::jit::v2::WireValue;
    sb.Add(n, WireTag(val), WireValue(val));
  });
  return sb.Str();
}

// ---------------------------------------------------------------------------
// Source rendering. The Cfg text IS the specialisation and IS the cache key --
// there is no other channel by which a config can reach hipcc. Geometry is
// deliberately absent from the rendered text; see the note on EpInterNodeCfg.
// ---------------------------------------------------------------------------
std::string EpInterNodeEntryName(const EpInterNodeCfg& cfg, EpInterNodeKernel kind) {
  const KernelDesc d = DescFor(kind);
  std::string s = "mori_ep_internode_";
  s += d.tag;
  s += '_';
  s += EpInterNodeDTypeTag(cfg.dtype);
  if (d.takesStdMoE && cfg.enableStdMoE) s += "_stdmoe";
  return s;
}

std::string EpInterNodeRenderSource(const EpInterNodeCfg& cfg, EpInterNodeKernel kind) {
  // Once the cfg is an NTTP its divisors are literals, so a zero here is a
  // division by constant zero inside hipcc -- diagnosed, if at all, against
  // generated source no one can trace back to the caller. MakeEpInterNodeCfg already
  // rejects it, but a Cfg can also be aggregate-initialised by hand, and this is
  // the one funnel every compile goes through.
  if (!EpInterNodeKernelCfgIsValid(cfg.kernelCfg)) {
    throw std::runtime_error("mori ep v1 jit: unusable config " + Render(cfg.kernelCfg));
  }

  const KernelDesc d = DescFor(kind);
  const std::string entry = EpInterNodeEntryName(cfg, kind);

  // The two names the entry macros expand against, in the same order and with
  // the same spelling ep_spec.cpp uses for the intranode kernels.
  std::string src =
      "// mori jit — generated, do not edit.\n"
      "#include \"src/ops/dispatch_combine_v2/ep_internode_kernel.hpp\"\n"
      "constexpr ::mori::ops::v2::EpInterNodeKernelCfg kConfig = ";
  src += Render(cfg.kernelCfg);
  src += ";\nusing TokT = ";
  src += EpInterNodeDTypeName(cfg.dtype);
  src += ";\n";

  if (!d.takesComm) {
    src += "MORI_EP_INTERNODE_CCO_ENTRY_LOCAL(";
    src += entry;
    src += ", ";
    src += d.body;
    src += ")\n";
  } else if (d.takesStdMoE) {
    src += "MORI_EP_INTERNODE_CCO_ENTRY_STDMOE(";
    src += entry;
    src += ", ";
    src += d.body;
    src += ", ";
    src += cfg.enableStdMoE ? "true" : "false";
    src += ")\n";
  } else {
    src += "MORI_EP_INTERNODE_CCO_ENTRY(";
    src += entry;
    src += ", ";
    src += d.body;
    src += ")\n";
  }
  return src;
}

// The grids v1 launches its passes with, as launch.cpp sizes them for the AOT
// symbols: the payload passes take blockNum, the ones that fan out over the
// device take the multiprocessor count, and the barrier is a single wavefront.
mori::jit::v2::LaunchGeometry EpInterNodeGeometry(const EpInterNodeCfg& cfg,
                                                  EpInterNodeKernel kind) {
  mori::jit::v2::LaunchGeometry g;
  g.blockX = static_cast<unsigned>(EpInterNodeBlockThreads(cfg));
  switch (kind) {
    case EpInterNodeKernel::CopyToStaging:
      g.gridX = static_cast<unsigned>(cfg.mpCount);
      g.sharedBytes = 0;
      break;
    case EpInterNodeKernel::Dispatch:
    case EpInterNodeKernel::DispatchLL:
      g.gridX = static_cast<unsigned>(cfg.blockNum);
      g.sharedBytes = static_cast<unsigned>(EpInterNodeDispatchSharedBytes(cfg));
      break;
    case EpInterNodeKernel::CombineSync:
      g.gridX = static_cast<unsigned>(cfg.mpCount);
      g.sharedBytes = 0;
      break;
    case EpInterNodeKernel::CombineSyncBarrier:
      // One wavefront, by construction: the barrier is a single-block fan-in.
      g.gridX = 1;
      g.blockX = static_cast<unsigned>(cfg.waveSize);
      g.sharedBytes = 0;
      break;
    case EpInterNodeKernel::Combine:
    case EpInterNodeKernel::CombineLL:
      g.gridX = static_cast<unsigned>(cfg.blockNum);
      g.sharedBytes = static_cast<unsigned>(EpInterNodeCombineSharedBytes(cfg));
      break;
    case EpInterNodeKernel::CombineAll:
      g.gridX = static_cast<unsigned>(cfg.mpCount);
      g.sharedBytes = static_cast<unsigned>(EpInterNodeCombineSharedBytes(cfg));
      break;
  }
  return g;
}

// ---------------------------------------------------------------------------
// The eight Specs. Every one of them is the same three delegations.
// ---------------------------------------------------------------------------
#define MORI_EP_INTERNODE_DEFINE_SPEC(ClassName, KIND)                \
  std::string ClassName::EntryName(const Cfg& cfg) {                  \
    return EpInterNodeEntryName(cfg, EpInterNodeKernel::KIND);        \
  }                                                                   \
  std::string ClassName::RenderSource(const Cfg& cfg) {               \
    return EpInterNodeRenderSource(cfg, EpInterNodeKernel::KIND);     \
  }                                                                   \
  mori::jit::v2::LaunchGeometry ClassName::Geometry(const Cfg& cfg) { \
    return EpInterNodeGeometry(cfg, EpInterNodeKernel::KIND);         \
  }                                                                   \
  const std::vector<std::string>& ClassName::SourceDeps() { return EpInterNodeDeps(); }

MORI_EP_INTERNODE_DEFINE_SPEC(EpInterNodeCopyToStagingSpec, CopyToStaging)
MORI_EP_INTERNODE_DEFINE_SPEC(EpInterNodeDispatchSpec, Dispatch)
MORI_EP_INTERNODE_DEFINE_SPEC(EpInterNodeDispatchLLSpec, DispatchLL)
MORI_EP_INTERNODE_DEFINE_SPEC(EpInterNodeCombineSyncSpec, CombineSync)
MORI_EP_INTERNODE_DEFINE_SPEC(EpInterNodeCombineSyncBarrierSpec, CombineSyncBarrier)
MORI_EP_INTERNODE_DEFINE_SPEC(EpInterNodeCombineSpec, Combine)
MORI_EP_INTERNODE_DEFINE_SPEC(EpInterNodeCombineLLSpec, CombineLL)
MORI_EP_INTERNODE_DEFINE_SPEC(EpInterNodeCombineAllSpec, CombineAll)

#undef MORI_EP_INTERNODE_DEFINE_SPEC

}  // namespace v2
}  // namespace ops
}  // namespace mori

// ===========================================================================
// Plan registration. Eight kernels, one Cfg, one Request, one Args schema --
// the only thing that differs is which Spec and which geometry.
// ===========================================================================

#include "mori/jit/v2/plan_api.hpp"

namespace {

mori::ops::v2::EpInterNodeCfg EpInterNodeCfgFromFields(const mori::jit::v2::FieldBag& f,
                                                       mori::ops::v2::EpInterNodeKernel kind) {
  using namespace mori::ops::v2;
  EpInterNodeRequest req;
  const EpInterNodeRequest defaults{};
  VisitFields(req, defaults, [&f](const char* n, auto& slot, const auto&) {
    using mori::jit::v2::WireAssign;
    if (f.Has(n)) WireAssign(slot, f.Get(n, 0));
  });
  return MakeEpInterNodeCfg(mori::jit::v2::GetToolchain().arch, req, kind);
}

// No C++-side AOT: a precompiled entry only helps if it renders the Cfg a live
// caller renders, and the geometry comes from the caller's tuning schedule.
int EpInterNodeNoPrecompile(const std::string&) { return 0; }

}  // namespace

#define MORI_EP_INTERNODE_DEFINE_PLAN(planName, ClassName, KIND)                         \
  namespace {                                                                            \
  mori::ops::v2::EpInterNodeCfg planName##FromFields(const mori::jit::v2::FieldBag& f) { \
    return EpInterNodeCfgFromFields(f, mori::ops::v2::EpInterNodeKernel::KIND);          \
  }                                                                                      \
  }                                                                                      \
  MORI_JIT_DEFINE_PLAN(planName, mori::ops::v2::ClassName, planName##FromFields,         \
                       mori::ops::v2::EpInterNodeRequestSchema, mori::ops::v2::Describe, \
                       EpInterNodeNoPrecompile, mori::ops::v2::EpInterNodeCcoArgs,       \
                       mori::ops::v2::EpInterNodeArgsSchema())

MORI_EP_INTERNODE_DEFINE_PLAN(ep_internode_copystaging, EpInterNodeCopyToStagingSpec, CopyToStaging)
MORI_EP_INTERNODE_DEFINE_PLAN(ep_internode_dispatch, EpInterNodeDispatchSpec, Dispatch)
MORI_EP_INTERNODE_DEFINE_PLAN(ep_internode_dispatch_ll, EpInterNodeDispatchLLSpec, DispatchLL)
MORI_EP_INTERNODE_DEFINE_PLAN(ep_internode_combinesync, EpInterNodeCombineSyncSpec, CombineSync)
MORI_EP_INTERNODE_DEFINE_PLAN(ep_internode_combinesyncbarrier, EpInterNodeCombineSyncBarrierSpec,
                              CombineSyncBarrier)
MORI_EP_INTERNODE_DEFINE_PLAN(ep_internode_combine, EpInterNodeCombineSpec, Combine)
MORI_EP_INTERNODE_DEFINE_PLAN(ep_internode_combine_ll, EpInterNodeCombineLLSpec, CombineLL)
MORI_EP_INTERNODE_DEFINE_PLAN(ep_internode_combineall, EpInterNodeCombineAllSpec, CombineAll)

#undef MORI_EP_INTERNODE_DEFINE_PLAN
