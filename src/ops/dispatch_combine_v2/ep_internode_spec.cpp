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

struct KernelDescriptor {
  const char* tag;   // goes in the entry name
  const char* body;  // the *_body function in ep_internode_kernel.hpp
  bool takesComm;    // false for the passes with no cross-node traffic
};

KernelDescriptor DescriptorFor(EpInterNodeKernel kind) {
  switch (kind) {
    case EpInterNodeKernel::CopyToStaging:
      return {"copystaging", "EpDispatchCopyToStaging_body", false};
    case EpInterNodeKernel::Dispatch:
      return {"dispatch", "EpDispatchInterNodeV2_body", true};
    case EpInterNodeKernel::DispatchLL:
      return {"dispatch_ll", "EpDispatchInterNodeV2LL_body", true};
    case EpInterNodeKernel::CombineSync:
      return {"combinesync", "EpCombineSync_body", false};
    case EpInterNodeKernel::CombineSyncBarrier:
      return {"combinesyncbarrier", "EpCombineSyncBarrier_body", false};
    case EpInterNodeKernel::Combine:
      return {"combine", "EpCombineInterNodeV2_body", true};
    case EpInterNodeKernel::CombineLL:
      return {"combine_ll", "EpCombineInterNodeV2LL_body", true};
    case EpInterNodeKernel::CombineAll:
      return {"combineall", "EpCombineAll_body", false};
  }
  throw std::runtime_error("mori ep internode v2 jit: unknown kernel");
}

// Header subtrees whose contents invalidate a compiled module. Coarser than the
// real include graph on purpose (see IncludeTreeHash): over-invalidating costs a
// rebuild, under-invalidating ships stale code.
const std::vector<std::string>& EpInterNodeSourceDeps() {
  static const std::vector<std::string> deps{"include/mori", "src/ops/dispatch_combine_v2",
                                             "src/cco"};
  return deps;
}

}  // namespace

// ---------------------------------------------------------------------------
// Request -> Cfg
// ---------------------------------------------------------------------------
EpInterNodeCfg MakeEpInterNodeCfg(const std::string& arch, const EpInterNodeRequest& request,
                                  EpInterNodeKernel kind) {
  EpInterNodeCfg cfg;
  cfg.kernelCfg.worldSize = request.worldSize;
  cfg.kernelCfg.hiddenDim = request.hiddenDim;
  cfg.kernelCfg.scaleDim = request.scaleDim;
  cfg.kernelCfg.scaleTypeSize = request.scaleTypeSize;
  cfg.kernelCfg.maxTokenTypeSize = request.maxTokenTypeSize;
  cfg.kernelCfg.maxNumInpTokenPerRank = request.maxNumInpTokenPerRank;
  cfg.kernelCfg.numExpertPerRank = request.numExpertPerRank;
  cfg.kernelCfg.numExpertPerToken = request.numExpertPerToken;
  cfg.kernelCfg.maxTotalRecvTokens = request.maxTotalRecvTokens;
  cfg.kernelCfg.gpuPerNode = request.gpuPerNode;
  cfg.kernelCfg.numQpPerPe = request.numQpPerPe;
  cfg.kernelCfg.quantType = request.quantType;

  cfg.dtype = request.dtype;

  cfg.waveSize = mori::jit::v2::WaveSizeForArch(arch);

  // Placeholders for a bare C++ caller only. v1 retunes blocks and warps per
  // token count, so a real caller always passes them; unlike the intranode
  // defaults these are not measured optima, they only have to be launchable.
  cfg.blockNum = 64;
  cfg.warpPerBlock = 8;
  cfg.rdmaBlockNum = 8;
  cfg.mpCount = 64;

  if (request.blockNum > 0) cfg.blockNum = request.blockNum;
  if (request.warpPerBlock > 0) cfg.warpPerBlock = request.warpPerBlock;
  if (request.rdmaBlockNum > 0) cfg.rdmaBlockNum = request.rdmaBlockNum;
  if (request.mpCount > 0) cfg.mpCount = request.mpCount;

  if (!EpInterNodeKernelCfgIsValid(cfg.kernelCfg)) {
    throw std::runtime_error(
        "mori ep internode v2: unusable config " + Render(cfg.kernelCfg) +
        "; every divisor must be positive and worldSize must be a multiple of gpuPerNode");
  }

  // Caught here rather than by hipModuleLaunchKernel, which reports it as a
  // generic launch failure with no mention of which knob was too large.
  const int threadsPerBlock = EpInterNodeBlockThreads(cfg);
  if (threadsPerBlock <= 0 || threadsPerBlock > 1024) {
    throw std::runtime_error(
        "mori ep internode v2: warpPerBlock " + std::to_string(cfg.warpPerBlock) + " x waveSize " +
        std::to_string(cfg.waveSize) + " = " + std::to_string(threadsPerBlock) +
        " threads per block, which exceeds 1024");
  }

  // Wave64 only. DispatchInterNodeSend builds its intra-warp prefix count as
  // __popcll(mask << (warpSize - laneId)) on a uint64_t, which drops the lanes
  // at or above laneId only when the shift width equals the container width.
  // On wave32 the discarded bits stay inside the 64-bit value and every set bit
  // is counted, so the send slot is wrong rather than the code merely being
  // slow. Rejected here rather than left to produce bad offsets.
  if (cfg.waveSize != 64) {
    throw std::runtime_error("mori ep internode v2: waveSize " + std::to_string(cfg.waveSize) +
                             " is unsupported; the internode kernels are wave64 only");
  }

  // The grid is split: blocks below rdmaBlockNum take the RDMA leg, the rest the
  // intra-node one. rdmaBlockNum >= blockNum leaves the intra-node half with
  // nothing AND makes the dispatch fan-in wait on rdmaBlockNum * warpNum
  // arrivals that can never occur, so every peer spins forever. Strict, because
  // rdmaBlockNum == blockNum is equally broken: it leaves xgmiBlockNum == 0.
  if (cfg.rdmaBlockNum >= cfg.blockNum) {
    throw std::runtime_error("mori ep internode v2: rdmaBlockNum " +
                             std::to_string(cfg.rdmaBlockNum) + " must be < blockNum " +
                             std::to_string(cfg.blockNum) +
                             "; the intra-node half would get no blocks and the dispatch "
                             "barrier would never complete");
  }
  (void)kind;
  return cfg;
}

std::string EpInterNodeRequestSchema() {
  mori::jit::v2::SchemaBuilder builder;
  const EpInterNodeRequest defaults{};
  VisitFields(defaults, defaults, [&builder](const char* name, const auto& value, const auto&) {
    using mori::jit::v2::WireTag;
    using mori::jit::v2::WireValue;
    builder.Add(name, WireTag(value), WireValue(value));
  });
  return builder.Str();
}

// ---------------------------------------------------------------------------
// Source rendering. The Cfg text IS the specialisation and IS the cache key --
// there is no other channel by which a config can reach hipcc. Geometry is
// deliberately absent from the rendered text; see the note on EpInterNodeCfg.
// ---------------------------------------------------------------------------
std::string EpInterNodeEntryName(const EpInterNodeCfg& cfg, EpInterNodeKernel kind) {
  const KernelDescriptor descriptor = DescriptorFor(kind);
  std::string name = "mori_ep_internode_";
  name += descriptor.tag;
  name += '_';
  name += EpInterNodeDTypeTag(cfg.dtype);
  return name;
}

std::string EpInterNodeRenderSource(const EpInterNodeCfg& cfg, EpInterNodeKernel kind) {
  // Once the cfg is an NTTP its divisors are literals, so a zero here is a
  // division by constant zero inside hipcc -- diagnosed, if at all, against
  // generated source no one can trace back to the caller. MakeEpInterNodeCfg already
  // rejects it, but a Cfg can also be aggregate-initialised by hand, and this is
  // the one funnel every compile goes through.
  if (!EpInterNodeKernelCfgIsValid(cfg.kernelCfg)) {
    throw std::runtime_error("mori ep internode v2 jit: unusable config " + Render(cfg.kernelCfg));
  }

  const KernelDescriptor descriptor = DescriptorFor(kind);
  const std::string entry = EpInterNodeEntryName(cfg, kind);

  // The two names the entry macros expand against, in the same order and with
  // the same spelling ep_spec.cpp uses for the intranode kernels.
  std::string source =
      "// mori jit — generated, do not edit.\n"
      "#include \"src/ops/dispatch_combine_v2/ep_internode_kernel.hpp\"\n"
      "constexpr ::mori::ops::v2::EpInterNodeKernelCfg kConfig = ";
  source += Render(cfg.kernelCfg);
  source += ";\nusing TokT = ";
  source += EpInterNodeDTypeName(cfg.dtype);
  source += ";\n";

  if (!descriptor.takesComm) {
    source += "MORI_EP_INTERNODE_CCO_ENTRY_LOCAL(";
    source += entry;
    source += ", ";
    source += descriptor.body;
    source += ")\n";
  } else {
    source += "MORI_EP_INTERNODE_CCO_ENTRY(";
    source += entry;
    source += ", ";
    source += descriptor.body;
    source += ")\n";
  }
  return source;
}

// The grids v1 launches its passes with, as launch.cpp sizes them for the AOT
// symbols: the payload passes take blockNum, the ones that fan out over the
// device take the multiprocessor count, and the barrier is a single wavefront.
mori::jit::v2::LaunchGeometry EpInterNodeGeometry(const EpInterNodeCfg& cfg,
                                                  EpInterNodeKernel kind) {
  mori::jit::v2::LaunchGeometry geometry;
  geometry.blockX = static_cast<unsigned>(EpInterNodeBlockThreads(cfg));
  switch (kind) {
    case EpInterNodeKernel::CopyToStaging:
      geometry.gridX = static_cast<unsigned>(cfg.mpCount);
      geometry.sharedBytes = 0;
      break;
    case EpInterNodeKernel::Dispatch:
    case EpInterNodeKernel::DispatchLL:
      geometry.gridX = static_cast<unsigned>(cfg.blockNum);
      // No dispatch pass declares dynamic shared memory.
      geometry.sharedBytes = 0;
      break;
    case EpInterNodeKernel::CombineSync:
      geometry.gridX = static_cast<unsigned>(cfg.mpCount);
      geometry.sharedBytes = 0;
      break;
    case EpInterNodeKernel::CombineSyncBarrier:
      // One wavefront, by construction: the barrier is a single-block fan-in.
      geometry.gridX = 1;
      geometry.blockX = static_cast<unsigned>(cfg.waveSize);
      geometry.sharedBytes = 0;
      break;
    case EpInterNodeKernel::Combine:
    case EpInterNodeKernel::CombineLL:
      geometry.gridX = static_cast<unsigned>(cfg.blockNum);
      geometry.sharedBytes = static_cast<unsigned>(EpInterNodeCombineSharedBytes(cfg));
      break;
    case EpInterNodeKernel::CombineAll:
      geometry.gridX = static_cast<unsigned>(cfg.mpCount);
      geometry.sharedBytes = static_cast<unsigned>(EpInterNodeCombineSharedBytes(cfg));
      break;
  }
  return geometry;
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
  const std::vector<std::string>& ClassName::SourceDeps() { return EpInterNodeSourceDeps(); }

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

mori::ops::v2::EpInterNodeCfg EpInterNodeCfgFromFields(const mori::jit::v2::FieldBag& fields,
                                                       mori::ops::v2::EpInterNodeKernel kind) {
  using namespace mori::ops::v2;
  EpInterNodeRequest request;
  const EpInterNodeRequest defaults{};
  VisitFields(request, defaults, [&fields](const char* name, auto& slot, const auto&) {
    using mori::jit::v2::WireAssign;
    if (fields.Has(name)) WireAssign(slot, fields.Get(name, 0));
  });
  return MakeEpInterNodeCfg(mori::jit::v2::GetToolchain().arch, request, kind);
}

// No C++-side AOT: a precompiled entry only helps if it renders the Cfg a live
// caller renders, and the geometry comes from the caller's tuning schedule.
int EpInterNodeNoPrecompile(const std::string&) { return 0; }

}  // namespace

#define MORI_EP_INTERNODE_DEFINE_PLAN(planName, ClassName, KIND)                              \
  namespace {                                                                                 \
  mori::ops::v2::EpInterNodeCfg planName##FromFields(const mori::jit::v2::FieldBag& fields) { \
    return EpInterNodeCfgFromFields(fields, mori::ops::v2::EpInterNodeKernel::KIND);          \
  }                                                                                           \
  }                                                                                           \
  MORI_JIT_DEFINE_PLAN(planName, mori::ops::v2::ClassName, planName##FromFields,              \
                       mori::ops::v2::EpInterNodeRequestSchema, mori::ops::v2::Describe,      \
                       EpInterNodeNoPrecompile, mori::ops::v2::EpInterNodeCcoArgs,            \
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
