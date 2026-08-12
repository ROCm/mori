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

#include "mori/ops/dispatch_combine_v2/ep_spec.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "mori/jit/v2/toolchain.hpp"

namespace mori {
namespace ops {
namespace v2 {

namespace {

int EnvInt(const char* name, int current) {
  const char* v = std::getenv(name);
  if (!v || !*v) return current;
  char* end = nullptr;
  long parsed = std::strtol(v, &end, 10);
  if (end == v || parsed <= 0) return current;
  return static_cast<int>(parsed);
}

}  // namespace

EpCfg MakeEpCfg(const std::string& arch, const EpRequest& req, EpKernelKind kind) {
  EpCfg c;
  c.worldSize = req.worldSize;
  c.hiddenDim = req.hiddenDim;
  c.maxTokPerRank = req.maxTokPerRank;
  c.numExpertPerRank = req.numExpertPerRank;
  c.numExpertPerToken = req.numExpertPerToken;
  c.maxRecv = req.maxRecv;
  c.dtype = req.dtype;
  c.useWeights = req.useWeights;

  c.waveSize = mori::jit::v2::WaveSizeForArch(arch);

  // Geometry defaults, split by kernel because the two are bound by different
  // things: dispatch is a copy engine and wants warps, combine's per-token
  // reduction saturates sooner.
  //
  // Combine's 8 warps is measured, not inherited: on gfx950 EP8 at
  // hidden=7168/topk=8, combine runs 75.4/164.1/978 us at 8 warps against
  // 87.8/173.2/1008 at v1's 4 and 102.8/201.3/1100 at 16 (128/512/4096 tokens).
  //
  // Combine's 64 blocks replaces v1's 80. The full sweep behind
  // hip_tuning_configs measured 64x8 as the winner at every token count and both
  // topk on mi355x -- 2-3% over 80x8 with 16 fewer blocks -- and 80 was never a
  // measured optimum, just the value v1 carried. Fewer blocks is not free either:
  // 32x8 costs 37% at ct=4096, so this is the optimum and not merely the smallest
  // thing tried. The Python path takes the tuning table and never sees this; what
  // it fixes is the bare C++ caller, who was getting a geometry the tables already
  // knew was worse.
  //
  // Still one shape -- a default, not a tuning table. Anything tuned belongs in
  // hip_tuning_configs, which is keyed by device, shape, topk and dtype.
  const bool isDispatch = kind == EpKernelKind::Dispatch;
  c.blockNum = 64;
  c.warpPerBlock = isDispatch ? 16 : 8;

  if (req.blockNum > 0) c.blockNum = req.blockNum;
  if (req.warpPerBlock > 0) c.warpPerBlock = req.warpPerBlock;

  // Overrides. The ONLY place the environment is read.
  const char* blkVar = isDispatch ? "MORI_V2_EP_DISP_BLOCKS" : "MORI_V2_EP_COMB_BLOCKS";
  const char* wrpVar = isDispatch ? "MORI_V2_EP_DISP_WARPS" : "MORI_V2_EP_COMB_WARPS";
  c.blockNum = EnvInt(blkVar, c.blockNum);
  c.warpPerBlock = EnvInt(wrpVar, c.warpPerBlock);

  // Byte8 is transport-only. Dispatch copies its payload untouched, but combine
  // sums across sources, so a byte type there would compile and silently reduce
  // garbage. Reject it here rather than let a Cfg like that reach hipcc.
  if (!isDispatch && c.dtype == EpDType::Byte8) {
    throw std::runtime_error(
        "mori v2 ep: combine reduces its input, so it needs an arithmetic dtype; "
        "fp8/fp4 are dispatch-transport only (pair them with a bf16/fp32 combine)");
  }

  if (!EpCfgIsValid(c)) {
    throw std::runtime_error(
        "mori v2 ep: inconsistent config (world=" + std::to_string(c.worldSize) +
        " hidden=" + std::to_string(c.hiddenDim) +
        " topk=" + std::to_string(c.numExpertPerToken) + " wave=" + std::to_string(c.waveSize) +
        " warps=" + std::to_string(c.warpPerBlock) + " blocks=" + std::to_string(c.blockNum) +
        "); token bytes must be 16 B aligned, topk and worldSize must fit in a wavefront");
  }
  return c;
}

std::string EpRequestSchema() {
  mori::jit::v2::SchemaBuilder sb;
  const EpRequest def{};
  VisitFields(def, def,
              [&](const char* n, const auto& val, const auto&) { EpEmitSchema(sb, n, val); });
  return sb.Str();
}

// ---------------------------------------------------------------------------
// Source rendering. The Cfg text IS the specialisation and IS the cache key --
// there is no other channel by which a config can reach hipcc.
// ---------------------------------------------------------------------------
namespace {

// gfx125x -> the TDM body + its LDS geometry.
bool EpArchIs1250() { return mori::jit::v2::GetToolchain().arch.rfind("gfx125", 0) == 0; }

// Host-side arch routing: gfx125x renders the TDM body (ep_intranode_1250x.hpp,
// which pulls the gfx1250 TDM header), every other arch renders the portable one.
// The render-time arch is the same GetToolchain().arch the toolchain compiles
// with (--offload-arch), so host and device never disagree, and the choice is in
// the rendered text -> in the cache key.
std::string RenderEpSource(const EpCfg& cfg, const char* portableBody, const char* gfx1250Body) {
  const bool is1250 = EpArchIs1250();
  const char* header = is1250 ? "src/ops/dispatch_combine_v2/ep_intranode_1250x.hpp"
                              : "src/ops/dispatch_combine_v2/ep_intranode_kernel.hpp";
  const char* body = is1250 ? gfx1250Body : portableBody;
  return std::string(
             "// mori jit v2 — generated, do not edit.\n"
             "#include \"") +
         header +
         "\"\n"
         "using namespace mori::ops::v2;\n"
         "constexpr EpCfg kCfg = " +
         Render(cfg) +
         ";\n"
         "using TokT = " +
         EpDTypeName(cfg.dtype) +
         ";\n"
         "extern \"C\" __global__ void __launch_bounds__(EpBlockThreads(kCfg))\n"
         "mori_jit_entry(EpArgs args) { " +
         body + "<kCfg, TokT>(args); }\n";
}

const std::vector<std::string>& EpSourceDeps() {
  static const std::vector<std::string> deps{"include/mori", "src/ops/dispatch_combine_v2",
                                             "src/cco"};
  return deps;
}

}  // namespace

std::string EpDispatchSpec::RenderSource(const Cfg& cfg) {
  return RenderEpSource(cfg, "EpDispatchBody", "EpDispatch1250xBody");
}

std::string EpCombineSpec::RenderSource(const Cfg& cfg) {
  return RenderEpSource(cfg, "EpCombineBody", "EpCombine1250xBody");
}

const std::vector<std::string>& EpDispatchSpec::SourceDeps() { return EpSourceDeps(); }
const std::vector<std::string>& EpCombineSpec::SourceDeps() { return EpSourceDeps(); }

mori::jit::v2::LaunchGeometry EpDispatchSpec::Geometry(const Cfg& cfg) {
  mori::jit::v2::LaunchGeometry g;
  g.gridX = static_cast<unsigned>(cfg.blockNum);
  g.blockX = static_cast<unsigned>(EpBlockThreads(cfg));
  // Portable dispatch keeps everything in registers; the gfx1250 TDM dispatch
  // stages one hidden-dim token tile per warp in dynamic LDS.
  g.sharedBytes = EpArchIs1250() ? static_cast<unsigned>(EpDispatch1250xLdsBytes(cfg)) : 0;
  return g;
}

mori::jit::v2::LaunchGeometry EpCombineSpec::Geometry(const Cfg& cfg) {
  mori::jit::v2::LaunchGeometry g;
  g.gridX = static_cast<unsigned>(cfg.blockNum);
  g.blockX = static_cast<unsigned>(EpBlockThreads(cfg));
  // Portable combine only needs the per-warp pointer arrays; the gfx1250 PULL/QUAD
  // paths size their tiles against the whole LDS budget at runtime.
  g.sharedBytes = EpArchIs1250() ? static_cast<unsigned>(EpCombine1250xLdsBudget)
                                 : static_cast<unsigned>(EpCombineSharedBytes(cfg));
  return g;
}

}  // namespace v2
}  // namespace ops
}  // namespace mori

// ===========================================================================
// Plan registration. Two kernels, one Cfg, one Request, one Args schema -- the
// only thing that differs is which Spec and which geometry.
// ===========================================================================

#include "mori/jit/v2/plan_api.hpp"

namespace {

mori::ops::v2::EpCfg EpCfgFromFields(const mori::jit::v2::FieldBag& f,
                                     mori::ops::v2::EpKernelKind kind) {
  using namespace mori::ops::v2;
  EpRequest req;
  EpApplyFields(req, /*prefix=*/"", [&](const std::string& n) { return f.Has(n.c_str()); },
                [&](const std::string& n) { return f.Get(n.c_str(), 0); });
  return MakeEpCfg(mori::jit::v2::GetToolchain().arch, req, kind);
}

mori::ops::v2::EpCfg EpDispatchFromFields(const mori::jit::v2::FieldBag& f) {
  return EpCfgFromFields(f, mori::ops::v2::EpKernelKind::Dispatch);
}
mori::ops::v2::EpCfg EpCombineFromFields(const mori::jit::v2::FieldBag& f) {
  return EpCfgFromFields(f, mori::ops::v2::EpKernelKind::Combine);
}

// No C++-side AOT for these kernels, and it is not an oversight: a precompiled
// entry only helps if it renders the Cfg a live op renders, and the launch
// geometry comes from the Python tuning schedule. Warming the cache therefore
// means constructing the op once at build time, which needs no table here.
int EpNoPrecompile(const std::string&) { return 0; }

}  // namespace

// Field order and types must match EpArgs; the binding builds its ctypes struct
// from this string and asserts sizeof against the vtable, so drift is a startup
// error rather than silent corruption.
#define MORI_EP_ARGS_SCHEMA                                                     \
  "window:u64,"                                                                  \
  "offTokOff:u64,offRecvNum:u64,offRecvToSrc:u64,offOutIdx:u64,"                 \
  "offOutWts:u64,offDispOut:u64,offOutTok:u64,offXdb:u64,rank:i32,"              \
  "tokenIndices:p,inpTokenBuf:p,weightsBuf:p,outTokenBuf:p,outWeightsBuf:p,"     \
  "dispDestTokIdMap:p,destPeTokenCounter:p,totalRecvTokenNum:p,"                 \
  "gridBarrier:p,xdbFlag:p,combineBarrierFan:p,numTokens:i32"

MORI_JIT_DEFINE_PLAN(ep_dispatch, mori::ops::v2::EpDispatchSpec, EpDispatchFromFields,
                     mori::ops::v2::EpRequestSchema, mori::ops::v2::Describe, EpNoPrecompile,
                     mori::ops::v2::EpArgs, MORI_EP_ARGS_SCHEMA)

MORI_JIT_DEFINE_PLAN(ep_combine, mori::ops::v2::EpCombineSpec, EpCombineFromFields,
                     mori::ops::v2::EpRequestSchema, mori::ops::v2::Describe, EpNoPrecompile,
                     mori::ops::v2::EpArgs, MORI_EP_ARGS_SCHEMA)
