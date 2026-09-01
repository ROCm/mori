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
// What the host and the JIT-generated device TU BOTH need for the v1 internode
// kernels: the specialised-on config, its rendering, and the argument struct.
//
// The counterpart of ep_cfg.hpp for the intranode pair, and it exists for the
// same reason (docs/MORI_JIT_V2_DESIGN.md §3.5): ep_internode_spec.hpp pulls
// in jit/v2/spec.hpp and the Compiler, and the device TU has no business seeing
// either. Attribute-free and standard C++ -- no __host__/__device__ here, or
// every host TU that touches a Cfg would need hipcc.
//
// Kept separate from ep_cfg.hpp rather than merged: dispatch_combine.hpp below
// costs ~200 headers (`hipcc -E -H`: 14 for ep_cfg.hpp, 219 here), and merging
// would charge that, and the warpSize dance, to the intranode kernels on every
// JIT compile.
// ---------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <string>

#include "mori/jit/v2/render.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"

// cco.hpp declares mori::cco::impl::warpSize(); mori/core/utils/utils.hpp may
// already have defined `warpSize` as a macro. See ep_internode_kernel.hpp.
#pragma push_macro("warpSize")
#undef warpSize
#include "mori/cco/cco.hpp"
#pragma pop_macro("warpSize")

namespace mori {
namespace moe {

// jit::v2::Fields calls RenderValue unqualified, so the overload for a config
// enum has to live in the enum's own namespace for ADL to find it.
inline std::string RenderValue(QuantType q) {
  switch (q) {
    case QuantType::Fp8DirectCast:
      return "::mori::moe::QuantType::Fp8DirectCast";
    case QuantType::Fp8BlockwiseQuant:
      return "::mori::moe::QuantType::Fp8BlockwiseQuant";
    case QuantType::Fp4BlockwiseQuant:
      return "::mori::moe::QuantType::Fp4BlockwiseQuant";
    default:
      return "::mori::moe::QuantType::None";
  }
}

}  // namespace moe

namespace ops {
namespace v2 {

// ---------------------------------------------------------------------------
// Transported element type. v1 keeps the two fp8 encodings and fp4 apart, which
// is why this is not EpDType: the intranode pair folds all three into Byte8
// because dispatch only copies bytes, whereas the v1 bodies hand T on to
// convert.hpp and to the quant paths.
// ---------------------------------------------------------------------------
enum class EpInterNodeDType { Bf16, Fp32, Fp8Fnuz, Fp8Ocp, Fp4 };

// Identifier-safe short tag for the kernel symbol. Deliberately the spellings
// launch.cpp uses for the AOT symbols, so the two builds can be lined up.
inline const char* EpInterNodeDTypeTag(EpInterNodeDType d) {
  switch (d) {
    case EpInterNodeDType::Fp32:
      return "f32";
    case EpInterNodeDType::Fp8Fnuz:
      return "fp8_fnuz";
    case EpInterNodeDType::Fp8Ocp:
      return "fp8_ocp";
    case EpInterNodeDType::Fp4:
      return "fp4";
    default:
      return "bf16";
  }
}

// The C++ type the generated TU aliases as TokT. Separate from the tag: these
// spellings contain characters that would not survive in a symbol name.
inline const char* EpInterNodeDTypeName(EpInterNodeDType d) {
  switch (d) {
    case EpInterNodeDType::Fp32:
      return "float";
    case EpInterNodeDType::Fp8Fnuz:
      return "__hip_fp8_e4m3_fnuz";
    case EpInterNodeDType::Fp8Ocp:
      return "__hip_fp8_e4m3";
    case EpInterNodeDType::Fp4:
      return "mori::mori_fp4x2_e2m1";
    default:
      return "hip_bfloat16";
  }
}

inline std::string RenderValue(EpInterNodeDType d) {
  switch (d) {
    case EpInterNodeDType::Fp32:
      return "::mori::ops::v2::EpInterNodeDType::Fp32";
    case EpInterNodeDType::Fp8Fnuz:
      return "::mori::ops::v2::EpInterNodeDType::Fp8Fnuz";
    case EpInterNodeDType::Fp8Ocp:
      return "::mori::ops::v2::EpInterNodeDType::Fp8Ocp";
    case EpInterNodeDType::Fp4:
      return "::mori::ops::v2::EpInterNodeDType::Fp4";
    default:
      return "::mori::ops::v2::EpInterNodeDType::Bf16";
  }
}

// ---------------------------------------------------------------------------
// The part of EpDispatchCombineConfig the kernel is specialised on -- what gets
// rendered into the TU, the way EpCfg is for the intranode pair. The generated
// TU names it `kConfig` and the bodies take it as
// `template <EpInterNodeKernelCfg kConfig, typename T>`.
//
// v1 reads its whole configuration out of args.config, so every shape query in
// the kernel -- `flat / config.MaxNumTokensToRecv()`, `expert /
// config.numExpertPerRank / config.gpuPerNode`, `dstTokId % config.numQpPerPe`
// -- is a scalar load feeding a full integer division. Passed as an NTTP they
// become literals, the divisions collapse to multiply-shift, and the quant
// branches fold away.
//
// Two groups of fields are deliberately NOT here:
//
//   rank -- would give every rank on a node its own copy of an identical
//   kernel, so eight hipcc runs and eight module loads for one binary's worth
//   of code. EpCfg keeps rank a launch argument for the same reason.
//
//   launch geometry -- it lives in EpInterNodeCfg, one level up in the host-only
//   header, because the kernel reads its grid from gridDim/blockDim rather than
//   from kConfig. See the note there.
//
// Defaults mirror EpDispatchCombineConfig's, except worldSize: 0 there means
// "not filled in yet", which is nothing that can be compiled for.
// ---------------------------------------------------------------------------
struct EpInterNodeKernelCfg {
  int worldSize{8};
  int hiddenDim{4096};
  int scaleDim{32};
  int scaleTypeSize{1};
  int maxTokenTypeSize{4};
  int maxNumInpTokenPerRank{128};
  int numExpertPerRank{1};
  int numExpertPerToken{2};
  int maxTotalRecvTokens{0};
  int gpuPerNode{8};
  int numQpPerPe{1};
  mori::moe::QuantType quantType{mori::moe::QuantType::None};
};

template <typename Self, typename Visit>
inline void VisitFields(Self& c, const EpInterNodeKernelCfg& d, Visit&& v) {
#define MORI_FIELD(x) v(#x, c.x, d.x)
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
#undef MORI_FIELD
}

MORI_JIT_ASSERT_FIELD_COUNT(
    EpInterNodeKernelCfg, 12,
    "added an EpInterNodeKernelCfg field -- update VisitFields(EpInterNodeKernelCfg) "
    "too, or the kernel silently compiles against its default");

// Designated-initialiser text for the generated TU. Only non-default fields are
// emitted, so a field added with a behaviour-preserving default leaves every
// existing cache entry addressed the same way.
inline std::string Render(const EpInterNodeKernelCfg& c) {
  const EpInterNodeKernelCfg d{};
  mori::jit::v2::Fields f;
  VisitFields(c, d, [&f](const char* name, const auto& value, const auto& dflt) {
    f.Put(name, value, dflt);
  });
  return mori::jit::v2::BraceInit("::mori::ops::v2::EpInterNodeKernelCfg", f);
}

// Lets Fields::Put treat a nested EpInterNodeKernelCfg like any other value, without a
// second field list: the rendered text is what distinguishes two of them.
inline std::string RenderValue(const EpInterNodeKernelCfg& c) { return Render(c); }

inline bool operator==(const EpInterNodeKernelCfg& a, const EpInterNodeKernelCfg& b) {
  // Compares the rendered text rather than the fields: the text IS what selects
  // the cached binary, and it cannot drift out of sync with VisitFields the way
  // a hand-written field comparison does.
  return Render(a) == Render(b);
}

// Project a live config onto the subset the kernel specialises on. For a caller
// that already holds an EpDispatchCombineHandle; the plan API builds the same
// thing out of an EpInterNodeRequest instead.
//
// Call this on the config a launch is actually about to run with, never on one
// that merely resembles it. hiddenDim in particular is a per-call argument in
// v1: LaunchDispatch writes it into args.config, not back into handle.config, so
// those two disagree the moment a caller passes hidden_dim -- and since the
// kernel's EpInterNodeBindConfig OVERWRITES args.config with the NTTP, sourcing the
// constants from the stale one silently runs the kernel against other numbers.
inline EpInterNodeKernelCfg MakeEpInterNodeKernelCfg(const mori::moe::EpDispatchCombineConfig& c) {
  EpInterNodeKernelCfg s;
  s.worldSize = c.worldSize;
  s.hiddenDim = c.hiddenDim;
  s.scaleDim = c.scaleDim;
  s.scaleTypeSize = c.scaleTypeSize;
  s.maxTokenTypeSize = c.maxTokenTypeSize;
  s.maxNumInpTokenPerRank = c.maxNumInpTokenPerRank;
  s.numExpertPerRank = c.numExpertPerRank;
  s.numExpertPerToken = c.numExpertPerToken;
  s.maxTotalRecvTokens = c.maxTotalRecvTokens;
  s.gpuPerNode = c.gpuPerNode;
  s.numQpPerPe = c.numQpPerPe;
  s.quantType = c.quantType;
  return s;
}

// A zero in any divisor is a division by a literal zero once the cfg is an
// NTTP -- hipcc either rejects the TU or emits a poison value, and neither
// diagnoses back to the caller that built it wrong.
inline bool EpInterNodeKernelCfgIsValid(const EpInterNodeKernelCfg& s) {
  return s.worldSize > 0 && s.gpuPerNode > 0 && s.numQpPerPe > 0 && s.numExpertPerRank > 0 &&
         s.numExpertPerToken > 0 && s.maxNumInpTokenPerRank > 0 && s.hiddenDim > 0 &&
         (s.worldSize % s.gpuPerNode) == 0;
}

// ---------------------------------------------------------------------------
// The single by-value kernel argument, shared with the device side:
// ep_internode_kernel.hpp includes this header rather than redeclaring it, so
// the two cannot disagree about the layout.
//
// EpDispatchCombineArgsRaw is what the AOT launcher already passes, and it is
// static_asserted to share a layout with EpDispatchCombineArgs<T>. The device
// communicator rides alongside it because that is how a JIT module gets its
// endpoints.
// ---------------------------------------------------------------------------
struct EpInterNodeCcoArgs {
  mori::moe::EpDispatchCombineArgsRaw raw;
  ::mori::cco::ccoDevComm devComm;
};

static_assert(
    sizeof(EpInterNodeCcoArgs) ==
        sizeof(mori::moe::EpDispatchCombineArgsRaw) + sizeof(::mori::cco::ccoDevComm),
    "EpInterNodeCcoArgs has interior padding -- EpInterNodeArgsSchema() describes it as two "
    "back-to-back byte ranges and would place devComm at the wrong offset");

// The args schema, in the form plan_api publishes.
//
// EpArgs crosses as 22 named scalars because it was designed for this boundary.
// These args cannot: every value in them is produced by EpDispatchCombineHandle
// (53 fields, 15 of them SymmMemObjPtr, 3 nested ShmemBufs*), and a caller on
// the far side of the ABI holds them only as the opaque buffer BuildArgs hands
// back. So they cross as byte ranges -- named, sized, and size-checked against
// C++'s own sizeof like any other schema, but copied wholesale rather than
// filled field by field.
inline const char* EpInterNodeArgsSchema() {
  static const std::string s = "raw:b" +
                               std::to_string(sizeof(mori::moe::EpDispatchCombineArgsRaw)) +
                               ",devComm:b" + std::to_string(sizeof(::mori::cco::ccoDevComm)) + ",";
  return s.c_str();
}

}  // namespace v2
}  // namespace ops
}  // namespace mori
