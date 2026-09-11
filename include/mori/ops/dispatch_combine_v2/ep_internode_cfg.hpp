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
// What the host and the JIT-generated device translation unit (TU) BOTH need for
// the v2 internode kernels: the specialised-on config, its rendering, and the
// argument struct.
//
// The counterpart of ep_cfg.hpp for the intranode pair, and it exists for the
// same reason (docs/MORI_JIT_V2_DESIGN.md §3.5): ep_internode_spec.hpp pulls
// in jit/v2/spec.hpp and the Compiler, and the device TU has no business seeing
// either. Attribute-free and standard C++ -- no __host__/__device__ here, or
// every host TU that touches a Cfg would need hipcc.
//
// Kept separate from ep_cfg.hpp rather than merged: this one pulls cco.hpp (and
// the warpSize dance around it) for the device communicator, and merging would
// charge that to the intranode kernels on every JIT compile. It no longer pulls
// v1 -- ep_internode_args.hpp owns the argument struct, the config the bodies
// read, index_t and QuantType.
// ---------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <string>

#include "mori/jit/v2/render.hpp"
#include "mori/ops/dispatch_combine_v2/ep_internode_args.hpp"

// cco.hpp declares mori::cco::impl::warpSize(); mori/core/utils/utils.hpp may
// already have defined `warpSize` as a macro. See ep_internode_kernel.hpp.
#pragma push_macro("warpSize")
#undef warpSize
#include "mori/cco/cco.hpp"
#pragma pop_macro("warpSize")

namespace mori {
namespace ops {
namespace v2 {

// jit::v2::Fields calls RenderValue unqualified, so the overload for a config
// enum has to live in the enum's own namespace for ADL to find it. That is
// mori::ops::v2 now that EpQuantType is v2's own enum rather than v1's.
inline std::string RenderValue(EpQuantType quantType) {
  switch (quantType) {
    case EpQuantType::Fp8DirectCast:
      return "::mori::ops::v2::EpQuantType::Fp8DirectCast";
    case EpQuantType::Fp8BlockwiseQuant:
      return "::mori::ops::v2::EpQuantType::Fp8BlockwiseQuant";
    case EpQuantType::Fp4BlockwiseQuant:
      return "::mori::ops::v2::EpQuantType::Fp4BlockwiseQuant";
    default:
      return "::mori::ops::v2::EpQuantType::None";
  }
}

// ---------------------------------------------------------------------------
// Transported element type. v1 keeps the two fp8 encodings and fp4 apart, which
// is why this is not EpDType: the intranode pair folds all three into Byte8
// because dispatch only copies bytes, whereas the v1 bodies hand T on to
// convert.hpp and to the quant paths.
// ---------------------------------------------------------------------------
enum class EpInterNodeDType { Bf16, Fp32, Fp8Fnuz, Fp8Ocp, Fp4 };

// Identifier-safe short tag for the kernel symbol. Deliberately the spellings
// launch.cpp uses for the AOT symbols, so the two builds can be lined up.
inline const char* EpInterNodeDTypeTag(EpInterNodeDType dtype) {
  switch (dtype) {
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
inline const char* EpInterNodeDTypeName(EpInterNodeDType dtype) {
  switch (dtype) {
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

inline std::string RenderValue(EpInterNodeDType dtype) {
  switch (dtype) {
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
// -- is a scalar load feeding a full integer division. Passed as a non-type
// template parameter (NTTP) they become literals, the divisions collapse to
// multiply-shift, and the quant branches fold away.
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
  EpQuantType quantType{EpQuantType::None};
};

// The device-side config the bodies read: purely the compiled-in shape, so the
// result is constexpr and every field folds to a literal at its use.
constexpr EpInterNodeDeviceCfg EpInterNodeDeviceCfgOf(const EpInterNodeKernelCfg& kernelCfg) {
  EpInterNodeDeviceCfg deviceCfg{};
  deviceCfg.worldSize = kernelCfg.worldSize;
  deviceCfg.hiddenDim = kernelCfg.hiddenDim;
  deviceCfg.scaleDim = kernelCfg.scaleDim;
  deviceCfg.scaleTypeSize = kernelCfg.scaleTypeSize;
  deviceCfg.maxTokenTypeSize = kernelCfg.maxTokenTypeSize;
  deviceCfg.maxNumInpTokenPerRank = kernelCfg.maxNumInpTokenPerRank;
  deviceCfg.numExpertPerRank = kernelCfg.numExpertPerRank;
  deviceCfg.numExpertPerToken = kernelCfg.numExpertPerToken;
  deviceCfg.maxTotalRecvTokens = kernelCfg.maxTotalRecvTokens;
  deviceCfg.gpuPerNode = kernelCfg.gpuPerNode;
  deviceCfg.numQpPerPe = kernelCfg.numQpPerPe;
  deviceCfg.quantType = kernelCfg.quantType;
  return deviceCfg;
}

template <typename Self, typename Visit>
inline void VisitFields(Self& cfg, const EpInterNodeKernelCfg& defaults, Visit&& visit) {
#define MORI_FIELD(x) visit(#x, cfg.x, defaults.x)
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
inline std::string Render(const EpInterNodeKernelCfg& cfg) {
  const EpInterNodeKernelCfg defaults{};
  mori::jit::v2::Fields fields;
  VisitFields(cfg, defaults,
              [&fields](const char* name, const auto& value, const auto& defaultValue) {
                fields.Put(name, value, defaultValue);
              });
  return mori::jit::v2::BraceInit("::mori::ops::v2::EpInterNodeKernelCfg", fields);
}

// Lets Fields::Put treat a nested EpInterNodeKernelCfg like any other value, without a
// second field list: the rendered text is what distinguishes two of them.
inline std::string RenderValue(const EpInterNodeKernelCfg& cfg) { return Render(cfg); }

// A zero in any divisor is a division by a literal zero once the cfg is an
// NTTP -- hipcc either rejects the TU or emits a poison value, and neither
// diagnoses back to the caller that built it wrong.
inline bool EpInterNodeKernelCfgIsValid(const EpInterNodeKernelCfg& cfg) {
  return cfg.worldSize > 0 && cfg.gpuPerNode > 0 && cfg.numQpPerPe > 0 &&
         cfg.numExpertPerRank > 0 && cfg.numExpertPerToken > 0 && cfg.maxNumInpTokenPerRank > 0 &&
         cfg.hiddenDim > 0 && (cfg.worldSize % cfg.gpuPerNode) == 0;
}

// The args schema, in the form plan_api publishes.
//
// Named fields, exactly like EpArgs: the field list, this string and the
// offset-order assert are all generated from MORI_EP_INTERNODE_ARGS_FIELDS in
// ep_internode_args.hpp, so the binding builds its ctypes struct from what C++
// declares rather than from a parallel copy. The one byte range is devComm,
// which is cco's struct and not EP's to name.
inline const char* EpInterNodeArgsSchema() {
  static const std::string schema = std::string(MORI_EP_INTERNODE_ARGS_SCHEMA) + "devComm:b" +
                                    std::to_string(sizeof(::mori::cco::ccoDevComm)) + ",";
  return schema.c_str();
}

}  // namespace v2
}  // namespace ops
}  // namespace mori
