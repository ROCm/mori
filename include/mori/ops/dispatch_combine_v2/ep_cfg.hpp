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
// EP intranode dispatch/combine: the specialisation identity, the runtime
// arguments, and the arithmetic both sides share.
//
// Ported from origin/main src/ops/dispatch_combine/intranode.hpp. That kernel's
// whole symmetric-memory surface is `SymmMemObjPtr::GetAs<T*>(pe)`, a two-load
// table lookup into a per-region peer-pointer array. Here it is one arena, one
// cco window, and a compile-time offset per region:
//
//     memObj->GetAs<T*>(pe)  ->  ccoGetLsaPeerPtr(win, pe, args.offRegion)
//
// so the 13 SymmMemObjPtr fields of EpDispatchCombineArgs collapse to a single
// window handle plus eight offsets, all runtime arguments.
//
// HIP-free and attribute-free: host compiles this with a plain C++ compiler,
// the generated device TU with hipcc. See MORI_JIT_V2_DESIGN.md §3.5.

#pragma once

#include <string>

#include "mori/jit/v2/render.hpp"

namespace mori {
namespace ops {
namespace v2 {

// ---------------------------------------------------------------------------
// dtype tag. The generated source expands it to a real type name; nothing here
// includes a HIP header.
//
// The VALUES are load-bearing: an `e`-tagged field crosses the boundary as a
// bare integer, and the binding has exactly ONE dtype name->int table
// (mori.jit.v2.plan_api DTYPES) that it applies to every enum field of every kernel.
// So this enumeration must agree with mori::ops::v2::DType numerically, or a
// caller asking for fp32 gets whatever this enum happens to call 1. Renumbering
// either enum independently is a silent-wrong-answer change, not a refactor.
// ---------------------------------------------------------------------------
enum class EpDType : int { Bf16 = 0, Fp32 = 1 };

inline const char* EpDTypeName(EpDType d) {
  return d == EpDType::Fp32 ? "float" : "hip_bfloat16";
}

constexpr int EpElemSize(EpDType d) { return d == EpDType::Fp32 ? 4 : 2; }

inline std::string RenderValue(EpDType d) {
  return d == EpDType::Fp32 ? "::mori::ops::v2::EpDType::Fp32"
                            : "::mori::ops::v2::EpDType::Bf16";
}

// ---------------------------------------------------------------------------
// Runtime arguments. One struct for both kernels: the union is small and a
// single published schema keeps the binding generic. Pointers that are NOT in
// the arena stay here (they are plain local buffers no peer reads).
// ---------------------------------------------------------------------------
struct EpArgs {
  // The cco window handle (ccoWindow_t). Runtime, not Cfg: the base address is
  // only known once the arena exists, and it differs per rank.
  unsigned long long window = 0;

  // Arena region byte offsets, matching the region list the Python op builds
  // (see SymmArena). RUNTIME, not Cfg: baking them in made every arena layout a
  // separate binary and every rank compile its own, and the gfx942 micro-benchmark
  // measured the constant form as a REGRESSION (VGPR 9 -> 22, because the compiler
  // loses "the base is uniform" and rematerialises the address per lane).
  unsigned long long offTokOff = 0;     // index_t[1]            slot allocator
  unsigned long long offRecvNum = 0;    // index_t[worldSize]    recv-count signal
  unsigned long long offRecvToSrc = 0;  // index_t[maxRecv]      slot -> src token
  unsigned long long offOutIdx = 0;     // index_t[maxRecv*topk] forwarded expert ids
  unsigned long long offOutWts = 0;     // float[maxRecv*topk]   forwarded weights
  unsigned long long offDispOut = 0;    // T[maxRecv*hidden]     dispatch landing zone
  unsigned long long offOutTok = 0;     // T[maxRecv*hidden]     combine staging
  unsigned long long offXdb = 0;        // uint64[worldSize]     barrier slots

  // Which LSA rank this is. Runtime for the same reason: as a Cfg field it made
  // all eight ranks compile their own copy of an identical kernel.
  int rank = 0;

  const int* tokenIndices = nullptr;  // [numTokens * topk] expert ids, <0 drops
  const void* inpTokenBuf = nullptr;  // dispatch: source tokens; combine: post-expert tokens
  const float* weightsBuf = nullptr;  // [numTokens * topk]
  void* outTokenBuf = nullptr;        // combine output, local
  float* outWeightsBuf = nullptr;     // combine weight output, local

  int* dispDestTokIdMap = nullptr;    // [numTokens * topk] flat dest index per (token, k)
  int* destPeTokenCounter = nullptr;  // [worldSize] per-dest send count
  int* totalRecvTokenNum = nullptr;   // [1]
  unsigned int* gridBarrier = nullptr;  // [1] intra-kernel grid rendezvous
  unsigned long long* xdbFlag = nullptr;  // [1] monotone cross-device barrier epoch

  int numTokens = 0;  // tokens this rank contributes this call
};

// ---------------------------------------------------------------------------
// Cfg. Shared by dispatch and combine: they run over the same arena and the
// same shape, and only the launch geometry differs. Two Specs, one Cfg.
// ---------------------------------------------------------------------------
struct EpCfg {
  // ---- topology / shape ----
  int worldSize = 8;
  int hiddenDim = 7168;
  int maxTokPerRank = 128;      // per-rank input token capacity
  int numExpertPerRank = 8;
  int numExpertPerToken = 8;    // topk
  int maxRecv = 0;              // 0 = worldSize * maxTokPerRank
  EpDType dtype = EpDType::Bf16;

  // ---- launch geometry (host-derived; see MakeEpCfg) ----
  int blockNum = 64;
  int warpPerBlock = 16;
  int waveSize = 64;

  // ---- algorithm ----
  bool useWeights = true;
};

template <typename Self, typename Visit>
inline void VisitFields(Self& c, const EpCfg& d, Visit&& v) {
#define MORI_FIELD(x) v(#x, c.x, d.x)
  MORI_FIELD(worldSize);
  MORI_FIELD(hiddenDim);
  MORI_FIELD(maxTokPerRank);
  MORI_FIELD(numExpertPerRank);
  MORI_FIELD(numExpertPerToken);
  MORI_FIELD(maxRecv);
  MORI_FIELD(dtype);
  MORI_FIELD(blockNum);
  MORI_FIELD(warpPerBlock);
  MORI_FIELD(waveSize);
  MORI_FIELD(useWeights);
#undef MORI_FIELD
}

MORI_JIT_ASSERT_FIELD_COUNT(EpCfg, 11, "added an EpCfg field -- update VisitFields(EpCfg) too");

inline std::string Render(const EpCfg& c) {
  const EpCfg d{};
  mori::jit::v2::Fields f;
  VisitFields(c, d, [&f](const char* name, const auto& value, const auto& dflt) {
    f.Put(name, value, dflt);
  });
  return mori::jit::v2::BraceInit("EpCfg", f);
}

// Every field, default or not -- what the plan reports back through `info`.
inline std::string Describe(const EpCfg& c) {
  std::string out;
  VisitFields(c, c, [&out](const char* name, const auto& value, const auto&) {
    using mori::jit::v2::RenderValue;  // ADL for the Ep types, jit's for scalars
    out += name;
    out += "=";
    out += RenderValue(value);
    out += "\n";
  });
  return out;
}

// ---------------------------------------------------------------------------
// Wire schema + generic apply, driven by the same VisitFields walk Render uses.
//
// Deliberately Ep-prefixed rather than overloading combine's EmitSchema /
// ApplyFields: those live in this same namespace, and an unqualified call with
// an Ep type would pick them up by ADL. Two kernels sharing a namespace should
// not have to reason about which overload set wins.
// ---------------------------------------------------------------------------
template <typename T>
inline void EpEmitSchema(mori::jit::v2::SchemaBuilder& sb, const std::string& name, const T& v) {
  using mori::jit::v2::WireTag;
  using mori::jit::v2::WireValue;
  sb.Add(name, WireTag(v), WireValue(v));
}
// Apply named request values onto a struct. EpRequest is all scalars, so this is
// one flat walk -- no nested-aggregate recursion to arrange.
template <typename T, typename Has, typename Get>
inline void EpApplyFields(T& dst, const std::string& prefix, const Has& has, const Get& get) {
  using mori::jit::v2::WireAssign;
  const T defaults{};
  VisitFields(dst, defaults, [&](const char* n, auto& slot, const auto&) {
    const std::string key = prefix.empty() ? std::string(n) : prefix + "." + n;
    if (has(key)) WireAssign(slot, get(key));
  });
}

// ---------------------------------------------------------------------------
// Shared arithmetic. One definition, used by the host to size the launch and by
// the device as a compile-time constant. Attribute-free constexpr on purpose:
// __host__ __device__ here would drag every host TU through hipcc.
// ---------------------------------------------------------------------------
constexpr int EpBlockThreads(const EpCfg& c) { return c.warpPerBlock * c.waveSize; }

// Recv-slot capacity. The flat token index encodes (pe, localTokId) with this
// stride, so host and device must agree exactly.
constexpr int EpMaxRecv(const EpCfg& c) {
  return c.maxRecv > 0 ? c.maxRecv : c.worldSize * c.maxTokPerRank;
}

// Combine's shared memory: one pointer array per warp for the topk sources,
// plus a second one for the weight pointers when weights are enabled.
constexpr int EpCombineSharedBytes(const EpCfg& c) {
  return static_cast<int>(sizeof(void*)) * c.warpPerBlock * c.numExpertPerToken *
         (c.useWeights ? 2 : 1);
}

constexpr int EpTokenBytes(const EpCfg& c) { return c.hiddenDim * EpElemSize(c.dtype); }

// A Cfg that cannot launch is a host-side error, not a kernel that misbehaves.
constexpr bool EpCfgIsValid(const EpCfg& c) {
  // rank is not checked here any more -- it is a launch argument, so the op
  // layer owns that bound (it is the rank it was constructed for).
  return c.worldSize > 0 && c.hiddenDim > 0 &&
         c.maxTokPerRank > 0 && c.numExpertPerToken > 0 && c.numExpertPerRank > 0 &&
         c.blockNum > 0 && (c.waveSize == 32 || c.waveSize == 64) && c.warpPerBlock > 0 &&
         EpBlockThreads(c) <= 1024 &&
         // The recv capacity must cover the worst case. The dispatch slot
         // counter is unbounded on the device (v1 asserted, which NDEBUG strips
         // anyway), and because EpMaxRecv is ALSO the flat-index stride, an
         // overflow does not just overrun the region -- it re-encodes to the
         // next peer and combine folds in a stranger's token. Reject the
         // under-sized cap at construction instead: dropping tokens is a
         // feature this port does not implement.
         EpMaxRecv(c) >= c.worldSize * c.maxTokPerRank &&
         // The dedup ballot and the grid-barrier peer loop both assume one lane
         // per peer / per top-k slot within a single wavefront.
         c.numExpertPerToken < c.waveSize && c.worldSize <= c.waveSize &&
         // WarpCopy moves whole 16 B chunks.
         (EpTokenBytes(c) % 16) == 0;
}

}  // namespace v2
}  // namespace ops
}  // namespace mori
