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
// Ported from src/ops/dispatch_combine/intranode.hpp, whose symmetric-memory
// surface is `memObj->GetAs<T*>(pe)`. Here that is
// `ccoGetLsaPeerPtr(win, pe, args.offRegion)`: one arena, one window, one offset
// per region, so 13 SymmMemObjPtr fields become a handle plus eight offsets.
//
// HIP-free and attribute-free: host compiles this with a plain C++ compiler,
// the generated device TU with hipcc. See MORI_JIT_V2_DESIGN.md §3.5.

#pragma once

#include <cstddef>
#include <string>

#include "mori/jit/v2/render.hpp"

namespace mori {
namespace ops {
namespace v2 {

// ---------------------------------------------------------------------------
// dtype tag. The generated source expands it to a real type name; nothing here
// includes a HIP header.
//
// The VALUES are load-bearing: an `e`-tagged field crosses as a bare integer and
// the binding has one dtype name->int table (plan_api.DTYPES) for every kernel,
// so this must agree numerically with mori::ops::v2::DType. Renumbering either
// alone is a silent wrong answer, not a refactor.
// ---------------------------------------------------------------------------
// Byte8 is a TRANSPORT type: dispatch only copies its payload, so fp8 and fp4 (2
// e2m1 per byte, caller halves hiddenDim) both move as bytes. Combine reduces and
// cannot use it -- MakeEpCfg rejects it there.
enum class EpDType : int { Bf16 = 0, Fp32 = 1, Byte8 = 2 };

inline const char* EpDTypeName(EpDType d) {
  switch (d) {
    case EpDType::Fp32:
      return "float";
    case EpDType::Byte8:
      return "unsigned char";
    default:
      return "hip_bfloat16";
  }
}

// Identifier-safe short tag, for the kernel symbol name. Separate from
// EpDTypeName, which yields a C++ type -- "unsigned char" has a space in it and
// would not be a legal symbol.
inline const char* EpDTypeTag(EpDType d) {
  switch (d) {
    case EpDType::Fp32:
      return "fp32";
    case EpDType::Byte8:
      return "byte8";
    default:
      return "bf16";
  }
}

constexpr int EpElemSize(EpDType d) {
  return d == EpDType::Fp32 ? 4 : (d == EpDType::Byte8 ? 1 : 2);
}

inline std::string RenderValue(EpDType d) {
  switch (d) {
    case EpDType::Fp32:
      return "::mori::ops::v2::EpDType::Fp32";
    case EpDType::Byte8:
      return "::mori::ops::v2::EpDType::Byte8";
    default:
      return "::mori::ops::v2::EpDType::Bf16";
  }
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

  // Arena region byte offsets, matching the region list SymmArena builds. Runtime,
  // not Cfg: as constants they made every arena layout a separate binary AND
  // measured slower on gfx942 (VGPR 9 -> 22 -- the compiler loses "the base is
  // uniform" and rematerialises the address per lane).
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

  int* dispDestTokIdMap = nullptr;        // [numTokens * topk] flat dest index per (token, k)
  int* destPeTokenCounter = nullptr;      // [worldSize] per-dest send count
  int* totalRecvTokenNum = nullptr;       // [1]
  unsigned int* gridBarrier = nullptr;    // [1] intra-kernel grid rendezvous
  unsigned long long* xdbFlag = nullptr;  // [1] monotone cross-device barrier epoch
  int* combineBarrierFan =
      nullptr;  // [blockNum*16] gfx1250 combine intra-grid fan-out (local scratch)

  int numTokens = 0;  // tokens this rank contributes this call
};

// The wire schema, generated from the field list rather than kept parallel to it.
// The binding builds its ctypes struct from `name:tag` in this order and checks
// sizeof -- which cannot see two same-type fields swapped, and 8 of the 22 are
// bare pointers. So the static_asserts below take the offsets in SCHEMA order:
// any disagreement with the declaration order stops the sequence increasing.
#define MORI_EP_ARGS_FIELDS(X) \
  X(window, "u64")             \
  X(offTokOff, "u64")          \
  X(offRecvNum, "u64")         \
  X(offRecvToSrc, "u64")       \
  X(offOutIdx, "u64")          \
  X(offOutWts, "u64")          \
  X(offDispOut, "u64")         \
  X(offOutTok, "u64")          \
  X(offXdb, "u64")             \
  X(rank, "i32")               \
  X(tokenIndices, "p")         \
  X(inpTokenBuf, "p")          \
  X(weightsBuf, "p")           \
  X(outTokenBuf, "p")          \
  X(outWeightsBuf, "p")        \
  X(dispDestTokIdMap, "p")     \
  X(destPeTokenCounter, "p")   \
  X(totalRecvTokenNum, "p")    \
  X(gridBarrier, "p")          \
  X(xdbFlag, "p")              \
  X(combineBarrierFan, "p")    \
  X(numTokens, "i32")

#define MORI_EP_ARGS_SCHEMA_ENTRY(name, tag) #name ":" tag ","
// Trailing comma: the binding skips empty items, and a separator rule that does
// not special-case the last element is one less thing to get wrong.
#define MORI_EP_ARGS_SCHEMA MORI_EP_ARGS_FIELDS(MORI_EP_ARGS_SCHEMA_ENTRY)

namespace detail {

#define MORI_EP_ARGS_OFFSET(name, tag) offsetof(::mori::ops::v2::EpArgs, name),
inline constexpr size_t kEpArgsOffsets[] = {MORI_EP_ARGS_FIELDS(MORI_EP_ARGS_OFFSET)};
#undef MORI_EP_ARGS_OFFSET

constexpr size_t kEpArgsFieldCount = sizeof(kEpArgsOffsets) / sizeof(kEpArgsOffsets[0]);

constexpr bool EpArgsOffsetsAscend() {
  for (size_t i = 1; i < kEpArgsFieldCount; ++i)
    if (kEpArgsOffsets[i] <= kEpArgsOffsets[i - 1]) return false;
  return true;
}

}  // namespace detail

static_assert(detail::kEpArgsFieldCount == 22,
              "added an EpArgs field -- add it to MORI_EP_ARGS_FIELDS in the same position "
              "and bump this count");
static_assert(detail::EpArgsOffsetsAscend(),
              "MORI_EP_ARGS_FIELDS is not in EpArgs declaration order -- the binding would "
              "write each argument into the wrong slot");

// ---------------------------------------------------------------------------
// Cfg. Shared by dispatch and combine: they run over the same arena and the
// same shape, and only the launch geometry differs. Two Specs, one Cfg.
// ---------------------------------------------------------------------------
struct EpCfg {
  // ---- topology / shape ----
  int worldSize = 8;
  int hiddenDim = 7168;
  int maxTokPerRank = 128;  // per-rank input token capacity
  int numExpertPerRank = 8;
  int numExpertPerToken = 8;  // topk
  int maxRecv = 0;            // 0 = worldSize * maxTokPerRank
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
// Ep-prefixed rather than overloading combine's EmitSchema / ApplyFields: those
// share this namespace, so an unqualified call would find them by ADL.
// ---------------------------------------------------------------------------
template <typename T>
inline void EpEmitSchema(mori::jit::v2::SchemaBuilder& sb, const std::string& name, const T& v) {
  using mori::jit::v2::WireTag;
  using mori::jit::v2::WireValue;
  sb.Add(name, WireTag(v), WireValue(v));
}
// Apply named request values onto a struct. EpRequest is all scalars: one flat walk.
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

// Tokens a warp takes per iteration on the gfx125x dispatch path. `tpi` is the
// natural quota (waveSize/topk, which lets COUNT read tokenIndices with every
// lane); it only fills the grid once there are aWarps*tpi tokens to go round,
// so below that the work piles onto the low warps and most blocks send nothing.
// Capping at ceil(numTokens/aWarps) spreads a scarce batch over every warp.
//
// Here rather than inline in the kernel so the two properties that matter can be
// tested on a host with no gfx125x: the result is never < 1 (a step of
// `aWarps * 0` never advances -- an unkillable hang, and no correctness check
// can report it because the check hangs too), and above the threshold it is
// exactly `tpi`, so the large-token path is identical and not merely close.
constexpr int EpWarpTokenQuota(int numTokens, int aWarps, int tpi) {
  if (tpi <= 1 || aWarps <= 0) return tpi;
  const int q = static_cast<int>((static_cast<long long>(numTokens) + aWarps - 1) / aWarps);
  return (q >= 1 && q < tpi) ? q : tpi;
}

// gfx1250 launch LDS. Dispatch stages one token tile per warp through the TDM
// engine; combine reserves the whole budget and sizes its tiles at runtime.
// EpCombine1250xLdsBudget must match MORI_COMB_LDS_BUDGET in ep_intranode_1250x.hpp.
constexpr int EpCombine1250xLdsBudget = 327680;
constexpr int EpDispatch1250xLdsBytes(const EpCfg& c) {
  return c.warpPerBlock * c.hiddenDim * EpElemSize(c.dtype);
}

// A Cfg that cannot launch is a host-side error, not a kernel that misbehaves.
// rank is not checked: it is a launch argument, so the op layer owns that bound.
constexpr bool EpCfgIsValid(const EpCfg& c) {
  return c.worldSize > 0 && c.hiddenDim > 0 && c.maxTokPerRank > 0 && c.numExpertPerToken > 0 &&
         c.numExpertPerRank > 0 && c.blockNum > 0 && (c.waveSize == 32 || c.waveSize == 64) &&
         c.warpPerBlock > 0 && EpBlockThreads(c) <= 1024 &&
         // The recv capacity must cover the worst case: the device slot counter is
         // unbounded, and since EpMaxRecv is also the flat-index stride an overflow
         // re-encodes to the next peer and combine folds in a stranger's token.
         // Token dropping is not implemented, so reject the cap at construction.
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
