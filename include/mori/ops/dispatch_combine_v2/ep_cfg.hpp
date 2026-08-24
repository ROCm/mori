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
  // uint8[maxRecv*scaleBytes]. Only read when Cfg.scaleBytes > 0; the arena does
  // not carry the region otherwise, so this stays 0 and nothing dereferences it.
  unsigned long long offOutScales = 0;

  // Which LSA rank this is. Runtime for the same reason: as a Cfg field it made
  // all eight ranks compile their own copy of an identical kernel.
  int rank = 0;

  const int* tokenIndices = nullptr;  // [numTokens * topk] expert ids, <0 drops
  const void* inpTokenBuf = nullptr;  // dispatch: source tokens; combine: post-expert tokens
  const float* weightsBuf = nullptr;  // [numTokens * topk]
  const void* scalesBuf = nullptr;    // [numTokens * scaleBytes] per-token scale rows
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
// sizeof -- which cannot see two same-type fields swapped, and 9 of the 24 are
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
  X(offOutScales, "u64")       \
  X(rank, "i32")               \
  X(tokenIndices, "p")         \
  X(inpTokenBuf, "p")          \
  X(weightsBuf, "p")           \
  X(scalesBuf, "p")            \
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

static_assert(detail::kEpArgsFieldCount == 24,
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
  // Per-token scale row carried alongside the payload, in bytes. 0 = off, and off
  // is free: Render omits default-valued fields, so the Cfg text -- which IS the
  // JIT cache key -- is byte-identical to a build without this feature.
  int scaleBytes = 0;
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
  MORI_FIELD(scaleBytes);
#undef MORI_FIELD
}

MORI_JIT_ASSERT_FIELD_COUNT(EpCfg, 12, "added an EpCfg field -- update VisitFields(EpCfg) too");

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

// The scale row's SLOT stride, which is not the row the caller handed us.
//
// A scale transfer is a run of consecutive destination slots, so the slot stride
// is what decides whether a run STARTS on a 128 B boundary -- and TdmWholeOrSplit128
// only yields a `body` for the part of a run that does; the rest degrades to the
// scalar global->global fallback. At the natural hidden/32 = 224 B only every 4th
// run starts aligned. Measured at 512 tokens/rank, hidden 7168, EP4 on gfx1250:
// dispatch 157.7us at 224 B, 94.8us at 256 B.
//
// Derived rather than a second Cfg field, so a caller states only the row it has
// and mori states the row it lays down. It is padded on every arch, not just the
// TDM one: the portable body does not need the alignment, but a destination
// layout that changed with the arch would fork every consumer of it, and the
// padding is ~14% of a side channel that is itself ~3% of an fp8 payload.
//
// ANY reader of the destination -- the receiving kernel, the arena sizing, a test
// -- must use this, not Cfg.scaleBytes.
constexpr int EpScaleAlign = 128;
constexpr int EpScaleStride(const EpCfg& c) {
  return c.scaleBytes <= 0 ? 0 : (c.scaleBytes + EpScaleAlign - 1) / EpScaleAlign * EpScaleAlign;
}

// gfx1250 launch LDS. Dispatch stages one token tile per warp through the TDM
// engine; combine reserves the whole budget and sizes its tiles at runtime.
// EpCombine1250xLdsBudget must match MORI_COMB_LDS_BUDGET in ep_intranode_1250x.hpp.
//
// Ep1250xLdsBytes is the physical ceiling; combine's budget happens to be all of
// it. Anything else that needs the ceiling should say so directly rather than
// reach for combine's number -- otherwise retuning combine silently resizes it.
constexpr int Ep1250xLdsBytes = 327680;
constexpr int EpCombine1250xLdsBudget = Ep1250xLdsBytes;
// xdbFlag slots for the per-block combine entry barrier: one uint64 per block, so a
// block is the only writer of its own epoch. 256 == the CU count, which caps the
// combine block_num; the host allocates this many and every call keeps them in step.
constexpr int EpXdbFlagSlots = 256;

// Per-warp LDS slab. The metadata tile and the payload tile share it (same
// address, different phases), so its size bounds BOTH -- and the metadata batch
// size is (slab - headroom) / bytes-per-token.
//
// Sizing it off the payload dtype alone is fine until a scale row joins the
// metadata: an fp8 payload halves the slab exactly when per-token metadata grows
// from 52 to 276 bytes, and the batch collapses ~11x.
//
// The floor is EMPIRICAL, and from one shape: hidden 7168, topk 6, EP4, 512
// tokens/rank on gfx1250, where it moved dispatch from 93.1us back to 36.7us.
// The mechanism (a narrower slab shrinks the metadata batch) is general; the
// conclusion that the bf16 width is the right target is not necessarily so for a
// very different hidden size, and this is the first thing to re-measure if one
// shows up.
//
// Capped by the physical LDS rather than by combine's budget: the two are the
// same number today, but for a shared physical reason, not because dispatch
// follows combine.
constexpr int EpDispatch1250xSlabBytes(const EpCfg& c) {
  const int payload = c.hiddenDim * EpElemSize(c.dtype);
  if (c.scaleBytes <= 0) return payload;
  const int wide = c.hiddenDim * EpElemSize(EpDType::Bf16);
  const long long total = (long long)wide * c.warpPerBlock;
  return (wide > payload && total <= Ep1250xLdsBytes) ? wide : payload;
}
constexpr int EpDispatch1250xLdsBytes(const EpCfg& c) {
  return c.warpPerBlock * EpDispatch1250xSlabBytes(c);
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
         (EpTokenBytes(c) % 16) == 0 &&
         // Scale rows are copied as dwords, so the row must be dword-sized.
         c.scaleBytes >= 0 && (c.scaleBytes % 4) == 0;
}

}  // namespace v2
}  // namespace ops
}  // namespace mori
